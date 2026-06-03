#!/usr/bin/env python3
"""Generate BEV soccer minimaps from SoccerNet GameState-style annotations.

The renderer never uses bbox_pitch as object positions.  bbox_pitch can be used
only to estimate an image-to-pitch homography.  Player, referee, and optional
ball dots are rendered from bbox_image points projected through that homography.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
HALF_LENGTH_M = PITCH_LENGTH_M / 2.0
HALF_WIDTH_M = PITCH_WIDTH_M / 2.0

PERSON_CATEGORY_IDS = {1, 2}
REFEREE_CATEGORY_ID = 3
BALL_CATEGORY_ID = 4


@dataclass(frozen=True)
class FrameInfo:
    image_id: str
    frame_number: int
    file_name: str
    width: Optional[int] = None
    height: Optional[int] = None


@dataclass(frozen=True)
class Detection:
    image_id: str
    frame_number: int
    category_id: int
    role: str
    team: Optional[str]
    track_id: Optional[int]
    bbox_image: Dict[str, float]
    bbox_pitch: Optional[Dict[str, float]]


@dataclass
class HomographyInfo:
    matrix: Optional[np.ndarray]
    inliers: int = 0
    points: int = 0
    source: str = "missing"
    jump_m: Optional[float] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render original, BEV/homography, and side-by-side videos from SoccerNetGS annotations."
    )
    parser.add_argument("--labels", required=True, type=Path, help="Path to Labels-GameState.json.")
    parser.add_argument("--video", type=Path, default=None, help="Optional source video; frames are used as fallback/input.")
    parser.add_argument("--output", required=True, type=Path, help="Output BEV/homography minimap video.")
    parser.add_argument(
        "--original-output",
        type=Path,
        default=None,
        help="Original-frame output path. Defaults to '<output_stem>_original<suffix>'.",
    )
    parser.add_argument(
        "--side-by-side-output",
        type=Path,
        default=None,
        help="Side-by-side output path. Defaults to '<output_stem>_side_by_side<suffix>'.",
    )
    parser.add_argument("--homography-csv", type=Path, default=None, help="Load homographies from CSV instead of estimating.")
    parser.add_argument("--save-homography-csv", type=Path, default=None, help="Optional homography CSV output.")
    parser.add_argument("--debug-csv", type=Path, default=None, help="Optional per-frame debug CSV output.")
    parser.add_argument("--debug", action="store_true", help="Write derived debug and homography CSV files.")

    parser.add_argument("--start-frame", type=int, default=None, help="First 1-based frame number to render.")
    parser.add_argument("--end-frame", type=int, default=None, help="Last 1-based frame number to render.")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of selected frames to render.")
    parser.add_argument("--fps", type=float, default=None, help="Override output FPS.")
    parser.add_argument("--map-width", type=int, default=960, help="BEV output width.")
    parser.add_argument("--map-height", type=int, default=620, help="BEV output height.")

    parser.add_argument("--min-homography-points", type=int, default=4, help="Minimum correspondences for homography.")
    parser.add_argument("--ransac-threshold", type=float, default=3.0, help="RANSAC threshold in pitch units.")
    parser.add_argument("--include-referees-in-homography", action="store_true", help="Use referees for calibration too.")
    parser.add_argument(
        "--homography-footpoints",
        choices=("bottom-line", "middle"),
        default="bottom-line",
        help="Use bbox bottom-left/middle/right points, or only bottom-middle, for calibration.",
    )
    parser.add_argument("--no-reuse-last-homography", action="store_true", help="Do not fill missing H with last valid H.")
    parser.add_argument("--homography-smoothing", type=float, default=0.18, help="EMA alpha for homography smoothing.")
    parser.add_argument(
        "--max-homography-jump-m",
        type=float,
        default=10.0,
        help="Large H jumps reduce smoothing alpha.",
    )

    parser.add_argument("--position-smoothing", type=float, default=0.55, help="EMA alpha for per-track dot smoothing.")
    parser.add_argument("--max-position-step-m", type=float, default=8.0, help="Clamp per-frame displayed dot step.")
    parser.add_argument("--velocity", action="store_true", help="Draw homography-projected speed labels for tracked people.")
    parser.add_argument("--velocity-window", type=int, default=5, help="Number of previous frames to average for velocity.")
    parser.add_argument("--velocity-csv", type=Path, default=None, help="Optional per-track velocity CSV output.")
    parser.add_argument("--pitch-margin-m", type=float, default=8.0, help="Hide dots outside pitch plus this margin.")
    parser.add_argument("--trail-frames", type=int, default=0, help="Track trail length. Defaults to points only.")
    parser.add_argument("--show-track-ids", action="store_true", help="Draw track IDs next to BEV/source dots.")
    parser.add_argument("--no-source-boxes", action="store_true", help="Do not draw source boxes on original/side-by-side.")

    parser.add_argument("--ball", action="store_true", help="Include ball dots/boxes. Otherwise render players/referees only.")
    parser.add_argument("--ball-point", choices=("bottom", "center"), default="bottom", help="Image point for ball projection.")
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def frame_number_from_file_name(file_name: str, fallback: int) -> int:
    try:
        return int(Path(file_name).stem)
    except ValueError:
        return fallback


def finite_pair(x: Any, y: Any) -> bool:
    try:
        return math.isfinite(float(x)) and math.isfinite(float(y))
    except (TypeError, ValueError):
        return False


def load_annotations(
    labels_path: Path,
    keep_bbox_pitch: bool = True,
) -> Tuple[List[FrameInfo], DefaultDict[str, List[Detection]], Dict[str, Any]]:
    labels = read_json(labels_path)
    info = labels.get("info", {})
    frames: List[FrameInfo] = []
    image_id_to_frame: Dict[str, FrameInfo] = {}

    for index, image in enumerate(labels.get("images", []), start=1):
        image_id = str(image.get("image_id", index))
        file_name = str(image.get("file_name", f"{index:06d}.jpg"))
        frame = FrameInfo(
            image_id=image_id,
            frame_number=frame_number_from_file_name(file_name, index),
            file_name=file_name,
            width=image.get("width"),
            height=image.get("height"),
        )
        frames.append(frame)
        image_id_to_frame[image_id] = frame

    annotations_by_image: DefaultDict[str, List[Detection]] = defaultdict(list)
    synthetic_frame = 1
    for ann in labels.get("annotations", []):
        if "bbox_image" not in ann:
            continue
        image_id = str(ann.get("image_id", ""))
        frame = image_id_to_frame.get(image_id)
        if frame is None:
            frame = FrameInfo(image_id=image_id, frame_number=synthetic_frame, file_name=f"{synthetic_frame:06d}.jpg")
            image_id_to_frame[image_id] = frame
            frames.append(frame)
            synthetic_frame += 1

        attributes = ann.get("attributes") or {}
        annotations_by_image[image_id].append(
            Detection(
                image_id=image_id,
                frame_number=frame.frame_number,
                category_id=int(ann.get("category_id", -1)),
                role=str(attributes.get("role") or ""),
                team=attributes.get("team"),
                track_id=ann.get("track_id"),
                bbox_image=ann.get("bbox_image") or {},
                bbox_pitch=ann.get("bbox_pitch") if keep_bbox_pitch else None,
            )
        )

    frames.sort(key=lambda item: (item.frame_number, item.image_id))
    return frames, annotations_by_image, info


def bbox_bottom_middle(bbox: Dict[str, float]) -> Optional[Tuple[float, float]]:
    try:
        x = float(bbox["x"])
        y = float(bbox["y"])
        w = float(bbox["w"])
        h = float(bbox["h"])
    except (KeyError, TypeError, ValueError):
        return None
    return x + w * 0.5, y + h


def bbox_center(bbox: Dict[str, float]) -> Optional[Tuple[float, float]]:
    if "x_center" in bbox and "y_center" in bbox:
        try:
            return float(bbox["x_center"]), float(bbox["y_center"])
        except (TypeError, ValueError):
            return None
    try:
        x = float(bbox["x"])
        y = float(bbox["y"])
        w = float(bbox["w"])
        h = float(bbox["h"])
    except (KeyError, TypeError, ValueError):
        return None
    return x + w * 0.5, y + h * 0.5


def bbox_bottom_line_points(bbox: Dict[str, float], mode: str) -> List[Tuple[str, Tuple[float, float]]]:
    try:
        x = float(bbox["x"])
        y = float(bbox["y"])
        w = float(bbox["w"])
        h = float(bbox["h"])
    except (KeyError, TypeError, ValueError):
        return []
    bottom_y = y + h
    if mode == "middle":
        return [("middle", (x + w * 0.5, bottom_y))]
    return [("left", (x, bottom_y)), ("middle", (x + w * 0.5, bottom_y)), ("right", (x + w, bottom_y))]


def pitch_bottom_line_points(bbox_pitch: Optional[Dict[str, float]], mode: str) -> Dict[str, Tuple[float, float]]:
    if not bbox_pitch:
        return {}
    keys = {
        "left": ("x_bottom_left", "y_bottom_left"),
        "middle": ("x_bottom_middle", "y_bottom_middle"),
        "right": ("x_bottom_right", "y_bottom_right"),
    }
    wanted = ("middle",) if mode == "middle" else ("left", "middle", "right")
    points: Dict[str, Tuple[float, float]] = {}
    for name in wanted:
        x_key, y_key = keys[name]
        x = bbox_pitch.get(x_key)
        y = bbox_pitch.get(y_key)
        if finite_pair(x, y):
            points[name] = (float(x), float(y))
    return points


def is_person_for_homography(det: Detection, include_referees: bool) -> bool:
    return det.category_id in PERSON_CATEGORY_IDS or (include_referees and det.category_id == REFEREE_CATEGORY_ID)


def estimate_homography(
    detections: Sequence[Detection],
    min_points: int,
    ransac_threshold: float,
    include_referees: bool,
    footpoint_mode: str,
) -> HomographyInfo:
    src_points: List[Tuple[float, float]] = []
    dst_points: List[Tuple[float, float]] = []
    seen_tracks: set[int] = set()

    for det in detections:
        if not is_person_for_homography(det, include_referees):
            continue
        image_points = dict(bbox_bottom_line_points(det.bbox_image, footpoint_mode))
        pitch_points = pitch_bottom_line_points(det.bbox_pitch, footpoint_mode)
        common = [name for name in ("left", "middle", "right") if name in image_points and name in pitch_points]
        if not common:
            continue
        if det.track_id is not None:
            track_id = int(det.track_id)
            if track_id in seen_tracks:
                continue
            seen_tracks.add(track_id)
        for name in common:
            src_points.append(image_points[name])
            dst_points.append(pitch_points[name])

    if len(src_points) < min_points:
        return HomographyInfo(None, points=len(src_points), source="insufficient")

    src = np.asarray(src_points, dtype=np.float32).reshape(-1, 1, 2)
    dst = np.asarray(dst_points, dtype=np.float32).reshape(-1, 1, 2)
    matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_threshold)
    if matrix is None or not np.isfinite(matrix).all():
        return HomographyInfo(None, points=len(src_points), source="failed")
    inliers = int(mask.sum()) if mask is not None else len(src_points)
    if inliers < min_points:
        return HomographyInfo(None, inliers=inliers, points=len(src_points), source="few-inliers")
    return HomographyInfo(normalize_homography(matrix), inliers=inliers, points=len(src_points), source="estimated")


def normalize_homography(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(np.float64)
    if abs(matrix[2, 2]) > 1e-12:
        matrix = matrix / matrix[2, 2]
    return matrix


def load_homographies(path: Path) -> Dict[int, HomographyInfo]:
    homographies: Dict[int, HomographyInfo] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame_number = int(row["frame_number"])
                values = [float(row[f"h{r}{c}"]) for r in range(3) for c in range(3)]
            except (KeyError, TypeError, ValueError):
                continue
            matrix = normalize_homography(np.asarray(values, dtype=np.float64).reshape(3, 3))
            homographies[frame_number] = HomographyInfo(
                matrix=matrix,
                inliers=int(float(row.get("inliers", 0) or 0)),
                points=int(float(row.get("points", 0) or 0)),
                source=row.get("source") or "loaded",
                jump_m=float(row["jump_m"]) if row.get("jump_m") else None,
            )
    return homographies


def write_homographies(path: Path, rows: Sequence[Tuple[FrameInfo, HomographyInfo]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame_number",
        "image_id",
        "source",
        "inliers",
        "points",
        "jump_m",
        "h00",
        "h01",
        "h02",
        "h10",
        "h11",
        "h12",
        "h20",
        "h21",
        "h22",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for frame, info in rows:
            row: Dict[str, Any] = {
                "frame_number": frame.frame_number,
                "image_id": frame.image_id,
                "source": info.source,
                "inliers": info.inliers,
                "points": info.points,
                "jump_m": "" if info.jump_m is None else f"{info.jump_m:.6f}",
            }
            if info.matrix is None:
                row.update({f"h{r}{c}": "" for r in range(3) for c in range(3)})
            else:
                row.update({f"h{r}{c}": f"{info.matrix[r, c]:.12g}" for r in range(3) for c in range(3)})
            writer.writerow(row)


def project_points(matrix: np.ndarray, points: Sequence[Tuple[float, float]]) -> List[Optional[Tuple[float, float]]]:
    if not points:
        return []
    src = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(src, matrix)
    out: List[Optional[Tuple[float, float]]] = []
    for x, y in dst.reshape(-1, 2):
        if math.isfinite(float(x)) and math.isfinite(float(y)):
            out.append((float(x), float(y)))
        else:
            out.append(None)
    return out


def homography_anchor_points(frame: FrameInfo) -> List[Tuple[float, float]]:
    width = float(frame.width or 1920)
    height = float(frame.height or 1080)
    return [
        (width * 0.25, height * 0.70),
        (width * 0.50, height * 0.70),
        (width * 0.75, height * 0.70),
        (width * 0.25, height * 0.88),
        (width * 0.50, height * 0.88),
        (width * 0.75, height * 0.88),
        (width * 0.50, height * 0.98),
    ]


def homography_jump_m(previous: np.ndarray, current: np.ndarray, frame: FrameInfo) -> Optional[float]:
    prev_points = project_points(previous, homography_anchor_points(frame))
    curr_points = project_points(current, homography_anchor_points(frame))
    distances: List[float] = []
    for prev_point, curr_point in zip(prev_points, curr_points):
        if prev_point is None or curr_point is None:
            continue
        distances.append(float(np.linalg.norm(np.asarray(curr_point) - np.asarray(prev_point))))
    return float(np.median(distances)) if distances else None


def smooth_homography(
    previous: Optional[np.ndarray],
    current: np.ndarray,
    frame: FrameInfo,
    alpha: float,
    max_jump_m: float,
) -> Tuple[np.ndarray, Optional[float]]:
    if previous is None or alpha >= 1.0:
        return normalize_homography(current), None
    alpha = min(max(alpha, 0.0), 1.0)
    jump = homography_jump_m(previous, current, frame)
    effective_alpha = alpha
    if jump is not None and max_jump_m > 0 and jump > max_jump_m:
        effective_alpha = max(0.02, alpha * max_jump_m / jump)
    return normalize_homography(previous * (1.0 - effective_alpha) + current * effective_alpha), jump


def detection_image_point(det: Detection, ball_point: str) -> Optional[Tuple[float, float]]:
    if det.category_id == BALL_CATEGORY_ID and ball_point == "center":
        return bbox_center(det.bbox_image)
    return bbox_bottom_middle(det.bbox_image)


def filter_frames(
    frames: Sequence[FrameInfo],
    start_frame: Optional[int],
    end_frame: Optional[int],
    max_frames: Optional[int],
) -> List[FrameInfo]:
    selected = [
        frame
        for frame in frames
        if (start_frame is None or frame.frame_number >= start_frame)
        and (end_frame is None or frame.frame_number <= end_frame)
    ]
    return selected[:max_frames] if max_frames is not None else selected


def video_fps(video_path: Optional[Path]) -> Optional[float]:
    if video_path is None:
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps) if fps and math.isfinite(fps) and fps > 0 else None


def output_fps(args: argparse.Namespace, info: Dict[str, Any]) -> float:
    if args.fps is not None and args.fps > 0:
        return float(args.fps)
    try:
        if info.get("frame_rate") and float(info["frame_rate"]) > 0:
            return float(info["frame_rate"])
    except (TypeError, ValueError):
        pass
    return video_fps(args.video) or 25.0


def field_rect(width: int, height: int) -> Tuple[int, int, int, int]:
    margin = max(24, min(width, height) // 18)
    usable_w = width - 2 * margin
    usable_h = height - 2 * margin
    target_ratio = PITCH_LENGTH_M / PITCH_WIDTH_M
    if usable_w / usable_h > target_ratio:
        field_h = usable_h
        field_w = int(round(field_h * target_ratio))
    else:
        field_w = usable_w
        field_h = int(round(field_w / target_ratio))
    return (width - field_w) // 2, (height - field_h) // 2, field_w, field_h


def pitch_to_canvas(x_m: float, y_m: float, rect: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x0, y0, field_w, field_h = rect
    px = x0 + (x_m + HALF_LENGTH_M) / PITCH_LENGTH_M * field_w
    # Vertically mirrored compared with the raw pitch coordinates, per current UI convention.
    py = y0 + (y_m + HALF_WIDTH_M) / PITCH_WIDTH_M * field_h
    return int(round(px)), int(round(py))


def draw_line_m(
    canvas: np.ndarray,
    rect: Tuple[int, int, int, int],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    cv2.line(canvas, pitch_to_canvas(*p1, rect), pitch_to_canvas(*p2, rect), color, thickness, cv2.LINE_AA)


def draw_rect_m(
    canvas: np.ndarray,
    rect: Tuple[int, int, int, int],
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    p1 = pitch_to_canvas(x_min, y_min, rect)
    p2 = pitch_to_canvas(x_max, y_max, rect)
    cv2.rectangle(
        canvas,
        (min(p1[0], p2[0]), min(p1[1], p2[1])),
        (max(p1[0], p2[0]), max(p1[1], p2[1])),
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_pitch(width: int, height: int, frame_number: int, source: str) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    canvas = np.full((height, width, 3), (35, 111, 58), dtype=np.uint8)
    rect = field_rect(width, height)
    x0, y0, field_w, field_h = rect
    white = (238, 238, 238)

    cv2.rectangle(canvas, (x0, y0), (x0 + field_w, y0 + field_h), white, 2, cv2.LINE_AA)
    draw_line_m(canvas, rect, (0.0, -HALF_WIDTH_M), (0.0, HALF_WIDTH_M), white, 2)
    center = pitch_to_canvas(0.0, 0.0, rect)
    cv2.circle(canvas, center, int(round(9.15 / PITCH_WIDTH_M * field_h)), white, 2, cv2.LINE_AA)
    cv2.circle(canvas, center, 3, white, -1, cv2.LINE_AA)

    penalty_width = 40.32
    goal_area_width = 18.32
    draw_rect_m(canvas, rect, -HALF_LENGTH_M, -penalty_width / 2, -HALF_LENGTH_M + 16.5, penalty_width / 2, white)
    draw_rect_m(canvas, rect, HALF_LENGTH_M - 16.5, -penalty_width / 2, HALF_LENGTH_M, penalty_width / 2, white)
    draw_rect_m(canvas, rect, -HALF_LENGTH_M, -goal_area_width / 2, -HALF_LENGTH_M + 5.5, goal_area_width / 2, white)
    draw_rect_m(canvas, rect, HALF_LENGTH_M - 5.5, -goal_area_width / 2, HALF_LENGTH_M, goal_area_width / 2, white)
    cv2.circle(canvas, pitch_to_canvas(-HALF_LENGTH_M + 11.0, 0.0, rect), 3, white, -1, cv2.LINE_AA)
    cv2.circle(canvas, pitch_to_canvas(HALF_LENGTH_M - 11.0, 0.0, rect), 3, white, -1, cv2.LINE_AA)

    cv2.putText(
        canvas,
        f"frame {frame_number:06d} | H: {source}",
        (18, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return canvas, rect


def detection_color(det: Detection) -> Tuple[int, int, int]:
    if det.category_id == BALL_CATEGORY_ID:
        return (245, 245, 245)
    if det.category_id == REFEREE_CATEGORY_ID:
        return (0, 220, 255)
    if det.team == "left":
        return (255, 122, 56)
    if det.team == "right":
        return (62, 77, 255)
    return (235, 235, 235)


def detection_radius(det: Detection) -> int:
    if det.category_id == BALL_CATEGORY_ID:
        return 5
    if det.category_id == REFEREE_CATEGORY_ID:
        return 6
    return 7


def scene_image_path(labels_path: Path, info: Dict[str, Any], frame: FrameInfo) -> Path:
    return labels_path.parent / str(info.get("im_dir") or "img1") / frame.file_name


def placeholder_source_frame(height: int = 1080, width: int = 1920) -> np.ndarray:
    frame = np.full((height, width, 3), (24, 24, 24), dtype=np.uint8)
    cv2.putText(frame, "source frame unavailable", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (220, 220, 220), 2)
    return frame


def read_source_frame(
    labels_path: Path,
    info: Dict[str, Any],
    frame: FrameInfo,
    video_capture: Optional[cv2.VideoCapture],
) -> np.ndarray:
    if video_capture is not None and video_capture.isOpened():
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame.frame_number - 1))
        ok, image = video_capture.read()
        if ok and image is not None:
            return image
    image = cv2.imread(str(scene_image_path(labels_path, info, frame)))
    if image is not None:
        return image
    return placeholder_source_frame(frame.height or 1080, frame.width or 1920)


def draw_source_boxes(source: np.ndarray, detections: Sequence[Detection], show_track_ids: bool) -> np.ndarray:
    image = source.copy()
    for det in detections:
        try:
            x = int(round(float(det.bbox_image["x"])))
            y = int(round(float(det.bbox_image["y"])))
            w = int(round(float(det.bbox_image["w"])))
            h = int(round(float(det.bbox_image["h"])))
        except (KeyError, TypeError, ValueError):
            continue
        color = detection_color(det)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)
        point = bbox_center(det.bbox_image) if det.category_id == BALL_CATEGORY_ID else bbox_bottom_middle(det.bbox_image)
        if point is not None:
            cv2.circle(image, (int(round(point[0])), int(round(point[1]))), 4, color, -1, cv2.LINE_AA)
        if show_track_ids and det.track_id is not None:
            cv2.putText(image, str(det.track_id), (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image


def resize_to_height(image: np.ndarray, target_height: int, target_width: Optional[int] = None) -> np.ndarray:
    if target_width is None:
        target_width = int(round(image.shape[1] * target_height / image.shape[0]))
    target_width = max(2, target_width + (target_width % 2))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def compose_side_by_side(source: np.ndarray, bev: np.ndarray, source_width: int) -> np.ndarray:
    source_resized = resize_to_height(source, bev.shape[0], source_width)
    return np.hstack([source_resized, bev])


def derived_output_path(output_path: Path, suffix: str) -> Path:
    return output_path.with_name(f"{output_path.stem}_{suffix}{output_path.suffix}")


def even_dimensions(width: int, height: int) -> Tuple[int, int]:
    width = width - 1 if width % 2 == 1 else width
    height = height - 1 if height % 2 == 1 else height
    return max(width, 2), max(height, 2)


def resize_to_even_frame(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    width, height = size
    if image.shape[1] == width and image.shape[0] == height:
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def inside_pitch_with_margin(point: Tuple[float, float], margin_m: float) -> bool:
    x, y = point
    return -HALF_LENGTH_M - margin_m <= x <= HALF_LENGTH_M + margin_m and -HALF_WIDTH_M - margin_m <= y <= HALF_WIDTH_M + margin_m


def smooth_detection_point(
    det: Detection,
    point: Tuple[float, float],
    smoothed_positions: Dict[Tuple[int, Optional[int]], Tuple[float, float]],
    position_smoothing: float,
    max_position_step_m: float,
) -> Tuple[float, float]:
    key = (det.category_id, det.track_id)
    if det.track_id is None:
        return point
    previous = smoothed_positions.get(key)
    if previous is None:
        smoothed_positions[key] = point
        return point

    prev_arr = np.asarray(previous, dtype=np.float64)
    curr_arr = np.asarray(point, dtype=np.float64)
    delta = curr_arr - prev_arr
    distance = float(np.linalg.norm(delta))
    if max_position_step_m > 0 and distance > max_position_step_m:
        curr_arr = prev_arr + delta / distance * max_position_step_m
    alpha = min(max(position_smoothing, 0.0), 1.0)
    smoothed = tuple((prev_arr * (1.0 - alpha) + curr_arr * alpha).tolist())
    smoothed_positions[key] = smoothed
    return smoothed


def compute_velocity_mps(
    track_positions: Dict[Tuple[int, int], List[Tuple[int, float, float]]],
    key: Tuple[int, int],
    window: int,
    fps: float,
) -> Optional[float]:
    history = track_positions.get(key)
    if not history or len(history) < 2 or fps <= 0:
        return None

    current = history[-1]
    past_index = max(0, len(history) - 1 - max(1, window))
    past = history[past_index]
    d_frames = current[0] - past[0]
    if d_frames <= 0:
        return None

    distance_m = math.hypot(current[1] - past[1], current[2] - past[2])
    dt = d_frames / fps
    if dt <= 0:
        return None
    return distance_m / dt


def draw_detections(
    canvas: np.ndarray,
    rect: Tuple[int, int, int, int],
    detections: Sequence[Detection],
    projected: Sequence[Optional[Tuple[float, float]]],
    histories: Dict[Tuple[int, Optional[int]], deque],
    smoothed_positions: Dict[Tuple[int, Optional[int]], Tuple[float, float]],
    track_positions: Dict[Tuple[int, int], List[Tuple[int, float, float]]],
    velocity_rows: List[Dict[str, Any]],
    frame_number: int,
    fps: float,
    trail_frames: int,
    show_track_ids: bool,
    pitch_margin_m: float,
    position_smoothing: float,
    max_position_step_m: float,
    draw_velocity: bool,
    velocity_window: int,
) -> int:
    drawn = 0
    for det, point in zip(detections, projected):
        if point is None or not inside_pitch_with_margin(point, pitch_margin_m):
            continue
        key = (det.category_id, det.track_id)
        point = smooth_detection_point(det, point, smoothed_positions, position_smoothing, max_position_step_m)

        if trail_frames > 0 and det.track_id is not None:
            history = histories.setdefault(key, deque(maxlen=trail_frames))
            history.append(point)
            color = detection_color(det)
            pts = [pitch_to_canvas(x, y, rect) for x, y in history]
            for idx in range(1, len(pts)):
                alpha = idx / max(1, len(pts) - 1)
                trail_color = tuple(int(channel * (0.25 + 0.55 * alpha)) for channel in color)
                cv2.line(canvas, pts[idx - 1], pts[idx], trail_color, 2, cv2.LINE_AA)

        px, py = pitch_to_canvas(point[0], point[1], rect)
        color = detection_color(det)
        radius = detection_radius(det)
        cv2.circle(canvas, (px, py), radius + 2, (24, 42, 31), -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), radius, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), radius, (255, 255, 255), 1, cv2.LINE_AA)

        if draw_velocity and det.track_id is not None and det.category_id != BALL_CATEGORY_ID:
            velocity_key = (det.category_id, int(det.track_id))
            track_positions.setdefault(velocity_key, []).append((frame_number, point[0], point[1]))
            speed_mps = compute_velocity_mps(track_positions, velocity_key, velocity_window, fps)
            if speed_mps is not None:
                speed_text = f"{speed_mps:.1f}"
                cv2.putText(
                    canvas,
                    speed_text,
                    (px - 11, py - radius - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                velocity_rows.append(
                    {
                        "frame_number": frame_number,
                        "image_id": det.image_id,
                        "track_id": det.track_id,
                        "category_id": det.category_id,
                        "role": det.role,
                        "team": det.team or "",
                        "x_projected_m": round(point[0], 6),
                        "y_projected_m": round(point[1], 6),
                        "speed_mps": round(speed_mps, 6),
                        "speed_kmh": round(speed_mps * 3.6, 6),
                        "window_frames": max(1, velocity_window),
                    }
                )

        if show_track_ids and det.track_id is not None and det.category_id != BALL_CATEGORY_ID:
            cv2.putText(canvas, str(det.track_id), (px + radius + 4, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1)
        drawn += 1
    return drawn


def fourcc_for_output(path: Path) -> int:
    return cv2.VideoWriter_fourcc(*"XVID") if path.suffix.lower() == ".avi" else cv2.VideoWriter_fourcc(*"mp4v")


def build_homographies(
    frames: Sequence[FrameInfo],
    annotations_by_image: Dict[str, List[Detection]],
    args: argparse.Namespace,
) -> Tuple[Dict[int, HomographyInfo], List[Tuple[FrameInfo, HomographyInfo]]]:
    if args.homography_csv is not None:
        loaded = load_homographies(args.homography_csv)
        rows = [(frame, loaded.get(frame.frame_number, HomographyInfo(None, source="missing-loaded"))) for frame in frames]
        return loaded, rows

    homographies: Dict[int, HomographyInfo] = {}
    rows: List[Tuple[FrameInfo, HomographyInfo]] = []
    last_valid: Optional[np.ndarray] = None
    last_smoothed: Optional[np.ndarray] = None
    reuse_last = not args.no_reuse_last_homography

    for frame in frames:
        info = estimate_homography(
            annotations_by_image.get(frame.image_id, []),
            args.min_homography_points,
            args.ransac_threshold,
            args.include_referees_in_homography,
            args.homography_footpoints,
        )
        if info.matrix is not None:
            raw_matrix = normalize_homography(info.matrix)
            smoothed, jump = smooth_homography(last_smoothed, raw_matrix, frame, args.homography_smoothing, args.max_homography_jump_m)
            source = info.source
            if last_smoothed is not None and args.homography_smoothing < 1.0:
                source = f"smoothed-{info.source}"
                if jump is not None and jump > args.max_homography_jump_m:
                    source = f"guarded-{source}"
            info = HomographyInfo(smoothed, inliers=info.inliers, points=info.points, source=source, jump_m=jump)
            last_valid = raw_matrix.copy()
            last_smoothed = smoothed.copy()
        elif reuse_last and last_valid is not None:
            reuse_matrix = last_smoothed if last_smoothed is not None else last_valid
            info = HomographyInfo(reuse_matrix.copy(), inliers=0, points=info.points, source=f"reused-{info.source}")
        homographies[frame.frame_number] = info
        rows.append((frame, info))
    return homographies, rows


def write_debug_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render(args: argparse.Namespace) -> None:
    keep_bbox_pitch = args.homography_csv is None
    frames, annotations_by_image, info = load_annotations(args.labels, keep_bbox_pitch=keep_bbox_pitch)
    selected_frames = filter_frames(frames, args.start_frame, args.end_frame, args.max_frames)
    if not selected_frames:
        raise SystemExit("No frames selected. Check --start-frame/--end-frame/--max-frames.")

    fps = output_fps(args, info)
    original_output = args.original_output or derived_output_path(args.output, "original")
    side_by_side_output = args.side_by_side_output or derived_output_path(args.output, "side_by_side")
    debug_csv = args.debug_csv
    velocity_csv = args.velocity_csv
    homography_csv_output = args.save_homography_csv
    if args.debug:
        debug_csv = debug_csv or derived_output_path(args.output, "debug").with_suffix(".csv")
        if args.velocity:
            velocity_csv = velocity_csv or derived_output_path(args.output, "velocity").with_suffix(".csv")
        homography_csv_output = homography_csv_output or derived_output_path(args.output, "homographies").with_suffix(".csv")

    homographies, homography_rows = build_homographies(selected_frames, annotations_by_image, args)

    source_video_capture: Optional[cv2.VideoCapture] = None
    if args.video is not None:
        source_video_capture = cv2.VideoCapture(str(args.video))
        if not source_video_capture.isOpened():
            source_video_capture.release()
            source_video_capture = None

    first_source = read_source_frame(args.labels, info, selected_frames[0], source_video_capture)
    original_size = even_dimensions(first_source.shape[1], first_source.shape[0])
    source_side_width = int(round(first_source.shape[1] * args.map_height / first_source.shape[0]))
    source_side_width += source_side_width % 2
    side_width = source_side_width + args.map_width
    side_width += side_width % 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    original_output.parent.mkdir(parents=True, exist_ok=True)
    side_by_side_output.parent.mkdir(parents=True, exist_ok=True)

    homography_writer = cv2.VideoWriter(str(args.output), fourcc_for_output(args.output), fps, (args.map_width, args.map_height))
    original_writer = cv2.VideoWriter(str(original_output), fourcc_for_output(original_output), fps, original_size)
    side_writer = cv2.VideoWriter(str(side_by_side_output), fourcc_for_output(side_by_side_output), fps, (side_width, args.map_height))
    if not homography_writer.isOpened():
        raise SystemExit(f"Could not open output video writer: {args.output}")
    if not original_writer.isOpened():
        homography_writer.release()
        raise SystemExit(f"Could not open original video writer: {original_output}")
    if not side_writer.isOpened():
        homography_writer.release()
        original_writer.release()
        raise SystemExit(f"Could not open side-by-side video writer: {side_by_side_output}")

    histories: Dict[Tuple[int, Optional[int]], deque] = {}
    smoothed_positions: Dict[Tuple[int, Optional[int]], Tuple[float, float]] = {}
    track_positions: Dict[Tuple[int, int], List[Tuple[int, float, float]]] = {}
    debug_rows: List[Dict[str, Any]] = []
    velocity_rows: List[Dict[str, Any]] = []
    rendered_categories = {1, 2, 3}
    if args.ball:
        rendered_categories.add(BALL_CATEGORY_ID)

    try:
        for frame in selected_frames:
            homography_info = homographies.get(frame.frame_number, HomographyInfo(None))
            canvas, rect = draw_pitch(args.map_width, args.map_height, frame.frame_number, homography_info.source)

            detections = [det for det in annotations_by_image.get(frame.image_id, []) if det.category_id in rendered_categories]
            image_points = [detection_image_point(det, args.ball_point) for det in detections]
            valid_detections = [det for det, point in zip(detections, image_points) if point is not None]
            valid_points = [point for point in image_points if point is not None]

            if homography_info.matrix is not None and valid_points:
                projected = project_points(homography_info.matrix, valid_points)
            else:
                projected = [None for _ in valid_points]

            drawn = draw_detections(
                canvas,
                rect,
                valid_detections,
                projected,
                histories,
                smoothed_positions,
                track_positions,
                velocity_rows,
                frame.frame_number,
                fps,
                args.trail_frames,
                args.show_track_ids,
                args.pitch_margin_m,
                args.position_smoothing,
                args.max_position_step_m,
                args.velocity,
                args.velocity_window,
            )
            homography_writer.write(canvas)

            source = read_source_frame(args.labels, info, frame, source_video_capture)
            if not args.no_source_boxes:
                source = draw_source_boxes(source, detections, args.show_track_ids)
            original_writer.write(resize_to_even_frame(source, original_size))
            side_writer.write(compose_side_by_side(source, canvas, source_side_width))

            debug_rows.append(
                {
                    "frame_number": frame.frame_number,
                    "image_id": frame.image_id,
                    "homography_source": homography_info.source,
                    "homography_points": homography_info.points,
                    "homography_inliers": homography_info.inliers,
                    "homography_jump_m": "" if homography_info.jump_m is None else round(homography_info.jump_m, 6),
                    "detections": len(detections),
                    "projected": len(valid_points),
                    "drawn": drawn,
                }
            )
    finally:
        homography_writer.release()
        original_writer.release()
        side_writer.release()
        if source_video_capture is not None:
            source_video_capture.release()

    if homography_csv_output is not None:
        write_homographies(homography_csv_output, homography_rows)
    if debug_csv is not None:
        write_debug_rows(debug_csv, debug_rows)
    if velocity_csv is not None:
        write_debug_rows(velocity_csv, velocity_rows)

    valid_h = sum(1 for _, h in homography_rows if h.matrix is not None)
    print(f"Wrote {len(selected_frames)} frames")
    print(f"Original video: {original_output}")
    print(f"Homography video: {args.output}")
    print(f"Side-by-side video: {side_by_side_output}")
    print(f"Homography frames: {valid_h}/{len(selected_frames)}")
    if debug_csv is not None:
        print(f"Debug CSV: {debug_csv}")
    if velocity_csv is not None:
        print(f"Velocity CSV: {velocity_csv}")
    if homography_csv_output is not None:
        print(f"Homography CSV: {homography_csv_output}")


def main() -> None:
    render(parse_args())


if __name__ == "__main__":
    main()
