#!/usr/bin/env python3
"""Generate Bird's Eye View (BEV) videos from SoccerNet GameState 2024 annotations.

Reads Labels-GameState.json, renders a top-down pitch diagram with player/ball
positions for each frame, and compiles frames into an MP4 video.
"""

import json
import math
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont

# --- Pitch dimensions (meters) ---
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
HALF_LENGTH = PITCH_LENGTH / 2  # 52.5
HALF_WIDTH = PITCH_WIDTH / 2    # 34.0

# Penalty area: 16.5m from goal line, 20.16m each side of center
PENALTY_AREA_DEPTH = 16.5
PENALTY_AREA_HALF_WIDTH = 20.16

# Goal area: 5.5m from goal line, 9.16m each side of center
GOAL_AREA_DEPTH = 5.5
GOAL_AREA_HALF_WIDTH = 9.16

# Center circle radius
CENTER_CIRCLE_RADIUS = 9.15

# Penalty spot distance from goal line
PENALTY_SPOT_DIST = 11.0

# Goal dimensions
GOAL_HALF_WIDTH = 3.66
GOAL_DEPTH = 2.0

# --- Rendering config ---
SCALE = 10  # pixels per meter
MARGIN = 30  # pixels padding around the pitch
PITCH_PX_W = int(PITCH_LENGTH * SCALE)  # 1050
PITCH_PX_H = int(PITCH_WIDTH * SCALE)   # 680
IMG_W = PITCH_PX_W + 2 * MARGIN         # 1110
IMG_H = PITCH_PX_H + 2 * MARGIN         # 740

# Colors
COLOR_FIELD = (34, 139, 34)         # forest green
COLOR_LINES = (255, 255, 255)       # white
COLOR_TEAM_LEFT = (30, 100, 220)    # blue
COLOR_TEAM_RIGHT = (220, 40, 40)    # red
COLOR_REFEREE = (255, 220, 50)      # yellow
COLOR_BALL = (255, 255, 255)        # white
COLOR_BALL_OUTLINE = (0, 0, 0)      # black
COLOR_GK_LEFT = (100, 200, 255)     # light blue
COLOR_GK_RIGHT = (255, 150, 100)    # light orange/salmon
COLOR_BACKGROUND = (20, 25, 20)     # dark background

PLAYER_RADIUS = 8
BALL_RADIUS = 6
LINE_WIDTH = 2
FONT_SIZE = 10

# --- Velocity config ---
VELOCITY_WINDOW = 5       # number of frames to average over for smoothing
COLOR_VELOCITY = (255, 255, 0)  # yellow text for velocity


def pitch_to_pixel(x_meter, y_meter):
    """Convert pitch coordinates (meters) to pixel coordinates on the BEV image.

    Pitch coords: X = -52.5..+52.5 (left goal to right goal, origin at center)
                  Y = -34..+34 (bottom touchline to top touchline, origin at center)

    Pixel coords: (0,0) is top-left of image.
                  X_pixel increases rightward, Y_pixel increases downward.
    """
    px = MARGIN + (x_meter + HALF_LENGTH) * SCALE
    # Y mapping: pitch Y=-34 maps to pixel top, Y=+34 maps to pixel bottom
    py = MARGIN + (HALF_WIDTH + y_meter) * SCALE
    return int(px), int(py)


def draw_pitch_template():
    """Draw a blank soccer pitch with all standard markings. Returns a PIL Image."""
    img = Image.new("RGB", (IMG_W, IMG_H), COLOR_BACKGROUND)
    draw = ImageDraw.Draw(img)

    # Green field
    draw.rectangle(
        [MARGIN, MARGIN, MARGIN + PITCH_PX_W, MARGIN + PITCH_PX_H],
        fill=COLOR_FIELD,
    )

    # Outer boundary (touchlines and goal lines)
    draw.rectangle(
        [MARGIN, MARGIN, MARGIN + PITCH_PX_W, MARGIN + PITCH_PX_H],
        outline=COLOR_LINES, width=LINE_WIDTH,
    )

    # Halfway line
    mid_x = MARGIN + int(PITCH_LENGTH / 2 * SCALE)
    draw.line([(mid_x, MARGIN), (mid_x, MARGIN + PITCH_PX_H)],
              fill=COLOR_LINES, width=LINE_WIDTH)

    # Center circle
    cx, cy = pitch_to_pixel(0, 0)
    r = int(CENTER_CIRCLE_RADIUS * SCALE)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                 outline=COLOR_LINES, width=LINE_WIDTH)

    # Center spot
    draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=COLOR_LINES)

    # --- Left penalty area ---
    pa_left = MARGIN
    pa_right = MARGIN + int(PENALTY_AREA_DEPTH * SCALE)
    pa_top = MARGIN + int((HALF_WIDTH - PENALTY_AREA_HALF_WIDTH) * SCALE)
    pa_bottom = MARGIN + int((HALF_WIDTH + PENALTY_AREA_HALF_WIDTH) * SCALE)
    draw.rectangle([pa_left, pa_top, pa_right, pa_bottom],
                   outline=COLOR_LINES, width=LINE_WIDTH)

    # Left goal area
    ga_right = MARGIN + int(GOAL_AREA_DEPTH * SCALE)
    ga_top = MARGIN + int((HALF_WIDTH - GOAL_AREA_HALF_WIDTH) * SCALE)
    ga_bottom = MARGIN + int((HALF_WIDTH + GOAL_AREA_HALF_WIDTH) * SCALE)
    draw.rectangle([pa_left, ga_top, ga_right, ga_bottom],
                   outline=COLOR_LINES, width=LINE_WIDTH)

    # Left penalty spot
    ps_x, ps_y = pitch_to_pixel(-HALF_LENGTH + PENALTY_SPOT_DIST, 0)
    draw.ellipse([ps_x - 3, ps_y - 3, ps_x + 3, ps_y + 3], fill=COLOR_LINES)

    # Left penalty arc (the arc outside the penalty area)
    arc_cx, arc_cy = pitch_to_pixel(-HALF_LENGTH + PENALTY_SPOT_DIST, 0)
    arc_r = int(CENTER_CIRCLE_RADIUS * SCALE)
    # Draw arc only outside penalty area
    draw.arc([arc_cx - arc_r, arc_cy - arc_r, arc_cx + arc_r, arc_cy + arc_r],
             start=-50, end=50, fill=COLOR_LINES, width=LINE_WIDTH)

    # --- Right penalty area ---
    rpa_left = MARGIN + int((PITCH_LENGTH - PENALTY_AREA_DEPTH) * SCALE)
    rpa_right = MARGIN + PITCH_PX_W
    draw.rectangle([rpa_left, pa_top, rpa_right, pa_bottom],
                   outline=COLOR_LINES, width=LINE_WIDTH)

    # Right goal area
    rga_left = MARGIN + int((PITCH_LENGTH - GOAL_AREA_DEPTH) * SCALE)
    draw.rectangle([rga_left, ga_top, rpa_right, ga_bottom],
                   outline=COLOR_LINES, width=LINE_WIDTH)

    # Right penalty spot
    rps_x, rps_y = pitch_to_pixel(HALF_LENGTH - PENALTY_SPOT_DIST, 0)
    draw.ellipse([rps_x - 3, rps_y - 3, rps_x + 3, rps_y + 3], fill=COLOR_LINES)

    # Right penalty arc
    rarc_cx, rarc_cy = pitch_to_pixel(HALF_LENGTH - PENALTY_SPOT_DIST, 0)
    draw.arc([rarc_cx - arc_r, rarc_cy - arc_r, rarc_cx + arc_r, rarc_cy + arc_r],
             start=130, end=230, fill=COLOR_LINES, width=LINE_WIDTH)

    # --- Goals (behind the goal line) ---
    for goal_x in [0, PITCH_LENGTH]:
        g_top = MARGIN + int((HALF_WIDTH - GOAL_HALF_WIDTH) * SCALE)
        g_bottom = MARGIN + int((HALF_WIDTH + GOAL_HALF_WIDTH) * SCALE)
        if goal_x == 0:
            g_left = MARGIN - int(GOAL_DEPTH * SCALE)
            g_right = MARGIN
        else:
            g_left = MARGIN + PITCH_PX_W
            g_right = MARGIN + PITCH_PX_W + int(GOAL_DEPTH * SCALE)
        draw.rectangle([g_left, g_top, g_right, g_bottom],
                       outline=COLOR_LINES, width=LINE_WIDTH)

    # Corner arcs (quarter circles at corners, radius ~1m)
    corner_r = int(1.0 * SCALE)
    corners = [
        (-HALF_LENGTH, -HALF_WIDTH),       # bottom-left
        (-HALF_LENGTH, HALF_WIDTH),        # top-left
        (HALF_LENGTH, -HALF_WIDTH),        # bottom-right
        (HALF_LENGTH, HALF_WIDTH),         # top-right
    ]
    arc_angles = [(270, 360), (0, 90), (180, 270), (90, 180)]
    for (cx_m, cy_m), (start, end) in zip(corners, arc_angles):
        ccx, ccy = pitch_to_pixel(cx_m, cy_m)
        draw.arc([ccx - corner_r, ccy - corner_r, ccx + corner_r, ccy + corner_r],
                 start=start, end=end, fill=COLOR_LINES, width=LINE_WIDTH)

    return img


def is_on_pitch(x, y, tolerance=3.0):
    """Check if a position is within the pitch bounds (with tolerance in meters)."""
    return (-HALF_LENGTH - tolerance <= x <= HALF_LENGTH + tolerance and
            -HALF_WIDTH - tolerance <= y <= HALF_WIDTH + tolerance)


def get_color(ann):
    """Get the rendering color for an annotation."""
    role = ann["attributes"]["role"]
    team = ann["attributes"].get("team")
    if role == "ball":
        return COLOR_BALL
    if role == "goalkeeper":
        return COLOR_GK_LEFT if team == "left" else COLOR_GK_RIGHT
    if role == "referee":
        return COLOR_REFEREE
    # Player
    if team == "left":
        return COLOR_TEAM_LEFT
    return COLOR_TEAM_RIGHT


def compute_velocity(track_positions, track_id, frame_idx, fps, window=VELOCITY_WINDOW):
    """Compute smoothed velocity (m/s) for a tracked entity.

    Uses displacement of pixel positions on the BEV/homography map over
    the last `window` frames, then converts pixel distance to real-world
    meters using the rendering scale factor.

    Conversion: distance_meters = distance_pixels / SCALE
    where SCALE = pixels per meter (derived from pitch 105x68m mapped
    to PITCH_PX_W x PITCH_PX_H pixels).

    Args:
        track_positions: dict mapping track_id -> list of (frame_idx, px, py)
                         where px, py are pixel coords on the BEV map
        track_id: the track to compute velocity for
        frame_idx: current frame index (0-based)
        fps: video frame rate
        window: number of past frames to look back

    Returns:
        Speed in m/s, or None if insufficient history.
    """
    history = track_positions.get(track_id)
    if not history or len(history) < 2:
        return None

    # history is sorted by frame_idx (we append in order)
    cur = history[-1]
    # Look back up to `window` frames
    past_idx = max(0, len(history) - 1 - window)
    past = history[past_idx]

    d_frames = cur[0] - past[0]
    if d_frames <= 0:
        return None

    # Displacement in pixels on the BEV map
    dpx = cur[1] - past[1]
    dpy = cur[2] - past[2]
    dist_pixels = math.sqrt(dpx * dpx + dpy * dpy)

    # Convert pixel displacement to meters using the scale factor
    # SCALE = pixels per meter (e.g. 10 px/m for 105m pitch -> 1050px)
    dist_meters = dist_pixels / SCALE

    dt = d_frames / fps  # seconds
    speed = dist_meters / dt  # m/s
    return speed


def generate_bev_video(scene_dir, output_dir, progress_callback=None):
    """Generate a BEV video for a single scene.

    Args:
        scene_dir: Path to scene directory (e.g., .../SNGS-116/)
        output_dir: Path to write output files (e.g., output/SNGS-116/)
        progress_callback: Optional callable(current_frame, total_frames)

    Returns:
        Path to the generated MP4 file, or None on failure.
    """
    json_path = os.path.join(scene_dir, "Labels-GameState.json")
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found")
        return None

    with open(json_path, "r") as f:
        data = json.load(f)

    info = data["info"]
    fps = info.get("frame_rate", 25)
    scene_name = info["name"]

    # Index annotations by image_id
    anns_by_frame = defaultdict(list)
    for ann in data["annotations"]:
        if ann.get("category_id") in (1, 2, 3, 4):  # player, gk, ref, ball
            anns_by_frame[ann["image_id"]].append(ann)

    # Create pitch template (reused for every frame)
    pitch_template = draw_pitch_template()

    # Try to load a small font for jersey numbers
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", FONT_SIZE)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Slightly smaller font for velocity display
    try:
        font_velocity = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
    except (OSError, IOError):
        font_velocity = font

    os.makedirs(output_dir, exist_ok=True)

    # Use a temp directory for BEV frames
    tmp_dir = os.path.join(output_dir, "_bev_frames")
    os.makedirs(tmp_dir, exist_ok=True)

    # Track positions history: track_id -> [(frame_idx, px, py), ...]
    # Stores pixel coordinates on the BEV map (after pitch_to_pixel projection)
    track_positions = defaultdict(list)

    total_frames = len(data["images"])
    for idx, img_info in enumerate(data["images"]):
        frame_id = img_info["image_id"]
        frame_num = idx + 1

        # Copy template
        frame_img = pitch_template.copy()
        draw = ImageDraw.Draw(frame_img)

        # Draw frame number / scene info
        draw.text((5, 5), f"{scene_name}  Frame {frame_num}/{total_frames}",
                  fill=COLOR_LINES, font=font)

        frame_anns = anns_by_frame.get(frame_id, [])

        # Separate ball from others so we draw it on top
        others = []
        balls = []
        for ann in frame_anns:
            if ann["attributes"]["role"] == "ball":
                balls.append(ann)
            else:
                others.append(ann)

        # Draw players/referees first
        for ann in others:
            bp = ann.get("bbox_pitch")
            if bp is None:
                continue
            x = bp.get("x_bottom_middle")
            y = bp.get("y_bottom_middle")
            if x is None or y is None:
                continue
            if not is_on_pitch(x, y):
                continue

            px, py = pitch_to_pixel(x, y)
            color = get_color(ann)

            # Record BEV pixel position for velocity tracking
            tid = ann.get("track_id")
            if tid is not None:
                track_positions[tid].append((idx, px, py))

            # Draw player dot with outline
            draw.ellipse(
                [px - PLAYER_RADIUS, py - PLAYER_RADIUS,
                 px + PLAYER_RADIUS, py + PLAYER_RADIUS],
                fill=color, outline=(0, 0, 0), width=1,
            )

            # Draw jersey number (inside the dot)
            jersey = ann["attributes"].get("jersey")
            if jersey:
                draw.text((px - 4, py - 5), str(jersey),
                          fill=(255, 255, 255), font=font)

            # Draw velocity above the player dot
            tid = ann.get("track_id")
            if tid is not None:
                speed = compute_velocity(track_positions, tid, idx, fps)
                if speed is not None:
                    speed_text = f"{speed:.1f}"
                    # Position text above the player dot
                    draw.text((px - 8, py - PLAYER_RADIUS - 12), speed_text,
                              fill=COLOR_VELOCITY, font=font_velocity)

        # Draw ball on top
        for ann in balls:
            bp = ann.get("bbox_pitch")
            if bp is None:
                continue
            x = bp.get("x_bottom_middle")
            y = bp.get("y_bottom_middle")
            if x is None or y is None:
                continue
            if not is_on_pitch(x, y):
                continue

            px, py = pitch_to_pixel(x, y)
            draw.ellipse(
                [px - BALL_RADIUS, py - BALL_RADIUS,
                 px + BALL_RADIUS, py + BALL_RADIUS],
                fill=COLOR_BALL, outline=COLOR_BALL_OUTLINE, width=2,
            )

        # Save frame
        frame_path = os.path.join(tmp_dir, f"{frame_num:06d}.png")
        frame_img.save(frame_path)

        if progress_callback:
            progress_callback(frame_num, total_frames)

    # Compile to MP4 with ffmpeg
    output_path = os.path.join(output_dir, "bev_video.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(tmp_dir, "%06d.png"),
        "-c:v", "libopenh264",
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr}")
        return None

    # Clean up temp frames
    for f in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, f))
    os.rmdir(tmp_dir)

    print(f"Generated: {output_path}")
    return output_path


def generate_source_video(scene_dir, output_dir):
    """Compile source JPEG frames into an MP4 for web playback."""
    json_path = os.path.join(scene_dir, "Labels-GameState.json")
    with open(json_path, "r") as f:
        data = json.load(f)

    fps = data["info"].get("frame_rate", 25)
    img_dir = os.path.join(scene_dir, data["info"]["im_dir"])

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "source_video.mp4")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(img_dir, "%06d.jpg"),
        "-c:v", "libopenh264",
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr}")
        return None

    print(f"Generated: {output_path}")
    return output_path


def generate_sidebyside_video(scene_dir, output_dir):
    """Combine source and BEV videos side-by-side into a single MP4."""
    source_path = os.path.join(output_dir, "source_video.mp4")
    bev_path = os.path.join(output_dir, "bev_video.mp4")
    
    if not os.path.exists(source_path) or not os.path.exists(bev_path):
        print("Missing source or BEV video for side-by-side generation.")
        return None

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sidebyside.mp4")

    # The source is 1920x1080, BEV is 1110x740.
    # We scale BEV to height 1080, maintaining aspect ratio, then horizontally stack them.
    cmd = [
        "ffmpeg", "-y",
        "-i", source_path,
        "-i", bev_path,
        "-filter_complex", "[1:v]scale=-1:1080[v2];[0:v][v2]hstack=inputs=2[v]",
        "-map", "[v]",
        "-c:v", "libopenh264",
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr}")
        return None

    print(f"Generated: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate BEV videos from SoccerNet GameState annotations")
    parser.add_argument("--scene", type=str, nargs="+", help="Scene name(s) (e.g., SNGS-116 SNGS-132). If omitted, processes all.")
    parser.add_argument("--data-dir", type=str, default="data/SoccerNetGS/gamestate-2024",
                        help="Path to the gamestate-2024 directory")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Output directory for generated videos")
    parser.add_argument("--source-video", action="store_true",
                        help="Also generate source video MP4 from frames")
    args = parser.parse_args()

    if args.scene:
        scenes = args.scene
    else:
        scenes = sorted([
            d for d in os.listdir(args.data_dir)
            if d.startswith("SNGS-") and os.path.isdir(os.path.join(args.data_dir, d))
        ])

    for scene_name in scenes:
        scene_dir = os.path.join(args.data_dir, scene_name)
        out_dir = os.path.join(args.output_dir, scene_name)

        def progress(cur, total, name=scene_name):
            print(f"\r  [{name}] Rendering frame {cur}/{total}", end="", flush=True)

        print(f"Processing {scene_name}...")
        generate_bev_video(scene_dir, out_dir, progress_callback=progress)
        print()

        if args.source_video:
            generate_source_video(scene_dir, out_dir)
            generate_sidebyside_video(scene_dir, out_dir)
