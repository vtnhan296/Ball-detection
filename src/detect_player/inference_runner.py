"""Folder image sequence inference to annotated mp4."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import cv2
import numpy as np

from .results import serialize_results
from .visualization import draw_results


def load_frames(
    images: Union[str, Path, np.ndarray, list[Union[str, Path, np.ndarray]]],
) -> list[np.ndarray]:
    if isinstance(images, (str, Path, np.ndarray)):
        image_items: Iterable[Union[str, Path, np.ndarray]] = [images]
    else:
        image_items = images

    frames: list[np.ndarray] = []
    for image in image_items:
        if isinstance(image, (str, Path)):
            frame = cv2.imread(str(image))
            if frame is None:
                raise FileNotFoundError(f"Cannot read image: {image}")
        else:
            frame = image
        if not isinstance(frame, np.ndarray) or frame.ndim < 2:
            raise ValueError("Each image must be a path or a BGR numpy array.")
        frames.append(frame)
    return frames


def list_images(
    folder: Path,
    extensions: Iterable[str],
    recursive: bool = False,
) -> list[Path]:
    extension_set = {ext.lower() for ext in extensions}
    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in folder.glob(pattern)
        if path.is_file() and path.suffix.lower() in extension_set
    )


def predict_folder_to_mp4(
    classifier: Any,
    folder_path: Union[str, Path],
    output_video_path: Optional[Union[str, Path]] = None,
    output_json_path: Optional[Union[str, Path]] = None,
    image_extensions: Optional[Iterable[str]] = None,
    recursive: bool = False,
    max_images: Optional[int] = None,
    write_json: bool = True,
    output_fps: float = 25.0,
    progress: bool = True,
) -> dict[str, Any]:
    """Run inference on sorted folder images and write one annotated mp4."""

    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Cannot open image folder: {folder}")
    if max_images is not None and max_images < 1:
        raise ValueError("max_images must be >= 1 when provided")

    extensions = [
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in (image_extensions or [".jpg", ".jpeg", ".png", ".bmp", ".webp"])
    ]
    images = list_images(folder, extensions=extensions, recursive=recursive)
    if max_images is not None:
        images = images[:max_images]
    if not images:
        raise FileNotFoundError(f"No images found in folder: {folder}")

    default_stem = f"{folder.name}_team_classifier"
    output_video = (
        Path(output_video_path)
        if output_video_path is not None
        else classifier.config.output_dir / f"{default_stem}.mp4"
    )
    output_json = (
        Path(output_json_path)
        if output_json_path is not None
        else classifier.config.output_dir / f"{default_stem}.json"
    )
    output_video.parent.mkdir(parents=True, exist_ok=True)
    if write_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)

    writer: Optional[cv2.VideoWriter] = None
    writer_size: Optional[tuple[int, int]] = None
    image_records: list[dict[str, Any]] = []
    total_detections = 0

    try:
        for image_index, image_path in enumerate(images):
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise FileNotFoundError(f"Cannot read image: {image_path}")

            results = classifier.predict(frame)
            total_detections += len(results)
            annotated = draw_results(frame, results)

            if writer is None:
                height, width = annotated.shape[:2]
                writer_size = (width, height)
                writer = cv2.VideoWriter(
                    str(output_video),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    output_fps,
                    writer_size,
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot create video writer: {output_video}")
            elif writer_size is not None:
                width, height = writer_size
                if annotated.shape[1] != width or annotated.shape[0] != height:
                    annotated = cv2.resize(annotated, (width, height))

            writer.write(annotated)
            image_records.append(
                {
                    "image_index": image_index,
                    "image_path": str(image_path),
                    "relative_path": str(image_path.relative_to(folder)),
                    "detections": serialize_results(results),
                }
            )

            if progress and (image_index + 1) % 25 == 0:
                print(
                    f"Processed {image_index + 1} images "
                    f"({total_detections} detections)"
                )
    finally:
        if writer is not None:
            writer.release()

    summary = {
        "folder_path": str(folder),
        "output_video_path": str(output_video),
        "output_json_path": str(output_json) if write_json else None,
        "images_found": len(images),
        "images_processed": len(image_records),
        "detections": total_detections,
        "image_extensions": extensions,
        "output_fps": output_fps,
    }

    if write_json:
        output_json.write_text(
            json.dumps({"summary": summary, "images": image_records}, indent=2),
            encoding="utf-8",
        )

    return summary
