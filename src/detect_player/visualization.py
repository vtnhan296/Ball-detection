"""Drawing helpers for detection results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import cv2
import numpy as np

from .results import DetectionResult, result_to_dict


def draw_image(
    image: Union[str, Path, np.ndarray],
    results: list[Union[DetectionResult, dict[str, Any]]],
    output_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    frame = cv2.imread(str(image)) if isinstance(image, (str, Path)) else image.copy()
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {image}")

    return draw_results(frame, results, output_path=output_path)


def draw_results(
    frame: np.ndarray,
    results: list[Union[DetectionResult, dict[str, Any]]],
    output_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    colors = {
        0: (255, 100, 0),
        1: (0, 100, 255),
        None: (200, 200, 200),
    }

    for result in results:
        item = result_to_dict(result)
        x1, y1, x2, y2 = [int(v) for v in item["bbox_xyxy"]]
        team_id = item.get("team_id")
        color = colors.get(team_id, colors[None])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        role = item.get("role", "unknown")
        conf = float(item.get("role_confidence", 0.0))
        label = f"{role} {conf:.2f}"
        if item.get("side_label") is not None:
            label += f" [{item['side_label']}]"

        cv2.putText(
            frame,
            label,
            (x1, max(15, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), frame)
    return frame
