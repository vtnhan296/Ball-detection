"""Result dataclasses and JSON-safe serialization helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np


@dataclass
class DetectionResult:
    """One detected object after role and team assignment."""

    image_idx: int
    bbox_xyxy: list[int]
    detection_confidence: float
    yolo_class_id: int
    yolo_class_name: str
    role: str
    role_source: str
    role_confidence: float
    reid_role: Optional[str] = None
    reid_role_confidence: Optional[float] = None
    team_id: Optional[int] = None
    side_label: Optional[str] = None
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self, include_embedding: bool = False) -> dict[str, Any]:
        item: dict[str, Any] = {
            "image_idx": self.image_idx,
            "bbox_xyxy": [int(v) for v in self.bbox_xyxy],
            "detection_confidence": float(self.detection_confidence),
            "yolo_class_id": int(self.yolo_class_id),
            "yolo_class_name": self.yolo_class_name,
            "role": self.role,
            "role_source": self.role_source,
            "role_confidence": float(self.role_confidence),
            "reid_role": self.reid_role,
            "reid_role_confidence": (
                None
                if self.reid_role_confidence is None
                else float(self.reid_role_confidence)
            ),
            "team_id": self.team_id,
            "side_label": self.side_label,
        }
        if include_embedding and self.embedding is not None:
            item["embedding"] = self.embedding.astype(float).tolist()
        return item


def result_to_dict(result: Union[DetectionResult, dict[str, Any]]) -> dict[str, Any]:
    if isinstance(result, DetectionResult):
        return result.to_dict(include_embedding=False)
    return result


def serialize_dict_result(
    result: dict[str, Any],
    include_embedding: bool = False,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in result.items():
        if key == "embedding" and not include_embedding:
            continue
        if isinstance(value, np.ndarray):
            output[key] = value.astype(float).tolist()
        elif isinstance(value, np.generic):
            output[key] = value.item()
        else:
            output[key] = value
    return output


def serialize_results(
    results: list[Union[DetectionResult, dict[str, Any]]],
    include_embedding: bool = False,
) -> list[dict[str, Any]]:
    return [
        (
            result.to_dict(include_embedding=include_embedding)
            if isinstance(result, DetectionResult)
            else serialize_dict_result(result, include_embedding=include_embedding)
        )
        for result in results
    ]
