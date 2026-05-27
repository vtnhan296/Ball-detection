"""YOLO + PRTReID player role and team inference facade.

The public API is intentionally small:

    clf = PlayerRoleTeamClassifier.from_project_defaults()
    results = clf.predict("frame.jpg")
    clf.predict_folder("frames/", output_video_path="annotated.mp4")

Embeddings stay in memory by default and are omitted from JSON unless
explicitly requested.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import cv2
import numpy as np
import torch

from .config import PlayerTeamClassifierConfig, default_device, find_project_root
from .inference_runner import (
    load_frames,
    predict_folder_to_mp4,
)
from .reid_backend import MissingDependencyError, PRTReIDBackend, require_dependency
from .results import (
    DetectionResult,
    serialize_results,
)
from .team_assignment import (
    assign_roles,
    assign_teams,
    resolve_yolo_classes,
    yolo_model_has_roles,
)
from .visualization import draw_image


class PlayerRoleTeamClassifier:
    """End-to-end player role and two-team inference."""

    def __init__(self, config: PlayerTeamClassifierConfig):
        self.config = config
        self.device = config.device
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.config.yolo_weights.exists():
            raise FileNotFoundError(f"Missing YOLO weights: {self.config.yolo_weights}")

        require_dependency("ultralytics", "pip install ultralytics")
        from ultralytics import YOLO

        self.yolo = YOLO(str(self.config.yolo_weights))
        self.yolo_class_names = {
            int(key): str(value) for key, value in self.yolo.names.items()
        }
        self.yolo_classes = resolve_yolo_classes(
            self.yolo_class_names,
            config.yolo_classes,
        )
        self.yolo_has_roles = yolo_model_has_roles(self.yolo_class_names)

        self.reid_backend = PRTReIDBackend(config)
        self.prtreid_ckpt = self.reid_backend.prtreid_ckpt
        self.hrnet_pretrained = self.reid_backend.hrnet_pretrained

    @classmethod
    def from_project_defaults(
        cls,
        project_root: Optional[Union[str, Path]] = None,
        **overrides: Any,
    ) -> "PlayerRoleTeamClassifier":
        config = PlayerTeamClassifierConfig.from_project_defaults(
            project_root=project_root,
            **overrides,
        )
        return cls(config)

    @torch.no_grad()
    def predict(
        self,
        images: Union[str, Path, np.ndarray, list[Union[str, Path, np.ndarray]]],
    ) -> list[DetectionResult]:
        frames = load_frames(images)
        crops, metadata = self._detect_yolo(frames)
        if not crops:
            return []

        embeddings, role_scores = self.reid_backend.extract_features(crops)
        results = assign_roles(
            metadata=metadata,
            embeddings=embeddings,
            role_scores=role_scores,
            yolo_has_roles=self.yolo_has_roles,
        )
        assign_teams(results)
        return results

    def predict_folder(
        self,
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
        return predict_folder_to_mp4(
            self,
            folder_path=folder_path,
            output_video_path=output_video_path,
            output_json_path=output_json_path,
            image_extensions=image_extensions,
            recursive=recursive,
            max_images=max_images,
            write_json=write_json,
            output_fps=output_fps,
            progress=progress,
        )

    def _detect_yolo(
        self,
        frames: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
        crops: list[np.ndarray] = []
        metadata: list[dict[str, Any]] = []

        for image_idx, frame in enumerate(frames):
            predict_kwargs: dict[str, Any] = {
                "conf": self.config.yolo_conf,
                "verbose": False,
                "device": self.device,
            }
            if self.yolo_classes:
                predict_kwargs["classes"] = sorted(self.yolo_classes)

            yolo_result = self.yolo.predict(frame, **predict_kwargs)[0]
            boxes = yolo_result.boxes
            height, width = frame.shape[:2]

            for box_idx in range(len(boxes)):
                class_id = int(boxes.cls[box_idx].item())
                if class_id not in self.yolo_classes:
                    continue

                x1, y1, x2, y2 = boxes.xyxy[box_idx].cpu().numpy().astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crops.append(crop_rgb)
                metadata.append(
                    {
                        "image_idx": image_idx,
                        "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                        "detection_confidence": float(boxes.conf[box_idx].item()),
                        "yolo_class_id": class_id,
                        "yolo_class_name": self.yolo_class_names.get(
                            class_id, "unknown"
                        ).lower(),
                    }
                )

        return crops, metadata

    def draw(
        self,
        image: Union[str, Path, np.ndarray],
        results: list[Union[DetectionResult, dict[str, Any]]],
        output_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        return draw_image(image, results, output_path=output_path)

    def save_json(
        self,
        results: list[Union[DetectionResult, dict[str, Any]]],
        output_path: Union[str, Path],
        include_embedding: bool = False,
    ) -> None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = serialize_results(results, include_embedding=include_embedding)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    from .cli import main

    raise SystemExit(main())
