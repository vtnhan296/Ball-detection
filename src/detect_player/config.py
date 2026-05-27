"""Configuration and path defaults for detect-player inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import torch


def find_project_root(start: Optional[Union[str, Path]] = None) -> Path:
    """Find the SoccerNet workspace root from root or any child path."""

    current = Path(start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent

    marker_sets = [
        ("data", "notebooks", "outputs"),
        ("data", "outputs"),
        ("README.md", "notebooks"),
    ]
    for candidate in [current, *current.parents]:
        for markers in marker_sets:
            if all((candidate / marker).exists() for marker in markers):
                return candidate
    return current


def default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@dataclass
class PlayerTeamClassifierConfig:
    """Runtime config for role/team inference."""

    project_root: Path
    yolo_weights: Path
    reid_weights_dir: Path
    output_dir: Path
    device: str = "cpu"
    yolo_conf: float = 0.30
    yolo_classes: Optional[list[int]] = None
    reid_batch_size: int = 32
    download_reid_weights: bool = True

    @classmethod
    def from_project_defaults(
        cls,
        project_root: Optional[Union[str, Path]] = None,
        **overrides: Any,
    ) -> "PlayerTeamClassifierConfig":
        root = find_project_root(project_root)
        device = overrides.pop("device", default_device())
        yolo_weights = (
            root
            / "outputs"
            / "detect_player"
            / "runs"
            / "E1_yolo_fullframe_img960"
            / "weights"
            / "best.pt"
        )
        config = cls(
            project_root=root,
            yolo_weights=yolo_weights,
            reid_weights_dir=root / "models" / "reid",
            output_dir=root / "outputs" / "detect_player" / "infer" / "team_classifier",
            device=device,
        )

        for key, value in overrides.items():
            if not hasattr(config, key):
                raise TypeError(f"Unknown config override: {key}")
            setattr(config, key, value)

        config.yolo_weights = Path(config.yolo_weights)
        config.reid_weights_dir = Path(config.reid_weights_dir)
        config.output_dir = Path(config.output_dir)
        return config
