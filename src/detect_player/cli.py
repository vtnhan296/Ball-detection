"""Command-line entrypoint for folder-to-mp4 inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .config import default_device
from .player_role_team_classifier import PlayerRoleTeamClassifier


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run player role/team inference on an image folder and write mp4."
    )
    parser.add_argument("folder_path", type=str, help="Folder of input images.")
    parser.add_argument("--output-video", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--no-json", action="store_true")
    args = parser.parse_args(argv)

    classifier = PlayerRoleTeamClassifier.from_project_defaults(
        device=args.device,
        yolo_conf=args.conf,
    )
    summary = classifier.predict_folder(
        Path(args.folder_path),
        output_video_path=args.output_video,
        output_json_path=args.output_json,
        max_images=args.max_images,
        write_json=not args.no_json,
        output_fps=args.fps,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
