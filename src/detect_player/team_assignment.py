"""Role parsing and two-team assignment logic."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from .reid_backend import require_dependency
from .results import DetectionResult

log = logging.getLogger(__name__)

# PRTReID's role classifier was trained with five SoccerNet role labels.
# The local YOLO detector only predicts player/goalkeeper/referee.
PRTREID_ROLE_MAPPING = {
    0: "ball",
    1: "goalkeeper",
    2: "other",
    3: "player",
    4: "referee",
}

YOLO_ROLE_NAMES = {"player", "goalkeeper", "referee"}


def resolve_yolo_classes(
    yolo_class_names: dict[int, str],
    requested: Optional[list[int]],
) -> set[int]:
    if requested is not None:
        return {int(value) for value in requested}

    names_lower = {
        class_id: name.lower() for class_id, name in yolo_class_names.items()
    }
    if any(name in YOLO_ROLE_NAMES for name in names_lower.values()):
        return set(names_lower.keys())
    return {0}


def yolo_model_has_roles(yolo_class_names: dict[int, str]) -> bool:
    return any(
        name.lower() in YOLO_ROLE_NAMES for name in yolo_class_names.values()
    )


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 1:
        scores = scores.reshape(-1)
    if scores.size == 0:
        return scores
    if np.any(scores < 0) or not np.isclose(float(scores.sum()), 1.0, atol=1e-2):
        shifted = scores - np.max(scores)
        exp = np.exp(shifted)
        denom = exp.sum()
        return exp / denom if denom else exp
    return scores


def role_from_scores(scores: Optional[np.ndarray]) -> tuple[str, float]:
    if scores is None:
        return "unknown", 0.0

    probs = normalize_scores(scores)
    if probs.size == 0:
        return "unknown", 0.0

    role_idx = int(np.argmax(probs))
    role_name = PRTREID_ROLE_MAPPING.get(role_idx, "unknown")
    return role_name, float(probs[role_idx])


def assign_roles(
    metadata: list[dict[str, Any]],
    embeddings: np.ndarray,
    role_scores: Optional[np.ndarray],
    yolo_has_roles: bool,
) -> list[DetectionResult]:
    results: list[DetectionResult] = []

    for index, meta in enumerate(metadata):
        reid_role, reid_conf = role_from_scores(
            None if role_scores is None else role_scores[index]
        )
        yolo_role = str(meta["yolo_class_name"]).lower()

        if yolo_has_roles and yolo_role in YOLO_ROLE_NAMES:
            role = yolo_role
            role_source = "yolo"
            role_confidence = float(meta["detection_confidence"])
        else:
            role = reid_role
            role_source = "prtreid"
            role_confidence = reid_conf

        results.append(
            DetectionResult(
                image_idx=int(meta["image_idx"]),
                bbox_xyxy=list(meta["bbox_xyxy"]),
                detection_confidence=float(meta["detection_confidence"]),
                yolo_class_id=int(meta["yolo_class_id"]),
                yolo_class_name=str(meta["yolo_class_name"]),
                role=role,
                role_source=role_source,
                role_confidence=float(role_confidence),
                reid_role=reid_role,
                reid_role_confidence=float(reid_conf),
                embedding=embeddings[index],
            )
        )

    return results


def assign_teams(results: list[DetectionResult]) -> None:
    player_indices = [
        idx for idx, result in enumerate(results) if result.role == "player"
    ]
    goalkeeper_indices = [
        idx for idx, result in enumerate(results) if result.role == "goalkeeper"
    ]

    if len(player_indices) == 1:
        result = results[player_indices[0]]
        result.team_id = 0
        result.side_label = "left"
        assign_goalkeepers_by_position(results, goalkeeper_indices)
        return

    if len(player_indices) < 2:
        assign_goalkeepers_by_position(results, goalkeeper_indices)
        return

    require_dependency("sklearn", "pip install scikit-learn")
    from sklearn.cluster import KMeans

    player_embs = np.stack([results[idx].embedding for idx in player_indices])
    if player_embs.ndim > 2:
        player_embs = player_embs.reshape(len(player_indices), -1)

    try:
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(player_embs)
        labels = kmeans.labels_
    except Exception as exc:
        log.warning("KMeans team clustering failed (%s); using x-position.", exc)
        assign_players_by_position(results, player_indices)
        assign_goalkeepers_by_position(results, goalkeeper_indices)
        return

    cluster_x = {0: [], 1: []}
    for label, result_idx in zip(labels, player_indices):
        cluster_x[int(label)].append(bbox_center_x(results[result_idx].bbox_xyxy))

    avg_x = {
        label: float(np.mean(values)) if values else float("inf")
        for label, values in cluster_x.items()
    }
    left_label = 0 if avg_x[0] <= avg_x[1] else 1
    right_label = 1 - left_label
    label_to_team = {left_label: 0, right_label: 1}
    label_to_side = {left_label: "left", right_label: "right"}

    for label, result_idx in zip(labels, player_indices):
        label_int = int(label)
        results[result_idx].team_id = label_to_team[label_int]
        results[result_idx].side_label = label_to_side[label_int]

    centers = np.asarray(kmeans.cluster_centers_)
    assign_goalkeepers_by_embedding(
        results=results,
        goalkeeper_indices=goalkeeper_indices,
        centers=centers,
        label_to_team=label_to_team,
        label_to_side=label_to_side,
    )


def assign_players_by_position(
    results: list[DetectionResult],
    player_indices: list[int],
) -> None:
    centers = [(bbox_center_x(results[idx].bbox_xyxy), idx) for idx in player_indices]
    centers.sort()
    midpoint = len(centers) // 2
    for rank, (_, idx) in enumerate(centers):
        is_left = rank < midpoint
        results[idx].team_id = 0 if is_left else 1
        results[idx].side_label = "left" if is_left else "right"


def assign_goalkeepers_by_embedding(
    results: list[DetectionResult],
    goalkeeper_indices: list[int],
    centers: np.ndarray,
    label_to_team: dict[int, int],
    label_to_side: dict[int, str],
) -> None:
    if not goalkeeper_indices:
        return

    for idx in goalkeeper_indices:
        embedding = results[idx].embedding
        if embedding is None:
            continue
        flat_embedding = np.asarray(embedding).reshape(1, -1)
        distances = np.linalg.norm(centers - flat_embedding, axis=1)
        label = int(np.argmin(distances))
        results[idx].team_id = label_to_team[label]
        results[idx].side_label = label_to_side[label]


def assign_goalkeepers_by_position(
    results: list[DetectionResult],
    goalkeeper_indices: list[int],
) -> None:
    if not goalkeeper_indices:
        return

    player_centers = [
        bbox_center_x(result.bbox_xyxy) for result in results if result.role == "player"
    ]
    pivot = float(np.median(player_centers)) if player_centers else None

    for idx in goalkeeper_indices:
        cx = bbox_center_x(results[idx].bbox_xyxy)
        if pivot is None:
            results[idx].team_id = 0
            results[idx].side_label = "left"
        elif cx <= pivot:
            results[idx].team_id = 0
            results[idx].side_label = "left"
        else:
            results[idx].team_id = 1
            results[idx].side_label = "right"


def bbox_center_x(bbox_xyxy: list[int]) -> float:
    return (float(bbox_xyxy[0]) + float(bbox_xyxy[2])) / 2.0
