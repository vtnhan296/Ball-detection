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


def assign_teams_temporal_prototypes(
    results: list[DetectionResult],
    metadata: list[dict[str, Any]],
    calibration_frames: int = 50,
    use_track_pooling: bool = True,
) -> None:
    """Assign teams using early-frame position only to orient ReID prototypes.

    The first `calibration_frames` selected frames define which embedding cluster
    is visually left/right. All detections are then assigned by cosine distance
    to the fixed prototypes, so later field position changes do not flip teams.
    """

    if len(results) != len(metadata):
        log.warning(
            "Temporal team assignment needs aligned results/metadata; "
            "falling back to global team assignment."
        )
        assign_teams(results)
        return

    frame_numbers = [
        int(meta.get("frame_number", meta.get("image_idx", 0)))
        for meta in metadata
    ]
    if not frame_numbers:
        return

    first_frame = min(frame_numbers)
    calibration_end = first_frame + max(1, int(calibration_frames)) - 1
    player_indices = [
        idx
        for idx, result in enumerate(results)
        if result.role == "player" and result.embedding is not None
    ]
    calibration_player_indices = [
        idx
        for idx in player_indices
        if frame_numbers[idx] <= calibration_end
    ]

    if len(calibration_player_indices) < 2:
        log.warning(
            "Temporal team calibration has fewer than 2 player embeddings; "
            "falling back to global team assignment."
        )
        assign_teams(results)
        return

    require_dependency("sklearn", "pip install scikit-learn")
    from sklearn.cluster import KMeans

    calibration_embs = stack_normalized_embeddings(
        [results[idx].embedding for idx in calibration_player_indices]
    )
    try:
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(calibration_embs)
        calibration_labels = kmeans.labels_
    except Exception as exc:
        log.warning(
            "Temporal team calibration KMeans failed (%s); "
            "falling back to global team assignment.",
            exc,
        )
        assign_teams(results)
        return

    label_counts = {
        label: int(np.sum(calibration_labels == label)) for label in (0, 1)
    }
    if label_counts[0] == 0 or label_counts[1] == 0:
        log.warning(
            "Temporal team calibration produced an empty cluster; "
            "falling back to global team assignment."
        )
        assign_teams(results)
        return

    cluster_x = {0: [], 1: []}
    for label, result_idx in zip(calibration_labels, calibration_player_indices):
        cluster_x[int(label)].append(bbox_center_x(results[result_idx].bbox_xyxy))

    avg_x = {
        label: float(np.mean(values)) if values else float("inf")
        for label, values in cluster_x.items()
    }
    left_label = 0 if avg_x[0] <= avg_x[1] else 1
    right_label = 1 - left_label
    label_to_team = {left_label: 0, right_label: 1}
    label_to_side = {left_label: "left", right_label: "right"}

    team_member_embeddings: dict[int, list[np.ndarray]] = {0: [], 1: []}
    for label, result_idx in zip(calibration_labels, calibration_player_indices):
        team_id = label_to_team[int(label)]
        team_member_embeddings[team_id].append(
            normalize_embedding(results[result_idx].embedding)
        )

    prototypes = {
        team_id: normalize_embedding(np.mean(embeddings, axis=0))
        for team_id, embeddings in team_member_embeddings.items()
        if embeddings
    }
    if 0 not in prototypes or 1 not in prototypes:
        log.warning(
            "Temporal team calibration could not build both prototypes; "
            "falling back to global team assignment."
        )
        assign_teams(results)
        return

    if use_track_pooling:
        assign_players_by_track_prototypes(results, metadata, player_indices, prototypes)
    else:
        for idx in player_indices:
            assign_result_by_prototypes(results[idx], prototypes)

    goalkeeper_indices = [
        idx
        for idx, result in enumerate(results)
        if result.role == "goalkeeper" and result.embedding is not None
    ]
    for idx in goalkeeper_indices:
        assign_result_by_prototypes(results[idx], prototypes)


def stack_normalized_embeddings(embeddings: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack([normalize_embedding(embedding) for embedding in embeddings])
    if stacked.ndim > 2:
        stacked = stacked.reshape(len(embeddings), -1)
    return stacked


def normalize_embedding(embedding: Any) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def cosine_distances_to_prototypes(
    embedding: Any,
    prototypes: dict[int, np.ndarray],
) -> dict[int, float]:
    vector = normalize_embedding(embedding)
    return {
        team_id: float(1.0 - np.dot(vector, prototype))
        for team_id, prototype in prototypes.items()
    }


def assign_result_by_prototypes(
    result: DetectionResult,
    prototypes: dict[int, np.ndarray],
    source: str = "prototype",
) -> None:
    distances = cosine_distances_to_prototypes(result.embedding, prototypes)
    if not distances:
        return
    team_id = min(distances, key=distances.get)
    result.team_id = int(team_id)
    result.side_label = "left" if int(team_id) == 0 else "right"
    result.team_assignment_source = source
    result.distance_to_left = distances.get(0)
    result.distance_to_right = distances.get(1)


def assign_players_by_track_prototypes(
    results: list[DetectionResult],
    metadata: list[dict[str, Any]],
    player_indices: list[int],
    prototypes: dict[int, np.ndarray],
) -> None:
    track_embeddings: dict[int, list[np.ndarray]] = {}
    no_track_indices: list[int] = []

    for idx in player_indices:
        track_id = metadata[idx].get("track_id")
        if track_id is None:
            no_track_indices.append(idx)
            continue
        track_embeddings.setdefault(int(track_id), []).append(
            normalize_embedding(results[idx].embedding)
        )

    track_team: dict[int, int] = {}
    for track_id, embeddings in track_embeddings.items():
        pooled = normalize_embedding(np.mean(embeddings, axis=0))
        distances = cosine_distances_to_prototypes(pooled, prototypes)
        track_team[track_id] = min(distances, key=distances.get)

    for idx in player_indices:
        track_id = metadata[idx].get("track_id")
        if track_id is None:
            assign_result_by_prototypes(results[idx], prototypes, source="prototype")
            continue
        team_id = int(track_team[int(track_id)])
        distances = cosine_distances_to_prototypes(results[idx].embedding, prototypes)
        results[idx].team_id = team_id
        results[idx].side_label = "left" if team_id == 0 else "right"
        results[idx].team_assignment_source = "track_prototype"
        results[idx].distance_to_left = distances.get(0)
        results[idx].distance_to_right = distances.get(1)


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
