
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ============================================================
# Models
# ============================================================

DEFAULT_SFACE_MODEL = (
    "models/face_recognition_sface_2021dec.onnx"
)

DEFAULT_YUNET_MODEL = (
    "models/face_detection_yunet_2023mar.onnx"
)


# ============================================================
# Face / embedding settings
# ============================================================

MAX_SAMPLES_PER_TRACK = 8
MIN_VALID_EMBEDDINGS = 2

# SFace pair thresholds.
PAIR_MEDIAN_THRESHOLD = 0.68
PAIR_P25_THRESHOLD = 0.50

PAIR_STRONG_MEDIAN = 0.75
PAIR_STRONG_P25 = 0.60

# Same-time faces are normally different people.
SIMULTANEOUS_OVERLAP_RATIO = 0.20
SIMULTANEOUS_STRONG_SIMILARITY = 0.92


# ============================================================
# Cluster settings
# ============================================================

CLUSTER_MEDIAN_THRESHOLD = 0.68
CLUSTER_MINIMUM_THRESHOLD = 0.58

MAX_PERSON_TRACKS = 12


# ============================================================
# Speaker identity settings
# ============================================================

# A speaker-face association above this is considered meaningful.
SPEAKER_EVIDENCE_THRESHOLD = 0.35

# If both faces have confident but different speaker identities,
# this becomes a hard identity conflict.
SPEAKER_HARD_CONFLICT_THRESHOLD = 0.35

# Require a small margin over the second-best speaker.
SPEAKER_MIN_MARGIN = 0.03


# ============================================================
# JSON
# ============================================================

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ============================================================
# Face tracks
# ============================================================

def load_face_tracks(
    data: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:

    tracks: Dict[str, Dict[str, Any]] = {}

    raw_tracks = data.get(
        "tracks",
        [],
    )

    if not isinstance(
        raw_tracks,
        list,
    ):
        raise RuntimeError(
            "Face JSON does not contain a 'tracks' list."
        )

    for track in raw_tracks:

        if not isinstance(
            track,
            dict,
        ):
            continue

        face_id = track.get(
            "face_id",
            track.get("id"),
        )

        if face_id is None:
            continue

        face_id = str(face_id)

        detections = []

        for det in track.get(
            "detections",
            [],
        ):

            if not isinstance(
                det,
                dict,
            ):
                continue

            t = as_float(
                det.get("time")
            )

            x = as_float(
                det.get("x")
            )

            y = as_float(
                det.get("y")
            )

            w = as_float(
                det.get("width")
            )

            h = as_float(
                det.get("height")
            )

            if None in (
                t,
                x,
                y,
                w,
                h,
            ):
                continue

            if w <= 1 or h <= 1:
                continue

            detections.append(
                {
                    **det,
                    "time": float(t),
                    "x": float(x),
                    "y": float(y),
                    "width": float(w),
                    "height": float(h),
                }
            )

        if not detections:
            continue

        detections.sort(
            key=lambda d: d["time"]
        )

        tracks[face_id] = {
            "face_id": face_id,
            "detections": detections,
            "start": detections[0]["time"],
            "end": detections[-1]["time"],
        }

    return tracks


def representative_detections(
    track: Dict[str, Any],
) -> List[Dict[str, Any]]:

    detections = track["detections"]

    if len(detections) <= MAX_SAMPLES_PER_TRACK:
        return list(detections)

    indices = np.linspace(
        0,
        len(detections) - 1,
        MAX_SAMPLES_PER_TRACK,
    )

    return [
        detections[int(round(i))]
        for i in indices
    ]


# ============================================================
# YuNet
# ============================================================

def create_yunet(
    model_path: str,
) -> Any:

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"YuNet model not found: {model_path}"
        )

    return cv2.FaceDetectorYN.create(
        model_path,
        "",
        (320, 320),
        0.60,
        0.30,
        5000,
    )


def detect_faces(
    detector: Any,
    frame: np.ndarray,
) -> np.ndarray:

    h, w = frame.shape[:2]

    detector.setInputSize(
        (w, h)
    )

    _, detections = detector.detect(
        frame
    )

    if detections is None:
        return np.empty(
            (0, 15),
            dtype=np.float32,
        )

    return np.asarray(
        detections,
        dtype=np.float32,
    )


def bbox_iou(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ax1 = ax + aw
    ay1 = ay + ah

    bx1 = bx + bw
    by1 = by + bh

    ix0 = max(
        ax,
        bx,
    )

    iy0 = max(
        ay,
        by,
    )

    ix1 = min(
        ax1,
        bx1,
    )

    iy1 = min(
        ay1,
        by1,
    )

    iw = max(
        0.0,
        ix1 - ix0,
    )

    ih = max(
        0.0,
        iy1 - iy0,
    )

    intersection = iw * ih

    area_a = max(
        0.0,
        aw,
    ) * max(
        0.0,
        ah,
    )

    area_b = max(
        0.0,
        bw,
    ) * max(
        0.0,
        bh,
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 1e-9:
        return 0.0

    return float(
        intersection / union
    )


def choose_matching_detection(
    target: Dict[str, Any],
    detections: np.ndarray,
) -> Optional[np.ndarray]:

    if detections.size == 0:
        return None

    target_box = np.array(
        [
            target["x"],
            target["y"],
            target["width"],
            target["height"],
        ],
        dtype=np.float32,
    )

    tcx = (
        target_box[0]
        + target_box[2] / 2.0
    )

    tcy = (
        target_box[1]
        + target_box[3] / 2.0
    )

    target_size = max(
        target_box[2],
        target_box[3],
        1.0,
    )

    best = None
    best_score = -1.0

    for detection in detections:

        candidate = detection[:4]

        iou = bbox_iou(
            target_box,
            candidate,
        )

        ccx = (
            candidate[0]
            + candidate[2] / 2.0
        )

        ccy = (
            candidate[1]
            + candidate[3] / 2.0
        )

        distance = np.sqrt(
            (ccx - tcx) ** 2
            + (ccy - tcy) ** 2
        )

        center_score = max(
            0.0,
            1.0
            - distance
            / target_size,
        )

        target_area = max(
            1.0,
            target_box[2]
            * target_box[3],
        )

        candidate_area = max(
            1.0,
            candidate[2]
            * candidate[3],
        )

        scale_score = (
            min(
                target_area,
                candidate_area,
            )
            / max(
                target_area,
                candidate_area,
            )
        )

        score = (
            0.70 * iou
            + 0.20 * scale_score
            + 0.10 * center_score
        )

        if score > best_score:

            best_score = score
            best = detection

    return best


# ============================================================
# SFace
# ============================================================

def create_sface(
    model_path: str,
) -> Any:

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"SFace model not found: {model_path}"
        )

    return cv2.FaceRecognizerSF.create(
        model_path,
        "",
    )


def extract_embedding(
    recognizer: Any,
    frame: np.ndarray,
    detection: np.ndarray,
) -> Optional[np.ndarray]:

    try:

        aligned = recognizer.alignCrop(
            frame,
            detection,
        )

        if (
            aligned is None
            or aligned.size == 0
        ):
            return None

        feature = recognizer.feature(
            aligned
        )

        feature = np.asarray(
            feature,
            dtype=np.float32,
        ).flatten()

        norm = np.linalg.norm(
            feature
        )

        if norm <= 1e-9:
            return None

        return (
            feature / norm
        ).astype(
            np.float32
        )

    except Exception:

        return None


# ============================================================
# Embeddings per face track
# ============================================================

def extract_track_embeddings(
    video_path: str,
    tracks: Dict[str, Dict[str, Any]],
    sface_model: str,
    yunet_model: str,
) -> Dict[str, Dict[str, Any]]:

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = (
        cap.get(
            cv2.CAP_PROP_FPS
        )
        or 30.0
    )

    recognizer = create_sface(
        sface_model
    )

    detector = create_yunet(
        yunet_model
    )

    output: Dict[
        str,
        Dict[str, Any],
    ] = {}

    for face_id, track in tracks.items():

        embeddings = []
        sample_times = []

        for detection in representative_detections(
            track
        ):

            t = float(
                detection["time"]
            )

            frame_number = max(
                0,
                int(
                    round(
                        t * fps
                    )
                ),
            )

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_number,
            )

            ok, frame = cap.read()

            if not ok or frame is None:
                continue

            fresh = detect_faces(
                detector,
                frame,
            )

            matching = (
                choose_matching_detection(
                    detection,
                    fresh,
                )
            )

            if matching is None:
                continue

            embedding = extract_embedding(
                recognizer,
                frame,
                matching,
            )

            if embedding is None:
                continue

            embeddings.append(
                embedding
            )

            sample_times.append(
                t
            )

        output[face_id] = {
            "embeddings": embeddings,
            "sample_times": sample_times,
            "samples": len(
                embeddings
            ),
        }

    cap.release()

    return output


# ============================================================
# Similarity
# ============================================================

def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator <= 1e-9:
        return 0.0

    return float(
        np.clip(
            np.dot(a, b)
            / denominator,
            -1.0,
            1.0,
        )
    )


def pair_similarity_stats(
    a: Dict[str, Any],
    b: Dict[str, Any],
) -> Dict[str, float]:

    ea = a.get(
        "embeddings",
        [],
    )

    eb = b.get(
        "embeddings",
        [],
    )

    if not ea or not eb:

        return {
            "median": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "mean": 0.0,
            "best": 0.0,
        }

    # Best counterpart per sample, in both directions.
    directed_a = []

    for x in ea:

        directed_a.append(
            max(
                cosine_similarity(
                    x,
                    y,
                )
                for y in eb
            )
        )

    directed_b = []

    for y in eb:

        directed_b.append(
            max(
                cosine_similarity(
                    y,
                    x,
                )
                for x in ea
            )
        )

    values = np.asarray(
        directed_a + directed_b,
        dtype=np.float32,
    )

    return {
        "median": float(
            np.percentile(
                values,
                50,
            )
        ),
        "p25": float(
            np.percentile(
                values,
                25,
            )
        ),
        "p75": float(
            np.percentile(
                values,
                75,
            )
        ),
        "mean": float(
            np.mean(values)
        ),
        "best": float(
            np.max(values)
        ),
    }


# ============================================================
# Temporal
# ============================================================

def overlap_seconds(
    a: Dict[str, Any],
    b: Dict[str, Any],
) -> float:

    return max(
        0.0,
        min(
            float(a["end"]),
            float(b["end"]),
        )
        -
        max(
            float(a["start"]),
            float(b["start"]),
        ),
    )


def overlap_ratio(
    a: Dict[str, Any],
    b: Dict[str, Any],
) -> float:

    overlap = overlap_seconds(
        a,
        b,
    )

    if overlap <= 0:
        return 0.0

    da = (
        float(a["end"])
        - float(a["start"])
    )

    db = (
        float(b["end"])
        - float(b["start"])
    )

    denominator = min(
        da,
        db,
    )

    if denominator <= 0:
        return 0.0

    return float(
        overlap / denominator
    )


# ============================================================
# Speaker evidence
# ============================================================

def load_face_speaker_profiles(
    path: str,
) -> Dict[
    str,
    Dict[str, Any],
]:

    data = load_json(
        path
    )

    profiles: Dict[
        str,
        Dict[str, Any],
    ] = {}

    # Prefer the explicit face_identity section generated by
    # speaker_face_association_v4.
    face_identity = data.get(
        "face_identity",
        {},
    )

    if isinstance(
        face_identity,
        dict,
    ):

        for face_id, item in face_identity.items():

            if not isinstance(
                item,
                dict,
            ):
                continue

            speaker = item.get(
                "speaker"
            )

            confidence = as_float(
                item.get(
                    "confidence",
                    0.0,
                )
            )

            margin = as_float(
                item.get(
                    "margin_over_second",
                    0.0,
                )
            )

            if speaker is None:
                continue

            profiles[
                str(face_id)
            ] = {
                "speaker": str(
                    speaker
                ),
                "confidence": float(
                    confidence
                    or 0.0
                ),
                "margin": float(
                    margin
                    or 0.0
                ),
            }

    # Fallback to hypotheses if needed.
    if not profiles:

        hypotheses = data.get(
            "hypotheses",
            [],
        )

        grouped: Dict[
            str,
            List[Dict[str, Any]],
        ] = defaultdict(list)

        for item in hypotheses:

            if not isinstance(
                item,
                dict,
            ):
                continue

            face_id = item.get(
                "face_id"
            )

            speaker = item.get(
                "speaker"
            )

            confidence = as_float(
                item.get(
                    "confidence",
                    0.0,
                )
            )

            if (
                face_id is None
                or speaker is None
            ):
                continue

            grouped[
                str(face_id)
            ].append(
                {
                    "speaker": str(
                        speaker
                    ),
                    "confidence": float(
                        confidence
                        or 0.0
                    ),
                }
            )

        for face_id, values in grouped.items():

            values.sort(
                key=lambda x:
                    x["confidence"],
                reverse=True,
            )

            if not values:
                continue

            best = values[0]

            second = (
                values[1]["confidence"]
                if len(values) > 1
                else 0.0
            )

            profiles[
                face_id
            ] = {
                "speaker": best[
                    "speaker"
                ],
                "confidence": best[
                    "confidence"
                ],
                "margin": (
                    best["confidence"]
                    - second
                ),
            }

    return profiles


def reliable_speaker_profile(
    face_id: str,
    profiles: Dict[
        str,
        Dict[str, Any],
    ],
) -> Optional[
    Dict[str, Any]
]:

    profile = profiles.get(
        face_id
    )

    if profile is None:
        return None

    if (
        profile["confidence"]
        < SPEAKER_EVIDENCE_THRESHOLD
    ):
        return None

    if (
        profile["margin"]
        < SPEAKER_MIN_MARGIN
    ):
        return None

    return profile


def speaker_conflict(
    face_a: str,
    face_b: str,
    profiles: Dict[
        str,
        Dict[str, Any],
    ],
) -> Tuple[
    bool,
    Optional[str],
    Optional[str],
]:

    profile_a = reliable_speaker_profile(
        face_a,
        profiles,
    )

    profile_b = reliable_speaker_profile(
        face_b,
        profiles,
    )

    if (
        profile_a is None
        or profile_b is None
    ):
        return (
            False,
            None,
            None,
        )

    speaker_a = profile_a[
        "speaker"
    ]

    speaker_b = profile_b[
        "speaker"
    ]

    if speaker_a == speaker_b:

        return (
            False,
            speaker_a,
            speaker_b,
        )

    return (
        True,
        speaker_a,
        speaker_b,
    )


# ============================================================
# Joint pair decision
# ============================================================

def evaluate_pair(
    face_a: str,
    face_b: str,
    tracks: Dict[str, Dict[str, Any]],
    descriptors: Dict[str, Dict[str, Any]],
    pair_stats: Dict[
        Tuple[str, str],
        Dict[str, float],
    ],
    speaker_profiles: Dict[
        str,
        Dict[str, Any],
    ],
) -> Dict[str, Any]:

    key = (
        (face_a, face_b)
        if face_a < face_b
        else (face_b, face_a)
    )

    stats = pair_stats[key]

    ratio = overlap_ratio(
        tracks[face_a],
        tracks[face_b],
    )

    # -----------------------------------------------
    # Speaker constraint FIRST.
    # -----------------------------------------------

    conflict, speaker_a, speaker_b = (
        speaker_conflict(
            face_a,
            face_b,
            speaker_profiles,
        )
    )

    # -----------------------------------------------
    # Simultaneous faces.
    # -----------------------------------------------

    simultaneous = (
        ratio
        >= SIMULTANEOUS_OVERLAP_RATIO
    )

    if simultaneous:

        compatible = (
            not conflict
            and stats["median"]
            >= SIMULTANEOUS_STRONG_SIMILARITY
            and stats["p25"]
            >= PAIR_STRONG_P25
        )

        reason = (
            "simultaneous_exceptionally_strong"
            if compatible
            else "simultaneous_faces"
        )

    else:

        # -------------------------------------------
        # Strong and consistent SFace.
        # -------------------------------------------

        strong_sface = (
            stats["median"]
            >= PAIR_STRONG_MEDIAN
            and stats["p25"]
            >= PAIR_STRONG_P25
        )

        moderate_sface = (
            stats["median"]
            >= PAIR_MEDIAN_THRESHOLD
            and stats["p25"]
            >= PAIR_P25_THRESHOLD
        )

        compatible = (
            not conflict
            and (
                strong_sface
                or moderate_sface
            )
        )

        if conflict:

            reason = (
                "hard_speaker_identity_conflict"
            )

        elif strong_sface:

            reason = (
                "strong_sface_no_speaker_conflict"
            )

        elif moderate_sface:

            reason = (
                "moderate_sface_no_speaker_conflict"
            )

        else:

            reason = (
                "weak_sface"
            )

    # -----------------------------------------------
    # Joint score.
    #
    # Speaker conflict is now represented explicitly,
    # rather than being a tiny adjustment.
    # -----------------------------------------------

    if conflict:

        joint_score = min(
            0.45,
            (
                0.82
                * stats["median"]
                + 0.18
                * stats["p25"]
            ),
        )

    else:

        joint_score = (
            0.82
            * stats["median"]
            + 0.18
            * stats["p25"]
        )

    return {
        "face_a": face_a,
        "face_b": face_b,
        "sface_median": round(
            stats["median"],
            4,
        ),
        "sface_p25": round(
            stats["p25"],
            4,
        ),
        "sface_p75": round(
            stats["p75"],
            4,
        ),
        "sface_mean": round(
            stats["mean"],
            4,
        ),
        "sface_best": round(
            stats["best"],
            4,
        ),
        "temporal_overlap_ratio": round(
            ratio,
            4,
        ),
        "speaker_a": speaker_a,
        "speaker_b": speaker_b,
        "speaker_conflict": conflict,
        "joint_score": round(
            float(
                np.clip(
                    joint_score,
                    0.0,
                    1.0,
                )
            ),
            4,
        ),
        "compatible": compatible,
        "reason": reason,
    }


# ============================================================
# Cluster consistency
# ============================================================

def cluster_compatible(
    cluster: List[str],
    candidate: str,
    pair_map: Dict[
        Tuple[str, str],
        Dict[str, Any],
    ],
) -> Tuple[
    bool,
    float,
    float,
]:

    if not cluster:

        return (
            True,
            1.0,
            1.0,
        )

    scores = []

    for existing in cluster:

        key = (
            (existing, candidate)
            if existing < candidate
            else (candidate, existing)
        )

        result = pair_map.get(
            key
        )

        if result is None:
            return (
                False,
                0.0,
                0.0,
            )

        if not result["compatible"]:

            return (
                False,
                0.0,
                float(
                    result[
                        "joint_score"
                    ]
                ),
            )

        scores.append(
            float(
                result[
                    "joint_score"
                ]
            )
        )

    array = np.asarray(
        scores,
        dtype=np.float32,
    )

    median_score = float(
        np.median(array)
    )

    minimum_score = float(
        np.min(array)
    )

    ok = (
        median_score
        >= CLUSTER_MEDIAN_THRESHOLD
        and minimum_score
        >= CLUSTER_MINIMUM_THRESHOLD
    )

    return (
        ok,
        median_score,
        minimum_score,
    )


# ============================================================
# Clustering
# ============================================================

def build_clusters(
    tracks: Dict[str, Dict[str, Any]],
    descriptors: Dict[str, Dict[str, Any]],
    speaker_profiles: Dict[
        str,
        Dict[str, Any],
    ],
) -> Tuple[
    List[List[str]],
    List[Dict[str, Any]],
]:

    face_ids = sorted(
        tracks.keys(),
        key=lambda face_id:
            tracks[face_id]["start"],
    )

    # --------------------------------------------------------
    # Raw SFace statistics.
    # --------------------------------------------------------

    raw_pair_stats: Dict[
        Tuple[str, str],
        Dict[str, float],
    ] = {}

    for i in range(
        len(face_ids)
    ):

        a = face_ids[i]

        for j in range(
            i + 1,
            len(face_ids)
        ):

            b = face_ids[j]

            key = (
                (a, b)
                if a < b
                else (b, a)
            )

            raw_pair_stats[key] = (
                pair_similarity_stats(
                    descriptors[a],
                    descriptors[b],
                )
            )

    # --------------------------------------------------------
    # Joint pair decisions.
    # --------------------------------------------------------

    pair_map: Dict[
        Tuple[str, str],
        Dict[str, Any],
    ] = {}

    for key in raw_pair_stats:

        a, b = key

        pair_map[key] = evaluate_pair(
            a,
            b,
            tracks,
            descriptors,
            raw_pair_stats,
            speaker_profiles,
        )

    candidates = [
        result
        for result in pair_map.values()
        if result["compatible"]
    ]

    candidates.sort(
        key=lambda x: (
            x["joint_score"],
            x["sface_median"],
            x["sface_p25"],
        ),
        reverse=True,
    )

    # --------------------------------------------------------
    # One-to-one assignment.
    # --------------------------------------------------------

    assigned = set()

    clusters: List[
        List[str]
    ] = []

    for pair in candidates:

        a = pair["face_a"]
        b = pair["face_b"]

        a_assigned = (
            a in assigned
        )

        b_assigned = (
            b in assigned
        )

        # Both new -> seed identity.
        if (
            not a_assigned
            and not b_assigned
        ):

            clusters.append(
                [
                    a,
                    b,
                ]
            )

            assigned.add(a)
            assigned.add(b)

            continue

        # A assigned, B new.
        if (
            a_assigned
            and not b_assigned
        ):

            cluster_index = next(
                (
                    index
                    for index, cluster
                    in enumerate(
                        clusters
                    )
                    if a in cluster
                ),
                None,
            )

            if cluster_index is None:
                continue

            cluster = clusters[
                cluster_index
            ]

            if (
                len(cluster)
                >= MAX_PERSON_TRACKS
            ):
                continue

            ok, _, _ = cluster_compatible(
                cluster,
                b,
                pair_map,
            )

            if ok:

                cluster.append(
                    b
                )

                assigned.add(
                    b
                )

            continue

        # B assigned, A new.
        if (
            b_assigned
            and not a_assigned
        ):

            cluster_index = next(
                (
                    index
                    for index, cluster
                    in enumerate(
                        clusters
                    )
                    if b in cluster
                ),
                None,
            )

            if cluster_index is None:
                continue

            cluster = clusters[
                cluster_index
            ]

            if (
                len(cluster)
                >= MAX_PERSON_TRACKS
            ):
                continue

            ok, _, _ = cluster_compatible(
                cluster,
                a,
                pair_map,
            )

            if ok:

                cluster.append(
                    a
                )

                assigned.add(
                    a
                )

            continue

        # Both already assigned:
        # never merge two existing identities.
        continue

    # Remaining faces become separate identities.
    for face_id in face_ids:

        if face_id not in assigned:

            clusters.append(
                [
                    face_id
                ]
            )

            assigned.add(
                face_id
            )

    # Stable ordering by first appearance.
    clusters.sort(
        key=lambda cluster:
            min(
                tracks[
                    face_id
                ]["start"]
                for face_id in cluster
            )
    )

    return (
        clusters,
        list(pair_map.values()),
    )


# ============================================================
# Output
# ============================================================

def build_person_output(
    clusters: List[List[str]],
    tracks: Dict[str, Dict[str, Any]],
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, str],
]:

    persons = []

    face_to_person: Dict[
        str,
        str,
    ] = {}

    for index, cluster in enumerate(
        clusters
    ):

        person_id = (
            f"PERSON_{index:02d}"
        )

        ordered = sorted(
            set(cluster),
            key=lambda face_id:
                tracks[
                    face_id
                ]["start"],
        )

        for face_id in ordered:

            if face_id in face_to_person:

                raise RuntimeError(
                    "Face assigned twice: "
                    f"{face_id}"
                )

            face_to_person[
                face_id
            ] = person_id

        persons.append(
            {
                "person_id": person_id,
                "face_ids": ordered,
                "track_count": len(
                    ordered
                ),
                "start": round(
                    min(
                        tracks[
                            face_id
                        ]["start"]
                        for face_id
                        in ordered
                    ),
                    3,
                ),
                "end": round(
                    max(
                        tracks[
                            face_id
                        ]["end"]
                        for face_id
                        in ordered
                    ),
                    3,
                ),
            }
        )

    return (
        persons,
        face_to_person,
    )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Joint persistent person identity "
            "using SFace plus speaker constraints."
        )
    )

    parser.add_argument(
        "--video",
        required=True,
    )

    parser.add_argument(
        "--faces",
        required=True,
    )

    parser.add_argument(
        "--speaking",
        required=True,
    )

    parser.add_argument(
        "--speaker-association",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--sface-model",
        default=DEFAULT_SFACE_MODEL,
    )

    parser.add_argument(
        "--yunet-model",
        default=DEFAULT_YUNET_MODEL,
    )

    return parser.parse_args()


def main() -> int:

    args = parse_args()

    print(
        "Loading face tracks..."
    )

    face_data = load_json(
        args.faces
    )

    tracks = load_face_tracks(
        face_data
    )

    if not tracks:

        raise RuntimeError(
            "No face tracks found."
        )

    print(
        f"Face tracks: {len(tracks)}"
    )

    print()
    print(
        "Loading speaker-face evidence..."
    )

    speaker_profiles = (
        load_face_speaker_profiles(
            args.speaker_association
        )
    )

    print(
        f"Faces with speaker evidence: "
        f"{len(speaker_profiles)}"
    )

    reliable_profiles = {
        face_id: profile
        for face_id, profile
        in speaker_profiles.items()
        if reliable_speaker_profile(
            face_id,
            speaker_profiles,
        )
        is not None
    }

    print(
        f"Faces with reliable speaker identity: "
        f"{len(reliable_profiles)}"
    )

    print()
    print(
        "Extracting aligned SFace embeddings..."
    )

    descriptors = (
        extract_track_embeddings(
            args.video,
            tracks,
            args.sface_model,
            args.yunet_model,
        )
    )

    valid_embeddings = sum(
        1
        for item in descriptors.values()
        if item["samples"]
        >= MIN_VALID_EMBEDDINGS
    )

    print(
        f"Tracks with valid embeddings: "
        f"{valid_embeddings}/{len(tracks)}"
    )

    print()
    print(
        "Building joint identity graph..."
    )

    (
        clusters,
        pair_results,
    ) = build_clusters(
        tracks,
        descriptors,
        speaker_profiles,
    )

    persons, face_to_person = (
        build_person_output(
            clusters,
            tracks,
        )
    )

    # --------------------------------------------------------
    # Sanity checks.
    # --------------------------------------------------------

    all_faces = [
        face_id
        for person in persons
        for face_id in person["face_ids"]
    ]

    if len(all_faces) != len(
        set(all_faces)
    ):
        raise RuntimeError(
            "Duplicate face assignment detected."
        )

    if set(all_faces) != set(
        tracks.keys()
    ):
        raise RuntimeError(
            "Some face tracks were not assigned."
        )

    conflict_pairs = sum(
        1
        for pair in pair_results
        if pair["speaker_conflict"]
    )

    compatible_pairs = sum(
        1
        for pair in pair_results
        if pair["compatible"]
    )

    print()
    print(
        f"Accepted joint identity pairs: "
        f"{compatible_pairs}/"
        f"{len(pair_results)}"
    )

    print(
        f"Speaker-conflict pairs rejected: "
        f"{conflict_pairs}"
    )

    print()
    print(
        f"Persistent persons: "
        f"{len(persons)}"
    )

    print()

    for person in persons:

        print(
            f"  {person['person_id']}: "
            f"{', '.join(person['face_ids'])} "
            f"| "
            f"{person['start']:.1f}s -> "
            f"{person['end']:.1f}s "
            f"| tracks="
            f"{person['track_count']}"
        )

    print()
    print(
        "STRONGEST JOINT PAIRS"
    )

    strongest = sorted(
        pair_results,
        key=lambda x: (
            x["joint_score"],
            x["sface_median"],
            x["sface_p25"],
        ),
        reverse=True,
    )

    for pair in strongest[:30]:

        print(
            f"  {pair['face_a']} <-> "
            f"{pair['face_b']} "
            f"joint="
            f"{pair['joint_score']:.3f} "
            f"sface="
            f"{pair['sface_median']:.3f} "
            f"p25="
            f"{pair['sface_p25']:.3f} "
            f"speaker="
            f"{pair['speaker_a']}/"
            f"{pair['speaker_b']} "
            f"conflict="
            f"{pair['speaker_conflict']} "
            f"compatible="
            f"{pair['compatible']}"
        )

    output = {
        "version": 7,
        "mode": "joint_person_identity",
        "video": os.path.abspath(
            args.video
        ),
        "face_file": os.path.abspath(
            args.faces
        ),
        "speaking_file": os.path.abspath(
            args.speaking
        ),
        "speaker_association_file": os.path.abspath(
            args.speaker_association
        ),
        "sface_model": os.path.abspath(
            args.sface_model
        ),
        "yunet_model": os.path.abspath(
            args.yunet_model
        ),
        "appearance_model": "sface",
        "face_tracks": len(tracks),
        "valid_embedding_tracks": valid_embeddings,
        "faces_with_speaker_evidence": len(
            speaker_profiles
        ),
        "faces_with_reliable_speaker_identity": len(
            reliable_profiles
        ),
        "accepted_joint_pairs": compatible_pairs,
        "speaker_conflict_pairs_rejected": conflict_pairs,
        "persons": persons,
        "face_to_person": face_to_person,
        "pair_results": pair_results,
        "thresholds": {
            "pair_median":
                PAIR_MEDIAN_THRESHOLD,
            "pair_p25":
                PAIR_P25_THRESHOLD,
            "pair_strong_median":
                PAIR_STRONG_MEDIAN,
            "pair_strong_p25":
                PAIR_STRONG_P25,
            "simultaneous_overlap":
                SIMULTANEOUS_OVERLAP_RATIO,
            "simultaneous_strong":
                SIMULTANEOUS_STRONG_SIMILARITY,
            "cluster_median":
                CLUSTER_MEDIAN_THRESHOLD,
            "cluster_minimum":
                CLUSTER_MINIMUM_THRESHOLD,
            "speaker_evidence":
                SPEAKER_EVIDENCE_THRESHOLD,
            "speaker_conflict":
                SPEAKER_HARD_CONFLICT_THRESHOLD,
            "speaker_margin":
                SPEAKER_MIN_MARGIN,
        },
    }

    output_path = os.path.abspath(
        args.output
    )

    os.makedirs(
        os.path.dirname(
            output_path
        ) or ".",
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    print()
    print(
        f"Saved: {output_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
