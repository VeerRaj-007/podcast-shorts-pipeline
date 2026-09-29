
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# CONFIG
# ============================================================

# Face must actually be visible at this timestamp.
VISIBILITY_TOLERANCE = 0.40

# Candidate from speaking-face timeline is strong local evidence,
# but not absolute identity proof.
SPEAKING_WEIGHT = 0.45

# Aggregate speaker-face association prior.
ASSOCIATION_WEIGHT = 0.20

# Aggregate face identity prior.
IDENTITY_WEIGHT = 0.15

# Temporal continuity.
CONTINUITY_WEIGHT = 0.12

# Small contribution from the speaking-face module's
# active_face decision.
ACTIVE_FACE_WEIGHT = 0.08

# Minimum local score for a usable result.
USABLE_THRESHOLD = 0.45

# Minimum local score for aggressive reframing.
STRONG_THRESHOLD = 0.68

# Minimum lead over second-best candidate.
MIN_MARGIN = 0.06

# How long a previous face remains useful.
CONTINUITY_WINDOW = 2.5

SPEAKING_WEIGHT = 0.45
ASSOCIATION_WEIGHT = 0.20
IDENTITY_WEIGHT = 0.15
CONTINUITY_WEIGHT = 0.12
VISIBILITY_WEIGHT = 0.08
ACTIVE_FACE_WEIGHT = 0.08


# ============================================================
# JSON
# ============================================================

def load_json(
    path: str,
) -> Dict[str, Any]:

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def to_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:
        return float(value)
    except (
        TypeError,
        ValueError,
    ):
        return default


def clamp(
    value: float,
    low: float = 0.0,
    high: float = 1.0,
) -> float:

    return max(
        low,
        min(high, value),
    )


# ============================================================
# FACE TRACKS
# ============================================================

def load_face_tracks(
    path: str,
) -> Dict[str, Dict[str, Any]]:

    data = load_json(path)

    tracks: Dict[str, Dict[str, Any]] = {}

    for raw_track in data.get(
        "tracks",
        [],
    ):

        if not isinstance(
            raw_track,
            dict,
        ):
            continue

        face_id = (
            raw_track.get("face_id")
            or raw_track.get("id")
        )

        if face_id is None:
            continue

        face_id = str(
            face_id
        )

        raw_detections = (
            raw_track.get(
                "detections"
            )
            or raw_track.get(
                "samples"
            )
            or []
        )

        detections = []

        for detection in raw_detections:

            if not isinstance(
                detection,
                dict,
            ):
                continue

            time_value = (
                detection.get("time")
                if "time" in detection
                else detection.get(
                    "timestamp"
                )
            )

            try:
                time_value = float(
                    time_value
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            detections.append(
                {
                    **detection,
                    "time": time_value,
                }
            )

        if not detections:
            continue

        detections.sort(
            key=lambda x:
                x["time"]
        )

        tracks[face_id] = {
            "face_id": face_id,
            "start":
                detections[0]["time"],
            "end":
                detections[-1]["time"],
            "detections":
                detections,
        }

    return tracks


def nearest_detection(
    track: Dict[str, Any],
    time_value: float,
) -> Optional[Dict[str, Any]]:

    best = None
    best_distance = float("inf")

    for detection in track[
        "detections"
    ]:

        distance = abs(
            detection["time"]
            - time_value
        )

        if distance < best_distance:

            best_distance = distance
            best = detection

    if (
        best is None
        or best_distance
        > VISIBILITY_TOLERANCE
    ):
        return None

    result = dict(best)

    result[
        "_distance"
    ] = best_distance

    return result


def get_visible_faces(
    tracks: Dict[str, Dict[str, Any]],
    time_value: float,
) -> Dict[str, Dict[str, Any]]:

    visible = {}

    for face_id, track in tracks.items():

        detection = nearest_detection(
            track,
            time_value,
        )

        if detection is not None:

            visible[
                face_id
            ] = detection

    return visible


# ============================================================
# ASSOCIATION
# ============================================================

def load_association(
    path: str,
) -> Dict[str, Any]:

    return load_json(path)


def get_speaker_identity(
    association: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:

    result = {}

    for speaker, info in association.get(
        "speaker_identity",
        {},
    ).items():

        if isinstance(
            info,
            dict,
        ):
            result[
                str(speaker)
            ] = info

    return result


def get_face_identity(
    association: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:

    result = {}

    for face_id, info in association.get(
        "face_identity",
        {},
    ).items():

        if isinstance(
            info,
            dict,
        ):
            result[
                str(face_id)
            ] = info

    return result


# ============================================================
# AGGREGATE HYPOTHESES
# ============================================================

def load_hypothesis_priors(
    association: Dict[str, Any],
) -> Dict[str, Dict[str, Dict[str, float]]]:

    """
    Returns:

        priors[speaker][face_id] = {
            "confidence": ...,
            "active_face_rate": ...,
            "motion_score": ...,
            "temporal_visibility": ...,
            "exclusivity": ...,
            ...
        }
    """

    result = defaultdict(dict)

    for item in association.get(
        "hypotheses",
        [],
    ):

        if not isinstance(
            item,
            dict,
        ):
            continue

        speaker = item.get(
            "speaker"
        )

        face_id = item.get(
            "face_id"
        )

        if (
            speaker is None
            or face_id is None
        ):
            continue

        result[
            str(speaker)
        ][
            str(face_id)
        ] = {
            "confidence":
                clamp(
                    to_float(
                        item.get(
                            "confidence"
                        )
                    )
                ),
            "active_face_rate":
                clamp(
                    to_float(
                        item.get(
                            "active_face_rate"
                        )
                    )
                ),
            "motion_score":
                clamp(
                    to_float(
                        item.get(
                            "motion_score"
                        )
                    )
                ),
            "temporal_visibility":
                clamp(
                    to_float(
                        item.get(
                            "temporal_visibility"
                        )
                    )
                ),
            "exclusivity":
                clamp(
                    to_float(
                        item.get(
                            "exclusivity"
                        )
                    )
                ),
            "active_face_samples":
                int(
                    to_float(
                        item.get(
                            "active_face_samples"
                        )
                    )
                ),
            "speaker_samples":
                int(
                    to_float(
                        item.get(
                            "speaker_samples"
                        )
                    )
                ),
            "visible_speaker_samples":
                int(
                    to_float(
                        item.get(
                            "visible_speaker_samples"
                        )
                    )
                ),
        }

    return result


# ============================================================
# SPEAKING TIMELINE
# ============================================================

def load_speaking_timeline(
    path: str,
) -> List[Dict[str, Any]]:

    data = load_json(
        path
    )

    result = []

    for item in data.get(
        "timeline",
        [],
    ):

        if not isinstance(
            item,
            dict,
        ):
            continue

        try:
            time_value = float(
                item.get(
                    "time"
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        result.append(
            {
                **item,
                "time":
                    time_value,
            }
        )

    result.sort(
        key=lambda x:
            x["time"]
    )

    return result


# ============================================================
# LOCAL CANDIDATES
# ============================================================

def parse_timeline_candidates(
    item: Dict[str, Any],
) -> List[Dict[str, Any]]:

    """
    Extract the actual timestamp-local candidates from:

        timeline[].candidates[]

    Your current speaking-face JSON contains:

        face_id
        score
        mouth_motion
        speech_active
        speaker
    """

    candidates = []

    raw_candidates = item.get(
        "candidates",
        [],
    )

    if not isinstance(
        raw_candidates,
        list,
    ):
        return candidates

    for raw in raw_candidates:

        if not isinstance(
            raw,
            dict,
        ):
            continue

        face_id = raw.get(
            "face_id"
        )

        if face_id is None:
            continue

        candidates.append(
            {
                "face_id":
                    str(face_id),

                # This is local speaking/mouth evidence.
                "speaking_score":
                    clamp(
                        to_float(
                            raw.get(
                                "score"
                            )
                        )
                    ),

                "mouth_motion":
                    clamp(
                        to_float(
                            raw.get(
                                "mouth_motion"
                            )
                        )
                    ),

                "speech_active":
                    bool(
                        raw.get(
                            "speech_active",
                            False,
                        )
                    ),

                "speaker":
                    (
                        str(
                            raw.get(
                                "speaker"
                            )
                        )
                        if raw.get(
                            "speaker"
                        )
                        is not None
                        else None
                    ),
            }
        )

    return candidates


# ============================================================
# AUXILIARY SCORES
# ============================================================

def identity_prior(
    speaker: str,
    face_id: str,
    face_identity: Dict[
        str,
        Dict[str, Any],
    ],
) -> float:

    info = face_identity.get(
        face_id
    )

    if not info:
        return 0.0

    identity_speaker = info.get(
        "speaker"
    )

    confidence = clamp(
        to_float(
            info.get(
                "confidence"
            )
        )
    )

    if identity_speaker is None:
        return 0.0

    if str(
        identity_speaker
    ) == speaker:

        return confidence

    # Strongly penalize known conflicting speaker assignment.
    return confidence * 0.02


def association_prior(
    speaker: str,
    face_id: str,
    hypothesis_priors: Dict[
        str,
        Dict[str, Dict[str, float]],
    ],
) -> float:

    item = hypothesis_priors.get(
        speaker,
        {},
    ).get(
        face_id
    )

    if item is None:
        return 0.0

    confidence = item[
        "confidence"
    ]

    temporal_visibility = item[
        "temporal_visibility"
    ]

    exclusivity = item[
        "exclusivity"
    ]

    # Confidence is most important.
    score = (
        0.60 * confidence
        + 0.25 * temporal_visibility
        + 0.15 * exclusivity
    )

    return clamp(
        score
    )


def continuity_score(
    face_id: str,
    previous_face: Optional[str],
    current_time: float,
    previous_time: Optional[float],
) -> float:

    if (
        previous_face is None
        or previous_time is None
    ):
        return 0.0

    if face_id != previous_face:
        return 0.0

    delta = abs(
        current_time
        - previous_time
    )

    if delta > CONTINUITY_WINDOW:
        return 0.0

    return clamp(
        1.0
        - (
            delta
            / CONTINUITY_WINDOW
        )
    )


def visibility_score(
    detection: Optional[Dict[str, Any]],
) -> float:

    if detection is None:
        return 0.0

    distance = to_float(
        detection.get(
            "_distance"
        ),
        VISIBILITY_TOLERANCE,
    )

    return clamp(
        1.0
        - (
            distance
            / VISIBILITY_TOLERANCE
        )
    )


def active_face_score(
    face_id: str,
    active_face: Optional[str],
) -> float:

    if (
        active_face is None
        or face_id != active_face
    ):
        return 0.0

    return 1.0


# ============================================================
# MAIN LOCAL RESOLVER
# ============================================================

def resolve_timeline(
    tracks: Dict[str, Dict[str, Any]],
    timeline: List[Dict[str, Any]],
    speaker_identity: Dict[
        str,
        Dict[str, Any],
    ],
    face_identity: Dict[
        str,
        Dict[str, Any],
    ],
    hypothesis_priors: Dict[
        str,
        Dict[str, Dict[str, float]],
    ],
) -> List[Dict[str, Any]]:

    result = []

    previous_by_speaker: Dict[
        str,
        Tuple[str, float],
    ] = {}

    for item in timeline:

        time_value = item[
            "time"
        ]

        speaker = item.get(
            "active_speaker"
        )

        active_face = item.get(
            "active_face"
        )

        if speaker is not None:
            speaker = str(
                speaker
            )

        if active_face is not None:
            active_face = str(
                active_face
            )

        # ----------------------------------------------------
        # No active speaker.
        # ----------------------------------------------------

        if speaker is None:

            result.append(
                {
                    "time":
                        time_value,
                    "speaker":
                        None,
                    "face_id":
                        None,
                    "confidence":
                        0.0,
                    "strength":
                        "none",
                    "active_face":
                        active_face,
                    "candidate_count":
                        0,
                    "candidates":
                        [],
                }
            )

            continue

        # ----------------------------------------------------
        # Faces physically visible at this exact time.
        # ----------------------------------------------------

        visible = get_visible_faces(
            tracks,
            time_value,
        )

        # ----------------------------------------------------
        # Timestamp-local speaking candidates.
        # ----------------------------------------------------

        local_candidates = (
            parse_timeline_candidates(
                item
            )
        )

        # Only use candidates belonging to current speaker.
        speaker_candidates = [
            candidate
            for candidate in local_candidates
            if (
                candidate["speaker"]
                is None
                or candidate["speaker"]
                == speaker
            )
        ]

        # ----------------------------------------------------
        # CRITICAL:
        #
        # If the speaking-face timeline has candidates,
        # use those faces as the primary local candidate set.
        #
        # If no candidates exist, fall back to physically
        # visible faces.
        # ----------------------------------------------------

        candidate_face_ids = set()

        for candidate in speaker_candidates:

            face_id = candidate[
                "face_id"
            ]

            if face_id in visible:
                candidate_face_ids.add(
                    face_id
                )

        if not candidate_face_ids:

            candidate_face_ids = set(
                visible.keys()
            )

        previous = (
            previous_by_speaker.get(
                speaker
            )
        )

        previous_face = (
            previous[0]
            if previous
            else None
        )

        previous_time = (
            previous[1]
            if previous
            else None
        )

        scored = []

        for face_id in candidate_face_ids:

            detection = visible.get(
                face_id
            )

            # ----------------------------------------------
            # Local speaking evidence.
            # ----------------------------------------------

            speaking = 0.0
            mouth_motion = 0.0
            speech_active = False

            for candidate in speaker_candidates:

                if (
                    candidate[
                        "face_id"
                    ]
                    != face_id
                ):
                    continue

                speaking = max(
                    speaking,
                    candidate[
                        "speaking_score"
                    ],
                )

                mouth_motion = max(
                    mouth_motion,
                    candidate[
                        "mouth_motion"
                    ],
                )

                speech_active = (
                    speech_active
                    or candidate[
                        "speech_active"
                    ]
                )

            # If timeline has no candidate entry,
            # no local speaking evidence.
            if face_id not in {
                c["face_id"]
                for c in speaker_candidates
            }:
                speaking = 0.0

            # ------------------------------------------------
            # Aggregate identity evidence.
            # ------------------------------------------------

            identity = identity_prior(
                speaker,
                face_id,
                face_identity,
            )

            association = association_prior(
                speaker,
                face_id,
                hypothesis_priors,
            )

            continuity = continuity_score(
                face_id,
                previous_face,
                time_value,
                previous_time,
            )

            visibility_value = visibility_score(
                detection
            )

            active = active_face_score(
                face_id,
                active_face,
            )

            # ------------------------------------------------
            # Score.
            #
            # IMPORTANT:
            # "speaking" is local evidence.
            # "identity" + "association" are identity priors.
            # ------------------------------------------------

            score = (
                SPEAKING_WEIGHT
                * speaking
                +
                ASSOCIATION_WEIGHT
                * association
                +
                IDENTITY_WEIGHT
                * identity
                +
                CONTINUITY_WEIGHT
                * continuity
                +
                VISIBILITY_WEIGHT
                * visibility_value
                +
                ACTIVE_FACE_WEIGHT
                * active
            )

            # Speech being active is useful confirmation,
            # but not a standalone identity signal.
            if speech_active:
                score += 0.02

            # Reliable speaker anchor gets a bonus.
            speaker_info = speaker_identity.get(
                speaker
            )

            if (
                speaker_info
                and speaker_info.get(
                    "reliable",
                    False,
                )
                and str(
                    speaker_info.get(
                        "face_id"
                    )
                )
                == face_id
            ):

                score += 0.10

            score = clamp(
                score
            )

            scored.append(
                {
                    "face_id":
                        face_id,
                    "score":
                        score,
                    "speaking":
                        speaking,
                    "mouth_motion":
                        mouth_motion,
                    "speech_active":
                        speech_active,
                    "association":
                        association,
                    "identity":
                        identity,
                    "continuity":
                        continuity,
                    "visibility":
                        visibility_value,
                    "active_face":
                        active,
                }
            )

        scored.sort(
            key=lambda x:
                x["score"],
            reverse=True,
        )

        # ----------------------------------------------------
        # No candidate.
        # ----------------------------------------------------

        if not scored:

            result.append(
                {
                    "time":
                        time_value,
                    "speaker":
                        speaker,
                    "face_id":
                        None,
                    "confidence":
                        0.0,
                    "strength":
                        "unresolved",
                    "active_face":
                        active_face,
                    "candidate_count":
                        0,
                    "candidates":
                        [],
                }
            )

            continue

        best = scored[0]

        second_score = (
            scored[1]["score"]
            if len(scored) > 1
            else 0.0
        )

        margin = (
            best["score"]
            - second_score
        )

        # ----------------------------------------------------
        # Resolution decision.
        #
        # One candidate gets to be usable without a margin
        # contest.
        # ----------------------------------------------------

        if len(scored) == 1:

            usable = (
                best["score"]
                >= USABLE_THRESHOLD
            )

        else:

            usable = (
                best["score"]
                >= USABLE_THRESHOLD
                and margin
                >= MIN_MARGIN
            )

        resolved_face = (
            best["face_id"]
            if usable
            else None
        )

        strength = (
            "strong"
            if (
                usable
                and best["score"]
                >= STRONG_THRESHOLD
            )
            else (
                "usable"
                if usable
                else "ambiguous"
            )
        )

        if usable:

            previous_by_speaker[
                speaker
            ] = (
                best["face_id"],
                time_value,
            )

        result.append(
            {
                "time":
                    time_value,
                "speaker":
                    speaker,
                "face_id":
                    resolved_face,
                "confidence":
                    round(
                        best["score"],
                        4,
                    ),
                "margin":
                    round(
                        margin,
                        4,
                    ),
                "strength":
                    strength,
                "active_face":
                    active_face,
                "candidate_count":
                    len(scored),
                "candidates":
                    [
                        {
                            "face_id":
                                candidate[
                                    "face_id"
                                ],
                            "score":
                                round(
                                    candidate[
                                        "score"
                                    ],
                                    4,
                                ),
                            "speaking":
                                round(
                                    candidate[
                                        "speaking"
                                    ],
                                    4,
                                ),
                            "mouth_motion":
                                round(
                                    candidate[
                                        "mouth_motion"
                                    ],
                                    4,
                                ),
                            "speech_active":
                                candidate[
                                    "speech_active"
                                ],
                            "association":
                                round(
                                    candidate[
                                        "association"
                                    ],
                                    4,
                                ),
                            "identity":
                                round(
                                    candidate[
                                        "identity"
                                    ],
                                    4,
                                ),
                            "continuity":
                                round(
                                    candidate[
                                        "continuity"
                                    ],
                                    4,
                                ),
                            "visibility":
                                round(
                                    candidate[
                                        "visibility"
                                    ],
                                    4,
                                ),
                            "active_face":
                                round(
                                    candidate[
                                        "active_face"
                                    ],
                                    4,
                                ),
                        }
                        for candidate
                        in scored[:10]
                    ],
            }
        )

    return result


# ============================================================
# SEGMENTS
# ============================================================

def build_segments(
    timeline: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    if not timeline:
        return []

    segments = []

    current = None

    for item in timeline:

        identity = (
            item.get("speaker"),
            item.get("face_id"),
        )

        if current is None:

            current = {
                "start":
                    item["time"],
                "end":
                    item["time"],
                "speaker":
                    item.get(
                        "speaker"
                    ),
                "face_id":
                    item.get(
                        "face_id"
                    ),
                "confidence_sum":
                    0.0,
                "samples":
                    0,
                "strengths":
                    [],
            }

        current_identity = (
            current["speaker"],
            current["face_id"],
        )

        if identity == current_identity:

            current["end"] = (
                item["time"]
            )

        else:

            segments.append(
                current
            )

            current = {
                "start":
                    item["time"],
                "end":
                    item["time"],
                "speaker":
                    item.get(
                        "speaker"
                    ),
                "face_id":
                    item.get(
                        "face_id"
                    ),
                "confidence_sum":
                    0.0,
                "samples":
                    0,
                "strengths":
                    [],
            }

        current[
            "confidence_sum"
        ] += float(
            item.get(
                "confidence",
                0.0,
            )
        )

        current[
            "samples"
        ] += 1

        current[
            "strengths"
        ].append(
            item.get(
                "strength",
                "none",
            )
        )

    if current is not None:

        segments.append(
            current
        )

    result = []

    for segment in segments:

        samples = max(
            1,
            segment["samples"],
        )

        strengths = segment[
            "strengths"
        ]

        if "strong" in strengths:
            strength = "strong"
        elif "usable" in strengths:
            strength = "usable"
        else:
            strength = "unresolved"

        result.append(
            {
                "start":
                    round(
                        segment["start"],
                        3,
                    ),
                "end":
                    round(
                        segment["end"],
                        3,
                    ),
                "speaker":
                    segment["speaker"],
                "face_id":
                    segment["face_id"],
                "confidence":
                    round(
                        segment[
                            "confidence_sum"
                        ]
                        / samples,
                        4,
                    ),
                "samples":
                    samples,
                "strength":
                    strength,
            }
        )

    return result


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Resolve the currently speaking speaker "
            "to the currently visible face using "
            "timestamp-local speaking evidence plus "
            "speaker/face priors."
        )
    )

    parser.add_argument(
        "--faces",
        required=True,
        help="Face tracking JSON",
    )

    parser.add_argument(
        "--association",
        required=True,
        help="Speaker-face association JSON",
    )

    parser.add_argument(
        "--speaking",
        required=True,
        help="Speaking-face JSON",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    print(
        "Loading face tracks..."
    )

    tracks = load_face_tracks(
        args.faces
    )

    print(
        f"Face tracks: "
        f"{len(tracks)}"
    )

    print()

    print(
        "Loading association..."
    )

    association = load_association(
        args.association
    )

    speaker_identity = (
        get_speaker_identity(
            association
        )
    )

    face_identity = (
        get_face_identity(
            association
        )
    )

    reliable_anchors = {
        speaker: info
        for speaker, info
        in speaker_identity.items()
        if info.get(
            "reliable",
            False,
        )
    }

    print(
        f"Speakers: "
        f"{len(speaker_identity)}"
    )

    print(
        f"Reliable anchors: "
        f"{len(reliable_anchors)}"
    )

    for speaker, info in sorted(
        reliable_anchors.items()
    ):

        print(
            f"  {speaker} -> "
            f"{info.get('face_id')} | "
            f"confidence="
            f"{to_float(info.get('confidence')):.4f}"
        )

    print()

    print(
        "Loading aggregate speaker-face hypotheses..."
    )

    hypothesis_priors = (
        load_hypothesis_priors(
            association
        )
    )

    hypothesis_pairs = sum(
        len(value)
        for value in hypothesis_priors.values()
    )

    print(
        f"Hypothesis pairs: "
        f"{hypothesis_pairs}"
    )

    print()

    print(
        "Loading speaking timeline..."
    )

    speaking_timeline = (
        load_speaking_timeline(
            args.speaking
        )
    )

    print(
        f"Timeline samples: "
        f"{len(speaking_timeline)}"
    )

    print()

    print(
        "Resolving local speaker -> visible face..."
    )

    resolved = resolve_timeline(
        tracks=tracks,
        timeline=speaking_timeline,
        speaker_identity=
            speaker_identity,
        face_identity=
            face_identity,
        hypothesis_priors=
            hypothesis_priors,
    )

    usable = sum(
        1
        for item in resolved
        if item.get(
            "strength"
        )
        in {
            "usable",
            "strong",
        }
    )

    strong = sum(
        1
        for item in resolved
        if item.get(
            "strength"
        )
        == "strong"
    )

    print(
        f"Usable local matches: "
        f"{usable}/"
        f"{len(resolved)}"
    )

    print(
        f"Strong local matches: "
        f"{strong}/"
        f"{len(resolved)}"
    )

    print()

    segments = build_segments(
        resolved
    )

    resolved_segments = [
        segment
        for segment in segments
        if (
            segment.get(
                "speaker"
            ) is not None
            and segment.get(
                "face_id"
            ) is not None
        )
    ]

    print(
        f"Resolved local segments: "
        f"{len(resolved_segments)}"
    )

    print()

    print(
        "LOCAL SPEAKER -> FACE SEGMENTS"
    )

    for index, segment in enumerate(
        resolved_segments,
        start=1,
    ):

        print(
            f"  {index:02d}. "
            f"{segment['start']:.2f}s -> "
            f"{segment['end']:.2f}s | "
            f"{segment['speaker']} -> "
            f"{segment['face_id']} | "
            f"confidence="
            f"{segment['confidence']:.3f} | "
            f"{segment['strength']}"
        )

    output = {
        "version": 3,
        "mode":
            "local_speaker_face_resolution_v3",

        "faces_file":
            os.path.abspath(
                args.faces
            ),

        "association_file":
            os.path.abspath(
                args.association
            ),

        "speaking_file":
            os.path.abspath(
                args.speaking
            ),

        "timeline":
            resolved,

        "segments":
            segments,

        "resolved_segments":
            resolved_segments,

        "anchors":
            reliable_anchors,

        "stats": {
            "face_tracks":
                len(tracks),

            "speakers":
                len(speaker_identity),

            "reliable_anchors":
                len(reliable_anchors),

            "hypothesis_pairs":
                hypothesis_pairs,

            "timeline_samples":
                len(resolved),

            "usable_matches":
                usable,

            "strong_matches":
                strong,

            "resolution_rate":
                round(
                    usable
                    / len(resolved),
                    4,
                )
                if resolved
                else 0.0,

            "strong_rate":
                round(
                    strong
                    / len(resolved),
                    4,
                )
                if resolved
                else 0.0,

            "resolved_segments":
                len(
                    resolved_segments
                ),
        },

        "weights": {
            "speaking":
                SPEAKING_WEIGHT,

            "association":
                ASSOCIATION_WEIGHT,

            "identity":
                IDENTITY_WEIGHT,

            "continuity":
                CONTINUITY_WEIGHT,

            "visibility":
                VISIBILITY_WEIGHT,

            "active_face":
                ACTIVE_FACE_WEIGHT,
        },
    }

    output_path = os.path.abspath(
        args.output
    )

    os.makedirs(
        os.path.dirname(
            output_path
        ),
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


if __name__ == "__main__":
    main()
