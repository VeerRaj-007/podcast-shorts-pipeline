
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# CONFIG
# ============================================================

# A non-anchor face must be reasonably similar to its speaker's
# anchor face.
MEDIAN_THRESHOLD = 0.72
P25_THRESHOLD = 0.55

# Strong visual match.
STRONG_MEDIAN = 0.80
STRONG_P25 = 0.68

# If two speaker anchors are too close, do not force an identity.
MIN_MARGIN = 0.05

# If two face tracks overlap in time substantially, be careful
# about assigning them to the same speaker.
OVERLAP_CONFLICT_THRESHOLD = 0.20

# Very high SFace similarity can overcome simultaneous overlap.
SIMULTANEOUS_OVERRIDE = 0.92


# ============================================================
# JSON
# ============================================================

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# FACE TRACKS
# ============================================================

def load_face_tracks(
    path: str,
) -> Dict[str, Dict[str, Any]]:

    data = load_json(path)

    tracks: Dict[str, Dict[str, Any]] = {}

    for track in data.get("tracks", []):

        if not isinstance(track, dict):
            continue

        face_id = (
            track.get("face_id")
            or track.get("id")
        )

        if face_id is None:
            continue

        face_id = str(face_id)

        detections = []

        # v5 format
        raw_detections = track.get(
            "detections",
            track.get("samples", []),
        )

        for detection in raw_detections:

            if not isinstance(
                detection,
                dict,
            ):
                continue

            time_value = detection.get(
                "time",
                detection.get("timestamp"),
            )

            try:
                t = float(time_value)
            except (
                TypeError,
                ValueError,
            ):
                continue

            detections.append(
                {
                    **detection,
                    "time": t,
                }
            )

        if not detections:
            continue

        detections.sort(
            key=lambda x: x["time"]
        )

        tracks[face_id] = {
            "face_id": face_id,
            "start": detections[0]["time"],
            "end": detections[-1]["time"],
            "detections": detections,
        }

    return tracks


# ============================================================
# SFACE PAIR RESULTS
# ============================================================

def load_pair_results(
    path: str,
) -> Dict[
    Tuple[str, str],
    Dict[str, Any],
]:

    data = load_json(path)

    result = {}

    # Support common names.
    raw_pairs = (
        data.get("pair_results")
        or data.get("pairs")
        or data.get("comparisons")
        or []
    )

    for item in raw_pairs:

        if not isinstance(item, dict):
            continue

        face_a = (
            item.get("face_a")
            or item.get("face1")
            or item.get("a")
        )

        face_b = (
            item.get("face_b")
            or item.get("face2")
            or item.get("b")
        )

        if (
            face_a is None
            or face_b is None
        ):
            continue

        face_a = str(face_a)
        face_b = str(face_b)

        key = (
            (face_a, face_b)
            if face_a < face_b
            else (face_b, face_a)
        )

        result[key] = item

    return result


def get_pair(
    pair_results: Dict[
        Tuple[str, str],
        Dict[str, Any],
    ],
    face_a: str,
    face_b: str,
) -> Optional[Dict[str, Any]]:

    if face_a == face_b:
        return {
            "median": 1.0,
            "p25": 1.0,
            "overlap_ratio": 0.0,
        }

    key = (
        (face_a, face_b)
        if face_a < face_b
        else (face_b, face_a)
    )

    return pair_results.get(key)


def extract_similarity(
    pair: Dict[str, Any],
) -> Tuple[float, float]:

    median_keys = [
        "median",
        "sface_median",
        "similarity_median",
        "median_similarity",
    ]

    p25_keys = [
        "p25",
        "sface_p25",
        "similarity_p25",
        "p25_similarity",
    ]

    median = 0.0
    p25 = 0.0

    for key in median_keys:
        if key in pair:
            try:
                median = float(
                    pair[key]
                )
                break
            except (
                TypeError,
                ValueError,
            ):
                pass

    for key in p25_keys:
        if key in pair:
            try:
                p25 = float(
                    pair[key]
                )
                break
            except (
                TypeError,
                ValueError,
            ):
                pass

    return median, p25


def extract_overlap(
    pair: Dict[str, Any],
) -> float:

    keys = [
        "overlap_ratio",
        "temporal_overlap_ratio",
        "track_overlap_ratio",
    ]

    for key in keys:

        if key in pair:

            try:
                return float(
                    pair[key]
                )
            except (
                TypeError,
                ValueError,
            ):
                pass

    return 0.0


# ============================================================
# SPEAKER ANCHORS
# ============================================================

def load_reliable_speaker_anchors(
    association_path: str,
) -> Dict[str, Dict[str, Any]]:

    data = load_json(
        association_path
    )

    speaker_identity = data.get(
        "speaker_identity",
        {},
    )

    anchors = {}

    for speaker, info in speaker_identity.items():

        if not isinstance(
            info,
            dict,
        ):
            continue

        # THIS is the important field.
        if not info.get(
            "reliable",
            False,
        ):
            continue

        face_id = info.get(
            "face_id"
        )

        if face_id is None:
            continue

        anchors[
            str(speaker)
        ] = {
            "speaker":
                str(speaker),
            "face_id":
                str(face_id),
            "confidence":
                float(
                    info.get(
                        "confidence",
                        0.0,
                    )
                ),
            "margin":
                float(
                    info.get(
                        "margin_over_second",
                        0.0,
                    )
                ),
            "active_face_rate":
                float(
                    info.get(
                        "active_face_rate",
                        0.0,
                    )
                ),
            "motion_score":
                float(
                    info.get(
                        "motion_score",
                        0.0,
                    )
                ),
            "temporal_visibility":
                float(
                    info.get(
                        "temporal_visibility",
                        0.0,
                    )
                ),
            "exclusivity":
                float(
                    info.get(
                        "exclusivity",
                        0.0,
                    )
                ),
            "active_face_samples":
                int(
                    info.get(
                        "active_face_samples",
                        0,
                    )
                ),
            "speaker_samples":
                int(
                    info.get(
                        "speaker_samples",
                        0,
                    )
                ),
            "visible_speaker_samples":
                int(
                    info.get(
                        "visible_speaker_samples",
                        0,
                    )
                ),
        }

    return anchors


# ============================================================
# TEMPORAL OVERLAP
# ============================================================

def track_overlap_ratio(
    track_a: Dict[str, Any],
    track_b: Dict[str, Any],
) -> float:

    start_a = float(
        track_a["start"]
    )

    end_a = float(
        track_a["end"]
    )

    start_b = float(
        track_b["start"]
    )

    end_b = float(
        track_b["end"]
    )

    overlap_start = max(
        start_a,
        start_b,
    )

    overlap_end = min(
        end_a,
        end_b,
    )

    overlap = max(
        0.0,
        overlap_end
        - overlap_start,
    )

    duration_a = max(
        0.001,
        end_a - start_a,
    )

    duration_b = max(
        0.001,
        end_b - start_b,
    )

    denominator = min(
        duration_a,
        duration_b,
    )

    return overlap / denominator


# ============================================================
# FACE → SPEAKER SCORE
# ============================================================

def score_face_against_anchor(
    face_id: str,
    anchor_face_id: str,
    speaker: str,
    tracks: Dict[str, Dict[str, Any]],
    pair_results: Dict[
        Tuple[str, str],
        Dict[str, Any],
    ],
) -> Dict[str, Any]:

    pair = get_pair(
        pair_results,
        face_id,
        anchor_face_id,
    )

    if pair is None:

        return {
            "compatible": False,
            "score": 0.0,
            "median": 0.0,
            "p25": 0.0,
            "overlap_ratio": 0.0,
            "reason":
                "no_sface_pair",
        }

    median, p25 = extract_similarity(
        pair
    )

    overlap_ratio = extract_overlap(
        pair
    )

    # If the pair file doesn't contain overlap,
    # calculate it directly from the tracks.
    if (
        overlap_ratio <= 0.0
        and face_id in tracks
        and anchor_face_id in tracks
    ):
        overlap_ratio = track_overlap_ratio(
            tracks[face_id],
            tracks[anchor_face_id],
        )

    # --------------------------------------------------------
    # Two different faces visible at the same time.
    # Don't merge them unless appearance similarity
    # is exceptionally high.
    # --------------------------------------------------------

    if (
        overlap_ratio
        >= OVERLAP_CONFLICT_THRESHOLD
        and median
        < SIMULTANEOUS_OVERRIDE
    ):

        return {
            "compatible": False,
            "score": 0.0,
            "median": median,
            "p25": p25,
            "overlap_ratio":
                overlap_ratio,
            "reason":
                "simultaneous_face_conflict",
        }

    # --------------------------------------------------------
    # Appearance consistency gate.
    # --------------------------------------------------------

    if (
        median
        < MEDIAN_THRESHOLD
        or p25
        < P25_THRESHOLD
    ):

        return {
            "compatible": False,
            "score": 0.0,
            "median": median,
            "p25": p25,
            "overlap_ratio":
                overlap_ratio,
            "reason":
                "weak_sface_similarity",
        }

    # Median is more important than p25.
    score = (
        0.75 * median
        + 0.25 * p25
    )

    # Small bonus for strong consistency.
    if (
        median >= STRONG_MEDIAN
        and p25 >= STRONG_P25
    ):
        score += 0.02

    # Tiny bonus when tracks don't overlap.
    # This encourages sensible camera-cut propagation.
    if overlap_ratio < 0.05:
        score += 0.01

    score = min(
        1.0,
        score,
    )

    return {
        "compatible": True,
        "score": score,
        "median": median,
        "p25": p25,
        "overlap_ratio":
            overlap_ratio,
        "reason":
            "strong_sface_match"
            if (
                median >= STRONG_MEDIAN
                and p25 >= STRONG_P25
            )
            else "acceptable_sface_match",
    }


# ============================================================
# RESOLVE ALL FACE TRACKS
# ============================================================

def resolve_face_identities(
    tracks: Dict[str, Dict[str, Any]],
    anchors: Dict[str, Dict[str, Any]],
    pair_results: Dict[
        Tuple[str, str],
        Dict[str, Any],
    ],
) -> Dict[str, Any]:

    face_assignments = {}

    speaker_faces = defaultdict(list)

    # --------------------------------------------------------
    # Every face is compared against every reliable
    # speaker anchor.
    # --------------------------------------------------------

    for face_id in tracks.keys():

        candidates = []

        for speaker, anchor in anchors.items():

            anchor_face = anchor[
                "face_id"
            ]

            result = score_face_against_anchor(
                face_id=face_id,
                anchor_face_id=anchor_face,
                speaker=speaker,
                tracks=tracks,
                pair_results=pair_results,
            )

            if face_id == anchor_face:

                result = {
                    "compatible": True,
                    "score": 1.0,
                    "median": 1.0,
                    "p25": 1.0,
                    "overlap_ratio": 0.0,
                    "reason": "reliable_anchor",
                }

            if result[
                "compatible"
            ]:

                candidates.append(
                    {
                        "speaker":
                            speaker,
                        "anchor_face":
                            anchor_face,
                        **result,
                    }
                )

        candidates.sort(
            key=lambda item:
                item["score"],
            reverse=True,
        )

        # ----------------------------------------------------
        # No speaker-compatible identity.
        # ----------------------------------------------------

        if not candidates:

            face_assignments[
                face_id
            ] = {
                "face_id":
                    face_id,
                "speaker":
                    None,
                "confidence":
                    0.0,
                "margin":
                    0.0,
                "stable":
                    False,
                "reason":
                    "unresolved",
                "candidates":
                    [],
            }

            continue

        best = candidates[0]

        second_score = (
            candidates[1]["score"]
            if len(candidates) > 1
            else 0.0
        )

        margin = (
            best["score"]
            - second_score
        )

        # Anchor itself is always stable.
        is_anchor = (
            best["reason"]
            == "reliable_anchor"
        )

        stable = (
            is_anchor
            or (
                best["score"]
                >= MEDIAN_THRESHOLD
                and margin
                >= MIN_MARGIN
            )
        )

        assigned_speaker = (
            best["speaker"]
            if stable
            else None
        )

        face_assignments[
            face_id
        ] = {
            "face_id":
                face_id,
            "speaker":
                assigned_speaker,
            "anchor_face":
                (
                    best["anchor_face"]
                    if stable
                    else None
                ),
            "confidence":
                round(
                    float(
                        best[
                            "score"
                        ]
                    ),
                    4,
                ),
            "margin":
                round(
                    float(
                        margin
                    ),
                    4,
                ),
            "stable":
                stable,
            "reason":
                (
                    "anchor"
                    if is_anchor
                    else (
                        "sface_propagated"
                        if stable
                        else "ambiguous"
                    )
                ),
            "candidates":
                [
                    {
                        "speaker":
                            c[
                                "speaker"
                            ],
                        "anchor_face":
                            c[
                                "anchor_face"
                            ],
                        "score":
                            round(
                                float(
                                    c[
                                        "score"
                                    ]
                                ),
                                4,
                            ),
                        "median":
                            round(
                                float(
                                    c[
                                        "median"
                                    ]
                                ),
                                4,
                            ),
                        "p25":
                            round(
                                float(
                                    c[
                                        "p25"
                                    ]
                                ),
                                4,
                            ),
                        "overlap_ratio":
                            round(
                                float(
                                    c[
                                        "overlap_ratio"
                                    ]
                                ),
                                4,
                            ),
                        "reason":
                            c[
                                "reason"
                            ],
                    }
                    for c in candidates
                ],
        }

        if stable:
            speaker_faces[
                assigned_speaker
            ].append(
                face_id
            )

    # --------------------------------------------------------
    # Build PERSON objects.
    #
    # IMPORTANT:
    # A person is created only for a reliable speaker.
    # Unresolved visual tracks are NOT converted into fake
    # people.
    # --------------------------------------------------------

    persons = []

    face_to_person = {}

    for speaker, anchor in anchors.items():

        person_id = (
            f"PERSON_{speaker}"
        )

        faces = list(
            speaker_faces.get(
                speaker,
                [],
            )
        )

        anchor_face = anchor[
            "face_id"
        ]

        if anchor_face not in faces:
            faces.insert(
                0,
                anchor_face,
            )

        # Deduplicate.
        faces = list(
            dict.fromkeys(
                faces
            )
        )

        for face_id in faces:
            face_to_person[
                face_id
            ] = person_id

        valid_tracks = [
            tracks[face_id]
            for face_id in faces
            if face_id in tracks
        ]

        start = min(
            track["start"]
            for track in valid_tracks
        )

        end = max(
            track["end"]
            for track in valid_tracks
        )

        persons.append(
            {
                "person_id":
                    person_id,
                "speaker":
                    speaker,
                "anchor_face":
                    anchor_face,
                "anchor_confidence":
                    anchor[
                        "confidence"
                    ],
                "anchor_margin":
                    anchor[
                        "margin"
                    ],
                "face_ids":
                    sorted(
                        faces,
                        key=lambda fid:
                            tracks[
                                fid
                            ]["start"],
                    ),
                "track_count":
                    len(faces),
                "start":
                    round(
                        start,
                        3,
                    ),
                "end":
                    round(
                        end,
                        3,
                    ),
            }
        )

    return {
        "persons":
            persons,
        "face_to_person":
            face_to_person,
        "face_assignments":
            face_assignments,
        "speaker_faces":
            {
                speaker:
                    sorted(
                        faces,
                        key=lambda fid:
                            tracks[
                                fid
                            ]["start"],
                    )
                for speaker, faces
                in speaker_faces.items()
            },
    }


# ============================================================
# SPEAKER → PERSON → FACE TIMELINE
# ============================================================

def build_timeline(
    speaking_path: str,
    identity_data: Dict[str, Any],
) -> List[Dict[str, Any]]:

    data = load_json(
        speaking_path
    )

    face_assignments = identity_data[
        "face_assignments"
    ]

    face_to_person = identity_data[
        "face_to_person"
    ]

    timeline = []

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
                item.get("time")
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        active_face = item.get(
            "active_face"
        )

        active_speaker = item.get(
            "active_speaker"
        )

        if active_face is not None:
            active_face = str(
                active_face
            )

        if active_speaker is not None:
            active_speaker = str(
                active_speaker
            )

        assignment = (
            face_assignments.get(
                active_face
            )
            if active_face
            else None
        )

        person_id = (
            face_to_person.get(
                active_face
            )
            if active_face
            else None
        )

        # A face belongs to the speaker only when
        # its propagated identity agrees with diarization.
        speaker_agrees = (
            assignment is not None
            and assignment.get(
                "speaker"
            )
            == active_speaker
            and assignment.get(
                "stable",
                False,
            )
        )

        if not speaker_agrees:
            person_id = None

        confidence = (
            float(
                assignment.get(
                    "confidence",
                    0.0,
                )
            )
            if (
                assignment
                and speaker_agrees
            )
            else 0.0
        )

        timeline.append(
            {
                "time":
                    time_value,
                "speaker":
                    active_speaker,
                "person_id":
                    person_id,
                "face_id":
                    active_face,
                "confidence":
                    round(
                        confidence,
                        4,
                    ),
                "stable":
                    speaker_agrees,
            }
        )

    return timeline


# ============================================================
# SUMMARY SEGMENTS
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
            item.get("person_id"),
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
                "person_id":
                    item.get(
                        "person_id"
                    ),
                "faces":
                    [],
                "confidence_sum":
                    0.0,
                "samples":
                    0,
            }

        current_identity = (
            current["speaker"],
            current["person_id"],
        )

        # Continue current identity.
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
                "person_id":
                    item.get(
                        "person_id"
                    ),
                "faces":
                    [],
                "confidence_sum":
                    0.0,
                "samples":
                    0,
            }

        if item.get(
            "face_id"
        ) is not None:

            current[
                "faces"
            ].append(
                item[
                    "face_id"
                ]
            )

        if item.get(
            "stable",
            False,
        ):

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

    if current is not None:
        segments.append(
            current
        )

    # Clean + merge very short empty runs later.
    cleaned = []

    for segment in segments:

        faces = list(
            dict.fromkeys(
                segment[
                    "faces"
                ]
            )
        )

        samples = segment[
            "samples"
        ]

        mean_confidence = (
            segment[
                "confidence_sum"
            ]
            / samples
            if samples > 0
            else 0.0
        )

        cleaned.append(
            {
                "start":
                    round(
                        segment[
                            "start"
                        ],
                        3,
                    ),
                "end":
                    round(
                        segment[
                            "end"
                        ],
                        3,
                    ),
                "speaker":
                    segment[
                        "speaker"
                    ],
                "person_id":
                    segment[
                        "person_id"
                    ],
                "faces":
                    faces,
                "confidence":
                    round(
                        mean_confidence,
                        4,
                    ),
                "samples":
                    samples,
            }
        )

    return cleaned


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Build speaker-anchored persistent "
            "person identities using reliable "
            "speaker-face anchors + existing SFace "
            "pair comparisons."
        )
    )

    parser.add_argument(
        "--faces",
        required=True,
    )

    parser.add_argument(
        "--association",
        required=True,
    )

    parser.add_argument(
        "--person-matrix",
        required=True,
    )

    parser.add_argument(
        "--speaking",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
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
        f"Face tracks: {len(tracks)}"
    )

    print()

    print(
        "Loading reliable speaker anchors..."
    )

    anchors = (
        load_reliable_speaker_anchors(
            args.association
        )
    )

    print(
        f"Reliable speaker anchors: "
        f"{len(anchors)}"
    )

    for speaker, anchor in sorted(
        anchors.items()
    ):

        print(
            f"  {speaker} -> "
            f"{anchor['face_id']} | "
            f"confidence="
            f"{anchor['confidence']:.4f} | "
            f"margin="
            f"{anchor['margin']:.4f}"
        )

    print()

    print(
        "Loading existing SFace pair matrix..."
    )

    pair_results = load_pair_results(
        args.person_matrix
    )

    print(
        f"Pair comparisons: "
        f"{len(pair_results)}"
    )

    print()

    print(
        "Propagating speaker identities..."
    )

    identity_data = (
        resolve_face_identities(
            tracks=tracks,
            anchors=anchors,
            pair_results=pair_results,
        )
    )

    print()

    print(
        "SPEAKER → FACE TRACKS"
    )

    for speaker, faces in sorted(
        identity_data[
            "speaker_faces"
        ].items()
    ):

        print(
            f"  {speaker}: "
            f"{', '.join(faces)}"
        )

    print()

    print(
        "Building local timeline..."
    )

    timeline = build_timeline(
        speaking_path=args.speaking,
        identity_data=identity_data,
    )

    stable_samples = sum(
        1
        for item in timeline
        if item.get(
            "stable",
            False,
        )
    )

    print(
        f"Stable timeline samples: "
        f"{stable_samples}/"
        f"{len(timeline)}"
    )

    print()

    print(
        "Building identity segments..."
    )

    segments = build_segments(
        timeline
    )

    # Only useful resolved segments.
    resolved_segments = [
        segment
        for segment in segments
        if (
            segment.get(
                "speaker"
            )
            is not None
            and segment.get(
                "person_id"
            ) is not None
        )
    ]

    print(
        f"Resolved identity segments: "
        f"{len(resolved_segments)}"
    )

    print()

    print(
        "SPEAKER → PERSON SEGMENTS"
    )

    for segment in resolved_segments:

        print(
            f"  "
            f"{segment['start']:.2f}s -> "
            f"{segment['end']:.2f}s | "
            f"{segment['speaker']} -> "
            f"{segment['person_id']} | "
            f"faces="
            f"{', '.join(segment['faces'])} | "
            f"confidence="
            f"{segment['confidence']:.3f}"
        )

    output = {
        "version": 2,
        "mode":
            "speaker_anchored_identity_v2",
        "faces_file":
            os.path.abspath(
                args.faces
            ),
        "association_file":
            os.path.abspath(
                args.association
            ),
        "person_matrix_file":
            os.path.abspath(
                args.person_matrix
            ),
        "speaking_file":
            os.path.abspath(
                args.speaking
            ),

        "anchors":
            anchors,

        "persons":
            identity_data[
                "persons"
            ],

        "face_to_person":
            identity_data[
                "face_to_person"
            ],

        "face_assignments":
            identity_data[
                "face_assignments"
            ],

        "speaker_faces":
            identity_data[
                "speaker_faces"
            ],

        "timeline":
            timeline,

        "segments":
            segments,

        "resolved_segments":
            resolved_segments,

        "stats": {
            "face_tracks":
                len(tracks),
            "reliable_anchors":
                len(anchors),
            "persons":
                len(
                    identity_data[
                        "persons"
                    ]
                ),
            "timeline_samples":
                len(timeline),
            "stable_timeline_samples":
                stable_samples,
            "timeline_resolution":
                (
                    round(
                        stable_samples
                        / len(timeline),
                        4,
                    )
                    if timeline
                    else 0.0
                ),
            "resolved_segments":
                len(
                    resolved_segments
                ),
        },

        "thresholds": {
            "median":
                MEDIAN_THRESHOLD,
            "p25":
                P25_THRESHOLD,
            "strong_median":
                STRONG_MEDIAN,
            "strong_p25":
                STRONG_P25,
            "min_margin":
                MIN_MARGIN,
            "overlap_conflict":
                OVERLAP_CONFLICT_THRESHOLD,
            "simultaneous_override":
                SIMULTANEOUS_OVERRIDE,
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
        "FINAL PERSON IDENTITIES"
    )

    for person in output[
        "persons"
    ]:

        print(
            f"  {person['person_id']}: "
            f"{', '.join(person['face_ids'])}"
        )

    print()

    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()
