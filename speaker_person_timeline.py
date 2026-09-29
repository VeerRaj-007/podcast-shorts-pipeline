
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# Configuration
# ============================================================

# Minimum local confidence before we allow a mapping to drive
# camera decisions.
MIN_LOCAL_CONFIDENCE = 0.35

# Confidence supplied by speaking-face detector.
SPEAKING_CONFIDENCE_WEIGHT = 0.60

# Confidence supplied by speaker-face association.
ASSOCIATION_CONFIDENCE_WEIGHT = 0.25

# Small bonus when the selected face belongs to the candidate
# persistent person.
PERSON_CONSISTENCY_WEIGHT = 0.15

# A candidate must beat the second-best candidate by this amount
# before we call the local mapping confident.
MIN_LOCAL_MARGIN = 0.05

# Smooth rapid identity flicker.
SMOOTH_WINDOW = 5

# Require this many consecutive compatible samples before
# switching the active person.
MIN_SWITCH_SAMPLES = 2


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
# Person identity
# ============================================================

def load_person_identity(
    path: str,
) -> Tuple[
    Dict[str, str],
    Dict[str, List[str]],
]:
    data = load_json(path)

    face_to_person: Dict[str, str] = {}

    raw_mapping = data.get(
        "face_to_person",
        {},
    )

    if isinstance(raw_mapping, dict):

        for face_id, person_id in raw_mapping.items():

            face_to_person[
                str(face_id)
            ] = str(person_id)

    person_faces: Dict[
        str,
        List[str],
    ] = defaultdict(list)

    raw_persons = data.get(
        "persons",
        [],
    )

    if isinstance(
        raw_persons,
        list,
    ):

        for person in raw_persons:

            if not isinstance(
                person,
                dict,
            ):
                continue

            person_id = person.get(
                "person_id"
            )

            if person_id is None:
                continue

            person_id = str(
                person_id
            )

            for face_id in person.get(
                "face_ids",
                [],
            ):

                person_faces[
                    person_id
                ].append(
                    str(face_id)
                )

    # Keep mapping internally consistent.
    for face_id, person_id in (
        face_to_person.items()
    ):

        if face_id not in person_faces[
            person_id
        ]:

            person_faces[
                person_id
            ].append(
                face_id
            )

    for faces in person_faces.values():

        faces[:] = sorted(
            set(faces)
        )

    return (
        face_to_person,
        dict(person_faces),
    )


# ============================================================
# Speaking-face timeline
# ============================================================

def load_speaking_timeline(
    path: str,
) -> List[Dict[str, Any]]:

    data = load_json(path)

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

        t = as_float(
            item.get("time")
        )

        if t is None:
            continue

        active_face = item.get(
            "active_face"
        )

        active_speaker = item.get(
            "active_speaker"
        )

        confidence = as_float(
            item.get(
                "confidence",
                0.0,
            )
        )

        timeline.append(
            {
                "time": float(t),
                "active_face": (
                    str(active_face)
                    if active_face is not None
                    else None
                ),
                "active_speaker": (
                    str(active_speaker)
                    if active_speaker is not None
                    else None
                ),
                "confidence": float(
                    confidence or 0.0
                ),
            }
        )

    timeline.sort(
        key=lambda x: x["time"]
    )

    return timeline


# ============================================================
# Speaker-face association
# ============================================================

def load_association_hypotheses(
    path: str,
) -> Dict[
    str,
    Dict[str, float],
]:

    data = load_json(path)

    result: Dict[
        str,
        Dict[str, float],
    ] = defaultdict(dict)

    hypotheses = data.get(
        "hypotheses",
        [],
    )

    if not isinstance(
        hypotheses,
        list,
    ):
        return {}

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
            or confidence is None
        ):
            continue

        face_id = str(
            face_id
        )

        speaker = str(
            speaker
        )

        result[
            face_id
        ][
            speaker
        ] = max(
            result[
                face_id
            ].get(
                speaker,
                0.0,
            ),
            float(
                confidence
            ),
        )

    return dict(result)


# ============================================================
# Face observations
# ============================================================

def load_observations(
    path: str,
) -> Dict[
    float,
    List[Dict[str, Any]],
]:

    data = load_json(path)

    by_time: Dict[
        float,
        List[Dict[str, Any]],
    ] = defaultdict(list)

    for item in data.get(
        "observations",
        [],
    ):

        if not isinstance(
            item,
            dict,
        ):
            continue

        t = as_float(
            item.get("time")
        )

        face_id = item.get(
            "face_id"
        )

        if (
            t is None
            or face_id is None
        ):
            continue

        motion = as_float(
            item.get(
                "mouth_motion",
                0.0,
            )
        )

        by_time[
            round(
                float(t),
                3,
            )
        ].append(
            {
                "face_id": str(
                    face_id
                ),
                "mouth_motion": float(
                    motion or 0.0
                ),
            }
        )

    return dict(by_time)


# ============================================================
# Speaker diarization
# ============================================================

def collect_speaker_segments(
    data: Any,
) -> List[Dict[str, Any]]:

    result = []

    seen = set()

    def find_speaker(
        node: Dict[str, Any],
        inherited: Optional[str],
    ) -> Optional[str]:

        for key in (
            "speaker",
            "speaker_id",
            "speakerId",
            "label",
            "name",
        ):

            if node.get(key) is not None:
                return str(
                    node[key]
                )

        return inherited

    def find_times(
        node: Dict[str, Any],
    ) -> Optional[Tuple[float, float]]:

        start = as_float(
            node.get(
                "start",
                node.get(
                    "start_time",
                    node.get(
                        "begin"
                    ),
                ),
            )
        )

        end = as_float(
            node.get(
                "end",
                node.get(
                    "end_time",
                    node.get(
                        "stop"
                    ),
                ),
            )
        )

        if (
            start is None
            or end is None
            or end <= start
        ):
            return None

        return (
            start,
            end,
        )

    def walk(
        node: Any,
        inherited: Optional[str] = None,
    ) -> None:

        if isinstance(
            node,
            list,
        ):

            for child in node:

                walk(
                    child,
                    inherited,
                )

            return

        if not isinstance(
            node,
            dict,
        ):
            return

        speaker = find_speaker(
            node,
            inherited,
        )

        times = find_times(
            node
        )

        if (
            speaker is not None
            and times is not None
        ):

            start, end = times

            key = (
                round(start, 6),
                round(end, 6),
                speaker,
            )

            if key not in seen:

                seen.add(key)

                result.append(
                    {
                        "start": start,
                        "end": end,
                        "speaker": speaker,
                    }
                )

        for key, value in node.items():

            if key in {
                "version",
                "mode",
                "video",
                "audio",
            }:
                continue

            walk(
                value,
                speaker,
            )

    walk(data)

    result.sort(
        key=lambda x: (
            x["start"],
            x["end"],
        )
    )

    return result


# ============================================================
# Speaker lookup
# ============================================================

def active_speakers_at(
    t: float,
    segments: List[Dict[str, Any]],
) -> List[str]:

    speakers = []

    for segment in segments:

        if (
            segment["start"]
            <= t
            <= segment["end"]
        ):

            speakers.append(
                str(
                    segment["speaker"]
                )
            )

    return speakers


# ============================================================
# Timeline lookup
# ============================================================

def nearest_timeline_point(
    timeline: List[Dict[str, Any]],
    t: float,
) -> Optional[Dict[str, Any]]:

    if not timeline:
        return None

    return min(
        timeline,
        key=lambda point:
            abs(
                point["time"]
                - t
            ),
    )


# ============================================================
# Local candidate generation
# ============================================================

def build_local_candidates(
    timeline_point: Optional[Dict[str, Any]],
    speaker: str,
    face_to_person: Dict[str, str],
    association: Dict[
        str,
        Dict[str, float],
    ],
    observations: Dict[
        float,
        List[Dict[str, Any]],
    ],
) -> List[Dict[str, Any]]:

    candidate_faces: Dict[
        str,
        Dict[str, Any],
    ] = {}

    # --------------------------------------------------------
    # 1. Speaking-face detector selected face.
    # --------------------------------------------------------

    if (
        timeline_point is not None
        and timeline_point.get(
            "active_speaker"
        ) == speaker
        and timeline_point.get(
            "active_face"
        ) is not None
    ):

        face_id = str(
            timeline_point[
                "active_face"
            ]
        )

        confidence = float(
            timeline_point.get(
                "confidence",
                0.0,
            )
            or 0.0
        )

        candidate_faces[
            face_id
        ] = {
            "speaking_confidence":
                confidence,
            "association_confidence":
                0.0,
        }

    # --------------------------------------------------------
    # 2. Association hypotheses.
    # --------------------------------------------------------

    for face_id, speaker_scores in (
        association.items()
    ):

        confidence = float(
            speaker_scores.get(
                speaker,
                0.0,
            )
        )

        if confidence <= 0.0:
            continue

        existing = candidate_faces.get(
            face_id,
            {
                "speaking_confidence": 0.0,
                "association_confidence": 0.0,
            },
        )

        existing[
            "association_confidence"
        ] = max(
            existing[
                "association_confidence"
            ],
            confidence,
        )

        candidate_faces[
            face_id
        ] = existing

    # --------------------------------------------------------
    # 3. Convert FACE -> PERSON.
    # --------------------------------------------------------

    candidates = []

    timeline_face = (
        timeline_point.get(
            "active_face"
        )
        if timeline_point is not None
        else None
    )

    for face_id, evidence in (
        candidate_faces.items()
    ):

        person_id = face_to_person.get(
            face_id
        )

        if person_id is None:
            continue

        speaking_conf = float(
            evidence[
                "speaking_confidence"
            ]
        )

        association_conf = float(
            evidence[
                "association_confidence"
            ]
        )

        person_consistency = (
            1.0
            if (
                timeline_face is not None
                and face_id == timeline_face
            )
            else 0.0
        )

        score = (
            SPEAKING_CONFIDENCE_WEIGHT
            * speaking_conf
            +
            ASSOCIATION_CONFIDENCE_WEIGHT
            * association_conf
            +
            PERSON_CONSISTENCY_WEIGHT
            * person_consistency
        )

        candidates.append(
            {
                "speaker": speaker,
                "person_id": person_id,
                "face_id": face_id,
                "score": float(
                    np.clip(
                        score,
                        0.0,
                        1.0,
                    )
                ),
                "speaking_confidence":
                    speaking_conf,
                "association_confidence":
                    association_conf,
                "person_consistency":
                    person_consistency,
            }
        )

    # --------------------------------------------------------
    # Multiple face tracks belonging to the same PERSON:
    # keep the strongest face evidence.
    # --------------------------------------------------------

    best_by_person: Dict[
        str,
        Dict[str, Any],
    ] = {}

    for candidate in candidates:

        person = candidate[
            "person_id"
        ]

        current = best_by_person.get(
            person
        )

        if (
            current is None
            or candidate["score"]
            > current["score"]
        ):

            best_by_person[
                person
            ] = candidate

    result = list(
        best_by_person.values()
    )

    result.sort(
        key=lambda x:
            x["score"],
        reverse=True,
    )

    return result


# ============================================================
# Temporal smoothing
# ============================================================

def smooth_timeline(
    raw: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    if not raw:
        return []

    output = []

    previous_person = None
    pending_person = None
    pending_count = 0

    for index, point in enumerate(
        raw
    ):

        candidate = point.get(
            "person_id"
        )

        confidence = float(
            point.get(
                "confidence",
                0.0,
            )
        )

        # ----------------------------------------------------
        # No local identity.
        # ----------------------------------------------------

        if candidate is None:

            output.append(
                {
                    **point,
                    "smoothed_person": (
                        previous_person
                    ),
                }
            )

            pending_person = None
            pending_count = 0

            continue

        # ----------------------------------------------------
        # Same person as previous.
        # ----------------------------------------------------

        if candidate == previous_person:

            pending_person = None
            pending_count = 0

            output.append(
                {
                    **point,
                    "smoothed_person":
                        previous_person,
                }
            )

            continue

        # ----------------------------------------------------
        # Candidate wants to switch.
        # ----------------------------------------------------

        if candidate == pending_person:

            pending_count += 1

        else:

            pending_person = candidate
            pending_count = 1

        # Switch only after consistent evidence.
        should_switch = (
            pending_count
            >= MIN_SWITCH_SAMPLES
            and confidence
            >= MIN_LOCAL_CONFIDENCE
        )

        if should_switch:

            previous_person = (
                pending_person
            )

            pending_person = None
            pending_count = 0

        output.append(
            {
                **point,
                "smoothed_person":
                    previous_person,
            }
        )

    return output


# ============================================================
# Build local timeline
# ============================================================

def build_local_timeline(
    timeline: List[Dict[str, Any]],
    speaker_segments: List[Dict[str, Any]],
    face_to_person: Dict[str, str],
    association: Dict[
        str,
        Dict[str, float],
    ],
    observations: Dict[
        float,
        List[Dict[str, Any]],
    ],
) -> List[Dict[str, Any]]:

    output = []

    for point in timeline:

        t = float(
            point["time"]
        )

        speaker = point.get(
            "active_speaker"
        )

        active_face = point.get(
            "active_face"
        )

        if speaker is None:

            output.append(
                {
                    "time": t,
                    "speaker": None,
                    "face_id": None,
                    "person_id": None,
                    "confidence": 0.0,
                    "reason": "no_active_speaker",
                }
            )

            continue

        speaker = str(
            speaker
        )

        candidates = build_local_candidates(
            timeline_point=point,
            speaker=speaker,
            face_to_person=face_to_person,
            association=association,
            observations=observations,
        )

        best = (
            candidates[0]
            if candidates
            else None
        )

        second = (
            candidates[1]
            if len(candidates) > 1
            else None
        )

        if best is None:

            output.append(
                {
                    "time": t,
                    "speaker": speaker,
                    "face_id": (
                        str(active_face)
                        if active_face is not None
                        else None
                    ),
                    "person_id": None,
                    "confidence": 0.0,
                    "margin": 0.0,
                    "stable_local": False,
                    "reason":
                        "no_person_candidate",
                    "candidates": [],
                }
            )

            continue

        second_score = (
            float(
                second["score"]
            )
            if second is not None
            else 0.0
        )

        margin = (
            best["score"]
            - second_score
        )

        stable_local = (
            best["score"]
            >= MIN_LOCAL_CONFIDENCE
            and margin
            >= MIN_LOCAL_MARGIN
        )

        output.append(
            {
                "time": t,
                "speaker": speaker,
                "face_id": best[
                    "face_id"
                ],
                "person_id": best[
                    "person_id"
                ],
                "confidence": round(
                    best["score"],
                    4,
                ),
                "margin": round(
                    margin,
                    4,
                ),
                "stable_local":
                    stable_local,
                "reason": (
                    "strong_local_match"
                    if stable_local
                    else "weak_local_match"
                ),
                "candidates": [
                    {
                        **candidate,
                        "score": round(
                            candidate["score"],
                            4,
                        ),
                    }
                    for candidate
                    in candidates[:10]
                ],
            }
        )

    return output


# ============================================================
# Segment extraction
# ============================================================

def build_segments(
    timeline: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    if not timeline:
        return []

    segments = []

    current = None

    for point in timeline:

        person = point.get(
            "smoothed_person"
        )

        speaker = point.get(
            "speaker"
        )

        face_id = point.get(
            "face_id"
        )

        t = float(
            point["time"]
        )

        # Ignore points without an identified speaker/person.
        if (
            speaker is None
            or person is None
        ):

            if current is not None:

                current["end"] = t
                segments.append(
                    current
                )
                current = None

            continue

        # Start a segment.
        if current is None:

            current = {
                "speaker": speaker,
                "person_id": person,
                "start": t,
                "end": t,
                "face_ids": set(),
                "samples": 0,
                "confidence_sum": 0.0,
            }

        # Switch.
        elif (
            current["speaker"]
            != speaker
            or current["person_id"]
            != person
        ):

            current["end"] = t

            segments.append(
                current
            )

            current = {
                "speaker": speaker,
                "person_id": person,
                "start": t,
                "end": t,
                "face_ids": set(),
                "samples": 0,
                "confidence_sum": 0.0,
            }

        current["end"] = t
        current["samples"] += 1
        current["confidence_sum"] += float(
            point.get(
                "confidence",
                0.0,
            )
        )

        if face_id is not None:

            current[
                "face_ids"
            ].add(
                str(face_id)
            )

    if current is not None:

        segments.append(
            current
        )

    output = []

    for segment in segments:

        samples = max(
            1,
            int(
                segment["samples"]
            ),
        )

        output.append(
            {
                "speaker":
                    segment["speaker"],
                "person_id":
                    segment["person_id"],
                "start": round(
                    segment["start"],
                    3,
                ),
                "end": round(
                    segment["end"],
                    3,
                ),
                "duration": round(
                    max(
                        0.0,
                        segment["end"]
                        - segment["start"],
                    ),
                    3,
                ),
                "samples":
                    segment["samples"],
                "mean_confidence":
                    round(
                        segment[
                            "confidence_sum"
                        ]
                        / samples,
                        4,
                    ),
                "face_ids":
                    sorted(
                        segment[
                            "face_ids"
                        ]
                    ),
            }
        )

    return output


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Build a local temporal "
            "speaker -> person -> face timeline."
        )
    )

    parser.add_argument(
        "--speakers",
        required=True,
    )

    parser.add_argument(
        "--persons",
        required=True,
    )

    parser.add_argument(
        "--speaking",
        required=True,
    )

    parser.add_argument(
        "--association",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    return parser.parse_args()


def main() -> int:

    args = parse_args()

    # --------------------------------------------------------
    # Person identities.
    # --------------------------------------------------------

    print(
        "Loading person identities..."
    )

    (
        face_to_person,
        person_faces,
    ) = load_person_identity(
        args.persons
    )

    print(
        f"Persons: "
        f"{len(person_faces)}"
    )

    print(
        f"Mapped faces: "
        f"{len(face_to_person)}"
    )

    # --------------------------------------------------------
    # Speaking timeline.
    # --------------------------------------------------------

    print()
    print(
        "Loading speaking-face timeline..."
    )

    speaking_timeline = (
        load_speaking_timeline(
            args.speaking
        )
    )

    print(
        f"Speaking timeline: "
        f"{len(speaking_timeline)}"
    )

    # --------------------------------------------------------
    # Association.
    # --------------------------------------------------------

    print()
    print(
        "Loading speaker-face associations..."
    )

    association = (
        load_association_hypotheses(
            args.association
        )
    )

    association_count = sum(
        len(values)
        for values
        in association.values()
    )

    print(
        f"Association hypotheses: "
        f"{association_count}"
    )

    # --------------------------------------------------------
    # Speaking observations.
    # --------------------------------------------------------

    observations = (
        load_observations(
            args.speaking
        )
    )

    observation_count = sum(
        len(values)
        for values
        in observations.values()
    )

    print(
        f"Face observations: "
        f"{observation_count}"
    )

    # --------------------------------------------------------
    # Diarization.
    # --------------------------------------------------------

    print()
    print(
        "Loading speaker diarization..."
    )

    speaker_data = load_json(
        args.speakers
    )

    speaker_segments = (
        collect_speaker_segments(
            speaker_data
        )
    )

    speakers = sorted(
        {
            str(
                segment["speaker"]
            )
            for segment
            in speaker_segments
        }
    )

    print(
        f"Speaker segments: "
        f"{len(speaker_segments)}"
    )

    print(
        f"Speakers: "
        f"{len(speakers)}"
    )

    # --------------------------------------------------------
    # Build local timeline.
    # --------------------------------------------------------

    print()
    print(
        "Resolving local speaker -> person -> face..."
    )

    raw_timeline = (
        build_local_timeline(
            timeline=speaking_timeline,
            speaker_segments=speaker_segments,
            face_to_person=face_to_person,
            association=association,
            observations=observations,
        )
    )

    # --------------------------------------------------------
    # Smooth identity switching.
    # --------------------------------------------------------

    print(
        "Applying temporal smoothing..."
    )

    smoothed_timeline = smooth_timeline(
        raw_timeline
    )

    # --------------------------------------------------------
    # Build segments.
    # --------------------------------------------------------

    segments = build_segments(
        smoothed_timeline
    )

    # --------------------------------------------------------
    # Stats.
    # --------------------------------------------------------

    identified_samples = sum(
        1
        for item
        in smoothed_timeline
        if item.get(
            "smoothed_person"
        ) is not None
    )

    stable_local_samples = sum(
        1
        for item
        in raw_timeline
        if item.get(
            "stable_local",
            False,
        )
    )

    print()
    print(
        f"Identified timeline samples: "
        f"{identified_samples}/"
        f"{len(smoothed_timeline)}"
    )

    print(
        f"Strong local matches: "
        f"{stable_local_samples}/"
        f"{len(raw_timeline)}"
    )

    print(
        f"Resolved speaking segments: "
        f"{len(segments)}"
    )

    print()
    print(
        "SPEAKER -> PERSON SEGMENTS"
    )

    for segment in segments:

        print(
            f"  "
            f"{segment['start']:.2f}s -> "
            f"{segment['end']:.2f}s | "
            f"{segment['speaker']} -> "
            f"{segment['person_id']} | "
            f"faces="
            f"{', '.join(segment['face_ids'])} | "
            f"confidence="
            f"{segment['mean_confidence']:.3f}"
        )

    # --------------------------------------------------------
    # Output.
    # --------------------------------------------------------

    output = {
        "version": 1,
        "mode": "speaker_person_face_timeline",
        "speaker_file": os.path.abspath(
            args.speakers
        ),
        "person_file": os.path.abspath(
            args.persons
        ),
        "speaking_file": os.path.abspath(
            args.speaking
        ),
        "association_file": os.path.abspath(
            args.association
        ),
        "speakers": speakers,
        "persons": sorted(
            person_faces.keys()
        ),
        "face_to_person": face_to_person,
        "person_faces": person_faces,
        "timeline": smoothed_timeline,
        "segments": segments,
        "stats": {
            "timeline_samples":
                len(smoothed_timeline),
            "identified_samples":
                identified_samples,
            "stable_local_samples":
                stable_local_samples,
            "resolved_segments":
                len(segments),
        },
        "thresholds": {
            "min_local_confidence":
                MIN_LOCAL_CONFIDENCE,
            "min_local_margin":
                MIN_LOCAL_MARGIN,
            "speaking_weight":
                SPEAKING_CONFIDENCE_WEIGHT,
            "association_weight":
                ASSOCIATION_CONFIDENCE_WEIGHT,
            "person_consistency_weight":
                PERSON_CONSISTENCY_WEIGHT,
            "min_switch_samples":
                MIN_SWITCH_SAMPLES,
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
