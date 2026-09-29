
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

SPEAKER_EVIDENCE_THRESHOLD = 0.35

# Evidence weights.
WEIGHT_FACE_SPEAKER = 0.55
WEIGHT_ACTIVE_TIMELINE = 0.30
WEIGHT_PERSON_COVERAGE = 0.15

# Minimum confidence for a final speaker -> person identity.
FINAL_CONFIDENCE_THRESHOLD = 0.45

# Difference required between best and second-best PERSON.
MIN_PERSON_MARGIN = 0.05


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

    face_to_person = data.get(
        "face_to_person",
        {},
    )

    persons_raw = data.get(
        "persons",
        [],
    )

    clean_face_to_person = {
        str(face): str(person)
        for face, person in face_to_person.items()
    }

    person_faces: Dict[
        str,
        List[str],
    ] = defaultdict(list)

    if isinstance(persons_raw, list):

        for person in persons_raw:

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

                face_id = str(
                    face_id
                )

                person_faces[
                    person_id
                ].append(
                    face_id
                )

    # Make sure every face_to_person entry is reflected.
    for face_id, person_id in (
        clean_face_to_person.items()
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
        faces.sort()

    return (
        clean_face_to_person,
        dict(person_faces),
    )


# ============================================================
# Speaker-face association
# ============================================================

def load_speaker_face_hypotheses(
    path: str,
) -> List[Dict[str, Any]]:

    data = load_json(path)

    output = []

    for item in data.get(
        "hypotheses",
        [],
    ):

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

        output.append(
            {
                **item,
                "face_id": str(
                    face_id
                ),
                "speaker": str(
                    speaker
                ),
                "confidence": float(
                    confidence
                ),
            }
        )

    return output


# ============================================================
# Speaking timeline
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
        key=lambda x:
            x["time"]
    )

    return timeline


# ============================================================
# Speaker diarization
# ============================================================

def collect_speaker_segments(
    data: Any,
) -> List[Dict[str, Any]]:

    result = []
    seen = set()

    def find_speaker(
        item: Dict[str, Any],
        inherited: Optional[str],
    ) -> Optional[str]:

        for key in (
            "speaker",
            "speaker_id",
            "speakerId",
            "label",
            "name",
        ):

            if item.get(key) is not None:
                return str(
                    item[key]
                )

        return inherited

    def find_times(
        item: Dict[str, Any],
    ) -> Optional[Tuple[float, float]]:

        start = as_float(
            item.get(
                "start",
                item.get(
                    "start_time",
                    item.get("begin"),
                ),
            )
        )

        end = as_float(
            item.get(
                "end",
                item.get(
                    "end_time",
                    item.get("stop"),
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
        inherited_speaker: Optional[str] = None,
    ) -> None:

        if isinstance(
            node,
            list,
        ):

            for item in node:
                walk(
                    item,
                    inherited_speaker,
                )

            return

        if not isinstance(
            node,
            dict,
        ):
            return

        speaker = find_speaker(
            node,
            inherited_speaker,
        )

        times = find_times(
            node
        )

        if (
            times is not None
            and speaker is not None
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
# Evidence construction
# ============================================================

def build_face_speaker_confidence(
    hypotheses: List[Dict[str, Any]],
) -> Dict[
    str,
    Dict[str, float],
]:

    result: Dict[
        str,
        Dict[str, float],
    ] = defaultdict(dict)

    for item in hypotheses:

        face_id = item["face_id"]
        speaker = item["speaker"]
        confidence = float(
            item["confidence"]
        )

        # Keep the strongest hypothesis for this
        # face/speaker pair.
        current = result[
            face_id
        ].get(
            speaker,
            0.0,
        )

        result[
            face_id
        ][
            speaker
        ] = max(
            current,
            confidence,
        )

    return dict(result)


def build_timeline_evidence(
    timeline: List[Dict[str, Any]],
    face_to_person: Dict[str, str],
) -> Dict[
    str,
    Dict[str, Dict[str, float]],
]:

    result: Dict[
        str,
        Dict[str, Dict[str, float]],
    ] = defaultdict(
        lambda:
            defaultdict(
                lambda: {
                    "active_count": 0.0,
                    "total_count": 0.0,
                    "confidence_sum": 0.0,
                }
            )
    )

    for item in timeline:

        speaker = item[
            "active_speaker"
        ]

        face_id = item[
            "active_face"
        ]

        if (
            speaker is None
            or face_id is None
        ):
            continue

        person = face_to_person.get(
            face_id
        )

        if person is None:
            continue

        bucket = result[
            str(speaker)
        ][
            str(person)
        ]

        bucket[
            "active_count"
        ] += 1.0

        bucket[
            "total_count"
        ] += 1.0

        bucket[
            "confidence_sum"
        ] += float(
            item["confidence"]
        )

    return {
        speaker: dict(
            persons
        )
        for speaker, persons
        in result.items()
    }


# ============================================================
# Main resolver
# ============================================================

def resolve(
    face_to_person: Dict[str, str],
    person_faces: Dict[str, List[str]],
    hypotheses: List[Dict[str, Any]],
    timeline: List[Dict[str, Any]],
    speaker_segments: List[Dict[str, Any]],
) -> Dict[str, Any]:

    face_speaker = (
        build_face_speaker_confidence(
            hypotheses
        )
    )

    timeline_evidence = (
        build_timeline_evidence(
            timeline,
            face_to_person,
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

    persons = sorted(
        person_faces.keys()
    )

    # --------------------------------------------------------
    # Face-speaker evidence aggregated to PERSON.
    # --------------------------------------------------------

    person_speaker_scores: Dict[
        str,
        Dict[str, float],
    ] = defaultdict(dict)

    for person_id in persons:

        faces = person_faces[
            person_id
        ]

        for speaker in speakers:

            values = []

            for face_id in faces:

                confidence = (
                    face_speaker
                    .get(
                        face_id,
                        {},
                    )
                    .get(
                        speaker,
                        0.0,
                    )
                )

                if confidence > 0:
                    values.append(
                        float(
                            confidence
                        )
                    )

            if values:

                # Average of strongest face evidence,
                # not simply maximum.
                values.sort(
                    reverse=True
                )

                top = values[
                    : min(3, len(values))
                ]

                person_speaker_scores[
                    person_id
                ][
                    speaker
                ] = float(
                    np.mean(top)
                )

            else:

                person_speaker_scores[
                    person_id
                ][
                    speaker
                ] = 0.0

    # --------------------------------------------------------
    # Add active timeline evidence.
    # --------------------------------------------------------

    final_scores: Dict[
        str,
        Dict[str, float],
    ] = defaultdict(dict)

    evidence_details: Dict[
        str,
        Dict[str, Dict[str, Any]],
    ] = defaultdict(dict)

    for speaker in speakers:

        for person_id in persons:

            face_speaker_score = (
                person_speaker_scores[
                    person_id
                ].get(
                    speaker,
                    0.0,
                )
            )

            timeline_bucket = (
                timeline_evidence
                .get(
                    speaker,
                    {},
                )
                .get(
                    person_id,
                    {},
                )
            )

            active_count = (
                timeline_bucket
                .get(
                    "active_count",
                    0.0,
                )
            )

            total_count = (
                timeline_bucket
                .get(
                    "total_count",
                    0.0,
                )
            )

            if total_count > 0:

                active_rate = (
                    active_count
                    / total_count
                )

                timeline_confidence = (
                    timeline_bucket[
                        "confidence_sum"
                    ]
                    / total_count
                )

            else:

                active_rate = 0.0
                timeline_confidence = 0.0

            # Person coverage is intentionally weak.
            face_count = len(
                person_faces[
                    person_id
                ]
            )

            coverage = min(
                1.0,
                face_count / 3.0,
            )

            score = (
                WEIGHT_FACE_SPEAKER
                * face_speaker_score
                +
                WEIGHT_ACTIVE_TIMELINE
                * active_rate
                +
                WEIGHT_PERSON_COVERAGE
                * coverage
            )

            final_scores[
                speaker
            ][
                person_id
            ] = float(
                np.clip(
                    score,
                    0.0,
                    1.0,
                )
            )

            evidence_details[
                speaker
            ][
                person_id
            ] = {
                "face_speaker_score":
                    round(
                        face_speaker_score,
                        4,
                    ),
                "active_timeline_rate":
                    round(
                        active_rate,
                        4,
                    ),
                "timeline_confidence":
                    round(
                        timeline_confidence,
                        4,
                    ),
                "person_track_count":
                    face_count,
                "score":
                    round(
                        float(score),
                        4,
                    ),
            }

    # --------------------------------------------------------
    # Best person per speaker.
    # --------------------------------------------------------

    speaker_person = {}

    for speaker in speakers:

        ranked = sorted(
            (
                {
                    "person_id": person_id,
                    "score": score,
                }
                for person_id, score
                in final_scores[
                    speaker
                ].items()
            ),
            key=lambda x:
                x["score"],
            reverse=True,
        )

        if not ranked:

            speaker_person[
                speaker
            ] = {
                "person_id": None,
                "confidence": 0.0,
                "margin": 0.0,
                "reliable": False,
            }

            continue

        best = ranked[0]

        second_score = (
            ranked[1]["score"]
            if len(ranked) > 1
            else 0.0
        )

        margin = (
            best["score"]
            - second_score
        )

        reliable = (
            best["score"]
            >= FINAL_CONFIDENCE_THRESHOLD
            and margin
            >= MIN_PERSON_MARGIN
        )

        speaker_person[
            speaker
        ] = {
            "person_id": (
                best["person_id"]
                if reliable
                else None
            ),
            "candidate_person_id":
                best["person_id"],
            "confidence": round(
                float(
                    best["score"]
                ),
                4,
            ),
            "margin": round(
                float(margin),
                4,
            ),
            "reliable": reliable,
        }

    # --------------------------------------------------------
    # Determine all visible faces belonging to a speaker.
    # --------------------------------------------------------

    for speaker, mapping in (
        speaker_person.items()
    ):

        if not mapping["reliable"]:
            continue

        person_id = mapping[
            "person_id"
        ]

        mapping[
            "face_ids"
        ] = list(
            person_faces.get(
                person_id,
                [],
            )
        )

    return {
        "speaker_person": speaker_person,
        "scores": {
            speaker: dict(
                values
            )
            for speaker, values
            in final_scores.items()
        },
        "evidence": {
            speaker: dict(
                values
            )
            for speaker, values
            in evidence_details.items()
        },
    }


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Resolve diarized speakers to persistent "
            "visual person identities."
        )
    )

    parser.add_argument(
        "--speakers",
        required=True,
        help="WhisperX speaker diarization JSON",
    )

    parser.add_argument(
        "--faces",
        required=True,
        help="Face tracking JSON",
    )

    parser.add_argument(
        "--speaking",
        required=True,
        help="Speaking-face detection JSON",
    )

    parser.add_argument(
        "--association",
        required=True,
        help="Speaker-face association JSON",
    )

    parser.add_argument(
        "--persons",
        required=True,
        help="Person identity JSON",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON",
    )

    return parser.parse_args()


def main() -> int:

    args = parse_args()

    print(
        "Loading person identities..."
    )

    person_data = load_json(
        args.persons
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

    print()
    print(
        "Loading speaker-face associations..."
    )

    hypotheses = (
        load_speaker_face_hypotheses(
            args.association
        )
    )

    print(
        f"Speaker-face hypotheses: "
        f"{len(hypotheses)}"
    )

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
        f"Speaking timeline samples: "
        f"{len(speaking_timeline)}"
    )

    print()
    print(
        "Loading diarization..."
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
                item["speaker"]
            )
            for item
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

    print()
    print(
        "Resolving speaker -> person..."
    )

    result_data = resolve(
        face_to_person=face_to_person,
        person_faces=person_faces,
        hypotheses=hypotheses,
        timeline=speaking_timeline,
        speaker_segments=speaker_segments,
    )

    print()
    print(
        "SPEAKER -> PERSON"
    )

    for speaker in speakers:

        mapping = result_data[
            "speaker_person"
        ][speaker]

        print(
            f"  {speaker} -> "
            f"{mapping.get('person_id')} "
            f"candidate="
            f"{mapping.get('candidate_person_id')} "
            f"confidence="
            f"{mapping.get('confidence', 0.0):.3f} "
            f"margin="
            f"{mapping.get('margin', 0.0):.3f} "
            f"reliable="
            f"{mapping.get('reliable', False)}"
        )

        if mapping.get(
            "face_ids"
        ):

            print(
                "      faces: "
                + ", ".join(
                    mapping["face_ids"]
                )
            )

    output = {
        "version": 1,
        "mode": "speaker_person_resolution",
        "speaker_file": os.path.abspath(
            args.speakers
        ),
        "face_file": os.path.abspath(
            args.faces
        ),
        "speaking_file": os.path.abspath(
            args.speaking
        ),
        "association_file": os.path.abspath(
            args.association
        ),
        "person_file": os.path.abspath(
            args.persons
        ),
        "speakers": speakers,
        "persons": sorted(
            person_faces.keys()
        ),
        "face_to_person": face_to_person,
        "person_faces": person_faces,
        **result_data,
        "thresholds": {
            "speaker_evidence":
                SPEAKER_EVIDENCE_THRESHOLD,
            "final_confidence":
                FINAL_CONFIDENCE_THRESHOLD,
            "person_margin":
                MIN_PERSON_MARGIN,
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
