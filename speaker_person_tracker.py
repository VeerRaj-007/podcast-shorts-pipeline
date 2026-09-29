
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

# Only evidence above this confidence contributes strongly.
MIN_HYPOTHESIS_CONFIDENCE = 0.30

# Timeline evidence from speaking-face detector.
TIMELINE_WEIGHT = 0.65

# Face -> speaker association evidence.
ASSOCIATION_WEIGHT = 0.35

# Minimum amount of accumulated evidence before we consider
# a speaker/person relationship meaningful.
MIN_EVIDENCE = 1.00

# Final confidence required for a stable mapping.
STABLE_CONFIDENCE = 0.62

# Required gap between first and second candidate.
STABLE_MARGIN = 0.10

# Minimum number of independent face tracks supporting the mapping.
MIN_SUPPORTING_FACE_TRACKS = 1

# A very weak mapping should never become stable merely because
# it is the only candidate.
MIN_STABLE_DIRECT_EVIDENCE = 0.35


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

    for face_id, person_id in data.get(
        "face_to_person",
        {},
    ).items():

        face_to_person[
            str(face_id)
        ] = str(person_id)

    person_faces: Dict[
        str,
        List[str],
    ] = defaultdict(list)

    for person in data.get(
        "persons",
        [],
    ):

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

    # Ensure face_to_person is complete.
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
# Speaker-face association
# ============================================================

def load_association_hypotheses(
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

        confidence = float(
            confidence
        )

        if confidence < MIN_HYPOTHESIS_CONFIDENCE:
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
                "confidence": confidence,
            }
        )

    return output


# ============================================================
# Speaking-face timeline
# ============================================================

def load_speaking_timeline(
    path: str,
) -> List[Dict[str, Any]]:

    data = load_json(path)

    output = []

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

        if t is None:
            continue

        output.append(
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

    output.sort(
        key=lambda x:
            x["time"]
    )

    return output


# ============================================================
# Diarization speakers
# ============================================================

def collect_speaker_segments(
    data: Any,
) -> List[Dict[str, Any]]:

    result = []
    seen = set()

    def speaker_of(
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

            value = item.get(
                key
            )

            if value is not None:
                return str(
                    value
                )

        return inherited

    def times_of(
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

        speaker = speaker_of(
            node,
            inherited_speaker,
        )

        times = times_of(
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
# Diarization speaker set
# ============================================================

def speaker_names(
    segments: List[Dict[str, Any]],
) -> List[str]:

    return sorted(
        {
            str(
                segment["speaker"]
            )
            for segment in segments
        }
    )


# ============================================================
# Accumulator
# ============================================================

class EvidenceAccumulator:

    def __init__(self) -> None:

        self.score: Dict[
            str,
            Dict[str, float],
        ] = defaultdict(
            lambda:
                defaultdict(float)
        )

        self.face_support: Dict[
            str,
            Dict[str, set[str]],
        ] = defaultdict(
            lambda:
                defaultdict(set)
        )

        self.timeline_samples: Dict[
            str,
            Dict[str, int],
        ] = defaultdict(
            lambda:
                defaultdict(int)
        )

        self.association_support: Dict[
            str,
            Dict[str, int],
        ] = defaultdict(
            lambda:
                defaultdict(int)
        )

    def add_timeline_evidence(
        self,
        speaker: str,
        person: str,
        face_id: str,
        confidence: float,
    ) -> None:

        value = max(
            0.0,
            float(confidence),
        )

        self.score[
            speaker
        ][
            person
        ] += (
            TIMELINE_WEIGHT
            * value
        )

        self.face_support[
            speaker
        ][
            person
        ].add(
            face_id
        )

        self.timeline_samples[
            speaker
        ][
            person
        ] += 1

    def add_association_evidence(
        self,
        speaker: str,
        person: str,
        face_id: str,
        confidence: float,
    ) -> None:

        value = max(
            0.0,
            float(confidence),
        )

        self.score[
            speaker
        ][
            person
        ] += (
            ASSOCIATION_WEIGHT
            * value
        )

        self.face_support[
            speaker
        ][
            person
        ].add(
            face_id
        )

        self.association_support[
            speaker
        ][
            person
        ] += 1


# ============================================================
# Resolve
# ============================================================

def resolve(
    speakers: List[str],
    persons: List[str],
    face_to_person: Dict[str, str],
    associations: List[Dict[str, Any]],
    timeline: List[Dict[str, Any]],
) -> Dict[str, Any]:

    accumulator = EvidenceAccumulator()

    # --------------------------------------------------------
    # 1. Speaking timeline.
    #
    # The timeline tells us:
    #
    # speaker X was talking
    # active visual face = FACE_Y
    #
    # Convert FACE_Y -> PERSON_Z.
    # --------------------------------------------------------

    for item in timeline:

        speaker = item.get(
            "active_speaker"
        )

        face_id = item.get(
            "active_face"
        )

        if (
            speaker is None
            or face_id is None
        ):
            continue

        speaker = str(
            speaker
        )

        face_id = str(
            face_id
        )

        person = face_to_person.get(
            face_id
        )

        if person is None:
            continue

        confidence = float(
            item.get(
                "confidence",
                0.0,
            )
            or 0.0
        )

        accumulator.add_timeline_evidence(
            speaker,
            person,
            face_id,
            confidence,
        )

    # --------------------------------------------------------
    # 2. Association hypotheses.
    #
    # face -> speaker
    # then
    # face -> person
    # --------------------------------------------------------

    for item in associations:

        face_id = item[
            "face_id"
        ]

        speaker = item[
            "speaker"
        ]

        person = face_to_person.get(
            face_id
        )

        if person is None:
            continue

        confidence = float(
            item["confidence"]
        )

        accumulator.add_association_evidence(
            speaker,
            person,
            face_id,
            confidence,
        )

    # --------------------------------------------------------
    # 3. Build final mapping.
    # --------------------------------------------------------

    mappings = {}

    for speaker in speakers:

        candidate_scores = []

        for person in persons:

            score = float(
                accumulator.score[
                    speaker
                ].get(
                    person,
                    0.0,
                )
            )

            if score <= 0.0:
                continue

            supporting_faces = (
                accumulator.face_support[
                    speaker
                ].get(
                    person,
                    set(),
                )
            )

            timeline_count = (
                accumulator.timeline_samples[
                    speaker
                ].get(
                    person,
                    0,
                )
            )

            association_count = (
                accumulator.association_support[
                    speaker
                ].get(
                    person,
                    0,
                )
            )

            candidate_scores.append(
                {
                    "person_id": person,
                    "raw_score": score,
                    "supporting_faces": sorted(
                        supporting_faces
                    ),
                    "supporting_face_count": len(
                        supporting_faces
                    ),
                    "timeline_samples": timeline_count,
                    "association_hypotheses": association_count,
                }
            )

        candidate_scores.sort(
            key=lambda x:
                x["raw_score"],
            reverse=True,
        )

        if not candidate_scores:

            mappings[
                speaker
            ] = {
                "person_id": None,
                "candidate_person_id": None,
                "confidence": 0.0,
                "margin": 0.0,
                "stable": False,
                "reason": "no_evidence",
                "candidates": [],
            }

            continue

        total_score = sum(
            item["raw_score"]
            for item in candidate_scores
        )

        for item in candidate_scores:

            item[
                "normalized_confidence"
            ] = (
                item["raw_score"]
                / total_score
                if total_score > 0.0
                else 0.0
            )

        best = candidate_scores[0]

        second_score = (
            candidate_scores[1][
                "normalized_confidence"
            ]
            if len(candidate_scores) > 1
            else 0.0
        )

        best_confidence = float(
            best[
                "normalized_confidence"
            ]
        )

        margin = (
            best_confidence
            - second_score
        )

        # ----------------------------------------------------
        # Stability.
        # ----------------------------------------------------

        stable = (
            best_confidence
            >= STABLE_CONFIDENCE
            and margin
            >= STABLE_MARGIN
            and best[
                "supporting_face_count"
            ]
            >= MIN_SUPPORTING_FACE_TRACKS
            and best[
                "raw_score"
            ]
            >= MIN_STABLE_DIRECT_EVIDENCE
        )

        mappings[
            speaker
        ] = {
            "person_id": (
                best["person_id"]
                if stable
                else None
            ),
            "candidate_person_id":
                best["person_id"],
            "confidence": round(
                best_confidence,
                4,
            ),
            "margin": round(
                margin,
                4,
            ),
            "stable": stable,
            "supporting_faces":
                best[
                    "supporting_faces"
                ],
            "timeline_samples":
                best[
                    "timeline_samples"
                ],
            "association_hypotheses":
                best[
                    "association_hypotheses"
                ],
            "reason": (
                "stable"
                if stable
                else "insufficient_separation"
            ),
            "candidates": [
                {
                    **candidate,
                    "raw_score": round(
                        candidate[
                            "raw_score"
                        ],
                        4,
                    ),
                    "normalized_confidence":
                        round(
                            candidate[
                                "normalized_confidence"
                            ],
                            4,
                        ),
                }
                for candidate in candidate_scores[:10]
            ],
        }

    return {
        "speaker_person": mappings,
        "accumulated_scores": {
            speaker: dict(
                values
            )
            for speaker, values
            in accumulator.score.items()
        },
    }


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Aggregate speaker-person evidence "
            "over time into stable mappings."
        )
    )

    parser.add_argument(
        "--speakers",
        required=True,
        help="Speaker diarization JSON",
    )

    parser.add_argument(
        "--faces",
        required=True,
        help="Face identity JSON",
    )

    parser.add_argument(
        "--speaking",
        required=True,
        help="Speaking-face JSON",
    )

    parser.add_argument(
        "--association",
        required=True,
        help="Speaker-face association JSON",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output speaker-person map",
    )

    return parser.parse_args()


def main() -> int:

    args = parse_args()

    # --------------------------------------------------------
    # Load files.
    # --------------------------------------------------------

    print(
        "Loading person identities..."
    )

    face_identity_data = load_json(
        args.faces
    )

    (
        face_to_person,
        person_faces,
    ) = load_person_identity(
        args.faces
    )

    persons = sorted(
        person_faces.keys()
    )

    print(
        f"Persons: {len(persons)}"
    )

    print(
        f"Mapped faces: "
        f"{len(face_to_person)}"
    )

    print()
    print(
        "Loading speaker-face associations..."
    )

    associations = (
        load_association_hypotheses(
            args.association
        )
    )

    print(
        f"Usable association hypotheses: "
        f"{len(associations)}"
    )

    print()
    print(
        "Loading speaking timeline..."
    )

    timeline = (
        load_speaking_timeline(
            args.speaking
        )
    )

    print(
        f"Speaking timeline samples: "
        f"{len(timeline)}"
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

    speakers = speaker_names(
        speaker_segments
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
    # Resolve.
    # --------------------------------------------------------

    print()
    print(
        "Accumulating speaker-person evidence..."
    )

    result_data = resolve(
        speakers=speakers,
        persons=persons,
        face_to_person=face_to_person,
        associations=associations,
        timeline=timeline,
    )

    mappings = result_data[
        "speaker_person"
    ]

    print()
    print(
        "SPEAKER -> PERSON"
    )

    for speaker in speakers:

        mapping = mappings[
            speaker
        ]

        print(
            f"  {speaker} -> "
            f"{mapping['person_id']} "
            f"candidate="
            f"{mapping['candidate_person_id']} "
            f"confidence="
            f"{mapping['confidence']:.3f} "
            f"margin="
            f"{mapping['margin']:.3f} "
            f"stable="
            f"{mapping['stable']}"
        )

        if mapping.get(
            "supporting_faces"
        ):

            print(
                "      faces: "
                + ", ".join(
                    mapping[
                        "supporting_faces"
                    ]
                )
            )

    stable_count = sum(
        1
        for mapping in mappings.values()
        if mapping["stable"]
    )

    print()
    print(
        f"Stable speaker-person mappings: "
        f"{stable_count}/{len(speakers)}"
    )

    # --------------------------------------------------------
    # Output.
    # --------------------------------------------------------

    output = {
        "version": 1,
        "mode": "speaker_person_temporal_resolution",
        "speaker_file": os.path.abspath(
            args.speakers
        ),
        "face_identity_file": os.path.abspath(
            args.faces
        ),
        "speaking_file": os.path.abspath(
            args.speaking
        ),
        "association_file": os.path.abspath(
            args.association
        ),
        "speakers": speakers,
        "persons": persons,
        "face_to_person": face_to_person,
        "person_faces": person_faces,
        **result_data,
        "thresholds": {
            "min_hypothesis_confidence":
                MIN_HYPOTHESIS_CONFIDENCE,
            "timeline_weight":
                TIMELINE_WEIGHT,
            "association_weight":
                ASSOCIATION_WEIGHT,
            "stable_confidence":
                STABLE_CONFIDENCE,
            "stable_margin":
                STABLE_MARGIN,
            "min_supporting_face_tracks":
                MIN_SUPPORTING_FACE_TRACKS,
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
