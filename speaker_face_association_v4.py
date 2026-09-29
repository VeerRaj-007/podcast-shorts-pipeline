from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SAMPLE_INTERVAL = 0.25

WEIGHT_ACTIVE_FACE = 0.55
WEIGHT_MOTION = 0.20
WEIGHT_TEMPORAL = 0.15
WEIGHT_EXCLUSIVITY = 0.10

MIN_CONFIDENCE = 0.35


# ---------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def overlap(
    a0: float,
    a1: float,
    b0: float,
    b1: float,
) -> float:
    return max(
        0.0,
        min(a1, b1) - max(a0, b0),
    )


# ---------------------------------------------------------------------
# Speaker extraction
# ---------------------------------------------------------------------

def collect_speaker_segments(
    data: Any,
) -> List[Dict[str, Any]]:

    result: List[Dict[str, Any]] = []
    seen = set()

    def get_speaker(
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
            value = item.get(key)

            if value is not None:
                return str(value)

        return inherited

    def get_times(
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

        if start is None or end is None:
            return None

        if end <= start:
            return None

        return start, end

    def walk(
        node: Any,
        inherited_speaker: Optional[str] = None,
    ) -> None:

        if isinstance(node, list):

            for item in node:
                walk(
                    item,
                    inherited_speaker,
                )

            return

        if not isinstance(node, dict):
            return

        speaker = get_speaker(
            node,
            inherited_speaker,
        )

        times = get_times(node)

        if times is not None and speaker is not None:

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


# ---------------------------------------------------------------------
# Face tracks
# ---------------------------------------------------------------------

def load_face_tracks(
    data: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:

    output: Dict[
        str,
        Dict[str, Any],
    ] = {}

    for track in data.get(
        "tracks",
        [],
    ):

        if not isinstance(track, dict):
            continue

        face_id = track.get(
            "face_id",
            track.get("id"),
        )

        if face_id is None:
            continue

        face_id = str(face_id)

        detections = []

        for detection in track.get(
            "detections",
            [],
        ):

            if not isinstance(
                detection,
                dict,
            ):
                continue

            t = as_float(
                detection.get("time")
            )

            if t is None:
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

        output[face_id] = {
            "face_id": face_id,
            "detections": detections,
            "start": detections[0]["time"],
            "end": detections[-1]["time"],
        }

    return output


# ---------------------------------------------------------------------
# Speaking-face data
# ---------------------------------------------------------------------

def load_speaking_data(
    data: Dict[str, Any],
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:

    timeline = data.get(
        "timeline",
        [],
    )

    observations = data.get(
        "observations",
        [],
    )

    clean_timeline = []
    clean_observations = []

    for item in timeline:

        if not isinstance(item, dict):
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

        clean_timeline.append(
            {
                "time": t,
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
                    item.get(
                        "confidence",
                        0.0,
                    )
                    or 0.0
                ),
            }
        )

    for item in observations:

        if not isinstance(item, dict):
            continue

        t = as_float(
            item.get("time")
        )

        face_id = item.get(
            "face_id"
        )

        if t is None or face_id is None:
            continue

        clean_observations.append(
            {
                "time": t,
                "face_id": str(face_id),
                "mouth_motion": float(
                    item.get(
                        "mouth_motion",
                        0.0,
                    )
                    or 0.0
                ),
            }
        )

    clean_timeline.sort(
        key=lambda x: x["time"]
    )

    clean_observations.sort(
        key=lambda x: x["time"]
    )

    return (
        clean_timeline,
        clean_observations,
    )


# ---------------------------------------------------------------------
# Speaker activity
# ---------------------------------------------------------------------

def speaker_at_time(
    t: float,
    speaker_segments: List[Dict[str, Any]],
) -> List[str]:

    active = []

    for seg in speaker_segments:

        if (
            seg["start"]
            <= t
            <= seg["end"]
        ):
            active.append(
                str(seg["speaker"])
            )

    return active


# ---------------------------------------------------------------------
# Motion normalization
# ---------------------------------------------------------------------

def normalize_motion(
    observations: List[Dict[str, Any]],
) -> None:

    values = [
        float(x["mouth_motion"])
        for x in observations
    ]

    if not values:
        return

    lo = float(
        np.percentile(
            values,
            20,
        )
    )

    hi = float(
        np.percentile(
            values,
            90,
        )
    )

    if hi <= lo:
        hi = lo + 1.0

    for item in observations:

        value = float(
            item["mouth_motion"]
        )

        item["mouth_motion_normalized"] = float(
            np.clip(
                (value - lo)
                / (hi - lo),
                0.0,
                1.0,
            )
        )


# ---------------------------------------------------------------------
# Main association
# ---------------------------------------------------------------------

def associate(
    speaker_segments: List[Dict[str, Any]],
    face_tracks: Dict[str, Dict[str, Any]],
    timeline: List[Dict[str, Any]],
    observations: List[Dict[str, Any]],
) -> Dict[str, Any]:

    speakers = sorted(
        {
            str(x["speaker"])
            for x in speaker_segments
        }
    )

    face_ids = sorted(
        face_tracks.keys()
    )

    normalize_motion(
        observations
    )

    # -------------------------------------------------------------
    # Motion index
    # -------------------------------------------------------------

    motion_by_face: Dict[
        str,
        List[float],
    ] = defaultdict(list)

    for obs in observations:

        motion_by_face[
            obs["face_id"]
        ].append(
            float(
                obs[
                    "mouth_motion_normalized"
                ]
            )
        )

    # -------------------------------------------------------------
    # Build actual speaker -> active face evidence.
    #
    # CRITICAL:
    #
    # We ONLY give speaking credit to the face that the
    # speaking-face detector selected.
    #
    # We do not give every visible face credit merely because
    # someone is talking.
    # -------------------------------------------------------------

    pair_active_count: Dict[
        Tuple[str, str],
        int,
    ] = defaultdict(int)

    pair_possible_count: Dict[
        Tuple[str, str],
        int,
    ] = defaultdict(int)

    pair_motion: Dict[
        Tuple[str, str],
        List[float],
    ] = defaultdict(list)

    face_active_by_speaker: Dict[
        str,
        Dict[str, int],
    ] = defaultdict(
        lambda: defaultdict(int)
    )

    total_speaker_samples: Dict[
        str,
        int,
    ] = defaultdict(int)

    for point in timeline:

        t = float(
            point["time"]
        )

        active_face = point[
            "active_face"
        ]

        active_speaker = point[
            "active_speaker"
        ]

        active_speakers = speaker_at_time(
            t,
            speaker_segments,
        )

        if active_speaker is not None:

            active_speaker = str(
                active_speaker
            )

        for speaker in active_speakers:

            total_speaker_samples[
                speaker
            ] += 1

            for face_id in face_ids:

                face_start = face_tracks[
                    face_id
                ]["start"]

                face_end = face_tracks[
                    face_id
                ]["end"]

                if not (
                    face_start
                    <= t
                    <= face_end
                ):
                    continue

                pair_possible_count[
                    (
                        face_id,
                        speaker,
                    )
                ] += 1

            # Only the selected active face gets positive
            # speaking-face credit.
            if (
                active_face is not None
                and active_speaker == speaker
            ):

                active_face = str(
                    active_face
                )

                if active_face in face_tracks:

                    pair_active_count[
                        (
                            active_face,
                            speaker,
                        )
                    ] += 1

                    face_active_by_speaker[
                        speaker
                    ][
                        active_face
                    ] += 1

    # -------------------------------------------------------------
    # Attach motion to each selected face/speaker pair.
    # -------------------------------------------------------------

    motion_lookup: Dict[
        Tuple[str, float],
        float,
    ] = {}

    for obs in observations:

        motion_lookup[
            (
                obs["face_id"],
                round(
                    float(
                        obs["time"]
                    ),
                    3,
                ),
            )
        ] = float(
            obs[
                "mouth_motion_normalized"
            ]
        )

    for point in timeline:

        active_face = point[
            "active_face"
        ]

        active_speaker = point[
            "active_speaker"
        ]

        if (
            active_face is None
            or active_speaker is None
        ):
            continue

        key_face = str(
            active_face
        )

        key_speaker = str(
            active_speaker
        )

        motion = motion_lookup.get(
            (
                key_face,
                round(
                    float(
                        point["time"]
                    ),
                    3,
                ),
            ),
            0.0,
        )

        pair_motion[
            (
                key_face,
                key_speaker,
            )
        ].append(
            motion
        )

    # -------------------------------------------------------------
    # Calculate metrics.
    # -------------------------------------------------------------

    hypotheses = []

    for face_id in face_ids:

        for speaker in speakers:

            active_count = pair_active_count[
                (
                    face_id,
                    speaker,
                )
            ]

            possible_count = pair_possible_count[
                (
                    face_id,
                    speaker,
                )
            ]

            speaker_count = total_speaker_samples[
                speaker
            ]

            if speaker_count > 0:

                active_face_rate = (
                    active_count
                    / speaker_count
                )

            else:

                active_face_rate = 0.0

            # How often this face was visible while the
            # speaker was talking.
            #
            # This is NOT treated as identity by itself.
            if possible_count > 0:

                temporal_visibility = (
                    active_count
                    / possible_count
                )

            else:

                temporal_visibility = 0.0

            motion_values = pair_motion[
                (
                    face_id,
                    speaker,
                )
            ]

            motion_score = (
                float(
                    np.mean(
                        motion_values
                    )
                )
                if motion_values
                else 0.0
            )

            face_scores = (
                face_active_by_speaker[
                    speaker
                ]
            )

            total_face_activity = sum(
                face_scores.values()
            )

            if total_face_activity > 0:

                exclusivity = (
                    face_scores.get(
                        face_id,
                        0,
                    )
                    / total_face_activity
                )

            else:

                exclusivity = 0.0

            confidence = (
                WEIGHT_ACTIVE_FACE
                * active_face_rate
                +
                WEIGHT_MOTION
                * motion_score
                +
                WEIGHT_TEMPORAL
                * temporal_visibility
                +
                WEIGHT_EXCLUSIVITY
                * exclusivity
            )

            hypotheses.append(
                {
                    "face_id": face_id,
                    "speaker": speaker,
                    "confidence": round(
                        float(
                            np.clip(
                                confidence,
                                0.0,
                                1.0,
                            )
                        ),
                        4,
                    ),
                    "active_face_rate": round(
                        float(
                            active_face_rate
                        ),
                        4,
                    ),
                    "motion_score": round(
                        float(
                            motion_score
                        ),
                        4,
                    ),
                    "temporal_visibility": round(
                        float(
                            temporal_visibility
                        ),
                        4,
                    ),
                    "exclusivity": round(
                        float(
                            exclusivity
                        ),
                        4,
                    ),
                    "active_face_samples": int(
                        active_count
                    ),
                    "speaker_samples": int(
                        speaker_count
                    ),
                    "visible_speaker_samples": int(
                        possible_count
                    ),
                }
            )

    # -------------------------------------------------------------
    # Rank.
    # -------------------------------------------------------------

    by_speaker: Dict[
        str,
        List[Dict[str, Any]],
    ] = defaultdict(list)

    by_face: Dict[
        str,
        List[Dict[str, Any]],
    ] = defaultdict(list)

    for hypothesis in hypotheses:

        by_speaker[
            hypothesis["speaker"]
        ].append(
            hypothesis
        )

        by_face[
            hypothesis["face_id"]
        ].append(
            hypothesis
        )

    for values in by_speaker.values():

        values.sort(
            key=lambda x: x[
                "confidence"
            ],
            reverse=True,
        )

    for values in by_face.values():

        values.sort(
            key=lambda x: x[
                "confidence"
            ],
            reverse=True,
        )

    # -------------------------------------------------------------
    # Best identity for each speaker.
    # -------------------------------------------------------------

    speaker_identity = {}

    for speaker in speakers:

        ranked = by_speaker.get(
            speaker,
            [],
        )

        if not ranked:

            speaker_identity[
                speaker
            ] = None

            continue

        best = ranked[0]

        second_score = (
            ranked[1]["confidence"]
            if len(ranked) > 1
            else 0.0
        )

        margin = (
            best["confidence"]
            - second_score
        )

        # Only call it a strong identity when there is
        # actual active-face evidence.
        reliable = (
            best["active_face_samples"] >= 2
            and best["confidence"]
            >= MIN_CONFIDENCE
            and margin >= 0.05
        )

        speaker_identity[
            speaker
        ] = {
            **best,
            "margin_over_second": round(
                float(margin),
                4,
            ),
            "reliable": reliable,
        }

    # -------------------------------------------------------------
    # Best speaker for each face.
    # -------------------------------------------------------------

    face_identity = {}

    for face_id in face_ids:

        ranked = by_face.get(
            face_id,
            [],
        )

        if not ranked:

            face_identity[
                face_id
            ] = None

            continue

        best = ranked[0]

        second_score = (
            ranked[1]["confidence"]
            if len(ranked) > 1
            else 0.0
        )

        margin = (
            best["confidence"]
            - second_score
        )

        face_identity[
            face_id
        ] = {
            **best,
            "margin_over_second": round(
                float(margin),
                4,
            ),
        }

    return {
        "hypotheses": hypotheses,
        "speaker_identity": speaker_identity,
        "face_identity": face_identity,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Associate diarized speakers with "
            "tracked faces using speaking-face "
            "timeline evidence."
        )
    )

    parser.add_argument(
        "--speaker",
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
        "--output",
        required=True,
    )

    return parser.parse_args()


def main() -> int:

    args = parse_args()

    speaker_data = load_json(
        args.speaker
    )

    face_data = load_json(
        args.faces
    )

    speaking_data = load_json(
        args.speaking
    )

    speaker_segments = (
        collect_speaker_segments(
            speaker_data
        )
    )

    face_tracks = load_face_tracks(
        face_data
    )

    timeline, observations = (
        load_speaking_data(
            speaking_data
        )
    )

    if not speaker_segments:
        raise RuntimeError(
            "No speaker segments found."
        )

    if not face_tracks:
        raise RuntimeError(
            "No face tracks found."
        )

    if not timeline:
        raise RuntimeError(
            "No speaking-face timeline found."
        )

    result_data = associate(
        speaker_segments=speaker_segments,
        face_tracks=face_tracks,
        timeline=timeline,
        observations=observations,
    )

    speakers = sorted(
        {
            str(x["speaker"])
            for x in speaker_segments
        }
    )

    result = {
        "version": 4,
        "mode": "speaker_face_association",
        "speaker_file": os.path.abspath(
            args.speaker
        ),
        "face_file": os.path.abspath(
            args.faces
        ),
        "speaking_face_file": os.path.abspath(
            args.speaking
        ),
        "speakers": speakers,
        "faces": sorted(
            face_tracks.keys()
        ),
        "speaker_segments": len(
            speaker_segments
        ),
        "speaking_timeline_samples": len(
            timeline
        ),
        "speaking_observations": len(
            observations
        ),
        "weights": {
            "active_face":
                WEIGHT_ACTIVE_FACE,
            "motion":
                WEIGHT_MOTION,
            "temporal":
                WEIGHT_TEMPORAL,
            "exclusivity":
                WEIGHT_EXCLUSIVITY,
        },
        **result_data,
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
            result,
            f,
            indent=2,
        )

    print(
        f"Speaker segments: "
        f"{len(speaker_segments)}"
    )

    print(
        f"Face tracks: "
        f"{len(face_tracks)}"
    )

    print(
        f"Speaking timeline samples: "
        f"{len(timeline)}"
    )

    print(
        f"Speaking observations: "
        f"{len(observations)}"
    )

    print()
    print(
        "BEST FACE PER SPEAKER"
    )

    for speaker in speakers:

        info = result[
            "speaker_identity"
        ].get(speaker)

        if info is None:

            print(
                f"  {speaker}: none"
            )

            continue

        print(
            f"  {speaker} -> "
            f"{info['face_id']} "
            f"confidence="
            f"{info['confidence']:.3f} "
            f"active="
            f"{info['active_face_rate']:.3f} "
            f"motion="
            f"{info['motion_score']:.3f} "
            f"margin="
            f"{info['margin_over_second']:.3f} "
            f"reliable="
            f"{info['reliable']}"
        )

    print()
    print(
        "TOP HYPOTHESES"
    )

    ranked = sorted(
        result["hypotheses"],
        key=lambda x: x[
            "confidence"
        ],
        reverse=True,
    )

    for item in ranked[:20]:

        print(
            f"  {item['face_id']} -> "
            f"{item['speaker']} "
            f"confidence="
            f"{item['confidence']:.3f} "
            f"active="
            f"{item['active_face_rate']:.3f} "
            f"motion="
            f"{item['motion_score']:.3f} "
            f"exclusive="
            f"{item['exclusivity']:.3f}"
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