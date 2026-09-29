
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional


# ============================================================
# CONFIG
# ============================================================

# New face must remain the best candidate for at least this
# many consecutive timeline samples before switching.
MIN_SWITCH_SAMPLES = 3

# Or it must remain supported for at least this amount of time.
MIN_SWITCH_DURATION = 0.60

# Once a face is established, don't switch to a new face unless
# the new candidate has a meaningful score advantage.
SWITCH_SCORE_MARGIN = 0.08

# Keep the current face through tiny gaps.
MAX_GAP_DURATION = 0.50

# If the current face disappears for longer than this, allow
# the state to become unresolved.
DROP_CURRENT_AFTER = 1.25

# Very weak candidate should not start a switch.
MIN_CANDIDATE_SCORE = 0.45

# Strong candidate can switch slightly faster.
STRONG_SWITCH_SCORE = 0.72


# ============================================================
# JSON
# ============================================================

def load_json(path: str) -> Dict[str, Any]:
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


# ============================================================
# HELPERS
# ============================================================

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
# CANDIDATE LOOKUP
# ============================================================

def best_candidate(
    item: Dict[str, Any],
) -> Optional[Dict[str, Any]]:

    candidates = item.get(
        "candidates",
        [],
    )

    if not isinstance(
        candidates,
        list,
    ):
        return None

    usable = []

    for candidate in candidates:

        if not isinstance(
            candidate,
            dict,
        ):
            continue

        face_id = candidate.get(
            "face_id"
        )

        if face_id is None:
            continue

        score = clamp(
            to_float(
                candidate.get(
                    "score",
                    0.0,
                )
            )
        )

        usable.append(
            {
                **candidate,
                "face_id": str(
                    face_id
                ),
                "score": score,
            }
        )

    if not usable:
        return None

    usable.sort(
        key=lambda x:
            x["score"],
        reverse=True,
    )

    return usable[0]


def find_candidate(
    item: Dict[str, Any],
    face_id: Optional[str],
) -> Optional[Dict[str, Any]]:

    if face_id is None:
        return None

    for candidate in item.get(
        "candidates",
        [],
    ):

        if not isinstance(
            candidate,
            dict,
        ):
            continue

        candidate_face = candidate.get(
            "face_id"
        )

        if (
            candidate_face is not None
            and str(candidate_face)
            == face_id
        ):

            return {
                **candidate,
                "face_id":
                    str(candidate_face),
                "score":
                    clamp(
                        to_float(
                            candidate.get(
                                "score",
                                0.0,
                            )
                        )
                    ),
            }

    return None


# ============================================================
# SPEAKER STATE
# ============================================================

class SpeakerState:

    def __init__(self) -> None:

        self.current_face: Optional[str] = None

        self.current_score: float = 0.0

        self.last_seen_time: Optional[float] = None

        self.pending_face: Optional[str] = None

        self.pending_start: Optional[float] = None

        self.pending_samples: int = 0

        self.pending_best_score: float = 0.0


# ============================================================
# STABILIZER
# ============================================================

def should_switch(
    state: SpeakerState,
    new_face: str,
    new_score: float,
    current_candidate_score: float,
    current_time: float,
) -> bool:

    # No current identity: accept a good candidate.
    if state.current_face is None:

        return (
            new_score
            >= MIN_CANDIDATE_SCORE
        )

    # Same face = no switch required.
    if new_face == state.current_face:
        return False

    # New candidate is too weak.
    if new_score < MIN_CANDIDATE_SCORE:
        return False

    # Existing face is competitive.
    score_difference = (
        new_score
        - current_candidate_score
    )

    if (
        score_difference
        < SWITCH_SCORE_MARGIN
        and new_score
        < STRONG_SWITCH_SCORE
    ):
        return False

    # Need persistence.
    if state.pending_face != new_face:
        return False

    if (
        state.pending_samples
        >= MIN_SWITCH_SAMPLES
    ):
        return True

    if (
        state.pending_start is not None
        and (
            current_time
            - state.pending_start
        )
        >= MIN_SWITCH_DURATION
    ):
        return True

    return False


def stabilize_timeline(
    timeline: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    states: Dict[
        str,
        SpeakerState,
    ] = {}

    output = []

    for item in timeline:

        time_value = to_float(
            item.get("time")
        )

        speaker = item.get(
            "speaker"
        )

        if speaker is None:

            output.append(
                {
                    **item,
                    "stable_face_id":
                        None,
                    "stable_confidence":
                        0.0,
                    "stable_strength":
                        "none",
                    "switch":
                        False,
                }
            )

            continue

        speaker = str(
            speaker
        )

        if speaker not in states:
            states[speaker] = (
                SpeakerState()
            )

        state = states[
            speaker
        ]

        top = best_candidate(
            item
        )

        # ----------------------------------------------------
        # No candidate at this point.
        # ----------------------------------------------------

        if top is None:

            if (
                state.current_face
                is not None
                and state.last_seen_time
                is not None
                and (
                    time_value
                    - state.last_seen_time
                )
                <= MAX_GAP_DURATION
            ):

                output.append(
                    {
                        **item,
                        "stable_face_id":
                            state.current_face,
                        "stable_confidence":
                            round(
                                state.current_score,
                                4,
                            ),
                        "stable_strength":
                            "held",
                        "switch":
                            False,
                    }
                )

            else:

                if (
                    state.current_face
                    is not None
                    and state.last_seen_time
                    is not None
                    and (
                        time_value
                        - state.last_seen_time
                    )
                    > DROP_CURRENT_AFTER
                ):

                    state.current_face = None
                    state.current_score = 0.0

                output.append(
                    {
                        **item,
                        "stable_face_id":
                            None,
                        "stable_confidence":
                            0.0,
                        "stable_strength":
                            "unresolved",
                        "switch":
                            False,
                    }
                )

            continue

        top_face = str(
            top["face_id"]
        )

        top_score = clamp(
            top["score"]
        )

        current_candidate = (
            find_candidate(
                item,
                state.current_face,
            )
        )

        current_candidate_score = (
            to_float(
                current_candidate.get(
                    "score",
                    0.0,
                )
            )
            if current_candidate
            else 0.0
        )

        # ----------------------------------------------------
        # No current face.
        # ----------------------------------------------------

        if state.current_face is None:

            if (
                top_score
                >= MIN_CANDIDATE_SCORE
            ):

                state.current_face = (
                    top_face
                )

                state.current_score = (
                    top_score
                )

                state.last_seen_time = (
                    time_value
                )

                state.pending_face = None

                state.pending_start = None

                state.pending_samples = 0

                output.append(
                    {
                        **item,
                        "stable_face_id":
                            state.current_face,
                        "stable_confidence":
                            round(
                                state.current_score,
                                4,
                            ),
                        "stable_strength":
                            (
                                "strong"
                                if top_score
                                >= STRONG_SWITCH_SCORE
                                else "usable"
                            ),
                        "switch":
                            True,
                    }
                )

            else:

                output.append(
                    {
                        **item,
                        "stable_face_id":
                            None,
                        "stable_confidence":
                            0.0,
                        "stable_strength":
                            "unresolved",
                        "switch":
                            False,
                    }
                )

            continue

        # ----------------------------------------------------
        # Current face still visible.
        # ----------------------------------------------------

        if top_face == state.current_face:

            state.current_score = (
                top_score
            )

            state.last_seen_time = (
                time_value
            )

            state.pending_face = None
            state.pending_start = None
            state.pending_samples = 0
            state.pending_best_score = 0.0

            output.append(
                {
                    **item,
                    "stable_face_id":
                        state.current_face,
                    "stable_confidence":
                        round(
                            state.current_score,
                            4,
                        ),
                    "stable_strength":
                        (
                            "strong"
                            if top_score
                            >= STRONG_SWITCH_SCORE
                            else "usable"
                        ),
                    "switch":
                        False,
                }
            )

            continue

        # ----------------------------------------------------
        # New face candidate appeared.
        # ----------------------------------------------------

        if top_score >= MIN_CANDIDATE_SCORE:

            if state.pending_face != top_face:

                # Start persistence tracking.
                state.pending_face = (
                    top_face
                )

                state.pending_start = (
                    time_value
                )

                state.pending_samples = 1

                state.pending_best_score = (
                    top_score
                )

            else:

                state.pending_samples += 1

                state.pending_best_score = max(
                    state.pending_best_score,
                    top_score,
                )

        else:

            state.pending_face = None
            state.pending_start = None
            state.pending_samples = 0
            state.pending_best_score = 0.0

        switch = should_switch(
            state=state,
            new_face=top_face,
            new_score=top_score,
            current_candidate_score=
                current_candidate_score,
            current_time=time_value,
        )

        # ----------------------------------------------------
        # Commit the switch.
        # ----------------------------------------------------

        if switch:

            old_face = (
                state.current_face
            )

            state.current_face = (
                top_face
            )

            state.current_score = (
                top_score
            )

            state.last_seen_time = (
                time_value
            )

            state.pending_face = None
            state.pending_start = None
            state.pending_samples = 0
            state.pending_best_score = 0.0

            output.append(
                {
                    **item,
                    "stable_face_id":
                        state.current_face,
                    "stable_confidence":
                        round(
                            state.current_score,
                            4,
                        ),
                    "stable_strength":
                        (
                            "strong"
                            if top_score
                            >= STRONG_SWITCH_SCORE
                            else "usable"
                        ),
                    "switch":
                        True,
                    "switched_from":
                        old_face,
                }
            )

        else:

            state.last_seen_time = (
                time_value
                if (
                    current_candidate
                    is not None
                )
                else state.last_seen_time
            )

            output.append(
                {
                    **item,
                    "stable_face_id":
                        state.current_face,
                    "stable_confidence":
                        round(
                            current_candidate_score
                            if current_candidate
                            else state.current_score,
                            4,
                        ),
                    "stable_strength":
                        "held",
                    "switch":
                        False,
                    "pending_face":
                        state.pending_face,
                    "pending_samples":
                        state.pending_samples,
                }
            )

    return output


# ============================================================
# SEGMENTS
# ============================================================

def build_segments(
    timeline: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    segments = []

    if not timeline:
        return segments

    current = None

    for item in timeline:

        speaker = item.get(
            "speaker"
        )

        face_id = item.get(
            "stable_face_id"
        )

        identity = (
            speaker,
            face_id,
        )

        if current is None:

            current = {
                "start":
                    item["time"],
                "end":
                    item["time"],
                "speaker":
                    speaker,
                "face_id":
                    face_id,
                "confidence_sum":
                    0.0,
                "samples":
                    0,
                "switches":
                    0,
                "strengths":
                    [],
            }

        current_identity = (
            current["speaker"],
            current["face_id"],
        )

        if identity != current_identity:

            segments.append(
                current
            )

            current = {
                "start":
                    item["time"],
                "end":
                    item["time"],
                "speaker":
                    speaker,
                "face_id":
                    face_id,
                "confidence_sum":
                    0.0,
                "samples":
                    0,
                "switches":
                    0,
                "strengths":
                    [],
            }

        else:

            current["end"] = (
                item["time"]
            )

        current[
            "confidence_sum"
        ] += to_float(
            item.get(
                "stable_confidence",
                0.0,
            )
        )

        current[
            "samples"
        ] += 1

        if item.get(
            "switch",
            False,
        ):
            current[
                "switches"
            ] += 1

        current[
            "strengths"
        ].append(
            item.get(
                "stable_strength",
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

        if (
            "strong" in strengths
        ):
            strength = "strong"
        elif (
            "usable" in strengths
            or "held" in strengths
        ):
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
                "switches":
                    segment[
                        "switches"
                    ],
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
            "Temporally stabilize local "
            "speaker -> face assignments."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="local_speaker_face_resolver JSON",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="stabilized output JSON",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    print(
        "Loading local speaker-face resolution..."
    )

    data = load_json(
        args.input
    )

    timeline = data.get(
        "timeline",
        [],
    )

    print(
        f"Timeline samples: "
        f"{len(timeline)}"
    )

    print()

    print(
        "Applying temporal stabilization..."
    )

    stabilized = stabilize_timeline(
        timeline
    )

    stable = sum(
        1
        for item in stabilized
        if item.get(
            "stable_face_id"
        ) is not None
    )

    switches = sum(
        1
        for item in stabilized
        if item.get(
            "switch",
            False,
        )
    )

    print(
        f"Stable samples: "
        f"{stable}/"
        f"{len(stabilized)}"
    )

    print(
        f"Face switches: "
        f"{switches}"
    )

    print()

    segments = build_segments(
        stabilized
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
        f"Stable speaker-face segments: "
        f"{len(resolved_segments)}"
    )

    print()

    print(
        "STABLE SPEAKER -> FACE SEGMENTS"
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
        "version": 1,
        "mode":
            "speaker_face_temporal_stabilization",

        "input_file":
            os.path.abspath(
                args.input
            ),

        "config": {
            "min_switch_samples":
                MIN_SWITCH_SAMPLES,
            "min_switch_duration":
                MIN_SWITCH_DURATION,
            "switch_score_margin":
                SWITCH_SCORE_MARGIN,
            "max_gap_duration":
                MAX_GAP_DURATION,
            "drop_current_after":
                DROP_CURRENT_AFTER,
            "min_candidate_score":
                MIN_CANDIDATE_SCORE,
            "strong_switch_score":
                STRONG_SWITCH_SCORE,
        },

        "timeline":
            stabilized,

        "segments":
            segments,

        "resolved_segments":
            resolved_segments,

        "stats": {
            "input_samples":
                len(timeline),
            "stable_samples":
                stable,
            "resolution_rate":
                round(
                    stable
                    / len(timeline),
                    4,
                )
                if timeline
                else 0.0,
            "switches":
                switches,
            "stable_segments":
                len(
                    resolved_segments
                ),
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
