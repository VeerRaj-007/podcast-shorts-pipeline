#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

EPS = 1e-6
COVERAGE_EPS = 1e-3


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


def normalize_shots(data):
    shots = (
        data.get("shots", [])
        if isinstance(data, dict)
        else data
    )

    if not isinstance(shots, list):
        raise ValueError(
            "Invalid shots JSON: expected a list or {\"shots\": [...]}"
        )

    normalized = []

    for shot in shots:
        start = float(shot["start"])
        end = float(shot["end"])

        if end <= start:
            continue

        normalized.append(
            {
                **shot,
                "start": start,
                "end": end,
            }
        )

    normalized.sort(
        key=lambda s: (
            s["start"],
            s["end"],
        )
    )

    return normalized


def resolve_exact_interval(
    start,
    end,
    shots,
):
    """
    Map the exact editorial source interval onto real source shots.

    IMPORTANT:
    This function NEVER changes the requested editorial start/end.
    It only decomposes that exact source interval across shot boundaries.
    """

    if end <= start + EPS:
        raise ValueError(
            f"Invalid editorial interval: {start} -> {end}"
        )

    relevant = []

    for shot in shots:
        shot_start = float(shot["start"])
        shot_end = float(shot["end"])

        overlap_start = max(
            start,
            shot_start,
        )

        overlap_end = min(
            end,
            shot_end,
        )

        if overlap_end <= overlap_start + EPS:
            continue

        relevant.append(
            {
                "shot_id": shot.get("shot_id"),
                "shot_type": shot.get("shot_type"),
                "source_in": overlap_start,
                "source_out": overlap_end,
                "duration": overlap_end - overlap_start,
            }
        )

    if not relevant:
        return None

    # Verify exact chronological coverage.
    cursor = start

    for segment in relevant:
        source_in = float(
            segment["source_in"]
        )
        source_out = float(
            segment["source_out"]
        )

        if source_in > cursor + COVERAGE_EPS:
            return None

        if source_in < cursor - COVERAGE_EPS:
            return None

        cursor = source_out

    if abs(cursor - end) > COVERAGE_EPS:
        return None

    # Local output timeline starts at zero for this Short.
    output_cursor = 0.0

    for segment in relevant:
        duration = float(
            segment["duration"]
        )

        segment["output_start"] = output_cursor
        segment["output_end"] = (
            output_cursor + duration
        )

        output_cursor += duration

    expected_duration = end - start

    if abs(
        output_cursor - expected_duration
    ) > COVERAGE_EPS:
        return None

    return relevant


def validate_action(action):
    start = float(action["start"])
    end = float(action["end"])

    segments = action.get(
        "render_segments",
        [],
    )

    if not segments:
        return False

    # Exact editorial boundaries.
    if abs(
        float(segments[0]["source_in"]) - start
    ) > COVERAGE_EPS:
        return False

    if abs(
        float(segments[-1]["source_out"]) - end
    ) > COVERAGE_EPS:
        return False

    # Chronological, gap-free source coverage.
    cursor = start

    for segment in segments:
        source_in = float(
            segment["source_in"]
        )
        source_out = float(
            segment["source_out"]
        )

        if abs(
            source_in - cursor
        ) > COVERAGE_EPS:
            return False

        if source_out <= source_in:
            return False

        cursor = source_out

    return abs(
        cursor - end
    ) <= COVERAGE_EPS


def validate_global_source_ranges(actions):
    """
    Validate each Short independently.

    The same source timestamp MAY legitimately appear in different
    final Shorts because each Short is an independent product output.

    We only forbid reuse within the same Short.
    """

    errors = []

    for action in actions:
        segments = action.get(
            "render_segments",
            [],
        )

        previous_end = None

        for index, segment in enumerate(segments):
            start = float(
                segment["source_in"]
            )
            end = float(
                segment["source_out"]
            )

            if previous_end is not None:
                if start < previous_end - COVERAGE_EPS:
                    errors.append(
                        {
                            "short_id": action.get("id"),
                            "segment_index": index,
                            "reason": "backward_or_overlapping_source",
                            "previous_end": previous_end,
                            "current_start": start,
                        }
                    )

            previous_end = end

    return errors


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Resolve exact Groq editorial intervals onto "
            "the real source shot timeline."
        )
    )

    parser.add_argument(
        "--sequence",
        required=True,
    )

    parser.add_argument(
        "--shots",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    sequence = load_json(
        args.sequence
    )

    shot_data = load_json(
        args.shots
    )

    actions = sequence.get(
        "actions",
        []
    )

    if not isinstance(actions, list):
        raise ValueError(
            "Sequence does not contain a valid actions list."
        )

    shots = normalize_shots(
        shot_data
    )

    resolved_actions = []
    unresolved = []

    # IMPORTANT:
    # Each Short starts with its own local output timeline.
    for action in actions:
        start = float(
            action["start"]
        )

        end = float(
            action["end"]
        )

        segments = resolve_exact_interval(
            start,
            end,
            shots,
        )

        if segments is None:
            unresolved.append(
                {
                    "id": action.get("id"),
                    "rank": action.get("rank"),
                    "start": start,
                    "end": end,
                    "reason": (
                        "Exact editorial source interval "
                        "cannot be completely covered by "
                        "the available source shots."
                    ),
                }
            )
            continue

        resolved = dict(action)

        resolved[
            "render_segments"
        ] = segments

        resolved[
            "editorial_source_match"
        ] = {
            "exact": True,
            "editorial_start": start,
            "editorial_end": end,
            "source_start": segments[0][
                "source_in"
            ],
            "source_end": segments[-1][
                "source_out"
            ],
        }

        resolved[
            "render_safe"
        ] = validate_action(
            resolved
        )

        resolved_actions.append(
            resolved
        )

    validation_errors = (
        validate_global_source_ranges(
            resolved_actions
        )
    )

    # Every resolved action must preserve its exact boundaries.
    boundary_errors = []

    for action in resolved_actions:
        if not action.get(
            "editorial_source_match",
            {}
        ).get("exact", False):
            boundary_errors.append(
                {
                    "id": action.get("id"),
                    "reason": (
                        "editorial_source_match.exact "
                        "is not true"
                    ),
                }
            )

        if not action.get(
            "render_safe",
            False
        ):
            boundary_errors.append(
                {
                    "id": action.get("id"),
                    "reason": (
                        "exact source coverage "
                        "validation failed"
                    ),
                }
            )

    render_safe = (
        not unresolved
        and not validation_errors
        and not boundary_errors
        and len(resolved_actions) == len(actions)
    )

    output = {
        "version": "5.0",
        "mode": (
            "exact_editorial_interval_source_mapping"
        ),
        "video_start": sequence.get(
            "video_start",
            0.0,
        ),
        "video_end": sequence.get(
            "video_end",
        ),

        "input_actions": len(actions),
        "resolved_actions": len(
            resolved_actions
        ),

        "actions": resolved_actions,

        "unresolved_actions": unresolved,

        "validation_errors": (
            validation_errors
        ),

        "boundary_errors": (
            boundary_errors
        ),

        "render_safe": render_safe,

        "policy": {
            "editorial_boundaries": (
                "immutable"
            ),
            "source_mapping": (
                "exact_source_interval_only"
            ),
            "silent_editorial_remapping": (
                "forbidden"
            ),
            "source_reuse_between_shorts": (
                "allowed"
            ),
            "source_reuse_within_short": (
                "forbidden"
            ),
            "freeze": False,
            "loop": False,
            "stretch": False,
        },

        "stats": {
            "input_actions": len(actions),
            "resolved_actions": len(
                resolved_actions
            ),
            "unresolved_actions": len(
                unresolved
            ),
            "validation_error_count": len(
                validation_errors
            ),
            "boundary_error_count": len(
                boundary_errors
            ),
            "render_segments": sum(
                len(
                    a.get(
                        "render_segments",
                        [],
                    )
                )
                for a in resolved_actions
            ),
        },
    }

    save_json(
        args.output,
        output,
    )

    print(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        )
    )

    if not render_safe:
        raise SystemExit(
            "ERROR: exact source mapping failed."
        )

    print()
    print(
        "SUCCESS: Resolver v5 preserved "
        "every editorial interval exactly."
    )


if __name__ == "__main__":
    main()