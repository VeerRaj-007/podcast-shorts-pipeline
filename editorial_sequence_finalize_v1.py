#!/usr/bin/env python3
"""
Editorial Sequence Finalizer v1.1

Converts a strictly resolved AV plan into the exact format expected by
editorial_sequence_renderer_v2.py.

Important:
- Editorial source ranges are preserved exactly.
- Resolver render_segments are action-local.
- This finalizer converts action-local render timing into global output timing.
- No source ranges are changed.
- No source footage is reused.
- No unresolved actions are silently accepted.
- Duration is allowed to be whatever the editorial sequence selected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EPS = 1e-3


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_actions(
    data: dict[str, Any],
) -> list[dict[str, Any]]:
    actions = data.get("actions")

    if actions is None:
        actions = data.get(
            "timeline_actions",
            [],
        )

    if not isinstance(
        actions,
        list,
    ):
        return []

    return actions


def get_action_duration(
    action: dict[str, Any],
) -> float:

    duration = fnum(
        action.get("duration")
    )

    if duration > EPS:
        return duration

    start = fnum(
        action.get("start")
    )

    end = fnum(
        action.get("end")
    )

    return end - start


def convert_action(
    action: dict[str, Any],
    global_output_start: float,
) -> dict[str, Any]:
    """
    Convert one resolver action.

    Resolver render_segments are action-local:
        0.0 -> action duration

    Renderer needs global:
        global_start -> global_end
    """

    duration = get_action_duration(
        action
    )

    if duration <= EPS:
        raise ValueError(
            "Action has non-positive duration."
        )

    source_start = fnum(
        action.get(
            "source_start"
        )
    )

    source_end = fnum(
        action.get(
            "source_end"
        )
    )

    source_duration = (
        source_end
        - source_start
    )

    if abs(
        source_duration
        - duration
    ) > EPS:
        raise ValueError(
            "Editorial source duration does not "
            "match action duration: "
            f"{source_duration:.6f} vs "
            f"{duration:.6f}"
        )

    original_segments = action.get(
        "render_segments"
    )

    if (
        not isinstance(
            original_segments,
            list,
        )
        or not original_segments
    ):
        raise ValueError(
            "Action contains no render_segments."
        )

    converted_segments = []

    expected_local = 0.0

    for index, segment in enumerate(
        original_segments
    ):

        local_start = fnum(
            segment.get(
                "output_start"
            )
        )

        local_end = fnum(
            segment.get(
                "output_end"
            )
        )

        source_in = fnum(
            segment.get(
                "source_in"
            )
        )

        source_out = fnum(
            segment.get(
                "source_out"
            )
        )

        if abs(
            local_start
            - expected_local
        ) > EPS:
            raise ValueError(
                "Action render segments are not "
                "contiguous at segment "
                f"{index}: expected "
                f"{expected_local:.6f}, got "
                f"{local_start:.6f}"
            )

        segment_duration = (
            local_end
            - local_start
        )

        source_duration = (
            source_out
            - source_in
        )

        if segment_duration <= EPS:
            raise ValueError(
                f"Invalid render segment {index}."
            )

        if abs(
            segment_duration
            - source_duration
        ) > EPS:
            raise ValueError(
                "Render segment duration mismatch "
                f"at segment {index}: "
                f"output={segment_duration:.6f}, "
                f"source={source_duration:.6f}"
            )

        converted_segments.append(
            {
                "output_start": round(
                    global_output_start
                    + local_start,
                    6,
                ),
                "output_end": round(
                    global_output_start
                    + local_end,
                    6,
                ),
                "source_in": round(
                    source_in,
                    6,
                ),
                "source_out": round(
                    source_out,
                    6,
                ),
            }
        )

        expected_local = local_end

    if abs(
        expected_local
        - duration
    ) > EPS:
        raise ValueError(
            "Render segments do not cover the "
            f"whole action: "
            f"{expected_local:.6f} vs "
            f"{duration:.6f}"
        )

    output_action = {
        k: v
        for k, v in action.items()
        if k not in {
            "render_segments",
            "start",
            "end",
            "duration",
            "render_safe",
        }
    }

    output_action["start"] = round(
        global_output_start,
        6,
    )

    output_action["end"] = round(
        global_output_start + duration,
        6,
    )

    output_action["duration"] = round(
        duration,
        6,
    )

    output_action[
        "render_segments"
    ] = converted_segments

    output_action[
        "render_safe"
    ] = True

    return output_action


def validate_final_actions(
    actions: list[dict[str, Any]],
) -> float:

    expected_output = 0.0

    all_source_ranges = []

    for action_index, action in enumerate(
        actions
    ):

        start = fnum(
            action.get("start")
        )

        end = fnum(
            action.get("end")
        )

        duration = fnum(
            action.get("duration")
        )

        if abs(
            start
            - expected_output
        ) > EPS:
            raise ValueError(
                "Final output timeline gap/overlap "
                f"at action {action_index}: "
                f"expected {expected_output:.6f}, "
                f"got {start:.6f}"
            )

        if end <= start:
            raise ValueError(
                f"Invalid action {action_index}."
            )

        if abs(
            duration
            - (end - start)
        ) > EPS:
            raise ValueError(
                f"Duration mismatch at action "
                f"{action_index}."
            )

        segments = action.get(
            "render_segments"
        )

        if (
            not isinstance(
                segments,
                list,
            )
            or not segments
        ):
            raise ValueError(
                f"Action {action_index} has no "
                "render_segments."
            )

        expected_segment = start

        for segment in segments:

            output_start = fnum(
                segment.get(
                    "output_start"
                )
            )

            output_end = fnum(
                segment.get(
                    "output_end"
                )
            )

            source_in = fnum(
                segment.get(
                    "source_in"
                )
            )

            source_out = fnum(
                segment.get(
                    "source_out"
                )
            )

            if abs(
                output_start
                - expected_segment
            ) > EPS:
                raise ValueError(
                    "Render segment output gap/overlap "
                    f"at action {action_index}."
                )

            if output_end <= output_start:
                raise ValueError(
                    "Invalid output segment."
                )

            if source_out <= source_in:
                raise ValueError(
                    "Invalid source segment."
                )

            output_duration = (
                output_end
                - output_start
            )

            source_duration = (
                source_out
                - source_in
            )

            if abs(
                output_duration
                - source_duration
            ) > EPS:
                raise ValueError(
                    "Output/source duration mismatch."
                )

            all_source_ranges.append(
                (
                    source_in,
                    source_out,
                    action_index,
                )
            )

            expected_segment = output_end

        if abs(
            expected_segment
            - end
        ) > EPS:
            raise ValueError(
                f"Action {action_index} render "
                "segments do not end at action end."
            )

        expected_output = end

    # Check source reuse.
    for i, current in enumerate(
        all_source_ranges
    ):

        c0, c1, ca = current

        for previous in all_source_ranges[
            :i
        ]:

            p0, p1, pa = previous

            overlap = max(
                0.0,
                min(c1, p1)
                - max(c0, p0),
            )

            if overlap > EPS:
                raise ValueError(
                    "Source footage reused between "
                    f"actions {pa} and {ca}: "
                    f"{max(c0, p0):.6f} -> "
                    f"{min(c1, p1):.6f}"
                )

    return expected_output


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Finalize a strictly resolved "
            "editorial AV plan."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Strictly resolved AV sequence JSON",
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.0,
    )

    args = parser.parse_args()

    data = load_json(
        args.input
    )

    original_actions = get_actions(
        data
    )

    if not original_actions:
        raise ValueError(
            "Input contains no actions."
        )

    if data.get("render_safe") is not True:
        raise ValueError(
            "Input sequence is not "
            "render_safe=true."
        )

    if data.get(
        "unresolved_actions",
        0,
    ) != 0:
        raise ValueError(
            "Input contains unresolved actions."
        )

    finalized_actions = []

    global_output_cursor = 0.0

    for action in original_actions:

        finalized = convert_action(
            action,
            global_output_cursor,
        )

        finalized_actions.append(
            finalized
        )

        global_output_cursor = (
            finalized["end"]
        )

    final_duration = (
        validate_final_actions(
            finalized_actions
        )
    )

    if final_duration < args.min_duration:
        raise ValueError(
            f"Final duration "
            f"{final_duration:.3f}s is below "
            f"minimum "
            f"{args.min_duration:.3f}s."
        )

    requested_duration = fnum(
        data.get(
            "video_end",
            final_duration,
        )
    )

    output = {
        "version": "1.1",
        "mode": "editorial_sequence_finalized_short",
        "source_sequence": args.input,
        "video_start": 0.0,
        "video_end": round(
            final_duration,
            3,
        ),
        "requested_video_end": round(
            requested_duration,
            3,
        ),
        "duration_policy": {
            "target_is_maximum": True,
            "allowed_to_end_early": True,
            "min_duration_seconds": args.min_duration,
            "reason": (
                "Duration follows the editorially "
                "selected source ranges."
            ),
        },
        "render_safe": True,
        "unresolved_source_gaps": [],
        "source_duration_errors": [],
        "source_reuse": [],
        "backward_source_jumps": [],
        "output_timeline": [
            {
                "output_start": action["start"],
                "output_end": action["end"],
            }
            for action in finalized_actions
        ],
        "actions": finalized_actions,
        "stats": {
            "input_actions": len(
                original_actions
            ),
            "output_actions": len(
                finalized_actions
            ),
            "render_segments": sum(
                len(
                    action[
                        "render_segments"
                    ]
                )
                for action
                in finalized_actions
            ),
            "final_duration": round(
                final_duration,
                3,
            ),
            "requested_duration": round(
                requested_duration,
                3,
            ),
            "dropped_tail_seconds": round(
                max(
                    0.0,
                    requested_duration
                    - final_duration,
                ),
                3,
            ),
            "status": "finalized",
        },
    }

    save_json(
        args.output,
        output,
    )

    print(
        json.dumps(
            output["stats"],
            indent=2,
        )
    )

    print(
        "\nSUCCESS: finalized the exact "
        "editorial render plan."
    )


if __name__ == "__main__":
    main()