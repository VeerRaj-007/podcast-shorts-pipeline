from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EPS = 0.001

MICRO_FLASH_MAX_SECONDS = 0.30
MICRO_FLASH_BACKWARD_JUMP_SECONDS = 5.0

# Two source ranges are considered part of the same continuous insert idea
# when the next source begins close to where the previous one ends.
CONTIGUOUS_INSERT_GAP_SECONDS = 0.35

# Do not create a new cut merely to make an action boundary disappear.
MIN_RESULT_SEGMENT_SECONDS = 0.35


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_actions(plan: dict[str, Any]) -> list[dict[str, Any]]:
    actions = plan.get("actions")
    if actions is None:
        actions = plan.get("timeline_actions", [])

    if not isinstance(actions, list) or not actions:
        raise ValueError("Plan contains no actions.")

    return actions


def action_kind(action: dict[str, Any]) -> str:
    kind = action.get("kind")
    return kind if isinstance(kind, str) and kind else "base"


def is_insert_action(action: dict[str, Any]) -> bool:
    return action_kind(action) == "insert"


def segment_duration(segment: dict[str, Any]) -> float:
    return float(segment["output_end"]) - float(segment["output_start"])


def source_duration(segment: dict[str, Any]) -> float:
    return float(segment["source_out"]) - float(segment["source_in"])


def is_micro_flash(segment: dict[str, Any]) -> bool:
    duration = segment_duration(segment)

    if duration >= MICRO_FLASH_MAX_SECONDS:
        return False

    if bool(segment.get("intentional_flash", False)):
        return False

    source_in = float(segment["source_in"])
    source_out = float(segment["source_out"])

    # The segment itself must be a meaningful source interval even though it
    # occupies very little output time.
    return source_out > source_in


def find_flash_context(
    actions: list[dict[str, Any]],
    action_index: int,
    segment_index: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    previous_segment = None
    next_segment = None

    # Previous output segment.
    if segment_index > 0:
        previous_segment = actions[action_index]["render_segments"][
            segment_index - 1
        ]
    elif action_index > 0:
        prev_segments = actions[action_index - 1].get(
            "render_segments", []
        )
        if prev_segments:
            previous_segment = prev_segments[-1]

    # Next output segment.
    current_segments = actions[action_index].get(
        "render_segments", []
    )
    if segment_index + 1 < len(current_segments):
        next_segment = current_segments[segment_index + 1]
    elif action_index + 1 < len(actions):
        next_segments = actions[action_index + 1].get(
            "render_segments", []
        )
        if next_segments:
            next_segment = next_segments[0]

    return previous_segment, next_segment


def should_remove_flash(
    flash: dict[str, Any],
    previous: dict[str, Any] | None,
    next_segment: dict[str, Any] | None,
) -> bool:
    if previous is None or next_segment is None:
        return False

    flash_source = float(flash["source_in"])
    prev_source_out = float(previous["source_out"])
    next_source_in = float(next_segment["source_in"])

    backward_jump = prev_source_out - flash_source

    if backward_jump < MICRO_FLASH_BACKWARD_JUMP_SECONDS:
        return False

    # The key pattern:
    #
    # local source ... [flash from far earlier] ... local source
    #
    # The source before and after the flash should themselves be reasonably
    # close, meaning the flash is sitting inside a coherent local sequence.
    local_gap = next_source_in - prev_source_out

    if local_gap < -EPS:
        # The following local segment itself goes backward, so this is not a
        # clean micro-flash pattern.
        return False

    return True


def remove_segment(
    actions: list[dict[str, Any]],
    action_index: int,
    segment_index: int,
) -> None:
    segments = actions[action_index]["render_segments"]
    segments.pop(segment_index)


def rebuild_action_bounds(action: dict[str, Any]) -> None:
    segments = action.get("render_segments", [])

    if not segments:
        action["start"] = action["end"] = action["duration"] = 0.0
        action["render_safe"] = False
        return

    action["start"] = float(segments[0]["output_start"])
    action["end"] = float(segments[-1]["output_end"])
    action["duration"] = action["end"] - action["start"]
    action["render_safe"] = True


def reflow_output_timeline(actions: list[dict[str, Any]]) -> None:
    """
    Rebuild output timestamps after edits while preserving each segment's
    duration exactly. This pass does NOT alter source_in/source_out.
    """
    cursor = 0.0

    for action in actions:
        rebuilt_segments = []

        for segment in action.get("render_segments", []):
            duration = source_duration(segment)

            if duration <= 0:
                continue

            segment["output_start"] = cursor
            segment["output_end"] = cursor + duration
            segment["duration"] = duration

            rebuilt_segments.append(segment)
            cursor += duration

        action["render_segments"] = rebuilt_segments
        rebuild_action_bounds(action)


def flatten_segments(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []

    for action_index, action in enumerate(actions):
        for segment_index, segment in enumerate(
            action.get("render_segments", [])
        ):
            item = segment
            item["_action_index"] = action_index
            item["_segment_index"] = segment_index
            result.append(item)

    return result


def collapse_empty_actions(
    actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []

    for action in actions:
        if action.get("render_segments"):
            result.append(action)

    return result


def merge_contiguous_insert_boundaries(
    actions: list[dict[str, Any]],
) -> int:
    """
    Merge adjacent insert actions only when their source sequences are
    naturally contiguous. This removes unnecessary semantic boundaries but
    does not remove actual visual shot changes inside the action.
    """
    merged = 0
    result = []

    for action in actions:
        if not result:
            result.append(action)
            continue

        previous = result[-1]

        if not (
            is_insert_action(previous)
            and is_insert_action(action)
        ):
            result.append(action)
            continue

        prev_segments = previous.get("render_segments", [])
        cur_segments = action.get("render_segments", [])

        if not prev_segments or not cur_segments:
            result.append(action)
            continue

        prev_source_end = float(prev_segments[-1]["source_out"])
        current_source_start = float(cur_segments[0]["source_in"])

        source_gap = current_source_start - prev_source_end

        if abs(source_gap) <= CONTIGUOUS_INSERT_GAP_SECONDS:
            previous["render_segments"].extend(cur_segments)
            rebuild_action_bounds(previous)

            previous["reason_types"] = list(dict.fromkeys(
                list(previous.get("reason_types", []))
                + list(action.get("reason_types", []))
            ))

            merged += 1
        else:
            result.append(action)

    actions[:] = result
    return merged


def repair_micro_flashes(
    actions: list[dict[str, Any]],
) -> tuple[int, list[dict[str, Any]]]:
    repairs = []

    # Work on a local copy of action structure.
    for ai, action in enumerate(actions):
        segments = action.get("render_segments", [])

        # Walk backwards so removals don't invalidate later indices.
        for si in range(len(segments) - 1, -1, -1):
            segment = segments[si]

            if not is_micro_flash(segment):
                continue

            previous, next_segment = find_flash_context(
                actions,
                ai,
                si,
            )

            if not should_remove_flash(
                segment,
                previous,
                next_segment,
            ):
                continue

            # Preserve the editorial decision but mark it removed.
            repairs.append({
                "type": "micro_flash_removed",
                "action_index": ai,
                "segment_index": si,
                "shot_id": segment.get("shot_id"),
                "source_in": float(segment["source_in"]),
                "source_out": float(segment["source_out"]),
                "duration": segment_duration(segment),
                "reason": (
                    "Short source jump went far backward and was surrounded "
                    "by a coherent local source sequence."
                ),
            })

            remove_segment(actions, ai, si)

    return len(repairs), repairs


def validate_source_uniqueness(actions: list[dict[str, Any]]) -> None:
    segments = flatten_segments(actions)

    normal = [
        s for s in segments
        if not bool(s.get("replay", False))
    ]

    normal.sort(key=lambda x: float(x["output_start"]))

    for i, a in enumerate(normal):
        a0 = float(a["source_in"])
        a1 = float(a["source_out"])

        for b in normal[i + 1:]:
            b0 = float(b["source_in"])
            b1 = float(b["source_out"])

            if b0 >= a1 - EPS:
                break

            overlap = min(a1, b1) - max(a0, b0)

            if overlap > 0.05:
                raise ValueError(
                    "Source reuse detected after repair."
                )


def validate_timeline(actions: list[dict[str, Any]]) -> float:
    if not actions:
        raise ValueError("Repair produced zero actions.")

    expected = 0.0

    for ai, action in enumerate(actions):
        start = float(action["start"])
        end = float(action["end"])

        if abs(start - expected) > EPS:
            raise ValueError(
                f"Timeline gap/overlap at action {ai}: "
                f"expected {expected:.6f}, got {start:.6f}"
            )

        if end <= start:
            raise ValueError(
                f"Invalid action duration at {ai}."
            )

        segment_expected = start

        for si, segment in enumerate(action.get(
            "render_segments", []
        )):
            seg_start = float(segment["output_start"])
            seg_end = float(segment["output_end"])

            if abs(seg_start - segment_expected) > EPS:
                raise ValueError(
                    f"Segment gap at {ai}:{si}."
                )

            if seg_end <= seg_start:
                raise ValueError(
                    f"Invalid segment at {ai}:{si}."
                )

            segment_expected = seg_end

        if abs(segment_expected - end) > EPS:
            raise ValueError(
                f"Action {ai} end does not match segments."
            )

        expected = end

    return expected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair contextual micro-flash insert cuts."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    plan = load_json(input_path)
    original_actions = get_actions(plan)

    # Deep copy through JSON so the input object is never mutated.
    actions = json.loads(json.dumps(original_actions))

    print("=== CONTEXTUAL INSERT REPAIR ===")
    print(f"Input actions: {len(actions)}")

    repair_count, repairs = repair_micro_flashes(actions)

    # Remove empty actions created by removing their only segment.
    actions = collapse_empty_actions(actions)

    # Re-merge legitimate contiguous insert action boundaries.
    merged_insert_boundaries = merge_contiguous_insert_boundaries(
        actions
    )

    # Rebuild the output timeline from real source durations.
    reflow_output_timeline(actions)

    final_duration = validate_timeline(actions)
    validate_source_uniqueness(actions)

    result = dict(plan)
    result["version"] = "1.0"
    result["mode"] = "contextual_insert_repair"
    result["actions"] = actions
    result["timeline_actions"] = actions
    result["render_safe"] = True

    result["repair_policy"] = {
        "micro_flash_max_seconds": MICRO_FLASH_MAX_SECONDS,
        "micro_flash_backward_jump_seconds": (
            MICRO_FLASH_BACKWARD_JUMP_SECONDS
        ),
        "contiguous_insert_gap_seconds": (
            CONTIGUOUS_INSERT_GAP_SECONDS
        ),
        "source_reuse_forbidden": True,
        "explicit_intentional_flash_preserved": True,
        "source_times_unchanged": True,
    }

    result["repair_log"] = repairs

    result["stats"] = {
        "input_actions": len(original_actions),
        "output_actions": len(actions),
        "micro_flashes_removed": repair_count,
        "contiguous_insert_boundaries_merged": merged_insert_boundaries,
        "render_segments": sum(
            len(action.get("render_segments", []))
            for action in actions
        ),
        "final_duration": final_duration,
    }

    save_json(output_path, result)

    print(json.dumps(result["stats"], indent=2))

    if repair_count:
        print(
            f"\nSUCCESS: removed {repair_count} accidental "
            "micro-flash segment(s)."
        )
    else:
        print(
            "\nSUCCESS: no qualifying micro-flash segments found."
        )


if __name__ == "__main__":
        main()
