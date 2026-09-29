#!/usr/bin/env python3
"""
Editra Visual Continuity / No-Op Collapse v2.

Collapses an editorial action boundary when the last rendered visual state of
one action is the same shot as the first rendered visual state of the next
action. This is less conservative than v1 while still avoiding speculative
merges.

No source timing is changed. Render segments remain untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EPS = 0.001


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


def boundary_signature(action: dict[str, Any]) -> tuple[str | None, str | None]:
    segments = action.get("render_segments", [])
    if not segments:
        raise ValueError("Action contains no render_segments.")
    last = segments[-1]
    return last.get("shot_id"), last.get("shot_type")


def first_signature(action: dict[str, Any]) -> tuple[str | None, str | None]:
    segments = action.get("render_segments", [])
    if not segments:
        raise ValueError("Action contains no render_segments.")
    first = segments[0]
    return first.get("shot_id"), first.get("shot_type")


def merge_lists(first: Any, second: Any) -> list[Any]:
    items: list[Any] = []
    if isinstance(first, list):
        items.extend(first)
    elif first is not None:
        items.append(first)
    if isinstance(second, list):
        items.extend(second)
    elif second is not None:
        items.append(second)

    out: list[Any] = []
    seen: set[str] = set()
    for item in items:
        key = json.dumps(item, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def merge_actions(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    merged = dict(first)
    merged["render_segments"] = (
        list(first.get("render_segments", []))
        + list(second.get("render_segments", []))
    )
    merged["start"] = float(first["start"])
    merged["end"] = float(second["end"])
    merged["duration"] = merged["end"] - merged["start"]
    merged["render_safe"] = True
    merged["reason_types"] = merge_lists(
        first.get("reason_types"), second.get("reason_types")
    )
    merged["merged_noop_actions"] = (
        int(first.get("merged_noop_actions", 0))
        + int(second.get("merged_noop_actions", 0))
        + 1
    )
    merged["merged_kinds"] = merge_lists(
        first.get("kind"), second.get("kind")
    )

    for key, value in second.items():
        if key in {
            "start", "end", "duration", "render_segments", "render_safe",
            "reason_types", "merged_noop_actions", "merged_kinds"
        }:
            continue
        if key not in merged:
            merged[key] = value

    return merged


def should_collapse(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    if previous.get("replay", False) or current.get("replay", False):
        return False

    if abs(float(previous["end"]) - float(current["start"])) > EPS:
        return False

    prev_shot, prev_type = boundary_signature(previous)
    curr_shot, curr_type = first_signature(current)

    # Same shot_id is a definite visual-source match. We also accept a missing
    # shot type on one side because shot_type classification can be imperfect.
    if prev_shot is None or curr_shot is None:
        return False

    if prev_shot != curr_shot:
        return False

    if prev_type is not None and curr_type is not None and prev_type != curr_type:
        return False

    return True


def collapse_actions(actions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    result: list[dict[str, Any]] = []
    collapsed = 0

    for action in actions:
        if not action.get("render_segments"):
            raise ValueError("Action contains no render_segments.")

        if not result:
            result.append(dict(action))
            continue

        previous = result[-1]
        if should_collapse(previous, action):
            result[-1] = merge_actions(previous, action)
            collapsed += 1
        else:
            result.append(dict(action))

    return result, collapsed


def validate(actions: list[dict[str, Any]]) -> None:
    if not actions:
        raise ValueError("Output plan is empty.")

    expected = 0.0
    for ai, action in enumerate(actions):
        start = float(action["start"])
        end = float(action["end"])
        if abs(start - expected) > EPS:
            raise ValueError(f"Action gap/overlap at {ai}.")
        if end <= start:
            raise ValueError(f"Invalid action {ai}.")

        seg_expected = start
        for si, segment in enumerate(action.get("render_segments", [])):
            s0 = float(segment["output_start"])
            s1 = float(segment["output_end"])
            if abs(s0 - seg_expected) > EPS:
                raise ValueError(f"Segment gap/overlap at {ai}:{si}.")
            if s1 <= s0:
                raise ValueError(f"Invalid segment at {ai}:{si}.")
            seg_expected = s1

        if abs(seg_expected - end) > EPS:
            raise ValueError(f"Action {ai} end mismatch.")
        expected = end


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    plan = load_json(input_path)
    actions = get_actions(plan)

    print("=== VISUAL CONTINUITY V2 ===")
    print(f"Input actions: {len(actions)}")

    output_actions, collapsed = collapse_actions(actions)
    validate(output_actions)

    input_duration = float(actions[-1]["end"])
    output_duration = float(output_actions[-1]["end"])
    if abs(input_duration - output_duration) > EPS:
        raise ValueError("Visual continuity pass changed total duration.")

    result = dict(plan)
    result["version"] = "2.0"
    result["mode"] = "editorial_sequence_visual_continuity"
    result["actions"] = output_actions
    result["timeline_actions"] = output_actions
    result["output_timeline"] = [
        {"output_start": float(a["start"]), "output_end": float(a["end"])}
        for a in output_actions
    ]
    result["render_safe"] = True
    result["visual_continuity_policy"] = {
        "collapse_same_boundary_shot": True,
        "same_shot_id_required": True,
        "shot_type_mismatch_rejected_when_both_known": True,
        "explicit_replays_preserved": True,
        "source_times_unchanged": True,
        "render_segments_unchanged": True,
        "duration_unchanged": True,
    }
    result["stats"] = {
        "input_actions": len(actions),
        "output_actions": len(output_actions),
        "collapsed_noop_actions": collapsed,
        "render_segments": sum(len(a["render_segments"]) for a in output_actions),
        "duration": output_duration,
    }

    save_json(output_path, result)
    print(json.dumps(result["stats"], indent=2))
    print("\nSUCCESS: visual boundary continuity pass completed.")


if __name__ == "__main__":
    main()
