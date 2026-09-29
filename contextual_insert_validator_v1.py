from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EPS = 0.001

MIN_HOLD = {
    "base": 0.70,
    "speaker": 0.55,
    "reaction": 0.45,
    "framing": 0.35,
    "insert": 0.40,
}

# Inserts are allowed to jump to earlier source material, but very large jumps
# are suspicious when the insert is not clearly motivated by the surrounding
# editorial context.
INSERT_BACKWARD_RISK_SECONDS = 12.0

# Two consecutive inserts usually need a very strong reason. We flag them
# rather than deleting them automatically.
CONSECUTIVE_INSERT_MAX_GAP = 0.20

# Very short total action changes can create "flash cuts".
RAPID_CUT_WINDOW_SECONDS = 1.20
RAPID_CUT_MAX_CHANGES = 3


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


def action_duration(action: dict[str, Any]) -> float:
    return float(action["end"]) - float(action["start"])


def get_kind(action: dict[str, Any]) -> str:
    kind = action.get("kind")
    if isinstance(kind, str) and kind:
        return kind
    return "base"


def get_segments(action: dict[str, Any]) -> list[dict[str, Any]]:
    segments = action.get("render_segments", [])
    if not isinstance(segments, list):
        raise ValueError("Action render_segments must be a list.")
    return segments


def first_source(action: dict[str, Any]) -> float | None:
    segments = get_segments(action)
    if not segments:
        return None
    return float(segments[0]["source_in"])


def last_source(action: dict[str, Any]) -> float | None:
    segments = get_segments(action)
    if not segments:
        return None
    return float(segments[-1]["source_out"])


def is_insert(action: dict[str, Any]) -> bool:
    return get_kind(action) == "insert"


def action_signature(action: dict[str, Any]) -> tuple:
    """
    Source-aware visual signature.

    We include shot identity and shot type, but not source timestamps so that
    the same visual shot remains recognizable even when the selected interval
    changes slightly.
    """
    signature = []

    for segment in get_segments(action):
        signature.append(
            (
                segment.get("shot_id"),
                segment.get("shot_type"),
            )
        )

    return tuple(signature)


def validate_holds(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues = []

    for index, action in enumerate(actions):
        duration = action_duration(action)
        kind = get_kind(action)
        minimum = MIN_HOLD.get(kind, 0.50)

        # An action made of multiple segments can legitimately be short if the
        # visual state is changing inside it. We therefore only flag it when
        # the whole action is below the minimum.
        if duration + EPS < minimum:
            issues.append({
                "type": "short_hold",
                "severity": "high" if duration < minimum * 0.65 else "medium",
                "action_index": index,
                "kind": kind,
                "duration": duration,
                "minimum": minimum,
                "reason": "Action is too short to read comfortably.",
            })

    return issues


def validate_rapid_changes(
    actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    issues = []

    # Sliding window over output actions.
    for start_index in range(len(actions)):
        start_time = float(actions[start_index]["start"])

        count = 0
        end_index = start_index

        for index in range(start_index, len(actions)):
            end_time = float(actions[index]["end"])

            if end_time - start_time <= RAPID_CUT_WINDOW_SECONDS + EPS:
                count += 1
                end_index = index
            else:
                break

        if count >= RAPID_CUT_MAX_CHANGES:
            issues.append({
                "type": "rapid_cut_cluster",
                "severity": "high",
                "start_action": start_index,
                "end_action": end_index,
                "action_count": count,
                "window_seconds": RAPID_CUT_WINDOW_SECONDS,
                "reason": (
                    "Too many editorial changes occur in a short output "
                    "window; likely to feel like a montage/flash-cut sequence."
                ),
            })

    return issues


def validate_insert_context(
    actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    issues = []

    for index, action in enumerate(actions):
        if not is_insert(action):
            continue

        current_source = first_source(action)
        if current_source is None:
            continue

        # Look at the immediately preceding output action.
        if index > 0:
            previous = actions[index - 1]
            previous_source = last_source(previous)

            if previous_source is not None:
                backward_jump = previous_source - current_source

                if backward_jump > INSERT_BACKWARD_RISK_SECONDS:
                    issues.append({
                        "type": "large_insert_backward_jump",
                        "severity": "medium",
                        "action_index": index,
                        "source_jump_seconds": backward_jump,
                        "previous_action_kind": get_kind(previous),
                        "reason": (
                            "Insert jumps far backward in source chronology. "
                            "This is allowed, but requires strong semantic "
                            "motivation to avoid feeling like unrelated B-roll."
                        ),
                    })

        # Consecutive inserts need stronger context than one insert.
        if index > 0 and is_insert(actions[index - 1]):
            gap = (
                float(action["start"])
                - float(actions[index - 1]["end"])
            )

            if gap <= CONSECUTIVE_INSERT_MAX_GAP + EPS:
                issues.append({
                    "type": "consecutive_inserts",
                    "severity": "high",
                    "action_index": index - 1,
                    "next_action_index": index,
                    "gap_seconds": gap,
                    "reason": (
                        "Consecutive inserts are likely to feel like a "
                        "montage unless both are strongly context-motivated."
                    ),
                })

    return issues


def validate_visual_boundaries(
    actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    issues = []

    for index in range(1, len(actions)):
        previous = actions[index - 1]
        current = actions[index]

        previous_signature = action_signature(previous)
        current_signature = action_signature(current)

        if previous_signature and current_signature:
            if previous_signature == current_signature:
                issues.append({
                    "type": "redundant_visual_boundary",
                    "severity": "high",
                    "action_index": index,
                    "previous_action_index": index - 1,
                    "reason": (
                        "Adjacent actions have the same resolved visual "
                        "signature. There may be no meaningful visual change."
                    ),
                })

    return issues


def validate_source_continuity(
    actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Dialogue/base/speaker/reaction actions should normally move forward.
    Inserts are excluded because they are explicitly allowed to use earlier
    source footage.
    """
    issues = []

    previous_source = None

    for index, action in enumerate(actions):
        if is_insert(action):
            continue

        current_source = first_source(action)

        if current_source is None:
            continue

        if previous_source is not None and current_source < previous_source - 0.05:
            issues.append({
                "type": "backward_dialogue_jump",
                "severity": "high",
                "action_index": index,
                "previous_source": previous_source,
                "current_source": current_source,
                "reason": (
                    "Non-insert editorial footage moved backward in source "
                    "time."
                ),
            })

        previous_source = current_source

    return issues


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate contextual inserts and shot-hold quality."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    plan = load_json(input_path)
    actions = get_actions(plan)

    print("=== CONTEXTUAL EDIT VALIDATION ===")
    print(f"Actions: {len(actions)}")
    print(f"Duration: {action_duration(actions[-1]):.3f}s")

    issues = []
    issues.extend(validate_holds(actions))
    issues.extend(validate_rapid_changes(actions))
    issues.extend(validate_insert_context(actions))
    issues.extend(validate_visual_boundaries(actions))
    issues.extend(validate_source_continuity(actions))

    high = sum(1 for issue in issues if issue["severity"] == "high")
    medium = sum(1 for issue in issues if issue["severity"] == "medium")

    report = {
        "version": "1.0",
        "validator": "contextual_insert_and_shot_hold_validator",
        "input": str(input_path),
        "passed": high == 0,
        "policy": {
            "insert_backward_jumps_allowed": True,
            "insert_backward_risk_seconds": INSERT_BACKWARD_RISK_SECONDS,
            "consecutive_insert_gap_seconds": CONSECUTIVE_INSERT_MAX_GAP,
            "rapid_cut_window_seconds": RAPID_CUT_WINDOW_SECONDS,
            "rapid_cut_max_changes": RAPID_CUT_MAX_CHANGES,
            "source_continuity_for_non_inserts": "monotonic",
        },
        "stats": {
            "actions": len(actions),
            "issues": len(issues),
            "high_severity": high,
            "medium_severity": medium,
            "duration": action_duration(actions[-1]),
        },
        "issues": issues,
    }

    save_json(output_path, report)

    print(json.dumps(report["stats"], indent=2))
    print("\n" + ("VALIDATION PASSED." if report["passed"] else
                    "VALIDATION FAILED: editorial issues require review."))


if __name__ == "__main__":
    main()