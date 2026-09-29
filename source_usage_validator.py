import argparse
import json
import sys
from pathlib import Path

EPS = 0.001
REUSE_OVERLAP_RATIO = 0.60
DUPLICATE_GAP = 0.050


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_actions(sequence):
    actions = sequence.get("actions")
    if actions is None:
        actions = sequence.get("timeline_actions", [])
    if not isinstance(actions, list):
        raise ValueError("Sequence actions must be a list.")
    return sorted(actions, key=lambda x: float(x.get("start", 0.0)))


def flatten_segments(actions):
    segments = []
    for action_index, action in enumerate(actions):
        render_segments = action.get("render_segments", [])
        for segment_index, seg in enumerate(render_segments):
            item = dict(seg)
            item["_action_index"] = action_index
            item["_segment_index"] = segment_index
            item["_action_start"] = float(action.get("start", 0.0))
            item["_action_end"] = float(action.get("end", 0.0))
            item["_action_kind"] = action.get("kind", "unknown")
            item["_action_shot_id"] = action.get("shot_id")
            segments.append(item)
    segments.sort(key=lambda x: float(x.get("output_start", 0.0)))
    return segments


def interval_overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def overlap_ratio(current_start, current_end, old_start, old_end):
    current_len = max(0.0, current_end - current_start)
    old_len = max(0.0, old_end - old_start)
    overlap = interval_overlap(current_start, current_end, old_start, old_end)
    denom = min(current_len, old_len)
    if denom <= EPS:
        return 0.0
    return overlap / denom


def explicitly_allowed_reuse(segment):
    kind = str(segment.get("kind", segment.get("_action_kind", ""))).lower()
    reason = str(segment.get("reason", "")).lower()
    return (
        kind in {"replay", "flashback", "callback"}
        or bool(segment.get("allow_source_reuse", False))
        or "intentional replay" in reason
        or "intentional reuse" in reason
    )


def is_adjacent_continuation(prev, cur):
    prev_in = float(prev["source_in"])
    prev_out = float(prev["source_out"])
    cur_in = float(cur["source_in"])
    cur_out = float(cur["source_out"])
    prev_output_end = float(prev["output_end"])
    cur_output_start = float(cur["output_start"])

    return (
        abs(cur_in - prev_out) <= DUPLICATE_GAP
        and abs(cur_output_start - prev_output_end) <= DUPLICATE_GAP
        and cur_out > cur_in + EPS
    )


def validate(sequence):
    actions = get_actions(sequence)
    segments = flatten_segments(actions)

    issues = []
    source_reuse = []
    backward_jumps = []
    output_gaps = []

    expected_output = 0.0
    previous = None
    prior_segments = []

    for i, seg in enumerate(segments):
        try:
            output_start = float(seg["output_start"])
            output_end = float(seg["output_end"])
            source_in = float(seg["source_in"])
            source_out = float(seg["source_out"])
        except (KeyError, TypeError, ValueError) as exc:
            issues.append({
                "type": "invalid_segment",
                "segment_index": i,
                "error": str(exc),
            })
            continue

        if abs(output_start - expected_output) > EPS:
            output_gaps.append({
                "segment_index": i,
                "expected_output_start": expected_output,
                "actual_output_start": output_start,
                "error_seconds": output_start - expected_output,
            })

        if source_out <= source_in + EPS:
            issues.append({
                "type": "invalid_source_range",
                "segment_index": i,
                "source_in": source_in,
                "source_out": source_out,
            })

        if previous is not None:
            prev_source_in = float(previous["source_in"])
            prev_source_out = float(previous["source_out"])

            backward = source_in < prev_source_in - EPS
            if backward and not explicitly_allowed_reuse(seg):
                backward_jumps.append({
                    "segment_index": i,
                    "previous_segment_index": i - 1,
                    "previous_source": [prev_source_in, prev_source_out],
                    "current_source": [source_in, source_out],
                    "output_range": [output_start, output_end],
                    "action_kind": seg.get("_action_kind"),
                    "shot_id": seg.get("_action_shot_id"),
                })

        if not explicitly_allowed_reuse(seg):
            for old in prior_segments:
                old_in = float(old["source_in"])
                old_out = float(old["source_out"])
                ratio = overlap_ratio(source_in, source_out, old_in, old_out)

                if ratio >= REUSE_OVERLAP_RATIO:
                    contiguous = (
                        previous is old
                        and is_adjacent_continuation(old, seg)
                    )
                    if not contiguous:
                        source_reuse.append({
                            "current_segment_index": i,
                            "previous_segment_index": old["_flat_index"],
                            "overlap_ratio": ratio,
                            "current_source": [source_in, source_out],
                            "previous_source": [old_in, old_out],
                            "current_output": [output_start, output_end],
                            "previous_output": [
                                float(old["output_start"]),
                                float(old["output_end"]),
                            ],
                            "current_kind": seg.get("_action_kind"),
                            "previous_kind": old.get("_action_kind"),
                            "current_shot_id": seg.get("_action_shot_id"),
                            "previous_shot_id": old.get("_action_shot_id"),
                        })
                        break

        seg["_flat_index"] = i
        prior_segments.append(seg)
        previous = seg
        expected_output = output_end

    source_usage_policy = "unique_source_moment_once"

    passed = not issues and not output_gaps and not source_reuse

    return {
        "version": "1.0",
        "validator": "source_usage_validator",
        "passed": passed,
        "policy": {
            "source_usage": source_usage_policy,
            "reuse_overlap_ratio": REUSE_OVERLAP_RATIO,
            "duplicate_gap_seconds": DUPLICATE_GAP,
            "timeline_epsilon_seconds": EPS,
            "intentional_reuse_requires_explicit_marker": True,
            "chronology_check": "flag_backward_source_jumps",
        },
        "stats": {
            "actions": len(actions),
            "render_segments": len(segments),
            "source_reuse_count": len(source_reuse),
            "backward_jump_count": len(backward_jumps),
            "output_gap_count": len(output_gaps),
            "invalid_segment_count": len(issues),
        },
        "source_reuse": source_reuse,
        "backward_jumps": backward_jumps,
        "output_gaps": output_gaps,
        "issues": issues,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate resolved render segments for source reuse and chronology."
    )
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--allow-backward-jumps",
        action="store_true",
        help="Report backward jumps but do not fail validation because of them.",
    )
    args = parser.parse_args()

    sequence_path = Path(args.sequence)
    output_path = Path(args.output)

    if not sequence_path.exists():
        raise FileNotFoundError(sequence_path)

    sequence = load_json(sequence_path)
    report = validate(sequence)

    if args.allow_backward_jumps:
        report["passed"] = not (
            report["issues"]
            or report["output_gaps"]
            or report["source_reuse"]
        )
        report["policy"]["chronology_check"] = "report_only"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))

    if not report["passed"]:
        print("\nVALIDATION FAILED", file=sys.stderr)
        if report["source_reuse"]:
            print(
                f"Found {len(report['source_reuse'])} repeated source moment(s).",
                file=sys.stderr,
            )
        if report["backward_jumps"] and not args.allow_backward_jumps:
            print(
                f"Found {len(report['backward_jumps'])} backward source jump(s).",
                file=sys.stderr,
            )
        if report["output_gaps"]:
            print(
                f"Found {len(report['output_gaps'])} output timeline gap(s).",
                file=sys.stderr,
            )
        sys.exit(1)

    print("\nVALIDATION PASSED: no repeated source moments detected.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
