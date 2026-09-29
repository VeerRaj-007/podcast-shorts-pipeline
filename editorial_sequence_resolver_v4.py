import argparse
import json
from pathlib import Path

EPS = 1e-6
OVERLAP_EPS = 1e-4
DIALOGUE_KINDS = {"base", "speaker", "reaction", "framing"}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def subtract_used(start, end, used):
    """Return unused pieces of [start,end] after subtracting used source intervals."""
    pieces = [(start, end)]
    for u0, u1 in sorted(used):
        if u1 <= start + OVERLAP_EPS or u0 >= end - OVERLAP_EPS:
            continue
        next_pieces = []
        for p0, p1 in pieces:
            if u1 <= p0 + OVERLAP_EPS or u0 >= p1 - OVERLAP_EPS:
                next_pieces.append((p0, p1))
                continue
            if u0 > p0 + OVERLAP_EPS:
                next_pieces.append((p0, min(u0, p1)))
            if u1 < p1 - OVERLAP_EPS:
                next_pieces.append((max(u1, p0), p1))
        pieces = [(a, b) for a, b in next_pieces if b - a > OVERLAP_EPS]
        if not pieces:
            break
    return pieces


def preferred_index(action, shots):
    wanted = action.get("shot_id")
    if not wanted:
        return None
    for i, shot in enumerate(shots):
        if shot.get("shot_id") == wanted:
            return i
    return None


def ordered_candidates(action, shots, chronology_cursor, used_intervals):
    """Prefer the planned shot, but stay near the current source position."""
    idx = preferred_index(action, shots)
    kind = str(action.get("kind", "base")).lower()
    dialogue_like = kind in DIALOGUE_KINDS

    def candidate_distance(i, shot):
        start = float(shot["start"])
        end = float(shot["end"])
        unused = subtract_used(start, end, used_intervals)
        has_unused = bool(unused)
        before = end <= chronology_cursor + OVERLAP_EPS
        distance = abs(max(start, chronology_cursor) - chronology_cursor)
        preferred_penalty = 0 if idx == i else 1
        return (
            not has_unused,
            before if dialogue_like else False,
            preferred_penalty,
            distance,
            start,
        )

    candidates = list(enumerate(shots))
    candidates.sort(key=lambda pair: candidate_distance(pair[0], pair[1]))
    return [shot for _, shot in candidates]


def resolve_action(action, shots, used_intervals, chronology_cursor):
    output_start = float(action["start"])
    output_end = float(action["end"])
    needed = output_end - output_start
    if needed <= EPS:
        return [], chronology_cursor

    kind = str(action.get("kind", "base")).lower()
    dialogue_like = kind in DIALOGUE_KINDS
    candidates = ordered_candidates(action, shots, chronology_cursor, used_intervals)

    remaining = needed
    local_used = []
    segments = []

    for shot in candidates:
        if remaining <= EPS:
            break

        shot_start = float(shot["start"])
        shot_end = float(shot["end"])

        if dialogue_like and shot_end <= chronology_cursor + OVERLAP_EPS:
            continue

        start = max(shot_start, chronology_cursor) if dialogue_like else shot_start
        if start >= shot_end - OVERLAP_EPS:
            continue

        available_pieces = subtract_used(
            start,
            shot_end,
            used_intervals + local_used,
        )
        if not available_pieces:
            continue

        for piece_start, piece_end in available_pieces:
            if remaining <= EPS:
                break

            take = min(remaining, piece_end - piece_start)
            if take <= OVERLAP_EPS:
                continue

            source_in = piece_start
            source_out = piece_start + take

            segments.append({
                "shot_id": shot.get("shot_id"),
                "shot_type": shot.get("shot_type"),
                "source_in": source_in,
                "source_out": source_out,
                "duration": take,
            })
            local_used.append((source_in, source_out))
            remaining -= take

            if dialogue_like:
                chronology_cursor = source_out

    if remaining > 1e-4:
        return None, chronology_cursor

    cursor = output_start
    for seg in segments:
        seg["output_start"] = cursor
        seg["output_end"] = cursor + seg["duration"]
        cursor = seg["output_end"]

    return segments, chronology_cursor


def validate_action(action):
    expected = float(action["end"]) - float(action["start"])
    actual = sum(
        float(s["source_out"]) - float(s["source_in"])
        for s in action.get("render_segments", [])
    )
    return abs(expected - actual) <= 1e-3


def validate_output_timeline(actions):
    gaps, overlaps = [], []
    actions = sorted(actions, key=lambda a: float(a["start"]))
    if not actions:
        return gaps, overlaps

    cursor = float(actions[0]["start"])
    for action in actions:
        start = float(action["start"])
        end = float(action["end"])
        if start > cursor + 1e-4:
            gaps.append({"start": cursor, "end": start, "duration": start - cursor})
        elif start < cursor - 1e-4:
            overlaps.append({"start": start, "end": min(end, cursor), "duration": min(end, cursor) - start})
        cursor = max(cursor, end)
    return gaps, overlaps


def validate_global_source_usage(actions):
    segments = []
    for ai, action in enumerate(actions):
        for si, seg in enumerate(action.get("render_segments", [])):
            segments.append({
                **seg,
                "action_index": ai,
                "segment_index": si,
                "kind": action.get("kind"),
            })
    segments.sort(key=lambda s: float(s["output_start"]))

    reuse = []
    backward = []
    previous = None

    for i, cur in enumerate(segments):
        c0, c1 = float(cur["source_in"]), float(cur["source_out"])
        if previous is not None:
            p0, p1 = float(previous["source_in"]), float(previous["source_out"])
            if c0 < p0 - OVERLAP_EPS and str(cur.get("kind")) in DIALOGUE_KINDS:
                backward.append({
                    "segment_index": i,
                    "previous_segment_index": i - 1,
                    "previous_source": [p0, p1],
                    "current_source": [c0, c1],
                    "kind": cur.get("kind"),
                })

        for j, old in enumerate(segments[:i]):
            o0, o1 = float(old["source_in"]), float(old["source_out"])
            if overlap(c0, c1, o0, o1) > OVERLAP_EPS:
                adjacent = (
                    abs(c0 - o1) <= 0.05
                    and abs(float(cur["output_start"]) - float(old["output_end"])) <= 0.05
                )
                if not adjacent:
                    reuse.append({
                        "current_segment_index": i,
                        "previous_segment_index": j,
                        "current_source": [c0, c1],
                        "previous_source": [o0, o1],
                        "current_kind": cur.get("kind"),
                        "previous_kind": old.get("kind"),
                    })
                    break
        previous = cur

    return reuse, backward


def main():
    parser = argparse.ArgumentParser(
        description="Resolve an editorial sequence with strict source uniqueness and dialogue chronology."
    )
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--shots", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    sequence = load_json(args.sequence)
    shot_data = load_json(args.shots)
    actions = sequence.get("actions", sequence.get("timeline_actions", []))

    shots = shot_data.get("shots", []) if isinstance(shot_data, dict) else shot_data
    if not isinstance(actions, list) or not isinstance(shots, list):
        raise ValueError("Invalid sequence or shots JSON structure.")

    actions = sorted(actions, key=lambda a: float(a["start"]))
    shots = sorted(shots, key=lambda s: (float(s["start"]), float(s["end"])))

    resolved_actions = []
    unresolved = []
    used_intervals = []
    chronology_cursor = 0.0

    for action in actions:
        segments, chronology_cursor = resolve_action(
            action,
            shots,
            used_intervals,
            chronology_cursor,
        )

        if segments is None:
            unresolved.append({
                "start": action.get("start"),
                "end": action.get("end"),
                "shot_id": action.get("shot_id"),
                "kind": action.get("kind"),
                "reason": "No unused real source footage can satisfy this action under strict source uniqueness/chronology.",
            })
            continue

        resolved = dict(action)
        resolved["render_segments"] = segments
        resolved["render_safe"] = validate_action(resolved)
        resolved_actions.append(resolved)
        used_intervals.extend(
            (float(s["source_in"]), float(s["source_out"]))
            for s in segments
        )

    gaps, overlaps = validate_output_timeline(resolved_actions)
    source_reuse, backward = validate_global_source_usage(resolved_actions)

    source_duration_errors = []
    for action in resolved_actions:
        if not validate_action(action):
            expected = float(action["end"]) - float(action["start"])
            actual = sum(
                float(s["source_out"]) - float(s["source_in"])
                for s in action.get("render_segments", [])
            )
            source_duration_errors.append({
                "start": action["start"],
                "end": action["end"],
                "expected_duration": expected,
                "actual_source_duration": actual,
                "error_seconds": expected - actual,
            })

    render_safe = (
        not unresolved
        and not gaps
        and not overlaps
        and not source_reuse
        and not backward
        and not source_duration_errors
        and all(a.get("render_safe") for a in resolved_actions)
    )

    output = {
        "version": "4.0",
        "mode": "strict_source_unique_chronological",
        "video_start": sequence.get("video_start", 0.0),
        "video_end": sequence.get("video_end"),
        "input_actions": len(actions),
        "actions": resolved_actions,
        "unresolved_source_gaps": unresolved,
        "source_duration_errors": source_duration_errors,
        "source_reuse": source_reuse,
        "backward_source_jumps": backward,
        "render_safe": render_safe,
        "policy": {
            "source_reuse": "forbidden",
            "intentional_replay_requires_explicit_marker": True,
            "dialogue_chronology": "monotonic",
            "inserts_may_use_unused_source_elsewhere": True,
            "freeze": False,
            "loop": False,
            "stretch": False,
        },
        "stats": {
            "input_actions": len(actions),
            "resolved_actions": len(resolved_actions),
            "unresolved_actions": len(unresolved),
            "render_segments": sum(len(a.get("render_segments", [])) for a in resolved_actions),
            "source_reuse_count": len(source_reuse),
            "backward_jump_count": len(backward),
        },
    }

    save_json(args.output, output)
    print(json.dumps(output, indent=2))

    if not render_safe:
        raise SystemExit(1)

    print("\nSUCCESS: strict resolver produced a source-unique, chronological render plan.")


if __name__ == "__main__":
    main()
