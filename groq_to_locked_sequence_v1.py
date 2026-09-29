from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EPS = 1e-6
GAP_TOL = 1e-3


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def as_list(data: Any, *keys: str) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    if isinstance(data, dict):
        for key in keys:
            value = data.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]

    raise ValueError(
        f"Could not find a list in JSON. Expected one of: {keys}"
    )


def fnum(value: Any) -> float:
    return float(value)


def shot_id(shot: dict[str, Any]) -> str:
    value = shot.get("shot_id", shot.get("id"))
    if value is None:
        return ""
    return str(value)


def intersecting_actions(
    start: float,
    end: float,
    shots: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[tuple[float, float, dict[str, Any]]] = []

    for shot in shots:
        if shot.get("start") is None or shot.get("end") is None:
            continue

        ss = fnum(shot["start"])
        se = fnum(shot["end"])

        if se <= ss:
            continue

        a = max(start, ss)
        b = min(end, se)

        if b - a > EPS:
            sid = shot_id(shot)

            if not sid:
                raise ValueError(
                    f"Shot covering {a:.6f}->{b:.6f} has no shot_id."
                )

            candidates.append((a, b, shot))

    candidates.sort(key=lambda x: (x[0], x[1]))

    actions: list[dict[str, Any]] = []
    cursor = start

    for a, b, shot in candidates:
        # Never silently bridge source gaps.
        if a > cursor + GAP_TOL:
            raise ValueError(
                "Source-shot gap inside locked editorial interval: "
                f"{cursor:.6f}->{a:.6f}. "
                "Refusing to modify the Groq boundary."
            )

        # Clamp tiny floating-point gaps only.
        actual_start = cursor if abs(a - cursor) <= GAP_TOL else a

        if b <= actual_start + EPS:
            continue

        actions.append(
            {
                "action_index": len(actions),
                "action_type": "base",
                "editorial_role": "groq_locked_moment",
                "source_start": round(actual_start, 6),
                "source_end": round(b, 6),
                "shot_index": shots.index(shot),
                "shot_id": shot_id(shot),
                "reason_types": [
                    "groq_locked_moment",
                    "source_shot_intersection",
                ],
            }
        )

        cursor = b

        if cursor >= end - GAP_TOL:
            break

    if cursor < end - GAP_TOL:
        raise ValueError(
            "Source-shot inventory does not fully cover the locked "
            f"Groq interval {start:.6f}->{end:.6f}. "
            f"Covered only through {cursor:.6f}. "
            "Refusing to extend or invent source timing."
        )

    # Force only harmless floating-point normalization at the ends.
    if actions:
        actions[0]["source_start"] = round(start, 6)
        actions[-1]["source_end"] = round(end, 6)

    return actions


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Groq editorial moments into locked, "
            "source-shot sequence JSON."
        )
    )
    parser.add_argument("--editorial", required=True, type=Path)
    parser.add_argument("--shots", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if not args.editorial.exists():
        raise SystemExit(f"Editorial JSON not found: {args.editorial}")

    if not args.shots.exists():
        raise SystemExit(f"Shots JSON not found: {args.shots}")

    editorial_data = load_json(args.editorial)
    shots_data = load_json(args.shots)

    clips = as_list(editorial_data, "clips", "moments", "selected")
    # visual_intelligence_v7 stores the actual per-shot records under
    # "shot_visual_intelligence".
    shots = as_list(
        shots_data,
        "shot_visual_intelligence",
        "shots",
        "source_shots",
    )

    if not clips:
        raise SystemExit("No Groq editorial clips found.")

    if not shots:
        raise SystemExit("No source shots found.")

    # IMPORTANT:
    # We intentionally use ONLY shot timing + shot_id.
    # No shot_type classification is consulted here.
    shots = sorted(
        shots,
        key=lambda s: (
            fnum(s["start"]),
            fnum(s["end"]),
        ),
    )

    locked_moments: list[dict[str, Any]] = []
    total_actions = 0

    for rank, clip in enumerate(clips, 1):
        if clip.get("start") is None or clip.get("end") is None:
            raise SystemExit(
                f"Clip #{rank} has no start/end."
            )

        start = fnum(clip["start"])
        end = fnum(clip["end"])

        if end <= start:
            raise SystemExit(
                f"Clip #{rank} has invalid interval "
                f"{start}->{end}."
            )

        actions = intersecting_actions(
            start,
            end,
            shots,
        )

        if not actions:
            raise SystemExit(
                f"Clip #{rank} produced no source actions."
            )

        # Explicit locked-boundary invariant.
        if abs(actions[0]["source_start"] - start) > GAP_TOL:
            raise SystemExit(
                f"Clip #{rank}: first action changed Groq start."
            )

        if abs(actions[-1]["source_end"] - end) > GAP_TOL:
            raise SystemExit(
                f"Clip #{rank}: final action changed Groq end."
            )

        locked_moments.append(
            {
                "rank": rank,
                "candidate_id": clip.get("candidate_id"),
                "title": clip.get("title", ""),
                "summary": clip.get("summary", ""),
                "moment_type": clip.get("moment_type", ""),
                "editorial_score": clip.get("score"),
                "source_start": round(start, 6),
                "source_end": round(end, 6),
                "duration": round(end - start, 6),
                "actions": actions,
            }
        )

        total_actions += len(actions)

    output = {
        "version": "1.0",
        "engine": "groq_to_locked_sequence",
        "source_editorial": str(args.editorial),
        "source_shots": str(args.shots),
        "editorial_authority": "groq",
        "source_timing_locked": True,
        "timing_policy": {
            "groq_start_end_authoritative": True,
            "shot_inventory_used_for": "visual_source_mapping_only",
            "shot_type_used": False,
            "source_gap_behavior": "fail_closed",
            "source_reuse": False,
            "source_extension": False,
            "source_truncation": False,
        },
        "moment_count": len(locked_moments),
        "action_count": total_actions,
        "moments": locked_moments,
    }

    # This top-level actions field is deliberately present for compatibility
    # with editorial_av_resolver_v1.py, while each moment retains its own
    # independent locked action list.
    output["actions"] = [
        action
        for moment in locked_moments
        for action in moment["actions"]
    ]

    output["timeline_actions"] = output["actions"]

    save_json(args.output, output)

    print("=" * 72)
    print("GROQ → LOCKED SOURCE SEQUENCE v1")
    print("=" * 72)
    print(f"Editorial moments: {len(locked_moments)}")
    print(f"Source actions:    {total_actions}")
    print()

    for moment in locked_moments:
        print(
            f"#{moment['rank']} | "
            f"{moment['source_start']:.3f} -> "
            f"{moment['source_end']:.3f} | "
            f"{moment['duration']:.3f}s | "
            f"{len(moment['actions'])} shots | "
            f"{moment['title']}"
        )

    print()
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
