from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

EPS = 1e-3


def load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def f(v: Any, d: float = 0.0) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else d
    except Exception:
        return d


def main() -> None:
    p = argparse.ArgumentParser(description="Validate visual edit plan invariants")
    p.add_argument("--plan", required=True)
    p.add_argument("--sequence", required=True)
    args = p.parse_args()

    plan = load(Path(args.plan))
    seq = load(Path(args.sequence))
    visual = plan.get("visual_segments", [])
    if not isinstance(visual, list) or not visual:
        raise RuntimeError("No visual_segments found")

    expected = sum(
        f(r.get("output_end")) - f(r.get("output_start"))
        for a in seq.get("actions", []) if isinstance(a, dict)
        for r in a.get("render_segments", []) if isinstance(r, dict)
    )
    actual = sum(f(s.get("output_end")) - f(s.get("output_start")) for s in visual)
    errors: list[str] = []

    if abs(expected - actual) > EPS:
        errors.append(f"timeline duration mismatch: sequence={expected:.6f}, visual={actual:.6f}")

    prev_end = 0.0
    for i, s in enumerate(visual):
        os = f(s.get("output_start"), -1)
        oe = f(s.get("output_end"), -1)
        vs = f(s.get("visual_source_start"), -1)
        ve = f(s.get("visual_source_end"), -1)
        if os < -EPS or oe <= os + EPS:
            errors.append(f"visual {i}: invalid output range {os}->{oe}")
        if i and abs(os - prev_end) > EPS:
            errors.append(f"visual {i}: output gap/overlap around {prev_end:.6f}->{os:.6f}")
        if ve <= vs + EPS:
            errors.append(f"visual {i}: invalid visual source range {vs}->{ve}")
        expected_duration = oe - os
        source_duration = ve - vs
        if abs(expected_duration - source_duration) > 0.01:
            errors.append(f"visual {i}: visual source duration {source_duration:.6f} does not match output {expected_duration:.6f}")
        if s.get("visual_reuse") and str(s.get("visual_mode")) not in {"carry_previous"}:
            errors.append(f"visual {i}: visual_reuse marker has unexpected mode {s.get('visual_mode')}")
        # HARD INVARIANT: speaker-facing visuals cannot use audio from a different source time.
        mode = str(s.get("visual_mode", ""))
        audio_s = f(s.get("audio_source_start"), -1)
        audio_e = f(s.get("audio_source_end"), -1)
        if mode == "speaker_focus":
            if audio_s < -0.5 or audio_e <= audio_s + EPS:
                errors.append(f"visual {i}: speaker_focus missing valid audio source range")
            else:
                if abs(vs - audio_s) > EPS or abs(ve - audio_e) > EPS:
                    errors.append(
                        f"visual {i}: speaker_focus AV sync mismatch "
                        f"audio={audio_s:.3f}->{audio_e:.3f}, "
                        f"visual={vs:.3f}->{ve:.3f}"
                    )

        prev_end = oe

    if errors:
        print("VISUAL PLAN INVALID")
        for e in errors:
            print("-", e)
        raise SystemExit(1)

    print("VISUAL PLAN VALID")
    print(json.dumps({
        "visual_segments": len(visual),
        "duration": round(actual, 6),
        "visual_reuse_segments": sum(1 for s in visual if s.get("visual_reuse")),
        "source_shot_boundaries_not_authoritative": bool(plan.get("policy", {}).get("source_shot_boundaries_are_not_visual_cuts")),
    }, indent=2))


if __name__ == "__main__":
    main()
