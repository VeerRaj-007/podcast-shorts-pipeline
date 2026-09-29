#!/usr/bin/env python3
"""
FINAL face-aware camera director for the independent interview Shorts pipeline.

Design:
  - Groq/editorial sequence is authoritative for WHAT gets rendered.
  - visual_intelligence_v7 is authoritative for WHERE the people actually are.
  - speaker attribution is used only to decide WHO should be framed.
  - camera is STATIC inside each source shot.
  - NO crop stabilization, NO animated pan, NO xfade, NO guessing from
    new_inde_shots.json's old face fields.

The script splits a selected source segment at visual-intelligence shot
boundaries, then chooses one fixed portrait crop for each source shot.

Usage:
  python face_aware_camera_director_final.py \
      --sequence tmp/new_inde_short01_sequence_v2.json \
      --visual tmp/new_inde_visual_intelligence_v7.json \
      --speakers tmp/full_video_speaker_attribution_v1.json \
      --output tmp/new_inde_short01_camera_final.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


W = 1920
H = 1080
OUT_W = 1080
OUT_H = 1920
CROP_W = int(round(H * 9 / 16))  # 607.5 -> 608
CROP_H = H
MIN_SPEAKER_OVERLAP = 0.55
MIN_VISIBLE_AREA = 0.0015
NEUTRAL_LEFT = (W - CROP_W) // 2


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def first_number(d: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for k in keys:
        v = d.get(k)
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                pass
    return None


def norm_segment(d: dict[str, Any], fallback_output_start: float = 0.0) -> dict[str, Any] | None:
    sin_ = first_number(d, ("source_in", "source_start", "source_from", "in"))
    sout = first_number(d, ("source_out", "source_end", "source_to", "out"))

    # Some sequence versions store the source range inside an action object.
    if sin_ is None and isinstance(d.get("action"), dict):
        a = d["action"]
        sin_ = first_number(a, ("source_in", "source_start", "source_from", "in"))
        sout = first_number(a, ("source_out", "source_end", "source_to", "out"))

    if sin_ is None or sout is None or sout <= sin_:
        return None

    o0 = first_number(
        d,
        ("output_start", "render_start", "timeline_start", "out_start"),
    )
    o1 = first_number(
        d,
        ("output_end", "render_end", "timeline_end", "out_end"),
    )

    return {
        "source_in": sin_,
        "source_out": sout,
        "output_start": fallback_output_start if o0 is None else o0,
        "output_end": (
            fallback_output_start + (sout - sin_) if o1 is None else o1
        ),
        "raw": d,
    }


def collect_render_segments(seq: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Prefer explicit render_segments. Fall back to action-like objects.
    This is intentionally schema-tolerant because the v2 sequence files
    have evolved during development.
    """
    found: list[dict[str, Any]] = []

    def walk(obj: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "render_segments" and isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict):
                            found.append(item)
                elif isinstance(v, (dict, list)):
                    walk(v, path + (k,))
        elif isinstance(obj, list):
            for item in obj:
                walk(item, path)

    walk(seq)

    normalized: list[dict[str, Any]] = []
    cursor = 0.0
    seen: set[tuple[float, float]] = set()

    for item in found:
        n = norm_segment(item, cursor)
        if not n:
            continue
        key = (round(n["source_in"], 6), round(n["source_out"], 6))
        if key in seen:
            continue
        seen.add(key)
        normalized.append(n)
        cursor = n["output_end"]

    if normalized:
        normalized.sort(key=lambda x: (x["output_start"], x["source_in"]))
        return normalized

    # Fallback: collect dicts that look like renderable source actions.
    candidates: list[dict[str, Any]] = []

    def fallback_walk(obj: Any) -> None:
        if isinstance(obj, dict):
            if (
                any(k in obj for k in ("action", "type", "kind"))
                and first_number(obj, ("source_in", "source_start", "in")) is not None
                and first_number(obj, ("source_out", "source_end", "out")) is not None
            ):
                candidates.append(obj)
            for v in obj.values():
                if isinstance(v, (dict, list)):
                    fallback_walk(v)
        elif isinstance(obj, list):
            for item in obj:
                fallback_walk(item)

    fallback_walk(seq)

    out: list[dict[str, Any]] = []
    cursor = 0.0
    for item in candidates:
        n = norm_segment(item, cursor)
        if n:
            out.append(n)
            cursor = n["output_end"]

    out.sort(key=lambda x: (x["output_start"], x["source_in"]))
    return out


def build_shots(visual: dict[str, Any]) -> list[dict[str, Any]]:
    shots = visual.get("shot_visual_intelligence")
    if not isinstance(shots, list):
        raise RuntimeError("visual_intelligence_v7 is missing shot_visual_intelligence")

    out = []
    for s in shots:
        if not isinstance(s, dict):
            continue
        s0 = first_number(s, ("start",))
        s1 = first_number(s, ("end",))
        if s0 is None or s1 is None or s1 <= s0:
            continue
        out.append(s)
    out.sort(key=lambda s: (s["start"], s["end"]))
    return out


def build_observations(visual: dict[str, Any]) -> list[dict[str, Any]]:
    obs = visual.get("identity_clean_observations")
    if not isinstance(obs, list):
        raise RuntimeError("visual_intelligence_v7 is missing identity_clean_observations")
    return [x for x in obs if isinstance(x, dict)]


def build_speaker_turns(speakers: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    turns_raw = speakers.get("turns", [])
    turns = [x for x in turns_raw if isinstance(x, dict)]

    mapping: dict[str, str] = {}
    for key in (
        "speaker_to_identity",
        "speaker_identity_map",
        "speaker_mapping",
        "mapping",
        "speaker_identity",
    ):
        m = speakers.get(key)
        if isinstance(m, dict):
            for spk, val in m.items():
                if isinstance(val, str):
                    mapping[str(spk)] = val
                elif isinstance(val, dict):
                    ident = val.get("identity_id") or val.get("identity")
                    if isinstance(ident, str):
                        mapping[str(spk)] = ident

    # Also accept a common list-shaped mapping.
    if isinstance(speakers.get("speaker_identity_mapping"), list):
        for item in speakers["speaker_identity_mapping"]:
            if isinstance(item, dict):
                spk = item.get("speaker_id") or item.get("speaker")
                ident = item.get("identity_id") or item.get("identity")
                if isinstance(spk, str) and isinstance(ident, str):
                    mapping[spk] = ident

    return turns, mapping


def turn_range(t: dict[str, Any]) -> tuple[float, float] | None:
    a = first_number(t, ("start", "start_time", "time_start"))
    b = first_number(t, ("end", "end_time", "time_end"))
    if a is None or b is None or b <= a:
        return None
    return a, b


def turn_identity(t: dict[str, Any], mapping: dict[str, str]) -> str | None:
    direct = t.get("identity_id") or t.get("identity")
    if isinstance(direct, str):
        return direct
    spk = t.get("speaker_id") or t.get("speaker") or t.get("label")
    if isinstance(spk, str):
        return mapping.get(spk)
    return None


def dominant_identity(
    start: float,
    end: float,
    turns: list[dict[str, Any]],
    mapping: dict[str, str],
) -> tuple[str | None, float, dict[str, float]]:
    totals: dict[str, float] = {}
    for t in turns:
        r = turn_range(t)
        if not r:
            continue
        ident = turn_identity(t, mapping)
        if not ident:
            continue
        ov = overlap(start, end, r[0], r[1])
        if ov > 0:
            totals[ident] = totals.get(ident, 0.0) + ov

    total = end - start
    if not totals or total <= 0:
        return None, 0.0, totals

    ident = max(totals, key=totals.get)
    share = totals[ident] / total
    return ident, share, totals


def shot_for_time(shots: list[dict[str, Any]], t: float) -> dict[str, Any] | None:
    # Direct containment first.
    for s in shots:
        if s["start"] <= t < s["end"]:
            return s

    # Nearest shot if exactly on a boundary.
    best = None
    best_dist = float("inf")
    for s in shots:
        if t < s["start"]:
            dist = s["start"] - t
        elif t >= s["end"]:
            dist = t - s["end"]
        else:
            dist = 0.0
        if dist < best_dist:
            best = s
            best_dist = dist
    return best


def split_by_visual_shots(
    start: float,
    end: float,
    shots: list[dict[str, Any]],
) -> list[tuple[float, float, dict[str, Any]]]:
    boundaries = {start, end}
    touched = []
    for s in shots:
        ov = overlap(start, end, s["start"], s["end"])
        if ov > 1e-6:
            a = max(start, s["start"])
            b = min(end, s["end"])
            boundaries.add(a)
            boundaries.add(b)
            touched.append((a, b, s))

    cuts = sorted(boundaries)
    pieces = []
    for a, b in zip(cuts, cuts[1:]):
        if b - a <= 1e-6:
            continue
        mid = (a + b) / 2
        s = shot_for_time(touched and [x[2] for x in touched] or shots, mid)
        if s:
            pieces.append((a, b, s))
    return pieces


def representative_face_x(
    shot: dict[str, Any],
    start: float,
    end: float,
    identity: str | None,
    observations: list[dict[str, Any]],
) -> tuple[float | None, float]:
    """
    Return (normalized x, support score).

    We only use observations inside THIS source shot and THIS identity.
    That prevents an identity's position from another camera shot leaking
    into the current crop.
    """
    rows: list[tuple[float, float, float]] = []
    sid = shot.get("shot_id")

    for o in observations:
        if sid is not None and o.get("shot_id") != sid:
            continue
        ts = first_number(o, ("timestamp", "time"))
        if ts is None or ts < start - 0.03 or ts > end + 0.03:
            continue
        if identity and o.get("identity_id") != identity:
            continue

        c = o.get("center")
        if not (isinstance(c, list) and len(c) >= 1):
            continue
        try:
            x = float(c[0])
        except (TypeError, ValueError):
            continue
        area = float(o.get("area_normalized", 0.0) or 0.0)
        score = float(o.get("score", o.get("confidence", 1.0)) or 1.0)

        if area < MIN_VISIBLE_AREA:
            continue
        rows.append((x, area, score))

    if not rows:
        return None, 0.0

    xs = [x for x, _, _ in rows]
    med = statistics.median(xs)

    # Support is intentionally conservative: enough reliable samples
    # in the actual shot to trust the position.
    support = min(1.0, len(rows) / 5.0)
    return med, support


def fallback_identity_x(
    shot: dict[str, Any],
    observations: list[dict[str, Any]],
) -> float | None:
    sid = shot.get("shot_id")
    candidates = []
    for o in observations:
        if sid is not None and o.get("shot_id") != sid:
            continue
        c = o.get("center")
        if not (isinstance(c, list) and len(c) >= 1):
            continue
        area = float(o.get("area_normalized", 0.0) or 0.0)
        if area < MIN_VISIBLE_AREA:
            continue
        candidates.append((area, float(c[0])))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]


def crop_left_from_face_x(face_x: float | None) -> int:
    if face_x is None:
        return NEUTRAL_LEFT
    center_px = clamp(face_x, 0.0, 1.0) * W
    left = center_px - CROP_W / 2.0
    left = clamp(left, 0.0, W - CROP_W)
    # Force even x for yuv420p safety.
    return int(round(left / 2.0) * 2)


def plan_segment(
    start: float,
    end: float,
    shot: dict[str, Any],
    visual_obs: list[dict[str, Any]],
    turns: list[dict[str, Any]],
    mapping: dict[str, str],
) -> dict[str, Any]:
    composition = str(shot.get("composition", "unknown"))

    speaker, speaker_share, speaker_shares = dominant_identity(
        start, end, turns, mapping
    )

    face_x = None
    support = 0.0
    if speaker and speaker_share >= MIN_SPEAKER_OVERLAP:
        face_x, support = representative_face_x(
            shot, start, end, speaker, visual_obs
        )

    # For a single-face shot, if diarization is weak but there is exactly
    # one clean visible identity, use that person's face instead of a blind
    # center crop.
    if face_x is None and composition == "single_face":
        present = []
        sid = shot.get("shot_id")
        for o in visual_obs:
            if sid is not None and o.get("shot_id") != sid:
                continue
            c = o.get("center")
            if not (isinstance(c, list) and len(c) >= 1):
                continue
            area = float(o.get("area_normalized", 0.0) or 0.0)
            if area < MIN_VISIBLE_AREA:
                continue
            ident = o.get("identity_id")
            if ident:
                present.append(ident)
        identities = sorted(set(present))
        if len(identities) == 1:
            face_x, support = representative_face_x(
                shot, start, end, identities[0], visual_obs
            )
            if speaker is None:
                speaker = identities[0]

    left = crop_left_from_face_x(face_x)

    if face_x is not None:
        mode = "speaker_focus" if speaker else "face_focus"
        reason = (
            f"{composition}: fixed portrait crop around visible "
            f"{speaker or 'face'} at x={face_x:.4f}"
        )
    else:
        mode = "neutral"
        reason = f"{composition}: no reliable speaker face in this source shot"

    return {
        "source_start": round(start, 6),
        "source_end": round(end, 6),
        "duration": round(end - start, 6),
        "shot_id": shot.get("shot_id"),
        "composition": composition,
        "speaker_identity": speaker,
        "speaker_share": round(speaker_share, 4),
        "speaker_shares": {k: round(v, 4) for k, v in speaker_shares.items()},
        "speaker_face_x": None if face_x is None else round(face_x, 5),
        "face_support": round(support, 4),
        "camera_mode": mode,
        "crop": {
            "x": left,
            "y": 0,
            "width": CROP_W,
            "height": CROP_H,
            "output_width": OUT_W,
            "output_height": OUT_H,
        },
        "animated": False,
        "transition": "hard_cut",
        "reason": reason,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--visual", required=True)
    ap.add_argument("--speakers", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    seq = load_json(args.sequence)
    visual = load_json(args.visual)
    speakers = load_json(args.speakers)

    base_segments = collect_render_segments(seq)
    if not base_segments:
        raise RuntimeError(
            "No renderable source segments found in the sequence JSON. "
            "Expected render_segments with source_in/source_out."
        )

    shots = build_shots(visual)
    observations = build_observations(visual)
    turns, mapping = build_speaker_turns(speakers)

    output_segments: list[dict[str, Any]] = []
    output_cursor = 0.0

    for base in base_segments:
        s0 = base["source_in"]
        s1 = base["source_out"]

        pieces = split_by_visual_shots(s0, s1, shots)
        if not pieces:
            shot = shot_for_time(shots, (s0 + s1) / 2)
            if not shot:
                raise RuntimeError(f"No visual-intelligence shot for {s0:.3f}-{s1:.3f}")
            pieces = [(s0, s1, shot)]

        for p0, p1, shot in pieces:
            plan = plan_segment(
                p0, p1, shot, observations, turns, mapping
            )
            dur = p1 - p0
            plan["output_start"] = round(output_cursor, 6)
            plan["output_end"] = round(output_cursor + dur, 6)
            output_cursor += dur
            output_segments.append(plan)

    result = {
        "version": "final-1.0",
        "mode": "source_shot_static_face_aware",
        "sequence_source": str(args.sequence),
        "visual_intelligence_source": str(args.visual),
        "speaker_attribution_source": str(args.speakers),
        "video": {"width": W, "height": H, "fps": 30.0},
        "output": {
            "width": OUT_W,
            "height": OUT_H,
            "crop_width": CROP_W,
            "crop_height": CROP_H,
            "camera_animation": False,
            "video_transition": "hard_cut",
            "audio_transition": "none",
        },
        "statistics": {
            "base_segments": len(base_segments),
            "camera_segments": len(output_segments),
            "duration": round(output_cursor, 6),
        },
        "segments": output_segments,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("=== FINAL CAMERA DIRECTOR ===")
    print(f"base segments : {len(base_segments)}")
    print(f"camera shots  : {len(output_segments)}")
    print(f"duration      : {output_cursor:.3f}s")
    for i, s in enumerate(output_segments, 1):
        c = s["crop"]
        print(
            f"{i:02d} {s['shot_id']} "
            f"{s['source_start']:.3f}->{s['source_end']:.3f} "
            f"{s['camera_mode']} "
            f"speaker={s['speaker_identity']} "
            f"face_x={s['speaker_face_x']} "
            f"crop_x={c['x']}"
        )


if __name__ == "__main__":
    main()