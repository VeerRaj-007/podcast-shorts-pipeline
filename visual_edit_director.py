from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

EPS = 1e-6
DEFAULT_MIN_VISUAL_HOLD = 1.25
DEFAULT_MIN_INSERT_HOLD = 2.0
DEFAULT_MAX_FLASH = 1.15
DEFAULT_MIN_SPEAKER_CUT = 2.25
DEFAULT_SPEAKER_SHARE = 0.55
DEFAULT_FACE_COVERAGE = 0.035
DEFAULT_TRANSITION_COOLDOWN = 1.35


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def normalize_shots(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw = data.get("shots", data.get("render_segments", []))
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for shot in raw:
        if not isinstance(shot, dict):
            continue
        sid = shot.get("shot_id")
        if not sid:
            continue
        s = fnum(shot.get("source_start", shot.get("start")))
        e = fnum(shot.get("source_end", shot.get("end")), s)
        if e <= s + EPS:
            continue
        out.append({**shot, "shot_id": str(sid), "source_start": s, "source_end": e})
    out.sort(key=lambda x: (x["source_start"], x["source_end"]))
    return out


def normalize_observations(visual: dict[str, Any]) -> list[dict[str, Any]]:
    raw = visual.get("identity_clean_observations") or visual.get("observations_deduped") or []
    if not isinstance(raw, list):
        return []
    out = []
    for o in raw:
        if not isinstance(o, dict) or o.get("identity_id") not in {"PERSON_0", "PERSON_1"}:
            continue
        out.append(o)
    out.sort(key=lambda x: fnum(x.get("timestamp")))
    return out


def normalize_turns(speakers: dict[str, Any]) -> list[dict[str, Any]]:
    raw = speakers.get("turns", [])
    if not isinstance(raw, list):
        return []
    out = []
    for t in raw:
        if not isinstance(t, dict):
            continue
        if t.get("person_id") not in {"PERSON_0", "PERSON_1"}:
            continue
        s = fnum(t.get("start"))
        e = fnum(t.get("end"), s)
        if e > s + EPS:
            out.append({**t, "start": s, "end": e})
    out.sort(key=lambda x: x["start"])
    return out


def collect_audio_segments(sequence: dict[str, Any]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for action_index, action in enumerate(sequence.get("actions", [])):
        if not isinstance(action, dict):
            continue
        editorial_text = str(action.get("editorial_text", ""))
        editorial_start = fnum(action.get("editorial_source_start", action.get("source_start")), -1.0)
        editorial_end = fnum(action.get("editorial_source_end", action.get("source_end")), -1.0)
        for r in action.get("render_segments", []):
            if not isinstance(r, dict):
                continue
            s = fnum(r.get("source_in"))
            e = fnum(r.get("source_out"), s)
            os = fnum(r.get("output_start"))
            oe = fnum(r.get("output_end"), os)
            if e <= s + EPS or oe <= os + EPS:
                continue
            segments.append({
                "action_index": action.get("action_index", action_index),
                "output_start": os,
                "output_end": oe,
                "source_start": s,
                "source_end": e,
                "shot_id": str(r.get("shot_id")) if r.get("shot_id") else None,
                "shot_type": str(r.get("shot_type", "unknown")),
                "editorial_text": editorial_text,
                "editorial_source_start": editorial_start,
                "editorial_source_end": editorial_end,
            })
    segments.sort(key=lambda x: x["output_start"])
    return segments


def obs_in_range(observations: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    return [o for o in observations if start <= fnum(o.get("timestamp"), -1.0) <= end]


def bbox_area_ratio(o: dict[str, Any]) -> float:
    bbox = o.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return 0.0
    _, _, w, h = map(fnum, bbox)
    return max(0.0, w) * max(0.0, h) / (1920.0 * 1080.0)


def person_stats(observations: list[dict[str, Any]], start: float, end: float) -> dict[str, dict[str, float]]:
    duration = max(0.001, end - start)
    out: dict[str, dict[str, float]] = {}
    for person in ("PERSON_0", "PERSON_1"):
        items = [o for o in obs_in_range(observations, start, end) if o.get("identity_id") == person]
        if not items:
            out[person] = {"samples": 0.0, "coverage": 0.0, "area": 0.0, "center_x": 0.5, "center_y": 0.5}
            continue
        xs, ys, areas = [], [], []
        for o in items:
            bbox = o.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                x, y, w, h = map(fnum, bbox)
                xs.append((x + w / 2.0) / 1920.0)
                ys.append((y + h / 2.0) / 1080.0)
            areas.append(bbox_area_ratio(o))
        out[person] = {
            "samples": float(len(items)),
            "coverage": min(1.0, len(items) / max(1.0, duration * 2.0)),
            "area": sum(areas) / max(1, len(areas)),
            "center_x": sum(xs) / max(1, len(xs)),
            "center_y": sum(ys) / max(1, len(ys)),
        }
    return out


def speaker_share(turns: list[dict[str, Any]], start: float, end: float) -> tuple[str | None, dict[str, float]]:
    totals = {"PERSON_0": 0.0, "PERSON_1": 0.0}
    for t in turns:
        p = t["person_id"]
        totals[p] += overlap(start, end, t["start"], t["end"])
    total = sum(totals.values())
    if total <= EPS:
        return None, totals
    leader = max(totals, key=totals.get)
    share = totals[leader] / total
    return (leader if share >= DEFAULT_SPEAKER_SHARE else None), totals


def merge_candidate(curr: dict[str, Any], nxt: dict[str, Any], *, reason: str) -> dict[str, Any]:
    out = dict(curr)
    out["output_end"] = nxt["output_end"]
    out["audio_source_end"] = nxt["audio_source_end"]
    out["source_visual_end"] = max(fnum(curr.get("source_visual_end")), fnum(nxt.get("source_visual_end")))
    out["reasons"] = list(curr.get("reasons", [])) + [reason]
    return out


def find_shot(shots_by_id: dict[str, dict[str, Any]], shot_id: str | None) -> dict[str, Any] | None:
    if not shot_id:
        return None
    return shots_by_id.get(str(shot_id))


def choose_visual_role(
    seg: dict[str, Any],
    shot: dict[str, Any] | None,
    pstats: dict[str, dict[str, float]],
    focus: str | None,
    shot_content: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    duration = seg["output_end"] - seg["output_start"]
    face0 = pstats["PERSON_0"]["samples"] > 0
    face1 = pstats["PERSON_1"]["samples"] > 0
    shot_meta = shot_content.get(str(seg.get("shot_id")), {})
    shot_scene = str(shot_meta.get("scene_type", (shot or {}).get("scene_type", ""))).lower()
    shot_summary = str(shot_meta.get("visual_summary", (shot or {}).get("visual_summary", ""))).lower()
    editorial_uses = {str(x).lower() for x in (shot_meta.get("editorial_uses") or (shot or {}).get("editorial_uses") or [])}

    # Explicit visual-content metadata is required before a non-face source becomes
    # a supporting insert. Generic "insert_or_screen" is never enough.
    movie_words = ("jack nicholson", "film", "movie", "scene", "actor", "character")
    semantic_support = any(w in (shot_scene + " " + shot_summary) for w in movie_words) or bool(editorial_uses & {"supporting_example", "evidence", "reaction_reference"})
    if semantic_support and not (face0 or face1):
        return "supporting_insert", "semantic_supporting_visual"

    # Very short visual-only events are almost always bad vertical cuts.
    if not face0 and not face1 and duration <= DEFAULT_MAX_FLASH:
        return "hold_previous", "subsecond_visual_flash_suppressed"

    if focus in {"PERSON_0", "PERSON_1"} and pstats[focus]["samples"] > 0:
        return "speaker_focus", "active_speaker_present"

    if face0 and face1:
        return "two_person", "both_speakers_visible"

    if face0 or face1:
        person = "PERSON_0" if face0 else "PERSON_1"
        return "speaker_focus", f"single_visible_person_{person}"

    # Without semantic evidence, non-face footage is treated as a cutaway candidate
    # that should not interrupt the speaker by default. This is what prevents a
    # podcast Short from becoming a slideshow of source inserts.
    if duration >= DEFAULT_MIN_INSERT_HOLD and semantic_support:
        return "supporting_insert", "stable_semantic_non_face_visual"

    return "hold_previous", "nonsemantic_non_face_visual_suppressed"


def find_previous_usable(segments: list[dict[str, Any]]) -> dict[str, Any] | None:
    for s in reversed(segments):
        if s.get("visual_source_start") is not None and s.get("visual_source_end") is not None:
            return s
    return None


def find_next_speaker(segments: list[dict[str, Any]], observations: list[dict[str, Any]], start_index: int) -> dict[str, Any] | None:
    for j in range(start_index + 1, len(segments)):
        s = segments[j]
        stats = person_stats(observations, s["source_start"], s["source_end"])
        if stats["PERSON_0"]["samples"] + stats["PERSON_1"]["samples"] > 0:
            return s
    return None


def build_visual_plan(
    sequence: dict[str, Any],
    shots: list[dict[str, Any]],
    visual: dict[str, Any],
    speakers: dict[str, Any],
) -> dict[str, Any]:
    audio_segments = collect_audio_segments(sequence)
    if not audio_segments:
        raise RuntimeError("No render_segments found in resolved sequence")

    obs = normalize_observations(visual)
    turns = normalize_turns(speakers)
    shots_by_id = {s["shot_id"]: s for s in shots}

    shot_content: dict[str, dict[str, Any]] = {}
    if isinstance(sequence.get("shot_content"), list):
        shot_content = {str(x.get("shot_id")): x for x in sequence["shot_content"] if isinstance(x, dict) and x.get("shot_id")}

    visual_segments: list[dict[str, Any]] = []
    transition_cooldown = 0.0

    for i, seg in enumerate(audio_segments):
        start, end = seg["source_start"], seg["source_end"]
        pstats = person_stats(obs, start, end)
        focus, totals = speaker_share(turns, start, end)
        shot = find_shot(shots_by_id, seg.get("shot_id"))
        role, reason = choose_visual_role(seg, shot, pstats, focus, shot_content)

        item = {
            "visual_index": len(visual_segments),
            "output_start": seg["output_start"],
            "output_end": seg["output_end"],
            "audio_source_start": seg["source_start"],
            "audio_source_end": seg["source_end"],
            "visual_source_start": seg["source_start"],
            "visual_source_end": seg["source_end"],
            "visual_shot_id": seg.get("shot_id"),
            "visual_mode": role,
            "target_identity": focus if role == "speaker_focus" else None,
            "speaker_seconds": totals,
            "face_stats": pstats,
            "source_shot_type": seg.get("shot_type"),
            "source_visual_summary": str((shot or {}).get("visual_summary", "")),
            "source_scene_type": str((shot or {}).get("scene_type", "")),
            "source_editorial_text": seg.get("editorial_text", ""),
            "visual_reuse": False,
            "reasons": [reason],
        }

        # A short shot change inside the same active speaker should not automatically
        # become a vertical edit. Carry the previous visual when the new shot is too
        # brief to justify a composition change.
        if visual_segments and role == "speaker_focus":
            prev = visual_segments[-1]
            same_target = prev.get("target_identity") == focus
            current_duration = end - start
            if same_target and prev.get("visual_mode") in {"speaker_focus", "carry_previous"} and current_duration < DEFAULT_MIN_SPEAKER_CUT:
                prev_len = prev["visual_source_end"] - prev["visual_source_start"]
                if prev_len + 0.001 >= current_duration:
                    item["visual_source_start"] = prev["visual_source_end"] - current_duration
                    item["visual_source_end"] = prev["visual_source_end"]
                    item["visual_shot_id"] = prev.get("visual_shot_id")
                    item["visual_mode"] = "carry_previous"
                else:
                    # Keep current source; a source-duration-safe replacement does not exist.
                    item["reasons"] = ["short_speaker_shot_not_replaced_due_to_source_duration"]

                item["visual_reuse"] = True
                item["target_identity"] = prev.get("target_identity")
                item["source_visual_summary"] = prev.get("source_visual_summary", "")
                item["reasons"] = ["short_speaker_shot_suppressed", "carry_previous_visual"]

        if role == "hold_previous":
            prev = find_previous_usable(visual_segments)
            nxt = find_next_speaker(audio_segments, obs, i)
            # A short carry is intentionally used to remove a flash. The carry is
            # visual-only; the audio remains the exact editorial source range.
            if prev:
                carry_duration = end - start
                prev_len = prev["visual_source_end"] - prev["visual_source_start"]
                if prev_len + 0.001 >= carry_duration:
                    item["visual_source_start"] = prev["visual_source_end"] - carry_duration
                    item["visual_source_end"] = prev["visual_source_end"]
                    item["visual_shot_id"] = prev.get("visual_shot_id")
                    item["visual_mode"] = "carry_previous"
                    item["visual_reuse"] = True
                    item["target_identity"] = prev.get("target_identity")
                    item["source_visual_summary"] = prev.get("source_visual_summary", "")
                    item["reasons"] = [reason, "carry_previous_visual_to_avoid_flash"]
                else:
                    prev = None
            if prev is None and nxt:
                next_shot_len = nxt["source_end"] - nxt["source_start"]
                if next_shot_len + 0.001 >= (end - start):
                    item["visual_source_start"] = nxt["source_start"]
                    item["visual_source_end"] = nxt["source_start"] + (end - start)
                    item["visual_shot_id"] = nxt.get("shot_id")
                    item["visual_mode"] = "advance_to_next_visual"
                    item["visual_reuse"] = False
                    item["reasons"] = [reason, "use_next_stable_visual"]
                else:
                    nxt = None
            if prev is None and nxt is None:
                # No safe alternative with enough source duration. Do not invent
                # frames, loop, stretch, or overrun the source shot. Keep the current
                # source visual and let the camera director make it clean.
                item["visual_mode"] = "current_source"
                item["reasons"] = [reason, "no_safe_visual_replacement_available"]

        if visual_segments:
            prev = visual_segments[-1]
            same_mode = prev["visual_mode"] == item["visual_mode"]
            same_target = prev.get("target_identity") == item.get("target_identity")
            gap = item["output_start"] - prev["output_end"]
            same_visual_source = prev.get("visual_shot_id") == item.get("visual_shot_id") and abs(
                fnum(prev.get("visual_source_end")) - fnum(item.get("visual_source_start"))
            ) <= 0.001
            if same_mode and same_target and same_visual_source and gap <= EPS:
                # Merge only when the actual visual source is continuous. We never
                # hide a source cut merely by renaming it as the same camera mode.
                prev["output_end"] = item["output_end"]
                prev["audio_source_end"] = item["audio_source_end"]
                prev["visual_source_end"] = item["visual_source_end"]
                prev["reasons"].extend(item["reasons"])
                prev["visual_shot_ids"] = sorted(set(prev.get("visual_shot_ids", [prev.get("visual_shot_id")]) + [item.get("visual_shot_id")]))
                continue

        visual_segments.append(item)
        transition_cooldown = max(0.0, transition_cooldown - (end - start))
        if role not in {"hold_previous"}:
            transition_cooldown = DEFAULT_TRANSITION_COOLDOWN

    # Re-index and make timeline invariants explicit.
    for idx, item in enumerate(visual_segments):
        item["visual_index"] = idx
        item.setdefault("visual_shot_ids", [item.get("visual_shot_id")])
        item["duration"] = round(item["output_end"] - item["output_start"], 6)
        item["visual_source_duration"] = round(item["visual_source_end"] - item["visual_source_start"], 6)

    source_start = min(x["audio_source_start"] for x in visual_segments)
    source_end = max(x["audio_source_end"] for x in visual_segments)
    output_duration = max(x["output_end"] for x in visual_segments)
    changes = max(0, len(visual_segments) - 1)

    return {
        "version": "1.0",
        "module": "visual_edit_director",
        "method": "semantic_visual_timeline_over_locked_editorial_audio",
        "policy": {
            "editorial_story_locked": True,
            "audio_source_timeline_authoritative": True,
            "visual_timeline_independent": True,
            "source_shot_boundaries_are_not_visual_cuts": True,
            "subsecond_nonsemantic_visuals_suppressed": True,
            "speaker_focus_preferred_when_speaker_visible": True,
            "visual_reuse_requires_explicit_marker": True,
            "audio_reuse_forbidden": True,
        },
        "source_range": [round(source_start, 6), round(source_end, 6)],
        "output_duration": round(output_duration, 6),
        "visual_cut_count": changes,
        "visual_segments": visual_segments,
        "diagnostics": {
            "input_audio_segments": len(audio_segments),
            "output_visual_segments": len(visual_segments),
            "source_shot_reduction": len(audio_segments) - len(visual_segments),
            "visual_reuse_segments": sum(1 for x in visual_segments if x.get("visual_reuse")),
            "modes": {
                mode: sum(1 for x in visual_segments if x.get("visual_mode") == mode)
                for mode in sorted({str(x.get("visual_mode")) for x in visual_segments})
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a semantic visual edit timeline for a vertical Short")
    parser.add_argument("--sequence", required=True, help="Resolved AV sequence JSON")
    parser.add_argument("--shots", required=True, help="Source shot inventory JSON")
    parser.add_argument("--visual", required=True, help="Identity/face visual intelligence JSON")
    parser.add_argument("--speakers", required=True, help="Speaker attribution JSON")
    parser.add_argument("--shot-content", default=None, help="Optional semantic shot-content JSON")
    parser.add_argument("--output", required=True, help="Visual edit plan JSON")
    args = parser.parse_args()

    sequence = load_json(Path(args.sequence))
    if args.shot_content:
        content_payload = load_json(Path(args.shot_content))
        content_items = content_payload.get("shots", []) if isinstance(content_payload, dict) else []
        sequence["shot_content"] = content_items if isinstance(content_items, list) else []

    result = build_visual_plan(
        sequence,
        normalize_shots(load_json(Path(args.shots))),
        load_json(Path(args.visual)),
        load_json(Path(args.speakers)),
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== VISUAL EDIT DIRECTOR ===")
    print(json.dumps(result["diagnostics"], indent=2))
    for s in result["visual_segments"]:
        print(
            f"{s['output_start']:.3f}->{s['output_end']:.3f} | "
            f"{s['visual_mode']} | target={s.get('target_identity')} | "
            f"audio={s['audio_source_start']:.3f}->{s['audio_source_end']:.3f} | "
            f"visual={s['visual_source_start']:.3f}->{s['visual_source_end']:.3f} | "
            f"reason={'; '.join(s.get('reasons', []))}"
        )
    print(f"output: {out}")


if __name__ == "__main__":
    main()
