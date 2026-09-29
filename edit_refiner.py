#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def interval_distance(a0: float, a1: float, b0: float, b1: float) -> float:
    if a1 < b0:
        return b0 - a1
    if b1 < a0:
        return a0 - b1
    return 0.0


def center_distance(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    width: float,
    height: float,
) -> float:
    dx = (ax - bx) / max(1.0, width)
    dy = (ay - by) / max(1.0, height)
    return math.sqrt(dx * dx + dy * dy)


def stable_timeline(data: dict[str, Any]) -> list[dict[str, Any]]:
    items = data.get("timeline") or data.get("samples") or []
    out = [x for x in items if isinstance(x, dict) and "time" in x]
    out.sort(key=lambda x: float(x["time"]))
    return out


def nearest_stable(
    timeline: list[dict[str, Any]],
    t: float,
    max_distance: float = 0.75,
) -> dict[str, Any] | None:
    if not timeline:
        return None
    item = min(timeline, key=lambda x: abs(float(x["time"]) - t))
    if abs(float(item["time"]) - t) > max_distance:
        return None
    return item


def stable_face_detections_at(
    face_data: dict[str, Any],
    t: float,
    max_distance: float = 0.75,
) -> list[dict[str, Any]]:
    tracks = face_data.get("tracks") or []
    matches: list[dict[str, Any]] = []

    for track in tracks:
        if not isinstance(track, dict):
            continue

        start = float(track.get("start", 0.0))
        end = float(track.get("end", start))
        if t < start - max_distance or t > end + max_distance:
            continue

        best = None
        best_dist = 10**9
        for det in track.get("detections", []):
            if not isinstance(det, dict) or "time" not in det:
                continue
            d = abs(float(det["time"]) - t)
            if d < best_dist:
                best_dist = d
                best = det

        if best is not None and best_dist <= max_distance:
            item = dict(best)
            item["track_id"] = track.get("track_id")
            item["face_id"] = track.get("face_id")
            item["shot_id"] = track.get("shot_id")
            item["track_start"] = start
            item["track_end"] = end
            matches.append(item)

    return matches


def raw_observations_at(
    raw_data: dict[str, Any],
    t: float,
    max_distance: float = 0.35,
) -> list[dict[str, Any]]:
    observations = []

    for sample in raw_data.get("face_observations", []):
        if not isinstance(sample, dict) or "time" not in sample:
            continue

        d = abs(float(sample["time"]) - t)
        if d > max_distance:
            continue

        for face in sample.get("faces", []):
            if isinstance(face, dict):
                item = dict(face)
                item["_sample_time"] = float(sample["time"])
                observations.append(item)

    return observations


def match_vision_to_stable_face(
    vision_obs: dict[str, Any],
    stable_faces: list[dict[str, Any]],
    width: float,
    height: float,
) -> tuple[str | None, float]:
    vx = float(vision_obs.get("center_x", 0.0))
    vy = float(vision_obs.get("center_y", 0.0))

    # Vision center values in the raw file are normalized.
    best_id = None
    best_score = -1.0

    vw = float(vision_obs.get("width", 0.0))
    vh = float(vision_obs.get("height", 0.0))

    for sf in stable_faces:
        sx = float(sf.get("center_x", float(sf.get("x", 0.0)) + float(sf.get("width", 0.0)) / 2))
        sy = float(sf.get("center_y", float(sf.get("y", 0.0)) + float(sf.get("height", 0.0)) / 2))

        # Stable face track bboxes are pixel-space; convert center to normalized.
        if width > 1:
            sx /= width
            sy /= height

        dist = center_distance(vx, vy, sx, sy, 1.0, 1.0)

        sw = float(sf.get("width", 0.0)) / max(1.0, width)
        sh = float(sf.get("height", 0.0)) / max(1.0, height)

        size_similarity = 1.0 - clamp01(abs(vw - sw) / max(0.02, sw))
        size_similarity *= 1.0 - 0.5 * clamp01(abs(vh - sh) / max(0.02, sh))

        proximity = clamp01(1.0 - dist / 0.12)
        score = 0.78 * proximity + 0.22 * size_similarity

        if score > best_score:
            best_score = score
            best_id = sf.get("face_id")

    return best_id, clamp01(best_score)


def event_center(event: dict[str, Any]) -> float:
    return (float(event.get("start", 0.0)) + float(event.get("end", 0.0))) / 2.0


def dedupe_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Keep events ordered and merge overlapping candidates when they represent
    # the same editorial beat. Face IDs may differ because the lightweight
    # vision tracker can restart IDs around cuts.
    events = sorted(
        events,
        key=lambda e: (
            float(e.get("start", 0.0)),
            str(e.get("type", "")),
        ),
    )

    merged: list[dict[str, Any]] = []

    for event in events:
        if not merged:
            merged.append(event)
            continue

        prev = merged[-1]

        if event.get("type") != prev.get("type"):
            merged.append(event)
            continue

        a0, a1 = float(prev.get("start", 0.0)), float(prev.get("end", 0.0))
        b0, b1 = float(event.get("start", 0.0)), float(event.get("end", 0.0))

        overlap = interval_overlap(a0, a1, b0, b1)
        gap = interval_distance(a0, a1, b0, b1)

        same_shots = bool(
            set(prev.get("source_shot_ids") or []) &
            set(event.get("source_shot_ids") or [])
        )

        compatible = overlap > 0.15 or (gap <= 0.30 and same_shots)

        if compatible:
            prev["start"] = round(min(a0, b0), 3)
            prev["end"] = round(max(a1, b1), 3)
            prev["score"] = round(
                max(float(prev.get("score", 0.0)), float(event.get("score", 0.0))),
                6,
            )

            for field in ("face_ids", "source_shot_ids"):
                values = []
                for item in (prev.get(field) or []) + (event.get(field) or []):
                    if item not in values:
                        values.append(item)
                prev[field] = values

            reasons = list(prev.get("reason") or [])
            for reason in event.get("reason") or []:
                if reason not in reasons:
                    reasons.append(reason)
            prev["reason"] = reasons

            prev_signals = dict(prev.get("signals") or {})
            for key, value in (event.get("signals") or {}).items():
                if key.startswith("peak_") or key not in prev_signals:
                    prev_signals[key] = value
            prev["signals"] = prev_signals
        else:
            merged.append(event)

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refine clean visual events into reaction-aware editorial events."
    )
    parser.add_argument("--clean", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--stable", required=True)
    parser.add_argument("--faces", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reaction-threshold", type=float, default=0.68)
    parser.add_argument("--active-face-match-threshold", type=float, default=0.60)
    parser.add_argument("--cross-cut-penalty", type=float, default=0.12)
    args = parser.parse_args()

    clean = load_json(args.clean)
    raw = load_json(args.raw)
    stable_data = load_json(args.stable)
    faces_data = load_json(args.faces)

    stable = stable_timeline(stable_data)

    width = float(
        raw.get("video", {}).get("width")
        or clean.get("source", {}).get("width")
        or 1920
    )
    height = float(
        raw.get("video", {}).get("height")
        or clean.get("source", {}).get("height")
        or 1080
    )

    refined: list[dict[str, Any]] = []

    input_events = clean.get("events") or []

    for event in input_events:
        if not isinstance(event, dict):
            continue

        etype = str(event.get("type", ""))
        start = float(event.get("start", 0.0))
        end = float(event.get("end", start))
        center = (start + end) / 2.0

        current = dict(event)

        near = nearest_stable(stable, center)
        active_speaker = None
        active_face = None

        if near:
            active_speaker = near.get("active_speaker") or near.get("speaker")
            active_face = near.get("active_face")

        current["active_speaker"] = active_speaker
        current["active_face_hint"] = active_face

        # ----------------------------------------------------
        # Reaction validation
        # ----------------------------------------------------
        if etype == "reaction_candidate":
            raw_face_ids = list(event.get("face_ids") or [])
            obs = raw_observations_at(raw, center, max_distance=0.40)

            candidate_obs = [
                o for o in obs
                if o.get("face_id") in raw_face_ids
            ]

            if not candidate_obs:
                current["reaction_valid"] = False
                current["reaction_status"] = "missing_visual_observation"
                current["score"] = round(float(event.get("score", 0.0)) * 0.55, 6)
                current["reason"] = list(event.get("reason") or []) + [
                    "could not recover matching visual observation",
                ]
                refined.append(current)
                continue

            best_candidate = max(
                candidate_obs,
                key=lambda o: float(o.get("reaction_score", 0.0)),
            )

            stable_faces = stable_face_detections_at(faces_data, center)

            matched_stable_face, match_score = match_vision_to_stable_face(
                best_candidate,
                stable_faces,
                width,
                height,
            )

            is_active_face = (
                bool(active_face)
                and bool(matched_stable_face)
                and matched_stable_face == active_face
                and match_score >= args.active_face_match_threshold
            )

            same_active_speaker = (
                bool(active_speaker)
                and (
                    event.get("target_speaker") == active_speaker
                    or event.get("active_speaker") == active_speaker
                )
            )

            source_shot_ids = event.get("source_shot_ids") or []
            multi_person_context = (
                str(event.get("source_shot_type", "")) == "two_shot"
                or len(stable_faces) >= 2
                or len(candidate_obs) >= 2
            )

            base_score = float(event.get("score", 0.0))

            # A strong event on the active speaker is speaker activity,
            # not a reaction. This is the critical correction from v1.
            if is_active_face:
                current["reaction_valid"] = False
                current["reaction_status"] = "active_speaker_activity"
                current["reclassified_type"] = "speaker_facial_activity"
                current["matched_stable_face"] = matched_stable_face
                current["face_match_score"] = round(match_score, 6)
                current["score"] = round(base_score * 0.45, 6)
                current["reason"] = list(event.get("reason") or []) + [
                    "candidate maps to active face",
                    "reclassified from reaction to speaker facial activity",
                ]
            elif same_active_speaker and not multi_person_context:
                current["reaction_valid"] = False
                current["reaction_status"] = "speaker_motion_without_multiview_context"
                current["reclassified_type"] = "speaker_facial_activity"
                current["matched_stable_face"] = matched_stable_face
                current["face_match_score"] = round(match_score, 6)
                current["score"] = round(base_score * 0.50, 6)
                current["reason"] = list(event.get("reason") or []) + [
                    "event occurs on a single-speaker context",
                    "not sufficient evidence for a separate reaction cut",
                ]
            elif base_score >= args.reaction_threshold and multi_person_context:
                # A high-score event in a multi-person composition that is
                # NOT the active face is a legitimate reaction candidate.
                current["reaction_valid"] = True
                current["reaction_status"] = "non_active_face_reaction"
                current["matched_stable_face"] = matched_stable_face
                current["face_match_score"] = round(match_score, 6)
                current["target_speaker"] = None
                current["score"] = round(
                    clamp01(base_score + 0.05 * multi_person_context),
                    6,
                )
                current["reason"] = list(event.get("reason") or []) + [
                    "candidate is not mapped to active face",
                    "multi-person visual context supports reaction interpretation",
                ]
            else:
                current["reaction_valid"] = False
                current["reaction_status"] = "insufficient_reaction_evidence"
                current["reclassified_type"] = "speaker_facial_activity"
                current["matched_stable_face"] = matched_stable_face
                current["face_match_score"] = round(match_score, 6)
                current["score"] = round(base_score * 0.65, 6)
                current["reason"] = list(event.get("reason") or []) + [
                    "insufficient evidence for an isolated reaction",
                ]

        # ----------------------------------------------------
        # Speaker facial activity
        # ----------------------------------------------------
        elif etype == "speaker_facial_activity":
            # Keep these, but make their role explicit. They can inform
            # intensity/emotion without automatically forcing a camera cut.
            current["editorial_role"] = "speaker_performance_signal"
            if float(event.get("score", 0.0)) >= 0.80:
                current["reason"] = list(event.get("reason") or []) + [
                    "high speaker-expression signal available to director",
                ]

        # ----------------------------------------------------
        # Inserts / framing / source cuts
        # ----------------------------------------------------
        elif etype == "visual_insert":
            current["editorial_role"] = "possible_insert"

        elif etype == "framing_change":
            current["editorial_role"] = "source_composition_change"

        elif etype == "shot_boundary":
            current["editorial_role"] = "source_cut_candidate"

        refined.append(current)

    # Reclassifications become their new effective type.
    for event in refined:
        if event.get("reclassified_type"):
            event["original_type"] = event.get("type")
            event["type"] = event["reclassified_type"]
            event.pop("reclassified_type", None)

    # Drop weak invalid reaction candidates from the editorial event stream.
    filtered: list[dict[str, Any]] = []
    for event in refined:
        if event.get("original_type") == "reaction_candidate":
            if not event.get("reaction_valid", False):
                # Preserve strong transformed speaker activity; weak noise can go.
                if event.get("type") != "speaker_facial_activity":
                    continue
        filtered.append(event)

    filtered = dedupe_events(filtered)

    # Add a normalized reaction priority useful to the sequence director.
    for event in filtered:
        if event.get("type") == "reaction_candidate":
            score = float(event.get("score", 0.0))
            event["reaction_priority"] = round(
                clamp01(score * 0.75 + 0.25),
                6,
            )
        elif event.get("type") == "speaker_facial_activity":
            event["reaction_priority"] = 0.0

    filtered.sort(key=lambda e: (float(e.get("start", 0.0)), -float(e.get("score", 0.0))))

    summary = {
        "input_events": len(input_events),
        "output_events": len(filtered),
        "reaction_candidates": sum(
            1 for e in filtered if e.get("type") == "reaction_candidate"
        ),
        "valid_reactions": sum(
            1 for e in filtered
            if e.get("type") == "reaction_candidate"
            and e.get("reaction_valid")
        ),
        "speaker_facial_activity": sum(
            1 for e in filtered if e.get("type") == "speaker_facial_activity"
        ),
        "visual_inserts": sum(
            1 for e in filtered if e.get("type") == "visual_insert"
        ),
        "framing_changes": sum(
            1 for e in filtered if e.get("type") == "framing_change"
        ),
        "shot_boundaries": sum(
            1 for e in filtered if e.get("type") == "shot_boundary"
        ),
        "strong_reactions": sum(
            1 for e in filtered
            if e.get("type") == "reaction_candidate"
            and float(e.get("score", 0.0)) >= 0.75
        ),
    }

    output = {
        "version": "1.0",
        "mode": "refined_editorial_events",
        "sources": {
            "clean_events": args.clean,
            "raw_visual_reactions": args.raw,
            "stable_speaker_face": args.stable,
            "face_tracks": args.faces,
        },
        "policy": {
            "reaction_requires_non_active_face": True,
            "multi_person_context_required_for_ambiguous_reaction": True,
            "active_speaker_motion_can_be_reclassified": True,
        },
        "summary": summary,
        "events": filtered,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(json.dumps({
        "output": args.output,
        **summary,
    }, indent=2))


if __name__ == "__main__":
    main()
