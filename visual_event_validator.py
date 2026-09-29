#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


REACTION_MIN = 0.58
REACTION_STRONG = 0.78
MERGE_GAP = 0.50
MIN_DURATION = 0.40
SPEAKER_PERSISTENCE_PENALTY = 0.18


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return obj


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def nearest(items: list[dict[str, Any]], t: float, max_distance: float) -> dict[str, Any] | None:
    if not items:
        return None
    item = min(items, key=lambda x: abs(float(x.get("time", 0.0)) - t))
    if abs(float(item.get("time", 0.0)) - t) > max_distance:
        return None
    return item


def event_center(event: dict[str, Any]) -> float:
    return (float(event["start"]) + float(event["end"])) / 2.0


def shot_boundaries(events: list[dict[str, Any]]) -> list[float]:
    return sorted(
        float(e["start"])
        for e in events
        if e.get("type") == "shot_boundary"
    )


def crosses_cut(start: float, end: float, cuts: list[float]) -> bool:
    return any(start < c < end for c in cuts)


def dedupe_reaction_events(
    raw_reactions: list[dict[str, Any]],
    cuts: list[float],
) -> list[dict[str, Any]]:
    # First pass: discard events that are mostly the ordinary active speaker.
    # We do NOT call these true reactions. They remain useful as facial activity.
    candidates: list[dict[str, Any]] = []

    for e in raw_reactions:
        score = float(e.get("score", 0.0))
        if score < REACTION_MIN:
            continue

        start = float(e["start"])
        end = float(e["end"])
        duration = max(0.0, end - start)

        active_speaker = e.get("active_speaker")
        target_speaker = e.get("target_speaker")

        # The v1 resolver is not face-specific: target_speaker is often simply
        # the currently active speaker. Such an event is evidence of facial
        # activity, not proof of a listener reaction.
        likely_speaker_facial_activity = (
            target_speaker is not None
            and active_speaker is not None
            and target_speaker == active_speaker
        )

        adjusted = score
        reasons = list(e.get("reason") or [])

        if likely_speaker_facial_activity:
            adjusted *= (1.0 - SPEAKER_PERSISTENCE_PENALTY)
            reasons.append("likely active-speaker facial activity, not isolated reaction")

        if crosses_cut(start, end, cuts):
            adjusted *= 0.88
            reasons.append("crosses source cut; confidence reduced")

        if duration > 2.0:
            adjusted *= 0.92
            reasons.append("long facial event; confidence reduced")

        item = dict(e)
        item["_adjusted_score"] = round(clamp01(adjusted), 6)
        item["_likely_speaker_activity"] = likely_speaker_facial_activity
        item["_reasons_internal"] = reasons
        candidates.append(item)

    # Group overlapping events from multiple vision IDs into one editorial beat.
    candidates.sort(key=lambda x: (float(x["start"]), -float(x["_adjusted_score"])))

    groups: list[list[dict[str, Any]]] = []

    for e in candidates:
        placed = False
        for group in reversed(groups):
            latest_end = max(float(x["end"]) for x in group)
            earliest_start = min(float(x["start"]) for x in group)

            if float(e["start"]) <= latest_end + MERGE_GAP:
                # Don't merge unrelated far-apart events just because they
                # happen during the same broad shot.
                if float(e["start"]) - earliest_start <= 2.5:
                    group.append(e)
                    placed = True
                    break

            if latest_end + MERGE_GAP < float(e["start"]):
                break

        if not placed:
            groups.append([e])

    merged: list[dict[str, Any]] = []

    for group in groups:
        start = min(float(e["start"]) for e in group)
        end = max(float(e["end"]) for e in group)
        peak = max(group, key=lambda e: float(e["_adjusted_score"]))

        # Require either persistence or multiple independent visual tracks.
        duration = end - start
        unique_faces = len({
            str(e.get("face_id"))
            for e in group
            if e.get("face_id") is not None
        })

        persistence = clamp01(duration / 1.0)
        multiplicity = clamp01(unique_faces / 2.0)

        # Strong evidence should not be watered down too much.
        score = (
            0.72 * float(peak["_adjusted_score"])
            + 0.18 * persistence
            + 0.10 * multiplicity
        )

        if score < REACTION_MIN:
            continue

        likely_active = all(bool(e["_likely_speaker_activity"]) for e in group)
        if likely_active:
            event_type = "speaker_facial_activity"
        else:
            event_type = "reaction_candidate"

        target_speakers = [
            e.get("target_speaker")
            for e in group
            if e.get("target_speaker")
        ]
        active_speakers = [
            e.get("active_speaker")
            for e in group
            if e.get("active_speaker")
        ]

        chosen_speaker = target_speakers[0] if target_speakers else None
        active_speaker = active_speakers[0] if active_speakers else None

        reasons = []
        for e in group:
            for r in e.get("_reasons_internal", []):
                if r not in reasons:
                    reasons.append(r)

        if unique_faces > 1:
            reasons.append("multiple independent visual tracks support the beat")

        if duration >= 0.75:
            reasons.append("visual behavior persists across multiple samples")

        if score >= REACTION_STRONG:
            reasons.append("strong editorial candidate")

        source_shots = sorted({
            str(e.get("source_shot_id"))
            for e in group
            if e.get("source_shot_id")
        })

        merged.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "type": event_type,
            "score": round(clamp01(score), 6),
            "face_ids": sorted({
                str(e.get("face_id"))
                for e in group
                if e.get("face_id")
            }),
            "target_speaker": chosen_speaker,
            "active_speaker": active_speaker,
            "source_shot_ids": source_shots,
            "source_shot_type": peak.get("source_shot_type"),
            "signals": {
                "peak_reaction_score": round(
                    max(float(e.get("signals", {}).get("reaction_score", 0.0)) for e in group),
                    6,
                ),
                "peak_facial_motion_score": round(
                    max(float(e.get("signals", {}).get("facial_motion_score", 0.0)) for e in group),
                    6,
                ),
                "persistence": round(persistence, 6),
                "face_multiplicity": round(multiplicity, 6),
                "raw_event_count": len(group),
            },
            "reason": reasons,
        })

    return merged


def build_clean_visual_events(
    data: dict[str, Any],
    stable_data: dict[str, Any] | None,
) -> dict[str, Any]:
    raw_events = [
        e for e in (data.get("events") or [])
        if isinstance(e, dict)
    ]

    cuts = shot_boundaries(raw_events)
    raw_reactions = [
        e for e in raw_events
        if e.get("type") == "reaction_candidate"
    ]

    clean_reactions = dedupe_reaction_events(raw_reactions, cuts)

    # Keep inserts and framing changes, but remove duplicate shot-boundary noise.
    inserts = [
        {
            "start": float(e["start"]),
            "end": float(e["end"]),
            "type": "visual_insert",
            "score": float(e.get("score", 0.0)),
            "source_shot_id": e.get("source_shot_id") or e.get("shot_id"),
            "source_shot_type": e.get("source_shot_type") or e.get("shot_type"),
            "reason": list(e.get("reason") or []),
        }
        for e in raw_events
        if e.get("type") == "visual_insert"
    ]

    framing = [
        {
            "start": float(e["start"]),
            "end": float(e["end"]),
            "type": "framing_change",
            "score": float(e.get("score", 0.0)),
            "source_shot_id": e.get("source_shot_id"),
            "source_shot_type": e.get("source_shot_type"),
            "reason": list(e.get("reason") or []),
        }
        for e in raw_events
        if e.get("type") == "framing_change"
    ]

    editorial_events = clean_reactions + inserts + framing
    editorial_events.sort(key=lambda e: (float(e["start"]), -float(e.get("score", 0.0))))

    # Optional stable speaker context.
    stable = []
    if stable_data:
        stable = stable_data.get("timeline") or stable_data.get("samples") or []

    enriched = []
    for e in editorial_events:
        item = dict(e)
        center = event_center(e)
        near = nearest(stable, center, 0.75)
        if near:
            item["nearby_active_speaker"] = (
                near.get("active_speaker") or near.get("speaker")
            )
            item["nearby_active_face"] = near.get("active_face")
        enriched.append(item)

    return {
        "version": "1.0",
        "mode": "clean_visual_editorial_events",
        "source": data.get("video"),
        "window": data.get("window"),
        "input_event_count": len(raw_events),
        "input_reaction_count": len(raw_reactions),
        "output_event_count": len(enriched),
        "summary": {
            "reaction_candidates": sum(
                1 for e in enriched if e.get("type") == "reaction_candidate"
            ),
            "speaker_facial_activity": sum(
                1 for e in enriched if e.get("type") == "speaker_facial_activity"
            ),
            "visual_inserts": sum(
                1 for e in enriched if e.get("type") == "visual_insert"
            ),
            "framing_changes": sum(
                1 for e in enriched if e.get("type") == "framing_change"
            ),
            "strong_reaction_candidates": sum(
                1
                for e in enriched
                if e.get("type") == "reaction_candidate"
                and float(e.get("score", 0.0)) >= REACTION_STRONG
            ),
        },
        "events": enriched,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and merge visual reaction events")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stable", default=None)
    args = parser.parse_args()

    data = load_json(args.input)
    stable = load_json(args.stable) if args.stable else None

    payload = build_clean_visual_events(data, stable)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps({
        "output": args.output,
        "input_events": payload["input_event_count"],
        "input_reactions": payload["input_reaction_count"],
        **payload["summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
