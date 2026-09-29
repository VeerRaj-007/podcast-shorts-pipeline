#!/usr/bin/env python3
"""
Editorial Beat Planner V2

Build conservative editorial intent beats from transcript + visual events.

Key V2 fix:
- Visual-event beats use TIMESTAMP-LOCAL transcript context rather than the
  entire transcript segment.
- Word-level timestamps are preferred when available.
- Segment-level text is used only as a fallback.
- Existing editorial policy remains conservative: visual interest alone does
  not authorize an insert.
- The final sequence director remains authoritative.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

VERSION = "2.0"

TRANSCRIPT_MERGE_GAP = 0.85
SPEECH_CONTEXT_RADIUS = 0.45
INSERT_CONTEXT_PRE = 0.90
INSERT_CONTEXT_POST = 0.45
REACTION_HOLD = 1.25
MIN_SPEAKER_HOLD = 0.75
MIN_INSERT_HOLD = 0.70
INSERT_COOLDOWN = 3.50
FRAMING_MIN_SCORE = 0.68
REACTION_MIN_SCORE = 0.72
STRONG_REACTION_SCORE = 0.86

VISUAL_CUES = {
    "look", "see", "watch", "show", "shows", "showing", "here", "there",
    "this", "that", "these", "those", "picture", "photo", "video", "screen",
    "camera", "edit", "editing", "clip", "logo", "game", "gameplay", "map",
    "menu", "button", "website", "page", "image",
}

REACTION_TERMS = {
    "what", "wow", "damn", "shit", "fuck", "omg", "oh", "god", "seriously",
    "no", "why", "bro", "jesus", "crazy", "insane", "haha", "laugh",
}

QUESTION_STARTS = {
    "what", "why", "how", "where", "when", "who", "which", "are", "is", "do",
    "did", "can", "could", "would", "will",
}

FILLER_PATTERNS = re.compile(
    r"\b(?:um+|uh+|erm+|like|you know|okay okay|yeah yeah)\b",
    re.IGNORECASE,
)


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out)


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalize_text(text: Any) -> str:
    value = str(text or "")
    return re.sub(r"\s+", " ", value).strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def transcript_density(text: str, duration: float) -> float:
    if duration <= 0.0:
        return 0.0
    words = len(tokenize(text))
    wpm = words / duration * 60.0
    return clamp01(wpm / 180.0)


def clean_filler_ratio(text: str) -> float:
    words = tokenize(text)
    if not words:
        return 0.0
    fillers = len(FILLER_PATTERNS.findall(text))
    return clamp01(fillers / max(1, len(words) / 6.0))


def visual_cue_score(text: str) -> float:
    words = tokenize(text)
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in VISUAL_CUES)
    return clamp01((hits / max(1, len(words))) * 5.0)


def question_score(text: str) -> float:
    words = tokenize(text)
    if not words:
        return 0.0
    starts = sum(1 for w in words[:3] if w in QUESTION_STARTS)
    qmark = 1.0 if "?" in text else 0.0
    return clamp01(0.55 * qmark + 0.45 * (starts / 2.0))


def excitement_score(text: str) -> float:
    words = tokenize(text)
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in REACTION_TERMS)
    repeats = 0
    for i in range(1, len(words)):
        if words[i] == words[i - 1] and words[i] in REACTION_TERMS:
            repeats += 1
    return clamp01(0.25 * min(1.0, hits / 3.0) + 0.75 * min(1.0, repeats / 2.0))


def normalize_words(words: Any) -> list[dict[str, Any]]:
    if not isinstance(words, list):
        return []
    out: list[dict[str, Any]] = []
    for word in words:
        if not isinstance(word, dict):
            continue
        text = normalize_text(word.get("word", word.get("text", "")))
        if not text:
            continue
        ws = fnum(word.get("start", word.get("start_time")), math.nan)
        we = fnum(word.get("end", word.get("end_time")), math.nan)
        if not math.isfinite(ws) or not math.isfinite(we) or we <= ws:
            continue
        out.append({"start": ws, "end": we, "text": text})
    return out


def normalize_transcript_segments(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw = data.get("segments", data.get("items", []))
    out: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return out
    for i, seg in enumerate(raw):
        if not isinstance(seg, dict):
            continue
        start = fnum(seg.get("start", seg.get("start_time")), 0.0)
        end = fnum(seg.get("end", seg.get("end_time")), start)
        if end <= start:
            continue
        out.append({
            "index": i,
            "start": start,
            "end": end,
            "text": normalize_text(seg.get("text")),
            "speaker": seg.get("speaker") or seg.get("speaker_id"),
            "words": normalize_words(seg.get("words")),
        })
    out.sort(key=lambda x: (x["start"], x["end"]))
    return out


def merge_transcript_beats(
    segments: list[dict[str, Any]], start: float, end: float
) -> list[dict[str, Any]]:
    selected = [s for s in segments if s["end"] >= start and s["start"] <= end]
    if not selected:
        return []

    groups: list[dict[str, Any]] = []
    for seg in selected:
        if not groups:
            groups.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "segments": [seg["index"]],
                "speakers": [seg["speaker"]] if seg.get("speaker") else [],
            })
            continue

        prev = groups[-1]
        gap = seg["start"] - prev["end"]
        if gap <= TRANSCRIPT_MERGE_GAP:
            prev["end"] = max(prev["end"], seg["end"])
            if seg["text"]:
                prev["text"] = (prev["text"] + " " + seg["text"]).strip()
            prev["segments"].append(seg["index"])
            if seg.get("speaker") and seg["speaker"] not in prev["speakers"]:
                prev["speakers"].append(seg["speaker"])
        else:
            groups.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "segments": [seg["index"]],
                "speakers": [seg["speaker"]] if seg.get("speaker") else [],
            })

    return [g for g in groups if g["end"] >= start and g["start"] <= end]


def normalize_events(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw = data.get("events", data.get("output_events", data.get("items", [])))
    out: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return out

    for i, event in enumerate(raw):
        if not isinstance(event, dict):
            continue
        start = fnum(event.get("start", event.get("start_time")), 0.0)
        end = fnum(event.get("end", event.get("end_time")), start)
        if end <= start:
            continue
        out.append({
            "id": i,
            "start": start,
            "end": end,
            "center": (start + end) / 2.0,
            "type": str(event.get("type", event.get("event_type", "unknown"))),
            "score": clamp01(fnum(event.get("score", event.get("confidence", 0.0)))),
            "target_speaker": event.get("target_speaker"),
            "active_speaker": event.get("active_speaker") or event.get("nearby_active_speaker"),
            "source_shot_id": event.get("source_shot_id"),
            "source_shot_type": event.get("source_shot_type"),
            "reason": list(event.get("reason") or []),
            "raw": event,
        })
    out.sort(key=lambda x: (x["start"], -x["score"]))
    return out


def normalize_shots(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = data.get("shots", data.get("items", []))
    result: dict[str, dict[str, Any]] = {}
    if not isinstance(raw, list):
        return result
    for shot in raw:
        if not isinstance(shot, dict):
            continue
        sid = str(shot.get("shot_id", shot.get("id", "")))
        if not sid:
            continue
        start = fnum(shot.get("start"), 0.0)
        end = fnum(shot.get("end"), start)
        result[sid] = {
            **shot,
            "shot_id": sid,
            "start": start,
            "end": end,
            "duration": max(0.0, end - start),
            "shot_type": str(shot.get("shot_type", "unknown")),
        }
    return result


def segments_for_window(
    transcripts: list[dict[str, Any]], start: float, end: float, radius: float
) -> list[dict[str, Any]]:
    lo = max(0.0, start - radius)
    hi = end + radius
    return [s for s in transcripts if s["end"] >= lo and s["start"] <= hi]


def local_context(
    transcripts: list[dict[str, Any]],
    start: float,
    end: float,
    pre: float,
    post: float,
) -> tuple[str, list[int]]:
    """Return words actually spoken near this visual event.

    Word timestamps are preferred. A tiny segment-level fallback is retained
    for sources without word timing, but the fallback is clipped to segments
    overlapping the local window instead of returning the whole transcript.
    """
    lo = max(0.0, start - max(0.0, pre))
    hi = end + max(0.0, post)

    selected = segments_for_window(transcripts, start, end, 0.0)
    if not selected:
        selected = segments_for_window(transcripts, start, end, max(pre, post))

    word_items: list[tuple[float, int, str]] = []
    for seg in selected:
        for word in seg.get("words", []):
            if word["end"] >= lo and word["start"] <= hi:
                center = (word["start"] + word["end"]) / 2.0
                word_items.append((center, seg["index"], word["text"]))

    if word_items:
        word_items.sort(key=lambda x: x[0])
        return " ".join(x[2] for x in word_items).strip(), sorted({x[1] for x in word_items})

    fallback = [
        seg for seg in transcripts
        if seg["end"] >= lo and seg["start"] <= hi
    ]
    fallback.sort(key=lambda x: (x["start"], x["end"]))
    return " ".join(x["text"] for x in fallback if x.get("text")).strip(), [x["index"] for x in fallback]


def transcript_context_text(items: list[dict[str, Any]]) -> str:
    return " ".join(x["text"] for x in items if x.get("text")).strip()


def has_semantic_visual_metadata(event: dict[str, Any], shot: dict[str, Any] | None) -> bool:
    candidates: list[dict[str, Any]] = [event.get("raw", {})]
    if shot:
        candidates.append(shot)
    semantic_keys = {
        "description", "caption", "visual_description", "visual_tags",
        "tags", "objects", "entities", "ocr_text", "scene_description",
        "semantic_labels", "content_labels",
    }
    for obj in candidates:
        for key in semantic_keys:
            value = obj.get(key)
            if value:
                return True
    return False


def event_semantic_relation(
    event: dict[str, Any], transcript_text: str, shot: dict[str, Any] | None
) -> tuple[str, float, list[str]]:
    et = event["type"].lower()
    score = event["score"]
    reasons: list[str] = []
    cue = visual_cue_score(transcript_text)
    excitement = excitement_score(transcript_text)
    question = question_score(transcript_text)
    semantic_metadata = has_semantic_visual_metadata(event, shot)

    different_person = (
        event.get("target_speaker")
        and event.get("active_speaker")
        and str(event["target_speaker"]) != str(event["active_speaker"])
    )

    if et == "reaction_candidate":
        if different_person and score >= REACTION_MIN_SCORE:
            reasons.append("target speaker differs from active speaker")
            if score >= STRONG_REACTION_SCORE:
                reasons.append("strong reaction confidence")
            return "reaction", clamp01(0.60 * score + 0.40 * excitement), reasons
        reasons.append("reaction candidate is not independently speaker-distinct")
        return "speaker_emphasis", clamp01(0.45 * score + 0.30 * excitement + 0.25 * question), reasons

    if et == "speaker_facial_activity":
        reasons.append("facial activity belongs to the active speaker")
        return "speaker_emphasis", clamp01(0.65 * score + 0.35 * excitement), reasons

    if et == "visual_insert":
        if semantic_metadata:
            reasons.append("visual event carries semantic metadata")
            if cue > 0.0:
                reasons.append("spoken context contains visual-reference cues")
            return "supporting_insert_candidate", clamp01(0.50 * score + 0.25 * cue + 0.25), reasons
        reasons.append("insert has only structural non-face evidence")
        if cue > 0.25:
            reasons.append("spoken context contains a visual-reference cue, but source content is unknown")
        return "supporting_insert_candidate", clamp01(0.20 * score + 0.35 * cue), reasons

    if et == "framing_change":
        if score >= FRAMING_MIN_SCORE:
            reasons.append("large framing/scene change")
            return "transition_candidate", clamp01(0.65 * score + 0.35 * cue), reasons
        reasons.append("weak framing change")
        return "reset", clamp01(0.35 * score), reasons

    return "reset", clamp01(0.25 * score), ["event type not treated as a direct editorial switch"]


def should_emit_insert(
    beat_start: float,
    last_insert_time: float | None,
    relation_confidence: float,
    has_metadata: bool,
) -> tuple[bool, str]:
    if not has_metadata:
        return False, "no semantic description/tags exist for the source insert"
    if relation_confidence < 0.65:
        return False, "semantic relation confidence below insert threshold"
    if last_insert_time is not None and beat_start - last_insert_time < INSERT_COOLDOWN:
        return False, "insert cooldown prevents visual over-editing"
    return True, "semantic visual relation is strong enough"


def build_beats(
    transcripts: list[dict[str, Any]],
    events: list[dict[str, Any]],
    shots_by_id: dict[str, dict[str, Any]],
    start: float,
    end: float,
) -> dict[str, Any]:
    speech_groups = merge_transcript_beats(transcripts, start, end)
    visual_events = [e for e in events if e["end"] >= start and e["start"] <= end]

    beats: list[dict[str, Any]] = []
    last_insert_time: float | None = None

    # Baseline speech beats stay segment-level: they describe the full dialogue
    # state and are not used as the semantic source for an individual insert.
    for group in speech_groups:
        gs = max(start, group["start"])
        ge = min(end, group["end"])
        if ge <= gs:
            continue
        text = group["text"]
        duration = ge - gs
        density = transcript_density(text, duration)
        excitement = excitement_score(text)
        question = question_score(text)
        fillers = clean_filler_ratio(text)
        cue = visual_cue_score(text)

        intent = "speaker_hold"
        confidence = clamp01(0.55 + 0.20 * density + 0.15 * (1.0 - fillers) + 0.10 * excitement)
        reason = ["default dialogue coverage"]

        if excitement >= 0.65 or question >= 0.75:
            intent = "speaker_emphasis"
            confidence = clamp01(confidence + 0.10)
            reason.append("speech contains elevated rhetorical/emotional energy")
        elif cue >= 0.55:
            intent = "speaker_emphasis"
            confidence = clamp01(confidence + 0.05)
            reason.append("speech contains an explicit visual-reference cue")

        beats.append({
            "start": round(gs, 3),
            "end": round(ge, 3),
            "intent": intent,
            "confidence": round(confidence, 6),
            "speaker": group["speakers"][0] if len(group["speakers"]) == 1 else None,
            "speech_excerpt": text,
            "transcript_segments": group["segments"],
            "allowed_profiles": ["medium", "close", "two_shot", "wide"] if intent == "speaker_emphasis" else ["medium", "two_shot", "wide", "close"],
            "insert_policy": "conditional",
            "reason": reason,
            "source_event_ids": [],
        })

    # Overlay visual-event beats using ONLY local speech context.
    for event in visual_events:
        es = max(start, event["start"])
        ee = min(end, event["end"])
        if ee <= es:
            continue

        et = event["type"].lower()
        if et == "visual_insert":
            text, ctx_ids = local_context(transcripts, es, ee, INSERT_CONTEXT_PRE, INSERT_CONTEXT_POST)
        else:
            text, ctx_ids = local_context(transcripts, es, ee, SPEECH_CONTEXT_RADIUS, SPEECH_CONTEXT_RADIUS)

        shot = shots_by_id.get(str(event.get("source_shot_id"))) if event.get("source_shot_id") else None
        intent, confidence, relation_reasons = event_semantic_relation(event, text, shot)

        has_meta = has_semantic_visual_metadata(event, shot)
        emit_insert = False
        insert_gate_reason = None
        if intent == "supporting_insert_candidate":
            emit_insert, insert_gate_reason = should_emit_insert(
                es, last_insert_time, confidence, has_meta
            )
            # Preserve V1/V2 conservative behavior: the planner does not consume
            # external shot-content metadata, so semantic matcher stays authoritative.
            emit_insert = False
            insert_gate_reason = "insert emission deferred to semantic matcher"
            intent = "supporting_insert_candidate"

        if intent == "reaction":
            ee = min(end, max(ee, es + REACTION_HOLD))
        elif intent == "supporting_insert":
            ee = min(end, max(ee, es + MIN_INSERT_HOLD))
        elif intent == "speaker_emphasis":
            ee = min(end, max(ee, es + MIN_SPEAKER_HOLD))

        reasons = relation_reasons[:]
        if insert_gate_reason:
            reasons.append(insert_gate_reason)
        if text:
            reasons.append("timestamp-local speech context used for visual intent")
        else:
            reasons.append("no timestamp-local speech context available")

        beats.append({
            "start": round(es, 3),
            "end": round(ee, 3),
            "intent": intent,
            "confidence": round(confidence, 6),
            "speaker": event.get("target_speaker") or event.get("active_speaker"),
            "speech_excerpt": text,
            "transcript_segments": ctx_ids,
            "allowed_profiles": (
                ["insert"] if intent == "supporting_insert"
                else ["close", "medium", "two_shot", "wide"] if intent in {"speaker_emphasis", "reaction"}
                else ["wide", "two_shot", "medium", "close"]
            ),
            "insert_policy": (
                "allowed" if intent == "supporting_insert"
                else "candidate_only" if intent == "supporting_insert_candidate"
                else "forbidden"
            ),
            "event_type": event["type"],
            "source_event_ids": [event["id"]],
            "source_shot_id": event.get("source_shot_id"),
            "source_shot_type": event.get("source_shot_type"),
            "reason": reasons,
            "metadata_available": has_meta,
        })

    beats.sort(key=lambda b: (b["start"], -b["confidence"]))
    beats = coalesce_beats(beats)

    return {
        "version": VERSION,
        "mode": "editorial_beat_planner_v2",
        "window": {"start": start, "end": end, "duration": max(0.0, end - start)},
        "policy": {
            "speech_is_default_coverage": True,
            "insert_requires_semantic_evidence": True,
            "structural_non_face_insert_is_not_semantic_proof": True,
            "reaction_requires_speaker_distinction_when_available": True,
            "insert_cooldown_seconds": INSERT_COOLDOWN,
            "reaction_hold_seconds": REACTION_HOLD,
            "min_speaker_hold_seconds": MIN_SPEAKER_HOLD,
            "min_insert_hold_seconds": MIN_INSERT_HOLD,
            "timestamp_local_visual_context": True,
            "insert_context_pre_seconds": INSERT_CONTEXT_PRE,
            "insert_context_post_seconds": INSERT_CONTEXT_POST,
            "word_timestamps_preferred": True,
            "semantic_matcher_remains_authoritative": True,
            "future_source_selection": "deferred_to_sequence_director",
        },
        "stats": {
            "speech_groups": len(speech_groups),
            "visual_events_considered": len(visual_events),
            "beats": len(beats),
            "speaker_hold": sum(b["intent"] == "speaker_hold" for b in beats),
            "speaker_emphasis": sum(b["intent"] == "speaker_emphasis" for b in beats),
            "reaction": sum(b["intent"] == "reaction" for b in beats),
            "supporting_insert": sum(b["intent"] == "supporting_insert" for b in beats),
            "supporting_insert_candidate": sum(b["intent"] == "supporting_insert_candidate" for b in beats),
            "transition_candidate": sum(b["intent"] == "transition_candidate" for b in beats),
            "reset": sum(b["intent"] == "reset" for b in beats),
        },
        "beats": beats,
    }


def coalesce_beats(beats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not beats:
        return []

    result: list[dict[str, Any]] = []
    for beat in beats:
        if not result:
            result.append(beat)
            continue
        prev = result[-1]
        same_intent = prev["intent"] == beat["intent"]
        touching = beat["start"] <= prev["end"] + 0.05
        same_shot = prev.get("source_shot_id") == beat.get("source_shot_id")
        both_speech = (
            prev["intent"] in {"speaker_hold", "speaker_emphasis"}
            and beat["intent"] in {"speaker_hold", "speaker_emphasis"}
        )
        same_event_type = prev.get("event_type") == beat.get("event_type")

        # Never merge two different visual insert events merely because they touch.
        # Preserve individual insert beats for the semantic matcher.
        mergeable = touching and (
            same_intent and same_shot and same_event_type
            or (both_speech and same_shot and not prev.get("source_event_ids") and not beat.get("source_event_ids"))
        )

        if mergeable:
            prev["end"] = max(prev["end"], beat["end"])
            prev["confidence"] = round(max(prev["confidence"], beat["confidence"]), 6)
            prev["transcript_segments"] = sorted(set(prev.get("transcript_segments", [])) | set(beat.get("transcript_segments", [])))
            prev["source_event_ids"] = sorted(set(prev.get("source_event_ids", [])) | set(beat.get("source_event_ids", [])))
            for reason in beat.get("reason", []):
                if reason not in prev["reason"]:
                    prev["reason"].append(reason)
        else:
            result.append(beat)

    for beat in result:
        beat["start"] = round(float(beat["start"]), 3)
        beat["end"] = round(float(beat["end"]), 3)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build timestamp-local editorial intent beats from transcript + visual events")
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--shots", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, required=True)
    args = parser.parse_args()

    if args.end <= args.start:
        raise ValueError("--end must be greater than --start")

    transcript_data = load_json(args.transcript)
    event_data = load_json(args.events)
    shot_data = load_json(args.shots)

    transcripts = normalize_transcript_segments(transcript_data)
    events = normalize_events(event_data)
    shots_by_id = normalize_shots(shot_data)

    result = build_beats(transcripts, events, shots_by_id, args.start, args.end)
    save_json(args.output, result)

    print("=== EDITORIAL BEAT PLANNER V2 ===")
    print(json.dumps({
        "output": args.output,
        "window": result["window"],
        "stats": result["stats"],
        "policy": result["policy"],
    }, indent=2))

    print("\nInsert/context decisions:")
    for b in result["beats"]:
        if b["intent"] in {"supporting_insert", "supporting_insert_candidate"}:
            excerpt = b.get("speech_excerpt", "")
            print(
                f"  {b['start']:.3f}-{b['end']:.3f} | "
                f"{b['intent']} | conf={b['confidence']:.2f} | "
                f"shot={b.get('source_shot_id')} | "
                f'speech="{excerpt}"'
            )


if __name__ == "__main__":
    main()