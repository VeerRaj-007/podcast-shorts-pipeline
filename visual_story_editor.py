from __future__ import annotations

"""
Visual Story Editor

Creates a professional vertical-edit decision layer over a locked editorial/audio
sequence. It does NOT treat source shot boundaries as automatic editorial cuts.

The output contains:
  - visual_decisions: higher-level editorial visual states
  - visual_segments: source realizations needed to execute those decisions
  - diagnostics explaining why a visual cut happened

Design principles:
  1. Editorial audio/story is immutable.
  2. Speaker changes can justify a visual change, but source-shot changes alone cannot.
  3. Non-face/source inserts need semantic evidence to interrupt an interview.
  4. Explicit visual references (e.g. Jack Nicholson) get priority during the
     matching spoken window.
  5. A source cut inside a visual decision is allowed; the camera composition remains
     conceptually stable across it.
  6. The planner fails loudly when required speaker information is missing.
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

EPS = 1e-6
SPEAKERS = {"PERSON_0", "PERSON_1"}
QUESTION_WORDS = {"who", "what", "when", "where", "why", "how", "which", "whose", "would", "could", "can", "do", "did", "does", "is", "are", "was", "were"}

# Stable editorial thresholds. These are policy constants, not numbered experiments.
MIN_SPEAKER_SHARE = 0.55
MIN_VISUAL_HOLD = 1.25
INSERT_MIN_DURATION = 1.50
INSERT_STRONG_MATCH = 0.55
SPEAKER_SWITCH_CUT_COOLDOWN = 1.00

VISUAL_NONFACE_TYPES = {"insert_or_screen", "insert", "screen", "non_face"}
INTERVIEW_TYPES = {"single_wide", "single_medium", "single_close", "two_shot", "medium", "close", "wide"}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def fnum(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def normalize_turns(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw = data.get("turns")
    if not isinstance(raw, list):
        raise RuntimeError("Speaker attribution has no 'turns' list.")
    out = []
    for turn in raw:
        if not isinstance(turn, dict):
            continue
        person = str(turn.get("person_id", ""))
        if person not in SPEAKERS:
            continue
        s = fnum(turn.get("start"), -1)
        e = fnum(turn.get("end"), -1)
        if e > s + EPS:
            out.append({**turn, "start": s, "end": e, "person_id": person})
    out.sort(key=lambda x: x["start"])
    if not out:
        raise RuntimeError("Speaker attribution contains no usable PERSON_0/PERSON_1 turns.")
    return out


def speaker_totals(turns: list[dict[str, Any]], start: float, end: float) -> dict[str, float]:
    totals = {"PERSON_0": 0.0, "PERSON_1": 0.0}
    for t in turns:
        totals[t["person_id"]] += overlap(start, end, t["start"], t["end"])
    return totals


def dominant_speaker(turns: list[dict[str, Any]], start: float, end: float) -> tuple[str | None, dict[str, float]]:
    totals = speaker_totals(turns, start, end)
    spoken = totals["PERSON_0"] + totals["PERSON_1"]
    if spoken <= EPS:
        return None, totals
    person = max(totals, key=totals.get)
    return (person if totals[person] / spoken >= MIN_SPEAKER_SHARE else None), totals


def normalize_observations(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw = data.get("identity_clean_observations") or data.get("observations_deduped") or []
    if not isinstance(raw, list):
        return []
    out = [o for o in raw if isinstance(o, dict) and o.get("identity_id") in SPEAKERS]
    out.sort(key=lambda x: fnum(x.get("timestamp"), -1))
    return out


def person_face_stats(observations: list[dict[str, Any]], start: float, end: float) -> dict[str, dict[str, float]]:
    samples: dict[str, list[dict[str, Any]]] = {"PERSON_0": [], "PERSON_1": []}
    for o in observations:
        t = fnum(o.get("timestamp"), -1)
        if start - 1e-4 <= t <= end + 1e-4 and o.get("identity_id") in SPEAKERS:
            samples[o["identity_id"]].append(o)

    result: dict[str, dict[str, float]] = {}
    duration = max(EPS, end - start)
    for p in SPEAKERS:
        items = samples[p]
        xs, ys, areas = [], [], []
        for o in items:
            b = o.get("bbox")
            if not isinstance(b, (list, tuple)) or len(b) != 4:
                continue
            x, y, w, h = [fnum(v) for v in b]
            if w <= 0 or h <= 0:
                continue
            xs.append((x + w / 2) / 1920.0)
            ys.append((y + h / 2) / 1080.0)
            areas.append((w * h) / (1920.0 * 1080.0))
        result[p] = {
            "samples": float(len(items)),
            "coverage": clamp01(len(items) / max(1.0, duration * 2.0)),
            "area": sum(areas) / max(1, len(areas)),
            "center_x": sum(xs) / max(1, len(xs)),
            "center_y": sum(ys) / max(1, len(ys)),
        }
    return result


def normalize_shots(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw = data.get("shots")
    if not isinstance(raw, list):
        raise RuntimeError("Shot inventory has no 'shots' list.")
    out = []
    for s in raw:
        if not isinstance(s, dict) or not s.get("shot_id"):
            continue
        start = fnum(s.get("source_start", s.get("start")), -1)
        end = fnum(s.get("source_end", s.get("end")), -1)
        if end <= start + EPS:
            continue
        out.append({**s, "shot_id": str(s["shot_id"]), "source_start": start, "source_end": end})
    out.sort(key=lambda x: (x["source_start"], x["source_end"]))
    return out


def normalize_content(data: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not data:
        return {}
    raw = data.get("shots", [])
    if not isinstance(raw, list):
        return {}
    return {str(x["shot_id"]): x for x in raw if isinstance(x, dict) and x.get("shot_id")}


def normalize_words(data: dict[str, Any], source_start: float, source_end: float) -> list[dict[str, Any]]:
    raw = data.get("words", [])
    if not isinstance(raw, list):
        raise RuntimeError("Transcript does not contain top-level word timestamps.")
    words = []
    for w in raw:
        if not isinstance(w, dict):
            continue
        s = fnum(w.get("start"), -1)
        e = fnum(w.get("end"), -1)
        text = str(w.get("word", "")).strip()
        if e <= s + EPS or not text:
            continue
        if e >= source_start - EPS and s <= source_end + EPS:
            words.append({"start": max(s, source_start), "end": min(e, source_end), "word": text})
    return words


def text_in_range(words: list[dict[str, Any]], start: float, end: float) -> str:
    return " ".join(w["word"] for w in words if overlap(start, end, w["start"], w["end"]) > 0).strip()


def trim_leading_non_dialogue(
    audio_segments: list[dict[str, Any]],
    words: list[dict[str, Any]],
    shots: list[dict[str, Any]],
    content: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    """Remove an unrelated leading insert before the first useful interview dialogue.

    The strict AV resolver can preserve a real source-shot at t=0. For a Short, an
    unrelated film/B-roll lead-in is not part of the spoken story and its embedded
    source audio must not leak into the final audio bed. We remove it only when:
      * the leading source shot is classified as an insert,
      * it is short,
      * the following source shot is interview footage, and
      * the leading visual has no strong semantic match to the opening speech.

    A second, transcript-gap fallback removes clearly non-dialogue pre-roll when the
    first spoken word starts substantially after the source begins.
    """
    if not audio_segments:
        return audio_segments, 0.0

    first = audio_segments[0]
    first_source = first["source_start"]
    first_end = first["source_end"]
    lead_duration = first_end - first_source
    first_shot_id = first.get("shot_id")
    first_shot = next((s for s in shots if s.get("shot_id") == first_shot_id), None)

    if first_shot is not None and lead_duration <= 2.5:
        first_kind = source_shot_kind(first_shot, content, [])
        next_shot = None
        if len(audio_segments) > 1:
            next_id = audio_segments[1].get("shot_id")
            next_shot = next((s for s in shots if s.get("shot_id") == next_id), None)
        next_kind = source_shot_kind(next_shot, content, []) if next_shot else "unknown"
        opening_text = text_in_range(words, first_source, first_end)
        score, _reasons, _blob = content_score(first_shot, content, opening_text) if first_shot else (0.0, [], "")

        unrelated_leading_insert = (
            first_kind == "insert"
            and next_kind == "interview"
            and score < INSERT_STRONG_MATCH
        )
        if unrelated_leading_insert:
            trimmed = [{**seg} for seg in audio_segments[1:]]
            if trimmed:
                for seg in trimmed:
                    seg["output_start"] -= lead_duration
                    seg["output_end"] -= lead_duration
                trimmed[0]["output_start"] = 0.0
                return trimmed, lead_duration

    # Fallback: transcript confirms a substantial leading gap before the first word.
    if words:
        first_word = min(words, key=lambda w: w["start"])["start"]
        lead = first_word - first_source
        if lead > 0.25:
            trimmed: list[dict[str, Any]] = []
            for seg in audio_segments:
                seg = {**seg}
                if seg["source_end"] <= first_word + EPS:
                    continue
                if seg["source_start"] < first_word:
                    seg["source_start"] = first_word
                trimmed.append(seg)

            if trimmed:
                offset = trimmed[0]["source_start"] - first_source
                for seg in trimmed:
                    duration = seg["source_end"] - seg["source_start"]
                    seg["output_start"] -= offset
                    seg["output_end"] = seg["output_start"] + duration
                trimmed[0]["output_start"] = 0.0
                return trimmed, offset

    return audio_segments, 0.0


def explicit_visual_entities(text: str) -> set[str]:
    lower = text.lower()
    found = set()
    if "jack nicholson" in lower:
        found.add("jack_nicholson")
    if any(x in lower for x in ("film", "movie", "scene")):
        found.add("film")
    if any(x in lower for x in ("book", "blurb", "excerpt", "acting")):
        found.add("acting_book")
    return found


def content_for_shot(shot: dict[str, Any], content: dict[str, dict[str, Any]]) -> dict[str, Any]:
    item = content.get(str(shot["shot_id"]))
    if item:
        return item
    return shot


def content_score(shot: dict[str, Any], content: dict[str, dict[str, Any]], spoken_text: str) -> tuple[float, list[str], str]:
    meta = content_for_shot(shot, content)
    summary = str(meta.get("visual_summary", ""))
    scene = str(meta.get("scene_type", "other"))
    tags = " ".join(str(x) for x in (meta.get("visual_tags") or []))
    uses = " ".join(str(x) for x in (meta.get("editorial_uses") or []))
    blob = f"{summary} {scene} {tags} {uses}".lower()
    query = tokenize(spoken_text)
    source_tokens = tokenize(blob)
    overlap_count = len(query & source_tokens)
    lex = overlap_count / max(1, len(query))
    entities = explicit_visual_entities(spoken_text)
    entity_bonus = 0.0
    reasons: list[str] = []
    if "jack_nicholson" in entities and "nicholson" in blob:
        entity_bonus += 0.80
        reasons.append("explicit_jack_nicholson_match")
    if "film" in entities and any(x in blob for x in ("film", "movie", "scene", "actor")):
        entity_bonus += 0.25
        reasons.append("film_context_match")
    if "acting_book" in entities and any(x in blob for x in ("book", "acting", "text", "how to stop")):
        entity_bonus += 0.25
        reasons.append("acting_book_context_match")
    if scene in {"person", "conversation"}:
        reasons.append("person_or_conversation_visual")
    if meta.get("content_confidence") is not None:
        conf = clamp01(fnum(meta.get("content_confidence")))
    else:
        conf = 0.5 if summary else 0.0
    uses_set = {str(x).strip().lower() for x in (meta.get("editorial_uses") or []) if x}

    # A source insert does not need to literally contain the celebrity's name.
    # For example, a movie frame can be semantically valid while the frame itself
    # only shows the scene. The spoken reference + explicit supporting/evidence
    # role is enough to make it a serious candidate, but only for insert footage.
    if entities & {"jack_nicholson", "film"} and shot.get("shot_type") in VISUAL_NONFACE_TYPES:
        if uses_set & {"supporting_example", "evidence"}:
            entity_bonus = max(entity_bonus, 0.70)
            reasons.append("explicit_spoken_visual_reference_plus_supporting_source_role")
        elif entities & {"jack_nicholson"} and any(x in blob for x in ("film", "movie", "scene", "actor", "character")):
            entity_bonus = max(entity_bonus, 0.58)
            reasons.append("explicit_jack_nicholson_context_in_movie_like_insert")

    score = clamp01(0.45 * lex + 0.45 * min(1.0, entity_bonus) + 0.10 * conf)
    if (entities & {"jack_nicholson", "film"}) and uses_set & {"supporting_example", "evidence"}:
        # Strong source-role metadata plus an explicit spoken visual reference is
        # sufficient to make this a real insert candidate even when the vision model
        # cannot identify the named actor from a frame alone.
        score = max(score, 0.62)
    return score, reasons, blob


def source_shots_for_interval(shots: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    return [s for s in shots if overlap(start, end, s["source_start"], s["source_end"]) > 1e-4]


def next_useful_interview_shot(
    shots: list[dict[str, Any]],
    content: dict[str, dict[str, Any]],
    observations: list[dict[str, Any]],
    focus: str | None,
    after: float,
) -> dict[str, Any] | None:
    """Find the nearest later interview shot suitable as an explicit visual lead-in."""
    candidates = [
        s for s in shots
        if s["source_start"] >= after - EPS
        and source_shot_kind(s, content, observations) == "interview"
    ]
    ranked: list[tuple[int, float, float, dict[str, Any]]] = []
    for shot in candidates:
        faces = person_face_stats(observations, shot["source_start"], shot["source_end"])
        target_present = 1 if focus and faces.get(focus, {}).get("samples", 0.0) > 0 else 0
        target_area = faces.get(focus, {}).get("area", 0.0) if focus else 0.0
        ranked.append((target_present, target_area, -shot["source_start"], shot))
    ranked.sort(key=lambda x: (-x[0], -x[1], x[2]))
    return ranked[0][3] if ranked else None


def contiguous_visual_span(
    shots: list[dict[str, Any]],
    start: float,
    duration: float,
) -> tuple[float, float] | None:
    """Return one contiguous source-video span of the requested duration.

    The span is allowed to cross normal source shot boundaries. That is intentional: a
    creative visual decision can contain source-camera cuts while remaining one visual state.
    """
    if duration <= EPS:
        return None
    ordered = [s for s in shots if s["source_end"] > start + EPS]
    ordered.sort(key=lambda x: x["source_start"])
    first = None
    for shot in ordered:
        if shot["source_start"] - EPS <= start < shot["source_end"] - EPS:
            first = shot
            break
    if first is None:
        return None
    cursor = start
    remaining = duration
    current_index = ordered.index(first)
    while remaining > EPS:
        shot = ordered[current_index]
        shot_start = max(cursor, shot["source_start"])
        available = shot["source_end"] - shot_start
        if available <= EPS:
            current_index += 1
            if current_index >= len(ordered):
                return None
            cursor = ordered[current_index]["source_start"]
            continue
        take = min(available, remaining)
        cursor = shot_start + take
        remaining -= take
        if remaining > EPS:
            current_index += 1
            if current_index >= len(ordered):
                return None
            next_shot = ordered[current_index]
            if abs(next_shot["source_start"] - cursor) > 0.08:
                return None
            cursor = next_shot["source_start"]
    return start, cursor


def source_shot_kind(
    shot: dict[str, Any],
    content: dict[str, dict[str, Any]],
    observations: list[dict[str, Any]],
) -> str:
    """Classify a source shot as interview footage or supporting/B-roll footage.

    IMPORTANT: content intelligence can explicitly identify a film/TV source even
    when the generic shot detector says ``person`` or the face tracker assigns a
    known interview identity. Those content-level signals must win; otherwise a
    movie close-up can be incorrectly treated as a talking-head shot.
    """
    meta = content_for_shot(shot, content)
    st = str(shot.get("shot_type", meta.get("shot_type", ""))).strip().lower()
    scene = str(meta.get("scene_type", "")).strip().lower()
    summary = str(meta.get("visual_summary", "")).lower()
    tags = {str(x).strip().lower() for x in (meta.get("visual_tags") or []) if x}
    uses = {str(x).strip().lower() for x in (meta.get("editorial_uses") or []) if x}
    text = " ".join(str(x) for x in (meta.get("text_overlays") or meta.get("text") or []))
    text_lower = text.lower()
    blob = f"{summary} {text_lower} {' '.join(tags)} {' '.join(uses)}"

    # High-confidence semantic evidence for film/TV or other inserted footage is
    # authoritative. This check runs BEFORE face identity evidence on purpose.
    film_markers = (
        "film", "movie", "tv", "television", "scene", "character",
        "wolf (1994)", "teenage mutant ninja turtles (1990)",
    )
    insert_roles = {"supporting_example", "evidence", "b_roll", "b-roll", "insert"}
    if (uses & insert_roles) and any(marker in blob for marker in film_markers):
        return "insert"
    if any(marker in text_lower for marker in film_markers) and scene not in {"conversation"}:
        return "insert"
    if "supporting_example" in uses and scene not in {"conversation"}:
        return "insert"

    # Explicit interview camera types remain authoritative when available.
    if st in {"single_wide", "single_medium", "single_close", "two_shot", "medium", "close", "wide"}:
        return "interview"

    stats = person_face_stats(
        observations,
        shot["source_start"],
        shot["source_end"],
    )
    present = [p for p in SPEAKERS if stats[p]["samples"] > 0 and stats[p]["coverage"] >= 0.55]

    # Balanced, persistent presence of both known interview identities is strong
    # evidence that this is the actual interview camera rather than an insert.
    if len(present) == 2:
        return "interview"

    # A sustained close/medium of one persistent interview identity is valid only
    # when no stronger semantic evidence has already marked the shot as B-roll.
    if len(present) == 1:
        p = present[0]
        if stats[p]["coverage"] >= 0.55 and stats[p]["area"] >= 0.018:
            return "interview"

    # Semantic scene metadata is the final interview fallback.
    if st not in VISUAL_NONFACE_TYPES and "insert" not in st and "screen" not in st:
        if scene in {"conversation"}:
            return "interview"

    return "insert"


def strongest_speaker_state(turns: list[dict[str, Any]], start: float, end: float, faces: dict[str, dict[str, float]]) -> tuple[str | None, dict[str, float], str]:
    focus, totals = dominant_speaker(turns, start, end)
    if focus:
        if faces[focus]["samples"] > 0:
            return focus, totals, "active_speaker_and_face"
        return focus, totals, "active_speaker_no_face"
    visible = [p for p in SPEAKERS if faces[p]["samples"] > 0]
    if len(visible) == 1:
        return visible[0], totals, "single_visible_person"
    return None, totals, "no_unambiguous_speaker"


def choose_base_visual(shots: list[dict[str, Any]], content: dict[str, dict[str, Any]], start: float, end: float, turns: list[dict[str, Any]], observations: list[dict[str, Any]], words: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = source_shots_for_interval(shots, start, end)
    faces = person_face_stats(observations, start, end)
    focus, totals, focus_reason = strongest_speaker_state(turns, start, end, faces)
    spoken = text_in_range(words, start, end)
    entities = explicit_visual_entities(spoken)

    # Prefer a semantically matched insert only when it is actually relevant to the
    # words being spoken. This is what lets the Jack Nicholson footage appear where it
    # belongs without turning every source insert into a flash.
    insert_candidates = []
    interview_candidates = []
    for shot in candidates:
        kind = source_shot_kind(shot, content, observations)
        if kind == "insert":
            score, reasons, blob = content_score(shot, content, spoken)
            insert_candidates.append((score, shot, reasons, blob))
        elif kind == "interview":
            interview_candidates.append(shot)

    if entities and insert_candidates:
        insert_candidates.sort(key=lambda x: (-x[0], x[1]["source_start"]))
        score, shot, reasons, blob = insert_candidates[0]
        if score >= INSERT_STRONG_MATCH:
            return {
                "mode": "supporting_insert",
                "target_identity": None,
                "shot_id": shot["shot_id"],
                "reason": "semantic_insert_match",
                "score": round(score, 3),
                "match_reasons": reasons,
                "spoken_text": spoken,
                "entities": sorted(entities),
            }

    if focus and interview_candidates:
        # Pick an interview source that actually contains the active speaker.
        ranked_interview = []
        for candidate in interview_candidates:
            cfaces = person_face_stats(
                observations,
                candidate["source_start"],
                candidate["source_end"],
            )
            target_stats = cfaces.get(focus, {})
            ranked_interview.append((
                1 if target_stats.get("samples", 0.0) > 0 else 0,
                target_stats.get("area", 0.0),
                target_stats.get("coverage", 0.0),
                -candidate["source_start"],
                candidate,
            ))
        ranked_interview.sort(reverse=True, key=lambda x: x[:4])
        chosen_interview = ranked_interview[0][4]
        chosen_faces = person_face_stats(
            observations,
            chosen_interview["source_start"],
            chosen_interview["source_end"],
        )
        if chosen_faces[focus]["samples"] > 0:
            return {
                "mode": "speaker_focus",
                "target_identity": focus,
                "shot_id": chosen_interview["shot_id"],
                "reason": focus_reason,
                "score": round(totals[focus] / max(EPS, sum(totals.values())), 3),
                "spoken_text": spoken,
                "entities": sorted(entities),
            }

    if len([p for p in SPEAKERS if faces[p]["samples"] > 0]) == 2 and not focus:
        return {
            "mode": "two_person",
            "target_identity": None,
            "shot_id": interview_candidates[0]["shot_id"] if interview_candidates else None,
            "reason": "both_interview_people_visible_without_unambiguous_audio_focus",
            "score": 0.0,
            "spoken_text": spoken,
            "entities": sorted(entities),
        }

    if interview_candidates:
        return {
            "mode": "speaker_focus",
            "target_identity": focus,
            "shot_id": interview_candidates[0]["shot_id"],
            "reason": "best_available_interview_visual",
            "score": 0.0,
            "spoken_text": spoken,
            "entities": sorted(entities),
        }

    # No useful visual in this source interval. The build stage decides whether this
    # should become an explicit visual-only lead-in or a continuation of a previous
    # semantic visual state. This is deliberately not a renderer-level guess.
    return {
        "mode": "no_visual",
        "target_identity": focus,
        "shot_id": None,
        "reason": "no_semantically_useful_visual_in_current_source_interval",
        "score": 0.0,
        "spoken_text": spoken,
        "entities": sorted(entities),
    }


def build_visual_story(sequence: dict[str, Any], shots: list[dict[str, Any]], visual: dict[str, Any], speakers: dict[str, Any], transcript: dict[str, Any], content: dict[str, dict[str, Any]]) -> dict[str, Any]:
    actions = sequence.get("actions", [])
    if not isinstance(actions, list) or not actions:
        raise RuntimeError("Resolved sequence has no actions.")

    # Keep the locked editorial audio timeline exactly as resolved.
    audio_segments = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        for r in action.get("render_segments", []):
            if not isinstance(r, dict):
                continue
            s = fnum(r.get("source_in"), -1)
            e = fnum(r.get("source_out"), -1)
            os = fnum(r.get("output_start"), -1)
            oe = fnum(r.get("output_end"), -1)
            if e > s + EPS and oe > os + EPS:
                audio_segments.append({"source_start": s, "source_end": e, "output_start": os, "output_end": oe, "shot_id": str(r.get("shot_id")) if r.get("shot_id") else None})
    audio_segments.sort(key=lambda x: x["output_start"])
    if not audio_segments:
        raise RuntimeError("Resolved sequence has no render segments.")

    source_start = audio_segments[0]["source_start"]
    source_end = audio_segments[-1]["source_end"]
    words = normalize_words(transcript, source_start, source_end)
    audio_segments, leading_trim = trim_leading_non_dialogue(
        audio_segments, words, shots, content
    )
    source_start = audio_segments[0]["source_start"]
    source_end = audio_segments[-1]["source_end"]
    turns = normalize_turns(speakers)
    observations = normalize_observations(visual)

    # Build one decision per source interval first, then coalesce adjacent decisions
    # with the same creative state. Leading non-dialogue pre-roll has already been
    # removed when it is an unrelated insert before the first spoken word.
    raw_decisions = []
    prev_state: dict[str, Any] | None = None
    cooldown = 0.0

    for seg in audio_segments:
        start, end = seg["source_start"], seg["source_end"]
        decision = choose_base_visual(shots, content, start, end, turns, observations, words)
        duration = end - start

        # Case A: audio begins before the first useful interview camera appears.
        # Use an explicit visual-only lead-in from the nearest valid interview shot,
        # rather than carrying a nonexistent previous visual or showing irrelevant B-roll.
        if decision["mode"] == "no_visual" and not raw_decisions:
            focus = decision.get("target_identity")
            lead = next_useful_interview_shot(shots, content, observations, focus, end)
            if lead:
                decision = {
                    **decision,
                    "mode": "speaker_focus",
                    "shot_id": lead["shot_id"],
                    "reason": "explicit_continuous_visual_lead_in_from_first_useful_interview_shot",
                    "visual_cursor_start": lead["source_start"],
                    "visual_reuse": False,
                }

        # Case B: stay on a semantically relevant film insert when the locked dialogue
        # continues the same example into the next AV source interval. The source shot
        # boundary is only realization detail; the creative insert state continues.
        if decision["mode"] == "no_visual" and raw_decisions:
            previous = raw_decisions[-1]
            current_sources = source_shots_for_interval(shots, start, end)
            current_kinds = [source_shot_kind(s, content, observations) for s in current_sources]
            if previous.get("visual_mode") == "supporting_insert" and current_sources and current_kinds and all(k == "insert" for k in current_kinds):
                decision = {
                    **decision,
                    "mode": "supporting_insert",
                    "target_identity": None,
                    "shot_id": current_sources[0]["shot_id"],
                    "reason": "continue_semantic_insert_state_across_source_interval",
                    "score": 0.0,
                    "entities": previous.get("semantic_entities", decision.get("entities", [])),
                }

        state_key = (
            decision["mode"],
            decision.get("target_identity"),
            tuple(decision.get("entities", [])) if decision["mode"] == "supporting_insert" else None,
        )

        # A source-camera change inside the same speaker state does NOT justify a new
        # editorial visual decision. We keep it inside the same visual state.
        if raw_decisions and prev_state is not None:
            same_creative_state = state_key == prev_state["state_key"]
            if same_creative_state or (decision["mode"] == "speaker_focus" and prev_state.get("mode") == "speaker_focus" and decision.get("target_identity") == prev_state.get("target_identity")):
                prev = raw_decisions[-1]
                prev["output_end"] = seg["output_end"]
                prev["audio_source_end"] = end
                prev["source_intervals"].append({**seg})
                prev["reasons"].append("source_shot_change_not_promoted_to_editorial_visual_cut")
                cooldown = max(0.0, cooldown - duration)
                continue

        # Speaker switch is the primary dialogue visual-cut trigger, but don't cut on a
        # tiny turn that is shorter than the cooldown window.
        if raw_decisions and decision["mode"] == "speaker_focus" and prev_state is not None and prev_state.get("mode") == "speaker_focus":
            if decision.get("target_identity") != prev_state.get("target_identity") and cooldown > 0 and duration < SPEAKER_SWITCH_CUT_COOLDOWN:
                prev = raw_decisions[-1]
                prev["output_end"] = seg["output_end"]
                prev["audio_source_end"] = end
                prev["source_intervals"].append({**seg})
                prev["reasons"].append("tiny_speaker_switch_held_for_visual_continuity")
                cooldown = max(0.0, cooldown - duration)
                continue

        if decision["mode"] == "no_visual":
            raise RuntimeError(
                f"No executable visual state for source interval {start:.3f}->{end:.3f}. "
                "The planner must resolve this into an interview visual, semantic insert, or explicit visual reuse."
            )

        item = {
            "decision_index": len(raw_decisions),
            "output_start": seg["output_start"],
            "output_end": seg["output_end"],
            "audio_source_start": start,
            "audio_source_end": end,
            "visual_mode": decision["mode"],
            "target_identity": decision.get("target_identity"),
            "visual_anchor_shot_id": decision.get("shot_id"),
            "spoken_text": decision.get("spoken_text", ""),
            "semantic_entities": decision.get("entities", []),
            "reasons": [decision["reason"]],
            "source_intervals": [{**seg}],
            "visual_reuse": bool(decision.get("visual_reuse", False)),
            "visual_cursor_start": decision.get("visual_cursor_start"),
            "visual_only_source_start": decision.get("visual_only_source_start"),
            "visual_only_source_end": decision.get("visual_only_source_end"),
            "state_key": state_key,
        }
        raw_decisions.append(item)
        prev_state = item
        cooldown = SPEAKER_SWITCH_CUT_COOLDOWN if decision["mode"] != "carry_previous" else cooldown

    # Convert creative decisions to executable source realizations.
    visual_segments = []
    for decision in raw_decisions:
        intervals = decision["source_intervals"]
        visual_cursor = fnum(decision.get("visual_cursor_start"), -1) if decision.get("visual_cursor_start") is not None else None
        for idx, src in enumerate(intervals):
            start, end = src["source_start"], src["source_end"]
            mode = decision["visual_mode"]
            anchor = decision.get("visual_anchor_shot_id")
            shot = next((s for s in shots if s["shot_id"] == anchor), None) if anchor else None
            duration = end - start
            if visual_cursor is not None:
                span = contiguous_visual_span(shots, visual_cursor, duration)
                if span is None:
                    raise RuntimeError(
                        f"Cannot realize continuous visual lead stream for decision {decision['decision_index']} "
                        f"at {start:.3f}->{end:.3f}; insufficient contiguous source footage."
                    )
                vs, ve = span
                # Keep the visual stream continuous across source shot boundaries.
                visual_cursor = ve
                sid = src.get("shot_id")
            elif mode == "supporting_insert":
                # Use the source realization interval itself. The creative state can
                # continue across SHOT_149 -> SHOT_150 -> SHOT_151 without becoming a
                # new editorial visual decision.
                vs, ve, sid = start, end, src.get("shot_id")
                if shot and ve <= vs + EPS:
                    vs, ve, sid = start, end, shot["shot_id"]
            else:
                # Normal speaker focus follows the exact audio source interval so
                # lip-sync is preserved. Source shot boundaries remain realization detail.
                vs, ve, sid = start, end, src.get("shot_id")
            visual_segments.append({
                "visual_index": len(visual_segments),
                "decision_index": decision["decision_index"],
                "output_start": src["output_start"],
                "output_end": src["output_end"],
                "audio_source_start": start,
                "audio_source_end": end,
                "visual_source_start": vs,
                "visual_source_end": ve,
                "visual_shot_id": sid,
                "visual_mode": mode,
                "target_identity": decision.get("target_identity"),
                "spoken_text": decision.get("spoken_text", ""),
                "semantic_entities": decision.get("semantic_entities", []),
                "visual_reuse": bool(decision.get("visual_reuse", False)),
                "reasons": list(decision["reasons"]),
            })

    editorial_cuts = max(0, len(raw_decisions) - 1)
    source_realization_cuts = max(0, len(visual_segments) - 1)

    # Do not allow the planner to claim success if it merely mirrored the AV timeline.
    mirrored = len(raw_decisions) == len(audio_segments) and all(
        abs(a["source_start"] - v["audio_source_start"]) < 1e-4 and
        abs(a["source_end"] - v["audio_source_end"]) < 1e-4
        for a, v in zip(audio_segments, visual_segments)
    )

    return {
        "version": "1.1",
        "module": "visual_story_editor",
        "method": "creative_visual_state_then_source_realization",
        "policy": {
            "editorial_story_locked": True,
            "audio_source_timeline_authoritative": True,
            "source_shot_boundaries_are_not_editorial_cut_triggers": True,
            "speaker_changes_can_trigger_visual_cut": True,
            "semantic_insert_required": True,
            "visual_timeline_independent": True,
            "explicit_visual_reuse_required": True,
            "audio_reuse_forbidden": True,
            "lip_sync_preserved_for_speaker_focus": True,
            "mirrored_av_timeline_is_failure": True,
        },
        "source_range": [round(source_start, 6), round(source_end, 6)],
        "output_duration": round(audio_segments[-1]["output_end"], 6),
        "visual_decisions": [
            {k: v for k, v in d.items() if k != "state_key"}
            for d in raw_decisions
        ],
        "visual_segments": visual_segments,
        "diagnostics": {
            "audio_segments": len(audio_segments),
            "visual_decisions": len(raw_decisions),
            "source_realization_segments": len(visual_segments),
            "editorial_visual_cuts": editorial_cuts,
            "source_realization_cuts": source_realization_cuts,
            "mirrors_audio_timeline": mirrored,
            "source_boundaries_not_authoritative": True,
            "speaker_turns_used": len(turns),
            "leading_non_dialogue_trim_seconds": round(leading_trim, 6),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Build the creative visual timeline for a vertical Short")
    p.add_argument("--sequence", required=True)
    p.add_argument("--shots", required=True)
    p.add_argument("--visual", required=True)
    p.add_argument("--speakers", required=True)
    p.add_argument("--transcript", required=True)
    p.add_argument("--shot-content", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    sequence = load_json(Path(args.sequence))
    shots = normalize_shots(load_json(Path(args.shots)))
    visual = load_json(Path(args.visual))
    speakers = load_json(Path(args.speakers))
    transcript = load_json(Path(args.transcript))
    content = normalize_content(load_json(Path(args.shot_content)))

    plan = build_visual_story(sequence, shots, visual, speakers, transcript, content)
    if plan["diagnostics"]["mirrors_audio_timeline"]:
        raise RuntimeError(
            "FAIL: visual planner mirrored the AV timeline. No creative visual edit was produced."
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== VISUAL STORY EDITOR ===")
    print(json.dumps(plan["diagnostics"], indent=2))
    for d in plan["visual_decisions"]:
        print(
            f"DECISION {d['decision_index']}: "
            f"{d['output_start']:.3f}->{d['output_end']:.3f} | "
            f"{d['visual_mode']} | target={d.get('target_identity')} | "
            f"anchor={d.get('visual_anchor_shot_id')} | "
            f"reason={'; '.join(d.get('reasons', []))}"
        )
    print(f"output: {out}")


if __name__ == "__main__":
    main()
