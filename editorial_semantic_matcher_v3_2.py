#!/usr/bin/env python3
"""
Editorial Semantic Matcher V3

Purpose
-------
Decide whether a detected visual insert is editorially supported by nearby
speech and/or by the visual role of the source shot.

V3 changes
----------
- keeps the event source shot as the primary candidate
- uses concept families instead of raw token overlap alone
- separates semantic relationship from editorial usefulness
- treats explicit visual-reference language as a cue, not proof
- handles speech-free inserts without pretending they have semantic dialogue
- records detailed score components for debugging
- never renders or edits video

The final sequence director remains authoritative.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

VERSION = "3.2"

MIN_CONTENT_CONFIDENCE = 0.55
LOCAL_CANDIDATE_THRESHOLD = 0.52
STRONG_LOCAL_THRESHOLD = 0.72
CONTEXTUAL_MAX_RANKING_SCORE = 0.35
REMOTE_THRESHOLD = 0.80
REMOTE_MARGIN = 0.14
MAX_REMOTE_CANDIDATES = 3

INSERT_INTENTS = {
    "supporting_insert",
    "supporting_insert_candidate",
}

STOPWORDS = {
    "a","an","and","are","as","at","be","been","but","by",
    "can","could","did","do","does","for","from","get","got",
    "had","has","have","he","her","here","him","his","how",
    "i","if","in","is","it","its","just","me","my","of",
    "oh","on","or","our","she","so","that","the","their",
    "them","there","they","this","to","up","us","was","we",
    "were","what","when","where","which","who","why","will",
    "with","you","your","yo","bro","sir","like","okay","ok",
    "yeah","really","thing","things","completely","think",
}

VISUAL_REFERENCE_WORDS = {
    "look", "see", "show", "showing", "shown", "picture", "photo",
    "image", "video", "screen", "logo", "bottle", "food", "chicken",
    "sauce", "trophy", "award", "fire", "flames", "edit", "clip",
    "game", "gameplay", "camera", "here", "there", "this", "that",
    "these", "those",
}

# Concept families. Matching is directional but intentionally conservative.
ALIASES: dict[str, set[str]] = {
    "trophy": {"trophy", "award", "cup", "prize", "championship", "winner"},
    "award": {"award", "trophy", "prize", "cup", "winner"},
    "chicken": {"chicken", "wing", "wings", "nugget", "nuggets", "fried", "poultry"},
    "food": {"food", "chicken", "wing", "wings", "nugget", "nuggets", "fried", "meal", "dish"},
    "spicy": {"spicy", "hot", "heat", "fire", "flame", "flames", "chili", "pepper"},
    "hot": {"hot", "spicy", "heat", "fire", "flame", "flames"},
    "fire": {"fire", "flame", "flames", "burning", "heat", "smoke", "explosion", "blast"},
    "sauce": {"sauce", "condiment", "dressing", "hot-sauce", "bottle"},
    "bottle": {"bottle", "bottles", "product", "container"},
    "mcdonalds": {"mcdonalds", "mcdonald", "fast-food", "burger", "nuggets", "mcnuggets"},
    "brand": {"brand", "logo", "sponsor", "sponsorship", "branding", "title-card"},
    "logo": {"logo", "brand", "branding", "sponsor", "sponsorship"},
    "screen": {"screen", "display", "monitor", "ui", "interface", "graphic"},
    "game": {"game", "gameplay", "gaming", "stream", "streamer"},
    "edit": {"edit", "editing", "editor", "clip", "video", "graphic", "title-card", "transition"},
    "graphic": {"graphic", "logo", "text", "title-card", "overlay", "screen"},
    "transition": {"transition", "graphic", "flames", "fire", "explosion", "title-card"},
    "people": {"person", "people", "man", "woman", "speaker", "face"},
}

PHRASES = (
    "hot sauce", "fried chicken", "spicy chicken", "spicy mcnuggets",
    "mcdonalds", "mcdonald", "hot ones", "round 1", "trophy",
    "look here", "what are you doing", "in an edit",
)

VISUAL_USE_WEIGHTS = {
    "product_reference": 1.0,
    "object_reference": 0.92,
    "supporting_example": 0.90,
    "screen_reference": 0.86,
    "gameplay_reference": 0.86,
    "evidence": 0.82,
    "branding": 0.70,
    "transition": 0.60,
    "atmosphere": 0.48,
    "location_reference": 0.80,
    "reaction_reference": 0.70,
    "unknown": 0.20,
}

SCENE_ROLE_WEIGHTS = {
    "product_closeup": 0.95,
    "object": 0.90,
    "graphic": 0.75,
    "screen_capture": 0.80,
    "document": 0.78,
    "gameplay": 0.88,
    "environment": 0.60,
    "conversation": 0.35,
    "person": 0.35,
    "montage": 0.55,
    "other": 0.35,
}


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def tokenize(text: Any) -> list[str]:
    raw = re.sub(r"[^a-z0-9' -]+", " ", str(text or "").lower())
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", re.sub(r"\s+", " ", raw))


def stem(token: str) -> str:
    t = token.strip("-'" )
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 5 and t.endswith("ing"):
        return t[:-3]
    if len(t) > 4 and t.endswith("ed"):
        return t[:-2]
    if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def concepts(text: Any) -> set[str]:
    out: set[str] = set()
    for token in tokenize(text):
        s = stem(token)
        if len(s) >= 2 and s not in STOPWORDS:
            out.add(s)
    return out


def expand(tokens: set[str]) -> set[str]:
    out = set(tokens)
    for token in list(tokens):
        out.update(ALIASES.get(token, {token}))
    return out


def phrase_bonus(qtext: str, vtext: str) -> float:
    q = " ".join(tokenize(qtext))
    v = " ".join(tokenize(vtext))
    matches = [p for p in PHRASES if p in q and p in v]
    return 1.0 if matches else 0.0


def explicit_visual_reference(qtext: str) -> bool:
    words = set(tokenize(qtext))
    return bool(words & VISUAL_REFERENCE_WORDS)


def shot_text(shot: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "visual_summary", "scene_type", "objects", "actions",
        "visual_tags", "editorial_uses", "text_visible", "local_ocr",
    ):
        value = shot.get(key)
        if isinstance(value, list):
            parts.extend(str(x) for x in value if x)
        elif value:
            parts.append(str(value))
    return " ".join(parts)


def concept_overlap(query: set[str], visual: set[str]) -> tuple[float, list[str]]:
    if not query or not visual:
        return 0.0, []

    qx = expand(query)
    vx = expand(visual)
    direct = len(query & visual) / max(1, len(query))
    broad = len(qx & vx) / max(1, len(qx))
    reverse = len(qx & vx) / max(1, len(vx))
    score = clamp01(0.50 * direct + 0.35 * broad + 0.15 * reverse)
    overlap = sorted(query & visual)
    return score, overlap


def temporal_proximity(b0: float, b1: float, s0: float, s1: float) -> float:
    if s1 >= b0 and s0 <= b1:
        return 1.0
    distance = b0 - s1 if s1 < b0 else s0 - b1
    return math.exp(-max(0.0, distance) / 8.0)


def editorial_role_score(shot: dict[str, Any]) -> float:
    uses = shot.get("editorial_uses")
    if not isinstance(uses, list):
        uses = []
    use_score = max((VISUAL_USE_WEIGHTS.get(str(u), 0.20) for u in uses), default=0.20)
    scene = str(shot.get("scene_type", "other"))
    scene_score = SCENE_ROLE_WEIGHTS.get(scene, 0.35)
    return clamp01(0.65 * use_score + 0.35 * scene_score)


def cue_strength(qtext: str) -> float:
    words = tokenize(qtext)
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in VISUAL_REFERENCE_WORDS)
    return clamp01((hits / max(1, len(words))) * 3.5)


def contextual_relationship(beat: dict[str, Any], shot: dict[str, Any]) -> tuple[float, list[str]]:
    qtext = str(beat.get("speech_excerpt", "")).strip()
    if not qtext:
        return 0.0, ["no speech in local event context"]

    q = concepts(qtext)
    v = concepts(shot_text(shot))
    sem, overlap = concept_overlap(q, v)
    cue = cue_strength(qtext)
    phrase = phrase_bonus(qtext, shot_text(shot))

    reasons: list[str] = []
    if overlap:
        reasons.append("concept overlap: " + ", ".join(overlap[:10]))
    if phrase:
        reasons.append("phrase-level relation present")
    if cue >= 0.55:
        reasons.append("strong visual-reference language")
    elif cue > 0.0:
        reasons.append("weak visual-reference language")

    # Semantic relation is mostly content, but explicit visual language can rescue
    # cases such as "look here" where the visual target is intentionally underspecified.
    relation = clamp01(0.78 * sem + 0.12 * phrase + 0.10 * cue)

    # Special editorial relationship: references to editing are naturally compatible
    # with graphic/title/transition visuals, but still require a real source shot.
    qx = expand(q)
    shot_scene = str(shot.get("scene_type", "other"))
    uses = {str(x) for x in (shot.get("editorial_uses") or [])}
    if "edit" in qx and (shot_scene == "graphic" or "transition" in uses):
        relation = max(relation, 0.64)
        reasons.append("dialogue references editing and source visual is editorially compatible")

    return relation, reasons


def score_pair(beat: dict[str, Any], shot: dict[str, Any]) -> dict[str, Any]:
    qtext = str(beat.get("speech_excerpt", ""))
    vtext = shot_text(shot)

    q = concepts(qtext)
    v = concepts(vtext)
    sem, overlap = concept_overlap(q, v)
    phrase = phrase_bonus(qtext, vtext)

    b0 = fnum(beat.get("start"))
    b1 = fnum(beat.get("end"), b0)
    s0 = fnum(shot.get("source_start"))
    s1 = fnum(shot.get("source_end"), s0)
    temporal = temporal_proximity(b0, b1, s0, s1)
    confidence = clamp01(fnum(shot.get("content_confidence")))
    role = editorial_role_score(shot)
    cue = cue_strength(qtext)
    relation, relation_reasons = contextual_relationship(beat, shot)
    has_speech = bool(qtext.strip())

    # Semantic relationship dominates when speech exists. Editorial role and timing
    # provide support, never enough by themselves to make a strong semantic match.
    if has_speech:
        score = clamp01(
            0.52 * relation
            + 0.16 * sem
            + 0.10 * phrase
            + 0.08 * cue
            + 0.08 * temporal
            + 0.06 * confidence
        )
    else:
        # Speech-free inserts can only become contextual candidates. This prevents
        # title/branding/transition shots from being mistaken for semantic proof.
        score = clamp01(
            0.34 * role
            + 0.20 * confidence
            + 0.46 * temporal
        )

    reasons = list(relation_reasons)
    if not has_speech:
        reasons.append("speech-free insert scored as contextual/editorial support only")
    if confidence >= MIN_CONTENT_CONFIDENCE:
        reasons.append("usable visual metadata confidence")
    if temporal >= 0.85:
        reasons.append("source timing is locally close")
    if role >= 0.75:
        reasons.append("strong editorial visual role")

    return {
        "shot_id": str(shot["shot_id"]),
        "source_start": round(s0, 6),
        "source_end": round(s1, 6),
        "score": round(score, 6),
        "semantic_score": round(sem, 6),
        "relationship_score": round(relation, 6),
        "phrase_score": round(phrase, 6),
        "cue_score": round(cue, 6),
        "temporal_score": round(temporal, 6),
        "editorial_role_score": round(role, 6),
        "content_confidence": round(confidence, 6),
        "speech_present": has_speech,
        "concept_overlap": overlap[:10],
        "visual_summary": str(shot.get("visual_summary", "")),
        "scene_type": str(shot.get("scene_type", "other")),
        "editorial_uses": list(shot.get("editorial_uses") or []),
        "reasons": reasons,
    }


def normalize_beats(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw = data.get("beats", [])
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for beat in raw:
        if not isinstance(beat, dict):
            continue
        start = fnum(beat.get("start"))
        end = fnum(beat.get("end"), start)
        if end <= start:
            continue
        out.append({**beat, "start": start, "end": end})
    return sorted(out, key=lambda x: (x["start"], x["end"]))


def normalize_shots(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw = data.get("shots", [])
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for shot in raw:
        if not isinstance(shot, dict) or not shot.get("shot_id"):
            continue
        out.append({
            **shot,
            "shot_id": str(shot["shot_id"]),
            "source_start": fnum(shot.get("source_start")),
            "source_end": fnum(shot.get("source_end"), fnum(shot.get("source_start"))),
            "content_confidence": clamp01(fnum(shot.get("content_confidence"))),
        })
    return sorted(out, key=lambda x: (x["source_start"], x["source_end"]))


def resolve_source_shot(
    beat: dict[str, Any],
    usable: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str, str | None]:
    source_shot_id = beat.get("source_shot_id")
    if source_shot_id:
        source = next((s for s in usable if str(s["shot_id"]) == str(source_shot_id)), None)
        if source:
            return source, "explicit_id", str(source["shot_id"])

    b0 = fnum(beat.get("start"))
    b1 = fnum(beat.get("end"), b0)

    boundary = [
        s for s in usable
        if abs(s["source_start"] - b0) <= 0.08
        or abs(s["source_end"] - b1) <= 0.08
    ]
    if boundary:
        source = min(boundary, key=lambda s: abs(s["source_start"] - b0))
        return source, "timing_recovered", str(source["shot_id"])

    containing = [
        s for s in usable
        if s["source_start"] - 0.08 <= b0
        and s["source_end"] + 0.08 >= b1
    ]
    if containing:
        source = min(containing, key=lambda s: abs(s["source_start"] - b0))
        return source, "timing_recovered", str(source["shot_id"])

    return None, "unresolved", str(source_shot_id) if source_shot_id else None


def build_match(
    beat: dict[str, Any],
    beats_index: int,
    all_shots: list[dict[str, Any]],
) -> dict[str, Any]:
    relevant = beat.get("intent") in INSERT_INTENTS

    base = {
        "beat_index": beats_index,
        "start": round(fnum(beat.get("start")), 3),
        "end": round(fnum(beat.get("end")), 3),
        "intent": str(beat.get("intent", "")),
        "speaker": beat.get("speaker"),
        "speech_excerpt": str(beat.get("speech_excerpt", "")),
        "insert_relevant": relevant,
    }

    if not relevant:
        return {
            **base,
            "decision": "not_applicable",
            "source_shot_id": None,
            "source_shot_resolution": "not_applicable",
            "source_shot_match": None,
            "remote_matches": [],
            "chosen_match": None,
            "decision_reason": [],
        }

    usable = [
        s for s in all_shots
        if s.get("visual_summary")
        and s.get("content_confidence", 0.0) >= MIN_CONTENT_CONFIDENCE
    ]

    source_shot, resolution, resolved_id = resolve_source_shot(beat, usable)
    local = score_pair(beat, source_shot) if source_shot else None
    visual_reference = explicit_visual_reference(base["speech_excerpt"])

    remotes: list[dict[str, Any]] = []
    if visual_reference:
        for shot in usable:
            if source_shot and shot["shot_id"] == source_shot["shot_id"]:
                continue
            scored = score_pair(beat, shot)
            if scored["score"] >= REMOTE_THRESHOLD:
                remotes.append(scored)
        remotes.sort(key=lambda x: -x["score"])
        remotes = remotes[:MAX_REMOTE_CANDIDATES]

    decision = "reject"
    chosen: dict[str, Any] | None = None
    reasons: list[str] = []

    if local:
        # A strong match must have actual semantic relationship to the spoken
        # context. Timing, confidence, and visual/editorial quality can raise
        # usefulness, but they cannot manufacture semantic proof.
        if base["speech_excerpt"].strip():
            if (
                local["score"] >= STRONG_LOCAL_THRESHOLD
                and local["relationship_score"] >= 0.55
            ):
                decision = "strong_match"
                chosen = dict(local)
                chosen["raw_score"] = local["score"]
                chosen["ranking_score"] = local["score"]
                chosen["match_strength"] = "semantic"
                reasons.append("local source visual has strong editorial-semantic support")
            elif (
                local["score"] >= LOCAL_CANDIDATE_THRESHOLD
                and local["relationship_score"] >= 0.25
            ):
                decision = "candidate"
                chosen = dict(local)
                chosen["raw_score"] = local["score"]
                chosen["ranking_score"] = local["score"]
                chosen["match_strength"] = "semantic"
                reasons.append("local source visual has enough semantic support to be evaluated by the sequence director")
            else:
                reasons.append("local source visual does not have enough semantic relationship to justify the insert")
        else:
            # Speech-free visuals are never semantic proof. They may remain a
            # contextual candidate, but deliberately cannot reach strong_match.
            if local["score"] >= LOCAL_CANDIDATE_THRESHOLD:
                contextual = dict(local)
                contextual["raw_score"] = local["score"]
                contextual["score"] = round(min(local["score"], CONTEXTUAL_MAX_RANKING_SCORE), 6)
                contextual["ranking_score"] = contextual["score"]
                contextual["match_strength"] = "contextual"
                chosen = contextual
                decision = "contextual_candidate"
                reasons.append("speech-free insert is contextual/editorial support only; timing cannot make it a strong match")
            else:
                reasons.append("speech-free source visual is not strong enough even as contextual support")
    else:
        reasons.append("event source shot has no usable semantic metadata")

    if local and not base["speech_excerpt"].strip():
        reasons.append("speech-free event cannot become a strong semantic match from timing alone")

    if remotes:
        best_remote = remotes[0]
        local_score = local["score"] if local else 0.0
        if (
            best_remote["score"] >= REMOTE_THRESHOLD
            and best_remote["score"] >= local_score + REMOTE_MARGIN
        ):
            decision = "remote_candidate"
            chosen = dict(best_remote)
            chosen["raw_score"] = best_remote["score"]
            chosen["ranking_score"] = best_remote["score"]
            chosen["match_strength"] = "semantic"
            reasons.append("remote visual is substantially stronger and speech contains an explicit visual-reference cue")

    return {
        **base,
        "decision": decision,
        "explicit_visual_reference": visual_reference,
        "source_shot_id": resolved_id,
        "source_shot_resolution": resolution,
        "source_shot_match": local,
        "remote_matches": remotes,
        "chosen_match": chosen,
        "decision_reason": reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Editorial semantic matcher v3")
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--beats", required=True)
    parser.add_argument("--shot-content", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Transcript is accepted for interface compatibility and future expansion.
    _ = load_json(args.transcript)
    beat_data = load_json(args.beats)
    shot_data = load_json(args.shot_content)

    beats = normalize_beats(beat_data)
    shots = normalize_shots(shot_data)

    matches = [build_match(beat, idx, shots) for idx, beat in enumerate(beats)]
    relevant = [m for m in matches if m["insert_relevant"]]

    result = {
        "version": VERSION,
        "mode": "editorial_semantic_matcher_v3",
        "policy": {
            "editing_decision_made_here": False,
            "source_shot_is_primary_candidate": True,
            "relationship_scoring_enabled": True,
            "concept_families_enabled": True,
            "speech_free_inserts_are_contextual_only": True,
            "semantic_relationship_required_for_strong_match": True,
            "contextual_candidates_rank_below_semantic_candidates": True,
            "remote_override_requires_explicit_visual_reference": True,
            "remote_override_requires_margin": REMOTE_MARGIN,
            "failed_vision_records_ignored": True,
            "final_sequence_director_remains_authoritative": True,
        },
        "thresholds": {
            "min_content_confidence": MIN_CONTENT_CONFIDENCE,
            "local_candidate": LOCAL_CANDIDATE_THRESHOLD,
            "strong_local": STRONG_LOCAL_THRESHOLD,
            "contextual_max_ranking_score": CONTEXTUAL_MAX_RANKING_SCORE,
            "remote": REMOTE_THRESHOLD,
            "remote_margin": REMOTE_MARGIN,
        },
        "stats": {
            "beats": len(beats),
            "insert_relevant_beats": len(relevant),
            "usable_visual_shots": sum(
                bool(s.get("visual_summary")) and s.get("content_confidence", 0.0) >= MIN_CONTENT_CONFIDENCE
                for s in shots
            ),
            "source_shots_with_semantic_metadata": sum(bool(m.get("source_shot_match")) for m in relevant),
            "source_shot_ids_recovered_from_timing": sum(
                m.get("source_shot_resolution") == "timing_recovered" for m in relevant
            ),
            "strong_matches": sum(m["decision"] == "strong_match" for m in relevant),
            "local_candidates": sum(m["decision"] == "candidate" for m in relevant),
            "contextual_candidates": sum(m["decision"] == "contextual_candidate" for m in relevant),
            "remote_candidates": sum(m["decision"] == "remote_candidate" for m in relevant),
            "rejected": sum(m["decision"] == "reject" for m in relevant),
        },
        "matches": matches,
    }

    save_json(args.output, result)

    print("=== EDITORIAL SEMANTIC MATCHER V3 ===")
    print(json.dumps({"output": args.output, **result["stats"]}, indent=2))
    print("\nInsert decisions:")
    for m in relevant:
        chosen = m.get("chosen_match")
        if chosen:
            raw = chosen.get("raw_score", chosen.get("score", 0.0))
            rank = chosen.get("ranking_score", chosen.get("score", 0.0))
            score_label = "rank" if abs(rank - raw) > 1e-9 else "score"
            raw_label = f"raw={raw:.2f} | " if abs(rank - raw) > 1e-9 else ""
            print(
                f"  {m['start']:.3f}-{m['end']:.3f} | {m['decision']} | "
                f"{score_label}={rank:.2f} | {raw_label}"
                f"shot={chosen['shot_id']} | relation={chosen['relationship_score']:.2f} | "
                f'\"{chosen["visual_summary"]}\"'
            )
        else:
            print(
                f"  {m['start']:.3f}-{m['end']:.3f} | reject | "
                f"source_shot={m.get('source_shot_id') or 'unknown'} | "
                f"reason={'; '.join(m.get('decision_reason', []))}"
            )


if __name__ == "__main__":
    main()
