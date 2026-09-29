#!/usr/bin/env python3
"""
Groq Editorial Engine v1

Purpose
-------
Turn a long-form transcript into a ranked set of genuinely self-contained
short-form moments WITHOUT lowering editorial standards to satisfy a count.

Architecture
------------
1. Load transcript units from JSON.
2. Ask Groq to discover potentially valuable complete moments from transcript windows.
3. Merge/deduplicate overlapping proposals.
4. Run a global boundary-refinement pass over the strongest proposals.
5. Run an adversarial critic pass on each refined proposal.
6. Deterministically validate timestamps, transcript coverage, completeness flags,
   and quality floor.
7. Rank surviving moments and return top-N requested by the user.
8. Never manufacture weak fillers just to satisfy N.

Important
---------
This script does NOT render video. It produces editorial decisions only.
Downstream AV/visual/render systems should consume the resulting JSON.

Environment
-----------
GROQ_API_KEY must be set.

Example
-------
python groq_editorial_engine_v1.py \
  --transcript tmp/new_aligned_transcript.json \
  --video-end 1001.371 \
  --count 3 \
  --output tmp/groq_editorial_v1.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from groq import Groq
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-120b"
VERSION = "1.0"

# Editorial policy. These are validation policies, NOT a quality-ranking substitute.
MIN_DURATION = 4.0
MAX_DURATION = 90.0
MIN_QUALITY_SCORE = 0.78
MIN_CONFIDENCE = 0.82
MAX_OVERLAP_RATIO = 0.55
DISCOVERY_WINDOW_UNITS = 45
DISCOVERY_OVERLAP_UNITS = 10
MAX_DISCOVERY_CANDIDATES_PER_WINDOW = 8
MAX_GLOBAL_CANDIDATES = 80
MAX_CRITIC_CANDIDATES = 30
TOP_RETURN_LIMIT = 100


class EngineError(RuntimeError):
    pass


@dataclass(frozen=True)
class Unit:
    index: int
    start: float
    end: float
    text: str
    speaker: str = ""


@dataclass
class Candidate:
    candidate_id: str
    start: float
    end: float
    title: str
    moment_type: str
    summary: str
    start_reason: str
    end_reason: str
    complete: bool
    quality_score: float
    confidence: float
    context_score: float
    payoff_score: float
    originality_score: float
    boundary_confidence: float
    source_unit_start: int
    source_unit_end: int
    transcript_text: str
    critic_status: str = "unreviewed"
    critic_notes: str = ""
    rejection_reasons: List[str] | None = None

    def __post_init__(self) -> None:
        if self.rejection_reasons is None:
            self.rejection_reasons = []

    @property
    def duration(self) -> float:
        return self.end - self.start


# ---------------------------------------------------------------------------
# JSON / transcript handling
# ---------------------------------------------------------------------------


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _find_unit_records(data: Any) -> List[Any]:
    if isinstance(data, list):
        return data

    if not isinstance(data, dict):
        raise EngineError("Transcript JSON must be an object or array.")

    for key in ("units", "segments", "transcript", "utterances"):
        value = data.get(key)
        if isinstance(value, list) and value:
            return value

    # Some alignment files store an object keyed by integer-like IDs.
    for key, value in data.items():
        if isinstance(value, list) and value:
            sample = value[0]
            if isinstance(sample, dict) and any(
                k in sample for k in ("start", "start_time", "timestamp")
            ):
                return value

    raise EngineError(
        "Could not locate transcript units. Expected one of: units, segments, "
        "transcript, utterances, or a top-level list."
    )


def load_units(path: Path, video_end: float) -> List[Unit]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = _find_unit_records(data)

    units: List[Unit] = []
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            continue

        start = _as_float(
            record.get("start", record.get("start_time", record.get("timestamp")))
        )
        end = _as_float(record.get("end", record.get("end_time")))

        text = record.get("text")
        if text is None:
            text = record.get("content", record.get("sentence", ""))

        if start is None or end is None:
            continue
        if end <= start:
            continue
        if not text or not str(text).strip():
            continue

        speaker = str(record.get("speaker", record.get("speaker_id", "")))

        units.append(
            Unit(
                index=idx,
                start=max(0.0, start),
                end=min(video_end, end),
                text=re.sub(r"\s+", " ", str(text)).strip(),
                speaker=speaker,
            )
        )

    units = [u for u in units if u.end > u.start]
    units.sort(key=lambda u: (u.start, u.end, u.index))

    if not units:
        raise EngineError("No usable timestamped transcript units were found.")

    return units


def units_in_range(units: Sequence[Unit], start: float, end: float) -> List[Unit]:
    return [u for u in units if u.start >= start - 1e-6 and u.end <= end + 1e-6]


def transcript_text(units: Sequence[Unit], start: float, end: float) -> str:
    selected = units_in_range(units, start, end)
    return " ".join(u.text for u in selected)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+(?:['’-]\w+)?\b", text, flags=re.UNICODE))


# ---------------------------------------------------------------------------
# Groq API
# ---------------------------------------------------------------------------


def groq_request(
    *,
    api_key: str,
    messages: List[Dict[str, str]],
    schema_name: str,
    schema: Dict[str, Any],
    temperature: float = 0.0,
    max_tokens: int = 12000,
    retries: int = 3,
) -> Dict[str, Any]:

    client = Groq(
        api_key=api_key,
    )

    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": schema,
                    },
                },
            )

            content = response.choices[0].message.content

            if not content:
                raise EngineError(
                    "Groq returned empty structured content."
                )

            parsed = json.loads(content)

            if not isinstance(parsed, dict):
                raise EngineError(
                    "Groq structured response was not a JSON object."
                )

            return parsed

        except Exception as exc:
            last_error = exc

            if attempt < retries:
                time.sleep(2 ** (attempt - 1))

    raise EngineError(
        f"Groq request failed after {retries} attempts: {last_error}"
    )


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


DISCOVERY_SCHEMA = {
    "type": "object",
    "properties": {
        "moments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_unit": {"type": "integer"},
                    "end_unit": {"type": "integer"},
                    "title": {"type": "string"},
                    "moment_type": {"type": "string"},
                    "summary": {"type": "string"},
                    "start_reason": {"type": "string"},
                    "end_reason": {"type": "string"},
                    "complete": {"type": "boolean"},
                    "quality_score": {"type": "number", "minimum": 0, "maximum": 1},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "context_score": {"type": "number", "minimum": 0, "maximum": 1},
                    "payoff_score": {"type": "number", "minimum": 0, "maximum": 1},
                    "originality_score": {"type": "number", "minimum": 0, "maximum": 1},
                    "boundary_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": [
                    "start_unit", "end_unit", "title", "moment_type", "summary",
                    "start_reason", "end_reason", "complete", "quality_score",
                    "confidence", "context_score", "payoff_score", "originality_score",
                    "boundary_confidence",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["moments"],
    "additionalProperties": False,
}


REFINEMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "start_unit": {"type": "integer"},
        "end_unit": {"type": "integer"},
        "complete": {"type": "boolean"},
        "start_reason": {"type": "string"},
        "end_reason": {"type": "string"},
        "boundary_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "quality_score": {"type": "number", "minimum": 0, "maximum": 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "start_unit", "end_unit", "complete", "start_reason", "end_reason",
        "boundary_confidence", "quality_score", "confidence",
    ],
    "additionalProperties": False,
}


CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "publishable": {"type": "boolean"},
        "quality_score": {"type": "number", "minimum": 0, "maximum": 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "boundary_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "has_context_problem": {"type": "boolean"},
        "has_incomplete_ending": {"type": "boolean"},
        "has_incomplete_setup": {"type": "boolean"},
        "has_wasted_material": {"type": "boolean"},
        "has_repetitive_material": {"type": "boolean"},
        "has_random_excerpt_feel": {"type": "boolean"},
        "has_duplicate_story": {"type": "boolean"},
        "notes": {"type": "string"},
    },
    "required": [
        "publishable", "quality_score", "confidence", "boundary_confidence",
        "has_context_problem", "has_incomplete_ending", "has_incomplete_setup",
        "has_wasted_material", "has_repetitive_material", "has_random_excerpt_feel",
        "has_duplicate_story", "notes",
    ],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

DISCOVERY_SYSTEM = r"""
You are the lead short-form editorial director for a professional AI video product.

Your job is NOT to manufacture viral clips. Your job is to identify genuinely
valuable, self-contained moments inside long-form conversation.

Editorial standard:
- Prefer complete ideas over arbitrary shortness.
- Start at the earliest point needed for the viewer to understand the moment.
- End only after the thought, answer, anecdote, example, realization, joke, or exchange
  has naturally landed.
- Never end mid-sentence, mid-example, mid-argument, or before the consequence/payoff.
- Do not add the next unrelated question merely to increase duration.
- Do not cut a strong moment just because it is longer than a typical Short.
- Do not invent anything that is not supported by the transcript.
- A weak moment must be omitted rather than promoted just to satisfy a requested count.
- Similar windows around the same story are one underlying editorial moment.

Think like a human editor selecting footage for publication.
"""

REFINEMENT_SYSTEM = r"""
You are a senior editorial boundary specialist.

Given a proposed moment and the surrounding transcript, refine its exact boundaries.
The output must identify the smallest contiguous transcript span that still gives a
complete, compelling, self-contained viewer experience.

Rules:
1. Preserve necessary setup.
2. Preserve the entire central example/story.
3. Preserve the actual payoff or realization.
4. End at the first natural landing after the payoff.
5. Never stop at a grammatically dependent phrase.
6. Never include the next unrelated question.
7. Do not alter or invent transcript text.
8. If the proposal is not genuinely complete and cannot be repaired locally, mark
   complete=false.
"""

CRITIC_SYSTEM = r"""
You are an adversarial professional short-form editor reviewing a proposed clip.
Your job is to reject clips that would make a paying customer think the product
produced a random, worthless, incomplete, repetitive, context-dependent, or badly
bounded Short.

Reject when:
- the viewer lacks required context;
- the ending cuts off a thought or story;
- the clip stops before the payoff;
- the clip contains unnecessary filler;
- the clip is just an arbitrary excerpt;
- the clip duplicates another story;
- the moment is interesting only because of the surrounding conversation;
- the timestamps do not represent the complete editorial unit.

Be strict. It is better to return fewer clips than weak clips.
"""


def make_discovery_prompt(window_units: Sequence[Unit], window_start: int, window_end: int) -> str:
    lines = []
    for u in window_units:
        speaker = f" [{u.speaker}]" if u.speaker else ""
        lines.append(f"UNIT {u.index} | {u.start:.3f} -> {u.end:.3f}{speaker} | {u.text}")

    return f"""
Analyze transcript units {window_start} through {window_end}.

Find up to {MAX_DISCOVERY_CANDIDATES_PER_WINDOW} genuinely strong self-contained
editorial moments.

Do NOT force a candidate for every interesting sentence.
Do NOT optimize for quantity.
Do NOT use a generic hook score as a substitute for completeness.

Transcript:
{chr(10).join(lines)}
"""


def make_refinement_prompt(candidate: Candidate, surrounding: Sequence[Unit]) -> str:
    lines = []
    for u in surrounding:
        marker = " <-- CURRENT CANDIDATE" if candidate.start - 1e-6 <= u.start and u.end <= candidate.end + 1e-6 else ""
        lines.append(f"UNIT {u.index} | {u.start:.3f} -> {u.end:.3f} | {u.text}{marker}")

    return f"""
Refine this proposed moment.

Proposed title: {candidate.title}
Proposed summary: {candidate.summary}
Proposed span: {candidate.start:.3f} -> {candidate.end:.3f}

Surrounding transcript:
{chr(10).join(lines)}

Return the exact start/end UNIT indexes for the complete editorial moment.
"""


def make_critic_prompt(candidate: Candidate, story_context: str, other_titles: Sequence[str]) -> str:
    titles = "\n".join(f"- {t}" for t in other_titles[:12]) or "- none"
    return f"""
Review this proposed Short.

TITLE: {candidate.title}
TYPE: {candidate.moment_type}
SPAN: {candidate.start:.3f} -> {candidate.end:.3f}

TRANSCRIPT:
{candidate.transcript_text}

WHY IT STARTS:
{candidate.start_reason}

WHY IT ENDS:
{candidate.end_reason}

SUMMARY:
{candidate.summary}

Other selected/proposed story titles for duplicate checking:
{titles}

Evaluate whether a professional editor would actually publish this exact span.
"""


# ---------------------------------------------------------------------------
# Candidate conversion / validation
# ---------------------------------------------------------------------------


def units_for_indices(units: Sequence[Unit], start_idx: int, end_idx: int) -> List[Unit]:
    selected = [u for u in units if start_idx <= u.index <= end_idx]
    selected.sort(key=lambda u: (u.start, u.end, u.index))
    return selected


def build_candidate(
    raw: Dict[str, Any],
    units: Sequence[Unit],
    candidate_id: str,
) -> Candidate | None:
    start_idx = int(raw["start_unit"])
    end_idx = int(raw["end_unit"])
    if end_idx < start_idx:
        return None

    selected = units_for_indices(units, start_idx, end_idx)
    if not selected:
        return None

    start = selected[0].start
    end = selected[-1].end
    text = normalize(" ".join(u.text for u in selected))

    return Candidate(
        candidate_id=candidate_id,
        start=start,
        end=end,
        title=normalize(str(raw["title"])),
        moment_type=normalize(str(raw["moment_type"])),
        summary=normalize(str(raw["summary"])),
        start_reason=normalize(str(raw["start_reason"])),
        end_reason=normalize(str(raw["end_reason"])),
        complete=bool(raw["complete"]),
        quality_score=float(raw["quality_score"]),
        confidence=float(raw["confidence"]),
        context_score=float(raw["context_score"]),
        payoff_score=float(raw["payoff_score"]),
        originality_score=float(raw["originality_score"]),
        boundary_confidence=float(raw["boundary_confidence"]),
        source_unit_start=selected[0].index,
        source_unit_end=selected[-1].index,
        transcript_text=text,
    )


def interval_overlap(a: Candidate, b: Candidate) -> float:
    left = max(a.start, b.start)
    right = min(a.end, b.end)
    intersection = max(0.0, right - left)
    shorter = min(a.duration, b.duration)
    if shorter <= 0:
        return 0.0
    return intersection / shorter


def story_similarity(a: Candidate, b: Candidate) -> float:
    a_tokens = set(re.findall(r"\b[a-z0-9']+\b", a.summary.lower()))
    b_tokens = set(re.findall(r"\b[a-z0-9']+\b", b.summary.lower()))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def dedupe_candidates(candidates: Sequence[Candidate]) -> List[Candidate]:
    kept: List[Candidate] = []
    ordered = sorted(
        candidates,
        key=lambda c: (
            c.quality_score,
            c.confidence,
            c.boundary_confidence,
            c.payoff_score,
        ),
        reverse=True,
    )

    for candidate in ordered:
        duplicate = False
        for existing in kept:
            if interval_overlap(candidate, existing) >= MAX_OVERLAP_RATIO:
                duplicate = True
                break
            if story_similarity(candidate, existing) >= 0.72:
                # Similar story concepts should not be returned as multiple Shorts.
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
        if len(kept) >= MAX_GLOBAL_CANDIDATES:
            break

    return kept


def validate_candidate(candidate: Candidate, video_end: float) -> List[str]:
    reasons: List[str] = []

    if candidate.start < 0:
        reasons.append("start_before_zero")
    if candidate.end > video_end + 1e-6:
        reasons.append("end_after_video")
    if candidate.end <= candidate.start:
        reasons.append("invalid_interval")
    if candidate.duration < MIN_DURATION:
        reasons.append("too_short")
    if candidate.duration > MAX_DURATION:
        reasons.append("too_long")
    if not candidate.complete:
        reasons.append("model_marked_incomplete")
    if candidate.confidence < MIN_CONFIDENCE:
        reasons.append("low_confidence")
    if candidate.boundary_confidence < MIN_CONFIDENCE:
        reasons.append("low_boundary_confidence")
    if candidate.quality_score < MIN_QUALITY_SCORE:
        reasons.append("below_quality_floor")
    if word_count(candidate.transcript_text) < 8:
        reasons.append("insufficient_content")

    # Strong textual continuation signals. This is only a safety net; semantic
    # completion remains an LLM editorial decision.
    tail = candidate.transcript_text.lower().strip()
    words = re.findall(r"\b\w+(?:['’-]\w+)?\b", tail)
    if words:
        last_word = words[-1]
        dangerous_endings = {
            "because", "although", "though", "if", "when", "while", "that",
            "which", "who", "where", "to", "for", "with", "without", "of",
            "from", "into", "between", "through", "and", "but", "or",
        }
        if last_word in dangerous_endings:
            reasons.append("linguistic_continuation_signal")

    if re.search(r"(?:,|:|;|[-—])\s*$", candidate.transcript_text):
        reasons.append("trailing_incomplete_punctuation")

    return sorted(set(reasons))


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def discover_candidates(units: Sequence[Unit], api_key: str) -> List[Candidate]:
    raw_candidates: List[Candidate] = []
    n = len(units)
    step = max(1, DISCOVERY_WINDOW_UNITS - DISCOVERY_OVERLAP_UNITS)

    for begin in range(0, n, step):
        end_exclusive = min(n, begin + DISCOVERY_WINDOW_UNITS)
        window = units[begin:end_exclusive]
        if not window:
            break

        result = groq_request(
            api_key=api_key,
            messages=[
                {"role": "system", "content": DISCOVERY_SYSTEM},
                {
                    "role": "user",
                    "content": make_discovery_prompt(
                        window,
                        window[0].index,
                        window[-1].index,
                    ),
                },
            ],
            schema_name="editorial_moment_discovery",
            schema=DISCOVERY_SCHEMA,
            temperature=0.0,
            max_tokens=10000,
        )

        for j, raw in enumerate(result.get("moments", []), 1):
            candidate = build_candidate(
                raw,
                units,
                f"d{begin:04d}_{j:02d}",
            )
            if candidate is not None:
                raw_candidates.append(candidate)

        if end_exclusive >= n:
            break

    return dedupe_candidates(raw_candidates)


def refine_candidates(candidates: Sequence[Candidate], units: Sequence[Unit], api_key: str) -> List[Candidate]:
    refined: List[Candidate] = []

    shortlist = sorted(
        candidates,
        key=lambda c: (
            c.quality_score,
            c.payoff_score,
            c.context_score,
            c.confidence,
        ),
        reverse=True,
    )[:MAX_CRITIC_CANDIDATES]

    for candidate in shortlist:
        neighborhood = [
            u for u in units
            if u.end >= candidate.start - 12.0
            and u.start <= candidate.end + 12.0
        ]

        result = groq_request(
            api_key=api_key,
            messages=[
                {"role": "system", "content": REFINEMENT_SYSTEM},
                {
                    "role": "user",
                    "content": make_refinement_prompt(candidate, neighborhood),
                },
            ],
            schema_name="editorial_boundary_refinement",
            schema=REFINEMENT_SCHEMA,
            temperature=0.0,
            max_tokens=4000,
        )

        selected = units_for_indices(
            units,
            int(result["start_unit"]),
            int(result["end_unit"]),
        )
        if not selected:
            candidate.rejection_reasons.append("refinement_no_units")
            continue

        candidate.start = selected[0].start
        candidate.end = selected[-1].end
        candidate.source_unit_start = selected[0].index
        candidate.source_unit_end = selected[-1].index
        candidate.transcript_text = normalize(" ".join(u.text for u in selected))
        candidate.complete = bool(result["complete"])
        candidate.start_reason = normalize(str(result["start_reason"]))
        candidate.end_reason = normalize(str(result["end_reason"]))
        candidate.boundary_confidence = float(result["boundary_confidence"])
        candidate.quality_score = min(candidate.quality_score, float(result["quality_score"]))
        candidate.confidence = min(candidate.confidence, float(result["confidence"]))

        refined.append(candidate)

    return dedupe_candidates(refined)


def critic_candidates(candidates: Sequence[Candidate], api_key: str) -> List[Candidate]:
    reviewed: List[Candidate] = []

    ordered = sorted(
        candidates,
        key=lambda c: (
            c.quality_score,
            c.payoff_score,
            c.boundary_confidence,
            c.confidence,
        ),
        reverse=True,
    )[:MAX_CRITIC_CANDIDATES]

    titles = [c.title for c in ordered]

    for candidate in ordered:
        result = groq_request(
            api_key=api_key,
            messages=[
                {"role": "system", "content": CRITIC_SYSTEM},
                {
                    "role": "user",
                    "content": make_critic_prompt(candidate, candidate.transcript_text, titles),
                },
            ],
            schema_name="editorial_moment_critic",
            schema=CRITIC_SCHEMA,
            temperature=0.0,
            max_tokens=4500,
        )

        candidate.critic_status = "passed" if result["publishable"] else "rejected"
        candidate.critic_notes = normalize(str(result["notes"]))
        candidate.quality_score = min(candidate.quality_score, float(result["quality_score"]))
        candidate.confidence = min(candidate.confidence, float(result["confidence"]))
        candidate.boundary_confidence = min(
            candidate.boundary_confidence,
            float(result["boundary_confidence"]),
        )

        if result["has_context_problem"]:
            candidate.rejection_reasons.append("critic_context_problem")
        if result["has_incomplete_ending"]:
            candidate.rejection_reasons.append("critic_incomplete_ending")
        if result["has_incomplete_setup"]:
            candidate.rejection_reasons.append("critic_incomplete_setup")
        if result["has_wasted_material"]:
            candidate.rejection_reasons.append("critic_wasted_material")
        if result["has_repetitive_material"]:
            candidate.rejection_reasons.append("critic_repetitive_material")
        if result["has_random_excerpt_feel"]:
            candidate.rejection_reasons.append("critic_random_excerpt")
        if result["has_duplicate_story"]:
            candidate.rejection_reasons.append("critic_duplicate_story")
        if not result["publishable"]:
            candidate.rejection_reasons.append("critic_rejected")

        reviewed.append(candidate)

    return reviewed


def final_score(candidate: Candidate) -> float:
    # Quality dominates. No duration reward is allowed to overpower editorial quality.
    score = (
        0.40 * candidate.quality_score
        + 0.18 * candidate.payoff_score
        + 0.14 * candidate.context_score
        + 0.12 * candidate.originality_score
        + 0.10 * candidate.boundary_confidence
        + 0.06 * candidate.confidence
    )
    return max(0.0, min(1.0, score))


def accept_candidate(candidate: Candidate, video_end: float) -> bool:
    reasons = validate_candidate(candidate, video_end)
    reasons.extend(candidate.rejection_reasons or [])
    candidate.rejection_reasons = sorted(set(reasons))

    return not candidate.rejection_reasons and candidate.critic_status == "passed"


def rank_and_select(candidates: Sequence[Candidate], requested_count: int, video_end: float) -> Tuple[List[Candidate], List[Candidate]]:
    accepted = [c for c in candidates if accept_candidate(c, video_end)]

    accepted.sort(
        key=lambda c: (
            final_score(c),
            c.quality_score,
            c.payoff_score,
            c.boundary_confidence,
            c.confidence,
        ),
        reverse=True,
    )

    # Final diversity pass. Do not return multiple near-identical stories.
    selected: List[Candidate] = []
    for candidate in accepted:
        too_close = False
        for existing in selected:
            if interval_overlap(candidate, existing) >= 0.45:
                too_close = True
                break
            if story_similarity(candidate, existing) >= 0.68:
                too_close = True
                break
        if too_close:
            candidate.rejection_reasons.append("final_diversity_duplicate")
            continue
        selected.append(candidate)
        if len(selected) >= min(requested_count, TOP_RETURN_LIMIT):
            break

    return selected, accepted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Groq Editorial Engine v1")
    parser.add_argument("--transcript", required=True, type=Path)
    parser.add_argument("--video-end", required=True, type=float)
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.count <= 0:
        raise SystemExit("--count must be greater than 0")
    if args.count > TOP_RETURN_LIMIT:
        raise SystemExit(f"--count cannot exceed {TOP_RETURN_LIMIT}")
    if not args.transcript.exists():
        raise SystemExit(f"Transcript not found: {args.transcript}")
    if args.video_end <= 0:
        raise SystemExit("--video-end must be positive")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY is not set.")

    units = load_units(args.transcript, args.video_end)

    print("=" * 72)
    print("GROQ EDITORIAL ENGINE v1")
    print("=" * 72)
    print(f"Transcript units: {len(units)}")
    print(f"Requested clips:  {args.count}")
    print()
    print("Stage 1/4: discovering complete editorial moments...")
    discovered = discover_candidates(units, api_key)
    print(f"  discovered: {len(discovered)}")

    print("Stage 2/4: refining exact editorial boundaries...")
    refined = refine_candidates(discovered, units, api_key)
    print(f"  refined: {len(refined)}")

    print("Stage 3/4: adversarial editorial review...")
    reviewed = critic_candidates(refined, api_key)
    print(f"  reviewed: {len(reviewed)}")

    print("Stage 4/4: deterministic validation + ranking...")
    selected, accepted = rank_and_select(reviewed, args.count, args.video_end)
    print(f"  accepted: {len(accepted)}")
    print(f"  selected: {len(selected)}")
    print()

    output = {
        "version": VERSION,
        "engine": "groq_editorial_engine",
        "model": MODEL,
        "video_end": args.video_end,
        "requested_count": args.count,
        "discovered_candidates": len(discovered),
        "refined_candidates": len(refined),
        "reviewed_candidates": len(reviewed),
        "accepted_candidates": len(accepted),
        "selected_count": len(selected),
        "quality_floor": MIN_QUALITY_SCORE,
        "confidence_floor": MIN_CONFIDENCE,
        "selection_policy": "requested_count_never_lowers_quality_floor",
        "abstained_from_fillers": len(selected) < args.count,
        "clips": [],
        "rejected_candidates": [],
    }

    for rank, candidate in enumerate(selected, 1):
        clip = asdict(candidate)
        clip["rank"] = rank
        clip["score"] = round(final_score(candidate), 6)
        clip["duration"] = round(candidate.duration, 3)
        output["clips"].append(clip)

    selected_ids = {c.candidate_id for c in selected}
    for candidate in reviewed:
        if candidate.candidate_id in selected_ids:
            continue
        rejected = asdict(candidate)
        rejected["score"] = round(final_score(candidate), 6)
        rejected["duration"] = round(candidate.duration, 3)
        output["rejected_candidates"].append(rejected)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("FINAL RANKING")
    print("-" * 72)
    if not selected:
        print("No publishable moments met the quality floor.")
    else:
        for clip in output["clips"]:
            print(
                f"#{clip['rank']} | {clip['start']:.3f} -> {clip['end']:.3f} | "
                f"score={clip['score']:.3f} | {clip['title']}"
            )

    print()
    if len(selected) < args.count:
        print(
            f"QUALITY ABSTENTION: requested {args.count}, but only "
            f"{len(selected)} moments passed the quality floor."
        )

    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
    except EngineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)