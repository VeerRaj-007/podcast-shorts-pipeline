"""
Step 3: General Content Intelligence Picker v2

Pipeline:

Transcript
    ↓
5-minute windows
    ↓
AI candidate generation
    ↓
Persistent candidate cache
    ↓
Candidate validation + deduplication
    ↓
Local ranking
    ↓
Optional compact AI final ranking
    ↓
Best N short-form clips

Designed to work with:
- Podcasts
- Interviews
- Gaming
- Tutorials
- Educational videos
- Commentary
- Debates
- Vlogs
- Documentaries
- Tech videos
- Storytelling
- General long-form content

Important v2 improvements:
- Persistent candidate caching
- Resumable window processing
- Transcript fingerprinting
- Immediate rate-limit detection
- No pointless retries after quota exhaustion
- Local ranking fallback
- Optional compact AI final ranking
- Same clips.json output format
- Same select_clips() public API
"""

import hashlib
import json
import os
import sys
import time
from typing import Any

from groq import Groq


# ============================================================
# CONFIG
# ============================================================

SELECTION_MODEL = "openai/gpt-oss-120b"

WINDOW_SECONDS = 5 * 60

# Only retry transient failures.
MAX_RETRIES = 2
RETRY_DELAY = 4

MIN_CLIP_DURATION = 15
MAX_CLIP_DURATION = 150

MAX_CANDIDATES_PER_WINDOW = 3

CANDIDATES_CACHE_PATH = "tmp/candidates.json"
CLIPS_OUTPUT_PATH = "tmp/clips.json"


# ============================================================
# HELPERS
# ============================================================


def _extract_json(text: str) -> Any:
    """
    Extract JSON from an LLM response.

    Handles:
    - plain JSON
    - ```json ... ```
    - ``` ... ```
    - extra surrounding text
    """

    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    first_object = text.find("{")
    last_object = text.rfind("}")

    if first_object != -1 and last_object != -1:
        candidate = text[first_object:last_object + 1]

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    first_array = text.find("[")
    last_array = text.rfind("]")

    if first_array != -1 and last_array != -1:
        candidate = text[first_array:last_array + 1]

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError(
        "Could not extract valid JSON from model response."
    )


def _is_rate_limit_error(exc: Exception) -> bool:
    """
    Detect Groq/API errors where retrying immediately is pointless.

    Covers:
    - 429
    - TPM
    - TPD
    - daily token limits
    - quota/rate-limit wording
    """

    text = str(exc).lower()

    rate_limit_terms = [
        "429",
        "rate limit",
        "rate_limit",
        "too many requests",
        "tokens per minute",
        "tpm",
        "tokens per day",
        "tpd",
        "daily token",
        "quota",
        "exceeded",
    ]

    return any(
        term in text
        for term in rate_limit_terms
    )


def _call_model(
    client: Groq,
    prompt: str,
    *,
    max_tokens: int = 2500,
) -> Any:
    """
    Call Groq.

    Rate-limit/quota errors are raised immediately.

    Only ordinary transient failures are retried.
    """

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=SELECTION_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert AI short-form video "
                            "editor and content intelligence system. "
                            "Return ONLY valid JSON."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.2,
                max_tokens=max_tokens,
            )

            content = response.choices[0].message.content

            if not content:
                raise ValueError(
                    "Model returned an empty response."
                )

            return _extract_json(content)

        except Exception as exc:
            last_error = exc

            if _is_rate_limit_error(exc):
                print(
                    "\n  API rate/quota limit detected."
                    "\n  Stopping API calls immediately."
                )

                raise

            print(
                f"  Model request failed "
                f"(attempt {attempt}/{MAX_RETRIES}): {exc}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    raise RuntimeError(
        f"Model request failed after "
        f"{MAX_RETRIES} attempts: {last_error}"
    )


# ============================================================
# TRANSCRIPT FINGERPRINT
# ============================================================


def _transcript_fingerprint(
    segments: list[dict],
) -> str:
    """
    Create a stable fingerprint for the transcript.

    This prevents candidates from one video being reused for
    another video.
    """

    normalized = []

    for index, segment in enumerate(segments):
        normalized.append(
            {
                "index": index,
                "start": float(
                    segment.get("start", 0)
                ),
                "end": float(
                    segment.get("end", 0)
                ),
                "text": str(
                    segment.get("text", "")
                ).strip(),
            }
        )

    raw = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


# ============================================================
# CACHE
# ============================================================


def _load_candidate_cache(
    transcript_hash: str,
) -> dict:
    """
    Load candidate cache if it belongs to the current transcript.
    """

    if not os.path.exists(CANDIDATES_CACHE_PATH):
        return {
            "transcript_hash": transcript_hash,
            "windows": {},
        }

    try:
        with open(
            CANDIDATES_CACHE_PATH,
            "r",
            encoding="utf-8",
        ) as file:
            cache = json.load(file)

    except Exception as exc:
        print(
            f"Warning: could not read candidate cache: {exc}"
        )

        return {
            "transcript_hash": transcript_hash,
            "windows": {},
        }

    if not isinstance(cache, dict):
        return {
            "transcript_hash": transcript_hash,
            "windows": {},
        }

    cached_hash = cache.get("transcript_hash")

    if cached_hash != transcript_hash:
        print(
            "\nCandidate cache belongs to a different transcript."
        )

        return {
            "transcript_hash": transcript_hash,
            "windows": {},
        }

    windows = cache.get("windows", {})

    if not isinstance(windows, dict):
        windows = {}

    return {
        "transcript_hash": transcript_hash,
        "windows": windows,
    }


def _save_candidate_cache(
    cache: dict,
) -> None:
    """
    Save candidate cache atomically.
    """

    os.makedirs(
        os.path.dirname(CANDIDATES_CACHE_PATH),
        exist_ok=True,
    )

    temp_path = CANDIDATES_CACHE_PATH + ".tmp"

    with open(
        temp_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            cache,
            file,
            indent=2,
            ensure_ascii=False,
        )

    os.replace(
        temp_path,
        CANDIDATES_CACHE_PATH,
    )


# ============================================================
# WINDOWING
# ============================================================


def _build_windows(
    segments: list[dict],
) -> list[list[tuple[int, dict]]]:
    """
    Split transcript into approximately 5-minute windows.

    Global segment indices are preserved.
    """

    windows: list[list[tuple[int, dict]]] = []

    current_window: list[tuple[int, dict]] = []
    current_window_start = None

    for index, segment in enumerate(segments):
        start = float(
            segment.get("start", 0)
        )

        if current_window_start is None:
            current_window_start = start

        if (
            current_window
            and start - current_window_start
            >= WINDOW_SECONDS
        ):
            windows.append(current_window)

            current_window = []
            current_window_start = start

        current_window.append(
            (index, segment)
        )

    if current_window:
        windows.append(current_window)

    return windows


def _format_window(
    window: list[tuple[int, dict]],
) -> str:
    """
    Format transcript window for the LLM.
    """

    lines = []

    for index, segment in window:
        start = float(
            segment.get("start", 0)
        )

        end = float(
            segment.get("end", start)
        )

        text = str(
            segment.get("text", "")
        ).strip()

        if not text:
            continue

        lines.append(
            f"[{index}] "
            f"({start:.2f}-{end:.2f}): "
            f"{text}"
        )

    return "\n".join(lines)


# ============================================================
# CANDIDATE VALIDATION
# ============================================================


def _safe_number(
    value: Any,
    default: float = 0.0,
) -> float:

    try:
        number = float(value)

        if number != number:
            return default

        return number

    except (TypeError, ValueError):
        return default


def _clamp_score(
    value: Any,
) -> int:

    score = _safe_number(
        value,
        0,
    )

    return max(
        0,
        min(
            10,
            round(score),
        ),
    )


def _valid_candidate(
    candidate: Any,
    segments: list[dict],
) -> bool:

    if not isinstance(candidate, dict):
        return False

    if (
        "start_index" not in candidate
        or "end_index" not in candidate
    ):
        return False

    try:
        start_index = int(
            candidate["start_index"]
        )

        end_index = int(
            candidate["end_index"]
        )

    except (TypeError, ValueError):
        return False

    if start_index < 0:
        return False

    if end_index >= len(segments):
        return False

    if start_index > end_index:
        return False

    start = float(
        segments[start_index].get(
            "start",
            0,
        )
    )

    end = float(
        segments[end_index].get(
            "end",
            start,
        )
    )

    duration = end - start

    if duration < MIN_CLIP_DURATION:
        return False

    if duration > MAX_CLIP_DURATION:
        return False

    return True


def _candidate_times(
    candidate: dict,
    segments: list[dict],
) -> tuple[float, float]:

    start_index = int(
        candidate["start_index"]
    )

    end_index = int(
        candidate["end_index"]
    )

    start = float(
        segments[start_index].get(
            "start",
            0,
        )
    )

    end = float(
        segments[end_index].get(
            "end",
            start,
        )
    )

    return start, end


def _candidate_overlaps(
    candidate_a: dict,
    candidate_b: dict,
    segments: list[dict],
) -> bool:

    start_a, end_a = _candidate_times(
        candidate_a,
        segments,
    )

    start_b, end_b = _candidate_times(
        candidate_b,
        segments,
    )

    overlap_start = max(
        start_a,
        start_b,
    )

    overlap_end = min(
        end_a,
        end_b,
    )

    overlap = max(
        0.0,
        overlap_end - overlap_start,
    )

    duration_a = max(
        0.01,
        end_a - start_a,
    )

    duration_b = max(
        0.01,
        end_b - start_b,
    )

    smaller_duration = min(
        duration_a,
        duration_b,
    )

    return (
        overlap / smaller_duration
        >= 0.5
    )


# ============================================================
# LOCAL QUALITY SCORING
# ============================================================


def _candidate_quality(
    candidate: dict,
) -> float:
    """
    Deterministic local score.

    Used for:
    - deduplication
    - fallback ranking
    - quota-safe operation
    """

    scores = candidate.get(
        "scores",
        {},
    )

    weights = {
        "hook": 1.5,
        "standalone": 1.4,
        "interestingness": 1.3,
        "emotional_impact": 1.0,
        "value": 1.0,
        "story_payoff": 1.1,
        "novelty": 1.0,
        "retention": 1.5,
        "visual_potential": 0.6,
    }

    total = 0.0
    weight_total = 0.0

    for key, weight in weights.items():
        score = _clamp_score(
            scores.get(
                key,
                0,
            )
        )

        total += score * weight
        weight_total += weight

    if weight_total == 0:
        return 0.0

    quality = (
        total / weight_total
    )

    # Context dependency is a penalty.
    context_dependency = _clamp_score(
        scores.get(
            "context_dependency",
            0,
        )
    )

    quality -= (
        context_dependency * 0.20
    )

    return quality


def _candidate_identity(
    candidate: dict,
) -> tuple[int, int]:

    return (
        int(candidate["start_index"]),
        int(candidate["end_index"]),
    )


def _deduplicate_candidates(
    candidates: list[dict],
    segments: list[dict],
) -> list[dict]:

    candidates = sorted(
        candidates,
        key=_candidate_quality,
        reverse=True,
    )

    kept: list[dict] = []

    seen = set()

    for candidate in candidates:
        identity = _candidate_identity(
            candidate
        )

        if identity in seen:
            continue

        seen.add(identity)

        overlaps = any(
            _candidate_overlaps(
                candidate,
                existing,
                segments,
            )
            for existing in kept
        )

        if not overlaps:
            kept.append(candidate)

    return kept


# ============================================================
# CANDIDATE GENERATION
# ============================================================


def _candidate_prompt(
    window_text: str,
    max_candidates: int,
) -> str:

    return f"""
You are analyzing one section of a long-form video.

The video can be ANY type of content:
- podcast
- interview
- gaming
- tutorial
- educational content
- commentary
- debate
- documentary
- vlog
- technology
- storytelling
- reaction
- lecture
- or another format.

Your job is to find the strongest moments that could become
high-quality YouTube Shorts, TikToks, or Instagram Reels.

Do NOT assume this is a podcast.

Look for:

- strong hooks
- surprising information
- powerful opinions
- interesting explanations
- useful insights
- funny moments
- emotional intensity
- conflict
- disagreement
- surprising reveals
- compelling stories
- strong conclusions
- fascinating facts
- clear transformations or lessons
- satisfying question/answer moments
- curiosity
- visually interesting events when apparent from transcript

Avoid:

- greetings
- introductions
- advertisements
- sponsor reads
- generic filler
- repetitive statements
- meaningless conversation
- long setup with no payoff
- excessive context dependency
- incomplete thoughts
- abrupt endings

A clip should generally be between
15 and 150 seconds.

Use ONLY the provided transcript segment indices.

Do not invent indices.

A clip may begin slightly before the strongest sentence
if that improves the hook.

A clip may include necessary setup before a payoff.

Generate up to {max_candidates} strong candidates.

If this section contains no genuinely good moments,
return an empty candidates array.

For every candidate provide:

- start_index
- end_index
- title
- reason
- content_type
- scores

Score every dimension from 0 to 10:

hook:
How strong is the opening?

standalone:
Can someone understand and enjoy this independently?

interestingness:
How compelling is the idea/event?

emotional_impact:
How much emotion, tension, humor, surprise, or excitement?

value:
Does the viewer learn or gain something?

story_payoff:
Does it contain setup/payoff or a satisfying progression?

novelty:
Does it feel fresh or unexpected?

retention:
How likely is a viewer to keep watching?

visual_potential:
Could this benefit from visual editing, reactions,
demonstrations, gameplay, or other visual elements?

context_dependency:
How much external context is required?
0 = completely understandable
10 = almost impossible without the full video.

Return ONLY this JSON:

{{
  "candidates": [
    {{
      "start_index": 10,
      "end_index": 20,
      "title": "Short descriptive title",
      "reason": "Why this moment has strong short-form potential.",
      "content_type": "story",
      "scores": {{
        "hook": 9,
        "standalone": 8,
        "interestingness": 9,
        "emotional_impact": 6,
        "value": 8,
        "story_payoff": 9,
        "novelty": 8,
        "retention": 9,
        "visual_potential": 5,
        "context_dependency": 2
      }}
    }}
  ]
}}

TRANSCRIPT:

{window_text}
""".strip()


def _normalize_candidate(
    candidate: dict,
) -> dict:

    candidate["start_index"] = int(
        candidate["start_index"]
    )

    candidate["end_index"] = int(
        candidate["end_index"]
    )

    candidate["title"] = str(
        candidate.get(
            "title",
            "Untitled clip",
        )
    ).strip()

    candidate["reason"] = str(
        candidate.get(
            "reason",
            "",
        )
    ).strip()

    candidate["content_type"] = str(
        candidate.get(
            "content_type",
            "general",
        )
    ).strip()

    raw_scores = candidate.get(
        "scores",
        {},
    )

    if not isinstance(
        raw_scores,
        dict,
    ):
        raw_scores = {}

    normalized_scores = {}

    for key in [
        "hook",
        "standalone",
        "interestingness",
        "emotional_impact",
        "value",
        "story_payoff",
        "novelty",
        "retention",
        "visual_potential",
        "context_dependency",
    ]:
        normalized_scores[key] = _clamp_score(
            raw_scores.get(
                key,
                0,
            )
        )

    candidate["scores"] = (
        normalized_scores
    )

    return candidate


def _generate_candidates(
    client: Groq,
    window: list[tuple[int, dict]],
    segments: list[dict],
) -> list[dict]:

    window_text = _format_window(
        window
    )

    if not window_text.strip():
        return []

    prompt = _candidate_prompt(
        window_text,
        MAX_CANDIDATES_PER_WINDOW,
    )

    result = _call_model(
        client,
        prompt,
        max_tokens=3000,
    )

    if not isinstance(
        result,
        dict,
    ):
        return []

    candidates = result.get(
        "candidates",
        [],
    )

    if not isinstance(
        candidates,
        list,
    ):
        return []

    valid = []

    for candidate in candidates[
        :MAX_CANDIDATES_PER_WINDOW
    ]:

        if not isinstance(
            candidate,
            dict,
        ):
            continue

        if not _valid_candidate(
            candidate,
            segments,
        ):
            continue

        try:
            candidate = _normalize_candidate(
                candidate
            )
        except Exception:
            continue

        valid.append(candidate)

    return valid


# ============================================================
# FINAL RANKING
# ============================================================


def _candidate_text(
    candidate: dict,
    segments: list[dict],
    candidate_id: int,
) -> str:

    start, end = _candidate_times(
        candidate,
        segments,
    )

    duration = end - start

    scores = candidate.get(
        "scores",
        {},
    )

    return (
        f"CANDIDATE {candidate_id}\n"
        f"time: {start:.2f}-{end:.2f}\n"
        f"duration: {duration:.2f}s\n"
        f"title: {candidate.get('title', '')}\n"
        f"reason: {candidate.get('reason', '')}\n"
        f"content_type: "
        f"{candidate.get('content_type', 'general')}\n"
        f"scores: "
        f"{json.dumps(scores, separators=(',', ':'))}"
    )


def _final_ranking_prompt(
    candidates: list[dict],
    segments: list[dict],
    n: int,
) -> str:

    formatted_candidates = "\n\n".join(
        _candidate_text(
            candidate,
            segments,
            i,
        )
        for i, candidate in enumerate(
            candidates
        )
    )

    return f"""
You are the final editor of an AI short-form
video generation system.

Select the BEST {n} candidates.

Prioritize:

1. Strong opening hook
2. Standalone understanding
3. Viewer curiosity
4. Retention potential
5. Interesting or surprising ideas
6. Emotional impact
7. Story/setup/payoff
8. Useful value when appropriate
9. Novelty
10. Natural beginning and ending

Avoid:

- excessive context dependency
- weak openings
- generic statements
- repetition
- weak payoff
- awkward endings
- filler
- advertisements
- greetings
- duplicate ideas
- overlapping clips

Judge different content types appropriately.

Gaming:
exciting plays, reactions, clutch moments,
funny events, unexpected outcomes.

Educational:
surprising explanations, useful concepts,
mistakes, counterintuitive facts.

Storytelling:
conflict, tension, turning points,
reveals, emotional moments, conclusions.

Interviews:
unusual answers, opinions, personal stories,
controversy, insight, humor, revelations.

Tutorials:
useful techniques, mistakes, shortcuts,
transformations, important explanations.

DIVERSITY:

If several candidates are similarly strong,
prefer different topics or different parts
of the video.

Do not select heavily overlapping clips.

Return ONLY:

{{
  "selected": [
    {{
      "candidate_id": 0,
      "reason": "Why this candidate should be selected."
    }}
  ]
}}

You may return fewer than {n} only when necessary.

CANDIDATES:

{formatted_candidates}
""".strip()


def _final_select_with_ai(
    client: Groq,
    candidates: list[dict],
    segments: list[dict],
    n: int,
) -> list[dict]:

    if not candidates:
        return []

    if len(candidates) <= n:
        return sorted(
            candidates,
            key=lambda candidate:
                _candidate_times(
                    candidate,
                    segments,
                )[0],
        )

    prompt = _final_ranking_prompt(
        candidates,
        segments,
        n,
    )

    result = _call_model(
        client,
        prompt,
        max_tokens=1200,
    )

    if not isinstance(
        result,
        dict,
    ):
        return []

    selected = result.get(
        "selected",
        [],
    )

    if not isinstance(
        selected,
        list,
    ):
        return []

    selected_candidates = []

    used_ids: set[int] = set()

    for item in selected:

        if not isinstance(
            item,
            dict,
        ):
            continue

        try:
            candidate_id = int(
                item["candidate_id"]
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        if (
            candidate_id < 0
            or candidate_id >= len(candidates)
        ):
            continue

        if candidate_id in used_ids:
            continue

        candidate = candidates[
            candidate_id
        ]

        if any(
            _candidate_overlaps(
                candidate,
                existing,
                segments,
            )
            for existing
            in selected_candidates
        ):
            continue

        selected_candidates.append(
            candidate
        )

        used_ids.add(
            candidate_id
        )

        if len(selected_candidates) >= n:
            break

    return selected_candidates


# ============================================================
# LOCAL RANKING FALLBACK
# ============================================================


def _local_rank(
    candidates: list[dict],
    segments: list[dict],
    n: int,
) -> list[dict]:
    """
    Rank candidates without making any API call.

    Uses AI-generated candidate scores and diversity.
    """

    if not candidates:
        return []

    ranked = sorted(
        candidates,
        key=_candidate_quality,
        reverse=True,
    )

    selected: list[dict] = []

    for candidate in ranked:

        if any(
            _candidate_overlaps(
                candidate,
                existing,
                segments,
            )
            for existing in selected
        ):
            continue

        selected.append(
            candidate
        )

        if len(selected) >= n:
            break

    return selected


def _fill_missing(
    selected: list[dict],
    candidates: list[dict],
    segments: list[dict],
    n: int,
) -> list[dict]:

    result = list(selected)

    remaining = [
        candidate
        for candidate in candidates
        if candidate not in result
    ]

    remaining.sort(
        key=_candidate_quality,
        reverse=True,
    )

    for candidate in remaining:

        if len(result) >= n:
            break

        overlaps = any(
            _candidate_overlaps(
                candidate,
                existing,
                segments,
            )
            for existing in result
        )

        if overlaps:
            continue

        result.append(
            candidate
        )

    return result


# ============================================================
# CANDIDATE CACHE PROCESSING
# ============================================================


def _collect_candidates(
    client: Groq,
    segments: list[dict],
    windows: list[list[tuple[int, dict]]],
    *,
    refresh: bool = False,
) -> tuple[list[dict], bool]:
    """
    Generate candidates with persistent per-window caching.

    Returns:
        candidates
        api_blocked
    """

    transcript_hash = _transcript_fingerprint(
        segments
    )

    cache = _load_candidate_cache(
        transcript_hash
    )

    if refresh:
        print(
            "\nRefreshing candidate cache..."
        )

        cache = {
            "transcript_hash":
                transcript_hash,
            "windows": {},
        }

        _save_candidate_cache(
            cache
        )

    all_candidates: list[dict] = []

    api_blocked = False

    for window_number, window in enumerate(
        windows,
        start=1,
    ):

        key = str(
            window_number
        )

        cached = cache["windows"].get(
            key
        )

        if (
            isinstance(cached, dict)
            and cached.get("status")
            == "complete"
        ):
            cached_candidates = (
                cached.get(
                    "candidates",
                    [],
                )
            )

            if not isinstance(
                cached_candidates,
                list,
            ):
                cached_candidates = []

            print(
                f"\nWindow "
                f"{window_number}/{len(windows)} "
                f"→ CACHE HIT "
                f"({len(cached_candidates)} candidate(s))"
            )

            all_candidates.extend(
                cached_candidates
            )

            continue

        if api_blocked:
            print(
                f"\nWindow "
                f"{window_number}/{len(windows)} "
                f"→ SKIPPED "
                f"(API unavailable)"
            )

            continue

        print(
            f"\nAnalyzing window "
            f"{window_number}/{len(windows)}..."
        )

        try:
            candidates = _generate_candidates(
                client,
                window,
                segments,
            )

            print(
                f"  Found "
                f"{len(candidates)} candidate(s)."
            )

            cache["windows"][key] = {
                "status": "complete",
                "candidates": candidates,
            }

            # SAVE IMMEDIATELY.
            _save_candidate_cache(
                cache
            )

            all_candidates.extend(
                candidates
            )

        except Exception as exc:

            if _is_rate_limit_error(exc):
                print(
                    "\n  Rate/quota limit reached."
                )

                print(
                    "  Saving completed windows "
                    "and stopping API generation."
                )

                api_blocked = True

                continue

            print(
                f"  Candidate generation failed: "
                f"{exc}"
            )

            # Do not mark failed windows complete.
            # They can be retried next run.

    # Reload everything from cache to ensure
    # resumability is respected.
    all_candidates = []

    for window_number in range(
        1,
        len(windows) + 1,
    ):

        cached = cache["windows"].get(
            str(window_number)
        )

        if not isinstance(
            cached,
            dict,
        ):
            continue

        if cached.get("status") != "complete":
            continue

        cached_candidates = cached.get(
            "candidates",
            [],
        )

        if isinstance(
            cached_candidates,
            list,
        ):
            all_candidates.extend(
                cached_candidates
            )

    return (
        all_candidates,
        api_blocked,
    )


# ============================================================
# PUBLIC API
# ============================================================


def select_clips(
    segments: list[dict],
    n: int = 3,
    api_key: str | None = None,
) -> list[dict]:
    """
    Select the best N short-form moments.

    Public API intentionally remains compatible
    with the previous picker.
    """

    if not segments:
        raise ValueError(
            "Transcript is empty."
        )

    if n <= 0:
        return []

    api_key = (
        api_key
        or os.getenv("GROQ_API_KEY")
    )

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set."
        )

    client = Groq(
        api_key=api_key
    )

    windows = _build_windows(
        segments
    )

    print(
        f"Transcript contains "
        f"{len(segments)} segments "
        f"across "
        f"{len(windows)} windows."
    )

    # --------------------------------------------------------
    # Stage 1: Candidate generation + cache
    # --------------------------------------------------------

    all_candidates, api_blocked = (
        _collect_candidates(
            client,
            segments,
            windows,
        )
    )

    print(
        f"\nGenerated/cached "
        f"{len(all_candidates)} raw candidates."
    )

    # --------------------------------------------------------
    # Stage 2: Deduplication
    # --------------------------------------------------------

    all_candidates = (
        _deduplicate_candidates(
            all_candidates,
            segments,
        )
    )

    print(
        f"After overlap removal: "
        f"{len(all_candidates)} candidates."
    )

    if not all_candidates:
        print(
            "\nNo valid candidates available."
        )

        return []

    # --------------------------------------------------------
    # Stage 3: Ranking
    # --------------------------------------------------------

    selected: list[dict] = []

    if api_blocked:
        print(
            "\nUsing LOCAL ranking because "
            "the API rate/quota limit was reached."
        )

        selected = _local_rank(
            all_candidates,
            segments,
            n,
        )

    else:
        try:
            print(
                "\nRunning compact AI final ranking..."
            )

            selected = _final_select_with_ai(
                client,
                all_candidates,
                segments,
                n,
            )

            print(
                f"AI ranking selected "
                f"{len(selected)} candidate(s)."
            )

        except Exception as exc:

            print(
                f"\nAI final ranking failed: {exc}"
            )

            print(
                "Falling back to local ranking."
            )

            selected = _local_rank(
                all_candidates,
                segments,
                n,
            )

    # --------------------------------------------------------
    # Stage 4: Fill missing slots
    # --------------------------------------------------------

    if len(selected) < n:

        selected = _fill_missing(
            selected,
            all_candidates,
            segments,
            n,
        )

    # --------------------------------------------------------
    # Stage 5: Downstream format
    # --------------------------------------------------------

    clips = []

    for candidate in selected:

        start, end = _candidate_times(
            candidate,
            segments,
        )

        start_index = int(
            candidate["start_index"]
        )

        end_index = int(
            candidate["end_index"]
        )

        clip_segments = segments[
            start_index:end_index + 1
        ]

        clips.append(
            {
                "start": start,
                "end": end,
                "title": candidate.get(
                    "title",
                    "Untitled clip",
                ),
                "reason": candidate.get(
                    "reason",
                    "",
                ),
                "segments": clip_segments,
            }
        )

    clips.sort(
        key=lambda clip:
            float(clip["start"])
    )

    return clips


# ============================================================
# CLI
# ============================================================


def main() -> None:

    transcript_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        and not sys.argv[1].startswith("--")
        else "tmp/transcript.json"
    )

    refresh = (
        "--refresh"
        in sys.argv
    )

    output_path = (
        CLIPS_OUTPUT_PATH
    )

    if not os.path.exists(
        transcript_path
    ):
        raise FileNotFoundError(
            f"Transcript not found: "
            f"{transcript_path}"
        )

    with open(
        transcript_path,
        "r",
        encoding="utf-8",
    ) as file:

        segments = json.load(
            file
        )

    if isinstance(
        segments,
        dict,
    ):
        segments = segments.get(
            "segments",
            [],
        )

    if not isinstance(
        segments,
        list,
    ):
        raise ValueError(
            "Transcript JSON must contain "
            "a list of segments."
        )

    print(
        f"Loaded "
        f"{len(segments)} "
        f"transcript segments."
    )

    if refresh:
        print(
            "Refresh mode enabled."
        )

    clips = select_clips(
        segments,
        n=3,
    )

    os.makedirs(
        os.path.dirname(
            output_path
        ),
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            clips,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nSaved "
        f"{len(clips)} clips to "
        f"{output_path}"
    )

    for i, clip in enumerate(
        clips,
        start=1,
    ):

        duration = (
            clip["end"]
            - clip["start"]
        )

        print(
            f"\nClip {i}"
            f"\n  "
            f"{clip['start']:.2f}s → "
            f"{clip['end']:.2f}s "
            f"({duration:.2f}s)"
            f"\n  Title: "
            f"{clip['title']}"
            f"\n  Reason: "
            f"{clip['reason']}"
        )


if __name__ == "__main__":
    main()