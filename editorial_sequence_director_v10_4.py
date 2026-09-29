from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


VERSION = "10.4"

MIN_CLIP = 2.0
MAX_CLIP = 60.0
MAX_LOCAL_CANDIDATE_UNITS = 20

# Event segmentation.
MAX_HARD_GAP = 3.0
SOFT_GAP = 1.6
MAX_EVENT_SECONDS = 60.0

WEIGHTS = {
    "hook": 1.30,
    "context": 1.00,
    "progression": 1.55,
    "continuity": 1.55,
    "payoff": 1.95,
    "closure": 1.25,
    "standalone": 1.25,
    "energy": 0.40,
    "visual": 0.25,
    "novelty": 0.25,
    "duration_fit": 0.03,
    "filler": -1.00,
    "repetition": -0.70,
    "context_tax": -1.70,
    "dead_air": -0.70,
}

GATES = {
    "hook": 0.30,
    "context": 0.38,
    "progression": 0.22,
    "continuity": 0.34,
    "payoff_or_closure": 0.31,
    "standalone": 0.45,
    "completeness": 0.50,
    "max_context_tax": 0.58,
    "max_filler": 0.46,
    "max_repetition": 0.80,
}

STOPWORDS = {
    "a","an","and","are","as","at","be","because","been","but","by","can",
    "could","did","do","does","for","from","get","go","got","had","has",
    "have","he","her","here","how","i","if","in","is","it","its","just",
    "like","me","my","no","not","of","on","or","our","so","that","the",
    "their","them","then","there","they","this","to","up","was","we",
    "were","what","when","where","who","why","will","with","would","you",
    "your","im","ive","id","ill","thats","dont","didnt","cant","couldnt",
    "wouldnt","wasnt","werent","isnt","arent","hes","shes","theyre",
}

FILLERS = {
    "uh","um","uhh","umm","okay","ok","yeah","yep","yes","no","well",
    "right","like","just","so","period","maybe","hmm","hey",
}

QUESTION_STARTS = {
    "what","why","how","where","when","who","which","whose",
    "is","are","am","was","were","do","does","did","can","could",
    "would","will","should","have","has","had",
}

CONTRAST_WORDS = {
    "but","however","though","although","instead","actually","yet",
    "except","rather","turns","apparently",
}

CONSEQUENCE_WORDS = {
    "so","then","therefore","thus","which","means","ended","finally",
    "until","after","before","because","that's",
}

EMOTION_WORDS = {
    "love","hate","scared","afraid","angry","mad","sad","happy","excited",
    "crazy","insane","amazing","terrible","awful","weird","wild",
    "ridiculous","shocked","surprised","funny","hilarious","brutal",
    "perfect","beautiful","horrible","worst","best","pain","fear",
    "awesome","disgusting","adorable","terrified",
}

INTENSIFIERS = {
    "really","very","so","too","literally","actually","absolutely",
    "completely","never","always","ever","nothing","everything",
}

BAD_START_WORDS = {
    "and","but","so","then","because","which","okay","ok","yeah","well",
    "right","maybe","period",
}

WEAK_OPEN_PATTERNS = (
    r"^(?:and|but|so|then|because|which|okay|ok|yeah|well|right)\b",
    r"^(?:you know|i mean|hold on|wait)\b",
)


@dataclass
class Unit:
    start: float
    end: float
    text: str
    index: int

    @property
    def words(self) -> List[str]:
        return re.findall(r"[A-Za-z0-9']+", self.text.lower())

    @property
    def word_count(self) -> int:
        return len(self.words)


@dataclass
class Event:
    event_index: int
    start_unit: int
    end_unit: int
    start: float
    end: float
    units: List[Unit]
    boundary_before: Optional[str]
    boundary_after: Optional[str]


@dataclass
class Candidate:
    start: float
    end: float
    unit_start: int
    unit_end: int
    event_index: int
    text: str
    event_type: str
    score: float
    accepted: bool
    rejection_reasons: List[str]
    metrics: Dict[str, Any]


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def mean(values: Sequence[float], default: float = 0.0) -> float:
    return statistics.fmean(values) if values else default


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", normalize(text).lower())


def content_tokens(text: str) -> set[str]:
    return {
        w for w in tokens(text)
        if len(w) > 2 and w not in STOPWORDS
    }


def overlap_ratio(a: str, b: str) -> float:
    aa = content_tokens(a)
    bb = content_tokens(b)
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / max(1, len(aa | bb))


def temporal_gaps(units: Sequence[Unit]) -> List[float]:
    return [
        max(0.0, b.start - a.end)
        for a, b in zip(units, units[1:])
    ]


def silence_between(a: Unit, b: Unit) -> float:
    return max(0.0, b.start - a.end)


def filler_ratio(text: str) -> float:
    words = tokens(text)
    if not words:
        return 0.0

    count = 0
    i = 0

    while i < len(words):
        if (
            i + 1 < len(words)
            and f"{words[i]} {words[i + 1]}"
            in {"you know", "i mean", "hold on"}
        ):
            count += 2
            i += 2
            continue

        if words[i] in FILLERS:
            count += 1

        i += 1

    return clamp(count / len(words))


def repetition_ratio(text: str) -> float:
    words = tokens(text)
    if len(words) < 2:
        return 0.0

    immediate = sum(
        1
        for a, b in zip(words, words[1:])
        if a == b
    )

    bigrams: Dict[Tuple[str, str], int] = {}

    for i in range(len(words) - 1):
        pair = (words[i], words[i + 1])
        bigrams[pair] = bigrams.get(pair, 0) + 1

    repeated_bigrams = sum(
        c - 1
        for c in bigrams.values()
        if c > 1
    )

    return clamp(
        (immediate + 1.5 * repeated_bigrams)
        / max(1.0, len(words) * 0.38)
    )


def speech_activity(units: Sequence[Unit]) -> float:
    if not units:
        return 0.0

    duration = units[-1].end - units[0].start
    if duration <= 0:
        return 0.0

    wps = sum(u.word_count for u in units) / duration
    return clamp(wps / 2.8)


# ---------------------------------------------------------------------------
# Rhetorical signals
# ---------------------------------------------------------------------------

def is_question(text: str) -> bool:
    clean = normalize(text)
    words = tokens(clean)

    if not words:
        return False

    if clean.endswith("?"):
        return True

    if words[0] in QUESTION_STARTS:
        return True

    low = clean.lower()

    patterns = (
        r"\bwhat\s+(?:are|is|do|did|does|would|could|can|was|were)\b",
        r"\bhow\s+(?:do|did|does|are|is|can|could|would)\b",
        r"\bwhy\s+(?:do|did|does|are|is|would|could|can)\b",
        r"\bwhere\s+(?:do|did|does|is|are|can|could|would)\b",
        r"\bwho\s+(?:is|are|did|does|do|would|could|can)\b",
        r"\b(?:is|are|was|were|did|does|do|can|could|would)\s+\w+\b",
    )

    return any(re.search(pattern, low) for pattern in patterns)


def question_strength(text: str) -> float:
    if normalize(text).endswith("?"):
        return 1.0
    return 0.80 if is_question(text) else 0.0


def contrast_strength(text: str) -> float:
    words = tokens(text)
    return clamp(
        sum(1 for w in words if w in CONTRAST_WORDS) / 2.0
    )


def consequence_strength(text: str) -> float:
    words = tokens(text)
    return clamp(
        sum(1 for w in words if w in CONSEQUENCE_WORDS) / 2.0
    )


def emotion_strength(text: str) -> float:
    words = tokens(text)
    if not words:
        return 0.0

    return clamp(
        sum(1 for w in words if w in EMOTION_WORDS)
        / max(1.0, len(words) / 9.0)
    )


def intensity_strength(text: str) -> float:
    words = tokens(text)
    if not words:
        return 0.0

    return clamp(
        sum(1 for w in words if w in INTENSIFIERS) / 3.0
    )


def first_person_strength(text: str) -> float:
    words = tokens(text)
    if not words:
        return 0.0

    return clamp(
        sum(1 for w in words if w in {"i", "we"}) / 3.0
    )


def beat_profile(unit: Unit) -> Dict[str, float]:
    words = unit.words

    q = question_strength(unit.text)
    contrast = contrast_strength(unit.text)
    consequence = consequence_strength(unit.text)
    emotion = emotion_strength(unit.text)
    intensity = intensity_strength(unit.text)

    weak = (
        len(words) <= 3
        or all(
            w in STOPWORDS or w in FILLERS
            for w in words
        )
        or words[0] in BAD_START_WORDS
    )

    declarative = 1.0 - q

    finality = (
        1.0
        if normalize(unit.text).endswith(("!", "."))
        else 0.45
    )

    landing_signal = clamp(
        0.36 * declarative
        + 0.21 * consequence
        + 0.19 * emotion
        + 0.12 * intensity
        + 0.12 * finality
    )

    return {
        "question": q,
        "contrast": contrast,
        "consequence": consequence,
        "emotion": emotion,
        "intensity": intensity,
        "first_person": first_person_strength(unit.text),
        "declarative": declarative,
        "weak": 1.0 if weak else 0.0,
        "landing_signal": landing_signal,
    }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_units(
    path: Path,
    video_end: float,
) -> List[Unit]:

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    if isinstance(data, dict):
        for key in (
            "segments",
            "utterances",
            "items",
            "results",
        ):
            if isinstance(data.get(key), list):
                data = data[key]
                break

    if not isinstance(data, list):
        raise ValueError(
            "Transcript must be a list or contain "
            "segments/utterances/items/results."
        )

    units: List[Unit] = []

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue

        start = item.get(
            "start",
            item.get(
                "start_time",
                item.get("startTime"),
            ),
        )

        end = item.get(
            "end",
            item.get(
                "end_time",
                item.get("endTime"),
            ),
        )

        text = item.get(
            "text",
            item.get(
                "content",
                item.get("word", ""),
            ),
        )

        if start is None or end is None or not text:
            continue

        try:
            start_f = float(start)
            end_f = float(end)
        except (TypeError, ValueError):
            continue

        if end_f <= start_f:
            continue

        if start_f >= video_end:
            continue

        units.append(
            Unit(
                start=start_f,
                end=min(end_f, video_end),
                text=normalize(text),
                index=i,
            )
        )

    return sorted(
        units,
        key=lambda x: (x.start, x.end),
    )


def load_shots(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in (
            "shots",
            "final_shots",
            "inventory",
        ):
            if isinstance(data.get(key), list):
                return data[key]

    return []


# ---------------------------------------------------------------------------
# Event segmentation
# ---------------------------------------------------------------------------

def classify_boundary(
    previous: Unit,
    current: Unit,
) -> Optional[str]:

    gap = silence_between(
        previous,
        current,
    )

    previous_profile = beat_profile(previous)
    current_profile = beat_profile(current)

    lexical = overlap_ratio(
        previous.text,
        current.text,
    )

    # Hard silence.
    if gap >= MAX_HARD_GAP:
        return "long_silence"

    # Strong pause after a completed beat.
    if (
        gap >= SOFT_GAP
        and previous_profile["landing_signal"] >= 0.57
        and lexical < 0.10
    ):
        return "completed_then_paused"

    # A direct answer/consequence/contrast after a question remains in event.
    if (
        previous_profile["question"] > 0
        and current_profile["declarative"] > 0.55
        and gap < SOFT_GAP
    ):
        return None

    if (
        current_profile["contrast"] > 0
        or current_profile["consequence"] > 0
    ) and gap < SOFT_GAP:
        return None

    # Very short acknowledgment/reaction normally belongs to current event.
    if (
        current_profile["weak"] > 0
        and len(current.words) <= 3
        and gap < SOFT_GAP
    ):
        return None

    # If both beats have real content, no relation, and the previous beat
    # already lands, begin a new event.
    if (
        len(content_tokens(previous.text)) >= 3
        and len(content_tokens(current.text)) >= 3
        and lexical < 0.06
        and previous_profile["landing_signal"] >= 0.60
    ):
        return "new_content"

    # New question after a completed thought.
    if (
        current_profile["question"] > 0
        and previous_profile["landing_signal"] >= 0.62
        and gap >= 0.6
    ):
        return "new_question"

    return None


def segment_events(
    units: Sequence[Unit],
) -> List[Event]:

    if not units:
        return []

    events: List[Event] = []

    start = 0
    boundary_before: Optional[str] = None
    event_index = 0

    for i in range(1, len(units)):
        reason = classify_boundary(
            units[i - 1],
            units[i],
        )

        current_duration = (
            units[i - 1].end
            - units[start].start
        )

        if (
            reason is not None
            or current_duration >= MAX_EVENT_SECONDS
        ):
            actual_reason = (
                reason
                if reason is not None
                else "event_duration_cap"
            )

            events.append(
                Event(
                    event_index=event_index,
                    start_unit=start,
                    end_unit=i - 1,
                    start=units[start].start,
                    end=units[i - 1].end,
                    units=list(units[start:i]),
                    boundary_before=boundary_before,
                    boundary_after=actual_reason,
                )
            )

            event_index += 1
            start = i
            boundary_before = actual_reason

    events.append(
        Event(
            event_index=event_index,
            start_unit=start,
            end_unit=len(units) - 1,
            start=units[start].start,
            end=units[-1].end,
            units=list(units[start:]),
            boundary_before=boundary_before,
            boundary_after=None,
        )
    )

    return events


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def valid_opening(
    units: Sequence[Unit],
) -> Tuple[bool, float, List[str]]:

    if not units:
        return False, 0.0, ["empty"]

    first = units[0]
    words = first.words
    profile = beat_profile(first)

    reasons = []

    if not words:
        return False, 0.0, ["empty_opening"]

    if words[0] in BAD_START_WORDS:
        # "and I..." can be valid only when the previous sentence is included,
        # so this window cannot start here.
        reasons.append("starts_as_continuation")

    if any(
        re.match(pattern, first.text.lower())
        for pattern in WEAK_OPEN_PATTERNS
    ):
        reasons.append("weak_sentence_start")

    if len(words) <= 3:
        reasons.append("too_short_opening")

    density = clamp(len(words) / 11.0)

    rhetorical = (
        0.58 * profile["question"]
        + 0.14 * profile["contrast"]
        + 0.14 * profile["emotion"]
        + 0.14 * profile["intensity"]
    )

    score = clamp(
        0.55 * rhetorical
        + 0.45 * density
        - 0.25 * profile["weak"]
    )

    return (
        not reasons,
        score,
        reasons,
    )


def event_candidate_ranges(
    event: Event,
) -> List[Tuple[int, int]]:

    units = event.units
    n = len(units)

    if not n:
        return []

    ranges: set[Tuple[int, int]] = set()

    def add(lo: int, hi: int) -> None:
        if lo < 0 or hi >= n or lo > hi:
            return

        duration = (
            units[hi].end
            - units[lo].start
        )

        if MIN_CLIP <= duration <= MAX_CLIP:
            ranges.add((lo, hi))

    # Start candidates must themselves be potentially valid openings.
    for i, unit in enumerate(units):
        _, opening_score, opening_reasons = valid_opening(
            units[i:i + 1]
        )

        profile = beat_profile(unit)

        structurally_interesting = (
            opening_score >= 0.25
            and not opening_reasons
        ) or (
            profile["question"] > 0
            or profile["emotion"] > 0.30
            or profile["contrast"] > 0
            or profile["intensity"] > 0.30
        )

        if not structurally_interesting:
            continue

        # Local candidate lengths.
        #
        # Short windows alone are insufficient for self-contained
        # answers that contain setup -> example -> explanation/payoff
        # across several transcript units.
        #
        # Generate progressively wider local windows. The time-based
        # MAX_CLIP guard still prevents candidates from becoming too long.
        for width in range(
            1,
            MAX_LOCAL_CANDIDATE_UNITS + 1,
        ):
            add(
                i,
                i + width,
            )

    # Always consider complete event only when its full span is reasonable.
    # It cannot cross into another event by construction.
    if (event.end - event.start) <= MAX_CLIP:
        add(0, n - 1)

    return sorted(ranges)


# ---------------------------------------------------------------------------
# Candidate scoring
# ---------------------------------------------------------------------------

def ending_continuation_risk(
    text: str,
) -> float:
    """
    Estimate whether the candidate ending sounds like it is
    grammatically or rhetorically waiting for more speech.

    This is intentionally conservative. It is not a full parser;
    it catches strong continuation signals that are especially
    dangerous for standalone Shorts.
    """

    t = normalize(text).strip().lower()

    if not t:
        return 1.0

    risk = 0.0

    # Strong grammatical continuation words.
    continuation_starters = (
        "and",
        "but",
        "because",
        "so",
        "if",
        "when",
        "while",
        "although",
        "though",
        "that",
        "which",
        "who",
        "where",
        "to",
        "than",
        "as",
        "for",
        "with",
        "without",
    )

    first_word = t.split()[0]

    if first_word in continuation_starters:
        risk += 0.35

    # Strong lexical endings that normally require something after them.
    continuation_endings = (
        "because",
        "although",
        "though",
        "if",
        "when",
        "while",
        "than",
        "that",
        "which",
        "who",
        "where",
        "to",
        "for",
        "with",
        "without",
        "like",
        "about",
        "of",
        "from",
        "into",
        "through",
        "between",
        "as",
    )

    words = t.split()

    if words and words[-1] in continuation_endings:
        risk += 0.65

    # Auxiliary / modal endings that strongly suggest continuation.
    auxiliary_endings = (
        "can",
        "could",
        "would",
        "should",
        "will",
        "may",
        "might",
        "must",
        "have",
        "has",
        "had",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
    )

    if words and words[-1] in auxiliary_endings:
        risk += 0.55

    # Explicitly incomplete punctuation.
    if t.endswith((",", ":", ";", "-", "—")):
        risk += 0.70

    # A trailing preposition is a classic continuation signal.
    if words and words[-1] in {
        "to",
        "of",
        "for",
        "with",
        "from",
        "in",
        "on",
        "at",
        "by",
        "into",
        "about",
        "between",
        "through",
    }:
        risk += 0.45

    return clamp(risk)


def score_candidate(
    window: Sequence[Unit],
    shots: Sequence[Dict[str, Any]],
) -> Tuple[
    float,
    Dict[str, Any],
    List[str],
]:

    text = normalize(
        " ".join(
            unit.text
            for unit in window
        )
    )

    profiles = [
        beat_profile(unit)
        for unit in window
    ]

    first = profiles[0]
    last = profiles[-1]

    # Opening must belong to this exact candidate.
    _, opening, opening_reasons = valid_opening(window)

    # Context.
    content = content_tokens(text)
    deictic_count = len(
        re.findall(
            r"\b(?:this|that|it|he|she|they|there|here|again)\b",
            text.lower(),
        )
    )

    context_presence = clamp(
        len(content) / 15.0
    )

    deictic_tax = clamp(
        (deictic_count - 1) / 5.0
    )

    context = clamp(
        0.80 * context_presence
        + 0.20 * (1.0 - deictic_tax)
    )

    # Adjacent relationship.
    adjacent_overlap = [
        overlap_ratio(a.text, b.text)
        for a, b in zip(
            window,
            window[1:],
        )
    ]

    lexical_continuity = mean(
        adjacent_overlap
    )

    rhetorical_links = []

    for a, b in zip(
        profiles,
        profiles[1:],
    ):
        link = 0.0

        if (
            a["question"] > 0
            and b["declarative"] > 0.55
        ):
            link = max(link, 1.0)

        if (
            b["contrast"] > 0
            or b["consequence"] > 0
        ):
            link = max(link, 0.88)

        if (
            a["emotion"] > 0.0
            and b["emotion"] > 0.20
        ):
            link = max(link, 0.66)

        if (
            a["intensity"] > 0.0
            and b["intensity"] > 0.20
        ):
            link = max(link, 0.58)

        rhetorical_links.append(link)

    rhetorical_continuity = mean(
        rhetorical_links
    )

    continuity = clamp(
        0.58 * clamp(
            lexical_continuity * 2.0
        )
        + 0.42 * rhetorical_continuity
    )

    # Progression.
    has_question = any(
        p["question"] > 0
        for p in profiles[:-1]
    )

    has_later_declaration = any(
        p["declarative"] > 0.55
        for p in profiles[1:]
    )

    has_contrast = any(
        p["contrast"] > 0
        for p in profiles[1:]
    )

    has_consequence = any(
        p["consequence"] > 0
        for p in profiles[1:]
    )

    transitions = 0

    for a, b in zip(
        profiles,
        profiles[1:],
    ):
        before = (
            a["question"] > 0,
            a["emotion"] > 0.30,
            a["contrast"] > 0,
            a["consequence"] > 0,
        )

        after = (
            b["question"] > 0,
            b["emotion"] > 0.30,
            b["contrast"] > 0,
            b["consequence"] > 0,
        )

        if before != after:
            transitions += 1

    question_to_declaration = (
        has_question
        and has_later_declaration
    )

    progression = clamp(
        0.25 * min(
            1.0,
            transitions / 2.0,
        )
        + 0.30 * (
            1.0
            if question_to_declaration
            else 0.0
        )
        + 0.20 * (
            1.0
            if has_contrast
            else 0.0
        )
        + 0.20 * (
            1.0
            if has_consequence
            else 0.0
        )
        + 0.05 * (
            1.0
            if len(window) >= 2
            else 0.0
        )
    )

    # Setup/payoff relationship.
    setup = normalize(
        " ".join(
            unit.text
            for unit in window[:-1]
        )
    )

    ending = window[-1].text

    # Ending completeness.
    #
    # A candidate should not receive a good landing score merely because
    # its final transcript unit contains several words. Some transcript
    # units are grammatically dependent on what follows.
    continuation_risk = ending_continuation_risk(
        ending
    )

    setup_end_overlap = overlap_ratio(
        setup,
        ending,
    ) if setup else 0.0

    midpoint = max(
        1,
        len(window) // 2,
    )

    early = normalize(
        " ".join(
            unit.text
            for unit in window[:midpoint]
        )
    )

    late = normalize(
        " ".join(
            unit.text
            for unit in window[midpoint:]
        )
    )

    global_relation = overlap_ratio(
        early,
        late,
    )

    # A payoff must have evidence connecting the ending to what came before.
    # Generic "yes/no/okay" does not count.
    ending_content = content_tokens(ending)
    setup_content = content_tokens(setup)

    meaningful_ending = (
        (
            len(ending_content) >= 2
            or last["emotion"] >= 0.30
            or last["contrast"] > 0
            or last["consequence"] > 0
            or last["intensity"] > 0.30
        )
        and continuation_risk < 0.55
    )

    answer_relationship = (
        1.0
        if (
            has_question
            and has_later_declaration
            and meaningful_ending
        )
        else 0.0
    )

    payoff = clamp(
        0.30 * answer_relationship
        + 0.24 * clamp(
            setup_end_overlap * 2.4
        )
        + 0.18 * clamp(
            global_relation * 2.0
        )
        + 0.12 * (
            1.0
            if has_consequence
            else 0.0
        )
        + 0.08 * (
            1.0
            if has_contrast
            else 0.0
        )
        + 0.05 * last["emotion"]
        + 0.03 * last["intensity"]
    )

    # A grammatically dependent ending is not a genuine payoff.
    #
    # Apply the penalty before candidate ranking so a candidate that
    # merely happens to contain many words cannot beat a complete answer.
    if continuation_risk >= 0.55:
        payoff *= max(
            0.05,
            1.0 - continuation_risk,
        )

    # Explicitly punish endings that introduce an unrelated lexical island.
    if (
        len(ending_content) >= 2
        and setup_content
        and len(ending_content & setup_content) == 0
        and not (
            has_consequence
            or has_contrast
            or answer_relationship
        )
    ):
        payoff *= 0.35

    # Closure.
    closure = clamp(
        0.45 * last["landing_signal"]
        + 0.22 * (
            1.0 - last["question"]
        )
        + 0.18 * last["emotion"]
        + 0.15 * last["consequence"]
    )

    # Closure must reflect actual linguistic landing.
    if continuation_risk >= 0.55:
        closure *= max(
            0.05,
            1.0 - 1.15 * continuation_risk,
        )

    # Standalone.
    standalone = clamp(
        0.50
        + 0.23 * context
        + 0.18 * continuity
        + 0.09 * progression
        - 0.28 * deictic_tax
        - 0.26 * first["weak"]
    )

    # Energy.
    energy = clamp(
        0.45 * mean(
            [p["emotion"] for p in profiles]
        )
        + 0.20 * mean(
            [p["intensity"] for p in profiles]
        )
        + 0.18 * mean(
            [p["question"] for p in profiles]
        )
        + 0.17 * min(
            1.0,
            transitions / 2.0,
        )
    )

    novelty = clamp(
        0.45 * min(
            1.0,
            transitions / 2.0,
        )
        + 0.30 * min(
            1.0,
            len(content) / 15.0,
        )
        + 0.25 * (
            1.0
            - repetition_ratio(text)
        )
    )

    # Timing.
    gaps = temporal_gaps(window)

    largest_gap = max(
        gaps,
        default=0.0,
    )

    average_gap = mean(
        gaps
    )

    dead_air = clamp(
        0.65 * (
            largest_gap - 0.70
        ) / 3.5
        + 0.35 * (
            average_gap - 0.40
        ) / 2.5
    )

    duration = (
        window[-1].end
        - window[0].start
    )

    # Tiny duration preference only.
    if 4.0 <= duration <= 18.0:
        duration_fit = 1.0
    elif duration < 4.0:
        duration_fit = clamp(
            duration / 4.0
        )
    else:
        duration_fit = clamp(
            1.0
            - (duration - 18.0) / 14.0
        )

    filler = filler_ratio(text)
    repetition = repetition_ratio(text)
    activity = speech_activity(window)

    # Context tax.
    context_tax = 0.0

    if first["weak"]:
        context_tax += 0.22

    if deictic_count >= 3 and context_presence < 0.50:
        context_tax += 0.28

    if opening < 0.30:
        context_tax += 0.14

    if (
        first["question"] > 0
        and continuity < 0.20
    ):
        context_tax += 0.18

    if (
        len(window) <= 2
        and len(content) < 4
    ):
        context_tax += 0.20

    context_tax = clamp(
        context_tax
    )

    completeness = clamp(
        0.16 * opening
        + 0.12 * context
        + 0.16 * progression
        + 0.18 * continuity
        + 0.22 * payoff
        + 0.16 * closure
    )

    # Event type based ONLY on this candidate.
    if (
        has_question
        and payoff >= 0.56
        and closure >= 0.45
    ):
        event_type = "question_to_payoff"
    elif (
        has_consequence
        and payoff >= 0.45
    ):
        event_type = "setup_to_consequence"
    elif (
        has_contrast
        and progression >= 0.45
        and closure >= 0.42
    ):
        event_type = "reversal"
    elif (
        energy >= 0.48
        and closure >= 0.50
    ):
        event_type = "emotional_landing"
    elif (
        progression >= 0.52
        and closure >= 0.45
    ):
        event_type = "escalation"
    elif (
        standalone >= 0.60
        and closure >= 0.55
    ):
        event_type = "standalone_quote"
    elif has_question:
        event_type = "open_question"
    else:
        event_type = "conversation_fragment"

    # Quality reasons.
    reasons = list(opening_reasons)

    if opening < GATES["hook"]:
        reasons.append("weak_opening")

    if context < GATES["context"]:
        reasons.append("insufficient_context")

    if progression < GATES["progression"]:
        reasons.append("weak_progression")

    if continuity < GATES["continuity"]:
        reasons.append("weak_event_continuity")

    if max(payoff, closure) < GATES["payoff_or_closure"]:
        reasons.append("weak_landing")

    if standalone < GATES["standalone"]:
        reasons.append("context_dependent")

    if completeness < GATES["completeness"]:
        reasons.append("incomplete_event")

    if context_tax > GATES["max_context_tax"]:
        reasons.append("high_context_dependency")

    if filler > GATES["max_filler"]:
        reasons.append("too_much_filler")

    if repetition > GATES["max_repetition"]:
        reasons.append("too_repetitive")

    if continuation_risk >= 0.55:
        reasons.append("unfinished_ending")

    if (
        last["question"] > 0
        and payoff < 0.42
        and closure < 0.48
    ):
        reasons.append("unresolved_question")

    if (
        activity < 0.16
        and duration > 6.0
    ):
        reasons.append("low_information_density")

    metrics = {
        "opening": clamp(opening),
        "context": clamp(context),
        "progression": clamp(progression),
        "continuity": clamp(continuity),
        "payoff": clamp(payoff),
        "closure": clamp(closure),
        "continuation_risk": clamp(continuation_risk),
        "standalone": clamp(standalone),
        "energy": clamp(energy),
        "novelty": clamp(novelty),
        "speech_activity": clamp(activity),
        "filler": clamp(filler),
        "repetition": clamp(repetition),
        "dead_air": clamp(dead_air),
        "duration_fit": clamp(duration_fit),
        "context_tax": clamp(context_tax),
        "event_completeness": clamp(completeness),
        "setup_end_overlap": clamp(setup_end_overlap * 2.4),
        "global_relationship": clamp(global_relation * 2.0),
        "meaningful_ending": meaningful_ending,
        "duration": duration,
        "largest_gap": largest_gap,
        "average_gap": average_gap,
        "content_token_count": len(content),
        "event_type": event_type,
        "visual": {},
    }

    return (
        score_from_metrics(
            metrics,
            visual_score=0.0,
        ),
        metrics,
        sorted(set(reasons)),
    )


def score_from_metrics(
    m: Dict[str, Any],
    visual_score: float,
) -> float:

    return (
        WEIGHTS["hook"] * m["opening"]
        + WEIGHTS["context"] * m["context"]
        + WEIGHTS["progression"] * m["progression"]
        + WEIGHTS["continuity"] * m["continuity"]
        + WEIGHTS["payoff"] * m["payoff"]
        + WEIGHTS["closure"] * m["closure"]
        + WEIGHTS["standalone"] * m["standalone"]
        + WEIGHTS["energy"] * m["energy"]
        + WEIGHTS["visual"] * visual_score
        + WEIGHTS["novelty"] * m["novelty"]
        + WEIGHTS["duration_fit"] * m["duration_fit"]
        + WEIGHTS["filler"] * m["filler"]
        + WEIGHTS["repetition"] * m["repetition"]
        + WEIGHTS["context_tax"] * m["context_tax"]
        + WEIGHTS["dead_air"] * m["dead_air"]
    )


# ---------------------------------------------------------------------------
# Visual features
# ---------------------------------------------------------------------------

def visual_features(
    start: float,
    end: float,
    shots: Sequence[Dict[str, Any]],
) -> Dict[str, float]:

    overlapping = []

    for shot in shots:
        if (
            shot.get("start") is None
            or shot.get("end") is None
        ):
            continue

        s = float(shot["start"])
        e = float(shot["end"])

        amount = max(
            0.0,
            min(end, e)
            - max(start, s),
        )

        if amount > 0:
            overlapping.append(
                (amount, shot)
            )

    if not overlapping:
        return {
            "shot_change": 0.0,
            "face": 0.0,
            "visual_variety": 0.0,
            "insert_opportunity": 0.0,
            "shot_count": 0.0,
        }

    total = sum(
        x[0] for x in overlapping
    )

    faces_score = 0.0
    types = set()
    insert = 0.0

    for amount, shot in overlapping:
        fraction = amount / max(
            total,
            0.001,
        )

        faces = shot.get(
            "faces",
            shot.get(
                "face_detection",
                shot.get(
                    "face_count",
                    0,
                ),
            ),
        )

        if isinstance(
            faces,
            (int, float),
        ):
            fs = clamp(
                float(faces) / 2.0
            )
        elif isinstance(
            faces,
            list,
        ):
            fs = clamp(
                len(faces) / 2.0
            )
        elif isinstance(
            faces,
            dict,
        ):
            detections = faces.get(
                "detections",
                faces.get(
                    "faces",
                    [],
                ),
            )

            fs = (
                clamp(
                    len(detections) / 2.0
                )
                if isinstance(
                    detections,
                    list,
                )
                else 0.0
            )
        else:
            fs = 0.0

        faces_score += (
            fs * fraction
        )

        st = shot.get(
            "shot_type",
            shot.get(
                "type",
                shot.get(
                    "classification"
                ),
            ),
        )

        if st:
            types.add(
                str(st)
            )

        ci = shot.get(
            "cut_in_score",
            shot.get(
                "cutInScore"
            ),
        )

        if isinstance(
            ci,
            (int, float),
        ):
            insert = max(
                insert,
                float(ci),
            )

    return {
        "shot_change": clamp(
            (len(overlapping) - 1)
            / 3.0
        ),
        "face": clamp(
            faces_score
        ),
        "visual_variety": clamp(
            (len(types) - 1)
            / 2.0
        ),
        "insert_opportunity": clamp(
            insert
        ),
        "shot_count": float(
            len(overlapping)
        ),
    }


# ---------------------------------------------------------------------------
# Structural ending quality / endpoint repair
# ---------------------------------------------------------------------------

_DANGLING_ENDING_WORDS = {
    "a", "an", "the", "of", "to", "for", "from", "with", "by", "at", "in",
    "on", "into", "onto", "through", "about", "because", "if", "when", "while",
    "and", "or", "but", "so", "as", "than", "that", "which", "who", "where",
    "what", "why", "how", "then", "until",
}

_DANGLING_VERB_PRECEDERS = {
    "took", "takes", "take", "waited", "wait", "spent", "lasted", "last",
    "needed", "needs", "used", "uses", "gave", "give", "made", "makes",
    "went", "go", "started", "start", "kept", "keep",
}


def structurally_incomplete_ending(text: str) -> Tuple[bool, str]:
    """
    Detect endpoint truncation using grammar/structure rather than a fixed
    payoff keyword list.

    This deliberately catches numeric truncation such as:
        "Jack Nicholson took 30"
    because a trailing number after a duration-taking verb is usually an
    unfinished quantity. It also catches ordinary dangling conjunctions,
    prepositions, and other open grammatical forms.
    """
    clean = normalize(text)
    words = tokens(clean)

    if not words:
        return True, "empty_endpoint"

    # A question is not a completed answer endpoint.
    if clean.endswith("?") or is_question(clean):
        return True, "question_endpoint"

    # Explicit dangling grammatical words.
    last = words[-1]
    if last in _DANGLING_ENDING_WORDS:
        return True, "dangling_function_word"

    # A lone number / number with punctuation is incomplete.
    if re.fullmatch(r"\d+(?:\.\d+)?", last):
        if len(words) == 1:
            return True, "numeric_endpoint"
        if len(words) >= 2 and words[-2] in _DANGLING_VERB_PRECEDERS:
            return True, "numeric_after_duration_verb"

    # Numeric amount/duration phrases with no unit are commonly incomplete.
    if re.search(
        r"\b(?:took|waited|spent|lasted|needed|for)\s+\d+(?:\.\d+)?$",
        clean.lower(),
    ):
        return True, "unfinished_numeric_phrase"

    # Hyphen / dash endpoint usually indicates the thought is still opening.
    if clean.endswith(("-", "—", "–", "/")):
        return True, "open_punctuation"

    # Speech-like continuation markers.
    if re.search(
        r"\b(?:because|although|if|when|while|that|which|who|where|what|why|how)$",
        clean.lower(),
    ):
        return True, "unfinished_subordinate_clause"

    return False, ""


def _candidate_with_window(
    original: Candidate,
    window: Sequence[Unit],
    shots: Sequence[Dict[str, Any]],
    ending_repair: Dict[str, Any],
) -> Candidate:
    """Re-score a repaired candidate without changing its opening or event."""
    base_score, metrics, reasons = score_candidate(window, shots)

    visual = visual_features(
        window[0].start,
        window[-1].end,
        shots,
    )
    metrics["visual"] = visual

    visual_score = (
        0.35 * visual["shot_change"]
        + 0.30 * visual["face"]
        + 0.20 * visual["visual_variety"]
        + 0.15 * visual["insert_opportunity"]
    )

    final_score = score_from_metrics(metrics, visual_score)
    metrics["ending_repair"] = ending_repair

    # The repaired candidate is represented by the exact transcript window.
    return Candidate(
        start=window[0].start,
        end=window[-1].end,
        unit_start=original.unit_start,
        unit_end=original.unit_end + (len(window) - 1),
        event_index=original.event_index,
        text=normalize(" ".join(u.text for u in window)),
        event_type=metrics["event_type"],
        score=final_score,
        accepted=not reasons,
        rejection_reasons=sorted(set(reasons)),
        metrics=metrics,
    )


def repair_selected_ending(
    selected: Optional[Candidate],
    units: Sequence[Unit],
    events: Sequence[Event],
    shots: Sequence[Dict[str, Any]],
    min_score: float,
) -> Optional[Candidate]:
    """
    Extend an otherwise strong selected candidate when its final transcript
    unit is structurally incomplete.

    Repair is semantic/structural:
      * preserves the original opening;
      * only advances chronologically;
      * never crosses an event boundary;
      * searches for the earliest natural landing that restores completeness;
      * permits longer clips because duration itself is not the objective.
    """
    if selected is None:
        return None

    incomplete, reason = structurally_incomplete_ending(selected.text)
    if not incomplete:
        return selected

    event = next(
        (e for e in events if e.event_index == selected.event_index),
        None,
    )
    if event is None:
        return selected

    start_pos = selected.unit_start
    current_end_pos = selected.unit_end

    if current_end_pos >= event.end_unit:
        return selected

    # Keep the original selected score as a baseline. A repaired landing must
    # still satisfy the normal editorial gates and should not become a much
    # weaker clip simply because it is longer.
    baseline_score = selected.score
    max_extra_units = min(20, event.end_unit - current_end_pos)

    best = None

    for extra in range(1, max_extra_units + 1):
        end_pos = current_end_pos + extra
        window = list(units[start_pos:end_pos + 1])

        duration = window[-1].end - window[0].start
        if duration > MAX_CLIP:
            break

        endpoint_incomplete, endpoint_reason = structurally_incomplete_ending(
            " ".join(u.text for u in window)
        )
        if endpoint_incomplete:
            continue

        candidate = _candidate_with_window(
            selected,
            window,
            shots,
            {
                "applied": True,
                "original_start": round(selected.start, 3),
                "original_end": round(selected.end, 3),
                "new_end": round(window[-1].end, 3),
                "reason": reason,
                "endpoint_reason": endpoint_reason,
                "extra_units": extra,
            },
        )

        # Normal editorial quality still governs the repaired result.
        if not candidate.accepted:
            continue
        if candidate.score < min_score:
            continue
        if candidate.metrics["event_completeness"] < 0.54:
            continue
        if (
            candidate.metrics["payoff"] < 0.34
            and candidate.metrics["closure"] < 0.56
        ):
            continue
        if (
            candidate.metrics["continuity"] < 0.35
            and candidate.metrics["standalone"] < 0.68
        ):
            continue
        if candidate.metrics["opening"] < 0.30:
            continue

        # Prefer the earliest complete landing, but reject a repair that
        # materially degrades the original editorial quality.
        score_drop = baseline_score - candidate.score
        if score_drop > 0.85:
            continue

        best = candidate
        break

    if best is None:
        # No structurally complete landing passed the ordinary quality gates.
        # Do not silently manufacture an ending.
        return selected

    return best


# ---------------------------------------------------------------------------
# Candidate building
# ---------------------------------------------------------------------------

def _make_candidate_from_window(
    window: Sequence[Unit],
    unit_start: int,
    unit_end: int,
    event_index: int,
    shots: Sequence[Dict[str, Any]],
) -> Candidate:
    """Score one global transcript window using the same v10 editorial logic."""
    base_score, metrics, reasons = score_candidate(
        window,
        shots,
    )

    visual = visual_features(
        window[0].start,
        window[-1].end,
        shots,
    )
    metrics["visual"] = visual

    visual_score = (
        0.35 * visual["shot_change"]
        + 0.30 * visual["face"]
        + 0.20 * visual["visual_variety"]
        + 0.15 * visual["insert_opportunity"]
    )

    final_score = score_from_metrics(
        metrics,
        visual_score,
    )

    return Candidate(
        start=window[0].start,
        end=window[-1].end,
        unit_start=unit_start,
        unit_end=unit_end,
        event_index=event_index,
        text=normalize(
            " ".join(
                u.text
                for u in window
            )
        ),
        event_type=metrics["event_type"],
        score=final_score,
        accepted=not reasons,
        rejection_reasons=sorted(set(reasons)),
        metrics=metrics,
    )


def _can_chain_boundary(event: Event, next_event: Event) -> bool:
    """
    Cross ordinary discourse segmentation boundaries when the next event is
    still part of the same conversational answer.

    Hard boundaries stop chaining:
      * long_silence
      * completed_then_paused
      * new_question
      * event_duration_cap

    A `new_content` boundary is therefore chainable, but so are other soft
    boundaries when there is no evidence that the speaker has started a new
    question or deliberately closed the thought.
    """
    hard = {
        "long_silence",
        "completed_then_paused",
        "new_question",
        "event_duration_cap",
    }

    if event.boundary_after in hard:
        return False

    # Never chain directly into a new question.
    if next_event.boundary_before == "new_question":
        return False

    # A question at the start of the next event should remain a new editorial
    # opening rather than being swallowed by the previous answer.
    next_first = next_event.units[0] if next_event.units else None
    if next_first is not None and is_question(next_first.text):
        return False

    return True


def build_chained_candidates(
    events: Sequence[Event],
    units: Sequence[Unit],
    shots: Sequence[Dict[str, Any]],
) -> List[Candidate]:
    """
    Generate context-complete candidates that span adjacent soft event
    boundaries when the conversation clearly continues.

    This fixes cases such as:
        question -> setup -> example -> consequence
    where the event segmenter correctly detects local beat changes but an
    editor would still regard the whole span as one answer.

    We deliberately do NOT cross:
      - new_question
      - long_silence
      - completed_then_paused
      - event_duration_cap
    """
    chained: List[Candidate] = []

    for start_idx, start_event in enumerate(events):
        # The opening must be legitimate at the beginning of the first event.
        first_unit = units[start_event.start_unit]
        valid, _, _ = valid_opening([first_unit])
        if not valid:
            continue

        end_idx = start_idx

        while end_idx < len(events):
            current = events[end_idx]

            window_start = start_event.start_unit
            window_end = current.end_unit
            window = list(units[window_start:window_end + 1])

            if not window:
                break

            duration = window[-1].end - window[0].start
            if duration > MAX_CLIP:
                break

            # Only emit a chained candidate when more than one event is used.
            if end_idx > start_idx:
                candidate = _make_candidate_from_window(
                    window,
                    window_start,
                    window_end,
                    start_event.event_index,
                    shots,
                )
                candidate.metrics["semantic_event_chain"] = True
                candidate.metrics["chained_event_start"] = start_event.event_index
                candidate.metrics["chained_event_end"] = current.event_index
                candidate.metrics["crossed_boundaries"] = end_idx - start_idx

                # Chaining is valuable only when the combined span is
                # substantively better than a disconnected local fragment.
                if (
                    candidate.accepted
                    and candidate.metrics["event_completeness"] >= 0.54
                ):
                    chained.append(candidate)

            if end_idx >= len(events) - 1:
                break

            # The boundary belongs to the current event and describes the
            # relation to the next one. Cross only "new_content".
            if not _can_chain_boundary(current, events[end_idx + 1]):
                break

            end_idx += 1

    return chained


def build_candidates(
    events: Sequence[Event],
    shots: Sequence[Dict[str, Any]],
) -> List[Candidate]:

    candidates: List[Candidate] = []

    # Existing event-local candidates remain unchanged.
    for event in events:
        for lo, hi in event_candidate_ranges(event):
            window = event.units[
                lo:hi + 1
            ]

            base_score, metrics, reasons = score_candidate(
                window,
                shots,
            )

            visual = visual_features(
                window[0].start,
                window[-1].end,
                shots,
            )

            metrics["visual"] = visual

            visual_score = (
                0.35 * visual["shot_change"]
                + 0.30 * visual["face"]
                + 0.20 * visual["visual_variety"]
                + 0.15 * visual["insert_opportunity"]
            )

            final_score = score_from_metrics(
                metrics,
                visual_score,
            )

            candidates.append(
                Candidate(
                    start=window[0].start,
                    end=window[-1].end,
                    unit_start=(
                        event.start_unit
                        + lo
                    ),
                    unit_end=(
                        event.start_unit
                        + hi
                    ),
                    event_index=event.event_index,
                    text=normalize(
                        " ".join(
                            u.text
                            for u in window
                        )
                    ),
                    event_type=metrics["event_type"],
                    score=final_score,
                    accepted=not reasons,
                    rejection_reasons=reasons,
                    metrics=metrics,
                )
            )

    # NEW: context-complete chains across soft "new_content" boundaries.
    # This is the generalized fix for answers split into setup/example/payoff
    # events by local discourse segmentation.
    #
    # We need the global unit list here. Event units are slices of that same
    # sorted transcript, so rebuilding it from events preserves global order.
    global_units: List[Unit] = []
    seen_indices = set()

    for event in events:
        for unit in event.units:
            if unit.index not in seen_indices:
                global_units.append(unit)
                seen_indices.add(unit.index)

    global_units.sort(
        key=lambda u: (u.start, u.end, u.index)
    )

    candidates.extend(
        build_chained_candidates(
            events,
            global_units,
            shots,
        )
    )

    # Exact interval dedupe.
    unique: Dict[
        Tuple[float, float],
        Candidate,
    ] = {}

    for candidate in candidates:
        key = (
            round(candidate.start, 3),
            round(candidate.end, 3),
        )

        old = unique.get(key)

        if (
            old is None
            or candidate.score > old.score
        ):
            unique[key] = candidate

    return sorted(
        unique.values(),
        key=lambda c: (
            c.accepted,
            c.score,
            c.metrics[
                "event_completeness"
            ],
            c.metrics["payoff"],
            c.metrics["closure"],
        ),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Final selector
# ---------------------------------------------------------------------------

def select_best(
    candidates: Sequence[Candidate],
    min_score: float,
) -> Optional[Candidate]:

    accepted = [
        c
        for c in candidates
        if c.accepted
    ]

    if not accepted:
        return None

    robust = []

    for candidate in accepted:
        m = candidate.metrics

        critical = sorted(
            [
                m["opening"],
                m["progression"],
                m["continuity"],
                m["payoff"],
                m["closure"],
                m["standalone"],
            ],
            reverse=True,
        )

        # One-dimensional winner protection.
        if (
            critical[0] > 0.90
            and sum(critical[1:4]) < 1.00
        ):
            continue

        # Long candidates need genuinely high completeness.
        if (
            m["duration"] > 18.0
            and m["event_completeness"] < 0.66
        ):
            continue

        # A candidate with low actual setup/payoff relationship should not
        # win merely because it has good generic landing/energy.
        if (
            m["payoff"] < 0.30
            and m["standalone"] < 0.70
        ):
            continue

        # Never let a clearly unfinished transcript unit win over a
        # context-complete candidate.
        if m.get("continuation_risk", 0.0) >= 0.75:
            continue

        robust.append(candidate)

    if not robust:
        robust = accepted

    ranked = sorted(
        robust,
        key=lambda c: (
            c.score,
            c.metrics[
                "event_completeness"
            ],
            c.metrics["payoff"]
            + c.metrics["closure"],
            c.metrics["continuity"]
            + c.metrics["progression"],
            c.metrics["opening"]
            + c.metrics["standalone"],
        ),
        reverse=True,
    )

    winner = ranked[0]
    m = winner.metrics

    # Final quality floor.
    if winner.score < min_score:
        return None

    if m["event_completeness"] < 0.54:
        return None

    if (
        m["payoff"] < 0.34
        and m["closure"] < 0.56
    ):
        return None

    if (
        m["continuity"] < 0.35
        and m["standalone"] < 0.68
    ):
        return None

    # Very important: opening must itself be legitimate.
    if m["opening"] < 0.30:
        return None

    return winner


# ---------------------------------------------------------------------------
# Render-safe actions
# ---------------------------------------------------------------------------

def build_actions(
    selected: Candidate,
    shots: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    actions = []

    for shot_index, shot in enumerate(
        shots
    ):
        if (
            shot.get("start") is None
            or shot.get("end") is None
        ):
            continue

        shot_start = float(
            shot["start"]
        )
        shot_end = float(
            shot["end"]
        )

        start = max(
            selected.start,
            shot_start,
        )
        end = min(
            selected.end,
            shot_end,
        )

        if end - start <= 0.001:
            continue

        actions.append(
            {
                "action_index": len(
                    actions
                ),
                "action_type": "base",
                "editorial_role": "event",
                "source_start": round(
                    start,
                    3,
                ),
                "source_end": round(
                    end,
                    3,
                ),
                "shot_index": shot_index,
                "shot_id": shot.get(
                    "shot_id",
                    shot.get("id"),
                ),
                "reason_types": [
                    "editorial_event",
                    selected.event_type,
                ],
            }
        )

    return actions


def validate_actions(
    actions: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:

    errors = []
    previous = -float("inf")
    ranges = []

    for action in actions:
        start = float(
            action["source_start"]
        )
        end = float(
            action["source_end"]
        )

        if end <= start:
            errors.append(
                "invalid_interval"
            )

        if start < previous - 0.001:
            errors.append(
                "backward_source_jump"
            )

        previous = start

        for old_start, old_end in ranges:
            if (
                min(end, old_end)
                - max(start, old_start)
                > 0.002
            ):
                errors.append(
                    "source_reuse"
                )

        ranges.append(
            (start, end)
        )

    return {
        "valid": not errors,
        "errors": sorted(
            set(errors)
        ),
        "video_reuse": sum(
            1
            for e in errors
            if e == "source_reuse"
        ),
        "backward_jumps": sum(
            1
            for e in errors
            if e == "backward_source_jump"
        ),
    }


# ---------------------------------------------------------------------------
# Output / CLI
# ---------------------------------------------------------------------------

def write_output(
    path: Path,
    transcript_path: Path,
    shots_path: Path,
    video_end: float,
    units: Sequence[Unit],
    events: Sequence[Event],
    candidates: Sequence[Candidate],
    selected: Optional[Candidate],
    actions: Sequence[Dict[str, Any]],
    min_score: float,
) -> None:

    validation = validate_actions(
        actions
    )

    data = {
        "director": {
            "name": "Editorial Sequence Director",
            "version": VERSION,
            "philosophy": [
                "one clip equals one conversational event",
                "candidate cannot cross hard event boundaries",
                "candidate opening must be valid by itself",
                "payoff must relate to candidate setup",
                "event type comes only from candidate-local evidence",
                "duration is not an objective",
                "zero-output is valid",
                "AV legality belongs to downstream resolver",
            ],
        },
        "source": {
            "transcript": str(
                transcript_path
            ),
            "shots": str(shots_path),
            "video_end": video_end,
            "transcript_units": len(
                units
            ),
            "events": len(events),
        },
        "policy": {
            "min_clip_seconds": MIN_CLIP,
            "max_clip_seconds": MAX_CLIP,
            "max_event_seconds": MAX_EVENT_SECONDS,
            "max_hard_gap_seconds": MAX_HARD_GAP,
            "soft_gap_seconds": SOFT_GAP,
            "duration_is_objective": False,
            "quality_gated": True,
            "structural_ending_repair": True,
            "semantic_event_chaining": True,
            "chain_boundary_allowed": "soft_boundaries_except_hard",
            "max_repair_extra_units": 20,
            "max_local_candidate_units": MAX_LOCAL_CANDIDATE_UNITS,
            "min_score": min_score,
        },
        "events": [
            {
                "event_index": e.event_index,
                "start_unit": e.start_unit,
                "end_unit": e.end_unit,
                "start": e.start,
                "end": e.end,
                "duration": e.end - e.start,
                "boundary_before": e.boundary_before,
                "boundary_after": e.boundary_after,
                "text": normalize(
                    " ".join(
                        u.text
                        for u in e.units
                    )
                ),
            }
            for e in events
        ],
        "candidate_count": len(
            candidates
        ),
        "accepted_candidate_count": sum(
            1
            for c in candidates
            if c.accepted
        ),
        "rejected_candidate_count": sum(
            1
            for c in candidates
            if not c.accepted
        ),
        "selected_moment": (
            asdict(selected)
            if selected
            else None
        ),
        "timeline_actions": list(
            actions
        ),
        "validation": validation,
        "render_ready_candidate": bool(
            selected
            and actions
            and validation["valid"]
        ),
        "top_candidates": [
            asdict(c)
            for c in candidates[:40]
        ],
    }

    path.write_text(
        json.dumps(
            data,
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Editorial Sequence Director v10.4"
        )
    )

    parser.add_argument(
        "--transcript",
        required=True,
    )

    parser.add_argument(
        "--shots",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--video-end",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--min-score",
        type=float,
        default=4.8,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    transcript_path = Path(
        args.transcript
    )

    shots_path = Path(
        args.shots
    )

    output_path = Path(
        args.output
    )

    units = load_units(
        transcript_path,
        args.video_end,
    )

    shots = load_shots(
        shots_path
    )

    events = segment_events(
        units
    )

    candidates = build_candidates(
        events,
        shots,
    )

    selected = select_best(
        candidates,
        min_score=args.min_score,
    )

    selected = repair_selected_ending(
        selected,
        units,
        events,
        shots,
        min_score=args.min_score,
    )

    actions = (
        build_actions(
            selected,
            shots,
        )
        if selected
        else []
    )

    write_output(
        output_path,
        transcript_path,
        shots_path,
        args.video_end,
        units,
        events,
        candidates,
        selected,
        actions,
        args.min_score,
    )

    summary = {
        "output": str(
            output_path
        ),
        "version": VERSION,
        "transcript_units": len(
            units
        ),
        "events": len(events),
        "candidate_moments": len(
            candidates
        ),
        "accepted_candidates": sum(
            1
            for c in candidates
            if c.accepted
        ),
        "selected_score": (
            round(
                selected.score,
                6,
            )
            if selected
            else None
        ),
        "selected_start": (
            round(
                selected.start,
                3,
            )
            if selected
            else None
        ),
        "selected_end": (
            round(
                selected.end,
                3,
            )
            if selected
            else None
        ),
        "selected_duration": (
            round(
                selected.end
                - selected.start,
                3,
            )
            if selected
            else None
        ),
        "event_type": (
            selected.event_type
            if selected
            else None
        ),
        "timeline_actions": len(
            actions
        ),
        "render_ready_candidate": bool(
            selected and actions
        ),
    }

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    if selected:
        print(
            "\nSELECTED TEXT\n"
            + selected.text
        )

        print(
            "\nSELECTED METRICS"
        )

        for key in (
            "opening",
            "context",
            "progression",
            "continuity",
            "payoff",
            "closure",
            "standalone",
            "event_completeness",
            "context_tax",
        ):
            print(
                f"{key}: "
                f"{selected.metrics[key]:.3f}"
            )

    else:
        print(
            "\nNO CLIP PASSED THE "
            "EDITORIAL QUALITY GATE."
        )


if __name__ == "__main__":
    main()