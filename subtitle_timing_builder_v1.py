from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_START = 951.88
DEFAULT_END = 961.66

# Readability constraints for short-form captions.
MAX_WORDS = 6
MAX_CHARS = 34
MIN_CAPTION_DURATION = 0.35
MAX_CAPTION_DURATION = 2.50
MAX_SILENCE = 0.55

# Words that commonly end a natural phrase.
BREAK_WORDS = {
    "and",
    "but",
    "or",
    "so",
    "because",
    "that",
    "then",
    "when",
    "if",
    "although",
}

# Strong punctuation gets priority as a grouping boundary.
STRONG_PUNCT = (".", "?", "!", ":", ";")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_words(data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract word-level objects from common WhisperX JSON layouts.
    """
    candidates: list[dict[str, Any]] = []

    top_words = data.get("word_segments")
    if isinstance(top_words, list):
        candidates.extend(
            x for x in top_words if isinstance(x, dict)
        )

    if not candidates:
        for segment in data.get("segments", []):
            if not isinstance(segment, dict):
                continue
            words = segment.get("words")
            if isinstance(words, list):
                candidates.extend(
                    x for x in words if isinstance(x, dict)
                )

    normalized: list[dict[str, Any]] = []

    for idx, word in enumerate(candidates):
        start = word.get("start")
        end = word.get("end")
        text = word.get("word", word.get("text", ""))

        if start is None or end is None:
            continue

        try:
            start_f = float(start)
            end_f = float(end)
        except (TypeError, ValueError):
            continue

        text_s = normalize_text(str(text))

        if not text_s:
            continue

        if end_f <= start_f:
            continue

        normalized.append(
            {
                "index": idx,
                "start": start_f,
                "end": end_f,
                "text": text_s,
                "score": word.get("score"),
            }
        )

    normalized.sort(key=lambda x: (x["start"], x["end"]))
    return normalized


def clip_words(
    words: list[dict[str, Any]],
    start: float,
    end: float,
) -> list[dict[str, Any]]:
    clipped: list[dict[str, Any]] = []

    for word in words:
        if word["end"] <= start:
            continue
        if word["start"] >= end:
            continue

        # Keep original WhisperX timing but clip only the first/last word
        # boundaries when they cross the editorial source range.
        clipped_start = max(float(word["start"]), start)
        clipped_end = min(float(word["end"]), end)

        if clipped_end <= clipped_start:
            continue

        item = dict(word)
        item["start"] = clipped_start
        item["end"] = clipped_end
        clipped.append(item)

    return clipped


def punctuation_ends(text: str) -> bool:
    stripped = text.rstrip()
    return stripped.endswith(STRONG_PUNCT)


def build_caption_groups(
    words: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal current
        if current:
            groups.append(current)
            current = []

    for word in words:
        if not current:
            current.append(word)
            continue

        proposed = current + [word]

        proposed_text = " ".join(x["text"] for x in proposed)
        proposed_duration = proposed[-1]["end"] - proposed[0]["start"]
        silence = word["start"] - current[-1]["end"]

        previous_text = current[-1]["text"].rstrip()

        should_break = False

        # Hard readability limits.
        if len(proposed) > MAX_WORDS:
            should_break = True

        if len(proposed_text) > MAX_CHARS:
            should_break = True

        if proposed_duration > MAX_CAPTION_DURATION:
            should_break = True

        # Large natural speech gap.
        if silence > MAX_SILENCE:
            should_break = True

        # Strong punctuation should normally close a caption.
        if punctuation_ends(previous_text):
            should_break = True

        # Avoid captions ending with weak conjunctions when possible.
        if previous_text.lower().strip(" ,") in BREAK_WORDS:
            # Only split here when the next word keeps the proposed caption
            # reasonably large/readable.
            if len(current) >= 2:
                should_break = True

        if should_break:
            flush()
            current.append(word)
        else:
            current.append(word)

    flush()

    result: list[dict[str, Any]] = []

    for idx, group in enumerate(groups):
        text = " ".join(x["text"] for x in group).strip()
        start = group[0]["start"]
        end = group[-1]["end"]

        result.append(
            {
                "caption_index": idx,
                "start": start,
                "end": end,
                "duration": end - start,
                "text": text,
                "word_indices": [x["index"] for x in group],
                "words": [
                    {
                        "start": x["start"],
                        "end": x["end"],
                        "text": x["text"],
                        "score": x.get("score"),
                    }
                    for x in group
                ],
            }
        )

    return result


def merge_tiny_groups(
    groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(groups) < 2:
        return groups

    merged: list[dict[str, Any]] = []

    for group in groups:
        duration = float(group["duration"])

        if (
            duration < MIN_CAPTION_DURATION
            and merged
            and len(merged[-1]["words"]) + len(group["words"]) <= MAX_WORDS
        ):
            prev = merged[-1]

            combined_words = prev["words"] + group["words"]
            combined_text = " ".join(
                word["text"] for word in combined_words
            )

            prev["end"] = group["end"]
            prev["duration"] = prev["end"] - prev["start"]
            prev["text"] = combined_text
            prev["words"] = combined_words
            prev["word_indices"] += group["word_indices"]
        else:
            merged.append(group)

    for idx, group in enumerate(merged):
        group["caption_index"] = idx

    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--start", type=float, default=DEFAULT_START)
    parser.add_argument("--end", type=float, default=DEFAULT_END)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    transcript_path = Path(args.transcript)
    output_path = Path(args.output)

    if not transcript_path.exists():
        raise FileNotFoundError(transcript_path)

    if args.end <= args.start:
        raise ValueError("--end must be greater than --start")

    data = load_json(transcript_path)
    all_words = get_words(data)

    if not all_words:
        raise RuntimeError(
            "No word-level timestamps found. Expected "
            "'word_segments' or 'segments[*].words'."
        )

    selected_words = clip_words(
        all_words,
        args.start,
        args.end,
    )

    if not selected_words:
        raise RuntimeError(
            f"No aligned words overlap {args.start:.3f} -> {args.end:.3f}."
        )

    groups = build_caption_groups(selected_words)
    groups = merge_tiny_groups(groups)

    # Re-number and calculate useful diagnostics.
    for idx, group in enumerate(groups):
        group["caption_index"] = idx

    total_words = len(selected_words)
    covered_start = selected_words[0]["start"]
    covered_end = selected_words[-1]["end"]

    output_data = {
        "version": "1.0",
        "module": "subtitle_timing_builder",
        "timing_source": {
            "provider": "WhisperX",
            "source_file": str(transcript_path),
            "word_timestamps_preserved": True,
            "timestamps_retimed": False,
        },
        "source_range": [
            args.start,
            args.end,
        ],
        "selected_word_range": [
            covered_start,
            covered_end,
        ],
        "caption_policy": {
            "max_words": MAX_WORDS,
            "max_chars": MAX_CHARS,
            "min_caption_duration": MIN_CAPTION_DURATION,
            "max_caption_duration": MAX_CAPTION_DURATION,
            "max_silence": MAX_SILENCE,
        },
        "statistics": {
            "aligned_words_selected": total_words,
            "caption_count": len(groups),
            "duration": args.end - args.start,
            "first_word_start": covered_start,
            "last_word_end": covered_end,
        },
        "captions": groups,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=== SUBTITLE TIMING BUILDER v1 ===")
    print(f"source range: {args.start:.3f} -> {args.end:.3f}")
    print(f"words selected: {total_words}")
    print(f"caption groups: {len(groups)}")

    for group in groups:
        print(
            f'{group["start"]:.3f} -> {group["end"]:.3f} | '
            f'{group["text"]}'
        )

    print(f"output: {output_path}")


if __name__ == "__main__":
    main()