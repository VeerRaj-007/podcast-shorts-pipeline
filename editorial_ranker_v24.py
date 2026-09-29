#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def clean_token(token):
    return re.sub(r"\s+", " ", str(token).strip())


def words_in_time_range(words, start, end):
    return [
        w for w in words
        if float(w.get("end", 0)) > start
        and float(w.get("start", 0)) < end
    ]


def text_from_ordered_words(words):
    text = ""

    for w in words:
        token = clean_token(w.get("word", ""))

        if not token:
            continue

        if not text:
            text = token
        elif re.match(r"^[,.;:!?)]", token):
            text += token
        else:
            text += " " + token

    text = re.sub(r"\s+", " ", text).strip()

    text = text.replace("I lovethe", "I love the")
    text = text.replace("Andthat's", "And that's")
    text = text.replace("Andthe", "And the")

    return text


def find_candidate(data, approx_start):
    candidates = data.get("selected", [])

    if not isinstance(candidates, list):
        raise ValueError("Input does not contain selected candidates.")

    for c in candidates:
        start = float(c.get("start", -999))

        if abs(start - approx_start) < 1.0:
            return c

    raise ValueError(
        f"Could not find candidate near {approx_start}"
    )


def normalize_word(token):
    token = str(token).strip().lower()
    return re.sub(r"[^\w']+", "", token)


def find_word_sequence(
    words,
    phrase_tokens,
    min_time,
    max_time,
):
    """
    Find BOTH the start and end index of an exact ordered word sequence.

    The previous implementation returned only the end index and could
    therefore produce a slice that started in the middle of the phrase.

    We also choose the occurrence whose first word is closest to min_time.
    """

    target = [
        normalize_word(token)
        for token in phrase_tokens
    ]

    target = [x for x in target if x]

    matches = []

    for i in range(len(words) - len(target) + 1):

        first = words[i]

        first_start = float(first.get("start", 0))

        if first_start < min_time:
            continue

        if first_start > max_time:
            break

        sequence = []

        valid = True

        for j, target_token in enumerate(target):
            current = words[i + j]

            current_token = normalize_word(
                current.get("word", "")
            )

            if current_token != target_token:
                valid = False
                break

            sequence.append(current_token)

        if valid:
            matches.append(
                (
                    abs(first_start - min_time),
                    i,
                    i + len(target) - 1,
                )
            )

    if not matches:
        return None

    matches.sort(key=lambda x: x[0])

    _, start_index, end_index = matches[0]

    return start_index, end_index

def build_editorial_text(
    words,
    start_index,
    end_index,
):
    return text_from_ordered_words(
        words[start_index:end_index + 1]
    )
def make_result(candidate, start, end, text, reason):
    result = dict(candidate)

    result["start"] = round(start, 3)
    result["end"] = round(end, 3)
    result["duration"] = round(end - start, 3)

    result["editorial_text_v24"] = text

    result["boundary_resolution"] = {
        "version": "v24-final",
        "reason": reason,
    }

    return result


def resolve_performance(words, candidate):

    start = 951.940
    end = 964.760

    start_match = find_word_sequence(
        words,
        ["The", "lie", "is", "that"],
        951.5,
        952.5,
    )

    end_match = find_word_sequence(
        words,
        [
            "And",
            "you",
            "fool",
            "people",
            "into",
            "making",
            "them",
            "think",
            "that's",
            "real",
        ],
        962.0,
        965.0,
    )

    if start_match is None:
        raise RuntimeError(
            "Could not locate performance START phrase."
        )

    if end_match is None:
        raise RuntimeError(
            "Could not locate performance END phrase."
        )

    start_index = start_match[0]
    end_index = end_match[1]

    text = build_editorial_text(
        words,
        start_index,
        end_index,
    )

    return make_result(
        candidate,
        start,
        end,
        text,
        "complete performance thought including consequence",
    )

def resolve_direction(words, candidate):

    start = 840.500
    end = 880.680

    start_match = find_word_sequence(
        words,
        [
            "What's",
            "the",
            "best",
            "piece",
            "of",
            "direction",
            "that",
            "you've",
            "received",
        ],
        840.0,
        842.0,
    )

    end_match = find_word_sequence(
        words,
        [
            "that",
            "has",
            "to",
            "do",
            "with",
            "subtext",
        ],
        876.0,
        881.0,
    )

    if start_match is None:
        raise RuntimeError(
            "Could not locate direction START phrase."
        )

    if end_match is None:
        raise RuntimeError(
            "Could not locate direction END phrase."
        )

    start_index = start_match[0]
    end_index = end_match[1]

    text = build_editorial_text(
        words,
        start_index,
        end_index,
    )

    return make_result(
        candidate,
        start,
        end,
        text,
        (
            "complete direction/subtext story; "
            "editorial text ends at subtext despite "
            "overlapping next-question timestamps"
        ),
    )

def resolve_iconic_role(words, candidate):

    start = 806.420
    end = 822.600

    start_match = find_word_sequence(
        words,
        [
            "Who",
            "is",
            "there",
            "an",
            "iconic",
            "role",
        ],
        806.0,
        808.0,
    )

    end_match = find_word_sequence(
        words,
        [
            "I",
            "can",
            "see",
            "it",
        ],
        821.0,
        823.0,
    )

    if start_match is None:
        raise RuntimeError(
            "Could not locate iconic-role START phrase."
        )

    if end_match is None:
        raise RuntimeError(
            "Could not locate iconic-role END phrase."
        )

    start_index = start_match[0]
    end_index = end_match[1]

    text = build_editorial_text(
        words,
        start_index,
        end_index,
    )

    return make_result(
        candidate,
        start,
        end,
        text,
        (
            "complete iconic-role answer; "
            "editorial text ends at I can see it "
            "before interviewer reaction"
        ),
    )

def resolve(words, candidate):
    start = float(candidate["start"])

    if abs(start - 951.880) < 1.0:
        return resolve_performance(words, candidate)

    if abs(start - 841.000) < 1.0:
        return resolve_direction(words, candidate)

    if abs(start - 806.360) < 1.0:
        return resolve_iconic_role(words, candidate)

    raise RuntimeError(
        f"Unexpected candidate start: {start}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    print("=== EDITORIAL RANKER v24 FINAL ===")

    source = load_json(args.input)
    transcript = load_json(args.transcript)

    words = transcript.get("words", [])

    if not words:
        raise RuntimeError(
            "No top-level words found in transcript."
        )

    candidates = source.get("selected", [])

    if not isinstance(candidates, list):
        raise RuntimeError(
            "Input does not contain selected candidates."
        )

    candidates = candidates[:3]

    print("verified word tokens:", len(words))
    print("v21 selected candidates:", len(candidates))

    results = []

    for rank, candidate in enumerate(candidates, 1):

        result = resolve(words, candidate)

        print()
        print(f"#{rank}")
        print(
            f"{result['start']:.3f} -> "
            f"{result['end']:.3f} "
            f"({result['duration']:.3f}s)"
        )
        print(result["editorial_text_v24"])

        results.append(result)

    output = {
        "version": "24.0-final",
        "module": "editorial_ranker_v24_final",
        "source_of_candidates": "v21_selected_top3",
        "word_timestamp_source": (
            "new_aligned_transcript.json:words"
        ),
        "selected": results,
    }

    save_json(args.output, output)

    print()
    print("selected:", len(results))
    print("saved:", args.output)


if __name__ == "__main__":
    main()