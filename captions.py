import json
import os
import re
from typing import Dict, List


MAX_WORDS = 2
MAX_CHARS = 18

MIN_DURATION = 0.06
MAX_DURATION = 1.05


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def load_transcript(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        segments = data.get("segments")

        if isinstance(segments, list):
            return segments

    raise ValueError(
        f"Could not find transcript segments in {path}"
    )


def get_words(segment: Dict) -> List[Dict]:
    """
    Use WhisperX word-level timestamps.

    If WhisperX could not align a particular segment,
    fall back to estimating the word timestamps from
    the segment timing.
    """

    real_words = segment.get("words")

    if isinstance(real_words, list) and real_words:

        result = []

        for word in real_words:
            try:
                text = clean_text(
                    str(word.get("word", ""))
                )

                start = float(word["start"])
                end = float(word["end"])

                if not text:
                    continue

                if end <= start:
                    continue

                result.append({
                    "text": text,
                    "start": start,
                    "end": end,
                })

            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                continue

        if result:
            return result

    # -----------------------------------------------------
    # Fallback for segments WhisperX couldn't align
    # -----------------------------------------------------

    text = clean_text(
        str(segment.get("text", ""))
    )

    if not text:
        return []

    try:
        start = float(segment["start"])
        end = float(segment["end"])
    except (
        KeyError,
        TypeError,
        ValueError,
    ):
        return []

    if end <= start:
        return []

    words = text.split()

    if not words:
        return []

    duration = end - start

    weights = [
        max(
            1.0,
            len(re.sub(r"[^\w]", "", word))
        )
        for word in words
    ]

    total_weight = sum(weights)

    if total_weight <= 0:
        return []

    result = []

    current = start

    for word, weight in zip(words, weights):

        word_duration = (
            duration
            * weight
            / total_weight
        )

        result.append({
            "text": word,
            "start": current,
            "end": current + word_duration,
        })

        current += word_duration

    return result


def create_clip_captions(
    clip: Dict,
    transcript_path: str,
) -> List[Dict]:

    clip_start = float(clip["start"])
    clip_end = float(clip["end"])

    segments = load_transcript(
        transcript_path
    )

    words = []

    for segment in segments:

        try:
            seg_start = float(segment["start"])
            seg_end = float(segment["end"])
        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

        if seg_end <= clip_start:
            continue

        if seg_start >= clip_end:
            continue

        for word in get_words(segment):

            if word["end"] <= clip_start:
                continue

            if word["start"] >= clip_end:
                continue

            words.append(word)

    # Make sure chronological
    words.sort(
        key=lambda x: x["start"]
    )

    captions = []

    i = 0

    while i < len(words):

        group = words[
            i:i + MAX_WORDS
        ]

        # Keep captions short
        while (
            len(group) > 1
            and len(
                " ".join(
                    x["text"]
                    for x in group
                )
            ) > MAX_CHARS
        ):
            group.pop()

        if not group:
            i += 1
            continue

        absolute_start = max(
            clip_start,
            group[0]["start"],
        )

        absolute_end = min(
            clip_end,
            group[-1]["end"],
        )

        if absolute_end <= absolute_start:
            i += len(group)
            continue

        duration = (
            absolute_end
            - absolute_start
        )

        duration = min(
            duration,
            MAX_DURATION,
        )

        duration = max(
            duration,
            MIN_DURATION,
        )

        relative_start = (
            absolute_start
            - clip_start
        )

        relative_end = min(
            clip_end - clip_start,
            relative_start + duration,
        )

        text = " ".join(
            word["text"]
            for word in group
        ).upper()

        captions.append({
            "start": relative_start,
            "end": relative_end,
            "text": text,
        })

        i += len(group)

    return captions


def ass_time(seconds: float) -> str:

    if seconds < 0:
        seconds = 0

    hours = int(
        seconds // 3600
    )

    minutes = int(
        (seconds % 3600) // 60
    )

    whole_seconds = int(
        seconds % 60
    )

    centiseconds = int(
        round(
            (
                seconds
                - int(seconds)
            ) * 100
        )
    )

    if centiseconds >= 100:
        whole_seconds += 1
        centiseconds = 0

    return (
        f"{hours}:"
        f"{minutes:02d}:"
        f"{whole_seconds:02d}."
        f"{centiseconds:02d}"
    )


def escape_ass(text: str) -> str:

    text = text.replace(
        "\\",
        r"\\",
    )

    text = text.replace(
        "{",
        r"\{",
    )

    text = text.replace(
        "}",
        r"\}",
    )

    return text


def create_clip_ass(
    clip: Dict,
    output_path: str,
    transcript_path: str,
):

    captions = create_clip_captions(
        clip,
        transcript_path,
    )

    print(
        f"      Generated "
        f"{len(captions)} caption chunks"
    )

    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Shorts,Arial,68,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,-1,0,0,100,100,0,0,1,5,2,5,60,60,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    os.makedirs(
        os.path.dirname(output_path)
        or ".",
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(header)

        for caption in captions:

            start = ass_time(
                caption["start"]
            )

            end = ass_time(
                caption["end"]
            )

            text = escape_ass(
                caption["text"]
            )

            styled_text = (
                r"{\pos(540,1130)\bord5\shad2}"
                + text
            )

            f.write(
                f"Dialogue: 0,"
                f"{start},"
                f"{end},"
                f"Shorts,,0,0,0,,"
                f"{styled_text}\n"
            )