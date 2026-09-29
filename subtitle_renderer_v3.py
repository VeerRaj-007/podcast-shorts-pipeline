from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any


STYLES: dict[str, dict[str, Any]] = {
    "clean_white": {
        "font": "Arial",
        "size": 54,
        "bold": True,
        "primary": "&H00FFFFFF",
        "secondary": "&H00FFFFFF",
        "outline": "&H00000000",
        "outline_width": 3,
        "shadow": 2,
        "alignment": 2,
        "margin_v": 245,
        "margin_l": 70,
        "margin_r": 70,
        "mode": "static",
        "box": False,
        "tracking": 0,
    },
    "bold_tiktok": {
        "font": "Arial",
        "size": 67,
        "bold": True,
        "primary": "&H00FFFFFF",
        "secondary": "&H0000E6FF",
        "outline": "&H00000000",
        "outline_width": 5,
        "shadow": 3,
        "alignment": 2,
        "margin_v": 280,
        "margin_l": 55,
        "margin_r": 55,
        "mode": "keyword",
        "box": True,
        "box_color": "&HB0000000",
        "tracking": -1,
    },
    "high_energy": {
        "font": "Arial",
        "size": 70,
        "bold": True,
        "primary": "&H00FFFFFF",
        "secondary": "&H00303BFF",
        "outline": "&H00000000",
        "outline_width": 6,
        "shadow": 3,
        "alignment": 2,
        "margin_v": 300,
        "margin_l": 45,
        "margin_r": 45,
        "mode": "keyword",
        "box": False,
        "tracking": -1,
    },
    "karaoke": {
        "font": "Arial",
        "size": 58,
        "bold": True,
        "primary": "&H00FFFFFF",
        "secondary": "&H00E5FF00",
        "outline": "&H00000000",
        "outline_width": 4,
        "shadow": 2,
        "alignment": 2,
        "margin_v": 275,
        "margin_l": 60,
        "margin_r": 60,
        "mode": "karaoke",
        "box": False,
        "tracking": 0,
    },
    "cinematic": {
        "font": "Georgia",
        "size": 48,
        "bold": False,
        "primary": "&H00FFFFFF",
        "secondary": "&H00FFFFFF",
        "outline": "&H00111111",
        "outline_width": 2,
        "shadow": 2,
        "alignment": 2,
        "margin_v": 235,
        "margin_l": 100,
        "margin_r": 100,
        "mode": "static",
        "box": False,
        "tracking": 1,
    },
    "podcast_host": {
        "font": "Arial",
        "size": 55,
        "bold": True,
        "primary": "&H00FFFFFF",
        "secondary": "&H0000FF85",
        "outline": "&H00000000",
        "outline_width": 3,
        "shadow": 2,
        "alignment": 2,
        "margin_v": 250,
        "margin_l": 75,
        "margin_r": 75,
        "mode": "keyword",
        "box": False,
        "tracking": 0,
    },
    "gaming": {
        "font": "Arial",
        "size": 68,
        "bold": True,
        "primary": "&H00FFFFFF",
        "secondary": "&H0066FF00",
        "outline": "&H00000000",
        "outline_width": 6,
        "shadow": 3,
        "alignment": 2,
        "margin_v": 285,
        "margin_l": 45,
        "margin_r": 45,
        "mode": "keyword",
        "box": True,
        "box_color": "&H90000000",
        "tracking": -1,
    },
    "news": {
        "font": "Arial",
        "size": 51,
        "bold": True,
        "primary": "&H00FFFFFF",
        "secondary": "&H00FFFFFF",
        "outline": "&H00000000",
        "outline_width": 3,
        "shadow": 1,
        "alignment": 1,
        "margin_v": 235,
        "margin_l": 90,
        "margin_r": 60,
        "mode": "static",
        "box": True,
        "box_color": "&HB0000000",
        "tracking": 0,
    },
    "typewriter": {
        "font": "Courier New",
        "size": 50,
        "bold": True,
        "primary": "&H00FFFFFF",
        "secondary": "&H000AD6FF",
        "outline": "&H00000000",
        "outline_width": 3,
        "shadow": 1,
        "alignment": 2,
        "margin_v": 235,
        "margin_l": 70,
        "margin_r": 70,
        "mode": "karaoke",
        "box": True,
        "box_color": "&H90000000",
        "tracking": 0,
    },
    "minimal_upper": {
        "font": "Arial",
        "size": 52,
        "bold": True,
        "primary": "&H00FFFFFF",
        "secondary": "&H00FFFFFF",
        "outline": "&H00000000",
        "outline_width": 3,
        "shadow": 2,
        "alignment": 5,
        "margin_v": 150,
        "margin_l": 65,
        "margin_r": 65,
        "mode": "static",
        "box": False,
        "tracking": 0,
    },
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ass_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    cs = int(round(seconds * 100))
    h = cs // 360000
    cs %= 360000
    m = cs // 6000
    cs %= 6000
    s = cs // 100
    c = cs % 100
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def escape_ass(text: str) -> str:
    return (
        str(text)
        .replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\n", " ")
    )


def keyword_indices(words: list[dict[str, Any]]) -> set[int]:
    """
    Lightweight emphasis heuristic.
    In production this can be replaced by an LLM/editorial emphasis model.
    """
    stop = {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
        "from", "has", "have", "how", "i", "if", "in", "is", "it", "me",
        "my", "of", "on", "or", "so", "that", "the", "to", "was", "we",
        "were", "what", "when", "you", "your",
    }

    scored: list[tuple[float, int]] = []

    for i, word in enumerate(words):
        raw = str(word.get("text", "")).strip(".,!?;:'\"").lower()

        if not raw or raw in stop:
            continue

        duration = max(
            0.01,
            float(word["end"]) - float(word["start"]),
        )
        score = math.log1p(len(raw)) + min(duration, 0.6)

        if len(raw) >= 7:
            score += 0.8

        scored.append((score, i))

    scored.sort(reverse=True)

    # Emphasize at most two words per caption.
    return {i for _, i in scored[:2]}


def karaoke_text(
    words: list[dict[str, Any]],
    source_start: float,
) -> str:
    parts: list[str] = []

    for word in words:
        start = float(word["start"]) - source_start
        end = float(word["end"]) - source_start
        duration_cs = max(1, int(round((end - start) * 100)))
        text = escape_ass(word.get("text", ""))

        if text:
            parts.append(f"{{\\k{duration_cs}}}{text}")

    return " ".join(parts)


def keyword_text(
    words: list[dict[str, Any]],
    style: dict[str, Any],
) -> str:
    emphasized = keyword_indices(words)
    parts: list[str] = []

    for i, word in enumerate(words):
        text = escape_ass(word.get("text", ""))

        if i in emphasized:
            color = style["secondary"]
            parts.append(
                f"{{\\c{color}\\fscx106\\fscy106}}{text}{{\\rDefault}}"
            )
        else:
            parts.append(text)

    return " ".join(parts)


def static_text(
    words: list[dict[str, Any]],
    caption_text: str,
) -> str:
    if words:
        return " ".join(
            escape_ass(word.get("text", ""))
            for word in words
        )
    return escape_ass(caption_text)


def choose_position(
    style: dict[str, Any],
    caption: dict[str, Any],
    camera_data: dict[str, Any] | None,
) -> tuple[int, int]:
    """
    Returns (alignment, margin_v).

    Default is lower-center. If the camera plan says we're focused on a
    subject and the subject occupies the lower-central area, lift the caption
    slightly. This is a conservative automatic placement heuristic.
    """
    alignment = int(style["alignment"])
    margin_v = int(style["margin_v"])

    if camera_data:
        segs = camera_data.get("camera_segments", [])

        start = float(caption["start"])
        end = float(caption["end"])

        for seg in segs:
            seg_start = float(seg.get("source_start", 0))
            seg_end = float(seg.get("source_end", 0))

            if end <= seg_start or start >= seg_end:
                continue

            if seg.get("composition") == "speaker_focus":
                # Don't cover chest/hand gestures in a close portrait.
                if alignment == 2:
                    margin_v = max(margin_v, 335)

            break

    return alignment, margin_v


def build_ass(
    timing_data: dict[str, Any],
    style_id: str,
    style: dict[str, Any],
    camera_data: dict[str, Any] | None,
) -> str:
    source_start = float(timing_data["source_range"][0])
    captions = timing_data.get("captions", [])

    bold = -1 if style["bold"] else 0
    border_style = 3 if style.get("box") else 1

    if style.get("box"):
        back_color = style.get("box_color", "&H90000000")
    else:
        back_color = "&H00000000"

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes
WrapStyle: 2
YCbCr Matrix: TV.709
Title: Subtitle Renderer v3
"""

    header += f"""
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{style['font']},{style['size']},{style['primary']},{style['secondary']},{style['outline']},{back_color},{bold},0,0,0,100,100,{style['tracking']},0,{border_style},{style['outline_width']},{style['shadow']},{style['alignment']},{style['margin_l']},{style['margin_r']},{style['margin_v']},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events: list[str] = []

    for caption in captions:
        abs_start = float(caption["start"])
        abs_end = float(caption["end"])

        start = abs_start - source_start
        end = abs_end - source_start

        if end <= start:
            continue

        words = caption.get("words", [])

        if style["mode"] == "karaoke" and words:
            text = karaoke_text(words, source_start)
        elif style["mode"] == "keyword" and words:
            text = keyword_text(words, style)
        else:
            text = static_text(words, caption.get("text", ""))

        alignment, margin_v = choose_position(
            style,
            caption,
            camera_data,
        )

        # Per-event override lets the same style move safely without changing
        # the global preset definition.
        position_tag = f"{{\\an{alignment}\\q2}}"
        if margin_v != style["margin_v"]:
            position_tag += f"{{\\pos(540,{1920 - margin_v})}}"

        # Small, restrained entrance rather than an aggressive whole-caption
        # scale jump.
        if style_id in {"bold_tiktok", "high_energy", "gaming"}:
            animation = r"{\fad(70,50)\t(0,90,\fscx103\fscy103)}"
        else:
            animation = r"{\fad(110,70)}"

        final_text = position_tag + animation + text

        events.append(
            f"Dialogue: 0,{ass_time(start)},{ass_time(end)},"
            f"Default,,0,0,0,,{final_text}"
        )

    return header + "\n".join(events) + "\n"


def run_ffmpeg(
    video: Path,
    ass_file: Path,
    output: Path,
) -> None:
    ass_path = ass_file.resolve().as_posix()
    ass_path = ass_path.replace(":", r"\:")

    vf = f"ass='{ass_path}'"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(output),
    ]

    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", required=True)
    parser.add_argument("--timing", required=True)
    parser.add_argument("--style", choices=sorted(STYLES), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--camera", default=None)

    args = parser.parse_args()

    video = Path(args.video)
    timing = Path(args.timing)
    output = Path(args.output)

    if not video.exists():
        raise FileNotFoundError(video)

    if not timing.exists():
        raise FileNotFoundError(timing)

    timing_data = load_json(timing)

    camera_data = None
    if args.camera:
        camera_path = Path(args.camera)
        if camera_path.exists():
            camera_data = load_json(camera_path)

    if not timing_data.get("captions"):
        raise RuntimeError("No captions found in timing JSON.")

    source_range = timing_data.get("source_range")
    if not isinstance(source_range, list) or len(source_range) != 2:
        raise RuntimeError(
            "Timing JSON must contain source_range: [start, end]."
        )

    style = STYLES[args.style]

    print("=== SUBTITLE RENDERER v3 ===")
    print(f"style: {args.style}")
    print(f"captions: {len(timing_data['captions'])}")
    print(f"source range: {source_range[0]} -> {source_range[1]}")
    print("timing: WhisperX aligned words")

    with tempfile.TemporaryDirectory(
        prefix="subtitle_renderer_v3_"
    ) as td:
        ass_file = Path(td) / "captions.ass"

        ass_text = build_ass(
            timing_data,
            args.style,
            style,
            camera_data,
        )

        ass_file.write_text(
            ass_text,
            encoding="utf-8",
        )

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        run_ffmpeg(
            video,
            ass_file,
            output,
        )

    print(f"output: {output}")


if __name__ == "__main__":
    main()