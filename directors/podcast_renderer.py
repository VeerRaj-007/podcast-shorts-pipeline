
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List


OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920
OUTPUT_ASPECT = OUTPUT_WIDTH / OUTPUT_HEIGHT


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_ffmpeg(command: List[str]) -> None:
    print("Running FFmpeg...")

    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if process.returncode != 0:
        print(process.stdout)
        raise RuntimeError(
            f"FFmpeg failed with exit code "
            f"{process.returncode}"
        )

    if process.stdout:
        print(process.stdout)


def build_cover_filter() -> str:
    """
    Scale the source so the output frame is completely covered,
    then center-crop safely to 1080x1920.

    The crop dimensions are based on the post-scale frame,
    so they cannot exceed the scaled dimensions.
    """

    return (
        "scale="
        f"w={OUTPUT_WIDTH}:"
        f"h={OUTPUT_HEIGHT}:"
        f"force_original_aspect_ratio=increase:"
        f"eval=frame,"
        "crop="
        f"w={OUTPUT_WIDTH}:"
        f"h={OUTPUT_HEIGHT}:"
        "x=(iw-1080)/2:"
        "y=(ih-1920)/2,"
        "setsar=1,"
        "format=yuv420p"
    )


def clamp01(value: float) -> float:
    return max(
        0.0,
        min(1.0, value),
    )


def build_two_shot_filter() -> str:
    """
    Base podcast framing.

    For the first renderer test we use a full-frame vertical
    presentation of the source, filling 1080x1920 without
    requesting an impossible crop before scaling.
    """

    return build_cover_filter()


def build_speaker_closeup_filter(
    action: Dict[str, Any],
) -> str:
    """
    Render a static face close-up.

    The director stores normalized crop coordinates. We convert
    them into a safe normalized crop and let FFmpeg perform the
    crop after a scale-to-fit stage.
    """

    crop = action.get("crop")

    if not isinstance(crop, dict):
        raise ValueError(
            "speaker_closeup action has no crop data."
        )

    x = clamp01(
        float(crop.get("x", 0.0))
    )
    y = clamp01(
        float(crop.get("y", 0.0))
    )
    width = clamp01(
        float(crop.get("width", 1.0))
    )
    height = clamp01(
        float(crop.get("height", 1.0))
    )

    # Protect against invalid / degenerate crops.
    width = max(
        0.05,
        min(width, 1.0),
    )

    height = max(
        0.05,
        min(height, 1.0),
    )

    # Clamp top-left after width/height are known.
    x = min(
        x,
        1.0 - width,
    )

    y = min(
        y,
        1.0 - height,
    )

    # We first scale the entire source to a large working frame
    # while preserving aspect ratio. Then we crop using safe
    # fractions of iw/ih.
    #
    # The crop dimensions are guaranteed to be <= iw/ih.
    #
    return (
        # Keep enough resolution for the closeup.
        "scale="
        "w=iw*1.0:"
        "h=ih*1.0:"
        "eval=frame,"
        "crop="
        f"w=max(2\\,trunc(iw*{width}/2)*2):"
        f"h=max(2\\,trunc(ih*{height}/2)*2):"
        f"x=max(0\\,min(iw-(max(2\\,trunc(iw*{width}/2)*2))\\,"
        f"trunc(iw*{x}/2)*2)):"
        f"y=max(0\\,min(ih-(max(2\\,trunc(ih*{height}/2)*2))\\,"
        f"trunc(ih*{y}/2)*2)),"
        f"scale="
        f"w={OUTPUT_WIDTH}:"
        f"h={OUTPUT_HEIGHT}:"
        "force_original_aspect_ratio=decrease:"
        "eval=frame,"
        f"pad={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:"
        "(ow-iw)/2:"
        "(oh-ih)/2,"
        "setsar=1,"
        "format=yuv420p"
    )


def build_video_filter(
    index: int,
    action: Dict[str, Any],
) -> str:
    start = float(
        action["start"]
    )

    end = float(
        action["end"]
    )

    trim = (
        "[0:v]"
        f"trim=start={start}:end={end},"
        "setpts=PTS-STARTPTS"
    )

    shot = action.get(
        "shot",
        "two_shot",
    )

    if shot == "two_shot":
        body = build_two_shot_filter()

    elif shot == "speaker_closeup":
        body = build_speaker_closeup_filter(
            action
        )

    else:
        raise ValueError(
            f"Unsupported shot type: {shot}"
        )

    return (
        f"{trim},"
        f"{body}"
        f"[v{index}]"
    )


def build_audio_filter(
    index: int,
    action: Dict[str, Any],
) -> str:
    start = float(
        action["start"]
    )

    end = float(
        action["end"]
    )

    return (
        "[0:a]"
        f"atrim=start={start}:end={end},"
        "asetpts=PTS-STARTPTS,"
        "aformat="
        "sample_fmts=fltp:"
        "sample_rates=48000:"
        "channel_layouts=stereo"
        f"[a{index}]"
    )


def build_concat_filter(
    count: int,
) -> str:

    inputs = []

    for index in range(count):
        inputs.append(
            f"[v{index}]"
        )
        inputs.append(
            f"[a{index}]"
        )

    return (
        "".join(inputs)
        + f"concat=n={count}:v=1:a=1"
        "[vout][aout]"
    )


def sanitize_plan(
    plan: List[Dict[str, Any]],
    video_start: float,
    video_end: float,
) -> List[Dict[str, Any]]:

    result = []

    for item in plan:

        start = max(
            video_start,
            float(
                item.get(
                    "start",
                    0.0,
                )
            ),
        )

        end = min(
            video_end,
            float(
                item.get(
                    "end",
                    0.0,
                )
            ),
        )

        if end <= start:
            continue

        cleaned = dict(
            item
        )

        cleaned["start"] = start
        cleaned["end"] = end

        result.append(
            cleaned
        )

    result.sort(
        key=lambda item:
            item["start"]
    )

    return result


def ensure_coverage(
    plan: List[Dict[str, Any]],
    video_start: float,
    video_end: float,
) -> List[Dict[str, Any]]:

    result = []

    cursor = video_start

    for action in plan:

        start = float(
            action["start"]
        )

        end = float(
            action["end"]
        )

        if start > cursor + 0.001:

            result.append(
                {
                    "start":
                        cursor,
                    "end":
                        start,
                    "shot":
                        "two_shot",
                    "speaker":
                        None,
                    "face_id":
                        None,
                    "confidence":
                        1.0,
                    "importance":
                        0.0,
                    "reason":
                        "gap_fill",
                    "crop":
                        None,
                }
            )

        if start < cursor:
            action = dict(
                action
            )
            action["start"] = cursor

        if action["end"] <= action["start"]:
            continue

        result.append(
            action
        )

        cursor = max(
            cursor,
            action["end"],
        )

    if cursor < video_end - 0.001:

        result.append(
            {
                "start":
                    cursor,
                "end":
                    video_end,
                "shot":
                    "two_shot",
                "speaker":
                    None,
                "face_id":
                    None,
                "confidence":
                    1.0,
                "importance":
                    0.0,
                "reason":
                    "tail_fill",
                "crop":
                    None,
            }
        )

    return result


def validate_plan(
    plan: List[Dict[str, Any]],
    video_start: float,
    video_end: float,
) -> None:

    if not plan:
        raise ValueError(
            "No actions in camera plan."
        )

    previous_end = video_start

    for index, action in enumerate(
        plan
    ):

        start = float(
            action["start"]
        )

        end = float(
            action["end"]
        )

        if start < video_start - 0.01:
            raise ValueError(
                f"Action {index} starts before requested window."
            )

        if end > video_end + 0.01:
            raise ValueError(
                f"Action {index} ends after requested window."
            )

        if end <= start:
            raise ValueError(
                f"Action {index} has non-positive duration."
            )

        if start < previous_end - 0.01:
            raise ValueError(
                f"Action {index} overlaps the previous action."
            )

        previous_end = end

    if abs(
        previous_end
        - video_end
    ) > 0.05:

        raise ValueError(
            "Camera plan does not cover the complete window."
        )


def render(
    video_path: str,
    plan_path: str,
    output_path: str,
    video_start: float,
    video_end: float,
) -> None:

    data = load_json(
        plan_path
    )

    plan = data.get(
        "plan",
        [],
    )

    plan = sanitize_plan(
        plan,
        video_start,
        video_end,
    )

    plan = ensure_coverage(
        plan,
        video_start,
        video_end,
    )

    validate_plan(
        plan,
        video_start,
        video_end,
    )

    print()
    print(
        f"Rendering window: "
        f"{video_start:.2f}s -> "
        f"{video_end:.2f}s"
    )

    print(
        f"Actions: {len(plan)}"
    )

    for index, action in enumerate(
        plan,
        start=1,
    ):

        print(
            f"  {index:02d}. "
            f"{action['start']:.2f}s -> "
            f"{action['end']:.2f}s | "
            f"{action.get('shot', 'two_shot')} | "
            f"{action.get('speaker') or '-'} | "
            f"{action.get('face_id') or '-'}"
        )

    filter_parts = []

    for index, action in enumerate(
        plan
    ):

        filter_parts.append(
            build_video_filter(
                index,
                action,
            )
        )

        filter_parts.append(
            build_audio_filter(
                index,
                action,
            )
        )

    filter_parts.append(
        build_concat_filter(
            len(plan)
        )
    )

    filter_complex = ";".join(
        filter_parts
    )

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        str(video_start),
        "-to",
        str(video_end),
        "-i",
        video_path,
        "-filter_complex",
        filter_complex,
        "-map",
        "[vout]",
        "-map",
        "[aout]",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        output_path,
    ]

    print()
    print(
        "Encoding..."
    )

    run_ffmpeg(
        command
    )

    print()
    print(
        f"Saved: "
        f"{Path(output_path).resolve()}"
    )


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Render an AI-directed podcast camera plan."
        )
    )

    parser.add_argument(
        "--video",
        required=True,
    )

    parser.add_argument(
        "--plan",
        required=True,
    )

    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--end",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    video_path = Path(
        args.video
    )

    plan_path = Path(
        args.plan
    )

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    if not plan_path.exists():
        raise FileNotFoundError(
            f"Plan not found: {plan_path}"
        )

    if args.end <= args.start:
        raise ValueError(
            "--end must be greater than --start."
        )

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    render(
        video_path=str(video_path),
        plan_path=str(plan_path),
        output_path=str(output_path),
        video_start=args.start,
        video_end=args.end,
    )


if __name__ == "__main__":
    main()
