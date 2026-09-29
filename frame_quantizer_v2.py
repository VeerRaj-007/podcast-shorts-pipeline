from __future__ import annotations

import argparse
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

EPS = 0.001


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def parse_fps(value: str) -> Fraction:
    if "/" in value:
        num, den = value.split("/", 1)
        return Fraction(int(num), int(den))
    return Fraction(value)


def probe_video(path: Path) -> tuple[Fraction, int, float]:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=r_frame_rate,nb_frames",
        "-show_entries",
        "format=duration",
        "-of", "json",
        str(path),
    ]

    data = json.loads(subprocess.check_output(cmd, text=True))

    streams = data.get("streams", [])
    if not streams:
        raise ValueError("No video stream found.")

    stream = streams[0]
    fps = parse_fps(stream["r_frame_rate"])

    reported_frames = int(stream.get("nb_frames") or 0)

    format_data = data.get("format", {})
    duration_value = format_data.get("duration")

    if duration_value is None:
        raise ValueError("FFprobe did not provide container duration.")

    duration = float(duration_value)

    if fps <= 0:
        raise ValueError("Invalid video frame rate.")

    if duration <= 0:
        raise ValueError(
            f"Invalid container duration: {duration}"
        )

    if reported_frames > 0:
        frame_count = reported_frames
    else:
        frame_count = int(
            Fraction(str(duration)) * fps + Fraction(1, 2)
        )

    return fps, frame_count, duration


def get_actions(plan: dict[str, Any]) -> list[dict[str, Any]]:
    actions = plan.get("actions")
    if actions is None:
        actions = plan.get("timeline_actions")

    if not isinstance(actions, list) or not actions:
        raise ValueError("Plan contains no actions.")

    return actions


def quantize_time(seconds: float, fps: Fraction) -> int:
    value = Fraction(str(seconds)) * fps
    return int(value + Fraction(1, 2))


def frame_to_time(frame: int, fps: Fraction) -> float:
    return float(Fraction(frame, 1) / fps)


def source_ranges_overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    a0 = int(a["source_start_frame"])
    a1 = int(a["source_end_frame"])
    b0 = int(b["source_start_frame"])
    b1 = int(b["source_end_frame"])
    return min(a1, b1) > max(a0, b0)


def merge_zero_frame_segment(
    segments: list[dict[str, Any]],
    index: int,
) -> None:
    """
    A zero-frame segment contains no real video frame.

    We remove it while preserving the surrounding source coverage.
    If it is between two segments whose source ranges touch at the same
    frame boundary, those segments are merged when they belong to the same
    editorial action. Otherwise the zero-frame item is simply dropped.
    """
    if not segments:
        return

    segment = segments[index]

    if int(segment["source_frame_count"]) != 0:
        return

    # Prefer merging the boundary into the previous segment if possible.
    if index > 0:
        prev = segments[index - 1]

        if (
            int(prev["source_end_frame"])
            == int(segment["source_start_frame"])
            or int(prev["source_end_frame"])
            == int(segment["source_end_frame"])
        ):
            prev["source_end_frame"] = max(
                int(prev["source_end_frame"]),
                int(segment["source_end_frame"]),
            )
            prev["source_frame_count"] = (
                int(prev["source_end_frame"])
                - int(prev["source_start_frame"])
            )

            segments.pop(index)
            return

    # Otherwise try the next segment.
    if index + 1 < len(segments):
        nxt = segments[index + 1]

        if (
            int(segment["source_end_frame"])
            == int(nxt["source_start_frame"])
            or int(segment["source_start_frame"])
            == int(nxt["source_start_frame"])
        ):
            nxt["source_start_frame"] = min(
                int(nxt["source_start_frame"]),
                int(segment["source_start_frame"]),
            )
            nxt["source_frame_count"] = (
                int(nxt["source_end_frame"])
                - int(nxt["source_start_frame"])
            )

            segments.pop(index)
            return

    # No compatible neighbor: the segment contains no frame and is removed.
    segments.pop(index)


def normalize_action_segments(
    original_segments: list[dict[str, Any]],
    fps: Fraction,
) -> list[dict[str, Any]]:
    quantized: list[dict[str, Any]] = []

    for segment in original_segments:
        source_in = float(segment["source_in"])
        source_out = float(segment["source_out"])

        start_frame = quantize_time(source_in, fps)
        end_frame = quantize_time(source_out, fps)

        item = {
            key: value
            for key, value in segment.items()
            if not key.startswith("_")
        }

        item["source_start_frame"] = start_frame
        item["source_end_frame"] = end_frame
        item["source_frame_count"] = end_frame - start_frame

        quantized.append(item)

    # Repeatedly eliminate zero-frame segments.
    changed = True
    while changed:
        changed = False

        for index, segment in enumerate(quantized):
            if int(segment["source_frame_count"]) == 0:
                merge_zero_frame_segment(
                    quantized,
                    index,
                )
                changed = True
                break

    # Recalculate all source durations.
    for segment in quantized:
        start_frame = int(segment["source_start_frame"])
        end_frame = int(segment["source_end_frame"])

        if end_frame <= start_frame:
            raise ValueError(
                "Internal normalization error: zero-frame segment survived."
            )

        segment["source_frame_count"] = end_frame - start_frame
        segment["source_in"] = frame_to_time(
            start_frame,
            fps,
        )
        segment["source_out"] = frame_to_time(
            end_frame,
            fps,
        )

    return quantized


def build_frame_plan(
    actions: list[dict[str, Any]],
    fps: Fraction,
) -> tuple[list[dict[str, Any]], int, int]:
    output_frame = 0
    frame_merged_segments = 0
    final_actions: list[dict[str, Any]] = []

    for action_index, original_action in enumerate(actions):
        original_segments = original_action.get("render_segments")

        if not isinstance(original_segments, list):
            raise ValueError(
                f"Action {action_index} has no render_segments."
            )

        raw_count = len(original_segments)

        segments = normalize_action_segments(
            original_segments,
            fps,
        )

        frame_merged_segments += raw_count - len(segments)

        if not segments:
            continue

        clean_segments: list[dict[str, Any]] = []

        for segment in segments:
            frame_count = int(segment["source_frame_count"])

            output_start_frame = output_frame
            output_end_frame = (
                output_start_frame + frame_count
            )

            item = dict(segment)

            item.update({
                "output_start_frame": output_start_frame,
                "output_end_frame": output_end_frame,
                "output_start": frame_to_time(
                    output_start_frame,
                    fps,
                ),
                "output_end": frame_to_time(
                    output_end_frame,
                    fps,
                ),
                "duration": frame_to_time(
                    frame_count,
                    fps,
                ),
            })

            clean_segments.append(item)
            output_frame = output_end_frame

        action = {
            key: value
            for key, value in original_action.items()
            if key not in {
                "render_segments",
                "render_safe",
                "start",
                "end",
                "duration",
                "start_frame",
                "end_frame",
                "frame_count",
                "frame_quantized",
            }
        }

        action_start = int(
            clean_segments[0]["output_start_frame"]
        )
        action_end = int(
            clean_segments[-1]["output_end_frame"]
        )

        action.update({
            "start_frame": action_start,
            "end_frame": action_end,
            "frame_count": action_end - action_start,
            "start": frame_to_time(action_start, fps),
            "end": frame_to_time(action_end, fps),
            "duration": frame_to_time(
                action_end - action_start,
                fps,
            ),
            "render_segments": clean_segments,
            "render_safe": True,
            "frame_quantized": True,
        })

        final_actions.append(action)

    return final_actions, output_frame, frame_merged_segments


def validate_plan(
    actions: list[dict[str, Any]],
    fps: Fraction,
    source_frame_count: int,
) -> None:
    if not actions:
        raise ValueError("Frame plan contains zero actions.")

    expected_output_frame = 0
    used_source: list[dict[str, Any]] = []

    for action_index, action in enumerate(actions):
        action_start = int(action["start_frame"])
        action_end = int(action["end_frame"])

        if action_start != expected_output_frame:
            raise ValueError(
                f"Output gap/overlap at action {action_index}: "
                f"expected frame {expected_output_frame}, "
                f"got {action_start}"
            )

        expected_segment = action_start

        for segment_index, segment in enumerate(
            action["render_segments"]
        ):
            source_start = int(segment["source_start_frame"])
            source_end = int(segment["source_end_frame"])
            source_count = int(segment["source_frame_count"])

            output_start = int(segment["output_start_frame"])
            output_end = int(segment["output_end_frame"])

            if source_start < 0 or source_end > source_frame_count:
                raise ValueError(
                    f"Source frame range outside video at "
                    f"{action_index}:{segment_index}: "
                    f"{source_start}->{source_end}"
                )

            if source_end <= source_start:
                raise ValueError(
                    f"Zero/negative source frames survived at "
                    f"{action_index}:{segment_index}."
                )

            if source_end - source_start != source_count:
                raise ValueError(
                    f"Source frame count mismatch at "
                    f"{action_index}:{segment_index}."
                )

            if output_start != expected_segment:
                raise ValueError(
                    f"Output frame gap/overlap at "
                    f"{action_index}:{segment_index}."
                )

            if output_end - output_start != source_count:
                raise ValueError(
                    f"Output/source frame mismatch at "
                    f"{action_index}:{segment_index}."
                )

            # Actual accidental source reuse check.
            if not bool(segment.get("replay", False)):
                for old in used_source:
                    if source_ranges_overlap(old, segment):
                        raise ValueError(
                            f"Source reuse detected at "
                            f"{action_index}:{segment_index}."
                        )
                used_source.append(segment)

            expected_segment = output_end

        if expected_segment != action_end:
            raise ValueError(
                f"Action {action_index} end does not match segments."
            )

        expected_output_frame = action_end

    if expected_output_frame <= 0:
        raise ValueError("Final plan contains zero output frames.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frame-quantize an Editra editorial sequence."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    video = Path(args.video)
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not video.exists():
        raise FileNotFoundError(video)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    plan = load_json(input_path)
    actions = get_actions(plan)

    fps, source_frame_count, source_duration = probe_video(video)

    print("=== SOURCE VIDEO ===")
    print(f"FPS: {fps.numerator}/{fps.denominator}")
    print(f"FPS decimal: {float(fps):.9f}")
    print(f"Source frames: {source_frame_count}")
    print(f"Source duration: {source_duration:.6f}s")

    print("=== FRAME QUANTIZATION ===")

    final_actions, output_frames, merged_count = build_frame_plan(
        actions,
        fps,
    )

    validate_plan(
        final_actions,
        fps,
        source_frame_count,
    )

    final_duration = frame_to_time(
        output_frames,
        fps,
    )

    result = {
        "version": "2.0",
        "mode": "frame_quantized_editorial_sequence",
        "source_sequence": str(input_path),
        "source_video": str(video),
        "render_safe": True,
        "frame_quantized": True,
        "fps": f"{fps.numerator}/{fps.denominator}",
        "fps_decimal": float(fps),
        "source_frame_count": source_frame_count,
        "source_duration": source_duration,
        "requested_video_end": float(
            plan.get(
                "requested_video_end",
                plan.get("video_end", 90.0),
            )
        ),
        "video_end": final_duration,
        "total_output_frames": output_frames,
        "output_timeline": [
            {
                "output_start": action["start"],
                "output_end": action["end"],
                "start_frame": action["start_frame"],
                "end_frame": action["end_frame"],
            }
            for action in final_actions
        ],
        "actions": final_actions,
        "timeline_actions": final_actions,
        "timing_policy": {
            "source_boundaries": "nearest_frame",
            "output_timeline": "integer_frame_accumulation",
            "source_output_duration": "exact_frame_count",
            "zero_frame_segments": "merged_or_removed",
        },
        "stats": {
            "input_actions": len(actions),
            "output_actions": len(final_actions),
            "input_render_segments": sum(
                len(a.get("render_segments", []))
                for a in actions
            ),
            "output_render_segments": sum(
                len(a["render_segments"])
                for a in final_actions
            ),
            "segments_removed_or_merged": merged_count,
            "total_output_frames": output_frames,
            "final_duration": final_duration,
            "duration_vs_previous_plan": (
                final_duration
                - float(plan.get("video_end", final_duration))
            ),
        },
    }

    save_json(output_path, result)

    print(json.dumps(result["stats"], indent=2))
    print(
        "\nSUCCESS: frame-accurate plan created; "
        "sub-frame segments normalized."
    )


if __name__ == "__main__":
        main()