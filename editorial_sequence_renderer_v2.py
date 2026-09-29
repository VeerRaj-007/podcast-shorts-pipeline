from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

TIMELINE_EPS = 0.001
DURATION_EPS = 0.01
FINAL_DURATION_TOLERANCE = 0.10


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def probe_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    return float(subprocess.check_output(cmd, text=True).strip())


def run(cmd: list[str]) -> None:
    print(">", " ".join(map(str, cmd)))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed with exit code {result.returncode}"
        )


def get_actions(plan: dict) -> list[dict]:
    actions = plan.get("actions")
    if actions is None:
        actions = plan.get("timeline_actions")

    if not isinstance(actions, list) or not actions:
        raise ValueError(
            "Finalized plan contains no actions. "
            "Refusing to render an empty plan."
        )

    return actions


def collect_segments(actions: list[dict]) -> list[dict]:
    segments: list[dict] = []

    for action_index, action in enumerate(actions):
        action_segments = action.get("render_segments")

        if not isinstance(action_segments, list) or not action_segments:
            raise ValueError(
                f"Action {action_index} contains no render_segments."
            )

        for segment_index, original in enumerate(action_segments):
            segment = dict(original)
            segment["_action_index"] = action_index
            segment["_segment_index"] = segment_index
            segments.append(segment)

    return segments


def validate_plan(actions: list[dict], source_duration: float) -> float:
    expected_output = 0.0

    for action_index, action in enumerate(actions):
        action_start = float(action["start"])
        action_end = float(action["end"])

        if abs(action_start - expected_output) > TIMELINE_EPS:
            raise ValueError(
                f"Action timeline gap/overlap at {action_index}: "
                f"expected {expected_output:.6f}, got {action_start:.6f}"
            )

        if action_end <= action_start:
            raise ValueError(f"Invalid action {action_index}.")

        expected_segment = action_start

        for segment_index, segment in enumerate(action["render_segments"]):
            required = ("output_start", "output_end", "source_in", "source_out")
            for key in required:
                if key not in segment:
                    raise ValueError(
                        f"Missing {key} at {action_index}:{segment_index}"
                    )

            out_start = float(segment["output_start"])
            out_end = float(segment["output_end"])
            source_in = float(segment["source_in"])
            source_out = float(segment["source_out"])

            if abs(out_start - expected_segment) > TIMELINE_EPS:
                raise ValueError(
                    f"Segment gap/overlap at {action_index}:{segment_index}: "
                    f"expected {expected_segment:.6f}, got {out_start:.6f}"
                )

            if out_end <= out_start:
                raise ValueError(f"Invalid output range at {action_index}:{segment_index}")

            if source_in < -TIMELINE_EPS or source_out > source_duration + TIMELINE_EPS:
                raise ValueError(
                    f"Source range outside video at {action_index}:{segment_index}: "
                    f"{source_in:.6f}->{source_out:.6f}"
                )

            if source_out <= source_in:
                raise ValueError(f"Invalid source range at {action_index}:{segment_index}")

            output_len = out_end - out_start
            source_len = source_out - source_in

            if abs(output_len - source_len) > DURATION_EPS:
                raise ValueError(
                    f"Duration mismatch at {action_index}:{segment_index}: "
                    f"output={output_len:.6f}, source={source_len:.6f}"
                )

            expected_segment = out_end

        if abs(expected_segment - action_end) > TIMELINE_EPS:
            raise ValueError(
                f"Action {action_index} end does not match render segments."
            )

        expected_output = action_end

    return expected_output


def validate_no_source_reuse(segments: list[dict]) -> None:
    candidates = [
        s for s in segments
        if not bool(s.get("replay", False))
    ]

    ordered = sorted(
        candidates,
        key=lambda s: float(s["output_start"]),
    )

    active: list[dict] = []

    for current in ordered:
        current_start = float(current["source_in"])
        current_end = float(current["source_out"])

        active = [
            old for old in active
            if float(old["source_out"]) > current_start + TIMELINE_EPS
        ]

        for old in active:
            old_start = float(old["source_in"])
            old_end = float(old["source_out"])

            overlap = min(current_end, old_end) - max(
                current_start, old_start
            )

            if overlap > 0.05:
                raise ValueError(
                    "Accidental source reuse detected: "
                    f"{old['_action_index']}:{old['_segment_index']} "
                    f"overlaps "
                    f"{current['_action_index']}:{current['_segment_index']} "
                    f"by {overlap:.3f}s."
                )

        active.append(current)


def build_filter_graph(segments: list[dict]) -> str:
    lines: list[str] = []
    video_labels: list[str] = []
    audio_labels: list[str] = []

    for index, segment in enumerate(segments):
        source_in = float(segment["source_in"])
        source_out = float(segment["source_out"])

        v = f"v{index}"
        a = f"a{index}"

        lines.append(
            f"[0:v]trim=start={source_in:.6f}:end={source_out:.6f},"
            f"setpts=PTS-STARTPTS[{v}]"
        )

        lines.append(
            f"[0:a]atrim=start={source_in:.6f}:end={source_out:.6f},"
            f"asetpts=PTS-STARTPTS[{a}]"
        )

        video_labels.append(f"[{v}]")
        audio_labels.append(f"[{a}]")

    concat_inputs = "".join(
        video_labels[i] + audio_labels[i]
        for i in range(len(segments))
    )

    lines.append(
        f"{concat_inputs}concat=n={len(segments)}:v=1:a=1[outv][outa]"
    )

    return ";\n".join(lines) + ";\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a finalized Editra editorial sequence."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()

    video = Path(args.video)
    sequence_path = Path(args.sequence)
    output = Path(args.output)

    if not video.exists():
        raise FileNotFoundError(f"Source video not found: {video}")
    if not sequence_path.exists():
        raise FileNotFoundError(f"Sequence not found: {sequence_path}")

    plan = load_json(sequence_path)

    if plan.get("render_safe") is not True:
        raise ValueError("Sequence is not render_safe=true.")

    actions = get_actions(plan)
    segments = collect_segments(actions)
    source_duration = probe_duration(video)

    print("=== PLAN VALIDATION ===")
    expected_duration = validate_plan(actions, source_duration)

    print(f"Actions: {len(actions)}")
    print(f"Render segments: {len(segments)}")
    print(f"Source duration: {source_duration:.6f}s")
    print(f"Expected output duration: {expected_duration:.6f}s")

    print("=== SOURCE UNIQUENESS VALIDATION ===")
    validate_no_source_reuse(segments)
    print("Source reuse: 0 overlapping intervals")

    output.parent.mkdir(parents=True, exist_ok=True)

    filter_file = output.with_suffix(".filter_complex.txt")
    filter_file.write_text(
        build_filter_graph(segments),
        encoding="utf-8",
    )

    filter_graph = filter_file.read_text(
    encoding="utf-8"
)

    command = [
    "ffmpeg", "-y",
    "-i", str(video),
    "-filter_complex", filter_graph,
    "-map", "[outv]",
    "-map", "[outa]",
    "-t", f"{expected_duration:.6f}",
    "-c:v", "libx264",
    "-preset", "fast",
    "-crf", "18",
    "-c:a", "aac",
    "-b:a", "192k",
    "-avoid_negative_ts", "make_zero",
    "-movflags", "+faststart",
    str(output),
    ]

    print("=== SINGLE-PASS RENDER ===")
    run(command)

    if not output.exists():
        raise RuntimeError("Renderer completed without creating output.")

    actual_duration = probe_duration(output)
    duration_error = actual_duration - expected_duration

    print("=== FINAL DURATION VALIDATION ===")
    print(f"Expected: {expected_duration:.6f}s")
    print(f"Actual:   {actual_duration:.6f}s")
    print(f"Error:    {duration_error:+.6f}s")

    if abs(duration_error) > FINAL_DURATION_TOLERANCE:
        raise RuntimeError(
            f"Duration validation failed: "
            f"{actual_duration:.6f}s vs {expected_duration:.6f}s"
        )

    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else Path(str(output) + ".manifest.json")
    )

    manifest = {
        "renderer": "editra_editorial_sequence_renderer_v2",
        "source_video": str(video),
        "sequence": str(sequence_path),
        "output": str(output),
        "single_final_encode": True,
        "concat_demuxer": False,
        "independent_segment_encoding": False,
        "actions": len(actions),
        "render_segments": len(segments),
        "expected_duration": expected_duration,
        "actual_duration": actual_duration,
        "duration_error": duration_error,
        "duration_tolerance": FINAL_DURATION_TOLERANCE,
        "source_reuse_count": 0,
        "status": "validated",
    }

    save_json(manifest_path, manifest)

    print(json.dumps(manifest, indent=2))
    print("\nSUCCESS: Editra sequence rendered and duration-validated.")


if __name__ == "__main__":
        main()
