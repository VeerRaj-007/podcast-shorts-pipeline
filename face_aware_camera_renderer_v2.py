from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


OUT_W = 1080
OUT_H = 1920


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run(cmd):
    print(">", " ".join(str(x) for x in cmd))

    result = subprocess.run(
        cmd,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed with code {result.returncode}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    source = Path(args.video)
    plan_path = Path(args.plan)
    output = Path(args.output)

    plan = load_json(plan_path)
    segments = plan["camera_segments"]

    if not segments:
        raise RuntimeError(
            "Camera plan contains no segments."
        )

    # ---------------------------------------------------------
    # IMPORTANT:
    # Every action has its own local output timeline.
    # Therefore output_start alone cannot determine global order.
    #
    # We preserve editorial/action order first, then local timing.
    # ---------------------------------------------------------
    segments = sorted(
        segments,
        key=lambda s: (
            int(s.get("action_index", 0)),
            float(s.get("output_start", 0.0)),
        ),
    )

    filters = []

    video_labels = []
    audio_labels = []

    total_duration = 0.0

    # ---------------------------------------------------------
    # BUILD VIDEO + AUDIO FOR EACH SEGMENT
    # ---------------------------------------------------------
    for i, seg in enumerate(segments):
        source_start = float(seg["source_start"])
        source_end = float(seg["source_end"])

        if source_end <= source_start:
            raise RuntimeError(
                f"Invalid segment {i}: "
                f"{source_start} -> {source_end}"
            )

        duration = source_end - source_start
        total_duration += duration

        crop = seg["crop"]

        left = float(crop["left"])
        top = float(crop["top"])
        width = float(crop["width"])
        height = float(crop["height"])

        # -----------------------------------------------------
        # VIDEO
        # -----------------------------------------------------
        video_label = f"v{i}"

        filters.append(
            f"[0:v]"
            f"trim="
            f"start={source_start:.6f}:"
            f"end={source_end:.6f},"
            f"setpts=PTS-STARTPTS,"
            f"crop="
            f"{width:.6f}:"
            f"{height:.6f}:"
            f"{left:.6f}:"
            f"{top:.6f},"
            f"scale={OUT_W}:{OUT_H}:"
            f"flags=lanczos,"
            f"setsar=1"
            f"[{video_label}]"
        )

        video_labels.append(
            f"[{video_label}]"
        )

        # -----------------------------------------------------
        # AUDIO
        #
        # Each source segment gets its own audio slice.
        # This keeps audio synchronized with the reordered video.
        # -----------------------------------------------------
        audio_label = f"a{i}"

        filters.append(
            f"[0:a]"
            f"atrim="
            f"start={source_start:.6f}:"
            f"end={source_end:.6f},"
            f"asetpts=PTS-STARTPTS"
            f"[{audio_label}]"
        )

        audio_labels.append(
            f"[{audio_label}]"
        )

    # ---------------------------------------------------------
    # CONCATENATE VIDEO
    # ---------------------------------------------------------
    filters.append(
        "".join(video_labels)
        + f"concat=n={len(video_labels)}:v=1:a=0"
        "[video]"
    )

    # ---------------------------------------------------------
    # CONCATENATE AUDIO
    # ---------------------------------------------------------
    filters.append(
        "".join(audio_labels)
        + f"concat=n={len(audio_labels)}:v=0:a=1"
        "[audio]"
    )

    filter_complex = ";".join(filters)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-filter_complex",
        filter_complex,

        "-map",
        "[video]",

        "-map",
        "[audio]",

        "-t",
        f"{total_duration:.6f}",

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

        "-ar",
        "44100",

        "-ac",
        "2",

        "-movflags",
        "+faststart",

        str(output),
    ]

    run(command)

    print()
    print("=== CAMERA RENDER v2 COMPLETE ===")
    print(f"segments     : {len(segments)}")
    print(f"duration     : {total_duration:.3f}s")
    print(f"output       : {output}")


if __name__ == "__main__":
    main()