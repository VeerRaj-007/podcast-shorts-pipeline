from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def probe(path: Path) -> tuple[int, int, float]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries",
            "format=duration",
            "-show_entries",
            "stream=width,height",
            "-of", "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed for {path}"
        )

    data = json.loads(result.stdout)

    video = next(
        (
            s for s in data.get("streams", [])
            if s.get("width") is not None
        ),
        None,
    )

    if video is None:
        raise RuntimeError(
            f"No video stream found in {path}"
        )

    return (
        int(video["width"]),
        int(video["height"]),
        float(data["format"]["duration"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render final subtitled Shorts."
    )
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--timing-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--renderer",
        default="subtitle_renderer_v3.py",
    )
    parser.add_argument(
        "--style",
        default="podcast_host",
    )
    args = parser.parse_args()

    videos = sorted(
        args.video_dir.glob("short_*_camera.mp4")
    )

    if not videos:
        raise SystemExit(
            f"No camera videos found in {args.video_dir}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "version": "1.0",
        "style": args.style,
        "rendered_count": 0,
        "outputs": [],
    }

    print("=" * 72)
    print("FINAL SUBTITLE RENDER")
    print("=" * 72)
    print(f"Shorts: {len(videos)}")
    print(f"Style:  {args.style}")
    print()

    for i, video in enumerate(videos, 1):
        timing = (
            args.timing_dir
            / f"short_{i:02d}_timing.json"
        )

        if not timing.exists():
            raise SystemExit(
                f"Missing subtitle timing: {timing}"
            )

        output = (
            args.output_dir
            / f"short_{i:02d}_final.mp4"
        )

        print(f"Short #{i:02d}")
        print(f"  video:  {video}")
        print(f"  timing: {timing}")
        print(f"  output: {output}")

        cmd = [
            "python",
            args.renderer,
            "--video",
            str(video),
            "--timing",
            str(timing),
            "--style",
            args.style,
            "--output",
            str(output),
        ]

        result = subprocess.run(cmd)

        if result.returncode != 0:
            raise SystemExit(
                f"Subtitle renderer failed for Short #{i:02d}"
            )

        width, height, duration = probe(output)

        if width != 1080 or height != 1920:
            raise SystemExit(
                f"Wrong output dimensions for Short #{i:02d}: "
                f"{width}x{height}"
            )

        manifest["rendered_count"] += 1
        manifest["outputs"].append(
            {
                "rank": i,
                "camera_video": str(video),
                "subtitle_timing": str(timing),
                "output": str(output),
                "duration": duration,
                "width": width,
                "height": height,
            }
        )

        print(
            f"  PASS: {width}x{height}, "
            f"{duration:.3f}s"
        )
        print()

    manifest_path = (
        args.output_dir / "manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 72)
    print("FINAL SHORTS COMPLETE")
    print("=" * 72)
    print(
        f"Passed: "
        f"{manifest['rendered_count']}/{len(videos)}"
    )
    print(f"Manifest: {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
