from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render all camera plans into vertical Shorts."
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--camera-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--renderer",
        default="face_aware_camera_renderer_final.py",
    )
    parser.add_argument("--crf", default=None)
    args = parser.parse_args()

    plans = sorted(
        args.camera_dir.glob("short_*_camera_plan.json")
    )

    if not plans:
        raise SystemExit(
            f"No camera plans found in {args.camera_dir}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "version": "1.0",
        "source_video": str(args.video),
        "rendered_count": 0,
        "outputs": [],
    }

    print("=" * 72)
    print("BATCH CAMERA RENDER")
    print("=" * 72)
    print(f"Plans: {len(plans)}")
    print()

    for i, plan in enumerate(plans, 1):
        output = (
            args.output_dir
            / f"short_{i:02d}_camera.mp4"
        )

        print(f"Short #{i:02d}")
        print(f"  plan:   {plan}")
        print(f"  output: {output}")

        cmd = [
            "python",
            args.renderer,
            "--video",
            str(args.video),
            "--camera",
            str(plan),
            "--output",
            str(output),
        ]

        if args.crf is not None:
            cmd += ["--crf", args.crf]

        result = subprocess.run(cmd)

        if result.returncode != 0:
            raise SystemExit(
                f"Camera renderer failed for Short #{i:02d}"
            )

        # Basic media verification.
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                str(output),
            ],
            capture_output=True,
            text=True,
        )

        if probe.returncode != 0:
            raise SystemExit(
                f"ffprobe failed for Short #{i:02d}"
            )

        media = json.loads(probe.stdout)

        streams = media.get("streams", [])
        video_stream = next(
            (
                s for s in streams
                if s.get("width") is not None
            ),
            None,
        )

        duration = float(
            media["format"]["duration"]
        )

        if not video_stream:
            raise SystemExit(
                f"No video stream in Short #{i:02d}"
            )

        width = int(video_stream["width"])
        height = int(video_stream["height"])

        if width != 1080 or height != 1920:
            raise SystemExit(
                f"Wrong dimensions for Short #{i:02d}: "
                f"{width}x{height}"
            )

        manifest["rendered_count"] += 1
        manifest["outputs"].append(
            {
                "rank": i,
                "camera_plan": str(plan),
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
    print("ALL CAMERA SHORTS RENDERED")
    print("=" * 72)
    print(
        f"Passed: "
        f"{manifest['rendered_count']}/{len(plans)}"
    )
    print(f"Manifest: {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
