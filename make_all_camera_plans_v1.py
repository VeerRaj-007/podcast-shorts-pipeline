from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build camera plans for every resolved Groq Short."
    )
    parser.add_argument("--resolved-dir", required=True, type=Path)
    parser.add_argument("--visual", required=True, type=Path)
    parser.add_argument("--speakers", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--camera-director",
        default="face_aware_camera_director_final.py",
    )
    args = parser.parse_args()

    resolved_files = sorted(
        args.resolved_dir.glob("short_*_resolved_sequence.json")
    )

    if not resolved_files:
        raise SystemExit(
            f"No resolved sequences found in {args.resolved_dir}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "version": "1.0",
        "resolved_count": len(resolved_files),
        "camera_plans": [],
    }

    print("=" * 72)
    print("BATCH FACE-AWARE CAMERA DIRECTOR")
    print("=" * 72)
    print(f"Shorts: {len(resolved_files)}")
    print()

    for i, sequence in enumerate(resolved_files, 1):
        output = (
            args.output_dir
            / f"short_{i:02d}_camera_plan.json"
        )

        print(f"Short #{i:02d}")
        print(f"  sequence: {sequence}")
        print(f"  output:   {output}")

        cmd = [
            "python",
            args.camera_director,
            "--sequence",
            str(sequence),
            "--visual",
            str(args.visual),
            "--speakers",
            str(args.speakers),
            "--output",
            str(output),
        ]

        result = subprocess.run(cmd, text=True)

        if result.returncode != 0:
            raise SystemExit(
                f"Camera director failed for Short #{i:02d}"
            )

        data = json.loads(
            output.read_text(encoding="utf-8")
        )

        segments = data.get("segments", [])
        duration = data.get("output", {}).get(
            "duration",
            data.get("video", {}).get("duration"),
        )

        manifest["camera_plans"].append(
            {
                "rank": i,
                "sequence": str(sequence),
                "camera_plan": str(output),
                "segment_count": len(segments),
                "duration": duration,
            }
        )

        print("  STATUS: PASS")
        print()

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 72)
    print("ALL CAMERA PLANS CREATED")
    print("=" * 72)
    print(f"Passed: {len(manifest['camera_plans'])}/{len(resolved_files)}")
    print(f"Manifest: {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
