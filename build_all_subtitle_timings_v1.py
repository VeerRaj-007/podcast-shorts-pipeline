from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build subtitle timing files for all locked Shorts."
    )
    parser.add_argument("--locked", required=True, type=Path)
    parser.add_argument("--transcript", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--builder",
        default="subtitle_timing_builder_v1.py",
    )
    args = parser.parse_args()

    locked = load_json(args.locked)
    moments = locked.get("moments")

    if not isinstance(moments, list) or not moments:
        raise SystemExit("No locked moments found.")

    if not args.transcript.exists():
        raise SystemExit(f"Transcript source not found: {args.transcript}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "version": "1.0",
        "source_locked_sequence": str(args.locked),
        "subtitle_source": str(args.transcript),
        "timing_count": 0,
        "timings": [],
    }

    print("=" * 72)
    print("BATCH SUBTITLE TIMING")
    print("=" * 72)
    print(f"Shorts: {len(moments)}")
    print()

    for i, moment in enumerate(moments, 1):
        start = float(moment["source_start"])
        end = float(moment["source_end"])

        output = (
            args.output_dir
            / f"short_{i:02d}_timing.json"
        )

        print(
            f"Short #{i:02d} | "
            f"{start:.3f} -> {end:.3f} | "
            f"{end - start:.3f}s"
        )

        cmd = [
            "python",
            args.builder,
            "--transcript",
            str(args.transcript),
            "--start",
            str(start),
            "--end",
            str(end),
            "--output",
            str(output),
        ]

        result = subprocess.run(cmd, text=True)

        if result.returncode != 0:
            raise SystemExit(
                f"Subtitle timing failed for Short #{i:02d}"
            )

        timing = load_json(output)

        if not isinstance(timing, dict):
            raise SystemExit(
                f"Invalid timing JSON for Short #{i:02d}"
            )

        manifest["timing_count"] += 1
        manifest["timings"].append(
            {
                "rank": i,
                "source_start": start,
                "source_end": end,
                "duration": end - start,
                "timing": str(output),
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
    print("ALL SUBTITLE TIMINGS CREATED")
    print("=" * 72)
    print(
        f"Passed: {manifest['timing_count']}/{len(moments)}"
    )
    print(f"Manifest: {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
