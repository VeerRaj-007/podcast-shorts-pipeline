from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resolve each Groq editorial moment independently."
    )
    parser.add_argument("--locked", required=True, type=Path)
    parser.add_argument("--shots", required=True, type=Path)
    parser.add_argument("--resolver", default="editorial_av_resolver_v1.py")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    locked = load_json(args.locked)

    moments = locked.get("moments")
    if not isinstance(moments, list) or not moments:
        raise SystemExit("No locked moments found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "version": "1.0",
        "engine": "resolve_groq_moments",
        "source_locked_sequence": str(args.locked),
        "source_shots": str(args.shots),
        "moment_count": len(moments),
        "resolved": [],
    }

    print("=" * 72)
    print("GROQ MOMENT RESOLVER")
    print("=" * 72)
    print(f"Independent moments: {len(moments)}")
    print()

    for index, moment in enumerate(moments, 1):
        title = str(moment.get("title", ""))

        actions = moment.get("actions")
        if not isinstance(actions, list) or not actions:
            raise SystemExit(
                f"Moment #{index} has no actions."
            )

        source_start = float(moment["source_start"])
        source_end = float(moment["source_end"])

        # One independent sequence per Short.
        sequence = {
            "version": "1.0",
            "sequence_type": "independent_short",
            "editorial_authority": "groq",
            "source_timing_locked": True,
            "moment_rank": index,
            "candidate_id": moment.get("candidate_id"),
            "title": title,
            "summary": moment.get("summary", ""),
            "moment_type": moment.get("moment_type", ""),
            "source_start": source_start,
            "source_end": source_end,
            "actions": actions,
            "timeline_actions": actions,
        }

        sequence_path = (
            args.output_dir
            / f"short_{index:02d}_locked_sequence.json"
        )

        resolved_path = (
            args.output_dir
            / f"short_{index:02d}_resolved_sequence.json"
        )

        save_json(sequence_path, sequence)

        cmd = [
            "python",
            args.resolver,
            "--sequence",
            str(sequence_path),
            "--shots",
            str(args.shots),
            "--output",
            str(resolved_path),
        ]

        print(
            f"Short #{index:02d} | "
            f"{source_start:.3f} -> {source_end:.3f} | "
            f"{source_end - source_start:.3f}s | {title}"
        )

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.stdout:
            # Keep resolver output visible without duplicating huge logs.
            lines = [
                line
                for line in result.stdout.splitlines()
                if line.strip()
            ]

            for line in lines[-8:]:
                print(f"  {line}")

        if result.returncode != 0:
            print(
                result.stderr,
                end="",
            )
            raise SystemExit(
                f"Resolver failed for Short #{index}."
            )

        resolved = load_json(resolved_path)

        if resolved.get("unresolved_actions", 0) != 0:
            raise SystemExit(
                f"Short #{index}: unresolved actions remain."
            )

        if not resolved.get("render_safe", False):
            raise SystemExit(
                f"Short #{index}: resolver did not mark sequence render_safe."
            )

        manifest["resolved"].append(
            {
                "rank": index,
                "candidate_id": moment.get("candidate_id"),
                "title": title,
                "source_start": source_start,
                "source_end": source_end,
                "duration": source_end - source_start,
                "sequence": str(sequence_path),
                "resolved": str(resolved_path),
            }
        )

        print("  STATUS: PASS")
        print()

    manifest_path = args.output_dir / "manifest.json"
    save_json(manifest_path, manifest)

    print("=" * 72)
    print("ALL MOMENTS RESOLVED")
    print("=" * 72)
    print(f"Passed: {len(manifest['resolved'])}/{len(moments)}")
    print(f"Manifest: {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
