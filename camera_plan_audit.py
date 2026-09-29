import argparse
import json
from pathlib import Path


EPS = 1e-3
MIN_FOCUS_SHARE = 0.40


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Audit face-aware camera director output."
    )

    parser.add_argument("--camera", required=True)
    parser.add_argument("--av", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    camera = load_json(args.camera)
    av = load_json(args.av)

    camera_segments = camera.get(
        "camera_segments",
        [],
    )

    av_actions = av.get(
        "actions",
        [],
    )

    errors = []
    warnings = []

    # ------------------------------------------------------------
    # 1. Camera segments exist
    # ------------------------------------------------------------

    if not camera_segments:
        errors.append("No camera segments found.")

    # ------------------------------------------------------------
    # 2. Build AV source map
    # ------------------------------------------------------------

    av_segments = []

    for action in av_actions:
        for segment in action.get(
            "render_segments",
            [],
        ):
            av_segments.append(
                {
                    "source_start": float(
                        segment["source_in"]
                    ),
                    "source_end": float(
                        segment["source_out"]
                    ),
                    "output_start": float(
                        segment["output_start"]
                    ),
                    "output_end": float(
                        segment["output_end"]
                    ),
                }
            )

    # ------------------------------------------------------------
    # 3. Camera source ranges must match AV source ranges
    # ------------------------------------------------------------

    for index, cam in enumerate(camera_segments, 1):
        c_start = float(
            cam["source_start"]
        )
        c_end = float(
            cam["source_end"]
        )

        matching = [
            s
            for s in av_segments
            if abs(
                s["source_start"] - c_start
            ) <= EPS
            and abs(
                s["source_end"] - c_end
            ) <= EPS
        ]

        if not matching:
            errors.append(
                f"camera segment {index}: "
                f"source range {c_start}->{c_end} "
                f"does not exactly match an AV segment."
            )

        # --------------------------------------------------------
        # 4. Speaker-focus quality
        # --------------------------------------------------------

        composition = cam.get(
            "composition"
        )

        focus_share = float(
            cam.get(
                "focus_share",
                0.0,
            )
        )

        target = cam.get(
            "target_identity"
        )

        if composition == "centered_speaker_focus":
            if not target:
                errors.append(
                    f"camera segment {index}: "
                    "speaker_focus has no target_identity."
                )

            if focus_share < MIN_FOCUS_SHARE:
                warnings.append(
                    f"camera segment {index}: "
                    f"speaker_focus has weak focus_share "
                    f"{focus_share:.3f}."
                )

        elif composition == "centered_two_person":
            if target is not None:
                errors.append(
                    f"camera segment {index}: "
                    "two-person composition unexpectedly "
                    f"has target={target}."
                )

        else:
            errors.append(
                f"camera segment {index}: "
                f"unknown composition={composition!r}."
            )

        # --------------------------------------------------------
        # 5. Crop validity
        # --------------------------------------------------------

        crop = cam.get("crop")

        if not isinstance(crop, dict):
            errors.append(
                f"camera segment {index}: missing crop."
            )
            continue

        for key in (
            "left",
            "top",
            "width",
            "height",
        ):
            if key not in crop:
                errors.append(
                    f"camera segment {index}: "
                    f"crop missing {key}."
                )

        try:
            left = float(crop["left"])
            top = float(crop["top"])
            width = float(crop["width"])
            height = float(crop["height"])

            if left < -EPS:
                errors.append(
                    f"camera segment {index}: crop left < 0."
                )

            if top < -EPS:
                errors.append(
                    f"camera segment {index}: crop top < 0."
                )

            if width <= 0:
                errors.append(
                    f"camera segment {index}: crop width <= 0."
                )

            if height <= 0:
                errors.append(
                    f"camera segment {index}: crop height <= 0."
                )

            if left + width > 1920 + EPS:
                errors.append(
                    f"camera segment {index}: crop exceeds source width."
                )

            if top + height > 1080 + EPS:
                errors.append(
                    f"camera segment {index}: crop exceeds source height."
                )

        except (KeyError, TypeError, ValueError):
            errors.append(
                f"camera segment {index}: invalid crop values."
            )

    # ------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------

    speaker_focus_count = sum(
        1
        for s in camera_segments
        if s.get("composition")
        == "centered_speaker_focus"
    )

    two_person_count = sum(
        1
        for s in camera_segments
        if s.get("composition")
        == "centered_two_person"
    )

    output = {
        "audit_version": "1.0",
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "camera_segments": len(camera_segments),
            "speaker_focus_segments": speaker_focus_count,
            "two_person_segments": two_person_count,
            "errors": len(errors),
            "warnings": len(warnings),
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 72)
    print("CAMERA PLAN AUDIT")
    print("=" * 72)
    print()
    print(
        "STATUS:",
        "PASS" if not errors else "FAIL",
    )
    print(
        "Camera segments:",
        len(camera_segments),
    )
    print(
        "Speaker-focus:",
        speaker_focus_count,
    )
    print(
        "Two-person:",
        two_person_count,
    )
    print(
        "Errors:",
        len(errors),
    )
    print(
        "Warnings:",
        len(warnings),
    )
    print()
    print(
        "Output:",
        output_path,
    )

    if errors:
        print()
        print("ERRORS")
        for error in errors:
            print(" -", error)

    if warnings:
        print()
        print("WARNINGS")
        for warning in warnings:
            print(" -", warning)

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()