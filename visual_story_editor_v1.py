#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


EPS = 1e-3


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


def clamp(value, lo=0.0, hi=1.0):
    return max(lo, min(hi, value))


def shot_has_face(shot):
    return (
        float(shot.get("max_face_count", 0) or 0) > 0
        or float(shot.get("face_coverage_max", 0) or 0) > 0
    )


def shot_midpoint(shot):
    return (
        float(shot["start"])
        + float(shot["end"])
    ) / 2.0


def find_shot_for_time(shots, time_value):
    candidates = []

    for shot in shots:
        start = float(shot["start"])
        end = float(shot["end"])

        if (
            start - EPS
            <= time_value
            <= end + EPS
        ):
            candidates.append(shot)

    if not candidates:
        return None

    # Prefer the narrowest containing shot.
    candidates.sort(
        key=lambda s: (
            float(s["end"]) - float(s["start"]),
            float(s["start"]),
        )
    )

    return candidates[0]


def build_coverage(action, shots):
    """
    Convert AV source segments into visual coverage intervals.

    This is intentionally conservative:
    - editorial start/end remain unchanged
    - audio source ranges remain unchanged
    - visual source ranges exactly equal the audio source ranges
      for speaker-focused coverage
    """

    segments = action.get(
        "render_segments",
        [],
    )

    visual_segments = []

    for index, segment in enumerate(segments):
        source_start = float(
            segment["source_in"]
        )
        source_end = float(
            segment["source_out"]
        )

        shot = find_shot_for_time(
            shots,
            (
                source_start
                + source_end
            ) / 2.0,
        )

        if shot is None:
            raise ValueError(
                "Could not resolve visual coverage for "
                f"{action.get('id')} "
                f"{source_start} -> {source_end}"
            )

        has_face = shot_has_face(shot)

        # Conservative v1:
        # If the source has a detectable person, use speaker_focus.
        # Otherwise preserve the source as a supporting visual.
        if has_face:
            mode = "speaker_focus"
        else:
            mode = "source_visual"

        visual_segments.append(
            {
                "index": index,
                "output_start": float(
                    segment["output_start"]
                ),
                "output_end": float(
                    segment["output_end"]
                ),

                # HARD SYNC INVARIANT.
                "audio_source_start": source_start,
                "audio_source_end": source_end,

                "visual_source_start": source_start,
                "visual_source_end": source_end,

                "mode": mode,

                "source_shot_id": shot.get(
                    "shot_id"
                ),

                "source_shot_type": shot.get(
                    "shot_type"
                ),

                "reason": (
                    "person_detected"
                    if mode == "speaker_focus"
                    else "no_detected_face"
                ),
            }
        )

    return visual_segments


def merge_adjacent_segments(segments):
    """
    Merge adjacent segments only when their visual decision is identical.

    We do NOT merge across different source ranges or different modes.
    """

    if not segments:
        return []

    merged = [dict(segments[0])]

    for current in segments[1:]:
        previous = merged[-1]

        same_mode = (
            current["mode"]
            == previous["mode"]
        )

        same_shot = (
            current["source_shot_id"]
            == previous["source_shot_id"]
        )

        adjacent_output = abs(
            float(current["output_start"])
            - float(previous["output_end"])
        ) <= EPS

        adjacent_source = abs(
            float(current["visual_source_start"])
            - float(previous["visual_source_end"])
        ) <= EPS

        if (
            same_mode
            and same_shot
            and adjacent_output
            and adjacent_source
        ):
            previous["output_end"] = (
                current["output_end"]
            )

            previous["audio_source_end"] = (
                current["audio_source_end"]
            )

            previous["visual_source_end"] = (
                current["visual_source_end"]
            )
        else:
            merged.append(dict(current))

    return merged


def validate_visual_action(action):
    start = float(action["start"])
    end = float(action["end"])

    segments = action.get(
        "visual_segments",
        [],
    )

    if not segments:
        return [
            "no_visual_segments"
        ]

    errors = []

    # Output coverage must begin at zero and end at exact duration.
    expected_duration = end - start

    if abs(
        float(segments[0]["output_start"])
    ) > EPS:
        errors.append(
            "visual_timeline_does_not_start_at_zero"
        )

    if abs(
        float(segments[-1]["output_end"])
        - expected_duration
    ) > EPS:
        errors.append(
            "visual_timeline_does_not_end_at_editorial_duration"
        )

    previous_output_end = 0.0

    for segment in segments:
        output_start = float(
            segment["output_start"]
        )
        output_end = float(
            segment["output_end"]
        )

        audio_start = float(
            segment["audio_source_start"]
        )
        audio_end = float(
            segment["audio_source_end"]
        )

        visual_start = float(
            segment["visual_source_start"]
        )
        visual_end = float(
            segment["visual_source_end"]
        )

        if (
            abs(output_start - previous_output_end)
            > EPS
        ):
            errors.append(
                "visual_output_gap_or_overlap"
            )

        if output_end <= output_start:
            errors.append(
                "invalid_visual_output_range"
            )

        # Speaker-focus audio/visual synchronization.
        if segment["mode"] == "speaker_focus":
            if abs(
                audio_start - visual_start
            ) > EPS:
                errors.append(
                    "speaker_focus_source_start_not_synced"
                )

            if abs(
                audio_end - visual_end
            ) > EPS:
                errors.append(
                    "speaker_focus_source_end_not_synced"
                )

        if visual_end <= visual_start:
            errors.append(
                "invalid_visual_source_range"
            )

        previous_output_end = output_end

    return sorted(set(errors))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a conservative visual story plan from "
            "an exact AV editorial plan."
        )
    )

    parser.add_argument(
        "--av",
        required=True,
    )

    parser.add_argument(
        "--shots",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    av = load_json(
        args.av
    )

    shot_data = load_json(
        args.shots
    )

    shots = (
        shot_data.get("shots", [])
        if isinstance(shot_data, dict)
        else shot_data
    )

    if not isinstance(shots, list):
        raise ValueError(
            "Invalid shot inventory."
        )

    actions = av.get(
        "actions",
        []
    )

    if not isinstance(actions, list):
        raise ValueError(
            "AV plan does not contain actions."
        )

    visual_actions = []
    validation_errors = []

    for action in actions:
        visual_segments = build_coverage(
            action,
            shots,
        )

        visual_segments = (
            merge_adjacent_segments(
                visual_segments
            )
        )

        visual_action = dict(action)

        visual_action[
            "visual_segments"
        ] = visual_segments

        # Preserve editorial boundaries exactly.
        visual_action[
            "editorial_source_match"
        ] = action.get(
            "editorial_source_match"
        )

        errors = validate_visual_action(
            visual_action
        )

        visual_action[
            "visual_render_safe"
        ] = not errors

        if errors:
            validation_errors.append(
                {
                    "id": action.get("id"),
                    "errors": errors,
                }
            )

        visual_actions.append(
            visual_action
        )

    output = {
        "version": "1.0",
        "engine": "visual_story_editor",

        "mode": (
            "conservative_exact_source_visual_direction"
        ),

        "source_av_plan": {
            "version": av.get("version"),
            "mode": av.get("mode"),
            "render_safe": av.get(
                "render_safe"
            ),
        },

        "video_start": av.get(
            "video_start",
            0.0,
        ),

        "video_end": av.get(
            "video_end"
        ),

        "actions": visual_actions,

        "validation_errors": (
            validation_errors
        ),

        "render_safe": (
            bool(av.get("render_safe"))
            and not validation_errors
        ),

        "policy": {
            "editorial_boundaries": (
                "immutable"
            ),

            "audio_source_ranges": (
                "immutable"
            ),

            "speaker_focus": {
                "audio_visual_source_sync": (
                    "required"
                ),
                "maximum_source_offset_seconds": 0.001,
            },

            "source_frame_invention": (
                "forbidden"
            ),

            "source_time_stretch": (
                "forbidden"
            ),

            "source_looping": (
                "forbidden"
            ),

            "silent_editorial_changes": (
                "forbidden"
            ),

            "visual_mode_v1": (
                "conservative"
            ),
        },

        "stats": {
            "input_actions": len(actions),
            "visual_actions": len(
                visual_actions
            ),
            "validation_error_count": len(
                validation_errors
            ),
            "visual_segments": sum(
                len(
                    a.get(
                        "visual_segments",
                        [],
                    )
                )
                for a in visual_actions
            ),
        },
    }

    save_json(
        args.output,
        output,
    )

    print("=" * 72)
    print("VISUAL STORY EDITOR v1")
    print("=" * 72)
    print()
    print(
        "Input actions:",
        len(actions),
    )
    print(
        "Visual actions:",
        len(visual_actions),
    )
    print(
        "Visual segments:",
        output["stats"]["visual_segments"],
    )
    print(
        "Validation errors:",
        len(validation_errors),
    )
    print(
        "Render safe:",
        output["render_safe"],
    )
    print()
    print(
        "Output:",
        args.output,
    )

    if not output["render_safe"]:
        raise SystemExit(
            "ERROR: visual plan failed validation."
        )

    print()
    print(
        "SUCCESS: visual plan preserved "
        "editorial and A/V source boundaries exactly."
    )


if __name__ == "__main__":
    main()