from pathlib import Path

src = Path("face_aware_camera_director_v5.py")
out = Path("face_aware_camera_director_v5_1.py")

if not src.exists():
    raise FileNotFoundError(src)

text = src.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. VERSION
# ---------------------------------------------------------------------------

# The original file currently emits version "5.0".
if text.count('"version": "5.0"') != 1:
    raise RuntimeError(
        'Expected exactly one camera-director version "5.0".'
    )

text = text.replace(
    '"version": "5.0"',
    '"version": "5.1"',
    1,
)


# ---------------------------------------------------------------------------
# 2. ADD SPEAKER COVERAGE VALIDATOR
# ---------------------------------------------------------------------------

anchor = '''def choose_speaker(turns, by_person, start, end):
'''

helper = '''def speaker_coverage(turns, start, end):
    """
    Measure how much of [start, end] is covered by speaker turns.

    Returns:
        {
            "covered_seconds": float,
            "required_seconds": float,
            "coverage_ratio": float,
            "turn_count": int,
        }

    This is diagnostics/validation only. It does not alter source timing.
    """

    required = max(
        0.0,
        float(end) - float(start),
    )

    intervals = []

    for turn in turns:
        try:
            turn_start = float(turn["start"])
            turn_end = float(turn["end"])
        except (KeyError, TypeError, ValueError):
            continue

        overlap_start = max(
            float(start),
            turn_start,
        )

        overlap_end = min(
            float(end),
            turn_end,
        )

        if overlap_end > overlap_start:
            intervals.append(
                (overlap_start, overlap_end)
            )

    if not intervals:
        return {
            "covered_seconds": 0.0,
            "required_seconds": required,
            "coverage_ratio": 0.0 if required > 0 else 1.0,
            "turn_count": 0,
        }

    # Merge overlapping speaker intervals so overlapping diarization
    # turns do not inflate the coverage value.
    intervals.sort()

    merged = [intervals[0]]

    for current_start, current_end in intervals[1:]:
        previous_start, previous_end = merged[-1]

        if current_start <= previous_end + 1e-6:
            merged[-1] = (
                previous_start,
                max(
                    previous_end,
                    current_end,
                ),
            )
        else:
            merged.append(
                (current_start, current_end)
            )

    covered = sum(
        end_value - start_value
        for start_value, end_value in merged
    )

    ratio = (
        covered / required
        if required > 0
        else 1.0
    )

    return {
        "covered_seconds": covered,
        "required_seconds": required,
        "coverage_ratio": ratio,
        "turn_count": len(intervals),
    }


def choose_speaker(turns, by_person, start, end):
'''

if text.count(anchor) != 1:
    raise RuntimeError(
        "Could not find choose_speaker() anchor."
    )

text = text.replace(
    anchor,
    helper,
    1,
)


# ---------------------------------------------------------------------------
# 3. ADD REQUIRED CLI FALLBACK FLAG
# ---------------------------------------------------------------------------

old = '''    parser.add_argument("--sequence", required=True)
    parser.add_argument("--visual", required=True)
    parser.add_argument("--speakers", required=True)
    parser.add_argument("--output", required=True)
'''

new = '''    parser.add_argument("--sequence", required=True)
    parser.add_argument("--visual", required=True)
    parser.add_argument("--speakers", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument(
        "--allow-two-person-fallback",
        action="store_true",
        help=(
            "Allow centered two-person framing when speaker-turn "
            "coverage is unavailable. Disabled by default."
        ),
    )
'''

if text.count(old) != 1:
    raise RuntimeError(
        "Could not find CLI argument block."
    )

text = text.replace(
    old,
    new,
    1,
)


# ---------------------------------------------------------------------------
# 4. REPLACE CAMERA SEGMENT DECISION BLOCK
# ---------------------------------------------------------------------------

old = '''        focus, speaker_seconds, focus_share = choose_speaker(
            turns,
            by_person,
            start,
            end,
        )

        if (
            focus
            and by_person.get(focus)
        ):
            crop = centered_speaker_crop(
                by_person[focus]
            )

            composition = "centered_speaker_focus"
            motion = "hold_centered"

        else:
            crop = centered_two_person_crop(
                by_person["PERSON_0"],
                by_person["PERSON_1"],
            )

            composition = "centered_two_person"
            motion = "hold"
            focus = None
            focus_share = 0.0
'''

new = '''        coverage = speaker_coverage(
            turns,
            start,
            end,
        )

        focus, speaker_seconds, focus_share = choose_speaker(
            turns,
            by_person,
            start,
            end,
        )

        # Production safety:
        # never silently hide missing speaker attribution.
        #
        # If speaker turns do not cover this source interval, fail unless
        # the caller explicitly opted into two-person fallback.
        if coverage["coverage_ratio"] < 0.98:
            if not args.allow_two_person_fallback:
                raise RuntimeError(
                    "Insufficient speaker-turn coverage for "
                    f"{start:.3f} -> {end:.3f}: "
                    f"{coverage['covered_seconds']:.3f}s / "
                    f"{coverage['required_seconds']:.3f}s "
                    f"({coverage['coverage_ratio']:.3f}). "
                    "Provide a full-video speaker attribution file."
                )

            crop = centered_two_person_crop(
                by_person["PERSON_0"],
                by_person["PERSON_1"],
            )

            composition = "centered_two_person"
            motion = "hold"
            focus = None
            focus_share = 0.0

        elif (
            focus
            and by_person.get(focus)
        ):
            crop = centered_speaker_crop(
                by_person[focus]
            )

            composition = "centered_speaker_focus"
            motion = "hold_centered"

        else:
            if not args.allow_two_person_fallback:
                raise RuntimeError(
                    "Speaker turns cover "
                    f"{start:.3f} -> {end:.3f}, but no "
                    "speaker with a visible face could be selected. "
                    "Refusing silent two-person fallback."
                )

            crop = centered_two_person_crop(
                by_person["PERSON_0"],
                by_person["PERSON_1"],
            )

            composition = "centered_two_person"
            motion = "hold"
            focus = None
            focus_share = 0.0
'''

if text.count(old) != 1:
    raise RuntimeError(
        "Could not find camera decision block."
    )

text = text.replace(
    old,
    new,
    1,
)


# ---------------------------------------------------------------------------
# 5. ADD COVERAGE TO CAMERA SEGMENT OUTPUT
# ---------------------------------------------------------------------------

old = '''            "focus_share": round(focus_share, 4),
            "crop": crop,
'''

new = '''            "focus_share": round(focus_share, 4),
            "speaker_coverage": coverage,
            "crop": crop,
'''

if text.count(old) != 1:
    raise RuntimeError(
        "Could not find camera segment metadata block."
    )

text = text.replace(
    old,
    new,
    1,
)


# ---------------------------------------------------------------------------
# 6. UPDATE OUTPUT CONSTRAINTS
# ---------------------------------------------------------------------------

old = '''            "speaker_fallback_threshold": MIN_SPEAKER_SHARE,
            "focus_zoom": FOCUS_ZOOM,
'''

new = '''            "speaker_fallback_threshold": MIN_SPEAKER_SHARE,
            "focus_zoom": FOCUS_ZOOM,
            "speaker_turn_coverage_required": 0.98,
            "silent_two_person_fallback": False,
'''

if text.count(old) != 1:
    raise RuntimeError(
        "Could not find output constraints block."
    )

text = text.replace(
    old,
    new,
    1,
)


# ---------------------------------------------------------------------------
# 7. UPDATE PRINTED VERSION
# ---------------------------------------------------------------------------

text = text.replace(
    "FACE-AWARE CAMERA DIRECTOR v5",
    "FACE-AWARE CAMERA DIRECTOR v5.1",
)


# ---------------------------------------------------------------------------
# 8. WRITE + COMPILE
# ---------------------------------------------------------------------------

out.write_text(
    text,
    encoding="utf-8",
)

compile(
    text,
    str(out),
    "exec",
)

print("=" * 72)
print("FACE-AWARE CAMERA DIRECTOR v5.1 CREATED")
print("=" * 72)
print()
print(f"Output: {out}")
print()
print("Changes:")
print("  - full speaker-turn coverage validation")
print("  - silent fallback disabled")
print("  - explicit --allow-two-person-fallback flag")
print("  - speaker coverage diagnostics")
print("  - source timestamps remain immutable")
print("  - existing crop algorithms unchanged")
print()
print("Syntax check: PASSED")