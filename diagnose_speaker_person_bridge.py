import json
from collections import defaultdict
from pathlib import Path

SPEAKER_FILE = Path("tmp/interview_speaker_tracks.json")
VISUAL_FILE = Path("tmp/new_inde_visual_intelligence_v7.json")

with SPEAKER_FILE.open(encoding="utf-8") as f:
    speaker_data = json.load(f)

with VISUAL_FILE.open(encoding="utf-8") as f:
    visual_data = json.load(f)

speaker_tracks = speaker_data.get("tracks", [])
observations = visual_data.get(
    "identity_clean_observations",
    [],
)

speakers = sorted({
    str(x["speaker"])
    for x in speaker_tracks
    if x.get("speaker")
})

persons = sorted({
    str(x["identity_id"])
    for x in observations
    if x.get("identity_id")
})

print("=" * 90)
print("SPEAKER -> VISUAL PERSON DIAGNOSTIC")
print("=" * 90)
print()
print("Speakers:", speakers)
print("Persons :", persons)
print()

# ------------------------------------------------------------
# Build speaker intervals
# ------------------------------------------------------------

speaker_intervals = defaultdict(list)

for track in speaker_tracks:
    speaker = track.get("speaker")

    if not speaker:
        continue

    start = float(track["start"])
    end = float(track["end"])

    if end > start:
        speaker_intervals[speaker].append(
            (start, end)
        )


# ------------------------------------------------------------
# Build visual observations by person
# ------------------------------------------------------------

person_observations = defaultdict(list)

for obs in observations:
    person = obs.get("identity_id")
    timestamp = obs.get("timestamp")

    if person is None or timestamp is None:
        continue

    person_observations[person].append(
        obs
    )


def overlap(a0, a1, b0, b1):
    return max(
        0.0,
        min(a1, b1) - max(a0, b0),
    )


def speaker_active_at(
    intervals,
    t,
):
    for start, end in intervals:
        if start <= t <= end:
            return True
    return False


def nearest_person_observations(
    person_obs,
    t,
    tolerance=0.35,
):
    return [
        obs
        for obs in person_obs
        if abs(
            float(obs["timestamp"]) - t
        ) <= tolerance
    ]


# ------------------------------------------------------------
# Evaluate temporal evidence
# ------------------------------------------------------------

print("PAIR SCORES")
print("-" * 90)

results = []

for speaker in speakers:
    intervals = speaker_intervals[speaker]

    speaker_duration = sum(
        end - start
        for start, end in intervals
    )

    for person in persons:
        person_obs = person_observations[
            person
        ]

        visible_samples = 0
        visible_seconds_est = 0.0

        # Sample each speaker interval at 0.25s.
        for start, end in intervals:
            t = start

            while t < end:
                if nearest_person_observations(
                    person_obs,
                    t,
                ):
                    visible_samples += 1
                    visible_seconds_est += 0.25

                t += 0.25

        # Stronger evidence:
        # moments where exactly ONE of the two people
        # is visible while this speaker is talking.
        exclusive_samples = 0
        total_samples = 0

        other_persons = [
            p for p in persons
            if p != person
        ]

        for start, end in intervals:
            t = start

            while t < end:
                total_samples += 1

                current_visible = bool(
                    nearest_person_observations(
                        person_obs,
                        t,
                    )
                )

                other_visible = any(
                    nearest_person_observations(
                        person_observations[op],
                        t,
                    )
                    for op in other_persons
                )

                if current_visible and not other_visible:
                    exclusive_samples += 1

                t += 0.25

        visibility_rate = (
            visible_seconds_est / speaker_duration
            if speaker_duration > 0
            else 0.0
        )

        exclusive_rate = (
            exclusive_samples / total_samples
            if total_samples > 0
            else 0.0
        )

        results.append({
            "speaker": speaker,
            "person": person,
            "speaker_duration": speaker_duration,
            "visible_seconds_est": visible_seconds_est,
            "visibility_rate": visibility_rate,
            "exclusive_rate": exclusive_rate,
        })

for r in sorted(
    results,
    key=lambda x: (
        x["speaker"],
        -x["exclusive_rate"],
        -x["visibility_rate"],
    ),
):
    print(
        f'{r["speaker"]:12s} -> {r["person"]:12s} | '
        f'visibility={r["visibility_rate"]:.3f} | '
        f'exclusive={r["exclusive_rate"]:.3f} | '
        f'visible_sec≈{r["visible_seconds_est"]:.1f} | '
        f'speech={r["speaker_duration"]:.1f}s'
    )

# ------------------------------------------------------------
# Check whether PERSON_0 / PERSON_1 are actually the only
# identities covering the important interview regions.
# ------------------------------------------------------------

print()
print("PERSON TIMELINES")
print("-" * 90)

for person in persons:
    obs = person_observations[person]

    times = [
        float(x["timestamp"])
        for x in obs
    ]

    if times:
        print(
            f"{person}: "
            f"{min(times):.3f} -> {max(times):.3f} "
            f"({len(times)} observations)"
        )

print()
print("=" * 90)
print("IMPORTANT")
print("=" * 90)
print(
    "This is evidence only. It does NOT automatically declare a mapping."
)
print(
    "We will use the evidence to determine whether a speaker->person "
    "mapping is reliable enough for camera framing."
)
