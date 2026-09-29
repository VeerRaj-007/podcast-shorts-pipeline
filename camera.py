import json
from pathlib import Path


MOMENTS_PATH = Path("tmp/moments.json")
OUTPUT_PATH = Path("tmp/camera_plan.json")


BIN_SIZE = 0.5

# ---------------------------------------------------------
# Reaction takeover tuning
# ---------------------------------------------------------

# Minimum intensity required for a fullscreen reaction.
EVENT_THRESHOLD = 0.55

# How much stronger the peak must be than its surroundings.
MIN_PROMINENCE = 0.06

# Don't create another takeover too close to the previous one.
MIN_EVENT_GAP = 6.0

# Start slightly before the actual peak.
PRE_EVENT_SECONDS = 0.6

# Give the reaction some time after the peak.
POST_EVENT_SECONDS = 1.0

# Minimum fullscreen duration.
MIN_HOLD_SECONDS = 1.8

# Maximum fullscreen duration.
MAX_HOLD_SECONDS = 4.5


def clamp(value, low, high):
    return max(low, min(high, value))


def load_moments():
    with MOMENTS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_timeline(data):
    timeline = data.get("timeline", [])

    if not timeline:
        raise RuntimeError(
            "moments.json does not contain a timeline."
        )

    return timeline


def get_intensity(point):
    return float(point.get("intensity", 0.0))


def is_local_peak(timeline, index, radius=3):
    current = get_intensity(timeline[index])

    start = max(0, index - radius)
    end = min(len(timeline), index + radius + 1)

    for i in range(start, end):
        if i == index:
            continue

        if get_intensity(timeline[i]) > current:
            return False

    return True


def calculate_prominence(timeline, index, radius=8):
    peak = get_intensity(timeline[index])

    left_start = max(0, index - radius)
    left_end = index

    right_start = index + 1
    right_end = min(len(timeline), index + radius + 1)

    surrounding = []

    for i in range(left_start, left_end):
        surrounding.append(get_intensity(timeline[i]))

    for i in range(right_start, right_end):
        surrounding.append(get_intensity(timeline[i]))

    if not surrounding:
        return peak

    baseline = sum(surrounding) / len(surrounding)

    return peak - baseline


def find_candidates(timeline):
    candidates = []

    for i, point in enumerate(timeline):

        intensity = get_intensity(point)

        if intensity < EVENT_THRESHOLD:
            continue

        if not is_local_peak(timeline, i):
            continue

        prominence = calculate_prominence(
            timeline,
            i,
        )

        if prominence < MIN_PROMINENCE:
            continue

        candidates.append(
            {
                "index": i,
                "time": float(point["start"]),
                "intensity": intensity,
                "prominence": prominence,
            }
        )

    return candidates


def suppress_nearby_candidates(candidates):
    candidates = sorted(
        candidates,
        key=lambda x: x["intensity"],
        reverse=True,
    )

    selected = []

    for candidate in candidates:

        too_close = False

        for existing in selected:

            if abs(
                candidate["time"] - existing["time"]
            ) < MIN_EVENT_GAP:

                too_close = True
                break

        if not too_close:
            selected.append(candidate)

    return sorted(
        selected,
        key=lambda x: x["time"],
    )


def estimate_hold_duration(
    timeline,
    peak_index,
):
    peak = get_intensity(
        timeline[peak_index]
    )

    threshold = max(
        EVENT_THRESHOLD * 0.82,
        peak * 0.68,
    )

    end_index = peak_index

    while end_index + 1 < len(timeline):

        next_intensity = get_intensity(
            timeline[end_index + 1]
        )

        if next_intensity < threshold:
            break

        end_index += 1

    duration = (
        end_index - peak_index + 1
    ) * BIN_SIZE

    return clamp(
        duration + POST_EVENT_SECONDS,
        MIN_HOLD_SECONDS,
        MAX_HOLD_SECONDS,
    )


def build_events(timeline, candidates):

    events = []

    for candidate in candidates:

        peak_index = candidate["index"]
        peak_time = candidate["time"]

        hold_duration = estimate_hold_duration(
            timeline,
            peak_index,
        )

        start = max(
            0.0,
            peak_time - PRE_EVENT_SECONDS,
        )

        hold_start = peak_time

        hold_end = (
            peak_time + hold_duration
        )

        end = hold_end

        events.append(
            {
                "type": "fullscreen_reaction",

                "start": round(start, 3),

                "attack_end": round(
                    hold_start,
                    3,
                ),

                "hold_start": round(
                    hold_start,
                    3,
                ),

                "hold_end": round(
                    hold_end,
                    3,
                ),

                "end": round(
                    end,
                    3,
                ),

                "peak_time": round(
                    peak_time,
                    3,
                ),

                "intensity": round(
                    candidate["intensity"],
                    4,
                ),

                "prominence": round(
                    candidate["prominence"],
                    4,
                ),

                "target": "webcam",
            }
        )

    return events


def build_plan(data):

    timeline = get_timeline(data)

    print(
        f"Timeline points: {len(timeline)}"
    )

    candidates = find_candidates(
        timeline
    )

    print(
        f"Raw reaction candidates: "
        f"{len(candidates)}"
    )

    candidates = suppress_nearby_candidates(
        candidates
    )

    print(
        f"Selected reaction events: "
        f"{len(candidates)}"
    )

    events = build_events(
        timeline,
        candidates,
    )

    for i, event in enumerate(events[:20], 1):

        print(
            f"#{i:02d} "
            f"{event['start']:.1f}s -> "
            f"{event['end']:.1f}s | "
            f"peak={event['peak_time']:.1f}s | "
            f"intensity={event['intensity']:.2f}"
        )

    return {
        "version": 2,

        "mode": "event_camera",

        "timeline": timeline,

        "events": events,
    }


def main():

    data = load_moments()

    plan = build_plan(data)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            plan,
            f,
            indent=2,
        )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()