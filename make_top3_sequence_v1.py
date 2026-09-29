import json
from pathlib import Path

EDITORIAL = Path("tmp/new_inde_editorial_v24_top3.json")
SHOTS = Path("tmp/new_inde_shots.json")
OUTPUT = Path("tmp/new_inde_top3_sequence_v1.json")


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


editorial = load(EDITORIAL)
shot_data = load(SHOTS)
shots = shot_data["shots"]

selected = editorial["selected"]

actions = []
action_index = 0

for short_rank, item in enumerate(selected, start=1):
    start = float(item["start"])
    end = float(item["start"] + item["duration"])

    # Use actual source shot boundaries only.
    overlapping = [
        shot for shot in shots
        if float(shot["end"]) > start
        and float(shot["start"]) < end
    ]

    if not overlapping:
        raise RuntimeError(
            f"No source shots found for SHORT {short_rank}: "
            f"{start} -> {end}"
        )

    short_actions = []

    for shot in overlapping:
        shot_start = max(
            start,
            float(shot["start"]),
        )

        shot_end = min(
            end,
            float(shot["end"]),
        )

        if shot_end <= shot_start:
            continue

        action = {
            "id": f"short_{short_rank:02d}_part_{len(short_actions):02d}",
            "action_index": action_index,
            "short_rank": short_rank,
            "rank": short_rank,

            # These are the exact source ranges to render.
            "start": shot_start,
            "end": shot_end,
            "source_start": shot_start,
            "source_end": shot_end,

            "shot_id": shot["shot_id"],
            "shot_type": shot.get("shot_type"),

            "kind": "base",

            "editorial_lock": {
                "locked": True,
                "parent_start": start,
                "parent_end": end,
            },

            "editorial": {
                "text": item.get("text"),
                "event_type": item.get("event_type"),
                "story_type": item.get("story_type"),
                "score": item.get("score"),
                "v16_publishability_score": item.get(
                    "v16_publishability_score"
                ),
            },
        }

        short_actions.append(action)
        actions.append(action)
        action_index += 1

    # Safety check: the split pieces must reconstruct the
    # exact editorial interval.
    reconstructed = sum(
        a["end"] - a["start"]
        for a in short_actions
    )

    expected = end - start

    if abs(reconstructed - expected) > 0.01:
        raise RuntimeError(
            f"SHORT {short_rank} split mismatch: "
            f"{reconstructed:.6f} != {expected:.6f}"
        )

print()
print("=== TOP 3 SEQUENCE v1 ===")

for short_rank in range(1, len(selected) + 1):
    parts = [
        a for a in actions
        if a["short_rank"] == short_rank
    ]

    start = parts[0]["start"]
    end = parts[-1]["end"]

    print(
        f"SHORT {short_rank}: "
        f"{start:.3f} -> {end:.3f} "
        f"({end - start:.3f}s) "
        f"| {len(parts)} camera parts"
    )

output = {
    "version": "1.0",
    "mode": "groq_v24_top3_locked",
    "source_editorial": str(EDITORIAL),
    "source_shots": str(SHOTS),

    "video": {
        "start": 0.0,
        "end": float(shot_data["end"]),
        "width": shot_data["width"],
        "height": shot_data["height"],
        "fps": shot_data["fps"],
    },

    "shorts": [
        {
            "short_rank": i,
            "start": float(selected[i - 1]["start"]),
            "end": float(
                selected[i - 1]["start"]
                + selected[i - 1]["duration"]
            ),
            "duration": float(selected[i - 1]["duration"]),
        }
        for i in range(1, len(selected) + 1)
    ],

    "actions": actions,

    "policy": {
        "editorial_source_ranges": "locked",
        "shot_boundaries": "preserved",
        "source_reuse": "forbidden_within_short",
        "duration_modification": False,
    },
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with open(
    OUTPUT,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        output,
        f,
        indent=2,
        ensure_ascii=False,
    )

print()
print(f"output: {OUTPUT}")
print(f"actions: {len(actions)}")
