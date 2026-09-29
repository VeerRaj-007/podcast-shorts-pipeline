import json
from pathlib import Path


EDITORIAL = Path("tmp/new_inde_editorial_v24_top3.json")
SHOTS = Path("tmp/new_inde_shots.json")
OUTPUT_DIR = Path("tmp")


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


editorial = load(EDITORIAL)
shot_data = load(SHOTS)
shots = shot_data["shots"]


for short_rank, item in enumerate(editorial["selected"], start=1):

    start = float(item["start"])
    end = start + float(item["duration"])

    overlapping = [
        shot
        for shot in shots
        if float(shot["end"]) > start
        and float(shot["start"]) < end
    ]

    if not overlapping:
        raise RuntimeError(
            f"No shots found for short {short_rank}"
        )

    actions = []

    local_time = 0.0

    for part_index, shot in enumerate(overlapping):

        source_in = max(
            start,
            float(shot["start"]),
        )

        source_out = min(
            end,
            float(shot["end"]),
        )

        if source_out <= source_in:
            continue

        duration = source_out - source_in

        action = {
            "id": (
                f"short_{short_rank:02d}"
                f"_part_{part_index:02d}"
            ),

            "action_index": part_index,

            "short_rank": short_rank,

            "kind": "base",

            "shot_id": shot["shot_id"],

            "render_segments": [
                {
                    "output_start": local_time,
                    "output_end": local_time + duration,

                    "source_in": source_in,
                    "source_out": source_out,
                }
            ],

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
                "duration": float(item["duration"]),
            },
        }

        actions.append(action)

        local_time += duration

    expected_duration = end - start

    if abs(local_time - expected_duration) > 0.01:
        raise RuntimeError(
            f"Short {short_rank} duration mismatch: "
            f"{local_time:.6f} != {expected_duration:.6f}"
        )

    output = {
        "version": "2.0",
        "mode": "groq_v24_top3_short",
        "short_rank": short_rank,

        "source_editorial": str(
            EDITORIAL
        ),

        "source_shots": str(
            SHOTS
        ),

        "editorial_lock": {
            "start": start,
            "end": end,
            "duration": expected_duration,
            "locked": True,
        },

        "video": {
            "start": 0.0,
            "end": float(shot_data["end"]),
            "width": shot_data["width"],
            "height": shot_data["height"],
            "fps": shot_data["fps"],
        },

        "actions": actions,

        "policy": {
            "editorial_boundaries": "locked",
            "shot_boundaries": "preserved",
            "resolver_may_change_editorial_start": False,
            "resolver_may_change_editorial_end": False,
        },
    }

    output_path = (
        OUTPUT_DIR
        / f"new_inde_short{short_rank:02d}_sequence_v2.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"SHORT {short_rank}: "
        f"{start:.3f} -> {end:.3f} "
        f"| {expected_duration:.3f}s "
        f"| {len(actions)} parts "
        f"| {output_path}"
    )
