#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


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


def main():
    parser = argparse.ArgumentParser(
        description="Convert Groq editorial decisions into resolver sequence JSON."
    )

    parser.add_argument(
        "--editorial",
        required=True,
    )

    parser.add_argument(
        "--video-end",
        required=True,
        type=float,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    editorial = load_json(
        args.editorial
    )

    clips = editorial.get(
        "clips",
        []
    )

    if not isinstance(clips, list):
        raise ValueError(
            "Editorial JSON does not contain a valid clips list."
        )

    actions = []

    for rank, clip in enumerate(clips, start=1):
        start = float(
            clip["start"]
        )

        end = float(
            clip["end"]
        )

        if start < 0:
            raise ValueError(
                f"Clip {rank} has negative start: {start}"
            )

        if end <= start:
            raise ValueError(
                f"Clip {rank} has invalid interval: "
                f"{start} -> {end}"
            )

        if end > args.video_end:
            raise ValueError(
                f"Clip {rank} exceeds video end: "
                f"{end} > {args.video_end}"
            )

        # Every Groq-selected editorial moment becomes one
        # resolver action. We deliberately do NOT ask the resolver
        # to invent a different editorial interval.
        actions.append(
            {
                "id": f"short_{rank:02d}",
                "action_index": rank - 1,
                "rank": rank,

                # Exact editorial decision from Groq.
                "start": start,
                "end": end,

                "kind": "base",

                "editorial": {
                    "candidate_id": clip.get(
                        "candidate_id"
                    ),
                    "title": clip.get(
                        "title"
                    ),
                    "moment_type": clip.get(
                        "moment_type"
                    ),
                    "summary": clip.get(
                        "summary"
                    ),
                    "start_reason": clip.get(
                        "start_reason"
                    ),
                    "end_reason": clip.get(
                        "end_reason"
                    ),
                    "complete": clip.get(
                        "complete"
                    ),
                    "quality_score": clip.get(
                        "quality_score"
                    ),
                    "confidence": clip.get(
                        "confidence"
                    ),
                    "boundary_confidence": clip.get(
                        "boundary_confidence"
                    ),
                },

                # Explicitly tell downstream systems that these are
                # editorially locked boundaries.
                "editorial_lock": {
                    "start": start,
                    "end": end,
                    "locked": True,
                },
            }
        )

    output = {
        "version": "1.0",
        "mode": "groq_editorial_locked",
        "video_start": 0.0,
        "video_end": args.video_end,

        "source_editorial": {
            "engine": editorial.get(
                "engine"
            ),
            "version": editorial.get(
                "version"
            ),
            "model": editorial.get(
                "model"
            ),
            "requested_count": editorial.get(
                "requested_count"
            ),
            "accepted_candidates": editorial.get(
                "accepted_candidates"
            ),
            "selected_count": editorial.get(
                "selected_count"
            ),
            "quality_floor": editorial.get(
                "quality_floor"
            ),
            "confidence_floor": editorial.get(
                "confidence_floor"
            ),
        },

        "actions": actions,

        "policy": {
            "editorial_boundaries": (
                "locked_from_groq"
            ),
            "resolver_may_remap_source": True,
            "resolver_may_change_editorial_start": False,
            "resolver_may_change_editorial_end": False,
            "source_reuse": "forbidden",
            "dialogue_chronology": "monotonic",
        },
    }

    save_json(
        args.output,
        output,
    )

    print(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print(
        f"SUCCESS: converted {len(actions)} "
        "Groq editorial moments into resolver actions."
    )


if __name__ == "__main__":
    main()