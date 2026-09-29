import argparse
import json
from pathlib import Path


EPS = 1e-6


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


def overlap(a0, a1, b0, b1):
    return max(
        0.0,
        min(a1, b1) - max(a0, b0),
    )


def build_speaker_activity(diarization):
    segments = diarization.get(
        "segments",
        [],
    )

    if not isinstance(segments, list):
        raise ValueError(
            "Diarization does not contain a valid segments list."
        )

    result = []

    for segment in segments:
        start = float(segment["start"])
        end = float(segment["end"])
        speaker = str(segment["speaker"])

        if end <= start:
            continue

        result.append(
            {
                "start": start,
                "end": end,
                "speaker": speaker,
            }
        )

    return result


def identity_observations(visual):
    observations = visual.get(
        "identity_clean_observations",
        [],
    )

    if not isinstance(observations, list):
        raise ValueError(
            "Visual intelligence does not contain "
            "identity_clean_observations."
        )

    return observations


def observation_identity_score(
    observations,
    speaker_start,
    speaker_end,
):
    scores = {
        "PERSON_0": 0.0,
        "PERSON_1": 0.0,
    }

    counts = {
        "PERSON_0": 0,
        "PERSON_1": 0,
    }

    for obs in observations:
        person = obs.get("identity_id")
        timestamp = obs.get("timestamp")

        if person not in scores or timestamp is None:
            continue

        t = float(timestamp)

        if (
            t < speaker_start - 0.5
            or t > speaker_end + 0.5
        ):
            continue

        # Each observation represents visual evidence that this
        # identity is visible while the speaker is active.
        score = float(
            obs.get("score", 1.0) or 1.0
        )

        area = float(
            obs.get("area_normalized", 0.0) or 0.0
        )

        # Confidence-weighted evidence. We don't assume that
        # visual presence alone means "this person is speaking";
        # the global mapping is accumulated over many intervals.
        weight = score * (
            1.0 + min(area * 20.0, 1.0)
        )

        scores[person] += weight
        counts[person] += 1

    return scores, counts


def infer_global_mapping(
    speaker_segments,
    observations,
):
    speakers = sorted(
        {
            s["speaker"]
            for s in speaker_segments
        }
    )

    if len(speakers) != 2:
        raise ValueError(
            f"Expected exactly 2 speakers, found: {speakers}"
        )

    evidence = {}

    for speaker in speakers:
        total_scores = {
            "PERSON_0": 0.0,
            "PERSON_1": 0.0,
        }

        total_counts = {
            "PERSON_0": 0,
            "PERSON_1": 0,
        }

        for segment in speaker_segments:
            if segment["speaker"] != speaker:
                continue

            scores, counts = observation_identity_score(
                observations,
                segment["start"],
                segment["end"],
            )

            for person in total_scores:
                total_scores[person] += scores[person]
                total_counts[person] += counts[person]

        ranked = sorted(
            total_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        best_person, best_score = ranked[0]
        second_person, second_score = ranked[1]

        total = best_score + second_score

        confidence = (
            best_score / total
            if total > EPS
            else 0.0
        )

        margin = (
            (best_score - second_score) / total
            if total > EPS
            else 0.0
        )

        evidence[speaker] = {
            "scores": total_scores,
            "observation_counts": total_counts,
            "candidate_person": best_person,
            "confidence": confidence,
            "margin": margin,
        }

    # Evaluate both possible one-to-one assignments.
    s0, s1 = speakers

    direct_score = (
        evidence[s0]["scores"]["PERSON_0"]
        + evidence[s1]["scores"]["PERSON_1"]
    )

    swapped_score = (
        evidence[s0]["scores"]["PERSON_1"]
        + evidence[s1]["scores"]["PERSON_0"]
    )

    if direct_score >= swapped_score:
        mapping = {
            s0: "PERSON_0",
            s1: "PERSON_1",
        }
        mapping_score = direct_score
    else:
        mapping = {
            s0: "PERSON_1",
            s1: "PERSON_0",
        }
        mapping_score = swapped_score

    return mapping, evidence, mapping_score


def build_turns(
    speaker_segments,
    mapping,
):
    """
    Merge adjacent diarization segments belonging to the same
    mapped person when they touch or nearly touch.
    """

    ordered = sorted(
        speaker_segments,
        key=lambda x: (
            x["start"],
            x["end"],
        ),
    )

    turns = []

    for segment in ordered:
        person = mapping.get(
            segment["speaker"]
        )

        if person is None:
            continue

        start = float(
            segment["start"]
        )
        end = float(
            segment["end"]
        )

        if not turns:
            turns.append(
                {
                    "start": start,
                    "end": end,
                    "duration": end - start,
                    "speaker_label": segment["speaker"],
                    "person_id": person,
                }
            )
            continue

        previous = turns[-1]

        same_person = (
            previous["person_id"]
            == person
        )

        close = (
            start
            <= previous["end"] + 0.12
        )

        if same_person and close:
            previous["end"] = max(
                previous["end"],
                end,
            )

            previous["duration"] = (
                previous["end"]
                - previous["start"]
            )
        else:
            turns.append(
                {
                    "start": start,
                    "end": end,
                    "duration": end - start,
                    "speaker_label": segment["speaker"],
                    "person_id": person,
                }
            )

    return turns


def validate_turns(
    turns,
    video_end,
):
    errors = []

    previous_end = 0.0

    for i, turn in enumerate(turns):
        start = float(turn["start"])
        end = float(turn["end"])

        if start < -EPS:
            errors.append(
                f"turn_{i}: negative start"
            )

        if end > video_end + EPS:
            errors.append(
                f"turn_{i}: exceeds video"
            )

        if end <= start:
            errors.append(
                f"turn_{i}: invalid interval"
            )

        if start < previous_end - 0.5:
            # Small overlaps can happen in diarization.
            # Large backward movement is an integrity problem.
            errors.append(
                f"turn_{i}: large backward/overlap jump"
            )

        previous_end = max(
            previous_end,
            end,
        )

    return sorted(set(errors))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--diarization",
        required=True,
    )

    parser.add_argument(
        "--visual",
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

    diarization = load_json(
        args.diarization
    )

    visual = load_json(
        args.visual
    )

    speaker_segments = build_speaker_activity(
        diarization
    )

    observations = identity_observations(
        visual
    )

    mapping, evidence, mapping_score = (
        infer_global_mapping(
            speaker_segments,
            observations,
        )
    )

    turns = build_turns(
        speaker_segments,
        mapping,
    )

    validation_errors = validate_turns(
        turns,
        args.video_end,
    )

    speaker_durations = {}

    for segment in speaker_segments:
        speaker = segment["speaker"]

        speaker_durations[speaker] = (
            speaker_durations.get(
                speaker,
                0.0,
            )
            + (
                segment["end"]
                - segment["start"]
            )
        )

    output = {
        "version": "1.0",
        "module": "full_video_speaker_attribution",

        "source_range": {
            "start": 0.0,
            "end": args.video_end,
        },

        "speaker_mapping": {
            speaker: {
                "person_id": person,
                "evidence": evidence[speaker],
            }
            for speaker, person in mapping.items()
        },

        "mapping_score": mapping_score,

        "turns": turns,

        "statistics": {
            "diarization_segments": len(
                speaker_segments
            ),
            "merged_turns": len(turns),
            "speaker_durations": speaker_durations,
            "visual_observations": len(
                observations
            ),
            "validation_error_count": len(
                validation_errors
            ),
        },

        "validation_errors": validation_errors,
    }

    save_json(
        args.output,
        output,
    )

    print("=" * 72)
    print("FULL VIDEO SPEAKER ATTRIBUTION v1")
    print("=" * 72)
    print()

    print("GLOBAL MAPPING")
    for speaker, person in mapping.items():
        ev = evidence[speaker]
        print(
            f"  {speaker} -> {person}"
            f" | confidence={ev['confidence']:.4f}"
            f" | margin={ev['margin']:.4f}"
        )

    print()
    print("DIARIZATION SEGMENTS:", len(
        speaker_segments
    ))

    print("MERGED TURNS:", len(turns))

    print(
        "VALIDATION ERRORS:",
        len(validation_errors),
    )

    print()
    print(
        "Output:",
        args.output,
    )

    if validation_errors:
        print()
        print("ERRORS")
        for error in validation_errors:
            print(" -", error)

        raise SystemExit(1)

    print()
    print(
        "SUCCESS: full-video speaker attribution "
        "created."
    )


if __name__ == "__main__":
    main()