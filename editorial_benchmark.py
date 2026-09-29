from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


EPS = 0.001


@dataclass
class Unit:
    index: int
    start: float
    end: float
    text: str
    speaker: str = ""


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def find_unit_records(data: Any) -> List[Any]:
    if isinstance(data, list):
        return data

    if not isinstance(data, dict):
        raise ValueError("Transcript JSON must be an object or array.")

    for key in ("units", "segments", "transcript", "utterances"):
        value = data.get(key)
        if isinstance(value, list) and value:
            return value

    raise ValueError(
        "Could not find transcript units. Expected units/segments/"
        "transcript/utterances."
    )


def load_units(path: Path, video_end: float) -> List[Unit]:
    data = load_json(path)
    records = find_unit_records(data)

    units: List[Unit] = []

    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            continue

        start = record.get("start", record.get("start_time"))
        end = record.get("end", record.get("end_time"))

        try:
            start = float(start)
            end = float(end)
        except (TypeError, ValueError):
            continue

        text = record.get("text", record.get("content", ""))
        if not str(text).strip() or end <= start:
            continue

        speaker = str(
            record.get("speaker", record.get("speaker_id", ""))
        )

        units.append(
            Unit(
                index=idx,
                start=max(0.0, start),
                end=min(video_end, end),
                text=" ".join(str(text).split()),
                speaker=speaker,
            )
        )

    units.sort(key=lambda u: (u.start, u.end, u.index))

    if not units:
        raise ValueError("No usable transcript units found.")

    return units


def units_for_clip(
    units: Sequence[Unit],
    start: float,
    end: float,
) -> List[Unit]:
    return [
        u
        for u in units
        if u.start >= start - EPS
        and u.end <= end + EPS
    ]


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    intersection = max(
        0.0,
        min(a_end, b_end) - max(a_start, b_start),
    )

    union = max(a_end, b_end) - min(a_start, b_start)

    if union <= 0:
        return 0.0

    return intersection / union


def audit_output(
    units: Sequence[Unit],
    data: Dict[str, Any],
    video_end: float,
) -> Dict[str, Any]:

    errors: List[str] = []
    warnings: List[str] = []

    clips = data.get("clips", [])
    rejected = data.get("rejected_candidates", [])

    quality_floor = float(
        data.get("quality_floor", 0.0)
    )

    confidence_floor = float(
        data.get("confidence_floor", 0.0)
    )

    requested_count = int(
        data.get("requested_count", len(clips))
    )

    # ------------------------------------------------------------
    # Top-level contract
    # ------------------------------------------------------------

    if not isinstance(clips, list):
        errors.append("clips is not a list.")
        clips = []

    selected_count = data.get(
        "selected_count",
        len(clips),
    )

    if selected_count != len(clips):
        errors.append(
            f"selected_count={selected_count} but clips has {len(clips)} items."
        )

    if len(clips) > requested_count:
        errors.append(
            f"selected {len(clips)} clips for requested_count={requested_count}."
        )

    abstained = bool(
        data.get("abstained_from_fillers", False)
    )

    if len(clips) < requested_count and not abstained:
        errors.append(
            "Fewer clips than requested but abstained_from_fillers is false."
        )

    # ------------------------------------------------------------
    # Per-clip contract
    # ------------------------------------------------------------

    seen_ids = set()
    per_clip: List[Dict[str, Any]] = []

    for position, clip in enumerate(clips, 1):
        cid = str(clip.get("candidate_id", ""))

        clip_errors: List[str] = []
        clip_warnings: List[str] = []

        if not cid:
            clip_errors.append("missing_candidate_id")
        elif cid in seen_ids:
            clip_errors.append("duplicate_candidate_id")
        seen_ids.add(cid)

        try:
            start = float(clip["start"])
            end = float(clip["end"])
        except (KeyError, TypeError, ValueError):
            clip_errors.append("invalid_timestamp")
            per_clip.append(
                {
                    "rank_position": position,
                    "candidate_id": cid,
                    "errors": clip_errors,
                    "warnings": clip_warnings,
                }
            )
            continue

        if start < -EPS:
            clip_errors.append("negative_start")

        if end > video_end + EPS:
            clip_errors.append("end_beyond_video")

        if end <= start + EPS:
            clip_errors.append("non_positive_duration")

        duration = end - start

        if duration < 4.0:
            clip_warnings.append("very_short_duration")

        complete = clip.get("complete")
        if complete is not True:
            clip_errors.append("complete_not_true")

        critic_status = str(
            clip.get("critic_status", "")
        )
        if critic_status != "passed":
            clip_errors.append(
                f"critic_status={critic_status!r}"
            )

        quality = float(
            clip.get("quality_score", 0.0)
        )

        confidence = float(
            clip.get("confidence", 0.0)
        )

        boundary_confidence = float(
            clip.get("boundary_confidence", 0.0)
        )

        if quality + EPS < quality_floor:
            clip_errors.append(
                f"quality_below_floor:{quality:.4f}<{quality_floor:.4f}"
            )

        if confidence + EPS < confidence_floor:
            clip_errors.append(
                f"confidence_below_floor:{confidence:.4f}<{confidence_floor:.4f}"
            )

        if boundary_confidence + EPS < confidence_floor:
            clip_errors.append(
                "boundary_confidence_below_floor:"
                f"{boundary_confidence:.4f}<{confidence_floor:.4f}"
            )

        reasons = clip.get("rejection_reasons", [])
        if reasons:
            clip_errors.append("rejection_reasons_not_empty")

        clip_units = units_for_clip(
            units,
            start,
            end,
        )

        if not clip_units:
            clip_errors.append("no_transcript_units_covered")

        source_start = clip.get("source_unit_start")
        source_end = clip.get("source_unit_end")

        if source_start is not None and source_end is not None:
            actual_indices = {u.index for u in clip_units}

            if source_start not in actual_indices:
                clip_warnings.append(
                    f"source_unit_start_not_found:{source_start}"
                )

            if source_end not in actual_indices:
                clip_warnings.append(
                    f"source_unit_end_not_found:{source_end}"
                )

        # Compare declared transcript with source transcript.
        actual_text = " ".join(
            u.text
            for u in clip_units
        )

        declared_text = " ".join(
            str(clip.get("transcript_text", "")).split()
        )

        if declared_text and actual_text:
            # Exact normalized equality is intentionally strict for this
            # machine audit. Boundary refinement should produce deterministic
            # source text.
            if declared_text != actual_text:
                clip_warnings.append(
                    "declared_transcript_differs_from_source_units"
                )

        per_clip.append(
            {
                "rank_position": position,
                "candidate_id": cid,
                "start": start,
                "end": end,
                "duration": duration,
                "quality_score": quality,
                "confidence": confidence,
                "boundary_confidence": boundary_confidence,
                "complete": complete,
                "critic_status": critic_status,
                "errors": clip_errors,
                "warnings": clip_warnings,
            }
        )

        if clip_errors:
            errors.extend(
                f"{cid}: {err}"
                for err in clip_errors
            )

        warnings.extend(
            f"{cid}: {warning}"
            for warning in clip_warnings
        )

    # ------------------------------------------------------------
    # Ranking contract
    # ------------------------------------------------------------

    score_sequence = [
        float(c.get("score", 0.0))
        for c in clips
    ]

    for i in range(1, len(score_sequence)):
        if score_sequence[i] > score_sequence[i - 1] + EPS:
            errors.append(
                "ranking_not_descending:"
                f"rank {i} -> rank {i + 1}"
            )

    # ------------------------------------------------------------
    # Inter-clip overlap
    # ------------------------------------------------------------

    overlap_pairs: List[Dict[str, Any]] = []

    for i in range(len(clips)):
        for j in range(i + 1, len(clips)):
            a = clips[i]
            b = clips[j]

            try:
                ov = overlap(
                    float(a["start"]),
                    float(a["end"]),
                    float(b["start"]),
                    float(b["end"]),
                )
            except (KeyError, TypeError, ValueError):
                continue

            if ov >= 0.55:
                overlap_pairs.append(
                    {
                        "a": a.get("candidate_id"),
                        "b": b.get("candidate_id"),
                        "overlap_ratio": round(ov, 4),
                    }
                )

    if overlap_pairs:
        errors.append(
            f"{len(overlap_pairs)} clip pairs exceed the "
            "0.55 overlap threshold."
        )

    # ------------------------------------------------------------
    # Selected-vs-rejected consistency
    # ------------------------------------------------------------

    rejected_ids = {
        str(c.get("candidate_id"))
        for c in rejected
    }

    selected_rejected_conflicts = [
        str(c.get("candidate_id"))
        for c in clips
        if str(c.get("candidate_id")) in rejected_ids
    ]

    if selected_rejected_conflicts:
        errors.append(
            "candidate appears in both clips and rejected_candidates:"
            + ",".join(selected_rejected_conflicts)
        )

    # ------------------------------------------------------------
    # Overall
    # ------------------------------------------------------------

    passed = not errors

    return {
        "audit_version": "1.0",
        "passed": passed,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "requested_count": requested_count,
            "selected_count": len(clips),
            "accepted_candidates": data.get("accepted_candidates"),
            "discovered_candidates": data.get("discovered_candidates"),
            "refined_candidates": data.get("refined_candidates"),
            "reviewed_candidates": data.get("reviewed_candidates"),
            "quality_floor": quality_floor,
            "confidence_floor": confidence_floor,
            "abstained_from_fillers": abstained,
        },
        "clips": per_clip,
        "overlap_pairs": overlap_pairs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit Groq Editorial Engine output."
    )

    parser.add_argument(
        "--transcript",
        required=True,
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

    transcript_path = Path(args.transcript)
    editorial_path = Path(args.editorial)
    output_path = Path(args.output)

    units = load_units(
        transcript_path,
        args.video_end,
    )

    data = load_json(
        editorial_path
    )

    if not isinstance(data, dict):
        raise ValueError(
            "Editorial output must be a JSON object."
        )

    audit = audit_output(
        units,
        data,
        args.video_end,
    )

    output_path.write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 72)
    print("EDITORIAL BENCHMARK / AUDIT v1")
    print("=" * 72)
    print()
    print(
        "STATUS:",
        "PASS" if audit["passed"] else "FAIL",
    )
    print()
    print(
        "Requested:",
        audit["summary"]["requested_count"],
    )
    print(
        "Selected:",
        audit["summary"]["selected_count"],
    )
    print(
        "Errors:",
        len(audit["errors"]),
    )
    print(
        "Warnings:",
        len(audit["warnings"]),
    )
    print()
    print("Output:", output_path)

    if audit["errors"]:
        print()
        print("ERRORS")
        for error in audit["errors"]:
            print(" -", error)

    if audit["warnings"]:
        print()
        print("WARNINGS")
        for warning in audit["warnings"]:
            print(" -", warning)

    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())