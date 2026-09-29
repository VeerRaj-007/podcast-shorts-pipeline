#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path
from typing import Any


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def interval_mid(start: float, end: float) -> float:
    return (float(start) + float(end)) / 2.0


def overlap_duration(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def overlap_ratio(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = overlap_duration(a0, a1, b0, b1)
    union = max(1e-9, max(a1, b1) - min(a0, b0))
    return inter / union


def distance_to_interval(value: float, start: float, end: float) -> float:
    if start <= value <= end:
        return 0.0
    return min(abs(value - start), abs(value - end))


def gaussian(distance: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return math.exp(-0.5 * (distance / scale) ** 2)


def build_stable_lookup(stable: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not stable:
        return []
    timeline = stable.get("timeline") or stable.get("samples") or []
    result: list[dict[str, Any]] = []
    for item in timeline:
        if isinstance(item, dict) and "time" in item:
            result.append(item)
    result.sort(key=lambda x: float(x["time"]))
    return result


def nearest_stable_sample(
    samples: list[dict[str, Any]],
    t: float,
    max_distance: float | None = None,
) -> dict[str, Any] | None:
    if not samples:
        return None
    item = min(samples, key=lambda s: abs(float(s.get("time", 0.0)) - t))
    if max_distance is not None and abs(float(item.get("time", 0.0)) - t) > max_distance:
        return None
    return item


def infer_target_speaker(
    moment: dict[str, Any],
    stable_samples: list[dict[str, Any]],
) -> str | None:
    explicit = moment.get("speaker") or moment.get("active_speaker")
    if explicit:
        return str(explicit)

    speech = float(moment.get("speech", 0.0) or 0.0)
    if speech <= 0.0:
        # Never invent a speaker for a non-speech moment using a distant timeline sample.
        return None

    m0 = float(moment.get("start", 0.0))
    m1 = float(moment.get("end", m0))
    mt = interval_mid(m0, m1)
    near = nearest_stable_sample(stable_samples, mt, max_distance=1.0)
    if not near:
        return None
    speaker = near.get("active_speaker") or near.get("speaker")
    return str(speaker) if speaker else None


def infer_shot_profile(shot: dict[str, Any]) -> str:
    shot_type = str(shot.get("shot_type", ""))
    avg_faces = float(shot.get("avg_face_count", 0.0) or 0.0)
    coverage = float(shot.get("face_coverage_avg", 0.0) or 0.0)

    if shot_type in {"insert_or_screen", "non_face"}:
        return "insert"
    if avg_faces >= 1.70:
        return "two_shot"
    if coverage >= 0.075:
        return "close"
    if coverage >= 0.045:
        return "medium"
    return "wide"


def classify_role(shot: dict[str, Any]) -> str:
    profile = infer_shot_profile(shot)
    if profile == "insert":
        return "insert"
    if profile == "two_shot":
        return "conversation_wide"
    if profile == "close":
        return "speaker_close"
    if profile == "medium":
        return "speaker_medium"
    return "speaker_wide"


def desired_profiles(moment: dict[str, Any], target_speaker: str | None) -> list[str]:
    intensity = float(moment.get("intensity", 0.0) or 0.0)
    speech = float(moment.get("speech", 0.0) or 0.0)
    reaction = float(moment.get("reaction_score", 0.0) or 0.0)
    visual = float(moment.get("visual_score", 0.0) or 0.0)
    emotion = float(moment.get("emotion_score", 0.0) or 0.0)

    if visual >= 0.75:
        return ["insert", "close", "medium", "two_shot", "wide"]
    if reaction >= 0.65 or emotion >= 0.75:
        return ["close", "medium", "two_shot", "wide", "insert"]
    if speech > 0.0 and target_speaker:
        if intensity >= 0.70:
            return ["close", "medium", "two_shot", "wide"]
        return ["medium", "close", "two_shot", "wide"]
    if intensity >= 0.72:
        return ["two_shot", "close", "medium", "wide", "insert"]
    return ["two_shot", "medium", "wide", "close", "insert"]


def speaker_alignment_for_shot(
    shot: dict[str, Any],
    target_speaker: str | None,
    stable_samples: list[dict[str, Any]],
) -> tuple[float, float]:
    if not target_speaker or not stable_samples:
        return 0.0, 0.0

    s0 = float(shot.get("start", 0.0))
    s1 = float(shot.get("end", s0))
    samples = [
        s for s in stable_samples
        if s0 - 0.25 <= float(s.get("time", 0.0)) <= s1 + 0.25
    ]
    if not samples:
        return 0.0, 0.0

    matches = 0
    usable = 0
    for sample in samples:
        active = sample.get("active_speaker") or sample.get("speaker")
        if not active:
            continue
        usable += 1
        if str(active) == target_speaker:
            matches += 1

    if usable == 0:
        return 0.0, 0.0
    return matches / usable, usable / max(1, len(samples))


def temporal_candidate_kind(
    shot: dict[str, Any],
    moment: dict[str, Any],
    local_radius: float,
) -> tuple[bool, str, float, float]:
    m0 = float(moment.get("start", 0.0))
    m1 = float(moment.get("end", m0))
    mt = interval_mid(m0, m1)
    s0 = float(shot.get("start", 0.0))
    s1 = float(shot.get("end", s0))

    direct_overlap = overlap_ratio(s0, s1, m0, m1)
    distance = distance_to_interval(mt, s0, s1)
    local = gaussian(distance, max(0.5, local_radius * 0.55))

    # A local shot can start/end just before/after the detected moment.
    is_local = distance <= local_radius or direct_overlap > 0.0
    if direct_overlap > 0.0:
        kind = "overlap"
    elif is_local:
        kind = "nearby"
    else:
        kind = "remote"
    return is_local, kind, direct_overlap, local


def score_shot(
    shot: dict[str, Any],
    moment: dict[str, Any],
    target_speaker: str | None,
    stable_samples: list[dict[str, Any]],
    used_recently: set[str],
    local_radius: float,
) -> tuple[float, dict[str, float], list[str], bool, str]:
    profile = infer_shot_profile(shot)
    role = classify_role(shot)
    preferred = desired_profiles(moment, target_speaker)

    if profile in preferred:
        rank = preferred.index(profile)
        profile_score = 1.0 - rank / max(1, len(preferred) - 1)
    else:
        profile_score = 0.0

    is_local, temporal_kind, direct_overlap, temporal_locality = temporal_candidate_kind(
        shot, moment, local_radius
    )

    visual = float(moment.get("visual_score", 0.0) or 0.0)
    reaction = float(moment.get("reaction_score", 0.0) or 0.0)
    emotion = float(moment.get("emotion_score", 0.0) or 0.0)
    intensity = float(moment.get("intensity", 0.0) or 0.0)
    speech = float(moment.get("speech", 0.0) or 0.0)

    no_face = float(shot.get("no_face_ratio", 0.0) or 0.0)
    face_presence = clamp(1.0 - no_face)

    speaker_match, speaker_observation_coverage = speaker_alignment_for_shot(
        shot, target_speaker, stable_samples
    )

    # Inserts are globally reusable only for strongly visual moments.
    globally_reusable_insert = profile == "insert" and max(visual, reaction) >= 0.75

    temporal_score = direct_overlap * 0.72 + temporal_locality * 0.28
    if not is_local and not globally_reusable_insert:
        temporal_score = 0.0

    # Strict guard: ordinary dialogue/reaction shots must remain local.
    locality_gate = 1.0 if is_local or globally_reusable_insert else 0.0

    close_value = clamp(max(reaction, emotion, intensity)) if profile in {"close", "medium"} else 0.0
    insert_fit = 0.0
    if profile == "insert":
        insert_fit = clamp(max(visual, reaction))

    recent_penalty = 0.18 if shot.get("shot_id") in used_recently else 0.0

    # Penalize remote source reuse heavily. A remote non-insert shot cannot win.
    remote_penalty = 0.85 if not is_local and not globally_reusable_insert else 0.0

    total = (
        0.27 * profile_score
        + 0.23 * temporal_score
        + 0.14 * speaker_match
        + 0.09 * speaker_observation_coverage
        + 0.09 * face_presence
        + 0.08 * close_value
        + 0.07 * insert_fit
        + 0.03 * locality_gate
        - recent_penalty
        - remote_penalty
    )
    total = max(0.0, total)

    reasons: list[str] = []
    if profile_score >= 0.75:
        reasons.append(f"preferred {profile} framing")
    if direct_overlap >= 0.25:
        reasons.append("overlaps current moment")
    elif temporal_kind == "nearby":
        reasons.append("near current moment")
    if speaker_match >= 0.70:
        reasons.append("matches active speaker locally")
    elif speaker_match >= 0.40:
        reasons.append("partially matches active speaker locally")
    if close_value >= 0.70 and profile in {"close", "medium"}:
        reasons.append("supports high-impact reaction/emotion")
    if insert_fit >= 0.75:
        reasons.append("strong visual/insertion candidate")
    if not is_local and globally_reusable_insert:
        reasons.append("remote insert allowed for visual moment")
    if not reasons:
        reasons.append("best local source shot")

    components = {
        "profile": clamp(profile_score),
        "temporal_overlap": clamp(direct_overlap),
        "temporal_locality": clamp(temporal_locality),
        "temporal_score": clamp(temporal_score),
        "face_presence": face_presence,
        "speaker_alignment": speaker_match,
        "speaker_observation_coverage": speaker_observation_coverage,
        "close_value": close_value,
        "insert_fit": insert_fit,
        "locality_gate": locality_gate,
        "repeat_penalty": recent_penalty,
        "remote_penalty": remote_penalty,
    }
    return total, components, reasons, is_local or globally_reusable_insert, role


def rank_candidates(
    moment: dict[str, Any],
    shots: list[dict[str, Any]],
    stable_samples: list[dict[str, Any]],
    used_recently: set[str],
    local_radius: float,
) -> list[dict[str, Any]]:
    target_speaker = infer_target_speaker(moment, stable_samples)

    scored: list[dict[str, Any]] = []
    for shot in shots:
        score, components, reasons, eligible, role = score_shot(
            shot,
            moment,
            target_speaker,
            stable_samples,
            used_recently,
            local_radius,
        )
        scored.append({
            "shot_id": shot["shot_id"],
            "score": round(score, 6),
            "eligible": eligible,
            "role": role,
            "shot_type": shot.get("shot_type"),
            "profile": infer_shot_profile(shot),
            "start": shot.get("start"),
            "end": shot.get("end"),
            "duration": shot.get("duration"),
            "target_speaker": target_speaker,
            "reasons": reasons,
            "components": {k: round(v, 6) for k, v in components.items()},
        })

    # Eligible candidates first. This prevents a remote shot from beating a valid local shot.
    scored.sort(key=lambda x: (x["eligible"], x["score"]), reverse=True)
    return scored


def select_for_moment(
    moment: dict[str, Any],
    shots: list[dict[str, Any]],
    stable_samples: list[dict[str, Any]],
    used_recently: set[str],
    local_radius: float,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    ranked = rank_candidates(moment, shots, stable_samples, used_recently, local_radius)
    eligible = [c for c in ranked if c["eligible"]]
    if not eligible:
        # This should be rare; don't silently select an unrelated shot.
        return None, ranked[:5]
    return eligible[0], eligible[1:5]


def build_plan(
    data: dict[str, Any],
    moments_data: dict[str, Any],
    stable_data: dict[str, Any] | None,
    local_radius: float,
) -> dict[str, Any]:
    shots = list(data.get("shots", []))
    moments = list(moments_data.get("moments", []))
    stable_samples = build_stable_lookup(stable_data)

    shots.sort(key=lambda s: float(s.get("start", 0.0)))
    moments.sort(key=lambda m: float(m.get("start", m.get("time", 0.0))))

    used_recently: deque[str] = deque(maxlen=4)
    selections: list[dict[str, Any]] = []
    rejected_remote = 0
    selected_count = 0

    for moment in moments:
        winner, alternatives = select_for_moment(
            moment,
            shots,
            stable_samples,
            set(used_recently),
            local_radius,
        )
        if winner is None:
            rejected_remote += 1
            continue

        used_recently.append(winner["shot_id"])
        selected_count += 1

        selections.append({
            "moment": {
                "start": moment.get("start"),
                "end": moment.get("end"),
                "time": moment.get("time"),
                "intensity": moment.get("intensity"),
                "speech": moment.get("speech", 0.0),
                "reaction_score": moment.get("reaction_score", 0.0),
                "visual_score": moment.get("visual_score", 0.0),
                "emotion_score": moment.get("emotion_score", 0.0),
            },
            "winner": winner,
            "alternatives": alternatives,
        })

    local_winners = sum(1 for s in selections if s["winner"]["components"]["locality_gate"] >= 1.0)
    remote_allowed_winners = selected_count - local_winners

    return {
        "version": "2.0",
        "mode": "editorial_shot_selection",
        "shot_inventory": data.get("video"),
        "moment_count": len(moments),
        "selected_count": selected_count,
        "selection_policy": {
            "type": "deterministic_v2",
            "purpose": "choose real source shots while enforcing temporal editorial locality",
            "no_synthetic_zoom": True,
            "uses_real_source_shots": True,
            "local_radius_seconds": local_radius,
            "remote_shots_allowed_only_for_strong_visual_inserts": True,
            "non_local_dialogue_shots_can_not_win": True,
            "speaker_inference_requires_nearby_timeline_evidence": True,
        },
        "diagnostics": {
            "local_winners": local_winners,
            "remote_insert_winners": remote_allowed_winners,
            "moments_without_eligible_local_shot": rejected_remote,
        },
        "selections": selections,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal editorial shot selector v2")
    parser.add_argument("--shots", required=True)
    parser.add_argument("--moments", required=True)
    parser.add_argument("--stable", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--local-radius",
        type=float,
        default=2.25,
        help="Maximum distance from the moment for ordinary source-shot eligibility.",
    )
    args = parser.parse_args()

    shots_data = load_json(args.shots)
    moments_data = load_json(args.moments)
    stable_data = load_json(args.stable) if args.stable else None

    plan = build_plan(
        shots_data,
        moments_data,
        stable_data,
        local_radius=max(0.5, args.local_radius),
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    print(json.dumps({
        "output": str(out),
        "shots": len(shots_data.get("shots", [])),
        "moments": plan["moment_count"],
        "selected": plan["selected_count"],
        "local_winners": plan["diagnostics"]["local_winners"],
        "remote_insert_winners": plan["diagnostics"]["remote_insert_winners"],
        "without_eligible_local_shot": plan["diagnostics"]["moments_without_eligible_local_shot"],
    }, indent=2))


if __name__ == "__main__":
    main()
