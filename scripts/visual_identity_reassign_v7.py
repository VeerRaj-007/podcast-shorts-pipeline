#!/usr/bin/env python3
"""Reassign the two persistent interview identities from an existing v6 JSON.

No video/model inference is required. This v7 pass fixes identity swaps caused by
appearance-only clustering by combining:
  - anchor identities from v6
  - SFace prototype similarity
  - screen-position (especially x-position) continuity
  - hard two-person overlap constraints
  - per-timestamp duplicate suppression

Usage:
  python scripts/visual_identity_reassign_v7.py \
      --input tmp/new_inde_visual_intelligence_v6.json \
      --output tmp/new_inde_visual_intelligence_v7.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


EPS = 1e-9
EXPECTED_PEOPLE = 2
MIN_OBS_TRACK = 3
OVERLAP_HARD = 0.55
SPATIAL_WEIGHT = 0.35
APPEARANCE_WEIGHT = 0.65


def l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / max(n, EPS)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), EPS))


def overlap(a: dict[str, Any], b: dict[str, Any]) -> float:
    start = max(float(a["first_time"]), float(b["first_time"]))
    end = min(float(a["last_time"]), float(b["last_time"]))
    inter = max(0.0, end - start)
    shorter = max(
        0.001,
        min(
            float(a["last_time"]) - float(a["first_time"]),
            float(b["last_time"]) - float(b["first_time"]),
        ),
    )
    return min(1.0, inter / shorter)


def center_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax, ay = map(float, a["mean_center"])
    bx, by = map(float, b["mean_center"])
    return float(np.hypot(ax - bx, ay - by))


def track_score(
    track: dict[str, Any],
    proto: np.ndarray,
    anchor_center: np.ndarray,
    identity_side_x: float,
) -> float:
    emb = np.asarray(track["appearance_prototype"], dtype=np.float32)
    center = np.asarray(track["mean_center"], dtype=np.float32)
    app = cosine(emb, proto)

    # X position is particularly informative for this fixed two-person interview.
    x_dist = abs(float(center[0]) - identity_side_x)
    x_sim = 1.0 - min(1.0, x_dist / 0.35)

    # Small y contribution prevents pathological matches without overpowering x.
    spatial_dist = float(np.linalg.norm(center - anchor_center))
    spatial = 1.0 - min(1.0, spatial_dist / 0.50)

    return APPEARANCE_WEIGHT * app + SPATIAL_WEIGHT * (0.80 * x_sim + 0.20 * spatial)


def choose_anchors(tracks: list[dict[str, Any]], ids: list[int]) -> tuple[int, int]:
    by_id = {int(t["track_id"]): i for i, t in enumerate(tracks)}
    if all(tid in by_id for tid in ids[:2]):
        return by_id[ids[0]], by_id[ids[1]]

    ordered = sorted(
        range(len(tracks)),
        key=lambda i: (int(tracks[i]["observations"]), float(tracks[i]["duration"])),
        reverse=True,
    )
    return ordered[0], ordered[1]


def reassign(input_path: Path, output_path: Path) -> dict[str, Any]:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    tracks = [t for t in data.get("tracks", []) if int(t.get("observations", 0)) >= MIN_OBS_TRACK]
    observations = data.get("observations", [])
    diagnostics_old = data.get("global_identity_diagnostics", {})
    old_anchor_ids = diagnostics_old.get("anchor_track_ids", [])
    anchor_a_idx, anchor_b_idx = choose_anchors(tracks, [int(x) for x in old_anchor_ids])

    anchor_tracks = [tracks[anchor_a_idx], tracks[anchor_b_idx]]
    prototypes = [
        l2(np.asarray(anchor_tracks[0]["appearance_prototype"], dtype=np.float32)),
        l2(np.asarray(anchor_tracks[1]["appearance_prototype"], dtype=np.float32)),
    ]
    anchor_centers = [
        np.asarray(anchor_tracks[0]["mean_center"], dtype=np.float32),
        np.asarray(anchor_tracks[1]["mean_center"], dtype=np.float32),
    ]
    side_x = [float(anchor_centers[0][0]), float(anchor_centers[1][0])]

    # v6's group ordering means PERSON_0 is the evidence-heavy identity. We preserve
    # that external convention after solving the two labels.
    assignments = [-1] * len(tracks)
    track_to_idx = {int(t["track_id"]): i for i, t in enumerate(tracks)}

    # Seed anchors explicitly.
    assignments[anchor_a_idx] = 0
    assignments[anchor_b_idx] = 1

    order = sorted(
        range(len(tracks)),
        key=lambda i: (int(tracks[i]["observations"]), float(tracks[i]["duration"])),
        reverse=True,
    )

    for _ in range(6):
        for i in order:
            if i in (anchor_a_idx, anchor_b_idx):
                continue
            tr = tracks[i]
            raw = [
                track_score(tr, prototypes[0], anchor_centers[0], side_x[0]),
                track_score(tr, prototypes[1], anchor_centers[1], side_x[1]),
            ]

            # Hard two-person constraint: if this track substantially overlaps a
            # strong track already assigned to one identity, prefer the other.
            penalties = [0.0, 0.0]
            for j in order:
                if j == i:
                    continue
                aj = assignments[j]
                if aj not in (0, 1):
                    continue
                other = tracks[j]
                ov = overlap(tr, other)
                if ov >= OVERLAP_HARD and int(other["observations"]) >= 5:
                    # Stronger tracks act as evidence for the opposite identity.
                    strength = min(1.0, int(other["observations"]) / 50.0)
                    penalties[aj] = max(penalties[aj], 0.75 * ov * strength)
            scores = [raw[k] - penalties[k] for k in (0, 1)]
            assignments[i] = 0 if scores[0] >= scores[1] else 1

        # Re-estimate identity prototypes, weighted heavily toward long tracks.
        for k in (0, 1):
            vectors: list[np.ndarray] = []
            weights: list[float] = []
            for i, a in enumerate(assignments):
                if a != k:
                    continue
                vectors.append(np.asarray(tracks[i]["appearance_prototype"], dtype=np.float32))
                weights.append(float(tracks[i]["observations"]))
            if vectors:
                acc = np.zeros_like(vectors[0])
                for v, w in zip(vectors, weights):
                    acc += v * w
                prototypes[k] = l2(acc)

    # Ensure deterministic external labels: PERSON_0 gets the larger evidence group.
    group_obs = [
        sum(int(tracks[i]["observations"]) for i, a in enumerate(assignments) if a == k)
        for k in (0, 1)
    ]
    if group_obs[1] > group_obs[0]:
        assignments = [1 - a for a in assignments]
        prototypes.reverse()
        anchor_centers.reverse()
        side_x.reverse()
        anchor_a_idx, anchor_b_idx = anchor_b_idx, anchor_a_idx
        group_obs.reverse()

    track_identity: dict[int, str] = {}
    for i, a in enumerate(assignments):
        track_identity[int(tracks[i]["track_id"])] = f"PERSON_{a}"

    # Add identity ids to tracks and observations.
    for t in data.get("tracks", []):
        tid = int(t["track_id"])
        t["identity_id"] = track_identity.get(tid)

    # Per timestamp, suppress duplicate local detections of the same global identity.
    # Keep the highest confidence/area observation; this is what downstream framing needs.
    by_time: dict[float, list[dict[str, Any]]] = {}
    for o in observations:
        ident = track_identity.get(int(o["track_id"]))
        o["identity_id"] = ident
        by_time.setdefault(float(o["timestamp"]), []).append(o)

    deduped: list[dict[str, Any]] = []
    duplicate_count = 0
    for ts in sorted(by_time):
        items = by_time[ts]
        best_by_identity: dict[str, dict[str, Any]] = {}
        for o in items:
            ident = o.get("identity_id")
            if ident not in ("PERSON_0", "PERSON_1"):
                continue
            if ident in best_by_identity:
                duplicate_count += 1
                old = best_by_identity[ident]
                old_key = (float(old.get("score", 0.0)), float(old.get("area_normalized", 0.0)))
                new_key = (float(o.get("score", 0.0)), float(o.get("area_normalized", 0.0)))
                if new_key > old_key:
                    best_by_identity[ident] = o
            else:
                best_by_identity[ident] = o
        deduped.extend(best_by_identity.values())

    # Rebuild identities from finalized track labels.
    identities: list[dict[str, Any]] = []
    for ident in ("PERSON_0", "PERSON_1"):
        member_tracks = [t for t in data.get("tracks", []) if t.get("identity_id") == ident]
        total_obs = sum(int(t["observations"]) for t in member_tracks)
        if not member_tracks:
            continue
        first_seen = min(float(t["first_time"]) for t in member_tracks)
        last_seen = max(float(t["last_time"]) for t in member_tracks)
        proto = np.zeros(128, dtype=np.float32)
        for t in member_tracks:
            proto += np.asarray(t["appearance_prototype"], dtype=np.float32) * float(t["observations"])
        proto = l2(proto)
        identities.append({
            "identity_id": ident,
            "source_track_ids": sorted(int(t["track_id"]) for t in member_tracks),
            "track_count": len(member_tracks),
            "observations": total_obs,
            "first_seen": round(first_seen, 4),
            "last_seen": round(last_seen, 4),
            "duration": round(last_seen - first_seen, 4),
            "appearance_prototype": [round(float(x), 6) for x in proto],
        })

    data["version"] = "7.0"
    data["global_identity_diagnostics"] = {
        "method": "two_person_constrained_spatial_overlap_v7",
        "expected_people": 2,
        "identity_count": len(identities),
        "anchor_track_ids": [int(anchor_tracks[0]["track_id"]), int(anchor_tracks[1]["track_id"])],
        "anchor_mean_centers": [
            [round(float(x), 5) for x in anchor_tracks[0]["mean_center"]],
            [round(float(x), 5) for x in anchor_tracks[1]["mean_center"]],
        ],
        "spatial_weight": SPATIAL_WEIGHT,
        "appearance_weight": APPEARANCE_WEIGHT,
        "hard_overlap_ratio": OVERLAP_HARD,
        "duplicate_observations_removed": duplicate_count,
        "group_observations": group_obs,
    }
    data["global_identities"] = identities
    data["observations_deduped"] = sorted(deduped, key=lambda o: (float(o["timestamp"]), str(o["identity_id"])))

    # Keep the full observation stream intact, but expose the identity-clean stream as
    # the preferred downstream representation.
    data["identity_clean_observations"] = sorted(deduped, key=lambda o: (float(o["timestamp"]), str(o["identity_id"])))
    data["statistics"]["global_identity_count"] = len(identities)
    data["statistics"]["identity_clean_observations"] = len(deduped)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = reassign(Path(args.input), Path(args.output))
    print("=== VISUAL IDENTITY v7 ===")
    print("global_identity_count:", result["statistics"]["global_identity_count"])
    print("identity_clean_observations:", result["statistics"]["identity_clean_observations"])
    print("diagnostics:", result["global_identity_diagnostics"])
    for ident in result["global_identities"]:
        print(ident["identity_id"], "tracks:", ident["track_count"], "observations:", ident["observations"])
    print("Output:", args.output)


if __name__ == "__main__":
    main()
