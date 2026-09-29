from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TORCH_LIB = ROOT / ".venv-whisperx" / "Lib" / "site-packages" / "torch" / "lib"
if not TORCH_LIB.exists():
    raise RuntimeError(f"CUDA runtime directory not found: {TORCH_LIB}")
os.environ["PATH"] = str(TORCH_LIB) + os.pathsep + os.environ.get("PATH", "")
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(TORCH_LIB))

import cv2
import numpy as np
import onnxruntime as ort

DEFAULT_VIDEO = ROOT / "Leon Kennedy meets Grace Ashcroft  Nick Apostolides and Angela Sant'Albano Interview Each Other - FGS (1080p, h264).mp4"
DEFAULT_SHOTS = ROOT / "tmp" / "new_inde_shots.json"
DEFAULT_YUNET = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
DEFAULT_SFACE = ROOT / "models" / "face_recognition_sface_2021dec.onnx"
DEFAULT_OUTPUT = ROOT / "tmp" / "new_inde_visual_intelligence_v6.json"

YUNET_W = YUNET_H = 640
SFACE_W = SFACE_H = 112
STRIDES = (8, 16, 32)
SAMPLE_EVERY = 0.5
FACE_THRESHOLD = 0.60
NMS_THRESHOLD = 0.45
TOP_K = 100

# Tracking is intentionally stricter than the SFace demo's verification
# threshold. The lower verification threshold is not appropriate for
# deciding which of two simultaneously visible people is the same identity.
TRACK_COSINE = 0.58
REID_COSINE = 0.60
MAX_ACTIVE_MISSES = 8          # 4 seconds at 2 FPS
MAX_REID_GAP = 120.0           # allow long gaps / shot changes
GALLERY_SIZE = 12
EMA_ALPHA = 0.20
MIN_FACE_AREA = 0.0015
MIN_OBS = 3
MAX_CENTER_DISTANCE = 0.25

# Global identity clustering. This is deliberately separate from local tracking.
# Tracks with compatible SFace prototypes are merged into persistent PERSON_X IDs.
GLOBAL_COSINE_STRONG = 0.62
GLOBAL_COSINE_WEAK = 0.54
MAX_CLUSTER_SIZE = 64

SFACE_TARGET = np.asarray(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


@dataclass
class Detection:
    x: float
    y: float
    w: float
    h: float
    score: float
    landmarks: np.ndarray

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)


@dataclass
class Track:
    track_id: int
    first_time: float
    last_time: float
    last_detection: Detection
    prototype: np.ndarray
    gallery: list[np.ndarray] = field(default_factory=list)
    observations: int = 0
    misses: int = 0
    timestamps: list[float] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    centers: list[tuple[float, float]] = field(default_factory=list)
    areas: list[float] = field(default_factory=list)
    shot_ids: set[str] = field(default_factory=set)

    def identity_similarity(self, emb: np.ndarray) -> float:
        values = [cosine(emb, self.prototype)]
        values.extend(cosine(emb, g) for g in self.gallery)
        return max(values)

    def update(
        self,
        t: float,
        det: Detection,
        emb: np.ndarray,
        fw: int,
        fh: int,
        shot_id: str | None,
    ) -> None:
        self.last_time = t
        self.last_detection = det
        self.prototype = l2_normalize(
            (1.0 - EMA_ALPHA) * self.prototype + EMA_ALPHA * emb
        )
        self.gallery.append(emb.copy())
        if len(self.gallery) > GALLERY_SIZE:
            self.gallery.pop(0)
        self.observations += 1
        self.misses = 0
        self.timestamps.append(t)
        self.confidences.append(det.score)
        self.centers.append((det.cx / fw, det.cy / fh))
        self.areas.append(det.area / max(1, fw * fh))
        if shot_id:
            self.shot_ids.add(shot_id)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(x))
    return x if n <= 1e-12 else x / n


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(l2_normalize(a), l2_normalize(b)))


def iou(a: Detection, b: Detection) -> float:
    ax0, ay0, ax1, ay1 = a.x, a.y, a.x + a.w, a.y + a.h
    bx0, by0, bx1, by1 = b.x, b.y, b.x + b.w, b.y + b.h
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def center_distance(a: Detection, b: Detection, fw: int, fh: int) -> float:
    dx = (a.cx - b.cx) / max(1, fw)
    dy = (a.cy - b.cy) / max(1, fh)
    return float(np.hypot(dx, dy))


def load_shots(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    shots = data.get("shots", []) if isinstance(data, dict) else data
    out = []
    for s in shots:
        if not isinstance(s, dict):
            continue
        start = float(s.get("start", 0.0))
        end = float(s.get("end", 0.0))
        if end > start:
            out.append({**s, "start": start, "end": end})
    return sorted(out, key=lambda s: s["start"])


def shot_at(shots: list[dict[str, Any]], t: float) -> dict[str, Any] | None:
    for s in shots:
        if s["start"] <= t < s["end"]:
            return s
    return None


def letterbox_frame(frame: np.ndarray, target_w: int, target_h: int) -> tuple[np.ndarray, float, float, float]:
    """Resize without distortion and pad to the fixed YuNet input size."""
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    resized_w = max(1, int(round(w * scale)))
    resized_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    pad_x = (target_w - resized_w) / 2.0
    pad_y = (target_h - resized_h) / 2.0
    canvas = np.zeros((target_h, target_w, 3), dtype=frame.dtype)
    left = int(round(pad_x))
    top = int(round(pad_y))
    canvas[top:top + resized_h, left:left + resized_w] = resized
    return canvas, scale, pad_x, pad_y


def non_max_suppression(
    detections: list[Detection],
    threshold: float,
    top_k: int,
) -> list[Detection]:
    """Greedy IoU NMS for YuNet detections."""
    ordered = sorted(detections, key=lambda d: d.score, reverse=True)
    if top_k > 0:
        ordered = ordered[:top_k]

    keep: list[Detection] = []
    for det in ordered:
        if all(iou(det, kept) <= threshold for kept in keep):
            keep.append(det)
    return keep


def decode_yunet(
    outputs: list[np.ndarray],
    fw: int,
    fh: int,
    scale: float,
    pad_x: float,
    pad_y: float,
    score_threshold: float,
    nms_threshold: float,
    top_k: int,
) -> list[Detection]:
    if len(outputs) != 12:
        raise RuntimeError(f"Expected 12 YuNet outputs, got {len(outputs)}")

    detections: list[Detection] = []

    for level, stride in enumerate(STRIDES):
        cls = outputs[level][0]
        obj = outputs[3 + level][0]
        box = outputs[6 + level][0]
        kps = outputs[9 + level][0]

        gw = YUNET_W // stride
        gh = YUNET_H // stride
        expected = gw * gh

        if cls.shape[0] != expected:
            raise RuntimeError(
                f"Unexpected YuNet anchor count at stride {stride}: "
                f"{cls.shape[0]} vs {expected}"
            )

        for i in range(expected):
            # OpenCV reference: clamp each confidence to [0,1], then sqrt(cls*obj).
            cls_score = float(np.clip(cls[i, 0], 0.0, 1.0))
            obj_score = float(np.clip(obj[i, 0], 0.0, 1.0))
            score = float(np.sqrt(cls_score * obj_score))

            if score < score_threshold:
                continue

            gx = i % gw
            gy = i // gw

            dx, dy, dw, dh = map(float, box[i, :4])

            cx_net = (gx + dx) * stride
            cy_net = (gy + dy) * stride
            w_net = float(np.exp(np.clip(dw, -20.0, 20.0)) * stride)
            h_net = float(np.exp(np.clip(dh, -20.0, 20.0)) * stride)

            # Map network coordinates back through the letterbox transform.
            cx = (cx_net - pad_x) / max(scale, 1e-12)
            cy = (cy_net - pad_y) / max(scale, 1e-12)
            w = w_net / max(scale, 1e-12)
            h = h_net / max(scale, 1e-12)

            points = []
            for j in range(5):
                lx_net = (kps[i, 2 * j] + gx) * stride
                ly_net = (kps[i, 2 * j + 1] + gy) * stride
                lx = (float(lx_net) - pad_x) / max(scale, 1e-12)
                ly = (float(ly_net) - pad_y) / max(scale, 1e-12)
                points.append((lx, ly))

            x = cx - w / 2.0
            y = cy - h / 2.0

            # Clip boxes only after mapping back to the real image.
            x0 = max(0.0, x)
            y0 = max(0.0, y)
            x1 = min(float(fw), x + w)
            y1 = min(float(fh), y + h)
            clipped_w = x1 - x0
            clipped_h = y1 - y0

            if clipped_w <= 1.0 or clipped_h <= 1.0:
                continue

            points_array = np.asarray(points, dtype=np.float32)
            points_array[:, 0] = np.clip(points_array[:, 0], 0.0, float(fw))
            points_array[:, 1] = np.clip(points_array[:, 1], 0.0, float(fh))

            detections.append(
                Detection(
                    x0,
                    y0,
                    clipped_w,
                    clipped_h,
                    score,
                    points_array,
                )
            )

    return non_max_suppression(
        detections,
        nms_threshold,
        top_k,
    )


def align_sface(frame: np.ndarray, det: Detection) -> np.ndarray | None:
    if det.landmarks.shape != (5, 2):
        return None
    M, _ = cv2.estimateAffinePartial2D(
        det.landmarks.astype(np.float32),
        SFACE_TARGET,
        method=cv2.LMEDS,
    )
    if M is None:
        return None
    return cv2.warpAffine(frame, M, (SFACE_W, SFACE_H), flags=cv2.INTER_LINEAR)


def sface_embedding(
    session: ort.InferenceSession,
    input_name: str,
    frame: np.ndarray,
    det: Detection,
) -> np.ndarray | None:
    aligned = align_sface(frame, det)
    if aligned is None:
        return None
    rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32).transpose(2, 0, 1)[None, ...]
    out = session.run(None, {input_name: blob})
    if not out:
        return None
    return l2_normalize(np.asarray(out[0], dtype=np.float32).reshape(-1))


def active_match(
    det: Detection,
    emb: np.ndarray,
    tracks: list[Track],
    used: set[int],
    fw: int,
    fh: int,
) -> tuple[int | None, float]:
    candidates: list[tuple[float, float, int]] = []
    for idx, tr in enumerate(tracks):
        if idx in used or tr.misses > MAX_ACTIVE_MISSES:
            continue
        sim = tr.identity_similarity(emb)
        ov = iou(det, tr.last_detection)
        dist = center_distance(det, tr.last_detection, fw, fh)
        # Appearance is the main identity signal. Spatial information only
        # breaks ties and protects against obvious cross-person swaps.
        if sim < TRACK_COSINE:
            continue
        if sim < 0.68 and dist > MAX_CENTER_DISTANCE and ov < 0.15:
            continue
        score = sim + 0.12 * ov + 0.03 * (1.0 - min(1.0, dist))
        candidates.append((score, sim, idx))
    if not candidates:
        return None, -1.0
    score, sim, idx = max(candidates)
    return idx, sim


def reidentify(
    t: float,
    emb: np.ndarray,
    tracks: list[Track],
    used: set[int],
) -> tuple[int | None, float]:
    candidates: list[tuple[float, float, int]] = []
    for idx, tr in enumerate(tracks):
        if idx in used:
            continue
        if t - tr.last_time > MAX_REID_GAP:
            continue
        sim = tr.identity_similarity(emb)
        if sim < REID_COSINE:
            continue
        # Prefer the strongest appearance match; tiny recency preference helps
        # when an identity has multiple fragments with similar embeddings.
        recency = max(0.0, 1.0 - (t - tr.last_time) / MAX_REID_GAP)
        score = sim + 0.02 * recency
        candidates.append((score, sim, idx))
    if not candidates:
        return None, -1.0
    score, sim, idx = max(candidates)
    return idx, sim


def start_track(
    track_id: int,
    t: float,
    det: Detection,
    emb: np.ndarray,
    fw: int,
    fh: int,
    shot_id: str | None,
) -> Track:
    tr = Track(
        track_id=track_id,
        first_time=t,
        last_time=t,
        last_detection=det,
        prototype=emb.copy(),
        gallery=[emb.copy()],
        observations=1,
    )
    tr.timestamps.append(t)
    tr.confidences.append(det.score)
    tr.centers.append((det.cx / fw, det.cy / fh))
    tr.areas.append(det.area / max(1, fw * fh))
    if shot_id:
        tr.shot_ids.add(shot_id)
    return tr


def finalize_tracks(tracks: list[Track], min_obs: int) -> list[dict[str, Any]]:
    result = []
    for tr in tracks:
        if tr.observations < min_obs:
            continue
        c = np.asarray(tr.centers, dtype=np.float32)
        result.append(
            {
                "track_id": tr.track_id,
                "first_time": round(tr.first_time, 4),
                "last_time": round(tr.last_time, 4),
                "duration": round(tr.last_time - tr.first_time, 4),
                "observations": tr.observations,
                "mean_confidence": round(float(np.mean(tr.confidences)), 4),
                "mean_center": [round(float(c[:, 0].mean()), 5), round(float(c[:, 1].mean()), 5)],
                "center_range": {
                    "min_x": round(float(c[:, 0].min()), 5),
                    "max_x": round(float(c[:, 0].max()), 5),
                    "min_y": round(float(c[:, 1].min()), 5),
                    "max_y": round(float(c[:, 1].max()), 5),
                },
                "mean_area_normalized": round(float(np.mean(tr.areas)), 6),
                "shot_ids": sorted(tr.shot_ids),
                # Compact appearance representation used by the global identity
                # stage. We keep only one prototype in the JSON, not every frame embedding.
                "appearance_prototype": [round(float(x), 6) for x in tr.prototype],
            }
        )
    return result


def _track_similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    return cosine(
        np.asarray(a["appearance_prototype"], dtype=np.float32),
        np.asarray(b["appearance_prototype"], dtype=np.float32),
    )


def _interval_overlap_ratio(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Fraction of the shorter track interval that overlaps the other track."""
    start = max(float(a["first_time"]), float(b["first_time"]))
    end = min(float(a["last_time"]), float(b["last_time"]))
    overlap = max(0.0, end - start)
    shorter = max(
        0.001,
        min(
            float(a["last_time"]) - float(a["first_time"]),
            float(b["last_time"]) - float(b["first_time"]),
        ),
    )
    return min(1.0, overlap / shorter)


def _track_center_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax, ay = map(float, a["mean_center"])
    bx, by = map(float, b["mean_center"])
    return float(np.hypot(ax - bx, ay - by))


def _pick_two_person_anchors(final_tracks: list[dict[str, Any]]) -> tuple[int, int]:
    """Pick two long tracks that are likely to represent the two interview participants."""
    if len(final_tracks) == 1:
        return 0, 0

    anchor_a = max(
        range(len(final_tracks)),
        key=lambda i: (
            int(final_tracks[i]["observations"]),
            float(final_tracks[i]["duration"]),
        ),
    )

    a = final_tracks[anchor_a]
    candidates: list[tuple[float, int]] = []
    for i, b in enumerate(final_tracks):
        if i == anchor_a:
            continue
        overlap = _interval_overlap_ratio(a, b)
        spatial = min(1.0, _track_center_distance(a, b) / 0.30)
        appearance_difference = 1.0 - _track_similarity(a, b)
        obs_factor = min(1.0, float(b["observations"]) / max(1, int(a["observations"])))
        score = (
            0.45 * obs_factor
            + 0.25 * overlap
            + 0.18 * spatial
            + 0.12 * appearance_difference
        )
        candidates.append((score, i))

    candidates.sort(reverse=True)
    return anchor_a, candidates[0][1]


def global_identity_clustering(
    final_tracks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[int, str], dict[str, Any]]:
    """
    Build persistent identities for this source as a constrained two-person interview.

    The previous graph-threshold approach produced dozens of PERSON_* clusters because
    appearance embeddings vary with pose, crop, lighting, and camera angle. For a known
    two-person interview, forcing exactly two global identities is much more stable.

    We initialize from two strong anchors, iteratively assign every local track to the
    closer identity using appearance plus screen-position consistency, and discourage
    assigning temporally overlapping tracks to the same person.
    """
    n = len(final_tracks)
    if n == 0:
        return [], {}, {
            "method": "two_person_constrained",
            "expected_people": 2,
            "identity_count": 0,
            "anchor_track_ids": [],
            "iterations": 0,
        }

    if n == 1:
        track = final_tracks[0]
        identity_id = "PERSON_0"
        return [
            {
                "identity_id": identity_id,
                "source_track_ids": [int(track["track_id"])],
                "track_count": 1,
                "observations": int(track["observations"]),
                "first_seen": round(float(track["first_time"]), 4),
                "last_seen": round(float(track["last_time"]), 4),
                "duration": round(float(track["last_time"] - track["first_time"]), 4),
                "appearance_prototype": track["appearance_prototype"],
            }
        ], {int(track["track_id"]): identity_id}, {
            "method": "two_person_constrained",
            "expected_people": 2,
            "identity_count": 1,
            "anchor_track_ids": [int(track["track_id"])],
            "iterations": 0,
            "note": "Only one stable track was available.",
        }

    anchor_a, anchor_b = _pick_two_person_anchors(final_tracks)

    prototypes = [
        np.asarray(final_tracks[anchor_a]["appearance_prototype"], dtype=np.float32),
        np.asarray(final_tracks[anchor_b]["appearance_prototype"], dtype=np.float32),
    ]
    anchor_centers = [
        np.asarray(final_tracks[anchor_a]["mean_center"], dtype=np.float32),
        np.asarray(final_tracks[anchor_b]["mean_center"], dtype=np.float32),
    ]

    assignments = [-1] * n
    margin_history: list[float] = []

    # Longer tracks are assigned first, so persistent identity prototypes are driven
    # by the most reliable observations.
    order = sorted(
        range(n),
        key=lambda i: (
            int(final_tracks[i]["observations"]),
            float(final_tracks[i]["duration"]),
        ),
        reverse=True,
    )

    for iteration in range(8):
        for i in order:
            track = final_tracks[i]
            emb = np.asarray(track["appearance_prototype"], dtype=np.float32)
            center = np.asarray(track["mean_center"], dtype=np.float32)

            scores: list[float] = []
            for identity_idx in (0, 1):
                sim = cosine(emb, prototypes[identity_idx])
                spatial = 1.0 - min(
                    1.0,
                    float(np.linalg.norm(center - anchor_centers[identity_idx])) / 0.45,
                )
                score = sim + 0.06 * spatial

                # If this track overlaps another already-assigned long track, strongly
                # discourage placing both into the same identity. This is the key
                # two-person constraint for simultaneous interview shots.
                conflict = 0.0
                for j in order:
                    if j == i or assignments[j] != identity_idx:
                        continue
                    other = final_tracks[j]
                    if int(other["observations"]) < 5:
                        continue
                    overlap = _interval_overlap_ratio(track, other)
                    if overlap >= 0.55:
                        conflict = max(conflict, overlap)
                if conflict > 0.0:
                    score -= 0.22 * conflict

                scores.append(score)

            chosen = 0 if scores[0] >= scores[1] else 1
            assignments[i] = chosen
            margin_history.append(abs(scores[0] - scores[1]))

        # Recompute observation-weighted identity prototypes.
        new_prototypes: list[np.ndarray] = []
        for identity_idx in (0, 1):
            vectors = []
            weights = []
            for i, assignment in enumerate(assignments):
                if assignment != identity_idx:
                    continue
                vectors.append(
                    np.asarray(final_tracks[i]["appearance_prototype"], dtype=np.float32)
                )
                weights.append(float(final_tracks[i]["observations"]))
            if not vectors:
                new_prototypes.append(prototypes[identity_idx])
                continue
            accumulator = np.zeros_like(vectors[0])
            for vector, weight in zip(vectors, weights):
                accumulator += vector * weight
            new_prototypes.append(l2_normalize(accumulator))
        prototypes = new_prototypes

    groups: list[list[int]] = [[], []]
    for i, identity_idx in enumerate(assignments):
        groups[identity_idx].append(i)

    # Keep PERSON_0 as the identity with the larger evidence volume.
    groups.sort(
        key=lambda members: (
            sum(int(final_tracks[i]["observations"]) for i in members),
            len(members),
        ),
        reverse=True,
    )

    track_to_identity: dict[int, str] = {}
    identities: list[dict[str, Any]] = []
    for identity_idx, members in enumerate(groups):
        identity_id = f"PERSON_{identity_idx}"
        if not members:
            continue

        total_obs = sum(int(final_tracks[i]["observations"]) for i in members)
        first_seen = min(float(final_tracks[i]["first_time"]) for i in members)
        last_seen = max(float(final_tracks[i]["last_time"]) for i in members)

        prototype = np.zeros(128, dtype=np.float32)
        for i in members:
            prototype += (
                np.asarray(final_tracks[i]["appearance_prototype"], dtype=np.float32)
                * float(final_tracks[i]["observations"])
            )
        prototype = l2_normalize(prototype)

        for i in members:
            track_to_identity[int(final_tracks[i]["track_id"])] = identity_id

        identities.append(
            {
                "identity_id": identity_id,
                "source_track_ids": sorted(int(final_tracks[i]["track_id"]) for i in members),
                "track_count": len(members),
                "observations": total_obs,
                "first_seen": round(first_seen, 4),
                "last_seen": round(last_seen, 4),
                "duration": round(last_seen - first_seen, 4),
                "appearance_prototype": [round(float(x), 6) for x in prototype],
            }
        )

    diagnostics = {
        "method": "two_person_constrained",
        "expected_people": 2,
        "identity_count": len(identities),
        "anchor_track_ids": [
            int(final_tracks[anchor_a]["track_id"]),
            int(final_tracks[anchor_b]["track_id"]),
        ],
        "anchor_mean_centers": [
            [round(float(x), 5) for x in final_tracks[anchor_a]["mean_center"]],
            [round(float(x), 5) for x in final_tracks[anchor_b]["mean_center"]],
        ],
        "iterations": 8,
        "mean_assignment_margin": round(
            float(np.mean(margin_history)) if margin_history else 0.0,
            5,
        ),
        "min_assignment_margin": round(
            float(np.min(margin_history)) if margin_history else 0.0,
            5,
        ),
        "cluster_sizes": [len(members) for members in groups],
    }

    return identities, track_to_identity, diagnostics

def build_shot_summaries(
    shots: list[dict[str, Any]],
    obs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for s in shots:
        items = [o for o in obs if s["start"] <= o["timestamp"] < s["end"]]
        times = sorted({o["timestamp"] for o in items})
        max_faces = max((o["face_count"] for o in items), default=0)
        composition = (
            "no_face" if max_faces == 0
            else "single_face" if max_faces == 1
            else "two_faces" if max_faces == 2
            else "multiple_faces"
        )
        track_summaries = []
        for tid in sorted({int(o["track_id"]) for o in items}):
            xs = [o for o in items if int(o["track_id"]) == tid]
            centers = np.asarray([o["center"] for o in xs], dtype=np.float32)
            areas = np.asarray([o["area_normalized"] for o in xs], dtype=np.float32)
            scores = np.asarray([o["score"] for o in xs], dtype=np.float32)
            track_summaries.append(
                {
                    "track_id": tid,
                    "observations": len(xs),
                    "presence_ratio": round(
                        len({x["timestamp"] for x in xs}) / max(1, len(times)), 4
                    ),
                    "mean_confidence": round(float(scores.mean()), 4),
                    "mean_center": [
                        round(float(centers[:, 0].mean()), 4),
                        round(float(centers[:, 1].mean()), 4),
                    ],
                    "center_range": {
                        "min_x": round(float(centers[:, 0].min()), 4),
                        "max_x": round(float(centers[:, 0].max()), 4),
                        "min_y": round(float(centers[:, 1].min()), 4),
                        "max_y": round(float(centers[:, 1].max()), 4),
                    },
                    "mean_face_area_normalized": round(float(areas.mean()), 5),
                    "max_face_area_normalized": round(float(areas.max()), 5),
                }
            )
        track_summaries.sort(
            key=lambda x: (x["presence_ratio"], x["mean_face_area_normalized"]),
            reverse=True,
        )
        result.append(
            {
                "shot_id": str(s.get("shot_id", "")),
                "start": s["start"],
                "end": s["end"],
                "duration": round(s["end"] - s["start"], 4),
                "sample_count": len(times),
                "max_face_count": max_faces,
                "mean_face_count": round(
                    float(np.mean([o["face_count"] for o in items])) if items else 0.0,
                    4,
                ),
                "composition": composition,
                "tracks": track_summaries,
            }
        )
    return result


def main() -> None:
    p = argparse.ArgumentParser(
        description="GPU YuNet + SFace local tracking plus global identity clustering for interview footage."
    )
    p.add_argument("--video", default=str(DEFAULT_VIDEO))
    p.add_argument("--shots", default=str(DEFAULT_SHOTS))
    p.add_argument("--yunet", default=str(DEFAULT_YUNET))
    p.add_argument("--sface", default=str(DEFAULT_SFACE))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--sample-every", type=float, default=SAMPLE_EVERY)
    p.add_argument("--face-threshold", type=float, default=FACE_THRESHOLD)
    p.add_argument("--nms-threshold", type=float, default=NMS_THRESHOLD)
    p.add_argument("--track-cosine", type=float, default=TRACK_COSINE)
    p.add_argument("--reid-cosine", type=float, default=REID_COSINE)
    args = p.parse_args()

    video_path, shots_path, yunet_path, sface_path, output_path = map(
        Path, (args.video, args.shots, args.yunet, args.sface, args.output)
    )
    for path, label in (
        (video_path, "video"),
        (shots_path, "shots"),
        (yunet_path, "YuNet"),
        (sface_path, "SFace"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    print("=== VISUAL INTELLIGENCE v6 ===")
    print("ONNX Runtime:", ort.__version__)
    print("Available providers:", ort.get_available_providers())
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    yu = ort.InferenceSession(str(yunet_path), providers=providers)
    sf = ort.InferenceSession(str(sface_path), providers=providers)
    if "CUDAExecutionProvider" not in yu.get_providers():
        raise RuntimeError("YuNet did not initialize with CUDAExecutionProvider")
    if "CUDAExecutionProvider" not in sf.get_providers():
        raise RuntimeError("SFace did not initialize with CUDAExecutionProvider")
    print("YuNet providers:", yu.get_providers())
    print("SFace providers:", sf.get_providers())

    yu_input = yu.get_inputs()[0]
    sf_input = sf.get_inputs()[0]
    print("YuNet input:", yu_input.name, yu_input.shape)
    print("SFace input:", sf_input.name, sf_input.shape)
    if list(yu_input.shape) != [1, 3, 640, 640]:
        raise RuntimeError(f"Unexpected YuNet input shape: {yu_input.shape}")
    if list(sf_input.shape) != [1, 3, 112, 112]:
        raise RuntimeError(f"Unexpected SFace input shape: {sf_input.shape}")

    shots = load_shots(shots_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frames / max(fps, 1e-6)
    print(f"Video: {fw}x{fh} {fps:.3f} fps {duration:.3f}s")

    tracks: list[Track] = []
    observations: list[dict[str, Any]] = []
    next_track_id = 0
    analyzed = 0
    raw_faces = 0
    embedding_count = 0
    active_matches = 0
    reid_matches = 0
    new_identity_tracks = 0
    next_sample = 0.0
    frame_index = 0
    sample_every = max(0.05, float(args.sample_every))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_index / max(fps, 1e-6)
        frame_index += 1
        if t + 1e-9 < next_sample:
            continue
        next_sample += sample_every
        analyzed += 1

        letterboxed, lb_scale, lb_pad_x, lb_pad_y = letterbox_frame(
            frame, YUNET_W, YUNET_H
        )
        blob = letterboxed.astype(np.float32).transpose(2, 0, 1)[None, ...]
        outputs = yu.run(None, {yu_input.name: blob})
        detections = decode_yunet(
            outputs,
            fw,
            fh,
            lb_scale,
            lb_pad_x,
            lb_pad_y,
            float(args.face_threshold),
            float(args.nms_threshold),
            TOP_K,
        )
        usable = [d for d in detections if d.area / max(1, fw * fh) >= MIN_FACE_AREA]
        raw_faces += len(usable)

        emb_list = [sface_embedding(sf, sf_input.name, frame, d) for d in usable]
        embedding_count += sum(e is not None for e in emb_list)

        # Every frame advances track miss counters. A track may be dormant for
        # longer than MAX_ACTIVE_MISSES and still be recoverable through re-ID.
        for tr in tracks:
            tr.misses += 1

        shot = shot_at(shots, t)
        shot_id = str(shot.get("shot_id", "")) if shot else None
        order = sorted(range(len(usable)), key=lambda i: usable[i].score, reverse=True)
        used_tracks: set[int] = set()
        assignment: dict[int, int] = {}

        # Pass 1: local temporal tracking.
        for di in order:
            emb = emb_list[di]
            if emb is None:
                continue
            ti, _sim = active_match(
                usable[di], emb, tracks, used_tracks, fw, fh
            )
            if ti is None:
                continue
            tracks[ti].update(t, usable[di], emb, fw, fh, shot_id)
            used_tracks.add(ti)
            assignment[di] = tracks[ti].track_id
            active_matches += 1

        # Pass 2: global identity re-identification. This is the major change
        # from v2: a shot cut or several seconds without a detection no longer
        # automatically creates a new person.
        for di in order:
            if di in assignment:
                continue
            emb = emb_list[di]
            if emb is None:
                continue
            ti, _sim = reidentify(
                t, emb, tracks, used_tracks
            )
            if ti is None:
                continue
            tracks[ti].update(t, usable[di], emb, fw, fh, shot_id)
            used_tracks.add(ti)
            assignment[di] = tracks[ti].track_id
            reid_matches += 1

        # Pass 3: genuinely new identity.
        for di in order:
            if di in assignment:
                continue
            emb = emb_list[di]
            if emb is None:
                continue
            tr = start_track(
                next_track_id,
                t,
                usable[di],
                emb,
                fw,
                fh,
                shot_id,
            )
            tracks.append(tr)
            assignment[di] = next_track_id
            next_track_id += 1
            new_identity_tracks += 1

        for di, tid in assignment.items():
            d = usable[di]
            observations.append(
                {
                    "timestamp": round(t, 4),
                    "shot_id": shot_id,
                    "track_id": int(tid),
                    "score": round(d.score, 5),
                    "bbox": [round(d.x, 2), round(d.y, 2), round(d.w, 2), round(d.h, 2)],
                    "center": [round(d.cx / fw, 5), round(d.cy / fh, 5)],
                    "area_normalized": round(d.area / max(1, fw * fh), 6),
                    "landmarks": [
                        [round(float(pt[0]), 2), round(float(pt[1]), 2)]
                        for pt in d.landmarks
                    ],
                    "face_count": len(usable),
                }
            )

        if analyzed % 100 == 0:
            live = sum(tr.misses <= MAX_ACTIVE_MISSES for tr in tracks)
            print(
                f"Analyzed: {analyzed} | time: {t:.1f}s | faces: {len(usable)} | "
                f"tracks created: {next_track_id} | live tracks: {live} | "
                f"active matches: {active_matches} | re-ID: {reid_matches}"
            )

    cap.release()

    final_tracks = finalize_tracks(tracks, MIN_OBS)

    # Global identity layer: merge fragmented local tracks into persistent people.
    global_identities, track_to_identity, identity_diagnostics = global_identity_clustering(
        final_tracks
    )

    for observation in observations:
        tid = int(observation["track_id"])
        observation["identity_id"] = track_to_identity.get(tid)

    for track in final_tracks:
        track["identity_id"] = track_to_identity.get(
            int(track["track_id"])
        )

    shot_summaries = build_shot_summaries(shots, observations)

    # Add persistent identity composition to each shot.
    for shot in shot_summaries:
        identity_counts: dict[str, int] = {}
        for observation in observations:
            if not (
                shot["start"]
                <= observation["timestamp"]
                < shot["end"]
            ):
                continue
            identity_id = observation.get("identity_id")
            if identity_id is not None:
                identity_counts[identity_id] = identity_counts.get(identity_id, 0) + 1
        shot["identities_present"] = sorted(
            identity_counts,
            key=lambda identity: identity_counts[identity],
            reverse=True,
        )

    result = {
        "version": "6.0",
        "module": "visual_intelligence_yunet_sface",
        "source_video": str(video_path),
        "source_shots": str(shots_path),
        "models": {"yunet": str(yunet_path), "sface": str(sface_path)},
        "inference": {
            "runtime": "onnxruntime",
            "execution_provider": "CUDAExecutionProvider",
            "gpu_required": True,
            "sample_every_seconds": sample_every,
            "yunet_input": [640, 640],
            "sface_input": [112, 112],
            "face_threshold": float(args.face_threshold),
            "nms_threshold": float(args.nms_threshold),
            "track_cosine": float(args.track_cosine),
            "reid_cosine": float(args.reid_cosine),
            "gallery_size": GALLERY_SIZE,
            "max_active_misses": MAX_ACTIVE_MISSES,
            "max_reid_gap_seconds": MAX_REID_GAP,
        },
        "video": {
            "width": fw,
            "height": fh,
            "fps": round(fps, 4),
            "frame_count": frames,
            "duration": round(duration, 4),
        },
        "statistics": {
            "analyzed_frames": analyzed,
            "face_observations": len(observations),
            "sface_embeddings": embedding_count,
            "tracks_created": next_track_id,
            "stable_tracks": len(final_tracks),
            "active_matches": active_matches,
            "reid_matches": reid_matches,
            "new_identity_tracks": new_identity_tracks,
            "shots": len(shots),
            "global_identity_count": len(global_identities),
        },
        "global_identity_diagnostics": identity_diagnostics,
        "global_identities": global_identities,
        "tracks": final_tracks,
        "observations": observations,
        "shot_visual_intelligence": shot_summaries,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== COMPLETE ===")
    for k, v in result["statistics"].items():
        print(f"{k}: {v}")
    print("Output:", output_path)


if __name__ == "__main__":
    main()
