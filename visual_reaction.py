#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ============================================================
# Visual / Reaction Intelligence v1
#
# Goal:
#   Convert raw video vision signals into timestamped editorial
#   events that the sequence director can consume.
#
# This is intentionally a signal engine, not an "emotion AI"
# black box. Every event keeps its evidence so we can inspect
# why it was emitted.
#
# Signals:
#   - face presence / count
#   - face size / framing
#   - face motion
#   - lower-face (mouth/jaw) motion
#   - upper-face (eyes/eyebrow) motion
#   - face appearance change
#   - shot boundaries from shot inventory
#   - insert / non-face shots
#   - speaker context from stable speaker-face timeline
#
# Optional:
#   - YuNet face detector
#
# Output:
#   {
#       "events": [
#           {
#               "start": 12.0,
#               "end": 12.75,
#               "type": "reaction_candidate",
#               "face_id": "VISION_FACE_01",
#               "score": 0.82,
#               "signals": {...},
#               "reason": [...]
#           }
#       ],
#       "timeline": [...]
#   }
#
# The next sequence layer can consume:
#   reaction_candidate
#   facial_motion
#   visual_insert
#   framing_change
#   face_entry
#   face_exit
#   shot_boundary
# ============================================================


DEFAULT_SAMPLE_INTERVAL = 0.25
DEFAULT_FACE_INTERVAL = 0.25
DEFAULT_REACTION_WINDOW = 1.00
DEFAULT_MERGE_GAP = 0.50
DEFAULT_MIN_EVENT = 0.50

REACTION_THRESHOLD = 0.58
MOTION_THRESHOLD = 0.42
FRAME_CHANGE_THRESHOLD = 0.38

EVENT_TYPES = {
    "reaction_candidate",
    "facial_motion",
    "visual_insert",
    "framing_change",
    "face_entry",
    "face_exit",
    "shot_boundary",
}


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def robust_z(values: list[float]) -> list[float]:
    if not values:
        return []
    med = statistics.median(values)
    dev = [abs(v - med) for v in values]
    mad = statistics.median(dev) if dev else 0.0
    scale = max(1e-6, 1.4826 * mad)
    return [(v - med) / scale for v in values]


def robust_normalize(values: list[float], low_q: float = 0.10, high_q: float = 0.90) -> list[float]:
    if not values:
        return []
    lo = float(np.quantile(values, low_q))
    hi = float(np.quantile(values, high_q))
    if hi - lo < 1e-9:
        return [0.5 for _ in values]
    return [clamp01((v - lo) / (hi - lo)) for v in values]


def interval_distance(a0: float, a1: float, b0: float, b1: float) -> float:
    if a1 < b0:
        return b0 - a1
    if b1 < a0:
        return a0 - b1
    return 0.0


def interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def resize_gray(img: np.ndarray, width: int = 96, height: int = 96) -> np.ndarray:
    if img is None or img.size == 0:
        return np.zeros((height, width), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)


def crop_safe(frame: np.ndarray, x: float, y: float, w: float, h: float) -> np.ndarray:
    H, W = frame.shape[:2]
    x0 = max(0, min(W - 1, int(round(x))))
    y0 = max(0, min(H - 1, int(round(y))))
    x1 = max(x0 + 1, min(W, int(round(x + w))))
    y1 = max(y0 + 1, min(H, int(round(y + h))))
    return frame[y0:y1, x0:x1]


def face_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax0 = float(a["x"])
    ay0 = float(a["y"])
    ax1 = ax0 + float(a["width"])
    ay1 = ay0 + float(a["height"])

    bx0 = float(b["x"])
    by0 = float(b["y"])
    bx1 = bx0 + float(b["width"])
    by1 = by0 + float(b["height"])

    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)

    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih

    aa = max(1.0, (ax1 - ax0) * (ay1 - ay0))
    bb = max(1.0, (bx1 - bx0) * (by1 - by0))
    return inter / max(1.0, aa + bb - inter)


def center_distance(a: dict[str, Any], b: dict[str, Any], W: int, H: int) -> float:
    ax = float(a.get("center_x", float(a["x"]) + float(a["width"]) / 2))
    ay = float(a.get("center_y", float(a["y"]) + float(a["height"]) / 2))
    bx = float(b.get("center_x", float(b["x"]) + float(b["width"]) / 2))
    by = float(b.get("center_y", float(b["y"]) + float(b["height"]) / 2))
    dx = (ax - bx) / max(1.0, W)
    dy = (ay - by) / max(1.0, H)
    return math.sqrt(dx * dx + dy * dy)


@dataclass
class FaceSample:
    time: float
    face_id: str
    x: float
    y: float
    width: float
    height: float
    area_ratio: float
    center_x: float
    center_y: float
    detector_score: float
    crop_gray: np.ndarray


class FaceTracker:
    def __init__(self, max_gap: float = 0.75) -> None:
        self.max_gap = max_gap
        self.next_id = 0
        self.active: dict[str, FaceSample] = {}
        self.tracks: dict[str, list[FaceSample]] = {}

    def update(
        self,
        time_s: float,
        detections: list[dict[str, Any]],
        W: int,
        H: int,
        frame: np.ndarray,
    ) -> list[FaceSample]:
        current: list[FaceSample] = []

        candidates = list(self.active.items())
        used_prev: set[str] = set()

        ordered = sorted(
            detections,
            key=lambda d: float(d.get("width", 0.0)) * float(d.get("height", 0.0)),
            reverse=True,
        )

        for det in ordered:
            best_id = None
            best_score = -1e9

            for face_id, prev in candidates:
                if face_id in used_prev:
                    continue

                dt = time_s - prev.time
                if dt < 0.0 or dt > self.max_gap:
                    continue

                iou = face_iou(det, {
                    "x": prev.x,
                    "y": prev.y,
                    "width": prev.width,
                    "height": prev.height,
                    "center_x": prev.center_x,
                    "center_y": prev.center_y,
                })
                dist = center_distance(det, {
                    "x": prev.x,
                    "y": prev.y,
                    "width": prev.width,
                    "height": prev.height,
                    "center_x": prev.center_x,
                    "center_y": prev.center_y,
                }, W, H)

                # IoU is the main identity cue; distance breaks ties.
                score = 0.72 * iou + 0.28 * clamp01(1.0 - dist / 0.35)
                if score > best_score:
                    best_score = score
                    best_id = face_id

            if best_id is None or best_score < 0.12:
                best_id = f"VISION_FACE_{self.next_id:02d}"
                self.next_id += 1

            crop = crop_safe(
                frame,
                float(det["x"]),
                float(det["y"]),
                float(det["width"]),
                float(det["height"]),
            )
            gray = resize_gray(crop)

            fs = FaceSample(
                time=time_s,
                face_id=best_id,
                x=float(det["x"]),
                y=float(det["y"]),
                width=float(det["width"]),
                height=float(det["height"]),
                area_ratio=float(
                    (float(det["width"]) * float(det["height"])) / max(1.0, W * H)
                ),
                center_x=float(det.get("center_x", float(det["x"]) + float(det["width"]) / 2)),
                center_y=float(det.get("center_y", float(det["y"]) + float(det["height"]) / 2)),
                detector_score=float(det.get("score", 0.0)),
                crop_gray=gray,
            )

            current.append(fs)
            self.tracks.setdefault(best_id, []).append(fs)
            used_prev.add(best_id)

        new_active = {fs.face_id: fs for fs in current}
        self.active = new_active
        return current


class YuNetDetector:
    def __init__(self, model_path: str | None) -> None:
        self.detector = None
        self.backend = "none"

        if not model_path:
            return

        p = Path(model_path)
        if not p.exists():
            return

        try:
            self.detector = cv2.FaceDetectorYN.create(
                str(p),
                "",
                (320, 320),
                score_threshold=0.55,
                nms_threshold=0.45,
                top_k=5000,
            )
            self.backend = "yunet"
        except Exception:
            self.detector = None
            self.backend = "none"

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        if self.detector is None:
            return []

        H, W = frame.shape[:2]
        self.detector.setInputSize((W, H))
        _, faces = self.detector.detect(frame)

        if faces is None:
            return []

        out: list[dict[str, Any]] = []
        for row in faces:
            x, y, w, h = map(float, row[:4])
            score = float(row[-1])
            out.append({
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "center_x": x + w / 2.0,
                "center_y": y + h / 2.0,
                "score": score,
            })
        return out


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))) / 255.0)


def crop_motion(prev: np.ndarray, curr: np.ndarray) -> float:
    if prev is None or curr is None or prev.size == 0 or curr.size == 0:
        return 0.0

    p = prev.astype(np.float32) / 255.0
    c = curr.astype(np.float32) / 255.0

    gx_p = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3)
    gy_p = cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3)
    gx_c = cv2.Sobel(c, cv2.CV_32F, 1, 0, ksize=3)
    gy_c = cv2.Sobel(c, cv2.CV_32F, 0, 1, ksize=3)

    flow_like = (
        np.abs(gx_p - gx_c)
        + np.abs(gy_p - gy_c)
        + np.abs(p - c)
    )
    return clamp01(float(np.mean(flow_like) * 2.4))


def region_motion(prev: np.ndarray, curr: np.ndarray, y0: float, y1: float) -> float:
    if prev is None or curr is None:
        return 0.0
    H = min(prev.shape[0], curr.shape[0])
    ya = max(0, min(H - 1, int(round(H * y0))))
    yb = max(ya + 1, min(H, int(round(H * y1))))
    return crop_motion(prev[ya:yb], curr[ya:yb])


def load_stable_timeline(stable: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not stable:
        return []
    items = stable.get("timeline") or stable.get("samples") or []
    out = []
    for item in items:
        if isinstance(item, dict) and "time" in item:
            out.append(item)
    out.sort(key=lambda x: float(x["time"]))
    return out


def nearest_stable(
    timeline: list[dict[str, Any]],
    t: float,
    max_distance: float = 0.75,
) -> dict[str, Any] | None:
    if not timeline:
        return None
    item = min(timeline, key=lambda x: abs(float(x["time"]) - t))
    if abs(float(item["time"]) - t) > max_distance:
        return None
    return item


def load_shot_inventory(shots: dict[str, Any]) -> list[dict[str, Any]]:
    items = shots.get("shots") or []
    return [s for s in items if isinstance(s, dict)]


def nearest_shot(
    shots: list[dict[str, Any]],
    t: float,
) -> dict[str, Any] | None:
    if not shots:
        return None

    def dist(s: dict[str, Any]) -> float:
        a = float(s.get("start", 0.0))
        b = float(s.get("end", a))
        return interval_distance(t, t, a, b)

    return min(shots, key=dist)


def timeline_shot_events(shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events = []
    for i, shot in enumerate(shots):
        start = float(shot.get("start", 0.0))
        end = float(shot.get("end", start))
        shot_id = str(shot.get("shot_id", f"SHOT_{i:03d}"))
        shot_type = str(shot.get("shot_type", ""))

        if i > 0:
            events.append({
                "start": round(start, 3),
                "end": round(min(end, start + DEFAULT_MIN_EVENT), 3),
                "type": "shot_boundary",
                "score": clamp01(float(shot.get("cut_in_score") or 0.0)),
                "shot_id": shot_id,
                "signals": {
                    "cut_in_score": float(shot.get("cut_in_score") or 0.0),
                    "duration": max(0.0, end - start),
                },
                "reason": ["source shot boundary"],
            })

        if shot_type in {"insert_or_screen", "non_face"}:
            events.append({
                "start": round(start, 3),
                "end": round(min(end, start + DEFAULT_MIN_EVENT), 3),
                "type": "visual_insert",
                "score": 0.72,
                "shot_id": shot_id,
                "signals": {
                    "shot_type": shot_type,
                    "duration": max(0.0, end - start),
                },
                "reason": ["source shot has no stable face focus", "candidate visual insert"],
            })

    return events


def analyze_video(
    video_path: str,
    start: float,
    end: float,
    sample_interval: float,
    face_interval: float,
    detector: YuNetDetector,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps) if fps > 0 else 0.0

    tracker = FaceTracker()
    face_samples_by_time: list[dict[str, Any]] = []

    t = max(0.0, float(start))
    prev_full: np.ndarray | None = None
    prev_faces: dict[str, FaceSample] = {}

    while t < float(end) - 1e-6:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        Hf, Wf = frame.shape[:2]
        faces = detector.detect(frame)

        current = tracker.update(t, faces, Wf, Hf, frame)

        global_change = 0.0
        if prev_full is not None:
            curr_small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
            prev_small = cv2.resize(prev_full, (160, 90), interpolation=cv2.INTER_AREA)
            global_change = mean_abs_diff(
                cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY),
            )

        observations: list[dict[str, Any]] = []

        current_map = {fs.face_id: fs for fs in current}

        for fs in current:
            prev = prev_faces.get(fs.face_id)

            full_motion = 0.0
            lower_motion = 0.0
            upper_motion = 0.0
            appearance_change = 0.0
            center_speed = 0.0
            area_change = 0.0

            if prev is not None:
                dt = max(1e-3, fs.time - prev.time)
                full_motion = crop_motion(prev.crop_gray, fs.crop_gray)
                lower_motion = region_motion(prev.crop_gray, fs.crop_gray, 0.48, 0.92)
                upper_motion = region_motion(prev.crop_gray, fs.crop_gray, 0.05, 0.50)

                appearance_change = mean_abs_diff(prev.crop_gray, fs.crop_gray)

                dx = (fs.center_x - prev.center_x) / max(1.0, Wf)
                dy = (fs.center_y - prev.center_y) / max(1.0, Hf)
                center_speed = clamp01(math.sqrt(dx * dx + dy * dy) / 0.08 / dt)

                area_change = clamp01(
                    abs(fs.area_ratio - prev.area_ratio)
                    / max(1e-5, prev.area_ratio * 0.40)
                )

            observations.append({
                "time": round(t, 3),
                "face_id": fs.face_id,
                "x": round(fs.x, 3),
                "y": round(fs.y, 3),
                "width": round(fs.width, 3),
                "height": round(fs.height, 3),
                "area_ratio": round(fs.area_ratio, 6),
                "center_x": round(fs.center_x / max(1.0, Wf), 6),
                "center_y": round(fs.center_y / max(1.0, Hf), 6),
                "detector_score": round(fs.detector_score, 4),
                "full_face_motion": round(full_motion, 6),
                "mouth_region_motion": round(lower_motion, 6),
                "upper_face_motion": round(upper_motion, 6),
                "appearance_change": round(appearance_change, 6),
                "center_speed": round(center_speed, 6),
                "area_change": round(area_change, 6),
            })

        face_samples_by_time.append({
            "time": round(t, 3),
            "face_count": len(current),
            "global_change": round(global_change, 6),
            "faces": observations,
        })

        prev_full = frame
        prev_faces = current_map

        t += sample_interval

    cap.release()

    # Build per-face robust normalized reaction signals.
    all_obs: list[dict[str, Any]] = []
    by_face: dict[str, list[dict[str, Any]]] = {}

    for sample in face_samples_by_time:
        for obs in sample["faces"]:
            all_obs.append(obs)
            by_face.setdefault(obs["face_id"], []).append(obs)

    for face_id, obs_list in by_face.items():
        for field in (
            "full_face_motion",
            "mouth_region_motion",
            "upper_face_motion",
            "appearance_change",
            "center_speed",
            "area_change",
        ):
            vals = [float(o[field]) for o in obs_list]
            norm = robust_normalize(vals)
            for o, n in zip(obs_list, norm):
                o[field + "_norm"] = round(n, 6)

        for o in obs_list:
            o["reaction_score"] = round(
                clamp01(
                    0.28 * float(o["mouth_region_motion_norm"])
                    + 0.22 * float(o["upper_face_motion_norm"])
                    + 0.20 * float(o["appearance_change_norm"])
                    + 0.15 * float(o["center_speed_norm"])
                    + 0.15 * float(o["area_change_norm"])
                ),
                6,
            )
            o["facial_motion_score"] = round(
                clamp01(
                    0.46 * float(o["full_face_motion_norm"])
                    + 0.28 * float(o["mouth_region_motion_norm"])
                    + 0.26 * float(o["upper_face_motion_norm"])
                ),
                6,
            )

    return face_samples_by_time, all_obs, {
        "fps": fps,
        "width": W,
        "height": H,
        "duration": duration,
        "face_backend": detector.backend,
        "sample_interval": sample_interval,
        "face_interval": face_interval,
    }


def build_face_events(
    face_timeline: list[dict[str, Any]],
    stable: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []

    by_face: dict[str, list[dict[str, Any]]] = {}
    for sample in face_timeline:
        for obs in sample["faces"]:
            by_face.setdefault(obs["face_id"], []).append(obs)

    for face_id, obs_list in by_face.items():
        obs_list.sort(key=lambda o: float(o["time"]))

        reaction_flags = [
            float(o.get("reaction_score", 0.0)) >= REACTION_THRESHOLD
            for o in obs_list
        ]
        motion_flags = [
            float(o.get("facial_motion_score", 0.0)) >= MOTION_THRESHOLD
            for o in obs_list
        ]

        def emit_runs(flags: list[bool], event_type: str, threshold_field: str) -> None:
            run_start: int | None = None

            for i, flag in enumerate(flags + [False]):
                if flag and run_start is None:
                    run_start = i
                elif not flag and run_start is not None:
                    j = i - 1
                    start_t = float(obs_list[run_start]["time"])
                    end_t = float(obs_list[j]["time"]) + DEFAULT_FACE_INTERVAL

                    if end_t - start_t >= DEFAULT_MIN_EVENT:
                        block = obs_list[run_start:i]
                        peak = max(block, key=lambda x: float(x[threshold_field]))
                        score = clamp01(float(peak[threshold_field]))

                        near = nearest_stable(stable, float(peak["time"]))
                        target_speaker = None
                        if near:
                            target_speaker = near.get("active_speaker") or near.get("speaker")

                        reasons = []
                        if event_type == "reaction_candidate":
                            reasons.append("face motion spike")
                            if float(peak.get("mouth_region_motion_norm", 0.0)) >= 0.75:
                                reasons.append("strong lower-face movement")
                            if float(peak.get("upper_face_motion_norm", 0.0)) >= 0.75:
                                reasons.append("strong upper-face movement")
                            if float(peak.get("appearance_change_norm", 0.0)) >= 0.75:
                                reasons.append("strong facial appearance change")
                        else:
                            reasons.append("facial motion above baseline")

                        if target_speaker:
                            reasons.append("nearby speaker timeline evidence")

                        events.append({
                            "start": round(start_t, 3),
                            "end": round(end_t, 3),
                            "type": event_type,
                            "face_id": face_id,
                            "target_speaker": target_speaker,
                            "score": round(score, 6),
                            "signals": {
                                "reaction_score": float(peak.get("reaction_score", 0.0)),
                                "facial_motion_score": float(peak.get("facial_motion_score", 0.0)),
                                "mouth_motion": float(peak.get("mouth_region_motion_norm", 0.0)),
                                "upper_face_motion": float(peak.get("upper_face_motion_norm", 0.0)),
                                "appearance_change": float(peak.get("appearance_change_norm", 0.0)),
                                "center_speed": float(peak.get("center_speed_norm", 0.0)),
                                "area_change": float(peak.get("area_change_norm", 0.0)),
                            },
                            "reason": reasons,
                        })

                    run_start = None

        emit_runs(reaction_flags, "reaction_candidate", "reaction_score")
        emit_runs(motion_flags, "facial_motion", "facial_motion_score")

        # Presence transitions.
        for i in range(1, len(obs_list)):
            prev = obs_list[i - 1]
            curr = obs_list[i]
            dt = float(curr["time"]) - float(prev["time"])

            if dt <= 0.75:
                continue

            events.append({
                "start": round(float(curr["time"]), 3),
                "end": round(float(curr["time"]) + DEFAULT_MIN_EVENT, 3),
                "type": "face_entry",
                "face_id": face_id,
                "target_speaker": None,
                "score": 0.62,
                "signals": {"gap_seconds": round(dt, 3)},
                "reason": ["face reappeared after a visual gap"],
            })

    return events


def build_frame_level_visual_events(
    face_timeline: list[dict[str, Any]],
    shots: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []

    for sample in face_timeline:
        t = float(sample["time"])
        fc = int(sample["face_count"])
        change = float(sample["global_change"])

        if change >= FRAME_CHANGE_THRESHOLD:
            events.append({
                "start": round(t, 3),
                "end": round(t + DEFAULT_MIN_EVENT, 3),
                "type": "framing_change",
                "score": round(clamp01(change / 0.75), 6),
                "signals": {
                    "global_change": change,
                    "face_count": fc,
                },
                "reason": ["large visual change between sampled frames"],
            })

    return events


def merge_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not events:
        return []

    events = sorted(
        events,
        key=lambda e: (
            float(e.get("start", 0.0)),
            str(e.get("type", "")),
            str(e.get("face_id", "")),
        ),
    )

    merged: list[dict[str, Any]] = []

    for event in events:
        if not merged:
            merged.append(event)
            continue

        prev = merged[-1]

        same_type = event.get("type") == prev.get("type")
        same_face = event.get("face_id") == prev.get("face_id")

        gap = float(event.get("start", 0.0)) - float(prev.get("end", 0.0))

        if same_type and same_face and gap <= DEFAULT_MERGE_GAP:
            prev["end"] = round(
                max(float(prev["end"]), float(event["end"])),
                3,
            )
            prev["score"] = round(
                max(float(prev.get("score", 0.0)), float(event.get("score", 0.0))),
                6,
            )

            prev_reasons = list(prev.get("reason") or [])
            for reason in event.get("reason") or []:
                if reason not in prev_reasons:
                    prev_reasons.append(reason)
            prev["reason"] = prev_reasons

            prev_signals = dict(prev.get("signals") or {})
            for key, value in (event.get("signals") or {}).items():
                prev_signals[f"peak_{key}"] = value
            prev["signals"] = prev_signals
        else:
            merged.append(event)

    return merged


def add_context(
    events: list[dict[str, Any]],
    shots: list[dict[str, Any]],
    stable: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out = []

    for event in events:
        center = (float(event["start"]) + float(event["end"])) / 2.0
        shot = nearest_shot(shots, center)
        near = nearest_stable(stable, center)

        enriched = dict(event)

        if shot:
            enriched["source_shot_id"] = shot.get("shot_id")
            enriched["source_shot_type"] = shot.get("shot_type")
            enriched["source_shot_start"] = round(float(shot.get("start", 0.0)), 3)
            enriched["source_shot_end"] = round(float(shot.get("end", 0.0)), 3)

        if near:
            enriched["active_speaker"] = (
                near.get("active_speaker") or near.get("speaker")
            )
            if near.get("active_face"):
                enriched["active_face_hint"] = near.get("active_face")

        out.append(enriched)

    return out


def build_timeline(
    face_timeline: list[dict[str, Any]],
    shots: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    timeline = []

    shot_map = shots

    for sample in face_timeline:
        t = float(sample["time"])

        shot = nearest_shot(shot_map, t)
        face_count = int(sample["face_count"])

        max_reaction = 0.0
        max_motion = 0.0
        max_mouth = 0.0
        max_upper = 0.0

        for face in sample["faces"]:
            max_reaction = max(max_reaction, float(face.get("reaction_score", 0.0)))
            max_motion = max(max_motion, float(face.get("facial_motion_score", 0.0)))
            max_mouth = max(max_mouth, float(face.get("mouth_region_motion_norm", 0.0)))
            max_upper = max(max_upper, float(face.get("upper_face_motion_norm", 0.0)))

        timeline.append({
            "time": round(t, 3),
            "face_count": face_count,
            "global_change": float(sample["global_change"]),
            "max_reaction": round(max_reaction, 6),
            "max_facial_motion": round(max_motion, 6),
            "max_mouth_motion": round(max_mouth, 6),
            "max_upper_face_motion": round(max_upper, 6),
            "source_shot_id": shot.get("shot_id") if shot else None,
            "source_shot_type": shot.get("shot_type") if shot else None,
        })

    return timeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual / Reaction Intelligence v1")
    parser.add_argument("--video", required=True)
    parser.add_argument("--shots", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stable", default=None)
    parser.add_argument("--yunet", default=None)

    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--sample", type=float, default=DEFAULT_SAMPLE_INTERVAL)
    parser.add_argument("--face-interval", type=float, default=DEFAULT_FACE_INTERVAL)

    args = parser.parse_args()

    shot_data = load_json(args.shots)
    shots = load_shot_inventory(shot_data)

    stable_data = load_json(args.stable) if args.stable else None
    stable = load_stable_timeline(stable_data)

    detector = YuNetDetector(args.yunet)

    # Probe duration before deciding the default end.
    cap_probe = cv2.VideoCapture(args.video)
    if not cap_probe.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = float(cap_probe.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = float(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    duration = frame_count / fps if fps > 0 else 0.0
    cap_probe.release()

    start = max(0.0, float(args.start))
    end = duration if args.end is None else min(duration, float(args.end))
    if end <= start:
        raise ValueError("End must be greater than start")

    face_timeline, all_obs, video_meta = analyze_video(
        video_path=args.video,
        start=start,
        end=end,
        sample_interval=max(0.10, float(args.sample)),
        face_interval=max(0.10, float(args.face_interval)),
        detector=detector,
    )

    events = []
    events.extend(timeline_shot_events(shots))
    events.extend(build_face_events(face_timeline, stable))
    events.extend(build_frame_level_visual_events(face_timeline, shots))

    events = [
        e for e in events
        if float(e.get("start", 0.0)) >= start - 1e-6
        and float(e.get("start", 0.0)) < end + 1e-6
    ]

    events = merge_events(events)
    events = add_context(events, shots, stable)
    events.sort(key=lambda e: (float(e["start"]), str(e.get("type", ""))))

    reaction_events = [e for e in events if e.get("type") == "reaction_candidate"]
    motion_events = [e for e in events if e.get("type") == "facial_motion"]
    visual_events = [e for e in events if e.get("type") == "visual_insert"]

    face_ids = sorted({str(o["face_id"]) for o in all_obs})
    summary = {
        "event_count": len(events),
        "reaction_candidates": len(reaction_events),
        "facial_motion_events": len(motion_events),
        "visual_insert_events": len(visual_events),
        "shot_boundary_events": sum(1 for e in events if e.get("type") == "shot_boundary"),
        "framing_change_events": sum(1 for e in events if e.get("type") == "framing_change"),
        "face_entries": sum(1 for e in events if e.get("type") == "face_entry"),
        "face_ids": face_ids,
        "strong_reactions": sum(
            1 for e in reaction_events if float(e.get("score", 0.0)) >= 0.75
        ),
    }

    payload = {
        "version": "1.0",
        "mode": "visual_reaction_intelligence",
        "video": args.video,
        "window": {
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
        },
        "models": {
            "face_detector": video_meta["face_backend"],
            "yunet": args.yunet,
        },
        "sampling": {
            "sample_interval": float(args.sample),
            "face_interval": float(args.face_interval),
        },
        "video": {
            **video_meta,
            "duration": round(duration, 3),
        },
        "summary": summary,
        "events": events,
        "timeline": build_timeline(face_timeline, shots),
        "face_observations": face_timeline,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps({
        "output": args.output,
        "window": [round(start, 3), round(end, 3)],
        "face_backend": video_meta["face_backend"],
        "face_ids": len(face_ids),
        "events": len(events),
        "reaction_candidates": len(reaction_events),
        "strong_reactions": summary["strong_reactions"],
        "facial_motion_events": len(motion_events),
        "visual_insert_events": len(visual_events),
        "shot_boundary_events": summary["shot_boundary_events"],
        "framing_change_events": summary["framing_change_events"],
    }, indent=2))


if __name__ == "__main__":
    main()
