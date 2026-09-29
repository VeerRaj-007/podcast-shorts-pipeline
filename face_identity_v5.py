from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


SAMPLE_INTERVAL = 0.20
NMS_IOU_THRESHOLD = 0.55
DUPLICATE_CENTER_THRESHOLD = 0.035
DUPLICATE_SIZE_RATIO = 1.8
MAX_NEW_TRACK_DISTANCE = 0.10
MAX_MISSED_SAMPLES = 10
MIN_FACE_SIZE = 24
DETECTION_SCORE = 0.60
TRACK_MATCH_THRESHOLD = 0.46
REID_THRESHOLD = 0.34
MAX_REID_GAP = 3.0
MERGE_MAX_GAP = 2.5
MIN_TRACK_SAMPLES = 4
MIN_TRACK_DURATION = 0.75
SHOT_CHANGE_THRESHOLD = 0.42
HIST_WEIGHT = 0.10
EMBED_WEIGHT = 0.45
CENTER_WEIGHT = 0.25
SIZE_WEIGHT = 0.10
MOTION_WEIGHT = 0.10


@dataclass
class Detection:
    x: float
    y: float
    w: float
    h: float
    score: float
    landmarks: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None
    histogram: Optional[np.ndarray] = None

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def area(self) -> float:
        return max(self.w * self.h, 1.0)


@dataclass
class Track:
    track_id: int
    start: float
    end: float
    last_time: float
    last_det: Detection
    missed: int = 0
    detections: List[Dict] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    histogram: Optional[np.ndarray] = None
    velocity: Tuple[float, float] = (0.0, 0.0)
    shot_id: int = 0

    def update(self, t: float, det: Detection) -> None:
        dt = max(t - self.last_time, 1e-6)
        self.velocity = (
            (det.cx - self.last_det.cx) / dt,
            (det.cy - self.last_det.cy) / dt,
        )
        self.last_det = det
        self.last_time = t
        self.end = t
        self.missed = 0
        self.detections.append(det_to_dict(t, det))
        self.embedding = blend_feature(self.embedding, det.embedding, 0.18)
        self.histogram = blend_feature(self.histogram, det.histogram, 0.15)



def blend_feature(old: Optional[np.ndarray], new: Optional[np.ndarray], alpha: float) -> Optional[np.ndarray]:
    if new is None:
        return old
    if old is None:
        out = new.astype(np.float32, copy=True)
    else:
        out = (1.0 - alpha) * old + alpha * new
    norm = float(np.linalg.norm(out))
    return out / norm if norm > 1e-8 else out



def det_to_dict(t: float, d: Detection) -> Dict:
    return {
        "time": round(float(t), 3),
        "x": round(float(d.x), 2),
        "y": round(float(d.y), 2),
        "width": round(float(d.w), 2),
        "height": round(float(d.h), 2),
        "center_x": round(float(d.cx), 2),
        "center_y": round(float(d.cy), 2),
        "score": round(float(d.score), 4),
        "has_embedding": d.embedding is not None,
    }



def crop_safe(frame: np.ndarray, x: float, y: float, w: float, h: float) -> Optional[np.ndarray]:
    H, W = frame.shape[:2]
    x0 = max(0, int(round(x)))
    y0 = max(0, int(round(y)))
    x1 = min(W, int(round(x + w)))
    y1 = min(H, int(round(y + h)))
    if x1 <= x0 or y1 <= y0:
        return None
    return frame[y0:y1, x0:x1]



def compute_histogram(face: np.ndarray) -> Optional[np.ndarray]:
    if face is None or face.size == 0:
        return None
    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    hist = cv2.normalize(hist, None).flatten().astype(np.float32)
    return hist



def load_sface(model_path: Optional[str]):
    if not model_path or not os.path.exists(model_path):
        return None
    try:
        return cv2.FaceRecognizerSF.create(model_path, "")
    except Exception:
        return None



def normalize_feature(feature: np.ndarray) -> np.ndarray:
    feature = np.asarray(feature, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(feature))
    return feature / norm if norm > 1e-8 else feature



def compute_embedding(recognizer, frame: np.ndarray, row: np.ndarray) -> Optional[np.ndarray]:
    if recognizer is None:
        return None
    try:
        aligned = recognizer.alignCrop(frame, row)
        feature = recognizer.feature(aligned)
        return normalize_feature(feature)
    except Exception:
        return None



def detect_yunet(detector, recognizer, frame: np.ndarray) -> List[Detection]:
    H, W = frame.shape[:2]
    detector.setInputSize((W, H))
    _, faces = detector.detect(frame)
    if faces is None:
        return []

    detections: List[Detection] = []
    for row in faces:
        x, y, w, h, score = [float(v) for v in row[:5]]
        if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE or score < DETECTION_SCORE:
            continue
        d = Detection(
            x=x,
            y=y,
            w=w,
            h=h,
            score=score,
            landmarks=np.asarray(row[5:15], dtype=np.float32) if len(row) >= 15 else None,
        )
        face = crop_safe(frame, x, y, w, h)
        d.histogram = compute_histogram(face)
        d.embedding = compute_embedding(recognizer, frame, np.asarray(row, dtype=np.float32))
        detections.append(d)
    return detections



def cosine_distance(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 1.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 1.0
    return float(1.0 - np.dot(a, b) / denom)



def histogram_distance(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 1.0
    return float(cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA))



def center_distance(pred_cx: float, pred_cy: float, det: Detection, W: int, H: int) -> float:
    dx = (pred_cx - det.cx) / max(W, 1)
    dy = (pred_cy - det.cy) / max(H, 1)
    return min(1.0, math.hypot(dx, dy) / 0.22)



def size_distance(last: Detection, det: Detection) -> float:
    return min(1.0, abs(math.log(last.area / det.area)) / math.log(3.0))



def association_cost(track: Track, det: Detection, t: float, W: int, H: int) -> Tuple[float, Dict[str, float]]:
    dt = max(t - track.last_time, 0.0)
    pred_cx = track.last_det.cx + track.velocity[0] * dt
    pred_cy = track.last_det.cy + track.velocity[1] * dt
    c = center_distance(pred_cx, pred_cy, det, W, H)
    s = size_distance(track.last_det, det)
    e = cosine_distance(track.embedding, det.embedding) if track.embedding is not None and det.embedding is not None else 0.5
    h = histogram_distance(track.histogram, det.histogram) if track.histogram is not None and det.histogram is not None else 0.5
    m = min(1.0, dt / MAX_REID_GAP)
    cost = CENTER_WEIGHT * c + SIZE_WEIGHT * s + EMBED_WEIGHT * e + HIST_WEIGHT * h + MOTION_WEIGHT * m
    return cost, {"center": c, "size": s, "embedding": e, "histogram": h, "gap": m}



def greedy_match(tracks: List[Track], detections: List[Detection], t: float, W: int, H: int):
    candidates = []
    for ti, track in enumerate(tracks):
        if track.missed > MAX_MISSED_SAMPLES:
            continue
        if t - track.last_time > MAX_REID_GAP:
            continue
        for di, det in enumerate(detections):
            cost, parts = association_cost(track, det, t, W, H)
            threshold = TRACK_MATCH_THRESHOLD if t - track.last_time <= SAMPLE_INTERVAL * 1.5 else REID_THRESHOLD
            if cost <= threshold:
                candidates.append((cost, ti, di, parts))
    candidates.sort(key=lambda x: x[0])
    used_t, used_d, chosen = set(), set(), []
    for item in candidates:
        _, ti, di, parts = item
        if ti in used_t or di in used_d:
            continue
        used_t.add(ti)
        used_d.add(di)
        chosen.append(item)
    return chosen, used_t, used_d



def estimate_shot_change(prev_frame: Optional[np.ndarray], frame: np.ndarray) -> float:
    if prev_frame is None or prev_frame.shape != frame.shape:
        return 0.0
    a = cv2.resize(prev_frame, (160, 90), interpolation=cv2.INTER_AREA)
    b = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return float(np.mean(np.abs(a - b)))



def iou(a: Dict, b: Dict) -> float:
    ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"] + a["width"], a["y"] + a["height"]
    bx1, by1, bx2, by2 = b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a["width"] * a["height"] + b["width"] * b["height"] - inter
    return inter / union if union > 0 else 0.0



def rect_iou(a: Detection, b: Detection) -> float:
    ax1, ay1 = a.x, a.y
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx1, by1 = b.x, b.y
    bx2, by2 = b.x + b.w, b.y + b.h
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def suppress_duplicate_detections(detections: List[Detection]) -> List[Detection]:
    if len(detections) < 2:
        return detections
    order = sorted(range(len(detections)), key=lambda i: detections[i].score, reverse=True)
    kept: List[Detection] = []
    for idx in order:
        d = detections[idx]
        duplicate = False
        for k in kept:
            iou_v = rect_iou(d, k)
            if iou_v >= NMS_IOU_THRESHOLD:
                duplicate = True
                break
            diag = math.hypot(max(k.w, 1.0), max(k.h, 1.0))
            center = math.hypot(d.cx - k.cx, d.cy - k.cy) / max(diag, 1.0)
            size_ratio = max(d.area, k.area) / max(min(d.area, k.area), 1.0)
            if center <= DUPLICATE_CENTER_THRESHOLD and size_ratio <= DUPLICATE_SIZE_RATIO:
                duplicate = True
                break
        if not duplicate:
            kept.append(d)
    return kept


def too_close_to_existing_track(det: Detection, tracks: List[Track], t: float, W: int, H: int) -> bool:
    for tr in tracks:
        if tr.missed > 2 or (t - tr.last_time) > SAMPLE_INTERVAL * 2.5:
            continue
        if rect_iou(det, tr.last_det) >= 0.40:
            return True
        center = math.hypot(det.cx - tr.last_det.cx, det.cy - tr.last_det.cy) / max(math.hypot(W, H), 1.0)
        size_ratio = max(det.area, tr.last_det.area) / max(min(det.area, tr.last_det.area), 1.0)
        if center <= MAX_NEW_TRACK_DISTANCE and size_ratio <= 2.0:
            return True
    return False


def merge_fragments(tracks: List[Track], max_gap: float) -> List[Track]:
    if len(tracks) < 2:
        return tracks
    tracks = sorted(tracks, key=lambda t: (t.start, t.track_id))
    active = tracks[:]
    changed = True
    while changed:
        changed = False
        for i in range(len(active)):
            merged_this_round = False
            for j in range(i + 1, len(active)):
                a, b = active[i], active[j]
                gap = b.start - a.end
                if gap < 0 or gap > max_gap:
                    continue
                if a.shot_id != b.shot_id:
                    continue
                if a.embedding is not None and b.embedding is not None:
                    if cosine_distance(a.embedding, b.embedding) > REID_THRESHOLD:
                        continue
                elif a.detections and b.detections:
                    if iou(a.detections[-1], b.detections[0]) < 0.08:
                        continue
                active[i] = merge_track_pair(a, b)
                del active[j]
                changed = True
                merged_this_round = True
                break
            if merged_this_round:
                break
    return active


def merge_track_pair(a: Track, b: Track) -> Track:
    if b.start < a.start:
        a, b = b, a
    a.end = max(a.end, b.end)
    a.detections.extend(b.detections)
    a.embedding = blend_feature(a.embedding, b.embedding, 0.35)
    a.histogram = blend_feature(a.histogram, b.histogram, 0.30)
    a.last_det = b.last_det
    a.last_time = max(a.last_time, b.last_time)
    return a


def track_video(video_path: str, model_path: str, sface_path: Optional[str], start: float, end: float) -> Dict:
    detector = cv2.FaceDetectorYN.create(
        model_path, "", (320, 320), DETECTION_SCORE, 0.30, 5000
    )
    recognizer = load_sface(sface_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if W <= 0 or H <= 0:
        raise RuntimeError("Could not determine video dimensions")

    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
    tracks: List[Track] = []
    next_id = 0
    t = start
    samples = 0
    shot_boundaries: List[float] = []
    prev_frame = None
    shot_id = 0

    while t < end - 1e-6:
        frame_idx = max(0, int(round(t * fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        shot_score = estimate_shot_change(prev_frame, frame)
        if prev_frame is not None and shot_score >= SHOT_CHANGE_THRESHOLD:
            shot_id += 1
            shot_boundaries.append(round(t, 3))
            for tr in tracks:
                tr.missed = MAX_MISSED_SAMPLES + 1

        detections = suppress_duplicate_detections(detect_yunet(detector, recognizer, frame))
        chosen, used_t, used_d = greedy_match(tracks, detections, t, W, H)

        for _, ti, di, _ in chosen:
            tracks[ti].shot_id = shot_id
            tracks[ti].update(t, detections[di])

        for ti, tr in enumerate(tracks):
            if ti not in used_t:
                tr.missed += 1

        for di, det in enumerate(detections):
            if di in used_d:
                continue
            if too_close_to_existing_track(det, tracks, t, W, H):
                continue
            tracks.append(
                Track(
                    track_id=next_id,
                    start=t,
                    end=t,
                    last_time=t,
                    last_det=det,
                    detections=[det_to_dict(t, det)],
                    embedding=det.embedding,
                    histogram=det.histogram,
                    shot_id=shot_id,
                )
            )
            next_id += 1

        samples += 1
        prev_frame = frame
        t += SAMPLE_INTERVAL

    cap.release()

    finished = [
        tr for tr in tracks
        if tr.detections
        and len(tr.detections) >= MIN_TRACK_SAMPLES
        and (tr.end - tr.start) >= MIN_TRACK_DURATION
    ]
    finished = merge_fragments(finished, MERGE_MAX_GAP)

    # Drop pathological fragments that still survived after duplicate suppression.
    finished = [
        tr for tr in finished
        if len(tr.detections) >= MIN_TRACK_SAMPLES
        and (tr.end - tr.start) >= MIN_TRACK_DURATION
    ]
    finished.sort(key=lambda tr: (-len(tr.detections), tr.start, tr.track_id))

    output_tracks = []
    for out_id, tr in enumerate(finished):
        tr.track_id = out_id
        output_tracks.append({
            "face_id": f"FACE_{out_id:02d}",
            "track_id": out_id,
            "shot_id": tr.shot_id,
            "start": round(tr.start, 3),
            "end": round(tr.end, 3),
            "sample_count": len(tr.detections),
            "has_appearance_embedding": tr.embedding is not None,
            "detections": tr.detections,
        })

    return {
        "version": 5,
        "mode": "persistent_face_tracking",
        "sample_interval": SAMPLE_INTERVAL,
        "appearance_model": "SFace" if recognizer is not None else "fallback_histogram",
        "video_width": W,
        "video_height": H,
        "fps": fps,
        "start": start,
        "end": end,
        "samples": samples,
        "shot_boundaries": shot_boundaries,
        "tracks": output_tracks,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Persistent face identity tracking v5")
    p.add_argument("--project-root", default=".")
    p.add_argument("--video", required=True)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--sface-model", default=None)
    return p.parse_args()



def main() -> int:
    args = parse_args()
    root = os.path.abspath(args.project_root)
    video_path = os.path.abspath(args.video)
    output_path = os.path.abspath(args.output)

    model_path = args.model or os.path.join(root, "models", "face_detection_yunet_2023mar.onnx")
    if not os.path.exists(model_path):
        raise RuntimeError(f"YuNet model not found: {model_path}")

    sface_path = args.sface_model or os.path.join(root, "models", "face_recognition_sface_2021dec.onnx")
    if not os.path.exists(sface_path):
        sface_path = None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    duration = frame_count / fps if fps > 0 else 0.0
    cap.release()

    end = args.end if args.end is not None else duration
    if end <= args.start:
        raise ValueError("--end must be greater than --start")

    result = track_video(video_path, model_path, sface_path, args.start, end)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Persistent face tracks: {len(result['tracks'])}")
    print(f"Samples: {result['samples']}")
    print(f"Shot boundaries: {len(result['shot_boundaries'])}")
    print(f"Appearance model: {result['appearance_model']}")
    for tr in result["tracks"]:
        print(f"  {tr['face_id']}: {tr['start']:.2f}s -> {tr['end']:.2f}s ({tr['sample_count']} samples)")
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
