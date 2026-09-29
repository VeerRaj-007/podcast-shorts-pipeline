from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_SAMPLE_INTERVAL = 0.25
DEFAULT_FACE_INTERVAL = 0.50
DEFAULT_MIN_SHOT = 0.75


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def robust_threshold(values: list[float], floor: float = 0.48) -> float:
    if not values:
        return floor
    med = statistics.median(values)
    deviations = [abs(v - med) for v in values]
    mad = statistics.median(deviations) if deviations else 0.0
    # Keep enough sensitivity for clean hard cuts while filtering normal motion.
    return min(0.88, max(floor, med + 5.0 * max(mad, 0.015)))


def hsv_hist(frame: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def visual_change(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    ha = hsv_hist(a)
    hb = hsv_hist(b)
    hist_d = float(cv2.compareHist(ha, hb, cv2.HISTCMP_BHATTACHARYYA))

    aa = cv2.resize(a, (160, 90), interpolation=cv2.INTER_AREA)
    bb = cv2.resize(b, (160, 90), interpolation=cv2.INTER_AREA)
    aa = cv2.cvtColor(aa, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bb = cv2.cvtColor(bb, cv2.COLOR_BGR2GRAY).astype(np.float32)
    pixel_d = float(np.mean(np.abs(aa - bb)) / 255.0)

    # Histogram is more reliable for edits/cuts; pixels help with same-colour hard cuts.
    score = 0.72 * clamp01(hist_d) + 0.28 * clamp01(pixel_d * 3.0)
    return score, hist_d


def local_peaks(scores: list[float], threshold: float, min_gap_frames: int) -> list[int]:
    peaks: list[int] = []
    last = -10**9
    for i in range(1, len(scores) - 1):
        s = scores[i]
        if s < threshold:
            continue
        if s < scores[i - 1] or s < scores[i + 1]:
            continue
        if i - last < min_gap_frames:
            if peaks and s > scores[peaks[-1]]:
                peaks[-1] = i
                last = i
            continue
        peaks.append(i)
        last = i
    return peaks


class FaceAnalyzer:
    def __init__(self, model_path: Path | None) -> None:
        self.detector = None
        self.backend = "none"
        self.model_path = str(model_path) if model_path else None

        if model_path and model_path.exists():
            try:
                self.detector = cv2.FaceDetectorYN.create(
                    str(model_path),
                    "",
                    (320, 320),
                    score_threshold=0.55,
                    nms_threshold=0.45,
                    top_k=5000,
                )
                self.backend = "yunet"
                return
            except Exception:
                self.detector = None

        cascade_path = getattr(cv2.data, "haarcascades", "")
        if cascade_path:
            path = Path(cascade_path) / "haarcascade_frontalface_default.xml"
            if path.exists():
                cascade = cv2.CascadeClassifier(str(path))
                if not cascade.empty():
                    self.detector = cascade
                    self.backend = "haar"

    def detect(self, frame: np.ndarray) -> list[dict[str, float]]:
        if self.detector is None:
            return []
        h, w = frame.shape[:2]
        max_dim = 720
        scale = min(1.0, max_dim / max(w, h))
        det_frame = frame if scale == 1.0 else cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        dh, dw = det_frame.shape[:2]

        if self.backend == "yunet":
            self.detector.setInputSize((dw, dh))
            _, faces = self.detector.detect(det_frame)
            if faces is None:
                return []
            result = []
            for row in faces:
                x, y, fw, fh = [float(v) / scale for v in row[:4]]
                x = max(0.0, min(x, w - 1))
                y = max(0.0, min(y, h - 1))
                fw = max(1.0, min(fw, w - x))
                fh = max(1.0, min(fh, h - y))
                score = float(row[-1]) if len(row) >= 15 else 1.0
                result.append({
                    "x": x,
                    "y": y,
                    "width": fw,
                    "height": fh,
                    "area_ratio": (fw * fh) / (w * h),
                    "center_x": (x + fw / 2.0) / w,
                    "center_y": (y + fh / 2.0) / h,
                    "score": score,
                })
            return result

        gray = cv2.cvtColor(det_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        boxes = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.10,
            minNeighbors=5,
            minSize=(max(24, w // 30), max(24, h // 30)),
        )
        result = []
        for x, y, fw, fh in boxes:
            x, y, fw, fh = [float(v) / scale for v in (x, y, fw, fh)]
            result.append({
                "x": x,
                "y": y,
                "width": fw,
                "height": fh,
                "area_ratio": float(fw * fh / (w * h)),
                "center_x": float((x + fw / 2.0) / w),
                "center_y": float((y + fh / 2.0) / h),
                "score": 1.0,
            })
        return result


def shot_type(face_counts: list[int], face_coverages: list[float], no_face_ratio: float) -> str:
    if not face_counts:
        return "unknown"
    avg_faces = statistics.mean(face_counts)
    avg_cov = statistics.mean(face_coverages) if face_coverages else 0.0

    # Preserve a useful distinction between multi-person interview framing and close-ups.
    if avg_faces >= 1.6:
        return "two_shot" if avg_faces < 2.8 else "multi_person"
    if avg_faces >= 0.6:
        if avg_cov >= 0.15:
            return "single_closeup"
        if avg_cov >= 0.055:
            return "single_medium"
        return "single_wide"
    if no_face_ratio >= 0.7:
        return "insert_or_screen"
    return "non_face"


def summarize_shot(samples: list[dict[str, Any]], width: int, height: int) -> dict[str, Any]:
    if not samples:
        return {}
    face_counts = [int(s["face_count"]) for s in samples]
    coverages = [float(s["face_coverage"]) for s in samples]
    no_face_ratio = sum(1 for s in samples if s["face_count"] == 0) / len(samples)

    center_x = [
        s["largest_face"]["center_x"]
        for s in samples
        if s.get("largest_face") is not None
    ]
    center_y = [
        s["largest_face"]["center_y"]
        for s in samples
        if s.get("largest_face") is not None
    ]

    def mean_or_none(values: list[float]) -> float | None:
        return float(statistics.mean(values)) if values else None

    return {
        "sample_count": len(samples),
        "shot_type": shot_type(face_counts, coverages, no_face_ratio),
        "avg_face_count": float(statistics.mean(face_counts)),
        "max_face_count": int(max(face_counts)),
        "face_coverage_avg": float(statistics.mean(coverages)),
        "face_coverage_max": float(max(coverages)),
        "no_face_ratio": float(no_face_ratio),
        "largest_face_center_x": mean_or_none(center_x),
        "largest_face_center_y": mean_or_none(center_y),
        "faces_stable": bool(len(face_counts) >= 2 and max(face_counts) - min(face_counts) <= 1),
        "source_width": width,
        "source_height": height,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a source-shot inventory for the AI Video Director.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--sample", type=float, default=DEFAULT_SAMPLE_INTERVAL)
    parser.add_argument("--face-interval", type=float, default=DEFAULT_FACE_INTERVAL)
    parser.add_argument("--min-shot", type=float, default=DEFAULT_MIN_SHOT)
    parser.add_argument("--yunet", default=None)
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0
    start = max(0.0, args.start)
    end = duration if args.end is None else min(duration, args.end)
    if end <= start:
        raise ValueError("--end must be greater than --start")

    # Dense enough for cuts, cheap enough for long videos.
    step_frames = max(1, int(round(args.sample * fps)))
    seek_start = max(0, int(round(start * fps)))
    seek_end = min(frame_count - 1, int(round(end * fps)))
    min_gap_frames = max(1, int(round(args.min_shot / args.sample)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, seek_start)
    prev = None
    times: list[float] = []
    scores: list[float] = []

    i = seek_start
    while i <= seek_end:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        if prev is not None:
            score, _ = visual_change(prev, frame)
            scores.append(score)
            times.append(t)
        prev = frame
        # Skip frames in-place rather than repeatedly seeking.
        for _ in range(step_frames - 1):
            if i >= seek_end:
                break
            ok, _ = cap.read()
            i += 1
            if not ok:
                break
        i += 1

    threshold = robust_threshold(scores)
    peak_idx = local_peaks(scores, threshold, min_gap_frames)
    boundaries = [start] + [times[i] for i in peak_idx] + [end]

    # Remove tiny fragments and de-duplicate nearby boundaries.
    clean = [boundaries[0]]
    for b in boundaries[1:]:
        if b - clean[-1] < args.min_shot and b < end:
            continue
        clean.append(b)
    if clean[-1] != end:
        clean.append(end)

    face_analyzer = FaceAnalyzer(Path(args.yunet) if args.yunet else None)
    face_interval_frames = max(1, int(round(args.face_interval * fps)))

    shots: list[dict[str, Any]] = []
    face_sample_total = 0

    for idx in range(len(clean) - 1):
        shot_start = float(clean[idx])
        shot_end = float(clean[idx + 1])
        if shot_end - shot_start < 0.20:
            continue

        sample_frames: list[dict[str, Any]] = []
        f0 = int(round(shot_start * fps))
        f1 = int(round(shot_end * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        f = f0
        while f <= f1:
            ok, frame = cap.read()
            if not ok:
                break
            t = f / fps
            if face_analyzer.detector is not None:
                faces = face_analyzer.detect(frame)
                face_sample_total += 1
            else:
                faces = []
            coverage = sum(float(x["area_ratio"]) for x in faces)
            largest = max(faces, key=lambda x: x["area_ratio"]) if faces else None
            sample_frames.append({
                "time": round(t, 3),
                "face_count": len(faces),
                "face_coverage": round(float(coverage), 6),
                "faces": faces,
                "largest_face": largest,
            })
            for _ in range(face_interval_frames - 1):
                if f >= f1:
                    break
                ok, _ = cap.read()
                f += 1
                if not ok:
                    break
            f += 1

        summary = summarize_shot(sample_frames, width, height)
        cut_score = None
        if idx > 0:
            # Boundary is one of the detected peak times; use nearest cut score.
            boundary = shot_start
            if times:
                nearest = min(range(len(times)), key=lambda j: abs(times[j] - boundary))
                cut_score = round(float(scores[nearest]), 4)

        shots.append({
            "shot_id": f"SHOT_{idx:03d}",
            "start": round(shot_start, 3),
            "end": round(shot_end, 3),
            "duration": round(shot_end - shot_start, 3),
            "cut_in_score": cut_score,
            **summary,
            "samples": sample_frames,
        })

    cap.release()

    out = {
        "version": "1.0",
        "mode": "source_shot_inventory",
        "video": str(video_path),
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(end - start, 3),
        "fps": fps,
        "width": width,
        "height": height,
        "cut_detection": {
            "sample_interval": args.sample,
            "min_shot": args.min_shot,
            "threshold": round(float(threshold), 4),
            "candidate_cuts": len(peak_idx),
            "final_shots": len(shots),
        },
        "face_detection": {
            "backend": face_analyzer.backend,
            "model": face_analyzer.model_path,
            "interval": args.face_interval,
            "sample_total": face_sample_total,
        },
        "shots": shots,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    counts: dict[str, int] = {}
    for s in shots:
        counts[s["shot_type"]] = counts.get(s["shot_type"], 0) + 1

    print(json.dumps({
        "output": str(out_path),
        "video_duration": round(duration, 3),
        "window": [round(start, 3), round(end, 3)],
        "cut_threshold": round(float(threshold), 4),
        "candidate_cuts": len(peak_idx),
        "final_shots": len(shots),
        "face_backend": face_analyzer.backend,
        "shot_types": counts,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
