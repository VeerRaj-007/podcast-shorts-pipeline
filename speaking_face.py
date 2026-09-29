
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict, deque
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SAMPLE_INTERVAL = 0.25

# A face detection can be slightly away from our exact sample timestamp.
MATCH_TOLERANCE = 0.30

# Relative mouth region inside the detected face.
MOUTH_X = 0.18
MOUTH_Y = 0.52
MOUTH_W = 0.64
MOUTH_H = 0.38

MIN_FACE_SIZE = 32

# Temporal smoothing.
MOTION_EMA_ALPHA = 0.40
SMOOTH_WINDOW = 5

# Speech is only a prior. Mouth motion is what identifies the face.
SPEECH_BASE_WEIGHT = 0.20
MOTION_WEIGHT = 0.80

# Minimum score to consider a face a speaking candidate.
ACTIVE_THRESHOLD = 0.32

# Require at least this much normalized mouth motion when speech exists.
MIN_MOTION_WITH_SPEECH = 0.12

# Ignore extremely tiny motion that is usually compression/noise.
RAW_MOTION_FLOOR = 1.5


# ---------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


# ---------------------------------------------------------------------
# Speaker-track extraction
# ---------------------------------------------------------------------

def _speaker_id(item: Dict[str, Any]) -> Optional[str]:
    for key in (
        "speaker",
        "speaker_id",
        "speakerId",
        "label",
        "id",
        "name",
    ):
        value = item.get(key)
        if value is not None:
            return str(value)
    return None


def _segment_times(item: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    start = as_float(
        item.get("start", item.get("start_time", item.get("begin")))
    )
    end = as_float(
        item.get("end", item.get("end_time", item.get("stop")))
    )

    if start is None or end is None:
        return None

    if end <= start:
        return None

    return start, end


def collect_speaker_segments(node: Any) -> List[Dict[str, Any]]:
    """
    Recursively find objects containing:
      start / end / speaker

    This intentionally supports several possible JSON schemas:
      {
        "segments": [...]
      }

      {
        "tracks": [...]
      }

      {
        "speaker_segments": [...]
      }

      {
        "speakers": [
          {"speaker": "...", "segments": [...]}
        ]
      }

    We recursively inspect nested lists/dicts so the exact top-level
    schema is not critical.
    """

    result: List[Dict[str, Any]] = []
    seen = set()

    def walk(value: Any, inherited_speaker: Optional[str] = None) -> None:
        if isinstance(value, list):
            for item in value:
                walk(item, inherited_speaker)
            return

        if not isinstance(value, dict):
            return

        local_speaker = _speaker_id(value) or inherited_speaker

        times = _segment_times(value)
        if times is not None and local_speaker is not None:
            start, end = times

            key = (
                round(start, 6),
                round(end, 6),
                local_speaker,
            )

            if key not in seen:
                seen.add(key)
                result.append(
                    {
                        "start": start,
                        "end": end,
                        "speaker": local_speaker,
                    }
                )

        for child_key, child_value in value.items():
            # Do not treat metadata as speaker segments.
            if child_key in {
                "version",
                "mode",
                "video",
                "audio",
                "samples",
                "timeline",
                "observations",
            }:
                continue

            walk(child_value, local_speaker)

    walk(node)

    result.sort(key=lambda x: (x["start"], x["end"]))
    return result


def load_speaker_segments(path: str) -> List[Dict[str, Any]]:
    data = load_json(path)
    segments = collect_speaker_segments(data)

    if not segments:
        raise RuntimeError(
            "Could not find any speaker segments in "
            f"{os.path.abspath(path)}. "
            "Expected objects containing start/end plus speaker information."
        )

    return segments


# ---------------------------------------------------------------------
# Speaker activity
# ---------------------------------------------------------------------

def active_speakers_at(
    t: float,
    speaker_segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Return all speakers whose diarization segment contains/near t.

    We keep all active speakers because overlaps can occur.
    """

    active: List[Dict[str, Any]] = []

    window_start = t - 0.10
    window_end = t + 0.10

    for seg in speaker_segments:
        s = float(seg["start"])
        e = float(seg["end"])

        ov = overlap(s, e, window_start, window_end)

        if ov > 0.0:
            active.append(
                {
                    "speaker": seg["speaker"],
                    "overlap": round(float(ov), 4),
                    "start": s,
                    "end": e,
                }
            )

    active.sort(key=lambda x: x["overlap"], reverse=True)
    return active


# ---------------------------------------------------------------------
# Face detections
# ---------------------------------------------------------------------

def build_detection_index(
    tracks: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:

    output: Dict[str, List[Dict[str, Any]]] = {}

    raw_tracks = tracks.get("tracks", [])

    if not isinstance(raw_tracks, list):
        return output

    for track in raw_tracks:
        if not isinstance(track, dict):
            continue

        face_id = track.get("face_id")

        if face_id is None:
            face_id = track.get("id")

        if face_id is None:
            continue

        face_id = str(face_id)

        detections = track.get("detections", [])

        if not isinstance(detections, list):
            continue

        cleaned = []

        for detection in detections:
            if not isinstance(detection, dict):
                continue

            t = as_float(detection.get("time"))

            if t is None:
                continue

            cleaned.append(detection)

        cleaned.sort(key=lambda d: float(d["time"]))

        if cleaned:
            output[face_id] = cleaned

    return output


def nearest_detection(
    detections: List[Dict[str, Any]],
    t: float,
) -> Optional[Dict[str, Any]]:

    if not detections:
        return None

    best = min(
        detections,
        key=lambda d: abs(float(d["time"]) - t),
    )

    distance = abs(float(best["time"]) - t)

    if distance > MATCH_TOLERANCE:
        return None

    return best


# ---------------------------------------------------------------------
# Mouth analysis
# ---------------------------------------------------------------------

def mouth_crop(
    frame: np.ndarray,
    detection: Dict[str, Any],
) -> Optional[np.ndarray]:

    x = as_float(detection.get("x"))
    y = as_float(detection.get("y"))
    w = as_float(detection.get("width"))
    h = as_float(detection.get("height"))

    if None in (x, y, w, h):
        return None

    assert x is not None
    assert y is not None
    assert w is not None
    assert h is not None

    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
        return None

    x0 = max(
        0,
        int(round(x + MOUTH_X * w)),
    )

    y0 = max(
        0,
        int(round(y + MOUTH_Y * h)),
    )

    x1 = min(
        frame.shape[1],
        int(round(x + (MOUTH_X + MOUTH_W) * w)),
    )

    y1 = min(
        frame.shape[0],
        int(round(y + (MOUTH_Y + MOUTH_H) * h)),
    )

    if x1 <= x0 or y1 <= y0:
        return None

    crop = frame[y0:y1, x0:x1]

    if crop.size == 0:
        return None

    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    crop = cv2.resize(
        crop,
        (96, 64),
        interpolation=cv2.INTER_AREA,
    )

    # Normalize brightness slightly so exposure changes have
    # less influence on frame-to-frame motion.
    crop = cv2.GaussianBlur(
        crop,
        (5, 5),
        0,
    )

    return crop


def mouth_motion(
    previous: Optional[np.ndarray],
    current: Optional[np.ndarray],
) -> float:

    if previous is None or current is None:
        return 0.0

    diff = cv2.absdiff(previous, current).astype(np.float32)

    # Ignore the noisiest pixels.
    p95 = float(np.percentile(diff, 95))
    p50 = float(np.percentile(diff, 50))

    value = max(0.0, p95 - p50)

    if value < RAW_MOTION_FLOOR:
        return 0.0

    return value


# ---------------------------------------------------------------------
# Motion normalization
# ---------------------------------------------------------------------

def normalize_track_scores(
    rows: List[Dict[str, Any]],
) -> None:

    by_face: Dict[str, List[float]] = defaultdict(list)

    for row in rows:
        by_face[str(row["face_id"])].append(
            float(row["mouth_motion_raw"])
        )

    for face_id, values in by_face.items():

        if not values:
            continue

        # Robust per-face baseline.
        lo = float(np.percentile(values, 25))
        hi = float(np.percentile(values, 90))

        if hi <= lo + 1e-9:
            hi = lo + 1.0

        for row in rows:
            if str(row["face_id"]) != face_id:
                continue

            raw = float(row["mouth_motion_raw"])

            normalized = np.clip(
                (raw - lo) / (hi - lo),
                0.0,
                1.0,
            )

            row["mouth_motion"] = round(
                float(normalized),
                4,
            )


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------

def smooth(values: deque[float]) -> float:
    if not values:
        return 0.0

    return float(sum(values) / len(values))


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------

def analyze(
    video_path: str,
    face_tracks_path: str,
    speaker_tracks_path: str,
    start: float,
    end: float,
) -> Dict[str, Any]:

    faces = load_json(face_tracks_path)
    speaker_segments = load_speaker_segments(
        speaker_tracks_path
    )

    detections = build_detection_index(faces)

    if not detections:
        raise RuntimeError(
            "No face tracks found in "
            f"{os.path.abspath(face_tracks_path)}"
        )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    rows: List[Dict[str, Any]] = []

    previous_mouth: Dict[
        str,
        Optional[np.ndarray],
    ] = {}

    ema_motion: Dict[str, float] = {}

    histories: Dict[
        str,
        deque[float],
    ] = defaultdict(
        lambda: deque(maxlen=SMOOTH_WINDOW)
    )

    samples = 0

    t = float(start)

    while t < end - 1e-6:

        frame_index = max(
            0,
            int(round(t * fps)),
        )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_index,
        )

        ok, frame = cap.read()

        if not ok or frame is None:
            break

        active_speakers = active_speakers_at(
            t,
            speaker_segments,
        )

        best_speaker = (
            active_speakers[0]["speaker"]
            if active_speakers
            else None
        )

        speech_active = bool(active_speakers)

        for face_id, face_detections in detections.items():

            detection = nearest_detection(
                face_detections,
                t,
            )

            if detection is None:

                previous_mouth[face_id] = None
                histories[face_id].clear()
                ema_motion.pop(face_id, None)

                continue

            mouth = mouth_crop(
                frame,
                detection,
            )

            raw_motion = mouth_motion(
                previous_mouth.get(face_id),
                mouth,
            )

            previous_mouth[face_id] = mouth

            if face_id not in ema_motion:
                ema = raw_motion
            else:
                ema = (
                    (1.0 - MOTION_EMA_ALPHA)
                    * ema_motion[face_id]
                    + MOTION_EMA_ALPHA
                    * raw_motion
                )

            ema_motion[face_id] = float(ema)

            histories[face_id].append(
                float(ema)
            )

            smoothed_raw = smooth(
                histories[face_id]
            )

            rows.append(
                {
                    "time": round(float(t), 3),
                    "face_id": face_id,
                    "mouth_motion_raw": round(
                        float(smoothed_raw),
                        4,
                    ),
                    "mouth_motion": 0.0,
                    "speech_active": speech_active,
                    "active_speaker": best_speaker,
                    "active_speakers": [
                        s["speaker"]
                        for s in active_speakers
                    ],
                }
            )

        samples += 1
        t += SAMPLE_INTERVAL

    cap.release()

    normalize_track_scores(rows)

    # -----------------------------------------------------------------
    # Build timeline
    # -----------------------------------------------------------------

    timeline: List[Dict[str, Any]] = []

    by_time: Dict[
        float,
        List[Dict[str, Any]],
    ] = defaultdict(list)

    for row in rows:
        by_time[
            float(row["time"])
        ].append(row)

    for timestamp, frame_rows in sorted(
        by_time.items()
    ):

        candidates = []

        for row in frame_rows:

            speech = bool(
                row["speech_active"]
            )

            motion = float(
                row["mouth_motion"]
            )

            # The key idea:
            #
            # speech is only a prior.
            #
            # Without visible mouth motion, a face should NOT
            # automatically become the speaking face merely
            # because somebody is talking somewhere in the frame.
            if speech and motion >= MIN_MOTION_WITH_SPEECH:
                speech_component = (
                    SPEECH_BASE_WEIGHT
                )
            else:
                speech_component = 0.0

            motion_component = (
                MOTION_WEIGHT * motion
            )

            score = (
                speech_component
                + motion_component
            )

            candidates.append(
                {
                    "face_id": row["face_id"],
                    "score": round(
                        float(score),
                        4,
                    ),
                    "mouth_motion": round(
                        motion,
                        4,
                    ),
                    "speech_active": speech,
                    "speaker": row["active_speaker"],
                }
            )

        candidates.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        active = None

        if candidates:
            strongest = candidates[0]

            if (
                strongest["score"]
                >= ACTIVE_THRESHOLD
            ):
                active = strongest

        timeline.append(
            {
                "time": timestamp,
                "active_face": (
                    active["face_id"]
                    if active
                    else None
                ),
                "active_speaker": (
                    active["speaker"]
                    if active
                    else None
                ),
                "confidence": round(
                    float(
                        active["score"]
                    )
                    if active
                    else 0.0,
                    4,
                ),
                "candidates": candidates,
            }
        )

    # -----------------------------------------------------------------
    # Per-face summary
    # -----------------------------------------------------------------

    per_face: Dict[
        str,
        Dict[str, Any],
    ] = {}

    for face_id in detections:

        face_rows = [
            row
            for row in rows
            if row["face_id"] == face_id
        ]

        speaking_rows = 0
        speech_rows = 0
        motion_rows = 0

        scores: List[float] = []

        for row in face_rows:

            motion = float(
                row["mouth_motion"]
            )

            speech = bool(
                row["speech_active"]
            )

            if speech:
                speech_rows += 1

            if motion >= MIN_MOTION_WITH_SPEECH:
                motion_rows += 1

            if speech and motion >= MIN_MOTION_WITH_SPEECH:
                speech_component = (
                    SPEECH_BASE_WEIGHT
                )
            else:
                speech_component = 0.0

            score = (
                speech_component
                + MOTION_WEIGHT * motion
            )

            scores.append(score)

            if score >= ACTIVE_THRESHOLD:
                speaking_rows += 1

        max_score = (
            max(scores)
            if scores
            else 0.0
        )

        mean_score = (
            float(np.mean(scores))
            if scores
            else 0.0
        )

        per_face[face_id] = {
            "face_id": face_id,
            "samples": len(face_rows),
            "speech_samples": speech_rows,
            "motion_samples": motion_rows,
            "speaking_samples": speaking_rows,
            "speaking_fraction": round(
                speaking_rows / len(face_rows),
                4,
            )
            if face_rows
            else 0.0,
            "mean_score": round(
                mean_score,
                4,
            ),
            "max_score": round(
                max_score,
                4,
            ),
        }

    return {
        "version": 2,
        "mode": "speaking_face_detection",
        "video": os.path.abspath(video_path),
        "start": float(start),
        "end": float(end),
        "sample_interval": SAMPLE_INTERVAL,
        "weights": {
            "speech_base": SPEECH_BASE_WEIGHT,
            "mouth_motion": MOTION_WEIGHT,
        },
        "active_threshold": ACTIVE_THRESHOLD,
        "min_motion_with_speech": MIN_MOTION_WITH_SPEECH,
        "raw_motion_floor": RAW_MOTION_FLOOR,
        "samples": samples,
        "face_ids": sorted(
            detections.keys()
        ),
        "speaker_segments": len(
            speaker_segments
        ),
        "speakers": sorted(
            {
                str(seg["speaker"])
                for seg in speaker_segments
            }
        ),
        "per_face": per_face,
        "timeline": timeline,
        "observations": rows,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Speaking-face detection v2"
        )
    )

    parser.add_argument(
        "--video",
        required=True,
    )

    parser.add_argument(
        "--faces",
        required=True,
    )

    parser.add_argument(
        "--speakers",
        required=True,
    )

    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--end",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    return parser.parse_args()


def main() -> int:

    args = parse_args()

    result = analyze(
        args.video,
        args.faces,
        args.speakers,
        args.start,
        args.end,
    )

    output_path = os.path.abspath(
        args.output
    )

    os.makedirs(
        os.path.dirname(output_path)
        or ".",
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )

    print(
        f"Speaking-face samples: "
        f"{result['samples']}"
    )

    print(
        f"Faces analyzed: "
        f"{len(result['face_ids'])}"
    )

    print(
        f"Speaker segments: "
        f"{result['speaker_segments']}"
    )

    print(
        f"Speakers detected: "
        f"{len(result['speakers'])}"
    )

    print()

    for face_id, info in result[
        "per_face"
    ].items():

        print(
            f"  {face_id}: "
            f"speaking_fraction="
            f"{info['speaking_fraction']:.3f} "
            f"mean="
            f"{info['mean_score']:.3f} "
            f"max="
            f"{info['max_score']:.3f}"
        )

    print()
    print(
        f"Saved: {output_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
