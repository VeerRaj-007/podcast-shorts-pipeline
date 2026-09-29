import json
import os
import sys
import urllib.request
from statistics import median
from typing import Dict, List, Optional, Tuple

import cv2


VIDEO_PATH = os.path.join("tmp", "source.mp4")
CAMERA_PLAN_PATH = os.path.join("tmp", "camera_plan.json")
OUTPUT_PATH = os.path.join("tmp", "face_plan.json")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "face_detection_yunet_2023mar.onnx",
)
MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)

SAMPLE_INTERVAL = 0.20
MIN_CONFIDENCE = 0.55
MAX_FACES = 5
MAX_SEEK_EVENTS = 1000


def download_model() -> None:
    if os.path.exists(MODEL_PATH):
        return

    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Downloading YuNet face detector...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"Saved model: {MODEL_PATH}")


def load_camera_events() -> List[Dict]:
    if not os.path.exists(CAMERA_PLAN_PATH):
        return []

    with open(CAMERA_PLAN_PATH, "r", encoding="utf-8") as f:
        plan = json.load(f)

    events = plan.get("events", [])
    return events if isinstance(events, list) else []


def create_detector(width: int, height: int):
    detector = cv2.FaceDetectorYN.create(
        MODEL_PATH,
        "",
        (width, height),
        score_threshold=MIN_CONFIDENCE,
        nms_threshold=0.3,
        top_k=5000,
    )
    return detector


def detect_faces(detector, frame) -> List[Dict]:
    height, width = frame.shape[:2]
    detector.setInputSize((width, height))

    _, detections = detector.detect(frame)

    if detections is None:
        return []

    results: List[Dict] = []

    for detection in detections[:MAX_FACES]:
        x, y, w, h = [float(v) for v in detection[:4]]
        confidence = float(detection[-1])

        if confidence < MIN_CONFIDENCE:
            continue

        center_x = x + (w / 2.0)
        center_y = y + (h / 2.0)

        results.append(
            {
                "center_x": center_x,
                "center_y": center_y,
                "width": max(1.0, w),
                "height": max(1.0, h),
                "confidence": confidence,
            }
        )

    return results


def distance(a: Dict, b: Dict) -> float:
    dx = float(a["center_x"]) - float(b["center_x"])
    dy = float(a["center_y"]) - float(b["center_y"])
    return (dx * dx + dy * dy) ** 0.5


def select_temporal_face(
    detections: List[Dict],
    previous: Optional[Dict],
) -> Optional[Dict]:
    if not detections:
        return None

    if previous is None:
        return max(
            detections,
            key=lambda item: (
                float(item["confidence"]),
                float(item["width"]),
            ),
        )

    # Prefer spatial continuity, with confidence and size as
    # secondary signals. This prevents the tracker from jumping
    # to a different face for one sample.
    def score(face: Dict) -> float:
        dist = distance(face, previous)
        return (
            -dist
            + float(face["confidence"]) * 100.0
            + float(face["width"]) * 0.10
        )

    return max(detections, key=score)


def summarize_event(samples: List[Dict]) -> Optional[Dict]:
    if not samples:
        return None

    centers_x = [float(s["center_x"]) for s in samples]
    centers_y = [float(s["center_y"]) for s in samples]
    widths = [float(s["width"]) for s in samples]
    heights = [float(s["height"]) for s in samples]
    confidences = [float(s["confidence"]) for s in samples]

    # Robust location/size summary.
    center_x = median(centers_x)
    center_y = median(centers_y)
    face_width = median(widths)
    face_height = median(heights)

    # Movement is the median frame-to-frame displacement in pixels
    # divided by the median elapsed sample time. This is later
    # normalized by webcam width in cut.py.
    displacements = []
    for previous, current in zip(samples, samples[1:]):
        dt = max(
            0.001,
            float(current["time"]) - float(previous["time"]),
        )
        displacements.append(
            distance(previous, current) / dt
        )

    movement_px_per_sec = (
        median(displacements)
        if displacements
        else 0.0
    )

    # Position variance is useful for distinguishing a stable
    # centered reaction from someone moving around the frame.
    pos_spread = (
        ((max(centers_x) - min(centers_x)) ** 2
         + (max(centers_y) - min(centers_y)) ** 2)
        ** 0.5
    )

    size_change = (
        max(widths) - min(widths)
        if widths
        else 0.0
    )

    return {
        "center_x": round(center_x, 2),
        "center_y": round(center_y, 2),
        "face_width": round(face_width, 2),
        "face_height": round(face_height, 2),
        "confidence": round(median(confidences), 4),
        "movement_px_per_sec": round(
            movement_px_per_sec,
            2,
        ),
        "position_spread_px": round(
            pos_spread,
            2,
        ),
        "size_change_px": round(
            size_change,
            2,
        ),
        "samples": len(samples),
    }


def analyze_event(
    capture: cv2.VideoCapture,
    event: Dict,
    fps: float,
    detector,
) -> Optional[Dict]:
    start = float(event["start"])
    end = float(event["end"])

    if end <= start:
        return None

    # One seek per event. Then decode sequentially.
    # This is dramatically faster than seeking for every 0.2s sample.
    capture.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)

    sample_every = max(
        1,
        int(round(fps * SAMPLE_INTERVAL)),
    )

    samples: List[Dict] = []
    previous_face: Optional[Dict] = None
    frame_index = 0

    max_frames = max(
        1,
        int((end - start) * fps) + 2,
    )

    while frame_index <= max_frames:
        success, frame = capture.read()
        if not success:
            break

        timestamp = float(
            capture.get(cv2.CAP_PROP_POS_MSEC)
        ) / 1000.0

        if timestamp < start:
            frame_index += 1
            continue

        if timestamp > end:
            break

        if frame_index % sample_every != 0:
            frame_index += 1
            continue

        detections = detect_faces(detector, frame)
        face = select_temporal_face(
            detections,
            previous_face,
        )

        if face is not None:
            tracked = {
                **face,
                "time": timestamp,
            }
            samples.append(tracked)
            previous_face = tracked

        frame_index += 1

    summary = summarize_event(samples)
    if summary is None:
        return None

    return {
        "start": start,
        "end": end,
        "peak": float(event.get("peak", (start + end) / 2.0)),
        "intensity": float(event.get("intensity", 0.0)),
        "face": summary,
    }


def main() -> int:
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}"
        )

    events = load_camera_events()
    if not events:
        print("Camera events to analyze: 0")
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": 2,
                    "mode": "face_aware_camera",
                    "events": [],
                },
                f,
                indent=2,
            )
        return 0

    if len(events) > MAX_SEEK_EVENTS:
        events = events[:MAX_SEEK_EVENTS]

    download_model()

    capture = cv2.VideoCapture(VIDEO_PATH)
    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not fps or fps <= 0:
        fps = 30.0

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = create_detector(width, height)

    print(f"Camera events to analyze: {len(events)}")
    print(
        f"Video: {width}x{height} @ {fps:.2f} fps"
    )
    print(
        f"Sampling: every {SAMPLE_INTERVAL:.2f}s"
    )

    output_events: List[Dict] = []

    try:
        for index, event in enumerate(events, start=1):
            start = float(event["start"])
            end = float(event["end"])

            print(
                f"[{index}/{len(events)}] "
                f"Analyzing {start:.1f}s → {end:.1f}s"
            )

            result = analyze_event(
                capture,
                event,
                fps,
                detector,
            )

            if result is None:
                print("    No stable face detected")
                continue

            face = result["face"]
            output_events.append(result)

            print(
                f"    Face: "
                f"x={face['center_x']:.1f}, "
                f"y={face['center_y']:.1f}, "
                f"width={face['face_width']:.1f}, "
                f"movement={face['movement_px_per_sec']:.1f}px/s"
            )
    finally:
        capture.release()

    plan = {
        "version": 2,
        "mode": "face_aware_camera_v2",
        "sample_interval": SAMPLE_INTERVAL,
        "events": output_events,
    }

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            plan,
            f,
            indent=2,
        )

    print(
        f"Face-aware events: {len(output_events)}"
    )
    print(f"Saved: {OUTPUT_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
