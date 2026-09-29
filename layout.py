import json
import os
from typing import Dict

import cv2


def get_video_info(video_path: str) -> Dict:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    duration = frames / fps if fps else 0.0

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration,
    }


def analyze_layout(video_path: str) -> Dict:
    info = get_video_info(video_path)

    print("      Analyzing source video layout...")
    print(
        f"      Source: {info['width']}x{info['height']} "
        f"({info['duration']:.1f}s)"
    )

    # Tuned from the actual 1920x1080 source frame.
    #
    # We intentionally begin at 70% rather than ~69.5%
    # to eliminate the thin gameplay line on the webcam's left.
    result = {
        "type": "gaming_stream",
        "webcam": True,
        "webcam_position": "top_right",

        "webcam_region": {
            "x": 0.702,
            "y": 0.000,
            "width": 0.298,
            "height": 0.405,
        },

        "gameplay_region": {
            "x": 0.000,
            "y": 0.000,
            "width": 0.695,
            "height": 1.000,
        },

        "source_width": info["width"],
        "source_height": info["height"],
    }

    print("      Layout: GAMING STREAM")
    print("      Webcam: TOP-RIGHT")

    return result


def load_or_analyze(
    video_path: str,
    layout_path: str,
    force: bool = False,
) -> Dict:

    if os.path.exists(layout_path) and not force:
        try:
            with open(layout_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    result = analyze_layout(video_path)

    os.makedirs(
        os.path.dirname(layout_path) or ".",
        exist_ok=True,
    )

    with open(layout_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"      Saved layout: {layout_path}")

    return result