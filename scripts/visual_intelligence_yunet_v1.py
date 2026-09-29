from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# IMPORTANT:
# Use the CUDA/cuDNN DLLs already bundled with the working WhisperX
# PyTorch installation.
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TORCH_LIB = (
    PROJECT_ROOT
    / ".venv-whisperx"
    / "Lib"
    / "site-packages"
    / "torch"
    / "lib"
)

if not TORCH_LIB.exists():
    raise RuntimeError(
        f"Could not find CUDA DLL directory:\n{TORCH_LIB}"
    )

os.environ["PATH"] = (
    str(TORCH_LIB)
    + os.pathsep
    + os.environ.get("PATH", "")
)

if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(TORCH_LIB))


import cv2
import numpy as np
import onnxruntime as ort


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = (
    PROJECT_ROOT
    / "models"
    / "face_detection_yunet_2023mar.onnx"
)

DEFAULT_SHOTS = (
    PROJECT_ROOT
    / "tmp"
    / "new_inde_shots.json"
)

DEFAULT_VIDEO = PROJECT_ROOT / (
    "Leon Kennedy meets Grace Ashcroft  Nick Apostolides and "
    "Angela Sant'Albano Interview Each Other - FGS (1080p, h264).mp4"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "tmp"
    / "new_inde_visual_intelligence_yunet_v1.json"
)

INPUT_W = 640
INPUT_H = 640

STRIDES = (8, 16, 32)

DEFAULT_SCORE_THRESHOLD = 0.60
DEFAULT_NMS_THRESHOLD = 0.45
DEFAULT_TOP_K = 100

DEFAULT_SAMPLE_EVERY = 0.5

TRACK_MAX_MISSES = 3
TRACK_IOU_THRESHOLD = 0.20
TRACK_CENTER_THRESHOLD = 0.12


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    x: float
    y: float
    w: float
    h: float
    score: float
    landmarks: list[tuple[float, float]] = field(
        default_factory=list
    )

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def area(self) -> float:
        return max(
            0.0,
            self.w,
        ) * max(
            0.0,
            self.h,
        )


@dataclass
class Track:
    track_id: int
    first_time: float
    last_time: float
    last_detection: Detection
    observations: int = 1
    misses: int = 0

    scores: list[float] = field(
        default_factory=list
    )

    centers: list[tuple[float, float]] = field(
        default_factory=list
    )

    areas: list[float] = field(
        default_factory=list
    )

    def add(
        self,
        timestamp: float,
        detection: Detection,
    ) -> None:

        self.last_time = timestamp
        self.last_detection = detection
        self.observations += 1
        self.misses = 0

        self.scores.append(
            detection.score
        )

        self.centers.append(
            (
                detection.cx,
                detection.cy,
            )
        )

        self.areas.append(
            detection.area
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    return max(
        minimum,
        min(
            maximum,
            value,
        ),
    )


def iou(
    a: Detection,
    b: Detection,
) -> float:

    ax0 = a.x
    ay0 = a.y
    ax1 = a.x + a.w
    ay1 = a.y + a.h

    bx0 = b.x
    by0 = b.y
    bx1 = b.x + b.w
    by1 = b.y + b.h

    ix0 = max(
        ax0,
        bx0,
    )
    iy0 = max(
        ay0,
        by0,
    )
    ix1 = min(
        ax1,
        bx1,
    )
    iy1 = min(
        ay1,
        by1,
    )

    iw = max(
        0.0,
        ix1 - ix0,
    )

    ih = max(
        0.0,
        iy1 - iy0,
    )

    intersection = iw * ih

    if intersection <= 0.0:
        return 0.0

    union = (
        a.w * a.h
        + b.w * b.h
        - intersection
    )

    if union <= 0.0:
        return 0.0

    return intersection / union


def normalized_center_distance(
    a: Detection,
    b: Detection,
) -> float:

    dx = a.cx - b.cx
    dy = a.cy - b.cy

    distance = math.sqrt(
        dx * dx + dy * dy
    )

    diagonal = math.sqrt(
        INPUT_W * INPUT_W
        + INPUT_H * INPUT_H
    )

    return distance / diagonal


# ---------------------------------------------------------------------------
# YuNet decoding
# ---------------------------------------------------------------------------

def decode_yunet(
    outputs: list[np.ndarray],
    image_width: int,
    image_height: int,
    score_threshold: float,
    nms_threshold: float,
    top_k: int,
) -> list[Detection]:
    """
    Decode OpenCV Zoo YuNet 2023mar outputs.

    Output order:
        cls_8, cls_16, cls_32,
        obj_8, obj_16, obj_32,
        bbox_8, bbox_16, bbox_32,
        kps_8, kps_16, kps_32

    Input is fixed 640x640 BGR, 0-255.
    """

    if len(outputs) != 12:
        raise RuntimeError(
            f"Expected 12 YuNet outputs, got {len(outputs)}"
        )

    detections: list[Detection] = []

    for level, stride in enumerate(
        STRIDES
    ):

        cls = outputs[
            level
        ][0]

        obj = outputs[
            3 + level
        ][0]

        bbox = outputs[
            6 + level
        ][0]

        kps = outputs[
            9 + level
        ][0]

        feature_w = INPUT_W // stride
        feature_h = INPUT_H // stride

        expected_anchors = (
            feature_w
            * feature_h
        )

        if cls.shape[0] != expected_anchors:
            raise RuntimeError(
                f"Unexpected anchor count at stride {stride}: "
                f"{cls.shape[0]} vs {expected_anchors}"
            )

        for index in range(
            expected_anchors
        ):

            class_score = float(
                cls[index, 0]
            )

            objectness = float(
                obj[index, 0]
            )

            score = (
                class_score
                * objectness
            )

            if score < score_threshold:
                continue

            grid_x = (
                index
                % feature_w
            )

            grid_y = (
                index
                // feature_w
            )

            anchor_x = (
                grid_x * stride
            )

            anchor_y = (
                grid_y * stride
            )

            dx = float(
                bbox[index, 0]
            )

            dy = float(
                bbox[index, 1]
            )

            dw = float(
                bbox[index, 2]
            )

            dh = float(
                bbox[index, 3]
            )

            center_x = (
                dx * stride
                + anchor_x
            )

            center_y = (
                dy * stride
                + anchor_y
            )

            width = (
                math.exp(
                    clamp(
                        dw,
                        -20.0,
                        20.0,
                    )
                )
                * stride
            )

            height = (
                math.exp(
                    clamp(
                        dh,
                        -20.0,
                        20.0,
                    )
                )
                * stride
            )

            x = (
                center_x
                - width / 2.0
            )

            y = (
                center_y
                - height / 2.0
            )

            landmark_points: list[
                tuple[float, float]
            ] = []

            for landmark_index in range(5):

                lx = float(
                    kps[
                        index,
                        landmark_index * 2,
                    ]
                )

                ly = float(
                    kps[
                        index,
                        landmark_index * 2
                        + 1,
                    ]
                )

                landmark_points.append(
                    (
                        lx * stride
                        + anchor_x,
                        ly * stride
                        + anchor_y,
                    )
                )

            # Convert 640x640 coordinates back into
            # the original frame dimensions.
            sx = (
                image_width
                / INPUT_W
            )

            sy = (
                image_height
                / INPUT_H
            )

            x *= sx
            y *= sy
            width *= sx
            height *= sy

            scaled_landmarks = [
                (
                    px * sx,
                    py * sy,
                )
                for px, py
                in landmark_points
            ]

            detections.append(
                Detection(
                    x=x,
                    y=y,
                    w=width,
                    h=height,
                    score=score,
                    landmarks=scaled_landmarks,
                )
            )

    return non_max_suppression(
        detections,
        nms_threshold,
        top_k,
    )


def non_max_suppression(
    detections: list[Detection],
    threshold: float,
    top_k: int,
) -> list[Detection]:

    if not detections:
        return []

    ordered = sorted(
        detections,
        key=lambda d: d.score,
        reverse=True,
    )

    if top_k > 0:
        ordered = ordered[
            :top_k
        ]

    selected: list[Detection] = []

    for candidate in ordered:

        suppressed = False

        for previous in selected:

            if (
                iou(
                    candidate,
                    previous,
                )
                > threshold
            ):
                suppressed = True
                break

        if not suppressed:
            selected.append(
                candidate
            )

    return selected


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

def match_detections_to_tracks(
    detections: list[Detection],
    tracks: list[Track],
) -> dict[int, int]:
    """
    Greedy global matching.

    Returns:
        detection_index -> track_index
    """

    candidates: list[
        tuple[
            float,
            int,
            int,
        ]
    ] = []

    for detection_index, detection in enumerate(
        detections
    ):

        for track_index, track in enumerate(
            tracks
        ):

            previous = (
                track.last_detection
            )

            overlap_score = iou(
                detection,
                previous,
            )

            distance = normalized_center_distance(
                detection,
                previous,
            )

            if (
                overlap_score
                >= TRACK_IOU_THRESHOLD
                or distance
                <= TRACK_CENTER_THRESHOLD
            ):

                # Higher IoU is better.
                # Lower distance is better.
                matching_cost = (
                    overlap_score
                    - distance
                )

                candidates.append(
                    (
                        matching_cost,
                        detection_index,
                        track_index,
                    )
                )

    candidates.sort(
        reverse=True
    )

    matched_detections: set[
        int
    ] = set()

    matched_tracks: set[
        int
    ] = set()

    matches: dict[
        int,
        int,
    ] = {}

    for (
        _cost,
        detection_index,
        track_index,
    ) in candidates:

        if detection_index in matched_detections:
            continue

        if track_index in matched_tracks:
            continue

        matches[
            detection_index
        ] = track_index

        matched_detections.add(
            detection_index
        )

        matched_tracks.add(
            track_index
        )

    return matches


# ---------------------------------------------------------------------------
# Shot helpers
# ---------------------------------------------------------------------------

def load_shots(
    path: Path,
) -> list[dict[str, Any]]:

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if isinstance(
        data,
        dict,
    ):
        shots = data.get(
            "shots",
            [],
        )
    else:
        shots = data

    if not isinstance(
        shots,
        list,
    ):
        raise ValueError(
            "Shot inventory does not contain a list."
        )

    cleaned = []

    for shot in shots:

        if not isinstance(
            shot,
            dict,
        ):
            continue

        start = float(
            shot.get(
                "start",
                0.0,
            )
        )

        end = float(
            shot.get(
                "end",
                0.0,
            )
        )

        if end <= start:
            continue

        cleaned.append(
            {
                **shot,
                "start": start,
                "end": end,
            }
        )

    return sorted(
        cleaned,
        key=lambda x: x[
            "start"
        ],
    )


def find_shot(
    shots: list[dict[str, Any]],
    timestamp: float,
) -> dict[str, Any] | None:

    for shot in shots:

        if (
            shot["start"]
            <= timestamp
            < shot["end"]
        ):
            return shot

    return None


# ---------------------------------------------------------------------------
# Visual intelligence aggregation
# ---------------------------------------------------------------------------

def build_shot_summary(
    shots: list[dict[str, Any]],
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    summaries = []

    for shot in shots:

        shot_observations = [
            item
            for item in observations
            if (
                shot["start"]
                <= item["timestamp"]
                < shot["end"]
            )
        ]

        unique_tracks = sorted(
            {
                int(
                    item["track_id"]
                )
                for item
                in shot_observations
                if item["track_id"] is not None
            }
        )

        track_stats = []

        for track_id in unique_tracks:

            track_obs = [
                item
                for item in shot_observations
                if item["track_id"]
                == track_id
            ]

            if not track_obs:
                continue

            scores = [
                item["score"]
                for item
                in track_obs
            ]

            centers = [
                item["center"]
                for item
                in track_obs
            ]

            areas = [
                item["area_normalized"]
                for item
                in track_obs
            ]

            xs = [
                center[0]
                for center in centers
            ]

            ys = [
                center[1]
                for center in centers
            ]

            track_stats.append(
                {
                    "track_id": track_id,
                    "observations": len(
                        track_obs
                    ),
                    "presence_ratio": round(
                        len(track_obs)
                        / max(
                            1,
                            len(
                                {
                                    item[
                                        "timestamp"
                                    ]
                                    for item
                                    in shot_observations
                                }
                            ),
                        ),
                        4,
                    ),
                    "mean_confidence": round(
                        float(
                            np.mean(scores)
                        ),
                        4,
                    ),
                    "max_confidence": round(
                        float(
                            np.max(scores)
                        ),
                        4,
                    ),
                    "center_mean": [
                        round(
                            float(
                                np.mean(xs)
                            ),
                            4,
                        ),
                        round(
                            float(
                                np.mean(ys)
                            ),
                            4,
                        ),
                    ],
                    "center_min": [
                        round(
                            float(
                                np.min(xs)
                            ),
                            4,
                        ),
                        round(
                            float(
                                np.min(ys)
                            ),
                            4,
                        ),
                    ],
                    "center_max": [
                        round(
                            float(
                                np.max(xs)
                            ),
                            4,
                        ),
                        round(
                            float(
                                np.max(ys)
                            ),
                            4,
                        ),
                    ],
                    "mean_area_normalized": round(
                        float(
                            np.mean(areas)
                        ),
                        4,
                    ),
                }
            )

        if track_stats:

            # Sort most visually persistent face first.
            track_stats.sort(
                key=lambda x: (
                    x[
                        "presence_ratio"
                    ],
                    x[
                        "mean_confidence"
                    ],
                ),
                reverse=True,
            )

        face_count_samples = [
            item["face_count"]
            for item
            in shot_observations
        ]

        if face_count_samples:

            mean_face_count = float(
                np.mean(
                    face_count_samples
                )
            )

            max_face_count = int(
                np.max(
                    face_count_samples
                )
            )

        else:

            mean_face_count = 0.0
            max_face_count = 0

        if max_face_count == 0:
            composition = "no_face"

        elif max_face_count == 1:
            composition = "single_face"

        elif max_face_count == 2:
            composition = "two_faces"

        else:
            composition = "multiple_faces"

        summaries.append(
            {
                "shot_id": str(
                    shot.get(
                        "shot_id",
                        "",
                    )
                ),
                "start": shot[
                    "start"
                ],
                "end": shot[
                    "end"
                ],
                "duration": round(
                    shot["end"]
                    - shot["start"],
                    4,
                ),
                "shot_type": shot.get(
                    "shot_type"
                ),
                "sample_count": len(
                    {
                        item[
                            "timestamp"
                        ]
                        for item
                        in shot_observations
                    }
                ),
                "mean_face_count": round(
                    mean_face_count,
                    3,
                ),
                "max_face_count": max_face_count,
                "composition": composition,
                "tracks": track_stats,
            }
        )

    return summaries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "GPU YuNet visual intelligence for INDE interviews."
        )
    )

    parser.add_argument(
        "--video",
        default=str(
            DEFAULT_VIDEO
        ),
    )

    parser.add_argument(
        "--model",
        default=str(
            DEFAULT_MODEL
        ),
    )

    parser.add_argument(
        "--shots",
        default=str(
            DEFAULT_SHOTS
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT
        ),
    )

    parser.add_argument(
        "--sample-every",
        type=float,
        default=DEFAULT_SAMPLE_EVERY,
        help="Seconds between analyzed frames.",
    )

    parser.add_argument(
        "--score",
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
    )

    parser.add_argument(
        "--nms",
        type=float,
        default=DEFAULT_NMS_THRESHOLD,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
    )

    args = parser.parse_args()

    video_path = Path(
        args.video
    )

    model_path = Path(
        args.model
    )

    shots_path = Path(
        args.shots
    )

    output_path = Path(
        args.output
    )

    for path, label in (
        (
            video_path,
            "video",
        ),
        (
            model_path,
            "YuNet model",
        ),
        (
            shots_path,
            "shot inventory",
        ),
    ):

        if not path.exists():
            raise FileNotFoundError(
                f"{label} not found:\n{path}"
            )

    # -----------------------------------------------------------------------
    # ONNX Runtime / CUDA
    # -----------------------------------------------------------------------

    print("=== VISUAL INTELLIGENCE v1 ===")

    print(
        "ONNX Runtime:",
        ort.__version__,
    )

    print(
        "Available providers:",
        ort.get_available_providers(),
    )

    session = ort.InferenceSession(
        str(model_path),
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    session_providers = (
        session.get_providers()
    )

    print(
        "Session providers:",
        session_providers,
    )

    if (
        "CUDAExecutionProvider"
        not in session_providers
    ):
        raise RuntimeError(
            "CUDAExecutionProvider failed. "
            "Refusing to run Visual Intelligence on CPU."
        )

    input_meta = session.get_inputs()[0]

    print(
        "Model input:",
        input_meta.name,
        input_meta.shape,
    )

    if input_meta.shape != [
        1,
        3,
        640,
        640,
    ]:
        raise RuntimeError(
            "Unexpected YuNet model input shape: "
            f"{input_meta.shape}"
        )

    # -----------------------------------------------------------------------
    # Shots
    # -----------------------------------------------------------------------

    shots = load_shots(
        shots_path
    )

    print(
        "Shots:",
        len(shots),
    )

    # -----------------------------------------------------------------------
    # Video
    # -----------------------------------------------------------------------

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video:\n{video_path}"
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = capture.get(
        cv2.CAP_PROP_FRAME_COUNT
    )

    duration = capture.get(
        cv2.CAP_PROP_FRAME_COUNT
    ) / max(
        fps,
        1e-6,
    )

    reported_duration = capture.get(
        cv2.CAP_PROP_POS_MSEC
    )

    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    print(
        "Video:",
        f"{width}x{height}",
        f"{fps:.3f} fps",
        f"{duration:.3f}s",
    )

    sample_every = max(
        0.05,
        float(
            args.sample_every
        ),
    )

    # -----------------------------------------------------------------------
    # Tracking state
    # -----------------------------------------------------------------------

    tracks: list[Track] = []

    next_track_id = 0

    observations: list[
        dict[str, Any]
    ] = []

    frame_index = 0

    next_sample_time = 0.0

    analyzed_frames = 0

    total_faces = 0

    # -----------------------------------------------------------------------
    # Sequential frame sampling
    # -----------------------------------------------------------------------

    while True:

        ok, frame = capture.read()

        if not ok:
            break

        timestamp = (
            frame_index
            / max(
                fps,
                1e-6,
            )
        )

        frame_index += 1

        if (
            timestamp
            + 1e-9
            < next_sample_time
        ):
            continue

        next_sample_time += sample_every

        analyzed_frames += 1

        # IMPORTANT:
        # YuNet 2023mar expects BGR [0..255], NCHW.
        resized = cv2.resize(
            frame,
            (
                INPUT_W,
                INPUT_H,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        tensor = np.transpose(
            resized,
            (
                2,
                0,
                1,
            ),
        )

        tensor = np.expand_dims(
            tensor.astype(
                np.float32
            ),
            axis=0,
        )

        outputs = session.run(
            None,
            {
                input_meta.name: tensor,
            },
        )

        detections = decode_yunet(
            outputs,
            width,
            height,
            float(
                args.score
            ),
            float(
                args.nms
            ),
            int(
                args.top_k
            ),
        )

        total_faces += len(
            detections
        )

        # ---------------------------------------------------------------
        # Track matching
        # ---------------------------------------------------------------

        matches = match_detections_to_tracks(
            detections,
            tracks,
        )

        matched_track_indices = set(
            matches.values()
        )

        # Update existing tracks.
        for (
            detection_index,
            track_index,
        ) in matches.items():

            tracks[
                track_index
            ].add(
                timestamp,
                detections[
                    detection_index
                ],
            )

        # Mark unmatched tracks as missed.
        for track_index, track in enumerate(
            tracks
        ):

            if track_index not in matched_track_indices:

                track.misses += 1

        # Create new tracks.
        for detection_index, detection in enumerate(
            detections
        ):

            if detection_index in matches:
                continue

            track = Track(
                track_id=next_track_id,
                first_time=timestamp,
                last_time=timestamp,
                last_detection=detection,
            )

            track.scores.append(
                detection.score
            )

            track.centers.append(
                (
                    detection.cx,
                    detection.cy,
                )
            )

            track.areas.append(
                detection.area
            )

            tracks.append(
                track
            )

            next_track_id += 1

        # Remove tracks that have disappeared.
        tracks = [
            track
            for track in tracks
            if track.misses
            <= TRACK_MAX_MISSES
        ]

        # ---------------------------------------------------------------
        # Save observations for this sampled frame.
        # ---------------------------------------------------------------

        current_track_detections: list[
            tuple[int, Detection]
        ] = []

        for detection_index, detection in enumerate(
            detections
        ):

            track_index = matches.get(
                detection_index
            )

            if (
                track_index is None
                or track_index >= len(tracks)
            ):
                # New-track lookup by closest latest detection.
                candidates = [
                    track
                    for track
                    in tracks
                    if track.last_time
                    == timestamp
                    and track.last_detection
                    is detection
                ]

                if candidates:
                    track_id = candidates[
                        0
                    ].track_id
                else:
                    track_id = None

            else:
                track_id = tracks[
                    track_index
                ].track_id

            current_track_detections.append(
                (
                    track_id,
                    detection,
                )
            )

        # If tracks changed because of pruning, resolve new-track IDs
        # robustly from current detections.
        for (
            track_id,
            detection,
        ) in current_track_detections:

            if track_id is None:
                continue

            center_x = (
                detection.cx
                / width
            )

            center_y = (
                detection.cy
                / height
            )

            area_normalized = (
                detection.area
                / (
                    width
                    * height
                )
            )

            shot = find_shot(
                shots,
                timestamp,
            )

            observations.append(
                {
                    "timestamp": round(
                        timestamp,
                        4,
                    ),
                    "shot_id": (
                        str(
                            shot.get(
                                "shot_id",
                                "",
                            )
                        )
                        if shot
                        else None
                    ),
                    "track_id": int(
                        track_id
                    ),
                    "score": round(
                        detection.score,
                        5,
                    ),
                    "bbox": [
                        round(
                            detection.x,
                            2,
                        ),
                        round(
                            detection.y,
                            2,
                        ),
                        round(
                            detection.w,
                            2,
                        ),
                        round(
                            detection.h,
                            2,
                        ),
                    ],
                    "center": [
                        round(
                            center_x,
                            5,
                        ),
                        round(
                            center_y,
                            5,
                        ),
                    ],
                    "area_normalized": round(
                        area_normalized,
                        6,
                    ),
                    "landmarks": [
                        [
                            round(
                                x,
                                2,
                            ),
                            round(
                                y,
                                2,
                            ),
                        ]
                        for x, y
                        in detection.landmarks
                    ],
                    "face_count": len(
                        detections
                    ),
                }
            )

        if analyzed_frames % 100 == 0:

            print(
                f"Analyzed frames: "
                f"{analyzed_frames} | "
                f"time: {timestamp:.1f}s | "
                f"faces: {len(detections)} | "
                f"tracks: {len(tracks)}"
            )

    capture.release()

    # -----------------------------------------------------------------------
    # Shot summaries
    # -----------------------------------------------------------------------

    shot_summaries = build_shot_summary(
        shots,
        observations,
    )

    # -----------------------------------------------------------------------
    # Aggregate tracks
    # -----------------------------------------------------------------------

    track_summary = []

    for track in tracks:

        if not track.centers:
            continue

        xs = [
            point[0]
            for point in track.centers
        ]

        ys = [
            point[1]
            for point in track.centers
        ]

        track_summary.append(
            {
                "track_id": track.track_id,
                "first_time": round(
                    track.first_time,
                    4,
                ),
                "last_time": round(
                    track.last_time,
                    4,
                ),
                "duration": round(
                    track.last_time
                    - track.first_time,
                    4,
                ),
                "observations": track.observations,
                "mean_confidence": round(
                    float(
                        np.mean(
                            track.scores
                        )
                    ),
                    4,
                ),
                "mean_center": [
                    round(
                        float(
                            np.mean(xs)
                        ),
                        5,
                    ),
                    round(
                        float(
                            np.mean(ys)
                        ),
                        5,
                    ),
                ],
                "mean_area_normalized": round(
                    float(
                        np.mean(
                            [
                                area
                                / (
                                    width
                                    * height
                                )
                                for area
                                in track.areas
                            ]
                        )
                    ),
                    6,
                ),
            }
        )

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------

    result = {
        "version": "1.0",
        "module": "visual_intelligence_yunet",
        "source_video": str(
            video_path
        ),
        "source_shots": str(
            shots_path
        ),
        "model": str(
            model_path
        ),
        "inference": {
            "runtime": "onnxruntime",
            "execution_provider": "CUDAExecutionProvider",
            "input_size": [
                INPUT_W,
                INPUT_H,
            ],
            "input_format": "BGR float32 0-255 NCHW",
            "sample_every_seconds": sample_every,
            "score_threshold": float(
                args.score
            ),
            "nms_threshold": float(
                args.nms
            ),
            "top_k": int(
                args.top_k
            ),
        },
        "video": {
            "width": width,
            "height": height,
            "fps": round(
                float(fps),
                4,
            ),
            "frame_count": int(
                frame_count
            ),
            "duration": round(
                float(duration),
                4,
            ),
        },
        "statistics": {
            "analyzed_frames": analyzed_frames,
            "total_face_observations": total_faces,
            "tracks_created": next_track_id,
            "shots": len(shots),
        },
        "tracks": sorted(
            track_summary,
            key=lambda x: x[
                "track_id"
            ],
        ),
        "observations": observations,
        "shot_visual_intelligence": shot_summaries,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== COMPLETE ===")

    print(
        "Analyzed frames:",
        analyzed_frames,
    )

    print(
        "Face observations:",
        total_faces,
    )

    print(
        "Tracks created:",
        next_track_id,
    )

    print(
        "Shots:",
        len(shots),
    )

    print(
        "Output:",
        output_path,
    )


if __name__ == "__main__":
    main()
