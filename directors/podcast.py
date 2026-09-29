
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# CONFIGURATION
# ============================================================

# Stable speaker-face confidence needed for an ordinary
# speaker close-up.
MIN_SPEAKER_CONFIDENCE = 0.60

# Strong moments can tolerate slightly lower confidence.
STRONG_MOMENT_CONFIDENCE = 0.50

# Minimum intensity for an event to influence the camera.
MIN_MOMENT_INTENSITY = 0.45

# Intensity at which the event is considered strong.
STRONG_MOMENT_INTENSITY = 0.68

# Minimum useful duration of a camera takeover.
MIN_CLOSEUP_DURATION = 1.00

# Add context around the detected moment.
PRE_ROLL = 0.75
POST_ROLL = 0.90

# Don't create multiple cuts immediately next to each other.
MIN_GAP_BETWEEN_EVENTS = 2.0

# Don't use extremely tiny face detections for closeups.
MIN_FACE_HEIGHT_RATIO = 0.08

# Desired face size in closeup.
TARGET_FACE_HEIGHT_RATIO = 0.52

# Face-track lookup tolerance.
FACE_TIME_TOLERANCE = 0.50

# Stable timeline lookup tolerance.
TIMELINE_TIME_TOLERANCE = 0.75

# How close an already-resolved speaker-face segment must be to
# a moment.
MAX_SPEAKER_FACE_DISTANCE = 1.25


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class FaceBox:
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0


@dataclass
class DirectorAction:
    start: float
    end: float
    shot: str
    speaker: Optional[str]
    face_id: Optional[str]
    confidence: float
    reason: str
    importance: float
    crop: Optional[Dict[str, float]] = None


# ============================================================
# GENERIC HELPERS
# ============================================================

def load_json(path: str) -> Any:
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def to_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        return float(value)
    except (
        TypeError,
        ValueError,
    ):
        return default


def clamp(
    value: float,
    low: float = 0.0,
    high: float = 1.0,
) -> float:
    return max(
        low,
        min(high, value),
    )


# ============================================================
# FACE BOX
# ============================================================

def extract_face_box(
    detection: Dict[str, Any],
) -> Optional[FaceBox]:

    # --------------------------------------------------------
    # Exact YuNet v5 schema discovered in your file:
    #
    # x
    # y
    # width
    # height
    # --------------------------------------------------------

    if all(
        key in detection
        for key in (
            "x",
            "y",
            "width",
            "height",
        )
    ):

        x = to_float(
            detection["x"]
        )

        y = to_float(
            detection["y"]
        )

        width = to_float(
            detection["width"]
        )

        height = to_float(
            detection["height"]
        )

        if width > 0 and height > 0:

            return FaceBox(
                x=x,
                y=y,
                w=width,
                h=height,
            )

    # --------------------------------------------------------
    # Generic bbox fallback.
    # --------------------------------------------------------

    bbox = detection.get(
        "bbox"
    )

    if (
        isinstance(
            bbox,
            (list, tuple),
        )
        and len(bbox) >= 4
    ):

        x = to_float(
            bbox[0]
        )

        y = to_float(
            bbox[1]
        )

        width = to_float(
            bbox[2]
        )

        height = to_float(
            bbox[3]
        )

        if width > 0 and height > 0:

            return FaceBox(
                x=x,
                y=y,
                w=width,
                h=height,
            )

    # --------------------------------------------------------
    # x/y/w/h fallback.
    # --------------------------------------------------------

    if all(
        key in detection
        for key in (
            "x",
            "y",
            "w",
            "h",
        )
    ):

        width = to_float(
            detection["w"]
        )

        height = to_float(
            detection["h"]
        )

        if width > 0 and height > 0:

            return FaceBox(
                x=to_float(
                    detection["x"]
                ),
                y=to_float(
                    detection["y"]
                ),
                w=width,
                h=height,
            )

    return None


# ============================================================
# FACE TRACKS
# ============================================================

def load_face_tracks(
    path: str,
) -> Dict[
    str,
    List[
        Tuple[
            float,
            FaceBox,
        ]
    ],
]:

    data = load_json(path)

    if not isinstance(
        data,
        dict,
    ):
        raise ValueError(
            "Face file must contain a JSON object."
        )

    raw_tracks = data.get(
        "tracks",
        [],
    )

    tracks: Dict[
        str,
        List[
            Tuple[
                float,
                FaceBox,
            ]
        ],
    ] = {}

    print(
        f"  Raw tracks in JSON: "
        f"{len(raw_tracks)}"
    )

    for raw_track in raw_tracks:

        if not isinstance(
            raw_track,
            dict,
        ):
            continue

        face_id = raw_track.get(
            "face_id"
        )

        if face_id is None:

            face_id = raw_track.get(
                "id"
            )

        if face_id is None:
            continue

        face_id = str(
            face_id
        )

        raw_detections = raw_track.get(
            "detections",
            [],
        )

        if not isinstance(
            raw_detections,
            list,
        ):
            continue

        samples: List[
            Tuple[
                float,
                FaceBox,
            ]
        ] = []

        for detection in raw_detections:

            if not isinstance(
                detection,
                dict,
            ):
                continue

            if detection.get(
                "time"
            ) is None:

                continue

            time_value = to_float(
                detection.get(
                    "time"
                ),
                -1.0,
            )

            if time_value < 0:
                continue

            box = extract_face_box(
                detection
            )

            if box is None:
                continue

            samples.append(
                (
                    time_value,
                    box,
                )
            )

        samples.sort(
            key=lambda item:
                item[0]
        )

        if samples:

            tracks[
                face_id
            ] = samples

    print(
        f"  Tracks with usable detections: "
        f"{len(tracks)}"
    )

    if tracks:

        first_face = next(
            iter(tracks)
        )

        print(
            f"  Example: "
            f"{first_face} -> "
            f"{len(tracks[first_face])} detections"
        )

    return tracks


def bbox_at_time(
    samples: List[
        Tuple[
            float,
            FaceBox,
        ]
    ],
    time_value: float,
) -> Optional[FaceBox]:

    best_box = None
    best_distance = float(
        "inf"
    )

    for sample_time, box in samples:

        distance = abs(
            sample_time
            - time_value
        )

        if distance < best_distance:

            best_distance = distance
            best_box = box

    if (
        best_box is None
        or best_distance
        > FACE_TIME_TOLERANCE
    ):
        return None

    return best_box


# ============================================================
# STABLE SPEAKER-FACE TIMELINE
# ============================================================

def load_stable_timeline(
    path: str,
) -> List[Dict[str, Any]]:

    data = load_json(path)

    if not isinstance(
        data,
        dict,
    ):
        raise ValueError(
            "Stable timeline must be a JSON object."
        )

    timeline = data.get(
        "timeline",
        [],
    )

    result = []

    for item in timeline:

        if not isinstance(
            item,
            dict,
        ):
            continue

        time_value = item.get(
            "time"
        )

        if time_value is None:
            continue

        result.append(
            {
                **item,
                "time":
                    to_float(
                        time_value
                    ),
            }
        )

    result.sort(
        key=lambda x:
            x["time"]
    )

    return result


# ============================================================
# MOMENTS
# ============================================================

def load_moments(
    path: str,
) -> List[Dict[str, Any]]:

    data = load_json(path)

    if isinstance(
        data,
        dict,
    ):

        raw_moments = data.get(
            "moments",
            [],
        )

    elif isinstance(
        data,
        list,
    ):

        raw_moments = data

    else:

        raw_moments = []

    result = []

    for item in raw_moments:

        if not isinstance(
            item,
            dict,
        ):
            continue

        if "start" not in item:
            continue

        start = to_float(
            item.get(
                "start"
            )
        )

        end = to_float(
            item.get(
                "end",
                start,
            )
        )

        intensity = clamp(
            to_float(
                item.get(
                    "intensity",
                    item.get(
                        "score",
                        0.0,
                    ),
                )
            )
        )

        speech = clamp(
            to_float(
                item.get(
                    "speech",
                    0.0,
                )
            )
        )

        result.append(
            {
                **item,
                "start":
                    start,
                "end":
                    max(
                        start,
                        end,
                    ),
                "intensity":
                    intensity,
                "score":
                    intensity,
                "speech":
                    speech,
            }
        )

    result.sort(
        key=lambda x:
            x["start"]
    )

    return result


# ============================================================
# LOCAL TIMELINE LOOKUPS
# ============================================================

def nearest_timeline_item(
    time_value: float,
    timeline: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:

    best = None
    best_distance = float(
        "inf"
    )

    for item in timeline:

        distance = abs(
            item["time"]
            - time_value
        )

        if distance < best_distance:

            best_distance = distance
            best = item

    if (
        best is None
        or best_distance
        > TIMELINE_TIME_TOLERANCE
    ):
        return None

    return best


def active_speaker_at(
    time_value: float,
    timeline: List[Dict[str, Any]],
) -> Optional[str]:

    item = nearest_timeline_item(
        time_value,
        timeline,
    )

    if item is None:
        return None

    speaker = item.get(
        "speaker"
    )

    if speaker is None:
        return None

    return str(
        speaker
    )


# ============================================================
# SPEAKER → FACE SEGMENTS
# ============================================================

def build_speaker_face_segments(
    timeline: List[Dict[str, Any]],
) -> Dict[
    str,
    List[
        Tuple[
            float,
            float,
            str,
            float,
        ]
    ],
]:

    result: Dict[
        str,
        List[
            Tuple[
                float,
                float,
                str,
                float,
            ]
        ],
    ] = defaultdict(list)

    if not timeline:
        return result

    current = None

    for item in timeline:

        speaker = item.get(
            "speaker"
        )

        face_id = item.get(
            "stable_face_id"
        )

        identity = (
            speaker,
            face_id,
        )

        if current is None:

            current = {
                "start":
                    item["time"],
                "end":
                    item["time"],
                "speaker":
                    speaker,
                "face_id":
                    face_id,
                "confidence_sum":
                    0.0,
                "samples":
                    0,
            }

        current_identity = (
            current["speaker"],
            current["face_id"],
        )

        if identity != current_identity:

            if (
                current["speaker"]
                is not None
                and current["face_id"]
                is not None
            ):

                confidence = (
                    current[
                        "confidence_sum"
                    ]
                    / max(
                        1,
                        current[
                            "samples"
                        ],
                    )
                )

                result[
                    str(
                        current[
                            "speaker"
                        ]
                    )
                ].append(
                    (
                        current[
                            "start"
                        ],
                        current[
                            "end"
                        ],
                        str(
                            current[
                                "face_id"
                            ]
                        ),
                        confidence,
                    )
                )

            current = {
                "start":
                    item["time"],
                "end":
                    item["time"],
                "speaker":
                    speaker,
                "face_id":
                    face_id,
                "confidence_sum":
                    0.0,
                "samples":
                    0,
            }

        else:

            current["end"] = (
                item["time"]
            )

        current[
            "confidence_sum"
        ] += to_float(
            item.get(
                "stable_confidence",
                0.0,
            )
        )

        current[
            "samples"
        ] += 1

    if current is not None:

        if (
            current["speaker"]
            is not None
            and current["face_id"]
            is not None
        ):

            confidence = (
                current[
                    "confidence_sum"
                ]
                / max(
                    1,
                    current[
                        "samples"
                    ],
                )
            )

            result[
                str(
                    current[
                        "speaker"
                    ]
                )
            ].append(
                (
                    current[
                        "start"
                    ],
                    current[
                        "end"
                    ],
                    str(
                        current[
                            "face_id"
                        ]
                    ),
                    confidence,
                )
            )

    return result


def face_for_speaker_at(
    speaker: str,
    time_value: float,
    segments: Dict[
        str,
        List[
            Tuple[
                float,
                float,
                str,
                float,
            ]
        ],
    ],
) -> Optional[Tuple[str, float]]:

    candidates = segments.get(
        speaker,
        [],
    )

    if not candidates:
        return None

    # First priority: currently inside a segment.
    for (
        start,
        end,
        face_id,
        confidence,
    ) in candidates:

        if (
            start
            <= time_value
            <= end
        ):

            return (
                face_id,
                confidence,
            )

    # Second priority: nearest segment, but only when close.
    best = None
    best_distance = float(
        "inf"
    )

    for (
        start,
        end,
        face_id,
        confidence,
    ) in candidates:

        if time_value < start:

            distance = (
                start
                - time_value
            )

        else:

            distance = (
                time_value
                - end
            )

        if distance < best_distance:

            best_distance = distance
            best = (
                face_id,
                confidence,
            )

    if (
        best is None
        or best_distance
        > MAX_SPEAKER_FACE_DISTANCE
    ):
        return None

    return best


# ============================================================
# FACE FRAMING
# ============================================================

def make_closeup_crop(
    bbox: FaceBox,
    source_width: float,
    source_height: float,
) -> Dict[str, float]:

    # Normalize face dimensions.
    face_w = (
        bbox.w
        / source_width
    )

    face_h = (
        bbox.h
        / source_height
    )

    center_x = (
        bbox.cx
        / source_width
    )

    center_y = (
        bbox.cy
        / source_height
    )

    # Desired crop height.
    crop_h = clamp(
        face_h
        / TARGET_FACE_HEIGHT_RATIO,
        0.25,
        1.0,
    )

    # 9:16 vertical crop.
    crop_w = clamp(
        crop_h
        * (9.0 / 16.0),
        0.20,
        1.0,
    )

    # Small upward bias to keep head in a natural position.
    target_center_y = clamp(
        center_y
        - 0.04,
        0.0,
        1.0,
    )

    crop_x = clamp(
        center_x
        - crop_w / 2.0,
        0.0,
        1.0 - crop_w,
    )

    crop_y = clamp(
        target_center_y
        - crop_h / 2.0,
        0.0,
        1.0 - crop_h,
    )

    return {
        "x":
            round(
                crop_x,
                5,
            ),
        "y":
            round(
                crop_y,
                5,
            ),
        "width":
            round(
                crop_w,
                5,
            ),
        "height":
            round(
                crop_h,
                5,
            ),
        "face_width_ratio":
            round(
                face_w,
                5,
            ),
        "face_height_ratio":
            round(
                face_h,
                5,
            ),
        "mode":
            "face_closeup",
    }


# ============================================================
# DIRECTOR
# ============================================================

class PodcastDirector:

    def __init__(
        self,
        stable_timeline: List[Dict[str, Any]],
        face_tracks: Dict[
            str,
            List[
                Tuple[
                    float,
                    FaceBox,
                ]
            ],
        ],
        moments: List[Dict[str, Any]],
        source_width: float = 1080.0,
        source_height: float = 1920.0,
    ) -> None:

        self.stable_timeline = (
            stable_timeline
        )

        self.face_tracks = (
            face_tracks
        )

        self.moments = (
            moments
        )

        self.source_width = (
            source_width
        )

        self.source_height = (
            source_height
        )

        self.speaker_face_segments = (
            build_speaker_face_segments(
                stable_timeline
            )
        )

    # --------------------------------------------------------
    # Face lookup.
    # --------------------------------------------------------

    def closeup_possible(
        self,
        face_id: str,
        time_value: float,
    ) -> Optional[
        Dict[str, Any]
    ]:

        samples = self.face_tracks.get(
            str(face_id)
        )

        if not samples:
            return None

        bbox = bbox_at_time(
            samples,
            time_value,
        )

        if bbox is None:
            return None

        face_height_ratio = (
            bbox.h
            / self.source_height
        )

        if (
            face_height_ratio
            < MIN_FACE_HEIGHT_RATIO
        ):
            return None

        crop = make_closeup_crop(
            bbox,
            self.source_width,
            self.source_height,
        )

        return {
            "bbox": {
                "x":
                    bbox.x,
                "y":
                    bbox.y,
                "width":
                    bbox.w,
                "height":
                    bbox.h,
            },
            "crop":
                crop,
        }

    # --------------------------------------------------------
    # Generate candidate speaker closeups.
    # --------------------------------------------------------

    def generate_candidates(
        self,
    ) -> List[DirectorAction]:

        candidates: List[
            DirectorAction
        ] = []

        for moment in self.moments:

            intensity = clamp(
                to_float(
                    moment.get(
                        "intensity",
                        moment.get(
                            "score",
                            0.0,
                        ),
                    )
                )
            )

            speech = clamp(
                to_float(
                    moment.get(
                        "speech",
                        0.0,
                    )
                )
            )

            if (
                intensity
                < MIN_MOMENT_INTENSITY
            ):
                continue

            raw_start = to_float(
                moment.get(
                    "start"
                )
            )

            raw_end = to_float(
                moment.get(
                    "end",
                    raw_start,
                )
            )

            center = (
                raw_start
                + raw_end
            ) / 2.0

            # ------------------------------------------------
            # IMPORTANT:
            #
            # Use speaker state at the actual acoustic moment.
            # ------------------------------------------------

            speaker = active_speaker_at(
                center,
                self.stable_timeline,
            )

            if speaker is None:
                continue

            # ------------------------------------------------
            # A speaker close-up requires speech.
            # Audio-only moments are ignored for now.
            # ------------------------------------------------

            if speech < 0.40:
                continue

            face_result = face_for_speaker_at(
                speaker,
                center,
                self.speaker_face_segments,
            )

            if face_result is None:
                continue

            face_id, confidence = (
                face_result
            )

            required_confidence = (
                STRONG_MOMENT_CONFIDENCE
                if intensity
                >= STRONG_MOMENT_INTENSITY
                else MIN_SPEAKER_CONFIDENCE
            )

            if (
                confidence
                < required_confidence
            ):
                continue

            visual = self.closeup_possible(
                face_id,
                center,
            )

            if visual is None:
                continue

            start = max(
                0.0,
                raw_start
                - PRE_ROLL,
            )

            end = (
                raw_end
                + POST_ROLL
            )

            if (
                end - start
                < MIN_CLOSEUP_DURATION
            ):
                continue

            reason = (
                "strong_speaker_moment"
                if intensity
                >= STRONG_MOMENT_INTENSITY
                else
                "important_speaker_moment"
            )

            candidates.append(
                DirectorAction(
                    start=start,
                    end=end,
                    shot="speaker_closeup",
                    speaker=speaker,
                    face_id=face_id,
                    confidence=confidence,
                    reason=reason,
                    importance=intensity,
                    crop=visual[
                        "crop"
                    ],
                )
            )

        return candidates

    # --------------------------------------------------------
    # Select non-overlapping camera events.
    # --------------------------------------------------------

    def merge_actions(
        self,
        actions: List[DirectorAction],
    ) -> List[DirectorAction]:

        if not actions:
            return []

        actions = sorted(
            actions,
            key=lambda action:
                (
                    action.start,
                    -action.importance,
                ),
        )

        selected: List[
            DirectorAction
        ] = []

        for action in actions:

            if not selected:

                selected.append(
                    action
                )

                continue

            previous = selected[
                -1
            ]

            overlap = (
                action.start
                <= previous.end
            )

            too_close = (
                action.start
                - previous.end
                < MIN_GAP_BETWEEN_EVENTS
            )

            if overlap or too_close:

                # Prefer the stronger event.
                if (
                    action.importance
                    > previous.importance
                ):

                    selected[
                        -1
                    ] = action

                else:

                    previous.end = max(
                        previous.end,
                        action.end,
                    )

            else:

                selected.append(
                    action
                )

        return selected

    # --------------------------------------------------------
    # Fill gaps with the base two-shot.
    # --------------------------------------------------------

    def build_full_plan(
        self,
        actions: List[DirectorAction],
        video_start: float,
        video_end: float,
    ) -> List[DirectorAction]:

        if (
            video_end
            <= video_start
        ):
            return []

        plan: List[
            DirectorAction
        ] = []

        cursor = video_start

        for action in actions:

            start = max(
                video_start,
                min(
                    action.start,
                    video_end,
                ),
            )

            end = max(
                start,
                min(
                    action.end,
                    video_end,
                ),
            )

            if start > cursor:

                plan.append(
                    DirectorAction(
                        start=cursor,
                        end=start,
                        shot="two_shot",
                        speaker=None,
                        face_id=None,
                        confidence=1.0,
                        reason=(
                            "default_conversation_framing"
                        ),
                        importance=0.0,
                        crop=None,
                    )
                )

            if end > start:

                plan.append(
                    DirectorAction(
                        start=start,
                        end=end,
                        shot=action.shot,
                        speaker=action.speaker,
                        face_id=action.face_id,
                        confidence=action.confidence,
                        reason=action.reason,
                        importance=action.importance,
                        crop=action.crop,
                    )
                )

            cursor = max(
                cursor,
                end,
            )

        if cursor < video_end:

            plan.append(
                DirectorAction(
                    start=cursor,
                    end=video_end,
                    shot="two_shot",
                    speaker=None,
                    face_id=None,
                    confidence=1.0,
                    reason=(
                        "default_conversation_framing"
                    ),
                    importance=0.0,
                    crop=None,
                )
            )

        return plan

    # --------------------------------------------------------
    # PUBLIC API.
    # --------------------------------------------------------

    def build_plan(
        self,
        video_start: float = 0.0,
        video_end: float = 0.0,
    ) -> List[DirectorAction]:

        if video_end <= video_start:

            raise ValueError(
                "video_end must be greater than video_start."
            )

        print(
            "Generating event-based camera plan..."
        )

        candidates = (
            self.generate_candidates()
        )

        print(
            f"Candidate camera events: "
            f"{len(candidates)}"
        )

        selected = self.merge_actions(
            candidates
        )

        print(
            f"Selected camera events: "
            f"{len(selected)}"
        )

        return self.build_full_plan(
            actions=selected,
            video_start=video_start,
            video_end=video_end,
        )


# ============================================================
# SERIALIZATION
# ============================================================

def serialize_plan(
    plan: List[DirectorAction],
) -> List[Dict[str, Any]]:

    output = []

    for action in plan:

        item = asdict(
            action
        )

        item[
            "start"
        ] = round(
            action.start,
            3,
        )

        item[
            "end"
        ] = round(
            action.end,
            3,
        )

        item[
            "confidence"
        ] = round(
            action.confidence,
            4,
        )

        item[
            "importance"
        ] = round(
            action.importance,
            4,
        )

        output.append(
            item
        )

    return output


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Event-based AI podcast camera director."
        )
    )

    parser.add_argument(
        "--stable",
        required=True,
        help=(
            "Temporally stabilized speaker-face JSON"
        ),
    )

    parser.add_argument(
        "--faces",
        required=True,
        help=(
            "YuNet v5 face tracks JSON"
        ),
    )

    parser.add_argument(
        "--moments",
        required=True,
        help=(
            "Moment intelligence JSON"
        ),
    )

    parser.add_argument(
        "--video-start",
        type=float,
        default=0.0,
        help=(
            "Video region start in seconds"
        ),
    )

    parser.add_argument(
        "--video-end",
        type=float,
        required=True,
        help=(
            "Video region end in seconds"
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Output camera plan JSON"
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    print(
        "Loading stable speaker-face timeline..."
    )

    stable_data = load_json(
        args.stable
    )

    stable_timeline = stable_data.get(
        "timeline",
        [],
    )

    print(
        f"Stable timeline samples: "
        f"{len(stable_timeline)}"
    )

    print()

    print(
        "Loading face tracks..."
    )

    face_data = load_json(
        args.faces
    )

    source_width = to_float(
        face_data.get(
            "video_width",
            1080,
        ),
        1080,
    )

    source_height = to_float(
        face_data.get(
            "video_height",
            1920,
        ),
        1920,
    )

    face_tracks = load_face_tracks(
        args.faces
    )

    print(
        f"Face tracks: "
        f"{len(face_tracks)}"
    )

    print()

    print(
        "Loading moments..."
    )

    moments = load_moments(
        args.moments
    )

    # --------------------------------------------------------
    # Restrict moments to requested video window.
    # --------------------------------------------------------

    filtered_moments = []

    for moment in moments:

        if (
            moment["end"]
            < args.video_start
        ):
            continue

        if (
            moment["start"]
            > args.video_end
        ):
            continue

        filtered_moments.append(
            moment
        )

    print(
        f"Moments in requested window: "
        f"{len(filtered_moments)}"
    )

    for moment in filtered_moments[:10]:

        print(
            f"  "
            f"{moment['start']:.2f}s -> "
            f"{moment['end']:.2f}s | "
            f"intensity="
            f"{moment['intensity']:.3f} | "
            f"speech="
            f"{moment['speech']:.1f}"
        )

    print()

    print(
        "Building podcast director..."
    )

    director = PodcastDirector(
        stable_timeline=stable_timeline,
        face_tracks=face_tracks,
        moments=filtered_moments,
        source_width=source_width,
        source_height=source_height,
    )

    plan = director.build_plan(
        video_start=args.video_start,
        video_end=args.video_end,
    )

    camera_takeovers = [
        action
        for action in plan
        if action.shot
        != "two_shot"
    ]

    two_shots = [
        action
        for action in plan
        if action.shot
        == "two_shot"
    ]

    print()

    print(
        f"Total camera actions: "
        f"{len(plan)}"
    )

    print(
        f"Speaker close-ups: "
        f"{len(camera_takeovers)}"
    )

    print(
        f"Two-shot sections: "
        f"{len(two_shots)}"
    )

    print()

    print(
        "CAMERA PLAN"
    )

    for index, action in enumerate(
        plan,
        start=1,
    ):

        print(
            f"  {index:02d}. "
            f"{action.start:.2f}s -> "
            f"{action.end:.2f}s | "
            f"{action.shot:18s} | "
            f"speaker="
            f"{action.speaker or '-'} | "
            f"face="
            f"{action.face_id or '-'} | "
            f"conf="
            f"{action.confidence:.3f} | "
            f"importance="
            f"{action.importance:.3f}"
        )

    output = {
        "version": 2,
        "mode":
            "podcast_event_based_director",

        "stable_file":
            os.path.abspath(
                args.stable
            ),

        "faces_file":
            os.path.abspath(
                args.faces
            ),

        "moments_file":
            os.path.abspath(
                args.moments
            ),

        "video_start":
            args.video_start,

        "video_end":
            args.video_end,

        "source_width":
            source_width,

        "source_height":
            source_height,

        "plan":
            serialize_plan(
                plan
            ),

        "stats": {
            "timeline_samples":
                len(stable_timeline),

            "face_tracks":
                len(face_tracks),

            "all_moments":
                len(moments),

            "window_moments":
                len(
                    filtered_moments
                ),

            "camera_actions":
                len(plan),

            "speaker_closeups":
                len(
                    camera_takeovers
                ),

            "two_shots":
                len(
                    two_shots
                ),
        },

        "config": {
            "min_speaker_confidence":
                MIN_SPEAKER_CONFIDENCE,

            "strong_moment_confidence":
                STRONG_MOMENT_CONFIDENCE,

            "min_moment_intensity":
                MIN_MOMENT_INTENSITY,

            "strong_moment_intensity":
                STRONG_MOMENT_INTENSITY,

            "min_closeup_duration":
                MIN_CLOSEUP_DURATION,

            "pre_roll":
                PRE_ROLL,

            "post_roll":
                POST_ROLL,

            "min_gap_between_events":
                MIN_GAP_BETWEEN_EVENTS,
        },
    }

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    print()

    print(
        f"Saved: "
        f"{output_path.resolve()}"
    )


if __name__ == "__main__":
    main()
