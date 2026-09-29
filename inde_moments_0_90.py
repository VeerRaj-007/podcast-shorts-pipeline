
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


# ============================================================
# CONFIG
# ============================================================

BIN_SIZE = 0.5
SAMPLE_RATE = 8000
TOP_MOMENTS = 30

# Minimum distance between selected moment centers.
MIN_MOMENT_SEPARATION = 2.0


# ============================================================
# AUDIO
# ============================================================

def load_audio_pcm(
    audio_path: str,
) -> np.ndarray:
    """
    Convert audio to mono 8 kHz signed PCM using FFmpeg.
    """

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        audio_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "s16le",
        "-",
    ]

    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    audio = np.frombuffer(
        process.stdout,
        dtype=np.int16,
    ).astype(
        np.float32
    )

    if audio.size == 0:
        raise RuntimeError(
            "FFmpeg returned no audio samples."
        )

    return audio / 32768.0


# ============================================================
# NORMALIZATION
# ============================================================

def robust_normalize(
    values: np.ndarray,
) -> np.ndarray:
    """
    Robust percentile normalization to 0..1.
    """

    if values.size == 0:
        return values

    low = float(
        np.percentile(
            values,
            10,
        )
    )

    high = float(
        np.percentile(
            values,
            95,
        )
    )

    if high <= low:
        return np.zeros_like(
            values
        )

    normalized = (
        values - low
    ) / (
        high - low
    )

    return np.clip(
        normalized,
        0.0,
        1.0,
    )


def smooth(
    values: np.ndarray,
    radius: int = 3,
) -> np.ndarray:
    """
    Moving-average smoothing.
    """

    if len(values) < 2:
        return values.copy()

    kernel_size = (
        radius * 2 + 1
    )

    kernel = (
        np.ones(
            kernel_size,
            dtype=np.float32,
        )
        / kernel_size
    )

    padded = np.pad(
        values,
        (
            radius,
            radius,
        ),
        mode="edge",
    )

    return np.convolve(
        padded,
        kernel,
        mode="valid",
    )


# ============================================================
# AUDIO TIMELINE
# ============================================================

def compute_bins(
    audio: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Compute 0.5-second acoustic intensity bins.
    """

    samples_per_bin = int(
        BIN_SIZE * SAMPLE_RATE
    )

    bin_count = int(
        np.ceil(
            len(audio)
            / samples_per_bin
        )
    )

    rms_values = []

    for index in range(
        bin_count
    ):

        start = (
            index
            * samples_per_bin
        )

        end = min(
            len(audio),
            start
            + samples_per_bin,
        )

        chunk = audio[
            start:end
        ]

        if chunk.size == 0:

            rms = 0.0

        else:

            rms = float(
                np.sqrt(
                    np.mean(
                        chunk * chunk
                    )
                )
            )

        rms_values.append(
            rms
        )

    rms = np.asarray(
        rms_values,
        dtype=np.float32,
    )

    normalized_audio = (
        robust_normalize(
            rms
        )
    )

    smoothed_audio = smooth(
        normalized_audio,
        radius=3,
    )

    # --------------------------------------------------------
    # Temporal acoustic change.
    # --------------------------------------------------------

    delta = np.zeros_like(
        smoothed_audio
    )

    if len(
        smoothed_audio
    ) > 1:

        delta[1:] = np.abs(
            np.diff(
                smoothed_audio
            )
        )

    delta = robust_normalize(
        delta
    )

    # --------------------------------------------------------
    # Combined intensity.
    #
    # Absolute loudness = 70%
    # Sudden acoustic change = 30%
    # --------------------------------------------------------

    intensity = (
        0.70
        * smoothed_audio
        + 0.30
        * delta
    )

    intensity = smooth(
        intensity,
        radius=2,
    )

    bins = []

    for index, value in enumerate(
        intensity
    ):

        start = (
            index
            * BIN_SIZE
        )

        end = (
            start
            + BIN_SIZE
        )

        bins.append(
            {
                "index":
                    index,
                "start":
                    round(
                        start,
                        3,
                    ),
                "end":
                    round(
                        end,
                        3,
                    ),
                "duration":
                    BIN_SIZE,
                "intensity":
                    round(
                        float(value),
                        4,
                    ),
                "audio":
                    round(
                        float(
                            smoothed_audio[
                                index
                            ]
                        ),
                        4,
                    ),
            }
        )

    return bins


# ============================================================
# SPEECH
# ============================================================

def speech_at_time(
    time_value: float,
    timeline: List[Dict[str, Any]],
) -> float:
    """
    Determine whether speech is active near a given time
    using the already-generated speaking-face timeline.
    """

    best = None
    best_distance = float(
        "inf"
    )

    for item in timeline:

        try:

            item_time = float(
                item.get(
                    "time",
                    0.0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            continue

        distance = abs(
            item_time
            - time_value
        )

        if distance < best_distance:

            best_distance = distance
            best = item

    if (
        best is None
        or best_distance > 0.75
    ):

        return 0.0

    return (
        1.0
        if best.get(
            "active_speaker"
        ) is not None
        else 0.0
    )


# ============================================================
# MOMENT SELECTION
# ============================================================

def select_moments(
    bins: List[Dict[str, Any]],
    speaking_timeline: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Select the strongest temporally-separated moments.

    The moment itself remains 0.5 seconds, while the director
    will later expand it into a natural camera event.
    """

    if not bins:
        return []

    ranked = sorted(
        bins,
        key=lambda item:
            item["intensity"],
        reverse=True,
    )

    selected = []

    for item in ranked:

        center = (
            item["start"]
            + item["end"]
        ) / 2.0

        too_close = False

        for previous in selected:

            previous_center = (
                previous["start"]
                + previous["end"]
            ) / 2.0

            if (
                abs(
                    center
                    - previous_center
                )
                < MIN_MOMENT_SEPARATION
            ):

                too_close = True
                break

        if too_close:
            continue

        speech = speech_at_time(
            center,
            speaking_timeline,
        )

        selected.append(
            {
                **item,
                "speech":
                    speech,
            }
        )

        if len(
            selected
        ) >= TOP_MOMENTS:

            break

    # Highest intensity first for output.
    selected.sort(
        key=lambda item:
            item["intensity"],
        reverse=True,
    )

    moments = []

    for rank, item in enumerate(
        selected,
        start=1,
    ):

        moments.append(
            {
                "index":
                    item["index"],
                "start":
                    item["start"],
                "end":
                    item["end"],
                "intensity":
                    item["intensity"],
                "audio":
                    item["audio"],
                "speech":
                    item["speech"],
                "camera_scale":
                    1.0,
                "rank":
                    rank,
                "duration":
                    item["duration"],
            }
        )

    return moments


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    # ========================================================
    # ARGUMENTS
    # ========================================================

    parser = argparse.ArgumentParser(
        description=(
            "Generate local moment intelligence "
            "for the Inde interview test window."
        )
    )

    parser.add_argument(
        "--audio",
        required=True,
        help="Input audio WAV/MP3/etc.",
    )

    parser.add_argument(
        "--speaking",
        required=True,
        help=(
            "Speaking-face timeline JSON, e.g. "
            "tmp/inde_speaking_faces_0_90_v2.json"
        ),
    )

    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Start time in seconds.",
    )

    parser.add_argument(
        "--end",
        type=float,
        default=90.0,
        help="End time in seconds.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path.",
    )

    args = parser.parse_args()

    # ========================================================
    # LOAD SPEAKING TIMELINE
    # ========================================================

    print(
        "Loading speaking timeline..."
    )

    with open(
        args.speaking,
        "r",
        encoding="utf-8",
    ) as f:

        speaking_data = json.load(
            f
        )

    speaking_timeline = (
        speaking_data.get(
            "timeline",
            [],
        )
    )

    print(
        f"Speaking timeline samples: "
        f"{len(speaking_timeline)}"
    )

    # ========================================================
    # LOAD AUDIO
    # ========================================================

    print()
    print(
        "Loading audio..."
    )

    audio = load_audio_pcm(
        args.audio
    )

    # ========================================================
    # CROP REQUESTED WINDOW
    # ========================================================

    start_sample = max(
        0,
        int(
            args.start
            * SAMPLE_RATE
        ),
    )

    end_sample = min(
        len(audio),
        int(
            args.end
            * SAMPLE_RATE
        ),
    )

    if (
        end_sample
        <= start_sample
    ):

        raise ValueError(
            "Invalid time range: "
            f"{args.start} -> {args.end}"
        )

    audio = audio[
        start_sample:end_sample
    ]

    actual_duration = (
        len(audio)
        / SAMPLE_RATE
    )

    print(
        f"Analyzing audio: "
        f"{args.start:.2f}s -> "
        f"{args.end:.2f}s"
    )

    print(
        f"Audio duration: "
        f"{actual_duration:.2f}s"
    )

    # ========================================================
    # COMPUTE TIMELINE
    # ========================================================

    print()
    print(
        "Computing acoustic timeline..."
    )

    bins = compute_bins(
        audio
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # The audio was cropped to args.start.
    # Therefore shift each bin back into original video time.
    # --------------------------------------------------------

    for item in bins:

        item["start"] = round(
            item["start"]
            + args.start,
            3,
        )

        item["end"] = round(
            item["end"]
            + args.start,
            3,
        )

    # ========================================================
    # SELECT MOMENTS
    # ========================================================

    print()
    print(
        "Selecting strongest moments..."
    )

    moments = select_moments(
        bins,
        speaking_timeline,
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    output = {
        "version": 2,
        "mode":
            "inde_local_moment_intelligence",

        "audio":
            str(
                Path(
                    args.audio
                ).resolve()
            ),

        "speaking_file":
            str(
                Path(
                    args.speaking
                ).resolve()
            ),

        "start":
            args.start,

        "end":
            args.end,

        "duration":
            round(
                args.end
                - args.start,
                3,
            ),

        "bin_size":
            BIN_SIZE,

        "timeline_bins":
            len(bins),

        "timeline":
            bins,

        "moments":
            moments,
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

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print(
        f"Audio bins: "
        f"{len(bins)}"
    )

    print(
        f"Selected moments: "
        f"{len(moments)}"
    )

    print()
    print(
        "TOP INDE 0-90s MOMENTS"
    )

    for moment in moments:

        print(
            f"  "
            f"{moment['start']:.2f}s -> "
            f"{moment['end']:.2f}s | "
            f"intensity="
            f"{moment['intensity']:.3f} | "
            f"audio="
            f"{moment['audio']:.3f} | "
            f"speech="
            f"{moment['speech']:.1f}"
        )

    print()
    print(
        f"Saved: "
        f"{output_path.resolve()}"
    )


if __name__ == "__main__":
    main()
