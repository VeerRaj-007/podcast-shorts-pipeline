import json
import math
import subprocess
from pathlib import Path

import numpy as np


# ============================================================
# CONFIG
# ============================================================

AUDIO_PATH = Path("tmp/audio.mp3")
TRANSCRIPT_PATH = Path("tmp/aligned_transcript.json")
OUTPUT_PATH = Path("tmp/moments.json")

BIN_SIZE = 0.5
SAMPLE_RATE = 8000

# Number of audio samples in one intelligence bin
SAMPLES_PER_BIN = int(SAMPLE_RATE * BIN_SIZE)

# How many moments to report
TOP_MOMENTS = 30


# ============================================================
# HELPERS
# ============================================================

def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def robust_normalize(values):
    """
    Normalize values using percentile-based bounds.

    This prevents one extremely loud spike from destroying
    the useful range of the rest of the signal.
    """

    if not values:
        return []

    arr = np.asarray(values, dtype=np.float32)

    low = float(np.percentile(arr, 5))
    high = float(np.percentile(arr, 95))

    if high - low < 1e-8:
        return [0.5 for _ in arr]

    normalized = (arr - low) / (high - low)
    normalized = np.clip(normalized, 0.0, 1.0)

    return normalized.tolist()


def smooth(values, radius=2):
    """
    Simple moving average.
    """

    if not values:
        return []

    arr = np.asarray(values, dtype=np.float32)

    if radius <= 0:
        return arr.tolist()

    kernel_size = radius * 2 + 1
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size

    padded = np.pad(arr, (radius, radius), mode="edge")
    result = np.convolve(padded, kernel, mode="valid")

    return result.tolist()


# ============================================================
# LOAD TRANSCRIPT
# ============================================================

def load_transcript():
    print("Loading aligned transcript...")

    with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    words = data.get("word_segments", [])

    print(f"Words loaded: {len(words):,}")
    print(f"Segments loaded: {len(segments):,}")

    return data


# ============================================================
# TIMELINE
# ============================================================

def get_duration(data):
    """
    Determine timeline duration from the aligned transcript.
    """

    words = data.get("word_segments", [])

    if words:
        return max(
            float(word["end"])
            for word in words
            if "end" in word
        )

    segments = data.get("segments", [])

    if segments:
        return max(
            float(segment["end"])
            for segment in segments
            if "end" in segment
        )

    return 0.0


def build_timeline(duration):
    count = int(math.ceil(duration / BIN_SIZE))

    timeline = []

    for i in range(count):
        start = i * BIN_SIZE
        end = min(start + BIN_SIZE, duration)

        timeline.append(
            {
                "index": i,
                "start": start,
                "end": end,
            }
        )

    print(f"Timeline bins: {len(timeline):,}")

    return timeline


# ============================================================
# SPEECH FEATURES
# ============================================================

def calculate_speech_features(timeline, data):
    print("Calculating speech features...")

    words = data.get("word_segments", [])

    speech_density = [0.0] * len(timeline)
    emphasis = [0.0] * len(timeline)
    questions = [0.0] * len(timeline)
    exclamations = [0.0] * len(timeline)

    # --------------------------------------------------------
    # Word-level features
    # --------------------------------------------------------

    for word in words:
        if "start" not in word or "end" not in word:
            continue

        start = float(word["start"])
        end = float(word["end"])

        index = int(start / BIN_SIZE)

        if index < 0 or index >= len(timeline):
            continue

        text = str(word.get("word", "")).strip()

        speech_density[index] += 1.0

        upper = text.upper()

        # Emphasis indicators
        if len(text) >= 7:
            emphasis[index] += 0.15

        if upper in {
            "WOW",
            "WHAT",
            "WHY",
            "NO",
            "YES",
            "OH",
            "OMG",
            "OHMYGOD",
            "NEVER",
            "ALWAYS",
        }:
            emphasis[index] += 0.5

    # --------------------------------------------------------
    # Segment-level features
    # --------------------------------------------------------

    for segment in data.get("segments", []):
        if "start" not in segment or "end" not in segment:
            continue

        start = float(segment["start"])
        end = float(segment["end"])
        text = str(segment.get("text", "")).strip()

        index = int(start / BIN_SIZE)

        if index < 0 or index >= len(timeline):
            continue

        if "?" in text:
            questions[index] += 1.0

        if "!" in text:
            exclamations[index] += 1.0

    speech_density = robust_normalize(speech_density)
    emphasis = robust_normalize(emphasis)
    questions = robust_normalize(questions)
    exclamations = robust_normalize(exclamations)

    return {
        "speech_density": speech_density,
        "emphasis": emphasis,
        "questions": questions,
        "exclamations": exclamations,
    }


# ============================================================
# AUDIO INTELLIGENCE
# ============================================================

def calculate_audio_rms():
    """
    Decode the audio with FFmpeg into raw mono PCM and calculate
    RMS energy for every BIN_SIZE interval.

    This is much more reliable than trying to parse FFmpeg's
    human-readable stderr statistics.
    """

    print("Analyzing audio energy with FFmpeg...")
    print(f"Audio sample rate: {SAMPLE_RATE} Hz")
    print(f"Audio bin size: {BIN_SIZE}s")

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",

        "-i",
        str(AUDIO_PATH),

        "-vn",

        "-ac",
        "1",

        "-ar",
        str(SAMPLE_RATE),

        "-f",
        "s16le",

        "-"
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    rms_values = []

    bytes_per_sample = 2
    bytes_per_bin = SAMPLES_PER_BIN * bytes_per_sample

    while True:
        raw = process.stdout.read(bytes_per_bin)

        if not raw:
            break

        samples = np.frombuffer(
            raw,
            dtype=np.int16,
        ).astype(np.float32)

        if samples.size == 0:
            continue

        # Convert int16 to [-1, 1]
        samples /= 32768.0

        # Root Mean Square
        rms = float(
            np.sqrt(
                np.mean(
                    samples * samples
                )
            )
        )

        rms_values.append(rms)

    stderr = process.stderr.read().decode(
        "utf-8",
        errors="ignore",
    )

    return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            "FFmpeg audio decoding failed:\n"
            + stderr
        )

    print(f"Audio measurements: {len(rms_values):,}")

    if rms_values:
        print(
            "Raw RMS range: "
            f"{min(rms_values):.6f} → "
            f"{max(rms_values):.6f}"
        )

    return rms_values


def prepare_audio_features(timeline):
    rms = calculate_audio_rms()

    if not rms:
        print("WARNING: No audio measurements found.")
        return [0.0] * len(timeline)

    normalized = robust_normalize(rms)

    # Smooth tiny frame-to-frame fluctuations
    normalized = smooth(
        normalized,
        radius=2,
    )

    # Match audio timeline to transcript timeline
    if len(normalized) < len(timeline):
        last = normalized[-1]

        normalized.extend(
            [last] * (
                len(timeline) - len(normalized)
            )
        )

    elif len(normalized) > len(timeline):
        normalized = normalized[:len(timeline)]

    print(
        "Normalized audio range: "
        f"{min(normalized):.3f} → "
        f"{max(normalized):.3f}"
    )

    return normalized


# ============================================================
# DERIVED INTELLIGENCE
# ============================================================

def calculate_derivatives(values):
    """
    Measures how quickly a signal changes.

    Positive values mean rising intensity.
    """

    if not values:
        return []

    result = [0.0]

    for i in range(1, len(values)):
        result.append(
            max(
                0.0,
                values[i] - values[i - 1],
            )
        )

    return robust_normalize(result)


def calculate_combined_intensity(
    speech,
    emphasis,
    questions,
    exclamations,
    audio,
):
    print("Combining intelligence signals...")

    audio_rise = calculate_derivatives(audio)
    speech_rise = calculate_derivatives(speech)

    intensity = []

    for i in range(len(speech)):

        score = (
            speech[i] * 0.25
            + audio[i] * 0.30
            + audio_rise[i] * 0.15
            + speech_rise[i] * 0.10
            + emphasis[i] * 0.10
            + questions[i] * 0.05
            + exclamations[i] * 0.05
        )

        intensity.append(
            clamp(score)
        )

    intensity = smooth(
        intensity,
        radius=2,
    )

    return intensity


# ============================================================
# CAMERA SCALE
# ============================================================

def intensity_to_camera_scale(intensity):
    """
    Convert moment intensity into a subtle camera zoom.

    0.0 → 1.00x
    1.0 → 1.16x
    """

    return round(
        1.0 + (
            clamp(intensity) * 0.16
        ),
        4,
    )


# ============================================================
# TOP MOMENTS
# ============================================================

def find_top_moments(
    timeline,
    intensity,
    audio,
    speech,
):
    """
    Select local peaks instead of simply selecting the loudest
    individual bins.
    """

    candidates = []

    for i in range(1, len(timeline) - 1):

        current = intensity[i]

        if current < intensity[i - 1]:
            continue

        if current < intensity[i + 1]:
            continue

        candidates.append(
            {
                "index": i,
                "start": timeline[i]["start"],
                "end": timeline[i]["end"],
                "intensity": current,
                "audio": audio[i],
                "speech": speech[i],
                "camera_scale": intensity_to_camera_scale(
                    current
                ),
            }
        )

    candidates.sort(
        key=lambda x: x["intensity"],
        reverse=True,
    )

    # --------------------------------------------------------
    # Non-maximum suppression.
    #
    # Don't return 10 peaks that are all from the same
    # exciting 5-second moment.
    # --------------------------------------------------------

    selected = []

    MIN_DISTANCE = 5.0

    for candidate in candidates:

        too_close = False

        for existing in selected:

            distance = abs(
                candidate["start"]
                - existing["start"]
            )

            if distance < MIN_DISTANCE:
                too_close = True
                break

        if too_close:
            continue

        selected.append(candidate)

        if len(selected) >= TOP_MOMENTS:
            break

    return selected


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("MOMENT INTELLIGENCE ENGINE")
    print("=" * 60)

    if not AUDIO_PATH.exists():
        raise FileNotFoundError(
            f"Missing audio file: {AUDIO_PATH}"
        )

    if not TRANSCRIPT_PATH.exists():
        raise FileNotFoundError(
            f"Missing transcript: {TRANSCRIPT_PATH}"
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    data = load_transcript()

    duration = get_duration(data)

    print(
        f"Timeline duration: "
        f"{duration:.2f}s"
    )

    timeline = build_timeline(
        duration
    )

    # --------------------------------------------------------
    # Speech intelligence
    # --------------------------------------------------------

    speech_features = calculate_speech_features(
        timeline,
        data,
    )

    # --------------------------------------------------------
    # Audio intelligence
    # --------------------------------------------------------

    audio = prepare_audio_features(
        timeline
    )

    # --------------------------------------------------------
    # Combined intelligence
    # --------------------------------------------------------

    intensity = calculate_combined_intensity(
        speech=speech_features["speech_density"],
        emphasis=speech_features["emphasis"],
        questions=speech_features["questions"],
        exclamations=speech_features["exclamations"],
        audio=audio,
    )

    # --------------------------------------------------------
    # Find peaks
    # --------------------------------------------------------

    moments = find_top_moments(
        timeline=timeline,
        intensity=intensity,
        audio=audio,
        speech=speech_features["speech_density"],
    )

    # --------------------------------------------------------
    # Add human-readable information
    # --------------------------------------------------------

    for rank, moment in enumerate(
        moments,
        start=1,
    ):

        moment["rank"] = rank

        moment["duration"] = round(
            moment["end"]
            - moment["start"],
            3,
        )

        moment["intensity"] = round(
            moment["intensity"],
            4,
        )

        moment["audio"] = round(
            moment["audio"],
            4,
        )

        moment["speech"] = round(
            moment["speech"],
            4,
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

        # --------------------------------------------------------
    # Build full intelligence timeline
    # --------------------------------------------------------

    intelligence_timeline = []

    for i, point in enumerate(timeline):

        intelligence_timeline.append(
            {
                "start": round(point["start"], 3),
                "end": round(point["end"], 3),

                "intensity": round(
                    intensity[i],
                    4,
                ),

                "audio": round(
                    audio[i],
                    4,
                ),

                "speech": round(
                    speech_features["speech_density"][i],
                    4,
                ),

                "emphasis": round(
                    speech_features["emphasis"][i],
                    4,
                ),

                "camera_scale": intensity_to_camera_scale(
                    intensity[i]
                ),
            }
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output = {
        "version": 4,

        "bin_size": BIN_SIZE,

        "duration": round(
            duration,
            3,
        ),

        "timeline_bins": len(timeline),

        "audio_measurements": len(audio),

        # Full continuous intelligence curve
        "timeline": intelligence_timeline,

        # Highest local peaks
        "moments": moments,
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Debug output
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("TOP MOMENTS")
    print("=" * 60)

    for moment in moments[:10]:

        print(
            f"#{moment['rank']:02d} "
            f"{moment['start']:.2f}s → "
            f"{moment['end']:.2f}s | "
            f"intensity={moment['intensity']:.3f} | "
            f"audio={moment['audio']:.3f} | "
            f"speech={moment['speech']:.3f} | "
            f"zoom={moment['camera_scale']:.3f}x"
        )

    print()
    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()