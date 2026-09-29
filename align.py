import json
import os
import sys

import torch
import whisperx


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

AUDIO_PATH = "tmp/audio.mp3"
TRANSCRIPT_PATH = "tmp/transcript.json"
OUTPUT_PATH = "tmp/aligned_transcript.json"


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

LANGUAGE = "en"

# Automatically use GPU if CUDA is available.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CPU uses int8.
# GPU uses float16.
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"


def load_transcript(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        segments = data.get("segments")

        if isinstance(segments, list):
            return segments

    raise ValueError(
        f"Could not find transcript segments in {path}"
    )


def clean_segments(segments):
    """
    Keep only the fields WhisperX needs.
    """

    cleaned = []

    for segment in segments:

        try:
            start = float(segment["start"])
            end = float(segment["end"])
            text = str(segment.get("text", "")).strip()
        except (KeyError, TypeError, ValueError):
            continue

        if not text:
            continue

        if end <= start:
            continue

        cleaned.append(
            {
                "start": start,
                "end": end,
                "text": text,
            }
        )

    return cleaned


def main():

    print()
    print("=" * 60)
    print("WHISPERX FORCED WORD ALIGNMENT")
    print("=" * 60)
    print()

    # -----------------------------------------------------
    # Check files
    # -----------------------------------------------------

    if not os.path.exists(AUDIO_PATH):
        print(f"ERROR: Audio not found:")
        print(f"  {AUDIO_PATH}")
        sys.exit(1)

    if not os.path.exists(TRANSCRIPT_PATH):
        print(f"ERROR: Transcript not found:")
        print(f"  {TRANSCRIPT_PATH}")
        sys.exit(1)

    # -----------------------------------------------------
    # Load transcript
    # -----------------------------------------------------

    print("[1/4] Loading transcript...")

    original_segments = load_transcript(
        TRANSCRIPT_PATH
    )

    segments = clean_segments(
        original_segments
    )

    print(
        f"      Loaded {len(segments)} transcript segments"
    )

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    print()
    print("[2/4] Selecting device...")

    print(f"      Device: {DEVICE}")
    print(f"      Compute type: {COMPUTE_TYPE}")

    if DEVICE == "cuda":
        print(
            f"      GPU: {torch.cuda.get_device_name(0)}"
        )
    else:
        print(
            "      Using CPU. Alignment will be slower."
        )

    # -----------------------------------------------------
    # Load audio
    # -----------------------------------------------------

    print()
    print("[3/4] Loading audio...")

    audio = whisperx.load_audio(
        AUDIO_PATH
    )

    print("      Audio loaded.")

    # -----------------------------------------------------
    # Load alignment model
    # -----------------------------------------------------

    print()
    print("[4/4] Loading English alignment model...")

    align_model, metadata = (
        whisperx.load_align_model(
            language_code=LANGUAGE,
            device=DEVICE,
        )
    )

    print("      Alignment model loaded.")

    # -----------------------------------------------------
    # FORCE ALIGNMENT
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("ALIGNING WORDS")
    print("=" * 60)
    print()
    print(
        "This may take some time for a long recording."
    )
    print()

    result = whisperx.align(
        segments,
        align_model,
        metadata,
        audio,
        DEVICE,
        return_char_alignments=False,
    )

    # -----------------------------------------------------
    # Extract aligned segments
    # -----------------------------------------------------

    aligned_segments = result.get(
        "segments",
        []
    )

    word_segments = result.get(
        "word_segments",
        []
    )

    print()
    print(
        f"Aligned segments: {len(aligned_segments)}"
    )

    print(
        f"Aligned words:    {len(word_segments)}"
    )

    # -----------------------------------------------------
    # Convert to the format used by captions.py
    # -----------------------------------------------------

    output_segments = []

    for segment in aligned_segments:

        text = str(
            segment.get("text", "")
        ).strip()

        if not text:
            continue

        output_segment = {
            "start": float(
                segment["start"]
            ),
            "end": float(
                segment["end"]
            ),
            "text": text,
            "words": [],
        }

        for word in segment.get(
            "words",
            []
        ):

            try:
                word_text = str(
                    word.get("word", "")
                ).strip()

                word_start = float(
                    word["start"]
                )

                word_end = float(
                    word["end"]
                )

            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                continue

            if not word_text:
                continue

            if word_end <= word_start:
                continue

            output_segment["words"].append(
                {
                    "word": word_text,
                    "start": word_start,
                    "end": word_end,
                }
            )

        output_segments.append(
            output_segment
        )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    output = {
        "language": LANGUAGE,
        "segments": output_segments,
        "word_segments": word_segments,
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
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    total_words = sum(
        len(segment["words"])
        for segment in output_segments
    )

    print()
    print("=" * 60)
    print("ALIGNMENT COMPLETE")
    print("=" * 60)
    print()
    print(
        f"Segments: {len(output_segments)}"
    )
    print(
        f"Words with timestamps: {total_words}"
    )
    print()
    print(
        f"Saved to:"
    )
    print(
        f"  {OUTPUT_PATH}"
    )
    print()


if __name__ == "__main__":
    main()