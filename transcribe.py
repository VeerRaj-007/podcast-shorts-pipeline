"""
Step 2: Extract audio + transcribe with segment + word-level timestamps.

Long podcasts are split into chunks before sending to Groq.
Each chunk gets its timestamps shifted back to the original
video timeline.
"""

import os
import subprocess
import time
from typing import Optional

from groq import Groq


CHUNK_SECONDS = 10 * 60
MAX_RETRIES = 4
RETRY_DELAY = 5


def extract_audio(video_path: str, out_dir: str) -> str:
    """Extract compressed mono audio."""

    os.makedirs(out_dir, exist_ok=True)

    audio_path = os.path.join(out_dir, "audio.mp3")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "64k",
        audio_path,
    ]

    subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )

    return audio_path


def get_duration(audio_path: str) -> float:
    """Get audio duration using ffprobe."""

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]

    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )

    return float(result.stdout.strip())


def split_audio(
    audio_path: str,
    out_dir: str,
    chunk_seconds: int = CHUNK_SECONDS,
) -> list[tuple[str, float]]:
    """
    Split audio into chunks.

    Returns:
        [
            (chunk_path, original_start_time),
            ...
        ]
    """

    chunks_dir = os.path.join(out_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    duration = get_duration(audio_path)

    chunks = []

    start = 0.0
    index = 0

    while start < duration:

        chunk_path = os.path.join(
            chunks_dir,
            f"chunk_{index:04d}.mp3",
        )

        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start),
            "-i",
            audio_path,
            "-t",
            str(chunk_seconds),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "64k",
            chunk_path,
        ]

        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )

        chunks.append((chunk_path, start))

        print(
            f"    Created chunk {index + 1}: "
            f"{start / 60:.1f} -> "
            f"{min(start + chunk_seconds, duration) / 60:.1f} min"
        )

        start += chunk_seconds
        index += 1

    return chunks


def _get_value(obj, key, default=None):
    """Read a value from either an object or dictionary."""

    if isinstance(obj, dict):
        return obj.get(key, default)

    return getattr(obj, key, default)


def transcribe_chunk(
    client: Groq,
    chunk_path: str,
) -> list[dict]:
    """
    Transcribe one chunk with REAL word-level timestamps.
    """

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            print(
                f"      Uploading {os.path.basename(chunk_path)} "
                f"(attempt {attempt}/{MAX_RETRIES})..."
            )

            with open(chunk_path, "rb") as f:

                result = client.audio.transcriptions.create(
                    file=f,
                    model="whisper-large-v3",
                    response_format="verbose_json",
                    timestamp_granularities=[
                        "segment",
                        "word",
                    ],
                )

            segments = _get_value(result, "segments", [])

            if segments is None:
                segments = []

            output = []

            for segment in segments:

                start = float(
                    _get_value(segment, "start", 0.0)
                )

                end = float(
                    _get_value(segment, "end", 0.0)
                )

                text = _get_value(
                    segment,
                    "text",
                    "",
                )

                raw_words = _get_value(
                    segment,
                    "words",
                    [],
                )

                words = []

                for word in raw_words or []:

                    word_start = _get_value(
                        word,
                        "start",
                        None,
                    )

                    word_end = _get_value(
                        word,
                        "end",
                        None,
                    )

                    word_text = _get_value(
                        word,
                        "word",
                        "",
                    )

                    if (
                        word_start is None
                        or word_end is None
                        or not word_text
                    ):
                        continue

                    words.append(
                        {
                            "start": float(word_start),
                            "end": float(word_end),
                            "word": str(word_text).strip(),
                        }
                    )

                output.append(
                    {
                        "start": start,
                        "end": end,
                        "text": str(text).strip(),
                        "words": words,
                    }
                )

            return output

        except Exception as e:

            print(
                f"      Transcription failed: {e}"
            )

            if attempt == MAX_RETRIES:
                raise

            wait = RETRY_DELAY * attempt

            print(
                f"      Retrying in {wait} seconds..."
            )

            time.sleep(wait)

    return []


def transcribe(
    audio_path: str,
    api_key: Optional[str] = None,
    tmp_dir: Optional[str] = None,
) -> list[dict]:
    """
    Transcribe the entire audio file.

    Returns segment timestamps AND word-level timestamps.
    """

    if tmp_dir is None:
        tmp_dir = os.path.dirname(audio_path) or "tmp"

    client = Groq(api_key=api_key)

    print("    Splitting audio into chunks...")

    chunks = split_audio(
        audio_path,
        tmp_dir,
        CHUNK_SECONDS,
    )

    print(f"    Created {len(chunks)} chunks")

    all_segments = []

    for index, (chunk_path, offset) in enumerate(chunks):

        print(
            f"\n    [{index + 1}/{len(chunks)}] "
            f"Transcribing chunk..."
        )

        segments = transcribe_chunk(
            client,
            chunk_path,
        )

        for segment in segments:

            # Shift segment timestamps
            # back to original video timeline.

            segment["start"] += offset
            segment["end"] += offset

            # Shift every WORD timestamp too.

            for word in segment.get("words", []):

                word["start"] += offset
                word["end"] += offset

            all_segments.append(segment)

        print(
            f"      -> {len(segments)} segments"
        )

    # Clean up chunks.

    for chunk_path, _ in chunks:

        try:
            os.remove(chunk_path)

        except OSError:
            pass

    try:
        os.rmdir(
            os.path.join(
                tmp_dir,
                "chunks",
            )
        )

    except OSError:
        pass

    return all_segments


if __name__ == "__main__":

    import sys
    import json

    if len(sys.argv) < 2:

        print(
            "Usage: python transcribe.py <video_path>"
        )

        sys.exit(1)

    os.makedirs(
        "tmp",
        exist_ok=True,
    )

    audio = extract_audio(
        sys.argv[1],
        "tmp",
    )

    segs = transcribe(
        audio,
        tmp_dir="tmp",
    )

    with open(
        "tmp/transcript.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            segs,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nTranscribed {len(segs)} segments "
        f"-> tmp/transcript.json"
    )