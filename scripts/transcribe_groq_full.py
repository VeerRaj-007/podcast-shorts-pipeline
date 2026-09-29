from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import requests


VIDEO = Path(
    r"Leon Kennedy meets Grace Ashcroft  Nick Apostolides and Angela Sant'Albano Interview Each Other - FGS (1080p, h264).mp4"
)

OUTPUT = Path("tmp/new_aligned_transcript.json")

MODEL = "whisper-large-v3"
API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

# Keep chunks comfortably below Groq's file-size limit.
CHUNK_SECONDS = 300


def run(cmd: list[str]) -> None:
    print("$", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def get_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def transcribe_file(
    audio_path: Path,
    prompt: str | None = None,
) -> dict[str, Any]:

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set.\n"
            "Windows Git Bash:\n"
            "export GROQ_API_KEY='your_key_here'"
        )

    data = {
        "model": MODEL,
        "response_format": "verbose_json",
        "timestamp_granularities[]": ["segment", "word"],
        "language": "en",
        "temperature": "0",
    }

    if prompt:
        data["prompt"] = prompt

    with audio_path.open("rb") as f:
        response = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
            },
            files={
                "file": (
                    audio_path.name,
                    f,
                    "audio/m4a",
                ),
            },
            data=data,
            timeout=600,
        )

    if not response.ok:
        raise RuntimeError(
            f"Groq error {response.status_code}: "
            f"{response.text}"
        )

    return response.json()


def main() -> None:
    if not VIDEO.exists():
        raise FileNotFoundError(
            f"Video not found:\n{VIDEO.resolve()}"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    duration = get_duration(VIDEO)

    print(f"Video duration: {duration:.3f}s")
    print(f"Chunks: {((duration - 1) // CHUNK_SECONDS) + 1:.0f}")

    all_segments: list[dict[str, Any]] = []
    all_words: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        chunk_start = 0.0
        chunk_index = 0

        while chunk_start < duration:
            chunk_end = min(
                duration,
                chunk_start + CHUNK_SECONDS,
            )

            chunk_file = (
                tmp_path
                / f"chunk_{chunk_index:03d}.m4a"
            )

            run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{chunk_start:.3f}",
                    "-i",
                    str(VIDEO),
                    "-t",
                    f"{chunk_end - chunk_start:.3f}",
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "64k",
                    str(chunk_file),
                ]
            )

            print(
                f"\nTranscribing chunk "
                f"{chunk_index}: "
                f"{chunk_start:.2f}-{chunk_end:.2f}"
            )

            result = transcribe_file(
                chunk_file,
                prompt=(
                    "This is an English interview. "
                    "Preserve names, game titles, people names, "
                    "and conversational wording."
                ),
            )

            for segment in result.get("segments", []):
                item = dict(segment)

                item["start"] = (
                    float(item["start"]) + chunk_start
                )
                item["end"] = (
                    float(item["end"]) + chunk_start
                )

                all_segments.append(item)

            for word in result.get("words", []):
                item = dict(word)

                item["start"] = (
                    float(item["start"]) + chunk_start
                )
                item["end"] = (
                    float(item["end"]) + chunk_start
                )

                all_words.append(item)

            chunk_index += 1
            chunk_start = chunk_end

    result = {
        "source": str(VIDEO),
        "duration": duration,
        "model": MODEL,
        "segments": all_segments,
        "words": all_words,
    }

    OUTPUT.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("\nDone.")
    print(f"Transcript: {OUTPUT}")


if __name__ == "__main__":
    main()