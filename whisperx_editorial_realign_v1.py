from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any


def run_ffmpeg_extract_audio(
    video: Path,
    wav_path: Path,
    start: float,
    end: float,
) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]

    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def get_wav_duration(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()

    if rate <= 0:
        raise RuntimeError("Invalid WAV sample rate.")

    return frames / rate


def shift_segments(
    aligned: dict[str, Any],
    source_start: float,
    source_end: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    segments = aligned.get("segments", [])
    words_out: list[dict[str, Any]] = []
    segments_out: list[dict[str, Any]] = []

    for segment_index, segment in enumerate(segments):
        local_start = segment.get("start")
        local_end = segment.get("end")

        if local_start is None or local_end is None:
            continue

        absolute_start = source_start + float(local_start)
        absolute_end = source_start + float(local_end)

        absolute_start = max(source_start, absolute_start)
        absolute_end = min(source_end, absolute_end)

        segment_out = {
            "segment_index": segment_index,
            "start": absolute_start,
            "end": absolute_end,
            "text": segment.get("text", ""),
        }

        segments_out.append(segment_out)

        for word_index, word in enumerate(segment.get("words", [])):
            word_start = word.get("start")
            word_end = word.get("end")

            if word_start is None or word_end is None:
                continue

            abs_word_start = source_start + float(word_start)
            abs_word_end = source_start + float(word_end)

            if abs_word_end <= source_start:
                continue
            if abs_word_start >= source_end:
                continue

            abs_word_start = max(source_start, abs_word_start)
            abs_word_end = min(source_end, abs_word_end)

            if abs_word_end <= abs_word_start:
                continue

            words_out.append(
                {
                    "word_index": len(words_out),
                    "segment_index": segment_index,
                    "segment_word_index": word_index,
                    "start": abs_word_start,
                    "end": abs_word_end,
                    "word": word.get("word", ""),
                    "score": word.get("score"),
                }
            )

    return segments_out, words_out


def normalize_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split()).strip()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", required=True)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--language", default="en")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
    )

    args = parser.parse_args()

    video = Path(args.video)
    output = Path(args.output)

    if not video.exists():
        raise FileNotFoundError(video)

    if args.end <= args.start:
        raise ValueError("--end must be greater than --start.")

    editorial_text = normalize_text(args.text)

    if not editorial_text:
        raise ValueError("Editorial text is empty.")

    try:
        import torch
        import whisperx
    except Exception as exc:
        raise RuntimeError(
            "WhisperX/PyTorch is unavailable. Run this inside .venv-whisperx."
        ) from exc

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("=== WHISPERX EDITORIAL RE-ALIGNMENT v1 ===")
    print(f"source range: {args.start:.3f} -> {args.end:.3f}")
    print(f"device: {device}")
    print("editorial text:")
    print(editorial_text)

    with tempfile.TemporaryDirectory(prefix="whisperx_realign_v1_") as td:
        wav_path = Path(td) / "selected.wav"

        run_ffmpeg_extract_audio(
            video,
            wav_path,
            args.start,
            args.end,
        )

        audio_duration = get_wav_duration(wav_path)
        print(f"audio duration: {audio_duration:.3f}s")

        print("loading WhisperX alignment model...")
        model_a, metadata = whisperx.load_align_model(
            language_code=args.language,
            device=device,
        )

        audio = whisperx.load_audio(str(wav_path))

        # IMPORTANT:
        # This segment is NOT a transcription. It is the corrected editorial
        # text that we want WhisperX to force-align to the source audio.
        transcript_segments = [
            {
                "start": 0.0,
                "end": audio_duration,
                "text": editorial_text,
            }
        ]

        print("running forced alignment...")
        aligned = whisperx.align(
            transcript_segments,
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )

    segments, words = shift_segments(
        aligned,
        args.start,
        args.end,
    )

    if not words:
        raise RuntimeError(
            "WhisperX returned no aligned words. "
            "The corrected text may not match the source audio closely enough."
        )

    # Basic sanity checks against impossible/non-monotonic output.
    for previous, current in zip(words, words[1:]):
        if current["start"] < previous["start"]:
            raise RuntimeError("Word timestamps are not monotonic.")
        if current["end"] < current["start"]:
            raise RuntimeError("Invalid word timestamp interval.")

    output_data = {
        "version": "1.0",
        "module": "whisperx_editorial_realign",
        "method": "corrected_editorial_text_forced_aligned_to_source_audio",
        "source_video": str(video),
        "source_range": [
            args.start,
            args.end,
        ],
        "language": args.language,
        "timing_source": {
            "provider": "WhisperX",
            "forced_alignment": True,
            "original_asr_text_used_for_words": False,
            "editorial_text_used_as_alignment_target": True,
        },
        "editorial_text": editorial_text,
        "segments": segments,
        "word_segments": words,
        "statistics": {
            "word_count": len(words),
            "segment_count": len(segments),
            "first_word_start": words[0]["start"],
            "last_word_end": words[-1]["end"],
            "coverage_seconds": words[-1]["end"] - words[0]["start"],
            "selected_duration": args.end - args.start,
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(
            output_data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(f"aligned words: {len(words)}")
    print(
        f"coverage: "
        f"{words[0]['start']:.3f} -> {words[-1]['end']:.3f}"
    )
    print(f"output: {output}")

    print()
    print("WORD TIMELINE")
    for word in words:
        print(
            f'{word["start"]:.3f} -> '
            f'{word["end"]:.3f} | '
            f'{word["word"]}'
        )


if __name__ == "__main__":
    main()