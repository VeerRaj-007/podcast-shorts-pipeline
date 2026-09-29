
#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
import subprocess
import tempfile
import os


VIDEO = Path(
    "Leon Kennedy meets Grace Ashcroft  Nick Apostolides and Angela Sant'Albano Interview Each Other - FGS (1080p, h264).mp4"
)

EXPECTED_VIDEO_END = 1001.371


def run(cmd):
    print("\n$", " ".join(map(str, cmd)))

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    print(result.stdout)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}"
        )


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate full-video speaker attribution for the "
            "current interview."
        )
    )

    parser.add_argument(
        "--video",
        default=str(VIDEO),
    )

    parser.add_argument(
        "--video-end",
        type=float,
        default=EXPECTED_VIDEO_END,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    video = Path(args.video)

    if not video.exists():
        raise FileNotFoundError(
            f"Video not found: {video}"
        )

    # ------------------------------------------------------------
    # 1. Extract audio from THIS video.
    # ------------------------------------------------------------

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="speaker_attribution_"
        )
    )

    audio_path = temp_dir / "audio.wav"

    print("=" * 72)
    print("FULL-VIDEO SPEAKER ATTRIBUTION")
    print("=" * 72)
    print()
    print("Video:", video)
    print("Expected video end:", args.video_end)
    print("Temporary audio:", audio_path)

    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(audio_path),
        ]
    )

    # ------------------------------------------------------------
    # 2. WhisperX diarization.
    #
    # Requires:
    #   pip install whisperx
    #
    # And a Hugging Face token for the pyannote diarization model.
    # ------------------------------------------------------------

    try:
        import whisperx
    except ImportError as exc:
        raise RuntimeError(
            "whisperx is not installed.\n\n"
            "Install it with:\n"
            "pip install whisperx"
        ) from exc

    token = os.environ.get(
        "HF_TOKEN"
    )

    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set.\n\n"
            "Set your Hugging Face token first:\n"
            'export HF_TOKEN="YOUR_HUGGINGFACE_TOKEN"'
        )

    device = "cuda"

    print()
    print("Loading WhisperX diarization model...")

    diarize_model = whisperx.DiarizationPipeline(
        use_auth_token=token,
        device=device,
    )

    print()
    print("Running diarization on CURRENT VIDEO AUDIO...")

    diarize_segments = diarize_model(
        str(audio_path)
    )

    if not hasattr(
        diarize_segments,
        "iterrows",
    ):
        raise RuntimeError(
            "Unexpected WhisperX diarization output."
        )

    turns = []

    for _, row in diarize_segments.iterrows():
        start = float(
            row["start"]
        )

        end = float(
            row["end"]
        )

        speaker = str(
            row["speaker"]
        )

        if end <= start:
            continue

        start = max(
            0.0,
            start,
        )

        end = min(
            args.video_end,
            end,
        )

        if end <= start:
            continue

        turns.append(
            {
                "start": start,
                "end": end,
                "duration": end - start,
                "speaker_label": speaker,
            }
        )

    if not turns:
        raise RuntimeError(
            "Diarization returned zero usable speaker turns."
        )

    # ------------------------------------------------------------
    # 3. Validate the diarization timeline.
    # ------------------------------------------------------------

    turns.sort(
        key=lambda x: (
            x["start"],
            x["end"],
        )
    )

    min_start = min(
        x["start"]
        for x in turns
    )

    max_end = max(
        x["end"]
        for x in turns
    )

    speakers = sorted(
        {
            x["speaker_label"]
            for x in turns
        }
    )

    print()
    print("=" * 72)
    print("DIARIZATION RESULT")
    print("=" * 72)
    print()
    print("Turn count:", len(turns))
    print("Speakers:", speakers)
    print(
        "Timeline:",
        f"{min_start:.3f}",
        "->",
        f"{max_end:.3f}",
    )

    # Hard protection against accidentally using another source.
    if max_end > args.video_end + 2.0:
        raise RuntimeError(
            "Diarization timeline is longer than the current video. "
            "Refusing to create attribution file."
        )

    if min_start < -0.001:
        raise RuntimeError(
            "Diarization produced an invalid negative start time."
        )

    # ------------------------------------------------------------
    # 4. DO NOT GUESS SPEAKER -> PERSON mapping.
    #
    # WhisperX gives SPEAKER_x labels.
    #
    # Camera Director needs:
    #     PERSON_0 / PERSON_1
    #
    # That mapping must be verified separately using visual evidence.
    # ------------------------------------------------------------

    output = {
        "version": "1.0",
        "module": "full_video_speaker_attribution",
        "method": "whisperx_diarization_current_video",

        "source_video": str(
            video
        ),

        "source_range": {
            "start": 0.0,
            "end": args.video_end,
        },

        "speaker_mapping": {
            speaker: {
                "person_id": None,
                "confidence": 0.0,
                "verified": False,
            }
            for speaker in speakers
        },

        "turns": turns,

        "statistics": {
            "turn_count": len(turns),
            "speaker_count": len(speakers),
            "speakers": speakers,
            "timeline_start": min_start,
            "timeline_end": max_end,
            "video_end": args.video_end,
        },

        "warning": (
            "Speaker labels were diarized from the CURRENT VIDEO "
            "AUDIO. SPEAKER_x -> PERSON_x visual identity mapping "
            "is intentionally unverified and MUST NOT be assumed."
        ),
    }

    # ------------------------------------------------------------
    # 5. Save output.
    # ------------------------------------------------------------

    output_path = Path(
        args.output
    )

    save_json(
        output_path,
        output,
    )

    print()
    print("=" * 72)
    print("FULL SPEAKER DIARIZATION FILE CREATED")
    print("=" * 72)
    print()
    print("Output:", output_path)
    print("Turns:", len(turns))
    print("Speakers:", len(speakers))
    print(
        "Timeline:",
        f"{min_start:.3f} -> {max_end:.3f}",
    )
    print()
    print(
        "IMPORTANT:"
    )
    print(
        "SPEAKER_x -> PERSON_x mapping is NOT verified."
    )
    print(
        "Do not use this mapping directly for camera framing."
    )


if __name__ == "__main__":
    main()
