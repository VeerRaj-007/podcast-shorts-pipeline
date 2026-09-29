"""
Orchestrator:

    YouTube URL
        ↓
    Download
        ↓
    Extract audio
        ↓
    Whisper transcription
        ↓
    LLM clip selection
        ↓
    FFmpeg clip generation

Cached intermediate files are reused so failed later stages
don't force us to repeat expensive earlier stages.
"""

import argparse
import json
import os

from download import download_video
from transcribe import extract_audio, transcribe
from pick import select_clips
from cut import cut_clip


def run(
    url: str,
    n_clips: int,
    tmp_dir: str = "tmp",
    out_dir: str = "output",
):
    os.makedirs(
        tmp_dir,
        exist_ok=True,
    )

    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    video_path = os.path.join(
        tmp_dir,
        "source.mp4",
    )

    audio_path = os.path.join(
        tmp_dir,
        "audio.mp3",
    )

    transcript_path = os.path.join(
        tmp_dir,
        "transcript.json",
    )

    clips_path = os.path.join(
        tmp_dir,
        "clips.json",
    )

    # =========================================================
    # STEP 1 — Download
    # =========================================================

    print("[1/4] Downloading video...")

    if os.path.exists(video_path):

        print(
            f"    Using existing video: "
            f"{video_path}"
        )

    else:

        video_path = download_video(
            url,
            tmp_dir,
        )

        print(
            f"    Downloaded: {video_path}"
        )

    # =========================================================
    # STEP 2 — Extract + transcribe
    # =========================================================

    print(
        "[2/4] Extracting audio + "
        "transcribing (Whisper)..."
    )

    if os.path.exists(transcript_path):

        print(
            f"    Using existing transcript: "
            f"{transcript_path}"
        )

        with open(
            transcript_path,
            "r",
            encoding="utf-8",
        ) as f:
            segments = json.load(f)

        print(
            f"    Loaded {len(segments)} segments"
        )

    else:

        # Extract audio only if it doesn't already exist.
        if os.path.exists(audio_path):

            print(
                f"    Using existing audio: "
                f"{audio_path}"
            )

        else:

            audio_path = extract_audio(
                video_path,
                tmp_dir,
            )

        segments = transcribe(
            audio_path,
            tmp_dir=tmp_dir,
        )

        with open(
            transcript_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                segments,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f"    Saved {len(segments)} segments "
            f"-> {transcript_path}"
        )

    # =========================================================
    # STEP 3 — Select clips
    # =========================================================

    print(
        "[3/4] Selecting best moments (LLM)..."
    )

    if os.path.exists(clips_path):

        print(
            f"    Using existing clips: "
            f"{clips_path}"
        )

        with open(
            clips_path,
            "r",
            encoding="utf-8",
        ) as f:
            clips = json.load(f)

        # If existing clips.json has fewer clips than requested,
        # regenerate it.
        if len(clips) < n_clips:

            print(
                f"    Existing file contains only "
                f"{len(clips)} clips; regenerating..."
            )

            clips = select_clips(
                segments,
                n=n_clips,
            )

            with open(
                clips_path,
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    clips,
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

    else:

        clips = select_clips(
            segments,
            n=n_clips,
        )

        with open(
            clips_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                clips,
                f,
                indent=2,
                ensure_ascii=False,
            )

    print(
        f"    Selected {len(clips)} clips"
    )

    for i, clip in enumerate(
        clips,
        start=1,
    ):
        print(
            f"    {i}. "
            f"{clip['start']:.1f}-"
            f"{clip['end']:.1f} "
            f'"{clip["title"]}"'
        )

    # =========================================================
    # STEP 4 — Cut + caption
    # =========================================================

    print(
        "[4/4] Cutting + captioning clips..."
    )

    for i, clip in enumerate(
        clips,
        start=1,
    ):

        out_path = os.path.join(
            out_dir,
            f"clip_{i}.mp4",
        )

        # Don't regenerate an existing clip.
        if os.path.exists(out_path):

            print(
                f"    -> {out_path} already exists"
            )

            continue

        cut_clip(
            video_path,
            clip,
            out_path,
            tmp_dir,
            layout="auto",
        )

        print(
            f"    -> {out_path} "
            f'"{clip["title"]}" '
            f"({clip['reason']})"
        )

    print(
        "\nDone. Check the output/ folder."
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Turn a YouTube podcast into "
            "short-form clips."
        )
    )

    parser.add_argument(
        "url",
        help="YouTube URL of the podcast/video",
    )

    parser.add_argument(
        "--clips",
        type=int,
        default=3,
        help="Number of clips to generate",
    )

    args = parser.parse_args()

    if not os.environ.get("GROQ_API_KEY"):

        raise SystemExit(
            "Set GROQ_API_KEY environment variable first."
        )

    run(
        args.url,
        args.clips,
    )