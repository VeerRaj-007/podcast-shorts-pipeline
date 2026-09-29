#!/usr/bin/env python3
"""
Speaker Attribution v1
- Runs pyannote Community-1 speaker diarization.
- Restricts the result to two speakers.
- Uses the known visual anchor around 956-958s to map pyannote labels
  to PERSON_0 / PERSON_1.
- Emits a clean JSON timeline for the selected editorial moment.

Requirements:
    pip install pyannote.audio

Before first run:
    1) Accept the model conditions for:
       https://huggingface.co/pyannote/speaker-diarization-community-1
    2) Create a Hugging Face access token.
    3) Set HUGGINGFACE_TOKEN in your shell.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any


DEFAULT_START = 951.88
DEFAULT_END = 961.66
# We visually verified that PERSON_0 is speaking around this part of the
# selected shot, so this acts as the identity calibration anchor.
DEFAULT_ANCHOR_START = 956.0
DEFAULT_ANCHOR_END = 958.0


def run_ffmpeg_extract_audio(video: Path, wav_path: Path, start: float, end: float) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-i", str(video),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(wav_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def load_wav_as_waveform(wav_path: Path) -> tuple[Any, int]:
    """Load PCM16 mono WAV without torchcodec/torchaudio."""
    import torch

    with wave.open(str(wav_path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frame_count = wf.getnframes()
        raw = wf.readframes(frame_count)

    if sample_width != 2:
        raise RuntimeError(
            f"Expected 16-bit PCM WAV, got sample width={sample_width} bytes."
        )

    audio = torch.frombuffer(bytearray(raw), dtype=torch.int16).float() / 32768.0

    if channels > 1:
        if audio.numel() % channels != 0:
            raise RuntimeError("Invalid multichannel WAV sample count.")
        audio = audio.reshape(-1, channels).mean(dim=1)

    waveform = audio.unsqueeze(0).contiguous()
    return waveform, sample_rate


def load_visual_identities(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    obs = data.get("identity_clean_observations", [])
    selected = []

    for item in obs:
        t = item.get("timestamp")
        ident = item.get("identity_id")
        if t is None or ident not in {"PERSON_0", "PERSON_1"}:
            continue
        selected.append(item)

    return selected


def identity_at_time(
    observations: list[dict[str, Any]],
    start: float,
    end: float,
) -> str | None:
    scores = {"PERSON_0": 0.0, "PERSON_1": 0.0}
    for obs in observations:
        t = float(obs["timestamp"])
        if start <= t <= end:
            ident = obs["identity_id"]
            # Each half-second visual observation contributes equally.
            scores[ident] += 1.0

    if scores["PERSON_0"] == 0 and scores["PERSON_1"] == 0:
        return None
    return max(scores, key=scores.get)


def diarization_to_turns(diarization: Any) -> list[dict[str, Any]]:
    turns = []
    for turn, speaker in diarization:
        turns.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker_label": str(speaker),
            }
        )
    return turns


def map_pyannote_to_people(
    turns: list[dict[str, Any]],
    anchor_start: float,
    anchor_end: float,
) -> dict[str, str]:
    # Whichever pyannote speaker overlaps the verified PERSON_0 speaking
    # anchor the most becomes PERSON_0.
    overlap_by_speaker: dict[str, float] = {}

    for turn in turns:
        ov = overlap(turn["start"], turn["end"], anchor_start, anchor_end)
        if ov > 0:
            speaker = turn["speaker_label"]
            overlap_by_speaker[speaker] = overlap_by_speaker.get(speaker, 0.0) + ov

    if not overlap_by_speaker:
        raise RuntimeError(
            "No diarization turn overlaps the calibration anchor "
            f"{anchor_start:.2f}-{anchor_end:.2f}s."
        )

    person0_speaker = max(overlap_by_speaker, key=overlap_by_speaker.get)
    speakers = sorted({t["speaker_label"] for t in turns})

    if len(speakers) == 1:
        return {person0_speaker: "PERSON_0"}

    if len(speakers) > 2:
        remaining = [s for s in speakers if s != person0_speaker]
        return {
            person0_speaker: "PERSON_0",
            remaining[0]: "PERSON_1",
        }

    person1_candidates = [s for s in speakers if s != person0_speaker]
    person1_speaker = person1_candidates[0]

    return {
        person0_speaker: "PERSON_0",
        person1_speaker: "PERSON_1",
    }


def merge_adjacent(turns: list[dict[str, Any]], gap: float = 0.08) -> list[dict[str, Any]]:
    if not turns:
        return []

    out = [dict(turns[0])]

    for cur in turns[1:]:
        prev = out[-1]
        if (
            cur["person_id"] == prev["person_id"]
            and cur["start"] - prev["end"] <= gap
        ):
            prev["end"] = max(prev["end"], cur["end"])
            prev["duration"] = prev["end"] - prev["start"]
        else:
            out.append(dict(cur))

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--visual", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", type=float, default=DEFAULT_START)
    parser.add_argument("--end", type=float, default=DEFAULT_END)
    parser.add_argument("--anchor-start", type=float, default=DEFAULT_ANCHOR_START)
    parser.add_argument("--anchor-end", type=float, default=DEFAULT_ANCHOR_END)
    args = parser.parse_args()

    video = Path(args.video)
    visual = Path(args.visual)
    output = Path(args.output)

    if not video.exists():
        raise FileNotFoundError(video)
    if not visual.exists():
        raise FileNotFoundError(visual)

    token = os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError(
            "HUGGINGFACE_TOKEN is not set. In Git Bash run:\n"
            'export HUGGINGFACE_TOKEN="hf_your_token_here"'
        )

    try:
        from pyannote.audio import Pipeline
    except Exception as exc:
        raise RuntimeError(
            "pyannote.audio is not available in this Python environment.\n"
            "Install with: pip install pyannote.audio"
        ) from exc

    print("=== SPEAKER ATTRIBUTION v1 ===")
    print(f"selected range: {args.start:.3f} -> {args.end:.3f}")
    print("loading pyannote/speaker-diarization-community-1...")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=token,
    )

    # Community-1 pipelines run on CPU by default. Use the RTX GPU when
    # available in this environment.
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            pipeline.to(torch.device("cuda"))
        else:
            print("GPU: unavailable; using CPU")
    except Exception as exc:
        print(f"GPU setup skipped: {exc}")

    # Extract a small PCM WAV and load it ourselves. This avoids pyannote's
    # TorchCodec file decoder, which is broken in some Windows/PyTorch setups.
    with tempfile.TemporaryDirectory(prefix="speaker_attr_v1_") as td:
        wav_path = Path(td) / "selected.wav"
        run_ffmpeg_extract_audio(video, wav_path, args.start, args.end)
        waveform, sample_rate = load_wav_as_waveform(wav_path)

        print(f"audio waveform: {tuple(waveform.shape)} @ {sample_rate} Hz")

        result = pipeline(
            {
                "waveform": waveform,
                "sample_rate": sample_rate,
            },
            num_speakers=2,
        )

        diarization = getattr(result, "exclusive_speaker_diarization", None)
        if diarization is None:
            diarization = result.speaker_diarization

        relative_turns = diarization_to_turns(diarization)

    # Convert excerpt-relative time to absolute source time.
    turns = []
    for turn in relative_turns:
        abs_start = args.start + turn["start"]
        abs_end = args.start + turn["end"]
        if abs_end <= args.start or abs_start >= args.end:
            continue
        abs_start = max(abs_start, args.start)
        abs_end = min(abs_end, args.end)
        if abs_end <= abs_start:
            continue
        turns.append(
            {
                "start": abs_start,
                "end": abs_end,
                "speaker_label": turn["speaker_label"],
            }
        )

    visual_obs = load_visual_identities(visual)

    # The calibration anchor is absolute source time.
    mapping = map_pyannote_to_people(
        turns,
        args.anchor_start,
        args.anchor_end,
    )

    attributed = []
    for turn in turns:
        attributed.append(
            {
                "start": turn["start"],
                "end": turn["end"],
                "duration": turn["end"] - turn["start"],
                "speaker_label": turn["speaker_label"],
                "person_id": mapping.get(turn["speaker_label"], "UNKNOWN"),
            }
        )

    attributed = merge_adjacent(attributed)

    # Attach lightweight visual evidence around each turn.
    for turn in attributed:
        visual_counts = {"PERSON_0": 0, "PERSON_1": 0}
        for obs in visual_obs:
            t = float(obs["timestamp"])
            if turn["start"] <= t <= turn["end"]:
                visual_counts[obs["identity_id"]] += 1
        turn["visual_identity_observations"] = visual_counts

    result_out = {
        "version": "1.0",
        "module": "speaker_attribution_v1",
        "method": "pyannote_community_1_exclusive_speaker_diarization",
        "source_video": str(video),
        "source_range": [args.start, args.end],
        "anchor": {
            "start": args.anchor_start,
            "end": args.anchor_end,
            "known_identity": "PERSON_0",
            "reason": "verified_visual_speaking_anchor",
        },
        "speaker_mapping": mapping,
        "turns": attributed,
        "statistics": {
            "turn_count": len(attributed),
            "person_0_seconds": sum(
                t["duration"] for t in attributed if t["person_id"] == "PERSON_0"
            ),
            "person_1_seconds": sum(
                t["duration"] for t in attributed if t["person_id"] == "PERSON_1"
            ),
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result_out, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(result_out["statistics"], indent=2))
    print(f"mapping: {mapping}")
    print(f"output: {output}")


if __name__ == "__main__":
    main()