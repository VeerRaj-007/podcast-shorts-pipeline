from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def resolve_whisperx_python(project_root: Path) -> str:
    candidates = [
        project_root / ".venv-whisperx" / "Scripts" / "python.exe",
        project_root / ".venv-whisperx" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def normalize_speaker(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def build_speaker_tracks(raw_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge nearby contiguous diarized segments from the same speaker."""
    rows: List[Tuple[float, float, str]] = []
    for item in raw_segments:
        speaker = normalize_speaker(item.get("speaker"))
        if not speaker:
            continue
        try:
            start = float(item["start"])
            end = float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        rows.append((start, end, speaker))

    rows.sort(key=lambda x: (x[0], x[1]))
    merged: List[Dict[str, Any]] = []

    for start, end, speaker in rows:
        if merged and merged[-1]["speaker"] == speaker and start <= merged[-1]["end"] + 0.35:
            merged[-1]["end"] = max(merged[-1]["end"], end)
            merged[-1]["duration"] = merged[-1]["end"] - merged[-1]["start"]
            continue
        merged.append({
            "speaker": speaker,
            "start": start,
            "end": end,
            "duration": end - start,
        })
    return merged


def speaker_stats(tracks: List[Dict[str, Any]]) -> Dict[str, Any]:
    totals: Dict[str, float] = {}
    for track in tracks:
        speaker = track["speaker"]
        totals[speaker] = totals.get(speaker, 0.0) + float(track["duration"])
    ordered = sorted(totals.items(), key=lambda item: item[0])
    return {
        "speaker_count": len(ordered),
        "speakers": [
            {"speaker": speaker, "speech_seconds": round(seconds, 3)}
            for speaker, seconds in ordered
        ],
    }


def diarize_with_whisperx(
    *,
    audio_path: Path,
    output_dir: Path,
    project_root: Path,
    hf_token: str,
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> List[Dict[str, Any]]:
    """Run WhisperX diarization in the dedicated WhisperX environment.

    This uses the current WhisperX API (whisperx.diarize.DiarizationPipeline)
    while remaining compatible with older constructor names where possible.
    """
    runner = output_dir / "_run_diarization.py"
    result_json = output_dir / "_diarization_raw.json"

    script = r'''
import inspect
import json
import sys
from pathlib import Path

from whisperx.diarize import DiarizationPipeline
import whisperx
import torch


audio_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
token = sys.argv[3]
min_speakers = int(sys.argv[4]) if sys.argv[4] != "None" else None
max_speakers = int(sys.argv[5]) if sys.argv[5] != "None" else None

try:
    import torch
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
except Exception:
    device = "cpu"

params = inspect.signature(DiarizationPipeline).parameters
kwargs = {"device": device}
if "token" in params:
    kwargs["token"] = token
elif "use_auth_token" in params:
    kwargs["use_auth_token"] = token
else:
    raise RuntimeError(
        "Unsupported WhisperX DiarizationPipeline constructor. "
        f"Available parameters: {list(params)}"
    )

pipeline = DiarizationPipeline(**kwargs)
call_params = inspect.signature(pipeline.__call__).parameters
call_kwargs = {}
if "min_speakers" in call_params and min_speakers is not None:
    call_kwargs["min_speakers"] = min_speakers
if "max_speakers" in call_params and max_speakers is not None:
    call_kwargs["max_speakers"] = max_speakers

audio = whisperx.load_audio(str(audio_path))
# The installed WhisperX DiarizationPipeline expects the loaded NumPy
# waveform directly and internally converts it to the pyannote input dict.
result = pipeline(audio, **call_kwargs)

# Current WhisperX returns a pandas DataFrame. Older variants returned
# a pyannote Annotation. Support both.
rows = []
if isinstance(result, tuple):
    result = result[0]

if hasattr(result, "iterrows"):
    for _, row in result.iterrows():
        try:
            start = float(row["start"])
            end = float(row["end"])
            speaker = str(row["speaker"])
        except Exception:
            continue
        rows.append({"start": start, "end": end, "speaker": speaker})
elif hasattr(result, "itertracks"):
    for item in result.itertracks(yield_label=True):
        turn, _, speaker = item
        rows.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": str(speaker),
        })
else:
    raise RuntimeError(f"Unsupported diarization output type: {type(result)!r}")

with output_path.open("w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2)
'''

    runner.write_text(script, encoding="utf-8")
    python_bin = resolve_whisperx_python(project_root)

    command = [
        python_bin,
        str(runner),
        str(audio_path),
        str(result_json),
        hf_token,
        str(min_speakers),
        str(max_speakers),
    ]

    print(f"WhisperX Python: {python_bin}")
    print(f"Audio: {audio_path}")
    print("Loading current WhisperX diarization API...")

    proc = subprocess.run(
        command,
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "WhisperX diarization failed.\n"
            f"stdout:\n{proc.stdout}\n\n"
            f"stderr:\n{proc.stderr}"
        )

    if not result_json.exists():
        raise RuntimeError("WhisperX diarization finished without producing a result file.")

    return load_json(result_json)


def build_or_load(
    *,
    audio_path: Path,
    output_path: Path,
    project_root: Path,
    refresh: bool = False,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> Dict[str, Any]:
    if output_path.exists() and not refresh:
        return load_json(output_path)

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError(
            "Speaker diarization requires HF_TOKEN. Set it in your shell, then rerun."
        )

    raw = diarize_with_whisperx(
        audio_path=audio_path,
        output_dir=output_path.parent,
        project_root=project_root,
        hf_token=token,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    tracks = build_speaker_tracks(raw)
    result = {
        "version": 2,
        "mode": "whisperx_diarization",
        "audio": str(audio_path),
        "speaker_count": speaker_stats(tracks)["speaker_count"],
        "tracks": tracks,
        "stats": speaker_stats(tracks),
    }
    save_json(output_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build speaker tracks with WhisperX diarization.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--audio", default="tmp/audio.mp3")
    parser.add_argument("--output", default="tmp/speaker_tracks.json")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    audio_path = (project_root / args.audio).resolve()
    output_path = (project_root / args.output).resolve()

    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")

    result = build_or_load(
        audio_path=audio_path,
        output_path=output_path,
        project_root=project_root,
        refresh=args.refresh,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )

    print(f"Speaker tracks: {len(result['tracks'])}")
    print(f"Speakers detected: {result['stats']['speaker_count']}")
    for speaker in result["stats"]["speakers"]:
        print(f"  {speaker['speaker']}: {speaker['speech_seconds']:.1f}s")
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
