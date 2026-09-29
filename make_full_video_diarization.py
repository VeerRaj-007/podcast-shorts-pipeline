import os
import json
from pathlib import Path

import whisperx
from whisperx.diarize import DiarizationPipeline


VIDEO = Path(
    "Leon Kennedy meets Grace Ashcroft  Nick Apostolides and Angela Sant'Albano Interview Each Other - FGS (1080p, h264).mp4"
)

OUTPUT = Path(
    "tmp/full_video_diarization_v1.json"
)

DEVICE = "cuda"
HF_TOKEN = os.environ.get("HF_TOKEN")


if not VIDEO.exists():
    raise FileNotFoundError(
        f"Video not found: {VIDEO}"
    )

if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN environment variable is not set."
    )


print("=" * 72)
print("FULL-VIDEO WHISPERX DIARIZATION")
print("=" * 72)
print()
print("Input:", VIDEO)
print("Output:", OUTPUT)
print("Device:", DEVICE)
print("Speaker constraint: exactly 2")
print()


print("Loading audio...")
audio = whisperx.load_audio(
    str(VIDEO)
)

print(
    "Audio loaded:",
    len(audio),
    "samples",
)


print()
print("Loading diarization model...")

diarize_model = DiarizationPipeline(
    token=HF_TOKEN,
    device=DEVICE,
)


print()
print("Running diarization...")
print("This is the full 1001-second source, not a Short crop.")
print()

diarize_segments = diarize_model(
    audio,
    min_speakers=2,
    max_speakers=2,
)


print()
print("Diarization complete.")

# WhisperX diarization output may be a DataFrame-like object.
if hasattr(diarize_segments, "to_dict"):
    records = diarize_segments.to_dict(
        orient="records"
    )
else:
    records = diarize_segments


normalized = []

for item in records:
    start = float(item["start"])
    end = float(item["end"])
    speaker = str(
        item.get("speaker", "")
    )

    if end <= start:
        continue

    normalized.append(
        {
            "start": start,
            "end": end,
            "duration": end - start,
            "speaker": speaker,
        }
    )


normalized.sort(
    key=lambda x: (
        x["start"],
        x["end"],
    )
)


speakers = sorted(
    {
        x["speaker"]
        for x in normalized
        if x["speaker"]
    }
)


timeline_start = (
    min(x["start"] for x in normalized)
    if normalized
    else None
)

timeline_end = (
    max(x["end"] for x in normalized)
    if normalized
    else None
)


output = {
    "version": "1.0",
    "module": "full_video_whisperx_diarization",
    "source_video": str(VIDEO),
    "speaker_count_requested": 2,
    "speaker_labels_found": speakers,
    "timeline": {
        "start": timeline_start,
        "end": timeline_end,
    },
    "segments": normalized,
    "statistics": {
        "segment_count": len(normalized),
        "speaker_count_found": len(speakers),
    },
}


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT.write_text(
    json.dumps(
        output,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


print()
print("=" * 72)
print("RESULT")
print("=" * 72)
print()
print("Segments:", len(normalized))
print("Speakers:", speakers)
print(
    "Timeline:",
    timeline_start,
    "->",
    timeline_end,
)
print()
print("Output:", OUTPUT)
