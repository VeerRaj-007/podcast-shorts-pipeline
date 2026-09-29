#!/usr/bin/env python3
"""
FINAL renderer for the independent interview Shorts pipeline.

Properties:
  - one source video input
  - static crop per camera-plan segment
  - NO crop animation
  - NO xfade
  - NO acrossfade
  - video is concatenated from exact trimmed source-shot pieces
  - when source pieces are contiguous, AUDIO IS TRIMMED ONCE as one stream
    instead of being cut/re-encoded independently at every camera shot

Usage:
  python face_aware_camera_renderer_final.py \
      --video "SOURCE.mp4" \
      --camera tmp/new_inde_short01_camera_final.json \
      --output out/short01_final.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


FPS = 30
AUDIO_GAP_TOLERANCE = 1.0 / FPS + 0.01


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    p = subprocess.run(cmd, text=True)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def ff(v: float) -> str:
    return f"{v:.6f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--crf", type=int, default=18)
    args = ap.parse_args()

    plan = load_json(args.camera)
    segments = plan.get("segments", [])
    if not isinstance(segments, list) or not segments:
        raise RuntimeError("Camera plan contains no segments.")

    for i, s in enumerate(segments):
        for key in ("source_start", "source_end"):
            if key not in s:
                raise RuntimeError(f"Segment {i} missing {key}")
        crop = s.get("crop", {})
        for key in ("x", "y", "width", "height", "output_width", "output_height"):
            if key not in crop:
                raise RuntimeError(f"Segment {i} crop missing {key}")

    # ---- Video graph ----
    filters: list[str] = []
    vlabels: list[str] = []

    for i, s in enumerate(segments):
        sin_ = float(s["source_start"])
        sout = float(s["source_end"])
        c = s["crop"]

        x = int(c["x"])
        y = int(c["y"])
        w = int(c["width"])
        h = int(c["height"])
        ow = int(c["output_width"])
        oh = int(c["output_height"])

        label = f"v{i}"
        filters.append(
            f"[0:v]trim=start={ff(sin_)}:end={ff(sout)},"
            f"setpts=PTS-STARTPTS,"
            f"crop={w}:{h}:{x}:{y},"
            f"scale={ow}:{oh}:flags=lanczos,"
            f"fps={FPS},"
            f"format=yuv420p[{label}]"
        )
        vlabels.append(f"[{label}]")

    filters.append(
        "".join(vlabels)
        + f"concat=n={len(vlabels)}:v=1:a=0[vout]"
    )

    # ---- Audio ----
    contiguous = True
    for a, b in zip(segments, segments[1:]):
        gap = float(b["source_start"]) - float(a["source_end"])
        if abs(gap) > AUDIO_GAP_TOLERANCE:
            contiguous = False
            break

    if contiguous:
        first = float(segments[0]["source_start"])
        last = float(segments[-1]["source_end"])
        filters.append(
            f"[0:a]atrim=start={ff(first)}:end={ff(last)},"
            f"asetpts=PTS-STARTPTS[aout]"
        )
    else:
        # Non-contiguous fallback: one exact audio trim per segment.
        alabels: list[str] = []
        for i, s in enumerate(segments):
            sin_ = float(s["source_start"])
            sout = float(s["source_end"])
            label = f"a{i}"
            filters.append(
                f"[0:a]atrim=start={ff(sin_)}:end={ff(sout)},"
                f"asetpts=PTS-STARTPTS[{label}]"
            )
            alabels.append(f"[{label}]")
        filters.append(
            "".join(alabels)
            + f"concat=n={len(alabels)}:v=0:a=1[aout]"
        )

    filter_complex = ";".join(filters)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        args.video,
        "-filter_complex",
        filter_complex,
        "-map",
        "[vout]",
        "-map",
        "[aout]",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(args.crf),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        "-shortest",
        str(out),
    ]

    print("=== FINAL CAMERA RENDERER ===")
    print(f"camera segments : {len(segments)}")
    print(f"audio strategy  : {'single contiguous trim' if contiguous else 'per-segment concat'}")
    print(f"output          : {out}")

    run(cmd)
    print("\nDONE:", out)


if __name__ == "__main__":
    main()