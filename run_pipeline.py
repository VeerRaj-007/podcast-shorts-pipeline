#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
GROQ_EDITOR = ROOT / "groq_editorial_engine_v1_1.py"
LOCKED_BRIDGE = ROOT / "groq_to_locked_sequence_v1.py"
MOMENT_RESOLVER = ROOT / "resolve_groq_moments_v1.py"
CAMERA_BATCH = ROOT / "make_all_camera_plans_v1.py"
CAMERA_RENDER_BATCH = ROOT / "render_all_camera_shorts_v1.py"
SUBTITLE_TIMING_BATCH = ROOT / "build_all_subtitle_timings_v1.py"
SUBTITLE_RENDER_BATCH = ROOT / "render_all_final_shorts_v1.py"
YUNET = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
SFACE = ROOT / "models" / "face_recognition_sface_2021dec.onnx"
VISUAL_IDENTITY = ROOT / "scripts" / "visual_identity_reassign_v7.py"
VISUAL_CANDIDATES = [
    ROOT / "scripts" / "visual_intelligence_yunet_sface_v6.py",
    ROOT / "scripts" / "visual_intelligence_yunet_sface_v5.py",
]
SPEAKER = ROOT / "speaker_attribution_v1.py"
DEFAULT_STYLE = "podcast_host"
STYLES = {
    "bold_tiktok", "cinematic", "clean_white", "gaming", "high_energy",
    "karaoke", "minimal_upper", "news", "podcast_host", "typewriter",
}
PRESETS = {
    "podcast": "podcast_host",
    "tiktok": "bold_tiktok",
    "cinematic": "cinematic",
    "clean": "clean_white",
}


def log(msg: str = "") -> None:
    print(msg, flush=True)


def die(msg: str) -> None:
    raise SystemExit(msg)


def require(path: Path, label: str) -> None:
    if not path.exists():
        die(f"{label} not found: {path}")


def run(cmd: list[str], capture: bool = False) -> subprocess.CompletedProcess[str]:
    log("$ " + " ".join(f'"{x}"' if " " in x else x for x in cmd))
    r = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=capture)
    if r.returncode != 0:
        if capture:
            if r.stdout: log(r.stdout)
            if r.stderr: log(r.stderr)
        die(f"Command failed with exit code {r.returncode}.")
    return r


def capture(cmd: list[str]) -> str:
    r = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr or "command failed")
    return r.stdout


def is_url(value: str) -> bool:
    return urlparse(value).scheme in {"http", "https"}


def slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value.strip("._-")[:80] or "video"


def short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def probe_duration(path: Path) -> float:
    return float(capture([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]).strip())


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def checkpoint(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def find_visual_script() -> Path:
    for p in VISUAL_CANDIDATES:
        if p.exists():
            return p
    die("No visual intelligence script found under scripts/.")


def whisper_python() -> Path:
    for p in [ROOT / ".venv-whisperx" / "Scripts" / "python.exe", ROOT / ".venv-whisperx" / "bin" / "python"]:
        if p.exists():
            return p
    return Path(sys.executable)

def vision_python() -> Path:
    p = ROOT / ".venv-vision" / "Scripts" / "python.exe"
    if not p.exists():
        die(f"Vision Python not found: {p}")
    return p

def detect_shots(video: Path, output: Path, threshold: float) -> None:
    duration = probe_duration(video)
    r = subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "info", "-i", str(video),
        "-filter:v", f"select='gt(scene\\,{threshold})',showinfo",
        "-an", "-f", "null", "-",
    ], cwd=ROOT, text=True, capture_output=True)
    text = r.stderr
    if r.returncode != 0 and "Output file is empty" not in text:
        raise RuntimeError(text)
    times = [0.0]
    for m in re.finditer(r"pts_time:([0-9]+(?:\.[0-9]+)?)", text):
        t = float(m.group(1))
        if 0.2 < t < duration - 0.2 and t - times[-1] >= 0.20:
            times.append(t)
    times.append(duration)
    shots = []
    for i, (a, b) in enumerate(zip(times, times[1:])):
        if b - a > 0.05:
            shots.append({"shot_id": f"SHOT_{i:03d}", "start": a, "end": b, "duration": b-a})
    write_json(output, {
        "version": "1.0", "module": "source_shot_detector",
        "source_video": str(video), "video_duration": duration,
        "scene_threshold": threshold, "shot_count": len(shots), "shots": shots,
    })


def write_whisper_helper(path: Path) -> None:
    path.write_text(r'''
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path
import torch
import whisperx

video = Path(sys.argv[1])
audio = Path(sys.argv[2])
out = Path(sys.argv[3])
model_name = os.environ.get("WHISPER_MODEL", "large-v3")
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
audio_data = whisperx.load_audio(str(audio))
model = whisperx.load_model(model_name, device=device, compute_type=compute_type, language="en")
result = model.transcribe(audio_data, batch_size=int(os.environ.get("WHISPER_BATCH_SIZE", "8")))
lang = result.get("language", "en")
model_a, metadata = whisperx.load_align_model(language_code=lang, device=device)
aligned = whisperx.align(result["segments"], model_a, metadata, audio_data, device, return_char_alignments=False)
words = []
for seg in aligned.get("segments", []):
    for w in seg.get("words", []) or []:
        if w.get("start") is not None and w.get("end") is not None:
            words.append({"word": str(w.get("word", "")), "start": float(w["start"]), "end": float(w["end"])})
duration = float(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1",str(video)], text=True).strip())
payload = {"source": str(video), "duration": duration, "model": model_name, "language": lang, "segments": aligned.get("segments", []), "words": words}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"WhisperX complete: {len(payload['segments'])} segments, {len(words)} words")
''', encoding="utf-8")


def make_subtitle_source(transcript: Path, output: Path) -> None:
    d = read_json(transcript)
    write_json(output, {
        "language": d.get("language", "en"),
        "word_segments": [
            {"word": str(w.get("word", "")), "start": float(w["start"]), "end": float(w["end"])}
            for w in d.get("words", []) if isinstance(w, dict) and w.get("start") is not None and w.get("end") is not None
        ],
    })


def fallback_speakers(path: Path) -> None:
    write_json(path, {
        "version": "1.0", "mode": "speaker_attribution_fallback", "turns": [],
        "speaker_to_identity": {}, "speaker_identity_mapping": [], "available": False,
    })


def main() -> int:
    ap = argparse.ArgumentParser(description="One-command long-video -> Shorts pipeline")
    ap.add_argument("input", help="Local video path or YouTube/HTTP(S) URL")
    ap.add_argument("--style", choices=sorted(STYLES), default=DEFAULT_STYLE)
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None)
    ap.add_argument("--output", type=Path, default=ROOT / "out")
    ap.add_argument("--no-subtitles", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--scene-threshold", type=float, default=0.30)
    args = ap.parse_args()
    style = PRESETS[args.preset] if args.preset and args.style == DEFAULT_STYLE else args.style

    for p, label in [
        (GROQ_EDITOR, "Groq editor"), (LOCKED_BRIDGE, "locked bridge"),
        (MOMENT_RESOLVER, "moment resolver"), (CAMERA_BATCH, "camera batch"),
        (CAMERA_RENDER_BATCH, "camera render batch"),
        (VISUAL_IDENTITY, "visual identity reassignment")]:
        require(p, label)
    if not args.no_subtitles:
        require(SUBTITLE_TIMING_BATCH, "subtitle timing batch")
        require(SUBTITLE_RENDER_BATCH, "subtitle render batch")
    require(YUNET, "YuNet model")
    require(SFACE, "SFace model")

    if is_url(args.input):
        key = "url_" + short_hash(args.input)
        name = slug(urlparse(args.input).netloc + "_" + urlparse(args.input).path)
    else:
        src = Path(args.input).expanduser().resolve()
        if not src.exists():
            die(f"Input video not found: {src}")
        key = "file_" + short_hash(str(src) + str(src.stat().st_size) + str(src.stat().st_mtime_ns))
        name = slug(src.stem)

    state = ROOT / ".pipeline" / key
    work = state / "work"
    resolved = work / "resolved_moments"
    camera_plans = work / "camera_plans"
    camera_shorts = work / "camera_shorts"
    subtitle_timings = work / "subtitle_timings"
    out_dir = args.output.expanduser().resolve() / name

    if args.force and state.exists():
        shutil.rmtree(state)
    for d in [work, resolved, camera_plans, camera_shorts, subtitle_timings, out_dir]:
        d.mkdir(parents=True, exist_ok=True)

    log("=" * 72)
    log("MASTER PODCAST SHORTS PIPELINE")
    log("=" * 72)
    log(f"Input:      {args.input}")
    log(f"Style:      {style}")
    log(f"Subtitles:  {'off' if args.no_subtitles else 'on'}")
    log(f"Output:     {out_dir}")
    log(f"Checkpoint: {state}")

    # 01 Input
    source = work / "source.mp4"
    if not checkpoint(state / "01_input.done") or not source.exists():
        if is_url(args.input):
            run(["yt-dlp", "--no-playlist", "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b", "--merge-output-format", "mp4", "-o", str(source), args.input])
        else:
            shutil.copy2(Path(args.input).expanduser().resolve(), source)
        dur = probe_duration(source)
        (state / "01_input.done").write_text(str(dur), encoding="utf-8")
    duration = probe_duration(source)

    # 02 Audio
    audio = work / "audio.wav"
    if not checkpoint(state / "02_audio.done") or not audio.exists():
        run(["ffmpeg", "-y", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(audio)])
        (state / "02_audio.done").write_text("ok", encoding="utf-8")

    # 03 WhisperX
    transcript = work / "aligned_transcript.json"
    if not checkpoint(state / "03_transcript.done") or not transcript.exists():
        helper = work / "whisperx_stage.py"
        write_whisper_helper(helper)
        run([str(whisper_python()), str(helper), str(source), str(audio), str(transcript)])
        d = read_json(transcript)
        if not d.get("words"):
            die("WhisperX produced no aligned words.")
        (state / "03_transcript.done").write_text("ok", encoding="utf-8")

    # 04 Visual
    shots = work / "shots.json"
    visual_raw = work / "visual_intelligence_raw.json"
    visual = work / "visual_intelligence.json"
    if not checkpoint(state / "04_visual.done") or not visual.exists():
        log("[04] Detecting source shots...")
        detect_shots(source, shots, args.scene_threshold)

        visual_script = find_visual_script()

        log("[04] YuNet + SFace visual intelligence...")
        run([
            str(vision_python()), str(visual_script),
            "--video", str(source),
            "--shots", str(shots),
            "--yunet", str(YUNET),
            "--sface", str(SFACE),
            "--output", str(visual_raw),
        ])

        raw_data = read_json(visual_raw)
        if not raw_data.get("shot_visual_intelligence"):
            die("Visual intelligence output missing shot_visual_intelligence.")

        log("[04] Reassigning / cleaning visual identities...")
        run([
            sys.executable,
            str(VISUAL_IDENTITY),
            "--input", str(visual_raw),
            "--output", str(visual),
        ])

        clean_data = read_json(visual)
        if "identity_clean_observations" not in clean_data:
            die("Visual identity reassignment output missing identity_clean_observations.")
        if not clean_data.get("identity_clean_observations"):
            die("Visual identity reassignment produced no identity_clean_observations.")

        if not clean_data.get("shot_visual_intelligence"):
            die("Clean visual output lost shot_visual_intelligence.")

        (state / "04_visual.done").write_text("ok", encoding="utf-8")

    # 05 Speakers (optional safe fallback)
    speakers = work / "speakers.json"
    if not checkpoint(state / "05_speakers.done") or not speakers.exists():
        if SPEAKER.exists() and os.environ.get("HUGGINGFACE_TOKEN"):
            try:
                run([str(whisper_python()), str(SPEAKER), "--video", str(source), "--visual", str(visual), "--output", str(speakers)])
            except SystemExit:
                log("Speaker attribution failed; continuing with visual-only fallback.")
                fallback_speakers(speakers)
        else:
            log("Speaker attribution skipped (no script or HUGGINGFACE_TOKEN); using visual-only fallback.")
            fallback_speakers(speakers)
        (state / "05_speakers.done").write_text("ok", encoding="utf-8")

    # 06 Groq
    editorial = work / "groq_editorial.json"
    if not checkpoint(state / "06_groq.done") or not editorial.exists():
        if not os.environ.get("GROQ_API_KEY"):
            die("GROQ_API_KEY is not set.")
        run([sys.executable, str(GROQ_EDITOR), "--transcript", str(transcript), "--video-end", f"{duration:.6f}", "--output", str(editorial)])
        if not read_json(editorial).get("clips"):
            die("Groq returned no clips.")
        (state / "06_groq.done").write_text("ok", encoding="utf-8")

    # 07 Locked sequence
    locked = work / "locked_sequence.json"
    if not checkpoint(state / "07_locked.done") or not locked.exists():
        run([sys.executable, str(LOCKED_BRIDGE), "--editorial", str(editorial), "--shots", str(visual), "--output", str(locked)])
        if not read_json(locked).get("source_timing_locked"):
            die("Locked sequence did not declare source_timing_locked=true.")
        (state / "07_locked.done").write_text("ok", encoding="utf-8")

    # 08 Resolve each Short independently
    if not checkpoint(state / "08_resolved.done") or not (resolved / "manifest.json").exists():
        run([sys.executable, str(MOMENT_RESOLVER), "--locked", str(locked), "--shots", str(shots), "--output-dir", str(resolved)])
        rm = read_json(resolved / "manifest.json")
        if not rm.get("resolved"):
            die("No Shorts were resolved.")
        (state / "08_resolved.done").write_text("ok", encoding="utf-8")

    # 09 Camera plans + renders
    if not checkpoint(state / "09_camera.done") or not (camera_shorts / "manifest.json").exists():
        run([sys.executable, str(CAMERA_BATCH), "--resolved-dir", str(resolved), "--visual", str(visual), "--speakers", str(speakers), "--output-dir", str(camera_plans)])
        run([sys.executable, str(CAMERA_RENDER_BATCH), "--video", str(source), "--camera-dir", str(camera_plans), "--output-dir", str(camera_shorts)])
        if int(read_json(camera_shorts / "manifest.json").get("rendered_count", 0)) == 0:
            die("Camera stage produced no videos.")
        (state / "09_camera.done").write_text("ok", encoding="utf-8")

    # 10 Subtitle timings
    subtitle_source = work / "subtitle_source.json"
    if args.no_subtitles:
        log("[10] Subtitles disabled.")
    elif not checkpoint(state / "10_subtitle_timing.done") or not (subtitle_timings / "manifest.json").exists():
        make_subtitle_source(transcript, subtitle_source)
        run([sys.executable, str(SUBTITLE_TIMING_BATCH), "--locked", str(locked), "--transcript", str(subtitle_source), "--output-dir", str(subtitle_timings)])
        if int(read_json(subtitle_timings / "manifest.json").get("timing_count", 0)) == 0:
            die("Subtitle timing stage produced no files.")
        (state / "10_subtitle_timing.done").write_text("ok", encoding="utf-8")

    # 11 Final render / promotion
    if args.no_subtitles:
        for src in sorted(camera_shorts.glob("short_*_camera.mp4")):
            rank = re.search(r"short_(\d+)_camera\.mp4$", src.name).group(1)
            shutil.copy2(src, out_dir / f"short_{int(rank):02d}_final.mp4")
    elif not checkpoint(state / "11_final.done"):
        run([sys.executable, str(SUBTITLE_RENDER_BATCH), "--video-dir", str(camera_shorts), "--timing-dir", str(subtitle_timings), "--output-dir", str(out_dir), "--style", style])
        (state / "11_final.done").write_text("ok", encoding="utf-8")

    final_videos = sorted(out_dir.glob("short_*_final.mp4"))
    locked_data = read_json(locked)
    expected = len(locked_data.get("moments", []))
    if len(final_videos) != expected:
        die(f"Final QC failed: expected {expected} videos, found {len(final_videos)}.")

    rows = []
    for p in final_videos:
        probe = json.loads(capture(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-show_entries", "stream=width,height", "-of", "json", str(p)]))
        stream = next((s for s in probe.get("streams", []) if s.get("width") is not None), None)
        if not stream or int(stream["width"]) != 1080 or int(stream["height"]) != 1920:
            die(f"Final QC failed dimensions: {p}")
        rows.append({"output": str(p), "duration": float(probe["format"]["duration"]), "width": 1080, "height": 1920})

    write_json(out_dir / "manifest.json", {
        "version": "1.0", "pipeline": "run_pipeline", "input": args.input,
        "style": style, "subtitles": not args.no_subtitles,
        "selected_moments": expected, "output_directory": str(out_dir),
        "checkpoint_directory": str(state), "qc": {"count": len(rows), "videos": rows},
        "completed_at": time.time(),
    })

    log("\n" + "=" * 72)
    log("🎉 PIPELINE COMPLETE")
    log("=" * 72)
    log(f"Good moments selected: {expected}")
    log(f"Final Shorts:          {len(rows)}")
    log(f"Output:                {out_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)