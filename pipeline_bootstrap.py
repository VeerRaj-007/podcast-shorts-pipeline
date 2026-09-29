#!/usr/bin/env python3
"""Podcast Shorts pipeline bootstrap / discovery tool.

Run from the project root. It does not modify existing project scripts.
It discovers likely pipeline components, validates basic external tools/env vars,
and writes .pipeline/discovery.json for the next wiring step.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path.cwd()
OUT = ROOT / ".pipeline"
DISCOVERY = OUT / "discovery.json"

SKIP_DIRS = {".git", ".venv", ".venv-whisperx", "__pycache__", ".pipeline", "node_modules"}

PATTERNS = {
    "transcription": ["whisperx", "load_model", "transcribe"],
    "alignment": ["whisperx", "align", "align_model"],
    "groq": ["from groq import Groq", "Groq(", "openai/gpt-oss-120b", "groq"],
    "visual_intelligence": ["yunet", "sface", "onnxruntime", "face_recognition"],
    "speaker_attribution": ["pyannote", "speaker-diarization", "speaker_attribution"],
    "camera_director": ["camera director", "camera_director", "speaker_focus", "crop"],
    "camera_renderer": ["camera renderer", "filter_complex", "crop=", "concat=n="],
    "subtitle_timing": ["subtitle timing", "word_segments", "max_words", "caption"],
    "subtitle_renderer": ["subtitle renderer", "libass", "ass", "subtitles"],
    "av_resolver": ["av resolver", "source_in", "source_out", "render_segments"],
    "editorial_sequence": ["editorial sequence", "selected_moment", "timeline_actions"],
}

LIKELY_NAME_HINTS = {
    "transcription": ["transcribe", "whisper"],
    "alignment": ["align", "whisper"],
    "groq": ["groq", "editorial", "selector"],
    "visual_intelligence": ["visual_intelligence", "yunet", "sface"],
    "speaker_attribution": ["speaker_attribution", "speaker"],
    "camera_director": ["camera_director"],
    "camera_renderer": ["camera_renderer", "vertical_short_renderer"],
    "subtitle_timing": ["subtitle_timing", "caption"],
    "subtitle_renderer": ["subtitle_renderer", "subtitle"],
    "av_resolver": ["av_resolver", "resolver"],
    "editorial_sequence": ["editorial_sequence", "sequence_director"],
}


def tool_version(cmd: str) -> str | None:
    exe = shutil.which(cmd)
    if not exe:
        return None
    try:
        p = subprocess.run([cmd, "--version"], capture_output=True, text=True, timeout=10)
        text = (p.stdout or p.stderr).strip().splitlines()
        return text[0] if text else exe
    except Exception:
        return exe


def env_presence() -> dict[str, bool]:
    return {
        "GROQ_API_KEY": bool(os.getenv("GROQ_API_KEY")),
        "HUGGINGFACE_TOKEN": bool(os.getenv("HUGGINGFACE_TOKEN")),
    }


def file_score(text: str, name: str, patterns: list[str], hints: list[str]) -> int:
    low = text.lower()
    base = 0
    for p in patterns:
        base += 2 if p.lower() in low else 0
    n = name.lower()
    for h in hints:
        base += 3 if h.lower() in n else 0
    return base


def discover_py_files() -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for p in ROOT.rglob("*.py"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        roles: dict[str, int] = {}
        for role, pats in PATTERNS.items():
            score = file_score(text, p.name, pats, LIKELY_NAME_HINTS[role])
            if score >= 4:
                roles[role] = score
        if roles:
            found.append({
                "path": str(p.relative_to(ROOT)).replace("\\", "/"),
                "roles": dict(sorted(roles.items(), key=lambda kv: (-kv[1], kv[0]))),
                "size": p.stat().st_size,
            })
    found.sort(key=lambda x: (-max(x["roles"].values()), x["path"]))
    return found


def find_existing_artifacts() -> dict[str, list[str]]:
    buckets = {
        "transcripts": ["transcript", "aligned"],
        "visual": ["visual_intelligence", "visual"],
        "shots": ["shots", "shot"],
        "speaker": ["speaker", "attribution"],
        "camera": ["camera"],
        "subtitle": ["subtitle", "caption"],
        "editorial": ["editorial", "sequence"],
        "outputs": [".mp4"],
    }
    out: dict[str, list[str]] = {k: [] for k in buckets}
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        rel = str(p.relative_to(ROOT)).replace("\\", "/")
        low = rel.lower()
        for bucket, needles in buckets.items():
            if any(n in low for n in needles):
                out[bucket].append(rel)
    for v in out.values():
        v.sort()
    return out


def main() -> int:
    print("=== PODCAST SHORTS PIPELINE BOOTSTRAP ===")
    print(f"project root: {ROOT}")
    print()

    OUT.mkdir(exist_ok=True)

    tools = {k: tool_version(k) for k in ("ffmpeg", "ffprobe", "yt-dlp")}
    envs = env_presence()
    py_files = discover_py_files()
    artifacts = find_existing_artifacts()

    print("TOOLS")
    for k, v in tools.items():
        print(f"  {k:8s}: {'OK - ' + v if v else 'MISSING'}")

    print("\nENV")
    for k, v in envs.items():
        print(f"  {k:18s}: {'SET' if v else 'MISSING'}")

    print("\nLIKELY PIPELINE SCRIPTS")
    shown = set()
    for item in py_files:
        best_role = next(iter(item["roles"]))
        key = (best_role, item["path"])
        if key in shown:
            continue
        shown.add(key)
        print(f"  {best_role:20s} {item['path']}  score={item['roles'][best_role]}")

    payload = {
        "version": "bootstrap-1.0",
        "project_root": str(ROOT),
        "python": sys.version,
        "tools": tools,
        "environment": envs,
        "python_candidates": py_files,
        "existing_artifacts": artifacts,
        "notes": [
            "No existing project script was modified.",
            "Candidate discovery is heuristic; the next wiring step must verify interfaces before execution.",
            "The intended pipeline keeps Groq editorial source boundaries authoritative.",
            "Source visuals only; no external/generated inserts in v1.",
        ],
    }
    DISCOVERY.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nWROTE: {DISCOVERY}")
    print("\nNEXT: paste the complete terminal output here. We will use it to wire the master runner without guessing script interfaces.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())