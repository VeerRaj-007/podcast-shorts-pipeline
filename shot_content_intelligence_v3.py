#!/usr/bin/env python3
"""
Shot Content Intelligence V3

Quota-safe visual description for insert / non-face source shots.

Important design change from V2:
- NO response_format=json_object
- NO structured JSON generation from the vision model
- one image per request
- one shot per request
- reasoning disabled
- compact labeled plain-text response
- deterministic local parser
- no automatic retries by default
- save after every shot
- resume from existing output
- failed shots are recorded transparently

This stage DESCRIBES visuals only. It does not decide whether to edit them.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from groq import Groq

try:
    import pytesseract
except Exception:
    pytesseract = None


VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.6-27b")
MAX_COMPLETION_TOKENS = int(os.getenv("SHOT_CONTENT_MAX_TOKENS", "300"))
FRAME_WIDTH = int(os.getenv("SHOT_CONTENT_FRAME_WIDTH", "768"))

INSERT_TYPES = {
    "insert_or_screen",
    "non_face",
    "insert",
    "screen",
}

SCENE_TYPES = {
    "product_closeup",
    "person",
    "conversation",
    "gameplay",
    "screen_capture",
    "graphic",
    "object",
    "environment",
    "document",
    "montage",
    "other",
}

EDITORIAL_USES = {
    "supporting_example",
    "product_reference",
    "location_reference",
    "object_reference",
    "screen_reference",
    "gameplay_reference",
    "reaction_reference",
    "atmosphere",
    "transition",
    "branding",
    "evidence",
    "unknown",
}


PROMPT = r"""
Analyze the single video frame.

DESCRIPTION ONLY. Do not discuss editing, cuts, pacing, or what the editor should do.
Describe ONLY what is visibly supported by the image.

Return exactly these 9 labeled lines and nothing else:
SUMMARY: one short factual sentence
SCENE: one of product_closeup, person, conversation, gameplay, screen_capture, graphic, object, environment, document, montage, other
OBJECTS: comma-separated visible objects, or none
TAGS: comma-separated useful visual tags, or none
USES: up to 3 of supporting_example, product_reference, location_reference, object_reference, screen_reference, gameplay_reference, reaction_reference, atmosphere, transition, branding, evidence, unknown
PEOPLE: yes or no
COUNT: integer 0 or greater
TEXT: short exact visible text snippets, comma-separated, or none
CONFIDENCE: number from 0 to 1

Rules:
- Keep every line short.
- Never invent text.
- Never invent objects or people.
- If uncertain, use other and/or unknown rather than guessing.
""".strip()


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return default


def center_time(shot: dict[str, Any]) -> float:
    start = fnum(shot.get("start"))
    end = fnum(shot.get("end"), start)
    return start + max(0.0, end - start) / 2.0


def is_insert_shot(shot: dict[str, Any]) -> bool:
    shot_type = str(shot.get("shot_type", "")).strip().lower()
    if shot_type in INSERT_TYPES:
        return True
    if "insert" in shot_type or "screen" in shot_type:
        return True

    no_face_ratio = fnum(shot.get("no_face_ratio"), 0.0)
    avg_faces = fnum(shot.get("avg_face_count"), 0.0)
    return no_face_ratio >= 0.95 and avg_faces < 0.25


def extract_frame(video: str, time_sec: float, output_path: str) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{max(0.0, time_sec):.6f}",
        "-i",
        video,
        "-frames:v",
        "1",
        "-vf",
        f"scale={FRAME_WIDTH}:-2:flags=lanczos",
        "-q:v",
        "4",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("ffmpeg frame extraction failed:\n" + result.stderr[-3000:])


def local_ocr(image_path: str) -> list[str]:
    if pytesseract is None:
        return []
    try:
        text = pytesseract.image_to_string(image_path).strip()
    except Exception:
        return []

    lines: list[str] = []
    for raw in text.splitlines():
        value = re.sub(r"\s+", " ", raw).strip()
        if value and len(value) <= 100:
            lines.append(value)
    return lines[:6]


def encode_data_url(image_path: str) -> str:
    raw = Path(image_path).read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def normalize_text(value: str) -> str:
    value = value.strip()
    value = re.sub(r"^```(?:text)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value, flags=re.IGNORECASE)
    return value.strip()


def parse_kv_response(raw: str) -> dict[str, Any]:
    text = normalize_text(raw)
    fields: dict[str, str] = {}

    wanted = {
        "summary",
        "scene",
        "objects",
        "tags",
        "uses",
        "people",
        "count",
        "text",
        "confidence",
    }

    for line in text.splitlines():
        match = re.match(r"^\s*([A-Za-z_]+)\s*:\s*(.*?)\s*$", line)
        if not match:
            continue
        key = match.group(1).strip().lower()
        if key in wanted:
            fields[key] = match.group(2).strip()

    missing = [key for key in wanted if key not in fields]
    if missing:
        raise ValueError(f"Missing labeled fields: {missing}; raw={text[:1200]!r}")

    summary = fields["summary"].strip()
    if not summary:
        raise ValueError("SUMMARY is empty")

    scene = fields["scene"].strip().lower()
    if scene not in SCENE_TYPES:
        scene = "other"

    def split_items(value: str, limit: int) -> list[str]:
        value = value.strip()
        if not value or value.lower() == "none":
            return []
        parts = [p.strip() for p in value.split(",")]
        out: list[str] = []
        for part in parts:
            if part and part.lower() != "none" and part not in out:
                out.append(part)
        return out[:limit]

    objects = split_items(fields["objects"], 5)
    tags = split_items(fields["tags"], 6)

    uses_raw = split_items(fields["uses"], 3)
    uses = [u for u in uses_raw if u in EDITORIAL_USES]
    if not uses:
        uses = ["unknown"]

    people = fields["people"].strip().lower()
    people_present = people in {"yes", "true", "1"}

    try:
        count = max(0, min(int(re.sub(r"[^0-9-]", "", fields["count"]) or "0"), 20))
    except Exception:
        count = 0
    if count > 0:
        people_present = True

    visible_text = split_items(fields["text"], 6)

    confidence = max(0.0, min(1.0, fnum(fields["confidence"], 0.0)))

    return {
        "visual_summary": summary[:400],
        "scene_type": scene,
        "objects": objects,
        "visual_tags": tags,
        "editorial_uses": uses,
        "people_present": people_present,
        "person_count": count,
        "text_visible": visible_text,
        "content_confidence": confidence,
    }


def call_vision(
    client: Groq,
    shot_id: str,
    image_path: str,
    ocr: list[str],
) -> dict[str, Any]:
    ocr_note = ""
    if ocr:
        safe_ocr = " | ".join(x.replace("\n", " ") for x in ocr[:4])
        ocr_note = (
            "\nLocal OCR hint (may be wrong). Use it only to help notice possible text; "
            f"never copy it unless visible: {safe_ocr}"
        )

    content = [
        {
            "type": "text",
            "text": f"SHOT_ID={shot_id}\n{PROMPT}{ocr_note}",
        },
        {
            "type": "image_url",
            "image_url": {"url": encode_data_url(image_path)},
        },
    ]

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": content}],
        temperature=0.2,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        reasoning_effort="none",
    )

    raw = response.choices[0].message.content or ""
    if not raw.strip():
        raise ValueError("Empty vision response")

    return parse_kv_response(raw)


def build_inventory(shots_data: Any) -> list[dict[str, Any]]:
    shots = shots_data.get("shots", []) if isinstance(shots_data, dict) else shots_data
    if not isinstance(shots, list):
        raise ValueError("Expected shots list or object containing 'shots'")

    selected: list[dict[str, Any]] = []
    for shot in shots:
        if not isinstance(shot, dict) or not shot.get("shot_id"):
            continue
        if not is_insert_shot(shot):
            continue

        start = fnum(shot.get("start"))
        end = fnum(shot.get("end"), start)
        if end <= start:
            continue

        selected.append(
            {
                "shot_id": str(shot["shot_id"]),
                "source_start": start,
                "source_end": end,
                "sample_time": center_time(shot),
                "shot_type": str(shot.get("shot_type", "unknown")),
            }
        )

    selected.sort(key=lambda x: (x["source_start"], x["source_end"], x["shot_id"]))
    return selected


def create_base_payload(
    video: str,
    output: str,
    start: float | None,
    end: float | None,
    inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "version": "3.0",
        "mode": "shot_content_intelligence",
        "video": video,
        "vision_model": VISION_MODEL,
        "analysis_window": {"start": start, "end": end},
        "policy": {
            "purpose": "describe source visuals for context-aware editorial decisions",
            "editing_decision_made_here": False,
            "one_representative_frame_per_shot": True,
            "one_image_per_request": True,
            "output_format": "labeled_text_parsed_locally",
            "reasoning_effort": "none",
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
        },
        "stats": {
            "candidate_shots": len(inventory),
            "analyzed_shots": 0,
            "failed_shots": 0,
            "skipped_existing_shots": 0,
        },
        "shots": [],
        "failures": [],
    }


def normalize_existing_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Existing output is not a JSON object")
    payload.setdefault("shots", [])
    payload.setdefault("failures", [])
    payload.setdefault("stats", {})
    if not isinstance(payload["shots"], list):
        payload["shots"] = []
    if not isinstance(payload["failures"], list):
        payload["failures"] = []
    return payload


def analyzed_count(payload: dict[str, Any]) -> int:
    return sum(
        1
        for item in payload["shots"]
        if isinstance(item, dict)
        and item.get("status") == "analyzed"
        and item.get("visual_summary")
        and fnum(item.get("content_confidence")) > 0
    )


def analyze(
    video: str,
    shots_data: Any,
    output: str,
    start: float | None,
    end: float | None,
) -> dict[str, Any]:
    inventory = build_inventory(shots_data)

    if start is not None:
        inventory = [x for x in inventory if x["source_end"] >= start]
    if end is not None:
        inventory = [x for x in inventory if x["source_start"] <= end]

    if not inventory:
        raise RuntimeError("No insert/non-face source shots found in requested window")

    output_path = Path(output)
    if output_path.exists():
        payload = normalize_existing_payload(load_json(output_path))
        print(f"  Resuming existing output: {output}")
    else:
        payload = create_base_payload(video, output, start, end, inventory)

    existing_by_id = {
        str(item.get("shot_id")): item
        for item in payload["shots"]
        if isinstance(item, dict) and item.get("shot_id") and item.get("status") == "analyzed"
    }

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    with tempfile.TemporaryDirectory(prefix="shot_content_intel_v3_") as temp_dir:
        total = len(inventory)
        for index, shot in enumerate(inventory, start=1):
            shot_id = shot["shot_id"]

            if shot_id in existing_by_id:
                payload["stats"]["skipped_existing_shots"] = int(
                    payload["stats"].get("skipped_existing_shots", 0)
                ) + 1
                print(f"    [{index}/{total}] {shot_id} -> already analyzed, skipping")
                continue

            frame_path = str(Path(temp_dir) / f"{shot_id}.jpg")
            print(
                f"    [{index}/{total}] {shot_id} "
                f"{shot['source_start']:.3f}-{shot['source_end']:.3f}s"
            )

            try:
                extract_frame(video, shot["sample_time"], frame_path)
                ocr = local_ocr(frame_path)
                result = call_vision(client, shot_id, frame_path, ocr)

                record = {
                    "shot_id": shot_id,
                    "source_start": shot["source_start"],
                    "source_end": shot["source_end"],
                    "sample_time": shot["sample_time"],
                    "shot_type": shot["shot_type"],
                    **result,
                    "local_ocr": ocr,
                    "status": "analyzed",
                }

                payload["shots"] = [
                    item for item in payload["shots"] if str(item.get("shot_id")) != shot_id
                ] + [record]
                existing_by_id[shot_id] = record
                payload["failures"] = [
                    item for item in payload["failures"] if str(item.get("shot_id")) != shot_id
                ]

                print(
                    f"      OK: {record['scene_type']} | "
                    f"conf={record['content_confidence']:.2f} | "
                    f"{record['visual_summary']}"
                )

            except Exception as exc:
                failure = {
                    "shot_id": shot_id,
                    "source_start": shot["source_start"],
                    "source_end": shot["source_end"],
                    "error": str(exc),
                }
                payload["failures"] = [
                    item for item in payload["failures"] if str(item.get("shot_id")) != shot_id
                ] + [failure]
                print(f"      FAILED: {shot_id}: {exc}")

            payload["shots"].sort(
                key=lambda x: (fnum(x.get("source_start")), str(x.get("shot_id")))
            )
            payload["failures"].sort(
                key=lambda x: (fnum(x.get("source_start")), str(x.get("shot_id")))
            )
            payload["stats"]["candidate_shots"] = len(inventory)
            payload["stats"]["analyzed_shots"] = analyzed_count(payload)
            payload["stats"]["failed_shots"] = len(payload["failures"])
            save_json(output_path, payload)

    payload["stats"]["candidate_shots"] = len(inventory)
    payload["stats"]["analyzed_shots"] = analyzed_count(payload)
    payload["stats"]["failed_shots"] = len(payload["failures"])
    save_json(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Quota-safe semantic shot content intelligence v3")
    parser.add_argument("--video", required=True)
    parser.add_argument("--shots", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", type=float, default=None)
    parser.add_argument("--end", type=float, default=None)
    args = parser.parse_args()

    print("=== SHOT CONTENT INTELLIGENCE V3 ===")
    print(f"model: {VISION_MODEL}")
    print(f"max_completion_tokens: {MAX_COMPLETION_TOKENS}")
    print("one image/request: yes")
    print("response format: plain labeled text (local parse)")
    print("reasoning_effort: none")
    print("retries: none")

    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set")

    payload = analyze(
        video=args.video,
        shots_data=load_json(args.shots),
        output=args.output,
        start=args.start,
        end=args.end,
    )

    print(json.dumps({
        "output": args.output,
        "vision_model": payload["vision_model"],
        **payload["stats"],
    }, indent=2))

    if payload["stats"]["failed_shots"]:
        print("\nCOMPLETED WITH FAILURES: see `failures` in the output JSON.")
    else:
        print("\nSUCCESS: all candidate shots have semantic visual metadata.")


if __name__ == "__main__":
    main()
