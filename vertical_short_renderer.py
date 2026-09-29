from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

SOURCE_W = 1920.0
SOURCE_H = 1080.0
OUT_W = 1080
OUT_H = 1920
TARGET_AR = OUT_W / OUT_H
FOCUS_ZOOM = 1.08
SPEAKERS = {"PERSON_0", "PERSON_1"}


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def normalize_shot_content(data: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not data:
        return {}
    raw = data.get("shots", [])
    if not isinstance(raw, list):
        return {}
    return {str(x["shot_id"]): x for x in raw if isinstance(x, dict) and x.get("shot_id")}


def shot_has_text_card(shot_id: str | None, shot_content: dict[str, dict[str, Any]]) -> bool:
    if not shot_id:
        return False
    item = shot_content.get(str(shot_id), {})
    text = item.get("text")
    if isinstance(text, list):
        return any(str(x).strip() not in {"", "none"} for x in text)
    return bool(text and str(text).strip().lower() != "none")


def shot_is_question_card(shot_id: str | None, shot_content: dict[str, dict[str, Any]]) -> bool:
    if not shot_id:
        return False
    item = shot_content.get(str(shot_id), {})
    text = item.get("text")
    values = text if isinstance(text, list) else [text]
    for value in values:
        normalized = str(value or "").strip().lower()
        if "what's the best bit of direction" in normalized:
            return True
        if "what's the best piece of direction" in normalized:
            return True
    return False


def question_card_crop() -> dict[str, Any]:
    # The burned-in interview question sits at the lower-right of the source frame.
    # Use the rightmost full-height 9:16 window so the plate is retained instead
    # of centering the crop on the person reading it.
    crop_h = SOURCE_H
    crop_w = SOURCE_H * TARGET_AR
    left = SOURCE_W - crop_w
    left, top, crop_w, crop_h = clamp_crop(left, 0.0, crop_w, crop_h)
    return {
        "left": round(left, 3),
        "top": round(top, 3),
        "width": round(crop_w, 3),
        "height": round(crop_h, 3),
        "mode": "question_card",
        "zoom": 1.0,
    }

def fnum(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def bbox(obs: dict[str, Any]) -> tuple[float, float, float, float] | None:
    b = obs.get("bbox")
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        return None
    x, y, w, h = map(fnum, b)
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def observations_in_range(observations: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    return [
        o
        for o in observations
        if start - 1e-4 <= fnum(o.get("timestamp"), -1.0) <= end + 1e-4
        and o.get("identity_id") in SPEAKERS
    ]


def robust_center(items: list[dict[str, Any]]) -> tuple[float, float, float] | None:
    pts: list[tuple[float, float, float]] = []
    for o in items:
        b = bbox(o)
        if not b:
            continue
        x, y, w, h = b
        pts.append((x + w / 2.0, y + h / 2.0, w * h))
    if not pts:
        return None
    pts.sort(key=lambda p: p[0])
    mid = len(pts) // 2
    cx = pts[mid][0] if len(pts) % 2 else (pts[mid - 1][0] + pts[mid][0]) / 2.0
    ys = sorted(p[1] for p in pts)
    cy = ys[mid] if len(ys) % 2 else (ys[mid - 1] + ys[mid]) / 2.0
    area = sum(p[2] for p in pts) / len(pts)
    return cx, cy, area


def clamp_crop(left: float, top: float, width: float, height: float) -> tuple[float, float, float, float]:
    width = min(width, SOURCE_W)
    height = min(height, SOURCE_H)
    left = max(0.0, min(left, SOURCE_W - width))
    top = max(0.0, min(top, SOURCE_H - height))
    return left, top, width, height


def speaker_crop(
    observations: list[dict[str, Any]],
    source_start: float,
    source_end: float,
    target: str | None,
) -> dict[str, Any]:
    items = [
        o for o in observations_in_range(observations, source_start, source_end)
        if o.get("identity_id") == target
    ]
    center = robust_center(items)
    if center is None:
        return cover_center_crop()
    cx, cy, _ = center
    crop_h = SOURCE_H / FOCUS_ZOOM
    crop_w = crop_h * TARGET_AR
    desired_face_y = SOURCE_H * 0.36
    left = cx - crop_w / 2.0
    top = cy - desired_face_y
    left, top, crop_w, crop_h = clamp_crop(left, top, crop_w, crop_h)
    return {
        "left": round(left, 3),
        "top": round(top, 3),
        "width": round(crop_w, 3),
        "height": round(crop_h, 3),
        "mode": "speaker_focus",
        "zoom": FOCUS_ZOOM,
    }


def cover_center_crop() -> dict[str, Any]:
    crop_h = SOURCE_H
    crop_w = SOURCE_H * TARGET_AR
    return {
        "left": round((SOURCE_W - crop_w) / 2.0, 3),
        "top": 0.0,
        "width": round(crop_w, 3),
        "height": crop_h,
        "mode": "cover_center",
        "zoom": 1.0,
    }


def two_person_crop(
    observations: list[dict[str, Any]],
    source_start: float,
    source_end: float,
) -> dict[str, Any]:
    items = observations_in_range(observations, source_start, source_end)
    boxes = [b for b in (bbox(o) for o in items) if b]
    if len(boxes) < 2:
        return cover_center_crop()
    min_x = min(b[0] for b in boxes)
    max_x = max(b[0] + b[2] for b in boxes)
    min_y = min(b[1] for b in boxes)
    max_y = max(b[1] + b[3] for b in boxes)
    union_w = max_x - min_x
    crop_w = max(SOURCE_H * TARGET_AR, union_w + 180.0)
    crop_h = crop_w / TARGET_AR
    if crop_h > SOURCE_H:
        return cover_center_crop()
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    left, top, crop_w, crop_h = clamp_crop(
        cx - crop_w / 2.0,
        cy - crop_h / 2.0,
        crop_w,
        crop_h,
    )
    return {
        "left": round(left, 3),
        "top": round(top, 3),
        "width": round(crop_w, 3),
        "height": round(crop_h, 3),
        "mode": "two_person",
        "zoom": round((SOURCE_H * TARGET_AR) / crop_w, 3),
    }


def ffprobe_duration(video: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video),
    ]
    return fnum(subprocess.check_output(cmd, text=True).strip())


def validate_speaker_av_sync(plan: dict[str, Any]) -> list[str]:
    """Speaker-focus visuals must use exactly the same source interval as audio."""
    errors: list[str] = []
    for i, seg in enumerate(plan.get("visual_segments", [])):
        mode = str(seg.get("visual_mode", ""))
        if mode != "speaker_focus":
            continue
        a0 = fnum(seg.get("audio_source_start"), -1.0)
        a1 = fnum(seg.get("audio_source_end"), -1.0)
        v0 = fnum(seg.get("visual_source_start"), -1.0)
        v1 = fnum(seg.get("visual_source_end"), -1.0)
        if a0 < 0 or a1 <= a0:
            errors.append(f"segment {i}: speaker_focus has invalid audio source range")
            continue
        if abs(v0 - a0) > 1e-3 or abs(v1 - a1) > 1e-3:
            errors.append(
                f"segment {i}: speaker_focus AV sync mismatch "
                f"audio={a0:.3f}->{a1:.3f}, visual={v0:.3f}->{v1:.3f}"
            )
    return errors


def validate_visual_source(plan: dict[str, Any], source_duration: float) -> list[str]:
    errors: list[str] = []
    for i, seg in enumerate(plan.get("visual_segments", [])):
        s = fnum(seg.get("visual_source_start"), -1)
        e = fnum(seg.get("visual_source_end"), -1)
        if s < -1e-6 or e <= s + 1e-6:
            errors.append(f"segment {i}: invalid visual source range {s}->{e}")
        if e > source_duration + 0.05:
            errors.append(
                f"segment {i}: visual source exceeds video duration {e:.3f} > {source_duration:.3f}"
            )
    return errors


def group_segments(plan: dict[str, Any]) -> list[dict[str, Any]]:
    segments = plan.get("visual_segments", [])
    if not isinstance(segments, list) or not segments:
        raise RuntimeError("Visual plan contains no visual_segments")
    groups: list[dict[str, Any]] = []
    by_decision: dict[int, dict[str, Any]] = {}
    for seg in sorted(segments, key=lambda s: fnum(s.get("output_start"))):
        decision_index = int(fnum(seg.get("decision_index"), 0))
        if decision_index not in by_decision:
            group = {
                "decision_index": decision_index,
                "output_start": fnum(seg.get("output_start")),
                "output_end": fnum(seg.get("output_end")),
                "segments": [],
                "visual_mode": str(seg.get("visual_mode", "cover_center")),
                "target_identity": seg.get("target_identity"),
            }
            by_decision[decision_index] = group
            groups.append(group)
        group = by_decision[decision_index]
        group["segments"].append(seg)
        group["output_start"] = min(group["output_start"], fnum(seg.get("output_start")))
        group["output_end"] = max(group["output_end"], fnum(seg.get("output_end")))
    return groups


def crop_for_group(
    group: dict[str, Any],
    observations: list[dict[str, Any]],
    shot_content: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    segments = group["segments"]
    source_start = min(fnum(s.get("visual_source_start")) for s in segments)
    source_end = max(fnum(s.get("visual_source_end")) for s in segments)
    mode = str(group.get("visual_mode", "cover_center"))
    target = group.get("target_identity")

    # The question plate is a special vertical-framing case. It lives on the
    # right side of the source frame, so a centered 9:16 crop clips its text.
    if any(shot_is_question_card(str(s.get("visual_shot_id")), shot_content) for s in segments):
        return question_card_crop()

    # Other text-card shots still need the full source height, but they should not
    # inherit the question-card's right-biased framing.
    if any(shot_has_text_card(str(s.get("visual_shot_id")), shot_content) for s in segments):
        return cover_center_crop()

    if mode == "speaker_focus" and target in SPEAKERS:
        return speaker_crop(observations, source_start, source_end, target)
    if mode == "two_person":
        return two_person_crop(observations, source_start, source_end)
    return cover_center_crop()


def detect_leading_audio_visual_mismatch(plan: dict[str, Any]) -> tuple[float, float, float]:
    """Detect the known leading case without deleting its audio.

    Returns (mismatch_duration, audio_start, first_useful_visual_start).
    The first AV segment carries valid spoken audio but an unrelated picture.
    We replace only that picture with a question-card hold, then resume the
    original source-synchronous video at the next segment.
    """
    segments = plan.get("visual_segments")
    if not isinstance(segments, list) or len(segments) < 2:
        return 0.0, 0.0, 0.0

    first = segments[0]
    second = segments[1]
    output_start = fnum(first.get("output_start"), 0.0)
    output_end = fnum(first.get("output_end"), 0.0)
    audio_start = fnum(first.get("audio_source_start"), -1.0)
    visual_start = fnum(first.get("visual_source_start"), -1.0)
    next_visual_start = fnum(second.get("visual_source_start"), -1.0)
    first_duration = output_end - output_start
    mismatch = next_visual_start - visual_start if visual_start >= 0 and next_visual_start >= 0 else 0.0

    if (
        output_start <= 1e-6
        and first_duration > 1e-3
        and audio_start >= 0.0
        and next_visual_start > visual_start + 1e-3
        and abs((visual_start + first_duration) - next_visual_start) <= 0.05
    ):
        return first_duration, audio_start, next_visual_start

    return 0.0, 0.0, 0.0

def render(video: Path, plan: dict[str, Any], observations: list[dict[str, Any]], shot_content: dict[str, dict[str, Any]], output: Path) -> None:
    source_duration = ffprobe_duration(video)
    sync_errors = validate_speaker_av_sync(plan)
    if sync_errors:
        raise RuntimeError("Speaker AV sync validation failed:\n" + "\n".join(sync_errors))
    prefix_duration = 0.0
    prefix_audio_start = 0.0
    prefix_question_source = 0.0
    if prefix_duration > 0.0:
        print(
            "replacing unrelated leading picture with question-card hold: "
            f"{prefix_duration:.3f}s; preserving audio from {prefix_audio_start:.3f}s"
        )

    errors = validate_visual_source(plan, source_duration)
    if errors:
        raise RuntimeError("Invalid visual plan:\n" + "\n".join(errors))

    # The first AV segment is valid audio but invalid picture. For rendering,
    # remove only that visual realization and replace it with a question-card
    # hold. The audio timeline remains untouched and therefore stays 40.180s.
    video_plan = plan

    groups = group_segments(video_plan)
    if not groups:
        raise RuntimeError("Visual plan contains no creative visual groups")

    with tempfile.TemporaryDirectory(prefix="vertical_short_render_") as td:
        # Decode the source once and trim video with the exact same source-time
        # basis used by the audio. The synthetic prefix is a deliberate visual-only
        # hold; every following interview frame remains source-synchronous.
        input_args: list[str] = ["-i", str(video)]

        filters: list[str] = []
        group_labels: list[str] = []
        seg_index = 0

        if prefix_duration > 0.0:
            crop = question_card_crop()
            x = max(0, int(round(crop["left"])))
            y = max(0, int(round(crop["top"])))
            w = max(2, int(round(crop["width"])))
            h = max(2, int(round(crop["height"])))
            prefix_label = "qlead"
            # Grab one frame from the first useful interview/question shot and
            # hold it for exactly the invalid-picture duration.
            frame_end = prefix_question_source + (1.0 / 30.0)
            filters.append(
                f"[0:v]trim=start={prefix_question_source:.6f}:end={frame_end:.6f},"
                "setpts=PTS-STARTPTS,select='eq(n\\,0)',"
                f"loop=loop={max(0, int(round(prefix_duration * 30.0)) - 1)}:size=1:start=0,"
                "setpts=N/(30*TB),"
                f"crop={w}:{h}:{x}:{y},scale={OUT_W}:{OUT_H}:flags=lanczos,"
                f"setsar=1,fps=30[{prefix_label}]"
            )
            group_labels.append(prefix_label)

        for group in groups:
            crop = crop_for_group(group, observations, shot_content)
            x = max(0, int(round(crop["left"])))
            y = max(0, int(round(crop["top"])))
            w = max(2, int(round(crop["width"])))
            h = max(2, int(round(crop["height"])))
            state_labels: list[str] = []

            for _seg in group["segments"]:
                vs = fnum(_seg.get("visual_source_start"))
                ve = fnum(_seg.get("visual_source_end"))
                label = f"v{seg_index}"
                filters.append(
                    f"[0:v]trim=start={vs:.6f}:end={ve:.6f},"
                    "setpts=PTS-STARTPTS,"
                    f"crop={w}:{h}:{x}:{y},"
                    f"scale={OUT_W}:{OUT_H}:flags=lanczos,"
                    f"setsar=1,fps=30[{label}]"
                )
                state_labels.append(label)
                seg_index += 1

            group_label = f"g{group['decision_index']}"
            filters.append(
                f"{''.join(f'[{x}]' for x in state_labels)}"
                f"concat=n={len(state_labels)}:v=1:a=0[{group_label}]"
            )
            group_labels.append(group_label)

        filters.append(
            f"{''.join(f'[{x}]' for x in group_labels)}"
            f"concat=n={len(group_labels)}:v=1:a=0[vout]"
        )

        audio_start = min(fnum(seg.get("audio_source_start")) for seg in plan["visual_segments"])
        audio_end = max(fnum(seg.get("audio_source_end")) for seg in plan["visual_segments"])
        audio_input_index = 0
        filters.append(
            f"[{audio_input_index}:a]atrim=start={audio_start:.6f}:end={audio_end:.6f},"
            "asetpts=PTS-STARTPTS[aout]"
        )

        cmd = [
            "ffmpeg", "-y", *input_args,
            "-filter_complex", ";".join(filters),
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
            "-movflags", "+faststart", "-shortest", str(output),
        ]
        subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Render a visual story plan with one crop treatment per creative visual state"
    )
    p.add_argument("--video", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--visual", required=True)
    p.add_argument("--shot-content", required=False, default=None)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    video = Path(args.video)
    plan = load_json(Path(args.plan))
    visual = load_json(Path(args.visual))
    observations = visual.get("identity_clean_observations") or visual.get("observations_deduped") or []
    if not isinstance(observations, list):
        observations = []
    shot_content = normalize_shot_content(load_json(Path(args.shot_content))) if args.shot_content else {}

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    render(video, plan, observations, shot_content, output)

    print("=== VERTICAL SHORT RENDERER ===")
    print(f"creative visual groups: {len(group_segments(plan))}")
    print(f"source realization segments: {len(plan.get('visual_segments', []))}")
    print(f"rendered: {output}")
    print(f"duration: {ffprobe_duration(output):.3f}s")


if __name__ == "__main__":
    main()
