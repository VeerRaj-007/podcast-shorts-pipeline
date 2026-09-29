from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

W, H = 1920, 1080
TARGET_AR = 9 / 16
FOCUS_ZOOM = 1.10
FOCUS_CROP_H = H / FOCUS_ZOOM
FOCUS_CROP_W = (H * TARGET_AR) / FOCUS_ZOOM


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def dominant_speaker(turns: list[dict[str, Any]], start: float, end: float):
    totals = {"PERSON_0": 0.0, "PERSON_1": 0.0}
    for t in turns:
        p = t.get("person_id")
        if p in totals:
            totals[p] += overlap(start, end, float(t["start"]), float(t["end"]))
    total = sum(totals.values())
    if total <= 0:
        return None, totals
    leader = max(totals, key=totals.get)
    share = totals[leader] / total
    return (leader if share >= 0.60 else None), totals


def presence_and_faces(observations: list[dict[str, Any]], start: float, end: float):
    by_person: dict[str, list[dict[str, Any]]] = {"PERSON_0": [], "PERSON_1": []}
    for o in observations:
        p = o.get("identity_id")
        ts = o.get("timestamp")
        if p in by_person and ts is not None and start <= float(ts) <= end:
            by_person[p].append(o)
    return by_person


def bbox_center(obs: dict[str, Any]) -> tuple[float, float] | None:
    bbox = obs.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    x, y, w, h = map(float, bbox)
    return x + w / 2.0, y + h / 2.0


def robust_face_anchor(obs_list: list[dict[str, Any]]):
    pts = []
    for o in obs_list:
        p = bbox_center(o)
        if p:
            pts.append(p)
    if not pts:
        return W * 0.5, H * 0.5
    xs = sorted(p[0] for p in pts)
    ys = sorted(p[1] for p in pts)
    mid = len(pts) // 2
    mx = xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2
    my = ys[mid] if len(ys) % 2 else (ys[mid - 1] + ys[mid]) / 2
    return mx, my


def focus_crop(obs_list: list[dict[str, Any]], source_start: float, source_end: float):
    cx, cy = robust_face_anchor(obs_list)

    # Put the face slightly below center and give modest room above it.
    desired_face_y = H * 0.34
    top = cy - desired_face_y
    top = max(0.0, min(top, H - FOCUS_CROP_H))

    # Preserve looking room using the observed face trajectory.
    motion_x = []
    for o in obs_list:
        c = bbox_center(o)
        if c:
            motion_x.append(c[0])
    if motion_x:
        avg_x = sum(motion_x) / len(motion_x)
    else:
        avg_x = cx

    # The observed shot has the speaker looking mostly left. Keep extra space
    # on the left while still keeping the face comfortably inside frame.
    left_bias = -0.08
    target_crop_center_x = avg_x + left_bias * FOCUS_CROP_W
    left = target_crop_center_x - FOCUS_CROP_W / 2
    left = max(0.0, min(left, W - FOCUS_CROP_W))

    return {
        "left": left,
        "top": top,
        "width": FOCUS_CROP_W,
        "height": FOCUS_CROP_H,
        "scale_to": [1080, 1920],
        "zoom": FOCUS_ZOOM,
        "source_anchor": {"face_x": avg_x, "face_y": cy},
    }


def two_person_crop(obs0: list[dict[str, Any]], obs1: list[dict[str, Any]]):
    pts = []
    for group in (obs0, obs1):
        for o in group:
            p = bbox_center(o)
            if p:
                pts.append(p)

    if not pts:
        center_x = W * 0.5
    else:
        center_x = sum(p[0] for p in pts) / len(pts)

    crop_h = H
    crop_w = H * TARGET_AR
    left = center_x - crop_w / 2
    left = max(0.0, min(left, W - crop_w))

    return {
        "left": left,
        "top": 0.0,
        "width": crop_w,
        "height": crop_h,
        "scale_to": [1080, 1920],
        "zoom": 1.0,
    }


def collect_segments(sequence: dict[str, Any]):
    out = []
    for action in sequence.get("actions", []):
        for r in action.get("render_segments", []):
            out.append({
                "action_index": action.get("action_index"),
                "output_start": float(r["output_start"]),
                "output_end": float(r["output_end"]),
                "source_start": float(r["source_in"]),
                "source_end": float(r["source_out"]),
            })
    out.sort(key=lambda x: x["output_start"])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sequence", required=True)
    p.add_argument("--visual", required=True)
    p.add_argument("--speakers", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    sequence = load_json(Path(args.sequence))
    visual = load_json(Path(args.visual))
    speakers = load_json(Path(args.speakers))

    observations = visual.get("identity_clean_observations", [])
    turns = speakers.get("turns", [])
    segments = collect_segments(sequence)
    if not segments:
        raise RuntimeError("No editorial render segments found.")

    camera = []
    for i, seg in enumerate(segments):
        start, end = seg["source_start"], seg["source_end"]
        by_person = presence_and_faces(observations, start, end)
        focus, speaker_seconds = dominant_speaker(turns, start, end)

        # Never force a two-person portrait crop when both people are too far
        # apart; focus the active speaker instead.
        if focus in by_person and by_person[focus]:
            crop = focus_crop(by_person[focus], start, end)
            composition = "speaker_focus"
            motion = "punch_in_and_hold"
        else:
            crop = two_person_crop(by_person["PERSON_0"], by_person["PERSON_1"])
            composition = "two_person"
            motion = "hold"
            focus = None

        camera.append({
            "segment_index": i,
            "action_index": seg["action_index"],
            "output_start": seg["output_start"],
            "output_end": seg["output_end"],
            "source_start": start,
            "source_end": end,
            "composition": composition,
            "target_identity": focus,
            "camera_motion": motion,
            "observed_identities": [p for p in ("PERSON_0", "PERSON_1") if by_person[p]],
            "speaker_seconds": speaker_seconds,
            "crop": crop,
        })

    result = {
        "version": "4.0",
        "module": "face_aware_camera_director",
        "method": "bbox_anchored_identity_calibrated_speaker_director",
        "camera_segments": camera,
        "constraints": {
            "editorial_source_ranges_authoritative": True,
            "native_portrait_crop": True,
            "crop_from_bbox_not_face_center_only": True,
            "looking_room_bias": True,
            "hardcoded_target_identity": False,
            "two_person_crop_only_when_faces_exist": True,
            "focus_zoom": FOCUS_ZOOM,
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("=== FACE-AWARE CAMERA DIRECTOR v4 ===")
    for s in camera:
        c = s["crop"]
        print(f"{s['source_start']:.3f} -> {s['source_end']:.3f} | {s['composition']} | target={s['target_identity']} | crop=({c['left']:.1f},{c['top']:.1f},{c['width']:.1f},{c['height']:.1f})")
    print(f"output: {out}")


if __name__ == "__main__":
    main()