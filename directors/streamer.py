from __future__ import annotations

from typing import Any, Dict, List, Optional

from .models import DirectorAction, DirectorPlan


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _face_event_for_camera_event(
    face_plan: Optional[Dict[str, Any]], camera_event: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    if not face_plan:
        return None

    cs = _float(camera_event.get("start"), -1)
    ce = _float(camera_event.get("end"), -1)
    best = None
    best_overlap = 0.0

    for face_event in face_plan.get("events", []):
        fs = _float(face_event.get("start"), -1)
        fe = _float(face_event.get("end"), -1)
        overlap = min(ce, fe) - max(cs, fs)
        if overlap > best_overlap:
            best_overlap = overlap
            best = face_event

    return best


def build_streamer_plan(
    clip_start: float,
    clip_end: float,
    camera_plan: Dict[str, Any],
    face_plan: Optional[Dict[str, Any]],
) -> DirectorPlan:
    """Convert existing event intelligence into renderer-independent actions.

    This preserves the behavior we want from the streamer reference:
    normal layout -> big face reaction -> hold -> normal layout.
    There is NO continuous zoom action here.
    """

    actions: List[DirectorAction] = []

    for event in camera_plan.get("events", []):
        start = _float(event.get("start"), -1)
        end = _float(event.get("end"), -1)
        if end <= clip_start or start >= clip_end or end <= start:
            continue

        face_event = _face_event_for_camera_event(face_plan, event)
        if not face_event:
            continue

        face = face_event.get("face", {}) or {}
        face_width = _float(face.get("face_width"), 0.0)
        movement = _float(
            face.get("movement_score", face_event.get("movement_score", 0.0))
        )

        # The existing intelligence already selected the event. The director's
        # job is to translate it into an editing instruction.
        # Very tiny detections are not good takeover candidates.
        if face_width < 55:
            continue

        if face_width >= 130:
            framing = "extreme"
        elif face_width >= 80:
            framing = "close"
        else:
            framing = "medium"

        intensity = _float(event.get("peak_intensity", event.get("intensity", 0.0)))
        reason_parts = ["high-energy streamer moment"]
        if intensity >= 0.65:
            reason_parts.append("strong intensity spike")
        if movement >= 0.20:
            reason_parts.append("visible face movement")

        actions.append(
            DirectorAction(
                start=max(start, clip_start),
                end=min(end, clip_end),
                action="reaction_takeover",
                reason="; ".join(reason_parts),
                priority=100,
                params={
                    "framing": framing,
                    "face_width": face_width,
                    "movement": movement,
                    "source_start": start,
                    "source_end": end,
                },
            )
        )

    actions.sort(key=lambda a: (a.start, -a.priority))

    return DirectorPlan(
        version=1,
        genre="streamer",
        confidence=0.92 if actions else 0.70,
        actions=actions,
        metadata={
            "camera_language": "event_based",
            "continuous_zoom": False,
            "base_layout": "webcam_top_gameplay_bottom",
        },
    )
