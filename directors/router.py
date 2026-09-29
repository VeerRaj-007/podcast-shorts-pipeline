from __future__ import annotations

from typing import Any, Dict, Optional

from .models import DirectorPlan
from .streamer import build_streamer_plan
from .podcast import build_podcast_plan
from .universal import build_universal_plan


def _guess_genre(layout: Dict[str, Any], requested: Optional[str]) -> str:
    if requested:
        return requested.lower().strip()

    layout_type = str(layout.get("type", "")).lower()
    if layout.get("webcam") and "gaming" in layout_type:
        return "streamer"
    if "podcast" in layout_type or "interview" in layout_type:
        return "podcast"
    if layout.get("webcam") and "stream" in layout_type:
        return "streamer"
    return "universal"


def route_director(
    *,
    clip_start: float,
    clip_end: float,
    layout: Dict[str, Any],
    camera_plan: Optional[Dict[str, Any]] = None,
    face_plan: Optional[Dict[str, Any]] = None,
    transcript: Optional[Dict[str, Any]] = None,
    requested_genre: Optional[str] = None,
    speaker_tracks=None,
    face_tracks=None,
) -> DirectorPlan:
    genre = _guess_genre(layout, requested_genre)
    camera_plan = camera_plan or {"events": []}

    if genre == "streamer":
        return build_streamer_plan(
            clip_start,
            clip_end,
            camera_plan,
            face_plan,
        )

    if genre in {"podcast", "interview"}:
        plan = build_podcast_plan(
            clip_start,
            clip_end,
            transcript,
            speaker_tracks,
            face_tracks,
        )
        plan.genre = genre
        return plan

    return build_universal_plan(clip_start, clip_end)
