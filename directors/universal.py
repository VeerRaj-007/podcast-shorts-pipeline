from __future__ import annotations

from typing import Any, Dict, List, Optional

from .models import DirectorAction, DirectorPlan


def build_universal_plan(
    clip_start: float,
    clip_end: float,
    content_signals: Optional[Dict[str, Any]] = None,
) -> DirectorPlan:
    """Conservative director for genres not yet specialized."""

    return DirectorPlan(
        version=1,
        genre="universal",
        confidence=0.50,
        actions=[],
        metadata={
            "camera_language": "content_adaptive",
            "continuous_zoom": False,
            "fallback": "preserve_source_framing",
        },
    )
