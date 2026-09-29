"""
Editorial A/V Resolver v1.2

Purpose
-------
Convert an editorial sequence into a render-safe timeline.

IMPORTANT ARCHITECTURAL RULE
----------------------------
The editorial sequence is authoritative.

For a normal/base interview action:
    editorial source_start/source_end
        ==
    video source_in/source_out
        ==
    audio source_in/source_out

The resolver MUST NOT move, expand, shrink, or otherwise reinterpret
the editorially selected source range.

For insert actions:
    video and audio may come from independent source ranges.

The renderer executes the resolved plan.
The resolver does not make editorial decisions.

Safety rules
------------
- No silent source-range modification.
- No source reuse.
- No backward source jumps.
- No looping.
- No freezing.
- No stretching.
- No fake duration.
- No empty resolved segments.
- render_safe can only be true after strict validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EPS = 1e-6
TOL = 1e-3
OVERLAP_TOL = 1e-4

INSERT_KINDS = {"insert"}

INSERT_SHOT_TYPES = {
    "insert",
    "insert_or_screen",
    "non_face",
    "graphic",
    "product",
    "object",
    "screen",
    "screen_capture",
}


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def fnum(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def norm(value: Any) -> str:
    return str(value or "").strip().lower()


def overlap(
    a0: float,
    a1: float,
    b0: float,
    b1: float,
) -> float:
    return max(
        0.0,
        min(a1, b1) - max(a0, b0),
    )


def valid_interval(start: float, end: float) -> bool:
    return end > start + EPS


def is_insert_shot(shot: dict[str, Any]) -> bool:
    return norm(
        shot.get("shot_type")
    ) in INSERT_SHOT_TYPES


# ---------------------------------------------------------------------------
# Editorial source timing
# ---------------------------------------------------------------------------

def action_source_start(action: dict[str, Any]) -> float:
    """
    v9.x editorial plans use source_start.
    Older plans may use start.
    """
    if "source_start" in action:
        return fnum(
            action.get("source_start")
        )

    return fnum(
        action.get("start")
    )


def action_source_end(action: dict[str, Any]) -> float:
    """
    v9.x editorial plans use source_end.
    Older plans may use end.
    """
    if "source_end" in action:
        return fnum(
            action.get("source_end")
        )

    return fnum(
        action.get("end")
    )


def action_duration(action: dict[str, Any]) -> float:
    return (
        action_source_end(action)
        - action_source_start(action)
    )


# ---------------------------------------------------------------------------
# Shot lookup
# ---------------------------------------------------------------------------

def find_shot_by_id(
    shots: list[dict[str, Any]],
    shot_id: str,
) -> dict[str, Any] | None:

    wanted = str(
        shot_id or ""
    )

    if not wanted:
        return None

    for shot in shots:
        if str(
            shot.get("shot_id") or ""
        ) == wanted:
            return shot

    return None


# ---------------------------------------------------------------------------
# Exact base-action resolution
# ---------------------------------------------------------------------------

def resolve_base_action(
    action: dict[str, Any],
    shots: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]] | None,
    list[dict[str, Any]] | None,
    str | None,
]:

    source_start = action_source_start(action)
    source_end = action_source_end(action)

    if not valid_interval(
        source_start,
        source_end,
    ):
        return (
            None,
            None,
            "Editorial action has non-positive duration.",
        )

    shot_id = str(
        action.get("shot_id") or ""
    )

    shot = find_shot_by_id(
        shots,
        shot_id,
    )

    if shot is None:
        return (
            None,
            None,
            f"Referenced shot_id not found: {shot_id}",
        )

    shot_start = fnum(
        shot.get("start")
    )

    shot_end = fnum(
        shot.get("end")
    )

    # IMPORTANT:
    # We do not change source_start/source_end.
    # We only verify that the editorial range belongs to the shot.
    if source_start < shot_start - TOL:
        return (
            None,
            None,
            (
                "Editorial source_start lies before the "
                f"referenced shot: "
                f"{source_start:.6f} < {shot_start:.6f}"
            ),
        )

    if source_end > shot_end + TOL:
        return (
            None,
            None,
            (
                "Editorial source_end lies after the "
                f"referenced shot: "
                f"{source_end:.6f} > {shot_end:.6f}"
            ),
        )

    duration = source_end - source_start

    segment = {
        "shot_id": shot_id,
        "shot_type": shot.get("shot_type"),
        "source_in": round(
            source_start,
            6,
        ),
        "source_out": round(
            source_end,
            6,
        ),
        "duration": round(
            duration,
            6,
        ),
    }

    # Base interview action:
    # audio MUST match the exact selected video range.
    video_segments = [dict(segment)]
    audio_segments = [dict(segment)]

    return (
        video_segments,
        audio_segments,
        None,
    )


# ---------------------------------------------------------------------------
# Insert resolution
# ---------------------------------------------------------------------------

def subtract_used(
    start: float,
    end: float,
    used: list[tuple[float, float]],
) -> list[tuple[float, float]]:

    pieces = [
        (start, end)
    ]

    for used_start, used_end in sorted(
        used
    ):

        if (
            used_end <= start + OVERLAP_TOL
            or used_start >= end - OVERLAP_TOL
        ):
            continue

        next_pieces: list[
            tuple[float, float]
        ] = []

        for p0, p1 in pieces:

            if (
                used_end <= p0 + OVERLAP_TOL
                or used_start >= p1 - OVERLAP_TOL
            ):
                next_pieces.append(
                    (p0, p1)
                )
                continue

            if used_start > p0 + OVERLAP_TOL:
                next_pieces.append(
                    (
                        p0,
                        min(
                            used_start,
                            p1,
                        ),
                    )
                )

            if used_end < p1 - OVERLAP_TOL:
                next_pieces.append(
                    (
                        max(
                            used_end,
                            p0,
                        ),
                        p1,
                    )
                )

        pieces = [
            (a, b)
            for a, b in next_pieces
            if b - a > OVERLAP_TOL
        ]

        if not pieces:
            break

    return pieces


def preferred_shot(
    action: dict[str, Any],
    shots: list[dict[str, Any]],
) -> dict[str, Any] | None:

    wanted = str(
        action.get("shot_id") or ""
    )

    return find_shot_by_id(
        shots,
        wanted,
    )


def resolve_insert_video(
    action: dict[str, Any],
    shots: list[dict[str, Any]],
    used_video: list[tuple[float, float]],
) -> tuple[
    list[dict[str, Any]] | None,
    str | None,
]:

    need = action_duration(action)

    if need <= EPS:
        return (
            None,
            "Insert action has non-positive duration.",
        )

    planned = preferred_shot(
        action,
        shots,
    )

    candidates: list[
        dict[str, Any]
    ] = []

    if planned is not None:
        candidates.append(
            planned
        )

    candidates.extend(
        shot
        for shot in shots
        if shot is not planned
        and is_insert_shot(shot)
    )

    remaining = need

    result: list[
        dict[str, Any]
    ] = []

    local_used: list[
        tuple[float, float]
    ] = []

    for shot in candidates:

        if shot is None:
            continue

        shot_start = fnum(
            shot.get("start")
        )
        shot_end = fnum(
            shot.get("end")
        )

        if not valid_interval(
            shot_start,
            shot_end,
        ):
            continue

        pieces = subtract_used(
            shot_start,
            shot_end,
            used_video + local_used,
        )

        for p0, p1 in pieces:

            if remaining <= EPS:
                break

            take = min(
                remaining,
                p1 - p0,
            )

            if take <= EPS:
                continue

            result.append(
                {
                    "shot_id": str(
                        shot.get("shot_id")
                        or ""
                    ),
                    "shot_type": shot.get(
                        "shot_type"
                    ),
                    "source_in": round(
                        p0,
                        6,
                    ),
                    "source_out": round(
                        p0 + take,
                        6,
                    ),
                    "duration": round(
                        take,
                        6,
                    ),
                }
            )

            local_used.append(
                (
                    p0,
                    p0 + take,
                )
            )

            remaining -= take

        if remaining <= EPS:
            break

    if remaining > TOL:
        return (
            None,
            (
                "Unable to resolve insert video without "
                "source reuse."
            ),
        )

    return (
        result,
        None,
    )


def resolve_insert_audio(
    action: dict[str, Any],
    source_shots: list[dict[str, Any]],
    used_audio: list[tuple[float, float]],
    audio_cursor: float,
) -> tuple[
    list[dict[str, Any]] | None,
    float,
    str | None,
]:

    need = action_duration(action)

    if need <= EPS:
        return (
            None,
            audio_cursor,
            "Insert action has non-positive duration.",
        )

    remaining = need

    result: list[
        dict[str, Any]
    ] = []

    local_used: list[
        tuple[float, float]
    ] = []

    for shot in source_shots:

        shot_start = fnum(
            shot.get("start")
        )
        shot_end = fnum(
            shot.get("end")
        )

        if not valid_interval(
            shot_start,
            shot_end,
        ):
            continue

        if (
            shot_end
            <= audio_cursor
            + OVERLAP_TOL
        ):
            continue

        start = max(
            shot_start,
            audio_cursor,
        )

        if start >= shot_end - OVERLAP_TOL:
            continue

        pieces = subtract_used(
            start,
            shot_end,
            used_audio + local_used,
        )

        for p0, p1 in pieces:

            if remaining <= EPS:
                break

            take = min(
                remaining,
                p1 - p0,
            )

            if take <= EPS:
                continue

            result.append(
                {
                    "shot_id": str(
                        shot.get("shot_id")
                        or ""
                    ),
                    "shot_type": shot.get(
                        "shot_type"
                    ),
                    "source_in": round(
                        p0,
                        6,
                    ),
                    "source_out": round(
                        p0 + take,
                        6,
                    ),
                    "duration": round(
                        take,
                        6,
                    ),
                }
            )

            local_used.append(
                (
                    p0,
                    p0 + take,
                )
            )

            audio_cursor = (
                p0 + take
            )

            remaining -= take

        if remaining <= EPS:
            break

    if remaining > TOL:
        return (
            None,
            audio_cursor,
            (
                "Unable to resolve insert audio "
                "without reuse/backward movement."
            ),
        )

    return (
        result,
        audio_cursor,
        None,
    )


# ---------------------------------------------------------------------------
# Render segments
# ---------------------------------------------------------------------------

def build_render_segments(
    video_segments: list[dict[str, Any]],
    audio_segments: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    """
    Renderer v2 supports a common source range for video+audio.

    Therefore render_segments can only be emitted when the A/V ranges
    are aligned one-for-one.
    """

    if not video_segments:
        return None

    if not audio_segments:
        return None

    if (
        len(video_segments)
        != len(audio_segments)
    ):
        return None

    render_segments: list[
        dict[str, Any]
    ] = []

    output_cursor = 0.0

    for video_segment, audio_segment in zip(
        video_segments,
        audio_segments,
    ):

        video_in = fnum(
            video_segment.get(
                "source_in"
            )
        )

        video_out = fnum(
            video_segment.get(
                "source_out"
            )
        )

        audio_in = fnum(
            audio_segment.get(
                "source_in"
            )
        )

        audio_out = fnum(
            audio_segment.get(
                "source_out"
            )
        )

        video_duration = (
            video_out - video_in
        )

        audio_duration = (
            audio_out - audio_in
        )

        if not valid_interval(
            video_in,
            video_out,
        ):
            return None

        if not valid_interval(
            audio_in,
            audio_out,
        ):
            return None

        if (
            abs(
                video_in - audio_in
            )
            > TOL
            or abs(
                video_out - audio_out
            )
            > TOL
            or abs(
                video_duration
                - audio_duration
            )
            > TOL
        ):
            return None

        render_segments.append(
            {
                "output_start": round(
                    output_cursor,
                    6,
                ),
                "output_end": round(
                    output_cursor
                    + video_duration,
                    6,
                ),
                "source_in": round(
                    video_in,
                    6,
                ),
                "source_out": round(
                    video_out,
                    6,
                ),
            }
        )

        output_cursor += video_duration

    return render_segments


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_stream_source_reuse(
    segments: list[dict[str, Any]],
    stream_name: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:

    reuse: list[
        dict[str, Any]
    ] = []

    backward: list[
        dict[str, Any]
    ] = []

    ordered = sorted(
        segments,
        key=lambda segment: (
            fnum(
                segment.get(
                    "source_in"
                )
            ),
            fnum(
                segment.get(
                    "source_out"
                )
            ),
        ),
    )

    for i, current in enumerate(
        ordered
    ):

        current_start = fnum(
            current.get(
                "source_in"
            )
        )

        current_end = fnum(
            current.get(
                "source_out"
            )
        )

        if i > 0:

            previous = ordered[
                i - 1
            ]

            previous_end = fnum(
                previous.get(
                    "source_out"
                )
            )

            if (
                current_start
                < previous_end - TOL
            ):
                backward.append(
                    {
                        "stream": stream_name,
                        "previous_end": previous_end,
                        "current_start": current_start,
                    }
                )

        for j, previous in enumerate(
            ordered[:i]
        ):

            previous_start = fnum(
                previous.get(
                    "source_in"
                )
            )

            previous_end = fnum(
                previous.get(
                    "source_out"
                )
            )

            amount = overlap(
                current_start,
                current_end,
                previous_start,
                previous_end,
            )

            if amount > OVERLAP_TOL:

                reuse.append(
                    {
                        "stream": stream_name,
                        "current": {
                            "action": current.get(
                                "_action_index"
                            ),
                            "source": [
                                current_start,
                                current_end,
                            ],
                        },
                        "previous": {
                            "action": previous.get(
                                "_action_index"
                            ),
                            "source": [
                                previous_start,
                                previous_end,
                            ],
                        },
                        "overlap_seconds": round(
                            amount,
                            6,
                        ),
                    }
                )

    return reuse, backward


def validate_plan(
    actions: list[dict[str, Any]],
) -> dict[str, Any]:

    invalid: list[
        dict[str, Any]
    ] = []

    all_video: list[
        dict[str, Any]
    ] = []

    all_audio: list[
        dict[str, Any]
    ] = []

    expected_output = 0.0

    for action_index, action in enumerate(
        actions
    ):

        output_start = fnum(
            action.get("start")
        )

        output_end = fnum(
            action.get("end")
        )

        duration = (
            output_end
            - output_start
        )

        if abs(
            output_start
            - expected_output
        ) > TOL:

            invalid.append(
                {
                    "action": action_index,
                    "reason": "output_gap_or_overlap",
                    "expected": round(
                        expected_output,
                        6,
                    ),
                    "actual": round(
                        output_start,
                        6,
                    ),
                }
            )

        if duration <= EPS:

            invalid.append(
                {
                    "action": action_index,
                    "reason": "non_positive_output_duration",
                }
            )

        source_start = action_source_start(
            action
        )

        source_end = action_source_end(
            action
        )

        source_duration = (
            source_end
            - source_start
        )

        if (
            abs(
                source_duration
                - duration
            )
            > TOL
        ):

            invalid.append(
                {
                    "action": action_index,
                    "reason": "editorial_duration_changed",
                    "editorial_source_duration": round(
                        source_duration,
                        6,
                    ),
                    "output_duration": round(
                        duration,
                        6,
                    ),
                }
            )

        video_segments = action.get(
            "video_segments",
            [],
        )

        audio_segments = action.get(
            "audio_segments",
            [],
        )

        render_segments = action.get(
            "render_segments",
            [],
        )

        if not video_segments:

            invalid.append(
                {
                    "action": action_index,
                    "reason": "empty_video_segments",
                }
            )

        if not audio_segments:

            invalid.append(
                {
                    "action": action_index,
                    "reason": "empty_audio_segments",
                }
            )

        if not render_segments:

            invalid.append(
                {
                    "action": action_index,
                    "reason": "empty_render_segments",
                }
            )

        video_duration = sum(
            fnum(
                seg.get(
                    "source_out"
                )
            )
            - fnum(
                seg.get(
                    "source_in"
                )
            )
            for seg in video_segments
        )

        audio_duration = sum(
            fnum(
                seg.get(
                    "source_out"
                )
            )
            - fnum(
                seg.get(
                    "source_in"
                )
            )
            for seg in audio_segments
        )

        render_duration = sum(
            fnum(
                seg.get(
                    "output_end"
                )
            )
            - fnum(
                seg.get(
                    "output_start"
                )
            )
            for seg in render_segments
        )

        if abs(
            video_duration
            - duration
        ) > TOL:

            invalid.append(
                {
                    "action": action_index,
                    "reason": "video_duration_mismatch",
                    "expected": round(
                        duration,
                        6,
                    ),
                    "actual": round(
                        video_duration,
                        6,
                    ),
                }
            )

        if abs(
            audio_duration
            - duration
        ) > TOL:

            invalid.append(
                {
                    "action": action_index,
                    "reason": "audio_duration_mismatch",
                    "expected": round(
                        duration,
                        6,
                    ),
                    "actual": round(
                        audio_duration,
                        6,
                    ),
                }
            )

        if abs(
            render_duration
            - duration
        ) > TOL:

            invalid.append(
                {
                    "action": action_index,
                    "reason": "render_duration_mismatch",
                    "expected": round(
                        duration,
                        6,
                    ),
                    "actual": round(
                        render_duration,
                        6,
                    ),
                }
            )

        for segment in video_segments:
            all_video.append(
                {
                    "_action_index": action_index,
                    **segment,
                }
            )

        for segment in audio_segments:
            all_audio.append(
                {
                    "_action_index": action_index,
                    **segment,
                }
            )

        expected_output = output_end

    video_reuse, video_backward = (
        validate_stream_source_reuse(
            all_video,
            "video",
        )
    )

    audio_reuse, audio_backward = (
        validate_stream_source_reuse(
            all_audio,
            "audio",
        )
    )

    return {
        "invalid": invalid,
        "video_source_reuse": video_reuse,
        "audio_source_reuse": audio_reuse,
        "video_backward_source_jumps": video_backward,
        "audio_backward_source_jumps": audio_backward,
        "render_safe_candidate": (
            not invalid
            and not video_reuse
            and not audio_reuse
            and not video_backward
            and not audio_backward
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Resolve editorial actions into exact, "
            "render-safe A/V source ranges."
        )
    )

    parser.add_argument(
        "--sequence",
        required=True,
    )

    parser.add_argument(
        "--shots",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    sequence = load_json(
        args.sequence
    )

    shot_data = load_json(
        args.shots
    )

    actions = sequence.get(
        "actions",
        sequence.get(
            "timeline_actions",
            [],
        ),
    )

    if not isinstance(
        actions,
        list,
    ):
        actions = []

    raw_shots = (
        shot_data.get(
            "shots",
            [],
        )
        if isinstance(
            shot_data,
            dict,
        )
        else shot_data
    )

    if not isinstance(
        raw_shots,
        list,
    ):
        raw_shots = []

    shots = sorted(
        [
            shot
            for shot in raw_shots
            if (
                isinstance(
                    shot,
                    dict,
                )
                and valid_interval(
                    fnum(
                        shot.get(
                            "start"
                        )
                    ),
                    fnum(
                        shot.get(
                            "end"
                        )
                    ),
                )
            )
        ],
        key=lambda shot: (
            fnum(
                shot.get(
                    "start"
                )
            ),
            fnum(
                shot.get(
                    "end"
                )
            ),
        ),
    )

    # Audio source fallback:
    # if the detector incorrectly classifies all interview shots as
    # insert_or_screen, they are still valid source audio.
    non_insert_shots = [
        shot
        for shot in shots
        if not is_insert_shot(
            shot
        )
    ]

    audio_source_shots = (
        non_insert_shots
        if non_insert_shots
        else shots
    )

    sorted_actions = sorted(
        actions,
        key=lambda action: (
            action_source_start(
                action
            ),
            action_source_end(
                action
            ),
        ),
    )

    resolved: list[
        dict[str, Any]
    ] = []

    unresolved: list[
        dict[str, Any]
    ] = []

    used_video: list[
        tuple[float, float]
    ] = []

    used_audio: list[
        tuple[float, float]
    ] = []

    output_cursor = 0.0

    audio_cursor = 0.0

    # -----------------------------------------------------------------------
    # Resolve each editorial action
    # -----------------------------------------------------------------------

    for action_index, action in enumerate(
        sorted_actions
    ):

        kind = norm(
            action.get(
                "kind",
                "base",
            )
        )

        source_start = action_source_start(
            action
        )

        source_end = action_source_end(
            action
        )

        duration = (
            source_end
            - source_start
        )

        # ---------------------------------------------------------------
        # Base action
        # ---------------------------------------------------------------

        if kind not in INSERT_KINDS:

            video_segments, audio_segments, error = (
                resolve_base_action(
                    action,
                    shots,
                )
            )

            if error is not None:

                unresolved.append(
                    {
                        "action_index": action_index,
                        "shot_id": action.get(
                            "shot_id"
                        ),
                        "kind": kind,
                        "source_start": source_start,
                        "source_end": source_end,
                        "reason": error,
                    }
                )

                continue

        # ---------------------------------------------------------------
        # Insert action
        # ---------------------------------------------------------------

        else:

            video_segments, video_error = (
                resolve_insert_video(
                    action,
                    shots,
                    used_video,
                )
            )

            if video_segments is None:

                unresolved.append(
                    {
                        "action_index": action_index,
                        "shot_id": action.get(
                            "shot_id"
                        ),
                        "kind": kind,
                        "source_start": source_start,
                        "source_end": source_end,
                        "reason": video_error
                        or "Insert video resolution failed.",
                    }
                )

                continue

            (
                audio_segments,
                audio_cursor,
                audio_error,
            ) = resolve_insert_audio(
                action,
                audio_source_shots,
                used_audio,
                audio_cursor,
            )

            if audio_segments is None:

                unresolved.append(
                    {
                        "action_index": action_index,
                        "shot_id": action.get(
                            "shot_id"
                        ),
                        "kind": kind,
                        "source_start": source_start,
                        "source_end": source_end,
                        "reason": audio_error
                        or "Insert audio resolution failed.",
                    }
                )

                continue

        # ---------------------------------------------------------------
        # Strict duration checks
        # ---------------------------------------------------------------

        if duration <= EPS:

            unresolved.append(
                {
                    "action_index": action_index,
                    "shot_id": action.get(
                        "shot_id"
                    ),
                    "kind": kind,
                    "source_start": source_start,
                    "source_end": source_end,
                    "reason": (
                        "Editorial action has "
                        "non-positive duration."
                    ),
                }
            )

            continue

        video_duration = sum(
            fnum(
                seg.get(
                    "source_out"
                )
            )
            - fnum(
                seg.get(
                    "source_in"
                )
            )
            for seg in (
                video_segments
                or []
            )
        )

        audio_duration = sum(
            fnum(
                seg.get(
                    "source_out"
                )
            )
            - fnum(
                seg.get(
                    "source_in"
                )
            )
            for seg in (
                audio_segments
                or []
            )
        )

        if (
            abs(
                video_duration
                - duration
            )
            > TOL
        ):

            unresolved.append(
                {
                    "action_index": action_index,
                    "shot_id": action.get(
                        "shot_id"
                    ),
                    "kind": kind,
                    "source_start": source_start,
                    "source_end": source_end,
                    "reason": "Video duration does not match editorial duration.",
                    "expected": duration,
                    "actual": video_duration,
                }
            )

            continue

        if (
            abs(
                audio_duration
                - duration
            )
            > TOL
        ):

            unresolved.append(
                {
                    "action_index": action_index,
                    "shot_id": action.get(
                        "shot_id"
                    ),
                    "kind": kind,
                    "source_start": source_start,
                    "source_end": source_end,
                    "reason": "Audio duration does not match editorial duration.",
                    "expected": duration,
                    "actual": audio_duration,
                }
            )

            continue

        # ---------------------------------------------------------------
        # Renderer v2 compatibility
        # ---------------------------------------------------------------

        render_segments = build_render_segments(
            video_segments,
            audio_segments,
        )

        if render_segments is None:

            unresolved.append(
                {
                    "action_index": action_index,
                    "shot_id": action.get(
                        "shot_id"
                    ),
                    "kind": kind,
                    "source_start": source_start,
                    "source_end": source_end,
                    "reason": (
                        "Resolved A/V streams are not aligned "
                        "enough for renderer v2."
                    ),
                }
            )

            continue

        # ---------------------------------------------------------------
        # Build final action
        # ---------------------------------------------------------------

        output_action = dict(
            action
        )

        # Editorial source timing is preserved EXACTLY.
        output_action[
            "source_start"
        ] = round(
            source_start,
            6,
        )

        output_action[
            "source_end"
        ] = round(
            source_end,
            6,
        )

        # Output timeline timing is generated here.
        output_action[
            "start"
        ] = round(
            output_cursor,
            6,
        )

        output_action[
            "end"
        ] = round(
            output_cursor
            + duration,
            6,
        )

        output_action[
            "duration"
        ] = round(
            duration,
            6,
        )

        output_action[
            "video_segments"
        ] = video_segments

        output_action[
            "audio_segments"
        ] = audio_segments

        output_action[
            "render_segments"
        ] = render_segments

        output_action[
            "source_model"
        ] = "independent_av"

        output_action[
            "audio_continues_under_insert"
        ] = (
            kind in INSERT_KINDS
        )

        resolved.append(
            output_action
        )

        # Reserve source intervals.
        for segment in video_segments:
            used_video.append(
                (
                    fnum(
                        segment.get(
                            "source_in"
                        )
                    ),
                    fnum(
                        segment.get(
                            "source_out"
                        )
                    ),
                )
            )

        for segment in audio_segments:
            used_audio.append(
                (
                    fnum(
                        segment.get(
                            "source_in"
                        )
                    ),
                    fnum(
                        segment.get(
                            "source_out"
                        )
                    ),
                )
            )

        output_cursor += duration

    # -----------------------------------------------------------------------
    # IMPORTANT:
    #
    # If an action was unresolved, the plan is NOT render safe.
    # We do not silently drop it and call the remaining actions safe.
    # -----------------------------------------------------------------------

    validation = validate_plan(
        resolved
    )

    render_safe = (
        len(unresolved) == 0
        and validation[
            "render_safe_candidate"
        ]
    )

    output = {
        "version": "1.2",
        "mode": "independent_av_resolver",
        "source_sequence": args.sequence,
        "video_start": sequence.get(
            "video_start",
            0.0,
        ),
        "video_end": round(
            output_cursor,
            6,
        ),
        "input_actions": len(
            actions
        ),
        "resolved_actions": len(
            resolved
        ),
        "unresolved_actions": len(
            unresolved
        ),
        "actions": resolved,
        "unresolved": unresolved,
        "validation": validation,
        "render_safe": render_safe,
        "policy": {
            "editorial_ranges_are_authoritative": True,
            "base_video_uses_exact_editorial_range": True,
            "base_audio_uses_exact_editorial_range": True,
            "insert_video_may_be_independent": True,
            "insert_audio_may_be_independent": True,
            "audio_chronology_monotonic": True,
            "video_chronology_monotonic": True,
            "audio_source_reuse": "forbidden",
            "video_source_reuse": "forbidden",
            "freeze": False,
            "loop": False,
            "stretch": False,
        },
    }

    save_json(
        args.output,
        output,
    )

    print(
        json.dumps(
            {
                "output": args.output,
                "input_actions": len(
                    actions
                ),
                "resolved_actions": len(
                    resolved
                ),
                "unresolved_actions": len(
                    unresolved
                ),
                "video_reuse": len(
                    validation[
                        "video_source_reuse"
                    ]
                ),
                "audio_reuse": len(
                    validation[
                        "audio_source_reuse"
                    ]
                ),
                "video_backward": len(
                    validation[
                        "video_backward_source_jumps"
                    ]
                ),
                "audio_backward": len(
                    validation[
                        "audio_backward_source_jumps"
                    ]
                ),
                "invalid": len(
                    validation[
                        "invalid"
                    ]
                ),
                "render_safe": render_safe,
            },
            indent=2,
        )
    )

    if unresolved:

        print(
            "\nUNRESOLVED ACTIONS:"
        )

        print(
            json.dumps(
                unresolved,
                indent=2,
                ensure_ascii=False,
            )
        )

    if not render_safe:
        raise SystemExit(1)


if __name__ == "__main__":
    main()