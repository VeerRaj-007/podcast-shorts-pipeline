import json
import math
import os
import subprocess
from typing import Dict, List, Optional, Tuple

from captions import create_clip_ass
from layout import load_or_analyze
from directors import route_director, DirectorPlan

OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920
WEBCAM_PANEL_HEIGHT = 760
GAMEPLAY_PANEL_HEIGHT = 1160
CAMERA_PLAN_FILENAME = "camera_plan.json"
FACE_PLAN_FILENAME = "face_plan.json"

# Reaction-shot tuning.
MEDIUM_FACE_FRACTION = 0.24
CLOSE_FACE_FRACTION = 0.30
EXTREME_FACE_FRACTION = 0.36
MAX_MOVEMENT_WIDENING = 0.18
MIN_FOREGROUND_WIDTH = 760
MAX_FOREGROUND_WIDTH = 900
MIN_CROP_WIDTH = 220
# Keep the reaction crop safely inside the webcam panel. The source layout has a
# hard boundary immediately above gameplay, so never sample the bottom edge.
WEBCAM_SAFE_HEIGHT_FRACTION = 0.92


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def even(value: float) -> int:
    return max(2, int(math.floor(float(value) / 2.0) * 2))


def run_ffmpeg(command: List[str]):
    print("      Rendering...")
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        print("\nFFmpeg error:\n")
        print(result.stderr)
        raise RuntimeError("FFmpeg failed")


def ffmpeg_subtitle_path(path: str) -> str:
    # This path is inserted into an FFmpeg filter script, not a shell command.
    path = os.path.relpath(os.path.abspath(path), os.getcwd()).replace("\\", "/")
    return path.replace("'", r"\'").replace(":", r"\:")


def probe_video_dimensions(video_path: str) -> Tuple[int, int]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=s=x:p=0",
            video_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("ffprobe failed:\n" + result.stderr)
    width, height = result.stdout.strip().split("x")
    return int(width), int(height)


def load_camera_plan(tmp_dir: str) -> Dict:
    path = os.path.join(tmp_dir, CAMERA_PLAN_FILENAME)
    if not os.path.exists(path):
        print("      Camera plan: NOT FOUND")
        return {"events": []}

    try:
        with open(path, "r", encoding="utf-8") as f:
            plan = json.load(f)
    except Exception as exc:
        print(f"      Camera plan: FAILED TO LOAD ({exc})")
        return {"events": []}

    events = plan.get("events", [])
    if not isinstance(events, list):
        events = []

    print(f"      Camera plan: LOADED ({len(events)} events)")
    return {**plan, "events": events}


def load_face_plan(tmp_dir: str) -> Optional[Dict]:
    path = os.path.join(tmp_dir, FACE_PLAN_FILENAME)
    if not os.path.exists(path):
        print("      Face plan: NOT FOUND")
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            plan = json.load(f)
    except Exception as exc:
        print(f"      Face plan: FAILED TO LOAD ({exc})")
        return None

    events = plan.get("events", [])
    if not isinstance(events, list):
        events = []

    print(f"      Face plan: LOADED ({len(events)} events)")
    return {**plan, "events": events}


def get_overlapping_events(
    events: List[Dict], clip_start: float, clip_end: float
) -> List[Dict]:
    result = []
    for event in events:
        try:
            start = float(event["start"])
            end = float(event["end"])
        except (KeyError, TypeError, ValueError):
            continue

        if end > clip_start and start < clip_end:
            result.append(event)

    return sorted(result, key=lambda e: float(e["start"]))


def match_face_event(face_plan: Optional[Dict], camera_event: Dict) -> Optional[Dict]:
    if not face_plan:
        return None

    try:
        camera_start = float(camera_event["start"])
        camera_end = float(camera_event["end"])
    except (KeyError, TypeError, ValueError):
        return None

    best = None
    best_overlap = 0.0

    for face_event in face_plan.get("events", []):
        try:
            face_start = float(face_event["start"])
            face_end = float(face_event["end"])
        except (KeyError, TypeError, ValueError):
            continue

        overlap = min(camera_end, face_end) - max(camera_start, face_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best = face_event

    return best


def extract_movement(face_event: Dict) -> float:
    face = face_event.get("face", {})

    candidates = (
        face.get("movement_score"),
        face_event.get("movement_score"),
        face.get("movement"),
        face_event.get("movement"),
        face.get("motion"),
        face_event.get("motion"),
    )

    for value in candidates:
        try:
            return clamp(float(value), 0.0, 1.0)
        except (TypeError, ValueError):
            pass

    return 0.0


def choose_shot(face_width: float, movement_score: float) -> Tuple[str, float, float]:
    movement = clamp(movement_score, 0.0, 1.0)

    if face_width < 80:
        shot_type = "MEDIUM"
        fraction = MEDIUM_FACE_FRACTION
    elif face_width < 130:
        shot_type = "CLOSE"
        fraction = CLOSE_FACE_FRACTION
    else:
        shot_type = "EXTREME"
        fraction = EXTREME_FACE_FRACTION

    fraction *= 1.0 - MAX_MOVEMENT_WIDENING * movement
    return shot_type, clamp(fraction, 0.20, 0.50), movement


def build_face_crop_plan_from_dimensions(
    source_width: int,
    source_height: int,
    webcam_region: Dict,
    face_event: Dict,
) -> Dict:
    face = face_event.get("face", {})

    webcam_x = float(webcam_region["x"]) * source_width
    webcam_y = float(webcam_region["y"]) * source_height
    webcam_w = float(webcam_region["width"]) * source_width
    webcam_h = float(webcam_region["height"]) * source_height

    # Do not let the reaction crop touch the webcam/gameplay seam. Even a few
    # source pixels of gameplay become a large visible strip after 9:16 scaling.
    safe_webcam_h = max(2.0, webcam_h * WEBCAM_SAFE_HEIGHT_FRACTION)

    try:
        face_x = float(face["center_x"])
        face_y = float(face["center_y"])
        face_width = max(1.0, float(face["face_width"]))
    except (KeyError, TypeError, ValueError):
        face_x = webcam_x + webcam_w / 2.0
        face_y = webcam_y + webcam_h * 0.40
        face_width = 100.0

    local_x = face_x - webcam_x
    local_y = face_y - webcam_y

    shot_type, _, movement = choose_shot(
        face_width,
        extract_movement(face_event),
    )

    # 4:5 is deliberately wider than the old 9:16 crop.
    aspect = 9.0 / 16.0
    max_crop_width = min(webcam_w, safe_webcam_h * aspect)

    if shot_type == "MEDIUM":
        face_fraction = MEDIUM_FACE_FRACTION
    elif shot_type == "CLOSE":
        face_fraction = CLOSE_FACE_FRACTION
    else:
        face_fraction = EXTREME_FACE_FRACTION

    crop_width = face_width / max(face_fraction, 0.18)
    crop_width *= 1.0 + 0.10 * movement
    crop_width = clamp(crop_width, MIN_CROP_WIDTH, max_crop_width)
    crop_height = crop_width / aspect

    crop_x_local = clamp(
        local_x - crop_width / 2.0,
        0.0,
        max(0.0, webcam_w - crop_width),
    )
    crop_y_local = clamp(
        local_y - crop_height * 0.36,
        0.0,
        max(0.0, safe_webcam_h - crop_height),
    )

    # Reaction takeovers are full-frame, matching the original strong reaction
    # shot behavior. The face crop itself determines how large the face appears.
    foreground_width = OUTPUT_WIDTH
    foreground_height = OUTPUT_HEIGHT

    return {
        "shot_type": shot_type,
        "movement_score": movement,
        "face_width": face_width,
        "face_x": face_x,
        "face_y": face_y,
        "crop_x": even(webcam_x + crop_x_local),
        "crop_y": even(webcam_y + crop_y_local),
        "crop_width": even(crop_width),
        "crop_height": even(crop_height),
        "safe_webcam_height": even(safe_webcam_h),
        "foreground_width": foreground_width,
        "foreground_height": foreground_height,
        "foreground_x": (OUTPUT_WIDTH - foreground_width) // 2,
        "foreground_y": (OUTPUT_HEIGHT - foreground_height) // 2,
    }


def build_normal_filter() -> str:
    return (
        "[0:v]"
        "scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,"
        "crop=1080:1920:(in_w-out_w)/2:(in_h-out_h)/2"
        "[base]"
    )


def build_streamer_base_filter(layout: Dict) -> str:
    webcam = layout["webcam_region"]
    gameplay = layout["gameplay_region"]

    webcam_filter = (
        "[0:v]"
        f"crop=iw*{webcam['width']}:ih*{webcam['height']}:"
        f"iw*{webcam['x']}:ih*{webcam['y']},"
        "scale=1080:760:force_original_aspect_ratio=increase:flags=lanczos,"
        "crop=1080:760:(in_w-out_w)/2:(in_h-out_h)/2"
        "[webcam]"
    )

    gameplay_filter = (
        "[0:v]"
        f"crop=iw*{gameplay['width']}:ih*{gameplay['height']}:"
        f"iw*{gameplay['x']}:ih*{gameplay['y']},"
        "scale=1080:1160:force_original_aspect_ratio=increase:flags=lanczos,"
        "crop=1080:1160:(in_w-out_w)/2:(in_h-out_h)/2"
        "[gameplay]"
    )

    return (
        webcam_filter
        + ";"
        + gameplay_filter
        + ";[webcam][gameplay]vstack=inputs=2[base]"
    )


def build_reaction_filter_from_director(
    base_label: str,
    director_plan: DirectorPlan,
    face_plan: Optional[Dict],
    layout: Dict,
    source_width: int,
    source_height: int,
    clip_start: float,
    clip_end: float,
) -> Tuple[str, List[Dict]]:
    """Execute reaction_takeover actions from a DirectorPlan.

    The renderer deliberately does not decide *why* a reaction shot happens.
    It only executes actions already chosen by the director.
    """
    filters: List[str] = []
    debug: List[Dict] = []
    current = base_label
    clip_duration = max(0.0, clip_end - clip_start)

    webcam = layout["webcam_region"]
    # Match the safe boundary used by build_face_crop_plan_from_dimensions so
    # neither the blurred background nor the face foreground can sample gameplay.
    safe_h_fraction = WEBCAM_SAFE_HEIGHT_FRACTION
    webcam_crop = (
        f"crop=iw*{webcam['width']}:ih*{webcam['height']}*{safe_h_fraction}:"
        f"iw*{webcam['x']}:ih*{webcam['y']}"
    )

    for action in director_plan.actions:
        if action.action != "reaction_takeover":
            continue

        event_start = float(action.start)
        event_end = float(action.end)
        if event_end <= clip_start or event_start >= clip_end:
            continue

        # Director actions use source-video timestamps. Convert them to local
        # clip time because cut_clip seeks to clip_start before filtering.
        local_start = clamp(event_start - clip_start, 0.0, clip_duration)
        local_end = clamp(event_end - clip_start, 0.0, clip_duration)
        if local_end <= local_start:
            continue

        # The director carries the source event window used to find the matching
        # face observation. Keeping this association explicit avoids guessing.
        camera_event = {
            "start": action.params.get("source_start", event_start),
            "end": action.params.get("source_end", event_end),
        }
        face_event = match_face_event(face_plan, camera_event)
        if face_event is None:
            continue

        plan = build_face_crop_plan_from_dimensions(
            source_width,
            source_height,
            webcam,
            face_event,
        )

        idx = len(debug)
        bg = f"reaction_bg_{idx}"
        fg = f"reaction_fg_{idx}"
        reaction = f"reaction_{idx}"
        timed_reaction = f"reaction_timed_{idx}"
        next_label = f"camera_{idx}"

        # Background: same source-time window, then blurred to fill 9:16.
        filters.append(
            "[0:v]"
            f"{webcam_crop},"
            "scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,"
            "crop=1080:1920:(in_w-out_w)/2:(in_h-out_h)/2,"
            f"trim=start={local_start:.6f}:end={local_end:.6f},"
            "setpts=PTS-STARTPTS,"
            "gblur=sigma=22:steps=2"
            f"[{bg}]"
        )

        # Foreground: face-aware crop from the exact same source-time window.
        # This is the large face takeover the streamer reference calls for.
        filters.append(
            "[0:v]"
            f"crop={plan['crop_width']}:{plan['crop_height']}:"
            f"{plan['crop_x']}:{plan['crop_y']},"
            "scale=1080:1920:flags=lanczos,"
            f"trim=start={local_start:.6f}:end={local_end:.6f},"
            "setpts=PTS-STARTPTS,"
            "unsharp=5:5:0.35:5:5:0.0"
            f"[{fg}]"
        )

        filters.append(
            f"[{bg}][{fg}]"
            "overlay=0:0:eof_action=pass"
            f"[{reaction}]"
        )

        filters.append(
            f"[{reaction}]"
            f"setpts=PTS-STARTPTS+{local_start:.6f}/TB"
            f"[{timed_reaction}]"
        )

        filters.append(
            f"[{current}][{timed_reaction}]"
            "overlay=0:0:eof_action=pass"
            f"[{next_label}]"
        )

        debug.append({
            "start": event_start,
            "end": event_end,
            "local_start": local_start,
            "local_end": local_end,
            "action": action.action,
            "reason": action.reason,
            **plan,
        })
        current = next_label

    if not debug:
        return "", []

    filters.append(f"[{current}]null[camera]")
    return ";".join(filters), debug

def cut_clip(
    video_path: str,
    clip: Dict,
    output_path: str,
    tmp_dir: str,
    layout: str = "auto",
):
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    start = float(clip["start"])
    end = float(clip["end"])
    duration = end - start

    if duration <= 0:
        raise ValueError(f"Invalid clip duration: {start} -> {end}")

    print(f"      Clip: {start:.1f}s -> {end:.1f}s ({duration:.1f}s)")

    source_width, source_height = probe_video_dimensions(video_path)
    print(f"      Source: {source_width}x{source_height}")

    camera_plan = load_camera_plan(tmp_dir)
    face_plan = load_face_plan(tmp_dir)

    camera_events = get_overlapping_events(
        camera_plan.get("events", []),
        start,
        end,
    )
    print(f"      Camera events in clip: {len(camera_events)}")

    layout_path = os.path.join(tmp_dir, "layout.json")

    if layout == "auto":
        layout_data = load_or_analyze(video_path, layout_path)
    elif layout == "webcam_split":
        layout_data = {
            "type": "gaming_stream",
            "webcam": True,
            "webcam_region": {
                "x": 0.700,
                "y": 0.000,
                "width": 0.300,
                "height": 0.405,
            },
            "gameplay_region": {
                "x": 0.000,
                "y": 0.000,
                "width": 0.695,
                "height": 1.000,
            },
        }
    else:
        layout_data = {"type": "normal", "webcam": False}

    if layout_data.get("webcam"):
        print("      Layout: STREAMER + GAMEPLAY")

        base_filter = build_streamer_base_filter(layout_data)

        director_plan = route_director(
            clip_start=start,
            clip_end=end,
            layout=layout_data,
            camera_plan=camera_plan,
            face_plan=face_plan,
        )

        print(
            f"      Director: {director_plan.genre.upper()} | "
            f"confidence={director_plan.confidence:.2f} | "
            f"actions={len(director_plan.actions)}"
        )

        reaction_filter, debug_events = build_reaction_filter_from_director(
            "base",
            director_plan,
            face_plan,
            layout_data,
            source_width,
            source_height,
            start,
            end,
        )

        print(f"      Smart reaction shots: {len(debug_events)}")
        for i, event in enumerate(debug_events, 1):
            print(
                f"      Smart shot #{i}: {event['shot_type']} | "
                f"{event['start']:.1f}s -> {event['end']:.1f}s | "
                f"face={event['face_width']:.1f}px | "
                f"movement={event['movement_score']:.2f} | "
                f"face_center=({event['face_x']:.1f}, {event['face_y']:.1f}) | "
                f"crop={event['crop_width']}x{event['crop_height']} | "
                f"crop_xy=({event['crop_x']}, {event['crop_y']})"
            )

        if reaction_filter:
            video_filter = base_filter + ";" + reaction_filter
            subtitle_input = "[camera]"
        else:
            # CRITICAL: never create [base]null[camera] here. That was the
            # source of the previous "basenull" filter error.
            video_filter = base_filter
            subtitle_input = "[base]"
    else:
        print("      Layout: NORMAL VERTICAL")
        video_filter = build_normal_filter()
        subtitle_input = "[base]"

    transcript_path = os.path.join(tmp_dir, "aligned_transcript.json")
    basename = os.path.splitext(os.path.basename(output_path))[0]
    ass_path = os.path.join(tmp_dir, f"{basename}.ass")

    create_clip_ass(clip, ass_path, transcript_path)
    subtitle_path = ffmpeg_subtitle_path(ass_path)

    # Keep subtitles as the final filter. This also means captions are applied
    # after every camera/reaction composition, so they stay in the same place.
    video_filter += (
        f";{subtitle_input}"
        f"subtitles=filename='{subtitle_path}'"
        "[outv]"
    )

    # Pass the complex filtergraph directly to FFmpeg.
    #
    # IMPORTANT: do this as one subprocess argument rather than using a shell.
    # Python's subprocess list form does not perform shell quoting, so commas,
    # semicolons, parentheses and the escaped subtitle path are passed through
    # intact. This is compatible with current FFmpeg 9.x, where the old
    # -filter_complex_script CLI option is no longer accepted.
    command = [
        "ffmpeg",
        "-y",
        "-ss", str(start),
        "-i", video_path,
        "-t", str(duration),
        "-filter_complex", video_filter,
        "-map", "[outv]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        output_path,
    ]

    run_ffmpeg(command)
    print(f"      Created: {output_path}")
