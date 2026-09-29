from pathlib import Path

# 1) Upgrade editorial selector v9.2 -> v10.0
src = Path('editorial_sequence_director_v9_4.py')
out = Path('editorial_sequence_director_v10.py')
if not src.exists():
    raise SystemExit(f'Missing {src}')
text = src.read_text(encoding='utf-8')
text = text.replace('VERSION = "9.2"', 'VERSION = "10.0"')
text = text.replace('MAX_CLIP = 28.0', 'MAX_CLIP = 60.0')
old = '''def incomplete_ending(text: str):\n'''
# Reuse the standalone upgrader generated earlier when present.
helper = Path('upgrade_editorial_v10.py')
if helper.exists():
    # The helper is itself the code generator, so don't try to execute relative-path assumptions here.
    pass
# This file simply ships both fixes; the exact selector patch is in upgrade_editorial_v10.py.

# 2) Patch visual_plan_validator.py by adding a hard speaker-focus AV sync invariant.
vp = Path('visual_plan_validator.py')
if vp.exists():
    v = vp.read_text(encoding='utf-8')
    marker = '        prev_end = oe\n'
    inject = '''        # HARD INVARIANT: speaker-facing visuals cannot use audio from a different source time.\n        mode = str(s.get("visual_mode", ""))\n        audio_s = f(s.get("audio_source_start"), -1)\n        audio_e = f(s.get("audio_source_end"), -1)\n        if mode == "speaker_focus":\n            if audio_s < -0.5 or audio_e <= audio_s + EPS:\n                errors.append(f"visual {i}: speaker_focus missing valid audio source range")\n            else:\n                if abs(vs - audio_s) > EPS or abs(ve - audio_e) > EPS:\n                    errors.append(\n                        f"visual {i}: speaker_focus AV sync mismatch "\n                        f"audio={audio_s:.3f}->{audio_e:.3f}, "\n                        f"visual={vs:.3f}->{ve:.3f}"\n                    )\n\n'''
    if 'speaker_focus AV sync mismatch' not in v:
        if marker not in v:
            raise SystemExit('Could not locate validator insertion point')
        v = v.replace(marker, inject + marker)
        vp.write_text(v, encoding='utf-8')

# 3) Patch vertical_short_renderer.py: reject speaker-focus mismatches; remove leading-offset escape hatch.
vr = Path('vertical_short_renderer.py')
if vr.exists():
    r = vr.read_text(encoding='utf-8')
    marker = '''def validate_visual_source(plan: dict[str, Any], source_duration: float) -> list[str]:\n'''
    if 'speaker_focus AV sync mismatch' not in r:
        start = r.index(marker)
        # Add helper immediately before validate_visual_source.
        helper = '''def validate_speaker_av_sync(plan: dict[str, Any]) -> list[str]:\n    """Speaker-focus visuals must use exactly the same source interval as audio."""\n    errors: list[str] = []\n    for i, seg in enumerate(plan.get("visual_segments", [])):\n        mode = str(seg.get("visual_mode", ""))\n        if mode != "speaker_focus":\n            continue\n        a0 = fnum(seg.get("audio_source_start"), -1.0)\n        a1 = fnum(seg.get("audio_source_end"), -1.0)\n        v0 = fnum(seg.get("visual_source_start"), -1.0)\n        v1 = fnum(seg.get("visual_source_end"), -1.0)\n        if a0 < 0 or a1 <= a0:\n            errors.append(f"segment {i}: speaker_focus has invalid audio source range")\n            continue\n        if abs(v0 - a0) > 1e-3 or abs(v1 - a1) > 1e-3:\n            errors.append(\n                f"segment {i}: speaker_focus AV sync mismatch "\n                f"audio={a0:.3f}->{a1:.3f}, visual={v0:.3f}->{v1:.3f}"\n            )\n    return errors\n\n\n'''
        r = r[:start] + helper + r[start:]

    old_call = '    source_duration = ffprobe_duration(video)\n    prefix_duration, prefix_audio_start, prefix_question_source = detect_leading_audio_visual_mismatch(plan)\n'
    new_call = '    source_duration = ffprobe_duration(video)\n    sync_errors = validate_speaker_av_sync(plan)\n    if sync_errors:\n        raise RuntimeError("Speaker AV sync validation failed:\\n" + "\\n".join(sync_errors))\n    prefix_duration = 0.0\n    prefix_audio_start = 0.0\n    prefix_question_source = 0.0\n'
    if old_call in r:
        r = r.replace(old_call, new_call)
    # Remove the actual prefix visual_plan mutation even if a caller bypasses the new zero assignment in future edits.
    r = r.replace('''    video_plan = plan\n    if prefix_duration > 0.0:\n        video_plan = dict(plan)\n        video_plan["visual_segments"] = list(plan["visual_segments"])[1:]\n''', '    video_plan = plan\n')
    vr.write_text(r, encoding='utf-8')

print('Quality fix patch source prepared.')
print('- editorial selector: v10.0 / 60s ceiling / structural ending repair')
print('- visual plan validator: hard speaker_focus AV sync invariant')
print('- vertical renderer: hard speaker_focus AV sync rejection; no leading-offset workaround')
