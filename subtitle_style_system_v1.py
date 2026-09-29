from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


STYLE_CATALOG: list[dict[str, Any]] = [
    {
        "id": "clean_white",
        "name": "Clean White",
        "category": "minimal",
        "description": "Clean creator-style captions with strong readability.",
        "font": {
            "family": "Arial",
            "weight": "bold",
            "size": 58,
        },
        "fill": {
            "type": "solid",
            "color": "#FFFFFF",
        },
        "stroke": {
            "enabled": True,
            "color": "#000000",
            "width": 3,
        },
        "shadow": {
            "enabled": True,
            "offset_x": 2,
            "offset_y": 3,
            "blur": 2,
        },
        "highlight": {
            "enabled": False,
        },
        "layout": {
            "alignment": "center",
            "anchor": "lower_center",
            "bottom_margin": 250,
            "max_lines": 2,
            "max_words_per_caption": 6,
        },
        "animation": {
            "enter": "fade_up",
            "exit": "fade",
            "word_pop": False,
        },
    },
    {
        "id": "bold_tiktok",
        "name": "Bold TikTok",
        "category": "social",
        "description": "Large high-energy captions with highlighted keywords.",
        "font": {
            "family": "Arial",
            "weight": "black",
            "size": 64,
        },
        "fill": {
            "type": "solid",
            "color": "#FFFFFF",
        },
        "stroke": {
            "enabled": True,
            "color": "#000000",
            "width": 5,
        },
        "shadow": {
            "enabled": True,
            "offset_x": 3,
            "offset_y": 4,
            "blur": 2,
        },
        "highlight": {
            "enabled": True,
            "color": "#FFE600",
            "mode": "keyword",
        },
        "layout": {
            "alignment": "center",
            "anchor": "lower_center",
            "bottom_margin": 270,
            "max_lines": 2,
            "max_words_per_caption": 5,
        },
        "animation": {
            "enter": "scale_pop",
            "exit": "fade",
            "word_pop": True,
        },
    },
    {
        "id": "mrbeast_energy",
        "name": "High Energy",
        "category": "creator",
        "description": "Aggressive emphasis and fast word-level movement.",
        "font": {
            "family": "Arial",
            "weight": "black",
            "size": 68,
        },
        "fill": {
            "type": "solid",
            "color": "#FFFFFF",
        },
        "stroke": {
            "enabled": True,
            "color": "#000000",
            "width": 6,
        },
        "shadow": {
            "enabled": True,
            "offset_x": 4,
            "offset_y": 4,
            "blur": 2,
        },
        "highlight": {
            "enabled": True,
            "color": "#FF3B30",
            "mode": "keyword",
        },
        "layout": {
            "alignment": "center",
            "anchor": "lower_center",
            "bottom_margin": 290,
            "max_lines": 2,
            "max_words_per_caption": 4,
        },
        "animation": {
            "enter": "bounce",
            "exit": "fade",
            "word_pop": True,
        },
    },
    {
        "id": "karaoke",
        "name": "Karaoke Highlight",
        "category": "word_sync",
        "description": "Words activate one by one in sync with speech.",
        "font": {
            "family": "Arial",
            "weight": "bold",
            "size": 60,
        },
        "fill": {
            "type": "solid",
            "color": "#FFFFFF",
        },
        "stroke": {
            "enabled": True,
            "color": "#000000",
            "width": 4,
        },
        "shadow": {
            "enabled": True,
            "offset_x": 2,
            "offset_y": 3,
            "blur": 2,
        },
        "highlight": {
            "enabled": True,
            "color": "#00E5FF",
            "mode": "current_word",
        },
        "layout": {
            "alignment": "center",
            "anchor": "lower_center",
            "bottom_margin": 270,
            "max_lines": 2,
            "max_words_per_caption": 5,
        },
        "animation": {
            "enter": "fade_up",
            "exit": "fade",
            "word_pop": True,
        },
    },
    {
        "id": "cinematic",
        "name": "Cinematic",
        "category": "cinema",
        "description": "Elegant captions with restrained movement.",
        "font": {
            "family": "Georgia",
            "weight": "bold",
            "size": 52,
        },
        "fill": {
            "type": "solid",
            "color": "#FFFFFF",
        },
        "stroke": {
            "enabled": True,
            "color": "#111111",
            "width": 2,
        },
        "shadow": {
            "enabled": True,
            "offset_x": 1,
            "offset_y": 2,
            "blur": 3,
        },
        "highlight": {
            "enabled": False,
        },
        "layout": {
            "alignment": "center",
            "anchor": "lower_center",
            "bottom_margin": 230,
            "max_lines": 2,
            "max_words_per_caption": 8,
        },
        "animation": {
            "enter": "fade",
            "exit": "fade",
            "word_pop": False,
        },
    },
    {
        "id": "podcast_host",
        "name": "Podcast Host",
        "category": "podcast",
        "description": "Professional interview captions with restrained emphasis.",
        "font": {
            "family": "Arial",
            "weight": "bold",
            "size": 56,
        },
        "fill": {
            "type": "solid",
            "color": "#FFFFFF",
        },
        "stroke": {
            "enabled": True,
            "color": "#000000",
            "width": 3,
        },
        "shadow": {
            "enabled": True,
            "offset_x": 2,
            "offset_y": 2,
            "blur": 2,
        },
        "highlight": {
            "enabled": True,
            "color": "#00FF85",
            "mode": "keyword",
        },
        "layout": {
            "alignment": "center",
            "anchor": "lower_center",
            "bottom_margin": 250,
            "max_lines": 2,
            "max_words_per_caption": 6,
        },
        "animation": {
            "enter": "fade_up",
            "exit": "fade",
            "word_pop": False,
        },
    },
    {
        "id": "gaming",
        "name": "Gaming",
        "category": "gaming",
        "description": "Large punchy captions for fast speech and reactions.",
        "font": {
            "family": "Arial",
            "weight": "black",
            "size": 66,
        },
        "fill": {
            "type": "solid",
            "color": "#FFFFFF",
        },
        "stroke": {
            "enabled": True,
            "color": "#000000",
            "width": 6,
        },
        "shadow": {
            "enabled": True,
            "offset_x": 4,
            "offset_y": 4,
            "blur": 1,
        },
        "highlight": {
            "enabled": True,
            "color": "#00FF66",
            "mode": "keyword",
        },
        "layout": {
            "alignment": "center",
            "anchor": "lower_center",
            "bottom_margin": 285,
            "max_lines": 2,
            "max_words_per_caption": 4,
        },
        "animation": {
            "enter": "scale_pop",
            "exit": "fade",
            "word_pop": True,
        },
    },
    {
        "id": "news",
        "name": "News / Interview",
        "category": "editorial",
        "description": "Professional lower-risk style for interviews and commentary.",
        "font": {
            "family": "Arial",
            "weight": "bold",
            "size": 54,
        },
        "fill": {
            "type": "solid",
            "color": "#FFFFFF",
        },
        "stroke": {
            "enabled": True,
            "color": "#000000",
            "width": 3,
        },
        "shadow": {
            "enabled": False,
        },
        "highlight": {
            "enabled": False,
        },
        "layout": {
            "alignment": "left",
            "anchor": "lower_left",
            "left_margin": 90,
            "bottom_margin": 240,
            "max_lines": 2,
            "max_words_per_caption": 7,
        },
        "animation": {
            "enter": "fade",
            "exit": "fade",
            "word_pop": False,
        },
    },
    {
        "id": "typewriter",
        "name": "Typewriter",
        "category": "stylized",
        "description": "Subtle character-driven style with restrained reveal animation.",
        "font": {
            "family": "Courier New",
            "weight": "bold",
            "size": 52,
        },
        "fill": {
            "type": "solid",
            "color": "#FFFFFF",
        },
        "stroke": {
            "enabled": True,
            "color": "#000000",
            "width": 3,
        },
        "shadow": {
            "enabled": False,
        },
        "highlight": {
            "enabled": True,
            "color": "#FFD60A",
            "mode": "current_word",
        },
        "layout": {
            "alignment": "center",
            "anchor": "lower_center",
            "bottom_margin": 235,
            "max_lines": 2,
            "max_words_per_caption": 6,
        },
        "animation": {
            "enter": "type_on",
            "exit": "fade",
            "word_pop": False,
        },
    },
    {
        "id": "minimal_upper",
        "name": "Minimal Upper",
        "category": "minimal",
        "description": "Captions placed higher to avoid lower UI overlays.",
        "font": {
            "family": "Arial",
            "weight": "bold",
            "size": 54,
        },
        "fill": {
            "type": "solid",
            "color": "#FFFFFF",
        },
        "stroke": {
            "enabled": True,
            "color": "#000000",
            "width": 3,
        },
        "shadow": {
            "enabled": True,
            "offset_x": 1,
            "offset_y": 2,
            "blur": 2,
        },
        "highlight": {
            "enabled": False,
        },
        "layout": {
            "alignment": "center",
            "anchor": "center",
            "vertical_offset": 160,
            "max_lines": 2,
            "max_words_per_caption": 6,
        },
        "animation": {
            "enter": "fade_up",
            "exit": "fade",
            "word_pop": False,
        },
    },
]


def build_catalog() -> dict[str, Any]:
    return {
        "version": "1.0",
        "module": "subtitle_style_system",
        "timing_source": {
            "provider": "WhisperX",
            "source_of_truth": "aligned_transcript.json",
            "word_level_timestamps": True,
            "timing_mutated_by_style": False,
        },
        "render_target": {
            "width": 1080,
            "height": 1920,
            "safe_area": {
                "top": 180,
                "bottom": 220,
                "left": 60,
                "right": 60,
            },
        },
        "style_selection": {
            "consumer_selects_one_preset": True,
            "preset_count": len(STYLE_CATALOG),
            "customization_supported_later": True,
        },
        "presets": STYLE_CATALOG,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build_catalog(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=== SUBTITLE STYLE SYSTEM v1 ===")
    print(f"presets: {len(STYLE_CATALOG)}")
    for style in STYLE_CATALOG:
        print(f'- {style["id"]}: {style["name"]}')
    print(f"output: {output}")


if __name__ == "__main__":
    main()
