"""
Step 1: Download the source video.

Why yt-dlp and not the YouTube API: the official API doesn't give you
the actual video file, just metadata. yt-dlp handles the messy part
(format negotiation, throttling, etc.) for free.

We cap resolution at 1080p so downloads stay fast and ffmpeg doesn't
choke later — we're producing 9:16 shorts, not archiving in 4K.
"""

import os
import yt_dlp


def download_video(url: str, out_dir: str) -> str:
    """Downloads `url` into `out_dir`, returns the path to the mp4 file."""
    os.makedirs(out_dir, exist_ok=True)
    out_template = os.path.join(out_dir, "source.%(ext)s")

    ydl_opts = {
        "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": out_template,
        "merge_output_format": "mp4",
        "quiet": False,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filepath = ydl.prepare_filename(info)
        # merge_output_format can change the extension after download
        base, _ = os.path.splitext(filepath)
        mp4_path = base + ".mp4"
        return mp4_path if os.path.exists(mp4_path) else filepath


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python download.py <youtube_url>")
        sys.exit(1)
    path = download_video(sys.argv[1], "tmp")
    print(f"Downloaded to: {path}")
