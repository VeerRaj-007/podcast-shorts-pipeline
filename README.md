# podcast-shorts

YouTube URL in → N vertical (9:16) captioned clips out.

## Pipeline (each file = one step, run standalone or via main.py)

1. `download.py` — pulls the video with yt-dlp
2. `transcribe.py` — extracts audio, transcribes with Groq's `whisper-large-v3` (segment timestamps)
3. `pick.py` — sends the transcript to `llama-3.3-70b-versatile` on Groq, gets back N best moments as segment index ranges (not raw seconds — see comment in the file for why)
4. `cut.py` — ffmpeg cuts each moment, center-crops to 9:16, burns in captions

## Setup (run this on your own machine, not in a sandbox)

```bash
pip install -r requirements.txt
brew install ffmpeg   # or apt install ffmpeg on Linux
```

Get a free Groq API key: https://console.groq.com/keys

```bash
export GROQ_API_KEY=your_key_here
python main.py "https://youtube.com/watch?v=SOME_PODCAST" --clips 3
```

Output lands in `output/clip_1.mp4`, `clip_2.mp4`, etc.

## What's deliberately NOT here yet

No database, no auth, no frontend, no queue, no dashboard. This is
step 1 from the roadmap: "one podcast → one genuinely good short."
Don't add infrastructure until you've watched a handful of outputs
and confirmed the clip *selection* is actually good — that's the
part that decides whether this product idea has legs, and no amount
of architecture fixes a bad "pick the best moment" prompt.

## Known rough edges to expect on first runs

- Center-crop assumes the speaker is roughly centered in frame. If
  the source video has a wide two-shot (two people side by side),
  the crop will cut someone off — you'll want a face-tracking crop
  eventually, not now.
- The selection prompt has no concept of audio energy/laughter/
  emphasis, only text. It'll sometimes pick something that reads well
  but was delivered flatly. Worth comparing text-only selection
  against a version that also gets word-level timing/pauses later.
- Long podcasts (90+ min) produce a long segment list — past a
  certain transcript length you may need to chunk it before sending
  to the LLM instead of one giant prompt.
