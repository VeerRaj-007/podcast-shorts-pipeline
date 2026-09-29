from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path

ROOT=Path('/mnt/data')
DEFAULT_VIDEO=ROOT/"Leon Kennedy meets Grace Ashcroft  Nick Apostolides and Angela Sant'Albano Interview Each Other - FGS (1080p, h264).mp4"
DEFAULT_PLAN=ROOT/'new_inde_camera_plan_v1.json'
DEFAULT_OUTPUT=ROOT/'new_inde_face_aware_render_v1.mp4'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--video',default=str(DEFAULT_VIDEO))
    p.add_argument('--plan',default=str(DEFAULT_PLAN))
    p.add_argument('--output',default=str(DEFAULT_OUTPUT))
    args=p.parse_args()
    plan=json.loads(Path(args.plan).read_text(encoding='utf-8'))
    segs=plan['camera_segments']
    cmd=['ffmpeg','-y','-hide_banner','-loglevel','warning']
    for s in segs:
        a=float(s['source_start']); b=float(s['source_end'])
        cmd += ['-ss',str(a),'-to',str(b),'-i',args.video]
    filters=[]
    for i,s in enumerate(segs):
        c=s['crop']; left=int(round(c['left'])); top=int(round(c['top'])); w=int(round(c['width'])); h=int(round(c['height']))
        filters.append(f'[{i}:v]setpts=PTS-STARTPTS,crop={w}:{h}:{left}:{top},scale=1080:1920:flags=lanczos,setsar=1[v{i}]')
        filters.append(f'[{i}:a]asetpts=PTS-STARTPTS[a{i}]')
    inputs=''.join(f'[v{i}][a{i}]' for i in range(len(segs)))
    filters.append(inputs+f'concat=n={len(segs)}:v=1:a=1[v][a]')
    cmd += ['-filter_complex',';'.join(filters),'-map','[v]','-map','[a]','-c:v','libx264','-preset','medium','-crf','18','-c:a','aac','-b:a','192k','-movflags','+faststart',args.output]
    print('=== FACE-AWARE CAMERA RENDER v1 ===')
    print('segments:',len(segs))
    subprocess.run(cmd,check=True)
    print('Output:',args.output)

if __name__=='__main__': main()