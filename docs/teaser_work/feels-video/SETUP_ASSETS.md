# Asset setup (regenerates ignored `public/` binaries)

`public/*.mp4` and `public/*.webp` are git-ignored derivatives. Recreate them
from the tracked sources in `docs/images` (paths below run from `feels-video/`):

```bash
SKILL=/home/ari/.claude/skills/ffmpeg-skill/scripts  # adjust to your install

# Byte-identical copies (already SDR 1080p30 / stills)
cp ../images/feels_screen_view.mp4 public/screen.mp4
cp ../images/team-venue.webp public/team.webp
cp ../images/robot-cup.webp public/robot-cup.webp

# Swirl source is 4K HLG HDR 29.97fps: tonemap to SDR, then conform
python3 "$SKILL/color.py" ../images/feels_swirl.mp4 --to-sdr -o /tmp/swirl_sdr.mp4
python3 "$SKILL/fit.py" /tmp/swirl_sdr.mp4 --height 1080 --fps 30 -o public/swirl.mp4
rm /tmp/swirl_sdr.mp4
```

Verify with: `python3 "$SKILL/probe.py" public/swirl.mp4 --compact`
 Expect: 24.1s, 1920x1080 @ 30fps, h264 yuv420p, no audio.
