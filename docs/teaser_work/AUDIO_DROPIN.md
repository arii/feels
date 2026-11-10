# Voiceover drop-in contract (v3 script)

The `FeelsShowcase` composition (900 frames, 30fps, 1920x1080) is visual-final
with the v3 layout: slim top headers (scenes 1–3), right-side two-metric
validation panel (scene 4), centered outro card over dimmed team photo
(scene 5).

When voiceover is ready, drop files here:

- `feels-video/public/audio/scene1_hook.mp3`
- `feels-video/public/audio/scene2_motion.mp3`
- `feels-video/public/audio/scene3_telemetry.mp3`
- `feels-video/public/audio/scene4_benchmark.mp3`
- `feels-video/public/audio/scene5_outro.mp3`

…or a single pre-stitched master as either
`feels-video/public/audio/master_vo.mp3` or
`feels-video/public/audio/master_vo.wav` (48 kHz stereo preferred).

## Exact VO script (v3, supersedes v2)

1. Hook (0–6s): "At the Viam Physical AI hackathon, Dhyuti and I wanted to
   see if a robot arm could guess what's inside a sealed coffee cup using
   only motor torques."
2. Motion (6–13s): "No cameras inside the cup. The arm picks it up, runs a
   deterministic swirl motion, and streams joint states in real time."
3. Telemetry (13–20s): "We don't model internal fluid dynamics. We just
   measure joint torque deviations from a nominal baseline trajectory to
   build signature vectors."
4. Benchmark (20–25s): "On 21 held-out test cups, nearest-neighbor
   classification got 52% on 3-way categories, and 86 to 89% separating
   empty from full."
5. Outro (25–30s): "Full writeup, code, and dataset links are at
   arii.github.io/feels."

## Agent integration steps (when files land)

1. Loudness-normalize to a single master with the ffmpeg skill
   (`loudness.py`, -16 LUFS / -1.5 dBTP for the mix target the project
   owner chose) → `feels-video/public/master_audio.mp3`.
2. Mount one `<Audio src={staticFile("master_audio.mp3")} />` at the root
   of `FeelsShowcase`, wrapped in the project's fade envelope.
3. If any VO line overruns its scene, resize that
   `TransitionSeries.Sequence` (durations are adjacent-safe) and re-render.
4. Render `out/feels_linkedin_final.mp4`, retag bt709 via
   `color.py --retag bt709` (stream copy), re-run the LinkedIn check.
