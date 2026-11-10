# Agent Notes: Teaser review and Remotion rebuild workflow

General playbook for two jobs: (A) review an existing video teaser for
completeness and delivery readiness, (B) rebuild it as a Remotion project.
No repo-specific paths or project details below — adapt names to the task.

## Skills and load order

1. `remotion-best-practices` (router) → its `remotion-create/REFERENCE.md`
   before scaffolding, `remotion-markup/REFERENCE.md` before writing JSX.
2. Before registering compositions or timelines: `compositions.md`,
   `video-editing.md`, `multi-scene-video.md` from remotion-markup.
3. `tailwind.md` when specs use Tailwind classes; `voiceover.md` when VO
   comes up (default TTS guidance: ElevenLabs + API key — ask the user
   for their provider/key rather than assuming).
4. `ffmpeg-skill` for every media op (probe, convert, retag, check,
   contact sheets). Never hand-roll raw `ffmpeg`/`ffprobe` — if no script
   exposes the operation, say so and name the closest built-in option.

## Phase 1 — Review existing footage before proposing anything

- Inventory everything: `probe.py` in compact mode on all clips (duration,
  fps, resolution, codecs, audio presence). Silent clips vs. sounding clips
  matters for what follows.
- Render contact sheets (`look.py`): dense grids for finalists, sparse
  grids for variants/endcards. View every sheet as an image before judging.
- If a "master" and a "published/delivered" copy both exist, diff them
  (`cmp` + check moov-before-mdat faststart on both) to prove whether the
  published file matches the master or is a stale/derived copy.
- Run the platform compliance check (`check.py --platform <target>`) — it
  is the delivery gate, not your eyeball.
- Match video content against any copy/cue/shot-list docs: every promised
  caption, metric, credit line, and URL must appear verbatim; every linked
  URL target must exist.
- Key audio learning: a silent audio track FAILs audio and loudness checks,
  while no audio stream at all only WARNs. Never loudness-normalize
  silence — either add real audio or ship streamless.

## Phase 2 — Scaffold the Remotion project

- Never scaffold into a non-empty folder: inspect it (including dotfiles)
  first, then scaffold into a fresh subfolder
  (`npx create-video@latest --yes --blank --no-tailwind <name>` + `npm i`).
- Add packages with `npx remotion add` for all `@remotion/*` packages
  (`@remotion/media`, `@remotion/transitions`, …).
- Tailwind v4: install `@remotion/tailwind-v4` + `tailwindcss`, add the
  `enableTailwind` bundler override to the config file, create a CSS entry
  with `@import 'tailwindcss'`, import it in the root file.
- Delete scaffold boilerplate compositions once replaced.

## Phase 3 — Prep assets into public/

- Conform hostile sources BEFORE editing: HDR/HLG → SDR tonemap
  (`color.py --to-sdr`), then scale/conform (`fit.py --height 1080
  --fps 30`) into `public/`. Phone footage is routinely 4K / 29.97fps /
  10-bit HLG — all three must go for predictable editing.
- Already-conformant clips and stills: plain copy, no re-encode.
- Do NOT pre-trim: trimming lives in JSX (`trimBefore` on the sequence
  boundary) so cuts stay adjustable without re-encoding.
- Contact-sheet tools often write next to their input — move strays back
  into your working folder.

## Phase 4 — Compose

- One `<Composition>` with inline metadata; for multi-scene work make the
  parent a `TransitionSeries` so scenes stay adjacent when later retiming
  (e.g. fitting scenes to voiceover lengths) changes durations.
- One component per scene; each scene's video + overlays under its own
  timing boundary; `name` props on series/sequences/clips for Studio.
- `<Video>`/`<Audio>` from `@remotion/media` — `objectFit` is a direct
  prop, never via `style` (lint warns). Images via `CanvasImage`. Assets
  via `staticFile()` from `public/`.
- Motion: `useCurrentFrame()` in the smallest component that needs it,
  `spring()` entrances (damping ~14–16, small frame delay), `interpolate()`
  inline in style, `scale`/`translate`/`rotate` properties — never
  `transform` strings, CSS transitions, or Tailwind animate classes.
- Type scale for 1080p video: pasted specs almost always arrive in webpage
  px (24px headlines) that are unreadable on video and hopeless on mobile
  feeds. Scale up aggressively (scene headlines ~60px, details ~30–36px,
  badges ~20px, hero numerals 64–110px) and keep key text ~140px+ from
  frame edges. Verify by rendering stills, never by reasoning.
- Baked-in text cards must each read as a complete standalone statement so
  the video works on mute in autoplaying feeds.

## Phase 5 — Verify without relying on Studio preview

- If the environment cannot serve ports to a host browser (sandboxed
  network namespaces, disabled approvals), do not fight it: Studio live
  preview is unavailable, so verify with renders instead.
- `npm run lint` must be 0 errors; fix warnings rather than carrying them.
- Render one still per scene at each scene's midpoint; inspect every still
  as an image before proceeding.
- Full render, then probe, contact sheet, and platform check; inspect the
  sheet before reporting done.
- Visual fixes MUST be re-rendered to verify: gradient/scrim math is
  unreliable unrendered (a bottom-55% scrim once visibly did nothing while
  72% worked). Never trust opacity math without a still.

## Phase 6 — Iterate on revised specs

- Keep every prior final render versioned (`v1`, `v2`, …) for comparison;
  never overwrite the deliverable name — version it.
- Pasted specs routinely contain bugs (wrong-cased identifiers), internal
  conflicts (two subtexts, flipped name orders), and webpage-scale layouts.
  Fix the unambiguous, follow scene-local copy verbatim, and flag each
  genuine conflict in one line rather than guessing user-owned choices
  (names, venues, numeric claims).
- Layout revisions should only touch overlay components and scene JSX;
  trims, timing, and prepared assets stay put.

## Phase 7 — Deliver

- Render → retag color to the delivery spec via stream copy (no re-encode)
  → platform check → contact sheet.
- Target: 0 failed. Common acceptable WARNs: subtitle track (covered when
  baked-in cards are genuinely standalone), missing audio until VO lands.
- Voiceover integration pattern: keep a written drop-in contract in the
  workspace (exact filenames, exact script lines, mix target). When files
  land: loudness-normalize to one master with the ffmpeg skill, mount a
  single `<Audio>` at the composition root inside a fade envelope, resize
  any overrun sequences (adjacent-safe), re-render, retag, re-check.
- Never mount audio files that do not exist yet — an empty drop folder
  means the silent + standalone-cards fallback stays the deliverable.
