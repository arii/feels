# TODO — current state (Thu night, demos Fri 3:30 PM)

## Done
- [x] 10-D swirl feature extractor (`feature_extractor.py`): real `z`,
  per-feature scales, L2 embedding `e`, plain-language descriptions, diagnostics
  (tare QC, rattle band-limit flag), `--calibrate --require-swirl`
- [x] Smooth swirl macro in `repeatable_grasp.py` (pick → swirl → place → back-off;
      file is **locked** — don't modify)
- [x] Dashboard (`realtime_plot.py` + `static/index.html`): Mode 1 identify, Mode 2
      teach, correction flow, feature-space panel, "why" physics (residuals + phase
      lag, honest observable-band spectrum), vector-space card, `/docs` site, mock mode
- [x] Atlas layer (`atlas.py`): teach/identify, material/category taxonomy,
      margin/ambiguity guards, shrinkage-LDA decision head (`predict_lda`),
      `--smoke` gating on the fake signature
- [x] `teach_csv.py`: batch teaching with `--require-swirl`, `--dry-run`, `--force`, `--list`
- [x] `predict_mystery.py`: offline classifier + `--eval` / `--qc` tare gate
- [x] Packaging: `setup.sh`, `Makefile`, `pyproject.toml`, `--help` epilogs, README quickstart
- [x] Lint/test green (pyflakes; mock smoke suite)

## Friday (before 3:30 PM demos)
- [ ] Rebuild the Atlas library from swirl recordings (`teach_csv.py --require-swirl --force --session fri`)
- [ ] Teach the demo set via Mode 2: `empty_cup`, `water_075pct` (several), `solid_020g`/`solid_050g`, `screws_100g`
- [ ] QC every batch (`predict_mystery.py --qc`); discard INIT_DRIFT / INIT_MOTION runs
- [ ] 3 dress rehearsals; confirm the `make mock` fallback works on the demo machine

## After the hackathon
- [ ] Stability-gated tare in `repeatable_grasp.py`
- [ ] Deploy `control_service.py` (100 Hz) → real rattle-band features
- [ ] Merge `feat/swirl-extractor-dashboard` to main
- [ ] `src/` package layout (deferred deliberately for demo-week stability)

## Archived (deliberately scoped out)
Pushing / impedance sliding, violent linear shakes, tilt sweeps, vision grasping,
on-site deep learning, Bayesian policy planning — see `archive/` for the exploration
scripts and superseded planning docs.
