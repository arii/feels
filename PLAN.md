# PLAN.md — finish the hackathon

Concrete next steps, in order. Demos Friday 3:30 PM; hard close tonight 9 PM.

## Tonight (before 9 PM hard close)

1. **Merge or hold `feat/swirl-extractor-dashboard`.** Review the diff, merge to main if clean — everything after tonight should run off one branch.
2. **Atlas ready.** `MONGODB_URI` in `.env`; run `python src/atlas.py` setup so the `signatures` collection and the `cup_dynamics_index` vector index (10 dims, cosine) exist before any teach call. Then **delete the fake setup signature** the script inserts:
   `python -c "import atlas; db = atlas.get_db(); print(db.signatures.delete_many({'label': 'water_half', 'motion_type': '3in1_macro'}).deleted_count)"`
   (matched by `motion_type` so a real `water_half` taught later is untouched).
2b. **Swirl protocol only.** Teach exclusively from recordings with a true SWIRL phase — `teach_csv.py --require-swirl` enforces this, and the dashboard always records with `--swirl-smooth`. Mixed protocols confound the embedding (verified: same cup scores ~0.4 cosine across protocols vs ~0.9 for different items within one protocol).
3. **Run the P0 data collection.** `P0_ONLY=1 bash run_plan.sh` — E1, E3–E5, E9, E12, E17, E21–E24. The confound set (E21–E24: solid/water/candy/soap at 150 g) is the demo that matters: same mass, different contents. Prioritize it if time is short.
4. **Gate every run.** `python src/predict_mystery.py --qc` after each batch — discard runs with INIT_DRIFT / INIT_MOTION flags. The tare baseline is the #1 data-quality risk (see BASELINE_FINDINGS.md).
5. **Recalibrate and freeze.** `python src/feature_extractor.py --calibrate` on the P0 swirl recordings, paste the new `FEATURE_SCALES` into `feature_extractor.py`, commit. Rescaling later invalidates Atlas signatures already taught.

## Friday morning

6. **Teach the item library** via dashboard Mode 2 (needs the robot + Atlas) or in batch with `teach_csv.py recordings/ --require-swirl --session thu`: empty, water half, candy, soap, + the confound set. Each dashboard teach is one swirl macro — budget ~2 min per item with setup.
7. **Dry-run Mode 1** on a true mystery cup. If confidence < 82%, the UI says so — teach that item on the spot (that's the online-learning story for the judges).
8. **Rehearse the fallback.** `FEELS_MOCK=1 python src/realtime_plot.py` runs the entire 2-mode demo from recorded data with the MOCK badge shown. If the robot or network misbehaves at 3:30, this is the backup — never present it as live.

## Demo talk track (2 min)

1. The problem: lidded cup, can't see inside — identify contents by *feel*.
2. The move: one continuous pick-up → swirl → place (no fragile multi-part dance).
3. The loop: swirl → 10-D torque signature → Atlas vector search → prediction + neighbors; teach new items live, no retraining.
4. The proof: confound set — four cups, same 150 g, different contents. Mass alone can't solve it; dynamics can.

## Honest limits (don't overclaim to judges)

- Static hold torques carry **no** payload signal at this pose (verified identical across contents); everything discriminative comes from the swirl dynamics.
- The 5–25 Hz rattle band is **not** observable at 4–5 Hz cloud sampling — the extractor reports `rattle_band_limited`. The real band needs the 100 Hz on-robot module (`control_service.py`, not yet deployed).
- Mass-offset features are noise until the tare is stability-gated (tare only after the settle gate passes, never mid-motion).
- **Motion protocol must match between library and query.** The embedding encodes the excitation motion as much as the contents: same item across swirl/fallback protocols scores ~0.4 cosine, while different items within one protocol reach ~0.9. Never mix; `--require-swirl` guards the ingest path.
- Current offline LORO is 43% on 7 pre-P0 recordings — the P0 swirl dataset is what makes the classifier real.

## After the hackathon

- Stability-gated tare in `repeatable_grasp.py` (tare at grasp-approach after settle, not INIT).
- Deploy `control_service.py` for 100 Hz sampling → real rattle-band features.
- Merge the branch, archive the superseded docs (`docs/04` 3-motion macro, `docs/05` tilt/FFT pipeline) into `docs/archive/`.
