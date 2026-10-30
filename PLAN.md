# PLAN.md — Friday runbook (demos 3:30 PM)

Branch `feat/swirl-extractor-dashboard` holds the full system. All Thursday-night
build items are done (LDA decision head, docs site, packaging, lint/test green).

## Friday morning

1. **Pull + rebuild the Atlas library.** The library must be rebuilt from
   swirl-protocol recordings or the LDA head silently falls back and old
   embeddings are stale-scale:
   ```bash
   git pull
   python feature_extractor.py --calibrate --require-swirl   # paste into FEATURE_SCALES if changed
   python teach_csv.py recordings/ --require-swirl --force --session fri
   ```
2. **Teach the demo set** via dashboard Mode 2 (needs robot + Atlas): `empty_cup`,
   `water_075pct` (most separable item — teach several), `solid_020g`/`solid_050g`,
   `screws_100g`. Budget ~2 min per item. Or batch: `teach_csv.py --require-swirl`.
3. **Gate every run.** `python predict_mystery.py --qc` after each batch — discard
   runs with INIT_DRIFT / INIT_MOTION flags. Tare baseline is the #1 data-quality risk.
4. **Dry-run Mode 1** on a true mystery cup. Close calls are flagged in the UI —
   teach the item on the spot (that's the online-learning story).
5. **Rehearse the fallback.** `make mock` runs the entire 2-mode demo from recorded
   data with the MOCK badge shown. If the robot or network misbehaves at 3:30,
   this is the backup — never present it as live.

## Demo talk track (2 min)

1. The problem: lidded cup, can't see inside — identify contents by *feel*.
2. The move: one continuous pick-up → swirl → place → back-off (no fragile multi-part dance).
3. The loop: swirl → 10-D torque signature → Atlas vector search → LDA head predicts; teach new items live, no retraining.
4. The proof: confound set — cups at the same mass, different contents. Mass alone can't solve it; dynamics can.

## Honest limits (don't overclaim to judges)

- Static hold torques carry **no** payload signal at this pose (verified identical across contents); everything discriminative comes from the swirl dynamics.
- The 5–25 Hz rattle band is **not** observable at 4–5 Hz cloud sampling — the extractor reports `rattle_band_limited`. The real band needs the 100 Hz on-robot module (`control_service.py`, not yet deployed).
- **Motion protocol must match between library and query.** Same item across swirl/fallback protocols scores ~0.4 cosine; different items within one protocol reach ~0.9. Never mix; `--require-swirl` guards ingest.
- Category accuracy is ~52% measured on 21 held-out cups (the dashboard shows this number, not a per-prediction score). A brim-full cup is haptically a solid block — no free surface, no slosh.

## After the hackathon

- Stability-gated tare in `repeatable_grasp.py` (tare at grasp-approach after settle, not INIT).
- Deploy `control_service.py` for 100 Hz sampling → real rattle-band features.
- Merge the branch; full `src/` package layout (deferred — flat layout kept deliberately for demo-week stability).
