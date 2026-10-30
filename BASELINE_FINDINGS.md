# Baseline findings: why the static-hold signal doesn't discriminate (2026-10-29)

Checked offline against the 8 real grasp-cycle recordings in `recordings/`
(7 labeled + 1 mystery) with the same cleaning/feature pipeline as
`notebooks/content_discrimination.ipynb`. Two verified problems:

## 1. The tare baseline is contaminated

`extract_features` tares on the INIT-phase median. In the recorded data:

- **New-format recordings** (`recording_20261029_180225_empty_cup.csv`): INIT is
  sampled while the arm is still moving to home (q2 sweeps 18.3° during INIT,
  tcp starts at the travel pose, not home). tau2 drifts ±37 N·m inside INIT.
- **Old-format recordings**: INIT is at home but the torques are still settling
  from the previous motion — tau2 drifts 11–20 N·m inside INIT while the arm
  is stationary (e.g. `154108_candy_mixed`: 51.7 → 40.2 N·m).

Baseline noise (~10 N·m) dwarfs the payload signal (≤200 g ≈ 0.5–2 N·m on τ3
at this pose). Every Δτ feature inherits it.

## 2. The static hold has no payload signal

Absolute hold torques are **identical across all contents to 3 decimals**:

| recording | contents | HOLD τ2 | HOLD τ3 |
|---|---|---|---|
| all 8 recordings | empty / water_half / candy_mixed / mystery | 43.566 N·m | −13.269 N·m |

At the current travel pose the payload does not move the static joint torques
above the noise floor. Retaring on TRAVEL_SETTLING instead of INIT does not
help: it produces two fake clusters (+4 vs +0 N·m on Δτ3) that split
empty_cup across both groups — a settle transient, not contents.

## Consequence

Leave-one-recording-out k-NN on the 12-D joint vector: **3/7 = 43%**.
The mystery prediction (empty_cup at 67%, nearest neighbor candy_mixed at
essentially tied distance) is not trustworthy. More static-hold data will not
fix this — the signal isn't there.

## Recommended fixes (in priority order)

1. **Stability-gated tare.** Tare only after the arm is confirmed settled at a
   known pose — reuse `wait_for_stability_in_travel()` at the grasp-approach
   pose *before* descending, or tare at travel after the settle gate passes.
   Never tare while any joint is moving or within ~2 s of a motion.
2. **Discriminate on dynamics, not statics.** The 4–5 Hz cloud sampling can't
   see the 5–25 Hz rattle band (notebook §6). The SWIRL/shake phases are the
   right direction — compare swirl-phase Δτ spectra per class once
   `run_plan.sh` P0 data (E1, E3–E5, E9, E12, E17, E21–E24) exists.
3. **Gate runs with `predict_mystery.py --qc`** before trusting a prediction —
   it flags INIT drift, mid-motion tares, and settle transients per recording.
4. **The confound set (E21–E24: solid_150g vs water_150g vs candy_150g vs
   soap_150g) is the demo that matters** — same mass, different contents. Mass
   alone can't solve it; that matrix is the proof the judges need.

## Deliverable

`predict_mystery.py` — standalone offline classifier + evaluator + QC gate.
Mirrors the notebook's cleaning pipeline exactly:

```bash
python predict_mystery.py                        # classify mystery recordings
python predict_mystery.py recordings/foo.csv     # classify one CSV
python predict_mystery.py --eval                 # LORO accuracy on references
python predict_mystery.py --qc                   # tare-baseline quality gate
```

Reproduces the notebook's mystery result; honest about confidence.
