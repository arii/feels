# Content-classification models

## The system (final setup)

Cup contents are inferred as a **2-D state spectrum**, not a brittle hard label:

```
              loadedness = P(loaded)   [Stage 1, mass]      empty  <------> full
              rigidity   = P(solid)    [Stage 2, swirl]     liquid <------> solid
```

| file | role |
|---|---|
| `../train_classifier.py` | trains/evaluates the 2-stage cascade leak-free; `--save` writes `cascade.joblib` |
| `../predict_content.py` | runs the cascade on one recording → empty / solid / liquid + confidence |
| `../state_spectrum.py` | the **final representation**: 2-D spectrum bars + **class-filtered** nearest items (a solid is only compared against solids) |
| `../openset.py` | Atlas open-set layer: **one-shot teach** a new item + **novelty detection** (no retrain) |

**What works (honest, leak-free):**
- **empty vs loaded** — reliable (~82% acc), from the mass/loadedness axis.
- **solid vs liquid** — works **when the cup is swirled** (~75%); the rigidity axis
  separates liquid (slosh) from solid (rigid); flagged low-confidence without a swirl.
- **class-filtered retrieval** rescues fine matches: e.g. a screws recording reads as
  SOLID and retrieves other solids/screws, where a raw global kNN drowned it in liquids.
- **Atlas** adds *capability* not accuracy: one-shot learning of new items + novelty.
  On proprioception-only embeddings it does not improve discrimination (items don't
  cluster) — that needs a richer (vision/audio) embedding.

---

## Trained model artifact — `cascade.joblib`

A 2-stage leak-free cascade trained by [`../train_classifier.py`](../train_classifier.py):

```
Stage 1   empty  vs  loaded     (mass contrast)          GradientBoosting
Stage 2   solid  vs  liquid     (slosh/rattle dynamics)  LogisticRegression (all-loaded)
```

Bundle keys: `stage1`, `stage2_all` (default predictor), `stage2_swirl`
(higher isolated score but too few training recs to generalize — kept as an
optional high-confidence reporter), `feature_order`.

**Leak-free results** (within-class time split, earliest 70% train / latest 30% test;
small test folds → approximate):

| decision | macro-F1 | acc | baseline |
|---|---|---|---|
| Stage 1 empty vs loaded | ~0.57 | ~0.82 | 0.46 |
| Stage 2 solid vs liquid (all loaded) | ~0.59 | ~0.64 | 0.33 |
| Stage 2 solid vs liquid (swirl only) | ~0.75 | ~0.75 | 0.37 |
| **End-to-end 3-class** | **~0.47** | **~0.58** | 0.20 |

Per-class (end-to-end): liquid F1 0.72 (recall 1.0), solid F1 0.40 (recall 0.29),
empty F1 0.29 (recall 0.20). Strong on "is it loaded" and on liquid; weak on
solids/empties — the known sensing limit.

Regenerate / re-evaluate:
```
python src/train_classifier.py          # leak-free evaluation of every stage
python src/train_classifier.py --save   # + refit on all data -> models/cascade.joblib
```

---

## `prev_attempts/` — exploratory analyses (NOT in use, kept as the record)

These are the diagnostic scripts from working out *why* naive content
classification failed, before arriving at the cascade. They document the dead
ends honestly. All use a leak-free (time-grouped) evaluation.

These ran against the earlier flat-3-class `train_classifier.py`; the cascade
refactor removed some helpers they imported, so `01`/`02` still run as-is while
`03`/`04` are kept as records only (their results are quoted below). All findings
are reproduced by the main `train_classifier.py` and `content_classifier.ipynb`.

| script | runs? | what it showed |
|---|---|---|
| `01_confound_audit_and_mass_regression.py` | ✅ | **Temporal confound**: recording-time alone predicts class at ~100% (classes were recorded in time-blocks), so any random/label-stratified split leaks the label. Also first mass-regression (R²<0). |
| `02_leak_free_timesplit.py` | ✅ | Removing the leak (within-class earliest→train/latest→test) **collapses flat 3-class to ~baseline** — the first honest number. |
| `03_mass_regression_and_knn.py` | record | Aggressive-denoise mass regression → **Pearson r≈0, R²<0** (no monotonic payload signal). KNN lands at the same confounded ~0.3 / baseline-honest. |
| `04_data_augmentation.py` | record | Jitter + class-balance + mixup → **collapses to baseline** both leaky and leak-free (augmentation resamples existing points; it cannot add missing information). |

### Why the flat approach was abandoned
Seven independent methods (mean-diff, regression R², confound audit, leak-free
split, augmentation, KL/Jensen-Shannon divergence, learned-metric LDA/NCA) all
agreed: **flat solid/water/empty content ID is not supported by this data** — the
between-class divergence sits at the random-split noise floor, and material
dynamics are aliased away by 4–5 Hz cloud sampling.

The **cascade** recovers real signal by splitting the problem: "is it loaded"
(big mass contrast) and "liquid vs solid" (swirl dynamics) are each learnable
leak-free, where the flat 3-class was not.

### What would lift results further (data-side, not modeling)
- Interleave recording order (kills the temporal confound — free).
- Cantilever grasp pose (larger moment arm → payload torque above the noise floor).
- 100 Hz on-robot capture (`control_service.py`) — exposes slosh/rattle for liquid vs solid.
- Always swirl (so the strong Stage-2 swirl path applies to every pick).
