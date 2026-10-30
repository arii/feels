# ML Pipeline — Feature Extraction, Embeddings & Atlas Vector Search

> [!IMPORTANT]
> **On-Site Deep Learning Training Archived:**
> We do NOT train CNNs, Transformers, or PyTorch neural networks on-site. Training deep networks under limited time and sample constraints risks severe overfitting and instability. Instead, we compute deterministic, hand-crafted physical features from the pick-up, swirl, place, and back-off telemetry and map them to a normalized unit embedding $e \in \mathbb{S}^9$ for non-parametric MongoDB Atlas Vector Search (`$vectorSearch`).

---

## 1. Feature Extraction Pipeline (`feature_extractor.py`)

The feature extraction engine transforms variable-length time-series rollouts $D = \{s_t\}_{t=0}^T$ into a fixed 10-dimensional physical feature vector $z \in \mathbb{R}^{10}$, then calculates the normalized metric embedding $e \in \mathbb{S}^9$.

```
Telemetry Rollout D
  ├── Static lift hold ──► mass_j2, mass_j3 (gravity torque offsets)
  ├── Swirl orbit       ──► swirl_var_j2, swirl_var_j3, swirl_ptp_j3,
  │                        phase_lag_s, rattle_e (observable band only)
  ├── Lift transient   ──► lift_dyn_j3 (inertial cue)
  ├── Grip             ──► grip_comp (compliance under load)
  └── Excitation       ──► swirl_speed (normalization)
                                        ↓
                         Raw 10D Feature Vector z
                                        ↓
            Per-feature scaled, L2-normalized embedding e = Wz / ‖Wz‖₂
                                        ↓
                     MongoDB Atlas $vectorSearch Retrieval
```

### Phase 1: Static Hold (NOT discriminative — audit finding)
Static hold torques were verified identical across contents to 3 decimals
($\tau_2 = 43.566$, $\tau_3 = -13.269$ N·m): at this pose the payload does not
move the static joint torques above the tare-noise floor. The `mass_j2` /
`mass_j3` features are kept in the vector but carry no payload signal until
the tare is stability-gated. Everything discriminative comes from the swirl
*dynamics* below. See `BASELINE_FINDINGS.md`.

### Phase 2: Horizontal Orbital Swirl (Slosh, Chatter, and Viscosity)
During the smooth horizontal swirl orbit:
- **Swirl Torque Variance:**
  $$\text{Var}(\tau_{3,\text{swirl}}) = \frac{1}{N}\sum (\tau_3(t) - \bar{\tau}_3)^2$$
- **Swirl Torque Spectrum (observable band only):** FFT on the detrended J3 residual,
  $$Y(f) = \mathcal{F}\{r_3(t)\},$$
  integrated over the *observable* band only:
  $$E_{\text{rattle}} = \sum_{1.5\,\text{Hz} \le f \le f_s/2} |Y(f)|^2.$$
  At 4–5 Hz cloud sampling the 5–25 Hz rattle band is **not** observable
  (Nyquist ~2 Hz); the extractor flags `rattle_band_limited`. The true band
  needs the 100 Hz on-robot module (`control_service.py`, not yet deployed).
- **Swirl Phase Lag ($\phi_{\text{lag}}$):** lag of max |cross-correlation|
  between TCP speed and the J3 torque residual, within ±2 s:
  $$\phi_{\text{lag}} = \arg\max_{|\delta| \le 2} |R_{v, r_3}(\delta)|.$$
  - Distinguishes lagging fluid vortex slosh (water, viscous soap) from rigid lumped mass (near-zero lag).

### Full 10-Dimensional Feature Vector ($z$)

As implemented in `feature_extractor.py` (`FEATURE_NAMES`):

| Index | Feature | Physical Meaning | Item Separation Property |
|:-----:|---------|------------------|--------------------------|
| 0 | `mass_j2` | Static gravity torque offset, shoulder (J2) | Payload mass estimate |
| 1 | `mass_j3` | Static gravity torque offset, elbow (J3) | Payload mass estimate |
| 2 | `swirl_var_j2` | Torque variance at J2 during swirl | Slosh / loose-content energy |
| 3 | `swirl_var_j3` | Torque variance at J3 during swirl | Slosh / loose-content energy |
| 4 | `swirl_ptp_j3` | Peak-to-peak J3 torque during swirl | Shifting-mass swing |
| 5 | `phase_lag_s` | Lag of max \|xcorr(TCP speed, J3 residual)\| | **Large for viscous fluids; near-zero for rigid mass** |
| 6 | `rattle_e` | High-frequency energy, *observable band only* (1.5 Hz–Nyquist) | Loose-item energy; **band-limited at 4–5 Hz sampling** |
| 7 | `grip_comp` | Gripper position std during swirl | Squishy vs rigid contents |
| 8 | `lift_dyn_j3` | Dynamic J3 torque during lift | Inertial mass cue |
| 9 | `swirl_speed` | Mean TCP speed during swirl | Excitation normalization |

---

## 2. Normalized Metric Embedding ($e \in \mathbb{S}^9$)

Rather than relying on learned neural network parameters, the vector is
per-feature scaled (90th-percentile scales in `FEATURE_SCALES`, calibrated on
real recordings and frozen) and normalized to the unit hypersphere:

$$e = \frac{Wz}{\|Wz\|_2}$$

This ensures that MongoDB Atlas Vector Search computes exact cosine similarity without needing neural network training or backpropagation.

---

## 3. MongoDB Atlas Vector Search Pipeline

Atlas operates as our metric retrieval engine. Queries are executed using the native `$vectorSearch` aggregation stage against the `cup_dynamics_index`:

```javascript
[
  {
    $vectorSearch: {
      index: "cup_dynamics_index",
      path: "feature_vector",
      queryVector: query_embedding,
      numCandidates: 20,
      limit: 5,
      filter: { session: "thu" } // optional session filter
    }
  },
  {
    $project: {
      label: 1,
      properties: 1,
      score: { $meta: "vectorSearchScore" }
    }
  }
]
```

### Classification Verdict
Atlas `$vectorSearch` retrieves the neighborhood; the call is made by a
shrinkage-LDA head (`atlas.predict_lda`) fit in milliseconds on the whole
labeled library at predict time — no training step, so teaching takes effect
immediately. Measured accuracy on 21 held-out cups: category (liquid / solid /
empty) 52%, material 43%, empty/nonempty 86%. The LDA's own probabilities are
uncalibrated, so the dashboard shows the *measured* accuracy instead of a
per-prediction score, plus close-call flagging on thin margins. Falls back to
pure vector-search retrieval if the library can't support the LDA fit.

---

## 4. Online Recording & Instant Teaching (UX)

When registering a new or updated item:
1. Operator places the item on the taped square and inputs its label (e.g. `dish_soap_half`).
2. Robot executes the single pick-up, swirl, place, and back-off routine.
3. `feature_extractor.py` computes $e$.
4. A single call to `atlas.teach(label, e, raw_samples)` inserts the new signature into the `signatures` collection.
5. The item is **immediately indexable and queryable** by subsequent runs without retraining or downtime.
