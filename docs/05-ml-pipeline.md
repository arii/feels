# ML Pipeline — Feature Extraction, Embeddings & Atlas Vector Search

> [!IMPORTANT]
> **On-Site Deep Learning Training Archived:**
> We do NOT train CNNs, Transformers, or PyTorch neural networks on-site. Training deep networks under limited time and sample constraints risks severe overfitting and instability. Instead, we compute deterministic, hand-crafted physical features from the pick-up, swirl, and hold telemetry and map them to a normalized unit embedding $e \in \mathbb{S}^9$ for non-parametric MongoDB Atlas Vector Search (`$vectorSearch`).

---

## 1. Feature Extraction Pipeline (`feature_extractor.py`)

The feature extraction engine transforms variable-length time-series rollouts $D = \{s_t\}_{t=0}^T$ into a fixed 10-dimensional physical feature vector $z \in \mathbb{R}^{10}$, then calculates the normalized metric embedding $e \in \mathbb{S}^9$.

```
Telemetry Rollout D
  ├── Phase 1: Static Hold  ──► Δτ_gravity (Mass)
  ├── Phase 2: Swirl Orbit  ──► Var(τ_swirl), f_peak, P_peak, E_rattle, Δφ_lag
  └── Compliance            ──► Δg_grip, Δi_grip
                                        ↓
                         Raw 10D Feature Vector z
                                        ↓
                       Normalized Unit Embedding e = z / ‖z‖₂
                                        ↓
                     MongoDB Atlas $vectorSearch Retrieval
```

### Phase 1: Static Hold (Gravitational Mass Proxy)
- Measures steady-state load torque difference on Joint 3:
  $$\Delta \tau_{\text{gravity}} = \bar{\tau}_{3,\text{hold}} - \tau_{\text{tare}}$$
- Directly estimates net payload mass:
  $$\hat{m} = \frac{|\Delta \tau_{\text{gravity}}|}{g \cdot L_3}$$

### Phase 2: Horizontal Orbital Swirl (Slosh, Chatter, and Viscosity)
During the smooth horizontal swirl orbit:
- **Swirl Torque Variance:**
  $$\text{Var}(\tau_{3,\text{swirl}}) = \frac{1}{N}\sum (\tau_3(t) - \bar{\tau}_3)^2$$
- **FFT Spectral Resonances:** Fast Fourier Transform on detrended torque:
  $$Y(f) = \mathcal{F}\{\tau_3(t) - \bar{\tau}_3\}$$
  - Dominant frequency $f_{\text{peak}} = \arg\max |Y(f)|^2$
  - Dominant peak power $P_{\text{peak}} = \max |Y(f)|^2$
- **High-Frequency Rattle Energy (5–25 Hz):** Flags loose granular candy or small objects rattling inside the cup:
  $$E_{\text{rattle}} = \sum_{5\,\text{Hz} \le f \le 25\,\text{Hz}} |Y(f)|^2$$
- **Swirl Phase Lag ($\Delta \phi_{\text{lag}}$):** Cross-correlation phase shift between the commanded orbital circular trajectory and the measured actuator torque response:
  $$\Delta \phi_{\text{lag}} = \arg\max_\delta (R_{\tau, \text{cmd}}(\delta)) \cdot \frac{2\pi f_{\text{swirl}}}{f_s}$$
  - Distinguishes lagging fluid vortex slosh (water, viscous soap) from rigid lumped mass (zero phase lag).

### Full 10-Dimensional Feature Vector ($z$)

| Index | Feature | Physical Meaning | Item Separation Property |
|:-----:|---------|------------------|--------------------------|
| 0 | $\Delta \tau_{\text{gravity}}$ | Steady-state mass offset (Nm) | Separates empty cup from loaded cups |
| 1 | $\text{Var}(\tau_{\text{swirl}})$ | Swirl residual torque variance | High for liquids and loose items, low for rigid mass |
| 2 | $f_{\text{peak}}$ | Dominant spectral frequency (Hz) | Natural sloshing or rattle oscillation frequency |
| 3 | $P_{\text{peak}}$ | Dominant spectral power | Intensity of internal dynamic oscillation |
| 4 | $E_{\text{rattle}}$ | Integrated 5–25 Hz rattle energy | **Sharp spikes for loose candy; near-zero for liquids** |
| 5 | $\Delta \phi_{\text{lag}}$ | Orbital swirl phase lag (rad) | **Large for viscous fluids; zero for solid weights** |
| 6 | $\tau_{\text{centripetal}}$ | Lateral centripetal torque coupling | Measures rotational fluid vortex inertia |
| 7 | $\Delta g_{\text{grip}}$ | Gripper stroke deflection (mm) | Senses container compliance / compression |
| 8 | $\Delta i_{\text{grip}}$ | Gripper motor load current | Grip resistance during motion |
| 9 | $S_{\text{stability}}$ | Steady-state stability metric | Percentage of time within resting bounds |

---

## 2. Normalized Metric Embedding ($e \in \mathbb{S}^9$)

Rather than relying on learned neural network parameters, the vector is directly normalized to the unit hypersphere:

$$e = \frac{z}{\|z\|_2 + \epsilon}$$

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
The nearest neighbors are grouped by label, and the prediction is determined by:
1. **Predicted Label:** Class with the highest mean cosine similarity score among the top-k matches.
2. **Confidence Percentage:** Mean similarity score $\times 100\%$.
3. **Low-Confidence Flag:** Triggered if top confidence is below $0.80$, alerting the operator to check cup placement.

---

## 4. Online Recording & Instant Teaching (UX)

When registering a new or updated item:
1. Operator places the item on the taped square and inputs its label (e.g. `dish_soap_half`).
2. Robot executes the single pick-up, swirl, and place routine.
3. `feature_extractor.py` computes $e$.
4. A single call to `atlas.teach(label, e, raw_samples)` inserts the new signature into the `signatures` collection.
5. The item is **immediately indexable and queryable** by subsequent runs without retraining or downtime.
