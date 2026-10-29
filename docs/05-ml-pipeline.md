# ML Pipeline — Feature Extraction, Embeddings & Atlas Vector Search

## Overview

The ML pipeline converts raw multi-modal robot telemetry into a compact embedding vector, then uses MongoDB Atlas Vector Search for k-NN classification and online item registration — no gradient retraining needed between demo runs.

---

## 1. Feature Extraction Pipeline (Edge Compute)

Convert variable-length time-series signals from the motion library (a_lift, a_shake, a_tilt) into a fixed-dimensional feature vector e ∈ R^d.

### Static Hold
- Steady-state gravity torque difference Δτ_g to estimate net payload mass m

### Periodic Shake / Tilt (FFT Features)
- **Dominant resonant frequencies and spectral spread** — distinguishes sloshing liquid from rigid bodies
- **Phase lag Δφ** between commanded acceleration and measured torque — viscosity indicator
- **High-frequency transient power (5–25 Hz rattle band)** — loose candy/pens vs. solid objects

### Grasp Aperture & Viscoelasticity
- Change in gripper stroke Δg and grip stall current during shake — detects compliant vs. rigid cup responses

### Full 10-Dimensional Feature Vector

| Feature | Description |
|---------|-------------|
| Δτ_gravity | Steady-state mass offset during static lift |
| Var(τ̃_shake) | Residual torque variance during sine shake |
| f_peak | Dominant spectral frequency via FFT |
| P_peak | Dominant spectral power via FFT |
| E_rattle | Integrated spectral energy in high-frequency 5–25 Hz band |
| Δφ_lag | Cross-correlation phase shift between input acceleration and measured torque |
| Δτ_hyst | Torque hysteresis loop area during tilt |
| Δg_grip | Grip deformation metric |
| Δi_grip | Grip compliance stall metric |
| (reserved) | Additional hand-crafted feature slot |

---

## 2. Embedding Model

Pass the extracted features through a lightweight encoder:
- **Option A:** Small MLP trained via PyTorch on the Linux compute host
- **Option B:** Temporal contrastive network (1D-CNN or Mini-LSTM / Temporal Transformer)

Output: normalized d-dimensional embedding vector e = Wz / ‖Wz‖₂ ∈ S^(d-1)

where W is an empirical scaling/whitening projection matrix.

---

## 3. Online Training via MongoDB Atlas Vector Search

Atlas serves as a metric-learning memory store, avoiding traditional slow gradient retraining between demo runs.

### Atlas Document Schema

```json
{
  "_id": "run_20261029_001",
  "item_label": "dish_soap_half_full",
  "category": "viscous_liquid",
  "properties": {
    "estimated_mass_g": 185,
    "viscosity_level": "high",
    "fill_ratio": 0.5
  },
  "motion_type": "fig8_sweep",
  "feature_vector": [0.142, 0.891, -0.321, "... d-dim normalized embedding ..."],
  "created_at": "2026-10-29T11:00:00Z"
}
```

### Atlas Collections
- **`runs`** — Raw arrays, one document per run
- **`signatures`** — Features + embedding (indexed for vector search)
- **`items`** — Label, description, photo

### Index Configuration

Configure an **Atlas Vector Search** index (cosine or Euclidean similarity) on `feature_vector`:

```js
$vectorSearch: {
  index: "cup_dynamics_index",
  path: "feature_vector",
  queryVector: <embedding>,
  numCandidates: 20,
  limit: 5
}
$project: {
  item_label: 1,
  category: 1,
  score: { $meta: "vectorSearchScore" }
}
```

---

## 4. Inference & Novel Item Registration

### Prediction (k-NN Query)
Execute the probing motions on an unknown cup, extract its embedding e_query, and issue an Atlas aggregation pipeline (`$vectorSearch`, k = 5, then `$group` by label).

**Confidence** = mean cosine similarity of the winning label's k neighbours.

### Active Identification & Uncertainty
If the top-1 cosine score is below a certainty threshold (e.g., < 0.82), the system selects the next most informative motion (a_push or a_tilt) to disambiguate the state.

### Online "One-Shot" Teaching
When registering a new item during the live demo:
1. Trigger the standard 3-motion diagnostic suite via the web UI
2. Embed the resulting trajectories
3. Insert the document into Atlas tagged with the user's label
4. The item is immediately queryable via Vector Search without restarting the server or retraining a classifier

---

## 5. Inference & Online Learning Formulation

### Bayesian Filtering / Interactive Belief Update

Maintain a belief state b_k(θ) = p(θ | D_{1:k}) updated sequentially over exploratory actions a_1, ..., a_K:

```
p(θ | D_{1:k}) ∝ p(D_k | a_k, θ) · p(θ | D_{1:k-1})
```

- **Likelihood p(D_k | a_k, θ):** Evaluated either via an analytical reduced-order model (mass-spring-dashpot slosh model) or a learned surrogate feature extractor

### Feature Representation

Trajectory rollouts are mapped to compact dynamic descriptors:

- **Time domain:** Mean steady torque, variance/kurtosis of residual torque τ̃ = τ_meas − τ_nominal_empty, peak transient acceleration
- **Frequency domain (FFT/PSD):** Dominant spectral frequencies, power spectral density in the 5–25 Hz rattle band, spectral phase difference between commanded input and force response

### Online Adaptation / Few-Shot Incremental Learning

Train a temporal backbone (1D-CNN or Mini-LSTM / Temporal Transformer) to map trajectory features z_k into an embedding space Z ⊂ R^d via contrastive loss.

**Online Registration:** When introducing a new object θ_new during the live demo, execute the canonical motion suite, project the rollout embeddings into Z, and update class prototypes or append to a k-NN / Gaussian Process classifier without full network retraining.

### Active Exploration Policy (Next-Best Action)

Select the subsequent action a_{k+1} that maximizes expected information gain (mutual information) regarding θ:

```
a*_{k+1} = argmax_{a ∈ A} I(θ; D_{k+1} | a, b_k)
           = argmax_{a ∈ A} [H(b_k(θ)) − E_{D_{k+1}}[H(b_{k+1}(θ))]]
```

---

## 6. Dynamic System Model

The combined robot-container dynamics follow the manipulator equation:

```
M(q)q̈ + C(q,q̇)q̇ + G(q) + J^T(q) · w_ext(θ) = τ
```

Where the external wrench w_ext(θ) acting on the end-effector is governed by the internal contents:

- **Rigid solid:** F_slosh = 0; wrench aligns strictly with rigid body acceleration a_tcp
- **Loose items (candy/pencils):** Non-linear impulse spikes (chatter) in τ and w_t caused by internal impacts
- **Liquid payload:** Frequency-dependent phase lag Δφ between commanded acceleration and measured torque τ, with amplitude attenuation scaling with viscosity μ

---

## 7. Viam Agent Tool Definitions (Runtime Agent Loop)

Structure the core identification logic as an **LLM/Agent Tool-Use loop**:

### `execute_probing_motion(motion_type: "lift" | "tilt" | "sine_shake" | "fig8")`
- Invokes the Viam Motion Service / Cartesian controller
- Automatically coordinates joint trajectory streaming with background sensor buffering
- Returns: `execution_status`, time-series duration, and raw stream handle

### `extract_dynamic_signature(stream_handle)`
- Calculates tare-subtracted steady-state gravity offset (mass estimate)
- Computes FFT spectral resonance and phase lag Δφ between command and torque response
- Calculates internal impulse chatter (rattle score in 5–25 Hz band)
- Returns: Extracted feature summary + normalized d-dimensional embedding vector

### `query_atlas_memory(embedding_vector, top_k=3)`
- Runs a `$vectorSearch` query against the MongoDB Atlas cluster
- Returns: Nearest matched object categories, confidence similarity scores, and historical physical parameters

### `register_novel_item(embedding_vector, label, properties)`
- Inserts the completed multi-probe embedding and metadata into Atlas for zero-shot dynamic memory
