# Formal Problem Definition

## Interactive Multi-Modal Dynamic Identification of Enclosed Payloads via Non-Parametric Metric Retrieval

---

## 1. Problem Statement

The goal is to determine the latent physical and categorical identity θ* ∈ Θ of an unobservable payload enclosed inside a sealed, rigid or semi-rigid container (a covered coffee cup). Because direct visual line-of-sight to the interior is occluded, the manipulator interacts physically with the container by executing a deterministic dynamic trajectory a ∈ A. The system logs multi-modal time-series telemetry D — chiefly high-frequency proprioception (joint kinematics, drive currents, torques, and wrist wrench) alongside grasp telemetry and visual stability constraints — and maps D to an embedding space Z ⊂ R^d to perform non-parametric nearest-neighbor classification and open-set incremental registration using a dynamic vector memory.

---

## 2. Latent Parameter Space (Θ)

Each enclosed payload is parameterized by: θ = (y, φ) ∈ Θ

### Categorical Taxonomy (y ∈ Y)
- Empty container
- Discrete loose solids (e.g., candy, pencils, hardware fasteners)
- Continuous low-viscosity Newtonian fluids (e.g., water)
- Continuous high-viscosity non-Newtonian/viscous fluids (e.g., liquid soap)
- Rigid lumped-mass solids (e.g., solid weights, dense tools)

### Continuous Intrinsic Properties (φ ∈ R^k)
- Net payload mass: m ∈ [0, M_max]
- Center of mass offset: r_com ∈ R³ relative to the container frame
- Damping coefficient / viscous lag: μ ≥ 0
- Internal chatter / impact index: σ_chatter ≥ 0 (quantifying internal collision impulses)
- Container fill ratio: α ∈ [0, 1]

---

## 3. State & Observation Formulations

### System State (s_t ∈ S) at time t ∈ [0, T]

```
s_t = [q_t^T,  q̇_t^T,  τ_t^T,  w_t^T,  g_t,  ġ_t]^T
```

- q_t, q̇_t ∈ R^n — Manipulator joint positions and velocities
- τ_t ∈ R^n — Measured joint actuator torques / motor drive currents
- w_t = [f_t^T, τ_wrist,t^T]^T ∈ R^6 — End-effector 6-axis force-torque wrench
- g_t, ġ_t ∈ R — Gripper aperture position and motor load current (monitoring container deformation or micro-slippage)

### Observation Trajectory (D): For execution horizon of duration T sampled at frequency f_s

```
D = {s_t, I_t}_{t=0}^T
```

where I_t represents visual verification frames from the camera to monitor external container pose and slip boundaries.

---

## 4. Action Space (A)

To ensure real-time determinism, the action space is formalized as a composite multi-phase excitation trajectory a_composite = [a_lift ∘ a_sine ∘ a_tilt] executed in Cartesian space:

### a_lift — Static Gravitational Excitation, t ∈ [0, T_1]
Constant vertical hold at fixed height z_hold to isolate static gravity loading G(q, m).

### a_sine — Harmonic Kinematic Excitation, t ∈ (T_1, T_2)
1D periodic horizontal oscillation x(t) = x_0 + A·sin(2πf_0·t)·ê_x designed to trigger fluid slosh phase lag and dry solid inertial rattling.

### a_tilt — Quasi-Static Angular Sweep, t ∈ (T_2, T_3)
Bounded angular tilt θ_pitch(t) ∈ [−θ_max, +θ_max] about the end-effector roll/pitch axis to measure center-of-mass migration and torsional hysteresis.

---

## 5. Dynamic Interaction Model

The manipulator-container-payload dynamic equations are governed by:

```
M(q)q̈ + C(q,q̇)q̇ + G(q) + J^T(q) · w_ext(t; θ) = τ(t)
```

The unmodeled external wrench w_ext(t; θ) transmits the signature of the hidden payload:

- **Newtonian fluid:** F_int(t) exhibits a hydrodynamic slosh resonance with phase delay Δφ = ∠(a_tcp, w_ext) > 0 governed by fluid height α
- **Viscous fluid:** Shows overdamped transient response with attenuated resonance amplitude and substantial viscous dissipation μ
- **Loose dry objects:** Exhibit piecewise discontinuous contact collisions, injecting high-frequency transient power spectra into τ(t) and w_ext(t)
- **Rigid mass:** Displays zero relative internal motion (F_int = 0, Δφ ≈ 0), acting strictly as a rigid parameter update to the manipulator's link inertia

---

## 6. Feature Projection & Non-Parametric Metric Retrieval

### Deterministic Feature Map (ψ: D → R^d)

The continuous rollout D is transformed into an explicit feature descriptor z:

```
z = [
  mass_j2,       # static gravity torque offset, shoulder (J2)
  mass_j3,       # static gravity torque offset, elbow (J3)
  swirl_var_j2,  # torque variance at J2 during swirl
  swirl_var_j3,  # torque variance at J3 during swirl
  swirl_ptp_j3,  # peak-to-peak J3 torque during swirl
  phase_lag_s,   # lag of max |xcorr(TCP speed, J3 residual)|, ±2 s window
  rattle_e,      # high-frequency energy, observable band only (band-limited at 4–5 Hz)
  grip_comp,     # gripper position std during swirl (compliance)
  lift_dyn_j3,   # dynamic J3 torque during lift (inertial cue)
  swirl_speed,   # mean TCP speed during swirl (excitation normalization)
]
```

### Normalized Metric Embedding

```
e = Wz / ‖Wz‖₂  ∈ S^(d-1)
```

where W is an empirical scaling/whitening projection matrix.

### Metric Retrieval & Classification Objective

Given an indexed memory bank M = {(e_i, y_i, φ_i)}_{i=1}^N stored within MongoDB Atlas Vector Search:

```
ŷ = argmax_{y ∈ Y}  Σ_{i ∈ Top-k(e, M)}  I(y_i = y) · cos(e, e_i)

Confidence(ŷ) = (1/k) · Σ_{i ∈ Top-k, y_i=ŷ}  cos(e, e_i)
```

where cos(u, v) = u^T·v denotes cosine similarity.

### Open-Set Continual Registration

If an operator submits an unobserved item label y_new during runtime:

```
M ← M ∪ {(e_new, y_new, φ_estimated)}
```

The item becomes instantaneously retrievable in subsequent identification cycles without retraining weights.
