# Formal Problem Definition

## Dynamic Identification of Enclosed Payloads via Non-Parametric Metric Retrieval under Fixed Swirl Excitation

---

## 1. Problem Statement

The objective is to determine the latent categorical and physical identity $\theta^* \in \Theta$ of an unobservable payload enclosed within a sealed container (a covered cup). Because direct line-of-sight to the interior is occluded, the manipulator interacts with the container by executing a deterministic, repeatable dynamic trajectory:

$$a^* = [a_{\text{approach}} \circ a_{\text{grasp}} \circ a_{\text{lift}} \circ a_{\text{swirl}} \circ a_{\text{place}} \circ a_{\text{release}}] \in \mathcal{A}$$

The system logs high-frequency proprioceptive time-series telemetry $D = \{s_t\}_{t=0}^T$ (joint torques, actuator angles, and gripper state), maps $D$ via a deterministic feature extractor $\psi(D) \to z \in \mathbb{R}^{10}$, normalizes the vector to the unit hypersphere $e = z / \|z\|_2 \in \mathbb{S}^9$, and performs non-parametric k-NN retrieval via MongoDB Atlas Vector Search against an online signature database.

---

## 2. Latent Parameter Space ($\Theta$)

Each enclosed payload is modeled as $\theta = (y, \phi) \in \Theta$:

### Categorical Taxonomy ($y \in \mathcal{Y}$)
- **`empty_cup`**: Baseline empty container ($\Delta \tau \approx 0$).
- **`water_half`**: Low-viscosity Newtonian fluid exhibiting swirl vortex dynamics.
- **`dish_soap_half`**: High-viscosity fluid exhibiting strong viscous swirl damping.
- **`candy_mixed`**: Discrete loose solids exhibiting high-frequency impact rattle (5–25 Hz).
- **`solid_weight`**: Rigid lumped-mass solid exhibiting zero swirl phase lag.

### Continuous Intrinsic Properties ($\phi \in \mathbb{R}^k$)
- Net payload mass: $m \in [0, M_{\max}]$
- Viscous damping / swirl phase lag: $\Delta \phi_{\text{lag}} \ge 0$
- Granular chatter / rattle spectral power: $E_{\text{rattle}} \ge 0$
- Dynamic swirl torque variance: $\sigma^2_{\tau} \ge 0$

---

## 3. Scoping Decisions & Action Formulation

> [!IMPORTANT]
> **Rejection of Pushing, Violent Shaking, and Dynamic Policy Iteration:**
> 1. **No Pushing or Impedance Sliding:** Surface contact sliding introduces unpredictable Coulomb friction variations, micro-chatter, and table slippage.
> 2. **No Violent Linear Shake or Tilt Sweeps:** Replaced with smooth horizontal orbital swirling $a_{\text{swirl}}$ which generates continuous centripetal and tangential excitation without risking fluid spills.
> 3. **No Dynamic Bayesian Planning:** Trajectory decisions are not generated dynamically; the system executes a single, deterministic, certified routine $a^*$.

### Action Breakdown ($a^*$)

1. **$a_{\text{grasp}}$ (Fixed Taped-Square Acquisition):** Vertical alignment and gripper closure at calibrated coordinates.
2. **$a_{\text{lift}}$ (Static Hold, $t \in [0, T_1]$):** Vertical translation to elevated transit height $z_{\text{travel}}$ and fixed dwell to isolate gravitational torque:
   $$G(q, m) = \Delta \tau_{\text{gravity}}$$
3. **$a_{\text{swirl}}$ (Orbital Excitation, $t \in (T_1, T_2)$):** Continuous planar circular orbit at constant elevation:
   $$x(t) = x_0 + R \cos(2\pi f_{\text{swirl}} t), \quad y(t) = y_0 + R \sin(2\pi f_{\text{swirl}} t), \quad z(t) = z_{\text{travel}}$$
4. **$a_{\text{place}}$ (Controlled Placement, $t \in (T_2, T_3)$):** Smooth descent back to table square, jaw release, and return to home staging.

---

## 4. Feature Space & Metric Retrieval

### Deterministic Feature Mapping
$$\psi: \mathcal{D} \to \mathbb{R}^{10}$$

Extracts:
1. $\Delta \tau_{\text{gravity}}$ (Static mass)
2. $\text{Var}(\tau_{3,\text{swirl}})$ (Dynamic swirl variance)
3. $f_{\text{peak}}$ (FFT peak frequency)
4. $P_{\text{peak}}$ (FFT peak power)
5. $E_{\text{rattle}}$ (5–25 Hz chatter energy)
6. $\Delta \phi_{\text{lag}}$ (Swirl cross-correlation phase shift)
7. $\tau_{\text{centripetal}}$ (Lateral/longitudinal coupling)
8. $\Delta g_{\text{grip}}$ (Aperture deformation)
9. $\Delta i_{\text{grip}}$ (Motor current load)
10. $S_{\text{stability}}$ (Steady-state ratio)

### Normalized Metric Embedding & Atlas Vector Search
$$e = \frac{\psi(D)}{\|\psi(D)\|_2} \in \mathbb{S}^9$$

Predicted category:
$$\hat{y} = \arg\max_{y \in \mathcal{Y}} \frac{1}{|\mathcal{N}_k(e, y)|} \sum_{e_i \in \mathcal{N}_k(e, y)} \cos(e, e_i)$$

where $\mathcal{N}_k(e)$ represents the top-$k$ nearest neighbors returned by MongoDB Atlas `$vectorSearch`.
