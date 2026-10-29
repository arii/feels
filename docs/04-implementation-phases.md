# Implementation Phases & Roadmap

## The Minimal Viable Demo — "3-Motion Macro"

Program one fixed, repeatable 6-second trajectory:

1. **Lift & Hold (1.5 s):** Reads steady-state joint torque / motor current to calculate mass
2. **Sine Shake (2.5 s):** Shakes horizontally along one axis at 2–3 Hz to capture rattle spikes (candy/pencil) vs. fluid slosh lag (water/soap)
3. **Slow Tilt (2.0 s):** Tilts end-effector 25° and returns to level to measure static CoM shift

---

## Streamlined Item Library (Distinct Physical Signatures)

Stick to 4–5 items that have dramatically different physical responses:

| Item | Signature |
|------|-----------|
| Empty cup | Low mass, near-zero shake resistance, no rattle |
| Water (half-full) | Medium mass, dynamic phase lag, low high-frequency rattle |
| Dish soap (half-full) | Medium mass, heavy viscous damping, delayed peak torque |
| Halloween fun-size candy / loose screws | Sharp, high-frequency impact chatter (5–25 Hz rattle peaks) |
| Solid heavy weight (e.g., small metal tool / solid roll of tape) | High mass, rigid coupling, zero rattle, zero phase lag |

---

## Simplified Tech Stack & Roles

- **Robot Control:** Write a single async Python script using Viam Python SDK (`MoveToPosition` or joint waypoints)
- **Telemetry Buffer:** Record a single Pandas DataFrame per run (`timestamp`, `joint_torques`, `tcp_wrench`, `gripper_load`)
- **Database & Memory:** Push the 10-dimension feature vector directly to MongoDB Atlas. Query with a straightforward `$vectorSearch` aggregation pipeline
- **UI:** A single-page lightweight dashboard (Streamlit or Vite/React) with two buttons:
  - **"Identify Cup"** — Runs macro → extracts features → queries Atlas → displays top 3 matches with % confidence
  - **"Teach New Item"** — Runs macro → asks for text label → inserts document into Atlas

---

## Phase 1: MVP (Target: Thursday Dinner / 6:00 PM)

*Goal: Prove the full end-to-end loop — pick a cup, run the fixed 3-in-1 motion, extract a basic feature vector, store/retrieve from MongoDB Atlas, and print the match.*

- [ ] **Taped Workspace Setup:** Tape a fixed target square on the table for consistent cup placement (bypassing visual servoing)
- [ ] **Hardcoded Pick & Place Script:** Write a Python script using the Viam Arm API to approach the taped square, close the gripper to grab the cup, lift 15 cm, and place it back down
- [ ] **Telemetry Logging:** In a background thread, log timestamps, joint torques/currents, and wrist F/T readings to a local Pandas DataFrame or CSV during motion
- [ ] **Standard 3-in-1 Motion Macro:**
  - Hold static (1.5 s)
  - 1D horizontal sine shake at 2–3 Hz (2.5 s)
  - Tilt 25° and return (2.0 s)
- [ ] **Baseline Feature Extractor:** Calculate an initial 6-dimensional vector:
  - Static lift torque Δτ_lift (mass proxy)
  - Shake torque variance σ²_τ
  - Dominant FFT peak frequency f_max
  - FFT peak power P_max
  - Rattle band energy (5–25 Hz)
  - Tilt torque hysteresis Δτ_tilt
- [ ] **Atlas Vector Pipeline:**
  - Seed Atlas with 3 baseline items (Empty cup, Water, Candy)
  - Write a Python script to send a query vector via `$vectorSearch` and print the top match and cosine similarity score

---

## Phase 2: Live Operator UI & Visual Proof (Target: Friday 12:30 PM)

*Goal: Replace the terminal script with a browser dashboard to make the physics and data flow visible to judges.*

- [ ] **Web Dashboard (Vite / React or Streamlit):**
  - Live camera stream showing the cup on the table
  - Big action buttons: "Run Diagnostic", "Tare/Home", and "Emergency Stop"
  - Telemetry graph: Plotting commanded velocity vs. measured torque in real time to show phase lag
  - FFT spectrum chart highlighting rattle spikes
- [ ] **Interactive Teaching Loop:**
  - Add an input box: "Label New Item" + button "Record & Save to Atlas"
  - Allows registering a new item live during the demo without code restarts
- [ ] **Expanded Physical Item Suite:** Calibrate 5 items with distinct dynamic signatures:
  - Empty cup
  - Water (half-full)
  - Dish soap / shampoo (viscous fluid)
  - Fun-size candy or screws (chattering solid)
  - Solid rigid weight (rigid coupling)

---

## Phase 3: Polish & Expansion (Target: Friday 2:30 PM — Before 3:30 PM Demos)

*Goal: Add intelligence, agent interaction, and safety layers if time permits.*

- [ ] **Confidence Threshold & Fallback Probing:** If the top Atlas vector search score is < 0.85, trigger an auxiliary action (e.g., a secondary vertical jiggle or push) to disambiguate close matches
- [ ] **LLM / Viam Agent Skill Integration:** Wrap the diagnostic routine and Atlas query as callable tools in an LLM agent that outputs an explanation (e.g., *"Detected low high-frequency rattle but high dynamic lag, matching viscous liquid properties"*)
- [ ] **Slip / Deformation Detection:** Use gripper stroke percentage and motor load to detect if a lightweight cup is being crushed or sliding during high-acceleration shakes
- [ ] **3-Cup Mystery Game (Stretch Goal):** Allow the user to place three identical cups on marked spots; the robot inspects each one in sequence to find which one holds the target object

---

## What to Cut vs. What to Keep

| Cut | Keep |
|-----|------|
| Dynamic vision tracking (visual servoing / continuous cup tracking) | Fixed taped square for placement; camera only for static 2D/depth check or live dashboard stream |
| Cartesian impedance pushing & three-cup shuffles | Single vertical pick-and-place routine |
| Deep learning & neural network training on-site (CNNs/Transformers) | Hand-crafted signal features (mass, FFT peak, spectral energy, phase lag) → 8–12 dim vector |
| Multi-step active Bayesian planning in real time | Single, deterministic 3-in-1 diagnostic macro every time |
