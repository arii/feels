# Hackathon Task Schedule

## Day 1: Thursday, 11:30 AM to 10:00 PM

**Goal:** Day 1 ends with the full loop working from a terminal: pick, shake, store, identify.
The 1:30 PM go/no-go test decides whether the signal is real before anything else is built on it.

---

### 11:30 AM – 12:30 PM: Foundations

#### Robotics
- [x] Connect with the Viam Python SDK; read joint positions; verify gripper and camera
- [x] Build and launch real-time 6-joint visualizer with tare calibration and CSV recording (`realtime_plot.py`)
- [x] Complete preliminary signal verification: proved $\Delta \tau = -6.20\,\text{N}\cdot\text{m}$ load reaction on Joint 3 (Recordings 1–4)
- [ ] Tape a square for the cup. Find a side grasp that holds a lidded cup firmly without crushing it (lower gripper force/speed)
- [ ] Posture optimization: Reposition arm to cantilever angle ($q_2 \in [45^\circ, 60^\circ]$) to maximize torque sensitivity on both shoulder and elbow
- [ ] On-robot data daemon: Deploy high-rate sampling (100–200 Hz via socket port 30000 on `sfhack12`)
- [ ] Simulation: Test URDF/PyBullet/Viam mock for offline trajectory validation and collision avoidance
- [ ] Alternative controller investigation: Evaluate velocity control and Cartesian impedance/compliance modes

#### AI
- [ ] Create Atlas collections: `runs` (raw arrays, one document per run), `signatures` (features + embedding), `items` (label, description, photo)
- [ ] Write a fake-run generator in the contract format: water rings and decays slowly, soap decays fast, candy has sharp spikes
- [ ] Feature extractor v1: the 10 hand-crafted features from the event guide (lift torque, shake variance, FFT peak frequency and power, 5–25 Hz rattle energy, post-stop decay rate, tilt hysteresis, grip change)

---

### 12:30 PM – 1:30 PM: Go/No-Go Signal Test

#### Robotics
- [ ] With the cup placed in the gripper by hand, run the shake-and-freeze motion 5 times each for water, dish soap, and candy at equal weight (use a kitchen scale)
- [ ] Save each run with its label

#### AI
- [ ] Plot post-stop torque per joint for all 15 runs on one figure
- [ ] Compute leave-one-out k-NN accuracy for both the hand-crafted features and the spectrum embedding

**Gate at 1:30 PM:** 80% or better across 3 classes = go. Below that, try currents instead of torque, a bigger shake, or another axis until 2:30 PM, then fall back (see Risks).

---

### 1:30 PM – 4:00 PM: Build the Halves

#### Robotics
- [ ] `run_macro()`: pick from the taped square, lift 15 cm, hold 1.5 s, shake 2.5 s, tilt 25° and back over 2 s, place back
- [ ] Write phase marks into each run. Start shake speed and acceleration low; raise them in steps

#### AI
- [ ] Create the Atlas vector index on `signatures.embedding` (cosine) with `label` and `session` as filter fields
- [ ] `teach()`: insert run + signature. `identify()`: `$vectorSearch` (k = 5) then `$group` by label; confidence = mean similarity of the winning label

---

### 4:00 PM – 6:00 PM: Integrate to MVP

#### Both
- [ ] Swap fake runs for real ones; fix format mismatches
- [ ] Teach 5 runs for each of the 5 items (about 25 runs, roughly 15 minutes)
- [ ] Blind test: one person fills, the other runs `identify`. Record the accuracy

**Checkpoint 6:00 PM:** The loop works end to end from the terminal.

---

### 7:00 PM – 9:00 PM: Strengthen

#### Robotics
- [ ] Wrap the port 30000 reader as a Viam sensor component so torque appears in the Viam app
- [ ] Mark fill lines on cups so every refill matches. Collect 10 more runs per item

#### AI
- [ ] Re-run the offline evaluation with the bigger dataset; keep the embedding that scores best
- [ ] Dashboard skeleton (Streamlit): Identify and Teach buttons, top 3 neighbours with similarity, torque overlay against neighbours

**Before leaving:** Record a backup video of one full successful run.

---

## Day 2: Friday, 8:30 AM to 3:30 PM

**Goal:** Day 2 turns the working loop into a show — a dashboard judges can read, live teaching, and three clean rehearsals. Code freezes at 2:00 PM.

---

### 8:30 AM – 9:00 AM: Morning Check

#### Both
- [ ] Re-tare with an empty cup; refill cups to their marked lines
- [ ] Run 5 blind tests. If accuracy dropped overnight, re-teach 3 fresh runs per item tagged `session: fri`, and filter the vector search to today's session

---

### 9:00 AM – 12:30 PM: The Show Layer

#### Robotics
- [ ] Wire Home, Release Grip, and Stop into one-click functions the dashboard can call
- [ ] Add slip detection: if gripper position changes during the shake, abort and retry
- [ ] Stretch (only if the loop is solid by 10:30 AM): find the cup with a YOLO "cup" detector through Viam's vision service instead of the taped square
- [ ] Run 20 back-to-back cycles to shake out reliability bugs

#### AI
- [ ] Finish the dashboard: camera feed, live torque trace, FFT chart, decision card (label, confidence bar, top 3 Atlas neighbours with similarity %)
- [ ] Drive dashboard updates from an Atlas Change Stream on `signatures`, so the screen reacts to the database
- [ ] Teach flow: type a label, press Teach, robot runs 3 shakes, signatures inserted; show a "saved to Atlas" message
- [ ] Low-confidence fallback: below 0.85, run one more shake on another axis and combine both results

**Checkpoint 12:30 PM:** Demo runs entirely from the dashboard.

---

### 12:30 PM – 2:00 PM: Rehearse and Fix

#### Both
- [ ] 3 full dress rehearsals of the demo script with a timer. Fix only what breaks
- [ ] Record a fresh backup video of a perfect run
- [ ] Pick the "new item" for live teaching and test it twice (paperclips or rice work well)

---

### 2:00 PM – 3:30 PM: Freeze and Prepare

#### Robotics
- [ ] Code freeze. Prepare 3 identical lidded cups plus the teaching item; check the arm is homed

#### AI
- [ ] Code freeze. Clear test clutter from Atlas (keep the reference signatures). Open the dashboard and Atlas UI on the demo screen
- [ ] Rehearse the 60-second pitch
