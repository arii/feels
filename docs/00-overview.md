# Feels Hackathon — Project Overview

> [!IMPORTANT]
> **Key Architecture Decisions & Scope Clarifications:**
> 1. **Fixed Motion Routine — Pick Up, Swirl, and Place:** All experiments use a single, deterministic pick-up, horizontal swirl excitation, and placement sequence executed from a fixed taped square.
> 2. **No Pushing or Impedance Manipulation:** We do not push, slide, or shuffle cups across the table. This avoids non-holonomic friction slippage, table surface irregularities, and cup tipping.
> 3. **No Shake or Tilt:** High-frequency linear sine shaking and aggressive angular tilting are eliminated. Swirling provides smooth, continuous centripetal and tangential excitation to differentiate sloshing liquids from tumbling granular solids and rigid masses without spilling.
> 4. **No Dynamic Vision Grasping:** Vision-guided visual servoing is replaced with fixed, mechanically repeatable arm-position-saver waypoints on a marked square.
> 5. **No Deep Learning Training On-Site:** We do not train CNNs or Transformers during the hackathon. We use deterministic, hand-crafted physical signal features mapped to a normalized metric embedding.
> 6. **Primary Focus:** High-reliability **MongoDB Atlas feature extraction** + **Operator UX** for live telemetry visualization, detection reporting, and online dataset recording/teaching.

---

## Links & Resources

- Event: https://luma.com/ta40197s?tk=ZtkRNO
- Mongo Learn Robot Hackathon: mdb.link/learn-robot-hackathon
- GitHub Repo: https://github.com/arii/feels
- Notebook: https://notebook.google.com/notebook/437061ae-a2d6-4709-96a8-21211aef4b83
- Viam Python SDK Docs: https://python.viam.dev/

---

## Problem Statement

Use **proprioception** (actuator torques, end-effector load, and arm kinematics) to identify what is inside a covered, lidded coffee cup (water, viscous soap, loose candy, rigid weight, or empty). The contents are occluded from vision — but by executing a fixed **pick up $\to$ swirl $\to$ place** routine, the robot extracts distinctive physical dynamic signatures and matches them via **MongoDB Atlas Vector Search**.

### Core Information-Gathering Routine
- **Fixed Pick Up:** Approach and grasp cup at fixed taped square; vertical lift to transit altitude.
- **Controlled Swirl:** Smooth circular/orbital horizontal swirl excitation to induce fluid vortex vs. solid rattle dynamics.
- **Fixed Place:** Controlled descent to place cup back on target square, open gripper, and return home.

*(Archived: Pushing, multi-cup games, violent sine shaking, angular tilt sweeps, and audio).*

### Object Properties to Identify

**Liquid**
- Low viscosity Newtonian fluid (water — dynamic swirl slosh)
- High viscosity non-Newtonian fluid (dish soap — heavy viscous swirl damping)

**Dry Object**
- Discrete loose solids (Halloween candy fun size — sharp tumbling chatter)
- Solid rigid weight (rigid coupling, high mass, zero swirl lag)

**Baseline / Empty**
- Empty cup (baseline tare mass, zero internal dynamics)

---

## System Diagram (Deterministic Swirl & Retrieval Pipeline)

```
[Taped Table Square]
        ↓
[Standard Approach & Grasp] (Fixed Waypoint)
        ↓
[Vertical Lift & Static Hold] (Steady-state mass Δτ_gravity)
        ↓
[Orbital Swirl Excitation] (Fluid vortex lag, granular tumble chatter, torque variance)
        ↓
[Safe Descent & Placement] (Descend, release gripper, return home)
        ↓
[Feature Extraction Engine] (Computes 10D physical vector z & normalized embedding e)
        ↓
[MongoDB Atlas $vectorSearch] (Cosine k-NN retrieval against 'signatures' collection)
        ↓
[Web Dashboard UX] (Live detection report, confidence score, and 1-click teach/record)
```

---

## Hardware & System Setup

### Robot
- **Robot name:** sfhack12
- **Arm:** UFactory xArm x850 (6-DOF)
- **Gripper:** UFactory parallel-jaw gripper
- **Camera:** Intel RealSense D435 (monitoring / dashboard visual verification)

### Viam Server / Agent Versions
- viam-server: 1.14.3
- viam-agent: 1.2.0
- OS: linux/amd64
- Remote address: `sfhack12-main.ty1cywh7tq.viam.cloud`
