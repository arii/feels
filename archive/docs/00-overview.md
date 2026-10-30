# Feels Hackathon — Project Overview

## Links & Resources

- Event: https://luma.com/ta40197s?tk=ZtkRNO
- Mongo Learn Robot Hackathon: mdb.link/learn-robot-hackathon
- GitHub Repo: https://github.com/arii/feels
- Notebook: https://notebook.google.com/notebook/437061ae-a2d6-4709-96a8-21211aef4b83
- Viam Python SDK Docs: https://python.viam.dev/

---

## Problem Statement

Use **proprioception + vision** to identify what the robot is holding inside a lidded coffee cup (liquid, toy, candy, partial fill, etc.). The cup is covered so we cannot see the contents — but we can pick it up, move through motions, and collect data (vision + mostly proprioception) to determine what is inside.

Ideally we set up **online training** to identify new items and then run the experiment.

### Information-Gathering Actions
- Pick up — top grasp
- Pick up — side grasp
- Pushing
- Waving through set motion (figure-8, up/down, jiggle)
- Potentially audio if it makes noise

### Object Properties to Identify

**Liquid**
- High viscosity (dish soap)
- Water

**Dry object**
- Halloween candy fun size
- Pencil
- Heavy item

**Capacity**
- Think of visual cross-section
- Vertical filling (full cup of water)
- Horizontal filling (candy uses small amount of horizontal shape)

---

## Formal Research Problem

**Interactive Dynamic Identification of Hidden Contents**

**Interactive perception** (active dynamic identification) on an unobservable payload θ* sealed inside an opaque container. The robot executes a sequence of dynamic exploratory motions (probing actions) and records multi-modal time-series signals — primarily proprioception (joint positions, velocities, torques/efforts, wrist wrench) along with external/wrist camera observations — to infer the latent physical properties and classify the object category.

### Problem Setting & Formulations
- **Simplified:** Training a classifier
- **Formal:** Interactive Bayesian identification of kinematic mechanisms (ref: Barragán, Patrick R. — *Interactive Bayesian identification of kinematic mechanisms*)

---

## System Diagram (Active Exploration Loop)

```
[Target Cup in Scene]
  Vision: Grasp Pose Detection via Viam
          ↓
[Initial Approach & Grasp]
  (Slip / Deformation Test)
          ↓
  ┌─────────────────────────────────────────────────────────────┐
  │                   Active Exploration Loop                   │
  │                                                             │
  │  1. Action      2. Trajectory   3. Data       4. Feature   │
  │  Selection:  →  Execution:   →  Collection: → Extraction: →│
  │  aₖ ~ argmax    Cartesian       High-rate       FFT phase  │
  │  Info Gain      Oscillation     Joint Torque/   lag,       │
  │                                 F/T Wrench      transient  │
  │                                                 spikes     │
  │                         ↓                                  │
  │                  5. Belief Update: p(θ | D₁:ₖ)            │
  └─────────────────────────────────────────────────────────────┘
          ↓ (Threshold certainty reached)
[Output: Object Class + Estimated Fill / Viscosity]
```

---

## Action Items (Initial Backlog)
- [ ] GitHub repo config setup
- [ ] Networking / get rough demo with pick-and-place task ASAP
- [ ] MongoDB data collection
- [ ] Creating set motions
- [ ] Training / interactive Bayesian analysis
- [ ] UX interaction website

---

## Hardware & System Setup

### Robot
- **Robot name:** sfhack12
- **Arm:** UFactory xArm x850
- **Camera:** RealSense

### Viam Server / Agent Versions
- viam-server: 1.14.3
- viam-agent: 1.2.0
- OS: linux/amd64
- Remote address: `sfhack12-main.ty1cywh7tq.viam.cloud`
