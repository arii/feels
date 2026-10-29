# Viam Module Config — FeelsControlService

Paste this as the **JSON attributes** block when adding the `control-logic` service
in the Viam app (Configuration → Services → Add).

## Service attributes

```json
{
  "arm":    "arm",
  "gripper": "gripper",
  "camera": "cam",
  "vision": "yolo",

  "sampling_rate_hz": 50,
  "lever_arm_m": 0.45,
  "tare_j3_nm": -16.821,

  "cup_x_mm":       300,
  "cup_y_mm":       0,
  "cup_grasp_z_mm": 150,
  "approach_z_mm":  310,
  "lift_height_mm": 420,

  "shake_amplitude_mm": 20,
  "shake_freq_hz": 2.5
}
```

## Tuning the cup pose

The values above are defaults. Before the first run:

1. Jog the arm manually (Viam app → Control) to the taped square.
2. Record the end-effector pose (x, y, z in mm).
3. Update `cup_x_mm`, `cup_y_mm`, `cup_grasp_z_mm` to match.
4. Set `approach_z_mm` ≈ `cup_grasp_z_mm + 160` (clears the cup rim).
5. Set `lift_height_mm` ≈ `cup_grasp_z_mm + 270` (comfortable shake height).

## Motion sequence (run_macro)

```
detect_cup  → vision confirm (skipped if shape-detector not wired)
pick_cup    → open gripper → hover at approach_z → descend to cup_grasp_z
            → grab → lift to lift_height
─── recording starts here ──────────────────────────────────────────────
Phase 1     → steady hold 1.5 s           (baseline torques)
Phase 2     → ±20 mm X-axis sine 2.5 s    (rattle / slosh)
Phase 3     → 25° tilt + restore 2.0 s    (inertia / liquid dynamics)
─── recording ends ─────────────────────────────────────────────────────
place_cup   → lower to cup_grasp_z → open → retract to approach_z
features    → 10-dim FFT extraction → Atlas identify()
```

## Service dependencies to declare in Viam

| Name           | Type          |
|----------------|---------------|
| arm            | Arm           |
| gripper        | Gripper       |
| cam            | Camera        |
| yolo           | VisionService |
