## Description
<!-- Provide a brief overview of changes introduced in this PR -->

## Physical AI Safety Checklist
All pull requests must satisfy these safety verifications prior to review and merge:

- [ ] **1. Simulation Tested**: Feature was executed and verified against mock models (`arm / fake`, `gripper / fake`).
- [ ] **2. Hard Z Boundary**: Confirmed no $Z$-coordinates below $15\text{ mm}$ (`MIN_SAFE_Z_MM = 15.0`).
- [ ] **3. Motion Routing & Obstacles**: All arm motions are routed through `MotionClient.move()` with a valid `WorldState` (e.g. table bounding box obstacle). No direct unconstrained `Arm.move_to_position` calls.
- [ ] **4. Context Managed Client Lifecycle**: All robot interactions are wrapped in persistent `async with` sessions to maintain the 2-second heartbeat and prevent actuator drops.
- [ ] **5. Timezone Compliance**: All timestamps in telemetry and database payloads use timezone-aware UTC datetime (`datetime.now(timezone.utc)`).
- [ ] **6. Two-Phase Grasping**: Grasp sequence employs pre-grasp offset ($Z + 100\text{ mm}$, downward tool orientation) followed by grip check (`await gripper.is_holding_something()`).
- [ ] **7. Team Hardware Lock**: Hardware lock confirmed with team before staging or merging for live robot runs.

## Testing Performed
<!-- Detail test commands run (e.g., `pytest`, `python scripts/test_simulated_grasp.py`) -->
- `pytest tests/`
- `python scripts/test_simulated_grasp.py`
