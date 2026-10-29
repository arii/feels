# feels

Lightweight physical AI and robotics repository powered by the [Viam Python SDK](https://github.com/viamrobotics/viam-python-sdk) and AI agent workflows.

---

## Overview

This repository is designed for rapid development and testing of robotic workflows with Viam. It includes:
- **`viam-python`** async client scripts to interface with machine hardware.
- **Agentic Skills & Plugins** from [`viam-devrel/agent-skills`](https://github.com/viam-devrel/agent-skills) to provide LLM agents (Claude Code, Antigravity, Cursor) with deep robotics SDK expertise.
- **Safety Directives** configured for collision avoidance, safe Z-boundaries, and heartbeat management.

### Target Machine & Components

- **Machine Address:** `sfhack12-main.ty1cywh7tq.viam.cloud`
- **Configured Components:**
  - `arm` (`Arm`): Robotic arm position and trajectory control
  - `cam` (`Camera`): Vision stream and image acquisition
  - `gripper` (`Gripper`): End-effector grasp and state detection
  - `table` (`Gripper`): Auxiliary fixture / table gripper
  - `ceiling` (`Gripper`): Overhead gripper mechanism

---

## Agentic Skills & Marketplace

This repo integrates with the official **Viam Agent Skills** repository: [`viam-devrel/agent-skills`](https://github.com/viam-devrel/agent-skills).

### 1. Claude Code Plugin Marketplace

To enable Viam agent skills inside **Claude Code**, run:

```bash
# Add the marketplace
/plugin marketplace add viam-devrel/agent-skills

# Install the Python SDK skill
/plugin install viam-python@viam-agent-skills

# (Optional) Install the full bundle or other skills
/plugin install viam-skills@viam-agent-skills
/plugin install viam-modules-fleet@viam-agent-skills
/plugin install local-viam-server@viam-agent-skills
```

### 2. Antigravity / Cursor Agents

- **Workspace Skill:** The repository includes `.agents/skills/viam-python/` containing verified SDK references and cheatsheets. Agents running in Antigravity or Cursor automatically discover these instructions.
- **CLI Plugin (agy):**
  ```bash
  agy plugin install /path/to/agent-skills/plugins/viam-python
  ```

### Available Skills in `viam-devrel/agent-skills`

| Skill | Description |
|---|---|
| `viam-python` | Async patterns, component/service APIs, module development, image transforms |
| `viam-go-motion-vision` | Motion planning, WorldState, frame system, and spatial math (Go) |
| `viam-modules-fleet` | CLI, module packaging, registry, fleet configuration |
| `local-viam-server` | Spin up a local `viam-server` linked to cloud machine parts |
| `viam-machine-config` | Read and deploy machine configuration via the Viam app API |
| `viam-ml` | Training data capture and model deployment |

---

## SDKs & Prerequisites

- **Python:** 3.10+ (Python 3.12 recommended)
- **Core SDKs:**
  - `viam-sdk` (`>=0.85.0`)
  - `python-dotenv` (`>=1.0.0`)
  - `pymongo` (`>=4.10.0`) *(optional, for Atlas telemetry & state persistence)*

---

## Getting Started

### 1. Virtual Environment Setup

Create and activate a Python virtual environment:

```bash
# Create venv
python3 -m venv .venv

# Activate venv (Linux/macOS)
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure Credentials

Copy the template environment file and fill in your machine credentials:

```bash
cp .env.example .env
```

Edit `.env`:
```env
VIAM_API_KEY=your_api_key_here
VIAM_API_KEY_ID=your_api_key_id_here
VIAM_MACHINE_ADDRESS=sfhack12-main.ty1cywh7tq.viam.cloud
```

### 3. Run the Hardware Inspection Script

Test connection and query active components:

```bash
python main.py
```

### 4. Interactive Real-Time Visualizer & Weight Detection Dashboard

Launch the 10-second rolling telemetry dashboard with separated 6-joint axes and 1-click session recording:

```bash
python realtime_plot.py --port 8080
```
Open **`http://localhost:8080`** in any browser.

### 5. Inspect All Sensors and Encoders

```bash
python sensors.py
```

### 6. Force & Weight Sensitivity Analysis

```bash
python effective_force.py --duration 15.0
```

---

## Experimental Findings: Weight Detection via Joint Torques

Even without a dedicated 6-axis F/T flange sensor, the xArm850 reports high-resolution joint loads via `arm.do_command({"load": ""})`.

1. **Posture Dynamics:**
   - In upright postures ($q_2 \approx 86^\circ$), the upper arm acts as an axial compression column, so vertical weight creates minimal angular moment on Joint 2 (Shoulder).
   - **Joint 3 (Elbow Pitch)** acts as the cantilever reaction pivot and absorbs the downward bending moment.
2. **Experimental Verification (Recordings 3 vs 4):**
   - **Empty / Resting State:** Joint 3 settles firmly to `-16.82 N·m`.
   - **Holding Payload:** Joint 3 deflects to **`-22.0` to `-23.02 N·m`** (a **`-6.20 N·m` shift**, equivalent to $\approx 1.4\,\text{kg}$ downward force).
   - When released, torque crisply snaps back from $-20.7 \to -16.82\,\text{N}\cdot\text{m}$.
   - CSV datasets and comparative plots are saved in [`recordings/`](recordings/).

---

## Baseline Code Sample

The baseline connection pattern in [`main.py`](main.py):

```python
import asyncio
from viam.robot.client import RobotClient
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.services.generic import Generic as GenericService

async def connect():
    opts = RobotClient.Options.with_api_key(
        api_key='<API-KEY>',
        api_key_id='<API-KEY-ID>'
    )
    return await RobotClient.at_address('sfhack12-main.ty1cywh7tq.viam.cloud', opts)

async def main():
    async with await connect() as machine:
        print('Resources:')
        print(machine.resource_names)
        
        # arm
        arm: Arm = Arm.from_robot(machine, "arm")
        arm_return_value = await arm.get_end_position()
        print(f"arm get_end_position return value: {arm_return_value}")

        # cam
        cam: Camera = Camera.from_robot(machine, "cam")
        cam_return_value = await cam.get_images()
        print(f"cam get_images return value: {cam_return_value}")

        # gripper
        gripper: Gripper = Gripper.from_robot(machine, "gripper")
        gripper_return_value = await gripper.is_moving()
        print(f"gripper is_moving return value: {gripper_return_value}")

        # table
        table: Gripper = Gripper.from_robot(machine, "table")
        table_return_value = await table.is_moving()
        print(f"table is_moving return value: {table_return_value}")

        # ceiling
        ceiling: Gripper = Gripper.from_robot(machine, "ceiling")
        ceiling_return_value = await ceiling.is_moving()
        print(f"ceiling is_moving return value: {ceiling_return_value}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Safety Directives

When writing code that moves physical hardware, follow the rules defined in [`.cursorrules`](.cursorrules):

1. **Client Lifecycle & Heartbeat:** Always use `async with await connect() as machine:` to maintain the 2-second heartbeat and prevent estop triggers.
2. **Motion Planning:** Use `MotionClient.move()` with a valid `WorldState` rather than unconstrained raw arm motion.
3. **Hard Z-Boundary:** Always enforce $Z \ge 15\text{ mm}$ (`MIN_SAFE_Z_MM = 15.0`) to prevent collisions with the work surface.
4. **Two-Phase Grasping:** Always plan a pre-grasp pose $100\text{ mm}$ above the target before descending, and verify grasp using `await gripper.is_holding_something()`.
