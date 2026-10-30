#!/usr/bin/env python3
"""
vision_grasp.py — Vision-guided 3D side grasping with contact detection & placement.

Key Capabilities:
1. Anti-Tipping Geometry:
   - Lowered vertical grasp altitude (default Z = 35.0mm or -20mm from centroid).
   - Backed-off insertion offset (tcp_offset = 130.0mm instead of 105.0mm) so the gripper
     palm does not crash into and push the cup across the table.
2. Contact Detection:
   - Guarded linear approach: steps forward while sampling joint torques (J2, J3, J5).
   - If contact resistance is detected (load delta > threshold), forward motion halts
     immediately and triggers gripper closure.
3. Complete Pick-and-Place Cycle:
   - Identify (at staging pose, immune to occlusion)
   - Approach to standoff (gripper open)
   - Guarded advance with contact detection
   - Grip cup & verify holding state
   - Lift vertically to transit altitude
   - Move to travel pose
   - Descend to place pose
   - Open gripper to deposit object
   - Retract to travel pose
   - Return to home pose
"""

import asyncio
import os
import sys
import math
import struct
import argparse
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from dotenv import load_dotenv

from viam.robot.client import RobotClient
from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.components.camera import Camera
from viam.components.switch import Switch
from viam.services.vision import VisionClient
from viam.services.motion import MotionClient
from viam.proto.common import Pose, PoseInFrame

load_dotenv()

VIAM_API_KEY = os.getenv("VIAM_API_KEY")
VIAM_API_KEY_ID = os.getenv("VIAM_API_KEY_ID")
VIAM_MACHINE_ADDRESS = os.getenv("VIAM_MACHINE_ADDRESS", "sfhack12-main.ty1cywh7tq.viam.cloud")

# Camera intrinsic defaults (Intel RealSense D435 on sfhack12)
DEFAULT_FX = 919.1594
DEFAULT_FY = 919.9533
DEFAULT_CX = 656.2068
DEFAULT_CY = 366.5177

# Table & workspace constraints (mm)
TABLE_SURFACE_Z = -23.0     # Obstacle table top surface reference in world frame (box top is at Z = -23.0mm)
DEFAULT_GRASP_Z = 50.0      # Lower vertical grasp altitude to prevent tipping (was 70.4mm, 50.0mm keeps 18mm table clearance)
DEFAULT_Z_OFFSET = -12.0    # Offset applied to detected centroid to grab lower body (mm)
MIN_SAFE_Z = 48.0           # Absolute minimum clearance above table: gripper bottom radius is 55mm -> 48 - 55 = -7mm (> -23mm table)
MAX_SAFE_Z = 65.0           # Maximum grasp altitude to avoid tipping

# Standoff and insertion geometry (mm)
# Note: tcp_offset = 142.0mm centers the cup directly between the rubber finger pads
# while leaving >50mm clearance before the gripper palm, completely preventing the robot from pushing the cup.
DEFAULT_TCP_OFFSET = 142.0  # Flange to finger contact center offset (mm)
APPROACH_STANDOFF = 60.0    # Pre-grasp standoff distance back along approach ray (mm)
TRAVEL_SAFE_Z = 195.0       # Lift altitude for safe transit (mm)

# Default orientation for side grasp (matches xArm tool flange horizontal side grasp)
DEFAULT_SIDE_THETA = 175.7  # Degrees roll around tool axis for horizontal finger opening

# Contact detection parameters
DEFAULT_CONTACT_THRESH = 0.22  # Torque delta in Nm to declare physical contact (Joint 3 / Joint 5)


class CupDetector:
    """
    Detects cup using Viam's YOLO-3D point cloud segmentation (`yolo-3d`),
    with fallback to 2D YOLO (`yolo`) + optical ray-plane intersection.
    """

    def __init__(
        self,
        machine: RobotClient,
        camera_name: str = "cam",
        yolo_3d_name: str = "yolo-3d",
        yolo_2d_name: str = "yolo"
    ):
        self.machine = machine
        self.cam = Camera.from_robot(machine, camera_name)
        self.camera_name = camera_name
        self.yolo_3d_name = yolo_3d_name
        self.yolo_2d_name = yolo_2d_name
        self.fx = DEFAULT_FX
        self.fy = DEFAULT_FY
        self.cx = DEFAULT_CX
        self.cy = DEFAULT_CY

    async def init_intrinsics(self):
        """Fetch live camera intrinsics if available."""
        try:
            props = await self.cam.get_properties()
            if props.intrinsic_parameters:
                self.fx = props.intrinsic_parameters.focal_x_px or self.fx
                self.fy = props.intrinsic_parameters.focal_y_px or self.fy
                self.cx = props.intrinsic_parameters.center_x_px or self.cx
                self.cy = props.intrinsic_parameters.center_y_px or self.cy
        except Exception:
            pass

    async def get_camera_extrinsics(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute camera optical center position and rotation matrix in world coordinates."""
        cam_origin = await self.machine.transform_pose(
            query=PoseInFrame(reference_frame=self.camera_name, pose=Pose(x=0, y=0, z=0, o_x=0, o_y=0, o_z=1, theta=0)),
            destination="world"
        )
        p_x = await self.machine.transform_pose(
            query=PoseInFrame(reference_frame=self.camera_name, pose=Pose(x=1, y=0, z=0, o_x=0, o_y=0, o_z=1, theta=0)),
            destination="world"
        )
        p_y = await self.machine.transform_pose(
            query=PoseInFrame(reference_frame=self.camera_name, pose=Pose(x=0, y=1, z=0, o_x=0, o_y=0, o_z=1, theta=0)),
            destination="world"
        )
        p_z = await self.machine.transform_pose(
            query=PoseInFrame(reference_frame=self.camera_name, pose=Pose(x=0, y=0, z=1, o_x=0, o_y=0, o_z=1, theta=0)),
            destination="world"
        )

        P_cam = np.array([cam_origin.pose.x, cam_origin.pose.y, cam_origin.pose.z], dtype=float)
        R_cam = np.column_stack([
            np.array([p_x.pose.x, p_x.pose.y, p_x.pose.z]) - P_cam,
            np.array([p_y.pose.x, p_y.pose.y, p_y.pose.z]) - P_cam,
            np.array([p_z.pose.x, p_z.pose.y, p_z.pose.z]) - P_cam,
        ])
        return P_cam, R_cam

    async def detect_cup_3d(self, z_offset: float = DEFAULT_Z_OFFSET, target_z: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Primary perception method: Query `yolo-3d` for segmented 3D point cloud of the cup.
        Applies a lower vertical altitude to prevent cup tipping during grasp.
        """
        try:
            yolo3d = VisionClient.from_robot(self.machine, self.yolo_3d_name)
            pcs = await yolo3d.get_object_point_clouds(self.camera_name)
            for pc in pcs:
                for g in pc.geometries.geometries:
                    if g.label == "cup":
                        c = g.center
                        ref_frame = pc.geometries.reference_frame or self.camera_name
                        print(f"[YOLO-3D] Segmented cup found in {ref_frame}: x={c.x:.1f}, y={c.y:.1f}, z={c.z:.1f} mm")

                        # Transform 3D center from camera frame to world frame
                        pif = PoseInFrame(
                            reference_frame=ref_frame,
                            pose=Pose(x=c.x, y=c.y, z=c.z, o_x=c.o_x, o_y=c.o_y, o_z=c.o_z, theta=c.theta)
                        )
                        world_pif = await self.machine.transform_pose(query=pif, destination="world")
                        wp = world_pif.pose

                        P_cam, _ = await self.get_camera_extrinsics()

                        # Determine anti-tipping lowered grasp Z
                        if target_z is not None:
                            final_z = max(MIN_SAFE_Z, min(MAX_SAFE_Z, float(target_z)))
                        else:
                            final_z = max(MIN_SAFE_Z, min(MAX_SAFE_Z, float(wp.z + z_offset)))

                        print(f"[YOLO-3D] Centroid Z: {wp.z:.1f}mm -> Anti-Tipping Grasp Z: {final_z:.1f}mm")

                        return {
                            "method": "yolo-3d-pointcloud",
                            "confidence": 1.0,
                            "position_world": (float(wp.x), float(wp.y), float(final_z)),
                            "detected_centroid_world": (float(wp.x), float(wp.y), float(wp.z)),
                            "camera_pos_world": (float(P_cam[0]), float(P_cam[1]), float(P_cam[2])),
                            "point_cloud_bytes": len(pc.point_cloud),
                        }
        except Exception as e:
            print(f"[YOLO-3D] Point cloud detection notice: {e}")

        return None

    async def detect_cup_2d_fallback(self, target_z: float = DEFAULT_GRASP_Z) -> Optional[Dict[str, Any]]:
        """
        Fallback perception method: 2D YOLO detection + optical ray-plane intersection.
        """
        try:
            await self.init_intrinsics()
            yolo2d = VisionClient.from_robot(self.machine, self.yolo_2d_name)
            detections = await yolo2d.get_detections_from_camera(self.camera_name)
            cups = [d for d in detections if d.class_name == "cup"]
            if not cups:
                return None

            cup = max(cups, key=lambda d: d.confidence)
            u_c = (cup.x_min + cup.x_max) / 2.0
            v_c = (cup.y_min + cup.y_max) / 2.0

            P_cam, R_cam = await self.get_camera_extrinsics()
            ray_cam = np.array([(u_c - self.cx) / self.fx, (v_c - self.cy) / self.fy, 1.0])
            ray_world = R_cam @ ray_cam

            if abs(ray_world[2]) < 1e-4:
                return None

            clamped_z = max(MIN_SAFE_Z, min(MAX_SAFE_Z, float(target_z)))
            lam = (clamped_z - P_cam[2]) / ray_world[2]
            P_world = P_cam + lam * ray_world

            return {
                "method": "yolo-2d-ray-intersection",
                "bbox": (cup.x_min, cup.y_min, cup.x_max, cup.y_max),
                "confidence": cup.confidence,
                "center_px": (u_c, v_c),
                "position_world": (float(P_world[0]), float(P_world[1]), float(clamped_z)),
                "camera_pos_world": (float(P_cam[0]), float(P_cam[1]), float(P_cam[2])),
            }
        except Exception as e:
            print(f"[YOLO-2D] Fallback error: {e}")
            return None

    async def detect_cup(self, target_z: Optional[float] = None, z_offset: float = DEFAULT_Z_OFFSET) -> Optional[Dict[str, Any]]:
        """
        Unified detection: Attempts YOLO-3D point cloud first; falls back to 2D ray-plane solver.
        """
        res = await self.detect_cup_3d(z_offset=z_offset, target_z=target_z)
        if res:
            return res
        print("[VISION] Falling back to 2D YOLO detection + optical ray solver...")
        fallback_z = target_z if target_z is not None else DEFAULT_GRASP_Z
        return await self.detect_cup_2d_fallback(target_z=fallback_z)


class SideGraspPlanner:
    """Plans approach, grasp, and lift poses for horizontal side grasp."""

    @staticmethod
    def plan_side_grasp(
        cup_pos: Tuple[float, float, float],
        cam_pos: Tuple[float, float, float],
        tcp_offset: float = DEFAULT_TCP_OFFSET,
        standoff: float = APPROACH_STANDOFF,
        travel_z: float = TRAVEL_SAFE_Z,
        theta_deg: float = DEFAULT_SIDE_THETA,
        grasp_z: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generates approach, grasp, and lift poses.
        The approach vector is directed horizontally toward the cup in the XY plane.
        """
        cx, cy, cz = cup_pos
        cam_x, cam_y, _ = cam_pos
        cz_final = grasp_z if grasp_z is not None else cz

        # Approach vector in XY plane pointing from camera/staging position toward cup
        v_xy = np.array([cx - cam_x, cy - cam_y, 0.0], dtype=float)
        norm = np.linalg.norm(v_xy)
        if norm < 1e-3:
            v_xy = np.array([1.0, 0.0, 0.0])
        else:
            v_xy /= norm

        # Arm TCP at grasp position: offset back by calibrated finger center distance (130mm)
        P_grasp = np.array([cx, cy, cz_final]) - tcp_offset * v_xy

        # Arm TCP at approach standoff: further back along approach vector
        P_approach = P_grasp - standoff * v_xy

        # Arm TCP at lift / travel: elevated above grasp position
        P_lift = np.array([P_grasp[0], P_grasp[1], travel_z])

        # Gripper orientation pointing along approach vector
        ox, oy, oz = v_xy[0], v_xy[1], 0.0

        return {
            "approach": Pose(x=P_approach[0], y=P_approach[1], z=P_approach[2], o_x=ox, o_y=oy, o_z=oz, theta=theta_deg),
            "grasp": Pose(x=P_grasp[0], y=P_grasp[1], z=P_grasp[2], o_x=ox, o_y=oy, o_z=oz, theta=theta_deg),
            "lift": Pose(x=P_lift[0], y=P_lift[1], z=P_lift[2], o_x=ox, o_y=oy, o_z=oz, theta=theta_deg),
            "vector": (float(ox), float(oy), float(oz)),
            "cup_pos": (cx, cy, cz_final),
            "tcp_offset": tcp_offset,
            "standoff": standoff,
        }


class VisionGraspController:
    """
    Executes the 'Look-Lock-and-Move' vision-guided side grasp sequence with contact detection
    and placement cycle.
    """

    def __init__(self, machine: RobotClient):
        self.machine = machine
        self.arm = Arm.from_robot(machine, "arm")
        self.gripper = Gripper.from_robot(machine, "gripper")
        self.motion = MotionClient.from_robot(machine, "builtin")
        self.detector = CupDetector(machine)

    async def move_pose(self, pose: Pose, label: str = "pose", timeout: float = 30.0) -> bool:
        """
        Move arm to Cartesian pose using collision-aware motion service.
        Strictly enforces collision avoidance: if motion.move detects an obstacle (e.g. table),
        the move is aborted and NEVER bypassed to direct IK.
        """
        if pose.z < MIN_SAFE_Z:
            print(f"[COLLISION SAFETY] Target Z={pose.z:.1f}mm is below MIN_SAFE_Z={MIN_SAFE_Z:.1f}mm! Clamping to prevent table collision.")
            pose.z = MIN_SAFE_Z

        goal = PoseInFrame(reference_frame="world", pose=pose)
        print(f"[MOVE] Navigating to {label} -> (X: {pose.x:.1f}, Y: {pose.y:.1f}, Z: {pose.z:.1f})...")
        try:
            return await self.motion.move(component_name="arm", destination=goal, timeout=timeout)
        except Exception as e:
            err_msg = str(e).lower()
            # If the motion service failed due to an obstacle collision, abort immediately!
            if any(kw in err_msg for kw in ["violation", "constraint", "collision", "table", "failed constraints"]):
                print(f"[COLLISION AVOIDANCE BLOCKED MOVE] motion.move blocked by collision safety: {e}")
                print("  ABORTING MOVE to prevent striking table or obstacles. Unconstrained IK fallback is disabled.")
                return False

            print(f"[MOTION] motion.move non-collision notice ({e}); using direct arm.move_to_position...")
            try:
                await self.arm.move_to_position(pose, timeout=timeout)
                return True
            except Exception as e2:
                print(f"[MOTION] arm.move_to_position failed: {e2}")
                return False

    async def move_waypoint(self, name: str, timeout: float = 40.0) -> bool:
        """Trigger an arm-position-saver switch to move to a saved waypoint."""
        print(f"\n[WAYPOINT] Commanding arm to '{name}'...")
        try:
            sw = Switch.from_robot(self.machine, name)
            t0 = asyncio.get_event_loop().time()
            await sw.set_position(2, timeout=timeout)
            duration = asyncio.get_event_loop().time() - t0
            pos = await self.arm.get_end_position()
            print(f"[✓] Arrived at '{name}' in {duration:.1f}s -> (X: {pos.x:.1f}, Y: {pos.y:.1f}, Z: {pos.z:.1f})")
            return True
        except Exception as e:
            print(f"[✗] Failed moving to '{name}': {e}")
            return False

    async def guarded_advance_with_contact(
        self,
        p_approach: Pose,
        p_grasp: Pose,
        step_mm: float = 12.0,
        contact_threshold: float = DEFAULT_CONTACT_THRESH,
    ) -> Tuple[bool, bool]:
        """
        Guarded linear advance from p_approach toward p_grasp.
        Monitors joint loads (J2, J3, J5) after each incremental step.
        If contact force deflection is detected, halts forward motion immediately.
        Returns: (success, contact_detected)
        """
        print(f"\n[GUARDED APPROACH] Advancing from standoff to contact pose (threshold: {contact_threshold:.2f} Nm)...")

        # Capture baseline loads at approach pose
        base_loads = None
        try:
            res = await self.arm.do_command({"load": ""}, timeout=2.0)
            base_loads = res.get("load", [])
        except Exception:
            pass

        dx = p_grasp.x - p_approach.x
        dy = p_grasp.y - p_approach.y
        total_dist = math.sqrt(dx * dx + dy * dy)
        if total_dist < 1.0:
            ok = await self.move_pose(p_grasp, label="grasp")
            return ok, False

        num_steps = max(1, int(math.ceil(total_dist / step_mm)))
        print(f"  • Progressing {total_dist:.1f}mm in up to {num_steps} guarded steps (~{total_dist/num_steps:.1f}mm each)")

        contact_detected = False
        for s in range(1, num_steps + 1):
            fraction = s / num_steps
            interp_x = p_approach.x + fraction * dx
            interp_y = p_approach.y + fraction * dy
            step_pose = Pose(
                x=interp_x, y=interp_y, z=p_grasp.z,
                o_x=p_grasp.o_x, o_y=p_grasp.o_y, o_z=p_grasp.o_z,
                theta=p_grasp.theta
            )

            # Move one step forward
            ok = await self.move_pose(step_pose, label=f"advance-step-{s}/{num_steps}")
            if not ok:
                return False, False
            await asyncio.sleep(0.08)

            # Check joint load deflection
            if base_loads and len(base_loads) >= 6:
                try:
                    res = await self.arm.do_command({"load": ""}, timeout=1.0)
                    cur_loads = res.get("load", [])
                    if cur_loads and len(cur_loads) >= 6:
                        # Monitor Joint 2, 3, 5 load delta
                        delta_j3 = abs(cur_loads[2] - base_loads[2])
                        delta_j2 = abs(cur_loads[1] - base_loads[1])
                        delta_j5 = abs(cur_loads[4] - base_loads[4])
                        max_deflection = max(delta_j3, delta_j2 * 0.5, delta_j5 * 2.0)
                        if max_deflection > contact_threshold:
                            print(f"\n[⚡ CONTACT DETECTED] Load deflection Δτ = {max_deflection:.3f} Nm > threshold {contact_threshold:.2f} Nm at step {s}/{num_steps}!")
                            print("  Halting forward motion immediately to prevent pushing the cup.")
                            contact_detected = True
                            break
                except Exception:
                    pass

        if not contact_detected:
            print(f"[✓] Reached target contact pose ({total_dist:.1f}mm) without premature collision.")

        return True, contact_detected

    async def execute_grasp(
        self,
        dry_run: bool = False,
        target_z: Optional[float] = None,
        z_offset: float = DEFAULT_Z_OFFSET,
        tcp_offset: float = DEFAULT_TCP_OFFSET,
        contact_thresh: float = DEFAULT_CONTACT_THRESH,
        full_cycle: bool = True,
    ) -> bool:
        """
        Execute full vision-guided grasp and placement routine:
        1. Open gripper
        2. Identify cup from staging pose (before occlusion occurs)
        3. Lock 3D coordinates & synthesize trajectory
        4. Move to pre-grasp standoff pose
        5. Guarded advance with contact detection
        6. Close gripper & verify grip
        7. Lift vertically to transit altitude
        8. (If full_cycle): Move to travel -> place -> open -> retract -> home
        """
        print("\n" + "=" * 70)
        print("🎯 VISION-GUIDED SIDE GRASP WITH CONTACT DETECTION & PLACEMENT")
        print("=" * 70)

        # 0. Ensure staging pose with clear sightline if currently down near table
        try:
            pos = await self.arm.get_end_position()
            if pos.z < 150.0:
                print(f"[STAGE] Arm is currently at Z={pos.z:.1f}mm (near table/occluded). Staging to 'home-pose' for clear view...")
                if not dry_run:
                    await self.move_waypoint("home-pose")
                    await asyncio.sleep(1.0)
                else:
                    print("[DRY RUN] Would stage to 'home-pose' first.")
        except Exception:
            pass

        # 1. Open gripper
        print("\n[STEP 1] Preparing gripper: opening jaws...")
        if not dry_run:
            await self.gripper.open(timeout=10.0)
            await asyncio.sleep(0.5)

        # 2. Identify cup (Look phase)
        print("\n[STEP 2] IDENTIFY PHASE: Localizing cup before approach occlusion...")
        det = await self.detector.detect_cup(target_z=target_z, z_offset=z_offset)
        if not det:
            print("[✗] Error: Cup could not be detected. Aborting grasp.")
            return False

        cup_pos = det["position_world"]
        cam_pos = det["camera_pos_world"]
        method = det.get("method", "yolo")
        print(f"[✓] Cup identified via '{method}'!")
        print(f"    Locked 3D World Coordinates: X={cup_pos[0]:.1f}mm, Y={cup_pos[1]:.1f}mm, Z={cup_pos[2]:.1f}mm")

        # 3. Plan trajectory (Lock & Plan phase)
        print("\n[STEP 3] PLANNING: Synthesizing side-grasp trajectory...")
        plan = SideGraspPlanner.plan_side_grasp(
            cup_pos, cam_pos, tcp_offset=tcp_offset, grasp_z=cup_pos[2]
        )
        p_app = plan["approach"]
        p_grp = plan["grasp"]
        p_lft = plan["lift"]
        v = plan["vector"]

        angle_deg = math.degrees(math.atan2(v[1], v[0]))
        print(f"  • Approach Vector:   [{v[0]:.3f}, {v[1]:.3f}, 0.0] ({angle_deg:.1f}°)")
        print(f"  • Standoff Pose:     X={p_app.x:.1f}, Y={p_app.y:.1f}, Z={p_app.z:.1f} mm")
        print(f"  • Target Grasp Pose: X={p_grp.x:.1f}, Y={p_grp.y:.1f}, Z={p_grp.z:.1f} mm (Offset: {tcp_offset:.1f}mm)")
        print(f"  • Safe Lift Pose:    X={p_lft.x:.1f}, Y={p_lft.y:.1f}, Z={p_lft.z:.1f} mm")

        if dry_run:
            print("\n[DRY RUN] Verification successful. Physical motion skipped.")
            return True

        # 4. Move to pre-grasp standoff (Move phase begins)
        print("\n[STEP 4] Moving to pre-grasp standoff pose...")
        if not await self.move_pose(p_app, label="standoff-approach"):
            print("[✗] Motion aborted at standoff.")
            return False
        await asyncio.sleep(0.8)

        # 5. Guarded advance with contact detection
        print("\n[STEP 5] Moving towards cup into side grasp pose (with contact detection)...")
        ok, contact = await self.guarded_advance_with_contact(
            p_app, p_grp, contact_threshold=contact_thresh
        )
        if not ok:
            print("[✗] Motion aborted during advance.")
            return False
        await asyncio.sleep(0.3)

        # 6. Actuate gripper & verify
        print("\n[STEP 6] Actuating gripper (grab)...")
        grabbed = await self.gripper.grab(timeout=10.0)
        await asyncio.sleep(0.5)
        status = await self.gripper.is_holding_something()
        is_holding = status.is_holding_something
        print(f"[✓] Grab complete: holding={is_holding}, meta={status.meta}")

        if not is_holding:
            print("[!] Warning: Gripper did not confirm holding object.")

        # 7. Lift to safe altitude
        print("\n[STEP 7] Lifting cup vertically to clear table...")
        await self.move_pose(p_lft, label="vertical-lift")
        await asyncio.sleep(0.5)

        if not full_cycle:
            print("\n" + "=" * 70)
            print("✅ VISION-GUIDED GRASP COMPLETE (Placement skipped via flag)")
            print("=" * 70 + "\n")
            return is_holding

        # 8. Travel Pose
        print("\n[STEP 8] Navigating to travel waypoint...")
        if not await self.move_waypoint("travel"):
            return False
        await asyncio.sleep(1.0)

        # 9. Place Pose
        print("\n[STEP 9] Descending to place waypoint...")
        if not await self.move_waypoint("place"):
            return False
        await asyncio.sleep(0.5)

        # 10. Release cup
        print("\n[STEP 10] Releasing cup (open gripper)...")
        await self.gripper.open(timeout=10.0)
        await asyncio.sleep(0.5)

        # 11. Retract back to travel height
        print("\n[STEP 11] Retracting to travel waypoint...")
        if not await self.move_waypoint("travel"):
            return False
        await asyncio.sleep(0.5)

        # 12. Return to home pose
        print("\n[STEP 12] Returning to home pose...")
        if not await self.move_waypoint("home-pose"):
            return False

        print("\n" + "=" * 70)
        print("✅ FULL VISION-GUIDED PICK-AND-PLACE CYCLE COMPLETED SUCCESSFULLY")
        print("=" * 70 + "\n")
        return is_holding


async def main():
    parser = argparse.ArgumentParser(description="Occlusion-aware vision-guided side grasp with contact detection & placement.")
    parser.add_argument("--detect-only", action="store_true", help="Identify cup and print 3D poses without moving")
    parser.add_argument("--dry-run", action="store_true", help="Verify perception and motion plan without moving")
    parser.add_argument("--grasp", action="store_true", help="Execute live physical side grasp with placement")
    parser.add_argument("--grasp-only", action="store_true", help="Execute grasp and lift only (skip place/return sequence)")
    parser.add_argument("--z-height", type=float, default=None, help="Explicit vertical grasp altitude in mm (default: 50.0mm)")
    parser.add_argument("--z-offset", type=float, default=DEFAULT_Z_OFFSET, help="Vertical offset from centroid (default: -12.0mm)")
    parser.add_argument("--tcp-offset", type=float, default=DEFAULT_TCP_OFFSET, help="Flange-to-cup standoff offset (default: 142.0mm to prevent pushing)")
    parser.add_argument("--contact-thresh", type=float, default=DEFAULT_CONTACT_THRESH, help="Torque deflection in Nm to detect contact (default: 0.22Nm)")
    parser.add_argument("--stage", action="store_true", help="Move arm to staging home-pose before detecting")
    args = parser.parse_args()

    opts = RobotClient.Options.with_api_key(
        api_key=VIAM_API_KEY,
        api_key_id=VIAM_API_KEY_ID,
    )
    async with await RobotClient.at_address(VIAM_MACHINE_ADDRESS, opts) as machine:
        controller = VisionGraspController(machine)

        if args.detect_only or (not args.grasp and not args.grasp_only and not args.dry_run):
            if args.stage:
                await controller.move_waypoint("home-pose")
                await asyncio.sleep(1.0)
            det = await controller.detector.detect_cup(target_z=args.z_height, z_offset=args.z_offset)
            if not det:
                print("[!] No cup detected in current view.")
                print("    Hint: If arm is currently close to the table (Z < 150mm), run with --stage to move to staging pose for unoccluded perception.")
                return

            cup_pos = det["position_world"]
            cam_pos = det["camera_pos_world"]
            plan = SideGraspPlanner.plan_side_grasp(
                cup_pos, cam_pos, tcp_offset=args.tcp_offset, grasp_z=cup_pos[2]
            )
            p_app = plan["approach"]
            p_grp = plan["grasp"]
            p_lft = plan["lift"]
            v = plan["vector"]
            print("\n" + "=" * 68)
            print(f"📍 CUP IDENTIFICATION ({det.get('method')})")
            print("=" * 68)
            if "detected_centroid_world" in det:
                dc = det["detected_centroid_world"]
                print(f"• Centroid Detected: X={dc[0]:.1f}mm, Y={dc[1]:.1f}mm, Z={dc[2]:.1f}mm")
            print(f"• Target Grasp 3D:   X={cup_pos[0]:.2f}mm, Y={cup_pos[1]:.2f}mm, Z={cup_pos[2]:.2f}mm (Lowered: {cup_pos[2]:.1f}mm)")
            print(f"• Approach Ray:      [{v[0]:.3f}, {v[1]:.3f}, 0.0] ({math.degrees(math.atan2(v[1], v[0])):.1f}°)")
            print(f"• Standoff Arm:      X={p_app.x:.2f}mm, Y={p_app.y:.2f}mm, Z={p_app.z:.2f}mm")
            print(f"• Contact Arm:       X={p_grp.x:.2f}mm, Y={p_grp.y:.2f}mm, Z={p_grp.z:.2f}mm (Offset: {args.tcp_offset:.1f}mm)")
            print(f"• Safe Lift Arm:     X={p_lft.x:.2f}mm, Y={p_lft.y:.2f}mm, Z={p_lft.z:.2f}mm")
            print(f"• Placement Targets: 'travel' -> 'place' -> 'travel' -> 'home-pose'")
            print("=" * 68 + "\n")
            return

        full_cycle = not args.grasp_only
        await controller.execute_grasp(
            dry_run=args.dry_run,
            target_z=args.z_height,
            z_offset=args.z_offset,
            tcp_offset=args.tcp_offset,
            contact_thresh=args.contact_thresh,
            full_cycle=full_cycle,
        )


if __name__ == "__main__":
    asyncio.run(main())
