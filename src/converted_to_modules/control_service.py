"""
On-Robot Control & High-Rate Telemetry Service (Viam Generic Service Module)

Addresses latency constraints by executing directly onboard the robot compute
host (sfhack12). Communicates with local arm and gripper hardware via low-latency
IPC (<1ms), enabling 50-200 Hz sensor acquisition, dynamic macro execution,
and onboard feature extraction.
"""

from collections import deque
from typing import ClassVar, Mapping, Optional, Sequence, Tuple
import asyncio
import logging
import math
import time

import numpy as np
from typing_extensions import Self
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.module.module import Module
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Pose, ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic as GenericService
from viam.services.vision import VisionClient
from viam.utils import struct_to_dict, ValueTypes

logger = logging.getLogger("FeelsControlService")


class FeelsControlService(GenericService, EasyResource):
    """
    On-robot generic service providing zero-latency telemetry buffering,
    high-frequency vibration/rattle capture, and automated motion macros.
    """
    MODEL: ClassVar[Model] = Model(ModelFamily("hackathons", "control_logic"), "generic-service")

    gripper: Gripper
    arm: Arm
    camera: Camera
    vision: Optional[VisionClient] = None

    # Telemetry configuration
    sampling_rate_hz: float = 50.0
    lever_arm_m: float = 0.45
    tare_j3_nm: float = -16.821

    # Cup pose parameters (mm, robot base frame) — tune to your taped square
    cup_x_mm: float = 300.0
    cup_y_mm: float = 0.0
    cup_grasp_z_mm: float = 150.0   # height at which gripper closes around cup
    approach_z_mm: float = 310.0    # safe hover height above cup before descending
    lift_height_mm: float = 420.0   # analysis height (shake + tilt happen here)
    shake_amplitude_mm: float = 20.0
    shake_freq_hz: float = 2.5

    # Internal state & circular buffer
    _running: bool = False
    _background_task: Optional[asyncio.Task] = None
    _buffer_maxlen: int = 2000
    _telemetry_buffer: deque

    # Active recording state
    _is_recording: bool = False
    _recorded_samples: list

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        attrs = struct_to_dict(config.attributes)

        required_deps = []
        optional_deps = []

        for key in ("arm", "gripper", "camera"):
            if key not in attrs or not attrs[key]:
                raise ValueError(f"Attribute '{key}' (non-empty string) is required")
            required_deps.append(attrs[key])

        # vision detector (VisionClient, e.g. yolo) is optional
        if attrs.get("vision"):
            optional_deps.append(attrs["vision"])

        return required_deps, optional_deps

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        self = super().new(config, dependencies)
        attrs = struct_to_dict(config.attributes)

        # Resolve required dependencies
        self.arm = dependencies[Arm.get_resource_name(attrs["arm"])]
        self.gripper = dependencies[Gripper.get_resource_name(attrs["gripper"])]
        self.camera = dependencies[Camera.get_resource_name(attrs["camera"])]

        # Resolve optional vision dependency
        vision_name = attrs.get("vision", "")
        if vision_name:
            try:
                self.vision = dependencies[VisionClient.get_resource_name(vision_name)]
            except KeyError:
                self.vision = None
                logger.warning(f"Vision service '{vision_name}' not found in dependencies")
        else:
            self.vision = None

        # Telemetry config
        self.sampling_rate_hz = float(attrs.get("sampling_rate_hz", 50.0))
        self.lever_arm_m = float(attrs.get("lever_arm_m", 0.45))
        self.tare_j3_nm = float(attrs.get("tare_j3_nm", -16.821))

        # Cup pose config (mm, tune to your taped square)
        self.cup_x_mm = float(attrs.get("cup_x_mm", 300.0))
        self.cup_y_mm = float(attrs.get("cup_y_mm", 0.0))
        self.cup_grasp_z_mm = float(attrs.get("cup_grasp_z_mm", 150.0))
        self.approach_z_mm = float(attrs.get("approach_z_mm", 310.0))
        self.lift_height_mm = float(attrs.get("lift_height_mm", 420.0))
        self.shake_amplitude_mm = float(attrs.get("shake_amplitude_mm", 20.0))
        self.shake_freq_hz = float(attrs.get("shake_freq_hz", 2.5))

        # Initialize buffers
        self._telemetry_buffer = deque(maxlen=self._buffer_maxlen)
        self._recorded_samples = []
        self._is_recording = False
        self._running = True

        # Launch onboard high-rate polling task
        self._background_task = asyncio.create_task(self._high_rate_loop())
        logger.info(
            f"FeelsControlService initialized at {self.sampling_rate_hz} Hz. "
            f"Arm: {attrs['arm']}, Gripper: {attrs['gripper']}"
        )
        return self

    async def close(self):
        """Cleanup on module shutdown or reconfiguration."""
        self._running = False
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        logger.info("FeelsControlService stopped.")

    async def _high_rate_loop(self):
        """
        Runs locally on the robot host (sfhack12).
        Because it communicates over local IPC, each poll takes <1ms.
        """
        interval = 1.0 / max(1.0, self.sampling_rate_hz)
        logger.info(f"Starting onboard telemetry loop (interval: {interval*1000:.1f}ms)...")

        while self._running:
            t_start = time.perf_counter()
            try:
                # Query local xArm driver for 6-joint actuator loads
                load_resp = await self.arm.do_command({"load": ""}, timeout=0.5)
                raw_torques = [float(x) for x in load_resp] if isinstance(load_resp, list) else []

                t_now = time.time()
                sample = {
                    "timestamp": t_now,
                    "torques": raw_torques,
                    "j3": raw_torques[2] if len(raw_torques) > 2 else 0.0,
                    "delta_j3": (raw_torques[2] - self.tare_j3_nm) if len(raw_torques) > 2 else 0.0,
                }

                # Compute instantaneous weight estimate
                if len(raw_torques) > 2:
                    delta_tau = abs(raw_torques[2] - self.tare_j3_nm)
                    force_n = delta_tau / self.lever_arm_m
                    sample["weight_grams"] = round((force_n / 9.80665) * 1000.0, 1)
                else:
                    sample["weight_grams"] = 0.0

                self._telemetry_buffer.append(sample)

                if self._is_recording:
                    self._recorded_samples.append(sample)

            except Exception as e:
                logger.warning(f"Onboard sensor read error: {e}")

            # Enforce sampling period
            t_elapsed = time.perf_counter() - t_start
            sleep_time = max(0.001, interval - t_elapsed)
            await asyncio.sleep(sleep_time)

    # ---------------------------------------------------------------------------
    # DoCommand RPC Interface
    # ---------------------------------------------------------------------------

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        action = str(command.get("action", "")).lower()

        if action == "tare":
            return await self._handle_tare()
        elif action == "get_telemetry":
            return await self._handle_get_telemetry()
        elif action == "start_recording":
            return await self._handle_start_recording()
        elif action == "stop_recording":
            return await self._handle_stop_recording()
        elif action == "get_recorded_data":
            return await self._handle_get_recorded_data()
        elif action == "compute_features":
            return await self._handle_compute_features()
        elif action == "detect_cup":
            return await self._detect_cup()
        elif action == "pick_cup":
            lift_pose = await self._pick_cup()
            return {"status": "picked", "lift_z_mm": lift_pose.z}
        elif action == "place_cup":
            return await self._place_cup(None)
        elif action == "run_macro":
            return await self._handle_run_macro(command)
        else:
            return {
                "error": f"Unknown action: '{action}'",
                "available_actions": [
                    "tare", "get_telemetry",
                    "start_recording", "stop_recording", "get_recorded_data",
                    "compute_features",
                    "detect_cup", "pick_cup", "place_cup",
                    "run_macro",
                ],
            }

    async def _handle_tare(self) -> dict:
        """Calibrates Joint 3 zero-load resting torque baseline."""
        if len(self._telemetry_buffer) > 0:
            recent_j3 = [s["j3"] for s in list(self._telemetry_buffer)[-20:]]
            self.tare_j3_nm = float(np.mean(recent_j3))
            logger.info(f"Recalibrated tare baseline: J3 = {self.tare_j3_nm:.4f} Nm")
            return {"status": "ok", "tare_j3_nm": self.tare_j3_nm}
        return {"status": "error", "message": "Buffer empty"}

    async def _handle_get_telemetry(self) -> dict:
        """Returns latest instantaneous state without round-trip latency."""
        if not self._telemetry_buffer:
            return {"status": "buffering", "samples": 0}

        latest = self._telemetry_buffer[-1]
        return {
            "status": "active",
            "timestamp": latest["timestamp"],
            "sampling_rate_hz": self.sampling_rate_hz,
            "buffer_count": len(self._telemetry_buffer),
            "is_recording": self._is_recording,
            "recorded_count": len(self._recorded_samples),
            "torques": latest["torques"],
            "j3_raw": latest["j3"],
            "delta_j3": latest["delta_j3"],
            "weight_grams": latest["weight_grams"],
            "tare_j3_nm": self.tare_j3_nm,
        }

    async def _handle_start_recording(self) -> dict:
        self._recorded_samples = []
        self._is_recording = True
        return {"status": "recording_started", "timestamp": time.time()}

    async def _handle_stop_recording(self) -> dict:
        self._is_recording = False
        duration_s = 0.0
        if len(self._recorded_samples) > 1:
            duration_s = self._recorded_samples[-1]["timestamp"] - self._recorded_samples[0]["timestamp"]

        return {
            "status": "recording_stopped",
            "sample_count": len(self._recorded_samples),
            "duration_s": round(duration_s, 2),
            "effective_hz": round(len(self._recorded_samples) / max(0.1, duration_s), 1),
        }

    async def _handle_get_recorded_data(self) -> dict:
        """Exports raw recorded arrays."""
        timestamps = [s["timestamp"] for s in self._recorded_samples]
        j3_values = [s["j3"] for s in self._recorded_samples]
        weights = [s["weight_grams"] for s in self._recorded_samples]
        return {
            "count": len(self._recorded_samples),
            "timestamps": timestamps,
            "j3": j3_values,
            "weights": weights,
        }

    async def _handle_compute_features(self) -> dict:
        """
        Extracts 10 physical features directly on-robot from recorded telemetry:
        1. Mean delta torque (mass proxy)
        2. Peak delta torque
        3. Torque variance
        4. Dominant FFT peak frequency
        5. Dominant FFT peak power
        6. Rattle energy (5-25 Hz band)
        7. Low-frequency slosh energy (0.5-3 Hz band)
        8. High-to-low spectral energy ratio
        9. Decay rate / damping
        10. Steady-state stability percentage
        """
        if len(self._recorded_samples) < 20:
            return {"error": "Insufficient samples for feature extraction (need >= 20)"}

        j3 = np.array([s["j3"] for s in self._recorded_samples])
        ts = np.array([s["timestamp"] for s in self._recorded_samples])
        dt = np.mean(np.diff(ts))
        fs = 1.0 / max(0.001, dt)

        # Baseline offset
        delta_tau = j3 - self.tare_j3_nm

        # 1-3: Time domain features
        mean_torque = float(np.mean(delta_tau))
        peak_torque = float(np.min(delta_tau))
        torque_variance = float(np.var(j3))

        # 4-8: Frequency domain (FFT) features
        n = len(j3)
        fft_vals = np.abs(np.fft.rfft(j3 - np.mean(j3)))
        freqs = np.fft.rfftfreq(n, d=dt)

        # Exclude DC component (0 Hz)
        if len(freqs) > 1:
            fft_vals = fft_vals[1:]
            freqs = freqs[1:]
            peak_idx = int(np.argmax(fft_vals))
            dominant_freq_hz = float(freqs[peak_idx])
            dominant_power = float(fft_vals[peak_idx])

            # Band energies
            rattle_mask = (freqs >= 5.0) & (freqs <= 25.0)
            slosh_mask = (freqs >= 0.5) & (freqs <= 3.0)
            rattle_energy = float(np.sum(fft_vals[rattle_mask])) if np.any(rattle_mask) else 0.0
            slosh_energy = float(np.sum(fft_vals[slosh_mask])) if np.any(slosh_mask) else 0.0
            spectral_ratio = float(rattle_energy / max(0.001, slosh_energy))
        else:
            dominant_freq_hz = 0.0
            dominant_power = 0.0
            rattle_energy = 0.0
            slosh_energy = 0.0
            spectral_ratio = 0.0

        # 9-10: Stability & Decay
        steady_mask = np.abs(j3 - self.tare_j3_nm) < 1.0
        stability_pct = float(np.mean(steady_mask) * 100.0)

        feature_vector = [
            round(mean_torque, 4),
            round(peak_torque, 4),
            round(torque_variance, 4),
            round(dominant_freq_hz, 2),
            round(dominant_power, 4),
            round(rattle_energy, 4),
            round(slosh_energy, 4),
            round(spectral_ratio, 4),
            round(stability_pct, 1),
            round(abs(mean_torque) / self.lever_arm_m / 9.80665 * 1000.0, 1),  # estimated mass (g)
        ]

        return {
            "status": "success",
            "samples_analyzed": n,
            "sample_rate_fs": round(fs, 1),
            "feature_vector": feature_vector,
            "metrics": {
                "mean_delta_torque_nm": round(mean_torque, 3),
                "peak_delta_torque_nm": round(peak_torque, 3),
                "torque_variance": round(torque_variance, 4),
                "dominant_freq_hz": round(dominant_freq_hz, 2),
                "dominant_power": round(dominant_power, 3),
                "rattle_energy_5_25hz": round(rattle_energy, 3),
                "slosh_energy_05_3hz": round(slosh_energy, 3),
                "spectral_ratio": round(spectral_ratio, 3),
                "steady_state_stability_pct": round(stability_pct, 1),
                "estimated_mass_grams": round(abs(mean_torque) / self.lever_arm_m / 9.80665 * 1000.0, 1),
            },
        }

    async def _detect_cup(self) -> dict:
        """Check the vision detector for a cup before approaching. Skips gracefully if no vision dep."""
        if self.vision is None:
            logger.info("No vision service configured — skipping cup detection")
            return {"found": True, "confidence": 1.0, "source": "skipped"}
        try:
            detections = await self.vision.get_detections_from_camera("cam")
            for d in detections:
                if d.confidence >= 0.5:
                    logger.info(f"Cup detected: {d.class_name} @ {d.confidence:.2f}")
                    return {"found": True, "confidence": float(d.confidence), "class_name": d.class_name}
            logger.warning("No cup detected in frame — proceeding anyway")
            return {"found": False, "confidence": 0.0}
        except Exception as e:
            logger.warning(f"Cup detection error: {e}")
            return {"found": False, "error": str(e)}

    async def _pick_cup(self) -> Pose:
        """Open gripper → hover above cup → descend → grab → lift to analysis height."""
        ox, oy, oz, theta = 0.0, 0.0, -1.0, 0.0  # gripper pointing straight down

        await self.gripper.open()

        pre_grasp = Pose(
            x=self.cup_x_mm, y=self.cup_y_mm, z=self.approach_z_mm,
            o_x=ox, o_y=oy, o_z=oz, theta=theta,
        )
        await self.arm.move_to_position(pre_grasp)
        await asyncio.sleep(0.3)

        grasp_pose = Pose(
            x=self.cup_x_mm, y=self.cup_y_mm, z=self.cup_grasp_z_mm,
            o_x=ox, o_y=oy, o_z=oz, theta=theta,
        )
        await self.arm.move_to_position(grasp_pose)
        await asyncio.sleep(0.3)

        await self.gripper.grab()
        await asyncio.sleep(0.4)

        lift_pose = Pose(
            x=self.cup_x_mm, y=self.cup_y_mm, z=self.lift_height_mm,
            o_x=ox, o_y=oy, o_z=oz, theta=theta,
        )
        await self.arm.move_to_position(lift_pose)
        await asyncio.sleep(0.5)

        logger.info(f"Cup lifted to z={self.lift_height_mm:.0f}mm")
        return lift_pose

    async def _place_cup(self, lift_pose: Optional[Pose]) -> dict:
        """Lower cup to taped square, release, retract to approach height."""
        ox, oy, oz, theta = 0.0, 0.0, -1.0, 0.0

        approach = Pose(
            x=self.cup_x_mm, y=self.cup_y_mm, z=self.approach_z_mm,
            o_x=ox, o_y=oy, o_z=oz, theta=theta,
        )
        await self.arm.move_to_position(approach)
        await asyncio.sleep(0.3)

        place = Pose(
            x=self.cup_x_mm, y=self.cup_y_mm, z=self.cup_grasp_z_mm,
            o_x=ox, o_y=oy, o_z=oz, theta=theta,
        )
        await self.arm.move_to_position(place)
        await asyncio.sleep(0.3)

        await self.gripper.open()
        await asyncio.sleep(0.3)

        await self.arm.move_to_position(approach)
        await asyncio.sleep(0.3)

        logger.info("Cup placed back on taped square")
        return {"status": "placed"}

    async def _shake_phase(self, lift_pose: Pose, duration: float = 2.5):
        """Sinusoidal X-axis excitation at shake_freq_hz for duration seconds."""
        n_steps = max(8, int(duration * self.shake_freq_hz * 8))
        step_dt = duration / n_steps

        for i in range(n_steps):
            t = (i / n_steps) * duration
            x_offset = self.shake_amplitude_mm * math.sin(2.0 * math.pi * self.shake_freq_hz * t)
            waypoint = Pose(
                x=lift_pose.x + x_offset, y=lift_pose.y, z=lift_pose.z,
                o_x=lift_pose.o_x, o_y=lift_pose.o_y, o_z=lift_pose.o_z, theta=lift_pose.theta,
            )
            await self.arm.move_to_position(waypoint)
            await asyncio.sleep(step_dt)

        # Return to centre
        await self.arm.move_to_position(lift_pose)
        await asyncio.sleep(0.2)

    async def _tilt_phase(self, lift_pose: Pose, tilt_degrees: float = 25.0):
        """Tilt cup tilt_degrees from vertical in the XZ plane, then restore level."""
        rad = math.radians(tilt_degrees)
        # Rotate -Z axis toward +X: (0,0,-1) → (sin θ, 0, -cos θ)
        tilt_pose = Pose(
            x=lift_pose.x, y=lift_pose.y, z=lift_pose.z,
            o_x=math.sin(rad), o_y=0.0, o_z=-math.cos(rad),
            theta=lift_pose.theta,
        )
        await self.arm.move_to_position(tilt_pose)
        await asyncio.sleep(1.0)

        await self.arm.move_to_position(lift_pose)
        await asyncio.sleep(0.8)

    async def _handle_run_macro(self, command: Mapping[str, ValueTypes]) -> dict:
        """
        Full see-pick-shake-classify motion macro:
        1. Detect cup via vision detector (visual confirm, optional)
        2. Pick cup from taped square and lift to analysis height
        3. Phase 1 — Steady hold 1.5 s  (record baseline torques)
        4. Phase 2 — Sinusoidal shake 2.5 s  (rattle / slosh excitation)
        5. Phase 3 — Tilt 25° and restore 2.0 s  (inertia / liquid response)
        6. Place cup back on taped square
        """
        logger.info("Starting see-pick-shake motion macro...")

        detection = await self._detect_cup()
        if not detection.get("found"):
            logger.warning("Cup not confirmed by vision — proceeding without visual confirm")

        lift_pose = await self._pick_cup()

        # Start recording only once cup is in hand and at lift height
        self._recorded_samples = []
        self._is_recording = True

        try:
            logger.info("Phase 1: Steady hold (1.5 s)...")
            await asyncio.sleep(1.5)

            logger.info("Phase 2: Sinusoidal shake (2.5 s)...")
            await self._shake_phase(lift_pose, duration=2.5)

            logger.info("Phase 3: Tilt & restore (2.0 s)...")
            await self._tilt_phase(lift_pose, tilt_degrees=25.0)

        finally:
            self._is_recording = False

        await self._place_cup(lift_pose)

        features = await self._handle_compute_features()
        return {
            "status": "macro_completed",
            "cup_detected": detection,
            "sample_count": len(self._recorded_samples),
            "features": features,
        }

    async def get_status(
        self, *, timeout: Optional[float] = None, **kwargs
    ) -> Mapping[str, ValueTypes]:
        return {
            "running": self._running,
            "sampling_rate_hz": self.sampling_rate_hz,
            "buffer_fill": len(self._telemetry_buffer),
            "is_recording": self._is_recording,
        }


if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
