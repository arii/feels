#!/usr/bin/env python3
"""
torque_service.py — Viam Generic Service module: proprioceptive TORQUE FEATURES.

A reusable on-robot sensing module for ANY load-sensing arm. It polls joint
torques at a configurable rate into a ring buffer and exposes, over do_command:

    {"action": "telemetry"}         latest torques + pose-normalized force + est. weight
    {"action": "features"}          time + FFT features over the recorded window
    {"action": "effective_force"}   pose-normalized vertical force F_z (gravity-cancelled by reach)
    {"action": "tare"}              recalibrate the empty baseline
    {"action": "record_start"} / {"action": "record_stop"} / {"action": "record_data"}
    {"action": "status"}

Why pose-normalized? Joint torque depends on arm configuration; dividing the
shoulder torque by the TCP reach r_xy = sqrt(x²+y²) cancels how far the arm is
reaching, giving a configuration-robust effective vertical force (and thus a
payload/weight estimate). See analysis/effective_force.py for the physics.

── Machine config (Viam app → Services → Add → this model) ───────────────────
{
  "model": "braveorangerover:sensing:torque-features", "type": "generic",
  "attributes": {
    "arm": "arm",
    "sampling_rate_hz": 50,
    "lever_arm_m": 0.45,
    "tare_j2_nm": 43.57, "tare_j3_nm": -16.82,
    "force_joint": 2
  }
}
Self-contained: needs only the `arm` dependency (no camera / cloud / frame system).
"""
from collections import deque
from typing import ClassVar, Mapping, Optional, Sequence, Tuple
import asyncio
import math
import time

import numpy as np
from typing_extensions import Self
from viam.components.arm import Arm
from viam.module.module import Module
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic as GenericService
from viam.utils import struct_to_dict, ValueTypes
from viam.logging import getLogger

logger = getLogger("TorqueFeatures")
GRAVITY = 9.80665


class TorqueFeatures(GenericService, EasyResource):
    """On-robot proprioceptive torque sampling + feature extraction."""

    MODEL: ClassVar[Model] = Model(ModelFamily("braveorangerover", "sensing"), "torque-features")

    arm: Arm
    sampling_rate_hz: float = 50.0
    lever_arm_m: float = 0.45
    tare_j2_nm: float = 43.57
    tare_j3_nm: float = -16.82
    force_joint: int = 2          # 1-based joint used for the effective-force estimate (2 = shoulder)

    _running: bool = False
    _task: Optional[asyncio.Task] = None
    _buf: deque
    _rec: list
    _recording: bool = False

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Tuple[Sequence[str], Sequence[str]]:
        attrs = struct_to_dict(config.attributes)
        if not attrs.get("arm"):
            raise ValueError("attribute 'arm' (component name) is required")
        return [attrs["arm"]], []

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        self = super().new(config, dependencies)
        a = struct_to_dict(config.attributes)
        self.arm = dependencies[Arm.get_resource_name(a["arm"])]
        self.sampling_rate_hz = float(a.get("sampling_rate_hz", 50.0))
        self.lever_arm_m = float(a.get("lever_arm_m", 0.45))
        self.tare_j2_nm = float(a.get("tare_j2_nm", 43.57))
        self.tare_j3_nm = float(a.get("tare_j3_nm", -16.82))
        self.force_joint = int(a.get("force_joint", 2))
        self._buf = deque(maxlen=int(a.get("buffer_size", 2000)))
        self._rec = []
        self._recording = False
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(f"TorqueFeatures up @ {self.sampling_rate_hz} Hz on arm '{a['arm']}'")
        return self

    async def close(self):
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    # ── high-rate sampling loop ───────────────────────────────────────────────
    async def _poll_loop(self):
        interval = 1.0 / max(1.0, self.sampling_rate_hz)
        while self._running:
            t0 = time.perf_counter()
            try:
                res = await self.arm.do_command({"load": ""}, timeout=0.5)
                loads = res.get("load", []) if isinstance(res, dict) else []
                loads = [float(x) for x in loads[:6]]
                if len(loads) >= 6:
                    s = {"t": time.time(), "loads": loads}
                    self._buf.append(s)
                    if self._recording:
                        self._rec.append(s)
            except Exception as e:
                logger.debug(f"load read error: {e}")
            await asyncio.sleep(max(0.001, interval - (time.perf_counter() - t0)))

    async def _effective_force(self) -> dict:
        """Pose-normalized vertical force from the shoulder/elbow torque."""
        if not self._buf:
            return {"status": "buffering"}
        loads = self._buf[-1]["loads"]
        try:
            pose = await self.arm.get_end_position()
            r_xy = max(0.05, math.sqrt(pose.x ** 2 + pose.y ** 2) / 1000.0)
        except Exception:
            r_xy = self.lever_arm_m
        j = self.force_joint - 1
        tare = self.tare_j2_nm if self.force_joint == 2 else self.tare_j3_nm
        dtau = loads[j] - tare
        f_z = -dtau / r_xy
        return {
            "effective_force_n": round(f_z, 3),
            "weight_grams": round((f_z / GRAVITY) * 1000.0, 1),
            "lever_arm_m": round(r_xy, 4),
            "delta_torque_nm": round(dtau, 4),
            "force_joint": self.force_joint,
        }

    # ── RPC ───────────────────────────────────────────────────────────────────
    async def do_command(self, command: Mapping[str, ValueTypes], *,
                         timeout: Optional[float] = None, **kwargs) -> Mapping[str, ValueTypes]:
        action = str(command.get("action", "")).lower()

        if action == "status":
            return {"running": self._running, "sampling_rate_hz": self.sampling_rate_hz,
                    "buffer_fill": len(self._buf), "recording": self._recording}

        if action == "telemetry":
            if not self._buf:
                return {"status": "buffering"}
            latest = self._buf[-1]
            return {"status": "active", "timestamp": latest["t"],
                    "torques_nm": [round(x, 4) for x in latest["loads"]],
                    **(await self._effective_force())}

        if action == "effective_force":
            return await self._effective_force()

        if action == "tare":
            if len(self._buf) < 5:
                return {"status": "error", "message": "not enough samples"}
            recent = np.array([s["loads"] for s in list(self._buf)[-20:]])
            self.tare_j2_nm = float(np.mean(recent[:, 1]))
            self.tare_j3_nm = float(np.mean(recent[:, 2]))
            return {"status": "ok", "tare_j2_nm": round(self.tare_j2_nm, 4),
                    "tare_j3_nm": round(self.tare_j3_nm, 4)}

        if action == "record_start":
            self._rec = []
            self._recording = True
            return {"status": "recording"}
        if action == "record_stop":
            self._recording = False
            dur = (self._rec[-1]["t"] - self._rec[0]["t"]) if len(self._rec) > 1 else 0.0
            return {"status": "stopped", "samples": len(self._rec), "duration_s": round(dur, 2),
                    "effective_hz": round(len(self._rec) / max(0.1, dur), 1)}
        if action == "record_data":
            return {"count": len(self._rec),
                    "timestamps": [s["t"] for s in self._rec],
                    "torques_nm": [s["loads"] for s in self._rec]}

        if action == "features":
            return self._compute_features()

        return {"error": f"unknown action '{action}'",
                "available": ["telemetry", "features", "effective_force", "tare",
                              "record_start", "record_stop", "record_data", "status"]}

    def _compute_features(self) -> dict:
        """Time + frequency features over the recorded window (or the live buffer)."""
        src = self._rec if len(self._rec) >= 20 else list(self._buf)
        if len(src) < 20:
            return {"error": "need >= 20 samples (record_start → move → record_stop, or wait for buffer)"}
        ts = np.array([s["t"] for s in src])
        L = np.array([s["loads"] for s in src])
        dt = float(np.mean(np.diff(ts)))
        fs = 1.0 / max(1e-3, dt)
        j3 = L[:, 2]
        dtau3 = j3 - self.tare_j3_nm

        # FFT on J3 residual (drop DC)
        n = len(j3)
        fft = np.abs(np.fft.rfft(j3 - j3.mean()))
        freqs = np.fft.rfftfreq(n, d=dt)
        dom_f = dom_p = rattle = slosh = 0.0
        if len(freqs) > 1:
            fft, freqs = fft[1:], freqs[1:]
            k = int(np.argmax(fft)); dom_f, dom_p = float(freqs[k]), float(fft[k])
            rattle = float(np.sum(fft[(freqs >= 5) & (freqs <= 25)]))
            slosh = float(np.sum(fft[(freqs >= 0.5) & (freqs <= 3)]))

        return {
            "status": "success", "samples": n, "fs_hz": round(fs, 1),
            "mean_delta_tau3_nm": round(float(np.mean(dtau3)), 4),
            "peak_delta_tau3_nm": round(float(np.min(dtau3)), 4),
            "std_tau3_nm": round(float(np.std(j3)), 4),
            "ptp_tau3_nm": round(float(np.ptp(j3)), 4),
            "dominant_freq_hz": round(dom_f, 2),
            "rattle_energy_5_25hz": round(rattle, 3),
            "slosh_energy_05_3hz": round(slosh, 3),
            "est_mass_grams": round(abs(float(np.mean(dtau3))) / self.lever_arm_m / GRAVITY * 1000.0, 1),
            "nyquist_hz": round(fs / 2.0, 2),
            "note": "rattle band (5-25Hz) only observable if fs >= 50Hz; cloud polling (~5Hz) cannot see it",
        }


if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
