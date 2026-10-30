#!/usr/bin/env python3
"""
pick_service.py — Viam Generic Service module: a vision-guided OBJECT PICKER.

Deploy this as a Viam module and configure the target object class (default
"cup"). Then pick objects over `do_command`:

    do_command({"action": "detect"})                    # multi-view locate the object
    do_command({"action": "pick"})                      # detect -> grasp -> lift
    do_command({"action": "pick_place"})                # full pick-and-place cycle
    do_command({"action": "pick", "object": "bottle"})  # override the class per-call

It reuses the robust multi-view detector + contact-guarded side grasp from
`vision_grasp.py`, generalized so any YOLO-detectable class works — "say cup, it
picks cups; say bottle, it picks bottles."

── Machine config (Viam app → Services → Add → this model) ───────────────────
{
  "model": "braveorangerover:pick_logic:yolo3d-pick", "type": "generic",
  "attributes": {
    "arm": "arm", "gripper": "gripper", "camera": "cam", "motion": "builtin",
    "vision_3d": "yolo-3d", "vision_2d": "yolo",
    "target_class": "cup",
    "machine_address": "sfhack12-main.ty1cywh7tq.viam.cloud",
    "views": 5, "vantage_spread": 40.0,
    "tcp_offset": 142.0, "grasp_z": 50.0, "contact_thresh": 0.22
  }
}
Requires VIAM_API_KEY / VIAM_API_KEY_ID in the module environment (used to open a
client connection for frame-system transforms). Register with meta.json.
"""
import asyncio
import os
from typing import ClassVar, Mapping, Optional, Sequence, Tuple

from typing_extensions import Self
from viam.module.module import Module
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.robot.client import RobotClient
from viam.services.generic import Generic as GenericService
from viam.utils import struct_to_dict, ValueTypes
from viam.logging import getLogger

import vision_grasp as vg

logger = getLogger("ObjectPicker")


class ObjectPicker(GenericService, EasyResource):
    """Vision-guided object picker. Configure `target_class` and call `do_command`."""

    MODEL: ClassVar[Model] = Model(ModelFamily("braveorangerover", "pick_logic"), "yolo3d-pick")

    # ── config / state ───────────────────────────────────────────────────────
    target_class: str = "cup"
    machine_address: str = ""
    arm_name: str = "arm"
    gripper_name: str = "gripper"
    camera_name: str = "cam"
    motion_name: str = "builtin"
    vision_3d_name: str = "yolo-3d"
    vision_2d_name: str = "yolo"
    views: int = 5
    vantage_spread: float = 40.0
    tcp_offset: float = 142.0
    grasp_z: Optional[float] = None
    contact_thresh: float = 0.22
    _machine: Optional[RobotClient] = None

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Tuple[Sequence[str], Sequence[str]]:
        attrs = struct_to_dict(config.attributes)
        required, optional = [], []
        for key in ("arm", "gripper", "camera"):
            if not attrs.get(key):
                raise ValueError(f"attribute '{key}' (component name) is required")
            required.append(attrs[key])
        for key in ("motion", "vision_3d", "vision_2d"):
            if attrs.get(key):
                optional.append(attrs[key])
        return required, optional

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        self = super().new(config, dependencies)
        self.reconfigure(config, dependencies)
        return self

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]):
        a = struct_to_dict(config.attributes)
        self.arm_name = a.get("arm", "arm")
        self.gripper_name = a.get("gripper", "gripper")
        self.camera_name = a.get("camera", "cam")
        self.motion_name = a.get("motion", "builtin")
        self.vision_3d_name = a.get("vision_3d", "yolo-3d")
        self.vision_2d_name = a.get("vision_2d", "yolo")
        self.target_class = a.get("target_class", "cup")
        self.machine_address = a.get("machine_address", os.getenv("VIAM_MACHINE_ADDRESS", ""))
        self.views = int(a.get("views", 5))
        self.vantage_spread = float(a.get("vantage_spread", 40.0))
        self.tcp_offset = float(a.get("tcp_offset", 142.0))
        self.grasp_z = float(a["grasp_z"]) if a.get("grasp_z") is not None else None
        self.contact_thresh = float(a.get("contact_thresh", 0.22))
        self._machine = None
        logger.info(f"ObjectPicker configured: target='{self.target_class}', views={self.views}")

    # ── machine client (for frame-system transforms) ─────────────────────────
    async def _machine_client(self) -> RobotClient:
        if self._machine is not None:
            return self._machine
        key, kid = os.getenv("VIAM_API_KEY"), os.getenv("VIAM_API_KEY_ID")
        if not (key and kid and self.machine_address):
            raise RuntimeError("set VIAM_API_KEY / VIAM_API_KEY_ID in the module env and "
                               "'machine_address' in the config for frame-system transforms")
        opts = RobotClient.Options.with_api_key(api_key=key, api_key_id=kid)
        self._machine = await RobotClient.at_address(self.machine_address, opts)
        return self._machine

    def _controller(self, machine, target_class: str) -> "vg.VisionGraspController":
        return vg.VisionGraspController(
            machine, arm_name=self.arm_name, gripper_name=self.gripper_name,
            motion_name=self.motion_name, camera_name=self.camera_name,
            yolo_3d_name=self.vision_3d_name, yolo_2d_name=self.vision_2d_name,
            target_class=target_class)

    # ── RPC interface ─────────────────────────────────────────────────────────
    async def do_command(self, command: Mapping[str, ValueTypes], *,
                         timeout: Optional[float] = None, **kwargs) -> Mapping[str, ValueTypes]:
        action = str(command.get("action", "")).lower()
        target = str(command.get("object", self.target_class)) or self.target_class
        views = int(command.get("views", self.views))

        if action not in ("detect", "pick", "pick_place", "status"):
            return {"error": f"unknown action '{action}'",
                    "available": ["detect", "pick", "pick_place", "status"]}

        if action == "status":
            return {"target_class": self.target_class, "views": self.views,
                    "vision_3d": self.vision_3d_name, "vision_2d": self.vision_2d_name,
                    "ready": bool(os.getenv("VIAM_API_KEY") and self.machine_address)}

        machine = await self._machine_client()
        ctrl = self._controller(machine, target)

        if action == "detect":
            det = await ctrl.detect_cup_robust(
                n_views=views, vantage_spread=self.vantage_spread,
                target_z=self.grasp_z, dry_run=True)   # detect = no arm motion
            if not det:
                return {"found": False, "object": target}
            p = det["position_world"]
            return {"found": True, "object": target, "method": det.get("method"),
                    "x_mm": round(p[0], 1), "y_mm": round(p[1], 1), "z_mm": round(p[2], 1),
                    "n_views": det.get("n_views"), "spread_mm": det.get("spread_mm")}

        # pick / pick_place
        ok = await ctrl.execute_grasp(
            dry_run=False, target_z=self.grasp_z, tcp_offset=self.tcp_offset,
            contact_thresh=self.contact_thresh, full_cycle=(action == "pick_place"),
            views=views, vantage_spread=self.vantage_spread)
        return {"action": action, "object": target, "success": bool(ok)}

    async def close(self):
        if self._machine is not None:
            await self._machine.close()
            self._machine = None


if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
