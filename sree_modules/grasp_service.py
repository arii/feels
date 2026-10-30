#!/usr/bin/env python3
"""
grasp_service.py — Viam Generic Service: deterministic repeatable grasp.

Self-contained: no subprocess, no external scripts. Implements the
pick → swirl → place macro directly using the injected arm/gripper
dependencies. Waypoints are configurable via module attributes.

    do_command({"action": "swirl", "object": "water_075pct"})
    do_command({"action": "status"})

Pairs with torque_service for 50 Hz data collection.
"""
import asyncio
import math
import time
from typing import ClassVar, Mapping, Optional, Sequence, Tuple

from typing_extensions import Self
from viam.module.module import Module
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName, Pose
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic as GenericService
from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.utils import ValueTypes
from viam.logging import getLogger

logger = getLogger("RepeatableGrasp")


class RepeatableGrasp(GenericService, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily("feels", "grasp"), "repeatable")

    def __init__(self, name: ResourceName, arm: Arm, gripper: Gripper):
        super().__init__(name)
        self.arm = arm
        self.gripper = gripper
        self._running = False
        # Waypoints (mm) — override via config attributes
        self.waypoints = {
            "home": (0, 0, 400),
            "approach": (300, 0, 150),
            "grasp": (300, 0, 80),
            "lift": (300, 0, 200),
            "travel": (0, 200, 200),
            "place": (0, 200, 80),
        }

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Tuple[Sequence[str], Sequence[str]]:
        return ["arm", "gripper"], []

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        arm = dependencies[Arm.get_resource_name(config.attributes.fields["arm"].string_value)]
        gripper = dependencies[Gripper.get_resource_name(config.attributes.fields["gripper"].string_value)]
        inst = cls(config.resource_name, arm, gripper)
        # Override waypoints from config if provided
        for k in inst.waypoints:
            if k in config.attributes.fields:
                v = config.attributes.fields[k]
                if len(v.list_value.values) == 3:
                    inst.waypoints[k] = tuple(v.list_value.values[i].number_value for i in range(3))
        inst._running = True
        logger.info("RepeatableGrasp ready")
        return inst

    async def close(self):
        self._running = False

    async def _move(self, name: str):
        x, y, z = self.waypoints[name]
        await self.arm.move_to_position(Pose(x=x, y=y, z=z, o_x=0, o_y=0, o_z=1, theta=0))
        logger.debug(f"Moved to {name}")

    async def _swirl(self, radius_mm: float = 60.0, revolutions: float = 2.0, steps: int = 36):
        """Circular swirl at current TCP XY, fixed Z."""
        pos = await self.arm.get_end_position()
        cx, cy, z = pos.x, pos.y, pos.z
        total = max(8, int(steps * revolutions))
        for i in range(total + 1):
            ang = 2.0 * math.pi * revolutions * (i / total)
            x = cx + radius_mm * math.cos(ang)
            y = cy + radius_mm * math.sin(ang)
            await self.arm.move_to_position(
                Pose(x=x, y=y, z=z, o_x=pos.o_x, o_y=pos.o_y, o_z=pos.o_z, theta=pos.theta))
            await asyncio.sleep(0.05)

    async def do_command(self, command: Mapping[str, ValueTypes], *,
                         timeout: Optional[float] = None, **kwargs) -> Mapping[str, ValueTypes]:
        action = str(command.get("action", "")).lower()

        if action == "status":
            return {"running": self._running}

        if action == "swirl":
            label = str(command.get("object", "mystery"))
            try:
                # Pick → swirl → place sequence
                await self._move("home")
                await self._move("approach")
                await self._move("grasp")
                await self.gripper.close()
                await self._move("lift")
                await self._move("travel")
                await self._swirl()  # The money move
                await self._move("place")
                await self.gripper.open()
                await self._move("lift")
                await self._move("home")
                return {"status": "ok", "object": label}
            except Exception as e:
                return {"status": "error", "message": f"{type(e).__name__}: {e}"}

        return {"error": f"unknown action '{action}'", "available": ["swirl", "status"]}


if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
