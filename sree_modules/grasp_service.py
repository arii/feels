#!/usr/bin/env python3
"""
grasp_service.py — Viam Generic Service module: deterministic REPEATABLE GRASP.

Exposes the locked repeatable_grasp.py macro (pick → swirl → place) as a Viam
service. No vision, no YOLO — the motion is fixed and deterministic, so every
recording is strictly comparable. Pairs with torque_service for 50 Hz data
collection.

    do_command({"action": "swirl", "object": "water_075pct"})  # full macro + CSV
    do_command({"action": "status"})

The macro runs as a subprocess (repeatable_grasp.py is locked — do not modify).
Output CSV lands in recordings/ for the Atlas pipeline.
"""
import asyncio
import os
import subprocess
import sys
from typing import ClassVar, Mapping, Optional, Sequence, Tuple

from typing_extensions import Self
from viam.module.module import Module
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic as GenericService
from viam.utils import ValueTypes
from viam.logging import getLogger

logger = getLogger("RepeatableGrasp")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RepeatableGrasp(GenericService, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily("feels", "grasp"), "repeatable")

    def __init__(self, name: ResourceName):
        super().__init__(name)
        self._running = False

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Tuple[Sequence[str], Sequence[str]]:
        return [], []

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        inst = cls(config.resource_name)
        inst._running = True
        logger.info("RepeatableGrasp ready")
        return inst

    async def close(self):
        self._running = False

    async def do_command(self, command: Mapping[str, ValueTypes], *,
                         timeout: Optional[float] = None, **kwargs) -> Mapping[str, ValueTypes]:
        action = str(command.get("action", "")).lower()

        if action == "status":
            return {"running": self._running, "macro": "repeatable_grasp.py"}

        if action == "swirl":
            # Run the locked macro as a subprocess — never import it (it's a script)
            label = str(command.get("object", "mystery"))
            cmd = [sys.executable, os.path.join(REPO, "repeatable_grasp.py"),
                   "--object", label, "--record", "--cycles", "1", "--swirl-smooth"]
            logger.info(f"Running macro for '{label}'")
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                out, err = await asyncio.wait_for(proc.communicate(), timeout=300)
                if proc.returncode != 0:
                    return {"status": "error",
                            "message": err.decode()[-500:] if err else "macro failed"}
                # Find the newest recording
                rec_dir = os.path.join(REPO, "recordings")
                csvs = sorted([f for f in os.listdir(rec_dir) if f.endswith(".csv")],
                              key=lambda f: os.path.getmtime(os.path.join(rec_dir, f)))
                return {"status": "ok", "csv": csvs[-1] if csvs else None,
                        "object": label}
            except asyncio.TimeoutError:
                return {"status": "error", "message": "macro timed out after 300s"}
            except Exception as e:
                return {"status": "error", "message": f"{type(e).__name__}: {e}"}

        return {"error": f"unknown action '{action}'",
                "available": ["swirl", "status"]}


if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
