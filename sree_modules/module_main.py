#!/usr/bin/env python3
"""Module entrypoint — serves the feels Viam models.

Importing each EasyResource subclass auto-registers its model:
  • feels:sensing:torque-features  (torque_service.TorqueFeatures)  — 50 Hz torque polling
  • feels:grasp:repeatable         (grasp_service.RepeatableGrasp)   — deterministic macro
"""
import asyncio
from viam.module.module import Module

import torque_service  # noqa: F401  (registers TorqueFeatures)
import grasp_service   # noqa: F401  (registers RepeatableGrasp)

if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
