#!/usr/bin/env python3
"""Module entrypoint — serves all three feels models in one Viam module.

Importing each EasyResource subclass auto-registers its model:
  • braveorangerover:pick_logic:yolo3d-pick   (pick_service.ObjectPicker)
  • braveorangerover:sensing:torque-features  (torque_service.TorqueFeatures)
  • braveorangerover:content:classifier       (content_service.ContentClassifier)
"""
import asyncio
from viam.module.module import Module

import pick_service      # noqa: F401  (registers ObjectPicker / yolo3d-pick)
import torque_service    # noqa: F401  (registers TorqueFeatures)
import content_service   # noqa: F401  (registers ContentClassifier)

if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
