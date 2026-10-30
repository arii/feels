#!/usr/bin/env python3
"""
content_service.py — Viam Generic Service module: CUP-CONTENT CLASSIFIER.

Predicts what's in the cup — **solid / liquid / empty** — from the proprioceptive
torque signal, using the 2-stage leak-free cascade (empty→loaded, solid↔liquid)
and reporting it as a 2-D state spectrum (loadedness × rigidity). Pairs with the
`torque-features` sensing module: torque-features *produces* the signal, this model
*classifies* it.

    do_command({"action": "predict", "recording": "recording_..._water.csv"})
    do_command({"action": "status"})

Heavy ML deps (scipy / pandas / scikit-learn / joblib) are imported LAZILY inside
predict, so a missing dep degrades only this one action — the lightweight
torque-features / yolo3d-pick models in the same module keep working.

── Machine config ───────────────────────────────────────────────────────────
{ "model": "braveorangerover:content:classifier", "type": "generic", "attributes": {} }

HONEST NOTE: on this 4–5 Hz proprioception data, leak-free accuracy is ~baseline
(see BASELINE_FINDINGS.md). "empty vs loaded" is the reliable axis; "solid vs
liquid" works only with a real swirl. The pipeline is correct; the signal is the
limit — a richer (vision/audio) embedding is the fix.
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
from viam.services.generic import Generic as GenericService
from viam.utils import struct_to_dict, ValueTypes
from viam.logging import getLogger

logger = getLogger("ContentClassifier")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root


class ContentClassifier(GenericService, EasyResource):
    """Predicts solid / liquid / empty from the torque-feature cascade."""

    MODEL: ClassVar[Model] = Model(ModelFamily("braveorangerover", "content"), "classifier")

    model_path: str = os.path.join(ROOT, "models", "cascade.joblib")
    recordings_dir: str = os.path.join(ROOT, "recordings")

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Tuple[Sequence[str], Sequence[str]]:
        return [], []

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        self = super().new(config, dependencies)
        a = struct_to_dict(config.attributes)
        self.model_path = a.get("model_path", os.path.join(ROOT, "models", "cascade.joblib"))
        self.recordings_dir = a.get("recordings_dir", os.path.join(ROOT, "recordings"))
        logger.info(f"ContentClassifier ready (model={self.model_path})")
        return self

    async def do_command(self, command: Mapping[str, ValueTypes], *,
                         timeout: Optional[float] = None, **kwargs) -> Mapping[str, ValueTypes]:
        action = str(command.get("action", "")).lower()

        if action == "status":
            return {"model_exists": os.path.isfile(self.model_path),
                    "classes": ["empty", "solid", "liquid"],
                    "note": "empty-vs-loaded reliable; solid-vs-liquid needs a swirl"}

        if action == "predict":
            return await asyncio.to_thread(self._predict, command)

        return {"error": f"unknown action '{action}'", "available": ["predict", "status"]}

    def _predict(self, command) -> dict:
        """Lazy heavy-dep prediction so it never blocks module startup."""
        try:
            import joblib
            import numpy as np
            import pandas as pd
            import train_classifier as tc   # in src/ (on path via module entrypoint)
        except Exception as e:
            return {"error": f"ML deps/model unavailable: {e}"}

        rec = str(command.get("recording", ""))
        if not rec:
            return {"error": "provide 'recording' (a recordings/*.csv basename)"}
        path = rec if os.path.isabs(rec) else os.path.join(self.recordings_dir, os.path.basename(rec))
        if not os.path.isfile(path):
            return {"error": f"recording not found: {os.path.basename(path)}"}
        if not os.path.isfile(self.model_path):
            return {"error": "cascade.joblib missing — train with: python src/train_classifier.py --save"}

        df = pd.read_csv(path)
        b = joblib.load(self.model_path)
        order = b["feature_order"]
        x = np.array([[tc.features(df)[k] for k in order]])

        def p(pipe, cls):
            return float(dict(zip(pipe.classes_, pipe.predict_proba(x)[0])).get(cls, 0.0))
        loadedness = p(b["stage1"], "loaded")
        rigidity = p(b["stage2_all"], "solid")
        coarse = "empty" if loadedness < 0.5 else ("solid" if rigidity >= 0.5 else "liquid")
        has_swirl = bool((df["phase"] == "SWIRL").any()) if "phase" in df else False

        return {
            "prediction": coarse,
            "loadedness": round(loadedness, 3),    # empty 0 .. 1 full
            "rigidity": round(rigidity, 3),        # liquid 0 .. 1 solid
            "has_swirl": has_swirl,
            "confidence_note": "" if has_swirl or coarse == "empty"
                               else "low-confidence liquid/solid (no swirl in recording)",
            "recording": os.path.basename(path),
            "true_label": str(df["object_label"].iloc[0]) if "object_label" in df else None,
        }


if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
