"""Regression tests for issue #10: deduplicate core modules across root and src/.

Verifies that root compatibility shims and canonical src/ implementations
are completely aligned and produce identical feature vectors and configurations.
"""
import os
import sys
import unittest
import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import feature_extractor as root_fe
import src.feature_extractor as src_fe
import repeatable_grasp as root_rg
import src.repeatable_grasp as src_rg


class TestCanonicalModules(unittest.TestCase):
    def test_feature_extractor_symbols_identical(self):
        """Feature names, descriptions, scales, and weights must match exactly."""
        self.assertEqual(root_fe.FEATURE_NAMES, src_fe.FEATURE_NAMES)
        self.assertEqual(root_fe.FEATURE_DESCRIPTIONS, src_fe.FEATURE_DESCRIPTIONS)
        self.assertEqual(root_fe.FEATURE_SCALES, src_fe.FEATURE_SCALES)
        self.assertEqual(root_fe.FEATURE_WEIGHTS, src_fe.FEATURE_WEIGHTS)

        # Ensure dead/nuisance weights are excluded (0.0) as proven in baseline audit
        self.assertEqual(src_fe.FEATURE_WEIGHTS["mass_j2"], 0.0)
        self.assertEqual(src_fe.FEATURE_WEIGHTS["mass_j3"], 0.0)
        self.assertEqual(src_fe.FEATURE_WEIGHTS["swirl_speed"], 0.0)

    def test_extract_features_parity(self):
        """Synthetic recording extract_features must produce identical results."""
        n_samples = 30
        t = np.linspace(0, 6, n_samples)
        phases = ["INIT"] * 6 + ["LIFT_TO_TRAVEL"] * 6 + ["SWIRL"] * 18
        df = pd.DataFrame({
            "phase": phases,
            "elapsed_sec": t,
            "tcp_x_mm": np.sin(t) * 10,
            "tcp_y_mm": np.cos(t) * 10,
            "tcp_z_mm": np.full(n_samples, 150.0),
            "tau1_nm": np.full(n_samples, 0.1),
            "tau2_nm": np.full(n_samples, 2.5),
            "tau3_nm": np.full(n_samples, 1.2),
            "tau4_nm": np.full(n_samples, 0.05),
            "tau5_nm": np.full(n_samples, 0.02),
            "tau6_nm": np.full(n_samples, 0.01),
            "q2_deg": np.full(n_samples, 15.0),
            "gripper_pos": np.full(n_samples, 50.0),
            "object_label": ["test_cup"] * n_samples,
        })

        out_root = root_fe.extract_features(df)
        out_src = src_fe.extract_features(df)

        self.assertEqual(out_root["z"], out_src["z"])
        self.assertEqual(out_root["e"], out_src["e"])
        self.assertEqual(out_root["diagnostics"], out_src["diagnostics"])

    def test_repeatable_grasp_paths(self):
        """RECORDINGS_DIR and SWIRL_CACHE must resolve to repo root regardless of import origin."""
        expected_rec = os.path.join(_REPO_ROOT, "recordings")
        expected_swirl = os.path.join(_REPO_ROOT, "swirl_joints.json")

        self.assertEqual(os.path.abspath(root_rg.RECORDINGS_DIR), os.path.abspath(expected_rec))
        self.assertEqual(os.path.abspath(src_rg.RECORDINGS_DIR), os.path.abspath(expected_rec))
        self.assertEqual(os.path.abspath(root_rg.SWIRL_CACHE), os.path.abspath(expected_swirl))
        self.assertEqual(os.path.abspath(src_rg.SWIRL_CACHE), os.path.abspath(expected_swirl))


if __name__ == "__main__":
    unittest.main()
