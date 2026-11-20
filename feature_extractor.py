#!/usr/bin/env python3
"""Compatibility shim: canonical module is src/feature_extractor.py (#10).

Re-exports all public symbols for root-level scripts and tooling.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from src.feature_extractor import *  # noqa: F401, F403
from src.feature_extractor import (  # explicit exports for IDEs & linters
    FALLBACK_DYN_PHASES,
    FEATURE_DESCRIPTIONS,
    FEATURE_NAMES,
    FEATURE_SCALES,
    FEATURE_WEIGHTS,
    JOINTS,
    SWIRL_PHASE,
    _fs,
    _lowpass,
    _phase_lag,
    _tare_with_qc,
    _tcp_speed,
    extract_features,
    main,
    swirl_window_series,
)

if __name__ == "__main__":
    main()
