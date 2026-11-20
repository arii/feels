#!/usr/bin/env python3
"""Compatibility shim: canonical module is src/repeatable_grasp.py (#10).

Re-exports all public symbols for root-level scripts and tooling.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from src.repeatable_grasp import *  # noqa: F401, F403

if __name__ == "__main__":
    import asyncio
    from src.repeatable_grasp import main
    asyncio.run(main())
