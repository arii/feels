#!/usr/bin/env bash
# Viam module entrypoint — serves 3 models:
#   braveorangerover:pick_logic:yolo3d-pick · :sensing:torque-features · :content:classifier
set -euo pipefail
cd "$(dirname "$0")"

VENV="${VIAM_MODULE_VENV:-.venv-module}"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip
  # Required for module startup + the lightweight pick/torque models:
  "$VENV/bin/pip" install "viam-sdk>=0.85.0" "numpy>=1.24.0" "python-dotenv>=1.0.0" "typing-extensions>=4.0.0"
  # Heavier ML stack, only for content:classifier's predict (non-fatal: startup still succeeds):
  "$VENV/bin/pip" install "scipy>=1.10.0" "pandas>=2.0.0" "scikit-learn>=1.3.0" "joblib>=1.3.0" "pymongo>=4.10.0" "certifi>=2024.0.0" \
    || echo "[run.sh] warn: content ML deps failed to install — content:classifier predict disabled, other models OK"
fi

# src/ on sys.path (script dir) so module imports resolve.
exec "$VENV/bin/python" src/module_main.py "$@"
