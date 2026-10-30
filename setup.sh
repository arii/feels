#!/usr/bin/env bash
# feels — one-command dev setup.
#   bash setup.sh
# Creates .venv, installs requirements, seeds .env, and sanity-checks the
# offline pipeline (no robot or Atlas needed). Run `bash setup.sh` — the file
# may arrive without the exec bit depending on how it was synced.
set -euo pipefail
cd "$(dirname "$0")"

say() { printf '\n==> %s\n' "$*"; }

# 1. Python version -----------------------------------------------------------
if ! command -v python3 >/dev/null; then
  echo "ERROR: python3 not found. Install Python 3.10+ first." >&2; exit 1
fi
PYV=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
say "python3 $PYV detected"

# 2. Virtualenv ---------------------------------------------------------------
if [ ! -d .venv ]; then
  say "creating .venv"
  python3 -m venv .venv
else
  say ".venv already exists — reusing"
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# 3. Dependencies -------------------------------------------------------------
say "installing requirements.txt"
pip install -q -r requirements.txt
pip freeze | grep -Ei "^(viam-sdk|pymongo|aiohttp|scikit-learn)==" || true

# 4. Credentials --------------------------------------------------------------
if [ ! -f .env ]; then
  say "creating .env from .env.example — fill in your credentials"
  cp .env.example .env
else
  say ".env already exists — leaving it alone"
fi

# 5. Sanity check (offline — no robot, no Atlas) -------------------------------
say "sanity check: feature extractor on a bundled recording"
CSV=$(ls recordings/recording_*.csv 2>/dev/null | head -1 || true)
if [ -n "$CSV" ]; then
  python -c "
import sys; sys.path.insert(0, '.')
import feature_extractor as fe
out = fe.extract_features(__import__('pandas').read_csv('$CSV'))
print('  features:', len(out['z']), '| embedding dim:', len(out['e']), '| phase:', out['diagnostics']['phase_source'])
print('  OK — extractor works')
"
else
  echo "  (no recordings/*.csv found — skipping)"
fi

say "teach_csv dry-run (parses the bundled recordings)"
python teach_csv.py --dry-run recordings/ 2>/dev/null | head -4 || true

cat <<'EOF'

Setup complete. Next steps:
  source .venv/bin/activate          # every new shell

  # UI without hardware (recommended first run):
  FEELS_MOCK=1 python realtime_plot.py --port 8080   # open http://localhost:8080

  # With hardware + Atlas (fill in .env first):
  python realtime_plot.py --port 8080

  # Batch-teach recordings into Atlas:
  python teach_csv.py recordings/ --require-swirl --dry-run   # preview
  python teach_csv.py recordings/ --require-swirl --session fri

See README.md (Quickstart) for the full workflow.
EOF
