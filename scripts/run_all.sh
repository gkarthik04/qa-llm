#!/usr/bin/env bash
# End-to-end run on Linux/macOS: venv -> install -> train -> index -> evaluate -> tests.
# Usage (from the repo root):  bash scripts/run_all.sh
set -euo pipefail
VENV_DIR="${VENV_DIR:-.venv}"

[ -x "$VENV_DIR/bin/python" ] || python3 -m venv "$VENV_DIR"
PY="$VENV_DIR/bin/python"

"$PY" -m pip install --upgrade pip
if ! command -v nvidia-smi >/dev/null 2>&1; then
  "$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
fi
"$PY" -m pip install -r requirements.txt

"$PY" -m qa.train
"$PY" -m qa.build_index
"$PY" -m qa.evaluate
"$PY" -m qa.evaluate_advanced
"$PY" -m pytest -q

echo
echo "Done. Try:  $PY -m qa.cli ask \"Who was the Super Bowl 50 MVP?\""
echo "API:        $PY -m uvicorn qa.api:app --port 8000"
