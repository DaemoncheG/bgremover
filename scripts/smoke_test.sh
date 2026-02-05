#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=""
if [[ -x "./venv/bin/python" ]]; then
  PYTHON_BIN="./venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "python3/python not found"
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found; skipping CLI smoke tests"
  exit 0
fi

"$PYTHON_BIN" main.py --list-models | grep -qx "u2net"
echo "Smoke tests passed."
