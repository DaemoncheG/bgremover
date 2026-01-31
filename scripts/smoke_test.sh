#!/usr/bin/env bash
set -euo pipefail

if ! command -v python >/dev/null 2>&1; then
  echo "python not found"
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found; skipping CLI smoke tests"
  exit 0
fi

python main.py --list-models | grep -qx "u2net"
echo "Smoke tests passed."
