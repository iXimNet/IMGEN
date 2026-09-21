#!/usr/bin/env bash
# IMGEN setup for macOS (Apple Silicon recommended).
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v python3 >/dev/null; then
  echo "Python 3.10+ is required." >&2
  exit 1
fi

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision
pip install -r requirements.txt

echo
echo "Image21-INT8 requires NVIDIA CUDA and is not available on macOS."
echo "Start the studio:     python -m imgen"
echo "UI only, no weights:  python -m imgen --demo"
