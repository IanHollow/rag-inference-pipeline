#!/usr/bin/env bash
set -euo pipefail

echo "Running install.sh..."

if [ -d ".venv" ]; then
	rm -rf .venv
fi

python3 -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo "Installation complete!"
