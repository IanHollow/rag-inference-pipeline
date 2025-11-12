#!/usr/bin/env bash
set -euo pipefail

echo "Running install.sh..."

if [ -d ".venv" ]; then
	rm -rf .venv
fi

python3 -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .

echo "Installation complete!"
