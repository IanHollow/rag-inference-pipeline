#!/usr/bin/env bash
set -euo pipefail

echo "Running install.sh..."

if [ -d ".venv" ]; then
	rm -rf .venv
fi

python3 -m virtualenv .venv
# Activate the virtual environment
# shellcheck disable=SC1091
. .venv/bin/activate
./.venv/bin/python -m pip install -e .

echo "Installation complete!"
