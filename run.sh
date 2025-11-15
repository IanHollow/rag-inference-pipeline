#!/usr/bin/env bash
set -euo pipefail

# Activate the virtual environment
# shellcheck disable=SC1091
. .venv/bin/activate

# Run the pipeline runtime
python -m pipeline.runtime
