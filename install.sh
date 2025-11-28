#!/usr/bin/env bash
set -euo pipefail

echo "Running install.sh..."

# Detect platform and GPU availability
PLATFORM="$(uname -s)"
PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')
FAISS_EXTRA="cpu"

# Check if we should use faiss-gpu (Linux + NVIDIA CUDA + Python <3.13 only)
if [ "$PLATFORM" = "Linux" ]; then
	if command -v nvidia-smi &> /dev/null && [ "$PYTHON_MINOR" -lt 13 ]; then
		echo "Detected Linux with NVIDIA GPU and Python 3.$PYTHON_MINOR - will install faiss-gpu"
		FAISS_EXTRA="gpu"
	elif command -v nvidia-smi &> /dev/null; then
		echo "Detected Linux with NVIDIA GPU but Python 3.$PYTHON_MINOR >= 3.13 - faiss-gpu not available, using faiss-cpu"
	else
		echo "Detected Linux without NVIDIA GPU - will use faiss-cpu"
	fi
elif [ "$PLATFORM" = "Darwin" ]; then
	echo "Detected macOS - will use faiss-cpu (faiss-gpu not supported on macOS)"
else
	echo "Detected $PLATFORM - will use faiss-cpu"
fi

if [ -d ".venv" ]; then
	rm -rf .venv
fi

python3 -m virtualenv .venv
# Activate the virtual environment
# shellcheck disable=SC1091
. .venv/bin/activate

echo "Installing with [$FAISS_EXTRA] FAISS backend..."
./.venv/bin/python -m pip install -e ".[$FAISS_EXTRA]"

echo "Installation complete!"
