#!/usr/bin/env bash
set -euo pipefail

echo "Running install.sh..."

if ! command -v uv &> /dev/null; then
	if command -v curl &> /dev/null; then
		curl -LsSf https://astral.sh/uv/install.sh | sh
	elif command -v wget &> /dev/null; then
		wget -qO- https://astral.sh/uv/install.sh | sh
	else
		echo "Error: Neither curl nor wget is available."
		exit 1
	fi
	# Add uv to PATH for the current script session
	export PATH="$HOME/.local/bin:$PATH"
else
	echo "uv is already installed. Updating..."
	uv self update &> /dev/null || echo "Warning: uv self update failed, continuing with existing version..."
fi

if [ -d ".venv" ]; then
	rm -rf .venv
fi

uv venv --no-project
uv pip install -r requirements.txt

echo "Installation complete!"
