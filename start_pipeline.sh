#!/usr/bin/env bash
set -euo pipefail

echo "Starting ML Pipeline (3 nodes)..."
echo ""

# Load shared environment variables
if [ -f .env.shared ]; then
	echo "Loading shared environment from .env.shared"
	set -a
	# shellcheck disable=SC1091
	source .env.shared
	set +a
else
	echo "Warning: .env.shared not found, using default values"
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment
if [ ! -d ".venv" ]; then
	echo "Virtual environment not found. Run ./install.sh first."
	exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "Logs will be written to:"
echo "    - logs/node0-gateway.log"
echo "    - logs/node1-retrieval.log"
echo "    - logs/node2-generation.log"
echo ""

# Array to store PIDs
declare -a PIDS=()

# Function to cleanup on exit
cleanup() {
	echo ""
	echo "Stopping all nodes..."
	for pid in "${PIDS[@]}"; do
		if kill -0 "$pid" 2> /dev/null; then
			kill "$pid" 2> /dev/null || true
		fi
	done
	wait 2> /dev/null || true
	echo "All nodes stopped"
	exit 0
}

# Trap Ctrl+C and other termination signals
trap cleanup SIGINT SIGTERM EXIT

# Start Node 0 (Gateway)
echo "Starting Node 0 (Gateway)..."
NODE_NUMBER=0 python -m pipeline.runtime > logs/node0-gateway.log 2>&1 &
PIDS+=($!)
sleep 1

# Start Node 1 (Retrieval)
echo "Starting Node 1 (Retrieval)..."
NODE_NUMBER=1 python -m pipeline.runtime > logs/node1-retrieval.log 2>&1 &
PIDS+=($!)
sleep 1

# Start Node 2 (Generation)
echo "Starting Node 2 (Generation)..."
NODE_NUMBER=2 python -m pipeline.runtime > logs/node2-generation.log 2>&1 &
PIDS+=($!)
sleep 1

echo ""
echo "All nodes started!"
echo "Monitor logs with: tail -f logs/*.log"
echo "Press Ctrl+C to stop all nodes"
echo ""

wait
