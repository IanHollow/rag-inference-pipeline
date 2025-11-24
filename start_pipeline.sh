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
# shellcheck disable=SC2329
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
NODE_NUMBER=0 ./run.sh > logs/node0-gateway.log 2>&1 &
PIDS+=($!)
sleep 1

# Start Node 1 (Retrieval)
echo "Starting Node 1 (Retrieval)..."
NODE_NUMBER=1 ./run.sh > logs/node1-retrieval.log 2>&1 &
PIDS+=($!)
sleep 1

# Start Node 2 (Generation)
echo "Starting Node 2 (Generation)..."
NODE_NUMBER=2 ./run.sh > logs/node2-generation.log 2>&1 &
PIDS+=($!)
sleep 1

echo ""
echo "Waiting for all nodes to become healthy..."

# Health check function
check_health() {
	local port=$1
	if curl -s -o /dev/null -w "%{http_code}" "http://localhost:$port/health" | grep -q "200"; then
		return 0
	else
		return 1
	fi
}

# Wait for services to be ready
MAX_RETRIES=60
count=0
nodes_ready=0

while [ $count -lt $MAX_RETRIES ]; do
	nodes_ready=0

	# Check Gateway
	if check_health 8000; then
		((nodes_ready += 1))
	fi

	# Check Retrieval
	if check_health 8001; then
		((nodes_ready += 1))
	fi

	# Check Generation
	if check_health 8002; then
		((nodes_ready += 1))
	fi

	if [ $nodes_ready -eq 3 ]; then
		echo "All nodes are healthy and ready to accept requests!"
		echo "Monitor logs with: tail -f logs/*.log"
		echo "Press Ctrl+C to stop all nodes"
		echo ""
		wait
		exit 0
	fi

	sleep 1
	((count += 1))
	echo -ne "Waiting for nodes... ($count/$MAX_RETRIES)\r"
done

echo ""
echo "Error: Timeout waiting for nodes to become healthy."
echo "Check logs for details:"
echo "  tail -n 20 logs/*.log"
exit 1
