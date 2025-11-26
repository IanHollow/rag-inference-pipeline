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

# Health check function - returns "healthy" or error status
check_health() {
	local port=$1
	local response
	response=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$port/health" 2>/dev/null) || response="000"
	if [ "$response" = "200" ]; then
		echo "healthy"
	elif [ "$response" = "000" ] || [ -z "$response" ]; then
		echo "not reachable"
	else
		echo "HTTP $response"
	fi
}

# Wait for services to be ready (no timeout - waits indefinitely)
count=0

while true; do
	nodes_ready=0

	# Clear line and move cursor up to overwrite previous status
	if [ $count -gt 0 ]; then
		# Move cursor up 4 lines to overwrite previous status block
		echo -ne "\033[4A\033[K"
	fi

	echo "Health Status (attempt $((count + 1))):"

	# Check Gateway
	gateway_status=$(check_health 8000)
	if [ "$gateway_status" = "healthy" ]; then
		nodes_ready=$((nodes_ready + 1))
		echo -e "  Node 0 (Gateway)    :8000 - ✓ healthy\033[K"
	else
		echo -e "  Node 0 (Gateway)    :8000 - ✗ $gateway_status\033[K"
	fi

	# Check Retrieval
	retrieval_status=$(check_health 8001)
	if [ "$retrieval_status" = "healthy" ]; then
		nodes_ready=$((nodes_ready + 1))
		echo -e "  Node 1 (Retrieval)  :8001 - ✓ healthy\033[K"
	else
		echo -e "  Node 1 (Retrieval)  :8001 - ✗ $retrieval_status\033[K"
	fi

	# Check Generation
	generation_status=$(check_health 8002)
	if [ "$generation_status" = "healthy" ]; then
		nodes_ready=$((nodes_ready + 1))
		echo -e "  Node 2 (Generation) :8002 - ✓ healthy\033[K"
	else
		echo -e "  Node 2 (Generation) :8002 - ✗ $generation_status\033[K"
	fi

	if [ $nodes_ready -eq 3 ]; then
		echo ""
		echo "All nodes are healthy and ready to accept requests!"
		echo "Monitor logs with: tail -f logs/*.log"
		echo "Press Ctrl+C to stop all nodes"
		echo ""
		wait
		exit 0
	fi

	sleep 2
	count=$((count + 1))
done
