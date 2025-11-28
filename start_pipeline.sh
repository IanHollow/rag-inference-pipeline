#!/usr/bin/env bash
set -euo pipefail

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

# Use environment variables or defaults
TOTAL_NODES=${TOTAL_NODES:-3}
NODE_0_IP=${NODE_0_IP:-"localhost:8000"}
NODE_1_IP=${NODE_1_IP:-"localhost:8001"}
NODE_2_IP=${NODE_2_IP:-"localhost:8002"}

echo "Starting ML Pipeline ($TOTAL_NODES node(s))..."
echo ""

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
for ((i = 0; i < TOTAL_NODES; i++)); do
	echo "    - logs/node${i}.log"
done
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

# Helper function to get node IP
get_node_ip() {
	local node_num=$1
	case $node_num in
		0) echo "$NODE_0_IP" ;;
		1) echo "$NODE_1_IP" ;;
		2) echo "$NODE_2_IP" ;;
		*) echo "localhost:$((8000 + node_num))" ;;
	esac
}

# Helper function to get port from IP:port string
get_port() {
	local ip_port=$1
	echo "${ip_port##*:}"
}

# Start all nodes
for ((i = 0; i < TOTAL_NODES; i++)); do
	node_ip=$(get_node_ip $i)
	echo "Starting Node $i on $node_ip..."
	TOTAL_NODES=$TOTAL_NODES NODE_NUMBER=$i NODE_0_IP=$NODE_0_IP NODE_1_IP=$NODE_1_IP NODE_2_IP=$NODE_2_IP ./run.sh > "logs/node${i}.log" 2>&1 &
	PIDS+=($!)
	sleep 1
done

echo ""
echo "Waiting for all nodes to become healthy..."

# Health check function - returns "healthy" or error status
check_health() {
	local host=$1
	local port=$2
	local response
	response=$(curl -s -o /dev/null -w "%{http_code}" "http://${host}:${port}/health" 2> /dev/null) || response="000"
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
		# Move cursor up to overwrite previous status block (TOTAL_NODES + 1 lines)
		lines_to_clear=$((TOTAL_NODES + 1))
		echo -ne "\033[${lines_to_clear}A\033[K"
	fi

	echo "Health Status (attempt $((count + 1))):"

	# Check all nodes dynamically
	for ((i = 0; i < TOTAL_NODES; i++)); do
		node_ip=$(get_node_ip $i)
		node_host="${node_ip%:*}"
		node_port=$(get_port "$node_ip")
		node_status=$(check_health "$node_host" "$node_port")
		if [ "$node_status" = "healthy" ]; then
			nodes_ready=$((nodes_ready + 1))
			echo -e "  Node $i :$node_port - ✓ healthy\033[K"
		else
			echo -e "  Node $i :$node_port - ✗ $node_status\033[K"
		fi
	done

	if [ $nodes_ready -eq "$TOTAL_NODES" ]; then
		echo ""
		echo "All $TOTAL_NODES node(s) are healthy and ready to accept requests!"
		echo "Monitor logs with: tail -f logs/*.log"
		echo "Press Ctrl+C to stop all nodes"
		echo ""
		wait
		exit 0
	fi

	sleep 2
	count=$((count + 1))
done
