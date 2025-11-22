#!/usr/bin/env bash
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting monitoring stack...${NC}"

# Detect container engine and compose command
if command -v docker &> /dev/null; then
	CONTAINER_CMD="docker"
	if docker compose version &> /dev/null; then
		COMPOSE_CMD="docker compose"
	elif command -v docker-compose &> /dev/null; then
		COMPOSE_CMD="docker-compose"
	else
		echo -e "${RED}Error: docker found but no compose plugin/command found${NC}"
		exit 1
	fi
elif command -v podman &> /dev/null; then
	CONTAINER_CMD="podman"
	if command -v podman-compose &> /dev/null; then
		COMPOSE_CMD="podman-compose"
	elif podman compose version &> /dev/null; then
		COMPOSE_CMD="podman compose"
	else
		echo -e "${RED}Error: podman found but no podman-compose found${NC}"
		exit 1
	fi
else
	echo -e "${RED}Error: neither docker nor podman is installed${NC}"
	exit 1
fi

echo "Using container engine: $CONTAINER_CMD"
echo "Using compose command: $COMPOSE_CMD"

# Navigate to monitoring directory
cd "$(dirname "$0")/../monitoring"

# Launch compose stack
echo "Launching services..."
$COMPOSE_CMD up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."

wait_for_service() {
	local url=$1
	local name=$2
	local max_retries=30
	local count=0

	echo -n "Waiting for $name ($url)..."
	while ! curl -s --fail "$url" > /dev/null; do
		sleep 1
		count=$((count + 1))
		if [ $count -ge $max_retries ]; then
			echo -e " ${RED}FAILED${NC}"
			return 1
		fi
		echo -n "."
	done
	echo -e " ${GREEN}OK${NC}"
	return 0
}

# Wait for Prometheus
wait_for_service "http://localhost:9090/-/healthy" "Prometheus"

# Wait for Grafana
wait_for_service "http://localhost:3000/api/health" "Grafana"

echo -e "${GREEN}Monitoring stack is up and running!${NC}"
echo "Prometheus: http://localhost:9090"
echo "Grafana:    http://localhost:3000"
