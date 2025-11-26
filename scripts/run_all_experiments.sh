#!/usr/bin/env bash
set -e

# Default to skipping monitoring (Docker) unless --with-monitoring is specified
SKIP_MONITORING="--skip-monitoring"
MANIFESTS=()

# Parse arguments
for arg in "$@"; do
	if [ "$arg" = "--with-monitoring" ]; then
		SKIP_MONITORING=""
	else
		MANIFESTS+=("$arg")
	fi
done

# Default to all experiments if no manifests provided
# Use nullglob to handle case where no files match
shopt -s nullglob
if [ ${#MANIFESTS[@]} -eq 0 ]; then
	MANIFESTS=(configs/experiments/*.yaml)
fi

if [ ${#MANIFESTS[@]} -eq 0 ]; then
	echo "No experiment manifests found in configs/experiments/"
	exit 1
fi

for manifest in "${MANIFESTS[@]}"; do
	echo "Running experiment from manifest: $manifest"
	python3 scripts/run_experiment.py $SKIP_MONITORING "$manifest"
	echo "Experiment completed: $manifest"
	echo "----------------------------------------"
done
