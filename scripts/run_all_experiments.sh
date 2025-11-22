#!/usr/bin/env bash
set -e

# Default to all experiments if no args provided
# Use nullglob to handle case where no files match
shopt -s nullglob
MANIFESTS=("${@}")
if [ ${#MANIFESTS[@]} -eq 0 ]; then
	MANIFESTS=(configs/experiments/*.yaml)
fi

if [ ${#MANIFESTS[@]} -eq 0 ]; then
	echo "No experiment manifests found in configs/experiments/"
	exit 1
fi

for manifest in "${MANIFESTS[@]}"; do
	echo "Running experiment from manifest: $manifest"
	python3 scripts/run_experiment.py "$manifest"
	echo "Experiment completed: $manifest"
	echo "----------------------------------------"
done
