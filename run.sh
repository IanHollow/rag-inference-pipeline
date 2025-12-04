#!/usr/bin/env bash
set -euo pipefail

# Activate the virtual environment
# shellcheck disable=SC1091
. .venv/bin/activate

# CRITICAL: Set OpenMP threads to 1 to prevent deadlocks.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Disable tokenizer parallelism to avoid fork deadlocks and semaphore leaks
export TOKENIZERS_PARALLELISM=false

# Allow duplicate OpenMP libraries (PyTorch and FAISS both link their own libomp)
# Without this, the process will crash with "OMP: Error #15" on macOS
export KMP_DUPLICATE_LIB_OK=TRUE

# Check if profiling is enabled
if [[ ${PROFILE_WITH_SCALENE:-0} == "1"   ]]; then
	# Ensure required environment variables are set
	RUN_ID=${PROFILING_RUN_ID:-"manual_run"}
	NODE_NUM=${NODE_NUMBER:-"0"}

	# Create output directory
	OUTPUT_DIR="artifacts/scalene/${RUN_ID}"
	mkdir -p "${OUTPUT_DIR}"

	OUTPUT_FILE="${OUTPUT_DIR}/node_${NODE_NUM}.html"

	echo "Starting with Scalene profiling. Output: ${OUTPUT_FILE}"

	# Build Scalene arguments
	SCALENE_ARGS=(
		"--cli"
		"--html"
		"--reduced-profile"
		"--cpu"
		"--gpu"
		"--memory"
		"--profile-interval" "5"
		"--outfile" "${OUTPUT_FILE}"
	)

	# Add optional profile-only flag
	if [[ -n ${SCALENE_PROFILE_ONLY:-} ]]; then
		SCALENE_ARGS+=("--profile-only" "${SCALENE_PROFILE_ONLY}")
	else
		SCALENE_ARGS+=("--profile-all")
	fi

	# Add optional cpu-percent-threshold flag
	if [[ -n ${SCALENE_CPU_PERCENT_THRESHOLD:-} ]]; then
		SCALENE_ARGS+=("--cpu-percent-threshold" "${SCALENE_CPU_PERCENT_THRESHOLD}")
	fi

	# Run with Scalene
	exec python -m scalene "${SCALENE_ARGS[@]}" -m pipeline.runtime
else
	# Run the pipeline runtime normally
	exec python -m pipeline.runtime
fi
