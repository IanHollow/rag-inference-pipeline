#!/usr/bin/env python3
import copy
from pathlib import Path
import subprocess
import sys

import yaml

REPO_ROOT = Path(__file__).parent.parent
CONFIGS_DIR = REPO_ROOT / "configs" / "experiments"

BASE_MANIFEST = {
    "run_id": "parallelism_benchmark",
    "description": "Benchmark CPU parallelism settings",
    "nodes": [
        {
            "number": 0,
            "profile": "configs/baseline_gateway.yaml",
            "env": {"GATEWAY_BATCH_SIZE": 32, "ENABLE_ADAPTIVE_BATCHING": True},
        },
        {"number": 1, "profile": "configs/retrieval.yaml", "env": {}},
        {"number": 2, "profile": "configs/generation.yaml", "env": {}},
    ],
    "workload": {"requests": 100, "concurrency": 8, "rate_limit": 0.0},
}


def run_benchmark() -> None:
    thread_counts = [4, 8, 12, 16]

    for threads in thread_counts:
        print(f"Running benchmark with {threads} threads...")

        manifest = copy.deepcopy(BASE_MANIFEST)
        manifest["run_id"] = f"parallelism_benchmark_{threads}threads"

        # Set env vars for all nodes
        env_vars = {
            "CPU_INFERENCE_THREADS": threads,
            "CPU_WORKER_THREADS": threads,
            "MAX_PARALLEL_GENERATION": min(4, max(1, threads // 2)),  # Scale generation concurrency
        }

        for node in manifest["nodes"]:
            if isinstance(node, dict) and "env" in node:
                node["env"].update(env_vars)

        # Write temporary manifest
        manifest_path = CONFIGS_DIR / f"temp_benchmark_{threads}.yaml"
        with manifest_path.open("w") as f:
            yaml.dump(manifest, f)

        try:
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "run_experiment.py"),
                    str(manifest_path),
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Benchmark failed for {threads} threads: {e}")
        finally:
            if manifest_path.exists():
                manifest_path.unlink()


if __name__ == "__main__":
    run_benchmark()
