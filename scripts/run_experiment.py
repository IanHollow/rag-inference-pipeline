#!/usr/bin/env python3
import argparse
import contextlib
import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

import psutil
import requests
import yaml

REPO_ROOT = Path(__file__).parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "experiments"


def load_manifest(path: str) -> dict[str, Any]:
    with Path(path).open() as f:
        return yaml.safe_load(f)


def get_process_stats(child: psutil.Process) -> dict[str, Any] | None:
    try:
        with child.oneshot():
            return {
                "pid": child.pid,
                "name": child.name(),
                "cmdline": child.cmdline(),
                "cpu_percent": child.cpu_percent(),
                "memory_info": child.memory_info()._asdict(),
                "create_time": child.create_time(),
            }
    except psutil.NoSuchProcess:
        return None


def get_children_stats(pid: int) -> list[dict[str, Any]]:
    stats = []
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            child_stats = get_process_stats(child)
            if child_stats:
                stats.append(child_stats)
    except psutil.NoSuchProcess:
        pass
    return stats


def start_monitoring() -> None:
    print("Starting monitoring stack...")
    subprocess.run([str(REPO_ROOT / "scripts" / "start_monitoring.sh")], check=True)


def wait_for_health(nodes: list[dict[str, Any]], timeout: int = 300) -> None:
    print("Waiting for nodes to be healthy...")
    start_time = time.time()
    ready_nodes: set[int] = set()

    while len(ready_nodes) < len(nodes):
        if time.time() - start_time > timeout:
            raise TimeoutError("Timed out waiting for nodes to be healthy")

        for node in nodes:
            node_num = node["number"]
            if node_num in ready_nodes:
                continue

            # Assume port is 8000 + node_num based on config.py defaults
            port = 8000 + node_num
            url = f"http://localhost:{port}/health"
            try:
                resp = requests.get(url, timeout=1)
                if resp.status_code == 200:
                    print(f"Node {node_num} is ready.")
                    ready_nodes.add(node_num)
            except requests.RequestException:
                pass

        time.sleep(2)


def run_experiment(
    manifest_path: str, skip_monitoring: bool = False, reuse_env: bool = False
) -> None:
    manifest = load_manifest(manifest_path)
    run_id = manifest.get("run_id", "unknown")
    # Append timestamp to run_id to make it unique
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    full_run_id = f"{run_id}_{timestamp}"

    output_dir = ARTIFACTS_DIR / full_run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy manifest to output dir
    shutil.copy(manifest_path, output_dir / "manifest.yaml")

    print(f"Starting experiment: {full_run_id}")
    print(f"Output directory: {output_dir}")

    if not skip_monitoring:
        start_monitoring()

    processes = []
    nodes = manifest["nodes"]
    total_nodes = len(nodes)

    experiment_start_time = time.time()

    try:
        # Start Nodes
        for node in nodes:
            node_num = node["number"]
            profile_path = node["profile"]

            # Resolve profile path relative to repo root
            abs_profile_path = (REPO_ROOT / profile_path).resolve()

            env = os.environ.copy()
            env["NODE_NUMBER"] = str(node_num)
            env["TOTAL_NODES"] = str(total_nodes)
            env["ROLE_PROFILE_OVERRIDE_PATH"] = str(abs_profile_path)
            env["PROFILING_RUN_ID"] = full_run_id

            # Apply env overrides
            if "env" in node:
                for k, v in node["env"].items():
                    env[k] = str(v)

            log_file = (output_dir / f"node_{node_num}.log").open("w")

            print(f"Starting Node {node_num} with profile {profile_path}...")
            p = subprocess.Popen(
                [str(REPO_ROOT / "run.sh")],
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=REPO_ROOT,
                preexec_fn=os.setsid,  # Create new process group for easier cleanup
            )
            processes.append((p, log_file))

        # Wait for health
        wait_for_health(nodes)

        # Run Workload
        workload = manifest.get("workload", {})
        requests_count = workload.get("requests", 10)

        print(f"Running workload: {requests_count} requests...")
        profile_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "profile_pipeline.py"),
            "--requests",
            str(requests_count),
            "--no-interactive",
            "--output-dir",
            str(output_dir / "profile_data"),
        ]

        profile_env = os.environ.copy()
        profile_env["PROFILING_RUN_ID"] = full_run_id

        # Attempt to detect batch size from manifest or profiles to report correctly
        detected_batch_size = "32"  # Default in ProfileFile schema

        # Check manifest env overrides
        for node in nodes:
            if "env" in node and "GATEWAY_BATCH_SIZE" in node["env"]:
                detected_batch_size = str(node["env"]["GATEWAY_BATCH_SIZE"])
                break

        # If not in manifest, check node 0 profile (assumed gateway)
        if detected_batch_size == "32":
            for node in nodes:
                if node["number"] == 0:
                    with contextlib.suppress(Exception):
                        p_path = (REPO_ROOT / node["profile"]).resolve()
                        with p_path.open() as f:
                            p_data = yaml.safe_load(f)
                            if "batch_size" in p_data:
                                detected_batch_size = str(p_data["batch_size"])
                    break

        profile_env["GATEWAY_BATCH_SIZE"] = detected_batch_size

        subprocess.run(profile_cmd, env=profile_env, check=True)

        # Capture Metrics
        print("Capturing metrics...")
        end_time = time.time()

        metrics_output = output_dir / "metrics.csv"
        capture_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "capture_metrics.py"),
            "--run-id",
            full_run_id,
            "--start-time",
            str(experiment_start_time),
            "--end-time",
            str(end_time),
            "--output-file",
            str(metrics_output),
        ]
        subprocess.run(capture_cmd, check=False)  # Don't fail if metrics fail

        # Snapshot process stats
        print("Snapshotting process stats...")
        stats = []
        for p, _ in processes:
            stats.extend(get_children_stats(p.pid))

        with (output_dir / "process_stats.json").open("w") as f:
            json.dump(stats, f, indent=2)

    except Exception as e:
        print(f"Experiment failed: {e}")
    finally:
        print("Tearing down nodes...")
        for p, f in processes:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                p.wait(timeout=5)
            except Exception:
                with contextlib.suppress(Exception):
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", help="Path to experiment manifest YAML")
    parser.add_argument(
        "--skip-monitoring", action="store_true", help="Skip starting monitoring stack"
    )
    parser.add_argument(
        "--reuse-env", action="store_true", help="Reuse existing environment (not implemented)"
    )
    args = parser.parse_args()

    run_experiment(args.manifest, args.skip_monitoring, args.reuse_env)
