#!/usr/bin/env python3
import argparse
import contextlib
import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Any

import psutil
import requests
import yaml


REPO_ROOT = Path(__file__).parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "experiments"

# Marker to identify our pipeline processes
PIPELINE_PROCESS_MARKER = "cs5416-ml-pipeline"


def is_port_in_use(port: int) -> bool:
    """Check if a port is currently in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("localhost", port))
        except OSError:
            return True
        else:
            return False


def _is_pipeline_process(proc: psutil.Process) -> bool:
    """Check if a process is a pipeline process from this project."""
    try:
        cmdline = proc.info.get("cmdline") or []
        cwd = proc.info.get("cwd") or ""
        name = proc.info.get("name") or ""
        cmdline_str = " ".join(cmdline) if cmdline else ""

        # Check if this is a Python process (name can be python, python3, Python, python3.10, etc.)
        is_python = name.lower().startswith("python")

        if is_python:
            # Check if it's running from our project directory or has our marker
            is_our_project = (
                PIPELINE_PROCESS_MARKER in cwd or PIPELINE_PROCESS_MARKER in cmdline_str
            )
            # Check if it's a pipeline node (runs pipeline.runtime or src.pipeline)
            is_pipeline = (
                "pipeline.runtime" in cmdline_str
                or "src.pipeline" in cmdline_str
                or "uvicorn" in cmdline_str
            )

            return is_our_project and is_pipeline
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        pass
    return False


def get_pipeline_processes() -> list[psutil.Process]:
    """Find all running pipeline processes from this project."""
    return [
        proc
        for proc in psutil.process_iter(["pid", "name", "cmdline", "cwd"])
        if _is_pipeline_process(proc)
    ]


def _terminate_process(proc: psutil.Process) -> None:
    """Terminate a single process safely."""
    try:
        print(f"  Terminating PID {proc.pid}: {' '.join(proc.cmdline()[:3])}...")
        proc.terminate()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass


def _kill_process(proc: psutil.Process) -> None:
    """Force kill a single process safely."""
    try:
        print(f"  Force killing PID {proc.pid}...")
        proc.kill()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass


def kill_existing_pipeline_processes() -> None:
    """Kill any existing pipeline processes from previous runs."""
    procs = get_pipeline_processes()
    if not procs:
        return

    print(f"Found {len(procs)} existing pipeline process(es), terminating...")
    for proc in procs:
        _terminate_process(proc)

    # Wait for processes to terminate gracefully
    _gone, alive = psutil.wait_procs(procs, timeout=5)

    # Force kill any remaining
    for proc in alive:
        _kill_process(proc)

    # Final wait
    if alive:
        psutil.wait_procs(alive, timeout=3)

    print("Cleanup complete.")


def ensure_ports_available(ports: list[int], max_wait: int = 10) -> None:
    """Ensure the specified ports are available, waiting if necessary."""
    start_time = time.time()
    while time.time() - start_time < max_wait:
        busy_ports = [p for p in ports if is_port_in_use(p)]
        if not busy_ports:
            return
        print(f"Waiting for ports {busy_ports} to become available...")
        time.sleep(1)

    busy_ports = [p for p in ports if is_port_in_use(p)]
    if busy_ports:
        msg = (
            f"Ports {busy_ports} are still in use after {max_wait}s. "
            "Please manually kill the processes using these ports."
        )
        raise RuntimeError(msg)


def load_manifest(path: str) -> dict[str, Any]:
    with Path(path).open() as f:
        result: dict[str, Any] = yaml.safe_load(f)
        return result


def _setup_node_environment(
    node: dict[str, Any],
    nodes: list[dict[str, Any]],
    manifest: dict[str, Any],
    full_run_id: str,
    skip_monitoring: bool,
) -> dict[str, str]:
    """Set up the environment variables for a pipeline node."""
    node_num = node["number"]
    total_nodes = len(nodes)
    profile_path = node["profile"]
    abs_profile_path = (REPO_ROOT / profile_path).resolve()

    env = os.environ.copy()
    env["NODE_NUMBER"] = str(node_num)
    env["TOTAL_NODES"] = str(total_nodes)
    env["ROLE_PROFILE_OVERRIDE_PATH"] = str(abs_profile_path)
    env["PROFILING_RUN_ID"] = full_run_id

    # Disable Prometheus/OTLP profiling when monitoring is skipped
    if skip_monitoring:
        env["ENABLE_PROFILING"] = "0"
        env["ENABLE_TRACING"] = "0"

    # Set NODE_*_IP for all nodes so they can communicate
    # When all nodes are on localhost, enable local dev mode
    all_localhost = True
    for i in range(total_nodes):
        ip_key = f"NODE_{i}_IP"
        if ip_key not in env:
            env[ip_key] = f"localhost:{8000 + i}"
        if "localhost" not in env[ip_key] and "127.0.0.1" not in env[ip_key]:
            all_localhost = False

    # Enable local dev mode for single-machine testing
    if all_localhost and "LOCAL_DEV_MODE" not in env:
        env["LOCAL_DEV_MODE"] = "true"
        # Calculate threads per node to avoid oversubscription
        cpu_count = os.cpu_count() or 8
        threads_per_node = max(1, (cpu_count - 2) // total_nodes)
        if "CPU_INFERENCE_THREADS" not in env:
            env["CPU_INFERENCE_THREADS"] = str(threads_per_node)
        if "FAISS_THREADS" not in env:
            env["FAISS_THREADS"] = str(threads_per_node)
        if "CPU_WORKER_THREADS" not in env:
            env["CPU_WORKER_THREADS"] = str(threads_per_node)

    # Check for profiling configuration in manifest
    profiling_config = manifest.get("profiling", {})
    if profiling_config.get("enabled", False) or manifest.get("profile_with_scalene", False):
        env["PROFILE_WITH_SCALENE"] = "1"
        env["ENABLE_PROFILING"] = "0"
        env["ENABLE_TRACING"] = "0"
        if "env" in profiling_config:
            for k, v in profiling_config["env"].items():
                env[k] = str(v)

    # Apply env overrides from node config
    if "env" in node:
        for k, v in node["env"].items():
            env[k] = str(v)

    return env


def _start_pipeline_nodes(
    nodes: list[dict[str, Any]],
    manifest: dict[str, Any],
    full_run_id: str,
    skip_monitoring: bool,
    output_dir: Path,
) -> list[tuple[subprocess.Popen, Any]]:
    """Start all pipeline nodes and return the list of processes with their log files."""
    processes = []

    for node in nodes:
        node_num = node["number"]
        profile_path = node["profile"]

        env = _setup_node_environment(node, nodes, manifest, full_run_id, skip_monitoring)
        log_file = (output_dir / f"node_{node_num}.log").open("w")

        print(f"Starting Node {node_num} with profile {profile_path}...")
        p = subprocess.Popen(
            [str(REPO_ROOT / "run.sh")],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=REPO_ROOT,
            start_new_session=True,
        )
        processes.append((p, log_file))

        # Add delay between starting nodes to avoid resource contention
        # during model loading (especially for nodes with heavy ML models)
        if node_num < len(nodes) - 1:
            time.sleep(2)

    return processes


def _detect_batch_size(nodes: list[dict[str, Any]]) -> str:
    """Detect the batch size from manifest or profile configuration."""
    detected_batch_size = "32"  # Default in ProfileFile schema

    # Check manifest env overrides
    for node in nodes:
        if "env" in node and "GATEWAY_BATCH_SIZE" in node["env"]:
            return str(node["env"]["GATEWAY_BATCH_SIZE"])

    # Check node 0 profile (assumed gateway)
    for node in nodes:
        if node["number"] == 0:
            with contextlib.suppress(Exception):
                p_path = (REPO_ROOT / node["profile"]).resolve()
                with p_path.open() as f:
                    p_data = yaml.safe_load(f)
                    if "batch_size" in p_data:
                        return str(p_data["batch_size"])
            break

    return detected_batch_size


def _run_workload(
    manifest: dict[str, Any],
    nodes: list[dict[str, Any]],
    full_run_id: str,
    output_dir: Path,
) -> None:
    """Run the workload defined in the manifest."""
    workload = manifest.get("workload", {})
    requests_count = workload.get("requests", 10)
    concurrency = workload.get("concurrency", 1)
    rate_limit = workload.get("rate_limit", 0.1)

    print(f"Running workload: {requests_count} requests (concurrency={concurrency})...")
    profile_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "profile_pipeline.py"),
        "--requests",
        str(requests_count),
        "--concurrency",
        str(concurrency),
        "--rate-limit",
        str(rate_limit),
        "--no-interactive",
        "--output-dir",
        str(output_dir / "profile_data"),
    ]

    profile_env = os.environ.copy()
    profile_env["PROFILING_RUN_ID"] = full_run_id
    profile_env["GATEWAY_BATCH_SIZE"] = _detect_batch_size(nodes)

    subprocess.run(profile_cmd, env=profile_env, check=True)


def _capture_metrics(
    full_run_id: str,
    output_dir: Path,
    experiment_start_time: float,
) -> None:
    """Capture Prometheus metrics for the experiment."""
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


def _cleanup_processes(
    processes: list[tuple[subprocess.Popen, Any]],
    output_dir: Path,
    full_run_id: str,
) -> None:
    """Clean up all pipeline processes and copy artifacts."""
    print("Tearing down nodes...")
    for p, f in processes:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            p.wait(timeout=15)
        except Exception:
            with contextlib.suppress(Exception):
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        f.close()

    # Final cleanup: ensure all pipeline processes are dead
    remaining = get_pipeline_processes()
    if remaining:
        print(f"Cleaning up {len(remaining)} remaining pipeline process(es)...")
        for proc in remaining:
            with contextlib.suppress(Exception):
                proc.kill()
        psutil.wait_procs(remaining, timeout=5)

    # Copy Scalene reports if they exist
    scalene_src_dir = REPO_ROOT / "artifacts" / "scalene" / full_run_id
    if scalene_src_dir.exists():
        print(f"Copying Scalene reports from {scalene_src_dir}...")
        scalene_dest_dir = output_dir / "scalene"
        scalene_dest_dir.mkdir(exist_ok=True)
        for item in scalene_src_dir.iterdir():
            if item.is_file():
                shutil.copy(item, scalene_dest_dir / item.name)
        print(f"Scalene reports copied to {scalene_dest_dir}")


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
            msg = "Timed out waiting for nodes to be healthy"
            raise TimeoutError(msg)

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
    manifest_path: str, skip_monitoring: bool = False, _reuse_env: bool = False
) -> None:
    manifest = load_manifest(manifest_path)
    run_id = manifest.get("run_id", "unknown")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    full_run_id = f"{run_id}_{timestamp}"

    output_dir = ARTIFACTS_DIR / full_run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(manifest_path, output_dir / "manifest.yaml")

    print(f"Starting experiment: {full_run_id}")
    print(f"Output directory: {output_dir}")

    kill_existing_pipeline_processes()

    nodes = manifest["nodes"]
    required_ports = [8000 + node["number"] for node in nodes]
    ensure_ports_available(required_ports)

    if not skip_monitoring:
        start_monitoring()

    processes: list[tuple[subprocess.Popen, Any]] = []
    experiment_start_time = time.time()

    try:
        processes = _start_pipeline_nodes(nodes, manifest, full_run_id, skip_monitoring, output_dir)
        wait_for_health(nodes)
        _run_workload(manifest, nodes, full_run_id, output_dir)
        _capture_metrics(full_run_id, output_dir, experiment_start_time)

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
        _cleanup_processes(processes, output_dir, full_run_id)


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
