#!/usr/bin/env python3
import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import httpx

PROMETHEUS_URL = "http://localhost:9090"


def query_prometheus_range(query: str, start: float, end: float, step: str = "5s") -> list[dict]:
    """Query Prometheus range API."""
    url = f"{PROMETHEUS_URL}/api/v1/query_range"
    params: dict[str, Any] = {
        "query": query,
        "start": start,
        "end": end,
        "step": step,
    }
    try:
        resp = httpx.get(url, params=params, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        if data["status"] != "success":
            print(f"Prometheus query failed: {data}", file=sys.stderr)
            return []
        return data["data"]["result"]
    except Exception as e:
        print(f"Error querying Prometheus: {e}", file=sys.stderr)
        return []


def capture_metrics(run_id: str, start_time: float, end_time: float, output_file: str) -> None:
    print(f"Capturing metrics for run {run_id} from {start_time} to {end_time}...")

    rows: list[dict[str, Any]] = []

    # Memory Usage (RSS)
    mem_results = query_prometheus_range(
        f'pipeline_memory_bytes{{run_id="{run_id}", type="rss"}}', start_time, end_time
    )
    for result in mem_results:
        metric = result["metric"]
        values = result["values"]

        for ts, val in values:
            rows.append(
                {
                    "timestamp": datetime.fromtimestamp(float(ts)).isoformat(),
                    "run_id": run_id,
                    "node": metric.get("node", "unknown"),
                    "service": metric.get("service", "unknown"),
                    "stage": "",
                    "metric": "memory_rss_bytes",
                    "value": float(val),
                }
            )

    # Stage Duration
    stage_results = query_prometheus_range(
        f'pipeline_stage_duration_seconds{{run_id="{run_id}"}}', start_time, end_time
    )
    for result in stage_results:
        metric = result["metric"]
        values = result["values"]

        for ts, val in values:
            rows.append(
                {
                    "timestamp": datetime.fromtimestamp(float(ts)).isoformat(),
                    "run_id": run_id,
                    "node": metric.get("node", "unknown"),
                    "service": "",
                    "stage": metric.get("stage", "unknown"),
                    "metric": "stage_duration_seconds",
                    "value": float(val),
                }
            )

    # Request Rate (Throughput)
    rate_results = query_prometheus_range(
        f'rate(pipeline_requests_total{{run_id="{run_id}"}}[30s])', start_time, end_time
    )
    for result in rate_results:
        metric = result["metric"]
        values = result["values"]

        for ts, val in values:
            rows.append(
                {
                    "timestamp": datetime.fromtimestamp(float(ts)).isoformat(),
                    "run_id": run_id,
                    "node": metric.get("node", "unknown"),
                    "service": metric.get("service", "unknown"),
                    "stage": "",
                    "metric": "requests_per_second",
                    "value": float(val),
                }
            )

    # Write to CSV
    output_path = Path(output_file)

    # Sort by timestamp
    rows.sort(key=lambda x: x["timestamp"])

    with output_path.open("w", newline="") as f:
        fieldnames = ["timestamp", "run_id", "node", "service", "stage", "metric", "value"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Captured {len(rows)} metric data points to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--start-time", type=float, required=True)
    parser.add_argument("--end-time", type=float, required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    capture_metrics(args.run_id, args.start_time, args.end_time, args.output_file)
