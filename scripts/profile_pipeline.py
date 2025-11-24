#!/usr/bin/env python3
"""
Profiling script for measuring pipeline performance across different batch sizes.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import requests

# Path configuration
REPO_ROOT = Path(__file__).parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "profile"

# Test queries
TEST_QUERIES = [
    "How do I return a defective product?",
    "What is your refund policy?",
    "My order hasn't arrived yet, tracking number is ABC123",
    "How do I update my billing information?",
    "Is there a warranty on electronic items?",
    "Can I change my shipping address after placing an order?",
    "What payment methods do you accept?",
    "How long does shipping typically take?",
    "How do I track my order?",
    "What is your customer service phone number?",
    "Do you ship internationally?",
    "How do I cancel my order?",
    "What is your privacy policy?",
    "Can I use multiple payment methods?",
    "How do I create an account?",
    "What are your business hours?",
]


def send_request(
    server_url: str, request_id: str, query: str, timeout: int = 300
) -> dict[str, Any]:
    """
    Send a single request to the pipeline.

    Returns:
        Dict with keys: request_id, query, success, latency_ms, response, error
    """
    payload = {"request_id": request_id, "query": query}

    start_time = time.perf_counter()
    result: dict[str, Any] = {
        "request_id": request_id,
        "query": query,
        "timestamp": start_time,
    }

    try:
        response = requests.post(server_url, json=payload, timeout=timeout)
        latency = (time.perf_counter() - start_time) * 1000  # ms

        if response.status_code == 200:
            response_data = response.json()
            result.update(
                {
                    "success": True,
                    "latency_ms": round(latency, 2),
                    "response": response_data,
                    "error": None,
                }
            )
        else:
            result.update(
                {
                    "success": False,
                    "latency_ms": round(latency, 2),
                    "response": None,
                    "error": f"HTTP {response.status_code}: {response.text}",
                }
            )

    except requests.exceptions.Timeout:
        latency = (time.perf_counter() - start_time) * 1000
        result.update(
            {
                "success": False,
                "latency_ms": round(latency, 2),
                "response": None,
                "error": "Request timeout",
            }
        )
    except Exception as e:
        latency = (time.perf_counter() - start_time) * 1000
        result.update(
            {
                "success": False,
                "latency_ms": round(latency, 2),
                "response": None,
                "error": str(e),
            }
        )

    return result


def clear_caches(server_url: str) -> None:
    """Clear caches on all services via Gateway."""
    try:
        url = server_url.replace("/query", "/clear_cache")
        response = requests.post(url, timeout=10)
        if response.status_code == 200:
            print(f"Caches cleared: {response.json()}")
        else:
            print(f"Failed to clear caches: {response.status_code} {response.text}")
    except Exception as e:
        print(f"Failed to clear caches: {e}")


def run_profiling_batch(
    server_url: str,
    num_requests: int,
    output_file: Path,
    rate_limit: float = 0.1,
    concurrency: int = 1,
    randomize_queries: bool = False,
) -> list[dict[str, Any]]:
    """
    Run a profiling batch by sending N requests.

    Args:
        server_url: URL of Node 0
        num_requests: Number of requests to send
        output_file: Path to write JSONL results
        rate_limit: Minimum seconds between requests (to avoid overwhelming)
        concurrency: Number of concurrent requests
        randomize_queries: Whether to append random suffix to queries

    Returns:
        List of result dictionaries
    """
    print(f"\n{'=' * 70}")
    print(f"Sending {num_requests} requests to {server_url}")
    print(f"Concurrency: {concurrency}")
    print(f"Randomize Queries: {randomize_queries}")
    print(f"Results will be written to: {output_file}")
    print(f"{'=' * 70}\n")

    results = []

    with output_file.open("w") as f, ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for i in range(num_requests):
            # Cycle through queries
            query = TEST_QUERIES[i % len(TEST_QUERIES)]
            if randomize_queries:
                import uuid

                query = f"{query} {uuid.uuid4()}"

            request_id = f"profile_{int(time.time() * 1000)}_{i}"

            print(f"[{i + 1}/{num_requests}] Submitting request {request_id}")

            future = executor.submit(send_request, server_url, request_id, query)
            futures.append(future)

            # Rate limiting (only if concurrency is low, otherwise we want to flood)
            if concurrency == 1 and i < num_requests - 1:
                time.sleep(rate_limit)

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            # Write to JSONL immediately (thread-safe enough for single writer loop)
            f.write(json.dumps(result) + "\n")
            f.flush()

            if result["success"]:
                print(f"  ✓ Success: {result['latency_ms']:.2f}ms")
            else:
                print(f"  ✗ Failed: {result['error']}")

    return results


def analyze_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Analyze profiling results and compute statistics.

    Returns:
        Dict with summary statistics
    """
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    if not successful:
        return {
            "total_requests": len(results),
            "successful": 0,
            "failed": len(failed),
            "success_rate": 0.0,
            "throughput_req_per_min": 0.0,
            "latency_avg_ms": 0.0,
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
            "latency_p99_ms": 0.0,
            "latency_min_ms": 0.0,
            "latency_max_ms": 0.0,
        }

    latencies = [r["latency_ms"] for r in successful]

    # Calculate throughput
    start_time = min(r["timestamp"] for r in results)
    end_time = max(r["timestamp"] for r in results)
    duration_min = (end_time - start_time) / 60.0 if end_time > start_time else 1.0 / 60.0

    return {
        "total_requests": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "success_rate": round(len(successful) / len(results) * 100, 2),
        "throughput_req_per_min": round(len(successful) / duration_min, 2),
        "latency_avg_ms": round(statistics.mean(latencies), 2),
        "latency_p50_ms": round(statistics.median(latencies), 2),
        "latency_p95_ms": round(statistics.quantiles(latencies, n=20)[18], 2),  # 95th percentile
        "latency_p99_ms": round(statistics.quantiles(latencies, n=100)[98], 2),  # 99th percentile
        "latency_min_ms": round(min(latencies), 2),
        "latency_max_ms": round(max(latencies), 2),
    }


def write_summary_csv(summary_data: list[dict[str, Any]], output_file: Path) -> None:
    """Write summary statistics to CSV."""
    if not summary_data:
        print("No summary data to write")
        return

    with output_file.open("w", newline="") as f:
        fieldnames = summary_data[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_data)

    print(f"\nSummary CSV written to: {output_file}")


def update_batch_size_env(batch_size: int) -> None:
    """
    Update environment variable for batch size.
    This would need to restart the pipeline, so we'll just set it for the next run.
    """
    os.environ["GATEWAY_BATCH_SIZE"] = str(batch_size)
    # Note: In practice, you'd need to restart the pipeline with this new config


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Profile pipeline performance across different batch sizes"
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default=None,
        help="Comma-separated list of batch sizes to test (e.g., '1,4,8')",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=12,
        help="Number of requests to send per batch size (default: 12)",
    )
    parser.add_argument(
        "--server",
        type=str,
        default=None,
        help="Server URL (default: from NODE_0_IP env var or localhost:8000)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.1,
        help="Minimum seconds between requests (default: 0.1)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent requests (default: 1)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["default", "saturation", "latency"],
        default=None,
        help="Workload preset (overrides requests and concurrency)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACTS_DIR,
        help="Output directory for results (default: artifacts/profile)",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip interactive prompts",
    )
    parser.add_argument(
        "--randomize-queries",
        action="store_true",
        help="Randomize queries to bypass cache",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear caches before each run",
    )

    args = parser.parse_args()

    # Apply presets
    if args.preset == "saturation":
        print("Using 'saturation' preset: 200 requests, 50 concurrency")
        args.requests = 200
        args.concurrency = 50
    elif args.preset == "latency":
        print("Using 'latency' preset: 50 requests, 1 concurrency")
        args.requests = 50
        args.concurrency = 1

    # Determine batch sizes to test
    if args.batch_sizes:
        batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
    elif "BATCH_SIZES" in os.environ:
        batch_sizes = [int(x.strip()) for x in os.environ["BATCH_SIZES"].split(",")]
    else:
        # Default: test current configuration only
        batch_sizes = [int(os.environ.get("GATEWAY_BATCH_SIZE", "1"))]

    # Determine server URL
    if args.server:
        server_url = args.server if args.server.startswith("http") else f"http://{args.server}"
    else:
        node_0_ip = os.environ.get("NODE_0_IP", "localhost:8000")
        server_url = f"http://{node_0_ip}/query"

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Test server connectivity
    try:
        health_url = server_url.replace("/query", "/health")
        response = requests.get(health_url, timeout=5)
        if response.status_code != 200:
            print(f"Warning: Health check returned status {response.status_code}")
    except Exception as e:
        print(f"Warning: Could not connect to server health endpoint: {e}")
        print("Continuing anyway...\n")

    # Run profiling for each batch size
    all_summaries = []

    for batch_size in batch_sizes:
        print(f"\n{'#' * 70}")
        print(f"# PROFILING WITH BATCH_SIZE = {batch_size}")
        print(f"{'#' * 70}")

        # Note: In a real scenario, you'd restart the pipeline with the new batch size
        # For now, we just document which batch size we're testing
        print(f"\nNote: Ensure pipeline is running with GATEWAY_BATCH_SIZE={batch_size}")
        if not args.no_interactive:
            input("Press Enter when ready to continue...")

        if args.clear_cache:
            clear_caches(server_url)

        # Output files
        jsonl_file = args.output_dir / f"batch_{batch_size}.jsonl"

        # Run profiling
        results = run_profiling_batch(
            server_url,
            args.requests,
            jsonl_file,
            args.rate_limit,
            concurrency=args.concurrency,
            randomize_queries=args.randomize_queries,
        )

        # Analyze results
        summary = analyze_results(results)
        summary["batch_size"] = batch_size
        all_summaries.append(summary)

        # Print summary
        print(f"\n{'=' * 70}")
        print(f"RESULTS FOR BATCH_SIZE = {batch_size}")
        print(f"{'=' * 70}")
        for key, value in summary.items():
            print(f"  {key:25s}: {value}")
        print(f"{'=' * 70}\n")

    # Write aggregated summary
    summary_csv = args.output_dir / "summary.csv"
    write_summary_csv(all_summaries, summary_csv)

    # Print final summary table
    print(f"\n{'=' * 70}")
    print("AGGREGATE SUMMARY")
    print(f"{'=' * 70}")
    print(
        f"{'Batch Size':>12} {'Success%':>10} {'Throughput':>12} {'Avg Latency':>14} {'P95 Latency':>14}"
    )
    print(f"{'-' * 70}")
    for summary in all_summaries:
        print(
            f"{summary['batch_size']:>12} "
            f"{summary['success_rate']:>9.1f}% "
            f"{summary['throughput_req_per_min']:>11.2f}/m "
            f"{summary['latency_avg_ms']:>11.2f}ms "
            f"{summary['latency_p95_ms']:>11.2f}ms"
        )
    print(f"{'=' * 70}\n")

    print(f"All results written to: {args.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
