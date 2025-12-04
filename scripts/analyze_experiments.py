#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml


REPO_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "artifacts" / "experiments"
ANALYSIS_DIR = REPO_ROOT / "analysis"


def analyze_experiments() -> None:
    ANALYSIS_DIR.mkdir(exist_ok=True)

    runs = []

    if not EXPERIMENTS_DIR.exists():
        print(f"No experiments directory found at {EXPERIMENTS_DIR}")
        return

    for run_dir in EXPERIMENTS_DIR.iterdir():
        if not run_dir.is_dir():
            continue

        manifest_path = run_dir / "manifest.yaml"
        if not manifest_path.exists():
            continue

        with manifest_path.open() as f:
            manifest = yaml.safe_load(f)

        run_id = manifest.get("run_id", run_dir.name)
        notes = manifest.get("notes", "")

        # Load profile summary
        profile_dir = run_dir / "profile_data"
        summary_csv = profile_dir / "summary.csv"

        if not summary_csv.exists():
            print(f"Warning: No summary.csv found for {run_dir.name}")
            continue

        try:
            df_summary = pd.read_csv(summary_csv)

            for _, row in df_summary.iterrows():
                run_data = {
                    "run_id": run_id,
                    "full_run_id": run_dir.name,
                    "notes": notes,
                    "batch_size": row.get("batch_size"),
                    "throughput": row.get("throughput_req_per_min"),
                    "latency_p50": row.get("latency_p50_ms"),
                    "latency_p95": row.get("latency_p95_ms"),
                    "success_rate": row.get("success_rate"),
                }

                # Load process stats
                stats_file = run_dir / "process_stats.json"
                if stats_file.exists():
                    with stats_file.open() as f:
                        stats = json.load(f)
                        # Sum RSS for all python processes
                        total_rss = sum(s["memory_info"]["rss"] for s in stats) / (
                            1024 * 1024
                        )  # MB
                        run_data["total_rss_mb"] = total_rss

                runs.append(run_data)
        except Exception as e:
            print(f"Error processing {run_dir.name}: {e}")

    if not runs:
        print("No experiment data found.")
        return

    df = pd.DataFrame(runs)
    print("Consolidated Results:")
    print(df)

    df.to_csv(ANALYSIS_DIR / "consolidated_results.csv", index=False)

    # Plots
    try:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df, x="throughput", y="latency_p95", hue="run_id", style="batch_size", s=100
        )
        plt.title("Throughput vs P95 Latency")
        plt.xlabel("Throughput (req/min)")
        plt.ylabel("P95 Latency (ms)")
        plt.grid(True)
        plt.savefig(ANALYSIS_DIR / "throughput_vs_latency.png")
        print(f"Saved plot to {ANALYSIS_DIR / 'throughput_vs_latency.png'}")

        # Bar chart for throughput
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x="run_id", y="throughput", hue="batch_size")
        plt.title("Throughput by Configuration")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(ANALYSIS_DIR / "throughput_bar.png")
        print(f"Saved plot to {ANALYSIS_DIR / 'throughput_bar.png'}")
    except Exception as e:
        print(f"Error generating plots: {e}")


if __name__ == "__main__":
    analyze_experiments()
