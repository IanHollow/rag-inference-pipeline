#!/usr/bin/env python3
import json
from pathlib import Path
import traceback

from matplotlib.container import BarContainer
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml


REPO_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "artifacts" / "experiments"
ANALYSIS_DIR = REPO_ROOT / "analysis"


def _is_numeric(value: str) -> bool:
    """Check if a string represents a numeric value."""
    try:
        float(value)
    except ValueError:
        return False
    else:
        return True


def _generate_throughput_vs_latency_plot(df: pd.DataFrame) -> None:
    """Generate scatter plot of throughput vs P95 latency."""
    _fig, ax = plt.subplots(figsize=(14, 8))

    sns.scatterplot(
        data=df,
        x="throughput",
        y="latency_p95",
        hue="run_id",
        style="batch_size",
        s=120,
        ax=ax,
        legend=False,
    )

    plt.title("Throughput vs P95 Latency", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Throughput (req/min)", fontsize=12)
    plt.ylabel("P95 Latency (ms)", fontsize=12)
    plt.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()

    run_id_handles = []
    run_id_labels = []
    batch_handles = []
    batch_labels = []

    for handle, label in zip(handles, labels, strict=False):
        if _is_numeric(label):
            batch_handles.append(handle)
            batch_labels.append(f"Batch: {label}")
        else:
            run_id_handles.append(handle)
            run_id_labels.append(label)

    ax.legend(
        run_id_handles[:10] + batch_handles,
        run_id_labels[:10] + batch_labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        title="Configuration",
        title_fontsize=9,
        framealpha=0.9,
        ncol=1,
    )

    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "throughput_vs_latency.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {ANALYSIS_DIR / 'throughput_vs_latency.png'}")


def _generate_throughput_bar_plot(df: pd.DataFrame) -> None:
    """Generate bar chart of throughput by configuration."""
    _fig, ax = plt.subplots(figsize=(16, 8))

    df_sorted = df.sort_values("throughput", ascending=False)

    sns.barplot(data=df_sorted, x="run_id", y="throughput", hue="batch_size", palette="RdPu", ax=ax)

    plt.title("Throughput by Configuration", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Run ID", fontsize=12)
    plt.ylabel("Throughput (req/min)", fontsize=12)
    plt.xticks(rotation=60, ha="right", fontsize=9)
    plt.legend(title="Batch Size", loc="upper right", fontsize=9, title_fontsize=10)

    for container in ax.containers:
        if isinstance(container, BarContainer):
            ax.bar_label(container, fmt="%.1f", fontsize=7, padding=2)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)

    plt.savefig(ANALYSIS_DIR / "throughput_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {ANALYSIS_DIR / 'throughput_bar.png'}")


def _generate_latency_comparison_plot(df: pd.DataFrame) -> None:
    """Generate bar chart comparing P50 and P95 latency."""
    _fig, ax = plt.subplots(figsize=(16, 8))

    df_sorted = df.sort_values("throughput", ascending=False)
    df_latency = df_sorted[["run_id", "latency_p50", "latency_p95"]].melt(
        id_vars=["run_id"],
        value_vars=["latency_p50", "latency_p95"],
        var_name="Percentile",
        value_name="Latency (ms)",
    )
    df_latency["Percentile"] = df_latency["Percentile"].map(
        {"latency_p50": "P50", "latency_p95": "P95"}
    )

    sns.barplot(
        data=df_latency,
        x="run_id",
        y="Latency (ms)",
        hue="Percentile",
        palette=["#4CAF50", "#FF5722"],
        ax=ax,
    )

    plt.title("Latency by Configuration (P50 vs P95)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Run ID", fontsize=12)
    plt.ylabel("Latency (ms)", fontsize=12)
    plt.xticks(rotation=60, ha="right", fontsize=9)
    plt.legend(title="Percentile", loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)

    plt.savefig(ANALYSIS_DIR / "latency_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {ANALYSIS_DIR / 'latency_comparison.png'}")


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
        # Set a clean style
        sns.set_style("whitegrid")

        _generate_throughput_vs_latency_plot(df)
        _generate_throughput_bar_plot(df)
        _generate_latency_comparison_plot(df)

    except Exception as e:
        print(f"Error generating plots: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    analyze_experiments()
