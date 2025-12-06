#!/usr/bin/env python3
"""
Regenerate improved plots from tier2-test-data consolidated results.
"""

from pathlib import Path

from matplotlib.container import BarContainer
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).parent.parent
TIER2_ANALYSIS_DIR = REPO_ROOT / "tier2-test-data" / "analysis"
CSV_FILE = TIER2_ANALYSIS_DIR / "consolidated_results.csv"


def generate_plots() -> None:
    if not CSV_FILE.exists():
        print(f"CSV file not found: {CSV_FILE}")
        return

    df = pd.read_csv(CSV_FILE)
    print(f"Loaded {len(df)} records from {CSV_FILE}")
    print(df.head())

    # Set a clean style
    sns.set_style("whitegrid")
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12

    # ============================================================
    # Scatter plot: Throughput vs P95 Latency
    # ============================================================
    _fig, ax = plt.subplots(figsize=(12, 9))

    # Create scatter plot
    sns.scatterplot(
        data=df,
        x="throughput",
        y="latency_p95",
        hue="run_id",
        style="batch_size",
        s=120,
        ax=ax,
    )

    plt.title("Throughput vs P95 Latency", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Throughput (req/min)", fontsize=12)
    plt.ylabel("P95 Latency (ms)", fontsize=12)
    plt.grid(True, alpha=0.3)

    # Move legend outside the plot to the right
    plt.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=7,
        title="run_id / batch_size",
        title_fontsize=8,
        framealpha=0.9,
        ncol=1,
        markerscale=0.8,
    )

    plt.tight_layout()
    output_path = TIER2_ANALYSIS_DIR / "throughput_vs_latency.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

    # ============================================================
    # Bar chart: Throughput by Configuration
    # ============================================================
    _fig, ax = plt.subplots(figsize=(16, 8))

    # Sort by throughput for better visualization
    df_sorted = df.sort_values("throughput", ascending=False)

    # Create bar plot
    sns.barplot(
        data=df_sorted,
        x="run_id",
        y="throughput",
        hue="batch_size",
        palette="RdPu",
        ax=ax,
        dodge=False,  # Don't dodge since each run_id has one batch_size
    )

    plt.title("Throughput by Configuration", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("", fontsize=12)  # Remove xlabel since it's obvious
    plt.ylabel("Throughput (req/min)", fontsize=12)

    # Rotate x-axis labels significantly and align them
    plt.setp(ax.get_xticklabels(), rotation=55, ha="right", fontsize=8)

    # Move legend to upper right inside plot
    plt.legend(title="Batch Size", loc="upper right", fontsize=9, title_fontsize=10)

    # Add value labels on top of bars
    for container in ax.containers:
        if isinstance(container, BarContainer):
            ax.bar_label(container, fmt="%.1f", fontsize=7, padding=2, rotation=0)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28)

    output_path = TIER2_ANALYSIS_DIR / "throughput_bar.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

    # ============================================================
    # Additional: Latency comparison (P50 vs P95)
    # ============================================================
    _fig, ax = plt.subplots(figsize=(16, 8))

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
    plt.xlabel("", fontsize=12)
    plt.ylabel("Latency (ms)", fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=55, ha="right", fontsize=8)
    plt.legend(title="Percentile", loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28)

    output_path = TIER2_ANALYSIS_DIR / "latency_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

    # ============================================================
    # Memory usage bar chart
    # ============================================================
    if "total_rss_mb" in df.columns:
        _fig, ax = plt.subplots(figsize=(16, 8))

        df_mem = df_sorted[["run_id", "total_rss_mb", "throughput"]].copy()

        sns.barplot(data=df_mem, x="run_id", y="total_rss_mb", palette="Blues_d", ax=ax)

        plt.title("Memory Usage by Configuration", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("", fontsize=12)
        plt.ylabel("Total RSS (MB)", fontsize=12)
        plt.setp(ax.get_xticklabels(), rotation=55, ha="right", fontsize=8)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.28)

        output_path = TIER2_ANALYSIS_DIR / "memory_usage.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")

    # ============================================================
    # Summary scatter: Throughput vs Memory with size = latency
    # ============================================================
    if "total_rss_mb" in df.columns:
        fig, ax = plt.subplots(figsize=(12, 8))

        scatter_plot = ax.scatter(
            df["throughput"],
            df["total_rss_mb"],
            c=df["latency_p95"],
            s=100,
            cmap="RdYlGn_r",
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
        )

        # Add colorbar
        cbar = fig.colorbar(scatter_plot, ax=ax)
        cbar.set_label("P95 Latency (ms)", fontsize=11)

        # Annotate top performers
        top_throughput = df.nlargest(3, "throughput")
        for _, row in top_throughput.iterrows():
            ax.annotate(
                row["run_id"],
                (row["throughput"], row["total_rss_mb"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
                alpha=0.8,
            )

        plt.title(
            "Throughput vs Memory Usage\n(Color = P95 Latency)",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )
        plt.xlabel("Throughput (req/min)", fontsize=12)
        plt.ylabel("Total RSS (MB)", fontsize=12)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = TIER2_ANALYSIS_DIR / "throughput_memory_latency.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    generate_plots()
