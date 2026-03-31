"""
experiments/analyze_results.py

Compare results across multiple simulation runs.
Generates comparison plots and summary statistics for SER hypothesis testing.

Usage:
    python -m experiments.analyze_results --run-dirs output/run1 output/run2 ...
    python -m experiments.analyze_results --run-dirs output/PURE_RATIONAL_* output/SELF_REPLICATING_*
    python -m experiments.analyze_results --output-dir output/ser_comparison_*/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sim.replay_loader import ReplayLoader
from viz.plots import plot_scenario_comparison, plot_ser_summary


def load_experiment_results(run_dirs: list[str]) -> dict[str, list[pd.DataFrame]]:
    """
    Load metrics from multiple run directories, grouped by scenario.
    Returns: {scenario_name: [metrics_df, ...]}
    """
    results: dict[str, list[pd.DataFrame]] = {}

    for run_dir in run_dirs:
        try:
            loader = ReplayLoader(run_dir)
            cfg = loader.load_config()
            scenario = cfg.scenario.value
            df = loader.load_metrics()
            if not df.empty:
                results.setdefault(scenario, []).append(df)
        except Exception as e:
            print(f"Warning: could not load {run_dir}: {e}")

    return results


def average_metrics(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Average metrics across multiple seeds for the same scenario."""
    if not dfs:
        return pd.DataFrame()
    if len(dfs) == 1:
        return dfs[0]

    # Align on tick index
    max_tick = min(df["tick"].max() for df in dfs)
    aligned = [df[df["tick"] <= max_tick].set_index("tick") for df in dfs]
    avg = pd.concat(aligned).groupby(level=0).mean().reset_index()
    return avg


def compute_summary_statistics(
    results: dict[str, list[pd.DataFrame]],
) -> pd.DataFrame:
    """
    Compute summary statistics per scenario for key SER metrics.
    Returns a DataFrame with one row per scenario.
    """
    rows = []
    for scenario, dfs in results.items():
        for df in dfs:
            if df.empty:
                continue
            final = df.iloc[-1]
            rows.append({
                "scenario": scenario,
                "final_tick": int(final["tick"]),
                "final_alive_agents": int(final.get("num_alive_agents", 0)),
                "final_lineages": int(final.get("num_active_lineages", 0)),
                "final_equilibrium_score": float(final.get("equilibrium_score", 0)),
                "mean_replication_rate": float(df["replication_rate"].mean()),
                "total_replications": float(df["replication_rate"].sum()),
                "max_lineage_diversity": float(df["num_active_lineages"].max()),
                "persistence_time": int(final.get("population_persistence_time", 0)),
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def generate_comparison_plots(
    results: dict[str, list[pd.DataFrame]],
    output_dir: str,
) -> None:
    """Generate all SER comparison plots."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Average across seeds
    avg_results = {s: average_metrics(dfs) for s, dfs in results.items()}

    # Main 4-panel SER summary
    plot_ser_summary(avg_results, save_dir=output_dir)
    print(f"  Saved: {output_dir}/ser_summary.png")

    # Individual metric comparisons
    for metric in [
        "total_population_mass",
        "equilibrium_score",
        "replication_rate",
        "num_active_lineages",
        "lineage_diversity",
        "dominance_index",
    ]:
        save_path = str(Path(output_dir) / f"comparison_{metric}.png")
        plot_scenario_comparison(avg_results, metric, save_path=save_path)
        print(f"  Saved: {save_path}")


def generate_report(
    results: dict[str, list[pd.DataFrame]],
    output_dir: str,
) -> None:
    """Write a markdown summary report."""
    stats = compute_summary_statistics(results)
    report_path = Path(output_dir) / "report.md"

    with open(report_path, "w") as f:
        f.write("# SER Simulation Analysis Report\n\n")
        f.write("## Scenario Summary Statistics\n\n")

        if not stats.empty:
            # Group by scenario for mean stats
            grouped = stats.groupby("scenario").agg({
                "final_alive_agents": ["mean", "std"],
                "final_equilibrium_score": ["mean", "std"],
                "mean_replication_rate": ["mean", "std"],
                "total_replications": ["mean", "std"],
                "persistence_time": ["mean", "std"],
            }).round(3)
            f.write(grouped.to_markdown())
            f.write("\n\n")

        f.write("## Data-driven Notes\n\n")
        f.write(
            "- This report intentionally avoids hard-coded expected outcomes.\n"
            "- Compare scenarios using observed statistics above (means/std across runs).\n"
            "- If variance is high, increase seeds before drawing conclusions.\n\n"
        )

    print(f"  Saved: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SER simulation results")
    parser.add_argument(
        "--run-dirs", nargs="+", required=True,
        help="Run directories to analyze (supports glob patterns via shell)",
    )
    parser.add_argument(
        "--output", default="output/analysis",
        help="Output directory for plots and report",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    print(f"Loading {len(args.run_dirs)} run(s)...")
    results = load_experiment_results(args.run_dirs)

    if not results:
        print("No valid results found.")
        return

    print(f"Scenarios found: {list(results.keys())}")

    stats = compute_summary_statistics(results)
    if not stats.empty:
        print("\nSummary statistics:")
        print(stats.groupby("scenario")[
            ["final_alive_agents", "final_equilibrium_score", "mean_replication_rate", "persistence_time"]
        ].mean().round(3).to_string())

    if not args.no_plots:
        print(f"\nGenerating plots → {args.output}")
        generate_comparison_plots(results, args.output)

    generate_report(results, args.output)
    print(f"\nAnalysis complete. Output: {args.output}")


if __name__ == "__main__":
    main()
