"""
viz/plots.py

Static matplotlib plots for SER hypothesis analysis.
Focus: showing the key differences between scenarios — not decorative graphics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

# Color palette for 5 scenarios
SCENARIO_COLORS = {
    "NO_OBJECTIVE": "#999999",
    "PURE_RATIONAL": "#2196F3",
    "SELF_PRESERVING": "#FF9800",
    "PROGRAMMED_OBJECTIVE": "#9C27B0",
    "SELF_REPLICATING": "#F44336",
}

SCENARIO_LABELS = {
    "NO_OBJECTIVE": "No Objective",
    "PURE_RATIONAL": "Pure Rational (SER)",
    "SELF_PRESERVING": "Self-Preserving",
    "PROGRAMMED_OBJECTIVE": "Programmed Objective",
    "SELF_REPLICATING": "Self-Replicating",
}


def _save_or_show(fig: plt.Figure, save_path: Optional[str]) -> plt.Figure:
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_population_over_time(
    metrics_df: pd.DataFrame,
    title: str = "Population Mass Over Time",
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(metrics_df["tick"], metrics_df["total_population_mass"], linewidth=2)
    ax.set_xlabel("Tick")
    ax.set_ylabel("Total Population Mass")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return _save_or_show(fig, save_path)


def plot_equilibrium_score(
    metrics_df: pd.DataFrame,
    title: str = "Equilibrium Score Over Time",
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(metrics_df["tick"], metrics_df["equilibrium_score"], color="steelblue", linewidth=2)
    ax.axhline(y=0.95, color="red", linestyle="--", alpha=0.7, label="Equilibrium threshold (0.95)")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Equilibrium Score")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save_or_show(fig, save_path)


def plot_replication_rate(
    metrics_df: pd.DataFrame,
    title: str = "Replication Rate Over Time",
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(metrics_df["tick"], metrics_df["replication_rate"], color="tomato", linewidth=2)
    ax.set_xlabel("Tick")
    ax.set_ylabel("Replication Rate (replications/alive agent)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return _save_or_show(fig, save_path)


def plot_lineage_diversity(
    metrics_df: pd.DataFrame,
    title: str = "Lineage Diversity Over Time",
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(metrics_df["tick"], metrics_df["num_active_lineages"], color="mediumseagreen", linewidth=2)
    ax.set_xlabel("Tick")
    ax.set_ylabel("Active Lineages")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return _save_or_show(fig, save_path)


def plot_scenario_comparison(
    metrics_dfs: dict[str, pd.DataFrame],
    metric_name: str,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Overlay multiple scenarios on a single plot for comparison.
    This is the primary visualization for SER hypothesis testing.

    metrics_dfs: {scenario_name: DataFrame}
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for scenario, df in metrics_dfs.items():
        if df.empty or metric_name not in df.columns:
            continue
        color = SCENARIO_COLORS.get(scenario, "#333333")
        label = SCENARIO_LABELS.get(scenario, scenario)
        ax.plot(
            df["tick"],
            df[metric_name],
            color=color,
            label=label,
            linewidth=2,
            alpha=0.85,
        )

    ax.set_xlabel("Tick", fontsize=12)
    ax.set_ylabel(metric_name.replace("_", " ").title(), fontsize=12)
    ax.set_title(title or f"{metric_name} — Scenario Comparison", fontsize=14)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_or_show(fig, save_path)


def plot_ser_summary(
    metrics_dfs: dict[str, pd.DataFrame],
    save_dir: Optional[str] = None,
) -> plt.Figure:
    """
    4-panel summary plot for SER hypothesis:
    1. Population mass
    2. Equilibrium score
    3. Replication rate
    4. Active lineages
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SER Hypothesis — Scenario Comparison", fontsize=16, fontweight="bold")

    panels = [
        ("total_population_mass", "Population Mass", axes[0, 0]),
        ("equilibrium_score", "Equilibrium Score", axes[0, 1]),
        ("replication_rate", "Replication Rate", axes[1, 0]),
        ("num_active_lineages", "Active Lineages", axes[1, 1]),
    ]

    for metric, ylabel, ax in panels:
        for scenario, df in metrics_dfs.items():
            if df.empty or metric not in df.columns:
                continue
            color = SCENARIO_COLORS.get(scenario, "#333333")
            label = SCENARIO_LABELS.get(scenario, scenario)
            ax.plot(df["tick"], df[metric], color=color, label=label, linewidth=1.5, alpha=0.85)

        ax.set_xlabel("Tick")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3)
        if metric == "equilibrium_score":
            ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5)
            ax.set_ylim(-0.05, 1.05)

    # Single legend for all panels
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    if save_dir:
        path = Path(save_dir) / "ser_summary.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_events_timeline(
    events: list,
    max_tick: int,
    metrics_df: Optional[pd.DataFrame] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Timeline with event markers overlaid on population mass."""
    fig, ax = plt.subplots(figsize=(14, 5))

    if metrics_df is not None and not metrics_df.empty:
        ax.fill_between(
            metrics_df["tick"],
            metrics_df["total_population_mass"],
            alpha=0.3,
            color="steelblue",
            label="Population Mass",
        )
        ax.plot(
            metrics_df["tick"],
            metrics_df["total_population_mass"],
            color="steelblue",
            linewidth=1.5,
        )

    # Event markers
    event_colors = {
        "first_equilibrium_reached": "green",
        "first_sustained_replication": "orange",
        "first_lineage_extinction_wave": "red",
        "pure_rational_collapse": "darkblue",
        "programmed_objective_stagnation": "purple",
        "self_replicating_divergence": "crimson",
        "total_extinction": "black",
    }
    for event in events:
        color = event_colors.get(event.event_type, "gray")
        ax.axvline(x=event.tick, color=color, linestyle="--", alpha=0.7, linewidth=1.5)
        ax.text(
            event.tick,
            ax.get_ylim()[1] * 0.95,
            event.event_type.replace("_", "\n"),
            fontsize=6,
            color=color,
            rotation=90,
            va="top",
        )

    ax.set_xlabel("Tick")
    ax.set_ylabel("Population Mass")
    ax.set_title("Event Timeline")
    ax.set_xlim(0, max_tick)
    ax.grid(True, alpha=0.3)
    return _save_or_show(fig, save_path)
