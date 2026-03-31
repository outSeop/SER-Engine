"""
viz/timeline.py

Event-annotated timeline visualization.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def render_timeline(
    metrics_df: pd.DataFrame,
    events: list,
    scenario_name: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Render population mass as area chart with event vertical lines.
    Shows the key narrative moments of a simulation run.
    """
    if metrics_df.empty:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Simulation Timeline — {scenario_name}", fontsize=14)

    ticks = metrics_df["tick"]

    # Top: population mass + alive agents
    ax1.fill_between(ticks, metrics_df["total_population_mass"], alpha=0.35, color="steelblue")
    ax1.plot(ticks, metrics_df["total_population_mass"], color="steelblue", linewidth=1.5, label="Population Mass")
    ax1_twin = ax1.twinx()
    ax1_twin.plot(ticks, metrics_df["num_alive_agents"], color="darkorange", linewidth=1.2, linestyle="--", label="Alive Agents")
    ax1.set_ylabel("Population Mass", color="steelblue")
    ax1_twin.set_ylabel("Alive Agents", color="darkorange")

    # Bottom: equilibrium score + replication rate
    ax2.plot(ticks, metrics_df["equilibrium_score"], color="mediumseagreen", linewidth=1.5, label="Equilibrium Score")
    ax2.axhline(y=0.95, color="mediumseagreen", linestyle=":", alpha=0.5)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(ticks, metrics_df["replication_rate"], color="tomato", linewidth=1.2, linestyle="--", label="Replication Rate")
    ax2.set_ylabel("Equilibrium Score", color="mediumseagreen")
    ax2_twin.set_ylabel("Replication Rate", color="tomato")
    ax2.set_xlabel("Tick")
    ax2.set_ylim(-0.05, 1.05)

    # Event markers on both axes
    event_colors = {
        "first_equilibrium_reached": "green",
        "first_sustained_replication": "orange",
        "first_lineage_extinction_wave": "red",
        "pure_rational_collapse": "navy",
        "programmed_objective_stagnation": "purple",
        "self_replicating_divergence": "crimson",
        "total_extinction": "black",
    }
    max_y1 = metrics_df["total_population_mass"].max()
    for event in events:
        color = event_colors.get(event.event_type, "gray")
        for ax in [ax1, ax2]:
            ax.axvline(x=event.tick, color=color, linestyle="--", alpha=0.6, linewidth=1.2)
        # Label on top panel
        ax1.text(
            event.tick, max_y1 * 0.9,
            event.event_type.replace("_", "\n"),
            fontsize=6, color=color, rotation=90, va="top", ha="right",
        )

    ax1.grid(True, alpha=0.25)
    ax2.grid(True, alpha=0.25)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig
