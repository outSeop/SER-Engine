"""
viz/graph_renderer.py

Renders the graph environment state at a given tick.
Node size = population mass, color = resource level, edge = flow intensity.

Also provides animate_graph() to produce a GIF showing how the graph evolves
tick-by-tick across the whole simulation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import networkx as nx
import numpy as np
import pandas as pd


def render_graph_state(
    world_df: pd.DataFrame,
    agents_df: pd.DataFrame,
    graph: nx.Graph,
    tick: int,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Render the simulation graph at a specific tick.

    Visual encoding:
    - Node size: total population_mass of agents on this node
    - Node color: resource_level (green=rich, red=depleted)
    - Edge width: trade_efficiency (not saved per tick, uses uniform default)
    - Agent dots: colored by lineage (up to 20 lineages)

    world_df: DataFrame from ReplayLoader.load_world_at_tick()
    agents_df: DataFrame from ReplayLoader.load_agents_at_tick()
    """
    fig, ax = plt.subplots(figsize=(12, 9))

    if world_df.empty:
        ax.text(0.5, 0.5, "No world data at this tick", ha="center", va="center")
        ax.set_title(title or f"Graph State — Tick {tick}")
        return fig

    # Build node_id → resource_level mapping
    resource_by_node: dict[int, float] = {}
    capacity_by_node: dict[int, float] = {}
    for _, row in world_df.iterrows():
        nid = int(row["node_id"])
        resource_by_node[nid] = float(row.get("resource_level", 0))
        capacity_by_node[nid] = float(row.get("resource_capacity", 1))

    # Build node_id → population_mass mapping from agents
    mass_by_node: dict[int, float] = {}
    if not agents_df.empty:
        for _, row in agents_df.iterrows():
            nid = int(row["node_id"])
            mass_by_node[nid] = mass_by_node.get(nid, 0) + float(row.get("population_mass", 0))

    # Layout
    pos = nx.spring_layout(graph, seed=42)

    node_ids = list(graph.nodes())

    # Node sizes: proportional to population mass (min size for visibility)
    max_mass = max(mass_by_node.values(), default=1.0)
    node_sizes = [
        max(100, (mass_by_node.get(n, 0) / max(max_mass, 1)) * 1500)
        for n in node_ids
    ]

    # Node colors: resource fraction (green=1.0, red=0.0)
    resource_fractions = [
        resource_by_node.get(n, 0) / max(capacity_by_node.get(n, 1), 1)
        for n in node_ids
    ]
    cmap = plt.cm.RdYlGn
    node_colors = [cmap(f) for f in resource_fractions]

    nx.draw_networkx_edges(
        graph, pos, ax=ax,
        alpha=0.3, width=0.8, edge_color="#888888",
    )
    nx.draw_networkx_nodes(
        graph, pos, ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.85,
    )
    nx.draw_networkx_labels(
        graph, pos, ax=ax,
        font_size=6, font_color="black", alpha=0.7,
    )

    # Colorbar for resource level
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04, label="Resource Level (fraction)")

    ax.set_title(title or f"Graph State — Tick {tick}", fontsize=13)
    ax.axis("off")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def _draw_frame(
    ax: plt.Axes,
    world_df: pd.DataFrame,
    agents_df: pd.DataFrame,
    graph: nx.Graph,
    pos: dict,
    tick: int,
    scenario_name: str,
    global_max_mass: float,
) -> None:
    """Draw one animation frame into ax (clears first)."""
    ax.clear()

    resource_by_node: dict[int, float] = {}
    capacity_by_node: dict[int, float] = {}
    if not world_df.empty:
        for _, row in world_df.iterrows():
            nid = int(row["node_id"])
            resource_by_node[nid] = float(row.get("resource_level", 0))
            capacity_by_node[nid] = float(row.get("resource_capacity", 1))

    mass_by_node: dict[int, float] = {}
    lineage_by_node: dict[int, list[str]] = {}
    if not agents_df.empty:
        for _, row in agents_df.iterrows():
            nid = int(row["node_id"])
            mass_by_node[nid] = mass_by_node.get(nid, 0) + float(row.get("population_mass", 0))
            lineage_by_node.setdefault(nid, []).append(str(row.get("lineage_id", "")))

    node_ids = list(graph.nodes())
    node_sizes = [
        max(60, (mass_by_node.get(n, 0) / max(global_max_mass, 1)) * 1200)
        for n in node_ids
    ]
    resource_fractions = [
        resource_by_node.get(n, 0) / max(capacity_by_node.get(n, 1), 1)
        for n in node_ids
    ]
    cmap = plt.cm.RdYlGn
    node_colors = [cmap(f) for f in resource_fractions]

    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.25, width=0.6, edge_color="#aaaaaa")
    nx.draw_networkx_nodes(
        graph, pos, ax=ax,
        node_size=node_sizes, node_color=node_colors, alpha=0.88,
    )

    # Highlight occupied nodes with a ring
    occupied = [n for n in node_ids if mass_by_node.get(n, 0) > 0]
    if occupied:
        occ_sizes = [mass_by_node.get(n, 0) / max(global_max_mass, 1) * 1500 + 100 for n in occupied]
        occ_pos = {n: pos[n] for n in occupied}
        nx.draw_networkx_nodes(
            graph, occ_pos, nodelist=occupied, ax=ax,
            node_size=occ_sizes, node_color="none",
            edgecolors="steelblue", linewidths=1.5, alpha=0.7,
        )

    total_alive = len(agents_df) if not agents_df.empty else 0
    total_lineages = agents_df["lineage_id"].nunique() if not agents_df.empty and "lineage_id" in agents_df.columns else 0
    ax.set_title(
        f"{scenario_name}  |  Tick {tick:>4d}  |  Agents: {total_alive}  |  Lineages: {total_lineages}",
        fontsize=10,
    )
    ax.axis("off")


def animate_graph(
    run_dir: str | Path,
    save_path: Optional[str] = None,
    fps: int = 4,
    max_frames: int = 100,
    dpi: int = 100,
) -> None:
    """
    Generate an animated GIF showing how the graph evolves tick by tick.

    - Node size  : total population_mass on each node (globally normalised)
    - Node color : resource level fraction (red=depleted → green=rich)
    - Blue ring  : nodes that have at least one alive agent

    Parameters
    ----------
    run_dir    : path to a completed simulation run directory
    save_path  : output .gif path (default: <run_dir>/graph_animation.gif)
    fps        : frames per second in the GIF
    max_frames : cap the number of frames (subsamples if necessary)
    dpi        : resolution
    """
    from sim.replay_loader import ReplayLoader
    from sim.config import load_config

    run_dir = Path(run_dir)
    loader = ReplayLoader(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    scenario_name = cfg.scenario.value

    available = loader.available_ticks()
    if not available:
        print("No snapshots found in run directory.")
        return

    # Subsample if too many frames
    if len(available) > max_frames:
        step = len(available) // max_frames
        available = available[::step]

    # Fixed graph layout (same seed as engine)
    graph = nx.watts_strogatz_graph(
        n=cfg.world.num_nodes,
        k=cfg.world.k_nearest,
        p=cfg.world.rewire_probability,
        seed=cfg.seed,
    )
    pos = nx.spring_layout(graph, seed=42)

    # Pre-compute global max mass for consistent node sizes
    print(f"Loading {len(available)} snapshots for animation…")
    frames_data: list[tuple[pd.DataFrame, pd.DataFrame, int]] = []
    global_max_mass = 1.0
    for tick in available:
        w = loader.load_world_at_tick(tick)
        a = loader.load_agents_at_tick(tick)
        frames_data.append((w, a, tick))
        if not a.empty and "population_mass" in a.columns:
            node_total = a.groupby("node_id")["population_mass"].sum()
            if not node_total.empty:
                global_max_mass = max(global_max_mass, node_total.max())

    # Build animation
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("#f8f8f8")

    # Colourbar (static)
    cmap = plt.cm.RdYlGn
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Resource level (fraction)", fontsize=8)

    def update(frame_idx: int):
        w, a, tick = frames_data[frame_idx]
        _draw_frame(ax, w, a, graph, pos, tick, scenario_name, global_max_mass)

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(frames_data),
        interval=int(1000 / fps),
        blit=False,
    )

    out_path = save_path or str(run_dir / "graph_animation.gif")
    print(f"Saving animation → {out_path}")
    ani.save(out_path, writer="pillow", fps=fps, dpi=dpi)
    plt.close(fig)
    print("Done.")
