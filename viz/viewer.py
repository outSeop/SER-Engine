"""
viz/viewer.py

Simple replay viewer. Loads a completed run and provides navigation
through ticks and events. Matplotlib-based (no web framework required).

Extension points (TODO):
- lineage highlight mode
- hub capture emphasis
- strategy-type filtering
- metric overlay
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from sim.replay_loader import ReplayLoader
from viz.graph_renderer import render_graph_state
from viz.timeline import render_timeline


class ReplayViewer:
    """
    Interactive matplotlib-based replay viewer.

    Usage:
        viewer = ReplayViewer("output/SELF_REPLICATING_42_...")
        viewer.show()
    """

    def __init__(self, run_dir: str | Path) -> None:
        self.loader = ReplayLoader(run_dir)
        self.metrics_df = self.loader.load_metrics()
        self.events = self.loader.load_events()
        self.available_ticks = self.loader.available_ticks()

        cfg = self.loader.load_config()
        # Rebuild the graph topology from config for layout
        import networkx as nx
        self._graph = nx.watts_strogatz_graph(
            n=cfg.world.num_nodes,
            k=cfg.world.k_nearest,
            p=cfg.world.rewire_probability,
            seed=cfg.seed,
        )
        self._current_tick_idx: int = 0

    def show(self) -> None:
        """Launch the interactive matplotlib viewer."""
        if not self.available_ticks:
            print("No snapshots found. Run the simulation first.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle("SER Simulation Replay", fontsize=14)
        plt.subplots_adjust(bottom=0.15)

        # Add tick slider
        from matplotlib.widgets import Button, Slider
        ax_slider = plt.axes([0.15, 0.04, 0.6, 0.03])
        slider = Slider(
            ax_slider, "Tick",
            valmin=0, valmax=max(self.available_ticks),
            valinit=self.available_ticks[0],
            valstep=1,
        )

        # Event jump buttons
        ax_prev = plt.axes([0.05, 0.04, 0.08, 0.04])
        ax_next = plt.axes([0.80, 0.04, 0.08, 0.04])
        btn_prev = Button(ax_prev, "◀ Event")
        btn_next = Button(ax_next, "Event ▶")

        def redraw(tick: int) -> None:
            for ax in axes:
                ax.clear()

            world_df = self.loader.load_world_at_tick(tick)
            agents_df = self.loader.load_agents_at_tick(tick)

            # Left: graph view
            render_graph_state(
                world_df=world_df,
                agents_df=agents_df,
                graph=self._graph,
                tick=tick,
                title=f"Graph — Tick {tick}",
            )
            # Right: metrics timeline up to current tick
            if not self.metrics_df.empty:
                subset = self.metrics_df[self.metrics_df["tick"] <= tick]
                axes[1].plot(subset["tick"], subset["total_population_mass"], color="steelblue")
                axes[1].axvline(x=tick, color="red", linestyle="--", alpha=0.5)
                axes[1].set_xlabel("Tick")
                axes[1].set_ylabel("Population Mass")
                axes[1].set_title("Population Mass")
                axes[1].grid(True, alpha=0.3)

            fig.canvas.draw_idle()

        def on_slider(val: float) -> None:
            tick = int(val)
            nearest = min(self.available_ticks, key=lambda t: abs(t - tick))
            redraw(nearest)

        def on_prev(event) -> None:
            current = int(slider.val)
            prev_events = [e for e in self.events if e.tick < current]
            if prev_events:
                target = prev_events[-1].tick
                slider.set_val(target)

        def on_next(event) -> None:
            current = int(slider.val)
            next_events = [e for e in self.events if e.tick > current]
            if next_events:
                target = next_events[0].tick
                slider.set_val(target)

        slider.on_changed(on_slider)
        btn_prev.on_clicked(on_prev)
        btn_next.on_clicked(on_next)

        redraw(self.available_ticks[0])
        plt.show()

    def jump_to_tick(self, tick: int) -> None:
        """Show a single static frame at the given tick."""
        world_df = self.loader.load_world_at_tick(tick)
        agents_df = self.loader.load_agents_at_tick(tick)
        render_graph_state(world_df, agents_df, self._graph, tick)
        plt.show()

    def show_timeline(self) -> None:
        """Show the full event timeline for this run."""
        cfg = self.loader.load_config()
        render_timeline(
            self.metrics_df,
            self.events,
            scenario_name=cfg.scenario.value,
        )
        plt.show()
