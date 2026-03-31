"""
sim/replay_loader.py

Loads saved simulation data for visualization and analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sim.config import SimulationConfig, load_config
from sim.event_detector import SimEvent


class ReplayLoader:
    """
    Provides read access to a completed simulation run's output directory.
    Used by viz/ and experiments/analyze_results.py.
    """

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        if not self.run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

    def load_config(self) -> SimulationConfig:
        cfg_path = self.run_dir / "config.yaml"
        return load_config(cfg_path)

    def load_metrics(self) -> pd.DataFrame:
        """Load the full metrics timeseries."""
        path = self.run_dir / "metrics.parquet"
        if not path.exists():
            # Try partial (run still in progress or crashed)
            path = self.run_dir / "metrics_partial.parquet"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def load_agents_at_tick(self, tick: int) -> pd.DataFrame:
        """Load agent snapshot at the nearest saved tick."""
        agents_dir = self.run_dir / "agents"
        if not agents_dir.exists():
            return pd.DataFrame()
        snapshots = sorted(agents_dir.glob("tick_*.parquet"))
        if not snapshots:
            return pd.DataFrame()
        # Find nearest saved tick
        ticks = [int(p.stem.split("_")[1]) for p in snapshots]
        nearest = min(ticks, key=lambda t: abs(t - tick))
        return pd.read_parquet(agents_dir / f"tick_{nearest:06d}.parquet")

    def load_world_at_tick(self, tick: int) -> pd.DataFrame:
        """Load world snapshot at the nearest saved tick."""
        world_dir = self.run_dir / "world_snapshots"
        if not world_dir.exists():
            return pd.DataFrame()
        snapshots = sorted(world_dir.glob("tick_*.parquet"))
        if not snapshots:
            return pd.DataFrame()
        ticks = [int(p.stem.split("_")[1]) for p in snapshots]
        nearest = min(ticks, key=lambda t: abs(t - tick))
        return pd.read_parquet(world_dir / f"tick_{nearest:06d}.parquet")

    def load_events(self) -> list[SimEvent]:
        path = self.run_dir / "events.json"
        if not path.exists():
            return []
        with open(path) as f:
            raw = json.load(f)
        return [
            SimEvent(
                tick=e["tick"],
                event_type=e["event_type"],
                description=e["description"],
                data=e.get("data", {}),
            )
            for e in raw
        ]

    def load_summary(self) -> dict:
        path = self.run_dir / "summary.json"
        if not path.exists():
            return {}
        with open(path) as f:
            return json.load(f)

    def available_ticks(self) -> list[int]:
        """Return list of ticks with saved agent snapshots."""
        agents_dir = self.run_dir / "agents"
        if not agents_dir.exists():
            return []
        return sorted(
            int(p.stem.split("_")[1])
            for p in agents_dir.glob("tick_*.parquet")
        )
