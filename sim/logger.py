"""
sim/logger.py

Persists simulation data to disk efficiently.
- Metrics: saved every tick (small rows)
- Agent/world snapshots: saved every N ticks (larger)
- Events: saved as JSON
- Config: saved as YAML
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from sim.config import LoggingConfig, SimulationConfig
from sim.event_detector import SimEvent
from sim.metrics import TickMetrics


class SimLogger:
    """
    Handles all I/O for a simulation run.
    Call log_tick() each tick, flush() periodically, finalize() at the end.
    """

    def __init__(self, config: LoggingConfig, run_id: str) -> None:
        self.config = config
        self.run_id = run_id
        self._run_dir = Path(config.output_dir) / run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)
        (self._run_dir / "agents").mkdir(exist_ok=True)
        (self._run_dir / "world_snapshots").mkdir(exist_ok=True)

        # In-memory accumulators
        self._metrics_buffer: list[dict] = []
        self._agent_buffer: dict[int, list[dict]] = {}   # tick -> rows
        self._world_buffer: dict[int, list[dict]] = {}   # tick -> rows
        self._events: list[dict] = []

    # ------------------------------------------------------------------
    # Per-tick logging
    # ------------------------------------------------------------------

    def log_tick(
        self,
        tick: int,
        metrics: TickMetrics,
        agents_snapshot: list[dict],
        world_snapshot: dict,
    ) -> None:
        """Accumulate tick data. Flushes to disk every save_every_n_ticks."""
        self._metrics_buffer.append(metrics.to_dict())

        # Only snapshot agents/world every N ticks
        if tick % self.config.save_every_n_ticks == 0:
            self._agent_buffer[tick] = agents_snapshot
            self._world_buffer[tick] = world_snapshot.get("nodes", [])

        if len(self._metrics_buffer) >= 100:
            self._flush_metrics()

    def log_events(self, events: list[SimEvent]) -> None:
        for e in events:
            self._events.append(e.to_dict())

    # ------------------------------------------------------------------
    # Flush / Finalize
    # ------------------------------------------------------------------

    def _flush_metrics(self) -> None:
        if not self._metrics_buffer:
            return
        df = pd.DataFrame(self._metrics_buffer)
        path = self._run_dir / "metrics_partial.parquet"
        if path.exists():
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df], ignore_index=True)
        df.to_parquet(path, index=False)
        self._metrics_buffer.clear()

    def _flush_snapshots(self) -> None:
        for tick, rows in self._agent_buffer.items():
            if rows:
                df = pd.DataFrame(rows)
                df.to_parquet(
                    self._run_dir / "agents" / f"tick_{tick:06d}.parquet",
                    index=False,
                )
        self._agent_buffer.clear()

        for tick, rows in self._world_buffer.items():
            if rows:
                df = pd.DataFrame(rows)
                df.to_parquet(
                    self._run_dir / "world_snapshots" / f"tick_{tick:06d}.parquet",
                    index=False,
                )
        self._world_buffer.clear()

    def flush(self) -> None:
        self._flush_metrics()
        self._flush_snapshots()

    def save_config(self, config: SimulationConfig) -> None:
        """Save config as YAML in the run directory (matches default.yaml structure)."""
        cfg_dict = {
            "simulation": {
                "seed": config.seed,
                "max_ticks": config.max_ticks,
                "scenario": config.scenario.value,
            },
            "world": config.world.__dict__,
            "agent": config.agent.__dict__,
            "replication": config.replication.__dict__,
            "mutation": {
                "magnitude": config.mutation.magnitude,
                "targets": config.mutation.targets,
            },
            "metrics": config.metrics.__dict__,
            "logging": config.logging.__dict__,
        }
        with open(self._run_dir / "config.yaml", "w") as f:
            yaml.dump(cfg_dict, f, default_flow_style=False)

    def finalize(self, final_metrics: TickMetrics | None = None) -> None:
        """Final flush and write summary + events."""
        self.flush()

        # Merge partial metrics file to final
        partial = self._run_dir / "metrics_partial.parquet"
        final_path = self._run_dir / "metrics.parquet"
        if partial.exists():
            partial.rename(final_path)

        # Events JSON
        with open(self._run_dir / "events.json", "w") as f:
            json.dump(self._events, f, indent=2)

        # Summary JSON
        summary: dict = {
            "run_id": self.run_id,
            "completed_at": datetime.now().isoformat(),
            "num_events": len(self._events),
            "events": [e["event_type"] for e in self._events],
        }
        if final_metrics:
            summary["final_metrics"] = final_metrics.to_dict()

        with open(self._run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    @property
    def run_dir(self) -> Path:
        return self._run_dir
