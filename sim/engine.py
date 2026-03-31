"""
sim/engine.py

Top-level simulation runner.
Creates and wires all subsystems, runs the tick loop.
Single entry point for executing one simulation run.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np

from sim.agent_system import AgentSystem
from sim.config import SimulationConfig
from sim.event_detector import EventDetector
from sim.graph_world import GraphWorld
from sim.logger import SimLogger
from sim.metrics import MetricsCollector
from sim.scheduler import Scheduler


class SimulationEngine:
    """
    Assembles all simulation components and runs the tick loop.

    Usage:
        config = load_config("config/default.yaml", {"simulation.scenario": "SELF_REPLICATING"})
        engine = SimulationEngine(config)
        run_dir = engine.run()
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self._run_id: str | None = None

        # All components wired in setup()
        self.rng: np.random.Generator | None = None
        self.world: GraphWorld | None = None
        self.agent_system: AgentSystem | None = None
        self.metrics_collector: MetricsCollector | None = None
        self.event_detector: EventDetector | None = None
        self.logger: SimLogger | None = None
        self.scheduler: Scheduler | None = None

    def setup(self) -> None:
        """Initialize all subsystems. Must be called before run()."""
        config = self.config

        # Single seeded RNG shared by all subsystems → full reproducibility
        self.rng = np.random.default_rng(config.seed)

        self._run_id = self._make_run_id()

        self.world = GraphWorld(config.world, self.rng)
        self.agent_system = AgentSystem(config, self.world, self.rng)
        self.agent_system.initialize_agents()

        self.metrics_collector = MetricsCollector(config.metrics)
        self.event_detector = EventDetector(config)
        self.logger = SimLogger(config.logging, self._run_id)
        self.logger.save_config(config)

        self.scheduler = Scheduler(
            config=config,
            world=self.world,
            agent_system=self.agent_system,
            metrics_collector=self.metrics_collector,
            event_detector=self.event_detector,
            logger=self.logger,
            rng=self.rng,
        )

    def run(self, verbose: bool = False) -> str:
        """
        Run the simulation until completion.
        Returns the run directory path as a string.
        """
        if self.scheduler is None:
            self.setup()

        try:
            while self.scheduler.step():
                if verbose and self.scheduler.current_tick % 100 == 0:
                    m = self.metrics_collector.latest()
                    print(
                        f"  tick={self.scheduler.current_tick:5d} | "
                        f"alive={m.num_alive_agents:4d} | "
                        f"lineages={m.num_active_lineages:4d} | "
                        f"equilibrium={m.equilibrium_score:.3f} | "
                        f"replication_rate={m.replication_rate:.3f}"
                    )
        finally:
            final = self.metrics_collector.latest()
            self.logger.finalize(final)

        return str(self.logger.run_dir)

    @property
    def run_id(self) -> str:
        if self._run_id is None:
            self._run_id = self._make_run_id()
        return self._run_id

    def _make_run_id(self) -> str:
        """Generate a unique run ID from scenario + seed + timestamp."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        scenario = self.config.scenario.value
        seed = self.config.seed
        return f"{scenario}_{seed}_{ts}"
