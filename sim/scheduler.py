"""
sim/scheduler.py

Orchestrates the strict 8-step tick update order.
The ordering is critical for correctness:
- Step 6 (death) must precede Step 7 (replication) so dead agents cannot replicate.
- Replication requests are collected in Step 3/4 but executed only in Step 7.
"""

from __future__ import annotations

import numpy as np

from sim.agent_system import AgentSystem
from sim.config import SimulationConfig
from sim.event_detector import EventDetector
from sim.evolution import process_replications
from sim.graph_world import GraphWorld
from sim.logger import SimLogger
from sim.metrics import MetricsCollector


class Scheduler:
    """
    Manages one simulation tick at a time, strictly following:
    1. Environment update
    2. Observation (implicit in step 3)
    3. Action selection
    4. Action resolution
    5. Cost/reward update
    6. Death/extinction
    7. Replication/mutation
    8. Metrics/event update
    """

    def __init__(
        self,
        config: SimulationConfig,
        world: GraphWorld,
        agent_system: AgentSystem,
        metrics_collector: MetricsCollector,
        event_detector: EventDetector,
        logger: SimLogger,
        rng: np.random.Generator,
    ) -> None:
        self.config = config
        self.world = world
        self.agent_system = agent_system
        self.metrics = metrics_collector
        self.event_detector = event_detector
        self.logger = logger
        self.rng = rng
        self.current_tick: int = 0

    def step(self) -> bool:
        """
        Execute one full tick.
        Returns False if the simulation should terminate (all dead or max_ticks reached).
        """
        # --- Step 1: Environment update ---
        self.world.regenerate_resources()

        # --- Step 3: Action selection + Step 4: partial resolution ---
        regular_actions, replication_requests = self.agent_system.collect_actions(
            self.current_tick
        )

        # --- Step 4: Resolve STAY/GATHER/MOVE ---
        self.agent_system.resolve_regular_actions(regular_actions)

        # --- Step 5: Cost/reward update ---
        self.agent_system.apply_maintenance_costs()

        # --- Step 6: Death/extinction ---
        dead_ids = self.agent_system.process_deaths()

        # --- Step 7: Replication/mutation (AFTER death to prevent zombie replication) ---
        new_agents = process_replications(
            self.agent_system,
            replication_requests,
            self.config,
            self.current_tick,
            self.rng,
        )

        # --- Step 8: Metrics/event update ---
        prev_metrics = self.metrics.latest()
        m = self.metrics.compute(
            tick=self.current_tick,
            agent_system=self.agent_system,
            world=self.world,
            replications_this_tick=len(new_agents),
            deaths_this_tick=len(dead_ids),
            new_lineages_this_tick=len(new_agents),
        )
        new_events = self.event_detector.check(
            self.current_tick, m, prev_metrics, self.agent_system
        )

        # Log this tick
        save_snap = (self.current_tick % self.config.logging.save_every_n_ticks == 0)
        self.logger.log_tick(
            tick=self.current_tick,
            metrics=m,
            agents_snapshot=self.agent_system.snapshot() if save_snap else [],
            world_snapshot=self.world.snapshot() if save_snap else {"nodes": []},
        )
        if new_events:
            self.logger.log_events(new_events)

        self.current_tick += 1

        # Termination conditions
        if m.num_alive_agents == 0:
            return False
        if self.current_tick >= self.config.max_ticks:
            return False
        return True
