"""
sim/metrics.py

Metric computation for each simulation tick.
Focuses on the metrics that directly test the SER hypothesis:
equilibrium convergence, replication persistence, lineage dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sim.config import MetricsConfig


# ---------------------------------------------------------------------------
# TickMetrics dataclass
# ---------------------------------------------------------------------------

@dataclass
class TickMetrics:
    tick: int

    # Population-level
    total_population_mass: float
    num_alive_agents: int
    num_active_lineages: int
    mean_energy: float

    # Evolution
    replication_rate: float          # replications this tick / alive agents
    extinction_count: int            # cumulative deaths so far
    lineage_branching_events: int    # new lineages born this tick

    # Graph dynamics
    active_node_count: int           # nodes with at least one alive agent

    # SER-specific: equilibrium detection
    equilibrium_score: float         # 0.0 = dynamic, 1.0 = static convergence
    inertness_score: float           # low activity proxy
    structural_drift_score: float    # distributional change proxy

    # Persistence
    population_persistence_time: int  # ticks since first alive agent

    # Diversity
    lineage_diversity: float         # unique lineages / alive agents
    dominance_index: float           # largest lineage mass / total mass
    centralization_score: float      # fraction of mass in top-3 nodes

    def to_dict(self) -> dict:
        return {
            "tick": self.tick,
            "total_population_mass": self.total_population_mass,
            "num_alive_agents": self.num_alive_agents,
            "num_active_lineages": self.num_active_lineages,
            "mean_energy": self.mean_energy,
            "replication_rate": self.replication_rate,
            "extinction_count": self.extinction_count,
            "lineage_branching_events": self.lineage_branching_events,
            "active_node_count": self.active_node_count,
            "equilibrium_score": self.equilibrium_score,
            "inertness_score": self.inertness_score,
            "structural_drift_score": self.structural_drift_score,
            "population_persistence_time": self.population_persistence_time,
            "lineage_diversity": self.lineage_diversity,
            "dominance_index": self.dominance_index,
            "centralization_score": self.centralization_score,
        }


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------

class MetricsCollector:
    """
    Computes and accumulates TickMetrics each tick.
    Provides the history as a pandas DataFrame for analysis and logging.
    """

    def __init__(self, config: MetricsConfig) -> None:
        self.config = config
        self.history: list[TickMetrics] = []
        self._cumulative_extinctions: int = 0
        self._first_alive_tick: int | None = None
        self._prev_node_distribution: dict[int, float] | None = None
        self._prev_lineage_distribution: dict[str, float] | None = None

    def compute(
        self,
        tick: int,
        agent_system,
        world,
        replications_this_tick: int,
        deaths_this_tick: int,
        new_lineages_this_tick: int,
    ) -> TickMetrics:
        """Compute all metrics for this tick and append to history."""
        self._cumulative_extinctions += deaths_this_tick
        alive = agent_system.get_alive_agents()

        if alive and self._first_alive_tick is None:
            self._first_alive_tick = tick

        # --- Population ---
        num_alive = len(alive)
        total_mass = sum(a.population_mass for a in alive)
        mean_energy = float(np.mean([a.energy for a in alive])) if alive else 0.0

        # --- Lineage ---
        lineage_ids = [a.lineage_id for a in alive]
        unique_lineages = len(set(lineage_ids))
        lineage_diversity = unique_lineages / max(num_alive, 1)

        # Dominance: largest single lineage's mass fraction
        if alive and total_mass > 0:
            lineage_masses: dict[str, float] = {}
            for a in alive:
                lineage_masses[a.lineage_id] = lineage_masses.get(a.lineage_id, 0) + a.population_mass
            dominance_index = max(lineage_masses.values()) / total_mass
        else:
            dominance_index = 0.0

        # --- Graph ---
        node_masses: dict[int, float] = {}
        for a in alive:
            node_masses[a.node_id] = node_masses.get(a.node_id, 0) + a.population_mass
        active_node_count = len(node_masses)

        # Centralization: fraction of mass in top-3 nodes
        if node_masses and total_mass > 0:
            sorted_masses = sorted(node_masses.values(), reverse=True)
            top3 = sum(sorted_masses[:3])
            centralization_score = top3 / total_mass
        else:
            centralization_score = 0.0

        # --- Replication rate ---
        replication_rate = replications_this_tick / max(num_alive, 1)

        # --- Inertness & structural drift ---
        inertness_score = self._compute_inertness_score(
            replications_this_tick=replications_this_tick,
            deaths_this_tick=deaths_this_tick,
            num_alive=max(num_alive, 1),
        )
        structural_drift_score = self._compute_structural_drift_score(
            node_masses=node_masses,
            lineage_masses=lineage_masses if alive and total_mass > 0 else {},
        )

        # --- Equilibrium score ---
        equilibrium_score = self._compute_equilibrium_score(
            inertness_score=inertness_score,
            structural_drift_score=structural_drift_score,
        )

        # --- Persistence time ---
        if self._first_alive_tick is not None:
            persistence_time = tick - self._first_alive_tick
        else:
            persistence_time = 0

        m = TickMetrics(
            tick=tick,
            total_population_mass=total_mass,
            num_alive_agents=num_alive,
            num_active_lineages=unique_lineages,
            mean_energy=mean_energy,
            replication_rate=replication_rate,
            extinction_count=self._cumulative_extinctions,
            lineage_branching_events=new_lineages_this_tick,
            active_node_count=active_node_count,
            equilibrium_score=equilibrium_score,
            inertness_score=inertness_score,
            structural_drift_score=structural_drift_score,
            population_persistence_time=persistence_time,
            lineage_diversity=lineage_diversity,
            dominance_index=dominance_index,
            centralization_score=centralization_score,
        )
        self.history.append(m)
        return m

    def _compute_inertness_score(
        self,
        replications_this_tick: int,
        deaths_this_tick: int,
        num_alive: int,
    ) -> float:
        """
        High score means low event activity (replication/extinction) this tick.
        """
        activity = (replications_this_tick + deaths_this_tick) / max(num_alive, 1)
        return float(np.clip(1.0 - activity, 0.0, 1.0))

    def _compute_structural_drift_score(
        self,
        node_masses: dict[int, float],
        lineage_masses: dict[str, float],
    ) -> float:
        """
        Normalized total-variation distance between consecutive distributions.
        """
        current_node_dist = self._normalize_distribution(node_masses)
        current_lineage_dist = self._normalize_distribution(lineage_masses)

        node_drift = self._tv_distance(self._prev_node_distribution, current_node_dist)
        lineage_drift = self._tv_distance(self._prev_lineage_distribution, current_lineage_dist)

        self._prev_node_distribution = current_node_dist
        self._prev_lineage_distribution = current_lineage_dist
        return float(np.clip((node_drift + lineage_drift) * 0.5, 0.0, 1.0))

    def _normalize_distribution(self, masses: dict) -> dict:
        total = float(sum(masses.values()))
        if total <= 0:
            return {}
        return {k: float(v / total) for k, v in masses.items()}

    def _tv_distance(self, prev: dict | None, cur: dict) -> float:
        if prev is None:
            return 0.0
        keys = set(prev.keys()) | set(cur.keys())
        if not keys:
            return 0.0
        return 0.5 * sum(abs(prev.get(k, 0.0) - cur.get(k, 0.0)) for k in keys)

    def _compute_equilibrium_score(
        self,
        inertness_score: float,
        structural_drift_score: float,
    ) -> float:
        """
        Measures proximity to static equilibrium.

        Uses coefficient of variation (CV = std/mean) over a sliding window
        for [total_population_mass, num_alive_agents, active_node_count].

        equilibrium_score = weighted blend of:
          - aggregate stability (windowed CVs)
          - inertness (low local activity)
          - inverse structural drift
        Score ≈ 1.0: system is static (converged).
        Score ≈ 0.0: system is highly dynamic.

        This directly operationalizes "static equilibrium" from the SER hypothesis.
        """
        window = self.config.equilibrium_window
        if len(self.history) < window:
            return 0.0

        recent = self.history[-window:]
        fields = ["total_population_mass", "num_alive_agents", "active_node_count"]
        cvs = []
        for f in fields:
            values = np.array([getattr(m, f) for m in recent], dtype=float)
            mean_val = np.mean(values)
            if mean_val == 0:
                cvs.append(0.0)
            else:
                cvs.append(float(np.std(values) / mean_val))

        aggregate_stability = float(np.clip(1.0 - np.mean(cvs), 0.0, 1.0))
        return float(
            np.clip(
                0.5 * aggregate_stability + 0.25 * inertness_score + 0.25 * (1.0 - structural_drift_score),
                0.0,
                1.0,
            )
        )

    def get_history_df(self) -> pd.DataFrame:
        """Return full metrics history as a pandas DataFrame."""
        return pd.DataFrame([m.to_dict() for m in self.history])

    def latest(self) -> TickMetrics | None:
        return self.history[-1] if self.history else None
