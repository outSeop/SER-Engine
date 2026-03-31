"""
sim/config.py

Configuration loading and typed dataclass definitions for the SER simulation engine.
All simulation parameters flow through SimulationConfig.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class StrategyType(Enum):
    NO_OBJECTIVE = "NO_OBJECTIVE"
    PURE_RATIONAL = "PURE_RATIONAL"
    SELF_PRESERVING = "SELF_PRESERVING"
    PROGRAMMED_OBJECTIVE = "PROGRAMMED_OBJECTIVE"
    SELF_REPLICATING = "SELF_REPLICATING"


class Action(Enum):
    STAY = "STAY"
    GATHER = "GATHER"
    MOVE = "MOVE"
    REPLICATE = "REPLICATE"


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WorldConfig:
    num_nodes: int = 50
    k_nearest: int = 4
    rewire_probability: float = 0.1
    resource_capacity_min: float = 50.0
    resource_capacity_max: float = 200.0
    resource_regen_rate_min: float = 0.5
    resource_regen_rate_max: float = 3.0
    hazard_level_min: float = 0.0
    hazard_level_max: float = 0.3
    carrying_capacity_min: int = 5
    carrying_capacity_max: int = 20
    oversight_level_min: float = 0.0
    oversight_level_max: float = 0.5
    infrastructure_level_min: float = 0.0
    infrastructure_level_max: float = 1.0
    travel_cost_min: float = 1.0
    travel_cost_max: float = 5.0
    communication_bandwidth_min: float = 0.5
    communication_bandwidth_max: float = 1.0
    trade_efficiency_min: float = 0.5
    trade_efficiency_max: float = 1.0
    conflict_risk_min: float = 0.0
    conflict_risk_max: float = 0.2


@dataclass
class AgentConfig:
    initial_count: int = 10
    initial_energy: float = 100.0
    initial_population_mass: float = 1000.0
    maintenance_cost_per_tick: float = 1.0
    gather_efficiency: float = 5.0
    move_energy_cost: float = 3.0
    death_energy_threshold: float = 0.0
    death_mass_threshold: float = 1.0


@dataclass
class ReplicationConfig:
    energy_cost: float = 40.0
    mass_split_ratio: float = 0.5
    min_energy_to_replicate: float = 60.0
    min_mass_to_replicate: float = 200.0
    cooldown_ticks: int = 5


@dataclass
class MutationConfig:
    magnitude: float = 0.05
    targets: list[str] = field(default_factory=lambda: [
        "survival_weight",
        "replication_weight",
        "objective_weight",
        "mutation_rate",
    ])


@dataclass
class MetricsConfig:
    equilibrium_window: int = 50
    equilibrium_threshold: float = 0.01
    persistence_check_interval: int = 100


@dataclass
class LoggingConfig:
    output_dir: str = "output"
    save_every_n_ticks: int = 10
    format: str = "parquet"


@dataclass
class SimulationConfig:
    seed: int = 42
    max_ticks: int = 1000
    scenario: StrategyType = StrategyType.PURE_RATIONAL
    world: WorldConfig = field(default_factory=WorldConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    replication: ReplicationConfig = field(default_factory=ReplicationConfig)
    mutation: MutationConfig = field(default_factory=MutationConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def make_rng(self) -> np.random.Generator:
        """Create and return a seeded numpy RNG. Call once at engine startup."""
        return np.random.default_rng(self.seed)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base dict."""
    result = dict(base)
    for k, v in overrides.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _apply_dotted_overrides(data: dict, overrides: dict[str, Any]) -> dict:
    """Apply dotted-key overrides like {'replication.energy_cost': 60.0}."""
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        target = data
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return data


def load_config(
    path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> SimulationConfig:
    """
    Load a YAML config file and return a SimulationConfig.

    overrides: flat dict with dotted keys, e.g. {'replication.energy_cost': 60.0}
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    if overrides:
        raw = _apply_dotted_overrides(raw, overrides)

    sim_raw = raw.get("simulation", {})
    world_raw = raw.get("world", {})
    agent_raw = raw.get("agent", {})
    rep_raw = raw.get("replication", {})
    mut_raw = raw.get("mutation", {})
    metrics_raw = raw.get("metrics", {})
    log_raw = raw.get("logging", {})

    scenario_str = sim_raw.get("scenario", "PURE_RATIONAL")
    scenario = StrategyType(scenario_str)

    return SimulationConfig(
        seed=sim_raw.get("seed", 42),
        max_ticks=sim_raw.get("max_ticks", 1000),
        scenario=scenario,
        world=WorldConfig(**world_raw) if world_raw else WorldConfig(),
        agent=AgentConfig(**agent_raw) if agent_raw else AgentConfig(),
        replication=ReplicationConfig(**rep_raw) if rep_raw else ReplicationConfig(),
        mutation=MutationConfig(**mut_raw) if mut_raw else MutationConfig(),
        metrics=MetricsConfig(**metrics_raw) if metrics_raw else MetricsConfig(),
        logging=LoggingConfig(**log_raw) if log_raw else LoggingConfig(),
    )
