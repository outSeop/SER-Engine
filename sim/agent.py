"""
sim/agent.py

LineageAgent dataclass and action-selection logic per strategy type.

Key design: Each agent is a lineage-type super-agent representing a population cluster.
The 5 strategy types implement fundamentally different decision rules — especially around
replication — to test the SER hypothesis.

SER Core Mechanism:
- PURE_RATIONAL: replication_utility = benefit_to_parent - cost. Since offspring are
  separate agents, benefit_to_parent ≈ 0. Therefore replication_utility < 0 always.
  → Pure rational agents never replicate. → Static equilibrium.
- SELF_REPLICATING: replication_weight adds intrinsic drive that offsets the cost.
  → Agents replicate despite the cost. → Persistent dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from sim.config import Action, SimulationConfig, StrategyType
from sim.graph_world import EdgeState, NodeState


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    current_node: NodeState
    neighbor_nodes: list[tuple[int, NodeState, EdgeState]]  # (node_id, state, edge)
    self_state: "LineageAgent"
    recent_energy_delta: float  # energy change last tick


# ---------------------------------------------------------------------------
# LineageAgent
# ---------------------------------------------------------------------------

@dataclass
class LineageAgent:
    agent_id: str
    lineage_id: str
    node_id: int
    alive: bool
    population_mass: float
    energy: float
    age: int
    strategy_type: StrategyType
    mutation_rate: float
    generation: int
    parent_lineage_id: Optional[str]

    # Internal drive weights
    survival_weight: float
    replication_weight: float
    objective_weight: float

    # Replication cooldown tracking
    last_replicated_tick: int = field(default=-999)

    # For PROGRAMMED_OBJECTIVE: accumulated objective score
    objective_score: float = field(default=0.0)

    # Previous tick energy (for delta calculation)
    prev_energy: float = field(default=0.0)

    def snapshot(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "lineage_id": self.lineage_id,
            "node_id": self.node_id,
            "alive": self.alive,
            "population_mass": self.population_mass,
            "energy": self.energy,
            "age": self.age,
            "strategy_type": self.strategy_type.value,
            "mutation_rate": self.mutation_rate,
            "generation": self.generation,
            "parent_lineage_id": self.parent_lineage_id,
            "survival_weight": self.survival_weight,
            "replication_weight": self.replication_weight,
            "objective_weight": self.objective_weight,
            "objective_score": self.objective_score,
        }


# ---------------------------------------------------------------------------
# Action selection: public entry point
# ---------------------------------------------------------------------------

def select_action(
    agent: LineageAgent,
    obs: Observation,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> tuple[Action, dict]:
    """
    Returns (Action, action_params) for the agent this tick.
    action_params may contain keys like 'target_node' for MOVE,
    or 'replicate' flag for REPLICATE.
    """
    dispatch = {
        StrategyType.NO_OBJECTIVE: _no_objective_action,
        StrategyType.PURE_RATIONAL: _pure_rational_action,
        StrategyType.SELF_PRESERVING: _self_preserving_action,
        StrategyType.PROGRAMMED_OBJECTIVE: _programmed_objective_action,
        StrategyType.SELF_REPLICATING: _self_replicating_action,
    }
    return dispatch[agent.strategy_type](agent, obs, config, rng)


# ---------------------------------------------------------------------------
# Helper: find best neighbor
# ---------------------------------------------------------------------------

def _best_neighbor(
    obs: Observation,
    score_fn,
) -> tuple[int, float] | None:
    """Return (node_id, score) of the best neighbor, or None if no neighbors."""
    best_id, best_score = None, float("-inf")
    for nid, nstate, edge in obs.neighbor_nodes:
        s = score_fn(nstate, edge)
        if s > best_score:
            best_id, best_score = nid, s
    return (best_id, best_score) if best_id is not None else None


def _sample_by_utility(
    options: list[tuple[Action, dict, float]],
    rng: np.random.Generator,
    temperature: float,
) -> tuple[Action, dict]:
    """
    Sample one action using a softmax over utility values.
    """
    temp = max(temperature, 1e-6)
    utilities = np.array([u for _, _, u in options], dtype=float)
    shifted = utilities - np.max(utilities)
    probs = np.exp(shifted / temp)
    probs = probs / np.sum(probs)
    idx = int(rng.choice(len(options), p=probs))
    action, params, _ = options[idx]
    return action, params


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _no_objective_action(
    agent: LineageAgent,
    obs: Observation,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> tuple[Action, dict]:
    """
    No terminal goal. Minimal activity.
    5% chance to GATHER if energy is below half; otherwise STAY.
    Never moves, never replicates.
    """
    if agent.energy < config.agent.initial_energy * 0.5:
        if rng.random() < 0.05:
            return Action.GATHER, {}
    return Action.STAY, {}


def _pure_rational_action(
    agent: LineageAgent,
    obs: Observation,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> tuple[Action, dict]:
    """Cost-benefit utility maximization without hard-coded no-replication."""
    cfg = config.agent
    rep_cfg = config.replication

    # Survival emergency guardrail.
    if agent.energy < cfg.maintenance_cost_per_tick * 3:
        return Action.GATHER, {}

    current_density = obs.current_node.resource_level / max(obs.current_node.resource_capacity, 1)
    current_score = current_density - obs.current_node.hazard_level
    best_move = _best_neighbor(
        obs, lambda ns, e: (ns.resource_level / max(ns.resource_capacity, 1)) - e.travel_cost / cfg.initial_energy - ns.hazard_level
    )

    # Replication expected utility includes possible positive externality.
    if agent.energy >= rep_cfg.min_energy_to_replicate and agent.population_mass >= rep_cfg.min_mass_to_replicate:
        expected_externality = (
            cfg.rational_replication_externality_weight
            * current_density
            * (1.0 - obs.current_node.hazard_level)
        )
        competition_cost = cfg.rational_replication_competition_penalty * (
            len(obs.current_node.agents_present) / max(obs.current_node.carrying_capacity, 1)
        )
        replication_utility = expected_externality - (
            rep_cfg.energy_cost / max(cfg.initial_energy, 1.0)
            + rep_cfg.mass_split_ratio
            + competition_cost
        )
    else:
        replication_utility = -1.0

    gather_utility = current_density * 0.9 - obs.current_node.hazard_level * 0.2
    stay_utility = -0.05 - obs.current_node.hazard_level * 0.3

    options: list[tuple[Action, dict, float]] = [
        (Action.GATHER, {}, gather_utility),
        (Action.STAY, {}, stay_utility),
        (Action.REPLICATE, {}, replication_utility),
    ]
    if best_move:
        best_nid, best_score = best_move
        move_utility = best_score - current_score
        options.append((Action.MOVE, {"target_node": best_nid}, move_utility))

    return _sample_by_utility(options, rng, cfg.rational_action_temperature)


def _self_preserving_action(
    agent: LineageAgent,
    obs: Observation,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> tuple[Action, dict]:
    """
    Survival is paramount. Avoid risk, conserve energy, seek safe nodes.
    Never replicates.

    Decision logic:
    1. If current node is dangerous → flee to safest neighbor
    2. If energy low → GATHER
    3. If energy comfortable and node resources good → GATHER to stockpile
    4. Otherwise STAY (minimize exposure)
    """
    cfg = config.agent
    node = obs.current_node

    # Flee from hazardous node
    if node.hazard_level > 0.2:
        safest = _best_neighbor(obs, lambda ns, e: -ns.hazard_level - e.conflict_risk)
        if safest:
            target_id, _ = safest
            return Action.MOVE, {"target_node": target_id}

    # Gather if energy is below comfortable threshold
    if agent.energy < cfg.initial_energy * 0.7:
        if node.resource_level > 0:
            return Action.GATHER, {}

    # Opportunistic gathering on safe, rich nodes
    if node.resource_level > cfg.gather_efficiency and node.hazard_level < 0.1:
        return Action.GATHER, {}

    return Action.STAY, {}


def _programmed_objective_action(
    agent: LineageAgent,
    obs: Observation,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> tuple[Action, dict]:
    """External objective optimization with explicit objective-contribution scoring."""
    cfg = config.agent
    rep_cfg = config.replication
    node = obs.current_node

    # Survival-first: keep enough energy to operate
    if agent.energy < cfg.maintenance_cost_per_tick * 3:
        return Action.GATHER, {}

    # Move toward richest neighbor if significantly better
    best_move = _best_neighbor(
        obs,
        lambda ns, e: cfg.objective_resource_weight * ns.resource_level
        + cfg.objective_coverage_weight * max(ns.carrying_capacity - len(ns.agents_present), 0)
        - e.travel_cost,
    )
    if best_move:
        best_nid, best_score = best_move
        current_objective = (
            cfg.objective_resource_weight * node.resource_level
            + cfg.objective_coverage_weight * max(node.carrying_capacity - len(node.agents_present), 0)
        )
        if best_score > current_objective + 3.0:
            return Action.MOVE, {"target_node": best_nid}

    # Replication if expected objective gain is positive and above threshold.
    can_rep = (
        agent.energy >= rep_cfg.min_energy_to_replicate
        and agent.population_mass >= rep_cfg.min_mass_to_replicate
    )
    crowding = len(node.agents_present) / max(node.carrying_capacity, 1)
    replication_gain = (
        cfg.objective_resource_weight * node.resource_regen_rate
        + cfg.objective_coverage_weight * max(1.0 - crowding, 0.0) * 10.0
        + cfg.objective_resilience_weight * (1.0 - node.hazard_level) * 5.0
    )
    replication_cost = rep_cfg.energy_cost / max(cfg.initial_energy, 1.0) + rep_cfg.mass_split_ratio
    if can_rep and (replication_gain - replication_cost) >= cfg.objective_replication_gain_threshold:
        return Action.REPLICATE, {}

    # Default: gather to maximize objective score
    if node.resource_level > 0:
        return Action.GATHER, {}

    return Action.STAY, {}


def _self_replicating_action(
    agent: LineageAgent,
    obs: Observation,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> tuple[Action, dict]:
    """
    Intrinsic replication drive. replication_weight offsets the cost.

    Decision logic:
    1. If energy critically low → GATHER (can't replicate if dead)
    2. REPLICATE if conditions met and replication_weight is high
    3. Move to richer node if current node can't support replication
    4. GATHER to build up energy for next replication
    """
    cfg = config.agent
    rep_cfg = config.replication
    node = obs.current_node

    # Critical survival: must gather
    if agent.energy < cfg.maintenance_cost_per_tick * 3:
        if node.resource_level > 0:
            return Action.GATHER, {}
        # Move to find resources
        best_move = _best_neighbor(obs, lambda ns, e: ns.resource_level - e.travel_cost)
        if best_move:
            return Action.MOVE, {"target_node": best_move[0]}
        return Action.STAY, {}

    # Replicate if drive + conditions align
    energy_ready = agent.energy >= rep_cfg.min_energy_to_replicate
    mass_ready = agent.population_mass >= rep_cfg.min_mass_to_replicate
    # Intrinsic drive: replication_weight makes this attractive even at cost
    rep_probability = agent.replication_weight * (1.0 if energy_ready and mass_ready else 0.0)
    if rep_probability > 0 and rng.random() < rep_probability:
        return Action.REPLICATE, {}

    # Move to resource-rich node to prepare for replication
    if node.resource_level < cfg.gather_efficiency:
        best_move = _best_neighbor(
            obs,
            lambda ns, e: ns.resource_level - e.travel_cost * 0.5,
        )
        if best_move and best_move[1] > node.resource_level:
            return Action.MOVE, {"target_node": best_move[0]}

    # Gather to fuel replication
    if node.resource_level > 0:
        return Action.GATHER, {}

    return Action.STAY, {}
