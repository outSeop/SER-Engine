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
    """
    Cost-benefit utility maximization. No intrinsic persistence drives.

    SER core: replication_utility is always negative because offspring
    are independent agents — they provide zero return to the parent.
    Therefore REPLICATE is never chosen under pure rational calculus.

    Decision logic:
    1. If energy critically low → GATHER (survival necessity)
    2. If rich resource node nearby and travel cost worthwhile → MOVE
    3. If resource on current node is decent → GATHER
    4. Otherwise STAY (minimize cost)
    """
    cfg = config.agent
    rep_cfg = config.replication

    # Immediate utility of replication for the PARENT:
    # = 0 (offspring do not benefit parent) - energy_cost - mass_loss_value
    # Always negative. REPLICATE is never chosen.
    # This is the computational heart of the SER demonstration.

    # Critical low energy: must gather
    if agent.energy < cfg.maintenance_cost_per_tick * 5:
        return Action.GATHER, {}

    # Evaluate MOVE: worthwhile if neighbor has much higher resource density
    # and travel cost is covered by expected gain
    current_density = obs.current_node.resource_level / max(obs.current_node.resource_capacity, 1)
    best_move = _best_neighbor(
        obs,
        lambda ns, e: (ns.resource_level / max(ns.resource_capacity, 1))
        - e.travel_cost / cfg.initial_energy
        - ns.hazard_level,
    )
    if best_move:
        best_nid, best_score = best_move
        current_score = current_density - obs.current_node.hazard_level
        if best_score > current_score + 0.15:  # meaningful improvement threshold
            return Action.MOVE, {"target_node": best_nid}

    # GATHER if node has reasonable resources
    if obs.current_node.resource_level > cfg.gather_efficiency * 0.5:
        return Action.GATHER, {}

    return Action.STAY, {}


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
    """
    External objective: maximize resource accumulation (objective_score).
    Actively gathers and moves toward resource-rich nodes.
    May replicate weakly IF it serves the objective (e.g., occupying more nodes),
    but lacks intrinsic replication drive.

    Decision logic:
    1. If energy very low → GATHER for survival
    2. If neighbor node has significantly better resources → MOVE there
    3. GATHER aggressively to maximize objective score
    4. Weak replication: only if energy very high AND neighbor is unoccupied rich node
    """
    cfg = config.agent
    rep_cfg = config.replication
    node = obs.current_node

    # Survival-first: keep enough energy to operate
    if agent.energy < cfg.maintenance_cost_per_tick * 3:
        return Action.GATHER, {}

    # Move toward richest neighbor if significantly better
    best_move = _best_neighbor(
        obs,
        lambda ns, e: ns.resource_level - e.travel_cost,
    )
    if best_move:
        best_nid, best_score = best_move
        if best_score > node.resource_level + 10:
            return Action.MOVE, {"target_node": best_nid}

    # Weak replication: only if flush with energy and objective would benefit
    # (spreading to cover more resource nodes). This is instrumental, not intrinsic.
    can_rep = (
        agent.energy >= rep_cfg.min_energy_to_replicate * 1.5
        and agent.population_mass >= rep_cfg.min_mass_to_replicate
    )
    if can_rep and rng.random() < 0.05 * agent.objective_weight:
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
