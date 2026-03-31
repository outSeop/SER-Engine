"""
sim/evolution.py

Replication cost model and mutation mechanics.

Key design decisions:
- Replication is ALWAYS costly: energy cost + mass split + implicit competition.
- Mutation applies to offspring only, never to the parent.
- mutation_rate itself is mutable (meta-evolution), enabling runaway lineages.
- Selection is implicit: surviving + replicating lineages spread naturally.
"""

from __future__ import annotations

import uuid

import numpy as np

from sim.agent import LineageAgent
from sim.config import MutationConfig, ReplicationConfig, SimulationConfig, StrategyType


# ---------------------------------------------------------------------------
# Replication eligibility check
# ---------------------------------------------------------------------------

def can_replicate(
    agent: LineageAgent,
    config: ReplicationConfig,
    current_tick: int,
) -> bool:
    """Check all preconditions for replication."""
    if not agent.alive:
        return False
    if agent.energy < config.min_energy_to_replicate:
        return False
    if agent.population_mass < config.min_mass_to_replicate:
        return False
    if current_tick - agent.last_replicated_tick < config.cooldown_ticks:
        return False
    return True


# ---------------------------------------------------------------------------
# Replication execution
# ---------------------------------------------------------------------------

def execute_replication(
    parent: LineageAgent,
    config: SimulationConfig,
    rng: np.random.Generator,
    new_agent_id: str,
    current_tick: int,
) -> LineageAgent:
    """
    Execute replication: apply costs to parent, create mutated child.

    Cost model (always expensive):
    1. Energy cost: parent loses energy_cost
    2. Mass split: child gets mass_split_ratio of parent's mass; parent loses it
    3. Implicit competition: child occupies same node, competes for resources

    Returns the new child LineageAgent (not yet added to AgentSystem).
    """
    rep = config.replication

    # --- Apply costs to parent ---
    parent.energy -= rep.energy_cost
    child_mass = parent.population_mass * rep.mass_split_ratio
    parent.population_mass -= child_mass
    parent.last_replicated_tick = current_tick

    # --- Create child ---
    child = LineageAgent(
        agent_id=new_agent_id,
        lineage_id=str(uuid.uuid4())[:8],
        node_id=parent.node_id,  # born on same node as parent
        alive=True,
        population_mass=child_mass,
        energy=parent.energy * 0.1,  # small energy endowment from parent
        age=0,
        strategy_type=parent.strategy_type,
        mutation_rate=parent.mutation_rate,
        generation=parent.generation + 1,
        parent_lineage_id=parent.lineage_id,
        survival_weight=parent.survival_weight,
        replication_weight=parent.replication_weight,
        objective_weight=parent.objective_weight,
        last_replicated_tick=-999,
        objective_score=0.0,
        prev_energy=0.0,
    )

    # Small energy transfer from parent (parent gives up this energy)
    parent.energy -= child.energy

    # --- Apply mutation to child ---
    mutate(child, config.mutation, rng)

    return child


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

def mutate(
    agent: LineageAgent,
    config: MutationConfig,
    rng: np.random.Generator,
) -> None:
    """
    Apply Gaussian perturbation to mutable fields with probability = agent.mutation_rate.

    mutation_rate itself is mutable (meta-evolution):
    lineages can evolve higher or lower mutation rates, creating second-order dynamics.

    Clamping:
    - weight fields: [0.0, 1.0]
    - mutation_rate: [0.01, 0.5]
    """
    for target in config.targets:
        if rng.random() < agent.mutation_rate:
            current_val = getattr(agent, target)
            noise = rng.normal(0.0, config.magnitude)
            new_val = current_val + noise

            # Clamp to valid range
            if target == "mutation_rate":
                new_val = float(np.clip(new_val, 0.01, 0.5))
            else:
                new_val = float(np.clip(new_val, 0.0, 1.0))

            setattr(agent, target, new_val)


# ---------------------------------------------------------------------------
# Batch replication processing
# ---------------------------------------------------------------------------

def process_replications(
    agent_system,
    replication_requests: list[tuple[LineageAgent, dict]],
    config: SimulationConfig,
    current_tick: int,
    rng: np.random.Generator,
) -> list[LineageAgent]:
    """
    Process all replication requests for this tick.
    Validates eligibility (post-death-processing), executes replication,
    adds children to the AgentSystem.

    Returns list of newly created agents.
    """
    new_agents: list[LineageAgent] = []

    for parent, _params in replication_requests:
        # Re-check: parent might have died in Step 6 before Step 7 runs
        if not parent.alive:
            continue
        if not can_replicate(parent, config.replication, current_tick):
            continue

        new_id = agent_system.new_agent_id()
        child = execute_replication(parent, config, rng, new_id, current_tick)
        agent_system.add_agent(child)
        new_agents.append(child)

    return new_agents
