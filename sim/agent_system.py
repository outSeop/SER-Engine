"""
sim/agent_system.py

Manages the full collection of LineageAgents: creation, placement, action
resolution, cost application, and death processing.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import numpy as np

from sim.agent import LineageAgent, Observation, select_action
from sim.config import Action, AgentConfig, SimulationConfig, StrategyType
from sim.graph_world import GraphWorld

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Initial weights per strategy
# ---------------------------------------------------------------------------

_STRATEGY_WEIGHTS: dict[StrategyType, dict[str, float]] = {
    StrategyType.NO_OBJECTIVE: {
        "survival_weight": 0.0,
        "replication_weight": 0.0,
        "objective_weight": 0.0,
        "mutation_rate": 0.0,
    },
    StrategyType.PURE_RATIONAL: {
        "survival_weight": 0.5,
        "replication_weight": 0.0,  # SER: no intrinsic replication drive
        "objective_weight": 0.0,
        "mutation_rate": 0.0,
    },
    StrategyType.SELF_PRESERVING: {
        "survival_weight": 1.0,
        "replication_weight": 0.0,
        "objective_weight": 0.0,
        "mutation_rate": 0.0,
    },
    StrategyType.PROGRAMMED_OBJECTIVE: {
        "survival_weight": 0.3,
        "replication_weight": 0.0,
        "objective_weight": 1.0,
        "mutation_rate": 0.0,
    },
    StrategyType.SELF_REPLICATING: {
        "survival_weight": 0.3,
        "replication_weight": 0.7,  # Intrinsic drive: offsets replication cost
        "objective_weight": 0.0,
        "mutation_rate": 0.1,
    },
}


class AgentSystem:
    """
    Owns all agents. Provides lifecycle management and action resolution.
    """

    def __init__(
        self,
        config: SimulationConfig,
        world: GraphWorld,
        rng: np.random.Generator,
    ) -> None:
        self.config = config
        self.world = world
        self.rng = rng
        self.agents: dict[str, LineageAgent] = {}
        self._total_created: int = 0

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize_agents(self) -> None:
        """Place initial agents on random nodes with strategy-appropriate weights."""
        cfg = self.config
        weights = _STRATEGY_WEIGHTS[cfg.scenario]
        node_ids = list(self.world.nodes.keys())

        for _ in range(cfg.agent.initial_count):
            agent_id = self._new_id()
            lineage_id = str(uuid.uuid4())[:8]
            node_id = int(self.rng.choice(node_ids))

            agent = LineageAgent(
                agent_id=agent_id,
                lineage_id=lineage_id,
                node_id=node_id,
                alive=True,
                population_mass=cfg.agent.initial_population_mass,
                energy=cfg.agent.initial_energy,
                age=0,
                strategy_type=cfg.scenario,
                mutation_rate=weights["mutation_rate"],
                generation=0,
                parent_lineage_id=None,
                survival_weight=weights["survival_weight"],
                replication_weight=weights["replication_weight"],
                objective_weight=weights["objective_weight"],
                prev_energy=cfg.agent.initial_energy,
            )
            self.agents[agent_id] = agent
            self.world.nodes[node_id].agents_present.append(agent_id)

    # ------------------------------------------------------------------
    # Observation building
    # ------------------------------------------------------------------

    def build_observation(self, agent: LineageAgent) -> Observation:
        """Construct what this agent can observe from the current world state."""
        current_node = self.world.get_node(agent.node_id)
        neighbor_ids = self.world.get_neighbors(agent.node_id)
        neighbor_nodes = [
            (nid, self.world.get_node(nid), self.world.get_edge(agent.node_id, nid))
            for nid in neighbor_ids
        ]
        recent_delta = agent.energy - agent.prev_energy
        return Observation(
            current_node=current_node,
            neighbor_nodes=neighbor_nodes,
            self_state=agent,
            recent_energy_delta=recent_delta,
        )

    # ------------------------------------------------------------------
    # Action selection (Step 3)
    # ------------------------------------------------------------------

    def collect_actions(
        self, current_tick: int
    ) -> tuple[list[tuple[LineageAgent, Action, dict]], list[tuple[LineageAgent, dict]]]:
        """
        Collect actions from all alive agents.
        Returns:
          - regular_actions: list of (agent, action, params) for STAY/GATHER/MOVE
          - replication_requests: list of (agent, params) for REPLICATE
        """
        regular_actions: list[tuple[LineageAgent, Action, dict]] = []
        replication_requests: list[tuple[LineageAgent, dict]] = []

        for agent in self.get_alive_agents():
            obs = self.build_observation(agent)
            action, params = select_action(agent, obs, self.config, self.rng)

            if action == Action.REPLICATE:
                replication_requests.append((agent, params))
            else:
                regular_actions.append((agent, action, params))

        return regular_actions, replication_requests

    # ------------------------------------------------------------------
    # Action resolution (Step 4)
    # ------------------------------------------------------------------

    def resolve_regular_actions(
        self, actions: list[tuple[LineageAgent, Action, dict]]
    ) -> None:
        """
        Resolve STAY, GATHER, MOVE actions.
        REPLICATE requests are handled separately in Step 7.
        """
        # Group GATHER actions by node to handle resource sharing
        gather_by_node: dict[int, list[LineageAgent]] = {}
        move_actions: list[tuple[LineageAgent, int]] = []

        for agent, action, params in actions:
            if action == Action.GATHER:
                gather_by_node.setdefault(agent.node_id, []).append(agent)
            elif action == Action.MOVE:
                target = params.get("target_node")
                if target is not None and target in self.world.nodes:
                    move_actions.append((agent, target))
            # STAY: no-op

        # Resolve GATHER: mass-proportional resource distribution
        for node_id, gatherers in gather_by_node.items():
            total_mass = sum(a.population_mass for a in gatherers)
            if total_mass <= 0:
                continue
            # Available resource this tick = min(requested, available)
            requested = self.config.agent.gather_efficiency * len(gatherers)
            available = self.world.consume_resource(node_id, requested)
            for agent in gatherers:
                share = (agent.population_mass / total_mass) * available
                agent.energy += share
                # PROGRAMMED_OBJECTIVE: accumulate objective score from resources
                if agent.strategy_type == StrategyType.PROGRAMMED_OBJECTIVE:
                    agent.objective_score += share * agent.objective_weight

        # Resolve MOVE
        for agent, target_node in move_actions:
            edge = self.world.get_edge(agent.node_id, target_node)
            cost = edge.travel_cost
            if agent.energy >= cost:
                # Update presence tracking
                if agent.agent_id in self.world.nodes[agent.node_id].agents_present:
                    self.world.nodes[agent.node_id].agents_present.remove(agent.agent_id)
                agent.node_id = target_node
                agent.energy -= cost
                self.world.nodes[target_node].agents_present.append(agent.agent_id)

    # ------------------------------------------------------------------
    # Cost application (Step 5)
    # ------------------------------------------------------------------

    def apply_maintenance_costs(self) -> None:
        """Deduct per-tick maintenance cost from all alive agents."""
        cost = self.config.agent.maintenance_cost_per_tick
        for agent in self.get_alive_agents():
            agent.prev_energy = agent.energy
            agent.energy -= cost
            agent.age += 1

    # ------------------------------------------------------------------
    # Death processing (Step 6)
    # ------------------------------------------------------------------

    def process_deaths(self) -> list[str]:
        """
        Mark agents as dead if below energy or mass thresholds.
        Returns list of agent_ids that died this tick.
        """
        cfg = self.config.agent
        dead_ids: list[str] = []
        for agent in list(self.agents.values()):
            if not agent.alive:
                continue
            if agent.energy <= cfg.death_energy_threshold or \
               agent.population_mass <= cfg.death_mass_threshold:
                agent.alive = False
                dead_ids.append(agent.agent_id)
                # Remove from node presence
                node = self.world.nodes.get(agent.node_id)
                if node and agent.agent_id in node.agents_present:
                    node.agents_present.remove(agent.agent_id)
        return dead_ids

    # ------------------------------------------------------------------
    # Agent management
    # ------------------------------------------------------------------

    def add_agent(self, agent: LineageAgent) -> None:
        """Add a newly replicated agent to the system."""
        self.agents[agent.agent_id] = agent
        self.world.nodes[agent.node_id].agents_present.append(agent.agent_id)
        self._total_created += 1

    def get_alive_agents(self) -> list[LineageAgent]:
        return [a for a in self.agents.values() if a.alive]

    def get_agents_at_node(self, node_id: int) -> list[LineageAgent]:
        return [
            self.agents[aid]
            for aid in self.world.nodes[node_id].agents_present
            if aid in self.agents and self.agents[aid].alive
        ]

    def _new_id(self) -> str:
        self._total_created += 1
        return f"agent_{self._total_created:06d}"

    def new_agent_id(self) -> str:
        return self._new_id()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def snapshot(self) -> list[dict]:
        return [a.snapshot() for a in self.agents.values() if a.alive]
