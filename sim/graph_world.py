"""
sim/graph_world.py

Graph-based environment representation.
Graph is the implementation medium, not the research topic.

Nodes represent: regions, resource clusters, server clusters, social units.
Edges represent: movement paths, resource flows, communication channels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from sim.config import WorldConfig


# ---------------------------------------------------------------------------
# State dataclasses
# ---------------------------------------------------------------------------

@dataclass
class NodeState:
    node_id: int
    resource_capacity: float
    resource_level: float
    resource_regen_rate: float
    hazard_level: float
    carrying_capacity: int
    oversight_level: float
    infrastructure_level: float
    # Agents currently on this node (maintained by AgentSystem)
    agents_present: list[str] = field(default_factory=list)

    def snapshot(self) -> dict:
        return {
            "node_id": self.node_id,
            "resource_capacity": self.resource_capacity,
            "resource_level": self.resource_level,
            "resource_regen_rate": self.resource_regen_rate,
            "hazard_level": self.hazard_level,
            "carrying_capacity": self.carrying_capacity,
            "oversight_level": self.oversight_level,
            "infrastructure_level": self.infrastructure_level,
            "num_agents": len(self.agents_present),
        }


@dataclass
class EdgeState:
    src: int
    dst: int
    travel_cost: float
    communication_bandwidth: float
    trade_efficiency: float
    conflict_risk: float

    def snapshot(self) -> dict:
        return {
            "src": self.src,
            "dst": self.dst,
            "travel_cost": self.travel_cost,
            "communication_bandwidth": self.communication_bandwidth,
            "trade_efficiency": self.trade_efficiency,
            "conflict_risk": self.conflict_risk,
        }


# ---------------------------------------------------------------------------
# GraphWorld
# ---------------------------------------------------------------------------

class GraphWorld:
    """
    Manages the simulation environment: graph topology, node/edge attributes,
    resource regeneration, and resource consumption.
    """

    def __init__(self, config: WorldConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng
        self.graph: nx.Graph = nx.Graph()
        self.nodes: dict[int, NodeState] = {}
        self.edges: dict[tuple[int, int], EdgeState] = {}
        self.build_small_world()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def build_small_world(self) -> None:
        """Build a Watts-Strogatz small-world graph and populate attributes."""
        cfg = self.config
        self.graph = nx.watts_strogatz_graph(
            n=cfg.num_nodes,
            k=cfg.k_nearest,
            p=cfg.rewire_probability,
            seed=int(self.rng.integers(0, 2**31)),
        )

        for node_id in self.graph.nodes():
            node_capacity = float(self.rng.uniform(
                cfg.resource_capacity_min, cfg.resource_capacity_max
            ))
            self.nodes[node_id] = NodeState(
                node_id=node_id,
                resource_capacity=node_capacity,
                resource_level=self.rng.uniform(
                    cfg.resource_capacity_min * 0.5, node_capacity
                ),
                resource_regen_rate=self.rng.uniform(
                    cfg.resource_regen_rate_min, cfg.resource_regen_rate_max
                ),
                hazard_level=self.rng.uniform(
                    cfg.hazard_level_min, cfg.hazard_level_max
                ),
                carrying_capacity=int(self.rng.integers(
                    cfg.carrying_capacity_min, cfg.carrying_capacity_max + 1
                )),
                oversight_level=self.rng.uniform(
                    cfg.oversight_level_min, cfg.oversight_level_max
                ),
                infrastructure_level=self.rng.uniform(
                    cfg.infrastructure_level_min, cfg.infrastructure_level_max
                ),
            )

        for src, dst in self.graph.edges():
            key = (min(src, dst), max(src, dst))
            self.edges[key] = EdgeState(
                src=src,
                dst=dst,
                travel_cost=self.rng.uniform(
                    cfg.travel_cost_min, cfg.travel_cost_max
                ),
                communication_bandwidth=self.rng.uniform(
                    cfg.communication_bandwidth_min, cfg.communication_bandwidth_max
                ),
                trade_efficiency=self.rng.uniform(
                    cfg.trade_efficiency_min, cfg.trade_efficiency_max
                ),
                conflict_risk=self.rng.uniform(
                    cfg.conflict_risk_min, cfg.conflict_risk_max
                ),
            )

    # ------------------------------------------------------------------
    # Environment update
    # ------------------------------------------------------------------

    def regenerate_resources(self) -> None:
        """Tick Step 1: Regenerate node resources, capped at capacity."""
        for node in self.nodes.values():
            node.resource_level = min(
                node.resource_level + node.resource_regen_rate,
                node.resource_capacity,
            )

    # ------------------------------------------------------------------
    # Resource access
    # ------------------------------------------------------------------

    def consume_resource(self, node_id: int, amount: float) -> float:
        """
        Consume up to `amount` from a node. Returns actual amount consumed.
        Resources cannot go negative.
        """
        node = self.nodes[node_id]
        actual = min(amount, node.resource_level)
        node.resource_level -= actual
        return actual

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_node(self, node_id: int) -> NodeState:
        return self.nodes[node_id]

    def get_neighbors(self, node_id: int) -> list[int]:
        return list(self.graph.neighbors(node_id))

    def get_edge(self, n1: int, n2: int) -> EdgeState:
        key = (min(n1, n2), max(n1, n2))
        return self.edges[key]

    def num_nodes(self) -> int:
        return len(self.nodes)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        return {
            "nodes": [n.snapshot() for n in self.nodes.values()],
            "edges": [e.snapshot() for e in self.edges.values()],
        }
