"""tests/test_metrics.py"""

import numpy as np
import pytest

from sim.metrics import MetricsCollector, TickMetrics
from sim.config import MetricsConfig


def make_metrics(tick: int, mass: float, alive: int, lineages: int, nodes: int) -> TickMetrics:
    return TickMetrics(
        tick=tick,
        total_population_mass=mass,
        num_alive_agents=alive,
        num_active_lineages=lineages,
        mean_energy=50.0,
        replication_rate=0.0,
        extinction_count=0,
        lineage_branching_events=0,
        active_node_count=nodes,
        equilibrium_score=0.0,
        inertness_score=1.0,
        structural_drift_score=0.0,
        population_persistence_time=tick,
        lineage_diversity=lineages / max(alive, 1),
        dominance_index=1.0 / max(lineages, 1),
        centralization_score=0.5,
    )


class FakeAgentSystem:
    def __init__(self, agents):
        self._agents = agents

    def get_alive_agents(self):
        return self._agents


class FakeAgent:
    def __init__(self, lineage_id, population_mass, energy, node_id):
        self.lineage_id = lineage_id
        self.population_mass = population_mass
        self.energy = energy
        self.node_id = node_id


def test_equilibrium_score_is_zero_when_insufficient_history():
    cfg = MetricsConfig(equilibrium_window=50)
    collector = MetricsCollector(cfg)
    # Only 10 entries, window=50 → not enough history
    for i in range(10):
        collector.history.append(make_metrics(i, 1000, 10, 5, 20))
    assert collector._compute_equilibrium_score(
        inertness_score=1.0,
        structural_drift_score=0.0,
    ) == pytest.approx(0.0)


def test_equilibrium_score_near_one_for_constant_population():
    cfg = MetricsConfig(equilibrium_window=50)
    collector = MetricsCollector(cfg)
    # Perfect constant: no variation at all
    for i in range(60):
        collector.history.append(make_metrics(i, 1000.0, 10, 5, 20))
    score = collector._compute_equilibrium_score(
        inertness_score=1.0,
        structural_drift_score=0.0,
    )
    assert score > 0.95, f"Expected high equilibrium score, got {score}"


def test_equilibrium_score_near_zero_for_fluctuating_population():
    cfg = MetricsConfig(equilibrium_window=50)
    collector = MetricsCollector(cfg)
    rng = np.random.default_rng(99)
    # All three metrics fluctuate wildly
    for i in range(60):
        mass = rng.uniform(100, 2000)
        alive = int(rng.integers(1, 50))
        nodes = int(rng.integers(1, 40))  # also fluctuate active_node_count
        collector.history.append(make_metrics(i, mass, alive, 3, nodes))
    score = collector._compute_equilibrium_score(
        inertness_score=0.5,
        structural_drift_score=0.5,
    )
    assert score < 0.7, f"Expected low equilibrium score, got {score}"


def test_lineage_diversity_calculation():
    cfg = MetricsConfig(equilibrium_window=50)
    collector = MetricsCollector(cfg)
    agents = [
        FakeAgent("lin_a", 100, 50, 0),
        FakeAgent("lin_a", 100, 50, 1),
        FakeAgent("lin_b", 100, 50, 2),
        FakeAgent("lin_c", 100, 50, 3),
    ]
    agent_sys = FakeAgentSystem(agents)

    class FakeWorld:
        nodes = {0: type("N", (), {"agents_present": []})()}

    m = collector.compute(0, agent_sys, FakeWorld(), 0, 0, 0)
    # 3 unique lineages / 4 agents = 0.75
    assert m.lineage_diversity == pytest.approx(0.75)


def test_dominance_index_single_lineage():
    cfg = MetricsConfig(equilibrium_window=50)
    collector = MetricsCollector(cfg)
    agents = [
        FakeAgent("lin_a", 500, 50, 0),
        FakeAgent("lin_a", 500, 50, 1),
    ]
    agent_sys = FakeAgentSystem(agents)

    class FakeWorld:
        nodes = {}

    m = collector.compute(0, agent_sys, FakeWorld(), 0, 0, 0)
    # Single lineage dominates 100%
    assert m.dominance_index == pytest.approx(1.0)


def test_dominance_index_equal_two_lineages():
    cfg = MetricsConfig(equilibrium_window=50)
    collector = MetricsCollector(cfg)
    agents = [
        FakeAgent("lin_a", 500, 50, 0),
        FakeAgent("lin_b", 500, 50, 1),
    ]
    agent_sys = FakeAgentSystem(agents)

    class FakeWorld:
        nodes = {}

    m = collector.compute(0, agent_sys, FakeWorld(), 0, 0, 0)
    assert m.dominance_index == pytest.approx(0.5)


def test_active_node_count():
    cfg = MetricsConfig(equilibrium_window=50)
    collector = MetricsCollector(cfg)
    agents = [
        FakeAgent("lin_a", 100, 50, 0),
        FakeAgent("lin_a", 100, 50, 0),  # same node
        FakeAgent("lin_b", 100, 50, 5),  # different node
    ]
    agent_sys = FakeAgentSystem(agents)

    class FakeWorld:
        nodes = {}

    m = collector.compute(0, agent_sys, FakeWorld(), 0, 0, 0)
    assert m.active_node_count == 2  # nodes 0 and 5


def test_metrics_to_dict_has_all_keys():
    m = make_metrics(5, 1000, 10, 3, 15)
    d = m.to_dict()
    expected_keys = [
        "tick", "total_population_mass", "num_alive_agents", "num_active_lineages",
        "mean_energy", "replication_rate", "extinction_count", "lineage_branching_events",
        "active_node_count", "equilibrium_score", "inertness_score", "structural_drift_score",
        "population_persistence_time",
        "lineage_diversity", "dominance_index", "centralization_score",
    ]
    for k in expected_keys:
        assert k in d, f"Missing key: {k}"
