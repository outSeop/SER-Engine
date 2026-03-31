"""tests/test_events.py"""

import pytest

from sim.config import SimulationConfig, StrategyType, load_config
from sim.event_detector import EventDetector, SimEvent
from sim.metrics import TickMetrics


def make_metrics(
    tick: int = 0,
    equilibrium_score: float = 0.0,
    replication_rate: float = 0.0,
    num_alive: int = 10,
    num_lineages: int = 5,
    extinction_count: int = 0,
    inertness_score: float = 1.0,
    structural_drift_score: float = 0.0,
) -> TickMetrics:
    return TickMetrics(
        tick=tick,
        total_population_mass=1000.0,
        num_alive_agents=num_alive,
        num_active_lineages=num_lineages,
        mean_energy=50.0,
        replication_rate=replication_rate,
        extinction_count=extinction_count,
        lineage_branching_events=0,
        active_node_count=10,
        equilibrium_score=equilibrium_score,
        inertness_score=inertness_score,
        structural_drift_score=structural_drift_score,
        population_persistence_time=tick,
        lineage_diversity=0.5,
        dominance_index=0.2,
        centralization_score=0.3,
    )


class FakeAgentSystem:
    def get_alive_agents(self):
        return []


@pytest.fixture
def pure_rational_config():
    return load_config(
        "config/default.yaml",
        {"simulation.scenario": "PURE_RATIONAL"},
    )


@pytest.fixture
def self_replicating_config():
    return load_config(
        "config/default.yaml",
        {"simulation.scenario": "SELF_REPLICATING"},
    )


def test_first_equilibrium_reached_fires_once(pure_rational_config):
    detector = EventDetector(pure_rational_config)
    agt = FakeAgentSystem()

    # Trigger equilibrium event
    m = make_metrics(tick=100, equilibrium_score=0.97, num_alive=5)
    events = detector.check(100, m, None, agt)
    eq_events = [e for e in events if e.event_type == "first_equilibrium_reached"]
    assert len(eq_events) == 1

    # Should NOT fire again on next tick
    events2 = detector.check(101, m, m, agt)
    eq_events2 = [e for e in events2 if e.event_type == "first_equilibrium_reached"]
    assert len(eq_events2) == 0


def test_first_equilibrium_not_fires_below_threshold(pure_rational_config):
    detector = EventDetector(pure_rational_config)
    agt = FakeAgentSystem()

    m = make_metrics(tick=50, equilibrium_score=0.80)
    events = detector.check(50, m, None, agt)
    eq_events = [e for e in events if e.event_type == "first_equilibrium_reached"]
    assert len(eq_events) == 0


def test_first_sustained_replication_requires_10_consecutive(self_replicating_config):
    detector = EventDetector(self_replicating_config)
    agt = FakeAgentSystem()
    prev = None

    # 9 consecutive ticks with replication → should NOT fire yet
    for tick in range(9):
        m = make_metrics(tick=tick, replication_rate=0.5)
        events = detector.check(tick, m, prev, agt)
        rep_events = [e for e in events if e.event_type == "first_sustained_replication"]
        assert len(rep_events) == 0, f"Should not fire at tick {tick}"
        prev = m

    # Tick 9 (10th consecutive) → should fire
    m = make_metrics(tick=9, replication_rate=0.5)
    events = detector.check(9, m, prev, agt)
    rep_events = [e for e in events if e.event_type == "first_sustained_replication"]
    assert len(rep_events) == 1


def test_sustained_replication_streak_resets_on_zero(self_replicating_config):
    detector = EventDetector(self_replicating_config)
    agt = FakeAgentSystem()
    prev = None

    # 8 consecutive with replication
    for tick in range(8):
        m = make_metrics(tick=tick, replication_rate=0.5)
        detector.check(tick, m, prev, agt)
        prev = m

    # One tick with zero replication → streak resets
    m_zero = make_metrics(tick=8, replication_rate=0.0)
    detector.check(8, m_zero, prev, agt)
    assert detector._replication_streak == 0


def test_pure_rational_collapse_fires(pure_rational_config):
    detector = EventDetector(pure_rational_config)
    agt = FakeAgentSystem()

    # All agents die
    m = make_metrics(tick=200, num_alive=0, equilibrium_score=0.0)
    events = detector.check(200, m, None, agt)
    collapse = [e for e in events if e.event_type == "pure_rational_collapse"]
    assert len(collapse) == 1


def test_total_extinction_fires_regardless_of_scenario(self_replicating_config):
    detector = EventDetector(self_replicating_config)
    agt = FakeAgentSystem()

    m = make_metrics(tick=500, num_alive=0)
    events = detector.check(500, m, None, agt)
    total_ext = [e for e in events if e.event_type == "total_extinction"]
    assert len(total_ext) == 1


def test_lineage_extinction_wave_fires_on_mass_death(pure_rational_config):
    detector = EventDetector(pure_rational_config)
    agt = FakeAgentSystem()

    prev = make_metrics(tick=9, extinction_count=2)
    m = make_metrics(tick=10, extinction_count=7)  # 5 deaths this tick
    events = detector.check(10, m, prev, agt)
    wave = [e for e in events if e.event_type == "first_lineage_extinction_wave"]
    assert len(wave) == 1


def test_self_replicating_divergence_fires(self_replicating_config):
    detector = EventDetector(self_replicating_config)
    agt = FakeAgentSystem()
    initial_count = self_replicating_config.agent.initial_count  # 10

    # 20 lineages = 2x initial_count → should fire
    m = make_metrics(tick=100, num_lineages=initial_count * 2)
    events = detector.check(100, m, None, agt)
    divergence = [e for e in events if e.event_type == "self_replicating_divergence"]
    assert len(divergence) == 1
