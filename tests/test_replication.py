"""tests/test_replication.py"""

import numpy as np
import pytest

from sim.agent import LineageAgent
from sim.config import ReplicationConfig, SimulationConfig, StrategyType
from sim.evolution import can_replicate, execute_replication, mutate


def make_agent(
    energy: float = 200.0,
    mass: float = 1000.0,
    last_replicated: int = -999,
    strategy: StrategyType = StrategyType.SELF_REPLICATING,
    mutation_rate: float = 0.0,
) -> LineageAgent:
    return LineageAgent(
        agent_id="test_001",
        lineage_id="lin_abc",
        node_id=0,
        alive=True,
        population_mass=mass,
        energy=energy,
        age=10,
        strategy_type=strategy,
        mutation_rate=mutation_rate,
        generation=1,
        parent_lineage_id=None,
        survival_weight=0.3,
        replication_weight=0.7,
        objective_weight=0.0,
        last_replicated_tick=last_replicated,
    )


@pytest.fixture
def rep_config():
    return ReplicationConfig(
        energy_cost=40.0,
        mass_split_ratio=0.5,
        min_energy_to_replicate=60.0,
        min_mass_to_replicate=200.0,
        cooldown_ticks=5,
    )


@pytest.fixture
def sim_config():
    from sim.config import load_config
    return load_config("config/default.yaml", {"simulation.scenario": "SELF_REPLICATING"})


def test_can_replicate_passes_when_all_conditions_met(rep_config):
    agent = make_agent(energy=200.0, mass=1000.0, last_replicated=-999)
    assert can_replicate(agent, rep_config, current_tick=10)


def test_can_replicate_fails_if_energy_too_low(rep_config):
    agent = make_agent(energy=50.0, mass=1000.0)  # below min_energy_to_replicate=60
    assert not can_replicate(agent, rep_config, current_tick=10)


def test_can_replicate_fails_if_mass_too_low(rep_config):
    agent = make_agent(energy=200.0, mass=100.0)  # below min_mass_to_replicate=200
    assert not can_replicate(agent, rep_config, current_tick=10)


def test_can_replicate_fails_if_cooldown_not_elapsed(rep_config):
    agent = make_agent(energy=200.0, mass=1000.0, last_replicated=8)
    # cooldown=5, current_tick=10, so 10-8=2 < 5
    assert not can_replicate(agent, rep_config, current_tick=10)


def test_can_replicate_passes_after_cooldown(rep_config):
    agent = make_agent(energy=200.0, mass=1000.0, last_replicated=4)
    # cooldown=5, current_tick=10, so 10-4=6 >= 5
    assert can_replicate(agent, rep_config, current_tick=10)


def test_can_replicate_fails_if_dead(rep_config):
    agent = make_agent(energy=200.0, mass=1000.0)
    agent.alive = False
    assert not can_replicate(agent, rep_config, current_tick=10)


def test_execute_replication_deducts_energy_cost(sim_config):
    rng = np.random.default_rng(42)
    agent = make_agent(energy=200.0, mass=1000.0)
    energy_before = agent.energy

    child = execute_replication(agent, sim_config, rng, "child_001", current_tick=10)

    rep = sim_config.replication
    # Parent lost energy_cost + child's initial endowment
    assert agent.energy < energy_before - rep.energy_cost


def test_execute_replication_splits_mass(sim_config):
    rng = np.random.default_rng(42)
    agent = make_agent(energy=200.0, mass=1000.0)
    original_mass = agent.population_mass
    rep = sim_config.replication

    child = execute_replication(agent, sim_config, rng, "child_001", current_tick=10)

    expected_child_mass = original_mass * rep.mass_split_ratio
    assert child.population_mass == pytest.approx(expected_child_mass)
    assert agent.population_mass == pytest.approx(original_mass - expected_child_mass)


def test_execute_replication_child_has_correct_lineage(sim_config):
    rng = np.random.default_rng(42)
    agent = make_agent(energy=200.0, mass=1000.0)

    child = execute_replication(agent, sim_config, rng, "child_001", current_tick=10)

    assert child.parent_lineage_id == agent.lineage_id
    assert child.lineage_id != agent.lineage_id  # new lineage
    assert child.generation == agent.generation + 1
    assert child.age == 0


def test_execute_replication_child_on_same_node(sim_config):
    rng = np.random.default_rng(42)
    agent = make_agent(energy=200.0, mass=1000.0)
    agent.node_id = 5

    child = execute_replication(agent, sim_config, rng, "child_001", current_tick=10)
    assert child.node_id == 5


def test_execute_replication_updates_cooldown(sim_config):
    rng = np.random.default_rng(42)
    agent = make_agent(energy=200.0, mass=1000.0)
    execute_replication(agent, sim_config, rng, "child_001", current_tick=42)
    assert agent.last_replicated_tick == 42


def test_mutation_changes_values_with_rate_1():
    from sim.config import MutationConfig
    rng = np.random.default_rng(42)
    agent = make_agent(mutation_rate=1.0)
    agent.survival_weight = 0.5
    original = agent.survival_weight

    cfg = MutationConfig(magnitude=0.1, targets=["survival_weight"])
    mutate(agent, cfg, rng)

    # With mutation_rate=1.0, the value MUST change
    assert agent.survival_weight != pytest.approx(original)


def test_mutation_no_change_with_rate_0():
    from sim.config import MutationConfig
    rng = np.random.default_rng(42)
    agent = make_agent(mutation_rate=0.0)
    agent.survival_weight = 0.5

    cfg = MutationConfig(magnitude=0.1, targets=["survival_weight"])
    mutate(agent, cfg, rng)

    assert agent.survival_weight == pytest.approx(0.5)


def test_mutation_clamps_weights_to_valid_range():
    from sim.config import MutationConfig
    rng = np.random.default_rng(0)
    agent = make_agent(mutation_rate=1.0)
    agent.survival_weight = 0.99  # near boundary

    # Force large magnitude to try to push out of [0, 1]
    cfg = MutationConfig(magnitude=10.0, targets=["survival_weight"])
    for _ in range(100):
        mutate(agent, cfg, rng)

    assert 0.0 <= agent.survival_weight <= 1.0


def test_mutation_rate_itself_can_mutate():
    from sim.config import MutationConfig
    rng = np.random.default_rng(42)
    agent = make_agent(mutation_rate=1.0)
    original_rate = agent.mutation_rate

    cfg = MutationConfig(magnitude=0.1, targets=["mutation_rate"])
    mutate(agent, cfg, rng)

    assert agent.mutation_rate != pytest.approx(original_rate)
    assert 0.01 <= agent.mutation_rate <= 0.5
