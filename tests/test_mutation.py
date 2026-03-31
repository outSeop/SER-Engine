"""tests/test_mutation.py — covered by test_replication.py mutation tests.
Additional edge case tests here."""

import numpy as np
import pytest

from sim.config import MutationConfig
from sim.evolution import mutate
from sim.agent import LineageAgent
from sim.config import StrategyType


def make_agent(mutation_rate: float = 0.5) -> LineageAgent:
    return LineageAgent(
        agent_id="m001",
        lineage_id="lin",
        node_id=0,
        alive=True,
        population_mass=1000.0,
        energy=100.0,
        age=0,
        strategy_type=StrategyType.SELF_REPLICATING,
        mutation_rate=mutation_rate,
        generation=0,
        parent_lineage_id=None,
        survival_weight=0.5,
        replication_weight=0.5,
        objective_weight=0.0,
    )


def test_all_targets_can_mutate():
    rng = np.random.default_rng(1)
    agent = make_agent(mutation_rate=1.0)
    cfg = MutationConfig(
        magnitude=0.05,
        targets=["survival_weight", "replication_weight", "objective_weight", "mutation_rate"],
    )
    original = {t: getattr(agent, t) for t in cfg.targets}
    # With mutation_rate=1.0 and many runs, all should change eventually
    changed = {t: False for t in cfg.targets}
    for _ in range(50):
        a = make_agent(mutation_rate=1.0)
        for t in cfg.targets:
            setattr(a, t, original.get(t, 0.5))
        mutate(a, cfg, rng)
        for t in cfg.targets:
            if getattr(a, t) != original.get(t, 0.5):
                changed[t] = True
    for t in cfg.targets:
        assert changed[t], f"Target '{t}' never changed across 50 mutations"


def test_weights_stay_in_0_1(  ):
    rng = np.random.default_rng(77)
    agent = make_agent(mutation_rate=1.0)
    cfg = MutationConfig(magnitude=0.5, targets=["survival_weight", "replication_weight"])
    for _ in range(200):
        mutate(agent, cfg, rng)
    assert 0.0 <= agent.survival_weight <= 1.0
    assert 0.0 <= agent.replication_weight <= 1.0


def test_mutation_rate_stays_in_bounds():
    rng = np.random.default_rng(55)
    agent = make_agent(mutation_rate=1.0)
    cfg = MutationConfig(magnitude=1.0, targets=["mutation_rate"])
    for _ in range(200):
        mutate(agent, cfg, rng)
    assert 0.01 <= agent.mutation_rate <= 0.5
