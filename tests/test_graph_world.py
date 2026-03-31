"""tests/test_graph_world.py"""

import numpy as np
import pytest

from sim.config import WorldConfig
from sim.graph_world import GraphWorld


@pytest.fixture
def world():
    cfg = WorldConfig(num_nodes=20, k_nearest=4, rewire_probability=0.1)
    rng = np.random.default_rng(42)
    return GraphWorld(cfg, rng)


def test_graph_has_correct_node_count(world):
    assert len(world.nodes) == 20


def test_all_nodes_have_required_fields(world):
    for node in world.nodes.values():
        assert node.resource_capacity > 0
        assert 0 <= node.resource_level <= node.resource_capacity
        assert node.resource_regen_rate > 0
        assert 0 <= node.hazard_level <= 1


def test_node_attributes_within_config_range(world):
    cfg = world.config
    for node in world.nodes.values():
        assert cfg.resource_capacity_min <= node.resource_capacity <= cfg.resource_capacity_max
        assert cfg.hazard_level_min <= node.hazard_level <= cfg.hazard_level_max
        assert cfg.oversight_level_min <= node.oversight_level <= cfg.oversight_level_max


def test_resource_regeneration_does_not_exceed_capacity(world):
    # Deplete all nodes first
    for node in world.nodes.values():
        node.resource_level = 0.0
    # Regenerate
    world.regenerate_resources()
    for node in world.nodes.values():
        assert node.resource_level <= node.resource_capacity
        assert node.resource_level >= 0


def test_resource_regeneration_increases_level(world):
    for node in world.nodes.values():
        node.resource_level = 0.0
    world.regenerate_resources()
    for node in world.nodes.values():
        assert node.resource_level > 0 or node.resource_regen_rate == 0


def test_consume_resource_returns_min_of_requested_and_available(world):
    node_id = 0
    world.nodes[node_id].resource_level = 10.0
    # Request more than available
    consumed = world.consume_resource(node_id, 50.0)
    assert consumed == pytest.approx(10.0)
    assert world.nodes[node_id].resource_level == pytest.approx(0.0)


def test_consume_resource_no_negative(world):
    node_id = 0
    world.nodes[node_id].resource_level = 5.0
    world.consume_resource(node_id, 100.0)
    assert world.nodes[node_id].resource_level >= 0.0


def test_get_neighbors_returns_valid_nodes(world):
    for node_id in world.nodes:
        neighbors = world.get_neighbors(node_id)
        for n in neighbors:
            assert n in world.nodes


def test_snapshot_is_serializable(world):
    snap = world.snapshot()
    assert "nodes" in snap
    assert "edges" in snap
    assert len(snap["nodes"]) == 20
