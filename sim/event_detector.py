"""
sim/event_detector.py

Detects and records key narrative events during simulation.
Events are the primary observation points for the SER hypothesis.

One-time events fire at most once per run.
Recurring events can fire multiple times.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sim.config import SimulationConfig, StrategyType
from sim.metrics import TickMetrics


# ---------------------------------------------------------------------------
# SimEvent
# ---------------------------------------------------------------------------

@dataclass
class SimEvent:
    tick: int
    event_type: str
    description: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "tick": self.tick,
            "event_type": self.event_type,
            "description": self.description,
            "data": self.data,
        }


# ---------------------------------------------------------------------------
# EventDetector
# ---------------------------------------------------------------------------

class EventDetector:
    """
    Checks simulation state each tick and emits events.
    Events are ordered by importance for SER hypothesis observation.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.events: list[SimEvent] = []
        # One-time flags
        self._fired: set[str] = set()
        # Counters for streak-based events
        self._replication_streak: int = 0
        self._stagnation_streak: int = 0

    def check(
        self,
        tick: int,
        metrics: TickMetrics,
        prev_metrics: TickMetrics | None,
        agent_system,
    ) -> list[SimEvent]:
        """Check all event conditions and return new events fired this tick."""
        new_events: list[SimEvent] = []

        new_events.extend(self._check_equilibrium(tick, metrics))
        new_events.extend(self._check_replication(tick, metrics))
        new_events.extend(self._check_extinctions(tick, metrics, prev_metrics, agent_system))
        new_events.extend(self._check_scenario_specific(tick, metrics))
        new_events.extend(self._check_lineage_divergence(tick, metrics))

        self.events.extend(new_events)
        return new_events

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_equilibrium(self, tick: int, m: TickMetrics) -> list[SimEvent]:
        events = []
        if "first_equilibrium_reached" not in self._fired:
            if m.equilibrium_score >= 0.95:
                self._fired.add("first_equilibrium_reached")
                events.append(SimEvent(
                    tick=tick,
                    event_type="first_equilibrium_reached",
                    description=f"System reached static equilibrium (score={m.equilibrium_score:.3f})",
                    data={"equilibrium_score": m.equilibrium_score, "num_alive": m.num_alive_agents},
                ))
        return events

    def _check_replication(self, tick: int, m: TickMetrics) -> list[SimEvent]:
        events = []
        if m.replication_rate > 0:
            self._replication_streak += 1
        else:
            self._replication_streak = 0

        if "first_sustained_replication" not in self._fired:
            if self._replication_streak >= 10:
                self._fired.add("first_sustained_replication")
                events.append(SimEvent(
                    tick=tick,
                    event_type="first_sustained_replication",
                    description=f"Sustained replication for 10+ consecutive ticks",
                    data={"streak": self._replication_streak, "replication_rate": m.replication_rate},
                ))
        return events

    def _check_extinctions(
        self,
        tick: int,
        m: TickMetrics,
        prev: TickMetrics | None,
        agent_system,
    ) -> list[SimEvent]:
        events = []

        if "first_lineage_extinction_wave" not in self._fired and prev is not None:
            deaths_this_tick = m.extinction_count - prev.extinction_count
            if deaths_this_tick >= 3:
                self._fired.add("first_lineage_extinction_wave")
                events.append(SimEvent(
                    tick=tick,
                    event_type="first_lineage_extinction_wave",
                    description=f"Extinction wave: {deaths_this_tick} agents died this tick",
                    data={"deaths_this_tick": deaths_this_tick},
                ))

        # Total population collapse
        if "total_extinction" not in self._fired and m.num_alive_agents == 0:
            self._fired.add("total_extinction")
            events.append(SimEvent(
                tick=tick,
                event_type="total_extinction",
                description="All agents have gone extinct",
                data={"final_tick": tick},
            ))

        return events

    def _check_scenario_specific(self, tick: int, m: TickMetrics) -> list[SimEvent]:
        """Scenario-specific hypothesis events."""
        events = []
        scenario = self.config.scenario

        # SER core: pure rational collapse (expected behavior)
        if scenario == StrategyType.PURE_RATIONAL:
            if "pure_rational_collapse" not in self._fired and m.num_alive_agents == 0:
                self._fired.add("pure_rational_collapse")
                events.append(SimEvent(
                    tick=tick,
                    event_type="pure_rational_collapse",
                    description="PURE_RATIONAL population collapsed — SER hypothesis observed",
                    data={"persistence_time": m.population_persistence_time},
                ))

        # Programmed objective stagnation
        if scenario == StrategyType.PROGRAMMED_OBJECTIVE:
            if m.equilibrium_score >= 0.90 and m.replication_rate == 0:
                self._stagnation_streak += 1
            else:
                self._stagnation_streak = 0

            if "programmed_objective_stagnation" not in self._fired and self._stagnation_streak >= 50:
                self._fired.add("programmed_objective_stagnation")
                events.append(SimEvent(
                    tick=tick,
                    event_type="programmed_objective_stagnation",
                    description="PROGRAMMED_OBJECTIVE reached stagnation: active but not replicating",
                    data={"stagnation_streak": self._stagnation_streak},
                ))

        return events

    def _check_lineage_divergence(self, tick: int, m: TickMetrics) -> list[SimEvent]:
        events = []
        initial = self.config.agent.initial_count

        if "self_replicating_divergence" not in self._fired:
            if m.num_active_lineages >= initial * 2:
                self._fired.add("self_replicating_divergence")
                events.append(SimEvent(
                    tick=tick,
                    event_type="self_replicating_divergence",
                    description=f"Lineage count doubled initial population: {m.num_active_lineages} lineages",
                    data={"num_lineages": m.num_active_lineages, "initial_count": initial},
                ))

        return events

    def get_events_df(self) -> list[dict]:
        return [e.to_dict() for e in self.events]
