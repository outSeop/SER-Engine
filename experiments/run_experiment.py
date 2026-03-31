"""
experiments/run_experiment.py

Run a single simulation experiment (one scenario, one seed).

Usage:
    python -m experiments.run_experiment --scenario PURE_RATIONAL --seed 42
    python -m experiments.run_experiment --config config/default.yaml --scenario SELF_REPLICATING --seed 123 --ticks 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def run_single(
    config_path: str = "config/default.yaml",
    scenario: str | None = None,
    seed: int | None = None,
    max_ticks: int | None = None,
    output_dir: str | None = None,
    verbose: bool = True,
) -> str:
    """
    Run a single simulation experiment.
    Returns the run directory path.
    """
    from sim.config import load_config
    from sim.engine import SimulationEngine

    overrides: dict = {}
    if scenario:
        overrides["simulation.scenario"] = scenario
    if seed is not None:
        overrides["simulation.seed"] = seed
    if max_ticks is not None:
        overrides["simulation.max_ticks"] = max_ticks
    if output_dir is not None:
        overrides["logging.output_dir"] = output_dir

    config = load_config(config_path, overrides or None)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Scenario: {config.scenario.value}")
        print(f"Seed:     {config.seed}")
        print(f"Ticks:    {config.max_ticks}")
        print(f"Nodes:    {config.world.num_nodes}")
        print(f"Agents:   {config.agent.initial_count}")
        print(f"{'='*60}")

    engine = SimulationEngine(config)
    run_dir = engine.run(verbose=verbose)

    if verbose:
        print(f"\nCompleted. Results: {run_dir}")
        _print_summary(run_dir)

    return run_dir


def _print_summary(run_dir: str) -> None:
    """Print a brief summary of the run results."""
    import json
    summary_path = Path(run_dir) / "summary.json"
    if not summary_path.exists():
        return

    with open(summary_path) as f:
        summary = json.load(f)

    print(f"\nEvents fired: {summary.get('events', [])}")

    final = summary.get("final_metrics", {})
    if final:
        print(f"Final tick:         {final.get('tick', '?')}")
        print(f"Alive agents:       {final.get('num_alive_agents', '?')}")
        print(f"Active lineages:    {final.get('num_active_lineages', '?')}")
        print(f"Equilibrium score:  {final.get('equilibrium_score', '?'):.3f}")
        print(f"Replication rate:   {final.get('replication_rate', '?'):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single SER simulation experiment")
    parser.add_argument("--config", default="config/default.yaml", help="Config YAML path")
    parser.add_argument("--scenario", default=None, help="Override scenario type")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--ticks", type=int, default=None, help="Override max_ticks")
    parser.add_argument("--output", default=None, help="Override output directory")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    run_single(
        config_path=args.config,
        scenario=args.scenario,
        seed=args.seed,
        max_ticks=args.ticks,
        output_dir=args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
