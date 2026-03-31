"""
experiments/batch_runner.py

Run multiple experiments from experiments.yaml.
Supports scenarios × seeds cross-product and parameter sweeps.

Usage:
    python -m experiments.batch_runner --config config/experiments.yaml --experiment ser_comparison
    python -m experiments.batch_runner --config config/experiments.yaml --experiment replication_cost_sweep
    python -m experiments.batch_runner --list  # list available experiments
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

import yaml


def load_experiment_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_batch(
    experiment_config_path: str,
    experiment_name: str,
    parallel: bool = False,
    verbose: bool = True,
) -> list[str]:
    """
    Run all combinations for a named experiment.
    Returns list of completed run_dir paths.

    parallel=True uses multiprocessing (TODO: future enhancement).
    """
    from experiments.run_experiment import run_single

    raw = load_experiment_config(experiment_config_path)
    experiments = raw.get("experiments", {})

    if experiment_name not in experiments:
        available = list(experiments.keys())
        raise ValueError(f"Experiment '{experiment_name}' not found. Available: {available}")

    exp = experiments[experiment_name]
    base_config = exp.get("base_config", "config/default.yaml")
    seeds = exp.get("seeds", [42])
    scenarios = exp.get("scenarios", ["PURE_RATIONAL"])
    max_ticks = exp.get("max_ticks", None)
    sweep: dict[str, list] = exp.get("parameter_sweep", {})

    # Build all combinations
    combos: list[dict[str, Any]] = []

    if sweep:
        # Parameter sweep: one param at a time × seeds × scenarios
        for param_key, param_values in sweep.items():
            for param_val in param_values:
                for scenario in scenarios:
                    for seed in seeds:
                        combos.append({
                            "scenario": scenario,
                            "seed": seed,
                            "max_ticks": max_ticks,
                            "overrides": {param_key: param_val},
                            "label": f"{scenario}_seed{seed}_{param_key.split('.')[-1]}={param_val}",
                        })
    else:
        for scenario in scenarios:
            for seed in seeds:
                combos.append({
                    "scenario": scenario,
                    "seed": seed,
                    "max_ticks": max_ticks,
                    "overrides": {},
                    "label": f"{scenario}_seed{seed}",
                })

    total = len(combos)
    if verbose:
        print(f"\nExperiment: {experiment_name}")
        print(f"Description: {exp.get('description', '')}")
        print(f"Total runs: {total}")
        print("=" * 60)

    run_dirs: list[str] = []
    for i, combo in enumerate(combos, 1):
        if verbose:
            print(f"\n[{i}/{total}] {combo['label']}")

        overrides = dict(combo.get("overrides", {}))
        if combo["scenario"]:
            overrides["simulation.scenario"] = combo["scenario"]
        if combo["seed"] is not None:
            overrides["simulation.seed"] = combo["seed"]
        if combo["max_ticks"] is not None:
            overrides["simulation.max_ticks"] = combo["max_ticks"]

        from sim.config import load_config
        from sim.engine import SimulationEngine

        config = load_config(base_config, overrides)
        engine = SimulationEngine(config)
        run_dir = engine.run(verbose=verbose)
        run_dirs.append(run_dir)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Batch complete. {len(run_dirs)} runs finished.")
        print(f"Output directories:")
        for d in run_dirs:
            print(f"  {d}")

    return run_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch SER experiments")
    parser.add_argument(
        "--config", default="config/experiments.yaml",
        help="Experiments YAML path",
    )
    parser.add_argument("--experiment", default=None, help="Experiment name to run")
    parser.add_argument("--list", action="store_true", help="List available experiments")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    if args.list:
        raw = load_experiment_config(args.config)
        print("Available experiments:")
        for name, exp in raw.get("experiments", {}).items():
            print(f"  {name}: {exp.get('description', '')}")
        return

    if not args.experiment:
        print("Error: --experiment is required (or use --list to see available experiments)")
        return

    run_batch(args.config, args.experiment, verbose=not args.quiet)


if __name__ == "__main__":
    main()
