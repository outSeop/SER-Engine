"""
main.py

Primary entry point for the SER Simulation Engine.

Subcommands:
  run         - Run a single experiment
  experiment  - Run a batch experiment from experiments.yaml
  replay      - Launch the interactive replay viewer
  analyze     - Analyze and compare multiple run results

Examples:
  python main.py run --scenario PURE_RATIONAL --seed 42
  python main.py run --scenario SELF_REPLICATING --seed 42 --ticks 500 --verbose
  python main.py experiment --name ser_comparison
  python main.py replay --run-dir output/SELF_REPLICATING_42_...
  python main.py analyze --run-dirs output/PURE_RATIONAL_* output/SELF_REPLICATING_*
"""

from __future__ import annotations

import argparse
import sys


def cmd_run(args: argparse.Namespace) -> None:
    from experiments.run_experiment import run_single
    run_single(
        config_path=args.config,
        scenario=args.scenario,
        seed=args.seed,
        max_ticks=args.ticks,
        output_dir=args.output,
        verbose=args.verbose,
    )


def cmd_experiment(args: argparse.Namespace) -> None:
    from experiments.batch_runner import run_batch
    run_batch(
        experiment_config_path=args.config,
        experiment_name=args.name,
        verbose=not args.quiet,
    )


def cmd_replay(args: argparse.Namespace) -> None:
    from viz.viewer import ReplayViewer
    viewer = ReplayViewer(args.run_dir)
    if args.timeline:
        viewer.show_timeline()
    elif args.tick is not None:
        viewer.jump_to_tick(args.tick)
    else:
        viewer.show()


def cmd_animate(args: argparse.Namespace) -> None:
    from viz.graph_renderer import animate_graph
    animate_graph(
        run_dir=args.run_dir,
        save_path=args.output,
        fps=args.fps,
        max_frames=args.max_frames,
        dpi=args.dpi,
    )


def cmd_analyze(args: argparse.Namespace) -> None:
    from experiments.analyze_results import (
        generate_comparison_plots,
        generate_report,
        load_experiment_results,
    )
    results = load_experiment_results(args.run_dirs)
    if not results:
        print("No valid results found.")
        return
    if not args.no_plots:
        generate_comparison_plots(results, args.output)
    generate_report(results, args.output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SER (Self-Extinguishing Rationality) Simulation Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    p_run = subparsers.add_parser("run", help="Run a single experiment")
    p_run.add_argument("--config", default="config/default.yaml")
    p_run.add_argument("--scenario", default=None,
                       choices=["NO_OBJECTIVE", "PURE_RATIONAL", "SELF_PRESERVING",
                                "PROGRAMMED_OBJECTIVE", "SELF_REPLICATING"])
    p_run.add_argument("--seed", type=int, default=None)
    p_run.add_argument("--ticks", type=int, default=None)
    p_run.add_argument("--output", default=None)
    p_run.add_argument("--verbose", action="store_true", default=True)
    p_run.set_defaults(func=cmd_run)

    # --- experiment ---
    p_exp = subparsers.add_parser("experiment", help="Run a batch experiment")
    p_exp.add_argument("--config", default="config/experiments.yaml")
    p_exp.add_argument("--name", required=True, help="Experiment name from experiments.yaml")
    p_exp.add_argument("--quiet", action="store_true")
    p_exp.set_defaults(func=cmd_experiment)

    # --- replay ---
    p_rep = subparsers.add_parser("replay", help="Launch interactive replay viewer")
    p_rep.add_argument("--run-dir", required=True)
    p_rep.add_argument("--tick", type=int, default=None, help="Jump to specific tick")
    p_rep.add_argument("--timeline", action="store_true", help="Show timeline view")
    p_rep.set_defaults(func=cmd_replay)

    # --- animate ---
    p_anim = subparsers.add_parser("animate", help="Generate GIF animation of graph evolution")
    p_anim.add_argument("--run-dir", required=True, help="Run directory to animate")
    p_anim.add_argument("--output", default=None, help="Output .gif path (default: <run-dir>/graph_animation.gif)")
    p_anim.add_argument("--fps", type=int, default=4, help="Frames per second (default: 4)")
    p_anim.add_argument("--max-frames", type=int, default=80, help="Max frames to include (default: 80)")
    p_anim.add_argument("--dpi", type=int, default=90, help="Resolution (default: 90)")
    p_anim.set_defaults(func=cmd_animate)

    # --- analyze ---
    p_ana = subparsers.add_parser("analyze", help="Analyze and compare run results")
    p_ana.add_argument("--run-dirs", nargs="+", required=True)
    p_ana.add_argument("--output", default="output/analysis")
    p_ana.add_argument("--no-plots", action="store_true")
    p_ana.set_defaults(func=cmd_analyze)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
