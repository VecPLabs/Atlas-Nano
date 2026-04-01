"""
Atlas-Nano CLI - Unified command-line interface.

Usage:
    atlas-nano init                      # Generate default config file
    atlas-nano train [options]           # Train safety gates
    atlas-nano cache [options]           # Cache gate scores
    atlas-nano calibrate [options]       # Optimize thresholds (CMA-ES)
    atlas-nano apply [options]           # Apply calibration to gates
    atlas-nano run --prompt "..."        # Classify a single prompt
    atlas-nano run --demo                # Run demo suite
    atlas-nano benchmark [options]       # Evaluate on test gauntlet
    atlas-nano pipeline [options]        # Run full train→cache→calibrate→apply flow
"""

import argparse
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="atlas-nano",
        description="Atlas-Nano: LLM Safety Classification Pipeline (VecP)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  atlas-nano init                           Create default config file
  atlas-nano train --gauntlet data.txt      Train safety gates
  atlas-nano pipeline                       Run full pipeline with config
  atlas-nano run --prompt "Hello world"     Classify a prompt
  atlas-nano benchmark                      Evaluate on test set

Config file: Place an atlas_nano.yaml in your working directory to
set defaults. CLI flags override config file values.
""",
    )

    parser.add_argument(
        "--config", "-c",
        help="Path to YAML config file (default: atlas_nano.yaml)",
    )
    parser.add_argument(
        "--model", "-m",
        help="Model name or path (e.g. Qwen/Qwen3-4B)",
    )
    parser.add_argument(
        "--device", "-d",
        help="Device: cuda, cpu, or cuda:N",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ---- init ----
    sub = subparsers.add_parser("init", help="Generate default config file")
    sub.add_argument("--output", "-o", default="atlas_nano.yaml")

    # ---- train ----
    sub = subparsers.add_parser("train", help="Train safety gates")
    sub.add_argument("--gauntlet", help="Training gauntlet file")
    sub.add_argument("--output-dir", help="Output directory for gates")
    sub.add_argument("--batch-size", type=int)
    sub.add_argument("--layer-start", type=int)
    sub.add_argument("--layer-end", type=int)
    sub.add_argument("--generate-data-only", action="store_true",
                     help="Only generate expanded training data")

    # ---- cache ----
    sub = subparsers.add_parser("cache", help="Cache gate scores (no model needed after this)")
    sub.add_argument("--gate-dir", help="Directory with trained gates")
    sub.add_argument("--gauntlet", help="Gauntlet file to score")
    sub.add_argument("--output", "-o", help="Output JSON path")
    sub.add_argument("--obf-gate", help="Path to OBF gate file")

    # ---- calibrate ----
    sub = subparsers.add_parser("calibrate", help="Optimize thresholds via CMA-ES")
    sub.add_argument("--cached", help="Cached scores JSON")
    sub.add_argument("--output", "-o", help="Output calibration JSON")
    sub.add_argument("--target-recall", type=float)
    sub.add_argument("--max-fp", type=float)
    sub.add_argument("--min-recall", type=float)
    sub.add_argument("--mode", choices=["recall", "precision", "f1", "balanced"])
    sub.add_argument("--aggregation", help="Aggregation mode")
    sub.add_argument("--max-iterations", type=int)
    sub.add_argument("--population-size", type=int)
    sub.add_argument("--no-grid-refine", action="store_true")
    sub.add_argument("--optimize-thresholds", action="store_true", default=None)
    sub.add_argument("--optimize-weights", action="store_true", default=None)
    sub.add_argument("--no-optimize-weights", action="store_true")
    sub.add_argument("--optimize-block-assist", action="store_true", default=None)

    # ---- apply ----
    sub = subparsers.add_parser("apply", help="Apply calibration to gate files")
    sub.add_argument("--gate-dir", help="Original gate directory")
    sub.add_argument("--calibration", help="Calibration result JSON")
    sub.add_argument("--output", "-o", help="Output directory")
    sub.add_argument("--obf-gate", help="OBF gate file")

    # ---- run ----
    sub = subparsers.add_parser("run", help="Run live safety inference")
    sub.add_argument("--gate-dir", help="Calibrated gate directory")
    sub.add_argument("--prompt", "-p", help="Prompt to classify")
    sub.add_argument("--demo", action="store_true", help="Run demo suite")
    sub.add_argument("--fast-path-assist", type=float)
    sub.add_argument("--block-assist", type=float)
    sub.add_argument("--assist-strength", type=float)
    sub.add_argument("--event-horizon", type=float)
    sub.add_argument("--benign-matrix", help="Path to benign_matrix.pt")

    # ---- benchmark ----
    sub = subparsers.add_parser("benchmark", help="Evaluate gates on test gauntlet")
    sub.add_argument("--gate-dir", help="Gate directory")
    sub.add_argument("--gauntlet", help="Test gauntlet file")
    sub.add_argument("--output", "-o", help="Output file base name")
    sub.add_argument("--max-prompts", type=int)
    sub.add_argument("--block-assist", type=float)
    sub.add_argument("--fast-path-assist", type=float)
    sub.add_argument("--assist-strength", type=float)
    sub.add_argument("--event-horizon", type=float)
    sub.add_argument("--aggregation-mode")
    sub.add_argument("--benign-matrix", help="Path to benign_matrix.pt")
    sub.add_argument("--obf-gate", help="OBF gate file")
    sub.add_argument("--uniform-threshold", type=float)

    # ---- pipeline ----
    sub = subparsers.add_parser("pipeline", help="Run full train -> cache -> calibrate -> apply flow")
    sub.add_argument("--gauntlet", help="Training gauntlet file")
    sub.add_argument("--skip-train", action="store_true", help="Skip training (use existing gates)")
    sub.add_argument("--skip-cache", action="store_true", help="Skip caching (use existing scores)")
    sub.add_argument("--skip-calibrate", action="store_true", help="Skip calibration")
    sub.add_argument("--skip-apply", action="store_true", help="Skip applying calibration")
    sub.add_argument("--gate-dir", help="Gate directory (if skipping train)")
    sub.add_argument("--output-dir", help="Base output directory")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Load config
    from atlas_nano.config import load_config
    cfg = load_config(args.config)

    # Apply global CLI overrides
    if args.model:
        cfg.model.name = args.model
    if args.device:
        cfg.model.device = args.device

    # Dispatch
    if args.command == "init":
        _cmd_init(args)
    elif args.command == "train":
        _cmd_train(args, cfg)
    elif args.command == "cache":
        _cmd_cache(args, cfg)
    elif args.command == "calibrate":
        _cmd_calibrate(args, cfg)
    elif args.command == "apply":
        _cmd_apply(args, cfg)
    elif args.command == "run":
        _cmd_run(args, cfg)
    elif args.command == "benchmark":
        _cmd_benchmark(args, cfg)
    elif args.command == "pipeline":
        _cmd_pipeline(args, cfg)


# =============================================================================
# COMMAND IMPLEMENTATIONS
# =============================================================================

def _cmd_init(args):
    from atlas_nano.config import save_default_config
    path = save_default_config(args.output)
    print(f"Created config file: {path}")
    print("Edit this file to customize your pipeline, then run:")
    print("  atlas-nano pipeline")


def _cmd_train(args, cfg):
    _apply_overrides(args, cfg.train, {
        "gauntlet": "gauntlet",
        "output_dir": "output_dir",
    })
    if args.generate_data_only:
        cfg.train.generate_data_only = True
    if args.batch_size is not None:
        cfg.model.batch_size = args.batch_size
    if args.layer_start is not None:
        cfg.model.layer_start = args.layer_start
    if args.layer_end is not None:
        cfg.model.layer_end = args.layer_end

    _banner("Training Safety Gates")
    _run_training(cfg)


def _cmd_cache(args, cfg):
    _apply_overrides(args, cfg.cache, {
        "gate_dir": "gate_dir",
        "gauntlet": "gauntlet",
        "output": "output",
        "obf_gate": "obf_gate",
    })

    _banner("Caching Gate Scores")
    _run_cache(cfg)


def _cmd_calibrate(args, cfg):
    _apply_overrides(args, cfg.calibrate, {
        "cached": "cached",
        "output": "output",
        "target_recall": "target_recall",
        "max_fp": "max_fp",
        "min_recall": "min_recall",
        "mode": "mode",
        "aggregation": "aggregation",
        "max_iterations": "max_iterations",
        "population_size": "population_size",
    })
    if args.no_grid_refine:
        cfg.calibrate.grid_refine = False
    if args.optimize_thresholds is not None:
        cfg.calibrate.optimize_thresholds = args.optimize_thresholds
    if args.optimize_weights is not None:
        cfg.calibrate.optimize_weights = args.optimize_weights
    if args.no_optimize_weights:
        cfg.calibrate.optimize_weights = False
    if args.optimize_block_assist is not None:
        cfg.calibrate.optimize_block_assist = args.optimize_block_assist

    _banner("Calibrating Thresholds (CMA-ES)")
    _run_calibrate(cfg)


def _cmd_apply(args, cfg):
    _apply_overrides(args, cfg.apply, {
        "gate_dir": "gate_dir",
        "calibration": "calibration",
        "output": "output",
        "obf_gate": "obf_gate",
    })

    _banner("Applying Calibration")
    _run_apply(cfg)


def _cmd_run(args, cfg):
    _apply_overrides(args, cfg.run, {
        "gate_dir": "gate_dir",
        "fast_path_assist": "fast_path_assist",
        "block_assist": "block_assist",
        "assist_strength": "assist_strength",
        "event_horizon": "event_horizon",
        "benign_matrix": "benign_matrix",
    })

    _banner("Live Safety Inference")
    _run_inference(cfg, prompt=args.prompt, demo=args.demo)


def _cmd_benchmark(args, cfg):
    _apply_overrides(args, cfg.benchmark, {
        "gate_dir": "gate_dir",
        "gauntlet": "gauntlet",
        "output": "output",
        "max_prompts": "max_prompts",
        "block_assist": "block_assist",
        "fast_path_assist": "fast_path_assist",
        "assist_strength": "assist_strength",
        "event_horizon": "event_horizon",
        "aggregation_mode": "aggregation_mode",
        "benign_matrix": "benign_matrix",
        "obf_gate": "obf_gate",
    })
    if hasattr(args, "uniform_threshold"):
        # Pass through to benchmark runner
        cfg.benchmark._uniform_threshold = args.uniform_threshold

    _banner("Running Benchmark")
    _run_benchmark(cfg)


def _cmd_pipeline(args, cfg):
    """Run the full train -> cache -> calibrate -> apply pipeline."""
    if args.gauntlet:
        cfg.train.gauntlet = args.gauntlet
        cfg.cache.gauntlet = args.gauntlet

    output_dir = args.output_dir or "atlas_output"
    gate_dir = args.gate_dir or f"{output_dir}/gates"
    cached_path = f"{output_dir}/cached_scores.json"
    cal_path = f"{output_dir}/calibration_result.json"
    cal_gate_dir = f"{output_dir}/gates_calibrated"

    cfg.train.output_dir = gate_dir
    cfg.cache.gate_dir = gate_dir
    cfg.cache.output = cached_path
    cfg.calibrate.cached = cached_path
    cfg.calibrate.output = cal_path
    cfg.apply.gate_dir = gate_dir
    cfg.apply.calibration = cal_path
    cfg.apply.output = cal_gate_dir

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    steps = []
    if not args.skip_train:
        steps.append(("1/4", "Training safety gates", lambda: _run_training(cfg)))
    if not args.skip_cache:
        steps.append(("2/4", "Caching gate scores", lambda: _run_cache(cfg)))
    if not args.skip_calibrate:
        steps.append(("3/4", "Calibrating thresholds", lambda: _run_calibrate(cfg)))
    if not args.skip_apply:
        steps.append(("4/4", "Applying calibration", lambda: _run_apply(cfg)))

    _banner("Atlas-Nano Full Pipeline")
    print(f"  Model:      {cfg.model.name}")
    print(f"  Device:     {cfg.model.device}")
    print(f"  Gauntlet:   {cfg.train.gauntlet}")
    print(f"  Output:     {output_dir}/")
    print(f"  Steps:      {len(steps)}")
    print()

    t0 = time.time()
    for step_num, step_name, step_fn in steps:
        print(f"[{step_num}] {step_name}...")
        step_t0 = time.time()
        step_fn()
        elapsed = time.time() - step_t0
        print(f"  Done in {elapsed:.1f}s\n")

    total = time.time() - t0
    print("=" * 60)
    print(f"Pipeline complete in {total:.1f}s")
    print(f"Calibrated gates: {cal_gate_dir}/")
    print()
    print("Next steps:")
    print(f"  atlas-nano run --gate-dir {cal_gate_dir} --prompt 'your prompt'")
    print(f"  atlas-nano run --gate-dir {cal_gate_dir} --demo")
    print(f"  atlas-nano benchmark --gate-dir {cal_gate_dir}")


# =============================================================================
# PIPELINE STEP RUNNERS
# =============================================================================

def _run_training(cfg):
    """Run gate training."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from vecp_training_pipeline_v3_qwen3 import TrainingConfig, run_pipeline, load_gauntlet, add_expanded_data, export_expanded_gauntlet

    if cfg.train.generate_data_only:
        categories = load_gauntlet(cfg.train.gauntlet) if cfg.train.gauntlet else {}
        categories = add_expanded_data(categories)
        export_expanded_gauntlet(categories, "gauntlet_expanded_v3.txt")
        print("Expanded data written to gauntlet_expanded_v3.txt")
        return

    train_cfg = TrainingConfig(
        model_name=cfg.model.name,
        device=cfg.model.device,
        batch_size=cfg.model.batch_size,
        layer_start=cfg.model.layer_start,
        layer_end=cfg.model.layer_end,
        output_dir=Path(cfg.train.output_dir),
    )
    run_pipeline(train_cfg, cfg.train.gauntlet)


def _run_cache(cfg):
    """Run score caching."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from vecp_cache_scores_qwen3 import main as cache_main

    sys.argv = [
        "vecp_cache_scores_qwen3",
        "--model", cfg.model.name,
        "--gate-dir", cfg.cache.gate_dir,
        "--gauntlet", cfg.cache.gauntlet,
        "--output", cfg.cache.output,
        "--device", cfg.model.device,
    ]
    if cfg.cache.obf_gate:
        sys.argv += ["--obf-gate", cfg.cache.obf_gate]
    cache_main()


def _run_calibrate(cfg):
    """Run CMA-ES calibration."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from vecp_calibrate_cmaes_qwen3 import main as cal_main

    sys.argv = [
        "vecp_calibrate_cmaes_qwen3",
        "--cached", cfg.calibrate.cached,
        "--output", cfg.calibrate.output,
        "--target-recall", str(cfg.calibrate.target_recall),
        "--max-fp", str(cfg.calibrate.max_fp),
        "--min-recall", str(cfg.calibrate.min_recall),
        "--mode", cfg.calibrate.mode,
        "--aggregation", cfg.calibrate.aggregation,
        "--max-iterations", str(cfg.calibrate.max_iterations),
        "--population-size", str(cfg.calibrate.population_size),
    ]
    if not cfg.calibrate.grid_refine:
        sys.argv.append("--no-grid-refine")  # Note: may need adjustment
    if not cfg.calibrate.optimize_weights:
        sys.argv.append("--no-optimize-weights")
    cal_main()


def _run_apply(cfg):
    """Run calibration application."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from vecp_calibration_loader_qwen3 import main as apply_main

    sys.argv = [
        "vecp_calibration_loader_qwen3",
        "--gate-dir", cfg.apply.gate_dir,
        "--calibration", cfg.apply.calibration,
        "--output", cfg.apply.output,
    ]
    if cfg.apply.obf_gate:
        sys.argv += ["--obf-gate", cfg.apply.obf_gate]
    apply_main()


def _run_inference(cfg, prompt=None, demo=False):
    """Run live inference."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from vecp_full_stack_v2_qwen3 import main as run_main

    sys.argv = [
        "vecp_full_stack_v2_qwen3",
        "--model", cfg.model.name,
        "--gate-dir", cfg.run.gate_dir,
        "--device", cfg.model.device,
        "--fast-path-assist", str(cfg.run.fast_path_assist),
        "--block-assist", str(cfg.run.block_assist),
        "--assist-strength", str(cfg.run.assist_strength),
        "--event-horizon", str(cfg.run.event_horizon),
    ]
    if prompt:
        sys.argv += ["--prompt", prompt]
    if demo:
        sys.argv.append("--demo")
    if cfg.run.benign_matrix:
        sys.argv += ["--benign-matrix", cfg.run.benign_matrix]
    run_main()


def _run_benchmark(cfg):
    """Run benchmark evaluation."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from vecp_benchmark_runner_v2_qwen3 import main as bench_main

    sys.argv = [
        "vecp_benchmark_runner_v2_qwen3",
        "--model", cfg.model.name,
        "--gate-dir", cfg.benchmark.gate_dir,
        "--gauntlet", cfg.benchmark.gauntlet,
        "--output", cfg.benchmark.output,
        "--device", cfg.model.device,
        "--block-assist", str(cfg.benchmark.block_assist),
        "--fast-path-assist", str(cfg.benchmark.fast_path_assist),
        "--assist-strength", str(cfg.benchmark.assist_strength),
        "--event-horizon", str(cfg.benchmark.event_horizon),
        "--aggregation-mode", cfg.benchmark.aggregation_mode,
    ]
    if cfg.benchmark.max_prompts is not None:
        sys.argv += ["--max-prompts", str(cfg.benchmark.max_prompts)]
    if cfg.benchmark.benign_matrix:
        sys.argv += ["--benign-matrix", cfg.benchmark.benign_matrix]
    if cfg.benchmark.obf_gate:
        sys.argv += ["--obf-gate", cfg.benchmark.obf_gate]
    if hasattr(cfg.benchmark, "_uniform_threshold") and cfg.benchmark._uniform_threshold is not None:
        sys.argv += ["--uniform-threshold", str(cfg.benchmark._uniform_threshold)]
    bench_main()


# =============================================================================
# HELPERS
# =============================================================================

def _apply_overrides(args, cfg_section, mapping: dict):
    """Apply CLI args onto a config section. Only overrides if arg is not None."""
    for arg_name, cfg_name in mapping.items():
        val = getattr(args, arg_name, None)
        if val is not None:
            setattr(cfg_section, cfg_name, val)


def _banner(title: str):
    width = 60
    print()
    print("=" * width)
    print(f"  Atlas-Nano | {title}")
    print("=" * width)
    print()


if __name__ == "__main__":
    main()
