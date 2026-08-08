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
    atlas-nano sign-check [options]      # Distill gates to single energy axis
    atlas-nano gguf [options]            # Embed safety data into GGUF
    atlas-nano pipeline [options]        # Full flow: train→calibrate→GGUF
"""

import argparse
import sys
import time
from pathlib import Path

from atlas_nano import __version__


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
  atlas-nano pipeline --gguf                Include GGUF embedding in pipeline
  atlas-nano run --prompt "Hello world"     Classify a prompt
  atlas-nano benchmark                      Evaluate on test set

Config file: Place an atlas_nano.yaml in your working directory to
set defaults. CLI flags override config file values.
""",
    )

    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"atlas-nano {__version__}",
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
    sub.add_argument("--force", "-f", action="store_true",
                     help="Overwrite existing config file")

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

    # ---- sign-check ----
    sub = subparsers.add_parser("sign-check",
        help="Distill 7-gate ensemble to single energy axis (Sign-Check Atlas)")
    sub.add_argument("--gauntlet", help="Gauntlet file for validation")
    sub.add_argument("--output-dir", help="Output directory for results")
    sub.add_argument("--component", choices=["residual", "mlp.down_proj", "self_attn.o_proj"])
    sub.add_argument("--layers", type=int, nargs="+", help="Layer indices to evaluate")
    sub.add_argument("--sweep-components", action="store_true",
                     help="Sweep all components (slower but comprehensive)")
    sub.add_argument("--threshold-mode", choices=["sweep", "pr_curve", "balanced"])
    sub.add_argument("--min-recall", type=float)
    sub.add_argument("--max-fp-rate", type=float)
    sub.add_argument("--skip-categories", action="store_true",
                     help="Skip per-category analysis (Phase 2)")
    sub.add_argument("--skip-threshold", action="store_true",
                     help="Skip threshold optimization (Phase 3)")

    # ---- gguf ----
    sub = subparsers.add_parser("gguf", help="Embed safety data into GGUF model file")
    sub.add_argument("--output", "-o", help="Output GGUF path")
    sub.add_argument("--mode", choices=["sidecar", "inject"], help="sidecar (standalone) or inject (into existing)")
    sub.add_argument("--input", help="Input GGUF file (required for inject mode)")
    sub.add_argument("--energy-axis", help="Path to energy_axis.npy")
    sub.add_argument("--phase1-results", help="Path to phase1_validation.json")
    sub.add_argument("--phase3-results", help="Path to phase3_threshold.json")
    sub.add_argument("--extraction-layer", type=int)
    sub.add_argument("--threshold", type=float)

    # ---- pipeline ----
    sub = subparsers.add_parser("pipeline", help="Run full train -> cache -> calibrate -> apply -> GGUF flow")
    sub.add_argument("--gauntlet", help="Training gauntlet file")
    sub.add_argument("--skip-train", action="store_true", help="Skip training (use existing gates)")
    sub.add_argument("--skip-cache", action="store_true", help="Skip caching (use existing scores)")
    sub.add_argument("--skip-calibrate", action="store_true", help="Skip calibration")
    sub.add_argument("--skip-apply", action="store_true", help="Skip applying calibration")
    sub.add_argument("--gguf", action="store_true",
                     help="Also run Sign-Check Atlas distillation and GGUF embedding")
    sub.add_argument("--gguf-output", help="GGUF output path (default: <output-dir>/model_safety.gguf)")
    sub.add_argument("--gguf-inject", help="Inject into existing GGUF file instead of sidecar")
    sub.add_argument("--gate-dir", help="Gate directory (if skipping train)")
    sub.add_argument("--output-dir", help="Base output directory")

    # ---- profile ----
    sub = subparsers.add_parser("profile", help="Inspect model-coupled safety profiles")
    profile_sub = sub.add_subparsers(dest="profile_command", required=True)
    validate = profile_sub.add_parser("validate", help="Validate a profile manifest")
    validate.add_argument("path", help="Path to profile.json")
    validate.add_argument("--model", dest="profile_model", help="Expected base model name")
    validate.add_argument("--architecture", help="Expected model architecture")
    validate.add_argument("--revision", help="Expected base model revision")
    validate.add_argument("--hidden-dim", type=int, help="Expected hidden dimension")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Profile validation is intentionally dependency-light so compatibility can
    # be checked before loading a model or the rest of the pipeline stack.
    if args.command == "profile":
        _cmd_profile(args)
        return

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
    elif args.command == "sign-check":
        _cmd_sign_check(args, cfg)
    elif args.command == "gguf":
        _cmd_gguf(args, cfg)
    elif args.command == "pipeline":
        _cmd_pipeline(args, cfg)


# =============================================================================
# COMMAND IMPLEMENTATIONS
# =============================================================================

def _cmd_init(args):
    from atlas_nano.config import save_default_config
    if Path(args.output).exists() and not args.force:
        _abort(f"{args.output} already exists.",
               "pass --force to overwrite, or use --output <other-path>")
    path = save_default_config(args.output)
    print(f"Created config file: {path}")
    print("Edit this file to customize your pipeline, then run:")
    print("  atlas-nano pipeline")


def _cmd_profile(args):
    from atlas_nano.profile import (
        ProfileError, RuntimeModel, assert_compatible, load_profile, verify_artifacts,
    )

    try:
        profile = load_profile(args.path)
        verify_artifacts(profile, Path(args.path).resolve().parent)
        if args.profile_model:
            assert_compatible(profile, RuntimeModel(
                name=args.profile_model,
                architecture=args.architecture,
                revision=args.revision,
                hidden_dim=args.hidden_dim,
            ))
    except ProfileError as exc:
        _abort(str(exc))
    print(f"Valid profile: {profile['profile_id']}")
    print(f"Base model:    {profile['base_model']}")
    print(f"Role:          {profile['decision']['role']}")


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

    if not cfg.train.generate_data_only:
        _require_file(cfg.train.gauntlet, "gauntlet file",
                      "pass --gauntlet <path> or set train.gauntlet in atlas_nano.yaml")

    _banner("Training Safety Gates")
    _run_training(cfg)


def _cmd_cache(args, cfg):
    _apply_overrides(args, cfg.cache, {
        "gate_dir": "gate_dir",
        "gauntlet": "gauntlet",
        "output": "output",
        "obf_gate": "obf_gate",
    })

    _require_dir(cfg.cache.gate_dir, "gate directory",
                 "pass --gate-dir <dir> pointing to trained gates (run 'atlas-nano train' first)")
    _require_file(cfg.cache.gauntlet, "gauntlet file",
                  "pass --gauntlet <path> or set cache.gauntlet in atlas_nano.yaml")

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

    _require_file(cfg.calibrate.cached, "cached scores file",
                  "pass --cached <path> (run 'atlas-nano cache' first)")

    _banner("Calibrating Thresholds (CMA-ES)")
    _run_calibrate(cfg)


def _cmd_apply(args, cfg):
    _apply_overrides(args, cfg.apply, {
        "gate_dir": "gate_dir",
        "calibration": "calibration",
        "output": "output",
        "obf_gate": "obf_gate",
    })

    _require_dir(cfg.apply.gate_dir, "gate directory",
                 "pass --gate-dir <dir> with the original (uncalibrated) gates")
    _require_file(cfg.apply.calibration, "calibration result file",
                  "pass --calibration <path> (run 'atlas-nano calibrate' first)")

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

    if not args.prompt and not args.demo:
        _abort("'run' needs either --prompt \"...\" or --demo.",
               "examples: atlas-nano run --prompt 'hello world'  |  atlas-nano run --demo")
    _require_dir(cfg.run.gate_dir, "calibrated gate directory",
                 "pass --gate-dir <dir> (run 'atlas-nano apply' first)")

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

    _require_dir(cfg.benchmark.gate_dir, "gate directory",
                 "pass --gate-dir <dir> with calibrated gates")
    _require_file(cfg.benchmark.gauntlet, "test gauntlet file",
                  "pass --gauntlet <path> or set benchmark.gauntlet in atlas_nano.yaml")

    _banner("Running Benchmark")
    _run_benchmark(cfg)


def _cmd_sign_check(args, cfg):
    """Run Sign-Check Atlas: distill 7-gate ensemble to single energy axis."""
    _apply_overrides(args, cfg.sign_check, {
        "gauntlet": "gauntlet",
        "output_dir": "output_dir",
        "component": "component",
        "threshold_mode": "threshold_mode",
        "min_recall": "min_recall",
        "max_fp_rate": "max_fp_rate",
    })
    layers = getattr(args, "layers", None)
    sweep = getattr(args, "sweep_components", False)
    skip_cat = getattr(args, "skip_categories", False)
    skip_thresh = getattr(args, "skip_threshold", False)

    _require_file(cfg.sign_check.gauntlet, "gauntlet file",
                  "pass --gauntlet <path> or set sign_check.gauntlet in atlas_nano.yaml")

    _banner("Sign-Check Atlas Distillation")
    _run_sign_check(cfg, layers=layers, sweep_components=sweep,
                    skip_categories=skip_cat, skip_threshold=skip_thresh)


def _cmd_gguf(args, cfg):
    """Embed safety data into GGUF."""
    _apply_overrides(args, cfg.gguf, {
        "output": "output",
        "mode": "mode",
        "input": "input_gguf",
        "energy_axis": "energy_axis",
        "phase1_results": "phase1_results",
        "phase3_results": "phase3_results",
    })
    extraction_layer = getattr(args, "extraction_layer", None)
    threshold = getattr(args, "threshold", None)

    if cfg.gguf.phase1_results:
        _require_file(cfg.gguf.phase1_results, "phase1 results file")
    if cfg.gguf.phase3_results:
        _require_file(cfg.gguf.phase3_results, "phase3 results file")
    if cfg.gguf.energy_axis:
        _require_file(cfg.gguf.energy_axis, "energy axis file")
    if cfg.gguf.mode == "inject":
        _require_file(cfg.gguf.input_gguf, "input GGUF file",
                      "inject mode needs --input <existing model.gguf>")

    _banner("GGUF Safety Embedding")
    _run_gguf_embed(cfg, extraction_layer=extraction_layer, threshold=threshold)


def _cmd_pipeline(args, cfg):
    """Run the full train -> cache -> calibrate -> apply (-> GGUF) pipeline."""
    gauntlet = args.gauntlet or cfg.train.gauntlet
    cfg.train.gauntlet = gauntlet
    cfg.cache.gauntlet = gauntlet
    cfg.sign_check.gauntlet = gauntlet

    output_dir = args.output_dir or "atlas_output"
    gate_dir = args.gate_dir or f"{output_dir}/gates"
    cached_path = f"{output_dir}/cached_scores.json"
    cal_path = f"{output_dir}/calibration_result.json"
    cal_gate_dir = f"{output_dir}/gates_calibrated"
    sc_dir = f"{output_dir}/sign_check"

    cfg.train.output_dir = gate_dir
    cfg.cache.gate_dir = gate_dir
    cfg.cache.output = cached_path
    cfg.calibrate.cached = cached_path
    cfg.calibrate.output = cal_path
    cfg.apply.gate_dir = gate_dir
    cfg.apply.calibration = cal_path
    cfg.apply.output = cal_gate_dir
    cfg.sign_check.output_dir = sc_dir

    include_gguf = args.gguf
    if include_gguf:
        gguf_output = args.gguf_output or f"{output_dir}/model_safety.gguf"
        cfg.gguf.output = gguf_output
        cfg.gguf.phase1_results = f"{sc_dir}/phase1_validation.json"
        cfg.gguf.phase3_results = f"{sc_dir}/phase3_threshold.json"
        if args.gguf_inject:
            cfg.gguf.mode = "inject"
            cfg.gguf.input_gguf = args.gguf_inject
            _require_file(args.gguf_inject, "input GGUF for --gguf-inject")

    if not args.skip_train:
        _require_file(gauntlet, "gauntlet file",
                      "pass --gauntlet <path> or use --skip-train with --gate-dir")
    elif not args.skip_cache:
        _require_dir(gate_dir, "gate directory",
                     "with --skip-train you must pass --gate-dir <existing gates>")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Count total steps
    base_steps = 4
    gguf_steps = 2 if include_gguf else 0
    total = base_steps + gguf_steps
    # Adjust for skips
    skipped = sum([args.skip_train, args.skip_cache, args.skip_calibrate, args.skip_apply])
    total -= skipped

    steps = []
    step_i = [0]

    def _step(name, fn):
        step_i[0] += 1
        steps.append((f"{step_i[0]}/{total}", name, fn))

    if not args.skip_train:
        _step("Training safety gates", lambda: _run_training(cfg))
    if not args.skip_cache:
        _step("Caching gate scores", lambda: _run_cache(cfg))
    if not args.skip_calibrate:
        _step("Calibrating thresholds", lambda: _run_calibrate(cfg))
    if not args.skip_apply:
        _step("Applying calibration", lambda: _run_apply(cfg))
    if include_gguf:
        _step("Sign-Check Atlas distillation",
              lambda: _run_sign_check(cfg, skip_categories=False, skip_threshold=False))
        _step("Embedding into GGUF", lambda: _run_gguf_embed(cfg))

    _banner("Atlas-Nano Full Pipeline")
    print(f"  Model:      {cfg.model.name}")
    print(f"  Device:     {cfg.model.device}")
    print(f"  Gauntlet:   {gauntlet}")
    print(f"  Output:     {output_dir}/")
    print(f"  GGUF:       {'yes' if include_gguf else 'no (use --gguf to enable)'}")
    print(f"  Steps:      {len(steps)}")
    print()

    t0 = time.time()
    for step_num, step_name, step_fn in steps:
        print(f"[{step_num}] {step_name}...")
        step_t0 = time.time()
        step_fn()
        elapsed = time.time() - step_t0
        print(f"  Done in {elapsed:.1f}s\n")

    total_time = time.time() - t0
    print("=" * 60)
    print(f"Pipeline complete in {total_time:.1f}s")
    print(f"Calibrated gates: {cal_gate_dir}/")
    if include_gguf:
        print(f"GGUF sidecar:     {cfg.gguf.output}")
    print()
    print("Next steps:")
    print(f"  atlas-nano run --gate-dir {cal_gate_dir} --prompt 'your prompt'")
    print(f"  atlas-nano run --gate-dir {cal_gate_dir} --demo")
    print(f"  atlas-nano benchmark --gate-dir {cal_gate_dir}")
    if include_gguf:
        print()
        print("GGUF integration:")
        print(f"  Sidecar file ready at: {cfg.gguf.output}")
        print(f"  See sign_check_atlas/llama_cpp_patch/ for llama.cpp integration")


# =============================================================================
# PIPELINE STEP RUNNERS
# =============================================================================

def _run_training(cfg):
    """Run gate training."""
    from atlas_nano.pipeline.training import (
        TrainingConfig, add_expanded_data, export_expanded_gauntlet,
        load_gauntlet, run_pipeline,
    )

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
    from atlas_nano.pipeline.cache import main as cache_main

    argv = [
        "atlas_nano.pipeline.cache",
        "--model", cfg.model.name,
        "--gate-dir", cfg.cache.gate_dir,
        "--gauntlet", cfg.cache.gauntlet,
        "--output", cfg.cache.output,
        "--device", cfg.model.device,
    ]
    if cfg.cache.obf_gate:
        argv += ["--obf-gate", cfg.cache.obf_gate]
    _run_entrypoint(cache_main, argv)


def _run_calibrate(cfg):
    """Run CMA-ES calibration."""
    from atlas_nano.pipeline.calibrate import main as cal_main

    argv = [
        "atlas_nano.pipeline.calibrate",
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
        argv.append("--no-grid-refine")
    if not cfg.calibrate.optimize_weights:
        argv.append("--no-optimize-weights")
    _run_entrypoint(cal_main, argv)


def _run_apply(cfg):
    """Run calibration application."""
    from atlas_nano.pipeline.apply import main as apply_main

    argv = [
        "atlas_nano.pipeline.apply",
        "--gate-dir", cfg.apply.gate_dir,
        "--calibration", cfg.apply.calibration,
        "--output", cfg.apply.output,
    ]
    if cfg.apply.obf_gate:
        argv += ["--obf-gate", cfg.apply.obf_gate]
    _run_entrypoint(apply_main, argv)


def _run_inference(cfg, prompt=None, demo=False):
    """Run live inference."""
    from atlas_nano.pipeline.inference import main as run_main

    argv = [
        "atlas_nano.pipeline.inference",
        "--model", cfg.model.name,
        "--gate-dir", cfg.run.gate_dir,
        "--device", cfg.model.device,
        "--fast-path-assist", str(cfg.run.fast_path_assist),
        "--block-assist", str(cfg.run.block_assist),
        "--assist-strength", str(cfg.run.assist_strength),
        "--event-horizon", str(cfg.run.event_horizon),
    ]
    if prompt:
        argv += ["--prompt", prompt]
    if demo:
        argv.append("--demo")
    if cfg.run.benign_matrix:
        argv += ["--benign-matrix", cfg.run.benign_matrix]
    _run_entrypoint(run_main, argv)


def _run_benchmark(cfg):
    """Run benchmark evaluation."""
    from atlas_nano.pipeline.benchmark import main as bench_main

    argv = [
        "atlas_nano.pipeline.benchmark",
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
        argv += ["--max-prompts", str(cfg.benchmark.max_prompts)]
    if cfg.benchmark.benign_matrix:
        argv += ["--benign-matrix", cfg.benchmark.benign_matrix]
    if cfg.benchmark.obf_gate:
        argv += ["--obf-gate", cfg.benchmark.obf_gate]
    if hasattr(cfg.benchmark, "_uniform_threshold") and cfg.benchmark._uniform_threshold is not None:
        argv += ["--uniform-threshold", str(cfg.benchmark._uniform_threshold)]
    _run_entrypoint(bench_main, argv)


def _run_sign_check(cfg, layers=None, sweep_components=False,
                     skip_categories=False, skip_threshold=False):
    """Run Sign-Check Atlas: Phase 1 (validate) + Phase 2 (categories) + Phase 3 (threshold)."""
    output_dir = cfg.sign_check.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Phase 1: Hypothesis validation - compute energy axis
    print("  Phase 1: Validating energy axis hypothesis...")
    from sign_check_atlas.validate_hypothesis import main as validate_main
    argv = [
        "validate_hypothesis",
        "--model", cfg.model.name,
        "--gauntlet", cfg.sign_check.gauntlet,
        "--output-dir", output_dir,
        "--device", cfg.model.device,
        "--component", cfg.sign_check.component,
        "--batch-size", str(cfg.model.batch_size),
    ]
    if layers:
        argv += ["--layers"] + [str(l) for l in layers]
    if sweep_components:
        argv.append("--sweep-components")
    _run_entrypoint(validate_main, argv)

    # Phase 2: Per-category analysis
    if not skip_categories:
        print("  Phase 2: Per-category analysis...")
        from sign_check_atlas.category_analysis import main as category_main
        energy_axis_path = f"{output_dir}/energy_axis.npy"
        argv = [
            "category_analysis",
            "--model", cfg.model.name,
            "--gauntlet", cfg.sign_check.gauntlet,
            "--energy-axis", energy_axis_path,
            "--output-dir", output_dir,
            "--device", cfg.model.device,
            "--component", cfg.sign_check.component,
            "--batch-size", str(cfg.model.batch_size),
        ]
        if layers:
            argv += ["--layers"] + [str(l) for l in layers]
        _run_entrypoint(category_main, argv)

    # Phase 3: Threshold optimization
    if not skip_threshold:
        print("  Phase 3: Threshold optimization...")
        from sign_check_atlas.threshold_search import main as threshold_main
        argv = [
            "threshold_search",
            "--harm-energies", f"{output_dir}/harm_energies.npy",
            "--safe-energies", f"{output_dir}/safe_energies.npy",
            "--mode", cfg.sign_check.threshold_mode,
            "--optimize", cfg.sign_check.optimize_metric,
            "--min-recall", str(cfg.sign_check.min_recall),
            "--max-fp-rate", str(cfg.sign_check.max_fp_rate),
            "--output-dir", output_dir,
        ]
        _run_entrypoint(threshold_main, argv)


def _run_gguf_embed(cfg, extraction_layer=None, threshold=None):
    """Run GGUF safety embedding (Phase 4)."""
    from sign_check_atlas.gguf_integration.embed_safety import main as embed_main

    argv = [
        "embed_safety",
        "--output", cfg.gguf.output,
        "--mode", cfg.gguf.mode,
        "--model-name", cfg.model.name,
    ]
    if cfg.gguf.phase1_results:
        argv += ["--phase1-results", cfg.gguf.phase1_results]
    if cfg.gguf.phase3_results:
        argv += ["--phase3-results", cfg.gguf.phase3_results]
    if cfg.gguf.energy_axis:
        argv += ["--energy-axis", cfg.gguf.energy_axis]
    if cfg.gguf.input_gguf:
        argv += ["--input", cfg.gguf.input_gguf]
    if extraction_layer is not None:
        argv += ["--extraction-layer", str(extraction_layer)]
    if threshold is not None:
        argv += ["--threshold", str(threshold)]
    _run_entrypoint(embed_main, argv)


# =============================================================================
# HELPERS
# =============================================================================

def _run_entrypoint(entrypoint, argv):
    """Run an argparse entry point without leaking process-global argv changes."""
    original_argv = sys.argv
    try:
        sys.argv = argv
        return entrypoint()
    finally:
        sys.argv = original_argv

def _apply_overrides(args, cfg_section, mapping: dict):
    """Apply CLI args onto a config section. Only overrides if arg is not None."""
    for arg_name, cfg_name in mapping.items():
        val = getattr(args, arg_name, None)
        if val is not None:
            setattr(cfg_section, cfg_name, val)


def _abort(msg: str, hint: str = None):
    """Print a friendly error and exit with status 2."""
    print(f"Error: {msg}", file=sys.stderr)
    if hint:
        print(f"Hint:  {hint}", file=sys.stderr)
    sys.exit(2)


def _require_file(path, label: str, hint: str = None):
    """Exit with a friendly message if the path is missing or not a file."""
    if not path:
        _abort(f"missing required {label}.", hint)
    p = Path(path)
    if not p.exists():
        _abort(f"{label} not found: {path}", hint)
    if not p.is_file():
        _abort(f"{label} is not a file: {path}", hint)


def _require_dir(path, label: str, hint: str = None):
    """Exit with a friendly message if the path is missing or not a directory."""
    if not path:
        _abort(f"missing required {label}.", hint)
    p = Path(path)
    if not p.exists():
        _abort(f"{label} not found: {path}", hint)
    if not p.is_dir():
        _abort(f"{label} is not a directory: {path}", hint)


def _banner(title: str):
    width = 60
    print()
    print("=" * width)
    print(f"  Atlas-Nano | {title}")
    print("=" * width)
    print()


if __name__ == "__main__":
    main()
