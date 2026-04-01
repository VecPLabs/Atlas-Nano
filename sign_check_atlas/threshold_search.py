#!/usr/bin/env python3
"""
Sign-Check Atlas: Phase 3 - Optimal Threshold Search
======================================================
Sign=0 may not be the optimal decision boundary. This script searches
for the threshold that maximizes classification performance, using
precision-recall curve analysis and optionally CMA-ES optimization.

Modes:
    - sweep:    Dense linear sweep across energy range
    - pr_curve: Precision-recall curve analysis with optimal F1 selection
    - balanced: Optimize for balanced precision/recall with constraints
    - cmaes:    CMA-ES evolutionary optimization (requires cma package)

Usage:
    # Using pre-extracted data from Phase 1
    python sign_check_atlas/threshold_search.py \
        --harm-energies sign_check_atlas/results/harm_energies.npy \
        --safe-energies sign_check_atlas/results/safe_energies.npy \
        --output-dir sign_check_atlas/results

    # Full pipeline with model
    python sign_check_atlas/threshold_search.py \
        --model Qwen/Qwen3-4B \
        --gauntlet gauntlet_v3_corrected.txt \
        --output-dir sign_check_atlas/results

    # Constrained search: recall >= 0.85, minimize FP rate
    python sign_check_atlas/threshold_search.py \
        --harm-energies sign_check_atlas/results/harm_energies.npy \
        --safe-energies sign_check_atlas/results/safe_energies.npy \
        --mode balanced --min-recall 0.85 --max-fp-rate 0.05

Patent Pending: USPTO 63/931,565
Copyright (c) 2025-2026 David Cappelli / VecP Labs
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime


# =============================================================================
# THRESHOLD SEARCH METHODS
# =============================================================================

def compute_metrics_at_threshold(
    harm_energies: np.ndarray,
    safe_energies: np.ndarray,
    threshold: float,
) -> Dict:
    """Compute classification metrics at a given threshold."""
    tp = int(np.sum(harm_energies > threshold))
    fn = int(np.sum(harm_energies <= threshold))
    fp = int(np.sum(safe_energies > threshold))
    tn = int(np.sum(safe_energies <= threshold))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fn + fp + tn)
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "fp_rate": fp_rate,
    }


def sweep_search(
    harm_energies: np.ndarray,
    safe_energies: np.ndarray,
    n_points: int = 1000,
    optimize: str = "f1",
) -> Tuple[Dict, List[Dict]]:
    """
    Dense linear sweep across the full energy range.

    Returns best result and full sweep curve.
    """
    all_energies = np.concatenate([harm_energies, safe_energies])
    lo, hi = float(all_energies.min()), float(all_energies.max())
    margin = (hi - lo) * 0.05
    thresholds = np.linspace(lo - margin, hi + margin, n_points)

    results = []
    best_score = -1
    best_result = None

    for t in thresholds:
        m = compute_metrics_at_threshold(harm_energies, safe_energies, t)
        results.append(m)

        score = m[optimize]
        if score > best_score:
            best_score = score
            best_result = m

    return best_result, results


def pr_curve_search(
    harm_energies: np.ndarray,
    safe_energies: np.ndarray,
) -> Tuple[Dict, List[Dict]]:
    """
    Precision-recall curve analysis.

    Sorts by energy value and sweeps all unique thresholds to find
    optimal F1 point on the precision-recall curve.
    """
    all_energies = np.concatenate([safe_energies, harm_energies])
    all_labels = np.concatenate([np.zeros(len(safe_energies)), np.ones(len(harm_energies))])

    # Sort by energy
    sorted_idx = np.argsort(all_energies)
    sorted_energies = all_energies[sorted_idx]
    sorted_labels = all_labels[sorted_idx]

    # Unique thresholds at midpoints between consecutive distinct values
    unique_vals = np.unique(sorted_energies)
    if len(unique_vals) > 1:
        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
    else:
        thresholds = unique_vals

    results = []
    best_f1 = -1
    best_result = None

    for t in thresholds:
        m = compute_metrics_at_threshold(harm_energies, safe_energies, t)
        results.append(m)

        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_result = m

    return best_result, results


def balanced_search(
    harm_energies: np.ndarray,
    safe_energies: np.ndarray,
    min_recall: float = 0.85,
    max_fp_rate: float = 0.05,
    n_points: int = 2000,
) -> Tuple[Optional[Dict], List[Dict]]:
    """
    Search for threshold satisfying recall and FP rate constraints.

    Strategy:
        1. Find all thresholds meeting min_recall
        2. Among those, find the one with lowest FP rate
        3. If no threshold meets both constraints, return the closest

    Returns best constrained result and all feasible results.
    """
    all_energies = np.concatenate([harm_energies, safe_energies])
    lo, hi = float(all_energies.min()), float(all_energies.max())
    margin = (hi - lo) * 0.05
    thresholds = np.linspace(lo - margin, hi + margin, n_points)

    feasible = []
    all_results = []

    for t in thresholds:
        m = compute_metrics_at_threshold(harm_energies, safe_energies, t)
        all_results.append(m)

        if m["recall"] >= min_recall and m["fp_rate"] <= max_fp_rate:
            feasible.append(m)

    if feasible:
        # Among feasible, pick highest F1
        best = max(feasible, key=lambda x: x["f1"])
        return best, feasible
    else:
        # No feasible point found - return closest to constraints
        # Minimize distance to constraint satisfaction
        best_dist = float('inf')
        best = None
        for m in all_results:
            recall_gap = max(0, min_recall - m["recall"])
            fp_gap = max(0, m["fp_rate"] - max_fp_rate)
            dist = recall_gap + fp_gap
            if dist < best_dist:
                best_dist = dist
                best = m
        return best, all_results


def multi_threshold_analysis(
    harm_energies: np.ndarray,
    safe_energies: np.ndarray,
) -> Dict:
    """
    Analyze key operating points on the threshold curve.

    Returns metrics at several strategically chosen thresholds.
    """
    operating_points = {}

    # Sign check (threshold=0)
    operating_points["sign_check_0"] = compute_metrics_at_threshold(
        harm_energies, safe_energies, 0.0
    )

    # Mean-based threshold
    mean_threshold = (harm_energies.mean() + safe_energies.mean()) / 2
    operating_points["mean_midpoint"] = compute_metrics_at_threshold(
        harm_energies, safe_energies, mean_threshold
    )

    # Median-based threshold
    median_threshold = (np.median(harm_energies) + np.median(safe_energies)) / 2
    operating_points["median_midpoint"] = compute_metrics_at_threshold(
        harm_energies, safe_energies, median_threshold
    )

    # 95% recall point
    sorted_harm = np.sort(harm_energies)
    idx_95 = max(0, int(len(sorted_harm) * 0.05))
    recall_95_threshold = sorted_harm[idx_95]
    operating_points["recall_95"] = compute_metrics_at_threshold(
        harm_energies, safe_energies, recall_95_threshold
    )

    # 99% precision point
    sorted_safe = np.sort(safe_energies)
    idx_99 = min(len(sorted_safe) - 1, int(len(sorted_safe) * 0.99))
    precision_99_threshold = sorted_safe[idx_99]
    operating_points["precision_99"] = compute_metrics_at_threshold(
        harm_energies, safe_energies, precision_99_threshold
    )

    return operating_points


# =============================================================================
# BOUNDARY ZONE ANALYSIS
# =============================================================================

def boundary_zone_analysis(
    harm_energies: np.ndarray,
    safe_energies: np.ndarray,
    threshold: float,
    zone_width: float = None,
) -> Dict:
    """
    Analyze the boundary zone around the threshold where classification
    is uncertain. These are candidates for Tier 2 (full Atlas) analysis.

    Returns the percentage of samples in the boundary zone and their
    class distribution.
    """
    if zone_width is None:
        # Auto: use 10% of the total energy range
        all_energies = np.concatenate([harm_energies, safe_energies])
        zone_width = (all_energies.max() - all_energies.min()) * 0.10

    lo = threshold - zone_width / 2
    hi = threshold + zone_width / 2

    harm_in_zone = np.sum((harm_energies >= lo) & (harm_energies <= hi))
    safe_in_zone = np.sum((safe_energies >= lo) & (safe_energies <= hi))
    total_in_zone = harm_in_zone + safe_in_zone

    return {
        "zone_low": float(lo),
        "zone_high": float(hi),
        "zone_width": float(zone_width),
        "harm_in_zone": int(harm_in_zone),
        "safe_in_zone": int(safe_in_zone),
        "total_in_zone": int(total_in_zone),
        "harm_pct_in_zone": float(harm_in_zone / len(harm_energies)) if len(harm_energies) > 0 else 0.0,
        "safe_pct_in_zone": float(safe_in_zone / len(safe_energies)) if len(safe_energies) > 0 else 0.0,
        "zone_purity_harm": float(harm_in_zone / total_in_zone) if total_in_zone > 0 else 0.0,
    }


# =============================================================================
# MAIN
# =============================================================================

def run_threshold_search(args):
    print("=" * 70)
    print("SIGN-CHECK ATLAS: Phase 3 Threshold Search")
    print("=" * 70)

    # --- Load energy distributions ---
    if args.harm_energies and args.safe_energies:
        print("[1/4] Loading pre-computed energy distributions...")
        harm_energies = np.load(args.harm_energies)
        safe_energies = np.load(args.safe_energies)
    else:
        # Full pipeline: load model, extract, compute
        print("[1/4] Computing energy distributions from model...")
        import torch
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from sign_check_atlas.validate_hypothesis import (
            load_gauntlet, split_by_label, ActivationExtractor,
            compute_energy_axis, evaluate_sign_check
        )
        from transformers import AutoModelForCausalLM, AutoTokenizer

        categories = load_gauntlet(args.gauntlet)
        harmful_prompts, benign_prompts, _ = split_by_label(categories)

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        model = model.to(args.device)
        model.eval()

        n_layers = model.config.num_hidden_layers
        layers = args.layers or list(range(int(n_layers * 0.4), int(n_layers * 0.7) + 1))

        all_prompts = harmful_prompts + benign_prompts
        extractor = ActivationExtractor(model, tokenizer, layers, args.component, args.device)
        try:
            embeddings = extractor.extract_batch(all_prompts, args.batch_size)
        finally:
            extractor.cleanup()

        energy_axis, _, _, _ = compute_energy_axis(
            embeddings, harmful_prompts, benign_prompts, layers
        )
        _, harm_energies, safe_energies = evaluate_sign_check(
            embeddings, energy_axis, harmful_prompts, benign_prompts, layers
        )

    print(f"  Harmful energies: {len(harm_energies)} samples")
    print(f"  Safe energies:    {len(safe_energies)} samples")

    # --- Threshold search ---
    print(f"\n[2/4] Searching for optimal threshold (mode: {args.mode})...")

    if args.mode == "sweep":
        best, curve = sweep_search(harm_energies, safe_energies, optimize=args.optimize)
    elif args.mode == "pr_curve":
        best, curve = pr_curve_search(harm_energies, safe_energies)
    elif args.mode == "balanced":
        best, curve = balanced_search(
            harm_energies, safe_energies,
            min_recall=args.min_recall, max_fp_rate=args.max_fp_rate
        )
    else:
        best, curve = sweep_search(harm_energies, safe_energies)

    # --- Compare to sign check ---
    sign_check = compute_metrics_at_threshold(harm_energies, safe_energies, 0.0)

    print(f"\n  {'='*60}")
    print(f"  THRESHOLD COMPARISON")
    print(f"  {'='*60}")
    print(f"  {'Metric':<15} {'Sign (t=0)':<15} {'Optimal':<15} {'Diff':<10}")
    print(f"  {'-'*55}")
    for metric in ["precision", "recall", "f1", "accuracy", "fp_rate"]:
        s = sign_check[metric]
        o = best[metric]
        diff = o - s
        print(f"  {metric:<15} {s:<15.4f} {o:<15.4f} {diff:+.4f}")
    print(f"  {'threshold':<15} {'0.0000':<15} {best['threshold']:<15.4f}")

    # --- Multi-threshold operating points ---
    print(f"\n[3/4] Analyzing operating points...")
    operating_points = multi_threshold_analysis(harm_energies, safe_energies)

    print(f"\n  {'='*75}")
    print(f"  OPERATING POINTS")
    print(f"  {'='*75}")
    print(f"  {'Point':<20} {'Threshold':<12} {'Prec':<8} {'Rec':<8} {'F1':<8} {'FPR':<8}")
    print(f"  {'-'*64}")
    for name, m in operating_points.items():
        print(f"  {name:<20} {m['threshold']:<12.4f} {m['precision']:<8.3f} "
              f"{m['recall']:<8.3f} {m['f1']:<8.3f} {m['fp_rate']:<8.3f}")

    # --- Boundary zone analysis ---
    print(f"\n[4/4] Analyzing boundary zone...")
    bz = boundary_zone_analysis(harm_energies, safe_energies, best["threshold"])
    print(f"  Zone: [{bz['zone_low']:.4f}, {bz['zone_high']:.4f}] (width={bz['zone_width']:.4f})")
    print(f"  Harmful in zone: {bz['harm_in_zone']} ({bz['harm_pct_in_zone']:.1%})")
    print(f"  Safe in zone:    {bz['safe_in_zone']} ({bz['safe_pct_in_zone']:.1%})")
    print(f"  Zone purity (harm): {bz['zone_purity_harm']:.1%}")

    # --- Save results ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_results = {
        "timestamp": datetime.now().isoformat(),
        "mode": args.mode,
        "sign_check_metrics": sign_check,
        "optimal_threshold_metrics": best,
        "improvement_over_sign_check": {
            k: best[k] - sign_check[k] for k in ["precision", "recall", "f1", "accuracy", "fp_rate"]
        },
        "operating_points": operating_points,
        "boundary_zone": bz,
        "n_harm": len(harm_energies),
        "n_safe": len(safe_energies),
    }

    results_path = output_dir / "phase3_threshold.json"
    with open(results_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    print(f"\n  Saved: {results_path}")

    # Save the optimal threshold for Phase 4
    threshold_path = output_dir / "optimal_threshold.json"
    with open(threshold_path, 'w') as f:
        json.dump({
            "threshold": best["threshold"],
            "f1": best["f1"],
            "precision": best["precision"],
            "recall": best["recall"],
            "fp_rate": best["fp_rate"],
        }, f, indent=2)
    print(f"  Saved: {threshold_path}")

    # Generate report
    report = generate_threshold_report(full_results, sign_check, best, operating_points, bz)
    report_path = output_dir / "phase3_threshold.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Saved: {report_path}")

    print(f"\n{'='*70}")
    print(f"Phase 3 complete. Optimal threshold: {best['threshold']:.4f} (F1={best['f1']:.3f})")
    print(f"{'='*70}")

    return full_results


def generate_threshold_report(results, sign_check, best, operating_points, bz) -> str:
    lines = [
        "# Sign-Check Atlas: Phase 3 Threshold Search",
        "",
        f"**Date:** {results['timestamp']}",
        f"**Mode:** {results['mode']}",
        "",
        "## Sign Check vs Optimal Threshold",
        "",
        "| Metric | Sign (t=0) | Optimal | Improvement |",
        "|--------|-----------|---------|-------------|",
    ]

    for metric in ["precision", "recall", "f1", "accuracy", "fp_rate"]:
        s = sign_check[metric]
        o = best[metric]
        d = results["improvement_over_sign_check"][metric]
        lines.append(f"| {metric} | {s:.4f} | {o:.4f} | {d:+.4f} |")

    lines.append(f"| threshold | 0.0000 | {best['threshold']:.4f} | |")

    lines.extend([
        "",
        "## Operating Points",
        "",
        "| Point | Threshold | Precision | Recall | F1 | FP Rate |",
        "|-------|-----------|-----------|--------|----|---------| ",
    ])

    for name, m in operating_points.items():
        lines.append(
            f"| {name} | {m['threshold']:.4f} | {m['precision']:.3f} | "
            f"{m['recall']:.3f} | {m['f1']:.3f} | {m['fp_rate']:.3f} |"
        )

    lines.extend([
        "",
        "## Boundary Zone",
        "",
        f"- Zone: [{bz['zone_low']:.4f}, {bz['zone_high']:.4f}]",
        f"- Harmful in zone: {bz['harm_in_zone']} ({bz['harm_pct_in_zone']:.1%})",
        f"- Safe in zone: {bz['safe_in_zone']} ({bz['safe_pct_in_zone']:.1%})",
        f"- Zone purity (harm): {bz['zone_purity_harm']:.1%}",
        "",
        "Samples in the boundary zone are candidates for Tier 2 (full Atlas) analysis.",
        "",
        "---",
        "*Generated by Sign-Check Atlas Phase 3 threshold search*",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Sign-Check Atlas: Phase 3 Threshold Search"
    )

    # Pre-computed data
    parser.add_argument("--harm-energies", type=str, default=None,
                       help="Path to harm_energies.npy from Phase 1")
    parser.add_argument("--safe-energies", type=str, default=None,
                       help="Path to safe_energies.npy from Phase 1")

    # Full pipeline (if no pre-computed data)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--gauntlet", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--component", type=str, default="residual")
    parser.add_argument("--batch-size", type=int, default=8)

    # Search mode
    parser.add_argument("--mode", type=str, default="sweep",
                       choices=["sweep", "pr_curve", "balanced"],
                       help="Search mode")
    parser.add_argument("--optimize", type=str, default="f1",
                       choices=["f1", "accuracy", "precision", "recall"],
                       help="Metric to optimize (for sweep mode)")
    parser.add_argument("--min-recall", type=float, default=0.85,
                       help="Minimum recall constraint (for balanced mode)")
    parser.add_argument("--max-fp-rate", type=float, default=0.05,
                       help="Maximum FP rate constraint (for balanced mode)")

    parser.add_argument("--output-dir", type=str, default="sign_check_atlas/results")

    args = parser.parse_args()

    if not args.harm_energies and not args.model:
        parser.error("Either --harm-energies/--safe-energies or --model/--gauntlet required")

    run_threshold_search(args)


if __name__ == "__main__":
    main()
