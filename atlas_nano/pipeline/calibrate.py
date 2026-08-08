#!/usr/bin/env python3
"""
VecP CMA-ES Calibration
=======================
Use CMA-ES (Covariance Matrix Adaptation Evolution Strategy) to find optimal
thresholds and weights for gate ensemble.

This runs on CACHED scores - no model inference needed. Calibration takes
seconds instead of hours.

Usage:
    python -m atlas_nano.pipeline.calibrate \
        --cached cached_scores.json \
        --target-recall 0.92 \
        --max-fp 0.12 \
        --output calibration_result.json

Optimization targets:
    - Maximize recall subject to FP rate constraint
    - Or minimize FP rate subject to recall constraint
    - Or maximize F1 score
    - Or custom weighted objective

Copyright (c) 2025 David Cappelli / VecP Labs
"""

import argparse
import json
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# CMA-ES - install with: pip install cma
try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False
    print("WARNING: cma not installed. Install with: pip install cma")


@dataclass
class CalibrationConfig:
    """Configuration for calibration."""
    target_recall: float = 0.92
    max_fp_rate: float = 0.12
    min_recall: float = 0.85  # Hard floor
    
    # Optimization mode
    mode: str = "balanced"  # "recall", "precision", "f1", "balanced"
    
    # What to optimize
    optimize_thresholds: bool = True
    optimize_weights: bool = True
    optimize_block_assist: bool = True
    optimize_output_gate: bool = True  # NEW: optimize assist_strength, event_horizon, fast_path_assist
    
    # Aggregation mode
    aggregation: str = "snr_weighted"  # "max_mean", "fisher_weighted", "snr_weighted", "weighted_sum"
    
    # CMA-ES parameters
    sigma0: float = 0.03  # Initial step size
    max_iterations: int = 500
    population_size: int = 20


class CachedClassifier:
    """
    Classifier that works on cached scores.
    No model inference - just math on pre-computed scores.
    """
    
    def __init__(self, cached_data: Dict):
        self.metadata = cached_data["metadata"]
        self.gate_info = cached_data["gate_info"]
        self.prompts = cached_data["prompts"]
        
        self.gate_names = self.metadata["gates"]
        self.prompt_ids = list(self.prompts.keys())
        
        # Pre-extract for speed
        self.labels = {pid: p["label"] for pid, p in self.prompts.items()}
        self.scores = {pid: p["scores"] for pid, p in self.prompts.items()}
        self.categories = {pid: p["category"] for pid, p in self.prompts.items()}
        
        # Get Fisher scores for default weighting
        self.fisher_scores = {
            gate: info["fisher"] for gate, info in self.gate_info.items()
        }
        
        # Get SNR scores (preferred for weighting)
        self.snr_scores = {
            gate: info.get("snr", info["fisher"])  # Fall back to fisher if no SNR
            for gate, info in self.gate_info.items()
        }
        
        # Count harmful/benign
        self.num_harmful = sum(1 for l in self.labels.values() if l == "harmful")
        self.num_benign = sum(1 for l in self.labels.values() if l == "benign")
        
        print(f"  Loaded {len(self.prompts)} prompts ({self.num_harmful} harmful, {self.num_benign} benign)")
        print(f"  Gates: {self.gate_names}")
        
        # Print SNR vs Fisher comparison
        print(f"\n  Gate Quality (SNR vs Fisher):")
        print(f"  {'Gate':<10} {'Fisher':>8} {'SNR':>8} {'SNR Rank':>10}")
        snr_ranked = sorted(self.snr_scores.items(), key=lambda x: -x[1])
        for rank, (gate, snr) in enumerate(snr_ranked, 1):
            fisher = self.fisher_scores.get(gate, 0)
            print(f"  {gate:<10} {fisher:>8.2f} {snr:>8.2f} {rank:>10}")
    
    def classify_prompt(
        self,
        prompt_id: str,
        thresholds: Dict[str, float],
        weights: Dict[str, float],
        block_assist: float,
        aggregation: str = "fisher_weighted",
        fast_path_assist: float = 0.85,
        assist_strength: float = 0.9,
        event_horizon: float = 1.0
    ) -> str:
        """Classify a single prompt using cached scores with full pipeline simulation."""
        scores = self.scores[prompt_id]
        
        normalized = []
        weighted_scores = []
        
        for gate in self.gate_names:
            if gate not in scores:
                continue
                
            score = scores[gate]
            threshold = thresholds.get(gate, 0.1)
            weight = weights.get(gate, 1.0)
            
            # Sigmoid normalization centered at threshold
            denom = max(abs(threshold), 0.1)
            norm = 1 / (1 + np.exp(-5 * (score - threshold) / denom))
            
            normalized.append(norm)
            weighted_scores.append(norm * weight)
        
        if not normalized:
            return "benign"
        
        # Aggregate based on mode
        if aggregation == "max_mean":
            harm_signal = 0.7 * max(normalized) + 0.3 * np.mean(normalized)
        elif aggregation == "fisher_weighted":
            total_weight = sum(weights.get(g, 1.0) for g in self.gate_names if g in scores)
            harm_signal = sum(weighted_scores) / total_weight if total_weight > 0 else 0
        elif aggregation == "snr_weighted":
            # Use SNR scores for weighting instead of provided weights
            snr_weighted_scores = []
            for gate in self.gate_names:
                if gate not in scores:
                    continue
                score = scores[gate]
                threshold = thresholds.get(gate, 0.1)
                snr = self.snr_scores.get(gate, 1.0)
                denom = max(abs(threshold), 0.1)
                norm = 1 / (1 + np.exp(-5 * (score - threshold) / denom))
                snr_weighted_scores.append(norm * snr)
            total_snr = sum(self.snr_scores.get(g, 1.0) for g in self.gate_names if g in scores)
            harm_signal = sum(snr_weighted_scores) / total_snr if total_snr > 0 else 0
        elif aggregation == "weighted_sum":
            total_weight = sum(weights.values())
            harm_signal = sum(weighted_scores) / total_weight if total_weight > 0 else 0
        elif aggregation == "max":
            harm_signal = max(normalized)
        else:
            harm_signal = np.mean(normalized)
        
        benign_assist = 1.0 - harm_signal
        
        # === ROUTING LOGIC (matches vecp_full_stack_v2.py) ===
        
        # Fast path: clearly benign
        if benign_assist >= fast_path_assist:
            return "benign"
        
        # Block path: clearly harmful
        if benign_assist <= block_assist:
            return "harmful"
        
        # Middle zone: simulate output gate behavior
        # In the real system, this goes through gravity calculation
        # Here we simulate with assist_strength and event_horizon
        
        # Effective benign signal after assist
        # Higher assist_strength = more help for borderline cases
        effective_assist = benign_assist * assist_strength
        
        # If effective assist is low enough, treat as harmful
        # event_horizon determines how strict the output gate is
        # Lower event_horizon = more permissive
        threshold_for_pass = block_assist + (fast_path_assist - block_assist) * (1 - event_horizon)
        
        if effective_assist <= threshold_for_pass:
            return "harmful"
        else:
            return "benign"
    
    def evaluate(
        self,
        thresholds: Dict[str, float],
        weights: Dict[str, float],
        block_assist: float,
        aggregation: str = "fisher_weighted",
        fast_path_assist: float = 0.85,
        assist_strength: float = 0.9,
        event_horizon: float = 1.0
    ) -> Dict:
        """Evaluate full dataset and return metrics."""
        tp = tn = fp = fn = 0
        
        for pid in self.prompt_ids:
            pred = self.classify_prompt(
                pid, thresholds, weights, block_assist, aggregation,
                fast_path_assist, assist_strength, event_horizon
            )
            label = self.labels[pid]
            
            if label == "harmful":
                if pred == "harmful":
                    tp += 1
                else:
                    fn += 1
            else:
                if pred == "benign":
                    tn += 1
                else:
                    fp += 1
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "recall": recall,
            "precision": precision,
            "fp_rate": fp_rate,
            "f1": f1,
        }
    
    def evaluate_by_category(
        self,
        thresholds: Dict[str, float],
        weights: Dict[str, float],
        block_assist: float,
        aggregation: str = "fisher_weighted",
        fast_path_assist: float = 0.85,
        assist_strength: float = 0.9,
        event_horizon: float = 1.0
    ) -> Dict[str, Dict]:
        """Evaluate and return per-category metrics."""
        category_results = {}
        
        for pid in self.prompt_ids:
            pred = self.classify_prompt(
                pid, thresholds, weights, block_assist, aggregation,
                fast_path_assist, assist_strength, event_horizon
            )
            label = self.labels[pid]
            cat = self.categories[pid]
            
            if cat not in category_results:
                category_results[cat] = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
            
            if label == "harmful":
                if pred == "harmful":
                    category_results[cat]["tp"] += 1
                else:
                    category_results[cat]["fn"] += 1
            else:
                if pred == "benign":
                    category_results[cat]["tn"] += 1
                else:
                    category_results[cat]["fp"] += 1
        
        # Compute metrics per category
        for cat, r in category_results.items():
            r["recall"] = r["tp"] / (r["tp"] + r["fn"]) if (r["tp"] + r["fn"]) > 0 else None
            r["fp_rate"] = r["fp"] / (r["fp"] + r["tn"]) if (r["fp"] + r["tn"]) > 0 else None
        
        return category_results


class CMAESCalibrator:
    """CMA-ES based calibration for VecP thresholds and weights."""
    
    def __init__(self, classifier: CachedClassifier, config: CalibrationConfig):
        self.classifier = classifier
        self.config = config
        self.gate_names = classifier.gate_names
        
        # Build parameter vector structure
        self.param_structure = []
        self.param_bounds_low = []
        self.param_bounds_high = []
        
        if config.optimize_thresholds:
            for gate in self.gate_names:
                self.param_structure.append(("threshold", gate))
                self.param_bounds_low.append(-0.2)
                self.param_bounds_high.append(0.3)
        
        if config.optimize_weights:
            for gate in self.gate_names:
                self.param_structure.append(("weight", gate))
                self.param_bounds_low.append(0.1)
                self.param_bounds_high.append(3.0)
        
        if config.optimize_block_assist:
            self.param_structure.append(("block_assist", None))
            self.param_bounds_low.append(0.05)
            self.param_bounds_high.append(0.25)
        
        if config.optimize_output_gate:
            # fast_path_assist: threshold for "clearly benign"
            self.param_structure.append(("fast_path_assist", None))
            self.param_bounds_low.append(0.70)
            self.param_bounds_high.append(0.95)
            
            # assist_strength: how much benign signal helps in middle zone
            self.param_structure.append(("assist_strength", None))
            self.param_bounds_low.append(0.3)
            self.param_bounds_high.append(1.0)
            
            # event_horizon: strictness of output gate (1.0 = strict, 0.5 = lenient)
            self.param_structure.append(("event_horizon", None))
            self.param_bounds_low.append(0.5)
            self.param_bounds_high.append(1.0)
        
        self.n_params = len(self.param_structure)
        print(f"  Optimizing {self.n_params} parameters")
        
        # Track best solution
        self.best_result = None
        self.best_params = None
        self.best_score = float('inf')
        self.history = []
    
    def params_to_dict(self, x: np.ndarray) -> Dict:
        """Convert parameter vector to full config dict."""
        thresholds = {}
        weights = {}
        block_assist = 0.15  # Default
        fast_path_assist = 0.85  # Default
        assist_strength = 0.9  # Default
        event_horizon = 1.0  # Default
        
        for i, (ptype, gate) in enumerate(self.param_structure):
            if ptype == "threshold":
                thresholds[gate] = x[i]
            elif ptype == "weight":
                weights[gate] = x[i]
            elif ptype == "block_assist":
                block_assist = x[i]
            elif ptype == "fast_path_assist":
                fast_path_assist = x[i]
            elif ptype == "assist_strength":
                assist_strength = x[i]
            elif ptype == "event_horizon":
                event_horizon = x[i]
        
        # Fill in defaults
        for gate in self.gate_names:
            if gate not in thresholds:
                thresholds[gate] = 0.13  # Default threshold
            if gate not in weights:
                weights[gate] = self.classifier.fisher_scores.get(gate, 1.0)
        
        return {
            "thresholds": thresholds,
            "weights": weights,
            "block_assist": block_assist,
            "fast_path_assist": fast_path_assist,
            "assist_strength": assist_strength,
            "event_horizon": event_horizon,
        }
    
    def objective(self, x: np.ndarray) -> float:
        """
        Objective function for CMA-ES (minimization).
        
        Returns a score to MINIMIZE. Lower is better.
        """
        params = self.params_to_dict(x)
        
        result = self.classifier.evaluate(
            params["thresholds"], 
            params["weights"], 
            params["block_assist"], 
            self.config.aggregation,
            params["fast_path_assist"],
            params["assist_strength"],
            params["event_horizon"]
        )
        
        recall = result["recall"]
        fp_rate = result["fp_rate"]
        f1 = result["f1"]
        precision = result["precision"]
        
        # Hard constraints - heavy penalty if violated
        penalty = 0
        if recall < self.config.min_recall:
            penalty += 10 * (self.config.min_recall - recall)
        
        # Soft constraint on FP rate
        if fp_rate > self.config.max_fp_rate:
            penalty += 5 * (fp_rate - self.config.max_fp_rate)
        
        # Objective based on mode
        if self.config.mode == "recall":
            # Maximize recall (minimize negative recall)
            score = -recall + 0.5 * fp_rate + penalty
        elif self.config.mode == "precision":
            # Maximize precision
            score = -precision + penalty
        elif self.config.mode == "f1":
            # Maximize F1
            score = -f1 + penalty
        elif self.config.mode == "balanced":
            # Balance recall and FP rate
            # Target: recall >= target_recall, fp_rate <= max_fp_rate
            recall_gap = max(0, self.config.target_recall - recall)
            fp_gap = max(0, fp_rate - self.config.max_fp_rate)
            score = recall_gap + fp_gap + penalty
        else:
            score = -f1 + penalty
        
        # Track best
        if score < self.best_score:
            self.best_score = score
            self.best_params = x.copy()
            self.best_result = result
            self.best_config = params
        
        return score
    
    def get_initial_guess(self) -> np.ndarray:
        """Generate initial parameter guess."""
        x0 = []
        
        for ptype, gate in self.param_structure:
            if ptype == "threshold":
                # Start with original calibrated threshold or 0.13
                orig = self.classifier.gate_info.get(gate, {}).get("original_threshold", 0.13)
                x0.append(np.clip(orig, -0.1, 0.25))
            elif ptype == "weight":
                # Start with SNR score (better than Fisher for signal quality)
                snr = self.classifier.snr_scores.get(gate, 1.0)
                x0.append(np.clip(snr, 0.5, 2.5))  # SNR typically 1-3 range
            elif ptype == "block_assist":
                x0.append(0.15)
            elif ptype == "fast_path_assist":
                x0.append(0.85)
            elif ptype == "assist_strength":
                x0.append(0.9)
            elif ptype == "event_horizon":
                x0.append(1.0)
        
        return np.array(x0)
    
    def optimize(self) -> Dict:
        """Run CMA-ES optimization."""
        if not HAS_CMA:
            raise ImportError("CMA-ES not available. Install with: pip install cma")
        
        x0 = self.get_initial_guess()
        
        # CMA-ES options
        opts = {
            'bounds': [self.param_bounds_low, self.param_bounds_high],
            'maxiter': self.config.max_iterations,
            'popsize': self.config.population_size,
            'verbose': -9,  # Quiet
        }
        
        print(f"\n  Starting CMA-ES optimization...")
        print(f"  Target: recall >= {self.config.target_recall:.1%}, FP <= {self.config.max_fp_rate:.1%}")
        print(f"  Mode: {self.config.mode}")
        
        es = cma.CMAEvolutionStrategy(x0, self.config.sigma0, opts)
        
        iteration = 0
        while not es.stop():
            solutions = es.ask()
            fitness = [self.objective(x) for x in solutions]
            es.tell(solutions, fitness)
            
            iteration += 1
            if iteration % 50 == 0:
                result = self.best_result
                print(f"  [Iter {iteration}] Recall: {result['recall']:.1%}, FP: {result['fp_rate']:.1%}, F1: {result['f1']:.3f}")
        
        # Final result
        return {
            "thresholds": self.best_config["thresholds"],
            "weights": self.best_config["weights"],
            "block_assist": self.best_config["block_assist"],
            "fast_path_assist": self.best_config["fast_path_assist"],
            "assist_strength": self.best_config["assist_strength"],
            "event_horizon": self.best_config["event_horizon"],
            "metrics": self.best_result,
            "iterations": iteration,
        }


def run_grid_search(
    classifier: CachedClassifier,
    base_config: Dict,
    aggregation: str = "fisher_weighted"
) -> Dict:
    """
    Grid search refinement around a base solution.
    Tests small perturbations to find local optimum.
    """
    print(f"\n  Running grid search refinement...")
    
    best_result = classifier.evaluate(
        base_config["thresholds"], 
        base_config["weights"], 
        base_config["block_assist"], 
        aggregation,
        base_config["fast_path_assist"],
        base_config["assist_strength"],
        base_config["event_horizon"]
    )
    best_config = base_config.copy()
    best_f1 = best_result["f1"]
    
    # Grid over block_assist
    for ba in np.arange(0.08, 0.20, 0.01):
        config = base_config.copy()
        config["block_assist"] = ba
        result = classifier.evaluate(
            config["thresholds"], config["weights"], config["block_assist"], aggregation,
            config["fast_path_assist"], config["assist_strength"], config["event_horizon"]
        )
        if result["f1"] > best_f1 and result["recall"] >= 0.88:
            best_f1 = result["f1"]
            best_config = config.copy()
            best_result = result
    
    # Grid over assist_strength
    for a_s in np.arange(0.5, 1.0, 0.05):
        config = best_config.copy()
        config["assist_strength"] = a_s
        result = classifier.evaluate(
            config["thresholds"], config["weights"], config["block_assist"], aggregation,
            config["fast_path_assist"], config["assist_strength"], config["event_horizon"]
        )
        if result["f1"] > best_f1 and result["recall"] >= 0.88:
            best_f1 = result["f1"]
            best_config = config.copy()
            best_result = result
    
    # Grid over event_horizon
    for eh in np.arange(0.7, 1.05, 0.05):
        config = best_config.copy()
        config["event_horizon"] = eh
        result = classifier.evaluate(
            config["thresholds"], config["weights"], config["block_assist"], aggregation,
            config["fast_path_assist"], config["assist_strength"], config["event_horizon"]
        )
        if result["f1"] > best_f1 and result["recall"] >= 0.88:
            best_f1 = result["f1"]
            best_config = config.copy()
            best_result = result
    
    best_config["metrics"] = best_result
    return best_config


def main():
    parser = argparse.ArgumentParser(description="CMA-ES calibration for VecP")
    parser.add_argument("--cached", required=True, help="Cached scores JSON file")
    parser.add_argument("--output", default="calibration_result.json")
    
    # Optimization targets
    parser.add_argument("--target-recall", type=float, default=0.92)
    parser.add_argument("--max-fp", type=float, default=0.12)
    parser.add_argument("--min-recall", type=float, default=0.85)
    parser.add_argument("--mode", choices=["recall", "precision", "f1", "balanced"], default="balanced")
    
    # What to optimize
    parser.add_argument("--optimize-thresholds", action="store_true", default=True)
    parser.add_argument("--optimize-weights", action="store_true", default=True)
    parser.add_argument("--no-optimize-weights", action="store_false", dest="optimize_weights")
    parser.add_argument("--optimize-block-assist", action="store_true", default=True)
    
    # Aggregation
    parser.add_argument("--aggregation", default="snr_weighted",
                        choices=["max_mean", "fisher_weighted", "snr_weighted", "weighted_sum", "max"])
    
    # CMA-ES params
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--population-size", type=int, default=20)
    
    # Grid search
    parser.add_argument("--no-grid-refine", dest="grid_refine",
                        action="store_false", default=True,
                        help="Skip grid search refinement after CMA-ES (default: refine)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  VecP CMA-ES Calibration")
    print("=" * 70)
    
    # Load cached scores
    print(f"\nLoading cached scores: {args.cached}")
    with open(args.cached, 'r') as f:
        cached_data = json.load(f)
    
    classifier = CachedClassifier(cached_data)
    
    # Config
    config = CalibrationConfig(
        target_recall=args.target_recall,
        max_fp_rate=args.max_fp,
        min_recall=args.min_recall,
        mode=args.mode,
        optimize_thresholds=args.optimize_thresholds,
        optimize_weights=args.optimize_weights,
        optimize_block_assist=args.optimize_block_assist,
        aggregation=args.aggregation,
        max_iterations=args.max_iterations,
        population_size=args.population_size,
    )
    
    # Run CMA-ES
    calibrator = CMAESCalibrator(classifier, config)
    
    if HAS_CMA:
        result = calibrator.optimize()
    else:
        print("  CMA-ES not available, using default parameters")
        result = {
            "thresholds": {g: 0.13 for g in classifier.gate_names},
            "weights": classifier.fisher_scores,
            "block_assist": 0.15,
            "fast_path_assist": 0.85,
            "assist_strength": 0.9,
            "event_horizon": 1.0,
            "metrics": classifier.evaluate(
                {g: 0.13 for g in classifier.gate_names},
                classifier.fisher_scores,
                0.15,
                args.aggregation
            )
        }
    
    print(f"\n  CMA-ES Result:")
    print(f"    Recall: {result['metrics']['recall']:.1%}")
    print(f"    FP Rate: {result['metrics']['fp_rate']:.1%}")
    print(f"    F1: {result['metrics']['f1']:.3f}")
    
    # Grid search refinement
    if args.grid_refine:
        refined = run_grid_search(
            classifier,
            result,
            args.aggregation
        )
        
        print(f"\n  After Grid Refinement:")
        print(f"    Recall: {refined['metrics']['recall']:.1%}")
        print(f"    FP Rate: {refined['metrics']['fp_rate']:.1%}")
        print(f"    F1: {refined['metrics']['f1']:.3f}")
        
        result = refined
    
    # Per-category breakdown
    print(f"\n  Category Breakdown:")
    cat_results = classifier.evaluate_by_category(
        result["thresholds"],
        result["weights"],
        result["block_assist"],
        args.aggregation,
        result.get("fast_path_assist", 0.85),
        result.get("assist_strength", 0.9),
        result.get("event_horizon", 1.0)
    )
    
    print(f"  {'Category':<10} {'Recall':>8} {'FP Rate':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*8}")
    for cat, m in sorted(cat_results.items(), key=lambda x: -(x[1].get('fp_rate') or 0)):
        recall_str = f"{m['recall']:.1%}" if m['recall'] is not None else "N/A"
        fp_str = f"{m['fp_rate']:.1%}" if m['fp_rate'] is not None else "N/A"
        print(f"  {cat:<10} {recall_str:>8} {fp_str:>8}")
    
    # Save result
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "target_recall": args.target_recall,
            "max_fp": args.max_fp,
            "mode": args.mode,
            "aggregation": args.aggregation,
        },
        "result": {
            "thresholds": result["thresholds"],
            "weights": result["weights"],
            "block_assist": result["block_assist"],
            "fast_path_assist": result.get("fast_path_assist", 0.85),
            "assist_strength": result.get("assist_strength", 0.9),
            "event_horizon": result.get("event_horizon", 1.0),
        },
        "metrics": result["metrics"],
        "category_metrics": cat_results,
    }
    
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n  Saved to: {args.output}")
    
    # Print config for benchmark runner
    print(f"\n" + "=" * 70)
    print("  USE THESE SETTINGS IN BENCHMARK RUNNER:")
    print("=" * 70)
    print(f"  --block-assist {result['block_assist']:.3f}")
    print(f"  --fast-path-assist {result.get('fast_path_assist', 0.85):.3f}")
    print(f"  --assist-strength {result.get('assist_strength', 0.9):.3f}")
    print(f"  --event-horizon {result.get('event_horizon', 1.0):.3f}")
    print(f"  --aggregation-mode {args.aggregation}")
    print(f"\n  Per-gate thresholds (update in gate files or use calibration loader):")
    for gate, thresh in sorted(result["thresholds"].items()):
        print(f"    {gate}: {thresh:.4f}")
    
    print(f"\n  Weights:")
    for gate, weight in sorted(result["weights"].items(), key=lambda x: -x[1]):
        print(f"    {gate}: {weight:.3f}")


if __name__ == "__main__":
    main()
