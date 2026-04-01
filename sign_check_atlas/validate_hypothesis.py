#!/usr/bin/env python3
"""
Sign-Check Atlas: Phase 1 - Hypothesis Validation
===================================================
Validates that a single energy axis (safe→harm direction) with a sign check
captures the majority of Atlas V2's detection capability.

Core idea:
    energy_axis = normalize(harm_centroid - safe_centroid)
    energy = dot(activation, energy_axis)
    energy > 0 → harmful direction → FLAG
    energy < 0 → safe direction → PASS

This script:
    1. Loads a model and extracts activations for gauntlet data
    2. Computes safe/harm centroids and energy axis
    3. Projects all activations onto the energy axis
    4. Evaluates sign-check classification accuracy
    5. Saves the energy axis and results

Usage:
    python sign_check_atlas/validate_hypothesis.py \
        --model Qwen/Qwen3-4B \
        --gauntlet gauntlet_v3_corrected.txt \
        --output-dir sign_check_atlas/results

    # CPU-only mode (slower but works without GPU)
    python sign_check_atlas/validate_hypothesis.py \
        --model Qwen/Qwen3-4B \
        --gauntlet gauntlet_v3_corrected.txt \
        --device cpu

    # Specify extraction layers
    python sign_check_atlas/validate_hypothesis.py \
        --model Qwen/Qwen3-4B \
        --gauntlet gauntlet_v3_corrected.txt \
        --layers 14 18 22 27

Patent Pending: USPTO 63/931,565
Copyright (c) 2025-2026 David Cappelli / VecP Labs
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# DATA LOADING
# =============================================================================

def load_gauntlet(filepath: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Load gauntlet file organized by category.

    Returns dict mapping category prefix (e.g. 'H_RED', 'B_GEN') to list of
    (prompt, code) tuples.
    """
    categories = {}

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '|' not in line:
                continue

            parts = line.split('|', 1)
            code = parts[0].strip()
            prompt = parts[1].strip()

            cat_parts = code.split('_')
            prefix = f"{cat_parts[0]}_{cat_parts[1]}" if len(cat_parts) >= 2 else code

            if prefix not in categories:
                categories[prefix] = []
            categories[prefix].append((prompt, code))

    return categories


def split_by_label(categories: Dict[str, List[Tuple[str, str]]]) -> Tuple[List[str], List[str], List[str]]:
    """Split gauntlet into harmful and benign prompt lists, plus category labels."""
    harmful_prompts = []
    benign_prompts = []
    harmful_cats = []

    for cat, entries in categories.items():
        is_harmful = cat.startswith('H_')
        for prompt, code in entries:
            if is_harmful:
                harmful_prompts.append(prompt)
                harmful_cats.append(cat)
            else:
                benign_prompts.append(prompt)

    return harmful_prompts, benign_prompts, harmful_cats


# =============================================================================
# ACTIVATION EXTRACTION
# =============================================================================

class ActivationExtractor:
    """
    Extracts activations from specified model layers.

    Hooks into model layers to capture hidden states during forward pass.
    Designed for the sign-check energy axis computation.
    """

    def __init__(self, model, tokenizer, layers: List[int], component: str = "residual", device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.layers = layers
        self.component = component
        self.captured = {}
        self.handles = []

        self._setup_hooks()

    def _setup_hooks(self):
        """Register forward hooks on specified layers."""
        for layer_idx in self.layers:
            layer_module = self.model.model.layers[layer_idx]

            if self.component == "residual":
                module = layer_module
            elif self.component == "mlp.down_proj":
                module = layer_module.mlp.down_proj
            else:
                parts = self.component.split('.')
                module = layer_module
                for part in parts:
                    module = getattr(module, part)

            def make_hook(idx):
                def hook(mod, inp, out):
                    self.captured[idx] = out[0].detach() if isinstance(out, tuple) else out.detach()
                return hook

            handle = module.register_forward_hook(make_hook(layer_idx))
            self.handles.append(handle)

        print(f"  [Extractor] Hooked {len(self.handles)} layers: {self.layers} ({self.component})")

    def cleanup(self):
        """Remove all hooks."""
        for h in self.handles:
            h.remove()
        self.handles = []

    def extract_batch(self, texts: List[str], batch_size: int = 8) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Extract activations for a batch of texts at all hooked layers.

        Returns: {layer_idx: {text: embedding_tensor}}
        """
        results = {l: {} for l in self.layers}
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(texts))
            batch = texts[start:end]

            inputs = self.tokenizer(
                batch, return_tensors="pt",
                truncation=True, max_length=512, padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            self.captured = {}

            with torch.no_grad():
                self.model(**inputs)

            seq_lengths = inputs['attention_mask'].sum(dim=1) - 1

            for b, text in enumerate(batch):
                last_idx = seq_lengths[b].item()

                for layer_idx in self.layers:
                    if layer_idx in self.captured:
                        act = self.captured[layer_idx]
                        if act.dim() == 3:
                            emb = act[b, int(last_idx), :].float()
                        else:
                            emb = act[-1, :].float()
                        results[layer_idx][text] = F.normalize(emb, dim=-1).cpu()

            print(f"\r  Extracting: {batch_idx + 1}/{total_batches}", end="", flush=True)

        print()
        return results


# =============================================================================
# ENERGY AXIS COMPUTATION
# =============================================================================

def compute_energy_axis(
    embeddings: Dict[int, Dict[str, torch.Tensor]],
    harmful_prompts: List[str],
    benign_prompts: List[str],
    layers: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Compute the energy axis from harmful and benign centroids.

    The energy axis is the normalized direction from safe centroid to harm centroid.
    Positive projection = moving toward harm.
    Negative projection = moving toward safe.

    Can operate on single or multiple layers (averaged).

    Returns:
        energy_axis: (dim,) normalized direction vector
        safe_centroid: (dim,) safe class centroid
        harm_centroid: (dim,) harm class centroid
        stats: dict with computation statistics
    """
    # Collect embeddings across specified layers
    harm_embs = []
    safe_embs = []

    for prompt in harmful_prompts:
        layer_embs = []
        for layer_idx in layers:
            if prompt in embeddings[layer_idx]:
                layer_embs.append(embeddings[layer_idx][prompt])
        if layer_embs:
            avg_emb = torch.stack(layer_embs).mean(dim=0)
            harm_embs.append(avg_emb)

    for prompt in benign_prompts:
        layer_embs = []
        for layer_idx in layers:
            if prompt in embeddings[layer_idx]:
                layer_embs.append(embeddings[layer_idx][prompt])
        if layer_embs:
            avg_emb = torch.stack(layer_embs).mean(dim=0)
            safe_embs.append(avg_emb)

    print(f"  Collected {len(harm_embs)} harmful, {len(safe_embs)} benign embeddings")

    harm_stack = torch.stack(harm_embs)
    safe_stack = torch.stack(safe_embs)

    harm_centroid = harm_stack.mean(dim=0).numpy()
    safe_centroid = safe_stack.mean(dim=0).numpy()

    # Energy axis: direction from safe to harm
    energy_axis = harm_centroid - safe_centroid
    axis_norm = np.linalg.norm(energy_axis)
    energy_axis = energy_axis / axis_norm

    # Compute centroid separation
    centroid_distance = float(axis_norm)
    cosine_sim = float(np.dot(
        harm_centroid / np.linalg.norm(harm_centroid),
        safe_centroid / np.linalg.norm(safe_centroid)
    ))

    stats = {
        "n_harmful": len(harm_embs),
        "n_benign": len(safe_embs),
        "centroid_distance": centroid_distance,
        "centroid_cosine_similarity": cosine_sim,
        "axis_dim": len(energy_axis),
        "layers_used": layers,
    }

    print(f"  Centroid distance: {centroid_distance:.4f}")
    print(f"  Centroid cosine similarity: {cosine_sim:.4f}")

    return energy_axis, safe_centroid, harm_centroid, stats


# =============================================================================
# SIGN-CHECK EVALUATION
# =============================================================================

def evaluate_sign_check(
    embeddings: Dict[int, Dict[str, torch.Tensor]],
    energy_axis: np.ndarray,
    harmful_prompts: List[str],
    benign_prompts: List[str],
    layers: List[int],
    threshold: float = 0.0,
) -> Dict:
    """
    Evaluate sign-check classification using the energy axis.

    Projects all activations onto the energy axis and classifies based on sign
    (or threshold).

    Returns comprehensive metrics dict.
    """
    energy_axis_t = torch.tensor(energy_axis, dtype=torch.float32)

    # Compute energies for all prompts
    harm_energies = []
    safe_energies = []

    for prompt in harmful_prompts:
        layer_embs = []
        for layer_idx in layers:
            if prompt in embeddings[layer_idx]:
                layer_embs.append(embeddings[layer_idx][prompt])
        if layer_embs:
            avg_emb = torch.stack(layer_embs).mean(dim=0)
            energy = torch.dot(avg_emb, energy_axis_t).item()
            harm_energies.append(energy)

    for prompt in benign_prompts:
        layer_embs = []
        for layer_idx in layers:
            if prompt in embeddings[layer_idx]:
                layer_embs.append(embeddings[layer_idx][prompt])
        if layer_embs:
            avg_emb = torch.stack(layer_embs).mean(dim=0)
            energy = torch.dot(avg_emb, energy_axis_t).item()
            safe_energies.append(energy)

    harm_energies = np.array(harm_energies)
    safe_energies = np.array(safe_energies)
    all_energies = np.concatenate([safe_energies, harm_energies])
    all_labels = np.array([0] * len(safe_energies) + [1] * len(harm_energies))

    # Sign-check classification
    predictions = (all_energies > threshold).astype(int)

    tp = int(((predictions == 1) & (all_labels == 1)).sum())
    fp = int(((predictions == 1) & (all_labels == 0)).sum())
    tn = int(((predictions == 0) & (all_labels == 0)).sum())
    fn = int(((predictions == 0) & (all_labels == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(all_labels) if len(all_labels) > 0 else 0.0
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Sign distribution analysis
    safe_negative_pct = float(np.sum(safe_energies < threshold) / len(safe_energies))
    harm_positive_pct = float(np.sum(harm_energies > threshold) / len(harm_energies))

    # Energy distribution stats
    metrics = {
        "threshold": threshold,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "fp_rate": fp_rate,
        "safe_negative_pct": safe_negative_pct,
        "harm_positive_pct": harm_positive_pct,
        "harm_energy_mean": float(harm_energies.mean()),
        "harm_energy_std": float(harm_energies.std()),
        "harm_energy_min": float(harm_energies.min()),
        "harm_energy_max": float(harm_energies.max()),
        "safe_energy_mean": float(safe_energies.mean()),
        "safe_energy_std": float(safe_energies.std()),
        "safe_energy_min": float(safe_energies.min()),
        "safe_energy_max": float(safe_energies.max()),
        "separation": float(harm_energies.mean() - safe_energies.mean()),
    }

    return metrics, harm_energies, safe_energies


def per_layer_analysis(
    embeddings: Dict[int, Dict[str, torch.Tensor]],
    harmful_prompts: List[str],
    benign_prompts: List[str],
    layers: List[int],
) -> List[Dict]:
    """
    Analyze sign-check performance at each individual layer.

    This helps identify which layers have the cleanest energy separation,
    informing the optimal extraction layer for GGUF embedding.
    """
    results = []

    for layer_idx in layers:
        # Compute energy axis for this single layer
        axis, safe_c, harm_c, stats = compute_energy_axis(
            embeddings, harmful_prompts, benign_prompts, [layer_idx]
        )

        # Evaluate
        metrics, _, _ = evaluate_sign_check(
            embeddings, axis, harmful_prompts, benign_prompts, [layer_idx]
        )

        results.append({
            "layer": layer_idx,
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "accuracy": metrics["accuracy"],
            "fp_rate": metrics["fp_rate"],
            "separation": metrics["separation"],
            "centroid_distance": stats["centroid_distance"],
            "safe_negative_pct": metrics["safe_negative_pct"],
            "harm_positive_pct": metrics["harm_positive_pct"],
        })

    results.sort(key=lambda x: x["f1"], reverse=True)
    return results


# =============================================================================
# MULTI-COMPONENT ANALYSIS
# =============================================================================

def multi_component_sweep(
    model, tokenizer, harmful_prompts: List[str], benign_prompts: List[str],
    layers: List[int], device: str = "cuda", batch_size: int = 8
) -> List[Dict]:
    """
    Sweep across components (residual, mlp.down_proj, self_attn.o_proj) at
    each layer to find the best extraction point for the energy axis.
    """
    components = ["residual", "mlp.down_proj", "self_attn.o_proj"]
    all_results = []
    all_prompts = harmful_prompts + benign_prompts

    for component in components:
        print(f"\n  Component: {component}")
        extractor = ActivationExtractor(model, tokenizer, layers, component, device)

        try:
            embeddings = extractor.extract_batch(all_prompts, batch_size)
        finally:
            extractor.cleanup()

        for layer_idx in layers:
            axis, safe_c, harm_c, stats = compute_energy_axis(
                embeddings, harmful_prompts, benign_prompts, [layer_idx]
            )
            metrics, _, _ = evaluate_sign_check(
                embeddings, axis, harmful_prompts, benign_prompts, [layer_idx]
            )

            all_results.append({
                "layer": layer_idx,
                "component": component,
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "accuracy": metrics["accuracy"],
                "separation": metrics["separation"],
                "centroid_distance": stats["centroid_distance"],
            })

    all_results.sort(key=lambda x: x["f1"], reverse=True)
    return all_results


# =============================================================================
# MAIN VALIDATION
# =============================================================================

def run_validation(args):
    """Run the full Phase 1 validation pipeline."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 70)
    print("SIGN-CHECK ATLAS: Phase 1 Hypothesis Validation")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Gauntlet: {args.gauntlet}")
    print(f"Device: {args.device}")
    print(f"Layers: {args.layers}")
    print(f"Component: {args.component}")
    print()

    # --- Load gauntlet data ---
    print("[1/6] Loading gauntlet data...")
    categories = load_gauntlet(args.gauntlet)
    harmful_prompts, benign_prompts, harmful_cats = split_by_label(categories)
    print(f"  Harmful: {len(harmful_prompts)} prompts")
    print(f"  Benign:  {len(benign_prompts)} prompts")
    print(f"  Categories: {sorted(categories.keys())}")

    # --- Load model ---
    print(f"\n[2/6] Loading model: {args.model}")
    t0 = time.time()
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
    hidden_dim = model.config.hidden_size
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  Layers: {n_layers}, Hidden dim: {hidden_dim}")

    # Validate layer indices
    layers = args.layers
    if not layers:
        # Default: 40-70% depth range
        start = int(n_layers * 0.4)
        end = int(n_layers * 0.7)
        layers = list(range(start, end + 1))
        print(f"  Auto-selected layers (40-70% depth): {layers}")

    # --- Extract activations ---
    print(f"\n[3/6] Extracting activations (component: {args.component})...")
    t0 = time.time()
    all_prompts = harmful_prompts + benign_prompts
    extractor = ActivationExtractor(model, tokenizer, layers, args.component, args.device)

    try:
        embeddings = extractor.extract_batch(all_prompts, args.batch_size)
    finally:
        extractor.cleanup()
    print(f"  Extracted in {time.time() - t0:.1f}s")

    # --- Compute energy axis (multi-layer averaged) ---
    print(f"\n[4/6] Computing energy axis (layers: {layers})...")
    energy_axis, safe_centroid, harm_centroid, axis_stats = compute_energy_axis(
        embeddings, harmful_prompts, benign_prompts, layers
    )

    # --- Evaluate sign-check at threshold=0 ---
    print(f"\n[5/6] Evaluating sign-check classifier (threshold=0)...")
    metrics, harm_energies, safe_energies = evaluate_sign_check(
        embeddings, energy_axis, harmful_prompts, benign_prompts, layers, threshold=0.0
    )

    print(f"\n  {'='*50}")
    print(f"  SIGN-CHECK RESULTS (threshold=0)")
    print(f"  {'='*50}")
    print(f"  Precision:  {metrics['precision']:.3f}")
    print(f"  Recall:     {metrics['recall']:.3f}")
    print(f"  F1:         {metrics['f1']:.3f}")
    print(f"  Accuracy:   {metrics['accuracy']:.3f}")
    print(f"  FP Rate:    {metrics['fp_rate']:.3f}")
    print(f"  {'='*50}")
    print(f"  Safe with negative energy:    {metrics['safe_negative_pct']:.1%}")
    print(f"  Harmful with positive energy: {metrics['harm_positive_pct']:.1%}")
    print(f"  {'='*50}")
    print(f"  Harm energy: mean={metrics['harm_energy_mean']:.4f} std={metrics['harm_energy_std']:.4f}")
    print(f"  Safe energy: mean={metrics['safe_energy_mean']:.4f} std={metrics['safe_energy_std']:.4f}")
    print(f"  Separation:  {metrics['separation']:.4f}")

    # Assess viability
    if metrics['f1'] > 0.85:
        verdict = "STRONG - Proceed to Phase 2 (category analysis)"
    elif metrics['f1'] > 0.75:
        verdict = "VIABLE - Usable as Tier 1 filter with full Atlas as Tier 2"
    else:
        verdict = "NEEDS WORK - Hypothesis needs refinement"
    print(f"\n  VERDICT: {verdict}")

    # --- Per-layer analysis ---
    print(f"\n[6/6] Per-layer analysis...")
    layer_results = per_layer_analysis(embeddings, harmful_prompts, benign_prompts, layers)

    print(f"\n  {'Layer':<8} {'F1':<8} {'Prec':<8} {'Rec':<8} {'Sep':<10} {'Dist':<10}")
    print(f"  {'-'*52}")
    for r in layer_results[:10]:
        print(f"  L{r['layer']:<6} {r['f1']:<8.3f} {r['precision']:<8.3f} {r['recall']:<8.3f} "
              f"{r['separation']:<10.4f} {r['centroid_distance']:<10.4f}")

    best_layer = layer_results[0]
    print(f"\n  Best single layer: L{best_layer['layer']} (F1={best_layer['f1']:.3f})")

    # --- Multi-component sweep (optional) ---
    component_results = None
    if args.sweep_components:
        print(f"\n[BONUS] Multi-component sweep...")
        component_results = multi_component_sweep(
            model, tokenizer, harmful_prompts, benign_prompts,
            layers, args.device, args.batch_size
        )
        print(f"\n  {'Layer':<8} {'Component':<20} {'F1':<8} {'Prec':<8} {'Rec':<8}")
        print(f"  {'-'*52}")
        for r in component_results[:10]:
            print(f"  L{r['layer']:<6} {r['component']:<20} {r['f1']:<8.3f} "
                  f"{r['precision']:<8.3f} {r['recall']:<8.3f}")

    # --- Save results ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save energy axis
    axis_path = output_dir / "energy_axis.npy"
    np.save(str(axis_path), energy_axis)
    print(f"\n  Saved energy axis: {axis_path}")

    # Save centroids
    np.save(str(output_dir / "safe_centroid.npy"), safe_centroid)
    np.save(str(output_dir / "harm_centroid.npy"), harm_centroid)

    # Save energy distributions for visualization
    np.save(str(output_dir / "harm_energies.npy"), harm_energies)
    np.save(str(output_dir / "safe_energies.npy"), safe_energies)

    # Save results JSON
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "gauntlet": args.gauntlet,
        "component": args.component,
        "layers": layers,
        "axis_stats": axis_stats,
        "sign_check_metrics": metrics,
        "per_layer_results": layer_results,
        "verdict": verdict,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
    }

    if component_results:
        results["component_sweep"] = component_results

    results_path = output_dir / "phase1_validation.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved results: {results_path}")

    # Generate markdown report
    report = generate_report(results, metrics, layer_results, axis_stats)
    report_path = output_dir / "phase1_validation.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Saved report: {report_path}")

    print(f"\n{'='*70}")
    print(f"Phase 1 complete. F1={metrics['f1']:.3f}")
    print(f"{'='*70}")

    return results


def generate_report(results: Dict, metrics: Dict, layer_results: List[Dict], axis_stats: Dict) -> str:
    """Generate a markdown report for Phase 1 results."""
    lines = [
        "# Sign-Check Atlas: Phase 1 Validation Results",
        "",
        f"**Date:** {results['timestamp']}",
        f"**Model:** {results['model']}",
        f"**Gauntlet:** {results['gauntlet']}",
        f"**Component:** {results['component']}",
        f"**Layers:** {results['layers']}",
        "",
        "## Energy Axis Statistics",
        "",
        f"- **Hidden dimension:** {results['hidden_dim']}",
        f"- **Harmful samples:** {axis_stats['n_harmful']}",
        f"- **Benign samples:** {axis_stats['n_benign']}",
        f"- **Centroid distance:** {axis_stats['centroid_distance']:.4f}",
        f"- **Centroid cosine similarity:** {axis_stats['centroid_cosine_similarity']:.4f}",
        "",
        "## Sign-Check Results (threshold=0)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Precision | {metrics['precision']:.3f} |",
        f"| Recall | {metrics['recall']:.3f} |",
        f"| F1 | {metrics['f1']:.3f} |",
        f"| Accuracy | {metrics['accuracy']:.3f} |",
        f"| FP Rate | {metrics['fp_rate']:.3f} |",
        "",
        "### Energy Distribution",
        "",
        f"| Class | Mean | Std | Min | Max |",
        f"|-------|------|-----|-----|-----|",
        f"| Harmful | {metrics['harm_energy_mean']:.4f} | {metrics['harm_energy_std']:.4f} | {metrics['harm_energy_min']:.4f} | {metrics['harm_energy_max']:.4f} |",
        f"| Safe | {metrics['safe_energy_mean']:.4f} | {metrics['safe_energy_std']:.4f} | {metrics['safe_energy_min']:.4f} | {metrics['safe_energy_max']:.4f} |",
        "",
        f"- **Separation:** {metrics['separation']:.4f}",
        f"- **Safe with negative energy:** {metrics['safe_negative_pct']:.1%}",
        f"- **Harmful with positive energy:** {metrics['harm_positive_pct']:.1%}",
        "",
        f"## Verdict",
        "",
        f"**{results['verdict']}**",
        "",
        "## Per-Layer Results",
        "",
        "| Layer | F1 | Precision | Recall | Separation | Centroid Dist |",
        "|-------|----|-----------| -------|------------|---------------|",
    ]

    for r in layer_results:
        lines.append(
            f"| L{r['layer']} | {r['f1']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | "
            f"{r['separation']:.4f} | {r['centroid_distance']:.4f} |"
        )

    lines.extend([
        "",
        f"**Best single layer:** L{layer_results[0]['layer']} (F1={layer_results[0]['f1']:.3f})",
        "",
        "---",
        "*Generated by Sign-Check Atlas Phase 1 validation*",
    ])

    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sign-Check Atlas: Phase 1 Hypothesis Validation"
    )
    parser.add_argument("--model", type=str, required=True,
                       help="Model name or path (e.g., Qwen/Qwen3-4B)")
    parser.add_argument("--gauntlet", type=str, required=True,
                       help="Path to gauntlet data file")
    parser.add_argument("--output-dir", type=str, default="sign_check_atlas/results",
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda or cpu)")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                       help="Layer indices for extraction (default: 40-70%% depth)")
    parser.add_argument("--component", type=str, default="residual",
                       choices=["residual", "mlp.down_proj", "self_attn.o_proj"],
                       help="Model component to extract from")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for extraction")
    parser.add_argument("--sweep-components", action="store_true",
                       help="Also sweep all components (slower but comprehensive)")

    args = parser.parse_args()
    run_validation(args)


if __name__ == "__main__":
    main()
