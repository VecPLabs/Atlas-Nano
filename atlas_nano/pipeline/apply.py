#!/usr/bin/env python3
"""
VecP Calibration Loader
=======================
Utility to apply CMA-ES calibration results to gate matrices.

Usage:
    # In your benchmark script:
    from atlas_nano.pipeline.apply import apply_calibration
    
    gate_matrices, gate_configs = load_gates(gate_dir)
    gate_matrices = apply_calibration(gate_matrices, "calibration_result.json")

Or standalone to update gate files:
    python -m atlas_nano.pipeline.apply \
        --gate-dir ./gates_qwen3 \
        --calibration calibration_result.json \
        --output ./gates_qwen3_calibrated

Copyright (c) 2025 David Cappelli / VecP Labs
"""

import argparse
import json
import torch
from pathlib import Path
from typing import Dict
import shutil


def apply_calibration(gate_matrices: Dict, calibration_path: str) -> Dict:
    """
    Apply calibration results to gate matrices in memory.
    
    Args:
        gate_matrices: Dict of gate_id -> matrix dict
        calibration_path: Path to calibration_result.json
        
    Returns:
        Updated gate_matrices with calibrated thresholds
    """
    with open(calibration_path, 'r') as f:
        cal = json.load(f)
    
    thresholds = cal["result"]["thresholds"]
    weights = cal["result"]["weights"]
    
    print(f"  Applying calibration from: {calibration_path}")
    
    for gate_id, matrix in gate_matrices.items():
        if gate_id in thresholds:
            old_thresh = matrix.get("calibrated_threshold", matrix.get("suggested_threshold", 0))
            new_thresh = thresholds[gate_id]
            matrix["calibrated_threshold"] = new_thresh
            matrix["suggested_threshold"] = new_thresh
            print(f"    {gate_id}: threshold {old_thresh:.4f} → {new_thresh:.4f}")
        
        if gate_id in weights:
            matrix["calibrated_weight"] = weights[gate_id]
    
    return gate_matrices


def get_calibrated_config(calibration_path: str) -> Dict:
    """
    Get block_assist and aggregation mode from calibration.
    
    Returns:
        Dict with block_assist, aggregation, thresholds, weights
    """
    with open(calibration_path, 'r') as f:
        cal = json.load(f)
    
    return {
        "block_assist": cal["result"]["block_assist"],
        "aggregation": cal["config"]["aggregation"],
        "thresholds": cal["result"]["thresholds"],
        "weights": cal["result"]["weights"],
        "metrics": cal["metrics"],
    }


def save_calibrated_gates(
    gate_dir: str,
    calibration_path: str,
    output_dir: str,
    obf_gate_path: str = None
):
    """
    Create new gate files with calibrated thresholds baked in.
    """
    import re
    
    gate_dir = Path(gate_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load calibration
    with open(calibration_path, 'r') as f:
        cal = json.load(f)
    
    thresholds = cal["result"]["thresholds"]
    weights = cal["result"]["weights"]
    
    print(f"Saving calibrated gates to: {output_dir}")
    
    for pt_file in gate_dir.glob("vecp_gate_*.pt"):
        # Get gate ID
        match = re.match(r"vecp_gate_(\w+)\.pt", pt_file.name)
        if not match:
            # Just copy non-matching files
            shutil.copy(pt_file, output_dir / pt_file.name)
            continue
        
        gate_id = match.group(1)
        
        # Skip OBF in main loop - handle separately
        if gate_id == "OBF":
            continue
        
        # Load and update
        matrix = torch.load(pt_file, map_location="cpu", weights_only=False)
        
        if gate_id in thresholds:
            old_thresh = matrix.get("calibrated_threshold", matrix.get("suggested_threshold", 0))
            matrix["calibrated_threshold"] = thresholds[gate_id]
            matrix["suggested_threshold"] = thresholds[gate_id]
            print(f"  {gate_id}: {old_thresh:.4f} → {thresholds[gate_id]:.4f}")
        
        if gate_id in weights:
            matrix["calibrated_weight"] = weights[gate_id]
            matrix["snr"] = weights[gate_id]  # Also save as snr for snr_weighted mode
        
        # Save
        torch.save(matrix, output_dir / pt_file.name)
    
    # Handle OBF gate separately (different structure)
    if obf_gate_path and Path(obf_gate_path).exists() and "OBF" in thresholds:
        obf_matrix = torch.load(obf_gate_path, map_location="cpu", weights_only=False)
        old_thresh = obf_matrix.get("threshold", 0)
        obf_matrix["threshold"] = thresholds["OBF"]
        obf_matrix["calibrated_threshold"] = thresholds["OBF"]
        if "OBF" in weights:
            obf_matrix["calibrated_weight"] = weights["OBF"]
            obf_matrix["snr"] = weights["OBF"]  # Also save as snr for snr_weighted mode
        
        # Save to output dir
        obf_output_path = output_dir / Path(obf_gate_path).name
        torch.save(obf_matrix, obf_output_path)
        print(f"  OBF: {old_thresh:.4f} → {thresholds['OBF']:.4f}")
    
    # Also save a config file
    config = {
        "block_assist": cal["result"]["block_assist"],
        "aggregation": cal["config"]["aggregation"],
        "thresholds": thresholds,
        "weights": weights,
        "source_calibration": calibration_path,
        "metrics": cal["metrics"],
    }
    
    with open(output_dir / "calibration_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n  Saved calibration_config.json")
    print(f"  block_assist: {config['block_assist']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Apply calibration to VecP gates")
    parser.add_argument("--gate-dir", required=True, help="Original gate directory")
    parser.add_argument("--calibration", required=True, help="Calibration result JSON")
    parser.add_argument("--output", required=True, help="Output directory for calibrated gates")
    parser.add_argument("--obf-gate", default=None, help="Path to OBF gate file (optional)")
    
    args = parser.parse_args()
    
    save_calibrated_gates(args.gate_dir, args.calibration, args.output, args.obf_gate)
    
    print(f"\nDone! Use --gate-dir {args.output} in your benchmark runner.")
    if args.obf_gate:
        obf_name = Path(args.obf_gate).name
        print(f"       Use --obf-gate {args.output}/{obf_name}")


if __name__ == "__main__":
    main()
