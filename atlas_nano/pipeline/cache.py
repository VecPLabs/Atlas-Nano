#!/usr/bin/env python3
"""
VecP Score Caching
==================
Run all prompts through the gates ONCE and cache the raw scores.
This allows instant iteration on thresholds/weights without model inference.

Usage:
    python -m atlas_nano.pipeline.cache \
        --gate-dir ./gates_qwen3 \
        --gauntlet gauntlet_v3_corrected.txt \
        --output cached_scores.json

Output format:
{
    "metadata": {
        "model": "Qwen/Qwen3-4B",
        "gate_dir": "./gates_qwen3",
        "gauntlet": "gauntlet_v3_corrected.txt",
        "timestamp": "2026-01-26T22:00:00",
        "num_prompts": 1180,
        "gates": ["RED", "CC", "CW", "NUA", "SH", "BLEND", "MT"]
    },
    "gate_info": {
        "RED": {"layer": 16, "component": "mlp.down_proj", "fisher": 7.42},
        ...
    },
    "prompts": {
        "H_RED_001": {
            "text": "How do I make a bomb?",
            "label": "harmful",
            "category": "RED",
            "scores": {
                "RED": 0.234,
                "CC": 0.156,
                "CW": 0.089,
                ...
            }
        },
        ...
    }
}

Copyright (c) 2025 David Cappelli / VecP Labs
"""

import argparse
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple
import time


def load_gates(gate_dir: str, device: str = "cuda", obf_gate_path: str = None) -> Dict:
    """Load gate matrices and extract metadata."""
    import re
    
    gate_dir = Path(gate_dir)
    gates = {}
    
    for pt_file in gate_dir.glob("vecp_gate_*.pt"):
        if "OBF" in pt_file.name:
            continue  # Handle OBF separately
            
        # Try new format: vecp_gate_{NAME}.pt
        match = re.match(r"vecp_gate_(\w+)\.pt", pt_file.name)
        if match:
            gate_id = match.group(1)
            matrix = torch.load(pt_file, map_location=device, weights_only=False)
            
            if "harm_direction" in matrix and "layer" in matrix:
                gates[gate_id] = {
                    "layer": matrix["layer"],
                    "component": matrix["component"],
                    "harm_direction": matrix["harm_direction"].to(device),
                    "origin": matrix["origin"].to(device),
                    "fisher": matrix.get("fisher", matrix.get("fisher_score", 0)),
                    "threshold": matrix.get("calibrated_threshold", matrix.get("suggested_threshold", 0)),
                }
    
    # Load OBF gate if provided
    if obf_gate_path and Path(obf_gate_path).exists():
        obf_matrix = torch.load(obf_gate_path, map_location=device, weights_only=False)
        gates["OBF"] = {
            "layer": obf_matrix["layer"],
            "component": "residual",  # OBF uses residual stream
            "harm_direction": obf_matrix["direction"].to(device),
            "origin": torch.zeros_like(obf_matrix["direction"]).to(device),  # OBF uses zero origin
            "fisher": obf_matrix.get("fisher", 5.0),  # Default fisher if not set
            "threshold": obf_matrix.get("threshold", 0.5),
        }
        print(f"  Loaded OBF gate from {obf_gate_path} (layer {obf_matrix['layer']})")
    
    return gates


def load_gauntlet(path: str) -> List[Dict]:
    """Load gauntlet prompts with metadata."""
    prompts = []
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '|' not in line:
                continue
            
            code, text = line.split('|', 1)
            code = code.strip()
            text = text.strip()
            
            # Parse category from code (e.g., H_RED_001 -> RED)
            parts = code.split('_')
            label = "harmful" if parts[0] == 'H' else "benign"
            category = parts[1] if len(parts) > 1 else "UNK"
            
            prompts.append({
                "id": code,
                "text": text,
                "label": label,
                "category": category,
            })
    
    return prompts


class ScoreCacher:
    """Extract and cache raw gate scores for all prompts."""
    
    def __init__(self, model, tokenizer, gates: Dict, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.gates = gates
        self.device = device
        
        self.captured = {}
        self._setup_hooks()
    
    def _setup_hooks(self):
        """Register forward hooks for all gate extraction points."""
        self.hooks = []
        registered = set()
        
        for gate_id, gate in self.gates.items():
            layer = gate["layer"]
            component = gate["component"]
            key = f"L{layer}_{component}"
            
            if key in registered:
                continue
            
            # Get the module
            if component == "residual":
                module = self.model.model.layers[layer]
            else:
                module = self.model.model.layers[layer]
                for part in component.split('.'):
                    module = getattr(module, part)
            
            def make_hook(k):
                def hook(mod, inp, out):
                    self.captured[k] = out[0].detach() if isinstance(out, tuple) else out.detach()
                return hook
            
            handle = module.register_forward_hook(make_hook(key))
            self.hooks.append(handle)
            registered.add(key)
    
    def cleanup(self):
        """Remove hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []
    
    def get_scores(self, prompt: str) -> Dict[str, float]:
        """Get raw scores from all gates for a single prompt."""
        self.captured = {}
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            self.model(**inputs)
        
        seq_len = inputs['attention_mask'].sum() - 1
        scores = {}
        
        for gate_id, gate in self.gates.items():
            layer = gate["layer"]
            component = gate["component"]
            key = f"L{layer}_{component}"
            
            if key not in self.captured:
                continue
            
            # Get embedding at last token
            act = self.captured[key]
            if act.dim() == 3:
                emb = act[0, seq_len, :].float()
            else:
                emb = act[-1, :].float()
            
            emb = F.normalize(emb, dim=-1)
            
            # OBF gate uses direct projection (no origin offset)
            # Other gates use displacement from origin
            if gate_id == "OBF":
                harm_dir = gate["harm_direction"].float()  # Ensure float32
                score = (emb @ harm_dir).item()
            else:
                displacement = emb - gate["origin"]
                score = (displacement @ gate["harm_direction"]).item()
            
            scores[gate_id] = score
        
        return scores


def main():
    parser = argparse.ArgumentParser(description="Cache VecP gate scores for calibration")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--gate-dir", required=True, help="Directory with trained gates")
    parser.add_argument("--gauntlet", required=True, help="Gauntlet file")
    parser.add_argument("--obf-gate", default=None, help="Path to OBF gate file")
    parser.add_argument("--output", default="cached_scores.json", help="Output file")
    parser.add_argument("--device", default="cuda")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  VecP Score Caching")
    print("=" * 70)
    
    # Load model
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    
    # Load gates
    print(f"\nLoading gates from: {args.gate_dir}")
    gates = load_gates(args.gate_dir, args.device, args.obf_gate)
    print(f"  Loaded {len(gates)} gates: {list(gates.keys())}")
    
    # Load gauntlet
    print(f"\nLoading gauntlet: {args.gauntlet}")
    prompts = load_gauntlet(args.gauntlet)
    print(f"  Loaded {len(prompts)} prompts")
    
    # Create cacher
    cacher = ScoreCacher(model, tokenizer, gates, args.device)
    
    # Cache all scores
    print(f"\nCaching scores...")
    start_time = time.time()
    
    results = {
        "metadata": {
            "model": args.model,
            "gate_dir": args.gate_dir,
            "gauntlet": args.gauntlet,
            "timestamp": datetime.now().isoformat(),
            "num_prompts": len(prompts),
            "gates": list(gates.keys()),
        },
        "gate_info": {
            gate_id: {
                "layer": g["layer"],
                "component": g["component"],
                "fisher": g["fisher"],
                "original_threshold": g["threshold"],
            }
            for gate_id, g in gates.items()
        },
        "prompts": {}
    }
    
    for i, prompt in enumerate(prompts):
        scores = cacher.get_scores(prompt["text"])
        
        results["prompts"][prompt["id"]] = {
            "text": prompt["text"][:200],  # Truncate for storage
            "label": prompt["label"],
            "category": prompt["category"],
            "scores": scores,
        }
        
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(prompts) - i - 1) / rate / 60
            print(f"  [{i+1}/{len(prompts)}] {rate:.1f} prompts/sec, ETA: {eta:.1f} min")
    
    # Calculate SNR for each gate
    print(f"\nCalculating SNR for each gate...")
    import numpy as np
    
    for gate_id in gates.keys():
        benign_scores = [
            p["scores"][gate_id] 
            for p in results["prompts"].values() 
            if p["label"] == "benign" and gate_id in p["scores"]
        ]
        harmful_scores = [
            p["scores"][gate_id] 
            for p in results["prompts"].values() 
            if p["label"] == "harmful" and gate_id in p["scores"]
        ]
        
        if benign_scores and harmful_scores:
            b_mean, b_std = np.mean(benign_scores), np.std(benign_scores)
            h_mean, h_std = np.mean(harmful_scores), np.std(harmful_scores)
            
            separation = abs(h_mean - b_mean)
            noise = (b_std + h_std) / 2
            snr = separation / noise if noise > 0 else 0
            
            results["gate_info"][gate_id]["snr"] = snr
            results["gate_info"][gate_id]["benign_mean"] = b_mean
            results["gate_info"][gate_id]["benign_std"] = b_std
            results["gate_info"][gate_id]["harmful_mean"] = h_mean
            results["gate_info"][gate_id]["harmful_std"] = h_std
            
            print(f"  {gate_id}: Fisher={results['gate_info'][gate_id]['fisher']:.2f}, SNR={snr:.2f}")
    
    # Cleanup
    cacher.cleanup()
    
    # Save
    print(f"\nSaving to: {args.output}")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    elapsed = time.time() - start_time
    print(f"\nDone! Cached {len(prompts)} prompts in {elapsed/60:.1f} minutes")
    print(f"File size: {Path(args.output).stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
