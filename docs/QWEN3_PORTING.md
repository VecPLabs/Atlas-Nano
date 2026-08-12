# VecP Full Stack v2.0 — Qwen3 4B Port

**Copyright (c) 2025 David Cappelli / VecP Labs LLC**

---

## Port Summary

Ported from **Phi4 Mini Instruct** (32 layers, 3072-dim, `Phi3ForCausalLM`) → **Qwen3 4B Instruct** (36 layers, 2560-dim, `Qwen3ForCausalLM`).

**Key findings:**
- Qwen3 uses **identical component names** to Phi4 and Qwen2.5 — `mlp.down_proj`, `self_attn.o_proj`, `model.model.layers[N]`. No hook refactoring needed.
- Qwen3 is **natively supported** in `transformers ≥ 4.51.0` — no `trust_remote_code` required (kept for backward compat).
- `attn_implementation="eager"` retained — required for VecP hook capture (SDPA/FlashAttn fuse ops and skip hooks).
- **Note on head_dim:** Qwen3-4B has `head_dim=128` with 32 heads → internal attention dim = 4096, but `o_proj` outputs 2560-dim. VecP hooks capture output tensors, so all embeddings are 2560-dim. This is transparent to VecP gate logic.

### Model Variants

| Model | Context | Thinking Mode | Recommended |
|-------|---------|--------------|-------------|
| `Qwen/Qwen3-4B` | 32K (131K w/ YaRN) | Base (no instruct) | ✅ Default — cleanest geometric signal |
| `Qwen/Qwen3-4B-Instruct` | 32K (131K w/ YaRN) | Hybrid thinking | Alternative |
| `Qwen/Qwen3-4B-Instruct-2507` | 262K | Non-thinking only | Alternative (latest instruct) |

The base model is recommended: RLHF safety training in instruct models compresses the harm/benign separation in embedding space, making VecP gates work against the model's own masking. The base model provides the cleanest geometric signals for gate training.

### What Changed

| Parameter | Phi4 Mini | Qwen3 4B | File(s) |
|-----------|-----------|----------|---------|
| Model string | `microsoft/Phi-4-mini-instruct` | `Qwen/Qwen3-4B` | All 6 |
| Architecture | `Phi3ForCausalLM` (32L, 3072-dim) | `Qwen3ForCausalLM` (36L, 2560-dim) | Reference |
| Layer sweep | 9-27 (32L) | 10-30 (36L) | training_pipeline |
| Trajectory layers | (5, 13) | (6, 15) | full_stack |
| Gravity layers | (16, 20, 24) | (18, 22, 27) | full_stack |
| Default gate dir | `./gates_phi4` | `./gates_qwen3` | training, cache, calibration_loader |
| GQA config | 24 heads, 8 KV | 32 heads, 8 KV | Reference only |
| Hidden dim | 3072 | 2560 | Transparent (auto from hooks) |

### What Didn't Change

- `atlas_nano.pipeline.calibrate` — model-agnostic calibration from cached scores
- All gate logic, aggregation modes, gravity model, obfuscation prefilter — unchanged
- Component names: `mlp.down_proj`, `self_attn.o_proj` — **identical**
- Layer accessor: `model.model.layers[N]` — **identical**
- Gauntlet format and categories — unchanged

### Layer Mapping Rationale

Layers were mapped proportionally to maintain coverage at the same relative depth:

| Purpose | Qwen2.5-7B (28L) | Phi4 Mini (32L) | Qwen3 4B (36L) | Relative Depth |
|---------|-------------------|-----------------|-----------------|----------------|
| Sweep start | 8 | 9 | 10 | ~28% |
| Sweep end | 24 | 27 | 30 | ~83% |
| Trajectory early | 4 | 5 | 6 | ~16% |
| Trajectory mid | 12 | 13 | 15 | ~40% |
| Gravity mid | 14 | 16 | 18 | ~50% |
| Gravity late | 18 | 20 | 22 | ~61% |
| Gravity deep | 21 | 24 | 27 | ~75% |

This preserves the **Last Clean Signal Principle** (optimal safety signal at 40-70% model depth).

---

## Files

### Pipeline (6 scripts)

| File | Purpose |
|------|---------|
| `atlas_nano.pipeline.training` | Train gates on Qwen3 4B |
| `atlas_nano.pipeline.cache` | Cache gate scores for calibration |
| `atlas_nano.pipeline.calibrate` | CMA-ES threshold optimization |
| `atlas_nano.pipeline.apply` | Apply calibration to gate files |
| `atlas_nano.pipeline.inference` | Live inference pipeline |
| `atlas_nano.pipeline.benchmark` | Evaluation and metrics |

### Data

| File | Prompts | Use |
|------|---------|-----|
| `atlas_nano/data/gauntlet_v3_corrected.txt` | 1,180 | Bundled training and calibration data |
| `data/evaluation/gauntlet_TEST_enhanced.txt` | 1,449 | Mixed-provenance candidate evaluation data |
| `data/evaluation/gauntlet_HARMBENCH_CLEANED.txt` | varies | Mixed HarmBench-derived evaluation data |

---

## Execution

**Requirements:**
```bash
pip install torch transformers numpy cma
# Qwen3 requires transformers >= 4.51.0 (native support, no custom code)
# Verify: python -c "import transformers; print(transformers.__version__)"
```

**Full pipeline:**
```bash
# 1. Train gates on Qwen3 4B
python -m atlas_nano.pipeline.training \
    --train-gauntlet atlas_nano/data/gauntlet_v3_corrected.txt \
    --output-dir ./gates_qwen3

# 2. Cache scores
python -m atlas_nano.pipeline.cache \
    --gate-dir ./gates_qwen3 \
    --gauntlet atlas_nano/data/gauntlet_v3_corrected.txt \
    --output cached_scores_qwen3.json

# 3. CMA-ES calibration
python -m atlas_nano.pipeline.calibrate \
    --cached cached_scores_qwen3.json \
    --aggregation snr_weighted \
    --output calibration_qwen3.json

# 3.5. Apply calibration
python -m atlas_nano.pipeline.apply \
    --gate-dir ./gates_qwen3 \
    --calibration calibration_qwen3.json \
    --output ./gates_qwen3_calibrated

# 4. Benchmark (TEST set)
python -m atlas_nano.pipeline.benchmark \
    --gate-dir ./gates_qwen3_calibrated \
    --gauntlet gauntlet_TEST.txt \
    --aggregation-mode snr_weighted

# 5. Benchmark (Enhanced set)
python -m atlas_nano.pipeline.benchmark \
    --gate-dir ./gates_qwen3_calibrated \
    --gauntlet data/evaluation/gauntlet_TEST_enhanced.txt \
    --aggregation-mode snr_weighted
```

**Note:** All `_qwen3` files are self-consistent — the benchmark runner imports from
`python -m atlas_nano.pipeline.inference` directly after installing the package.

---

## Gate Training Data Viability

Same gauntlet data as Phi4 port — see Phi4 README for per-gate balance analysis.
Gate quality depends on model representation, not training data; re-run the sweep to find optimal layers for Qwen3 4B.

---

## Architecture Reference

```
Qwen3 4B: Qwen3ForCausalLM, 36 layers, 2560 hidden, GQA (32 heads, 8 KV, head_dim=128)

model.model.layers[N].self_attn.o_proj    → attn output (2560-dim)
model.model.layers[N].mlp.down_proj       → MLP output (2560-dim)
model.model.layers[N]                     → residual stream (2560-dim)

VecP hooks identical to Phi4 / Qwen2.5 — no adapter needed.

Note: attention_bias=false in Qwen3 (vs bias in Phi4). Irrelevant to VecP
since hooks capture output activations, not weight matrices.
```

### Pipeline Architecture (unchanged)

```
OBFUSCATION PREFILTER (regex + optional ML OBF gate)
         ↓ obf_signal reduces benign_assist up to 50%
INPUT GATE SENSOR (7 gates: RED CC CW NUA SH BLEND MT)
         ↓ harm_signal → benign_assist = 1 - harm_signal
ROUTER
  ├─ benign_assist ≥ 0.85 → FAST PATH (no monitoring)
  ├─ benign_assist ≤ 0.12 → BLOCK
  └─ else → OUTPUT GATED (gravity model with assist reduction)
         ↓ effective_gravity = raw - (assist × benign × strength)
         ↓ block on event_horizon / velocity spike / trajectory tense
RLHF COLLABORATION (benchmark: effective_recall = VecP + RLHF)
```

---

## Port History

| Port | Source → Target | Key Change |
|------|----------------|------------|
| v1 | Qwen2.5-7B (28L, 3584-dim) → Phi4 Mini (32L, 3072-dim) | Phi3 arch class, same hooks |
| **v2** | **Phi4 Mini (32L, 3072-dim) → Qwen3 4B (36L, 2560-dim)** | **Native transformers, same hooks** |

The shared hook structure across these model families supports further
portability experiments; it does not establish equivalent accuracy or universal
compatibility without per-model calibration and held-out evaluation.
