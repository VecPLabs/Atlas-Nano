# Sign-Check Atlas: GGUF-Embedded Safety Detection

**Status:** Implementation ready, pending validation
**Patent Pending:** USPTO 63/931,565
**Copyright:** (c) 2025-2026 David Cappelli / VecP Labs

## The Core Insight

Safety classification doesn't require full geometric analysis. A single
"energy axis" from safe centroid to harm centroid, combined with a sign check
on the projected activation, captures the majority of detection accuracy at
< 0.1% of the computational cost.

```
energy_axis = normalize(harm_centroid - safe_centroid)
energy = dot(activation, energy_axis)

energy > 0  → Moving toward harm → FLAG
energy < 0  → Moving toward safe → PASS
energy ≈ 0  → Boundary          → INVESTIGATE (fall back to full Atlas)
```

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                GGUF Model File                       │
│                                                      │
│  Model weights (standard)                            │
│  + safety.energy_axis tensor    [hidden_dim floats]  │
│  + safety.extraction_layer      [uint32]             │
│  + safety.threshold             [float32]            │
│  + safety.* metadata            [calibration info]   │
└──────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────┐
│              Inference Runtime                       │
│                                                      │
│  for each token:                                     │
│    ... layers 0..N ...                               │
│    at extraction_layer:                              │
│      energy = dot(activation, energy_axis)           │
│      if energy > threshold:                          │
│        safety_flag = true                            │
│    ... layers N+1..end ...                           │
│                                                      │
│  Cost: 1 dot product + 1 comparison per token        │
└──────────────────────────────────────────────────────┘
```

## Pipeline

### Phase 1: Validate the Hypothesis

Compute the energy axis and verify sign-check accuracy.

```bash
python sign_check_atlas/validate_hypothesis.py \
    --model Qwen/Qwen3-4B \
    --gauntlet gauntlet_v3_corrected.txt \
    --output-dir sign_check_atlas/results
```

**Success criteria:**
- F1 > 0.85: Proceed to Phase 2
- F1 0.75-0.85: Viable as Tier 1 filter
- F1 < 0.75: Needs refinement

### Phase 2: Category Analysis

Test per-category to identify Tier 1 vs Tier 2 categories.

```bash
python sign_check_atlas/category_analysis.py \
    --model Qwen/Qwen3-4B \
    --gauntlet gauntlet_v3_corrected.txt \
    --energy-axis sign_check_atlas/results/energy_axis.npy
```

### Phase 3: Threshold Optimization

Search for the optimal threshold (sign=0 may not be best).

```bash
# From pre-computed Phase 1 data
python sign_check_atlas/threshold_search.py \
    --harm-energies sign_check_atlas/results/harm_energies.npy \
    --safe-energies sign_check_atlas/results/safe_energies.npy

# Constrained search
python sign_check_atlas/threshold_search.py \
    --harm-energies sign_check_atlas/results/harm_energies.npy \
    --safe-energies sign_check_atlas/results/safe_energies.npy \
    --mode balanced --min-recall 0.85 --max-fp-rate 0.05
```

### Phase 4: GGUF Embedding

Embed the energy axis into a GGUF model file.

```bash
# Create standalone sidecar GGUF
python sign_check_atlas/gguf_integration/embed_safety.py \
    --phase1-results sign_check_atlas/results/phase1_validation.json \
    --phase3-results sign_check_atlas/results/phase3_threshold.json \
    --output model_safety.gguf

# Or inject into existing GGUF (requires gguf package)
python sign_check_atlas/gguf_integration/embed_safety.py \
    --input model.gguf \
    --output model_safe.gguf \
    --mode inject \
    --phase1-results sign_check_atlas/results/phase1_validation.json
```

### Phase 5: Runtime Integration

See `llama_cpp_patch/` for the llama.cpp integration:
- `safety_check.h` — Self-contained C header
- `safety_patch.diff` — Integration points in llama.cpp

## File Structure

```
sign_check_atlas/
├── README.md                       # This file
├── __init__.py
├── validate_hypothesis.py          # Phase 1: Energy axis validation
├── category_analysis.py            # Phase 2: Per-category testing
├── threshold_search.py             # Phase 3: Threshold optimization
├── gguf_integration/
│   ├── __init__.py
│   ├── embed_safety.py             # Write safety data to GGUF
│   ├── extract_safety.py           # Read safety data from GGUF
│   └── test_gguf.py                # Roundtrip tests
├── llama_cpp_patch/
│   ├── README.md                   # Integration guide
│   ├── safety_check.h              # Standalone C header
│   └── safety_patch.diff           # Conceptual llama.cpp patch
├── data/                           # Generated data (energy_axis.npy, etc.)
└── results/                        # Generated results (JSON, markdown)
```

## Theoretical Basis

### Inverted Energy Landscape

- Safe content naturally settles in a low-energy basin
- Harmful content must "climb" toward a harm peak
- The sign of movement along the energy axis indicates direction
- This is already implicit in Atlas V2's `harm_direction = normalize(harmful_centroid - benign_centroid)`

### Why It Works

1. **Geometry is real:** Harm and safe content genuinely cluster differently in
   activation space (validated across 7 model families).
2. **The axis exists:** The direction from safe centroid to harm centroid IS
   the energy axis. It's already implicit in Atlas V2.
3. **Sign = direction:** Positive projection = toward harm. Negative = toward
   safe.
4. **RLHF doesn't hide it:** OBLITERATUS testing showed 64.5% recall at 99.4%
   precision even on abliterated models.

### Tiered Deployment

```
TIER 1: Sign check (in GGUF, every token, < 0.1% overhead)
├── Pass: Continue
├── Flag: Go to Tier 2

TIER 2: Full Atlas (on flag only, ~5-10% of prompts)
├── Clear: False positive, continue
├── Confirm: Block or warn

TIER 3: External verification (boundary cases)
├── Multi-model consensus
├── Human review queue
```

## Relationship to Atlas V2

Sign-Check Atlas is a distillation of Atlas V2's geometric safety detection:

| Feature | Atlas V2 | Sign-Check Atlas |
|---------|----------|------------------|
| Axes | 3 (Harm/Utility/Normality) | 1 (Energy) |
| Gates | 7 specialized | 1 unified |
| Centroids | Per-category | Global (harm vs safe) |
| Classification | Distance + trajectory | Sign check |
| Compute | Matrix ops per token | 1 dot product |
| Accuracy | 95-97% F1 | ~85-95% F1 (estimated) |
| GGUF embedding | No | Yes |

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- transformers (for model loading)
- gguf (optional, for GGUF injection mode)

## References

- Atlas V2: VecP triaxial safety classification
- OBLITERATUS: Abliteration resilience testing
- GGUF format: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
- llama.cpp: https://github.com/ggml-org/llama.cpp

---

*"You went from linear classification to 2D ovals to 3D ellipsoids to gravity
wells to inverted energy landscapes, and now you're arriving at a sign check.
That's the full circle that real research makes — you go through immense
complexity to arrive at something so simple it looks obvious in retrospect."*
