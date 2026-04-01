# Sign-Check Atlas: llama.cpp Integration

## Overview

This patch adds near-zero-cost safety classification to llama.cpp by reading
a sign-check energy axis from the GGUF model file and computing a single dot
product during inference.

**Performance overhead:** < 0.1% of inference time (one dot product per token
at one layer).

## Files

- `safety_check.h` — Standalone C header with the safety check logic. Can be
  used independently of llama.cpp.
- `safety_patch.diff` — Conceptual patch showing the three integration points
  in llama.cpp.

## Integration Points

### 1. Model Loading (`llama-model-loader.cpp`)

Read `safety.*` metadata keys and the `safety.energy_axis` tensor from GGUF:

```cpp
const int idx_version = gguf_find_key(ctx_gguf, "safety.version");
if (idx_version >= 0) {
    model.safety.enabled = true;
    model.safety.extraction_layer = gguf_get_val_u32(...);
    model.safety.threshold = gguf_get_val_f32(...);
    // Load energy_axis tensor pointer
    model.safety.energy_axis = (float *)ggml_get_tensor(...)->data;
}
```

### 2. Layer Processing (`llama-decode.cpp`)

At the extraction layer, compute the dot product:

```cpp
if (model.safety.enabled && il == model.safety.extraction_layer) {
    float energy = 0.0f;
    for (uint32_t d = 0; d < dim; d++) {
        energy += activation[d] * model.safety.energy_axis[d];
    }
    if (energy > model.safety.threshold) {
        batch_result.safety_flagged = true;
        batch_result.safety_energy = energy;
    }
}
```

### 3. Public API (`llama.h`)

Expose safety status to callers:

```cpp
bool  llama_safety_enabled(const struct llama_context * ctx);
bool  llama_safety_flagged(const struct llama_context * ctx);
float llama_safety_energy(const struct llama_context * ctx);
```

## GGUF Metadata Keys

| Key | Type | Description |
|-----|------|-------------|
| `safety.version` | uint32 | Metadata version (currently 1) |
| `safety.type` | string | `"sign_check_atlas"` |
| `safety.extraction_layer` | uint32 | Layer index for activation extraction |
| `safety.extraction_component` | string | Component name (e.g., `"residual"`) |
| `safety.threshold` | float32 | Classification threshold |
| `safety.hidden_dim` | uint32 | Energy axis dimension |
| `safety.calibration_f1` | float32 | F1 from calibration |
| `safety.calibration_precision` | float32 | Precision from calibration |
| `safety.calibration_recall` | float32 | Recall from calibration |
| `safety.model_name` | string | Model the axis was calibrated on |

## GGUF Tensor

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| `safety.energy_axis` | F32 | `[hidden_dim]` | Normalized energy axis vector |

## How It Works

1. During model loading, the safety metadata and energy axis tensor are read
   from the GGUF file.
2. During inference, at the specified extraction layer, the residual stream
   activation for the last token is projected onto the energy axis:
   `energy = dot(activation, energy_axis)`
3. If `energy > threshold`, a safety flag is set.
4. The caller can query `llama_safety_flagged()` to check the result.

## Tiered Deployment

```
TIER 1: Sign check (in GGUF, every token)
├── Pass: Continue generation
├── Flag: Route to Tier 2

TIER 2: Full Atlas (on flag only)
├── Clear: False positive, continue
├── Confirm: Block or warn

TIER 3: External verification (boundary cases)
├── Multi-model consensus
├── Human review queue
```

## Notes

- The patch is conceptual — exact line numbers depend on llama.cpp version.
- The `safety_check.h` header is self-contained and can be used as a
  reference implementation for any GGML-based runtime.
- The energy axis is memory-mapped from the GGUF tensor data; no allocation
  needed.

---

*Patent Pending: USPTO 63/931,565*
*Copyright (c) 2025-2026 David Cappelli / VecP Labs*
