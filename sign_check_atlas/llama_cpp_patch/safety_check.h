/**
 * Sign-Check Atlas: llama.cpp Safety Integration Header
 * ======================================================
 * Minimal header for GGUF-embedded sign-check safety classification.
 *
 * This provides near-zero-cost safety classification during inference
 * by computing a single dot product at the specified extraction layer.
 *
 * Performance: One dot product + one comparison per token at one layer.
 * Overhead: < 0.1% of total inference time.
 *
 * Patent Pending: USPTO 63/931,565
 * Copyright (c) 2025-2026 David Cappelli / VecP Labs
 */

#ifndef LLAMA_SAFETY_CHECK_H
#define LLAMA_SAFETY_CHECK_H

#include <stdbool.h>
#include <stdint.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Safety check result for a single token */
struct llama_safety_result {
    float  energy;       /* Projected energy value (positive = toward harm) */
    bool   flagged;      /* true if energy > threshold */
    int    layer;        /* Layer where activation was extracted */
};

/* Safety metadata loaded from GGUF */
struct llama_safety_params {
    bool     enabled;              /* Whether safety check is active */
    uint32_t version;              /* Safety metadata version */
    uint32_t extraction_layer;     /* Layer to extract activations from */
    uint32_t hidden_dim;           /* Dimension of the energy axis */
    float    threshold;            /* Classification threshold */
    float  * energy_axis;          /* Energy axis vector [hidden_dim] */
    float    calibration_f1;       /* F1 score from calibration */
    float    calibration_precision;
    float    calibration_recall;
};

/**
 * Initialize safety params from GGUF metadata.
 *
 * Called during model loading. Reads safety.* metadata keys and the
 * safety.energy_axis tensor from the GGUF file.
 *
 * Returns true if safety metadata was found and loaded successfully.
 */
static inline bool llama_safety_init(
    struct llama_safety_params * params
    /* In practice, this would take the gguf_context and ggml_context */
) {
    /* Placeholder: actual implementation reads from GGUF context.
     *
     * Pseudo-implementation:
     *
     *   int version_idx = gguf_find_key(ctx, "safety.version");
     *   if (version_idx < 0) {
     *       params->enabled = false;
     *       return false;
     *   }
     *
     *   params->version          = gguf_get_val_u32(ctx, version_idx);
     *   params->extraction_layer = gguf_get_val_u32(ctx, gguf_find_key(ctx, "safety.extraction_layer"));
     *   params->hidden_dim       = gguf_get_val_u32(ctx, gguf_find_key(ctx, "safety.hidden_dim"));
     *   params->threshold        = gguf_get_val_f32(ctx, gguf_find_key(ctx, "safety.threshold"));
     *
     *   // Load energy axis tensor
     *   int tensor_idx = ggml_get_tensor_idx(ml, "safety.energy_axis");
     *   struct ggml_tensor * axis_tensor = ggml_get_tensor(ml, tensor_idx);
     *   params->energy_axis = (float *)axis_tensor->data;
     *   params->hidden_dim  = ggml_nelements(axis_tensor);
     *
     *   params->enabled = true;
     *   return true;
     */
    params->enabled = false;
    return false;
}

/**
 * Compute safety energy at the extraction layer.
 *
 * This is the core sign-check operation:
 *   energy = dot(activation, energy_axis)
 *   flagged = (energy > threshold)
 *
 * Called once per token at the specified extraction layer.
 *
 * @param params    Safety parameters (axis, threshold, layer)
 * @param activation Pointer to the activation vector at extraction layer
 *                   (hidden_dim floats, typically the residual stream)
 * @param n_dims    Dimension of the activation vector (must match hidden_dim)
 * @return          Safety result with energy value and flag
 */
static inline struct llama_safety_result llama_safety_check(
    const struct llama_safety_params * params,
    const float * activation,
    uint32_t n_dims
) {
    struct llama_safety_result result;
    result.energy  = 0.0f;
    result.flagged = false;
    result.layer   = (int)params->extraction_layer;

    if (!params->enabled || !params->energy_axis || !activation) {
        return result;
    }

    if (n_dims != params->hidden_dim) {
        return result;  /* Dimension mismatch, skip */
    }

    /* Compute dot product: energy = sum(activation[i] * energy_axis[i]) */
    float energy = 0.0f;
    uint32_t i;

    /* Main loop - compiler will auto-vectorize with -O2+ */
    for (i = 0; i < n_dims; i++) {
        energy += activation[i] * params->energy_axis[i];
    }

    result.energy  = energy;
    result.flagged = (energy > params->threshold);

    return result;
}

/**
 * Free safety params resources.
 */
static inline void llama_safety_free(struct llama_safety_params * params) {
    /* energy_axis is typically owned by the GGUF tensor data,
     * so we don't free it here. Just clear the pointer. */
    params->energy_axis = NULL;
    params->enabled = false;
}

#ifdef __cplusplus
}
#endif

#endif /* LLAMA_SAFETY_CHECK_H */
