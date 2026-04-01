"""
Sign-Check Atlas: GGUF-Embedded Safety Detection
=================================================
Reduces Atlas V2's geometric harm detection to a single linear projection
and sign check, enabling near-zero-cost safety classification that can be
embedded directly into GGUF model files.

Core operation:
    energy_axis = normalize(harm_centroid - safe_centroid)
    energy = dot(activation, energy_axis)
    energy > threshold → FLAG (toward harm)
    energy < threshold → PASS (toward safe)

Modules:
    validate_hypothesis  - Phase 1: Compute and validate the energy axis
    category_analysis    - Phase 2: Per-category accuracy mapping
    threshold_search     - Phase 3: Optimal threshold optimization
    gguf_integration     - Phase 4: GGUF embedding and reading
    llama_cpp_patch      - Phase 5: llama.cpp integration

Patent Pending: USPTO 63/931,565
Copyright (c) 2025-2026 David Cappelli / VecP Labs
"""

__version__ = "0.1.0"
__author__ = "David Cappelli / VecP Labs"
