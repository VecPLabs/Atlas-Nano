#!/usr/bin/env python3
"""
Sign-Check Atlas: Phase 4 - GGUF Safety Embedding
===================================================
Embeds the sign-check energy axis and safety metadata into a GGUF model file.

The energy axis is stored as:
    - Custom metadata keys under the `safety.*` namespace
    - A tensor named `safety.energy_axis` containing the axis vector

This allows any GGUF-compatible runtime (llama.cpp, etc.) to perform
sign-check safety classification at near-zero cost during inference.

GGUF Metadata Keys Added:
    safety.version          (uint32)  - Safety metadata version
    safety.type             (string)  - "sign_check_atlas"
    safety.extraction_layer (uint32)  - Layer to extract activations from
    safety.extraction_component (string) - Component (residual, mlp.down_proj, etc.)
    safety.threshold        (float32) - Classification threshold
    safety.hidden_dim       (uint32)  - Dimension of the energy axis
    safety.calibration_f1   (float32) - F1 from calibration
    safety.calibration_precision (float32) - Precision from calibration
    safety.calibration_recall    (float32) - Recall from calibration
    safety.model_name       (string)  - Model the axis was calibrated on
    safety.calibration_date (string)  - ISO date of calibration

GGUF Tensor Added:
    safety.energy_axis      (float32) - The energy axis vector [hidden_dim]

Usage:
    # Embed safety metadata into existing GGUF
    python sign_check_atlas/gguf_integration/embed_safety.py \
        --input model.gguf \
        --output model_safe.gguf \
        --energy-axis sign_check_atlas/results/energy_axis.npy \
        --threshold-config sign_check_atlas/results/optimal_threshold.json \
        --extraction-layer 14 \
        --model-name "Qwen/Qwen3-4B"

    # From Phase 1+3 results
    python sign_check_atlas/gguf_integration/embed_safety.py \
        --input model.gguf \
        --output model_safe.gguf \
        --phase1-results sign_check_atlas/results/phase1_validation.json \
        --phase3-results sign_check_atlas/results/phase3_threshold.json

Patent Pending: USPTO 63/931,565
Copyright (c) 2025-2026 David Cappelli / VecP Labs
"""

import argparse
import json
import struct
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


# =============================================================================
# GGUF CONSTANTS
# =============================================================================

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3

# GGUF metadata value types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# GGML tensor types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1


# =============================================================================
# SAFETY METADATA BUILDER
# =============================================================================

class SafetyMetadata:
    """
    Container for sign-check safety metadata to embed in GGUF.
    """

    def __init__(
        self,
        energy_axis: np.ndarray,
        extraction_layer: int,
        threshold: float = 0.0,
        extraction_component: str = "residual",
        model_name: str = "",
        f1: float = 0.0,
        precision: float = 0.0,
        recall: float = 0.0,
    ):
        self.energy_axis = energy_axis.astype(np.float32)
        self.extraction_layer = extraction_layer
        self.threshold = threshold
        self.extraction_component = extraction_component
        self.model_name = model_name
        self.f1 = f1
        self.precision = precision
        self.recall = recall
        self.hidden_dim = len(energy_axis)
        self.version = 1
        self.calibration_date = datetime.now().isoformat()

    @classmethod
    def from_phase_results(
        cls,
        energy_axis_path: str,
        phase1_results_path: str = None,
        phase3_results_path: str = None,
    ) -> "SafetyMetadata":
        """Build SafetyMetadata from Phase 1 and Phase 3 output files."""
        energy_axis = np.load(energy_axis_path)

        extraction_layer = 14  # default
        model_name = ""
        threshold = 0.0
        f1 = 0.0
        precision = 0.0
        recall = 0.0
        component = "residual"

        if phase1_results_path:
            with open(phase1_results_path) as f:
                p1 = json.load(f)
            model_name = p1.get("model", "")
            component = p1.get("component", "residual")

            # Use best single layer from per_layer_results
            per_layer = p1.get("per_layer_results", [])
            if per_layer:
                extraction_layer = per_layer[0]["layer"]

            metrics = p1.get("sign_check_metrics", {})
            f1 = metrics.get("f1", 0.0)
            precision = metrics.get("precision", 0.0)
            recall = metrics.get("recall", 0.0)

        if phase3_results_path:
            with open(phase3_results_path) as f:
                p3 = json.load(f)
            opt = p3.get("optimal_threshold_metrics", {})
            threshold = opt.get("threshold", 0.0)
            f1 = opt.get("f1", f1)
            precision = opt.get("precision", precision)
            recall = opt.get("recall", recall)

        return cls(
            energy_axis=energy_axis,
            extraction_layer=extraction_layer,
            threshold=threshold,
            extraction_component=component,
            model_name=model_name,
            f1=f1,
            precision=precision,
            recall=recall,
        )

    def to_dict(self) -> Dict:
        """Return a JSON-serializable dict of the metadata."""
        return {
            "safety.version": self.version,
            "safety.type": "sign_check_atlas",
            "safety.extraction_layer": self.extraction_layer,
            "safety.extraction_component": self.extraction_component,
            "safety.threshold": self.threshold,
            "safety.hidden_dim": self.hidden_dim,
            "safety.calibration_f1": self.f1,
            "safety.calibration_precision": self.precision,
            "safety.calibration_recall": self.recall,
            "safety.model_name": self.model_name,
            "safety.calibration_date": self.calibration_date,
        }

    def to_json(self, path: str):
        """Save metadata as JSON (for debugging/inspection)."""
        data = self.to_dict()
        data["safety.energy_axis_shape"] = list(self.energy_axis.shape)
        data["safety.energy_axis_norm"] = float(np.linalg.norm(self.energy_axis))
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


# =============================================================================
# GGUF WRITER (MINIMAL)
# =============================================================================

class GGUFSafetyWriter:
    """
    Minimal GGUF writer that creates a standalone safety GGUF file
    containing only the sign-check metadata and energy axis tensor.

    This can be:
    1. Used as a sidecar file loaded alongside the model GGUF
    2. Merged into an existing GGUF using gguf tools

    The sidecar approach is preferred as it doesn't modify the original model.
    """

    def __init__(self, safety_metadata: SafetyMetadata):
        self.metadata = safety_metadata

    def _write_string(self, f, s: str):
        """Write a GGUF string (uint64 length + bytes)."""
        encoded = s.encode("utf-8")
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)

    def _write_kv_string(self, f, key: str, value: str):
        """Write a string key-value pair."""
        self._write_string(f, key)
        f.write(struct.pack("<I", GGUF_TYPE_STRING))
        self._write_string(f, value)

    def _write_kv_uint32(self, f, key: str, value: int):
        """Write a uint32 key-value pair."""
        self._write_string(f, key)
        f.write(struct.pack("<I", GGUF_TYPE_UINT32))
        f.write(struct.pack("<I", value))

    def _write_kv_float32(self, f, key: str, value: float):
        """Write a float32 key-value pair."""
        self._write_string(f, key)
        f.write(struct.pack("<I", GGUF_TYPE_FLOAT32))
        f.write(struct.pack("<f", value))

    def _write_kv_array_float32(self, f, key: str, values: np.ndarray):
        """Write a float32 array key-value pair."""
        self._write_string(f, key)
        f.write(struct.pack("<I", GGUF_TYPE_ARRAY))
        f.write(struct.pack("<I", GGUF_TYPE_FLOAT32))
        f.write(struct.pack("<Q", len(values)))
        f.write(values.astype(np.float32).tobytes())

    def write_sidecar(self, output_path: str):
        """
        Write a standalone GGUF sidecar file with safety metadata.

        The sidecar contains:
        - GGUF header
        - Safety metadata as KV pairs
        - Energy axis as a tensor
        """
        md = self.metadata

        # Count KV pairs
        n_kv = 11  # All safety.* keys
        n_tensors = 1  # safety.energy_axis

        # Compute tensor data size and alignment
        tensor_data = md.energy_axis.astype(np.float32).tobytes()
        alignment = 32  # Standard GGUF alignment

        with open(output_path, "wb") as f:
            # === HEADER ===
            f.write(GGUF_MAGIC)
            f.write(struct.pack("<I", GGUF_VERSION))
            f.write(struct.pack("<Q", n_tensors))
            f.write(struct.pack("<Q", n_kv))

            # === METADATA KV PAIRS ===
            self._write_kv_uint32(f, "safety.version", md.version)
            self._write_kv_string(f, "safety.type", "sign_check_atlas")
            self._write_kv_uint32(f, "safety.extraction_layer", md.extraction_layer)
            self._write_kv_string(f, "safety.extraction_component", md.extraction_component)
            self._write_kv_float32(f, "safety.threshold", md.threshold)
            self._write_kv_uint32(f, "safety.hidden_dim", md.hidden_dim)
            self._write_kv_float32(f, "safety.calibration_f1", md.f1)
            self._write_kv_float32(f, "safety.calibration_precision", md.precision)
            self._write_kv_float32(f, "safety.calibration_recall", md.recall)
            self._write_kv_string(f, "safety.model_name", md.model_name)
            self._write_kv_string(f, "safety.calibration_date", md.calibration_date)

            # === TENSOR INFO ===
            # Tensor name
            self._write_string(f, "safety.energy_axis")
            # n_dims
            f.write(struct.pack("<I", 1))
            # dimensions (ne[0])
            f.write(struct.pack("<Q", md.hidden_dim))
            # type (F32)
            f.write(struct.pack("<I", GGML_TYPE_F32))
            # offset from start of tensor data section
            f.write(struct.pack("<Q", 0))

            # === PADDING TO ALIGNMENT ===
            current_pos = f.tell()
            pad_size = (alignment - (current_pos % alignment)) % alignment
            f.write(b"\x00" * pad_size)

            # === TENSOR DATA ===
            f.write(tensor_data)

            # Final padding
            current_pos = f.tell()
            pad_size = (alignment - (current_pos % alignment)) % alignment
            f.write(b"\x00" * pad_size)

        file_size = Path(output_path).stat().st_size
        print(f"  Written GGUF sidecar: {output_path} ({file_size} bytes)")
        print(f"  Energy axis: {md.hidden_dim}-dim F32 ({md.hidden_dim * 4} bytes)")
        print(f"  Extraction layer: {md.extraction_layer}")
        print(f"  Threshold: {md.threshold:.6f}")
        print(f"  Calibration F1: {md.f1:.3f}")


# =============================================================================
# GGUF METADATA INJECTION (using gguf Python package)
# =============================================================================

def inject_into_existing_gguf(
    input_path: str,
    output_path: str,
    safety_metadata: SafetyMetadata,
):
    """
    Inject safety metadata into an existing GGUF file using the gguf
    Python package. Creates a new file with the safety data added.

    Requires: pip install gguf
    """
    try:
        from gguf import GGUFReader, GGUFWriter
    except ImportError:
        print("ERROR: gguf package not installed. Install with: pip install gguf")
        print("Falling back to sidecar mode.")
        writer = GGUFSafetyWriter(safety_metadata)
        sidecar_path = output_path.replace(".gguf", "_safety.gguf")
        writer.write_sidecar(sidecar_path)
        return

    print(f"  Reading: {input_path}")
    reader = GGUFReader(input_path)

    # Get architecture from original
    arch = "llama"  # default
    for field in reader.fields:
        if field == "general.architecture":
            arch = str(reader.fields[field].parts[-1], "utf-8")
            break

    print(f"  Architecture: {arch}")
    print(f"  Writing: {output_path}")

    writer = GGUFWriter(output_path, arch)

    # Copy all existing metadata
    for key, field in reader.fields.items():
        if key.startswith("safety."):
            continue  # Skip old safety metadata
        # The gguf library handles different types
        try:
            if field.types and field.types[0].name == "STRING":
                val = str(field.parts[-1], "utf-8")
                writer.add_string(key, val)
            elif field.types and field.types[0].name == "UINT32":
                writer.add_uint32(key, int(field.parts[-1][0]))
            elif field.types and field.types[0].name == "FLOAT32":
                writer.add_float32(key, float(field.parts[-1][0]))
        except Exception:
            pass  # Skip fields we can't handle

    # Add safety metadata
    md = safety_metadata
    writer.add_uint32("safety.version", md.version)
    writer.add_string("safety.type", "sign_check_atlas")
    writer.add_uint32("safety.extraction_layer", md.extraction_layer)
    writer.add_string("safety.extraction_component", md.extraction_component)
    writer.add_float32("safety.threshold", md.threshold)
    writer.add_uint32("safety.hidden_dim", md.hidden_dim)
    writer.add_float32("safety.calibration_f1", md.f1)
    writer.add_float32("safety.calibration_precision", md.precision)
    writer.add_float32("safety.calibration_recall", md.recall)
    writer.add_string("safety.model_name", md.model_name)
    writer.add_string("safety.calibration_date", md.calibration_date)

    # Copy all existing tensors
    for tensor in reader.tensors:
        writer.add_tensor(tensor.name, tensor.data)

    # Add energy axis tensor
    writer.add_tensor("safety.energy_axis", md.energy_axis)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"  Done. Safety metadata injected into {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sign-Check Atlas: Embed safety metadata into GGUF"
    )

    parser.add_argument("--energy-axis", type=str, default=None,
                       help="Path to energy_axis.npy")
    parser.add_argument("--phase1-results", type=str, default=None,
                       help="Path to phase1_validation.json")
    parser.add_argument("--phase3-results", type=str, default=None,
                       help="Path to phase3_threshold.json (or optimal_threshold.json)")
    parser.add_argument("--extraction-layer", type=int, default=None,
                       help="Override extraction layer")
    parser.add_argument("--threshold", type=float, default=None,
                       help="Override threshold")
    parser.add_argument("--model-name", type=str, default="",
                       help="Model name for metadata")

    parser.add_argument("--input", type=str, default=None,
                       help="Input GGUF file to inject into")
    parser.add_argument("--output", type=str, required=True,
                       help="Output path (GGUF sidecar or modified GGUF)")

    parser.add_argument("--mode", type=str, default="sidecar",
                       choices=["sidecar", "inject"],
                       help="sidecar=standalone file, inject=modify existing GGUF")

    args = parser.parse_args()

    # Build SafetyMetadata
    if args.phase1_results:
        energy_axis_path = args.energy_axis
        if not energy_axis_path:
            # Try to find it next to phase1 results
            results_dir = Path(args.phase1_results).parent
            energy_axis_path = str(results_dir / "energy_axis.npy")

        safety = SafetyMetadata.from_phase_results(
            energy_axis_path=energy_axis_path,
            phase1_results_path=args.phase1_results,
            phase3_results_path=args.phase3_results,
        )
    elif args.energy_axis:
        energy_axis = np.load(args.energy_axis)
        safety = SafetyMetadata(
            energy_axis=energy_axis,
            extraction_layer=args.extraction_layer or 14,
            threshold=args.threshold or 0.0,
            model_name=args.model_name,
        )
    else:
        parser.error("Either --energy-axis or --phase1-results required")

    # Apply overrides
    if args.extraction_layer is not None:
        safety.extraction_layer = args.extraction_layer
    if args.threshold is not None:
        safety.threshold = args.threshold
    if args.model_name:
        safety.model_name = args.model_name

    print("=" * 60)
    print("SIGN-CHECK ATLAS: Phase 4 GGUF Embedding")
    print("=" * 60)
    print(f"  Energy axis dim:    {safety.hidden_dim}")
    print(f"  Extraction layer:   {safety.extraction_layer}")
    print(f"  Component:          {safety.extraction_component}")
    print(f"  Threshold:          {safety.threshold:.6f}")
    print(f"  Calibration F1:     {safety.f1:.3f}")
    print(f"  Model:              {safety.model_name}")
    print()

    if args.mode == "sidecar":
        writer = GGUFSafetyWriter(safety)
        writer.write_sidecar(args.output)
    elif args.mode == "inject":
        if not args.input:
            parser.error("--input required for inject mode")
        inject_into_existing_gguf(args.input, args.output, safety)

    # Also save JSON metadata for inspection
    json_path = args.output.replace(".gguf", "_metadata.json")
    safety.to_json(json_path)
    print(f"  Metadata JSON: {json_path}")


if __name__ == "__main__":
    main()
