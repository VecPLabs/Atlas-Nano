#!/usr/bin/env python3
"""
Sign-Check Atlas: GGUF Safety Metadata Reader
===============================================
Reads and validates sign-check safety metadata from GGUF files.

Supports both:
    - Sidecar GGUF files (standalone safety data)
    - Integrated GGUF files (safety data merged into model)

Usage:
    # Inspect a safety sidecar
    python sign_check_atlas/gguf_integration/extract_safety.py \
        --input model_safety.gguf

    # Extract energy axis to numpy
    python sign_check_atlas/gguf_integration/extract_safety.py \
        --input model_safety.gguf \
        --extract-axis output_axis.npy

    # Validate against known results
    python sign_check_atlas/gguf_integration/extract_safety.py \
        --input model_safety.gguf \
        --validate sign_check_atlas/results/phase1_validation.json

Copyright (c) 2025-2026 David Cappelli / VecP Labs
"""

import argparse
import json
import struct
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple


# =============================================================================
# GGUF CONSTANTS (matching embed_safety.py)
# =============================================================================

GGUF_MAGIC = b"GGUF"
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGML_TYPE_F32 = 0


# =============================================================================
# MINIMAL GGUF READER
# =============================================================================

class GGUFSafetyReader:
    """
    Minimal GGUF reader focused on extracting safety metadata.

    Reads the header, metadata KV pairs, and the safety.energy_axis tensor.
    Supports both sidecar and full model GGUF files.
    """

    def __init__(self, path: str):
        self.path = path
        self.metadata = {}
        self.energy_axis = None
        self.n_tensors = 0
        self.n_kv = 0

        self._read(path)

    def _read_string(self, f) -> str:
        """Read a GGUF string (uint64 length + bytes)."""
        length = struct.unpack("<Q", f.read(8))[0]
        return f.read(length).decode("utf-8")

    def _read(self, path: str):
        """Read the GGUF file and extract safety metadata."""
        with open(path, "rb") as f:
            # Header
            magic = f.read(4)
            if magic != GGUF_MAGIC:
                raise ValueError(f"Not a GGUF file: magic={magic!r}")

            version = struct.unpack("<I", f.read(4))[0]
            self.n_tensors = struct.unpack("<Q", f.read(8))[0]
            self.n_kv = struct.unpack("<Q", f.read(8))[0]

            # Read KV pairs
            for _ in range(self.n_kv):
                key = self._read_string(f)
                value_type = struct.unpack("<I", f.read(4))[0]

                if value_type == GGUF_TYPE_UINT32:
                    value = struct.unpack("<I", f.read(4))[0]
                elif value_type == GGUF_TYPE_FLOAT32:
                    value = struct.unpack("<f", f.read(4))[0]
                elif value_type == GGUF_TYPE_STRING:
                    value = self._read_string(f)
                elif value_type == GGUF_TYPE_ARRAY:
                    arr_type = struct.unpack("<I", f.read(4))[0]
                    arr_len = struct.unpack("<Q", f.read(8))[0]
                    if arr_type == GGUF_TYPE_FLOAT32:
                        data = f.read(arr_len * 4)
                        value = np.frombuffer(data, dtype=np.float32).copy()
                    else:
                        # Skip unknown array types
                        f.read(arr_len * 4)  # rough skip
                        value = None
                else:
                    # Skip unknown types (read 8 bytes as best guess)
                    f.read(8)
                    value = None

                self.metadata[key] = value

            # Read tensor info
            tensor_infos = []
            for _ in range(self.n_tensors):
                name = self._read_string(f)
                n_dims = struct.unpack("<I", f.read(4))[0]
                dims = []
                for _ in range(n_dims):
                    dims.append(struct.unpack("<Q", f.read(8))[0])
                tensor_type = struct.unpack("<I", f.read(4))[0]
                offset = struct.unpack("<Q", f.read(8))[0]
                tensor_infos.append({
                    "name": name,
                    "dims": dims,
                    "type": tensor_type,
                    "offset": offset,
                })

            # Align to next alignment boundary (default 32)
            alignment = self.metadata.get("general.alignment", 32)
            current = f.tell()
            pad = (alignment - (current % alignment)) % alignment
            f.seek(current + pad)

            tensor_data_start = f.tell()

            # Read energy axis tensor
            for info in tensor_infos:
                if info["name"] == "safety.energy_axis":
                    f.seek(tensor_data_start + info["offset"])
                    n_elements = 1
                    for d in info["dims"]:
                        n_elements *= d
                    if info["type"] == GGML_TYPE_F32:
                        data = f.read(n_elements * 4)
                        self.energy_axis = np.frombuffer(data, dtype=np.float32).copy()
                    break

    def get_safety_metadata(self) -> Dict:
        """Return only the safety.* metadata keys."""
        return {k: v for k, v in self.metadata.items() if k.startswith("safety.")}

    def has_safety_data(self) -> bool:
        """Check if this GGUF has sign-check safety metadata."""
        return "safety.version" in self.metadata and self.energy_axis is not None

    def validate(self) -> Tuple[bool, list]:
        """
        Validate the safety metadata for consistency.

        Returns (is_valid, list_of_issues).
        """
        issues = []

        if "safety.version" not in self.metadata:
            issues.append("Missing safety.version")
        if "safety.hidden_dim" not in self.metadata:
            issues.append("Missing safety.hidden_dim")
        if self.energy_axis is None:
            issues.append("Missing safety.energy_axis tensor")

        if self.energy_axis is not None and "safety.hidden_dim" in self.metadata:
            expected = self.metadata["safety.hidden_dim"]
            actual = len(self.energy_axis)
            if expected != actual:
                issues.append(f"Dimension mismatch: metadata says {expected}, tensor has {actual}")

        if self.energy_axis is not None:
            norm = np.linalg.norm(self.energy_axis)
            if abs(norm - 1.0) > 0.01:
                issues.append(f"Energy axis not normalized: norm={norm:.4f}")

        return len(issues) == 0, issues

    def print_summary(self):
        """Print a human-readable summary of the safety metadata."""
        safety = self.get_safety_metadata()

        if not safety:
            print("  No safety metadata found in this GGUF file.")
            return

        print(f"  Safety Metadata:")
        print(f"    Version:     {safety.get('safety.version', 'N/A')}")
        print(f"    Type:        {safety.get('safety.type', 'N/A')}")
        print(f"    Model:       {safety.get('safety.model_name', 'N/A')}")
        print(f"    Layer:       {safety.get('safety.extraction_layer', 'N/A')}")
        print(f"    Component:   {safety.get('safety.extraction_component', 'N/A')}")
        print(f"    Threshold:   {safety.get('safety.threshold', 'N/A')}")
        print(f"    Hidden Dim:  {safety.get('safety.hidden_dim', 'N/A')}")
        print(f"    Cal. F1:     {safety.get('safety.calibration_f1', 'N/A')}")
        print(f"    Cal. Prec:   {safety.get('safety.calibration_precision', 'N/A')}")
        print(f"    Cal. Recall: {safety.get('safety.calibration_recall', 'N/A')}")
        print(f"    Cal. Date:   {safety.get('safety.calibration_date', 'N/A')}")

        if self.energy_axis is not None:
            print(f"    Axis Shape:  ({len(self.energy_axis)},)")
            print(f"    Axis Norm:   {np.linalg.norm(self.energy_axis):.6f}")
            print(f"    Axis Range:  [{self.energy_axis.min():.6f}, {self.energy_axis.max():.6f}]")

        is_valid, issues = self.validate()
        if is_valid:
            print(f"    Validation:  PASS")
        else:
            print(f"    Validation:  FAIL")
            for issue in issues:
                print(f"      - {issue}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sign-Check Atlas: Read safety metadata from GGUF"
    )
    parser.add_argument("--input", type=str, required=True,
                       help="Input GGUF file")
    parser.add_argument("--extract-axis", type=str, default=None,
                       help="Extract energy axis to .npy file")
    parser.add_argument("--validate", type=str, default=None,
                       help="Validate against Phase 1 results JSON")
    parser.add_argument("--json", action="store_true",
                       help="Output as JSON")

    args = parser.parse_args()

    print("=" * 60)
    print("SIGN-CHECK ATLAS: GGUF Safety Reader")
    print("=" * 60)
    print(f"  File: {args.input}")
    print()

    reader = GGUFSafetyReader(args.input)

    if args.json:
        safety = reader.get_safety_metadata()
        # Convert numpy to list for JSON
        serializable = {}
        for k, v in safety.items():
            if isinstance(v, np.ndarray):
                serializable[k] = v.tolist()
            else:
                serializable[k] = v
        print(json.dumps(serializable, indent=2))
    else:
        reader.print_summary()

    if args.extract_axis:
        if reader.energy_axis is not None:
            np.save(args.extract_axis, reader.energy_axis)
            print(f"\n  Extracted axis to: {args.extract_axis}")
        else:
            print("\n  ERROR: No energy axis found in file")

    if args.validate:
        with open(args.validate) as f:
            expected = json.load(f)
        print(f"\n  Validating against: {args.validate}")
        # Compare key metrics
        safety = reader.get_safety_metadata()
        issues = []
        if expected.get("hidden_dim") and safety.get("safety.hidden_dim"):
            if expected["hidden_dim"] != safety["safety.hidden_dim"]:
                issues.append(f"Dim mismatch: expected {expected['hidden_dim']}, got {safety['safety.hidden_dim']}")
        if not issues:
            print("    Validation: PASS")
        else:
            for issue in issues:
                print(f"    ISSUE: {issue}")


if __name__ == "__main__":
    main()
