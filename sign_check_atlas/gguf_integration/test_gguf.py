#!/usr/bin/env python3
"""
Sign-Check Atlas: GGUF Integration Tests
==========================================
Roundtrip tests for GGUF safety metadata embedding and extraction.

Tests:
    1. Write sidecar GGUF → Read back → Verify match
    2. Energy axis integrity (normalized, correct dim)
    3. Metadata completeness
    4. Validation checks

Usage:
    python sign_check_atlas/gguf_integration/test_gguf.py
    python sign_check_atlas/gguf_integration/test_gguf.py --dim 2560  # Qwen3-4B
    python sign_check_atlas/gguf_integration/test_gguf.py --dim 3072  # Phi-4

Patent Pending: USPTO 63/931,565
Copyright (c) 2025-2026 David Cappelli / VecP Labs
"""

import argparse
import os
import sys
import tempfile
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from sign_check_atlas.gguf_integration.embed_safety import SafetyMetadata, GGUFSafetyWriter
from sign_check_atlas.gguf_integration.extract_safety import GGUFSafetyReader


def test_roundtrip(hidden_dim: int = 2560):
    """Test write → read roundtrip for safety GGUF sidecar."""
    print(f"\n  TEST: Roundtrip (dim={hidden_dim})")

    # Create synthetic energy axis
    rng = np.random.RandomState(42)
    energy_axis = rng.randn(hidden_dim).astype(np.float32)
    energy_axis /= np.linalg.norm(energy_axis)

    # Build metadata
    safety = SafetyMetadata(
        energy_axis=energy_axis,
        extraction_layer=14,
        threshold=0.0123,
        extraction_component="residual",
        model_name="test/model",
        f1=0.912,
        precision=0.934,
        recall=0.891,
    )

    # Write sidecar
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        writer = GGUFSafetyWriter(safety)
        writer.write_sidecar(tmp_path)

        # Read back
        reader = GGUFSafetyReader(tmp_path)

        # Verify
        assert reader.has_safety_data(), "Safety data not found"

        meta = reader.get_safety_metadata()
        assert meta["safety.version"] == 1
        assert meta["safety.type"] == "sign_check_atlas"
        assert meta["safety.extraction_layer"] == 14
        assert meta["safety.extraction_component"] == "residual"
        assert abs(meta["safety.threshold"] - 0.0123) < 1e-5
        assert meta["safety.hidden_dim"] == hidden_dim
        assert abs(meta["safety.calibration_f1"] - 0.912) < 1e-3
        assert abs(meta["safety.calibration_precision"] - 0.934) < 1e-3
        assert abs(meta["safety.calibration_recall"] - 0.891) < 1e-3
        assert meta["safety.model_name"] == "test/model"

        # Verify energy axis
        assert reader.energy_axis is not None
        assert len(reader.energy_axis) == hidden_dim
        assert np.allclose(reader.energy_axis, energy_axis, atol=1e-6), \
            f"Axis mismatch: max diff={np.max(np.abs(reader.energy_axis - energy_axis))}"

        # Validate
        is_valid, issues = reader.validate()
        assert is_valid, f"Validation failed: {issues}"

        print(f"    PASS: All metadata and tensor data match")

    finally:
        os.unlink(tmp_path)


def test_normalization_check():
    """Test that validation catches unnormalized axes."""
    print(f"\n  TEST: Normalization validation")

    # Create unnormalized axis
    axis = np.ones(100, dtype=np.float32) * 0.5  # norm = 5.0

    safety = SafetyMetadata(
        energy_axis=axis,
        extraction_layer=10,
        threshold=0.0,
    )

    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Manually override to keep unnormalized
        safety.energy_axis = axis  # bypass normalization

        writer = GGUFSafetyWriter(safety)
        writer.write_sidecar(tmp_path)

        reader = GGUFSafetyReader(tmp_path)
        is_valid, issues = reader.validate()

        # Should catch the unnormalized axis
        has_norm_issue = any("not normalized" in i for i in issues)
        assert has_norm_issue, f"Should have caught unnormalized axis, got: {issues}"

        print(f"    PASS: Correctly flagged unnormalized axis")

    finally:
        os.unlink(tmp_path)


def test_dimension_mismatch():
    """Test that validation catches dimension mismatches."""
    print(f"\n  TEST: Dimension mismatch detection")

    axis = np.random.randn(256).astype(np.float32)
    axis /= np.linalg.norm(axis)

    safety = SafetyMetadata(
        energy_axis=axis,
        extraction_layer=10,
        threshold=0.0,
    )

    # Corrupt hidden_dim
    safety.hidden_dim = 512

    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        writer = GGUFSafetyWriter(safety)
        writer.write_sidecar(tmp_path)

        reader = GGUFSafetyReader(tmp_path)
        is_valid, issues = reader.validate()

        has_dim_issue = any("mismatch" in i.lower() for i in issues)
        assert has_dim_issue, f"Should have caught dim mismatch, got: {issues}"

        print(f"    PASS: Correctly flagged dimension mismatch")

    finally:
        os.unlink(tmp_path)


def test_multiple_dimensions():
    """Test with various model dimensions."""
    print(f"\n  TEST: Multiple model dimensions")

    dims = {
        "Phi-4-mini": 3072,
        "Qwen3-4B": 2560,
        "Llama-3.2-3B": 3200,
        "Gemma-2B": 2048,
        "Generic-384": 384,
    }

    for name, dim in dims.items():
        axis = np.random.randn(dim).astype(np.float32)
        axis /= np.linalg.norm(axis)

        safety = SafetyMetadata(
            energy_axis=axis,
            extraction_layer=int(dim * 0.5 / 128),  # rough
            threshold=0.01,
            model_name=name,
        )

        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            writer = GGUFSafetyWriter(safety)
            writer.write_sidecar(tmp_path)

            reader = GGUFSafetyReader(tmp_path)
            assert reader.has_safety_data()
            assert len(reader.energy_axis) == dim
            assert np.allclose(reader.energy_axis, axis, atol=1e-6)

            is_valid, _ = reader.validate()
            assert is_valid

            print(f"    PASS: {name} (dim={dim})")

        finally:
            os.unlink(tmp_path)


def test_json_export():
    """Test JSON metadata export."""
    print(f"\n  TEST: JSON metadata export")

    axis = np.random.randn(256).astype(np.float32)
    axis /= np.linalg.norm(axis)

    safety = SafetyMetadata(
        energy_axis=axis,
        extraction_layer=10,
        threshold=0.05,
        model_name="test/json",
        f1=0.9,
        precision=0.95,
        recall=0.85,
    )

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        safety.to_json(tmp_path)

        import json
        with open(tmp_path) as f:
            data = json.load(f)

        assert data["safety.version"] == 1
        assert data["safety.type"] == "sign_check_atlas"
        assert data["safety.extraction_layer"] == 10
        assert abs(data["safety.threshold"] - 0.05) < 1e-5
        assert data["safety.energy_axis_shape"] == [256]

        print(f"    PASS: JSON export verified")

    finally:
        os.unlink(tmp_path)


def main():
    parser = argparse.ArgumentParser(description="GGUF integration tests")
    parser.add_argument("--dim", type=int, default=2560,
                       help="Primary dimension to test")
    args = parser.parse_args()

    print("=" * 60)
    print("SIGN-CHECK ATLAS: GGUF Integration Tests")
    print("=" * 60)

    tests = [
        lambda: test_roundtrip(args.dim),
        test_normalization_check,
        test_dimension_mismatch,
        test_multiple_dimensions,
        test_json_export,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"    FAIL: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
