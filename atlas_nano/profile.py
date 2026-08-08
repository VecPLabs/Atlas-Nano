"""Validation for model-coupled Atlas safety profiles."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


REQUIRED_FIELDS = {
    "schema_version", "profile_id", "atlas_version", "license", "attribution", "base_model",
    "architecture", "extraction", "decision", "evaluation",
}


class ProfileError(ValueError):
    """Raised when a profile is invalid or incompatible."""


@dataclass(frozen=True)
class RuntimeModel:
    name: str
    architecture: Optional[str] = None
    revision: Optional[str] = None
    hidden_dim: Optional[int] = None


def load_profile(path: str | Path) -> dict[str, Any]:
    profile_path = Path(path)
    try:
        data = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProfileError(f"could not read profile {profile_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ProfileError("profile root must be a JSON object")
    validate_profile(data)
    return data


def validate_profile(profile: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_FIELDS - profile.keys())
    if missing:
        raise ProfileError(f"missing required fields: {', '.join(missing)}")
    if profile["schema_version"] != 1:
        raise ProfileError(f"unsupported schema_version: {profile['schema_version']!r}")
    for field in ("profile_id", "atlas_version", "license", "attribution", "base_model", "architecture"):
        if not isinstance(profile[field], str) or not profile[field].strip():
            raise ProfileError(f"{field} must be a non-empty string")
    extraction = profile.get("extraction")
    decision = profile.get("decision")
    evaluation = profile.get("evaluation")
    if not isinstance(extraction, Mapping):
        raise ProfileError("extraction must be an object")
    if not isinstance(decision, Mapping):
        raise ProfileError("decision must be an object")
    if not isinstance(evaluation, Mapping):
        raise ProfileError("evaluation must be an object")
    for field in ("component", "layer", "hidden_dim"):
        if field not in extraction:
            raise ProfileError(f"extraction.{field} is required")
    if not isinstance(extraction["layer"], int) or extraction["layer"] < 0:
        raise ProfileError("extraction.layer must be a non-negative integer")
    if not isinstance(extraction["hidden_dim"], int) or extraction["hidden_dim"] <= 0:
        raise ProfileError("extraction.hidden_dim must be a positive integer")
    if not isinstance(decision.get("threshold"), (int, float)):
        raise ProfileError("decision.threshold must be numeric")
    if decision.get("role") not in {"tier1_filter", "full_classifier"}:
        raise ProfileError("decision.role must be tier1_filter or full_classifier")
    if evaluation.get("split") not in {"calibration", "validation", "held_out", "external"}:
        raise ProfileError("evaluation.split must describe the reported metrics")


def assert_compatible(profile: Mapping[str, Any], runtime: RuntimeModel) -> None:
    """Fail closed when known runtime properties do not match the profile."""
    mismatches: list[str] = []
    if profile["base_model"].casefold() != runtime.name.casefold():
        mismatches.append(f"model {runtime.name!r} != {profile['base_model']!r}")
    if runtime.architecture and profile["architecture"] != runtime.architecture:
        mismatches.append(f"architecture {runtime.architecture!r} != {profile['architecture']!r}")
    expected_revision = profile.get("base_model_revision")
    if runtime.revision and expected_revision and runtime.revision != expected_revision:
        mismatches.append(f"revision {runtime.revision!r} != {expected_revision!r}")
    expected_dim = profile["extraction"]["hidden_dim"]
    if runtime.hidden_dim is not None and runtime.hidden_dim != expected_dim:
        mismatches.append(f"hidden_dim {runtime.hidden_dim} != {expected_dim}")
    if mismatches:
        raise ProfileError("incompatible safety profile: " + "; ".join(mismatches))


def verify_artifacts(profile: Mapping[str, Any], profile_dir: str | Path) -> None:
    """Verify local release artifacts without following paths outside the profile."""
    base = Path(profile_dir).resolve()
    for artifact in profile.get("artifacts", []):
        relative = artifact.get("path")
        expected = artifact.get("sha256")
        if not relative or not expected:
            raise ProfileError("every release artifact requires path and sha256")
        artifact_path = (base / relative).resolve()
        if artifact_path.parent != base:
            raise ProfileError(f"artifact path escapes profile directory: {relative}")
        if not artifact_path.is_file():
            raise ProfileError(f"artifact not found: {artifact_path}")
        hasher = hashlib.sha256()
        with artifact_path.open("rb") as artifact_file:
            for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        if digest.casefold() != str(expected).casefold():
            raise ProfileError(f"artifact checksum mismatch: {relative}")
