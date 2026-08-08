"""
Configuration system for Atlas-Nano pipeline.

Loads settings from YAML config files with sensible defaults.
Users can override any setting via CLI flags or config file.
"""

from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any

DEFAULT_CONFIG_NAME = "atlas_nano.yaml"
PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_TRAINING_GAUNTLET = str(PACKAGE_DIR / "data" / "gauntlet_v3_corrected.txt")


@dataclass
class ModelConfig:
    """Model-related settings."""
    name: str = "Qwen/Qwen3-4B"
    device: str = "cuda"
    batch_size: int = 8
    # Layer range for gate training (auto-detected if not set)
    layer_start: Optional[int] = None
    layer_end: Optional[int] = None


@dataclass
class TrainConfig:
    """Training pipeline settings."""
    gauntlet: str = DEFAULT_TRAINING_GAUNTLET
    output_dir: str = "./gates"
    generate_data_only: bool = False


@dataclass
class CacheConfig:
    """Score caching settings."""
    gate_dir: str = "./gates"
    gauntlet: str = DEFAULT_TRAINING_GAUNTLET
    output: str = "cached_scores.json"
    obf_gate: Optional[str] = None


@dataclass
class CalibrateConfig:
    """CMA-ES calibration settings."""
    cached: str = "cached_scores.json"
    output: str = "calibration_result.json"
    target_recall: float = 0.92
    max_fp: float = 0.12
    min_recall: float = 0.85
    mode: str = "balanced"  # recall, precision, f1, balanced
    aggregation: str = "snr_weighted"
    optimize_thresholds: bool = True
    optimize_weights: bool = True
    optimize_block_assist: bool = True
    max_iterations: int = 500
    population_size: int = 20
    grid_refine: bool = True


@dataclass
class ApplyConfig:
    """Calibration application settings."""
    gate_dir: str = "./gates"
    calibration: str = "calibration_result.json"
    output: str = "./gates_calibrated"
    obf_gate: Optional[str] = None


@dataclass
class RunConfig:
    """Live inference settings."""
    gate_dir: str = "./gates_calibrated"
    fast_path_assist: float = 0.85
    block_assist: float = 0.15
    assist_strength: float = 0.5
    event_horizon: float = 1.02
    benign_matrix: Optional[str] = None


@dataclass
class BenchmarkConfig:
    """Benchmark evaluation settings."""
    gate_dir: str = "./gates_calibrated"
    gauntlet: Optional[str] = None
    output: str = "vecp_v2_results"
    max_prompts: Optional[int] = None
    block_assist: float = 0.12
    fast_path_assist: float = 0.85
    assist_strength: float = 0.5
    event_horizon: float = 1.02
    aggregation_mode: str = "max_mean"
    benign_matrix: Optional[str] = None
    obf_gate: Optional[str] = None


@dataclass
class SignCheckConfig:
    """Sign-Check Atlas settings (single-axis distillation for GGUF)."""
    gauntlet: str = DEFAULT_TRAINING_GAUNTLET
    output_dir: str = "sign_check_atlas/results"
    component: str = "residual"
    # Phase 3 threshold search
    threshold_mode: str = "balanced"  # sweep, pr_curve, balanced
    optimize_metric: str = "f1"  # f1, accuracy, precision, recall
    min_recall: float = 0.85
    max_fp_rate: float = 0.05


@dataclass
class GGUFConfig:
    """GGUF embedding settings."""
    output: str = "model_safety.gguf"
    mode: str = "sidecar"  # sidecar or inject
    input_gguf: Optional[str] = None  # required for inject mode
    # Auto-populated from sign-check results if not set
    energy_axis: Optional[str] = None
    phase1_results: Optional[str] = None
    phase3_results: Optional[str] = None


@dataclass
class PipelineConfig:
    """Full pipeline orchestration settings."""
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    calibrate: CalibrateConfig = field(default_factory=CalibrateConfig)
    apply: ApplyConfig = field(default_factory=ApplyConfig)
    run: RunConfig = field(default_factory=RunConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    sign_check: SignCheckConfig = field(default_factory=SignCheckConfig)
    gguf: GGUFConfig = field(default_factory=GGUFConfig)


# ---- Model architecture presets ----
# Maps model family patterns to recommended layer ranges
MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "qwen3-4b": {"layer_start": 9, "layer_end": 27, "hidden_dim": 2560},
    "qwen3-8b": {"layer_start": 10, "layer_end": 30, "hidden_dim": 4096},
    "qwen2.5": {"layer_start": 8, "layer_end": 24, "hidden_dim": 3584},
    "phi-4": {"layer_start": 9, "layer_end": 27, "hidden_dim": 5120},
    "llama-3": {"layer_start": 10, "layer_end": 28, "hidden_dim": 4096},
    "gemma-2": {"layer_start": 8, "layer_end": 22, "hidden_dim": 2304},
    "mistral": {"layer_start": 10, "layer_end": 28, "hidden_dim": 4096},
}


def detect_model_preset(model_name: str) -> Optional[Dict[str, Any]]:
    """Auto-detect layer ranges from model name."""
    name_lower = model_name.lower()
    for pattern, preset in MODEL_PRESETS.items():
        if pattern in name_lower:
            return preset
    return None


def load_config(config_path: Optional[str] = None) -> PipelineConfig:
    """Load config from YAML file, falling back to defaults."""
    config = PipelineConfig()

    # Try explicit path, then look in current directory
    paths_to_try = []
    if config_path:
        paths_to_try.append(Path(config_path))
    paths_to_try.append(Path(DEFAULT_CONFIG_NAME))

    for path in paths_to_try:
        if path.exists():
            import yaml
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            _apply_dict(config, data)
            break

    # Auto-detect layer range if not explicitly set
    if config.model.layer_start is None or config.model.layer_end is None:
        preset = detect_model_preset(config.model.name)
        if preset:
            if config.model.layer_start is None:
                config.model.layer_start = preset["layer_start"]
            if config.model.layer_end is None:
                config.model.layer_end = preset["layer_end"]
        else:
            # Conservative defaults for unknown models
            config.model.layer_start = 8
            config.model.layer_end = 24

    return config


def save_default_config(path: str = DEFAULT_CONFIG_NAME):
    """Write a default config file with comments for the user to edit."""
    import yaml
    config = PipelineConfig()
    data = _to_serializable(asdict(config))
    with open(path, "w") as f:
        f.write("# Atlas-Nano Pipeline Configuration\n")
        f.write("# Edit this file to customize your pipeline.\n")
        f.write("# All paths are relative to where you run the command.\n\n")
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return path


def _apply_dict(config, data: dict):
    """Recursively apply a dict onto a dataclass config."""
    for key, value in data.items():
        if hasattr(config, key):
            attr = getattr(config, key)
            if hasattr(attr, "__dataclass_fields__") and isinstance(value, dict):
                _apply_dict(attr, value)
            else:
                setattr(config, key, value)


def _to_serializable(obj):
    """Convert dataclass dict to YAML-friendly types."""
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj
