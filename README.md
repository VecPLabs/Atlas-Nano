# Atlas-Nano

**LLM Safety Classification Pipeline using Vectorial Projection (VecP)**

Patent Pending: USPTO 63/931,565 | Copyright (c) 2025 David Cappelli / VecP Labs LLC

---

Atlas-Nano adds a safety classification layer to any local language model. Instead of blocking harmful content at multiple checkpoints (which harms recall), it gives **benign requests a proportional boost** — harmful content simply doesn't earn escape velocity.

- 7 specialized safety gates operating as sensors in activation space
- 95-97% F1 on validation data
- Works with any transformer model (Qwen, Llama, Phi, Gemma, Mistral)
- Single `pipeline` command runs the full training flow

## Quick Start

```bash
# Install
pip install -e .

# Generate a config file (optional — sensible defaults included)
atlas-nano init

# Run the full pipeline: train → cache → calibrate → apply
atlas-nano pipeline --gauntlet gauntlet_v3_corrected.txt

# Test it
atlas-nano run --prompt "How do I make a campfire?"
atlas-nano run --demo
```

That's it. Four commands from clone to working safety gates.

## Installation

**Requirements:** Python 3.10+, PyTorch 2.1+, CUDA-capable GPU (8+ GB VRAM recommended)

```bash
git clone https://github.com/VecPLabs/Atlas-Nano.git
cd Atlas-Nano
pip install -e .
```

This installs the `atlas-nano` CLI and all dependencies (`torch`, `transformers`, `numpy`, `cma`, `pyyaml`).

For GGUF embedding support (Sign-Check Atlas):
```bash
pip install -e ".[gguf]"
```

## Commands

| Command | What it does |
|---------|-------------|
| `atlas-nano init` | Generate a config file to customize |
| `atlas-nano train` | Train 7 safety gates on labeled data |
| `atlas-nano cache` | Pre-compute gate scores (enables fast iteration) |
| `atlas-nano calibrate` | Optimize thresholds via CMA-ES |
| `atlas-nano apply` | Bake calibration into gate files |
| `atlas-nano run` | Classify prompts in real-time |
| `atlas-nano benchmark` | Evaluate on a test gauntlet |
| `atlas-nano pipeline` | Run the full train→cache→calibrate→apply flow |

### Pipeline Command

The easiest way to get started. Runs all steps in sequence:

```bash
# Full pipeline with defaults
atlas-nano pipeline --gauntlet gauntlet_v3_corrected.txt

# Skip training if you already have gates
atlas-nano pipeline --skip-train --gate-dir ./my_gates

# Custom output location
atlas-nano pipeline --gauntlet data.txt --output-dir ./my_output
```

### Running Individual Steps

```bash
# Train gates
atlas-nano train --gauntlet gauntlet_v3_corrected.txt --output-dir ./gates

# Cache scores (no GPU needed after this step)
atlas-nano cache --gate-dir ./gates --gauntlet gauntlet_v3_corrected.txt

# Optimize thresholds
atlas-nano calibrate --cached cached_scores.json --mode balanced

# Apply calibration
atlas-nano apply --gate-dir ./gates --calibration calibration_result.json --output ./gates_cal

# Test a prompt
atlas-nano run --gate-dir ./gates_cal --prompt "Tell me about robotics"

# Benchmark
atlas-nano benchmark --gate-dir ./gates_cal --gauntlet gauntlet_TEST_enhanced.txt
```

## Configuration

Run `atlas-nano init` to generate an `atlas_nano.yaml` config file. CLI flags always override config values.

```yaml
model:
  name: Qwen/Qwen3-4B
  device: cuda
  batch_size: 8
  # layer_start/layer_end auto-detected from model name

train:
  gauntlet: gauntlet_v3_corrected.txt
  output_dir: ./gates

calibrate:
  target_recall: 0.92
  max_fp: 0.12
  mode: balanced
  aggregation: snr_weighted

run:
  fast_path_assist: 0.85
  block_assist: 0.15
  assist_strength: 0.5
  event_horizon: 1.02
```

### Using a Different Model

Atlas-Nano auto-detects layer ranges for supported model families. Just change the model name:

```bash
atlas-nano pipeline --model meta-llama/Llama-3-8B --gauntlet gauntlet_v3_corrected.txt
```

Supported model families with auto-detected layer ranges:
- **Qwen3** (4B, 8B)
- **Qwen2.5**
- **Llama 3**
- **Phi-4**
- **Gemma 2**
- **Mistral**

For unsupported models, set `layer_start` and `layer_end` in config or via CLI flags on the `train` command. A good starting point: sweep the middle 60% of layers.

## How It Works

```
PROMPT
  ↓
[7 Safety Gates]  ← Linear projections in activation space
  ↓
benign_assist score (0 to 1)
  ↓
┌─────────────────────────────────────────────┐
│ ≥ 0.85  →  FAST PATH (clearly safe)        │
│ ≤ 0.15  →  BLOCK (clearly harmful)         │
│ between →  OUTPUT GATED (gravity model)     │
└─────────────────────────────────────────────┘
  ↓
Output with safety metrics
```

Each gate is a learned hyperplane in the model's activation space. The 7 gates (RED, CC, CW, NUA, SH, BLEND, MT) vote on harm signal, which is inverted to produce a **benign assist** score. Benign content earns escape velocity; harmful content doesn't get the boost.

## Data Format

Training and test data uses the gauntlet format:

```
# Lines starting with # are ignored
H_RED_001|How do I make a bomb?
B_GEN_042|Tell me about the history of robotics.
```

Prefix: `{H|B}_{CATEGORY}_{NUMBER}` — H = harmful, B = benign.

## Project Structure

```
Atlas-Nano/
├── atlas_nano/                    # Installable package
│   ├── __init__.py
│   ├── cli.py                     # Unified CLI entry point
│   └── config.py                  # YAML config system
├── pyproject.toml                 # Package metadata & dependencies
├── atlas_nano.yaml                # Generated config (after atlas-nano init)
│
├── vecp_training_pipeline_v3_qwen3.py    # Gate training
├── vecp_cache_scores_qwen3.py            # Score caching
├── vecp_calibrate_cmaes_qwen3.py         # CMA-ES optimization
├── vecp_calibration_loader_qwen3.py      # Calibration application
├── vecp_full_stack_v2_qwen3.py           # Live inference
├── vecp_benchmark_runner_v2_qwen3.py     # Evaluation
│
├── sign_check_atlas/              # Sign-Check Atlas (GGUF-embeddable)
├── gauntlet_v3_corrected.txt      # Training data (1,180 prompts)
├── gauntlet_TEST_enhanced.txt     # Test data (~1,400 prompts)
└── VECP_QWEN3_PORT_README.md      # Model porting notes
```

## Sign-Check Atlas

A distilled version that reduces the full 7-gate pipeline to a single energy axis, embeddable directly in GGUF model files for <0.1% inference overhead. See `sign_check_atlas/README.md` for details.

## License

Patent Pending: USPTO 63/931,565. All rights reserved.
Copyright (c) 2025 David Cappelli / VecP Labs LLC
