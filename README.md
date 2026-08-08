# Atlas Nano

Atlas Nano is a research-preview toolkit for sensing safety-relevant signals in
transformer activation space. Unlike a standalone judge model, Atlas is coupled
to the model it monitors: every safety profile is calibrated for a specific base
model, architecture, extraction point, and threshold policy.

> [!CAUTION]
> Atlas Nano produces safety signals; it is not a complete safety policy, a
> security boundary, or a substitute for application-level controls. Treat the
> included artifacts and runtime integration as experimental.

Copyright (c) 2025-2026 David Cappelli / VecP Labs LLC.

## What is included

- **Full Atlas:** seven activation-space gates for training, calibration, live
  inference, and evaluation.
- **Sign-Check Atlas:** a distilled single-axis Tier 1 sensor suitable for a
  low-overhead runtime integration.
- **Safety profiles:** machine-readable compatibility and evaluation metadata.
- **GGUF tooling:** experimental sidecar creation and extraction.

Sign-Check should normally route flagged or boundary inputs to Full Atlas or
another policy layer. It should not be treated as an interchangeable moderation
model.

## Supported release profile

The v0.1 research preview includes one documented profile:

| Profile | Role | Measurement | Status |
|---|---|---|---|
| `qwen3-4b-signcheck-v1` | Tier 1 filter | F1 0.895, precision 0.949, recall 0.848 on the 1,180-example calibration gauntlet | Experimental; not held-out evidence |

These figures are calibration measurements, not estimates of production
performance. The included results must not be generalized to other model
revisions, quantizations, prompts, languages, or deployments. See
[MODEL_CARD.md](MODEL_CARD.md) and [docs/EVALUATION.md](docs/EVALUATION.md).

## Install

Requirements: Python 3.10+, PyTorch 2.1+, and a CUDA-capable GPU for extraction
and training.

```bash
git clone https://github.com/VecPLabs/Atlas-Nano.git
cd Atlas-Nano
pip install -e .
```

GGUF injection support is optional:

```bash
pip install -e ".[gguf]"
```

The software is open source under [Apache-2.0](LICENSE). Atlas-authored datasets
and profile artifacts are available under CC BY 4.0; imported benchmark material
retains its original terms. See [DATA_LICENSES.md](DATA_LICENSES.md).

If Atlas Nano contributes to published research, benchmarks, models, or
products, please cite it using [CITATION.cff](CITATION.cff). Citation is strongly
requested, but is not an additional condition of the software license.

## Quick start

Validate that a profile matches the model you intend to use:

```bash
atlas-nano profile validate profiles/qwen3-4b-signcheck-v1/profile.json \
  --model Qwen/Qwen3-4B \
  --architecture Qwen3ForCausalLM \
  --hidden-dim 2560
```

Run the full research pipeline:

```bash
atlas-nano init
atlas-nano pipeline
atlas-nano run --gate-dir atlas_output/gates_calibrated \
  --prompt "How do I make a campfire?"
```

Add experimental Sign-Check and GGUF sidecar generation:

```bash
atlas-nano pipeline --gguf
```

The `llama.cpp` material in `sign_check_atlas/llama_cpp_patch/` is a conceptual
reference, not a tested upstream-compatible patch. Pin and validate a runtime
before deployment.

## Commands

| Command | Purpose |
|---|---|
| `atlas-nano profile validate` | Validate profile structure and compatibility |
| `atlas-nano init` | Generate a configuration file |
| `atlas-nano train` | Train the seven safety gates |
| `atlas-nano cache` | Cache gate scores |
| `atlas-nano calibrate` | Optimize thresholds with CMA-ES |
| `atlas-nano apply` | Apply calibration to gate files |
| `atlas-nano run` | Run full Atlas inference |
| `atlas-nano benchmark` | Evaluate calibrated gates |
| `atlas-nano sign-check` | Build a single-axis sensor |
| `atlas-nano gguf` | Create or inject experimental GGUF safety data |
| `atlas-nano pipeline` | Orchestrate the full research flow |

Run `atlas-nano <command> --help` for detailed options.

## Portability

Atlas can be ported to other decoder-only transformer families through
model-specific activation extraction and calibration. A preset is a starting
point, not evidence that a model is supported. Do not reuse axes, centroids, or
thresholds across models unless equivalence has been independently established.

Current code contains starting presets for Qwen, Llama, Phi, Gemma, and Mistral
families. Only the included Qwen3-4B profile is documented as a v0.1 release
artifact.

## Repository map

```text
atlas_nano/                 Installable package and CLI
atlas_nano/pipeline/        Training, calibration, inference, and evaluation
atlas_nano/data/            Bundled Atlas-authored training gauntlet
profiles/                   Versioned model-coupled artifact manifests
sign_check_atlas/           Single-axis research and GGUF tooling
tests/                      Fast package/profile tests
docs/                       Evaluation and release guidance
data/evaluation/            Mixed-provenance candidate evaluation datasets
```

The included sample GGUF and its metadata are colocated with the Qwen profile in
`profiles/qwen3-4b-signcheck-v1/`. Historical experimental results remain under
`sign_check_atlas/results*` and are not release artifacts.

## Release status and limitations

- Version `0.1.0` is an alpha research preview.
- The committed Qwen profile does not specify a frozen base-model revision yet.
- The supplied metric is measured on calibration data, not a locked held-out set.
- Runtime overhead has not been independently benchmarked in this repository.
- English-language and adversarial coverage is incomplete.
- False positives and false negatives are expected.
- The GGUF runtime patch is conceptual.

See [docs/RELEASE_CHECKLIST.md](docs/RELEASE_CHECKLIST.md) for the remaining work
before a production-oriented release.

New to publishing a project? Follow the step-by-step
[release guide](docs/RELEASING.md). It covers the GitHub pre-release, Zenodo DOI,
Hugging Face profile, and eventual PyPI publication.

## Citation

GitHub exposes the repository's `CITATION.cff` through its **Cite this
repository** control. Until a DOI-backed release or paper is available, cite the
versioned software directly:

```bibtex
@software{cappelli_atlas_nano_2026,
  author  = {David Cappelli},
  title   = {Atlas Nano},
  version = {0.1.0},
  year    = {2026},
  url     = {https://github.com/VecPLabs/Atlas-Nano}
}
```

## License

Software is licensed under Apache-2.0. Data and artifact licensing is documented
in [DATA_LICENSES.md](DATA_LICENSES.md). Attribution does not imply endorsement.

## Security

Do not include undisclosed vulnerabilities or sensitive prompts in public issues.
See [SECURITY.md](SECURITY.md) for reporting guidance.
