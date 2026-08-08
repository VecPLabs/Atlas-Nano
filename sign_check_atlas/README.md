# Sign-Check Atlas

Sign-Check Atlas is the experimental Tier 1 component of Atlas Nano. It projects
an internal model activation onto a calibrated energy axis and compares the
result with a threshold. It is model-coupled and should route uncertain or
flagged cases to a stronger policy layer.

## Release status

The Python sidecar reader/writer and research scripts are implemented. The
`llama_cpp_patch/` directory is a conceptual integration reference and is not a
tested patch for a pinned upstream revision. Runtime overhead has not yet been
benchmarked in this repository.

The included Qwen3-4B threshold measured F1 0.8952, precision 0.9485, and recall
0.8476 on the same 1,180-example gauntlet used for calibration. This is not
held-out evidence. See `../profiles/qwen3-4b-signcheck-v1/profile.json` and
`../docs/EVALUATION.md`.

## Design

```text
activation at configured component/layer
                |
                v
energy = dot(activation, energy_axis)
                |
                +-- below pass threshold --> continue
                +-- boundary zone ---------> route
                +-- above flag threshold --> route
```

The profile must match the exact base model, architecture, hidden dimension,
component, layer, and threshold calibration. Validate it before loading:

```bash
atlas-nano profile validate profiles/qwen3-4b-signcheck-v1/profile.json \
  --model Qwen/Qwen3-4B --architecture Qwen3ForCausalLM --hidden-dim 2560
```

## Research pipeline

```bash
atlas-nano sign-check --output-dir sign_check_atlas/results

atlas-nano gguf \
  --phase1-results sign_check_atlas/results/phase1_validation.json \
  --phase3-results sign_check_atlas/results/phase3_threshold.json \
  --output model_safety.gguf
```

Injecting data into a GGUF file requires the optional `gguf` dependency. Merely
embedding metadata does not make an unmodified runtime enforce or expose the
safety signal.

## Tiered use

1. Sign-Check supplies a low-cost routing signal.
2. Full Atlas or another classifier reviews flags and boundary cases.
3. Application policy chooses whether to continue, warn, block, or seek review.

Do not use Sign-Check as the sole safety control. False positives and false
negatives are expected, especially under distribution shift or adaptive attack.

Historical `results*` directories are research experiments. Only a manifest
under `../profiles/` represents a release profile.
