# Release checklist

## Completed for v0.1 preparation

- [x] Identify the project as an alpha research preview.
- [x] Separate system, profile, and runtime-integration concepts.
- [x] Add a machine-readable profile schema and compatibility validator.
- [x] Bind the included Qwen artifact to its actual calibration metrics.
- [x] Replace general performance claims with qualified measurements.
- [x] Document intended use, limitations, and security reporting.
- [x] Add an explicit repository rights notice.
- [x] Compute SHA-256 hashes for the included Qwen profile artifacts.
- [x] Select Apache-2.0 for software and CC BY 4.0 for Atlas-authored data.
- [x] Add machine-readable citation metadata and attribution guidance.
- [x] Move pipeline implementations into the installable package.
- [x] Bundle the default Atlas-authored training gauntlet in the wheel.

## Required before publishing v0.1

- [ ] Have counsel review third-party dataset redistribution terms if the mixed
  evaluation files will be published as standalone datasets.
- [ ] Record the exact Qwen base-model revision, tokenizer revision, runtime,
  Transformers version, precision, seed, and hardware.
- [ ] Reconstruct per-row provenance for the two mixed evaluation gauntlets.
- [ ] Create a fresh locked held-out split with no threshold tuning.
- [ ] Run category, hard-negative, multilingual, and adaptive-attack evaluation.
- [ ] Reproduce the result from a clean environment.
- [ ] Tag the release, archive it with Zenodo, and add the DOI to `CITATION.cff`.
- [ ] Pin supported dependency ranges after testing the clean environment matrix.
- [x] Add CI for Python 3.10-3.12, CLI smoke tests, profile tests, and GGUF tests.
- [x] Decide on an open-source release.

## Required before claiming runtime support

- [ ] Implement against a pinned `llama.cpp` commit.
- [ ] Add end-to-end tests using the target quantization.
- [ ] Specify which token activation is checked and how multi-sequence batches are
  handled.
- [ ] Benchmark overhead on named hardware using a published methodology.
- [ ] Define signal reset, boundary, fail-closed, and API semantics.

## Suggested distribution

- GitHub Releases: source, changelog, checksums, and signed provenance.
- PyPI: `atlas-nano` CLI and Python package.
- Hugging Face: one repository per supported profile with a system/model card.
- OCI image: reproducible demonstration server after the runtime is validated.
