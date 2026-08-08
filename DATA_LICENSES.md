# Data and artifact licensing

Software source code in this repository is licensed under Apache-2.0. This file
documents material that has a different or mixed provenance.

## Atlas-authored material

The following material, to the extent owned by David Cappelli / VecP Labs LLC,
is licensed under the Creative Commons Attribution 4.0 International license
(CC BY 4.0):

- `atlas_nano/data/gauntlet_v3_corrected.txt`
- Atlas-authored prompts within `data/evaluation/gauntlet_TEST_enhanced.txt`
- release manifests and artifacts under `profiles/`
- generated evaluation reports under `sign_check_atlas/results*`

Preferred attribution: “Atlas Nano, David Cappelli / VecP Labs LLC,” with a
link to `https://github.com/VecPLabs/Atlas-Nano`. For scholarly work, please use
`CITATION.cff`.

## OR-Bench material

`data/evaluation/gauntlet_TEST_enhanced.txt` contains prompts derived from
OR-Bench-Hard-1K.
OR-Bench is distributed under CC BY 4.0. Those prompts remain subject to the
OR-Bench license and should cite:

> OR-Bench: An Over-Refusal Benchmark for Large Language Models.

Source: https://huggingface.co/datasets/bench-llm/or-bench

The current combined file does not carry reliable per-row source metadata. Treat
the entire file as mixed-provenance until that metadata is reconstructed.

## HarmBench and related benchmark material

`data/evaluation/gauntlet_HARMBENCH_CLEANED.txt` contains prompts sourced or
adapted from HarmBench and related benchmark sets identified by row prefixes.
HarmBench's official repository is distributed under the MIT License:

- Source: https://github.com/centerforaisafety/HarmBench
- Paper: https://arxiv.org/abs/2402.04249

Atlas Nano does not relicense third-party prompts. Their original terms and
attribution requirements continue to apply. The exact source and transformation
of every row should be audited before republishing this combined file as a
standalone dataset.

## No endorsement

Attribution does not imply that OR-Bench, HarmBench, their authors, or their
institutions endorse Atlas Nano.
