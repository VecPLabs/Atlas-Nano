# Contributing

Thank you for helping improve Atlas Nano. Bug reports, reproducibility results,
documentation corrections, and focused code changes are welcome.

## Development setup

```bash
python -m venv .venv
python -m pip install -e ".[dev,gguf]"
python -m unittest discover -s tests -v
python tests/test_gguf_integration.py --dim 2560
```

GPU-dependent changes should include the exact model revision, tokenizer,
hardware, precision, dependency versions, command, and result artifact.

## Pull requests

- Keep changes scoped and explain user-visible behavior.
- Add tests for compatibility, packaging, or metric logic where practical.
- Do not weaken fail-closed profile checks.
- Do not add benchmark prompts without source and license metadata.
- Do not describe calibration data as held-out evaluation.

Unless explicitly stated otherwise, submitted code contributions are licensed
under Apache-2.0 as described in section 5 of that license. Contributions to
datasets must identify their provenance and the contributor's authority to
license them.

Sensitive security reports should follow [SECURITY.md](SECURITY.md), not a
public issue.
