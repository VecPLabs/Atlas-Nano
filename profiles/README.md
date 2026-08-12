# Safety profiles

A profile binds Atlas artifacts and metrics to an exact model contract. Profiles
are not portable weights: changing the base model, model revision, architecture,
hidden dimension, extraction component, layer, tokenizer behavior, or runtime
may invalidate the calibration.

Each release profile lives in its own directory and contains `profile.json` plus
any separately distributed artifacts and hashes. Validate a manifest with:

```bash
atlas-nano profile validate profiles/qwen3-4b-signcheck-v1/profile.json
```

Validation also checks every listed local artifact against its SHA-256 digest.

For production use, `base_model_revision` and artifact SHA-256 hashes must be
populated. The v0.1 manifest leaves the revision unset because it was not recorded
during the original experiment; that is an explicit release limitation.
