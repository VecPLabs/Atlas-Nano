# Releasing Atlas Nano

This guide assumes no prior release-management experience. Version `0.1.0`
should be published as a **GitHub pre-release / research preview**, not as a
production release.

## What each service is for

- **GitHub Release:** canonical source release, notes, and downloadable files.
- **Zenodo:** permanent archive and DOI so Atlas Nano can be cited reliably.
- **Hugging Face:** discoverable home for each model-specific profile artifact.
- **PyPI:** convenient Python installation. Publish here after the GitHub tag has
  passed CI and a clean installation test.

None of these services establishes that Atlas Nano is production-safe. The model
card and evaluation report define what the release actually supports.

## 1. Prepare the release branch

Do not commit the current release work directly to `main`. From the repository:

```bash
git switch -c codex/release-v0.1.0
python scripts/release_check.py --allow-dirty
git diff --check
git status --short
```

Review the changes, especially `LICENSE`, `DATA_LICENSES.md`, `MODEL_CARD.md`,
and the reported metrics. Then commit:

```bash
git add -A
git commit -m "Prepare Atlas Nano v0.1.0 research preview"
git push -u origin codex/release-v0.1.0
```

Open a pull request into `main`. Confirm that CI passes on Python 3.10, 3.11,
and 3.12. Merge the pull request without changing the prepared version.

## 2. Run the final preflight on `main`

After merging:

```bash
git switch main
git pull --ff-only
python -m pip install -e ".[dev,gguf]"
python scripts/release_check.py --build
```

The command refuses to approve a dirty worktree, runs the lightweight test
suite and GGUF round trips, validates the release profile and checksums, builds
the wheel and source distribution, and checks the wheel layout. Release files
are written to `dist/`.

## 3. Tag and create the GitHub pre-release

Create an annotated tag only after the final preflight passes:

```bash
git tag -a v0.1.0 -m "Atlas Nano v0.1.0 research preview"
git push origin v0.1.0
```

In GitHub, open **Releases → Draft a new release**:

- Choose tag `v0.1.0`.
- Title it `Atlas Nano v0.1.0 — Research Preview`.
- Select **Set as a pre-release**.
- Copy the `0.1.0` section from `CHANGELOG.md` into the notes.
- Upload the wheel and source archive from `dist/`.
- State prominently that the Qwen metric is a calibration result, the runtime
  patch is conceptual, and Atlas is not a sole safety control.

If GitHub CLI is installed, the equivalent command is:

```bash
gh release create v0.1.0 dist/* \
  --title "Atlas Nano v0.1.0 — Research Preview" \
  --notes-file CHANGELOG.md \
  --prerelease
```

## 4. Archive with Zenodo

Connect the GitHub repository through Zenodo and enable Atlas Nano before
publishing the GitHub release. Zenodo will archive the tagged release and assign
a version DOI plus a concept DOI.

The first tag cannot contain a DOI that does not exist yet. After Zenodo creates
it:

1. Add the **concept DOI** to `CITATION.cff` so it remains valid across versions.
2. Add the Zenodo badge and DOI citation to `README.md`.
3. Commit that metadata to `main`; include it in the next tag.

## 5. Publish the Qwen profile on Hugging Face

Create a model repository such as `VecPLabs/atlas-nano-qwen3-4b-signcheck-v1`.
Upload only the contents of `profiles/qwen3-4b-signcheck-v1/` plus a card derived
from `MODEL_CARD.md`.

The card must include:

- exact compatible base model and revision;
- extraction component, layer, and hidden dimension;
- calibration-versus-held-out status of every metric;
- Apache-2.0 software and CC BY 4.0 artifact terms;
- Atlas Nano and upstream dataset citations;
- artifact SHA-256 checksums;
- a warning that the profile is not interchangeable across models.

Do not publish the mixed evaluation gauntlets as a Hugging Face dataset until
their row-level provenance has been reconstructed.

## 6. Publish to PyPI later

PyPI is intentionally last. First configure a PyPI project and a GitHub Actions
trusted publisher; avoid storing a long-lived API token in the repository.

Test on TestPyPI, install the wheel into a clean environment, and run:

```bash
atlas-nano --version
atlas-nano profile validate path/to/profile.json
```

Then publish the exact distributions created from tag `v0.1.0`. Never rebuild a
tagged version after publication; bump the version for any correction.

## If something goes wrong

- Before pushing a tag: fix the issue normally and rerun preflight.
- After pushing but before publishing a release: delete and recreate the tag only
  if nobody could reasonably depend on it.
- After publishing: do not replace files silently. Document the problem, mark the
  release accordingly, fix it, and publish `0.1.1`.
