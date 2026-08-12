"""Local release preflight for Atlas Nano."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles" / "qwen3-4b-signcheck-v1" / "profile.json"
FORBIDDEN_RELEASE_PHRASES = (
    "patent pending",
    "uspto 63/931,565",
    "proprietary - patent pending",
    "all rights reserved",
)


def run(*args: str) -> None:
    print("+", " ".join(args))
    subprocess.run(args, cwd=ROOT, check=True)


def read_version() -> str:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    if not match:
        raise RuntimeError("could not find project version in pyproject.toml")
    return match.group(1)


def check_version(version: str) -> None:
    package_init = (ROOT / "atlas_nano" / "__init__.py").read_text(encoding="utf-8")
    if f'__version__ = "{version}"' not in package_init:
        raise RuntimeError("atlas_nano.__version__ does not match pyproject.toml")
    if f"## {version} -" not in (ROOT / "CHANGELOG.md").read_text(encoding="utf-8"):
        raise RuntimeError("CHANGELOG.md has no section for the project version")
    if f"version: {version}" not in (ROOT / "CITATION.cff").read_text(encoding="utf-8"):
        raise RuntimeError("CITATION.cff version does not match pyproject.toml")


def check_release_language() -> None:
    suffixes = {".py", ".md", ".toml", ".json", ".cff", ".txt", ".h", ".diff"}
    violations: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if ".git" in path.parts or path.name == "LICENSE" or path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8", errors="replace").casefold()
        for phrase in FORBIDDEN_RELEASE_PHRASES:
            if phrase in text:
                violations.append(f"{path.relative_to(ROOT)}: {phrase}")
    if violations:
        raise RuntimeError("stale release language:\n  " + "\n  ".join(violations))


def check_profile() -> None:
    sys.path.insert(0, str(ROOT))
    from atlas_nano.profile import load_profile, verify_artifacts

    profile = load_profile(PROFILE)
    verify_artifacts(profile, PROFILE.parent)


def check_git_clean(allow_dirty: bool) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout
    if status.strip() and not allow_dirty:
        raise RuntimeError("worktree is dirty; commit the release or use --allow-dirty while preparing")


def build_and_check_wheel(version: str) -> None:
    run(sys.executable, "-m", "build")
    wheels = sorted((ROOT / "dist").glob(f"atlas_nano-{version}-*.whl"))
    if not wheels:
        raise RuntimeError("build produced no matching wheel")
    required = {
        "atlas_nano/data/gauntlet_v3_corrected.txt",
        "atlas_nano/pipeline/training.py",
        f"atlas_nano-{version}.dist-info/licenses/LICENSE",
        f"atlas_nano-{version}.dist-info/licenses/NOTICE",
    }
    with zipfile.ZipFile(wheels[-1]) as wheel:
        names = set(wheel.namelist())
    missing = sorted(required - names)
    stale = sorted(
        name for name in names
        if name.startswith("vecp_") or name == "sign_check_atlas/gguf_integration/test_gguf.py"
    )
    if missing or stale:
        raise RuntimeError(f"invalid wheel contents; missing={missing}, stale={stale}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-dirty", action="store_true", help="allow uncommitted preparation changes")
    parser.add_argument("--build", action="store_true", help="also build and inspect wheel/sdist")
    args = parser.parse_args()

    version = read_version()
    check_version(version)
    check_release_language()
    check_profile()
    check_git_clean(args.allow_dirty)
    run(sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v")
    run(sys.executable, "tests/test_gguf_integration.py", "--dim", "2560")
    run("git", "diff", "--check")
    if args.build:
        build_and_check_wheel(version)
    print(f"\nAtlas Nano {version} release preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
