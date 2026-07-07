"""Stage a clean OCR97 public repository candidate.

The active OCR97 working directory contains runtime artifacts and internal run
history. This script copies the public product surface into a sibling directory
that can be reviewed before creating a standalone GitHub repository.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = ROOT.parent / "ocr97-public"

DIRECTORIES = [
    ".github",
    "benchmarks",
    "examples",
    "src",
    "tests",
]

FILES = [
    ".gitignore",
    "LICENSE",
    "MANIFEST.in",
    "PUBLIC_README.md",
    "constraints-ocr.txt",
    "pyproject.toml",
    "requirements-ocr.txt",
]

PUBLIC_TOOL_FILES = [
    "ocr97_public_release_audit.py",
    "stage_ocr97_public_repo.py",
]

PUBLIC_DOC_FILES = [
    "BENCHMARKS.md",
    "GITHUB_PUBLICATION_NOTES.md",
    "ROADMAP.md",
]

IGNORED_NAMES = {
    ".pytest_cache",
    "__pycache__",
    "artifacts",
    "build",
    "dist",
    "_tmp",
}

IGNORED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".log",
}


def ignore_names(_dir: str, names: list[str]) -> set[str]:
    ignored = set()
    for name in names:
        if name in IGNORED_NAMES:
            ignored.add(name)
            continue
        if Path(name).suffix.lower() in IGNORED_SUFFIXES:
            ignored.add(name)
    return ignored


def remove_managed_target_paths(target: Path) -> None:
    managed = set(DIRECTORIES + FILES + ["README.md", "docs", "tools"] + list(IGNORED_NAMES))
    for name in managed:
        path = target / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

    for path in list(target.rglob("*")):
        if path.is_dir() and path.name in IGNORED_NAMES:
            shutil.rmtree(path)


def copy_tree(source: Path, destination: Path) -> None:
    if source.exists():
        shutil.copytree(source, destination, ignore=ignore_names)


def stage(target: Path) -> dict:
    target = target.resolve()
    if target == ROOT.resolve():
        raise ValueError("target must not be the active OCR97 working directory")
    target.mkdir(parents=True, exist_ok=True)
    remove_managed_target_paths(target)

    for directory in DIRECTORIES:
        copy_tree(ROOT / directory, target / directory)

    docs_dir = target / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    for filename in PUBLIC_DOC_FILES:
        source = ROOT / "docs" / filename
        if source.exists():
            shutil.copy2(source, docs_dir / filename)

    for filename in FILES:
        source = ROOT / filename
        if not source.exists():
            continue
        destination_name = "README.md" if filename == "PUBLIC_README.md" else filename
        shutil.copy2(source, target / destination_name)

    tools_dir = target / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    for filename in PUBLIC_TOOL_FILES:
        source = ROOT / "tools" / filename
        if source.exists():
            shutil.copy2(source, tools_dir / filename)

    copied_files = sum(1 for path in target.rglob("*") if path.is_file())
    return {
        "target": str(target),
        "copied_files": copied_files,
        "readme": str(target / "README.md"),
        "audit_command": f"python tools/ocr97_public_release_audit.py --root {target}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage a clean OCR97 public repo candidate")
    parser.add_argument("--target", default=str(DEFAULT_TARGET), help="destination directory")
    args = parser.parse_args()
    result = stage(Path(args.target))
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
