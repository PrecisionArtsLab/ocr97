"""Audit OCR97 public-release readiness.

This checks the files intended for a standalone public repository. It is
conservative by design: failing the audit means "review before publishing," not
necessarily that the package cannot run locally.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable


DEFAULT_ROOT = Path(__file__).resolve().parents[1]

TEXT_SUFFIXES = {".md", ".py", ".json", ".toml", ".txt", ".in", ".yml", ".yaml"}

GENERATED_DIR_NAMES = {
    ".pytest_cache",
    "__pycache__",
    "artifacts",
    "build",
    "dist",
    "_tmp",
}

FORBIDDEN_PATTERNS = {
    "absolute_user_path": re.compile(r"C:\\Users\\[^\\\s`]+", re.IGNORECASE),
    "env_secret_assignment": re.compile(
        r"(?i)(api[_-]?key|token|password|secret)\s*=\s*['\"][^'\"]{6,}"
    ),
    "private_env_file": re.compile(r"(^|[\\/])\.env($|[\\/])", re.IGNORECASE),
    "precision_internal": re.compile(r"precisionartslab|percisionartslab|louisa", re.IGNORECASE),
    "live_account_workflow": re.compile(r"\bebay\b|\bwhatsapp\b|\bgarmin\b", re.IGNORECASE),
}

MISLEADING_CLAIM_PATTERNS = {
    "unqualified_97_accuracy": re.compile(
        r"(?i)(?:97\s*%|97/100).{0,80}(?:accuracy|accurate|grade|score)"
    ),
}

REQUIRED_COMMON = [
    "LICENSE",
    "pyproject.toml",
    "MANIFEST.in",
    "src/ocr97/__init__.py",
    "src/ocr97/diagnostics.py",
    "tests/test_package_smoke.py",
    "benchmarks/truth10_manifest.json",
    "examples/doctor_payload_demo.py",
    "tools/ocr97_public_release_audit.py",
]


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def iter_text_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        parts = set(path.relative_to(root).parts)
        if parts & GENERATED_DIR_NAMES:
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {"MANIFEST.in"}:
            yield path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def required_files(root: Path) -> list[str]:
    required = list(REQUIRED_COMMON)
    if (root / "README.md").exists():
        required.append("README.md")
    else:
        required.append("PUBLIC_README.md")
    return required


def audit(root: Path) -> dict:
    root = root.resolve()
    missing = [item for item in required_files(root) if not (root / item).exists()]
    generated_dirs = [
        rel(path, root)
        for path in root.rglob("*")
        if path.is_dir() and path.name in GENERATED_DIR_NAMES
    ]

    findings = []
    scanned = 0
    for path in iter_text_files(root):
        scanned += 1
        text = read_text(path)
        path_rel = rel(path, root)

        if path_rel != "tools/ocr97_public_release_audit.py":
            for kind, pattern in FORBIDDEN_PATTERNS.items():
                if pattern.search(text):
                    findings.append(
                        {
                            "file": path_rel,
                            "kind": kind,
                            "detail": "public candidate contains private, local, or live-account wording",
                        }
                    )

        if path.name in {"README.md", "PUBLIC_README.md"}:
            for kind, pattern in MISLEADING_CLAIM_PATTERNS.items():
                if pattern.search(text) and "not a blanket claim" not in text.lower():
                    findings.append(
                        {
                            "file": path_rel,
                            "kind": kind,
                            "detail": "OCR97 quality claims must be tied to measured benchmark evidence",
                        }
                    )
            if "Native Desktop And VM Use" not in text:
                findings.append(
                    {
                        "file": path_rel,
                        "kind": "native_desktop_section_missing",
                        "detail": "public README must explain native desktop and VM use",
                    }
                )
            if "not a blanket claim" not in text.lower():
                findings.append(
                    {
                        "file": path_rel,
                        "kind": "ocr97_name_boundary_missing",
                        "detail": "public README must state OCR97 is not an unqualified 97% accuracy claim",
                    }
                )

    ok = not missing and not findings and not generated_dirs
    return {
        "ok": ok,
        "root": str(root),
        "scanned_files": scanned,
        "missing_required_files": missing,
        "generated_dirs_present": generated_dirs,
        "findings": findings,
        "release_label": "v0.1.0-alpha",
        "next_action": (
            "Ready for owner review and optional standalone GitHub initialization."
            if ok
            else "Fix findings, remove generated directories, then rerun the audit."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit OCR97 public-release readiness")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="repo root to audit")
    args = parser.parse_args()
    result = audit(Path(args.root))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
