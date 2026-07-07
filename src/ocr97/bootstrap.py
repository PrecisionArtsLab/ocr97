from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from . import diagnostics as diag

REPO_ROOT = Path(__file__).resolve().parents[2]


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _module_version(name: str) -> str:
    package = diag.MODULE_PACKAGES.get(name, name)
    return diag.package_version(package)


def _module_available(name: str) -> bool:
    return diag.module_available(name)


def _pip_install_requirements(requirements_path: Path, constraints_path: Path) -> Dict[str, Any]:
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)]
    if constraints_path.exists():
        cmd.extend(["-c", str(constraints_path)])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "ok": proc.returncode == 0,
        "command": cmd,
        "stdout": (proc.stdout or "")[-8000:],
        "stderr": (proc.stderr or "")[-8000:],
        "returncode": proc.returncode,
    }


def _parse_pins(path: Path) -> Dict[str, str]:
    pins: Dict[str, str] = {}
    if not path.exists():
        return pins
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            continue
        name, version = line.split("==", 1)
        key = str(name or "").strip().lower().replace("-", "_")
        value = str(version or "").strip()
        if key and value:
            pins[key] = value
    return pins


def _installed_version(package_name: str) -> str:
    try:
        return str(importlib_metadata.version(package_name))
    except Exception:
        return ""


def _maybe_download_model(model_id: str, target_dir: Path) -> Dict[str, Any]:
    model_id = str(model_id or "").strip()
    if not model_id:
        return {"ok": False, "reason": "model_id_unset"}
    if target_dir.exists() and any(target_dir.iterdir()):
        return {"ok": True, "reason": "already_present", "target_dir": str(target_dir)}
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception:
        return {"ok": False, "reason": "huggingface_hub_unavailable", "target_dir": str(target_dir)}
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        return {"ok": True, "reason": "downloaded", "target_dir": str(target_dir)}
    except Exception as exc:
        return {"ok": False, "reason": f"download_failed:{type(exc).__name__}:{exc}", "target_dir": str(target_dir)}


def _engine_row(name: str, status: Dict[str, Any], module_name: str, model_id: str, model_dir: Path) -> Dict[str, Any]:
    return {
        "name": name,
        "module": module_name,
        "module_version": _module_version(module_name) if _module_available(module_name) else "",
        "model_id": model_id,
        "model_dir": str(model_dir),
        "model_hash": diag.hash_dir_metadata(model_dir),
        "status": status,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap/check OCR97 local OCR and optional heavy lanes.")
    parser.add_argument("--install", action="store_true", help="Install OCR dependencies from requirements-ocr.txt.")
    parser.add_argument("--download-models", action="store_true", help="Download model assets (explicit opt-in).")
    parser.add_argument("--check-only", action="store_true", help="Compatibility alias. Equivalent to default behavior.")
    parser.add_argument("--ci", action="store_true", help="Fail hard when pinned critical dependency versions drift.")
    parser.add_argument(
        "--requirements-file",
        default=str(REPO_ROOT / "requirements-ocr.txt"),
        help="Path to OCR requirements file.",
    )
    parser.add_argument(
        "--constraints-file",
        default=str(REPO_ROOT / "constraints-ocr.txt"),
        help="Path to OCR constraints file.",
    )
    parser.add_argument("--instance-name", default="ocr97", help="Instance name for install metadata location.")
    parser.add_argument("--output", default="", help="Optional explicit metadata output path.")
    args = parser.parse_args()
    check_only = bool(args.check_only or not args.install)
    requirements_path = Path(args.requirements_file).expanduser()
    constraints_path = Path(args.constraints_file).expanduser()

    local_package_map = {
        "tesseract": ("pytesseract", "pytesseract"),
        "rapidocr": ("rapidocr_onnxruntime", "rapidocr-onnxruntime"),
        "native_pdf_text": ("fitz", "pymupdf"),
        "local_image_preprocessed_best": ("pytesseract", "pytesseract"),
    }
    optional_package_map = {
        "gb10_paddleocr_vl": ("paddleocr", "paddleocr"),
        "mineru2_5": ("mineru", "mineru"),
        "olmocr2": ("olmocr", "olmocr"),
    }

    install_action: Dict[str, Any] = {
        "mode": "check_only" if check_only else "install",
        "requirements_file": str(requirements_path),
        "constraints_file": str(constraints_path),
        "executed": False,
    }
    if not check_only:
        if not requirements_path.exists():
            install_action.update({"ok": False, "error": "requirements_file_missing"})
        else:
            install_action.update({"executed": True, "result": _pip_install_requirements(requirements_path, constraints_path)})
    else:
        install_action.update({"ok": True, "reason": "check_only_default"})

    model_downloads: Dict[str, Any] = {}
    if args.download_models:
        model_downloads["mineru2_5"] = _maybe_download_model(diag.model_id("mineru2_5"), diag.model_dir("mineru2_5"))
        model_downloads["olmocr2"] = _maybe_download_model(diag.model_id("olmocr2"), diag.model_dir("olmocr2"))
        paddle_dir = diag.model_dir("gb10_paddleocr_vl")
        paddle_dir.mkdir(parents=True, exist_ok=True)
        model_downloads["gb10_paddleocr_vl"] = {"ok": True, "reason": "runtime_managed", "target_dir": str(paddle_dir)}

    install_actions: Dict[str, Any] = {}
    for engine_name, (module_name, package_name) in {**local_package_map, **optional_package_map}.items():
        install_actions[engine_name] = {
            "module": module_name,
            "package": package_name,
            "installed": _module_available(module_name),
        }

    critical_constraints = _parse_pins(constraints_path)
    local_critical_names = ["pillow"]
    optional_critical_names = ["rich", "huggingface_hub", "numpy"]
    critical_names = [*local_critical_names, *optional_critical_names]
    critical_versions: Dict[str, Any] = {}
    local_version_conflicts: list[str] = []
    optional_version_conflicts: list[str] = []
    for pkg in critical_names:
        pin = str(critical_constraints.get(pkg.replace("-", "_"), "")).strip()
        installed = _installed_version(pkg)
        critical_versions[pkg] = {"pinned": pin, "installed": installed}
        if pin and installed and installed != pin:
            conflict = f"{pkg}:{installed}!={pin}"
            if pkg in local_critical_names:
                local_version_conflicts.append(conflict)
            else:
                optional_version_conflicts.append(conflict)

    local_status = {engine: diag.engine_status(engine) for engine in local_package_map}
    optional_status = {
        "gb10_paddleocr_vl": diag.engine_status("gb10_paddleocr_vl"),
        "mineru2_5": diag.engine_status("mineru2_5"),
        "olmocr2": diag.engine_status("olmocr2"),
    }

    metadata = {
        "updated_at": _utc_iso(),
        "source": "ocr97_bootstrap",
        "python": sys.version.split(" ")[0],
        "check_only": bool(check_only),
        "install_requested": bool(args.install),
        "download_models": bool(args.download_models),
        "release_profile": "portable_local_default",
        "requirements": {
            "requirements_file": str(requirements_path),
            "constraints_file": str(constraints_path),
            "constraints_present": constraints_path.exists(),
        },
        "install_action": install_action,
        "install_actions": install_actions,
        "model_downloads": model_downloads,
        "critical_dependency_versions": critical_versions,
        "local_version_conflicts": local_version_conflicts,
        "optional_version_conflicts": optional_version_conflicts,
        "critical_version_conflicts": [*local_version_conflicts, *optional_version_conflicts],
        "local_engines": {
            name: _engine_row(name, status, local_package_map[name][0], "", diag.model_dir(name))
            for name, status in local_status.items()
        },
        "optional_engines": {
            name: _engine_row(name, status, optional_package_map[name][0], diag.model_id(name), diag.model_dir(name))
            for name, status in optional_status.items()
        },
    }
    metadata["engines"] = {**metadata["local_engines"], **metadata["optional_engines"]}
    metadata["ok"] = bool(not local_version_conflicts and all(bool((row or {}).get("status", {}).get("ready")) for row in metadata["local_engines"].values()))
    metadata["optional_ready"] = all(bool((row or {}).get("status", {}).get("ready")) for row in metadata["optional_engines"].values())

    default_output = diag.model_dir("ocr97").parent / f"{args.instance_name.lower()}_ocr_install_metadata.json"
    output_path = Path(args.output).expanduser() if args.output else default_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    summary = {k: bool(v["status"].get("ready")) for k, v in metadata["local_engines"].items()}
    optional_summary = {k: bool(v["status"].get("ready")) for k, v in metadata["optional_engines"].items()}
    print(json.dumps({"ok": bool(metadata.get("ok")), "optional_ready": bool(metadata.get("optional_ready")), "output": str(output_path), "summary": summary, "optional_summary": optional_summary, "local_version_conflicts": local_version_conflicts, "optional_version_conflicts": optional_version_conflicts, "critical_version_conflicts": [*local_version_conflicts, *optional_version_conflicts]}, indent=2))
    if args.ci and [*local_version_conflicts, *optional_version_conflicts]:
        return 2
    return 0 if bool(metadata.get("ok")) or check_only else 1


if __name__ == "__main__":
    raise SystemExit(main())

