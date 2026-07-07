from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import importlib.util
import os
import shutil
from pathlib import Path
from typing import Any, Dict

from .engine_registry import (
    engine_assets_required,
    engine_model_dir,
    engine_model_id,
    engine_module_name,
    engine_names,
    engine_package_name,
    engine_ready_override_env,
    engine_is_optional,
)
from .paths import ensure_paths
from .profiles import (
    AUTO_HARDWARE_PROFILE,
    CPU_HARDWARE_PROFILE,
    LOCAL_GPU_HARDWARE_PROFILE,
    REMOTE_MODEL_HARDWARE_PROFILE,
    WORKSTATION_HARDWARE_PROFILE,
    requested_hardware_profile,
)


MODULE_PACKAGES = {
    "flask": "flask",
    "requests": "requests",
    "pytesseract": "pytesseract",
    "rapidocr_onnxruntime": "rapidocr-onnxruntime",
    "fitz": "pymupdf",
    "torch": "torch",
    "transformers": "transformers",
    "paddleocr": "paddleocr",
    "mineru": "mineru",
    "olmocr": "olmocr",
}


def module_available(module_name: str) -> bool:
    if not module_name:
        return False
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def package_version(package_name: str) -> str:
    try:
        return str(importlib_metadata.version(package_name))
    except Exception:
        return ""


def module_status(module_name: str, package_name: str = "") -> Dict[str, Any]:
    package = package_name or MODULE_PACKAGES.get(module_name, module_name)
    available = module_available(module_name)
    return {
        "module": module_name,
        "package": package,
        "available": available,
        "version": package_version(package) if available else "",
    }


def model_id(engine: str) -> str:
    return engine_model_id(engine)


def model_dir(engine: str) -> Path:
    return engine_model_dir(engine)


def has_model_assets(path: Path) -> bool:
    try:
        if not path.exists():
            return False
        return any(True for _ in path.iterdir())
    except Exception:
        return False


def hash_dir_metadata(root: Path, *, limit: int = 3000) -> str:
    if not root.exists():
        return ""
    dig = hashlib.sha256()
    seen = 0
    for item in sorted(root.rglob("*")):
        if not item.is_file():
            continue
        try:
            stat = item.stat()
        except Exception:
            continue
        dig.update(str(item.relative_to(root)).encode("utf-8", errors="ignore"))
        dig.update(str(stat.st_size).encode("utf-8"))
        dig.update(str(int(stat.st_mtime)).encode("utf-8"))
        seen += 1
        if seen >= limit:
            break
    return dig.hexdigest()[:24]


def engine_status(engine: str) -> Dict[str, Any]:
    override_env = engine_ready_override_env(engine)
    override = str(os.getenv(override_env, "")).strip().lower()
    module_name = engine_module_name(engine)
    module_ready = module_available(module_name) if module_name else False
    path = model_dir(engine)
    assets_required = engine_assets_required(engine)
    assets_ready = has_model_assets(path) if assets_required else True
    if override in {"1", "true", "yes"}:
        ready = True
        reason = "override_ready"
    elif override in {"0", "false", "no"}:
        ready = False
        reason = "override_not_ready"
    elif engine == "gb10_qwen_ocr":
        ready = remote_model_configured()
        reason = "remote_model_endpoint_configured" if ready else "remote_model_endpoint_missing"
    elif engine in {"local_image_best", "local_image_preprocessed_best"}:
        rapid_ready = module_available("rapidocr_onnxruntime")
        tess_ready = module_available("pytesseract")
        pil_ready = module_available("PIL")
        preprocess_ready = True if engine == "local_image_best" else pil_ready
        ready = bool((rapid_ready or tess_ready) and preprocess_ready)
        reason = "module_ready" if ready else ("pillow_missing" if not preprocess_ready else "module_or_assets_missing")
    else:
        ready = bool(module_ready and assets_ready)
        reason = "module_ready" if ready and not assets_required else ("module_and_assets_ready" if ready else "module_or_assets_missing")
    return {
        "ready": ready,
        "reason": reason,
        "module": module_name,
        "module_available": module_ready,
        "model_id": model_id(engine),
        "model_dir": str(path),
        "model_assets_present": assets_ready,
        "runtime_loaded": False,
        "diagnostic_mode": "lightweight_no_model_import",
    }


def remote_model_configured() -> bool:
    keys = {
        "OCR97_REMOTE_OCR_GATEWAY_URL",
        "OCR97_OLLAMA_URL",
        "OLLAMA_URL",
        "OLLAMA_HOST",
        "OCR97_GB10_OCR_ENABLED",
        "OCR97_GB10_OCR_USE_GATEWAY",
    }
    return any(str(os.getenv(key, "")).strip() for key in keys)


def local_gpu_hint() -> Dict[str, Any]:
    cuda_visible = str(os.getenv("CUDA_VISIBLE_DEVICES", "")).strip()
    nvidia_smi = shutil.which("nvidia-smi")
    gpu_disabled = cuda_visible in {"-1", "none", "None"}
    return {
        "nvidia_smi_available": bool(nvidia_smi),
        "cuda_visible_devices": cuda_visible,
        "gpu_disabled_by_env": gpu_disabled,
        "available": bool(nvidia_smi and not gpu_disabled),
    }


def hardware_scaling_payload(engines: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    requested = requested_hardware_profile()
    gpu = local_gpu_hint()
    remote_ready = remote_model_configured()

    if requested != AUTO_HARDWARE_PROFILE:
        effective = requested
        reason = "explicit_OCR97_HARDWARE_PROFILE"
    elif remote_ready:
        effective = REMOTE_MODEL_HARDWARE_PROFILE
        reason = "remote_model_endpoint_configured"
    elif gpu["available"]:
        effective = LOCAL_GPU_HARDWARE_PROFILE
        reason = "local_gpu_hint_available"
    else:
        effective = CPU_HARDWARE_PROFILE
        reason = "portable_cpu_default"

    baseline_lanes = [name for name in engine_names(include_optional=False)]
    optional_lanes = []
    if effective in {LOCAL_GPU_HARDWARE_PROFILE, WORKSTATION_HARDWARE_PROFILE}:
        optional_lanes.extend(
            name for name in engine_names(include_optional=True) if engine_is_optional(name) and engines.get(name, {}).get("ready")
        )
    if effective in {REMOTE_MODEL_HARDWARE_PROFILE, WORKSTATION_HARDWARE_PROFILE}:
        optional_lanes.extend(
            name
            for name in engine_names(include_optional=True)
            if engine_is_optional(name) and (engines.get(name, {}).get("ready") or remote_ready)
        )

    return {
        "requested_profile": requested,
        "effective_profile": effective,
        "reason": reason,
        "scale_policy": "portable_cpu_first_then_enable_optional_lanes_when_proven_ready",
        "local_gpu": gpu,
        "remote_model_configured": remote_ready,
        "baseline_lanes": baseline_lanes,
        "optional_lanes": sorted(set(optional_lanes)),
        "does_not_require_specific_hardware": True,
    }


def doctor_payload() -> Dict[str, Any]:
    paths = ensure_paths()
    engines = {name: engine_status(name) for name in engine_names(include_optional=True)}
    return {
        "ok": True,
        "diagnostic_mode": "lightweight_no_model_import",
        "paths": {
            "cache_dir": str(paths.cache_dir),
            "data_dir": str(paths.data_dir),
            "state_dir": str(paths.state_dir),
        },
        "modules": {name: module_status(name) for name in MODULE_PACKAGES},
        "engines": engines,
        "hardware_scaling": hardware_scaling_payload(engines),
    }

