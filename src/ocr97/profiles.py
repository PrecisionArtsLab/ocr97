from __future__ import annotations

import os


PUBLIC_PROFILE = "github-release"
LOCAL_PRODUCTION_PROFILE = "local-production"

AUTO_HARDWARE_PROFILE = "auto"
CPU_HARDWARE_PROFILE = "cpu"
LOCAL_GPU_HARDWARE_PROFILE = "local-gpu"
REMOTE_MODEL_HARDWARE_PROFILE = "remote-model"
WORKSTATION_HARDWARE_PROFILE = "workstation"

SUPPORTED_HARDWARE_PROFILES = {
    AUTO_HARDWARE_PROFILE,
    CPU_HARDWARE_PROFILE,
    LOCAL_GPU_HARDWARE_PROFILE,
    REMOTE_MODEL_HARDWARE_PROFILE,
    WORKSTATION_HARDWARE_PROFILE,
}


def active_profile() -> str:
    raw = str(os.getenv("OCR97_PROFILE", PUBLIC_PROFILE)).strip().lower()
    if raw in {"local", "local_prod", "local-production", "production-local"}:
        return LOCAL_PRODUCTION_PROFILE
    return PUBLIC_PROFILE


def local_production_enabled() -> bool:
    return active_profile() == LOCAL_PRODUCTION_PROFILE


def env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def gb10_default_enabled() -> bool:
    return env_flag("OCR97_GB10_OCR_ENABLED", default=local_production_enabled())


def gb10_gateway_default_enabled() -> bool:
    return env_flag("OCR97_GB10_OCR_USE_GATEWAY", default=local_production_enabled())


def requested_hardware_profile() -> str:
    raw = str(os.getenv("OCR97_HARDWARE_PROFILE", AUTO_HARDWARE_PROFILE)).strip().lower()
    aliases = {
        "default": AUTO_HARDWARE_PROFILE,
        "portable": CPU_HARDWARE_PROFILE,
        "cpu-only": CPU_HARDWARE_PROFILE,
        "gpu": LOCAL_GPU_HARDWARE_PROFILE,
        "cuda": LOCAL_GPU_HARDWARE_PROFILE,
        "remote": REMOTE_MODEL_HARDWARE_PROFILE,
        "ollama": REMOTE_MODEL_HARDWARE_PROFILE,
        "gb10": REMOTE_MODEL_HARDWARE_PROFILE,
        "gx10": REMOTE_MODEL_HARDWARE_PROFILE,
    }
    normalized = aliases.get(raw, raw)
    if normalized in SUPPORTED_HARDWARE_PROFILES:
        return normalized
    return AUTO_HARDWARE_PROFILE
