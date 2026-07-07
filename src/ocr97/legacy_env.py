from __future__ import annotations

import os
from typing import Dict, Iterable

# Canonical OCR97 env vars plus short compatibility aliases.
ENV_ALIASES: Dict[str, Iterable[str]] = {
    "OCR97_ROUTE_MODE": ("OCR97_OCR_ROUTE_MODE",),
    "OCR97_OCR_FINGERPRINT_PATH": ("OCR97_OCR_FINGERPRINT_PATH",),
    "OCR97_OCR_USE_GATEWAY": ("OCR97_GB10_OCR_USE_GATEWAY",),
    "OCR97_OCR_GATEWAY_URL": ("OCR97_GB10_OCR_GATEWAY_URL",),
    "OCR97_QWEN_MODEL": ("OCR97_GB10_QWEN_OCR_MODEL",),
    "OCR97_QWEN_FALLBACK_MODEL": ("OCR97_GB10_QWEN_OCR_FALLBACK_MODEL",),
    "OCR97_QWEN_OLLAMA_URL": ("OCR97_GB10_QWEN_OLLAMA_URL",),
    "OCR97_GOT_MODEL_ID": ("OCR97_GOT_OCR2_MODEL_ID",),
    "OCR97_GOT_DEVICE": ("OCR97_GOT_OCR2_DEVICE",),
    "OCR97_FINBERT_MODEL_ID": ("OCR97_FINBERT_MODEL_ID",),
    "OCR97_TABLEFORMER_MODEL_ID": ("OCR97_TABLEFORMER_MODEL_ID",),
    "OCR97_PADDLE_MODEL_ID": ("OCR97_PADDLEOCR_VL_MODEL_ID",),
    "OCR97_MINERU_MODEL_ID": ("OCR97_MINERU2_5_MODEL_ID",),
    "OCR97_OLMOCR_MODEL_ID": ("OCR97_OLMOCR2_MODEL_ID",),
    "OCR97_PADDLE_MODEL_DIR": ("OCR97_PADDLEOCR_VL_MODEL_DIR",),
    "OCR97_MINERU_MODEL_DIR": ("OCR97_MINERU2_5_MODEL_DIR",),
    "OCR97_OLMOCR_MODEL_DIR": ("OCR97_OLMOCR2_MODEL_DIR",),
    "OCR97_OCR_UPLOAD_DIR": ("OCR_UPLOAD_DIR",),
    "OCR97_INSTALL_METADATA_PATH": ("OCR97_OCR_INSTALL_METADATA_PATH",),
    "OCR97_SMOKE_FIXTURES": ("OCR97_OCR_SMOKE_FIXTURES",),
    "OCR97_SMOKE_REPORT_PATH": ("OCR97_OCR_SMOKE_REPORT_PATH",),
    "OCR97_GATEWAY_TIMEOUT_SEC": ("OCR97_OCR_GATEWAY_TIMEOUT_SEC",),
}


def apply_legacy_env_aliases() -> None:
    # OCR97_* -> legacy names
    for new_key, legacy_keys in ENV_ALIASES.items():
        new_val = str(os.getenv(new_key, "")).strip()
        if new_val:
            for legacy_key in legacy_keys:
                if not str(os.getenv(legacy_key, "")).strip():
                    os.environ[legacy_key] = new_val

    # legacy names -> OCR97_* for diagnostics/consistency
    for new_key, legacy_keys in ENV_ALIASES.items():
        if str(os.getenv(new_key, "")).strip():
            continue
        for legacy_key in legacy_keys:
            legacy_val = str(os.getenv(legacy_key, "")).strip()
            if legacy_val:
                os.environ[new_key] = legacy_val
                break

