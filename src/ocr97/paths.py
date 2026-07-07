from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from platformdirs import user_cache_dir, user_data_dir, user_state_dir
except Exception:
    user_cache_dir = None
    user_data_dir = None
    user_state_dir = None


@dataclass(frozen=True)
class OCR97Paths:
    cache_dir: Path
    data_dir: Path
    state_dir: Path


def _fallback_base() -> Path:
    return Path.home() / ".ocr97"


def get_paths() -> OCR97Paths:
    cache = str(os.getenv("OCR97_CACHE_DIR", "")).strip()
    data = str(os.getenv("OCR97_DATA_DIR", "")).strip()
    state = str(os.getenv("OCR97_STATE_DIR", "")).strip()
    if cache and data and state:
        return OCR97Paths(cache_dir=Path(cache), data_dir=Path(data), state_dir=Path(state))

    if user_cache_dir and user_data_dir and user_state_dir:
        cache_dir = Path(cache) if cache else Path(user_cache_dir("ocr97", appauthor=False))
        data_dir = Path(data) if data else Path(user_data_dir("ocr97", appauthor=False))
        state_dir = Path(state) if state else Path(user_state_dir("ocr97", appauthor=False))
    else:
        base = _fallback_base()
        cache_dir = Path(cache) if cache else base / "cache"
        data_dir = Path(data) if data else base / "data"
        state_dir = Path(state) if state else base / "state"
    return OCR97Paths(cache_dir=cache_dir, data_dir=data_dir, state_dir=state_dir)


def ensure_paths() -> OCR97Paths:
    paths = get_paths()
    paths.cache_dir.mkdir(parents=True, exist_ok=True)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    return paths

