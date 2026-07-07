from __future__ import annotations

from typing import Any, Callable, Dict

REGISTRY: Dict[str, Callable[..., Any]] = {}


def register(name: str, fn: Callable[..., Any]) -> None:
    """Register callable in local OCR97 registry."""
    REGISTRY[str(name)] = fn

