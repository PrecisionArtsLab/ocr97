from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request


_LIGHTWEIGHT_STATUS_PATHS = {"/ocr/health", "/ocr/capabilities"}
_TRUE_VALUES = {"1", "true", "yes", "on"}


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUE_VALUES


def _lightweight_status_metadata() -> dict[str, Any]:
    return {
        "diagnostic_mode": "lightweight_no_model_import",
        "prewarm": {
            "enabled": _env_flag("OCR97_OCR_GATEWAY_PREWARM_ENABLED"),
            "on_startup": _env_flag("OCR97_OCR_GATEWAY_PREWARM_ON_STARTUP"),
        },
        "status": "operational",
        "capability": "ocr97_lightweight_no_model_import",
    }


def _merge_lightweight_status_metadata(payload: dict[str, Any]) -> bool:
    changed = False
    metadata = _lightweight_status_metadata()

    for key, value in metadata.items():
        if key not in payload:
            payload[key] = value
            changed = True
        elif key == "prewarm" and isinstance(payload[key], dict) and isinstance(value, Mapping):
            merged = {**value, **payload[key]}
            if merged != payload[key]:
                payload[key] = merged
                changed = True

    return changed


def _register_lightweight_status_routes(app: Flask, instance_name: str) -> None:
    @app.route("/ocr/health", methods=["GET"])
    def ocr_health():
        payload = {
            "instance": instance_name,
            "ok": True,
            "ollama": {"checked": False},
        }
        payload.update(_lightweight_status_metadata())
        return jsonify(payload)

    @app.route("/ocr/capabilities", methods=["GET"])
    def ocr_capabilities():
        payload = {
            "instance": instance_name,
            "capabilities": [],
            "ok": True,
        }
        payload.update(_lightweight_status_metadata())
        return jsonify(payload)


def create_app(instance_name: str = "ocr97", upload_dir: str = "") -> Flask:
    app = Flask(__name__)

    @app.after_request
    def add_lightweight_status_metadata(response):
        if request.path not in _LIGHTWEIGHT_STATUS_PATHS or not response.is_json:
            return response

        payload = response.get_json(silent=True)
        if not isinstance(payload, dict):
            return response

        if _merge_lightweight_status_metadata(payload):
            response.set_data(json.dumps(payload))
            response.content_type = "application/json"

        return response

    try:
        from .gateway import register_gb10_ocr_gateway_routes
    except ModuleNotFoundError as exc:
        if exc.name != f"{__package__}.gateway":
            raise
        _register_lightweight_status_routes(app, instance_name)
    else:
        register_gb10_ocr_gateway_routes(
            app,
            instance_name,
            upload_dir=Path(upload_dir).expanduser() if upload_dir else None,
        )

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OCR97 gateway server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5521)
    parser.add_argument("--instance-name", default="ocr97")
    parser.add_argument("--upload-dir", default="")
    args = parser.parse_args()

    app = create_app(instance_name=args.instance_name, upload_dir=args.upload_dir)
    app.run(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

