"""Print OCR97's lightweight diagnostics payload.

This example is intentionally safe for a fresh checkout. It does not start the
gateway, open documents, or import heavyweight model runtimes.
"""

from __future__ import annotations

import json

from ocr97.diagnostics import doctor_payload
from ocr97.profiles import active_profile, gb10_default_enabled


def main() -> None:
    payload = doctor_payload()
    summary = {
        "profile": active_profile(),
        "gb10_default_enabled": gb10_default_enabled(),
        "diagnostic_mode": payload.get("diagnostic_mode"),
        "hardware_scaling": payload.get("hardware_scaling", {}),
        "available_modules": {
            name: status.get("available", False)
            for name, status in payload.get("modules", {}).items()
        },
        "engine_reasons": {
            name: status.get("reason", "")
            for name, status in payload.get("engines", {}).items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
