from __future__ import annotations

import argparse
import json
from pathlib import Path

from .diagnostics import doctor_payload
from .paths import ensure_paths


def _doctor() -> int:
    ensure_paths()
    payload = doctor_payload()
    print(json.dumps(payload, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="OCR97 CLI")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("doctor", help="Print runtime diagnostics")

    serve_parser = sub.add_parser("serve", help="Run OCR97 gateway")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=5521)
    serve_parser.add_argument("--instance-name", default="ocr97")
    serve_parser.add_argument("--upload-dir", default="")

    args = parser.parse_args()
    if args.cmd == "doctor":
        return _doctor()
    if args.cmd == "serve":
        import sys
        from .server import main as serve_main

        sys.argv = ["ocr97-serve", "--host", args.host, "--port", str(args.port), "--instance-name", args.instance_name]
        if args.upload_dir:
            sys.argv.extend(["--upload-dir", str(Path(args.upload_dir).expanduser())])
        return serve_main()

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

