from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_examples(examples: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    rows = list(examples.get("examples") or [])
    if len(rows) < 8:
        errors.append("expected_at_least_8_examples")
    seen = set()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            errors.append(f"example_{index}_not_object")
            continue
        example_id = str(row.get("id") or "").strip()
        if not example_id:
            errors.append(f"example_{index}_missing_id")
        if example_id in seen:
            errors.append(f"duplicate_example:{example_id}")
        seen.add(example_id)
        for key in ("document_type", "stressors", "expected_signal", "why_it_matters"):
            if not str(row.get(key) or "").strip():
                errors.append(f"{example_id or index}_missing_{key}")
    return errors


def validate_queue(queue: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    runs = list(queue.get("runs") or [])
    if len(runs) < 3:
        errors.append("expected_at_least_3_queue_runs")
    seen = set()
    for index, row in enumerate(runs, start=1):
        if not isinstance(row, dict):
            errors.append(f"run_{index}_not_object")
            continue
        run_id = str(row.get("id") or "").strip()
        if not run_id:
            errors.append(f"run_{index}_missing_id")
        if run_id in seen:
            errors.append(f"duplicate_run:{run_id}")
        seen.add(run_id)
        if not str(row.get("run_dir") or "").strip():
            errors.append(f"{run_id or index}_missing_run_dir")
        if not str(row.get("broad_variants") or "").strip():
            errors.append(f"{run_id or index}_missing_broad_variants")
        if bool(row.get("include_heavy", True)) and not str(row.get("focus_variants") or "").strip():
            errors.append(f"{run_id or index}_missing_focus_variants")
    return errors


def build_plan(examples: Mapping[str, Any], queue: Mapping[str, Any]) -> str:
    lines = [
        "# Spark OCR97 3090 Long-Form Capability Test Run",
        "",
        "## Purpose",
        "",
        "Measure what OCR97 can do on this RTX 3090 machine when it is allowed to use the full local stack, not just the portable default path.",
        "This is not a public 97-grade claim. It is a ceiling test that should identify the work required to move from the current public grade toward a defensible 3090-backed grade.",
        "",
        "## Scope",
        "",
        "- Rebuild the 120-case release manifest.",
        "- Run the corrected long-form queue in `benchmarks/ocr97_3090_longform_queue.json`.",
        "- Capture GPU/driver evidence before the run.",
        "- Audit all generated result files with `ocr97.capability_audit`.",
        "- Report which failures are OCR text misses, table structure misses, field consensus misses, GPU-lane readiness failures, or latency problems.",
        "",
        "## Required Commands",
        "",
        "Plan and validate only:",
        "",
        "```powershell",
        "powershell -ExecutionPolicy Bypass -File tools/run_spark_ocr97_3090_longform.ps1 -PlanOnly",
        "```",
        "",
        "Full run:",
        "",
        "```powershell",
        "powershell -ExecutionPolicy Bypass -File tools/run_spark_ocr97_3090_longform.ps1 -ResetQueue",
        "```",
        "",
        "Status check while running:",
        "",
        "```powershell",
        "powershell -ExecutionPolicy Bypass -File tools/run_spark_ocr97_3090_longform.ps1 -Status",
        "```",
        "",
        "Audit-only rerun after artifacts exist:",
        "",
        "```powershell",
        "python -m ocr97.capability_audit --run-dir artifacts/ocr97_3090_longform --output artifacts/ocr97_3090_longform/OCR97_3090_CAPABILITY_AUDIT.md --json-output artifacts/ocr97_3090_longform/OCR97_3090_CAPABILITY_AUDIT.json",
        "```",
        "",
        "## Test Sequence",
        "",
        "1. Hardware preflight: record `nvidia-smi`, Python version, installed OCR97 package path, and CUDA visibility.",
        "2. Manifest rebuild: regenerate `benchmarks/release_97_gate_manifest.json` from `benchmarks/mixed_corpus_manifest.json`.",
        "3. Queue run A: broad release baseline with clean, mild degraded, low contrast, and hard focus variants.",
        "4. Queue run B: table and field consensus stress on invoices, bank statements, brokerage, purchase orders, EOBs, logistics, and legal forms.",
        "5. Queue run C: hard image stress using rotated, noisy scan, blurred, small text, and low contrast variants.",
        "6. Queue run D: 3090 escalation proxy run. Spark should inspect low-confidence cases and document whether a GPU/VLM lane was available or blocked.",
        "7. Capability audit: generate the markdown and JSON audit reports from all result files.",
        "8. Final Spark report: write pass/fail, current 3090 grade estimate, blockers to 93+, and blockers to 97.",
        "",
        "## Example Targets Spark Should Use",
        "",
        "| ID | Type | Stressors | Expected Signal |",
        "|---|---|---|---|",
    ]
    for row in list(examples.get("examples") or []):
        stressors = ", ".join(str(item) for item in list(row.get("stressors") or []))
        lines.append(f"| `{row.get('id')}` | {row.get('document_type')} | {stressors} | {row.get('expected_signal')} |")
    lines.extend(
        [
            "",
            "## Final Report Requirements",
            "",
            "Spark must write `artifacts/ocr97_3090_longform/OCR97_3090_LONGFORM_SPARK_REPORT.md` with:",
            "",
            "- hardware evidence",
            "- queue statuses and summary paths",
            "- best and worst engine/variant averages",
            "- weakest 25 cases from the capability audit",
            "- whether field consensus was used enough to matter",
            "- whether table row recovery improved or remained a blocker",
            "- whether any 3090/GPU lane actually ran",
            "- grade estimate for current implementation on this 3090",
            "- top five changes required to reach 93+ and then 97",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and write the OCR97 3090 Spark long-form handoff.")
    parser.add_argument("--examples", default="benchmarks/ocr97_3090_longform_examples.json")
    parser.add_argument("--queue", default="benchmarks/ocr97_3090_longform_queue.json")
    parser.add_argument("--output", default="SPARK_OCR97_3090_LONGFORM_TEST_RUN.md")
    parser.add_argument("--json-output", default="")
    args = parser.parse_args(argv)

    examples = _load_json(Path(args.examples).expanduser())
    queue = _load_json(Path(args.queue).expanduser())
    errors = validate_examples(examples) + validate_queue(queue)
    result = {
        "ok": not errors,
        "errors": errors,
        "examples": len(list(examples.get("examples") or [])),
        "runs": len(list(queue.get("runs") or [])),
        "output": args.output,
    }
    if errors:
        print(json.dumps(result, indent=2))
        return 1
    output = Path(args.output).expanduser()
    output.write_text(build_plan(examples, queue), encoding="utf-8")
    if args.json_output:
        json_output = Path(args.json_output).expanduser()
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
