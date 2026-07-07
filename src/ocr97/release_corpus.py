from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .truth_benchmark import load_manifest, score_manifest


DEFAULT_VARIANTS = [
    ("baseline", "Baseline document"),
    ("dense", "Dense layout"),
    ("rotated", "Rotated scan target"),
    ("noisy", "Noisy scan target"),
    ("small_text", "Small text target"),
    ("table_first", "Table-first target"),
]


def _case_variant(case: Mapping[str, Any], *, variant_id: str, variant_label: str, ordinal: int) -> Dict[str, Any]:
    base = dict(case)
    original_id = str(base.get("id") or f"case_{ordinal}")
    sample_text = str(base.get("sample_text") or "")
    control_line = f"Release Gate Ref: RG-{ordinal:03d}-{variant_id.upper()}"
    required_tokens = [str(item) for item in list(base.get("required_tokens") or []) if str(item).strip()]
    required_tokens.append(control_line)

    base["id"] = f"{original_id}_{variant_id}"
    base["label"] = f"{base.get('label') or original_id} - {variant_label}"
    base["source_case_id"] = original_id
    base["release_variant"] = variant_id
    base["required_tokens"] = required_tokens
    base["sample_text"] = f"{sample_text}\n{control_line}".strip()
    return base


def expand_manifest(base_manifest: Mapping[str, Any], *, variants: Optional[List[tuple[str, str]]] = None) -> Dict[str, Any]:
    selected_variants = list(variants or DEFAULT_VARIANTS)
    expanded_cases: List[Dict[str, Any]] = []
    ordinal = 1
    for case in list(base_manifest.get("cases") or []):
        if not isinstance(case, dict):
            continue
        for variant_id, variant_label in selected_variants:
            expanded_cases.append(_case_variant(case, variant_id=variant_id, variant_label=variant_label, ordinal=ordinal))
            ordinal += 1

    return {
        "name": "ocr97_release_97_gate_corpus",
        "description": (
            "Deterministic 120-case release gate expanded from the mixed corpus. "
            "Variants are synthetic but exercise table, rotation, noisy scan, and small-text lanes."
        ),
        "source_manifest": str(base_manifest.get("name") or ""),
        "variant_count": len(selected_variants),
        "cases": expanded_cases,
    }


def write_release_manifest(base_path: Path, output_path: Path) -> Dict[str, Any]:
    manifest = expand_manifest(load_manifest(base_path))
    score = score_manifest(manifest)
    if int(score.get("score_avg") or 0) != 100:
        raise ValueError(f"expanded_manifest_self_score_failed: {score.get('score_avg')}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"output": str(output_path), "case_count": len(manifest["cases"]), "self_score": int(score["score_avg"])}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the OCR97 120-case release-grade benchmark manifest.")
    parser.add_argument("--base", default="benchmarks/mixed_corpus_manifest.json", help="Base mixed corpus manifest.")
    parser.add_argument("--output", default="benchmarks/release_97_gate_manifest.json", help="Expanded release gate manifest.")
    args = parser.parse_args(argv)
    result = write_release_manifest(Path(args.base).expanduser(), Path(args.output).expanduser())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
