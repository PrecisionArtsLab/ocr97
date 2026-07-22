from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .sroie_runner import score_sroie_payload


def _safe_name(value: Any, default: str = "case") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("._")
    return cleaned or default


def _resolve_local_path(raw: str, manifest_path: Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    repo_candidate = manifest_path.resolve().parents[1] / path
    if repo_candidate.exists():
        return repo_candidate
    return manifest_path.parent / path


def _save_image_value(value: Any, output_path: Path) -> None:
    from PIL import Image

    image = None
    if hasattr(value, "save"):
        image = value
    elif isinstance(value, dict) and value.get("bytes"):
        image = Image.open(BytesIO(value["bytes"]))
    elif isinstance(value, dict) and value.get("path"):
        image = Image.open(value["path"])
    elif isinstance(value, (bytes, bytearray)):
        image = Image.open(BytesIO(value))
    if image is None:
        raise ValueError("unsupported_parquet_image_value")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path, quality=95)


def _materialize_case(case: Mapping[str, Any], manifest_path: Path, source_dir: Path) -> tuple[Path, Dict[str, Any]]:
    case_id = _safe_name(case.get("id"))
    source_path = str(case.get("source_path") or "").strip()
    provenance: Dict[str, Any] = {"source_kind": "local"}
    expected_from_source: Dict[str, Any] = {}

    if source_path:
        source = _resolve_local_path(source_path, manifest_path)
        if not source.exists():
            raise FileNotFoundError(f"source_path_missing:{source}")
    elif case.get("hf_repo") and case.get("hf_path"):
        from huggingface_hub import hf_hub_download

        source = Path(
            hf_hub_download(
                repo_id=str(case["hf_repo"]),
                filename=str(case["hf_path"]),
                repo_type="dataset",
            )
        )
        provenance = {
            "source_kind": "huggingface_file",
            "hf_repo": str(case["hf_repo"]),
            "hf_path": str(case["hf_path"]),
        }
    elif case.get("hf_repo") and case.get("hf_parquet_path"):
        import pandas as pd
        from huggingface_hub import hf_hub_download

        parquet_path = Path(
            hf_hub_download(
                repo_id=str(case["hf_repo"]),
                filename=str(case["hf_parquet_path"]),
                repo_type="dataset",
            )
        )
        row_index = int(case.get("row_index") or 0)
        frame = pd.read_parquet(parquet_path)
        row = frame.iloc[row_index]
        output = source_dir / f"{case_id}.jpg"
        _save_image_value(row[str(case.get("image_column") or "image")], output)
        source = output
        text_column = str(case.get("text_column") or "text")
        if text_column in frame.columns:
            expected_from_source["transcription"] = str(row[text_column] or "")
        provenance = {
            "source_kind": "huggingface_parquet_row",
            "hf_repo": str(case["hf_repo"]),
            "hf_path": str(case["hf_parquet_path"]),
            "row_index": row_index,
        }
    else:
        raise ValueError(f"case_source_missing:{case_id}")

    if bool(case.get("as_image_pdf")):
        from PIL import Image

        pdf_path = source_dir / f"{case_id}.image_only.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            image.convert("RGB").save(pdf_path, "PDF", resolution=150.0)
        source = pdf_path
        provenance["image_only_pdf"] = True
    elif source.parent != source_dir:
        copied = source_dir / f"{case_id}{source.suffix.lower()}"
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, copied)
        source = copied

    return source, {"provenance": provenance, **expected_from_source}


def _extract_auto(client: Any, path: Path, case: Mapping[str, Any]) -> Dict[str, Any]:
    started = time.perf_counter()
    with path.open("rb") as handle:
        response = client.post(
            "/ocr/extract",
            data={
                "file": (io.BytesIO(handle.read()), path.name),
                "goal": str(case.get("goal") or "Extract all readable text and preserve exact values."),
                "model": "auto",
                "requested_lane_strict": "0",
                "route_mode": "quality_first",
                "max_pages": str(case.get("max_pages") or 3),
                "max_chars": str(case.get("max_chars") or 20000),
            },
            content_type="multipart/form-data",
        )
    return {
        "status_code": response.status_code,
        "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
        "payload": response.get_json(silent=True) or {},
    }


def _token_coverage(expected: str, actual: str) -> float:
    wanted = [token for token in re.findall(r"[A-Za-z0-9]+", expected.lower()) if token]
    if not wanted:
        return 1.0
    got = set(re.findall(r"[A-Za-z0-9]+", actual.lower()))
    return sum(1 for token in wanted if token in got) / float(len(wanted))


def _score(case: Mapping[str, Any], payload: Mapping[str, Any], materialized: Mapping[str, Any]) -> Dict[str, Any]:
    validation = str(case.get("validation") or "nonempty").strip().lower()
    text = str(payload.get("markdown") or payload.get("text") or "")
    if validation == "sroie_fields":
        return {"validation": validation, **score_sroie_payload(payload, dict(case.get("expected") or {}))}
    if validation == "transcription":
        expected = str(case.get("expected_text") or materialized.get("transcription") or "")
        coverage = _token_coverage(expected, text)
        return {
            "validation": validation,
            "score": int(round(coverage * 100.0)),
            "token_coverage": round(coverage, 4),
            "expected_text": expected,
        }
    min_chars = int(case.get("min_chars") or 20)
    chars = len(text.strip())
    return {
        "validation": "routing_smoke_nonempty",
        "score": 100 if bool(payload.get("ok")) and chars >= min_chars else 0,
        "chars": chars,
        "min_chars": min_chars,
        "accuracy_claim": False,
    }


def write_real_routing_report(output_dir: Path, summary: Mapping[str, Any]) -> Path:
    report_path = output_dir / "REAL_ROUTING_REPORT.md"
    lines = [
        "# OCR97 Real Auto-Routing Validation",
        "",
        "> This is a bounded real-corpus validation, not the OCR97 release grade.",
        "",
        f"- Cases: `{summary.get('case_count', 0)}`",
        f"- Accuracy-scored cases: `{summary.get('accuracy_case_count', 0)}`",
        f"- Accuracy score average: `{summary.get('accuracy_score_avg', 0)}`",
        f"- Combined score average: `{summary.get('score_avg', 0)}` (includes nonempty routing-smoke cases)",
        f"- Routes using fallback: `{summary.get('fallback_count', 0)}`",
        f"- Degraded Tesseract fallbacks: `{summary.get('degraded_fallback_count', 0)}`",
        f"- Failed requests: `{summary.get('failed_count', 0)}`",
        "",
        "| Case | Modality | Validation | Score | Selected engine | Attempt | Depth | Degraded | Latency ms |",
        "|---|---|---|---:|---|---:|---:|---|---:|",
    ]
    for row in list(summary.get("results") or []):
        score = dict(row.get("score") or {})
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("id") or ""),
                    str(row.get("modality") or ""),
                    str(score.get("validation") or row.get("validation") or ""),
                    str(score.get("score") or 0),
                    str(row.get("selected_engine") or row.get("engine") or ""),
                    str(row.get("selected_attempt_number") or 0),
                    str(row.get("chain_depth") or 0),
                    "yes" if row.get("degraded_fallback") else "no",
                    str(row.get("latency_ms") or 0),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Routing-smoke cases verify that real scanned or signed documents produce usable output, but do not claim transcription accuracy. "
            "Only `sroie_fields` and `transcription` cases contribute to the separate accuracy average.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_real_routing_benchmark(manifest_path: Path, output_dir: Path, *, limit: int = 0) -> Dict[str, Any]:
    os.environ.setdefault("OCR97_PROFILE", "local-production")
    os.environ.setdefault("OCR97_GB10_OCR_ENABLED", "1")
    os.environ.setdefault("OCR97_OCR_SMOKE_REQUIRED", "0")
    os.environ.setdefault("OCR97_OCR_GATEWAY_PREWARM_ENABLED", "0")
    os.environ.setdefault("OCR97_OCR_GATEWAY_PREWARM_ON_STARTUP", "0")
    os.environ.setdefault("OCR97_OCR_SLO_P95_IMAGE_PREPROCESSOR_MS", "180000")
    from .server import create_app

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = [dict(item) for item in list(manifest.get("cases") or []) if isinstance(item, dict)]
    if limit > 0:
        cases = cases[:limit]
    source_dir = output_dir / "sources"
    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(instance_name="ocr97_real_routing_benchmark")
    client = app.test_client()
    results: List[Dict[str, Any]] = []

    for case in cases:
        case_id = _safe_name(case.get("id"))
        try:
            source, materialized = _materialize_case(case, manifest_path, source_dir)
            extraction = _extract_auto(client, source, case)
            payload = dict(extraction.get("payload") or {})
            score = _score(case, payload, materialized)
            result = {
                "id": case_id,
                "label": str(case.get("label") or case_id),
                "corpus_kind": "real",
                "modality": str(case.get("modality") or "unknown"),
                "validation": str(score.get("validation") or ""),
                "input_path": str(source),
                "provenance": dict(materialized.get("provenance") or {}),
                "status_code": extraction["status_code"],
                "latency_ms": extraction["latency_ms"],
                "ok": bool(payload.get("ok")),
                "engine": str(payload.get("engine") or ""),
                "router": str(payload.get("router") or ""),
                "selected_engine": str(payload.get("selected_engine") or payload.get("engine") or ""),
                "engine_chain": list(payload.get("engine_chain") or []),
                "attempts": list(payload.get("attempts") or []),
                "attempted_engines": list(payload.get("attempted_engines") or []),
                "chain_depth": int(payload.get("chain_depth") or 0),
                "selected_attempt_index": int(payload.get("selected_attempt_index") if payload.get("selected_attempt_index") is not None else -1),
                "selected_attempt_number": int(payload.get("selected_attempt_number") or 0),
                "fallback_used": bool(payload.get("fallback_used")),
                "fallback_reason": str(payload.get("fallback_reason") or ""),
                "degraded_fallback": bool(payload.get("degraded_fallback")),
                "fallback_status": str(payload.get("fallback_status") or ""),
                "confidence": payload.get("confidence"),
                "confidence_tier": str(payload.get("confidence_tier") or ""),
                "score": score,
            }
            artifact = {**result, "text": str(payload.get("markdown") or payload.get("text") or "")}
        except Exception as exc:
            result = {
                "id": case_id,
                "label": str(case.get("label") or case_id),
                "corpus_kind": "real",
                "modality": str(case.get("modality") or "unknown"),
                "ok": False,
                "score": {"validation": str(case.get("validation") or ""), "score": 0},
                "error": f"{type(exc).__name__}:{exc}",
            }
            artifact = dict(result)
        artifact_path = artifact_dir / f"{case_id}.json"
        artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        results.append({**result, "artifact_path": str(artifact_path)})

    scores = [int((row.get("score") or {}).get("score") or 0) for row in results]
    accuracy_scores = [
        int((row.get("score") or {}).get("score") or 0)
        for row in results
        if str((row.get("score") or {}).get("validation") or "") in {"sroie_fields", "transcription"}
    ]
    summary = {
        "name": str(manifest.get("name") or "real_routing_benchmark"),
        "benchmark_kind": "end_to_end_auto_route",
        "corpus_kind": "real",
        "manifest": str(manifest_path),
        "case_count": len(results),
        "score_avg": round(sum(scores) / float(len(scores)), 2) if scores else 0.0,
        "accuracy_case_count": len(accuracy_scores),
        "accuracy_score_avg": round(sum(accuracy_scores) / float(len(accuracy_scores)), 2) if accuracy_scores else 0.0,
        "fallback_count": sum(1 for row in results if row.get("fallback_used")),
        "degraded_fallback_count": sum(1 for row in results if row.get("degraded_fallback")),
        "failed_count": sum(1 for row in results if not row.get("ok")),
        "results": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary["report"] = str(write_real_routing_report(output_dir, summary))
    (output_dir / "real_routing_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run OCR97 end-to-end auto routing against real document artifacts.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    result = run_real_routing_benchmark(
        Path(args.manifest).expanduser(),
        Path(args.output_dir).expanduser(),
        limit=int(args.limit or 0),
    )
    print(json.dumps({key: value for key, value in result.items() if key != "results"}, indent=2))
    return 1 if result["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
