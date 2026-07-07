from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
from urllib.parse import urlparse

import requests

from .gateway import _native_pdf_text_extract
from .truth_benchmark import benchmark_claim_readiness, load_manifest, score_case


DEFAULT_TIMEOUT = 45
DEFAULT_MAX_PAGES = 12
DEFAULT_MAX_CHARS = 80000


def _utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_name(value: Any, default: str = "document") -> str:
    clean = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value or "")).strip("._")
    return clean or default


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_must_be_object:{path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _case_filename(case: Mapping[str, Any]) -> str:
    explicit = str(case.get("filename") or "").strip()
    if explicit:
        return _safe_name(explicit)
    parsed = urlparse(str(case.get("url") or ""))
    name = Path(parsed.path).name
    if name:
        return _safe_name(name)
    return f"{_safe_name(case.get('id'))}.pdf"


def fetch_document(case: Mapping[str, Any], docs_dir: Path, *, timeout: int = DEFAULT_TIMEOUT, force: bool = False) -> Dict[str, Any]:
    docs_dir.mkdir(parents=True, exist_ok=True)
    output_path = docs_dir / _case_filename(case)
    source_path = str(case.get("source_path") or "").strip()
    if source_path:
        src = Path(source_path).expanduser()
        if not src.exists():
            return {"ok": False, "case_id": str(case.get("id") or ""), "path": str(output_path), "error": f"source_path_missing:{src}"}
        if force or not output_path.exists():
            shutil.copy2(src, output_path)
        return {"ok": True, "case_id": str(case.get("id") or ""), "path": str(output_path), "source": str(src), "mode": "source_path"}

    url = str(case.get("url") or "").strip()
    if not url:
        return {"ok": False, "case_id": str(case.get("id") or ""), "path": str(output_path), "error": "missing_url"}
    if output_path.exists() and not force:
        return {"ok": True, "case_id": str(case.get("id") or ""), "path": str(output_path), "url": url, "mode": "cached"}

    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": "OCR97 real-corpus builder/0.1"})
        response.raise_for_status()
    except Exception as exc:
        return {"ok": False, "case_id": str(case.get("id") or ""), "path": str(output_path), "url": url, "error": f"download_failed:{type(exc).__name__}:{exc}"}
    output_path.write_bytes(response.content)
    return {
        "ok": True,
        "case_id": str(case.get("id") or ""),
        "path": str(output_path),
        "url": url,
        "mode": "downloaded",
        "bytes": len(response.content),
        "content_type": response.headers.get("content-type", ""),
    }


def fetch_manifest_documents(
    manifest: Mapping[str, Any],
    output_dir: Path,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    force: bool = False,
) -> Dict[str, Any]:
    docs_dir = output_dir / "docs"
    rows = [fetch_document(case, docs_dir, timeout=timeout, force=force) for case in list(manifest.get("cases") or []) if isinstance(case, dict)]
    summary = {
        "stage": "real_document_fetch",
        "created_at": _utc_iso(),
        "manifest_name": str(manifest.get("name") or ""),
        "case_count": len(rows),
        "downloaded": sum(1 for row in rows if row.get("ok")),
        "failed": sum(1 for row in rows if not row.get("ok")),
        "documents": rows,
    }
    _write_json(output_dir / "fetch_manifest.json", summary)
    return summary


def _artifact_payload(case: Mapping[str, Any], source: Mapping[str, Any], extraction: Mapping[str, Any], score: Mapping[str, Any]) -> Dict[str, Any]:
    text = str(extraction.get("markdown") or extraction.get("text") or "")
    return {
        "ok": bool(extraction.get("ok")),
        "case_id": str(case.get("id") or ""),
        "label": str(case.get("label") or ""),
        "source": dict(source),
        "engine": str(extraction.get("engine") or "native_pdf_text"),
        "route": str(extraction.get("route") or ""),
        "pages": extraction.get("pages"),
        "extracted_text": text,
        "markdown": text,
        "score": dict(score),
        "error": str(extraction.get("error") or ""),
    }


def score_real_documents(
    manifest: Mapping[str, Any],
    output_dir: Path,
    *,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    docs_by_id = {str(row.get("case_id") or ""): row for row in list((output_dir / "fetch_manifest.json").exists() and _read_json(output_dir / "fetch_manifest.json").get("documents") or [])}
    results: List[Dict[str, Any]] = []
    extracted_text_by_id: Dict[str, str] = {}

    for case in [dict(item) for item in list(manifest.get("cases") or []) if isinstance(item, dict)]:
        case_id = str(case.get("id") or "")
        source = dict(docs_by_id.get(case_id) or {})
        path = Path(str(source.get("path") or output_dir / "docs" / _case_filename(case)))
        if not path.exists():
            score = score_case(case, extracted_text="")
            artifact_path = output_dir / "artifacts" / f"{_safe_name(case_id)}.ocr97.json"
            _write_json(artifact_path, _artifact_payload(case, source, {"ok": False, "engine": "native_pdf_text", "error": "document_missing"}, score))
            results.append(
                {
                    "id": case_id,
                    "input_path": str(path),
                    "artifact_path": str(artifact_path),
                    "ok": False,
                    "engine": "native_pdf_text",
                    "score": score,
                    "error": "document_missing",
                }
            )
            continue
        extraction = _native_pdf_text_extract(path, max_pages=max_pages, max_chars=max_chars)
        text = str(extraction.get("markdown") or extraction.get("text") or "")
        extracted_text_by_id[case_id] = text
        score = score_case(case, extracted_text=text)
        artifact_path = output_dir / "artifacts" / f"{_safe_name(case_id)}.ocr97.json"
        _write_json(artifact_path, _artifact_payload(case, source, extraction, score))
        results.append(
            {
                "id": case_id,
                "input_path": str(path),
                "artifact_path": str(artifact_path),
                "ok": bool(extraction.get("ok")),
                "engine": str(extraction.get("engine") or "native_pdf_text"),
                "score": score,
                "error": str(extraction.get("error") or ""),
            }
        )

    score_avg = 0 if not results else int(round(sum(int(row["score"].get("score") or 0) for row in results) / float(len(results))))
    failure_buckets: Dict[str, int] = {}
    field_totals: Dict[str, Dict[str, int]] = {}
    for result in results:
        score = dict(result.get("score") or {})
        for bucket, count in dict(score.get("failure_buckets") or {}).items():
            failure_buckets[str(bucket)] = failure_buckets.get(str(bucket), 0) + int(count)
        for field in list(score.get("fields") or []):
            name = str(field.get("name") or "unknown")
            field_totals.setdefault(name, {"hits": 0, "total": 0})
            field_totals[name]["total"] += 1
            if field.get("matched"):
                field_totals[name]["hits"] += 1
    field_accuracy = {
        name: {
            "hits": totals["hits"],
            "total": totals["total"],
            "accuracy": 100 if totals["total"] == 0 else round((totals["hits"] / float(totals["total"])) * 100.0, 2),
        }
        for name, totals in sorted(field_totals.items())
    }
    comparison = {
        "name": str(manifest.get("name") or "ocr97_real_document_corpus"),
        "stage": "real_document_score",
        "created_at": _utc_iso(),
        "case_count": len(results),
        "variant": "real_pdf_native",
        "summary": {
            "score_avg": score_avg,
            "field_accuracy": field_accuracy,
            "failure_buckets": failure_buckets,
            "readiness": benchmark_claim_readiness({"score_avg": score_avg, "case_count": len(results), "failure_buckets": failure_buckets}),
        },
        "engines": [
            {
                "engine": "ocr97",
                "extractor": "native_pdf_text",
                "available": True,
                "score_avg": score_avg,
                "scored_case_count": len(results),
                "results": results,
            }
        ],
    }
    _write_json(output_dir / "baseline_comparison.json", comparison)
    _write_json(output_dir / "extracted_text_by_id.json", extracted_text_by_id)
    return comparison


def build_failure_summary(comparison: Mapping[str, Any], output_dir: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for engine in list(comparison.get("engines") or []):
        for result in list(engine.get("results") or []):
            score = dict(result.get("score") or {})
            for field in list(score.get("fields") or []):
                if field.get("matched"):
                    continue
                rows.append(
                    {
                        "case_id": str(result.get("id") or ""),
                        "field": str(field.get("name") or ""),
                        "expected": field.get("expected"),
                        "failure_bucket": str(field.get("failure_bucket") or ""),
                        "partial_score": float(field.get("partial_score") or 0.0),
                        "artifact_path": str(result.get("artifact_path") or ""),
                        "input_path": str(result.get("input_path") or ""),
                        "top_candidate": (list(field.get("ranked_candidates") or [])[:1] or [None])[0],
                    }
                )
    summary = {
        "stage": "real_document_failure_summary",
        "created_at": _utc_iso(),
        "failure_count": len(rows),
        "failures": rows,
    }
    _write_json(output_dir / "real_document_failures.json", summary)
    lines = [
        "# OCR97 Real Document Failure Summary",
        "",
        f"- generated_at: `{summary['created_at']}`",
        f"- failure_count: `{len(rows)}`",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['case_id']} / {row['field']}",
                "",
                f"- bucket: `{row['failure_bucket']}`",
                f"- expected: `{row['expected']}`",
                f"- partial_score: `{row['partial_score']}`",
                f"- input: `{row['input_path']}`",
                f"- artifact: `{row['artifact_path']}`",
                "",
            ]
        )
    (output_dir / "real_document_failures.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return summary


def fetch_score(
    manifest_path: Path,
    output_dir: Path,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    force: bool = False,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> Dict[str, Any]:
    manifest = load_manifest(manifest_path)
    fetch = fetch_manifest_documents(manifest, output_dir, timeout=timeout, force=force)
    comparison = score_real_documents(manifest, output_dir, max_pages=max_pages, max_chars=max_chars)
    failures = build_failure_summary(comparison, output_dir)
    summary = {
        "stage": "real_document_fetch_score",
        "created_at": _utc_iso(),
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "fetch": fetch,
        "score_avg": comparison["summary"]["score_avg"],
        "failure_count": failures["failure_count"],
        "comparison_path": str(output_dir / "baseline_comparison.json"),
        "failure_summary_path": str(output_dir / "real_document_failures.md"),
    }
    _write_json(output_dir / "run_summary.json", summary)
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch and score public real-document OCR97 corpus cases.")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch_score_cmd = sub.add_parser("fetch-score", help="Download corpus PDFs, score them, and write Helix-compatible comparison JSON.")
    fetch_score_cmd.add_argument("--manifest", required=True, help="Path to real-document corpus manifest.")
    fetch_score_cmd.add_argument("--output-dir", required=True, help="Directory for downloaded PDFs and artifacts.")
    fetch_score_cmd.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    fetch_score_cmd.add_argument("--force", action="store_true", help="Re-download documents even if cached.")
    fetch_score_cmd.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES)
    fetch_score_cmd.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)

    args = parser.parse_args(argv)
    if args.command == "fetch-score":
        summary = fetch_score(
            Path(args.manifest).expanduser(),
            Path(args.output_dir).expanduser(),
            timeout=args.timeout,
            force=bool(args.force),
            max_pages=int(args.max_pages),
            max_chars=int(args.max_chars),
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
