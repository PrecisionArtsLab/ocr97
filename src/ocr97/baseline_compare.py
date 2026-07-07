from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from .truth_benchmark import benchmark_claim_readiness, load_manifest, score_case
from .truth_runner import generate_image_fixtures, run_gateway_image_truth_benchmark


BaselineFn = Callable[[Path], Dict[str, Any]]


def _safe_id(value: Any) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value or "case")).strip("_") or "case"


def _extract_tesseract(path: Path) -> Dict[str, Any]:
    try:
        import pytesseract
        from PIL import Image
    except Exception as exc:
        return {"ok": False, "skip": True, "error": f"tesseract_unavailable:{type(exc).__name__}:{exc}"}
    started = time.perf_counter()
    try:
        text = pytesseract.image_to_string(Image.open(path))
    except Exception as exc:
        return {"ok": False, "skip": True, "error": f"tesseract_runtime_unavailable:{type(exc).__name__}:{exc}"}
    return {
        "ok": True,
        "engine": "tesseract",
        "text": str(text or ""),
        "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
    }


def _extract_easyocr(path: Path) -> Dict[str, Any]:
    try:
        import easyocr  # type: ignore
    except Exception as exc:
        return {"ok": False, "skip": True, "error": f"easyocr_unavailable:{type(exc).__name__}:{exc}"}
    started = time.perf_counter()
    try:
        cache = globals().setdefault("_EASYOCR_READER_CACHE", {})
        reader = cache.get("en_cpu")
        if reader is None:
            reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            cache["en_cpu"] = reader
        rows = reader.readtext(str(path), detail=0, paragraph=True)
    except Exception as exc:
        return {"ok": False, "skip": True, "error": f"easyocr_runtime_unavailable:{type(exc).__name__}:{exc}"}
    return {
        "ok": True,
        "engine": "easyocr",
        "text": "\n".join(str(row) for row in rows),
        "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
    }


def _flatten_paddle_rows(payload: Any) -> str:
    rows: List[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, str):
            if value.strip():
                rows.append(value.strip())
            return
        if isinstance(value, Mapping):
            for key in ("text", "rec_text", "label"):
                if key in value:
                    visit(value[key])
            for key in ("res", "data", "results"):
                if key in value:
                    visit(value[key])
            return
        if isinstance(value, (list, tuple)):
            if len(value) >= 2 and isinstance(value[1], (list, tuple)) and value[1] and isinstance(value[1][0], str):
                rows.append(value[1][0])
                return
            for item in value:
                visit(item)

    visit(payload)
    return "\n".join(rows)


def _extract_paddleocr(path: Path) -> Dict[str, Any]:
    os.environ.setdefault("FLAGS_use_onednn", "0")
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    try:
        from paddleocr import PaddleOCR  # type: ignore
    except Exception as exc:
        return {"ok": False, "skip": True, "error": f"paddleocr_unavailable:{type(exc).__name__}:{exc}"}
    started = time.perf_counter()
    try:
        try:
            engine = PaddleOCR(lang="en", use_textline_orientation=True)
        except Exception:
            engine = PaddleOCR(lang="en")
        if hasattr(engine, "predict"):
            payload = engine.predict(str(path))
        else:
            try:
                payload = engine.ocr(str(path), cls=True)
            except TypeError:
                payload = engine.ocr(str(path))
            except AttributeError:
                payload = engine.ocr(str(path))
    except Exception as exc:
        try:
            engine = PaddleOCR(lang="en")
            payload = engine.ocr(str(path))
        except Exception as fallback_exc:
            return {"ok": False, "skip": True, "error": f"paddleocr_runtime_unavailable:{type(fallback_exc).__name__}:{fallback_exc}"}
    return {
        "ok": True,
        "engine": "paddleocr",
        "text": _flatten_paddle_rows(payload),
        "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
    }


def _engine_registry() -> Dict[str, BaselineFn]:
    return {
        "tesseract": _extract_tesseract,
        "easyocr": _extract_easyocr,
        "paddleocr": _extract_paddleocr,
    }


def _score_direct_baseline(
    *,
    engine: str,
    manifest: Mapping[str, Any],
    fixture_paths: Mapping[str, str],
    output_dir: Path,
    extractor: BaselineFn,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [case for case in list(manifest.get("cases") or []) if isinstance(case, dict)]
    skip_reason = ""
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            continue
        case_id = str(case.get("id") or "").strip()
        if not case_id or case_id not in fixture_paths:
            continue
        if skip_reason:
            extraction = {"ok": False, "skip": True, "error": skip_reason}
            text = ""
        else:
            path = Path(fixture_paths[case_id])
            extraction = extractor(path)
            text = str(extraction.get("text") or extraction.get("markdown") or "")
            if index == 0 and extraction.get("skip"):
                skip_reason = str(extraction.get("error") or f"{engine}_unavailable")
        text = str(extraction.get("text") or extraction.get("markdown") or "")
        score = score_case(case, extracted_text=text) if extraction.get("ok") else {
            "id": case_id,
            "score": 0,
            "field_score": 0,
            "fields": [],
            "failure_buckets": {"engine_unavailable": 1} if extraction.get("skip") else {"engine_failed": 1},
        }
        artifact = {
            "id": case_id,
            "input_path": str(fixture_paths[case_id]),
            "engine": engine,
            "ok": bool(extraction.get("ok")),
            "skip": bool(extraction.get("skip")),
            "error": str(extraction.get("error") or ""),
            "latency_ms": float(extraction.get("latency_ms") or 0.0),
            "score": score,
            "extracted_text": text,
        }
        artifact_path = output_dir / f"{_safe_id(case_id)}.{engine}.json"
        artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        rows.append({k: v for k, v in artifact.items() if k != "extracted_text"} | {"artifact_path": str(artifact_path)})

    available_rows = [row for row in rows if not row.get("skip")]
    scored_rows = [row for row in available_rows if row.get("ok")]
    score_avg = 0 if not scored_rows else int(round(sum(int((row.get("score") or {}).get("score") or 0) for row in scored_rows) / float(len(scored_rows))))
    skipped = len([row for row in rows if row.get("skip")])
    return {
        "engine": engine,
        "available": skipped < len(rows) if rows else False,
        "unavailable_reason": skip_reason,
        "case_count": len(rows),
        "scored_case_count": len(scored_rows),
        "skipped_case_count": skipped,
        "score_avg": score_avg,
        "results": rows,
    }


def _score_ocr97(
    *,
    manifest: Mapping[str, Any],
    fixture_dir: Path,
    output_dir: Path,
    variant: str,
    ocr97_engine: str = "local_image_preprocessed_best",
) -> Dict[str, Any]:
    result = run_gateway_image_truth_benchmark(
        manifest,
        fixture_dir=fixture_dir,
        output_dir=output_dir,
        variant=variant,
        engine=ocr97_engine,
    )
    return {
        "engine": "ocr97",
        "available": True,
        "case_count": int(result.get("case_count") or 0),
        "scored_case_count": int(result.get("case_count") or 0),
        "skipped_case_count": 0,
        "score_avg": int(result.get("score_avg") or 0),
        "mode": result.get("mode"),
        "ocr97_engine": ocr97_engine,
        "results": list(result.get("results") or []),
    }


def _field_summary(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    totals: Dict[str, Dict[str, int]] = {}
    for row in rows:
        score = dict(row.get("score") or {})
        for field in list(score.get("fields") or []):
            name = str(field.get("name") or "unknown")
            totals.setdefault(name, {"hits": 0, "total": 0})
            totals[name]["total"] += 1
            if field.get("matched"):
                totals[name]["hits"] += 1
    return {
        name: {
            "hits": data["hits"],
            "total": data["total"],
            "accuracy": 0 if data["total"] == 0 else round((data["hits"] / float(data["total"])) * 100.0, 2),
        }
        for name, data in sorted(totals.items())
    }


def summarize_comparison(results: List[Mapping[str, Any]]) -> Dict[str, Any]:
    scored = [dict(row) for row in results if row.get("available") and int(row.get("scored_case_count") or 0) > 0]
    ranked = sorted(scored, key=lambda row: (-int(row.get("score_avg") or 0), str(row.get("engine") or "")))
    ocr97 = next((row for row in scored if row.get("engine") == "ocr97"), None)
    best_baseline = next((row for row in ranked if row.get("engine") != "ocr97"), None)
    delta = None
    verdict = "no_scored_baseline"
    if ocr97 and best_baseline:
        delta = int(ocr97.get("score_avg") or 0) - int(best_baseline.get("score_avg") or 0)
        if delta >= 10:
            verdict = "ocr97_leads"
        elif delta >= 0:
            verdict = "ocr97_comparable"
        else:
            verdict = "baseline_leads"
    elif ocr97:
        verdict = "ocr97_scored_no_baseline_available"
    return {
        "ranked": [{"engine": row.get("engine"), "score_avg": row.get("score_avg"), "scored_case_count": row.get("scored_case_count")} for row in ranked],
        "best_engine": ranked[0].get("engine") if ranked else "",
        "best_baseline": best_baseline.get("engine") if best_baseline else "",
        "ocr97_vs_best_baseline_delta": delta,
        "verdict": verdict,
    }


def markdown_report(payload: Mapping[str, Any]) -> str:
    lines = [
        f"# OCR97 Baseline Comparison - {payload.get('name')}",
        "",
        f"Variant: `{payload.get('variant')}`",
        f"Cases: `{payload.get('case_count')}`",
        "",
        "| Engine | Available | Score | Scored Cases | Skipped | Note |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for result in list(payload.get("engines") or []):
        lines.append(
            f"| `{result.get('engine')}` | {bool(result.get('available'))} | {int(result.get('score_avg') or 0)}/100 | "
            f"{int(result.get('scored_case_count') or 0)} | {int(result.get('skipped_case_count') or 0)} | "
            f"{str(result.get('unavailable_reason') or result.get('error') or '').replace('|', '/')[:120]} |"
        )
    summary = dict(payload.get("summary") or {})
    lines.extend([
        "",
        "## Conclusion",
        "",
        f"Verdict: `{summary.get('verdict')}`.",
        f"Best engine: `{summary.get('best_engine') or 'none'}`.",
        f"Best baseline: `{summary.get('best_baseline') or 'none'}`.",
        f"OCR97 delta versus best baseline: `{summary.get('ocr97_vs_best_baseline_delta')}`.",
        "",
        "## Field Evidence",
        "",
    ])
    for result in list(payload.get("engines") or []):
        lines.append(f"### {result.get('engine')}")
        field_summary = dict(result.get("field_accuracy") or {})
        if not field_summary:
            lines.append("No scored field evidence.")
            lines.append("")
            continue
        for field, data in field_summary.items():
            lines.append(f"- `{field}`: {data.get('accuracy')}% ({data.get('hits')}/{data.get('total')})")
        lines.append("")
    return "\n".join(lines)


def run_baseline_comparison(
    manifest: Mapping[str, Any],
    *,
    fixture_dir: Path,
    output_dir: Path,
    variant: str = "mild_degraded",
    engines: Optional[List[str]] = None,
    max_cases: int = 0,
    ocr97_engine: str = "local_image_preprocessed_best",
) -> Dict[str, Any]:
    cases = [dict(case) for case in list(manifest.get("cases") or []) if isinstance(case, dict)]
    if max_cases > 0:
        cases = cases[:max_cases]
    scoped_manifest = {**dict(manifest), "cases": cases}
    fixture_paths = generate_image_fixtures(scoped_manifest, fixture_dir, variant=variant)
    requested = engines or ["ocr97", "tesseract", "easyocr", "paddleocr"]
    registry = _engine_registry()
    results: List[Dict[str, Any]] = []
    for engine in requested:
        clean = str(engine or "").strip().lower()
        if clean == "ocr97":
            result = _score_ocr97(
                manifest=scoped_manifest,
                fixture_dir=fixture_dir,
                output_dir=output_dir / "ocr97",
                variant=variant,
                ocr97_engine=ocr97_engine,
            )
        elif clean in registry:
            result = _score_direct_baseline(
                engine=clean,
                manifest=scoped_manifest,
                fixture_paths=fixture_paths,
                output_dir=output_dir / clean,
                extractor=registry[clean],
            )
        else:
            result = {"engine": clean, "available": False, "case_count": len(cases), "scored_case_count": 0, "skipped_case_count": len(cases), "score_avg": 0, "results": [], "error": "unknown_engine"}
        result["field_accuracy"] = _field_summary(list(result.get("results") or []))
        result["readiness"] = benchmark_claim_readiness({
            "score_avg": int(result.get("score_avg") or 0),
            "case_count": int(result.get("scored_case_count") or 0),
            "failure_buckets": {},
        })
        results.append(result)
    payload = {
        "name": str(scoped_manifest.get("name") or "ocr97_baseline_comparison"),
        "variant": variant,
        "ocr97_engine": ocr97_engine,
        "case_count": len(cases),
        "fixture_dir": str(fixture_dir),
        "artifact_dir": str(output_dir),
        "engines": results,
        "summary": summarize_comparison(results),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "baseline_comparison.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (output_dir / "baseline_comparison.md").write_text(markdown_report(payload) + "\n", encoding="utf-8")
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run OCR97 against actual OCR outputs and compare against open-source baselines.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--fixture-dir", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--variant", default="mild_degraded")
    parser.add_argument("--engines", default="ocr97,tesseract,easyocr,paddleocr")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--ocr97-engine", default="local_image_preprocessed_best")
    args = parser.parse_args(argv)

    payload = run_baseline_comparison(
        load_manifest(Path(args.manifest).expanduser()),
        fixture_dir=Path(args.fixture_dir).expanduser(),
        output_dir=Path(args.artifact_dir).expanduser(),
        variant=args.variant,
        engines=[item.strip() for item in str(args.engines or "").split(",") if item.strip()],
        max_cases=int(args.max_cases or 0),
        ocr97_engine=str(args.ocr97_engine or "local_image_preprocessed_best"),
    )
    print(json.dumps({"name": payload["name"], "case_count": payload["case_count"], "summary": payload["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
