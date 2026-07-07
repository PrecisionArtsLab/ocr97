from __future__ import annotations

import argparse
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import requests

from .receipt_fields import normalize_receipt_date, normalize_receipt_key, receipt_date_variants, receipt_field_value


SROIE_DATASET = "jsdnrs/ICDAR2019-SROIE"
SROIE_SOURCE_PAGE = "https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE"
SROIE_ROWS_URL = "https://datasets-server.huggingface.co/rows"


def _normalize_number(value: Any) -> str:
    raw = str(value or "").replace(",", "").strip()
    match = re.search(r"\d+(?:\.\d+)?", raw)
    if not match:
        return ""
    try:
        return f"{float(match.group(0)):.2f}"
    except Exception:
        return match.group(0)


def _receipt_text(payload: Mapping[str, Any]) -> str:
    return str(payload.get("markdown") or payload.get("text") or "")


def _contains_company(text: str, receipt_fields: Iterable[Mapping[str, Any]], expected: Any) -> bool:
    want = normalize_receipt_key(expected)
    if not want:
        return True
    haystacks = [normalize_receipt_key(text)]
    haystacks.extend(normalize_receipt_key(item.get("value")) for item in receipt_fields if str(item.get("field") or "") == "company")
    for haystack in haystacks:
        if want and (want in haystack or haystack in want):
            return True
    want_tokens = set(want.split())
    for haystack in haystacks:
        got_tokens = set(haystack.split())
        if want_tokens and len(want_tokens & got_tokens) / float(len(want_tokens)) >= 0.8:
            return True
    return False


def _contains_total(text: str, receipt_fields: Iterable[Mapping[str, Any]], expected: Any) -> bool:
    want = _normalize_number(expected)
    if not want:
        return True
    values = re.findall(r"\d+(?:[,.]\d{2})", str(text or ""))
    values.extend(str(item.get("value") or "") for item in receipt_fields if str(item.get("field") or "") == "total")
    return any(_normalize_number(value) == want for value in values)


def _contains_date(text: str, receipt_fields: Iterable[Mapping[str, Any]], expected: Any) -> bool:
    want = normalize_receipt_date(expected)
    if not want:
        return True
    possible = set(receipt_date_variants(want))
    for item in receipt_fields:
        if str(item.get("field") or "") == "date" and normalize_receipt_date(item.get("value")) == want:
            return True
    normalized_text = str(text or "")
    for variant in sorted(possible, key=len, reverse=True):
        if variant and variant in normalized_text:
            return True
    for match in re.finditer(r"[0-9OoQDISsl|B]{1,2}\s*[/-]\s*[0-9OoQDISsl|Be]{1,2}\s*[/-]?\s*(?:20)?[0-9OoQDISsl|B]{2,4}", normalized_text):
        if normalize_receipt_date(match.group(0)) == want:
            return True
    return False


def _address_coverage(text: str, expected: Any) -> float:
    want_tokens = [token for token in normalize_receipt_key(expected).split() if len(token) > 1]
    if not want_tokens:
        return 1.0
    haystack = set(normalize_receipt_key(text).split())
    hits = sum(1 for token in want_tokens if token in haystack)
    return hits / float(len(want_tokens))


def score_sroie_payload(payload: Mapping[str, Any], expected: Mapping[str, Any]) -> Dict[str, Any]:
    text = _receipt_text(payload)
    receipt_fields = list(payload.get("receipt_fields") or [])
    fields = [
        {"name": "company", "expected": expected.get("company"), "matched": _contains_company(text, receipt_fields, expected.get("company"))},
        {"name": "total", "expected": expected.get("total"), "matched": _contains_total(text, receipt_fields, expected.get("total"))},
        {"name": "date", "expected": expected.get("date"), "matched": _contains_date(text, receipt_fields, expected.get("date"))},
    ]
    address_text = text + "\n" + "\n".join(str(item.get("value") or "") for item in receipt_fields if str(item.get("field") or "") == "address")
    coverage = _address_coverage(address_text, expected.get("address"))
    fields.append({"name": "address", "expected": expected.get("address"), "matched": coverage >= 0.65, "token_coverage": round(coverage, 3)})
    hits = sum(1 for row in fields if row["matched"])
    return {"score": int(round((hits / 4.0) * 100.0)), "field_hits": hits, "field_total": 4, "fields": fields}


def _fetch_rows(*, split: str, offset: int, length: int, dataset: str = SROIE_DATASET) -> list[Dict[str, Any]]:
    response = requests.get(
        SROIE_ROWS_URL,
        params={"dataset": dataset, "config": "default", "split": split, "offset": int(offset), "length": int(length)},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    return [dict(item) for item in list(payload.get("rows") or [])]


def _download_image(url: str, path: Path, *, retries: int = 3, backoff_sec: float = 1.5) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    last_error: Exception | None = None
    for attempt in range(max(1, int(retries))):
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
            return
        except Exception as exc:
            last_error = exc
            if attempt + 1 < max(1, int(retries)):
                time.sleep(backoff_sec * float(attempt + 1))
    if last_error:
        raise last_error
    path.parent.mkdir(parents=True, exist_ok=True)


def _extract_with_gateway(client: Any, image_path: Path, *, engine: str) -> Dict[str, Any]:
    started = time.perf_counter()
    with image_path.open("rb") as handle:
        response = client.post(
            "/ocr/extract",
            data={
                "file": (io.BytesIO(handle.read()), image_path.name),
                "goal": "Extract exact receipt fields: company, date, total, address. Preserve OCR text and numbers.",
                "model": engine,
                "requested_lane_strict": "1",
                "route_mode": "balanced",
                "max_pages": "1",
                "max_chars": "12000",
            },
            content_type="multipart/form-data",
        )
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
    return {"status_code": response.status_code, "latency_ms": elapsed_ms, "payload": response.get_json(silent=True) or {}}


def run_sroie_benchmark(
    *,
    output_dir: Path,
    split: str = "test",
    offset: int = 0,
    length: int = 10,
    engine: str = "local_image_preprocessed_best",
    dataset: str = SROIE_DATASET,
) -> Dict[str, Any]:
    os.environ.setdefault("OCR97_OCR_SMOKE_REQUIRED", "0")
    os.environ.setdefault("OCR97_OCR_GATEWAY_PREWARM_ENABLED", "0")
    os.environ.setdefault("OCR97_OCR_GATEWAY_PREWARM_ON_STARTUP", "0")
    os.environ.setdefault("OCR97_OCR_SLO_P95_IMAGE_PREPROCESSOR_MS", "120000")
    os.environ.setdefault("OCR97_OCR_PREPROCESS_INCLUDE_TEXT", "1")
    from .server import create_app

    rows = _fetch_rows(split=split, offset=offset, length=length, dataset=dataset)
    image_dir = output_dir / "images"
    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(instance_name="ocr97_sroie_runner")
    client = app.test_client()
    results: List[Dict[str, Any]] = []

    for ordinal, item in enumerate(rows):
        row = dict(item.get("row") or {})
        row_idx = int(item.get("row_idx") if item.get("row_idx") is not None else offset + ordinal)
        key = str(row.get("key") or f"row_{row_idx}")
        image = dict(row.get("image") or {})
        source = str(image.get("src") or "")
        expected = dict(row.get("entities") or {})
        safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", key).strip("_") or f"row_{row_idx}"
        image_path = image_dir / f"{row_idx:04d}_{safe_key}.jpg"
        _download_image(source, image_path)
        extraction = _extract_with_gateway(client, image_path, engine=engine)
        payload = dict(extraction.get("payload") or {})
        score = score_sroie_payload(payload, expected)
        artifact = {
            "dataset": dataset,
            "source_page": SROIE_SOURCE_PAGE,
            "split": split,
            "offset": row_idx,
            "key": key,
            "source": source.split("?")[0],
            "input_path": str(image_path),
            "status_code": extraction["status_code"],
            "latency_ms": extraction["latency_ms"],
            "ok": bool(payload.get("ok")),
            "engine": str(payload.get("engine") or ""),
            "router": str(payload.get("router") or ""),
            "selected_engine": str(payload.get("selected_engine") or ""),
            "selected_preprocess": str(payload.get("selected_preprocess") or ""),
            "field_consensus_used": bool(payload.get("field_consensus_used")),
            "receipt_fields_used": bool(payload.get("receipt_fields_used")),
            "receipt_fields": list(payload.get("receipt_fields") or []),
            "expected": expected,
            "score": score,
            "text": _receipt_text(payload),
        }
        artifact_path = artifact_dir / f"{row_idx:04d}_{safe_key}.json"
        artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        results.append({k: v for k, v in artifact.items() if k != "text"} | {"artifact_path": str(artifact_path)})

    avg = 0 if not results else round(sum(float((row.get("score") or {}).get("score") or 0) for row in results) / float(len(results)), 2)
    field_totals: Dict[str, Dict[str, int]] = {}
    for row in results:
        for field in list((row.get("score") or {}).get("fields") or []):
            name = str(field.get("name") or "")
            bucket = field_totals.setdefault(name, {"hits": 0, "total": 0})
            bucket["hits"] += 1 if field.get("matched") else 0
            bucket["total"] += 1
    return {
        "dataset": dataset,
        "source_page": SROIE_SOURCE_PAGE,
        "split": split,
        "offset": offset,
        "case_count": len(results),
        "engine": engine,
        "score_avg": avg,
        "field_totals": {
            name: {**bucket, "accuracy": round((bucket["hits"] / float(max(1, bucket["total"]))) * 100.0, 2)}
            for name, bucket in sorted(field_totals.items())
        },
        "output_dir": str(output_dir),
        "artifact_dir": str(artifact_dir),
        "results": results,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run OCR97 against Hugging Face SROIE receipt rows and score company/date/total/address fields.")
    parser.add_argument("--output-dir", required=True, help="Directory for downloaded images and artifacts.")
    parser.add_argument("--output", default="", help="Optional summary JSON path.")
    parser.add_argument("--split", default="test", help="SROIE split.")
    parser.add_argument("--offset", type=int, default=0, help="Dataset row offset.")
    parser.add_argument("--length", type=int, default=10, help="Number of rows to test.")
    parser.add_argument("--engine", default="local_image_preprocessed_best", help="Gateway OCR engine/router.")
    args = parser.parse_args(argv)

    result = run_sroie_benchmark(
        output_dir=Path(args.output_dir).expanduser(),
        split=args.split,
        offset=args.offset,
        length=args.length,
        engine=args.engine,
    )
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": result["dataset"], "split": result["split"], "offset": result["offset"], "case_count": result["case_count"], "score_avg": result["score_avg"], "field_totals": result["field_totals"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


