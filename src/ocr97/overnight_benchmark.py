from __future__ import annotations

import argparse
import gc
import io
import json
import os
import re
import time
import traceback
import unicodedata
import tempfile
from collections import Counter
from difflib import SequenceMatcher
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import requests

from .receipt_fields import append_receipt_fields, receipt_fields_from_candidates
from .sroie_runner import SROIE_DATASET, SROIE_SOURCE_PAGE, _download_image, _fetch_rows, score_sroie_payload


DEFAULT_ENGINES = ["tesseract", "rapidocr", "local_image_preprocessed_best"]
OPTIONAL_DIRECT_ENGINES = {"easyocr", "paddleocr", "surya", "doctr"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("_") or "item"


def _normalize_engine_name(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _artifact_payload_text(payload: Mapping[str, Any]) -> str:
    return str(payload.get("markdown") or payload.get("text") or "")


def _field_totals(results: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    totals: Dict[str, Dict[str, int]] = {}
    for row in results:
        score = row.get("score") if isinstance(row.get("score"), dict) else {}
        for field in list((score or {}).get("fields") or []):
            name = str(field.get("name") or "")
            bucket = totals.setdefault(name, {"hits": 0, "total": 0})
            bucket["hits"] += 1 if field.get("matched") else 0
            bucket["total"] += 1
    return {
        name: {**bucket, "accuracy": round((bucket["hits"] / float(max(1, bucket["total"]))) * 100.0, 2)}
        for name, bucket in sorted(totals.items())
    }


def _score_avg(results: Iterable[Mapping[str, Any]]) -> float:
    rows = list(results)
    if not rows:
        return 0.0
    return round(sum(float((row.get("score") or {}).get("score") or 0.0) for row in rows) / float(len(rows)), 2)


def _normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    return " ".join(re.findall(r"[\w]+(?:[.,:/-][\w]+)*", text, flags=re.UNICODE))


def _edit_distance(left: list[str], right: list[str]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, token in enumerate(left, 1):
        current = [i]
        for j, other in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (token != other)))
        previous = current
    return previous[-1]


def text_quality_metrics(extracted: str, expected_lines: Iterable[Any]) -> Dict[str, float]:
    """Score transcript accuracy plus receipt-layout proxies from SROIE line truth."""
    clean_text = str(extracted or "").split("\n\nReceipt fields:", 1)[0]
    truth_lines = [_normalize_text(line) for line in expected_lines if _normalize_text(line)]
    predicted_lines = [_normalize_text(line) for line in clean_text.splitlines() if _normalize_text(line)]
    truth = " ".join(truth_lines)
    predicted = " ".join(predicted_lines)
    truth_chars = list(truth)
    predicted_chars = list(predicted)
    truth_words = truth.split()
    predicted_words = predicted.split()
    cer = _edit_distance(truth_chars, predicted_chars) / float(max(1, len(truth_chars)))
    wer = _edit_distance(truth_words, predicted_words) / float(max(1, len(truth_words)))
    recovered = sum(1 for line in truth_lines if any(SequenceMatcher(None, line, got).ratio() >= 0.80 for got in predicted_lines))
    line_recall = recovered / float(max(1, len(truth_lines)))
    order_score = SequenceMatcher(None, truth_lines, predicted_lines).ratio()
    expected_numbers = Counter(re.findall(r"\d+(?:[.,:/-]\d+)*", truth))
    actual_numbers = Counter(re.findall(r"\d+(?:[.,:/-]\d+)*", predicted))
    number_hits = sum(min(count, actual_numbers[token]) for token, count in expected_numbers.items())
    number_recall = number_hits / float(max(1, sum(expected_numbers.values())))
    return {
        "cer": round(cer, 4),
        "wer": round(wer, 4),
        "line_recall": round(line_recall, 4),
        "reading_order_score": round(order_score, 4),
        "numeric_token_recall": round(number_recall, 4),
        "layout_table_proxy_score": round(((line_recall + order_score + number_recall) / 3.0) * 100.0, 2),
    }


class OCR97BenchAutopilot:
    """Resumable benchmark coordinator with bounded diagnostic authority."""

    def __init__(
        self,
        *,
        output_dir: Path,
        engines: list[str],
        split: str,
        offset: int,
        length: int,
        retries: int,
        enable_model_debug: bool,
        debug_model: str,
        ollama_url: str,
    ) -> None:
        self.output_dir = output_dir
        self.engines = [_normalize_engine_name(engine) for engine in engines if str(engine or "").strip()]
        self.split = split
        self.offset = int(offset)
        self.length = int(length)
        self.retries = max(0, int(retries))
        self.enable_model_debug = bool(enable_model_debug)
        self.debug_model = debug_model.strip()
        self.ollama_url = ollama_url.rstrip("/")
        self.image_dir = self.output_dir / "images"
        self.artifact_dir = self.output_dir / "artifacts"
        self.debug_dir = self.output_dir / "OCR97_debug"
        self.state_path = self.output_dir / "state.json"
        self.report_json_path = self.output_dir / "report.json"
        self.report_md_path = self.output_dir / "REPORT.md"
        self._client: Any = None
        self._app: Any = None
        self._direct_cache: Dict[str, Any] = {}
        self.state: Dict[str, Any] = {}

    def prepare(self, *, reset: bool = False) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        if reset and self.state_path.exists():
            self.state_path.unlink()
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
        else:
            self.state = {
                "started_at": _utc_now(),
                "updated_at": _utc_now(),
                "dataset": SROIE_DATASET,
                "source_page": SROIE_SOURCE_PAGE,
                "split": self.split,
                "offset": self.offset,
                "length": self.length,
                "engines": self.engines,
                "completed": {},
                "failures": [],
                "events": [],
            }
            self._save_state()

    def _save_state(self) -> None:
        self.state["updated_at"] = _utc_now()
        fd, raw = tempfile.mkstemp(prefix=f".{self.state_path.name}.", suffix=".tmp", dir=str(self.state_path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.state, handle, indent=2)
                handle.write("\n")
            os.replace(raw, self.state_path)
        finally:
            if os.path.exists(raw):
                os.unlink(raw)

    def _event(self, kind: str, message: str, **extra: Any) -> None:
        event = {"ts": _utc_now(), "kind": kind, "message": message, **extra}
        self.state.setdefault("events", []).append(event)
        self._save_state()

    def _completed_key(self, engine: str, row_idx: int) -> str:
        return f"{engine}:{row_idx}"

    def _image_path(self, *, row_idx: int, key: str) -> Path:
        return self.image_dir / f"{row_idx:04d}_{_safe_name(key)}.jpg"

    def _get_gateway_client(self) -> Any:
        if self._client is None:
            os.environ.setdefault("OCR97_OCR_SMOKE_REQUIRED", "0")
            os.environ.setdefault("OCR97_OCR_GATEWAY_PREWARM_ENABLED", "0")
            os.environ.setdefault("OCR97_OCR_GATEWAY_PREWARM_ON_STARTUP", "0")
            os.environ.setdefault("OCR97_OCR_SLO_P95_IMAGE_PREPROCESSOR_MS", "180000")
            os.environ.setdefault("OCR97_OCR_SLO_P95_SEMANTIC_MS", "1800000")
            os.environ.setdefault("OCR97_OCR_PREPROCESS_INCLUDE_TEXT", "1")
            from .server import create_app

            self._app = create_app(instance_name="ocr97_overnight_benchmark")
            self._client = self._app.test_client()
        return self._client

    def _reset_gateway_client(self, reason: str) -> None:
        self._event("gateway_reset", f"Resetting gateway client: {reason}")
        self._client = None
        self._app = None
        gc.collect()

    def _drop_gateway_client(self) -> None:
        # Silently release the test client after a successful row so the next
        # row starts with a fresh client. The Werkzeug test client holds
        # internal state that becomes invalid after one long-running multipart
        # POST; reusing it causes an instant failure on the next request.
        self._client = None
        self._app = None
        gc.collect()

    def _extract_gateway(self, image_path: Path, *, engine: str) -> Dict[str, Any]:
        client = self._get_gateway_client()
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
        payload = response.get_json(silent=True) or {}
        return {"status_code": response.status_code, "latency_ms": elapsed_ms, "payload": payload}

    def _extract_easyocr(self, image_path: Path) -> Dict[str, Any]:
        started = time.perf_counter()
        try:
            import easyocr  # type: ignore

            reader = self._direct_cache.get("easyocr")
            if reader is None:
                reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                self._direct_cache["easyocr"] = reader
            rows = reader.readtext(str(image_path), detail=0, paragraph=True)
            text = "\n".join(str(item) for item in rows if str(item).strip())
            payload = self._payload_from_text(text, engine="easyocr")
            return {"status_code": 200, "latency_ms": round((time.perf_counter() - started) * 1000.0, 2), "payload": payload}
        except Exception as exc:
            return self._direct_failure("easyocr", started, exc)

    def _extract_paddleocr(self, image_path: Path) -> Dict[str, Any]:
        started = time.perf_counter()
        try:
            from paddleocr import PaddleOCR  # type: ignore

            ocr = self._direct_cache.get("paddleocr")
            if ocr is None:
                try:
                    ocr = PaddleOCR(lang="en", use_textline_orientation=True)
                except Exception:
                    ocr = PaddleOCR(lang="en")
                self._direct_cache["paddleocr"] = ocr
            result = ocr.predict(str(image_path)) if hasattr(ocr, "predict") else ocr.ocr(str(image_path))
            from .baseline_compare import _flatten_paddle_rows
            text = _flatten_paddle_rows(result)
            payload = self._payload_from_text(text, engine="paddleocr")
            return {"status_code": 200, "latency_ms": round((time.perf_counter() - started) * 1000.0, 2), "payload": payload}
        except Exception as exc:
            return self._direct_failure("paddleocr", started, exc)

    def _extract_doctr(self, image_path: Path) -> Dict[str, Any]:
        started = time.perf_counter()
        try:
            from doctr.io import DocumentFile  # type: ignore
            from doctr.models import ocr_predictor  # type: ignore

            doc = DocumentFile.from_images(str(image_path))
            predictor = self._direct_cache.get("doctr")
            if predictor is None:
                predictor = ocr_predictor(pretrained=True)
                self._direct_cache["doctr"] = predictor
            result = predictor(doc)
            text = str(result.render() or "")
            payload = self._payload_from_text(text, engine="doctr")
            return {"status_code": 200, "latency_ms": round((time.perf_counter() - started) * 1000.0, 2), "payload": payload}
        except Exception as exc:
            return self._direct_failure("doctr", started, exc)

    def _extract_surya(self, image_path: Path) -> Dict[str, Any]:
        started = time.perf_counter()
        try:
            from PIL import Image
            from surya.common.surya.schema import TaskNames  # type: ignore
            from surya.detection import DetectionPredictor  # type: ignore
            from surya.foundation import FoundationPredictor  # type: ignore
            from surya.recognition import RecognitionPredictor  # type: ignore

            cached = self._direct_cache.get("surya")
            if cached is None:
                foundation = FoundationPredictor()
                cached = (RecognitionPredictor(foundation), DetectionPredictor())
                self._direct_cache["surya"] = cached
            recognition, detection = cached
            image = Image.open(image_path).convert("RGB")
            prediction = recognition(
                [image],
                task_names=[TaskNames.ocr_with_boxes],
                det_predictor=detection,
                math_mode=False,
                sort_lines=True,
            )[0]
            text = "\n".join(str(line.text) for line in prediction.text_lines if str(line.text).strip())
            payload = self._payload_from_text(text, engine="surya")
            return {"status_code": 200, "latency_ms": round((time.perf_counter() - started) * 1000.0, 2), "payload": payload}
        except Exception as exc:
            return self._direct_failure("surya", started, exc)

    def _direct_failure(self, engine: str, started: float, exc: Exception) -> Dict[str, Any]:
        return {
            "status_code": 599,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "payload": {"ok": False, "engine": engine, "error": f"{type(exc).__name__}: {exc}"},
        }

    def _payload_from_text(self, text: str, *, engine: str) -> Dict[str, Any]:
        candidates = [{"ok": bool(text.strip()), "engine": engine, "preprocess": "direct", "text": text, "markdown": text, "_selection_score": 0.0}]
        receipt_fields = receipt_fields_from_candidates(candidates)
        merged = append_receipt_fields(text, receipt_fields) if receipt_fields else text
        return {
            "ok": bool(text.strip()),
            "engine": engine,
            "text": merged,
            "markdown": merged,
            "receipt_fields": receipt_fields,
            "receipt_fields_used": bool(receipt_fields),
        }

    def _extract(self, image_path: Path, *, engine: str) -> Dict[str, Any]:
        if engine == "easyocr":
            return self._extract_easyocr(image_path)
        if engine == "paddleocr":
            return self._extract_paddleocr(image_path)
        if engine == "surya":
            return self._extract_surya(image_path)
        if engine == "doctr":
            return self._extract_doctr(image_path)
        return self._extract_gateway(image_path, engine=engine)

    def _debug_with_model(self, failure: Mapping[str, Any]) -> Dict[str, Any]:
        prompt = (
            "You are OCR97 operating under bounded overnight OCR benchmark authority.\n"
            "You may diagnose errors, recommend safe retries, dependency checks, engine skips, or resume actions.\n"
            "You may not edit source code, delete benchmark artifacts, or fabricate benchmark scores.\n\n"
            "Return concise JSON with keys: diagnosis, likely_cause, safe_next_action, retry_recommended.\n\n"
            f"Failure:\n{json.dumps(failure, indent=2)[:6000]}"
        )
        prompt_path = self.debug_dir / f"debug_prompt_{_safe_name(failure.get('engine'))}_{failure.get('row_idx', 'row')}_{int(time.time())}.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        if not self.enable_model_debug:
            return {"ok": False, "reason": "model_debug_disabled", "prompt_path": str(prompt_path)}
        try:
            payload = {
                "model": self.debug_model or os.getenv("OLLAMA_MODEL_CHAT", "gemma3:12b"),
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 600},
            }
            response = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=120)
            data = response.json() if response.ok else {"error": response.text}
            out_path = prompt_path.with_name(prompt_path.stem.replace("debug_prompt", "debug_reply") + ".json")
            out_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            return {"ok": bool(response.ok), "model": payload["model"], "prompt_path": str(prompt_path), "reply_path": str(out_path), "response": data}
        except Exception as exc:
            return {"ok": False, "prompt_path": str(prompt_path), "error": f"{type(exc).__name__}: {exc}"}

    def run(self) -> Dict[str, Any]:
        rows = _fetch_rows(split=self.split, offset=self.offset, length=self.length, dataset=SROIE_DATASET)
        self._event("start", "Starting overnight OCR benchmark", engines=self.engines, rows=len(rows))
        for engine in self.engines:
            for ordinal, item in enumerate(rows):
                row = dict(item.get("row") or {})
                row_idx = int(item.get("row_idx") if item.get("row_idx") is not None else self.offset + ordinal)
                completed_key = self._completed_key(engine, row_idx)
                if completed_key in dict(self.state.get("completed") or {}):
                    continue
                self._run_one(engine=engine, row_idx=row_idx, row=row)
        report = self._write_reports()
        self._event("complete", "Overnight OCR benchmark complete", report=str(self.report_md_path))
        return report

    def _run_one(self, *, engine: str, row_idx: int, row: Mapping[str, Any]) -> None:
        key = str(row.get("key") or f"row_{row_idx}")
        image = dict(row.get("image") or {})
        source = str(image.get("src") or "")
        expected = dict(row.get("entities") or {})
        expected_lines = list(row.get("words") or [])
        image_path = self._image_path(row_idx=row_idx, key=key)
        artifact_path = self.artifact_dir / f"{engine}_{row_idx:04d}_{_safe_name(key)}.json"

        try:
            _download_image(source, image_path)
        except Exception as exc:
            payload = {"ok": False, "engine": engine, "error": f"data_fetch_failed:{type(exc).__name__}: {exc}"}
            score = score_sroie_payload(payload, expected)
            artifact = {
                "dataset": SROIE_DATASET,
                "source_page": SROIE_SOURCE_PAGE,
                "engine_requested": engine,
                "split": self.split,
                "offset": row_idx,
                "key": key,
                "source": source.split("?")[0],
                "input_path": str(image_path),
                "status_code": 0,
                "latency_ms": 0,
                "attempt": 0,
                "ok": False,
                "engine": engine,
                "router": "",
                "selected_engine": "",
                "selected_preprocess": "",
                "receipt_fields_used": False,
                "receipt_fields": [],
                "field_consensus_used": False,
                "field_consensus": [],
                "expected": expected,
                "score": score,
                "text_metrics": text_quality_metrics("", expected_lines),
                "error": str(payload.get("error") or ""),
                "text": "",
            }
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
            completed = dict(self.state.get("completed") or {})
            completed[self._completed_key(engine, row_idx)] = {
                "engine": engine,
                "row_idx": row_idx,
                "key": key,
                "artifact_path": str(artifact_path),
                "score": score,
                "text_metrics": text_quality_metrics("", expected_lines),
                "latency_ms": 0,
                "ok": False,
            }
            self.state["completed"] = completed
            failure = {
                "engine": engine,
                "row_idx": row_idx,
                "key": key,
                "attempt": 0,
                "error": str(payload["error"]),
                "traceback": traceback.format_exc(limit=4),
                "artifact_path": str(artifact_path),
            }
            self.state.setdefault("failures", []).append(failure)
            self._save_state()
            self._event("case_data_fetch_failed", f"Dataset image fetch failed for {key}", engine=engine, row_idx=row_idx, error=failure["error"])
            return

        last_error: Dict[str, Any] = {}
        for attempt in range(self.retries + 1):
            try:
                self._event("case_start", f"Running {engine} on {key}", engine=engine, row_idx=row_idx, attempt=attempt)
                extraction = self._extract(image_path, engine=engine)
                payload = dict(extraction.get("payload") or {})
                score = score_sroie_payload(payload, expected)
                metrics = text_quality_metrics(_artifact_payload_text(payload), expected_lines)
                artifact = {
                    "dataset": SROIE_DATASET,
                    "source_page": SROIE_SOURCE_PAGE,
                    "engine_requested": engine,
                    "split": self.split,
                    "offset": row_idx,
                    "key": key,
                    "source": source.split("?")[0],
                    "input_path": str(image_path),
                    "status_code": extraction.get("status_code"),
                    "latency_ms": extraction.get("latency_ms"),
                    "attempt": attempt,
                    "ok": bool(payload.get("ok")),
                    "engine": str(payload.get("engine") or engine),
                    "router": str(payload.get("router") or ""),
                    "selected_engine": str(payload.get("selected_engine") or ""),
                    "selected_preprocess": str(payload.get("selected_preprocess") or ""),
                    "receipt_fields_used": bool(payload.get("receipt_fields_used")),
                    "receipt_fields": list(payload.get("receipt_fields") or []),
                    "field_consensus_used": bool(payload.get("field_consensus_used")),
                    "field_consensus": list(payload.get("field_consensus") or []),
                    "expected": expected,
                    "score": score,
                    "text_metrics": metrics,
                    "error": str(payload.get("error") or ""),
                    "text": _artifact_payload_text(payload),
                }
                artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
                completed = dict(self.state.get("completed") or {})
                completed[self._completed_key(engine, row_idx)] = {
                    "engine": engine,
                    "row_idx": row_idx,
                    "key": key,
                    "artifact_path": str(artifact_path),
                    "score": score,
                    "text_metrics": metrics,
                    "latency_ms": extraction.get("latency_ms"),
                    "ok": bool(payload.get("ok")),
                }
                self.state["completed"] = completed
                self._save_state()
                if not payload.get("ok"):
                    raise RuntimeError(str(payload.get("error") or f"{engine}_returned_not_ok"))
                self._drop_gateway_client()
                return
            except Exception as exc:
                last_error = {
                    "engine": engine,
                    "row_idx": row_idx,
                    "key": key,
                    "attempt": attempt,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=8),
                    "artifact_path": str(artifact_path),
                }
                self.state.setdefault("failures", []).append(last_error)
                debug = self._debug_with_model(last_error)
                self.state.setdefault("debug", []).append(debug)
                self._save_state()
                self._reset_gateway_client(f"{engine}:{key}:attempt_{attempt}_failed")
                if attempt >= self.retries:
                    self._event("case_failed", f"{engine} failed on {key}", engine=engine, row_idx=row_idx, error=last_error["error"])

    def _write_reports(self) -> Dict[str, Any]:
        completed = list((self.state.get("completed") or {}).values())
        by_engine: Dict[str, list[Mapping[str, Any]]] = {}
        for row in completed:
            by_engine.setdefault(str(row.get("engine") or ""), []).append(row)
        engine_rows = []
        for engine, rows in sorted(by_engine.items()):
            metric_rows = [dict(row.get("text_metrics") or {}) for row in rows]
            latencies = sorted(float(row.get("latency_ms") or 0.0) for row in rows)
            p95_index = max(0, min(len(latencies) - 1, int(round(0.95 * len(latencies) + 0.4999)) - 1)) if latencies else 0
            engine_rows.append(
                {
                    "engine": engine,
                    "case_count": len(rows),
                    "score_avg": _score_avg(rows),
                    "failure_count": sum(1 for row in rows if not row.get("ok")),
                    "field_totals": _field_totals(rows),
                    "latency_avg_ms": round(sum(float(row.get("latency_ms") or 0.0) for row in rows) / float(max(1, len(rows))), 2),
                    "latency_p95_ms": round(latencies[p95_index], 2) if latencies else 0.0,
                    "text_metrics": {
                        key: round(sum(float(metric.get(key) or 0.0) for metric in metric_rows) / float(max(1, len(metric_rows))), 4)
                        for key in ("cer", "wer", "line_recall", "reading_order_score", "numeric_token_recall", "layout_table_proxy_score")
                    },
                }
            )
        report = {
            "generated_at": _utc_now(),
            "dataset": SROIE_DATASET,
            "source_page": SROIE_SOURCE_PAGE,
            "split": self.split,
            "offset": self.offset,
            "length": self.length,
            "engines": self.engines,
            "output_dir": str(self.output_dir),
            "artifact_dir": str(self.artifact_dir),
            "state_path": str(self.state_path),
            "engine_results": engine_rows,
            "failure_count": len(list(self.state.get("failures") or [])),
            "failures": list(self.state.get("failures") or [])[-20:],
        }
        self.report_json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        self.report_md_path.write_text(self._render_markdown(report), encoding="utf-8")
        return report

    def _render_markdown(self, report: Mapping[str, Any]) -> str:
        lines = [
            "# OCR97 Overnight Benchmark Report",
            "",
            f"- Generated: `{report.get('generated_at')}`",
            f"- Dataset: `{report.get('dataset')}`",
            f"- Source: {report.get('source_page')}",
            f"- Split/offset/length: `{report.get('split')}` / `{report.get('offset')}` / `{report.get('length')}`",
            f"- Artifacts: `{report.get('artifact_dir')}`",
            f"- State: `{report.get('state_path')}`",
            "",
            "## Engine Summary",
            "",
            "| Engine | Cases | Score Avg | CER | WER | Layout proxy | Avg / p95 ms | Field Totals |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for row in list(report.get("engine_results") or []):
            fields = ", ".join(
                f"{name} {bucket.get('hits')}/{bucket.get('total')} ({bucket.get('accuracy')}%)"
                for name, bucket in dict(row.get("field_totals") or {}).items()
            )
            metrics = dict(row.get("text_metrics") or {})
            lines.append(f"| `{row.get('engine')}` | {row.get('case_count')} | {row.get('score_avg')} | {metrics.get('cer')} | {metrics.get('wer')} | {metrics.get('layout_table_proxy_score')} | {row.get('latency_avg_ms')} / {row.get('latency_p95_ms')} | {fields} |")
        lines.extend(["", "## OCR97 Benchmark Authority", ""])
        lines.extend(
            [
                "During this run OCR97 is authorized to retry failed cases, recreate the in-process OCR97 gateway client, write diagnostic prompts/replies, skip unavailable optional engines after captured failures, and resume from `state.json`.",
                "",
                "OCR97 is not authorized to edit source code, delete artifacts, change expected labels, or fabricate missing field values while the run is unattended.",
                "",
                f"- Failure count: `{report.get('failure_count')}`",
            ]
        )
        failures = list(report.get("failures") or [])
        if failures:
            lines.extend(["", "## Recent Failures", ""])
            for failure in failures[-10:]:
                lines.append(f"- `{failure.get('engine')}` row `{failure.get('row_idx')}` `{failure.get('key')}`: {failure.get('error')}")
        lines.append("")
        return "\n".join(lines)


def _parse_engines(raw: str) -> list[str]:
    engines = [_normalize_engine_name(item) for item in str(raw or "").split(",") if item.strip()]
    return engines or list(DEFAULT_ENGINES)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run a resumable overnight OCR97 head-to-head SROIE benchmark with OCR97 diagnostics.")
    parser.add_argument("--output-dir", required=True, help="Output directory for state, artifacts, and reports.")
    parser.add_argument("--split", default="test", help="SROIE split.")
    parser.add_argument("--offset", type=int, default=0, help="Dataset offset.")
    parser.add_argument("--length", type=int, default=50, help="Number of rows to benchmark.")
    parser.add_argument("--engines", default=",".join(DEFAULT_ENGINES), help="Comma-separated engines. Gateway engines plus optional easyocr,paddleocr,surya,doctr.")
    parser.add_argument("--retries", type=int, default=1, help="Retries per engine/case.")
    parser.add_argument("--reset", action="store_true", help="Start a new run instead of resuming state.json.")
    parser.add_argument("--model-debug", action="store_true", help="Ask a local Ollama model for diagnostic notes after failures.")
    parser.add_argument("--debug-model", default=os.getenv("OCR97_BENCH_DEBUG_MODEL", os.getenv("OLLAMA_MODEL_CHAT", "gemma3:12b")), help="Ollama model for failure diagnostics.")
    parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"), help="Ollama base URL.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare state and print plan without running OCR.")
    args = parser.parse_args(argv)

    autopilot = OCR97BenchAutopilot(
        output_dir=Path(args.output_dir).expanduser(),
        engines=_parse_engines(args.engines),
        split=args.split,
        offset=args.offset,
        length=args.length,
        retries=args.retries,
        enable_model_debug=args.model_debug,
        debug_model=args.debug_model,
        ollama_url=args.ollama_url,
    )
    autopilot.prepare(reset=args.reset)
    if args.dry_run:
        print(json.dumps({"ok": True, "state_path": str(autopilot.state_path), "engines": autopilot.engines, "length": args.length}, indent=2))
        return 0
    report = autopilot.run()
    print(json.dumps({"ok": True, "report": str(autopilot.report_md_path), "engine_results": report.get("engine_results")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
