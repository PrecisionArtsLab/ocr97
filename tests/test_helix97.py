import json
from pathlib import Path

from ocr97 import helix97


def _write_comparison(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    artifact = tmp_path / "case.ocr97.json"
    artifact.write_text(
        json.dumps(
            {
                "ok": True,
                "extracted_text": "Invoice summary Subtotal: $90.00 Total: $100.00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    comparison = {
        "summary": {"ocr97_vs_best_baseline_delta": 3},
        "engines": [
            {
                "engine": "ocr97",
                "score_avg": 77,
                "results": [
                    {
                        "id": "invoice_summary",
                        "artifact_path": str(artifact),
                        "input_path": str(tmp_path / "case.png"),
                        "score": {
                            "id": "invoice_summary",
                            "label": "Invoice summary",
                            "score": 77,
                            "field_score": 50,
                            "fields": [
                                {
                                    "name": "total",
                                    "type": "money",
                                    "expected": "100.00",
                                    "matched": False,
                                    "partial_score": 0.0,
                                    "failure_bucket": "candidate_found_but_not_selected",
                                    "ranked_candidates": [
                                        {
                                            "field": "total",
                                            "value": "$90.00",
                                            "normalized_value": "90",
                                            "source_line": "Invoice summary Subtotal: $90.00 Total: $100.00",
                                            "line_index": 0,
                                            "confidence": 0.9,
                                            "reason": "numeric candidate from label/context",
                                        },
                                        {
                                            "field": "total",
                                            "value": "$100.00",
                                            "normalized_value": "100",
                                            "source_line": "Invoice summary Subtotal: $90.00 Total: $100.00",
                                            "line_index": 0,
                                            "confidence": 0.7,
                                            "reason": "numeric candidate near requested label",
                                        },
                                    ],
                                    "source_evidence": {
                                        "source_line": "Invoice summary Subtotal: $90.00 Total: $100.00",
                                        "line_index": 0,
                                    },
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }
    path = tmp_path / "baseline_comparison.json"
    path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    return path


def _write_comparison_with_delta(tmp_path: Path, *, delta: int) -> Path:
    path = _write_comparison(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("summary", {})["ocr97_vs_best_baseline_delta"] = delta
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _write_clean_manifest(tmp_path: Path, *, score_avg: int, failure_count: int) -> Path:
    path = tmp_path / "run_summary.json"
    path.write_text(
        json.dumps({"score_avg": score_avg, "failure_count": failure_count}) + "\n",
        encoding="utf-8",
    )
    return path


def _write_strict_summary(tmp_path: Path, deltas) -> Path:
    path = tmp_path / "strict_hard_matrix_summary.json"
    path.write_text(
        json.dumps(
            {
                "variants": [
                    {"summary": {"ocr97_vs_best_baseline_delta": int(delta)}}
                    for delta in deltas
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_helix97_collects_failures_as_training_records(tmp_path):
    comparison = _write_comparison(tmp_path)

    manifest = helix97.collect_failure_records(comparison, output_dir=tmp_path / "helix")
    dataset = Path(manifest["dataset_path"])
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()]

    assert manifest["pipeline"] == "Helix97"
    assert manifest["record_count"] == 1
    assert rows[0]["field"]["failure_bucket"] == "candidate_found_but_not_selected"
    assert rows[0]["training_target"]["positive_candidate_index"] == 1
    assert rows[0]["gb10"]["role"] == "teacher_rationale_and_synthetic_variants"
    assert rows[0]["raw_ocr_training_gate"]["decision"] == "defer"


def test_helix97_trains_field_ranker_and_exports_layout_regions(tmp_path):
    comparison = _write_comparison(tmp_path)
    manifest = helix97.collect_failure_records(comparison, output_dir=tmp_path / "helix")
    dataset = Path(manifest["dataset_path"])

    model = helix97.train_field_ranker(dataset, output_dir=tmp_path / "helix")
    layout = helix97.export_layout_region_examples(dataset, output_dir=tmp_path / "helix")

    assert Path(model["model_path"]).exists()
    assert model["training"]["records_trainable"] == 1
    assert model["training"]["accuracy_on_trainable"] == 1.0
    assert Path(layout["layout_dataset_path"]).exists()
    assert layout["example_count"] == 1


def test_helix97_collects_many_and_evaluates_field_ranker(tmp_path):
    comparison_a = _write_comparison(tmp_path / "a")
    comparison_b = _write_comparison(tmp_path / "b")

    manifest = helix97.collect_failure_records_many(
        [comparison_a, comparison_b],
        output_dir=tmp_path / "helix",
    )
    model = helix97.train_field_ranker(Path(manifest["dataset_path"]), output_dir=tmp_path / "helix")
    evaluation = helix97.evaluate_field_ranker(
        Path(manifest["dataset_path"]),
        Path(model["model_path"]),
        output_dir=tmp_path / "helix",
    )

    assert manifest["record_count"] == 2
    assert evaluation["evaluated"] == 2
    assert evaluation["accuracy"] == 1.0
    assert Path(evaluation["markdown_path"]).exists()


def test_helix97_pipeline_writes_gb10_plan_and_raw_ocr_gate(tmp_path, monkeypatch):
    comparison = _write_comparison(tmp_path)
    monkeypatch.setattr(helix97, "gb10_status", lambda url: {"url": url, "reachable": True, "models": ["qwen2.5:72b"]})

    summary = helix97.run_pipeline(
        comparison,
        output_dir=tmp_path / "helix",
        gb10_url="http://gb10.local:11434",
        gb10_model="qwen2.5:72b",
    )

    assert summary["pipeline"] == "Helix97"
    assert summary["raw_ocr_gate"]["decision"] == "defer_raw_ocr_training"
    assert summary["promotion_gate"]["decision"] in {"hold_raw_ocr_promotion", "block_raw_ocr_promotion"}
    assert "checks" in summary["promotion_gate"]
    assert summary["gb10_training_plan"]["gb10"]["status"]["reachable"] is True
    assert Path(tmp_path / "helix" / "helix97_gb10_training_plan.md").exists()


def test_helix97_promotion_gate_blocks_when_evidence_missing(tmp_path, monkeypatch):
    comparison = _write_comparison_with_delta(tmp_path, delta=-2)
    monkeypatch.setattr(helix97, "gb10_status", lambda url: {"url": url, "reachable": True, "models": ["qwen2.5:72b"]})

    summary = helix97.run_pipeline(
        comparison,
        output_dir=tmp_path / "helix",
        gb10_url="http://gb10.local:11434",
        gb10_model="qwen2.5:72b",
    )

    assert summary["promotion_gate"]["status"] == "fail"
    gate_states = {item["status"] for item in summary["promotion_gate"]["checks"]}
    assert "missing" in gate_states


def test_helix97_promotion_gate_allows_when_clean_and_strict_pass(tmp_path, monkeypatch):
    comparison = _write_comparison_with_delta(tmp_path, delta=-2)
    clean_manifest = _write_clean_manifest(tmp_path, score_avg=99, failure_count=0)
    strict_summary = _write_strict_summary(tmp_path, deltas=[1, 2, 0])
    monkeypatch.setattr(helix97, "gb10_status", lambda url: {"url": url, "reachable": True, "models": ["qwen2.5:72b"]})

    summary = helix97.run_pipeline(
        comparison,
        output_dir=tmp_path / "helix",
        gb10_url="http://gb10.local:11434",
        gb10_model="qwen2.5:72b",
        clean_manifest_path=clean_manifest,
        strict_matrix_summary_path=strict_summary,
    )

    assert summary["promotion_gate"]["decision"] == "allow_raw_ocr_promotion"
    assert summary["promotion_gate"]["status"] == "pass"
    assert summary["gb10_training_plan"]["phases"][-1]["status"] == "allowed"


def test_helix97_gb10_teacher_augmentation_uses_local_endpoint(tmp_path, monkeypatch):
    comparison = _write_comparison(tmp_path)
    manifest = helix97.collect_failure_records(comparison, output_dir=tmp_path / "helix")

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "response": json.dumps(
                    {
                        "correction_reason_summary": "The value next to Total is the target.",
                        "hard_negative_patterns": ["Subtotal competing with Total"],
                        "synthetic_variant_prompts": ["Create invoice where subtotal appears before total."],
                        "layout_region_hint": "Prefer the label-value span after Total.",
                    }
                )
            }

    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return Response()

    monkeypatch.setattr(helix97.requests, "post", fake_post)

    result = helix97.augment_with_gb10_teacher(
        Path(manifest["dataset_path"]),
        output_dir=tmp_path / "helix",
        gb10_url="http://gb10.local:11434",
        gb10_model="qwen2.5:72b",
    )

    assert result["augmented"] == 1
    assert calls[0]["url"] == "http://gb10.local:11434/api/generate"
    row = json.loads((tmp_path / "helix" / "helix97_gb10_teacher_augmented.jsonl").read_text(encoding="utf-8"))
    assert row["teacher"]["correction_reason_summary"]


def test_helix97_raw_ocr_gate_accepts_multi_variant_summary(tmp_path):
    summary = {
        "rows": [
            {"variant": "clean", "ocr97_vs_best_baseline_delta": 5},
            {"variant": "noisy", "ocr97_vs_best_baseline_delta": 2},
        ]
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary) + "\n", encoding="utf-8")

    result = helix97.raw_ocr_training_gate(path, output_path=tmp_path / "gate.json")

    assert result["decision"] == "defer_raw_ocr_training"
    assert result["delta_vs_best_baseline"] == 2
