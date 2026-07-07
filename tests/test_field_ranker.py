import json

from ocr97.field_evidence import field_candidates
from ocr97.field_ranker import DEFAULT_WEIGHTS, rerank_candidates


def test_learned_ranker_can_reorder_candidates():
    model = {
        "weights": {
            "bias": 0.0,
            "candidate_confidence": 0.0,
            "value_near_field_alias": 2.0,
            "immediate_after_field_alias": 1.0,
            "labels_between_alias_and_value": -1.0,
        }
    }
    candidates = [
        {
            "field": "total",
            "value": "$90.00",
            "normalized_value": "90",
            "source_line": "Subtotal: $90.00 Total: $100.00",
            "line_index": 0,
            "confidence": 0.95,
            "reason": "numeric candidate from label/context",
        },
        {
            "field": "total",
            "value": "$100.00",
            "normalized_value": "100",
            "source_line": "Subtotal: $90.00 Total: $100.00",
            "line_index": 0,
            "confidence": 0.70,
            "reason": "numeric candidate from label/context",
        },
    ]

    ranked = rerank_candidates(candidates, {"name": "total", "aliases": ["Total"], "type": "money"}, model=model)

    assert ranked[0]["normalized_value"] == "100"
    assert ranked[0]["learned_ranker_used"] is True


def test_field_candidates_uses_configured_helix97_model(tmp_path, monkeypatch):
    model_path = tmp_path / "model.json"
    model_path.write_text(
        json.dumps(
            {
                "weights": {
                    "bias": 0.0,
                    "candidate_confidence": 0.1,
                    "value_near_field_alias": 2.0,
                    "immediate_after_field_alias": 1.0,
                    "labels_between_alias_and_value": -1.0,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OCR97_HELIX97_FIELD_RANKER_MODEL", str(model_path))

    candidates = field_candidates(
        "Invoice summary Subtotal: $90.00 Total: $100.00",
        {"name": "total", "aliases": ["Total"], "type": "money"},
    )

    assert candidates[0]["normalized_value"] == "100"
    assert candidates[0]["learned_ranker_used"] is True
    assert "learned_rank_score" in candidates[0]


def test_learned_ranker_prefers_identifier_text_with_digits():
    candidates = [
        {
            "field": "invoice_number",
            "value": "summary",
            "normalized_value": "summary",
            "source_line": "Invoice: summary",
            "line_index": 0,
            "confidence": 0.85,
            "reason": "text candidate from label",
        },
        {
            "field": "invoice_number",
            "value": "INV 2048",
            "normalized_value": "inv 2048",
            "source_line": "Invoice: INV 2048",
            "line_index": 1,
            "confidence": 0.85,
            "reason": "text candidate from label",
        },
    ]

    ranked = rerank_candidates(candidates, {"name": "invoice_number", "aliases": ["Invoice"], "type": "text"}, model={"weights": DEFAULT_WEIGHTS})

    assert ranked[0]["normalized_value"] == "inv 2048"
