from pathlib import Path

from ocr97.overnight_benchmark import text_quality_metrics
from ocr97.week_benchmark_campaign import WeekCampaign, comparative_grade, safe_json


def test_text_quality_metrics_exact_text() -> None:
    result = text_quality_metrics("Invoice 42\nTotal 19.99", ["Invoice 42", "Total 19.99"])
    assert result["cer"] == 0
    assert result["wer"] == 0
    assert result["line_recall"] == 1
    assert result["numeric_token_recall"] == 1
    assert result["layout_table_proxy_score"] == 100


def test_comparative_grade_is_mechanical_and_bounded() -> None:
    rows = []
    for engine, latency in [
        ("local_image_preprocessed_best", 100), ("tesseract", 200),
        ("paddleocr", 220), ("surya", 250), ("doctr", 260),
    ]:
        rows.append({
            "engine": engine, "case_count": 50, "failure_count": 0, "score_avg": 90,
            "latency_avg_ms": latency,
            "text_metrics": {"cer": 0.05, "wer": 0.1, "layout_table_proxy_score": 90},
        })
    result = comparative_grade(rows, [{"engine": "ocr97", "score_avg": 90}])
    assert 85 <= result["score"] <= 100
    assert sum(bucket["max"] for bucket in result["rubric"].values()) == 100


def test_campaign_state_is_resumable(tmp_path: Path) -> None:
    first = WeekCampaign(tmp_path, notify=False, run_id="test_run")
    assert first.next_phase()["id"] == "preflight"
    first.state["phases"][0]["status"] = "complete"
    first.save()
    second = WeekCampaign(tmp_path, notify=False)
    assert second.next_phase()["id"] == "sroie_tesseract"
    assert safe_json(tmp_path / "campaign_state.json")["active_run_id"] == "test_run"


def test_failed_phase_is_bounded(tmp_path: Path, monkeypatch) -> None:
    campaign = WeekCampaign(tmp_path, max_phase_attempts=1, notify=False)
    monkeypatch.setattr(campaign, "_preflight", lambda: {"ok": False, "error": "offline"})
    result = campaign.run_next()
    assert result["phase_status"] == "blocked"
    assert campaign.next_phase()["id"] == "sroie_tesseract"
