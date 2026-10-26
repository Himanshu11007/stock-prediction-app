"""
tests/test_news_catalyst_review.py — the read-only news-call review script
(scripts/research/news_catalyst_review.py): collects NEWS_CATALYST calls with
their headlines, link type, the move before the call and the results; the env
file is read literally; nothing secret is printed.
"""
import importlib.util
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from db.models.news import EventEntity
from db.models.prediction import EventClassification, MarketEvent, Prediction, PredictionOutcome, PredictionRun

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ncr", ROOT / "scripts" / "research" / "news_catalyst_review.py")
ncr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ncr)


def test_collects_news_calls_with_headline_link_and_results():
    e = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(e)
    now = datetime.now(timezone.utc)
    today = now.date().isoformat()
    with Session(e) as s:
        s.add(PredictionRun(run_id="R1", idempotency_key="TOMORROW_EOD:x", engine_version="v2", rule_version="r",
                            feature_set_version="f", run_type="TOMORROW_EOD", trading_date=today,
                            target_session_date=today, data_cutoff_at=now, config_hash="h"))
        ev = MarketEvent(source="drishti", source_event_id="1", symbol="ABC.NS", event_type="EARNINGS",
                         title="ABC profit jumps 40%", published_at=now, effective_available_at=now)
        s.add(ev)
        s.commit()
        s.add(EventClassification(event_id=ev.id, classifier_version="c1", direction="POSITIVE"))
        s.add(EventEntity(event_id=ev.id, entity_type="COMPANY", entity_key="ABC.NS", symbol="ABC.NS",
                          relation="SUBJECT"))
        for pid, setup in (("P1", "NEWS_CATALYST"), ("P2", "MOMENTUM_CONTINUATION")):
            s.add(Prediction(prediction_id=pid, run_id="R1", symbol="ABC.NS" if pid == "P1" else "XYZ.NS",
                             direction="UP", setup_type=setup, features={"ret_1": 0.045, "ret_5": 0.08},
                             reasons=["positive news", "technical setup disagrees (DOWN)"], event_ids=[ev.id]))
        s.commit()
        s.add(PredictionOutcome(prediction_id="P1", horizon_sessions=1, evaluator_version="e",
                                outcome_status="EVALUATED", directional_return=-0.03, hit=False))
        s.commit()
        calls = ncr.collect(s, 30)
    assert len(calls) == 1                                                   # only news calls
    c = calls[0]
    assert c["symbol"] == "ABC.NS" and c["already_moved"] and not c["inferred_only"]
    assert c["news"][0]["title"] == "ABC profit jumps 40%" and c["news"][0]["link"] == "named (subject)"
    assert c["results"][1]["hit"] is False
    summary = ncr._summary(calls, lambda c: "already moved" if c["already_moved"] else "not yet moved")
    assert summary["already moved"]["1d"] == {"evaluated": 1, "right": 0, "avg_return_in_call_direction": -0.03}


def test_env_file_is_read_literally(tmp_path, monkeypatch):
    f = tmp_path / "x.env"
    f.write_text("# c\nNCR_TEST_VALUE=a&b$(touch PWNED);c\nlower=no\n", encoding="utf-8")
    monkeypatch.delenv("NCR_TEST_VALUE", raising=False)
    ncr.load_env_file(f)
    import os
    assert os.environ["NCR_TEST_VALUE"] == "a&b$(touch PWNED);c" and "lower" not in os.environ
    assert not (tmp_path / "PWNED").exists()
    monkeypatch.delenv("NCR_TEST_VALUE")


def test_script_never_prints_the_database_url():
    text = (ROOT / "scripts" / "research" / "news_catalyst_review.py").read_text(encoding="utf-8")
    assert "print(os.environ" not in text and "DATABASE_URL\"]" not in text
