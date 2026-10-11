"""
tests/test_signal_eval.py — strategy comparison: coverage, drawdown, sample
gates, news overlay on v2, category breakdown and a conservative verdict.
Synthetic trades only.
"""
import datetime as dt

from catalysts import signal_eval as se
from catalysts.replay import ReplayCall
from prediction_v2.backtest import Trade

D = [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(30)]


def t(strategy, day, ret, sym="A.NS", split="holdout", setup=""):
    return Trade(strategy=strategy, symbol=sym, cutoff=day, direction="UP", ret=ret, split=split, setup=setup)


def test_drawdown_is_peak_to_trough_of_the_daily_book():
    tr = [t("x", D[0], 0.02), t("x", D[1], -0.03), t("x", D[1], -0.01), t("x", D[2], 0.01)]
    dd = se.max_drawdown(tr, cost_bps=0)
    assert dd["sessions_traded"] == 3 and abs(dd["max_drawdown"] - 0.02) < 1e-9 and abs(dd["cumulative_net"] - 0.01) < 1e-9


def test_overlay_replays_and_gates():
    calls = [ReplayCall(D[0], "A.NS", "DOWN", 0.01, 0.005, -0.5, [1], "r", "EARNINGS"),
             ReplayCall(D[1], "B.NS", "UP", None, None, 0.5)]                     # unscored: dropped
    news = se.trades_from_replay(calls, "news_x", {D[0]: "holdout", D[1]: "holdout"}, {"A.NS": "Energy"})
    assert len(news) == 1 and news[0].setup == "EARNINGS" and news[0].sector == "Energy"
    v2 = [t("v2", D[0], -0.02), t("v2", D[2], 0.01, sym="C.NS")]
    both = se.overlay(v2, news, "v2_plus_news")
    assert {(x.symbol, x.direction) for x in both} == {("A.NS", "DOWN"), ("C.NS", "UP")}
    cmp = se.compare(v2 + news + both, {"holdout": D[:3]}, {"holdout": 60}, cost_bps=20, min_calls=2,
                     category_arms=("news_x",))
    arms = cmp["splits"]["holdout"]["arms"]
    assert arms["v2"]["status"] == "OK" and arms["news_x"]["status"] == "INSUFFICIENT_SAMPLE"
    assert arms["v2"]["coverage"] == round(2 / 60, 4)
    assert cmp["by_category"]["news_x"]["EARNINGS"]["calls"] == 1
    assert se.verdict(cmp, "news_x", "v2")["result"] == "INSUFFICIENT_SAMPLE"


def test_verdict_needs_an_interval_above_zero_and_the_comparator():
    good = [t("a", D[i], 0.01 + 0.001 * (i % 3)) for i in range(30)]
    flat = [t("b", D[i], 0.0 if i % 2 else 0.001) for i in range(30)]
    cmp = se.compare(good + flat, {"holdout": D}, {"holdout": 100}, cost_bps=10, min_calls=20)
    assert se.verdict(cmp, "a", "b")["result"] == "IMPROVES"
    assert se.verdict(cmp, "b", "a")["result"] == "NO_EVIDENCE_OF_IMPROVEMENT"
    assert se.verdict(cmp, "zzz", "a")["result"] == "NOT_AVAILABLE"
