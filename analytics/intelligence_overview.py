"""
analytics/intelligence_overview.py — the user-facing AI Intelligence page.

Separates the kinds of "intelligence" and states plainly what each one is
worth, so nothing implies predictive skill that was not demonstrated
(docs/ML_EXPERIMENT_REGISTRY.md, docs/RANKING_VALIDATION_V1.md).
"""
from __future__ import annotations

from typing import Any

from sqlmodel import Session

import engine_runs.service as runs
import masters.service as masters
from ranking import presenter
from utils.market_session import market_status


def overview(session: Session) -> dict[str, Any]:
    features = masters.get_config(session, "app.features")
    weights, _ = masters.ranking_config(session)
    return {
        "sections": [
            {"key": "market", "title": "Market intelligence",
             "status": "Used in the StockLens Score",
             "text": "The NIFTY 50 regime (Bullish / Sideways / Bearish / High Volatility) from trend, momentum "
                     "and volatility of the index, recomputed at every analysis run.",
             "regime": presenter.regime_payload(runs.latest_market_regime(session)),
             "market_status": market_status(holidays=masters.get_config(session, "market.holidays"))},
            {"key": "fundamental", "title": "Fundamental Quality & Value Framework (FQVF)",
             "status": "Used in the StockLens Score",
             "text": "18 fixed checks on earnings, returns on capital, valuation, leverage and cash flow. Missing "
                     "data is shown as 'Not available', never as a failure. See each stock's Quality & Value checks."},
            {"key": "technical", "title": "Technical information",
             "status": "Used in the StockLens Score",
             "text": "Daily and weekly trend, 60-day momentum, volatility and drawdown relative to the analysed "
                     "universe. See each stock's Market tab."},
            {"key": "ml_signal", "title": "ML direction signal",
             "status": f"Informational only - weight {weights.get('ml_signal', 0):g} in the StockLens Score",
             "text": "A next-day up/down probability from a machine-learning model. Research on this model did not "
                     "demonstrate reliable predictive skill, so it does not affect scores or rankings and is shown "
                     "only for transparency.",
             "displayed": bool(features.get("ml_signal_display"))},
            {"key": "news", "title": "News sentiment",
             "status": "Not used in scoring",
             "text": "News items do not carry reliable publication timestamps in our data source, so they cannot "
                     "be shown as information that was known before any analysis date. News is not an input to "
                     "FQVF or the StockLens Score."},
            {"key": "legacy", "title": "Legacy signal analytics",
             "status": "Retired methodology",
             "text": "Analytics of BUY/SELL/HOLD signals from the earlier engine, mostly recorded before a "
                     "look-ahead defect was fixed. Shown for transparency only; not comparable to the current "
                     "StockLens Score and not evidence of skill."},
        ],
        "note": "StockLens provides analysis, not investment advice. No signal or score guarantees returns.",
    }
