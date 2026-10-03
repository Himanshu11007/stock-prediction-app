"""
ranking/service.py — StockRankingService: the StockAI Score.

The score is a weighted average of component scores (each 0-100), computed
only over the components that are available for that stock; the weights of
missing components are dropped and the remaining weights renormalised.
`coverage` (available weight / total weight) is always reported with it.
Full methodology: docs/RANKING_METHODOLOGY.md.

Components
  quality           FQVF checks 1, 2, 5, 12, 16   (PASS 100 / WARNING 50 / FAIL 0)
  valuation         FQVF checks 6-11, 14
  financial_health  FQVF checks 13, 15, 17
  sector_outlook    FQVF check 18
  technical_trend   (daily + weekly trend score, -1..+1) mapped to 0-100
  momentum          percentile of 60-day return within the analysed universe
  market_regime     per-stock regime score (-1..+1) mapped to 0-100
  risk              100 - mean percentile of (annual volatility, 1y max drawdown)
  ml_signal         ensemble P(up) x 100 — default weight 0 (see technical.ML_DISCLAIMER)

The score ranks stocks against each other on the same inputs. It is not a
probability, a price target or a return forecast.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from config import MARKET_DATA_STALE_DAYS, MIN_AVG_VOLUME, RANKING_ENGINE_VERSION

STATUS_POINTS = {"PASS": 100.0, "WARNING": 50.0, "FAIL": 0.0}

FQVF_COMPONENTS = {
    "quality": (1, 2, 5, 12, 16),
    "valuation": (6, 7, 8, 9, 10, 11, 14),
    "financial_health": (13, 15, 17),
    "sector_outlook": (18,),
}

COMPONENT_LABELS = {
    "quality": "Business quality (stability, EPS, ROE, ROCE)",
    "valuation": "Valuation (PE, peers, intrinsic value, PEG, PB, PSR)",
    "financial_health": "Financial health (leverage, cash flow, dividend)",
    "sector_outlook": "Sector outlook",
    "technical_trend": "Technical trend (daily and weekly)",
    "momentum": "Momentum (60-day return vs universe)",
    "market_regime": "Stock price regime",
    "risk": "Risk (volatility and drawdown vs universe)",
    "ml_signal": "ML direction signal (informational)",
}

DEFAULT_WEIGHTS: dict[str, float] = {
    "quality": 25.0,
    "valuation": 20.0,
    "financial_health": 15.0,
    "technical_trend": 10.0,
    "momentum": 10.0,
    "risk": 10.0,
    "sector_outlook": 5.0,
    "market_regime": 5.0,
    "ml_signal": 0.0,
}

DEFAULT_RULES: dict[str, float] = {
    "min_score_coverage": 0.60,       # share of total weight that must be available
    "min_fqvf_coverage": 0.50,        # share of the 16 scored FQVF checks evaluated
    "min_avg_volume_20d": float(MIN_AVG_VOLUME),
    "max_market_data_age_days": float(MARKET_DATA_STALE_DAYS),
    "strong_component": 70.0,         # component score shown as a positive factor
    "weak_component": 30.0,           # component score shown as a risk
}


def validate_weights(weights: dict[str, float]) -> dict[str, float]:
    unknown = set(weights) - set(DEFAULT_WEIGHTS)
    if unknown:
        raise ValueError(f"Unknown ranking components: {sorted(unknown)}")
    merged = {**DEFAULT_WEIGHTS, **{k: float(v) for k, v in weights.items()}}
    if any(v < 0 for v in merged.values()):
        raise ValueError("Ranking weights must be >= 0")
    if sum(merged.values()) <= 0:
        raise ValueError("At least one ranking weight must be positive")
    return merged


def validate_rules(rules: dict[str, float]) -> dict[str, float]:
    unknown = set(rules) - set(DEFAULT_RULES)
    if unknown:
        raise ValueError(f"Unknown ranking rules: {sorted(unknown)}")
    merged = {**DEFAULT_RULES, **{k: float(v) for k, v in rules.items()}}
    for k in ("min_score_coverage", "min_fqvf_coverage"):
        if not 0 <= merged[k] <= 1:
            raise ValueError(f"{k} must be between 0 and 1")
    return merged


@dataclass
class RankingInput:
    symbol: str
    name: str
    sector: Optional[str]
    fqvf: Optional[dict]                 # FQVFResult.to_dict()
    technical: Optional[dict]            # ranking.technical.technical_snapshot
    ml: Optional[dict]                   # ranking.technical.ml_signal
    market_status: str                   # OK / STALE / UNAVAILABLE / ERROR
    market_age_days: Optional[int]
    fundamentals_status: str             # OK / PARTIAL / UNAVAILABLE / ERROR
    company_active: bool = True
    company_tradable: bool = True
    freshness: dict = field(default_factory=dict)


def _fqvf_component(fqvf: Optional[dict], ids: tuple[int, ...]) -> tuple[Optional[float], list[int]]:
    if not fqvf:
        return None, []
    pts = [(c["id"], STATUS_POINTS[c["status"]]) for c in fqvf["checks"]
           if c["id"] in ids and c["status"] in STATUS_POINTS]
    if not pts:
        return None, []
    return round(sum(p for _, p in pts) / len(pts), 1), [i for i, _ in pts]


MIN_PERCENTILE_POOL = 10


def _percentiles(values: dict[str, Optional[float]]) -> dict[str, float]:
    """Percentile rank (0-100, ties averaged) among symbols with a value.
    Fewer than MIN_PERCENTILE_POOL values: no percentile (a rank among a
    handful of stocks is not meaningful, and a default would be fabricated)."""
    s = pd.Series({k: v for k, v in values.items() if v is not None}, dtype=float)
    if len(s) < MIN_PERCENTILE_POOL:
        return {}
    ranks = s.rank(method="average")
    return {k: round(float((r - 1) / (len(s) - 1) * 100), 1) for k, r in ranks.items()}


class StockRankingService:
    def __init__(self, weights: Optional[dict] = None, rules: Optional[dict] = None):
        self.weights = validate_weights(weights or {})
        self.rules = validate_rules(rules or {})
        self.version = RANKING_ENGINE_VERSION

    def rank(self, inputs: list[RankingInput], reference: Optional[dict[str, dict]] = None) -> list[dict]:
        """`reference`: technical metrics of other stocks that join the
        momentum/risk percentile pools but are not scored (single-stock runs)."""
        tech = {**(reference or {}), **{i.symbol: (i.technical or {}) for i in inputs}}
        momentum_pct = _percentiles({s: t.get("return_60d") for s, t in tech.items()})
        vol_pct = _percentiles({s: t.get("volatility_annual") for s, t in tech.items()})
        dd_pct = _percentiles({s: (abs(t["max_drawdown_1y"]) if t.get("max_drawdown_1y") is not None else None)
                               for s, t in tech.items()})

        results = [self._score_one(i, momentum_pct, vol_pct, dd_pct) for i in inputs]
        eligible = sorted((r for r in results if r["eligible"]),
                          key=lambda r: (-r["stockai_score"], r["symbol"]))
        for n, r in enumerate(eligible, 1):
            r["rank"] = n
        return results

    def _score_one(self, i: RankingInput, momentum_pct, vol_pct, dd_pct) -> dict:
        t = i.technical or {}
        comps: dict[str, dict[str, Any]] = {}

        for name, ids in FQVF_COMPONENTS.items():
            score, used = _fqvf_component(i.fqvf, ids)
            comps[name] = {"score": score, "basis": f"FQVF checks {list(ids)}; evaluated {used}" if used
                           else "no FQVF check in this group could be evaluated"}

        ts = t.get("trend_score")
        comps["technical_trend"] = {
            "score": round((ts + 1) * 50, 1) if ts is not None else None,
            "basis": f"daily {t.get('trend_daily')}, weekly {t.get('trend_weekly')}" if ts is not None
            else "insufficient price history for trend"}
        comps["momentum"] = {
            "score": momentum_pct.get(i.symbol),
            "basis": (f"60-day return {t['return_60d']:.1%}, percentile within analysed universe"
                      if i.symbol in momentum_pct else
                      "60-day return unavailable" if t.get("return_60d") is None else
                      f"fewer than {MIN_PERCENTILE_POOL} stocks to rank against")}
        rs = t.get("regime_score")
        comps["market_regime"] = {
            "score": round((rs + 1) * 50, 1) if rs is not None else None,
            "basis": f"{t.get('regime')}: {t.get('regime_reason')}" if rs is not None else "regime unavailable"}
        if i.symbol in vol_pct and i.symbol in dd_pct:
            comps["risk"] = {"score": round(100 - (vol_pct[i.symbol] + dd_pct[i.symbol]) / 2, 1),
                             "basis": (f"annual volatility {t['volatility_annual']:.1%}, 1y max drawdown "
                                       f"{t['max_drawdown_1y']:.1%}, ranked within analysed universe")}
        else:
            comps["risk"] = {"score": None, "basis": "volatility/drawdown unavailable, or fewer than "
                                                     f"{MIN_PERCENTILE_POOL} stocks to rank against"}
        ml = i.ml or {}
        comps["ml_signal"] = {
            "score": round(ml["probability_up"] * 100, 1) if ml.get("available") else None,
            "basis": ml.get("disclaimer") if ml.get("available") else (ml.get("reason") or "not computed")}

        total_w = sum(self.weights.values())
        avail = {k: c for k, c in comps.items() if c["score"] is not None and self.weights[k] > 0}
        avail_w = sum(self.weights[k] for k in avail)
        score = round(sum(self.weights[k] * c["score"] for k, c in avail.items()) / avail_w, 1) if avail_w else None
        coverage = round(avail_w / total_w, 3) if total_w else 0.0
        for k, c in comps.items():
            c["weight"] = self.weights[k]
            c["label"] = COMPONENT_LABELS[k]
            c["contribution"] = (round(self.weights[k] * c["score"] / avail_w, 2)
                                 if k in avail and avail_w else None)

        reasons = self._ineligible(i, score, coverage, t)
        return {
            "symbol": i.symbol, "name": i.name, "sector": i.sector,
            "stockai_score": score, "score_coverage": coverage,
            "eligible": not reasons, "ineligible_reasons": reasons, "rank": None,
            "components": comps,
            "positives": self._positives(i, comps),
            "risks": self._risks(i, comps, t),
            "freshness": i.freshness,
            "engine_version": self.version,
        }

    def _ineligible(self, i: RankingInput, score, coverage, t) -> list[str]:
        r = self.rules
        reasons = []
        if not i.company_active:
            reasons.append("stock is inactive in the Stock Master")
        if not i.company_tradable:
            reasons.append("stock is marked non-tradable")
        if i.market_status != "OK":
            reasons.append(f"market data {i.market_status.lower()}")
        elif i.market_age_days is not None and i.market_age_days > r["max_market_data_age_days"]:
            reasons.append(f"market data is {i.market_age_days} days old")
        if i.fundamentals_status not in ("OK", "PARTIAL"):
            reasons.append(f"fundamentals {i.fundamentals_status.lower()}")
        fq_cov = (i.fqvf or {}).get("coverage", 0.0)
        if fq_cov < r["min_fqvf_coverage"]:
            reasons.append(f"FQVF coverage {fq_cov:.0%} below {r['min_fqvf_coverage']:.0%}")
        if score is None or coverage < r["min_score_coverage"]:
            reasons.append(f"score coverage {coverage:.0%} below {r['min_score_coverage']:.0%}")
        vol = t.get("avg_volume_20d")
        if vol is None or vol < r["min_avg_volume_20d"]:
            reasons.append(f"20-day average volume below {r['min_avg_volume_20d']:,.0f} (liquidity)")
        return reasons

    def _positives(self, i: RankingInput, comps) -> list[str]:
        out = [c["explanation"] for c in (i.fqvf or {}).get("checks", [])
               if c["scored"] and c["status"] == "PASS"]
        out += [f"Strong {c['label'].lower()} ({c['score']:.0f}/100)"
                for k, c in comps.items()
                if c["score"] is not None and c["weight"] > 0 and c["score"] >= self.rules["strong_component"]
                and k not in FQVF_COMPONENTS]
        return out

    def _risks(self, i: RankingInput, comps, t) -> list[str]:
        out = [f"{c['name']}: {c['explanation']}" for c in (i.fqvf or {}).get("checks", [])
               if c["scored"] and c["status"] in ("FAIL", "WARNING")]
        out += [f"Weak {c['label'].lower()} ({c['score']:.0f}/100)"
                for k, c in comps.items()
                if c["score"] is not None and c["weight"] > 0 and c["score"] <= self.rules["weak_component"]
                and k not in FQVF_COMPONENTS]
        na = (i.fqvf or {}).get("counts", {}).get("NOT_AVAILABLE", 0)
        if na:
            out.append(f"{na} of 18 FQVF checks could not be evaluated (data not available)")
        if t.get("volatility_annual") is not None and t["volatility_annual"] > 0.45:
            out.append(f"High volatility ({t['volatility_annual']:.0%} annualised)")
        return out
