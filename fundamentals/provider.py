"""
fundamentals/provider.py — market and fundamental data from Yahoo Finance.

Every value returned is either what the provider reported (after explicit,
documented unit conversion) or None. Nothing is estimated, filled or
defaulted. Invalid numbers (NaN, ±inf, non-numeric) become None and are
listed in `issues`, so data health can report them.

Unit conversions (verified against provider output, see docs/FQVF.md):
  debtToEquity      provider reports a percentage (46.3 = 0.463) -> ratio / 100
  dividendYield     provider reports a percentage (0.51 = 0.51%) -> / 100,
                    used only when dividendRate (absolute, per share) is absent
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import pandas as pd

ANNUAL_FIELDS = {
    "diluted_eps":          ("income_stmt", ("Diluted EPS", "Basic EPS")),
    "net_income":           ("income_stmt", ("Net Income", "Net Income Common Stockholders")),
    "revenue":              ("income_stmt", ("Total Revenue", "Operating Revenue")),
    "ebit":                 ("income_stmt", ("EBIT",)),
    "total_assets":         ("balance_sheet", ("Total Assets",)),
    "current_liabilities":  ("balance_sheet", ("Current Liabilities",)),
    "stockholders_equity":  ("balance_sheet", ("Stockholders Equity", "Common Stock Equity")),
    "total_debt":           ("balance_sheet", ("Total Debt",)),
    "operating_cash_flow":  ("cashflow", ("Operating Cash Flow",)),
    "free_cash_flow":       ("cashflow", ("Free Cash Flow",)),
    "capital_expenditure":  ("cashflow", ("Capital Expenditure",)),
}

INFO_FIELDS = {
    "sector":                "sector",
    "industry":              "industry",
    "currency":              "currency",
    "provider_price":        "currentPrice",
    "trailing_pe":           "trailingPE",
    "price_to_book":         "priceToBook",
    "book_value_per_share":  "bookValue",
    "price_to_sales":        "priceToSalesTrailing12Months",
    "provider_peg":          "pegRatio",
    "payout_ratio":          "payoutRatio",
    "trailing_eps":          "trailingEps",
    "market_cap":            "marketCap",
    "shares_outstanding":    "sharesOutstanding",
    "dividend_rate":         "dividendRate",
}


def clean_number(value: Any) -> float | None:
    """A finite float, or None. Never substitutes a value."""
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _annual_series(frames: dict[str, pd.DataFrame], issues: list[str]) -> list[dict]:
    """One dict per fiscal year (newest first); years with no data dropped."""
    by_year: dict[str, dict] = {}
    for field, (frame_name, labels) in ANNUAL_FIELDS.items():
        frame = frames.get(frame_name)
        if frame is None or frame.empty:
            continue
        label = next((l for l in labels if l in frame.index), None)
        if label is None:
            continue
        for col, raw in frame.loc[label].items():
            year = pd.Timestamp(col).date().isoformat()
            value = clean_number(raw)
            if raw is not None and value is None and not (isinstance(raw, float) and math.isnan(raw)):
                issues.append(f"invalid {field} for FY {year}: {raw!r}")
            by_year.setdefault(year, {"fiscal_year_end": year})[field] = value
    rows = [r for r in by_year.values() if any(v is not None for k, v in r.items() if k != "fiscal_year_end")]
    return sorted(rows, key=lambda r: r["fiscal_year_end"], reverse=True)


def normalise(info: dict, frames: dict[str, pd.DataFrame]) -> dict:
    """Pure transformation of provider payloads (testable without network)."""
    issues: list[str] = []
    data: dict[str, Any] = {}
    for key, provider_key in INFO_FIELDS.items():
        raw = info.get(provider_key)
        if key in ("sector", "industry", "currency"):
            data[key] = raw.strip() if isinstance(raw, str) and raw.strip() else None
            continue
        value = clean_number(raw)
        if raw is not None and value is None:
            issues.append(f"invalid {provider_key}: {raw!r}")
        data[key] = value

    de_pct = clean_number(info.get("debtToEquity"))
    data["debt_to_equity"] = de_pct / 100 if de_pct is not None else None

    price = data.get("provider_price")
    if data.get("dividend_rate") is not None and price:
        data["dividend_yield"] = data["dividend_rate"] / price
        data["dividend_yield_basis"] = "dividendRate / currentPrice"
    else:
        dy_pct = clean_number(info.get("dividendYield"))
        data["dividend_yield"] = dy_pct / 100 if dy_pct is not None else None
        data["dividend_yield_basis"] = "dividendYield / 100" if dy_pct is not None else None

    data["annual"] = _annual_series(frames, issues)
    data["issues"] = issues
    return data


def fetch_fundamentals(symbol: str) -> dict:
    """
    Returns {"status", "error", "fiscal_period_end", "data", "fetched_at"}.
    status: OK (summary + statements), PARTIAL (one of them missing),
    UNAVAILABLE (provider returned nothing usable), ERROR (exception).
    """
    import yfinance as yf

    fetched_at = datetime.now(timezone.utc)
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        frames = {}
        for name in ("income_stmt", "balance_sheet", "cashflow"):
            try:
                frames[name] = getattr(ticker, name)
            except Exception:  # one statement failing must not lose the others
                frames[name] = None
    except Exception as e:
        return {"status": "ERROR", "error": f"{type(e).__name__}: {e}"[:500],
                "fiscal_period_end": None, "data": None, "fetched_at": fetched_at}

    data = normalise(info, frames)
    has_summary = any(data.get(k) is not None for k in ("trailing_pe", "price_to_book", "market_cap"))
    has_annual = bool(data["annual"])
    if not has_summary and not has_annual:
        status, error = "UNAVAILABLE", "provider returned no fundamental data"
    elif has_summary and has_annual:
        status, error = "OK", None
    else:
        status = "PARTIAL"
        error = "no annual statements" if not has_annual else "no summary valuation fields"
    return {
        "status": status, "error": error,
        "fiscal_period_end": data["annual"][0]["fiscal_year_end"] if has_annual else None,
        "data": data, "fetched_at": fetched_at,
    }


def fetch_price_history(symbols: list[str], period: str = "2y") -> dict[str, pd.DataFrame | None]:
    """Daily adjusted OHLCV per symbol (one batched request). Missing symbols
    map to None; rows without a Close are dropped."""
    import yfinance as yf

    out: dict[str, pd.DataFrame | None] = {s: None for s in symbols}
    if not symbols:
        return out
    raw = yf.download(symbols, period=period, interval="1d", auto_adjust=True,
                      group_by="ticker", threads=True, progress=False)
    if raw is None or raw.empty:
        return out
    for s in symbols:
        try:
            df = raw[s] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            continue
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
        if df.empty:
            continue
        df.index = pd.to_datetime(df.index).tz_localize(None)
        out[s] = df
    return out
