"""
api/routes/analysis.py — Single-stock analysis endpoint.

Routes contain no business logic — they call services.py and wrap the
result in the standard response envelope. ValueError / KeyError / generic
Exception raised by services.py are caught by the centralized handlers in
api/main.py (400 / 404 / 500 respectively), so no local try/except is
needed here for those cases.
"""
from __future__ import annotations

from fastapi import APIRouter

from api import services
from api.schemas import AnalyzeStockRequest, success_envelope

router = APIRouter()


@router.post("/analyze-stock")
def analyze_stock(payload: AnalyzeStockRequest):
    """
    Run the full StockAI Pro analysis pipeline for one symbol.

    Reuses the exact same step order as app.py's "Analyse Stock" tab:
    load data → engineer features → train model → predict → fetch news →
    sentiment → regime → multi-timeframe trend → confluence signal → risk.

    Raises ValueError (→ 400) for invalid/unknown symbols or insufficient
    data — handled centrally in api/main.py.
    """
    symbol = payload.symbol.strip()
    if not symbol:
        raise ValueError("symbol must not be empty")

    result = services.analyze_stock(symbol)
    return success_envelope(result, message=f"Analysis complete for {symbol}")