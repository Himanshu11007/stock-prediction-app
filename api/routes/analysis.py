"""
api/routes/analysis.py — Single-stock analysis endpoint.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api import services
from api.schemas import AnalyzeStockRequest, AnalyzeStockResponse
from utils.logger import get_logger, log_exception

logger = get_logger(__name__)
router = APIRouter()


@router.post("/analyze-stock", response_model=AnalyzeStockResponse)
def analyze_stock(payload: AnalyzeStockRequest):
    """
    Run the full StockAI Pro analysis pipeline for one symbol.

    Reuses the exact same step order as app.py's "Analyse Stock" tab:
    load data → engineer features → train model → predict → fetch news →
    sentiment → regime → multi-timeframe trend → confluence signal → risk.
    """
    symbol = payload.symbol.strip()
    if not symbol:
        raise HTTPException(status_code=400, detail="symbol must not be empty")

    try:
        result = services.analyze_stock(symbol)
        return result

    except ValueError as e:
        # Invalid symbol / no data / bad target distribution
        raise HTTPException(status_code=400, detail=str(e))

    except RuntimeError as e:
        log_exception(logger, f"analyze_stock pipeline error for {symbol}", e)
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        log_exception(logger, f"analyze_stock unexpected error for {symbol}", e)
        raise HTTPException(status_code=500, detail="Internal error during analysis")