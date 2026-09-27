"""api/routes/stocks.py — stock discovery for normal authenticated users.

Distinct from api/routes/admin.py's /admin/stocks (ADMIN-only, exposes
active/analysis_enabled toggles and bookkeeping timestamps). This is the
endpoint the mobile app's stock search/selection is meant to use - any
authenticated user, active stocks only, minimal fields.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

import stocks.service as stocks_service
from api.schemas_stocks import StockSearchResultResponse
from auth.dependencies import get_current_user
from db.session import get_session

router = APIRouter(prefix="/stocks", dependencies=[Depends(get_current_user)])


@router.get("", response_model=list[StockSearchResultResponse])
def search_stocks(
    search: str | None = None, limit: int = 50, offset: int = 0, session: Session = Depends(get_session)
):
    return stocks_service.search_stocks(session, search=search, limit=limit, offset=offset)


@router.get("/{symbol}", response_model=StockSearchResultResponse)
def get_stock(symbol: str, session: Session = Depends(get_session)):
    company = stocks_service.get_stock(session, symbol)
    if company is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Stock not found")
    return company
