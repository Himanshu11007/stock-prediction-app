"""Pydantic request/response models for /api/v1/watchlist/*."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class WatchlistItemResponse(BaseModel):
    id: int
    symbol: str
    stock_name: str
    # Optional: a watchlist entry is for following a stock's analysis; a
    # purchase price/date is an optional personal note.
    buy_price: Optional[float] = None
    buy_date: Optional[str] = None
    quantity: float
    created_at: datetime


class WatchlistAddRequest(BaseModel):
    symbol: str
    buy_price: Optional[float] = Field(default=None, gt=0)
    buy_date: Optional[str] = None
    quantity: float = Field(default=1, gt=0)


class WatchlistAlertUpdate(BaseModel):
    score_changes: Optional[bool] = None
    rank_changes: Optional[bool] = None
    fqvf_changes: Optional[bool] = None
    status_changes: Optional[bool] = None
    muted: Optional[bool] = None
