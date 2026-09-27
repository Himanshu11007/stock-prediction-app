"""Pydantic request/response models for /api/v1/watchlist/*."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class WatchlistItemResponse(BaseModel):
    id: int
    symbol: str
    stock_name: str
    buy_price: float
    buy_date: str
    quantity: float
    created_at: datetime


class WatchlistAddRequest(BaseModel):
    symbol: str
    buy_price: float = Field(..., gt=0)
    buy_date: str
    quantity: float = Field(default=1, gt=0)
