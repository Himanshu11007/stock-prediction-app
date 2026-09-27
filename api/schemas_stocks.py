"""Pydantic request/response models for /api/v1/stocks/* (normal-user stock
discovery - see api/schemas_admin.py:AdminStockResponse for the separate,
richer admin-only shape)."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class StockSearchResultResponse(BaseModel):
    symbol: str
    name: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    exchange: str
    analysis_enabled: bool
