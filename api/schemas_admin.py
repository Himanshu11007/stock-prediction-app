"""Pydantic request/response models for /api/v1/admin/*."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel


# ── Dashboard ────────────────────────────────────────────────────────────────

class AdminDashboardResponse(BaseModel):
    total_users: int
    active_users: int
    admin_users: int
    total_stocks: int
    enabled_stocks: int
    analysis_enabled_stocks: int
    total_recommendations: int
    total_watchlist_items: int


# ── Users ────────────────────────────────────────────────────────────────────

class AdminUserResponse(BaseModel):
    id: int
    email: str
    is_active: bool
    roles: list[str]
    created_at: datetime


class AssignRoleRequest(BaseModel):
    role: str


# ── Stock master ─────────────────────────────────────────────────────────────

class AdminStockResponse(BaseModel):
    symbol: str
    name: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    exchange: str
    active: bool
    analysis_enabled: bool
    created_at: datetime
    updated_at: datetime


class AdminStockUpdateRequest(BaseModel):
    active: Optional[bool] = None
    analysis_enabled: Optional[bool] = None


# ── Recommendations (read-only) ─────────────────────────────────────────────

class AdminRecommendationResponse(BaseModel):
    id: int
    saved_date: str
    symbol: str
    stock: str
    signal: str
    cmp: float
    confluence_score: Optional[float] = None
    ml_confidence: Optional[float] = None
    news_score: Optional[float] = None
    scan_id: Optional[str] = None
    sector: Optional[str] = None
    market_regime: Optional[str] = None
    engine_version: Optional[str] = None
    is_legacy_migration: bool
    created_at: datetime


# ── Watchlist (read-only) ───────────────────────────────────────────────────

class AdminWatchlistItemResponse(BaseModel):
    id: int
    user_id: int
    symbol: str
    stock_name: str
    buy_price: float
    buy_date: str
    quantity: float
    is_legacy_migration: bool
    created_at: datetime


# ── Audit logs (read-only) ──────────────────────────────────────────────────

class AdminAuditLogResponse(BaseModel):
    id: int
    admin_user_id: int
    action: str
    entity: str
    entity_id: Optional[str] = None
    extra_data: Optional[dict[str, Any]] = None
    created_at: datetime
