"""Stock master data: companies (symbol/name/sector master) and the
category-based stock universe (large/mid/small cap membership lists).

Replaces data/nse_stocks.csv (-> Company) and data/{large,mid,small}cap.csv
(-> StockUniverseMember) as the source of truth. See migration plan for the
CSV audit this schema is based on.
"""
from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel, UniqueConstraint


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Company(SQLModel, table=True):
    """Stock master. One row per NSE-listed symbol. Source of truth for
    stock search, sector lookups, and any table that needs a company FK."""

    __tablename__ = "companies"

    symbol: str = Field(primary_key=True)
    name: str
    sector: Optional[str] = Field(default=None, index=True)
    industry: Optional[str] = Field(default=None)
    isin: Optional[str] = Field(default=None)
    exchange: str = Field(default="NSE")
    active: bool = Field(default=True)
    analysis_enabled: bool = Field(default=True)
    # Admin-controlled: False for suspended/delisted/non-tradable symbols.
    tradable: bool = Field(default=True)
    # Data availability from the latest engine run: OK / PARTIAL / STALE /
    # UNAVAILABLE / ERROR (NULL = never checked), with a human-readable reason.
    data_status: Optional[str] = Field(default=None)
    data_status_reason: Optional[str] = Field(default=None)
    data_checked_at: Optional[datetime] = Field(default=None)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class StockUniverseMember(SQLModel, table=True):
    """Category membership (Large/Mid/Small Cap) for a symbol. A symbol can
    belong to at most one row per category; kept separate from Company since
    membership is a many-per-symbol relationship (a symbol could in principle
    appear in more than one list) while Company is one row per symbol."""

    __tablename__ = "stock_universe"
    __table_args__ = (
        UniqueConstraint("symbol", "category", name="uq_stock_universe_symbol_category"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(foreign_key="companies.symbol", index=True)
    category: str = Field(index=True)
    created_at: datetime = Field(default_factory=utcnow)
