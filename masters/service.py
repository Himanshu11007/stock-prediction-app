"""
masters/service.py — Sector / Industry masters and application configuration.

Mutations follow admin/service.py's rule: the entity change and its
admin_audit_logs row are committed in one transaction (admin.audit.log_action).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlmodel import Session, func, select

from admin.audit import log_action
from db.models.market import SECTOR_OUTLOOKS, AppConfig, Industry, Sector
from db.models.stock import Company
from db.models.user import User
from ranking.service import DEFAULT_RULES, DEFAULT_WEIGHTS, validate_rules, validate_weights


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── Application configuration ────────────────────────────────────────────────

# key -> (default, description). Only these keys can be set.
CONFIG_KEYS: dict[str, tuple[Any, str]] = {
    "ranking.weights": (DEFAULT_WEIGHTS, "StockAI Score component weights (docs/RANKING_METHODOLOGY.md)"),
    "ranking.rules": (DEFAULT_RULES, "Top Picks eligibility rules"),
    "top_picks.limit": (20, "Maximum number of Top Investment Candidates returned to clients"),
    "app.features": ({"top_picks": True, "stock_analysis": True, "watchlist": True,
                      "performance": True, "intelligence": True, "ml_signal_display": True},
                     "Feature availability flags read by the mobile app"),
    "app.disclaimer": ("StockAI Pro provides research and analysis, not investment advice. "
                       "Scores rank stocks on available data; they are not predictions or "
                       "guarantees of returns. Past performance does not indicate future results.",
                       "Disclaimer shown by clients"),
    "app.announcement": (None, "Optional message shown on the mobile home screen (null = none)"),
}


def _validate_config(key: str, value: Any) -> Any:
    if key not in CONFIG_KEYS:
        raise ValueError(f"Unknown configuration key: {key}")
    if key == "ranking.weights":
        if not isinstance(value, dict):
            raise ValueError("ranking.weights must be an object")
        return validate_weights(value)
    if key == "ranking.rules":
        if not isinstance(value, dict):
            raise ValueError("ranking.rules must be an object")
        return validate_rules(value)
    if key == "top_picks.limit":
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 100:
            raise ValueError("top_picks.limit must be an integer between 1 and 100")
    if key == "app.features":
        default = CONFIG_KEYS[key][0]
        if not isinstance(value, dict) or set(value) - set(default) or \
                not all(isinstance(v, bool) for v in value.values()):
            raise ValueError(f"app.features must map known features {sorted(default)} to booleans")
        return {**default, **value}
    if key in ("app.disclaimer", "app.announcement"):
        if value is not None and (not isinstance(value, str) or len(value) > 2000):
            raise ValueError(f"{key} must be a string of at most 2000 characters or null")
    return value


def get_config(session: Session, key: str) -> Any:
    if key not in CONFIG_KEYS:
        raise KeyError(key)
    row = session.get(AppConfig, key)
    return row.value if row is not None else CONFIG_KEYS[key][0]


def list_config(session: Session) -> list[dict]:
    rows = {r.key: r for r in session.exec(select(AppConfig)).all()}
    out = []
    for key, (default, desc) in CONFIG_KEYS.items():
        row = rows.get(key)
        out.append({"key": key, "value": row.value if row else default, "description": desc,
                    "is_default": row is None, "updated_at": row.updated_at if row else None,
                    "updated_by": row.updated_by if row else None})
    return out


def set_config(session: Session, admin_user: User, key: str, value: Any) -> dict:
    value = _validate_config(key, value)
    row = session.get(AppConfig, key)
    old = row.value if row else CONFIG_KEYS[key][0]
    if row is None:
        row = AppConfig(key=key, description=CONFIG_KEYS[key][1])
    row.value, row.updated_at, row.updated_by = value, _now(), admin_user.id
    session.add(row)
    log_action(session, admin_user, "CONFIG_UPDATED", "app_config", key, {"old": old, "new": value})
    session.commit()
    return next(c for c in list_config(session) if c["key"] == key)


def ranking_config(session: Session) -> tuple[dict, dict]:
    return (validate_weights(get_config(session, "ranking.weights")),
            validate_rules(get_config(session, "ranking.rules")))


# ── Sectors and industries ───────────────────────────────────────────────────

def upsert_classification(session: Session, sector: Optional[str], industry: Optional[str]) -> None:
    """Create Sector/Industry master rows for a provider classification
    (no commit; called inside an engine-run transaction)."""
    sector_row = None
    if sector:
        sector_row = session.exec(select(Sector).where(Sector.name == sector)).first()
        if sector_row is None:
            sector_row = Sector(name=sector)
            session.add(sector_row)
            session.flush()
    if industry:
        ind = session.exec(select(Industry).where(Industry.name == industry)).first()
        if ind is None:
            session.add(Industry(name=industry, sector_id=sector_row.id if sector_row else None))
            session.flush()
        elif ind.sector_id is None and sector_row is not None:
            ind.sector_id = sector_row.id
            session.add(ind)


def list_sectors(session: Session) -> list[dict]:
    counts = dict(session.exec(select(Company.sector, func.count()).group_by(Company.sector)).all())
    return [{**s.model_dump(), "stock_count": counts.get(s.name, 0)}
            for s in session.exec(select(Sector).order_by(Sector.name)).all()]


def update_sector(session: Session, admin_user: User, sector_id: int, *, outlook: Optional[str] = None,
                  outlook_notes: Optional[str] = None, active: Optional[bool] = None,
                  clear_outlook: bool = False) -> Sector:
    sector = session.get(Sector, sector_id)
    if sector is None:
        raise KeyError(f"Sector {sector_id}")
    changes: dict[str, Any] = {}
    if clear_outlook:
        changes["outlook"] = [sector.outlook, None]
        sector.outlook, sector.outlook_notes = None, None
        sector.outlook_updated_at, sector.outlook_updated_by = _now(), admin_user.id
    elif outlook is not None:
        outlook = outlook.strip().upper()
        if outlook not in SECTOR_OUTLOOKS:
            raise ValueError(f"outlook must be one of {SECTOR_OUTLOOKS}")
        changes["outlook"] = [sector.outlook, outlook]
        sector.outlook = outlook
        sector.outlook_notes = outlook_notes
        sector.outlook_updated_at, sector.outlook_updated_by = _now(), admin_user.id
    if active is not None and active != sector.active:
        changes["active"] = [sector.active, active]
        sector.active = active
    if not changes:
        return sector
    sector.updated_at = _now()
    session.add(sector)
    log_action(session, admin_user, "SECTOR_UPDATED", "sector", sector.name, changes)
    session.commit()
    session.refresh(sector)
    return sector


def list_industries(session: Session) -> list[dict]:
    sectors = {s.id: s.name for s in session.exec(select(Sector)).all()}
    counts = dict(session.exec(select(Company.industry, func.count()).group_by(Company.industry)).all())
    return [{**i.model_dump(), "sector": sectors.get(i.sector_id), "stock_count": counts.get(i.name, 0)}
            for i in session.exec(select(Industry).order_by(Industry.name)).all()]


def update_industry(session: Session, admin_user: User, industry_id: int, *,
                    active: Optional[bool] = None, sector_id: Optional[int] = None) -> Industry:
    ind = session.get(Industry, industry_id)
    if ind is None:
        raise KeyError(f"Industry {industry_id}")
    changes: dict[str, Any] = {}
    if sector_id is not None and sector_id != ind.sector_id:
        if session.get(Sector, sector_id) is None:
            raise ValueError(f"Sector {sector_id} does not exist")
        changes["sector_id"] = [ind.sector_id, sector_id]
        ind.sector_id = sector_id
    if active is not None and active != ind.active:
        changes["active"] = [ind.active, active]
        ind.active = active
    if not changes:
        return ind
    ind.updated_at = _now()
    session.add(ind)
    log_action(session, admin_user, "INDUSTRY_UPDATED", "industry", ind.name, changes)
    session.commit()
    session.refresh(ind)
    return ind
