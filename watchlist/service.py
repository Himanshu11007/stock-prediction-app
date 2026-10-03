"""Authenticated user watchlist business logic - user-scoped CRUD over the
new DB's watchlist_items table (db/models/tracker.py:WatchlistItem).

This is deliberately separate from the legacy storage/watchlist.py (global,
single-user, raw sqlite3 against tracker.db, still used by the running
Streamlit app) - the two are independent until the application is
repointed to the new DB (a later phase). Nothing here touches tracker.db.

Ownership is enforced entirely server-side: every function takes the
authenticated User (from the JWT-backed auth dependency) and scopes every
query/mutation to that user's own rows. No function here accepts a raw
user_id from a caller.
"""
from typing import Optional

from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from db.models.stock import Company
from db.models.tracker import WatchlistItem
from db.models.user import User


class UnknownSymbolError(Exception):
    """Raised when the symbol doesn't exist in the stock master (companies)."""


class DuplicateWatchlistItemError(Exception):
    """Raised when the user already has this symbol in their watchlist."""


def list_watchlist(session: Session, user: User) -> list[WatchlistItem]:
    stmt = select(WatchlistItem).where(WatchlistItem.user_id == user.id).order_by(WatchlistItem.id)
    return list(session.exec(stmt).all())


def add_watchlist_item(
    session: Session,
    user: User,
    symbol: str,
    buy_price: Optional[float] = None,
    buy_date: Optional[str] = None,
    quantity: float = 1,
) -> WatchlistItem:
    """company/stock_name is looked up from the stock master (companies),
    never trusted from the client - avoids ever pairing one stock's symbol
    with a different stock's name."""
    symbol = symbol.strip().upper()

    company = session.get(Company, symbol)
    if company is None or not company.active:
        # Same treatment for "doesn't exist" and "exists but inactive": the
        # stock-search endpoint (stocks/service.py) only ever surfaces
        # active stocks, so a client should never be able to add a symbol
        # here that it could not have discovered through search.
        raise UnknownSymbolError(f"Stock symbol {symbol!r} is not available")

    existing = session.exec(
        select(WatchlistItem).where(WatchlistItem.user_id == user.id, WatchlistItem.symbol == symbol)
    ).first()
    if existing is not None:
        raise DuplicateWatchlistItemError(f"{symbol} is already in your watchlist")

    item = WatchlistItem(
        user_id=user.id,
        symbol=symbol,
        stock_name=company.name,
        buy_price=buy_price,
        buy_date=buy_date,
        quantity=quantity,
    )
    session.add(item)
    try:
        session.commit()
    except IntegrityError:
        # Defensive backstop against the uq_watchlist_items_user_symbol
        # constraint for a rare concurrent-request race; the check above
        # handles the normal case.
        session.rollback()
        raise DuplicateWatchlistItemError(f"{symbol} is already in your watchlist")
    session.refresh(item)
    return item


def remove_watchlist_item(session: Session, user: User, item_id: int) -> bool:
    """Returns True if removed. False if the item doesn't exist OR belongs
    to a different user - deliberately not distinguished, so a request
    can't be used to probe whether another user has a given item id."""
    item = session.get(WatchlistItem, item_id)
    if item is None or item.user_id != user.id:
        return False
    from db.models.notifications import WatchlistAlertSetting
    for a in session.exec(select(WatchlistAlertSetting).where(WatchlistAlertSetting.user_id == user.id,
                                                              WatchlistAlertSetting.symbol == item.symbol)).all():
        session.delete(a)
    session.delete(item)
    session.commit()
    return True


def overview(session: Session, user: User) -> list[dict]:
    """Each watched stock with its latest analysis, change since the previous
    full ranking run, and alert settings. Missing data stays missing."""
    from db.models.market import MarketSnapshot, StockAnalysisResult
    from engine_runs.service import latest_result
    from notifications import detector
    from notifications.service import alert_payload, alert_setting, get_preferences
    from ranking import presenter

    items = list_watchlist(session, user)
    pref = get_preferences(session, user)
    cur_run = detector.latest_full_run(session)
    prev_run = detector.previous_full_run(session, cur_run) if cur_run else None

    def in_run(run, symbol):
        if run is None:
            return None
        return session.exec(select(StockAnalysisResult).where(StockAnalysisResult.run_id == run.run_id,
                                                              StockAnalysisResult.symbol == symbol)).first()

    out = []
    for it in items:
        latest = latest_result(session, it.symbol)
        cur, prev = in_run(cur_run, it.symbol), in_run(prev_run, it.symbol)
        snap = session.exec(select(MarketSnapshot).where(MarketSnapshot.symbol == it.symbol)
                            .order_by(MarketSnapshot.fetched_at.desc())).first()
        alerts = alert_setting(session, user.id, it.symbol)
        change = None
        if cur is not None and prev is not None and cur.stockai_score is not None and prev.stockai_score is not None:
            change = round(cur.stockai_score - prev.stockai_score, 1)
        fq = (latest.fqvf or {}) if latest else {}
        company = session.get(Company, it.symbol)
        out.append({
            "id": it.id, "symbol": it.symbol, "stock_name": it.stock_name,
            "sector": company.sector if company else None,
            "buy_price": it.buy_price, "buy_date": it.buy_date, "quantity": it.quantity,
            "created_at": it.created_at.isoformat(),
            "analysed": latest is not None,
            "stockai_score": latest.stockai_score if latest else None,
            "rank": cur.rank if cur else None,
            "eligible": cur.eligible if cur else None,
            "fqvf_score": fq.get("score"), "fqvf_passed": (fq.get("counts") or {}).get("PASS"),
            "score_change": change,
            "previous_score": prev.stockai_score if prev else None,
            "price": snap.close if snap else None, "price_as_of": snap.as_of_date if snap else None,
            "analysis_computed_at": presenter._iso(latest.computed_at) if latest else None,
            "freshness_status": presenter.freshness_status(latest) if latest else None,
            "labels": presenter.labels(latest, None) if latest else [],
            "alerts": alert_payload(alerts),
            "alerts_active": bool(pref.watchlist_alerts and not alerts.muted),
        })
    return out
