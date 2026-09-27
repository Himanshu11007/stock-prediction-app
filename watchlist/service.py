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
    buy_price: float,
    buy_date: str,
    quantity: float = 1,
) -> WatchlistItem:
    """company/stock_name is looked up from the stock master (companies),
    never trusted from the client - avoids ever pairing one stock's symbol
    with a different stock's name."""
    symbol = symbol.strip().upper()

    company = session.get(Company, symbol)
    if company is None:
        raise UnknownSymbolError(f"Unknown stock symbol: {symbol!r}")

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
    session.delete(item)
    session.commit()
    return True
