"""Stock discovery for normal authenticated users - read-only, over the
same DB-backed stock master (db/models/stock.py:Company) that
scripts/migrate_stock_universe.py populated from the CSV universe.

Distinct from admin/service.py's list_stocks/get_stock: this only ever
returns active stocks and only the fields appropriate for a normal user
(no created_at/updated_at/active bookkeeping columns), and stays available
to any authenticated user rather than ADMIN-only.
"""
from typing import Optional

from sqlmodel import Session, func, select

from db.models.stock import Company


def search_stocks(
    session: Session, search: Optional[str] = None, limit: int = 50, offset: int = 0
) -> list[Company]:
    stmt = select(Company).where(Company.active == True)  # noqa: E712
    if search:
        pattern = f"%{search.strip().upper()}%"
        stmt = stmt.where(func.upper(Company.symbol).like(pattern) | func.upper(Company.name).like(pattern))
    stmt = stmt.order_by(Company.symbol).offset(offset).limit(limit)
    return list(session.exec(stmt).all())


def get_stock(session: Session, symbol: str) -> Optional[Company]:
    company = session.get(Company, symbol.strip().upper())
    if company is None or not company.active:
        return None
    return company
