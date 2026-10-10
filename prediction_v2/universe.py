"""
prediction_v2/universe.py — v2 universe membership (independent of v1).

v1 ranks only `stock_universe` members (seeded from data/{large,mid,small}cap.csv);
a stock created in the Admin Console is a Company but not a member, so it is
never ranked in a full v1 run (finding F1). v2 keeps its own list:

  * seed: a COPY of the v1 members (v1 is read, never written);
  * administrators add/remove any existing active Company (e.g. a refiner
    missing from v1) without affecting v1;
  * bulk import of a constituent list (e.g. NIFTY 500, F&O) from a CSV the
    operator supplies - no list is bundled because no licensed source is
    configured.
"""
from __future__ import annotations

from typing import Iterable, Optional

from sqlmodel import Session, select

from db.models.prediction import V2UniverseMember
from db.models.stock import Company, StockUniverseMember


def seed_from_v1(session: Session) -> int:
    """Add every v1 member that is not already in v2. Returns rows added."""
    have = set(session.exec(select(V2UniverseMember.symbol)).all())
    added = 0
    for sym in sorted(set(session.exec(select(StockUniverseMember.symbol)).all()) - have):
        session.add(V2UniverseMember(symbol=sym, source="SEED"))
        added += 1
    session.commit()
    return added


def add_symbols(session: Session, symbols: Iterable[str], *, source: str = "ADMIN",
                added_by: Optional[int] = None, note: Optional[str] = None) -> dict[str, list[str]]:
    """Add existing Companies to v2 (re-activates inactive members)."""
    out: dict[str, list[str]] = {"added": [], "reactivated": [], "already": [], "unknown": []}
    for raw in symbols:
        sym = raw.strip().upper()
        if not sym:
            continue
        company = session.get(Company, sym)
        if company is None:
            out["unknown"].append(sym)
            continue
        row = session.exec(select(V2UniverseMember).where(V2UniverseMember.symbol == sym)).first()
        if row is None:
            session.add(V2UniverseMember(symbol=sym, source=source, added_by=added_by, note=note))
            out["added"].append(sym)
        elif not row.active:
            row.active, row.note = True, note or row.note
            session.add(row)
            out["reactivated"].append(sym)
        else:
            out["already"].append(sym)
    session.commit()
    return out


def deactivate(session: Session, symbols: Iterable[str]) -> list[str]:
    done = []
    for sym in (s.strip().upper() for s in symbols):
        row = session.exec(select(V2UniverseMember).where(V2UniverseMember.symbol == sym)).first()
        if row and row.active:
            row.active = False
            session.add(row)
            done.append(sym)
    session.commit()
    return done


def members(session: Session) -> list[Company]:
    """Active v2 members whose Company is active and tradable."""
    stmt = (select(Company).join(V2UniverseMember, V2UniverseMember.symbol == Company.symbol)
            .where(V2UniverseMember.active == True, Company.active == True, Company.tradable == True)  # noqa: E712
            .order_by(Company.symbol))
    return list(session.exec(stmt).all())
