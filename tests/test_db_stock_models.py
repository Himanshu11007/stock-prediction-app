"""tests/test_db_stock_models.py — schema-level tests for db/models/stock.py

Uses an isolated in-memory SQLite engine (not storage/app.db) so these tests
never touch real data and can run in any order/CI environment.
"""
import pytest
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, SQLModel, create_engine

from db.models.stock import Company, StockUniverseMember


@pytest.fixture()
def session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def test_company_roundtrip(session):
    session.add(Company(symbol="TCS.NS", name="Tata Consultancy Services Ltd."))
    session.commit()
    fetched = session.get(Company, "TCS.NS")
    assert fetched.name == "Tata Consultancy Services Ltd."
    assert fetched.active is True
    assert fetched.exchange == "NSE"


def test_stock_universe_requires_existing_company(session):
    session.add(Company(symbol="RELIANCE.NS", name="Reliance Industries Ltd."))
    session.commit()
    session.add(StockUniverseMember(symbol="RELIANCE.NS", category="Large Cap"))
    session.commit()

    member = session.exec(
        StockUniverseMember.__table__.select().where(
            StockUniverseMember.symbol == "RELIANCE.NS"
        )
    ).first()
    assert member is not None


def test_stock_universe_duplicate_symbol_category_rejected(session):
    session.add(Company(symbol="INFY.NS", name="Infosys Ltd."))
    session.commit()
    session.add(StockUniverseMember(symbol="INFY.NS", category="Large Cap"))
    session.commit()

    session.add(StockUniverseMember(symbol="INFY.NS", category="Large Cap"))
    with pytest.raises(IntegrityError):
        session.commit()
