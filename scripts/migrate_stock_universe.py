"""One-off data migration: CSV stock-universe files -> companies / stock_universe tables.

Source files (all read-only, none written by the app — see migration audit):
  data/nse_stocks.csv          -> companies (broad symbol/name master list)
  data/{large,mid,small}cap.csv -> companies (upsert) + stock_universe (membership)

Idempotent: safe to re-run. Existing rows are updated in place, not duplicated.
Does not touch or delete the source CSVs.

Usage:
    venv/Scripts/python.exe -m scripts.migrate_stock_universe
"""
import sys
from pathlib import Path

import pandas as pd
from sqlmodel import Session, select

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from db.models.stock import Company, StockUniverseMember
from db.session import engine


def _clean(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    bad = df[df["Symbol"].isna() | df["Company"].isna()]
    if len(bad):
        print(f"  skipping {len(bad)} malformed row(s) in {source}: {bad['Symbol'].tolist()}")
        df = df.drop(bad.index)
    df["Symbol"] = df["Symbol"].astype(str).str.strip()
    df["Company"] = df["Company"].astype(str).str.strip()
    return df[(df["Symbol"] != "") & (df["Company"] != "")]


def _upsert_company(session: Session, symbol: str, name: str) -> None:
    existing = session.get(Company, symbol)
    if existing is None:
        session.add(Company(symbol=symbol, name=name))
    elif existing.name != name:
        existing.name = name
        session.add(existing)


def _upsert_universe_membership(session: Session, symbol: str, category: str) -> None:
    stmt = select(StockUniverseMember).where(
        StockUniverseMember.symbol == symbol, StockUniverseMember.category == category
    )
    if session.exec(stmt).first() is None:
        session.add(StockUniverseMember(symbol=symbol, category=category))


def migrate() -> None:
    with Session(engine) as session:
        # 1. Broad master list -> companies
        master_df = _clean(pd.read_csv(config.DATA_DIR / "nse_stocks.csv"), "nse_stocks.csv")
        for row in master_df.itertuples(index=False):
            _upsert_company(session, row.Symbol, row.Company)
        session.commit()
        print(f"companies: upserted {len(master_df)} rows from nse_stocks.csv")

        # 2. Category CSVs -> companies (upsert, in case of symbols missing from
        #    nse_stocks.csv) + stock_universe membership
        category_files = {
            "Large Cap": config.UNIVERSE_LARGECAP,
            "Mid Cap": config.UNIVERSE_MIDCAP,
            "Small Cap": config.UNIVERSE_SMALLCAP,
        }
        for category, path in category_files.items():
            df = _clean(pd.read_csv(path), path.name)
            for row in df.itertuples(index=False):
                _upsert_company(session, row.Symbol, row.Company)
                _upsert_universe_membership(session, row.Symbol, category)
            session.commit()
            print(f"stock_universe: upserted {len(df)} '{category}' rows from {path.name}")

        total_companies = session.exec(select(Company)).all()
        total_universe = session.exec(select(StockUniverseMember)).all()
        print(f"Done. companies={len(total_companies)} stock_universe={len(total_universe)}")


if __name__ == "__main__":
    migrate()
