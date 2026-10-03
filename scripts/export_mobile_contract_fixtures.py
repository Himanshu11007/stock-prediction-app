"""
scripts/export_mobile_contract_fixtures.py — capture real API responses for
the mobile app's contract tests.

Runs the real FastAPI application against an in-memory database seeded with
two ranking runs, a watchlist entry, notifications and preferences, and
writes each response body to
  <mobile repo>/StockAIPro.Mobile.Tests/Fixtures/<name>.json
so StockAIPro.Mobile.Tests deserialises exactly what the backend sends.

Usage: python scripts/export_mobile_contract_fixtures.py [--mobile-repo PATH]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402
from sqlmodel import Session, SQLModel, create_engine  # noqa: E402

import auth.service as auth_service  # noqa: E402
import notifications.service as notify  # noqa: E402
from api.main import app  # noqa: E402
from auth.security import create_access_token  # noqa: E402
from db.models.market import MarketSnapshot  # noqa: E402
from db.models.stock import Company  # noqa: E402
from db.models.tracker import WatchlistItem  # noqa: E402
from db.session import get_session  # noqa: E402
from notifications.push import PushResult, PushRouter, SENT  # noqa: E402
from tests.test_notifications import _base_scores, _run  # noqa: E402

DEFAULT_MOBILE = ROOT.parent / "StockAIPro-Mobile" / "StockAIPro.Mobile"


class _Ok:
    def configured(self):
        return True

    def send(self, token, message):
        return PushResult(SENT, "m")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mobile-repo", default=str(DEFAULT_MOBILE))
    out = Path(ap.parse_args().mobile_repo) / "StockAIPro.Mobile.Tests" / "Fixtures"
    out.mkdir(parents=True, exist_ok=True)

    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(eng)
    now = dt.datetime.now(dt.timezone.utc)
    with Session(eng) as s:
        auth_service.ensure_roles_exist(s)
        user = auth_service.create_user(s, "user@example.com", "UserPass1!", roles=[auth_service.USER_ROLE])
        for i in range(30):
            s.add(Company(symbol=f"S{i:02d}.NS", name=f"Stock {i:02d}", sector="Energy" if i % 2 else "IT",
                          industry="Oil & Gas" if i % 2 else "Software"))
        s.commit()
        prev = _base_scores()
        _run(s, "R1", now - dt.timedelta(days=1, hours=2), prev, risk={"S03.NS": 20.0})
        _run(s, "R2", now - dt.timedelta(hours=2), {**prev, "S10.NS": 99.0, "S03.NS": 75.0},
             regime=("Sideways", 0.1), risk={"S03.NS": 20.0})
        s.add(MarketSnapshot(symbol="S03.NS", fetched_at=now - dt.timedelta(hours=3),
                             as_of_date=(now - dt.timedelta(hours=3)).date().isoformat(), close=1234.5,
                             technical={"return_60d": 0.05, "trend_daily": {"trend": "UP", "score": 0.5},
                                        "trend_weekly": {"trend": "SIDEWAYS", "score": 0.0}}))
        s.add(WatchlistItem(user_id=user.id, symbol="S03.NS", stock_name="Stock 03"))
        s.commit()
        notify.update_preferences(s, user, {"score_changes": True})
        notify.register_device(s, user, "fixture-device", "android", "fixture-token", "1.0", "granted")
        notify.process_ranking_run(s, "R2", now=now, router=PushRouter({"fcm": _Ok(), "apns": _Ok()}))

    def _session():
        with Session(eng) as s:
            yield s
    app.dependency_overrides[get_session] = _session
    client = TestClient(app)
    h = {"Authorization": f"Bearer {create_access_token(subject='user@example.com', roles=['USER'])}"}
    calls = {
        "app_config": client.get("/api/v1/app/config"),
        "market_status": client.get("/api/v1/market/status", headers=h),
        "top_picks": client.get("/api/v1/top-picks", headers=h),
        "stock_analysis": client.get("/api/v1/stocks/S03.NS/analysis", headers=h),
        "watchlist_overview": client.get("/api/v1/watchlist/overview", headers=h),
        "notifications": client.get("/api/v1/notifications", headers=h),
        "notification_preferences": client.get("/api/v1/notifications/preferences", headers=h),
        "devices": client.get("/api/v1/devices", headers=h),
        "performance_overview": client.get("/api/v1/performance/overview", headers=h),
        "intelligence_overview": client.get("/api/v1/intelligence/overview", headers=h),
        "feedback_created": client.post("/api/v1/feedback", headers=h,
                                        json={"category": "STALE_DATA", "message": "Price looks old", "symbol": "S03.NS"}),
    }
    app.dependency_overrides.clear()
    for name, r in calls.items():
        if r.status_code >= 400:
            raise SystemExit(f"{name}: HTTP {r.status_code} {r.text[:200]}")
        (out / f"{name}.json").write_text(json.dumps(r.json(), indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {len(calls)} fixtures to {out}")


if __name__ == "__main__":
    main()
