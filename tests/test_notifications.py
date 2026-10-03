"""
tests/test_notifications.py — notification engine (detection, fan-out,
preferences, deduplication, rate limiting, quiet hours, push delivery,
devices), the user/admin notification API, watchlist alerts, feedback,
market status, performance/intelligence overviews and the scheduler's
calendar rules.
"""
import datetime as dt

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

import auth.service as auth_service
import masters.service as masters
import notifications.service as notify
from api.main import app
from auth.security import create_access_token
from db.models.admin import AdminAuditLog
from db.models.market import EngineRun, MarketRegimeSnapshot, MarketSnapshot, StockAnalysisResult
from db.models.notifications import (Notification, NotificationDelivery, NotificationPreference, NotificationRun,
                                     PushDevice, WatchlistAlertSetting)
from db.models.stock import Company
from db.models.tracker import WatchlistItem
from db.session import get_session
from notifications.push import (FAILED, INVALID_TOKEN, NOT_CONFIGURED, SENT, ApnsProvider, FcmProvider, PushMessage,
                                PushResult, PushRouter)
from notifications.settings import validate_templates
from utils.market_session import IST, market_status

UTC = dt.timezone.utc
NOON_IST = dt.datetime(2026, 10, 5, 12, 0, tzinfo=IST)          # Monday, outside quiet hours


class FakeProvider:
    def __init__(self, name, result=SENT):
        self.name, self.result, self.sent = name, result, []

    def configured(self):
        return True

    def send(self, token, message):
        self.sent.append((token, message))
        return PushResult(self.result, message_id="m1" if self.result == SENT else None)


@pytest.fixture()
def router():
    return PushRouter({"fcm": FakeProvider("fcm"), "apns": FakeProvider("apns")})


@pytest.fixture()
def db():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(eng)
    with Session(eng) as s:
        auth_service.ensure_roles_exist(s)
        auth_service.create_user(s, "admin@example.com", "AdminPass1!", roles=[auth_service.ADMIN_ROLE])
        auth_service.create_user(s, "user@example.com", "UserPass1!", roles=[auth_service.USER_ROLE])
        auth_service.create_user(s, "other@example.com", "OtherPass1!", roles=[auth_service.USER_ROLE])
        for i in range(30):
            s.add(Company(symbol=f"S{i:02d}.NS", name=f"Stock {i:02d}", sector="Energy" if i % 2 else "IT"))
        s.commit()
    return eng


def _uid(s, email):
    return auth_service.get_user_by_email(s, email).id


def _fq(passes):
    return {"score": passes / 18 * 100, "coverage": 0.9, "counts": {"PASS": passes, "FAIL": 18 - passes},
            "checks": [], "summary": f"{passes} passed"}


def _comps(score, risk=50.0):
    keys = ("quality", "valuation", "financial_health", "technical_trend", "momentum", "risk", "sector_outlook",
            "market_regime", "ml_signal")
    return {k: {"score": (risk if k == "risk" else score), "weight": 0 if k == "ml_signal" else 10,
                "label": k, "basis": "test"} for k in keys}


def _run(s, run_id, started, scores, regime=("Bullish", 0.6), kind="RANKING", config=None, passes=None,
         ineligible=(), risk=None):
    """scores: symbol -> StockLens Score; rank by score among eligible."""
    s.add(EngineRun(run_id=run_id, kind=kind, status="COMPLETED", started_at=started, finished_at=started,
                    engine_version="ranking-v1.0", fqvf_version="fqvf-v1.0", config=config or {}))
    ranked = sorted((sym for sym in scores if sym not in ineligible), key=lambda x: -scores[x])
    for sym, sc in scores.items():
        s.add(StockAnalysisResult(
            run_id=run_id, symbol=sym, computed_at=started, stockai_score=sc, score_coverage=0.95,
            eligible=sym not in ineligible, ineligible_reasons=["market data stale"] if sym in ineligible else [],
            rank=ranked.index(sym) + 1 if sym in ranked else None,
            components=_comps(sc, (risk or {}).get(sym, 50.0)), fqvf=_fq((passes or {}).get(sym, 12)),
            freshness={"market_data_as_of": started.date().isoformat()}, engine_version="ranking-v1.0",
            fqvf_version="fqvf-v1.0"))
    s.add(MarketRegimeSnapshot(run_id=run_id, regime=regime[0], regime_score=regime[1], computed_at=started,
                               as_of_date=started.date().isoformat()))
    s.commit()


def _base_scores():
    return {f"S{i:02d}.NS": 90.0 - i for i in range(30)}       # S00 best ... S29 worst


def _two_runs(s, **kw):
    prev, cur = _base_scores(), _base_scores()
    for k, v in kw.pop("changes", {}).items():
        cur[k] = v
    t0 = dt.datetime(2026, 10, 2, 11, 0, tzinfo=UTC)
    _run(s, "R1", t0, prev, regime=kw.pop("prev_regime", ("Bullish", 0.6)), passes=kw.pop("prev_passes", None),
         risk=kw.pop("prev_risk", None))
    _run(s, "R2", t0 + dt.timedelta(days=1), cur, **kw)


def _set(s, key, value):
    admin = auth_service.get_user_by_email(s, "admin@example.com")
    masters.set_config(s, admin, key, value)


def _limit(s, n):
    _set(s, "top_picks.limit", n)


# ── detection and fan-out ────────────────────────────────────────────────────

def test_new_and_removed_top_candidates_are_detected_with_reasons(db, router):
    with Session(db) as s:
        _limit(s, 5)
        _two_runs(s, changes={"S10.NS": 99.0, "S01.NS": 70.0}, ineligible=("S02.NS",))
        uid = _uid(s, "user@example.com")
        notify.update_preferences(s, s.get(type(auth_service.get_user_by_email(s, "user@example.com")), uid),
                                  {"top_candidate_removed": True})
        nrun = notify.process_ranking_run(s, "R2", now=NOON_IST, router=router)
        assert nrun.status == "COMPLETED"
        assert set(nrun.detail["entered_top"]) == {"S10.NS", "S05.NS"}
        assert set(nrun.detail["left_top"]) == {"S01.NS", "S02.NS"}
        rows = s.exec(select(Notification).where(Notification.user_id == uid)).all()
        new = {n.symbol: n for n in rows if n.type == "NEW_TOP_CANDIDATE"}
        assert set(new) == {"S10.NS", "S05.NS"}
        assert "Stock 10 has entered the Top Investment Candidates at rank 1" in new["S10.NS"].body
        assert "StockLens Score: 99.0/100" in new["S10.NS"].body
        assert new["S10.NS"].route == "/stock/S10.NS" and new["S10.NS"].ranking_run_id == "R2"
        removed = {n.symbol: n for n in rows if n.type == "TOP_CANDIDATE_REMOVED"}
        assert "StockLens Score decreased from 89.0 to 70.0" in removed["S01.NS"].body
        assert "data quality / eligibility changed (market data stale)" in removed["S02.NS"].body
        # no "buy" language anywhere
        assert not any("buy" in (n.title + n.body).lower() for n in rows)


def test_conservative_defaults_do_not_notify_score_or_fqvf_changes(db, router):
    with Session(db) as s:
        _limit(s, 5)
        _two_runs(s, changes={"S03.NS": 70.0}, passes={"S03.NS": 16})
        notify.process_ranking_run(s, "R2", now=NOON_IST, router=router)
        uid = _uid(s, "user@example.com")
        types = {n.type for n in s.exec(select(Notification).where(Notification.user_id == uid)).all()}
        assert "SCORE_CHANGE" not in types and "FQVF_CHANGE" not in types and "TOP_CANDIDATE_REMOVED" not in types
        pref = notify.get_preferences(s, auth_service.get_user_by_email(s, "user@example.com"))
        assert pref.new_top_candidate and pref.watchlist_alerts and not pref.daily_summary and not pref.market_regime


def test_score_change_threshold_is_configurable_and_small_changes_are_ignored(db, router):
    with Session(db) as s:
        _limit(s, 5)
        user = auth_service.get_user_by_email(s, "user@example.com")
        notify.update_preferences(s, user, {"score_changes": True, "new_top_candidate": False})
        _two_runs(s, changes={"S00.NS": 85.0, "S01.NS": 95.0})        # -5 and +6 points
        notify.process_ranking_run(s, "R2", now=NOON_IST, router=router)
        assert not s.exec(select(Notification).where(Notification.type == "SCORE_CHANGE")).all()
    with Session(db) as s:
        s.exec(select(NotificationRun)).all()
        for n in s.exec(select(NotificationRun)).all():
            s.delete(n)
        s.commit()
        _set(s, "notifications.settings", {"score_change_threshold": 5})
        notify.process_ranking_run(s, "R2", now=NOON_IST, router=router)
        changed = {n.symbol: n for n in s.exec(select(Notification).where(Notification.type == "SCORE_CHANGE")).all()}
        assert set(changed) == {"S00.NS", "S01.NS"}
        assert "from 90.0 to 85.0 (-5.0)" in changed["S00.NS"].body


def test_fqvf_change_and_market_regime_change(db, router):
    with Session(db) as s:
        _limit(s, 5)
        user = auth_service.get_user_by_email(s, "user@example.com")
        notify.update_preferences(s, user, {"fqvf_changes": True, "market_regime": True, "new_top_candidate": False})
        _two_runs(s, prev_passes={"S00.NS": 15}, passes={"S00.NS": 17, "S01.NS": 13},
                  prev_regime=("Bullish", 0.6), regime=("Bearish", -0.5))
        notify.process_ranking_run(s, "R2", now=NOON_IST, router=router)
        rows = s.exec(select(Notification).where(Notification.user_id == user.id)).all()
        fq = [n for n in rows if n.type == "FQVF_CHANGE"]
        assert [n.symbol for n in fq] == ["S00.NS"]                       # 12->13 is below 2 checks
        assert "changed from 15/18 to 17/18" in fq[0].body
        regime = [n for n in rows if n.type == "MARKET_REGIME_CHANGE"]
        assert len(regime) == 1 and "from Bullish to Bearish" in regime[0].body and regime[0].route == "/"


def test_insignificant_regime_fluctuation_is_not_notified(db, router):
    with Session(db) as s:
        user = auth_service.get_user_by_email(s, "user@example.com")
        notify.update_preferences(s, user, {"market_regime": True})
        _two_runs(s, prev_regime=("Sideways", 0.05), regime=("Bullish", 0.2))     # delta 0.15 < 0.3
        notify.process_ranking_run(s, "R2", now=NOON_IST, router=router)
        assert not s.exec(select(Notification).where(Notification.type == "MARKET_REGIME_CHANGE")).all()


def test_many_new_candidates_are_aggregated_into_one_notification(db, router):
    with Session(db) as s:
        _limit(s, 10)
        _two_runs(s, changes={f"S{i}.NS": 100.0 + i for i in range(20, 26)})    # 6 new entrants
        notify.process_ranking_run(s, "R2", now=NOON_IST, router=router)
        uid = _uid(s, "user@example.com")
        rows = s.exec(select(Notification).where(Notification.user_id == uid,
                                                 Notification.type == "NEW_TOP_CANDIDATE")).all()
        assert len(rows) == 1 and rows[0].route == "/top-picks"
        assert rows[0].title == "New StockLens Top Candidates" and "6 stocks entered" in rows[0].body


def test_watchlist_alert_combines_changes_and_respects_per_stock_switches(db, router):
    with Session(db) as s:
        _limit(s, 5)
        user = auth_service.get_user_by_email(s, "user@example.com")
        other = auth_service.get_user_by_email(s, "other@example.com")
        s.add(WatchlistItem(user_id=user.id, symbol="S20.NS", stock_name="Stock 20"))
        s.add(WatchlistItem(user_id=user.id, symbol="S21.NS", stock_name="Stock 21"))
        s.add(WatchlistItem(user_id=other.id, symbol="S20.NS", stock_name="Stock 20"))
        s.commit()
        notify.update_alert_setting(s, user, "S21.NS", {"muted": True})
        notify.update_alert_setting(s, other, "S20.NS", {"score_changes": False, "rank_changes": False})
        _two_runs(s, changes={"S20.NS": 95.0, "S21.NS": 86.0}, prev_passes={"S20.NS": 10}, passes={"S20.NS": 13})
        notify.process_ranking_run(s, "R2", now=NOON_IST, router=router)
        mine = s.exec(select(Notification).where(Notification.user_id == user.id,
                                                 Notification.type == "WATCHLIST_ALERT")).all()
        assert [n.symbol for n in mine] == ["S20.NS"]                      # S21 muted
        body = mine[0].body
        assert "StockLens Score 70.0 -> 95.0 (+25.0)" in body and "rank 21 -> 1" in body
        assert "entered the Top Investment Candidates" in body
        assert "FQVF 10/18 -> 13/18" in body and mine[0].title == "Watchlist: Stock 20"
        # The watched stock is not notified a second time as a new candidate.
        assert not s.exec(select(Notification).where(Notification.user_id == user.id,
                                                     Notification.symbol == "S20.NS",
                                                     Notification.type == "NEW_TOP_CANDIDATE")).all()
        theirs = s.exec(select(Notification).where(Notification.user_id == other.id,
                                                   Notification.type == "WATCHLIST_ALERT")).one()
        assert "StockLens Score" not in theirs.body and "FQVF 10/18 -> 13/18" in theirs.body


# ── dedup, rate limits, cooldown, quiet hours, global switch ─────────────────

def test_processing_the_same_run_twice_never_duplicates(db, router):
    with Session(db) as s:
        _limit(s, 5)
        _two_runs(s, changes={"S10.NS": 99.0})
        first = notify.process_ranking_run(s, "R2", now=NOON_IST, router=router)
        count = len(s.exec(select(Notification)).all())
        again = notify.process_ranking_run(s, "R2", now=NOON_IST, router=router)
        assert again.id == first.id and len(s.exec(select(Notification)).all()) == count
        # even if the run record were removed, (user, dedup_key) blocks duplicates
        s.delete(s.get(NotificationRun, first.id))
        for n in s.exec(select(Notification)).all():
            n.notification_run_id = None
            s.add(n)
        s.commit()
        notify.process_ranking_run(s, "R2", now=NOON_IST, router=router)
        assert len(s.exec(select(Notification)).all()) == count


def test_push_rate_limit_keeps_extra_notifications_in_the_inbox(db, router):
    with Session(db) as s:
        _limit(s, 10)
        _set(s, "notifications.settings", {"max_push_per_user_per_run": 2, "aggregate_above": 10})
        user = auth_service.get_user_by_email(s, "user@example.com")
        notify.register_device(s, user, "dev-1", "android", "tok-user")
        _two_runs(s, changes={f"S{i}.NS": 100.0 + i for i in range(20, 25)})    # 5 entrants
        notify.process_ranking_run(s, "R2", now=NOON_IST, router=router)
        rows = s.exec(select(Notification).where(Notification.user_id == user.id)).all()
        assert len(rows) == 5
        assert sorted(n.push_status for n in rows).count("SENT") == 2
        assert sum(n.push_status == "SUPPRESSED_RATE_LIMIT" for n in rows) == 3
        assert len(router.providers["fcm"].sent) == 2


def test_per_stock_cooldown_blocks_repeated_pushes(db, router):
    with Session(db) as s:
        _limit(s, 5)
        user = auth_service.get_user_by_email(s, "user@example.com")
        notify.register_device(s, user, "dev-1", "android", "tok-user")
        _two_runs(s, changes={"S10.NS": 99.0})
        notify.process_ranking_run(s, "R2", now=NOON_IST, router=router)
        t = dt.datetime(2026, 10, 4, 11, 0, tzinfo=UTC)
        _run(s, "R3", t, _base_scores())                                    # S10 drops out
        _run(s, "R4", t + dt.timedelta(hours=1), {**_base_scores(), "S10.NS": 99.0})   # and re-enters
        notify.process_ranking_run(s, "R3", now=NOON_IST, router=router)
        notify.process_ranking_run(s, "R4", now=NOON_IST + dt.timedelta(hours=1), router=router)
        s10 = s.exec(select(Notification).where(Notification.user_id == user.id, Notification.symbol == "S10.NS",
                                                Notification.type == "NEW_TOP_CANDIDATE")
                     .order_by(Notification.id)).all()
        assert [n.push_status for n in s10] == ["SENT", "SUPPRESSED_COOLDOWN"]


def test_quiet_hours_queue_then_dispatch_and_expire(db, router):
    with Session(db) as s:
        _limit(s, 5)
        user = auth_service.get_user_by_email(s, "user@example.com")
        notify.register_device(s, user, "dev-1", "ios", "apns-tok")
        _two_runs(s, changes={"S10.NS": 99.0})
        night = dt.datetime(2026, 10, 5, 23, 30, tzinfo=IST)
        notify.process_ranking_run(s, "R2", now=night, router=router)
        n = s.exec(select(Notification).where(Notification.user_id == user.id)).one()
        assert n.push_status == "QUEUED_QUIET_HOURS"
        assert n.push_after.astimezone(IST).replace(tzinfo=None) == dt.datetime(2026, 10, 6, 7, 0)
        assert notify.dispatch_pending(s, night + dt.timedelta(hours=1), router)["sent"] == 0
        assert notify.dispatch_pending(s, dt.datetime(2026, 10, 6, 7, 5, tzinfo=IST), router)["sent"] == 1
        s.refresh(n)
        assert n.push_status == "SENT" and router.providers["apns"].sent[0][0] == "apns-tok"
        assert router.providers["apns"].sent[0][1].data["route"] == "/stock/S10.NS"


def test_quiet_window_math():
    pref = NotificationPreference(user_id=1, quiet_hours_start="22:00", quiet_hours_end="07:00")
    assert notify.quiet_until(pref, dt.datetime(2026, 10, 5, 12, 0, tzinfo=IST)) is None
    assert notify.quiet_until(pref, dt.datetime(2026, 10, 6, 6, 0, tzinfo=IST)).astimezone(IST).hour == 7
    day = NotificationPreference(user_id=1, quiet_hours_start="13:00", quiet_hours_end="14:00")
    assert notify.quiet_until(day, dt.datetime(2026, 10, 5, 13, 30, tzinfo=IST)) is not None
    off = NotificationPreference(user_id=1, quiet_hours_enabled=False)
    assert notify.quiet_until(off, dt.datetime(2026, 10, 5, 23, 0, tzinfo=IST)) is None


def test_global_emergency_switch_creates_nothing(db, router):
    with Session(db) as s:
        _set(s, "notifications.settings", {"enabled": False})
        _two_runs(s, changes={"S10.NS": 99.0})
        nrun = notify.process_ranking_run(s, "R2", now=NOON_IST, router=router)
        assert nrun.status == "SKIPPED" and not s.exec(select(Notification)).all()


def test_push_disabled_by_user_still_records_in_app(db, router):
    with Session(db) as s:
        _limit(s, 5)
        user = auth_service.get_user_by_email(s, "user@example.com")
        notify.register_device(s, user, "dev-1", "android", "tok")
        notify.update_preferences(s, user, {"push_enabled": False})
        _two_runs(s, changes={"S10.NS": 99.0})
        notify.process_ranking_run(s, "R2", now=NOON_IST, router=router)
        n = s.exec(select(Notification).where(Notification.user_id == user.id)).one()
        assert n.push_status == "SUPPRESSED_PREFERENCE" and not router.providers["fcm"].sent


def test_partial_runs_and_first_run_are_not_compared(db, router):
    with Session(db) as s:
        t0 = dt.datetime(2026, 10, 2, 11, 0, tzinfo=UTC)
        _run(s, "R1", t0, _base_scores())
        assert notify.process_ranking_run(s, "R1", router=router).status == "SKIPPED"
        _run(s, "RP", t0 + dt.timedelta(hours=1), {"S10.NS": 99.0}, config={"symbols": ["S10.NS"]})
        assert notify.process_ranking_run(s, "RP", router=router).detail["reason"].startswith("partial run")


# ── push delivery and devices ────────────────────────────────────────────────

def test_invalid_token_deactivates_device_and_no_device_is_recorded(db):
    bad = PushRouter({"fcm": FakeProvider("fcm", INVALID_TOKEN), "apns": FakeProvider("apns")})
    with Session(db) as s:
        _limit(s, 5)
        user = auth_service.get_user_by_email(s, "user@example.com")
        other = auth_service.get_user_by_email(s, "other@example.com")
        notify.register_device(s, user, "dev-1", "android", "stale-token")
        _two_runs(s, changes={"S10.NS": 99.0})
        notify.process_ranking_run(s, "R2", now=NOON_IST, router=bad)
        dev = s.exec(select(PushDevice).where(PushDevice.user_id == user.id)).one()
        assert not dev.active and dev.push_token is None and dev.invalid_reason
        assert s.exec(select(NotificationDelivery)).one().status == INVALID_TOKEN
        n_other = s.exec(select(Notification).where(Notification.user_id == other.id)).one()
        assert n_other.push_status == "NO_DEVICE"


def test_unconfigured_provider_is_reported_not_faked(db):
    router = PushRouter({"fcm": FcmProvider(project_id="", service_account_file=""),
                         "apns": ApnsProvider(key_file="", key_id="", team_id="", bundle_id="")})
    with Session(db) as s:
        _limit(s, 5)
        user = auth_service.get_user_by_email(s, "user@example.com")
        notify.register_device(s, user, "dev-1", "android", "tok")
        _two_runs(s, changes={"S10.NS": 99.0})
        notify.process_ranking_run(s, "R2", now=NOON_IST, router=router)
        n = s.exec(select(Notification).where(Notification.user_id == user.id)).one()
        assert n.push_status == "PROVIDER_NOT_CONFIGURED" and n.pushed_at is None


def test_device_registration_replacement_and_cross_account_token(db):
    with Session(db) as s:
        user = auth_service.get_user_by_email(s, "user@example.com")
        other = auth_service.get_user_by_email(s, "other@example.com")
        d = notify.register_device(s, user, "phone", "android", "token-A", "1.0", "granted")
        assert d.provider == "fcm" and notify.device_payload(d)["token"] == "...oken-A"
        d2 = notify.register_device(s, user, "phone", "android", "token-B", "1.1", "granted")
        assert d2.id == d.id and d2.push_token == "token-B"                  # replacement, same row
        notify.register_device(s, other, "phone", "android", "token-B")       # account switch on the device
        s.refresh(d2)
        assert not d2.active and d2.push_token is None
        with pytest.raises(ValueError):
            notify.register_device(s, user, "x", "windows", "t")
        assert notify.unregister_device(s, other.id, "phone")
        assert not s.exec(select(PushDevice).where(PushDevice.user_id == other.id)).one().active


class _Resp:
    def __init__(self, status, body=None, headers=None):
        self.status_code, self._body, self.headers = status, body or {}, headers or {}

    def json(self):
        return self._body


class _Http:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def post(self, url, **kw):
        self.calls.append((url, kw))
        return self.responses.pop(0)


def test_fcm_provider_sends_and_maps_unregistered_tokens(tmp_path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048).private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()).decode()
    sa = tmp_path / "sa.json"
    sa.write_text('{"client_email": "svc@p.iam.gserviceaccount.com", "private_key": %s}' % repr(key).replace("'", '"'))
    http = _Http(_Resp(200, {"access_token": "at", "expires_in": 3600}), _Resp(200, {"name": "projects/p/messages/1"}),
                 _Resp(404, {"error": {"status": "NOT_FOUND", "details": [{"errorCode": "UNREGISTERED"}]}}))
    p = FcmProvider("p", str(sa), http=http)
    msg = PushMessage("t", "b", {"route": "/stock/TCS.NS"})
    assert p.send("tok", msg).status == SENT
    url, kw = http.calls[1]
    assert url.endswith("/projects/p/messages:send") and kw["json"]["message"]["data"]["route"] == "/stock/TCS.NS"
    assert kw["headers"]["Authorization"] == "Bearer at"
    assert p.send("tok", msg).status == INVALID_TOKEN                    # access token cached, no 2nd OAuth call
    assert len(http.calls) == 3


def test_apns_provider_uses_token_auth_and_maps_410(tmp_path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    key = ec.generate_private_key(ec.SECP256R1()).private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()).decode()
    kf = tmp_path / "AuthKey.p8"
    kf.write_text(key)
    http = _Http(_Resp(200, headers={"apns-id": "abc"}), _Resp(410, {"reason": "Unregistered"}),
                 _Resp(500, {"reason": "InternalServerError"}))
    p = ApnsProvider(str(kf), "KEYID", "TEAMID", "com.example.app", sandbox=True, http=http)
    msg = PushMessage("t", "b", {"route": "/top-picks"})
    assert p.send("devtoken", msg).status == SENT
    url, kw = http.calls[0]
    assert url == "https://api.sandbox.push.apple.com/3/device/devtoken"
    assert kw["headers"]["apns-topic"] == "com.example.app" and kw["headers"]["authorization"].startswith("bearer ")
    assert kw["json"]["route"] == "/top-picks" and kw["json"]["aps"]["alert"]["title"] == "t"
    assert p.send("devtoken", msg).status == INVALID_TOKEN
    assert p.send("devtoken", msg).status == FAILED


# ── daily summary ────────────────────────────────────────────────────────────

def test_daily_summary_opt_in_time_once_per_day_and_trading_days_only(db, router):
    with Session(db) as s:
        user = auth_service.get_user_by_email(s, "user@example.com")
        notify.update_preferences(s, user, {"daily_summary": True, "daily_summary_time": "08:30"})
        t = dt.datetime(2026, 10, 2, 11, 0, tzinfo=UTC)
        _run(s, "R1", t, _base_scores())
        early = dt.datetime(2026, 10, 5, 8, 0, tzinfo=IST)
        assert notify.send_daily_summaries(s, early, router)["created"] == 0
        on_time = dt.datetime(2026, 10, 5, 8, 45, tzinfo=IST)
        assert notify.send_daily_summaries(s, on_time, router)["created"] == 1
        assert notify.send_daily_summaries(s, on_time + dt.timedelta(hours=1), router)["created"] == 0
        n = s.exec(select(Notification).where(Notification.type == "DAILY_SUMMARY")).one()
        assert n.user_id == user.id and n.route == "/top-picks"
        assert n.body.startswith("Analysis of 2026-10-02: 1. Stock 00 (90.0); 2. Stock 01 (89.0)")
        saturday = dt.datetime(2026, 10, 10, 9, 0, tzinfo=IST)
        assert notify.send_daily_summaries(s, saturday, router)["reason"] == "not a trading day"


# ── settings validation ──────────────────────────────────────────────────────

def test_settings_and_templates_are_validated():
    from notifications.settings import validate_settings
    with pytest.raises(ValueError):
        validate_settings({"score_change_threshold": 0})
    with pytest.raises(ValueError):
        validate_settings({"unknown": 1})
    with pytest.raises(ValueError):
        validate_templates({"NEW_TOP_CANDIDATE": {"title": "Hi {password}"}})
    t = validate_templates({"NEW_TOP_CANDIDATE": {"title": "{name} is a Top Candidate"}})
    assert t["NEW_TOP_CANDIDATE"]["title"] == "{name} is a Top Candidate" and t["NEW_TOP_CANDIDATE"]["body"]


# ── market status ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("when,holidays,expected", [
    (dt.datetime(2026, 10, 5, 9, 5, tzinfo=IST), [], "PRE_OPEN"),
    (dt.datetime(2026, 10, 5, 11, 0, tzinfo=IST), [], "OPEN"),
    (dt.datetime(2026, 10, 5, 15, 45, tzinfo=IST), [], "POST_CLOSE"),
    (dt.datetime(2026, 10, 5, 20, 0, tzinfo=IST), [], "CLOSED"),
    (dt.datetime(2026, 10, 4, 11, 0, tzinfo=IST), [], "WEEKEND"),
    (dt.datetime(2026, 10, 2, 11, 0, tzinfo=IST), ["2026-10-02"], "HOLIDAY"),
    (dt.datetime(2026, 10, 5, 5, 31, tzinfo=UTC), [], "OPEN"),          # 11:01 IST
])
def test_market_status(when, holidays, expected):
    st = market_status(when, holidays)
    assert st["status"] == expected and st["timezone"].startswith("Asia/Kolkata")
    assert st["holiday_calendar_configured"] == bool(holidays)


# ── API ──────────────────────────────────────────────────────────────────────

@pytest.fixture()
def client(db):
    def _session():
        with Session(db) as s:
            yield s
    app.dependency_overrides[get_session] = _session
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


def H(email="user@example.com", roles=("USER",)):
    return {"Authorization": f"Bearer {create_access_token(subject=email, roles=list(roles))}"}


ADMIN = lambda: H("admin@example.com", ("ADMIN",))  # noqa: E731


def test_notification_center_api_read_state_and_isolation(client, db, router):
    with Session(db) as s:
        _limit(s, 5)
        _two_runs(s, changes={"S10.NS": 99.0})
        notify.process_ranking_run(s, "R2", now=NOON_IST, router=router)
    r = client.get("/api/v1/notifications", headers=H())
    data = r.json()["data"]
    assert r.status_code == 200 and data["unread"] == 1 and data["items"][0]["route"] == "/stock/S10.NS"
    item = data["items"][0]
    assert item["type"] == "NEW_TOP_CANDIDATE" and item["symbol"] == "S10.NS" and item["engine_version"] == "ranking-v1.0"
    assert client.post(f"/api/v1/notifications/{item['id']}/read", headers=H("other@example.com")).status_code == 404
    assert client.post(f"/api/v1/notifications/{item['id']}/read", headers=H()).json()["data"]["read"] is True
    assert client.get("/api/v1/notifications/unread-count", headers=H()).json()["data"]["unread"] == 0
    other = client.get("/api/v1/notifications", headers=H("other@example.com")).json()["data"]
    assert other["unread"] == 1
    assert client.post("/api/v1/notifications/read-all", headers=H("other@example.com")).json()["data"]["marked"] == 1
    empty = client.get("/api/v1/notifications?unread_only=true", headers=H())
    assert empty.json()["message"] == "You're all caught up."


def test_preferences_api_validates_and_persists(client):
    r = client.get("/api/v1/notifications/preferences", headers=H()).json()["data"]
    assert r["new_top_candidate"] is True and r["daily_summary"] is False and r["timezone"] == "Asia/Kolkata"
    bad = client.put("/api/v1/notifications/preferences", json={"quiet_hours_start": "25:00"}, headers=H())
    assert bad.status_code == 400
    ok = client.put("/api/v1/notifications/preferences",
                    json={"daily_summary": True, "daily_summary_time": "09:00"}, headers=H())
    assert ok.status_code == 200 and ok.json()["data"]["daily_summary_time"] == "09:00"
    assert client.get("/api/v1/notifications/preferences", headers=H("other@example.com")).json()["data"][
        "daily_summary"] is False


def test_device_api_masks_tokens_and_logout_removes_device(client, db):
    body = {"device_id": "dev-x", "platform": "android", "push_token": "secret-push-token-123",
            "app_version": "1.0", "permission": "granted"}
    r = client.post("/api/v1/devices", json=body, headers=H())
    assert r.status_code == 200 and r.json()["data"]["token"] == "...en-123"
    assert "secret-push-token" not in client.get("/api/v1/devices", headers=H()).text
    assert client.get("/api/v1/devices", headers=H("other@example.com")).json()["data"] == []
    assert client.post("/api/v1/devices", json={**body, "platform": "web"}, headers=H()).status_code == 422
    # Real login/logout with a device id: logout deactivates the push device.
    login = client.post("/api/v1/auth/login", data={"username": "user@example.com", "password": "UserPass1!",
                                                    "device_id": "dev-x"})
    assert login.status_code == 200
    assert client.post("/api/v1/auth/logout", json={"refresh_token": login.json()["refresh_token"]}).status_code == 204
    with Session(db) as s:
        d = s.exec(select(PushDevice).where(PushDevice.device_id == "dev-x")).one()
        assert not d.active and d.push_token is None
    assert client.delete("/api/v1/devices/unknown", headers=H()).status_code == 404


def test_watchlist_without_buy_price_overview_and_alerts(client, db):
    with Session(db) as s:
        _two_runs(s, changes={"S03.NS": 75.0})
    r = client.post("/api/v1/watchlist", json={"symbol": "S03.NS"}, headers=H())
    assert r.status_code == 201 and r.json()["buy_price"] is None
    ov = client.get("/api/v1/watchlist/overview", headers=H()).json()["data"]["items"][0]
    assert ov["symbol"] == "S03.NS" and ov["stockai_score"] == 75.0 and ov["score_change"] == -12.0
    assert ov["previous_score"] == 87.0 and ov["rank"] is not None and ov["fqvf_passed"] == 12
    assert ov["alerts"] == {"score_changes": True, "rank_changes": True, "fqvf_changes": True,
                            "status_changes": True, "muted": False} and ov["alerts_active"] is True
    upd = client.put(f"/api/v1/watchlist/{r.json()['id']}/alerts", json={"muted": True}, headers=H())
    assert upd.status_code == 200 and upd.json()["data"]["muted"] is True
    assert client.put(f"/api/v1/watchlist/{r.json()['id']}/alerts", json={"muted": True},
                      headers=H("other@example.com")).status_code == 404
    assert client.get("/api/v1/watchlist/overview", headers=H("other@example.com")).json()["message"] == \
        "Your watchlist is empty."


def test_feedback_and_account_deletion_request_reach_admin(client):
    r = client.post("/api/v1/feedback", json={"category": "STALE_DATA", "message": "Price looks a week old",
                                              "symbol": "s03.ns"}, headers=H())
    assert r.status_code == 201 and r.json()["data"]["symbol"] == "S03.NS"
    assert "recorded" in r.json()["message"] and "sent" not in r.json()["message"].lower()
    d = client.post("/api/v1/account/deletion-request", json={"reason": "no longer needed"}, headers=H())
    assert d.status_code == 201 and d.json()["data"]["category"] == "ACCOUNT_DELETION"
    assert client.get("/api/v1/admin/feedback", headers=H()).status_code == 403
    rows = client.get("/api/v1/admin/feedback", headers=ADMIN()).json()
    assert {x["category"] for x in rows} == {"STALE_DATA", "ACCOUNT_DELETION"}
    fid = rows[0]["id"]
    assert client.patch(f"/api/v1/admin/feedback/{fid}", json={"status": "RESOLVED"}, headers=ADMIN()).status_code == 200
    mine = client.get("/api/v1/feedback", headers=H()).json()["data"]
    assert any(x["status"] == "RESOLVED" for x in mine)
    for _ in range(8):
        client.post("/api/v1/feedback", json={"category": "APP_BUG", "message": "Something broke"}, headers=H())
    assert client.post("/api/v1/feedback", json={"category": "APP_BUG", "message": "Again broke"},
                       headers=H()).status_code == 429


def test_admin_notification_controls_are_authorized_and_audited(client, db, router):
    from notifications import push
    push.set_router(router)
    try:
        for path in ("/admin/notifications/stats", "/admin/notifications/runs", "/admin/notifications/recent"):
            assert client.get("/api/v1" + path, headers=H()).status_code == 403
            assert client.get("/api/v1" + path, headers=ADMIN()).status_code == 200
        for path in ("/admin/notifications/process-run", "/admin/notifications/daily-summary",
                     "/admin/notifications/dispatch", "/admin/notifications/test"):
            assert client.post("/api/v1" + path, json={}, headers=H()).status_code == 403
        bad = client.put("/api/v1/admin/config/notifications.settings", json={"value": {"score_change_threshold": -1}},
                         headers=ADMIN())
        assert bad.status_code == 400
        ok = client.put("/api/v1/admin/config/notifications.settings", json={"value": {"enabled": False}},
                        headers=ADMIN())
        assert ok.status_code == 200
        assert client.post("/api/v1/admin/notifications/test", headers=ADMIN()).status_code == 200
        with Session(db) as s:
            _two_runs(s)
        r = client.post("/api/v1/admin/notifications/process-run", json={}, headers=ADMIN())
        assert r.status_code == 200 and r.json()["status"] == "SKIPPED"          # globally disabled
        with Session(db) as s:
            actions = {a.action for a in s.exec(select(AdminAuditLog)).all()}
        assert {"CONFIG_UPDATED", "NOTIFICATIONS_PROCESSED", "TEST_NOTIFICATION_SENT"} <= actions
    finally:
        push.set_router(None)


def test_market_status_performance_and_intelligence_overviews(client):
    st = client.get("/api/v1/market/status", headers=H())
    assert st.status_code == 200 and st.json()["data"]["status"] in (
        "PRE_OPEN", "OPEN", "CLOSING", "POST_CLOSE", "CLOSED", "WEEKEND", "HOLIDAY")
    assert client.get("/api/v1/market/status").status_code == 401
    perf = client.get("/api/v1/performance/overview", headers=H()).json()["data"]
    sec = perf["sections"]
    assert perf["order"][0] == "ranking_v1_prospective"
    assert sec["ranking_v1_prospective"]["status"] == "COLLECTING"
    assert sec["ranking_v1_prospective"]["message"] == \
        "Performance tracking will appear after enough observations are available."
    assert "not establish" in sec["ranking_v1_validation"]["conclusion"]
    assert "Retired" in sec["legacy_signals"]["caveat"] and "success_rate" not in sec["legacy_signals"]
    intel = client.get("/api/v1/intelligence/overview", headers=H()).json()["data"]
    keys = [x["key"] for x in intel["sections"]]
    assert keys == ["market", "fundamental", "technical", "ml_signal", "news", "legacy"]
    ml = next(x for x in intel["sections"] if x["key"] == "ml_signal")
    assert "Informational only - weight 0" in ml["status"]
    cfg = client.get("/api/v1/app/config").json()["data"]
    assert len(cfg["onboarding"]) >= 5 and "privacy_summary" in cfg["legal"]


def test_stock_analysis_explanation_labels_freshness_and_single_run_rank(client, db):
    with Session(db) as s:
        _limit(s, 5)
        t = dt.datetime.now(UTC) - dt.timedelta(hours=2)
        _run(s, "R1", t, _base_scores(), risk={"S03.NS": 20.0})
        s.add(MarketSnapshot(symbol="S03.NS", fetched_at=t - dt.timedelta(minutes=1), as_of_date=t.date().isoformat(),
                             close=123.45, technical={}))
        s.commit()
    a = client.get("/api/v1/stocks/S03.NS/analysis", headers=H()).json()["data"]
    assert a["explanation"]["summary"].startswith("Ranked 4 of 30 eligible stocks, inside the Top 5")
    labels = {x["label"] for x in a["labels"]}
    assert {"Top Candidate", "Strong Quality", "Attractive Valuation", "High Risk"} <= labels
    assert a["freshness_status"]["status"] == "OK" and a["reference_price"]["close"] == 123.45
    assert a["market_regime"]["regime"] == "Bullish"
    tp = client.get("/api/v1/top-picks", headers=H()).json()["data"]["items"]
    assert tp[0]["labels"][0]["label"] == "Top Candidate" and len(tp[0]["components"]) == 6
    # An on-demand single-stock run must not show "rank 1 of 1".
    with Session(db) as s:
        _run(s, "SINGLE-1", dt.datetime.now(UTC) - dt.timedelta(minutes=5), {"S03.NS": 70.0}, kind="SINGLE",
             config={"symbols": ["S03.NS"]})
    a2 = client.get("/api/v1/stocks/S03.NS/analysis", headers=H()).json()["data"]
    assert a2["ranking"]["rank"] == 4 and a2["ranking"]["universe_rank"]["run_id"] == "R1"
    assert a2["explanation"]["summary"].startswith("On-demand analysis of this stock. In the last full ranking run")


def test_stale_market_data_is_flagged():
    from ranking.presenter import freshness_status
    r = StockAnalysisResult(run_id="x", symbol="X", computed_at=dt.datetime.now(UTC) - dt.timedelta(days=10),
                            freshness={"market_data_as_of": "2020-01-01"}, engine_version="v", fqvf_version="v")
    f = freshness_status(r)
    assert f["status"] == "STALE" and any("may be stale" in i for i in f["issues"])
    r2 = StockAnalysisResult(run_id="x", symbol="X", computed_at=dt.datetime.now(UTC), freshness={},
                             engine_version="v", fqvf_version="v")
    assert freshness_status(r2)["status"] == "UNAVAILABLE"


def test_top_picks_prefers_latest_full_run_over_partial(client, db):
    with Session(db) as s:
        t = dt.datetime.now(UTC) - dt.timedelta(hours=3)
        _run(s, "FULL", t, _base_scores())
        _run(s, "PART", t + dt.timedelta(hours=1), {"S29.NS": 99.0}, config={"limit": 1})
    data = client.get("/api/v1/top-picks", headers=H()).json()["data"]
    assert data["run"]["run_id"] == "FULL" and data["items"][0]["symbol"] == "S00.NS"


# ── scheduler calendar rules ─────────────────────────────────────────────────

def test_scheduled_ranking_job_respects_calendar_and_data(db, monkeypatch):
    import pandas as pd
    import scripts.scheduled_jobs as jobs
    monkeypatch.setattr(jobs, "engine", db)
    started = []
    monkeypatch.setattr(jobs.runs, "execute_run", lambda e, run_id: started.append(run_id))
    monkeypatch.setattr(jobs.runs, "notify_after_run", lambda e, run_id: None)
    with Session(db) as s:
        _set(s, "market.holidays", ["2026-10-02"])
    no_call = lambda: (_ for _ in ()).throw(AssertionError("provider must not be called"))  # noqa: E731
    assert "not a trading day" in jobs.ranking_job(dt.datetime(2026, 10, 3, 17, 0, tzinfo=IST), no_call)["reason"]
    assert "not a trading day" in jobs.ranking_job(dt.datetime(2026, 10, 2, 17, 0, tzinfo=IST), no_call)["reason"]
    assert "before 16:00" in jobs.ranking_job(dt.datetime(2026, 10, 5, 15, 0, tzinfo=IST), no_call)["reason"]
    stale = pd.DataFrame({"Close": [1.0]}, index=pd.to_datetime(["2026-10-01"]))
    assert "not available yet" in jobs.ranking_job(dt.datetime(2026, 10, 5, 17, 0, tzinfo=IST), lambda: stale)["reason"]
    fresh = pd.DataFrame({"Close": [1.0]}, index=pd.to_datetime(["2026-10-05"]))
    jobs.ranking_job(dt.datetime(2026, 10, 5, 17, 0, tzinfo=IST), lambda: fresh)
    assert len(started) == 1
