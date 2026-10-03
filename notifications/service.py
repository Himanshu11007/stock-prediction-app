"""
notifications/service.py — the notification engine, separate from ranking.

Pipeline for one completed ranking run (process_ranking_run):

  ranking run -> compare with the previous full run (detector) -> material
  changes -> per-user drafts (preferences, watchlist, per-stock switches)
  -> deduplicate (user + dedup key) -> rate limit / cooldown / quiet hours
  -> in-app Notification rows -> push dispatch -> NotificationDelivery rows

Every notification is stored in the user's Notification Center, even when
its push is suppressed, queued for quiet hours, or no device/provider is
available, so an important event is never lost because a push was missed.

Deduplication rule: Notification has UNIQUE(user_id, dedup_key) with
  dedup_key = "<TYPE>:<SYMBOL or MULTI>:<ranking run_id>"   ranking events
              "MARKET_REGIME_CHANGE:<ranking run_id>"
              "DAILY_SUMMARY:<IST date>"
and NotificationRun has UNIQUE(kind, source_key), so re-processing the same
ranking run (retry, second scheduler tick) creates nothing new.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Optional

from sqlalchemy import func
from sqlmodel import Session, select

import masters.service as masters
from config import NOTIFICATION_ENGINE_VERSION
from db.models.market import EngineRun, StockAnalysisResult
from db.models.notifications import (Notification, NotificationDelivery, NotificationPreference, NotificationRun,
                                     PushDevice, WatchlistAlertSetting)
from db.models.stock import Company
from db.models.tracker import WatchlistItem
from db.models.user import User
from notifications import detector
from notifications.push import INVALID_TOKEN, NOT_CONFIGURED, SENT, PushMessage, PushRouter, get_router
from notifications.settings import render
from utils.logger import get_logger
from utils.market_session import IST, is_trading_day

logger = get_logger(__name__)

PREF_FIELDS = ("push_enabled", "new_top_candidate", "top_candidate_removed", "score_changes", "fqvf_changes",
               "watchlist_alerts", "daily_summary", "market_regime", "quiet_hours_enabled", "quiet_hours_start",
               "quiet_hours_end", "daily_summary_time")
ALERT_FIELDS = ("score_changes", "rank_changes", "fqvf_changes", "status_changes", "muted")
# Order in which a user's notifications of one run claim the push budget.
PRIORITY = {"WATCHLIST_ALERT": 0, "NEW_TOP_CANDIDATE": 1, "TOP_CANDIDATE_REMOVED": 2, "SCORE_CHANGE": 3,
            "FQVF_CHANGE": 4, "MARKET_REGIME_CHANGE": 5, "DAILY_SUMMARY": 6, "TEST": 7}


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _aware(t: Optional[dt.datetime]) -> Optional[dt.datetime]:
    return None if t is None else (t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc))


def settings(session: Session) -> dict:
    return masters.get_config(session, "notifications.settings")


def templates(session: Session) -> dict:
    return masters.get_config(session, "notifications.templates")


# ── preferences ──────────────────────────────────────────────────────────────

def default_preferences(session: Session, user_id: int) -> NotificationPreference:
    s = settings(session)
    q = s["default_quiet_hours"]
    return NotificationPreference(user_id=user_id, quiet_hours_enabled=q["enabled"], quiet_hours_start=q["start"],
                                  quiet_hours_end=q["end"], daily_summary_time=s["default_daily_summary_time"])


def get_preferences(session: Session, user: User) -> NotificationPreference:
    return session.get(NotificationPreference, user.id) or default_preferences(session, user.id)


def preferences_payload(p: NotificationPreference) -> dict:
    return {**{k: getattr(p, k) for k in PREF_FIELDS}, "timezone": "Asia/Kolkata",
            "updated_at": _aware(p.updated_at).isoformat() if p.updated_at else None}


def update_preferences(session: Session, user: User, changes: dict) -> NotificationPreference:
    from notifications.settings import is_hhmm
    unknown = set(changes) - set(PREF_FIELDS)
    if unknown:
        raise ValueError(f"Unknown preference fields: {sorted(unknown)}")
    for k, v in changes.items():
        if k in ("quiet_hours_start", "quiet_hours_end", "daily_summary_time"):
            if not is_hhmm(v):
                raise ValueError(f"{k} must be HH:MM (24-hour, India time)")
        elif not isinstance(v, bool):
            raise ValueError(f"{k} must be true or false")
    pref = session.get(NotificationPreference, user.id) or default_preferences(session, user.id)
    for k, v in changes.items():
        setattr(pref, k, v)
    pref.updated_at = _now()
    session.add(pref)
    session.commit()
    session.refresh(pref)
    return pref


def _hhmm(s: str) -> dt.time:
    h, m = s.split(":")
    return dt.time(int(h), int(m))


def quiet_until(pref: NotificationPreference, now: dt.datetime) -> Optional[dt.datetime]:
    """End of the current quiet window (UTC) if `now` falls inside it."""
    if not pref.quiet_hours_enabled:
        return None
    local = now.astimezone(IST)
    start, end, t = _hhmm(pref.quiet_hours_start), _hhmm(pref.quiet_hours_end), local.time()
    if start == end:
        return None
    if start < end:
        inside, end_day = start <= t < end, local.date()
    else:  # overnight window, e.g. 22:00-07:00
        inside = t >= start or t < end
        end_day = local.date() + dt.timedelta(days=1) if t >= start else local.date()
    if not inside:
        return None
    return dt.datetime.combine(end_day, end, tzinfo=IST).astimezone(dt.timezone.utc)


# ── devices ──────────────────────────────────────────────────────────────────

PLATFORM_PROVIDER = {"android": "fcm", "ios": "apns"}


def mask_token(token: Optional[str]) -> Optional[str]:
    return None if not token else f"...{token[-6:]}"


def device_payload(d: PushDevice) -> dict:
    return {"device_id": d.device_id, "platform": d.platform, "provider": d.provider,
            "token": mask_token(d.push_token), "app_version": d.app_version, "permission": d.permission,
            "active": d.active, "created_at": _aware(d.created_at).isoformat(),
            "last_active_at": _aware(d.last_active_at).isoformat(),
            "invalid_reason": d.invalid_reason}


def register_device(session: Session, user: User, device_id: str, platform: str, push_token: Optional[str],
                    app_version: Optional[str] = None, permission: str = "unknown",
                    provider: Optional[str] = None) -> PushDevice:
    platform = (platform or "").lower()
    if platform not in PLATFORM_PROVIDER:
        raise ValueError("platform must be 'android' or 'ios'")
    provider = (provider or PLATFORM_PROVIDER[platform]).lower()
    if provider not in ("fcm", "apns"):
        raise ValueError("provider must be 'fcm' or 'apns'")
    if permission not in ("granted", "denied", "unknown"):
        raise ValueError("permission must be granted, denied or unknown")
    if not device_id or len(device_id) > 200:
        raise ValueError("device_id is required (at most 200 characters)")
    if push_token is not None and (not push_token.strip() or len(push_token) > 4096):
        raise ValueError("push_token must be a non-empty string of at most 4096 characters")
    now = _now()
    if push_token:
        # A token belongs to one installation: if another account registered
        # it earlier (shared device, account switch), stop pushing to them.
        for other in session.exec(select(PushDevice).where(PushDevice.push_token == push_token,
                                                           PushDevice.active == True)).all():  # noqa: E712
            if other.user_id != user.id or other.device_id != device_id:
                other.active, other.push_token = False, None
                other.invalidated_at, other.invalid_reason = now, "token registered by another session"
                session.add(other)
    dev = session.exec(select(PushDevice).where(PushDevice.user_id == user.id,
                                                PushDevice.device_id == device_id)).first()
    if dev is None:
        dev = PushDevice(user_id=user.id, device_id=device_id, platform=platform, provider=provider)
    dev.platform, dev.provider, dev.app_version, dev.permission = platform, provider, app_version, permission
    dev.push_token = push_token
    dev.active, dev.last_active_at, dev.invalidated_at, dev.invalid_reason = True, now, None, None
    session.add(dev)
    session.commit()
    session.refresh(dev)
    return dev


def unregister_device(session: Session, user_id: int, device_id: str, reason: str = "signed out") -> bool:
    dev = session.exec(select(PushDevice).where(PushDevice.user_id == user_id,
                                                PushDevice.device_id == device_id)).first()
    if dev is None:
        return False
    dev.active, dev.push_token, dev.invalidated_at, dev.invalid_reason = False, None, _now(), reason
    session.add(dev)
    session.commit()
    return True


def list_devices(session: Session, user: User) -> list[PushDevice]:
    return list(session.exec(select(PushDevice).where(PushDevice.user_id == user.id)
                             .order_by(PushDevice.last_active_at.desc())).all())


# ── watchlist alert switches ─────────────────────────────────────────────────

def alert_setting(session: Session, user_id: int, symbol: str) -> WatchlistAlertSetting:
    return session.exec(select(WatchlistAlertSetting).where(WatchlistAlertSetting.user_id == user_id,
                                                            WatchlistAlertSetting.symbol == symbol)).first() \
        or WatchlistAlertSetting(user_id=user_id, symbol=symbol)


def update_alert_setting(session: Session, user: User, symbol: str, changes: dict) -> WatchlistAlertSetting:
    unknown = set(changes) - set(ALERT_FIELDS)
    if unknown:
        raise ValueError(f"Unknown alert fields: {sorted(unknown)}")
    if not all(isinstance(v, bool) for v in changes.values()):
        raise ValueError("alert settings must be true or false")
    row = alert_setting(session, user.id, symbol)
    for k, v in changes.items():
        setattr(row, k, v)
    row.updated_at = _now()
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def alert_payload(a: WatchlistAlertSetting) -> dict:
    return {k: getattr(a, k) for k in ALERT_FIELDS}


# ── inbox ────────────────────────────────────────────────────────────────────

def notification_payload(n: Notification) -> dict:
    return {"id": n.id, "type": n.type, "title": n.title, "body": n.body, "symbol": n.symbol, "route": n.route,
            "data": n.data or {}, "ranking_run_id": n.ranking_run_id, "engine_version": n.engine_version,
            "created_at": _aware(n.created_at).isoformat(), "read": n.read_at is not None,
            "read_at": _aware(n.read_at).isoformat() if n.read_at else None, "push_status": n.push_status}


def list_notifications(session: Session, user: User, unread_only: bool = False, limit: int = 50,
                       offset: int = 0) -> tuple[list[Notification], int, int]:
    q = select(Notification).where(Notification.user_id == user.id)
    if unread_only:
        q = q.where(Notification.read_at.is_(None))
    rows = session.exec(q.order_by(Notification.created_at.desc(), Notification.id.desc())
                        .offset(offset).limit(limit)).all()
    total = session.exec(select(func.count()).select_from(Notification).where(Notification.user_id == user.id)).one()
    return list(rows), int(total), unread_count(session, user)


def unread_count(session: Session, user: User) -> int:
    return int(session.exec(select(func.count()).select_from(Notification).where(
        Notification.user_id == user.id, Notification.read_at.is_(None))).one())


def mark_read(session: Session, user: User, notification_id: int) -> Optional[Notification]:
    n = session.get(Notification, notification_id)
    if n is None or n.user_id != user.id:
        return None
    if n.read_at is None:
        n.read_at = _now()
        session.add(n)
        session.commit()
        session.refresh(n)
    return n


def mark_all_read(session: Session, user: User) -> int:
    rows = session.exec(select(Notification).where(Notification.user_id == user.id,
                                                   Notification.read_at.is_(None))).all()
    now = _now()
    for n in rows:
        n.read_at = now
        session.add(n)
    session.commit()
    return len(rows)


# ── fan-out ──────────────────────────────────────────────────────────────────

def _score(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:.1f}"


def _short_list(items: list[str], n: int = 5) -> str:
    return ", ".join(items[:n]) + (f" +{len(items) - n} more" if len(items) > n else "")


def _draft(type_: str, dedup: str, title: str, body: str, *, symbol=None, route="/notifications", data=None):
    return {"type": type_, "dedup_key": dedup, "title": title, "body": body, "symbol": symbol, "route": route,
            "data": data or {}}


def _stock_route(symbol: str) -> str:
    return f"/stock/{symbol}"


def _watchlist_text(c: detector.StockChange, a: WatchlistAlertSetting) -> list[str]:
    parts = []
    if a.score_changes and c.score:
        parts.append(f"StockLens Score {_score(c.score['old'])} -> {_score(c.score['new'])} "
                     f"({c.score['delta']:+.1f})")
    if a.rank_changes and c.rank:
        parts.append(f"rank {c.rank['old']} -> {c.rank['new']}")
    if a.rank_changes and c.entered_top:
        parts.append("entered the Top Investment Candidates")
    if a.rank_changes and c.left_top:
        parts.append("left the Top Investment Candidates")
    if a.fqvf_changes and c.fqvf:
        parts.append(f"FQVF {c.fqvf['old_passes']}/18 -> {c.fqvf['new_passes']}/18 checks passed")
    if a.status_changes and c.status:
        if "eligible" in c.status:
            parts.append("now eligible for Top Candidates" if c.status["eligible"]
                         else "no longer eligible (" + "; ".join(c.status.get("reasons") or ["data rules"]) + ")")
        if "high_risk" in c.status:
            parts.append("now rated high risk" if c.status["high_risk"] else "no longer rated high risk")
    return parts


def build_drafts(cmp: detector.RunComparison, pref: NotificationPreference, watched: set[str],
                 alerts: dict[str, WatchlistAlertSetting], s: dict, tpl: dict) -> list[dict]:
    run_id, types, agg = cmp.run.run_id, s["event_types"], s["aggregate_above"]
    drafts: list[dict] = []
    handled: set[str] = set()

    # Watchlist first: one combined alert per watched stock.
    if pref.watchlist_alerts and types["WATCHLIST_ALERT"]:
        for sym in sorted(watched):
            c = cmp.changes.get(sym)
            a = alerts.get(sym) or WatchlistAlertSetting(symbol=sym)
            if c is None or a.muted:
                continue
            parts = _watchlist_text(c, a)
            if parts:
                text = "; ".join(parts)
                title, body = render(tpl, "WATCHLIST_ALERT", name=c.name, symbol=sym, changes=text[0].upper() + text[1:])
                drafts.append(_draft("WATCHLIST_ALERT", f"WATCHLIST_ALERT:{sym}:{run_id}", title, body, symbol=sym,
                                     route=_stock_route(sym), data={"changes": parts}))
                handled.add(sym)

    def membership(kind: str, key: str, pref_on: bool):
        items = [c for c in cmp.changes.values() if getattr(c, key) and c.symbol not in handled]
        if not (pref_on and types[kind] and items):
            return
        if len(items) > agg:
            names = [c.name for c in items]
            title, body = render(tpl, f"{kind}_MULTI", count=len(items), list=_short_list(names))
            drafts.append(_draft(kind, f"{kind}:MULTI:{run_id}", title, body, route="/top-picks",
                                 data={"symbols": [c.symbol for c in items]}))
        else:
            for c in items:
                v = getattr(c, key)
                title, body = render(tpl, kind, name=c.name, symbol=c.symbol, rank=v.get("rank"),
                                     score=_score(v.get("score")), reason=v.get("reason"))
                drafts.append(_draft(kind, f"{kind}:{c.symbol}:{run_id}", title, body, symbol=c.symbol,
                                     route=_stock_route(c.symbol), data=v))
        handled.update(c.symbol for c in items)

    membership("NEW_TOP_CANDIDATE", "entered_top", pref.new_top_candidate)
    membership("TOP_CANDIDATE_REMOVED", "left_top", pref.top_candidate_removed)

    # Score / FQVF changes of (current or previous) Top Candidates not already covered.
    relevant = set(cmp.current_top) | set(cmp.previous_top)
    for kind, key, pref_on in (("SCORE_CHANGE", "score", pref.score_changes),
                               ("FQVF_CHANGE", "fqvf", pref.fqvf_changes)):
        items = [c for c in cmp.changes.values() if getattr(c, key) and c.symbol in relevant
                 and c.symbol not in handled]
        if not (pref_on and types[kind] and items):
            continue
        if len(items) > agg:
            label = (lambda c: f"{c.name} ({c.score['delta']:+.1f})") if key == "score" else \
                    (lambda c: f"{c.name} ({c.fqvf['old_passes']}->{c.fqvf['new_passes']})")
            title, body = render(tpl, f"{kind}_MULTI", count=len(items), list=_short_list([label(c) for c in items]))
            drafts.append(_draft(kind, f"{kind}:MULTI:{run_id}", title, body, route="/top-picks",
                                 data={"symbols": [c.symbol for c in items]}))
        else:
            for c in items:
                v = getattr(c, key)
                title, body = render(tpl, kind, name=c.name, symbol=c.symbol, old_score=_score(v.get("old")),
                                     new_score=_score(v.get("new")),
                                     delta=f"{v['delta']:+.1f}" if "delta" in v else None,
                                     old_passes=v.get("old_passes"), new_passes=v.get("new_passes"))
                drafts.append(_draft(kind, f"{kind}:{c.symbol}:{run_id}", title, body, symbol=c.symbol,
                                     route=_stock_route(c.symbol), data=v))

    if cmp.regime and pref.market_regime and types["MARKET_REGIME_CHANGE"]:
        title, body = render(tpl, "MARKET_REGIME_CHANGE", old_regime=cmp.regime["old"], new_regime=cmp.regime["new"])
        drafts.append(_draft("MARKET_REGIME_CHANGE", f"MARKET_REGIME_CHANGE:{run_id}", title, body, route="/",
                             data=cmp.regime))
    return drafts


def _decide_push(session: Session, user_id: int, draft: dict, pref: NotificationPreference, s: dict,
                 now: dt.datetime, pushed_this_run: int) -> tuple[str, Optional[dt.datetime]]:
    if not s["push_enabled"]:
        return "DISABLED_GLOBALLY", None
    if not pref.push_enabled:
        return "SUPPRESSED_PREFERENCE", None
    if draft["type"] == "MARKET_REGIME_CHANGE" and s["regime_cooldown_hours"]:
        since = now - dt.timedelta(hours=s["regime_cooldown_hours"])
        if session.exec(select(Notification.id).where(Notification.user_id == user_id,
                                                      Notification.type == "MARKET_REGIME_CHANGE",
                                                      Notification.pushed_at >= since)).first():
            return "SUPPRESSED_COOLDOWN", None
    if draft["symbol"] and s["stock_cooldown_hours"]:
        since = now - dt.timedelta(hours=s["stock_cooldown_hours"])
        if session.exec(select(Notification.id).where(Notification.user_id == user_id,
                                                      Notification.symbol == draft["symbol"],
                                                      Notification.type == draft["type"],
                                                      Notification.pushed_at >= since)).first():
            return "SUPPRESSED_COOLDOWN", None
    day_ago = now - dt.timedelta(hours=24)
    pushed_today = session.exec(select(func.count()).select_from(Notification).where(
        Notification.user_id == user_id, Notification.pushed_at >= day_ago)).one()
    if pushed_this_run >= s["max_push_per_user_per_run"] or pushed_today + pushed_this_run >= s["max_push_per_user_per_day"]:
        return "SUPPRESSED_RATE_LIMIT", None
    until = quiet_until(pref, now)
    if until is not None:
        return "QUEUED_QUIET_HOURS", until
    return "PENDING", None


def _store(session: Session, user_id: int, drafts: list[dict], pref: NotificationPreference, s: dict,
           now: dt.datetime, nrun: NotificationRun, run_id: Optional[str], engine_version: Optional[str]) -> list[Notification]:
    existing = set(session.exec(select(Notification.dedup_key).where(
        Notification.user_id == user_id, Notification.dedup_key.in_([d["dedup_key"] for d in drafts]))).all())
    created, budget_used = [], 0
    for d in sorted(drafts, key=lambda d: PRIORITY.get(d["type"], 9)):
        if d["dedup_key"] in existing:
            continue
        status, after = _decide_push(session, user_id, d, pref, s, now, budget_used)
        if status in ("PENDING", "QUEUED_QUIET_HOURS"):
            budget_used += 1
        n = Notification(user_id=user_id, notification_run_id=nrun.id, type=d["type"], dedup_key=d["dedup_key"],
                         title=d["title"], body=d["body"], symbol=d["symbol"], route=d["route"], data=d["data"],
                         ranking_run_id=run_id, engine_version=engine_version, created_at=now,
                         push_status=status, push_after=after)
        session.add(n)
        created.append(n)
        existing.add(d["dedup_key"])
    return created


def _users_and_context(session: Session) -> tuple[list[User], dict, dict, dict]:
    users = list(session.exec(select(User).where(User.is_active == True)).all())  # noqa: E712
    prefs = {p.user_id: p for p in session.exec(select(NotificationPreference)).all()}
    watched: dict[int, set[str]] = {}
    for w in session.exec(select(WatchlistItem)).all():
        watched.setdefault(w.user_id, set()).add(w.symbol)
    alerts: dict[int, dict[str, WatchlistAlertSetting]] = {}
    for a in session.exec(select(WatchlistAlertSetting)).all():
        alerts.setdefault(a.user_id, {})[a.symbol] = a
    return users, prefs, watched, alerts


def process_ranking_run(session: Session, run_id: str, now: Optional[dt.datetime] = None,
                        router: Optional[PushRouter] = None) -> NotificationRun:
    """Detect changes for a completed ranking run and notify users. Idempotent."""
    now = now or _now()
    existing = session.exec(select(NotificationRun).where(NotificationRun.kind == "RANKING_CHANGES",
                                                          NotificationRun.source_key == run_id)).first()
    if existing is not None:
        return existing
    s = settings(session)
    run = session.exec(select(EngineRun).where(EngineRun.run_id == run_id)).first()
    nrun = NotificationRun(kind="RANKING_CHANGES", source_key=run_id, engine_version=NOTIFICATION_ENGINE_VERSION,
                           started_at=now)

    def skip(reason: str) -> NotificationRun:
        nrun.status, nrun.finished_at, nrun.detail = "SKIPPED", _now(), {"reason": reason}
        session.add(nrun)
        session.commit()
        session.refresh(nrun)
        return nrun

    if not s["enabled"]:
        return skip("notifications globally disabled")
    if run is None or run.status not in detector.COMPLETED:
        return skip("ranking run not found or not completed")
    if not detector.is_full_universe(run):
        return skip("partial run (explicit symbols or limit); not compared")
    previous = detector.previous_full_run(session, run)
    if previous is None:
        return skip("no previous full ranking run to compare against")
    nrun.previous_run_id = previous.run_id
    session.add(nrun)
    session.commit()
    session.refresh(nrun)
    try:
        cmp = detector.compare(session, run, previous, s, masters.get_config(session, "top_picks.limit"))
        tpl = templates(session)
        users, prefs, watched, alerts = _users_and_context(session)
        created: list[Notification] = []
        for u in users:
            pref = prefs.get(u.id) or default_preferences(session, u.id)
            drafts = build_drafts(cmp, pref, watched.get(u.id, set()), alerts.get(u.id, {}), s, tpl)
            if drafts:
                created += _store(session, u.id, drafts, pref, s, now, nrun, run_id, run.engine_version)
        session.commit()
        nrun.events_detected = len(cmp.changes) + (1 if cmp.regime else 0)
        nrun.notifications_created = len(created)
        nrun.detail = {"entered_top": [c.symbol for c in cmp.changes.values() if c.entered_top],
                       "left_top": [c.symbol for c in cmp.changes.values() if c.left_top],
                       "score_changes": sum(1 for c in cmp.changes.values() if c.score),
                       "fqvf_changes": sum(1 for c in cmp.changes.values() if c.fqvf),
                       "regime_change": cmp.regime}
        result = dispatch_pending(session, now, router, notification_ids=[n.id for n in created])
        nrun.pushes_sent = result["sent"]
        nrun.pushes_suppressed = sum(1 for n in created if n.push_status.startswith("SUPPRESSED")
                                     or n.push_status in ("DISABLED_GLOBALLY",))
        nrun.status = "COMPLETED"
    except Exception as e:
        session.rollback()
        logger.exception("NOTIFICATION_RUN_FAILED | %s", run_id)
        nrun = session.get(NotificationRun, nrun.id)
        nrun.status, nrun.detail = "FAILED", {"error": f"{type(e).__name__}: {e}"[:500]}
    nrun.finished_at = _now()
    session.add(nrun)
    session.commit()
    session.refresh(nrun)
    return nrun


def send_daily_summaries(session: Session, now: Optional[dt.datetime] = None,
                         router: Optional[PushRouter] = None) -> dict:
    """Create today's summary for every opted-in user whose summary time has
    passed (India time). Trading days only; content from the latest full run."""
    now = now or _now()
    s = settings(session)
    local = now.astimezone(IST)
    today = local.date()
    if not (s["enabled"] and s["event_types"]["DAILY_SUMMARY"]):
        return {"created": 0, "reason": "daily summary disabled"}
    if not is_trading_day(today, masters.get_config(session, "market.holidays")):
        return {"created": 0, "reason": "not a trading day"}
    run = detector.latest_full_run(session)
    if run is None or run.finished_at is None:
        return {"created": 0, "reason": "no completed analysis run"}
    if now - _aware(run.finished_at) > dt.timedelta(hours=96):
        return {"created": 0, "reason": "latest analysis is older than 4 days; summary not sent"}
    active = {c.symbol: c.name for c in session.exec(select(Company).where(Company.active == True)).all()}  # noqa: E712
    rows = [r for r in session.exec(select(StockAnalysisResult).where(
        StockAnalysisResult.run_id == run.run_id, StockAnalysisResult.eligible == True)  # noqa: E712
        .order_by(StockAnalysisResult.rank)).all() if r.symbol in active][:s["daily_summary_size"]]
    if not rows:
        return {"created": 0, "reason": "no eligible candidates in the latest run"}
    listing = "; ".join(f"{r.rank}. {active[r.symbol]} ({_score(r.stockai_score)})" for r in rows)
    analysis_date = _aware(run.finished_at).astimezone(IST).date().isoformat()
    title, body = render(templates(session), "DAILY_SUMMARY", list=listing, date=analysis_date)
    nrun = session.exec(select(NotificationRun).where(NotificationRun.kind == "DAILY_SUMMARY",
                                                      NotificationRun.source_key == today.isoformat())).first()
    if nrun is None:
        nrun = NotificationRun(kind="DAILY_SUMMARY", source_key=today.isoformat(), previous_run_id=run.run_id,
                               engine_version=NOTIFICATION_ENGINE_VERSION, status="COMPLETED", started_at=now)
        session.add(nrun)
        session.commit()
        session.refresh(nrun)
    users, prefs, _, _ = _users_and_context(session)
    created = []
    for u in users:
        pref = prefs.get(u.id)
        if pref is None or not pref.daily_summary or local.time() < _hhmm(pref.daily_summary_time):
            continue
        d = _draft("DAILY_SUMMARY", f"DAILY_SUMMARY:{today.isoformat()}", title, body, route="/top-picks",
                   data={"symbols": [r.symbol for r in rows], "analysis_date": analysis_date})
        created += _store(session, u.id, [d], pref, s, now, nrun, run.run_id, run.engine_version)
    session.commit()
    result = dispatch_pending(session, now, router, notification_ids=[n.id for n in created])
    nrun.notifications_created += len(created)
    nrun.pushes_sent += result["sent"]
    nrun.finished_at = _now()
    session.add(nrun)
    session.commit()
    return {"created": len(created), "sent": result["sent"]}


# ── push dispatch ────────────────────────────────────────────────────────────

def dispatch_pending(session: Session, now: Optional[dt.datetime] = None, router: Optional[PushRouter] = None,
                     notification_ids: Optional[list[int]] = None) -> dict:
    """Push PENDING notifications and quiet-hour notifications whose window
    has ended. Queued pushes older than quiet_queue_max_hours expire (they
    stay in the Notification Center)."""
    now = now or _now()
    router = router or get_router()
    s = settings(session)
    q = select(Notification).where(Notification.push_status.in_(("PENDING", "QUEUED_QUIET_HOURS")))
    if notification_ids is not None:
        if not notification_ids:
            return {"sent": 0, "attempted": 0}
        q = q.where(Notification.id.in_(notification_ids))
    sent = attempted = 0
    for n in session.exec(q.order_by(Notification.id)).all():
        if n.push_status == "QUEUED_QUIET_HOURS":
            if _aware(n.push_after) and _aware(n.push_after) > now:
                continue
            if now - _aware(n.created_at) > dt.timedelta(hours=s["quiet_queue_max_hours"]):
                n.push_status = "EXPIRED"
                session.add(n)
                continue
        if not s["enabled"] or not s["push_enabled"]:
            n.push_status = "DISABLED_GLOBALLY"
            session.add(n)
            continue
        devices = session.exec(select(PushDevice).where(PushDevice.user_id == n.user_id,
                                                        PushDevice.active == True,  # noqa: E712
                                                        PushDevice.push_token.is_not(None),
                                                        PushDevice.permission != "denied")).all()
        if not devices:
            n.push_status = "NO_DEVICE"
            session.add(n)
            continue
        message = PushMessage(title=n.title, body=n.body, data={
            "route": n.route, "notification_id": str(n.id), "type": n.type, "symbol": n.symbol or ""})
        statuses = []
        for dev in devices:
            attempted += 1
            res = router.send(dev.provider, dev.push_token, message)
            statuses.append(res.status)
            session.add(NotificationDelivery(notification_id=n.id, device_id=dev.id, provider=dev.provider,
                                             status=res.status, provider_message_id=res.message_id,
                                             error=res.error, attempted_at=now))
            if res.status == INVALID_TOKEN:
                dev.active, dev.push_token = False, None
                dev.invalidated_at, dev.invalid_reason = now, "provider rejected the token"
                session.add(dev)
        ok = statuses.count(SENT)
        if ok:
            sent += 1
            n.pushed_at = now
            n.push_status = "SENT" if ok == len(statuses) else "PARTIAL"
        elif all(st == NOT_CONFIGURED for st in statuses):
            n.push_status = "PROVIDER_NOT_CONFIGURED"
        else:
            n.push_status = "FAILED"
        session.add(n)
    session.commit()
    return {"sent": sent, "attempted": attempted}


def send_test(session: Session, user: User, router: Optional[PushRouter] = None) -> Notification:
    """A test notification to the user's own devices (admin tool / support)."""
    now = _now()
    title, body = render(templates(session), "TEST")
    n = Notification(user_id=user.id, type="TEST", dedup_key=f"TEST:{now.isoformat()}", title=title, body=body,
                     route="/notifications", engine_version=NOTIFICATION_ENGINE_VERSION, created_at=now)
    session.add(n)
    session.commit()
    session.refresh(n)
    dispatch_pending(session, now, router, notification_ids=[n.id])
    session.refresh(n)
    return n


def admin_stats(session: Session, router: Optional[PushRouter] = None) -> dict[str, Any]:
    by_status = dict(session.exec(select(Notification.push_status, func.count()).group_by(Notification.push_status)).all())
    by_type = dict(session.exec(select(Notification.type, func.count()).group_by(Notification.type)).all())
    devices = dict(session.exec(select(PushDevice.platform, func.count()).where(PushDevice.active == True)  # noqa: E712
                                .group_by(PushDevice.platform)).all())
    return {"engine_version": NOTIFICATION_ENGINE_VERSION, "providers_configured": (router or get_router()).status(),
            "notifications_by_push_status": by_status, "notifications_by_type": by_type,
            "active_devices_by_platform": devices,
            "users_with_preferences": session.exec(select(func.count()).select_from(NotificationPreference)).one()}
