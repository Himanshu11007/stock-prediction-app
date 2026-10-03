"""
notifications/settings.py — administrator-controlled notification settings
and templates (app_config keys "notifications.settings" and
"notifications.templates"; changed through PUT /admin/config/{key}, which
writes the audit log).

Defaults are deliberately conservative: material thresholds, aggregation of
multi-stock events, per-run and per-day push caps, a per-stock cooldown and
quiet hours. See docs/NOTIFICATIONS.md.
"""
from __future__ import annotations

import re
import string
from typing import Any

EVENT_TYPES = ("NEW_TOP_CANDIDATE", "TOP_CANDIDATE_REMOVED", "SCORE_CHANGE", "FQVF_CHANGE",
               "WATCHLIST_ALERT", "DAILY_SUMMARY", "MARKET_REGIME_CHANGE")

DEFAULT_SETTINGS: dict[str, Any] = {
    "enabled": True,                         # global emergency switch (False = nothing is created or pushed)
    "push_enabled": True,                    # False = Notification Center only, no push
    "event_types": {t: True for t in EVENT_TYPES},
    "score_change_threshold": 10.0,          # StockLens Score points (0-100 scale)
    "rank_change_threshold": 15,             # places, watchlist alerts
    "fqvf_change_min_checks": 2,             # change in FQVF checks passed
    "regime_min_score_change": 0.3,          # regime score is -1..+1
    "regime_cooldown_hours": 72,
    "aggregate_above": 3,                    # >N stocks entering/leaving -> one combined notification
    "max_push_per_user_per_run": 5,
    "max_push_per_user_per_day": 10,
    "stock_cooldown_hours": 20,              # same user + stock + type: at most one push per window
    "quiet_queue_max_hours": 12,             # pushes queued in quiet hours expire after this
    "daily_summary_size": 5,
    "default_quiet_hours": {"enabled": True, "start": "22:00", "end": "07:00"},
    "default_daily_summary_time": "08:30",
}

DEFAULT_TEMPLATES: dict[str, dict[str, str]] = {
    "NEW_TOP_CANDIDATE": {
        "title": "New StockLens Top Candidate",
        "body": "{name} has entered the Top Investment Candidates at rank {rank}. "
                "StockLens Score: {score}/100. Tap to view analysis."},
    "NEW_TOP_CANDIDATE_MULTI": {
        "title": "New StockLens Top Candidates",
        "body": "{count} stocks entered the Top Investment Candidates: {list}. Tap to view."},
    "TOP_CANDIDATE_REMOVED": {
        "title": "StockLens Candidate Update",
        "body": "{name} is no longer in the Top Investment Candidates. Reason: {reason}"},
    "TOP_CANDIDATE_REMOVED_MULTI": {
        "title": "StockLens Candidate Update",
        "body": "{count} stocks left the Top Investment Candidates: {list}. Tap to review."},
    "SCORE_CHANGE": {
        "title": "StockLens Score Update",
        "body": "{name}'s StockLens Score changed from {old_score} to {new_score} ({delta}). Tap to review."},
    "SCORE_CHANGE_MULTI": {
        "title": "StockLens Score Updates",
        "body": "{count} candidates changed materially: {list}. Tap to review."},
    "FQVF_CHANGE": {
        "title": "FQVF Update",
        "body": "{name}'s FQVF changed from {old_passes}/18 to {new_passes}/18 checks passed. Tap to review."},
    "FQVF_CHANGE_MULTI": {
        "title": "FQVF Updates",
        "body": "{count} candidates had FQVF changes: {list}. Tap to review."},
    "WATCHLIST_ALERT": {
        "title": "Watchlist: {name}",
        "body": "{changes}. Tap to view analysis."},
    "DAILY_SUMMARY": {
        "title": "Today's StockLens Top Candidates",
        "body": "Analysis of {date}: {list}. Tap to view today's analysis."},
    "MARKET_REGIME_CHANGE": {
        "title": "Market Regime Update",
        "body": "The NIFTY 50 regime changed from {old_regime} to {new_regime}. Tap for details."},
    "TEST": {
        "title": "StockLens test notification",
        "body": "Push notifications are working on this device."},
}

TEMPLATE_FIELDS = {"name", "symbol", "score", "rank", "old_score", "new_score", "delta", "reason", "old_passes",
                   "new_passes", "changes", "list", "date", "old_regime", "new_regime", "count"}

_HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


def is_hhmm(value: Any) -> bool:
    return isinstance(value, str) and bool(_HHMM.match(value))


def validate_settings(value: Any) -> dict:
    if not isinstance(value, dict):
        raise ValueError("notifications.settings must be an object")
    unknown = set(value) - set(DEFAULT_SETTINGS)
    if unknown:
        raise ValueError(f"Unknown notification settings: {sorted(unknown)}")
    merged = {**DEFAULT_SETTINGS, **value}
    for k in ("enabled", "push_enabled"):
        if not isinstance(merged[k], bool):
            raise ValueError(f"{k} must be true or false")
    et = merged["event_types"]
    if not isinstance(et, dict) or set(et) - set(EVENT_TYPES) or not all(isinstance(v, bool) for v in et.values()):
        raise ValueError(f"event_types must map {list(EVENT_TYPES)} to booleans")
    merged["event_types"] = {**DEFAULT_SETTINGS["event_types"], **et}
    bounds = {"score_change_threshold": (1, 100), "rank_change_threshold": (1, 500), "fqvf_change_min_checks": (1, 18),
              "regime_min_score_change": (0, 2), "regime_cooldown_hours": (0, 24 * 30), "aggregate_above": (1, 50),
              "max_push_per_user_per_run": (0, 50), "max_push_per_user_per_day": (0, 100),
              "stock_cooldown_hours": (0, 24 * 14), "quiet_queue_max_hours": (0, 48), "daily_summary_size": (1, 20)}
    for k, (lo, hi) in bounds.items():
        v = merged[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
            raise ValueError(f"{k} must be a number between {lo} and {hi}")
    q = merged["default_quiet_hours"]
    if not isinstance(q, dict) or set(q) != {"enabled", "start", "end"} or not isinstance(q["enabled"], bool) \
            or not is_hhmm(q["start"]) or not is_hhmm(q["end"]):
        raise ValueError('default_quiet_hours must be {"enabled": bool, "start": "HH:MM", "end": "HH:MM"}')
    if not is_hhmm(merged["default_daily_summary_time"]):
        raise ValueError("default_daily_summary_time must be HH:MM")
    return merged


def validate_templates(value: Any) -> dict:
    if not isinstance(value, dict):
        raise ValueError("notifications.templates must be an object")
    unknown = set(value) - set(DEFAULT_TEMPLATES)
    if unknown:
        raise ValueError(f"Unknown notification templates: {sorted(unknown)}")
    merged = {k: dict(v) for k, v in DEFAULT_TEMPLATES.items()}
    for key, tpl in value.items():
        if not isinstance(tpl, dict) or set(tpl) - {"title", "body"}:
            raise ValueError(f"template {key} must be an object with 'title' and/or 'body'")
        for part, text in tpl.items():
            if not isinstance(text, str) or not text.strip() or len(text) > 300:
                raise ValueError(f"template {key}.{part} must be a non-empty string of at most 300 characters")
            fields = {f for _, f, _, _ in string.Formatter().parse(text) if f is not None}
            if fields - TEMPLATE_FIELDS:
                raise ValueError(f"template {key}.{part} uses unknown fields {sorted(fields - TEMPLATE_FIELDS)}; "
                                 f"allowed: {sorted(TEMPLATE_FIELDS)}")
            merged[key][part] = text
    return merged


class _Blank(dict):
    def __missing__(self, key):
        return "-"


def render(templates: dict, key: str, **fields) -> tuple[str, str]:
    tpl = templates.get(key) or DEFAULT_TEMPLATES[key]
    values = _Blank({k: ("-" if v is None else v) for k, v in fields.items()})
    return (string.Formatter().vformat(tpl["title"], (), values)[:120],
            string.Formatter().vformat(tpl["body"], (), values)[:500])
