"""
admin_console/app.py — StockAI Pro Admin / Master Control console (Streamlit).

Run:   streamlit run admin_console/app.py
API:   STOCKAI_API_URL (default http://127.0.0.1:8000/api/v1)

Every page reads and writes through the REST API as an ADMIN user
(admin_console/client.py); the backend enforces authorization and writes the
audit log. The console shows stored data only and never invents values: a
missing value is shown as empty, with the backend's stated reason.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from admin_console.client import DEFAULT_API_URL, AdminApiClient, ApiError  # noqa: E402

st.set_page_config(page_title="StockAI Pro - Admin", layout="wide")

PAGES = [
    "Admin Dashboard", "Stock Master", "Sector Master", "Industry Master", "Fundamental Data",
    "Market Data", "Valuation Data", "Technical / Market Signals", "Market Regime", "FQVF Reference",
    "Ranking Configuration", "Top Picks", "Users", "Roles", "Watchlist Administration",
    "Recommendation History", "Data Validation", "Data Health", "API Health", "Engine Runs",
    "Audit Logs", "Engine Versions", "Application Configuration", "Notifications", "User Reports",
]


def client() -> AdminApiClient:
    if "client" not in st.session_state:
        st.session_state.client = AdminApiClient(st.session_state.get("api_url", DEFAULT_API_URL))
    return st.session_state.client


def call(fn, *args, **kwargs):
    """Run an API call; show the backend's error instead of crashing."""
    try:
        return fn(*args, **kwargs)
    except ApiError as e:
        st.error(f"API error {e.status}: {e.message}")
        return None


def table(rows, columns=None, empty="No records."):
    if not rows:
        st.info(empty)
        return
    df = pd.json_normalize(rows) if isinstance(rows, list) else pd.DataFrame(rows)
    if columns:
        df = df[[c for c in columns if c in df.columns]]
    st.dataframe(df, use_container_width=True, hide_index=True)


# ── authentication ───────────────────────────────────────────────────────────

def login_page():
    st.title("StockAI Pro - Admin Console")
    st.caption("Sign in with an account that has the ADMIN role.")
    with st.form("login"):
        api_url = st.text_input("API base URL", value=st.session_state.get("api_url", DEFAULT_API_URL))
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in"):
            st.session_state.api_url = api_url
            st.session_state.client = AdminApiClient(api_url)
            me = call(client().login, email, password)
            if me:
                st.session_state.me = me
                st.rerun()


# ── pages ────────────────────────────────────────────────────────────────────

def page_dashboard(c):
    d = call(c.get, "/admin/dashboard")
    health = call(c.get, "/admin/api-health")
    if d:
        cols = st.columns(4)
        for i, (k, v) in enumerate(d.items()):
            cols[i % 4].metric(k.replace("_", " ").title(), v)
    if health:
        st.subheader(f"System status: {health['status']}")
        st.json(health["checks"])


def page_stock_master(c):
    summary = call(c.get, "/admin/stock-master/summary")
    if summary:
        st.caption("Data status: " + ", ".join(f"{k}: {v}" for k, v in summary.items()))
    search = st.text_input("Search symbol or name")
    rows = call(c.get, "/admin/stocks", search=search or None, limit=200)
    table(rows, ["symbol", "name", "sector", "industry", "exchange", "active", "analysis_enabled", "tradable",
                 "data_status", "data_status_reason", "data_checked_at"])
    st.subheader("Edit stock")
    with st.form("edit_stock"):
        symbol = st.text_input("Symbol (e.g. RELIANCE.NS)")
        col1, col2, col3 = st.columns(3)
        active = col1.selectbox("Active", ["unchanged", True, False])
        analysis = col2.selectbox("Analysis enabled", ["unchanged", True, False])
        tradable = col3.selectbox("Tradable", ["unchanged", True, False])
        name = st.text_input("Company name (blank = unchanged)")
        sector = st.text_input("Sector (blank = unchanged)")
        industry = st.text_input("Industry (blank = unchanged)")
        if st.form_submit_button("Save") and symbol:
            body = {k: v for k, v in {"active": active, "analysis_enabled": analysis, "tradable": tradable}.items()
                    if v != "unchanged"}
            body.update({k: v for k, v in {"name": name, "sector": sector, "industry": industry}.items() if v})
            if call(c.patch, f"/admin/stocks/{symbol.strip().upper()}", body):
                st.success("Saved (audited)")
    st.subheader("Add stock")
    with st.form("add_stock"):
        sym = st.text_input("New symbol")
        nm = st.text_input("Company name")
        if st.form_submit_button("Create") and sym and nm:
            if call(c.post, "/admin/stocks", {"symbol": sym, "name": nm}):
                st.success("Created (audited)")


def page_sectors(c):
    rows = call(c.get, "/admin/sectors")
    table(rows, ["id", "name", "active", "stock_count", "outlook", "outlook_notes", "outlook_updated_at"],
          empty="No sectors yet - they are created from provider classifications during engine runs.")
    st.subheader("Set sector outlook (FQVF check 18)")
    st.caption("POSITIVE = PASS, NEUTRAL = WARNING, NEGATIVE = FAIL. Unset = NOT_AVAILABLE. "
               "Takes effect from the next engine run.")
    if rows:
        with st.form("outlook"):
            names = {r["name"]: r["id"] for r in rows}
            name = st.selectbox("Sector", list(names))
            outlook = st.selectbox("Outlook", ["POSITIVE", "NEUTRAL", "NEGATIVE", "(clear)"])
            notes = st.text_area("Rationale / notes")
            if st.form_submit_button("Save"):
                body = {"clear_outlook": True} if outlook == "(clear)" else {"outlook": outlook, "outlook_notes": notes}
                if call(c.patch, f"/admin/sectors/{names[name]}", body):
                    st.success("Saved (audited)")


def page_industries(c):
    rows = call(c.get, "/admin/industries")
    table(rows, ["id", "name", "sector", "active", "stock_count"])
    if rows:
        with st.form("industry"):
            names = {r["name"]: r["id"] for r in rows}
            name = st.selectbox("Industry", list(names))
            active = st.selectbox("Active", [True, False])
            if st.form_submit_button("Save"):
                if call(c.patch, f"/admin/industries/{names[name]}", {"active": active}):
                    st.success("Saved (audited)")


def page_fundamentals(c):
    symbol = st.text_input("Symbol (blank = latest snapshot of every stock)")
    rows = call(c.get, "/admin/fundamentals", symbol=symbol or None, limit=500)
    if rows:
        flat = [{"symbol": r["symbol"], "fetched_at": r["fetched_at"], "status": r["status"], "error": r["error"],
                 "fiscal_period_end": r["fiscal_period_end"],
                 **{k: (r.get("data") or {}).get(k) for k in ("sector", "industry", "trailing_pe", "price_to_book",
                                                              "price_to_sales", "debt_to_equity", "dividend_yield")},
                 "fiscal_years": len((r.get("data") or {}).get("annual") or []),
                 "issues": len((r.get("data") or {}).get("issues") or [])} for r in rows]
        table(flat)
        if symbol and rows:
            st.subheader("Annual series (latest snapshot)")
            table((rows[0].get("data") or {}).get("annual") or [])
    else:
        st.info("No fundamentals snapshots yet.")


def page_market(c):
    symbol = st.text_input("Symbol (blank = all stocks)")
    rows = call(c.get, "/admin/market-data", symbol=symbol or None, limit=500)
    table(rows, ["symbol", "status", "as_of_date", "close", "volume", "fetched_at", "issues"])


def page_valuation(c):
    data = call(c.get, "/admin/valuation")
    if not data or not data["items"]:
        st.info("No completed engine run yet.")
        return
    flat = []
    for item in data["items"]:
        row = {"symbol": item["symbol"]}
        for ch in item["checks"]:
            row[ch["name"]] = f"{ch['status']}" + (f" ({ch['grade']})" if ch.get("grade") else "")
        flat.append(row)
    st.caption(f"Run {data['run_id']}")
    table(flat)


def page_technical(c):
    rows = call(c.get, "/admin/technical", limit=500)
    table(rows, ["symbol", "as_of_date", "return_20d", "return_60d", "volatility_annual", "max_drawdown_1y",
                 "rsi", "trend_score", "regime", "regime_score", "ml_signal.direction", "ml_signal.probability_up"])
    st.caption("ML signal is informational only (no demonstrated predictive discrimination).")


def page_regime(c):
    rows = call(c.get, "/admin/market-regime")
    table(rows, ["computed_at", "as_of_date", "index", "status", "regime", "regime_score", "reason"])


def page_fqvf(c):
    ref = call(c.get, "/admin/fqvf/reference")
    if ref:
        st.subheader(f"{ref['name']} ({ref['short_name']}) - fixed 18 checks")
        table(ref["checks"])
        st.write("Statuses", ref["statuses"])
        st.write("Score", ref["score"])
        st.subheader("Thresholds (read-only; the framework is fixed)")
        table([{"threshold": k, "value": v} for k, v in ref["thresholds"].items()])


def page_ranking(c):
    cfg = call(c.get, "/admin/ranking/config")
    if not cfg:
        return
    table(cfg["components"])
    st.write("Eligibility rules", cfg["rules"])
    st.subheader("Edit weights")
    st.caption("Weights are relative (renormalised over available components). Changes apply from the next run "
               "and are audited. ml_signal defaults to 0: the classifier showed no out-of-sample skill.")
    with st.form("weights"):
        new = {comp["key"]: st.number_input(comp["label"], min_value=0.0, max_value=100.0,
                                            value=float(comp["weight"]), step=1.0) for comp in cfg["components"]}
        if st.form_submit_button("Save weights"):
            if call(c.put, "/admin/config/ranking.weights", {"value": new}):
                st.success("Saved (audited)")


def page_top_picks(c):
    eligible_only = st.checkbox("Eligible only", value=False)
    data = call(c.get, "/admin/analysis-results", eligible=True if eligible_only else None, limit=1000)
    if not data or not data.get("run"):
        st.info("No completed engine run yet.")
        return
    st.caption(f"Run {data['run']['run_id']} - {data['run']['status']} - finished {data['run']['finished_at']}")
    table(data["items"], ["rank", "symbol", "stockai_score", "score_coverage", "fqvf_score", "fqvf_summary",
                          "eligible", "ineligible_reasons"])


def page_users(c):
    rows = call(c.get, "/admin/users", limit=200)
    table(rows)
    with st.form("user_active"):
        uid = st.number_input("User id", min_value=1, step=1)
        action = st.selectbox("Action", ["activate", "deactivate"])
        if st.form_submit_button("Apply"):
            if call(c.post, f"/admin/users/{int(uid)}/{action}"):
                st.success("Done (audited)")


def page_roles(c):
    roles = call(c.get, "/admin/roles") or []
    table(roles)
    with st.form("roles"):
        uid = st.number_input("User id", min_value=1, step=1)
        role = st.selectbox("Role", [r["name"] for r in roles] or ["USER", "ADMIN"])
        action = st.selectbox("Action", ["assign", "remove"])
        if st.form_submit_button("Apply"):
            if action == "assign":
                ok = call(c.post, f"/admin/users/{int(uid)}/roles", {"role": role})
            else:
                ok = call(c.delete, f"/admin/users/{int(uid)}/roles/{role}")
            if ok is not None:
                st.success("Done (audited)")


def page_watchlist(c):
    table(call(c.get, "/admin/watchlist", limit=500))
    st.caption("Read-only: administrators cannot edit users' watchlists.")


def page_recommendations(c):
    symbol = st.text_input("Symbol filter")
    table(call(c.get, "/admin/recommendations", symbol=symbol or None, limit=500))
    st.caption("Rows with engine_version NULL or v1.0 predate the temporal-integrity fix and must not be "
               "used to evaluate the current engine.")


def page_validation(c):
    d = call(c.get, "/admin/data-health")
    if not d:
        return
    rows = [{"category": cat, **f} for cat, items in d["findings"].items() for f in items]
    category = st.selectbox("Category", ["(all)"] + list(d["findings"]))
    table([r for r in rows if category == "(all)" or r["category"] == category], empty="No findings.")


def page_health(c):
    d = call(c.get, "/admin/data-health")
    if not d:
        return
    st.caption(f"{d['scope']} - {d['stocks_in_scope']} stocks - generated {d['generated_at']}")
    cols = st.columns(4)
    for i, (k, v) in enumerate(d["summary"].items()):
        cols[i % 4].metric(k.replace("_", " "), v)
    st.write("Never checked:", d["never_checked"])
    st.write("News timestamps:", d["news_timestamps"])
    st.write("Latest run:", d["latest_run"])


def page_api_health(c):
    h = call(c.get, "/admin/api-health")
    if h:
        st.subheader(h["status"])
        st.json(h)


def page_runs(c):
    with st.form("start_run"):
        st.write("Start a ranking run (runs in the background; one at a time).")
        symbols = st.text_input("Symbols (comma-separated; blank = Large/Mid/Small Cap universe)")
        limit = st.number_input("Limit (0 = no limit)", min_value=0, step=10)
        include_ml = st.checkbox("Compute informational ML signal", value=True)
        refresh = st.checkbox("Re-fetch fundamentals even if fresh", value=False)
        if st.form_submit_button("Start run"):
            body = {"symbols": [s.strip() for s in symbols.split(",") if s.strip()] or None,
                    "limit": int(limit) or None, "include_ml": include_ml, "refresh_fundamentals": refresh}
            r = call(c.post, "/admin/engine-runs", body)
            if r:
                st.success(f"Started {r['run_id']}")
    runs = call(c.get, "/admin/engine-runs")
    table(runs, ["run_id", "kind", "status", "started_at", "finished_at", "total", "processed", "succeeded",
                 "skipped", "failed", "error_count", "engine_version", "fqvf_version"])
    run_id = st.text_input("Run id for details")
    if run_id:
        detail = call(c.get, f"/admin/engine-runs/{run_id}")
        if detail:
            table(detail.get("errors") or [], empty="No errors.")
            st.json(detail.get("config"))


def page_audit(c):
    table(call(c.get, "/admin/audit-logs", limit=500))


def page_versions(c):
    v = call(c.get, "/admin/engine-versions")
    if v:
        st.json(v)


def page_config(c):
    rows = call(c.get, "/admin/config") or []
    table(rows, ["key", "value", "is_default", "updated_at", "updated_by", "description"])
    with st.form("config"):
        key = st.selectbox("Key", [r["key"] for r in rows])
        current = next((r["value"] for r in rows if r["key"] == key), None)
        raw = st.text_area("Value (JSON)", value=json.dumps(current, indent=2))
        if st.form_submit_button("Save"):
            try:
                value = json.loads(raw)
            except ValueError as e:
                st.error(f"Invalid JSON: {e}")
            else:
                if call(c.put, f"/admin/config/{key}", {"value": value}):
                    st.success("Saved (audited)")


def page_notifications(c):
    stats = call(c.get, "/admin/notifications/stats")
    if stats:
        st.subheader("Status")
        settings = stats.pop("settings", {})
        st.json(stats)
        st.subheader("Settings (audited)")
        st.caption("Global switch, event types, thresholds, rate limits and quiet-hour defaults. "
                   "Templates: key notifications.templates on the Application Configuration page.")
        with st.form("notification_settings"):
            enabled = st.checkbox("Notifications enabled (global emergency switch)", value=settings.get("enabled", True))
            push_enabled = st.checkbox("Push delivery enabled", value=settings.get("push_enabled", True))
            types = {t: st.checkbox(f"Event type {t}", value=v) for t, v in settings.get("event_types", {}).items()}
            score = st.number_input("Score change threshold (points)", 1.0, 100.0,
                                    float(settings.get("score_change_threshold", 10)))
            rank = st.number_input("Rank change threshold (places)", 1, 500, int(settings.get("rank_change_threshold", 15)))
            fqvf = st.number_input("FQVF change (checks passed)", 1, 18, int(settings.get("fqvf_change_min_checks", 2)))
            per_run = st.number_input("Max pushes per user per run", 0, 50, int(settings.get("max_push_per_user_per_run", 5)))
            per_day = st.number_input("Max pushes per user per day", 0, 100, int(settings.get("max_push_per_user_per_day", 10)))
            cooldown = st.number_input("Per-stock cooldown (hours)", 0, 336, int(settings.get("stock_cooldown_hours", 20)))
            if st.form_submit_button("Save settings"):
                body = {"enabled": enabled, "push_enabled": push_enabled, "event_types": types,
                        "score_change_threshold": score, "rank_change_threshold": int(rank),
                        "fqvf_change_min_checks": int(fqvf), "max_push_per_user_per_run": int(per_run),
                        "max_push_per_user_per_day": int(per_day), "stock_cooldown_hours": int(cooldown)}
                if call(c.put, "/admin/config/notifications.settings", {"value": {**settings, **body}}):
                    st.success("Saved (audited)")
    st.subheader("Actions (audited)")
    cols = st.columns(4)
    if cols[0].button("Process latest ranking run"):
        r = call(c.post, "/admin/notifications/process-run", {})
        if r:
            st.success(f"{r['status']}: {r['notifications_created']} notification(s), {r['pushes_sent']} pushed")
    if cols[1].button("Send daily summaries now"):
        r = call(c.post, "/admin/notifications/daily-summary")
        if r is not None:
            st.success(str(r))
    if cols[2].button("Dispatch queued pushes"):
        r = call(c.post, "/admin/notifications/dispatch")
        if r is not None:
            st.success(str(r))
    if cols[3].button("Send test to my devices"):
        r = call(c.post, "/admin/notifications/test")
        if r:
            st.info(f"Push status: {r['push_status']}")
    st.subheader("Notification runs")
    table(call(c.get, "/admin/notifications/runs"), ["id", "kind", "source_key", "previous_run_id", "status",
                                                      "events_detected", "notifications_created", "pushes_sent",
                                                      "pushes_suppressed", "started_at", "engine_version"])
    st.subheader("Recent notifications")
    table(call(c.get, "/admin/notifications/recent", limit=200),
          ["id", "user_id", "type", "title", "symbol", "push_status", "read", "created_at"])


def page_reports(c):
    status = st.selectbox("Status", ["", "NEW", "IN_REVIEW", "RESOLVED", "REJECTED"])
    rows = call(c.get, "/admin/feedback", **({"status": status} if status else {})) or []
    table(rows, ["id", "created_at", "user_id", "category", "symbol", "message", "status", "admin_note",
                 "platform", "app_version"], empty="No reports.")
    with st.form("report_update"):
        rid = st.number_input("Report id", min_value=0, step=1)
        new_status = st.selectbox("New status", ["IN_REVIEW", "RESOLVED", "REJECTED", "NEW"])
        note = st.text_area("Admin note")
        if st.form_submit_button("Update (audited)") and rid:
            if call(c.patch, f"/admin/feedback/{int(rid)}", {"status": new_status, "admin_note": note or None}):
                st.success("Updated")


HANDLERS = {
    "Admin Dashboard": page_dashboard, "Stock Master": page_stock_master, "Sector Master": page_sectors,
    "Industry Master": page_industries, "Fundamental Data": page_fundamentals, "Market Data": page_market,
    "Valuation Data": page_valuation, "Technical / Market Signals": page_technical, "Market Regime": page_regime,
    "FQVF Reference": page_fqvf, "Ranking Configuration": page_ranking, "Top Picks": page_top_picks,
    "Users": page_users, "Roles": page_roles, "Watchlist Administration": page_watchlist,
    "Recommendation History": page_recommendations, "Data Validation": page_validation,
    "Data Health": page_health, "API Health": page_api_health, "Engine Runs": page_runs,
    "Audit Logs": page_audit, "Engine Versions": page_versions, "Application Configuration": page_config,
    "Notifications": page_notifications, "User Reports": page_reports,
}
assert list(HANDLERS) == PAGES


def main():
    if "me" not in st.session_state:
        login_page()
        return
    c = client()
    with st.sidebar:
        st.write(f"Signed in as **{st.session_state.me.get('email') or st.session_state.me.get('id')}**")
        page = st.radio("Admin pages", PAGES)
        if st.button("Sign out"):
            c.logout()
            for k in ("me", "client"):
                st.session_state.pop(k, None)
            st.rerun()
    st.title(page)
    HANDLERS[page](c)


main()
