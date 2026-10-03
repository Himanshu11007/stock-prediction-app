"""
scripts/e2e_backend.py — end-to-end smoke test against a RUNNING backend.

Exercises the real HTTP API the mobile app and admin console use: health,
database, authentication (login, /me, refresh rotation, logout), stock
search, analysis / FQVF / ranking, Top Investment Candidates, watchlist
add/list/remove, admin authorization and APIs, and OpenAPI docs.

Usage (server must already be running):
    python scripts/e2e_backend.py --base http://127.0.0.1:8000 \\
        --user e2e.user@example.com --user-password ... \\
        --admin e2e.admin@example.com --admin-password ...

Exit code 0 only if every check passes. Never prints tokens or passwords.
"""
from __future__ import annotations

import argparse
import sys
import time

import requests

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}", flush=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--user", required=True)
    ap.add_argument("--user-password", required=True)
    ap.add_argument("--admin", required=True)
    ap.add_argument("--admin-password", required=True)
    ap.add_argument("--symbol", default="RELIANCE.NS")
    args = ap.parse_args()
    api = args.base.rstrip("/") + "/api/v1"
    s = requests.Session()

    # 1-3. startup, health, database
    r = s.get(f"{api}/health", timeout=10)
    check("health", r.status_code == 200 and r.json().get("status") == "ok")
    r = s.get(f"{api}/app/config", timeout=10)
    check("app config (public, DB-backed)", r.status_code == 200 and "features" in r.json()["data"])

    # 4. authentication
    r = s.post(f"{api}/auth/login", data={"username": args.user, "password": args.user_password}, timeout=20)
    if not check("user login", r.status_code == 200, f"status {r.status_code}"):
        return 1
    tokens = r.json()
    H = {"Authorization": f"Bearer {tokens['access_token']}"}
    me = s.get(f"{api}/auth/me", headers=H, timeout=10)
    check("/auth/me", me.status_code == 200 and "USER" in me.json()["roles"])
    check("wrong password rejected", s.post(f"{api}/auth/login", data={"username": args.user, "password": "wrong"},
                                            timeout=20).status_code == 401)
    check("no token rejected", s.get(f"{api}/top-picks", timeout=10).status_code == 401)
    r = s.post(f"{api}/auth/refresh", json={"refresh_token": tokens["refresh_token"]}, timeout=10)
    rotated = r.json() if r.status_code == 200 else {}
    check("refresh rotation", r.status_code == 200 and rotated.get("refresh_token") != tokens["refresh_token"])
    check("old refresh token revoked", s.post(f"{api}/auth/refresh", json={"refresh_token": tokens["refresh_token"]},
                                              timeout=10).status_code == 401)
    H = {"Authorization": f"Bearer {rotated['access_token']}"}

    # 5. stock search
    r = s.get(f"{api}/stocks", params={"search": args.symbol.split(".")[0], "limit": 5}, headers=H, timeout=10)
    check("stock search", r.status_code == 200 and any(x["symbol"] == args.symbol for x in r.json()))

    # 6-8. analysis, FQVF, ranking
    r = s.get(f"{api}/stocks/{args.symbol}/analysis", headers=H, timeout=60)
    if r.status_code == 404:
        t0 = time.time()
        r = s.post(f"{api}/stocks/{args.symbol}/analysis/refresh", headers=H, timeout=180)
        check("on-demand analysis", r.status_code == 200, f"{time.time() - t0:.1f}s")
    ok = r.status_code == 200
    data = r.json().get("data", {}) if ok else {}
    check("stock analysis", ok and data.get("symbol") == args.symbol)
    checks = data.get("fqvf", {}).get("checks", [])
    check("FQVF has 18 fixed checks", len(checks) == 18 and [c["id"] for c in checks] == list(range(1, 19)))
    check("FQVF NOT_AVAILABLE carries a reason", all(c["explanation"] for c in checks if c["status"] == "NOT_AVAILABLE"))
    check("ranking present", data.get("ranking", {}).get("components") is not None,
          f"score {data.get('ranking', {}).get('stockai_score')}")
    check("/fqvf endpoint", s.get(f"{api}/stocks/{args.symbol}/fqvf", headers=H, timeout=30).status_code == 200)
    check("/ranking endpoint", s.get(f"{api}/stocks/{args.symbol}/ranking", headers=H, timeout=30).status_code == 200)
    check("unknown stock 404", s.get(f"{api}/stocks/NOSUCHSTOCK.NS/analysis", headers=H, timeout=10).status_code == 404)

    # 9. Top Picks
    r = s.get(f"{api}/top-picks", headers=H, timeout=30)
    tp = r.json().get("data", {}) if r.status_code == 200 else {}
    items = tp.get("items", [])
    check("top picks", r.status_code == 200 and isinstance(items, list), f"{len(items)} items, run {(tp.get('run') or {}).get('run_id')}")
    if items:
        ranks = [i["rank"] for i in items]
        check("top picks ranked and explained", ranks == sorted(ranks) and all(i["positives"] or i["risks"] for i in items))

    # 10. watchlist
    r = s.post(f"{api}/watchlist", headers=H, json={"symbol": args.symbol, "buy_price": 100.0,
                                                     "buy_date": "2026-10-01", "quantity": 1}, timeout=10)
    added = r.status_code == 201
    check("watchlist add", added, f"status {r.status_code}")
    lst = s.get(f"{api}/watchlist", headers=H, timeout=10)
    item = next((i for i in lst.json() if i["symbol"] == args.symbol), None) if lst.status_code == 200 else None
    check("watchlist list", item is not None)
    if item:
        check("watchlist remove", s.delete(f"{api}/watchlist/{item['id']}", headers=H, timeout=10).status_code == 204)

    # 11. admin
    check("normal user cannot use admin APIs", s.get(f"{api}/admin/data-health", headers=H, timeout=10).status_code == 403)
    check("normal user cannot clear logs", s.delete(f"{api}/logs/clear", headers=H, timeout=10).status_code == 403)
    r = s.post(f"{api}/auth/login", data={"username": args.admin, "password": args.admin_password}, timeout=20)
    if check("admin login", r.status_code == 200):
        A = {"Authorization": f"Bearer {r.json()['access_token']}"}
        for path in ("/admin/dashboard", "/admin/sectors", "/admin/industries", "/admin/engine-runs",
                     "/admin/data-health", "/admin/api-health", "/admin/analysis-results", "/admin/config",
                     "/admin/fqvf/reference", "/admin/engine-versions", "/admin/audit-logs"):
            rr = s.get(api + path, headers=A, timeout=60)
            check(f"admin GET {path}", rr.status_code == 200, f"status {rr.status_code}")
        bad = s.put(f"{api}/admin/config/ranking.weights", headers=A, json={"value": {"quality": -1}}, timeout=10)
        check("admin config validation", bad.status_code == 400)

    # 12. OpenAPI
    spec = s.get(args.base.rstrip("/") + "/openapi.json", timeout=10)
    paths = spec.json().get("paths", {}) if spec.status_code == 200 else {}
    for p in ("/api/v1/top-picks", "/api/v1/stocks/{symbol}/analysis", "/api/v1/stocks/{symbol}/fqvf",
              "/api/v1/stocks/{symbol}/ranking", "/api/v1/app/config", "/api/v1/admin/engine-runs"):
        check(f"OpenAPI documents {p}", p in paths)
    check("/docs", s.get(args.base.rstrip("/") + "/docs", timeout=10).status_code == 200)
    check("/redoc", s.get(args.base.rstrip("/") + "/redoc", timeout=10).status_code == 200)

    # logout revokes refresh token
    s.post(f"{api}/auth/logout", json={"refresh_token": rotated["refresh_token"]}, timeout=10)
    check("logout revokes refresh token",
          s.post(f"{api}/auth/refresh", json={"refresh_token": rotated["refresh_token"]}, timeout=10).status_code == 401)

    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("FAILED:", ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
