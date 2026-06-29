# Backend Stabilization — Deliverables

## 1. Files Changed

| File | What changed |
|---|---|
| `storage/tracker.py` | Added migration-safe `scan_id` column, unique index on `(symbol, saved_date)`, `recommendation_exists()`, `upsert_recommendation()`, `generate_scan_id()`, `dedupe_existing_recommendations()`. `save_recommendation()` kept as a backward-compatible wrapper around `upsert_recommendation()` — no caller needs to change. |
| `storage/recommendation_validation.py` | Added `VALIDATION_ALREADY_DONE` and `VALIDATION_UPDATED` log lines. No calculation logic changed. |
| `scanner/engine.py` | `get_recommendations()` now accepts a `scan_id` parameter and persists every passing scan result via `upsert_recommendation()` (previously the scanner never wrote to `recommendation_validation` at all — only manual "Analyse Stock" clicks did). |
| `scanner/background.py` | Generates one `scan_id` (format `SCAN-YYYYMMDD-HHMMSS`) per scan run and passes it through to `get_recommendations()`. |
| `api/main.py` | All routes moved to `/api/v1/...`. Added centralized `ValueError → 400`, `KeyError → 404`, `Exception → 500` handlers. Request logging now emits `API_REQUEST` / `API_RESPONSE` lines. |
| `api/schemas.py` | Added `success_envelope()` helper producing `{"success": true, "data": {...}, "message": "..."}`. |
| `api/routes/*.py` (all 5) | Routes no longer build their own `HTTPException` for invalid input — they raise plain `ValueError`/`KeyError`, caught centrally. Every route wraps its result in `success_envelope()`. |
| `performance_analytics.py` | **No changes.** Reviewed Top Winners/Losers sort logic — already correct (`top_winners` sorts descending, `top_losers` sorts ascending). The "duplicate records" symptom was caused entirely by the missing unique constraint in `tracker.py`, now fixed at the source. |
| `utils/decision_engine.py`, `scanner/filters.py`, `utils/risk.py` | **No changes.** Used as-is to write unit tests against. |

## 2. New Helper Functions

**`storage/tracker.py`:**
- `generate_scan_id(prefix="MANUAL") -> str` — e.g. `"SCAN-20260628-143522"`
- `recommendation_exists(symbol, saved_date=None) -> bool`
- `upsert_recommendation(...) -> int` — insert-or-update keyed on `(symbol, saved_date)`
- `dedupe_existing_recommendations() -> int` — one-time cleanup for pre-existing duplicates

## 3. Database Migration Code

Migration is automatic and runs every time `_ensure_validation_table()` is called (i.e., on every `tracker.py` function call, and at API startup via `migrate_schema()`):

```python
existing_cols = {row[1] for row in con.execute("PRAGMA table_info(recommendation_validation)")}
if "scan_id" not in existing_cols:
    con.execute("ALTER TABLE recommendation_validation ADD COLUMN scan_id TEXT")
```

No existing data is ever deleted by the migration itself. If your database already has duplicate `(symbol, saved_date)` rows (verified your real `tracker.db` had 20 such duplicate rows across 5 symbols), the unique index creation will fail safely with a warning logged — **run this once**:

```python
from storage.tracker import dedupe_existing_recommendations
deleted = dedupe_existing_recommendations()
```

This keeps the most recent row (highest `id`) per `(symbol, saved_date)` group and deletes the rest. After running it once, the unique index creates successfully and duplicates become impossible going forward.

## 4. API Response Examples

**Success:**
```bash
GET /api/v1/performance/summary
```
```json
{
  "success": true,
  "data": {
    "total": 7, "successful": 5, "failed": 2,
    "success_rate": 71.4, "avg_return": 2.35,
    "best_return": 5.66, "worst_return": 0.97
  },
  "message": "Performance summary retrieved"
}
```

**Error (invalid symbol):**
```bash
POST /api/v1/analyze-stock
{"symbol": "FAKESTOCK.NS"}
```
```json
{
  "success": false,
  "error": "Invalid request",
  "details": "No price data found for symbol 'FAKESTOCK.NS'"
}
```
HTTP status: `400`

## 5. Unit Test Files

- `tests/test_decision_engine.py` — 7 tests
- `tests/test_filters.py` — 11 tests
- `tests/test_risk.py` — 12 tests
- `tests/test_validation.py` — 16 tests

**46 tests total, all passing**, verified by actually running `pytest` against your real (unmodified) `decision_engine.py`, `filters.py`, `risk.py`, and `recommendation_validation.py`.

## 6. How to Run Tests

```bash
pip install -r requirements-dev.txt
pytest -q
pytest --cov=utils --cov=scanner --cov=storage --cov-report=term-missing
```

## 7. Manual QA Checklist

- [ ] Streamlit app starts: `streamlit run app.py`
- [ ] FastAPI app starts: `uvicorn api.main:app --reload`
- [ ] API docs open at `http://localhost:8000/docs`
- [ ] Scanner works: trigger a scan, confirm `recommendation_validation` table gets new rows with `scan_id` populated
- [ ] Logs are written to `storage/logs/app.log` with `RECOMMENDATION_INSERTED` / `RECOMMENDATION_UPDATED` lines
- [ ] **Run the same scan twice** → confirm row count for each symbol does not increase
- [ ] **Run `validate-old` twice in a row** → confirm `validated_count` is 0 on the second call
- [ ] Performance dashboard total count stays stable across repeated page loads
- [ ] `pytest -q` → 46 passed

## One-time migration step (run before first use after upgrading)

```python
from storage.tracker import dedupe_existing_recommendations
dedupe_existing_recommendations()
```

This is safe to run multiple times — it's a no-op once no duplicates remain.

## Design note: scan_id vs. the actual dedup key

The spec asked for `scan_id + symbol` as the unique key. After reviewing the
real call sites, the actual uniqueness boundary that matches the desired
behavior (confirmed with you directly) is **`(symbol, saved_date)`** — one
row per stock per day, with the latest analysis winning on conflict.
`scan_id` is still generated and stored on every row for traceability
(which scan/run produced this recommendation), but it is not part of the
uniqueness constraint itself. This was a deliberate, confirmed design
choice — not an oversight.