# Fundamental Quality & Value Framework (FQVF)

Implementation: `fqvf/service.py` (`FQVFInputs`, `FQVFCheck`, `FQVFResult`,
`evaluate`). Version: `fqvf-v1.0` (`config.FQVF_ENGINE_VERSION`).
Tests: `tests/test_fqvf.py`. Reference API: `GET /api/v1/fqvf/reference`.

FQVF is a **fixed 18-check** evaluation framework. The checks, their order and
their meaning are defined once, in the backend. The mobile app and admin
console only display results; nothing recomputes them.

## Statuses

| Status | Meaning |
|---|---|
| `PASS` | Meets the preferred threshold. |
| `WARNING` | Tolerable / acceptable band, or a mixed result. `grade` names the band (e.g. `TOLERABLE`). |
| `FAIL` | Does not meet the threshold, or a business fact disqualifies it (e.g. negative earnings for PE). |
| `NOT_AVAILABLE` | The data needed is missing. **This is not a failure**; the explanation states what is missing. |

Every check returns: `id`, `name`, `value`, `threshold`, `status`, `grade`,
`explanation`, `source`, `source_timestamp`, `data_available`, `scored`, plus
the result's `calculated_at` and `version`.

## The 18 checks

Data sources: *summary* = provider summary fields (Yahoo Finance `info`);
*statements* = annual income statement, balance sheet and cash flow (Yahoo
Finance, typically 4 fiscal years for NSE stocks); *price* = daily adjusted
close; *Sector Master* = administrator input. "[spec]" marks thresholds given
by the product specification; others are documented project defaults.

| # | Check | Value / formula | PASS | WARNING | FAIL | NOT_AVAILABLE when |
|---|---|---|---|---|---|---|
| 1 | Stable Stock / Business Stability | Net income per fiscal year (statements) | profit in every year (≥ 3 years) | exactly one loss year, not the latest | latest year a loss, or ≥ 2 loss years | < 3 years of net income |
| 2 | 5-Year EPS Progression | Diluted EPS per fiscal year | no year-on-year decline | one decline and EPS above the first year | ≥ 2 declines | < 4 years of EPS, or EPS not comparable across years (split/bonus detected) |
| 3 | Industry Classification *(informational)* | sector / industry (summary) | both known | — | — | either missing |
| 4 | Current Market Price (CMP) *(informational)* | last daily close | valid and ≤ 4 days old | stale (> 4 days) | — | no valid price |
| 5 | EPS Growth | EPS CAGR = (last / first)^(1/years) − 1 | ≥ 10% | 0–10% | < 0, or latest EPS negative | < 3 EPS years; first EPS ≤ 0 (CAGR undefined); split/bonus detected |
| 6 | PE Ratio | trailing PE (summary) | 0 < PE ≤ 25 | 25 < PE ≤ 40 | PE > 40, PE ≤ 0, or trailing EPS ≤ 0 | not reported |
| 7 | Industry PE | PE vs **median trailing PE of other analysed stocks in the same industry** (stock itself excluded; snapshots ≤ 7 days old) | PE ≤ median | up to 20% above | > 20% above | < 3 peers with positive PE, or no positive PE |
| 8 | Historical Average PE (5–7 years) | PE vs own average of year-end price / annual EPS | PE ≤ average | up to 20% above | > 20% above | < 5 fiscal years [spec] — **usually not available** (provider supplies ~4 years) |
| 9 | Intrinsic Value | Graham number √(22.5 × EPS × BVPS) vs CMP | CMP ≤ value | up to 20% above | > 20% above, or EPS/BVPS ≤ 0 (undefined) | EPS, BVPS or price missing |
| 10 | PEG Ratio | PE ÷ (EPS CAGR × 100); provider PEG only if CAGR unavailable and no split/bonus | ≤ 1 [spec] | ≤ 2 | > 2, or growth ≤ 0 | no positive PE or growth rate |
| 11 | Price-to-Book (PB) | summary | < 1 `PREFERRED` [spec] | 1–3 `TOLERABLE` [spec] | > 3 or ≤ 0 | not reported |
| 12 | Return on Equity (ROE) | net income ÷ shareholders' equity (latest FY) | > 15% [spec] | — | ≤ 15%, or equity ≤ 0 | data missing |
| 13 | Debt-to-Equity (D/E) | summary (percent → ratio ÷ 100); fallback total debt ÷ equity | < 0.5 `STRONG` [spec]; < 2 `ACCEPTABLE` [spec] | — | ≥ 2 or negative | not reported (common for banks) |
| 14 | Price-to-Sales (PSR) | summary | < 1.5 `PREFERRED` [spec] | ≤ 2 `TOLERABLE` [spec] | > 2 | not reported |
| 15 | Free Cash Flow Strength | **FCF conversion = free cash flow ÷ operating cash flow** (latest FY) | > 50% [spec] | 0–50% | ≤ 0, or operating cash flow ≤ 0 | not reported (common for banks) |
| 16 | Return on Capital Employed (ROCE) | EBIT ÷ (total assets − current liabilities) | > 15% | — | ≤ 15% | EBIT / capital employed missing or capital employed ≤ 0 |
| 17 | Dividend Strength | yield = dividend rate ÷ price (else provider yield ÷ 100); payout ratio | yield ≥ 1% and payout ≤ 75% | pays a dividend but low yield or high payout | no dividend | dividend data not reported |
| 18 | Sector Outlook | Sector Master outlook set by an administrator | `POSITIVE` | `NEUTRAL` | `NEGATIVE` | not assessed, or sector unknown |

### Score and coverage

- **FQVF score** = 100 × (PASS + 0.5 × WARNING) ÷ (scored checks evaluated).
  Checks 3 and 4 are informational and excluded. `NOT_AVAILABLE` checks are
  excluded, never counted as failures.
- **Coverage** = scored checks evaluated ÷ 16. A score must always be read
  with its coverage.

### Data-integrity rules

- Values are provider data or `None`; NaN/∞/non-numeric values become `None`
  and are listed in the snapshot's `issues` (see data health).
- **Split / bonus detection:** provider EPS history is not restated. If, between
  consecutive fiscal years, the implied share count (net-income ratio ÷ EPS
  ratio) changes by more than 35%, checks 2 and 5 (and the derived PEG) are
  `NOT_AVAILABLE` with that reason rather than reporting a false EPS collapse.
- **5-Year EPS Progression** is evaluated with 4 fiscal years when only 4
  exist, and the explanation says so.
- Sector Outlook is never inferred: it is an explicit, audited administrator input.

## Limitations

- One provider (Yahoo Finance); statements usually cover 4 fiscal years.
- Financial companies (banks/NBFCs) do not report EBIT or FCF in the same
  structure, so checks 15–16 (and often 13) are `NOT_AVAILABLE` for them.
- Graham number is a conservative heuristic for value, not a valuation model.
- Thresholds are fixed by design; changing them is a framework version change.
