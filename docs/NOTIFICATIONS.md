# Notifications, Devices, Preferences and Reports

The notification engine is separate from ranking. The ranking engine
produces runs; the notification engine reads completed runs and decides
what, if anything, users are told.

| Piece | Location |
|---|---|
| Change detection | `notifications/detector.py` |
| Fan-out, deduplication, rate limits, quiet hours, dispatch | `notifications/service.py` |
| Push providers (FCM, APNs) | `notifications/push.py` |
| Admin settings and templates | `notifications/settings.py` (app_config `notifications.settings`, `notifications.templates`) |
| Tables | `notifications`, `notification_runs`, `notification_deliveries`, `notification_preferences`, `push_devices`, `watchlist_alert_settings`, `user_feedback` (migration `ff56f2194d59`) |
| User API | `api/routes/notifications.py`, `api/routes/watchlist.py` |
| Admin API | `api/routes/admin_notifications.py`; admin console pages **Notifications** and **User Reports** |
| Tests | `tests/test_notifications.py`; mobile `NotificationAndNavigationTests.cs` |

## Pipeline

1. A completed ranking run.
2. The run is compared with the previous **full-universe** RANKING run. Runs
   with explicit symbols or a limit are never compared.
3. Material changes are detected.
4. Per-user drafts are built, applying preferences, the watchlist and
   per-stock alert switches.
5. Duplicates are removed: `UNIQUE(user_id, dedup_key)`.
6. Push decisions are made: global switch, user push switch, per-stock
   cooldown, per-run and per-day limits, quiet hours.
7. Every draft becomes a `notifications` row (Notification Center), whatever
   happens to its push.
8. Push dispatch writes one `notification_deliveries` row per device.

**When it runs:**

- After every background ranking run (`engine_runs.service.notify_after_run`).
- From the scheduled ranking job.
- On demand: `POST /admin/notifications/process-run`.

**Idempotency.** A `notification_runs` row is unique per
`(kind, source_key)`. Processing the same ranking run again returns the
existing record and creates nothing.

## Events (when they are created)

| Type | Rule (thresholds are admin-configurable) | Audience | Opens |
|---|---|---|---|
| NEW_TOP_CANDIDATE | Enters the Top Candidates list (eligible, rank ≤ `top_picks.limit`) | Users with "New Top Candidate" on (default **on**) | `/stock/{symbol}`; more than `aggregate_above` (3) entrants become one combined notification that opens `/top-picks` |
| TOP_CANDIDATE_REMOVED | Leaves the list. The reason is one of: score decreased X→Y; risk increased; eligibility/data quality changed (with the reasons); not analysed; other stocks ranked higher | Users with this switch on (default off) | `/stock/{symbol}` or `/top-picks` |
| SCORE_CHANGE | \|Δ StockAI Score\| ≥ `score_change_threshold` (10 points), for current or previous Top Candidates | Users with "StockAI Score changes" on (default off) | `/stock/{symbol}` |
| FQVF_CHANGE | \|Δ checks passed\| ≥ `fqvf_change_min_checks` (2), for current or previous Top Candidates | Users with "FQVF changes" on (default off) | `/stock/{symbol}` |
| WATCHLIST_ALERT | For each watched stock, **one** combined message covering: score change ≥ threshold; rank change ≥ `rank_change_threshold` (15); entering or leaving the Top Candidates; FQVF change; gaining or losing eligibility; crossing the high-risk line | Owner, if "Watchlist alerts" is on (default **on**) and the stock is not muted; each kind can be switched per stock | `/stock/{symbol}` |
| DAILY_SUMMARY | Trading days only, after the user's time (IST). Content: top `daily_summary_size` (5) candidates from the latest full run. Skipped if that run is more than 4 days old | Opt-in (default off) | `/top-picks` |
| MARKET_REGIME_CHANGE | NIFTY 50 regime label changed **and** \|Δ regime score\| ≥ `regime_min_score_change` (0.3). Pushed at most once per `regime_cooldown_hours` (72) | Opt-in (default off) | `/` |

A watched stock is never notified twice in the same run: the watchlist
alert takes precedence over the general event types.

## Deduplication rule

A notification is unique by **user + dedup key**:

| Notification | Dedup key |
|---|---|
| Stock events | `TYPE:SYMBOL:ranking_run_id` |
| Combined notifications | `TYPE:MULTI:ranking_run_id` |
| Regime change | `MARKET_REGIME_CHANGE:run_id` |
| Daily summary | `DAILY_SUMMARY:<IST date>` |

The same user, stock, event type and ranking snapshot therefore never
produce a second notification.

## Rate limiting (configurable)

- **Per-stock cooldown:** at most one push per user, stock and type within
  `stock_cooldown_hours` (20). Later ones show `SUPPRESSED_COOLDOWN`.
- **Push caps:** `max_push_per_user_per_run` (5) and
  `max_push_per_user_per_day` (10). Notifications are prioritised in this
  order: watchlist, new candidate, removed, score, FQVF, regime. Extras show
  `SUPPRESSED_RATE_LIMIT`.
- **Quiet hours:** per user, default 22:00–07:00 IST. Pushes are queued
  (`QUEUED_QUIET_HOURS`) and sent by the next dispatch after the window. A
  queued push expires after `quiet_queue_max_hours` (12).
- **Not pushed, still recorded:**
  - user turned push off: `SUPPRESSED_PREFERENCE`;
  - global push switch off: `DISABLED_GLOBALLY`;
  - no device with a token: `NO_DEVICE`;
  - provider not configured: `PROVIDER_NOT_CONFIGURED`.

All of these notifications remain in the user's Notification Center.

## Admin controls (all audited)

Set through `PUT /admin/config/notifications.settings` or the
**Notifications** page:

- global emergency switch (`enabled`), push switch, per-type switches;
- thresholds, rate limits, quiet-hour and daily-summary defaults.

Templates are set through `PUT /admin/config/notifications.templates`.
Only placeholders from the documented field list are accepted.

Actions: process a run, send daily summaries, dispatch queued pushes, send
a test to your own devices. The notification engine version is
`notify-v1.0` (`config.NOTIFICATION_ENGINE_VERSION`).

## Devices

`POST /devices` takes `device_id`, `platform` (android/ios), `push_token`,
`app_version` and `permission`. The provider is FCM for Android and APNs
for iOS.

- **Token replacement:** each `(user, device_id)` has one row; a new token
  replaces the old one.
- **Account switch:** if another account registers the same token, the old
  registration is deactivated.
- **Invalid tokens:** a provider "invalid token" response (FCM 404 /
  UNREGISTERED, APNs 410 / BadDeviceToken) deactivates the device.
- **Sign-out:** `POST /auth/logout`, revoking a session, and "sign out all
  devices" deactivate that device's registration and delete its token.
- **Privacy:** tokens are only returned to their owner, masked (`...abc123`),
  and are never logged.

## Push providers: production configuration still required

Credentials come only from environment variables. Never commit them.

| Platform | Backend environment | App side |
|---|---|---|
| Android (FCM HTTP v1) | `FCM_PROJECT_ID`, `FCM_SERVICE_ACCOUNT_FILE` (Firebase service-account JSON with the Firebase Cloud Messaging API enabled) | Either put `Platforms/Android/google-services.json` (gitignored) in the mobile repo, or build with `-p:FirebaseApplicationId=… -p:FirebaseApiKey=… -p:FirebaseProjectId=… -p:FirebaseSenderId=…`. The Firebase Android app's package name must match `ApplicationId`. |
| iOS (APNs, token-based) | `APNS_KEY_FILE` (.p8), `APNS_KEY_ID`, `APNS_TEAM_ID`, `APNS_BUNDLE_ID`, `APNS_USE_SANDBOX=true` for development builds | Push Notifications capability on the App ID; a provisioning profile with `aps-environment` (Entitlements.plist has `development`); build and sign on a Mac with Xcode |

`GET /admin/notifications/stats` shows which providers are configured.
`POST /admin/notifications/test` sends a test to the administrator's own
devices.

Without credentials the whole pipeline still runs. Notifications appear in
the app and the push status says why no push was sent.

## Reports (feedback)

- **Submitting:** `POST /feedback` takes a category (incorrect stock data,
  stale data, company information, ranking issue, app problem, other), an
  optional symbol, and a message. Each user is limited to 10 reports a day.
- **Account deletion:** `POST /account/deletion-request` records a request
  for an administrator. Deletion is not automated.
- **Admin review:** reports are listed at `/admin/feedback` and on the
  **User Reports** page. Status changes are audited.
- **No email:** no email or ticketing integration exists. The confirmation
  text says the report was recorded for review.
