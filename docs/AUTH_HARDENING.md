# Authentication, concurrency and reliability hardening (2026-10)

Scope: Google sign-in end to end, forgot/reset password, refresh-token
rotation, OTP verification, engine-run concurrency, authorization review,
logging, migrations. Apple Sign-In was deliberately not touched.
Repositories: backend (this repo), `StockAIPro.Mobile`, `StockLens.Web`.

## 1. Current architecture

```
Mobile (MAUI Blazor)          Web (Blazor WASM)              Backend (FastAPI + SQLModel, PostgreSQL)
 Credential Manager ─┐         Google Identity Services ─┐
   (Android only)    │ id_token                          │ id_token
                     ▼                                    ▼
 StockAIPro.Mobile.Core (shared by both clients, web via git submodule)
   AuthService ─► AuthApiClient ─► POST /api/v1/auth/google | /login | /otp/verify | /refresh ...
                                        │
                                        ▼
            auth/external_identity.py (OIDC verify) ─► find_or_create_user_for_identity
            auth/otp_service.py, auth/service.py (passwords, refresh tokens)
            auth/password_reset.py (new)        auth/throttle.py (new)
                                        │
                                        ▼
            auth/token_issuance.py: JWT access token (15 min, HS256, sub = user id)
                                    + opaque refresh token (30 days, SHA-256 at rest, rotated)
```

- Tokens are stored by the clients in `ITokenStore` (SecureStorage on mobile,
  browser storage on web); `AuthenticatedHttpMessageHandler` refreshes on 401.
- Authorization (`auth/dependencies.py`) re-reads the user and its roles from
  the database on every request; the `roles` claim in the JWT is never trusted.
- The ranking engine runs in the API process (admin "start run", background
  thread) **and** in the separate Render cron container (`scripts/scheduled_jobs.py`).

## 2. Root causes found

| # | Area | Root cause | Severity |
|---|------|-----------|----------|
| G1 | Google | Only one `aud` accepted (`GOOGLE_OAUTH_CLIENT_ID`). Any other client of the same project (an iOS client, a separate web client) is rejected - no way to configure more without code. | High (blocks iOS/web variants) |
| G2 | Google | `email_verified` was parsed but **never enforced**. An unverified Google email created a StockLens account owning that address - an attacker could pre-register a victim's email (account pre-hijacking), and the victim's later password reset / OTP login would land in the attacker-controlled account. | High |
| G3 | Google | `PyJWKClient` was constructed per request, so every login re-downloaded Google's JWKS (extra latency; any googleapis.com hiccup = failed login). | Medium (reliability) |
| G4 | Google | No clock-skew leeway; a phone a few seconds ahead fails with "token used before issued". `iat`/`iss`/`aud` not in `require`. | Medium |
| G5 | Google | First sign-in created the user (commit) and then the identity (second commit). Two simultaneous first sign-ins → second insert hits `uq_external_identity_subject` → HTTP 500 and an orphan password-less user. A second TOCTOU: link lookup → (other request commits) → email check → spurious 409. Reproduced on PostgreSQL. | Medium |
| G6 | Mobile | Android `GoogleSignInService`: exceptions after the picker returned (null data, non-Google credential, `GoogleIdTokenCredential.CreateFrom` failures, no activity) escaped to `Login.razor`, which only catches the `GoogleSignIn*` exceptions → unhandled error UI. `NoCredentialException` (the usual setup error: package/SHA-1 not registered, or no Google account) produced a generic message. | Medium |
| G7 | Config | The mobile app only enables Google when built with `-p:GoogleServerClientId=…`; the Android OAuth client must match package `com.companyname.stockaipro.mobile` + the signing SHA-1. Either missing = "not configured" / "No credentials available". This is the most likely reason Google "does not work" on a given build (see §7). | High (config) |
| G8 | iOS | There is **no** Google Sign-In implementation for iOS (the default `GoogleSignInService` fails closed). Not a bug - unimplemented (see §7.3). | Info |
| R1 | Refresh | `rotate_refresh_token`: `SELECT` → check `revoked_at` → `UPDATE`. Two concurrent requests both pass the check and both get a new token (reproduced: N concurrent requests → several successes). | P0 |
| O1 | OTP | `verify_otp`: read challenge → compare → set `consumed_at`. Concurrent requests with the right code all succeed (reproduced). Additionally `attempt_count += 1` was a lost update, so concurrent wrong guesses exceeded `OTP_MAX_ATTEMPTS` (reproduced: 15 concurrent guesses all compared). | P0 |
| E1 | Engine | `create_run` checks for a RUNNING row, then inserts. The in-process `threading.Lock` does not cover the cron container or other workers → two RUNNING ranking runs (reproduced). | P1 |
| P1 | Password | No forgot/reset flow existed; `change_password` exists but password-less (Google/OTP) accounts could never set one. | Feature |
| L1 | Logging | `OTP_SENT` logged the full destination email address. | Low |

The authorization review (§5) found the design already correct: roles and
`is_active` are read from the database on every request.

## 3. What changed

### Backend
- **Google / OIDC** (`auth/external_identity.py`): audiences from
  `GOOGLE_ALLOWED_AUDIENCES`; process-wide cached JWKS client; `require`
  `exp, iat, iss, aud, sub`; RS256 only; 60 s leeway (configurable); blank `sub`
  rejected; `email_verified` accepted as bool or `"true"`. New
  `ExternalEmailNotVerifiedError` (→ **403**) - a new account is only created
  from a provider-verified email; returning users are matched by `(provider,
  sub)` regardless of email. User + identity + role are created in **one
  transaction**; `IntegrityError` → re-resolve (same identity → sign in; same
  email → 409). `link_identity` maps a concurrent duplicate to 409 instead of 500.
- **Password reset** (`auth/password_reset.py`, `api/routes/auth_password.py`,
  `auth/password_reset_delivery.py`, `auth/throttle.py`) - see §4.
- **Refresh rotation** (`auth/service.py`): one conditional
  `UPDATE … SET revoked_at=now WHERE token_hash=? AND revoked_at IS NULL AND expires_at>now`;
  only `rowcount == 1` proceeds; revoke + new token in **one commit**; replay of
  a revoked token logs `REFRESH_TOKEN_REUSE`. Logout uses the same atomic update.
- **OTP** (`auth/otp_service.py`): attempt counted with
  `UPDATE … SET attempt_count=attempt_count+1 WHERE … AND attempt_count<MAX AND not consumed AND not expired`;
  consumption with `UPDATE … SET consumed_at=now WHERE consumed_at IS NULL`;
  constant-time hash comparison; `OTP_REUSE` / `OTP_MAX_ATTEMPTS` events.
- **Engine runs** (`db/models/market.py`, `engine_runs/service.py`): partial
  unique index `uq_engine_runs_one_running_ranking ON engine_runs(kind) WHERE
  status='RUNNING' AND kind='RANKING'` (PostgreSQL and SQLite). `create_run`
  turns the `IntegrityError` into the existing `RunInProgressError`, so every
  caller keeps its behaviour: admin API → **409**, cron job → `SKIPPED`,
  product API → existing handling. Stale-run expiry (3 h) still frees the slot.
  Single-stock runs are unaffected.
- **Sessions end on password reset**: `users.token_valid_after`; access tokens
  now carry a fractional `iat` and `get_current_user` rejects tokens issued
  before the reset (no 15-minute window).
- **Security events** (`auth/security_events.py`): one `SECURITY_EVENT | NAME |
  k=v` line on the `stocklens.security` logger; emails masked (`j***@x.com`),
  secret-looking fields dropped. Events: `GOOGLE_LOGIN_SUCCEEDED/FAILED`,
  `EXTERNAL_ACCOUNT_CREATED`, `EXTERNAL_IDENTITY_LINKED/UNLINKED`,
  `PASSWORD_RESET_REQUESTED/COMPLETED/FAILED`, `REFRESH_TOKEN_REUSE`,
  `OTP_REUSE`, `OTP_MAX_ATTEMPTS`.
- Startup warnings when Google, reset links or email are not configured.

### Mobile (`StockAIPro.Mobile`)
- `IAuthApiClient.ForgotPasswordAsync/ResetPasswordAsync`,
  `IAuthService.RequestPasswordResetAsync/ResetPasswordAsync`; new
  `/forgot-password` page, "Forgot password?" on Login.
- Android `GoogleSignInService`: every failure becomes
  `GoogleSignInFailedException`; `NoCredentialException` gets an actionable
  message. Type-checked against the real binding assemblies.

### Web (`StockLens.Web`)
- `/forgot-password` and `/reset-password?token=…` pages (token removed from
  the address bar immediately, client-side 8-72 + confirmation check); "Forgot
  password?" on Login; Google 403/409 show the backend's explanation.
- Submodule `external/StockAIPro.Mobile` bumped to the mobile commit above.

## 4. Password reset design

```
POST /api/v1/auth/forgot-password {"email"}      → 202 {"message": "If an account exists for this email address, a password reset link has been sent."}
POST /api/v1/auth/reset-password  {"token","new_password"} → 200 {"message"} | 400 invalid/expired/used | 422 password policy | 429
```

- Same 202 body for existing, unknown, deactivated and per-account-throttled
  emails; the email is sent by a `BackgroundTasks` job after the response, so
  response time does not reveal SMTP work either.
- Token: `secrets.token_urlsafe(32)` (256 bits); only SHA-256 stored
  (`password_reset_tokens.token_hash`, unique). Expires after
  `PASSWORD_RESET_TOKEN_EXPIRY_MINUTES` (30). A new request supersedes older
  unused links. Consumption is one conditional `UPDATE … WHERE used_at IS NULL
  AND expires_at > now`; exactly one of N concurrent requests wins.
- The new password is hashed **before** the token is consumed, so a policy
  rejection does not burn the link. Policy = registration's (8-72 chars, ≤72
  bytes for bcrypt).
- On success, in the same transaction: password hash, `used_at`, all other
  links invalidated, **all refresh tokens revoked**, `token_valid_after = now`;
  then push devices are unregistered (same as "sign out all devices").
- Rate limits (database-backed, shared by all workers, keys stored hashed):
  `PASSWORD_RESET_MAX_REQUESTS_PER_IP` (10/h, **429**, counted for every
  request so it reveals nothing), `PASSWORD_RESET_MAX_EMAILS_PER_ACCOUNT` (3/h,
  **silent**), `PASSWORD_RESET_MAX_ATTEMPTS_PER_IP` (20/h on reset, 429).
- **Password-less accounts** (Google/OTP): allowed to set a password through
  this flow. The link goes to the account's own address, and control of that
  mailbox already grants sign-in via email OTP, so this adds no new way in.
  Combined with G2 (no account is ever created from an unverified provider
  email), the address on the account is one its owner has proven.
- Email: the existing SMTP integration (`OTP_SMTP_*`), via a shared
  `send_smtp_message`. Contains instructions, link, expiry, "signs you out
  everywhere", and a do-not-forward warning. The link/token is never logged; a
  delivery failure logs only the exception type.

## 5. Authorization review

- Admin routes: router-level `Depends(require_admin)` on `/admin/*`
  (admin, admin_masters, admin_notifications). `require_role` re-reads roles
  from `user_roles` on every request - **preserved**.
- `get_current_user`: signature, `type == "access"`, user exists, `is_active`,
  and now `token_valid_after`. Deleted users → 401; disabled users → 401
  (and refresh/login rejected).
- Horizontal access: per-user resources key on `current_user.id`
  (watchlist alerts 404 for another user's item; session revocation ignores
  other users' sessions).
- `tests/test_authorization_security.py` proves: role removed → old JWT 403;
  role granted → old JWT 200; forged `roles:["ADMIN"]` ignored; wrong secret,
  `alg:none`, wrong token type, expired → 401; refresh token not usable as
  access token; deactivated/deleted users locked out.

## 6. Error contract (auth endpoints)

| Situation | Status | Body |
|-----------|--------|------|
| Bad password / unknown email (login) | 401 | `Invalid email or password` (same for both) |
| Invalid/expired/forged Google token, wrong aud/iss | 401 | `Invalid Google identity token` |
| Google email not verified | 403 | explanation |
| Google email belongs to an unlinked account | 409 | "sign in and link" instructions |
| Refresh token unknown/expired/reused | 401 | `Invalid or expired refresh token` |
| OTP wrong / expired / used / too many attempts | 401 | generic per case |
| Reset token unknown/expired/used | 400 | one message for all three |
| Password policy | 422 | validation detail |
| Rate limited | 429 | message + `Retry-After` (reset endpoints) |
| Engine run already running | 409 (admin API) | `An engine run is already in progress` |

`forgot-password` and `otp/request` never reveal account existence.
Registration's 409 on a duplicate email is inherent to self-service sign-up and unchanged.

## 7. Google Sign-In configuration (exact)

All client ids must belong to **one** Google Cloud project.

### 7.1 Google Cloud Console
1. APIs & Services → OAuth consent screen: External; add test users while in
   *Testing* (or publish).
2. Credentials → Create client → **Web application** ("StockLens server").
   Its client id is the **audience**: backend `GOOGLE_OAUTH_CLIENT_ID`, mobile
   `-p:GoogleServerClientId`, web `StockLens:GoogleClientId`
   (`wwwroot/appsettings.json`). For the web button also add the web app's
   origin under *Authorized JavaScript origins*.
3. Credentials → Create client → **Android**: package
   `com.companyname.stockaipro.mobile` (the `ApplicationId` in
   `StockAIPro.Mobile.csproj`) and the **SHA-1 of the key that signs the APK**:
   - debug: `keytool -list -v -keystore ~/.android/debug.keystore -alias androiddebugkey -storepass android -keypass android`
     (on Windows `%USERPROFILE%\.android\debug.keystore`; the dev laptop's is
     `2A:89:D9:F5:90:C3:E3:13:07:55:40:38:AA:E5:D2:07:A0:89:C6:8A`)
   - release: your release keystore's SHA-1, **and** for Play distribution
     the *App signing key* SHA-1 from Play Console → Setup → App integrity.
   One Android client per (package, SHA-1); create one for each key. No
   secret, no `google-services.json` is needed for sign-in. (SHA-256 is not
   used by this OAuth client type.)

### 7.2 Android build
```
dotnet publish StockAIPro.Mobile -f net10.0-android -c Release \
  -p:StockAIProApiBaseUrl=https://<api-host> \
  -p:GoogleServerClientId=<WEB client id>.apps.googleusercontent.com
```
Requirements on the device: Google Play services, a Google account; the app
must be signed with a key whose SHA-1 is registered (7.1 step 3).

| Symptom | Cause |
|---------|-------|
| "Google sign-in is not configured on this device" | built without `GoogleServerClientId` (or not a `…apps.googleusercontent.com` value) |
| "No Google account is available for sign-in…" | `NoCredentialException`: no account on device, **or package/SHA-1 not registered**, or the account is not a test user of an app in *Testing* |
| 401 "Invalid Google identity token" | backend `GOOGLE_OAUTH_CLIENT_ID` ≠ the app's `GoogleServerClientId`, or not set |
| 403 email not verified | the Google account's email is unverified |
| 409 | that email already has a StockLens account: sign in to it, Account → link Google |

### 7.3 iOS (not implemented - what it would take)
There is no iOS Google implementation; the button reports "not configured".
To add it: create an **iOS** OAuth client (bundle id = `ApplicationId`); add
its *reversed client id* (`com.googleusercontent.apps.<id>`) as a
`CFBundleURLTypes` URL scheme in `Platforms/iOS/Info.plist`; implement
`IGoogleSignInService` for iOS (Google Sign-In SDK binding, or an
authorization-code + PKCE flow with `WebAuthenticator` using the iOS client id
and the reversed-client-id redirect, exchanging the code at
`https://oauth2.googleapis.com/token` for the `id_token`); and add the iOS
client id to the backend via `GOOGLE_IOS_CLIENT_ID` (or
`GOOGLE_ALLOWED_AUDIENCES`) - the backend side is ready. App Store Review
Guideline 4.8 requires an equivalent privacy-focused login option when a
third-party login is offered on iOS; confirm the current rules before release.

## 8. Configuration reference (new/changed)

| Variable | Default | Notes |
|----------|---------|-------|
| `GOOGLE_OAUTH_CLIENT_ID` | - | Web client id; always an allowed audience |
| `GOOGLE_WEB_CLIENT_ID`, `GOOGLE_ANDROID_CLIENT_ID`, `GOOGLE_IOS_CLIENT_ID` | - | optional extra audiences |
| `GOOGLE_ALLOWED_AUDIENCES` | - | comma-separated extra audiences |
| `EXTERNAL_ID_TOKEN_LEEWAY_SECONDS` | 60 | clock skew for provider tokens |
| `FRONTEND_BASE_URL` | - | e.g. `https://stocklens-web.onrender.com`; link = `<base>/reset-password?token=…` |
| `PASSWORD_RESET_URL` | `FRONTEND_BASE_URL/reset-password` | full override |
| `PASSWORD_RESET_TOKEN_EXPIRY_MINUTES` | 30 | |
| `PASSWORD_RESET_WINDOW_SECONDS` | 3600 | window for the three limits |
| `PASSWORD_RESET_MAX_EMAILS_PER_ACCOUNT` | 3 | silent |
| `PASSWORD_RESET_MAX_REQUESTS_PER_IP` | 10 | 429 |
| `PASSWORD_RESET_MAX_ATTEMPTS_PER_IP` | 20 | 429 |
| `OTP_SMTP_*` | - | the one email provider, now also for reset emails |

Secrets (`JWT_SECRET_KEY`, `OTP_SMTP_PASSWORD`, `DATABASE_URL`) stay in the
deployment's secret store. OAuth client ids are public identifiers; no client
secret is used anywhere.

## 9. Database migrations

- `b7e4c2a91d35`: `password_reset_tokens` (unique `token_hash`, indexes on
  `user_id`, `created_at`), `auth_throttle_events` (composite index
  `(bucket, key_hash, created_at)` + `created_at`), nullable
  `users.token_valid_after`. Additive only.
- `c3f9d8e0a4b2`: marks all but the newest duplicate RUNNING ranking run as
  FAILED (normally none), then creates the partial unique index.

Both are reversible and verified on PostgreSQL 16 and SQLite: upgrade from
the previous head, `alembic check` (no drift), downgrade, re-upgrade; the
duplicate-cleanup path was exercised with two RUNNING rows. Deploy order:
`alembic upgrade head` (Render `preDeployCommand`) before the new code serves.

## 10. Tests

| File | What |
|------|------|
| `tests/test_google_id_token_security.py` | real RS256 verification with a local JWKS: valid, 2nd audience, wrong/missing aud, wrong/missing iss, expired, skew, future iat, wrong key, tampered payload, unknown kid, `alg:none`, HS256 key confusion, missing claims, blank sub; endpoint: tokens + device session + refresh + logout, returning user with changed email, case-variant emails, unverified (403), existing password account (409 → link → login), identity owned by another user, deactivated user, last-method unlink, no token/email in logs; concurrent first logins → one user |
| `tests/test_password_reset.py` | enumeration-safe responses, case-insensitive lookup, deactivated, hash-at-rest, expiry, reuse, unknown, supersede, policy (token not burned), session + access-token invalidation, per-account/IP limits, attempt limit, Google/OTP accounts, email content, SMTP, delivery failure, nothing secret logged |
| `tests/test_auth_concurrency.py` | 10 simultaneous refreshes → 1 success; 10 OTP verifications → 1; 15 concurrent wrong guesses → exactly 5 compared; 10 resets → 1; 10 ranking runs → 1 RUNNING; raw duplicate insert rejected by the DB; stale/finished runs; single-stock runs unaffected |
| `tests/test_authorization_security.py` | §5 |

Concurrency tests run against a file-backed SQLite database and, when
`TEST_DATABASE_URL` points at a disposable PostgreSQL database, PostgreSQL:

```
TEST_DATABASE_URL=postgresql://user:pass@127.0.0.1/stocklens_test pytest tests/test_auth_concurrency.py
```

All four race tests **fail against the previous implementation** (verified)
and pass now. Mobile Core: `dotnet test StockAIPro.Mobile.Tests`; web:
`dotnet test tests/StockLens.Web.Tests`.

## 11. Risks and compatibility

- **No API contract was broken.** New endpoints only; existing responses
  unchanged except: Google login with an *unverified* email now gets 403
  (previously an account was created - the vulnerability G2). Existing
  accounts created that way keep working (matched by `sub`).
- Access tokens' `iat` is now fractional; no client parses it (checked
  backend, mobile, web).
- Password reset revokes all sessions, including on the device that
  requested it - intended.
- `change-password` behaviour is unchanged (it does not revoke other
  sessions); consider doing so later.
- Rate limits are per client IP as seen by uvicorn. Behind Render's proxy,
  run uvicorn with `--proxy-headers` **and** `--forwarded-allow-ips` set to the
  proxy range (or `*` if the platform guarantees it) or every request shares
  one IP and the per-IP limits apply globally. The existing OTP limits have
  the same dependency.
- Reset links in a query string can appear in static-host access logs of the
  web app; the token is single-use, 30-minute, and the page strips it from
  the URL immediately; `Referrer-Policy: strict-origin-when-cross-origin`
  keeps it out of cross-origin referrers.
- Refresh-token reuse is logged, not punished (no family revocation): a
  client race would otherwise sign users out. Revisit if reuse events appear
  in production logs.
- Not verified here: a live Google sign-in (needs real OAuth clients and a
  device), real SMTP delivery, an Android APK build (Google Maven is not
  reachable from this environment; the changed Android file was type-checked
  against the real binding assemblies instead), and any iOS build.
