# StockAI Pro Authentication — Phase 8

This document covers the Phase 8 authentication expansion: Google/Apple
sign-in, OTP login, device/session management, account linking, and the
mobile PIN model. It assumes the Phase 1-6 foundation (users/roles, bcrypt
passwords, JWT access tokens, opaque hashed refresh tokens with rotation)
already described in the codebase itself (`auth/service.py`,
`auth/security.py`).

**The backend is the sole authentication authority.** The mobile app never
independently decides a user is authenticated — it only ever holds and
presents tokens the backend issued, and always re-validates against the
backend (via refresh) when in doubt.

## 1. Architecture

```
Google / Apple id_token  ──┐
Phone/email OTP          ──┼──► backend verifies ──► find-or-create user ──► JWT + refresh token
Password (existing)      ──┘
```

Every login path — password, Google, Apple, OTP — ends up calling the same
`auth/token_issuance.py:issue_token_pair()`, so they all produce tokens
exactly the same way and all participate in device/session tracking
identically. There is no separate "SSO token" or "OTP token" format.

New tables (see `db/models/user.py`), each owned by one module:

| Table                 | Owning module              | Purpose |
|-----------------------|-----------------------------|---------|
| `external_identities` | `auth/external_identity.py` | Google/Apple `(provider, provider_subject)` → user |
| `otp_challenges`      | `auth/otp_service.py`       | Hashed, expiring, attempt-limited OTP codes |
| `trusted_devices`     | `auth/device_service.py`    | "this user completed full auth on this device"; `pin_enabled` flag only |
| `refresh_tokens.device_id` | `auth/service.py` (existing table, new column) | Lets sessions be listed/revoked per device |

`users.email`, `users.phone`, and `users.hashed_password` are all nullable:
a Google/Apple/OTP-only account may have no password, and a phone-only OTP
account may have no email.

**JWT subject**: access tokens now carry the user's numeric id as `sub`
(not email), since some accounts have no email at all. `get_current_user`
accepts *either* an id or an email in `sub`, so any already-issued
email-subject token keeps working until it naturally expires (15 minutes).

## 2. Google Sign-In

- Mobile obtains a Google **id_token** via the platform's native Google
  Sign-In SDK (not implemented here — see "Mobile" section below for what's
  still required).
- Backend verifies it as a standard OIDC JWT against Google's own rotating
  public keys (`https://www.googleapis.com/oauth2/v3/certs`), checking
  signature, `aud` (must equal `GOOGLE_OAUTH_CLIENT_ID`), and `iss`.
- The client-supplied email is **never** trusted by itself — only `sub`,
  once the signature is verified, is used as the identity key.
- See `auth/external_identity.py:GoogleIdentityVerifier` /
  `find_or_create_user_for_identity()`.

**Setup required for production**: register an OAuth 2.0 client in Google
Cloud Console for the mobile app (Android + iOS client IDs, or a single
server/web client ID depending on the chosen native SDK flow), and set:

```
GOOGLE_OAUTH_CLIENT_ID=<your OAuth client id>
```

Without this set, `/auth/google` and `/auth/link/google` always fail closed
(`401 Invalid Google identity token`) rather than silently accepting
anything.

## 3. Apple Sign-In

- Mobile obtains an Apple **identity_token** via Sign in with Apple.
- Backend verifies it as a standard OIDC JWT against Apple's own rotating
  public keys (`https://appleid.apple.com/auth/keys`), checking signature,
  `aud` (must equal `APPLE_SERVICES_ID`), and `iss`.
- Apple only includes `email` on the **first** authorization (or when the
  user chooses private relay) — `sub` is the only field guaranteed present
  on every login and is the only one used as the identity key.
- See `auth/external_identity.py:AppleIdentityVerifier`.

**Setup required for production**: register a Services ID with Sign in
with Apple enabled in the Apple Developer portal, and set:

```
APPLE_SERVICES_ID=<your Services ID>
```

Same fail-closed behavior as Google when unset.

## 4. OTP (email or phone)

`POST /auth/otp/request {destination}` → generates a code, stores only a
per-row-salted SHA-256 hash of it, and delivers it via whatever
`IOtpDeliveryService` is configured (see below). Always returns 204
regardless of whether `destination` has an account, so this endpoint can't
be used to enumerate registered users.

`POST /auth/otp/verify {destination, code}` → validates the code
(expiry, one-time-use, attempt limit), then finds-or-creates the StockAI
user for `destination` and issues tokens exactly like any other login path.

Configuration (`config.py`, all overridable via environment variables):

| Variable | Default | Meaning |
|---|---|---|
| `OTP_CODE_LENGTH` | 6 | Digits per code |
| `OTP_EXPIRE_SECONDS` | 300 | Code lifetime |
| `OTP_MAX_ATTEMPTS` | 5 | Wrong guesses before a challenge is burned |
| `OTP_RESEND_COOLDOWN_SECONDS` | 30 | Minimum gap between two codes for the same destination |
| `OTP_MAX_REQUESTS_PER_WINDOW` | 5 | Cap per destination AND per requesting IP |
| `OTP_REQUEST_WINDOW_SECONDS` | 1800 | Rolling window the above cap applies over |

### OTP provider setup

**No real SMS/email provider is wired up.** `auth/otp_delivery.py` defines
the `IOtpDeliveryService` abstraction; production needs a real
implementation plugged into `get_otp_delivery_service()` (e.g. Twilio for
SMS, any transactional email API for email) once an account/credentials
exist. Required production secrets would be supplied via environment
variables the same way `GOOGLE_OAUTH_CLIENT_ID`/`JWT_SECRET_KEY` are — never
committed to source.

For local development only, set:

```
OTP_DEV_LOG_CODES=true
```

which logs each generated code to the server's own log file instead of
delivering it anywhere. **This must never be set in production** — it
exists purely so a developer can exercise the full OTP flow without a real
provider account. All automated tests instead use
`auth/otp_delivery.py:FakeOtpDeliveryService`, a deterministic in-memory
double with no I/O at all.

## 5. Device / session management

Every login (password, Google, Apple, OTP) can optionally include a
client-generated `device_id` (+ `device_name`). When present:

- the resulting refresh token is tagged with that `device_id`, and stays
  tagged with it across every rotation (`auth/service.py:rotate_refresh_token`)
- a `trusted_devices` row is created/refreshed (`auth/device_service.py:upsert_trusted_device`)

Endpoints (all authenticated):

- `GET /auth/sessions` — list active sessions, newest first, with device
  name where known
- `POST /auth/sessions/revoke {session_id}` — revoke one session
- `POST /auth/sessions/revoke-all {}` — **sign out all devices**
- `POST /auth/sessions/revoke-all {except_current: true, current_device_id}`
  — sign out every *other* device
- `POST /auth/devices/pin-enabled {device_id, enabled}` — purely
  informational flag the mobile client flips after setting up/clearing its
  own local PIN (see below) — the backend never sees the PIN itself

Revoking a session only affects **refresh** capability. An already-issued
access token for that session keeps working until its own short expiry
(15 minutes) — there is no server-side access-token blacklist, by design
(see `api/routes/auth_devices.py` module docstring for why one isn't
needed).

## 6. The 4-digit PIN is NOT a remote credential

This is the most important security property in this phase:

```
Google / Apple / OTP / existing login
        ↓
Authenticated StockAI session (JWT + refresh token, as always)
        ↓
Trusted device established (trusted_devices row)
        ↓
User sets a 4-digit PIN                      ← entirely on-device
        ↓
PIN protects LOCAL access to the already-stored session
```

On every subsequent launch, the PIN is checked **entirely on the mobile
device**, using `Microsoft.Maui.Storage.SecureStorage` (same mechanism
`SecureTokenStore` already uses for tokens) — the raw PIN is never sent to
the backend, never stored server-side, and there is no `POST
/login-with-pin` endpoint. A correct PIN only unlocks whatever refresh
token is *already* cached locally; it never bypasses the server.

If that cached refresh token has since been revoked (sign-out-all-devices,
a revoked single session, or natural expiry), the very next API call 401s,
`AuthenticatedHttpMessageHandler` attempts the normal refresh, the server
rejects it, and the mobile app falls back to full re-authentication
(Google/Apple/OTP/password) exactly as if no PIN existed. **A four-digit
PIN never overrides server-side revocation.**

Why not verify the PIN on the server instead? A 4-digit PIN has only 10,000
possible values — treating it as a remote credential (`POST
/login-with-pin`) would make it brute-forceable server-side no matter how
aggressively that one endpoint were rate-limited, and would turn "guess a
4-digit number" into a full account-takeover primitive. Keeping it
local-only means the worst a leaked/guessed PIN can do is unlock a session
*that device already legitimately holds* — not establish a new one.

## 7. Account linking

An authenticated user can explicitly attach another sign-in method:

- `POST /auth/link/google {id_token}`
- `POST /auth/link/apple {identity_token}`
- `GET /auth/identities` — list linked providers
- `DELETE /auth/identities/{provider}` — unlink, rejected
  (`400`) if it would leave the account with no way to sign in at all

Linking is **never automatic**. If a brand-new Google/Apple/OTP identity's
email matches an existing StockAI account, login returns `409 Conflict`
with instructions to sign in to the existing account and link explicitly
from there (`auth/external_identity.py:AccountLinkingRequiredError`) —
email collision alone is never treated as proof of ownership.

## 8. Mobile — what's implemented vs. what's still required

Implemented: device identity generation, local PIN storage/verification
with lockout, OTP request/verify UI, the `IGoogleSignInService` /
`IAppleSignInService` client abstractions, AuthApiClient/AuthService
extensions for every endpoint above, and device/session management calls.

**Not implemented (requires real provider setup this environment doesn't
have)**: the native Google Sign-In SDK integration and Sign in with Apple
entitlement/URL-scheme wiring that actually produce a real `id_token` /
`identity_token` on-device. The mobile `IGoogleSignInService`/
`IAppleSignInService` implementations throw a clear "not configured" error
until a real OAuth client id (Google) and Services ID + Sign in with Apple
capability (Apple Developer account) are supplied — see
`StockAIPro.Mobile.Core/Services/Authentication/GoogleSignIn.cs` /
`AppleSignIn.cs`. No live Google or Apple sign-in was tested end-to-end;
only the backend verification logic and the client-side plumbing up to
that boundary were.

## 9. Production deployment checklist

- [ ] `JWT_SECRET_KEY` — already required pre-Phase-8; unchanged
- [ ] `GOOGLE_OAUTH_CLIENT_ID`
- [ ] `APPLE_SERVICES_ID`
- [ ] A real `IOtpDeliveryService` implementation wired into
      `auth/otp_delivery.py:get_otp_delivery_service()`, with its own
      provider credentials as environment variables (never committed)
- [ ] `OTP_DEV_LOG_CODES` unset/`false`
- [ ] `alembic upgrade head` run against the production database
