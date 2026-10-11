# Exposed NewsAPI key: status and remediation plan (updated 11 Oct 2026)

This file never contains the key. It is identified by its length (32
characters) and its SHA-256 fingerprint, `c5d151c4f231…`; the full digest is
in `catalysts/providers.py` as `EXPOSED_NEWSAPI_KEY_SHA256`. A one-way hash of
a random 32-character key cannot be turned back into the key.

## 1. Status

| Item | Status |
|---|---|
| Key validity | **NOT REVOKED.** Last verified **STILL ACTIVE** (HTTP 200) on 11 Oct 2026. The owner cannot revoke it through the NewsAPI account (no self-service option available). Until NewsAPI support disables it, assume anyone can use it |
| Repository | `Himanshu11007/stock-prediction-app` is **public**; assume the key is compromised |
| Where it is | 7 commits (`bcf4da6` hardcoded in `news/api.py`; `c9ca6e2` onwards in `.streamlit/secrets.toml`) on every remote branch, and in the tracked `.streamlit/secrets.toml` on `main` and the pushed review branches |
| Does StockLens use it? | **No.** Verified 11 Oct 2026: no application code reads `.streamlit/secrets.toml` or an `API_KEY` entry. The legacy Streamlit news (`news/api.py`) uses Google News RSS; the news pipeline uses official RSS feeds by default |

## 2. Done on `review/independent-review-prep` (local, not pushed)

| Change | Effect |
|---|---|
| NewsAPI **disabled by default** | `NewsApiProvider` refuses to run unless `NEWSAPI_ENABLED=true` **and** a new key is set in `NEWSAPI_KEY`. A key alone never enables it, and the registry and health check report it as `DISABLED`. Every refusal happens before any network request |
| **The exposed key is refused** | Even when enabled, a key whose SHA-256 matches the exposed key is rejected. There is no fallback to any file, `secrets.toml` entry or default |
| Benchmark runner | Queries NewsAPI only with the same flag and a new key |
| Manual checker (`scripts/security/check_exposed_newsapi_key.py`) | The only code that reads the exposed key. Without `--confirm-read-exposed-key` it reads nothing and contacts nothing (exit 3). Never run from CI, schedulers or tests |
| `.streamlit/secrets.toml` | **Untracked** (`git rm --cached`) and ignored by `.gitignore`. The local copy is kept |
| Secret scanning | `scripts/security/scan_secrets.py` (tests and CI) fails on tracked secret files or credentials. `tests/test_newsapi_disabled.py` also checks every tracked file for the exposed key **by fingerprint**, so the test never contains the key |

### Keeping your local copy of `.streamlit/secrets.toml`

When a branch that *untracks* the file is checked out or merged into a
working copy where the file is still tracked (for example `main` or
`feature/prediction-v2`), **Git deletes the working file**. If you want to
keep it, copy it outside the repository before switching or merging, and
copy it back afterwards; it is then ignored. Nothing in StockLens needs this
file.

## 3. Getting the key revoked (owner)

Self-service revocation is not available to you. Options, most effective
first:

1. **Recover the account, then regenerate the key.** Use the password reset
   on newsapi.org's login page with the e-mail address the key was
   registered with. Logged-in users see their key in the documentation; the
   account page is where a new key can be generated, if your plan offers it.
2. **E-mail NewsAPI support** at `support@newsapi.org` (listed on
   newsapi.org; paid plans advertise e-mail support):
   - send it **from the e-mail address the account was registered with**;
   - ask them to **disable or regenerate** the key, or close the account;
   - identify the key **without sending it again**: the account e-mail, and
     the public commit URLs where it is exposed (for example
     `https://github.com/Himanshu11007/stock-prediction-app/blob/c9ca6e2/.streamlit/secrets.toml`),
     which they can read themselves;
   - say it was exposed in a public repository and that you cannot revoke
     it yourself;
   - keep the reply. NewsAPI documents an `apiKeyDisabled` error, so it can
     disable keys on its side.
3. **If support does not respond:**
   - ask NewsAPI to **delete the account**, which invalidates its keys;
   - do not use that account for anything again. StockLens never needs it.
4. **Verify** once support confirms:
   `python scripts/security/check_exposed_newsapi_key.py --confirm-read-exposed-key`
   must print `REVOKED (HTTP 401 …)` and exit 0. Until then, no document may
   call the key revoked.

**Until it is revoked:**
- the risk is to the **NewsAPI account**: quota use, or use that breaches
  NewsAPI's terms in your name;
- it is not a risk to StockLens's data or users, because StockLens does not
  use the key.

## 4. Git history: prepared, NOT executed (needs separate approval)

Rewriting history changes every commit hash from `bcf4da6` onwards on all
branches, needs a force-push, and invalidates every clone, open review and
commit link.

Because the repository is public and the key cannot be revoked yet, a
rewrite **reduces further exposure but does not remove copies already made**
(clones, forks, caches, scrapers). It is worth considering **because**
revocation is unavailable, but only with your explicit approval.

Options, least to most invasive:
1. **Make the repository private** (owner, GitHub settings). This stops new
   viewers immediately, with no history change. Existing copies remain.
2. **Rewrite history** on a fresh mirror clone, never the working copy:
   ```bash
   git clone --mirror https://github.com/Himanshu11007/stock-prediction-app.git sp-mirror.git
   cd sp-mirror.git
   git filter-repo --invert-paths --path .streamlit/secrets.toml
   # replacements.txt OUTSIDE the repository, one line "<the key>==>REMOVED_NEWSAPI_KEY", deleted afterwards:
   git filter-repo --replace-text ../replacements.txt
   git log --all -S "<the key>" --oneline       # expect no output
   # ONLY with approval:
   git push --force --mirror
   ```
   Afterwards:
   - every collaborator re-clones;
   - open pull requests are recreated;
   - ask GitHub Support to purge cached views of the old commits;
   - forks keep the old history.

## 5. Related credentials (owner)

- **Production database password:** rotation still pending.
- **Marketaux and Alpha Vantage keys:** Windows user environment variables
  only; not present in any repository file (scan of 11 Oct 2026).
