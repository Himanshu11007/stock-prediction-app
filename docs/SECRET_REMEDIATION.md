# Exposed NewsAPI key: status and remediation plan (11 Oct 2026)

This file never contains the key. It is identified by its length (32
characters) and a fingerprint (`sha256` prefix `c5d151c4f231`).

## 1. Status, verified 11 Oct 2026

| Item | Status |
|---|---|
| Key validity | **STILL ACTIVE.** `python scripts/security/check_exposed_newsapi_key.py` returned `STILL ACTIVE (HTTP 200)`, exit 1. **Not revoked** |
| Repository | `Himanshu11007/stock-prediction-app` is **public**: assume the key is compromised |
| Where it is tracked today | `.streamlit/secrets.toml` (single entry `API_KEY`) on `origin/main` and every review branch. No code reads it: the NewsAPI adapter uses `NEWSAPI_KEY` from the environment |
| First introduced | `bcf4da6` (2026-05-10), hardcoded in `news/api.py`; moved to `.streamlit/secrets.toml` in `c9ca6e2` (2026-05-17) |
| Commits that add or remove it (`git log --all -S`) | `bcf4da6`, `c9ca6e2`, `1981816`, `9970acd`, `261f8f0`, `96051e6`, `f11e31b` |
| Remote branches whose history contains it | `main` and every branch on GitHub: `claude/stocklens-auth-security-hardening-qczhnr` and the four `review/*` branches |

## 2. Done on the local branch `review/independent-review-prep` (not pushed)

- `.streamlit/secrets.toml` is **untracked** (`git rm --cached`). The local
  copy remains on disk and is ignored by Git.
- `.gitignore` additionally ignores `.env`, `.env.*` (except
  `.env.example`), `*.pem`, `*.p12`, `*.pfx` and `client_secret*.json`.
- `scripts/security/scan_secrets.py` fails if any tracked file is a secret
  file or contains a credential. It runs in the test suite
  (`tests/test_secret_scan.py`) and in CI (`.github/workflows/ci.yml`).
  - It prints the file, line and rule only, never a value.
  - `--history` also scans every commit reachable from HEAD.

**Untracking does not make the key safe.** It is still in every commit
listed above, and anyone who cloned or forked the public repository has it.

## 3. Required order (owner)

1. **Revoke or regenerate the key** at newsapi.org (account > API key).
   This is the only step that removes the risk.
2. **Verify:** `python scripts/security/check_exposed_newsapi_key.py` must
   print `REVOKED (HTTP 401 apiKeyInvalid)` and exit 0. Until then, no
   document may call the key revoked.
3. If NewsAPI is still wanted, create a **new** key and keep it only in the
   environment (`NEWSAPI_KEY`), never in a file in the repository.
4. Approve the push of this branch, so the untracking and the scanner reach
   GitHub.
5. Optional: rewrite history (step 4 below). Only after revocation, and
   only with explicit approval.

## 4. History rewrite: prepared, NOT executed (needs separate approval)

A rewrite changes every commit hash from `bcf4da6` onwards on all branches.
It needs a force-push and invalidates every existing clone, open review and
commit link. Because the repository is public, it reduces future exposure
but cannot undo past exposure. Revocation is mandatory either way.

```bash
# on a fresh mirror clone, never the working copy
git clone --mirror https://github.com/Himanshu11007/stock-prediction-app.git sp-mirror.git
cd sp-mirror.git
# 1) drop the secrets file from every commit
git filter-repo --invert-paths --path .streamlit/secrets.toml
# 2) replace the hardcoded value in news/api.py (bcf4da6..c9ca6e2) with a placeholder:
#    put ONE line "<the revoked key>==>REMOVED_NEWSAPI_KEY" in a file outside the repository,
#    then (delete that file afterwards):
git filter-repo --replace-text ../replacements.txt
# 3) verify: no commit contains it any more (expect no output)
git log --all -S "<the revoked key>" --oneline
python scripts/security/scan_secrets.py --history     # run from a checkout of the rewritten mirror
# 4) ONLY with approval: git push --force --mirror
```

After an approved force-push:
- every collaborator re-clones;
- open pull requests are recreated;
- GitHub support can be asked to purge cached views of the old commits;
- forks keep the old history and cannot be cleaned by the owner.

## 5. Related credentials (owner)

- **Production database password:** it was exposed in an earlier session;
  rotation is still pending.
- **Marketaux and Alpha Vantage keys:** stored only as Windows user
  environment variables on the owner's PC. A scan of both repositories
  (11 Oct 2026) found neither key value in any file.
