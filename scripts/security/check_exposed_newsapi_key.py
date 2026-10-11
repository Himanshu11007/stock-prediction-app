"""
scripts/security/check_exposed_newsapi_key.py — MANUAL security check: is the
NewsAPI key that was committed to the public repository still valid? It is
the only code that reads that key, and it is not part of the application.

  python scripts/security/check_exposed_newsapi_key.py --confirm-read-exposed-key

Without the flag it does nothing: no Git read, no network request (exit 3).
With it, the key is read from the Git object where it was committed
(c9ca6e2:.streamlit/secrets.toml, the API_KEY entry), sent once to NewsAPI
in a header, and only the verdict and the HTTP status are printed; never the
key. Expected after NewsAPI support revokes it: "REVOKED (HTTP 401
apiKeyInvalid)". Exit code 0 = revoked, 1 = still active, 2 = could not
check, 3 = not confirmed.

Never run it from CI, schedulers or tests.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request

COMMIT_PATH = "c9ca6e2:.streamlit/secrets.toml"
CONFIRM_FLAG = "--confirm-read-exposed-key"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Manual check: is the exposed NewsAPI key still valid?")
    ap.add_argument(CONFIRM_FLAG, dest="confirm", action="store_true",
                    help="read the exposed key from Git history and send one request to NewsAPI")
    args = ap.parse_args(argv)
    if not args.confirm:
        print(f"not run: this reads the exposed key and contacts NewsAPI; pass {CONFIRM_FLAG} to run it manually")
        return 3
    text = subprocess.run(["git", "show", COMMIT_PATH], capture_output=True, text=True).stdout
    m = re.search(r'(?m)^\s*API_KEY\s*=\s*"([^"]+)"', text)
    if not m:
        print("could not find the committed key (git object unavailable)")
        return 2
    req = urllib.request.Request("https://newsapi.org/v2/top-headlines?country=us&pageSize=1",
                                 headers={"X-Api-Key": m.group(1), "User-Agent": "StockLens-security-check"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            print(f"STILL ACTIVE (HTTP {r.status}): the key is not revoked; see docs/SECRET_REMEDIATION.md")
            return 1
    except urllib.error.HTTPError as e:
        try:
            code = json.loads(e.read() or b"{}").get("code")
        except ValueError:
            code = None
        if e.code == 401:
            print(f"REVOKED (HTTP 401 {code})")
            return 0
        print(f"could not determine (HTTP {e.code} {code})")
        return 2
    except OSError as e:
        print(f"could not check: {type(e).__name__}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
