"""
scripts/security/check_exposed_newsapi_key.py — is the NewsAPI key that was
committed to the public repository still valid? Prints only the verdict and
the HTTP status; never the key.

  python scripts/security/check_exposed_newsapi_key.py

The key is read from the Git object where it was committed
(c9ca6e2:.streamlit/secrets.toml, the API_KEY entry). Run it after revoking
or regenerating the key at newsapi.org: the expected result is
"REVOKED (HTTP 401 apiKeyInvalid)". Exit code 0 = revoked, 1 = still active,
2 = could not check.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import urllib.error
import urllib.request

COMMIT_PATH = "c9ca6e2:.streamlit/secrets.toml"


def main() -> int:
    text = subprocess.run(["git", "show", COMMIT_PATH], capture_output=True, text=True).stdout
    m = re.search(r'(?m)^\s*API_KEY\s*=\s*"([^"]+)"', text)
    if not m:
        print("could not find the committed key (git object unavailable)")
        return 2
    req = urllib.request.Request("https://newsapi.org/v2/top-headlines?country=us&pageSize=1",
                                 headers={"X-Api-Key": m.group(1), "User-Agent": "StockLens-security-check"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            print(f"STILL ACTIVE (HTTP {r.status}): revoke or regenerate the key at newsapi.org now")
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
