"""
scripts/security/scan_secrets.py — fail when a tracked file contains a
credential or when a secret file is tracked at all. Safe to run anywhere:
it prints the file, line and rule only, never the value.

  python scripts/security/scan_secrets.py            # exit 0 clean, 1 findings
  python scripts/security/scan_secrets.py --history  # also every commit reachable from HEAD (slower)

Runs in the test suite (tests/test_secret_scan.py) and in CI. With
--history it reports commits that once contained a secret, which only a
history rewrite (owner approval) or key revocation can address.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Files that must never be tracked, whatever they contain.
FORBIDDEN_PATHS = (
    re.compile(r"(^|/)\.streamlit/secrets\.toml$"),
    re.compile(r"(^|/)\.env(\.[\w-]+)?$"),
    re.compile(r"(^|/)[^/]*\.(pem|p12|pfx|key)$"),
    re.compile(r"(^|/)(id_rsa|id_ed25519)$"),
    re.compile(r"(^|/)client_secret[^/]*\.json$"),
)
ALLOWED_PATHS = (re.compile(r"(^|/)\.env\.example$"),)

RULES: tuple[tuple[str, re.Pattern], ...] = (
    ("google-oauth-client-secret", re.compile(r"GOCSPX-[A-Za-z0-9_\-]{20,}")),
    ("aws-access-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("private-key-block", re.compile(r"-----BEGIN (RSA |EC |OPENSSH |DSA |)PRIVATE KEY-----")),
    ("github-token", re.compile(r"\b(ghp|gho|ghs|ghu|github_pat)_[A-Za-z0-9_]{30,}\b")),
    ("slack-token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("database-url-with-password", re.compile(r"\b(postgres(ql)?|mysql)(\+\w+)?://[^:\s/@]+:[^@\s/]{6,}@([\w.-]+)")),
    # name = "value" assignments of key / token / secret / password with a long opaque value
    ("credential-assignment", re.compile(
        r"""(?i)\b[\w.]*(api[_-]?key|api[_-]?token|secret[_-]?key|client[_-]?secret|access[_-]?token|auth[_-]?token|password)\b"""
        r"""\s*[:=]\s*["']([A-Za-z0-9_\-+/=]{20,})["']""")),
)
# Values that are clearly placeholders or test fixtures.
PLACEHOLDER = re.compile(r"(?i)(test|fake|dummy|example|sample|placeholder|changeme|secret-value|xxxx|0123456789|"
                         r"your[_-]|<|\$\{|abcdef)")
SKIP_SUFFIX = (".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".woff", ".woff2", ".ttf", ".zip", ".db", ".sqlite",
               ".parquet", ".pkl", ".joblib", ".xlsx")
# disposable local / CI-service databases: a password there is not a leaked credential
LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "postgres", "db", "host.docker.internal"}
LINE_ALLOW = "secret-scan: allow"          # explicit, reviewable per-line exception


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True,
                          encoding="utf-8", errors="replace").stdout


def scan_text(path: str, text: str) -> list[tuple[str, int, str]]:
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        if LINE_ALLOW in line:
            continue
        for rule, pat in RULES:
            for m in pat.finditer(line):
                if rule == "database-url-with-password" and m.group(m.lastindex).lower() in LOCAL_HOSTS:
                    continue
                value = m.group(m.lastindex) if rule == "credential-assignment" else m.group(0)
                if PLACEHOLDER.search(value) or (rule == "credential-assignment" and value.isalpha()):
                    continue
                out.append((path, n, rule))
    return out


def forbidden(path: str) -> bool:
    return any(p.search(path) for p in FORBIDDEN_PATHS) and not any(p.search(path) for p in ALLOWED_PATHS)


def scan_tracked() -> list[tuple[str, int, str]]:
    findings = []
    for path in _git("ls-files", "-z").split("\0"):
        if not path:
            continue
        if forbidden(path):
            findings.append((path, 0, "secret-file-tracked"))
            continue
        f = ROOT / path
        if path.lower().endswith(SKIP_SUFFIX) or not f.is_file() or f.stat().st_size > 5_000_000:
            continue
        findings += scan_text(path, f.read_text(encoding="utf-8", errors="replace"))
    return findings


def scan_history() -> list[tuple[str, str, str]]:
    """(commit, path, rule) for commits reachable from HEAD that tracked a
    secret file or added a matching line."""
    out = []
    for commit in _git("rev-list", "HEAD").split():
        for path in _git("show", "--pretty=format:", "--name-only", commit).split("\n"):
            if path and forbidden(path):
                out.append((commit[:10], path, "secret-file-tracked"))
        diff = _git("show", "--pretty=format:", "--unified=0", "--no-color", commit)
        current = ""
        for line in diff.splitlines():
            if line.startswith("+++ b/"):
                current = line[6:]
            elif line.startswith("+") and not line.startswith("+++"):
                for _, _, rule in scan_text(current, line[1:]):
                    out.append((commit[:10], current, rule))
    return sorted(set(out))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", action="store_true")
    args = ap.parse_args(argv)
    findings = scan_tracked()
    for path, line, rule in findings:
        print(f"{path}:{line}: {rule}")
    if args.history:
        hist = scan_history()
        for commit, path, rule in hist:
            print(f"history {commit} {path}: {rule}")
        print(f"{len(hist)} historical finding(s) (needs key revocation; history rewrite only with owner approval)")
    print(f"{len(findings)} finding(s) in tracked files")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
