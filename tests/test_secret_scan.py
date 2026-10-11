"""
tests/test_secret_scan.py — the repository tracks no secret file and no
credential (scripts/security/scan_secrets.py), and the scanner's rules catch
the credential shapes StockLens uses. Sample secrets are assembled at run
time so this file itself never matches. Values are never printed.
"""
from scripts.security import scan_secrets as ss


def test_repository_tracks_no_secret_files_or_credentials():
    findings = ss.scan_tracked()
    assert findings == [], f"secrets in tracked files (file:line:rule): {findings}"


def test_rules_catch_real_credential_shapes_and_skip_placeholders():
    hexkey = "c5d1" + "51c4f2" + "31aa" * 5 + "77"                 # 32 hex characters, like a NewsAPI key
    cases = {
        f'API_KEY = "{hexkey}"': "credential-assignment",
        "client_secret = 'GOC" + "SPX-" + "a1B2c3D4e5F6g7H8i9J0kLmN'": "google-oauth-client-secret",
        "url = 'postgresql://app:" + "s3cr3tPw9@" + "dpg-abc.oregon-postgres.render.com/db'": "database-url-with-password",
        "aws = 'AK" + "IA" + "Q7W2R9T4Y6U1I3O8'": "aws-access-key",
        "-----BEGIN " + "PRIVATE KEY-----": "private-key-block",
        "tok = 'gh" + "p_" + "a" * 36 + "'": "github-token",
    }
    for line, rule in cases.items():
        assert rule in {r for _, _, r in ss.scan_text("x.py", line)}, rule
    for line in ('api_key = "test-scheduler-token-0123456789abcdef"', 'password = "AdminPass1!"',
                 "TEST_DATABASE_URL=postgresql://stocklens:" + "localpw123@127.0.0.1/stocklens_test",
                 'API_KEY = "' + hexkey + '"  # secret-scan: allow', 'token = os.environ["NEWSAPI_KEY"]'):
        assert ss.scan_text("x.py", line) == [], line


def test_secret_files_are_forbidden_whatever_they_contain():
    for path in (".streamlit/secrets.toml", "app/.env", ".env.production", "deploy/server.pem", "client_secret_123.json"):
        assert ss.forbidden(path), path
    for path in (".env.example", "docs/SECRETS.md", "auth/token_issuance.py", ".streamlit/config.toml"):
        assert not ss.forbidden(path), path
