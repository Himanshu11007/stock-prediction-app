"""
admin_console/embedded.py — the Admin Console as the "Administration" section
of the main Streamlit app (app.py, st.navigation).

Same console, same pages (admin_console/app.py HANDLERS); nothing is
duplicated. Access:

  - Before sign-in the section has one entry, "Admin Console" (sign-in form).
    The admin pages are not registered, so their URLs do not exist.
  - Sign-in goes through the backend (POST /auth/login + GET /auth/me) and is
    accepted only for accounts with the ADMIN role (AdminApiClient.login).
  - Every admin page reads and writes through the REST API with that user's
    token; the backend's require_admin check is the authorization boundary.
    Streamlit holds no database URL, no secret and no business logic.

The API address comes from configuration, never from the sign-in form:
STOCKAI_API_URL (environment, or Streamlit secrets on Streamlit Community
Cloud), else the production API below.
"""
from __future__ import annotations

import os
import re
from typing import Callable

import streamlit as st

from admin_console import app as console
from config import PRODUCT_NAME

SECTION = "Administration"
PRODUCTION_API_URL = "https://stocklens-api-otrc.onrender.com/api/v1"
SIGN_IN_PATH = "admin"


def api_url() -> str:
    url = os.environ.get("STOCKAI_API_URL")
    if not url:
        try:
            url = st.secrets.get("STOCKAI_API_URL")
        except Exception:                   # no secrets file configured
            url = None
    return (url or PRODUCTION_API_URL).rstrip("/")


def is_admin() -> bool:
    me = st.session_state.get("me")
    return bool(me) and "ADMIN" in (me.get("roles") or [])


def _slug(name: str) -> str:
    return "admin-" + re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _configure() -> None:
    st.set_page_config(page_title=f"{PRODUCT_NAME} - Admin", page_icon=str(console.BRAND_ICON), layout="wide",
                       initial_sidebar_state="expanded")


def _sign_in() -> None:
    _configure()
    url = api_url()
    if st.session_state.get("api_url") != url:     # never reuse a client for another API
        console.sign_out()
        st.session_state.api_url = url
    console.login_page(api_url=url)
    st.caption(f"Backend: {url}")


def _admin_page(name: str) -> Callable[[], None]:
    def render() -> None:
        _configure()
        if not is_admin():                  # defence in depth; the page is not registered for non-admins
            st.error("Sign in with an ADMIN account to open the Admin Console.")
            st.stop()
        with st.sidebar:
            st.caption(f"Signed in as {st.session_state.me.get('email') or st.session_state.me.get('id')}")
        st.title(name)
        console.HANDLERS[name](console.client())
    render.__name__ = _slug(name).replace("-", "_")
    return render


def _sign_out() -> None:
    _configure()
    console.sign_out()
    st.success("Signed out of the Admin Console.")
    st.rerun()


def pages() -> list:
    """Navigation entries of the Administration section for this session."""
    if not is_admin():
        return [st.Page(_sign_in, title="Admin Console", icon=":material/admin_panel_settings:",
                        url_path=SIGN_IN_PATH)]
    entries = []
    for name in console.PAGES:
        # The dashboard keeps the sign-in URL, so signing in lands on it.
        path = SIGN_IN_PATH if name == "Admin Dashboard" else _slug(name)
        entries.append(st.Page(_admin_page(name), title=name, url_path=path))
    entries.append(st.Page(_sign_out, title="Sign out", icon=":material/logout:", url_path="admin-sign-out"))
    return entries
