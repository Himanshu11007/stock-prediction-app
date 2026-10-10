"""
tests/test_terminology.py — user-facing text never shows the bare acronym
FQVF. It is the Fundamental Quality & Value Framework (docs/FQVF.md); users
see "Fundamental Quality & Value Score" / "Quality & Value", or the full name
followed by "(FQVF)". Internal fields, enum values (FQVF_CHANGE), versions
(fqvf-v1.0) and frozen Ranking v1 explanation strings are out of scope.
"""
import re

from analytics import intelligence_overview  # noqa: F401  (module import must work)
from masters.service import CONFIG_KEYS
from notifications.settings import DEFAULT_TEMPLATES, render
from ranking.presenter import fqvf_reference

ALLOWED = re.compile(r"Fundamental Quality & Value (Framework|Score) \(FQVF\)")


def bare(text: str) -> list[str]:
    return re.findall(r".{0,30}\bFQVF\b.{0,30}", ALLOWED.sub("", text))


def test_notification_templates_use_the_plain_name():
    for key, tpl in DEFAULT_TEMPLATES.items():
        assert not bare(tpl["title"] + " " + tpl["body"]), key
    title, body = render(DEFAULT_TEMPLATES, "FQVF_CHANGE", name="TCS", old_passes=10, new_passes=13)
    assert title == "Quality & Value checks update" and "Fundamental Quality & Value checks" in body


def test_onboarding_defaults_explain_the_acronym():
    onboarding = CONFIG_KEYS["app.onboarding"][0]
    text = " ".join(p["title"] + " " + p["body"] for p in onboarding)
    assert not bare(text)
    assert "Fundamental Quality & Value Score (FQVF)" in text


def test_intelligence_overview_and_reference_name_it_in_full():
    src = open(intelligence_overview.__file__, encoding="utf-8").read()
    titles = re.findall(r'"title": "([^"]+)"', src)
    assert titles and not bare(" ".join(titles))
    ref = fqvf_reference()
    assert ref["name"] == "Fundamental Quality & Value Framework" and ref["short_name"] == "FQVF"   # unchanged
    assert ref["display_name"] == "Fundamental Quality & Value Score"
    assert "does not predict" in ref["not_a_prediction"]
