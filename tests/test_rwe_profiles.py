"""
Tests for FASE 13-14 — RWE profiles + Testimonianze endpoints.

Covers:
  - /rwe/extract: no key → error, happy path, dedup by external_id
  - /rwe/profiles: list, filter by source/treatment
  - /rwe/testimonianze: only community_forum profiles appear
  - /rwe/testimonianze/{id}/curate: 404 for missing, 400 for FAERS, 200 for forum
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest


def _ok_profile():
    return json.dumps({
        "experience_summary": "User reports shedding after finasteride.",
        "key_quotes": ["my hair started shedding"],
        "treatment": "finasteride",
        "condition": "androgenetic alopecia",
        "adverse_events": ["hair shedding"],
        "outcome": "unknown",
        "experience_type": "adverse_event",
        "extraction_confidence": "high",
        "demographics": {"age": None, "sex": "male", "country": None},
    })


def _forum_body(**kw):
    base = {
        "title": "My shedding experience",
        "text": "I started finasteride and my hair started shedding after 2 weeks.",
        "source": "reddit",
        "source_type": "community_forum",
        "evidence_tier": "anecdotal",
        "external_id": "t3_abc123",
        "treatment": "finasteride",
        "language": "en",
    }
    base.update(kw)
    return base


class TestRWEExtract:
    def test_no_key_returns_error(self, client):
        resp = client.post("/rwe/extract", json=_forum_body())
        assert resp.status_code == 200
        assert resp.json()["error"] == "OPENAI_API_KEY not set."

    def test_happy_path_persists_profile(self, client, monkeypatch, db_session):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = _mock_resp(_ok_profile())
            resp = client.post("/rwe/extract", json=_forum_body())

        data = resp.json()
        assert data["already_existed"] is False
        assert data["episode_id"].startswith("rwe-")
        assert data["source"] == "reddit"
        prof = data["extracted_profile"]
        assert prof["treatment"] == "finasteride"
        assert prof["experience_type"] == "adverse_event"

    def test_dedup_by_external_id(self, client, monkeypatch, db_session):
        """Second extract with same external_id returns already_existed."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = _mock_resp(_ok_profile())
            client.post("/rwe/extract", json=_forum_body())
            # Second call — should NOT call LLM again
            resp2 = client.post("/rwe/extract", json=_forum_body())

        data = resp2.json()
        assert data["already_existed"] is True

    def test_llm_called_once_for_new_item(self, client, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = _mock_resp(_ok_profile())
            client.post("/rwe/extract", json=_forum_body())
        assert mock_openai.return_value.chat.completions.create.call_count == 1


class TestRWEProfilesList:
    def test_empty_list(self, client):
        resp = client.get("/rwe/profiles")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_list_after_extract(self, client, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = _mock_resp(_ok_profile())
            client.post("/rwe/extract", json=_forum_body())

        resp = client.get("/rwe/profiles")
        data = resp.json()
        assert data["count"] == 1
        assert data["profiles"][0]["source"] == "reddit"
        assert data["profiles"][0]["is_testimonial"] is False

    def test_filter_by_source(self, client, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = _mock_resp(_ok_profile())
            client.post("/rwe/extract", json=_forum_body(source="reddit"))
            client.post("/rwe/extract", json=_forum_body(
                source="calvizie", external_id="calv-1", text="post in italian"
            ))

        resp = client.get("/rwe/profiles?source=calvizie")
        data = resp.json()
        assert data["count"] == 1
        assert data["profiles"][0]["source"] == "calvizie"


class TestTestimonianze:
    def test_empty_testimonials(self, client):
        resp = client.get("/rwe/testimonianze")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_curate_forum_profile(self, client, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = _mock_resp(_ok_profile())
            r = client.post("/rwe/extract", json=_forum_body())
        eid = r.json()["episode_id"]

        # Curate it
        cur = client.post(f"/rwe/testimonianze/{eid}/curate")
        assert cur.status_code == 200
        assert cur.json()["is_testimonial"] is True

        # Now appears in testimonials
        resp = client.get("/rwe/testimonianze")
        data = resp.json()
        assert data["count"] == 1
        assert data["testimonials"][0]["episode_id"] == eid

    def test_curate_faers_rejected(self, client, monkeypatch):
        """FAERS records cannot be testimonials."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = _mock_resp(_ok_profile())
            r = client.post("/rwe/extract", json=_forum_body(
                source="openfda_faers",
                source_type="pharmacovigilance",
                evidence_tier="spontaneous_report",
                external_id="FAERS-1",
            ))
        eid = r.json()["episode_id"]

        cur = client.post(f"/rwe/testimonianze/{eid}/curate")
        assert cur.status_code == 400
        assert "community_forum" in cur.json()["detail"]

        # Does NOT appear in testimonials
        resp = client.get("/rwe/testimonianze")
        assert resp.json()["count"] == 0

    def test_curate_missing_returns_404(self, client):
        resp = client.post("/rwe/testimonianze/rwe-nonexistent/curate")
        assert resp.status_code == 404


# ── Helper ───────────────────────────────────────────────────────────────────

def _mock_resp(content):
    m = type("M", (), {"content": content})()
    c = type("C", (), {"message": m})()
    return type("R", (), {"choices": [c]})()
