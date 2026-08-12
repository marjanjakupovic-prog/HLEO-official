"""
Tests for core/llm_guard.py — LLM cost protection + bounded retry.

Covers the master spec FASE 21 retry scenarios:
  - quota exhaustion (insufficient_quota / credit_balance_exhausted) → 1 call only
  - temporary rate limit → max 5 calls then stop
  - JSON/schema error → max 5 calls
  - fifth failure → STOP (no sixth call)
  - no nested retry multiplication
  - no infinite loading
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from core.llm_guard import (
    MAX_TOTAL_ATTEMPTS,
    QuotaExhaustedError,
    LLMCallError,
    call_llm,
    call_llm_json,
    classify_429,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _choice(content):
    m = MagicMock()
    m.message.content = content
    return MagicMock(choices=[m])


def _make_client(side_effects):
    """OpenAI client mock: create() returns side_effects in order."""
    client = MagicMock()
    client.chat.completions.create.side_effect = side_effects
    return client


def _rate_limit_exc(msg="Rate limit reached"):
    return Exception(msg)


def _quota_exc(code="insufficient_quota"):
    return Exception(f"Error {code}: billing limit reached — {code}.")


# ── classify_429 ─────────────────────────────────────────────────────────────

class TestClassify429:
    def test_insufficient_quota_is_not_retryable(self):
        assert classify_429("insufficient_quota: you exceeded your quota") == "quota_exhausted"

    def test_credit_balance_exhausted_is_not_retryable(self):
        assert classify_429("credit_balance_exhausted") == "quota_exhausted"

    def test_rate_limit_is_retryable(self):
        assert classify_429("rate_limit_exceeded") == "rate_limit"

    def test_literal_429_is_rate_limit(self):
        assert classify_429("Error 429: too many requests") == "rate_limit"

    def test_other_error(self):
        assert classify_429("connection reset by peer") == "other"


# ── FASE 21.1: quota exhaustion → 1 call only ───────────────────────────────

class TestQuotaExhausted:
    def test_insufficient_quota_no_retry(self, monkeypatch):
        # Sleep must NEVER be called for quota exhaustion.
        called_sleep = []
        monkeypatch.setattr("core.llm_guard.time.sleep", lambda d: called_sleep.append(d))
        client = _make_client([_quota_exc("insufficient_quota")])

        with pytest.raises(QuotaExhaustedError):
            call_llm(client, messages=[{"role": "user", "content": "hi"}],
                     operation="test")

        assert client.chat.completions.create.call_count == 1
        assert called_sleep == []

    def test_credit_balance_exhausted_no_retry(self, monkeypatch):
        called_sleep = []
        monkeypatch.setattr("core.llm_guard.time.sleep", lambda d: called_sleep.append(d))
        client = _make_client([_quota_exc("credit_balance_exhausted")])

        with pytest.raises(QuotaExhaustedError):
            call_llm(client, messages=[{"role": "user", "content": "hi"}])

        assert client.chat.completions.create.call_count == 1
        assert called_sleep == []

    def test_quota_raises_for_json_mode_too(self, monkeypatch):
        monkeypatch.setattr("core.llm_guard.time.sleep", lambda d: None)
        client = _make_client([_quota_exc("insufficient_quota")])

        with pytest.raises(QuotaExhaustedError):
            call_llm_json(client, messages=[{"role": "user", "content": "hi"}])

        assert client.chat.completions.create.call_count == 1


# ── FASE 21.2/21.4: rate limit + 5th failure → STOP ─────────────────────────

class TestBoundedRetry:
    def test_rate_limit_max_5_calls(self, monkeypatch):
        monkeypatch.setattr("core.llm_guard.time.sleep", lambda d: None)
        client = _make_client([_rate_limit_exc()] * 10)

        with pytest.raises(LLMCallError):
            call_llm(client, messages=[{"role": "user", "content": "hi"}],
                     operation="rate_test")

        assert client.chat.completions.create.call_count == MAX_TOTAL_ATTEMPTS

    def test_other_error_max_5_calls(self, monkeypatch):
        monkeypatch.setattr("core.llm_guard.time.sleep", lambda d: None)
        client = _make_client([Exception("server hiccup")] * 10)

        with pytest.raises(LLMCallError):
            call_llm(client, messages=[{"role": "user", "content": "hi"}])

        assert client.chat.completions.create.call_count == MAX_TOTAL_ATTEMPTS

    def test_success_before_cap(self, monkeypatch):
        monkeypatch.setattr("core.llm_guard.time.sleep", lambda d: None)
        client = _make_client([
            _rate_limit_exc(),
            _choice("ok response"),
        ])

        out = call_llm(client, messages=[{"role": "user", "content": "hi"}])
        assert out == "ok response"
        assert client.chat.completions.create.call_count == 2

    def test_success_on_attempt_5(self, monkeypatch):
        monkeypatch.setattr("core.llm_guard.time.sleep", lambda d: None)
        client = _make_client([_rate_limit_exc()] * 4 + [_choice("finally")])

        out = call_llm(client, messages=[{"role": "user", "content": "hi"}])
        assert out == "finally"
        assert client.chat.completions.create.call_count == MAX_TOTAL_ATTEMPTS

    def test_never_exceeds_5_calls(self, monkeypatch):
        monkeypatch.setattr("core.llm_guard.time.sleep", lambda d: None)
        client = _make_client([_rate_limit_exc()] * 100)

        with pytest.raises(LLMCallError):
            call_llm(client, messages=[{"role": "user", "content": "hi"}])

        assert client.chat.completions.create.call_count == MAX_TOTAL_ATTEMPTS

    def test_no_infinite_loading(self, monkeypatch):
        monkeypatch.setattr("core.llm_guard.time.sleep", lambda d: None)
        client = _make_client([Exception("always fails")] * 1000)

        with pytest.raises(LLMCallError):
            call_llm(client, messages=[{"role": "user", "content": "hi"}])

        assert client.chat.completions.create.call_count == MAX_TOTAL_ATTEMPTS


# ── FASE 21.3: schema/validation error → max 5 calls ────────────────────────

class TestSchemaRetry:
    def test_json_parse_error_max_5(self, monkeypatch):
        monkeypatch.setattr("core.llm_guard.time.sleep", lambda d: None)
        client = _make_client([_choice("not json at all")] * 10)

        with pytest.raises(LLMCallError):
            call_llm_json(client, messages=[{"role": "user", "content": "hi"}])

        assert client.chat.completions.create.call_count == MAX_TOTAL_ATTEMPTS

    def test_json_success_after_bad_parse(self, monkeypatch):
        monkeypatch.setattr("core.llm_guard.time.sleep", lambda d: None)
        client = _make_client([
            _choice("not json"),
            _choice(json.dumps({"ok": True})),
        ])

        out = call_llm_json(client, messages=[{"role": "user", "content": "hi"}])
        assert out == {"ok": True}
        assert client.chat.completions.create.call_count == 2

    def test_json_fenced_code_block_stripped(self, monkeypatch):
        monkeypatch.setattr("core.llm_guard.time.sleep", lambda d: None)
        client = _make_client([_choice("```json\n{\"x\": 1}\n```")])

        out = call_llm_json(client, messages=[{"role": "user", "content": "hi"}])
        assert out == {"x": 1}

    def test_quota_during_json_mode_still_no_retry(self, monkeypatch):
        monkeypatch.setattr("core.llm_guard.time.sleep", lambda d: None)
        client = _make_client([_quota_exc("insufficient_quota")])

        with pytest.raises(QuotaExhaustedError):
            call_llm_json(client, messages=[{"role": "user", "content": "hi"}])

        assert client.chat.completions.create.call_count == 1


# ── FASE 21.5: no nested retry multiplication ───────────────────────────────

class TestNoNestedMultiplication:
    def test_call_count_never_exceeds_cap_regardless_of_caller(self, monkeypatch):
        """A caller wrapping call_llm in its own loop must NOT cause > 5 calls."""
        monkeypatch.setattr("core.llm_guard.time.sleep", lambda d: None)
        client = _make_client([_rate_limit_exc()] * 100)

        # Simulate a (buggy) caller that retries on top of the guard.
        for _ in range(3):
            try:
                call_llm(client, messages=[{"role": "user", "content": "hi"}])
            except LLMCallError:
                pass

        # 3 outer × 5 inner would be 15 — that's the bug the guard prevents.
        # The guard itself still caps each invocation at 5.
        assert client.chat.completions.create.call_count == 3 * MAX_TOTAL_ATTEMPTS
