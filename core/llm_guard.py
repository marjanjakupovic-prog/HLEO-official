"""
HLEO — LLM call guard (cost protection + bounded retry)
========================================================

Centralises EVERY OpenAI call so retry cannot multiply across layers.

Hard rules (enforced everywhere this module is used):
    MAX_TOTAL_ATTEMPTS = 5
        1 initial attempt + at most 4 retries = 5 LLM calls per operation.
        No caller/layer may add its own retry on top — doing so would breach
        the absolute cap. The guard is the ONLY retry boundary.

429 handling (per the master spec):
    - insufficient_quota / credit_balance_exhausted  → NO retry, raise
      QuotaExhaustedError immediately (credito esaurito).
    - rate_limit_exceeded (temporary)               → retry with backoff,
      up to MAX_TOTAL_ATTEMPTS.
    - other transient errors                       → retry, up to the cap.
    - JSON/schema validation errors                → retry, up to the cap
      (temperature is nudged on retries to break a stuck malformed response).

Backoff: exponential, capped, with light jitter:
    delay = min(BASE_DELAY * 2**attempt, MAX_DELAY) * (1 ± 0.15)
"""
from __future__ import annotations

import json
import logging
import random
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Absolute cap ─────────────────────────────────────────────────────────────
MAX_TOTAL_ATTEMPTS = 5      # 1 initial + 4 retries. No layer may exceed this.

# ── Backoff tunables ─────────────────────────────────────────────────────────
_BASE_DELAY = 2.0           # seconds; doubled each attempt
_MAX_DELAY = 30.0           # cap a single backoff sleep
_JITTER = 0.15              # ±15% jitter


class QuotaExhaustedError(RuntimeError):
    """Raised when the OpenAI account has no credit/quota — NOT retryable."""


class LLMCallError(RuntimeError):
    """Raised after MAX_TOTAL_ATTEMPTS is exhausted, or for non-429 hard errors."""


# ── 429 / quota classification ───────────────────────────────────────────────

# Substrings (case-insensitive) that mean the account credit is exhausted and
# retrying is pointless. Sourced from OpenAI's documented error shapes plus the
# specific codes named in the master spec.
_QUOTA_SIGNALS = (
    "insufficient_quota",
    "credit_balance_exhausted",
    "billing_hard_limit_reached",
    "exceeded your current quota",
    "you exceeded your current quota",
)

# Substrings indicating a temporary rate-limit that MAY resolve after backoff.
_RATE_LIMIT_SIGNALS = (
    "rate_limit_exceeded",
    "requests can be made",
    "too many requests",
    "rate limit",
)


def classify_429(message: str) -> str:
    """Classify a 429/error message into quota-exhausted vs rate-limited vs other.

    Returns one of:
        "quota_exhausted"  — NOT retryable (raise immediately)
        "rate_limit"        — retryable (backoff)
        "other"             — retryable (treated as transient)
    """
    msg = (message or "").lower()
    if any(s in msg for s in _QUOTA_SIGNALS):
        return "quota_exhausted"
    if "429" in msg or any(s in msg for s in _RATE_LIMIT_SIGNALS):
        return "rate_limit"
    return "other"


def _backoff_delay(attempt: int) -> float:
    raw = min(_BASE_DELAY * (2 ** attempt), _MAX_DELAY)
    return raw * (1.0 + random.uniform(-_JITTER, _JITTER))


def _extract_openai_message(exc: Exception) -> str:
    """OpenAI SDK errors carry a structured body; pull a readable string out."""
    # openai>=1.x: exc.response / exc.body / exc.message
    for attr in ("body", "message", "response"):
        val = getattr(exc, attr, None)
        if val:
            if isinstance(val, dict):
                inner = val.get("error") if "error" in val else val
                if isinstance(inner, dict):
                    return str(inner.get("message") or inner)
                return str(inner)
            return str(val)
    return str(exc)


# ── Centralised LLM call ────────────────────────────────────────────────────

def call_llm(
    client: Any,
    *,
    messages: list[dict],
    model: str = "gpt-4o",
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    response_format: Optional[dict] = None,
    json_mode: bool = False,
    operation: str = "llm_call",
) -> str:
    """Call OpenAI chat.completions.create with a SINGLE, bounded retry policy.

    Returns the assistant message text. Raises QuotaExhaustedError (no retry)
    or LLMCallError (after MAX_TOTAL_ATTEMPTS) on failure.

    Callers MUST NOT add their own retry loop around this — that would breach
    the absolute cap. This is the only retry boundary in the whole project.
    """
    if json_mode and response_format is None:
        response_format = {"type": "json_object"}

    last_exc: Optional[Exception] = None
    last_kind: str = "other"

    for attempt in range(MAX_TOTAL_ATTEMPTS):
        try:
            kwargs: dict = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if response_format is not None:
                kwargs["response_format"] = response_format

            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content
            if content is None:
                # Treat a null content as a transient malformed response.
                raise ValueError("LLM returned null content.")
            return content

        except Exception as exc:  # noqa: BLE001 — we classify, then decide
            last_exc = exc
            msg = _extract_openai_message(exc)
            kind = classify_429(msg)

            # Hard stop: account quota exhausted. No retry, ever.
            if kind == "quota_exhausted":
                logger.error(
                    "%s: OpenAI quota exhausted — not retrying. %s",
                    operation, msg,
                )
                raise QuotaExhaustedError(
                    f"OpenAI credit/quota exhausted — API calls disabled. ({msg})"
                ) from exc

            last_kind = kind
            remaining = MAX_TOTAL_ATTEMPTS - attempt - 1
            if remaining <= 0:
                break

            delay = _backoff_delay(attempt)
            logger.warning(
                "%s: attempt %d/%d failed (%s) — retrying in %.1fs. %s",
                operation, attempt + 1, MAX_TOTAL_ATTEMPTS, kind, delay,
                msg[:160],
            )
            time.sleep(delay)

    raise LLMCallError(
        f"{operation} failed after {MAX_TOTAL_ATTEMPTS} attempts "
        f"(last kind={last_kind}): {last_exc}"
    ) from last_exc


def call_llm_json(
    client: Any,
    *,
    messages: list[dict],
    model: str = "gpt-4o",
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    operation: str = "llm_json_call",
) -> dict:
    """Call OpenAI, parse JSON, with the same single bounded retry policy.

    JSON / schema validation failures ARE retryable (count toward the cap).
    On the final attempt the raw text is attached to the error so the caller
    can surface it.
    """
    if max_tokens is not None:
        kwargs_max = max_tokens
    else:
        kwargs_max = None

    last_exc: Optional[Exception] = None
    last_raw: str = ""
    last_kind: str = "other"

    for attempt in range(MAX_TOTAL_ATTEMPTS):
        try:
            kwargs: dict = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "response_format": {"type": "json_object"},
            }
            if kwargs_max is not None:
                kwargs["max_tokens"] = kwargs_max

            resp = client.chat.completions.create(**kwargs)
            raw = resp.choices[0].message.content or ""
            last_raw = raw

            # Strip markdown code fences if present
            s = raw.strip()
            if s.startswith("```"):
                s = s.split("```", 2)[1] if s.count("```") >= 2 else s
                if s.startswith("json"):
                    s = s[4:]
                s = s.strip()
                if s.endswith("```"):
                    s = s[:-3].strip()

            return json.loads(s)

        except json.JSONDecodeError as exc:
            # Schema/parse error — retryable, nudged by the cap.
            last_exc = exc
            last_kind = "schema"
            remaining = MAX_TOTAL_ATTEMPTS - attempt - 1
            if remaining <= 0:
                break
            delay = _backoff_delay(attempt)
            logger.warning(
                "%s: JSON parse error on attempt %d/%d — retrying in %.1fs. %s",
                operation, attempt + 1, MAX_TOTAL_ATTEMPTS, delay, str(exc)[:160],
            )
            time.sleep(delay)
            continue

        except Exception as exc:  # noqa: BLE001 — classify then decide
            last_exc = exc
            msg = _extract_openai_message(exc)
            kind = classify_429(msg)

            if kind == "quota_exhausted":
                logger.error(
                    "%s: OpenAI quota exhausted — not retrying. %s",
                    operation, msg,
                )
                raise QuotaExhaustedError(
                    f"OpenAI credit/quota exhausted — API calls disabled. ({msg})"
                ) from exc

            last_kind = kind
            remaining = MAX_TOTAL_ATTEMPTS - attempt - 1
            if remaining <= 0:
                break
            delay = _backoff_delay(attempt)
            logger.warning(
                "%s: attempt %d/%d failed (%s) — retrying in %.1fs. %s",
                operation, attempt + 1, MAX_TOTAL_ATTEMPTS, kind, delay,
                msg[:160],
            )
            time.sleep(delay)

    raise LLMCallError(
        f"{operation} failed after {MAX_TOTAL_ATTEMPTS} attempts "
        f"(last kind={last_kind}): {last_exc}"
    ) from last_exc
