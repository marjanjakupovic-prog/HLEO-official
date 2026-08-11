"""Shared pytest fixtures for HLEO backend tests.

Creates tables on an in-memory SQLite DB so the API can be exercised via
TestClient without a live Postgres instance. The .env in the repo holds a real
OPENAI_API_KEY (loaded by core.database.load_dotenv at import), so we blank it
again here to keep tests hermetic.
"""
import os
import sys

# Force SQLite in-memory BEFORE importing the app (core.database reads env at import).
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ.pop("OPENAI_API_KEY", None)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Neutralize load_dotenv before any core module re-imports it during a request.
# Several core modules (core.extractor, core.article_extractor, core.database) call
# load_dotenv() at import time, which would re-leak the real .env key into the
# process env mid-request and defeat the per-test env pops.
import dotenv
_real_load_dotenv = dotenv.load_dotenv
def _noop_load_dotenv(*a, **k):
    return False
dotenv.load_dotenv = _noop_load_dotenv
# Also patch already-imported references in modules that did `from dotenv import load_dotenv`.
for _modname in ("core.database", "core.extractor", "core.article_extractor"):
    try:
        _m = sys.modules.get(_modname)
        if _m is not None and hasattr(_m, "load_dotenv"):
            _m.load_dotenv = _noop_load_dotenv
    except Exception:
        pass

import pytest
from fastapi.testclient import TestClient

import api.main as appmod
from core.database import Base, engine, SessionLocal


@pytest.fixture(scope="session", autouse=True)
def _create_tables():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(autouse=True)
def _no_openai_key():
    """Ensure no real LLM key leaks into any test (load_dotenv is neutralized)."""
    os.environ.pop("OPENAI_API_KEY", None)
    yield
    os.environ.pop("OPENAI_API_KEY", None)


@pytest.fixture()
def client():
    return TestClient(appmod.app)


@pytest.fixture()
def db_session():
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()
