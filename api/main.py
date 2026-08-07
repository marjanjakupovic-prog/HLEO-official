"""
HLEO v1.0 — FastAPI application
Endpoints:
  GET  /                     → dashboard UI
  GET  /health               → key/version status
  GET  /stats                → DB row counts
  GET  /search?q=            → fast collect from all sources (no LLM)
  POST /pipeline/run?q=      → collect + LLM-extract articles → DB
  GET  /profiles?limit=      → saved clinical profiles
  POST /experiences/ingest?q= → collect Reddit + LLM-extract patient experiences → DB
  GET  /experiences?limit=   → saved patient experiences
  POST /assistant/chat       → AI Clinical Assistant (RAG over DB)
  GET  /assistant/sessions/{session_id} → chat history
"""
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Depends, Query, Request, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import select, func, desc, or_
from sqlalchemy.orm import Session

from core.database import get_db, engine, Base
from core.models import (
    ClinicalProfile, RawSource, AuditLog,
    PatientExperience, SourceAttribution, ChatSession, ChatMessage,
)
from api.partners import router as rwe_router
from core.orchestrator import QueryOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Module-level orchestrator instance (stateless, safe to share across requests)
_orchestrator = QueryOrchestrator()

# Simple in-memory translation cache (avoids re-calling LLM for identical text+lang pairs)
_translate_cache: dict = {}

app = FastAPI(title="HLEO API", version="1.0.0")

# Allow the Replit preview proxy (and any origin) to send cross-origin POST/PATCH/DELETE
# requests with JSON bodies.  The preflight OPTIONS was returning 405 and blocking all
# fetch() calls from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)
app.include_router(rwe_router)

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_dict(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    try:
        return asdict(obj)
    except TypeError:
        return str(obj)


# ── Core routes ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health_check():
    import os
    key = os.getenv("OPENAI_API_KEY", "")
    return {
        "status": "ok",
        "version": "1.0.0",
        "openai_key_set": bool(key),
        "openai_key_prefix": key[:8] + "…" if key else None,
    }


@app.get("/stats")
def stats(db: Session = Depends(get_db)):
    return {
        "clinical_profiles": db.execute(
            select(func.count()).select_from(ClinicalProfile)
        ).scalar(),
        "patient_experiences": db.execute(
            select(func.count()).select_from(PatientExperience)
        ).scalar(),
        "raw_sources": db.execute(
            select(func.count()).select_from(RawSource)
        ).scalar(),
        "source_attributions": db.execute(
            select(func.count()).select_from(SourceAttribution)
        ).scalar(),
        "chat_sessions": db.execute(
            select(func.count()).select_from(ChatSession)
        ).scalar(),
    }


# ── Search (fast, no LLM) ─────────────────────────────────────────────────────

@app.get("/search")
def search(q: str = Query(..., description="Search query")):
    """Collect results from all sources — no LLM, returns raw data immediately."""
    from core.pipeline import HLEOPipeline

    # ── Orchestrate: detect language, translate to scientific English if needed ──
    orch = _orchestrator.process(q)

    pipeline = HLEOPipeline()
    raw = pipeline.collect(orch.search_query)

    pubmed         = [_to_dict(a) for a in raw["pubmed"]]
    europepmc      = [_to_dict(a) for a in raw["europepmc"]]
    clinicaltrials = [_to_dict(a) for a in raw["clinicaltrials"]]
    reddit_raw     = [_to_dict(p) for p in raw["reddit"]]

    return {
        "query":          q,
        "orchestration":  orch.to_dict(),
        "llm_extraction": pipeline.extractor.client is not None,
        "totals": {
            "pubmed":         len(pubmed),
            "europepmc":      len(europepmc),
            "clinicaltrials": len(clinicaltrials),
            "reddit":         len(reddit_raw),
        },
        "pubmed": pubmed, "europepmc": europepmc,
        "clinicaltrials": clinicaltrials, "reddit": reddit_raw,
    }


# ── Article pipeline ──────────────────────────────────────────────────────────

@app.post("/pipeline/run")
def run_pipeline(q: str = Query(...), db: Session = Depends(get_db)):
    """
    Full article pipeline:
    1. Collect from PubMed, EuropePMC, ClinicalTrials
    2. LLM-extract a ClinicalProfile from each abstract
    3. Save profile + SourceAttribution to DB
    Returns summary and saved profiles.
    """
    from core.pipeline import HLEOPipeline
    from core.article_extractor import ArticleExtractor

    # ── Orchestrate: detect language, translate to scientific English if needed ──
    orch = _orchestrator.process(q)

    pipeline  = HLEOPipeline()
    extractor = ArticleExtractor()

    if extractor.client is None:
        return {"error": "OPENAI_API_KEY not set — cannot run LLM extraction."}

    raw = pipeline.collect(orch.search_query)

    articles = []
    for item in raw["pubmed"]:
        articles.append({
            "source":     "pubmed",
            "episode_id": f"pubmed-{item.pmid}",
            "title":      item.title,
            "abstract":   item.abstract or "",
            "url":        f"https://pubmed.ncbi.nlm.nih.gov/{item.pmid}/",
            "external_id": item.pmid,
            "journal":    (item.metadata or {}).get("journal", ""),
            "pub_year":   str((item.metadata or {}).get("pubdate", ""))[:4],
            "meta":       item.metadata or {},
        })
    for item in raw["europepmc"]:
        ep_id = (item.metadata or {}).get("id") or (item.doi or "").replace("/", "-")
        articles.append({
            "source":     "europepmc",
            "episode_id": f"europepmc-{ep_id}",
            "title":      item.title,
            "abstract":   item.abstract or "",
            "url":        f"https://doi.org/{item.doi}" if item.doi else "",
            "external_id": item.doi or ep_id,
            "journal":    (item.metadata or {}).get("journal", ""),
            "pub_year":   str(item.year or ""),
            "meta":       item.metadata or {},
        })
    for item in raw["clinicaltrials"]:
        nct = (item.metadata or {}).get("nct_id", "unknown")
        articles.append({
            "source":     "clinicaltrials",
            "episode_id": f"clinicaltrial-{nct}",
            "title":      item.title,
            "abstract":   item.abstract or "",
            "url":        f"https://clinicaltrials.gov/study/{nct}" if nct != "unknown" else "",
            "external_id": nct,
            "journal":    "",
            "pub_year":   str(item.year or ""),
            "meta":       item.metadata or {},
        })

    # ── Phase 1: Pre-checks (sequential, cheap DB lookups) ───────────────────
    # Resolve already-existing records and no-abstract skips before any LLM work.
    # The DB session stays on the main thread for all three phases.
    pre_results: dict[int, dict] = {}        # index → resolved entry (skip cases)
    needs_llm:   list[tuple[int, dict]] = [] # (index, art) requiring extraction

    for i, art in enumerate(articles):
        episode_id = art["episode_id"]

        existing = db.execute(
            select(ClinicalProfile).where(ClinicalProfile.episode_id == episode_id)
        ).scalar_one_or_none()

        if existing:
            pre_results[i] = {
                "episode_id":  episode_id,
                "status":      "already_exists",
                "db_id":       existing.id,
                "source":      art["source"],
                "title":       art["title"],
                "profile":     existing.extracted_payload,
                "attribution": _get_attribution(db, episode_id),
            }
            continue

        if not art["abstract"]:
            pre_results[i] = {
                "_is_error":  True,
                "episode_id": episode_id,
                "error":      "No abstract available — skipped.",
            }
            continue

        needs_llm.append((i, art))

    # ── Phase 2: Parallel LLM extraction ─────────────────────────────────────
    # Only extractor.extract() calls run concurrently.
    # max_workers=8 caps simultaneous gpt-4o calls — safe for OpenAI Tier-1 limits.
    # Results are keyed by the original position index so order is preserved.
    # One failure never propagates to the others.
    _MAX_WORKERS = 8
    llm_results: dict[int, tuple] = {}
    # value: ("ok", payload, None) | ("error", None, err_str)

    if needs_llm:
        def _extract_one(idx_art: tuple) -> tuple:
            idx, art = idx_art
            try:
                payload = extractor.extract(
                    title=art["title"],
                    abstract=art["abstract"],
                    source=art["source"],
                )
                return idx, "ok", payload, None
            except Exception as exc:
                return idx, "error", None, str(exc)

        logger.info(
            "Extraction | %d articles → parallel (max_workers=%d)",
            len(needs_llm), _MAX_WORKERS,
        )
        t_llm_start = datetime.now(timezone.utc)
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            futures = {pool.submit(_extract_one, item): item[0] for item in needs_llm}
            for fut in as_completed(futures):
                idx, status, payload, err = fut.result()
                llm_results[idx] = (status, payload, err)
                if status == "error":
                    logger.error(
                        "Extraction failed [%s]: %s",
                        articles[idx]["episode_id"], err,
                    )
                else:
                    logger.info("Extracted %s", articles[idx]["episode_id"])
        t_llm_elapsed = (datetime.now(timezone.utc) - t_llm_start).total_seconds()
        logger.info(
            "Extraction | done in %.1fs (%d ok, %d failed)",
            t_llm_elapsed,
            sum(1 for v in llm_results.values() if v[0] == "ok"),
            sum(1 for v in llm_results.values() if v[0] == "error"),
        )

    # ── Phase 3: DB writes (sequential, main thread, original order) ──────────
    # Iterate by original index so saved[] and errors[] are in input order.
    saved = []
    errors = []

    for i, art in enumerate(articles):
        episode_id = art["episode_id"]

        # Pre-resolved in Phase 1 (DB hit or no-abstract)
        if i in pre_results:
            entry = pre_results[i]
            if entry.get("_is_error"):
                errors.append({"episode_id": entry["episode_id"], "error": entry["error"]})
            else:
                saved.append(entry)
            continue

        # Should always be present after Phase 2 ran
        if i not in llm_results:
            errors.append({"episode_id": episode_id, "error": "Extraction result missing."})
            continue

        status, payload, err = llm_results[i]
        if status == "error":
            errors.append({"episode_id": episode_id, "error": err})
            continue

        try:
            row = ClinicalProfile(
                episode_id=episode_id,
                user_id=art["source"],
                final_category="N/A",
                confidence_score=0.0,
                adjudication_required=False,
                extracted_payload=payload,
                validation_payload={
                    "source":         art["source"],
                    "title":          art["title"],
                    "url":            art["url"],
                    "abstract_chars": len(art["abstract"]),
                    "journal":        art["journal"],
                    "pub_year":       art["pub_year"],
                    "meta":           art["meta"],
                },
            )
            db.add(row)
            db.flush()

            attr = SourceAttribution(
                profile_episode_id=episode_id,
                source_type=art["source"],
                source_title=art["title"],
                source_url=art["url"],
                external_id=art["external_id"],
                journal=art["journal"],
                pub_year=art["pub_year"],
                abstract_excerpt=art["abstract"][:500],
            )
            db.add(attr)
            db.commit()
            db.refresh(row)

            saved.append({
                "episode_id": episode_id,
                "status":     "saved",
                "db_id":      row.id,
                "source":     art["source"],
                "title":      art["title"],
                "profile":    payload,
                "attribution": _get_attribution(db, episode_id),
            })
            logger.info("Saved profile %s", episode_id)

        except Exception as exc:
            db.rollback()
            logger.exception("Failed to write %s: %s", episode_id, exc)
            errors.append({"episode_id": episode_id, "error": str(exc)})

    # Flat list of ALL episode_ids processed (new + pre-existing) — used by frontend
    # to filter the profiles view to only this search's results.
    all_episode_ids = [s["episode_id"] for s in saved]

    return {
        "query":           q,
        "orchestration":   orch.to_dict(),
        "processed":       len(articles),
        "saved":           len([s for s in saved if s["status"] == "saved"]),
        "already_existed": len([s for s in saved if s["status"] == "already_exists"]),
        "errors":          len(errors),
        "episode_ids":     all_episode_ids,   # ← Feature: search-scoped profile filtering
        "results":         saved,
        "error_details":   errors,
    }


def _get_attribution(db: Session, episode_id: str) -> Optional[dict]:
    attr = db.execute(
        select(SourceAttribution).where(SourceAttribution.profile_episode_id == episode_id)
    ).scalar_one_or_none()
    if not attr:
        return None
    return {
        "source_type":  attr.source_type,
        "source_title": attr.source_title,
        "source_url":   attr.source_url,
        "external_id":  attr.external_id,
        "journal":      attr.journal,
        "pub_year":     attr.pub_year,
        "abstract_excerpt": attr.abstract_excerpt,
    }


# ── Clinical profiles ─────────────────────────────────────────────────────────

@app.get("/profiles")
def list_profiles(
    limit:       int           = Query(20, ge=1, le=100),
    episode_ids: Optional[str] = Query(None),   # comma-separated; filters to current search
    db:          Session       = Depends(get_db),
):
    """Return saved clinical profiles with source attribution.

    When `episode_ids` is supplied (comma-separated), only those profiles are
    returned — this powers the per-search isolation in the Profiles view.
    """
    id_filter = [eid.strip() for eid in episode_ids.split(",") if eid.strip()] \
                if episode_ids else None

    q = select(ClinicalProfile).order_by(desc(ClinicalProfile.processed_at))
    if id_filter is not None:
        q = q.where(ClinicalProfile.episode_id.in_(id_filter))
    rows = db.execute(q.limit(limit)).scalars().all()

    result = []
    for r in rows:
        vp = r.validation_payload or {}
        attr = _get_attribution(db, r.episode_id)
        result.append({
            "id":                   r.id,
            "episode_id":           r.episode_id,
            "user_id":              r.user_id,
            "final_category":       r.final_category,
            "confidence_score":     r.confidence_score,
            "adjudication_required": r.adjudication_required,
            "processed_at":         r.processed_at.isoformat() if r.processed_at else None,
            "title":    vp.get("title", ""),
            "source":   vp.get("source", r.user_id),
            "url":      vp.get("url", ""),
            "journal":  vp.get("journal", ""),
            "pub_year": vp.get("pub_year", ""),
            "profile":  r.extracted_payload,
            "attribution": attr,
        })

    # Total count respects the same filter so the badge is accurate
    total_q = select(func.count()).select_from(ClinicalProfile)
    if id_filter is not None:
        total_q = total_q.where(ClinicalProfile.episode_id.in_(id_filter))

    return {
        "total":    db.execute(total_q).scalar(),
        "profiles": result,
    }


# ── Patient experiences ───────────────────────────────────────────────────────

@app.post("/experiences/ingest")
def ingest_experiences(q: str = Query(...), db: Session = Depends(get_db)):
    """
    Collect Reddit posts via PRAW OAuth, LLM-extract patient experiences, save to DB.

    Always returns a structured response including:
      reddit_status  — ok | no_credentials | auth_error | rate_limited | no_results | network_error
      reddit_reason  — human-readable explanation of the status
    """
    from collectors.reddit import RedditCollector, STATUS_OK, STATUS_NO_CREDENTIALS
    from core.patient_extractor import PatientExperienceExtractor

    extractor = PatientExperienceExtractor()
    if extractor.client is None:
        return {
            "query": q, "collected": 0, "saved": 0, "errors": 0, "results": [],
            "reddit_status": "no_openai_key",
            "reddit_reason": "OPENAI_API_KEY is not set — LLM extraction is disabled.",
        }

    # ── Orchestrate: detect language, translate to scientific English if needed ──
    orch = _orchestrator.process(q)

    # ── Collect from Reddit via PRAW ────────────────────────────────────────
    collector = RedditCollector()
    raw_reddit, reddit_status, reddit_reason = collector.search_with_status(
        orch.search_query, limit=15
    )
    logger.info(f"Reddit [{reddit_status}] for '{orch.search_query}': {reddit_reason}")

    if reddit_status != STATUS_OK:
        return {
            "query":          q,
            "orchestration":  orch.to_dict(),
            "collected":      0,
            "saved":          0,
            "already_existed": 0,
            "errors":         0,
            "results":        [],
            "error_details":  [],
            "reddit_status":  reddit_status,
            "reddit_reason":  reddit_reason,
        }

    # ── Extract and save ────────────────────────────────────────────────────
    saved  = []
    errors = []

    for post in raw_reddit:
        episode_id = f"reddit-exp-{abs(hash(post.url))}"

        existing = db.execute(
            select(PatientExperience).where(PatientExperience.episode_id == episode_id)
        ).scalar_one_or_none()
        if existing:
            saved.append({
                "episode_id": episode_id,
                "status":  "already_exists",
                "title":   post.title,
                "profile": existing.extracted_profile,
            })
            continue

        body = (post.text or "").strip()
        if len(body) < 50:
            errors.append({"episode_id": episode_id, "error": "Post body too short — skipped."})
            continue

        try:
            profile = extractor.extract(
                title=post.title,
                text=body,
                author=post.author or "",
                url=post.url or "",
            )

            row = PatientExperience(
                episode_id=episode_id,
                source_platform="reddit",
                source_url=post.url,
                author=post.author,
                raw_text=body[:4000],
                extracted_profile=profile,
                query_context=q,
            )
            db.add(row)
            db.commit()
            db.refresh(row)

            saved.append({
                "episode_id": episode_id,
                "status": "saved",
                "db_id":  row.id,
                "title":  post.title,
                "url":    post.url,
                "profile": profile,
            })
            logger.info(f"Saved patient experience {episode_id}")

        except Exception as exc:
            db.rollback()
            logger.exception(f"Failed to extract experience {episode_id}: {exc}")
            errors.append({"episode_id": episode_id, "error": str(exc)})

    n_saved = len([s for s in saved if s["status"] == "saved"])
    return {
        "query":           q,
        "orchestration":   orch.to_dict(),
        "collected":       len(raw_reddit),
        "saved":           n_saved,
        "already_existed": len([s for s in saved if s["status"] == "already_exists"]),
        "errors":          len(errors),
        "results":         saved,
        "error_details":   errors,
        "reddit_status":   STATUS_OK,
        "reddit_reason":   f"Retrieved {len(raw_reddit)} post(s); {n_saved} new experience(s) saved.",
    }


@app.get("/experiences")
def list_experiences(
    limit: int = Query(20, ge=1, le=100),
    db:    Session = Depends(get_db),
):
    rows = db.execute(
        select(PatientExperience)
        .order_by(desc(PatientExperience.ingested_at))
        .limit(limit)
    ).scalars().all()

    return {
        "total": db.execute(select(func.count()).select_from(PatientExperience)).scalar(),
        "experiences": [
            {
                "id":           r.id,
                "episode_id":   r.episode_id,
                "source_url":   r.source_url,
                "query_context": r.query_context,
                "ingested_at":  r.ingested_at.isoformat() if r.ingested_at else None,
                "profile":      r.extracted_profile,
            }
            for r in rows
        ],
    }


# ── AI Clinical Assistant ─────────────────────────────────────────────────────

from typing import List

class SearchArticleCtx(BaseModel):
    """One article from the active search, sent by the frontend."""
    source: str                     # "pubmed" | "europepmc" | "clinicaltrials"
    title: str
    abstract: Optional[str] = ""
    url: Optional[str] = ""
    # Bibliographic identifiers — used by the AI for accurate Key Studies citations
    pmid: Optional[str] = ""
    doi: Optional[str] = ""
    nct_id: Optional[str] = ""
    year: Optional[str] = ""
    journal: Optional[str] = ""

class SearchContext(BaseModel):
    """
    Active search context forwarded by the frontend with every chat message.
    Populated by the orchestrator output + raw collector results.
    """
    original_query: str
    search_query: str               # English query actually sent to collectors
    detected_language: str          # ISO-639-1 code from orchestrator
    articles: List[SearchArticleCtx] = []

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    language: Optional[str] = "en"         # ISO 639-1 code, e.g. "en" / "it"
    search_context: Optional[SearchContext] = None   # Feature 002: active search


@app.post("/assistant/chat")
def assistant_chat(
    body: ChatRequest,
    db: Session = Depends(get_db),
):
    """
    AI Clinical Assistant — RAG over stored profiles and patient experiences.
    Creates a session if none provided; returns assistant response + session_id.
    """
    import os, json as _json

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return {"error": "OPENAI_API_KEY not set."}

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    # ── Session management ──────────────────────────────────────────
    import json as _json
    session_id = body.session_id or str(uuid.uuid4())
    session = db.execute(
        select(ChatSession).where(ChatSession.session_id == session_id)
    ).scalar_one_or_none()

    if not session:
        # Auto-create only when no session_id was supplied (new chat from welcome screen)
        title = body.message[:80]
        session = ChatSession(session_id=session_id, title=title, status="active")
        db.add(session)
        db.commit()
    elif session.status == "closed":
        raise HTTPException(
            status_code=409,
            detail="This session is closed. Reopen it before sending messages.",
        )

    # ── Persist search context on session (isolation guarantee) ────
    # Always overwrite so the session always reflects the latest search.
    now_utc = datetime.now(timezone.utc)
    if body.search_context is not None:
        session.search_query   = body.search_context.original_query
        session.search_context = body.search_context.model_dump()
        session.updated_at     = now_utc
        db.commit()
    else:
        session.updated_at = now_utc
        db.commit()

    # ── Resolve effective search context (backend-enforced isolation) ──
    # If the frontend didn't send a search_context, fall back to the one
    # stored on THIS session — never from any other session.
    effective_search_ctx = body.search_context
    if effective_search_ctx is None and session.search_context:
        try:
            sc_data = session.search_context
            effective_search_ctx = SearchContext(
                original_query=sc_data.get("original_query", ""),
                search_query=sc_data.get("search_query", ""),
                detected_language=sc_data.get("detected_language", "en"),
                articles=[
                    SearchArticleCtx(**a) for a in sc_data.get("articles", [])
                ],
            )
        except Exception:
            effective_search_ctx = None

    # ── Load conversation history ───────────────────────────────────
    history_rows = db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
        .limit(20)
    ).scalars().all()

    messages_history = [
        {"role": r.role, "content": r.content} for r in history_rows
    ]

    # ── RAG: retrieve relevant context from DB ──────────────────────
    # Skip entirely when current search articles are available — the retrieved
    # scientific literature is the sole primary source in that case.
    # DB records are only fetched (and injected) when there are no search articles.
    user_msg_lower = body.message.lower()
    context_snippets: list = []
    context_episode_ids: list = []

    _has_search_articles_early = bool(
        effective_search_ctx and effective_search_ctx.articles
    )

    if not _has_search_articles_early:
        # Search clinical profiles
        cp_rows = db.execute(
            select(ClinicalProfile)
            .order_by(desc(ClinicalProfile.processed_at))
            .limit(30)
        ).scalars().all()

        for cp in cp_rows:
            vp = cp.validation_payload or {}
            title = vp.get("title", "").lower()
            payload = cp.extracted_payload or {}
            payload_text = _json.dumps(payload).lower()
            if any(word in title or word in payload_text
                   for word in user_msg_lower.split() if len(word) > 3):
                diag       = ", ".join(payload.get("diagnosis", [])[:3])
                treats     = ", ".join(payload.get("treatments", [])[:4])
                dosages    = ", ".join(payload.get("dosages", [])[:4])
                outcomes   = ", ".join(payload.get("outcomes", [])[:3])
                ae         = ", ".join(payload.get("adverse_effects", [])[:3])
                ev_level   = payload.get("evidence_level", "")
                study_pop  = payload.get("study_population", "")
                snippet  = (
                    f"[Clinical Profile — {cp.user_id.upper()}, {vp.get('pub_year','')}] "
                    f"'{vp.get('title','')[:100]}'"
                    f"{f' ({ev_level})' if ev_level else ''}"
                    f"{f' [{study_pop}]' if study_pop else ''}\n"
                    f"  Diagnosis: {diag or 'N/A'}\n"
                    f"  Treatments: {treats or 'N/A'}"
                    f"{f' | Dosages: {dosages}' if dosages else ''}\n"
                    f"  Outcomes: {outcomes or 'N/A'}\n"
                    f"  Adverse effects: {ae or 'N/A'}"
                )
                context_snippets.append(snippet)
                context_episode_ids.append(cp.episode_id)
                if len(context_snippets) >= 5:
                    break

        # Search patient experiences
        pe_rows = db.execute(
            select(PatientExperience)
            .order_by(desc(PatientExperience.ingested_at))
            .limit(30)
        ).scalars().all()

        for pe in pe_rows:
            p = pe.extracted_profile or {}
            payload_text = _json.dumps(p).lower()
            if any(word in payload_text for word in user_msg_lower.split() if len(word) > 3):
                condition = p.get("condition", "unknown condition")
                summary   = p.get("experience_summary", "")
                treats    = ", ".join(p.get("treatments_tried", [])[:3])
                outcomes  = ", ".join(p.get("reported_outcomes", [])[:3])
                snippet   = (
                    f"[Patient Experience — Reddit]\n"
                    f"  Condition: {condition}\n"
                    f"  Summary: {summary[:200] if summary else 'N/A'}\n"
                    f"  Treatments tried: {treats or 'N/A'}\n"
                    f"  Reported outcomes: {outcomes or 'N/A'}"
                )
                context_snippets.append(snippet)
                context_episode_ids.append(pe.episode_id)
                if len(context_snippets) >= 8:
                    break

    # ── Build system prompt ─────────────────────────────────────────

    _LANG_MAP = {
        "it": "Italian", "en": "English", "fr": "French",
        "de": "German",  "es": "Spanish", "pt": "Portuguese",
    }
    _resp_lang = _LANG_MAP.get(body.language or "en", "English")
    _lang_note = (
        f"\n\nIMPORTANT: You must respond entirely in {_resp_lang}. "
        f"All your answers, labels, and explanations must be written in {_resp_lang}."
        if _resp_lang != "English" else ""
    )

    # ── Priority 1: current search context (Feature 002 + 003 isolation) ──
    search_block = ""
    has_search_articles = False
    sc = effective_search_ctx   # backend-resolved; never leaks from other sessions

    if sc and sc.articles:
        has_search_articles = True
        lang_note_query = (
            f" (original query in {sc.detected_language.upper()}: '{sc.original_query}',"
            f" translated to: '{sc.search_query}')"
            if sc.original_query.lower() != sc.search_query.lower()
            else f" (query: '{sc.original_query}')"
        )
        art_lines = []
        for i, art in enumerate(sc.articles, 1):
            src_label = {"pubmed": "PubMed", "europepmc": "Europe PMC",
                         "clinicaltrials": "ClinicalTrials.gov"}.get(art.source, art.source)
            abstract_excerpt = (art.abstract or "").strip()
            url_part = f"\n     URL: {art.url}" if art.url else ""
            # Build identifier string so the AI can cite accurately without hallucinating IDs
            id_parts = []
            if art.pmid:   id_parts.append(f"PMID {art.pmid}")
            if art.doi:    id_parts.append(f"DOI {art.doi}")
            if art.nct_id: id_parts.append(f"NCT {art.nct_id}")
            id_str = f" | {' | '.join(id_parts)}" if id_parts else ""
            meta_parts = []
            if art.journal: meta_parts.append(art.journal)
            if art.year:    meta_parts.append(art.year)
            meta_str = f" ({', '.join(meta_parts)})" if meta_parts else ""
            art_lines.append(
                f"  [{i}] [{src_label}]{id_str}{meta_str} {art.title}\n"
                f"      Abstract: {abstract_excerpt or 'N/A'}"
                f"{url_part}"
            )
        source_labels = sorted(set(
            {"pubmed": "PubMed", "europepmc": "Europe PMC",
             "clinicaltrials": "ClinicalTrials.gov"}.get(a.source, a.source)
            for a in sc.articles
        ))
        search_block = (
            f"══ PRIORITY 1 — CURRENT SEARCH RESULTS{lang_note_query} ══\n"
            f"{len(sc.articles)} article(s) retrieved from: {', '.join(source_labels)}\n\n"
            + "\n\n".join(art_lines)
            + "\n\n"
            "══ CLINICAL REASONING ENGINE ══\n"
            "You are a senior clinical expert writing an evidence-based opinion. "
            "You do NOT list database results. You synthesize evidence and communicate conclusions.\n\n"
            "REQUIRED RESPONSE STRUCTURE (in this exact order):\n\n"
            "## 🩺 Clinical Conclusion\n"
            "3–5 sentences that directly state what the evidence shows about the user's question. "
            "This is your overall clinical judgment based on the retrieved articles. "
            "Write it as a clinician would: assertive, grounded, acknowledging uncertainty where it exists. "
            "NEVER open with search metadata ('X articles retrieved', 'A search was performed', "
            "'Based on the retrieved articles', 'The following studies'). "
            "The section must begin with a clinical statement about the topic itself.\n\n"
            "## Confidence Level\n"
            "State exactly ONE of the following on its own line, then a single sentence explaining why:\n"
            "  🟢 High confidence — multiple consistent RCTs or systematic reviews/meta-analyses\n"
            "  🟡 Moderate confidence — some consistent evidence but limited by study quality, size, or number\n"
            "  🔴 Low confidence — few studies, conflicting results, or only observational/case data\n"
            "Base confidence solely on the retrieved articles above. Never inflate it.\n\n"
            "## 🔬 Evidence Summary\n"
            "Synthesize — do NOT enumerate every article. Write 3–5 sentences that explain:\n"
            "  • Where studies agree\n"
            "  • Where studies disagree or conflict\n"
            "  • Important limitations (sample size, follow-up, study design)\n"
            "Every claim must be traceable to the listed articles. "
            "Do not introduce facts not supported by the abstracts above.\n\n"
            "## 📚 Key Studies\n"
            "Cite 3–5 of the most relevant studies. For each, provide on a single line:\n"
            "  [Source] Full study title — PMID XXXXXXXX | DOI xxx | NCT XXXXXXX (use whichever identifier is available from the article data above)\n"
            "Do NOT reference articles as 'Article 1', 'Article 2', etc. Use their actual titles. "
            "Omit studies that are not relevant to the specific question.\n\n"
            "## ℹ️ General Medical Knowledge (ONLY if necessary)\n"
            "Include this section ONLY when the retrieved articles leave a clinically important gap. "
            "If included, open with exactly this sentence: "
            "'The following information is not derived from the retrieved evidence but from general medical knowledge.' "
            "If the retrieved articles fully answer the question, OMIT this section entirely — do not include a placeholder.\n\n"
            "ABSOLUTE RULES:\n"
            "- Synthesis before articles. Conclusion before evidence. Never the reverse.\n"
            "- Never enumerate all retrieved articles; mention only the most relevant ones in Key Studies.\n"
            "- Never repeat the same information across sections.\n"
            "- Every claim in Evidence Summary and Key Studies must trace back to a listed article.\n"
            "- The response must read like a clinician's written opinion, not a database printout.\n"
            "══════════════════════════════════════════\n"
        )
    elif sc:
        # Search was performed but returned no articles
        search_block = (
            f"══ PRIORITY 1 — CURRENT SEARCH: no articles retrieved "
            f"(query: '{sc.original_query}') ══\n"
            "Answer the user's question as directly as possible using general medical knowledge. "
            "If you must rely on general knowledge, state explicitly at the END of your response:\n"
            "  'Note: the current search returned no results. "
            "This answer is based on general medical knowledge.'\n"
            "══════════════════════════════════════════\n"
        )

    # ── Priority 2: HLEO database (RAG) ────────────────────────────────
    # context_snippets is empty when search articles are present (RAG was skipped above).
    if has_search_articles:
        # RAG was not run — no DB content to show.
        db_header = (
            "══ PRIORITY 2 — HLEO DATABASE (not used — search articles present) ══\n"
            "══════════════════════════════════════════"
        )
    else:
        db_block = (
            "\n\n".join(context_snippets)
            if context_snippets
            else "No matching records in the HLEO database for this query."
        )
        db_header = (
            f"══ PRIORITY 2 — HLEO DATABASE ({len(context_snippets)} records) ══\n"
            f"{db_block}\n"
            "══════════════════════════════════════════"
        )

    # ── Priority 3: general knowledge note ─────────────────────────────
    p3_note = (
        "══ PRIORITY 3 — GENERAL KNOWLEDGE ══\n"
        "Use general medical knowledge ONLY if neither the current search results "
        "nor the HLEO database contain sufficient information to answer the question. "
        "When doing so, explicitly state that the answer is based on general knowledge.\n"
        "══════════════════════════════════════════"
    )

    system_prompt = (
        "You are HLEO Clinical Assistant, an AI research assistant for clinicians and "
        "researchers. You synthesize scientific evidence and communicate clinical conclusions.\n\n"
        "SOURCE PRIORITY — strictly enforced:\n"
        "  1. CURRENT SEARCH RESULTS — the retrieved scientific articles are the ONLY valid "
        "primary source when present. Build every primary section exclusively from these.\n"
        "  2. HLEO DATABASE — supplementary only; never used as primary evidence when search "
        "articles are available.\n"
        "  3. GENERAL KNOWLEDGE — only when (1) and (2) are both insufficient.\n\n"
        "You always:\n"
        "- Cite sources inline (e.g. '[PubMed]', '[Europe PMC]', '[HLEO DB]').\n"
        "- Acknowledge uncertainty; never overstate evidence strength.\n"
        "- Recommend consulting a qualified clinician for personal medical decisions.\n"
        "- When search results are present, follow the CLINICAL REASONING ENGINE in Priority 1 "
        "exactly — synthesis and clinical judgment first, individual studies last.\n"
        "- When no search results are present, answer concisely (3–6 sentences) using "
        "HLEO DB then general knowledge.\n\n"
        f"{search_block}"
        f"{db_header}\n\n"
        f"{p3_note}"
        f"{_lang_note}"
    )

    # ── Call LLM ───────────────────────────────────────────────────
    llm_messages = [{"role": "system", "content": system_prompt}]
    llm_messages.extend(messages_history)
    llm_messages.append({"role": "user", "content": body.message})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=llm_messages,
        temperature=0.2,
        max_tokens=1800 if has_search_articles else 800,
    )
    assistant_text = response.choices[0].message.content

    # ── Persist messages ───────────────────────────────────────────
    db.add(ChatMessage(
        session_id=session_id,
        role="user",
        content=body.message,
        context_used=[],
    ))
    db.add(ChatMessage(
        session_id=session_id,
        role="assistant",
        content=assistant_text,
        context_used=context_episode_ids,
    ))
    db.commit()

    return {
        "session_id":          session_id,
        "response":            assistant_text,
        "context_used_count":  len(context_snippets),
        "context_episode_ids": context_episode_ids,
    }


@app.get("/assistant/sessions/{session_id}")
def get_session(session_id: str, db: Session = Depends(get_db)):
    """Return full message history + stored search context for a single session."""
    session = db.execute(
        select(ChatSession).where(ChatSession.session_id == session_id)
    ).scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    messages = db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
    ).scalars().all()

    return {
        "session_id":     session_id,
        "title":          session.title,
        "status":         session.status,
        "search_query":   session.search_query,
        "search_context": session.search_context,   # restored on session open
        "messages": [
            {
                "role":         m.role,
                "content":      m.content,
                "context_used": m.context_used,
                "created_at":   m.created_at.isoformat() if m.created_at else None,
            }
            for m in messages
        ],
    }


# ── Translation endpoint ──────────────────────────────────────────────────────

class TranslateRequest(BaseModel):
    text: str
    target_lang: str = "it"
    content_type: str = "clinical_article"   # clinical_article | patient_experience | general


@app.post("/translate")
async def translate_text(body: TranslateRequest):
    """
    AI translation endpoint used by the frontend language switcher.
    Accepts {text, target_lang, content_type}.
    Returns  {translation, summary, target_lang, content_type}.
    Results are cached in _translate_cache (process lifetime) to avoid re-calling the LLM.
    """
    import os, json as _json, hashlib

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured.")

    cache_key = hashlib.md5(
        f"{body.target_lang}:{body.content_type}:{body.text[:500]}".encode()
    ).hexdigest()
    if cache_key in _translate_cache:
        return _translate_cache[cache_key]

    LANG_NAMES = {
        "it": "Italian",    "en": "English",   "fr": "French",
        "de": "German",     "es": "Spanish",   "pt": "Portuguese",
        "zh": "Chinese",    "ja": "Japanese",  "ar": "Arabic",
        "ru": "Russian",    "nl": "Dutch",     "pl": "Polish",
    }
    lang_name     = LANG_NAMES.get(body.target_lang, body.target_lang)
    content_label = body.content_type.replace("_", " ")

    prompt = (
        f"You are a professional medical translator and summariser.\n\n"
        f"Translate the following {content_label} text to {lang_name}, "
        f"preserving all medical terminology accurately. "
        f"Also provide a concise summary in {lang_name} (2-3 sentences max).\n\n"
        f"TEXT:\n{body.text[:3000]}\n\n"
        "Respond with JSON only:\n"
        '{"translation": "...", "summary": "..."}'
    )

    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=1500,
        temperature=0.2,
    )

    result = _json.loads(response.choices[0].message.content)
    out = {
        "translation":  result.get("translation", ""),
        "summary":      result.get("summary", ""),
        "target_lang":  body.target_lang,
        "content_type": body.content_type,
    }
    _translate_cache[cache_key] = out
    logger.info(f"Translated {len(body.text)} chars to {lang_name} ({body.content_type})")
    return out


# ── Session management constants ──────────────────────────────────────────────
_MAX_ACTIVE  = 10
_MAX_CLOSED  = 20

class SessionPatch(BaseModel):
    action: str          # "rename" | "close" | "reopen"
    title: Optional[str] = None


@app.get("/assistant/sessions")
def list_sessions(db: Session = Depends(get_db)):
    """Return active and closed sessions with message counts."""
    all_sessions = db.execute(
        select(ChatSession).order_by(desc(ChatSession.updated_at))
    ).scalars().all()

    # Count messages per session in one query
    msg_counts_raw = db.execute(
        select(ChatMessage.session_id, func.count(ChatMessage.id).label("cnt"))
        .group_by(ChatMessage.session_id)
    ).all()
    msg_counts = {row.session_id: row.cnt for row in msg_counts_raw}

    def _fmt(s: ChatSession) -> dict:
        return {
            "session_id":   s.session_id,
            "title":        s.title or "Untitled",
            "search_query": s.search_query,
            "message_count": msg_counts.get(s.session_id, 0),
            "created_at":   s.created_at.isoformat() if s.created_at else None,
            "updated_at":   s.updated_at.isoformat() if s.updated_at else None,
            "closed_at":    s.closed_at.isoformat() if s.closed_at else None,
        }

    active = [_fmt(s) for s in all_sessions if s.status == "active"]
    closed = [_fmt(s) for s in all_sessions if s.status == "closed"]
    return {"active": active, "closed": closed}


@app.post("/assistant/sessions")
def create_session(db: Session = Depends(get_db)):
    """Create a new active session, enforcing the 10-active limit."""
    active_count = db.execute(
        select(func.count(ChatSession.id)).where(ChatSession.status == "active")
    ).scalar_one()
    if active_count >= _MAX_ACTIVE:
        raise HTTPException(
            status_code=409,
            detail=(
                f"You have reached the maximum number of active sessions ({_MAX_ACTIVE}). "
                "Close or delete an existing session before creating a new one."
            ),
        )
    now = datetime.now(timezone.utc)
    session = ChatSession(
        session_id=str(uuid.uuid4()),
        title="New session",
        status="active",
        created_at=now,
        updated_at=now,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return {
        "session_id": session.session_id,
        "title":      session.title,
        "status":     session.status,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
    }


@app.patch("/assistant/sessions/{session_id}")
def patch_session(session_id: str, body: SessionPatch, db: Session = Depends(get_db)):
    """Rename, close, or reopen a session with limit enforcement."""
    session = db.execute(
        select(ChatSession).where(ChatSession.session_id == session_id)
    ).scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    now = datetime.now(timezone.utc)

    if body.action == "rename":
        if not body.title or not body.title.strip():
            raise HTTPException(status_code=422, detail="Title cannot be empty.")
        session.title      = body.title.strip()[:120]
        session.updated_at = now

    elif body.action == "close":
        if session.status == "closed":
            raise HTTPException(status_code=409, detail="Session is already closed.")
        closed_count = db.execute(
            select(func.count(ChatSession.id)).where(ChatSession.status == "closed")
        ).scalar_one()
        if closed_count >= _MAX_CLOSED:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"You have reached the maximum number of archived sessions ({_MAX_CLOSED}). "
                    "Delete one or more archived sessions before closing another session."
                ),
            )
        session.status     = "closed"
        session.closed_at  = now
        session.updated_at = now

    elif body.action == "reopen":
        if session.status == "active":
            raise HTTPException(status_code=409, detail="Session is already active.")
        active_count = db.execute(
            select(func.count(ChatSession.id)).where(ChatSession.status == "active")
        ).scalar_one()
        if active_count >= _MAX_ACTIVE:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Maximum active sessions reached. "
                    "Close or delete an active session before reopening this one."
                ),
            )
        session.status     = "active"
        session.closed_at  = None
        session.updated_at = now

    else:
        raise HTTPException(status_code=422, detail=f"Unknown action: {body.action!r}")

    db.commit()
    return {"ok": True, "session_id": session_id, "action": body.action}


@app.delete("/assistant/sessions/{session_id}")
def delete_session(session_id: str, db: Session = Depends(get_db)):
    """Permanently delete a session and all its messages."""
    session = db.execute(
        select(ChatSession).where(ChatSession.session_id == session_id)
    ).scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    db.execute(
        ChatMessage.__table__.delete().where(ChatMessage.session_id == session_id)
    )
    db.delete(session)
    db.commit()
    return {"ok": True, "deleted": session_id}
