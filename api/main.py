"""
HLEO v1.0 — FastAPI application
Endpoints:
  GET  /                     → dashboard UI
  GET  /health               → key/version status
  GET  /stats                → DB row counts
  GET  /search?q=&mode=      → scientific (relational, L2) or global (keyword) collect
  POST /pipeline/run?q=&mode=→ collect + LLM-extract articles → DB (scientific: relevant-only)
  POST /synthesis            → Level 3 scientific synthesis (reuses L2 relevant articles, no re-search)
  GET  /rwe/search?q=        → RWE search (Reddit + openFDA FAERS + Calvizie.net) — patient experiences & community
  GET  /profiles?limit=       → saved clinical profiles
  POST /experiences/ingest?q= → collect Reddit + LLM-extract patient experiences → DB
  GET  /experiences?limit=   → saved patient experiences
  POST /assistant/chat       → AI Clinical Assistant (RAG over DB; accepts scientific + RWE context)
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
    RWEProfile,
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


def _article_from_pubmed(item: Any) -> dict:
    return {
        "source":      "pubmed",
        "episode_id":  f"pubmed-{item.pmid}",
        "title":        item.title,
        "abstract":     item.abstract or "",
        "url":          f"https://pubmed.ncbi.nlm.nih.gov/{item.pmid}/",
        "external_id":  item.pmid,
        "journal":      (item.metadata or {}).get("journal", ""),
        "pub_year":     str((item.metadata or {}).get("pubdate", ""))[:4],
        "meta":         item.metadata or {},
    }


def _article_from_europepmc(item: Any) -> dict:
    ep_id = (item.metadata or {}).get("id") or (item.doi or "").replace("/", "-")
    return {
        "source":      "europepmc",
        "episode_id":  f"europepmc-{ep_id}",
        "title":        item.title,
        "abstract":     item.abstract or "",
        "url":          f"https://doi.org/{item.doi}" if item.doi else "",
        "external_id":  item.doi or ep_id,
        "journal":      (item.metadata or {}).get("journal", ""),
        "pub_year":     str(item.year or ""),
        "meta":         item.metadata or {},
    }


def _article_from_clinicaltrials(item: Any) -> dict:
    nct = (item.metadata or {}).get("nct_id", "unknown")
    return {
        "source":      "clinicaltrials",
        "episode_id":  f"clinicaltrial-{nct}",
        "title":        item.title,
        "abstract":     item.abstract or "",
        "url":          f"https://clinicaltrials.gov/study/{nct}" if nct != "unknown" else "",
        "external_id":  nct,
        "journal":      "",
        "pub_year":     str(item.year or ""),
        "meta":         item.metadata or {},
    }


def _articles_from_raw(raw: dict) -> list[dict]:
    """Build the article list from a HLEOPipeline.collect() result (global mode)."""
    articles: list[dict] = []
    for item in raw["pubmed"]:
        articles.append(_article_from_pubmed(item))
    for item in raw["europepmc"]:
        articles.append(_article_from_europepmc(item))
    for item in raw["clinicaltrials"]:
        articles.append(_article_from_clinicaltrials(item))
    return articles


def _articles_from_relational(rel_out: dict, only_relevant: bool = True) -> list[dict]:
    """Build the article list from a RelationalSearch result.

    In scientific mode, profiles are extracted ONLY from articles the relational
    judge labeled `relevant` (the core rule). relevance_label/score/reason are
    carried into validation_payload so each profile stays anchored to the relation.
    """
    articles: list[dict] = []
    for source_key, builder in (
        ("pubmed", _article_from_pubmed),
        ("europepmc", _article_from_europepmc),
        ("clinicaltrials", _article_from_clinicaltrials),
    ):
        for item in rel_out.get(source_key, []):
            label = (getattr(item, "metadata", None) or {}).get("relevance_label")
            if only_relevant and label != "relevant":
                continue
            art = builder(item)
            md = getattr(item, "metadata", None) or {}
            art["relevance_label"]  = md.get("relevance_label")
            art["relevance_score"]  = md.get("relevance_score")
            art["relevance_reason"] = md.get("relevance_reason")
            articles.append(art)
    return articles


# ── Core routes ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(request, "index.html")


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
def search(q: str = Query(..., description="Search query"),
           mode: str = Query("scientific", description="Search mode: 'scientific' (relational, Level 2) | 'global' (broad keyword).")):
    """Collect results from all sources.

    Two modes:
      - scientific (default): relational pipeline (Level 2) — AI-extracted
        clinical relation → per-source structured queries → hard filter →
        LLM relational judge re-ranking. Requires OPENAI_API_KEY (503 if absent,
        no silent keyword fallback, so scientific profiles stay relation-pure).
      - global: broad exploratory keyword pipeline (orchestrator + collect),
        no relational filter, all sources including Reddit.

    When called without `mode`, defaults to scientific (preserves prior behavior
    when the LLM key is available).
    """
    from core.pipeline import HLEOPipeline

    if mode == "scientific":
        from core.relational_search import RelationalSearch
        rs = RelationalSearch()
        if rs._client is None:
            raise HTTPException(
                status_code=503,
                detail="Scientific mode requires OPENAI_API_KEY (relational search unavailable). Use Global mode.",
            )
        try:
            rel_out = rs.search(q)
        except Exception as exc:
            logger.warning(f"/search scientific: relational search failed ({exc}).")
            raise HTTPException(
                status_code=503,
                detail=f"Scientific search failed: {exc}. Try Global mode.",
            )
        if rel_out is None:
            raise HTTPException(
                status_code=503,
                detail="Scientific mode could not extract a clinical relation. Try rephrasing or use Global mode.",
            )
        pubmed         = [_to_dict(a) for a in rel_out["pubmed"]]
        europepmc      = [_to_dict(a) for a in rel_out["europepmc"]]
        clinicaltrials = [_to_dict(a) for a in rel_out["clinicaltrials"]]
        reddit_raw     = [_to_dict(p) for p in rel_out["reddit"]]
        rel = rel_out["relation"]
        orch_dict = {
            "original_query":      q,
            "search_query":        rel.scientific_query or q,
            "detected_language":   "",
            "translation_applied": True,
            "confidence":          1.0,
            "relation":            rel.to_dict(),
            "retrieval_stats":     rel_out["stats"],
        }
        return {
            "query":          q,
            "orchestration":  orch_dict,
            "llm_extraction": True,
            "totals": {
                "pubmed":         len(pubmed),
                "europepmc":      len(europepmc),
                "clinicaltrials": len(clinicaltrials),
                "reddit":         len(reddit_raw),
            },
            "pubmed": pubmed, "europepmc": europepmc,
            "clinicaltrials": clinicaltrials, "reddit": reddit_raw,
        }

    # ── Global mode: plain keyword pipeline (orchestrator + collect) ─────────
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
def run_pipeline(q: str = Query(...), db: Session = Depends(get_db),
                 mode: str = Query("scientific", description="Extraction mode: 'scientific' (relational, relevant-only) | 'global' (broad, all articles).")):
    """
    Full article pipeline:
    1. Collect from PubMed, EuropePMC, ClinicalTrials (Reddit only in global)
    2. LLM-extract a ClinicalProfile from each abstract
    3. Save profile + SourceAttribution to DB
    Returns summary and saved profiles.

    Modes:
      - scientific (default): uses RelationalSearch (Level 2) and extracts profiles
        ONLY from articles judged `relevant` by the relational judge. Requires
        OPENAI_API_KEY. The relation is preserved and relevance_label/score/reason
        are stored in validation_payload so profiles stay relation-pure.
      - global: broad keyword pipeline (orchestrator + collect), extracts from all
        retrieved articles, no relevance filter (prior behavior).
    """
    from core.article_extractor import ArticleExtractor

    extractor = ArticleExtractor()
    if extractor.client is None:
        return {"error": "OPENAI_API_KEY not set — cannot run LLM extraction."}

    # ── Retrieval dispatch (mode-aware) ──────────────────────────────────────
    rel_dict: Optional[dict] = None   # relation preserved from scientific search
    if mode == "scientific":
        from core.relational_search import RelationalSearch
        rs = RelationalSearch()
        if rs._client is None:
            return {"error": "Scientific mode requires OPENAI_API_KEY."}
        try:
            rel_out = rs.search(q)
        except Exception as exc:
            logger.warning(f"/pipeline/run scientific: relational search failed ({exc}).")
            return {"error": f"Scientific search failed: {exc}. Try Global mode."}
        if rel_out is None:
            return {"error": "Scientific mode could not extract a clinical relation. Try rephrasing or use Global mode."}
        rel = rel_out["relation"]
        rel_dict = rel.to_dict()
        orch_search_query = rel.scientific_query or q
        # Scientific rule: profiles only from articles judged 'relevant'.
        articles = _articles_from_relational(rel_out, only_relevant=True)
        if not articles:
            return {
                "query":           q,
                "orchestration":   {"original_query": q, "search_query": orch_search_query,
                                    "relation": rel_dict, "translation_applied": True},
                "processed":       0, "saved": 0, "already_existed": 0, "errors": 0,
                "episode_ids":     [], "results": [], "error_details": [],
                "warning":         "No articles judged relevant; no profiles extracted.",
            }
    else:
        from core.pipeline import HLEOPipeline
        orch = _orchestrator.process(q)
        orch_search_query = orch.search_query
        pipeline = HLEOPipeline()
        raw = pipeline.collect(orch_search_query)
        articles = _articles_from_raw(raw)

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
                    "relevance_label":  art.get("relevance_label"),
                    "relevance_score":  art.get("relevance_score"),
                    "relevance_reason": art.get("relevance_reason"),
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

    # Build orchestration dict for the response (mode-aware).
    if mode == "scientific":
        orchestration_out = {
            "original_query":      q,
            "search_query":        orch_search_query,
            "detected_language":   "",
            "translation_applied": True,
            "relation":            rel_dict,
        }
    else:
        orchestration_out = orch.to_dict()

    return {
        "query":           q,
        "orchestration":   orchestration_out,
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


# ── RWE search (Real World Evidence — patient experiences & community) ────────
#
# Independent from the scientific /search pipeline. Collects from Reddit
# (PRAW OAuth2) and openFDA FAERS (official API, no key required) only —
# sources that require authorization are registered in /rwe/sources but
# never scraped. RWE items are stamped evidence_tier=anecdotal /
# spontaneous_report and never presented as clinical evidence.

@app.get("/rwe/search")
def rwe_search(
    q: str = Query(..., description="RWE search query"),
    limit: int = Query(15, ge=1, le=50),
    sources: Optional[str] = Query(
        None, description="Comma-separated subset: reddit,openfda_faers,calvizie,hairlosstalk,hairlossexperiences,maladiesrares"
    ),
):
    """Run the RWE pipeline. Returns normalized RWE items with provenance."""
    from core.rwe.pipeline import RWEPipeline

    pipe = RWEPipeline()
    src_list = [s.strip() for s in sources.split(",")] if sources else None
    result = pipe.search(q, limit=limit, sources=src_list)
    return result.model_dump()


# ── FASE 13-14: RWE Profiles + Testimonianze ────────────────────────────────

class RWEExtractRequest(BaseModel):
    """FASE 13: extract a profile from a single RWE item."""
    title: str = ""
    text: str = ""
    source: str = ""
    source_type: str = ""
    evidence_tier: str = "anecdotal"
    source_url: str = ""
    external_id: str = ""
    treatment: str = ""
    condition: str = ""
    experience_type: str = "discussion"
    language: str = "en"
    query_context: str = ""


@app.post("/rwe/extract")
def rwe_extract_profile(body: RWEExtractRequest, db: Session = Depends(get_db)):
    """FASE 13: LLM-extract a structured profile from a single RWE item.

    Persists the result as an RWEProfile row. Returns the extracted profile.
    Skips items already extracted (external_id dedup).
    """
    import os, hashlib

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return {"error": "OPENAI_API_KEY not set."}

    # Dedup by external_id if present
    if body.external_id:
        existing = db.query(RWEProfile).filter(
            RWEProfile.external_id == body.external_id
        ).first()
        if existing:
            return {
                "episode_id": existing.episode_id,
                "extracted_profile": existing.extracted_profile,
                "already_existed": True,
                "source": existing.source,
            }

    from core.rwe.profile_extractor import RWEProfileExtractor
    extractor = RWEProfileExtractor()
    try:
        profile = extractor.extract(
            title=body.title,
            text=body.text,
            source=body.source,
            source_type=body.source_type,
            treatment=body.treatment,
            condition=body.condition,
        )
    except RuntimeError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        logger.exception(f"/rwe/extract failed — {exc}")
        return {"error": f"RWE extraction failed: {exc}"}

    episode_id = "rwe-" + hashlib.md5(
        f"{body.source}:{body.external_id or body.title}:{body.text[:100]}".encode()
    ).hexdigest()[:16]

    row = RWEProfile(
        episode_id=episode_id,
        source=body.source,
        source_type=body.source_type,
        evidence_tier=body.evidence_tier,
        source_url=body.source_url,
        external_id=body.external_id,
        title=body.title,
        raw_text=(body.text or "")[:5000],
        extracted_profile=profile,
        treatment=profile.get("treatment") or body.treatment,
        condition=profile.get("condition") or body.condition,
        experience_type=profile.get("experience_type") or body.experience_type,
        query_context=body.query_context,
        language=body.language,
        is_testimonial=False,
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    return {
        "episode_id": row.episode_id,
        "extracted_profile": profile,
        "already_existed": False,
        "source": row.source,
    }


@app.get("/rwe/profiles")
def list_rwe_profiles(
    db: Session = Depends(get_db),
    source: Optional[str] = None,
    treatment: Optional[str] = None,
    limit: int = 50,
):
    """FASE 13: list stored RWE profiles."""
    q = db.query(RWEProfile)
    if source:
        q = q.filter(RWEProfile.source == source)
    if treatment:
        q = q.filter(RWEProfile.treatment.ilike(f"%{treatment}%"))
    rows = q.order_by(RWEProfile.ingested_at.desc()).limit(limit).all()
    return {
        "count": len(rows),
        "profiles": [{
            "episode_id": r.episode_id,
            "source": r.source,
            "source_type": r.source_type,
            "evidence_tier": r.evidence_tier,
            "title": r.title,
            "treatment": r.treatment,
            "condition": r.condition,
            "experience_type": r.experience_type,
            "is_testimonial": r.is_testimonial,
            "extracted_profile": r.extracted_profile,
            "source_url": r.source_url,
            "external_id": r.external_id,
            "query_context": r.query_context,
            "language": r.language,
            "ingested_at": r.ingested_at.isoformat() if r.ingested_at else None,
        } for r in rows],
    }


@app.get("/rwe/testimonianze")
def list_rwe_testimonianze(
    db: Session = Depends(get_db),
    treatment: Optional[str] = None,
    limit: int = 20,
):
    """FASE 14: curated testimonials (is_testimonial=True) derived from RWE profiles.

    A testimonial is a community_forum RWE profile curated for display.
    FAERS records are excluded — spontaneous reports are NOT testimonials.
    """
    q = db.query(RWEProfile).filter(
        RWEProfile.is_testimonial.is_(True),
        RWEProfile.source_type == "community_forum",
    )
    if treatment:
        q = q.filter(RWEProfile.treatment.ilike(f"%{treatment}%"))
    rows = q.order_by(RWEProfile.ingested_at.desc()).limit(limit).all()
    return {
        "count": len(rows),
        "testimonials": [{
            "episode_id": r.episode_id,
            "source": r.source,
            "title": r.title,
            "treatment": r.treatment,
            "condition": r.condition,
            "extracted_profile": r.extracted_profile,
            "source_url": r.source_url,
            "language": r.language,
        } for r in rows],
    }


@app.post("/rwe/testimonianze/{episode_id}/curate")
def curate_testimonial(episode_id: str, db: Session = Depends(get_db)):
    """FASE 14: mark an RWE profile as a curated testimonial."""
    row = db.query(RWEProfile).filter(RWEProfile.episode_id == episode_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="RWE profile not found.")
    if row.source_type != "community_forum":
        raise HTTPException(
            status_code=400,
            detail="Only community_forum profiles can be testimonials "
                    "(FAERS records are not testimonials)."
        )
    row.is_testimonial = True
    db.commit()
    return {"episode_id": episode_id, "is_testimonial": True}


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

class RWEItemCtx(BaseModel):
    """One RWE item forwarded by the frontend with a chat message.

    RWE is kept strictly separate from scientific evidence: ``source_type``
    and ``evidence_tier`` let the Assistant distinguish testimonials /
    pharmacovigilance reports from clinical studies.
    """
    source: str
    source_type: str
    evidence_tier: str = "anecdotal"
    collection_method: str = "official_api"
    source_url: Optional[str] = ""
    external_id: Optional[str] = ""
    title: str = ""
    text: str = ""
    date: Optional[str] = ""
    language: str = "en"
    topic: str = ""
    treatment: Optional[str] = ""
    condition: Optional[str] = ""
    experience_type: str = "discussion"
    relevance: str = "unknown"
    relevance_reason: Optional[str] = ""
    privacy_status: str = "redacted"
    # Search-engine provenance (Phase: RWE Search Engine)
    matched_query: Optional[str] = ""
    matched_query_type: Optional[str] = ""
    source_language: str = "en"
    relevance_score: float = 0.0
    match_reason: Optional[str] = ""
    metadata: dict = {}

class SearchContext(BaseModel):
    """
    Active search context forwarded by the frontend with every chat message.
    Populated by the orchestrator output + raw collector results.

    ``articles`` is the scientific evidence; ``rwe_evidence`` is the optional
    RWE evidence. They are kept separate so the Assistant never conflates a
    testimonial with a clinical study.
    """
    original_query: str
    search_query: str               # English query actually sent to collectors
    detected_language: str          # ISO-639-1 code from orchestrator
    articles: List[SearchArticleCtx] = []
    rwe_evidence: List[RWEItemCtx] = []   # Feature: RWE convergence

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    language: Optional[str] = "en"         # ISO 639-1 code, e.g. "en" / "it"
    search_context: Optional[SearchContext] = None   # Feature 002: active search


# ── Level 3: Scientific Synthesis (reuses Level 2 results, no re-search) ──────

class SynthesisArticle(BaseModel):
    """One article already judged relevant by the Level 2 relational judge."""
    source: str
    title: str
    abstract: Optional[str] = ""
    url: Optional[str] = ""
    pmid: Optional[str] = ""
    doi: Optional[str] = ""
    nct_id: Optional[str] = ""
    year: Optional[str] = ""
    journal: Optional[str] = ""
    relevance_label: Optional[str] = ""
    relevance_score: Optional[float] = None
    relevance_reason: Optional[str] = ""


class SynthesisRelation(BaseModel):
    """The ClinicalRelation extracted by Level 2 (passed through from /search)."""
    original_query: str = ""
    agent: dict = {}
    event: dict = {}
    manifestation: dict = {}
    temporal: str = ""
    relation_type: str = "unknown"
    scientific_query: str = ""
    relation_phrases: list = []
    fallback_needed: bool = False


class SynthesisRequest(BaseModel):
    query: str
    search_query: Optional[str] = ""
    detected_language: Optional[str] = "en"
    language: Optional[str] = "en"
    relation: Optional[SynthesisRelation] = None
    articles: List[SynthesisArticle] = []


class CardSynthesisRequest(BaseModel):
    """On-demand per-card synthesis (FASE 8).

    Synthesises a SINGLE article or RWE record on demand — no auto-synthesis,
    no global batch. The user clicks 'Ricava sintesi' on a specific card.
    """
    query: str
    title: str
    abstract: str = ""
    text: str = ""
    source: str = ""
    url: str = ""
    pmid: str = ""
    doi: str = ""
    nct_id: str = ""
    external_id: str = ""
    source_type: str = ""       # scientific_article | community_forum | pharmacovigilance
    evidence_tier: str = ""     # RCT | anecdotal | spontaneous_report | …
    treatment: str = ""
    condition: str = ""
    language: str = "en"        # output language


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
                rwe_evidence=[
                    RWEItemCtx(**r) for r in sc_data.get("rwe_evidence", [])
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
    elif sc and sc.rwe_evidence:
        # No scientific articles, but RWE items are present — RWE drives the answer.
        search_block = (
            f"══ PRIORITY 1 — CURRENT SEARCH: no scientific articles retrieved "
            f"(query: '{sc.original_query}'). Real-world evidence is available below. ══\n"
            "══════════════════════════════════════════\n"
        )
    elif sc:
        # Search was performed but returned no articles and no RWE
        search_block = (
            f"══ PRIORITY 1 — CURRENT SEARCH: no articles retrieved "
            f"(query: '{sc.original_query}') ══\n"
            "Answer the user's question as directly as possible using general medical knowledge. "
            "If you must rely on general knowledge, state explicitly at the END of your response:\n"
            "  'Note: the current search returned no results. "
            "This answer is based on general medical knowledge.'\n"
            "══════════════════════════════════════════\n"
        )

    # ── RWE evidence block (kept strictly separate from scientific) ──
    # Built only when the frontend forwarded RWE items. RWE never overrides
    # scientific evidence; it supplements it in a distinct, labelled section.
    rwe_block = ""
    has_rwe = bool(sc and sc.rwe_evidence)
    if has_rwe:
        rwe_items = sc.rwe_evidence
        rwe_lines = []
        for i, it in enumerate(rwe_items, 1):
            tier_label = {
                "anecdotal": "anecdotal (community)",
                "spontaneous_report": "spontaneous report (pharmacovigilance)",
                "survey": "survey",
            }.get(it.evidence_tier, it.evidence_tier)
            src_label = {
                "reddit": "Reddit", "openfda_faers": "openFDA FAERS",
                "calvizie": "Calvizie.net", "hairlosstalk": "HairLossTalk",
                "hairlossexperiences": "HairLossExperiences",
                "maladiesrares": "MaladiesRaresInfo (FR)",
            }.get(it.source, it.source)
            date_part = f" ({it.date})" if it.date else ""
            treat_part = f" | Treatment: {it.treatment}" if it.treatment else ""
            text_excerpt = (it.text or "").strip()
            if len(text_excerpt) > 350:
                text_excerpt = text_excerpt[:340] + "…"
            # Search-engine provenance — lets the Assistant cite which query /
            # language surfaced each item and how it was matched.
            prov_parts = []
            if it.matched_query:
                prov_parts.append(f"matched_query='{it.matched_query}'")
            if it.matched_query_type:
                prov_parts.append(f"type={it.matched_query_type}")
            if it.source_language:
                prov_parts.append(f"lang={it.source_language}")
            if it.match_reason:
                prov_parts.append(f"match={it.match_reason}")
            if it.relevance_score:
                prov_parts.append(f"score={it.relevance_score:.2f}")
            prov_part = f" | {' '.join(prov_parts)}" if prov_parts else ""
            rwe_lines.append(
                f"  [{i}] [{src_label} — {tier_label}]{date_part}{treat_part}{prov_part}\n"
                f"      {it.title or '(no title)'}\n"
                f"      Text: {text_excerpt or 'N/A'}"
            )
        rwe_source_labels = sorted(set(
            {"reddit": "Reddit", "openfda_faers": "openFDA FAERS", "calvizie": "Calvizie.net", "hairlosstalk": "HairLossTalk", "hairlossexperiences": "HairLossExperiences", "maladiesrares": "MaladiesRaresInfo (FR)"}.get(i.source, i.source)
            for i in rwe_items
        ))
        rwe_block = (
            f"══ REAL-WORLD EVIDENCE (RWE) — {len(rwe_items)} item(s) from: "
            f"{', '.join(rwe_source_labels)} ══\n"
            "These are patient-reported experiences, community discussions, and "
            "spontaneous adverse-event reports. They are NOT clinical studies and must "
            "NOT be presented as scientific proof. Cite each as '[RWE: <Source>]'.\n\n"
            + "\n\n".join(rwe_lines)
            + "\n\n"
            "RWE RESPONSE RULES:\n"
            "- RWE is supplementary and explicitly lower-evidence than scientific studies.\n"
            "- NEVER present a testimonial or a FAERS report as established clinical fact.\n"
            "- If scientific articles are ALSO present (Priority 1 above), structure your "
            "answer with BOTH: a scientific section AND a 'Testimonianze e discussioni' "
            "section, plus a 'Confronto' (comparison) section noting where RWE agrees or "
            "diverges from the scientific evidence.\n"
            "- If ONLY RWE is present (no scientific articles), answer using the RWE items "
            "under a 'Testimonianze e discussioni' heading, and explicitly state that "
            "these are patient-reported experiences, not clinical evidence, and recommend "
            "consulting the scientific literature / a clinician.\n"
            "- Preserve provenance: every RWE claim must cite its [RWE: <Source>] origin.\n"
            "- Respect privacy_status: never attribute statements to named individuals.\n"
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
        "  2. REAL-WORLD EVIDENCE (RWE) — patient experiences, community discussions, and "
        "pharmacovigilance reports. Lower-evidence than scientific studies; never presented "
        "as clinical proof. Always kept in a distinct, labelled section.\n"
        "  3. HLEO DATABASE — supplementary only; never used as primary evidence when search "
        "articles are available.\n"
        "  4. GENERAL KNOWLEDGE — only when (1), (2) and (3) are all insufficient.\n\n"
        "You always:\n"
        "- Cite sources inline (e.g. '[PubMed]', '[Europe PMC]', '[RWE: Reddit]', "
        "'[RWE: openFDA FAERS]', '[HLEO DB]').\n"
        "- Keep scientific evidence and RWE strictly separate: never present a testimonial "
        "or a spontaneous adverse-event report as established clinical fact.\n"
        "- Acknowledge uncertainty; never overstate evidence strength.\n"
        "- Recommend consulting a qualified clinician for personal medical decisions.\n"
        "- When scientific search results are present, follow the CLINICAL REASONING ENGINE "
        "in Priority 1 exactly — synthesis and clinical judgment first, individual studies last.\n"
        "- When only RWE is present (no scientific articles), answer under a "
        "'Testimonianze e discussioni' heading using the RWE items, and explicitly state they "
        "are patient-reported experiences, not clinical evidence.\n"
        "- When both scientific and RWE are present, include BOTH a scientific section AND a "
        "'Testimonianze e discussioni' section, plus a 'Confronto' (comparison) section.\n"
        "- When no search results are present at all, answer concisely (3–6 sentences) using "
        "HLEO DB then general knowledge.\n\n"
        f"{search_block}"
        f"{rwe_block}"
        f"{db_header}\n\n"
        f"{p3_note}"
        f"{_lang_note}"
    )

    # ── Call LLM ───────────────────────────────────────────────────────────
    llm_messages = [{"role": "system", "content": system_prompt}]
    llm_messages.extend(messages_history)
    llm_messages.append({"role": "user", "content": body.message})

    from core.llm_guard import call_llm, LLMCallError, QuotaExhaustedError
    try:
        assistant_text = call_llm(
            client,
            messages=llm_messages,
            model="gpt-4o",
            temperature=0.2,
            max_tokens=1800 if (has_search_articles or has_rwe) else 800,
            operation="assistant_chat",
        )
    except QuotaExhaustedError as exc:
        return {"error": f"OpenAI quota esaurita: {exc}", "quota_exhausted": True}
    except LLMCallError as exc:
        logger.exception(f"/assistant/chat LLM failed — {exc}")
        return {"error": f"Assistant LLM call failed after retries: {exc}"}

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


# ── Level 3: Scientific Synthesis ─────────────────────────────────────────────

_SYNTHESIS_PROMPT = """You are a senior clinical evidence synthesizer. You are given:
1) a CLINICAL RELATION the user is investigating, and
2) a set of ALREADY-RETRIEVED, JUDGED-RELEVANT scientific articles (their titles, abstracts, and bibliographic IDs).

Your job: synthesize the evidence ACROSS articles into a structured scientific answer.
You must anchor every claim to the provided articles (cite by index and identifier). Do NOT invent articles, PMIDs, DOIs, or findings not present in the inputs. If the evidence is insufficient or contradictory, say so explicitly.

Return ONLY a JSON object with this exact schema:
{{
  "conclusion": "3-5 sentence clinical conclusion synthesizing what the evidence says about the relation",
  "evidence_summary": "where the studies agree/disagree, sample/population, study design limitations, gaps",
  "claims": [
    {{
      "article_index": <int, the [i] index>,
      "claim": "the specific finding/claim this article contributes regarding the relation",
      "evidence_type": "direct" | "indirect",
      "confidence": "high" | "moderate" | "low",
      "citation": "PMID xxx | DOI xxx | NCT xxx"
    }}
  ],
  "agreements": ["points where >=2 articles concur"],
  "contradictions": ["points where articles disagree"],
  "confidence": {{"level": "high" | "moderate" | "low", "rationale": "why this confidence level (n. articles, consistency, design)"}},
  "key_studies": ["3-5 most relevant citations"]
}}

Direct evidence = the article explicitly studies the relation (agent->manifestation).
Indirect evidence = the relation is inferable or only partially addressed.
"""

def _describe_synthesis_relation(rel: Optional[SynthesisRelation]) -> str:
    if rel is None:
        return "(no structured relation extracted — synthesize from the query and articles)"
    a = rel.agent.get("normalized", "") or rel.agent.get("term", "")
    m = rel.manifestation.get("normalized", "") or rel.manifestation.get("term", "")
    ev = rel.event.get("normalized", "") or rel.event.get("term", "")
    rt = rel.relation_type
    parts = [f"Original query: {rel.original_query}"]
    if rt == "adverse_effect":
        parts.append(f"Relation: {a} (via {ev}) may CAUSE/TRIGGER {m} as an adverse effect")
    elif rt == "efficacy":
        parts.append(f"Relation: {a} used to TREAT {m}")
    elif rt == "exposure_outcome":
        parts.append(f"Relation: {a} (via {ev}) leads to / is associated with {m}")
    else:
        parts.append(f"Relation: {a} -> {m}")
    if rel.temporal:
        parts.append(f"Temporal context: {rel.temporal}")
    if rel.relation_phrases:
        parts.append("Relation phrases: " + "; ".join(rel.relation_phrases))
    return "\n".join(parts)


@app.post("/synthesis")
def synthesize(body: SynthesisRequest):
    """Level 3: synthesize a structured scientific answer from Level 2 results.

    Receives the query, the ClinicalRelation extracted by /search (scientific),
    and the articles already judged relevant — it does NOT re-run retrieval.
    Returns conclusion, evidence summary, per-article claims (direct/indirect),
    agreements/contradictions, confidence, and key study citations.
    """
    import os, json as _json

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="Scientific synthesis requires OPENAI_API_KEY.")

    if not body.articles:
        return {"error": "No relevant articles provided for synthesis."}

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    # Format articles as a numbered block with identifiers for citation.
    art_lines = []
    for i, a in enumerate(body.articles):
        ident = a.pmid or a.doi or a.nct_id or a.url or ""
        ident_str = f"PMID {a.pmid}" if a.pmid else (f"DOI {a.doi}" if a.doi else (f"NCT {a.nct_id}" if a.nct_id else ident))
        meta_bits = [x for x in [a.journal, a.year] if x]
        meta_str = f" ({', '.join(meta_bits)})" if meta_bits else ""
        rel_str = ""
        if a.relevance_label:
            rel_str = f" | relevance: {a.relevance_label}"
            if a.relevance_score is not None:
                rel_str += f" ({a.relevance_score})"
        abstract = (a.abstract or "")[:1200]
        art_lines.append(
            f"[{i}] [{a.source}] {ident_str}{meta_str} — {a.title}\n    ABSTRACT: {abstract}{rel_str}"
        )
    articles_block = "\n".join(art_lines)

    relation_desc = _describe_synthesis_relation(body.relation)

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

    prompt = (
        f"{_SYNTHESIS_PROMPT}\n\n"
        f"=== CLINICAL RELATION ===\n{relation_desc}\n\n"
        f"=== RELEVANT ARTICLES ({len(body.articles)}) ===\n{articles_block}\n"
        f"{_lang_note}"
    )

    try:
        from core.llm_guard import call_llm_json, LLMCallError, QuotaExhaustedError
        parsed = call_llm_json(
            client,
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o",
            temperature=0.2,
            max_tokens=2200,
            operation="scientific_synthesis",
        )
    except QuotaExhaustedError as exc:
        return {"error": f"OpenAI quota esaurita: {exc}", "quota_exhausted": True}
    except LLMCallError as exc:
        logger.exception(f"/synthesis failed — {exc}")
        return {"error": f"Synthesis failed after retries: {exc}"}
    except _json.JSONDecodeError as exc:
        logger.exception(f"/synthesis: JSON parse error — {exc}")
        return {"error": f"Synthesis returned malformed JSON: {exc}"}
    except Exception as exc:
        logger.exception(f"/synthesis failed — {exc}")
        return {"error": f"Synthesis failed: {exc}"}

    return {
        **parsed,
        "relation": body.relation.model_dump() if body.relation else None,
        "query":    body.query,
        "language": body.language,
    }


# ── FASE 8: On-demand per-card synthesis ────────────────────────────────────
_CARD_SYNTHESIS_PROMPT = """You are a clinical evidence synthesiser. Produce a CONCISE on-demand synthesis of the SINGLE source below, in the context of the user's query.

Return ONLY a JSON object with these keys:
{
  "summary": "2-4 sentence summary of what THIS source says about the query",
  "key_points": ["point 1", "point 2", ...],
  "relevance_to_query": "direct|indirect|background",
  "evidence_type": "scientific|experiential|spontaneous_report",
  "confidence": {"level": "high|moderate|low", "rationale": "..."},
  "limitations": ["limitation 1", ...]
}

Rules:
- Summarise ONLY this one source. Do NOT invent data from other sources.
- For scientific articles: cite the study design and sample size if stated.
- For RWE/community posts: clearly label as patient-reported anecdotal evidence.
- For FAERS records: label as spontaneous adverse-event report, NOT clinical proof.
- key_points: 2-5 bullet points, each ≤120 chars.
- relevance_to_query: how directly this source addresses the user's question.
- Be neutral and evidence-based. Do NOT give medical advice.
"""


@app.post("/synthesis/card")
def synthesize_card(body: CardSynthesisRequest):
    """FASE 8: on-demand synthesis of a SINGLE article/RWE record.

    No auto-synthesis, no batch. The user explicitly requests synthesis for one
    card. Returns a compact structured summary with provenance (source id/type).
    """
    import os

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503,
                            detail="Card synthesis requires OPENAI_API_KEY.")

    content_text = (body.abstract or body.text or "").strip()
    if not content_text:
        return {"error": "No content to synthesise (abstract/text required)."}

    # Build provenance identifier
    ident = (body.pmid or body.doi or body.nct_id
             or body.external_id or body.url or body.source or "")
    if body.pmid:
        ident_str = f"PMID {body.pmid}"
    elif body.doi:
        ident_str = f"DOI {body.doi}"
    elif body.nct_id:
        ident_str = f"NCT {body.nct_id}"
    elif body.external_id:
        ident_str = f"ID {body.external_id}"
    else:
        ident_str = ident
    source_label = body.source or (body.source_type or "unknown")
    tier_label = body.evidence_tier or ""

    _LANG_MAP = {
        "it": "Italian", "en": "English", "fr": "French",
        "de": "German",  "es": "Spanish", "pt": "Portuguese",
    }
    _resp_lang = _LANG_MAP.get(body.language or "en", "English")
    _lang_note = (
        f"\n\nIMPORTANT: respond entirely in {_resp_lang}."
        if _resp_lang != "English" else ""
    )

    meta_bits = []
    if body.treatment:
        meta_bits.append(f"Treatment: {body.treatment}")
    if body.condition:
        meta_bits.append(f"Condition: {body.condition}")
    if tier_label:
        meta_bits.append(f"Evidence tier: {tier_label}")
    meta_str = "\n".join(meta_bits)

    prompt = (
        f"{_CARD_SYNTHESIS_PROMPT}\n\n"
        f"=== USER QUERY ===\n{body.query}\n\n"
        f"=== SOURCE ===\n"
        f"[{source_label}] {ident_str}\n"
        f"Title: {body.title}\n"
        + (f"{meta_str}\n" if meta_str else "")
        + f"Content:\n{content_text[:2000]}\n"
        f"{_lang_note}"
    )

    from openai import OpenAI
    from core.llm_guard import call_llm_json, LLMCallError, QuotaExhaustedError
    client = OpenAI(api_key=api_key)
    try:
        parsed = call_llm_json(
            client,
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o",
            temperature=0.2,
            max_tokens=800,
            operation="card_synthesis",
        )
    except QuotaExhaustedError as exc:
        return {"error": f"OpenAI quota esaurita: {exc}", "quota_exhausted": True}
    except LLMCallError as exc:
        logger.exception(f"/synthesis/card failed — {exc}")
        return {"error": f"Card synthesis failed after retries: {exc}"}
    except Exception as exc:
        logger.exception(f"/synthesis/card failed — {exc}")
        return {"error": f"Card synthesis failed: {exc}"}

    # Attach provenance
    parsed["source"] = source_label
    parsed["source_id"] = ident_str
    parsed["evidence_tier"] = tier_label
    parsed["query"] = body.query
    parsed["language"] = body.language
    return parsed


# ── FASE 15: AI Assistant — Scientific vs RWE structured comparison ────────

class CompareRequest(BaseModel):
    """FASE 15: structured comparison of scientific evidence vs RWE for a query."""
    query: str
    search_query: str = ""
    detected_language: str = "en"
    language: str = "en"
    scientific_articles: List[SearchArticleCtx] = []
    rwe_evidence: List[RWEItemCtx] = []


@app.post("/assistant/compare")
def assistant_compare(body: CompareRequest):
    """FASE 15: structured Scientific vs RWE comparison.

    Produces a structured comparison showing where scientific evidence and RWE
    agree, diverge, or where RWE fills gaps the literature doesn't cover.
    RWE is NEVER presented as clinical proof — always labelled as anecdotal /
    spontaneous report.
    """
    import os

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return {"error": "OPENAI_API_KEY not set."}

    if not body.scientific_articles and not body.rwe_evidence:
        return {"error": "No evidence provided for comparison."}

    from openai import OpenAI
    from core.llm_guard import call_llm_json, LLMCallError, QuotaExhaustedError
    client = OpenAI(api_key=api_key)

    # Format scientific articles
    sci_lines = []
    for i, a in enumerate(body.scientific_articles[:10]):
        ident = a.pmid or a.doi or a.nct_id or a.url or ""
        sci_lines.append(
            f"[S{i}] {a.source} | {ident} — {a.title}\n"
            f"    {(a.abstract or '')[:600]}"
        )
    sci_block = "\n".join(sci_lines) or "(no scientific articles)"

    # Format RWE items
    rwe_lines = []
    for i, r in enumerate(body.rwe_evidence[:10]):
        rwe_lines.append(
            f"[R{i}] {r.source} ({r.evidence_tier}) | {r.external_id or ''} — {r.title}\n"
            f"    treatment={r.treatment or '?'}; {(r.text or '')[:500]}"
        )
    rwe_block = "\n".join(rwe_lines) or "(no RWE items)"

    _LANG_MAP = {
        "it": "Italian", "en": "English", "fr": "French",
        "de": "German",  "es": "Spanish", "pt": "Portuguese",
    }
    _resp_lang = _LANG_MAP.get(body.language or "en", "English")
    _lang_note = (
        f"\n\nIMPORTANT: respond entirely in {_resp_lang}."
        if _resp_lang != "English" else ""
    )

    prompt = (
        "You are a clinical evidence analyst. Compare the SCIENTIFIC evidence "
        "with the REAL-WORLD EVIDENCE (RWE) for the user's query.\n\n"
        "Return ONLY a JSON object with these keys:\n"
        "{\n"
        '  "scientific_consensus": "1-3 sentences: what does the literature conclude?",\n'
        '  "rwe_consensus": "1-3 sentences: what do patients/reports describe?",\n'
        '  "agreements": ["point where RWE supports the literature", ...],\n'
        '  "divergences": ["point where RWE differs from the literature", ...],\n'
        '  "gaps_filled_by_rwe": ["aspect RWE covers that the literature does not", ...],\n'
        '  "evidence_quality_note": "reminder: RWE is anecdotal/spontaneous, not proof",\n'
        '  "practical_takeaway": "1-2 sentences: what this means for the patient",\n'
        '  "confidence": {"level": "high|moderate|low", "rationale": "..."}\n'
        "}\n\n"
        "Rules:\n"
        "- NEVER present RWE as clinical proof. Always label it as anecdotal or "
        "spontaneous report.\n"
        "- Be neutral and evidence-based. Do NOT give medical advice.\n"
        "- If one side is empty, state that explicitly.\n"
        f"{_lang_note}\n\n"
        f"=== USER QUERY ===\n{body.query}\n\n"
        f"=== SCIENTIFIC EVIDENCE ({len(body.scientific_articles)}) ===\n{sci_block}\n\n"
        f"=== REAL-WORLD EVIDENCE ({len(body.rwe_evidence)}) ===\n{rwe_block}\n"
    )

    try:
        parsed = call_llm_json(
            client,
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o",
            temperature=0.2,
            max_tokens=1500,
            operation="assistant_compare",
        )
    except QuotaExhaustedError as exc:
        return {"error": f"OpenAI quota esaurita: {exc}", "quota_exhausted": True}
    except LLMCallError as exc:
        logger.exception(f"/assistant/compare failed — {exc}")
        return {"error": f"Comparison failed after retries: {exc}"}
    except Exception as exc:
        logger.exception(f"/assistant/compare failed — {exc}")
        return {"error": f"Comparison failed: {exc}"}

    parsed["query"] = body.query
    parsed["language"] = body.language
    parsed["scientific_count"] = len(body.scientific_articles)
    parsed["rwe_count"] = len(body.rwe_evidence)
    return parsed


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
    from core.llm_guard import call_llm_json, LLMCallError, QuotaExhaustedError
    client = OpenAI(api_key=api_key)
    try:
        result = call_llm_json(
            client,
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o",
            temperature=0.2,
            max_tokens=1500,
            operation="translate",
        )
    except (LLMCallError, QuotaExhaustedError) as exc:
        raise HTTPException(status_code=503, detail=f"Translation failed: {exc}")

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
