"""
RWE search pipeline.

Flow (analogous to the scientific pipeline, kept fully separate):
    User Question
    → orchestrator (lang detect + translate to English)
    → RWE collectors (Reddit adapter, openFDA FAERS, ...)
    → normalization (RawTestimonial/FAERS → RWEItem)
    → deduplication (URL / external_id)
    → relevance filtering (keyword overlap, non-authoritative)
    → provenance stamped (source, collection_method, evidence_tier)
    → RWE results

Independent of the scientific search — calling /rwe/search never touches
PubMed/Europe PMC/ClinicalTrials.gov and never mutates scientific state.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

from core.orchestrator import QueryOrchestrator
from core.rwe.models import RWEItem, RWESearchResult

logger = logging.getLogger(__name__)


# ─── Relevance filtering ─────────────────────────────────────────────────────

_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "with",
    "my", "i", "is", "are", "was", "it", "this", "that", "from", "by",
    "di", "il", "la", "per", "che", "e", "un", "una", "del", "della",
}


def _tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-zà-öø-ÿ]+", (text or "").lower())
            if len(w) > 2 and w not in _STOPWORDS}


def relevance_filter(items: List[RWEItem], query: str) -> List[RWEItem]:
    """
    Lightweight keyword-overlap relevance filter (non-authoritative).
    Marks items relevant/irrelevant with a reason. Keeps only relevant.

    This is NOT a clinical judgment — it just discards obviously-off-topic
    items (e.g. generic forum chatter that doesn't mention the query terms).
    """
    q_tokens = _tokens(query)
    if not q_tokens:
        for it in items:
            it.relevance = "unknown"
            it.relevance_reason = "No query terms available."
        return items

    kept: List[RWEItem] = []
    for it in items:
        # NOTE: it.topic is the query that surfaced the item, so it always
        # overlaps with the query — exclude it from the relevance body to
        # avoid a false "relevant" signal on off-topic content.
        # Include treatment/condition so drug-name matches on FAERS reports count.
        body = f"{it.title} {it.text} {it.treatment or ''} {it.condition or ''}".lower()
        body_tokens = _tokens(body)
        overlap = q_tokens & body_tokens
        score = len(overlap)
        if score >= 1:
            it.relevance = "relevant"
            it.relevance_reason = (
                f"Keyword overlap ({score}/{len(q_tokens)}): "
                f"{', '.join(sorted(overlap))}"
            )
            kept.append(it)
        else:
            it.relevance = "irrelevant"
            it.relevance_reason = "No query-term overlap."
    return kept


# ─── Deduplication ──────────────────────────────────────────────────────────

def deduplicate(items: List[RWEItem]) -> List[RWEItem]:
    """Deduplicate by source + external_id (or source_url as fallback)."""
    seen = set()
    unique: List[RWEItem] = []
    for it in items:
        key = (it.source, it.external_id or it.source_url or it.title)
        if key in seen:
            continue
        seen.add(key)
        unique.append(it)
    return unique


# ─── Pipeline ───────────────────────────────────────────────────────────────


class RWEPipeline:
    """
    Orchestrates RWE collectors. Independent from the scientific HLEOPipeline.

    Usage:
        pipe = RWEPipeline()
        result = pipe.search("finasteride hair loss")
    """

    def __init__(self) -> None:
        from core.rwe.reddit_adapter import RedditRWEAdapter
        from core.rwe.openfda_collector import OpenFDACollector
        self.reddit = RedditRWEAdapter()
        self.openfda = OpenFDACollector()

    def search(
        self,
        query: str,
        limit: int = 15,
        sources: Optional[List[str]] = None,
    ) -> RWESearchResult:
        """
        Run the RWE pipeline.

        Args:
            query: user query (any language).
            limit: per-source item cap.
            sources: restrict to a subset; None = all usable sources.
        """
        orchestrator = QueryOrchestrator()
        orch = orchestrator.process(query)

        search_query = orch.search_query or query
        sources = sources or ["reddit", "openfda_faers"]

        all_items: List[RWEItem] = []
        source_status: dict = {}

        # Reddit (PRAW OAuth2)
        if "reddit" in sources:
            items, status, reason = self.reddit.search_with_status(
                search_query, limit=limit
            )
            source_status["reddit"] = status
            if status == self.reddit.STATUS_OK:
                all_items.extend(items)

        # openFDA FAERS (official API, no key required)
        if "openfda_faers" in sources:
            # FAERS queries work best with the drug name; use the search query.
            items, status, reason = self.openfda.search_with_status(
                search_query, limit=limit
            )
            source_status["openfda_faers"] = status
            # OpenFDACollector exposes STATUS_OK as a module constant; compare by value.
            if status == "ok":
                all_items.extend(items)

        # Normalize provenance is already stamped per-item by collectors.
        # Deduplicate across sources.
        before = len(all_items)
        all_items = deduplicate(all_items)
        after = len(all_items)

        # Relevance filter (non-authoritative keyword overlap)
        all_items = relevance_filter(all_items, search_query)

        totals = {
            "retrieved": before,
            "deduped_removed": before - after,
            "unique": after,
            "relevant": len(all_items),
        }

        return RWESearchResult(
            query=query,
            search_query=search_query,
            detected_language=orch.detected_language,
            totals=totals,
            items=all_items,
            source_status=source_status,
        )
