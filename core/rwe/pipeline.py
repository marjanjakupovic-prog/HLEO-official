"""
RWE search pipeline — autonomous search engine.

Flow (analogous to the scientific pipeline, kept fully separate):
    User Question
      → RWEQueryEngine (language detect + prepare + translate + expand)
      → RWE collectors (Reddit adapter, openFDA FAERS, ...) per expanded query
      → normalization (RawTestimonial/FAERS → RWEItem)
      → deduplication (URL / external_id, keep best matched_query)
      → semi-semantic relevance (authoritative source + token + entity + synonym)
      → provenance stamped (matched_query, source_language, match_reason, score)
      → RWE results

Independent of the scientific search — calling /rwe/search never touches
PubMed/Europe PMC/ClinicalTrials.gov and never mutates scientific state.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

from core.biomedical_kb import (
    DRUG_ALIASES,
    CONDITION_ALIASES,
    SYMPTOM_ALIASES,
)
from core.orchestrator import QueryOrchestrator  # noqa: F401  (patched by tests via core.rwe.pipeline.QueryOrchestrator)
from core.rwe.models import RWEItem, RWESearchResult
from core.rwe.query_engine import RWEQueryEngine

logger = logging.getLogger(__name__)


# ─── Relevance scoring (semi-semantic) ───────────────────────────────────────

_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "with",
    "my", "i", "is", "are", "was", "it", "this", "that", "from", "by",
    "di", "il", "la", "per", "che", "e", "un", "una", "del", "della",
    "can", "cause", "does", "may", "will", "you", "your",
}

# Sources whose own search is authoritative: when they return an item for a
# query, the match is trusted even if the query term is absent from the item's
# text fields (e.g. openFDA matched on a drug buried in the report's drug list).
_AUTHORITATIVE_SOURCES = {"openfda_faers"}


def _tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-zà-öø-ÿ]+", (text or "").lower())
            if len(w) > 2 and w not in _STOPWORDS}


def _entity_terms(entities) -> List[str]:
    """Flat list of lowercase canonical names + their aliases for matching."""
    terms: List[str] = []
    for etype, canonical, _ in entities or []:
        terms.append(canonical.lower())
        for d in (DRUG_ALIASES, CONDITION_ALIASES, SYMPTOM_ALIASES):
            if canonical in d:
                for alias in d[canonical][:5]:
                    if len(alias) > 3:
                        terms.append(alias.lower())
                break
    return terms


def _score_item(
    it: RWEItem,
    query: str,
    entities: list,
    is_authoritative_match: bool,
) -> Tuple[float, str]:
    """
    Semi-semantic relevance score (0.0–1.0) + human match reason.

    Combines:
      - authoritative source match (trusted server-side match → high floor)
      - exact token overlap with the matched query
      - entity overlap (recognised biomedical entities in the item text)
      - synonym overlap (aliases of recognised entities)
    """
    q_tokens = _tokens(query)
    body = f"{it.title} {it.text} {it.treatment or ''} {it.condition or ''}".lower()
    body_tokens = _tokens(body)
    overlap = q_tokens & body_tokens
    token_score = len(overlap) / max(1, len(q_tokens)) if q_tokens else 0.0

    ent_terms = _entity_terms(entities)
    ent_hits = [t for t in ent_terms if t and t in body]
    entity_score = min(1.0, len(ent_hits) / 3.0) if ent_terms else 0.0

    # Weighted blend: token overlap is the primary signal, entity overlap a
    # semantic booster. Authoritative source matches get a high floor so they
    # survive even when the text fields don't echo the query term verbatim
    # (the openFDA case: the drug is in the report but not in the reaction text).
    base = 0.5 * token_score + 0.5 * entity_score
    if is_authoritative_match:
        score = max(0.6, base + 0.35)
        reason = (
            f"authoritative_source_match (server-side match trusted; "
            f"token={token_score:.2f}, entity={entity_score:.2f})"
        )
    elif token_score >= 0.5 and entity_score > 0:
        score = max(base, 0.75)
        reason = f"exact_keyword+semantic ({len(overlap)} tokens, {len(ent_hits)} entities)"
    elif token_score > 0:
        score = base
        reason = f"exact_keyword ({len(overlap)}/{len(q_tokens)} tokens: {sorted(overlap)})"
    elif entity_score > 0:
        score = max(base, 0.4)
        reason = f"semantic_entity_match ({len(ent_hits)} entities: {ent_hits[:3]})"
    else:
        score = 0.0
        reason = "no keyword or entity overlap"
    return round(min(1.0, score), 3), reason


def relevance_filter(
    items: List[RWEItem],
    query: str,
    entities: Optional[list] = None,
    authoritative_sources: Optional[set] = None,
    min_score: float = 0.01,
) -> List[RWEItem]:
    """
    Semi-semantic relevance filter.

    Marks each item with ``relevance`` (relevant|irrelevant), ``relevance_reason``,
    ``relevance_score`` (0–1), and ``match_reason``. Keeps only relevant items
    (score ≥ min_score). Items from authoritative sources that matched the query
    server-side are always kept — their match is trusted even when the query term
    is absent from the item's text fields (fixes the openFDA shedding case).

    Backward compatible: ``entities`` and ``authoritative_sources`` default to
    None/empty so the call ``relevance_filter(items, query)`` still works.
    """
    q_tokens = _tokens(query)
    auth_sources = authoritative_sources or set()
    entities = entities or []

    if not q_tokens and not entities:
        for it in items:
            it.relevance = "unknown"
            it.relevance_reason = "No query terms or entities available."
            it.relevance_score = 0.0
            it.match_reason = "no_signal"
        return items

    kept: List[RWEItem] = []
    for it in items:
        # Authoritative = the source's own search is trusted (hardcoded set),
        # optionally restricted by the caller via authoritative_sources.
        is_auth = it.source in _AUTHORITATIVE_SOURCES and (
            not auth_sources or it.source in auth_sources
        )
        score, reason = _score_item(it, query, entities, is_auth)
        it.relevance_score = score
        it.match_reason = reason
        if score >= min_score:
            it.relevance = "relevant"
            it.relevance_reason = reason
            kept.append(it)
        else:
            it.relevance = "irrelevant"
            it.relevance_reason = reason
    return kept


# ─── Deduplication ──────────────────────────────────────────────────────────

def deduplicate(items: List[RWEItem]) -> List[RWEItem]:
    """
    Deduplicate by source + external_id (or source_url as fallback).

    When the same item is surfaced by multiple expanded queries, the copy with
    the highest relevance_score is kept (so the best matched_query provenance
    survives).
    """
    best: dict = {}
    order: List = []
    for it in items:
        key = (it.source, it.external_id or it.source_url or it.title)
        if key not in best:
            best[key] = it
            order.append(key)
        else:
            if (it.relevance_score or 0.0) > (best[key].relevance_score or 0.0):
                best[key] = it
    return [best[k] for k in order]


# ─── Pipeline ───────────────────────────────────────────────────────────────


class RWEPipeline:
    """
    Orchestrates RWE collectors behind an autonomous query engine.

    Independent from the scientific HLEOPipeline.

    Usage:
        pipe = RWEPipeline()
        result = pipe.search("La finasteride può causare shedding iniziale?")
    """

    def __init__(self) -> None:
        from core.rwe.reddit_adapter import RedditRWEAdapter
        from core.rwe.openfda_collector import OpenFDACollector
        self.reddit = RedditRWEAdapter()
        self.openfda = OpenFDACollector()
        self._engine = RWEQueryEngine()

    def search(
        self,
        query: str,
        limit: int = 15,
        sources: Optional[List[str]] = None,
    ) -> RWESearchResult:
        """
        Run the RWE search pipeline.

        Args:
            query: user query (any language).
            limit: per-source item cap per expanded query.
            sources: restrict to a subset; None = all usable sources.
        """
        # ── 1. Build the query plan (detect → prepare → translate → expand) ──
        plan = self._engine.plan(query)
        sources = sources or ["reddit", "openfda_faers"]

        per_source_limits = {
            "reddit": limit,
            "openfda_faers": limit,
        }

        all_items: List[RWEItem] = []
        source_status: dict = {}

        if "reddit" in sources:
            items, status = self._collect_source(
                "reddit", self.reddit, plan, per_source_limits["reddit"]
            )
            source_status["reddit"] = status
            all_items.extend(items)

        if "openfda_faers" in sources:
            items, status = self._collect_source(
                "openfda_faers", self.openfda, plan, per_source_limits["openfda_faers"]
            )
            source_status["openfda_faers"] = status
            all_items.extend(items)

        # ── 2. Deduplicate across (query × source), keep best matched_query ──
        before = len(all_items)
        all_items = deduplicate(all_items)
        after = len(all_items)

        # ── 3. Semi-semantic relevance filtering ────────────────────────────
        relevance_query = plan.translated_query or plan.original_query
        all_items = relevance_filter(
            all_items,
            relevance_query,
            entities=plan.entities,
        )
        all_items.sort(key=lambda it: it.relevance_score, reverse=True)

        totals = {
            "retrieved": before,
            "deduped_removed": before - after,
            "unique": after,
            "relevant": len(all_items),
            "queries_used": len(plan.expanded_queries),
        }

        return RWESearchResult(
            query=query,
            original_query=plan.original_query,
            search_query=relevance_query,
            detected_language=plan.detected_language,
            translated_query=plan.translated_query,
            translation_applied=plan.translation_applied,
            expanded_queries=plan.to_dict()["expanded_queries"],
            totals=totals,
            items=all_items,
            source_status=source_status,
        )

    # ── Source collection ─────────────────────────────────────────────────────

    def _collect_source(
        self,
        name: str,
        collector,
        plan,
        cap: int,
    ) -> Tuple[List[RWEItem], str]:
        """
        Run a collector across every expanded query, stamping each item with the
        matched_query provenance. Stops early once ``cap`` items are collected.

        Returns (items, aggregate_status). The aggregate status is the best
        status seen across the expanded queries.
        """
        collected: List[RWEItem] = []
        statuses: List[str] = []
        seen_ids: set = set()

        for eq in plan.expanded_queries:
            if len(collected) >= cap:
                break
            try:
                items, status, reason = collector.search_with_status(
                    eq.query, limit=max(1, cap - len(collected))
                )
            except Exception as exc:
                logger.warning(f"RWE collector {name} failed for '{eq.query}': {exc}")
                statuses.append("network_error")
                continue
            statuses.append(status)
            if status != "ok":
                continue
            for it in items:
                key = (it.source, it.external_id or it.source_url)
                if key in seen_ids:
                    continue
                seen_ids.add(key)
                it.matched_query = eq.query
                it.matched_query_type = eq.expansion_type
                it.source_language = eq.source_language
                it.topic = eq.query
                collected.append(it)
                if len(collected) >= cap:
                    break

        # Aggregate: prefer ok > no_results > no_credentials > error
        status_rank = {
            "ok": 0, "no_results": 1, "no_credentials": 2,
            "auth_error": 3, "rate_limited": 4, "network_error": 5,
        }
        agg = "no_results"
        if statuses:
            agg = min(statuses, key=lambda s: status_rank.get(s, 9))
        return collected, agg
