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


def _split_entities(entities) -> Tuple[List[str], List[str], List[str]]:
    """Split recognised entities into (drug_terms, event_terms, context_terms).

    event_terms = symptoms + conditions (the 'event' side of a drug→event query).
    drug_terms  = drugs + active ingredients (the 'exposure' side).
    context_terms = any other entity types (procedures, organs, …) — light context boost.
    Each list carries the canonical name plus a few aliases, lowercase.
    """
    drugs: List[str] = []
    events: List[str] = []
    context: List[str] = []
    for etype, canonical, _ in entities or []:
        canon = canonical.lower()
        if etype == "drug" or etype == "active_ingredient":
            drugs.append(canon)
            for alias in DRUG_ALIASES.get(canonical, [])[:5]:
                if len(alias) > 3:
                    drugs.append(alias.lower())
        elif etype in ("symptom", "adverse_effect", "disease", "condition"):
            events.append(canon)
            for alias in SYMPTOM_ALIASES.get(canonical, [])[:5]:
                if len(alias) > 3:
                    events.append(alias.lower())
            for alias in CONDITION_ALIASES.get(canonical, [])[:5]:
                if len(alias) > 3:
                    events.append(alias.lower())
        else:
            context.append(canon)
    return drugs, events, context


# Semantic equivalents: terms that, when the event side of the query mentions
# one of these, are treated as a match when found in a record's text/reactions.
# This is the "hair shedding ≈ hair loss ≈ alopecia ≈ hair fall" bridge.
_EVENT_SYNONYM_GROUPS: List[Tuple[List[str], List[str]]] = [
    (
        ["hair loss", "hair shedding", "alopecia", "hair fall", "hairfall",
         "caduta capelli", "perdita di capelli", "perdita capelli",
         "caduta dei capelli", "perdita dei capelli",
         "chute de cheveux", "perte de cheveux", "perte de cheveu",
         "haarausfall", "caída del cabello", "pérdida de cabello",
         "hairedropping", "hairfallout"],
        ["hair loss", "hair shedding", "alopecia", "hair fall", "hairfall",
         "caduta", "perdita", "chute", "haarausfall", "caída",
         "shedding", "thinning", "bald", "fell out", "fall out", "falling out",
         "capelli", "cheveux"],
    ),
    (
        ["sexual dysfunction", "disfunzione sessuale",
         "libido loss", "lost libido", "low sex drive",
         "calo del desiderio", "desiderio sessuale", "fame sessuale",
         "dysfonction sexuelle", "libido vermindert",
         "disfunción sexual", "disfunção sexual"],
        ["sexual dysfunction", "libido", "sex drive", "erection", "erectile",
         "impotence", "impotenza", "désir", "desiderio",
         "sessual", "sex", "libido"],
    ),
    (
        ["depression", "depresso", "deprimé", "deprimida"],
        ["depression", "depressive", "depressed", "depress", "depresso",
         "deprim", "mood"],
    ),
]


def _event_match_terms(query_event_terms: List[str]) -> set:
    """Expand query-side event terms to include their semantic equivalents.

    Returns a set of lowercase substrings/tokens to look for in the record.
    When a query event term (canonical or alias) belongs to a known synonym
    group, all members of that group (and their match-tokens) are admitted.
    """
    out = set(t.lower() for t in query_event_terms if t)
    for members, match_tokens in _EVENT_SYNONYM_GROUPS:
        # Does the query reference any member of this concept group?
        if any(m.lower() in out for m in members):
            out.update(m.lower() for m in members)
            out.update(t.lower() for t in match_tokens)
    return out


def _event_match(item_text_lower: str, query_event_terms: List[str]) -> Tuple[float, List[str]]:
    """Score how well the item's event/reaction text matches the query's event.

    Returns (0..1, matched_terms). A 0 means the record's reactions/text do not
    express the event the query asks about (the "dutasteride + loss of
    proprioception" for a "dutasteride induced hair shedding" query case).
    """
    if not query_event_terms:
        # No event was recognised in the query — treat event_match as neutral.
        return 0.5, []
    terms = _event_match_terms(query_event_terms)
    hits = [t for t in terms if t and t in item_text_lower]
    if not hits:
        return 0.0, []
    # Saturate quickly: a couple of distinct event hits = strong event match.
    score = min(1.0, 0.5 + 0.25 * len(set(hits)))
    return score, sorted(set(hits))[:6]


def _drug_match(item_text_lower: str, treatment_lower: str, query_drug_terms: List[str]) -> Tuple[float, List[str]]:
    """Score how well the item references the query's drug/exposure.

    For authoritative sources (openFDA), the drug is trusted to be present
    server-side, so the treatment field is checked too.
    """
    if not query_drug_terms:
        return 0.0, []
    hay = f"{item_text_lower} {treatment_lower}"
    hits = [t for t in query_drug_terms if t and t in hay]
    if not hits:
        return 0.0, []
    score = min(1.0, 0.6 + 0.2 * len(set(hits)))
    return score, sorted(set(hits))[:4]


def _score_item(
    it: RWEItem,
    query: str,
    entities: list,
    is_authoritative_match: bool,
) -> Tuple[float, str]:
    """
    Semi-semantic relevance score (0.0–1.0) + human match reason.

    Separates the two sides of a drug→event query:
      - drug_match   : does the record reference the query's drug/exposure?
      - event_match  : does the record's reaction/text express the query's
                       event (semantically, incl. IT/FR/ES/DE equivalents)?

    overall_relevance = weighted blend; drug-only matches (no event) score
    LOW so a dutasteride+proprioception record does not surface for a
    "dutasteride induced hair shedding" query.

    For authoritative sources (openFDA), the drug is trusted as present
    (server-side match), but the event is STILL verified against the record's
    reaction text — authoritative does not override the event requirement.
    """
    q_tokens = _tokens(query)
    body = f"{it.title} {it.text}".lower()
    treatment_lower = (it.treatment or "").lower()
    body_tokens = _tokens(f"{it.title} {it.text} {it.treatment or ''} {it.condition or ''}")
    overlap = q_tokens & body_tokens
    token_score = len(overlap) / max(1, len(q_tokens)) if q_tokens else 0.0

    drug_terms, event_terms, context_terms = _split_entities(entities)

    drug_score, drug_hits = _drug_match(body, treatment_lower, drug_terms)
    event_score, event_hits = _event_match(body, event_terms)

    # Context boost (light) — population/condition co-occurrence.
    ctx_hits = [t for t in context_terms if t and t in body]
    ctx_score = min(0.15, 0.05 * len(ctx_hits)) if context_terms else 0.0

    # ── overall_relevance ──────────────────────────────────────────────────
    # When the query has BOTH a drug and an event, both must be present for a
    # high score. A drug-only match (event_score=0) is heavily penalised.
    has_drug_and_event = bool(drug_terms) and bool(event_terms)

    if is_authoritative_match:
        # openFDA: trust the drug match (server-side), but still require the event.
        drug_component = 1.0   # authoritative drug match is trusted present
    else:
        drug_component = drug_score

    if has_drug_and_event:
        if event_score == 0.0:
            # The record does NOT express the queried event → low relevance.
            base = 0.15 * drug_component + 0.0 + ctx_score
            reason = (f"drug_match_only (drug={drug_hits[:2]}; "
                      f"event_missing — queried event not in record)")
            return round(min(1.0, base), 3), reason
        # Both sides present — strong relevance.
        base = 0.45 * drug_component + 0.45 * event_score + 0.10 * token_score + ctx_score
        reason = (f"drug+event_match (drug={drug_hits[:2]}; "
                  f"event={event_hits[:3]}; tokens={len(overlap)})")
        return round(min(1.0, base), 3), reason

    # Query has only a drug, or only an event, or neither recognised.
    if drug_terms and not event_terms:
        # Drug-only query: drug match is the signal.
        base = 0.6 * drug_component + 0.3 * token_score + ctx_score
        reason = f"drug_match (drug={drug_hits[:2]}; tokens={len(overlap)})"
        return round(min(1.0, base), 3), reason
    if event_terms and not drug_terms:
        base = 0.6 * event_score + 0.3 * token_score + ctx_score
        reason = f"event_match (event={event_hits[:3]}; tokens={len(overlap)})"
        return round(min(1.0, base), 3), reason

    # No structured entities — fall back to token/entity overlap.
    ent_terms = _entity_terms(entities)
    ent_hits = [t for t in ent_terms if t and t in body]
    entity_score = min(1.0, len(ent_hits) / 3.0) if ent_terms else 0.0
    if token_score >= 0.5 and entity_score > 0:
        base = max(0.5 * token_score + 0.5 * entity_score, 0.75)
        reason = f"exact_keyword+semantic ({len(overlap)} tokens, {len(ent_hits)} entities)"
    elif token_score > 0:
        base = 0.5 * token_score + 0.5 * entity_score
        reason = f"exact_keyword ({len(overlap)}/{len(q_tokens)} tokens: {sorted(overlap)})"
    elif entity_score > 0:
        base = max(0.5 * token_score + 0.5 * entity_score, 0.4)
        reason = f"semantic_entity_match ({len(ent_hits)} entities: {ent_hits[:3]})"
    else:
        base = 0.0
        reason = "no keyword or entity overlap"
    return round(min(1.0, base), 3), reason


def relevance_filter(
    items: List[RWEItem],
    query: str,
    entities: Optional[list] = None,
    authoritative_sources: Optional[set] = None,
    min_score: float = 0.20,
) -> List[RWEItem]:
    """
    Semi-semantic relevance filter — respects the FULL intent of the query.

    Splits the query into a DRUG/exposure side and an EVENT/symptom side, and
    requires BOTH for a high score. A record that merely references the queried
    drug but expresses an unrelated event (e.g. dutasteride + loss of
    proprioception for a "dutasteride induced hair shedding" query) scores LOW
    and is filtered out.

    Authoritative sources (openFDA): the drug match is trusted server-side,
    but the event is STILL verified against the record's reaction text —
    authoritative does not override the event requirement.

    Marks each item with ``relevance`` (relevant|irrelevant), ``relevance_reason``,
    ``relevance_score`` (0–1), and ``match_reason``. Keeps only relevant items
    (score ≥ min_score). Default min_score=0.20 filters drug-only false positives.

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
        from core.rwe.calvizie_collector import CalvizieCollector
        from core.rwe.hairlosstalk_collector import HairLossTalkCollector
        from core.rwe.hairlossexperiences_collector import HairLossExperiencesCollector
        from core.rwe.maladiesrares_collector import MaladiesRaresCollector
        self.reddit = RedditRWEAdapter()
        self.openfda = OpenFDACollector()
        self.calvizie = CalvizieCollector()
        self.hairlosstalk = HairLossTalkCollector()
        self.hairlossexperiences = HairLossExperiencesCollector()
        self.maladiesrares = MaladiesRaresCollector()
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
        sources = sources or [
            "reddit", "openfda_faers", "calvizie",
            "hairlosstalk", "hairlossexperiences", "maladiesrares",
        ]

        per_source_limits = {
            "reddit": limit,
            "openfda_faers": limit,
            "calvizie": limit,
            "hairlosstalk": limit,
            "hairlossexperiences": limit,
            "maladiesrares": limit,
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

        if "calvizie" in sources:
            items, status = self._collect_source(
                "calvizie", self.calvizie, plan, per_source_limits["calvizie"]
            )
            source_status["calvizie"] = status
            all_items.extend(items)

        if "hairlosstalk" in sources:
            items, status = self._collect_source(
                "hairlosstalk", self.hairlosstalk, plan, per_source_limits["hairlosstalk"]
            )
            source_status["hairlosstalk"] = status
            all_items.extend(items)

        if "hairlossexperiences" in sources:
            items, status = self._collect_source(
                "hairlossexperiences", self.hairlossexperiences,
                plan, per_source_limits["hairlossexperiences"],
            )
            source_status["hairlossexperiences"] = status
            all_items.extend(items)

        if "maladiesrares" in sources:
            items, status = self._collect_source(
                "maladiesrares", self.maladiesrares, plan, per_source_limits["maladiesrares"]
            )
            source_status["maladiesrares"] = status
            all_items.extend(items)

        # ── 2. Annotate relevance for ALL items (do not filter them out yet) ──
        before = len(all_items)

        relevance_query = plan.translated_query or plan.original_query
        # Compute relevance_score/reason for every item but keep all items (min_score=0.0)
        all_items = relevance_filter(
            all_items,
            relevance_query,
            entities=plan.entities,
            min_score=0.0,
        )

        # ── 3. Deduplicate across (query × source), keep the copy with highest relevance_score ──
        after_scored = len(all_items)
        all_items = deduplicate(all_items)
        after = len(all_items)

        # ── 4. Sort all items by relevance_score (descending)
        all_items.sort(key=lambda it: (it.relevance_score or 0.0), reverse=True)

        totals = {
            "retrieved": before,
            "deduped_removed": before - after,
            "unique": after,
            # Keep 'relevant' as a diagnostic: count of items with score >= 0.20
            "relevant": sum(1 for it in all_items if (it.relevance_score or 0.0) >= 0.20),
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
