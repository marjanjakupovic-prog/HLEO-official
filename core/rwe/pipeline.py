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
from typing import Dict, List, Optional, Tuple

from core.biomedical_kb import (
    DRUG_ALIASES,
    CONDITION_ALIASES,
    SYMPTOM_ALIASES,
)
from core.orchestrator import QueryOrchestrator  # noqa: F401  (patched by tests via core.rwe.pipeline.QueryOrchestrator)
from core.rwe.intent import GENERIC_EVENT_TERMS, merged_sides
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


# Generic "any adverse event" outcome terms (e.g. "finasteride side effects"):
# the user asks about ANY event, so the event side is satisfied by any known
# adverse-event vocabulary rather than by one literal phrase.
# Canonical definition lives in core.rwe.intent (shared with merged_sides,
# which must not inject vocabulary synonyms into generic-event side-sets).
_GENERIC_EVENT_TERMS = GENERIC_EVENT_TERMS

# Standard adverse-event vocabulary used ONLY for generic any-event queries
# ("finasteride side effects"): the event side is satisfied when the record
# mentions any common adverse reaction. MedDRA-style common terms, EN+IT.
_GENERIC_EVENT_VOCAB = {
    "nausea", "vomiting", "vomito", "diarrhoea", "diarrhea", "diarrea",
    "headache", "mal di testa", "dizziness", "dizzy", "vertigo", "vertigini",
    "capogiri", "insomnia", "insonnia", "fatigue", "stanchezza", "anxiety",
    "ansia", "anxious", "rash", "eruzione", "pruritus", "itching", "prurito",
    "palpitations", "palpitazioni", "brain fog", "nebbia mentale",
    "gynecomastia", "ginecomastia", "edema", "oedema", "gonfiore",
    "weight gain", "weight loss", "aumento di peso", "adverse",
}


def _v3_event_terms(event_side: set) -> set:
    """Match terms for the event side, with group semantics extended to QU
    phrasings the production exact-membership check would miss.

    Production ``_event_match_terms`` admits a synonym group only when a query
    term IS a group member. QU terms like "initial shedding" or
    "androgenetic alopecia" are not members, so V3 additionally triggers a
    group when a term CONTAINS a member ("androgenetic alopecia" ⊃ "alopecia")
    or a term's head word IS CONTAINED in a member ("shedding" ⊂ "hair
    shedding"). Split head words are used ONLY as trigger probes, never as
    match terms themselves (a word like "hair" would match everything).
    """
    terms = {t for t in event_side if t}
    # Head words are trigger PROBES only: a probe triggers a group when it is
    # a PROPER substring of a member ("shedding" ⊂ "hair shedding"). A probe
    # equal to a member ("alopecia" from "androgenetic alopecia") must NOT
    # widen the term to the whole generic group — the term is more specific.
    probes = set()
    for t in terms:
        probes.update(w for w in t.split() if len(w) > 3)
    out = set(terms)
    for members, match_tokens in _EVENT_SYNONYM_GROUPS:
        exact = any(t == m for t in terms for m in members)
        headword = any(p != m and p in m for p in probes for m in members)
        if exact or headword:
            out.update(m.lower() for m in members)
            out.update(t.lower() for t in match_tokens)
    return out


def _v3_event_score(body: str, condition_field: str, event_side: set,
                    is_authoritative_match: bool,
                    tiers: Optional[Dict[str, float]] = None,
                    ) -> Tuple[float, List[str]]:
    """Event-side score for V3, including the generic-any-event case.

    ``tiers`` (optional, from external vocabulary evidence) weights each
    matched term: terms absent from the map weight 1.0, so without
    vocabulary evidence the score is byte-identical to the pre-vocabulary
    behaviour."""
    def _score(hits: List[str]) -> float:
        if tiers:
            w = sum(tiers.get(h, 1.0) for h in hits)
        else:
            w = float(len(hits))
        return min(1.0, 0.5 + 0.25 * w)

    if not event_side:
        return 0.5, []
    if event_side <= _GENERIC_EVENT_TERMS:
        # "Any adverse event" query: authoritative reports ARE adverse-event
        # reports by definition; other sources must mention some known event
        # vocabulary (union of all semantic groups + the generic terms).
        if is_authoritative_match:
            return 1.0, ["(any adverse event — authoritative report)"]
        vocab = set(_GENERIC_EVENT_TERMS) | _GENERIC_EVENT_VOCAB
        for members, match_tokens in _EVENT_SYNONYM_GROUPS:
            vocab.update(m.lower() for m in members)
            vocab.update(t.lower() for t in match_tokens)
        hits = sorted({t for t in vocab if t and t in body})[:6]
        return (_score(hits), hits) if hits else (0.0, [])
    terms = _v3_event_terms(event_side)
    hits = sorted({t for t in terms if t and t in body})[:6]
    if not hits and condition_field:
        hits = sorted({t for t in terms if t and t in condition_field})[:6]
    if not hits:
        return 0.0, []
    return _score(hits), hits


def _score_item_v3(
    it: RWEItem,
    query: str,
    sides: Dict[str, set],
    is_authoritative_match: bool,
) -> Tuple[float, str]:
    """
    QU-aware relevance score (V3). Same contract as ``_score_item`` (V1):
    returns (0–1 score, human match reason) and never mutates the item.

    Differences vs V1:
    - the query sides come from the structured QU intent (interventions /
      outcomes / conditions + QU synonyms), UNIONED with the KB entities —
      the event side exists even when the KB does not recognise the concept
      (the "shedding" gap);
    - anchor side = interventions ∪ conditions; event side = outcomes ∪
      conditions (a condition anchors and also expresses the topic);
    - SYMMETRIC strictness: for a two-sided query, a record matching only
      one side scores low (V1 penalised only the missing-event case);
    - openFDA keeps the V1 rule: drug trusted server-side, event verified.
    """
    q_tokens = _tokens(query)
    body = f"{it.title} {it.text}".lower()
    treatment_lower = (it.treatment or "").lower()
    condition_lower = (it.condition or "").lower()
    body_tokens = _tokens(
        f"{it.title} {it.text} {treatment_lower} {condition_lower}"
    )
    overlap = q_tokens & body_tokens
    token_score = len(overlap) / max(1, len(q_tokens)) if q_tokens else 0.0

    iv, oc, cd = sides["iv"], sides["oc"], sides["cd"]
    tiers = sides.get("tiers") or {}

    # ── side resolution (avoids the degenerate cd-anchor case) ──────────────
    # Two-sided queries: the anchor must be the DRUG when one is present
    # (a condition mention alone cannot satisfy both sides of "finasteride
    # hair loss"); the event side is outcomes ∪ conditions. Queries without
    # an intervention are anchored by the condition itself.
    if iv:
        anchor_terms = iv
        event_side = oc | cd
    else:
        anchor_terms = cd
        event_side = oc if oc else cd
    anchor_side = anchor_terms

    # ── anchor side (intervention or condition) ──
    if is_authoritative_match and iv:
        anchor_score, anchor_hits = 1.0, ["(authoritative)"]
    else:
        hay = f"{body} {treatment_lower}"
        anchor_hits = sorted({t for t in anchor_terms if t and t in hay})
        # Vocabulary tiers weight each hit; absent terms weight 1.0 (unchanged
        # behaviour when no vocabulary evidence is present).
        anchor_w = sum(tiers.get(h, 1.0) for h in anchor_hits)
        anchor_score = min(1.0, 0.6 + 0.2 * anchor_w) if anchor_hits else 0.0
        # A condition anchor also matches semantically (multilingual groups),
        # e.g. an Italian post about "caduta capelli" anchors "hair loss".
        if cd and not iv:
            sem_score, sem_hits = _v3_event_score(body, condition_lower, cd,
                                                  False, tiers)
            if sem_score > anchor_score:
                anchor_score, anchor_hits = sem_score, sem_hits

    # ── event side (QU outcomes/conditions, extended semantic groups) ──
    event_score, event_hits = _v3_event_score(
        body, condition_lower, event_side, is_authoritative_match, tiers)

    # ── structured-field boost (small, bounded) ──
    boost = 0.0
    if treatment_lower and any(t in treatment_lower for t in anchor_side):
        boost += 0.05
    if condition_lower and any(t in condition_lower for t in event_side):
        boost += 0.05

    # ── combine ──
    if anchor_side and event_side:
        if event_score == 0.0:
            base = 0.15 * anchor_score + 0.05 * token_score
            reason = (f"v3 anchor_match_only (anchor={anchor_hits[:2]}; "
                      f"event_missing)")
        elif anchor_score == 0.0:
            base = 0.15 * event_score + 0.05 * token_score
            reason = (f"v3 event_match_only (event={event_hits[:3]}; "
                      f"anchor_missing)")
        else:
            base = (0.40 * anchor_score + 0.40 * event_score
                    + 0.10 * token_score + boost)
            reason = (f"v3 anchor+event (anchor={anchor_hits[:2]}; "
                      f"event={event_hits[:3]}; tokens={len(overlap)})")
    elif anchor_side:
        base = 0.55 * anchor_score + 0.25 * token_score + boost
        reason = f"v3 anchor_only (anchor={anchor_hits[:2]}; tokens={len(overlap)})"
    elif event_side:
        base = 0.55 * event_score + 0.30 * token_score + boost
        reason = f"v3 event_only (event={event_hits[:3]})"
    else:
        base = 0.8 * token_score
        reason = "v3 token_fallback"
    return round(min(1.0, base), 3), reason


def relevance_filter(
    items: List[RWEItem],
    query: str,
    entities: Optional[list] = None,
    authoritative_sources: Optional[set] = None,
    min_score: float = 0.20,
    intent=None,
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

    QU-aware mode (V3): when ``intent`` is a valid ``RWEQueryIntent`` (built by
    the query engine only under the HLEO_RWE_INTENT_SCORING feature flag), the
    scoring uses the structured intent sides via ``_score_item_v3``. When
    ``intent`` is None the behaviour is EXACTLY the V1 behaviour above.
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

    # V3 side sets are precomputed once per call (deterministic, no LLM here).
    v3_sides = merged_sides(intent, entities) if intent is not None else None

    kept: List[RWEItem] = []
    for it in items:
        # Authoritative = the source's own search is trusted (hardcoded set),
        # optionally restricted by the caller via authoritative_sources.
        is_auth = it.source in _AUTHORITATIVE_SOURCES and (
            not auth_sources or it.source in auth_sources
        )
        if v3_sides is not None:
            score, reason = _score_item_v3(it, query, v3_sides, is_auth)
        else:
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
    """Union duplicate records while retaining every query provenance."""
    best: dict = {}
    order: List = []
    for it in items:
        key = (it.source, it.external_id or it.source_url or it.title)
        prov = {
            "matched_query": it.matched_query,
            "matched_query_type": it.matched_query_type,
            "source_language": it.source_language,
            "original_term": (it.metadata or {}).get("original_term"),
            "expanded_term": (it.metadata or {}).get("expanded_term"),
            "match_kind": (it.metadata or {}).get("match_kind"),
            "tier": (it.metadata or {}).get("tier"),
            "provider": (it.metadata or {}).get("provider"),
            "source_entity": (it.metadata or {}).get("source_entity"),
            "query_origin": (it.metadata or {}).get("query_origin"),
        }
        if key not in best:
            best[key] = it
            order.append(key)
            it.metadata = dict(it.metadata or {})
            it.metadata["matched_queries"] = [it.matched_query] if it.matched_query else []
            it.metadata["match_provenance"] = [prov]
            continue
        winner = best[key]
        winner.metadata = dict(winner.metadata or {})
        queries = winner.metadata.setdefault("matched_queries", [])
        if it.matched_query and it.matched_query not in queries:
            queries.append(it.matched_query)
        provenance = winner.metadata.setdefault("match_provenance", [])
        if prov not in provenance:
            provenance.append(prov)
        if (it.relevance_score or 0.0) > (winner.relevance_score or 0.0):
            it.metadata = winner.metadata
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

        # Resolve effective sources: if caller provided an explicit list, prefer it;
        # otherwise read active RWE sources from SourceRegistry. Fall back to legacy
        # default list if no registry rows are present.
        from core.database import SessionLocal
        from core.models import SourceRegistry
        from sqlalchemy import select

        registry_map: Dict[str, SourceRegistry] = {}
        effective_sources: List[str] = []

        if sources:
            # normalize provided list and try to resolve registry rows for them
            requested = [s.strip() for s in sources if s and s.strip()]
            if requested:
                with SessionLocal() as db:
                    rows = db.execute(select(SourceRegistry).where(SourceRegistry.source_id.in_(requested))).scalars().all()
                rows_by_id = {r.source_id: r for r in rows}
                for s in requested:
                    if s in rows_by_id:
                        registry_map[s] = rows_by_id[s]
                        effective_sources.append(s)
                    else:
                        # allow direct collector keys (legacy) to be passed through
                        effective_sources.append(s)
        else:
            # discover active RWE sources from the registry
            with SessionLocal() as db:
                rows = db.execute(
                    select(SourceRegistry).where(
                        SourceRegistry.category == "rwe_experience",
                        SourceRegistry.status == "active",
                    )
                ).scalars().all()
            if rows:
                for r in rows:
                    registry_map[r.source_id] = r
                    effective_sources.append(r.source_id)
            else:
                effective_sources = [
                    "reddit", "openfda_faers", "calvizie",
                    "hairlosstalk", "hairlossexperiences", "maladiesrares",
                ]

        per_source_limits = {s: limit for s in effective_sources}
        # ensure default caps for legacy keys if present
        per_source_limits.update({
            "reddit": limit,
            "openfda_faers": limit,
            "calvizie": limit,
            "hairlosstalk": limit,
            "hairlossexperiences": limit,
            "maladiesrares": limit,
        })

        all_items: List[RWEItem] = []
        source_status: dict = {}

        # Collector instance map for known collectors
        collector_map = {
            "reddit": self.reddit,
            "openfda_faers": self.openfda,
            "calvizie": self.calvizie,
            "hairlosstalk": self.hairlosstalk,
            "hairlossexperiences": self.hairlossexperiences,
            "maladiesrares": self.maladiesrares,
        }

        for src in effective_sources:
            row = registry_map.get(src)
            collector_key = (row.runtime_collector if row and row.runtime_collector else src)
            collector = collector_map.get(collector_key)
            # If this is a generic REST-configured source, instantiate GenericRESTCollector
            if collector_key == "generic_rest" and row:
                try:
                    from collectors.generic_rest import GenericRESTCollector
                    collector = GenericRESTCollector(row.connection_spec or {}, source_id=row.source_id, category=row.category)
                except Exception as exc:
                    logger.exception("Failed to instantiate GenericRESTCollector for %s: %s", src, exc)
                    source_status[src] = "no_collector"
                    continue

            if not collector:
                source_status[src] = "no_collector"
                continue

            try:
                items, status = self._collect_source(src, collector, plan, per_source_limits.get(src, limit))
                source_status[src] = status
                all_items.extend(items)
            except Exception as exc:
                logger.exception("RWE collector failed for %s: %s", src, exc)
                source_status[src] = "network_error"
                # continue with other sources
                continue

        # ── 2. Deduplicate across (query × source), keep best matched_query ──
        before = len(all_items)
        all_items = deduplicate(all_items)
        after = len(all_items)

        # ── 3. Semi-semantic relevance filtering ────────────────────────────
        # intent is None unless the V3 feature flag built it → V1 behaviour.
        relevance_query = plan.translated_query or plan.original_query
        all_items = relevance_filter(
            all_items,
            relevance_query,
            entities=plan.entities,
            intent=getattr(plan, "intent", None),
            min_score=0.20,
        )
        all_items.sort(key=lambda it: it.relevance_score, reverse=True)
        relevant_before_cap = len(all_items)
        all_items = all_items[:400]

        totals = {
            "retrieved": before,
            "deduped_removed": before - after,
            "unique": after,
            "relevant": relevant_before_cap,
            "final": len(all_items),
            "score_threshold": 0.20,
            "max_results": 400,
            "queries_used": len(plan.expanded_queries),
        }

        return RWESearchResult(
            query=query,
            original_query=plan.original_query,
            search_query=relevance_query,
            canonical_query=plan.canonical_query,
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

        # Forum-feed collectors (XenForo RSS / phpBB Atom) are not full-text
        # searchable: every query re-fetches the same feed. Send them only the
        # primary queries (original/translated/canonical); full-text APIs
        # (openFDA, Reddit) receive the complete expansion set.
        from core.rwe.xenforo_base import XenForoRSSCollector
        from core.rwe.maladiesrares_collector import MaladiesRaresCollector
        feed_like = isinstance(collector, (XenForoRSSCollector, MaladiesRaresCollector))
        if feed_like:
            primary_types = {"original", "translated", "canonical"}
            queries = [eq for eq in plan.expanded_queries
                       if eq.expansion_type in primary_types] or plan.expanded_queries[:1]
        else:
            # Server-side search APIs (openFDA, Reddit) execute one request
            # per query; bound the number of distinct queries sent to keep
            # latency and rate limits healthy while still using the
            # vocabulary expansions. Original/translated/canonical rank first
            # in the plan ordering.
            queries = plan.expanded_queries[:6]

        for eq in queries:
            try:
                items, status, reason = collector.search_with_status(
                    eq.query, limit=None
                )
            except Exception as exc:
                logger.warning(f"RWE collector {name} failed for '{eq.query}': {exc}")
                statuses.append("network_error")
                continue
            statuses.append(status)
            if status != "ok":
                continue
            for it in items:
                it.matched_query = eq.query
                it.matched_query_type = eq.expansion_type
                it.source_language = eq.source_language
                it.topic = eq.query
                it.metadata = dict(it.metadata or {})
                it.metadata.update({
                    "original_term": eq.original_term,
                    "expanded_term": eq.expanded_term,
                    "match_kind": eq.match_kind,
                    "tier": eq.tier,
                    "provider": eq.provider,
                    "source_entity": eq.source_entity,
                    "query_origin": eq.query_origin,
                })
                collected.append(it)

        # Aggregate: prefer ok > no_results > no_credentials > error
        status_rank = {
            "ok": 0, "no_results": 1, "no_credentials": 2,
            "auth_error": 3, "rate_limited": 4, "network_error": 5,
        }
        agg = "no_results"
        if statuses:
            agg = min(statuses, key=lambda s: status_rank.get(s, 9))
        return collected, agg
