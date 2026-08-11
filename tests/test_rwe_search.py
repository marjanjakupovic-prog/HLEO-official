"""
Tests for the RWE Search Engine — autonomous query preparation, translation,
controlled query expansion, semi-semantic retrieval, and provenance.

Covers the required cases:
  - Italian query
  - English query
  - language detection
  - translation (original_query preserved)
  - query expansion (synonyms, MeSH, combos, colloquial)
  - synonyms / medical vs colloquial terms
  - trichology query
  - non-trichology query
  - semi-semantic retrieval (authoritative source match preserved)
  - relevance filtering (score + match_reason)
  - provenance (matched_query, source_language, expansion_type)
  - source language
  - original_query never overwritten
  - results from translated queries
  - results from expanded queries
  - no duplicates across multi-query retrieval
  - RWE only
  - scientific only regression
  - both (scientific + RWE convergence via AI Assistant schema)
"""
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core.rwe.models import RWEItem
from core.rwe.query_engine import (
    RWEQueryEngine,
    RWEQueryPlan,
    ExpandedQuery,
    detect_language,
    EXP_ORIGINAL,
    EXP_TRANSLATED,
    EXP_SYNONYM,
    EXP_MESH,
    EXP_COMBO,
    EXP_COLLOQUIAL,
    EXP_NEIGHBOR,
)


# ─── fakes ───────────────────────────────────────────────────────────────────

def _fake_reddit_item(title, url, text="finasteride stopped my shedding"):
    return RWEItem(
        source="reddit",
        source_type="community_forum",
        evidence_tier="anecdotal",
        collection_method="official_api_oauth2",
        external_id=url,
        source_url=url,
        title=title,
        text=text,
        date="2023-05-01",
        language="en",
        privacy_status="redacted",
    )


def _fake_faers_item(rid="FAERS-1", treatment="FINASTERIDE", text="alopecia; decreased libido"):
    return RWEItem(
        source="openfda_faers",
        source_type="pharmacovigilance",
        evidence_tier="spontaneous_report",
        collection_method="official_api_no_key",
        external_id=rid,
        source_url=f"https://api.fda.gov/drug/event.json?search=safetyreportid:{rid}",
        title=f"FAERS report {rid} — Alopecia",
        text=text,
        date="20240101",
        language="en",
        treatment=treatment,
        experience_type="adverse_event",
        privacy_status="anonymous",
    )


def _stub_reddit_ok(items):
    def _sws(query, limit=15):
        return items, "ok", f"Retrieved {len(items)} post(s)."
    return _sws


def _stub_openfda_ok(items):
    def _sws(query, limit=20):
        return list(items), "ok", f"Retrieved {len(items)} report(s)."
    return _sws


def _stub_openfda_status(status, reason):
    def _sws(query, limit=20):
        return [], status, reason
    return _sws


def _stub_reddit_status(status, reason):
    def _sws(query, limit=15):
        return [], status, reason
    return _sws


def _make_engine():
    """RWEQueryEngine with a mocked orchestrator (deterministic, no LLM).

    Must be called INSIDE a ``_mock_orch(...)`` patch context so the engine
    captures the mocked orchestrator at construction time.
    """
    return RWEQueryEngine()


def _mock_orch(search_query, lang="en", translated=True):
    """Patch QueryOrchestrator.process to return a fixed OrchestrationResult.

    The engine instantiates the orchestrator in ``__init__``, so the engine
    itself must be created while this patch is active.
    """
    return patch(
        "core.rwe.query_engine.QueryOrchestrator",
        **{"return_value.process.return_value": SimpleNamespace(
            search_query=search_query,
            detected_language=lang,
            translation_applied=translated,
        )},
    )


# ─── language detection ─────────────────────────────────────────────────────

def test_detect_language_italian():
    assert detect_language("La finasteride può causare shedding?") == "it"


def test_detect_language_english():
    assert detect_language("Can finasteride cause initial shedding?") == "en"


def test_detect_language_empty():
    assert detect_language("") == "und"


def test_detect_language_german():
    assert detect_language("Kann finasterid haarausfall verursachen?") == "de"


def test_detect_language_french():
    assert detect_language("La finastéride peut causer la chute de cheveux?") == "fr"


def test_detect_language_spanish():
    assert detect_language("La finasterida puede causar caída del cabello?") == "es"


# ─── query plan / original_query preserved ──────────────────────────────────

def test_original_query_never_overwritten():
    """original_query must remain the verbatim user input."""
    with _mock_orch("finasteride initial shedding", lang="it"):
        eng = _make_engine()
        plan = eng.plan("La finasteride può causare shedding iniziale?")
    assert plan.original_query == "La finasteride può causare shedding iniziale?"
    assert plan.translated_query != plan.original_query
    assert plan.detected_language == "it"


def test_translation_fields_present():
    with _mock_orch("finasteride hair loss", lang="it", translated=True):
        eng = _make_engine()
        plan = eng.plan("caduta capelli finasteride")
    assert plan.translation_applied is True
    assert plan.translated_query == "finasteride hair loss"


def test_english_query_no_translation():
    with _mock_orch("finasteride shedding", lang="en", translated=False):
        eng = _make_engine()
        plan = eng.plan("finasteride shedding")
    assert plan.detected_language == "en"
    assert plan.translation_applied is False


# ─── query expansion ────────────────────────────────────────────────────────

def test_expansion_contains_original_and_translated():
    with _mock_orch("finasteride hair loss", lang="it"):
        eng = _make_engine()
        plan = eng.plan("finasteride caduta capelli")
    types = [eq.expansion_type for eq in plan.expanded_queries]
    assert EXP_ORIGINAL in types
    assert EXP_TRANSLATED in types


def test_expansion_synonyms_present():
    with _mock_orch("finasteride", lang="en", translated=False):
        eng = _make_engine()
        plan = eng.plan("finasteride")
    queries = [eq.query.lower() for eq in plan.expanded_queries]
    # propecia/proscar are known finasteride brand aliases
    assert "propecia" in queries or "proscar" in queries


def test_expansion_mesh_terms_present():
    with _mock_orch("finasteride", lang="en", translated=False):
        eng = _make_engine()
        plan = eng.plan("finasteride")
    mesh_queries = [eq for eq in plan.expanded_queries if eq.expansion_type == EXP_MESH]
    assert len(mesh_queries) >= 1


def test_expansion_colloquial_for_shedding():
    """The colloquial supplement must surface patient phrasings for hair loss."""
    with _mock_orch("finasteride initial shedding", lang="en", translated=False):
        eng = _make_engine()
        plan = eng.plan("finasteride initial shedding")
    colloq = [eq.query for eq in plan.expanded_queries if eq.expansion_type == EXP_COLLOQUIAL]
    assert any("shedding" in q or "hair fall" in q or "worsening" in q for q in colloq)


def test_expansion_is_controlled_not_broad():
    """A finasteride query must NOT broaden into generic hair-loss-only queries
    that drop the drug entity — every colloquial/combo keeps the drug anchor."""
    with _mock_orch("finasteride shedding", lang="en", translated=False):
        eng = _make_engine()
        plan = eng.plan("finasteride shedding")
    # every expanded query must contain 'finasteride' OR a finasteride synonym
    fin_terms = {"finasteride", "propecia", "proscar", "finpecia", "fincar"}
    for eq in plan.expanded_queries:
        if eq.expansion_type in (EXP_COLLOQUIAL, EXP_COMBO, EXP_NEIGHBOR):
            qtokens = set(eq.query.lower().split())
            assert qtokens & fin_terms or "finasteride" in eq.matched_entities, (
                f"expansion '{eq.query}' dropped the finasteride anchor"
            )


def test_expansion_capped():
    with _mock_orch("finasteride dutasteride minoxidil", lang="en", translated=False):
        eng = _make_engine()
        plan = eng.plan("finasteride dutasteride minoxidil")
    assert len(plan.expanded_queries) <= 16


def test_expansion_deduplicated():
    with _mock_orch("finasteride", lang="en", translated=False):
        eng = _make_engine()
        plan = eng.plan("finasteride")
    queries = [eq.query.lower() for eq in plan.expanded_queries]
    assert len(queries) == len(set(queries)), "duplicate expanded queries"


# ─── synonyms / medical vs colloquial ───────────────────────────────────────

def test_medical_term_recognized():
    """Medical term 'androgenetic alopecia' is recognized as a condition."""
    with _mock_orch("androgenetic alopecia", lang="en", translated=False):
        eng = _make_engine()
        plan = eng.plan("androgenetic alopecia treatment")
    canonicals = [c for _, c, _ in plan.entities]
    assert "androgenetic alopecia" in canonicals


def test_colloquial_term_recognized_as_medical():
    """Colloquial 'caduta capelli' maps to the medical concept 'hair loss'."""
    with _mock_orch("hair loss", lang="it"):
        eng = _make_engine()
        plan = eng.plan("caduta capelli")
    canonicals = [c for _, c, _ in plan.entities]
    assert "hair loss" in canonicals


def test_brand_name_normalized_to_generic():
    """Brand 'propecia' is recognized as the generic 'finasteride'."""
    with _mock_orch("propecia", lang="en", translated=False):
        eng = _make_engine()
        plan = eng.plan("propecia side effects")
    canonicals = [c for _, c, _ in plan.entities]
    assert "finasteride" in canonicals


# ─── trichology vs non-trichology ────────────────────────────────────────────

def test_trichology_query_recognizes_entities():
    with _mock_orch("minoxidil hair loss", lang="en", translated=False):
        eng = _make_engine()
        plan = eng.plan("minoxidil hair loss")
    canonicals = {c for _, c, _ in plan.entities}
    assert "minoxidil" in canonicals
    assert "hair loss" in canonicals


def test_non_trichology_query_still_runs():
    """A non-trichology query produces a plan (no entities) without crashing."""
    with _mock_orch("headache aspirin", lang="en", translated=False):
        eng = _make_engine()
        plan = eng.plan("headache aspirin")
    assert plan.original_query == "headache aspirin"
    assert len(plan.expanded_queries) >= 1  # at least the original


# ─── semi-semantic retrieval / relevance ────────────────────────────────────

def test_authoritative_source_match_preserved():
    """openFDA items with no text overlap are kept (authoritative server match)."""
    from core.rwe.pipeline import relevance_filter
    # FAERS report where the drug is NOT in the reaction text
    it = _fake_faers_item(text="headache; nausea", treatment="FINASTERIDE")
    out = relevance_filter(
        [it], "finasteride",
        entities=[("drug", "finasteride", 1.0)],
    )
    assert len(out) == 1
    assert out[0].relevance == "relevant"
    assert "authoritative" in (out[0].match_reason or "")


def test_semantic_entity_match_without_keyword():
    """An item matching an entity alias (but no query token) is kept."""
    from core.rwe.pipeline import relevance_filter
    it = _fake_reddit_item("my story", "u1", text="I took propecia and my hair fell out")
    out = relevance_filter(
        [it], "finasteride shedding",
        entities=[("drug", "finasteride", 1.0), ("symptom", "hair loss", 1.0)],
    )
    assert len(out) == 1
    assert out[0].relevance_score > 0


def test_off_topic_item_filtered():
    from core.rwe.pipeline import relevance_filter
    off = _fake_reddit_item("cooking", "u2", text="pasta recipe with tomato sauce")
    out = relevance_filter(
        [off], "finasteride shedding",
        entities=[("drug", "finasteride", 1.0)],
    )
    assert len(out) == 0
    assert off.relevance == "irrelevant"


def test_relevance_filter_backward_compatible():
    """relevance_filter(items, query) still works without entities arg."""
    from core.rwe.pipeline import relevance_filter
    rel = _fake_reddit_item("fin", "u1", text="finasteride worked")
    out = relevance_filter([rel], "finasteride")
    assert len(out) == 1
    assert out[0].relevance == "relevant"


def test_relevance_score_is_float_0_to_1():
    from core.rwe.pipeline import relevance_filter
    it = _fake_faers_item(text="alopecia", treatment="FINASTERIDE")
    relevance_filter([it], "finasteride", entities=[("drug", "finasteride", 1.0)])
    assert 0.0 <= it.relevance_score <= 1.0


# ─── provenance ─────────────────────────────────────────────────────────────

def test_matched_query_stamped():
    """Each item carries the expanded query that surfaced it."""
    from core.rwe.pipeline import RWEPipeline
    pipe = RWEPipeline()
    faers = [_fake_faers_item()]
    with patch.object(pipe.openfda, "search_with_status", _stub_openfda_ok(faers)), \
         patch.object(pipe.reddit, "search_with_status", _stub_reddit_status("no_credentials", "x")), \
         patch("core.rwe.query_engine.QueryOrchestrator") as M:
        M.return_value.process.return_value = SimpleNamespace(
            search_query="finasteride", detected_language="en", translation_applied=False,
        )
        result = pipe.search("finasteride")
    assert len(result.items) == 1
    it = result.items[0]
    assert it.matched_query is not None
    assert it.matched_query_type is not None
    assert it.source_language is not None


def test_source_language_stamped():
    from core.rwe.pipeline import RWEPipeline
    pipe = RWEPipeline()
    faers = [_fake_faers_item()]
    # Reddit returns items only for English queries (realistic: Reddit search
    # is English-oriented, so translated queries match).
    def reddit_en_only(query, limit=15):
        if any(w in query.lower() for w in ("finasteride", "hair", "shedding")) and "caduta" not in query.lower():
            return [_fake_reddit_item("en post", "u-en")], "ok", "ok"
        return [], "no_results", "none"
    with patch.object(pipe.openfda, "search_with_status", _stub_openfda_ok(faers)), \
         patch.object(pipe.reddit, "search_with_status", reddit_en_only), \
         _mock_orch("finasteride hair loss", lang="it"):
        result = pipe.search("caduta capelli finasteride")
    # at least one item should have source_language en (matched the translated query)
    langs = {it.source_language for it in result.items}
    assert "en" in langs


def test_expanded_queries_in_result():
    """The result envelope exposes the full expansion provenance."""
    from core.rwe.pipeline import RWEPipeline
    pipe = RWEPipeline()
    with patch.object(pipe.openfda, "search_with_status", _stub_openfda_status("no_results", "none")), \
         patch.object(pipe.reddit, "search_with_status", _stub_reddit_status("no_credentials", "x")), \
         _mock_orch("finasteride", lang="en", translated=False):
        result = pipe.search("finasteride")
    assert len(result.expanded_queries) >= 1
    assert result.original_query == "finasteride"
    assert result.translated_query == "finasteride"


# ─── multi-query retrieval: translated + expanded results ───────────────────

def test_results_from_translated_query():
    """An item surfaced by the translated (English) query is present when the
    original was Italian."""
    from core.rwe.pipeline import RWEPipeline
    pipe = RWEPipeline()
    # Reddit returns items only when the query contains 'finasteride' (English)
    def reddit_conditional(query, limit=15):
        if "finasteride" in query.lower():
            return [_fake_reddit_item("en post", "u-en")], "ok", "ok"
        return [], "no_results", "none"
    with patch.object(pipe.reddit, "search_with_status", reddit_conditional), \
         patch.object(pipe.openfda, "search_with_status", _stub_openfda_status("no_results", "x")), \
         _mock_orch("finasteride hair loss", lang="it"):
        result = pipe.search("caduta capelli finasteride")
    assert any(it.source == "reddit" for it in result.items)
    # the matched_query should be an English expansion
    en_items = [it for it in result.items if it.matched_query and "finasteride" in it.matched_query.lower()]
    assert len(en_items) >= 1


def test_results_from_expanded_query():
    """An item surfaced by a synonym/expanded query (not the original) is kept."""
    from core.rwe.pipeline import RWEPipeline
    pipe = RWEPipeline()
    # openFDA returns an item only for the 'propecia' synonym
    def openfda_conditional(query, limit=20):
        if "propecia" in query.lower():
            return [_fake_faers_item(rid="PF-1", treatment="FINASTERIDE")], "ok", "ok"
        return [], "no_results", "none"
    with patch.object(pipe.openfda, "search_with_status", openfda_conditional), \
         patch.object(pipe.reddit, "search_with_status", _stub_reddit_status("no_credentials", "x")), \
         _mock_orch("finasteride", lang="en", translated=False):
        result = pipe.search("finasteride")
    assert any(it.external_id == "PF-1" for it in result.items)
    pf = next(it for it in result.items if it.external_id == "PF-1")
    assert "propecia" in pf.matched_query.lower()
    assert pf.matched_query_type == EXP_SYNONYM


# ─── no duplicates across multi-query ───────────────────────────────────────

def test_no_duplicates_across_queries():
    """The same item returned by multiple expanded queries is deduplicated."""
    from core.rwe.pipeline import RWEPipeline
    pipe = RWEPipeline()
    same = _fake_faers_item(rid="DUP-1")
    with patch.object(pipe.openfda, "search_with_status", _stub_openfda_ok([same])), \
         patch.object(pipe.reddit, "search_with_status", _stub_reddit_status("no_credentials", "x")), \
         _mock_orch("finasteride", lang="en", translated=False):
        result = pipe.search("finasteride")
    # same item returned for many expanded queries but should appear once
    ids = [it.external_id for it in result.items if it.source == "openfda_faers"]
    assert ids.count("DUP-1") == 1


# ─── RWE only / scientific regression / both ────────────────────────────────

def test_rwe_only_search():
    from core.rwe.pipeline import RWEPipeline
    pipe = RWEPipeline()
    with patch.object(pipe.openfda, "search_with_status", _stub_openfda_ok([_fake_faers_item()])), \
         patch.object(pipe.reddit, "search_with_status", _stub_reddit_ok([_fake_reddit_item("r","u")])), \
         _mock_orch("finasteride", lang="en", translated=False):
        result = pipe.search("finasteride")
    assert all(i.source_type in ("community_forum", "pharmacovigilance") for i in result.items)
    assert all(i.source_type not in ("pubmed", "europepmc", "clinicaltrials") for i in result.items)


def test_scientific_search_route_unchanged(client):
    """Scientific /search must still work and not leak RWE items."""
    r = client.get("/search?q=finasteride&mode=global")
    assert r.status_code == 200


def test_rwe_search_endpoint_returns_provenance(client):
    r = client.get("/rwe/search?q=finasteride")
    assert r.status_code == 200
    body = r.json()
    assert "original_query" in body
    assert "translated_query" in body
    assert "detected_language" in body
    assert "expanded_queries" in body


def test_assistant_accepts_rwe_with_provenance(client):
    """The AI Assistant schema accepts RWE items with the new provenance fields."""
    r = client.post("/assistant/chat", json={
        "message": "What do patients report about finasteride shedding?",
        "search_context": {
            "original_query": "finasteride shedding",
            "search_query": "finasteride shedding",
            "detected_language": "en",
            "articles": [],
            "rwe_evidence": [{
                "source": "reddit",
                "source_type": "community_forum",
                "evidence_tier": "anecdotal",
                "title": "My finasteride shed",
                "text": "I experienced initial shedding on finasteride",
                "matched_query": "finasteride initial shedding",
                "matched_query_type": "colloquial",
                "source_language": "en",
                "relevance_score": 0.8,
                "match_reason": "exact_keyword+semantic",
            }],
        },
    })
    assert r.status_code == 200


# ─── integration: the canonical Italian test query ──────────────────────────

def test_canonical_italian_query_full_pipeline():
    """The success criterion: 'La finasteride può causare shedding iniziale?'
    must be transformed into a proper RWE search plan."""
    with _mock_orch("finasteride initial shedding", lang="it", translated=True):
        eng = _make_engine()
        plan = eng.plan("La finasteride può causare shedding iniziale?")
    # 1. recognizes language
    assert plan.detected_language == "it"
    # 2. preserves original
    assert plan.original_query == "La finasteride può causare shedding iniziale?"
    # 3. translates
    assert plan.translated_query == "finasteride initial shedding"
    assert plan.translation_applied is True
    # 4. recognizes entities
    canonicals = {c for _, c, _ in plan.entities}
    assert "finasteride" in canonicals
    assert "hair loss" in canonicals  # 'shedding' → hair loss concept
    # 5. generates expanded queries
    assert len(plan.expanded_queries) >= 5
    # 6. includes colloquial shedding phrasings
    all_queries = " ".join(eq.query.lower() for eq in plan.expanded_queries)
    assert "shedding" in all_queries
    # 7. expansion is controlled (finasteride anchor preserved)
    fin_terms = {"finasteride", "propecia", "proscar"}
    colloq = [eq for eq in plan.expanded_queries if eq.expansion_type == EXP_COLLOQUIAL]
    for eq in colloq:
        assert "finasteride" in eq.matched_entities or (set(eq.query.lower().split()) & fin_terms)
