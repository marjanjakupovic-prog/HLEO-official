"""
hleo_v1/core/semantic_search.py
================================
HLEO v2 — Semantic Biomedical Search Engine (Phases 1–12)

This module is completely independent.  It does NOT modify any existing class.
It wraps the existing collectors and reuses existing dedup/ranking helpers.

Public API
----------
    engine = SemanticSearch()
    result = engine.search("caduta indotta dutasteride")
    # result.pubmed / .europepmc / .clinicaltrials / .reddit  ← same shape as
    # the existing /search endpoint output
    # result.expansion  ← metadata dict (additive)

Phases implemented here
-----------------------
    1  Intent recognition   (hybrid: KB → fuzzy → LLM only if needed)
    2  Knowledge graph      (biomedical_kb.KNOWLEDGE_GRAPH traversal)
    3  Query expansion      (KB synonyms + MeSH + LLM)
    4  Parallel search      (ThreadPoolExecutor across all queries × all sources)
    5  Merge / dedup        (reuses HLEOAggregator helpers)
    6  Semantic ranking     (clinical_rank + entity overlap bonus)
    7  LLM analysis         (scaffolded; full synthesis is done by ArticleExtractor)
    8  Real World Evidence  (Reddit uses same expanded queries)
    9  Sci vs RWE           (scaffolded; full synthesis is Task #27)
   10  Persistent cache     (PostgreSQL via SemanticSearchCache ORM)
   11  Fallback             (broadens scope when < MIN_ARTICLES found, max 2 rounds)
   12  Structured logging   (JSON line per search)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from core.biomedical_kb import (
    lookup_entity,
    get_neighbors,
    get_mesh_terms,
    quick_translate_it,
    DRUG_ALIASES,
    CONDITION_ALIASES,
    SYMPTOM_ALIASES,
)
from core.search_result import SearchResult

logger = logging.getLogger(__name__)

# ─── Tunables ────────────────────────────────────────────────────────────────
MIN_ARTICLES          = 5       # fallback triggers below this
TARGET_ARTICLES       = 25      # desired pool size
MAX_EXPANDED_QUERIES  = 20      # hard cap per search round
MAX_FALLBACK_ROUNDS   = 2       # maximum retry rounds
CACHE_TTL_MINUTES     = 60      # in-process cache TTL
COLLECTOR_LIMIT       = 12      # articles per (query, collector) call
SEMANTIC_BONUS        = 10.0    # score bonus for entity overlap in title/abstract
MAX_WORKERS           = 12      # ThreadPoolExecutor pool size


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IntentEntity:
    text:       str
    entity_type: str   # drug | condition | symptom | procedure | unknown
    normalized: str
    confidence: float = 1.0


@dataclass
class IntentResult:
    original_query:  str
    detected_language: str
    entities:        list[IntentEntity]
    category:        str        # e.g. "adverse_effect", "treatment", "general"
    used_llm:        bool = False

    def entity_names(self) -> list[str]:
        return [e.normalized for e in self.entities]

    def has_entities(self) -> bool:
        return bool(self.entities)


@dataclass
class SemanticResult:
    """Full output of SemanticSearch.search() — same four lists as the pipeline."""
    pubmed:         list[dict] = field(default_factory=list)
    europepmc:      list[dict] = field(default_factory=list)
    clinicaltrials: list[dict] = field(default_factory=list)
    reddit:         list[dict] = field(default_factory=list)
    expansion: dict = field(default_factory=dict)   # metadata (additive)


# ─────────────────────────────────────────────────────────────────────────────
# In-process TTL cache  (process-lifetime; DB cache is the persistent layer)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _CacheEntry:
    result:     SemanticResult
    expires_at: datetime


_MEM_CACHE: dict[str, _CacheEntry] = {}


def _cache_key(query: str) -> str:
    return hashlib.md5(query.lower().strip().encode()).hexdigest()


def _mem_get(key: str) -> Optional[SemanticResult]:
    entry = _MEM_CACHE.get(key)
    if entry and datetime.now(timezone.utc) < entry.expires_at:
        return entry.result
    _MEM_CACHE.pop(key, None)
    return None


def _mem_set(key: str, result: SemanticResult) -> None:
    _MEM_CACHE[key] = _CacheEntry(
        result=result,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=CACHE_TTL_MINUTES),
    )


# ─────────────────────────────────────────────────────────────────────────────
# SemanticSearch
# ─────────────────────────────────────────────────────────────────────────────

class SemanticSearch:
    """
    Semantic biomedical search engine.

    Usage
    -----
        engine = SemanticSearch()
        result = engine.search("problemi sessuali finasteride")
    """

    def __init__(self) -> None:
        self._client = None
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=api_key)
                logger.debug("SemanticSearch: OpenAI client ready.")
            except Exception as exc:
                logger.warning("SemanticSearch: OpenAI unavailable — %s", exc)

        # Lazy-import collectors so we don't break startup if something missing
        self._collectors: dict[str, Any] = {}
        self._init_collectors()

    # ── Collector init ────────────────────────────────────────────────────────

    def _init_collectors(self) -> None:
        try:
            from collectors.pubmed import PubMedCollector
            from collectors.europepmc import EuropePMCCollector
            from collectors.clinicaltrials import ClinicalTrialsCollector
            from collectors.reddit import RedditCollector
            self._collectors = {
                "pubmed":         PubMedCollector(),
                "europepmc":      EuropePMCCollector(),
                "clinicaltrials": ClinicalTrialsCollector(),
                "reddit":         RedditCollector(),
            }
        except Exception as exc:
            logger.error("SemanticSearch: collector init failed — %s", exc)

    # =========================================================================
    # PUBLIC: search()
    # =========================================================================

    def search(self, raw_query: str) -> SemanticResult:
        """
        Full 12-phase semantic search.  Returns a SemanticResult whose four
        article lists are directly substitutable for pipeline.collect() output.
        """
        t0 = time.perf_counter()
        q  = raw_query.strip()
        if not q:
            return SemanticResult()

        ck = _cache_key(q)

        # ── Phase 10: check in-process cache ──────────────────────────────────
        cached = _mem_get(ck)
        if cached:
            self._log(q, cached.expansion, cache_hit=True, elapsed=0.0)
            cached.expansion["cache_hit"] = True
            return cached

        # ── Phase 10: check DB cache ──────────────────────────────────────────
        db_cached = self._db_cache_get(ck)
        if db_cached:
            _mem_set(ck, db_cached)
            self._log(q, db_cached.expansion, cache_hit=True, elapsed=0.0)
            db_cached.expansion["cache_hit"] = True
            return db_cached

        errors: list[str] = []
        fallback_triggered = False

        try:
            # ── Phase 1: intent recognition ───────────────────────────────────
            intent = self._analyze_intent(q)

            # ── Phase 2+3: knowledge graph + query expansion ──────────────────
            expanded_queries = self._build_expansion(intent)

            # ── Phase 4+8: parallel search (scientific + Reddit) ──────────────
            raw_results = self._parallel_search(expanded_queries)

            # ── Phase 5: merge + dedup ────────────────────────────────────────
            merged = self._merge_deduplicate(raw_results)

            # ── Phase 11: fallback if too few results ─────────────────────────
            sci_count = (
                len(merged.get("pubmed", [])) +
                len(merged.get("europepmc", [])) +
                len(merged.get("clinicaltrials", []))
            )
            if sci_count < MIN_ARTICLES:
                fallback_triggered = True
                merged, extra_queries = self._fallback(intent, expanded_queries, merged)
                expanded_queries = expanded_queries + extra_queries

            # ── Phase 6: ranking with semantic bonus ──────────────────────────
            self._rank_with_semantic_bonus(merged, intent)

            # ── Serialise to plain dicts ──────────────────────────────────────
            result = self._to_result(
                merged, intent, expanded_queries,
                fallback_triggered=fallback_triggered,
            )

        except Exception as exc:
            logger.exception("SemanticSearch error: %s", exc)
            errors.append(str(exc))
            result = SemanticResult(expansion={"error": str(exc), "cache_hit": False})

        elapsed = round(time.perf_counter() - t0, 2)
        result.expansion["elapsed_s"]          = elapsed
        result.expansion["cache_hit"]          = False
        result.expansion["fallback_triggered"] = fallback_triggered
        result.expansion["errors"]             = errors

        # ── Phase 10: write to cache ──────────────────────────────────────────
        if not errors:
            _mem_set(ck, result)
            self._db_cache_set(ck, q, result)

        # ── Phase 12: structured log ──────────────────────────────────────────
        self._log(q, result.expansion, cache_hit=False, elapsed=elapsed)

        return result

    # =========================================================================
    # Phase 1 — Intent recognition  (hybrid: KB → fuzzy → LLM)
    # =========================================================================

    def _analyze_intent(self, query: str) -> IntentResult:
        """
        Hybrid intent recognition.
        1. Quick Italian→English token mapping (no LLM, instant)
        2. KB exact + fuzzy entity lookup  (no LLM)
        3. LLM extraction ONLY if < 1 entity found with conf ≥ 0.8
        """
        # Step 1: quick translation of Italian tokens
        query_prep = quick_translate_it(query)

        # Step 2: KB lookup
        raw_hits = lookup_entity(query_prep)
        # also try the original query in case IT alias handles it
        if not raw_hits or all(c < 0.8 for _, _, c in raw_hits):
            raw_hits2 = lookup_entity(query)
            # Merge — keep best confidence per canonical
            best: dict[str, tuple] = {h[1]: h for h in raw_hits}
            for h in raw_hits2:
                if h[1] not in best or h[2] > best[h[1]][2]:
                    best[h[1]] = h
            raw_hits = list(best.values())

        entities = [
            IntentEntity(
                text=canonical,
                entity_type=etype,
                normalized=canonical,
                confidence=conf,
            )
            for etype, canonical, conf in raw_hits
            if conf >= 0.5
        ]

        used_llm = False
        # Step 3: LLM fallback — only when KB found fewer than 1 confident entity
        high_conf = [e for e in entities if e.confidence >= 0.8]
        if not high_conf and self._client:
            try:
                llm_entities = self._llm_extract_intent(query)
                entities = llm_entities
                used_llm = True
            except Exception as exc:
                logger.warning("LLM intent extraction failed: %s", exc)

        category = self._infer_category(entities)
        lang      = self._detect_language_quick(query)

        return IntentResult(
            original_query=query,
            detected_language=lang,
            entities=entities,
            category=category,
            used_llm=used_llm,
        )

    def _infer_category(self, entities: list[IntentEntity]) -> str:
        types = {e.entity_type for e in entities}
        if "drug" in types and "symptom" in types:
            return "adverse_effect"
        if "drug" in types and "condition" in types:
            return "drug_condition"
        if "drug" in types:
            return "drug_query"
        if "condition" in types:
            return "condition_query"
        if "symptom" in types:
            return "symptom_query"
        return "general"

    def _detect_language_quick(self, text: str) -> str:
        """Fast language detection based on stopwords — avoids LLM call."""
        it_markers = {"di", "del", "dopo", "con", "per", "indotta", "indotto",
                      "perdita", "dolore", "caduta", "problemi", "sessuale",
                      "articolari", "cuore", "ginocchio", "memoria"}
        tokens = set(text.lower().split())
        if tokens & it_markers:
            return "it"
        return "en"

    def _llm_extract_intent(self, query: str) -> list[IntentEntity]:
        """LLM-based entity extraction (used only when KB lookup fails)."""
        prompt = (
            "You are a biomedical NLP system for a clinical research search engine.\n"
            "Extract all biomedical entities from the query below.\n"
            "Return JSON: {\"entities\": [{\"text\": str, \"type\": str, "
            "\"normalized\": str}]}\n"
            "Types: drug, active_ingredient, disease, symptom, adverse_effect, "
            "organ, treatment, procedure, biomarker, medical_concept\n"
            "Normalize drug names to their INN (International Nonproprietary Name).\n"
            "Output ONLY the JSON object, no explanation.\n"
            f"\nQuery: {query}"
        )
        resp = self._client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=300,
        )
        data = json.loads(resp.choices[0].message.content)
        result = []
        for ent in data.get("entities", []):
            result.append(IntentEntity(
                text=ent.get("text", ""),
                entity_type=ent.get("type", "unknown"),
                normalized=ent.get("normalized", ent.get("text", "")),
                confidence=0.85,  # LLM extraction assumed high but not perfect
            ))
        return result

    # =========================================================================
    # Phase 2+3 — Knowledge graph traversal + query expansion
    # =========================================================================

    def _build_expansion(self, intent: IntentResult) -> list[str]:
        """
        Generate all useful search queries:
          A) KB synonyms for each entity
          B) MeSH terms for each entity
          C) Knowledge graph neighbors (1-hop)
          D) LLM expansion (scientific variants, if client available)

        Returns a deduplicated list ordered by quality (most specific first).
        """
        queries: list[str] = []

        entity_names = intent.entity_names()

        # ── A: Direct synonyms from alias dictionaries ─────────────────────────
        for name in entity_names:
            # drug synonyms
            for canonical, aliases in DRUG_ALIASES.items():
                if canonical == name or name in [a.lower() for a in aliases]:
                    # top English aliases (skip obvious duplicates)
                    for alias in aliases[:6]:
                        if alias.lower() == alias and len(alias.split()) <= 4:
                            queries.append(alias)
                    # add combo queries with other entities
                    for other in entity_names:
                        if other != name:
                            queries.append(f"{canonical} {other}")
                    break
            # condition/symptom synonyms
            for d in (CONDITION_ALIASES, SYMPTOM_ALIASES):
                for canonical, aliases in d.items():
                    if canonical == name or name in [a.lower() for a in aliases]:
                        for alias in aliases[:4]:
                            if alias.lower() == alias and len(alias.split()) <= 4:
                                queries.append(alias)
                        break

        # ── B: MeSH terms ──────────────────────────────────────────────────────
        for name in entity_names:
            for mesh in get_mesh_terms(name):
                queries.append(mesh)
                # combo with other entities
                for other in entity_names:
                    if other != name:
                        queries.append(f"{mesh} {other}")

        # ── C: Knowledge graph neighbors ──────────────────────────────────────
        for name in entity_names:
            neighbors = get_neighbors(name, depth=1)
            # Only include neighbors that are recognisable entities
            all_known = (
                set(DRUG_ALIASES) |
                set(CONDITION_ALIASES) |
                set(SYMPTOM_ALIASES)
            )
            useful_neighbors = neighbors & all_known
            for neighbor in useful_neighbors:
                # build a combo query: drug + effect/condition
                for name2 in entity_names:
                    queries.append(f"{name2} {neighbor}")

        # ── D: LLM expansion ──────────────────────────────────────────────────
        if self._client and intent.has_entities():
            try:
                llm_queries = self._llm_expand(intent)
                queries.extend(llm_queries)
            except Exception as exc:
                logger.warning("LLM query expansion failed: %s", exc)

        # If we have no entities and no LLM, fall back to original query
        if not queries:
            queries.append(intent.original_query)

        # ── Deduplicate + order + cap ──────────────────────────────────────────
        return self._rank_queries(queries, intent)

    def _llm_expand(self, intent: IntentResult) -> list[str]:
        """Ask GPT-4o-mini for additional scientific query variants."""
        entity_list = ", ".join(
            f"{e.normalized} ({e.entity_type})" for e in intent.entities
        )
        prompt = (
            "You are a biomedical literature search specialist.\n"
            "Given the entities extracted from a user query, generate scientific "
            "PubMed-style search queries that would retrieve relevant studies.\n\n"
            "Include:\n"
            "- MeSH terms\n"
            "- Synonyms and spelling variants\n"
            "- Abbreviations\n"
            "- Related clinical terminology used in abstract writing\n"
            "- Specific adverse effect names\n\n"
            "Do NOT include the original Italian text.\n"
            "All queries must be in English.\n"
            "Return JSON: {\"queries\": [str, ...]}\n"
            "Generate as many useful, non-redundant queries as possible "
            "(aim for 10–15).\n\n"
            f"Entities: {entity_list}\n"
            f"Original query: {intent.original_query}"
        )
        resp = self._client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"},
            max_tokens=600,
        )
        data = json.loads(resp.choices[0].message.content)
        return [str(q).strip() for q in data.get("queries", []) if q]

    def _rank_queries(self, queries: list[str], intent: IntentResult) -> list[str]:
        """
        Deduplicate, normalise, and sort queries by estimated quality.
        Priority: queries containing known entity names > general terms.
        """
        seen:    set[str] = set()
        ranked:  list[tuple[int, str]] = []

        entity_names_lower = {e.normalized.lower() for e in intent.entities}

        for q in queries:
            q = q.strip()
            if not q or len(q) < 3:
                continue
            key = q.lower()
            if key in seen:
                continue
            seen.add(key)

            # Score: how many entity names does this query contain?
            hits = sum(1 for name in entity_names_lower if name in key)
            ranked.append((hits, q))

        # Stable sort: more entity overlap first
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [q for _, q in ranked[:MAX_EXPANDED_QUERIES]]

    # =========================================================================
    # Phase 4+8 — Parallel search (scientific + Reddit)
    # =========================================================================

    def _parallel_search(
        self,
        queries: list[str],
    ) -> dict[str, list[tuple[str, SearchResult]]]:
        """
        Run every (query × collector) combination concurrently.

        Returns
        -------
        {
            "pubmed":         [(query, SearchResult), ...],
            "europepmc":      [(query, SearchResult), ...],
            "clinicaltrials": [(query, SearchResult), ...],
            "reddit":         [(query, SearchResult), ...],
        }
        """
        sci_collectors  = ["pubmed", "europepmc", "clinicaltrials"]
        all_sources     = sci_collectors + ["reddit"]

        # Build task list: (source_name, query)
        tasks: list[tuple[str, str]] = []
        for source in all_sources:
            if source not in self._collectors:
                continue
            for q in queries:
                tasks.append((source, q))

        buckets: dict[str, list[tuple[str, SearchResult]]] = {
            s: [] for s in all_sources
        }

        def _call(source: str, q: str) -> tuple[str, str, list[SearchResult]]:
            collector = self._collectors[source]
            try:
                articles = collector.search(q, limit=COLLECTOR_LIMIT)
                return source, q, articles
            except Exception as exc:
                logger.warning("Collector %s failed for query '%s': %s", source, q, exc)
                return source, q, []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(_call, src, q): (src, q) for src, q in tasks}
            for fut in as_completed(futures):
                source, q, articles = fut.result()
                for art in articles:
                    buckets[source].append((q, art))

        return buckets

    # =========================================================================
    # Phase 5 — Merge + deduplicate  (reuses HLEOAggregator helpers)
    # =========================================================================

    def _merge_deduplicate(
        self,
        raw: dict[str, list[tuple[str, SearchResult]]],
    ) -> dict[str, list[SearchResult]]:
        """
        Flatten (query, article) pairs per source, deduplicate using
        HLEOAggregator.create_key() + completeness_score(), and return
        per-source lists of unique SearchResult objects.
        """
        from aggregator import HLEOAggregator
        agg = HLEOAggregator()

        merged: dict[str, list[SearchResult]] = {
            "pubmed": [], "europepmc": [], "clinicaltrials": [], "reddit": [],
        }

        for source in merged:
            pairs = raw.get(source, [])
            best_by_key: dict[str, tuple[float, str, SearchResult]] = {}

            for q, art in pairs:
                key = agg.create_key(art)
                if key is None:
                    key = f"notitle:{id(art)}"
                score = agg.completeness_score(art)
                existing = best_by_key.get(key)
                if existing is None or score > existing[0]:
                    best_by_key[key] = (score, q, art)

            # Attach the best-performing query to metadata for logging
            for completeness, best_q, art in best_by_key.values():
                art.metadata["_matched_query"] = best_q
                merged[source].append(art)

        return merged

    # =========================================================================
    # Phase 11 — Fallback (broaden scope when too few results)
    # =========================================================================

    def _fallback(
        self,
        intent: IntentResult,
        used_queries: list[str],
        current_merged: dict[str, list[SearchResult]],
    ) -> tuple[dict[str, list[SearchResult]], list[str]]:
        """
        Generate broader queries and retry search.  Merge new results with
        the existing pool.  Runs at most MAX_FALLBACK_ROUNDS rounds.
        """
        from aggregator import HLEOAggregator
        agg = HLEOAggregator()

        extra_queries: list[str] = []

        for round_n in range(MAX_FALLBACK_ROUNDS):
            sci_count = (
                len(current_merged.get("pubmed", [])) +
                len(current_merged.get("europepmc", [])) +
                len(current_merged.get("clinicaltrials", []))
            )
            if sci_count >= MIN_ARTICLES:
                break

            logger.info(
                "Fallback round %d — current: %d articles (target: %d)",
                round_n + 1, sci_count, MIN_ARTICLES,
            )

            new_queries = self._generate_fallback_queries(intent, used_queries, round_n)
            if not new_queries:
                logger.info("Fallback: no further expansions possible.")
                break

            extra_queries.extend(new_queries)
            raw2 = self._parallel_search(new_queries)
            new_merged = self._merge_deduplicate(raw2)

            # Merge new results into current pool (dedup again)
            for source in current_merged:
                existing_keys: set[str] = set()
                for art in current_merged[source]:
                    k = agg.create_key(art)
                    if k:
                        existing_keys.add(k)

                for art in new_merged.get(source, []):
                    k = agg.create_key(art)
                    if k is None or k not in existing_keys:
                        current_merged[source].append(art)
                        if k:
                            existing_keys.add(k)

        return current_merged, extra_queries

    def _generate_fallback_queries(
        self,
        intent: IntentResult,
        used_queries: list[str],
        round_n: int,
    ) -> list[str]:
        """Broaden scope: remove the most specific constraint each round."""
        used_lower = {q.lower() for q in used_queries}

        candidates: list[str] = []

        if round_n == 0:
            # Use only the primary entity (drop secondary entities)
            for ent in intent.entities:
                if ent.entity_type == "drug":
                    candidates.extend([
                        f"{ent.normalized} adverse effects",
                        f"{ent.normalized} side effects",
                        f"{ent.normalized} safety",
                        f"{ent.normalized} pharmacology",
                    ])
                    for mesh in get_mesh_terms(ent.normalized):
                        candidates.append(mesh)
                elif ent.entity_type in ("condition", "symptom"):
                    candidates.extend([
                        f"{ent.normalized} treatment",
                        f"{ent.normalized} etiology",
                        ent.normalized,
                    ])
        elif round_n == 1:
            # Very broad: just drug class or condition class
            for ent in intent.entities:
                neighbors = get_neighbors(ent.normalized, depth=2)
                for n in list(neighbors)[:5]:
                    candidates.append(n)
            # Also try LLM if available
            if self._client:
                try:
                    entity_list = ", ".join(intent.entity_names())
                    prompt = (
                        "Generate 5 broad PubMed search queries for the topic: "
                        f"{entity_list}. Make them wider in scope than the original "
                        "queries. Return JSON: {\"queries\": [str, ...]}."
                    )
                    resp = self._client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.5,
                        response_format={"type": "json_object"},
                        max_tokens=300,
                    )
                    data = json.loads(resp.choices[0].message.content)
                    candidates.extend(data.get("queries", []))
                except Exception as exc:
                    logger.warning("Fallback LLM expansion failed: %s", exc)

        return [q for q in candidates if q.lower() not in used_lower]

    # =========================================================================
    # Phase 6 — Ranking with semantic bonus
    # =========================================================================

    def _rank_with_semantic_bonus(
        self,
        merged: dict[str, list[SearchResult]],
        intent: IntentResult,
    ) -> None:
        """
        1. Apply existing clinical_rank() to every scientific article.
        2. Add SEMANTIC_BONUS for articles whose title/abstract contain
           at least one recognised entity from the intent.
        3. Sort each source list descending by score.
        Reddit is never touched (consistent with existing pipeline).
        """
        from core.ranker import clinical_rank

        entity_terms: list[str] = []
        for ent in intent.entities:
            entity_terms.append(ent.normalized.lower())
            # include aliases for matching
            from core.biomedical_kb import DRUG_ALIASES, CONDITION_ALIASES, SYMPTOM_ALIASES
            for d in (DRUG_ALIASES, CONDITION_ALIASES, SYMPTOM_ALIASES):
                if ent.normalized in d:
                    entity_terms.extend(
                        [a.lower() for a in d[ent.normalized][:5] if len(a) > 3]
                    )
                    break

        for source in ("pubmed", "europepmc", "clinicaltrials"):
            lst = merged.get(source, [])
            for art in lst:
                base = clinical_rank(art)
                # Semantic bonus
                haystack = (
                    (getattr(art, "title",    "") or "") + " " +
                    (getattr(art, "abstract", "") or "")
                ).lower()
                bonus = SEMANTIC_BONUS if any(t in haystack for t in entity_terms) else 0.0
                art.score = round(base + bonus, 2)

            lst.sort(key=lambda a: a.score, reverse=True)

    # =========================================================================
    # Phase 10 — Persistent DB cache
    # =========================================================================

    def _db_cache_get(self, cache_key: str) -> Optional[SemanticResult]:
        try:
            from core.database import SessionLocal
            from core.models import SemanticSearchCache
            with SessionLocal() as db:
                row = db.query(SemanticSearchCache).filter_by(query_hash=cache_key).first()
                if row is None:
                    return None
                if row.expires_at and datetime.now(timezone.utc) > row.expires_at:
                    db.delete(row)
                    db.commit()
                    return None
                row.hit_count = (row.hit_count or 0) + 1
                db.commit()
                return self._deserialise_result(row.result_json)
        except Exception as exc:
            logger.debug("DB cache get failed: %s", exc)
            return None

    def _db_cache_set(self, cache_key: str, query: str, result: SemanticResult) -> None:
        try:
            from core.database import SessionLocal
            from core.models import SemanticSearchCache
            from sqlalchemy import select as sa_select
            with SessionLocal() as db:
                existing = db.query(SemanticSearchCache).filter_by(query_hash=cache_key).first()
                payload = self._serialise_result(result)
                expires = datetime.now(timezone.utc) + timedelta(minutes=CACHE_TTL_MINUTES)
                if existing:
                    existing.result_json  = payload
                    existing.expires_at   = expires
                    existing.hit_count    = (existing.hit_count or 0) + 1
                else:
                    db.add(SemanticSearchCache(
                        query_hash       = cache_key,
                        original_query   = query,
                        normalized_query = query.lower().strip(),
                        entities_json    = [
                            {"text": e.text, "type": e.entity_type,
                             "normalized": e.normalized}
                            for e in result.expansion.get("_entities_obj", [])
                        ],
                        expanded_queries_json = result.expansion.get("expanded_queries", []),
                        result_json      = payload,
                        expires_at       = expires,
                    ))
                db.commit()
        except Exception as exc:
            logger.debug("DB cache set failed: %s", exc)

    @staticmethod
    def _serialise_result(result: SemanticResult) -> dict:
        from dataclasses import asdict
        def _ser(lst):
            out = []
            for item in lst:
                try:
                    out.append(asdict(item) if hasattr(item, "__dataclass_fields__") else item)
                except Exception:
                    out.append(item)
            return out
        return {
            "pubmed":         _ser(result.pubmed),
            "europepmc":      _ser(result.europepmc),
            "clinicaltrials": _ser(result.clinicaltrials),
            "reddit":         _ser(result.reddit),
            "expansion":      result.expansion,
        }

    @staticmethod
    def _deserialise_result(data: dict) -> Optional[SemanticResult]:
        if not data:
            return None
        return SemanticResult(
            pubmed         = data.get("pubmed", []),
            europepmc      = data.get("europepmc", []),
            clinicaltrials = data.get("clinicaltrials", []),
            reddit         = data.get("reddit", []),
            expansion      = data.get("expansion", {}),
        )

    # =========================================================================
    # Finalise result
    # =========================================================================

    def _to_result(
        self,
        merged:            dict[str, list[SearchResult]],
        intent:            IntentResult,
        expanded_queries:  list[str],
        fallback_triggered: bool,
    ) -> SemanticResult:
        from dataclasses import asdict

        def _ser_list(lst):
            out = []
            for art in lst:
                try:
                    d = asdict(art)
                except Exception:
                    d = {"title": getattr(art, "title", ""), "source": getattr(art, "source", "")}
                out.append(d)
            return out

        # Compute best-performing query (most matches)
        query_counts: dict[str, int] = {}
        for source in ("pubmed", "europepmc", "clinicaltrials"):
            for art in merged.get(source, []):
                q = art.metadata.get("_matched_query", "")
                if q:
                    query_counts[q] = query_counts.get(q, 0) + 1
        best_query = max(query_counts, key=query_counts.get) if query_counts else ""

        expansion = {
            "intent": {
                "original_query":   intent.original_query,
                "detected_language":intent.detected_language,
                "category":         intent.category,
                "used_llm":         intent.used_llm,
                "entities": [
                    {
                        "text":        e.text,
                        "type":        e.entity_type,
                        "normalized":  e.normalized,
                        "confidence":  e.confidence,
                    }
                    for e in intent.entities
                ],
            },
            "expanded_queries":   expanded_queries,
            "queries_used":       len(expanded_queries),
            "best_query":         best_query,
            "articles_per_source": {
                src: len(merged.get(src, []))
                for src in ("pubmed", "europepmc", "clinicaltrials", "reddit")
            },
            "total_articles": sum(
                len(merged.get(s, []))
                for s in ("pubmed", "europepmc", "clinicaltrials")
            ),
            "fallback_triggered": fallback_triggered,
            "_entities_obj":      intent.entities,  # internal; stripped before JSON
        }

        result = SemanticResult(
            pubmed         = _ser_list(merged.get("pubmed", [])),
            europepmc      = _ser_list(merged.get("europepmc", [])),
            clinicaltrials = _ser_list(merged.get("clinicaltrials", [])),
            reddit         = _ser_list(merged.get("reddit", [])),
            expansion      = expansion,
        )
        return result

    # =========================================================================
    # Phase 12 — Structured JSON logging
    # =========================================================================

    def _log(
        self,
        query:     str,
        expansion: dict,
        cache_hit: bool,
        elapsed:   float,
    ) -> None:
        record = {
            "event":              "semantic_search",
            "query":              query,
            "detected_language":  expansion.get("intent", {}).get("detected_language"),
            "entities":           [
                e.get("normalized") for e in
                expansion.get("intent", {}).get("entities", [])
            ],
            "expanded_queries":   expansion.get("expanded_queries", []),
            "queries_used":       expansion.get("queries_used", 0),
            "articles_per_source":expansion.get("articles_per_source", {}),
            "total_articles":     expansion.get("total_articles", 0),
            "best_query":         expansion.get("best_query", ""),
            "cache_hit":          cache_hit,
            "fallback_triggered": expansion.get("fallback_triggered", False),
            "elapsed_s":          elapsed,
            "errors":             expansion.get("errors", []),
        }
        # Remove internal key before logging
        record.pop("_entities_obj", None)
        logger.info("SEMANTIC_SEARCH %s", json.dumps(record, ensure_ascii=False))
