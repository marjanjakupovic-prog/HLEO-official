"""
RWE Query Engine — autonomous query preparation, translation, and controlled
query expansion for the Real World Evidence search pipeline.

Flow
----
    User Query (any language)
      → language detection (quick heuristic + QueryOrchestrator)
      → query preparation (normalise + quick Italian→English token map)
      → translation (QueryOrchestrator → scientific English; original kept)
      → intent / entity recognition (biomedical_kb.lookup_entity)
      → controlled query expansion (KB synonyms + MeSH + graph neighbours
        + entity combos + trichology supplement)
      → RWEQueryPlan

Design rules
------------
- ``original_query`` is NEVER overwritten.
- Expansion is CONTROLLED: it stays anchored to the recognised entities
  (a finasteride query remains about finasteride + its effects; it never
  broadens into generic hair-loss chatter).
- Reuses the existing ``biomedical_kb`` dictionaries and ``QueryOrchestrator``
  — no duplicated synonym/translation logic.
- KB-first, deterministic. LLM expansion is optional (only when an OpenAI key
  is present) and bounded, so tests run without network/API.
- Trichology-focused: a supplement of colloquial shedding/regrowth phrasings
  is used as an EXPANSION BASE, never as a rigid filter.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional

from core.biomedical_kb import (
    DRUG_ALIASES,
    CONDITION_ALIASES,
    SYMPTOM_ALIASES,
    get_mesh_terms,
    get_neighbors,
    lookup_entity,
    quick_translate_it,
)
from core.orchestrator import QueryOrchestrator
from core.rwe.intent import build_intent, intent_scoring_enabled

logger = logging.getLogger(__name__)

# ── Tunables ─────────────────────────────────────────────────────────────────
MAX_EXPANDED_QUERIES = 16          # hard cap per search plan
MIN_EXPANDED_QUERIES = 1           # always at least the prepared query
ENTITY_CONFIDENCE_FLOOR = 0.5      # entities below this are dropped
MAX_SYNONYMS_PER_ENTITY = 3        # keep expansion controlled & leave room for
                                   # mesh/combo/colloquial tiers

# Expansion type tags — carried as provenance on every expanded query.
EXP_ORIGINAL   = "original"
EXP_TRANSLATED = "translated"
EXP_CANONICAL  = "canonical"
EXP_SYNONYM    = "synonym"
EXP_MESH       = "mesh"
EXP_NEIGHBOR   = "neighbor"
EXP_COMBO      = "combo"
EXP_COLLOQUIAL = "colloquial"
EXP_LLM        = "llm"


# ── Trichology supplement ────────────────────────────────────────────────────
# Colloquial / patient-language phrasings that surface community testimony.
# These are anchored to a recognised symptom concept (canonical) so they only
# expand when that concept is present in the query — they never broaden scope
# on their own. This is the "medical vs colloquial" bridge.
_TRICHOLOGY_COLLOQUIAL: dict[str, list[str]] = {
    "hair loss": [
        "initial shedding", "increased hair loss", "hair shedding",
        "temporary worsening", "increased hair fall", "hair fall out",
        "caduta iniziale", "peggioramento della caduta",
        "caduta aumentata", "perdita maggiore",
    ],
    "sexual dysfunction": [
        "lost my libido", "low sex drive", "can't get it up",
        "erection problems", "calo del desiderio",
        "problemi in letto", "fame sessuale sparita",
    ],
    "telogen effluvium": [
        "diffuse shedding", "hair falling out everywhere",
        "caduta diffusa", "caduta massiccia",
    ],
    "androgenetic alopecia": [
        "thinning crown", "receding hairline", "bald spot",
        "diradamento", "tempie che arretrano", "chiericato",
    ],
}


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class ExpandedQuery:
    """A single expanded search query with full provenance."""
    query: str
    expansion_type: str          # EXP_* constant
    source_language: str = "en"  # language of this query string
    matched_entities: list = field(default_factory=list)  # canonical names
    original_term: str = ""
    expanded_term: str = ""
    match_kind: Optional[str] = None
    tier: Optional[float] = None
    provider: Optional[str] = None
    source_entity: Optional[str] = None
    query_origin: str = "user"


@dataclass
class RWEQueryPlan:
    """
    Full query plan consumed by the RWE pipeline.

    ``original_query`` is the verbatim user input — never mutated.
    ``translated_query`` is the English rendering (== original when already EN).
    ``expanded_queries`` is the controlled set of queries sent to collectors.
    """
    original_query: str
    detected_language: str
    translated_query: str
    translation_applied: bool
    canonical_query: str = ""
    entities: list = field(default_factory=list)        # (type, canonical, conf)
    vocabulary: dict = field(default_factory=dict)
    expanded_queries: List[ExpandedQuery] = field(default_factory=list)
    # QU-aware relevance (V3): structured intent, built ONLY when the
    # HLEO_RWE_INTENT_SCORING feature flag is on; None = pure V1 behaviour.
    intent: Optional[object] = None                   # RWEQueryIntent | None

    def query_strings(self) -> List[str]:
        """Flat list of query strings (deduplicated, order preserved)."""
        seen, out = set(), []
        for eq in self.expanded_queries:
            k = eq.query.lower().strip()
            if k and k not in seen:
                seen.add(k)
                out.append(eq.query)
        return out

    def to_dict(self) -> dict:
        return {
            "original_query": self.original_query,
            "detected_language": self.detected_language,
            "translated_query": self.translated_query,
            "translation_applied": self.translation_applied,
            "entities": [
                {"type": t, "canonical": c, "confidence": conf}
                for t, c, conf in self.entities
            ],
            "expanded_queries": [
                {
                    "query": eq.query,
                    "expansion_type": eq.expansion_type,
                    "source_language": eq.source_language,
                    "matched_entities": eq.matched_entities,
                    "original_term": eq.original_term,
                    "expanded_term": eq.expanded_term,
                    "match_kind": eq.match_kind,
                    "tier": eq.tier,
                    "provider": eq.provider,
                    "source_entity": eq.source_entity,
                    "query_origin": eq.query_origin,
                }
                for eq in self.expanded_queries
            ],
            "intent": (
                self.intent.model_dump()
                if self.intent is not None and hasattr(self.intent, "model_dump")
                else None
            ),
        }


# ── Language detection (non-LLM, instant) ───────────────────────────────────

# Stopword / marker sets per language. Used only for fast detection so the
# plan carries an honest detected_language even when no OpenAI key is present.
_LANG_MARKERS = {
    "it": {
        "di", "del", "della", "dei", "che", "per", "con", "senza", "dopo",
        "prima", "può", "possono", "causare", "indotta", "indotto", "perdita",
        "caduta", "dolore", "dolori", "problemi", "sessuale", "sessuali",
        "capelli", "diradamento", "effetti", "collaterali", "memoria",
    },
    "de": {
        "und", "oder", "von", "mit", "ohne", "nach", "vor", "kann", "können",
        "verursachen", "haarausfall", "haare", "nebenwirkungen", "schmerzen",
    },
    "fr": {
        "et", "ou", "de", "du", "avec", "sans", "après", "avant", "peut",
        "peuvent", "causer", "provoquer", "perte", "cheveux", "chute",
        "effets", "secondaires",
    },
    "es": {
        "y", "o", "de", "del", "con", "sin", "después", "antes", "puede",
        "pueden", "causar", "provocar", "pérdida", "cabello", "caída",
        "efectos", "secundarios",
    },
}


def detect_language(text: str) -> str:
    """
    Fast stopword/marker-based language detection.

    Returns an ISO-639-1 code. Falls back to ``en`` when no marker matches
    (English is the default source language of the RWE collectors).
    """
    if not text or not text.strip():
        return "und"
    tokens = set(re.findall(r"[a-zà-öø-ÿ]+", text.lower()))
    if not tokens:
        return "und"
    best_lang, best_score = "en", 0
    for lang, markers in _LANG_MARKERS.items():
        score = len(tokens & markers)
        if score > best_score:
            best_lang, best_score = lang, score
    return best_lang


# ── Helpers ──────────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    """Lower-case, strip diacritics, collapse whitespace."""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", ascii_text.lower().strip())


def _prepare(query: str, lang: str) -> str:
    """
    Query preparation: normalise whitespace + apply the quick Italian→English
    token map when the source language is Italian (reduces LLM dependency for
    common medical terms). Non-Italian queries are returned normalised only.
    """
    q = re.sub(r"\s+", " ", (query or "").strip())
    if lang == "it":
        q = quick_translate_it(q)
    return q


def _strip_punct(text: str) -> str:
    """Replace punctuation with spaces so biomedical_kb n-gram matching works
    (the KB's _norm keeps punctuation attached to tokens, e.g. 'shedding?')."""
    return re.sub(r"[^\w\sà-öø-ÿ-]", " ", text or "").strip()


# ── Query Engine ─────────────────────────────────────────────────────────────

class RWEQueryEngine:
    """
    Builds a controlled RWEQueryPlan from a user query.

    Reuses ``QueryOrchestrator`` (language detection + translation) and the
    ``biomedical_kb`` dictionaries (synonyms, MeSH, knowledge graph). Stateless
    apart from the orchestrator's internal cache.
    """

    def __init__(self, orchestrator: Optional[QueryOrchestrator] = None) -> None:
        self._orchestrator = orchestrator or QueryOrchestrator()

    # ── Public API ───────────────────────────────────────────────────────────

    def plan(self, query: str) -> RWEQueryPlan:
        """Build the full query plan for a user query (any language)."""
        q = (query or "").strip()
        if not q:
            return RWEQueryPlan(
                original_query=query or "",
                detected_language="und",
                translated_query="",
                translation_applied=False,
            )

        # ── 1. Language detection (quick, non-LLM) ───────────────────────────
        lang = detect_language(q)

        # ── 2. Query preparation (normalise + quick IT→EN token map) ────────
        prepared = _prepare(q, lang)

        # ── 3. Translation via the existing orchestrator ────────────────────
        # The orchestrator detects language + translates to English with an LLM
        # when an OpenAI key is present; otherwise it passes through. We trust
        # its English output as ``translated_query`` but ALWAYS keep the
        # original user input separate.
        orch = self._orchestrator.process(q)
        translated = (orch.search_query or prepared or q).strip()
        # Prefer the orchestrator's detected language when it is confident
        # (not "und"), otherwise keep our quick-detection result.
        if orch.detected_language and orch.detected_language != "und":
            lang = orch.detected_language
        translation_applied = bool(
            orch.translation_applied
            or translated.lower() != q.lower()
        )

        # ── 4. Canonicalization (the translated representation remains intact)
        entities = self._recognize_entities(translated, q)
        canonical = self._canonicalize(translated, entities)

        # ── 5. Vocabulary resolution, once per canonical entity ─────────────
        # This is deliberately before expansion and retrieval.  ``None`` means
        # the feature is off: no provider is instantiated or queried.
        vocabulary = {}
        resolutions = {}
        from core.vocab.resolver import build_resolver_from_env
        resolver = build_resolver_from_env()
        if resolver is not None:
            terms = list(dict.fromkeys([c for _, c, _ in entities]))
            resolutions = resolver.resolve_terms(terms, language=lang)
            vocabulary = self._slim_vocabulary(resolutions)
            entities = self._enrich_entities(entities, resolutions)

        # ── 6. Controlled query expansion (now consumes vocabulary evidence)
        expanded = self._expand(
            original=q,
            translated=translated,
            canonical_query=canonical,
            prepared=prepared,
            lang=lang,
            entities=entities,
            resolutions=resolutions,
        )

        # ── 7. QU intent for the existing V3 relevance scorer ────────────────
        intent = None
        if intent_scoring_enabled():
            intent = build_intent(translated, q, entities, use_llm=True)
            if vocabulary:
                intent.vocabulary = vocabulary

        return RWEQueryPlan(
            original_query=q,
            detected_language=lang,
            translated_query=translated,
            translation_applied=translation_applied,
            canonical_query=canonical,
            entities=entities,
            vocabulary=vocabulary,
            expanded_queries=expanded,
            intent=intent,
        )

    @staticmethod
    def _canonicalize(translated: str, entities: list) -> str:
        """Return a canonical representation without mutating translation."""
        out = translated
        dictionaries = (DRUG_ALIASES, CONDITION_ALIASES, SYMPTOM_ALIASES)
        for _etype, canonical, _ in entities:
            aliases = [canonical]
            for mapping in dictionaries:
                aliases.extend(mapping.get(canonical, []))
            for alias in sorted(set(aliases), key=len, reverse=True):
                if alias and alias.lower() != canonical.lower():
                    out = re.sub(rf"(?<!\w){re.escape(alias)}(?!\w)", canonical,
                                 out, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", out).strip()

    @staticmethod
    def _slim_vocabulary(resolutions: dict) -> dict:
        return {
            term: [m.model_dump() for m in resolution.matches]
            for term, resolution in resolutions.items()
            if resolution.matches
        }

    @staticmethod
    def _enrich_entities(entities: list, resolutions: dict) -> list:
        """Use provider semantic groups as typed evidence, never replacement."""
        out = list(entities)
        known = {canonical for _, canonical, _ in out}
        group_types = {"drug": "drug", "condition": "condition", "symptom": "symptom"}
        for _term, resolution in resolutions.items():
            for match in resolution.matches:
                etype = group_types.get(match.semantic_group)
                canonical = match.preferred_term.strip().lower()
                if etype and canonical and canonical not in known:
                    out.append((etype, canonical, match.confidence))
                    known.add(canonical)
        return out

    @staticmethod
    def _replace_entity(text: str, original: str, replacement: str) -> str:
        """Replace one anchored entity, retaining the remaining query context."""
        replaced = re.sub(rf"(?<!\w){re.escape(original)}(?!\w)", replacement,
                          text, count=1, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", replaced).strip()

    # ── Entity recognition ───────────────────────────────────────────────────

    def _recognize_entities(self, translated: str, original: str) -> list:
        """Hybrid KB entity lookup on translated + original text."""
        # Strip trailing/leading punctuation that would break n-gram matching
        # in biomedical_kb (e.g. "shedding?" != "shedding"). The KB's own _norm
        # does not remove punctuation, so we clean it here.
        translated = _strip_punct(translated)
        original = _strip_punct(original)
        hits = lookup_entity(translated)
        hits_orig = lookup_entity(original)
        # Merge — keep best confidence per canonical name
        best: dict[str, tuple] = {h[1]: h for h in hits}
        for h in hits_orig:
            if h[1] not in best or h[2] > best[h[1]][2]:
                best[h[1]] = h
        return [
            (etype, canonical, conf)
            for etype, canonical, conf in best.values()
            if conf >= ENTITY_CONFIDENCE_FLOOR
        ]

    # ── Controlled query expansion ───────────────────────────────────────────

    def _expand(
        self,
        original: str,
        translated: str,
        canonical_query: str,
        prepared: str,
        lang: str,
        entities: list,
        resolutions: Optional[dict] = None,
    ) -> List[ExpandedQuery]:
        """
        Build the controlled set of expanded queries.

        Anchored to recognised entities: every expanded query is either the
        original/translated query, a synonym of a recognised entity, a MeSH
        term of a recognised entity, a graph neighbour of a recognised entity,
        or a combo of recognised entities. Nothing broadens beyond them.
        """
        queries: List[ExpandedQuery] = []
        canonical_names = [c for _, c, _ in entities]

        # ── A: original + translated (always present) ───────────────────────
        queries.append(ExpandedQuery(
            query=original,
            expansion_type=EXP_ORIGINAL,
            source_language=lang,
            matched_entities=canonical_names,
            original_term=original,
            expanded_term=original,
            query_origin="user",
        ))
        if translated.lower() != original.lower():
            queries.append(ExpandedQuery(
                query=translated,
                expansion_type=EXP_TRANSLATED,
                source_language="en",
                matched_entities=canonical_names,
                original_term=original,
                expanded_term=translated,
                query_origin="translation",
            ))
        if canonical_query and canonical_query.lower() not in {original.lower(), translated.lower()}:
            queries.append(ExpandedQuery(
                query=canonical_query,
                expansion_type=EXP_CANONICAL,
                source_language="en",
                matched_entities=canonical_names,
                original_term=original,
                expanded_term=canonical_query,
                query_origin="canonicalization",
            ))

        if not entities:
            # No entities recognised — fall back to the prepared query so the
            # collectors still receive something usable.
            if prepared and prepared.lower() not in {q.query.lower() for q in queries}:
                queries.append(ExpandedQuery(
                    query=prepared,
                    expansion_type=EXP_TRANSLATED,
                    source_language="en",
                    matched_entities=[],
                ))
            return self._dedup_and_cap(queries, canonical_names)

        # ── B: KB synonyms for each entity ─────────────────────────────────
        for etype, canonical, _ in entities:
            alias_dict = self._alias_dict_for(etype, canonical)
            if alias_dict and canonical in alias_dict:
                for alias in alias_dict[canonical][:MAX_SYNONYMS_PER_ENTITY]:
                    if len(alias.split()) <= 4:
                        queries.append(ExpandedQuery(
                            query=self._replace_entity(canonical_query := (canonical_query or translated), canonical, alias),
                            expansion_type=EXP_SYNONYM,
                            source_language="en",
                            matched_entities=[canonical],
                            original_term=canonical,
                            expanded_term=alias,
                            match_kind="synonym",
                            tier=0.9,
                            source_entity=canonical,
                            query_origin="kb",
                        ))

        # ── C: MeSH terms ──────────────────────────────────────────────────
        for etype, canonical, _ in entities:
            for mesh in get_mesh_terms(canonical):
                queries.append(ExpandedQuery(
                    query=self._replace_entity(canonical_query or translated, canonical, mesh),
                    expansion_type=EXP_MESH,
                    source_language="en",
                    matched_entities=[canonical],
                    original_term=canonical,
                    expanded_term=mesh,
                    match_kind="concept",
                    tier=0.6,
                    source_entity=canonical,
                    query_origin="kb",
                ))

        # ── B2: external vocabulary terms used for real retrieval ──────────
        # Expand one entity at a time, preserving all other query context. A
        # related concept is deliberately not lexicalized as a synonym.
        from core.vocab.models import MATCH_TIERS
        for _etype, source_entity, _ in entities:
            resolution = (resolutions or {}).get(source_entity)
            if resolution is None:
                continue
            for match in resolution.matches:
                tier = MATCH_TIERS.get(match.match_kind)
                if tier is None:
                    continue
                for term in [match.preferred_term, *match.synonyms]:
                    term = (term or "").strip()
                    if len(term) < 3 or term.lower() == source_entity.lower():
                        continue
                    base = canonical_query or translated or original
                    expanded_query = self._replace_entity(base, source_entity, term)
                    if expanded_query == base:
                        expanded_query = " ".join(
                            [term] + [c for _, c, _ in entities if c != source_entity]
                        )
                    queries.append(ExpandedQuery(
                        query=expanded_query,
                        expansion_type=(EXP_SYNONYM if match.match_kind == "synonym"
                                        else match.match_kind),
                        source_language=match.language or "en",
                        matched_entities=[source_entity],
                        original_term=source_entity,
                        expanded_term=term,
                        match_kind=match.match_kind,
                        tier=tier,
                        provider=match.provider,
                        source_entity=source_entity,
                        query_origin="vocabulary",
                    ))

        # ── D: Knowledge graph neighbours (1-hop, anchored) ────────────────
        all_known = (
            set(DRUG_ALIASES) | set(CONDITION_ALIASES) | set(SYMPTOM_ALIASES)
        )
        for _etype, entity_name, _ in entities:
            neighbors = get_neighbors(entity_name, depth=1) & all_known
            for neighbor in neighbors:
                queries.append(ExpandedQuery(
                    query=f"{entity_name} {neighbor}",
                    expansion_type=EXP_NEIGHBOR,
                    source_language="en",
                    matched_entities=[entity_name, neighbor],
                ))

        # ── E: Entity combos (drug + symptom / drug + condition) ───────────
        if len(canonical_names) > 1:
            for i, (t1, c1, _) in enumerate(entities):
                for j, (t2, c2, _) in enumerate(entities):
                    if i < j and c1 != c2:
                        queries.append(ExpandedQuery(
                            query=f"{c1} {c2}",
                            expansion_type=EXP_COMBO,
                            source_language="en",
                            matched_entities=[c1, c2],
                        ))

        # ── F: Trichology colloquial supplement (anchored to symptoms) ─────
        for etype, canonical, _ in entities:
            colloquial = _TRICHOLOGY_COLLOQUIAL.get(canonical)
            if colloquial:
                # pair with any drug entity so the query stays specific
                drugs = [c for t, c, _ in entities if t == "drug"]
                for phrase in colloquial:
                    if drugs:
                        queries.append(ExpandedQuery(
                            query=f"{drugs[0]} {phrase}",
                            expansion_type=EXP_COLLOQUIAL,
                            source_language="en",
                            matched_entities=[canonical, drugs[0]],
                        ))
                    else:
                        queries.append(ExpandedQuery(
                            query=phrase,
                            expansion_type=EXP_COLLOQUIAL,
                            source_language="en",
                            matched_entities=[canonical],
                        ))

        return self._dedup_and_cap(queries, canonical_names)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _alias_dict_for(etype: str, canonical: str):
        """Return the alias dict that contains ``canonical`` for ``etype``."""
        for d in (DRUG_ALIASES, CONDITION_ALIASES, SYMPTOM_ALIASES):
            if canonical in d:
                return d
        return None

    @staticmethod
    def _dedup_and_cap(
        queries: List[ExpandedQuery], entity_names: list
    ) -> List[ExpandedQuery]:
        """
        Deduplicate by query text and cap to MAX_EXPANDED_QUERIES.

        Ordering preserves the provenance value of each expansion type:
        original/translated first, then synonym/mesh/combo (specific), then
        colloquial, then neighbour (broader). Within each tier, queries
        containing more entity names rank first.
        """
        seen: set[str] = set()
        unique: List[ExpandedQuery] = []
        for eq in queries:
            key = _norm(eq.query)
            if not key or len(key) < 2 or key in seen:
                continue
            seen.add(key)
            unique.append(eq)

        tier_order = {
            EXP_ORIGINAL: 0, EXP_TRANSLATED: 1, EXP_CANONICAL: 1,
            EXP_SYNONYM: 2, EXP_MESH: 2, EXP_COMBO: 2,
            "preferred": 2, "translation": 2, "abbreviation": 2,
            "orthographic_variant": 2, "normalized": 2,
            "colloquial": 3, "slang": 3, "concept": 4,
            EXP_COLLOQUIAL: 3, EXP_NEIGHBOR: 5, EXP_LLM: 6,
        }
        ent_lower = {n.lower() for n in entity_names}

        def _score(eq: ExpandedQuery) -> tuple:
            ent_hits = sum(1 for n in ent_lower if n in _norm(eq.query))
            return (tier_order.get(eq.expansion_type, 9), -ent_hits)

        unique.sort(key=_score)
        return unique[:MAX_EXPANDED_QUERIES] if len(unique) > MAX_EXPANDED_QUERIES else unique
