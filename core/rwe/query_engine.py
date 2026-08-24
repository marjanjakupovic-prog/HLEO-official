"""
RWE Query Engine — autonomous query preparation, translation, and controlled
query expansion for the Real World Evidence search pipeline.

Catena C — provider-first terminology
-------------------------------------
All terminology (entity recognition, canonicalisation, synonyms, MeSH,
colloquial/multilingual variants) is resolved through the EXTERNAL vocabulary
providers via ``core.vocab.resolver`` + ``core.vocab.entities`` (RxNorm, MeSH,
ConceptNet, Wikidata). NO internal hardcoded dictionaries are used.

Flow
----
    User Query (any language)
      → language detection (QueryOrchestrator LLM, any language; local
        heuristic only as a hint when the LLM is unavailable)
      → translation (QueryOrchestrator → scientific English; original kept)
      → entity recognition (external providers, EN + source language)
      → canonicalisation (provider preferred terms)
      → controlled query expansion (provider synonyms / translations /
        colloquial-slang / MeSH concepts + entity combos)
      → RWEQueryPlan

Design rules
------------
- ``original_query`` is NEVER overwritten.
- Expansion is CONTROLLED: it stays anchored to the recognised entities
  (a finasteride query remains about finasteride + its effects; it never
  broadens into generic hair-loss chatter).
- Providers never raise and are cache-backed; when no resolver is available
  (offline / HLEO_VOCAB_ENABLED=0) the plan degrades gracefully to the
  original/translated queries only.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional

from core.orchestrator import QueryOrchestrator
from core.rwe.intent import build_intent, intent_scoring_enabled
from core.vocab.entities import merge_recognitions, recognize
from core.vocab.resolver import build_resolver_from_env

logger = logging.getLogger(__name__)

# ── Tunables ─────────────────────────────────────────────────────────────────
MAX_EXPANDED_QUERIES = 16          # hard cap per search plan
MIN_EXPANDED_QUERIES = 1           # always at least the prepared query

# Expansion type tags — carried as provenance on every expanded query.
EXP_ORIGINAL   = "original"
EXP_TRANSLATED = "translated"
EXP_CANONICAL  = "canonical"
EXP_SYNONYM    = "synonym"
EXP_MESH       = "mesh"
EXP_NEIGHBOR   = "neighbor"      # legacy tag, no longer generated
EXP_COMBO      = "combo"
EXP_COLLOQUIAL = "colloquial"
EXP_LLM        = "llm"


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
    # QU-aware relevance (V3): structured intent; None = pure V1 behaviour.
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


# ── Language detection (local heuristic — hint only) ─────────────────────────
#
# The AUTHORITATIVE language detector is the QueryOrchestrator (LLM, any
# language). This stopword heuristic is only a fallback hint when the LLM has
# no opinion ("und", e.g. no API key); it returns "und" when it has no
# evidence rather than silently guessing "en".
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
    "en": {
        "the", "and", "or", "with", "without", "after", "before", "can",
        "could", "does", "do", "is", "are", "was", "were", "cause", "causes",
        "caused", "effect", "effects", "side", "hair", "loss", "shedding",
        "treatment", "my", "i", "me",
    },
}


def detect_language(text: str) -> str:
    """
    Fast stopword/marker-based language detection (HINT only).

    Returns an ISO-639-1 code, or ``und`` when no marker matches — the caller
    must not treat an unsupported language as English silently.
    """
    if not text or not text.strip():
        return "und"
    tokens = set(re.findall(r"[a-zà-öø-ÿ]+", text.lower()))
    if not tokens:
        return "und"
    best_lang, best_score = "und", 0
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


# ── Engine ───────────────────────────────────────────────────────────────────

class RWEQueryEngine:
    """
    Builds a controlled RWEQueryPlan from a user query.

    Terminology comes exclusively from the external vocabulary providers
    (Catena C). Stateless apart from the orchestrator's internal cache.
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

        # ── 1. Language detection + translation (orchestrator, any language) ─
        # The orchestrator's LLM detection is authoritative and NOT limited to
        # a fixed language set. The local heuristic is only a hint when the
        # orchestrator has no opinion ("und", e.g. no API key).
        orch = self._orchestrator.process(q)
        translated = (orch.search_query or q).strip()
        lang = (orch.detected_language or "und").lower()
        if lang == "und":
            lang = detect_language(q)
        translation_applied = bool(
            orch.translation_applied
            or translated.lower() != q.lower()
        )

        # ── 2. Entity recognition via external providers (Catena C) ─────────
        # The English rendering is resolved as English; the original text is
        # resolved in its own language so multilingual providers (ConceptNet)
        # can map native phrasings (cross-language edges → translations).
        resolver = build_resolver_from_env()
        entities: list = []
        resolutions: dict = {}
        surfaces: dict = {}
        if resolver is not None:
            rec_en = recognize(translated, "en", resolver)
            if lang not in {"und", "en"} and translated.lower() != q.lower():
                rec_src = recognize(q, lang, resolver)
                rec = merge_recognitions(rec_en, rec_src)
            else:
                rec = rec_en
            entities = rec.entities
            resolutions = rec.resolutions
            surfaces = rec.surfaces

        # ── 3. Canonicalisation (provider preferred terms; translation intact)
        canonical = self._canonicalize(translated, entities, surfaces)

        # ── 4. Controlled query expansion (provider terminology only) ────────
        vocabulary = self._slim_vocabulary(resolutions)
        expanded = self._expand(
            original=q,
            translated=translated,
            canonical_query=canonical,
            lang=lang,
            entities=entities,
            resolutions=resolutions,
        )

        # ── 5. QU intent for the V3 relevance scorer ─────────────────────────
        # The LLM-structured intent is feature-flagged; additionally, when the
        # query exposes a structured drug→event relation, a deterministic
        # entities-fallback intent is ALWAYS built (no LLM call) so the V3
        # relation-aware scorer activates even with the flag off. Queries
        # without a recognised relation keep intent=None → exact V1 behaviour.
        intent = None
        if intent_scoring_enabled():
            intent = build_intent(translated, q, entities, use_llm=True)
            if vocabulary:
                intent.vocabulary = vocabulary
        elif self._has_structured_relation(entities):
            intent = build_intent(translated, q, entities, use_llm=False)
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
    def _canonicalize(translated: str, entities: list, surfaces: dict) -> str:
        """Replace provider-verified surface forms with their canonical
        preferred term (e.g. "propecia" → "finasteride"). The translated
        representation itself is never mutated."""
        out = translated
        for _etype, canonical, _conf in entities or []:
            surface = (surfaces or {}).get(canonical, canonical)
            for term in {surface, canonical}:
                if term and term.lower() != canonical.lower():
                    out = re.sub(rf"(?<!\w){re.escape(term)}(?!\w)", canonical,
                                 out, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", out).strip()

    @staticmethod
    def _slim_vocabulary(resolutions: dict) -> dict:
        return {
            term: [m.model_dump() for m in resolution.matches]
            for term, resolution in (resolutions or {}).items()
            if resolution.matches
        }

    @staticmethod
    def _replace_entity(text: str, original: str, replacement: str) -> str:
        """Replace one anchored entity, retaining the remaining query context."""
        replaced = re.sub(rf"(?<!\w){re.escape(original)}(?!\w)", replacement,
                          text, count=1, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", replaced).strip()

    # ── Structured relation detection ────────────────────────────────────────

    @staticmethod
    def _has_structured_relation(entities: list) -> bool:
        """True when the query exposes a drug→event relation (an exposure AND
        a symptom/adverse-effect outcome). Conditions alone (e.g. "hair loss")
        do NOT count: they are the disease context, not a requested outcome."""
        has_drug = any(t in ("drug", "active_ingredient") for t, _, _ in entities or [])
        has_outcome = any(t in ("symptom", "adverse_effect") for t, _, _ in entities or [])
        return has_drug and has_outcome

    # ── Controlled query expansion ───────────────────────────────────────────

    def _expand(
        self,
        original: str,
        translated: str,
        canonical_query: str,
        lang: str,
        entities: list,
        resolutions: Optional[dict] = None,
    ) -> List[ExpandedQuery]:
        """
        Build the controlled set of expanded queries.

        Anchored to recognised entities: every expanded query is either the
        original/translated/canonical query, a provider-typed variant of a
        recognised entity (synonym / translation / colloquial / MeSH concept),
        or a combo of recognised entities. Nothing broadens beyond them.
        """
        queries: List[ExpandedQuery] = []
        canonical_names = [c for _, c, _ in entities]

        # ── A: original + translated + canonical (always present) ───────────
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
            return self._dedup_and_cap(queries, canonical_names)

        # ── B: provider-typed variants, one entity at a time ────────────────
        # Synonyms, translations, orthographic variants, colloquial/slang and
        # MeSH descriptor concepts — each preserving the rest of the query.
        # related_concept is deliberately NOT lexicalized (evidence only).
        from core.vocab.models import MATCH_TIERS
        for _etype, source_entity, _ in entities:
            resolution = (resolutions or {}).get(source_entity)
            if resolution is None:
                continue
            for match in resolution.matches:
                tier = MATCH_TIERS.get(match.match_kind)
                if tier is None:
                    continue
                if match.match_kind == "concept" and match.provider == "mesh":
                    exp_type = EXP_MESH
                elif match.match_kind in {"exact", "canonical", "preferred",
                                          "synonym", "normalized"}:
                    # identity-level variants of the entity → synonym tier
                    exp_type = EXP_SYNONYM
                else:
                    exp_type = match.match_kind
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
                        expansion_type=exp_type,
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

        # ── C: Entity combos (drug + symptom / drug + condition) ────────────
        if len(canonical_names) > 1:
            for i, (_t1, c1, _) in enumerate(entities):
                for j, (_t2, c2, _) in enumerate(entities):
                    if i < j and c1 != c2:
                        queries.append(ExpandedQuery(
                            query=f"{c1} {c2}",
                            expansion_type=EXP_COMBO,
                            source_language="en",
                            matched_entities=[c1, c2],
                        ))

        return self._dedup_and_cap(queries, canonical_names)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _dedup_and_cap(
        queries: List[ExpandedQuery], entity_names: list
    ) -> List[ExpandedQuery]:
        """
        Deduplicate by query text and cap to MAX_EXPANDED_QUERIES.

        Ordering preserves the provenance value of each expansion type:
        original/translated first, then synonym/mesh/combo (specific), then
        colloquial, then broader tiers. Within each tier, queries containing
        more entity names rank first.
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
