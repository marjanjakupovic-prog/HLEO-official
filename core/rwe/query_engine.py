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
    entities: list = field(default_factory=list)        # (type, canonical, conf)
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

        # ── 4. Intent / entity recognition (KB hybrid lookup) ───────────────
        # Look up entities on BOTH the prepared/translated English text and the
        # original (Italian aliases are indexed too). Merge best confidence.
        entities = self._recognize_entities(translated, q)

        # ── 5. Controlled query expansion ───────────────────────────────────
        expanded = self._expand(
            original=q,
            translated=translated,
            prepared=prepared,
            lang=lang,
            entities=entities,
        )

        # ── 6. QU intent for the QU-aware relevance scorer (V3) ─────────────
        # Feature-flagged: when disabled, intent stays None and the pipeline
        # behaves exactly like V1 (no LLM call, no behaviour change).
        intent = None
        if intent_scoring_enabled():
            # External vocabulary evidence is a SECOND, independent flag;
            # build_resolver_from_env() returns None when HLEO_VOCAB_ENABLED
            # is off, so the intent is built exactly as before.
            from core.vocab.resolver import build_resolver_from_env
            intent = build_intent(translated, q, entities, use_llm=True,
                                  resolver=build_resolver_from_env())

        return RWEQueryPlan(
            original_query=q,
            detected_language=lang,
            translated_query=translated,
            translation_applied=translation_applied,
            entities=entities,
            expanded_queries=expanded,
            intent=intent,
        )

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
        prepared: str,
        lang: str,
        entities: list,
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
        ))
        if translated.lower() != original.lower():
            queries.append(ExpandedQuery(
                query=translated,
                expansion_type=EXP_TRANSLATED,
                source_language="en",
                matched_entities=canonical_names,
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
                            query=alias,
                            expansion_type=EXP_SYNONYM,
                            source_language="en",
                            matched_entities=[canonical],
                        ))

        # ── C: MeSH terms ──────────────────────────────────────────────────
        for etype, canonical, _ in entities:
            for mesh in get_mesh_terms(canonical):
                queries.append(ExpandedQuery(
                    query=mesh,
                    expansion_type=EXP_MESH,
                    source_language="en",
                    matched_entities=[canonical],
                ))

        # ── D: Knowledge graph neighbours (1-hop, anchored) ────────────────
        all_known = (
            set(DRUG_ALIASES) | set(CONDITION_ALIASES) | set(SYMPTOM_ALIASES)
        )
        for etype, canonical, _ in entities:
            neighbors = get_neighbors(canonical, depth=1) & all_known
            for neighbor in neighbors:
                # combo: primary entity + neighbour (keeps scope tight)
                queries.append(ExpandedQuery(
                    query=f"{canonical} {neighbor}",
                    expansion_type=EXP_NEIGHBOR,
                    source_language="en",
                    matched_entities=[canonical, neighbor],
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
            EXP_ORIGINAL: 0, EXP_TRANSLATED: 1,
            EXP_SYNONYM: 2, EXP_MESH: 2, EXP_COMBO: 2,
            EXP_COLLOQUIAL: 3, EXP_NEIGHBOR: 4, EXP_LLM: 5,
        }
        ent_lower = {n.lower() for n in entity_names}

        def _score(eq: ExpandedQuery) -> tuple:
            ent_hits = sum(1 for n in ent_lower if n in _norm(eq.query))
            return (tier_order.get(eq.expansion_type, 9), -ent_hits)

        unique.sort(key=_score)
        return unique[:MAX_EXPANDED_QUERIES] if len(unique) > MAX_EXPANDED_QUERIES else unique
