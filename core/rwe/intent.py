"""
RWE Query Intent — structured QU (Query Understanding) extraction for the
QU-aware relevance scorer (V3).

Design constraints (agreed with the project owner):
- V3 is FEATURE-FLAGGED (``HLEO_RWE_INTENT_SCORING``). When the flag is off,
  no intent is built, no LLM call is made, and ``relevance_filter`` behaves
  exactly like V1.
- AT MOST ONE LLM call per query (never per document). The LLM only
  structures the query; it never scores items.
- The LLM output is validated as strict JSON via Pydantic. On any failure
  (no API key, network error, invalid JSON, schema mismatch) the fallback is
  a deterministic intent rebuilt from the KB entities — i.e. V1 behaviour.
- ``relation_type`` is extracted for explainability only; it is NEVER used
  as a ranking criterion.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from core.biomedical_kb import (
    CONDITION_ALIASES,
    DRUG_ALIASES,
    SYMPTOM_ALIASES,
)

logger = logging.getLogger(__name__)

INTENT_SCORING_ENV = "HLEO_RWE_INTENT_SCORING"
INTENT_MODEL = "gpt-4o-mini"

_ALLOWED_RELATIONS = {"side_effect", "treatment", "outcome", "comparison", "unknown"}

# Generic "any adverse event" terms (e.g. "finasteride side effects"): the
# user asks about ANY event. Such canonicals keep their generic semantics in
# scoring — external vocabulary synonyms must NOT be injected into their
# side-set (they stay as evidence in intent.vocabulary only), otherwise the
# generic-any-event branch of the scorer would be silently disabled.
GENERIC_EVENT_TERMS = {
    "side effect", "side effects", "adverse effect", "adverse effects",
    "adverse event", "adverse events", "effetti collaterali",
    "effetto collaterale", "reaction", "reactions",
}
_MAX_TERMS_PER_SIDE = 12
_MAX_SYNONYMS = 6
_MAX_TERM_LEN = 40


def intent_scoring_enabled() -> bool:
    """Feature flag — read at call time so tests can monkeypatch the env."""
    return os.getenv(INTENT_SCORING_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


class RWEQueryIntent(BaseModel):
    """Structured representation of the query intent for V3 scoring.

    ``interventions`` = drugs/exposures, ``outcomes`` = events/symptoms the
    user asks about, ``conditions`` = diseases/conditions. ``synonyms`` maps
    a canonical term (present in one of the three lists) to extra aliases.
    ``vocabulary`` (optional, feature-flagged) maps each canonical term to
    slim external-vocabulary evidence (provider, concept_id, match_kind,
    confidence, synonyms) — evidence, never auto-synonyms.
    """
    interventions: List[str] = Field(default_factory=list)
    outcomes: List[str] = Field(default_factory=list)
    conditions: List[str] = Field(default_factory=list)
    synonyms: Dict[str, List[str]] = Field(default_factory=dict)
    relation_type: Optional[str] = None   # explainability only, never ranked
    source: str = "kb_fallback"           # "llm" | "kb_fallback"
    confidence: float = 0.0
    vocabulary: Dict[str, List[dict]] = Field(default_factory=dict)

    @field_validator("relation_type")
    @classmethod
    def _check_relation(cls, v):
        if v is None:
            return None
        v = str(v).lower().strip()
        return v if v in _ALLOWED_RELATIONS else "unknown"


def _clean_term(t) -> Optional[str]:
    if not isinstance(t, str):
        return None
    t = t.lower().strip()
    if len(t) < 3 or len(t) > _MAX_TERM_LEN:
        return None
    return t


def _clean_terms(values, cap: int = _MAX_TERMS_PER_SIDE) -> List[str]:
    out: List[str] = []
    for v in values or []:
        t = _clean_term(v)
        if t and t not in out:
            out.append(t)
        if len(out) >= cap:
            break
    return out


_INTENT_PROMPT = """You are a biomedical query-understanding component for a
hair-loss real-world-evidence search engine. Analyse the user query and
return ONLY a JSON object with these keys:

- "interventions": list of drugs / treatments / active ingredients the query
  is about (canonical generic names, e.g. "finasteride"). Empty list if none.
- "outcomes": list of events / symptoms / side effects the user asks about
  (e.g. "hair shedding", "depression", "sexual dysfunction"). Empty list if
  the query does not ask about any event.
- "conditions": list of diseases / conditions mentioned (e.g. "hair loss",
  "androgenetic alopecia"). Empty list if none.
- "synonyms": object mapping each canonical term above to a list of up to 4
  common aliases / brand names / patient phrasings (e.g. "finasteride":
  ["propecia", "proscar"]).
- "relation_type": one of "side_effect", "treatment", "outcome",
  "comparison", "unknown".

Rules:
- Do NOT invent an outcome for a query that only names a drug or only names
  a condition (e.g. "propecia" has no outcome; "hair loss" is a condition,
  not an outcome).
- A query that EXPLICITLY names an outcome concept — even a generic one —
  DOES have an outcome: "finasteride side effects" → outcomes:
  ["side effects"]; "finasteride shedding" → outcomes: ["hair shedding"].
- Keep the user's clinical intent: a colloquial phrasing like "my hair is
  falling out since I started X" means outcome "hair shedding" / "hair loss".
- Answer in English canonical terms even if the query is in another language.
- JSON only, no markdown, no commentary.

Query (English rendering): {translated}
Original user query: {original}"""


def extract_intent_llm(
    translated_query: str,
    original_query: str = "",
    client=None,
    model: str = INTENT_MODEL,
) -> Optional[RWEQueryIntent]:
    """ONE LLM call to structure the query. Returns None on any failure.

    The optional ``client`` parameter exists for testability (a fake client
    can be injected); production code creates a real OpenAI client lazily.
    """
    if not (translated_query or original_query):
        return None
    try:
        if client is None:
            if not os.getenv("OPENAI_API_KEY"):
                return None
            from openai import OpenAI
            client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": _INTENT_PROMPT.format(
                    translated=translated_query or original_query,
                    original=original_query or translated_query,
                ),
            }],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = json.loads(resp.choices[0].message.content)
        if not isinstance(raw, dict):
            return None
        intent = RWEQueryIntent(
            interventions=_clean_terms(raw.get("interventions")),
            outcomes=_clean_terms(raw.get("outcomes")),
            conditions=_clean_terms(raw.get("conditions")),
            synonyms={
                c: _clean_terms(a, _MAX_SYNONYMS)
                for c, a in (raw.get("synonyms") or {}).items()
                if _clean_term(c)
            },
            relation_type=raw.get("relation_type"),
        )
        if not (intent.interventions or intent.outcomes or intent.conditions):
            return None  # empty extraction is not a usable signal
        intent.source = "llm"
        intent.confidence = 1.0
        return intent
    except Exception as exc:  # noqa: BLE001 — any failure → deterministic fallback
        logger.info("QU intent extraction failed (%s); falling back to KB", type(exc).__name__)
        return None


def intent_from_kb(entities: list) -> RWEQueryIntent:
    """Deterministic fallback: rebuild the three intent sides from the KB
    entities of the query plan. Equivalent information to what V1 uses."""
    iv: List[str] = []
    oc: List[str] = []
    cd: List[str] = []
    for etype, canonical, _conf in entities or []:
        c = canonical.lower()
        if etype in ("drug", "active_ingredient"):
            iv.append(c)
        elif etype in ("symptom", "adverse_effect"):
            oc.append(c)
        elif etype in ("condition", "disease"):
            cd.append(c)
    return RWEQueryIntent(
        interventions=iv, outcomes=oc, conditions=cd,
        synonyms={}, relation_type=None,
        source="kb_fallback", confidence=0.5,
    )


def _slim_vocabulary(resolutions) -> Dict[str, List[dict]]:
    """Slim serialisable view of the resolver output for the intent."""
    out: Dict[str, List[dict]] = {}
    for term, res in (resolutions or {}).items():
        entries = []
        for m in res.matches:
            entries.append({
                "provider": m.provider,
                "concept_id": m.concept_id,
                "preferred_term": m.preferred_term,
                "synonyms": m.synonyms[:8],
                "semantic_group": m.semantic_group,
                "language": m.language,
                "confidence": m.confidence,
                "match_kind": m.match_kind,
            })
        if entries:
            out[term] = entries
    return out


def build_intent(
    translated_query: str,
    original_query: str,
    entities: list,
    use_llm: bool = True,
    client=None,
    resolver=None,
) -> RWEQueryIntent:
    """Build the query intent: LLM extraction (1 call) with automatic
    deterministic KB fallback on any failure. When a VocabularyResolver is
    provided (feature flag), canonical terms are resolved against external
    vocabularies and attached as typed evidence."""
    intent = None
    if use_llm:
        intent = extract_intent_llm(translated_query, original_query, client=client)
    if intent is None:
        intent = intent_from_kb(entities)
    if resolver is not None:
        try:
            canonicals = (intent.interventions + intent.outcomes
                          + intent.conditions)
            resolutions = resolver.resolve_terms(canonicals)
            intent.vocabulary = _slim_vocabulary(resolutions)
        except Exception as exc:  # noqa: BLE001 — vocabulary is optional evidence
            logger.info("vocabulary resolution skipped (%s)", type(exc).__name__)
    return intent


def merged_sides(intent: RWEQueryIntent, entities: list) -> Dict[str, set]:
    """Term sets for V3 scoring: intent sides + their synonyms (QU and KB),
    UNIONED with the KB entities of the plan (the LLM supplements the KB,
    it never replaces it). Returns {"iv", "oc", "cd"} sets of lowercase terms
    plus "tiers": a term → evidence-weight map populated ONLY from external
    vocabulary evidence (feature-flagged). Terms absent from "tiers" default
    to weight 1.0, so without vocabulary evidence the scoring is unchanged.
    """
    iv = set(_clean_terms(intent.interventions))
    oc = set(_clean_terms(intent.outcomes))
    cd = set(_clean_terms(intent.conditions))

    # External vocabulary evidence (typed, tiered; related_concept excluded
    # by scored_terms in core.vocab.models). A provider term joins the SAME
    # side as the canonical it resolves — it never jumps sides.
    tiers: Dict[str, float] = {}
    vocabulary = getattr(intent, "vocabulary", None) or {}
    if vocabulary:
        from core.vocab.models import VocabularyMatch, VocabularyResolution
        side_of = {t: iv for t in iv}
        side_of.update({t: oc for t in oc})
        side_of.update({t: cd for t in cd})
        for canonical, entries in vocabulary.items():
            bucket = side_of.get(canonical)
            if bucket is None:
                continue
            if canonical in GENERIC_EVENT_TERMS:
                continue  # generic any-event canonical: evidence only, no injection
            res = VocabularyResolution(
                term=canonical,
                matches=[VocabularyMatch(**e) for e in entries],
            )
            for term, weight in res.scored_terms().items():
                if term != canonical:
                    bucket.add(term)
                if term not in tiers or weight > tiers[term]:
                    tiers[term] = weight

    # QU synonyms, bucketed by where their canonical lives
    for canonical, aliases in (intent.synonyms or {}).items():
        c = _clean_term(canonical)
        if not c:
            continue
        bucket = iv if c in iv else oc if c in oc else cd if c in cd else None
        if bucket is not None:
            bucket.update(_clean_terms(aliases, _MAX_SYNONYMS))

    # KB entities + aliases as a supplement (fallback vocabulary)
    kb_alias_map = (
        ({"drug", "active_ingredient"}, DRUG_ALIASES, iv),
        ({"symptom", "adverse_effect"}, SYMPTOM_ALIASES, oc),
        ({"condition", "disease"}, CONDITION_ALIASES, cd),
    )
    for etype, canonical, _conf in entities or []:
        for types, alias_dict, bucket in kb_alias_map:
            if etype in types:
                bucket.add(canonical.lower())
                for alias in alias_dict.get(canonical, [])[:_MAX_SYNONYMS]:
                    t = _clean_term(alias)
                    if t:
                        bucket.add(t)

    # KB aliases for QU canonicals known to the KB
    for bucket, alias_dict in ((iv, DRUG_ALIASES), (oc, SYMPTOM_ALIASES), (cd, CONDITION_ALIASES)):
        for term in list(bucket):
            for kb_canon, aliases in alias_dict.items():
                if term == kb_canon.lower():
                    for a in aliases[:_MAX_SYNONYMS]:
                        t = _clean_term(a)
                        if t:
                            bucket.add(t)
    return {"iv": iv, "oc": oc, "cd": cd, "tiers": tiers}
