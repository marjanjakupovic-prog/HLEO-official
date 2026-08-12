"""
HLEO — Relational Search (Level 2)
==================================
Transforms the AI-extracted clinical *relation* into precise per-source
retrieval, a tolerant hard filter, and a batched LLM relational judge that
re-ranks candidates so articles discussing the REQUESTED RELATIONSHIP surface
on top.

Pipeline
--------
    user query
      → (1) ClinicalRelation extraction        [1 LLM call, cached]
      → (2) per-source structured query build   [no LLM]
      → (3) retrieval via existing collectors   [PubMed/EuropePMC/ClinicalTrials]
      → (4) hard filter (agent + manifestation co-occurrence, synonym-tolerant)
      → (5) clinical_rank pre-sort              [deterministic, no LLM]
      → (6) LLM relational judge on top-N pool  [batched, ~2 calls/source]
      → (7) final re-rank by judge score        [no LLM]

Design notes
------------
- NO hardcoded clinical combinations: the LLM generates agent/manifestation
  search_terms, relation_type and relation_phrases per query.
- Fallback: if OPENAI_API_KEY is missing, or relation extraction / judge fail
  (incl. 429), the caller falls back to the existing keyword pipeline. This
  module never raises for those conditions — it returns None / degraded
  rankings so /search keeps working.
- Collectors are reused unchanged: PubMed accepts [tiab]/hasabstract syntax
  in `term`; EuropePMC accepts TITLE:/ABSTRACT: field syntax in `query`.
- Output shape is identical to core.pipeline.collect() (dict of SearchResult
  lists) so /search can swap it in with no frontend changes. Each article's
  `score` is set to the judge-driven combined score so the existing frontend
  sort (by data.score desc) immediately benefits.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from collectors.pubmed import PubMedCollector
from collectors.europepmc import EuropePMCCollector
from collectors.clinicaltrials import ClinicalTrialsCollector
from core.search_result import SearchResult

logger = logging.getLogger(__name__)

MODEL = "gpt-4o-mini"
JUDGE_BATCH = 5          # articles per judge LLM call
JUDGE_POOL_PER_SOURCE = 10  # top-N candidates judged per source


# ── Clinical relation model ──────────────────────────────────────────────────

@dataclass
class ClinicalRelation:
    """Structured clinical relationship extracted from the user query."""
    original_query: str
    agent: dict = field(default_factory=dict)           # {term,normalized,role,identified,search_terms}
    event: dict = field(default_factory=dict)           # {term,normalized}
    manifestation: dict = field(default_factory=dict)   # {term,normalized,role,search_terms}
    temporal: str = ""
    relation_type: str = "unknown"
    scientific_query: str = ""
    relation_phrases: list = field(default_factory=list)
    fallback_needed: bool = False

    def to_dict(self) -> dict:
        return {
            "original_query": self.original_query,
            "agent": self.agent,
            "event": self.event,
            "manifestation": self.manifestation,
            "temporal": self.temporal,
            "relation_type": self.relation_type,
            "scientific_query": self.scientific_query,
            "relation_phrases": self.relation_phrases,
            "fallback_needed": self.fallback_needed,
        }


# ── Prompt: relation extraction (refined, validated in /tmp prototypes) ──────

_RELATION_PROMPT = """You are a biomedical clinical-NLP system for a scientific literature search engine.

Interpret the CLINICAL RELATIONSHIP the user is asking about, so the engine can retrieve
articles that discuss THAT RELATIONSHIP (not mere co-mention of words).

The agent is NOT assumed to be a drug: it can be a drug, chemical, physical exposure, activity,
procedure, device, or generic/unspecified.

Return ONLY a JSON object (no markdown, no commentary) with this schema:
{
  "query_original": str,
  "agent": {"term": str, "normalized": str, "role": str, "identified": bool, "search_terms": [str]},
  "event": {"term": str, "normalized": str},
  "manifestation": {"term": str, "normalized": str, "role": str, "search_terms": [str]},
  "temporal": str,
  "relation_type": str,
  "scientific_query": str,
  "relation_phrases": [str],
  "fallback_needed": bool
}

relation_type STRICT ENUM (choose one):
- adverse_effect   : a DRUG/CHEMICAL/PROCEDURE may CAUSE/TRIGGER the manifestation (harm after/with the agent).
- efficacy         : the agent is used TO TREAT/FOR a condition, or asks if effective/regrowth.
- drug_condition   : agent studied vs condition, neutral.
- exposure_outcome : a NON-PHARMACOLOGICAL exposure/activity -> outcome (running->knee pain, sun->erythema). NOT adverse_effect.
- association      : a statistical/epidemiological link, no causal direction.
- diagnostic       : agent used to diagnose.
- prevention       : agent used to prevent a condition.
- unknown          : relationship unclear.

ROUTING (apply in order):
1. activity/physical_exposure/device -> outcome = exposure_outcome (NOT adverse_effect).
2. "to treat/for/effective/regrowth" -> efficacy.
3. drug/chemical/procedure followed by a harmful manifestation the user suspects caused -> adverse_effect.
Never default all negatives to adverse_effect.

search_terms: provide 2-5 English synonyms/variants for SEARCHING (INN for drugs; scientific
terms for manifestations, e.g. erythema, "skin irritation", "cutaneous irritation", dermatitis).
These are used to match articles by synonym, so include clinically equivalent terms.

scientific_query: REQUIRE the agent AND manifestation to co-occur, joined with AND. For
adverse_effect append a causal group using ONLY: "adverse effect","side effect",induced,
"caused by","triggered by","secondary to","drug-related","treatment-related",worsening.
DO NOT use bare "effect" alone, and DO NOT use "following" or "associated with" as mandatory
terms (they are too generic). For efficacy: (treatment OR efficacy OR therapeutic OR "response to").
For exposure_outcome: (following OR "due to" OR injury OR caused). English, parentheses + OR groups.

relation_phrases: 3-6 diverse natural scientific phrases expressing THIS specific relation
(vary grammar: induced / associated with / following / secondary to / adverse reaction to).

VAGUE-AGENT HANDLING: if the agent is not specifically named (e.g. "un nuovo farmaco"):
agent.identified=false, agent.role="generic_unspecified", agent.normalized="" (do NOT invent a
name), fallback_needed=true.

Normalize: drugs->INN; Italian lay terms -> scientific English (rossore->erythema/"skin irritation";
caduta dei capelli->"hair shedding"/"hair loss"; dolore articolare->arthralgia; mal di testa->headache;
ginocchio->knee; esposizione al sole->"sun exposure"; eritema->erythema).

Query: __QUERY__"""


# ── Prompt: relational LLM judge (batched) ───────────────────────────────────

_JUDGE_PROMPT = """You are a strict biomedical relevance judge.

The user is asking about this CLINICAL RELATION:
  agent: {agent} ({arole})
  event: {event}
  manifestation: {manifest} ({mrole})
  temporal: {temporal}
  relation_type: {rtype}
  relation description: {desc}

For EACH article below, decide whether it substantively discusses THIS RELATIONSHIP
(the agent causing/being associated with the manifestation as an adverse effect / exposure
outcome, or the agent treating the condition if efficacy), versus merely mentioning the
words separately or discussing a different topic.

Assign:
  label: "relevant" | "partial" | "not_relevant"
  score: 0.0-1.0  (1.0 = the relation is a central focus; 0.5 = both terms present but
                   relation not the focus; 0.0 = no real relation)
  reason: one short sentence

Return ONLY JSON: {{"results":[{{"i":int,"label":str,"score":float,"reason":str}}]}}

Articles:
{arts}"""


def _describe(rel: ClinicalRelation) -> str:
    a = rel.agent.get("normalized", "")
    m = rel.manifestation.get("normalized", "")
    ev = rel.event.get("normalized", "")
    rt = rel.relation_type
    if rt == "adverse_effect":
        return f"{a} (via {ev}) may CAUSE/TRIGGER {m} as an adverse effect"
    if rt == "efficacy":
        return f"{a} used to TREAT {m}"
    if rt == "exposure_outcome":
        return f"{a} (via {ev}) leads to / is associated with {m}"
    return f"{a} -> {m}"


# ── RelationalSearch ─────────────────────────────────────────────────────────

class RelationalSearch:
    """
    Relational retrieval + LLM judge re-ranking.

    Usage
    -----
        rs = RelationalSearch()
        out = rs.search("rossore dopo utilizzo di minoxidil")
        # out -> {"pubmed":[SearchResult], "europepmc":[...], "clinicaltrials":[...],
        #         "reddit":[], "relation": ClinicalRelation, "stats": {...}}
        # out is None when relational mode is unavailable (no key / extraction failed)
        # so the caller falls back to the existing keyword pipeline.
    """

    # Process-lifetime cache for relation extraction (identical query → same relation)
    _rel_cache: dict[str, ClinicalRelation] = {}

    def __init__(self) -> None:
        self._client = None
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=api_key)
            except Exception as exc:
                logger.warning("RelationalSearch: OpenAI init failed — %s", exc)
        self.pubmed = PubMedCollector()
        self.europepmc = EuropePMCCollector()
        self.clinicaltrials = ClinicalTrialsCollector()

    # ── Public API ───────────────────────────────────────────────────────────

    def search(self, raw_query: str) -> Optional[dict]:
        """
        Run the full relational pipeline. Returns a dict with the same four
        article lists as pipeline.collect() plus `relation` and `stats`.
        Returns None when relational mode cannot run (no key / extraction
        failure) so the caller falls back gracefully.
        """
        if self._client is None:
            return None

        t0 = time.perf_counter()
        stats: dict[str, Any] = {"openai_calls": 0, "judge_errors": [], "judge_used": True}

        # 1) relation extraction
        rel = self._extract_relation(raw_query)
        stats["openai_calls"] += 1
        if rel is None:
            return None  # fall back to keyword pipeline

        # 2-3) per-source retrieval
        pm_q = self._build_pubmed_query(rel)
        ep_q = self._build_epmc_query(rel)
        ct_q = self._build_ct_query(rel)

        pm = self.pubmed.search(pm_q, limit=20)
        ep = self.europepmc.search(ep_q, limit=20)
        ct = self.clinicaltrials.search(ct_q, limit=10)

        stats["candidates"] = {
            "pubmed": len(pm), "europepmc": len(ep), "clinicaltrials": len(ct),
        }

        # 4) hard filter
        pm = self._hard_filter(pm, rel)
        ep = self._hard_filter(ep, rel)
        ct = self._hard_filter(ct, rel)
        stats["after_hard_filter"] = {
            "pubmed": len(pm), "europepmc": len(ep), "clinicaltrials": len(ct),
        }

        # 5) clinical_rank pre-sort
        from core.ranker import clinical_rank
        for lst in (pm, ep, ct):
            for art in lst:
                art.score = clinical_rank(art)
            lst.sort(key=lambda a: a.score, reverse=True)

        # 6) LLM judge on top-N pool
        judged = self._judge_and_rank(pm, rel, stats)
        pm = judged
        ep = self._judge_and_rank(ep, rel, stats)
        ct = self._judge_and_rank(ct, rel, stats)

        # reddit untouched (relational search focuses on scientific literature)
        reddit: list = []

        stats["final"] = {"pubmed": len(pm), "europepmc": len(ep), "clinicaltrials": len(ct)}
        stats["elapsed_s"] = round(time.perf_counter() - t0, 2)

        return {
            "pubmed": pm, "europepmc": ep, "clinicaltrials": ct, "reddit": reddit,
            "relation": rel, "stats": stats,
        }

    # ── (1) Relation extraction ──────────────────────────────────────────────

    def _extract_relation(self, query: str) -> Optional[ClinicalRelation]:
        import hashlib
        ck = hashlib.md5(query.lower().strip().encode()).hexdigest()
        if ck in self._rel_cache:
            return self._rel_cache[ck]
        try:
            data = self._llm_json(_RELATION_PROMPT.replace("__QUERY__", query), max_tokens=700)
        except Exception as exc:
            logger.warning("RelationalSearch: relation extraction failed — %s", exc)
            return None
        rel = ClinicalRelation(
            original_query=data.get("query_original", query),
            agent=data.get("agent", {}) or {},
            event=data.get("event", {}) or {},
            manifestation=data.get("manifestation", {}) or {},
            temporal=data.get("temporal", ""),
            relation_type=data.get("relation_type", "unknown"),
            scientific_query=data.get("scientific_query", ""),
            relation_phrases=data.get("relation_phrases", []) or [],
            fallback_needed=bool(data.get("fallback_needed", False)),
        )
        self._rel_cache[ck] = rel
        return rel

    # ── (2) Per-source query builders ────────────────────────────────────────

    @staticmethod
    def _or_group(terms: list[str]) -> str:
        terms = [t for t in terms if t]
        if not terms:
            return ""
        if len(terms) == 1:
            return terms[0]
        return "(" + " OR ".join(terms) + ")"

    # Causal groups — tightened: no bare "effect", no "following"/"associated with" mandatory
    _CAUSAL = {
        "adverse_effect": '("adverse effect" OR "side effect" OR induced OR "caused by" OR "triggered by" OR "secondary to" OR "drug-related" OR "treatment-related" OR worsening)',
        "efficacy": '(treatment OR efficacy OR therapeutic OR "response to")',
        "exposure_outcome": '(following OR "due to" OR injury OR caused)',
        "drug_condition": "",
        "association": '("associated with" OR correlation OR linked)',
        "unknown": "",
        "diagnostic": "",
        "prevention": '(prevention OR preventive OR prophylaxis)',
    }

    def _build_pubmed_query(self, rel: ClinicalRelation) -> str:
        ag = self._or_group(rel.agent.get("search_terms") or [rel.agent.get("normalized", "")])
        mn = self._or_group(rel.manifestation.get("search_terms") or [rel.manifestation.get("normalized", "")])
        causal = self._CAUSAL.get(rel.relation_type, "")
        # [tiab] restricts to title/abstract; hasabstract ensures an abstract exists.
        parts = []
        if ag:
            parts.append(f"{ag}[tiab]")
        if mn:
            parts.append(f"{mn}[tiab]")
        if causal:
            parts.append(causal)
        parts.append("hasabstract")
        return " AND ".join(parts)

    def _build_epmc_query(self, rel: ClinicalRelation) -> str:
        ag = self._or_group(rel.agent.get("search_terms") or [rel.agent.get("normalized", "")])
        mn = self._or_group(rel.manifestation.get("search_terms") or [rel.manifestation.get("normalized", "")])
        causal = self._CAUSAL.get(rel.relation_type, "")
        # Force the agent into TITLE or ABSTRACT (measured ~100% agent presence),
        # require manifestation, optional causal group.
        parts = []
        if ag:
            parts.append(f"(TITLE:{ag} OR ABSTRACT:{ag})")
        if mn:
            parts.append(mn)
        if causal:
            parts.append(causal)
        return " AND ".join(parts)

    def _build_ct_query(self, rel: ClinicalRelation) -> str:
        # ClinicalTrials v2 query.term is free-text; use agent + manifestation.
        ag = rel.agent.get("normalized", "")
        mn = rel.manifestation.get("normalized", "")
        return f"{ag} {mn}".strip() or rel.original_query

    # ── (4) Hard filter (synonym-tolerant) ────────────────────────────────────

    def _hard_filter(self, items: list[SearchResult], rel: ClinicalRelation) -> list[SearchResult]:
        agent_terms = [t.lower() for t in (rel.agent.get("search_terms") or []) if t]
        if rel.agent.get("normalized"):
            agent_terms.append(rel.agent["normalized"].lower())
        mani_terms = [t.lower() for t in (rel.manifestation.get("search_terms") or []) if t]
        if rel.manifestation.get("normalized"):
            mani_terms.append(rel.manifestation["normalized"].lower())

        kept = []
        for it in items:
            blob = ((getattr(it, "title", "") or "") + " " + (getattr(it, "abstract", "") or "")).lower()
            has_agent = any(t and t in blob for t in agent_terms)
            has_mani = any(t and t in blob for t in mani_terms)
            # If we have no terms to match on, keep the item (don't over-filter).
            if not agent_terms and not mani_terms:
                kept.append(it)
            elif has_agent and has_mani:
                kept.append(it)
            elif has_agent and not mani_terms:
                kept.append(it)
            elif has_mani and not agent_terms:
                kept.append(it)
            # else: drop — clearly incompatible (neither agent nor manifestation present)
        return kept

    # ── (6) LLM judge + (7) re-rank ──────────────────────────────────────────

    def _judge_and_rank(self, items: list[SearchResult], rel: ClinicalRelation, stats: dict) -> list[SearchResult]:
        from core.ranker import clinical_rank
        if not items:
            return items
        pool = items[:JUDGE_POOL_PER_SOURCE]
        tail = items[JUDGE_POOL_PER_SOURCE:]

        judgements = self._judge_batched(pool, rel, stats)

        # Combine: judge score dominates; clinical_rank breaks ties.
        # Combined score = score*1000 + clinical_rank so judge ordering wins.
        for art, j in zip(pool, judgements):
            jscore = float(j.get("score", 0.0))
            combined = jscore * 1000.0 + clinical_rank(art)
            art.score = round(combined, 2)
            art.metadata = dict(art.metadata or {})
            art.metadata["relevance_label"] = j.get("label", "not_relevant")
            art.metadata["relevance_score"] = jscore
            art.metadata["relevance_reason"] = j.get("reason", "")

        pool.sort(key=lambda a: a.score, reverse=True)
        # tail keeps its clinical_rank score (already set), ranked after pool
        return pool + tail

    def _judge_batched(self, pool: list[SearchResult], rel: ClinicalRelation, stats: dict) -> list[dict]:
        if not pool or self._client is None:
            stats["judge_used"] = False
            return [{"label": "partial", "score": 0.5, "reason": "judge unavailable"} for _ in pool]
        out: list[dict] = []
        for i in range(0, len(pool), JUDGE_BATCH):
            batch = pool[i:i + JUDGE_BATCH]
            try:
                res = self._llm_judge(batch, rel)
                stats["openai_calls"] += 1
                # align by index 'i' in the returned JSON
                idx_map = {r.get("i"): r for r in res}
                for j, _art in enumerate(batch):
                    out.append(idx_map.get(j, {"label": "partial", "score": 0.5, "reason": "missing"}))
            except Exception as exc:
                stats["judge_used"] = False
                stats["judge_errors"].append(str(exc))
                logger.warning("RelationalSearch: judge batch failed — %s", exc)
                for _art in batch:
                    out.append({"label": "partial", "score": 0.5, "reason": f"judge error: {exc}"})
            time.sleep(1.0)  # be gentle with rate limits between batches
        return out

    def _llm_judge(self, batch: list[SearchResult], rel: ClinicalRelation) -> list[dict]:
        arts_txt = "\n".join(
            f"[{i}] TITLE: {getattr(a,'title','')}\n    ABSTRACT: {(getattr(a,'abstract','') or '')[:700]}"
            for i, a in enumerate(batch)
        )
        prompt = _JUDGE_PROMPT.format(
            agent=rel.agent.get("normalized", ""), arole=rel.agent.get("role", ""),
            event=rel.event.get("normalized", ""), manifest=rel.manifestation.get("normalized", ""),
            mrole=rel.manifestation.get("role", ""), temporal=rel.temporal,
            rtype=rel.relation_type, desc=_describe(rel), arts=arts_txt,
        )
        data = self._llm_json(prompt, max_tokens=900)
        return data.get("results", []) or []

    # ── LLM helper (delegates retry to the central llm_guard) ─────────────────
    #
    # The guard is the ONLY retry boundary: MAX_TOTAL_ATTEMPTS=5, no nested
    # retry. quota exhaustion raises QuotaExhaustedError (no retry); transient
    # 429s and JSON/parse errors retry up to the cap. This method must NOT add
    # its own retry loop on top — that would breach the absolute cap.

    def _llm_json(self, prompt: str, max_tokens: int = 700) -> dict:
        from core.llm_guard import call_llm_json
        return call_llm_json(
            self._client,
            messages=[{"role": "user", "content": prompt}],
            model=MODEL,
            temperature=0,
            max_tokens=max_tokens,
            operation="relational_search_llm",
        )
