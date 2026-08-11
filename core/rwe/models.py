"""
Real World Evidence (RWE) data model.

RWE items represent patient-reported experiences, community discussions, and
post-marketing adverse-event reports — explicitly distinct from scientific
evidence (PubMed / Europe PMC / ClinicalTrials.gov).

The Assistant uses ``source_type`` / ``evidence_tier`` to keep RWE separate
from scientific evidence and never presents a testimonial as clinical proof.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# Source classification — provenance + legal acquisition method.
RWE_SOURCES = {
    "reddit": {
        "source_type": "community_forum",
        "collection_method": "official_api_oauth2",
        "evidence_tier": "anecdotal",
        "language": "multi",
    },
    "openfda_faers": {
        "source_type": "pharmacovigilance",
        "collection_method": "official_api_no_key",
        "evidence_tier": "spontaneous_report",
        "language": "en",
    },
}


class RWEItem(BaseModel):
    """A single normalized RWE item — the unit exchanged with the Assistant."""

    source: str                          # reddit | openfda_faers | ...
    source_type: str                    # community_forum | pharmacovigilance | ...
    evidence_tier: str = "anecdotal"    # anecdotal | spontaneous_report | survey
    collection_method: str = "official_api"  # provenance of acquisition
    source_url: Optional[str] = None
    external_id: Optional[str] = None   # reddit permalink id / FAERS safety_report_id

    title: str = ""
    text: str = ""                      # post body / reaction description
    date: Optional[str] = None          # ISO-8601 when available
    language: str = "en"
    topic: str = ""                     # query topic that surfaced this item

    # Structured entities (best-effort, extracted from text)
    treatment: Optional[str] = None     # drug / intervention mentioned
    condition: Optional[str] = None    # condition discussed
    experience_type: str = "discussion"  # discussion | adverse_event | outcome_report

    relevance: str = "unknown"          # relevant | irrelevant | unknown
    relevance_reason: Optional[str] = None

    # Privacy / redaction status — RWE never carries direct identifiers.
    privacy_status: str = "redacted"    # redacted | anonymous
    metadata: dict = Field(default_factory=dict)


class RWESearchResult(BaseModel):
    """Envelope returned by the RWE pipeline."""

    query: str
    search_query: str                   # normalized query actually sent to collectors
    detected_language: str = "und"
    totals: dict = Field(default_factory=dict)
    items: List[RWEItem] = []
    source_status: dict = Field(default_factory=dict)  # per-source ok/unavailable/unauthorized
