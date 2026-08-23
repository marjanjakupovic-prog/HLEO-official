"""
UMLS provider — INTERFACE/STUB ONLY.

The UMLS license request has been submitted to NLM and is pending approval.
This provider is intentionally INACTIVE: it implements the VocabularyProvider
interface so the resolver can already orchestrate it, but:

- available() is True ONLY when HLEO_UMLS_API_KEY is set (UTS API key);
- until then every operation degrades to [] / None;
- no UMLS content is bundled, downloaded or scraped.

When the license is approved, the real implementation will target the UTS
REST API (https://utslogin.nlm.nih.gov + uts-ws.nlm.nih.gov/rest/) and will
not require any change in the resolver, the intent or the scorer.
"""
from __future__ import annotations

import os
from typing import List, Optional

from core.vocab.base import VocabularyProvider
from core.vocab.models import VocabularyMatch


class UMLSProvider(VocabularyProvider):
    name = "umls"
    semantic_group_hint = "general"

    def available(self) -> bool:
        return bool(os.getenv("HLEO_UMLS_API_KEY", "").strip())

    def _search(self, term: str, language: str,
                semantic_types: Optional[list],
                limit: int) -> List[VocabularyMatch]:
        # License pending — intentionally not implemented.
        return []
