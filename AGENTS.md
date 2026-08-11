# HLEO — Agent Memory

## Project overview
HLEO (Hair Loss Evidence Organizer) — FastAPI app: scientific literature search +
extraction + AI Clinical Assistant (RAG). Python 3.13.

## Key architecture facts (for RWE phase)
- **Scientific search (Level 3 / dual search)** lives in `api/main.py`:
  - `GET /search?q=&mode=scientific|global` — scientific = relational (Level 2, LLM,
    returns 503 without OPENAI_API_KEY, no silent fallback); global = keyword pipeline.
  - `POST /pipeline/run?q=&mode=...` — extract ClinicalProfiles → DB.
  - `POST /synthesis` — Level 3 synthesis (reuses Level 2 relevant articles).
- **Scientific sources**: PubMed, Europe PMC, ClinicalTrials.gov (`collectors/*.py`).
  Reddit is collected in `HLEOPipeline.collect()` but ONLY used for experiences,
  never enters scientific dedup/ranking (`aggregator.py` excludes reddit).
- **Result model**: `core/search_result.py` `SearchResult` dataclass.
- **DB models** (`core/models.py`): `ClinicalProfile`, `PatientExperience`,
  `RawSource`, `SourceAttribution`, `ChatSession`, `ChatMessage`, plus a full
  partner/source/license governance set (`PartnerRegistry`, `SourceRegistry`,
  `DataLicensing`, `PartnerAuthorization`, `EvidenceComparison`).
- **AI Assistant** (`POST /assistant/chat`): RAG over DB; receives `SearchContext`
  (articles) from frontend; when search articles present they are the ONLY primary
  source (DB RAG skipped). System prompt has strict SOURCE PRIORITY.
- **Existing RWE scaffolding**:
  - `collectors/base.py` `RawTestimonial` dataclass (source, url, title, text, author, created_at).
  - `collectors/reddit.py` `RedditCollector` (PRAW OAuth, status codes: ok/
    no_credentials/auth_error/rate_limited/no_results/network_error).
  - `core/patient_schema.py` `PatientExperienceProfile` (Pydantic).
  - `core/patient_extractor.py` `PatientExperienceExtractor` (LLM, gpt-4o).
  - `POST /experiences/ingest?q=` (Reddit → LLM extract → `PatientExperience` DB rows).
  - `GET /experiences?limit=` lists saved experiences.
  - `api/partners.py` router prefix `/rwe` — partner/source/license/comparison CRM
    (already references "community vs scientific" via `EvidenceComparison`).
- **Orchestrator** (`core/orchestrator.py`): `QueryOrchestrator.process(q)` →
  `OrchestrationResult` (detect lang, translate to scientific English).

## Where to put RWE pipeline (without contaminating scientific)
- New `core/rwe/` package: `models.py` (RWE item model), `pipeline.py`
  (collect→normalize→dedup→relevance→provenance), `collectors/` modular.
- New endpoint `GET /rwe/search?q=` (separate from `/search`).
- Reuse `RawTestimonial` + `PatientExperienceProfile` + `PatientExperience` DB row.
- Extend `SearchContext`/assistant prompt with optional `rwe_evidence` list,
  keep `articles` (scientific) untouched → no regression.

## Conventions
- Pydantic v2, SQLAlchemy 2.0 (declarative_base in `core.database`).
- Tests in `tests/`, conftest neutralizes `load_dotenv` + in-memory SQLite.
- No commit/push without explicit user authorization.

## RWE implementation (Phase 4–9) — DONE
- `core/rwe/` package: `models.py` (RWEItem/RWESearchResult Pydantic, RWE_SOURCES
  registry), `openfda_collector.py` (openFDA FAERS, official API no key),
  `reddit_adapter.py` (wraps `RedditCollector` → RWEItem), `pipeline.py`
  (RWEPipeline: collect → dedup → relevance_filter → provenance).
- **Endpoint** `GET /rwe/search?q=&limit=&sources=` in `api/main.py` — fully
  separate from scientific `/search`; never touches PubMed/EuropePMC/CT.gov.
- **AI Assistant convergence**: `SearchContext.rwe_evidence: List[RWEItemCtx]`
  (optional). System prompt has 4-tier SOURCE PRIORITY (scientific > RWE > DB >
  general). RWE block built separately; rules force "Testimonianze e discussioni"
  + "Confronto" sections when both present; RWE-only mode explicitly labels
  patient-reported experiences as non-clinical.
- **Frontend**: third search tab "RWE" in `templates/index.html` (EN/IT i18n),
  calls `/rwe/search`, renders RWE items with amber "RWE" badge + evidence_tier
  metadata; pipeline/synthesis buttons hidden in RWE mode; context badge shows
  RWE item count when no scientific articles.
- **openFDA query bug (fixed)**: openFDA returns 404 for `+OR+` syntax (requests
  encodes `+`→`%2B`); use spaces + `OR` instead. Test `test_openfda_query_syntax_live`
  guards against regression.
- **Sources usable**: openFDA/FAERS (public, no key), Reddit (PRAW OAuth).
  PatientsLikeMe/Carenity/Inspire/AskAPatient = NOT usable (no API / scrape ban).
- **Tests**: `tests/test_rwe.py` (14 tests: pipeline, dedup, relevance,
  provenance, separation, assistant schema, scientific regression, live openFDA).
  `tests/test_search_result.py` removed (was a broken non-test snippet since
  initial commit, blocked pytest collection).
