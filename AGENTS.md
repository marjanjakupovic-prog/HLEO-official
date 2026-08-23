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
  `reddit_adapter.py` (wraps `RedditCollector` → RWEItem),
  `xenforo_base.py` (shared `XenForoRSSCollector` base: fetch/parse/aggregate
  logic for any XenForo-RSS community), four XenForo/phpBB community collectors
  (see "Community source collectors" below), `pipeline.py` (RWEPipeline:
  collect → dedup → relevance_filter → provenance).
- **RWE_SOURCES registry**: `reddit` (community_forum, OAuth2 API),
  `openfda_faers` (pharmacovigilance, official_api_no_key),
  `calvizie` (community_forum, official_rss_feed, language=it),
  `hairlosstalk` (community_forum, official_rss_feed, language=en),
  `hairlossexperiences` (community_forum, official_rss_feed, language=en),
  `maladiesrares` (community_forum, official_atom_feed, language=fr).
  Default sources in `RWEPipeline.search()`:
  `["reddit","openfda_faers","calvizie","hairlosstalk","hairlossexperiences","maladiesrares"]`.
  **When writing pipeline tests, pass `sources=[...]` explicitly** — the default
  now triggers live community-feed fetches (calvizie/hairlosstalk/etc.) if not
  mocked/restricted.
- **Community source collectors** (all read-only, no API key/OAuth, no anti-bot):
  - `calvizie_collector.py` — Calvizie.net (IT, XenForo RSS, 18 non-surgical
    hair-loss sub-forums; transplant/cosmetic forums excluded).
  - `hairlosstalk_collector.py` — HairLossTalk.com (EN, XenForo RSS, 11
    non-surgical sub-forums: antiandrogens, minoxidil, alopecia areata/totalis/
    universalis, side effects, shedding, success stories, alternative
    treatments, men's/women's treatment tracks; transplant/concealer/wig/cosmetic
    forums excluded; scope audited 2026-08 against live forums index).
  - `hairlossexperiences_collector.py` — HairLossExperiences.com (EN, XenForo
    RSS, 2 non-surgical RWE sub-forums: hair-loss-medications + general-hair-loss;
    site is transplant-heavy overall so per-section inclusion is essential —
    female-hair-loss/FAQ/products forums excluded after live audit showed
    transplant-surgery/cosmetic content).
  - `maladiesrares_collector.py` — MaladiesRaresInfo.org (FR, phpBB Atom feed,
    `pelade universelle`/alopecia universalis sub-forum f173; small-volume but
    the only FR-language alopecia-areata RWE source).
  - `xenforo_base.py` — `XenForoRSSCollector` shared base; subclasses set
    `source`, `base_url`, `language`, `forum_slugs`. Status codes: ok /
    no_results / rate_limited / network_error.
- **Collector pattern**: each collector exposes `search_with_status(query, limit)`
  → `(items, status, reason)`. Status codes: ok / no_results / rate_limited /
  network_error (reddit also: no_credentials / auth_error).
- **Endpoint** `GET /rwe/search?q=&limit=&sources=` in `api/main.py` — fully
  separate from scientific `/search`; never touches PubMed/EuropePMC/CT.gov.
  `sources` accepts comma-separated: `reddit,openfda_faers,calvizie,hairlosstalk,hairlossexperiences,maladiesrares`.
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
- **Sources usable**: openFDA/FAERS (public, no key), Reddit (PRAW OAuth),
  Calvizie.net/HairLossTalk/HairLossExperiences (XenForo official RSS, no key),
  MaladiesRaresInfo (phpBB official Atom, no key). PatientsLikeMe/Carenity/
  Inspire/AskAPatient = NOT usable (no API / scrape ban). DEFERRED (anti-bot/
  down, not bypassed): SalusMaster (Cloudflare 403), HairRestorationNetwork
  (Cloudflare 403), BaldTruthTalk (503), International-HairLossForum (Cloudflare
  403), Alopezie.de (unreachable, exit 56), BelliCapelli (ForumFree UA-block,
  transplant-focused). haarerkrankungen.de software/RSS pending verification.
- **Tests**: `tests/test_rwe.py` (14 tests), `tests/test_rwe_search.py`
  (search-engine), `tests/test_rwe_calvizie.py` (19 Calvizie tests),
  `tests/test_rwe_xenforo_collectors.py` (17 HairLossTalk/HairLossExperiences
  tests), `tests/test_rwe_maladiesrares.py` (18 MaladiesRares tests).
  `tests/test_search_result.py` removed (was a broken non-test snippet since
  initial commit, blocked pytest collection).

## RWE Search Engine (Phase: autonomous retrieval) — DONE
- **Goal**: transform the RWE search from "query → collector → keyword match"
  into an autonomous pipeline:
  User Query → language detect → prepare → translate → controlled expansion
  → semi-semantic retrieval → RWE collectors → normalize → dedup → relevance
  → provenance → RWE evidence → AI Assistant.
- **New module** `core/rwe/query_engine.py`: `RWEQueryEngine` builds a
  `RWEQueryPlan` (`original_query` NEVER overwritten, `detected_language`,
  `translated_query`, `translation_applied`, `entities`, `expanded_queries`).
  Reuses `QueryOrchestrator` (lang detect + translate) and `biomedical_kb`
  (DRUG/CONDITION/SYMPTOM_ALIASES, MESH_MAP, KNOWLEDGE_GRAPH,
  lookup_entity/get_neighbors/get_mesh_terms/quick_translate_it). NO duplicated
  synonym/translation logic.
- **Language detection**: `detect_language()` (fast stopword/marker heuristic,
  IT/EN/DE/FR/ES) + orchestrator LLM detection when key present. Multi-language
  = source-language detection + English translation (collectors are EN-oriented;
  orchestrator only translates →English, which is the honestly-supported path).
- **Controlled expansion**: original + translated + KB synonyms (≤3/entity) +
  MeSH + graph neighbours (1-hop, anchored) + entity combos (drug+symptom) +
  trichology colloquial supplement (patient phrasings: "initial shedding",
  "increased hair fall", "temporary worsening", "caduta iniziale", etc.,
  mapped to recognised symptom concepts). Capped at 16 queries; every expanded
  query stays anchored to recognised entities (a finasteride query never
  broadens into generic hair-loss chatter). KB-first/deterministic; LLM
  expansion optional + bounded.
- **Semi-semantic relevance** (`relevance_filter` in `pipeline.py`): score =
  token overlap (50%) + entity/synonym overlap (50%). Authoritative sources
  (openFDA) trusted on their own server-side match → high score floor even when
  the query term is absent from the item's text fields (fixes the
  finasteride/shedding openFDA case). Each item gets `relevance_score` (0–1),
  `match_reason` (authoritative|exact_keyword|semantic_entity|...).
- **Provenance**: every `RWEItem` carries `matched_query`, `matched_query_type`
  (original|translated|synonym|mesh|combo|colloquial|neighbor), `source_language`,
  `relevance_score`, `match_reason`. `RWESearchResult` exposes `original_query`,
  `translated_query`, `detected_language`, `translation_applied`,
  `expanded_queries` (full transparent expansion list).
- **AI Assistant**: `RWEItemCtx` extended with the provenance fields; the RWE
  prompt block now includes `matched_query`, `type`, `lang`, `match`, `score`
  per item so the Assistant can cite origin precisely.
- **Frontend**: `/rwe/search` response mapped to `rwe_evidence` with provenance
  fields; search cards show `q:` (matched_query) + `rel` (relevance_score).
- **Tests**: `tests/test_rwe_search.py` (37 tests: language detection IT/EN/
  DE/FR/ES, original_query preservation, translation, expansion types, synonyms,
  MeSH, colloquial shedding, controlled-anchored expansion, cap, dedup,
  medical vs colloquial terms, brand→generic, trichology/non-trichology,
  authoritative source match, semantic entity match, off-topic filter,
  backward-compat relevance_filter, score range, provenance stamping,
  source_language, translated/expanded-query results, no duplicates across
  queries, RWE-only, scientific regression, endpoint provenance, assistant
  schema, canonical Italian query full pipeline). Full suite: 60 passed.


## Dependencies (2026-08-12 audit, refreshed 2026-08-10)
All deps in requirements.txt are at latest published versions and used in
code. Verified pip index versions for each: fastapi 0.141.1, uvicorn 0.52.1,
sqlalchemy 2.0.52, psycopg2-binary 2.9.12, pydantic 2.13.4, bcrypt 5.0.0,
openai 3.0.0, python-dotenv 1.2.2, pytest 9.1.1, jinja2 3.1.6, requests 2.34.2,
beautifulsoup4 4.15.0, praw 8.0.3, ddgs 9.14.4, httpx2 2.10.0.

- 2026-08-10 refresh: bcrypt 4.2.0 → 5.0.0 (major; API unchanged —
  hashpw/checkpw/gensalt all backward compatible, verified with admin Basic
  Auth flow end-to-end); praw 8.0.2 → 8.0.3 (patch). All 165 tests pass,
  pip-audit clean. install_hleo.py embedded requirements synced too.
- openai 2.54.0 -> 3.0.0: v3.0.0 makes HTTPX2 the default HTTP client (httpx
  no longer auto-installed). Codebase pins httpx2==2.10.0 and uses only
  `from openai import OpenAI` + `client.chat.completions.create(**kwargs)`
  (unchanged API), so the upgrade was a drop-in.
- No unused deps to prune: every requirements.txt entry is used directly
  (fastapi, sqlalchemy, pydantic, bcrypt, openai, dotenv, requests, bs4, ddgs,
  praw) or at runtime/CLI (uvicorn server, jinja2 via FastAPI Jinja2Templates,
  psycopg2 via SQLAlchemy driver string, httpx2 as openai transport,
  pytest for the test suite).
- Orphaned legacy code (NOT removed): app.py + ui/ (PySide6 Qt desktop
  UI) is from the initial commit, unused by tests/Dockerfile/run_pipeline, and
  depends on PySide6 which is NOT in requirements.txt and NOT installed. It is
  superseded by the FastAPI web app (api/main.py + templates/index.html).
  Left in place; flagged for user decision on whether to delete.

## Admin authentication — token-based login (2026-08-10)
- **Architecture**: HTTP Basic Auth (bcrypt-hashed password from env) is still
  supported for backward compatibility (API clients, curl), AND a stateless
  Bearer-token login flow was added for the web UI. No sessions, no users table,
  no hardcoded secrets. The admin identity is read from environment variables:
  - `HLEO_ADMIN_USERNAME` (default: none → admin disabled)
  - `HLEO_ADMIN_PASSWORD_HASH` (bcrypt hash; plaintext NEVER stored)
  - `HLEO_ADMIN_TOKEN_TTL` (optional, seconds; default 28800 = 8h)
- **Login flow (frontend → backend)**:
  1. `GET /admin/ping` (public, no auth) → `{admin_enabled: bool}`. Frontend
     uses this to show/hide the Admin nav tab.
  2. User clicks Admin tab → `loadAdminSources()` checks sessionStorage for a
     token. If none, shows the login form (username + password inputs).
  3. `POST /admin/login` `{username, password}` → validates against the same
     bcrypt hash, returns `{token, expires_at, username}`. 401 on wrong creds.
  4. Frontend stores the token in `sessionStorage` (cleared when tab closes;
     NEVER localStorage or cookies). All subsequent `/admin/*` fetches include
     `Authorization: Bearer <token>`.
  5. `POST /admin/logout` (requires valid token) → no-op server-side (stateless
     token), client clears sessionStorage.
- **Token**: HMAC-SHA256 signed payload `{u: username, exp: expiry}`. The HMAC
  key is the bcrypt password hash itself (server-side only, never exposed to
  clients). Changing the admin password invalidates all outstanding tokens.
  Tampered/expired tokens → 401.
- **`require_admin`** dependency accepts EITHER Bearer token (web UI) OR Basic
  Auth (backward compat). If admin not configured → 404 (section hidden). If no
  valid creds → 401 with `WWW-Authenticate`.
- **Security**: no credentials hardcoded in client code (verified by test).
  Password field cleared immediately after login. Token does not contain the
  password or hash (verified by test). Token in sessionStorage (not persisted to
  disk). In production, deploy behind HTTPS.
- **Files**: `core/admin_auth.py` (token create/verify, require_admin),
  `api/admin.py` (`/admin/login`, `/admin/logout`, source CRUD routes),
  `templates/index.html` (login form, token management JS).

## Browser smoke-test note (2026-08-15)
- Playwright smoke test `e2e_smoke.py` now passes on local SQLite/uvicorn.
- Important frontend detail: `runCompare()` merges `clinical_profile_episode_ids` from the scientific snapshot (`_searchStates.scientific.activeSearchCtx`) and `rwe_profile_episode_ids` from the RWE snapshot/current context.
- The active context after switching to RWE may no longer contain the scientific IDs, but the compare payload still includes both via snapshot merge.
- For reliable smoke tests, force `setSearchMode('scientific')` for the scientific search, `setSearchMode('rwe')` for the RWE search, and route `/pipeline/run` with `max_results=1` to keep the test fast.

- **Tests**: `tests/test_admin_auth.py` (24 tests covering all required
  scenarios: unauthenticated 401, wrong creds 401, correct creds 200 + token,
  bearer-token auth 200, admin tab visibility, no hardcoded creds, Basic Auth
  backward compat, tampered/expired token 401).

## Admin tab (frontend) — fixed 2026-08-10
- **Bug**: Admin nav tab appeared but clicking it showed the search page
  (page-admin never became visible). Root cause: the admin page `<div>` had
  `id="adminPage"` while `showPage(name)` toggles `hidden` on
  `document.getElementById(\`page-${name}\`)`. All other pages use the
  `page-<name>` convention. Fixed by renaming `id="adminPage"` →
  `id="page-admin"` in templates/index.html.
- Admin enabled via env `HLEO_ADMIN_USERNAME` + `HLEO_ADMIN_PASSWORD_HASH`
  (bcrypt; generate with `python -c "from core.admin_auth import hash_password; print(hash_password('your-password'))" `).

## Vocabulary Provider Layer (2026-08-23) — feature-flagged, OFF by default
- `core/vocab/`: external structured-vocabulary layer feeding V3 intent as
  TYPED EVIDENCE (never auto-synonyms). Flag `HLEO_VOCAB_ENABLED` (default off
  → pipeline byte-identical to plain V1/V3). `HLEO_VOCAB_PROVIDERS` default
  `rxnorm,mesh,conceptnet,wikidata`.
- Providers (all `VocabularyProvider` in `base.py`, never raise, timeout 10s,
  silent skip when unavailable): `rxnorm.py` (NLM RxNav REST, no key; drug
  canonicalisation brand↔generic — NOTE: RxNav rejects %2B, tty list must
  keep literal `+`, same pitfall as openFDA), `mesh.py` (NLM MeSH RDF REST:
  lookup/descriptor + `{UI}.json`; descriptor uses `label`, Term records use
  `prefLabel`; SPARQL endpoint does not answer label queries — do NOT use),
  `loinc.py` (FHIR fhir.loinc.org, env-only creds `HLEO_LOINC_USERNAME/
  PASSWORD`, absent → available()=False), `conceptnet.py` (general-purpose
  multilingual, CC BY-SA; API was 502-down on 2026-08-23 — fallback design
  absorbs it), `wikidata.py` (general-purpose aliases, CC0; FP-prone on
  abbreviations, confidence-capped). `umls.py` + `snomed.py` = INTERFACE
  STUBS ONLY, inactive until license (`HLEO_UMLS_API_KEY`, `HLEO_SNOMED_*`).
- `models.py`: `VocabularyMatch` (provider, concept_id, preferred_term,
  synonyms, semantic_group, language, confidence, match_kind, source_url,
  metadata) + `VocabularyResolution`. match_kind ∈ exact/canonical/preferred/
  synonym/translation/abbreviation/orthographic_variant/colloquial/slang/
  normalized/concept/related_concept; `MATCH_TIERS` weights; related_concept
  = evidence only, NEVER scored.
- `cache.py`: `VocabCache` TTL (env `HLEO_VOCAB_CACHE_TTL` default 24h),
  per-provider namespace, sha256 keys (NO credentials), optional JSON file
  via `HLEO_VOCAB_CACHE_PATH` (default memory-only), `HLEO_VOCAB_CACHE_DISABLE=1`.
  GOTCHA FIXED: `cache or VocabCache()` silently discards shared caches
  because empty VocabCache is falsy (`__len__`) — always use
  `cache if cache is not None else ...`. Resolver shares a process-wide
  default cache.
- Integration: `RWEQueryIntent.vocabulary` (slim dict); `build_intent(...,
  resolver=...)` wired in query_engine only when BOTH flags on;
  `merged_sides` returns extra `tiers` map (term→weight) and injects
  provider synonyms into the SAME side as their canonical; `_score_item_v3`
  weights hits by tier (absent=1.0 → identical scores without vocab).
  GENERIC_EVENT_TERMS moved to core/rwe/intent.py — vocabulary synonyms are
  NEVER injected into generic any-event side-sets ("side effects") or the
  generic scoring branch breaks (benchmark regression found+fixed 2026-08-23).
- Tests: `tests/test_vocab.py` (32 tests, all offline via fake _get_json).
  Full suite 249 passed.
- Benchmark: `benchmarks/vocab_ab_benchmark.py` (3 arms V1/V3/V3+vocab, 33
  queries incl. IT colloquial, abbreviations minox/dut/fin/rogaine, slang).
  Result: v3vocab ≡ v3 on kept sets (no harm); big gains are QU-level
  (minox→minoxidil canonicalisation, R 0.22→0.97 P=1.0). Vocabulary probe +
  cache stats in `vocab_ab_report.json`. Known unresolved: "dut" (no provider
  maps it, QU doesn't canonicalise), ConceptNet down at benchmark time.
