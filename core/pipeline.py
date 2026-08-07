import logging

from aggregator import HLEOAggregator
from collectors.reddit import RedditCollector
from collectors.pubmed import PubMedCollector
from collectors.europepmc import EuropePMCCollector
from collectors.clinicaltrials import ClinicalTrialsCollector
from core.extractor import LLMExtractor
from core.validator import HLEOValidator
from core.judge import HLEOJudge
from search.source_fetcher import SourceFetcher

logger = logging.getLogger(__name__)

_aggregator = HLEOAggregator()


class HLEOPipeline:

    def __init__(self):
        self.collector = RedditCollector()
        self.pubmed = PubMedCollector()
        self.europepmc = EuropePMCCollector()
        self.clinicaltrials = ClinicalTrialsCollector()
        self.extractor = LLMExtractor()
        self.validator = HLEOValidator()
        self.judge = HLEOJudge()
        self.fetcher = SourceFetcher()

    def collect(self, query: str) -> dict:
        """
        Collect raw data from all sources and deduplicate across sources.

        Flow
        ----
        1. Fetch from PubMed, Europe PMC, ClinicalTrials.gov, Reddit in parallel.
        2. Run HLEOAggregator.deduplicate_across_sources() over the three
           scientific lists (Reddit is never touched).
        3. Log Retrieved / Duplicates removed / Final unique papers.
        4. Return the cleaned per-source dict — identical structure to before,
           but with cross-source duplicates removed (winning copy only retained).
        """
        reddit_posts       = self.collector.search(query, limit=10)
        pubmed_articles    = self.pubmed.search(query, limit=20)
        europepmc_articles = self.europepmc.search(query, limit=15)
        clinical_trials    = self.clinicaltrials.search(query, limit=10)

        raw = {
            "reddit":         reddit_posts,
            "pubmed":         pubmed_articles,
            "europepmc":      europepmc_articles,
            "clinicaltrials": clinical_trials,
        }

        # ── Cross-source deduplication ────────────────────────────────────────
        deduped, stats = _aggregator.deduplicate_across_sources(raw)

        logger.info(
            "Dedup | Retrieved: %d | Duplicates removed: %d | Final unique papers: %d",
            stats["retrieved"],
            stats["removed"],
            stats["unique"],
        )

        if stats["removed"] > 0:
            for key, loser_src, winner_src in stats["duplicate_keys"]:
                logger.info(
                    "  dup removed: [%s] kept from %s, dropped from %s",
                    key, winner_src, loser_src,
                )

        # ── Clinical re-ranking ───────────────────────────────────────────────
        # Score every scientific article with clinical_rank() and sort each
        # per-source list by score descending.  Reddit is never touched.
        # This is the only mutation between dedup and returning the result dict.
        from core.ranker import rank_articles
        rank_articles(deduped)

        return deduped

    def process(self, query: str) -> list:
        """Full pipeline: collect → LLM extract → validate → judge."""
        logger.info("Pipeline avviata")

        data = self.collect(query)

        posts = data["reddit"]
        articles = data["pubmed"]
        europe_articles = data["europepmc"]
        clinical_trials = data["clinicaltrials"]

        logger.info(f"Post Reddit: {len(posts)}")
        logger.info(f"Articoli PubMed: {len(articles)}")

        if (
            not posts
            and not articles
            and not europe_articles
            and not clinical_trials
        ):
            logger.warning("Nessun dato trovato")
            return []

        results = []

        for article in articles:
            results.append({"type": "pubmed", "article": article})

        for article in europe_articles:
            results.append({"type": "europepmc", "article": article})

        for trial in clinical_trials:
            results.append({"type": "clinicaltrials", "trial": trial})

        for post in posts:
            try:
                raw_sources = self.fetcher.fetch(post.url)
                profile = self.extractor.extract(post.text)
                logger.info("Estrazione completata")

                validation = self.validator.validate(
                    profile,
                    raw_sources,
                    post.created_at,
                )
                logger.info("Validazione completata")

                judge_result = self.judge.evaluate(
                    profile.baseline_status.value,
                    profile.post_treatment_status.value,
                    validation.passed_validation,
                    profile.post_treatment_status.support_strength,
                    profile.conflict_detected,
                    profile.episode_id,
                )
                logger.info("Giudizio completato")

                results.append({
                    "type": "reddit",
                    "post": post,
                    "profile": profile,
                    "validation": validation,
                    "judge": judge_result,
                })

            except Exception as e:
                logger.exception(f"Errore elaborazione post {post.url}: {e}")

        return results
