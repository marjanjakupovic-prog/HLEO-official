import requests
import time

from core.search_result import SearchResult


class PubMedCollector:
    SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def search(self, query: str, limit: int = 10):
        # 1 — get IDs
        r = requests.get(
            self.SEARCH_URL,
            params={"db": "pubmed", "term": query, "retmax": limit, "retmode": "json"},
            timeout=15,
        )
        r.raise_for_status()
        ids = r.json()["esearchresult"]["idlist"]
        if not ids:
            return []

        time.sleep(0.4)

        # 2 — summary (title, authors, journal)
        r2 = requests.get(
            self.SUMMARY_URL,
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
            timeout=15,
        )
        r2.raise_for_status()
        details = r2.json()

        time.sleep(0.4)

        # 3 — fetch abstracts as plain text, one call for all IDs
        abstract_map: dict[str, str] = {}
        try:
            r3 = requests.get(
                self.FETCH_URL,
                params={
                    "db": "pubmed",
                    "id": ",".join(ids),
                    "rettype": "abstract",
                    "retmode": "text",
                },
                timeout=20,
            )
            if r3.status_code == 200:
                # Split on numbered entries like "\n\n1. " or "\n\n2. "
                blocks = r3.text.split("\n\n\n")
                for i, pmid in enumerate(ids):
                    if i < len(blocks):
                        abstract_map[pmid] = blocks[i].strip()
        except Exception:
            pass

        results = []
        for pmid in ids:
            art = details["result"].get(pmid, {})
            results.append(
                SearchResult(
                    title=art.get("title", ""),
                    source="PubMed",
                    authors=[a.get("name", "") for a in art.get("authors", [])],
                    pmid=pmid,
                    abstract=abstract_map.get(pmid, ""),
                    metadata={
                        "journal": art.get("fulljournalname", ""),
                        "pubdate": art.get("pubdate", ""),
                    },
                )
            )
        return results
