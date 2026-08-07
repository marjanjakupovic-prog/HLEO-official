import requests
import time

results.append(
    SearchResult(
        title=article.get("title", ""),
        source="PubMed",
        authors=[
            author.get("name", "")
            for author in article.get("authors", [])
        ],
        pmid=pmid,
        metadata={
            "journal": article.get("fulljournalname", ""),
            "pubdate": article.get("pubdate", ""),
        },
    )
)