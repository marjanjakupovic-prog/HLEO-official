import requests

from core.search_result import SearchResult


class EuropePMCCollector:
    BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    def search(self, query: str, limit: int = 5):
        r = requests.get(
            self.BASE_URL,
            params={"query": query, "format": "json", "pageSize": limit,
                    "resultType": "core"},   # "core" returns abstractText
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()

        results = []
        for item in data.get("resultList", {}).get("result", []):
            abstract = item.get("abstractText") or item.get("abstract", "")
            # Extract authors list (authorList → author[].fullName or .lastName + .firstName)
            author_list = item.get("authorList", {}).get("author", [])
            authors = []
            for a in author_list:
                name = a.get("fullName") or (
                    " ".join(filter(None, [a.get("firstName", ""), a.get("lastName", "")]))
                )
                if name:
                    authors.append(name)
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    source="Europe PMC",
                    abstract=abstract,
                    authors=authors,
                    year=int(item["pubYear"]) if item.get("pubYear") else None,
                    doi=item.get("doi"),
                    metadata={
                        "journal": item.get("journalTitle", ""),
                        "id": item.get("id"),
                    },
                )
            )
        return results
