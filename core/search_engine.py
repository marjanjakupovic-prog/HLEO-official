from aggregator import HLEOAggregator
from search.manager import SearchManager


class SearchEngine:

    def __init__(self):
        self.scientific = HLEOAggregator()
        self.web = SearchManager()

    def search(self, query: str, scientific_limit: int = 5):
        result = {
            "scientific": [],
            "web": []
        }

        try:
            result["scientific"] = self.scientific.search(
                query,
                limit=scientific_limit
            )
        except Exception as e:
            print(f"Scientific search error: {e}")

        try:
            result["web"] = self.web.search(query)
        except Exception as e:
            print(f"Web search error: {e}")

        return result