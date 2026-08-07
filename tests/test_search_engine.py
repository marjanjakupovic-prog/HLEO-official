from core.search_engine import SearchEngine

engine = SearchEngine()

results = engine.search("finasteride")

print("\n=== SCIENTIFIC ===")
print(len(results["scientific"]))

print("\n=== WEB ===")
print(len(results["web"]))