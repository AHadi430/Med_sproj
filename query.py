from pipeline import run_pipeline

query = "What is the hepatitis C prevalence in Punjab Pakistan?"

out = run_pipeline(query, debug=True)

print("\n" + "=" * 60)
print("mode        :", out["mode"])
print("query type  :", out["query_type"])
print("query used  :", out["query_used"])
print("expanded    :", out["expanded"])
print("num hits    :", len(out["hits"]))
print("=" * 60)
print("\nANSWER:\n")
print(out["answer"])

# optionally print sources
print("\nSOURCES:")
for h in out["hits"][:5]:
    print(" -", h["meta"].get("url", ""))