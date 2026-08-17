"""Score the full dataset with one model and report aggregate statistics.

For each item we measure the cosine similarity between the statement and three
comparison texts: its negation, a meaning-preserving paraphrase (positive
control) and an in-domain unrelated sentence (negative control).

Writes results/similarities.csv (per item) and results/summary.json (aggregates).
"""

import csv
import json
import statistics
from pathlib import Path

from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CONDITIONS = ("negation", "paraphrase", "unrelated")
# A cutoff in the range commonly used to decide "relevant enough to retrieve".
RETRIEVAL_THRESHOLD = 0.8

items = json.loads(Path("data/negation_pairs.json").read_text(encoding="utf-8"))

# One batch, so every text is embedded under identical conditions.
texts = [item[field] for item in items for field in ("statement", *CONDITIONS)]
model = SentenceTransformer(MODEL)
embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32)

rows = []
for index, item in enumerate(items):
    statement, *others = embeddings[index * 4 : index * 4 + 4]
    row = {"id": item["id"], "domain": item["domain"], "negation_type": item["negation_type"]}
    # Embeddings are normalized, so the dot product is the cosine similarity.
    row.update({name: float(statement @ other) for name, other in zip(CONDITIONS, others)})
    rows.append(row)

Path("results").mkdir(exist_ok=True)
with open("results/similarities.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def summarize(values):
    return {
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def share(predicate):
    return round(sum(predicate(row) for row in rows) / len(rows), 4)


summary = {
    "model": MODEL,
    "items": len(rows),
    "retrieval_threshold": RETRIEVAL_THRESHOLD,
    "similarity": {name: summarize([row[name] for row in rows]) for name in CONDITIONS},
    "negation_beats_paraphrase": share(lambda row: row["negation"] > row["paraphrase"]),
    "negation_above_threshold": share(lambda row: row["negation"] >= RETRIEVAL_THRESHOLD),
    "paraphrase_above_threshold": share(lambda row: row["paraphrase"] >= RETRIEVAL_THRESHOLD),
    "unrelated_above_threshold": share(lambda row: row["unrelated"] >= RETRIEVAL_THRESHOLD),
    "mean_by_domain": {
        domain: {
            name: round(
                statistics.mean([r[name] for r in rows if r["domain"] == domain]), 4
            )
            for name in CONDITIONS
        }
        for domain in sorted({row["domain"] for row in rows})
    },
}
Path("results/summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

print(f"model: {MODEL}   items: {len(rows)}\n")
print(f"{'condition':<12} {'mean':>7} {'median':>7} {'min':>7} {'max':>7}")
for name in CONDITIONS:
    stats = summary["similarity"][name]
    print(f"{name:<12} {stats['mean']:>7.4f} {stats['median']:>7.4f} {stats['min']:>7.4f} {stats['max']:>7.4f}")

print(f"\nnegation scores higher than the paraphrase:  {summary['negation_beats_paraphrase']:.1%}")
print(f"negation at or above {RETRIEVAL_THRESHOLD}:                  {summary['negation_above_threshold']:.1%}")
print(f"paraphrase at or above {RETRIEVAL_THRESHOLD}:                {summary['paraphrase_above_threshold']:.1%}")
print(f"unrelated at or above {RETRIEVAL_THRESHOLD}:                 {summary['unrelated_above_threshold']:.1%}")

print("\nmean similarity by domain:")
print(f"{'domain':<12} {'negation':>9} {'paraphrase':>11} {'unrelated':>10}")
for domain, means in summary["mean_by_domain"].items():
    print(f"{domain:<12} {means['negation']:>9.4f} {means['paraphrase']:>11.4f} {means['unrelated']:>10.4f}")
