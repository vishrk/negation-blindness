"""Score the full dataset with one model and report aggregate statistics.

For each item we measure the cosine similarity between the statement and three
comparison texts: its negation, a meaning-preserving paraphrase (positive
control) and an in-domain unrelated sentence (negative control).

Writes results/similarities.csv (per item) and results/summary.json (aggregates).
"""

import csv
import json
from pathlib import Path

from metrics import CONDITIONS, RETRIEVAL_THRESHOLD, load_items, score_model

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

items = load_items()
rows, summary = score_model(MODEL, items)

Path("results").mkdir(exist_ok=True)
with open("results/similarities.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
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
