"""Run the negation/paraphrase/unrelated scoring across several embedding models.

Answers: does a bigger or newer embedding model actually fix negation blindness?

Writes results/model_comparison.csv and results/model_comparison.json.
Edit MODELS to add or remove models from the sweep.
"""

import csv
import json

from metrics import load_items, score_model

MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-base-en-v1.5",
    "intfloat/e5-base-v2",
    "thenlper/gte-base",
]

items = load_items()
comparisons = []
for model_name in MODELS:
    print(f"scoring {model_name} ...")
    _, summary = score_model(model_name, items)
    comparisons.append(
        {
            "model": model_name,
            "negation_mean": summary["similarity"]["negation"]["mean"],
            "paraphrase_mean": summary["similarity"]["paraphrase"]["mean"],
            "unrelated_mean": summary["similarity"]["unrelated"]["mean"],
            "negation_beats_paraphrase": summary["negation_beats_paraphrase"],
            "negation_above_threshold": summary["negation_above_threshold"],
        }
    )

with open("results/model_comparison.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(comparisons[0]))
    writer.writeheader()
    writer.writerows(comparisons)
with open("results/model_comparison.json", "w", encoding="utf-8") as file:
    json.dump(comparisons, file, indent=2)
    file.write("\n")

print(f"\n{'model':<38} {'negation':>9} {'paraphrase':>11} {'unrelated':>10} {'neg>para':>9} {'neg>=0.8':>9}")
for row in comparisons:
    print(
        f"{row['model']:<38} {row['negation_mean']:>9.4f} {row['paraphrase_mean']:>11.4f} "
        f"{row['unrelated_mean']:>10.4f} {row['negation_beats_paraphrase']:>9.1%} "
        f"{row['negation_above_threshold']:>9.1%}"
    )
