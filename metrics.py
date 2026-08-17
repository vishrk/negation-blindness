"""Shared scoring logic: embed the dataset with a given model and summarize it.

Used standalone by run_metrics.py (one model) and in a loop by
run_multi_model.py (several models).
"""

import json
import statistics
from pathlib import Path

from sentence_transformers import SentenceTransformer

CONDITIONS = ("negation", "paraphrase", "unrelated")
# A cutoff in the range commonly used to decide "relevant enough to retrieve".
RETRIEVAL_THRESHOLD = 0.8


def load_items():
    return json.loads(Path("data/negation_pairs.json").read_text(encoding="utf-8"))


def score_model(model_name, items):
    """Embed every item with model_name and return (rows, summary)."""
    texts = [item[field] for item in items for field in ("statement", *CONDITIONS)]
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32)

    rows = []
    for index, item in enumerate(items):
        statement, *others = embeddings[index * 4 : index * 4 + 4]
        row = {"id": item["id"], "domain": item["domain"], "negation_type": item["negation_type"]}
        # Embeddings are normalized, so the dot product is the cosine similarity.
        row.update({name: float(statement @ other) for name, other in zip(CONDITIONS, others)})
        rows.append(row)

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
        "model": model_name,
        "items": len(rows),
        "retrieval_threshold": RETRIEVAL_THRESHOLD,
        "similarity": {name: summarize([row[name] for row in rows]) for name in CONDITIONS},
        "negation_beats_paraphrase": share(lambda row: row["negation"] > row["paraphrase"]),
        "negation_above_threshold": share(lambda row: row["negation"] >= RETRIEVAL_THRESHOLD),
        "paraphrase_above_threshold": share(lambda row: row["paraphrase"] >= RETRIEVAL_THRESHOLD),
        "unrelated_above_threshold": share(lambda row: row["unrelated"] >= RETRIEVAL_THRESHOLD),
        "mean_by_domain": {
            domain: {
                name: round(statistics.mean([r[name] for r in rows if r["domain"] == domain]), 4)
                for name in CONDITIONS
            }
            for domain in sorted({row["domain"] for row in rows})
        },
    }
    return rows, summary
