"""Simulate retrieval: does a natural question surface the answer, or its negation?

Builds one shared index out of every statement and every negation in the dataset
(80 documents), embeds each item's question, and finds the nearest document by
cosine similarity. This is the practical version of the earlier pairwise
similarity numbers: a real vector index containing both truths and their
contradictions, queried the way a RAG system actually gets queried.

Writes results/retrieval.csv (per item) and results/retrieval_summary.json.
"""

import csv
import json

import numpy as np
from sentence_transformers import SentenceTransformer

from metrics import load_items

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

items = load_items()

# The index: every statement and every negation in the dataset, as separate documents.
docs = []
for item in items:
    docs.append({"item_id": item["id"], "domain": item["domain"], "kind": "statement", "text": item["statement"]})
    docs.append({"item_id": item["id"], "domain": item["domain"], "kind": "negation", "text": item["negation"]})

model = SentenceTransformer(MODEL)
doc_embeddings = model.encode([doc["text"] for doc in docs], normalize_embeddings=True, batch_size=32)
question_embeddings = model.encode([item["question"] for item in items], normalize_embeddings=True, batch_size=32)

rows = []
for item, question_embedding in zip(items, question_embeddings):
    scores = doc_embeddings @ question_embedding
    top = docs[int(np.argmax(scores))]

    own_statement_idx = next(i for i, d in enumerate(docs) if d["item_id"] == item["id"] and d["kind"] == "statement")
    own_negation_idx = next(i for i, d in enumerate(docs) if d["item_id"] == item["id"] and d["kind"] == "negation")

    rows.append(
        {
            "id": item["id"],
            "domain": item["domain"],
            "question": item["question"],
            "sim_to_statement": round(float(scores[own_statement_idx]), 4),
            "sim_to_own_negation": round(float(scores[own_negation_idx]), 4),
            "top1_item_id": top["item_id"],
            "top1_kind": top["kind"],
            "top1_is_correct": top["item_id"] == item["id"] and top["kind"] == "statement",
            "top1_is_own_negation": top["item_id"] == item["id"] and top["kind"] == "negation",
        }
    )

with open("results/retrieval.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

n = len(rows)
summary = {
    "model": MODEL,
    "items": n,
    "index_size": len(docs),
    # Literal ask: within just {own statement, own negation}, which wins on similarity to the question?
    "pair_negation_wins": round(sum(r["sim_to_own_negation"] > r["sim_to_statement"] for r in rows) / n, 4),
    # Realistic ask: across the full 80-document index, what does top-1 actually return?
    "full_index_top1_correct": round(sum(r["top1_is_correct"] for r in rows) / n, 4),
    "full_index_top1_own_negation": round(sum(r["top1_is_own_negation"] for r in rows) / n, 4),
    "full_index_top1_other_item": round(
        sum(not r["top1_is_correct"] and not r["top1_is_own_negation"] for r in rows) / n, 4
    ),
}
with open("results/retrieval_summary.json", "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2)
    file.write("\n")

print(f"model: {MODEL}   items: {n}   index size: {len(docs)} documents\n")
print(f"negation wins over statement, question vs. the pair alone: {summary['pair_negation_wins']:.1%}\n")
print("full 80-document index, top-1 result:")
print(f"  correct statement retrieved:      {summary['full_index_top1_correct']:.1%}")
print(f"  own negation retrieved instead:    {summary['full_index_top1_own_negation']:.1%}")
print(f"  a different item retrieved:        {summary['full_index_top1_other_item']:.1%}")
