"""Test two mitigations against the retrieval failures from Phase 5.

Mitigation A — cross-encoder reranker: take the bi-encoder's top-5 candidates and
re-score them with a cross-encoder, which reads the query and document together
instead of comparing two separately-computed vectors. Does joint attention over
the pair recover the negation signal that the bi-encoder threw away?

Mitigation B — negation-aware ambiguity filter: cheap and model-free. If a query's
top-2 bi-encoder hits are the statement and negation of the very same item, that's
a structural tell that the answer is contested — flag it instead of silently
returning a possibly-wrong top-1. Measured as recall (does it catch the actual
failures?) and precision (does it cry wolf on results that were already correct?).

Writes results/mitigations.csv (per item) and results/mitigations_summary.json.
"""

import csv
import json

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from metrics import build_index, load_items

BIENCODER = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_K = 5

items = load_items()
docs = build_index(items)

bi_encoder = SentenceTransformer(BIENCODER)
doc_embeddings = bi_encoder.encode([d["text"] for d in docs], normalize_embeddings=True, batch_size=32)
question_embeddings = bi_encoder.encode([item["question"] for item in items], normalize_embeddings=True, batch_size=32)

cross_encoder = CrossEncoder(CROSS_ENCODER)

rows = []
for item, question_embedding in zip(items, question_embeddings):
    bi_scores = doc_embeddings @ question_embedding
    ranked = np.argsort(-bi_scores)

    baseline_idx = int(ranked[0])
    runner_up_idx = int(ranked[1])

    top_k_idx = ranked[:RERANK_K]
    cross_scores = cross_encoder.predict([(item["question"], docs[i]["text"]) for i in top_k_idx])
    reranked_idx = int(top_k_idx[int(np.argmax(cross_scores))])

    baseline_doc = docs[baseline_idx]
    reranked_doc = docs[reranked_idx]
    runner_up_doc = docs[runner_up_idx]

    ambiguous = runner_up_doc["item_id"] == baseline_doc["item_id"] and runner_up_doc["kind"] != baseline_doc["kind"]

    rows.append(
        {
            "id": item["id"],
            "question": item["question"],
            "baseline_correct": baseline_doc["item_id"] == item["id"] and baseline_doc["kind"] == "statement",
            "baseline_own_negation": baseline_doc["item_id"] == item["id"] and baseline_doc["kind"] == "negation",
            "reranked_correct": reranked_doc["item_id"] == item["id"] and reranked_doc["kind"] == "statement",
            "reranked_own_negation": reranked_doc["item_id"] == item["id"] and reranked_doc["kind"] == "negation",
            "flagged_ambiguous": ambiguous,
        }
    )

with open("results/mitigations.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

n = len(rows)
failures = [r for r in rows if r["baseline_own_negation"]]
successes = [r for r in rows if r["baseline_correct"]]

summary = {
    "bi_encoder": BIENCODER,
    "cross_encoder": CROSS_ENCODER,
    "rerank_k": RERANK_K,
    "items": n,
    "baseline": {
        "correct": round(sum(r["baseline_correct"] for r in rows) / n, 4),
        "own_negation_retrieved": round(sum(r["baseline_own_negation"] for r in rows) / n, 4),
    },
    "cross_encoder_reranked": {
        "correct": round(sum(r["reranked_correct"] for r in rows) / n, 4),
        "own_negation_retrieved": round(sum(r["reranked_own_negation"] for r in rows) / n, 4),
        "fixed_a_baseline_failure": sum(r["reranked_correct"] for r in failures),
        "broke_a_baseline_success": sum(not r["reranked_correct"] for r in successes),
    },
    "ambiguity_filter": {
        "recall_on_baseline_failures": round(sum(r["flagged_ambiguous"] for r in failures) / len(failures), 4)
        if failures
        else None,
        "false_positives_on_baseline_successes": sum(r["flagged_ambiguous"] for r in successes),
        "total_flagged": sum(r["flagged_ambiguous"] for r in rows),
    },
}
with open("results/mitigations_summary.json", "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2)
    file.write("\n")

print(f"bi-encoder: {BIENCODER}\ncross-encoder: {CROSS_ENCODER}  (reranking top {RERANK_K})\nitems: {n}\n")
print(f"{'':<22} {'correct':>9} {'own negation':>14}")
print(f"{'baseline':<22} {summary['baseline']['correct']:>9.1%} {summary['baseline']['own_negation_retrieved']:>14.1%}")
print(
    f"{'cross-encoder rerank':<22} {summary['cross_encoder_reranked']['correct']:>9.1%} "
    f"{summary['cross_encoder_reranked']['own_negation_retrieved']:>14.1%}"
)
print(
    f"\ncross-encoder fixed {summary['cross_encoder_reranked']['fixed_a_baseline_failure']}/"
    f"{len(failures)} baseline failures, broke {summary['cross_encoder_reranked']['broke_a_baseline_success']}/"
    f"{len(successes)} baseline successes"
)

print(f"\nambiguity filter - flags {summary['ambiguity_filter']['total_flagged']}/{n} queries as contested:")
recall = summary["ambiguity_filter"]["recall_on_baseline_failures"]
print(f"  catches {recall:.1%} of the actual baseline failures" if recall is not None else "  no baseline failures to catch")
print(f"  false-flags {summary['ambiguity_filter']['false_positives_on_baseline_successes']}/{len(successes)} correct results")
