"""Sanity checks for data/negation_pairs.json. Run after editing the dataset."""

import json
from collections import Counter
from pathlib import Path

FIELDS = ("id", "domain", "negation_type", "statement", "negation", "paraphrase", "unrelated", "question")

items = json.loads(Path("data/negation_pairs.json").read_text(encoding="utf-8"))

ids = set()
for item in items:
    assert set(item) == set(FIELDS), f"{item.get('id')}: fields are {sorted(item)}"
    assert all(item[f].strip() for f in FIELDS), f"{item['id']}: has an empty field"
    assert item["id"] not in ids, f"duplicate id {item['id']}"
    ids.add(item["id"])
    # The four texts must all differ, or the controls measure nothing.
    texts = {item["statement"], item["negation"], item["paraphrase"], item["unrelated"]}
    assert len(texts) == 4, f"{item['id']}: duplicate text across fields"

print(f"{len(items)} items, all well-formed")
for label, field in (("domain", "domain"), ("negation type", "negation_type")):
    counts = Counter(item[field] for item in items)
    print(f"\nby {label}:")
    for key, count in counts.most_common():
        print(f"  {key:<15} {count}")
