"""The one chart that tells the story. Reads results/summary.json (Phase 3 output).

Saves results/chart.png and results/chart.svg.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

# Same categorical colors used throughout the project's results log.
COLORS = {"negation": "#c4432b", "paraphrase": "#1f8a5f", "unrelated": "#5e72b8"}
LABELS = {"negation": "Negation", "paraphrase": "Paraphrase\n(correct restatement)", "unrelated": "Unrelated\n(different topic)"}

summary = json.loads(Path("results/summary.json").read_text(encoding="utf-8"))
conditions = ["negation", "paraphrase", "unrelated"]
means = [summary["similarity"][c]["mean"] for c in conditions]
threshold = summary["retrieval_threshold"]

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 13

fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=200)

bars = ax.bar(
    [LABELS[c] for c in conditions],
    means,
    width=0.55,
    color=[COLORS[c] for c in conditions],
    zorder=3,
)

for bar, value in zip(bars, means):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
        color="#171a1f",
    )

ax.axhline(threshold, color="#8890a0", linestyle="--", linewidth=1, zorder=2)
ax.text(
    2.32,
    threshold + 0.015,
    f"{threshold} retrieval threshold",
    ha="right",
    va="bottom",
    fontsize=10.5,
    color="#5b6270",
)

ax.set_ylim(0, 1.08)
ax.set_ylabel("Mean cosine similarity to the original statement")
ax.set_title(
    "Embedding models score a sentence and its negation\nas similar as a correct paraphrase",
    fontsize=16,
    fontweight="bold",
    pad=18,
)
ax.text(
    0.5,
    -0.16,
    f"all-MiniLM-L6-v2 · {summary['items']} statement/negation/paraphrase/unrelated items · negation-blindness",
    transform=ax.transAxes,
    ha="center",
    fontsize=9.5,
    color="#8890a0",
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", length=0)
ax.yaxis.grid(True, color="#e1e4e9", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig("results/chart.png", bbox_inches="tight")
fig.savefig("results/chart.svg", bbox_inches="tight")
print("wrote results/chart.png and results/chart.svg")
