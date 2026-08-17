"""The reranker chart. Reads results/mitigations_summary.json (Phase 6 output).

One grouped-bar chart carries both stories at once: the baseline bars already
show the blindness (high "correct", low "own negation retrieved"), and the
reranked bars show the fix converging both toward a coin flip instead of
closing the gap.

Saves results/mitigations_chart.png and results/mitigations_chart.svg.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Same categorical colors used throughout the project's results log.
COLORS = {"baseline": "#5e72b8", "reranked": "#c4432b"}
LABELS = {"baseline": "Baseline (bi-encoder)", "reranked": "Cross-encoder rerank"}
GROUPS = ["correct", "own_negation_retrieved"]
GROUP_LABELS = ["Correct retrieval", "Own negation\nretrieved instead"]

summary = json.loads(Path("results/mitigations_summary.json").read_text(encoding="utf-8"))
baseline = [summary["baseline"][g] for g in GROUPS]
reranked = [summary["cross_encoder_reranked"][g] for g in GROUPS]

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 13

fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=200)

x = np.arange(len(GROUPS))
width = 0.32

bars_b = ax.bar(x - width / 2, baseline, width, color=COLORS["baseline"], zorder=3, label=LABELS["baseline"])
bars_r = ax.bar(x + width / 2, reranked, width, color=COLORS["reranked"], zorder=3, label=LABELS["reranked"])

for bars in (bars_b, bars_r):
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color="#171a1f",
        )

ax.set_xticks(x)
ax.set_xticklabels(GROUP_LABELS)
ax.set_ylim(0, 1.08)
ax.set_ylabel("Share of 40 queries")
ax.set_title(
    "Cross-encoder reranking cuts accuracy from 92.5% to 50%\n"
    "fixes 1 of 3 failures, breaks 18 of 37 successes",
    fontsize=15,
    fontweight="bold",
    pad=18,
)
ax.text(
    0.5,
    -0.44,
    "all-MiniLM-L6-v2 -> cross-encoder/ms-marco-MiniLM-L-6-v2 (top-5 rerank) · negation-blindness",
    transform=ax.transAxes,
    ha="center",
    fontsize=9.5,
    color="#8890a0",
)

ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=10.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", length=0, pad=10)
ax.yaxis.grid(True, color="#e1e4e9", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

fig.tight_layout()
fig.subplots_adjust(bottom=0.28)
fig.savefig("results/mitigations_chart.png", bbox_inches="tight")
fig.savefig("results/mitigations_chart.svg", bbox_inches="tight")
print("wrote results/mitigations_chart.png and results/mitigations_chart.svg")
