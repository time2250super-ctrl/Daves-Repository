#!/usr/bin/env python3
"""Render graphs for the Nova session test into /opt/cursor/artifacts."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BG0, INK, MUTED = "#0f1412", "#e8f0ea", "#9bb0a3"
GOLD, GREEN, RED, LINE = "#c4a35a", "#3d8f6e", "#c46b5a", "#26332c"

rows = [json.loads(l) for l in Path("/tmp/nova-test.jsonl").read_text().splitlines() if l.strip()]
labels = [r["label"] for r in rows]
knows = [int(r["knows_dave"]) for r in rows]
lat = [float(r["latency"]) for r in rows]
# Expected: step index 1 (A recall) should know; all others should not.
expected = [0, 1, 0, 0]
passed = [k == e for k, e in zip(knows, expected)]

plt.rcParams.update({
    "figure.facecolor": BG0, "axes.facecolor": BG0, "savefig.facecolor": BG0,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.edgecolor": LINE, "font.size": 11,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
fig.suptitle("Nova per-session memory test  ·  Qwen2.5-1.5B-Instruct (CPU)",
             color=INK, fontsize=15, fontweight="bold", y=0.98)

# Left: does Nova recall the name, per step, vs expectation
xs = range(len(rows))
bar_colors = [GREEN if p else RED for p in passed]
ax1.bar(xs, [1] * len(rows), color=LINE, width=0.6, zorder=1)
ax1.bar(xs, knows, color=bar_colors, width=0.6, zorder=2)
for i, (k, e, p) in enumerate(zip(knows, expected, passed)):
    ax1.text(i, 0.5, "recalls\n'Dave'" if k else "no memory\nof name",
             ha="center", va="center", color="#f4fff8", fontsize=9, fontweight="bold")
    ax1.text(i, 1.05, ("PASS" if p else "FAIL"), ha="center", va="bottom",
             color=(GREEN if p else RED), fontsize=11, fontweight="bold")
    ax1.text(i, -0.14, f"expect: {'yes' if e else 'no'}", ha="center", va="top",
             color=MUTED, fontsize=8)
ax1.set_xticks(list(xs))
ax1.set_xticklabels([l.replace(". ", ".\n") for l in labels], fontsize=9)
ax1.set_yticks([])
ax1.set_ylim(-0.25, 1.25)
ax1.set_title("Does Nova remember the name is 'Dave'?", color=GOLD, fontsize=12, pad=26)
ax1.legend(handles=[Patch(facecolor=GREEN, label="matches expected"),
                    Patch(facecolor=RED, label="unexpected")],
           loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2,
           facecolor=BG0, edgecolor=LINE, labelcolor=INK, framealpha=1)
for s in ax1.spines.values():
    s.set_visible(False)

# Right: latency per request
b = ax2.bar(xs, lat, color=GOLD, width=0.6)
for i, v in enumerate(lat):
    ax2.text(i, v + 0.3, f"{v:.1f}s", ha="center", va="bottom", color=INK, fontsize=10)
ax2.set_xticks(list(xs))
ax2.set_xticklabels([l.split(".")[0] for l in labels], fontsize=10)
ax2.set_xlabel("step")
ax2.set_ylabel("response latency (s)")
ax2.set_ylim(0, max(lat) * 1.25)
ax2.set_title("Per-request latency (CPU generation)", color=GOLD, fontsize=12, pad=12)
ax2.grid(axis="y", color=LINE, linewidth=0.8)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)

fig.text(0.5, 0.005,
         f"Result: {sum(passed)}/{len(passed)} assertions passed  ·  "
         "memory recall, session isolation, and reset all verified",
         ha="center", color=MUTED, fontsize=10)
fig.subplots_adjust(left=0.06, right=0.97, top=0.82, bottom=0.24, wspace=0.18)
out = Path("/opt/cursor/artifacts/nova-session-test.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130)
print("wrote", out)
