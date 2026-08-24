#!/usr/bin/env python3
"""Render the competitive latency chart from bake-off results.

Input: JSON array written by scripts/competitive-benchmark.py
Output: docs/benchmarks/competitive-latency.png

Usage:
  python3 scripts/render-competitive-chart.py \
      --results .build/bakeoff/results.json --out docs/benchmarks/competitive-latency.png
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = [
    "MetalANNS",
    "numpy (exact scan)",
    "FAISS IndexFlatIP",
    "FAISS HNSW",
    "hnswlib",
    "USearch",
    "sqlite-vec",
]

DISPLAY = {
    "MetalANNS": "MetalANNS (exact)",
    "numpy (exact scan)": "NumPy scan (exact)",
    "FAISS IndexFlatIP": "FAISS FlatIP (exact)",
    "FAISS HNSW": "FAISS HNSW (@0.99 rec.)",
    "hnswlib": "hnswlib (@0.99 rec.)",
    "USearch": "USearch (@0.99 rec.)",
    "sqlite-vec": "sqlite-vec (exact)",
}

ACCENT = "#0066cc"
NEUTRAL = "#9ba3ad"
APPROX = "#c98a2d"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=".build/bakeoff/results.json")
    parser.add_argument(
        "--out", default="docs/benchmarks/competitive-latency.png"
    )
    args = parser.parse_args()

    rows = json.loads(Path(args.results).read_text())
    by_scale = defaultdict(dict)
    for r in rows:
        by_scale[r["scale"]][r["backend"]] = r

    scales = sorted(by_scale)
    fig, axes = plt.subplots(
        1, len(scales), figsize=(4.6 * len(scales), 4.6), sharey=False
    )

    for ax, n in zip(axes, scales):
        entries = by_scale[n]
        backends = [b for b in ORDER if b in entries]
        values = [entries[b]["p50_us"] / 1000.0 for b in backends]
        colors = [
            ACCENT if b == "MetalANNS"
            else APPROX if entries[b]["kind"] == "approx"
            else NEUTRAL
            for b in backends
        ]

        y = range(len(backends))
        ax.barh(y, values, color=colors, height=0.62)
        ax.set_yticks(list(y))
        ax.set_yticklabels([DISPLAY[b] for b in backends], fontsize=9)
        for tick, b in zip(ax.get_yticklabels(), backends):
            if b == "MetalANNS":
                tick.set_fontweight("bold")
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlabel("query p50 latency, ms (log)", fontsize=9)
        ax.set_title(f"{n:,} vectors", fontsize=11)
        ax.grid(axis="x", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        best_other = min(values[1:])
        speedup = best_other / values[0]
        ax.text(
            0.5,
            0.97,
            f"{speedup:.1f}\u00d7 faster than the fastest alternative",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.5,
            color=ACCENT,
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2),
        )

        xmax = max(values)
        for yi, (b, v) in enumerate(zip(backends, values)):
            label = f"{v:.2f}"
            if v < 0.1:
                label = f"{v * 1000:.0f} µs"
            elif v >= 10:
                label = f"{v:.0f}"
            ax.text(
                v * 1.12,
                yi,
                label,
                va="center",
                fontsize=8,
                color="#333333",
            )
        ax.set_xlim(0.03, xmax * 3.2)

    fig.suptitle(
        "Single-query vector search latency - Apple M3 Max, dim 384, cosine, k=10",
        fontsize=12,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor="white")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
