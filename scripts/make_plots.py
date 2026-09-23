"""Turn the benchmark JSONs into publication-quality figures (PNG + PDF).

Produces
--------
* ``figures/fig_index_benchmark.{png,pdf}`` — index lookup latency, the
  latency / memory trade-off, and the learned-index threshold sweep.
* ``figures/fig_buffer_pool.{png,pdf}`` — hit ratio vs. pool capacity for
  each replacement policy (with the Belady optimum as reference).

Requires ``scripts/run_benchmarks.py`` and ``scripts/bench_buffer.py`` to
have been run first.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
FIGURES = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures"))

DPI = 300
COLORS = {
    "sorted_array": "#4C72B0",
    "btree": "#DD8452",
    "learned_e16": "#55A868",
    "lru": "#4C72B0",
    "clock": "#DD8452",
    "random": "#55A868",
    "opt": "#C44E52",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "legend.frameon": False,
})


def _save(fig, name: str) -> None:
    os.makedirs(FIGURES, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(FIGURES, f"{name}.{ext}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"  saved {path}")
    plt.close(fig)


def fig_index_benchmark(payload: dict) -> None:
    variants = payload["variants"]
    names = [v["name"] for v in variants]
    x = range(len(variants))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # (a) lookup latency
    ax = axes[0]
    present = [v["present_ns"] for v in variants]
    absent = [v["absent_ns"] for v in variants]
    width = 0.38
    ax.bar([i - width / 2 for i in x], present, width, label="key present",
           color=[COLORS[n] for n in names], alpha=0.85)
    ax.bar([i + width / 2 for i in x], absent, width, label="key absent",
           color=[COLORS[n] for n in names], alpha=0.45)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylabel("lookup latency (ns / query)")
    ax.set_title("Point-lookup latency")
    ax.legend(loc="upper left")
    for i, v in enumerate(variants):
        ax.annotate(f"{v['lookup_ns']:.0f}", (i, v["lookup_ns"]),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)

    # (b) build time vs memory (log-log)
    ax = axes[1]
    for v in variants:
        ax.scatter(v["memory_bytes"], v["build_s"], s=90, color=COLORS[v["name"]],
                   label=v["name"], edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate(v["name"], (v["memory_bytes"], v["build_s"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("index memory (bytes)")
    ax.set_ylabel("build time (s)")
    ax.set_title("Build time vs. memory")

    # (c) threshold sweep
    ax = axes[2]
    sweep = payload["threshold_sweep"]
    ths = [int(t) for t in sweep]
    ns = [sweep[t]["lookup_ns"] for t in sweep]
    pieces = [sweep[t]["pieces"] for t in sweep]
    ax.plot(ths, ns, "o-", color=COLORS["learned_e16"], label="lookup ns")
    ax.set_xlabel("error bound E")
    ax.set_ylabel("lookup latency (ns / query)")
    ax.set_title("Learned index: E vs. latency")
    ax.set_xlim(min(ths) - 4, max(ths) + 4)
    ax2 = ax.twinx()
    ax2.plot(ths, pieces, "s--", color="#8172B3", label="linear pieces")
    ax2.set_ylabel("number of linear pieces", color="#8172B3")
    ax2.tick_params(axis="y", labelcolor="#8172B3")
    for t, n in zip(ths, ns):
        ax.annotate(f"{n:.0f}", (t, n), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8)
    for t, p in zip(ths, pieces):
        ax2.annotate(f"{p}", (t, p), textcoords="offset points",
                     xytext=(0, -12), ha="center", fontsize=7, color="#8172B3")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="upper right")

    fig.suptitle("Index structures: sorted array vs. B+ tree vs. learned index (RMI)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "fig_index_benchmark")


def fig_buffer_pool(payload: dict) -> None:
    hit = payload["hit_ratio"]
    caps = [int(c) for c in hit["lru"]]
    pcts = [c / payload["meta"]["n_pages"] * 100 for c in caps]

    fig, ax = plt.subplots(figsize=(8, 5))
    for policy in ["lru", "clock", "random", "opt"]:
        ys = [hit[policy][str(c)] for c in caps]
        style = dict(color=COLORS[policy], marker="o", markersize=5, linewidth=1.8)
        if policy == "opt":
            style.update(linestyle="--", linewidth=2.0, zorder=4)
        ax.plot(pcts, ys, label=policy.upper(), **style)
    ax.set_xlabel("buffer pool capacity (% of page set)")
    ax.set_ylabel("hit ratio")
    ax.set_title("Buffer-pool replacement policies under a Zipf workload")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", title="policy")
    fig.tight_layout()
    _save(fig, "fig_buffer_pool")


def main() -> int:
    idx_path = os.path.join(RESULTS, "index_benchmark.json")
    buf_path = os.path.join(RESULTS, "buffer_pool_benchmark.json")
    missing = [p for p in (idx_path, buf_path) if not os.path.exists(p)]
    if missing:
        print("missing benchmark results; run the benchmark scripts first:")
        for p in missing:
            print(f"  - {p}")
        return 1

    print("fig_index_benchmark:")
    with open(idx_path, encoding="utf-8") as fh:
        fig_index_benchmark(json.load(fh))
    print("fig_buffer_pool:")
    with open(buf_path, encoding="utf-8") as fh:
        fig_buffer_pool(json.load(fh))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
