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
    "learned_block": "#8172B3",
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


def fig_index_distribution(payload: dict) -> None:
    """Distribution sensitivity: latency / model pieces / prediction error per
    key distribution."""
    dists = [d["distribution"] for d in payload["per_dist"]]
    x = range(len(dists))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # (a) lookup latency per distribution, grouped by variant
    ax = axes[0]
    variant_names = [v["name"] for v in payload["per_dist"][0]["variants"]]
    width = 0.2
    for i, vn in enumerate(variant_names):
        ys = [d["variants"][i]["lookup_ns"] for d in payload["per_dist"]]
        ax.bar([xi + (i - 1.5) * width for xi in x], ys, width,
               label=vn, color=COLORS[vn], alpha=0.85)
        for xi, y in zip(x, ys):
            ax.annotate(f"{y:.0f}", (xi + (i - 1.5) * width, y),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(dists)
    ax.set_ylabel("lookup latency (ns / query)")
    ax.set_title("Point-lookup latency vs. distribution")
    ax.legend(loc="upper left", fontsize=8)

    # (b) linear pieces (model size proxy).  The piece counts span less than
    # 2x (349..638), so a log scale would compress the very difference we want
    # to show; linear keeps 349 vs 638 visually honest.
    ax = axes[1]
    learned_i = variant_names.index("learned_e16")
    block_i = variant_names.index("learned_block")
    for i, vn in zip((learned_i, block_i), ("learned_e16", "learned_block")):
        ys = [d["variants"][i].get("pieces", 0) for d in payload["per_dist"]]
        ax.bar([xi + (i - 0.5) * width for xi in x], ys, width,
               label=vn, color=COLORS[vn], alpha=0.85)
        for xi, y in zip(x, ys):
            ax.annotate(f"{y}", (xi + (i - 0.5) * width, y),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(dists)
    ax.set_ylabel("linear pieces")
    ax.set_title("Model size: pieces per distribution")
    ax.legend(loc="upper left", fontsize=8)

    # (c) prediction-error max vs. bound (single series: learned_e16)
    ax = axes[2]
    bound = payload["meta"].get("threshold", 16)
    for i, d in enumerate(payload["per_dist"]):
        li = d["variants"][2]
        err = li.get("error", {})
        ax.bar(i, err.get("max", 0), 0.5, label=d["distribution"],
               color=COLORS["learned_e16"], alpha=0.85)
        ax.annotate(f"p99={err.get('p99', 0)}", (i, err.get("max", 0)),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8)
    ax.axhline(bound, color="black", linestyle="--", linewidth=1.2)
    ax.text(len(dists) - 0.4, bound + 0.6, f"error bound E={bound}",
            fontsize=8, ha="right")
    ax.set_xticks(list(x))
    ax.set_xticklabels(dists)
    ax.set_ylabel("max prediction error")
    ax.set_title("Prediction error stays inside bound")
    ax.set_ylim(0, max(bound * 2, 4))

    fig.suptitle("Learned index: distribution sensitivity (uniform / clustered / zipf)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "fig_index_distribution")


def fig_index_block(payload: dict) -> None:
    """Block-routing sweep: latency / memory / comparisons / error bound vs.
    block size."""
    refs = payload["references"]
    sweep = payload["block_sweep"]
    bs = [s["block_size"] for s in sweep]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # (a) lookup latency vs block size (with reference lines)
    ax = axes[0]
    ax.plot(bs, [s["lookup_ns"] for s in sweep], "o-",
            color=COLORS["learned_block"], label="learned_block")
    for name, ls, lab in (("sorted", "--", "sorted_array"),
                          ("btree", "-.", "btree"),
                          ("learned_point", ":", "learned_e16")):
        ax.axhline(refs[name]["lookup_ns"], linestyle=ls,
                   color=COLORS[lab], label=lab)
    ax.set_xlabel("block size")
    ax.set_ylabel("lookup latency (ns / query)")
    ax.set_title("Block-routing: latency vs. block size")
    ax.legend(loc="upper right", fontsize=8)

    # (b) estimated comparisons per lookup
    ax = axes[1]
    ax.plot(bs, [s["comparisons"] for s in sweep], "s-",
            color=COLORS["learned_block"], label="learned_block")
    for name, ls, lab in (("sorted", "--", "sorted_array"),
                          ("btree", "-.", "btree"),
                          ("learned_point", ":", "learned_e16")):
        ax.axhline(refs[name]["comparisons"], linestyle=ls,
                   color=COLORS[lab], label=lab)
    ax.set_xlabel("block size")
    ax.set_ylabel("estimated comparisons / lookup")
    ax.set_title("Estimated comparisons vs. block size")
    ax.legend(loc="upper right", fontsize=8)

    # (c) memory vs block size (model bytes only)
    ax = axes[2]
    ax.plot(bs, [s["model_bytes"] for s in sweep], "o-",
            color=COLORS["learned_block"], label="model bytes")
    ax.set_xlabel("block size")
    ax.set_ylabel("model memory (bytes)")
    ax.set_title("Block-routing: model memory vs. block size")
    ax.annotate(f"max block error = {max(s['block_error_max'] for s in sweep)}",
                xy=(0.02, 0.95), xycoords="axes fraction", fontsize=9,
                va="top", ha="left")

    fig.suptitle("Learned index: block-routing sweep "
                 f"({payload['meta'].get('distribution', 'uniform')} keys)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "fig_index_block")


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
    dist_path = os.path.join(RESULTS, "index_distribution.json")
    block_path = os.path.join(RESULTS, "index_block.json")
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

    if os.path.exists(dist_path):
        print("fig_index_distribution:")
        with open(dist_path, encoding="utf-8") as fh:
            fig_index_distribution(json.load(fh))
    else:
        print("  (skip: index_distribution.json missing; run "
              "`run_benchmarks.py --distribution uniform clustered zipf`)")
    if os.path.exists(block_path):
        print("fig_index_block:")
        with open(block_path, encoding="utf-8") as fh:
            fig_index_block(json.load(fh))
    else:
        print("  (skip: index_block.json missing; run "
              "`run_benchmarks.py --block-sizes 32 64 128 256`)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
