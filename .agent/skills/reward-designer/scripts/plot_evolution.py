#!/usr/bin/env python3
"""Plot the Eureka paper's Figure 5 -- cumulative best success per iteration.

    python3 scripts/plot_evolution.py                                  # outputs/candidates
    python3 scripts/plot_evolution.py --run runs/r0 --run runs/r1 ...  # mean +- std over runs
    python3 scripts/plot_evolution.py --human outputs/baselines/human_metrics.json \
                                      --no-evolution runs/k32/candidates

Each --run is a candidates directory holding iter<i>_k<j>_metrics.json files
from ONE search. The y value at iteration i is the best success_rate seen in
iterations 0..i of that run (the "cumulative best" the paper plots); with more
than one --run the points are the mean and the error bars the std across runs.

--human draws Isaac Lab's hand-written reward as a horizontal line: a metrics
file from eval_headless.py run WITHOUT --reward-file, or a bare number.
--no-evolution draws the paper's "Eureka w/o Evolution" ablation: the best
candidate of a search run with --iterations 1 and K equal to the total samples
of the first two evolution iterations (2 x 16 = 32 here). Both are repeatable
and averaged.

The numbers are printed before plotting, so the table works without
matplotlib; only the figure needs it.
"""
import argparse
import json
import os
import statistics
import sys

from summarize_run import load_candidates

# Categorical slots 1-3 of the dataviz reference palette (light mode), in fixed
# order. Red/green are left out on purpose: they read as status colours, and
# the red/green pair is the one colour-blind readers lose first.
EUREKA_COLOR = "#2a78d6"
HUMAN_COLOR = "#eb6834"
ABLATION_COLOR = "#1baf7a"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"


def cumulative_best(candidates):
    """Return [best success_rate over iterations 0..i] for i in 0..last."""
    executed = [row for row in candidates if row[3].get("status") == "success"]
    if not candidates:
        return []
    last = max(row[0] for row in candidates)
    curve, best = [], None
    for iteration in range(last + 1):
        scores = [row[3].get("success_rate", 0.0) for row in executed if row[0] == iteration]
        if scores:
            best = max(scores) if best is None else max(best, max(scores))
        # An iteration where nothing executed carries the previous best forward:
        # the search still holds that reward, it just found nothing new.
        curve.append(best)
    return curve


def baseline_value(source):
    """A bare number, a metrics JSON, or a candidates directory -> success_rate."""
    try:
        return float(source)
    except ValueError:
        pass
    if os.path.isdir(source):
        executed = [row[3].get("success_rate", 0.0) for row in load_candidates(source)
                    if row[3].get("status") == "success"]
        if not executed:
            raise SystemExit(f"no executed candidates in {source}")
        return max(executed)
    with open(source) as handle:
        metrics = json.load(handle)
    if metrics.get("status") != "success":
        raise SystemExit(f"{source} did not succeed (status={metrics.get('status')})")
    return metrics["success_rate"]


def mean_and_std(values):
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def main():
    skill_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", action="append", default=None,
                        help="Candidates directory of one search (repeatable). "
                             "Default: outputs/candidates.")
    parser.add_argument("--human", action="append", default=[],
                        help="Human-reward baseline: metrics JSON or number (repeatable).")
    parser.add_argument("--no-evolution", action="append", default=[],
                        help="Ablation: candidates dir, metrics JSON or number (repeatable).")
    parser.add_argument("--title", default="Franka Reach")
    parser.add_argument("--out", default=os.path.join(skill_root, "outputs", "figure5_evolution.png"))
    args = parser.parse_args()

    run_dirs = args.run or [os.path.join(skill_root, "outputs", "candidates")]
    curves = []
    for directory in run_dirs:
        candidates = load_candidates(directory)
        if not candidates:
            raise SystemExit(f"no iter*_k*_metrics.json found in {directory}")
        curves.append(cumulative_best(candidates))

    # Runs of different lengths are compared only over the iterations they share.
    length = min(len(curve) for curve in curves)
    iterations = list(range(1, length + 1))  # 1-based, as the paper labels them
    means, stds = [], []
    for i in range(length):
        values = [curve[i] for curve in curves if curve[i] is not None]
        mean, std = mean_and_std(values) if values else (float("nan"), 0.0)
        means.append(mean)
        stds.append(std)

    human = mean_and_std([baseline_value(s) for s in args.human]) if args.human else None
    ablation = (mean_and_std([baseline_value(s) for s in args.no_evolution])
                if args.no_evolution else None)

    print(f"runs: {len(curves)}   (error bars: "
          f"{'std across runs' if len(curves) > 1 else 'none -- one run'})")
    print(f"{'iter':>4} {'cum_best':>9} {'std':>7}")
    for iteration, mean, std in zip(iterations, means, stds):
        print(f"{iteration:4d} {mean:9.4f} {std:7.4f}")
    if human:
        print(f"human        {human[0]:.4f} +- {human[1]:.4f}  (n={len(args.human)})")
    if ablation:
        print(f"no evolution {ablation[0]:.4f} +- {ablation[1]:.4f}  "
              f"(n={len(args.no_evolution)})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("\nmatplotlib is not installed; the table above is the data. "
                         "Install it (e.g. apt install python3-matplotlib) for the figure.")

    fig, ax = plt.subplots(figsize=(4.6, 3.4), dpi=200)
    ax.errorbar(iterations, means, yerr=stds if len(curves) > 1 else None,
                color=EUREKA_COLOR, linewidth=2, marker="o", markersize=6,
                markeredgecolor="white", markeredgewidth=1.5,
                capsize=3, elinewidth=1, label="Eureka", zorder=3)
    if human:
        ax.axhline(human[0], color=HUMAN_COLOR, linewidth=1.5, linestyle="--",
                   label="Human", zorder=2)
    if ablation:
        ax.axhline(ablation[0], color=ABLATION_COLOR, linewidth=1.5, linestyle="--",
                   label="Eureka w/o Evolution", zorder=2)

    # Fit the y range to what is drawn rather than 0..1: the curves of interest
    # sit close together near the top and would flatten into one line otherwise.
    drawn = [m - s for m, s in zip(means, stds)] + [m + s for m, s in zip(means, stds)]
    drawn += [line[0] for line in (human, ablation) if line]
    low, high = min(drawn), max(drawn)
    pad = max(0.05, (high - low) * 0.15)
    ax.set_ylim(max(0.0, low - pad), min(1.0, high + pad))
    ax.set_xticks(iterations)

    ax.set_title(args.title, color=INK_PRIMARY, fontsize=11)
    ax.set_xlabel("Eureka iteration", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("Success rate", color=INK_SECONDARY, fontsize=9)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)

    # Direct-label the final cumulative best so the headline needs no squinting.
    ax.annotate(f"{means[-1]:.3f}", (iterations[-1], means[-1]),
                textcoords="offset points", xytext=(0, 8), ha="center",
                fontsize=8, color=INK_PRIMARY)
    # matplotlib lists errorbar containers after plain lines; put Eureka first
    # as the paper does.
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: labels[i] != "Eureka")
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3,
              frameon=False, fontsize=8, labelcolor=INK_SECONDARY)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
