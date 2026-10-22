#!/usr/bin/env python3
"""Summarise a Stage 1 run from the per-candidate metrics files.

    python3 scripts/summarize_run.py                    # outputs/candidates
    python3 scripts/summarize_run.py --dir outputs/smoke/candidates
    python3 scripts/summarize_run.py --csv run.csv      # also write a CSV

Reads outputs/candidates/iter<i>_k<j>_metrics.json, which eval_headless.py
writes for every candidate, and reconstructs what the terminal printed plus the
per-iteration distribution. Pure stdlib, no Isaac Sim, runs on the host.

Why this exists: the run's own stdout is the only place the per-iteration story
appears live, and it is lost unless the run was piped through `tee`. The metrics
files persist, so the whole run can be reconstructed afterwards.
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics
import sys

CANDIDATE_RE = re.compile(r"^iter(\d+)_k(\d+)_metrics\.json$")


def load_candidates(directory):
    """Return [(iteration, k, candidate_id, metrics)] sorted by (iter, k)."""
    found = []
    for path in glob.glob(os.path.join(directory, "*_metrics.json")):
        match = CANDIDATE_RE.match(os.path.basename(path))
        if not match:
            continue  # smoketest / timing / other stray metrics files
        try:
            with open(path) as handle:
                metrics = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  (skipping {os.path.basename(path)}: {exc})", file=sys.stderr)
            continue
        found.append((int(match.group(1)), int(match.group(2)),
                      f"iter{match.group(1)}_k{match.group(2)}", metrics))
    found.sort(key=lambda row: (row[0], row[1]))
    return found


def summarize(candidates):
    iterations = sorted({row[0] for row in candidates})
    print(f"{'iter':>4} {'ran':>6} {'best':>11} {'best':>7} {'median':>7} "
          f"{'worst':>7} {'>0.5':>5} {'<0.05':>6} {'reward':>8} {'corr':>7} {'pos_err':>8}")

    for iteration in iterations:
        group = [row for row in candidates if row[0] == iteration]
        ok = [row for row in group if row[3].get("status") == "success"]
        if not ok:
            print(f"{iteration:4d} {0:>3}/{len(group):<2}   (no candidate executed)")
            continue

        scores = sorted((row[3].get("success_rate", 0.0) for row in ok), reverse=True)
        best = max(ok, key=lambda row: row[3].get("success_rate", 0.0))
        m = best[3]
        correlation = m.get("reward_correlation")
        print(f"{iteration:4d} {len(ok):>3}/{len(group):<2} {best[2]:>11} "
              f"{scores[0]:7.4f} {statistics.median(scores):7.4f} {scores[-1]:7.4f} "
              f"{sum(1 for s in scores if s > 0.5):5d} "
              f"{sum(1 for s in scores if s < 0.05):6d} "
              f"{m.get('mean_reward', 0.0):8.2f} "
              f"{(f'{correlation:+.3f}' if correlation is not None else 'n/a'):>7} "
              f"{m.get('mean_position_error', float('nan')):8.4f}")

    executed = [row for row in candidates if row[3].get("status") == "success"]
    print()
    print(f"candidates: {len(candidates)}   executed: {len(executed)}   "
          f"execute_rate: {len(executed) / max(len(candidates), 1):.2f}")

    failures = {}
    for row in candidates:
        status = row[3].get("status")
        if status != "success":
            failures[status] = failures.get(status, 0) + 1
    if failures:
        print("failures by status: "
              + ", ".join(f"{k}={v}" for k, v in sorted(failures.items())))

    if not executed:
        return None

    # The global best is what the run actually ships -- selection keeps the best
    # reward seen across ALL iterations, so a later iteration degrading does not
    # degrade the output.
    champion = max(executed, key=lambda row: row[3].get("success_rate", 0.0))
    cm = champion[3]
    print()
    print(f"GLOBAL BEST  {champion[2]}  success={cm.get('success_rate', 0):.4f}  "
          f"pos_err={cm.get('mean_position_error', float('nan')):.4f}  "
          f"reward={cm.get('mean_reward', 0):.2f}")
    print(f"  (from iteration {champion[0]} of {max(r[0] for r in candidates)} -- "
          f"a later iteration winning here would mean the search improved)")
    return champion


def report_final_training(skill_root):
    path = os.path.join(skill_root, "outputs", "final_metrics.json")
    if not os.path.isfile(path):
        return
    with open(path) as handle:
        m = json.load(handle)
    if m.get("status") != "success":
        print(f"FINAL TRAIN  did not produce a policy (status={m.get('status')})")
        return
    print(f"FINAL TRAIN  success={m.get('success_rate', 0):.4f}  "
          f"pos_err={m.get('mean_position_error', float('nan')):.4f}  "
          f"reward={m.get('mean_reward', 0):.2f}  "
          f"iters={m.get('train_iterations')}")


def write_csv(candidates, path):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", "k", "candidate", "status", "success_rate",
                         "mean_reward", "reward_correlation", "mean_position_error",
                         "min_position_error", "num_envs", "seed", "n_terms"])
        for iteration, k, cid, m in candidates:
            writer.writerow([
                iteration, k, cid, m.get("status"), m.get("success_rate"),
                m.get("mean_reward"), m.get("reward_correlation"),
                m.get("mean_position_error"), m.get("min_position_error"),
                m.get("num_envs"), m.get("seed"),
                len(m.get("reward_components") or {}),
            ])
    print(f"\nwrote {path}")


def main():
    skill_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dir", default=os.path.join(skill_root, "outputs/candidates"),
                        help="Directory of *_metrics.json files.")
    parser.add_argument("--csv", default=None, help="Also write a per-candidate CSV.")
    args = parser.parse_args()

    candidates = load_candidates(args.dir)
    if not candidates:
        raise SystemExit(f"no iter*_k*_metrics.json found in {args.dir}")

    summarize(candidates)
    report_final_training(skill_root)
    if args.csv:
        write_csv(candidates, args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
