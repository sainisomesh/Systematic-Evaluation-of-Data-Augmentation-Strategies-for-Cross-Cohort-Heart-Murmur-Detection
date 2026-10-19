"""Collect the per-run Kaggle outputs into one table.

Usage: python3 analysis/consolidate_kaggle.py [RUNS_DIR]

RUNS_DIR (default: results/kaggle_runs) holds one folder per notebook version (vNNN) with
global_results.json and/or results_<experiment>.json. For each experiment, the most recent
successful run is written to results/kaggle_runs_consolidated.csv.
"""

import csv
import glob
import json
import os
import sys

COHORTS = ["physionet22_val", "WC2023", "WVU", "OHH-2025"]


def load_runs(runs_dir):
    runs = {}
    version_dirs = glob.glob(os.path.join(runs_dir, "v*"))
    for d in sorted(version_dirs, key=lambda p: int(os.path.basename(p)[1:])):
        version = int(os.path.basename(d)[1:])
        records = {}
        global_file = os.path.join(d, "global_results.json")
        if os.path.exists(global_file):
            with open(global_file) as fh:
                data = json.load(fh)
            records.update({k: r for k, r in data.items() if isinstance(r, dict)})
        for f in glob.glob(os.path.join(d, "results_*.json")):
            with open(f) as fh:
                r = json.load(fh)
            records.setdefault(r.get("name", os.path.basename(f)[len("results_"):-len(".json")]), r)
        for name, r in records.items():
            if r.get("status") == "SUCCESS" and isinstance(r.get("WC2023"), dict):
                runs.setdefault(name, []).append((version, r))
    return runs


def flatten(r):
    row = {}
    for c in COHORTS:
        x = r.get(c) or {}
        ci = x.get("auc_ci") or [None, None]
        row[f"{c}_auc"] = x.get("auc")
        row[f"{c}_ci_low"], row[f"{c}_ci_high"] = ci[0], ci[1]
        row[f"{c}_n_patients"] = x.get("n_patients")
    external = [row[f"{c}_auc"] for c in COHORTS[1:]]
    row["mean_external_auc"] = sum(external) / 3 if None not in external else None
    row["wall_time_seconds"] = r.get("wall_time_seconds")
    return row


def is_paper_run(name):
    return "_prod_" in name or name.startswith("00")


def main():
    runs = load_runs(sys.argv[1] if len(sys.argv) > 1 else "results/kaggle_runs")
    rows = [{"experiment": name, "kaggle_version": runs[name][-1][0], **flatten(runs[name][-1][1])}
            for name in sorted(n for n in runs if is_paper_run(n))]
    with open("results/kaggle_runs_consolidated.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} experiments to results/kaggle_runs_consolidated.csv")


if __name__ == "__main__":
    main()
