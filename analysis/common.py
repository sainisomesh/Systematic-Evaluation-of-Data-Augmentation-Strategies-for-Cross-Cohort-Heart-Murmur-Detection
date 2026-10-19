"""Shared helpers for the analysis scripts."""

import csv
import re

RESULTS_CSV = "results/kaggle_runs_consolidated.csv"
COHORTS = ["physionet22_val", "WC2023", "WVU", "OHH-2025"]


def load_results():
    with open(RESULTS_CSV) as fh:
        return list(csv.DictReader(fh))


def sort_key(name):
    m = re.match(r"(\d+)([a-z]*)", name)
    return int(m.group(1)), m.group(2)
