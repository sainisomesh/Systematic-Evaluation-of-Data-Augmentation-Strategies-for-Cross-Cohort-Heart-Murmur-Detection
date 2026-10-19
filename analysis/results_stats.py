"""Derived statistics reported in the paper.

Usage: python3 analysis/results_stats.py

Reports the combination analysis, mean AUROC by number of components, the spread of the
baseline and gain-jitter runs, the gain-jitter run pairs, and Spearman correlations between evaluation sets.
"""

import statistics

from scipy.stats import spearmanr

from common import COHORTS, load_results

LABELS = ["CirCor", "WC2023", "WVU", "OHH-2025", "mean_ext"]
COMPONENTS = {
    "cspec_ultra_light": "cspec", "gain_jitter": "gain", "cardiac_crop": "crop",
    "quality_weight": "qw", "noisemixed": "nmix", "noise025": "n025", "noise100": "n100",
    "noise500": "n500", "focal": "focal", "hpss": "hpss", "spec": "spec",
}
SINGLE = {
    "focal": "04_prod_focal_loss", "spec": "05_prod_spec_augment", "n025": "06_prod_noise_025",
    "n100": "07_prod_noise_100", "n500": "08_prod_noise_500", "nmix": "09_prod_noise_mixed",
    "gain": "11_prod_gain_jitter", "hpss": "13_prod_hpss_eq", "crop": "14_prod_cardiac_crop",
    "qw": "16_prod_quality_weight", "cspec": "23_prod_cspec_ultra_light",
}
BASELINE_AND_GAIN = ["02_prod_processing_trainer_unfrozen", "11_prod_gain_jitter",
                  "11b_prod_gain_jitter_mild", "11c_prod_gain_jitter_extreme"]
GAIN_PAIRS = [
    ("04_prod_focal_loss", "49_prod_gain_jitter_focal"),
    ("05_prod_spec_augment", "52_prod_gain_jitter_spec"),
    ("06_prod_noise_025", "46_prod_gain_jitter_noise025"),
    ("07_prod_noise_100", "47_prod_gain_jitter_noise100"),
    ("08_prod_noise_500", "48_prod_gain_jitter_noise500"),
    ("09_prod_noise_mixed", "45_prod_gain_jitter_noisemixed"),
    ("13_prod_hpss_eq", "50_prod_gain_jitter_hpss"),
    ("14_prod_cardiac_crop", "51_prod_gain_jitter_cardiac_crop"),
    ("16_prod_quality_weight", "54_prod_gain_jitter_quality_weight"),
    ("23_prod_cspec_ultra_light", "53_prod_gain_jitter_cspec_ultra_light"),
    ("39_prod_hpss_focal", "78_prod_gain_jitter_hpss_focal"),
    ("41_prod_cspec_ultra_light_noise025", "75_prod_gain_jitter_cspec_ultra_light_noise025"),
    ("42_prod_cspec_ultra_light_noise100", "74_prod_gain_jitter_cspec_ultra_light_noise100"),
]

RUNS = {r["experiment"]: [float(r[f"{c}_auc"]) for c in COHORTS] + [float(r["mean_external_auc"])]
        for r in load_results()}


def components(name):
    rest, found = name.split("_prod_")[1], []
    for token, label in COMPONENTS.items():
        if token in rest:
            found.append(label)
            rest = rest.replace(token, "")
    return found


def is_combination(name):
    return name[:2].isdigit() and int(name[:2]) >= 24


def combination_summary(exclude_gain):
    groups = {"augmentations only": [], "with focal loss": []}
    for name, values in RUNS.items():
        if not is_combination(name):
            continue
        parts = components(name)
        if exclude_gain and "gain" in parts:
            continue
        parent_scores = [RUNS[SINGLE[p]][4] for p in parts]
        group = "with focal loss" if "focal" in parts else "augmentations only"
        groups[group].append((values[4], max(parent_scores), min(parent_scores)))
    for group, items in groups.items():
        above = sum(v > best for v, best, _ in items)
        below = sum(v < worst for v, _, worst in items)
        delta = 100 * statistics.mean(v - best for v, best, _ in items)
        print(f"  {group:19s} n={len(items):2d}  above strongest={above}  below weakest={below}  "
              f"mean change vs strongest={delta:+.2f}")


def main():
    print("Combinations (all):")
    combination_summary(exclude_gain=False)
    print("Combinations (excluding gain jitter):")
    combination_summary(exclude_gain=True)

    by_count = {1: [RUNS[n][4] for n in SINGLE.values()]}
    for name, values in RUNS.items():
        if is_combination(name):
            by_count.setdefault(len(components(name)), []).append(values[4])
    print("Mean external AUROC by number of components:",
          {k: (len(v), round(100 * statistics.mean(v), 2)) for k, v in sorted(by_count.items())})

    print("Baseline and gain-jitter runs (02, 11, 11b, 11c):")
    for i, label in enumerate(LABELS):
        v = [RUNS[n][i] for n in BASELINE_AND_GAIN]
        print(f"  {label:9s} min {100 * min(v):.2f}  max {100 * max(v):.2f}  "
              f"range {100 * (max(v) - min(v)):.2f}  SD {100 * statistics.stdev(v):.2f}")
    ceiling = max(RUNS[n][4] for n in BASELINE_AND_GAIN)
    print("  configurations above the highest of these runs:",
          [(n, round(100 * v[4], 2)) for n, v in RUNS.items() if v[4] > ceiling])

    print(f"Gain-jitter pairs (n={len(GAIN_PAIRS)}):")
    for i, label in enumerate(LABELS):
        d = [RUNS[b][i] - RUNS[a][i] for a, b in GAIN_PAIRS]
        rms = (sum(x * x for x in d) / len(d)) ** 0.5
        print(f"  {label:9s} mean |difference| {100 * statistics.mean(abs(x) for x in d):.2f}  "
              f"max {100 * max(abs(x) for x in d):.2f}  implied per-run SD {100 * rms / 2 ** 0.5:.2f}")

    fine_tuned = [v for n, v in RUNS.items() if not n.startswith(("00", "01_", "03_"))]
    print(f"Spearman correlations over {len(fine_tuned)} fully fine-tuned configurations:")
    for i, label in enumerate(LABELS[1:], 1):
        r = spearmanr([v[0] for v in fine_tuned], [v[i] for v in fine_tuned])
        print(f"  CirCor vs {label}: rho {r.correlation:.2f}, p {r.pvalue:.2f}")
    for a, b in [(1, 2), (2, 3), (1, 3)]:
        r = spearmanr([v[a] for v in fine_tuned], [v[b] for v in fine_tuned])
        print(f"  {LABELS[a]} vs {LABELS[b]}: rho {r.correlation:.2f}")


if __name__ == "__main__":
    main()
