"""Write results/prod_results.md: AUROC with 95% CI for all 82 configurations.

Usage: python3 analysis/make_results_md.py
"""

from common import load_results, sort_key

NAMES = {
    "00_dryrun_audioset_zeroshot": "AudioSet zero-shot, raw audio",
    "00b_dryrun_audioset_zeroshot_stdpreproc": "AudioSet zero-shot, preprocessed audio",
    "01_prod_processing_trainer_frozen": "Frozen backbone (head only)",
    "02_prod_processing_trainer_unfrozen": "Full fine-tuning (baseline)",
    "03_prod_processing_trainer_progressive": "Progressive unfreezing",
}
TOKENS = [
    ("cspec_ultra_light", "cardiac SpecAugment (5, 48)"), ("cspec_light", "cardiac SpecAugment (8, 48)"),
    ("cspec_moderate", "cardiac SpecAugment (15, 80)"), ("cspec_aggressive", "cardiac SpecAugment (24, 100)"),
    ("spec_moderate", "SpecAugment (24, 80)"), ("spec_aggressive", "SpecAugment (40, 100)"),
    ("spec_extreme", "SpecAugment (48, 150)"), ("spec_augment", "SpecAugment (12, 48)"),
    ("spec", "SpecAugment (12, 48)"), ("gain_jitter_mild", "gain jitter r=0.15"),
    ("gain_jitter_extreme", "gain jitter r=0.5"), ("gain_jitter", "gain jitter r=0.3"),
    ("hpss_eq", "HPSS-EQ"), ("hpss", "HPSS-EQ"), ("cardiac_crop", "cardiac-cycle crop"),
    ("resp_am", "respiratory modulation"), ("quality_weight", "quality weighting"),
    ("polarity_flip", "polarity inversion"), ("mixup", "mixup"),
    ("noisemixed", "noise, mixed SNR"), ("noise_mixed", "noise, mixed SNR"),
    ("noise025", "noise 32 dB"), ("noise_025", "noise 32 dB"), ("noise100", "noise 20 dB"),
    ("noise_100", "noise 20 dB"), ("noise500", "noise 6 dB"), ("noise_500", "noise 6 dB"),
    ("focal_loss", "focal loss"), ("focal", "focal loss"),
]
SECTIONS = [(3, "Zero-shot baselines and backbone adaptation"),
            (23, "Single components added to full fine-tuning"),
            (71, "Two-component combinations"),
            (999, "Three-component combinations")]


def describe(name):
    if name in NAMES:
        return NAMES[name]
    rest, parts = name.split("_prod_")[1], []
    for token, label in TOKENS:
        if token in rest:
            parts.append(label)
            rest = rest.replace(token, " ")
    return " + ".join(parts)


def section(name):
    number = sort_key(name)[0]
    return next(title for limit, title in SECTIONS if number <= limit)


def cell(row, cohort):
    auc, low, high = (100 * float(row[f"{cohort}_{k}"]) for k in ("auc", "ci_low", "ci_high"))
    return f"{auc:.2f} ({low:.1f}-{high:.1f})"


def main():
    rows = sorted(load_results(), key=lambda r: sort_key(r["experiment"]))
    lines = [
        "# Results",
        "",
        "Patient-level AUROC (%) with 95% bootstrap confidence intervals (1,000 patient resamples) for all",
        "82 configurations. Models were trained on CirCor DigiScope (PhysioNet 2022) only; WC2023 (108",
        "patients), WVU (76) and OHH-2025 (1,151) are external cohorts that were never used for training",
        "or model selection. CirCor (val) is the held-out 20% patient split, which was also used to select",
        "the checkpoint. Mean ext. is the mean AUROC over the three external cohorts.",
        "",
        "Experiment IDs match `unified_murmur_pipeline.py`. Source: `kaggle_runs_consolidated.csv`.",
    ]
    current = None
    for r in rows:
        title = section(r["experiment"])
        if title != current:
            current = title
            lines += ["", f"## {title}", "",
                      "| ID | Configuration | CirCor (val) | WC2023 | WVU | OHH-2025 | Mean ext. |",
                      "|---|---|---|---|---|---|---|"]
        lines.append(f"| `{r['experiment']}` | {describe(r['experiment'])} | {cell(r, 'physionet22_val')} | "
                     f"{cell(r, 'WC2023')} | {cell(r, 'WVU')} | {cell(r, 'OHH-2025')} | "
                     f"{100 * float(r['mean_external_auc']):.2f} |")
    with open("results/prod_results.md", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Wrote {len(rows)} rows to results/prod_results.md")


if __name__ == "__main__":
    main()
