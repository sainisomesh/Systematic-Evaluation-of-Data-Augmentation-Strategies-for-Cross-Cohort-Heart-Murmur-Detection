"""Write Supplementary Table S1: AUROC with 95% CI for every configuration.

Usage: python3 analysis/make_supplementary_table.py
Output: results/supplementary_table_S1.tex
"""

from common import load_results, sort_key


def cell(row, cohort):
    auc, low, high = (100 * float(row[f"{cohort}_{k}"]) for k in ("auc", "ci_low", "ci_high"))
    return f"{auc:.2f} ({low:.1f}--{high:.1f})"


def main():
    rows = sorted(load_results(), key=lambda r: sort_key(r["experiment"]))
    body = "\n".join(
        f"{r['experiment'].replace('_', chr(92) + '_')} & {cell(r, 'physionet22_val')} & {cell(r, 'WC2023')} & "
        f"{cell(r, 'WVU')} & {cell(r, 'OHH-2025')} & {100 * float(r['mean_external_auc']):.2f} \\\\"
        for r in rows)
    tex = r"""\documentclass[10pt]{article}
\usepackage[margin=1.6cm,landscape]{geometry}
\usepackage{booktabs,longtable}
\begin{document}
\section*{Supplementary Table S1}
Patient-level AUROC (\%) with 95\% bootstrap confidence intervals (1,000 patient resamples) for all 82
configurations, by experiment identifier. External cohorts: WC2023 ($n=108$), WVU ($n=76$), and OHH-2025
($n=1{,}151$ patients). Mean ext.: mean AUROC over the three external cohorts.
\begin{longtable}{lccccr}
\toprule
Experiment & CirCor (val) & WC2023 & WVU & OHH-2025 & Mean ext. \\
\midrule
\endhead
""" + body + r"""
\bottomrule
\end{longtable}
\end{document}
"""
    with open("results/supplementary_table_S1.tex", "w") as fh:
        fh.write(tex)
    print(f"Wrote {len(rows)} rows to results/supplementary_table_S1.tex")


if __name__ == "__main__":
    main()
