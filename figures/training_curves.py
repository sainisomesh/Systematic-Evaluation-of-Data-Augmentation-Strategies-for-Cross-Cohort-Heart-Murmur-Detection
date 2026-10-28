"""Reproduce the training-curve figure of the paper from the MLflow log of ./run.sh.

Run through the launcher after the full run:  ./run.sh --figures   (run.bat --figures on Windows)
or directly:  .venv/bin/python figures/training_curves.py

Reads the MLflow database that the full run writes (results_reproduced/mlflow.db; each training run is
identified by its output_dir parameter) and checks that the epoch selected by validation AUPRC gives
the validation AUROC that the run reported in results_reproduced/global_results.json. Writes
Fig_training_curves.pdf/.png and training_curves.csv to results_reproduced/figures/training_curves/.
"""
import argparse
import csv
import json
import os
import shutil
import sqlite3

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "results_reproduced")
OUT = os.path.join(RUNS, "figures", "training_curves")
MODELS = [  # (experiment, label, focal loss?)
    ("01_prod_processing_trainer_frozen", "Frozen backbone", False),
    ("02_prod_processing_trainer_unfrozen", "Full fine-tuning", False),
    ("13_prod_hpss_eq", "HPSS-EQ", False),
    ("23_prod_cspec_ultra_light", "Cardiac SpecAugment (5, 48)", False),
    ("29_prod_noisemixed_focal", "Mixed noise + focal loss", True),
    ("39_prod_hpss_focal", "HPSS-EQ + focal loss", True),
]
COLORS = ["#7f7f7f", "#000000", "#0072B2", "#009E73", "#E69F00", "#c0392b"]
USE_TEX = shutil.which("latex") is not None
TEXT_STYLE = ({"text.usetex": True, "font.family": "serif", "font.serif": ["Computer Modern Roman"]} if USE_TEX
              else {"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
                    "axes.formatter.use_mathtext": True, "axes.unicode_minus": False})


def history(dbs, exp):
    """Loss per logging step and validation metrics per epoch of the training run of one experiment.
    If the experiment was trained more than once (for example after a resumed run), the last finished
    run with evaluation results is used."""
    found = []
    for db in dbs:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        rows = con.execute(
            "SELECT r.run_uuid, r.end_time, p.value FROM runs r JOIN params p ON p.run_uuid = r.run_uuid "
            "WHERE p.key = 'output_dir' AND r.status = 'FINISHED'").fetchall()
        for run, end, out_dir in rows:
            if os.path.basename(os.path.normpath(out_dir)) != exp:  # exact folder name (no LIKE wildcards)
                continue
            n_eval = con.execute("SELECT COUNT(*) FROM metrics WHERE run_uuid = ? AND key = 'eval_auc_pr'",
                                 (run,)).fetchone()[0]
            if n_eval:
                found.append((end or 0, db, run))
        con.close()
    if not found:
        raise SystemExit(f"{exp}: no finished training run in {dbs}; run ./run.sh first.")
    _, db, run = max(found)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute("SELECT key, step, value FROM metrics WHERE run_uuid = ? ORDER BY step, timestamp",
                       (run,)).fetchall()
    con.close()
    epoch_at = {s: v for k, s, v in rows if k == "epoch"}
    get = lambda key: sorted({s: v for k, s, v in rows if k == key}.items())
    loss = [(epoch_at[s], v) for s, v in get("loss")]
    ev = {key: [(epoch_at[s], v) for s, v in get(key)] for key in ("eval_auc_roc", "eval_auc_pr", "eval_loss")}
    return loss, ev


def main():
    ap = argparse.ArgumentParser(description="Reproduce the training-curve figure.")
    ap.add_argument("--mlflow_db", nargs="*", default=[os.path.join(RUNS, "mlflow.db")],
                    help="MLflow database(s) of the training runs")
    ap.add_argument("--results", default=os.path.join(RUNS, "global_results.json"),
                    help="results of the same runs, used to check the selected epoch")
    args = ap.parse_args()
    for db in args.mlflow_db:
        if not os.path.exists(db):
            raise SystemExit(f"{db} not found: run ./run.sh first (it logs the training runs to MLflow).")
    reported = json.load(open(args.results)) if os.path.exists(args.results) else {}
    paper = {r["experiment"]: r for r in csv.DictReader(open(os.path.join(ROOT, "results", "kaggle_runs_consolidated.csv")))}

    plt.rcParams.update({**TEXT_STYLE, "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "axes.linewidth": 1.4, "xtick.major.width": 1.2,
                         "ytick.major.width": 1.2, "xtick.major.size": 4, "ytick.major.size": 4,
                         "xtick.direction": "out", "ytick.direction": "out", "savefig.facecolor": "white",
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.55))
    table = []
    for (exp, label, focal), color in zip(MODELS, COLORS):
        loss, ev = history(args.mlflow_db, exp)
        pr = ev["eval_auc_pr"]
        best = max(range(len(pr)), key=lambda i: pr[i][1])  # first maximum, as load_best_model_at_end
        roc_best = ev["eval_auc_roc"][best][1]
        rep = (reported.get(exp) or {}).get("physionet22_val", {}).get("auc")
        if isinstance(rep, (int, float)):
            assert abs(roc_best - rep) < 1e-6, (exp, roc_best, rep)
        print(f"{exp:40s} selected epoch {pr[best][0]:.0f}: validation AUROC {roc_best:.4f}"
              f" (run reported {f'{rep:.4f}' if rep is not None else 'n/a'}, paper {float(paper[exp]['physionet22_val_auc']):.4f})")
        ls = "--" if focal else "-"
        axes[0].plot(*zip(*loss), ls=ls, color=color, lw=1.3, label=label)
        for ax, key in ((axes[1], "eval_auc_roc"), (axes[2], "eval_auc_pr")):
            e, v = zip(*ev[key])
            ax.plot(e, [100 * x for x in v], ls=ls, color=color, lw=1.3, marker="o", ms=2.5)
            ax.plot(e[best], 100 * v[best], marker="o", ms=6, mfc="none", mec=color, mew=1.2)
        for key, series in [("train_loss", loss)] + list(ev.items()):
            table += [{"experiment": exp, "series": key, "epoch": round(e, 4), "value": v} for e, v in series]

    pct = "\\%" if USE_TEX else "%"
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Training loss")
    axes[1].set_ylabel(f"Validation AUROC ({pct})")
    axes[2].set_ylabel(f"Validation AUPRC ({pct})")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks(range(0, 11, 2))
    for ax, t in zip(axes, "ABC"):
        ax.text(-0.3, 1.04, t, transform=ax.transAxes, fontsize=10, va="bottom")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7.5, handlelength=2.6,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=1.2)
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"Fig_training_curves.{ext}"), dpi=220)
    with open(os.path.join(OUT, "training_curves.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    print("written:", os.path.join(OUT, "Fig_training_curves.pdf"))


if __name__ == "__main__":
    main()
