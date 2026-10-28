"""Checks for figures/training_curves.py (plain assertions, run directly).

Run from the repository root:  .venv/bin/python tests/test_training_curves.py
"""
import os
import sqlite3
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures"))
import training_curves as T  # noqa: E402


def make_db(path, runs):
    """runs: list of (run_uuid, output_dir, status, end_time, [(key, step, value), ...])."""
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE runs (run_uuid TEXT, status TEXT, end_time INTEGER)")
    con.execute("CREATE TABLE params (key TEXT, value TEXT, run_uuid TEXT)")
    con.execute("CREATE TABLE metrics (key TEXT, value REAL, timestamp INTEGER, run_uuid TEXT, step INTEGER)")
    for run, out_dir, status, end, metrics in runs:
        con.execute("INSERT INTO runs VALUES (?, ?, ?)", (run, status, end))
        con.execute("INSERT INTO params VALUES ('output_dir', ?, ?)", (out_dir, run))
        for k, step, v in metrics:
            con.execute("INSERT INTO metrics VALUES (?, ?, 0, ?, ?)", (k, v, run, step))
    con.commit()
    con.close()


def epoch_metrics(auc_pr, auc_roc):
    out = [("epoch", 10, 0.5), ("loss", 10, 0.9), ("epoch", 20, 1.0), ("eval_auc_pr", 20, auc_pr),
           ("eval_auc_roc", 20, auc_roc), ("eval_loss", 20, 0.3)]
    return out


def test_exact_experiment_match():
    """Experiment names contain underscores; a similar name must not be picked up (no LIKE wildcards)."""
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "mlflow.db")
        make_db(db, [("a", "./13_prod_hpss_eq", "FINISHED", 1, epoch_metrics(0.8, 0.90)),
                     ("b", "./13_prod_hpss_eqX", "FINISHED", 2, epoch_metrics(0.7, 0.10)),
                     ("c", "./13xprod_hpss_eq", "FINISHED", 3, epoch_metrics(0.7, 0.20))])
        _, ev = T.history([db], "13_prod_hpss_eq")
        assert ev["eval_auc_roc"] == [(1.0, 0.90)], ev


def test_latest_finished_run_with_evaluation():
    """If an experiment was trained more than once, use the last finished run that has evaluation results."""
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "mlflow.db")
        make_db(db, [("old", "/kaggle/working/39_prod_hpss_focal", "FINISHED", 1, epoch_metrics(0.8, 0.95)),
                     ("new", "./39_prod_hpss_focal", "FINISHED", 5, epoch_metrics(0.8, 0.97)),
                     ("crashed", "./39_prod_hpss_focal", "FAILED", 9, epoch_metrics(0.8, 0.10)),
                     ("noeval", "./39_prod_hpss_focal", "FINISHED", 7, [("loss", 10, 0.5)])])
        loss, ev = T.history([db], "39_prod_hpss_focal")
        assert ev["eval_auc_roc"] == [(1.0, 0.97)], ev
        assert loss == [(0.5, 0.9)], loss


def test_missing_log_fails_clearly():
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "training_curves.py")
    r = subprocess.run([sys.executable, script, "--mlflow_db", "/nonexistent/mlflow.db"], capture_output=True, text=True)
    assert r.returncode != 0 and "run ./run.sh first" in (r.stdout + r.stderr), r


if __name__ == "__main__":
    test_exact_experiment_match()
    test_latest_finished_run_with_evaluation()
    test_missing_log_fails_clearly()
    print("training curve tests passed")
