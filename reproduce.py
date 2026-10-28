"""Reproduce all experiments in the paper.

Usage:
    python reproduce.py            set up, download the data, run all 82 experiments, compare with the paper
    python reproduce.py --check    set up, download and verify the data, run the unit tests (no training)
    python reproduce.py --figures  after the full run: reproduce the attribution figure from the
                                   trained models (figures/attribution.py)

Running the command again resumes: verified downloads and finished experiments are kept.
Only the Python standard library is needed to start; everything else is installed into .venv.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sqlite3
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
import venv
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
RUN_DIR = os.path.join(ROOT, "results_reproduced")
VENV_DIR = os.path.join(ROOT, ".venv")
PIPELINE = os.path.join(ROOT, "unified_murmur_pipeline.py")

ZENODO_RECORD = "23460250"
DB_NAME = "hestia_wvu_wc2023_physionet22_ohh2025.db"
NOISE_ZIP = "ICBHI_noise.zip"
FILES = {
    DB_NAME: (6590955520, "c947d6bdc66fd4e2fe0b932136178413"),
    NOISE_ZIP: (91983966, "0e774dd91e1155e92f3cd73995495078"),
}
NOISE_FILES = 174
COHORTS = {  # project_name: (recordings, patients, patients with a murmur)
    "physionet22": (3163, 942, 179),
    "WC2023": (570, 108, 74),
    "WVU": (383, 76, 54),
    "OHH-2025": (5382, 1151, 41),
}
TORCH_VERSION = "2.10.0"
FIGURE_EXPERIMENTS = ["01_prod_processing_trainer_frozen", "02_prod_processing_trainer_unfrozen",
                      "23_prod_cspec_ultra_light", "29_prod_noisemixed_focal", "39_prod_hpss_focal"]
MIN_FREE_GB = 60
SSL_CONTEXT = None


def log(msg):
    print(f"[reproduce] {msg}", flush=True)


def fail(msg):
    print(f"[reproduce] ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def run(cmd, **kwargs):
    log("$ " + " ".join(cmd))
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        fail(f"command failed with exit code {result.returncode}: {' '.join(cmd)}")


def check_system():
    if not (3, 12) <= sys.version_info[:2] <= (3, 14):
        fail(f"Python 3.12 to 3.14 is required (found {platform.python_version()}); "
             "the pinned packages in requirements.txt are available for these versions.")
    if platform.system() == "Darwin" and platform.machine() == "x86_64":
        fail(f"PyTorch {TORCH_VERSION} is not available for Intel (x86_64) Python on macOS. "
             "Use an arm64 Python on Apple silicon, or Linux or Windows with an NVIDIA GPU.")
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    log(f"Python {platform.python_version()} on {platform.system()} {platform.machine()}, {free_gb:.0f} GB free")
    if free_gb < MIN_FREE_GB and not os.path.exists(os.path.join(DATA_DIR, DB_NAME)):
        fail(f"at least {MIN_FREE_GB} GB of free disk space is needed (data, environment and outputs).")


def venv_python():
    if os.environ.get("REPRO_PYTHON"):
        return os.environ["REPRO_PYTHON"]
    sub = "Scripts" if os.name == "nt" else "bin"
    exe = "python.exe" if os.name == "nt" else "python"
    return os.path.join(VENV_DIR, sub, exe)


def has_nvidia_gpu():
    return shutil.which("nvidia-smi") is not None and subprocess.run(
        ["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def setup_environment():
    py = venv_python()
    if os.environ.get("REPRO_PYTHON"):
        log(f"Using existing interpreter {py}")
        return py
    marker = os.path.join(VENV_DIR, ".installed")
    if os.path.exists(marker):
        log("Environment already set up (.venv)")
        return py
    log("Creating environment in .venv")
    try:
        venv.create(VENV_DIR, with_pip=True, clear=os.path.exists(VENV_DIR))
    except Exception as e:
        fail(f"could not create the environment ({e}). On Debian or Ubuntu, install python3-venv first.")
    run([py, "-m", "pip", "install", "--upgrade", "pip"])
    torch_cmd = [py, "-m", "pip", "install", f"torch=={TORCH_VERSION}", f"torchaudio=={TORCH_VERSION}"]
    if platform.system() == "Windows" and has_nvidia_gpu():
        torch_cmd += ["--index-url", "https://download.pytorch.org/whl/cu128"]
    run(torch_cmd)
    run([py, "-m", "pip", "install", "-r", os.path.join(ROOT, "requirements.txt")])
    with open(marker, "w") as fh:
        fh.write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    return py


def make_ssl_context(py):
    """Use the certifi bundle, since some Python installations ship without root certificates."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    out = subprocess.run([py, "-c", "import certifi; print(certifi.where())"], capture_output=True, text=True)
    cafile = out.stdout.strip()
    if out.returncode == 0 and os.path.exists(cafile):
        return ssl.create_default_context(cafile=cafile)
    return ssl.create_default_context()


def zenodo_files():
    url = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
    try:
        with urllib.request.urlopen(url, timeout=60, context=SSL_CONTEXT) as resp:
            record = json.load(resp)
    except (urllib.error.URLError, TimeoutError) as e:
        log(f"Could not read the Zenodo record ({e}).")
        return {}
    return {f["key"]: (f["links"]["self"], f["size"], f["checksum"].split(":", 1)[1])
            for f in record.get("files", [])}


def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def download(url, dest, size, retries=5):
    partial = dest + ".part"
    for attempt in range(1, retries + 1):
        start = os.path.getsize(partial) if os.path.exists(partial) else 0
        if start > size:
            os.remove(partial)
            start = 0
        if start == size:
            break
        req = urllib.request.Request(url, headers={"Range": f"bytes={start}-"} if start else {})
        try:
            with urllib.request.urlopen(req, timeout=120, context=SSL_CONTEXT) as resp:
                mode = "ab" if start and resp.status == 206 else "wb"
                done = start if mode == "ab" else 0
                with open(partial, mode) as out:
                    last = time.time()
                    while True:
                        chunk = resp.read(1 << 22)
                        if not chunk:
                            break
                        out.write(chunk)
                        done += len(chunk)
                        if time.time() - last > 10:
                            log(f"  {os.path.basename(dest)}: {100 * done / size:.0f}% of {size / 1e9:.2f} GB")
                            last = time.time()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            log(f"  download interrupted ({e}); retry {attempt}/{retries}")
            time.sleep(min(60, 5 * attempt))
    if not os.path.exists(partial) or os.path.getsize(partial) != size:
        fail(f"could not download {os.path.basename(dest)}. Run the command again to resume.")
    os.replace(partial, dest)


def fetch_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    remote = None
    for name, (size, md5) in FILES.items():
        dest = os.path.join(DATA_DIR, name)
        if os.path.exists(dest):
            log(f"Checking {name}")
            if os.path.getsize(dest) == size and md5sum(dest) == md5:
                continue
            if remote is None:
                remote = zenodo_files()
            if name in remote and os.path.getsize(dest) == remote[name][1] and md5sum(dest) == remote[name][2]:
                continue
            log(f"{name} does not match the published file; downloading it again")
            os.remove(dest)
        if remote is None:
            remote = zenodo_files()
        if name not in remote:
            fail(f"{name} is not available from Zenodo record {ZENODO_RECORD} "
                 f"(https://doi.org/10.5281/zenodo.{ZENODO_RECORD}). "
                 f"Download it manually into {DATA_DIR} and run the command again.")
        url, size, md5 = remote[name]
        log(f"Downloading {name} ({size / 1e9:.2f} GB)")
        download(url, dest, size)
        if md5sum(dest) != md5:
            os.remove(dest)
            fail(f"{name} failed the checksum after download. Run the command again.")
    noise_dir = os.path.join(DATA_DIR, "ICBHI_noise")
    if not os.path.isdir(noise_dir) or len([f for f in os.listdir(noise_dir) if f.endswith(".wav")]) != NOISE_FILES:
        log(f"Unpacking {NOISE_ZIP}")
        with zipfile.ZipFile(os.path.join(DATA_DIR, NOISE_ZIP)) as zf:
            zf.extractall(DATA_DIR)


def validate_data():
    noise_dir = os.path.join(DATA_DIR, "ICBHI_noise")
    n_noise = len([f for f in os.listdir(noise_dir) if f.endswith(".wav")])
    if n_noise != NOISE_FILES:
        fail(f"expected {NOISE_FILES} .wav files in {noise_dir}, found {n_noise}.")
    con = sqlite3.connect(f"file:{os.path.join(DATA_DIR, DB_NAME)}?mode=ro", uri=True)
    try:
        for cohort, expected in COHORTS.items():
            found = con.execute(
                "SELECT COUNT(*), COUNT(DISTINCT patient_id), "
                "(SELECT COUNT(*) FROM (SELECT patient_id FROM audio_info WHERE project_name = ? "
                "GROUP BY patient_id HAVING MAX(has_murmur) = 1)) "
                "FROM audio_info WHERE project_name = ?", (cohort, cohort)).fetchone()
            if tuple(found) != expected:
                fail(f"{cohort}: expected (recordings, patients, murmur patients) = {expected}, found {tuple(found)}.")
        missing = con.execute("SELECT COUNT(*) FROM audio_info a LEFT JOIN audio_raw_zstd b "
                              "ON a.uuid = b.uuid WHERE b.uuid IS NULL").fetchone()[0]
        if missing:
            fail(f"{missing} recordings in audio_info have no audio.")
    finally:
        con.close()
    log("Data verified: " + ", ".join(f"{c} {n[0]} recordings" for c, n in COHORTS.items())
        + f", {NOISE_FILES} noise files")


def report_device(py):
    code = ("import torch; print('CUDA GPU: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() "
            "else 'Apple GPU (MPS)' if torch.backends.mps.is_available() else 'CPU only')")
    out = subprocess.run([py, "-c", code], capture_output=True, text=True)
    log(f"Device: {out.stdout.strip() or out.stderr.strip()}")


def run_experiments(py):
    os.makedirs(RUN_DIR, exist_ok=True)
    log_path = os.path.join(RUN_DIR, "run.log")
    cmd = [py, PIPELINE, "--db_path", os.path.join(DATA_DIR, DB_NAME),
           "--noise_dir", os.path.join(DATA_DIR, "ICBHI_noise"), "--run_suite"]
    log(f"Running all experiments in {RUN_DIR} (log: {log_path}); trained models are saved in <experiment>/best_model")
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        proc = subprocess.Popen(cmd, cwd=RUN_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            fh.write(line)
        proc.wait()
    return proc.returncode


def compare_with_paper():
    reproduced = os.path.join(RUN_DIR, "global_results.json")
    if not os.path.exists(reproduced):
        log("No results to compare yet.")
        return
    with open(reproduced) as fh:
        new = json.load(fh)
    with open(os.path.join(ROOT, "results", "kaggle_runs_consolidated.csv")) as fh:
        paper = {r["experiment"]: r for r in csv.DictReader(fh)}
    rows, max_diff = [], 0.0
    for name, res in sorted(new.items()):
        if name not in paper or res.get("status") != "SUCCESS":
            continue
        row = {"experiment": name}
        for cohort in ("physionet22_val", "WC2023", "WVU", "OHH-2025"):
            got = (res.get(cohort) or {}).get("auc")
            ref = float(paper[name][f"{cohort}_auc"])
            row[f"{cohort}_paper"] = round(100 * ref, 2)
            row[f"{cohort}_reproduced"] = round(100 * got, 2) if isinstance(got, (int, float)) else ""
            if isinstance(got, (int, float)):
                max_diff = max(max_diff, abs(100 * (got - ref)))
        rows.append(row)
    if not rows:
        log("No finished experiments to compare yet.")
        return
    out = os.path.join(RUN_DIR, "comparison_with_paper.csv")
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log(f"Compared {len(rows)} experiments with the paper (largest AUROC difference {max_diff:.2f} points): {out}")


def run_tests(py):
    run([py, os.path.join(ROOT, "tests", "test_metrics_and_augmentation.py")], cwd=ROOT)


def make_figures(py):
    missing = [e for e in FIGURE_EXPERIMENTS
               if not os.path.exists(os.path.join(RUN_DIR, e, "best_model", "config.json"))]
    if missing:
        fail("--figures uses the models trained by the full run, which are not there yet: "
             + ", ".join(missing) + ". Run ./run.sh (run.bat on Windows) first, then ./run.sh --figures.")
    log("Installing the figure dependencies (figures/requirements.txt)")
    run([py, "-m", "pip", "install", "-r", os.path.join(ROOT, "figures", "requirements.txt")])
    run([py, os.path.join(ROOT, "tests", "test_attribution_helpers.py")], cwd=ROOT)
    out = os.path.join(RUN_DIR, "figures", "attribution")
    log(f"Reproducing the attribution figure (outputs in {out}); this takes several hours on a CPU")
    run([py, os.path.join(ROOT, "figures", "attribution.py"), "all"], cwd=ROOT)
    log(f"Figure written: {os.path.join(out, 'Fig_attribution_left_ABC.pdf')} and Fig_attribution_right_DEF.pdf")


def main():
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Reproduce the experiments in the paper.")
    parser.add_argument("--check", action="store_true",
                        help="set up, download and verify the data, and run the unit tests without training")
    parser.add_argument("--figures", action="store_true",
                        help="after the full run, reproduce the attribution figure from the trained models")
    args = parser.parse_args()

    check_system()
    global SSL_CONTEXT
    py = setup_environment()
    SSL_CONTEXT = make_ssl_context(py)
    fetch_data()
    validate_data()
    report_device(py)
    run_tests(py)
    if args.check:
        log("Check complete.")
        return
    if args.figures:
        make_figures(py)
        return
    code = run_experiments(py)
    compare_with_paper()
    if code != 0:
        fail(f"the pipeline exited with code {code}; see {os.path.join(RUN_DIR, 'run.log')}. "
             "Running the command again resumes from the last finished experiment.")
    log("All experiments finished.")


if __name__ == "__main__":
    main()
