import os
import argparse
import glob
import shutil
import json
import sqlite3
import random
import time
import sys
import subprocess

import importlib.util

REQUIRED_PACKAGES = ["mlflow", "loguru", "optuna", "zstandard", "soundfile", "librosa"]
missing = [pkg for pkg in REQUIRED_PACKAGES if importlib.util.find_spec(pkg) is None]
if missing:
    if os.environ.get("_DEPENDENCIES_INSTALLED") == "1":
        raise ImportError(f"Packages still missing after installation: {missing}")
    print(f"Installing missing packages: {' '.join(missing)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    os.environ["_DEPENDENCIES_INSTALLED"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import mlflow
import zstandard as zstd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torchaudio
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    matthews_corrcoef,
)
import scipy.signal
import csv
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from datasets import Dataset
from loguru import logger
from transformers import (
    EarlyStoppingCallback,
    TrainerCallback,
    Trainer,
    TrainingArguments,
    AutoFeatureExtractor,
)

TARGET_SAMPLE_RATE = 16000
MAX_LEN_SAMPLES = 160000
DEFAULT_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"


def safe_gcs_upload(local_path, bucket_name, gcs_path, gcs_client=None):
    if gcs_client is None:
        return False
    if not os.path.exists(local_path):
        print(f"[GCS] Skipping upload - file not found: {local_path}")
        return False
    try:
        from google.cloud import storage as gcs

        client = gcs_client or gcs.Client()
        client.bucket(bucket_name).blob(gcs_path).upload_from_filename(local_path)
        print(f"[GCS] Uploaded {local_path} -> gs://{bucket_name}/{gcs_path}")
        return True
    except Exception as e:
        print(f"[GCS] Upload failed for {local_path}: {e}")
        return False


# Losses
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.60, gamma=2.0, reduction="mean", label_smoothing=0.0):
        super().__init__()
        self.alpha, self.gamma, self.reduction, self.label_smoothing = (
            alpha,
            gamma,
            reduction,
            label_smoothing,
        )

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(
            logits, targets, reduction="none", label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        alpha_t = torch.where(
            targets == 1,
            torch.tensor(self.alpha, dtype=logits.dtype, device=logits.device),
            torch.tensor(1.0 - self.alpha, dtype=logits.dtype, device=logits.device),
        )
        focal_loss = alpha_t * ((1.0 - pt) ** self.gamma) * ce_loss
        return (
            focal_loss.mean()
            if self.reduction == "mean"
            else focal_loss.sum() if self.reduction == "sum" else focal_loss
        )


def _locate_db(path):
    """Return a path to a SQLite DB that contains an ``audio_info`` table, searching ``path`` (a .db file or a directory, recursively)."""
    if not path:
        return None
    candidates = []
    if os.path.isfile(path) and path.endswith(".db"):
        candidates.append(path)
    elif os.path.isdir(path):
        for root, _dirs, files in os.walk(path, followlinks=True):
            for f in files:
                if f.endswith(".db"):
                    candidates.append(os.path.join(root, f))
    for db in candidates:
        try:
            con = sqlite3.connect(db)
            row = con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='audio_info'"
            ).fetchone()
            con.close()
            if row:
                return db
        except Exception:
            continue
    return None


# Entry point
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db_path",
        type=str,
        default=None,
        help="Path to DB. Required unless using --use_kaggle_dataset",
    )
    parser.add_argument("--ohh_folder", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--effective_batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument(
        "--mil_strategy",
        type=str,
        default="earlyfusionkhead",
        choices=["earlyfusionkhead", "gated_attention", "latefusionkhead"],
    )
    parser.add_argument("--use_focal_loss", action="store_true")
    parser.add_argument("--focal_alpha", type=float, default=0.60)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--unfreeze_top", action="store_true")
    parser.add_argument(
        "--use_lora", action="store_true", help="Use LoRA for PEFT on AST backbone"
    )
    parser.add_argument("--experiment_name", type=str, default="default")
    parser.add_argument("--use_spec_augment", action="store_true")
    parser.add_argument("--use_random_eq", action="store_true")
    parser.add_argument("--use_lung_noise", action="store_true")
    parser.add_argument("--use_polarity_flip", action="store_true")
    parser.add_argument("--use_position_dropout", action="store_true")
    parser.add_argument("--use_freq_mixstyle", action="store_true")
    parser.add_argument("--use_gain_jitter", action="store_true")
    parser.add_argument(
        "--use_pre_emphasis",
        action="store_true",
        help="Apply pre-emphasis filtering (ignored if run_suite is used)",
    )
    parser.add_argument("--lambda_val", type=float, default=None)
    parser.add_argument("--gcs_bucket", type=str, default=None)
    parser.add_argument("--optuna_study_name", type=str, default="prod_study")
    parser.add_argument("--optuna_storage", type=str, default="sqlite:///optuna.db")
    parser.add_argument(
        "--noise_dir",
        type=str,
        default=None,
        help="Path to ICBHI noise folder. Required for noise augmentation experiments.",
    )
    parser.add_argument(
        "--run_suite", action="store_true", help="Run all experiments in the suite"
    )
    parser.add_argument(
        "--run_only",
        type=str,
        nargs="+",
        default=[],
        help="Experiment names to run (e.g. 13_prod_hpss_eq). If empty, all suite experiments run.",
    )
    parser.add_argument("--max_duration_seconds", type=float, default=10.0)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers. Max 4 on Kaggle T4x2 to avoid deadlocks.",
    )
    parser.add_argument(
        "--use_kaggle_dataset",
        action="store_true",
        help="Automatically download or mount the Kaggle dataset",
    )
    parser.add_argument(
        "--resume_dir",
        type=str,
        default=None,
        help="Directory to check for existing results to skip completed experiments",
    )
    parser.add_argument(
        "--skip_model_save",
        action="store_true",
        help="Do not save the trained model (metrics and logs are still written)",
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Dry run: first experiment only, 2 epochs",
    )
    args = parser.parse_args()
    run_start = time.time()

    print(
        f"Unified murmur pipeline, {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        flush=True,
    )
    print(f"Python {sys.version.split()[0]} | PyTorch {torch.__version__}", flush=True)
    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
            flush=True,
        )
    elif torch.backends.mps.is_available():
        print("Device: Apple MPS", flush=True)
    else:
        print("Device: CPU (no GPU detected)", flush=True)
    print(f"Args: {vars(args)}", flush=True)

    AIP_MODEL_DIR = os.environ.get("AIP_MODEL_DIR", None)
    is_vertex = AIP_MODEL_DIR is not None

    if is_vertex:
        print("[Vertex AI] Detected. Using local /tmp/results for intermediate files.")
        output_dir = "/tmp/results"
    elif os.path.exists("/kaggle/working"):
        print("[Kaggle] Detected /kaggle/working directory. Using it as output_dir.")
        output_dir = "/kaggle/working"
    else:
        output_dir = "."
    os.makedirs(output_dir, exist_ok=True)
    resume_dir = args.resume_dir if args.resume_dir else output_dir

    if args.use_kaggle_dataset:
        print(
            "[Kaggle] --use_kaggle_dataset specified. Attempting to locate or download dataset...",
            flush=True,
        )
        dataset_path = None

        candidates = [
            "/kaggle/input/mumurdetection",
            "/kaggle/input/someshbsaini/mumurdetection",
        ]

        if os.path.exists("/kaggle/input"):
            _found_db = _locate_db("/kaggle/input")
            if _found_db:
                candidates.insert(0, os.path.dirname(_found_db))

        for cand in candidates:
            if os.path.exists(cand):
                print(f"[Kaggle] Native Kaggle mount found at {cand}", flush=True)
                dataset_path = cand
                break

        if not dataset_path:
            try:
                import kagglehub

                print(
                    "[Kaggle] Downloading someshbsaini/mumurdetection via kagglehub...",
                    flush=True,
                )
                dataset_path = kagglehub.dataset_download("someshbsaini/mumurdetection")
                print(f"[Kaggle] Dataset downloaded to {dataset_path}", flush=True)
            except Exception as e:
                print(f"[Kaggle] kagglehub download failed: {e}")
                print(
                    "\n[ERROR] If you are running this in a non-interactive (background/Save Version) Kaggle session, "
                    "dynamic downloads are blocked by Kaggle. You must manually attach the dataset 'someshbsaini/mumurdetection' "
                    "in the Kaggle Notebook settings (Add Input -> search for 'someshbsaini/mumurdetection').",
                    flush=True,
                )
                sys.exit(1)

        if dataset_path:
            db_cand = _locate_db(dataset_path)
            if db_cand:
                args.db_path = db_cand
                print(f"[Kaggle] Auto-assigned db_path: {args.db_path}", flush=True)

            ohh_cand_1 = os.path.join(
                dataset_path, "ohh_2025_dataset", "onehearthealth_2025_per_patient"
            )
            ohh_cand_2 = os.path.join(dataset_path, "onehearthealth_2025_per_patient")
            ohh_zip_cand = os.path.join(dataset_path, "ohh_2025_dataset.zip")
            if os.path.exists(ohh_cand_1):
                args.ohh_folder = ohh_cand_1
            elif os.path.exists(ohh_cand_2):
                args.ohh_folder = ohh_cand_2
            elif os.path.exists(ohh_zip_cand):
                import zipfile

                print(
                    f"[Kaggle] Found zipped OHH dataset at {ohh_zip_cand}. Extracting to /tmp/ohh_unzipped...",
                    flush=True,
                )
                with zipfile.ZipFile(ohh_zip_cand, "r") as zip_ref:
                    zip_ref.extractall("/tmp/ohh_unzipped")

                extracted_root = "/tmp/ohh_unzipped"
                for sub in os.listdir(extracted_root):
                    cand = os.path.join(extracted_root, sub)
                    if os.path.isdir(cand) and os.path.isdir(
                        os.path.join(cand, "Abnormal")
                    ):
                        extracted_root = cand
                        break
                args.ohh_folder = extracted_root
                print(
                    f"[Kaggle] Extracted OHH dataset ready at {args.ohh_folder}",
                    flush=True,
                )

            if args.ohh_folder:
                print(
                    f"[Kaggle] Auto-assigned ohh_folder: {args.ohh_folder}", flush=True
                )

            noise_cand_1 = os.path.join(dataset_path, "ICBHI_noise")
            noise_cand_2 = os.path.join(dataset_path, "noise")
            noise_cand_3 = os.path.join(dataset_path, "ICBHI2017")
            noise_cand_3_nested = os.path.join(dataset_path, "ICBHI2017", "ICBHI2017")
            noise_cand_4 = "/kaggle/input/icbhi-noise-dataset/ICBHI2017"
            noise_cand_5 = "/kaggle/input/icbhi-noise-dataset"
            noise_zip_cand = os.path.join(dataset_path, "ICBHI2017.zip")

            if os.path.exists(noise_cand_1):
                args.noise_dir = noise_cand_1
                print(f"[Kaggle] Auto-assigned noise_dir: {args.noise_dir}", flush=True)
            elif os.path.exists(noise_cand_2):
                args.noise_dir = noise_cand_2
                print(f"[Kaggle] Auto-assigned noise_dir: {args.noise_dir}", flush=True)
            elif os.path.exists(noise_cand_3_nested):
                args.noise_dir = noise_cand_3_nested
                print(
                    f"[Kaggle] Auto-assigned noise_dir: {args.noise_dir} (Nested)",
                    flush=True,
                )
            elif os.path.exists(noise_cand_3):
                args.noise_dir = noise_cand_3
                print(f"[Kaggle] Auto-assigned noise_dir: {args.noise_dir}", flush=True)
            elif os.path.exists(noise_cand_4):
                args.noise_dir = noise_cand_4
                print(f"[Kaggle] Auto-assigned noise_dir: {args.noise_dir}", flush=True)
            elif os.path.exists(noise_cand_5):
                args.noise_dir = noise_cand_5
                print(f"[Kaggle] Auto-assigned noise_dir: {args.noise_dir}", flush=True)
            elif os.path.exists(noise_zip_cand):
                import zipfile

                print(
                    f"[Kaggle] Found zipped ICBHI dataset at {noise_zip_cand}. Extracting to /tmp/icbhi_unzipped...",
                    flush=True,
                )
                with zipfile.ZipFile(noise_zip_cand, "r") as zip_ref:
                    zip_ref.extractall("/tmp/icbhi_unzipped")

                extracted_root = "/tmp/icbhi_unzipped"
                for sub in os.listdir(extracted_root):
                    cand = os.path.join(extracted_root, sub)
                    if os.path.isdir(cand) and any(
                        f.endswith(".wav") for f in os.listdir(cand)
                    ):
                        extracted_root = cand
                        break
                args.noise_dir = extracted_root
                print(
                    f"[Kaggle] Extracted ICBHI dataset ready at {args.noise_dir}",
                    flush=True,
                )
            else:
                print(
                    "[Kaggle] Noise folder not found in mounted dataset. Downloading directly from public GCS bucket...",
                    flush=True,
                )
                import subprocess

                subprocess.run(
                    [
                        "gsutil",
                        "-m",
                        "cp",
                        "-r",
                        "gs://heart-quality-training-78e/data/ICBHI2017",
                        "/tmp/ICBHI_noise",
                    ]
                )
                args.noise_dir = "/tmp/ICBHI_noise/ICBHI2017"
                print(
                    f"[Kaggle] Auto-assigned downloaded noise_dir: {args.noise_dir}",
                    flush=True,
                )

    if args.db_path is None:
        print(
            "ERROR: --db_path is required if --use_kaggle_dataset is not used or auto-detection failed."
        )
        sys.exit(1)

    gcs_client = None
    if args.gcs_bucket:
        try:
            from google.cloud import storage as gcs

            gcs_client = gcs.Client()
            print(
                f"[GCS] Syncing previous results from gs://{args.gcs_bucket}/results/ to {resume_dir} ...",
                flush=True,
            )
            bucket = gcs_client.bucket(args.gcs_bucket)
            blobs = bucket.list_blobs(prefix="results/results_")
            for blob in blobs:
                local_path = os.path.join(resume_dir, os.path.basename(blob.name))
                if not os.path.exists(local_path):
                    blob.download_to_filename(local_path)
                    print(f"[GCS] Downloaded {blob.name}", flush=True)
            global_blob = bucket.blob("results/global_results.json")
            if global_blob.exists():
                global_local = os.path.join(resume_dir, "global_results.json")
                if not os.path.exists(global_local):
                    global_blob.download_to_filename(global_local)
                    print("[GCS] Downloaded global_results.json", flush=True)
        except Exception as e:
            print(
                f"[GCS] Client init or sync failed: {e}. GCS uploads/sync will be skipped."
            )

    if args.gcs_bucket:
        if not os.path.exists(args.db_path):
            try:
                if gcs_client is None:
                    from google.cloud import storage as gcs

                    gcs_client = gcs.Client()
                print(
                    f"Database {args.db_path} not found locally. Searching gs://{args.gcs_bucket}/{args.db_path}..."
                )
                local_db = "/tmp/data.db"
                gcs_client.bucket(args.gcs_bucket).blob(
                    args.db_path
                ).download_to_filename(local_db)
                args.db_path = local_db
                print(f"Successfully downloaded DB to {local_db}")
            except Exception as e:
                print(
                    f"[GCS] Database download failed: {e}"
                )
        else:
            print(f"Database {args.db_path} found locally. Skipping download.")

    if os.path.isdir(args.db_path):
        args.db_path = _locate_db(args.db_path) or args.db_path
    if not os.path.isfile(args.db_path):
        print(f"ERROR: database not found: {args.db_path}", flush=True)
        sys.exit(1)

    if args.ohh_folder and args.gcs_bucket:
        if not os.path.exists(args.ohh_folder):
            zip_name = "ohh_2025_dataset.zip"
            print(
                f"OHH folder {args.ohh_folder} not found. Searching for {zip_name} in GCS..."
            )
            try:
                from google.cloud import storage as gcs
                import zipfile

                client = gcs.Client()
                local_zip = "/tmp/ohh_data.zip"
                client.bucket(args.gcs_bucket).blob(zip_name).download_to_filename(
                    local_zip
                )
                print(f"Unzipping {zip_name} to /tmp/ohh_unzipped...")
                with zipfile.ZipFile(local_zip, "r") as zip_ref:
                    zip_ref.extractall("/tmp/ohh_unzipped")
                args.ohh_folder = "/tmp/ohh_unzipped"
                for sub in os.listdir(args.ohh_folder):
                    candidate = os.path.join(args.ohh_folder, sub)
                    if os.path.isdir(candidate) and os.path.isdir(
                        os.path.join(candidate, "Abnormal")
                    ):
                        print(f"Detected inner OHH folder: {candidate}")
                        args.ohh_folder = candidate
                        break
                print(f"OHH Dataset ready at {args.ohh_folder}")
            except Exception as e:
                print(
                    f"[GCS] OHH Dataset setup failed: {e}. OHH cohort will be skipped."
                )
    elif not args.ohh_folder:
        print("--ohh_folder not set; OHH-2025 is read from the database.")

    noise_rng = random.Random(43)
    if args.noise_dir and os.path.exists(args.noise_dir):
        noise_files = sorted(
            [
                os.path.join(args.noise_dir, f)
                for f in os.listdir(args.noise_dir)
                if f.endswith(".wav")
            ]
        )
    else:
        noise_files = []
        if args.run_suite:
            print(
                "[WARNING] --noise_dir not set or not found; noise augmentation is disabled."
            )
    noise_rng.shuffle(noise_files)
    n_n = len(noise_files)
    noise_splits = {
        "train": noise_files[: int(n_n * 0.8)],
        "test": noise_files[int(n_n * 0.8) :],
    }

    patient_rng = random.Random(42)
    with sqlite3.connect(args.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT patient_id, MAX(has_murmur) FROM audio_info WHERE project_name = 'physionet22' COLLATE NOCASE GROUP BY patient_id"
        )
        all_p = [(r[0], r[1], "physionet22") for r in cursor.fetchall()]
    murmur_p = [p for p in all_p if p[1] == 1]
    normal_p = [p for p in all_p if p[1] == 0]
    patient_rng.shuffle(murmur_p)
    patient_rng.shuffle(normal_p)
    m_split = int(len(murmur_p) * 0.8)
    n_split = int(len(normal_p) * 0.8)
    train_p = murmur_p[:m_split] + normal_p[:n_split]
    val_p = murmur_p[m_split:] + normal_p[n_split:]
    patient_rng.shuffle(train_p)
    patient_rng.shuffle(val_p)

    train_ids = {p[0] for p in train_p}
    val_ids = {p[0] for p in val_p}
    overlap = train_ids.intersection(val_ids)
    assert len(overlap) == 0, f"DATA LEAKAGE DETECTED: {len(overlap)} patients overlap."
    train_pos = sum(1 for p in train_p if p[1] == 1)
    val_pos = sum(1 for p in val_p if p[1] == 1)
    integrity_status = f"Verified: {len(train_ids)} train ({train_pos} pos, {len(train_ids)-train_pos} neg), {len(val_ids)} val ({val_pos} pos, {len(val_ids)-val_pos} neg), 0 overlap."
    print(integrity_status)

    if getattr(args, "test_mode", False):
        args.epochs = 2

    HA = 0.6
    HG = 2.0

    experiments = []
    if args.run_suite:
        experiments = [
            # Zero-shot baselines
            {
                "name": "00_dryrun_audioset_zeroshot",
                "zero_shot_pretrained": True,
                "prod_processing": False,
                "mil": "N/A",
                "cfg": {},
            },
            {
                "name": "00b_dryrun_audioset_zeroshot_stdpreproc",
                "zero_shot_pretrained": True,
                "prod_processing": False,
                "mil": "N/A",
                "cfg": {"use_standard_preprocessing": True},
            },
            # Backbone adaptation
            {"name": "01_prod_processing_trainer_frozen",     "prod_processing": True, "mil": "N/A", "focal": False, "fa": 0.6, "fg": 2.0, "cfg": {}, "unfreeze": False},
            {"name": "02_prod_processing_trainer_unfrozen",   "prod_processing": True, "mil": "N/A", "focal": False, "fa": 0.6, "fg": 2.0, "cfg": {}, "unfreeze": True},
            {"name": "03_prod_processing_trainer_progressive", "prod_processing": True, "mil": "N/A", "focal": False, "fa": 0.6, "fg": 2.0, "cfg": {}, "unfreeze": False, "progressive_unfreeze": True},
            # Single components
            {
                "name": "04_prod_focal_loss",
                "prod_processing": True,
                "mil": "N/A",
                "focal": True,
                "fa": HA,
                "fg": HG,
                "cfg": {},
                "unfreeze": True,
            },
            {
                "name": "05_prod_spec_augment",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {"use_spec_augment": True},
                "unfreeze": True,
            },
            {
                "name": "06_prod_noise_025",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {"use_lung_noise": True, "lambda_val": 0.25},
                "unfreeze": True,
            },
            {
                "name": "07_prod_noise_100",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {"use_lung_noise": True, "lambda_val": 1.0},
                "unfreeze": True,
            },
            {
                "name": "08_prod_noise_500",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {"use_lung_noise": True, "lambda_val": 5.0},
                "unfreeze": True,
            },
            {
                "name": "09_prod_noise_mixed",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {"use_lung_noise": True, "lambda_val": None},
                "unfreeze": True,
            },
            {
                "name": "10_prod_polarity_flip",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {"use_polarity_flip": True},
                "unfreeze": True,
            },
            {
                "name": "11_prod_gain_jitter",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {"use_gain_jitter": True},
                "unfreeze": True,
            },
            {"name": "11b_prod_gain_jitter_mild",    "prod_processing": True, "mil": "N/A", "focal": False, "fa": 0.6, "fg": 2.0, "cfg": {"use_gain_jitter": True, "gain_jitter_range": 0.15}, "unfreeze": True},
            {"name": "11c_prod_gain_jitter_extreme", "prod_processing": True, "mil": "N/A", "focal": False, "fa": 0.6, "fg": 2.0, "cfg": {"use_gain_jitter": True, "gain_jitter_range": 0.5}, "unfreeze": True},
            {
                "name": "12_prod_mixup",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {"use_mixup": True},
                "unfreeze": True,
            },
            {
                "name": "13_prod_hpss_eq",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {"use_hpss_eq": True},
                "unfreeze": True,
            },
            {
                "name": "14_prod_cardiac_crop",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {"use_cardiac_crop": True},
                "unfreeze": True,
            },
            {
                "name": "15_prod_resp_am",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {"use_resp_am": True},
                "unfreeze": True,
            },
            {
                "name": "16_prod_quality_weight",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {"use_quality_weight": True},
                "unfreeze": True,
            },
            {
                "name": "17_prod_spec_moderate",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {
                    "use_spec_augment": True,
                    "spec_freq_mask": 24,
                    "spec_time_mask": 80,
                },
                "unfreeze": True,
            },
            {
                "name": "18_prod_spec_aggressive",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {
                    "use_spec_augment": True,
                    "spec_freq_mask": 40,
                    "spec_time_mask": 100,
                },
                "unfreeze": True,
            },
            {
                "name": "19_prod_spec_extreme",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {
                    "use_spec_augment": True,
                    "spec_freq_mask": 48,
                    "spec_time_mask": 150,
                },
                "unfreeze": True,
            },
            {
                "name": "20_prod_cspec_light",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {
                    "use_spec_augment": True,
                    "spec_cardiac_constrained": True,
                    "spec_freq_mask": 8,
                    "spec_time_mask": 48,
                },
                "unfreeze": True,
            },
            {
                "name": "21_prod_cspec_moderate",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {
                    "use_spec_augment": True,
                    "spec_cardiac_constrained": True,
                    "spec_freq_mask": 15,
                    "spec_time_mask": 80,
                },
                "unfreeze": True,
            },
            {
                "name": "22_prod_cspec_aggressive",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {
                    "use_spec_augment": True,
                    "spec_cardiac_constrained": True,
                    "spec_freq_mask": 24,
                    "spec_time_mask": 100,
                },
                "unfreeze": True,
            },
            {
                "name": "23_prod_cspec_ultra_light",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {
                    "use_spec_augment": True,
                    "spec_cardiac_constrained": True,
                    "spec_freq_mask": 5,
                    "spec_time_mask": 48,
                },
                "unfreeze": True,
            },
            # Two-component combinations
            {"name": "24_prod_spec_noisemixed",   "prod_processing": True, "mil": "N/A", "focal": False, "fa": 0.6, "fg": 2.0, "cfg": {"use_spec_augment": True, "use_lung_noise": True, "lambda_val": None}, "unfreeze": True},
            {"name": "25_prod_spec_noise025",    "prod_processing": True, "mil": "N/A", "focal": False, "fa": 0.6, "fg": 2.0, "cfg": {"use_spec_augment": True, "use_lung_noise": True, "lambda_val": 0.25}, "unfreeze": True},
            {"name": "26_prod_spec_noise100",    "prod_processing": True, "mil": "N/A", "focal": False, "fa": 0.6, "fg": 2.0, "cfg": {"use_spec_augment": True, "use_lung_noise": True, "lambda_val": 1.0}, "unfreeze": True},
            {"name": "27_prod_spec_noise500",    "prod_processing": True, "mil": "N/A", "focal": False, "fa": 0.6, "fg": 2.0, "cfg": {"use_spec_augment": True, "use_lung_noise": True, "lambda_val": 5.0}, "unfreeze": True},
            {"name": "28_prod_spec_focal",        "prod_processing": True, "mil": "N/A", "focal": True,  "fa": HA,  "fg": HG, "cfg": {"use_spec_augment": True}, "unfreeze": True},
            {"name": "29_prod_noisemixed_focal",  "prod_processing": True, "mil": "N/A", "focal": True,  "fa": HA,  "fg": HG, "cfg": {"use_lung_noise": True, "lambda_val": None}, "unfreeze": True},
            {"name": "30_prod_noise025_focal",   "prod_processing": True, "mil": "N/A", "focal": True,  "fa": HA,  "fg": HG, "cfg": {"use_lung_noise": True, "lambda_val": 0.25}, "unfreeze": True},
            {"name": "31_prod_noise100_focal",   "prod_processing": True, "mil": "N/A", "focal": True,  "fa": HA,  "fg": HG, "cfg": {"use_lung_noise": True, "lambda_val": 1.0}, "unfreeze": True},
            {"name": "32_prod_noise500_focal",   "prod_processing": True, "mil": "N/A", "focal": True,  "fa": HA,  "fg": HG, "cfg": {"use_lung_noise": True, "lambda_val": 5.0}, "unfreeze": True},
            {
                "name": "33_prod_hpss_noisemixed",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": HA,
                "fg": HG,
                "cfg": {
                    "use_hpss_eq": True,
                    "use_lung_noise": True,
                    "lambda_val": None,
                },
                "unfreeze": True,
            },
            {
                "name": "34_prod_hpss_noise025",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": HA,
                "fg": HG,
                "cfg": {
                    "use_hpss_eq": True,
                    "use_lung_noise": True,
                    "lambda_val": 0.25,
                },
                "unfreeze": True,
            },
            {
                "name": "35_prod_hpss_noise100",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": HA,
                "fg": HG,
                "cfg": {"use_hpss_eq": True, "use_lung_noise": True, "lambda_val": 1.0},
                "unfreeze": True,
            },
            {
                "name": "36_prod_hpss_noise500",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": HA,
                "fg": HG,
                "cfg": {"use_hpss_eq": True, "use_lung_noise": True, "lambda_val": 5.0},
                "unfreeze": True,
            },
            {
                "name": "37_prod_hpss_spec",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": HA,
                "fg": HG,
                "cfg": {"use_hpss_eq": True, "use_spec_augment": True},
                "unfreeze": True,
            },
            {
                "name": "38_prod_hpss_cspec_ultra_light",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": HA,
                "fg": HG,
                "cfg": {
                    "use_hpss_eq": True,
                    "use_spec_augment": True,
                    "spec_cardiac_constrained": True,
                    "spec_freq_mask": 5,
                    "spec_time_mask": 48,
                },
                "unfreeze": True,
            },
            {
                "name": "39_prod_hpss_focal",
                "prod_processing": True,
                "mil": "N/A",
                "focal": True,
                "fa": HA,
                "fg": HG,
                "cfg": {"use_hpss_eq": True},
                "unfreeze": True,
            },
            {
                "name": "40_prod_cspec_ultra_light_noisemixed",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": HA,
                "fg": HG,
                "cfg": {
                    "use_spec_augment": True,
                    "spec_cardiac_constrained": True,
                    "spec_freq_mask": 5,
                    "spec_time_mask": 48,
                    "use_lung_noise": True,
                    "lambda_val": None,
                },
                "unfreeze": True,
            },
            {
                "name": "41_prod_cspec_ultra_light_noise025",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": HA,
                "fg": HG,
                "cfg": {
                    "use_spec_augment": True,
                    "spec_cardiac_constrained": True,
                    "spec_freq_mask": 5,
                    "spec_time_mask": 48,
                    "use_lung_noise": True,
                    "lambda_val": 0.25,
                },
                "unfreeze": True,
            },
            {
                "name": "42_prod_cspec_ultra_light_noise100",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": HA,
                "fg": HG,
                "cfg": {
                    "use_spec_augment": True,
                    "spec_cardiac_constrained": True,
                    "spec_freq_mask": 5,
                    "spec_time_mask": 48,
                    "use_lung_noise": True,
                    "lambda_val": 1.0,
                },
                "unfreeze": True,
            },
            {
                "name": "43_prod_cspec_ultra_light_noise500",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": HA,
                "fg": HG,
                "cfg": {
                    "use_spec_augment": True,
                    "spec_cardiac_constrained": True,
                    "spec_freq_mask": 5,
                    "spec_time_mask": 48,
                    "use_lung_noise": True,
                    "lambda_val": 5.0,
                },
                "unfreeze": True,
            },
            {
                "name": "44_prod_cspec_ultra_light_focal",
                "prod_processing": True,
                "mil": "N/A",
                "focal": True,
                "fa": HA,
                "fg": HG,
                "cfg": {
                    "use_spec_augment": True,
                    "spec_cardiac_constrained": True,
                    "spec_freq_mask": 5,
                    "spec_time_mask": 48,
                },
                "unfreeze": True,
            },
            {
                "name": "45_prod_gain_jitter_noisemixed",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {
                    "use_gain_jitter": True,
                    "use_lung_noise": True,
                    "lambda_val": None,
                },
                "unfreeze": True,
            },
            {
                "name": "46_prod_gain_jitter_noise025",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {
                    "use_gain_jitter": True,
                    "use_lung_noise": True,
                    "lambda_val": 0.25,
                },
                "unfreeze": True,
            },
            {
                "name": "47_prod_gain_jitter_noise100",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {
                    "use_gain_jitter": True,
                    "use_lung_noise": True,
                    "lambda_val": 1.0,
                },
                "unfreeze": True,
            },
            {
                "name": "48_prod_gain_jitter_noise500",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {
                    "use_gain_jitter": True,
                    "use_lung_noise": True,
                    "lambda_val": 5.0,
                },
                "unfreeze": True,
            },
            {
                "name": "49_prod_gain_jitter_focal",
                "prod_processing": True,
                "mil": "N/A",
                "focal": True,
                "fa": HA,
                "fg": HG,
                "cfg": {"use_gain_jitter": True},
                "unfreeze": True,
            },
            {
                "name": "50_prod_gain_jitter_hpss",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {"use_gain_jitter": True, "use_hpss_eq": True},
                "unfreeze": True,
            },
            {
                "name": "51_prod_gain_jitter_cardiac_crop",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {"use_gain_jitter": True, "use_cardiac_crop": True},
                "unfreeze": True,
            },
            {
                "name": "52_prod_gain_jitter_spec",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {"use_gain_jitter": True, "use_spec_augment": True},
                "unfreeze": True,
            },
            {
                "name": "53_prod_gain_jitter_cspec_ultra_light",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {
                    "use_gain_jitter": True,
                    "use_spec_augment": True,
                    "spec_cardiac_constrained": True,
                    "spec_freq_mask": 5,
                    "spec_time_mask": 48,
                },
                "unfreeze": True,
            },
            {
                "name": "54_prod_gain_jitter_quality_weight",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {"use_gain_jitter": True, "use_quality_weight": True},
                "unfreeze": True,
            },
            {
                "name": "55_prod_cardiac_crop_noisemixed",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {
                    "use_cardiac_crop": True,
                    "use_lung_noise": True,
                    "lambda_val": None,
                },
                "unfreeze": True,
            },
            {
                "name": "56_prod_cardiac_crop_noise025",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {
                    "use_cardiac_crop": True,
                    "use_lung_noise": True,
                    "lambda_val": 0.25,
                },
                "unfreeze": True,
            },
            {
                "name": "57_prod_cardiac_crop_noise100",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {
                    "use_cardiac_crop": True,
                    "use_lung_noise": True,
                    "lambda_val": 1.0,
                },
                "unfreeze": True,
            },
            {
                "name": "58_prod_cardiac_crop_noise500",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {
                    "use_cardiac_crop": True,
                    "use_lung_noise": True,
                    "lambda_val": 5.0,
                },
                "unfreeze": True,
            },
            {
                "name": "59_prod_cardiac_crop_focal",
                "prod_processing": True,
                "mil": "N/A",
                "focal": True,
                "fa": HA,
                "fg": HG,
                "cfg": {"use_cardiac_crop": True},
                "unfreeze": True,
            },
            {
                "name": "60_prod_cardiac_crop_hpss",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {"use_cardiac_crop": True, "use_hpss_eq": True},
                "unfreeze": True,
            },
            {
                "name": "61_prod_cardiac_crop_spec",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {"use_cardiac_crop": True, "use_spec_augment": True},
                "unfreeze": True,
            },
            {
                "name": "62_prod_cardiac_crop_cspec_ultra_light",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {
                    "use_cardiac_crop": True,
                    "use_spec_augment": True,
                    "spec_cardiac_constrained": True,
                    "spec_freq_mask": 5,
                    "spec_time_mask": 48,
                },
                "unfreeze": True,
            },
            {
                "name": "63_prod_cardiac_crop_quality_weight",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {"use_cardiac_crop": True, "use_quality_weight": True},
                "unfreeze": True,
            },
            {
                "name": "64_prod_quality_weight_noisemixed",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {
                    "use_quality_weight": True,
                    "use_lung_noise": True,
                    "lambda_val": None,
                },
                "unfreeze": True,
            },
            {
                "name": "65_prod_quality_weight_noise025",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {
                    "use_quality_weight": True,
                    "use_lung_noise": True,
                    "lambda_val": 0.25,
                },
                "unfreeze": True,
            },
            {
                "name": "66_prod_quality_weight_noise100",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {
                    "use_quality_weight": True,
                    "use_lung_noise": True,
                    "lambda_val": 1.0,
                },
                "unfreeze": True,
            },
            {
                "name": "67_prod_quality_weight_noise500",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {
                    "use_quality_weight": True,
                    "use_lung_noise": True,
                    "lambda_val": 5.0,
                },
                "unfreeze": True,
            },
            {
                "name": "68_prod_quality_weight_focal",
                "prod_processing": True,
                "mil": "N/A",
                "focal": True,
                "fa": HA,
                "fg": HG,
                "cfg": {"use_quality_weight": True},
                "unfreeze": True,
            },
            {
                "name": "69_prod_quality_weight_hpss",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {"use_quality_weight": True, "use_hpss_eq": True},
                "unfreeze": True,
            },
            {
                "name": "70_prod_quality_weight_spec",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {"use_quality_weight": True, "use_spec_augment": True},
                "unfreeze": True,
            },
            {
                "name": "71_prod_quality_weight_cspec_ultra_light",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {
                    "use_quality_weight": True,
                    "use_spec_augment": True,
                    "spec_cardiac_constrained": True,
                    "spec_freq_mask": 5,
                    "spec_time_mask": 48,
                },
                "unfreeze": True,
            },
            # Three-component combinations
            {
                "name": "72_prod_hpss_noisemixed_focal",
                "prod_processing": True,
                "mil": "N/A",
                "focal": True,
                "fa": HA,
                "fg": HG,
                "cfg": {"use_hpss_eq": True, "use_lung_noise": True, "lambda_val": None},
                "unfreeze": True,
            },
            {
                "name": "73_prod_hpss_noise025_focal",
                "prod_processing": True,
                "mil": "N/A",
                "focal": True,
                "fa": HA,
                "fg": HG,
                "cfg": {"use_hpss_eq": True, "use_lung_noise": True, "lambda_val": 0.25},
                "unfreeze": True,
            },
            {
                "name": "74_prod_gain_jitter_cspec_ultra_light_noise100",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {
                    "use_gain_jitter": True,
                    "use_spec_augment": True,
                    "spec_cardiac_constrained": True,
                    "spec_freq_mask": 5,
                    "spec_time_mask": 48,
                    "use_lung_noise": True,
                    "lambda_val": 1.0,
                },
                "unfreeze": True,
            },
            {
                "name": "75_prod_gain_jitter_cspec_ultra_light_noise025",
                "prod_processing": True,
                "mil": "N/A",
                "focal": False,
                "fa": 0.6,
                "fg": 2.0,
                "cfg": {
                    "use_gain_jitter": True,
                    "use_spec_augment": True,
                    "spec_cardiac_constrained": True,
                    "spec_freq_mask": 5,
                    "spec_time_mask": 48,
                    "use_lung_noise": True,
                    "lambda_val": 0.25,
                },
                "unfreeze": True,
            },
            {
                "name": "76_prod_hpss_noise100_focal",
                "prod_processing": True,
                "mil": "N/A",
                "focal": True,
                "fa": HA,
                "fg": HG,
                "cfg": {"use_hpss_eq": True, "use_lung_noise": True, "lambda_val": 1.0},
                "unfreeze": True,
            },
            {
                "name": "77_prod_cardiac_crop_noisemixed_focal",
                "prod_processing": True,
                "mil": "N/A",
                "focal": True,
                "fa": HA,
                "fg": HG,
                "cfg": {
                    "use_cardiac_crop": True,
                    "use_lung_noise": True,
                    "lambda_val": None,
                },
                "unfreeze": True,
            },
            {
                "name": "78_prod_gain_jitter_hpss_focal",
                "prod_processing": True,
                "mil": "N/A",
                "focal": True,
                "fa": HA,
                "fg": HG,
                "cfg": {"use_gain_jitter": True, "use_hpss_eq": True},
                "unfreeze": True,
            },
        ]
    else:
        experiments = [
            {
                "name": args.experiment_name,
                "mil": args.mil_strategy,
                "focal": args.use_focal_loss,
                "fa": args.focal_alpha,
                "fg": args.focal_gamma,
                "cfg": {
                    "use_spec_augment": args.use_spec_augment,
                    "spec_freq_mask": getattr(args, "spec_freq_mask", 12),
                    "spec_time_mask": getattr(args, "spec_time_mask", 48),
                    "spec_cardiac_constrained": getattr(
                        args, "spec_cardiac_constrained", False
                    ),
                    "use_random_eq": args.use_random_eq,
                    "use_lung_noise": args.use_lung_noise,
                    "lambda_val": args.lambda_val,
                    "use_pre_emphasis": getattr(args, "use_pre_emphasis", False),
                    "use_polarity_flip": args.use_polarity_flip,
                    "use_position_dropout": args.use_position_dropout,
                    "use_freq_mixstyle": getattr(args, "use_freq_mixstyle", False),
                    "use_gain_jitter": getattr(args, "use_gain_jitter", False),
                    "use_hpss_eq": getattr(args, "use_hpss_eq", False),
                    "use_mixup": getattr(args, "use_mixup", False),
                    "mixup_alpha": getattr(args, "mixup_alpha", 0.4),
                },
                "unfreeze": args.unfreeze_top,
            }
        ]

    global_results = {}

    resume_global_path = os.path.join(resume_dir, "global_results.json")
    if os.path.exists(resume_global_path):
        try:
            with open(resume_global_path, "r") as f:
                prev_global = json.load(f)
                global_results.update(prev_global)
                print(
                    f"[Resume] Loaded {len(prev_global)} previous experiments from {resume_global_path}",
                    flush=True,
                )
        except Exception as e:
            print(
                f"[Resume] Warning: failed to parse previous global_results.json: {e}",
                flush=True,
            )

    if getattr(args, "run_only", None):
        unknown = sorted(set(args.run_only) - {e["name"] for e in experiments})
        if unknown:
            print(f"ERROR: unknown experiment name(s): {', '.join(unknown)}", flush=True)
            sys.exit(1)
        experiments = [e for e in experiments if e["name"] in args.run_only]

    if any(e.get("cfg", {}).get("use_lung_noise") for e in experiments) and not noise_splits["train"]:
        print("ERROR: the selected experiments add respiratory noise, but no .wav files were found "
              "in --noise_dir.", flush=True)
        sys.exit(1)

    if getattr(args, "test_mode", False):
        experiments = experiments[:1]
        print(
            f"[TEST MODE] Running only {experiments[0]['name']} for 2 epochs",
            flush=True,
        )

    executed_count = 0
    for exp_idx, exp in enumerate(experiments):
        elapsed_time = time.time() - run_start
        if os.path.exists("/kaggle/working") and elapsed_time > (10 * 3600):
            print(
                "\n[QUOTA WARNING] 10 hours elapsed. Exiting so that Kaggle saves the outputs.",
                flush=True,
            )
            break

        if getattr(args, "run_only", None) and exp["name"] not in args.run_only:
            continue
        print(
            f"\nExperiment {exp_idx+1}/{len(experiments)}: {exp['name']}",
            flush=True,
        )

        resume_file = os.path.join(resume_dir, f"results_{exp['name']}.json")
        if os.path.exists(resume_file):
            try:
                with open(resume_file, "r") as f:
                    prev_results = json.load(f)
                if prev_results.get("status") == "SUCCESS":
                    print(
                        f"[Resume] Skipping {exp['name']} - found successful completion in {resume_file}.",
                        flush=True,
                    )
                    global_results[exp["name"]] = prev_results
                    continue
            except Exception as e:
                print(
                    f"[Resume] Warning: Could not read {resume_file}. Re-running experiment. Error: {e}",
                    flush=True,
                )

        if exp.get("zero_shot_pretrained", False):
            import logging

            logging.info(f"Running zero-shot dry-run experiment: {exp['name']}")
            exp_start = time.time()
            try:
                results = run_zero_shot_audioset_baseline(args, exp, output_dir)
                results["status"] = "SUCCESS"
                results["name"] = exp["name"]
            except Exception as e:
                print(
                    f"\n[ERROR] Zero-shot experiment {exp['name']} failed: {e}",
                    flush=True,
                )
                results = {"name": exp["name"], "status": "FAILED", "error": str(e)}

            results["split_integrity"] = "N/A (no training/eval split - zero-shot)"
            results["wall_time_seconds"] = round(time.time() - exp_start, 1)
            global_results[exp["name"]] = results
            executed_count += 1
            print(
                f"[DONE] {exp['name']} | status={results['status']} | cal_score={results.get('best_cal_score', 0):.4f} | wall_time={results['wall_time_seconds']:.0f}s",
                flush=True,
            )

            result_path = os.path.join(output_dir, f"results_{exp['name']}.json")
            with open(result_path, "w") as f:
                json.dump(results, f, indent=2)
            continue

        if exp.get("prod_processing", False):
            import logging

            logging.info(f"Running PROD processing experiment: {exp['name']}")
            exp_start = time.time()
            try:
                results = run_prod_pipeline_experiment(
                    args, exp, resume_dir, noise_splits, output_dir
                )
                results["status"] = "SUCCESS"
                results["name"] = exp["name"]
            except Exception as e:
                print(
                    f"\n[ERROR] Experiment {exp['name']} failed: {e}",
                    flush=True,
                )
                results = {"name": exp["name"], "status": "FAILED", "error": str(e)}

            results["split_integrity"] = "Verified inside PROD Pipeline"
            results["wall_time_seconds"] = round(time.time() - exp_start, 1)
            global_results[exp["name"]] = results
            executed_count += 1
            print(
                f"[DONE] {exp['name']} | status={results['status']} | cal_score={results.get('best_cal_score', 0):.4f} | wall_time={results['wall_time_seconds']:.0f}s",
                flush=True,
            )

            result_path = os.path.join(output_dir, f"results_{exp['name']}.json")
            with open(result_path, "w") as f:
                json.dump(results, f, indent=2)
            continue

        print(
            f"[SKIP] {exp['name']}: non-PROD (MIL) experiment path was removed; skipping.",
            flush=True,
        )
        global_results[exp["name"]] = {
            "name": exp["name"],
            "status": "SKIPPED",
            "reason": "MIL path removed",
        }
        executed_count += 1

    global_path = os.path.join(output_dir, "global_results.json")
    with open(global_path, "w") as f:
        json.dump(global_results, f, indent=2)

    if os.path.exists("/content/drive/MyDrive"):
        try:
            drive_results_dir = "/content/drive/MyDrive/OHH_Murmur_Results"
            shutil.copy(global_path, drive_results_dir)
            print(
                "[Colab Sync] Final global_results.json synced to Google Drive.",
                flush=True,
            )
        except Exception as e:
            print(
                f"[Colab Sync] Warning: failed to sync final global results: {e}",
                flush=True,
            )

    if args.gcs_bucket:
        safe_gcs_upload(
            global_path, args.gcs_bucket, "results/global_results.json", gcs_client
        )

    print(f"Completed {executed_count} experiments", flush=True)
    print(
        f"Experiment wall time {round((time.time() - run_start)/3600, 1)} hours {int(time.time() - run_start)}s",
        flush=True,
    )
    for name, res in global_results.items():
        st = res.get("status", "SUCCESS")
        cs = res.get("best_cal_score", 0)
        wt = res.get("wall_time_seconds", 0)
        pn_auc = res.get("physionet22_val", {}).get("auc", "-")
        ohh_auc = res.get("OHH-2025", {}).get("auc", "-")
        print(
            f"  {name:35s} | status={st:7s} | cal={cs:.4f} | PN-AUC={pn_auc} | OHH-AUC={ohh_auc} | {wt:.0f}s",
            flush=True,
        )

    total_time = time.time() - run_start
    print(
        f"\nTOTAL PIPELINE WALL TIME: {total_time/3600:.1f} hours ({int(total_time)}s)",
        flush=True,
    )
    failed = [name for name, res in global_results.items() if res.get("status") != "SUCCESS"]
    if failed:
        print(f"Failed experiments: {', '.join(failed)}", flush=True)
        sys.exit(1)


# Data loading, preprocessing, augmentation, training and evaluation
DEBUG_MODE = False
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_CI = 0.95
BOOTSTRAP_SEED = 42


@dataclass
class TrainConfig:
    data_root: str
    output_dir: str
    model_name: str
    seed: int
    max_trials: int
    patience_trials: int
    epochs: int
    feature_cfg: "FeatureConfig"
    max_length: int
    ohh_folder: str = None


@dataclass
class FeatureConfig:
    sample_rate: int = 16000
    normalize: bool = True
    remove_dc: bool = True
    pre_emphasis: float = 0.97
    bandpass_low: float = 20.0
    bandpass_high: float = 1200.0
    rms_normalize: bool = True
    rms_target: float = 0.1
    trim_silence: bool = True
    trim_db: float = -40.0
    max_duration_seconds: float = 0.0
    use_spec_augment: bool = False
    spec_freq_mask: int = 12
    spec_time_mask: int = 48
    spec_cardiac_constrained: bool = (
        False
    )
    use_lung_noise: bool = False
    use_polarity_flip: bool = False
    use_gain_jitter: bool = False
    gain_jitter_range: float = 0.3
    use_hpss_eq: bool = False
    use_mixup: bool = False
    mixup_alpha: float = 0.4
    use_cardiac_crop: bool = False
    use_resp_am: bool = False
    resp_am_depth: float = 0.3
    use_quality_weight: bool = False
    quality_weight_min: float = 0.5
    lambda_val: float = None
    noise_files: list = None
    is_train: bool = False


class FeatureDataset:
    def __init__(self, records: List[Tuple], is_train: bool = False) -> None:
        self.records = records
        self.is_train = is_train

    def to_hf_dataset(self) -> Dataset:
        data = []
        for r in self.records:
            item = {"is_train": self.is_train}
            if len(r) == 2:
                item.update({"rel_path": r[0], "label": r[1]})
            elif len(r) == 3 and isinstance(r[2], bytes):
                item.update({"rel_path": r[0], "label": r[1], "audio_bytes": r[2]})
            elif len(r) == 3 and not isinstance(r[2], bytes):
                item.update({"rel_path": r[0], "label": r[1], "patient_id": r[2]})
            elif len(r) == 4:
                item.update(
                    {
                        "rel_path": r[0],
                        "label": r[1],
                        "patient_id": r[2],
                        "audio_bytes": r[3],
                    }
                )
            data.append(item)
        return Dataset.from_list(data)


def setup_logging() -> None:
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )


def set_seed(seed: int) -> None:
    logger.info("Setting random seed: {}", seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.backends.mps.is_available():
        torch.use_deterministic_algorithms(True, warn_only=True)


def is_under_dir(rel_path: str, dir_prefix: str) -> bool:
    return rel_path.replace("\\\\", "/").startswith(dir_prefix.rstrip("/") + "/")


def label_from_path(rel_path: str) -> Optional[int]:
    if "/Abnormal/" in rel_path.replace("\\\\", "/"):
        return 1
    if "/Normal/" in rel_path.replace("\\\\", "/"):
        return 0
    return None


def extract_patient_id(rel_path: str, base_dir: str) -> Optional[str]:
    norm = rel_path.replace("\\\\", "/")
    prefix = base_dir.rstrip("/") + "/"
    if not norm.startswith(prefix):
        return None
    remainder = norm[len(prefix) :]
    parts = remainder.split("/")
    if len(parts) < 2:
        return None
    return parts[0]


def build_records(
    data_root: str,
) -> Tuple[List[Tuple], List[Tuple]]:
    db_path = _locate_db(data_root)
    if db_path and os.path.exists(db_path):
        logger.info("Found Kaggle SQLite DB. Loading data directly from database...")
        import sqlite3

        conn = sqlite3.connect(db_path, uri=True)
        c = conn.cursor()

        c.execute(
            "SELECT patient_id, MAX(has_murmur) FROM audio_info WHERE project_name = 'physionet22' COLLATE NOCASE GROUP BY patient_id"
        )
        all_p = c.fetchall()

        patient_rng = random.Random(42)
        murmur_p = [p for p in all_p if p[1] == 1]
        normal_p = [p for p in all_p if p[1] == 0]
        patient_rng.shuffle(murmur_p)
        patient_rng.shuffle(normal_p)
        m_split = int(len(murmur_p) * 0.8)
        n_split = int(len(normal_p) * 0.8)
        train_patients = set(p[0] for p in (murmur_p[:m_split] + normal_p[:n_split]))
        val_patients = set(p[0] for p in (murmur_p[m_split:] + normal_p[n_split:]))

        c.execute("""
            SELECT a.uuid, a.patient_id, a.has_murmur, b.audio_file 
            FROM audio_info a 
            JOIN audio_raw_zstd b ON a.uuid = b.uuid 
            WHERE a.project_name = 'physionet22' COLLATE NOCASE
        """)
        all_rows = c.fetchall()

        train_records = []
        val_records = []
        for row in all_rows:
            uuid_str, patient_id, label, audio_zstd = row
            if patient_id in train_patients:
                train_records.append((uuid_str, label, audio_zstd))
            elif patient_id in val_patients:
                val_records.append((uuid_str, label, patient_id, audio_zstd))

        logger.info("Train records: {}", len(train_records))
        logger.info("Val records: {}", len(val_records))
        return train_records, val_records

    raise FileNotFoundError(f"Kaggle SQLite DB not found at {db_path}")


class NoiseMixingError(RuntimeError):
    pass


def load_audio(path: str) -> Tuple[torch.Tensor, int]:
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T.copy())
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sr


def resample_if_needed(
    waveform: torch.Tensor, orig_sr: int, target_sr: int
) -> torch.Tensor:
    if orig_sr == target_sr:
        return waveform
    return torchaudio.functional.resample(waveform, orig_sr, target_sr)


def normalize_waveform(waveform: torch.Tensor) -> torch.Tensor:
    max_val = waveform.abs().max()
    if max_val > 0:
        return waveform / max_val
    return waveform


def remove_dc_offset(waveform: torch.Tensor) -> torch.Tensor:
    return waveform - waveform.mean(dim=1, keepdim=True)


def apply_pre_emphasis(waveform: torch.Tensor, coeff: float) -> torch.Tensor:
    if coeff <= 0:
        return waveform
    filtered = waveform.clone()
    filtered[:, 1:] = waveform[:, 1:] - coeff * waveform[:, :-1]
    return filtered


def apply_bandpass(
    waveform: torch.Tensor, sample_rate: int, low_hz: float, high_hz: float
) -> torch.Tensor:
    if low_hz and low_hz > 0:
        waveform = torchaudio.functional.highpass_biquad(waveform, sample_rate, low_hz)
    if high_hz and high_hz > 0 and high_hz < sample_rate / 2:
        waveform = torchaudio.functional.lowpass_biquad(waveform, sample_rate, high_hz)
    return waveform


def rms_normalize(waveform: torch.Tensor, target_rms: float) -> torch.Tensor:
    if target_rms <= 0:
        return waveform
    rms = torch.sqrt(torch.mean(waveform**2))
    if rms > 0:
        return waveform * (target_rms / rms)
    return waveform


def trim_silence(waveform: torch.Tensor, trim_db: float) -> torch.Tensor:
    if trim_db >= 0:
        return waveform
    max_val = waveform.abs().max()
    if max_val == 0:
        return waveform
    threshold = max_val * (10 ** (trim_db / 20))
    mask = (waveform.abs() >= threshold).any(dim=0)
    if not mask.any():
        return waveform
    indices = mask.nonzero(as_tuple=False).squeeze(1)
    start = int(indices[0].item())
    end = int(indices[-1].item()) + 1
    return waveform[:, start:end]


def crop_max_duration(
    waveform: torch.Tensor, sample_rate: int, max_seconds: float
) -> torch.Tensor:
    if max_seconds <= 0:
        return waveform
    max_samples = int(max_seconds * sample_rate)
    if waveform.size(1) > max_samples:
        return waveform[:, :max_samples]
    return waveform


def _cardiac_envelope(
    x: np.ndarray, sample_rate: int, env_sr: int = 200
) -> Tuple[np.ndarray, int]:
    """Rectified, smoothed, decimated amplitude envelope for cheap cycle analysis."""
    win = max(1, int(0.05 * sample_rate))
    kernel = np.ones(win, dtype=np.float32) / float(win)
    env = np.convolve(np.abs(x), kernel, mode="same")
    step = max(1, sample_rate // env_sr)
    return env[::step].astype(np.float32), step


def _envelope_band_autocorr(
    env_ds: np.ndarray,
    step: int,
    sample_rate: int,
    hr_min_bpm: float = 30.0,
    hr_max_bpm: float = 220.0,
):
    """Autocorrelation of the zero-mean envelope and the lag window for the heart-rate band."""
    e = env_ds - env_ds.mean()
    if e.size < 8 or not np.any(e):
        return None
    n = int(2 ** np.ceil(np.log2(e.size * 2)))
    spec = np.fft.rfft(e, n)
    ac = np.fft.irfft(spec * np.conj(spec), n)[: e.size]
    env_sr = sample_rate / float(step)
    lo = int(env_sr * 60.0 / hr_max_bpm)
    hi = min(int(env_sr * 60.0 / hr_min_bpm), ac.size - 1)
    if hi <= lo + 1:
        return None
    return ac, lo, hi


def _estimate_cycle_length(
    env_ds: np.ndarray,
    step: int,
    sample_rate: int,
    hr_min_bpm: float = 30.0,
    hr_max_bpm: float = 220.0,
) -> Optional[int]:
    """Estimate the cardiac cycle length in samples from the envelope autocorrelation (30-220 bpm)."""
    res = _envelope_band_autocorr(env_ds, step, sample_rate, hr_min_bpm, hr_max_bpm)
    if res is None:
        return None
    ac, lo, hi = res
    lag = lo + int(np.argmax(ac[lo:hi]))
    if lag <= 0:
        return None
    return lag * step


def _rhythmicity_score(
    x: np.ndarray,
    sample_rate: int,
    hr_min_bpm: float = 30.0,
    hr_max_bpm: float = 220.0,
) -> Optional[float]:
    """Recording-quality score in [0, 1]: the normalized envelope-autocorrelation peak in the heart-rate band."""
    env_ds, step = _cardiac_envelope(np.asarray(x, dtype=np.float32), sample_rate)
    res = _envelope_band_autocorr(env_ds, step, sample_rate, hr_min_bpm, hr_max_bpm)
    if res is None:
        return None
    ac, lo, hi = res
    if ac[0] <= 0:
        return None
    peak = float(ac[lo:hi].max() / ac[0])
    return float(min(1.0, max(0.0, peak)))


def crop_cardiac_cycles(
    waveform: torch.Tensor,
    sample_rate: int,
    target_len: int,
    rng: np.random.Generator,
    hr_min_bpm: float = 30.0,
    hr_max_bpm: float = 220.0,
) -> torch.Tensor:
    """Random crop aligned to whole cardiac cycles (training only)."""
    total = int(waveform.shape[-1])
    if target_len <= 0 or total <= target_len:
        return waveform

    x = waveform.squeeze(0).detach().cpu().numpy().astype(np.float32)
    env_ds, step = _cardiac_envelope(x, sample_rate)
    period = _estimate_cycle_length(env_ds, step, sample_rate, hr_min_bpm, hr_max_bpm)

    if period is not None and 0 < period <= target_len:
        n_cycles = max(1, target_len // period)
        crop_len = min(total, n_cycles * period)
    else:
        crop_len = target_len

    max_start = total - crop_len
    if max_start <= 0:
        return waveform

    start = None
    if period is not None and period > 0:
        try:
            min_dist = max(1, int(0.6 * period / step))
            peaks, _ = scipy.signal.find_peaks(env_ds, distance=min_dist)
            onsets = peaks * step
            onsets = onsets[onsets <= max_start]
            if onsets.size:
                start = int(onsets[int(rng.integers(0, onsets.size))])
        except Exception:
            start = None

    if start is None:
        start = int(rng.integers(0, max_start + 1))

    return waveform[:, start : start + crop_len]


def apply_respiratory_am(
    waveform: torch.Tensor,
    sample_rate: int,
    rng: np.random.Generator,
    depth: float = 0.3,
    rate_min_bpm: float = 12.0,
    rate_max_bpm: float = 20.0,
) -> torch.Tensor:
    """Respiratory amplitude modulation (training only)."""
    if depth <= 0:
        return waveform
    total = int(waveform.shape[-1])
    if total <= 0:
        return waveform
    rate_hz = float(rng.uniform(rate_min_bpm, rate_max_bpm)) / 60.0
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    t = np.arange(total, dtype=np.float32) / float(sample_rate)
    mod = (1.0 + float(depth) * np.sin(2.0 * np.pi * rate_hz * t + phase)).astype(
        np.float32
    )
    return waveform * torch.from_numpy(mod).to(waveform.dtype)


def apply_hpss_eq(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """HPSS with parametric band equalization."""
    try:
        import librosa
    except ImportError:
        return waveform

    wav_np = waveform.squeeze(0).numpy()

    h, p = librosa.effects.hpss(wav_np, margin=1.0)

    alpha = random.uniform(0.8, 1.2)
    beta = random.uniform(0.8, 1.2)
    wav_aug = (alpha * h) + (beta * p)

    num_bands = random.randint(1, 2)
    for _ in range(num_bands):
        fc = random.uniform(50.0, 1000.0)
        gain_db = random.uniform(-3.0, 3.0)
        Q = random.uniform(1.0, 3.0)

        w0 = fc / (sample_rate / 2.0)
        if w0 >= 1.0 or w0 <= 0.0:
            continue

        b, a = scipy.signal.iirpeak(w0, Q)

        gain_linear = 10.0 ** (gain_db / 20.0)
        filtered = scipy.signal.lfilter(b, a, wav_aug)
        wav_aug = wav_aug + filtered * (gain_linear - 1.0)

    if np.any(np.isnan(wav_aug)) or np.any(np.isinf(wav_aug)):
        return waveform

    return torch.from_numpy(wav_aug.astype(np.float32)).unsqueeze(0)


class DataCollatorAST:
    def __init__(
        self,
        feature_extractor: AutoFeatureExtractor,
        data_root: str,
        cfg: FeatureConfig,
        max_length: int,
        seed: int = 42,
    ) -> None:
        self.feature_extractor = feature_extractor
        self.data_root = data_root
        self.cfg = cfg
        self.max_length = max_length
        self._rng = np.random.default_rng(seed)

    def _prepare_waveform(self, item: Dict[str, object]) -> Optional[np.ndarray]:
        try:
            if "audio_bytes" in item:
                import io
                import soundfile as sf

                dctx = zstd.ZstdDecompressor()
                raw_wav = dctx.decompress(item["audio_bytes"])
                data, orig_sr = sf.read(io.BytesIO(raw_wav))
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                waveform = torch.from_numpy(data).float().unsqueeze(0)
            else:
                path = os.path.join(self.data_root, item["rel_path"])
                waveform, orig_sr = load_audio(path)

            waveform = resample_if_needed(waveform, orig_sr, self.cfg.sample_rate)
            if self.cfg.remove_dc:
                waveform = remove_dc_offset(waveform)

            is_train = item.get("is_train", False)

            if is_train and self.cfg.use_gain_jitter:
                if random.random() < 0.5:
                    r = self.cfg.gain_jitter_range
                    gain = random.uniform(1.0 - r, 1.0 + r)
                    waveform = waveform * gain

            if is_train and self.cfg.use_hpss_eq:
                if random.random() < 0.5:
                    waveform = apply_hpss_eq(waveform, self.cfg.sample_rate)

            waveform = apply_bandpass(
                waveform,
                self.cfg.sample_rate,
                self.cfg.bandpass_low,
                self.cfg.bandpass_high,
            )
            waveform = apply_pre_emphasis(waveform, self.cfg.pre_emphasis)
            if self.cfg.trim_silence:
                waveform = trim_silence(waveform, self.cfg.trim_db)
            if (
                is_train
                and self.cfg.use_cardiac_crop
                and self.cfg.max_duration_seconds > 0
            ):
                target_len = int(
                    round(self.cfg.max_duration_seconds * self.cfg.sample_rate)
                )
                waveform = crop_cardiac_cycles(
                    waveform, self.cfg.sample_rate, target_len, self._rng
                )
            else:
                waveform = crop_max_duration(
                    waveform, self.cfg.sample_rate, self.cfg.max_duration_seconds
                )
            if self.cfg.normalize:
                waveform = normalize_waveform(waveform)
            if self.cfg.rms_normalize:
                waveform = rms_normalize(waveform, self.cfg.rms_target)

            if is_train and self.cfg.use_lung_noise and self.cfg.noise_files:
                noise_path = random.choice(self.cfg.noise_files)
                try:
                    noise_wav, n_sr = load_audio(noise_path)
                    noise_wav = resample_if_needed(
                        noise_wav, n_sr, self.cfg.sample_rate
                    )
                    if self.cfg.remove_dc:
                        noise_wav = remove_dc_offset(noise_wav)
                    noise_wav = apply_bandpass(
                        noise_wav,
                        self.cfg.sample_rate,
                        self.cfg.bandpass_low,
                        self.cfg.bandpass_high,
                    )
                    noise_wav = apply_pre_emphasis(noise_wav, self.cfg.pre_emphasis)
                    if self.cfg.rms_normalize:
                        noise_wav = rms_normalize(noise_wav, self.cfg.rms_target)

                    if noise_wav.shape[-1] < waveform.shape[-1]:
                        pad_amt = waveform.shape[-1] - noise_wav.shape[-1]
                        noise_wav = F.pad(noise_wav, (0, pad_amt))
                    else:
                        noise_wav = noise_wav[:, : waveform.shape[-1]]

                    l_val = (
                        self.cfg.lambda_val
                        if self.cfg.lambda_val is not None
                        else random.choice([0.0, 0.25, 0.5, 1.0, 5.0, 10.0])
                    )
                    waveform = waveform + noise_wav * (l_val / 10.0)

                    if self.cfg.normalize:
                        waveform = normalize_waveform(waveform)
                    if self.cfg.rms_normalize:
                        waveform = rms_normalize(waveform, self.cfg.rms_target)
                except Exception as ne:
                    raise NoiseMixingError(f"noise mixing failed for {noise_path}: {ne}") from ne

            if is_train and self.cfg.use_polarity_flip:
                if random.random() < 0.5:
                    waveform = waveform * -1.0

            if is_train and self.cfg.use_resp_am:
                if self._rng.random() < 0.5:
                    waveform = apply_respiratory_am(
                        waveform,
                        self.cfg.sample_rate,
                        self._rng,
                        depth=self.cfg.resp_am_depth,
                    )

            if waveform.shape[-1] < 1600:
                return np.zeros(16000, dtype=np.float32)

            return waveform.squeeze(0).numpy()
        except NoiseMixingError:
            raise
        except Exception as e:
            logger.warning(
                "Failed to load audio for {}: {}", item.get("rel_path", "unknown"), e
            )
            return np.zeros(16000, dtype=np.float32)

    def __call__(self, features: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        valid_features = []
        inputs = []
        for f in features:
            arr = self._prepare_waveform(f)
            if arr is not None:
                if arr.size == 0:
                    arr = np.zeros(16000, dtype=np.float32)
                inputs.append(arr)
                valid_features.append(f)

        if not inputs:
            inputs = [np.zeros(16000, dtype=np.float32)]
            valid_features = [
                features[0] if features else {"label": 0, "rel_path": "dummy"}
            ]

        batch = self.feature_extractor(
            inputs,
            sampling_rate=self.cfg.sample_rate,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        if self.cfg.use_spec_augment:
            if any(f.get("is_train", False) for f in valid_features):
                input_values = batch["input_values"].clone()
                for i, f in enumerate(valid_features):
                    if f.get("is_train", False):
                        if random.random() < 0.5:
                            f_max = self.cfg.spec_freq_mask
                            if self.cfg.spec_cardiac_constrained:
                                CARDIAC_MAX_BIN = 52
                                f_mask = random.randint(0, min(f_max, CARDIAC_MAX_BIN))
                                f0 = random.randint(0, CARDIAC_MAX_BIN - f_mask)
                            else:
                                f_mask = random.randint(0, f_max)
                                f0 = random.randint(0, 128 - f_mask)
                            input_values[i, :, f0 : f0 + f_mask] = 0
                        if random.random() < 0.5:
                            t_max = self.cfg.spec_time_mask
                            t_mask = random.randint(0, t_max)
                            t0 = random.randint(0, 1024 - t_mask)
                            input_values[i, t0 : t0 + t_mask, :] = 0
                batch["input_values"] = input_values

        is_any_train = any(f.get("is_train", False) for f in valid_features)
        if self.cfg.use_mixup and is_any_train and len(valid_features) > 1:
            lam = np.random.beta(self.cfg.mixup_alpha, self.cfg.mixup_alpha)
            batch_size = batch["input_values"].size(0)
            index = torch.randperm(batch_size)
            batch["input_values"] = (
                lam * batch["input_values"] + (1 - lam) * batch["input_values"][index]
            )
            hard = torch.tensor(
                [f.get("label", 0) for f in valid_features], dtype=torch.long
            )
            y_onehot = F.one_hot(hard, num_classes=2).float()
            y_mixed = lam * y_onehot + (1 - lam) * y_onehot[index]
            batch["labels"] = y_mixed
        else:
            labels = torch.tensor(
                [f.get("label", 0) for f in valid_features], dtype=torch.long
            )
            batch["labels"] = labels

        if self.cfg.use_quality_weight and is_any_train:
            wmin = float(self.cfg.quality_weight_min)
            weights = []
            for arr, f in zip(inputs, valid_features):
                if f.get("is_train", False):
                    r = _rhythmicity_score(arr, self.cfg.sample_rate)
                    r = 1.0 if r is None else r
                    weights.append(wmin + (1.0 - wmin) * r)
                else:
                    weights.append(1.0)
            w = torch.tensor(weights, dtype=torch.float32)
            w = w * (w.numel() / w.sum().clamp_min(1e-8))
            batch["sample_weight"] = w

        return batch


class DataCollatorASTRaw:
    """Collator for the "00_dryrun_audioset_zeroshot" experiment only."""

    def __init__(
        self,
        feature_extractor: AutoFeatureExtractor,
        data_root: str,
        sample_rate: int = 16000,
        max_length: int = 1024,
    ) -> None:
        self.feature_extractor = feature_extractor
        self.data_root = data_root
        self.sample_rate = sample_rate
        self.max_length = max_length

    def _load_raw_waveform(self, item: Dict[str, object]) -> np.ndarray:
        try:
            if "audio_bytes" in item:
                import io
                import soundfile as sf

                dctx = zstd.ZstdDecompressor()
                raw_wav = dctx.decompress(item["audio_bytes"])
                data, orig_sr = sf.read(io.BytesIO(raw_wav))
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                waveform = torch.from_numpy(data).float().unsqueeze(0)
            else:
                path = os.path.join(self.data_root, item["rel_path"])
                waveform, orig_sr = load_audio(path)

            waveform = resample_if_needed(waveform, orig_sr, self.sample_rate)
            if waveform.shape[-1] < 1600:
                return np.zeros(16000, dtype=np.float32)
            return waveform.squeeze(0).numpy()
        except Exception as e:
            logger.warning(
                "[zero-shot] Failed to load audio for {}: {}",
                item.get("rel_path", "unknown"),
                e,
            )
            return np.zeros(16000, dtype=np.float32)

    def __call__(self, features: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        if not features:
            features = [{"label": 0, "rel_path": "dummy"}]
        inputs = [self._load_raw_waveform(f) for f in features]

        batch = self.feature_extractor(
            inputs,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        batch["labels"] = torch.tensor(
            [f.get("label", 0) for f in features], dtype=torch.long
        )
        return batch


class MetricsCheckpointCallback(TrainerCallback):
    """Saves validation metrics and checkpoint information to a CSV file."""

    def __init__(self, csv_path: str, calibrated_score_lambda: float) -> None:
        self.csv_path = csv_path
        self.calibrated_score_lambda = calibrated_score_lambda
        self.pending_metrics = None
        self._write_header()

    def _write_header(self) -> None:
        """Write CSV header if file doesn't exist."""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "epoch",
                        "step",
                        "checkpoint_path",
                        "eval_loss",
                        "eval_accuracy",
                        "eval_mcc",
                        "eval_sensitivity",
                        "eval_specificity",
                        "eval_auc_roc",
                        "eval_auc_pr",
                        "eval_brier",
                        "eval_calibrated_score",
                        "timestamp",
                    ]
                )
            logger.info("Created metrics CSV file: {}", self.csv_path)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Store metrics temporarily - will be written in on_save."""
        if metrics is None:
            return

        eval_auc_pr = metrics.get("eval_auc_pr")
        eval_brier = metrics.get("eval_brier")
        if eval_auc_pr is not None and eval_brier is not None:
            calibrated_score = float(
                eval_auc_pr
            ) - self.calibrated_score_lambda * float(eval_brier)
            metrics["eval_calibrated_score"] = calibrated_score
            mlflow.log_metric(
                "eval_calibrated_score", calibrated_score, step=state.global_step
            )

        self.pending_metrics = {
            "epoch": int(state.epoch) if hasattr(state, "epoch") else 0,
            "step": state.global_step,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }
        logger.info("Metrics captured for step {}", state.global_step)

    def on_save(self, args, state, control, **kwargs):
        """Write metrics to CSV when checkpoint is saved."""
        if self.pending_metrics is None:
            return

        checkpoint_name = f"checkpoint-{state.global_step}"

        candidates = []

        candidates.append(os.path.join(args.output_dir, checkpoint_name))

        if os.path.exists(args.output_dir):
            for d in os.listdir(args.output_dir):
                if d.startswith("run-"):
                    run_dir = os.path.join(args.output_dir, d)
                    if os.path.isdir(run_dir):
                        candidates.append(os.path.join(run_dir, checkpoint_name))

        existing_candidates = [p for p in candidates if os.path.isdir(p)]

        if existing_candidates:
            existing_candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            checkpoint_path = existing_candidates[0]
        else:
            checkpoint_path = os.path.join(args.output_dir, checkpoint_name)

        metrics = self.pending_metrics["metrics"]
        epoch = self.pending_metrics["epoch"]
        step = self.pending_metrics["step"]
        timestamp = self.pending_metrics["timestamp"]

        eval_loss = metrics.get("eval_loss", "")
        eval_accuracy = metrics.get("eval_accuracy", "")
        eval_mcc = metrics.get("eval_mcc", "")
        eval_sensitivity = metrics.get("eval_sensitivity", "")
        eval_specificity = metrics.get("eval_specificity", "")
        eval_auc_roc = metrics.get("eval_auc_roc", "")
        eval_auc_pr = metrics.get("eval_auc_pr", "")
        eval_brier = metrics.get("eval_brier", "")
        eval_calibrated_score = metrics.get("eval_calibrated_score", "")

        with open(self.csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    step,
                    checkpoint_path,
                    eval_loss,
                    eval_accuracy,
                    eval_mcc,
                    eval_sensitivity,
                    eval_specificity,
                    eval_auc_roc,
                    eval_auc_pr,
                    eval_brier,
                    eval_calibrated_score,
                    timestamp,
                ]
            )

        logger.info(
            "Wrote metrics to CSV for step {} with checkpoint: {}",
            step,
            checkpoint_path,
        )
        self.pending_metrics = None


class OptunaMLflowRunCallback(TrainerCallback):
    def __init__(self, study_name: str) -> None:
        self.study_name = study_name

    def on_train_begin(self, args, state, control, **kwargs):
        trial_name = getattr(state, "trial_name", None)
        if trial_name is not None:
            mlflow.set_tag("optuna_trial", str(trial_name))
            mlflow.set_tag("optuna_study", self.study_name)
            logger.info("Tagged MLflow run with trial info: {}", trial_name)

    def on_train_end(self, args, state, control, **kwargs):
        trial_name = getattr(state, "trial_name", None)
        checkpoint = getattr(state, "best_model_checkpoint", None)

        logger.info("Trial: {}, Checkpoint: {}", trial_name, checkpoint)

        if checkpoint:
            mlflow.log_param("best_checkpoint", checkpoint)
            logger.info(
                "Logged best_checkpoint for trial '{}' to MLflow: {}",
                trial_name,
                checkpoint,
            )
        else:
            logger.warning(
                "No best_model_checkpoint available for trial '{}'", trial_name
            )


def compute_patient_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    rel_paths: List[str],
    val_records: List[Tuple],
    patient_aggregation: str = "topk_mean",
    patient_top_k: int = 2,
    apply_temperature_scaling: bool = False,
    temperature_min: float = 0.8,
    temperature_max: float = 3.0,
    temperature_steps: int = 25,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    ci: float = BOOTSTRAP_CI,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, float]:
    def _build_patient_targets_and_scores(
        probabilities: np.ndarray,
    ) -> Tuple[List[int], List[int], List[float]]:
        patient_prob_lists: Dict[str, List[float]] = {}
        patient_labels: Dict[str, int] = {}
        lookup = {r[0]: (r[1], r[2]) for r in val_records}

        for rel_path, prob in zip(rel_paths, probabilities):
            match = lookup.get(rel_path)
            if match is None:
                continue
            label, patient_id = match
            patient_labels[patient_id] = label
            patient_prob_lists.setdefault(patient_id, []).append(float(prob))

        def _aggregate_patient_probability(values: List[float]) -> float:
            if not values:
                return 0.0
            if patient_aggregation == "max":
                return float(max(values))
            if patient_aggregation == "topk_mean":
                k = max(1, min(patient_top_k, len(values)))
                top_values = sorted(values, reverse=True)[:k]
                return float(np.mean(top_values))
            raise ValueError(
                "Unsupported patient_aggregation: "
                f"{patient_aggregation}. Use 'topk_mean' or 'max'."
            )

        y_true_local: List[int] = []
        y_pred_local: List[int] = []
        y_score_local: List[float] = []
        for patient_id, probs_for_patient in patient_prob_lists.items():
            prob = _aggregate_patient_probability(probs_for_patient)
            y_true_local.append(patient_labels[patient_id])
            y_pred_local.append(1 if prob >= 0.5 else 0)
            y_score_local.append(prob)
        return y_true_local, y_pred_local, y_score_local

    def _pick_temperature() -> float:
        if not apply_temperature_scaling:
            return 1.0
        if temperature_steps < 2 or temperature_max <= temperature_min:
            return 1.0

        candidate_temperatures = np.linspace(
            temperature_min,
            temperature_max,
            temperature_steps,
        )
        best_temperature = 1.0
        best_brier = float("inf")

        for temperature in candidate_temperatures:
            scaled_probs = torch.softmax(
                torch.tensor(logits, dtype=torch.float32) / float(temperature),
                dim=-1,
            )[:, 1].numpy()
            y_true_local, _, y_score_local = _build_patient_targets_and_scores(
                scaled_probs
            )
            if not y_true_local:
                continue
            brier_val = float(brier_score_loss(y_true_local, y_score_local))
            if brier_val < best_brier:
                best_brier = brier_val
                best_temperature = float(temperature)

        return best_temperature

    temperature = _pick_temperature()
    probs = torch.softmax(
        torch.tensor(logits, dtype=torch.float32) / temperature,
        dim=-1,
    )[:, 1].numpy()

    y_true, y_pred, y_score = _build_patient_targets_and_scores(probs)

    if not y_true:
        return {
            "mcc": 0.0,
            "accuracy": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "auc_roc": 0.0,
            "auc_pr": 0.0,
            "brier": 0.0,
            "temperature": temperature,
            "mcc_ci_low": 0.0,
            "mcc_ci_high": 0.0,
            "accuracy_ci_low": 0.0,
            "accuracy_ci_high": 0.0,
            "sensitivity_ci_low": 0.0,
            "sensitivity_ci_high": 0.0,
            "specificity_ci_low": 0.0,
            "specificity_ci_high": 0.0,
            "auc_roc_ci_low": 0.0,
            "auc_roc_ci_high": 0.0,
            "auc_pr_ci_low": 0.0,
            "auc_pr_ci_high": 0.0,
            "brier_ci_low": 0.0,
            "brier_ci_high": 0.0,
        }

    mcc = float(matthews_corrcoef(y_true, y_pred))
    accuracy = float(accuracy_score(y_true, y_pred))
    pos_total = sum(1 for y in y_true if y == 1)
    neg_total = sum(1 for y in y_true if y == 0)
    true_pos = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1)
    true_neg = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 0)
    sensitivity = float(true_pos / pos_total) if pos_total > 0 else 0.0
    specificity = float(true_neg / neg_total) if neg_total > 0 else 0.0
    if len(set(y_true)) < 2:
        auc_roc = 0.0
        auc_pr = 0.0
    else:
        auc_roc = float(roc_auc_score(y_true, y_score))
        auc_pr = float(average_precision_score(y_true, y_score))
    brier = float(brier_score_loss(y_true, y_score)) if y_true else 0.0

    rng = np.random.default_rng(seed)
    indices = np.arange(len(y_true))
    alpha = (1.0 - ci) / 2.0

    def _metric_dict(y_t, y_p, y_s) -> Dict[str, float]:
        mcc_val = float(matthews_corrcoef(y_t, y_p))
        acc_val = float(accuracy_score(y_t, y_p))
        pos_total = sum(1 for y in y_t if y == 1)
        neg_total = sum(1 for y in y_t if y == 0)
        true_pos = sum(1 for y, p in zip(y_t, y_p) if y == 1 and p == 1)
        true_neg = sum(1 for y, p in zip(y_t, y_p) if y == 0 and p == 0)
        sens_val = float(true_pos / pos_total) if pos_total > 0 else 0.0
        spec_val = float(true_neg / neg_total) if neg_total > 0 else 0.0
        if len(set(y_t)) < 2:
            auc_val = 0.0
            auc_pr_val = 0.0
        else:
            auc_val = float(roc_auc_score(y_t, y_s))
            auc_pr_val = float(average_precision_score(y_t, y_s))
        brier_val = float(brier_score_loss(y_t, y_s)) if y_t else 0.0
        return {
            "mcc": mcc_val,
            "accuracy": acc_val,
            "sensitivity": sens_val,
            "specificity": spec_val,
            "auc_roc": auc_val,
            "auc_pr": auc_pr_val,
            "brier": brier_val,
        }

    bootstrap_metrics = {
        "mcc": [],
        "accuracy": [],
        "sensitivity": [],
        "specificity": [],
        "auc_roc": [],
        "auc_pr": [],
        "brier": [],
    }
    for _ in range(bootstrap_samples):
        sample_idx = rng.choice(indices, size=len(indices), replace=True)
        y_t = [y_true[i] for i in sample_idx]
        y_p = [y_pred[i] for i in sample_idx]
        y_s = [y_score[i] for i in sample_idx]
        vals = _metric_dict(y_t, y_p, y_s)
        for key in bootstrap_metrics:
            bootstrap_metrics[key].append(vals[key])

    ci_bounds = {}
    for key, values in bootstrap_metrics.items():
        lower = float(np.quantile(values, alpha))
        upper = float(np.quantile(values, 1.0 - alpha))
        ci_bounds[f"{key}_ci_low"] = lower
        ci_bounds[f"{key}_ci_high"] = upper

    return {
        "mcc": mcc,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "brier": brier,
        "temperature": temperature,
        **ci_bounds,
    }


def build_datasets(
    train_records: List[Tuple[str, int]],
    val_records: List[Tuple],
) -> Tuple[Dataset, Dataset, List[str]]:
    logger.info("Building Hugging Face datasets")
    if DEBUG_MODE:
        logger.info("DEBUG_MODE enabled: sampling 100 train and 100 validation records")
        random.seed(42)
        if len(train_records) > 100:
            train_records = random.sample(train_records, 100)
        if len(val_records) > 100:
            val_records = random.sample(val_records, 100)
    train_dataset = FeatureDataset(train_records, is_train=True).to_hf_dataset()
    val_records_flat = []
    for r in val_records:
        if len(r) == 4:
            val_records_flat.append((r[0], r[1], r[3]))
        else:
            val_records_flat.append((r[0], r[1]))
    val_dataset = FeatureDataset(val_records_flat, is_train=False).to_hf_dataset()
    val_rel_paths = [r[0] for r in val_records]
    return train_dataset, val_dataset, val_rel_paths


from transformers import TrainerCallback


class ProgressiveUnfreezeCallback(TrainerCallback):
    def __init__(self, unfreeze_epoch: int = 3, lr_ratio: float = 0.1):
        self.unfreeze_epoch = unfreeze_epoch
        self.lr_ratio = lr_ratio
        self.unfrozen = False

    def on_epoch_begin(
        self,
        args,
        state,
        control,
        model=None,
        optimizer=None,
        lr_scheduler=None,
        **kwargs,
    ):
        if state.epoch >= self.unfreeze_epoch and not self.unfrozen:
            logger.info(
                f"[ProgressiveUnfreezeCallback] Epoch {state.epoch}: unfreezing the top 2 AST layers."
            )
            new_params = []

            unwrapped_model = getattr(model, "module", model)
            unwrapped_model = getattr(unwrapped_model, "_orig_mod", unwrapped_model)

            if hasattr(unwrapped_model, "audio_spectrogram_transformer"):
                backbone = unwrapped_model.audio_spectrogram_transformer
                blocks = backbone.layers if hasattr(backbone, "layers") else backbone.encoder.layer
                for layer in blocks[-2:]:
                    for p in layer.parameters():
                        p.requires_grad = True
                        new_params.append(p)

            if new_params and optimizer is not None:
                new_lr = args.learning_rate * self.lr_ratio
                optimizer.add_param_group({"params": new_params, "lr": new_lr})
                logger.info(
                    f"Added {len(new_params)} parameters to optimizer with base lr={new_lr}"
                )

                if lr_scheduler is not None:
                    if hasattr(lr_scheduler, "base_lrs"):
                        lr_scheduler.base_lrs.append(new_lr)
                    if (
                        hasattr(lr_scheduler, "lr_lambdas")
                        and len(lr_scheduler.lr_lambdas) > 0
                    ):
                        lr_scheduler.lr_lambdas.append(lr_scheduler.lr_lambdas[0])
                        logger.info(
                            "Patched lr_scheduler state to track the newly unfrozen backbone parameters."
                        )

            self.unfrozen = True


class FocalLossTrainer(Trainer):
    """HuggingFace Trainer subclass that uses FocalLoss instead of CrossEntropyLoss."""

    def __init__(self, focal_alpha=0.6, focal_gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.focal_loss_fn = FocalLoss(
            alpha=focal_alpha, gamma=focal_gamma, label_smoothing=0.05
        )
        logger.info(
            f"FocalLossTrainer active: alpha={focal_alpha}, gamma={focal_gamma}"
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        sample_weight = inputs.pop("sample_weight", None)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if labels.dim() == 2:
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -(labels * log_probs).sum(dim=-1).mean()
        elif sample_weight is not None:
            per = FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma,
                            reduction="none", label_smoothing=0.05)(logits, labels)
            loss = (per * sample_weight.to(dtype=per.dtype, device=per.device)).mean()
        else:
            loss = self.focal_loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


class MixupTrainer(Trainer):
    """HuggingFace Trainer that handles soft labels from Mixup using soft cross-entropy."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if labels.dim() == 2:
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -(labels * log_probs).sum(dim=-1).mean()
        else:
            loss = F.cross_entropy(logits, labels, label_smoothing=0.05)
        return (loss, outputs) if return_outputs else loss


class WeightedCETrainer(Trainer):
    """Trainer for the quality-aware ablation."""

    def __init__(self, label_smoothing=0.05, **kwargs):
        super().__init__(**kwargs)
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        sample_weight = inputs.pop("sample_weight", None)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if sample_weight is None:
            loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
        else:
            per = F.cross_entropy(
                logits, labels, label_smoothing=self.label_smoothing, reduction="none"
            )
            sw = sample_weight.to(dtype=per.dtype, device=per.device)
            loss = (per * sw).mean()
        return (loss, outputs) if return_outputs else loss


def run_zero_shot_audioset_baseline(args, exp, pipeline_output_dir=None):
    """Zero-shot baseline: score each recording with the pretrained AudioSet 'Heart murmur' class, without fine-tuning."""
    import torch
    import sqlite3
    import mlflow
    from torch.utils.data import DataLoader
    from transformers import ASTForAudioClassification

    logger.info(f"Starting zero-shot (no fine-tuning) AudioSet baseline: {exp['name']}")

    base_dir = (
        pipeline_output_dir
        if pipeline_output_dir
        else (args.output_dir if hasattr(args, "output_dir") else "checkpoints")
    )
    output_dir = os.path.join(base_dir, exp["name"])
    os.makedirs(output_dir, exist_ok=True)

    if getattr(args, "mlflow_tracking_uri", None):
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(getattr(args, "mlflow_experiment", "unified_prod_experiment"))

    set_seed(getattr(args, "seed", 42))

    model_name = getattr(args, "model_name", "MIT/ast-finetuned-audioset-10-10-0.4593")
    sample_rate = getattr(args, "sample_rate", 16000)
    max_length = getattr(args, "max_length", 1024)
    batch_size = getattr(args, "batch_size", 16)
    calibrated_score_lambda = getattr(args, "calibrated_score_lambda", 0.20)

    data_root = (
        os.path.dirname(args.db_path)
        if getattr(args, "db_path", None) and os.path.exists(args.db_path)
        else "/kaggle/input/mumurdetection"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("GPU is required for this script.")

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = ASTForAudioClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    murmur_idx = next(
        (k for k, v in id2label.items() if v.strip().lower() == "heart murmur"), None
    )
    heartbeat_idx = next(
        (
            k
            for k, v in id2label.items()
            if v.strip().lower() == "heart sounds, heartbeat"
        ),
        None,
    )
    if murmur_idx is None:
        raise RuntimeError(
            f"'Heart murmur' label not found among {len(id2label)} classes in "
            f"{model_name}'s id2label map."
        )
    logger.info(
        "[zero-shot] Using AudioSet class index {} ('Heart murmur') of {} "
        "total classes as the murmur score.{}",
        murmur_idx,
        len(id2label),
        (
            f" Index {heartbeat_idx} ('Heart sounds, heartbeat') is also logged."
            if heartbeat_idx is not None
            else ""
        ),
    )

    use_standard_preprocessing = exp.get("cfg", {}).get(
        "use_standard_preprocessing", False
    )
    if use_standard_preprocessing:
        feature_cfg = FeatureConfig(
            sample_rate=sample_rate,
            normalize=not getattr(args, "no_normalize", False),
            remove_dc=not getattr(args, "no_remove_dc", False),
            pre_emphasis=getattr(args, "pre_emphasis", 0.97),
            bandpass_low=getattr(args, "bandpass_low", 20.0),
            bandpass_high=getattr(args, "bandpass_high", 1200.0),
            rms_normalize=not getattr(args, "no_rms_normalize", False),
            rms_target=getattr(args, "rms_target", 0.1),
            trim_silence=not getattr(args, "no_trim_silence", False),
            trim_db=getattr(args, "trim_db", -40.0),
            max_duration_seconds=getattr(args, "max_duration_seconds", 10.0),
            is_train=False,
        )
        collator = DataCollatorAST(
            feature_extractor=feature_extractor,
            data_root=data_root,
            cfg=feature_cfg,
            max_length=max_length,
            seed=getattr(args, "seed", 42),
        )
    else:
        collator = DataCollatorASTRaw(
            feature_extractor, data_root, sample_rate=sample_rate, max_length=max_length
        )

    @torch.no_grad()
    def score_hf_dataset(hf_dataset):
        """Return two-column logits [0, murmur logit] in the row order of hf_dataset."""
        if len(hf_dataset) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        loader = DataLoader(
            hf_dataset,
            batch_size=batch_size,
            collate_fn=collator,
            shuffle=False,
        )
        murmur_logit_chunks = []
        for batch in loader:
            input_values = batch["input_values"].to(device)
            outputs = model(input_values=input_values)
            murmur_logit_chunks.append(
                outputs.logits[:, murmur_idx].detach().cpu().numpy()
            )
        murmur_logits = np.concatenate(murmur_logit_chunks)
        return np.stack([np.zeros_like(murmur_logits), murmur_logits], axis=-1).astype(
            np.float32
        )

    results = {"status": "SUCCESS"}
    if mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.start_run(run_name=exp["name"])
    try:
        train_records, val_records = build_records(
            args.db_path if getattr(args, "db_path", None) and os.path.exists(args.db_path) else data_root
        )
        _, val_dataset, val_rel_paths = build_datasets(train_records, val_records)
        pseudo_logits = score_hf_dataset(val_dataset)
        physio_metrics = compute_patient_metrics(
            logits=pseudo_logits,
            labels=np.array([r[1] for r in val_records]),
            rel_paths=val_rel_paths,
            val_records=val_records,
            apply_temperature_scaling=not getattr(
                args, "disable_temperature_scaling", False
            ),
            temperature_min=getattr(args, "temperature_min", 0.8),
            temperature_max=getattr(args, "temperature_max", 3.0),
            temperature_steps=getattr(args, "temperature_steps", 25),
        )
        cal_score = float(
            physio_metrics.get("auc_pr", 0)
        ) - calibrated_score_lambda * float(physio_metrics.get("brier", 0))
        mlflow.log_metric("final_eval_auc_pr", float(physio_metrics.get("auc_pr", 0)))
        mlflow.log_metric("final_eval_brier", float(physio_metrics.get("brier", 0)))
        mlflow.log_metric("final_eval_calibrated_score", cal_score)
        mlflow.log_metric(
            "final_eval_temperature", float(physio_metrics.get("temperature", 1.0))
        )

        results["best_cal_score"] = cal_score
        results["physionet22_val"] = {
            "auc": physio_metrics.get("auc_roc", 0),
            "auc_ci": [
                physio_metrics.get("auc_roc_ci_low", 0),
                physio_metrics.get("auc_roc_ci_high", 0),
            ],
            "auc_pr": physio_metrics.get("auc_pr", 0),
            "brier": physio_metrics.get("brier", 0),
            "cal_score": cal_score,
            "sens": physio_metrics.get("sensitivity", 0),
            "spec": physio_metrics.get("specificity", 0),
            "acc": physio_metrics.get("accuracy", 0),
            "mcc": physio_metrics.get("mcc", 0),
        }

        logger.info(
            "[zero-shot] Starting cross-cohort scoring on WC2023, WVU, OHH-2025"
        )
        cohorts = ["WC2023", "WVU", "OHH-2025"]
        db_path = (
            args.db_path
            if getattr(args, "db_path", None) and os.path.exists(args.db_path)
            else (_locate_db(data_root) or "")
        )

        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path, uri=True)
            c = conn.cursor()
            for cohort in cohorts:
                try:
                    cohort_records, cohort_val_records = [], []
                    use_db = True
                    if cohort == "OHH-2025":
                        if getattr(args, "ohh_folder", None) and os.path.exists(
                            args.ohh_folder
                        ):
                            full_ohh_dir = args.ohh_folder
                        else:
                            full_ohh_dir = os.path.join(
                                data_root,
                                "ohh_2025_dataset",
                                "onehearthealth_2025_per_patient",
                            )
                        if os.path.exists(full_ohh_dir):
                            use_db = (
                                False
                            )
                            for label_str, label_val in [
                                ("Abnormal", 1),
                                ("Normal", 0),
                            ]:
                                d_path = os.path.join(full_ohh_dir, label_str)
                                if not os.path.exists(d_path):
                                    continue
                                for patient_id in os.listdir(d_path):
                                    p_path = os.path.join(d_path, patient_id)
                                    if not os.path.isdir(p_path):
                                        continue
                                    for f in os.listdir(p_path):
                                        if f.endswith(".wav"):
                                            abs_path = os.path.join(p_path, f)
                                            cohort_records.append((abs_path, label_val))
                                            cohort_val_records.append(
                                                (abs_path, label_val, patient_id)
                                            )
                        else:
                            logger.warning(
                                "OHH-2025 folder not found at {}; falling back to the "
                                "SQLite DB copy (same audio - original WAV bytes "
                                "stored losslessly via zstd, scoped to "
                                "project_name='OHH-2025').",
                                full_ohh_dir,
                            )

                    if use_db:
                        c.execute(
                            "SELECT a.uuid, a.patient_id, a.has_murmur, b.audio_file "
                            "FROM audio_info a JOIN audio_raw_zstd b ON a.uuid = b.uuid "
                            "WHERE a.project_name = ? COLLATE NOCASE",
                            (cohort,),
                        )
                        for row in c.fetchall():
                            uuid_str, patient_id, label, audio_zstd = row
                            cohort_records.append((uuid_str, label, audio_zstd))
                            cohort_val_records.append(
                                (uuid_str, label, patient_id, audio_zstd)
                            )

                    if not cohort_records:
                        continue

                    cohort_dataset = FeatureDataset(cohort_records).to_hf_dataset()
                    cohort_pseudo_logits = score_hf_dataset(cohort_dataset)
                    cohort_metrics = compute_patient_metrics(
                        logits=cohort_pseudo_logits,
                        labels=np.array([r[1] for r in cohort_records]),
                        rel_paths=[r[0] for r in cohort_records],
                        val_records=cohort_val_records,
                        apply_temperature_scaling=not getattr(
                            args, "disable_temperature_scaling", False
                        ),
                        temperature_min=getattr(args, "temperature_min", 0.8),
                        temperature_max=getattr(args, "temperature_max", 3.0),
                        temperature_steps=getattr(args, "temperature_steps", 25),
                    )
                    n_patients = len(set(r[2] for r in cohort_val_records))
                    results[cohort] = {
                        "auc": cohort_metrics.get("auc_roc", "-"),
                        "auc_ci": [
                            cohort_metrics.get("auc_roc_ci_low", 0),
                            cohort_metrics.get("auc_roc_ci_high", 0),
                        ],
                        "n_patients": n_patients,
                    }
                except Exception as cohort_err:
                    logger.warning(
                        "[zero-shot] Cohort {} failed, skipping: {}",
                        cohort,
                        cohort_err,
                    )
                    results[f"{cohort}_error"] = str(cohort_err)
            conn.close()
    finally:
        mlflow.end_run()

    return results


def run_prod_pipeline_experiment(
    args, exp, resume_dir, noise_splits, pipeline_output_dir=None
):
    import optuna
    import mlflow
    import torch
    import sqlite3
    from transformers import ASTForAudioClassification

    logger.info(f"Starting PROD AST murmur fine-tuning for experiment: {exp['name']}")

    base_dir = (
        pipeline_output_dir
        if pipeline_output_dir
        else (args.output_dir if hasattr(args, "output_dir") else "checkpoints")
    )
    output_dir = os.path.join(base_dir, exp["name"])
    os.makedirs(output_dir, exist_ok=True)

    if getattr(args, "mlflow_tracking_uri", None):
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(getattr(args, "mlflow_experiment", "unified_prod_experiment"))
    if mlflow.active_run() is not None:
        mlflow.end_run()

    if not torch.cuda.is_available():
        logger.warning("GPU is required for this script.")

    set_seed(getattr(args, "seed", 42))

    cfg = exp.get("cfg", {})

    use_focal = exp.get("focal", False)
    focal_alpha = exp.get("fa", 0.6)
    focal_gamma = exp.get("fg", 2.0)
    active_augs = [k for k, v in cfg.items() if v and k.startswith("use_")]
    noise_lambda = cfg.get("lambda_val", None)
    print(f"\n  [PROD CONFIG] {exp['name']}")
    print(
        f"    Focal Loss:    {use_focal}"
        + (f" (alpha={focal_alpha}, gamma={focal_gamma})" if use_focal else "")
    )
    print(f"    Augmentations: {active_augs if active_augs else 'None (baseline)'}")
    print(f"    Noise Lambda:  {noise_lambda}")
    print(f"    Unfreeze:      {exp.get('unfreeze', False)}")
    print(
        f"    Noise Files:   {len(noise_splits.get('train', []))} available", flush=True
    )

    feature_cfg = FeatureConfig(
        sample_rate=getattr(args, "sample_rate", 16000),
        normalize=not getattr(args, "no_normalize", False),
        remove_dc=not getattr(args, "no_remove_dc", False),
        pre_emphasis=getattr(args, "pre_emphasis", 0.97),
        bandpass_low=getattr(args, "bandpass_low", 20.0),
        bandpass_high=getattr(args, "bandpass_high", 1200.0),
        rms_normalize=not getattr(args, "no_rms_normalize", False),
        rms_target=getattr(args, "rms_target", 0.1),
        trim_silence=not getattr(args, "no_trim_silence", False),
        trim_db=getattr(args, "trim_db", -40.0),
        max_duration_seconds=getattr(args, "max_duration_seconds", 10.0),
        use_spec_augment=cfg.get("use_spec_augment", False),
        spec_freq_mask=cfg.get("spec_freq_mask", 12),
        spec_time_mask=cfg.get("spec_time_mask", 48),
        spec_cardiac_constrained=cfg.get("spec_cardiac_constrained", False),
        use_lung_noise=cfg.get("use_lung_noise", False),
        use_polarity_flip=cfg.get("use_polarity_flip", False),
        use_gain_jitter=cfg.get("use_gain_jitter", False),
        gain_jitter_range=cfg.get("gain_jitter_range", 0.3),
        use_hpss_eq=cfg.get("use_hpss_eq", False),
        use_mixup=cfg.get("use_mixup", False),
        mixup_alpha=cfg.get("mixup_alpha", 0.4),
        use_cardiac_crop=cfg.get("use_cardiac_crop", False),
        use_resp_am=cfg.get("use_resp_am", False),
        resp_am_depth=cfg.get("resp_am_depth", 0.3),
        use_quality_weight=cfg.get("use_quality_weight", False),
        quality_weight_min=cfg.get("quality_weight_min", 0.5),
        lambda_val=cfg.get("lambda_val", None),
        noise_files=noise_splits.get("train", []),
        is_train=True,
    )

    data_root = (
        os.path.dirname(args.db_path)
        if getattr(args, "db_path", None) and os.path.exists(args.db_path)
        else "/kaggle/input/mumurdetection"
    )
    config = TrainConfig(
        data_root=data_root,
        output_dir=output_dir,
        model_name=getattr(
            args, "model_name", "MIT/ast-finetuned-audioset-10-10-0.4593"
        ),
        seed=getattr(args, "seed", 42),
        max_trials=getattr(args, "max_trials", 1),
        patience_trials=getattr(args, "patience_trials", 3),
        epochs=args.epochs,
        feature_cfg=feature_cfg,
        max_length=getattr(args, "max_length", 1024),
        ohh_folder=getattr(args, "ohh_folder", None),
    )

    train_records, val_records = build_records(
        args.db_path if getattr(args, "db_path", None) and os.path.exists(args.db_path) else config.data_root
    )
    train_dataset, val_dataset, val_rel_paths = build_datasets(
        train_records, val_records
    )

    feature_extractor = AutoFeatureExtractor.from_pretrained(config.model_name)

    def compute_metrics_fn(eval_pred):
        logits = eval_pred.predictions
        labels = eval_pred.label_ids
        return compute_patient_metrics(
            logits=logits,
            labels=labels,
            rel_paths=val_rel_paths,
            val_records=val_records,
            apply_temperature_scaling=not getattr(
                args, "disable_temperature_scaling", False
            ),
            temperature_min=getattr(args, "temperature_min", 0.8),
            temperature_max=getattr(args, "temperature_max", 3.0),
            temperature_steps=getattr(args, "temperature_steps", 25),
        )

    def model_init(trial: Optional[optuna.Trial] = None):
        label2id = {"Normal": 0, "Abnormal": 1}
        id2label = {0: "Normal", 1: "Abnormal"}
        model = ASTForAudioClassification.from_pretrained(
            config.model_name,
            num_labels=2,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
        )
        freeze_base = not exp.get("unfreeze", False)
        if exp.get("progressive_unfreeze", False):
            freeze_base = True

        if freeze_base:
            base = getattr(model, "audio_spectrogram_transformer", None)
            if base is not None:
                for param in base.parameters():
                    param.requires_grad = False
        return model

    num_train_samples = len(train_dataset)
    batches_per_epoch = max(1, num_train_samples // args.batch_size)
    eval_steps = max(1, batches_per_epoch // 5)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        label_smoothing_factor=getattr(args, "label_smoothing", 0.05),
        logging_steps=eval_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_auc_pr",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to=["mlflow"],
        remove_unused_columns=False,
        save_total_limit=1,
        save_only_model=True,
    )

    metrics_csv_path = os.path.join(config.output_dir, "metrics_checkpoints.csv")

    trainer_common_kwargs = dict(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorAST(
            feature_extractor=feature_extractor,
            data_root=config.data_root,
            cfg=config.feature_cfg,
            max_length=config.max_length,
            seed=getattr(args, "seed", 42),
        ),
        compute_metrics=compute_metrics_fn,
        callbacks=[
            OptunaMLflowRunCallback(getattr(args, "optuna_study_name", "prod_study")),
            MetricsCheckpointCallback(
                metrics_csv_path,
                calibrated_score_lambda=getattr(args, "calibrated_score_lambda", 0.20),
            ),
            EarlyStoppingCallback(
                early_stopping_patience=getattr(args, "early_stopping_patience", 2)
            ),
        ],
    )

    if exp.get("progressive_unfreeze", False):
        trainer_common_kwargs["callbacks"].append(
            ProgressiveUnfreezeCallback(unfreeze_epoch=3)
        )

    use_mixup = cfg.get("use_mixup", False)
    use_quality_weight = cfg.get("use_quality_weight", False)

    if use_focal:
        trainer = FocalLossTrainer(
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            **trainer_common_kwargs,
        )
    elif use_mixup:
        trainer = MixupTrainer(**trainer_common_kwargs)
    elif use_quality_weight:
        trainer = WeightedCETrainer(**trainer_common_kwargs)
    else:
        trainer = Trainer(**trainer_common_kwargs)

    if config.max_trials > 1:
        logger.info("Starting Optuna hyperparameter search with Trainer")

        def compute_objective(metrics: Dict[str, float]) -> float:
            return float(metrics.get("eval_auc_pr", 0.0))

        def hp_space(trial: optuna.Trial) -> Dict[str, float | int | bool]:
            return {
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-5, 5e-4, log=True
                ),
                "per_device_train_batch_size": trial.suggest_categorical(
                    "per_device_train_batch_size", [16, 64, 128]
                ),
                "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
                "num_train_epochs": trial.suggest_categorical(
                    "num_train_epochs", [1, 5, 10]
                ),
            }

        best_run = trainer.hyperparameter_search(
            direction="maximize",
            backend="optuna",
            n_trials=config.max_trials,
            hp_space=hp_space,
            compute_objective=compute_objective,
            study_name=getattr(args, "optuna_study_name", "prod_study"),
            storage=getattr(args, "optuna_storage", "sqlite:///optuna.db"),
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=getattr(args, "optuna_seed", 42)),
        )
        return {"status": "SUCCESS", "best_objective": best_run.objective}

    trainer.train()
    metrics = trainer.evaluate()

    eval_auc_pr = metrics.get("eval_auc_pr")
    eval_brier = metrics.get("eval_brier")
    eval_temperature = metrics.get("eval_temperature")
    if eval_auc_pr is not None and eval_brier is not None:
        final_calibrated_score = float(eval_auc_pr) - getattr(
            args, "calibrated_score_lambda", 0.20
        ) * float(eval_brier)
        mlflow.log_metric("final_eval_auc_pr", float(eval_auc_pr))
        mlflow.log_metric("final_eval_brier", float(eval_brier))
        mlflow.log_metric("final_eval_calibrated_score", final_calibrated_score)
    if eval_temperature is not None:
        mlflow.log_metric("final_eval_temperature", float(eval_temperature))

    logger.info("Starting blind cross-cohort analysis on WVU, WC2023, OHH-2025")
    cohorts = ["WC2023", "WVU", "OHH-2025"]
    db_path = (
        args.db_path
        if getattr(args, "db_path", None) and os.path.exists(args.db_path)
        else (_locate_db(config.data_root) or "")
    )
    results = {
        "status": "SUCCESS",
        "best_cal_score": metrics.get("eval_calibrated_score", 0),
    }

    results["physionet22_val"] = {
        "auc": metrics.get("eval_auc_roc", 0),
        "auc_ci": [
            metrics.get("eval_auc_roc_ci_low", 0),
            metrics.get("eval_auc_roc_ci_high", 0),
        ],
        "auc_pr": metrics.get("eval_auc_pr", 0),
        "brier": metrics.get("eval_brier", 0),
        "cal_score": metrics.get("eval_calibrated_score", 0),
        "sens": metrics.get("eval_sensitivity", 0),
        "spec": metrics.get("eval_specificity", 0),
        "acc": metrics.get("eval_accuracy", 0),
        "mcc": metrics.get("eval_mcc", 0),
    }

    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path, uri=True)
        c = conn.cursor()
        for cohort in cohorts:
            cohort_records, cohort_val_records = [], []
            use_db = True
            if cohort == "OHH-2025":
                if getattr(config, "ohh_folder", None) and os.path.exists(
                    config.ohh_folder
                ):
                    full_ohh_dir = config.ohh_folder
                else:
                    full_ohh_dir = os.path.join(
                        config.data_root,
                        "ohh_2025_dataset",
                        "onehearthealth_2025_per_patient",
                    )
                if os.path.exists(full_ohh_dir):
                    use_db = False
                    for label_str, label_val in [("Abnormal", 1), ("Normal", 0)]:
                        d_path = os.path.join(full_ohh_dir, label_str)
                        if not os.path.exists(d_path):
                            continue
                        for patient_id in os.listdir(d_path):
                            p_path = os.path.join(d_path, patient_id)
                            if not os.path.isdir(p_path):
                                continue
                            for f in os.listdir(p_path):
                                if f.endswith(".wav"):
                                    abs_path = os.path.join(p_path, f)
                                    cohort_records.append((abs_path, label_val))
                                    cohort_val_records.append(
                                        (abs_path, label_val, patient_id)
                                    )
                else:
                    logger.warning(
                        "OHH-2025 folder not found at {}; falling back to the "
                        "SQLite DB copy (same audio - original WAV bytes stored "
                        "losslessly via zstd, scoped to project_name='OHH-2025').",
                        full_ohh_dir,
                    )

            if use_db:
                c.execute(
                    "SELECT a.uuid, a.patient_id, a.has_murmur, b.audio_file FROM audio_info a JOIN audio_raw_zstd b ON a.uuid = b.uuid WHERE a.project_name = ? COLLATE NOCASE",
                    (cohort,),
                )
                for row in c.fetchall():
                    uuid_str, patient_id, label, audio_zstd = row
                    cohort_records.append((uuid_str, label, audio_zstd))
                    cohort_val_records.append((uuid_str, label, patient_id, audio_zstd))

            if cohort_records:
                cohort_dataset = FeatureDataset(cohort_records).to_hf_dataset()
                preds = trainer.predict(cohort_dataset)
                cohort_metrics = compute_patient_metrics(
                    logits=preds.predictions,
                    labels=preds.label_ids,
                    rel_paths=[r[0] for r in cohort_records],
                    val_records=cohort_val_records,
                    apply_temperature_scaling=not getattr(
                        args, "disable_temperature_scaling", False
                    ),
                    temperature_min=getattr(args, "temperature_min", 0.8),
                    temperature_max=getattr(args, "temperature_max", 3.0),
                    temperature_steps=getattr(args, "temperature_steps", 25),
                )
                n_patients = len(set(r[2] for r in cohort_val_records))
                results[cohort] = {
                    "auc": cohort_metrics.get("auc_roc", "-"),
                    "auc_ci": [
                        cohort_metrics.get("auc_roc_ci_low", 0),
                        cohort_metrics.get("auc_roc_ci_high", 0),
                    ],
                    "n_patients": n_patients,
                }
        conn.close()
    if not getattr(args, "skip_model_save", False):
        trainer.save_model(os.path.join(config.output_dir, "best_model"))
    for checkpoint in glob.glob(os.path.join(config.output_dir, "checkpoint-*")):
        shutil.rmtree(checkpoint, ignore_errors=True)
    if mlflow.active_run() is not None:
        mlflow.end_run()
    return results


if __name__ == "__main__":
    if os.path.exists("/kaggle/working"):
        prev_run = "/kaggle/input/murmur-pipeline-run-1"
        if os.path.exists(prev_run):
            print(
                f"Found previous Kaggle run at {prev_run}. Copying results to resume..."
            )
            prev_global = os.path.join(prev_run, "global_results.json")
            if os.path.exists(prev_global):
                shutil.copy(prev_global, "/kaggle/working/")
                print("Copied global_results.json to resume.")
            for f in os.listdir(prev_run):
                if f.startswith("results_") and f.endswith(".json"):
                    shutil.copy(os.path.join(prev_run, f), "/kaggle/working/")
        sys.argv = [
            "unified_murmur_pipeline.py",
            "--use_kaggle_dataset",
            "--skip_model_save",
            "--run_suite",
        ]
    elif os.path.exists("/content/drive/MyDrive"):
        drive_results_dir = "/content/drive/MyDrive/OHH_Murmur_Results"
        if os.path.exists(drive_results_dir):
            print(
                f"[Colab Resume] Found previous results in Google Drive at {drive_results_dir}. Restoring to /content/..."
            )
            for f in os.listdir(drive_results_dir):
                if (
                    (f.startswith("results_") and f.endswith(".json"))
                    or f == "global_results.json"
                    or (f.startswith("config_") and f.endswith(".json"))
                ):
                    try:
                        shutil.copy(os.path.join(drive_results_dir, f), "/content/")
                    except Exception as e:
                        print(f"[Colab Resume] Warning: failed to copy {f}: {e}")
            print("[Colab Resume] Progress restored.", flush=True)
    main()
