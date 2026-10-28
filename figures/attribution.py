"""Reproduce the attribution figure of the paper from the models trained by ./run.sh.

Run through the launcher after the full run:  ./run.sh --figures   (run.bat --figures on Windows)
or directly:  .venv/bin/python figures/attribution.py all

Integrated Gradients (Captum) explains each recording's murmur score (murmur logit minus no-murmur
logit) relative to the model input for silence: positive values raised the score, negative values
lowered it. The analysis uses CirCor validation patients with a systolic murmur, aligned to the
CirCor S1/systole/S2/diastole annotations, which are downloaded from PhysioNet and checked against
PhysioNet's SHA-256 checksums. Outputs are written to results_reproduced/figures/attribution/.

Stages: labels, check, audit, validate, average, stats, or all (in that order).
"""
import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import sqlite3
import ssl
import sys
import urllib.request

import numpy as np
import torch
from transformers import ASTForAudioClassification

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import unified_murmur_pipeline as P  # noqa: E402

HERE = ROOT  # the pipeline's data folder is ROOT/data
DATA_DIR = os.path.join(ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "hestia_wvu_wc2023_physionet22_ohh2025.db")
RUNS = os.path.join(ROOT, "results_reproduced")
OUT = os.path.join(RUNS, "figures", "attribution")
LABELS = os.path.join(DATA_DIR, "circor_v1.0.3_labels")
CIRCOR_LABELS = os.path.join(LABELS, "tsv")
PHYSIONET = "https://physionet.org/files/circor-heart-sound/1.0.3/"
CONSOLIDATED = os.path.join(ROOT, "results", "kaggle_runs_consolidated.csv")

# Models shown in the figure: (experiment, unused, unused, label); each is loaded from
# results_reproduced/<experiment>/best_model (the same tuple layout as the paper's analysis script).
MODELS = [
    ("01_prod_processing_trainer_frozen", None, "best_model", "Frozen backbone"),
    ("02_prod_processing_trainer_unfrozen", None, "best_model", "Full fine-tuning"),
    ("23_prod_cspec_ultra_light", None, "best_model", "Cardiac SpecAugment (5, 48)"),
    ("29_prod_noisemixed_focal", None, "best_model", "Mixed noise + focal loss"),
    ("39_prod_hpss_focal", None, "best_model", "HPSS-EQ + focal loss"),
]
REFERENCE_MODEL = "39_prod_hpss_focal"  # model used for the randomization sanity check
SR, HOP, WIN, MAX_LEN = 16000, 160, 400, 1024
LIMIT = None  # --limit: use only the first N patients (smoke test)

USE_TEX = shutil.which("latex") is not None
TEXT_STYLE = ({"text.usetex": True, "font.family": "serif", "font.serif": ["Computer Modern Roman"]} if USE_TEX
              else {"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm", "axes.unicode_minus": False,
                    "axes.formatter.use_mathtext": True})


def panel_label(letter, title):
    return (r"\textbf{%s}\quad %s" % (letter, title)) if USE_TEX else (r"$\mathbf{%s}$   %s" % (letter, title))


def device():
    if os.environ.get("ATTRIBUTION_DEVICE"):
        return torch.device(os.environ["ATTRIBUTION_DEVICE"])
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # MPS stalled under memory pressure


def load_model(exp, *_):
    path = os.path.join(RUNS, exp, "best_model")
    if not os.path.exists(os.path.join(path, "config.json")):
        raise SystemExit(f"{path} not found: run ./run.sh first (it trains the models and saves best_model).")
    model, info = ASTForAudioClassification.from_pretrained(path, output_loading_info=True)
    bad = {k: v for k, v in info.items() if v}
    if bad:
        raise RuntimeError(f"{exp}: model did not load cleanly: {bad}")
    return model.float().eval().to(device())


def db_path():
    return DB_PATH


def tsv_path(file_name):
    f = os.path.join(CIRCOR_LABELS, file_name.replace(".wav", ".tsv"))
    return f if file_name and os.path.exists(f) else None


def published():
    return {r["experiment"]: r for r in csv.DictReader(open(CONSOLIDATED))}


def _get(url):
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    with urllib.request.urlopen(url, timeout=120, context=ctx) as r:
        return r.read()


def cmd_labels(args):
    """Download training_data.csv and the segmentation files of the validation recordings from PhysioNet."""
    os.makedirs(CIRCOR_LABELS, exist_ok=True)
    sums = dict(reversed(l.split(" ", 1)) for l in _get(PHYSIONET + "SHA256SUMS.txt").decode().splitlines() if " " in l)

    def fetch(rel, dest):
        if os.path.exists(dest) and hashlib.sha256(open(dest, "rb").read()).hexdigest() == sums[rel]:
            return
        data = _get(PHYSIONET + rel)
        if hashlib.sha256(data).hexdigest() != sums[rel]:
            raise SystemExit(f"{rel}: checksum mismatch")
        with open(dest, "wb") as fh:
            fh.write(data)

    fetch("training_data.csv", os.path.join(LABELS, "training_data.csv"))
    meta = circor_meta()
    names = [meta[r[0]][0].replace(".wav", ".tsv") for r in circor_val_records()]
    for n in names:
        if f"training_data/{n}" in sums:
            fetch(f"training_data/{n}", os.path.join(CIRCOR_LABELS, n))
    print(f"labels ready: training_data.csv and {len(os.listdir(CIRCOR_LABELS))} segmentation files", flush=True)


def cmd_check(args):
    """Validation AUROC of each trained model, next to the value in the paper."""
    os.makedirs(OUT, exist_ok=True)
    collator, _ = make_collator()
    recs, pub, rows = circor_val_records(), published(), []
    for exp, _, _, label in MODELS:
        model = load_model(exp)
        logits = predict(model, collator, recs)
        m = P.compute_patient_metrics(logits=logits, labels=np.array([r[1] for r in recs]), rel_paths=[r[0] for r in recs],
                                      val_records=[r[:3] for r in recs], apply_temperature_scaling=True, bootstrap_samples=1)
        ref = float(pub[exp]["physionet22_val_auc"])
        rows.append({"experiment": exp, "val_auroc_reproduced": m["auc_roc"], "val_auroc_paper": ref})
        print(f"{exp:40s} validation AUROC {m['auc_roc']:.4f} (paper {ref:.4f})", flush=True)
        del model
        free_memory()
    with open(os.path.join(OUT, "check_auroc.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def cmd_stats(args):
    """Numbers quoted in the paper's attribution analysis, written to paper_numbers.json."""
    import statistics as st

    order = [e[:2] for e in LEFT + RIGHT]
    audit = list(csv.DictReader(open(os.path.join(OUT, "alignment_audit.csv"))))
    ok = [r for r in audit if r["aligned"] == "True" and int(r["cycles_used"]) > 0]
    summ = json.load(open(os.path.join(OUT, "cycle_average_summary.json")))
    rows = list(csv.DictReader(open(os.path.join(OUT, "validate_per_recording.csv"))))
    out = {"eligible_patients": len(audit), "aligned_patients": len(ok), "cycles": summ["n_cycles"],
           "mean_phase_ms": dict(zip(["S1", "systole", "S2", "diastole"], summ["mean_phase_ms"])),
           "max_completeness_error": max(float(r["ig_rel_delta"]) for r in rows)}
    hz = mel_centers_hz()
    nb = int((hz <= CARDIAC_HZ).sum())
    band = (hz[:nb] >= 100) & (hz[:nb] <= 250)
    cfg = feature_config()
    recs = {r[0][0]: r for r in aligned_recordings()}
    for k in order:
        r = [x for x in rows if x["experiment"].startswith(k)]
        ratio = lambda ph: [float(x[f"attr_share_{ph}"]) / float(x[f"time_share_{ph}"]) for x in r]
        m = {"model": r[0]["model"], "n": len(r)}
        for ph in ("S1", "systole", "S2", "diastole"):
            m[f"{ph}_ratio_median"] = round(st.median(ratio(ph)), 2)
        m["systole_ratio_iqr"] = [round(float(v), 2) for v in np.percentile(ratio("systole"), [25, 75])]
        m["systole_ratio_above_1"] = sum(v > 1 for v in ratio("systole"))
        m["S1_ratio_above_1"] = sum(v > 1 for v in ratio("S1"))
        m["diastole_ratio_below_1"] = sum(v < 1 for v in ratio("diastole"))
        m["deletion_top_gt_random"] = sum(float(x["drop_top10"]) > float(x["drop_random10_mean"]) for x in r)
        share, medf = [], []
        for x in r:
            rec, fn, _ = recs[x["uuid"]]
            ig = np.load(os.path.join(OUT, "validate", f"{x['experiment']}_{x['uuid']}.npz"))["ig_band"][:, :nb]
            pos = np.clip(ig[frame_states(fn, trim_offset_s(rec, cfg), ig.shape[0]) == 2], 0, None)
            if pos.sum() > 0:
                share.append(float(pos[:, band].sum() / pos.sum()))
                c = np.cumsum(pos.sum(0)) / pos.sum()
                medf.append(float(hz[np.searchsorted(c, 0.5)]))
        m["systole_share_100_250Hz_median"] = round(st.median(share), 2)
        m["systole_median_frequency_hz"] = round(st.median(medf))
        out[k] = m
    san = list(csv.DictReader(open(os.path.join(OUT, "validate_sanity.csv"))))
    for what in sorted({s["randomized"] for s in san}):
        v = [float(s["spearman_vs_trained"]) for s in san if s["randomized"] == what]
        out[f"sanity: {what}"] = [round(min(v), 2), round(max(v), 2)]
    json.dump(out, open(os.path.join(OUT, "paper_numbers.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def _limited(fn):
    def wrapper():
        recs = fn()
        return recs[:LIMIT] if LIMIT else recs
    return wrapper


CARDIAC_HZ = 1200.0


def free_memory():
    import gc

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def feature_config():
    # Eval path: every augmentation flag off; preprocessing as in run_prod_pipeline_experiment.
    return P.FeatureConfig(max_duration_seconds=10.0)


def make_collator():
    fe = P.AutoFeatureExtractor.from_pretrained(P.DEFAULT_MODEL)
    return P.DataCollatorAST(fe, data_root="", cfg=feature_config(), max_length=MAX_LEN), fe


_SPLIT = None


def split_records():
    """The pipeline's own CirCor train/validation split, built once per process."""
    global _SPLIT
    if _SPLIT is None:
        _SPLIT = P.build_records(os.path.join(HERE, "data"))
    return _SPLIT


def circor_val_records():
    return split_records()[1]


def cohort_records(cohort):
    conn = sqlite3.connect(db_path())
    rows = conn.execute(
        "SELECT a.uuid, a.patient_id, a.has_murmur, b.audio_file FROM audio_info a "
        "JOIN audio_raw_zstd b ON a.uuid = b.uuid WHERE a.project_name = ? COLLATE NOCASE",
        (cohort,),
    ).fetchall()
    conn.close()
    return [(u, lab, pid, blob) for u, pid, lab, blob in rows]


def item(rec):
    return {"rel_path": rec[0], "label": rec[1], "audio_bytes": rec[3], "is_train": False}


@torch.no_grad()
def predict(model, collator, records, batch=8):
    out = []
    for i in range(0, len(records), batch):
        b = collator([item(r) for r in records[i : i + batch]])
        out.append(model(input_values=b["input_values"].to(device())).logits.float().cpu().numpy())
    return np.concatenate(out)


def trim_offset_s(rec, cfg):
    """Seconds removed from the start by trim_silence, recomputed with the pipeline's own steps."""
    import soundfile as sf

    data, sr = sf.read(io.BytesIO(P.zstd.ZstdDecompressor().decompress(rec[3])))
    if data.ndim > 1:
        data = data.mean(axis=1)
    w = torch.from_numpy(data).float().unsqueeze(0)
    w = P.resample_if_needed(w, sr, cfg.sample_rate)
    w = P.remove_dc_offset(w)
    w = P.apply_bandpass(w, cfg.sample_rate, cfg.bandpass_low, cfg.bandpass_high)
    w = P.apply_pre_emphasis(w, cfg.pre_emphasis)
    thr = w.abs().max() * (10 ** (cfg.trim_db / 20))
    start = int((w.abs() >= thr).any(dim=0).nonzero()[0].item())
    trimmed = P.trim_silence(w, cfg.trim_db)
    assert torch.equal(trimmed[:, : min(100, trimmed.shape[1])], w[:, start : start + min(100, trimmed.shape[1])])
    return start / cfg.sample_rate


def n_valid_frames(collator, rec):
    wav = collator._prepare_waveform(item(rec))
    return max(0, min(MAX_LEN, 1 + (len(wav) - WIN) // HOP)), len(wav)


def silence_baseline(fe, x, n_valid):
    sil = fe([np.zeros(SR * 10, dtype=np.float32)], sampling_rate=SR, return_tensors="pt",
             padding="max_length", truncation=True, max_length=MAX_LEN)["input_values"]
    base = x.clone()
    base[:, :n_valid, :] = sil[:, :n_valid, :].to(x.device)
    return base


def mel_centers_hz(n_mels=128, low=20.0, high=SR / 2):
    mel = lambda f: 1127.0 * np.log(1 + f / 700.0)  # Kaldi mel scale, as in the AST feature extractor
    inv = lambda m: 700.0 * (np.exp(m / 1127.0) - 1)
    pts = np.linspace(mel(low), mel(high), n_mels + 2)
    return inv(pts[1:-1])


def circor_meta():
    conn = sqlite3.connect(db_path())
    meta = {u: (fn, loc, heard) for u, fn, loc, heard in conn.execute(
        "SELECT uuid, audio_file_name, recording_location, murmur_heard_this_location FROM audio_info "
        "WHERE project_name='physionet22' COLLATE NOCASE")}
    conn.close()
    return meta


BLOCK = 16


STATES = {1: "S1", 2: "systole", 3: "S2", 4: "diastole"}


def eligible_validation_recordings():
    labels = {r["Patient ID"]: r for r in csv.DictReader(open(os.path.join(ROOT, "data", "circor_v1.0.3_labels",
                                                                          "training_data.csv")))}
    meta = circor_meta()
    by_patient = {}
    for r in circor_val_records():
        t = labels[r[2]]
        if t["Murmur"] != "Present" or t["Systolic murmur timing"] in ("nan", "") \
                or t["Diastolic murmur timing"] not in ("nan", ""):
            continue
        fn = meta[r[0]][0]
        loc = fn.split("_")[1].split(".")[0]
        if loc not in t["Murmur locations"].split("+") or tsv_path(fn) is None:
            continue
        rank = 0 if loc == t["Most audible location"] else 1
        if r[2] not in by_patient or rank < by_patient[r[2]][0]:
            by_patient[r[2]] = (rank, r, fn, t["Systolic murmur timing"])
    return [v[1:] for _, v in sorted(by_patient.items())]


def aligned_recordings():
    """Eligible recordings that passed the alignment audit (stage 'audit'): heart-sound energy peaks
    within 20 ms of the annotated S1/S2 and at least one complete cycle inside the input window."""
    f = os.path.join(OUT, "alignment_audit.csv")
    if not os.path.exists(f):
        raise SystemExit("run the 'audit' stage first")
    ok = {r["file"] for r in csv.DictReader(open(f)) if r["aligned"] == "True" and int(r["cycles_used"]) > 0}
    return [r for r in eligible_validation_recordings() if r[1] in ok]


def frame_states(file_name, t0, n_valid):
    """TSV state (0 = not annotated) of every valid frame, from the frame centre time."""
    st = np.zeros(n_valid, dtype=int)
    centres = (np.arange(n_valid) * HOP + WIN / 2) / SR + t0
    with open(tsv_path(file_name)) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 3:
                st[(centres >= float(parts[0])) & (centres < float(parts[1]))] = int(parts[2])
    return st


def block_sums(a, n_valid, n_rows):
    nt = n_valid // BLOCK
    return a[: nt * BLOCK, : n_rows * BLOCK].reshape(nt, BLOCK, n_rows, BLOCK).sum(axis=(1, 3))


def cmd_validate(args):
    from captum.attr import IntegratedGradients

    vdir = os.path.join(OUT, "validate")
    os.makedirs(vdir, exist_ok=True)
    collator, fe = make_collator()
    cfg = feature_config()
    recs = aligned_recordings()
    global _SPLIT
    _SPLIT = None  # keep only the audio of the recordings explained here
    free_memory()
    # 16x16 blocks for the deletion test and the sanity check: 4 block rows = Mel bins 0-63, i.e. about
    # 0-1.8 kHz (the band-pass ends at 1.2 kHz with a 2nd-order roll-off). Heart-cycle shares use 0-1.2 kHz.
    n_rows = int(np.ceil((mel_centers_hz() <= CARDIAC_HZ).sum() / BLOCK))
    print(f"{len(recs)} eligible, aligned recordings (one per patient)", flush=True)
    rows = []
    models = [m for m in MODELS if not args.models or m[0][:2] in args.models]
    for exp, v, ck, label in models:
        model = load_model(exp, v, ck)
        fwd = lambda x: (lambda z: z[:, 1] - z[:, 0])(model(input_values=x).logits)
        ig = IntegratedGradients(fwd)
        for rec, fn, timing in recs:
            f = os.path.join(vdir, f"{exp}_{rec[0]}.npz")
            x = collator([item(rec)])["input_values"].to(device())
            n_valid, _ = n_valid_frames(collator, rec)
            base = silence_baseline(fe, x, n_valid)
            if not os.path.exists(f):
                steps = args.n_steps
                while True:
                    attr, delta = ig.attribute(x, baselines=base, n_steps=steps, internal_batch_size=args.ibs,
                                               return_convergence_delta=True)
                    with torch.no_grad():
                        fx, fb = fwd(x).item(), fwd(base).item()
                    # Completeness error relative to the attributed change, floored at 1 logit so that
                    # recordings scored almost like silence do not demand ever more steps.
                    rel = abs(delta.item()) / max(abs(fx - fb), 1.0)
                    if rel <= 0.05 or steps >= 256:
                        break
                    steps *= 2
                a = attr.detach().cpu().numpy()[0]
                blocks = block_sums(a, n_valid, n_rows)
                # Deletion test on 16x16 blocks inside the valid frames, Mel bins 0-63 (about 0-1.8 kHz).
                nt = blocks.shape[0]
                k = max(1, int(round(0.10 * blocks.size)))
                order = np.argsort(blocks.ravel())[::-1]
                rng = np.random.default_rng(int(rec[0][:8], 16) % (2**32))
                sets = [order[:k]] + [rng.choice(blocks.size, k, replace=False) for _ in range(20)]
                xs = []
                for idx in sets:
                    xm = x.clone()
                    for b in idx:
                        ti, fi = divmod(int(b), n_rows)
                        sl = (slice(None), slice(ti * BLOCK, (ti + 1) * BLOCK), slice(fi * BLOCK, (fi + 1) * BLOCK))
                        xm[sl] = base[sl]
                    xs.append(xm)
                with torch.no_grad():
                    m = torch.cat([fwd(torch.cat(xs[i : i + 3])) for i in range(0, len(xs), 3)]).cpu().numpy()
                np.savez_compressed(f, blocks=blocks, frame_attr=np.clip(a[:n_valid, : n_rows * BLOCK], 0, None).sum(1),
                                    ig_band=a[:n_valid, : n_rows * BLOCK].astype(np.float32),
                                    x_band=x.cpu().numpy()[0, :n_valid, : n_rows * BLOCK].astype(np.float32),
                                    fx=fx, fb=fb, rel_delta=rel, abs_delta=abs(delta.item()), steps=steps, drop_top=fx - m[0],
                                    drop_random=fx - m[1:], n_valid=n_valid)
                del attr, delta, xs, xm
                free_memory()
            d = np.load(f)
            st = frame_states(fn, trim_offset_s(rec, cfg), int(d["n_valid"]))
            nb_c = int((mel_centers_hz() <= CARDIAC_HZ).sum())
            pos = np.clip(d["ig_band"][:, :nb_c], 0, None).sum(1)  # positive attribution, 0-1,200 Hz
            ann = st > 0
            row = {"experiment": exp, "model": label, "uuid": rec[0], "patient_id": rec[2], "file": fn,
                   "timing": timing, "margin": float(d["fx"]), "margin_baseline": float(d["fb"]),
                   "ig_abs_delta": float(d["abs_delta"]) if "abs_delta" in d else float("nan"),
                   "ig_rel_delta": float(d["rel_delta"]),
                   "ig_steps": int(d["steps"]), "annotated_fraction": float(ann.mean()),
                   "drop_top10": float(d["drop_top"]), "drop_random10_mean": float(np.mean(d["drop_random"]))}
            for s_, name in STATES.items():
                time_share = (st == s_).sum() / max(ann.sum(), 1)
                attr_share = pos[st == s_].sum() / max(pos[ann].sum(), 1e-12)
                row[f"time_share_{name}"] = float(time_share)
                row[f"attr_share_{name}"] = float(attr_share)
            rows.append(row)
            print(f"{exp[:24]:24s} {rec[2]:>6s} {timing[:12]:12s} margin={row['margin']:+.2f} "
                  f"delta={row['ig_rel_delta']:.3f} systole attr/time={row['attr_share_systole']:.2f}/"
                  f"{row['time_share_systole']:.2f} drop top/random={row['drop_top10']:+.2f}/"
                  f"{row['drop_random10_mean']:+.2f}", flush=True)
        del model, ig, fwd
        free_memory()
    with open(os.path.join(OUT, "validate_per_recording.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if args.sanity:
        sanity_randomization(args, recs[: args.sanity], collator, fe, n_rows)


def reinit_top(model, n_layers):
    """Re-initialize the classifier and the top n_layers encoder layers in place, and check it worked."""
    ast_ = model.audio_spectrogram_transformer  # transformers 5.x: .layers; 4.x: .encoder.layer
    layers = ast_.layers if hasattr(ast_, "layers") else ast_.encoder.layer
    targets = [model.classifier] + (list(layers)[len(layers) - n_layers:] if n_layers else [])
    top_linear = next(mod for mod in layers[-1].modules() if isinstance(mod, torch.nn.Linear))
    before = [model.classifier.dense.weight.detach().clone(), top_linear.weight.detach().clone()]
    for mod in targets:
        # transformers 5.x skips parameters flagged as already initialized by from_pretrained
        for prm in mod.parameters():
            prm.__dict__.pop("_is_hf_initialized", None)
        mod.apply(model._init_weights)
    assert not torch.allclose(before[0], model.classifier.dense.weight), "classifier not re-initialized"
    if n_layers:
        assert not torch.allclose(before[1], top_linear.weight), "top layer not re-initialized"


def sanity_randomization(args, recs, collator, fe, n_rows):
    """Re-initialize the classifier and top encoder layers of the reference model; maps should change."""
    from captum.attr import IntegratedGradients
    from scipy.stats import spearmanr

    exp, v, ck, _ = next(m for m in MODELS if m[0] == REFERENCE_MODEL)
    out = []
    # None = trained model recomputed with the same IG settings (control: rho should be close to 1).
    for n_layers in (None, 0, 6, 12):
        model = load_model(exp, v, ck)
        torch.manual_seed(0)
        ast_ = model.audio_spectrogram_transformer  # transformers 5.x: .layers; 4.x: .encoder.layer
        layers = ast_.layers if hasattr(ast_, "layers") else ast_.encoder.layer
        if n_layers is not None:
            reinit_top(model, n_layers)
        fwd = lambda x: (lambda z: z[:, 1] - z[:, 0])(model(input_values=x).logits)
        ig = IntegratedGradients(fwd)
        for rec, fn, _ in recs:
            x = collator([item(rec)])["input_values"].to(device())
            n_valid, _ = n_valid_frames(collator, rec)
            base = silence_baseline(fe, x, n_valid)
            a = ig.attribute(x, baselines=base, n_steps=args.n_steps, internal_batch_size=args.ibs)
            rnd = block_sums(a.detach().cpu().numpy()[0], n_valid, n_rows).ravel()
            orig = np.load(os.path.join(OUT, "validate", f"{exp}_{rec[0]}.npz"))["blocks"].ravel()
            rho = float(spearmanr(orig, rnd).correlation)
            what = "none (trained, same IG settings)" if n_layers is None else f"classifier + top {n_layers} layers"
            out.append({"randomized": what, "uuid": rec[0], "spearman_vs_trained": rho})
            print(f"sanity: re-initialized {what}, {rec[2]}: block-level rho vs trained = {rho:+.2f}", flush=True)
        del model
    with open(os.path.join(OUT, "validate_sanity.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)


PHASE_LIMITS = {1: (0.04, 0.25), 2: (0.05, 0.50), 3: (0.03, 0.20), 4: (0.08, 1.50)}


CYCLE_LIMITS = (60 / 200, 60 / 40)


def cycles(file_name, t0, n_valid, shift=0.0, return_times=False):
    """Frame ranges [(i0, i1) per phase] of complete, contiguous, physiologically plausible
    S1-systole-S2-diastole cycles that lie inside the model input window."""
    centres = (np.arange(n_valid) * HOP + WIN / 2) / SR + t0 + shift
    with open(tsv_path(file_name)) as fh:
        seg = [(float(p[0]), float(p[1]), int(p[2])) for p in (l.split() for l in fh) if len(p) == 3]
    out, times = [], []
    for i in range(len(seg) - 3):
        c = seg[i : i + 4]
        if [x[2] for x in c] != [1, 2, 3, 4]:
            continue
        if any(abs(c[j][1] - c[j + 1][0]) > 1e-3 for j in range(3)):  # phases must touch
            continue
        if any(not (PHASE_LIMITS[st][0] <= b - a <= PHASE_LIMITS[st][1]) for a, b, st in c):
            continue
        if not (CYCLE_LIMITS[0] <= c[3][1] - c[0][0] <= CYCLE_LIMITS[1]):
            continue
        if c[0][0] < centres[0] or c[3][1] > centres[-1]:  # whole cycle inside the window
            continue
        ranges = []
        for a, b, _ in c:
            idx = np.where((centres >= a) & (centres < b))[0]
            ranges.append((idx[0], idx[-1] + 1) if len(idx) >= 2 else None)
        if all(ranges):
            out.append(ranges)
            times.append([b - a for a, b, _ in c])
    return (out, times) if return_times else out


def warp(m, ranges, bins):
    """Resample each phase's frames of map m (time x freq) to a fixed number of bins by area
    averaging: frame j covers [j, j+1); output bin b of a phase covers an equal share of the
    phase, so every frame contributes in proportion to its overlap (no frame skipped or reused
    unevenly, whether the phase is stretched or compressed)."""
    parts = []
    for (i0, i1), n in zip(ranges, bins):
        seg = m[i0:i1]
        cum = np.vstack([np.zeros((1, seg.shape[1])), np.cumsum(seg, axis=0)])  # integral at frame edges
        e = np.linspace(0, len(seg), n + 1)
        integ = np.stack([np.interp(e, np.arange(len(seg) + 1), cum[:, k]) for k in range(seg.shape[1])], axis=1)
        parts.append(np.diff(integ, axis=0) / np.diff(e)[:, None])
    return np.concatenate(parts)


LEFT = ["01", "02"]


RIGHT = ["39", "23", "29"]


def cmd_average(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vdir = os.path.join(OUT, "validate")
    cfg = feature_config()
    order = LEFT + RIGHT
    models = sorted([m for m in MODELS if m[0][:2] in order], key=lambda m: order.index(m[0][:2]))
    assert [m[0][:2] for m in models] == order, f"panel models missing from MODELS: {order}"
    hz = mel_centers_hz()
    nb = int((hz <= CARDIAC_HZ).sum())
    # Cycles of every audited recording that has a map for every model.
    info, times = {}, []
    for rec, fn, _ in aligned_recordings():
        fs = [os.path.join(vdir, f"{m[0]}_{rec[0]}.npz") for m in models]
        if not all(os.path.exists(f) for f in fs):
            continue
        nv = int(np.load(fs[0])["n_valid"])
        cyc, t = cycles(fn, trim_offset_s(rec, cfg), nv, return_times=True)
        if cyc:
            info[rec[0]] = cyc
            times += t
    n_bad = sum(float(np.load(os.path.join(vdir, f"{m[0]}_{u}.npz"))["rel_delta"]) > 0.05 for u in info for m in models)
    print(f"maps over the 5% completeness criterion: {n_bad}", flush=True)
    # Template: bins per phase in proportion to the mean phase durations of all cycles used.
    dur = np.mean(times, axis=0)
    bins = np.maximum(4, np.round(100 * dur / dur.sum()).astype(int))
    spec, maps = [], {m[0]: [] for m in models}
    for u, cyc in info.items():
        for k, (exp, *_) in enumerate(models):
            d = np.load(os.path.join(vdir, f"{exp}_{u}.npz"))
            ig = d["ig_band"][:, :nb]
            ig = ig / max(np.abs(ig).mean(), 1e-12)  # each recording counts equally
            maps[exp].append(np.mean([warp(ig, c, bins) for c in cyc], axis=0))  # patient average first
            if k == 0:
                spec.append(np.mean([warp(d["x_band"][:, :nb], c, bins) for c in cyc], axis=0))
    edges = np.concatenate([[0], np.cumsum(bins)])
    means = {exp: np.mean(maps[exp], 0) for exp, *_ in models}
    lo = int(np.searchsorted(hz, args.fmin))
    hi = min(int(np.searchsorted(hz, args.fmax, side="right")), nb)
    lim = float(max(np.percentile(np.abs(m[:, lo:hi]), 99) for m in means.values()))  # one scale for all models
    summary = {"n_recordings": len(info), "n_cycles": len(times), "mean_phase_ms": (1000 * dur).round(1).tolist(),
               "bins": bins.tolist(), "color_limit": lim}
    for exp, m in means.items():
        pos = np.clip(m, 0, None).sum(1)
        summary[exp] = {nm: float(pos[edges[i] : edges[i + 1]].sum() / pos.sum()) for i, nm in enumerate(STATES.values())}
    plt.rcParams.update({**TEXT_STYLE,
                         "font.size": 8, "axes.linewidth": 1.0, "xtick.major.width": 0.9, "ytick.major.width": 0.9,
                         "xtick.direction": "out", "ytick.direction": "out"})
    label = {m[0]: m[3] for m in models}
    left = [("A", "Mean input", np.mean(spec, 0), "magma", None)] + \
           [("BC"[i], label[e], means[e], "RdBu_r", lim) for i, e in enumerate(m[0] for m in models[: len(LEFT)])]
    right = [("DEF"[i], label[e], means[e], "RdBu_r", lim) for i, e in enumerate(m[0] for m in models[len(LEFT) :])]
    # Mel channels that receive no energy (narrower than the FFT resolution at low frequencies) are
    # constant in every input; they are drawn grey instead of as signal.
    _, fe = make_collator()
    noise = np.random.default_rng(0).standard_normal(SR * 10).astype(np.float32) * 0.1
    probe = fe([noise], sampling_rate=SR, return_tensors="np", padding="max_length", truncation=True,
               max_length=MAX_LEN)["input_values"][0][:900]
    empty = np.where(probe.std(0)[:nb] < 1e-3)[0]  # empty channel: std ~5e-6 (rounding); real channels >= 0.1
    summary["empty_mel_channels_hz"] = [round(float(hz[k]), 1) for k in empty]
    yt = [f for f in (50, 100, 200, 300, 500, 700, 1000) if args.fmin <= f <= args.fmax]
    ytick_pos = [np.interp(f, hz[:nb], np.arange(nb)) for f in yt]  # Mel bins are not evenly spaced in Hz

    def column(panels, name):
        # Fixed geometry (inches) so the two images line up exactly when placed side by side.
        W, ph, top, gap, bottom, left_m, cb_w, cb_gap, right_m = 3.3, 1.18, 0.22, 0.27, 0.28, 0.5, 0.07, 0.06, 0.28
        H = top + 3 * ph + 2 * gap + bottom
        fig = plt.figure(figsize=(W, H))
        aw = W - left_m - cb_gap - cb_w - right_m
        for k, (letter, title, m, cmap, l) in enumerate(panels):
            y0 = H - top - (k + 1) * ph - k * gap
            ax = fig.add_axes([left_m / W, y0 / H, aw / W, ph / H])
            cax = fig.add_axes([(left_m + aw + cb_gap) / W, y0 / H, cb_w / W, ph / H])
            kw = dict(vmin=-l, vmax=l) if l else {}
            shown = np.ma.masked_array(m[:, lo:hi].T.copy(), mask=False)
            shown.mask[[k_ - lo for k_ in empty if lo <= k_ < hi], :] = True
            cm = plt.get_cmap(cmap).copy()
            cm.set_bad("0.82")
            im = ax.imshow(shown, origin="lower", aspect="auto", cmap=cm, interpolation="nearest",
                           extent=[0, edges[-1], lo - 0.5, hi - 0.5], **kw)
            for e in edges[1:-1]:
                ax.axvline(e, color="white" if l is None else "0.35", lw=0.6, ls=(0, (3, 2)))
            ax.set_yticks(ytick_pos, [str(f) for f in yt])
            ax.set_ylabel("Hz", labelpad=2)
            ax.set_xticks((edges[:-1] + edges[1:]) / 2, ["S1", "Systole", "S2", "Diastole"] if k == 2 else [])
            ax.tick_params(axis="x", length=0)
            ax.text(0.0, 1.05, panel_label(letter, title), transform=ax.transAxes, ha="left")
            c = fig.colorbar(im, cax=cax)
            c.outline.set_linewidth(0.7)
            if l:
                c.set_ticks([-l, 0, l], labels=["$-$", "0", "$+$"])
            else:
                c.set_ticks([])
        for e in ("pdf", "png"):
            fig.savefig(os.path.join(OUT, f"{name}.{e}"), dpi=300)
        plt.close(fig)

    column(left, "Fig_attribution_left_ABC")
    if len(right) == 3:
        column(right, "Fig_attribution_right_DEF")
    print(json.dumps(summary, indent=1))
    json.dump(summary, open(os.path.join(OUT, "cycle_average_summary.json"), "w"), indent=1)


def cmd_audit(args):
    """Check the heart-cycle alignment of every eligible recording without running a model."""
    collator, _ = make_collator()
    cfg = feature_config()
    hz = mel_centers_hz()
    low = np.where((hz >= 25) & (hz <= 200))[0]  # S1/S2 energy band
    rows = []
    for rec, fn, timing in eligible_validation_recordings():
        x = collator([item(rec)])["input_values"][0].numpy()
        nv, _ = n_valid_frames(collator, rec)
        t0 = trim_offset_s(rec, cfg)
        e = x[:nv, low].mean(1)
        contrast = {}
        for sh in np.round(np.arange(-0.06, 0.0601, 0.01), 2):
            st = frame_states(fn, t0 + sh, nv)
            hs, dia = e[(st == 1) | (st == 3)], e[st == 4]
            contrast[sh] = float(hs.mean() - dia.mean()) if len(hs) and len(dia) else float("nan")
        best = max(contrast, key=lambda k: contrast[k])
        cyc, times = cycles(fn, t0, nv, return_times=True)
        with open(tsv_path(fn)) as fh:
            seg_all = [p for p in (l.split() for l in fh) if len(p) == 3]
        n_raw = sum(1 for i in range(len(seg_all) - 3) if [int(z[2]) for z in seg_all[i : i + 4]] == [1, 2, 3, 4])
        t = np.array(times) if times else np.full((1, 4), np.nan)
        rows.append({"patient_id": rec[2], "file": fn, "timing": timing, "trim_offset_s": round(t0, 4),
                     "cycles_in_tsv": n_raw, "cycles_used": len(cyc),
                     "heart_rate_bpm": round(float(60 / t.sum(1).mean()), 1) if times else float("nan"),
                     **{f"{n}_ms": round(float(t[:, i].mean() * 1000), 1) for i, n in enumerate(STATES.values())},
                     "contrast_at_0": round(contrast[0.0], 3), "best_shift_s": best,
                     "aligned": bool(abs(best) <= 0.02 and contrast[0.0] > 0)})
        r = rows[-1]
        print(f"{r['patient_id']:>6s} {fn:14s} cycles {r['cycles_used']:2d}/{r['cycles_in_tsv']:2d} HR {r['heart_rate_bpm']:5.1f} "
              f"S1 {r['S1_ms']:5.1f} sys {r['systole_ms']:5.1f} S2 {r['S2_ms']:5.1f} dia {r['diastole_ms']:6.1f} ms  "
              f"best shift {best:+.2f}s contrast {r['contrast_at_0']:+.2f} {'OK' if r['aligned'] else 'CHECK'}", flush=True)
    with open(os.path.join(OUT, "alignment_audit.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r["aligned"] and r["cycles_used"] > 0]
    print(f"{len(ok)}/{len(rows)} recordings aligned with usable cycles; "
          f"total cycles used {sum(r['cycles_used'] for r in ok)}")


eligible_validation_recordings = _limited(eligible_validation_recordings)


def cmd_all(args):
    for stage in (cmd_labels, cmd_check, cmd_audit, cmd_validate, cmd_average, cmd_stats):
        print(f"=== {stage.__name__[4:]}", flush=True)
        stage(args)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Reproduce the attribution figure.")
    ap.add_argument("stage", choices=["labels", "check", "audit", "validate", "average", "stats", "all"])
    ap.add_argument("--n_steps", type=int, default=32, help="initial Integrated Gradients steps (doubled up to 256)")
    ap.add_argument("--ibs", type=int, default=2, help="Integrated Gradients internal batch size")
    ap.add_argument("--models", nargs="*", default=[], help="experiment number prefixes for 'validate'")
    ap.add_argument("--sanity", type=int, default=3, help="recordings for the randomization check (0 = skip)")
    ap.add_argument("--fmin", type=float, default=25.0)
    ap.add_argument("--fmax", type=float, default=1000.0)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N patients (smoke test)")
    a = ap.parse_args()
    LIMIT = a.limit or None
    os.makedirs(OUT, exist_ok=True)
    {"labels": cmd_labels, "check": cmd_check, "audit": cmd_audit, "validate": cmd_validate,
     "average": cmd_average, "stats": cmd_stats, "all": cmd_all}[a.stage](a)
