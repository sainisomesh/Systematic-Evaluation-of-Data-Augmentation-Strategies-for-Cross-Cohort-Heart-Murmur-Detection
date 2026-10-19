"""Measure how much gain jitter changes the model input.

Usage: python analysis/check_gain_noop.py DATA_DIR

Applies the pipeline's preprocessing to five CirCor recordings with and without a gain factor
and reports the largest difference in the waveform and in the AST input features.
"""

import glob
import io
import sqlite3
import sys

import numpy as np
import soundfile as sf
import torch
import torchaudio
import zstandard as zstd
from transformers import AutoFeatureExtractor

SAMPLE_RATE = 16000


def remove_dc_offset(w):
    return w - w.mean(dim=1, keepdim=True)


def apply_bandpass(w, sr, low_hz=20.0, high_hz=1200.0):
    w = torchaudio.functional.highpass_biquad(w, sr, low_hz)
    return torchaudio.functional.lowpass_biquad(w, sr, high_hz)


def apply_pre_emphasis(w, coeff=0.97):
    out = w.clone()
    out[:, 1:] = w[:, 1:] - coeff * w[:, :-1]
    return out


def trim_bounds(w, trim_db=-40.0):
    peak = w.abs().max()
    if peak == 0:
        return 0, w.shape[-1]
    idx = (w.abs() >= peak * 10 ** (trim_db / 20)).any(dim=0).nonzero(as_tuple=False).squeeze(1)
    return int(idx[0]), int(idx[-1]) + 1


def load_mono(blob):
    data, sr = sf.read(io.BytesIO(zstd.ZstdDecompressor().decompress(blob)))
    if data.ndim > 1:
        data = data.mean(axis=1)
    return torch.from_numpy(data).float().unsqueeze(0), sr


def preprocess(wav, sr, gain):
    w = torchaudio.functional.resample(wav, sr, SAMPLE_RATE) if sr != SAMPLE_RATE else wav
    w = remove_dc_offset(w) * gain
    w = apply_pre_emphasis(apply_bandpass(w, SAMPLE_RATE))
    start, end = trim_bounds(w)
    w = w[:, start:end][:, : 10 * SAMPLE_RATE]
    if w.abs().max() > 0:
        w = w / w.abs().max()
    rms = torch.sqrt(torch.mean(w ** 2))
    if rms > 0:
        w = w * (0.1 / rms)
    return w.squeeze(0).numpy()


def open_db(data_dir):
    return sqlite3.connect(glob.glob(f"{data_dir}/*.db")[0]).cursor()


def main():
    cursor = open_db(sys.argv[1])
    cursor.execute("SELECT b.audio_file FROM audio_info a JOIN audio_raw_zstd b ON a.uuid = b.uuid "
                   "WHERE a.project_name = 'physionet22' LIMIT 5")
    extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

    def features(x):
        return extractor([x], sampling_rate=SAMPLE_RATE, return_tensors="np",
                         padding="max_length", max_length=1024)["input_values"][0]

    for (blob,) in cursor.fetchall():
        wav, sr = load_mono(blob)
        base = preprocess(wav, sr, 1.0)
        base_features = features(base)
        for gain in (0.7, 0.85, 1.3, 1.5):
            w = preprocess(wav, sr, gain)
            if w.shape != base.shape:
                print(f"gain {gain}: length changed ({w.shape[0]} vs {base.shape[0]} samples)")
                continue
            print(f"gain {gain}: max waveform difference {np.abs(w - base).max():.2e}, "
                  f"max feature difference {np.abs(features(w) - base_features).max():.2e} "
                  f"(feature SD {base_features.std():.2f})")


if __name__ == "__main__":
    main()
