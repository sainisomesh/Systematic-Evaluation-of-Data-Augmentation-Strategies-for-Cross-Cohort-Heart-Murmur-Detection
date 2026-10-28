"""Checks for the helpers in figures/attribution.py (plain assertions, run directly).

Run from the repository root:  .venv/bin/python tests/test_attribution_helpers.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures"))
import attribution as A  # noqa: E402


def test_valid_frame_count_matches_feature_extractor():
    """Frames 0..n_valid-1 carry the recording; every later frame is identical padding."""
    _, fe = A.make_collator()
    rng = np.random.default_rng(0)
    for seconds in (2.0, 5.37, 10.0):
        n = int(seconds * A.SR)
        x = fe([rng.standard_normal(n).astype(np.float32) * 0.1], sampling_rate=A.SR, return_tensors="pt",
               padding="max_length", truncation=True, max_length=A.MAX_LEN)["input_values"][0].numpy()
        n_valid = 1 + (n - A.WIN) // A.HOP
        assert x.shape == (A.MAX_LEN, 128)
        pad = x[n_valid:]
        assert np.allclose(pad, pad[0]), seconds  # constant padding rows
        assert not np.allclose(x[n_valid - 1], pad[0]), seconds  # last valid frame is real signal


def test_mel_centers():
    hz = A.mel_centers_hz()
    assert hz.shape == (128,) and np.all(np.diff(hz) > 0)
    assert 20 < hz[0] < 60 and 7000 < hz[-1] < 8000
    n_cardiac = int((hz <= A.CARDIAC_HZ).sum())
    assert 45 <= n_cardiac <= 55, n_cardiac  # the pipeline treats bins 0-51 as the cardiac range


def test_silence_baseline_keeps_padding():
    import torch

    _, fe = A.make_collator()
    x = torch.randn(1, A.MAX_LEN, 128)
    base = A.silence_baseline(fe, x, n_valid=300)
    assert torch.equal(base[:, 300:], x[:, 300:])
    sil = fe([np.zeros(A.SR * 10, dtype=np.float32)], sampling_rate=A.SR, return_tensors="pt", padding="max_length",
             truncation=True, max_length=A.MAX_LEN)["input_values"]
    assert torch.allclose(base[:, :300], sil[:, :300])  # the extractor's own spectrogram of silence


def test_warp_area_average():
    m = np.arange(150.0)[:, None] * np.ones((1, 3))
    for n in (36, 300):  # compressed and stretched phase
        w = A.warp(m, [(0, 150)], [n])
        assert np.allclose(w.mean(0), m.mean(0)) and np.all(np.diff(w[:, 0]) >= 0)
    c = A.warp(np.full((40, 2), 7.0), [(0, 10), (10, 25), (25, 31), (31, 40)], [5, 9, 4, 12])
    assert c.shape == (30, 2) and np.allclose(c, 7.0)
    z = np.zeros((20, 1))
    z[5] = 1.0
    wz = A.warp(z, [(0, 20)], [10])
    assert np.isclose(wz.sum() * 2, 1.0) and wz.argmax() == 2


def test_reinit_changes_loaded_weights():
    """attribution.reinit_top must re-initialize weights flagged as loaded (transformers 5.x skips
    parameters with _is_hf_initialized) and leave lower layers untouched."""
    import torch
    from transformers import ASTConfig, ASTForAudioClassification

    cfg = ASTConfig(num_hidden_layers=3, hidden_size=32, num_attention_heads=2, intermediate_size=64, num_labels=2)
    m = ASTForAudioClassification(cfg)
    for p in m.parameters():
        p.data.fill_(0.123)
        p._is_hf_initialized = True  # as left by from_pretrained
    layers = m.audio_spectrogram_transformer.layers
    A.reinit_top(m, 2)
    changed = lambda mod: not all(torch.allclose(p, torch.full_like(p, 0.123)) for p in mod.parameters())
    assert changed(m.classifier) and changed(layers[1]) and changed(layers[2])
    assert not changed(layers[0])  # below the re-initialized block

def test_predict_runs_without_gradients():
    """predict() must run under torch.no_grad() (it returns NumPy logits for whole cohorts)."""
    import torch
    from transformers import ASTConfig, ASTForAudioClassification

    cfg = ASTConfig(num_hidden_layers=1, hidden_size=32, num_attention_heads=2, intermediate_size=64, num_labels=2)
    model = ASTForAudioClassification(cfg).eval()
    collator = lambda items: {"input_values": torch.randn(len(items), A.MAX_LEN, 128)}
    records = [(f"u{i}", 0, "p", b"") for i in range(5)]
    out = A.predict(model, collator, records, batch=2)
    assert out.shape == (5, 2) and out.dtype.kind == "f"


if __name__ == "__main__":
    test_valid_frame_count_matches_feature_extractor()
    test_mel_centers()
    test_silence_baseline_keeps_padding()
    test_warp_area_average()
    test_reinit_changes_loaded_weights()
    test_predict_runs_without_gradients()
    print("attribution helper tests passed")
