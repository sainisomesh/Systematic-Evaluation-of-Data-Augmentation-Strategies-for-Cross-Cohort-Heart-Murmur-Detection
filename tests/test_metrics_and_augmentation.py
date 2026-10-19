"""Unit tests for unified_murmur_pipeline.py.

Run: python3 tests/test_metrics_and_augmentation.py
"""

import io
import os
import sys
import tempfile
import warnings

import numpy as np
import soundfile as sf
import zstandard as zstd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unified_murmur_pipeline as ump

warnings.filterwarnings("ignore", message="A single label was found", category=UserWarning)


def _logits_for_probs(probs):
    """Build (N, 2) logits whose softmax[:, 1] equals the requested probabilities."""
    probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.stack([np.zeros_like(probs), np.log(probs / (1 - probs))], axis=1)


def test_patient_metrics_order_independence():
    """Patient-level AUROC and AUPRC do not depend on the order of the recordings."""
    rel_paths = ["r1", "r2", "r3"]
    val_records = [("r1", 1, "A"), ("r2", 1, "A"), ("r3", 0, "B")]
    logits = _logits_for_probs([0.9, 0.8, 0.1])

    m = ump.compute_patient_metrics(
        logits=logits,
        labels=None,
        rel_paths=rel_paths,
        val_records=val_records,
        apply_temperature_scaling=False,
        bootstrap_samples=20,
    )
    assert m["auc_roc"] == 1.0, f"expected AUC 1.0, got {m['auc_roc']}"

    order = [2, 0, 1]
    m2 = ump.compute_patient_metrics(
        logits=logits[order],
        labels=None,
        rel_paths=[rel_paths[i] for i in order],
        val_records=val_records,
        apply_temperature_scaling=False,
        bootstrap_samples=20,
    )
    assert m2["auc_roc"] == m["auc_roc"], (m2["auc_roc"], m["auc_roc"])
    assert m2["auc_pr"] == m["auc_pr"], (m2["auc_pr"], m["auc_pr"])
    print("PASS: patient metrics are order-independent")


def _synthetic_audio_bytes():
    t = np.arange(16000 * 3) / 16000.0
    wav = (0.1 * np.sin(2 * np.pi * 100 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, wav, 16000, format="WAV")
    return zstd.ZstdCompressor().compress(buf.getvalue())


def test_no_augmentation_rng_during_eval():
    """Cardiac-cycle crop and respiratory modulation draw random numbers only for training items."""
    cfg = ump.FeatureConfig(
        use_cardiac_crop=True,
        use_resp_am=True,
        max_duration_seconds=2.0,
        is_train=False,
    )
    collator = ump.DataCollatorAST(
        feature_extractor=None,
        data_root=".",
        cfg=cfg,
        max_length=1024,
        seed=42,
    )
    audio = _synthetic_audio_bytes()

    eval_item = {"is_train": False, "rel_path": "x", "label": 1, "audio_bytes": audio}
    state_before = collator._rng.bit_generator.state
    collator._prepare_waveform(eval_item)
    state_after = collator._rng.bit_generator.state
    assert state_before == state_after, "evaluation advanced the augmentation random generator"

    train_item = {"is_train": True, "rel_path": "x", "label": 1, "audio_bytes": audio}
    collator._prepare_waveform(train_item)
    assert collator._rng.bit_generator.state != state_after, "training did not use the augmentation random generator"
    print("PASS: augmentation random generator is used only for training items")


def test_noise_loading_and_mixing():
    """Noise files are decoded exactly, mixed into training items, and a missing file is an error."""
    with tempfile.TemporaryDirectory() as tmp:
        pcm = (np.random.default_rng(0).integers(-20000, 20000, 16000 * 2)).astype(np.int16)
        noise_path = os.path.join(tmp, "noise.wav")
        sf.write(noise_path, pcm, 16000, subtype="PCM_16")
        waveform, sr = ump.load_audio(noise_path)
        assert sr == 16000 and waveform.shape == (1, pcm.size)
        assert np.array_equal(waveform.numpy()[0], pcm.astype(np.float32) / 32768.0)

        item = {"is_train": True, "rel_path": "x", "label": 1, "audio_bytes": _synthetic_audio_bytes()}
        base_cfg = dict(max_duration_seconds=2.0, is_train=True)
        clean = ump.DataCollatorAST(None, ".", ump.FeatureConfig(**base_cfg), 1024, 42)._prepare_waveform(item)
        noisy_cfg = ump.FeatureConfig(use_lung_noise=True, lambda_val=5.0, noise_files=[noise_path], **base_cfg)
        noisy = ump.DataCollatorAST(None, ".", noisy_cfg, 1024, 42)._prepare_waveform(item)
        assert noisy.shape == clean.shape and not np.allclose(noisy, clean), "noise was not added"

        missing_cfg = ump.FeatureConfig(use_lung_noise=True, lambda_val=1.0,
                                        noise_files=[os.path.join(tmp, "missing.wav")], **base_cfg)
        try:
            ump.DataCollatorAST(None, ".", missing_cfg, 1024, 42)._prepare_waveform(item)
        except ump.NoiseMixingError:
            pass
        else:
            raise AssertionError("a missing noise file should raise NoiseMixingError")
    print("PASS: noise files are decoded, mixed, and missing files raise an error")


def test_focal_loss_uses_quality_weights():
    """Focal loss applies per-sample quality weights and is unchanged without them."""
    import torch
    from transformers import TrainingArguments

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(8, 2)

        def forward(self, input_values, **kwargs):
            return type("Output", (), {"logits": self.linear(input_values)})()

    torch.manual_seed(0)
    model = TinyModel()
    with tempfile.TemporaryDirectory() as tmp:
        trainer = ump.FocalLossTrainer(model=model, args=TrainingArguments(output_dir=tmp, report_to=[], use_cpu=True),
                                       focal_alpha=0.6, focal_gamma=2.0)
        x, y = torch.randn(4, 8), torch.tensor([0, 1, 1, 0])
        w = torch.tensor([0.5, 1.5, 1.0, 1.0])
        plain = trainer.compute_loss(model, {"input_values": x, "labels": y})
        weighted = trainer.compute_loss(model, {"input_values": x, "labels": y, "sample_weight": w})
        per = ump.FocalLoss(alpha=0.6, gamma=2.0, reduction="none", label_smoothing=0.05)(model(x).logits, y)
        assert torch.allclose(plain, ump.FocalLoss(alpha=0.6, gamma=2.0, label_smoothing=0.05)(model(x).logits, y))
        assert torch.allclose(weighted, (per * w).mean())
        assert not torch.allclose(weighted, plain)
    print("PASS: focal loss applies quality weights")


def test_progressive_unfreeze_top_layers():
    """Progressive unfreezing makes only the top two AST layers trainable and adds them to the optimizer."""
    import torch
    from transformers import ASTConfig, ASTForAudioClassification

    config = ASTConfig(hidden_size=32, num_hidden_layers=4, num_attention_heads=2, intermediate_size=64,
                       max_length=64, num_mel_bins=32, num_labels=2)
    model = ASTForAudioClassification(config)
    for p in model.audio_spectrogram_transformer.parameters():
        p.requires_grad = False
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    args = type("Args", (), {"learning_rate": 1e-4})()
    state = type("State", (), {"epoch": 3.0})()

    ump.ProgressiveUnfreezeCallback(unfreeze_epoch=3).on_epoch_begin(args, state, None, model=model, optimizer=optimizer)

    backbone = model.audio_spectrogram_transformer
    blocks = backbone.layers if hasattr(backbone, "layers") else backbone.encoder.layer
    trainable = [all(p.requires_grad for p in b.parameters()) for b in blocks]
    assert trainable == [False, False, True, True], trainable
    assert not any(p.requires_grad for p in backbone.embeddings.parameters())
    assert len(optimizer.param_groups) == 2 and optimizer.param_groups[1]["lr"] == 1e-5
    print("PASS: progressive unfreezing trains only the top two layers")


if __name__ == "__main__":
    test_patient_metrics_order_independence()
    test_no_augmentation_rng_during_eval()
    test_noise_loading_and_mixing()
    test_focal_loss_uses_quality_weights()
    test_progressive_unfreeze_top_layers()
    print("All tests passed.")
