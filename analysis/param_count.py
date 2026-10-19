"""Count the parameters of the fine-tuned model.

Usage: python analysis/param_count.py
"""

from transformers import ASTForAudioClassification


def count(module):
    return sum(p.numel() for p in module.parameters())


def main():
    model = ASTForAudioClassification.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        num_labels=2,
        label2id={"Normal": 0, "Abnormal": 1},
        id2label={0: "Normal", 1: "Abnormal"},
        ignore_mismatched_sizes=True,
    )
    backbone = model.audio_spectrogram_transformer
    layers = backbone.layers if hasattr(backbone, "layers") else backbone.encoder.layer
    print(f"Total: {count(model):,}")
    print(f"Embeddings: {count(backbone.embeddings):,}")
    print(f"Encoder layers: {len(layers)} x {count(layers[0]):,}")
    print(f"Final layer norm: {count(backbone.layernorm):,}")
    print(f"Classifier head: {count(model.classifier):,}")


if __name__ == "__main__":
    main()
