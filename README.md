# Systematic Evaluation of Data Augmentation Strategies for Cross-Cohort Heart Murmur Detection

Code for the paper of the same title. The study design, methods and findings are described in the
paper. An Audio Spectrogram Transformer is fine-tuned on CirCor DigiScope (PhysioNet 2022) and tested
on three external cohorts (WC2023, WVU and OHH-2025). All 82 configurations from the paper are defined
in `unified_murmur_pipeline.py`.

## Reproducing the experiments

macOS or Linux:

```bash
./run.sh
```

Windows:

```bat
run.bat
```

This creates a Python environment in `.venv`, downloads the data from Zenodo
(https://doi.org/10.5281/zenodo.23460250) into `data/`, checks it, runs all 82 experiments, and writes
the outputs and a comparison with the paper to `results_reproduced/`. The trained model of each
experiment is saved in `results_reproduced/<experiment>/best_model/` (about 350 MB each). It needs
Python 3.12 to 3.14, about 60 GB of disk space, 16 GB of RAM, an internet connection, and an NVIDIA
GPU with about 16 GB of memory, such as the T4 used for the paper.

On an NVIDIA T4 an experiment takes about 1 hour (up to 5 hours), about 100 hours for all 82. If the
run stops, start the command again and it continues from the last finished experiment.

`./run.sh --check` sets everything up and verifies the data without training.

## Reproducing the training-curve and attribution figures

After the full run has finished, run:

```bash
./run.sh --figures        # Windows: run.bat --figures
```

It needs the outputs of the full run, so it stops with a message if `./run.sh` has not been run
first. It installs `figures/requirements.txt` (Captum and Matplotlib), then:

1. draws the training curves (training loss, and validation AUROC and AUPRC per epoch) of six
   configurations from the MLflow log of the full run (`results_reproduced/mlflow.db`), checks that
   the selected epoch of each gives the validation AUROC the run reported, and writes the figure to
   `results_reproduced/figures/training_curves/`;
2. makes the attribution figure from the trained models in `results_reproduced/<experiment>/best_model/`
   of five configurations (frozen backbone, full fine-tuning, HPSS-EQ with focal loss, cardiac
   SpecAugment (5, 48), and mixed noise with focal loss): it downloads the CirCor labels and heart-cycle
   segmentation files it needs from PhysioNet (checked against PhysioNet's SHA-256 checksums) and runs
   `figures/attribution.py`, which computes Integrated Gradients for each validation patient with a
   systolic murmur, aligns them to S1, systole, S2 and diastole, and averages them. The two halves of the
   figure (`Fig_attribution_left_ABC.pdf`, `Fig_attribution_right_DEF.pdf`), the per-patient results and
   the numbers quoted in the paper (`paper_numbers.json`) are written to
   `results_reproduced/figures/attribution/`. This step runs on an NVIDIA GPU if one is available and
   otherwise on the CPU (several hours).

Text is set with LaTeX if it is installed and with Matplotlib's Computer Modern fonts otherwise.

## Repository contents

- `unified_murmur_pipeline.py`: data loading, preprocessing, augmentation, training and evaluation
- `reproduce.py`, `run.sh`, `run.bat`: setup, data download and the full run
- `results/`: outputs of the original runs used in the paper
- `analysis/`: scripts that regenerate `results/prod_results.md` and the statistics reported in the
  paper from `results/` (run from this folder with `.venv`, for example
  `.venv/bin/python analysis/results_stats.py`)
- `figures/`: `training_curves.py`, `attribution.py` and their requirements, used by `./run.sh --figures`
- `tests/`: unit tests, run automatically by `reproduce.py`

## Citation

Data: Saini, S., Araujo, M. Dataset to Reproduce: Systematic Evaluation of Data Augmentation Strategies for
Cross-Cohort Heart Murmur Detection. Zenodo, 2026. https://doi.org/10.5281/zenodo.23460250

## License

MIT
