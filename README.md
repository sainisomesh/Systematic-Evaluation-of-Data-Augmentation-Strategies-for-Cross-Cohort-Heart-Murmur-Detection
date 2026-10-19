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

## Repository contents

- `unified_murmur_pipeline.py`: data loading, preprocessing, augmentation, training and evaluation
- `reproduce.py`, `run.sh`, `run.bat`: setup, data download and the full run
- `results/`: outputs of the original runs used in the paper
- `analysis/`: scripts that regenerate `results/prod_results.md` and the statistics reported in the
  paper from `results/` (run from this folder with `.venv`, for example
  `.venv/bin/python analysis/results_stats.py`)
- `tests/`: unit tests, run automatically by `reproduce.py`

## Citation

Data: Saini, S., Araujo, M. Dataset to Reproduce: Systematic Evaluation of Data Augmentation Strategies for
Cross-Cohort Heart Murmur Detection. Zenodo, 2026. https://doi.org/10.5281/zenodo.23460250

## License

MIT
