# Data

Download the two files from Zenodo: https://doi.org/10.5281/zenodo.23460250

Put `hestia_wvu_wc2023_physionet22_ohh2025.db` in this folder and unzip `ICBHI_noise.zip` here:

```text
data/
    hestia_wvu_wc2023_physionet22_ohh2025.db
    ICBHI_noise/        174 .wav files
```

The database holds the recordings and murmur labels of the four cohorts. CirCor (`physionet22`) is
used for training and validation; WC2023, WVU and OHH-2025 are used only for evaluation.

| Cohort | Recordings | Patients | Patients with murmur |
|---|---|---|---|
| physionet22 (CirCor) | 3,163 | 942 | 179 |
| WC2023 | 570 | 108 | 74 |
| WVU | 383 | 76 | 54 |
| OHH-2025 | 5,382 | 1,151 | 41 |

`ICBHI_noise/` contains respiratory recordings from the ICBHI 2017 Respiratory Sound Database, used
as noise during training.
