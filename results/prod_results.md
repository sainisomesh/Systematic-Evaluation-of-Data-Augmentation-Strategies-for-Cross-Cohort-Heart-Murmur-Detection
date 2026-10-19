# Results

Patient-level AUROC (%) with 95% bootstrap confidence intervals (1,000 patient resamples) for all
82 configurations. Models were trained on CirCor DigiScope (PhysioNet 2022) only; WC2023 (108
patients), WVU (76) and OHH-2025 (1,151) are external cohorts that were never used for training
or model selection. CirCor (val) is the held-out 20% patient split, which was also used to select
the checkpoint. Mean ext. is the mean AUROC over the three external cohorts.

Experiment IDs match `unified_murmur_pipeline.py`. Source: `kaggle_runs_consolidated.csv`.

## Zero-shot baselines and backbone adaptation

| ID | Configuration | CirCor (val) | WC2023 | WVU | OHH-2025 | Mean ext. |
|---|---|---|---|---|---|---|
| `00_dryrun_audioset_zeroshot` | AudioSet zero-shot, raw audio | 77.37 (67.6-85.8) | 89.11 (80.9-96.2) | 47.05 (31.9-62.7) | 53.12 (42.8-63.6) | 63.10 |
| `00b_dryrun_audioset_zeroshot_stdpreproc` | AudioSet zero-shot, preprocessed audio | 66.20 (53.8-77.4) | 84.62 (74.8-93.6) | 55.56 (40.8-70.0) | 46.76 (36.1-57.8) | 62.31 |
| `01_prod_processing_trainer_frozen` | Frozen backbone (head only) | 84.48 (75.6-91.8) | 70.75 (60.0-80.8) | 51.60 (37.1-66.4) | 61.08 (51.1-70.8) | 61.14 |
| `02_prod_processing_trainer_unfrozen` | Full fine-tuning (baseline) | 96.31 (92.9-98.9) | 77.74 (68.4-86.8) | 67.42 (54.3-79.9) | 80.51 (72.8-87.0) | 75.22 |
| `03_prod_processing_trainer_progressive` | Progressive unfreezing | 86.62 (77.6-93.9) | 76.03 (66.0-85.4) | 54.97 (39.7-70.2) | 56.08 (44.8-67.3) | 62.36 |

## Single components added to full fine-tuning

| ID | Configuration | CirCor (val) | WC2023 | WVU | OHH-2025 | Mean ext. |
|---|---|---|---|---|---|---|
| `04_prod_focal_loss` | focal loss | 96.79 (94.1-98.9) | 76.63 (67.3-84.9) | 57.07 (42.6-69.6) | 82.04 (74.3-88.3) | 71.91 |
| `05_prod_spec_augment` | SpecAugment (12, 48) | 96.56 (93.6-98.6) | 81.96 (72.8-90.1) | 67.00 (53.7-79.2) | 81.17 (73.2-88.7) | 76.71 |
| `06_prod_noise_025` | noise 32 dB | 95.67 (91.9-98.5) | 85.41 (77.4-92.5) | 63.80 (50.2-76.9) | 81.44 (72.5-89.0) | 76.89 |
| `07_prod_noise_100` | noise 20 dB | 96.06 (91.7-99.0) | 88.83 (81.4-94.9) | 66.33 (52.0-79.6) | 74.33 (65.0-83.2) | 76.50 |
| `08_prod_noise_500` | noise 6 dB | 94.27 (88.3-98.3) | 73.85 (63.2-83.2) | 61.53 (46.0-76.4) | 69.20 (59.0-78.9) | 68.19 |
| `09_prod_noise_mixed` | noise, mixed SNR | 97.26 (94.6-99.1) | 82.43 (73.1-90.0) | 62.71 (46.8-76.2) | 79.06 (70.9-86.3) | 74.73 |
| `10_prod_polarity_flip` | polarity inversion | 96.31 (92.9-98.9) | 77.74 (68.4-86.8) | 67.42 (54.3-79.9) | 80.51 (72.8-87.0) | 75.22 |
| `11_prod_gain_jitter` | gain jitter r=0.3 | 95.16 (91.1-98.3) | 90.26 (83.8-95.8) | 66.33 (52.4-78.9) | 83.98 (76.0-90.8) | 80.19 |
| `11b_prod_gain_jitter_mild` | gain jitter r=0.15 | 96.10 (92.3-98.8) | 76.79 (67.5-85.5) | 55.39 (41.2-68.8) | 80.57 (72.3-87.5) | 70.91 |
| `11c_prod_gain_jitter_extreme` | gain jitter r=0.5 | 96.97 (94.3-98.9) | 84.26 (76.1-91.5) | 64.06 (49.6-77.4) | 83.15 (75.6-89.4) | 77.16 |
| `12_prod_mixup` | mixup | 97.42 (95.2-99.0) | 77.90 (68.6-86.1) | 63.55 (49.0-76.3) | 64.46 (55.2-73.4) | 68.64 |
| `13_prod_hpss_eq` | HPSS-EQ | 96.00 (92.4-98.5) | 83.94 (75.2-91.7) | 71.80 (58.5-83.5) | 87.49 (80.3-93.1) | 81.08 |
| `14_prod_cardiac_crop` | cardiac-cycle crop | 97.88 (95.7-99.2) | 77.34 (68.4-85.8) | 69.87 (56.9-81.5) | 86.17 (79.8-91.5) | 77.79 |
| `15_prod_resp_am` | respiratory modulation | 94.58 (89.9-97.9) | 85.33 (77.9-92.0) | 69.44 (54.9-82.3) | 70.15 (60.7-79.1) | 74.98 |
| `16_prod_quality_weight` | quality weighting | 97.11 (94.7-98.9) | 85.61 (78.0-92.4) | 71.04 (57.3-83.0) | 76.24 (67.1-84.9) | 77.63 |
| `17_prod_spec_moderate` | SpecAugment (24, 80) | 97.07 (94.5-98.9) | 76.27 (66.4-85.2) | 63.30 (49.2-76.7) | 78.97 (70.3-86.5) | 72.85 |
| `18_prod_spec_aggressive` | SpecAugment (40, 100) | 95.84 (92.6-98.6) | 84.50 (77.1-91.6) | 55.89 (41.4-69.3) | 72.86 (64.1-80.6) | 71.08 |
| `19_prod_spec_extreme` | SpecAugment (48, 150) | 96.43 (92.7-98.9) | 77.62 (68.0-86.4) | 58.84 (43.8-73.2) | 67.07 (58.3-75.4) | 67.84 |
| `20_prod_cspec_light` | cardiac SpecAugment (8, 48) | 97.32 (94.9-99.0) | 79.25 (70.3-88.3) | 65.07 (52.5-77.7) | 85.09 (78.2-90.9) | 76.47 |
| `21_prod_cspec_moderate` | cardiac SpecAugment (15, 80) | 97.49 (95.3-99.1) | 81.60 (72.6-89.4) | 62.96 (49.0-75.8) | 83.72 (76.8-89.7) | 76.09 |
| `22_prod_cspec_aggressive` | cardiac SpecAugment (24, 100) | 97.49 (95.3-99.1) | 82.67 (73.7-90.8) | 65.57 (51.5-79.0) | 76.60 (67.0-85.0) | 74.95 |
| `23_prod_cspec_ultra_light` | cardiac SpecAugment (5, 48) | 96.78 (93.9-98.9) | 83.55 (75.0-91.2) | 68.52 (54.7-81.0) | 85.33 (78.3-91.2) | 79.13 |

## Two-component combinations

| ID | Configuration | CirCor (val) | WC2023 | WVU | OHH-2025 | Mean ext. |
|---|---|---|---|---|---|---|
| `24_prod_spec_noisemixed` | SpecAugment (12, 48) + noise, mixed SNR | 96.39 (93.4-98.7) | 82.51 (73.9-90.0) | 59.76 (45.4-73.3) | 83.56 (75.9-90.0) | 75.28 |
| `25_prod_spec_noise025` | SpecAugment (12, 48) + noise 32 dB | 96.56 (93.7-98.7) | 87.68 (80.6-94.1) | 63.55 (50.3-75.6) | 77.69 (69.6-85.5) | 76.31 |
| `26_prod_spec_noise100` | SpecAugment (12, 48) + noise 20 dB | 96.87 (94.1-98.9) | 86.76 (78.8-93.5) | 61.28 (47.4-74.5) | 75.96 (67.0-84.1) | 74.67 |
| `27_prod_spec_noise500` | SpecAugment (12, 48) + noise 6 dB | 93.98 (89.2-97.6) | 84.94 (76.5-92.1) | 58.50 (44.1-72.1) | 68.62 (58.6-78.0) | 70.68 |
| `28_prod_spec_focal` | SpecAugment (12, 48) + focal loss | 96.23 (92.5-98.8) | 84.38 (76.1-91.9) | 60.94 (46.1-73.9) | 81.31 (73.5-88.3) | 75.55 |
| `29_prod_noisemixed_focal` | noise, mixed SNR + focal loss | 96.89 (94.2-98.8) | 84.30 (76.4-91.2) | 75.08 (62.1-86.5) | 81.08 (73.7-88.1) | 80.15 |
| `30_prod_noise025_focal` | noise 32 dB + focal loss | 96.91 (94.2-98.9) | 82.75 (73.3-90.6) | 62.46 (47.4-75.2) | 73.80 (65.0-82.0) | 73.00 |
| `31_prod_noise100_focal` | noise 20 dB + focal loss | 96.35 (93.3-98.5) | 85.10 (77.2-92.4) | 63.47 (49.5-75.8) | 83.64 (76.1-90.1) | 77.40 |
| `32_prod_noise500_focal` | noise 6 dB + focal loss | 96.17 (92.7-98.6) | 82.63 (73.5-90.7) | 60.44 (46.9-73.3) | 72.08 (62.0-81.7) | 71.72 |
| `33_prod_hpss_noisemixed` | HPSS-EQ + noise, mixed SNR | 95.61 (90.9-98.7) | 83.62 (75.1-91.3) | 64.81 (50.6-77.8) | 77.17 (68.2-85.5) | 75.20 |
| `34_prod_hpss_noise025` | HPSS-EQ + noise 32 dB | 97.34 (94.8-99.0) | 85.02 (76.8-92.2) | 70.96 (58.3-82.9) | 80.71 (73.1-87.9) | 78.90 |
| `35_prod_hpss_noise100` | HPSS-EQ + noise 20 dB | 97.44 (95.0-99.2) | 83.35 (74.8-90.7) | 67.17 (53.1-79.4) | 78.71 (70.1-86.6) | 76.41 |
| `36_prod_hpss_noise500` | HPSS-EQ + noise 6 dB | 95.09 (90.9-98.1) | 77.98 (68.4-87.3) | 60.44 (46.1-73.4) | 83.09 (74.7-90.2) | 73.84 |
| `37_prod_hpss_spec` | SpecAugment (12, 48) + HPSS-EQ | 96.76 (93.6-99.1) | 78.50 (68.8-86.7) | 63.80 (49.8-76.7) | 80.78 (72.0-88.4) | 74.36 |
| `38_prod_hpss_cspec_ultra_light` | cardiac SpecAugment (5, 48) + HPSS-EQ | 95.75 (92.2-98.2) | 83.39 (74.7-90.9) | 63.64 (49.5-76.6) | 81.97 (73.6-88.9) | 76.33 |
| `39_prod_hpss_focal` | HPSS-EQ + focal loss | 97.09 (94.5-99.0) | 86.53 (78.4-93.5) | 68.60 (54.7-81.9) | 89.54 (83.4-94.6) | 81.56 |
| `40_prod_cspec_ultra_light_noisemixed` | cardiac SpecAugment (5, 48) + noise, mixed SNR | 96.70 (93.8-98.8) | 80.29 (70.0-88.7) | 62.88 (48.9-75.8) | 73.27 (64.1-81.9) | 72.15 |
| `41_prod_cspec_ultra_light_noise025` | cardiac SpecAugment (5, 48) + noise 32 dB | 95.67 (91.7-98.3) | 84.86 (76.1-92.0) | 66.58 (52.5-78.9) | 83.25 (75.8-90.0) | 78.23 |
| `42_prod_cspec_ultra_light_noise100` | cardiac SpecAugment (5, 48) + noise 20 dB | 96.91 (94.3-98.9) | 81.48 (73.2-89.6) | 68.35 (55.0-81.3) | 81.79 (74.1-88.5) | 77.20 |
| `43_prod_cspec_ultra_light_noise500` | cardiac SpecAugment (5, 48) + noise 6 dB | 95.45 (91.0-98.5) | 81.40 (71.7-89.9) | 58.16 (43.8-71.5) | 63.26 (52.9-73.5) | 67.61 |
| `44_prod_cspec_ultra_light_focal` | cardiac SpecAugment (5, 48) + focal loss | 95.59 (92.1-98.1) | 84.58 (76.9-91.7) | 70.96 (58.4-82.5) | 74.24 (64.0-84.0) | 76.59 |
| `45_prod_gain_jitter_noisemixed` | gain jitter r=0.3 + noise, mixed SNR | 96.13 (92.9-98.6) | 82.35 (73.2-90.4) | 65.32 (50.8-78.2) | 81.86 (73.2-89.1) | 76.51 |
| `46_prod_gain_jitter_noise025` | gain jitter r=0.3 + noise 32 dB | 96.62 (93.1-99.0) | 85.41 (76.8-92.7) | 63.47 (49.2-76.0) | 81.84 (73.4-89.0) | 76.91 |
| `47_prod_gain_jitter_noise100` | gain jitter r=0.3 + noise 20 dB | 96.60 (93.6-98.7) | 89.27 (82.2-95.3) | 67.42 (53.2-80.5) | 82.11 (73.7-89.3) | 79.60 |
| `48_prod_gain_jitter_noise500` | gain jitter r=0.3 + noise 6 dB | 95.63 (91.3-98.6) | 79.05 (68.8-87.6) | 60.44 (45.3-74.4) | 72.25 (61.9-81.9) | 70.58 |
| `49_prod_gain_jitter_focal` | gain jitter r=0.3 + focal loss | 97.32 (94.8-99.1) | 79.81 (70.0-87.8) | 62.21 (47.6-75.1) | 73.84 (64.5-82.5) | 71.95 |
| `50_prod_gain_jitter_hpss` | gain jitter r=0.3 + HPSS-EQ | 96.85 (93.9-98.9) | 81.96 (73.4-89.8) | 58.75 (45.4-72.1) | 75.52 (66.0-84.6) | 72.08 |
| `51_prod_gain_jitter_cardiac_crop` | gain jitter r=0.3 + cardiac-cycle crop | 95.45 (91.5-98.3) | 80.21 (71.2-88.8) | 57.49 (42.9-71.2) | 74.61 (65.3-83.3) | 70.77 |
| `52_prod_gain_jitter_spec` | SpecAugment (12, 48) + gain jitter r=0.3 | 95.45 (91.7-98.3) | 81.52 (72.8-89.8) | 69.36 (55.7-81.7) | 80.96 (73.6-87.9) | 77.28 |
| `53_prod_gain_jitter_cspec_ultra_light` | cardiac SpecAugment (5, 48) + gain jitter r=0.3 | 95.78 (92.4-98.4) | 86.33 (78.1-92.8) | 69.44 (55.0-81.7) | 77.49 (68.2-85.9) | 77.75 |
| `54_prod_gain_jitter_quality_weight` | gain jitter r=0.3 + quality weighting | 97.13 (94.4-99.0) | 83.55 (75.1-91.2) | 65.07 (50.1-76.7) | 81.90 (74.3-88.7) | 76.84 |
| `55_prod_cardiac_crop_noisemixed` | cardiac-cycle crop + noise, mixed SNR | 97.53 (95.2-99.1) | 84.34 (75.4-91.7) | 66.25 (52.8-78.4) | 78.02 (70.5-84.8) | 76.20 |
| `56_prod_cardiac_crop_noise025` | cardiac-cycle crop + noise 32 dB | 97.09 (94.5-99.0) | 81.04 (72.4-89.3) | 66.33 (52.2-79.5) | 78.92 (70.5-86.2) | 75.43 |
| `57_prod_cardiac_crop_noise100` | cardiac-cycle crop + noise 20 dB | 97.07 (94.4-98.8) | 82.87 (74.5-90.3) | 63.38 (49.6-76.2) | 76.89 (68.4-84.9) | 74.38 |
| `58_prod_cardiac_crop_noise500` | cardiac-cycle crop + noise 6 dB | 96.17 (91.9-98.9) | 79.05 (70.0-87.0) | 67.76 (53.3-80.5) | 77.07 (68.4-85.0) | 74.63 |
| `59_prod_cardiac_crop_focal` | cardiac-cycle crop + focal loss | 95.55 (91.9-98.3) | 75.00 (65.7-84.2) | 66.16 (52.9-78.4) | 80.88 (72.6-88.3) | 74.01 |
| `60_prod_cardiac_crop_hpss` | HPSS-EQ + cardiac-cycle crop | 95.28 (91.2-98.3) | 79.61 (69.7-88.0) | 59.43 (45.3-72.7) | 74.06 (65.2-82.7) | 71.03 |
| `61_prod_cardiac_crop_spec` | SpecAugment (12, 48) + cardiac-cycle crop | 95.59 (91.1-98.4) | 81.60 (72.6-89.7) | 57.15 (43.4-70.0) | 73.90 (64.8-82.5) | 70.88 |
| `62_prod_cardiac_crop_cspec_ultra_light` | cardiac SpecAugment (5, 48) + cardiac-cycle crop | 95.67 (92.1-98.3) | 82.19 (73.1-90.0) | 57.32 (42.6-70.5) | 79.44 (70.9-86.7) | 72.99 |
| `63_prod_cardiac_crop_quality_weight` | cardiac-cycle crop + quality weighting | 96.31 (92.1-98.9) | 76.11 (66.8-85.0) | 53.54 (39.2-66.3) | 62.64 (53.4-71.8) | 64.10 |
| `64_prod_quality_weight_noisemixed` | quality weighting + noise, mixed SNR | 95.24 (91.0-98.1) | 85.57 (78.3-92.1) | 65.66 (51.2-78.1) | 70.34 (61.5-78.9) | 73.86 |
| `65_prod_quality_weight_noise025` | quality weighting + noise 32 dB | 95.65 (91.3-98.7) | 80.80 (72.4-89.1) | 65.07 (52.0-77.9) | 79.22 (71.6-86.4) | 75.03 |
| `66_prod_quality_weight_noise100` | quality weighting + noise 20 dB | 96.50 (93.2-98.8) | 74.05 (64.3-83.2) | 64.56 (50.5-77.3) | 85.78 (79.2-91.5) | 74.80 |
| `67_prod_quality_weight_noise500` | quality weighting + noise 6 dB | 94.19 (89.6-97.6) | 82.75 (73.9-90.7) | 64.31 (50.4-77.2) | 81.72 (73.2-89.3) | 76.26 |
| `68_prod_quality_weight_focal` | quality weighting + focal loss | 96.87 (94.2-98.8) | 85.81 (78.0-92.6) | 65.24 (51.4-77.7) | 80.25 (72.5-87.4) | 77.10 |
| `69_prod_quality_weight_hpss` | HPSS-EQ + quality weighting | 95.26 (91.0-98.2) | 84.58 (76.2-91.9) | 57.49 (42.9-70.9) | 83.22 (76.3-89.9) | 75.10 |
| `70_prod_quality_weight_spec` | SpecAugment (12, 48) + quality weighting | 96.35 (93.4-98.4) | 79.13 (69.9-87.7) | 60.52 (46.6-73.6) | 78.79 (70.7-85.9) | 72.82 |
| `71_prod_quality_weight_cspec_ultra_light` | cardiac SpecAugment (5, 48) + quality weighting | 96.66 (94.0-98.8) | 83.78 (75.9-91.2) | 68.52 (56.4-80.8) | 84.95 (77.0-91.3) | 79.08 |

## Three-component combinations

| ID | Configuration | CirCor (val) | WC2023 | WVU | OHH-2025 | Mean ext. |
|---|---|---|---|---|---|---|
| `72_prod_hpss_noisemixed_focal` | HPSS-EQ + noise, mixed SNR + focal loss | 97.05 (94.5-99.0) | 88.59 (79.8-95.1) | 57.15 (41.6-71.2) | 60.69 (50.6-70.8) | 68.81 |
| `73_prod_hpss_noise025_focal` | HPSS-EQ + noise 32 dB + focal loss | 97.16 (94.7-99.0) | 86.69 (79.6-93.0) | 63.38 (49.0-77.0) | 78.66 (70.3-85.8) | 76.24 |
| `74_prod_gain_jitter_cspec_ultra_light_noise100` | cardiac SpecAugment (5, 48) + gain jitter r=0.3 + noise 20 dB | 95.82 (91.4-99.0) | 85.21 (76.8-92.7) | 66.67 (53.0-79.6) | 73.72 (63.8-83.1) | 75.20 |
| `75_prod_gain_jitter_cspec_ultra_light_noise025` | cardiac SpecAugment (5, 48) + gain jitter r=0.3 + noise 32 dB | 97.28 (94.6-99.0) | 88.47 (81.4-94.6) | 64.14 (49.8-76.6) | 87.72 (81.2-93.4) | 80.11 |
| `76_prod_hpss_noise100_focal` | HPSS-EQ + noise 20 dB + focal loss | 96.58 (93.3-99.0) | 76.83 (67.3-85.9) | 72.56 (58.4-84.7) | 83.05 (75.5-89.3) | 77.48 |
| `77_prod_cardiac_crop_noisemixed_focal` | cardiac-cycle crop + noise, mixed SNR + focal loss | 97.44 (95.1-99.0) | 84.38 (76.0-91.7) | 56.23 (42.6-69.3) | 64.24 (54.0-74.5) | 68.28 |
| `78_prod_gain_jitter_hpss_focal` | gain jitter r=0.3 + HPSS-EQ + focal loss | 96.48 (93.6-98.4) | 76.51 (66.7-85.2) | 61.45 (46.9-75.0) | 83.19 (75.8-90.0) | 73.72 |
