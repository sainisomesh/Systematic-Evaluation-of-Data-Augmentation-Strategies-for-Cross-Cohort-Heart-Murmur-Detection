# Design decisions

This note explains the main choices behind the code in this repository and why we made them. The
methods and results themselves are described in the paper; here we focus on the reasoning, including
a few things we would do differently next time.

## One pipeline for everything

All data loading, preprocessing, augmentation, training and evaluation live in a single script,
`unified_murmur_pipeline.py`. Earlier in the project we had separate scripts for different model
variants, and small differences between them (for example in preprocessing or in how scores were
pooled per patient) were hard to spot and changed the results. Keeping everything in one file means
every configuration goes through exactly the same code path, and the only thing that changes between
experiments is the entry in the `experiments` list.

Each experiment is one dictionary with a fixed name (`NN_prod_<description>`). We never renumber or
delete entries; finished experiments are commented out instead, because the names are used in the
results files and in the paper. The single-file layout also suited our compute: the experiments ran as
Kaggle notebooks, which execute one script, and the script can resume after a time limit by reading
the results it has already written.

## Data

- **One database for all cohorts.** All recordings are stored in one SQLite database. The audio is the
  original WAV file, compressed losslessly with zstd, so nothing is re-encoded or resampled before the
  pipeline sees it. Every query selects one cohort by name, so an external cohort cannot leak into
  training by accident. The pipeline finds the database by its `audio_info` table rather than by file
  name.
- **Training on CirCor only.** CirCor DigiScope (the PhysioNet 2022 Challenge training set) is the only
  training data. WC2023, WVU and OHH-2025 are used only for evaluation, never for training, early
  stopping or checkpoint selection. We wanted to measure how a model transfers to recordings from
  other clinics and devices, and any contact with those cohorts during training would have blurred that.
- **Patient-level split.** CirCor patients are split 80/20 into training and validation sets,
  stratified by label, with a fixed seed. Splitting by patient rather than by recording keeps
  recordings of the same person out of both sets.

## Preprocessing

Every recording, training or external, goes through the same steps: resampling to 16 kHz, removal of
the DC offset, a 20–1,200 Hz band-pass filter, pre-emphasis, trimming of silent lead-in and tail, a
10-second crop, and peak and RMS normalization. Since the cohorts were recorded with different
devices at 4, 8 and 16 kHz, we wanted the model to see them on a common footing. A 20–1,200 Hz band
keeps the heart sounds and most murmur energy while removing handling noise and hiss, and RMS
normalization evens out differences in recording level between devices.

One consequence we only understood later: because the normalization removes level differences, a
random gain change applied before it (our "gain jitter" augmentation) has almost no effect. Those runs
became near-repeats of the baseline, which turned out to be useful for estimating run-to-run
variability, but the lesson is that an augmentation must come after any step that would undo it.

## Model: why AST, and how much of it to train

We used the Audio Spectrogram Transformer checkpoint fine-tuned on AudioSet
(`MIT/ast-finetuned-audioset-10-10-0.4593`). A model pre-trained on a large and varied audio collection
seemed a better starting point for recordings from many devices than training from scratch on about
750 patients. We replaced its 527-class AudioSet output with a two-class layer (normal, murmur).

AudioSet already contains a "Heart murmur" class, so we first scored the unchanged model on that class
(the zero-shot baselines) to see how much murmur information the pre-trained model carries before any
training on heart sounds.

The main architectural decision was how much of the pre-trained network to update. We compared three
options before running any augmentation:

- **Frozen backbone, training only the head.** With a small training set, freezing is the cautious
  choice: few parameters change, overfitting is limited, and the general audio features learned on
  AudioSet are kept, which might in principle transfer well to other devices.
- **Full fine-tuning of all layers.** Heart sounds are very different from most of AudioSet: they are
  quiet, low-frequency, filtered to below 1.2 kHz, and the relevant detail lies in timing within the
  heart cycle. Full fine-tuning lets every layer, including the patch embedding, adapt to that.
- **Progressive unfreezing,** a compromise in which the backbone stays frozen for the first epochs and
  then only the top two layers are trained at a lower learning rate.

Full fine-tuning was clearly best, in-domain and on the external cohorts, by more than the spread we
saw between nearly identical runs, so every later experiment used it. The attribution analysis later
showed the same thing from a different angle: the frozen model did not focus on systole, while the
fine-tuned models did.

## Training

- **Shared settings.** All fully fine-tuned configurations share the same optimizer (AdamW, learning
  rate 1e-4, weight decay 1e-3), batch size (with one exception, noted in the paper), schedule (up to
  10 epochs, linear decay), label smoothing, data split and random seed. Fixing everything except the component under test keeps the comparisons
  fair; the trade-off is that each augmentation is tested at a common setting rather than at its own
  optimum.
- **Model selection on validation AUPRC.** About a fifth of CirCor patients have a murmur, so we chose
  the checkpoint with the highest patient-level AUPRC on the CirCor validation set, with early stopping
  (patience 2). The same validation set is also reported as the in-domain result, so that number is
  optimistic, which the paper states.
- **Loss variants as Trainer subclasses.** Focal loss, quality-weighted cross-entropy and mixup are
  implemented as small subclasses of the Hugging Face `Trainer`, chosen from the experiment's settings.
  Adding a new loss means adding a subclass rather than branching inside the training loop.
- **Reproducibility over speed where it mattered.** Random seeds are set once for Python, NumPy and
  PyTorch, and augmentations that we added later draw from a seeded generator. We switched off Hugging
  Face's `full_determinism` on GPUs: it forces synchronous CUDA calls, which made training so slow that
  runs hit the platform time limit. Reruns of the same configuration on the same hardware still gave
  identical results.

## Choosing augmentations without testing every combination

We started from the kinds of variation we expected between clinics and devices, and picked
augmentations that imitate them:

- different stethoscope frequency responses (HPSS-EQ),
- different recording levels (gain jitter),
- breathing and other body noise (respiratory noise from the ICBHI database at several levels, and
  respiratory amplitude modulation),
- recordings that start at different points in the heart cycle (cardiac-cycle crop),
- reversed signal polarity, which depends on the device (polarity inversion),
- partly missing or masked information (SpecAugment, and a version restricted to the 0–1,200 Hz band
  where heart sounds are, so that masks do not fall on empty frequencies),

plus three training-side components: mixup, a quality-based loss weight, and focal loss for the class
imbalance. That makes eleven components, and with several intensity levels for some of them, testing
every combination was out of the question: 2^11 = 2,048 subsets before intensities, at roughly one GPU
hour or more per run.

So we added components in stages and let each stage decide what went into the next:

1. **Baselines (5 runs):** the two zero-shot baselines and the three ways of training the backbone.
2. **One component at a time (22 runs):** each of the eleven components alone, plus extra intensity
   levels for gain jitter, both SpecAugment variants and respiratory noise, to see whether stronger
   is better.
3. **Pairs (48 runs):** combinations of the eight components that were worth combining. Polarity
   inversion was dropped because it gave results identical to the baseline (the model sees a magnitude
   spectrogram, so flipping the waveform changes nothing). Mixup and respiratory modulation were
   tested only alone because each lowered the OHH-2025 AUROC by more than 10 points on its own
   (to 64.46% and 70.15%, from 80.51%).
4. **Triples (7 runs):** chosen from the pair results.

This kept the study to 82 runs. Because the triples were picked from the pair results, the stage-4
runs were guided by earlier outcomes, which the paper states.

We trained each configuration once. Repeating every run with several seeds would have multiplied the
compute several times over, so we used the near-repeat runs (configurations that differ only by a
change the preprocessing removes) to estimate how much results vary between runs, and we read single
differences of a few points with that in mind.

## Evaluation

- **Per patient, as in the PhysioNet 2022 Challenge.** The model scores each recording, and the scores
  of a patient's recordings are combined (the mean of the two highest) into one patient score.
- **Primary outcome.** The patient-level AUROC on each external cohort and the mean over the three.
  AUROC is threshold-free, which matters because murmur prevalence ranges from about 4% to over 70%
  across the cohorts.
- **OHH-2025 as the cohort that matters most.** It is the largest external cohort and the closest to
  the setting in which One Heart Health intends to use such a model, so we report it separately.
- **Uncertainty.** 95% confidence intervals come from 1,000 bootstrap resamples of patients.

## Explaining what the models learned

To see which parts of the input the models use, we chose Integrated Gradients rather than a sampling
estimate of SHAP values. Its attributions add up exactly to the change in the model's score, which
gives a built-in check for every map, and the result does not depend on random sampling. The reference
input is silence. Red regions in the maps therefore raised the murmur score compared with silence, and
blue regions lowered it.

Single example maps are easy to over-read, so we averaged instead. We used CirCor validation patients
with a systolic murmur and the S1, systole, S2 and diastole annotations that come with CirCor, aligned
every heart cycle to a common time axis, and averaged per patient before averaging across patients.
Recordings whose annotations did not line up with the heart sounds were excluded by a fixed rule. We
then checked that the maps reflect the models: removing the regions with the highest attribution
lowered the score more than removing random regions, and re-initializing the classifier or the
transformer layers removed the pattern, as it should if the maps depend on what the network learned.

## Reproducibility

- `./run.sh` sets up a pinned environment, downloads the data from Zenodo with checksum checks, runs
  the unit tests and then all 82 experiments, resuming where it stopped.
- `./run.sh --figures`, run afterwards, reproduces the training-curve and attribution figures from the
  user's own trained models. Its extra packages are kept in a separate requirements file so that the
  training environment stays exactly as tested.
- The original outputs of our runs are in `results/`, so the numbers in the paper can be checked
  without retraining.
