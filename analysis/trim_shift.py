"""Count how often gain jitter moves the silence-trimming start point.

Usage: python analysis/trim_shift.py DATA_DIR

For 200 CirCor recordings, applies ten random gains in [0.7, 1.3] before the pipeline's filtering
and reports how far the -40 dB trimming start point moves.
"""

import sys

import numpy as np
import torchaudio

from check_gain_noop import (SAMPLE_RATE, apply_bandpass, apply_pre_emphasis, load_mono, open_db,
                             remove_dc_offset, trim_bounds)


def main():
    cursor = open_db(sys.argv[1])
    cursor.execute("SELECT b.audio_file FROM audio_info a JOIN audio_raw_zstd b ON a.uuid = b.uuid "
                   "WHERE a.project_name = 'physionet22' ORDER BY a.uuid LIMIT 200")
    rng = np.random.default_rng(0)
    shifts = []
    for (blob,) in cursor.fetchall():
        wav, sr = load_mono(blob)
        w = remove_dc_offset(torchaudio.functional.resample(wav, sr, SAMPLE_RATE))

        def trim_start(gain):
            return trim_bounds(apply_pre_emphasis(apply_bandpass(w * gain, SAMPLE_RATE)))[0]

        start = trim_start(1.0)
        shifts.append(max(abs(trim_start(g) - start) for g in rng.uniform(0.7, 1.3, 10)))

    shifts = np.array(shifts)
    shifted = shifts[shifts > 0]
    print(f"Recordings: {len(shifts)}")
    print(f"Trimming start unchanged for all gains: {int((shifts == 0).sum())}")
    print(f"Start moved: {len(shifted)} (more than 10 ms: {int((shifts > 160).sum())}, "
          f"more than 1 s: {int((shifts > SAMPLE_RATE).sum())})")
    if len(shifted):
        print(f"Median shift among moved recordings (samples): {np.median(shifted):.0f}")


if __name__ == "__main__":
    main()
