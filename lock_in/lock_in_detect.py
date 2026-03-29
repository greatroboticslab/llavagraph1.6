"""
Lock-in detection for piezoelectric actuator displacement signals.
Reproduces the example lock_in.png plot using real data from 2.20.2026.

For each reference frequency, plots lock-in magnitude over time for
sine, square, noise, pulse, and ramp signals.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import re
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', '2.20.2026')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'plots')

FREQUENCIES = [1, 100, 200, 300, 400]  # Hz
CATEGORIES = ['sine', 'Square', 'noise', 'pulse', 'ramp']
# Display names for plot legends
CAT_LABELS = {
    'sine': 'Sine',
    'Square': 'Square',
    'noise': 'Noise',
    'pulse': 'Pulse',
    'ramp': 'Ramp',
}
CAT_COLORS = {
    'sine': 'tab:blue',
    'Square': 'tab:orange',
    'noise': 'tab:green',
    'pulse': 'tab:red',
    'ramp': 'tab:purple',
}


def parse_signal(filepath):
    """Parse a .txt data file. Returns (fs, displacement_array)."""
    fs = None
    d_values = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Sample Frequency'):
                fs = float(line.split('=')[1].strip().split()[0])
            elif line.startswith('D:'):
                # D:135215013 N:811041
                d_val = int(line.split()[0].split(':')[1])
                d_values.append(d_val)
    return fs, np.array(d_values, dtype=np.float64)


def lock_in_detect(signal, fs, f_ref, tau_periods=5):
    """
    Digital lock-in detection.

    Parameters:
        signal: 1D array of displacement values
        fs: sampling frequency (Hz)
        f_ref: reference frequency (Hz)
        tau_periods: low-pass filter time constant in number of reference periods

    Returns:
        t: time array (seconds)
        R: lock-in magnitude over time
    """
    # Remove DC offset
    signal = signal - np.mean(signal)

    n = len(signal)
    t = np.arange(n) / fs

    # Reference signals
    ref_cos = np.cos(2 * np.pi * f_ref * t)
    ref_sin = np.sin(2 * np.pi * f_ref * t)

    # Multiply signal by references
    x_raw = signal * ref_cos
    y_raw = signal * ref_sin

    # Low-pass filter: exponential moving average
    # Time constant = tau_periods / f_ref
    tau = tau_periods / f_ref  # seconds
    alpha = 1.0 / (tau * fs)  # smoothing factor
    alpha = min(alpha, 1.0)   # clamp for very low frequencies

    x_filt = np.zeros(n)
    y_filt = np.zeros(n)
    x_filt[0] = x_raw[0]
    y_filt[0] = y_raw[0]
    for i in range(1, n):
        x_filt[i] = alpha * x_raw[i] + (1 - alpha) * x_filt[i - 1]
        y_filt[i] = alpha * y_raw[i] + (1 - alpha) * y_filt[i - 1]

    # Magnitude (multiply by 2 to get true amplitude, since mixing halves it)
    R = 2.0 * np.sqrt(x_filt**2 + y_filt**2)

    return t, R


def find_data_file(category, freq_hz, sample_idx=1):
    """Find a data file for the given category and frequency."""
    cat_dir = os.path.join(DATA_DIR, category)

    if category == 'noise':
        # Noise files don't have frequency: noise-1.txt, noise-2.txt, ...
        fname = f'noise-{sample_idx}.txt'
    else:
        # e.g., sine-100Hz-1.txt, Square-100Hz-1.txt
        prefix = category.lower() if category != 'Square' else 'Square'
        fname = f'{prefix}-{freq_hz}Hz-{sample_idx}.txt'

    path = os.path.join(cat_dir, fname)
    if os.path.exists(path):
        return path

    # Try alternate naming (e.g., sine-1-1.txt for 1Hz)
    if freq_hz == 1:
        prefix = category.lower() if category != 'Square' else 'Square'
        alt_fname = f'{prefix}-{sample_idx}-1.txt'
        alt_path = os.path.join(cat_dir, alt_fname)
        if os.path.exists(alt_path):
            return alt_path

    return None


def plot_lockin_for_frequency(freq_hz, sample_idx=1):
    """Generate lock-in detection plot for one reference frequency."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for cat in CATEGORIES:
        fpath = find_data_file(cat, freq_hz, sample_idx)
        if fpath is None:
            print(f"  Skipping {cat} at {freq_hz}Hz — file not found")
            continue

        fs, signal = parse_signal(fpath)
        t, R = lock_in_detect(signal, fs, freq_hz)

        # Normalize: convert raw displacement units to nm
        # The raw D values are ~135,215,000 range; variations are the signal
        # We work with the lock-in magnitude of the de-meaned signal directly
        ax.plot(t, R, color=CAT_COLORS[cat],
                label=f'{CAT_LABELS[cat]}', linewidth=2)

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Detected Amplitude (Lock-In Output)', fontsize=12)
    ax.set_title(f'Lock-In Detection at {freq_hz} Hz Reference', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outpath = os.path.join(OUTPUT_DIR, f'lock_in_{freq_hz}Hz.png')
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {outpath}")


def main():
    freqs = FREQUENCIES
    if len(sys.argv) > 1:
        freqs = [int(f) for f in sys.argv[1:]]

    print("Lock-in detection — generating plots")
    print(f"Data dir: {DATA_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print()

    for freq in freqs:
        print(f"Processing {freq} Hz...")
        plot_lockin_for_frequency(freq)

    print("\nDone!")


if __name__ == '__main__':
    main()
