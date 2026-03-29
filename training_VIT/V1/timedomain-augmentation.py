import re
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import glob
import argparse
from pathlib import Path

# --- Augmentation Parameter Settings ---
SEGMENT_OFFSETS = [100, 5000, 10000]
TIME_WINDOW_SIZE = 1024
TIME_WINDOW_SIZE_LARGE = 4096
MAX_INPUT_FREQ = 500


def load_time_series_range(csv_path, start_idx, n_points):
    """Load a segment of data from a CSV file."""
    try:
        with open(csv_path, 'r') as f:
            header = f.readline()
            lines = f.readlines()
        if start_idx + n_points > len(lines):
            return None, None
        time_ms = np.zeros(n_points)
        displacement = np.zeros(n_points)
        for i, line in enumerate(lines[start_idx:start_idx + n_points]):
            parts = line.strip().split(',')
            time_ms[i] = float(parts[0])
            displacement[i] = float(parts[1])
        return displacement, time_ms
    except Exception:
        return None, None


def generate_time_plot(data, time_ms, base_filename, segment_label, output_dir):
    """Generate time-domain displacement plot."""
    n = len(data)
    if n < 2: return None
    data_centered = data - np.mean(data)
    relative_time = time_ms - time_ms[0]

    plt.figure(figsize=(12, 6))
    plt.plot(relative_time, data_centered, linewidth=1.5, color='#0055aa')

    # Remove wave type prefix from title for cleaner display
    display_name = re.sub(r'^(sine|square|noise|pulse|ramp)[-_]', '', base_filename, flags=re.IGNORECASE)
    plt.title(f'Time Domain Signal: {display_name} ({segment_label})', fontsize=14, fontweight='bold')
    plt.xlabel('Relative Time (ms)', fontsize=12, fontweight='bold')
    plt.ylabel('Displacement (nm)', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)

    max_amp = np.max(np.abs(data_centered))
    if max_amp > 0:
        plt.ylim(-max_amp * 1.2, max_amp * 1.2)

    plt.tight_layout()
    out_name = f"{base_filename}_{segment_label}_TIME.png"
    plt.savefig(output_dir / out_name, dpi=300)
    plt.close()
    return out_name


def should_include_file(filename):
    """Filter files based on frequency and category."""
    fname = filename.lower()
    if any(x in fname for x in ['noise', 'pulse', 'ramp']): return True
    match = re.search(r'_(\d+)(?:hz|hx)', fname, re.IGNORECASE)
    if match:
        return int(match.group(1)) <= MAX_INPUT_FREQ
    return True


def process_directory(csv_root, output_dir):
    """
    Traverse the root directory and all subfolders to process CSV files.
    """
    csv_root = Path(csv_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gather all CSV files from the root and all subdirectories
    csv_files = []
    for root, dirs, files in os.walk(csv_root):
        for file in files:
            if file.endswith(".csv") and should_include_file(file):
                csv_files.append(os.path.join(root, file))

    csv_files = sorted(csv_files)
    print(f"Total CSV files found (including subfolders): {len(csv_files)}")

    count = 0
    for csv_path in csv_files:
        csv_path_obj = Path(csv_path)
        base_filename = csv_path_obj.stem

        # Use larger window for low-frequency signals to capture the full wave
        is_low_freq = 'trail' in base_filename.lower() or '_1Hz_' in base_filename
        win_size = TIME_WINDOW_SIZE_LARGE if is_low_freq else TIME_WINDOW_SIZE

        for i, offset in enumerate(SEGMENT_OFFSETS):
            segment_label = f"seg{i + 1}"
            data, time_ms = load_time_series_range(csv_path, offset, win_size)
            if data is not None:
                generate_time_plot(data, time_ms, base_filename, segment_label, output_dir)
                count += 1
        print(f"Finished processing: {base_filename}")

    print(f"\nSuccess! Total images generated: {count}")
    print(f"Check your results here: {output_dir}")


if __name__ == "__main__":
    # --- Paths to be modified ---
    input_path = "/Users/ilminurablikim/Desktop/csv"
    output_path = "/Users/ilminurablikim/Desktop/time_output"
    # ---------------------------

    process_directory(input_path, output_path)