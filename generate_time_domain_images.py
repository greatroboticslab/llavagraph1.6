#!/usr/bin/env python3
"""
Generate time-domain waveform images from raw txt/csv data files with augmentation.
Creates balanced train/val/test splits.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import random
import csv
from sklearn.model_selection import train_test_split

# Configuration
RAW_DATA_DIR_2026 = Path("/data/ilminur/llavagraph1.6-master/data/2.20.2026")
RAW_DATA_ARCHIVE = Path("/data/ilminur/llavagraph1.6-master/data/Archive")
OUTPUT_DIR = Path("/data/ilminur/llavagraph1.6-master/mambavision/data")

# Categories mapping
CATEGORIES = {
    "noise": ["noise", "noise"],
    "pulse": ["pulse", "pulse_csv"],
    "ramp": ["ramp", "ramp_csv"],
    "sine": ["sine", None],  # No archive folder for sine
    "Square": ["Square", None],
}

# Image settings
IMG_SIZE = 224
DPI = 72
N_AUGMENTATIONS = 7

def load_txt(filepath):
    """Parse .txt file (2.20.2026 format)."""
    fs = 1000.0
    d_values = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Sample Frequency'):
                try:
                    fs = float(line.split('=')[1].strip().split()[0])
                except:
                    fs = 1000.0
            elif line.startswith('D:'):
                try:
                    d_values.append(int(line.split()[0].split(':')[1]))
                except:
                    pass
    return fs, np.array(d_values, dtype=np.float64)

def load_csv(filepath):
    """Parse CSV file (Archive format)."""
    fs = 1000.0
    d_values = []
    
    # Try to detect format
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
        
    # Check if it has header
    if 'amplitude' in first_line.lower() or 'time' in first_line.lower():
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # Try common column names
                    if 'amplitude' in row:
                        d_values.append(float(row['amplitude']))
                    elif 'Amplitude' in row:
                        d_values.append(float(row['Amplitude']))
                    elif 'value' in row:
                        d_values.append(float(row['value']))
                except:
                    pass
    else:
        # Simple CSV with just values
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    val = float(line.strip())
                    d_values.append(val)
                except:
                    pass
    
    return fs, np.array(d_values, dtype=np.float64)

def augment_data(data):
    """Apply augmentations to the data."""
    augmentations = []
    augmentations.append(data.copy())  # Original
    
    # Add noise
    for noise_level in [0.01, 0.02]:
        noise = np.random.normal(0, np.std(data) * noise_level, len(data))
        augmentations.append(data + noise)
    
    # Scale amplitude
    for scale in [0.9, 1.1]:
        augmentations.append(data * scale)
    
    # Time shift
    shift = len(data) // 20
    augmentations.append(np.roll(data, shift))
    
    return augmentations[:N_AUGMENTATIONS]

def generate_time_domain_image(data, fs, output_path):
    """Generate a time-domain waveform image."""
    fig, ax = plt.subplots(figsize=(IMG_SIZE/DPI, IMG_SIZE/DPI), dpi=DPI)
    
    n_samples = len(data)
    time_ms = np.arange(n_samples) * (1000.0 / fs)
    
    ax.plot(time_ms, data, linewidth=0.5, color='blue')
    ax.set_xlim(0, time_ms[-1])
    ax.set_xlabel('Time (ms)', fontsize=8)
    ax.set_ylabel('Amplitude', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', pad_inches=0.02)
    plt.close()

def collect_all_data():
    """Collect all data from both sources."""
    all_files = []
    
    # From 2.20.2026 folder
    for raw_folder, class_name in CATEGORIES.items():
        if class_name[0] is None:
            continue
        raw_folder_path = RAW_DATA_DIR_2026 / class_name[0]
        if not raw_folder_path.exists():
            continue
        
        txt_files = list(raw_folder_path.glob("*.txt"))
        for txt_file in txt_files:
            all_files.append((txt_file, class_name[0].lower() if class_name[0] != "Square" else "square", 'txt'))
    
    # From Archive folder
    for class_name, archive_folder in CATEGORIES.items():
        if archive_folder[1] is None:
            continue
        
        if archive_folder[1] == "pulse_csv":
            archive_path = RAW_DATA_ARCHIVE / "pulse_csv"
            csv_files = list(archive_path.glob("*.csv"))
            for csv_file in csv_files:
                all_files.append((csv_file, "pulse", 'csv'))
        elif archive_folder[1] == "ramp_csv":
            archive_path = RAW_DATA_ARCHIVE / "ramp_csv"
            csv_files = list(archive_path.glob("*.csv"))
            for csv_file in csv_files:
                all_files.append((csv_file, "ramp", 'csv'))
    
    print(f"Total files collected: {len(all_files)}")
    return all_files

def main():
    print("=" * 70)
    print("  Time-Domain Image Generator (Combined Data Sources)")
    print("=" * 70)
    
    # Clean and create output directories
    import shutil
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    
    for split in ['train', 'val', 'test']:
        for class_name in ['noise', 'pulse', 'ramp', 'sine', 'square']:
            (OUTPUT_DIR / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Collect all data
    all_files = collect_all_data()
    
    if len(all_files) == 0:
        print("Error: No data files found!")
        return
    
    # Split into train/val/test (60/20/20)
    random.seed(42)
    np.random.seed(42)
    random.shuffle(all_files)
    
    train_files, temp_files = train_test_split(
        all_files, test_size=0.40, random_state=42
    )
    val_files, test_files = train_test_split(
        temp_files, test_size=0.50, random_state=42
    )
    
    print(f"\nSplits (original files):")
    print(f"  Train: {len(train_files)} files")
    print(f"  Val:   {len(val_files)} files")
    print(f"  Test:  {len(test_files)} files")
    
    # Process each split
    for split_name, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        print(f"\nProcessing {split_name} split...")
        
        success_count = 0
        error_count = 0
        use_augmentation = (split_name == 'train')
        
        for data_file, class_name, file_type in files:
            try:
                # Load data
                if file_type == 'txt':
                    fs, data = load_txt(data_file)
                else:
                    fs, data = load_csv(data_file)
                
                if len(data) < 100:
                    error_count += 1
                    continue
                
                if use_augmentation:
                    augmented_data_list = augment_data(data)
                    for aug_idx, aug_data in enumerate(augmented_data_list):
                        if aug_idx == 0:
                            output_filename = f"{class_name}_{data_file.stem}_orig.png"
                        else:
                            output_filename = f"{class_name}_{data_file.stem}_aug{aug_idx}.png"
                        output_path = OUTPUT_DIR / split_name / class_name / output_filename
                        generate_time_domain_image(aug_data, fs, output_path)
                        success_count += 1
                else:
                    output_filename = f"{class_name}_{data_file.stem}.png"
                    output_path = OUTPUT_DIR / split_name / class_name / output_filename
                    generate_time_domain_image(data, fs, output_path)
                    success_count += 1
                
            except Exception as e:
                print(f"  Error processing {data_file}: {e}")
                error_count += 1
        
        print(f"  {split_name}: {success_count} images, {error_count} errors")
    
    print("\n" + "=" * 70)
    print("  Generation Complete!")
    print("=" * 70)
    
    # Print final counts
    print("\nFinal dataset counts:")
    for split in ['train', 'val', 'test']:
        total = 0
        for class_name in ['noise', 'pulse', 'ramp', 'sine', 'square']:
            count = len(list((OUTPUT_DIR / split / class_name).glob("*.png")))
            total += count
            print(f"  {split}/{class_name}: {count} images")
        print(f"  {split} TOTAL: {total} images")

if __name__ == "__main__":
    main()
