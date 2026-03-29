import re
import random
import shutil
import argparse
from pathlib import Path
from collections import defaultdict


TRAIN_RATIO = 0.6
RANDOM_SEED = 42
EXCLUDE_500HZ = True


def extract_freq_group(filename):
    """Extract frequency for stratified sampling."""
    fname = filename.lower()
    # For categories without frequency labels, return the category name directly
    if any(x in fname for x in ["noise", "pulse", "ramp"]):
        for x in ["noise", "pulse", "ramp"]:
            if x in fname: return x

    # Match frequency in filename, e.g., 100Hz
    match = re.search(r'_(\d+)(?:hz|hx)', fname, re.IGNORECASE)
    if match:
        return f"{match.group(1)}Hz"
    return "unknown"


def extract_csv_id(filename):
    """
    Extract the unique identifier of the original CSV.
    Ensures all seg maps generated from the same CSV are assigned to the same set (Train or Test).
    """
    stem = Path(filename).stem
    # Remove suffixes, such as _seg1_TIME or _seg2_TIME
    return re.sub(r'_seg\d+_(TIME|FFT)$', '', stem)


def should_exclude(filename, category):
    """Determine if 500Hz data should be excluded."""
    if not EXCLUDE_500HZ:
        return False
    if category in ["noise", "pulse", "ramp"]:
        return False
    match = re.search(r'_500(?:hz|hx)', filename, re.IGNORECASE)
    return match is not None


def split_category(image_dir, category):
    """Split a specific category folder into training and testing sets."""
    cat_path = image_dir / category
    if not cat_path.exists():
        print(f"  [Skip] Folder does not exist: {cat_path}")
        return [], []

    # Get all PNGs in the folder
    all_files = sorted(cat_path.glob("*.png"))

    # Frequency filtering (500Hz)
    if category in ("sine", "square"):
        before = len(all_files)
        all_files = [f for f in all_files if not should_exclude(f.name, category)]
        excluded = before - len(all_files)
        if excluded:
            print(f"  [Filter] Excluded {excluded} images of 500Hz")

    # Group by original CSV identity
    csv_groups = defaultdict(list)
    for f in all_files:
        csv_id = extract_csv_id(f.name)
        csv_groups[csv_id].append(f)

    # Prepare for stratified sampling by frequency
    freq_to_csvs = defaultdict(list)
    for csv_id in csv_groups:
        freq = extract_freq_group(csv_id)
        freq_to_csvs[freq].append(csv_id)

    train_files = []
    test_files = []
    random.seed(RANDOM_SEED)

    # Iterate through each frequency group for splitting
    for freq, csv_ids in sorted(freq_to_csvs.items()):
        random.shuffle(csv_ids)

        # Calculate split count, ensuring at least 1
        n_train = max(1, round(len(csv_ids) * TRAIN_RATIO))
        if len(csv_ids) == 1:
            n_train = 1  # If there is only one CSV source, assign it to the training set

        train_csvs = csv_ids[:n_train]
        test_csvs = csv_ids[n_train:]

        for csv_id in train_csvs:
            train_files.extend(csv_groups[csv_id])
        for csv_id in test_csvs:
            test_files.extend(csv_groups[csv_id])

        print(f"    - {freq}: {len(csv_ids)} source files -> {len(train_csvs)} Train, {len(test_csvs)} Test")

    return train_files, test_files


def copy_files(files, dest_dir, category):
    """Copy files to the destination path."""
    cat_dest = dest_dir / category
    cat_dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, cat_dest / f.name)


def process_split(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)

    train_dir = output_path / "train"
    test_dir = output_path / "test"

    # Clean up previous split results every time the script runs
    for d in [train_dir, test_dir]:
        if d.exists():
            print(f"Cleaning up old directory: {d}")
            shutil.rmtree(d)

    # Define your 5 folder categories
    categories = ["noise", "sine", "square", "pulse", "ramp"]

    total_train = 0
    total_test = 0

    for cat in categories:
        print(f"\nProcessing category: {cat}")
        train_files, test_files = split_category(input_path, cat)

        if not train_files and not test_files:
            continue

        copy_files(train_files, train_dir, cat)
        copy_files(test_files, test_dir, cat)

        print(f"  [Complete] {cat}: {len(train_files)} training images, {len(test_files)} testing images")
        total_train += len(train_files)
        total_test += len(test_files)

    print(f"\n{'=' * 60}")
    print(f"Split task completed successfully!")
    print(f"Total training set: {total_train} images")
    print(f"Total testing set: {total_test} images")
    print(f"Results stored in: {output_path}")


if __name__ == "__main__":
    # ==========================================
    # --- Please modify your paths here ---
    # ==========================================
    # The root directory where your 5 category folders are located
    MY_INPUT_DIR = "/Users/ilminurablikim/Desktop/images_Timedomain_augmented"

    # The location where you want the train/test folders to be output
    MY_OUTPUT_DIR = "/Users/ilminurablikim/Desktop/split_dataset_aug"
    # ==========================================

    process_split(MY_INPUT_DIR, MY_OUTPUT_DIR)
