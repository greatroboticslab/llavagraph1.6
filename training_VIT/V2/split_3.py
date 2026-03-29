import re
import random
import shutil
from pathlib import Path
from collections import defaultdict

# --- Configuration ---
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2
RANDOM_SEED = 42
EXCLUDE_500HZ = True


def extract_freq_group(filename):
    fname = filename.lower()
    if any(x in fname for x in ["noise", "pulse", "ramp"]):
        for x in ["noise", "pulse", "ramp"]:
            if x in fname: return x
    match = re.search(r'_(\d+)(?:hz|hx)', fname, re.IGNORECASE)
    if match:
        return f"{match.group(1)}Hz"
    return "unknown"


def extract_csv_id(filename):
    stem = Path(filename).stem
    return re.sub(r'_seg\d+_(TIME|FFT)$', '', stem)


def should_exclude(filename, category):
    if not EXCLUDE_500HZ:
        return False
    if category in ["noise", "pulse", "ramp"]:
        return False
    match = re.search(r'_500(?:hz|hx)', filename, re.IGNORECASE)
    return match is not None


def split_category(image_dir, category):
    cat_path = image_dir / category
    if not cat_path.exists():
        print(f"  [Skip] Folder does not exist: {cat_path}")
        return [], [], []

    all_files = sorted(cat_path.glob("*.png"))

    if category in ("sine", "square"):
        before = len(all_files)
        all_files = [f for f in all_files if not should_exclude(f.name, category)]
        excluded = before - len(all_files)
        if excluded:
            print(f"  [Filter] Excluded {excluded} images of 500Hz")

    # Group by original CSV identity to prevent data leakage
    csv_groups = defaultdict(list)
    for f in all_files:
        csv_id = extract_csv_id(f.name)
        csv_groups[csv_id].append(f)

    freq_to_csvs = defaultdict(list)
    for csv_id in csv_groups:
        freq = extract_freq_group(csv_id)
        freq_to_csvs[freq].append(csv_id)

    train_files, val_files, test_files = [], [], []
    random.seed(RANDOM_SEED)

    for freq, csv_ids in sorted(freq_to_csvs.items()):
        random.shuffle(csv_ids)

        # Calculate split indices
        total_csvs = len(csv_ids)
        n_train = max(1, round(total_csvs * TRAIN_RATIO))
        n_val = max(1, round(total_csvs * VAL_RATIO)) if total_csvs > 2 else 0

        # Ensure at least 1 in test if possible
        train_csvs = csv_ids[:n_train]
        val_csvs = csv_ids[n_train:n_train + n_val]
        test_csvs = csv_ids[n_train + n_val:]

        for csv_id in train_csvs: train_files.extend(csv_groups[csv_id])
        for csv_id in val_csvs: val_files.extend(csv_groups[csv_id])
        for csv_id in test_csvs: test_files.extend(csv_groups[csv_id])

        print(
            f"    - {freq}: {total_csvs} sources -> {len(train_csvs)} Train, {len(val_csvs)} Val, {len(test_csvs)} Test")

    return train_files, val_files, test_files


def copy_files(files, dest_dir, category):
    cat_dest = dest_dir / category
    cat_dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, cat_dest / f.name)


def process_split(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)

    train_dir = output_path / "train"
    val_dir = output_path / "val"
    test_dir = output_path / "test"

    for d in [train_dir, val_dir, test_dir]:
        if d.exists():
            print(f"Cleaning up old directory: {d}")
            shutil.rmtree(d)

    categories = ["noise", "sine", "square", "pulse", "ramp"]
    stats = {"train": 0, "val": 0, "test": 0}

    for cat in categories:
        print(f"\nProcessing category: {cat}")
        tr, vl, ts = split_category(input_path, cat)

        if not any([tr, vl, ts]): continue

        copy_files(tr, train_dir, cat)
        copy_files(vl, val_dir, cat)
        copy_files(ts, test_dir, cat)

        print(f"  [Complete] {cat}: {len(tr)} Train, {len(vl)} Val, {len(ts)} Test")
        stats["train"] += len(tr)
        stats["val"] += len(vl)
        stats["test"] += len(ts)

    print(f"\n{'=' * 60}")
    print(f"Three-way split task completed successfully!")
    print(f"Total: {stats['train']} Train | {stats['val']} Val | {stats['test']} Test")
    print(f"Results stored in: {output_path}")


if __name__ == "__main__":
    # --- Modify these paths ---
    MY_INPUT_DIR = "augmented_images"
    MY_OUTPUT_DIR = ("split_dataset_3_aug")

    process_split(MY_INPUT_DIR, MY_OUTPUT_DIR)