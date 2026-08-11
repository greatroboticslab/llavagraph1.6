"""
prepare_classifier_data.py
===========================
Converts the existing data/{train,val,test}.jsonl (built for the
text-generation track) into a classification version: same system/input,
but the target is a single-character defect-category code instead of a
DIAGNOSIS/CORRECTION paragraph. Labels come from label_scheme.py — a fixed,
deterministic function of the measured features already in the input text.

Usage:
    python prepare_classifier_data.py
"""

import json
from pathlib import Path
from collections import Counter

from label_scheme import compute_label, LABEL_TO_CATEGORY

SYSTEM_PROMPT = (
    "You are a piezoelectric actuator defect classifier. Given the measured "
    "waveform features below, output ONLY the single-character code for the "
    "dominant defect category. No words, no punctuation — one character."
)


def convert(in_path, out_path):
    examples = [json.loads(l) for l in open(in_path) if l.strip()]
    counts = Counter()
    with open(out_path, "w") as f:
        for ex in examples:
            label = compute_label(ex["input"], ex["waveform"])
            counts[(ex["waveform"], label)] += 1
            f.write(json.dumps({
                "system": SYSTEM_PROMPT,
                "input": ex["input"],
                "label": label,
                "waveform": ex["waveform"],
            }) + "\n")
    return examples, counts


def main():
    data_dir = Path("./data")
    for split in ["train", "val", "test"]:
        in_path = data_dir / f"{split}.jsonl"
        out_path = data_dir / f"{split}_cls.jsonl"
        examples, counts = convert(in_path, out_path)

        print(f"{split}: {len(examples)} examples -> {out_path}")
        by_type = {}
        for (wt, lbl), c in counts.items():
            by_type.setdefault(wt, {})[lbl] = c
        for wt, d in sorted(by_type.items()):
            total = sum(d.values())
            maj = max(d.values())
            print(f"    {wt:<8} n={total:<4} categories={d}  "
                  f"majority_baseline={maj/total*100:.1f}%")


if __name__ == "__main__":
    main()
