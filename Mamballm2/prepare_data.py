"""
prepare_data.py
===============
Converts training_data_v3.csv into train / val / test JSONL files
for use with finetune.py.

Run locally before uploading to the cluster:
    python prepare_data.py --input /path/to/training_data_v3.csv --output_dir ./data

Each output line is a JSON object:
    {"system": "...", "input": "...", "output": "...", "waveform": "...", "split": "..."}
"""

import argparse
import json
from pathlib import Path

import pandas as pd

# Reuse prompts and feature formatting from generate_training_data
from generate_training_data import (
    SYSTEM_PROMPT,
    build_feature_text,
)


def row_to_example(row) -> dict:
    features = json.loads(row["features_json"]) if row["features_json"] else {}
    # Use the same build_feature_text() as generation — ensures train/inference consistency
    prompt = build_feature_text(row["waveform_label"], features)
    return {
        "system":   SYSTEM_PROMPT,
        "input":    prompt,
        "output":   row["description"].strip(),
        "waveform": row["waveform_label"],
        "split":    row["split"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default="./training_data.csv",
                        help="Path to training_data.csv")
    parser.add_argument("--output_dir", default="./data")
    args = parser.parse_args()

    df  = pd.read_csv(args.input)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"Split counts:\n{df['split'].value_counts().to_string()}\n")

    for split in ["train", "val", "test"]:
        subset   = df[df["split"] == split]
        out_file = out / f"{split}.jsonl"
        with open(out_file, "w") as f:
            for _, row in subset.iterrows():
                f.write(json.dumps(row_to_example(row)) + "\n")
        print(f"  {split}: {len(subset)} examples -> {out_file}")

    sample = row_to_example(df.iloc[0])
    print("\n--- Sample (row 0) ---")
    print(f"[system]  {sample['system'][:80]}...")
    print(f"[input]\n{sample['input'][:300]}...")
    print(f"[output]  {sample['output'][:150]}...")


if __name__ == "__main__":
    main()
