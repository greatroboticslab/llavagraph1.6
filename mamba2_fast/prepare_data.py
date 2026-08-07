"""
prepare_data.py
===============
Converts training_data_short.csv (output of compress_labels.py) into
train/val/test JSONL for fine-tuning mamba2-780m-hf.

Reuses build_feature_text() from mamba_llm/generate_training_data.py so the
feature formatting is identical to the Zamba2 track (train/inference
consistency). Does NOT reuse that module's SYSTEM_PROMPT — that prompt
tells the model to write 60-100 words across 2-3 sentences, which is the
old long-format target. Our target here is ~30-40 words / one sentence per
section, so the system prompt must say that instead, or the instruction and
the training targets would disagree with each other.

Usage:
    python prepare_data.py \
        --input  ./training_data_short.csv \
        --output_dir ./data
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "../mamba_llm")
from generate_training_data import build_feature_text  # noqa: E402

SYSTEM_PROMPT = (
    "You are a piezoelectric actuator control expert. "
    "The piezo was COMMANDED to produce the waveform stated below. "
    "Your job: write one short diagnosis and one short correction, "
    "for an engineer who needs the answer fast.\n\n"
    "FORMAT (use these exact headers):\n\n"
    "DIAGNOSIS: [ONE sentence] Name the single dominant physical mechanism "
    "causing the deviation (e.g. hysteresis saturation, bandwidth limitation, "
    "creep, resonance) and cite the one or two measured numbers that show it.\n\n"
    "CORRECTION: [ONE sentence] State the correction action and connect it to "
    "the physical cause (e.g. 'advance phase by X deg to counteract the lag').\n\n"
    "RULES:\n"
    "1. Total length: under 40 words across both sentences. Be concise.\n"
    "2. Every number must have a unit.\n"
    "3. English only. No bullet points. No other sections."
)


CV_MARKER = "CORRECTION VECTOR:"


def strip_cv(text: str) -> str:
    # The model never generates the correction vector — it's computed
    # analytically and appended after generation (see eval scripts' own
    # strip_cv()/format_cv()). Training on it would waste capacity and
    # risks teaching the model to hallucinate plausible-looking numbers.
    idx = text.find(CV_MARKER)
    return text[:idx].rstrip() if idx != -1 else text.rstrip()


def row_to_example(row) -> dict:
    features = json.loads(row["features_json"]) if row["features_json"] else {}
    prompt = build_feature_text(row["waveform_label"], features)
    return {
        "system":   SYSTEM_PROMPT,
        "input":    prompt,
        "output":   strip_cv(row["description"]),
        "waveform": row["waveform_label"],
        "split":    row["split"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default="./training_data_short.csv")
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
    print(f"[output]  {sample['output']}")


if __name__ == "__main__":
    main()
