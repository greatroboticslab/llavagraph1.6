"""
clean_training_data.py
======================
Remove Gemini artifacts from training_data.csv and normalize description format.

Run:
    python clean_training_data.py
"""

import re
import pandas as pd

INPUT  = "training_data.csv"
OUTPUT = "training_data_clean.csv"

# Patterns that indicate a contaminated Gemini response
ARTIFACT_PATTERNS = [
    r"Please respond with the format",
    r"your entire response should be",
    r"in all capital letters",
    r"the letter \w should appear",
    r"do not include",
    r"your response must contain",
    r"write \d+ paragraphs",
]

ARTIFACT_RE = re.compile("|".join(ARTIFACT_PATTERNS), re.IGNORECASE)

# Preambles Gemini added that aren't part of the actual description
PREAMBLE_RE = re.compile(
    r"^(Here'?s?( a| my| the)? (technical )?description[^:]*:\s*"
    r"|Here'?s?( a| my)? analysis[^:]*:\s*"
    r"|The following (is|describes)[^:]*:\s*"
    r"|output:\s*)",
    re.IGNORECASE,
)


def clean_description(text: str) -> str:
    text = text.strip()
    text = PREAMBLE_RE.sub("", text).strip()
    return text


def is_contaminated(text: str) -> bool:
    return bool(ARTIFACT_RE.search(text))


def main():
    df = pd.read_csv(INPUT)
    total = len(df)

    # Flag contaminated rows
    contaminated = df["description"].apply(is_contaminated)
    n_contaminated = contaminated.sum()
    print(f"Total rows      : {total}")
    print(f"Contaminated    : {n_contaminated} ({100*n_contaminated/total:.1f}%)")

    # Drop contaminated rows
    df = df[~contaminated].copy()

    # Strip preambles from remaining descriptions
    df["description"] = df["description"].apply(clean_description)

    # Drop rows where description became too short after cleaning
    too_short = df["description"].str.len() < 50
    print(f"Too short after clean: {too_short.sum()}")
    df = df[~too_short].copy()

    # Print split counts
    for split in ["train", "val", "test"]:
        n = (df["split"] == split).sum()
        print(f"  {split}: {n}")

    df.to_csv(OUTPUT, index=False)
    print(f"\nSaved cleaned data to {OUTPUT}")

    # Show a few cleaned examples
    print("\n--- Sample cleaned descriptions ---")
    for _, row in df.head(3).iterrows():
        print(f"[{row['waveform_label']}] {row['description'][:120]}...")
        print()


if __name__ == "__main__":
    main()
