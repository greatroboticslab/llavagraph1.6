"""
fix_flagged.py
===============
Re-compresses only the rows listed in flagged_for_review.csv (identified by
image_path) using the fixed prompt/token-budget in compress_labels.py, and
patches the results directly into training_data_short.csv in place.

For the 2026-08-04 run this fixes 2 rows out of 2445 that were truncated
mid-sentence (see flagged_for_review.csv) — not worth re-running the full
~3 hour job for, but cheap enough to patch individually.

Usage:
    python fix_flagged.py
"""

import os

import pandas as pd

from compress_labels import call_gemini_text, split_cv, hallucinated_numbers

MAIN_CSV    = "./training_data_short.csv"
FLAGGED_CSV = "./flagged_for_review.csv"


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "ERROR: GEMINI_API_KEY not set.\n"
            "  Run:  set -a; source ../mamba_llm/.env; set +a"
        )

    main_df = pd.read_csv(MAIN_CSV)
    flagged_df = pd.read_csv(FLAGGED_CSV)
    print(f"Re-processing {len(flagged_df)} flagged row(s)")

    for _, frow in flagged_df.iterrows():
        image_path = frow["image_path"]
        idx = main_df.index[main_df["image_path"] == image_path]
        if len(idx) == 0:
            print(f"  SKIP (not found in {MAIN_CSV}): {image_path}")
            continue
        idx = idx[0]

        original_full = frow["original"]  # the pre-compression text, saved earlier
        gemini_text, cv_block = split_cv(original_full)

        new_short = call_gemini_text(api_key, gemini_text)
        if new_short is None:
            print(f"  FAILED again: {image_path} — left as-is, needs manual look")
            continue

        bad = hallucinated_numbers(gemini_text, new_short)
        status = f"STILL FLAGGED {sorted(bad)}" if bad else "clean"
        print(f"  {image_path}: {status}")
        print(f"    old: {main_df.at[idx, 'description'][:120]!r}")
        print(f"    new: {new_short[:120]!r}")

        full_short = new_short.strip() + ("\n\n" + cv_block if cv_block else "")
        main_df.at[idx, "description"] = full_short

    main_df.to_csv(MAIN_CSV, index=False)
    print(f"\nPatched {MAIN_CSV}")


if __name__ == "__main__":
    main()
