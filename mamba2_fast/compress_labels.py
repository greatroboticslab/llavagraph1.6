"""
compress_labels.py
===================
Takes the existing Gemini-generated DIAGNOSIS+CORRECTION labels in
mamba_llm/training_data.csv and compresses each one to ~1-2 sentences
(~30-50 tokens), for the short-output Mamba2-780m fine-tuning target.

This does NOT re-run the vision pipeline — it sends only the existing text
to Gemini (text-only call, no image), asking it to shorten while preserving
the substance. The CORRECTION VECTOR block is never touched by the LLM: it
is split off before compression and reattached unchanged afterward, exactly
as it already flows through infer_zamba.py / eval_zamba.py.

Content-preservation safeguard: every number mentioned in the compressed
text is checked against the numbers in the original text. Any number that
appears in the compressed version but NOT in the original is flagged as a
likely hallucination in flagged_for_review.csv, for manual spot-checking
rather than silent trust.

Usage:
    python compress_labels.py \
        --input  ../mamba_llm/training_data.csv \
        --output ./training_data_short.csv
"""

import argparse
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
)

CV_MARKER = "CORRECTION VECTOR:"

COMPRESS_INSTRUCTION = (
    "You are compressing a piezoelectric actuator diagnostic report. Below is "
    "the ORIGINAL DIAGNOSIS and CORRECTION text.\n\n"
    "Rewrite it in EXACTLY this shape:\n"
    "DIAGNOSIS: <one sentence, the single primary defect and its key measured "
    "evidence>\n"
    "CORRECTION: <one sentence, the correction strategy>\n\n"
    "Hard rules:\n"
    "- Keep every number that appears in the ORIGINAL. Do not round, drop, or "
    "invent any numeric value.\n"
    "- Do not introduce any claim, defect, or number that is not already in "
    "the ORIGINAL text.\n"
    "- Cut only redundant phrasing and secondary/minor observations — keep "
    "the single most important defect, not a list of all of them.\n"
    "- Total output under 40 words.\n"
    "- Always write complete sentences — never leave a sentence unfinished, "
    "even if that means going slightly over 40 words.\n\n"
    "ORIGINAL:\n{original}"
)


def split_cv(full_text: str):
    idx = full_text.find(CV_MARKER)
    if idx == -1:
        return full_text.strip(), ""
    return full_text[:idx].strip(), full_text[idx:].strip()


def call_gemini_text(api_key: str, original_text: str, max_retries: int = 4):
    payload = {
        "contents": [{"parts": [
            {"text": COMPRESS_INSTRUCTION.format(original=original_text)},
        ]}],
        "generationConfig": {
            "maxOutputTokens": 160,
            "temperature": 0.1,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    for attempt in range(max_retries):
        try:
            r = requests.post(GEMINI_URL, params={"key": api_key},
                               json=payload, timeout=30)
            if r.status_code == 429:
                wait = 15 * (2 ** attempt)
                print(f"\n    [rate-limit] sleeping {wait}s ...", flush=True)
                time.sleep(wait)
                continue
            if r.status_code == 400:
                print(f"\n    [400 body] {r.text[:500]}", flush=True)
            r.raise_for_status()
            return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except requests.Timeout:
            print(f"\n    [timeout {attempt+1}/{max_retries}]", flush=True)
            time.sleep(5)
        except Exception as e:
            # str(e) on a requests exception includes the full request URL,
            # which includes ?key=<api_key> — scrub it before ever printing.
            safe_msg = re.sub(r"key=[^&\s'\")]+", "key=***REDACTED***", str(e))
            print(f"\n    [error {attempt+1}/{max_retries}] {safe_msg}", flush=True)
            if attempt == max_retries - 1:
                return None
            time.sleep(5)
    return None


NUMBER_RE = re.compile(r"-?\d+\.?\d*")


def extract_numbers(text: str) -> set:
    # Round to 1 decimal when comparing so trivial formatting differences
    # (206.7 vs 206.70) don't trigger false-positive hallucination flags.
    out = set()
    for m in NUMBER_RE.findall(text):
        try:
            out.add(round(float(m), 1))
        except ValueError:
            pass
    return out


def hallucinated_numbers(original: str, compressed: str) -> set:
    return extract_numbers(compressed) - extract_numbers(original)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../mamba_llm/training_data.csv")
    parser.add_argument("--output", default="./training_data_short.csv")
    parser.add_argument("--flagged", default="./flagged_for_review.csv")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "ERROR: GEMINI_API_KEY not set in the environment.\n"
            "  This script reads os.environ directly (no auto .env loading).\n"
            "  Run:  set -a; source ../mamba_llm/.env; set +a\n"
            "  ...then re-run this script."
        )

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    out_path = Path(args.output)
    done = set()
    results = []
    if out_path.exists() and out_path.stat().st_size > 0:
        existing = pd.read_csv(out_path)
        results = existing.to_dict("records")
        done = set(existing["image_path"].tolist())
        print(f"Resuming: {len(done)} already done, {len(df) - len(done)} remaining")

    flagged = []
    rows = df if args.limit is None else df.head(args.limit)
    todo = [r for _, r in rows.iterrows() if r["image_path"] not in done]
    print(f"To process: {len(todo)}")

    start = time.time()
    for i, row in enumerate(todo):
        gemini_text, cv_block = split_cv(row["description"])

        short_text = call_gemini_text(api_key, gemini_text)
        if short_text is None:
            print(f"[{i+1}/{len(todo)}] FAILED — keeping original (uncompressed)")
            short_text = gemini_text  # fail-safe: don't lose the sample

        bad_numbers = hallucinated_numbers(gemini_text, short_text)
        if bad_numbers:
            flagged.append({
                "image_path": row["image_path"],
                "original": gemini_text,
                "compressed": short_text,
                "suspect_numbers": sorted(bad_numbers),
            })

        full_short = short_text.strip() + ("\n\n" + cv_block if cv_block else "")
        results.append({**row.to_dict(), "description": full_short})

        elapsed = time.time() - start
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (len(todo) - i - 1) / rate if rate > 0 else 0
        flag_tag = f"  [FLAGGED: {sorted(bad_numbers)}]" if bad_numbers else ""
        print(f"[{i+1:4d}/{len(todo)}] {row['waveform_label']:<7} "
              f"{len(gemini_text)}->{len(short_text)} chars  "
              f"ETA {eta/60:.0f}m{flag_tag}")

        if (i + 1) % 25 == 0:
            pd.DataFrame(results).to_csv(out_path, index=False)
            if flagged:
                pd.DataFrame(flagged).to_csv(args.flagged, index=False)

        time.sleep(4)  # free-tier rate limit: 15 req/min

    pd.DataFrame(results).to_csv(out_path, index=False)
    if flagged:
        pd.DataFrame(flagged).to_csv(args.flagged, index=False)

    print(f"\nDone. {len(results)} rows -> {out_path}")
    print(f"Flagged for manual review (possible hallucinated numbers): "
          f"{len(flagged)} -> {args.flagged}")


if __name__ == "__main__":
    main()
