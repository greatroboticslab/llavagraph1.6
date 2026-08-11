"""
eval_classifier.py
===================
Evaluates the fine-tuned single-token defect classifier: per-waveform-type
accuracy against the majority-class baseline (so a high number can't be
mistaken for real learning when it's actually just always guessing the
majority class — see label_scheme.py for why that baseline matters here),
plus real per-sample latency.

Speed is measured the same way as eval_mamba2.py (wall-clock around
model.generate()) for comparability — the key expected difference is
max_new_tokens=1 here: a single forward pass, no autoregressive loop, so
this is the number that actually tests whether classification reaches
genuine ms-level latency where the generation track topped out at ~2.3s.

Usage:
    python eval_classifier.py                     # evaluates the weighted checkpoint (default)
    python eval_classifier.py --suffix ""          # evaluates the original unweighted checkpoint
"""

import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--suffix", default="_weighted",
                    help="Selects checkpoints_cls<suffix>/final and names "
                         "output files eval_results_cls<suffix>.jsonl etc. "
                         "Use '' for the original unweighted run.")
args = parser.parse_args()

MODEL_PATH  = f"/projects/ya4v/llavagraph1.6/mamba2_fast/checkpoints_cls{args.suffix}/final"
TEST_PATH   = "/projects/ya4v/llavagraph1.6/mamba2_fast/data/test_cls.jsonl"
RESULTS_DIR = Path("/projects/ya4v/llavagraph1.6/mamba2_fast/results")
RESULTS_DIR.mkdir(exist_ok=True)

print(f"Loading tokenizer: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading model: {MODEL_PATH}")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

try:
    import mamba_ssm, causal_conv1d  # noqa: F401
    print("Fast SSM kernels: available")
except ImportError:
    print("WARNING: mamba_ssm/causal_conv1d not installed — speed numbers "
          "below will NOT reflect Mamba's real potential (naive PyTorch scan).")

test_data = [json.loads(l) for l in open(TEST_PATH) if l.strip()]
print(f"Test samples: {len(test_data)}")


def build_prompt(ex):
    # Must match train_classifier.py's build_prompt() exactly.
    return f"{ex['system']}\n\n{ex['input']}\n\nCategory:"


def run_inference(ex):
    prompt_text = build_prompt(ex)
    input_ids = tokenizer(prompt_text, return_tensors="pt",
                          add_special_tokens=False).input_ids.to(model.device)

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    elapsed_s = time.time() - t0

    pred = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    return pred, elapsed_s


results_path = RESULTS_DIR / f"eval_results_cls{args.suffix}.jsonl"
metrics_path = RESULTS_DIR / f"metrics_summary_cls{args.suffix}.txt"

per_type_correct = defaultdict(int)
per_type_total = defaultdict(int)
per_type_ms = defaultdict(list)
per_type_label_counts = defaultdict(lambda: defaultdict(int))     # true label distribution
per_type_pred_counts = defaultdict(lambda: defaultdict(int))      # predicted label distribution

total_start = time.time()
with open(results_path, "w") as fout:
    for idx, ex in enumerate(test_data):
        wtype = ex["waveform"]
        true_label = ex["label"]
        try:
            pred, elapsed_s = run_inference(ex)
            elapsed_ms = elapsed_s * 1000
            correct = (pred == true_label)

            per_type_total[wtype] += 1
            per_type_correct[wtype] += int(correct)
            per_type_ms[wtype].append(elapsed_ms)
            per_type_label_counts[wtype][true_label] += 1
            per_type_pred_counts[wtype][pred] += 1

            fout.write(json.dumps({
                "idx": idx, "waveform": wtype, "true_label": true_label,
                "predicted": pred, "correct": correct, "ms": round(elapsed_ms, 2),
            }) + "\n")
            fout.flush()

            print(f"[{idx+1:3d}/{len(test_data)}] {wtype:<7} | "
                  f"true={true_label} pred={pred!r} {'OK' if correct else 'X '} | "
                  f"{elapsed_ms:.1f}ms")
        except Exception as e:
            print(f"[{idx+1:3d}/{len(test_data)}] {wtype:<7} | ERROR: {e}")
            fout.write(json.dumps({"idx": idx, "waveform": wtype, "error": str(e)}) + "\n")
            fout.flush()

total_elapsed = time.time() - total_start

lines = ["=" * 70, "EVALUATION SUMMARY — mamba2-780m-hf classifier (fine-tuned)",
          "=" * 70, f"Total samples: {len(test_data)}",
          f"Total time:    {total_elapsed/60:.1f} min\n"]

all_correct, all_total, all_ms = 0, 0, []
for wtype in ["sine", "square", "ramp", "pulse", "noise"]:
    total = per_type_total[wtype]
    if not total:
        continue
    correct = per_type_correct[wtype]
    acc = correct / total
    maj_label = max(per_type_label_counts[wtype], key=per_type_label_counts[wtype].get)
    maj_baseline = per_type_label_counts[wtype][maj_label] / total
    ms = per_type_ms[wtype]
    verdict = "ABOVE majority baseline" if acc > maj_baseline else "AT OR BELOW majority baseline — not real learning"
    lines.append(f"{wtype:<8} n={total:3d} | accuracy={acc*100:5.1f}%  "
                 f"majority_baseline={maj_baseline*100:5.1f}%  [{verdict}]")
    lines.append(f"{'':<8}          avg={sum(ms)/len(ms):.2f}ms  p50={sorted(ms)[len(ms)//2]:.2f}ms  "
                 f"true_dist={dict(per_type_label_counts[wtype])}  pred_dist={dict(per_type_pred_counts[wtype])}")
    all_correct += correct; all_total += total; all_ms.extend(ms)

lines.append("-" * 70)
lines.append(f"{'OVERALL':<8} n={all_total:3d} | accuracy={all_correct/all_total*100:.1f}%  "
             f"avg={sum(all_ms)/len(all_ms):.2f}ms  p50={sorted(all_ms)[len(all_ms)//2]:.2f}ms")
lines.append("=" * 70)
lines.append("\nFor comparison:")
lines.append("  Text-generation track (mamba2-780m, this repo): avg=2274ms/sample")
lines.append("  Zamba2-7B few-shot baseline: avg=~21600ms/sample")

summary = "\n".join(lines)
print("\n" + summary)
with open(metrics_path, "w") as f:
    f.write(summary + "\n")
print(f"\nResults: {results_path}\nMetrics: {metrics_path}")
