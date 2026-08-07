# mamba2_fast

Fast-inference counterpart to `mamba_zamba/`: same piezoelectric actuator
diagnostic task (waveform features in, DIAGNOSIS + CORRECTION text out, plus
an analytically computed CORRECTION VECTOR), but using a fine-tuned
**pure-Mamba2** model instead of a 7B Mamba+Transformer hybrid prompted
few-shot. The goal was to test whether a small, architecturally pure Mamba
model can be both faster and more accurate than the Zamba2-7B few-shot
baseline. Speed improved substantially; accuracy improved as well; the
original millisecond-level latency target was not reached.

---

## Result summary (487-sample test set)

| | Zamba2-7B (few-shot, no fine-tuning) | Mamba2-780M (fine-tuned, this track) |
|---|---|---|
| BLEU-1 | 0.382 | **0.507** |
| ROUGE-L | 0.209 | **0.341** |
| Avg. latency | ~21,600 ms/sample | **2,274 ms/sample** |
| p50 latency | ~21,600 ms/sample | **2,334 ms/sample** |
| Hardware | A100 80GB | RTX 5000 Ada |

Per-waveform-type breakdown (`results/metrics_summary.txt`):

| Waveform | n | BLEU-1 | ROUGE-L | avg ms |
|---|---|---|---|---|
| sine | 120 | 0.496 | 0.342 | 2246 |
| square | 120 | 0.510 | 0.347 | 2277 |
| ramp | 80 | 0.538 | 0.370 | 2279 |
| pulse | 75 | 0.477 | 0.304 | 2313 |
| noise | 92 | 0.514 | 0.337 | 2268 |

**Speed: ~9.5x faster. Accuracy: BLEU-1 +33% relative, ROUGE-L +63%
relative.**

![Learning curve](figures/learning_curve.png)

![Accuracy comparison](figures/accuracy_comparison.png)

![Speed comparison](figures/speed_comparison.png)

---

## Method

**Model:** `AntonV/mamba2-780m-hf` — a community HF-transformers-compatible
conversion of `state-spaces/mamba2-780m` (no official `-hf` conversion
exists for Mamba2 at the time of writing; Mamba-1 has one, Mamba-2 doesn't).
Verified before use (`sanity_check.py`) to load correctly and produce
coherent text, to rule out the class of conversion bug that cost significant
time on the Zamba2 track (the `shared_transformer` weight-tying issue
documented in `mamba_zamba/README.md`).

Pure Mamba-2 architecture, not a hybrid — chosen specifically because the
project's speed claim is about Mamba's architecture, and a hybrid model
(like Zamba2 or Nemotron-H) can't cleanly support that claim.

**Training data:** the same 493 raw hardware measurements as `mamba_zamba/`,
augmented to 2,445 labeled examples, split 1468/490/487 (train/val/test) —
but with the DIAGNOSIS/CORRECTION labels **re-compressed** from the original
60-100 word Gemini text down to ~40 words (`compress_labels.py`, a
text-only Gemini call, not a re-run of the vision pipeline). This was
necessary because generating short output is one of the two real levers on
inference latency (the other being model size); training on the original
long-form target would reproduce Zamba2's latency profile regardless of
model size. Compression preserved every number from the source text — a
hallucination check (`extract_numbers` diff between original and
compressed) flagged 2 of 2445 rows for a truncation artifact, both fixed
(`fix_flagged.py`).

**Fine-tuning:** full-parameter, not LoRA (`finetune_mamba2.py`). At 780M
parameters this is affordable on a single GPU, and it avoids depending on
PEFT/LoRA support for Mamba2, which is newer and less proven than
LoRA-on-attention-layers. Key settings: LR 2e-5, weight decay 0.01, label
smoothing 0.05, up to 10 epochs with early stopping (patience 3) on eval
loss — the epoch count is a ceiling, not a target, given 780M parameters
against only 1,468 training examples makes overfitting the primary risk,
not underfitting.

**Evaluation:** `eval_mamba2.py`, same measurement methodology as
`mamba_zamba/eval_zamba.py` (wall-clock around `model.generate()`, BLEU-1 /
ROUGE-L against the reference) for comparability. Zero-shot — this model is
fine-tuned directly on the task, so no in-context example is built into the
prompt (unlike Zamba2's 1-shot few-shot approach).

---

## What went wrong, and what fixed it

Documented here rather than only in the codebase's inline comments, because
several of these were non-obvious and are worth knowing before extending
this work.

**1. Training loss looked ~4x higher than eval loss.**
Normal training has train loss at or below eval loss; here train loss
converged around 6.7 while eval loss converged around 1.7. Root-caused with
a controlled experiment (`diagnose_loss_gap.py`): same fine-tuned model,
same data, only `gradient_accumulation_steps` changed from 4 to 1 — logged
loss dropped from ~6.8 to ~1.7, matching eval loss almost exactly. This
transformers version logs training loss inflated by roughly the
accumulation factor before averaging. **Confirmed to be a display/logging
artifact, not a real train/eval performance gap** — `finetune_mamba2.py`
now divides the logged value by `GRAD_ACCUM` before saving/plotting, so
`results/training_history.csv` and `figures/learning_curve.png` show the
corrected number. Model selection during training (`metric_for_best_model
="eval_loss"`, early stopping) was never affected by this bug — `eval_loss`
was never computed via the accumulation path — so no training run needed
to be redone for this reason.

**2. Training the model to emit a stop token collapsed generation to
empty output.**
Two independent attempts — (a) appending `eos_token_id` to the training
target plus left-padding (per Mamba2's own docs, which describe
right-padding as unreliable for this architecture), and (b) the same fix
alone — both resulted in the model emitting EOS as literally the first
generated token on every test sample (`tokens: 1`, empty diagnosis text,
BLEU/ROUGE = 0 across the board). A debug dump of the actual training
example (ids, labels, decoded target text) confirmed the training data
itself was correct in both attempts — labels were "\nDIAGNOSIS: ...
<|endoftext|>" exactly as intended. **The root cause of the training
collapse was not identified.** Given two clean attempts failed identically
despite verified-correct input data, further training-side iteration was
judged not worth the cost. The fix that was kept: don't train the model to
stop; instead, since every well-formed target is exactly two lines with no
blank line in it, `eval_mamba2.py` trims generated text at the first blank
line post-hoc. Verified across the full 487-sample test set
(`grep -c "MEASURED FEATURES\|COMMANDED WAVEFORM" results/eval_results.jsonl`
→ 0) that no hallucinated continuation text survives into the final output.

**3. Cluster GPU queueing.**
The `a100` partition on this cluster has exactly one node, so jobs queued
for 1-2 days. `research-gpu` (6 nodes, RTX 5000 Ada, mostly idle) is used
instead throughout this track — sufficient memory (~10GB needed, node has
62GB) for a 780M model. This is why speed numbers here are on a different
GPU than the Zamba2 baseline (see comparability caveats above).

---

## Known limitations / open questions

- **Latency target not met** (see above) — the two unexplored levers are
  further output compression (fewer generated tokens) and a smaller
  backbone (below 780M).
- **Why EOS training collapses training is still unknown.** Anyone
  revisiting this should look at the transformers version's interaction
  between `label_smoothing_factor`, gradient accumulation, and Mamba2's SSM
  state handling — the three variables changed together across the two
  failed attempts, so they weren't cleanly isolated from each other.
- **Prompt/output token-boundary assumption.** `WaveformDataset` computes
  where the trainable label region starts by tokenizing the prompt and the
  full text separately and taking `len(prompt_ids)` as the boundary. BPE
  tokenization is not always prefix-stable across concatenation, so this
  can in principle be off by a token for some fraction of examples. Spot
  checked correct on one example via `diagnose_loss_gap.py`'s debug output;
  not exhaustively verified across all 1,468 training examples.
  `MAX_SEQ_LEN=768` truncation is also unverified as a non-issue across the
  full dataset (no example has been confirmed to exceed it, but none has
  been confirmed not to, either).
- **The model sometimes mislabels a real number's meaning.** In the
  worked example below, the model's CORRECTION sentence says "18.62°
  phase lag" — 18.62 is a real number from the input (`Hysteresis (nm):
  18.62`), but it's hysteresis in nanometers, not a phase lag in degrees.
  The correction vector itself is unaffected (it's computed independently
  by formula, not extracted from the generated text), but the generated
  *prose* can attach a real number to the wrong physical quantity. Shown
  as-is in the example figure below rather than swapped for a cleaner run.
- **BLEU-1/ROUGE-L here are simplified implementations** (no n-gram
  clipping or brevity penalty on BLEU-1), matching `eval_zamba.py`'s
  methodology for internal comparability — not directly comparable to
  standard-library (sacrebleu / rouge-score) numbers reported elsewhere.

---

## Worked example

![Worked example: idx=1 sine at 100 Hz](figures/example_idx1_sine.png)

Test-set sample idx=1 (100 Hz sine, real
hardware measurement). Left panel is the real waveform image (matched to
this sample by its measured peak/RMS/phase-lag values against
`mamba_llm/features_full.csv`, not by filename pattern — an earlier
filename-based guess turned out to point at a different physical sample).
Center panel is the real DIAGNOSIS/CORRECTION text as generated by the
fine-tuned model plus the real BLEU-1/ROUGE-L/latency for that sample, read
programmatically from `results/eval_results.jsonl` — nothing in this panel
is hand-typed. Right panel is the predicted waveform after applying the
real correction vector, computed analytically (labeled as a prediction, not
a hardware re-measurement — no closed-loop hardware run has happened yet).

Regenerate with: `python make_example_figure.py --idx <n>` for any test
sample; add its hardware image to `REAL_IMAGES` in that script after
verifying the match against `features_full.csv` (see the script's
docstring — do not match by filename alone).

---

## Files

| File | Purpose |
|---|---|
| `sanity_check.py` / `run_sanity_check.sbatch` | Verify the base checkpoint loads and generates coherent text |
| `compress_labels.py` | Compress `mamba_llm/training_data.csv` labels to short-form targets |
| `fix_flagged.py` | Re-process rows flagged by `compress_labels.py`'s hallucination check |
| `prepare_data.py` | Build `data/{train,val,test}.jsonl` from the compressed labels |
| `finetune_mamba2.py` / `run_finetune.sbatch` | Full-parameter fine-tuning |
| `diagnose_loss_gap.py` / `run_diagnose.sbatch` | Controlled experiment isolating the loss-logging artifact (see above) |
| `eval_mamba2.py` / `run_eval.sbatch` | Speed + BLEU-1/ROUGE-L evaluation on the 487-sample test set |
| `make_figures.py` | Learning curve, accuracy comparison, speed comparison charts |
| `make_example_figure.py` | The 3-panel worked example above |

## Reproducing

```bash
python compress_labels.py          # ~3h, needs GEMINI_API_KEY
python fix_flagged.py              # only if compress_labels.py flags anything
python prepare_data.py
# sync to cluster, then:
sbatch run_finetune.sbatch         # ~20-30 min on RTX 5000 Ada
sbatch run_eval.sbatch             # ~20 min
# sync results/ back locally, then:
python make_figures.py
python make_example_figure.py --idx 1
```
