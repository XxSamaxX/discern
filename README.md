# semif-vision

![What it does](docs/hero.png)

Porting the **semantic if** — a decision read straight from a model's
answer-token logits, with no decoding loop — from text to **vision**, and
measuring what the reported probability actually tells you.

Built on [SemIf](https://github.com/TheoLeeCJ/SemIf) and the
[JEV-CPU](https://huggingface.co/Meanblock/JEV-CPU) port. The scoring engine is
vendored unchanged (MIT) so that image results are directly comparable to text
ones; see [THIRD_PARTY.md](THIRD_PARTY.md).

## What this is

A decision becomes a lettered multiple choice. One forward pass. Read the final
logits at the tokens `A`, `B`, `C`. No generation.

For images this means you can evaluate *predicates* a fixed-vocabulary detector
cannot express — "is this photo posed?", "is the white balance wrong?", "is the
meal already in progress?" — without training anything.

## Findings

**It works, and it is fast.** On **MMBench en/dev (4,329 items)** Qwen3-VL-4B
scores **87.4%** with one forward pass per item at **91 ms/item**, and **82.2%**
under MMBench's own CircularEval protocol. On our own 22-item probe it answers
22/22, including counting, absence, a false-premise question, an unanswerable
question, and reading an Adidas mark spanning ~20×15 px.

**The probability it reports is dangerous; the gap is not.** *(the bars in the
figure above are the gap, not the probability)* Over 4,329 MMBench items the
softmax reports **p ≥ 0.99 on 52% of the model's own errors** — it claims
certainty on more than half of what it gets wrong. The logit gap behind that
same number **predicts correctness with AUC 0.894** (median 22.4 nats when
right, 5.2 when wrong). **Threshold on the gap in nats, never on the
probability.** It buys a real operating point:

| gap threshold | you answer | accuracy |
|---|---|---|
| none | 100% | 87.4% |
| ≥ 12 nats | 75% | 96.8% |
| ≥ 16 nats | 66% | **98.4%** |
| ≥ 20 nats | 55% | 99.1% |

**Position bias is 1–2 nats, and the gap tells you exactly when it matters.**
Rotating the options changes the winner on **11.7% of MMBench items**. Those
items have a **median gap of 2.7 nats**; the stable ones, 22.8. The gap predicts
order-stability with **AUC 0.980**. So: rotate and average only below ~3 nats —
a rule we wrote from a 22-item probe before downloading the dataset, and which
lands on the empirical median of 4,329 items.

**The gap will not protect you from questions the model cannot answer.** On 12
normative questions (consent, retention, licensing — policy no model can
ground) the gap distribution shifts but overlaps badly: AUC 0.788, *p* = 0.006,
and the best possible threshold still admits **7 of 12**. The model claims a
photo needs legal consent at 15.6 nats, about as confidently as it reports
there is no dog in the picture at 26.4. **Use semantic ifs for perception;
keep policy in code.**

**A semantic if is an `if`, not a `switch`.** Our first cascade asked one
four-option policy question whose options were not mutually exclusive, and the
model oscillated between two *correct* answers. Independent binary predicates
composed in Python fixed it.

**Where it breaks.** The worst MMBench categories are `spatial_relationship`
(34.5% under CircularEval, near chance) and `image_quality` (57.3%). Our 22-item
probe had flagged exactly these two: its lowest gaps were the spatial-relation
item (8.1 nats) and the white-balance item (7.8). The confidence signal
identified the weak categories from a handful of examples.

**Systems note.** On a hybrid Intel CPU (i9-12900F, 8P+8E), restricting
inference to the 8 P-cores is **1.9× faster** than using 20 threads —
scheduling onto E-cores actively hurts. Overlapping a GPU vision stage with a
CPU text stage in two threads saved 22% wall clock.

Full write-up with method and statistics: [`paper/paper.pdf`](paper/paper.pdf).

## Layout

```
src/semif_vl.py        vision port; readout identical to upstream, + gap/rotation calibration
src/cascade.py         GPU predicates -> CPU policy, threaded pipeline
src/bench_cpu4b.py     CPU thread sweep and policy benchmark
scripts/eval_mmbench.py       MMBench harness (resumable JSONL)
scripts/analiza_mmbench.py    accuracy, AUC, abstention curves
src/semif_phase1/      vendored upstream engine (MIT)
scripts/exp_gap_por_tipo.py   the perceptual-vs-normative experiment
scripts/figuras.py     paper figures
data/                  benchmark decisions + 4 COCO val2017 images
results/               raw output of every run reported
paper/                 LaTeX sources and compiled PDF
```

## Running it

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch torchvision transformers \
    accelerate safetensors huggingface-hub numpy pillow matplotlib

.venv/bin/python src/semif_vl.py --rows data/vl_rows_hard.json --calibrate
.venv/bin/python scripts/exp_gap_por_tipo.py
.venv/bin/python src/bench_cpu4b.py          # needs ~21 GiB RAM

# MMBench: ~7 min vanilla, ~25 min CircularEval
.venv/bin/python scripts/eval_mmbench.py
.venv/bin/python scripts/eval_mmbench.py --circular
.venv/bin/python scripts/analiza_mmbench.py results/mmbench_*_4329.jsonl
```

First run downloads Qwen3-VL-4B-Instruct (~8 GB).

## Caveats

The MMBench numbers are one model on one split, single run, with one prompt
template; no confidence intervals on latency. MMBench dev is public and may
sit in the model's training data, so 87.4% is not evidence about generalisation
— the claims that matter here are about the *confidence signal*, which is
measured within the same run and does not depend on the absolute score.

Our own 22-item probe is small and author-annotated; one label was wrong (we
called a logo illegible, the model read it correctly) and was corrected. Its
22/22 never located the model's failure boundary, which is why the MMBench run
exists.

## Licence

MIT, matching upstream. See [LICENSE](LICENSE) and [THIRD_PARTY.md](THIRD_PARTY.md).
