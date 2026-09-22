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

**It works, and it is fast.** Qwen3-VL-4B answers **22/22** on our benchmark at
**110–140 ms/decision** on an RTX 3080 (8.4 GiB VRAM). That includes counting,
absence, a false-premise question, an unanswerable question, and reading an
Adidas mark spanning ~20×15 px. Option order never changed a winner.

**The probability it reports is useless.** *(the bars in the figure above are the gap, not the probability)* The softmax over 2–3 answer slots
gave **≥ 0.9993 on every single decision**, while the logit gap behind it spans
7.3–26.4 nats and orders the decisions by real difficulty.
**Threshold on the gap in nats, never on the probability.**

**Position bias is 1–2 nats.** Measured directly by rotating options. That is an
order of magnitude below perceptual gaps, so order calibration buys nothing
there — but it decides the answer outright for small text models, whose gaps
fall to 0.8–2.3 nats. Rule of thumb: rotate and average only below ~3 nats.

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
```

First run downloads Qwen3-VL-4B-Instruct (~8 GB).

## Caveats

Small benchmark: 4 images, 34 decisions, one model, one prompt template.
Ground truth is single-annotator and author-written — one label was wrong
(we called a logo illegible; the model read it correctly) and was corrected.
22/22 means our items never found the model's failure boundary, so the number
bounds nothing. Single machine, single run, no confidence intervals on latency.

## Licence

MIT, matching upstream. See [LICENSE](LICENSE) and [THIRD_PARTY.md](THIRD_PARTY.md).
