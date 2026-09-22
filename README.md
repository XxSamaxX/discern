# discern

[![PyPI](https://img.shields.io/pypi/v/discern-vl)](https://pypi.org/project/discern-vl/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

![What it does](docs/hero.png)

```bash
pip install discern-vl
```

Ask a vision model a question. Get the answer out of its logits in one forward
pass — no decoding, nothing trained, no fixed label set.

```python
from discern import discern

if discern("photo.jpg", "is there an identifiable person?"):
    blur_faces()

v = discern("photo.jpg", "what is the camera viewpoint?",
            ["from above", "at eye level", "from below"])
print(v)            # at eye level  [16.8 nats]
```

A predicate a detector cannot express — *"is this photo posed?"*, *"is the
white balance wrong?"*, *"is the meal already in progress?"* — costs one
prefill and about 90 ms.

**The confidence you get back is the logit gap in nats, not a probability.**
That is the whole point of this repo, and it is measured, not asserted: over
4,329 MMBench items the softmax probability was ≥ 0.99 on **52% of the model's
own errors**, while the gap predicts correctness with **AUC 0.894**. So
`discern` refuses to be used as a condition when the gap is too small:

```python
v = discern("blurry.jpg", "is the person wearing a helmet?")
bool(v)             # raises Uncertain: 1.2 nat gap, below the trust threshold
v.answer            # 'yes'  — still there if you want it anyway
v.trusted           # False
```

Below the threshold it also rotates the options and re-scores before answering,
because rotation changes the winner on 11.7% of items and those have a median
gap of 2.7 nats.

Several questions about one image:

```python
from discern import multi

for v in multi("photo.jpg", [
        "is there an identifiable person?",
        "is the white balance wrong?",
        ("what is the camera viewpoint?", ["above", "eye level", "below"]),
]):
    print(v, v.probabilities)
```

Built on [SemIf](https://github.com/TheoLeeCJ/SemIf) and the
[JEV-CPU](https://huggingface.co/Meanblock/JEV-CPU) port. The scoring engine is
vendored unchanged (MIT) so image results stay comparable to the published text
ones; see [THIRD_PARTY.md](THIRD_PARTY.md).

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

**The gap signal weakens as options shrink.** On POPE (9,000 binary
yes/no questions about object presence) the same readout scores **89.6%**
(F1 89.0), but the gap only reaches **AUC 0.777** against MMBench's 0.894, and
the softmax is ≥ 0.99 on **86% of the errors** rather than 52%. With two slots
the model is *confidently* wrong: its median gap when wrong is 16.4 nats, against
5.2 on MMBench. **Ask three or more options where you can**, and demand a much
higher threshold on binary questions.

POPE also shows where object hallucination actually hides. This model does not
over-assert presence — its yes-rate is 42.7–45.6%, *below* the 50% base rate, so
the usual headline metric says it is clean. But precision falls 98.2% → 94.0% →
92.0% from the random to the popular to the adversarial split while recall stays
at exactly 83.9% throughout. The co-occurring distractors do pull it into false
positives; the aggregate yes-rate just hides it.

**Where it breaks.** The worst MMBench categories are `spatial_relationship`
(34.5% under CircularEval, near chance) and `image_quality` (57.3%). Our 22-item
probe had flagged exactly these two: its lowest gaps were the spatial-relation
item (8.1 nats) and the white-balance item (7.8). The confidence signal
identified the weak categories from a handful of examples.

**An optimisation that does not work.** Questions about one image share a
prefix — the system turn plus ~300 visual tokens — so caching it and running
only each question's tail looks free. It is numerically exact (max probability
difference 0.0) and **13–14% slower**, flat across 384, 852 and 1,812-token
images, with no crossover: the cached path loses the fast attention kernel and
that costs more than recomputing the prefix. Image preprocessing is only 3% of
the time; the forward is 97%. Kept behind `use_cache=True` and a
`verify_cache()` checker in case flash-attn flips the balance.

**Systems note.** On a hybrid Intel CPU (i9-12900F, 8P+8E), restricting
inference to the 8 P-cores is **1.9× faster** than using 20 threads —
scheduling onto E-cores actively hurts. Overlapping a GPU vision stage with a
CPU text stage in two threads saved 22% wall clock.

Full write-up with method and statistics: [`paper/paper.pdf`](paper/paper.pdf).

## Layout

```
src/discern/__init__.py       the API: discern(image, question) -> Verdict
src/discern/_batch.py         multi(): several questions, one image
src/discern/_readout.py       the vision port; readout identical to upstream
scripts/eval_probe.py         runs the hand-built 22-item probe
src/discern/_semif/           vendored upstream engine (MIT)
scripts/cascade.py            GPU predicates -> CPU policy, threaded pipeline
scripts/bench_cpu4b.py        CPU thread sweep and policy benchmark
scripts/eval_mmbench.py       MMBench harness (resumable JSONL)
scripts/analiza_mmbench.py    accuracy, AUC, abstention curves
scripts/eval_pope.py          POPE harness (9,000 binary questions)
scripts/analiza_pope.py       F1, yes-rate, precision/recall by split
scripts/exp_gap_por_tipo.py   the perceptual-vs-normative experiment
scripts/figuras.py     paper figures
data/                  benchmark decisions + 4 COCO val2017 images
results/               raw output of every run reported
paper/                 LaTeX sources and compiled PDF
```

## Running it

The library is `pip install discern-vl`. To reproduce the measurements you want
the repo and the benchmark extras:

```bash
git clone https://github.com/XxSamaxX/discern && cd discern
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[bench]"

.venv/bin/python scripts/eval_probe.py --rows data/vl_rows_hard.json --calibrate
.venv/bin/python scripts/exp_gap_por_tipo.py
.venv/bin/python scripts/bench_cpu4b.py      # needs ~21 GiB RAM

# MMBench: ~7 min vanilla, ~25 min CircularEval
.venv/bin/python scripts/eval_mmbench.py
.venv/bin/python scripts/eval_mmbench.py --circular
.venv/bin/python scripts/analiza_mmbench.py results/mmbench_*_4329.jsonl

# POPE: ~34 min for 9,000 binary questions
.venv/bin/python scripts/eval_pope.py
.venv/bin/python scripts/analiza_pope.py
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
