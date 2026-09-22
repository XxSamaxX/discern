#!/usr/bin/env python3
"""Port de SemIf a vision-language: 'semantic ifs' sobre imagenes.

El readout es IDENTICO al de src/semif_phase1/direct.py (un forward,
logits de la ultima posicion, restringidos a los tokens de las letras).
Solo cambian dos cosas frente al original:
  1. el prompt lleva una imagen ademas del JSON de evidencia/criterio/opciones
  2. el loader usa un modelo image-text-to-text en GPU (bf16)

Anadido propio: calibracion por permutacion de opciones (--calibrate),
porque el repo declara sus probabilidades como "uncalibrated".
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

import torch
import transformers
from semif_phase1.core import DIRECT_SYSTEM, LETTERS, digest, softmax, validate_row

PROMPT_VERSION = "direct-options-vl-v1"
DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"


# ---------------------------------------------------------------- loader
def load_vl_model(source: str, dtype=torch.bfloat16, device: str = "cuda:0"):
    t0 = time.time()
    processor = transformers.AutoProcessor.from_pretrained(source)
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        source, dtype=dtype, device_map={"": device}, low_cpu_mem_usage=True,
    )
    model.eval()
    metadata = {
        "source": source,
        "dtype": str(dtype).replace("torch.", ""),
        "device": device,
        "load_seconds": round(time.time() - t0, 1),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    return model, processor, metadata


# ---------------------------------------------------------------- prompt
def _slot_ids(tokenizer, count: int) -> list[int]:
    """Identico a direct._slot_ids: exige round-trip de un solo token."""
    result = []
    for letter in LETTERS[:count]:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded) != letter:
            raise ValueError(f"Answer slot {letter!r} is not one exact round-trip token")
        result.append(encoded[0])
    if len(result) != len(set(result)):
        raise ValueError("Answer-slot tokens collide")
    return result


def vl_messages(row: dict, options: list[dict]) -> list[dict]:
    """Como core.direct_messages pero con la imagen en el turno de usuario."""
    payload = {
        "criterion": row["question"],
        "options": [
            {"letter": LETTERS[i], "description": o["description"]}
            for i, o in enumerate(options)
        ],
    }
    if row.get("state"):
        payload["evidence"] = row["state"]
    return [
        {"role": "system", "content": [{"type": "text", "text": DIRECT_SYSTEM}]},
        {"role": "user", "content": [
            {"type": "image", "url": row["image"]},
            {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
        ]},
    ]


def _check_boundary(tokenizer, text_prompt: str, slots: list[int]) -> None:
    """Anadir la letra no puede alterar la tokenizacion previa."""
    base = tokenizer.encode(text_prompt, add_special_tokens=False)
    for letter, token in zip(LETTERS, slots):
        if tokenizer.encode(text_prompt + letter, add_special_tokens=False) != base + [token]:
            raise ValueError(f"Answer boundary changes tokenization for slot {letter}")


# ---------------------------------------------------------------- scoring
def _forward_last_logits(model, inputs):
    with torch.inference_mode():
        out = model(**inputs, use_cache=False, return_dict=True)
    return out.logits[:, -1, :][0].float()


def score_once(model, processor, row: dict, options: list[dict]) -> dict:
    tokenizer = processor.tokenizer
    messages = vl_messages(row, options)
    slots = _slot_ids(tokenizer, len(options))

    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    _check_boundary(tokenizer, text_prompt, slots)

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    vocabulary = _forward_last_logits(model, inputs)
    torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - t0

    selected = vocabulary[slots].cpu().tolist()
    return {
        "logits": {o["id"]: lg for o, lg in zip(options, selected)},
        "letters": {LETTERS[i]: o["id"] for i, o in enumerate(options)},
        "option_ids": [o["id"] for o in options],
        "probabilities": softmax(selected),
        "option_logits": selected,
        "input_tokens": int(inputs["input_ids"].shape[1]),
        "forward_seconds": forward_seconds,
        "prompt_sha256": digest(text_prompt),
    }


def score(model, processor, row: dict, metadata: dict, calibrate: bool = False) -> dict:
    """calibrate=True promedia sobre rotaciones ciclicas del orden de opciones,
    para separar la senal del sesgo posicional A/B/C."""
    validate_row({**row, "state": row.get("state") or "image"})
    options = row["options"]
    n = len(options)
    started = time.perf_counter()

    rotations = list(range(n)) if calibrate else [0]
    logit_sum = {o["id"]: 0.0 for o in options}
    slot_logits: dict[str, list[float]] = {}
    per_rotation, tokens, fwd = [], 0, 0.0

    for shift in rotations:
        rotated = options[shift:] + options[:shift]
        r = score_once(model, processor, row, rotated)
        for oid, lg in r["logits"].items():
            logit_sum[oid] += lg / len(rotations)
        for letter, oid in r["letters"].items():
            slot_logits.setdefault(letter, []).append(r["logits"][oid])
        per_rotation.append({"shift": shift, "letters": r["letters"],
                             "logits": r["logits"],
                             "winner": max(r["logits"], key=r["logits"].get)})
        tokens, fwd = r["input_tokens"], fwd + r["forward_seconds"]

    ids = [o["id"] for o in options]
    mean_logits = [logit_sum[i] for i in ids]
    ranked = sorted(mean_logits, reverse=True)
    winners = {rot["winner"] for rot in per_rotation}
    return {
        "id": row["id"],
        "option_ids": ids,
        "probabilities": softmax(mean_logits),
        "mean_logits": mean_logits,
        "gap_nats": ranked[0] - ranked[1],
        "winner_flips": len(winners) > 1,
        "winners_por_rotacion": sorted(winners),
        "slot_logit_medio": {k: sum(v) / len(v) for k, v in slot_logits.items()},
        "input_tokens": tokens,
        "forward_seconds": fwd,
        "forwards": len(per_rotation),
        "total_seconds": time.perf_counter() - started,
        "calibrated": calibrate,
        "per_rotation": per_rotation,
        "prompt_version": PROMPT_VERSION,
        "model": metadata,
        "readout": "native full-vocabulary last-position logits restricted to declared answer slots",
        "probability_status": ("order-averaged over cyclic option rotations"
                               if calibrate else
                               "conditional option score; uncalibrated as decision confidence"),
    }


# ---------------------------------------------------------------- cli
def render(result: dict, truth: str | None = None) -> None:
    pairs = sorted(zip(result["option_ids"], result["probabilities"]), key=lambda x: -x[1])
    mark = ""
    if truth is not None:
        mark = "  [OK]" if pairs[0][0] == truth else f"  [FALLO, esperado {truth}]"
    print(f"\n=== {result['id']} ==={mark}")
    print(f"  -> {pairs[0][0]}   (gap {result['gap_nats']:.1f} nats)")
    for oid, p in pairs:
        print(f"    {oid:22s} {p*100:5.1f}%  {'#' * int(p * 30)}")
    print(f"  ({result['forwards']} fwd, {result['forward_seconds']*1000:.0f} ms total, "
          f"{result['input_tokens']} tok, calibrado={result['calibrated']})")
    if result["calibrated"]:
        flip = ("SI -> " + "/".join(result["winners_por_rotacion"])
                if result["winner_flips"] else "no")
        slots = "  ".join(f"{k}:{v:+.1f}" for k, v in sorted(result["slot_logit_medio"].items()))
        print(f"  cambia el ganador al rotar: {flip}")
        print(f"  logit medio por slot: {slots}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--rows", default=os.path.join(_ROOT, "data", "vl_rows.json"))
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--json", action="store_true", help="volcar resultados crudos")
    args = ap.parse_args()

    print(f"[load] {args.model} (bf16 / cuda)")
    model, processor, meta = load_vl_model(args.model)
    print(f"[load] {meta['load_seconds']}s, "
          f"VRAM {torch.cuda.memory_allocated()/2**30:.1f} GiB, "
          f"transformers={meta['transformers_version']}")

    with open(args.rows) as fh:
        rows = json.load(fh)
    # las rutas de imagen del JSON son relativas a data/
    for row in rows:
        if not os.path.isabs(row["image"]):
            row["image"] = os.path.join(_ROOT, "data", row["image"])

    results, hits, scored = [], 0, 0
    wall = time.perf_counter()
    for row in rows:
        r = score(model, processor, row, meta, calibrate=args.calibrate)
        r["truth"] = row.get("truth")
        results.append(r)
        render(r, row.get("truth"))
        if row.get("truth"):
            scored += 1
            best = max(zip(r["option_ids"], r["probabilities"]), key=lambda x: x[1])[0]
            hits += best == row["truth"]
    wall = time.perf_counter() - wall

    print(f"\n{'='*52}")
    print(f"ACIERTO   {hits}/{scored} ({hits/scored*100:.0f}%)   calibrado={args.calibrate}")
    tot_fwd = sum(r["forwards"] for r in results)
    tot_sec = sum(r["forward_seconds"] for r in results)
    print(f"FORWARDS  {tot_fwd} en {tot_sec:.1f}s  ->  {tot_sec/tot_fwd*1000:.0f} ms/forward")
    print(f"WALL      {wall:.1f}s para {len(results)} decisiones "
          f"({wall/len(results)*1000:.0f} ms/decision)")
    print(f"VRAM pico {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")

    if args.json:
        out = os.path.join(_ROOT, "results", "vl_results.json")
        with open(out, "w") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
        print(f"\n[out] {out}")


if __name__ == "__main__":
    main()
