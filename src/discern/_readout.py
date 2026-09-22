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

import json
import functools
import os
import time

import torch
import transformers

from ._semif.core import DIRECT_SYSTEM, LETTERS, digest, softmax, validate_row

PROMPT_VERSION = "direct-options-vl-v1"
DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
# En CPU el 4B no compensa, y no por poco: 5707 ms frente a 2458 del 2B en esta
# maquina (i9-12900F, 8 hilos), 21.5 GiB de RAM residente frente a 12.7, y un
# colapso de float32 del 62% frente al 26%. Pierde 4.1 puntos en MMBench y gana
# en POPE. El factor CPU/GPU ademas empeora al escalar: x44 en el 2B, x63 en el
# 4B. Medido en scripts/bench_cpu_vision.py.
DEFAULT_MODEL_CPU = "Qwen/Qwen3-VL-2B-Instruct"


# ---------------------------------------------------------------- loader
@functools.lru_cache(maxsize=16)
def checkpoint_bytes(source: str) -> int | None:
    """Tamano del checkpoint en el hub, para estimar si cabe en la VRAM.

    Cacheado: sin esto son ~160 ms de red por consulta, y choose() hace dos.
    """
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(source, files_metadata=True)
        return sum(f.size or 0 for f in info.siblings
                   if f.rfilename.endswith(".safetensors"))
    except Exception:
        return None


def pick_device(source: str, holgura_gib: float = 1.2) -> tuple[str, "torch.dtype"]:
    """Elige GPU si el checkpoint cabe con holgura; si no, CPU en float32.

    En CPU se usa float32 a proposito: bf16 va emulado salvo en CPUs con
    avx512_bf16 o amx, y ahi resulta mas lento que fp32.

    La holgura de 1.2 GiB esta calibrada: el 4B tiene un checkpoint de 8.3 GiB
    y su pico medido con imagenes normales es 8.4. Aun asi la estimacion puede
    fallar (otra cosa ocupando VRAM, imagenes enormes), asi que load_vl_model
    reintenta en CPU si la GPU da OOM.
    """
    if not torch.cuda.is_available():
        return "cpu", torch.float32
    libre = torch.cuda.mem_get_info()[0] / 2**30
    n = checkpoint_bytes(source)
    if n is None:
        return ("cuda:0", torch.bfloat16) if libre >= 6.0 else ("cpu", torch.float32)
    necesita = n / 2**30 + holgura_gib
    return ("cuda:0", torch.bfloat16) if libre >= necesita else ("cpu", torch.float32)


def choose(device: str | None = None) -> tuple[str, str]:
    """Elige modelo y dispositivo A LA VEZ, en una sola escalera.

    Decidirlos por separado permite que se contradigan: mirar si cabe el 4B,
    concluir "CPU", elegir por eso el modelo de CPU, y acabar poniendolo en la
    GPU porque ese si cabia. Aqui la decision es una.

      1. el 4B en GPU, si cabe        (mejor precision)
      2. el 2B en GPU, si cabe        (x1.6 mas rapido, menos precision)
      3. el 2B en CPU                 (~50x mas lento; el 4B en CPU es x2.3
                                       peor y pide 21.5 GiB de RAM)
    """
    if device == "cpu":
        return DEFAULT_MODEL_CPU, "cpu"
    for modelo in (DEFAULT_MODEL, DEFAULT_MODEL_CPU):
        dev, _ = pick_device(modelo)
        if dev != "cpu":
            return modelo, (device or dev)
    # caer a CPU cuesta ~50x: decirlo, no dejar que se note por el reloj
    libre = torch.cuda.mem_get_info()[0] / 2**30 if torch.cuda.is_available() else 0
    print(f"[discern] no hay VRAM para el modelo mas pequeno "
          f"({libre:.1f} GiB libres); usando CPU, ~50x mas lento. "
          f"Pasa device='cuda:0' para forzar la GPU.", flush=True)
    return DEFAULT_MODEL_CPU, "cpu"


def load_vl_model(source: str, dtype=None, device: str | None = None):
    if device is None or dtype is None:
        auto_dev, auto_dt = pick_device(source)
        device = device or auto_dev
        dtype = dtype or (torch.float32 if device == "cpu" else auto_dt)
    if device == "cpu" and torch.get_num_threads() > 8:
        # CPUs hibridas (P+E): cenirse a los P-cores es ~1.9x mas rapido,
        # ver scripts/bench_cpu4b.py
        torch.set_num_threads(min(8, os.cpu_count() or 8))
    t0 = time.time()
    processor = transformers.AutoProcessor.from_pretrained(source)
    try:
        model = transformers.AutoModelForImageTextToText.from_pretrained(
            source, dtype=dtype, device_map={"": device}, low_cpu_mem_usage=True,
        )
    except torch.OutOfMemoryError:
        # la estimacion de pick_device fallo: caer a CPU en vez de reventar
        if device == "cpu":
            raise
        torch.cuda.empty_cache()
        print(f"[discern] no cabe en la GPU, cargando en CPU (float32). "
              f"Sera ~50x mas lento; considera un modelo menor.", flush=True)
        device, dtype = "cpu", torch.float32
        if torch.get_num_threads() > 8:
            torch.set_num_threads(min(8, os.cpu_count() or 8))
        model = transformers.AutoModelForImageTextToText.from_pretrained(
            source, dtype=dtype, device_map={"": device}, low_cpu_mem_usage=True,
        )
    model.eval()
    metadata = {
        "source": source,
        "dtype": str(dtype).replace("torch.", ""),
        "device": device,
        "threads": torch.get_num_threads() if device == "cpu" else None,
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
            # una ruta/URL va como "url"; un PIL.Image (datasets) va como "image"
            {"type": "image", "url": row["image"]} if isinstance(row["image"], str)
            else {"type": "image", "image": row["image"]},
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
