#!/usr/bin/env python3
"""Cascada de dos motores: predicados visuales en GPU -> politica en CPU.

Etapa 1 (GPU, Qwen3-VL): lee la imagen y resuelve predicados visuales.
Etapa 2 (CPU, Qwen3-0.6B): recibe SOLO los veredictos de la etapa 1 como
evidencia textual y decide la accion. Nunca ve pixeles.

Las dos etapas corren en hilos con una cola en medio, asi que mientras la
GPU procesa la imagen k+1 la CPU decide la politica de la imagen k.
torch suelta el GIL durante el forward, asi que el solape es real.

Uso:
  python cascade.py              # pipeline (solapado)
  python cascade.py --secuencial # misma carga, sin solape, para comparar
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

import torch
from semif_phase1.direct import score as text_score

import semif_vl
from loaders import load_cpu


# --------------------------------------------------------------- predicados
VISUAL_PREDICATES = [
    {"key": "balance", "question": "Assess the white balance of this photograph.",
     "options": [{"id": "neutro", "description": "Neutral: whites look white, no colour cast"},
                 {"id": "dominante", "description": "Incorrect: a strong overall colour cast tints the whole image"}]},
    {"key": "personas", "question": "Are there identifiable people visible in the image?",
     "options": [{"id": "con_personas", "description": "Yes, one or more people are identifiable"},
                 {"id": "sin_personas", "description": "No identifiable people"}]},
    {"key": "encuadre", "question": "Is this photograph posed or candid?",
     "options": [{"id": "posada", "description": "Posed: subjects arranged for the camera"},
                 {"id": "espontanea", "description": "Candid: caught mid-action"}]},
]

# Predicados binarios INDEPENDIENTES sobre la evidencia en prosa.
# La logica de negocio se compone en Python (ver decidir()), no dentro del modelo.
POLICY_PREDICATES = [
    {"key": "color", "question": "Does this photograph need to be sent for colour correction?",
     "options": [{"id": "si", "description": "Yes, the colour needs correcting"},
                 {"id": "no", "description": "No, the colour is acceptable"}]},
    {"key": "consentimiento", "question": "Must publication be held until personal consent is obtained?",
     "options": [{"id": "si", "description": "Yes, consent is required before publishing"},
                 {"id": "no", "description": "No, consent is not required"}]},
]


def evidencia_en_prosa(v: dict) -> str:
    """Los veredictos visuales como frase, no como JSON: mide mejor (ver 2x2)."""
    trozos = []
    trozos.append("The image has a strong overall colour cast."
                  if v["balance"]["veredicto"] == "dominante"
                  else "The image has neutral colour balance.")
    trozos.append("Identifiable people appear in the image."
                  if v["personas"]["veredicto"] == "con_personas"
                  else "No identifiable people appear in the image.")
    return " ".join(trozos)


def decidir(flags: dict) -> str:
    """La politica, en codigo. Aqui no hay modelo."""
    if flags["color"] and flags["consentimiento"]:
        return "corregir_y_consentimiento"
    if flags["consentimiento"]:
        return "pedir_consentimiento"
    if flags["color"]:
        return "corregir_color"
    return "publicar"


# --------------------------------------------------------------- etapas
def gpu_stage(vl_model, vl_proc, vl_meta, image: str) -> dict:
    """Resuelve todos los predicados visuales de una imagen."""
    verdicts, seconds, tokens = {}, 0.0, 0
    for pred in VISUAL_PREDICATES:
        row = {"id": pred["key"], "image": image, "state": "",
               "question": pred["question"], "options": pred["options"]}
        r = semif_vl.score(vl_model, vl_proc, row, vl_meta, calibrate=False)
        best = max(zip(r["option_ids"], r["probabilities"]), key=lambda x: x[1])
        verdicts[pred["key"]] = {"veredicto": best[0], "p": round(best[1], 3)}
        seconds += r["forward_seconds"]
        tokens = r["input_tokens"]
    return {"image": os.path.basename(image), "verdicts": verdicts,
            "gpu_seconds": seconds, "input_tokens": tokens}


def cpu_stage(tx_model, tx_tok, tx_meta, item: dict) -> dict:
    """Predicados binarios sobre la evidencia en prosa, compuestos en Python.
    Cada predicado se puntua en sus 2 rotaciones: si el ganador cambia, se marca
    inestable en vez de devolver un valor en el que no se puede confiar."""
    prosa = evidencia_en_prosa(item["verdicts"])
    t0 = time.perf_counter()
    flags, detalle, tokens = {}, {}, 0
    for pred in POLICY_PREDICATES:
        ganadores, gaps = [], []
        for shift in (0, 1):
            rot = pred["options"][shift:] + pred["options"][:shift]
            row = {"id": pred["key"], "state": prosa,
                   "question": pred["question"], "options": rot}
            r = text_score(tx_model, tx_tok, row, tx_meta)
            pares = dict(zip(r["option_ids"], r["option_logits"]))
            ganadores.append(max(pares, key=pares.get))
            gaps.append(abs(pares["si"] - pares["no"]))
            tokens = r["input_tokens"]
        estable = len(set(ganadores)) == 1
        flags[pred["key"]] = estable and ganadores[0] == "si"
        detalle[pred["key"]] = {"estable": estable, "voto": ganadores,
                                "gap_nats": round(sum(gaps) / 2, 1)}
    return {**item, "accion": decidir(flags), "flags": flags, "detalle": detalle,
            "cpu_seconds": time.perf_counter() - t0, "cpu_tokens": tokens}


# --------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--secuencial", action="store_true")
    ap.add_argument("--vl-model", default=semif_vl.DEFAULT_MODEL)
    ap.add_argument("--cpu-model", default="Qwen/Qwen3-0.6B")
    # en CPU hibridos (P+E cores) conviene ceñirse a los P-cores: ver bench_cpu4b.py
    ap.add_argument("--cpu-threads", type=int, default=8)
    args = ap.parse_args()

    images = sorted(os.path.join(_ROOT, "data", "img", f)
                    for f in os.listdir(os.path.join(_ROOT, "data", "img"))
                    if f.endswith((".jpg", ".png")))

    torch.set_num_threads(args.cpu_threads)
    print(f"[cfg] {len(images)} imagenes | {len(VISUAL_PREDICATES)} predicados visuales "
          f"| hilos CPU={args.cpu_threads}/{os.cpu_count()}")

    print(f"[gpu] cargando {args.vl_model}...")
    vl_model, vl_proc, vl_meta = semif_vl.load_vl_model(args.vl_model)
    print(f"[gpu] {vl_meta['load_seconds']}s, VRAM {torch.cuda.memory_allocated()/2**30:.1f} GiB")

    print(f"[cpu] cargando {args.cpu_model} (fp32)...")
    t0 = time.time()
    tx_model, tx_tok, tx_meta = load_cpu(args.cpu_model)[:3]
    print(f"[cpu] {time.time()-t0:.1f}s, RAM ~{sum(p.numel() for p in tx_model.parameters())*4/2**30:.1f} GiB")

    results = []
    wall = time.perf_counter()

    if args.secuencial:
        for img in images:
            results.append(cpu_stage(tx_model, tx_tok, tx_meta,
                                     gpu_stage(vl_model, vl_proc, vl_meta, img)))
    else:
        q: queue.Queue = queue.Queue(maxsize=2)
        gpu_busy = cpu_busy = 0.0

        fallos: list = []

        def producer():
            nonlocal gpu_busy
            try:
                for img in images:
                    t = time.perf_counter()
                    item = gpu_stage(vl_model, vl_proc, vl_meta, img)
                    gpu_busy += time.perf_counter() - t
                    q.put(item)
            except BaseException as e:          # noqa: BLE001
                fallos.append(e)
            finally:
                q.put(None)                     # el centinela SIEMPRE se pone:
                                                # sin esto una excepcion aqui deja
                                                # al consumidor colgado en q.get()

        def consumer():
            nonlocal cpu_busy
            try:
                while True:
                    item = q.get()
                    if item is None:
                        break
                    t = time.perf_counter()
                    results.append(cpu_stage(tx_model, tx_tok, tx_meta, item))
                    cpu_busy += time.perf_counter() - t
            except BaseException as e:          # noqa: BLE001
                fallos.append(e)
                while q.get() is not None:      # vaciar para no bloquear al productor
                    pass

        p = threading.Thread(target=producer)
        c = threading.Thread(target=consumer)
        p.start(); c.start(); p.join(); c.join()
        if fallos:
            raise fallos[0]

    wall = time.perf_counter() - wall

    print(f"\n{'='*66}")
    for r in sorted(results, key=lambda x: x["image"]):
        v = "  ".join(f"{k}={d['veredicto']}({d['p']:.2f})" for k, d in r["verdicts"].items())
        print(f"{r['image']:<20} {v}")
        d = "  ".join(f"{k}={'si' if r['flags'][k] else 'no'}"
                      f"{'' if v['estable'] else '(INESTABLE)'}[{v['gap_nats']}n]"
                      for k, v in r["detalle"].items())
        print(f"{'':<20} {d}")
        print(f"{'':<20} -> {r['accion'].upper()}  "
              f"[gpu {r['gpu_seconds']*1000:.0f}ms / cpu {r['cpu_seconds']*1000:.0f}ms]")
    print("=" * 66)
    g = sum(r["gpu_seconds"] for r in results)
    c_ = sum(r["cpu_seconds"] for r in results)
    modo = "SECUENCIAL" if args.secuencial else "PIPELINE"
    print(f"{modo}   wall {wall:.2f}s   gpu_busy {g:.2f}s   cpu_busy {c_:.2f}s   "
          f"suma {g + c_:.2f}s")
    print(f"solape  {(g + c_ - wall):.2f}s ahorrados  ({(1 - wall/(g + c_))*100:.0f}% vs sumar)")
    print(f"VRAM pico {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")


if __name__ == "__main__":
    main()
