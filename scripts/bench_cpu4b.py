#!/usr/bin/env python3
"""Banco del modelo de referencia (Qwen3.5-4B, rol direct_option_logits) en CPU.

Mide tres cosas:
  1. latencia por forward segun numero de hilos (i9-12900F es 8P+8E: los E-cores
     pueden restar, hay que medirlo en vez de suponerlo)
  2. el mismo 2x2 de politica que fallo con el 0.6B (evidencia json vs prosa)
  3. RAM residente real
"""
from __future__ import annotations
import argparse, json, os, resource, sys, time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "src"))

import torch
from discern._semif.direct import score as text_score
from loaders import load_cpu
import cascade

MODEL = "Qwen/Qwen3.5-4B"


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


PROSA = {
    "cast+personas": "The automated check found a strong colour cast over the whole image, "
                     "and identifiable people appear in it.",
    "limpio+sin_personas": "The automated check found neutral colour balance and no "
                           "identifiable people in the image.",
}
JSONEV = {
    "cast+personas": json.dumps({"balance": "dominante", "personas": "con_personas"}),
    "limpio+sin_personas": json.dumps({"balance": "neutro", "personas": "sin_personas"}),
}
ESPERADO = {"cast+personas": {"si"}, "limpio+sin_personas": {"no"}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hilos", default="4,8,12,16,20")
    args = ap.parse_args()

    print(f"[load] {MODEL} fp32 CPU...")
    model, tok, meta, secs = load_cpu(MODEL)
    params = sum(p.numel() for p in model.parameters())
    print(f"[load] {secs:.0f}s | {params/1e9:.2f}B params | RSS {rss_gib():.1f} GiB")

    fila = {"id": "b", "state": PROSA["cast+personas"],
            "question": cascade.POLICY_PREDICATES[0]["question"],
            "options": cascade.POLICY_PREDICATES[0]["options"]}

    print(f"\n{'='*52}\n1) LATENCIA POR NUMERO DE HILOS")
    text_score(model, tok, fila, meta)          # warmup
    mejor = (1e9, 0)
    for h in [int(x) for x in args.hilos.split(",")]:
        torch.set_num_threads(h)
        ts = [text_score(model, tok, fila, meta)["forward_seconds"] for _ in range(3)]
        med = sorted(ts)[1]
        mejor = min(mejor, (med, h))
        print(f"  {h:>3} hilos  {med*1000:>7.0f} ms/forward")
    print(f"  -> mejor: {mejor[1]} hilos ({mejor[0]*1000:.0f} ms)")
    torch.set_num_threads(mejor[1])

    print(f"\n{'='*52}\n2) POLITICA: mismo 2x2 que fallo con el 0.6B")
    opts = cascade.POLICY_PREDICATES[1]["options"]          # consentimiento
    preg = cascade.POLICY_PREDICATES[1]["question"]
    for caso in PROSA:
        for fmt, banco in (("json", JSONEV), ("prosa", PROSA)):
            ganadores, gaps = [], []
            for shift in (0, 1):
                rot = opts[shift:] + opts[:shift]
                r = text_score(model, tok, {"id": "p", "state": banco[caso],
                                            "question": preg, "options": rot}, meta)
                pares = dict(zip(r["option_ids"], r["option_logits"]))
                ganadores.append(max(pares, key=pares.get))
                gaps.append(abs(pares["si"] - pares["no"]))
            est = len(set(ganadores)) == 1
            ok = est and set(ganadores) <= ESPERADO[caso]
            print(f"  {caso:<22} {fmt:<6} -> {'/'.join(sorted(set(ganadores))):<8} "
                  f"gap {sum(gaps)/2:>5.1f}n  {'estable' if est else 'INESTABLE'} "
                  f"{'OK' if ok else 'MAL'}")

    print(f"\n{'='*52}\nRSS pico {rss_gib():.1f} GiB")


if __name__ == "__main__":
    main()
