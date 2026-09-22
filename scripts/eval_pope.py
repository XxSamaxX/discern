#!/usr/bin/env python3
"""POPE: 9000 preguntas binarias sobre presencia de objetos (random/popular/
adversarial), 1500 si y 1500 no en cada particion.

Por que importa para este repo:
  - binario = 2 slots = el PEOR caso para la saturacion de la softmax
  - es la familia "ausencia / premisa falsa", donde nuestro probe de 22 items
    salio perfecto y sospechamos que a escala no lo sera
  - la metrica propia de POPE es el yes-rate: mide si el modelo dice "si" de mas,
    que es exactamente la alucinacion de objetos

Una sola corrida calibrada da los dos protocolos: la rotacion 0 es el orden
original (vanilla) y el acuerdo entre las dos rotaciones da el circular.
"""
from __future__ import annotations
import argparse, json, os, sys, time

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_R, "src"))
import torch
from discern import _readout as semif_vl

OPCIONES = [{"id": "yes", "description": "Yes, it is present in the image"},
            {"id": "no",  "description": "No, it is not present in the image"}]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="random,popular,adversarial")
    ap.add_argument("--limit", type=int, default=0, help="por particion; 0 = todo")
    ap.add_argument("--model", default=semif_vl.DEFAULT_MODEL)
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("lmms-lab-encoder/POPE", "Full")

    out = os.path.join(_R, "results", "pope.jsonl")
    hechos = set()
    if os.path.exists(out):
        with open(out) as fh:
            hechos = {(j["split"], j["id"]) for j in
                      (json.loads(l) for l in fh if l.strip())}
        print(f"[resumen] {len(hechos)} items ya hechos")

    model, proc, meta = semif_vl.load_vl_model(args.model)
    print(f"[gpu] {meta['load_seconds']}s, VRAM {torch.cuda.memory_allocated()/2**30:.1f} GiB")

    t0 = time.perf_counter()
    vistos = aciertos = 0
    with open(out, "a") as fh:
        for split in args.splits.split(","):
            d = ds[split]
            n = args.limit or len(d)
            print(f"\n[{split}] {n} items")
            for i in range(n):
                it = d[i]
                clave = (split, str(it["id"]))
                if clave in hechos:
                    continue
                fila = {"id": str(it["id"]), "image": it["image"], "state": "",
                        "question": str(it["question"]).strip(), "options": OPCIONES}
                try:
                    r = semif_vl.score(model, proc, fila, meta, calibrate=True)
                except Exception as e:
                    fh.write(json.dumps({"split": split, "id": str(it["id"]),
                                         "error": str(e)[:200]}) + "\n"); fh.flush()
                    continue
                verdad = it["answer"].strip().lower()
                gan = [rot["winner"] for rot in r["per_rotation"]]
                vanilla = gan[0]                      # rotacion 0 = orden original
                circular = gan[0] if len(set(gan)) == 1 else None   # None = inestable
                vistos += 1; aciertos += vanilla == verdad
                fh.write(json.dumps({
                    "split": split, "id": str(it["id"]), "verdad": verdad,
                    "vanilla": vanilla, "circular": circular,
                    "ok": vanilla == verdad,
                    "ok_circular": circular == verdad,
                    "gap_nats": r["gap_nats"], "p_ganador": max(r["probabilities"]),
                    "flip": r["winner_flips"], "tokens": r["input_tokens"],
                }) + "\n"); fh.flush()
                if vistos % 250 == 0:
                    dt = time.perf_counter() - t0
                    tot = n * len(args.splits.split(","))
                    print(f"  {vistos:>5}  acierto {aciertos/vistos*100:5.1f}%  "
                          f"{dt/vistos*1000:>5.0f} ms/item  "
                          f"eta {(tot-vistos)*dt/vistos/60:.0f} min", flush=True)

    dt = time.perf_counter() - t0
    print(f"\n{'='*56}\nvanilla {aciertos}/{vistos} = {aciertos/max(vistos,1)*100:.1f}%")
    print(f"tiempo {dt/60:.1f} min | salida {out}")


if __name__ == "__main__":
    main()
