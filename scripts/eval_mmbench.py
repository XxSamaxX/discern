#!/usr/bin/env python3
"""Evaluacion del readout de logits sobre MMBench (en/dev, 4329 items).

MMBench ya viene como question + A/B/C/D + answer, o sea el formato nativo de
este mecanismo. Su protocolo oficial, CircularEval, exige acertar bajo TODAS las
rotaciones de las opciones: es el mismo control de sesgo posicional que usamos,
pero como estandar comparable con la literatura.

Lo que mide, y que con un benchmark de 22 items no se podia medir:
  - VanillaEval (un forward) frente a CircularEval (n forwards, todas aciertan)
  - si el gap en logits PREDICE el acierto (AUC), que es la afirmacion central
  - saturacion de la softmax sobre miles de items, no sobre 22

Escribe JSONL incremental, asi que se puede interrumpir y reanudar.
"""
from __future__ import annotations
import argparse, json, os, sys, time

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_R, "src"))
import torch
from discern import _readout as semif_vl
from discern._semif.core import LETTERS

NO_OPCION = ("", "nan", "none", "null")


def opciones_reales(fila: dict) -> list[dict]:
    """Las opciones no usadas vienen como la cadena literal 'nan', no vacias."""
    out = []
    for k in "ABCD":
        v = fila.get(k)
        if v is not None and str(v).strip().lower() not in NO_OPCION:
            out.append({"id": k, "description": str(v).strip()})
    return out


def a_fila(item: dict, idx: int) -> dict | None:
    ops = opciones_reales(item)
    if len(ops) < 2:
        return None
    hint = item.get("hint")
    estado = str(hint).strip() if hint and str(hint).strip().lower() not in NO_OPCION else ""
    return {"id": str(item.get("index", idx)), "image": item["image"],
            "state": estado, "question": str(item["question"]).strip(),
            "options": ops, "truth": item["answer"], "category": item.get("category")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = todo el split")
    ap.add_argument("--circular", action="store_true", help="CircularEval (n forwards)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default=semif_vl.DEFAULT_MODEL)
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("lmms-lab-encoder/MMBench", "en", split="dev")
    total = args.limit or len(ds)

    sufijo = "circular" if args.circular else "vanilla"
    out = args.out or os.path.join(_R, "results", f"mmbench_{sufijo}_{total}.jsonl")
    hechos = set()
    if os.path.exists(out):
        with open(out) as fh:
            hechos = {json.loads(l)["id"] for l in fh if l.strip()}
        print(f"[resumen] {len(hechos)} items ya hechos en {os.path.basename(out)}")

    model, proc, meta = semif_vl.load_vl_model(args.model)
    print(f"[gpu] {meta['load_seconds']}s, VRAM {torch.cuda.memory_allocated()/2**30:.1f} GiB")
    print(f"[eval] {total} items, modo {sufijo}")

    aciertos = vistos = 0
    t0 = time.perf_counter()
    with open(out, "a") as fh:
        for i in range(total):
            fila = a_fila(ds[i], i)
            if fila is None or fila["id"] in hechos:
                continue
            try:
                r = semif_vl.score(model, proc, fila, meta, calibrate=args.circular)
            except Exception as e:                      # un item malo no tumba la corrida
                fh.write(json.dumps({"id": fila["id"], "error": str(e)[:200]}) + "\n")
                fh.flush()
                continue
            pred = max(zip(r["option_ids"], r["probabilities"]), key=lambda x: x[1])[0]
            # CircularEval: solo cuenta si acierta en TODAS las rotaciones
            if args.circular:
                ok = all(rot["winner"] == fila["truth"] for rot in r["per_rotation"])
            else:
                ok = pred == fila["truth"]
            vistos += 1; aciertos += ok
            fh.write(json.dumps({
                "id": fila["id"], "categoria": fila["category"],
                "n_opciones": len(fila["options"]), "verdad": fila["truth"],
                "prediccion": pred, "ok": bool(ok),
                "gap_nats": r["gap_nats"], "p_ganador": max(r["probabilities"]),
                "flip": r["winner_flips"], "tokens": r["input_tokens"],
                "ganadores": [rot["winner"] for rot in r["per_rotation"]],
            }) + "\n")
            fh.flush()
            if vistos % 100 == 0:
                dt = time.perf_counter() - t0
                print(f"  {vistos:>5}/{total}  acierto {aciertos/vistos*100:5.1f}%  "
                      f"{dt/vistos*1000:>6.0f} ms/item  eta {(total-vistos)*dt/vistos/60:.0f} min",
                      flush=True)

    dt = time.perf_counter() - t0
    print(f"\n{'='*56}")
    print(f"{sufijo.upper():<9} {aciertos}/{vistos} = {aciertos/max(vistos,1)*100:.1f}%")
    print(f"tiempo    {dt/60:.1f} min ({dt/max(vistos,1)*1000:.0f} ms/item)")
    print(f"salida    {out}")


if __name__ == "__main__":
    main()
