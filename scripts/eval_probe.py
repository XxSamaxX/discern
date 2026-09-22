#!/usr/bin/env python3
"""Corre el probe hecho a mano (data/vl_rows*.json) y reporta acierto y gaps.

Vive en scripts/ y no dentro del paquete: sus rutas son relativas al repo, que
es una suposicion valida para un script de investigacion e invalida para una
libreria instalada en site-packages.
"""
import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import torch
from discern._readout import DEFAULT_MODEL, load_vl_model, score
from discern._semif.core import LETTERS

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
