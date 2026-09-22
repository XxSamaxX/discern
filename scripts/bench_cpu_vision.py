#!/usr/bin/env python3
"""Cuanto cuesta el readout de vision en CPU. Sin esto no se puede prometer
una via CPU: el forward es el 97% del coste e incluye el ViT sobre los tokens
visuales, que es justo lo caro.
"""
import os, sys, time, resource
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_R, "src"))
import torch
from discern._readout import load_vl_model, score

MODELO = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-VL-2B-Instruct"
IMG = os.path.join(_R, "data", "img", "gatos_sofa.jpg")
OPS = [{"id": "yes", "description": "Yes"}, {"id": "no", "description": "No"}]
fila = {"id": "q", "image": IMG, "state": "",
        "question": "Is there a dog in the image?", "options": OPS}

print(f"[cpu] {MODELO} float32...")
t0 = time.time()
m, proc, meta = load_vl_model(MODELO, device="cpu", dtype=torch.float32)
print(f"[cpu] cargado en {time.time()-t0:.0f}s | RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.1f} GiB")
print(f"[cpu] hilos: {torch.get_num_threads()} de {os.cpu_count()}")

print("\n  barrido de hilos (3 medidas, mediana):")
mejor = (1e9, 0)
score(m, proc, fila, meta)                       # warmup
for h in (4, 8, 12, 16, 24):
    torch.set_num_threads(h)
    ts = sorted(score(m, proc, fila, meta)["forward_seconds"] for _ in range(3))
    mejor = min(mejor, (ts[1], h))
    print(f"    {h:>3} hilos  {ts[1]*1000:>8.0f} ms/decision")
print(f"    -> mejor: {mejor[1]} hilos, {mejor[0]*1000:.0f} ms")

torch.set_num_threads(mejor[1])
print("\n  con imagenes de distinto tamano:")
from PIL import Image
base = Image.open(IMG)
for lado in (448, 640, 1024):
    im = base.resize((lado, int(lado*base.height/base.width)), Image.LANCZOS)
    f2 = dict(fila, image=im)
    r = score(m, proc, f2, meta)
    print(f"    {lado:>5}px -> {r['input_tokens']:>5} tokens, {r['forward_seconds']*1000:>8.0f} ms")
