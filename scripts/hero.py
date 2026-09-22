#!/usr/bin/env python3
"""Figura de portada del README: foto -> predicado en texto libre -> respuesta + gap.

Los gaps se leen de results/gap_por_tipo.json, nunca se escriben a mano, para
que la figura no pueda desviarse de los datos medidos.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = {x["id"]: x["gap"] for x in json.load(
     open(os.path.join(_R, "results", "gap_por_tipo.json")))["perceptual"]}

AZUL, TINTA, TINTA2, GRIS, SUP = "#2a78d6", "#0b0b0b", "#52514e", "#d8d7d3", "#f4f4f2"
MAXGAP = 27.0

PANELES = [
    ("gatos_sofa.jpg", [
        ("How many cats are in the image?",            "two",               "H_contar_gatos"),
        ("Is there a dog in the image?",               "no",                "H_perro_ausente"),
        ("What are the animals doing?",                "resting",           "actividad_gatos"),
        ("Where is the narrow remote, "
         "relative to the two cats?",                  "between them",      "H_mando_espacial"),
        ("Are the two remotes the same model?",        "different models",  "H_mandos_iguales"),
    ]),
    ("tenis_ninos.jpg", [
        ("Is this photograph posed or candid?",        "posed",             "posado_tenis"),
        ("Is someone holding a trophy?",               "yes",               "trofeo_tenis"),
        ("What is the camera viewpoint?",              "eye level",         "vista_tenis"),
        ("What brand is printed on the white cap "
         "of the boy on the far right?",               "Adidas",            "H_logo_gorra"),
    ]),
]

FIG_W, FIG_H = 11.5, 7.0
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=130)
fig.patch.set_facecolor("white")
plt.rcParams["font.family"] = "DejaVu Sans"

fig.text(0.5, 0.965, "One forward pass. No decoding. No training set.",
         ha="center", fontsize=15, color=TINTA, weight="bold")
fig.text(0.5, 0.928,
         "Free-text predicates read straight from Qwen3-VL-4B's answer-token logits "
         "— ~120 ms each on one RTX 3080",
         ha="center", fontsize=9.5, color=TINTA2)

ALTO, IM_W = 0.375, 0.255
TOPE, PIE = 0.895, 0.075          # banda util entre subtitulo y pie
for idx, (imagen, filas) in enumerate(PANELES):
    y0 = TOPE - ALTO - idx * (ALTO + (TOPE - PIE - 2 * ALTO))

    im = Image.open(os.path.join(_R, "data", "img", imagen))
    # la caja de la foto se calcula para respetar su relacion de aspecto,
    # si no el marco no cine la imagen
    im_h = (IM_W * FIG_W) * (im.height / im.width) / FIG_H
    ax_im = fig.add_axes([0.035, y0 + (ALTO - im_h) / 2, IM_W, im_h])
    ax_im.imshow(im)
    ax_im.set_xticks([]); ax_im.set_yticks([])
    for s in ax_im.spines.values():
        s.set_edgecolor(GRIS); s.set_linewidth(1)

    ax = fig.add_axes([0.315, y0, 0.655, ALTO])
    ax.set_xlim(0, 1); ax.set_ylim(0, len(filas)); ax.axis("off")

    for i, (pregunta, respuesta, clave) in enumerate(filas):
        y = len(filas) - i - 0.5
        gap = G[clave]
        ax.add_patch(FancyBboxPatch((0, y - 0.42), 1, 0.84,
                     boxstyle="round,pad=0,rounding_size=0.012",
                     facecolor=SUP, edgecolor="none",
                     transform=ax.transData, zorder=0))
        ax.text(0.012, y + 0.12, f'"{pregunta}"', fontsize=8.6, color=TINTA2,
                va="center", style="italic")
        ax.text(0.012, y - 0.20, respuesta, fontsize=10.2, color=TINTA,
                va="center", weight="bold")
        # barra del gap: anclada a una linea base comun, extremo redondeado
        x0, ancho = 0.685, 0.24
        ax.plot([x0, x0 + ancho], [y, y], color=GRIS, linewidth=3,
                solid_capstyle="round", zorder=1)
        ax.plot([x0, x0 + ancho * gap / MAXGAP], [y, y], color=AZUL, linewidth=3,
                solid_capstyle="round", zorder=2)
        ax.text(x0 + ancho + 0.016, y, f"{gap:.1f} nats", fontsize=8.4,
                color=TINTA2, va="center")

fig.text(0.315, 0.022,
         "The bar is the logit gap — the real confidence signal. "
         "The softmax probability behind every one of these is $\\geq$ 0.9993.",
         fontsize=8.8, color=TINTA2)

out = os.path.join(_R, "docs", "hero.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight", pad_inches=0.12, facecolor="white")
print("escrito:", out)
