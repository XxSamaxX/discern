#!/usr/bin/env python3
"""Figuras del paper. Salida PDF vectorial para LaTeX."""
import json, os, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(_R, "results", "gap_por_tipo.json")))
OUT = os.path.join(_R, "paper", "figures")

# paleta validada (slots categoricos 1 y 2), tokens de texto recesivos
AZUL, NARANJA = "#2a78d6", "#eb6834"
TINTA, TINTA2, GRIS = "#0b0b0b", "#52514e", "#c9c8c4"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.edgecolor": TINTA2, "axes.linewidth": 0.6,
    "xtick.color": TINTA2, "ytick.color": TINTA2,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.labelcolor": TINTA, "text.color": TINTA,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

def recesivo(ax, eje="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=eje, color=GRIS, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

# ---------------------------------------------------------------- figura 1
# Mismas decisiones, mismo forward: dos lecturas. La softmax aplana; el gap no.
P = sorted(d["perceptual"], key=lambda x: x["gap"])
x = range(len(P))
fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.4, 3.1), sharex=True,
                             gridspec_kw={"hspace": 0.28})

a1.plot(x, [p["p_ganador"] for p in P], "o", color=AZUL, markersize=3.2,
        markeredgecolor="white", markeredgewidth=0.4)
a1.set_ylim(0, 1.06); a1.set_yticks([0, 0.5, 1.0])
a1.set_ylabel("softmax $p$\nof winning option")
a1.axhline(1.0, color=GRIS, linewidth=0.5, zorder=0)
a1.annotate("every decision $\\geq 0.9993$", xy=(0.5, 0.58),
            xycoords="axes fraction", ha="center", va="center",
            fontsize=7, color=TINTA2)
recesivo(a1)

a2.plot(x, [p["gap"] for p in P], "o", color=AZUL, markersize=3.2,
        markeredgecolor="white", markeredgewidth=0.4)
a2.set_ylabel("logit gap\n(nats)")
a2.set_xlabel("perceptual decisions, ordered by gap")
a2.set_ylim(0, 29)
recesivo(a2)
fig.savefig(os.path.join(OUT, "saturation.pdf"), bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

# ---------------------------------------------------------------- figura 2
# Solapamiento entre familias. n pequeno -> se dibujan todos los puntos.
fig, ax = plt.subplots(figsize=(3.4, 1.75))
fam = [("perceptual", d["perceptual"], AZUL, "o", 1.0),
       ("normative",  d["normativa"],  NARANJA, "^", 0.0)]
for nombre, datos, color, marca, y in fam:
    g = [v["gap"] for v in datos]
    # jitter determinista para que no se solapen los puntos identicos
    ys = [y + ((i % 5) - 2) * 0.052 for i in range(len(g))]
    ax.plot(g, ys, marca, color=color, markersize=4, alpha=0.85,
            markeredgecolor="white", markeredgewidth=0.4, linestyle="none")
    med = statistics.median(g)
    ax.plot([med, med], [y - 0.21, y + 0.21], color=color, linewidth=2, solid_capstyle="butt")
    ax.annotate(f"median {med:.1f}", xy=(med, y + 0.26), ha="center",
                fontsize=6.5, color=TINTA2)


ax.axvspan(7.3, 23.1, color=GRIS, alpha=0.35, zorder=0, linewidth=0)
ax.annotate("overlap", xy=(15.2, -0.42), ha="center", fontsize=6.5, color=TINTA2)
ax.set_xlim(0, 27.4); ax.set_ylim(-0.55, 1.5)
# etiqueta directa en el eje: la identidad nunca depende solo del color
ax.set_yticks([0, 1]); ax.set_yticklabels(["normative\n(n=12)", "perceptual\n(n=22)"],
                                          fontsize=7.5, color=TINTA)
ax.tick_params(axis="y", length=0)
ax.set_xlabel("logit gap (nats)")
ax.spines["left"].set_visible(False)
recesivo(ax, eje="x")
fig.savefig(os.path.join(OUT, "families.pdf"), bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("figuras:", os.listdir(OUT))
