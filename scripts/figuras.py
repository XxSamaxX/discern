#!/usr/bin/env python3
"""Figuras del paper. PDF vectorial, dos columnas, tipografia serif.

Los numeros se leen de results/ en tiempo de render: ninguna figura puede
desviarse de lo medido.
"""
import collections, itertools, json, math, os, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_R, "paper", "figures")
os.makedirs(OUT, exist_ok=True)

AZUL, NARANJA, AGUA = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA2, GRIS = "#0b0b0b", "#52514e", "#c9c8c4"
FLOAT32_NATS = 17.33         # -ln(2**-25): por encima, float32(p) == 1.0 exacto

plt.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.edgecolor": TINTA2, "axes.linewidth": 0.6,
    "xtick.color": TINTA2, "ytick.color": TINTA2,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.labelcolor": TINTA, "text.color": TINTA,
    "legend.frameon": False, "legend.fontsize": 7,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def carga(p):
    return [j for j in (json.loads(l) for l in open(os.path.join(_R, "results", p))
            if l.strip()) if "error" not in j]


def auc(pos, neg):
    w = sum(1 for a, b in itertools.product(pos, neg) if a > b)
    e = sum(1 for a, b in itertools.product(pos, neg) if a == b)
    return (w + 0.5 * e) / (len(pos) * len(neg))


def recesivo(ax, eje="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=eje, color=GRIS, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)


V = carga("mmbench_vanilla_4329.jsonl")
C = carga("mmbench_circular_4329.jsonl")
P = carga("pope.jsonl")

# ============================================================ fig 1 (2 cols)
# El gap y la probabilidad son la MISMA senal; solo una sobrevive a float32.
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.45))

sat = np.float32(1.0)
for etiqueta, color, marca, sub in (("correct", AZUL, "o", [r for r in V if r["ok"]]),
                                    ("incorrect", NARANJA, "^", [r for r in V if not r["ok"]])):
    g = np.array([r["gap_nats"] for r in sub])
    q = np.maximum(1 - np.array([r["p_ganador"] for r in sub]), 1e-16)
    a1.scatter(g, q, s=3.5, c=color, marker=marca, alpha=0.35, linewidths=0, label=etiqueta)
a1.set_yscale("log")
a1.axhspan(1e-16, 6e-8, color=GRIS, alpha=0.45, zorder=0, linewidth=0)
a1.axhline(6e-8, color=TINTA2, linewidth=0.7, linestyle=(0, (4, 2)), zorder=1)
a1.annotate("below float32 resolution:\n$p$ is exactly 1.0 here", xy=(2.0, 3e-12),
            fontsize=6.8, color=TINTA2, va="center")
a1.set_xlabel("logit gap (nats)")
a1.set_ylabel("$1-p$  of winning option")
a1.set_xlim(-0.5, 31); a1.set_ylim(1e-15, 1.2)
a1.legend(loc="upper right", markerscale=2.2, handletextpad=0.2)
recesivo(a1)
a1.set_title("(a)  one signal, two scales", fontsize=8, color=TINTA, loc="left", pad=6)

bins = np.arange(0, 32, 1.5)
for etiqueta, color, sub in (("correct", AZUL, [r for r in V if r["ok"]]),
                             ("incorrect", NARANJA, [r for r in V if not r["ok"]])):
    a2.hist([r["gap_nats"] for r in sub], bins=bins, density=True, color=color,
            alpha=0.62, edgecolor="white", linewidth=0.4, label=etiqueta)
pct = sum(1 for r in V if np.float32(r["p_ganador"]) == sat) / len(V) * 100
a2.axvline(FLOAT32_NATS, color=TINTA2, linewidth=0.9, linestyle=(0, (4, 2)))
a2.annotate(f"{pct:.0f}% of items lie\nright of this line",
            xy=(FLOAT32_NATS + 0.7, a2.get_ylim()[1] * 0.74), fontsize=6.8, color=TINTA2)
a2.set_xlabel("logit gap (nats)"); a2.set_ylabel("density")
a2.set_xlim(0, 31)
a2.legend(loc="upper left", handletextpad=0.5)
recesivo(a2)
a2.set_title(f"(b)  gap separates at AUC {auc([r['gap_nats'] for r in V if r['ok']], [r['gap_nats'] for r in V if not r['ok']]):.3f}",
             fontsize=8, color=TINTA, loc="left", pad=6)
fig.savefig(os.path.join(OUT, "signal.pdf"), bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

# ============================================================ fig 2 (1 col)
# Curva de abstencion: que compras al negarte a responder.
fig, ax = plt.subplots(figsize=(3.4, 2.35))
series = (("MMBench, vanilla", AZUL, "-", V, "ok"),
          ("MMBench, circular", AGUA, "--", C, "ok"),
          ("POPE", NARANJA, "-.", P, "ok"))
for etiqueta, color, estilo, R, campo in series:
    xs, ys = [], []
    for t in np.arange(0, 26, 0.5):
        s = [r for r in R if r["gap_nats"] >= t]
        if len(s) < len(R) * 0.25:
            break
        xs.append(len(s) / len(R) * 100); ys.append(sum(r[campo] for r in s) / len(s) * 100)
    ax.plot(xs, ys, estilo, color=color, linewidth=1.6, label=etiqueta)
    ax.plot(xs[0], ys[0], "o", color=color, markersize=4.5,
            markeredgecolor="white", markeredgewidth=0.6)
ax.set_xlabel("coverage: items answered (%)")
ax.set_ylabel("accuracy on those (%)")
ax.invert_xaxis()
ax.legend(loc="lower right")
# el marcador del extremo izquierdo de cada linea es el punto sin abstencion;
# se explica en el pie, no con una anotacion que pisaria las curvas
recesivo(ax)
fig.savefig(os.path.join(OUT, "abstention.pdf"), bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

# ============================================================ fig 3 (1 col)
# El gap predice si el orden de las opciones decide la respuesta.
fig, ax = plt.subplots(figsize=(3.4, 2.15))
filas = (("MMBench", C, 1.0), ("POPE", P, 0.0))
for nombre, R, y in filas:
    flip = [r["gap_nats"] for r in R if r.get("flip")]
    est = [r["gap_nats"] for r in R if not r.get("flip")]
    for datos, color, off, marca in ((est, AZUL, 0.13, "o"), (flip, NARANJA, -0.13, "^")):
        if not datos:
            continue
        b = ax.boxplot([datos], positions=[y + off], orientation="horizontal", widths=0.2,
                       patch_artist=True, showfliers=False, whis=(5, 95))
        for p_ in b["boxes"]:
            p_.set(facecolor=color, alpha=0.6, edgecolor=color, linewidth=0.8)
        for k in ("whiskers", "caps", "medians"):
            for p_ in b[k]:
                p_.set(color=color, linewidth=1.0)
    ax.annotate(f"{len(flip)/len(R)*100:.1f}%\nflip", xy=(31.6, y), fontsize=6.8,
                color=TINTA2, va="center", ha="right", linespacing=1.3)
ax.set_yticks([0, 1]); ax.set_yticklabels(["POPE\n(n=9000)", "MMBench\n(n=4329)"], fontsize=7.5)
ax.tick_params(axis="y", length=0)
ax.set_xlabel("logit gap (nats)"); ax.set_xlim(-0.5, 32); ax.set_ylim(-0.45, 1.5)
ax.spines["left"].set_visible(False)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], color=AZUL, marker="s", linestyle="none", label="order-stable"),
                   Line2D([], [], color=NARANJA, marker="s", linestyle="none", label="winner flips")],
          loc="upper left", handletextpad=0.4, ncol=2, columnspacing=1.0,
          bbox_to_anchor=(0.0, 1.16))
recesivo(ax, eje="x")
fig.savefig(os.path.join(OUT, "order.pdf"), bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

# ============================================================ fig 4 (1 col)
# Perceptual frente a normativa: el solapamiento que impide usar el gap de filtro.
d = json.load(open(os.path.join(_R, "results", "gap_por_tipo.json")))
fig, ax = plt.subplots(figsize=(3.4, 1.75))
for nombre, datos, color, marca, y in (("perceptual", d["perceptual"], AZUL, "o", 1.0),
                                       ("normative", d["normativa"], NARANJA, "^", 0.0)):
    g = [v["gap"] for v in datos]
    ys = [y + ((i % 5) - 2) * 0.052 for i in range(len(g))]
    ax.plot(g, ys, marca, color=color, markersize=4, alpha=0.85,
            markeredgecolor="white", markeredgewidth=0.4, linestyle="none")
    med = statistics.median(g)
    ax.plot([med, med], [y - 0.21, y + 0.21], color=color, linewidth=2, solid_capstyle="butt")
    ax.annotate(f"median {med:.1f}", xy=(med, y + 0.26), ha="center", fontsize=6.5, color=TINTA2)
ax.axvspan(7.3, 23.1, color=GRIS, alpha=0.35, zorder=0, linewidth=0)
ax.annotate("overlap", xy=(15.2, -0.42), ha="center", fontsize=6.5, color=TINTA2)
ax.set_xlim(0, 27.4); ax.set_ylim(-0.55, 1.5)
ax.set_yticks([0, 1]); ax.set_yticklabels(["normative\n(n=12)", "perceptual\n(n=22)"], fontsize=7.5)
ax.tick_params(axis="y", length=0)
ax.set_xlabel("logit gap (nats)")
ax.spines["left"].set_visible(False)
recesivo(ax, eje="x")
fig.savefig(os.path.join(OUT, "families.pdf"), bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

print("figuras:", sorted(f for f in os.listdir(OUT) if f.endswith(".pdf")))
print(f"float32 colapsa en {pct:.1f}% de MMBench")
