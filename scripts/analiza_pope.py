#!/usr/bin/env python3
"""Analisis de POPE. Ademas de acierto, las metricas propias del benchmark:
F1 y yes-rate (un modelo que alucina objetos dice "si" de mas), y lo que
interesa a este repo: si el gap sigue prediciendo el acierto cuando solo hay
dos slots, que es el peor caso para la saturacion de la softmax.
"""
import collections, itertools, json, math, os, statistics, sys

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FICH = sys.argv[1] if len(sys.argv) > 1 else "pope_qwen3-vl-4b-instruct.jsonl"
_RUTA = _FICH if os.path.isabs(_FICH) else os.path.join(_R, "results", _FICH)
R = [j for j in (json.loads(l) for l in open(_RUTA) if l.strip()) if "error" not in j]


def auc_mw(pos, neg):
    if not pos or not neg:
        return None
    w = sum(1 for a, b in itertools.product(pos, neg) if a > b)
    e = sum(1 for a, b in itertools.product(pos, neg) if a == b)
    auc = (w + 0.5 * e) / (len(pos) * len(neg))
    m, n = len(pos), len(neg)
    z = (auc * m * n - m * n / 2) / ((m * n * (m + n + 1) / 12) ** 0.5)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / 2 ** 0.5)))
    return auc, z, p


def metricas(rs):
    tp = sum(1 for r in rs if r["vanilla"] == "yes" and r["verdad"] == "yes")
    fp = sum(1 for r in rs if r["vanilla"] == "yes" and r["verdad"] == "no")
    fn = sum(1 for r in rs if r["vanilla"] == "no" and r["verdad"] == "yes")
    acc = sum(r["ok"] for r in rs) / len(rs)
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    yes = sum(1 for r in rs if r["vanilla"] == "yes") / len(rs)
    return acc, prec, rec, f1, yes


print(f"POPE — {os.path.basename(_RUTA)} — {len(R)} items\n")
print(f"{'particion':<14}{'n':>6}{'acc':>8}{'prec':>8}{'rec':>8}{'F1':>8}{'yes-rate':>10}")
print("-" * 62)
por = collections.defaultdict(list)
for r in R:
    por[r["split"]].append(r)
for sp in ("random", "popular", "adversarial"):
    if sp in por:
        a, p_, rc, f1, y = metricas(por[sp])
        print(f"{sp:<14}{len(por[sp]):>6}{a*100:>7.1f}%{p_*100:>7.1f}%"
              f"{rc*100:>7.1f}%{f1*100:>7.1f}%{y*100:>9.1f}%")
a, p_, rc, f1, y = metricas(R)
print("-" * 62)
print(f"{'TOTAL':<14}{len(R):>6}{a*100:>7.1f}%{p_*100:>7.1f}%{rc*100:>7.1f}%"
      f"{f1*100:>7.1f}%{y*100:>9.1f}%")
print(f"\n(yes-rate real del dataset: 50.0% — por encima = alucina presencia)")

ok = [r["gap_nats"] for r in R if r["ok"]]
mal = [r["gap_nats"] for r in R if not r["ok"]]
print(f"\n{'='*62}\nEL GAP CON SOLO DOS SLOTS")
print(f"  gap mediano: acierta {statistics.median(ok):.1f}n | falla {statistics.median(mal):.1f}n")
A = auc_mw(ok, mal)
print(f"  AUC gap->acierto: {A[0]:.3f}  (z={A[1]:.1f}, p={A[2]:.1e})")

ps = [r["p_ganador"] for r in R]
sat_mal = sum(1 for r in R if not r["ok"] and r["p_ganador"] >= 0.99)
print(f"\n  softmax: minima {min(ps):.4f} | p>=0.99 en {sum(x>=0.99 for x in ps)/len(ps)*100:.0f}% de items")
print(f"    y en {sat_mal}/{len(mal)} ({sat_mal/len(mal)*100:.0f}%) de los ERRORES")

print(f"\n  abstencion:")
print(f"    {'umbral':>7}{'cobertura':>11}{'acierto':>9}{'F1':>8}")
for t in (0, 3, 5, 8, 12, 16, 20):
    resp = [r for r in R if r["gap_nats"] >= t]
    if len(resp) < 20:
        continue
    aa, _, _, ff, _ = metricas(resp)
    print(f"    {t:>6}n{len(resp)/len(R)*100:>10.0f}%{aa*100:>8.1f}%{ff*100:>7.1f}%")

flips = sum(1 for r in R if r["flip"])
print(f"\n{'='*62}\nSENSIBILIDAD AL ORDEN")
print(f"  el ganador cambia al rotar: {flips}/{len(R)} ({flips/len(R)*100:.1f}%)")
if flips:
    gf = [r["gap_nats"] for r in R if r["flip"]]
    gn = [r["gap_nats"] for r in R if not r["flip"]]
    print(f"  gap mediano: con flip {statistics.median(gf):.1f}n | sin flip {statistics.median(gn):.1f}n")
    B = auc_mw(gn, gf)
    print(f"  AUC gap->estabilidad: {B[0]:.3f} (p={B[2]:.1e})")
acc_c = sum(1 for r in R if r["ok_circular"]) / len(R)
print(f"\n  vanilla {a*100:.1f}%  ->  circular {acc_c*100:.1f}%  (caida {(a-acc_c)*100:.1f} pp)")
