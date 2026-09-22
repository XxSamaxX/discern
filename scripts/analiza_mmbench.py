#!/usr/bin/env python3
"""Analisis de las corridas de MMBench: acierto, saturacion, y sobre todo
si el gap en logits predice el acierto (AUC) y a que coste de cobertura.
"""
import argparse, collections, itertools, json, math, os, statistics, sys

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cargar(p):
    return [r for r in (json.loads(l) for l in open(p) if l.strip()) if "error" not in r]


def auc_mw(pos, neg):
    """AUC = P(pos > neg) = Mann-Whitney U / mn, con p-valor por aprox. normal."""
    if not pos or not neg:
        return None
    pares = len(pos) * len(neg)
    wins = sum(1 for a, b in itertools.product(pos, neg) if a > b)
    empates = sum(1 for a, b in itertools.product(pos, neg) if a == b)
    auc = (wins + 0.5 * empates) / pares
    m, n = len(pos), len(neg)
    z = (auc * m * n - m * n / 2) / ((m * n * (m + n + 1) / 12) ** 0.5)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / 2 ** 0.5)))
    return auc, z, p


def informe(nombre, R):
    ok = [r["gap_nats"] for r in R if r["ok"]]
    mal = [r["gap_nats"] for r in R if not r["ok"]]
    print(f"\n{'='*62}\n{nombre}  —  {len(R)} items")
    print(f"  acierto: {len(ok)}/{len(R)} = {len(ok)/len(R)*100:.1f}%")
    if not mal:
        print("  (sin fallos: no se puede medir poder predictivo)")
        return
    print(f"  gap mediano  acierta {statistics.median(ok):5.1f}n   falla {statistics.median(mal):5.1f}n")
    a = auc_mw(ok, mal)
    print(f"  AUC gap->acierto: {a[0]:.3f}  (z={a[1]:.1f}, p={a[2]:.1e})")

    ps = [r["p_ganador"] for r in R]
    sat = sum(x >= 0.99 for x in ps)
    sat_mal = sum(1 for r in R if not r["ok"] and r["p_ganador"] >= 0.99)
    print(f"\n  softmax: minima {min(ps):.4f} | mediana {statistics.median(ps):.4f}")
    print(f"    p>=0.99 en {sat}/{len(R)} ({sat/len(R)*100:.0f}%) de todos los items")
    print(f"    p>=0.99 en {sat_mal}/{len(mal)} ({sat_mal/len(mal)*100:.0f}%) de los ERRORES"
          f"  <- por esto no sirve de umbral")

    print(f"\n  abstencion por umbral de gap:")
    print(f"    {'umbral':>7} {'cobertura':>10} {'acierto':>9} {'errores restantes':>18}")
    for t in (0, 3, 5, 8, 12, 16, 20):
        resp = [r for r in R if r["gap_nats"] >= t]
        if not resp:
            continue
        acc = sum(r["ok"] for r in resp) / len(resp)
        err = sum(1 for r in resp if not r["ok"])
        print(f"    {t:>6}n {len(resp)/len(R)*100:>9.0f}% {acc*100:>8.1f}% {err:>13} de {len(mal)}")

    print(f"\n  peores categorias (min 25 items):")
    porcat = collections.defaultdict(list)
    for r in R:
        porcat[r.get("categoria") or "?"].append(r["ok"])
    filas = [(sum(v) / len(v), k, len(v)) for k, v in porcat.items() if len(v) >= 25]
    for acc, k, n in sorted(filas)[:6]:
        print(f"    {acc*100:5.1f}%  {k:<38} (n={n})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ficheros", nargs="+")
    args = ap.parse_args()
    datos = {}
    for f in args.ficheros:
        R = cargar(f)
        datos[os.path.basename(f)] = R
        informe(os.path.basename(f), R)

    # si hay vanilla y circular del mismo tamano, comparar item a item
    v = next((R for n, R in datos.items() if "vanilla" in n), None)
    c = next((R for n, R in datos.items() if "circular" in n), None)
    if v and c:
        iv = {r["id"]: r for r in v}
        ic = {r["id"]: r for r in c}
        comun = set(iv) & set(ic)
        if comun:
            print(f"\n{'='*62}\nVANILLA vs CIRCULAR  ({len(comun)} items en comun)")
            av = sum(iv[i]["ok"] for i in comun) / len(comun)
            ac = sum(ic[i]["ok"] for i in comun) / len(comun)
            flips = sum(1 for i in comun if ic[i].get("flip"))
            print(f"  vanilla  {av*100:.1f}%")
            print(f"  circular {ac*100:.1f}%   (caida {(av-ac)*100:.1f} pp)")
            print(f"  el ganador cambia al rotar en {flips}/{len(comun)} "
                  f"({flips/len(comun)*100:.1f}%)")
            gf = [ic[i]["gap_nats"] for i in comun if ic[i].get("flip")]
            gn = [ic[i]["gap_nats"] for i in comun if not ic[i].get("flip")]
            if gf and gn:
                print(f"  gap mediano: con flip {statistics.median(gf):.1f}n, "
                      f"sin flip {statistics.median(gn):.1f}n")
                a = auc_mw(gn, gf)
                print(f"  AUC gap->estabilidad: {a[0]:.3f} (p={a[2]:.1e})")


if __name__ == "__main__":
    main()
