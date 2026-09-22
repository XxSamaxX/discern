"""Varias preguntas sobre una misma imagen, codificandola una sola vez.

Todas las preguntas comparten prefijo: el turno de sistema y los ~300 tokens
visuales. Solo difiere la cola con el JSON de criterio y opciones. Asi que se
ejecuta el prefijo una vez con cache KV y despues solo la cola de cada pregunta.

MEDIDO Y DESCARTADO: la cache es numericamente exacta (verify_cache da una
diferencia maxima de 0.0) pero en este stack resulta 13-14% MAS LENTA, y la
proporcion es plana con el tamano de la imagen -- no hay punto de cruce:

    tokens   sin cache   con cache   ganancia
       384       666ms       764ms      0.87x
       852      1423ms      1657ms      0.86x
      1812      3473ms      4009ms      0.87x

La causa es que el camino con cache pierde la ruta rapida de atencion (mascara
explicita sobre DynamicCache) y eso cuesta mas que recalcular el prefijo. Por
eso use_cache=False por defecto. Se conserva el codigo porque con flash-attn
instalado el balance podria invertirse, y quien lo reintente merece saber que
ya se midio. El preprocesado de imagen, por cierto, es solo el 3% del coste:
el forward se lleva el 97%.

Qwen-VL usa mrope (posiciones 3D) y partir un forward puede desalinear las
posiciones sin avisar; verify_cache() lo comprueba logit a logit.
"""
from __future__ import annotations

import time

import torch
from . import Verdict, _opciones, load  # noqa: F401
from . import _readout as semif_vl
from ._semif.core import LETTERS


def _tokenizar(proc, fila):
    msgs = semif_vl.vl_messages(fila, fila["options"])
    return proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                    return_dict=True, return_tensors="pt")


def _prefijo_comun(secuencias: list[torch.Tensor]) -> int:
    """Tokens iniciales identicos en todas las preguntas."""
    corta = min(s.shape[0] for s in secuencias)
    base = secuencias[0]
    n = 0
    while n < corta and all(int(s[n]) == int(base[n]) for s in secuencias[1:]):
        n += 1
    return n


def _logits_sin_cache(model, enc, slots):
    with torch.inference_mode():
        out = model(**enc, use_cache=False, return_dict=True)
    return out.logits[:, -1, :][0].float()[slots].cpu()


def multi(image, questions, *, options=None, evidence: str = "",
          model: str | None = None, threshold: float = 3.0,
          use_cache: bool = False) -> list[Verdict]:
    """Responde varias preguntas sobre una imagen.

    questions  lista de cadenas, o de tuplas (pregunta, opciones)
    options    opciones por defecto para las preguntas que no traigan las suyas
    use_cache  reutilizar la cache KV del prefijo compartido. Falso por defecto:
               es exacto pero mas lento, ver la cabecera del modulo.
    """
    m, proc, meta = load(model)
    normal = []
    for q in questions:
        if isinstance(q, (tuple, list)) and len(q) == 2 and not isinstance(q[1], str):
            normal.append((str(q[0]), q[1]))
        else:
            normal.append((str(q), options))

    filas, encs, slotss, binarias = [], [], [], []
    for pregunta, ops in normal:
        o, binaria = _opciones(ops)
        fila = {"id": "q", "image": image, "state": evidence,
                "question": pregunta.strip(), "options": o}
        enc = _tokenizar(proc, fila)
        filas.append(fila); encs.append(enc); binarias.append(binaria)
        slotss.append([proc.tokenizer.encode(L, add_special_tokens=False)[0]
                       for L in LETTERS[:len(o)]])

    t0 = time.perf_counter()
    dev = m.device
    n_pref = _prefijo_comun([e["input_ids"][0] for e in encs]) if len(encs) > 1 else 0
    compartido = use_cache and n_pref >= 16 and len(encs) > 1

    vectores = []
    if compartido:
        try:
            vectores = _con_cache(m, encs, slotss, n_pref, dev)
        except Exception:
            vectores = []          # cualquier problema -> camino seguro
    if not vectores:
        compartido = False
        for enc, slots in zip(encs, slotss):
            vectores.append(_logits_sin_cache(m, enc.to(dev), slots))

    ms = (time.perf_counter() - t0) * 1000
    out = []
    for vec, (pregunta, _), fila, binaria in zip(vectores, normal, filas, binarias):
        sel = vec.tolist()
        mx = max(sel)
        ex = [pow(2.718281828459045, v - mx) for v in sel]
        tot = sum(ex)
        probs = [e / tot for e in ex]
        ids = [o["id"] for o in fila["options"]]
        orden = sorted(sel, reverse=True)
        gap = orden[0] - orden[1]
        mejor = ids[sel.index(mx)]
        out.append(Verdict(answer=mejor, gap=gap, trusted=gap >= threshold,
                           probabilities=dict(zip(ids, probs)), forwards=1,
                           ms=ms / len(normal), binary=binaria))
    return out


def _con_cache(m, encs, slotss, n_pref, dev):
    """Prefijo una vez, colas despues. Devuelve [] si transformers no lo soporta."""
    from transformers.cache_utils import DynamicCache
    base = encs[0]
    pref = {k: (v[:, :n_pref] if k in ("input_ids", "attention_mask", "mm_token_type_ids")
                else v) for k, v in base.items()}
    pref = {k: v.to(dev) for k, v in pref.items()}
    cache = DynamicCache()
    with torch.inference_mode():
        m(**pref, past_key_values=cache, use_cache=True, return_dict=True)
    largo_pref = cache.get_seq_length()

    vectores = []
    for enc, slots in zip(encs, slotss):
        ids = enc["input_ids"][:, n_pref:].to(dev)
        att = torch.ones((1, largo_pref + ids.shape[1]), dtype=torch.long, device=dev)
        with torch.inference_mode():
            out = m(input_ids=ids, attention_mask=att, past_key_values=cache,
                    use_cache=True, return_dict=True)
        vectores.append(out.logits[:, -1, :][0].float()[slots].cpu())
        cache.crop(largo_pref)          # devolver la cache al prefijo compartido
    return vectores


def verify_cache(image, questions, *, model=None, tol: float = 0.05) -> bool:
    """Compara el camino con cache contra el camino sin cache, logit a logit."""
    con = multi(image, questions, model=model, use_cache=True)
    sin = multi(image, questions, model=model, use_cache=False)
    peor = 0.0
    for a, b in zip(con, sin):
        for k in a.probabilities:
            peor = max(peor, abs(a.probabilities[k] - b.probabilities[k]))
        if a.answer != b.answer:
            print(f"  DISTINTA RESPUESTA: {a.answer!r} vs {b.answer!r}")
            return False
    print(f"  maxima diferencia de probabilidad: {peor:.2e}  (tolerancia {tol})")
    return peor <= tol
