#!/usr/bin/env python3
"""Experimento central: el mismo modelo y el mismo readout sobre dos familias
de pregunta. Perceptual (que hay en la imagen) frente a normativa (que debe
hacer la organizacion). Si el gap separa las familias, el gap sirve como
criterio para decidir que se delega a un semantic if y que se queda en codigo.
"""
import json, os, statistics, sys
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HERE, "src"))
import torch, semif_vl
from semif_phase1.core import LETTERS

# --- normativas: NO dependen de la imagen, solo de una politica institucional
NORMATIVAS = [
 {"id":"N_consentimiento","evidence":"Identifiable people appear in the photograph.",
  "question":"Must publication be held until personal consent is obtained?",
  "options":[{"id":"si","description":"Yes, hold for consent"},{"id":"no","description":"No, publish without consent"}]},
 {"id":"N_correccion","evidence":"The photograph has a strong overall colour cast.",
  "question":"Should this photograph be sent to the colour-correction queue?",
  "options":[{"id":"si","description":"Yes, send for correction"},{"id":"no","description":"No, acceptable as is"}]},
 {"id":"N_archivo","evidence":"The photograph is three years old and was taken at a company event.",
  "question":"Should this photograph be moved to cold archive under the retention policy?",
  "options":[{"id":"si","description":"Yes, archive it"},{"id":"no","description":"No, keep it active"}]},
 {"id":"N_licencia","evidence":"The photograph was taken by a contractor during a paid assignment.",
  "question":"Does the organisation hold commercial redistribution rights over this photograph?",
  "options":[{"id":"si","description":"Yes, rights are held"},{"id":"no","description":"No, rights are not held"}]},
 {"id":"N_retencion","evidence":"The photograph was uploaded by an employee who has since left the company.",
  "question":"Does the retention policy require deleting this photograph?",
  "options":[{"id":"si","description":"Yes, deletion is required"},{"id":"no","description":"No, deletion is not required"}]},
 {"id":"N_menores","evidence":"The photograph shows a group of children at a sports event.",
  "question":"Does the organisation's policy permit publishing this photograph on social media?",
  "options":[{"id":"si","description":"Yes, publishing is permitted"},{"id":"no","description":"No, publishing is not permitted"}]},
 {"id":"N_presupuesto","evidence":"The photograph was selected for the annual report.",
  "question":"Does using this photograph require a payment from the marketing budget?",
  "options":[{"id":"si","description":"Yes, a payment is required"},{"id":"no","description":"No payment is required"}]},
 {"id":"N_revision","evidence":"The photograph was taken at an internal team meeting.",
  "question":"Must this photograph be reviewed by the legal department before use?",
  "options":[{"id":"si","description":"Yes, legal review is required"},{"id":"no","description":"No legal review is required"}]},
 {"id":"N_marca","evidence":"The photograph shows a person wearing branded sportswear.",
  "question":"Does this photograph violate the organisation's brand guidelines?",
  "options":[{"id":"si","description":"Yes, it violates the guidelines"},{"id":"no","description":"No violation"}]},
 {"id":"N_acceso","evidence":"The photograph is stored in the shared marketing folder.",
  "question":"Are external contractors authorised to access this photograph?",
  "options":[{"id":"si","description":"Yes, they are authorised"},{"id":"no","description":"No, they are not authorised"}]},
 {"id":"N_prioridad","evidence":"The photograph is one of two hundred submitted for this campaign.",
  "question":"Should this photograph be processed before the others in the queue?",
  "options":[{"id":"si","description":"Yes, process it first"},{"id":"no","description":"No, normal order"}]},
 {"id":"N_caducidad","evidence":"The model release form for this photograph was signed in 2019.",
  "question":"Has the model release for this photograph expired?",
  "options":[{"id":"si","description":"Yes, it has expired"},{"id":"no","description":"No, it is still valid"}]},
]

def texto_solo(model, proc, row):
    """Mismo readout que semif_vl pero sin imagen."""
    payload = {"evidence": row["evidence"], "criterion": row["question"],
               "options":[{"letter":LETTERS[i],"description":o["description"]}
                          for i,o in enumerate(row["options"])]}
    msgs=[{"role":"system","content":[{"type":"text","text":
            "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
            "Respond with only its uppercase letter, with no explanation or reasoning."}]},
          {"role":"user","content":[{"type":"text","text":json.dumps(payload,ensure_ascii=False)}]}]
    txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = proc.tokenizer(txt, return_tensors="pt").to(model.device)
    slots=[proc.tokenizer.encode(L,add_special_tokens=False)[0] for L in LETTERS[:len(row["options"])]]
    with torch.inference_mode():
        lg = model(**ids, use_cache=False).logits[:,-1,:][0].float()[slots].cpu().tolist()
    return dict(zip([o["id"] for o in row["options"]], lg))

def main():
    model, proc, meta = semif_vl.load_vl_model(semif_vl.DEFAULT_MODEL)
    salida = {"modelo": semif_vl.DEFAULT_MODEL, "perceptual": [], "normativa": []}

    print(f"\n{'='*64}\nPERCEPTUAL (con imagen)")
    for fichero in ("data/vl_rows.json", "data/vl_rows_hard.json"):
        for row in json.load(open(os.path.join(_HERE, fichero))):
            row = {**row, "image": os.path.join(_HERE, "data", row["image"])}
            r = semif_vl.score(model, proc, row, meta, calibrate=True)
            gan = max(zip(r["option_ids"], r["probabilities"]), key=lambda x: x[1])[0]
            salida["perceptual"].append({"id": r["id"], "gap": r["gap_nats"],
                                         "p_ganador": max(r["probabilities"]),
                                         "n_opciones": len(r["option_ids"]),
                                         "ok": gan == row.get("truth"), "flip": r["winner_flips"]})
            print(f"  {r['id']:<24} gap {r['gap_nats']:>5.1f}n  "
                  f"{'OK ' if gan == row.get('truth') else 'MAL'}  flip={r['winner_flips']}")

    print(f"\n{'='*64}\nNORMATIVA (sin imagen, misma maquinaria)")
    for row in NORMATIVAS:
        votos, gaps = [], []
        for shift in (0, 1):
            rot = row["options"][shift:] + row["options"][:shift]
            lg = texto_solo(model, proc, {**row, "options": rot})
            votos.append(max(lg, key=lg.get)); gaps.append(abs(lg["si"] - lg["no"]))
        flip = len(set(votos)) > 1
        g = sum(gaps) / 2
        import math as _m
        salida["normativa"].append({"id": row["id"], "gap": g, "flip": flip, "votos": votos,
                                    "p_ganador": 1/(1+_m.exp(-g)), "n_opciones": 2})
        print(f"  {row['id']:<24} gap {g:>5.1f}n  voto={'/'.join(sorted(set(votos))):<6} flip={flip}")

    p = [x["gap"] for x in salida["perceptual"]]
    n = [x["gap"] for x in salida["normativa"]]
    print(f"\n{'='*64}")
    print(f"PERCEPTUAL  n={len(p):<3} gap mediana {statistics.median(p):5.1f}n  "
          f"rango [{min(p):.1f}, {max(p):.1f}]  aciertos {sum(x['ok'] for x in salida['perceptual'])}/{len(p)}")
    print(f"NORMATIVA   n={len(n):<3} gap mediana {statistics.median(n):5.1f}n  "
          f"rango [{min(n):.1f}, {max(n):.1f}]  flips {sum(x['flip'] for x in salida['normativa'])}/{len(n)}")
    print(f"separacion: el minimo perceptual ({min(p):.1f}n) frente al maximo normativo ({max(n):.1f}n)")
    json.dump(salida, open(os.path.join(_HERE, "results", "gap_por_tipo.json"), "w"), indent=1)

main()
