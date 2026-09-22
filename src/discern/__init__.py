"""discern — ask a vision model a question, get the answer from its logits.

    from discern import discern

    if discern("photo.jpg", "is there a person?"):
        blur_faces()

    v = discern("photo.jpg", "what is the camera viewpoint?",
                ["from above", "at eye level", "from below"])
    print(v.answer, v.gap)

One forward pass, no decoding, nothing trained. The confidence you get back is
the logit gap in nats, not the softmax probability -- on 4,329 MMBench items
that probability was >= 0.99 on 52% of the model's own errors, while the gap
predicts correctness with AUC 0.894. See the repo README for the measurements.

Below `threshold` nats the answer is marked untrusted, and by default the
options are rotated and averaged before giving up on it: rotating changes the
winner on 11.7% of items, and those have a median gap of 2.7 nats.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

__version__ = "0.2.0"

__all__ = ["discern", "multi", "Verdict", "Uncertain", "load", "unload",
           "DEFAULT_THRESHOLD"]

class Uncertain(ValueError):
    """Se uso como booleano un veredicto por debajo del umbral de confianza."""


DEFAULT_THRESHOLD = 3.0          # nats; ver README, seccion de sesgo posicional
_MODELO = None                   # (model, processor, meta), perezoso


def load(model: str | None = None, device: str | None = None):
    """Carga el modelo (perezoso y cacheado). Llamalo tu si quieres controlar
    cuando se paga el coste de carga; si no, la primera llamada lo hace.

    Sin argumentos elige solo: GPU con el 4B si el checkpoint cabe en la VRAM,
    CPU con el 2B si no. device="cpu" o "cuda:0" fuerza el dispositivo."""
    global _MODELO
    from . import _readout as semif_vl
    if model is None:
        # con un modelo ya cargado y sin peticion explicita, no se vuelve a
        # decidir: choose() consulta el hub y son ~300 ms por llamada
        if _MODELO is not None and device is None:
            return _MODELO
        model, device = semif_vl.choose(device)
    if _MODELO is None or _MODELO[2]["source"] != model:
        _MODELO = semif_vl.load_vl_model(model, device=device)
    return _MODELO


def unload():
    """Libera la VRAM."""
    global _MODELO
    _MODELO = None
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


@dataclass
class Verdict:
    """El resultado de una pregunta. En binarias se puede usar como booleano."""
    answer: str
    gap: float
    trusted: bool
    probabilities: dict = field(repr=False)
    forwards: int = 1
    ms: float = 0.0
    binary: bool = False

    def __bool__(self) -> bool:
        if not self.binary:
            raise TypeError(
                f"{self.answer!r} is one of {list(self.probabilities)}; a "
                "multiple-choice verdict has no truth value. Compare .answer, "
                "or ask a yes/no question instead.")
        # Un veredicto por debajo del umbral NO se deja usar como condicion:
        # a 1-2 nats el sesgo posicional ya decide la respuesta (README), asi
        # que dejarlo pasar como True silencioso seria justo el error que este
        # proyecto documenta. .answer sigue disponible para inspeccionarlo.
        if not self.trusted:
            raise Uncertain(
                f"answer {self.answer!r} rests on a {self.gap:.1f} nat gap, below "
                f"the trust threshold. Read .answer if you want it anyway, or "
                f"pass threshold=0 to disable this check.")
        return self.answer == "yes"

    def __str__(self) -> str:
        marca = "" if self.trusted else "  (untrusted)"
        return f"{self.answer}  [{self.gap:.1f} nats]{marca}"


def _opciones(options) -> tuple[list[dict], bool]:
    if options is None:
        return ([{"id": "yes", "description": "Yes"},
                 {"id": "no", "description": "No"}], True)
    if isinstance(options, dict):
        ops = [{"id": str(k), "description": str(v)} for k, v in options.items()]
    else:
        ops = [{"id": str(o), "description": str(o)} for o in options]
    if len(ops) < 2:
        raise ValueError("need at least two options")
    if len({o["id"] for o in ops}) != len(ops):
        raise ValueError("option labels must be unique")
    return ops, False


def discern(image, question: str, options=None, *,
            evidence: str = "", model: str | None = None, device: str | None = None,
            threshold: float = DEFAULT_THRESHOLD, rotate: str | bool = "auto") -> Verdict:
    """Pregunta algo sobre una imagen y lee la respuesta de los logits.

    image     ruta, URL o PIL.Image
    question  la pregunta, en texto libre
    options   None -> si/no; lista de cadenas; o dict {etiqueta: descripcion}
    evidence  contexto textual opcional que acompana a la imagen
    device    None elige solo (GPU si cabe, si no CPU); "cpu" o "cuda:0" fuerza
    threshold nats por debajo de los cuales el veredicto se marca no fiable
    rotate    "auto" (rota solo si el gap queda corto), True (siempre), False
    """
    from . import _readout as semif_vl
    m, proc, meta = load(model, device)
    ops, binaria = _opciones(options)
    fila = {"id": "q", "image": image, "state": evidence,
            "question": question.strip(), "options": ops}

    t0 = time.perf_counter()
    r = semif_vl.score(m, proc, fila, meta, calibrate=(rotate is True))
    # "auto": solo se paga la rotacion cuando el gap cae donde el orden decide
    if rotate == "auto" and r["gap_nats"] < threshold and len(ops) > 1:
        r = semif_vl.score(m, proc, fila, meta, calibrate=True)

    mejor = max(zip(r["option_ids"], r["probabilities"]), key=lambda x: x[1])[0]
    return Verdict(answer=mejor, gap=r["gap_nats"], trusted=r["gap_nats"] >= threshold,
                   probabilities=dict(zip(r["option_ids"], r["probabilities"])),
                   forwards=r["forwards"], ms=(time.perf_counter() - t0) * 1000,
                   binary=binaria)


def multi(*a, **k):
    """Varias preguntas sobre una imagen, codificandola una sola vez.
    Implementado en discern._batch; se importa aqui de forma perezosa para que
    `import discern` no arrastre torch hasta que hace falta."""
    from ._batch import multi as _multi
    return _multi(*a, **k)
