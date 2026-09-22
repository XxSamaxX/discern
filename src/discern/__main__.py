"""discern — ask a vision model a question from the command line.

    discern IMAGE "question"                       yes/no
    discern IMAGE "question" opt1 opt2 [opt3 ...]  multiple choice

Options:
    --device cpu|cuda:0   force the device (default: pick whatever fits)
    --model  REPO_ID      force the model  (default: 4B on GPU, 2B on CPU)
    --threshold NATS      trust threshold, default 3.0

Exit code: 0 yes (or a resolved multiple choice), 1 no, 2 below the threshold.

    discern photo.jpg "is this posed?" && echo posed
    discern photo.jpg "is there a person?" --device cpu
"""
import sys


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    opciones = {}
    for clave in ("--device", "--model", "--threshold"):
        while clave in argv:
            i = argv.index(clave)
            if i + 1 >= len(argv):
                print(f"{clave} necesita un valor", file=sys.stderr)
                return 2
            opciones[clave.lstrip("-")] = argv.pop(i + 1)
            argv.pop(i)
    if len(argv) < 2:
        print(__doc__.strip())
        return 2

    from . import discern as _discern
    img, pregunta, *ops = argv
    v = _discern(img, pregunta, ops or None,
                 device=opciones.get("device"),
                 model=opciones.get("model"),
                 threshold=float(opciones.get("threshold", 3.0)))
    print(v)
    if not v.trusted:
        return 2
    return 0 if (not v.binary or bool(v)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
