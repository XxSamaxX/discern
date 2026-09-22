"""CLI:  discern IMAGE "question" [option option ...]

Codigo de salida: 0 = si (o multi-opcion resuelta), 1 = no, 2 = gap por debajo
del umbral. Pensado para encadenar en shell:

    discern foto.jpg "is this posed?" && echo posada
"""
import sys


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2:
        print(__doc__.strip())
        return 2
    from . import discern as _discern
    img, pregunta, *ops = argv
    v = _discern(img, pregunta, ops or None)
    print(v)
    if not v.trusted:
        return 2
    return 0 if (not v.binary or bool(v)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
