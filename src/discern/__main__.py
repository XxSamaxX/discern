"""CLI:  python -m discern foto.jpg "is this posed?" [opcion opcion ...]"""
import sys
from . import discern

if len(sys.argv) < 3:
    print(__doc__.strip()); raise SystemExit(2)
img, pregunta, *ops = sys.argv[1:]
v = discern(img, pregunta, ops or None)
print(v)
# codigo de salida usable en shell: 0 = si, 1 = no, 2 = por debajo del umbral
if not v.trusted:
    raise SystemExit(2)
raise SystemExit(0 if (not v.binary or bool(v)) else 1)
