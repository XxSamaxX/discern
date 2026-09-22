#!/usr/bin/env bash
# Empaqueta el envio a arXiv: fuentes, .bbl y figuras, sin auxiliares.
#
# arXiv no ejecuta BibTeX, asi que el .bbl es obligatorio y refs.bib no sirve
# de nada. Tampoco quiere .aux/.log/.out. El script compila primero para que
# el .bbl este al dia, y verifica que el paquete compila por si solo.
set -euo pipefail
R="$(cd "$(dirname "$0")/.." && pwd)"
S="$R/arxiv"; T="$R/arxiv-discern.tar.gz"

cd "$R/paper"
latexmk -pdf -interaction=nonstopmode paper.tex >/dev/null
[ -s paper.bbl ] || { echo "falta paper.bbl"; exit 1; }

rm -rf "$S"; mkdir -p "$S/figures"
cp paper.tex paper.bbl "$S/"
cp figures/*.pdf "$S/figures/"
tar czf "$T" -C "$S" .

D="$(mktemp -d)"; tar xzf "$T" -C "$D"
( cd "$D" && latexmk -pdf -interaction=nonstopmode paper.tex >/dev/null )
n=$(grep -ic undefined "$D/paper.log" || true)
p=$(pdfinfo "$D/paper.pdf" | awk '/Pages/{print $2}')
rm -rf "$D"
[ "$n" = "0" ] || { echo "hay $n referencias sin resolver"; exit 1; }
echo "$T listo: $p paginas, sin referencias rotas, $(du -h "$T" | cut -f1)"
