#!/bin/bash
# Build the submission bundle. Numbers are regenerated first so the PDF can
# never be stale with respect to the runs; the zip is built from the same
# sources that were just compiled, never assembled by hand.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src
NAME=robowm
REVIEW="$HOME/Downloads/CALTECH/conference drafts/review/$NAME"
DRAFTS="$HOME/Downloads/CALTECH/conference drafts"

python3 src/make_numbers.py --runs runs --output paper/numbers.tex
python3 src/make_figure.py --runs runs --output figs/criterion.png \
  --meta runs/figure1_meta.json
# The figure lives beside main.tex so the zip rebuilds without the repo.
cp figs/criterion.pdf paper/criterion.pdf
cd paper
mkdir -p build
tectonic -X compile main.tex --outdir build --keep-logs >/dev/null 2>&1
PAGES=$(grep -oE "Output written on.*\(([0-9]+) page" build/main.log | grep -oE "[0-9]+ page" | grep -oE "[0-9]+")
UNDEF=$(grep -ic "undefined" build/main.log || true)
echo "pages: $PAGES | undefined: $UNDEF"
if [ "$UNDEF" != "0" ]; then echo "REFUSING: undefined references"; exit 1; fi
if [ "$PAGES" -gt 5 ]; then echo "REFUSING: $PAGES pages, short-paper limit is 4 plus references"; exit 1; fi

mkdir -p "$REVIEW"
cp main.tex numbers.tex refs.bib neurips_2026.sty criterion.pdf "$REVIEW/"
cp build/main.pdf "$REVIEW/$NAME.pdf"
cd "$REVIEW"
rm -f "$DRAFTS/$NAME.zip"
zip -q -r "$DRAFTS/$NAME.zip" main.tex numbers.tex refs.bib neurips_2026.sty criterion.pdf
cp "$REVIEW/$NAME.pdf" "$DRAFTS/$NAME.pdf"
echo "bundled: $DRAFTS/$NAME.zip and $NAME.pdf"
unzip -l "$DRAFTS/$NAME.zip" | tail -3
