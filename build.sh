#!/bin/bash
set -eu
PYTHON_BIN="${PYTHON_BIN:-python3}"
BUILD_DIR=".build-figures"
mkdir -p "$BUILD_DIR"
export MPLBACKEND=Agg
export SOURCE_DATE_EPOCH=0
export MPLCONFIGDIR="$PWD/.matplotlib-cache"
export XDG_CACHE_HOME="$PWD/.cache"
(cd code/data/extended && shasum -a 256 -c SHA256SUMS)
(cd code/data/legacy-m20 && shasum -a 256 -c SHA256SUMS)
"$PYTHON_BIN" code/o25_analysis.py --data-dir code/data/extended --out-dir "$BUILD_DIR"
for figure in o25_fig1_sigma_pair o25_fig2_distribution o25_fig3_convergence; do
  cp "$BUILD_DIR/$figure.pdf" "code/$figure.pdf"
done
"$PYTHON_BIN" code/o25_scaling_plot.py --dir code/data/legacy-m20 --out code/o25_scaling.pdf
"$PYTHON_BIN" code/o25_scaling_extended_reconstruct.py
(cd code && shasum -a 256 -c ARTIFACT_SHA256SUMS)
bash compile.sh
