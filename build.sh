#!/bin/bash

set -e

PYTHON_BIN="${PYTHON_BIN:-python3}"

cd code
MPLCONFIGDIR="../.matplotlib-cache" "$PYTHON_BIN" o25_scaling_extended_reconstruct.py
shasum -a 256 -c ARTIFACT_SHA256SUMS
cd ..

bash compile.sh
