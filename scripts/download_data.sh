#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/raw
kaggle competitions download -c kkbox-churn-prediction-challenge -p data/raw
cd data/raw
for f in *.zip; do unzip -o "$f" && rm -f "$f"; done
# Kaggle serves this competition's larger files as .7z, not .zip.
for f in *.7z; do 7z x -y "$f" && rm -f "$f"; done
