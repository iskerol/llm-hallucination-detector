#!/bin/bash
set -e
conda env create -f environment.yml
conda run -n ruc-detect python -m spacy download en_core_web_sm
conda run -n ruc-detect python build_index.py --sample 5000
echo "Setup complete. Run: conda activate ruc-detect"
