#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ $# -ne 0 ]]; then
    echo "Usage: $0" >&2
    exit 2
fi

# A full fresh selection. Downloaded datasets are retained.
rm -rf -- xyz xyz_final data/tmp
rm -f -- \
    train_XTB.json \
    valid_XTB.json \
    data/pubchem_selected_conformers.json \
    data/des370k_selected_dimers.json \
    data/rowansci_selected_inputs.json \
    data/baseline_XTB.json
python pubchem.py
python des370k.py
python rowansci.py
python train_valid_split.py
