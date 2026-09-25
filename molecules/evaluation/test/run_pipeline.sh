#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ $# -ne 0 ]]; then
    echo "Usage: $0" >&2
    exit 2
fi

rm -rf -- xyz xyz_final data
rm -f -- pubchem_selected_conformers.json \
    conf_indices_1.json
python prepare_ood_pubchem.py
rm -rf -- data
