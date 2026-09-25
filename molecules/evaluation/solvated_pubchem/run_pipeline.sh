#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ $# -ne 0 ]]; then
    echo "Usage: $0" >&2
    exit 2
fi

rm -rf -- xyz xyz_final data
rm -f -- selected_conformers.json
python prepare_solvated_pubchem.py
rm -rf -- data
