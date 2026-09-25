#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ $# -ne 0 ]]; then
    echo "Usage: $0" >&2
    exit 2
fi

# Source downloads are retained; selection and GFN2-xTB optimization start fresh.
rm -rf -- xyz_init xyz xyz_final data
rm -f -- selected_conformers.json keys.txt
python prepare_dipeptides.py
rm -rf -- data
