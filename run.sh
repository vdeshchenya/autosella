#!/usr/bin/env bash
set -euo pipefail

# Activate the repository environment before invoking this script.  If the
# local ORCA installation requires an environment file, pass its path through
# ORCA_ENV_FILE; dft_molecular_system.py will read its simple export lines.
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 DATASET [MINIMIZER]"
  echo "DATASET may be 'test', 'dipeptides', 'amino_acid_ligand_pairs', 'solvated_pubchem', or an XYZ folder."
  echo "Example: $0 dipeptides"
  echo "Example: $0 test autosella_f"
  echo "MINIMIZER may be 'sella' (default), 'autosella_g'/'g', 'autosella_a'/'a', 'autosella_f'/'f', or a Python file path."
  exit 2
fi

if [[ -n "${ORCA_ENV_FILE:-}" && ! -f "${ORCA_ENV_FILE}" ]]; then
  echo "ORCA_ENV_FILE does not exist: ${ORCA_ENV_FILE}" >&2
  exit 2
fi

if [[ -z "${OPI_ORCA:-}" && -z "${ORCA_BINARY:-}" && -z "${ORCA_BIN:-}" && -z "${ORCA_PATH:-}" ]] \
  && ! command -v orca >/dev/null 2>&1; then
  if [[ -z "${ORCA_ENV_FILE:-}" ]]; then
    echo "ORCA is not configured." >&2
    echo "Set OPI_ORCA, ORCA_BINARY, ORCA_BIN, ORCA_PATH, ORCA_ENV_FILE, or add orca to PATH." >&2
    exit 2
  fi
fi

if [[ -d "inputs/$1" ]]; then
  input_dir="inputs/$1"
elif [[ -d "$1" ]]; then
  input_dir="$1"
else
  echo "Input folder does not exist: $1" >&2
  exit 2
fi
dataset="$(basename "${input_dir}")"
max_force_calls=200
if [[ "${dataset}" == solvated_pubchem ]]; then
  max_force_calls=500
fi

minimizer="${2:-sella}"
case "${minimizer}" in
  sella) minimizer_name=sella; minimizer_path=minimizers/sella_baseline.py ;;
  autosella_g|g) minimizer_name=autosella_g; minimizer_path=minimizers/autosella_g.py ;;
  autosella_a|a) minimizer_name=autosella_a; minimizer_path=minimizers/autosella_a.py ;;
  autosella_f|f) minimizer_name=autosella_f; minimizer_path=minimizers/autosella_f.py ;;
  *) minimizer_name="$(basename "${minimizer}" .py)"; minimizer_path="${minimizer}" ;;
esac

python minimize_dft_folder.py "${input_dir}" \
  --output-dir "results/${dataset}/${minimizer_name}" \
  --max-force-calls "${max_force_calls}" \
  --minimizer-path "${minimizer_path}"
