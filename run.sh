#!/usr/bin/env bash
# Convenience wrapper around minimize_folder.py.
#
#   ./run.sh DATASET POTENTIAL [MINIMIZER] [-- extra args]
#
# Results land in results/<dataset>/<potential>/<minimizer>/ -- flat, no
# batches: xTB and force-field calls are cheap enough to run a whole dataset on
# one host with a process pool.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat >&2 <<'USAGE'
Usage: ./run.sh DATASET POTENTIAL [MINIMIZER] [-- extra minimize_folder.py args]

  DATASET    test | dipeptides | amino_acid_ligand_pairs | solvated_pubchem
             or a folder containing XYZ files (relative or absolute path)
  POTENTIAL  xtb | ff
  MINIMIZER  sella (default) | autosella_g (g) | autosella_a (a) | autosella_f (f) | path/to.py

Examples:
  ./run.sh test xtb
  ./run.sh dipeptides ff f
  ./run.sh amino_acid_ligand_pairs ff a -- --procs 8
  ./run.sh solvated_pubchem xtb f  # 500 force calls per molecule
  ./run.sh /path/to/xyz xtb f -- --procs 8
USAGE
  exit 2
fi

dataset="$1"
potential="$2"
shift 2

minimizer="sella"
if [[ $# -gt 0 && "$1" != "--" ]]; then
  minimizer="$1"
  shift
fi
if [[ "${1:-}" == "--" ]]; then
  shift
fi

case "${dataset}" in
  test|dipeptides|amino_acid_ligand_pairs|solvated_pubchem)
    input_dir="inputs/${dataset}"
    ;;
  all)
    echo "'all' is not supported; select one dataset or input folder." >&2
    exit 2
    ;;
  *)
    if [[ ! -d "${dataset}" ]]; then
      echo "Unknown dataset or input folder does not exist: ${dataset}" >&2
      exit 2
    fi
    input_dir="$(cd -- "${dataset}" && pwd)"
    dataset="$(basename -- "${input_dir}")"
    ;;
esac

case "${potential}" in
  xtb|ff) ;;
  *)
    echo "Unknown potential: ${potential} (expected 'xtb' or 'ff')" >&2
    exit 2
    ;;
esac

case "${minimizer}" in
  sella)
    minimizer_name="sella"
    minimizer_arg=(--minimizer-path minimizers/sella_baseline.py)
    ;;
  autosella_g|g)
    minimizer_name="autosella_g"
    minimizer_arg=(--minimizer-path minimizers/autosella_g.py)
    ;;
  autosella_a|a)
    minimizer_name="autosella_a"
    minimizer_arg=(--minimizer-path minimizers/autosella_a.py)
    ;;
  autosella_f|f)
    minimizer_name="autosella_f"
    minimizer_arg=(--minimizer-path minimizers/autosella_f.py)
    ;;
  *)
    minimizer_name="$(basename "${minimizer}" .py)"
    minimizer_arg=(--minimizer-path "${minimizer}")
    ;;
esac

output_dir="results/${dataset}/${potential}/${minimizer_name}"

if [[ ! -d "${input_dir}" ]]; then
  echo "Input folder does not exist: ${input_dir}" >&2
  exit 2
fi

max_force_calls=200
if [[ "${dataset}" == "solvated_pubchem" ]]; then
  max_force_calls=500
fi

python minimize_folder.py "${input_dir}" \
  --max-force-calls "${max_force_calls}" \
  "$@" \
  --potential "${potential}" \
  --output-dir "${output_dir}" \
  "${minimizer_arg[@]}"
