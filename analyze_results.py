#!/usr/bin/env python
"""
Summarize <results-root>/<dataset>/<minimizer>/ against a Sella baseline.

Report the same metrics as the xTB evaluation analyzer:

  1. force-call cost      pooled and per-molecule ratios, plus recovered energy
  2. final-energy deltas  dE = E_final(method) - E_final(Sella), in kcal/mol
  3. agreement            heavy-atom RMSD and energy agreement with Sella's minimum

A recorded molecule counts as an error unless it finished cleanly and converged
with finite energies and a positive force-call count. Missing rows are reported
separately. Ratios use mutually converged pairs; agreement percentages use the
full input set, or the union of recorded molecules if inputs are unavailable.
DFT summary energies are in Hartree; reported energy differences are in kcal/mol
and RMSDs are in Angstrom. Numbered subsets are analyzed independently.

Usage:
    python analyze_results.py                       # every dataset found
    python analyze_results.py --dataset test_1
    python analyze_results.py --baseline sella_nofrag
    python analyze_results.py --results-root results --format markdown
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

from utils import ANGSTROM_TO_NM, HARTREE_TO_KJ, parse_xyz


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_INPUTS_ROOT = REPO_ROOT / "inputs"
HARTREE_TO_KCAL = HARTREE_TO_KJ / 4.184
BASELINE = "sella"
RMSD_TOL_ANG = 0.1
ENERGY_TOL_KCAL = 0.1


def read_summary(path: Path) -> dict[str, dict]:
    """Parse one summary.tsv into {mol_id: row}. Later rows win."""
    rows: dict[str, dict] = {}
    with path.open(newline="") as fh:
        reader = csv.DictReader(
            (line for line in fh if not line.startswith("#")), delimiter="\t"
        )
        required = {"mol", "status", "converged", "n_calls", "initial_Eh", "final_Eh"}
        if reader.fieldnames is None:
            return rows
        if not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: expected DFT summary columns {sorted(required)}")
        for row in reader:
            if not row.get("mol") or any(row.get(key) is None for key in required):
                print(f"warning: skipping incomplete row in {path}", file=sys.stderr)
                continue
            rows[row["mol"]] = row
    return rows


def as_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def is_ok(row: dict) -> bool:
    return (
        row.get("status") == "ok"
        and row.get("converged") == "True"
        and all(math.isfinite(as_float(row.get(key)))
                for key in ("n_calls", "initial_Eh", "final_Eh"))
        and as_float(row.get("n_calls")) > 0
    )


def heavy_atom_rmsd(path_a: Path, path_b: Path) -> float:
    """Aligned heavy-atom RMSD (A); use all atoms if fewer than three are heavy."""
    z_a, pos_a = parse_xyz(str(path_a))
    z_b, pos_b = parse_xyz(str(path_b))
    if z_a.shape != z_b.shape or not np.array_equal(z_a, z_b):
        return math.nan
    if not z_a.size or not np.isfinite(pos_a).all() or not np.isfinite(pos_b).all():
        return math.nan
    heavy = z_a > 1
    if heavy.sum() < 3:
        heavy = np.ones_like(heavy)
    a = pos_a[heavy] / ANGSTROM_TO_NM
    b = pos_b[heavy] / ANGSTROM_TO_NM
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, 1.0, d]) @ vt
    aligned = b @ rotation.T
    return float(np.sqrt(np.mean(np.sum((a - aligned) ** 2, axis=1))))


def compare(
    base_dir: Path, method_dir: Path, with_rmsd: bool,
    expected_molecules: list[str] | None = None,
) -> dict:
    base = read_summary(base_dir / "summary.tsv")
    method = read_summary(method_dir / "summary.tsv")
    recorded = set(base) | set(method)
    molecules = sorted(set(expected_molecules) if expected_molecules is not None else recorded)
    extra = recorded - set(molecules)
    if extra:
        print(f"warning: ignoring {len(extra)} molecule(s) absent from inputs in "
              f"{base_dir} / {method_dir}", file=sys.stderr)
    n_total = len(molecules)

    base_errors = sum(1 for m in molecules if m in base and not is_ok(base[m]))
    method_errors = sum(1 for m in molecules if m in method and not is_ok(method[m]))
    base_missing = sum(m not in base for m in molecules)
    method_missing = sum(m not in method for m in molecules)

    calls_b: list[float] = []
    calls_m: list[float] = []
    energy_ratios: list[float] = []
    deltas_kcal: list[float] = []
    rmsds: list[float] = []
    same_min = 0
    rmsd_ok = 0
    energy_ok = 0

    for mol in molecules:
        rb, rm = base.get(mol), method.get(mol)
        if rb is None or rm is None or not is_ok(rb) or not is_ok(rm):
            continue
        nb, nm = as_float(rb["n_calls"]), as_float(rm["n_calls"])
        if nb <= 0 or nm <= 0:
            continue
        calls_b.append(nb)
        calls_m.append(nm)

        e0 = as_float(rb["initial_Eh"])
        e1b = as_float(rb["final_Eh"])
        e1m = as_float(rm["final_Eh"])
        drop_b, drop_m = e0 - e1b, e0 - e1m
        if drop_b * HARTREE_TO_KCAL > 1e-9:
            energy_ratios.append(drop_m / drop_b)

        delta_kcal = (e1m - e1b) * HARTREE_TO_KCAL
        deltas_kcal.append(delta_kcal)
        if abs(delta_kcal) < ENERGY_TOL_KCAL:
            energy_ok += 1

        if with_rmsd:
            xyz_b = base_dir / f"{mol}_final.xyz"
            xyz_m = method_dir / f"{mol}_final.xyz"
            if xyz_b.is_file() and xyz_m.is_file():
                try:
                    value = heavy_atom_rmsd(xyz_b, xyz_m)
                except (OSError, ValueError, IndexError, KeyError, np.linalg.LinAlgError) as exc:
                    print(f"warning: cannot compare geometry for {mol}: {exc}", file=sys.stderr)
                    value = math.nan
                if math.isfinite(value):
                    rmsds.append(value)
                    if value < RMSD_TOL_ANG:
                        rmsd_ok += 1
                        if abs(delta_kcal) < ENERGY_TOL_KCAL:
                            same_min += 1

    b = np.asarray(calls_b, dtype=float)
    m = np.asarray(calls_m, dtype=float)
    d = np.asarray(deltas_kcal, dtype=float)
    r = np.asarray(rmsds, dtype=float)

    def pct(count: int) -> float:
        return 100.0 * count / n_total if n_total else math.nan

    return {
        "n_total": n_total,
        "n_scored": int(b.size),
        "base_errors": base_errors,
        "method_errors": method_errors,
        "base_missing": base_missing,
        "method_missing": method_missing,
        "pooled_pct": 100.0 * float(m.sum() / b.sum()) if b.size else math.nan,
        "permol_pct": 100.0 * float(np.mean(m / b)) if b.size else math.nan,
        "energy_ratio": float(np.mean(energy_ratios)) if energy_ratios else math.nan,
        "mean_dE": float(np.mean(d)) if d.size else math.nan,
        "median_dE": float(np.median(d)) if d.size else math.nan,
        "max_dE": float(np.max(d)) if d.size else math.nan,
        "min_dE": float(np.min(d)) if d.size else math.nan,
        "deeper_pct": pct(int((d < 0).sum())),
        "over_0p1_pct": pct(int((d > 0.1).sum())),
        "over_1p0_pct": pct(int((d > 1.0).sum())),
        "median_rmsd": float(np.median(r)) if r.size else math.nan,
        "rmsd_ok_pct": pct(rmsd_ok) if with_rmsd else math.nan,
        "energy_ok_pct": pct(energy_ok),
        "same_min_pct": pct(same_min) if with_rmsd else math.nan,
    }


def discover(
    results_root: Path, dataset: str | None, baseline: str = BASELINE
) -> list[tuple[str, list[str]]]:
    found: list[tuple[str, list[str]]] = []
    if not results_root.is_dir():
        return found
    for dataset_dir in sorted(results_root.iterdir()):
        if not dataset_dir.is_dir() or (dataset and dataset_dir.name != dataset):
            continue
        methods = sorted(
            p.name for p in dataset_dir.iterdir()
            if p.is_dir() and (p / "summary.tsv").is_file()
        )
        if baseline in methods:
            found.append((dataset_dir.name, methods))
    return found


def fmt(value: float, digits: int = 2) -> str:
    return "n/a" if value is None or math.isnan(value) else f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--baseline", default=BASELINE,
                        help="Baseline directory name (default: sella; e.g. sella_nofrag)")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Results tree to analyze (default: this repository's results/)",
    )
    parser.add_argument("--inputs-root", type=Path, default=DEFAULT_INPUTS_ROOT,
                        help="Input tree used for full dataset sizes (default: inputs/)")
    parser.add_argument("--format", default="text", choices=["text", "markdown"])
    parser.add_argument("--no-rmsd", action="store_true",
                        help="Skip RMSD calculations (much faster)")
    args = parser.parse_args()

    results_root = args.results_root.expanduser().resolve()
    inputs_root = args.inputs_root.expanduser().resolve()
    blocks = discover(results_root, args.dataset, args.baseline)
    if not blocks:
        print(f"no result blocks with a {args.baseline} baseline found under {results_root}")
        return 1

    md = args.format == "markdown"
    for dataset, methods in blocks:
        base_dir = results_root / dataset / args.baseline
        others = [m for m in methods if m != args.baseline]
        input_files = sorted(p for p in (inputs_root / dataset).glob("*.xyz") if p.is_file())
        expected = [p.stem for p in input_files] if input_files else None
        population = "input molecules" if expected is not None else "recorded molecule union"
        header = f"{dataset}   (baseline: {args.baseline}; N: {population})"
        print(f"\n{'## ' if md else ''}{header}")
        if not others:
            print("\nNo other minimizer summaries available for comparison.")
            continue
        if md:
            print("\n| Method | Err. b/a | Missing b/a | Pooled (%) | Per-mol. (%) | Energy | "
                  "Mean dE (kcal/mol) | Median dE | Max dE | Min dE | Deeper (%) | >0.1 (%) | >1.0 (%) | "
                  "Median RMSD (Å) | RMSD<0.1 (%) | \\|dE\\|<0.1 (%) | Same min. (%) | Scored | N |")
            print("|---|---|---|" + "---:|" * 16)
        else:
            print("-" * len(header))
        for method in others:
            stats = compare(base_dir, results_root / dataset / method,
                            with_rmsd=not args.no_rmsd, expected_molecules=expected)
            if md:
                print(
                    f"| {method} | {stats['base_errors']}/{stats['method_errors']} | "
                    f"{stats['base_missing']}/{stats['method_missing']} | "
                    f"{fmt(stats['pooled_pct'])} | {fmt(stats['permol_pct'])} | "
                    f"{fmt(stats['energy_ratio'], 6)} | {fmt(stats['mean_dE'], 4)} | "
                    f"{fmt(stats['median_dE'], 6)} | {fmt(stats['max_dE'])} | "
                    f"{fmt(stats['min_dE'])} | "
                    f"{fmt(stats['deeper_pct'])} | {fmt(stats['over_0p1_pct'])} | "
                    f"{fmt(stats['over_1p0_pct'])} | {fmt(stats['median_rmsd'], 4)} | "
                    f"{fmt(stats['rmsd_ok_pct'], 1)} | {fmt(stats['energy_ok_pct'], 1)} | "
                    f"{fmt(stats['same_min_pct'], 1)} | "
                    f"{stats['n_scored']} | {stats['n_total']} |"
                )
            else:
                print(
                    f"  {method:<10} N={stats['n_total']:<5} scored={stats['n_scored']:<5} "
                    f"err b/a={stats['base_errors']}/{stats['method_errors']:<4} "
                    f"missing b/a={stats['base_missing']}/{stats['method_missing']} "
                    f"pooled={fmt(stats['pooled_pct'])}%  per-mol={fmt(stats['permol_pct'])}%  "
                    f"energy={fmt(stats['energy_ratio'], 6)}  "
                    f"meandE={fmt(stats['mean_dE'], 4)} kcal/mol  "
                    f"medRMSD={fmt(stats['median_rmsd'], 4)} Å  "
                    f"energy-agree={fmt(stats['energy_ok_pct'], 1)}%  "
                    f"same-min={fmt(stats['same_min_pct'], 1)}%"
                )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
