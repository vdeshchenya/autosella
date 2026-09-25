#!/usr/bin/env python
"""
Summarize <results-root>/<dataset>/<potential>/<minimizer>/ against Sella.

Reproduces the three benchmark tables in README.md:

  1. force-call cost      pooled and per-molecule ratios, plus recovered energy
  2. final-energy deltas  dE = E_final(method) - E_final(Sella), in kcal/mol
  3. agreement            heavy-atom RMSD and energy agreement with Sella's minimum

A molecule counts as an *error* when its run did not finish cleanly and
converge: a failure, or a run that used its whole force-call budget without
meeting the convergence test. Ratios use molecules where both the baseline and
the method converged; percentages of the benchmark use the full molecule count,
so a failed optimization can never improve an agreement rate.

Usage:
    python analyze_results.py                       # every dataset/potential found
    python analyze_results.py --dataset test --potential ff
    python analyze_results.py --results-root results --format markdown
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from utils import parse_xyz


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"
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
        for row in reader:
            rows[row["mol"]] = row
    return rows


def as_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def is_ok(row: dict) -> bool:
    return row.get("status") == "ok" and row.get("converged") == "True"


def heavy_atom_rmsd(path_a: Path, path_b: Path) -> float:
    """Heavy-atom RMSD (A) after optimal rigid-body superposition."""
    z_a, pos_a = parse_xyz(str(path_a))
    z_b, pos_b = parse_xyz(str(path_b))
    if z_a.shape != z_b.shape or not np.array_equal(z_a, z_b):
        return math.nan
    heavy = z_a > 1
    if heavy.sum() < 3:
        heavy = np.ones_like(heavy)
    a = pos_a[heavy] / 0.1
    b = pos_b[heavy] / 0.1
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, 1.0, d]) @ vt
    aligned = b @ rotation.T
    return float(np.sqrt(np.mean(np.sum((a - aligned) ** 2, axis=1))))


def compare(
    base_dir: Path, method_dir: Path, with_rmsd: bool
) -> dict:
    base = read_summary(base_dir / "summary.tsv")
    method = read_summary(method_dir / "summary.tsv")
    molecules = sorted(set(base) | set(method))
    n_total = len(molecules)

    base_errors = sum(1 for m in molecules if m not in base or not is_ok(base[m]))
    method_errors = sum(1 for m in molecules if m not in method or not is_ok(method[m]))

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

        e0 = as_float(rb["initial_kcal_mol"])
        e1b = as_float(rb["final_kcal_mol"])
        e1m = as_float(rm["final_kcal_mol"])
        drop_b, drop_m = e0 - e1b, e0 - e1m
        if drop_b > 1e-9:
            energy_ratios.append(drop_m / drop_b)

        delta_kcal = e1m - e1b
        deltas_kcal.append(delta_kcal)
        if abs(delta_kcal) < ENERGY_TOL_KCAL:
            energy_ok += 1

        if with_rmsd:
            xyz_b = base_dir / f"{mol}_final.xyz"
            xyz_m = method_dir / f"{mol}_final.xyz"
            if xyz_b.is_file() and xyz_m.is_file():
                value = heavy_atom_rmsd(xyz_b, xyz_m)
                if not math.isnan(value):
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
    results_root: Path, dataset: str | None, potential: str | None
) -> list[tuple[str, str, list[str]]]:
    found: list[tuple[str, str, list[str]]] = []
    if not results_root.is_dir():
        return found
    for dataset_dir in sorted(results_root.iterdir()):
        if not dataset_dir.is_dir() or (dataset and dataset_dir.name != dataset):
            continue
        for potential_dir in sorted(dataset_dir.iterdir()):
            if not potential_dir.is_dir() or (potential and potential_dir.name != potential):
                continue
            methods = sorted(
                p.name
                for p in potential_dir.iterdir()
                if p.is_dir() and (p / "summary.tsv").is_file()
            )
            if BASELINE in methods:
                found.append((dataset_dir.name, potential_dir.name, methods))
    return found


def fmt(value: float, digits: int = 2) -> str:
    return "n/a" if value is None or math.isnan(value) else f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--potential", default=None, choices=["xtb", "ff"])
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Results tree to analyze (default: the published results/ tree)",
    )
    parser.add_argument("--format", default="text", choices=["text", "markdown"])
    parser.add_argument("--no-rmsd", action="store_true",
                        help="Skip the geometry-agreement table (much faster)")
    args = parser.parse_args()

    results_root = args.results_root.expanduser().resolve()
    blocks = discover(results_root, args.dataset, args.potential)
    if not blocks:
        print(f"no result blocks with a sella baseline found under {results_root}")
        return 1

    md = args.format == "markdown"
    for dataset, potential, methods in blocks:
        base_dir = results_root / dataset / potential / BASELINE
        others = [m for m in methods if m != BASELINE]
        header = f"{dataset} / {potential}   (baseline: {BASELINE})"
        print(f"\n{'## ' if md else ''}{header}")
        if md:
            print("\n| Method | Err. b/a | Pooled (%) | Per-mol. (%) | Energy | "
                  "Mean dE | Median dE | Max dE | Min dE | Deeper (%) | >0.1 (%) | >1.0 (%) | "
                  "Median RMSD | RMSD<0.1 (%) | \\|dE\\|<0.1 (%) | Same min. (%) | N |")
            print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        else:
            print("-" * len(header))
        for method in others:
            stats = compare(base_dir, results_root / dataset / potential / method,
                            with_rmsd=not args.no_rmsd)
            if md:
                print(
                    f"| {method} | {stats['base_errors']}/{stats['method_errors']} | "
                    f"{fmt(stats['pooled_pct'])} | {fmt(stats['permol_pct'])} | "
                    f"{fmt(stats['energy_ratio'], 6)} | {fmt(stats['mean_dE'], 4)} | "
                    f"{fmt(stats['median_dE'], 6)} | {fmt(stats['max_dE'])} | "
                    f"{fmt(stats['min_dE'])} | "
                    f"{fmt(stats['deeper_pct'])} | {fmt(stats['over_0p1_pct'])} | "
                    f"{fmt(stats['over_1p0_pct'])} | {fmt(stats['median_rmsd'], 4)} | "
                    f"{fmt(stats['rmsd_ok_pct'], 1)} | {fmt(stats['energy_ok_pct'], 1)} | "
                    f"{fmt(stats['same_min_pct'], 1)} | "
                    f"{stats['n_total']} |"
                )
            else:
                print(
                    f"  {method:<10} N={stats['n_total']:<5} scored={stats['n_scored']:<5} "
                    f"err b/a={stats['base_errors']}/{stats['method_errors']:<4} "
                    f"pooled={fmt(stats['pooled_pct'])}%  per-mol={fmt(stats['permol_pct'])}%  "
                    f"energy={fmt(stats['energy_ratio'], 6)}  "
                    f"meandE={fmt(stats['mean_dE'], 4)}  "
                    f"medRMSD={fmt(stats['median_rmsd'], 4)}  "
                    f"energy-agree={fmt(stats['energy_ok_pct'], 1)}%  "
                    f"same-min={fmt(stats['same_min_pct'], 1)}%"
                )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
