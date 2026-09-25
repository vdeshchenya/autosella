# AutoSella xTB and force-field evaluation

This branch contains scripts, input geometries, and result summaries for evaluating molecular geometry minimizers on the **GFN2-xTB** and **GFN-FF** potentials.

The included minimizers are frozen as follows:

| Public name | File                           | SHA-256                                                            |
| ----------- | ------------------------------ | ------------------------------------------------------------------ |
| Sella       | `minimizers/sella_baseline.py` | `f441448085d06cf26a6781498d6fd5fe4f4985603ed9c1ad29e05cac01257f29` |
| AutoSella-G | `minimizers/autosella_g.py`    | `6e2eab4b8c3a1323883c9295f3ae3a11dfe59cf1b45095cb637826e3845fa0d4` |
| AutoSella-A | `minimizers/autosella_a.py`    | `2e07cc802a8666d4e97bc3f1ed034469437ea87351f4a2a7054fa3450d67ba54` |
| AutoSella-F | `minimizers/autosella_f.py`    | `64115d8b46c8506ebeb246d6950643ff768fc17a8382fedc8f2a15e8957f04ce` |

GFN2-xTB single-point energies and gradients are computed through the standalone `xtb` executable using `--grad --acc 0.2`. GFN-FF uses the same executable with `--gfnff --grad`. Its topology is generated once from the original input geometry and charge using `xtb --gfnff --norestart`, then reused for every scoring and optimization call through the [GFN-FF topology restart](https://xtb-docs.readthedocs.io/en/latest/gfnff.html#topology-file). Both potentials use the same energy, gradient, and displacement convergence criteria.

## Evaluation sets

The `inputs` directory contains the four fixed geometry sets used by the evaluation. All XYZ files store the signed integer molecular charge on the second line as `charge=<integer>`. The complete selection and preparation code is maintained in the `main`-branch molecule evaluation pipeline.

| Dataset                                                     | Number of molecules | Main-branch source                                  |
| ----------------------------------------------------------- | ------------------- | --------------------------------------------------- |
| Test (`inputs/test/`)                                       | 500                 | `molecules/evaluation/test/xyz/`                    |
| Dipeptides (`inputs/dipeptides/`)                           | 673                 | `molecules/evaluation/dipeptides/xyz/`              |
| Amino-acid–ligand pairs (`inputs/amino_acid_ligand_pairs/`) | 475                 | `molecules/evaluation/amino_acid_ligand_pairs/xyz/` |
| Solvated PubChem (`inputs/solvated_pubchem/`)               | 100                 | `molecules/evaluation/solvated_pubchem/xyz/`        |

## Benchmark results

Sella 2.5.0 is the baseline. The results below compare AutoSella-G, AutoSella-A, and AutoSella-F with Sella on four input sets using GFN2-xTB and GFN-FF with a budget of 200 force evaluations per molecule, or 500 for solvated PubChem.

Relative force-call values below 100% indicate savings: `Pooled` weights molecules by optimization cost, while `Per-mol.` weights molecules equally. `Energy` is the mean ratio of recovered energy decrease relative to Sella. These ratios use only molecules where both minimizers converged. `Err. b/a` means baseline errors / algorithm errors, counting failures and nonconverged runs, and `N` is the full benchmark size. “Same min.” requires heavy-atom RMSD < 0.1 Å and |ΔE| < 0.1 kcal/mol.

### GFN2-xTB

| Dataset | N | Method | Err. b/a | Pooled (%) ↓ | Per-mol. (%) ↓ | Energy ↑ | Same min. (%) |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| Test | 500 | AutoSella-G | 0/1 | 99.05 | 99.27 | 1.000074 | 99.0 |
|  |  | AutoSella-A | 0/1 | 80.34 | 80.94 | 1.000044 | 97.6 |
|  |  | AutoSella-F | 0/0 | **72.05** | **73.25** | **1.000284** | 97.2 |
| Dipeptides | 673 | AutoSella-G | 0/0 | 99.71 | 100.49 | 1.000449 | 99.0 |
|  |  | AutoSella-A | 0/0 | 77.01 | 78.14 | 1.000908 | 95.7 |
|  |  | AutoSella-F | 0/0 | **64.06** | **65.28** | **1.001613** | 93.2 |
| Amino-acid–ligand pairs |  475| AutoSella-G | 0/2 | 140.32 | 148.77 | **1.027613** | 41.9 |
|  |  | AutoSella-A | 0/1 | 90.41 | 92.57 | 1.008460 | 69.5 |
|  |  | AutoSella-F | 0/0 | **67.21** | **69.79** | 1.024184 | 39.8 |
| Solvated PubChem | 100 | AutoSella-G | 2/68 | 216.52 | 239.85 | 0.983101 | 0.0 |
|  |  | AutoSella-A | 2/0 | 113.01 | 118.77 | 0.999125 | 18.0 |
|  |  | AutoSella-F | 2/0 | **38.95** | **39.86** | **1.014870** | 1.0 |

### GFN-FF — secondary results

| Dataset | N | Method | Err. b/a | Pooled (%) ↓ | Per-mol. (%) ↓ | Energy ↑ | Same min. (%) |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| Test | 500 | AutoSella-G | 0/2 | 99.69 | 99.93 | 1.002694 | 97.8 |
|  |  | AutoSella-A | 0/1 | 78.52 | 78.90 | 1.002606 | 94.8 |
|  |  | AutoSella-F | 0/1 | **74.31** | **74.14** | **1.002850** | 96.4 |
| Dipeptides | 673 | AutoSella-G | 0/0 | 100.17 | 100.26 | 1.000471 | 98.2 |
|  |  | AutoSella-A | 0/0 | 77.06 | 77.80 | **1.000674** | 82.6 |
|  |  | AutoSella-F | 0/0 | **67.91** | **68.91** | 0.999653 | 79.6 |
| Amino-acid–ligand pairs | 475 | AutoSella-G | 2/1 | 127.75 | 133.46 | 1.012540 | 35.4 |
|  |  | AutoSella-A | 2/1 | 86.63 | 88.68 | 0.997444 | 62.3 |
|  |  | AutoSella-F | 2/1 | **70.41** | **72.57** | **1.017843** | 31.8 |
| Solvated PubChem | 100 | AutoSella-G | 1/13 | 195.75 | 210.29 | 0.958138 | 0.0 |
|  |  | AutoSella-A | 1/1 | 109.63 | 114.59 | 0.999054 | 22.0 |
|  |  | AutoSella-F | 1/0 | **33.34** | **34.08** | **1.012442** | 5.0 |

Failure counts across all four datasets (runtime errors / nonconverged):

| Potential | Sella | AutoSella-G | AutoSella-A | AutoSella-F |
| --- | ---: | ---: | ---: | ---: |
| GFN2-xTB | 1/1 | 5/66 | 2/0 | 0/0 |
| GFN-FF | 1/2 | 4/12 | 0/3 | 0/2 |

## Running an evaluation

Create and activate the `autogeomopt` environment from `conda-lock.yml` or `environment.yml`. The environment files pin xTB 6.7.1 and Sella 2.5.0; the lock targets Linux x86-64. The `xtb` executable must be on `PATH`, or supplied through `XTB_BIN`.

Run from the repository root:

```bash
./run.sh <dataset> <potential> <minimizer>
```

Here `<dataset>` accepts `test`, `dipeptides`, `amino_acid_ligand_pairs`, `solvated_pubchem`, or a path to a folder containing `.xyz` files. `<potential>` accepts `xtb` or `ff`. `<minimizer>` accepts `sella` (default), `autosella_g`/`g`, `autosella_a`/`a`, `autosella_f`/`f`, or an explicit minimizer Python file path.

Results are written to `results/<dataset>/<potential>/<minimizer>/`. For a folder path, `<dataset>` is the folder's basename. Each run starts from the beginning and replaces results in that output folder.

The budget is 200 force calls per molecule, or 500 for `solvated_pubchem`. Molecules run in parallel using the detected physical-core count worker processes by default, with a minimum of one; each xTB calculation uses one thread. invocations can override these with `--max-force-calls` and `--procs`.

Build the comparison tables with:

```bash
python analyze_results.py --format markdown
```

## Repository layout

- `minimizers/`: frozen Sella and AutoSella-G/A/F implementations.
- `inputs/<dataset>/`: charge-tagged XYZ inputs.
- `run.sh`: launch one dataset/potential/minimizer combination.
- `minimize_folder.py`: evaluate a folder with xTB or force-field energies and forces.
- `xtb_molecular_system.py`, `ff_molecular_system.py`: potential backends.
- `convergence.py`, `utils.py`: convergence criteria, unit conversions, and XYZ I/O.
- `analyze_results.py`: compare results against the Sella baseline.
- `results/<dataset>/<potential>/<minimizer>/`: generated evaluation outputs.

Each output directory contains `summary.tsv`, per-molecule `*_info.txt`, and `*_final.xyz` when a final geometry is available. The summary columns are `mol`, `charge`, `status`, `converged`, `n_calls`, `initial_kcal_mol`, `final_kcal_mol`, `delta_kcal_mol`, `elapsed_s`, and `elapsed_s_per_step`. Energies are in kcal/mol. Per-molecule metadata records energies in kJ/mol and, for xTB, Hartree.

## License and attribution

The repository's software is distributed under the GNU Lesser General Public License v3.0 only. See `LICENSE`, `COPYING`, and `COPYING.LESSER`.

The optimizer implementations are derived from and substantially modify Sella  2.5.0. Upstream attribution and a summary of modifications are provided in  `NOTICE`. Dataset artifacts retain their respective upstream terms;  the software license does not relicense SPICE, RowanSci, or other third-party data.
