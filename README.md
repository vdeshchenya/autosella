# AutoSella DFT evaluation

This branch contains scripts and input geometries for evaluating molecular geometry minimizers with DFT forces.

The included minimizers are frozen as follows:

| Public name | File                           | SHA-256                                                            |
| ----------- | ------------------------------ | ------------------------------------------------------------------ |
| Sella       | `minimizers/sella_baseline.py` | `f441448085d06cf26a6781498d6fd5fe4f4985603ed9c1ad29e05cac01257f29` |
| AutoSella-G | `minimizers/autosella_g.py`    | `6e2eab4b8c3a1323883c9295f3ae3a11dfe59cf1b45095cb637826e3845fa0d4` |
| AutoSella-A | `minimizers/autosella_a.py`    | `2e07cc802a8666d4e97bc3f1ed034469437ea87351f4a2a7054fa3450d67ba54` |
| AutoSella-F | `minimizers/autosella_f.py`    | `64115d8b46c8506ebeb246d6950643ff768fc17a8382fedc8f2a15e8957f04ce` |

DFT single-point energies and gradients are computed through ORCA/OPI using the `r2SCAN-3c` composite method with ORCA `ENGRAD` and `TightSCF` jobs.

## Evaluation sets

The `inputs` directory contains the four fixed geometry sets used by the evaluation. All XYZ files store the signed integer molecular charge on the second line as `charge=<integer>`. The complete selection and preparation code is maintained in the `main`-branch molecule evaluation pipeline.

| Dataset                                                     | Number of molecules | Main-branch source                                  |
| ----------------------------------------------------------- | ------------------- | --------------------------------------------------- |
| Test (`inputs/test/`)                                       | 500                 | `molecules/evaluation/test/xyz/`                    |
| Dipeptides (`inputs/dipeptides/`)                           | 673                 | `molecules/evaluation/dipeptides/xyz/`              |
| Amino-acid–ligand pairs (`inputs/amino_acid_ligand_pairs/`) | 475                 | `molecules/evaluation/amino_acid_ligand_pairs/xyz/` |
| Solvated PubChem (`inputs/solvated_pubchem/`)               | 100                 | `molecules/evaluation/solvated_pubchem/xyz/`        |

## Running an evaluation

Create and activate the `autogeomopt` environment from `conda-lock.yml` or `environment.yml`. The environment files pin xTB 6.7.1 and Sella 2.5.0; the lock targets Linux x86-64. Install and configure ORCA 6.1.1 or newer separately.

Run from the repository root:

```bash
./run.sh <dataset> <minimizer>
```

Here `<dataset>` accepts `test`, `dipeptides`, `amino_acid_ligand_pairs`, `solvated_pubchem`, or a path to a folder containing `.xyz` files. `<minimizer>` accepts `sella` (default), `autosella_g`/`g`, `autosella_a`/`a`, `autosella_f`/`f`, or an explicit minimizer Python file path.

Results are written to `results/<dataset>/<minimizer>/`. For a folder path, `<dataset>` is the folder's basename. Each run starts from the beginning and replaces results in that output folder.

The budget is 200 force calls per molecule, or 500 for `solvated_pubchem`. ORCA uses the detected physical-core count by default. Direct `minimize_dft_folder.py` invocations can override these with `--max-force-calls` and `--ncores`.

## Repository layout

- `minimizers/`: frozen Sella and AutoSella-G/A/F implementations.
- `inputs/<dataset>/`: charge-tagged XYZ inputs.
- `run.sh`: launch one dataset/minimizer evaluation.
- `minimize_dft_folder.py`: evaluate a folder with ORCA/OPI DFT forces.
- `dft_molecular_system.py`, `convergence.py`, `utils.py`: evaluator and helpers.
- `analyze_results.py`: compare results against the Sella baseline.
- `results/<dataset>/<minimizer>/`: generated evaluation outputs.

Each output directory contains `summary.tsv`, per-molecule `*_info.txt`, and `*_final.xyz` when a final geometry is available. The summary columns are `mol`, `charge`, `status`, `converged`, `n_calls`, `initial_Eh`, `final_Eh`, `delta_Eh`, `elapsed_s`, and `elapsed_s_per_step`. Energies are in Hartree.

## License and attribution

The repository's software is distributed under the GNU Lesser General Public License v3.0 only. See `LICENSE`, `COPYING`, and `COPYING.LESSER`.

The optimizer implementations are derived from and substantially modify Sella  2.5.0. Upstream attribution and a summary of modifications are provided in  `NOTICE`. Dataset artifacts retain their respective upstream terms;  the software license does not relicense SPICE, RowanSci, or other third-party data.
