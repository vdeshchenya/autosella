# Evaluation molecule sets

All four pipelines use [SPICE 2.0.1](https://doi.org/10.5281/zenodo.10975225) (CC0) and the main pipeline's Sella/GFN2-xTB optimizer.

| Directory | Target | Selection |
| --- | --- | --- |
| `test` | 500 | PubChem molecules selected for diversity relative to the main dataset and accepted test molecules. |
| `dipeptides` | Up to 676 | Fixed dipeptide identities. |
| `amino_acid_ligand_pairs` | 475 | 25 pairs for each of 19 amino acids, with globally unique ligand IDs. |
| `solvated_pubchem` | 100 | Diversity selection by PubChem solute; optimizes the complete system with its 20 waters. |

Each system tries up to ten conformers in deterministic hash order, stopping at the first accepted one. All pipelines except dipeptides require their full target count.

PubChem selection is sequential because each acceptance changes the next molecule's diversity ranking. It maximizes the minimum Tanimoto distance using hydrogen-normalized, 1024-bit ECFP4 fingerprints. Initial references are train/validation PubChem molecules, selected DES monomers, and train/validation RowanSci molecules reconstructed from the main `xyz/` files. Reference and duplicate candidate SMILES are excluded, only accepted molecules extend the reference set.

## Run

Build the main dataset before either PubChem pipeline, which reads its manifests and train/validation files. Run inside the desired directory:

```bash
cd molecules/evaluation/test
./run_pipeline.sh
```

Each wrapper clears its generated geometries, manifests, and `data/` before running. Shared downloads and main-dataset outputs are retained. Dipeptides and amino-acid–ligand pairs run in parallel, set `XTB_WORKERS` to override the default physical-core worker count.

## Acceptance and outputs

All systems use the shared [energy, connectivity, and diameter checks](../README.md#optimization-and-acceptance): energy must decrease by at least `0.005 Ha`, connectivity must satisfy the covalent-radius rules, and final diameter must not exceed twice the input diameter.

Accepted systems save:

- `xyz/<name>.xyz`: raw SPICE input coordinates.
- `selected_conformers.json`: selected source conformers; named `pubchem_selected_conformers.json` in `test/`.
