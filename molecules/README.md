# Molecule dataset pipeline

This pipeline builds training and validation geometries from SPICE PubChem, SPICE DES370K, and the RowanSci benchmark. Inputs are optimized directly with Sella/GFN2-xTB. Accepted optimization inputs are saved in `xyz/`, with the last evaluated geometries in `xyz_final/`.

## Run

Activate the repository's conda environment, then run `./run_pipeline.sh` to rebuild the main dataset. Each run starts fresh: generated XYZ files, manifests, baselines, and train/validation JSON files are removed first. Downloaded source data are kept. It also verifies the pinned SPICE 2.0.1 HDF5 checksum and the pinned RowanSci archive, commit, filenames, and per-file hashes before use.

Set the worker count for parallel stages:

```bash
XTB_WORKERS=8 ./run_pipeline.sh
```

Without this variable, parallel stages use the number of physical CPU cores, with one numerical thread per worker. PubChem runs sequentially: it chooses one molecule, tries its conformers one at a time, and stops at the first passing conformer. Each accepted molecule changes the diversity ranking used to choose the next molecule, so selection waits for its outcome. DES and RowanSci batches try one conformer per molecule or class pair at a time.

## Selection

### PubChem

`pubchem.py` considers every numeric group in SPICE and represents each unique canonical SMILES with a 1,024-bit ECFP4 fingerprint. It starts with the smallest-ID molecule that passes preparation, then repeatedly chooses the molecule farthest from all previously accepted molecules.

Conformers are tried in deterministic hash order. A molecule affects later ranking only after a conformer passes GFN2-xTB optimization and the energy, connectivity, and system diameter checks described in [Optimization and acceptance](#optimization-and-acceptance). The pipeline requires exactly 475 successful groups.

### DES370K

`des370k.py` groups dimers by their two monomer classes. It attempts to select one train dimer per represented class pair and, when possible, a different valid dimer.

Train selection finishes across all class pairs before valid selection starts, so validation candidates are ranked against the complete selected training set. They are prioritized by how many of their monomers already occur in train: zero, then one, then two. Up to ten conformers are tried per dimer. A train dimer remains accepted if no valid dimer passes, so validation coverage may be smaller than train coverage.

### RowanSci

`rowansci.py` processes the fixed 25 structures from the pinned RowanSci benchmark. Before optimization, it adds reproducible random coordinate noise with a mean displacement of 0.02 Å per atom. This makes optimization a little harder because some raw conformers are already close to their GFN2-xTB minima.

The perturbed input is saved in `xyz/` on acceptance. The RowanSci manifest records the seed and mean displacement.

## Optimization and acceptance

Every main-dataset candidate runs one Sella/GFN2-xTB optimization. Atom order and charge are preserved. PubChem and DES use source coordinates; RowanSci uses the noisy coordinates described above.

The optimization has a budget of 200 force calls. A candidate is accepted when the calculation succeeds and its energy decreases by at least `0.005 Ha` from the raw input. The last evaluated geometry must have a system diameter (maximum distance between any two atoms) no greater than twice the raw input's diameter. This is a simple check to reject dissociated systems whose fragments have drifted apart. Final RDKit connectivity is compared with the source SMILES reference. For each mismatch, g_ij = r_ij / (R_i + R_j) uses RDKit covalent radii. A missing reference bond is allowed when g_ij < 1; an extra inferred bond is allowed when g_ij > 1. Other mismatches are rejected. Convergence within the budget is not required.

Reported energy, convergence, force-call count, and saved geometry all describe the last evaluated state. Energies in the baseline JSON are total Hartree energies.

## Train/validation split

`train_valid_split.py` reads the three selection manifests and `data/baseline_XTB.json`:

- train receives the first 225 accepted PubChem groups, 25 RowanSci structures, and DES entries marked `train`;
- validation receives the next 250 PubChem groups and DES entries marked `valid`.

Before writing, it checks the PubChem quota, duplicate keys, exact agreement between manifests and baselines, valid DES split labels, and zero train/validation overlap.

## Outputs


| Path                                    | Contents                                                  |
| --------------------------------------- | --------------------------------------------------------- |
| `data/pubchem_selected_conformers.json` | Accepted PubChem groups and conformers in diversity order |
| `data/des370k_selected_dimers.json`     | Accepted DES dimers, monomers, classes, and split         |
| `data/rowansci_selected_inputs.json`    | Accepted RowanSci inputs and source metadata              |
| `data/baseline_XTB.json`                | Combined GFN2-xTB baselines                               |
| `xyz/`                                  | Accepted optimization inputs                              |
| `xyz_final/`                            | Last evaluated GFN2-xTB geometries                        |
| `train_XTB.json`, `valid_XTB.json`      | Final split baselines                                     |


XYZ coordinates are in Angstrom and the second line contains `charge=<integer>`. PubChem files are named `<seed>.xyz`; DES files are named `des370k_<class1>__<class2>__<split>.xyz` (`<split>` = `train` or `valid`). Raw and final geometries use the same filename in their respective directories. Baseline keys are the filename stem, without a method suffix. Selected conformer indices remain in the manifests.

Rejected candidates are not published. Temporary attempts live under `data/tmp/`. Downloaded SPICE and RowanSci inputs are stored in ignored dataset-specific directories under `data/`.