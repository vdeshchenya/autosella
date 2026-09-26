# AutoSella

AutoSella is a family of molecular geometry optimizers discovered through evaluator-grounded LLM autoresearch. The optimizers retain Sella's redundant-internal-coordinate foundation while reducing the number of expensive energy-and-force evaluations needed to reach a local minimum.

The accompanying manuscript is titled *Optimizing the Optimizer: Language Models Discover Faster Molecular Relaxation Algorithms*.

## Optimizers

The repository includes three frozen paper-selected implementations:


| Name        | File                        | Origin                                         |
| ----------- | --------------------------- | ---------------------------------------------- |
| AutoSella-G | `minimizers/autosella_g.py` | Grok 4.6 (xhigh) autoresearch result                     |
| AutoSella-A | `minimizers/autosella_a.py` | GPT-6 Astra (high) autoresearch result                |
| AutoSella-F | `minimizers/autosella_f.py` | Claude Fable 5.1 (max) autoresearch result |

`minimizers/sella_wrapper.py` provides the Sella 2.5.0 reference using the same callback, unit, convergence, and force-call accounting contract.

## Installation

The checked-in Conda environment covers the complete repository, including the molecule preparation pipeline and the evaluation dependencies:

```bash
conda env create -f environment.yml
conda activate autogeomopt
```

The lock file records the resolved environment used for reproducibility:

```bash
conda-lock install --name autogeomopt conda-lock.yml
```

Use `environment.yml` as the readable environment specification and the lock file for exact reproduction.

## Quick start

The included example optimizes a molecular geometry with AutoSella-G and GFN2-xTB:

```bash
python examples/optimize_molecule.py \
  examples/mol.xyz \
  --optimizer g \
  --output examples/mol_optimized.xyz
```

Select `g`, `a`, `f`, or `sella` to run another implementation under the same 200-force-call budget.

The XYZ comment line must contain the signed integer molecular charge in the exact form `charge=<integer>`. Coordinates in XYZ files are in angstrom, the optimizer callback uses nm, kJ/mol, and kJ/(mol nm).

## Repository layout

```text
.
├── minimizers/       # Frozen AutoSella implementations and Sella reference
├── molecules/        # Molecule construction, manifests, and baseline data
├── examples/         # Runnable single-molecule example and small XYZ input
├── environment.yml                # Conda environment file
├── conda-lock.yml                 # Exact Linux x86-64 environment lock
├── LICENSE                        # Project license summary
├── COPYING                        # GNU GPL v3 license text
├── COPYING.LESSER                 # GNU LGPL v3 additional terms
└── NOTICE                         # Attribution and third-party notices
```

The publication repository has seven branches:

- `main`: stable optimizers, benchmark construction, and documentation;
- `autoresearch-base`: shared starting optimizer, research harness, prepared data, and evaluation service for the autoresearch runs;
- `autoresearch-grok-46-xhigh`: Grok 4.6 (xhigh) run artifacts and recorded results;
- `autoresearch-astra-high`: GPT-6 Astra (high) run artifacts and recorded results;
- `autoresearch-fable-51-max`: Claude Fable 5.1 (max) artifacts and recorded results;
- `xtb_ff_evaluation`: held-out GFN2-xTB and GFN-FF evaluation;
- `dft_evaluation`: ORCA/OPI r2SCAN-3c evaluation.

## Molecule data and benchmark construction

The main benchmark combines SPICE PubChem, SPICE DES370K, and RowanSci structures. Source conformers are optimized directly with Sella/GFN2-xTB under a 200-force-call budget. Raw inputs are saved in `xyz/`. The deterministic split contains 225 PubChem groups in train and 250 in validation, DES dimers assigned by monomer-class pair, and accepted inputs from the 25-structure RowanSci benchmark in train.

Run the main pipeline with:

```bash
cd molecules
XTB_WORKERS=8 ./run_pipeline.sh
```

`XTB_WORKERS` controls DES370K and RowanSci parallelism, omit it to use the number of physical CPU cores. PubChem runs sequentially.

Each run verifies pinned source downloads and replaces its generated geometries, manifests, and splits. See [molecules/README.md](molecules/README.md) for the main pipeline and [molecules/evaluation/README.md](molecules/evaluation/README.md) for evaluation-set construction.

## Results

The study evaluates the selected optimizers on out-of-domain drug-like molecules and dipeptides using GFN2-xTB, GFN-FF, and r2SCAN-3c DFT. Summary and per-molecule evaluation records are provided on the evaluation branches.

## Reproducibility

- Every optimizer exposes the same `minimize_func` entry point.
- One force call is one callback invocation returning energy and forces.
- The default hard budget is 200 force calls.

## License and attribution

The repository's software is distributed under the GNU Lesser General Public License v3.0 only. See `LICENSE`, `COPYING`, and `COPYING.LESSER`.

The optimizer implementations are derived from and substantially modify Sella  2.5.0. Upstream attribution and a summary of modifications are provided in  `NOTICE`. Dataset artifacts retain their respective upstream terms;  the software license does not relicense SPICE, RowanSci, or other third-party data.
