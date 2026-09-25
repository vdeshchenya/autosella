# Prepared data

Consume the operator-approved immutable train/valid release under `releases/`.
Preparation and reference generation happen upstream; changes require a new
release ID.

`release.json` records provenance, public settings and units. Exact convergence
criteria are listed in [program.md](../program.md) and enforced by the evaluator.
The release's pinned checksum
inventory covers the payload. `train_XTB.json` and `valid_XTB.json` supply the
references; `xyz/<reference-key>.xyz` supplies each starting geometry.

XYZ coordinates use Angstrom and explicit integer `charge=` metadata. Reference
energies are total GFN2-xTB energies in Hartree, starting from the supplied
source SPICE geometry or noisy RowanSci geometry (mean displacement 0.02 Å).
There is no GFN-FF preoptimization in the active release. Final energy describes
the last evaluated state;
force-call counts and convergence flags are retained without filtering.

Source-declared settings alone do not establish numerical reproduction. Preserve
release identity and hashes in evaluation evidence. Verify the approved payload
with `python -m pytest tests/test_data_integrity.py`.

The active release is `main-f59f2b89a-gfn2-v1`: 469 train molecules
(225 PubChem, 219 DES370K, 25 RowanSci) and 465 validation molecules
(250 PubChem, 215 DES370K), with xTB 6.7.1 declared upstream.
Earlier immutable releases are retained for earlier evidence. The existing
upstream OOD dataset overlaps this new selection and must not be used as an
independent held-out set until it is rebuilt.
