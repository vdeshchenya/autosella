# Research log — geometry optimizer (GFN2-xTB, force-call cost)

Objective: reach the reference energy lowering (aggregate `mean_rel_energy >= 1 - 1e-9`)
with fewer force evaluations than the champion (`mean_rel_steps`, lower is better),
train gate improvement >= 1e-4, valid gate improvement > 1e-4 over the champion.

## Cycle 1 — anchor (Sella 2.5.0, internal=True, order=0)

- Commit `62ff882376f00637eae97beb66cbee96274c8a38`, release `main-f59f2b89a-gfn2-v1`.
- Train eval `ee46bb9e3c2e47fda01c290fbaad87da`: 469/469 converged, mean_rel_steps 1.0,
  mean_rel_energy 1.0000000000000049. Valid eval `ff95b9de79ba411ca0e4871058aa67ac`:
  465/465, mean_rel_energy 0.9999999999999895.
- Cost structure (from `anchors/train.json`, `anchors/valid.json`, computed locally):
  train mean 28.2 force calls (median 22, range 7–106); valid mean 30.5 (median 25, 7–155).
  By source (train): pubchem 19.9, rowansci 23.8, des370k dimers 37.3 (median 34).
  Valid dimers 38.5 with the most expensive cases (155, 136, 128, 123 calls).
  Dimers are ~46 % of the molecules and the dominant cost centre; single-molecule cost
  correlates weakly with size. Each force call saved on one molecule is worth
  ~1/(n_ref * N) ≈ 0.05/N in `mean_rel_steps`.
- Mechanism of the champion (read from `algo.py`): redundant internals
  (bonds/angles/dihedrals; dummy atoms for near-linear angles), Fischer–Almlöf diagonal
  guess Hessian projected into the internal space, TS-BFGS update each step,
  quasi-Newton step with |eigenvalue| Levenberg shift and a max-internal-step trust
  region (delta0 = 0.1, grow ×1.15 for good steps, shrink ×0.9, no step rejection),
  LSODA back-transformation; disconnected fragments are joined by growing the
  covalent-radius scale by 5 % per pass until a single connected graph results
  (`allow_fragments=False`), so intermolecular contacts become long pseudo-bonds with
  angles/dihedrals through them (H-bonds are near-linear → dummy-atom machinery).
- Sources consulted:
  - geomeTRIC "How it works" (TRIC, guess Hessian, trust radius, BFGS reset):
    https://geometric.readthedocs.io/en/latest/how-it-works.html
  - L.-P. Wang, C. Song, "Geometry optimization made simple with translation and rotation
    coordinates", J. Chem. Phys. 144, 214108 (2016), https://doi.org/10.1063/1.4952956
  - M. Swart, F. M. Bickelhaupt, "Optimization of strong and weak coordinates",
    Int. J. Quantum Chem. 106, 2536 (2006), https://doi.org/10.1002/qua.21049
  - A. Shajan et al., "Geometry Optimization: A Comparison of Different Open-Source
    Geometry Optimizers", J. Chem. Theory Comput. 19, 7533 (2023),
    https://doi.org/10.1021/acs.jctc.3c00188 (Baker set: ASE/Sella 187 steps best,
    geomeTRIC 287, exact-Hessian reference 111).
  - E. D. Hermes et al., "Sella, an Open-Source Automation-Friendly Molecular Saddle Point
    Optimizer", J. Chem. Theory Comput. 18, 6974 (2022),
    https://doi.org/10.1021/acs.jctc.2c00395
  - ORCA manual, geometry optimization defaults (Almlöf model Hessian, BFGS, RFO,
    MaxStep 0.3 au): https://www.faccts.de/docs/orca/6.0/manual/contents/typical/geoopt.html
  - R. Lindh et al., "On the use of a Hessian model function in molecular geometry
    optimizations", Chem. Phys. Lett. 241, 423 (1995),
    https://doi.org/10.1016/0009-2614(95)00646-L

## Cycle 2 — fragment translation/rotation internals (`allow_fragments=True`)

### Hypothesis
Sella already implements the TRIC idea of Wang & Song (2016): with
`allow_fragments=True` each disconnected fragment gets 3 centroid translations and
3 rotation (exponential-map) coordinates with guess curvature 0.05 Ha, instead of being
stitched together by long inter-fragment pseudo-bonds (found by inflating the covalent
radii) plus all angles/dihedrals through them. For the des370k dimers (46 % of the set,
mean 37–38 reference calls versus ~20 for PubChem molecules) the pseudo-bond internals
are ill conditioned (near-linear D–H···A hydrogen bonds trigger the dummy-atom
machinery, and the intermolecular "bond" gets a covalent-strength Almlöf curvature that
is 10–100× too stiff), so the quasi-Newton path takes many small steps. Explicit
translation/rotation coordinates should reduce the dimer cost substantially while
leaving single-fragment molecules on an identical path (the fragment code is only
reached when the bond graph is disconnected; no PubChem entry is multi-fragment).

### Change
`minimize_func`: `Sella(atoms, internal=True, order=0, logfile=None, allow_fragments=True)`.
One-argument change; everything else identical to the champion.

### Control
The single-fragment molecules (PubChem, RowanSci) should reproduce the anchor step
counts exactly; any difference there would indicate an unintended code path.

### Result
- Candidate commit `a5b0bd56c0434408dc4c079157fd506862f43f59` vs champion
  `62ff882376f00637eae97beb66cbee96274c8a38`. Decision: **invalid**
  (train `internal_error`, 1 molecule). Evidence: `evaluation_results/cycle-2-train.json`
  (eval `583056e232854da0b0e024e20cd148a0`), diagnostics
  `diagnostics/583056e232854da0b0e024e20cd148a0/13fece29….json`, `results.tsv` row 2.
- Per-molecule comparison against `anchors/train.json` (computed locally): rowansci
  25/25 identical step counts; pubchem 224/225 identical (control holds — the fragment
  code path is only reached for disconnected graphs); dimers 37.3 → 26.1 mean calls
  (rel 0.79), overall rel-steps 0.90 on the 468 completed molecules.
- Energy gate would also have failed: mean rel_energy 0.9968 (< 1 − 1e-9).
  Sum(rel−1) = −1.52: 15 dimers lost > 1e-3 each (total −2.84; e.g.
  `des370k_amines__amines__train` converged after 6 calls 2.87 kcal/mol above the
  anchor's 99-call result, rel 0.12; `des370k_ammoniums__ketones__train` 11 calls,
  +1.19 kcal/mol), 13 dimers gained (total +1.32, e.g. `des370k_monoatomics__pyrrole__train`
  −2.49 kcal/mol, rel 1.73).
- Failure: `des370k_monoatomics__thiols__train` (Cl⁻ + dimethyl disulfide, Cl⁻ 6 Å away).
  The 7th force call raised an xtb SCC non-convergence (HOMO–LUMO gap 0.0075 Eh: Cl⁻ 3p
  vs S–S σ* near-degeneracy). The trajectory itself was tame (max atomic displacement
  0.42 Å, S–S 2.049 → 2.080 Å) but Cl⁻ moved only 0.2 Å in 6 calls: with the constant
  TRIC guess curvature 0.05 Ha/Å² and a Coulomb force of ~6e-4 Ha/Å the translation
  step is ~0.01 Å, so the ion lingers far away while the disulfide relaxes in an
  electronically unstable region. The anchor's inflated pseudo-bond (Almlöf curvature
  ~1e-8 at r − rcov = 4.6 Å) instead lets Cl⁻ approach at the trust-radius limit and
  bind (final Cl···H 2.16 Å, 8.6 kcal/mol) in 27 calls.

### Interpretation
The direction is sound (dimer cost −21 %, single-molecule control exact) but the
constant 0.05 Ha guess for fragment translations/rotations is the wrong stiffness for
intermolecular degrees of freedom: for dispersion-bound or far-apart fragments it is
10–1000× too stiff, so the intermolecular steps are tiny and the evaluator's
displacement/energy criteria declare convergence while the dimer is still far from the
minimum the anchor finds (energy gate loss), and ions approach too slowly (SCC failure
exposure). Under a displacement-based external convergence test, guesses that err on
the soft side are safer: soft guesses give trust-radius-limited steps that BFGS
corrects in one or two updates, whereas stiff guesses stall.
Alternative explanation for the early stops: symmetric starting arrangements where the
TRIC gradient vanishes exactly. Not testable from the retained evidence (no bundles for
converged molecules); the stiffness mechanism explains both the early stops and the
slow Cl⁻ approach, so it is the repair target.

### Next idea/control
Repair 1 (cycle 3): keep `allow_fragments=True` but derive the guess curvature of every
fragment translation/rotation coordinate from the actual inter-fragment contacts
(sum over inter-fragment atom pairs of the Almlöf stretch curvature k_ij(r) projected
on the rigid-body displacement of the fragment), so that far-apart or dispersion-bound
fragments get soft coordinates and ion pairs get stiff ones. Control: single-fragment
molecules must again match the anchor exactly.

## Cycle 3 — repair 1 of tric-allow-fragments: contact-derived curvature for fragment coordinates

### Hypothesis
Cycle 2 showed that TRIC fragment coordinates cut dimer cost by 21 % but that the
constant 0.05 Ha guess curvature is far too stiff for distant or dispersion-bound
fragments (tiny intermolecular steps → premature external convergence → energy-gate
loss; slow ion approach → SCC-failure exposure). Deriving each fragment
translation/rotation curvature from the actual inter-fragment contacts,
h_q = Σ_{i∈F, j∉F} k_ij(r) (u_ij · ∂x_i/∂q)², with the Almlöf stretch curvature
k_ij(r) = 0.3601·exp(−1.944 (r − rcov)/Bohr) Ha/Bohr² already used for bonds, gives
soft coordinates (trust-radius-limited steps, as the anchor's inflated pseudo-bonds
effectively had) when the fragments are far apart and stiff ones for ion pairs and
hydrogen bonds. Floor 1e-3 Ha. Expected: the energy gate is restored (no early stops),
the Cl⁻ case approaches at the trust-radius limit like the anchor, and the dimer step
reduction is retained or improved; single-fragment molecules unchanged.

### Change
`Internals._h0_fragment` (new), `Internals.guess_hessian` (translations/rotations use
it when `allow_fragments`), `minimize_func` (`allow_fragments=True`).

### Result (cycle 3)
- Candidate `13cbc5cca5ed795a1793d0eb993a685d1e0dec4b` vs champion
  `62ff882376f00637eae97beb66cbee96274c8a38`. Decision: **keep** (new champion).
  Train eval `162f84e0677c4f6f83ed1d9c92cb3ea1`: 469/469, mean_rel_steps 0.9389,
  mean_rel_energy 1.0000848 (`evaluation_results/cycle-3-train.json`). Valid eval
  `d96038c965ab481092aaaa24dee58c6a`: 465/465, mean_rel_steps 0.9600,
  mean_rel_energy 1.00565 (`evaluation_results/cycle-3-valid.json`); `results.tsv` row 3.
- Per source (local comparison with `anchors/*.json`, script `/tmp/compare.py`):
  train dimers 37.3 → 27.1 mean calls (rel 0.866), pubchem 224/225 identical
  (`104079126` 41 → 66: a mid-run re-initialization after `check_for_bad_internals`
  finds a disconnected graph at that geometry and switches to fragment coordinates),
  rowansci identical; valid dimers 38.5 → 27.9 (rel 0.913), pubchem all identical.
- Energy gate: train sum(rel−1) = +0.040 only (losers: `alkanes__carboxylates` −0.33,
  `pyridine__water` −0.28, `carboxylates__water` −0.21 — the anchor's long wandering
  runs reached deeper minima; gainers: `carboxylates__ethers` +0.36,
  `ketones__monoatomics` +0.23). Valid sum +2.63, dominated by
  `carboxylates__ketones__valid` (+3.2, a 12.9 kcal/mol deeper minimum, likely a proton
  transfer) against water-dimer losses (`amides__water` −0.35 at 10 kcal/mol,
  `ethers__water` −0.41, `esters__water` −0.41).
- Cost regressions where the fragment path finds a different, deeper minimum through a
  long trajectory: `carboxylates__ketones__train` 9 → 51, `ammoniums__esters` 19 → 66,
  `amides__ethers` 36 → 77.

### Interpretation (cycle 3)
The contact-derived curvature fixed both diagnosed failure modes: no SCC failure
(the Cl⁻ case converged in 19 calls vs 27) and the energy gate passed. The remaining
dimer cost (27 calls) still exceeds the single-molecule cost (20) and the energy margin
on train is thin (+8.5e-5 in the mean), so later changes must not trade minima depth
for steps on dimers. Uncertainty: the per-molecule energy outcome on flat dimer
surfaces is path-dependent and effectively random with respect to small changes;
the gate is protected mainly by the ion-pair gains.

## Cycle 4 — textbook trust-region policy (expand ×2 for ρ > 0.75, shrink ×0.5 for ρ < 0.25, cap 1.0)

### Hypothesis
Anchor step counts for PubChem molecules correlate strongly with the size of the
geometry change (corr 0.80 with the max aligned atomic displacement; molecules with
dmax > 0.8 Å need 27 calls on average versus 8–12 for dmax < 0.4 Å, computed from
`anchors/train.json`), the signature of a trust-radius-limited approach phase: Sella's
max-internal-step radius starts at 0.1 (Å or rad) and grows only ×1.15 per step, and
only when 0.75 < ρ < 1.33, so a 60° torsion change alone needs ~7 boundary steps.
The standard policy (Nocedal & Wright ch. 4; geomeTRIC uses √2 growth for Q > 0.75,
halving for Q < 0.25) doubles the radius after any step with ρ > 0.75 — including
ρ > 1.33, where the model underestimates the decrease because the guess Hessian is
too stiff — halves it after ρ < 0.25 (including energy increases, which Sella barely
penalised: ×0.9 only for ρ < 0.01), and caps it at 1.0. Expected: fewer steps in the
far-from-minimum phase for all sources; risk: larger steps change which dimer minimum
is found (energy gate) and bad steps are more frequent.

### Change
`_default_kwargs['minimum']` (sigma_inc 2.0, sigma_dec 0.5, rho_dec 4.0, new
delta_max 1.0), `Sella.__init__` (delta_max kwarg), `Sella.step` (trust update rule).

### Result (cycle 4)
- Candidate `617c6f77a320cdbb904b5b02f50309e8a2afe736` vs champion
  `13cbc5cca5ed795a1793d0eb993a685d1e0dec4b`. Decision: **invalid** (train
  `internal_error`, 1 molecule). Evidence: `evaluation_results/cycle-4-train.json`
  (eval `72011b44b9554327b47eccb17e4021a9`), `results.tsv` row 4.
- Crash: `des370k_alkenes__phenol__train`, `IndexError` in `Internals._h0_fragment`
  (cycle-3 code) during a mid-run re-initialization: `add_dummy_to_internals` appends a
  dummy-atom index to the fragment's Translation/Rotation coordinates, but
  `_h0_fragment` indexed `self.atoms.positions` (real atoms only). This is a latent
  contract bug of the current champion (not triggered on train/valid in cycle 3 because
  it needs a fragment + a near-linear angle at a re-initialization); it must ride
  along with the next candidate.
- Cost/energy on the 468 completed molecules vs the cycle-3 champion
  (`/tmp/compare.py`): overall rel-steps 1.039 (worse); pubchem 20.04 → 19.97
  (rel 0.996; 82 faster, 91 identical, 52 slower — flat across all geometry-change
  bins, from dmax < 0.2 Å to > 1.5 Å); rowansci 23.8 → 23.9; dimers 27.1 → 29.1
  (rel 1.087); mean rel_energy 0.99933 (sum −0.31, gate would fail).

### Interpretation (cycle 4)
The hypothesis is contradicted: the trust radius is not what limits the approach
phase of far-from-minimum PubChem molecules (doubling growth and adding a proper
shrink changed their cost by < 1 % in every displacement bin). The limiting factor
must be the accuracy of the curvature model along the path (soft torsional modes;
endgame under tight displacement criteria), not the step bound. On dimers the policy
hurt: on flat intermolecular surfaces ρ is noisy, so the ×0.5 shrink for ρ < 0.25
collapses the radius and the larger steps also alter which minimum is reached
(energy loss). Direction closed for now; a growth-only variant (no ×0.5 shrink)
is the only untested part and is expected to be roughly neutral.

### Next idea/control
Cycle 5: fix the dummy-index bug in `_h0_fragment` (use `all_positions`, restrict
contacts to real atoms) and test the Hessian update for minimization: BFGS whenever
the curvature condition sᵀy > 0 and sᵀBs > 0 hold (fall back to TS-BFGS otherwise),
replacing the unconditional TS-BFGS update. Control: molecules whose paths never hit
the fallback should differ from the champion only through the update formula.

## Cycle 5 — BFGS update for minimization (BFGS_auto with a subspace curvature test) + dummy-index repair of `_h0_fragment`

### Hypothesis
The champion updates the internal-coordinate Hessian with Bofill's TS-BFGS
(`ApproximateHessian` default), a compromise formula designed for saddle searches
that does not preserve positive definiteness and, when the model curvature along the
step is wrong, weights the `|B|s` direction differently from BFGS
(u ∝ y + (sᵀBs/sᵀy)·Bs instead of y + √(sᵀy/sᵀBs)·Bs). For pure minimization the
BFGS update is the standard choice (Gaussian/Berny, ORCA, geomeTRIC, Schlegel's
reviews) because it keeps the model positive definite whenever sᵀy > 0 and converges
superlinearly on the soft modes that dominate the tight endgame here. Expectation:
fewer force calls mainly on single-molecule PubChem/rowansci entries (endgame on soft
torsions), neutral-to-positive on dimers; risk: a different path on flat dimer
surfaces can change which minimum is reached (energy gate), as in cycle 4.
Sella's own `BFGS_auto` option could never select BFGS in internal coordinates
because it tests `np.all(eigvals(B) > 0)` on the projected H0 = P·H·P, which has a
null space; the test is replaced by generalized eigenvalues of SᵀY and SᵀBS w.r.t. SᵀS
(tolerance 1e-10), falling back to TS-BFGS when the curvature condition fails.

### Change (functions)
- `update_H` (`BFGS_auto` branch): curvature test on the step subspace as described;
  new keyword `bfgs_tol=1e-10`.
- `ApproximateHessian.__init__`: default `update_method='BFGS_auto'` (propagates through
  `project`, `__add__`, `PES.set_H`, `get_HL_projected`, including re-initialized PES
  objects after `check_for_bad_internals`).
- `Internals._h0_fragment` (repair of the cycle-4 crash, a latent bug of the champion):
  positions taken from `all_positions`; fragment indices restricted to real atoms
  (`idx_all < natoms`) for the contact sum; `mask` sized `natoms`; rotation centroid
  over all coordinate indices including dummies (as in `Rotation`); returns the floor
  if the fragment has no real atoms.
Control: single-molecule entries never touch `_h0_fragment`, so their change is the
update formula alone; the champion's behaviour is recovered only if the fallback
fires on every step (it should almost never fire on a descent path).

### Result (cycle 5)
- Candidate `e0360bc9b6d8a288b2a5d4636bbcf45c3f990653` vs champion
  `13cbc5cca5ed795a1793d0eb993a685d1e0dec4b`. Decision: **invalid**
  (`energy_below_baseline`, mean rel_energy 0.99993; C_S train 1.0545 vs 0.9389,
  i.e. −0.116). Evidence: `evaluation_results/cycle-5-train.json`
  (eval `458dcd4faaff48748f6fa1533a121af3`), `results.tsv` row 5, diagnostics
  `diagnostics/458dcd4faaff48748f6fa1533a121af3/` (molecule 8030412, 200 calls).
- Per source vs champion (`/tmp/compare.py` + distribution script): pubchem
  20.04 → 22.28 (median ratio 1.037; 6 % faster, 41 % identical, 53 % slower; still
  +4 % excluding the two blow-ups 8030412 34 → 200 and 135316214 33 → 143); dimers
  27.05 → 32.54 (84 % slower, 8 molecules ≥ 2×: amides__ethers 77 → 175,
  phenol__phenol 37 → 105, imidazolium__ketones 56 → 106, amides__pyridine 35 → 98);
  rowansci 23.8 → 24.8. Ratio grows with the anchor's cost (median 1.00 for runs
  < 15 calls, 1.08–1.10 above 20 calls).
- Trace of 8030412: after call 41 the run is frozen (max force 1.1e-3 Eh/Bohr constant,
  displacements 1e-10 Bohr for 150 calls) — the BFGS `y yᵀ/(yᵀs)` term with a tiny
  positive yᵀs produced enormous curvature and the Newton step vanished; earlier
  (calls 14–41) the run crept down 35 kJ/mol with max force ≈ 1e-3.

### Interpretation (cycle 5)
Contradicted: within Sella's machinery (parallel-transported gradients, redundant
internals, MIS trust region without step rejection) TS-BFGS is clearly the better
minimization update; BFGS is worse on the typical molecule (not only in the tails) and
occasionally catastrophic. The |B|s-weighted secant direction of TS-BFGS acts like a
damped BFGS, tolerating secant pairs with poor curvature (large steps in flat
intermolecular modes, noisy endgame gradients). A Powell-damped BFGS might remove the
stalls but the systematic +5 % suggests the update formula is not where cost is lost;
direction closed. Curvature information is more likely gained by extrapolating over
several past points (GDIIS) or by a better initial model (Lindh/Swart).

### Next idea/control
Cycle 6: controlled GDIIS (Farkas & Schlegel 2002) on top of the champion's QN/TS-BFGS
step, active only when the QN step is not trust-limited, with the standard safeguards;
`_h0_fragment` repair rides along.

## Cycle 6 — controlled GDIIS endgame (Farkas & Schlegel 2002) on the QN/TS-BFGS step + `_h0_fragment` repair

### Hypothesis
Under the tight displacement criterion (1.8e-3 Bohr) the endgame is limited by the
accuracy of the soft-mode curvature: a step along a soft mode with a 30 % curvature
error leaves 30 % of the step as residual, so the last force calls shrink the residual
geometrically. GDIIS (Császár & Pulay 1984) extrapolates over the last few points and
gradients — effectively a secant refinement along the recently sampled directions — and
Farkas & Schlegel (PCCP 4, 11 (2002)) showed that with their safeguards it reduces the
number of steps of a BFGS/RFO optimizer by roughly 10–30 %, most for floppy molecules,
which are the expensive PubChem/rowansci entries here. Expected: fewer calls on runs
with ≥ 15 calls, no change on runs that are trust-limited until the end; risk: altered
paths on flat dimer surfaces (energy lottery, cf. cycles 4–5).

### Change (functions)
- `Sella.__init__`: `gdiis=True, gdiis_nmax=5, gdiis_cmax=10` (active for order 0 only);
  history `self._diis_hist` of (internal x, internal g) per point.
- `Sella._predict_step`: keeps the restricted-step object and, after the QN step, calls
  `_gdiis_step`.
- `Sella._gdiis_record`: appends `pes.curr['x'], pes.curr['g']` (continuous dihedral
  branch), clears the history if the internal set size changed, keeps ≤ 5 points.
- `Sella._gdiis_step`: only when the QN step is not trust-limited (`smag < delta`) and
  the stepper is the QN stepper; error vectors e_i = −V (Vᵀ P (g_i + H·scons))/|λ| in the
  free subspace of the current point (same Hessian eigen-decomposition as the QN step);
  DIIS coefficients from the KKT system (scaled Gram matrix, lstsq); step
  = Ufree (Σc_i P(x_i − x_n) + Σc_i e_i) + scons. Safeguards: c_n > 0, Σ|c_i| ≤ 10,
  cos(GDIIS, QN) ≥ {2: 0.97, 3: 0.84, 4: 0.71, 5: 0.67, ≥6: 0.62}, |GDIIS| ≤ 10|QN|;
  otherwise the oldest point is dropped and the test repeated; the accepted step is
  scaled to the MIS trust radius if it exceeds it.
- `Sella.step`: history cleared on internal-coordinate re-initialization.
- `Internals._h0_fragment`: dummy-index repair (as in cycle 5).
Control: runs whose QN steps are trust-limited until convergence, and the first two
force calls of every run, are unchanged.

### Result (cycle 6)
- Candidate `ec3782e85afea3ee86f0a448002d4354345fb6d7`, champion `13cbc5cca5ed795a1793d0eb993a685d1e0dec4b`.
- Train (evaluation `74bee239890544298a6ab48cc4b4691c`,
  [cycle-6-train.json](evaluation_results/cycle-6-train.json)): 469/469 converged,
  C_S = 0.93860 (champion 0.93888; improvement 0.00028, below the 1e-4 gate only by
  magnitude, it would have passed), but mean rel_energy 0.99935 → **invalid**
  (`energy_below_baseline`). Decision: invalid ([results.tsv](results.tsv) cycle 6).
- Per source (mean calls, candidate vs champion): PubChem 20.16 vs 20.04, dimers 27.03
  vs 27.05, rowansci 23.64 vs 23.80. Distribution: runs with ≥ 60 champion calls fell
  71.4 → 61.2 (amides__ethers 77 → 53, carboxylates__esters 66 → 35,
  carboxylates__ketones 51 → 35, benzene__phenol 28 → 17), but the mid range (25–60
  calls) was slightly worse and a few runs blew up (carboxylates__water 20 → 57,
  amides__guanidiniums 24 → 52).
- Energy: 14 molecules changed rel_energy by > 1e-4; the loss is two runs:
  `des370k_ammoniums__ketones__train` 1.000 → 0.682 (48 → 12 calls, +1.19 kcal/mol,
  converged at a much shallower stationary point) and `404345632` 1.000 → 0.926
  (41 → 38 calls, different minimum); small losses acids__alcohols, acids__sulfides,
  252618428; gains amides__ethers 1.070 → 1.000, carboxylates__ketones 1.103 → 1.199.
- No diagnostics artifacts (all runs converged), so the two energy losers cannot be
  traced.

### Interpretation
The hypothesis that a secant-like extrapolation over the last points shortens the
endgame is not supported: overall cost is unchanged (−0.03 %), and the gains are
confined to a handful of very long runs where the QN/TS-BFGS iteration was evidently
stalling, offset by mid-range runs that got slightly longer. The likely reason is that
the TS-BFGS update already imposes the secant condition along the last step, so the
one-Hessian GDIIS error vectors add little new curvature information; what GDIIS adds
is a different path, which on flat dimer surfaces is again an energy lottery (two
runs converged early at shallower points, exactly the failure mode of cycles 4–5).
Alternatives: (a) the safeguards may be too loose for the flat intermolecular modes
(the 12-call ammoniums__ketones run suggests a DIIS step that landed on a point with
tiny forces and tiny displacement, ending the run) — a repair would restrict GDIIS to
runs/points where the gradient is already small or exclude fragment modes; (b) the
subspace (5) may be too large for a 20-call run. Uncertainty: single train evaluation;
the two energy losers dominate the gate margin (sum rel−1 = −0.31), so a repair that
only fixes them would still be cost-neutral, i.e. would not pass the 1e-4 cost gate
with any margin. Not worth more than one repair attempt.

### Next idea/control
Repair 1 of `gdiis-endgame` (if attempted): disable GDIIS while any fragment
translation/rotation coordinate has a non-negligible gradient and require the DIIS
step to be at least 0.25× the QN step; control = molecules with a single fragment
(PubChem/rowansci) whose paths would be unchanged by the fragment rule. Otherwise move
to the model-Hessian idea (Lindh/Swart pairwise model, which also covers
intermolecular contacts consistently) and to Hessian carry-over at re-initialization.
Champion restored to `13cbc5cca5ed795a1793d0eb993a685d1e0dec4b`.

## Cycle 7 — share the Fischer–Almlöf torsional constant over the redundant dihedrals of a bond (1/√n scaling) + `_h0_fragment` repair

### Hypothesis
Evidence so far says the approach phase is model-limited, not trust-limited (cycle 4
null), and the endgame is limited by soft-mode curvature accuracy (cycles 5–6). The
Fischer–Almlöf torsional formula (`_h0_dihedral`) was fitted as the curvature of the
rotation about a bond; Sella assigns that full value to every one of the
n = (n_b − 1)(n_c − 1) redundant dihedrals around the bond, so a rigid rotation, which
changes all n dihedrals together, sees n·h: for an sp³–sp³ C–C bond (n = 9,
h ≈ 0.0088 Ha/rad²) the model rotation constant is ≈ 0.08 Ha/rad² against ≈ 0.02 for
a 3 kcal/mol threefold barrier; C–O and C–N single bonds are 2–4× too stiff as well,
while double bonds (n = 4, shorter r) are about right. A too-stiff torsional model
makes the first quasi-Newton step along every soft torsion 2–4× too short, and the
rank-2 TS-BFGS update repairs only the direction just stepped, so floppy molecules
pay roughly one extra force call per torsional direction. Scaling each dihedral's
guess by 1/√n brings the model rotation constants of single bonds within ~2× of the
physical values while leaving double bonds only moderately softer. Prediction: fewer
calls for PubChem/rowansci molecules and for dimers with intramolecular relaxation,
unchanged behaviour for rigid dimers; risk: overshoot on softened ring/double-bond
torsions and the usual energy lottery on flat dimer surfaces. Also carries the
`_h0_fragment` dummy-index repair from cycles 5–6 (latent IndexError of the champion).

### Change (functions)
- `Internals.guess_hessian`: counts the proper (non-dummy) dihedrals per central bond
  and divides each `_h0_dihedral` value by √n; dummy dihedrals keep 0.5 Ha.
- `Internals._h0_fragment`: dummy-index repair (indices ≥ natoms restricted to real
  atoms for the contact sum; rotation centroid over all indices).
Control: molecules with no dihedrals (atoms, diatomics, some monoatomic-ion dimers)
are unchanged; rigid dimers change only through the intra-fragment torsional guess.

### Result (cycle 7)
- Candidate `6a301d6b92824d60a500ccf101e7772a07b7a871`, champion `13cbc5cca5ed795a1793d0eb993a685d1e0dec4b`.
- Train (evaluation `7e73f6c0de864c77b4c98eaab9cf7dfb`,
  [cycle-7-train.json](evaluation_results/cycle-7-train.json)): 469/469 converged,
  valid (mean rel_energy 1.00034), but C_S = 0.95112 vs 0.93888 → **discard**
  (train improvement −0.0122).
- Mean calls fell (PubChem 20.04 → 19.72, rowansci 23.80 → 22.76, dimers 27.05 →
  27.20) but the mean *ratio* rose because short runs got longer while long runs got
  shorter: PubChem anchor < 20 calls: ratio 1.04–1.05 (frac slower 0.23–0.42), anchor
  ≥ 20: 0.91–0.96 (frac faster 0.58–1.00); dimers < 20 calls: 1.08–1.15, ≥ 45: 0.87.
  Best: amides__ethers 77 → 39, esters__ethers 43 → 23, amides__phenol 61 → 42,
  160853090 89 → 74; worst: carboxylates__guanidiniums 14 → 30, 135094180 15 → 38,
  amides__guanidiniums 24 → 48.
- Split by chemistry (short heavy-atom bonds r < 0.92 Σr_cov as a double/aromatic
  proxy): molecules with no short bonds improved in every bin (PubChem < 20 calls:
  0.986, ≥ 20: 0.914; dimers 1.011 / 0.976), molecules with short bonds got worse
  for short runs (PubChem 1.065, dimers 1.125) and gained less for long runs
  (0.962 / 1.023); slowdown grows with the fraction of short bonds.
- Energy: net +0.15 (gainers carboxylates__ethers 1.356, ketones__monoatomics 1.233;
  losers alkanes__carboxylates 0.673, pyridine__water 0.723, carboxylates__water 0.797).

### Interpretation
The redundancy over-stiffening is real: wherever the torsions are sp³ single bonds,
sharing the constant over the n dihedrals cuts the cost (−1.4 % to −9 %), most on
long runs where the first torsional steps were 2–4× too short. But the uniform 1/√n
factor also softens double-bond and aromatic torsions, for which the Almlöf value
per dihedral was already about right (n = 4 and short r): model rotation constants
fell from 0.12 → 0.06 (C=C, physical ≈ 0.19) and 0.076 → 0.038 Ha/rad² (aromatic), so
near-minimum molecules with rings overshoot along ring/conjugated torsions and pay
extra endgame steps, and the C_S mean of ratios punishes these lengthened short runs
more than it rewards the shortened long ones. Alternatives considered: (a) the trust
radius interacting with the larger first steps — not supported, the slowdown tracks
the short-bond fraction, not the run length alone; (b) the energy lottery — the
energy side is positive here (+0.15), not the cause. Uncertainty: one train
evaluation; the bond-length proxy for bond order is crude.

### Next idea/control
Repair 1 of `torsion-redundancy`: keep the 1/√n sharing but multiply the torsional
guess by the bond-order factor exp(−Ct (r − r_cov)/Bohr) (the same exponential the
Almlöf formula already uses once), which restores the champion's stiffness for
short (double/aromatic/amide/ester) bonds — C=C 0.16, aromatic 0.072, amide 0.095,
ester 0.04 Ha/rad² per rotation — while sp³–sp³ and C–O/C–N single bonds stay at the
softened 0.02–0.025. Control: molecules without short bonds should keep the cycle-7
gains; molecules with short bonds should return to ≈ champion cost or better.

## Cycle 8 — repair 1 of torsion-redundancy: bond-order factor on the torsional guess (1/√n sharing kept) + `_h0_fragment` repair

### Hypothesis
Cycle 7 showed that sharing the Almlöf torsional constant over the n redundant
dihedrals helps every molecule without short bonds and hurts molecules with
double/aromatic bonds, whose torsions were already about right before the sharing.
Multiplying the torsional guess by the bond-order factor exp(−Ct (r − r_cov)/Bohr)
(Ct = 2.85, the exponential the formula already contains once) restores the
champion's stiffness for short bonds (model rotation constants: C=C 0.12 → 0.16,
aromatic 0.076 → 0.072, amide 0.10 → 0.095, ester 0.037 → 0.04 Ha/rad²) while the
sp³–sp³ and single C–O/C–N rotations keep the softened values (C–C 0.079 → 0.025,
methanol C–O 0.036 → 0.019). Prediction: the no-short-bond molecules keep the
cycle-7 gains (≈ −1.5 % to −9 % by bin) and the short-bond molecules return to about
the champion's cost, giving C_S below the 0.9379 gate; risk: stretched single bonds
(r ≫ r_cov) become very soft, and the energy lottery on flat dimer surfaces.

### Change (functions)
- `Internals._h0_dihedral`: returns `h0 * bo` with `bo = exp(−Ct (r_bc − r_cov)/Bohr)`.
- `Internals.guess_hessian`: 1/√n sharing (cycle 7); `Internals._h0_fragment`:
  dummy-index repair (cycles 5–7).
Control: molecules without dihedrals are unchanged; for molecules whose bonds all have
r ≈ r_cov the change reduces to cycle 7.

### Result
Candidate `c4972f1e0c508fafee12a2fdbeceb506786dd0d6` vs champion
`13cbc5cca5ed795a1793d0eb993a685d1e0dec4b`, train evaluation
`830f0fd8164e4765bc7d78936b8cd268` ([JSON](evaluation_results/cycle-8-train.json)):
469/469 converged, C_S = **0.9094** (champion 0.9389, −3.1 %, largest cost gain of
the project) but mean rel_energy 0.99962 → decision **invalid**
(`energy_below_baseline`). Cost by source: PubChem 0.965, rowansci 0.913, dimers
0.996 (fraction faster 0.49/0.68/0.45); by PubChem size: < 15 atoms 1.001, 15–25
0.970, 25–40 0.948, 40–80 0.909, ≥ 80 0.851 (2 molecules). The short-bond
molecules returned to the champion's cost as predicted (cycle 7's +6–13 % on
short-run molecules with double/aromatic bonds is gone). Best runs: amides__ethers
77 → 37, acalabrutinib 38 → 20, carboxylates__esters 66 → 35, 135095297 36 → 21.
Energy changes vs the champion (all other molecules identical to 1e-3): the seven
changes shared with cycle 7 sum to +0.17 (carboxylates__ketones +0.095, 135249644
+0.079, sulfides__thiols +0.042, amides__guanidiniums +0.018, carboxylates__water
+0.003, amides__ethers −0.070 (the champion's 1.070 basin is not found)) and two
new ones: **ketones__ketones 1.000 → 0.773** (38 → 31 calls, 2.7 kcal/mol) and
**135043047 1.000 → 0.839** (35 calls; 38 kcal/mol). Total Σ(rel − 1) = −0.18
(champion +0.04). No diagnostics artifacts (all runs converged).

### Interpretation
- ketones__ketones is a formaldehyde dimer with no proper dihedral at all. Its
  only internal coordinate touched by this cycle is the *improper* dihedral Sella
  adds at each three-neighbour centre without proper dihedrals (the umbrella
  mode); the bond-order factor applied to its "central bond" (a C–H, 0.85×; for
  a nitrate N–O it would be 2.2×) is an unintended side effect of repair 1, and
  that 15 % change of one umbrella curvature was enough to steer the dimer into a
  second, 2.7 kcal/mol shallower basin (centroid distance 3.59 Å instead of 2.82;
  the same basin cycle 4 found). Cycle 7, which left impropers untouched (n = 1),
  reproduced the champion's result bit for bit on this molecule, so the
  evaluator is deterministic and the flip is a code effect, not noise.
- 135043047 is C6N6O6 (benzotrifuroxan-like); three N–O bonds form during the
  run (235 kcal/mol). The champion closes all three, the candidate only two
  (0.839): the softened proper torsions changed the reactive path. No general
  fix within this idea; a bond-formation-aware rebuild of the internals is a
  separate backlog item.
- The other energy changes are exactly cycle 7's (same molecules, same values),
  i.e. they come from the 1/√n sharing and net +0.17; they also show that the
  paths of these flat dimer surfaces are reproducible and only flip under code
  changes that touch their coordinates.
- Alternatives: the improper side effect could also be "neutral" and the flip
  a chaotic response — but cycle 4 (different code, same shallow basin) shows
  the two basins are the only attractors, so removing the improper perturbation
  should return this molecule to the champion's path deterministically.
- Uncertainty: the energy gate after the repair rests on a thin margin: champion
  +0.04, cycle-7 changes +0.17, 135043047 −0.16 → ≈ +0.05, i.e. any further
  adverse basin flip of ≥ 0.05 fails it again. The valid set's champion margin
  is +2.6 but +3.2 of it is one molecule (carboxylates__ketones__valid, 4.2×).
The champion is restored (`git reset --hard 13cbc5c`); the implementation is
preserved at
[ideas/torsion-redundancy/c4972f1e0c508fafee12a2fdbeceb506786dd0d6/algo.py](ideas/torsion-redundancy/c4972f1e0c508fafee12a2fdbeceb506786dd0d6/algo.py).

### Next idea/control
Repair 2 of torsion-redundancy (cycle 9): apply the bond-order factor only to
proper dihedrals (central bond b–c with d bonded to c); impropers keep the
Fischer–Almlöf constant exactly as in the champion (the 1/√n sharing never
touched them, n = 1). Control: molecules without proper dihedrals must reproduce
the champion's calls and energies exactly (ketones__ketones 38/1.000);
success = mean rel_energy ≥ 1 and C_S ≤ 0.9379, expected ≈ 0.910.

## Cycle 9 — repair 2 of torsion-redundancy: bond-order factor and 1/√n sharing for proper dihedrals only + `_h0_fragment` repair
### Hypothesis
The cycle-8 energy failure had one mechanical cause: the bond-order factor was
also applied to Sella's improper dihedrals (umbrella modes at three-/four-
neighbour centres without proper dihedrals, and the impropers that replace
linear angles at atoms with ≥ 3 bonds), changing their curvature by 0.85–2.2×
depending on which neighbour bond happens to be the "axis"; that alone flipped
the formaldehyde dimer ketones__ketones (no proper dihedral) into a 2.7 kcal/mol
shallower basin (−0.227). Restricting the factor (and the 1/√n sharing, which
never applied to impropers anyway) to proper dihedrals — a–b–c–d with d bonded to
c — leaves every molecule without proper dihedrals bit-identical to the champion
and keeps the −3.1 % cost gain, which comes from proper torsions of floppy
molecules. Prediction: C_S ≈ 0.910, mean rel_energy ≈ 1.0001 (champion +0.04,
cycle-7 basin changes +0.17, 135043047 −0.16 if its path is unchanged).
Uncertainty: any further adverse basin flip ≥ 0.05 fails the energy gate again.

### Change (functions)
- `Internals._h0_dihedral(..., proper=True)`: `proper=False` returns the plain
  Fischer–Almlöf value (champion behaviour).
- `Internals.guess_hessian`: builds the bonded-pair set, `is_proper(dihedral)`
  tests `frozenset((c, d)) in bonded`; impropers get `_h0_dihedral(proper=False)`
  and are excluded from the redundancy count `ndih`.
- Cycles 5–8 `_h0_fragment` dummy-index repair retained.
Control: molecules without proper dihedrals (formaldehyde, NH3, CH4, monoatomics,
water dimers) must reproduce the champion's calls and energies exactly.
Candidate commit `19f0b01` (full SHA in results.tsv after recording).

### Result
Candidate `19f0b01645625b416beed9a66fbae55bf58b6d49` vs champion
`13cbc5cca5ed795a1793d0eb993a685d1e0dec4b`. Train evaluation
`9a0ce69315dd42e288e9ff6b1cce153a` ([JSON](evaluation_results/cycle-9-train.json)):
469/469 converged, **C_S = 0.90994** (champion 0.93888, −2.9 %), mean rel_energy
1.000103 (valid). Valid evaluation `bdeb5cf50b304c428a44870452780414`
([JSON](evaluation_results/cycle-9-valid.json)): 465/465 converged, **C_S =
0.93389** (champion 0.95997, −2.6 %), mean rel_energy 1.00717. Decision
**keep** — new champion.
- Train by source: rowansci 0.913, PubChem 0.965, dimers 0.997. 437/469 molecules
  are bit-identical to cycle 8; the 32 that differ are exactly the improper-
  bearing molecules, all trivially (same calls, rel_energy to 1e-10) except the
  control ketones__ketones, which returned to the champion's 38 calls / 1.000 as
  predicted. 135043047 stays at 35 calls / 0.839.
- Valid by source: PubChem 0.940 (134985308 89 → 67, 135011106 65 → 55), dimers
  1.018 — the valid dimers got slightly *slower* through basin changes that
  mostly found lower energies (esters__esters 74 → 115 with rel_energy 1.248,
  amides__ethers 19 → 39 / 1.314, guanidiniums__ketones 20 → 39 / 1.129,
  alkenes__amides 12 → 41 / 1.000); the champion's 4.2× outlier
  carboxylates__ketones__valid kept its basin (40 → 37 calls).

### Interpretation
- The prediction of repair 2 was exact on the diagnosed molecule and on the
  overall numbers (C_S 0.910 vs predicted ≈ 0.910, energy +0.048 vs predicted
  +0.05), which supports the causal picture: the torsional gain comes from
  proper torsions of floppy molecules (PubChem 40–80 atoms −9 %), and the
  improper side effect was the only mechanical cause of the cycle-8 failure.
- The valid dimers' +1.8 % shows that the torsional model changes intramolecular
  relaxation timing and therefore the intermolecular path of dimers with floppy
  monomers; on valid those flips happened to be energy-favourable and cost-
  unfavourable. This is the basin lottery, not a systematic cost effect (train
  dimers 0.997).
- Alternatives considered: 1/n instead of 1/√n (cycle-7 analysis suggests
  over-softening for methyl rotors next to carbonyls, which would still be
  4× too stiff, but ring torsions would become too soft); not pursued now.
- Uncertainty: single evaluations per split; the energy margin on train is
  +0.05 (carried by cycle-7-type basin changes) and on valid +3.3 (of which
  +3.2 is one molecule), so the next change that perturbs dimer paths again
  faces the same lottery.
Champion is now `19f0b01` (train 0.90994 / valid 0.93389).

### Next idea/control
The largest remaining systematic cost is the dimer endgame (219 train molecules,
26.4 calls mean, equal to the reference) and the reactive/connectivity-changing
molecules (23 molecules, 5.5 % of calls, 135043047 −0.16 energy). Next cycle:
bond-formation-aware rebuild of the internals (backlog item
`bond-formation-aware-internals`), a general mechanism that re-detects covalent
connectivity when a non-bonded pair enters the bonding radius and rebuilds the
coordinate set through the existing `initialize_pes` path (cached forces, no
extra call). Control: molecules whose connectivity never changes must be
bit-identical to the champion.

## Cycle 10 — angle-valence-scale: halve the Fischer–Almlöf bend guess

Candidate commit: `9d64f8b2809b7b9b878b18de1d8afc39a66fa40d` (anchor
`19f0b01645625b416beed9a66fbae55bf58b6d49`).

### Hypothesis
Cycle 9 showed that the accuracy of the diagonal guess for *angular*
primitives matters even with TS-BFGS updates (torsions −9 % on floppy
molecules). Evaluating the vendored Fischer–Almlöf bend formula gives
0.28 Ha/rad² for H–C–H, 0.32 for H–C–C, 0.35 for C–C–C, 0.30 for C–O–H and
0.40 for C–C=O. The corresponding valence force-field constants for the same
primitive angles are about half of that (methane 0.53, ethane/propane
0.65–1.0, water/alcohols 0.75, carbonyls ≈1.0 mdyn·Å/rad² = 0.12–0.23
Ha/rad²; Schlegel's classic bend guesses are 0.16/0.25 and geomeTRIC uses
0.16). Because Sella's model energy over a redundant primitive set,
½ Σ k_i δq_i², has exactly the form of a valence force field, the primitive
diagonal should match the valence constants. A 2× too stiff bend model halves
the first steps along bending modes, and since the error is not uniform across
stretch/bend/torsion subspaces the secant update only fixes one direction per
step; small rigid molecules (where we are identical to the reference, ratio
≈ 1.0, 8–15 calls) should lose 1–3 calls, larger ones more. Prediction: PubChem
and rowansci sets −3 to −6 %, dimers roughly unchanged in cost with the usual
basin lottery on energy (all paths change). This is the smallest bend-model
change that tests the stiffness question; refinement by hybridisation would
follow if the direction is confirmed.

### Change (functions)
- `Internals._h0_angle`: new keyword `valence_scale=0.5` multiplying the
  Fischer–Almlöf bend value (with a comment listing the reference constants).
  Nothing else changes; bonds, torsions, impropers, fragment translations and
  rotations keep the cycle-9 model.

### Result
- Train: `evaluation_results/cycle-10-train.json` (evaluation
  71a115ea593347e8bb76e46752995a2d): C_S 0.9049710 vs anchor 0.9099351
  (Δ = −0.00496), energy 1.000132 (Σ(rel−1) = +0.062), 469/469 converged.
- Valid: `evaluation_results/cycle-10-valid.json` (evaluation
  dd2cae46930a474a843d60d4a825feb0; the first two receipts were `pending`
  and the identical command was repeated): C_S 0.9218204 vs anchor 0.9338923
  (Δ = −0.01207), energy 1.007633, 465/465 converged.
- Decision (`scripts/cycle.py record`): **keep** — new champion
  `9d64f8b2809b7b9b878b18de1d8afc39a66fa40d` (train 0.90497 / valid 0.92182).
- Breakdown (mean ratio anchor → candidate; same-basin unless noted):
  train PubChem 0.9677 → 0.9700 (natoms <10: 1.004 → 0.964; 10–20: 0.993 →
  0.996; 20–35: 0.948 → 0.962; 35–60: 0.929 → 0.958, which includes the
  reactive molecule 104079126 63 → 137 calls with a 31 kcal/mol deeper final
  energy; excluding it PubChem is 0.9651 → 0.9594), rowansci 0.9125 →
  0.9191, dimers 0.8503 → 0.8366 (99 better / 63 same / 57 worse).
  Valid PubChem 0.9398 → 0.9382 (<10: 0.956 → 0.912; 10–20: 0.966 → 0.938;
  20–35: 0.929 → 0.920; 35–60: 0.935 → 0.953), dimers 0.9271 → 0.9028
  (105 better / 51 same / 59 worse).
- Large PubChem molecules (≥35 atoms) stay in the same basin (|ΔE| < 0.1
  kcal/mol for 59/59 train, 120/122 valid) and cost +0.8 (train) / +0.5
  (valid) calls on average (27 worse vs 18 better; 57 vs 39): a broad, mild
  systematic loss, not an outlier effect.

### Interpretation
- The direction of the bend-stiffness hypothesis holds for small molecules
  (<20 atoms −3 to −4 %) and for dimers (−1.6 % train, −2.4 % valid, with
  2:1 better:worse counts on both splits, mostly same-basin), i.e. wherever
  the bending modes are local and the true diagonal is close to the valence
  constants. It fails for large floppy molecules (≥35 atoms, +2–3 %): their
  slow modes are collective skeletal deformations made of many bends and
  torsions, where the diagonal model has no bend–bend or bend–torsion
  coupling, and a stiffer bend diagonal apparently acts as useful damping
  (the same reason 1/√n rather than 1/n worked for torsions in cycle 9).
- Alternative explanation: the valence ratio is not uniform (H–C–H 0.43,
  H–C–C 0.47 of the Almlöf value, but C–C–C 0.66, C–C=O 0.58, C–O–H 0.57),
  so a single ×0.5 over-softens the heavy-atom skeleton angles that dominate
  large molecules while being right for H-containing angles. A two-level
  scale (≈0.45 with a terminal H, ≈0.6–0.65 heavy-only) is the natural
  refinement and is logged in the backlog (`angle-valence-scale`).
- Uncertainty: single evaluations; the covalent effect is ±0.5–1 call per
  molecule and the dimer gain is partly path-dependent (basin lottery is
  favourable this time: train Σ(rel−1) +0.062, valid +3.5).

### Next idea/control
The bend model is now at valence magnitude; the remaining H0 refinement with
a physical basis is the two-level bend scale (heavy-atom angles stiffer than
H-containing ones), expected to recover part of the +0.5–0.8 calls lost on
large molecules while keeping the small-molecule and dimer gains. Larger
levers remain the dimer floor (near-minimum dimers 1.15–1.44× the reference)
and the reactive molecules (bond-formation-aware rebuild). Next cycle: the
two-level bend scale (cheap, same lottery exposure), then the dimer floor.
Control: molecules without heavy-only angles (none in practice) unchanged;
expect large PubChem (≥35 atoms) to move back toward the cycle-9 counts.

## Cycle 11 — angle-valence-scale refinement: two-level bend scale

Candidate commit: `ebfd6feffc3ac2d6cf4690bd31399615c0d8e83e` (anchor
`9d64f8b2809b7b9b878b18de1d8afc39a66fa40d`).

### Hypothesis
The cycle-10 factor 0.5 matches the valence/Almlöf ratio of angles with a
terminal hydrogen (H–C–H 0.43, H–C–C 0.47, H–N–H 0.50, H–O–C 0.57, H–O–H
0.62) but over-softens heavy-atom-only angles, whose ratio is 0.56–0.66
(C–O–C 0.56, C–C=O 0.58, C–C–C 0.66). Large molecules are dominated by such
skeleton angles and were the only group that lost calls in cycle 10 (+0.5–0.8
calls, same basin). Prediction: PubChem ≥35 atoms recovers roughly half of
that loss, small molecules and dimers (H-rich angles unchanged) keep their
cycle-10 counts; overall −0.2 to −0.5 %, subject to the basin lottery.

### Change (functions)
- `Internals._h0_angle`: `valence_scale` replaced by `valence_scale_h=0.5`
  (either terminal atom is hydrogen) and `valence_scale_heavy=0.62`
  (otherwise, including dummy-atom angles). Nothing else changes.

### Result
- Train: `evaluation_results/cycle-11-train.json` (evaluation
  c7f2fe103b0445569cddd6858a1286da): C_S 0.9073954 vs champion 0.9049710
  (Δ = +0.00242), energy 1.000342, 469/469 converged. Decision: **discard**
  (train improvement below 1e-4). Valid not run.
- Breakdown (c9 → c10 → c11 mean ratios): PubChem 0.9677 → 0.9700 → 0.9612
  (≥35 atoms 0.9285 → 0.9577 → 0.9335, i.e. the cycle-10 loss on large
  molecules is recovered as predicted; <10 atoms 1.004 → 0.964 → 0.969);
  rowansci 0.9125 → 0.9191 → 0.9325, entirely venetoclax 41 → 82 calls
  (different conformer, −8.2 kcal/mol); the other 12 changed rowansci
  molecules are 10 better / 2 worse by 1–3 calls. Dimers 0.8503 → 0.8366 →
  0.8493: 52 better / 116 same / 51 worse, 217/219 same basin, +58 calls in
  total of which +34 come from two molecules (amides__phenol 39 → 56,
  carboxylates__guanidiniums 16 → 33).

### Interpretation
- The physical prediction held on the covalent sets (PubChem −0.9 %, large
  molecules back to the cycle-9 level while keeping the small-molecule gain;
  rowansci −4 % outside one conformer flip). The train mean nevertheless
  rose by 0.24 % because dimers respond to any perturbation of the monomer
  bend model with a symmetric ±5-call endgame scatter (52:51) plus a few
  large same-basin outliers, and one rowansci conformer flip cost 41 calls.
- This is the third time a same-basin dimer scatter of ±0.5–1.5 % decided a
  cycle (cycles 8, 10, 11); the dimer endgame is not just slow but also
  chaotic with respect to the approach path. Reducing that sensitivity is
  the core of the dimer-floor idea and would also make small H0 refinements
  measurable.
- No bounded repair: the failure has no mechanistic cause in the candidate
  (the covalent effect is as designed; the loss is path scatter on dimers
  and one conformer flip), so a repair would amount to re-rolling.
  Implementation preserved in
  `ideas/angle-valence-scale/ebfd6feffc3ac2d6cf4690bd31399615c0d8e83e/algo.py`;
  the two-level scale should be re-tested as part of the next bend-model
  change rather than alone.

### Next idea/control
Champion restored to `9d64f8b`. Next: the dimer endgame — make the
intermolecular soft-mode convergence less path-dependent and shorter. First
candidate mechanism: the fragment rotation/translation guess curvature and
its update (soft modes converge geometrically at a rate set by the model /
true curvature ratio; the contact model is a rough guess, and TS-BFGS learns
one direction per step). Control: covalent molecules bit-identical.

## Cycle 12 — multi-secant-update: limited-memory multi-secant TS-BFGS

Candidate commit: `ac30253ee6257494156a9af03f75fe334c61a817` (anchor
`9d64f8b2809b7b9b878b18de1d8afc39a66fa40d`).

### Hypothesis
The dimer endgame (≈10–12 of the 17–22 calls of a near-minimum dimer) is a
quasi-Newton search in the 6-dimensional intermolecular soft subspace, where
the diagonal contact guess is rough and each single-secant TS-BFGS update
makes the model exact only along the newest step; earlier secant conditions
are progressively destroyed, so the soft block is learned asymptotically
(≈1.5–2 n steps for n soft modes). The vendored `update_H`/`_MS_TS_BFGS`
already implement the multi-secant form B S = Ỹ with the symmetrisation
`symmetrize_Y2`; imposing the last m independent pairs simultaneously makes
the model exact on the span of the recent steps (a quadratic 6-D block is
learned in 6 steps). Expected: shorter endgames for dimers (−1 to −3 calls)
and for floppy molecules with several soft torsions; small rigid molecules
(few steps) little change. Risk (cycle-5 lesson): the secant data are noisy
in curvilinear coordinates, and imposing several conditions at once may
amplify that; mitigated by normalising the pairs, dropping (nearly)
dependent older steps (tolerance 0.3 on the orthogonal remainder), memory 4,
and the TS-BFGS metric (|B|) which keeps the complement bounded.

### Change (functions)
- `PES.__init__`: `secant_memory=4`, `secant_dep_tol=0.3`, `_secant_pairs`.
- `PES._update_H`: collects the pair with `PES._collect_secant_pairs`
  (newest first, normalised by |dx|, Gram–Schmidt dependency filter) and
  calls `ApproximateHessian.update(S, Y)` with the column matrices; with a
  single surviving column the update is the previous one (the TS-BFGS
  formula is scale invariant). The history lives in the PES object, so a
  rebuild of the internals (`initialize_pes`) starts a fresh history.

### Result (candidate 1)
- Train: `evaluation_results/cycle-12-train.json` (evaluation
  0414ba73ec154cf2b4e11d676364017e): C_S 0.9173002 vs champion 0.9049710
  (Δ = +0.0123), energy 1.000135, 469/469 converged. Decision: **discard**.
- Breakdown: PubChem 0.9700 → 0.9602 (97 better / 67 same / 61 worse;
  10–20 atoms 0.996 → 0.967; ref ≥ 40 calls 1.14 → 1.03), rowansci 0.9191
  → 0.9462 (paliperidone_palmitate 72 → 128, otherwise 13 better / 6 worse),
  dimers 0.8366 → 0.8699 (83 better / 29 same / 107 worse; same-basin mean
  +0.97 calls; tails: amides__water 31 → 65, acids__esters 20 → 56,
  esters__ethers 21 → 46 vs ammoniums__esters 57 → 35, amides__carboxylates
  52 → 33). Near-minimum dimers (ref < 15) 1.44 → 1.63.

### Interpretation (candidate 1)
- The covalent sets behave as hypothesised (more secant conditions → shorter
  runs for mid-size molecules), but the dimer endgame gets longer and much
  more variable. So the recent secant pairs of dimers are not mutually
  consistent with one symmetric Hessian: the soft intermolecular modes are
  strongly anharmonic on the scale of the endgame steps and/or the
  rigid-body coordinates' gradient transport between steps is inexact.
  A single-secant update overwrites stale information within one step; the
  4-pair block keeps enforcing it. Diagnosed cause: inconsistent older
  pairs (energy/cost failure, not an implementation error).
- Repair 1 (bounded, general): keep an older pair only if it is
  quadratically consistent with every newer kept pair,
  |s_new·y_old − s_old·y_new| ≤ 0.25·sqrt((s_old·y_old)(s_new·y_new)),
  so anharmonic soft-mode pairs fall back to the single-secant update while
  consistent (quasi-quadratic) sequences still get the block update.
  Candidate `9c106c78fda1a4fcf8ec5a5d863bd5a770ca0d30`.

### Next idea/control
- The repair candidate is a new commit, so the decision writer bound cycle
  12 to candidate 1 (discard) and refused it; the repair was evaluated as
  cycle 13 below (the train receipt of candidate 1 was re-fetched with
  `client.py results --evaluation-id 0414ba73…` after it had been overwritten
  by the repair's first run).

## Cycle 13 — multi-secant-update repair 1: quadratic-consistency filter

Candidate commit: `9c106c78fda1a4fcf8ec5a5d863bd5a770ca0d30` (anchor
`9d64f8b2809b7b9b878b18de1d8afc39a66fa40d`). Same idea as cycle 12; only
`PES._collect_secant_pairs` changed.

### Hypothesis
See cycle 12. Repair 1: an older secant pair enters the block only if it
is quadratically consistent with every newer kept pair (the symmetry test
|s_new·y_old − s_old·y_new| ≤ 0.25·sqrt((s_old·y_old)(s_new·y_new))).
Expected: the dimer endgame reverts to the single-secant behaviour when
the soft modes are anharmonic on the scale of the steps, while the
covalent gains of candidate 1 (PubChem −4 %) survive.

### Change (functions)
- `PES.__init__`: `secant_cons_tol = 0.25`.
- `PES._collect_secant_pairs`: after the Gram–Schmidt dependency test the
  consistency test against all newer kept pairs; inconsistent pairs are
  skipped (not removed from the history, so they can still be used against
  later pairs).

### Result
- Train: `evaluation_results/cycle-13-train.json` (evaluation
  a30bfd3f6b044b89a36c5129b50a13b3): C_S 0.8844994 vs champion 0.9049710
  (Δ = −0.0205), energy 1.000478, 469/469 converged.
- Valid: `evaluation_results/cycle-13-valid.json` (evaluation
  7f0fa4abd934431f9284e1d6008f280c): C_S 0.8936084 vs champion 0.9218204
  (Δ = −0.0282), energy 1.007800, 465/465 converged.
- Decision: **keep** (results.tsv cycle 13). New champion `9c106c7`.
- Breakdown train: PubChem 0.9700 → 0.9318 (117 better / 74 same / 34
  worse), rowansci 0.9191 → 0.9137, dimers 0.8366 → 0.8326 (125 better /
  28 same / 62 worse); near-minimum dimers (ref < 15) 1.44 → 1.60; energy
  sum +0.22 (two dimers moved to deeper basins, none lost more than 0.1).
- Breakdown valid: dimers 27.27 → 26.30 mean calls (rel 0.979), PubChem
  22.12 → 21.47 (rel 0.975); energy sum +3.6 (carboxylates__ketones now
  −12.85 kcal/mol deeper; ethers/esters/amides__water each 4.7–10 kcal/mol
  shallower basins, −0.41/−0.35 each), 8 losers vs 8 gainers.

### Interpretation
- The consistency filter removed the dimer damage of candidate 1 (same-basin
  dimers now −0.9 calls instead of +1.0) and kept the covalent gain: the
  block update helps whenever the recent steps sample a quasi-quadratic
  region, and the filter switches it off exactly where it hurt. The
  strongest gains are the long PubChem runs (ref ≥ 40 calls) and the
  floppy dimers with several coupled soft modes (ammoniums__esters 46 → 20,
  amides__ethers 51 → 31).
- The near-minimum dimers (ref < 15 calls, 1.44 → 1.60) are still slightly
  worse: with 2–4 steps there is no history to exploit and the first block
  updates (2 pairs) are applied to steps that are still in the contact
  regime. Alternative explanation: the pairs of the very first steps are
  taken with the guess Hessian and the first real curvature, where the
  symmetric test is passed trivially (only 2 pairs) — a minimum of 3 pairs
  before imposing a block, or an age limit, could be tested.
- Uncertainty: the valid energy sum is dominated by one basin flip (+3.2);
  the call gains (−2.3 % on both splits, both sources) are consistent
  across splits and sources, so they are not lottery noise.

### Next idea/control
- Backlog `multi-secant-update` is now incorporated; open refinements:
  memory 6–8 (the covalent runs may profit from more pairs), a minimum block
  size of 3, and a tolerance sweep (0.15 / 0.4) — each a single-parameter
  control that should be bundled with another change because the expected
  effect is ≈0.5 %.
- Next: the bend-scale two-level variant (cycle 11) bundled with a memory-6
  multi-secant, or the hybridisation-aware torsion sharing.

## Cycle 14 — Lindh-type contact bending stiffness in the fragment TR guess (discard)

Champion at start: `9c106c78fda1a4fcf8ec5a5d863bd5a770ca0d30` (train 0.8844994 /
valid 0.8936084). Candidate: `6cc9adc8604c703313d050e93b5cb18c6a340a78`.

### Hypothesis
The fragment translation/rotation guess models only the stretch projections
of the inter-fragment contacts (cycle 3), so libration and sliding modes of
H-bonded and salt-bridge dimers sit at the 1e-3 Ha floor, 5–30× softer than
the physical curvature (water dimer libration 0.01–0.03 Ha/rad²). Because the
evaluator's displacement criterion is binding for these soft modes, the
endgame spends many steps creeping along them; a Lindh-type bending stiffness
k_φ·ρ_bond·ρ_contact (k_φ = 0.15 Ha/rad²) for every angle k–i···j and
i···j–l around a contact should shorten the dimer endgames.

### Change (functions)
- `Internals._h0_fragment`: computes the contact bond orders once, adds the
  bending contribution when a neighbour list is passed.
- New `Internals._h0_contact_bend`: for each contact pair with
  bo ≥ 1e-4, all bonded neighbours on both sides form angle terms; the
  stiffness of a TR coordinate is k_φ·bo_bond·bo_contact·|dε|² with the
  isotropic linear-bend vector ε = â + b̂ (only the moving fragment's
  atoms displaced).
- `Internals.guess_hessian`: builds the bonded neighbour lists from
  `internals['bonds']` and passes them to the translation and rotation
  guesses.

### Result
Train C_S 0.8926722 vs 0.8844994 → **discard** (train improvement below
1e-4). Energy ratio 1.000427. Evidence
`evaluation_results/cycle-14-train.json` (1b423bbcf946473ca508662153027436).
Breakdown: rowansci 1.000 (25 identical), PubChem 1.0002 (224/225
identical), dimers 1.034 (25.47 → 26.08 calls, 95 identical). Best:
ammoniums__carboxylates 29 → 11 (reference level), acids__ammoniums
30 → 20, acids__ketones 43 → 33, alcohols__carboxylates 36 → 26,
amides__carboxylates 42 → 32. Worst: acids__esters 16 → 36 (+ basin change),
alkanes__carboxylates 19 → 39 (basin change, energy 0.673), acids__acids
41 → 54, acids__guanidiniums 32 → 46, esters__imidazolium 20 → 32,
alcohols__pyridine 27 → 39, pyrrole__water 16 → 27, pyridine__pyridine
35 → 45. Contact-strength bins (bo_max of the closest contact): strong
(≥ 0.02, n = 67) +0.96 calls (23 better / 35 worse), moderate
(0.005–0.02, n = 31) +1.97 (8/20), weak (n = 114) +0.10 (79 identical).
Initial→final RMSD bins of the champion run (strong contacts): < 0.3 Å
+0.04 (9/11), 0.3–0.6 Å +0.14 (8/11), ≥ 0.6 Å +2.86 (6/13).

### Interpretation
- The physics is right where the guess matters most (near-minimum salt
  bridges reach the reference cost), but the implementation makes the guess
  too stiff along directions it should leave soft, and quasi-Newton punishes
  over-estimated curvature much more than under-estimated curvature: a soft
  guess overshoots once and the secant update repairs it; a stiff guess
  along a direction that the steps barely contain shrinks the step every
  iteration without the update ever learning it.
- Two concrete defects: (1) the diagonal-in-lab-frame model: the soft twist
  about the contact axis is a combination of the three lab-axis rotations,
  so every diagonal guess gives it the libration stiffness; (2) the
  isotropic ε = â + b̂ form counts the precession of one arm about the
  other (dθ = 0) as a bend with weight sin²θ, so the acceptor twist about
  the O···H axis gets ≈ 0.01 Ha/rad² instead of ≈ 0.001. Both explain the
  losses on very-short-contact dimers (bo_max 0.2–0.35, where the bending
  term is largest) and on far-from-minimum dimers (many steps with a wrong
  soft-mode curvature), and the deterministic +3 to +7 losses on
  near-minimum dimers that the floor was already solving faster than the
  reference (pyrrole__water, amides__pyrrole).
- A third, milder issue of the whole contact model (also of cycle 3's
  stretch term): each contact stiffness is assigned to both fragments
  independently, so a relative step overshoots by 2× relative to the model.
- Alternatives: the losses could partly be an approach-phase effect (stiffer
  librations reduce the exploratory 0.1-rad trust steps on anharmonic
  surfaces) rather than hidden-direction creep; the two hypotheses are
  distinguished by the next candidate — a full-block model keeps the
  librations stiff but leaves the twist at the floor, so if the losses
  persist the approach-phase explanation is the right one.
- Uncertainty: 2 basin changes (acids__esters, alkanes__carboxylates) are
  lottery; the remaining ±10-call deterministic changes are systematic.

### Next idea/control
- Repair 1 (planned as cycle 15): the full rigid-body block of the contact
  model, H_TR = Σ k_ij ∇q r_ij ∇q r_ijᵀ + Σ k_φ ∇q θ ∇q θᵀ over the
  translation+rotation coordinates of all fragments with the cross-fragment
  blocks, true-angle derivatives (two perpendicular bends for sin θ < 0.1),
  floor 1e-3 on the diagonal. Targets both diagnosed causes (hidden twist,
  precession overcount) and the 2× overshoot.
- Fallback if the repair fails: keep the stretch-only guess and revisit the
  endgame instead (multi-secant minimum block size, dimer trust growth).

## Cycle 15 — Full rigid-body block of the contact model in the fragment TR guess (non_generalizable)
Candidate `bb7ede468bab945528fd5010e2cc0ffeb8005471` vs champion
`9c106c78fda1a4fcf8ec5a5d863bd5a770ca0d30`. Repair 1 of the
`contact-bend-guess` idea (see cycle 14).

### Hypothesis
Cycle 14's losses come from two implementation defects of the diagonal
contact guess — the twist about a contact axis is hidden inside the three
lab-axis rotations and therefore gets the libration stiffness, and the
isotropic ε-form counts precession as bending — plus the 2× overshoot of the
relative step when each fragment carries the full pair stiffness. Building
the Gauss–Newton block of the pair model over the rigid-body coordinates of
all fragments (H_TR = Σ k_ij ∇q r_ij ∇q r_ijᵀ + Σ k_φ ∇q θ ∇q θᵀ + 1e-3 I,
cross-fragment terms included) removes all three, so the near-minimum
salt-bridge gains of cycle 14 should survive while the very-short-contact
and far-from-minimum losses disappear.

### Change (functions)
- `_h0_fragment` replaced by `_h0_fragment_block(slots, nbrs, ...)`: builds
  the Cartesian generators of every fragment translation/rotation slot
  (rotation generator e × (x − c) about the group centroid, dummies
  included in the centroid but excluded from contacts), the stretch part
  from all inter-fragment pairs with the Almlöf bond-order stiffness
  0.3601·exp(−1.944 (r − r_cov)/Bohr), and a bending part only for
  hydrogen-bond contacts X–H···Y (Y ∈ N, O, F, S, Cl, Br, I) around both the
  H and the Y vertex, k_φ = 0.15 × min(bo, cap) with cap 0.05 for N/O/S/halogen
  donors and 0.01 for C–H donors, using the true angle derivative plus an
  out-of-plane isotropic term weighted exp(−((π−θ)/0.5)²) for near-linear
  angles. Frame-invariant, symmetric, PSD (checked statically on saved
  geometries under random rotations).
- `guess_hessian`: collects the TR slots (`tr_slots`) instead of writing
  per-coordinate values, builds the bonded-neighbour lists from
  `internals['bonds']`, and writes the block into `H0[np.ix_(sl, sl)]`.

### Result
- Train: C_S **0.8772586** vs 0.8844994 (−0.0072, passes the 1e-4 gate),
  mean rel_energy 1.001254, all 469 converged. Evidence
  `evaluation_results/cycle-15-train.json` (e703db0a26724480a1a0ab7f9bfc4c15).
- Valid: C_S **0.9039743** vs 0.8936084 (+0.0104), mean rel_energy 1.007789.
  Evidence `evaluation_results/cycle-15-valid.json`
  (d4bc0cd9ba3d4c1395a8b623728aed3d). Decision: **non_generalizable**.
- Paired breakdown (basin change = |ΔE_final| > 0.2 kJ/mol): train −73 calls
  in total, of which 9 basin changes give −59 (alcohols__sulfides 40 → 15,
  amides__phenol 78 → 44, carboxylates__guanidiniums 29 → 13,
  benzene__guanidiniums 14 → 28) and the 460 same-basin runs give −14
  (80 better / 86 worse; dimers −0.35 per molecule, PubChem +0.02, rowansci
  0). Valid +81 in total: 10 basin changes −21, same-basin +102 (71 / 87;
  dimers +0.38 per molecule, PubChem identical paths). Same-basin dimers by
  reference cost, train / valid: ref < 15 −0.67 / +0.78, 15–25 −0.16 / +0.28,
  25–40 +0.33 / +0.71, ≥ 40 −0.15 / +0.34. Largest same-basin swings:
  ammoniums__esters 58 → 36, benzene__imidazolium 50 → 35, benzene__phenol
  28 → 14, phenol__thiols (valid) 29 → 12, acids__ammoniums (valid) 23 → 16
  against amides__ammoniums (valid, ref 10) 42 → 66, amides__phenol (valid)
  41 → 62, alkenes__pyridine 14 → 27, alkanes__carboxylates 19 → 30.

### Interpretation
- The block model does what it was designed to do (static checks: the twist
  about an O···H axis sits at the floor, librations at 0.005–0.03 Ha/rad²,
  frame invariance to 1e-12), and the result is a wash: the same-basin
  effect is −0.03 calls per molecule on train and +0.22 on valid with a
  ±10–20-call spread on individual dimers in both directions. The train
  pass was a basin lottery (−59 of −73 calls from 9 basin changes).
- Hence the hidden-direction over-stiffness of the champion's diagonal
  stretch model is not what costs the near-minimum dimers their calls. The
  decomposition of the champion's initial→final motion (Kabsch, per
  fragment) shows that for the ref < 15 dimers the relative rigid-body
  RMSD is 0.26 Å (median; up to 0.9 Å for phenol__thiols, alkenes__thiols,
  amides__ethers) while the intramolecular RMSD is 0.03 Å: our optimizer
  slides 0.3–0.9 Å along the flat intermolecular surface, gaining
  0.1–0.5 kcal/mol, while the reference stops within ≈ 10 calls. The
  reference must stop because its stiff model barely moves those
  coordinates and the displacement criterion (1.8e-3 Bohr) is then met as
  soon as the forces are below threshold; a physically soft model has to
  converge the soft-mode gradients to |g| ≈ k · 1.8e-3 Bohr, i.e. 100–300×
  below the force threshold, on an anharmonic surface, which no initial
  curvature can shortcut.
- Alternatives: the valid regression could be sampling noise of a
  high-variance path change rather than a systematic effect (same-basin
  dimer means differ by 0.7 calls between the splits with a spread of ±5);
  either way there is no evidence of a systematic gain to repair.
- Uncertainty: the bending caps (0.05 / 0.01) and the linear-bend weight
  were set from physical estimates without tuning; a different k_φ would
  move individual dimers but the class-level wash argues against a
  parameter fix.

### Next idea/control
- No further repairs of the contact guess (2 of 3 bounded repairs unused;
  the diagnosed causes were addressed and did not matter). Idea parked with
  the implementation preserved in
  `ideas/contact-bend-guess/bb7ede468bab945528fd5010e2cc0ffeb8005471/algo.py`.
- The near-minimum dimer endgame (43 train dimers with ref < 15 at C_S 1.75,
  ≈ 0.07 of C_S) is governed by convergence acceleration on flat anharmonic
  surfaces, not by the guess: revisit the endgame mechanisms in the
  backlog — multi-secant memory/minimum block size, controlled GDIIS with
  the fragment-aware safeguards — under the rule that steps must be the
  model's genuine steps (no damping to trigger the displacement criterion).

## Cycle 16 — energy-augmented secant condition (discard)

Champion at start: `9c106c78fda1a4fcf8ec5a5d863bd5a770ca0d30` (train 0.8844994 /
valid 0.8936084). Candidate: `7c0c154d68064fd1fe839244bf17fbfd2f376b5b`.

### Hypothesis
After cycles 14–15 showed that contact-guess physics cannot close the
near-minimum dimer gap, the remaining lever is convergence acceleration on
anharmonic modes. The plain secant condition B s = y makes the model
curvature along the step the average over the step, i.e. half a step stale
at the point where the next Newton step is taken. The modified secant
condition of Zhang–Deng–Chen (1999) / Wei–Li–Qi (2006),
y* = y + θ s/(s·s) with θ = 6(f0 − f1) + 3(g0 + g1)·s, uses the two
end-point energies to make s·B s equal to the curvature at the new point
to O(|s|⁴) (exact for cubics; checked analytically). On Morse-like
stretches and H-bond/salt-bridge approach coordinates this should remove
≈ 1 step per anharmonic mode chain, across all molecules (systematic,
small per molecule — the kind of effect the lottery cannot hide).

### Change (functions)
- `PES.__init__`: `secant_energy_clip = 0.5`.
- New `PES._secant_dg(s, y, df, g0, g1)`: θ from the end-point energies and
  gradients (g0 = transported gradient `g_par`, g1 = new gradient, s =
  `dx_final`), clipped to ±0.5|s·y| so that the sign of the curvature is
  preserved; returns y + θ s/(s·s).
- `PES.kick`: passes the corrected difference to `_update_H` (the
  multi-secant block then imposes it exactly on column 0).

### Result
Train C_S 0.8982460 vs 0.8844994 → **discard** (train improvement −0.0137).
Energy ratio 1.001287 (max Δ 38.0 kcal/mol unchanged). Evidence
`evaluation_results/cycle-16-train.json` (346b3366bd684533b9a9ba5d6d40f724).
Paired: 20 basin changes (−116 calls), same-basin 449 molecules mean
+0.75 calls (119 better / 194 worse); by source PubChem 0.932 → 0.944,
dimers 0.833 → 0.843, rowansci 0.914 → 0.966. Same-basin dimers by ref
bin: [0,15) +0.19, [15,25) +1.00, [25,40) +0.55, [40+) +1.75. The loss is
broad: |d| < 8 contributes +0.017 of the +0.023 same-basin C_S loss.
Six stretched ionic dimers stopped at the stretched geometry like the
reference (ammoniums__esters 58 → 15 (+0.8 kJ/mol), amides__carboxylates
42 → 15, carboxylates__water 44 → 20 (+12.1 kJ/mol), esters__ethers
36 → 18, acids__guanidiniums 32 → 21, carboxylates__guanidiniums 29 → 17);
best same-basin: imidazolium__ketones 58 → 47, acids__ketones 43 → 33,
esters__thiols 40 → 30, ketones__phenol 60 → 50; worst:
carboxylates__esters 26 → 54, ammoniums__water 21 → 40,
alkanes__carboxylates 19 → 35, amines__water 36 → 50.

### Interpretation
- The mechanism is implemented as derived (sign and magnitude verified on
  a cubic; units consistent: eV, eV/Å·Å), so this is an energy/cost failure
  of the idea, not a contract error.
- First diagnosis: near convergence |f0 − f1| falls to the 1e-7–1e-6 Ha
  level where the SCC energy noise, multiplied by six, exceeds |s·y| for
  steps of 1e-3 Å; the clip then turns θ into a random ±50 % rescaling of
  the curvature along every endgame step. The growth of the loss with run
  length (ref bins, rowansci) fitted this. → repair 1 (cycle 17).
- The stretched-dimer early stops show the corrected model stiffening the
  slide coordinate (θ > 0 when the curvature grows inward), which mimics
  the reference's behaviour on type-(ii) dimers — a large call gain that
  costs energy credit; not a mechanism to pursue deliberately.

### Next idea/control
Repair 1: skip the correction when |s·y| < 3e-4 eV (energy-resolution
floor); everything else unchanged.

## Cycle 17 — energy-secant repair 1: energy-resolution floor (discard)

Champion at start: `9c106c78fda1a4fcf8ec5a5d863bd5a770ca0d30`. Candidate:
`0eddb4a1ed75dba8f4dd72a8b006172d4ccf4e6e`.

### Hypothesis
If the cycle-16 loss comes from unresolved end-point energies at the
endgame, skipping the correction below |s·y| = 3e-4 eV (≈ 1.1e-5 Ha; six
times a 1e-7 Ha energy noise is then ≤ 5 % of the curvature) keeps the
approach/mid-phase benefit and removes the endgame damage.

### Change (functions)
- `PES.__init__`: `secant_energy_floor = 3e-4` (eV).
- `PES._secant_dg`: returns the plain y when |s·y| is below the floor.

### Result
Train C_S 0.9040092 vs 0.8844994 → **discard** (−0.0195). Energy ratio
1.001738. Evidence `evaluation_results/cycle-17-train.json`
(e4bfc2fb50da4d709cd4122c13fced9d). Paired vs champion: 18 basin changes
(−86 calls; the same stretched-dimer early stops as cycle 16), same-basin
451 molecules mean +0.91 (117 better / 200 worse); PubChem 0.944, dimers
0.855, rowansci 0.970; same-basin dimers by ref bin [0,15) +0.76,
[15,25) +1.00, [25,40) +0.54, [40+) +2.33; |d| < 8 contributes +0.019 of
+0.027.

### Interpretation
- The floor removed the endgame corrections and the result got slightly
  worse, with the early trajectory (basin changes) essentially unchanged:
  the endgame corrections were neutral-to-helpful and the approach/mid-phase
  corrections carry the harm. The noise diagnosis is refuted.
- Candidate causes for the mid-phase harm (untested): the end-point
  curvature of the new pair is inconsistent with the interval-average
  curvatures of the older pairs in the multi-secant block (more pairs
  dropped by the consistency filter or averaged by `symmetrize_Y2`); on
  trust-limited approach steps the cubic expansion is invalid and the clip
  still allows 50 % rank-1 curvature changes; on the compressed side of a
  Morse-like mode the next RFO step needs the average curvature over the
  *next* interval, which the stale interval average predicts better than
  the end-point curvature. None of these predicts a subset that could be
  repaired with a general rule, and the degradation is broad (86 molecules
  ≥ +3 calls vs 28 ≤ −3), so the failure is not promising: no further
  repairs (2 of 3 unused).
- Uncertainty: the gains on deep H-bonded dimers (−10 to −11 calls on four
  molecules) suggest a sign-restricted (max(θ,0)) variant might retain a
  benefit for contact formation, but the same molecules move by ±10 under
  any path change (lottery), so this is a weak signal; recorded in the
  backlog, not pursued.

### Next idea/control
Champion restored (`git reset --hard 9c106c78fda1a4fcf8ec5a5d863bd5a770ca0d30`).
Curvature-modification ideas from the energies are exhausted; next
candidates are step-control mechanisms that need no extra force call:
growth-only trust radius with a larger cap (cycle 4 tested growth *and*
shrink changes together), and step rejection with position restore on
rho < 0 (keeps the secant pair, avoids the uphill point becoming the
reference geometry).

## Cycle 18 — growth-only trust radius (invalid)

Champion at start: `9c106c78fda1a4fcf8ec5a5d863bd5a770ca0d30`. Candidate:
`be5daad3baf6b70011aa3f989d170e48c1111a39`.

### Hypothesis
The costliest dimers are trust-limited slides (Kabsch decomposition of the
champion's motion: rotations of 40–90° and centroid shifts of 1–2 Å at
0.1 rad / 0.1 Å per step, e.g. ammoniums__esters 58 vs 19 calls,
imidazolium__ketones 58 vs 23) and unrelaxed PubChem conformers need several
0.1-limited steps before the QN step fits inside the radius. Sella grows the
max-internal-step radius only ×1.15 and only for 0.75 < ρ < 1.33. A
textbook growth-only policy (Nocedal & Wright ch. 4: ×1.5 for ρ > 0.75,
including ρ > 1.33, cap 0.5 Å/rad, shrink rule unchanged) should shorten
the approach phase without the ×0.5 shrink that made cycle 4 harmful.

### Change (functions)
- `_default_kwargs['minimum']`: `sigma_inc = 1.5`, new `delta_max = 0.5`.
- `Sella.__init__`: `self.delta_max`.
- `Sella.step`: growth branch `rho > 1/rho_inc and (ord == 0 or rho <
  rho_inc)`; `delta = min(max(1.5·smag, delta), delta_max)`.

### Result
Train C_S 0.9117889 vs 0.8844994, mean rel_energy 0.999967 → **invalid**
(`energy_below_baseline`). Evidence `evaluation_results/cycle-18-train.json`
(fa1406ea4c73417db625e19b88ab985b). Paired vs champion: 14 basin changes
(+70 calls; 104079126 ends +130 kJ/mol above the champion's basin in 57
calls, i.e. at the reference minimum, losing the +0.127 energy credit;
phenol__pyrrole +10.1 kJ/mol; 440717067 +4.3 kJ/mol in 13 calls); same-basin
455 molecules mean +0.80 calls: PubChem −0.49, rowansci −0.48, dimers
+2.54 (all ref bins +0.8 … +2.4; worst alcohols__guanidiniums +26,
alkenes__phenol +26, amines__amines +24).

### Interpretation
- The growth helps connected molecules (−2.5 % calls) and hurts dimers
  (+10 %): the soft, anharmonic intermolecular surface does not tolerate
  0.3–0.5 Å/rad fragment translation/rotation steps; the quadratic model is
  only valid over ~0.1 rad there. The cycle-4 conclusion ("the approach
  phase is not trust-limited") holds for dimers only.
- The energy failure is a basin lottery on the reactive/fragmenting
  104079126 (the champion's −130 kJ/mol basin is a lucky escape; every path
  change so far loses it) plus conformer hopping on flexible PubChem
  molecules under 0.5 rad torsion steps (440717067).
- Repair target (diagnosed cause: TR coordinates over-grown): give the
  fragment translation/rotation internals their own radius under Sella's
  original policy.

### Next idea/control
Repair 1 (cycle 19): separate `delta_tr` for the TR internals.

## Cycle 19 — growth-only repair 1: separate radius for fragment TR internals (invalid)

Champion at start: `9c106c78fda1a4fcf8ec5a5d863bd5a770ca0d30`. Candidate:
`c5b89d72d82f36e02958ac43c35a1fd723945fdc` (on top of cycle 18).

### Hypothesis
With the TR internals on their own radius (`delta_tr`, Sella's ×1.15 window
policy) and the covalent internals on the growth-only policy, the dimer
loss of cycle 18 disappears while the PubChem gain stays.

### Change (functions)
- `Sella.__init__`: `self.delta_tr = delta0`, `self.sigma_inc_tr = 1.15`.
- `Sella._has_tr_internals`: TR coordinates present and `MaxInternalStep`.
- `Sella._predict_step`: `rs_kwargs['wx'] = delta/delta_tr` (mirrors the
  vendored `wc = delta/delta_cell` pattern; weights wx apply to the first
  `ntrans` and last `nrotations` components of `Internals._names` order).
- `Sella.step`: per-part step magnitudes (`smag_int` over bonds/angles/
  dihedrals/other, `smag_tr` over TR); shrink applies one common factor
  `0.9·max(smag_int/delta, smag_tr/delta_tr)` to both radii (Sella's rule
  generalised so the non-limiting part is not collapsed); growth ×1.5 (cap
  0.5) for the covalent radius, ×1.15 inside the window for `delta_tr`.

### Result
Train C_S 0.8769714 vs 0.8844994 (−0.0075), mean rel_energy 0.999799 →
**invalid** (`energy_below_baseline`). Evidence
`evaluation_results/cycle-19-train.json` (d97d8604efb64267846a0550ce0961ba).
Paired vs champion: same-basin 463 molecules mean −0.13 calls: PubChem
−0.51, rowansci −0.48, dimers −0.02 (ref bins +0.06 / −0.15 / −0.39 /
+0.08); same-basin C_S contribution −0.0060 (PubChem −0.0038, dimers
−0.0019, rowansci −0.0003), basin part −0.0015. Energy budget
sum(rel−1) = −0.094 vs +0.224: 104079126 −0.127 (rel 1.000 in 54 calls),
carboxylates__ketones −0.095 (rel 1.199 → 1.103, +1.5 kJ/mol, 35 → 47
calls), 440717067 −0.067 (rel 0.933, 13 calls), sulfides__thiols −0.042,
135104646 −0.010; phenol__phenol +0.023. PubChem results identical to
cycle 18 except 104079126 (which fragments during the run and therefore
takes the TR code path) — confirms the evaluator's determinism per code
path.

### Interpretation
- The repair works mechanically: dimers are back to neutral (−0.4 %
  same-basin), PubChem keeps −0.8 %. The candidate would be a new best on
  calls but fails the energy gate by lottery: the champion's +0.224 budget
  is two lucky basins (104079126 +0.127, carboxylates__ketones +0.199), and
  the dimer path changes (covalent radius of pre-relaxed monomers grown
  ×1.5 for no measurable benefit) erased the second.
- Diagnosed cause for repair 2: the growth-only policy applied to the
  covalent coordinates of multi-fragment systems changes their paths
  without benefit; restricting the policy to connected systems reproduces
  the champion's dimer results exactly (deterministic evaluator; predicted
  train C_S 0.87593, sum(rel−1) +0.021) while keeping the PubChem gain.

### Next idea/control
Repair 2 (cycle 20): growth-only policy only when the internal set has no
TR coordinates; multi-fragment systems keep Sella's single-radius policy.

## Cycle 20 — growth-only repair 2: connected systems only (non_generalizable)

Champion at start: `9c106c78fda1a4fcf8ec5a5d863bd5a770ca0d30`. Candidate:
`75cce0a859dd48530215d84187f7366099841410` (champion + minimal change; the
`delta_tr` machinery of cycle 19 dropped).

### Hypothesis
See cycle 19: dimers identical to the champion, connected molecules on the
growth-only policy → valid on train with C_S ≈ 0.876, and the PubChem gain
(−0.8 % same-basin) should transfer to the valid split.

### Change (functions)
- `_default_kwargs['minimum']`: `sigma_inc_mol = 1.5`, `delta_max_mol =
  0.5` (Sella's `sigma_inc = 1.15` kept for the window branch).
- `Sella._has_tr_internals`: `pes.int.ntrans + pes.int.nrotations > 0`.
- `Sella.step`: new branch `ord == 0 and rho > 0.75 and not
  _has_tr_internals()` → `delta = min(max(1.5·smag, delta), 0.5)`; the
  original window branch follows for everything else.

### Result
Train C_S 0.8771281 vs 0.8844994 (Δ +0.0074), mean rel_energy 1.000044
(valid, exactly as predicted); valid C_S 0.8939665 vs 0.8936084
(Δ −0.00036) → **non_generalizable**. Evidence
`evaluation_results/cycle-20-train.json` (df090c1600e046cdaa3de5546b06650a),
`evaluation_results/cycle-20-valid.json` (7481334f9a0a44f09533ae743b212288).
Train: 7 of 219 dimers differ from the champion (+16 calls; dimers detected
as one fragment or fragmenting mid-run); same-basin non-dimer 247 molecules
mean −0.15 calls, mean rel d −0.0078, 84 wins (−178 calls) / 59 losses
(+142); same-basin C_S contribution −0.0041. Valid: same-basin non-dimer
247 molecules mean +0.07 calls, mean rel d −0.0035, 96 wins (−182) / 59
losses (+200), same-basin C_S contribution −0.0019; basin changes +49
calls (135011106 51 → 87, 104130706 30 → 36, 135065494 109 → 116); heavy
loss tail 252662734 32 → 67 (ref 34), 252648738 36 → 54, 135145908 25 →
38 (train tail: 384221749 58 → 77, 135098013 18 → 28, 252657041 32 → 41,
osimertinib 31 → 39).

### Interpretation
- The mechanism's same-basin effect is real on both splits (−0.41 % / −0.19 %
  of C_S; 96 wins vs 59 losses on valid) but heavy-tailed: a few molecules
  lose 10–35 calls, enough to cancel the many 1–9-call wins on valid.
- Candidate cause of the tail: after ×1.5 growth to the 0.5 cap the model is
  poor over 0.5 rad torsion steps; Sella's shrink (×0.9 of the failed step,
  matched to ×1.15 growth) needs ~15 failed steps to return from 0.5 to 0.1,
  and every failed step is an accepted uphill move. Textbook policies pair
  fast growth with a ×0.25–0.5 shrink (cycle 4 tested ×0.5 on ρ < 0.25
  with growth ×2/cap 1.0 under the cycle-3 curvature model: neutral for
  PubChem then).
- Alternatives: the tail is a basin/path lottery unrelated to the radius
  (the losses of 252662734 etc. are within 0.2 kJ/mol of the champion's
  minimum, so the paths differ but end in the same minimum — consistent
  with both readings); the cap 0.5 itself may be too large for torsions.

### Next idea/control
Repair 3 (cycle 21, last of the bounded repairs): connected systems halve
the radius on a failed step (`sigma_dec_mol = 0.5`, Sella's ρ < 0.01 / ρ
> 100 trigger unchanged); multi-fragment systems keep ×0.9.

## Cycle 21 — growth-only repair 3: ×0.5 shrink on failure for connected systems (keep)

Champion at start: `9c106c78fda1a4fcf8ec5a5d863bd5a770ca0d30`. Candidate:
`630dd79df90bd3681e846ca2de6ef318667f4f46` (cycle 20 + shrink change).

### Hypothesis
The heavy loss tail of cycle 20 comes from slow recovery after the radius
has grown to the 0.5 cap: Sella's ×0.9-of-the-failed-step shrink needs ~15
accepted uphill steps to return to 0.1. Halving on failure (ρ < 0.01 or
ρ > 100, trigger unchanged) for connected systems should remove the tail
and keep the growth gain; multi-fragment systems stay on Sella's policy.

### Change (functions)
- `_default_kwargs['minimum']`: `sigma_dec_mol = 0.5`.
- `Sella.__init__`: `self.sigma_dec_mol`.
- `Sella.step`: shrink branch uses `sigma_dec_mol` when `ord == 0 and not
  _has_tr_internals()`.

### Result
Train C_S 0.8749174 vs 0.8844994 (Δ +0.0096), mean rel_energy 1.000099;
valid C_S 0.8861966 vs 0.8936084 (Δ +0.0074), mean rel_energy 1.007633 →
**keep**. Evidence `evaluation_results/cycle-21-train.json`
(a7bab73733bf45b2a080dafd8db2e8dd), `evaluation_results/cycle-21-valid.json`
(7944103de42d4be6a0f9445281af4376). Paired vs cycle-13 champion — train:
same-basin non-dimer 245 molecules mean −0.23 calls (rel −0.92 %), 92 wins
(−178) / 59 losses (+121), C_S contribution −0.0048; basin changes 5
(−103 calls: 104079126 131 → 58 at the reference minimum, 440717067 31 → 13,
135095518 27 → 17, venetoclax 53 → 54 at −5.8 kJ/mol); dimers +0.06.
Valid: same-basin non-dimer 248 molecules mean −0.29 calls (rel −1.34 %),
105 wins (−225) / 66 losses (+153), C_S contribution −0.0072; basin
changes 135011106 51 → 46 (+21.5 kJ/mol, rel still ≥ 1 budget-wise:
valid sum(rel−1) +3.55), 135065494 109 → 90 (−9.1 kJ/mol); dimers +0.005.
Tail vs cycle 20: 252662734 67 → 44 (champion 32), 135145908 38 → 33,
384221749 77 → 74, 135098013 28 → 33 (worse).

### Interpretation
- The textbook pairing (fast growth, fast shrink, cap) works for connected
  molecules under the current curvature model: −0.9 % / −1.3 % same-basin
  on train/valid, loss tail cut by a third to a half. Cycle 4's neutral
  result for PubChem was under the cycle-3 curvature model, with ×2 growth,
  cap 1.0 and ρ < 0.25 shrink trigger; the difference in outcome is most
  plausibly the curvature model (multi-secant block, bend scale) making
  larger steps reliable, with the cap (0.5 vs 1.0) as a second factor.
- Remaining tail (135098013 18 → 33, 384221749 58 → 74) is unexplained
  without trajectories; possible cause is a growth to 0.5 rad on a torsion
  that then needs several halvings; cap 0.3 or per-type caps (bonds 0.3 Å,
  angles/dihedrals 0.5 rad) are cheap follow-ups.
- The energy budget on train is thin (+0.0466 total, +0.0001 mean): the
  champion's lucky 104079126 basin is gone, carboxylates__ketones +0.199 and
  imidazolium__ketones +0.030 carry it. Any future dimer path change risks
  the gate; a candidate that changes dimer paths should be expected to
  fail the energy gate ~1/3 of the time by lottery.

### Next idea/control
New champion `630dd79df90bd3681e846ca2de6ef318667f4f46` (train 0.8749174,
valid 0.8861966). Follow-ups on the same mechanism (per-type caps, growth
trigger) are one-parameter experiments; the larger remaining cost is still
the dimers (train dimer C 0.83 vs PubChem 0.92): the slide phase is
trust-limited on TR coordinates but the quadratic model fails there —
so the next fundamental direction is a better model for the TR block
(anharmonic/Morse-like contact model or line-search-free RFO with a
non-quadratic radial model), not a larger radius.

## Cycle 22 — textbook trust-radius policy for multi-fragment systems, TR cap 0.25 (keep)

Champion at start: `630dd79df90bd3681e846ca2de6ef318667f4f46`. Candidate:
`a277fc1cbf2ebec46359133c7228210d71338a9d`.

### Hypothesis
Cycle 18 showed the growth-only policy with cap 0.5 and ×0.9 shrink gives
large gains on the reorientation dimers (amides__phenol −26,
ammoniums__benzene −17, imidazolium__ketones −14) and large losses
elsewhere (+15…+26 on alcohols__guanidiniums, alkenes__phenol,
amines__amines): the soft intermolecular surface does not tolerate 0.5
Å/rad steps and the ×0.9 shrink recovers too slowly. With the cycle-21
pairing (×0.5 shrink on failure) and a smaller cap (0.25 Å/rad) for
systems with fragment TR internals the gains should survive and the
blow-ups shrink.

### Change (functions)
- `_default_kwargs['minimum']`: `delta_max_tr = 0.25`.
- `Sella.__init__`: `self.delta_max_tr`.
- `Sella.step`: minimisation uses `sigma_dec_mol` (×0.5) and the growth-only
  branch for all systems; the cap is `delta_max_tr` when
  `_has_tr_internals()` else `delta_max_mol`. Connected systems unchanged
  (PubChem/rowansci results bit-identical to cycle 21 on both splits).

### Result
Train C_S 0.8658181 vs 0.8749174 (Δ +0.0091), mean rel_energy 1.000509;
valid C_S 0.8829537 vs 0.8861966 (Δ +0.0032), mean rel_energy 1.009161 →
**keep**. Evidence `evaluation_results/cycle-22-train.json`
(e2cb5469b28f4d76acc922eb1b6a966f), `evaluation_results/cycle-22-valid.json`
(7ddbf17ed37f43a29da86eae10025ccc). Paired vs cycle 21 — train dimers:
same-basin −0.37 calls (ref bins −1.00 / 0.00 / −0.14 / −0.05), 10 basin/
energy changes −52 calls (carboxylates__guanidiniums 29 → 12,
esters__guanidiniums 31 → 18, acids__guanidiniums 32 → 20, amides__phenol
78 → 54, carboxylates__pyrrole 45 → 33; acids__esters 16 → 42 at −1.6
kJ/mol); best same-basin ketones__phenol 67 → 50, imidazolium__ketones 58
→ 45, phenol__phenol 46 → 35, carboxylates__ketones 35 → 25; worst
carboxylates__esters 26 → 57, alkanes__carboxylates 19 → 38,
ammoniums__water 21 → 32. Valid dimers: same-basin −0.53 calls (ref ≥ 40
bin −1.71; the < 40 bins +0.05…+0.23), 17 energy changes −13 calls
(amides__ammoniums 42 → 14, carboxylates__guanidiniums 40 → 15,
ketones__ketones 65 → 43; ammoniums__esters 20 → 44, alcohols__esters 18 →
39). Energy budget train sum(rel−1) +0.239 (cycle 21: +0.047).

### Interpretation
- The textbook pairing also pays on dimers: net −1.4 % (train) / −2.0 %
  (valid) of dimer calls, with the long runs (ref ≥ 40) gaining most on
  valid. The variance per molecule is large (±10–30 calls) — path changes
  on the flat intermolecular surface remain a lottery.
- Five ionic/H-bonded dimers now stop at exactly the reference energy
  (rel 1.0000, 0.2–0.9 kJ/mol above the cycle-21 end points) in far fewer
  calls (carboxylates__guanidiniums 12 vs 29). Reading: after a failed
  step on the flat slide (ρ outside (0.01, 100), which near convergence is
  the energy resolution) the ×0.5 shrink limits the next step, the
  displacement criterion is met while the forces are already below
  threshold, and the run stops where the reference's stiff model stops.
  This is the standard trust-region update acting at all phases, not a
  convergence-targeted damping; 209 of 219 train dimers end within 0.1
  kJ/mol of cycle 21, and the mean energy ratio improved. Recorded as a
  boundary observation: no further tightening of the shrink rule for the
  endgame will be pursued, because a shrink rule tuned to trigger on the
  energy resolution would be exactly the prohibited damping.
- Uncertainty: the cap 0.25 was not scanned (0.15/0.35 untested); the
  split-radius variant (cycle 19 machinery) might let the covalent and TR
  parts of a dimer have separate caps, but the covalent part of a dimer
  is pre-relaxed so the single radius suffices.

### Next idea/control
New champion `a277fc1cbf2ebec46359133c7228210d71338a9d` (train 0.8658181,
valid 0.8829537). The trust-region line is now: cap scan (TR 0.15/0.35,
bonds 0.3) as cheap one-parameter cycles; the fundamental line remains the
curvature model of the intermolecular block and the reactive-molecule
internals (bond-formation-aware rebuild).

## Cycle 23 — step rejection (geomeTRIC rule) (discard)

### Hypothesis
Sella accepts every step: after a step whose energy rose by more than the
model's predicted drop (ρ < −1) it continues from the uphill point with the
radius shrunk. geomeTRIC instead undoes such a step (returns to the previous
geometry, whose energy/gradient are cached, so the undo costs no force call)
and retries with the halved radius. If the accepted uphill points of the
champion start detours (clash regions, other basins), undoing them should
save calls.

### Change (functions)
- `PES.reject_step`: restores the saved positions (`PES.save` is called in
  `Sella._predict_step`) and `curr = last.copy()`; `_update` sees the
  matching state hash and does not re-evaluate. The Hessian update from the
  failed pair is kept.
- `Sella.step`: after the bad-internals block, `rho < rho_reject (−1)` with
  `smag > delta_min` → `reject_step()`, `delta = max(0.5·smag, delta_min)`,
  return (trust update skipped).
- `_default_kwargs['minimum']`: `rho_reject = −1.0`; `Sella.__init__`:
  `self.rho_reject`.

### Result
Train C_S 0.8756767 vs 0.8658181 (Δ −0.0099), mean rel_energy 1.000067 →
**discard** (`af7101c283b901b6e22022dc6ecc251df8adf303`, evidence
`evaluation_results/cycle-23-train.json`,
5a7b1cd1ba314923bfcf4dbe41548eca). Paired vs the champion: 1 basin change
(carboxylates__water 43 → 19 at +12.1 kJ/mol, i.e. a higher basin);
same-basin mean +0.19 calls, 43 better / 101 worse; PubChem +0.15 (192 of
225 unchanged), dimers +0.17 (ref < 15 bin +1.79), rowansci set −0.32.
Big wins carboxylates__esters 57 → 27, alkanes__carboxylates 38 → 19,
benzene__phenol 29 → 13, osimertinib 38 → 31; big losses
carboxylates__ketones 25 → 47, 160853090 73 → 81, many +4/+5.

### Interpretation
- With the secant condition B′s = y the next model minimum is the same
  whether the base point is the undone or the uphill point (x₁ − B′⁻¹g₁ =
  x₀ − B′⁻¹g₀ exactly), so on a locally quadratic surface undoing only
  costs the wasted call; the accepted path additionally measures the next
  ρ against the high uphill energy, so it regrows the radius faster. The
  undo can only pay when the uphill point is a bad base (deep clash: the
  model is polluted by wall curvature and creeps out over several calls).
- The many small losses (+1…+5 in 101 molecules) versus few large wins
  match this picture; the wins in dimers are path re-rolls on the flat
  intermolecular surface (see the repairs below).
- Alternative not excluded: the retry from x₀ with the halved radius fails
  again on strongly anharmonic walls (cascade of 2–4 undos), which would
  explain the +4 clusters; no trajectories are available to confirm.

### Next idea/control
Bounded repairs: (1) size guard — reject only steps ≥ 0.05 Å/rad (endgame
overshoots are within the energy resolution); (2) rise threshold — reject
only rises ≥ 0.05 eV (deep clashes); (3) scope to connected systems.

## Cycle 24 — step rejection repair 1: size guard 0.05 Å/rad (discard)

### Hypothesis
If the losses of cycle 23 are endgame undos of small steps (energy changes
at the calculator's resolution), rejecting only steps whose max internal
component is ≥ 0.05 Å/rad removes them and keeps the mid-phase wins.

### Change (functions)
- `_default_kwargs['minimum']`: `reject_min_step = 0.05`; `Sella.__init__`:
  `self.reject_min_step`; `Sella.step`: `and smag >= self.reject_min_step`.

### Result
Train C_S 0.8733385 vs 0.8658181 (Δ −0.0075), mean rel_energy 1.000068 →
**discard** (`abf5ef6c2dd795255e1114962e1140eca7af2bdb`, evidence
`evaluation_results/cycle-24-train.json`,
fcbee25f6bc6472a90bb9ec4e43d0cb7). Versus cycle 23 only 8 molecules
changed (−11 calls); versus the champion 42 better / 99 worse, PubChem
+0.14 (33 changed, 7 better / 26 worse), dimers +0.15, rowansci −0.48.
Long champion runs (n ≥ 35, 48 molecules) −1.42 calls on average; short
runs (421) +0.29.

### Interpretation
Nearly all rejections happen at steps ≥ 0.05, i.e. in the mid phase;
the endgame is not where the calls are lost. The size guard is therefore
not the lever; the loss is the systematic cost of the undo on ordinary
overshoots (one wasted call plus the slower radius regrowth).

### Next idea/control
Repair 2: restrict the undo to failures with a large absolute energy rise.

## Cycle 25 — step rejection repair 2: rise threshold 0.05 eV (discard)

### Hypothesis
A shallow rise (overshoot on a soft mode) is recovered in one step from the
uphill point; a rise ≥ 0.05 eV (≈1.2 kcal/mol) marks a wall/clash from
which the accepted path creeps out with a polluted model. Rejecting only
the latter should keep the clash wins and drop the overshoot losses.

### Change (functions)
- `PES.kick`: stores `self.df_actual` (energy change of the kick, eV);
  `PES.__init__` initialises it.
- `_default_kwargs['minimum']`: `reject_min_rise = 0.05`; `Sella.__init__`:
  `self.reject_min_rise`; `Sella.step`: `and self.pes.df_actual >=
  self.reject_min_rise`.

### Result
Train C_S 0.8665735 vs 0.8658181 (Δ −0.0008), mean rel_energy 1.000503 →
**discard** (`560cf373b7bf0774db25c775533699b444295382`, evidence
`evaluation_results/cycle-25-train.json`,
a2eb96c9ef22410388429286eefd5799). 82 molecules changed: PubChem 17 (7
better / 10 worse, +1 call, ΔC_S −0.0007), rowansci 4 (all better, −18
calls: 384221749 74 → 67, osimertinib 38 → 31, venetoclax 54 → 48,
135098013 33 → 27; ΔC_S −0.0011), dimers 61 (23 better / 38 worse, +41
calls, ΔC_S +0.0025). The big dimer wins of cycles 23/24
(carboxylates__esters, alkanes__carboxylates) disappeared: they came from
shallow-rise undos, i.e. from path re-rolls, not from clash avoidance.

### Interpretation
Deep-rise undos help the large flexible molecules consistently on train
and are neutral-to-negative on the small PubChem molecules and on dimers.
On dimers, undoing merely re-rolls the path on the flat intermolecular
surface (as many losses as wins, with the losses concentrated in the long
runs, ref ≥ 25: +0.23/+0.28 calls).

### Next idea/control
Repair 3: scope the deep-rise undo to connected systems (no fragment TR
internals), which keeps the dimer paths and energy credits exactly
(deterministic evaluator); predicted train C_S 0.8640.

## Cycle 26 — step rejection repair 3: connected systems only (non_generalizable)

### Hypothesis
See cycle 25: keep the undo (step ≥ 0.05 Å/rad, rise ≥ 0.05 eV) for
connected systems only; dimers follow the champion's accept-always path.

### Change (functions)
- `_default_kwargs['minimum']`: `reject_tr = False`; `Sella.__init__`:
  `self.reject_tr`; `Sella.step`: `and (self.reject_tr or not
  self._has_tr_internals())`.

### Result
Train C_S 0.8643830 vs 0.8658181 (Δ +0.0014, predicted 0.8640; passes
1e-4), mean rel_energy 1.000509; valid C_S 0.8855485 vs 0.8829537
(Δ −0.0026), mean rel_energy 1.009147 → **non_generalizable**
(`6951d9e71fdd619a7ffcb906011d04cb61366479`, evidence
`evaluation_results/cycle-26-train.json` 9a6fd90369b44366a629a4d7a0f48fec,
`evaluation_results/cycle-26-valid.json` 0b7ab797eb57406890f153e2153e029f).
Valid: dimers bit-identical to the champion; PubChem 13 better / 23 worse
(+48 calls): 252633479 42 → 33, 135011106 46 → 39, 135145908 33 → 26,
135192963 41 → 35, 313065814 28 → 22 vs 135065494 90 → 113 (higher basin,
+9.05 kJ/mol), 252662126 31 → 41, 134985308 55 → 65, 135050336 51 → 60,
252653478 24 → 30, 103958198 21 → 26.

### Interpretation
- The deep-rise undo is a coin flip on connected molecules as well: the
  train gain rested on four large flexible molecules of the rowansci set
  (which the valid split does not contain) and the PubChem balance is
  slightly negative on both splits. The undone uphill point is, on
  average, as good a base as the previous one — the accepted path's fresh
  gradient at the uphill point is worth the call it cost.
- Uncertainty: the 0.05 eV threshold and the ×0.5 radius after the undo
  were not scanned; a much larger threshold (≥ 0.2 eV, true blow-ups
  only) would touch few molecules and might be neutral-positive, but the
  observed spread (±10 calls per molecule on valid) says the expected gain
  is below the gate. Trajectory-level evidence (which events cascade) is
  unavailable through the passive diagnostics (no artifacts for converged
  runs).

### Next idea/control
Repair budget for cycle 23 exhausted; champion restored
(`git reset --hard a277fc1cbf2ebec46359133c7228210d71338a9d`).
Implementations preserved in `ideas/step-rejection/` (cycle 23 and cycle
26 versions). Next: return to the trust-region cap scan / the curvature
model of the intermolecular block; the step-rejection line is closed
unless trajectory evidence becomes available.

## Cycle 27 — initial trust radius 0.25 Å/rad for connected systems (invalid)

### Hypothesis
Most connected molecules of both sets start far from their minimum (max
atomic displacement over the champion's run ≥ 1 Å for 99/250 train and
148/250 valid molecules, RMSD ~0.9 Å), so their first quasi-Newton steps
are trust-limited: from δ₀ = 0.1 the ×1.5 growth needs four accepted
steps to reach the 0.5 cap, from 0.25 only two. Starting connected
systems at δ₀ = 0.25 should save 1–3 calls on the far-from-minimum
molecules; multi-fragment systems keep 0.1 (their paths, and their
energy-gate credits, stay bit-identical).

### Change (functions)
- `_default_kwargs['minimum']`: `delta0_mol = 0.25`.
- `Sella.__init__`: `delta0 = default['delta0_mol']` when `order == 0`
  and `not self._has_tr_internals()`.

### Result
Train **invalid** (`73ea9a9c801d20e8a18ac7150d0cac1ead307346`, evidence
`evaluation_results/cycle-27-train.json` d3c74a7dc5054f1d84196935315088f9):
468/469 converged, one internal_error — rivaroxaban, xtb SCC failure at
call 2 (the first optimizer step). Provisional C_S of the other 468
molecules ≈ 0.857 (not scored).

### Interpretation
- Diagnostic artifact `diagnostics/d3c74a7dc5054f1d84196935315088f9/
  cef263ed…json`: the first step moved Cl by 3.33 Å and the whole
  chlorothiophene ring by 1.6–2.9 Å (RMS 1.07 Å); the minimum non-bonded
  distance dropped from 2.03 Å to 0.86 Å (C…H). Along the linear
  interpolation of that step the contact is still 2.0 Å at 0.67 Å max
  displacement, 1.68 Å at 1.0 Å and 1.16 Å at 1.67 Å. The champion's
  whole run moves rivaroxaban's atoms by at most 0.96 Å (RMS 0.37 Å).
- Cause: the max-internal-component trust region is blind to lever arms.
  A 0.25 rad change of several dihedrals along a chain compounds: a
  distal group swings by Σ rᵢ·θᵢ ≈ several Å even though every internal
  component is within the radius. The quadratic model (guessed Hessian,
  no non-bonded terms) is nowhere near valid over such moves.
- Alternatives considered: (a) scale dihedral weights by lever arm —
  ignores compounding; (b) revert to δ₀ = 0.1 — leaves the same
  failure mode for the 0.5 cap steps and gives up the hypothesis.

### Next idea/control
Repair 1 (cycle 28): add the linearised max atomic displacement of the
internal step (`Binv @ s`, real atoms) to the trust-region measure so the
radius bounds both the internal components and the Cartesian swing
(cap = 2δ Å), connected systems only.

## Cycle 28 — cycle 27 repair 1: Cartesian bound (2δ Å) on the internal trust region (keep)

### Hypothesis
Keep δ₀ = 0.25 for connected systems, but let the trust region also
bound the linearised Cartesian displacement of the internal step: the
step lies in range(B), so `Binv @ s` is the minimum-norm Cartesian move
realising it (no net translation/rotation), and its largest atomic norm
divided by 2 is treated as a second step-size measure. The radius then
means "max internal component ≤ δ and max atomic swing ≤ 2δ Å"
(0.5 Å on the first step, 1.0 Å at the 0.5 cap). This removes the
lever-arm/compounding blind spot diagnosed in cycle 27 while leaving
ordinary bond/angle steps (≤ 2δ Å by construction) unchanged. Dimer
paths stay identical unless their internal set loses the fragment
TR coordinates during the run.

### Change (functions)
- `_default_kwargs['minimum']`: `cart_ratio_mol = 2.0`; `Sella.__init__`:
  `self.cart_ratio_mol`.
- `Sella._predict_step`: passes `cart_ratio` to `MaxInternalStep` when
  `order == 0` and `not self._has_tr_internals()`.
- `MaxInternalStep.__init__`: `cart_ratio`, `_ncart_atoms`;
  `MaxInternalStep.cons`: value = max(max|s·w|, max_a‖(Binv s)_a‖ /
  cart_ratio) with the derivative of the active branch (the Newton/
  bisection solver of `BaseRestrictedStep.get_s` is unchanged); the
  returned `smag` therefore drives the radius update consistently.

### Result
Train C_S **0.8547638** vs 0.8658181 (Δ +0.01105), mean rel_energy
1.001053 (was 1.000509), 469/469 converged; valid C_S **0.8654398** vs
0.8829537 (Δ +0.01751), mean rel_energy 1.009366 (was 1.009161), 465/465
converged → **keep** (`291e48613748fb810ddabc67fb393ee65f2946a3`,
evidence `evaluation_results/cycle-28-train.json`
7e98ad43caf041dd9ffb3945b180eb78, `evaluation_results/cycle-28-valid.json`
5e4cdc96837545a3b7f8910a7d250ee7).
Paired vs cycle 22: train connected 250 molecules mean −0.51 calls
(106 better / 76 same / 68 worse; PubChem −0.40, rowansci/other −1.56;
ref ≥ 40 calls: −5.75), dimers 7 changed, net 0.00; valid connected
mean −0.88 calls (131 better / 56 same / 63 worse; ref [25,40): −1.24,
ref ≥ 40: −3.12), 12 dimers changed, net −0.01. Basin changes 5 (train)
/ 7 (valid). Largest gains: 104079126 58 → 29, 135098013 33 → 16,
venetoclax 54 → 39, osimertinib 38 → 24 (train); 251920090 59 → 34,
135011106 46 → 28, 135192963 41 → 27 (valid). Largest losses: 440717067
13 → 28, 135095518 17 → 26 (train); 135239978 20 → 40, 104130706 28 → 36
(valid). rivaroxaban 29 → 27, no SCC failures.

### Interpretation
- The gain is broad-based and grows with the reference length (the
  far-from-minimum molecules), as the cycle 27 hypothesis predicted; the
  Cartesian bound made the larger initial radius safe (no SCC failures,
  fewer basin changes than a typical dimer-path change). The two
  mechanisms were not separated: δ₀ = 0.25 alone is invalid (cycle 27),
  and the Cartesian bound alone at δ₀ = 0.1 is an open control (it
  would cut the champion's long-lever first steps, expected neutral or
  slightly positive).
- The changed dimers (7/12) are systems whose internal set is rebuilt
  during the run (fragments merging, TR coordinates disappearing), after
  which the Cartesian bound applies per step; their net effect is zero.
- Uncertainty: the ratio 2 and the scoping to connected systems were not
  scanned. The losses (a handful of molecules doubling their calls) look
  like path changes from the larger first steps rather than a systematic
  cost; a lower ratio (1.0–1.5) would trade those against the long-lever
  gains.

### Next idea/control
Follow-ups (backlog `cartesian-trust-region`): ratio scan (1.5 / 3),
extend the bound to multi-fragment systems (fragment rotations of
0.25 rad swing atoms by ~1 Å; a Cartesian cap may let the TR radius cap
rise above 0.25), and the δ₀ = 0.1 control. Champion is now
`291e48613748fb810ddabc67fb393ee65f2946a3`.

## Cycle 29 — Cartesian bound ratio 1.5 (non_generalizable)

### Hypothesis
Cycle 28 vs cycle 27 on the 468 common train molecules shows the bound
itself is worth −0.92 calls per connected molecule (73 paths changed,
43 better / 30 worse, 104079126 136 → 29, venetoclax 61 → 39), so the
ratio may not be at its optimum; a tighter bound (1.5δ: 0.375 Å on the
first step, 0.75 Å at the cap) tests the restrictive side.

### Change (functions)
- `_default_kwargs['minimum']`: `cart_ratio_mol = 1.5`.

### Result
Train C_S 0.8522352 vs 0.8547638 (Δ +0.0025), mean rel_energy 1.000895;
valid C_S 0.8675452 vs 0.8654398 (Δ −0.0021), mean rel_energy 1.009536
→ **non_generalizable** (`a25ea9c7953de0d518e145e57cbc82644630e22d`,
evidence `evaluation_results/cycle-29-train.json`
315b6aedf0844856959ac265fbcc2843 — first attempt returned an
infrastructure error (SSH exit 255), the identical command was repeated;
`evaluation_results/cycle-29-valid.json` d218791111294ea5a3db2da61b35cf35).
Paired vs cycle 28: train connected −0.12 calls (46 better / 50 worse),
valid connected +0.19 (49 better / 64 worse); dimers identical apart
from mid-run rebuild cases.

### Interpretation
- The ratio is on a flat optimum: 1.5 vs 2 moves ±0.1 calls per molecule
  with opposite signs on the two splits, i.e. noise-level path changes.
  Most of the bound's value comes from cutting the multi-Å swings
  (ratio ∞ → 2); the exact cap within 0.4–0.5 Å on the first step does
  not matter.
- Alternative: ratio 3 would sit between 2 and ∞; not worth a cycle.

### Next idea/control
Champion stays `291e48613748fb810ddabc67fb393ee65f2946a3` (restored with
`git reset --hard`). Next: extend the Cartesian bound to systems with
fragment TR internals (fragment rotations are the longest levers).

## Cycle 30 — adaptive per-type curvature scale of the model Hessian (discard)

### Hypothesis
The score is dominated by the bulk endgame (one call saved on every
molecule ≈ −0.05 C_S), whose length is set by the accuracy of the model
Hessian in the soft, not-yet-explored directions. The diagonal guess is
off by a molecule-dependent factor per coordinate type (cycles 9–11
showed ±3–4 % swings from global type scales with opposite optima for
small and large molecules). The newest secant pair measures the model
error along the step; decomposed by type, r_t = (s_t·y_t)/(s_t·(Hs)_t)
estimates the block's scale error, and H ← D H D (D = diag(√(r_t^0.5)),
clipped to [0.25, 4], types with ≥ 2 % of the predicted energy and
positive observed curvature only, first five updates, connected systems)
extrapolates the correction to the unexplored directions of the type; the
multi-secant update then re-imposes the recent pairs exactly.

### Change (functions)
- `InternalPES.__init__`: `type_scale_steps = 5`, `type_scale_power = 0.5`,
  `type_scale_clip = (0.25, 4.0)`, `type_scale_min_frac = 0.02`.
- `InternalPES._update_H` (new override): calls `_rescale_H_by_type`
  before `PES._update_H`; `InternalPES._type_slices`,
  `InternalPES._rescale_H_by_type` (new).

### Result
Train C_S 0.8695090 vs 0.8547638 (Δ −0.0147), mean rel_energy 1.001322
→ **discard** (`d1d43183addc53f9c1c9c6bc96c3e7867972dd94`, evidence
`evaluation_results/cycle-30-train.json` 44747228656a4add8236f89a950b7e7f).
Same-basin: PubChem +1.09 calls (83 better / 113 worse overall), rowansci
+0.04, dimers +0.07 (rebuild cases only); 104079126 left its deep basin
(29 → 123 calls, −130 kJ/mol). Best 384221749 73 → 53, worst 404345632
28 → 45, 135142801 35 → 51, 160853090 77 → 93.

### Interpretation
- The type-wise extrapolation is wrong more often than right: the model
  error along a step is not shared by the other coordinates of the same
  type (torsions of one molecule differ by conjugation and sterics more
  than molecules differ from each other), and the scaling distorts the
  curvature information the earlier updates had placed in the other
  directions (only the last four pairs are re-imposed).
- Alternatives: the first pair only (cycle 31), stronger damping, or
  dihedrals only — the first is the cleanest hypothesis and was tested.

### Next idea/control
Repair 1: `type_scale_steps = 1` (cycle 31).

## Cycle 31 — adaptive type scale repair 1: first secant update only (discard)

### Hypothesis
The first step is the largest and its pair the least noisy; applying the
type-wise correction once, before any update has placed information in
H, avoids distorting learned curvature.

### Change (functions)
- `InternalPES.__init__`: `type_scale_steps = 1`.

### Result
Train C_S 0.8605085 vs 0.8547638 (Δ −0.0057), mean rel_energy 1.001324
→ **discard** (`e783e4be52ab750cb28b15b15681d74edb953270`, evidence
`evaluation_results/cycle-31-train.json` 2ec64f33dab649c2854671f550f72a0f).
Same-basin PubChem +0.50 calls (79 better / 86 worse; molecules < 15
atoms −0.17, 15–40 atoms +0.24…+0.33, > 40 atoms −0.06), rowansci
−0.28; 104079126 again leaves its basin (29 → 124, 95 of the 118 extra
calls). Best 384221749 73 → 63, 135104646 30 → 24; worst 363892164
23 → 43.

### Interpretation
- Even a single, damped correction from the first pair is neutral at
  best on the bulk (+0.05 calls same-basin) and costs the reactive
  borane's lucky basin; the first step is also the most anharmonic one,
  so its secant curvature ratio is not the local curvature ratio.
- The line is closed: the per-type scale hypothesis (correlated guess
  errors within a type) is contradicted on both variants. Preserved in
  `ideas/adaptive-type-scale/d1d43183…/algo.py` for a future
  least-squares variant over several pairs restricted to the endgame,
  if trajectory evidence ever shows type-correlated errors.

### Next idea/control
Champion restored (`git reset --hard 291e48613748fb810ddabc67fb393ee65f2946a3`).
Next: the trust-region cap scan for connected systems now that the
Cartesian bound is in place (delta_max_mol 0.35 / 0.75), or the
bond-formation-aware internals for the reactive molecules.

## Cycle 32 — trust-radius cap 0.75 Å/rad for connected systems (discard)

### Hypothesis
With the Cartesian bound (2δ Å) guarding against lever-arm swings, the
0.5 cap of the connected-system trust region (chosen in cycle 21 without
the bound) may be too conservative for the far-from-minimum molecules:
0.75 rad / 1.5 Å would let them cover their ~1 Å RMSD in fewer accepted
steps.

### Change (functions)
- `_default_kwargs['minimum']`: `delta_max_mol=0.75` (was 0.5); one
  parameter, multi-fragment paths untouched (dimer results bit-identical).

### Result
Train C_S 0.8574170 vs 0.8547638 (Δ −0.0027), mean rel_energy 1.001053
(identical to the champion's) → **discard**
(`290dae279c8574f15d757ddb5ee1b6f9450264b8`, evidence
`evaluation_results/cycle-32-train.json` 5d9c6aedcaca4a43adf7559ffaf01932).
40 molecules changed (16 better / 24 worse), all same-basin: PubChem
+0.16 calls, rowansci +0.04, dimers 0. Best 135043047 36 → 26,
163350562 25 → 18; worst 135098427 31 → 43, 135095297 17 → 25,
135098013 16 → 23.

### Interpretation
- The radius only reaches 0.75 after three consecutive ρ > 0.75 steps;
  only ~40 of 250 connected molecules ever get there, and for them the
  larger steps are as often harmful (overshoot, longer endgame) as
  helpful. With the growth ×1.5 from δ₀ = 0.25, the 0.5 cap is at the
  flat optimum together with the Cartesian ratio (cycle 29): the
  trust-region parameters are exhausted (δ₀, cap, ratio, growth, shrink
  all scanned since cycle 21).
- Alternative: the far-from-minimum cost is not step length but model
  quality (the Hessian guess in the coupled torsion/bend directions),
  which no radius policy fixes.

### Next idea/control
Champion restored (`git reset --hard 291e48613748fb810ddabc67fb393ee65f2946a3`).
Next: leave the trust-region parameters; pursue endgame/model-quality
ideas (secant-pair noise guard, dimer Cartesian bound) or the
bond-formation-aware internals.

## Cycle 33 — safeguarded multi-secant Bofill (SR1/PSB) Hessian update (discard)

### Hypothesis
Sella's TS-BFGS is a saddle-search compromise (Bofill 2003: a Powell-type
rank-2 update weighted by y and |B|s). For minimisation in a trust-region
framework the textbook update is SR1 (Nocedal & Wright ch. 6; Conn, Gould
& Toint 1991; Byrd, Khalfan & Schnabel 1996): it corrects the model along
the observed error direction J = y − Bs, reproduces a quadratic Hessian
after n independent steps without line searches and yields more accurate
Hessian approximations than BFGS-type updates. Blending it with PSB per
secant direction (Bofill 1994, φ = (s·J)²/(|s|²|J|²)) removes the SR1
denominator blow-up. If the endgame of soft modes is limited by the
accuracy of the model in the not-yet-sampled directions, a self-correcting
update should shorten it.

### Change (functions)
- `_MS_Bofill` (new): block form of the update on the eigen-directions of
  M = JᵀS (s_i = S q_i, J_i = J q_i, dual vectors w_i = S(SᵀS)⁻¹q_i);
  per direction φ_i·J_iJ_iᵀ/μ_i + (1−φ_i)·(J_iw_iᵀ + w_iJ_iᵀ − μ_i w_iw_iᵀ),
  which satisfies the full block secant condition for any φ_i and bounds
  the SR1 term by |μ_i|/|s_i|² (the observed curvature error).
- `update_H`: `method='Bofill'` dispatch; `PES.update_method = 'Bofill'`
  and `PES.set_H` passes it to `ApproximateHessian` (all systems).

### Result
Train C_S 1.4483809 vs 0.8547638 (Δ −0.594), mean rel_energy 1.003991,
469/469 converged → **discard** (`8cc5bca9ae123781cd0f5dee7b4319489987ea36`,
evidence `evaluation_results/cycle-33-train.json`
4c9a07c63b404740827e2b22ebcfa673). 90 % of the molecules slower (median
ratio 1.60; dimers 25.2 → 39.4 calls, connected 17.7 → 33.4). Connected
molecules by max displacement of the champion run: < 0.2 Å ×1.12 (n 23),
0.2–0.8 Å ×1.73 (97), ≥ 0.8 Å ×1.86 (130). Worst 134979308 34 → 179,
monoatomics__sulfides 20 → 156, 135064751 31 → 134, paliperidone_palmitate
63 → 150; best 160853090 77 → 55. No run hit the call budget.

### Interpretation
- Decisive and systematic, like the BFGS test of cycle 5 (C_S 1.055):
  in this code the two "accurate" minimisation updates both lose badly to
  TS-BFGS, and the loss sits in the approach phase (large, anharmonic
  steps), not in the endgame (near-minimum molecules ×1.12). The secant
  data of trust-limited steps (0.25–0.5 rad torsions, frame-transported
  gradients y = g_new − B_new B_old⁺ g_old) carries large anharmonic and
  coupling components in J; SR1 imposes them as a rank-1 term along J
  (creating stiff-coordinate couplings and, for negative μ, indefinite
  models whose |λ| steps are of the wrong magnitude), whereas the
  |B|-weighted Powell form of TS-BFGS spreads the correction in the B
  metric and keeps the model close to the guess in the stiff block.
- Not an implementation defect: the single-pair limit is exactly Bofill's
  update, the block form satisfies B⁺S = Y by construction, and no run
  failed or froze. The bounded near-minimum loss also argues against a
  sign/indexing bug (which would break every run).
- Alternative not tested: pure PSB (rank-2 along s) or the Birkholz–
  Schlegel flowchart (SR1 → BFGS → PSB) — both lack the |B| metric, which
  the evidence now identifies as the property that matters here.

### Next idea/control
Champion restored (`git reset --hard 291e48613748fb810ddabc67fb393ee65f2946a3`).
Implementation preserved in `ideas/bofill-sr1-update/8cc5bca9…/algo.py`.
The Hessian-update family is closed (BFGS, SR1/Bofill both rejected).
Next: near-minimum dimers (ref < 20: rel 1.33–1.45 on both splits,
≈ 0.03–0.055 of C_S) — the reference (plain Sella, anchor rel 1.000
everywhere) stops within 10–19 calls while the TR-coordinate champion
slides along the flat intermolecular surface; and the trust-radius
collapse to 0.5·smag after a failed unconstrained endgame step.

## Cycle 34 — non-local contact curvature in the guess Hessian (Lindh-style pair term projected onto the internals) (keep)

Champion at start: `291e48613748fb810ddabc67fb393ee65f2946a3` (train
0.8547638, valid 0.8654398). Cycle number reserved with
`scripts/cycle.py begin` → 34.

### Hypothesis
Offline analysis of the champion's train/valid results (cycle-28 JSON vs
`anchors/*.json`, Kabsch-aligned final geometries): the slowest connected
class is the large folded molecules — 88 train / 101 valid molecules with
≥ 0.3 close non-bonded contacts per atom (graph distance ≥ 4, r below a
vdW-contact limit) run at rel 0.921 / 0.878 against 0.85–0.89 for the other
connected classes (train: 83 mols < 0.05 contacts/atom rel 0.886, 46 mols
0.15–0.3 rel 0.852). Their extra calls sit in a same-endpoint endgame
(rmsd to the reference minimum 0.00 Å for most of the worst cases:
384221749 73 vs 68, 8030412 37 vs 34, 135083911 28 vs 22). Sella's guess
Hessian is diagonal in the redundant internals — bonds, angles and
dihedrals carry the 1-2/1-3/1-4 curvature only. The steric and
hydrogen-bond curvature of the *non-local* contacts that hold a fold
together (1-6 and further apart in the covalent graph, 2.2–4 Å apart in
space) is absent, so the model is too soft along the torsional
combinations that compress those contacts: at a vdW contact the pair
curvature is ≈ 2–3e-4 Ha/Bohr² (LJ: 72ε/R₀²) and an H···O contact at 1.9 Å
≈ 0.012 Ha/Bohr²; with lever arms of 2–3 Å, 20–100 such pairs add
0.01–0.1 Ha/rad² to the fold torsions — the same order as, or larger than,
the torsional guess itself (0.025 Ha/rad² per sp³ bond). Lindh et al.
(CPL 241, 423, 1995) build their model Hessian from *all* atom pairs for
exactly this reason, and the Fischer–Almlöf stretch formula extrapolated
to non-bonded distances (which `_h0_fragment` already uses for
inter-fragment contacts) reproduces the vdW-contact curvature within a
factor 2 (C···C at 3.5 Å: 2.5e-4 Ha/Bohr²). Adding, for every pair of the
same fragment at graph distance ≥ 5 within 3 Å of its covalent-radius sum,
the pseudo-stretch quadratic form k_ij (u_ij·(Δx_j − Δx_i))² expressed in
the internal coordinates through the pseudo-inverse Jacobian
(H_nb = DᵀKD, D_pq = u_ij·(B⁺_j − B⁺_i)_q) gives the model the off-diagonal
information a diagonal guess cannot hold (which torsion combinations are
sterically blocked), without touching the fragment TR block (rigid motions
of a fragment leave its internal distances unchanged, so H_nb has zero
rows/columns there) and without double counting the 1-4/1-5 interactions
that the torsional constant already represents. Expected: the ≥ 0.3
contacts/atom class approaches the rel of the other classes (−0.5 to −1
call on ~90 molecules per split, ≈ −0.005 to −0.01 C_S), small molecules
and dimers nearly unchanged (few non-local pairs). Risk: over-stiff fold
modes (F–A is repulsive-only, no attractive tail) slow the approach phase
of far-from-minimum molecules; dimer path re-rolls through the fragments'
internal contacts.

### Change (functions)
- `Internals._h0_nonlocal_contacts` (new): covalent graph from
  `self.internals['bonds']` (real atoms only), connected-component labels,
  depth-limited BFS distances (cap `min_path=5`); pairs of the same
  component at graph distance ≥ 5 with r < r_cov,i + r_cov,j + 3 Å get
  k_ij = 0.3601·exp(−1.944 (r − r_cov)/Bohr) Ha/Bohr² (the `_h0_bond` /
  `_h0_fragment` stretch formula), converted to eV/Å²; B⁺ = pinv(B, rcond
  1e-6) reshaped per atom, D_pq = u_ij·(B⁺_j − B⁺_i)_q, returns DᵀKD
  (nint × nint) or `None` when no pair qualifies.
- `Internals.guess_hessian`: returns `diag(|h0|) + H_nb` instead of
  `diag(|h0|)`; the projection `P H P` in `InternalPES.__init__` is a
  no-op on H_nb (its rows lie in the column space of B).
- No change to updates, trust region, rebuild logic or the fragment TR
  block (rigid fragment motions leave the pair distances unchanged, so
  the TR rows/columns of H_nb are zero).

### Result
Train C_S 0.8468548 vs 0.8547638 (Δ −0.0079), mean rel_energy 1.001310;
valid C_S 0.8524317 vs 0.8654398 (Δ −0.0130), mean rel_energy 1.009560 →
**keep** (`eacac3d65ab9dc5b36840c4d9c3ff0624a61531d`, evidence
`evaluation_results/cycle-34-train.json` f1a90657e8694b3c986c57a17085d4b9,
`evaluation_results/cycle-34-valid.json` f70f297b74dd477f81a19078dedf7131;
all 469/465 converged). Paired against the cycle-28 files:
- Train: 142 molecules changed, 96 better / 44 worse same-basin (sum
  −139 calls), 2 basin changes (104079126 29 → 142, final energy
  −130 kJ/mol lower, rel_energy 1.127 — a rearrangement to a different
  minimum; 252089162 30 → 31, +5.6 kJ/mol). Connected molecules by
  non-local contact density (qualifying pairs per atom, computed from
  the input geometry): < 0.05 (96 mols) bit-identical; [0.05, 0.15)
  (21) −0.10; [0.15, 0.30) (28) +4.2 including the 104079126 outlier,
  +0.2 without it (8:9); [0.30, 0.60) (44) rel 0.903 → 0.840, −1.18
  calls, 33 better : 5 worse; ≥ 0.60 (61) rel 0.907 → 0.833, −1.79,
  40 : 11. Dimers +0.09 calls on average (219 molecules; 174 with
  < 0.05 pairs/atom essentially unchanged). Best: venetoclax 39 → 27,
  384221749 73 → 61, 135083911 28 → 17, des370k carboxylates__thiols
  29 → 18, 135104646 30 → 21, osimertinib 24 → 16; worst same-basin:
  alcohols__guanidiniums 20 → 29, ketones__phenol 53 → 61, 135097653
  19 → 25.
- Valid: 164 changed, 105 better / 55 worse same-basin (sum −104),
  4 basin changes (sum +1, 134985308 53 → 58 at −15.8 kJ/mol). Connected:
  [0.30, 0.60) (72) rel 0.887 → 0.854, 35 : 13; ≥ 0.60 (71) rel 0.898 →
  0.865, 35 : 20; [0.15, 0.30) (44) −0.14; smaller classes unchanged.
  Dimers +0.01 on average (215). Best 135065494 91 → 80, 103921822
  39 → 31, 135091752 18 → 10; worst 104139123 23 → 34, 135145908 22 → 31.

### Interpretation
- The mechanism landed where the hypothesis put it: the folded /
  contact-rich connected molecules (≥ 0.3 pairs per atom, 105 train /
  143 valid) gained 0.7–1.8 calls each with 3–7 : 1 better : worse
  ratios, and they now run at rel 0.83–0.87 like the other classes,
  i.e. the missing non-local curvature — not the trust region — was
  what held them back. Small molecules and most dimers are untouched,
  which makes the two splits agree (Δ −0.008 / −0.013), unlike the
  fragment-model re-rolls of cycles 13–15.
- The one large loss (104079126, an 18-atom hydroxamate/phosphonate
  anion) is a basin change to a minimum 130 kJ/mol lower, which costs
  113 calls (rebuilds after the rearrangement) and alone is worth
  +0.006 C_S on train; the same-basin gain is therefore ≈ −0.014 on
  train. Converged runs leave no diagnostic bundle. Such rearrangements
  in the first steps are a path effect of the modified early steps, not
  a systematic property of the term (the three molecules with a
  hydrogen-bond-strength pair, k_max ≥ 0.05 Ha/Bohr², improved by 7.5
  and 3 calls on average; 104079126 itself has k_max 0.008 and only 4
  qualifying pairs).
- Uncertainty: the stretch-only Fischer–Almlöf pair curvature has no
  attractive tail and no angular part; the term is evaluated once per
  (re)build at the input geometry, so contacts that form during the
  optimisation are not represented until a rebuild happens (only on
  near-linear angles). The 1-5 pairs (min_path 4) and a global scale are
  untested; the 44-molecule valid class [0.15, 0.30) gained only 0.14
  calls, so the benefit is concentrated in the dense folds.

### Next idea/control
New champion `eacac3d65ab9dc5b36840c4d9c3ff0624a61531d` (train 0.8468548,
valid 0.8524317). Follow-ups of the same mechanism: include the 1-5 pairs
(`min_path=4`) — the torsional constant covers them only around the
central bond; a larger cutoff (`max_excess`) is cheap but the exponential
already makes pairs beyond 4 Å negligible; the same pair term for the
*inter*-fragment contacts (coupling the TR coordinates of a dimer with its
internal torsions) would replace the diagonal `_h0_fragment` block by a
full Cartesian-derived quadratic form (risk: dimer re-rolls, cycles
13–15). The Hessian-carry-over across rebuilds and the bond-formation
rebuild remain the other open directions.

## Cycle 35 — geometry-tracked non-local contact term (H ← H + A(x) − A(x_prev) before every secant update)

Champion at start: `eacac3d65ab9dc5b36840c4d9c3ff0624a61531d` (train 0.8468548,
valid 0.8524317). Candidate: `fc3407d8ecc67de5fe3a9a7ad64eaf6d6f3bad56`.

### Hypothesis
The cycle-34 contact term is evaluated once, at the input geometry (and at
rebuilds). The remaining slowest connected class is the far-from-minimum
one (initial→final RMSD ≥ 0.8 Å: 45 train molecules at rel 0.900, 75 valid
at 0.861, ~30 calls each), and these molecules *fold* during the run: the
qualifying-pair curvature Σk at the champion's final geometry is 1.5–2.2×
that at the input (train 0.010 → 0.015 Ha/Bohr², valid 0.006 → 0.014;
pairs per molecule 28 → 37 / 23 → 34; e.g. 384221749 k_max 0.002 → 0.033
with 60 → 115 pairs, 135065494 0.004 → 0.035, 61 → 123 pairs — new
intramolecular hydrogen bonds), while the near-minimum classes hardly
change (RMSD < 0.4 Å: Σk ±0.0001). The secant updates only learn curvature
along the steps taken, so contacts that appear late act in unsampled
directions until the optimizer stumbles on them. Treating the analytic
part of the model as geometry dependent — before every TS-BFGS update,
replace A(x_prev) by A(x) computed with the current pseudo-inverse
Jacobian (which also carries the frame change of the internal
coordinates), then impose the secant conditions on the shifted matrix —
keeps the learned correction and puts the current contacts (including a
covalent-strength k for pairs that approach bonding distance) into the
model immediately. Expected: −1 to −3 calls on the folding molecules
(≈ −0.003 to −0.006 C_S per split), no change for molecules with a static
contact pattern (the shift vanishes when the geometry stops changing, so
no endgame effect), dimers changed only through their intramolecular
folds. Risk: the shift is indefinite when contacts break (the QN stepper
takes absolute eigenvalues), and it interacts with the |B|-weighted
multi-secant update (which re-imposes the last four secant pairs on the
shifted matrix); a step-to-step change of the analytic part could also
perturb the endgame of soft dimers through their internal contacts.

### Change (functions)
- `Internals._nonlocal_pairs(min_path)` (new): the same-fragment pair
  list at graph distance ≥ min_path, cached on the bond set so that the
  flood fill / depth-limited BFS is not repeated at every step.
- `Internals._h0_nonlocal_contacts(..., Binv=None)`: accepts an external
  pseudo-inverse Jacobian (the stepper's cached `_get_Binv()`), otherwise
  computes its own; `guess_hessian` keeps the term used in the guess as
  `_h0_nonlocal_last`.
- `InternalPES.__init__`: when the guess is built (`H0 is None`) it
  records `_nb_prev = A(x₀)` and enables tracking (`_track_nb`); a
  supplied H0 disables it.
- `InternalPES._track_nonlocal_contacts` (new) and `_update_H` override:
  before `PES._update_H` (the multi-secant TS-BFGS update) the model is
  shifted by A(x) − A(x_prev), shapes are checked against the current
  internal set, non-finite shifts are skipped.

### Result
Train `evaluation_results/cycle-35-train.json`
(7251b95c87de49f9838df92da65ee06f): **C_S 0.8324971** (Δ −0.0144 vs
0.8468548), energy 1.000665 (pass). Valid
`evaluation_results/cycle-35-valid.json`
(93e6f09374ac44cc8dc0461332ad3a64): **C_S 0.8397486** (Δ −0.0127 vs
0.8524317), energy 1.009417 (pass). Decision **keep**; champion
`fc3407d8ecc67de5fe3a9a7ad64eaf6d6f3bad56`.

Paired against cycle 34 (`/tmp/paired.py`):
- Train: total −209 calls. Four basin changes: 104079126 142 → 32 (back
  to the reference basin, +130 kJ/mol, the +0.127 energy credit of cycle
  34 is lost again; train margin Σ(r−1) ≈ 0.31), 134997543 38 → 32,
  404345632 32 → 24 (+10.7 kJ/mol), 440717067 29 → 13 (+4.3 kJ/mol).
  Same basin (465 molecules): mean −0.148 calls, 77 better / 38 worse;
  pubchem −0.80, other −0.76, dimers −0.04 (ref ≥ 40 dimers −0.15).
  Largest gains 135124603 31 → 21, 160853090 77 → 68,
  paliperidone_palmitate 58 → 50, venetoclax 27 → 19, 135097653 25 → 18,
  acids__alkenes 34 → 27; largest losses 363892164 24 → 40, 384221749
  61 → 75, carboxylates__thiols 18 → 28, 135241983 14 → 23, 135142801
  34 → 40, alcohols__sulfides 35 → 41.
- Valid: total −180 calls. Two basin changes (134985308 58 → 39 at
  +15.8 kJ/mol, 135255884 30 → 24 at +3.0 kJ/mol). Same basin (463):
  mean −0.335 calls, 101 better / 52 worse; pubchem −0.66, dimers −0.07
  (ref ≥ 40 dimers −0.26). Largest gains alcohols__water 33 → 19,
  135065494 80 → 67, 104139123 34 → 24, 135240396 25 → 15, 252648738
  42 → 33, 135130364 26 → 18; losses 135011106 26 → 40, 135050336
  48 → 56, 103943741 32 → 37, 163343378 28 → 33, 252658999 8 → 13,
  104130706 36 → 40.

By initial→final RMSD class (same-basin molecules, `/tmp/rmsd_paired.py`):

| class | train n | rel c34 → c35 | Δcalls (better:worse) | valid n | rel c34 → c35 | Δcalls (better:worse) |
|---|---|---|---|---|---|---|
| connected < 0.2 Å | 63 | 0.918 → 0.910 | −0.10 (5:2) | 27 | 0.813 → 0.828 | +0.19 (0:1) |
| connected 0.2–0.4 Å | 68 | 0.847 → 0.827 | −0.32 (18:4) | 64 | 0.847 → 0.829 | −0.28 (20:6) |
| connected 0.4–0.8 Å | 73 | 0.843 → 0.826 | −0.38 (23:13) | 83 | 0.828 → 0.796 | −0.77 (38:15) |
| connected ≥ 0.8 Å | 42 | 0.839 → 0.839 | −0.10 (17:12) | 74 | 0.864 → 0.836 | −0.84 (34:19) |
| dimers (all) | 219 | — | −0.05 (14:7) | 215 | — | −0.06 (9:11) |

### Interpretation
The tracked term helps the connected molecules that move 0.2–0.8 Å, and
on valid also the far-moving class (−0.84 calls, 34:19), i.e. exactly the
molecules whose contact pattern changes during the run; the near-minimum
class (< 0.2 Å, where A(x) ≈ A(x₀)) and the dimers are unchanged, as the
hypothesis required. The train far-moving class is flat on average
(17:12) because two of its members (363892164 +16, 384221749 +14) took
long detours — these are folding molecules with many new pairs, where the
indefinite shift when a contact breaks (or the interaction with the four
re-imposed secant pairs) can produce a poor step; the valid split, with
more members in that class, shows the average effect is positive. The
result is consistent across the splits (−0.014 / −0.013) and the gain is
larger than the hypothesis' range (−0.003 to −0.006) because the
0.2–0.8 Å classes, not only the ≥ 0.8 Å class, profit. Uncertainty: the
train energy margin is thinner (104079126 returned to the reference
basin, credit +0.127 lost; the margin Σ(r−1) ≈ 0.31 is now held by a few
dimer credits), so a future path re-roll of one large dimer credit could
fail the train gate without any change in call counts — this is the
energy lottery, not a property of this candidate. Alternative
explanation for part of the gain: the shift acts like a partial Hessian
refresh that counteracts stale secant information in the stiff block;
this would predict gains even for molecules without new contacts, which
the < 0.2 Å class (no gain) does not show.

### Next idea/control
Follow-ups to the tracked model: (a) a damped shift (only add the
positive part, or clip A(x) − A(x_prev) to the current curvature scale)
to remove the two large train detours — but the sign of the effect is
unknown and the current form is the cleanest; (b) extend tracking to the
whole analytic guess (bond/angle/dihedral diagonal, which changes with the
bond lengths and the dihedral bond-order factor) — the diagonal changes
are small for near-equilibrium bonds, low expected value; (c) the
multi-secant refinements (memory 6–8, minimum block 3) that were untested
before cycle 34 now operate on a better model and may interact
differently. Control for cycle 36: an idea that acts on a different
mechanism than the contact model (e.g. the multi-secant block size), so
that a gain or loss is attributable.

## Cycle 36 — row-dependent prefactor of the Fischer–Almlöf stretch guess (bonds to third-row and heavier atoms, and to boron, 0.45–0.65× softer)

Champion at start: `fc3407d8ecc67de5fe3a9a7ad64eaf6d6f3bad56` (train 0.8324971,
valid 0.8397486).

### Hypothesis
Connected molecules rich in bonds to S, P, Cl, Br, Si, I (and B) are the
slowest remaining class per unit size: with the champion, molecules whose
bonds are ≥ 30 % such "heavy" bonds need 4.0 calls/√N near the minimum
(RMSD < 0.3 Å, 46 train molecules, N ≈ 7) against 2.6 calls/√N for
molecules without heavy bonds (29 molecules, N ≈ 26); a regression of the
champion's calls on √N and the heavy-bond count gives +0.44 (train) /
+0.71 (valid) calls per heavy bond, +0.69 / +0.98 for the reference. Their
heavy bonds also start further from equilibrium (mean |Δr| 0.077 Å vs
0.044 Å for the light bonds of the same molecules). The Fischer–Almlöf
stretch guess `0.3601·exp(−1.944(r − r_cov))` was fitted to first-row
molecules; measured valence stretch constants of bonds to heavier atoms
are 0.54–0.66 (third row: C–S 3.0, S–H 4.0, S=O 10, C–Cl 3.4, C–P 2.9,
P=O 9, Si–C 3.0, Si–H 2.7 mdyn/Å vs 5.4, 6.1, 15.7, 5.8, 5.2, 14, 5.6, 4.5
from the formula), ≈ 0.5 (fourth row: C–Br, H–Br), ≈ 0.4 (fifth row: C–I,
H–I) and ≈ 0.6 for boron of the formula's value, and bonds between two
heavy atoms fall short by about the product (S–S, P–P 0.37, Cl–Cl 0.5).
A model that is 1.5–2.7× too stiff along a group of bonds makes the
quasi-Newton step cover only 40–65 % of the required bond relaxation
per iteration (error factor ≈ 0.4–0.6 per step until the secant pairs
have sampled the group), which for bonds that must relax by 0.05–0.1 Å to
within the 1.8e-3 Å displacement criterion costs 3–5 calls. Scaling the
stretch prefactor per atom by row (row 3: 0.65, row 4: 0.55, row ≥ 5:
0.45, boron 0.65; product for two heavy atoms) puts the guess within
±15 % of the measured constants. Expected: −1 to −3 calls on the 78 train
/ 37 valid molecules with ≥ 30 % heavy bonds and −0.5 to −1 on the 93 /
116 with fewer (≈ −0.01 to −0.02 C_S per split); molecules without such
bonds (79 / 96 connected, ~190 dimers) are bit-identical, so the large
energy credits (carboxylates__ethers, ketones__monoatomics with unbonded
Cl⁻, carboxylates__ketones, amides__amides) are preserved; dimers with
thiol/sulfide fragments (≈ 30 per split) change path through their
intramolecular S bonds. Risk: if GFN2-xTB bonds to heavy atoms are as
stiff as the unscaled formula, the softened guess overshoots (error
factor −0.5 per step, i.e. no faster than before, plus oscillation) and
the class gets slower — the decision is symmetric evidence about the
xTB stiffness of these bonds. The contact terms (`_h0_fragment`,
`_h0_nonlocal_contacts`) keep the unscaled tail, since their calibration
(cycles 3, 14/15, 34/35) is independent of covalent stretch constants.

### Change (functions)
Candidate commit `2326fbaf173207b1b6755f6bc6dc3f92d88271a6`
(+28/−1 lines vs the champion, `Internals` class only):
- `Internals._stretch_row_factor` (class attribute `{3: 0.65, 4: 0.55,
  5: 0.45, 6: 0.45, 7: 0.45}`) and `Internals._stretch_prefactor(numbers)`
  (classmethod): product over the bond's two atoms of 1 (Z ≤ 10, dummy),
  0.65 (boron) or the row factor (Z > 10; rows by the noble-gas
  boundaries 18/36/54).
- `Internals._h0_bond`: the Fischer–Almlöf value
  `Ab·exp(−Bb(r − r_cov)/Bohr)` is multiplied by `_stretch_prefactor` before
  the unit conversion. `_h0_fragment` and `_h0_nonlocal_contacts` keep the
  unscaled Ab/Bb tail; angles, dihedrals and the TR blocks are untouched.

### Result
- Train (`evaluation_results/cycle-36-train.json`, evaluation
  `c307c0f1aa804da59ab9f9664823d739`; an earlier identical run of the
  uncommitted file, evaluation `88f5d64a89674e3da2b8547ff4ea58da`, gave the
  same 0.8277394771237356 — the record step refused it because `algo.py`
  differed from HEAD, so the candidate was committed and the evaluation
  repeated): C_S **0.8277395** vs 0.8324971 (Δ −0.0048, gate passed),
  mean rel_energy 1.0010233 (valid), 469/469 converged.
- Valid (`evaluation_results/cycle-36-valid.json`, evaluation
  `7bd02ef01d9d4f07b8039b864c3a4cde`): C_S **0.8415559** vs 0.8397486
  (Δ +0.0018), mean rel_energy 1.0094680 (valid), 465/465 converged.
- Decision (`results.tsv`, cycle 36): **non_generalizable** ("valid
  improvement is not greater than 1e-4"); champion stays `fc3407d`.
- Paired (`/tmp/paired.py`, `/tmp/heavy3.py`, `/tmp/heavy4.py`): all 255
  train / 267 valid molecules without a heavy bond are bit-identical in
  calls and energy, as designed; the energy credits survived on both
  splits (train margin rose to Σ(r−1) ≈ 0.48 because 104079126 fell into
  its −130 kJ/mol basin again: 32 → 130 calls, +0.0051 C_S; without that
  flip train would be ≈ 0.8226, Δ −0.0099).
  Heavy-bond molecules: train connected (170, excl. 104079126) Σ −37 calls
  (65 better / 46 worse, median 0), train dimers with S (41) Σ −18; valid
  connected (154) Σ **+57** (43 better / 46 worse, median 0), dimers (44)
  Σ −6. The small changes (|Δ| ≤ 3 calls) sum to −34 on train (n 157,
  mean −0.22) but only +3 on valid (n 139, mean +0.02); the large ones
  (|Δ| ≥ 4) to −3 (train, 13 molecules) and +54 (valid, 15 molecules:
  135065494 +21 [S(VI) with S–O 1.40–1.69 Å and a 1.66 Å S–H], 135046932
  +18 [tri-thiol], 252648738 +12 [sulfonamide], 135239978 +11 [two
  sulfonate esters], 135011106 −11, 163343378 −10, 104130706 −8).
  By element/class the sign flips between the splits: Br train −16
  (11 better / 3 worse) vs valid +12 (1 / 8); P(V) −17 vs +7; thiols −6
  vs +16; S(VI)O₃₊ −5 (n 2) vs +18 (n 8); Cl −1 vs −27; thioethers −2 vs
  −14; sulfoxides −8 vs −3; S–S −3 vs −6.

### Interpretation
The softened stretch guess does not produce the predicted systematic
saving: even the |Δ| ≤ 3 part of the paired differences, which should
carry a 1–3 call saving per heavy-bond-rich molecule if bond relaxation
were the bottleneck, is −0.22 calls/molecule on train and +0.02 on valid,
and the per-class signs flip between splits. The plausible reason is
that the stiff stretch directions are exactly the ones the multi-secant
TS-BFGS update learns within the first two or three steps (their secant
pairs have the largest |y| and are sampled first), so the diagonal of
the guess along bonds matters only for the first step or two, while
every change of those first steps re-rolls the path of the soft-mode
approach and the endgame (±10–20 call detours on 4–8 molecules per
split). Cycles 34/35 gained because they added curvature that the update
cannot infer cheaply (off-diagonal steric coupling among many soft
coordinates); cycle 36 adjusted curvature the update already corrects.
Alternatives not excluded: (i) GFN2-xTB heavy-atom bonds could be
stiffer than the valence force-field constants used for the calibration
(the overshoot branch of the hypothesis) — the mixed signs argue against
a large systematic error in either direction; (ii) the 0.65 third-row
factor might be right for S=O/C–S but wrong for hypervalent S(VI)
centres (valid S(VI)O₃₊ +18 over 8 molecules, train −5 over 2), whose
S–O single bonds and O–S–O bends are poorly represented anyway — too few
molecules to tell; (iii) a smaller factor set (e.g. 0.8/0.7/0.6) would
change fewer paths but also cannot recover a saving that does not exist
in the small-|Δ| part. The train gain was therefore a favourable
re-roll, and the noise floor for guess-diagonal changes on connected
molecules is ≈ ±0.005 C_S per split — two independent split results are
needed before believing any effect of that size.

### Next idea / control
The candidate is preserved at
`ideas/stretch-row-factor/2326fbaf173207b1b6755f6bc6dc3f92d88271a6/algo.py`
(backlog `stretch-row-factor`). Diagonal-only guess refinements for
bonds (and, by the same argument, the "angle guess at heavy centres"
follow-up) are parked: the update learns stiff diagonals fast. The
remaining levers for connected molecules are off-diagonal/soft-mode
model curvature that the update cannot learn cheaply, and the endgame
(consecutive-criteria count) — the next cycle looks at the step
computation itself rather than the guess.

## Cycle 37 — class-resolved torsional guess for rotatable bonds (acyclic single bonds to a planar sp² centre scaled to their rotational-barrier curvature; connected systems only)
Champion at start: `fc3407d8ecc67de5fe3a9a7ad64eaf6d6f3bad56` (train 0.8324971, valid 0.8397486).

### Hypothesis
The cycle-36 "Next idea" pointed at the step computation, but the
endgame analysis done before this cycle says otherwise: the linear
endgame rate ε ≈ 0.5/step of near-minimum connected molecules is set by
the model accuracy along soft modes with a real gradient, i.e. angles and
torsions, because the multi-secant update learns the stiff stretch block
within the first steps (cycle 36) but cannot cover 10–20 torsions with a
handful of secant pairs. Working out the champion's rotational
stiffness Σ_d h_d per central bond (Fischer–Almlöf × bond-order factor,
1/√n over the n redundant dihedrals) against literature barrier
curvatures n²V_n/2 gives: sp³–sp³ C–C 0.026 Ha/rad² (physical 0.021,
fine), C=C 0.16 (0.19), aromatic ring bond 0.083 (UFF 0.086), amide C–N
0.095 (0.057), ester O–C(=O) 0.045 (0.035) — but sp²–sp³ links 0.025
(propene 0.014, acetaldehyde 0.008, ethylbenzene 0.004), aryl–O 0.034
(anisole/phenol 0.010), aryl–N 0.049 (aniline ~0.017), biphenyl/styrene/
butadiene 0.023–0.033 (0.008–0.019). The 1/√n sharing was calibrated by
the n = 9 sp³–sp³ case (cycle 9); bonds to a planar centre have n = 2–6
and are left 2–3.5× too stiff. For a mode whose model stiffness is ρ×
the true one the per-step error factor is |1 − 1/ρ| (ρ = 2.5 → 0.6,
ρ = 1.5 → 0.33; ρ = 0.5 → 1, so erring soft is worse than erring
stiff), which is why the classes are scaled to the stiff end of their
physical range: sp²–sp³ ×0.6, sp²–lone-pair heteroatom ×0.45, carbonyl–
lone-pair ×0.7, sp²–sp² single ×0.65, each fading back to 1 with the
bond-order factor between 1.5 and 2.2 (acyclic double bonds, amidinium/
guanidinium links keep the full constant). Ring bonds (≤ 8 atoms) are
untouched: their dihedrals describe puckering, and cycle 7 showed ring
torsions want the full constant (aromatic-bearing molecules +6–13 %
when the ring torsions were halved). Multi-fragment systems keep the
champion model so that the dimer basins (train energy margin +0.31 held
by dimer credits) are preserved bit for bit.
Static census of the affected set (train/valid connected molecules
whose covalent graph has ≥ 1 scaled acyclic bond, 1.25 × r_cov bonding
as in Sella): 138/257 train and 189/262 valid connected molecules, on
average 19.6/20.8 calls at rel 0.824/0.825 (scaled bonds per molecule:
1 → 30/33 molecules, 2–3 → 49/81, 4–6 → 45/59, ≥ 7 → 14/16; the drug-like
set venetoclax, acalabrutinib, osimertinib, nintedanib, lenvatinib carry
8–10 each). Prediction: −0.5 to −1.5 calls on the affected molecules
(leverage ≈ −0.012/−0.016 C_S per call on train/valid), so C_S ≈ 0.820
/ 0.825; molecules without scaled bonds and all dimers bit-identical.
Energy gate: the unaffected part carries +0.42 (train) / +4.29 (valid)
of the margin; the affected part −0.11 / +0.09 (train credits at risk:
135249644 +0.079; debits that could shrink: 404345632 −0.074, 440717067
−0.067). Failure modes: (i) the affected set is not slower than the
unaffected one (rel 0.824 vs 0.872 on train — so no visible "torsion
penalty" in the census; the gain, if any, is in the endgame length, not
in a pathology class); (ii) if the update already learns these torsions
in the first steps the change is another ±0.005 path re-roll (cycle 36
lesson); (iii) a class set too soft for some member (e.g. propene-type
sp²–sp³ at 0.015 vs physical 0.014 is at the edge) would overshoot.

### Change (functions)
- `Internals._torsion_centre_types(adj)`: per real atom, from element and
  covalent neighbour count: `'pi'` (C/B with 3 neighbours, N with 2, N with
  3 and a terminal O/S neighbour = nitro/N-oxide), `'lp'` (N with 3, O/S/Se
  with 2), `'sigma'` otherwise.
- `Internals._in_small_ring(b, c, adj, ring_max)`: BFS from b to c avoiding
  the bond itself, depth ≤ ring_max − 1 (ring_max 8).
- `Internals._torsion_class_factor(b, c, types, adj, bo, ...)`: 1 unless a
  central atom is `'pi'` and the bond is acyclic; class scale s = 0.6
  (pi–sigma), 0.45 (pi–lp), 0.7 (carbonyl pi–lp: the pi centre has a
  terminal O/S neighbour), 0.65 (pi–pi); returns
  `s + (1 − s)·clip((bo − 1.5)/0.7, 0, 1)`.
- `Internals.guess_hessian`: builds the real-atom adjacency from the bonded
  set, computes the factor once per central bond (bond-order factor
  `exp(−2.85 (r − r_cov)/Bohr)` as in `_h0_dihedral`) and multiplies the
  proper-dihedral guess `_h0_dihedral(...)/√n` by it; skipped (factor 1)
  when `ntrans + nrotations > 0` (multi-fragment systems). Impropers,
  dummy dihedrals, bonds, angles, fragment TR and the non-local contact
  term are unchanged.
Control: every multi-fragment system and every connected molecule without
an acyclic bond to a planar centre must reproduce the champion's calls
and energies exactly (119 train / 73 valid connected molecules plus all
dimers).

### Result
Candidate `3ca2b9dd7c07605f17b891694e506742d101e5a6` vs champion
`fc3407d8ecc67de5fe3a9a7ad64eaf6d6f3bad56`. Train evaluation
`a9c4f19458a640088df3d2ba085390dc` ([JSON](evaluation_results/cycle-37-train.json)):
469/469 converged, **C_S = 0.8274668** (champion 0.8324971, Δ −0.0050),
mean rel_energy 1.000898 (valid). Valid evaluation
`0d288340dd2c4df68d6b4a3a10ff75af` ([JSON](evaluation_results/cycle-37-valid.json)):
465/465 converged, **C_S = 0.8379440** (champion 0.8397486, Δ −0.0018),
mean rel_energy 1.009390. Decision **keep** — new champion `3ca2b9d`.
- Controls held: all 212 train / 203 valid multi-fragment systems and
  118/119 train, 73/73 valid connected molecules without a scaled bond are
  bit-identical (calls and energies). The one exception,
  des370k_monoatomics__phenol__train (Mg²⁺ over the phenol ring, one
  fragment at the 1.25 r_cov bonding rule, ring carbons 4-coordinated by
  the cation), kept its 43 calls with a 3e-5 kJ/mol energy difference —
  consistent with a mid-run internal-coordinate rebuild at a geometry
  where the Mg–C contacts have opened and the ring carbons count as
  planar centres (unverified; no bundle for converged runs).
- Affected set (138 train / 189 valid connected molecules with ≥ 1 scaled
  acyclic bond): train −0.37 calls (58 better / 33 worse / 47 same,
  dC_S −0.0050), valid −0.14 calls (73 / 53 / 63, −0.0018). The
  |Δ| < 6-call part is negative on both splits (−0.0041 / −0.0031), unlike
  cycle 36 where it flipped sign. By number of scaled bonds: 1 → +0.43 /
  −0.39; 2–3 → −0.67 / +0.42; 4–6 → −0.33 / −0.24; ≥ 7 → −1.14 / −2.12
  (7:1 and 11:2 better:worse). By size: < 20 atoms +0.08 / +0.47 calls,
  20–40 −0.23 / −0.18, 40–60 −0.91 / −0.47 (train named drugs rel 0.755 →
  0.726; PubChem 0.857 → 0.849 / 0.818 → 0.814).
- Class regression (least squares of Δcalls on the number of scaled bonds
  per class, and presence/absence splits) agrees in sign on both splits:
  pi–pi −0.35 / −0.26 calls per bond (molecules with a pi–pi bond −0.94
  vs +0.14 without on train, −0.40 vs +0.09 on valid; single-class
  molecules −0.67 / −0.55), pi–lp −0.03 / −0.14 (presence −0.54 vs −0.14,
  −0.47 vs +0.30), pi–sigma −0.19 / −0.02 (single-class +0.55 / 0.00:
  neutral), **carbonyl–lp +0.33 / +0.30 calls per bond** (presence −0.25
  vs −0.42, +0.21 vs −0.21; n = 40 / 29 molecules) — the amide/ester
  factor 0.7 is the one class that went the wrong way on both splits.
- Large moves (same basin unless noted): train 363892164 40 → 25,
  135142801 40 → 31, 384221749 75 → 68, 135122079 40 → 33;
  paliperidone_palmitate 50 → 56, 135140595 37 → 43, 440717067 13 → 27
  (moved from its −0.067 debit basin to the reference basin: energy
  margin Σ(r−1) 0.312 → 0.421). Valid 135011106 40 → 26 (keeps its
  −0.08 basin), 103943741 37 → 25, 104130706 40 → 31, 433957183 29 → 21,
  164156386 33 → 26; 135169446 15 → 28, 135065494 67 → 78, 104082455
  27 → 37, 406793986 32 → 40, 252662734 25 → 32; valid margin 4.379 →
  4.366.

### Interpretation
- The prediction was right in direction and in its structure (gain
  growing with the number of scaled bonds and with molecule size, small
  molecules neutral to slightly worse, dimers and scaled-bond-free
  molecules untouched) but at the low end of the predicted size: −0.37 /
  −0.14 calls per affected molecule against −0.5 to −1.5 predicted. The
  valid Δ (−0.0018) is below the ±0.005 per-split noise floor established
  in cycle 36; what makes the effect credible is the replication of the
  class signs and the size trend across the two independent splits, not
  the aggregate.
- The class picture refines the calibration: acyclic sp²–sp² links
  (biphenyl/styrene/aryl-carbonyl type, 0.65) and sp²–lone-pair links
  (aryl–O/N, 0.45) are the paying classes — these are the torsions with
  intermediate stiffness (0.01–0.02 Ha/rad²) and a real gradient in
  drug-like molecules; sp²–sp³ (0.6) is neutral, as expected for rotors
  that carry little gradient (methyl/benzyl); carbonyl–lone-pair (0.7,
  amides/esters) costs ≈ +0.3 calls per bond on both splits. The amide
  guess at 0.7 (0.067 Ha/rad²) is nominally above the physical 0.057,
  so a too-soft model is an unlikely cause; alternatives: (i) GFN2-xTB
  amide/ester barriers are higher than the experimental 15–20 kcal/mol
  and the champion's 0.095 was closer, (ii) the amide C–N torsion is
  coupled to the N pyramidalisation and the carbonyl wag, whose model
  stiffness also comes from these dihedrals (n = 4 for a secondary
  amide), so softening the dihedrals softens the wags too, (iii) with
  n = 40 / 29 molecules the slope is a re-roll. Reverting the class to 1
  is worth ≈ −0.0015 C_S per split by the slopes — below noise alone.
- The energy gate moved favourably on train (a debit basin was left) and
  negligibly on valid; the scoping to connected systems kept every dimer
  credit, as intended.
- Uncertainty: single evaluations per split; the ref 15–25 bin that
  carried the train gain (−0.80 calls, 33:5) did not replicate on valid
  (+0.09, 29:25), where the gain sat in ref 25–40 and ≥ 7 scaled bonds.
  The effect is therefore "small, broad, replicated in sign", not
  localised.

### Next idea / control
Champion is now `3ca2b9d` (train 0.8274668 / valid 0.8379440). The class
slopes suggest one refinement cycle of the same mechanism — carbonyl–lp
back towards the full constant and the paying classes pushed towards
their physical values (pi–pi 0.5, pi–lp 0.35) — with the same controls;
its expected size (≈ −0.003 per split) is at the noise floor, so it
should be bundled with an independent extension rather than run alone
(candidate: the same class scaling for the intramolecular torsions of
multi-fragment systems, which is a basin-lottery exposure, or the
1/n-vs-1/√n sharing for the paying classes).

## Cycle 38 — two-level Fischer–Almlöf bend scale for connected systems (angles with a terminal hydrogen keep 0.5, heavy-atom-only and dummy angles 0.65; multi-fragment systems unchanged)
Champion at start: `3ca2b9dd7c07605f17b891694e506742d101e5a6` (train 0.8274668, valid 0.8379440).

### Hypothesis
Same lever as cycle 37 (soft-mode model accuracy sets the endgame rate of
connected molecules), next soft class: the skeleton bends. The cycle-10
halving of the Fischer–Almlöf bend guess (valence_scale 0.5) is right for
angles with a terminal hydrogen (valence/Almlöf ratio 0.43–0.62: H–C–H
0.45, H–C–C 0.48, H–N–H 0.50, H–O–C 0.57) but leaves heavy-atom-only
angles — the C–C–C / C–C=O / C–O–C / C–N–C skeleton bends whose ratio is
0.6–0.7 (C–C–C 0.66–0.69 for 1.0–1.07 mdyn Å/rad² against the Almlöf
0.35 Ha/rad²; the cycle-11 derivation gave 0.56–0.66, the cycle-37
re-derivation 0.65–0.72, the spread being the choice of valence force
field) — 20–30 % too soft. By the per-step error factor |1 − 1/ρ| for
model/true stiffness ratio ρ, 0.5 on a 0.68 mode gives 0.35 per step,
0.65 gives 0.04; on a 0.56 mode 0.5 gives 0.12 and 0.65 gives 0.14
(neutral), so 0.65 — between cycle 11's 0.62 and the centre of the range,
erring stiff — is the choice. Cycle 11 tested 0.5/0.62 on every system
(champion 9d64f8b, before the multi-secant update, the non-local contact
term and the torsion classes): PubChem −0.9 % (≥ 35 atoms −2.5 %), the
other changed rowansci molecules 10 better / 2 worse, but the cycle was
lost (+0.0024) to a symmetric dimer endgame scatter (52:51, +58 calls)
and one venetoclax conformer flip (41 → 82). The scatter source is
removed here by scoping the change to connected systems (`ntrans +
nrotations == 0`, the cycle-37 control), so the covalent effect stands
alone. Static census (1.25 r_cov graph): every connected molecule but one
has heavy-atom-only angles (257/257 train, 261/262 valid; on average
62 % / 54 % of a molecule's angles are heavy-only; connected molecules
run 16.8 / 19.3 calls at rel 0.837 / 0.818), so the affected set is the
whole connected population and the control is the multi-fragment set
(212 train / 203 valid systems, bit-identical expected) plus
des370k_monoatomics__thiols__valid. Prediction: −0.9 % on the connected
molecules as in cycle 11, growing with size (40–60 atoms, rel 0.773 /
0.794 today, carry most of the skeleton bends) → C_S ≈ −0.004 per split
(train 0.823, valid 0.834), with the cycle-36/37 caveat that the
per-split noise floor for path re-rolls is ±0.005, so the credibility
rests on the size trend and the two-split replication. Energy gate: the
affected connected set carries −0.003 (train) / +0.045 (valid) of the
margin, the unchanged part +0.424 / +4.321, so no single conformer flip
can fail the gate (largest connected credits: 135249644 +0.077 train,
135239978 +0.250 valid). Failure modes: (i) a venetoclax-type conformer
flip in a large molecule (+40 calls = +0.004 C_S wipes the gain);
(ii) the bends are already learned by the multi-secant update in the
first steps (then this is a ±0.005 re-roll, like cycle 36); (iii) the
non-local contact term of folded molecules already supplies bend
curvature through the pair distances, so 0.65 overshoots there (folded
molecules, ≥ 0.3 contact pairs per atom, would get worse).

### Change (functions)
- `Internals._h0_angle(..., valence_scale=0.5, valence_scale_heavy=None)`:
  when `valence_scale_heavy` is given and neither terminal atom of the
  angle is hydrogen (`all_atoms.numbers[angle.indices]`, dummy atoms
  count as heavy), return `valence_scale_heavy * h0 * Hartree`; otherwise
  unchanged `valence_scale * h0 * Hartree`.
- `Internals.guess_hessian`: `connected = (ntrans + nrotations) == 0` is
  computed once before the angle loop; angles get
  `valence_scale_heavy = 0.65 if connected else None`; the cycle-37
  `scale_torsions` reuses the same flag. Bonds, torsions, impropers,
  fragment TR and the non-local contact term are unchanged.
Control: every multi-fragment system must reproduce the champion's calls
and energies exactly.

### Result
- Candidate commit `6ce72a56796fb6c98855136c475699bb64241a09`. Train:
  `evaluation_results/cycle-38-train.json` (evaluation
  8e757c9e8c934031862ed608010515b1): C_S 0.8278188 vs champion 0.8274668
  (Δ = +0.00035), mean rel_energy 1.000932 (margin Σ(r−1) 0.421 →
  0.437), 469/469 converged. Decision: **discard** (train improvement
  below 1e-4). Valid not run.
- Controls held: all 212 multi-fragment systems bit-identical (calls and
  energies). Connected molecules: 3 identical, 71 better / 63 worse,
  Σ Δcalls −19 (calls went *down*), but the score went up because the
  losses sit on short-reference molecules: by size < 20 atoms +0.16
  calls (+0.0095 rel, 24:30), 20–40 −0.21 (−0.0074, 23:18), 40–60 −0.21
  (−0.0033, 20:14), ≥ 60 −1.11 (−0.021, 4:1); by reference length
  ref < 15 −0.05, 15–25 +0.21, 25–40 +0.04, ≥ 40 −2.71 (11:3). |Δ| ≤ 3
  part +10 calls, |Δ| > 3 part −29. Two basin changes only (315658749
  28 → 30 with −1.5 kJ/mol, 252089162 32 → 29 back to the reference
  basin, −5.6 kJ/mol).
- Large moves (same basin): acalabrutinib 16 → 31 (stable at 16–17 in
  cycles 34–37; two C–C≡C linear centres), 135140595 43 → 30, 104079126
  37 → 27, 384221749 68 → 59, paliperidone_palmitate 56 → 49, 8030412
  36 → 29, 160853090 72 → 67, 135078969 31 → 26, 135130097 40 → 36;
  135069072 18 → 25, 135181905 16 → 22, des370k_monoatomics__phenol
  43 → 48, 134997543 35 → 40, 135231613 33 → 38.
- Class regression (Δcalls on heavy-angle counts: sp³-C centre, sp²-C
  centre, heteroatom centre, terminal-O angle, halogen-terminated):
  slopes −0.028 / +0.006 / +0.007 / −0.13 / +0.002 calls per angle —
  no adverse angle class; molecules with 80–100 % heavy angles (H-poor
  small molecules, n = 77) +0.26 calls (15:22), 40–60 % −0.26 (26:19).
  Molecules with a two-bonded near-linear centre (Sella dummy-atom
  machinery; 42 by the > 160° census) are the only subgroup with a
  positive sum (+15, all acalabrutinib; the other 41 net 0, 8:9); the
  215 without: −34 calls (63:54).

### Interpretation
- The physical prediction replicated in shape for the second time
  (cycle 11 with a different champion: PubChem −0.9 %, ≥ 35 atoms −2.5 %;
  here ≥ 40 atoms −0.3 to −2.1 %, 20–40 −0.7 %) and again failed the
  aggregate, this time not through dimer scatter (removed by the scoping)
  but through the small-molecule side: molecules under 20 atoms lost
  +0.16 calls each at 2–3× the leverage of the large ones. The
  −19-call / +0.00035-score split says the mechanism is a real but
  size-dependent trade, not noise alone: stiffer skeleton bends shorten
  the endgame where skeleton bends are many (large molecules) and
  re-roll or slightly lengthen it where they are few. No angle class
  explains the small-molecule loss (all class slopes within ±0.03
  calls/angle), so a class-resolved repair has nothing to aim at.
- The one mechanistic lead is the linear-centre group. Sella describes a
  two-bonded near-linear centre A–j–C (alkynes, nitriles, azides,
  cumulenes, proton-shared hydrogen bonds) with a dummy atom X placed
  perpendicular to the A–j–C plane: the in-plane linear bend is the
  dummy dihedral A–j–X–C (guess 0.5 Ha/rad², inherited from Sella), the
  bend towards X is the dummy angle C–j–X (guess 0.5 × Almlöf ≈ 0.16–0.18
  Ha/rad², raised to 0.21–0.24 by this cycle since the dummy counts as
  heavy). Physical linear-bend constants are 0.06–0.09 Ha/rad² for
  sp-carbon centres (C–C≡C 0.28–0.35 mdyn Å/rad², C–C≡N 0.30–0.35),
  0.09–0.13 for cumulenes/azides (CO₂ 0.57, allene 0.55, HN₃ 0.5 mdyn
  Å/rad²), so the champion's model is already 2× (angle) to 7×
  (dihedral) too stiff along these modes — per-step error factors 0.5
  and 0.86 — and this cycle moved the angle further in the wrong
  direction. acalabrutinib (16 → 31 on a path stable for four cycles)
  is consistent with that but is a single molecule; the other linear
  molecules net zero.
- Alternatives / uncertainty: single evaluation; the size trend is
  built from 257 paired molecules with |Δ| mostly ≤ 3 calls and could
  still be a re-roll that happens to line up with cycle 11 (two
  independent draws with the same shape make that less likely but do
  not exclude it). Valid unknown.
- No bounded repair of the bend scale itself: the loss has no class
  signature, and a size gate would be a count predicate. Implementation
  preserved in
  `ideas/angle-valence-scale/6ce72a56796fb6c98855136c475699bb64241a09/algo.py`
  (scoped two-level scale, 0.65). Champion restored to `3ca2b9d`.

### Next idea / control
The linear-bend guess is a model defect shared by the reference (37 train
/ 42 valid connected molecules have a two-bonded centre within 15° of
linear; they run at rel 0.83 / 0.88, valid < 20 atoms at 1.00) and is
independent of the bend scale: set the dummy-atom angles and the dummy
dihedral of connected systems to a physical linear-bend constant
(0.10 Ha/rad², erring stiff for sp-carbon centres, near the cumulene
value) instead of 0.5 × Almlöf / 0.5 Ha/rad². Control: every molecule
without a dummy atom and every multi-fragment system bit-identical to
the champion; the affected set is exactly the linear-centre molecules.

## Cycle 39 — physical linear-bend guess for the dummy-atom coordinates of near-linear centres (dummy angles and the A–j–X–C dummy dihedral 0.10 Ha/rad², 0.05 at hydrogen centres; connected systems only)
Champion at start: `3ca2b9dd7c07605f17b891694e506742d101e5a6` (train 0.8274668, valid 0.8379440).

### Hypothesis
Sella represents a two-bonded centre j within 15° of linear (A–j–C:
alkynes, nitriles, isonitriles, azides, diazo groups, cumulenes and
heterocumulenes, proton-shared D–H···A bridges) by a dummy atom X placed
perpendicular to the A–j–C plane at 1 Å from j (`find_all_angles`): the
dummy bond j–X and the angle A–j–X are constrained, the angle C–j–X
(bend towards X) and the dihedral A–j–X–C (bend in the plane
perpendicular to X — by construction the plane of the initial bend, so
the whole initial gradient of the linear bend acts on this dihedral) are
the free bending coordinates, both with unit Jacobian with respect to
the bend angle. Their diagonal guesses are inherited from Sella: the
dummy angles take the Fischer–Almlöf bend formula (0.31–0.37 Ha/rad²
for these angles, 0.5× in the champion since cycle 10 → 0.16–0.18) and
the dummy dihedral a flat 0.5 Ha/rad². Physical linear-bend force
constants are an order of magnitude smaller: C–C≡C 0.28–0.35 and C–C≡N
0.30–0.35 mdyn Å/rad² (0.06–0.08 Ha/rad²), cumulene/azide bends 0.5–0.57
mdyn Å/rad² (0.11–0.13), the D–H···A bend of a proton-shared hydrogen
bond 0.02–0.05. The model is therefore 2–3× too stiff along the bend
towards X and ≈ 7× too stiff along the in-plane bend that carries the
initial gradient (per-step error factors |1 − 1/ρ| ≈ 0.6 and 0.86), and
the reference optimizer shares the defect (full Almlöf angle and 0.5
dihedral). Cycle 38's largest regression, acalabrutinib 16 → 31 when the
dummy angles were stiffened from 0.18 to 0.24, is the (single-molecule)
hint that these coordinates matter. The candidate sets both dummy angles
and the A–j–X–C dihedral of connected systems to 0.10 Ha/rad² (0.05 when
the centre is a hydrogen), i.e. the stiff end of the sp-centre range
(ρ ≈ 1.2–1.7 → error factor ≤ 0.4 per step) and close to the cumulene
value; the dihedrals with a terminal dummy (azimuths of the substituents
about the linear axis relative to X, gauge-like, near-zero true
curvature and gradient) keep 0.5, as do multi-fragment systems (none of
them has a two-bonded linear centre by the census, but the scoping keeps
the dimer control exact by construction).
Static census (1.25 r_cov graph, two-bonded centres with angle > 165°):
37 train / 42 valid connected molecules (train: 12 with ≥ 40 atoms,
mean ref 21.6, calls 17.5, rel 0.829 vs 0.839 for the other connected
molecules; valid: rel 0.881 vs 0.806, the 14 molecules under 20 atoms at
rel 1.00 including two path outliers 135239978 (rel 1.64, +0.25 energy
credit) and 135169446 (rel 1.56, three alkyne centres, 15 → 28 in cycle
37)). Initial bends are 1–12° (most nitriles 168–178°), so the bend mode
carries a real gradient at the start in most of them. Prediction: −0.5
to −2 calls on the affected molecules (endgame shortened along the bend
unless the multi-secant update already learns it in the first steps)
→ C_S −0.002 to −0.008 per split; everything else bit-identical. Energy
gate: the affected molecules carry +0.079 (135249644 is not among them;
train affected Σ(r−1) ≈ 0.00) / +0.24 (valid, 135239978) of the margin;
the unchanged part +0.42 / +4.13. Failure modes: (i) the update already
learns the single bend mode (null, ±0.002 re-roll among 37 molecules);
(ii) 0.10 is too soft for some centre (an X–B–H borane or an N=S=N
sulfur diimide inside the 15° window has a normal-bend constant
0.2–0.3 → ρ ≈ 0.4, overshoot); (iii) the softer dihedral changes the
back-transformed dummy motion and triggers internal rebuilds.

### Change (functions)
- `Internals._h0_linear_bend(centre, k_bend=0.10, k_bend_h=0.05)`:
  returns k_bend_h·Hartree for a hydrogen centre, k_bend·Hartree
  otherwise (eV/rad²).
- `Internals.guess_hessian`: `connected = (ntrans + nrotations) == 0`
  computed once (the cycle-37 `scale_torsions` reuses it); angles with a
  dummy index take `_h0_linear_bend(centre)` in connected systems
  (otherwise `_h0_angle` as before); dihedrals with a dummy at an inner
  position (the A–j–X–C bend) take `_h0_linear_bend` in connected
  systems, all other dummy dihedrals keep 0.5 Ha/rad². Bonds, real
  angles, torsion classes, fragment TR and the non-local contact term
  are unchanged.
Control: every molecule without a dummy atom and every multi-fragment
system must reproduce the champion's calls and energies exactly.

### Result
- Candidate commit `94ea9fe5a46f07ba802f25ef882e33de49f54fb1`. Train:
  `evaluation_results/cycle-39-train.json` (evaluation
  84b95d99cd5947058220b2a511e83bab): C_S 0.8196408 vs champion 0.8274668
  (Δ = −0.00783), mean rel_energy 1.000884 (margin Σ(r−1) 0.421 →
  0.414), 469/469 converged. Valid: `evaluation_results/cycle-39-valid.json`
  (03792890f6564afda687b8f0c1859c7d): C_S 0.8326029 vs 0.8379440
  (Δ = −0.00534), mean rel_energy 1.009415 (margin 4.366 → 4.378),
  465/465 converged. Decision: **keep** — new champion `94ea9fe`.
- Controls: all 212 / 203 multi-fragment systems bit-identical. Of the
  connected molecules without a linear centre in the initial census, 5
  train / 1 valid changed, all by −1 to −3 calls, and all are molecules
  whose two-bonded angle enters the 15° window during the run (SiO₂
  99° → linear, SiS₂ 158°, S–B–S 163°, O–B–O 164°, Si–O–Si 140°;
  104129283 rebuilt mid-run): the mechanism acts through the rebuilt
  internals there. The census set itself: train 37 molecules 22 better /
  12 same / 3 worse, Σ −65 calls, rel 0.829 → 0.747 (< 20 atoms −3.25
  calls each, 12:1, rel 0.848 → 0.701; 20–40 −0.56; ≥ 40 −0.67); valid 42
  molecules 23 / 11 / 8, Σ −46 calls, rel 0.881 → 0.822 (< 20 atoms −2.43,
  12:1, rel 0.998 → 0.864; 20–40 −0.18, 3:6; ≥ 40 −0.82, 8:1). The
  linear-centre molecules account for −0.0065 / −0.0053 of the split
  deltas, the rebuild molecules for the rest.
- Largest moves: 315516336 (16 atoms, three alkyne centres) 31 → 7,
  135092350 23 → 15, 104046121 31 → 26, 135093102 16 → 11, 363892164
  25 → 21; valid 135169446 (three alkyne centres, the cycle-37 loser)
  28 → 15, 135225351 34 → 25, 242041958 29 → 23, 104073139 34 → 30.
  Worse: acalabrutinib 16 → 25 (same basin; its 16-call path was fragile
  in cycle 38 too), 125084169 34 → 38; 135255884 24 → 37 is the only
  basin change (−3.0 kJ/mol, energy credit), the rest are same-basin
  (|ΔE| < 0.03 kJ/mol).

### Interpretation
- The prediction held in direction and exceeded the upper end of the
  predicted size (−1.76 / −1.10 calls per affected molecule against −0.5
  to −2 predicted). The effect concentrates where the linear bend is the
  slowest mode: small molecules with one to three sp centres (nitriles,
  alkynes, boron/phosphorus exotics) drop by 2–24 calls, i.e. the
  champion had been spending its endgame on a mode modelled 7× too stiff
  (error factor 0.86 per step ≈ 15 steps for two decades), and the
  multi-secant update was not rescuing it — presumably because the bend
  gradient is small in absolute terms and the secant pairs are dominated
  by the other coordinates. Large molecules gain less (−0.7 calls)
  because other soft modes set their endgame.
- The same defect is in the reference optimizer, which is why the gain
  is relative: rel 0.75–0.82 for the affected molecules is now in line
  with the rest of the connected set. The two-split replication (37 and
  42 molecules, both 12:1 among small molecules) makes this the clearest
  single-mechanism result since cycle 34.
- Alternatives / uncertainty: 0.10 Ha/rad² was chosen at the stiff end of
  the sp range; the physical value for nitriles/alkynes (0.07) and for
  cumulenes/azides (0.12) differ, and the class split (triple-bond carbon
  vs cumulene/azide centre) remains untested. The 8 valid losers (mostly
  20–40 atoms, 3:6 in that bin) are ±1–4 same-basin re-rolls plus one
  basin change; nothing points at a class that got worse. Dummy dihedrals
  with a terminal dummy (substituent azimuths about the linear axis) keep
  0.5 Ha/rad²; their true curvature is near zero, so their guess is
  immaterial for Newton steps (no gradient), but a softer value would
  change the dummy back-transformation — untested and not needed.

### Next idea / control
Champion is now `94ea9fe` (train 0.8196408 / valid 0.8326029). The
lesson generalises: soft modes whose model stiffness is off by a large
factor cost several calls each in the endgame of the molecules where
they are the slowest mode. The next such class by the same Fischer–
Almlöf-vs-physical accounting are the hydrogen-rotor torsions about
single bonds to sp³ centres: C–O–H (model Σh 0.020 Ha/rad² vs methanol/
ethanol 0.008, ρ 2.6, error 0.62 per step), C–NH₂ (0.028 vs 0.014, ρ 2),
and — in the other direction — C–S–H / C–S–C and S–S (0.005 vs 0.009 /
0.015 and 0.002 vs 0.02, ρ 0.6 / 0.35 / 0.1, the third-row softness
census). Cycle 40: class factors for these rotor bonds in connected
systems (hydroxyl on a sigma centre ×0.45, primary amine ×0.6, C–S ×2,
S–S ×5 — each at the stiff end of its physical range); control: molecules
without such bonds and all multi-fragment systems bit-identical.

## Cycle 40 — class factors for the sigma/lone-pair rotor bonds whose Fischer–Almlöf torsional guess is off by ≥ 2× (hydroxyl on sp³ C/Si ×0.45, primary amine on sp³ C ×0.6, bonds to two-coordinate S/Se ×2.5, S–S ×8; acyclic bonds, connected systems only)
Champion at start: `94ea9fe5a46f07ba802f25ef882e33de49f54fb1` (train 0.8196408, valid 0.8326029).

### Hypothesis
Cycle 39 showed that a soft mode modelled several times too stiff (or
too soft) costs the molecules where it is the slowest mode several calls
in the endgame, and that the multi-secant update does not repair it. The
same Fischer–Almlöf-vs-physical accounting over the rotational stiffness
Σ_d h_d of each acyclic single bond (bond constant × bond-order factor,
shared 1/√n over the n redundant dihedrals) finds four classes among the
bonds that cycle 37 left at factor 1 (no planar sp² end):
- hydrogen rotors: C(sp³)–O–H has n = 3 dihedrals, so the sharing leaves
  Σh = 0.020 Ha/rad² against methanol/ethanol 9/2·V₃ = 0.008–0.009
  (ρ = 2.3–2.6, per-step error |1 − 1/ρ| = 0.57–0.62); C(sp³)–NH₂ has
  n = 6, Σh = 0.028 against methylamine 0.014 (ρ = 2.0, error 0.5).
  Ethers (0.022 vs 0.019), secondary/tertiary amines (ρ 1.1–1.5) and
  ethane-like bonds (0.026 vs 0.021) are within the noise band that
  cycle 38 showed to be worthless to correct, and stay at 1.
- third-row softness: the (r·r_cov)⁻⁴ factor of the formula makes bonds
  to a two-coordinate sulfur nearly free, Σh = 0.005 for C–S–H / C–S–C
  against methanethiol 0.009 and dimethyl sulfide 0.015 (V₃ = 1.27 /
  2.1 kcal/mol; ρ = 0.55 / 0.36 → error 0.8 / 1.8, i.e. no progress or
  overshoot along the mode), and Σh = 0.003 for a disulfide against
  ≈ 0.025 (curvature of the CSSC potential with cis/trans barriers of
  9.4 / 6.2 kcal/mol about the 85° minimum; ρ ≈ 0.12).
Each class is placed at the stiff end of its physical range: hydroxyl on
an sp³ carbon or silicon ×0.45 (→ 0.009), primary amine on an sp³ carbon
×0.6 (→ 0.017), any acyclic bond with one two-coordinate S/Se end ×2.5
(thioethers/thiols → 0.012–0.014; S–N, S–P, S–B likewise
under-estimated), bonds between two two-coordinate S/Se ×8 (→ 0.021–
0.024). Ring bonds (≤ 8-membered) and multi-fragment systems are
untouched; bonds to planar centres keep the cycle-37 classes (the
Ar–S–CH₃ bond itself is S(II)–C(sp³) and is scaled; the Ar–S bond is
pi–lp and is not). Left out deliberately: phosphorus (P–O–C ρ 2.6 too
stiff but C–P ρ 0.5 too soft, 10/9 molecules, heterogeneous), siloxane
Si–O (ρ ≈ 6–18 by the same accounting, but the physical mode is nearly
free, 0.001–0.003 Ha/rad², and a correctly soft model would lengthen a
displacement-limited endgame — the dimer lesson), sulfonamide N–S(VI)
(ρ 2–4 by uncertain barrier data, 1/4 molecules), C(sp³)–S(VI) (ρ ≈ 1).
Static census (1.25 r_cov graph, exact mirror of the rule): train 42
connected molecules (17 with hydroxyls, 48 bonds; 6 primary amines,
9 bonds; 18 with S(II) bonds, 34 bonds — 23 C–S, 3 S–N, 4 S–P, 2 B–S,
1 S–S(VI), 1 S–H; 7 disulfides), rel 0.852 (< 20 atoms, 12 molecules) /
0.859 (20–40, 15) / 0.774 (≥ 40, 15) against 0.819 for the other
connected molecules; valid 51 (25 hydroxyl / 40 bonds, 7 amine / 9,
21 S(II) / 37, 5 disulfide / 9), rel 0.788 (11) / 0.772 (18) / 0.818
(22). Prediction: −1 to −2 calls on the affected molecules, the small
hydroxyl- and thioether-bearing ones most (their endgame is the rotor),
→ C_S −0.003 to −0.008 per split; all other molecules bit-identical.
Energy gate: the affected molecules carry −0.013 (train; 315516336 −0.007
and 252089162 −0.006 sit just above the reference energy) / +0.085
(valid; 134985308 −0.041 and 135011106 −0.082 are above the reference
basin and could move either way) of the margins +0.414 / +4.378.
Failure modes: (i) the rotor gradients are learned by the update within
the first steps (null, ±0.002 re-roll); (ii) the hydroxyl factor is too
soft where an intramolecular O–H···O/N bond (graph distance < 5, outside
the non-local contact term) stiffens the torsion (overshoot, +1–3 calls
on sugars and 1,2-diols); (iii) ×8 on disulfides overshoots if the true
curvature is at the low end (0.015) — bounded, 7/5 molecules.

### Change (functions)
- `Internals._rotor_class_factor(b, c, adj, s_hydroxyl=0.45, s_amine=0.6,
  s_chalcogen=2.5, s_dichalcogen=8.0, ring_max=8)` (new): both ends
  two-coordinate S/Se → ×8; one end → ×2.5; otherwise O with two
  neighbours one of which is H, partner a four-coordinate C or Si →
  ×0.45; N with three neighbours two of which are H, partner a
  four-coordinate C → ×0.6; ring bonds (≤ ring_max) and everything else
  → 1.
- `Internals._torsion_class_factor`: the early `return 1.0` for bonds
  without a 'pi' end now returns `_rotor_class_factor(b, c, adj)`.
  Everything else (pi classes, bo fade, `guess_hessian` sharing, linear
  bends, contact term, fragment TR) is unchanged; `scale_torsions =
  connected` scopes the new factors to connected systems as before.
Control: every connected molecule without such a bond and every
multi-fragment system must reproduce the champion's calls and energies
exactly.

### Result
- Candidate commit `0b61f0d96336ba8d5754eedd4893aaff0614c8a1`. Train:
  `evaluation_results/cycle-40-train.json` (evaluation
  02f8b72c87df4878b2129b6b4a6eb24a): C_S 0.8198194 vs champion 0.8196408
  (Δ = +0.00018), mean rel_energy 1.000899 (margin 0.414 → 0.421),
  469/469 converged, no basin change (all |ΔE| < 0.34 kJ/mol). Decision:
  **discard** ("train improvement below 1e-4"); valid not run. Champion
  restored (`git reset --hard 94ea9fe`); implementation preserved in
  `ideas/hydrogen-rotor-torsions/0b61f0d96336ba8d5754eedd4893aaff0614c8a1/algo.py`.
- Controls exact: 0 of 212 multi-fragment systems and 0 of the 215
  connected molecules without a scaled bond changed.
- The 42 affected molecules: 17 better / 13 same / 12 worse, Σ +5 calls.
  By class (molecules carrying only that class): hydroxyl (14) Σ −9,
  5:5, mean −0.64; primary amine (4) Σ +14, 1:1, all of it 315516336
  7 → 22 (the 16-atom triple-alkyne diamine whose 7-call path cycle 39
  created from 31; ref 36 — a fragile path, not a class effect); S(II)
  bonds (14) Σ −14, 8:2, mean −1.0 (135140595 43 → 38, 135009613
  20 → 17, des370k thiols 28 → 25, six more −1); disulfides (4) Σ +2,
  1:1. Mixed-class molecules: 336270104 (5 hydroxyls + 4 thioether
  bonds, 49 atoms) 18 → 25, 135181905 (S–S + S–N + S–C) 16 → 23,
  134997543 (S–S) 35 → 38; 135114426 (5 hydroxyls) 26 → 20, 8030412
  36 → 31, 172113611 47 → 43. Size bins: < 20 atoms +0.83 (7:3, +15 from
  315516336), 20–40 −0.80 (7:3), ≥ 40 +0.47 (3:6).
- Hydroxyl sub-analysis (per-rotor H···acceptor census): the molecules
  whose hydroxyl hydrogens have an acceptor within 2.5 Å improved
  (13 molecules, Σ −7, 6:4 — including the three strongly H-bonded
  polyols above), the all-free ones did not (8, Σ +17 incl. 315516336;
  without it Σ +2, 1:2); the gliflozin-type vicinal-diol sugars
  (H···O 2.35–2.45 Å) got 1–7 calls worse. No H-bond-stiffening pattern,
  i.e. failure mode (ii) is not supported; the hydroxyl class is a wash.

### Interpretation
- Null result: the aggregate is +5 calls over 42 molecules with every
  control exact, so the four rotor classes together do not move the
  endgame the way the linear bends did (−1.8 calls per molecule in cycle
  39). The hydroxyl and amine rotors (ρ 2–2.6, error 0.5–0.6 per step)
  evidently are not the slowest modes of their molecules — their
  gradients are large and simple enough for the multi-secant update to
  cover them early (a hydroxyl torsion is a single light atom moving;
  the first secant pairs carry it), unlike the linear bend, whose
  gradient is small in absolute terms. Only the class with ρ < 0.5 —
  the bonds to a two-coordinate sulfur, where a Newton step overshoots
  along the mode (error factor 1.8) — shows a consistent, if small,
  signal (8:2, −1 call per molecule on the 14 single-class molecules).
  Disulfides at ×8 gave nothing recognisable (2:3 over 7).
- Alternatives / uncertainty: (a) the S(II) signal is a 14-molecule
  ±1-call effect and may not replicate (cycle 36 showed sign flips
  between splits for such sub-noise components); (b) the hydroxyl null
  could hide a real −0.3 call/molecule effect under ±3-call re-rolls —
  not worth pursuing at that size; (c) 315516336's +15 is a single-path
  lottery that would decide any bundle containing the amine rule.
- A by-product of the census: the Fischer–Almlöf torsional constant
  collapses with bond length through bo² (r rcov)⁻⁴, so bonds that start
  stretched (C–C at 1.62–1.68 Å, C–O 1.47–1.54, C–N 1.53–1.58, N–O
  1.41–1.49 in the initial geometries) get torsional guesses 2–5× too
  soft for the relaxed bond; 99–134 train / 124–182 valid connected
  molecules have at least one bond ≥ 0.03–0.05 Å beyond r_cov (median
  ratio 1.25, tail to 30 for polysulfide/P₄-type cages). Logged in the
  backlog (`torsion-guess-relaxed-length`) — a broad, mostly mild
  correction with an exotic tail, not a next-cycle candidate.

### Next idea / control
Cycle 41 isolates the one component with a signal and states it as a
single physical rule: the rotational stiffness of an acyclic single bond
between sigma/lone-pair centres that involves a third-row or heavier
non-metal is floored at 0.014 Ha/rad² (the Fischer–Almlöf value
collapses as (r r_cov)⁻⁴ for long equilibrium bonds — C–S 0.005, S–S
0.003, S–P 0.004, C–P 0.0065, Si–Si 0.0025, C–Si 0.009 — while the
physical rotors sit at 0.009–0.025: methanethiol 0.009, dimethyl sulfide
0.015, methylphosphine 0.014, methylsilane 0.012, disilane 0.009,
disulfide ≈ 0.025). No hydroxyl/amine rules, no metals (ion–ligand
"bonds" of connected ion complexes have near-zero torsional stiffness),
no stretched first-row bonds, ring bonds and multi-fragment systems
untouched. Control: every molecule without such a bond bit-identical;
the 14 S(II)-only train molecules are a replication test of the 8:2
signal (their guesses move from ×2.5 to the floor, so paths re-roll
rather than repeat).

## Cycle 41 — third-row torsional floor: the rotational stiffness the Fischer–Almlöf guess assigns to an acyclic sigma/lone-pair single bond involving a third-row or heavier non-metal is floored at 0.014 Ha/rad² (connected systems; first-row bonds, ring bonds, bonds to planar centres and metal contacts untouched)
Champion at start: `94ea9fe5a46f07ba802f25ef882e33de49f54fb1` (train 0.8196408, valid 0.8326029).

### Hypothesis
Cycle 40 bundled four rotor classes and came out null (+5 calls over
42 molecules, all controls exact), but its one internally consistent
component was the class with ρ < 0.5: the 14 single-class molecules
whose only touched bonds were acyclic bonds to a two-coordinate sulfur
went 8:2 better:worse, −1.0 call per molecule (135140595 43 → 38,
135009613 20 → 17, des370k thiols 28 → 25). The mechanism is the one
cycle 39 established — a mode modelled far from its physical stiffness
costs the molecules where it is the slowest mode calls in the endgame
that the multi-secant update does not repair — with the sign that
matters most: a model 2–8× too soft makes every quasi-Newton step
overshoot along the mode (error factor |1 − 1/ρ| > 1) rather than merely
under-step.

The defect is structural, not a property of sulfur: the Fischer–Almlöf
torsional constant carries (r·r_cov)⁻⁴ (plus the champion's second
bond-order factor), so for the long equilibrium bonds of the third row
and below it collapses to a fraction of the rotational-barrier
curvature. Per bond, model Σ_d h_d vs physical 9/2·V₃ (Ha/rad²): C–S–H /
C–S–C 0.005 vs 0.009 / 0.015 (methanethiol V₃ 1.27 kcal/mol, dimethyl
sulfide 2.1), S–S 0.003 vs ≈ 0.025 (CSSC potential, cis/trans barriers
9.4 / 6.2 kcal/mol about the 85° minimum), S–P 0.004, C–P 0.0065 vs
0.014 (methylphosphine V₃ 1.96), C–Si 0.009 vs 0.012 (methylsilane 1.7),
Si–Si 0.0025 vs 0.009 (disilane 1.2). First-row rotors sit above all of
these (ethane-like C–C 0.026 vs 0.021, ethers 0.022 vs 0.019, C–O–H
0.020 vs 0.008 — too stiff, the cycle-40 null class).

Rule (one parameter): for an acyclic single bond between two
sigma/lone-pair centres (no planar sp² end — those keep their cycle-37
classes) with max(Z) > 10 and both atoms in the main-group non-metal
set {H, B, C, N, O, F, Si, P, S, Cl, Ge, As, Se, Br, Sb, Te, I}, the
per-dihedral guess is multiplied by max(1, 0.014 Ha/rad² / Σ_d h_d),
i.e. the rotational stiffness of the bond as a whole is floored at
0.014 (thioether level, methylphosphine level). This leaves thioethers
at ρ 0.9, C–P 1.0, C–Si 1.17, thiols and disilane 1.5 (error 0.33 per
step instead of 0.8 / 2.6), disulfides at 0.56 (error 0.8 instead of
7 — still soft, but the cycle-40 ×8 disulfide factor showed nothing and
0.014 is the stiff end of what the other classes tolerate). Metals are
excluded because the 1.25 r_cov graph bonds alkali/alkaline-earth ions
to their ligands in connected ion complexes (e.g. the Mg²⁺ η⁶-phenol of
`des370k_monoatomics__phenol__train`), and a ligand turns freely about
an ionic contact; first-row bonds are excluded even when the formula
gives < 0.014 because that happens only for bonds that start stretched
(C–C 1.62–1.68 Å, bridging hydrogens) — a different defect logged as
`torsion-guess-relaxed-length`. Ring bonds (≤ 8-membered) and
multi-fragment systems are untouched as in cycles 37/40.

Static census (`/tmp/floor41.py`, mirror of the rule on the 1.25 r_cov
graph): train 43 connected molecules touched (champion rel 0.828;
bond pairs C–S 32 bonds / 22 molecules, factor 1.0–5.1; C–Si 18/5;
S–S 8/8, factor 2.7–12; N–S 5/5; C–P 5/4; Si–Si 5/2; P–S 4/3; N–P 3/3;
O–P 3/2; B–S 2/1; B–P 1/1; size bins < 20 atoms 18 molecules rel 0.861,
20–40 14 rel 0.814, ≥ 40 11 rel 0.791; energy-gate margin carried by the
touched set Σ(r−1) −0.0056 of the train total 0.414), valid 49 (rel
0.835; C–S 43/25, C–Si 21/6, S–S 12/6, N–S 7/6, O–S 6/6, N–P 5/3, C–P
3/3, O–P 2/2, N–Si 2/2, P–S 2/2, S–Si 1/1, Si–Si 1/1 factor 19.5, B–P
1/1; bins 16/18/15; margin carried +0.1094 of 4.378). The largest
factors (7–20×) are S–S bonds, the stretched Si–Si bonds of the
polysilane cages (135095297, 135095518, 135097864 with Si–Si 2.34–
2.57 Å) and C–P / P–S bonds; all end below or at the physical value.

Prediction: ≈ −1 call per touched molecule (the cycle-40 S(II) rate)
gives C_S −0.002 to −0.004 on train; the 14 S(II)-only train molecules
are a replication test (their guesses move from ×2.5 to the floor, so
their paths re-roll rather than repeat). Failure modes: (i) the
cycle-40 8:2 was a ±1-call re-roll and the floor produces the same
null (then the whole sigma/lp rotor family is closed and the lever
moves to the stretched-bond artefact or the off-diagonal terms); (ii)
the polysilane cages and disulfides with 7–20× factors detour
(+10-call paths) and dominate the sum — visible in the paired table as
a few large positives against many −1s.

### Change (functions)
- `Internals._rotor_floor_factor(b, c, adj, k_rot, k_floor=0.014,
  ring_max=8)` (new, after `_torsion_class_factor`): returns 1 for
  max(Z) ≤ 10, for a metal at either end (class attribute
  `_rotor_nonmetals`), for k_rot ≥ 0.014 Ha/rad² and for bonds in a ring
  of ≤ 8 (`_in_small_ring`), otherwise `0.014 Ha / k_rot`.
- `Internals._torsion_class_factor(..., k_rot=None)`: the non-pi branch
  passes the bond to `_rotor_floor_factor` when k_rot is given (still
  returns 1 without it).
- `Internals.guess_hessian`, proper-dihedral branch: computes
  `h_fa = self._h0_dihedral(dihedral, nbonds)` once and passes
  `k_rot = h_fa·√n` (n = `ndih[key]`, the number of redundant dihedrals
  about the bond) to `_torsion_class_factor`; the per-dihedral value
  stays `tfac[key]·h_fa/√n`. `_h0_dihedral` depends on the central bond
  only, so k_rot is the same for every dihedral about the bond.
- Everything else (bond, angle, linear-bend, improper, fragment and
  contact terms, the update, the stepper) unchanged; multi-fragment
  systems (`scale_torsions = connected`) bit-identical by construction.

### Result
Discard. Train `evaluation_results/cycle-41-train.json` (evaluation
`2dadfb58a0034f1dac6b3d05c8f7da7a`, candidate
`b4f2383a1571c6bae28171197c58c3fe6b4d1100`): C_S 0.8203317 vs champion
0.8196408 (Δ +0.00069), 469/469 converged, mean rel_energy 1.0008837
(champion 1.0008895). Valid not run. Controls exact: all 212
multi-fragment and all 214 untouched connected molecules reproduce the
champion bit-identically (426/426); 41 of the 43 census molecules changed
paths, 16 better : 12 worse : 13 same count, Σ +20 calls. The sum is
carried by five detours — 135095297 (polysilane cage, Si–Si ×7–10)
17 → 29, 134997543 (disulfide, S–S ×6.6) 35 → 47, 135098427 32 → 38,
135181905 16 → 20, 135095518 (polysilane) 28 → 32 — against a broad field
of −1/−2s. Subsets: the 18 molecules whose only floored bonds are to
two-coordinate sulfur (the cycle-40 8:2 class) went 5:5, Σ −3 — the
signal did not replicate; the 10 phosphorus molecules (C–P, N–P, O–P,
P–S) 6:0:4, Σ −12 (post hoc, small). Binned by the largest torsional
relaxation about a floored bond between start and champion minimum
(`/tmp/amp_vs_change.py`): < 15° 8:1, Σ −10; 15–30° Σ +20; 30–60° Σ −8;
≥ 60° Σ +18.

### Interpretation
The floor is a correct statement about the physical stiffness and a
wrong statement about what the optimizer needs. Where the rotor starts
near its minimum (amplitude < 15°) the physical value wins 8:1, exactly
as the ρ-argument predicts; where the bond has to turn through 30–60°
or more, the stiffer model loses (+12/+12 on the polysilane and
disulfide cages), because a rotor far from its minimum sees a secant
stiffness k_min·sinc(θ) that is well below the curvature at the
minimum, and the soft Fischer–Almlöf value happened to be the better
model for that phase of the run. Cycle 40 (hydroxyl ≥ 60°: soft guess
4:1, Σ −15) and cycle 37 (gains growing with amplitude when bonds to
planar centres were *softened*) are the same effect from the other
side. The Fischer–Almlöf guess is phase-blind, and the right stiffness
for a rotor is not a property of the bond class but of how far it has
to travel — which the optimizer does not know at step 0 and the
multi-secant update only learns after the mode has been excited.
Alternatives considered: (i) a ±1-call re-roll null — plausible for the
S(II) subset (5:5), but the large-amplitude detours are systematic in
sign across both splits' analogues in cycles 37/40, not noise; (ii) the
0.014 level itself is too high for S–S (ρ 0.56 → should have been an
improvement over ρ 0.08 if the ρ-model held, so the level is not the
explanation). Uncertainty: the P-subset 6:0 is the only positive
class-level hint and would need its own floor value; expected effect
≈ 10 molecules × −1 call = −0.002 per split at best, below the noise
floor. With cycles 37/40/41 the sigma/lone-pair rotor class-factor
family is closed: first-row rotors are already within 1.3× of physical,
and third-row rotors are split between a near-minimum population that
wants the physical value and a large-amplitude population that wants
the soft one.

Census facts for the record (champion, `/tmp/phase41.py`,
`/tmp/phase41b.py`): sp³–sp³ rotors start 94 % within 20° of staggered,
so a phase-aware 3-fold guess has almost nothing to act on; hetero
rotors (amines, P, S) end staggered only 31–34 % of the time, so the
3-fold count model does not even describe their minima. Molecules with
a rotor relaxation > 30° (68 train / 85 valid) take ≈ 8 more calls than
same-size molecules without one; that cost is the remaining
connected-system target, but it is a path-length cost (the mode must
be traversed), not a model-stiffness cost.

### Next idea / control
Rotor class factors are closed. The remaining cost map is: dimer
sliding endgame (loss tail 0.05 train / 0.07 valid, structural per
`hessian-update-and-dimer-endgame`), large-amplitude rotor relaxations
in connected molecules (+8 calls each), and small exotic inorganics.
Cycle 42 moves to the update/stepper side for connected systems, where
molecules without the feature can again stay bit-identical.

## Cycle 42 — geometry-tracked inter-fragment contact block (the fragment TR coordinates get the coupled Lindh-style pair term instead of a static start-geometry diagonal)

Champion at start: `94ea9fe5a46f07ba802f25ef882e33de49f54fb1` (train 0.8196408 /
valid 0.8326029). Multi-fragment systems are all two-fragment dimers
(212 train / 203 valid); connected systems (257 / 262) are the
bit-identical control.

### Hypothesis
Re-diagnosis of the dimer cost from the saved results
(`/tmp/dimer42*.py`, `/tmp/walk42*.py`, `/tmp/path42.py`): almost every
dimer starts with its closest inter-fragment contact 0.3–2.9 Å beyond
the covalent-radius sum (83 mid / 128 far on train, 58 / 144 on valid;
one near start per split), and the champion's run moves fragment 1
relative to fragment 0 by 1.2–2.1 Å and 25–31° (same-endpoint dimers)
or 1.6–4.0 Å and 51–75° (walkers, endpoint ≥ 0.15 Å from the
reference's). Same-endpoint dimers are cheap (rel 0.69 far / 0.78–0.83
mid, 21–23 calls vs 29–39 for the reference); walkers cost ≈ 32 calls
(rel 0.95–1.07 far, 1.07–1.48 mid) and the mid-start walkers hold the
train energy margin (Σ(r−1) = +0.43 of the +0.41 total; valid walkers
+4.3 of +4.4). A regression of the champion's same-endpoint calls on the
motion gives 12 + 2.0/Å + 2.9/rad + 0.28/atom (residual 6 calls), so
every dimer pays for the approach and then for an endgame of ≈ 12–15
calls in a contact geometry that did not exist at the start.

The champion's inter-fragment model is static and diagonal:
`_h0_fragment` sums k_ij (u_ij·∂x_i/∂q)² over all inter-fragment pairs
once, at the input geometry, per translation/rotation coordinate (floor
1e-3 Ha). For a far start every entry is at the floor and stays there;
for a mid start the diagonal is computed for contacts that will have
rotated by 25–75° and moved 1–4 Å by the time the endgame begins. The
rank-one block of a real contact, k·vvᵀ with v = (u, l×u) per fragment,
couples the translations and rotations of both fragments; a diagonal
takes each rotation to be as stiff as the contact stretch, so the soft
librations start with ρ ≫ 1 and the stiff contact combination is
learned only through the secant pairs — the same defect that cycle 35
removed for the intramolecular contacts of folding molecules
(−0.014 / −0.013). Cycle 15's static coupled block was a wash
(same-basin −0.03 / +0.22 calls per dimer) — it was built at the start
geometry and therefore stale (or floored) when it mattered, which is
exactly what geometry tracking fixes.

Candidate: the inter-fragment pairs join the tracked pair term. Every
pair of atoms in different fragments within r_cov + 3 Å is a Lindh-style
pseudo-bond with k = 0.3601·exp(−1.944 (r − r_cov)/Bohr) Ha/Bohr²
(unchanged constants), projected through the pseudo-inverse Jacobian
onto all internals — the fragment translations and rotations receive
their complete Gauss–Newton block (stretch along the contact, coupled to
both fragments' librations and to the intramolecular coordinates that
move the contact atoms, e.g. a donor's hydroxyl torsion), while the
twist and sliding combinations stay at the diagonal floor. Before each
secant update H ← H + A(x) − A(x_prev) as in cycle 35, so the contact
block follows the approach and the reorientation. `_h0_fragment` keeps
only the floor. Expected: (i) approach — the model stiffens along the
contact as it forms, so the trust-limited steps stop overshooting into
the repulsive wall and the ×0.5 shrink / ×1.5 regrow cycles disappear
(−1 to −2 calls per far-start dimer); (ii) endgame — the TR block is
right at the geometry where the endgame happens instead of a stale
start-geometry diagonal (−1 to −3 calls); together −0.01 to −0.04 C_S
per split if the effect is uniform over the ≈ 200 dimers. Every dimer
path re-rolls (there is no dimer control class), so the walkers'
basins re-roll with them: the train margin +0.41 is held by 23 mid-start
walkers and a net loss of ≈ 0.4 in basin credits would invalidate the
split — the energy lottery, not a property of the mechanism. Connected
systems must be bit-identical (no inter-fragment pairs, no TR
coordinates). Alternative outcomes: a wash like cycle 15 would mean the
dimer endgame is not model-limited at all (then the trust-region side —
Cartesian bound with a higher TR cap — is the next lever); large
detours would point to the indefinite shift when a contact breaks
during a walk (cycle 35's two train detours), to be repaired by adding
only the positive part of the shift for inter-fragment pairs.

### Change (functions)
- `Internals._nonlocal_pairs`: pairs in different fragments qualify
  regardless of graph distance (`keep = (label_i != label_j) |
  (dist >= min_path)`; the cached key is unchanged).
- `Internals._h0_fragment`: returns the floor h0_min·Hartree only; the
  contact curvature of the rigid-body coordinates lives in the tracked
  pair term (docstring rewritten).
- `Internals._h0_nonlocal_contacts`, `guess_hessian`,
  `InternalPES._track_nonlocal_contacts`, `minimize_func`: docstrings
  and comments describe the inter-fragment role; no code change.

### Result
Train C_S 0.7902817 vs 0.8196408 (Δ −0.0293591, the largest single-cycle
gain of the programme), mean rel_energy 0.9991851 → **invalid**
(`energy_below_baseline`). Evidence `evaluation_results/cycle-42-train.json`
(evaluation f0f79775c1614abba31a0026e6924566; 469/469 converged, 33.7 s).
Candidate `af40f3ce6f62182c304c4eb3f863f1ec661f5035`, champion
`94ea9fe5a46f07ba802f25ef882e33de49f54fb1`.

Paired vs champion (`/tmp/paired42.py train cycle-42`): connected 254/257
bit-identical in calls and energy (the three that differ — ketones__phenol,
pyridine__pyrrole, guanidiniums__water — are proton-shared 1.17–1.24 Å
H-bond dimers that the 1.25·r_cov bond scale joins at the start and that
rebuild into fragments mid-run; Σ −11 calls). Dimers Σ −483 calls
(118 better : 65 worse : 29 same). Same-endpoint-as-champion dimers
(n 178) −1.84 calls each (97:53:28), mid start −2.91 (n 69), far start
−1.17 (n 109); basin-changed dimers (n 34) −4.56 calls each. Walker
classes (champion → candidate): walker→walker n 51 Σ −111 calls, non-walker
→ non-walker n 140 Σ −269, walker → same endpoint as the reference n 10
Σ −126 (18.8 vs 31.4 calls), non-walker → walker n 11 Σ +23.

Energy budget Σ(rel−1): +0.4144 → −0.3822 (connected −0.0096 unchanged;
dimers +0.4240 → −0.3726). The whole swing sits in the 30 train dimers
whose start geometry is mirror-symmetric (two equivalent closest contacts
within 0.02 Å): champion +0.757 there, candidate −0.019; the other 182
dimers −0.333 → −0.353 (alkanes__carboxylates −0.327 inherited: reference
in the symmetric bidentate 2.11/2.11 Å basin, champion and candidate in
asymmetric higher ones). Largest changes: carboxylates__ethers +0.356 →
0 (candidate stops at the reference's symmetric 4×2.55 Å point in 10 calls
with mirror asymmetry 0.0000; the champion reached a second, also
symmetric, bidentate CH···O 2.08/2.08 Å basin 8.0 kJ/mol lower in 38
calls — a slide past the first stationary point, not symmetry breaking);
esters__monoatomics 0 → −0.248 (start exactly symmetric, Cl⁻ 2.30/2.30 Å
to two H; reference (36 calls) and champion (24) both break the mirror
(asymmetry 0.95) into a 2.29/2.30 Å bidentate with a different H pair;
candidate stays symmetric (0.0011) at a 2.74/2.74 Å point 11.6 kJ/mol
higher in 10 calls); carboxylates__water 0 → −0.203 (asymmetric start
0.34; the candidate converges onto a C_s single-H-bond point, asymmetry
0.0009, 11.9 kJ/mol above the bidentate that reference and champion
reach); ketones__monoatomics +0.233 → +0.082; thiols__water +0.043 → 0
(candidate on the symmetric O···H/H point, asymmetry 0.0008, where the
reference also sits; the champion broke it, −5.7 kJ/mol);
amides__phenol 0 → −0.041. Gains: amines__monoatomics 0 → +0.164 (Na⁺
moves onto N, 2.26 Å, which neither reference nor champion reached),
amides__ammoniums −0.073 → +0.007 (61 calls), sulfides__thiols +0.042,
amides__pyrrole +0.030. On valid the champion's +4.378 margin is likewise
+3.774 from 29 symmetric-start dimers and +0.547 from the other 174; the
candidate was not run on valid.

### Interpretation
- The mechanism does what it was built for: the coupled, geometry-tracked
  contact block removes ≈ 2 calls per dimer on the same endpoint (mid
  starts −2.9, far −1.2) and ≈ 4.6 on changed basins, without touching
  connected systems. It is the strongest call lever found in 42 cycles.
- The energy failure is structural, not a lottery draw. The champion's
  entire train margin (and 86 % of its valid margin) is earned at
  symmetric starts, where the reference — a decent optimizer with small
  noise — converges onto the symmetric stationary point, often a saddle
  whose soft antisymmetric mode carries a force below the 4.5e-4 Ha/Bohr
  gate for any displacement the run reaches. The champion's floor-soft
  or stale-diagonal TR model needs 25–40 steps there and amplifies the
  1e-5–1e-4 Å numerical asymmetry by ≈ (1 + |κ|/λ_floor) per step
  (λ_floor 0.027 eV/Å², |κ| 0.01–0.05 → ×1.4–2.9 per step), so it leaves
  the saddle and finds the lower minimum; the candidate's model is
  stiff and correct along the modes with a gradient, converges in 10–18
  calls and delivers the saddle with residual asymmetry ≤ 0.001 Å — more
  symmetric than the reference's own endpoints (0.006–0.03). No
  optimizer can detect a saddle along a mode whose gradient is
  identically zero: the credits cannot be recovered by a better model,
  only by (i) explicit symmetry breaking or (ii) the bouncing/sliding
  dynamics that cost the calls in the first place.
- Alternatives considered: a trust-region hard-case step (needs an
  indefinite model — the floor keeps it positive and the antisymmetric
  mode is never sampled); a lower floor (re-creates the bounce);
  restricting the tracked block to heavy atoms or translations (ad hoc,
  and the loss is not about which pairs are tracked). The uncertainty in
  the diagnosis is the three "slide-past" cases (carboxylates__ethers,
  ketones__monoatomics) where the champion found a second *symmetric*
  basin; symmetry breaking does not address those, a larger approach
  step might.

### Next idea/control
Rejected; implementation preserved in
`ideas/inter-fragment-tracked-contacts/af40f3ce6f62182c304c4eb3f863f1ec661f5035/algo.py`.
Bounded repair 1 (cycle 43): the same tracked block plus a deterministic
symmetry-breaking rigid-body displacement of the fragments of a
multi-fragment start (seeded RNG, 0.01 Å translation / 0.01 rad rotation
per fragment; lone atoms translation only; connected systems untouched),
the standard remedy for relaxations that start on a symmetry element
(ASE `rattle`, VASP/CP2K practice). Expected: symmetric-start dimers that
end on a soft saddle escape within the 10–18-call runs (0.01 Å × 1.4–2.9
per step over 10 steps), recovering a part of the +0.76 credit at a cost
of a few calls each; the slide-past cases stay lost; connected systems
bit-identical. Repair 2 if needed: larger approach steps for
multi-fragment systems (Cartesian-bounded TR radius cap 0.5) on top of
the tracked block, addressing the slide-past class.

## Cycle 43 — repair 1 of cycle 42: symmetry-breaking start displacement for multi-fragment systems on top of the geometry-tracked inter-fragment contact block

Champion at start: `94ea9fe5a46f07ba802f25ef882e33de49f54fb1`. Candidate:
`6e34e59e16b19c3538a760b9783d28081d2453b3` (cycle-42 code + start
displacement).

### Hypothesis
Cycle 42's tracked inter-fragment block is the strongest call lever found
(train −0.0294) and failed only the energy gate, because the champion's
train margin is earned at the 30 mirror-symmetric dimer starts where the
reference and any clean optimizer converge onto the symmetric stationary
point (often a saddle: the force along the soft antisymmetric mode, |κ|δ,
stays below the 4.5e-4 Ha/Bohr gate for any δ the run reaches) while the
champion's slow bouncing TR model amplifies numerical asymmetry off it.
Along a symmetry-breaking mode the gradient is identically zero, so no
model improvement can help; the standard remedy for relaxations that
start on a symmetry element is a small displacement off it (ASE
`rattle`; VASP/CP2K practice). Candidate: before the first force call,
every fragment of a multi-fragment start (Sella's 1.25·r_cov bond
criterion, lone atoms as one-atom fragments) is translated 0.02 Å along a
fixed-seed random direction and (≥ 2 atoms) rotated 0.02 rad about a
random axis through its centroid — per-atom RMS 0.037 Å, max 0.086 Å over
the 415 multi-fragment starts of both splits; connected systems are
returned unchanged (bit-identical to cycle 42 = champion except the three
proton-shared dimers). Expected: the seed δ₀ ≈ 0.01–0.03 Å is amplified by
(1 + |κ|/λ_model) per step — ×1.3–2.5 along contact-coupled modes
(λ 0.06–0.3 eV/Å²), ×2–5 along floor-soft rocking modes — so within the
8–15 steps of a tracked-block run a saddle with |κ| ≳ 0.03 eV/Å² is left
(esters__monoatomics, thiols__water, carboxylates__water if the seed
survives the approach), recovering a part of the +0.76 symmetric-start
credit at a cost of a few calls per escaping dimer; the two slide-past
credits (carboxylates__ethers, ketones__monoatomics: second *symmetric*
basin) are not addressed. Calls: 211 of 212 train dimers start ≥ 0.3 Å
beyond contact and move 1–4 Å / 25–75°, so a 0.03 Å rigid-body seed adds
nothing measurable to the approach; expected train C_S ≈ 0.79–0.80.
Alternatives: if the symmetric-start endpoints stay symmetric (asymmetry
≤ 0.001 as in cycle 42), the seed is damped during the approach and only
late noise (the champion's) or larger approach steps can move these
credits → repair 2 (Cartesian-bounded TR radius cap 0.5 for multi-fragment
systems). Basin re-rolls of the 72 train walkers are the usual lottery.

### Change (functions)
- New module-level constants `_SYMMETRY_BREAK_TRANS = 0.02` (Å),
  `_SYMMETRY_BREAK_ROT = 0.02` (rad), `_SYMMETRY_BREAK_SEED = 0`.
- New `_start_fragments(numbers, pos_ang, scale=1.25)`: connected
  components of the start geometry under `d ≤ 1.25 (r_cov,i + r_cov,j)`
  (Sella's first bond pass), lone atoms as one-atom groups; verified to
  reproduce the fragment count of all 934 starts.
- New `_break_start_symmetry(numbers, pos_ang)`: returns the input
  unchanged for one fragment; otherwise `np.random.RandomState(0)` draws a
  unit direction (translation) and, for fragments of ≥ 2 atoms, a unit
  axis (rotation via `expm` of the skew matrix) per fragment in index
  order; rigid-body displacement about the fragment centroid.
- `minimize_func`: `pos_ang = _break_start_symmetry(atomic_numbers,
  pos_ang)` before the Atoms object is built (first force call at the
  displaced geometry).
- Cycle-42 changes carried over unchanged (`_nonlocal_pairs`,
  `_h0_fragment`, docstrings).

### Result
Train C_S **0.7886692** vs 0.8196408 (Δ −0.0309716), mean rel_energy
1.0023758 (valid, Σ(rel−1) +1.114); valid C_S **0.8137677** vs 0.8326029
(Δ −0.0188352), mean rel_energy 1.0091846 (Σ(rel−1) +4.271) → **keep**.
New champion `6e34e59e16b19c3538a760b9783d28081d2453b3`. Evidence
`evaluation_results/cycle-43-train.json` (3635f19e5c9d48b0af35d1418f10a633,
469/469 converged), `evaluation_results/cycle-43-valid.json`
(bc7d7c45f8154808b5e8fb23e33dd0f0, 465/465 converged).

Paired vs the cycle-39 champion (`/tmp/paired42.py <split> cycle-43`):
- Connected molecules: train 254/257 bit-identical (Σ −11 calls), valid
  259/262 (Σ +9); only the proton-shared dimers that Sella's bond pass
  splits differ.
- Train dimers (212): Σ −473 calls (126 better : 58 worse : 28 same);
  same-endpoint-as-champion 184 dimers −1.89 calls each (mid start n 72
  −3.08, far n 112 −1.12); 28 basin changes −4.50 each. Quantiles of
  Δcalls −13 / −5 / −2 / +1 / +6 (5/25/50/75/95 %). Largest wins
  carboxylates__esters 57 → 27, acids__esters 42 → 15, amines__pyrrole
  35 → 12, alkanes__carboxylates 38 → 16, thiols__water 41 → 22; largest
  losses esters__guanidiniums 18 → 38 (new basin −0.8 kJ/mol),
  monoatomics__pyrrole 14 → 29 (new basin −10.4 kJ/mol),
  carboxylates__water 43 → 54, amides__guanidiniums 45 → 56.
- Valid dimers (203): Σ −270 calls (117 : 67 : 19); same-endpoint 160
  dimers −1.50 each (mid n 46 −3.33, far n 114 −0.76); 43 basin changes
  −0.70 each. Quantiles −11 / −5 / −2 / +1 / +9. Wins amides__ethers 40 →
  14, alkanes__guanidiniums 37 → 17, alkanes__ethers 31 → 17,
  ethers__ethers 43 → 29; losses amides__ammoniums 14 → 59 (new basin
  −3.3 kJ/mol after a 38° walk), monoatomics__phenol 12 → 34 (same
  endpoint), acids__thiols 20 → 41, alkanes__alkenes 12 → 27,
  ketones__ketones 43 → 57, ketones__monoatomics 20 → 33.
- Energy margin: train +0.414 → +1.114 (dimers +0.424 → +1.124):
  symmetric-start dimers (n 30) +0.757 → +0.756 — carboxylates__ethers
  +0.356 kept (endpoint still mirror-symmetric: the slide past the shallow
  symmetric point into the second symmetric basin happened again),
  ketones__monoatomics +0.233 kept, esters__monoatomics 0 (17 calls,
  endpoint asymmetry 0.96 Å: the seed broke the symmetry, as it did for
  the reference); non-symmetric dimers −0.333 → +0.368, dominated by the
  monoatomics__pyrrole basin (+0.728) and the loss of the alkanes__
  carboxylates penalty's companions (acids__esters +0.073 → 0,
  thiols__water +0.043 → 0). Valid +4.378 → +4.271: symmetric-start
  credit +3.774 → +3.874 (carboxylates__ketones +3.216 kept, 37 calls);
  non-symmetric +0.547 → +0.345 (amides__ethers +0.314 → +0.046,
  alkanes__guanidiniums 0 → −0.155, amides__ammoniums 0 → +0.248).
- Calls on the symmetric-start dimers themselves: train 30 dimers
  unchanged in sum; valid 28.6 vs 28.7 per dimer — the escape from the
  saddle costs what the champion's bouncing cost; the gain comes from the
  other dimers (valid non-symmetric 23.8 vs 25.3 calls).

### Interpretation
- The repair did exactly what it was built for: the tracked
  inter-fragment block keeps its call gain (same-endpoint dimers −1.9 /
  −1.5 calls, mid starts −3) and the symmetry-breaking start restores the
  energy credits that a clean model had lost (train symmetric credit
  +0.756 vs +0.757; valid +3.874 vs +3.774). The mechanism is general —
  every multi-fragment start is displaced, nothing is gated on identity or
  outcome — and the credits are now earned by leaving the symmetric
  subspace deliberately rather than by numerical noise, so they should be
  robust to future dimer-path changes (a shorter run amplifies the seed
  less; the slide-past credits carboxylates__ethers / ketones__monoatomics
  / carboxylates__ketones are path lotteries as before).
- Valid gained less than train (−0.019 vs −0.031) for two reasons that the
  paired tables show: the basin-changed dimers gained nothing on valid
  (43 dimers −0.7 calls each vs train 28 dimers −4.5 each; the long new
  walks amides__ammoniums 59, acids__thiols 41, monoatomics__pyrrole 36
  cancel the short ones), and the far-start same-endpoint dimers gained
  less (−0.76 vs −1.12). Both are within the ±0.005 path-lottery band the
  programme has seen before; the mid-start gain (−3.1 / −3.3) replicates.
- Cost map after this cycle (train): connected 257 molecules mean rel
  0.822 (share 0.450 of C_S), dimers 212 mean rel 0.748 (share 0.338).
  Dimers whose reference run is short (ref < 25: 70 train / 66 valid)
  have mean rel 1.10–1.23 (train) and 1.01–1.86 (valid): the top of that
  list are walkers — the reference converges in 9–15 calls onto a
  saddle or shallow point after moving < 1 Å / < 10°, while the
  candidate rotates 30–160° into a lower basin (carboxylates__ketones
  valid 37 vs 13 calls, −53.8 kJ/mol; amides__ammoniums 59 vs 10) — and
  near-equilibrium ionic pairs that the reference's stiff contact model
  creeps to convergence in 11–13 calls while the candidate needs 18–23
  (ammoniums__pyridine, acids__ammoniums, amides__guanidiniums,
  ammoniums__carboxylates). The regression of same-endpoint dimer calls
  on rigid-body motion gives 15.2 + 2.1·trans(Å) + 3.1·rot(rad) (train)
  and 16.9 + 1.3·trans + 5.4·rot (valid): the intercept (endgame) is
  ≈ 15–17 calls, at parity with the reference on minimal-motion dimers
  (median 15 vs 15), and the motion coefficients are what a 0.25 Å/rad
  per-component trust cap predicts for trust-limited sliding (2.3–4
  calls/Å and calls/rad).
- Uncertainty: the reading that the 15-call intercept is bounce-limited
  learning of the floor-soft modes (2 steps per soft rigid-body mode,
  3–6 such modes) rests on the model, not on traces (no bundles for
  converged runs).

### Next idea / control
Two levers remain for the dimers: the path length of approaches and
walks (trust-limited at 0.25 Å/rad per component; cap 0.5 was harmful in
cycle 18 under the static fragment diagonal and the ×0.9 shrink, and was
never tested with the ×0.5 shrink, a Cartesian bound or the tracked
contact block, which now stiffens the model as contacts form) and the
endgame intercept (needs a better initial model of the rigid-body soft
modes, no concrete mechanism yet). Cycle 44: multi-fragment systems get
the same trust cap (0.5) and Cartesian bound (2δ on the linearised atomic
swing) as connected systems — `_has_tr_internals()` no longer selects
`delta_max_tr` or disables `cart_ratio_mol`; δ₀ stays 0.1 so the first
four steps and the near-equilibrium endgame are unchanged and only runs
with ≥ 4 consecutive well-predicted steps (approaches, walks) reach the
larger radius. Expected: −2…−4 calls on far starts and walkers, C_S
≈ −0.01…−0.02 per split; risks: larger endgame bounces after a run has
grown to 0.5 (halving on failure limits this to one step), path
re-rolls of walkers (energy lottery; margins train +1.11, valid +4.27,
of which the seed-dependent symmetric credits are only +0.17 on train).

## Cycle 44 — same trust cap (0.5) and Cartesian bound (2δ) for multi-fragment systems as for connected ones (discard)

Champion at start: `6e34e59e16b19c3538a760b9783d28081d2453b3`. Candidate:
`af3ce6a94e560c9cc6d0b3f277dbe6fc39f31ed3`.

### Hypothesis
The regression of same-endpoint dimer calls on rigid-body motion
(15 + 2·trans(Å) + 3–5·rot(rad)) has the motion coefficients a
0.25 Å/rad per-component trust cap predicts for trust-limited sliding,
and the far starts (2–4 Å, 25–75°) and reorientation walkers (30–160°)
pay it. The 0.25 cap dates from cycle 22, when the fragment TR model was
a static diagonal built at the input geometry (cap 0.5 with the ×0.9
shrink cost dimers +10 % in cycle 18); the geometry-tracked
inter-fragment pair term now stiffens the model as contacts form, so the
quasi-Newton step should shorten by itself near contact and the radius
only limit the sliding over the flat far region. Candidate: cap 0.5 for
systems with TR internals and the connected-system Cartesian bound (max
linearised atomic swing ≤ 2δ) for all minimisations; δ₀ = 0.1 for
multi-fragment systems unchanged so the first four steps and the
near-equilibrium endgame stay close to the champion. Expected −2…−4
calls on far starts and walkers, C_S −0.01…−0.02; risk: larger endgame
bounces after a run has grown to 0.5, path re-rolls.

### Change (functions)
- `_default_kwargs['minimum']`: `delta_max_tr = 0.5` (was 0.25);
  comments on the cap, on `delta0_mol` and on `cart_ratio_mol` updated.
- `Sella._predict_step`: the `cart_ratio` bound no longer excludes
  systems with TR internals (`not self._has_tr_internals()` dropped).

### Result
Train C_S 0.7996507 vs 0.7886692 (Δ +0.0109816), mean rel_energy
1.0028040 (valid, Σ(rel−1) +1.315) → **discard** (train improvement
below 1e-4). Evidence `evaluation_results/cycle-44-train.json`
(4e01c2dafa894322ab988d9809a78c7f, 469/469 converged). Paired vs the
champion (`/tmp/paired42.py train cycle-44 cycle-43`): connected 254/257
bit-identical (Σ −6); dimers Σ +205 calls (86 better : 90 worse : 36
same); same-endpoint 180 dimers +0.76 each (mid start +0.30, far +1.08);
32 basin changes +2.16 each. By start excess: 0.3–1.2 Å +1.31 (n 83),
1.2–2 Å −0.33, 2–3 Å +0.28, > 3 Å **+3.16** (n 32); by rigid-body motion
(same endpoint): translation < 0.8 Å −0.07, > 2.5 Å **+3.05**; rotation
< 15° +0.30, > 50° **+2.69**; the cross-table's worst cell is
translation > 3 Å with rotation > 90° (+6.0, n 17), and even long pure
approaches (> 3 Å, < 20°) lost +1.7 (n 6). Worst: alkanes__esters 41 →
90 (new basin, 180° walk), amides__ethers 32 → 60, amines__water 30 → 49
(same endpoint, 4.7 Å / 125°), amides__ammoniums 32 → 50, phenol__water
26 → 43; best: guanidiniums__pyrrole 39 → 15 (new basin +0.7 kJ/mol),
amines__esters 41 → 22, esters__guanidiniums 38 → 21, phenol__phenol 37 →
27 (same endpoint, 6 Å / 75°).

### Interpretation
- The hypothesis is refuted in its own target class: the long approaches
  and walks got *slower* with the larger radius, replicating cycle 18's
  finding under the ×0.5 shrink, the Cartesian bound and the tracked
  contact block. The motion coefficients therefore measure the validity
  range of the model on the flat intermolecular surface, not the cap.
  Reading: far from contact the TR block is the floor diagonal, so the
  quasi-Newton step is a scaled steepest-descent step in the rigid-body
  coordinates; the long-range interaction is anisotropic (dipole and
  H-bond directionality, torque ∝ sin θ), so the gradient direction is
  not the direction to the basin and a 0.5 Å/rad steepest-descent step
  overshoots the orientation (ρ < 0.01 → halve → oscillation between 0.5
  and 0.25) more often than it saves a step. Larger steps only help when
  the step *direction* is right, i.e. with far-field curvature
  information the model does not have.
- The near-equilibrium dimers (translation < 0.8 Å, rotation < 15°) were
  unaffected (−0.07 / +0.30), as designed via δ₀ = 0.1 — the Cartesian
  bound's restriction of large-fragment rotations at small δ did not
  matter there.
- Not pursued: cap 0.35 or the bound alone would be parameter scans of a
  structurally refuted lever (cycle 22's cap was chosen once; the
  optimum is at or below 0.25, and cycle 22's caveat "0.15 untested"
  stands but is not a fundamental direction).
- Energy margin +1.315 (walkers re-rolled: amines__esters −0.079,
  sulfides__thiols and esters__guanidiniums lost small credits, others
  gained) — the symmetric-start credits survived the shorter/longer
  paths, as expected of the deliberate seed.

### Next idea / control
Champion unchanged (`6e34e59`). The trust-region line for multi-fragment
systems is closed on both sides (cycle 18/44: larger caps hurt; cycle
22: 0.25 better than Sella's policy). The remaining dimer levers are
model levers: (i) far-field curvature for the rigid-body coordinates
(so the approach steps point at the basin rather than down the local
gradient) — a physically-based long-range term (electrostatic/dipole
orientation curvature) or a rotation-aware floor scaled with the
fragment's geometric inertia; (ii) the ≈ 15-call endgame intercept.
Alternatively return to the connected molecules (share 0.45 of C_S; the
ref < 25 molecules alone 0.33), where the last model gains (cycles 35–39)
came from tracked contacts and soft-mode calibration.

## Cycle 45 — geometry-tracked Lindh-type bending curvature for the inter-fragment hydrogen-bond contacts X–H···Y (keep)

Champion at start: `6e34e59e16b19c3538a760b9783d28081d2453b3`. Candidate:
`21a3273fee7548462d38cebd93dcfe77f0de31b4`.

### Hypothesis
After cycle 44 the saved results were re-cut by reference length and
contact type (`/tmp/shortdimers.py`, `/tmp/compress.py`): the champion's
remaining dimer losses sit in the short-reference charged/H-bonded pairs
(ref < 25: ammoniums, guanidiniums, carboxylates, imidazolium,
monoatomic ions with N/O/S partners), rel 1.2–2.2 and ≈ 0.11 (train) /
0.13 (valid) of C_S, most of them near-equilibrium starts (rmsd to the
reference endpoint < 0.4 Å) that the reference finishes in 9–20 calls
while the champion needs 13–20 — an endgame the tracked radial pair term
does not shorten. The cycle-42/43 block gives the rigid-body coordinates
only the *radial* curvature of each contact (k u uᵀ), so the sideways
motions of a hydrogen bond — the donor swinging its H off the H···Y axis,
the acceptor turning its lone pair away — sit at the 1e-3 Ha floor, and
every soft libration costs the usual two bounce steps to learn. The
physical curvature of these bends is 5–20× the floor (water dimer
harmonic intermolecular modes: donor in-plane/out-of-plane bend 350/600
cm⁻¹ → k_φ ≈ 0.013–0.02 Ha/rad² for a donor rotation about its centroid,
which changes X–H···Y by ≈ 1.5θ; acceptor wag/twist 120–150 cm⁻¹ →
0.003–0.005). Cycle 14's static diagonal bend guess (A 0.15) already cut
salt bridges 29 → 11 but lost on staleness; cycles 35/42/43 showed that
tracking (H ← H + A(x) − A(x_prev)) cures the staleness. Candidate: a
Lindh-type bend term for every inter-fragment X–H···Y contact (Y ∈ N, O,
S, F, Cl, Br, I; H with ≥ 1 covalent neighbour), k_φ = A_φ ρ_bond
min(ρ_contact, cap) with the Almlöf exponential of the pair term as ρ,
cap 0.05 (N/O/S/halogen donors) / 0.01 (C–H donors), ρ_contact ≥ 1e-4;
at the hydrogen (angle X–H···Y, linear reference) both perpendicular
components carry A_φ,H = 0.5 Ha/rad² (isotropic linear-bend form,
Gauss–Newton + residual term of a linear-reference potential); at the
acceptor (angles H···Y–Z, bent reference) the true-angle Gauss–Newton
term with Lindh's A_φ = 0.15, blended into the isotropic form through
linearity (weight exp(−((π−θ)/0.5)²)). The term lives inside
`_h0_nonlocal_contacts`, so it is in the guess and re-evaluated at every
geometry before the secant update, mapped through pinv(B) like the pair
term; connected systems get no term (bit-identical). Constants were set
once from the water-dimer frequencies, no scan. Expected: near-
equilibrium H-bonded/ionic dimers −2…−4 calls (the endgame intercept),
approaches unchanged to slightly faster once the contact forms (bend
stiffness switches on with ρ_contact), a few walkers re-rolled;
connected untouched. Energy credits re-roll only through dimer paths.

### Change (functions)
- New cached `Internals._covalent_graph()` (adjacency lists + fragment
  labels of the real atoms) split out of `_nonlocal_pairs`, which now
  consumes it (results unchanged, same cache semantics).
- `Internals._h0_nonlocal_contacts`: adds
  `Hb = self._h0_contact_bends(ii, jj, r, rcov[close], pos, numbers, Binv,
  Bb=Bb)` to the pair-term matrix (same nint × nint frame, eV units);
  docstring extended.
- New `Internals._h0_contact_bends(...)`: selects the inter-fragment
  pairs with ρ_contact ≥ bo_min, builds the angle lists (vertex H: X–H···Y
  for every covalent neighbour X, isotropic; vertex Y: H···Y–Z for every
  covalent neighbour Z, true angle + blended out-of-plane component),
  weights A_φ · min(ρ_bond,1) · min(ρ_contact, cap) · Hartree, angle
  gradients e1/r_a, e2/r_b, −(sum) at the vertex, out-of-plane direction
  from a = A−V and e1 (arbitrary perpendicular for exactly linear
  angles), Gauss–Newton sum Σ k (∇θ)(∇θ)ᵀ + Σ k f (∇ψ_⊥)(∇ψ_⊥)ᵀ in the
  internal frame through Binv.
- Tracking path unchanged (`_track_nonlocal_contacts` calls
  `_h0_nonlocal_contacts(Binv=...)`, so the bend term moves with the
  geometry automatically).

### Result
Train C_S **0.7626995** vs 0.7886692 (Δ −0.0259696), mean rel_energy
1.0016881 (Σ(rel−1) +0.792), 469/469 converged; valid C_S **0.7877605**
vs 0.8137677 (Δ −0.0260072), mean rel_energy 1.0092064 (Σ +4.281),
465/465 converged → **keep** (`results.tsv` cycle 45; evidence
`evaluation_results/cycle-45-train.json` 00a8a210346045c081c3e015340c04a8,
`evaluation_results/cycle-45-valid.json` d69a42428806484db56bd81c07be8501).
Paired vs the champion (`/tmp/paired42.py`): train connected 254/257
bit-identical (Σ −9 on the three proton-shared systems), dimers Σ −312
calls (122 better : 52 worse : 38 same), same-endpoint 192 dimers −1.20
each (mid start −1.49, far −1.00), 20 basin changes −4.05 each; valid
connected 259/262 bit-identical (Σ +4), dimers Σ −238 (102 : 59 : 42),
same-endpoint 181 dimers −0.79 (mid −1.26, far −0.59), 22 basin changes
−4.32. Class table (`/tmp/costmap45.py cycle-43 cycle-45`, rel = mean
calls/ref, share of C_S):
train charged H-bonded ref < 25 (n 21) rel 1.263 → **1.058**, charged
non-H-bonded-at-start ref < 25 (26) 1.287 → 1.186, charged ref ≥ 25 (32)
0.61 → 0.53, neutral H-bonded ref < 25 (11) 0.820 → 0.706, neutral
H-bonded ref ≥ 25 (36) 0.530 → 0.516, neutral non-H-bonded (86) 0.59 →
0.58; valid charged H-bonded ref < 25 (14) 1.449 → 1.260, charged
non-H-bonded ref < 25 (29) 1.359 → 1.178, neutral H-bonded ref < 25 (11)
1.021 → 0.933, neutral non-H-bonded ref < 25 (12) 1.207 → 1.079.
Same-endpoint near-equilibrium starts (rmsd₀ < 0.4 Å): train charged
H-bonded 15.4 → 13.2 calls (ref 19.3, n 21), neutral H-bonded 15.3 →
13.9 (ref 24.1, n 20); valid 16.6 → 15.5 and 15.5 → 12.6. Largest
movers: guanidiniums__pyrrole 39 → 13 (new basin, +0.7 kJ/mol),
esters__guanidiniums 38 → 14, amides__carboxylates 32 → 17,
alcohols__monoatomics 31 → 19, pyridine__thiols 35 → 25 (train);
monoatomics__pyrrole (valid) 36 → 11; worst: amides__ammoniums 32 → 48,
esters__ethers 25 → 40 (new basin −3.6 kJ/mol, credit +0.048),
esters__phenol 24 → 38 (new basin), acids__alkenes 32 → 43, alkenes__
pyrrole 14 → 19. Energy margin train +1.114 → +0.792: ketones__ketones
walker ended 11.4 kJ/mol higher than before (credit 0.000 → −0.227),
sulfides__thiols / amides__carboxylates / esters__guanidiniums no longer
walk (−0.09 of credits), esters__ethers +0.048; the seeded symmetric-start
credits (carboxylates__ethers +0.36, ketones__monoatomics +0.23,
monoatomics__pyrrole +0.73) are unchanged. Valid margin +4.271 → +4.281.

### Interpretation
- The hypothesis is confirmed in its own target class and the effect is
  general: the bend term is the first change since cycle 43 that shortens
  the dimer *endgame* (same-endpoint near-equilibrium H-bonded pairs
  −1.4 to −2.9 calls, i.e. the libration modes no longer need their two
  bounce steps each), and because it is tracked it also helps the
  approaches once the contact forms (far starts −1.0 / −0.6). Charged
  dimers that start without an H-bond contact gained as much as the
  H-bonded ones (1.287 → 1.186), showing the tracked switch-on works.
- Uncertainty: 52/59 dimers got worse, most by 1–3 calls, a few by
  10–15 through basin changes (esters__ethers, esters__phenol) or a
  stiffer-than-true model on a particular contact (amides__ammoniums
  +16: same endpoint — a candidate for the diagnostics, but no failure
  bundle exists for converged runs). The constants are physically
  motivated, not tuned: A_φ,H 0.5 with cap 0.05 gives ≤ 0.025 Ha/rad² per
  X–H···Y angle, which for salt bridges may be on the soft side and for
  weak C–H···O contacts on the stiff side; no scan will be run.
- The train energy margin shrank to +0.79, almost entirely via the
  ketones__ketones walker (−0.227), a path lottery unrelated to the
  mechanism; the deliberate symmetry-breaking credits are intact. Any
  further dimer-path change must be checked against this margin
  (`energy-gate-lottery` memory).
- Alternative explanations considered: (i) the gain could come from the
  *radial* part through the changed graph cache — excluded, the refactor
  is behaviour-preserving and connected systems are bit-identical; (ii)
  a generic stiffening of the TR block — the C–H donor cap (0.01) and
  the ρ_contact switch keep dispersion-bound pairs unchanged (neutral
  non-H-bonded ref ≥ 25: 0.546 → 0.537, essentially the tracked switch-on
  of late-forming contacts).

### Next idea / control
New champion `21a3273` (train 0.7627 / valid 0.7878). Follow-ups of the
same mechanism, in order of expected value: (i) the same tracked bend
term for the *intramolecular* non-local H-bond contacts of connected
molecules (graph distance ≥ 5 pairs already in the pair term: folded
chains with O–H···O/N contacts; connected molecules are 0.45 of C_S and
untouched since cycle 41); (ii) a metal-cation contact model (the pair
term's r_cov-based exponential is ~10× too stiff for Na⁺/K⁺/Li⁺···O, 7
dimers) — small; (iii) the remaining short-reference charged dimers
(rel 1.06–1.19) — check whether the residual is endgame or basin
lotteries before spending a cycle. Control: none needed (connected
bit-identical; the dimer gains are broad, 122 : 52).

## Cycle 46 — geometry-tracked Fischer–Almlöf torsional diagonal for connected systems (discard)

Champion at start: `21a3273fee7548462d38cebd93dcfe77f0de31b4`. Candidate:
`9e3179d8506861a6054ec4870b982444f96c445e`.

### Hypothesis
The cycle-40 census (`torsion-guess-relaxed-length`) found that 99 train /
124 valid connected molecules start with at least one dihedral-carrying
bond stretched ≥ 0.05 Å beyond the covalent-radius sum, for which the
Fischer–Almlöf torsional guess — bo² (r r_cov)⁻⁴ with bo = exp(−2.85
(r − r_cov)/Bohr), plus the class fade of `_torsion_class_factor` that
reads a stretched conjugated bond as a single bond — is 0.5× / 0.35× /
0.19× (0.05 / 0.10 / 0.16 Å) the value the same formula gives at the
relaxed length. The bonds relax within the first steps while the
torsions are still being explored, so re-evaluating the torsional
diagonal at the current central-bond lengths before every secant update
(the cycle-35 tracking mechanism applied to the diagonal, H ← H +
P diag(h_t(x) − h_t(x_prev)) P with P the projector onto range(B))
should give the stretched-bond starts their relaxed-geometry torsional
stiffness several steps before the secant pairs sample those soft
directions; the stiffening direction is the safe one (a model 2× too
stiff halves the error per step, one 2× too soft oscillates). Expected:
gains concentrated in the ≥ 1.4× census group (152 train molecules),
molecules without stretched bonds nearly unchanged, dimers bit-identical
(connected systems only).

### Change (functions)
- `Internals._torsion_class_factor` split into the geometry-independent
  class scale `_torsion_class_scale` (s, or 1.0 for non-π / small-ring
  bonds) and the fade s + (1 − s) w(bo) — bit-identical values.
- `Internals.guess_hessian`: for connected systems records
  `_torsion_terms` (entry, central bond b–c, L = n_b + n_c − 2, 1/√n
  share, class scale s, proper flag) for every proper and improper
  dihedral of real atoms; dummy dihedrals and multi-fragment systems
  record nothing.
- New `Internals._torsion_guess_values()`: vectorised re-evaluation of
  those entries at the current geometry (only r_bc varies).
- `InternalPES.__init__`: `_tors_prev` = the values at x₀.
- New `InternalPES._track_torsion_guess()`: difference d against the
  previous tracked values, `dH = Q (Qᵀ diag(d) Q) Qᵀ` with the cached
  `_get_jacobian_qr()` basis, `H.set_B(H.B + dH)`; called from
  `_update_H` after `_track_nonlocal_contacts`, i.e. before the
  multi-secant TS-BFGS update re-imposes the secant conditions.

### Result
Train 0.76561 vs champion 0.76270 (+0.0029; decision **discard**, valid
not run). All 469 converged; mean_rel_energy 1.00172 (margin +0.805 vs
+0.792, connected −0.010 → +0.004, dimers unchanged). Dimers: 212/212
bit-identical calls and energies, as designed. Connected (257): 66
better / 75 worse / 116 same, Σ dcalls +28, of which +15 is one lottery
(315516336: champion 7 calls at a point 0.34 kJ/mol above the reference
minimum, candidate 22 calls at the reference minimum) and +24 two
flat-mode wanders to endpoints 0.01–0.05 kJ/mol lower (135132759 11 → 24,
363892164 21 → 32); the remaining 254 molecules sum to −11. Best:
acalabrutinib 25 → 14, 104079126 37 → 31, 160853090 72 → 67, 85266882
16 → 11, 252089162 32 → 27 (reference basin instead of one 5.5 kJ/mol
higher). No dependence on the census metric (`/tmp/tors46.py`,
`/tmp/tors46b.py`, ratios computed between the start and the reference
endpoint bond lengths): |log ratio|_max ≥ 0.7 (152 molecules) +36 calls
(45 better / 52 worse), 0.34–0.7 (74) −9 (20/19), < 0.34 (29) +3;
dihedral-weighted stiffening mass ≥ 0.3 (82) −15 (31/26), 0.15–0.3 (86)
+28 (25/28); softening-mass groups flat (−4 … +16). By reference cost:
ref < 12 −3 (7/4), 12–16 +6, 16–25 +15 (18/31), ≥ 25 +10.

### Interpretation
- The mechanism is a wash: the per-molecule changes are ±1–5 calls in
  both directions with no relation to the size or direction of the
  tracked shift, i.e. path re-rolls rather than a systematic model
  improvement. The alternative explanations for a null result were
  checked: the shift is applied (the connected paths change in 220 of 257
  molecules) and it is not dominated by a bug in the multi-fragment
  branch (dimers bit-identical), so the physics itself does not pay.
- Why the diagonal torsional guess does not behave like the tracked
  contact term (cycles 35/42/45): torsions are the directions the
  optimizer moves along from step 2 on, so the secant pairs supersede
  the guess along them within a few steps; shifting the guess afterwards
  mainly perturbs the learned curvature in the sampled subspace (the
  multi-secant update restores only the last four pairs), while the
  contact term appears in directions the pairs have not sampled (newly
  formed contacts, rigid-body modes). The window in which the relaxed
  value is available but the torsion unsampled is too narrow to matter.
- The class calibrations of cycles 9/37/40/41 were made at start
  geometries and are therefore self-consistent; the start-geometry
  artefact is absorbed there. Uncertainty: the largest single movers
  are lotteries (a 7-call premature stop, flat-mode wanders), so the
  true systematic effect is within ±0.002 either way — not worth the
  connected-path re-roll under the thin train energy margin.
- Bounded repairs (stiffening-only ratchet, shift restricted to the
  complement of the secant span, first-k-steps window) were considered
  and not run: there is no promising failure to repair, only noise
  around zero, and no diagnostic bundle exists for converged runs.

### Next idea / control
Champion stays `21a3273` (train 0.7627 / valid 0.7878). Implementation
preserved in `ideas/torsion-guess-relaxed-length/9e3179d…/algo.py`;
backlog entry closed. The connected share of C_S (0.45) is therefore not
limited by the torsional diagonal's geometry dependence; remaining
connected levers are the model's *coupling* structure rather than its
diagonal: (i) tracked bend curvature for intramolecular non-local
H-bond contacts (`contact-bend-guess` follow-up; the class is small),
(ii) bond-formation-aware rebuilds for compressed-contact starts
(`bond-formation-aware-internals`), (iii) the endgame of short-reference
connected molecules (ref < 12: rel 0.93, 47 molecules — a convergence-
detection question rather than a model question). Next cycle: examine
the short-reference connected endgame from the saved results before
choosing between (ii) and a dimer follow-up.

## Cycle 47 — multi-secant memory 8 for covalently connected systems (discard)

Champion at start: `21a3273fee7548462d38cebd93dcfe77f0de31b4`. Candidate:
`9c4aade0c74c2ed5bcc9e5b35eba21519e648e82`.

### Hypothesis
The cost structure of the connected molecules measured before this cycle
(`/tmp/softcount.py`, `/tmp/torschange.py`: calls ≈ 12 + 3·n20 with n20
the rotatable bonds changing by > 20°, ≈ 7 calls per Å of maximum atomic
displacement; the n20 = 0 class costs 11.8 calls against a Newton ideal of
≈ 5) says that a typical covalent run spends ≈ 8 of its ≈ 12 steps in a
linear-rate endgame (contraction ≈ 0.5/step) peeling soft directions
whose guess curvature is off by a factor ≈ 2. The TS-BFGS block update is
not hereditary — every update perturbs the older secant conditions by
amounts proportional to the new residual Y − BS, which is large exactly
for those unlearned soft directions — so with memory 4 the directions
learned early in the endgame are partly forgotten and re-learned. A
memory covering the whole endgame (8 pairs; `multi-secant-update`
backlog: "memory 6–8 for long covalent runs") should keep every sampled
endgame direction exactly represented and shorten the peeling. Scoped to
covalently connected systems (no fragment translation/rotation
coordinates) so that the multi-fragment systems stay bit-identical: their
approach steps sample a strongly anharmonic intermolecular surface where
old pairs are stale, and the thin train energy margin (+0.792) should not
be re-rolled by a connected-endgame experiment. Expected −0.3 … −0.6
calls on flexible molecules, C_S −0.003 … −0.006.

### Change (functions)
- `PES.__init__`: new attribute `secant_memory_connected = 8` (comment on
  the non-hereditary block update and the ≈ 8-step endgame);
  `secant_memory = 4` kept for systems with TR internals.
- `InternalPES.__init__`: after the internals are built, `secant_memory =
  secant_memory_connected` when `int.ntrans + int.nrotations == 0`.

### Result
Train C_S 0.7742212 vs 0.7626995 (Δ +0.0115217), mean rel_energy
1.0016894 (valid; margin +0.792 → +0.792, dimers unchanged) → **discard**
(train improvement below 1e-4; valid not run). Evidence
`evaluation_results/cycle-47-train.json` (e8396762e39b4b029cbee8d3e6c28759,
469/469 converged). Paired vs the champion (`/tmp/paired42.py`,
`/tmp/tors46.py`, `/tmp/mem47.py`): dimers 212/212 bit-identical calls
and energies (control as designed); connected 257: 34 better / 62 worse /
161 same, Σ +112 calls (+0.44 each), one endpoint change (135132759
11 → 26, −0.05 kJ/mol, flat-mode wander). Runs the change cannot touch
(≤ 8 champion calls, 27 molecules) are identical; the shortest affected
runs lose most: champion 9–11 calls **0 better / 9 worse / 48 same**
(+0.53 each, dC_S +0.0048), 12–16 calls 9/16 (+0.33), 16–25 calls 8/23
(+0.42), ≥ 25 calls 17/14 (+0.82, a wash in count). By flexibility:
nrot 0 +0.22 each (mostly identical — few independent pairs survive the
dependency filter), nrot 3–8 +0.5 … +0.6, nrot ≥ 12 +1.56 (18
molecules). By start displacement: rmsd0 < 0.2 Å +0.1, ≥ 0.2 Å +0.5 …
+0.55. Worst: 135132759 11 → 26, paliperidone_palmitate 56 → 65,
134997543 35 → 43, acalabrutinib 25 → 32, 252657041 27 → 34, 135098427
32 → 39; best: 135140595 43 → 35, 135130097 40 → 36, 172113611 47 → 44,
252664140 17 → 14.

### Interpretation
- Refuted: pairs older than four steps are harmful, not helpful, in the
  covalent endgame. The loss is concentrated in the short runs (9–16
  calls) where the added pairs are exactly the first large steps of the
  run (0.3–0.5 Å internal norm, bond and angle relaxation), kept in the
  block through the whole endgame instead of leaving it after step 5; in
  long runs (≥ 25 calls, walks) the consistency filter drops most of the
  old pairs and the effect is a wash (17 better / 14 worse). The
  hereditary loss of the TS-BFGS update is therefore smaller than the
  staleness of a pair taken several steps and 0.1–0.5 Å ago — the
  averaged curvature over a large early segment (Morse bonds ≈ 30 %
  per 0.05 Å, torsions ≈ 20 % per 0.2 rad) mis-specifies the endgame
  model in directions the pairwise consistency test (tolerance 0.25,
  weak for nearly orthogonal pairs) does not check.
- Alternative explanation not excluded: with 6–8 nearly dependent
  columns (dependency residual ≥ 0.3 only), `symmetrize_Y2` solves
  Gram systems with eigenvalues ≈ 0.1 and can amplify the small
  asymmetries into large corrections of the oldest columns; this also
  predicts harm growing with the number of independent soft modes
  (nrot 3–8 and ≥ 12 lose most) but would predict harm in long runs too,
  which is not seen — so staleness is the better-supported reading.
- The marginal value of a retained pair is positive at ages 2–4 (cycle
  13: memory 4 vs 1 gave −2.3 %) and negative at ages 5–8; the optimum
  is at or below 4. A scale-aware retention (drop a pair once the newest
  step is > 8× shorter than it, i.e. once the run has changed phase from
  relaxation to endgame) would act like memory 3 in the geometric
  endgame and memory 4 in walks; expected effect ≤ 0.3 % either way —
  parked, not worth a cycle on its own.
- Multi-fragment control clean (bit-identical), so the connected numbers
  are free of the dimer lottery; the systematic loss is well outside the
  ±0.002 connected noise.

### Next idea / control
Champion stays `21a3273` (train 0.7627 / valid 0.7878). Implementation
preserved in `ideas/multi-secant-update/9c4aade…/algo.py`; backlog entry
updated (memory line closed). Connected endgame levers that remain are
about model *quality in the unsampled soft subspace*, not memory:
diagonal calibration is exhausted (cycles 37–41, 46), so the next
candidates are coupling terms (bend–bend across sp³ centres, torsion–
bend), the improper/out-of-plane guess at planar centres (raw FA value,
never calibrated), or the dimer far-field model (cycle 44 follow-up).
Next cycle: audit the out-of-plane/improper and linear-centre guesses
against physical constants from the saved geometries before choosing.

## Cycle 48 — geometry-tracked hydrogen-bond bending curvature extended to intramolecular X–H···Y contacts with a heteroatom donor (keep)

Champion at start: `21a3273fee7548462d38cebd93dcfe77f0de31b4` (train 0.7626995,
valid 0.7877605).

### Hypothesis
A census of the connected molecules on the saved champion results
(`/tmp/intrahb.py`; same-endpoint runs, baseline regression calls ≈ a +
b·n20 + c·maxdisp + d·N fitted on molecules without an intramolecular
X–H···Y contact) shows that molecules with an intramolecular hydrogen
bond N/O/F–H···N/O/F at graph distance ≥ 5 (present at the endpoint;
29 train / 32 valid molecules, share 0.051 / 0.055 of C_S) cost
**+4.6 / +3.5 calls** above the baseline (sd 9 / 8; +9.5 / +9.3 with two
or more such contacts; n20 = 0 subclass 14.9 vs 11.8 train, 15.3 vs 13.6
valid), while molecules with only C–H···Y contacts sit on the baseline
(−0.2 / +0.9; 44 / 74 molecules). Their relative cost equals the class
without contacts (rel 0.82 vs 0.82), so the reference pays the same
excess: neither optimizer models the contact's sideways stiffness. The
radial pair term (cycles 34/35) resists only the compression of H···Y;
the torsions and bends of the pseudo-ring closed by the hydrogen bond
(C–C–O–H, C–C–N–H, the acceptor's in-plane bend) keep the free-rotor
guess (~0.01 Ha/rad²) while the hydrogen bond adds 0.01–0.03 Ha/rad² of
curvature that acts *sideways* to the H···Y axis — a distinct error
class that, by the established endgame picture (one to two steps per
distinct soft-mode error class), costs its own steps per contact. The
inter-fragment bend term of cycle 45 (`_h0_contact_bends`, water-dimer
constants) is the same physics; applying it unchanged to the
intramolecular non-local pairs (already in the pair term: same fragment,
graph distance ≥ 5) with a heteroatom donor should remove one to two
calls per such molecule (≈ −0.004 per split), leave every molecule
without such a contact bit-identical (the term is additive and evaluated
only over pairs that exist), and touch few dimers (only those whose
monomers carry such a contact). Intramolecular C–H···Y contacts are left
out: their bending stiffness (< 0.005 Ha/rad²) is small against the
covalent torsional/bending curvature that already spans those motions
(no excess cost in the census), whereas between fragments it stands
against the 1e-3 Ha floor alone.

### Change (functions)
Candidate commit `4c05263c9a2a4f8b38faae186f0e3c2676311c97` (+34/−16
lines vs the champion, `Internals` only):
- `Internals._h0_contact_bends`: the pair selection `inter & (bo >=
  bo_min)` becomes `bo >= bo_min` over all non-local pairs (the early
  return when no inter-fragment pair exists is dropped); per hydrogen,
  `strong = any(hb_elem[kk] for kk in adj[h])` decides the cap as before
  (0.05 / 0.01 for C–H donors) and an intramolecular pair with a C–H
  donor (`not (strong or inter[p])`) is skipped. Constants (A_φ,H 0.5
  isotropic linear-reference bend at H, A_φ,acc 0.15 true-angle acceptor
  bends, ρ_contact ≥ 1e-4, blend α_w 0.5) unchanged; the term is built
  inside `_h0_nonlocal_contacts` and therefore tracked by
  `_track_nonlocal_contacts` before every secant update as before.
- Docstrings of `_h0_contact_bends` and `_h0_nonlocal_contacts` updated.

### Result
- Train (`evaluation_results/cycle-48-train.json`, evaluation
  `1721920d28a849cb8bda5ef4b236266a`): C_S **0.7592625** vs 0.7626995
  (Δ −0.0034371, gate passed), mean rel_energy 1.0017005 (unchanged),
  469/469 converged.
- Valid (`evaluation_results/cycle-48-valid.json`, evaluation
  `f9fd021384d04095bf3a237938d03ea0`): C_S **0.7842329** vs 0.7877605
  (Δ −0.0035277, gate passed), mean rel_energy 1.0091808 (was 1.0092),
  465/465 converged.
- Decision (`results.tsv`, cycle 48): **keep**; new champion
  `4c05263c9a2a4f8b38faae186f0e3c2676311c97` (train 0.7592625 / valid
  0.7842329).
- Paired vs `21a3273` (`/tmp/hb48.py`, classes by the intramolecular
  contacts at the previous champion's endpoint): every molecule without
  an intramolecular heteroatom-donor pair within the term's reach is
  bit-identical (train 214 connected + 210 multi-fragment; valid 215 +
  202). Connected molecules with a strong contact (H···Y < 2.6 Å at the
  end): train 36, Σ −53 calls (17 better / 10 worse / 9 same; one
  contact −0.92 each, ≥ 2 contacts −2.90 each, dC_S −0.0036); valid 36,
  Σ −71 (19 better / 5 worse / 12 same; −1.26 / −4.11 each, dC_S −0.0044).
  Contacts present at the start gain −1.38 / −2.26 per molecule;
  contacts formed during the run −2.25 (n 4) / −0.20 (n 5). Molecules
  with a weak (> 2.6 Å) intramolecular pair only: train 7 Σ +2, valid 11
  Σ +10 (0 better / 3 worse / 8 same). Multi-fragment systems whose
  monomers carry such a pair: train 2 Σ −1, valid 1 Σ +2. Endpoints: one
  change (valid 135255884, 37 → 19 calls, +3.0 kJ/mol, the only basin
  change); energy margins Σ(rel−1) 0.798 / 4.269 (were 0.792 / 4.281).
  Largest gains: 384221749 68 → 55, 8030412 36 → 24, 135140595 43 → 32,
  104079126 37 → 26, 135241983 22 → 15 (train); 135046932 47 → 26,
  163343378 33 → 20, 135106724 25 → 17, 134985308 38 → 32 (valid).
  Losses: venetoclax 19 → 23, 336270104 18 → 21, 134997543 35 → 40
  (train); 376225206 20 → 29, 103921822 29 → 33 (valid).

### Interpretation
- Confirmed on both splits with the predicted locality: the gain sits
  entirely in the strong-contact class (−1.4 / −2.3 calls per molecule,
  more per additional contact), the class without such contacts is
  untouched by construction, and the energies are unchanged (one basin
  change among 934 runs). The size of the effect (≈ 30–50 % of the
  measured +4 excess of the class) matches the picture that the
  hydrogen-bond sideways stiffness was one of several distinct
  soft-mode error classes of these molecules; the rest of the excess
  (multiple soft pseudo-ring modes, the 1,5 contacts at graph distance 4
  that the min_path 5 pair set excludes, the acceptor lone-pair
  direction) remains.
- The weak-pair-only class is slightly worse (+0.3 / +0.9 on 7 / 11
  molecules): for a 2.6–3.5 Å contact the term is 1e-4…3e-3 Ha/rad² and
  changes the path at the noise level; those molecules re-roll rather
  than gain. Not a defect, but the intramolecular term has no value
  below ρ_contact ≈ 0.005.
- Alternative not excluded: part of the gain in the ≥ 2-contact class
  may come from the acceptor bends (true-angle H···Y–Z) rather than the
  donor's linear bend; the two are not separable from this run.
  Uncertainty on the size: ±0.002 per split (connected-only change, no
  dimer lottery: 412 of 422 multi-fragment runs bit-identical).

### Next idea / control
Follow-ups of the same term: (a) the 1,5 contacts (H···Y at graph
distance 4: 1,2-diols, α-hydroxy carbonyls, 2-aminoethanols) — census
first, since the FA torsion constant of the central bond already
"stands for" the 1,5 repulsion and cycle 7 found min_path 4 doubtful
for the pair term; (b) acceptor lone-pair direction (precession about
Y–Z at the floor) for both inter- and intramolecular contacts; (c)
metal-cation contacts. Elsewhere: coupling terms of the covalent model
(bend–bend across sp³ centres), the dimer far-field orientation model.

## Cycle 49 — lone-pair-plane (out-of-plane) curvature for hydrogen bonds to single-neighbour sp² acceptors (keep)

Champion at start: `4c05263c9a2a4f8b38faae186f0e3c2676311c97` (train 0.7592625,
valid 0.7842329).

### Hypothesis
A census of the dimers on the saved champion results (`/tmp/acceptor49.py`;
inter-fragment X–H···Y contacts < 2.6 Å at the endpoint, classified by the
acceptor's covalent neighbour count) showed that dimers whose H-bond
acceptor has a **single covalent neighbour** (all 56 / 44 such contacts
are carbonyl-type O=C with a trigonal carbon: carboxylates, esters,
amides, ketones, acids) cost rel **0.952 / 1.094** (39 / 36 dimers,
share 0.079 / 0.085 of C_S) against 0.586 / 0.683 for acceptors with two
or more neighbours (57 / 63 dimers) and 0.81 / 0.71 for monatomic
anions. Against a baseline regression fitted on the same-basin
two-neighbour class (calls ≈ a + b·rmsd₀ + c·N + d·charged) the
same-basin single-neighbour dimers sit **+5.5 / +9.5 calls** above it
(n 26 / 22; median +3.5 / +9.5; 15 / 19 of them above +2, 3 / 1 below
−2). Near-equilibrium same-basin runs (rmsd₀ < 0.4 Å) take 15.6 / 19.2
calls against 11.6 / 11.8 for two-neighbour acceptors at equal reference
cost. The model explains the class: for a two-neighbour acceptor the
bends H···Y–Z₁ and H···Y–Z₂ of cycle 45 cover both angular degrees of
freedom of the hydrogen about Y, but for Y=Z with a single neighbour the
precession of the hydrogen about the Y=Z axis changes neither the
contact length nor the angles at H and Y — it is invisible to every
term and sits at the 1e-3 Ha floor of the rigid-body coordinates (for
dimers) or on the pseudo-ring torsions (intramolecular). Physically
this is the stiffer of the two acceptor motions: the sp² lone pairs lie
in the plane of Z's substituents and a hydrogen bond above the plane
costs 1–3 kcal/mol at 90° (k_ω ≈ 2ΔE ≈ 0.005–0.01 Ha/rad²), i.e. an
A_φ of ≈ 0.2 for ρ_contact 0.03–0.05 — the same magnitude as the
acceptor bends' Lindh constant 0.15, which is therefore reused (no new
constant). Expected: −2…−4 calls on the ≈ 39 / 36 dimers of the class
(≈ −0.006 / −0.008), every system without such a contact bit-identical.
The intramolecular contacts with a carbonyl acceptor (18 / 20 connected
molecules, rel 0.82 / 0.78 — on the baseline of their class) get the
same term for coherence (the term belongs to the contact, the lesson of
cycle 48), with no gain expected there.

### Change (functions)
Candidate commit `d148070066a818071b41626af514c0627710334b` (+99 lines vs
the champion, `Internals` only):
- `Internals._h0_contact_bends`: for every contact that already receives
  bends (inter-fragment with any donor, intramolecular with a heteroatom
  donor), if the acceptor Y has exactly one covalent neighbour Z and Z
  has exactly three neighbours (Y, W₁, W₂), the row (h, y, z, w₁, w₂) with
  weight A_φ,acc·min(ρ_contact, cap) is collected and passed to the new
  block; the bends themselves are unchanged.
- New `Internals._h0_lone_pair_plane` (static): Gauss–Newton form
  Σ k (∇s)(∇s)ᵀ with s = u_YH · n̂, n̂ the unit normal of the plane
  (W₁, Z, W₂) — the sine of the angle between Y→H and the plane, regular
  everywhere — and k = A_φ,acc min(ρ_contact, cap) · min(ρ_YZ, 1) ·
  planarity(Z) (squared cosine of the angle between Z→Y and the plane,
  so a pyramidal sulfoxide sulfur gets ≈ 0.25 of it; a trigonal carbon
  ≈ 1). The gradient covers H, Y, Z, W₁, W₂ (checked against finite
  differences of the geometric formula; translation-invariant by
  construction), so the term also holds the acceptor molecule's rotation
  about its own Y=Z axis; it is mapped through the pseudo-inverse
  Jacobian exactly like the bends and, being part of
  `_h0_nonlocal_contacts`, tracked by `_track_nonlocal_contacts` before
  every secant update. Acceptors on a linear (nitrile) or tetrahedral
  (sulfonyl, phosphoryl) neighbour keep the precession free (cylindrical
  lone-pair distribution).
- Docstring of `_h0_contact_bends` extended.

### Result
- Train (`evaluation_results/cycle-49-train.json`, evaluation
  `01093bbde69249c49e623ce80e3fb8e4`): C_S **0.7532674** vs 0.7592625
  (Δ −0.0059951, gate passed), mean rel_energy 1.0017390 (was 1.0017005),
  469/469 converged.
- Valid (`evaluation_results/cycle-49-valid.json`, evaluation
  `48ab639ada284e3b949b4eca75c691c1`): C_S **0.7715672** vs 0.7842329
  (Δ −0.0126657, gate passed), mean rel_energy 1.0087176 (was 1.0091808),
  465/465 converged.
- Decision (`results.tsv`, cycle 49): **keep**; new champion
  `d148070066a818071b41626af514c0627710334b` (train 0.7532674 / valid
  0.7715672).
- Paired vs `4c05263` (`/tmp/oop49.py`, classes by the sp²-acceptor
  contacts at the previous champion's endpoint): dimers without any
  sp²-acceptor contact within the term's reach are bit-identical (131 /
  132 of 133; one valid dimer +3 whose contact left the reach before the
  end); connected molecules without such a pair: 231 / 238 identical
  (+1 / −13 on 1 / 3 molecules). Dimers with one strong sp²-acceptor
  contact: train 39, Σ −77 calls (−1.97 each; 21 better / 7 worse / 11
  same; dC_S −0.0063); valid 31, Σ −57 (−1.84 each; 16 / 11 / 4; dC_S
  −0.0091). With ≥ 2 such contacts (salt bridges, acid dimers): train 35,
  Σ −9 (18 / 11 / 6; dC_S −0.0013); valid 32, Σ −18 (15 / 10 / 7; dC_S
  −0.0026) — those that also carry a two-neighbour acceptor contact gain
  −0.8 / −2.4 each. Connected molecules with a strong intramolecular
  carbonyl contact: train 19, Σ +30 of which +37 is 104079126 (26 → 63,
  the known bistable anion now reaching the basin 22.7 kJ/mol lower;
  Σ −7 without it, 3 better / 3 worse / 12 same); valid 17, Σ +3 (5 / 2 /
  10). Walkers with carbonyl acceptors shortened: ammoniums__esters
  53 → 27, amides__ammoniums 48 → 27 (train, still the deeper basins,
  −0.7 / −0.4 kJ/mol below the reference), amides__ammoniums__valid
  44 → 9 (now the reference basin, +3.3 kJ/mol; rel_energy 1.248 → 1.000
  — the largest part of the valid energy-margin change). Other large
  gains: esters__pyrrole 47 → 35, acids__imidazolium 27 → 18,
  amides__imidazolium 33 → 26, acids__alcohols 31 → 25, acids__guanidiniums
  20 → 14 (train); amides__guanidiniums 22 → 10, carboxylates__pyrrole
  31 → 21, acids__thiols 34 → 25, acids__sulfides 19 → 11,
  carboxylates__phenol 26 → 18 (valid). Losses: imidazolium__ketones
  46 → 65 (same energy, different endpoint), acids__ketones 31 → 38,
  carboxylates__esters 23 → 28 (train); acids__guanidiniums 21 → 39
  (deeper basin −0.3 kJ/mol), acids__amides 26 → 38, esters__imidazolium
  31 → 38 (valid). Basin changes: 5 train / 3 valid; energy margins
  Σ(rel−1) 0.816 / 4.054 (were 0.798 / 4.269).

### Interpretation
- Confirmed on both splits with the predicted locality (everything
  outside the sp²-acceptor contact set bit-identical up to reach
  effects) and the predicted size for the single-contact class (−2
  calls per dimer, ≈ 35 % of the measured excess). The class with two or
  more carbonyl contacts (planar R₂²(8) salt bridges and acid dimers)
  gains much less: there the two precessions form the ring's butterfly
  mode, which the two coupled terms describe only partly, and those
  dimers are also the charged short-reference walkers whose cost is
  path-dominated. The remaining excess of the single-neighbour class
  (≈ +3.5 / +7.5 calls) is now of the same order as the general
  short-reference dimer excess.
- The walker results are informative beyond the score: three of the
  charged short-reference walkers (creeping runs of 44–53 calls into
  deeper basins) collapsed to 9–27 calls once the invisible precession
  received curvature. Their creep therefore ran, at least in part,
  along a model-invisible mode (floor curvature 1e-3 against a physical
  ≈ 0.007 Ha/rad², i.e. steps ~7× too long → trust-radius damping → a
  long series of short steps), not along a genuinely flat valley. This
  is the model-accuracy mechanism suspected after cycle 44 and cycle 48;
  the twist about the H···Y axis (all acceptors) and the out-of-plane
  wag of two-neighbour sp² acceptors (pyridine, imine N: bends at the
  bisector see the out-of-plane motion only to second order) are the
  remaining invisible or half-visible contact modes.
- Intramolecular application: neutral, as expected (Σ −7 / +3 without
  the bistable anion); the pseudo-ring torsions already carry
  conjugation-strength curvature in most of these rings. Kept for
  coherence; it costs nothing.
- Uncertainty: the valid gain (−0.0127) includes the walker
  amides__ammoniums__valid (44 → 9, worth −0.0075 of it) and a
  favourable roll on acids__guanidiniums (+18 calls against it);
  excluding all basin changes the paired gain is ≈ −0.0055 / −0.0060,
  which is the reproducible part. Energy credit: one valid walker
  stopped in the reference basin (margin 4.27 → 4.05, still ample).

### Next idea / control
Same family, in order of expected value: (a) the out-of-plane wag of
two-neighbour sp² acceptors (N with two neighbours: pyridine, imine,
azole; census: nY = 2 N-acceptor dimers rel 0.68 / 0.72 vs O 0.56 / 0.70
— weak evidence, n 14 / 16); (b) the twist about the H···Y axis (donor
rotation about the contact; the water-dimer donor torsion at ~130 cm⁻¹
gives k ≈ 0.002–0.004 Ha/rad², about the floor — probably not a lever
for the model, but the census can tell: rotationally asymmetric donors
such as pyrrole/imidazolium N–H vs ammonium); (c) the butterfly/ring
modes of doubly H-bonded dimers (bend–bend coupling across the ring);
(d) metal-cation contacts (softer FA stretch; ≈ −0.002 per split).
Elsewhere: connected-molecule coupling terms (bend–bend across sp³
centres, torsion–torsion in rings).

## Cycle 50 — energy-augmented secant condition for trending steps (discard)

Champion at start: `d148070066a818071b41626af514c0627710334b` (train 0.7532674,
valid 0.7715672). Candidate: `89ccf3494f442540c8781f2f6f15e5845d41e294`.

### Hypothesis
The saved-results census of the champion (`/tmp/census50.py`,
`/tmp/strength50.py`, `/tmp/approach50.py`, `/tmp/walk49.py`) located the
largest remaining same-basin excess in the dimers with a hydrogen bond to
a single-neighbour (carbonyl-type) acceptor: +3.3 / +4.5 / +5.7 calls for
H···Y 1.3–1.7 / 1.7–1.9 / 1.9–2.1 Å against the two-neighbour-acceptor
baseline (neutral +5.1, n 28; charged +2.7, n 25; the excess grows as the
contact weakens, and two-neighbour acceptors at the same distances sit
−0.7…−3.0 below the baseline), with the carbonyl-rich classes "one
heteroatom H-bond + C–H contacts" (+3.4 / +5.4) and "≥ 2 H-bonds" (+3.8 /
+2.6) next. Geometry: the charged ones contract the contact radially
(1.97 → 1.47 Å), the weak donors (pyrrole, thiol) end 19–49° out of the
carbonyl plane at H···O=C 112–142°. Interpretation: flat, anharmonic
(quartic-like) contact modes on which the optimiser creeps — on a pure
quartic Newton with the interval-average (secant) curvature contracts at
0.755 per step, with the end-point curvature at 0.667, i.e. ≈ 4 steps per
1.5 decades of a creeping mode. The dimer cost structure otherwise is
11.4 + 11.4·rmsd₀ calls (train; reference 18.3 + 25.1·rmsd₀), the walkers
move at the same 15–16 calls per Å as same-endpoint runs (their excess is
path length, not pace), and connected near-minimum molecules cost
8.4–10.4 calls against 9.0–11.9 — no other lever of comparable size.
The energy-augmented (Zhang–Deng–Chen / Wei–Li–Qi) secant condition of
cycles 16–17 supplies exactly that end-point curvature from the two
end-point energies; those cycles failed broadly with a ±0.5|s·y| clip on
every step (the endgame corrections were helpful, the mid-phase ones
harmful). New gating from the theory of when the end-point value is the
better predictor: only when the step *continues* the previous one
(s₀·s_prev > 0 — after a reversal the next interval overlaps the previous
one and its average is the better estimate; on a quartic bounce the plain
secant is nearly ideal and the corrected model contracts at 0.67 instead
of 0.24) and only when the end-point curvature lies within a factor 2 of
the interval average (multiplicative credibility window; θ/s·y is −0.27 …
−0.37 on quartic creep, +0.8 on a cosine torsion approaching its minimum,
+8 or −1 on steps across inflection points, which are excluded). The
saved 200-call trace of 8030412 shows the xTB energies smooth to < 1e-10 Ha
(monotone Δf sequence down to 1e-11), so energy noise is not a concern.
Expected: −2…−4 calls on the creeping carbonyl classes (≈ −0.005 … −0.01),
connected molecules near-neutral (harmonic endgame: θ ≈ 0).

### Change (functions)
Candidate commit `89ccf3494f442540c8781f2f6f15e5845d41e294` (+59/−2 lines):
- `PES.__init__`: `secant_energy_window = 2.0`.
- New `PES._secant_dg(s0, s, y, df, g0, g1)`: θ = 6(f0 − f1) + 3(g0 + g1)·s
  with g0 = transported gradient `g_par`, g1 = new gradient, s = `dx_final`;
  returns y + θ s/(s·s) if `_secant_pairs` is non-empty, ŝ_prev·s₀ > 0
  (s₀ = `dx_initial`, the requested step in the frame of the previous
  pair), and 1/2 ≤ 1 + θ/(s·y) ≤ 2; otherwise y (also for c = 0, non-finite
  values, |s|² < 1e-16). Verified on a synthetic cubic (s·y* equals
  s·G(x1)s exactly; orthogonal components untouched; reversed previous
  step, large θ and empty history return y).
- `PES.kick`: `g1 = self.get_g()` reused; `_update_H(dx_final,
  self._secant_dg(dx_initial, dx_final, dg_actual, df_actual, g_par, g1))`.

### Result
- Train (`evaluation_results/cycle-50-train.json`, evaluation
  `8ed606dee2e34c6b8f563db2c9be4385`): C_S **0.7696717** vs 0.7532674
  (Δ +0.0164, gate failed), mean rel_energy 1.0019465 (was 1.0017390),
  469/469 converged. Decision (`results.tsv`, cycle 50): **discard**.
- Paired vs the champion (`/tmp/paired_cls.py cycle-50 cycle-49`):
  same-basin 457 molecules mean **+0.13** calls (128 better / 137 worse /
  192 identical; dC_S +0.0052, of which |d| ≤ 2 contributes +0.0018);
  basin changes 12, dC_S **+0.0112** (acids__esters 11 → 51 into a basin
  1.6 kJ/mol deeper, amides__ammoniums 27 → 54, 135132759 11 → 26,
  amides__phenol 39 → 75, phenol__water 24 → 47; against
  imidazolium__ketones 65 → 47, carboxylates__water 50 → 35). Energy
  margin Σ(rel−1) 0.913 (was 0.816).
- Same-basin by class: connected +0.17 (54 / 65 / 135 identical; dC_S
  +0.0052), concentrated in the shortest runs — champion calls < 12:
  4 better / 20 worse / 61 identical, mean +0.35; 12–16: 17 / 12; 16–25:
  18 / 20; 25–40: 11 / 10 (+1.04); ≥ 40: 4 / 3 (−2.5). Dimers +0.08
  (74 / 72 / 57; dC_S −0.0000): the creeping classes gained as predicted —
  single carbonyl acceptor −0.79 (5 better / 2 worse, n 14), "≥ 2 H to one
  Y" −1.56 (6 / 1), "≥ 2 H-bonds" −1.29 (7 / 4), S acceptors −0.67 — while
  dispersion-bound dimers lost +0.78 (8 / 18, n 37), two-neighbour O
  acceptors +1.43 (4 / 8), metal contacts +2.67 (0 / 2).

### Interpretation
- The mechanism itself is close to neutral in the same basin (+0.13
  calls) with the predicted sign on the creeping carbonyl / salt-bridge
  classes, so the creep diagnosis is not refuted; but the correction also
  fires on the trending *relaxation* steps of every run, where it costs
  ≈ +0.35 calls in the short connected runs (the cycle-47 signature:
  information imposed by the first large steps stays in the memory-4
  block through the whole endgame of a 9–11-call run) and +0.8 on the
  flat dispersion-bound dimers whose trends end in overshoots. The
  retrospective trend test (s·s_prev > 0) admits the overshoot step that
  closes a trend — precisely the pair for which the interval average is
  the better predictor of the reversal that follows — and it admits the
  second and third relaxation steps of every short run.
- The discard is dominated by basin changes (+0.0112 of +0.0164): a
  curvature change on the early trending steps re-rolls the path lottery
  of the charged dimers (acids__esters into a 1.6 kJ/mol deeper basin at
  +40 calls). That part would re-roll under any variant.
- Uncertainty: with 192 identical runs the gate fires rarely; the
  class-level gains (n 14 / 9 / 14) are within the lottery. The
  same-basin sign of the connected short runs (4 : 20) is the one robust
  signal, and it points at the relaxation-phase corrections, not at the
  end-point-curvature idea as such.

### Next idea / control
Repair 1 (bounded, general rule): restrict the correction to persistent
creep — at least two consecutive same-direction steps before this one
(ŝ_prev·s₀ > 0 and ŝ_prev·ŝ_prev2 > 0) *and* the step still descending
along itself at its end point (g1·s < 0, g0·s < 0, f1 < f0: no overshoot,
so the next step continues along s and the end-point curvature is the
right estimate); window unchanged. This excludes the relaxation steps of
short runs (a three-step trend hardly occurs in a 9-call run) and the
overshoot that closes a trend, and keeps the 5–10-step creeps of the
carbonyl dimers. Champion restored (`git reset --hard d148070`).

## Cycle 51 — cycle-50 repair 1: energy-augmented secant restricted to persistent model-limited creeps (discard)

Champion at start: `d148070066a818071b41626af514c0627710334b` (train 0.7532674,
valid 0.7715672). Candidate: `b2bce1e5d2ff39f00b9cd28c072fc83797651c0c`.

### Hypothesis
The cycle-50 harm sat in the short connected runs (4 better / 20 worse
below 12 calls) and in dispersion-bound dimers, i.e. on relaxation steps
and on the overshoot step that closes a trend, while the creeping
carbonyl / salt-bridge classes gained ≈ 1 call. A creep detector — the
step is the model's own (not cut by the trust region), continues the
previous step, goes downhill and still descends along itself at both end
points (g0·s < 0, g1·s < 0), and the previous step qualified the same way
(two consecutive same-direction model-limited undershoots) — should keep
the end-point-curvature correction on the 5–10-step creeps of the flat
contact modes and remove it from relaxation steps and overshoots. Expected:
connected near-identical, creep classes −1…−2 calls (≈ −0.003 … −0.006).

### Change (functions)
Candidate `b2bce1e5d2ff39f00b9cd28c072fc83797651c0c` (+83/−2 lines on the
champion):
- `PES.__init__`: `secant_energy_window = 2.0`, `secant_creep_min = 2`,
  `_creep_run = 0`, `_step_restricted = False`.
- `PES._secant_dg`: `creep = cont and not _step_restricted and df < 0 and
  g0·s < 0 and g1·s < 0` (cont = ŝ_prev·s₀ > 0 with matching shapes);
  a non-creep step resets `_creep_run` to 0 and returns y; a creep step
  increments it and returns y unless `_creep_run ≥ 2`, c = s·y > 0 and
  1/2 ≤ 1 + θ/c ≤ 2, in which case y + θ s/(s·s), θ = 6(f0−f1) + 3(g0+g1)·s.
- `PES.kick`: as in cycle 50 (`g1` reused, corrected pair passed to
  `_update_H`).
- `Sella.step`: `self.pes._step_restricted = bool(smag >= self.delta)`
  before the kick (`get_s` returns the radius itself as `smag` when the
  step was cut).
- Static checks: py_compile; pyflakes diff vs champion only the
  `dpos_first` line shift; stub check of `_secant_dg`: cubic exactness
  (s·y* = s·G(x1)s to 1e-16, orthogonal components unchanged), counter
  and every gate (restricted, reversed, overshoot, window, empty history,
  shape mismatch) behave as designed; quartic creep θ/s·y = −0.28.

### Result
- Train (`evaluation_results/cycle-51-train.json`, evaluation
  `bc8cc2b69aa44d5fbad02c44abab5241`): C_S **0.7564724** vs 0.7532674
  (Δ +0.0032, gate failed), mean rel_energy 1.0017450 (was 1.0017390),
  469/469 converged. Decision (`results.tsv`, cycle 51): **discard**.
- Paired vs the champion (`/tmp/paired_cls.py cycle-51 cycle-49` and the
  run-length breakdown): 271 runs bit-identical, 198 changed; same-basin
  dC_S **+0.0004** (36 better / 32 worse), one near-degenerate path
  change (ammoniums__esters 27 → 52, −0.12 kJ/mol, +0.0028 — the whole
  discard). Connected −0.05 calls (21 better / 16 worse; < 12 calls
  3 : 1, 25–40 7 : 2 at −0.24, ≥ 40 4 : 2 at −1.9 — paliperidone_palmitate
  56 → 48, 160853090 72 → 65). Dimers +0.18 (15 / 17): the creep classes
  did not gain — single carbonyl acceptor 2 better / 4 worse (+0.40),
  "≥ 2 H to one Y" 0 / 1, "≥ 2 H-bonds" 2 / 0 (−0.33), two-neighbour O
  1 / 0, dispersion 2 / 2, S / sp² N / nY ≥ 3 / metal bit-identical.
  Largest dimer moves: carboxylates__water 50 → 46 against
  ammoniums__benzene 38 → 43, ammoniums__ketones 27 → 31, acids__alkenes
  40 → 44.

### Interpretation
- The repair did what it was designed to do — the short-connected harm
  disappeared (3 : 1 instead of 4 : 20) and the correction fired only
  inside genuine creeps (198 runs contain ≥ 2 consecutive model-limited
  same-direction undershoots, so such creeps are common in both classes)
  — but correcting the curvature along the creep direction to its
  end-point value does not shorten the creeps: dimers 15 : 17, and the
  large changes (−8, −7, +7, +5) have random sign. The creep diagnosis of
  cycle 50 was therefore wrong in its mechanism: the consecutive
  undershoots are not a stale 1-D curvature along one flattening mode
  (which the correction fixes exactly for cubic profiles) but a rotating
  step direction — each step's new component lies in a direction whose
  curvature the model has not yet sampled, so the secant information
  along the previous step (average or end-point) does not change the next
  step. That is the subspace-exploration picture of the connected
  endgame ([[adaptive-type-scale]] showed the per-type extrapolation of
  such errors also fails).
- Alternatives not excluded: (i) the correction pairs, stored with y*,
  shift the consistency test of `_collect_secant_pairs` and drop older
  pairs (a hidden cost that could mask a small gain); (ii) the factor-2
  window rejects the strongly anharmonic contact modes where the
  end-point value would matter most. Neither is worth a third evaluation:
  four variants of the energy-augmented secant (clip, resolution floor,
  trend gate, creep gate) have now covered the design space from "every
  step" to "only persistent creeps" without a systematic same-basin gain
  anywhere; the long-connected 11 : 4 (−0.6 calls, −0.0008) is the only
  hint and it is within the lottery.

### Next idea / control
Energy-secant line closed (backlog status). The creep-detector plumbing
(`_step_restricted`, `_creep_run`) is preserved in
`ideas/energy-secant/b2bce1e5d2ff39f00b9cd28c072fc83797651c0c/algo.py`
for any future remedy that needs to recognise creeps. Champion restored
(`git reset --hard d148070`). Next: return to the tracked-model-term route
(sp² N acceptor wag census; butterfly modes) or the connected subspace-
exploration cost (which directions remain unsampled after the relaxation
steps — coupling between adjacent internal coordinates that the diagonal
guess ignores).

## Cycle 52 — ionic reference radii for the group-1/2 metals in the model-Hessian stiffness formulas (keep)

Champion at start: `d148070066a818071b41626af514c0627710334b` (train 0.7532674,
valid 0.7715672). Candidate: `06e433a3b54220d8920e477ca43ed54ba1fef406`.

### Hypothesis
The Almlöf stretch curvature k = 0.3601 exp(−1.944 (r − r_ref)/Bohr) Ha/Bohr²
is that of a covalent single bond (0.36) at r = r_ref. For the s-block metals
the Cordero "covalent" radius is calibrated on their ionic contacts (Na 1.66
+ O 0.66 = 2.32 Å is the Na⁺···O distance; Li 1.28 + C 0.76 = 2.04 ≈ the
Li⁺···C(π) distance; Mg 1.41 + C 0.76 = 2.17 ≈ the Mg···C distance 2.22 Å of
the train Mg–phenol), so every metal–ligand contact of the tracked pair term
and every metal–ligand "bond" of a connected system (Mg–C ×6, Li–C ×6, Na–S,
K–S, Ca–C ×2) gets 0.1–0.4 Ha/Bohr² (K···O at 2.31 Å in the valid K–ether:
1.47), 5–15× the physical curvature of an ion–dipole / cation–π contact
(0.02–0.07 Ha/Bohr² from the 200–500 cm⁻¹ M⁺···OH₂ stretches; the ion–dipole
model (2n−4)|U|/r² with a Born exponent n = 9 gives 0.029 for Na⁺···OH₂ and
3(n−3)|U|/r² ≈ 0.05 for Na⁺–benzene). Sideways, the cage of radial pair
terms around a cation on a π face (6 × k sin²θ/2 ≈ 0.07 Ha/Bohr² with the
Cordero radii) is ~10× the physical slip curvature, and the tracked term
re-imposes it at every geometry — which is why the cation walkers advance
0.01–0.05 Å per step (Na–pyrrole train 29 calls vs ref 16, Li–phenol valid
34 vs 22). With the Shannon ionic radius (CN 6) as the metal's reference
radius the same formula gives 0.037 (Na⁺···O 2.3 Å), 0.062 (Li⁺···O 1.9),
0.032 (K⁺···O 2.7), 0.026 (Na···C(π) 2.5), 0.020 (Mg···O 2.16) Ha/Bohr² — the
physical range without a new constant; for the cation–π slip 6 × 0.009 ×
0.13 ≈ 0.007. Halide anions are the opposite case (covalent C–X radius;
Cl⁻···H at 2.1 Å is already 0.8 Å outside r_ref → 0.015 Ha/Bohr², right) and
keep the covalent radii. Census (`/tmp/metal52.py`, cycle-49 champion): 6
train / 8 valid molecules contain Li/Na/K/Mg/Ca (share 0.0135 / 0.0176, mean
rel 1.056 / 1.025 vs 0.75 overall). Expected: −0.002…−0.004 per split, every
metal-free molecule bit-identical; risk: a softer model may send a walker to
another basin (train energy margin 0.82 rel-energy units).

### Change (functions)
Candidate `06e433a3b54220d8920e477ca43ed54ba1fef406` (+31/−9 lines on the
champion):
- module table `_STIFFNESS_RADII` (before `class Internals`): copy of
  `ase.data.covalent_radii` with Li 0.76, Na 1.02, K 1.38, Rb 1.52, Cs 1.67,
  Be 0.45, Mg 0.72, Ca 1.00, Sr 1.18, Ba 1.35 Å (Shannon, CN 6).
- used instead of `covalent_radii` in `_h0_bond`, `_h0_angle`,
  `_h0_dihedral`, the torsion-class bond order in `guess_hessian`,
  `_h0_nonlocal_contacts` (pair cutoff r < r_ref + 3 Å and k) and the
  `rcov_all` bond-order factors of `_h0_contact_bends` /
  `_h0_lone_pair_plane`.
- unchanged: `find_all_bonds` (1.25 r_cov connectivity, fragments,
  rebuilds), `_start_fragments` / the symmetry-breaking start.
- Static checks: py_compile; pyflakes diff vs the champion only line shifts;
  the table equals the covalent radii for every other element, so
  metal-free molecules are bit-identical by construction.

### Result
- Train (`evaluation_results/cycle-52-train.json`, evaluation
  `4fdaec24fcef4aa1bf69718a87aae4d5`): C_S **0.7515590** vs 0.7532674
  (Δ −0.0017, gate passed), mean rel_energy 1.000186 (was 1.001739),
  469/469 converged. Exactly the 6 metal molecules changed, 463
  bit-identical: Mg–phenol (connected, 6 Mg–C bonds) 43 → 34, Na–sulfides
  (connected) 18 → 14, Na–pyrrole (cation–π walker) 29 → 23, K–alkenes
  (dimer) 22 → 14, Na–amines (dimer, Na on the N–H side) 11 → 13,
  Na–alkanes (connected, Na···H₃C) 14 → 16; Σ −23 calls, Σ Δrel −0.80.
  The Na–pyrrole walker now ends in the reference's basin (rel_energy
  1.7282 → 1.0000; the champion's endpoint was 10.4 kJ/mol lower, a 0.73
  rel-energy-unit credit that carried almost the whole train energy margin,
  now 0.087 units).
- Valid (`evaluation_results/cycle-52-valid.json`, evaluation
  `59ddfb5869d0416d81a43e82c72cfaed`): C_S **0.7677808** vs 0.7715672
  (Δ −0.0038, gate passed), mean rel_energy 1.008717 (unchanged),
  465/465 converged. Exactly the 8 metal molecules changed, 457
  bit-identical: Li–alkanes (connected, Li–C ×3) 32 → 17, K–sulfides
  (connected) 28 → 21, Li–phenol (cation–π walker) 34 → 16, Li–benzene
  (connected, Li–C ×6) 12 → 9, Ca–alkenes (connected) 10 → 6, Na–pyridine
  12 → 12, K–ethers (K···O 2.31 Å, lottery credit 1.85 kept) 12 → 14,
  K–thiols 19 → 21; Σ −43 calls, all same basin.
- Decision (`results.tsv`, cycle 52): **keep**. New champion
  `06e433a3b54220d8920e477ca43ed54ba1fef406` (train 0.7515590, valid
  0.7677808).

### Interpretation
- The physics statement was right in both coordinate systems: the two
  cation–π walkers (the largest metal costs) fell to the dimer average
  (29 → 23, 34 → 16), and the *connected* metal systems — whose metal
  bonds and angles are not tracked and were assumed to be "learned in the
  first steps" — gained even more (43 → 34, 32 → 17, 28 → 21, 10 → 6):
  with six Mg–C or Li–C "bonds" plus their 15 angles at 0.2–0.3 Ha/Bohr²
  and 0.24 Ha/rad², the ion's three degrees of freedom were spread over
  ~20 stiff redundant coordinates, and a diagonal secant correction along
  the first step does not undo a coupled overestimate of that size (the
  cycle-36 lesson holds for a 1.5× error on one bond, not for a 10× error
  on a cage). 14 of 14 metal systems changed, 11 better / 3 worse (+2
  each); the two-call losses are weak Na···H–C / K···S contacts where the
  softened model (0.005 Ha/Bohr² per pair) may now sit below the
  ion–induced-dipole curvature, i.e. within the ordinary path-change noise.
- The train energy margin was almost entirely one lottery credit
  (Na–pyrrole, 0.73 of 0.82 units); the softer model walks it into the
  reference's basin instead (23 calls, 10.4 kJ/mol higher), leaving
  0.087 units of train margin. Future candidates that change dimer paths
  therefore run a real risk of the `energy_below_baseline` verdict on
  train; nothing to be done about it except noting that a train-invalid
  result with a good valid result may be this margin rather than the idea.
- Alternatives not excluded: the gain might partly come from the shrunken
  pair cutoff (fewer distant metal···H pairs), but their k was < 1e-3
  Ha/Bohr² and the effect is negligible against the 10× change of the
  near contacts.

### Next idea / control
Metal line done (backlog `metal-ionic-radii` accepted; no constants to
scan — the table is the Shannon CN-6 set). The three +2 cases are not
worth a repair. Next: the remaining physical model errors of the same
kind — contacts whose Almlöf reference is miscalibrated (none obvious for
the non-metal elements: halide anions checked) — or the sp² N acceptor
wag / butterfly-mode census.

## Cycle 53 — geometry-tracked stretch diagonal and transport of the secant pairs with the analytic part of the model, connected systems (keep)

Champion at start: `06e433a3b54220d8920e477ca43ed54ba1fef406` (train 0.7515590,
valid 0.7677808). Candidate: `f38a8b6e79bc9856c9a87272d4b4e903ddfe2d7d`.

### Hypothesis
Two facts from the saved results and the algebra of the update. (i) The
multi-secant TS-BFGS update gives B⁺S = Ỹ exactly (UᵀS = I in `_MS_TS_BFGS`),
so a shift of the model along a *sampled* direction is always overridden by
the stored pair — and a pair is a path average over its segment. This is why
tracking the torsional diagonal (cycle 46) was a wash while tracking the
off-diagonal contact terms (cycles 35, 43, 45, 48, 49) paid: only the
unsampled part of the tracked model survives the update. (ii) Bond
anharmonicity is a measurable cost: on the cycle-52 champion (connected
same-endpoint, `/tmp/bondanh53.py`) the largest bond-length change costs
+7.6 (se 1.8) train / +14.0 (se 2.2) valid calls per Å on top of the
n20/maxdisp/N baseline (lengthening +7.4/+13.7, shortening n.s.), +1.0/+0.8
calls per bond changing by > 0.1 Å; small stiff inorganic molecules (3–5
atoms, no torsions) still take 7–10 calls at maxdisp 0.1–0.3 Å. After a
0.1 Å bond step the path-average (secant) curvature of a Morse bond lags the
local curvature by 25–35 % (Morse: 0.52 k_e at +0.1 Å, 1.76 k_e at −0.1 Å),
≈ one extra endgame step per anharmonic chain. The Fischer–Almlöf stretch
exponential k(r) = 0.3601 exp(−1.944 (r − r_ref)/Bohr) is a local
curvature–length relation that matches Badger's rule (d ln k/dr = −3/(r −
d_ij) ≈ −3.3…−4.3/Å against Bb = 3.7/Å; Morse would give 3a ≈ 5–6/Å), i.e. a
physically justified, conservative transport of the bond curvature with the
geometry — provided it also enters the *pairs*, not only the prior. Mechanism:
with the analytic part T(x) of the model (projected stretch diagonal plus the
contact block A(x)), move the model H ← H + T(x) − T(x_prev) before each
update as before, and transport every stored pair to the current point,
y_j ← y_j + (T(x) − ½(T(x_j) + T(x_{j+1}))) s_j, so that the secant conditions
describe the local curvature at x. Connected systems only, so that the
multi-fragment paths (thin train energy margin +0.087) stay bit-identical.
Expected −0.5…−1 call on molecules with bonds changing ≥ 0.1 Å (≈ 40–65 per
split), C_S −0.002…−0.005.

### Change (functions)
Candidate `f38a8b6e79bc9856c9a87272d4b4e903ddfe2d7d` (+120/−5 lines on the
champion):
- `Internals._h0_stretch_diagonal` (new): the Almlöf `_h0_bond` guesses of
  all bonds at the current geometry as a vector over the internal
  coordinates (translations first, as in `guess_hessian`).
- `PES._update_H(dx, dg, T_now=None, tbar=None)` and
  `PES._collect_secant_pairs(dx, dg, T_now=None, tbar=None)`: the pair store
  holds `(s, y, tbar)` with tbar = T_avg s / |dx| (None when no analytic
  model); before the dependence and consistency tests every pair with a
  tbar is transported, `y = y + (T_now @ s − tbar)`; the single-pair branch
  shifts dg the same way. Without T_now/tbar the code path is the old one.
- `InternalPES.__init__`: `self._transport = self._track_nb and ntrans +
  nrotations == 0`; `self._an_prev = self._analytic_model()` at x0.
- `InternalPES._analytic_model` (new): T = Q (Qᵀ diag(d) Q) Qᵀ + A(x) with d
  from `_h0_stretch_diagonal`, Q from `_get_jacobian_qr`, A from
  `_h0_nonlocal_contacts(Binv=self._get_Binv())`; None on any failure.
- `InternalPES._track_analytic_model(dx)` (new): H ← H + T(x) − T(x_prev),
  returns (T(x), ½(T(x_prev) + T(x)) dx).
- `InternalPES._update_H`: connected systems take the new path, otherwise
  the unchanged `_track_nonlocal_contacts()` + `PES._update_H`.
- Static checks: py_compile; pyflakes diff vs the champion empty;
  `/tmp/stub53.py` execs the real `_update_H`/`_collect_secant_pairs`
  source and confirms that trapezoid-averaged pairs of a model T(x) + C are
  transported to (T(x) + C) s exactly (1e-16), that the old path is
  unchanged without T, and that Q (Qᵀ D Q) Qᵀ = P D P.

### Result
- Train (`evaluation_results/cycle-53-train.json`, evaluation
  `170657c16c6c4979a5d77261a6bdcaab`): C_S **0.7446790** vs 0.7515590
  (Δ −0.0069, gate passed), mean rel_energy 1.000339 (was 1.000186), 469/469
  converged. All 212 multi-fragment systems bit-identical. Connected 257:
  92 better / 44 worse, mean −0.28 calls (same-basin −0.37); 5 basin changes
  (+0.0012: 315516336 7 → 21 now in the reference's basin −0.34 kJ/mol,
  363892164 21 → 35 to a basin 0.01 kJ/mol lower, monoatomics__sulfides
  14 → 16 to a basin 1.4 kJ/mol lower, 135093970 22 → 17 to a basin 2.1
  kJ/mol lower, 135095518 28 → 25 same energy).
- Valid (`evaluation_results/cycle-53-valid.json`, evaluation
  `26dda62539be4686ac3dbeb2896dc83a`): C_S **0.7482450** vs 0.7677808
  (Δ −0.0195, gate passed), mean rel_energy 1.008749 (was 1.008717),
  465/465 converged. All 203 multi-fragment systems bit-identical. Connected
  262: 113 better / 48 worse, mean −1.10 calls (same-basin −0.95); 5 basin
  changes (−0.0020), among them amides__imidazoles 37 → 21 and 104129283
  42 → 29 to basins 0.06–0.27 kJ/mol higher, 135100404 30 → 31 into the
  reference's basin (29 kJ/mol lower).
- Paired bins (`/tmp/pairbond53.py`, same-endpoint): by reference length
  ref < 12 −0.21 / −0.18, 12–16 −0.07 / −0.18, 16–30 −0.22 / −0.53, ≥ 30
  **−1.28 (se 0.69) / −2.65 (se 0.65)**; by heavy atoms < 6 −0.50 (18:5) /
  −0.62 (9:3), 6–12 −0.33 / −0.38, 12–20 +0.15 / −0.52, ≥ 20 −0.71 (32:7) /
  −1.43 (57:20); by largest bond change maxdb < 0.05 Å −0.07 / −0.67,
  0.05–0.12 −0.27 / −0.88, 0.12–0.2 −0.31 / −1.30, ≥ 0.2 −1.71 (se 0.93) /
  −1.05 (se 1.16); n10 ≥ 4 −2.33 / −2.83 (n 12 / 6).
- Decision (`results.tsv`, cycle 53): **keep**. New champion
  `f38a8b6e79bc9856c9a87272d4b4e903ddfe2d7d` (train 0.7446790, valid
  0.7482450).

### Interpretation
- The mechanism pays, and more broadly than the bond-anharmonicity census
  predicted. The predicted signature is there (small stiff molecules
  −0.5…−0.6 calls with 18:5 / 9:3 better:worse — molecules with no
  non-local contacts, so this is the stretch transport alone; maxdb ≥ 0.2 Å
  −1.7 on train; n10 ≥ 4 −2.3 / −2.8), but the largest gains are in long
  runs of large molecules (ref ≥ 30: −1.3 / −2.7 calls; ≥ 20 heavy atoms
  −0.7 / −1.4), where the bond changes are ordinary. There the transported
  part is mostly the contact block A(x): a folding chain changes its contact
  curvature 2–6× along the path, the champion moved the model but the update
  re-imposed the stale segment averages along every sampled direction (the
  directions the run actually moves in), and the transported pairs now carry
  the current contact curvature there. That the valid gain (−0.0195) is
  three times the train gain is consistent with the split composition
  (valid has 127 connected molecules with ≥ 20 heavy atoms and 60 with ref
  ≥ 30 against 83 / 43 on train); the per-molecule effects agree in sign
  across splits in every bin.
- Alternatives / uncertainty: the split between the stretch transport and
  the contact transport is inferred from the class pattern, not measured
  (one evaluation per cycle); a control that transports only A(x) would
  settle it but is not needed for the decision. Ten basin changes are the
  usual path-lottery noise (+0.0012 / −0.0020); the train energy margin
  rose to +0.159 by lottery, the valid one is unchanged (+4.07).
- The result also revises the cycle-47 memory lesson: pairs older than ~4
  steps were "stale" partly because of the analytic (predictable) part of
  the curvature change; with transport the useful memory may be longer.

### Next idea / control
The same transport for multi-fragment systems: the contact block of a dimer
changes most during the approach (k grows 2–6× as the H-bond forms) and the
walkers creep along the contact modes; the pairs measured during the
approach are the stale ones. Risk: the train energy margin (+0.159 now;
any dimer path change re-rolls the lottery). Second candidate: revisit the
secant memory (5–6) for connected systems now that the pairs are
transported (cycle 47's "stale beyond 4" was measured without transport).

## Cycle 54 — composite Simpson path average of the analytic model for the secant transport (non_generalizable)

Champion at start: `f38a8b6e79bc9856c9a87272d4b4e903ddfe2d7d` (train 0.7446790, valid 0.7482450).

### Hypothesis
The cycle-53 transport moves every stored pair by (T(x) − T_avg,j) s_j with T_avg,j
the trapezoid average ½(T(x_prev) + T(x)) of the analytic model over the step. The
terms of T are exponentials of distances (k ∝ exp(−3.7/Å · r)); for a contact or a
bond that opens by 0.5–1 Å in one step (early torsional steps of a folding chain, a
dissociating bond) the trapezoid overstates the exact path average by 0.1–0.25 k_prev
— more than the local k(x) itself for b·Δr ≥ 3 — leaving the transported pair too soft
or of the wrong sign along the step. Simpson's rule with T evaluated at the Cartesian
midpoint (composite, panels of ≤ 0.4 Å atomic displacement, at most four) is within
0.1 % of k_prev of the exact average for any step size (`/tmp/stub54.py`, run on the
real `_analytic_model_average` / `_track_analytic_model` source extracted from
algo.py: composite error ≤ 9e-4 k_prev for steps up to 2.5 Å vs trapezoid errors
0.09–0.39 k_prev for steps ≥ 0.4 Å). Expected: a small same-basin gain on the
molecules with large early steps (n10 ≥ 2, maxdb ≥ 0.2 Å, long folding runs) and no
change elsewhere; multi-fragment systems bit-identical (no transport there).

### Change (functions)
Candidate `394e459e6ab8eadc8bd60afce6f8a6d861796560`.
- `Internals.shadow_copy()`: a copy with its own `Atoms` (copied, calculator removed)
  and dummies, sharing the coordinate lists, `forbidden`, `_active` and the covalent
  graph (derived from the bonds list, so the non-local pair set is identical); its
  positions can be set freely and no force call can be triggered through it.
- `InternalPES.__init__`: `_pos_prev` (atom and dummy positions at the last model
  evaluation) and a lazily created `_shadow`.
- `InternalPES._analytic_model_from(int_obj, Q, Binv)` (static): T = Q (Qᵀ diag(d) Q) Qᵀ
  + Hnb for any Internals object; `_analytic_model()` now delegates to it with the
  cached Q/Binv (bitwise the cycle-53 T(x)). `_model_frame(B)` (static) reproduces the
  QR / SVD-truncation logic of `_get_jacobian_qr` / `_get_Binv` for the shadow Jacobian.
- `InternalPES._analytic_model_at(pos_prev, dpos_prev, frac)`: T on the Cartesian
  straight line at fraction `frac` of the step, via the shadow copy.
- `InternalPES._analytic_model_average(T_prev, T_now, pos_prev, dpos_prev)`: composite
  Simpson with `npanel = min(4, ceil(dmax / 0.4 Å))`, dmax the largest atomic
  displacement of the step; endpoint average if any interior point is unavailable.
- `InternalPES._track_analytic_model(dx)`: `tbar = T_avg @ dx` with the new average
  (was ½(T_prev + T_now)); `_pos_prev` refreshed together with `_an_prev`.
Multi-fragment systems: `_transport` is False, path unchanged.

### Result
- Train (`evaluation_results/cycle-54-train.json`, evaluation
  `39b185855ca644f8a472cda437aa1a53`): C_S **0.7420590** vs 0.7446790 (Δ −0.0026,
  gate passed), mean rel_energy 1.000292 (was 1.000339), 469/469 converged. All 212
  multi-fragment systems bit-identical. Connected 257: 30 better / 31 worse, mean
  −0.15 calls (same-basin −0.075, dC_S −0.0016); 2 basin changes (−0.0010): the
  bistable anion 104079126 44 → 25 now ends in the reference basin (22.7 kJ/mol above
  the champion's endpoint), 135039840 13 → 13.
- Valid (`evaluation_results/cycle-54-valid.json`, evaluation
  `154f265cd1fb42b28f2385e2e07c5810`): C_S **0.7481526** vs 0.7482450 (Δ −0.0001, gate
  1e-4 not passed), mean rel_energy 1.008769 (was 1.008749), 465/465 converged. All
  203 multi-fragment systems bit-identical. Connected 262: 40 better / 43 worse, mean
  +0.11 calls (same-basin −0.035, dC_S −0.0019); 3 basin changes (+0.0018), all three
  losing a deeper basin for the reference's: 125084169 26 → 37, 104129283 29 → 40,
  amides__imidazoles 21 → 36.
- Paired bins (`/tmp/pairbond53.py cycle-54 cycle-53 6`, same-endpoint): train n10
  [2,4) −0.47 (se 0.30, 8:6), maxdb ≥ 0.2 Å −0.41 (se 0.50), ref ≥ 30 −0.31 (se 0.49,
  15:16), everything else within ±0.1; valid n10 [1,2) −0.51 (se 0.24, 22:9), maxdb
  [0.12, 0.2) −0.54 (se 0.57, 15:7), n10 = 0 +0.11 (se 0.07, 11:21), ≥ 20 heavy atoms
  +0.03 (17:24). Predicted single cases: des370k_acids__alcohols (dissociating bond,
  maxdb 0.84 Å) 29 → 25; 135065494 (maxdb 1.53 Å) 55 → 63; 384221749 46 → 55 and
  252638725 36 → 19 are ±9-call path changes without a bond signature.
- Decision: **non_generalizable** (`scripts/cycle.py record`); champion stays
  `f38a8b6e79bc9856c9a87272d4b4e903ddfe2d7d`. Implementation preserved at
  `ideas/simpson-transport-average/394e459e6ab8eadc8bd60afce6f8a6d861796560/algo.py`.

### Interpretation
The quadrature error of the trapezoid transport is real but small in connected
molecules: the same-basin effect is −0.0016 / −0.0019 on the two splits (≈ −0.05
calls per molecule, sign counts even), the predicted classes move by −0.4 to −0.5
calls with standard errors of the same size, and the split-level numbers are decided
by the basin lottery (train −0.0010 from one lucky bistable anion, valid +0.0018 from
three deeper-basin runs lost). Why so little: the large steps (≥ 0.4 Å) happen in the
first two or three steps only, their pairs leave the memory-4 window within four
updates, and after that b·Δr ≤ 1 for every bond and contact, where trapezoid and
Simpson differ by < 3 % of k. The stale-average error that cycle 53 removed was the
first-order one (the local value vs the segment average, proportional to the whole
change of T along the path, accumulated over all pairs); the quadrature error is the
second-order remainder on the few large early segments. Alternatives not excluded:
the Cartesian straight-line path compresses rotating bonds at the chord midpoint
(harmless while the bond components of s vanish for torsional steps) and could
counteract part of the gain on molecules with combined stretch + torsion steps; the
dimer approach phase (0.5–1 Å steps routinely) is where the quadrature would matter,
but the champion has no transport there yet.

### Next idea / control
Restore the champion (`git reset --hard f38a8b6`). The transport line for
multi-fragment systems (contact block + stretch diagonal during the approach) remains
the largest known unexplored mechanism, but every dimer path change re-rolls the
train energy margin (+0.159, carried by carboxylates__ethers +0.356 and
ketones__monoatomics +0.233 against alkanes__carboxylates −0.327, pyridine__water
−0.277, ketones__ketones −0.227), so it is a ~50 % lottery on validity. Before it,
look for a connected-system lever with a census: the short-reference connected runs
(ref < 16, rel 0.85–0.90, share 0.19 train) and the linear endgame rate.

## Cycle 55 — secant-pair transport with the analytic model for multi-fragment systems (keep)
Champion at start: `f38a8b6e79bc9856c9a87272d4b4e903ddfe2d7d` (train 0.7447, valid 0.7482).
Candidate: `5d0a542040b30ac0f066135252c6bc80f13d072f` — new champion (train 0.7194, valid 0.7271).

### Hypothesis
Cycle 53 transported the stored secant pairs with the analytic part of the model
(y_j ← y_j + (T(x) − T_avg,j) s_j, T = P diag(d_Almlöf) P + A(x)) for connected systems
only; multi-fragment systems kept the older model shift (H ← H + A(x) − A(x_prev),
`_track_nonlocal_contacts`) without transport. Since the multi-secant TS-BFGS re-imposes
B s_j = y_j exactly along the sampled directions (`/tmp/stub53.py`), and every approach
step samples the rigid-body/contact directions, the dimer endgame model kept the
approach-phase averages of the contact curvature (2–6× softer than at the formed contact)
and bounced. Expected: fewer endgame calls on same-basin dimers, monotone in the approach
distance; the cost of connected systems bit-identical to the cycle-54 candidate (composite
Simpson average for all systems). Known risk: any dimer path change re-rolls the train
energy margin (+0.159 at the champion).

### Change (functions)
- `Internals.shadow_copy` / `Internals.sync_rotation_state` (new): a calculator-free copy
  of the coordinate set with independent `Rotation` objects (same indices, axis and
  bit-identical `refpos`; their `q_prev` quaternion-branch state is copied from the real
  objects before each use, so evaluating the shadow never changes the real object's
  branch sequence).
- `InternalPES.__init__`: `_transport = bool(self._track_nb)` (was: connected only);
  `_pos_prev`, `_shadow` initialised for every transported system.
- `InternalPES._analytic_model_at` (new): model T at an interior point of the Cartesian
  straight-line step, evaluated on the shadow (QR/SVD frame `_model_frame`, contact block
  and stretch diagonal `_analytic_model_from`).
- `InternalPES._analytic_model_average` (new, from cycle 54): composite Simpson average
  over the step with `ceil(dmax / 0.4 Å)` panels (≤ 4), endpoint average as fallback.
- `InternalPES._track_analytic_model`: uses the path average for T_avg; docstring covers
  the multi-fragment case. `InternalPES._update_H`: transport branch for all systems with
  the analytic model, `_track_nonlocal_contacts` only when the transport is off.
- Static checks: py_compile, pyflakes (no new messages), `/tmp/stub53.py` (transport
  algebra) and `/tmp/stub54.py` (composite Simpson accuracy, panel counts, fallbacks).

### Result
- Train 0.7194 (Δ −0.0253), mean rel energy 1.00157, 469/469 converged
  (`evaluation_results/cycle-55-train.json`, evaluation b6fb8eb0f1594e8d82e2bc85b5479c30).
  Valid 0.7271 (Δ −0.0212), mean rel energy 1.00825, 465/465 converged
  (`evaluation_results/cycle-55-valid.json`, evaluation 5de5a7b57d544968941d670ccc15ebc9).
  Decision **keep** (`scripts/cycle.py record`, both gates passed).
- Decomposition (`/tmp/pairdimer55.py cycle-55 cycle-53 8`, locally classified
  connected/multi-fragment): train connected −0.0030 (254 of 257 runs bit-identical to
  cycle 54; the three others are des370k pairs that Sella treats as fragments), dimer
  −0.0223 = same-basin −0.0222 + basin changes −0.0001; valid connected −0.0007, dimer
  −0.0205 = same-basin −0.0159 + basin changes −0.0046.
- Same-basin dimers (196 / 183 same-endpoint): −1.62 / −1.85 calls per dimer, 134:44 and
  127:45 better:worse. By approach distance (rmsd start → champion end): < 0.2 Å −0.93 /
  −0.30; 0.2–0.4 Å −0.61 / −0.16; 0.4–0.8 Å −1.33 / −1.54; 0.8–1.5 Å −2.69 / −2.30;
  > 1.5 Å −2.67 / −4.91. Charged −1.04 / −1.33, neutral −1.93 / −2.13. Reference bins:
  ref ≥ 24 uniformly −1.5 to −3.8; ref < 16 −0.96 (train, 11:3) but +0.32 (valid, 8:9).
- Basin changes: train 16 (carboxylates__water 50 → 13 into a basin 12 kJ/mol above the
  reference endpoint, credit −0.206; monoatomics__pyrrole 23 → 33 into one 11 kJ/mol below,
  credit +0.788; guanidiniums__pyrrole 13 → 35 and benzene__guanidiniums 11 → 27 now reach
  the reference basin instead of a shallower one); valid 20 (alkanes__guanidiniums 37 → 14
  shallower, −0.155; guanidiniums__ketones 32 → 14 lost a 5 kJ/mol deeper basin, −0.129;
  amides__benzene 46 → 34, ethers__pyridine 23 → 30).
- Energy margins of the new champion: train +0.735 = +2.19 − 1.46 (monoatomics__pyrrole
  +0.788, carboxylates__ethers +0.356, ketones__monoatomics +0.233, carboxylates__ketones
  +0.199 against alkanes__carboxylates −0.327, pyridine__water −0.277, ketones__ketones
  −0.227, carboxylates__water −0.206); valid +3.836 (carboxylates__ketones +3.216,
  ethers__monoatomics +0.852 against ethers__water −0.414, esters__water −0.411,
  amides__water −0.347).
- New cost map (train / valid): connected rel 0.799 / 0.761, share 0.438 / 0.429; dimer
  rel 0.622 / 0.683, share 0.281 / 0.298; dimer ref < 16 rel 1.02 / 1.45 (share 0.050 /
  0.084) is the only class above the reference; dimer ref ≥ 24 rel 0.35–0.61.

### Interpretation
The mechanism worked as predicted and is the largest dimer gain of the transport line:
the gain grows monotonically with the approach distance (i.e. with how much the contact
block changed along the sampled directions), is present for charged and neutral pairs, and
the connected part is the known cycle-54 remainder. The composite Simpson average was the
enabling piece for the 0.5–1 Å approach steps (cycle 54 showed it is nearly irrelevant for
connected molecules). Uncertainties: (i) the train validity was carried by one deeper
basin (monoatomics__pyrrole +0.788) — the intrinsic train margin without it is −0.05, so
the next dimer path change is again a lottery on train validity; (ii) the short-reference
dimers did not move (train −0.96 on 14 runs, valid +0.32 on 17): their cost is decided by
the first steps (slide-past into deeper basins, or charged pairs whose first steps
overshoot), which the endgame transport cannot touch; (iii) the transport is not carried
across a rebuild (`check_for_bad_internals` creates a fresh InternalPES with empty pairs),
which mostly affects dimers that form a new contact mid-run. Alternative explanation for
part of the gain — the model shift is now applied on top of transported pairs, so the
contact block is effectively tracked twice as fast during the approach — is the intended
behaviour (B⁺S = Ỹ made the shift ineffective along sampled directions before).

### Next idea / control
With the transport in place for all systems, re-test the secant memory (5–6 pairs): the
cycle-5/6 result (8 pairs hurt) was obtained with stale pairs, and transported pairs
should age better. Alternatively, a census of the first five steps of the short-reference
dimers with the new endpoints. The train energy margin is thin without the single credit;
a connected-only change would leave the dimer credits untouched.

## Cycle 56 — multi-secant memory 8 for all systems with transported pairs (discard)
Champion at start: `5d0a542040b30ac0f066135252c6bc80f13d072f` (train 0.7194, valid 0.7271).
Candidate: `e332641c7bb3ac695a7e1a9ebc7de5da3971b32d`.

### Hypothesis
Cycle 47 (memory 8, connected only, no transport) lost +0.44 calls per molecule, with
the loss concentrated in short runs where the first large relaxation steps stayed in
the block through the endgame with their averaged bond/contact curvature. The secant
transport of cycles 53/55 moves exactly that part of a stored pair to the current
geometry (composite Simpson average for the large early segments), so the marginal
value of pairs older than four should now be non-negative; for two-fragment systems
the six rigid-body modes exceed a memory of four, so a longer memory should keep the
endgame model exact along all of them. One constant: `PES.secant_memory = 8`.

### Change (functions)
- `PES.__init__`: `secant_memory = 8` (was 4), comment on the transported-pair
  rationale. No other change.

### Result
- Train 0.7384 (Δ +0.0190), mean rel energy 1.00169, 469/469 converged
  (`evaluation_results/cycle-56-train.json`, evaluation ee622eeac737434fab0dee38b5c11e2f).
  Decision **discard** (train improvement below 1e-4; valid not run). Champion restored
  (`git reset --hard 5d0a542`).
- Connected (`/tmp/mem47.py train cycle-56 cycle-55`): +0.0035, 38 better / 51 worse /
  168 identical (+0.18 calls each; cycle 47: +0.44). Loss is spread thinly: nrot ≥ 8
  +0.5…+0.7 each, N ≥ 45 +0.85 each, champion runs of 9–12 calls +0.22 (1:9); flexible
  mid-size molecules (N 30–45) slightly better (−0.33 each, 13:9).
- Multi-fragment (`/tmp/pairdimer55.py cycle-56 cycle-55`): +0.0155, 43 better / 97
  worse, +0.83 calls per same-basin dimer; by approach distance: < 0.2 Å +0.07 (3:4),
  0.2–0.4 Å +0.42, 0.4–0.8 Å +1.13 (8:43), 0.8–1.5 Å +0.30, > 1.5 Å +1.84 (7:15);
  charged and neutral alike (+0.82 / +0.84). Only 2 endpoint changes (energy credits
  +0.723 vs +0.667 — the train margin survived, monoatomics__pyrrole unchanged).

### Interpretation
The transport removed most of the connected-system harm of a long memory (+0.44 →
+0.18 calls) but not all of it — the remaining staleness is in the non-analytic part
(torsional anharmonicity, bends), as the flexible/large molecules lose most. For the
dimers a longer memory is clearly harmful even with transported pairs, and worst at
moderate approach distances: the pairs taken during the approach are stale beyond the
analytic contact block — the exponential pair term carries no long-range
electrostatic/dispersion curvature (which changes sign between 3 and 5 Å), and the
fragment rotation coordinates are defined relative to a reference orientation, so a
pair's s-vector from a substantially different orientation is misread in the current
frame; the pairwise consistency test (tolerance 0.25) does not catch either. The
marginal value of a stored pair is therefore negative beyond age 4 in both classes; the
memory line is closed for good. Alternative not excluded: the near-dependent columns
of an 8-pair block (dependency tolerance 0.3) amplify small asymmetries in
`symmetrize_Y2`, which would also predict harm growing with the number of soft modes;
the connected pattern (largest molecules lose most) is compatible with both readings.

### Next idea / control
Not a memory change. Remaining levers from the cost map of the new champion: the
short-reference dimers (decided in the first steps), the C=O-acceptor single-H-bond
class (+6 same-basin calls above the H-bond baseline on 9/11 dimers), and the generic
connected endgame (rel 0.75–0.80 in every reference bin).

## Cycle 57 — predictor-corrector step under the geometry-following analytic model, connected systems (discard)
Champion at start: `5d0a542040b30ac0f066135252c6bc80f13d072f` (train 0.7194, valid 0.7271).
Candidate: `ed2664b53496ba09f754803e49ef5f796d06d9b8`.

### Hypothesis
Since cycle 53 the model Hessian of connected systems follows the geometry, H(x) = H +
T(x) − T(x_k) (T = Almlöf stretch diagonal with the exponential curvature–length
relation, plus the non-local contact block), and the secant pairs are transported with
it, so H is the model curvature *at* x_k. The quasi-Newton step −H⁻¹g nevertheless
assumes that curvature over the whole step, although T says how it changes along the
way. The starting geometries of the sets have their bonds systematically 0.05–0.07 Å
too long (bond census of cycle 55: C–C shorten:lengthen 101:26 at |dr| > 0.06 Å, C–N
62:10, C–O 24:2): a bond 0.06 Å too long has 0.80 of its equilibrium curvature
(b = 3.67/Å), so the local quadratic step overshoots the model's own minimum by 12 %
of the step (0.007 Å, four times the displacement gate), and a contact that closes
during a step is met with the curvature it had before it. The step of the
path-dependent model solves g + [∫₀¹ H(x_k + t s) dt] s = 0, i.e. s = −(H + dT_g(s))⁻¹g
with dT_g = ∫₀¹[T(x(t)) − T(x_k)]dt along the linearised Cartesian path x + t·Binv·s;
it is found by fixed-point iteration from the quadratic step (sensitivity
≈ b|dr|/2 = 0.1–0.4 for bond steps of 0.06–0.2 Å, contractive; a second correction is
kept only if the iteration contracts, so a step into a contact wall keeps the
single-correction, Heun-type estimate). The trust ratio of the step is judged against
the same model's energy prediction, f(x + s) − f(x) = g·s + ½ sᵀ(H + dT_e)s with
dT_e = 2∫₀¹(1 − t)[T(x(t)) − T(x_k)]dt. Expected −0.2 to −0.5 calls per connected
molecule (C_S −0.003 to −0.008); multi-fragment systems bit-identical.

### Change (functions)
- `InternalPES._analytic_model_at_positions(pos, dpos)`: T on the shadow internals at
  arbitrary positions (factored out of `_analytic_model_at`, which now delegates to it
  with the interpolated positions — the transport path is unchanged).
- `InternalPES._analytic_model_ahead(T0, s)`: (dT_g, dT_e) by the same composite
  Simpson rule as `_analytic_model_average` (panels ≤ 0.4 Å of atomic displacement,
  at most four; weights checked exact for quadratic T on a panel, `/tmp/simpson57_check.py`).
- `Sella._restricted_step_with(B, rs_kwargs)`: the restricted step (same radius, stepper,
  Cartesian bound) under a substitute model matrix; the `ApproximateHessian` fields
  (`B`, eigen cache, `initialized`) are restored afterwards.
- `Sella._curved_step(s, smag, rs_kwargs)`: the fixed-point iteration (≤ 2 corrections),
  called from `_predict_step` after the quadratic step (no-inequality branch), guarded
  to minimisation, `curved_step_mol`, `pes._transport`, no TR internals; sets
  `pes._B_pred = H + dT_e` for the kick.
- `PES.__init__`/`PES.kick`: `_B_pred` (consumed by the kick's `df_pred`, reset in
  `_predict_step`).
- `_default_kwargs['minimum']['curved_step_mol'] = True`, `Sella.curved_step_mol`.

### Result
- Train 0.71929 (Δ −0.0001), mean rel energy 1.00148 (margin +0.69, was +0.74), 469/469
  converged, 56 s (was 35 s) (`evaluation_results/cycle-57-train.json`, evaluation
  e444d166623d4640be4e1c96a4e84cd2). Decision **discard** (train improvement below
  1e-4; valid not run). Champion restored (`git reset --hard 5d0a542`); implementation
  preserved in `ideas/curved-step-analytic-model/ed2664b5…/algo.py`.
- Paired (`/tmp/pair57.py cycle-57 cycle-55`): multi-fragment 212/212 bit-identical.
  Connected: 58 better / 51 worse / 148 identical, same-basin −0.12 ± 0.12 calls per
  molecule (−0.0012), three endpoint changes net +0.0011 (104079126 25 → 47 into a basin
  22.7 kJ/mol deeper, +0.022 credit; monoatomics__sulfides 16 → 14 and 135093970 18 → 19
  into the reference basin). By largest bond change: 0.03–0.05 Å −0.21 (12:6), 0.05–0.08
  −0.13 (17:15), 0.08–0.12 −0.09 (12:13), 0.12–0.2 −0.52 (11:7), ≥ 0.2 +0.90 (5:8). By
  largest lengthening: 0.03–0.06 Å −0.44 (25:12), ≥ 0.1 Å +0.71 (7:8). By champion run
  length: 6–9 calls +0.21 (2:6), 9–12 +0.16 (7:12), 12–16 −0.03, 16–25 −0.22 (21:12),
  ≥ 25 −0.88 (15:10). By start displacement: < 0.7 Å ≈ 0, ≥ 0.7 Å −0.45 (23:14). By
  size: N < 8 +0.20 (5:11), 8–12 +0.53, 12–20 −0.39, ≥ 45 −0.45. The N < 12 losers are
  hydrogen-free/exotic-bond molecules (B–H, N–S, O–P, C–Cl, C–I, Br–S, B–Br, S₄, B–F);
  largest gains are folding chains (135142801 40 → 29, acalabrutinib 25 → 15, 384221749
  55 → 48, 336270104 20 → 15 at rmsd0 0.85–1.8 Å); largest same-basin losses
  ketones__phenol 34 → 45 (maxdb 0.69 Å, an H-transfer path) and 135231613 34 → 40 (B–H).

### Interpretation
The hypothesis' main term is real but not binding: the organic shorten-by-0.06 Å
class (maxdb 0.05–0.12 Å, n 148) is a wash (−0.1 each, 29:28), because the 12 %
residual of a bond step (0.007 Å) is removed by the following step anyway, while the
soft modes still need 5–15 steps — the endgame of a connected molecule is set by the
soft-mode model quality (cycle 37/39 memory), not by the stretch anharmonicity. The
signal that exists is in the long, folding runs (contact block changing along the
step; ≥ 16 calls −0.2 to −0.9 each), and the losses are where T's geometry dependence
is least like the xTB bond (exotic single bonds of the small hydrogen-free molecules,
bonds that lengthen ≥ 0.1 Å): the correction applies the model's exponential a second
time (transport and step), so a wrong exponent now costs twice. Alternatives not
excluded: (i) the trust ratio judged with dT_e changes the radius sequence of the
short runs (their +0.2 could be a radius effect rather than a step-direction effect);
(ii) the second correction's contraction test is blind to a non-contracting *first*
correction (a step into a contact wall keeps the single correction, which can overshoot
the wall as much as the quadratic step undershoots it). Uncertainty: same-basin
−0.12 ± 0.12 — indistinguishable from zero; the one adverse basin change decides the
sign of the split total.

### Next idea / control
Not the stretch part. The contact-only correction (long folding runs) is bounded by
≈ −0.002 per split from this run — only as a component of a bundle. The multi-fragment
approach phase (0.5–1 Å steps into a contact wall whose curvature changes ~6× along
the step) is the one place where the step-averaged curvature differs qualitatively
from the local one; that extension carries the train energy-gate lottery (margin +0.74
with −0.05 without the monoatomics__pyrrole credit) and is the candidate for a later
cycle, not the next. Next: a different lever of the connected endgame (soft-mode model
quality along unsampled directions) or the short-reference dimers (decided in the first
3–5 steps).

## Cycle 58 — per-rotor secant learning of the torsional stiffness (connected systems) (discard)
Champion at start: `5d0a542040b30ac0f066135252c6bc80f13d072f` (train 0.7193841, valid 0.7270871).
Candidate: `50710f5f24e0facec71ab0fb309b185b1d2787a6`; decision **discard** (train 0.7435097, Δ +0.0241;
valid not run). Evidence: `evaluation_results/cycle-58-train.json` (evaluation
92e8fcb5254946659653314ed24ed27c, 469/469 converged, mean rel energy 1.00151). Implementation preserved in
`ideas/rotor-secant-stiffness/50710f5f24e0facec71ab0fb309b185b1d2787a6/algo.py`; backlog
`## rotor-secant-stiffness`.

### Hypothesis
The census of the cycle-55 endpoints (`/tmp/phase58.py`, `/tmp/phase58b.py`) killed the class-phase torsional
model (far-starting rotors end within the class minimum only 23–32 % of the time, rotate 26–28° on average to
unpredictable destinations), so the per-rotor stiffness cannot be predicted structurally beyond the class factors;
but it can be *measured*: the rigid rotation about an acyclic bond changes every dihedral of the bond by the same
angle and nothing else, its internal displacement is the indicator vector u of the bond's dihedrals (in the range
of the Jacobian), the torque is u·g, the rotation of a step u·dx/n and the torque change u·dg, so u·dg/dphi is the
secant torsional curvature — one fact per moving rotor per force call that the multi-secant update, which learns
the curvature along the whole step, structurally cannot extract for a soft component of a stiff-dominated step.
Expected: −0.005 to −0.02 per split if the rotor model error drives the mid-phase undershoot and the endgame rate;
rotor-free molecules and all multi-fragment systems bit-identical.

### Change (functions)
`Internals.guess_hessian`: records `_rotor_groups` (dihedral-index groups per acyclic bond of real atoms, ring
bonds up to eight atoms excluded via `_in_small_ring`) and `_h0_diag_last`. `InternalPES.__init__`: for connected
systems builds `_rotors` / `_rotor_k0` (guess rotational stiffness K0 = Σ|h0| over the group). New
`InternalPES._learn_rotor_stiffness(dx, dg, phi_min=0.05, k_lo=0.2, k_hi=5.0)`: for each rotor with |dphi| ≥
phi_min sets uᵀHu to clip(u·dg/dphi, 0.2 K0, 5 K0) by the rank-one block shift H[idx, idx] += (K_new − K_cur)/n²
(wag directions and all other coordinates untouched; pairs not transported), then `set_B`.
`InternalPES._update_H`: calls it after `_track_analytic_model` and before `PES._update_H`. Static checks:
py_compile, pyflakes identical to the baseline, numpy stub of the block shift (`/tmp/stub58.py`).

### Result
Train 0.74351 (Δ +0.0241), mean rel energy 1.00151 (margin +0.71, was +0.74). Paired (`/tmp/pair58.py cycle-58
cycle-55`, `/tmp/sum58.py`): controls hold — 212/212 multi-fragment and 68/68 rotor-free connected molecules
bit-identical, runs of < 8 calls identical. Connected 42 better / 88 worse / 84 identical; same-basin +1.18 ± 0.24
calls per molecule (+0.0208 of the split total), six endpoint changes +0.0034 (104079126 25 → 67 into a basin
49 kJ/mol deeper, credit +0.048; 104046121 25 → 42; 135186987 23 → 12; 135316214 30 → 40; 252089162 39 → 43).
By number of acyclic rotors: 1 +0.19 (3:5), 2 −0.33 (4:8), 3–4 +0.62 (15:14), 5–7 +2.22 (12:23), ≥ 8 +3.98
(8:38). By far-starting rotors: 0 far +0.13, 1 far +0.06, 2 far +0.48, ≥ 3 far +3.62 (17:53). By largest rotation
of the champion path: 3–10° +0.77 (0:6), 10–30° +1.14 (15:27), 30–60° +2.02 (15:25), ≥ 60° +2.95 (12:29). By
champion run length: 8–12 +0.23, 12–16 +1.01, 16–24 +1.03, 24–40 +4.72, ≥ 40 +9.33. Largest losses: 104079126
25 → 67 (basin change), 384221749 55 → 89 (17 rotors), 135267488 28 → 45 (20 rotors), 104046121 25 → 42,
paliperidone_palmitate 50 → 65 (22 rotors), 8030412 23 → 38. Wall time 45 s (champion 35 s).

### Interpretation
Refuted, and monotonically in the number of rotors and in the run length — the signature of a measurement that is
wrong more often than right, not of a wrong constant. Two intrinsic mechanisms: (i) for a soft rotor (K ≈ 0.02–
0.1 eV/rad²) the torque change of its own rotation, K·dphi ≈ 0.003 eV/rad at the 0.05–0.1 rad of a mid-phase
step, is of the size of the torque changes that the bends, stretches and the other rotors relaxing in the same
step induce through the 1–4/1–5 couplings; the docstring's premise that the couplings are small against K·dphi
holds for stiff coordinates only, so the small-rotation readings are noise (3–10° bin 0:6) and the noise grows
with the number of rotors moving together (≥ 8 rotors +4 calls). (ii) Over a large rotation the secant of a
cosine-type well is always below the curvature at the minimum (|sin a − sin b| ≤ |a − b|); a rotor that has
stopped turning is never re-measured, and the multi-secant update — by the hypothesis' own argument — barely
corrects a soft component, so the too-soft large-span value is frozen into the endgame, where it produces
oscillation instead of contraction (long runs +4.7 to +9.3). The hypothesis' premise was also half wrong: the
multi-secant update does learn the rotors where it matters, in the endgame, when the soft modes dominate the
steps and the pairs are nearly pure rotations. Alternatives not excluded but not worth a candidate: a
stiffen-only rail or a small-rotation window — both refuted by the same table (small rotations lose, and the
loss scales with simultaneously moving rotors, which no rail addresses); the residual form uᵀ(y − Hs)/dphi is
identical for a model without torsion–torsion couplings. This is the adaptive-type-scale lesson (cycles 30/31) in
a purer form: per-coordinate secant extraction from a full-molecule step is ill-posed; the multi-secant update
with its dependency and consistency filters is the right consumer of the pairs. No repair candidates
(uniformly negative, no diagnostics bundles — all runs converged).

### Next idea / control
The rotor model can only be improved structurally (the guess) or through the update's use of the pairs, not by
per-coordinate measurements — closed. Remaining levers from the cycle-58 census: the short-reference dimers
(decided in the first 3–5 steps, `/tmp/shortdimer58.py`), the ion-pair endgame (needs the cycle-45 constants,
not allowed), and the connected mid-phase (far rotors cost +1.64 calls per far bond but their destinations are
unpredictable — a trust-region/step-shape question rather than a model question).

## Cycle 59 — retrospective trust-radius shrink test for connected systems (discard)

Champion at start: `5d0a542040b30ac0f066135252c6bc80f13d072f` (train 0.7193841, valid 0.7270871). Candidate:
`d4a0cfde39ae85e0b945854754d31b24b4cfd44d`.

### Hypothesis
The cycle-55/58 census leaves "one badly modelled soft mode learned per step" as the main connected-system cost. A
step dominated by a mode whose true curvature is m > 2 times the model's fails Sella's test (ρ = 2 − m < 0.01) and
the ×0.5 shrink then caps the next step at half the failed step's largest component — but the multi-secant update
has just imposed the true curvature along that step, so the next model step is already the corrected one
(|1 − m|/m of the overshoot, > 0.5 for every m > 2) and the cap only truncates it (+1 call per such failure, more
when the cap binds through the LM damping of all modes). Retrospective trust-region idea (Bastin, Malmedy, Mouffe,
Toint, Tomanos, Math. Program. 123, 395 (2010)): judge the step again with the *updated* model. With the newest
secant pair exact (symmetrize_Y2 keeps column 0), the updated model's prediction over the step is the trapezoid
of the two gradients, g·s + ½ s·y, which equals the actual change on any quadratic surface, whatever the old model
was; the retrospective ratio ρ_post = ΔE/(g·s + ½ s·y) deviates from 1 only through anharmonicity, which no update
repairs. Rule: after a failed step, halve the radius only if ρ_post also lies outside Sella's good-agreement window
(1/rho_inc, rho_inc) = (0.75, 1.33); otherwise keep it (no growth either). Restricted to systems without fragment
TR internals in this cycle (multi-fragment paths bit-identical; train validity hinges on the pyrrole credit). Cubic
analysis before the run: for a Newton-like step with curvature ratio m and cubic anharmonicity ε = c s/k,
ρ_post = (½ − 1/m + ε)/(½ − 1/m + 1.5 ε) — hypersensitive to ε near m = 2 (→ 2/3 for any ε ≠ 0), tolerant
(|ε| ≲ 0.1) for m ≥ 3; wall (exponential) steps of one decay length give 0.92, cosine-rotor steps of 0.2/0.4/0.5
rad 1.03/1.14/1.24, overshoots past the minimum 1.24–1.66. Boundary: the rule removes shrinks and never adds
them, so it cannot make steps artificially small; Sella's trigger ρ < 0.01 / ρ > 100 is unchanged.

### Change (functions)
- `PES.get_df_post(dx, g, dg)` (trapezoid g·dx + ½ dx·dg) and the `InternalPES` override projected with
  `Unred` like `get_df_pred`; `PES.kick` stores `self.ratio_post = df_actual / df_post` (None when |df_post| <
  1e-14 or non-finite) from `dx_final`, the transported old gradient `g_par` and `dg_actual`; `PES.__init__`
  initialises `ratio_post`.
- `_default_kwargs['minimum']['retro_shrink'] = True`; `Sella.__init__` reads it.
- `Sella._retro_exempt()`: True iff minimisation, `retro_shrink`, no TR internals and 1/rho_inc < ρ_post < rho_inc.
- `Sella.step`: the shrink branch (ρ outside (0.01, 100)) is skipped when `_retro_exempt()`; growth branches
  untouched.

### Result
Train 0.7215782 (Δ +0.0022), mean rel energy 1.001634 (margin +0.77 incl. one new credit) → **discard** (train
improvement below 1e-4). Evidence `evaluation_results/cycle-59-train.json` (9762f55cbaab805a6d19d3b794ee8e76,
evaluation ee22d8e16cfb4ab793ad7d54fa4c1471). Paired (`/tmp/pair58.py cycle-59 cycle-55`): 212/212 multi-fragment
runs bit-identical (control holds); connected 257: 224 same call count (193 bit-identical), 14 better (−19 calls),
19 worse (+74), net +55 calls (+0.21 ± 0.10 per molecule); one endpoint change (252657041 25 → 34, 3.2 kJ/mol
deeper, credit +0.031). Runs of < 8 calls all identical; 8–12 calls −0.03, 12–16 +0.01, 16–24 +0.02, 24–40
+1.28 (3:8), ≥ 40 +2.33 (1:3). By acyclic rotors: 0 ±0, 1 −0.06, 2 −0.19 (3:0), 3–4 −0.05 (3:1), 5–7 +0.37 (4:4),
≥ 8 +0.78 (2:12); by far-starting rotors ≥ 3: +0.72 (4:16); by largest rotation of the champion path ≥ 60°:
+0.84 (4:11), < 60° ≤ +0.09. Largest losses: 134997543 32 → 51 (7 rotors, 107° rotation), 384221749 55 → 63,
paliperidone_palmitate 50 → 55, 135086807 29 → 34, 135100494 25 → 30. Wall time 39 s (champion 35 s).

### Interpretation
- The endgame case the rule was built for — a failed step whose energy change the secant model explains, followed
  by a truncated corrected step — is rare or cheap: in the runs of 8–24 calls (207 molecules) the exemption changed
  nothing net (−0.03…+0.02 calls, 10 better vs 8 worse), and the medium rotor bins gained at most −0.19 calls. By
  the cubic analysis this is expected: the common endgame failure is a moderate overshoot (m ≈ 2–2.5) whose
  quadratic energy change nearly cancels, so ρ_post ≈ 2/3 falls outside the window for any anharmonicity, while for
  m ≈ 2 the cap (0.5 of the failed step) equals the needed correction and costs nothing.
- The exemptions that did fire sit in the mid-phase of long, rotor-rich runs (≥ 8 rotors, ≥ 3 far-starting rotors,
  ≥ 60° rotations): 0.2–0.5 rad rotor steps on cosine surfaces are quadratic-consistent to within the window (ρ_post
  1.03–1.24 by the cosine estimate) but their secant curvature is the chord of the well, systematically below the
  curvature at the minimum (cycle 58's observation), so the "repaired" model still overshoots and the kept radius
  lets it: the ×0.5 shrink of cycle 21 was protecting exactly these steps (cycle 20's growth-only tail reappears in
  miniature: 134997543 +19, 384221749 +8, 24–40-call runs +1.28). The premise "the update has imposed the true
  curvature" holds for quadratic modes only, and those failures do not dominate the cost.
- Alternatives not excluded: a tighter window (e.g. ±5 %) would exclude the mid-phase cosine steps and keep the
  quadratic endgame exemptions, but the endgame gain is at the noise floor (≤ −0.001 of the score), so a repair
  cannot pay more than that; a step-size or curvature-change qualifier would need trajectories to design and
  none exist (all runs converged, no diagnostics bundles). Not extended to multi-fragment systems: cycle 22 showed
  their flat, anharmonic surfaces benefit from the fast shrink, the same regime that lost here.

### Next idea / control
No repair candidate (the bins where the mechanism should pay are neutral; the loss is a re-creation of the
growth-only tail). Trust-region line closed for connected systems in both directions: tightening the shrink is the
prohibited endgame damping (cycle 22 note), loosening it by model consistency loses the anharmonic mid-phase
protection. Remaining connected levers are model-structural (the rotor guess along a path, bend–bend coupling
across sp³ centres) or step-shape (the far-rotor mid-phase); the dimer levers stay behind the energy-gate
lottery.

## Cycle 60 — initial trust radius 0.25 Å/rad for multi-fragment systems (discard)

Champion at start: `5d0a542040b30ac0f066135252c6bc80f13d072f` (train 0.7193841, valid 0.7270871). Candidate:
`af12ce643644e02c49891dbd2dc93071ca5d02ef`.

### Hypothesis
The last untested control of the cycle-28 trust-region line. Connected systems start at δ₀ = 0.25 (cycle 28),
multi-fragment systems still at Sella's 0.1 with the ×1.5 growth ramp 0.1 → 0.15 → 0.225 → 0.25 (cap δ_max_tr).
Since cycle 46/55 the fragment translation/rotation block carries the tracked contact curvature and the contact
bends, so the first rigid-body steps of a near-equilibrium or mid-range start (0.15–1 Å / rad of remaining relative
motion) are directed by the model and limited only by the radius; the ramp should then cost one or two accepted
steps per dimer before the radius reaches the size of the remaining motion. Expected −0.005…−0.01 with the usual
multi-fragment lottery (P(train invalid) ≈ 20 %); cycle 44 (cap 0.5) is not a counter-argument because it changed
the mid-run radius of far starts, not the first steps.

### Change (functions)
- `_default_kwargs['minimum']`: comment of `delta0_mol` extended to multi-fragment systems (value 0.25 unchanged).
- `Sella.__init__`: `delta0 = default['delta0_mol']` for every `order == 0` run, dropping the
  `not self._has_tr_internals()` exclusion. Connected systems bit-identical by construction.

### Result
Train 0.7203612 vs 0.7193841 (Δ +0.0010), mean rel energy 1.00208 (valid, margin +0.97), 469/469 converged, 36.6 s.
Decision **discard** (train improvement below 1e-4); valid not run. Paired breakdown (`/tmp/pair60.py`,
`evaluation_results/cycle-60-train.json` vs `cycle-55-train.json`):
- connected 257/257 identical; multi-fragment 212: mean −0.06 calls, 102 better / 71 worse / 39 identical;
  same-basin (192) −0.40 calls = −0.0047 of C_S; basin changes +0.0057 (credits +0.667 → +0.906).
- same-basin by start class: near-equilibrium (tr < 1 Å, rot < 15°, 41) −0.17; mid-range (1–2.5 Å, 105) −0.52; far
  (46) −0.33; rotations ≥ 45° (47) −0.28; charged (53) −0.32, neutral (137) −0.42; champion runs of < 12 calls (36)
  +0.36, 22–35 calls (53) −1.13.
- basin changes: des370k_acids__esters 9 → 38 (ref 14; the candidate ends 0.38 kcal/mol below the reference,
  rel_e 1.073; start tr 0.47 Å / 5°), esters__guanidiniums 11 → 31 (ref 17; −0.20 kcal/mol, rel_e 1.022; tr 0.91 Å
  / 9°), alkanes__esters 38 → 76 (ref 96), benzene__thiols 22 → 32; favourable: ketones__ketones 21 → 32 (ref 43)
  now reaches the reference basin (champion was 2.72 kcal/mol short, rel_e 0.773 → 1.000), pyridine__pyridine
  26 → 18, benzene__imidazolium 47 → 32, ammoniums__esters 48 → 29, amides__ammoniums 25 → 19 but 1.40 kcal/mol
  short (rel_e 1.005 → 0.927). Largest same-basin moves: esters__pyrrole 40 → 24, monoatomics__pyrrole 33 → 25
  (kept its +0.79 credit), amines__water 29 → 40, acids__imidazolium 17 → 26, imidazolium__pyridine 22 → 28.

### Interpretation
- The mechanism exists but is small: same-basin −0.40 calls per multi-fragment run, and least where it was
  predicted (near-equilibrium −0.17). With the contact block directing the first step, a near-equilibrium start
  needs a first step of ~0.1–0.3 Å/rad; at δ₀ = 0.1 the ramp reaches 0.225 by the third accepted step, so at most one
  step is truncated, and the shortest runs (< 12 calls) even lost +0.36 — their 0.1 first step was already the
  right size and the 0.25 step overshoots the soft contact modes (ρ failure, ×0.5 shrink, one extra step). The
  mid-range class gains most (−0.52) because there the ramp binds for two steps.
- The score rose through the energy lottery of exactly the class the change targets: two near-equilibrium
  H-bonded starts left the reference basin with the larger first step (proton/acceptor re-pairing to minima
  0.2–0.4 kcal/mol deeper) and paid +29 / +20 calls (+0.0069 of C_S), more than the whole same-basin gain. A larger
  first step from a near-equilibrium start is a basin re-roll by construction, so the control cannot be separated
  from the lottery on one split, and its same-basin value (−0.005) is at the lottery amplitude (±0.005 per split).
- Alternatives not excluded: the truncation cost may be hidden in runs whose first steps fail for other reasons
  (SCC noise, contact bends too stiff), where the radius is halved anyway; no trajectories exist to tell (all runs
  converged, no bundles). An intermediate δ₀ (0.15–0.2) would be a parameter scan and is not pursued.

### Next idea / control
None on the radius: δ₀, growth, shrink and caps are now scanned or controlled for both system classes; the
memory note `trust-region-connected-systems` is updated (multi-fragment δ₀ line closed). The multi-fragment
cost that remains is model quality on the soft contact modes (short-reference ion pairs, shared-proton
contacts 1.3–1.4 Å, where the model is ~4× too stiff) and the lottery; the connected-system cost is the rotor
mid-phase. The next cycle should be model-structural rather than step-control.

## Cycle 61 — donor X–H stretch softening after the ν(XH)–d(H···Y) correlation (discard)

Champion at start: `5d0a542040b30ac0f066135252c6bc80f13d072f` (train 0.7193841, valid 0.7270871). Candidate:
`f4fd1efa1f715f94ffc5600745a7f8df1e9465a0`.

### Hypothesis
The idea search after cycle 60 (`/tmp/basin61.py`, `/tmp/ionres61.py`, `/tmp/sharedH61.py` on the champion's
train/valid endpoints) located the largest same-basin residuals of the multi-fragment class in the ionic
short-hydrogen-bond pairs, and among them the shared-proton complexes (N–H⁺···N/O⁻ and O–H···O⁻ with d(H···Y)
1.34–1.51 Å and X–H 1.07–1.24 Å at the endpoint: ammoniums/imidazolium/guanidiniums + pyridine/amines/carboxylates,
acids + carboxylates), which we finish 3–7 calls slower than the reference (ammoniums__pyridine 25 vs 19 train,
18 vs 11 valid; imidazolium__pyridine 22 vs 19; ammoniums__pyrrole 23 vs 17 valid). Along the proton coordinate
the tracked model has k_XH + k_HY = 0.36 exp(−3.67 Δr) + pair term ≈ 0.42 Ha/Bohr² at the endpoint of such a pair,
while the Novak/Libowitzky ν(OH)–d(H···O) correlation (ν = 3632 − 1.79·10⁶ exp(−d/0.2146 Å) cm⁻¹, Libowitzky 1999)
puts the X–H curvature at 0.03–0.15 Ha/Bohr² there: the bond is being transferred, not stretched, and the
bond-length exponential captures only the first 20–40 % of the softening. Expressed with the excess
e = d(H···Y) − r_H − r_Y, the correlation's curvature ratio f(e) = (1 − 5.37 exp(−e/0.2146))² agrees with the
tracked exponential at e ≈ 0.7 Å (d(H···O) 1.67 Å, r(OH) ≈ 1.0, both ≈ 0.34 Ha/Bohr²) and falls to 0.30 (vs 0.75)
at 1.5 Å and 0.08 (vs 0.63) at 1.4 Å; scaling the donor X–H stretch by f(e_min)/f(0.7) (floor 0.05, the ratio of
the lowest observed X–H stretches ≈ 800 cm⁻¹) should make the endgame steps along the proton coordinate 2–4× longer
where it still moves 0.1–0.2 Å, saving 1–3 calls in ~10 shared-proton pairs per split (pot ≈ 0.003–0.005) with
little lottery exposure (the factor acts in contacts that are already formed).

### Change (functions)
- `Internals._donor_stretch_factors` (new): per-bond factors; donors = hydrogens with a single covalent neighbour
  of the acceptor element set (N, O, F, S, Cl, Br, I); acceptors = the same elements among the non-local partners
  of `_nonlocal_pairs(5)` (other fragment, or same fragment at graph distance ≥ 5); e from `_STIFFNESS_RADII`;
  factor clip(f(e_min)/f(0.7), 0.05, 1) for e_min < 0.7 Å, else 1; None when nothing is affected.
- `Internals.guess_hessian` and `Internals._h0_stretch_diagonal`: the bond guess `_h0_bond(bond)` is multiplied
  by the factor, so the guess and the geometry-tracked/transported stretch diagonal stay consistent (the factor
  moves with the geometry through `_track_analytic_model` and the Simpson path average, via the shadow internals).
- Static checks: pyflakes diff empty; numpy stub check of the extracted function (N–H···N at e 0.40 → 0.05, e 0.54
  → 0.51, O–H···O at e 0.83 → None, C–H donor → None, bifurcated donor takes the closest acceptor).

### Result
Train 0.7200242 vs 0.7193841 (Δ +0.0006), mean rel energy 1.00158 (valid), 469/469 converged, 36.4 s. Decision
**discard** (train improvement below 1e-4); valid not run. Paired breakdown (`/tmp/pair61.py`, `/tmp/pair61b.py`,
`evaluation_results/cycle-61-train.json` vs `cycle-55-train.json`):
- connected 255/257 identical (−0.0002, two intramolecular short contacts: 384221749 55 → 47, 172113611 42 → 43);
  multi-fragment 20 better / 24 worse / 5 basin changes (net −0.0006 from the basin changes: ammoniums__esters
  48 → 15 stays in the reference basin where the champion had found one 0.19 kcal/mol deeper, amides__carboxylates
  15 → 35 to a basin 0.34 kcal/mol deeper, amides__ammoniums 25 → 36; same-basin +0.0013).
- The 51 affected molecules (e_min < 0.7 at the start or the champion endpoint) by where the short contact occurs:
  (a) short only at the start — compressed non-equilibrium contacts — 16 same-basin: +0.56 calls (4 better / 8
  worse; pyrrole__water 14 → 18, ethers__phenol 9 → 14, alcohols__pyridine 17 → 19, amides__sulfides 10 → 12,
  guanidiniums__sulfides 12 → 14); (b) short only at the endpoint — contacts formed during the run — 26 same-basin:
  +0.19 (10/10); of these the true shared protons (endpoint X–H ≥ 1.07 Å, 13 molecules): 0.00 on average
  (amines__imidazolium 23 → 20, carboxylates__guanidiniums 14 → 10, acids__ammoniums 15 → 12, carboxylates__pyrrole
  37 → 30, ammoniums__pyridine 25 → 24; acids__carboxylates 16 → 19, amides__imidazolium 21 → 25, ammoniums__ethers
  13 → 17, guanidiniums__pyrrole 35 → 45); (c) halide-anion acceptors (F⁻ +4, Cl⁻ +6) and the 0.7–1.0 group
  (no factor at the endpoints, 55 molecules) +0.04.
- By acceptor element (same-basin endpoint contacts): O −0.73 (15), N +0.75 (8), S +0.67 (3), F/Cl +4/+2.5.

### Interpretation
- The target mechanism did not materialise: the shared-proton endgames are not systematically shorter with a 2–6×
  softer X–H stretch (0.00 mean, ±3–4 scatter). The secant transport (cycles 53/55) learns the curvature along the
  proton coordinate within one or two steps, so the initial stiffness error costs nothing systematic; what those
  endgames pay for is the anharmonic (flat or double-well) surface along the proton coordinate, on which a quadratic
  model with any curvature keeps mispredicting the energy change (ρ failures, radius halving) — a curvature factor
  cannot fix that.
- The side effects are real and explain the small loss: (a) an equilibrium correlation applied to a compressed
  non-equilibrium contact softens a bond that is not weakened (the wall is already in the pair term), so the first
  X–H steps overshoot; (c) the halide anions' stiffness radius is the C–X covalent one, so e = 0.6 there is an
  ordinary N–H···Cl⁻ contact at 1.94 Å, not a shared proton; the pyrrole nitrogen counts as an acceptor although
  N–H···N(pyrrole) is a π contact.
- Alternatives not excluded: an elongation-gated variant (factor only when r(X–H) − r_cov > 0.03 Å too, halides
  excluded) would remove (a) and (c), but since (b) is null the ceiling is ≈ −0.001 and it is not pursued as a
  repair. A non-quadratic line model along X–H···Y (Morse or double well) is the only formulation that addresses
  the actual limitation, and it would be a stepper change, not a Hessian factor.

### Next idea / control
The stiffness line for short hydrogen bonds is closed (backlog `donor-stretch-softening`). Remaining
multi-fragment same-basin cost: the far-field walking (cap-limited, closed), the shared-proton anharmonic
endgames (needs a non-quadratic step, untested), and the rotor mid-phase of connected systems. Next cycle:
a different model-structural or stepper idea; the champion stays `5d0a542`.

## Cycle 62 — X–H···π contact bends for heteroatom donors between fragments (keep)

Champion at start: `5d0a542040b30ac0f066135252c6bc80f13d072f` (train 0.7193841, valid 0.7270871). Candidate:
`09bb577f62847bdc1fd519479f5456eb1b224841` — **new champion** (train 0.7179724, valid 0.7256903).

### Hypothesis
The idea search after cycle 61 (censuses `/tmp/ring5_62.py`, `/tmp/near62.py`, `/tmp/pi62.py`, `/tmp/pitouch62.py`
and analytic re-derivations, all on the champion's saved endpoints) closed the remaining soft-mode items — charged-dimer
far-field curvature (steps trust-limited, α ≫ λ), chord/cosine torsion curvature (same), shared protons (n 4/4),
saturated-ring pucker (share ≤ 0.003), inertia-scaled rotation floor (within 2× of physical), sp² N wag (covered by the
bend at H), butterfly modes of doubly H-bonded dimers (no missing mode), shrink relative to δ, force-field
pre-relaxation, and the valid near-equilibrium dimer excess (credit walkers) — and left one physical mode class the
contact model does not hold: the sideways motions of X–H···π contacts. A heteroatom donor pointing its hydrogen at a
double bond or an aromatic face sits 2.0–2.8 Å from the nearest unsaturated carbons, all inclined 15–30° from the
common normal, so the radial pair terms to those carbons give the donor's tilt and the face's sliding almost no
curvature and those librations (100–300 cm⁻¹, k ≈ 0.003–0.016 Ha/rad²) stay at the 0.001 Ha floor of the fragment
coordinates. Giving the π carbons the two kinds of bend that heteroatom acceptors already have (cycle 45), with the weak
cap (0.01) per carbon, should shorten the endgames of the 25 train / 24 valid dimers with such a contact (share 0.032 /
0.045) by 1–3 calls each where the libration is the last mis-modelled mode; expected −0.001…−0.003 per split, with the
usual basin lottery on top (train credits at stake +0.063 of +0.735, the monoatomics__pyrrole credit untouched).

### Change (functions)
- `Internals._h0_contact_bends`: a hydrogen of a heteroatom donor whose closest non-hydrogen atom of another fragment
  (by r − r_cov among the non-local pairs) is a carbon with two or three covalent neighbours is a π donor towards that
  fragment (`pi_donor` set keyed by (hydrogen, fragment label)); every such carbon of that fragment within range
  (bo ≥ bo_min) then gets the isotropic bend at the hydrogen (X–H···C, A_φ,h = 0.5 per donor arm) and the true-angle
  acceptor bends H···C–Z (A_φ,acc = 0.15 per arm) with the contact factor c = min(bo, bo_cap_weak). A hydrogen whose
  closest partner is a heteroatom keeps the heteroatom terms only. The acceptor-heteroatom branch, the lone-pair-plane
  term, connected molecules and dimers without a π donor are bit-identical to the champion.
- `_h0_nonlocal_contacts` docstring updated. Static checks: pyflakes diff empty, py_compile OK; numpy stub of the
  selection rule on the start geometries (`/tmp/stub62.py`): 16 train / 14 valid dimers carry π bends at the start,
  Σc per dimer 0.0003–0.079, hydrogen-bonded dimers untouched.

### Result
Train 0.7179724 vs 0.7193841 (Δ −0.0014), mean rel energy 1.00150 (valid), 469/469 converged, 39.5 s; valid
0.7256903 vs 0.7270871 (Δ −0.0014), mean rel energy 1.00782, 465/465 converged. Decision **keep** (train ≥ 1e-4,
valid > 1e-4, both valid). Paired breakdown (`/tmp/pair61.py cycle-62 cycle-55`, `/tmp/piface62.py`):
- Connected 257/262 bit-identical; every hydrogen-bond class (e < 1.0 Å) identical except three ±1 changes. All
  effects are in the 25 / 24 dimers with a π-face contact at the start or the endpoint.
- By the size of the π face touched (number of unsaturated carbons of the partner within range):
  alkene faces (≤ 2 carbons): train 10 dimers, all same basin, −0.80 calls (7 better, 2 worse: alkenes__ammoniums
  24 → 20, alkenes__thiols 33 → 30, alkenes__phenol 19 → 16, alkenes__water 20 → 17, alkenes__guanidiniums 19 → 17,
  alkenes__pyrrole 19 → 17; alcohols__alkenes 13 → 16, acids__alkenes 29 → 35), dscore −0.0010; valid 8 same-basin
  −1.12 calls (−0.0012: alkenes__phenol 27 → 22, alkenes__ammoniums 25 → 20, acids__alkenes 16 → 14) plus two basin
  changes (alkenes__amides 31 → 11 into a basin 0.97 kcal/mol higher, alkenes__thiols 34 → 26; −0.0021).
  Aromatic faces (≥ 3 carbons): train 13 same-basin +1.46 calls (+0.0005: imidazolium__phenol 25 → 33, benzene__water
  37 → 43, amides__benzene 23 → 28, benzene__pyrrole 11 → 15; alcohols__benzene 25 → 23, imidazolium__pyrrole 17 → 15)
  and two basin changes that supply the train gain (benzene__imidazolium 47 → 29 same energy, ammoniums__benzene
  37 → 21 giving up a 0.033 credit); valid 9 same-basin +1.89 calls (+0.0007: benzene__pyrrole 23 → 32,
  pyrrole__water 27 → 31, alcohols__benzene 25 → 28, acids__benzene 36 → 39; benzene__water 43 → 41) and five basin
  changes (benzene__thiols 26 → 17, benzene__imidazolium 28 → 23, amides__benzene 34 → 39, benzene__guanidiniums
  18 → 28, ammoniums__benzene 32 → 33).
- Net same-basin effect −0.0003 (train) / −0.0005 (valid); the rest of the −0.0014 per split is basin lottery
  (train −0.0011, valid −0.0010). Energy margins: train Σ(rel_e − 1) +0.735 → +0.703, valid +3.836 → +3.636.

### Interpretation
- The alkene faces behave as designed: with two carbons at c ≤ 0.01 the libration stiffness at the hydrogen is
  0.5·Σc ≈ 0.005–0.01 Ha/rad² (cations at the cap, neutrals at bo ≈ 0.005), the sliding stiffness from the three arms
  per carbon 0.15·Σc·3 ≈ 0.003–0.009 — both within the physical range estimated from I·ω² (cation libration
  0.006–0.016, neutral 0.003–0.005) — and they finish about one call earlier on both splits.
- The aromatic faces are over-stiffened: six carbons at the same bo give three times the alkene sum (cation contacts
  Σc 0.04–0.06 → k_H 0.02–0.03, neutral 0.012–0.018 → 0.006–0.009), 2–3× the physical libration and sliding stiffness
  of a cation–π or O–H···π contact (benzene binds a donor only 1.3–2× more strongly than an alkene), and those dimers
  finish 1.5–1.9 calls later; their share of the gain came from basin changes.
- Alternatives not excluded: the same-basin sums are small (13 and 9 dimers) and single dimers move by ±5, so the
  aromatic penalty is suggestive (consistent sign on both splits, 8 worse / 4 better) rather than established; the
  low-Σc regressions (benzene__water train, benzene__pyrrole valid) may be path changes rather than stiffness effects.

### Next idea / control
Bounded follow-up (cycle 63): normalise the π-face contact factors so that one (hydrogen, fragment) face contributes at
most a double bond's worth of bending curvature — scale the c_i of a face by min(1, 2·max_i c_i / Σ_i c_i). Alkene
faces, hydrogen-bonded contacts and connected molecules stay bit-identical; only the ≥ 3-carbon faces (15 train / 14
valid dimers) change. No constant is introduced; the "two carbons" equivalence is the alkene calibration that cycle 62
confirmed. Champion is now `09bb577`.

## Cycle 63 — π-face normalisation of the X–H···π contact bends (keep, marginal)

Champion at start: `09bb577f62847bdc1fd519479f5456eb1b224841` (train 0.7179724, valid 0.7256903). Candidate:
`a09dbb5d39a63bd0fdb4b4bdb176812d8af2c023` — **new champion** (train 0.7177764, valid 0.7252031).

### Hypothesis
Cycle 62's read-out by face size: alkene faces (two carbons, Σc ≤ 0.02 → libration stiffness at H ≤ 0.01 Ha/rad²,
sliding 0.15·Σc·3 ≤ 0.009 — both in the physical range from I·ω²) finished 0.8–1.1 calls earlier on both splits,
aromatic faces (six carbons at the same distance, three times the sum → 2–3× the physical stiffness; benzene binds a
donor only 1.3–2× more strongly than a double bond) finished 1.5–1.9 calls later. Scaling the contact factors of one
(hydrogen, fragment) face by min(1, 2·max_i c_i / Σ_i c_i) — a face is worth at most a double bond — should remove the
aromatic over-stiffening without touching anything else; expected −0.0005…−0.001 per split (15 / 14 aromatic-face
dimers, share 0.03 / 0.04), no constant introduced.

### Change (functions)
- `Internals._h0_contact_bends`: after the `pi_donor` set, a `face` dict collects min(bo, bo_cap_weak) over the
  unsaturated carbons in range (bo ≥ bo_min) per (hydrogen, fragment label) key; `pi_scale[key] = min(1, 2·max/sum)`;
  the π branch multiplies its contact factor by the scale (`scale` = 1 in the heteroatom branch). Docstring updated.
- Static checks: pyflakes diff empty; numpy stub (`/tmp/stub63.py`) on the start geometries: alkene faces scale 1.00,
  aromatic faces 0.33–0.93 (edge contacts where one carbon dominates stay near 1).

### Result
Train 0.7177764 vs 0.7179724 (Δ −0.0002), mean rel energy 1.00150 (identical), 469/469 converged; valid 0.7252031 vs
0.7256903 (Δ −0.0005), mean rel energy 1.00782 (identical), 465/465. Decision **keep** (train ≥ 1e-4, valid > 1e-4).
Paired read-out (`/tmp/piface62.py cycle-63 cycle-62`): alkene faces, hydrogen-bonded dimers and connected molecules
bit-identical (as designed); aromatic faces train 15/15 same basin −0.13 calls (imidazolium__phenol 33 → 27,
ammoniums__benzene 21 → 17, benzene__water 43 → 40, benzene__pyrrole 15 → 14; benzene__imidazolium 29 → 36,
amides__benzene 28 → 32), valid 13 same-basin +0.85 calls (benzene__water 41 → 46, benzene__pyrrole 32 → 36,
benzene__imidazolium 23 → 26; acids__benzene 39 → 37, amines__benzene 16 → 15, benzene__guanidiniums 28 → 27) plus
ammoniums__benzene 33 → 30 (basin change, same energy). The score gains come from the short-reference members
(ammoniums__benzene ref 11, benzene__guanidiniums ref 13, acids__benzene ref 25); in calls the valid set is a wash.

### Interpretation
- The over-stiffening explanation of cycle 62 is only weakly supported: with the aromatic faces at alkene-like
  stiffness the same dimers move by ±3–7 calls in both directions, i.e. the aromatic-face endgames are path-sensitive
  rather than stiffness-limited (the cation faces with Σc 0.05–0.06 — imidazolium__phenol, ammoniums__benzene — did
  improve on both splits, the low-Σc neutral faces — benzene__water, benzene__pyrrole — moved either way).
- The keep is marginal (train −0.0002 at the 1e-4 gate, valid −0.0005) and within the lottery noise; it is retained
  because it is the physically better-normalised model and bit-identical outside the 29 aromatic-face dimers.
- The π line is closed: the class share is 0.03–0.04, the same-basin effects are within ±1 call, and no further
  structural term (centroid pseudo-acceptor, dropped acceptor bends) is worth a cycle.

### Next idea / control
Row-dependent stretch stiffness for bonds to elements beyond the first row (Si–Cl, Ge–Br, Sn–I): the Almlöf single
exponential gives every single bond ≈ 0.36 Ha/Bohr² (5.6 mdyn/Å) at r = r_cov, but C–Cl/C–S/C–Si/C–P bonds are
3.0–3.5 mdyn/Å, C–Br 2.9, C–I 2.3, Si–Cl/P–Cl/S–S 2.5–3.0, Br–Br 2.5, I–I 1.7 — 1.6–3.5× too stiff. Census
(`/tmp/heavy63.py`): connected molecules with such bonds are 164 / 151 (share 0.28 / 0.25); the small mostly-heavy
molecules (ref ≤ 12, 21 train) run at rel 0.90 vs 0.85 for first-row molecules of the same size. Champion is now
`a09dbb5`.

## Cycle 64 — row factors of the Almlöf stretch curvature for bonds to heavier p-block elements (keep)

Champion at start: `a09dbb5d39a63bd0fdb4b4bdb176812d8af2c023` (train 0.7177764, valid 0.7252031). Candidate:
`6c6efdf2e03120e3dbb7daebae4ba6baa2f73377` — **new champion** (train 0.7095286, valid 0.7225279).

### Hypothesis
The Almlöf stretch guess A·exp(−B(r − r_cov)) gives every bond 0.36 Ha/Bohr² = 5.6 mdyn/Å at r = r_cov. That is
right for first-row bonds (C–C 4.5, C–N 5.2, C–O 5.2, C–H 5.0, N–H 6.4 mdyn/Å; C–F and O–H 0.75–0.8× under), but the
bonds of the heavier p-block elements are soft for their length: C–Si 2.9, C–P 3.0, C–S 3.1, C–Cl 3.4, C–Br 2.9, C–I
2.3, Si–Cl 3.0, P–Cl 2.5, S–S 2.5, Cl–Cl 3.2, Br–Br 2.5, I–I 1.7 mdyn/Å, i.e. the model is 1.6–5× too stiff (the
homonuclear bonds sit inside the Cordero radius sum, where the exponential grows: Br–Br 8.7, I–I 8.4 mdyn/Å). A
quasi-Newton step along a mode whose model stiffness is r× the true one leaves |1 − 1/r| of the error, and the
multi-secant memory learns roughly one mode per step, so the 3–6-atom inorganic molecules (NBr3, NI3, S4, I2S2,
PCl3-like, BBr3 …; ref ≤ 12, train n 21 at rel 0.90 = 8–9 calls, the reference's own count) were expected to gain
2–3 calls, heavy-rich organics a fraction of a call per heavy bond. Census (`/tmp/heavy63.py`): connected molecules
with such bonds are 164 / 151 (share 0.28 / 0.25). Rule: one factor per element row applied once per atom (product),
row 3 (Al–Ar) 0.58, row 4 (Ga–Kr) 0.50, row 5 (In–Xe) 0.42, row 6 (Tl–Rn) 0.35; calibrated on the C–X constants above
(C–Si 3.25, C–P 3.1, C–S 3.1, C–Cl 3.4, C–Br 3.0, C–I 2.4) and checked on X–X and H–X (S–S 2.3, Cl–Cl 2.3, Br–Br 2.2,
I–I 1.5, Si–Cl 2.8, P–Cl 2.3, S–H 3.5, Si–H 2.6, H–Cl 4.1, H–Br 4.1, H–I 3.3, Si–F 5.2, Si–O 5.4, S=O 9.1, C=S 6.8:
0.7–1.06× the textbook values; `/tmp/stub64.py` extracts the table from the code). Connected systems only (the
established class rule of guess_hessian: the fragments' intramolecular paths and basins stay as before, so the train
energy credits are safe); s-block keeps the cycle-52 ionic radii, d-block untouched.

### Change (functions)
- Module level: `_STIFFNESS_SCALE` (Z-indexed, ones except the p-block ranges 13–18, 31–36, 49–54, 81–86) with a
  comment giving the calibration.
- `Internals._h0_bond`: `h0 *= _STIFFNESS_SCALE[numbers].prod()` when `ntrans + nrotations == 0` and both atoms are
  real (dummy bonds excluded). `_h0_stretch_diagonal` calls `_h0_bond`, so the tracked stretch diagonal (cycle 55/61
  transport) follows automatically; `_h0_nonlocal_contacts`, bends and torsions unchanged.
- Static checks: py_compile, pyflakes diff empty, algebraic stub of the extracted table.

### Result
Train 0.7095286 vs 0.7177764 (Δ −0.0082, the largest gain since cycle 45), 469/469 converged, mean rel energy
1.0013962 vs 1.0014979 (Σ(rel_e − 1) +0.655 vs +0.703: monoatomics__sulfides — a cation counted as bonded, hence
connected — lost its +0.057 credit, 104079126 gained +0.022 at 25 → 48 calls, 252089162 closed a −0.006 deficit).
Valid 0.7225279 vs 0.7252031 (Δ −0.0027), 465/465, energies bit-identical. Decision **keep**.
Paired read-out (`/tmp/heavy64.py cycle-64 cycle-63`): multi-fragment systems and heavy-free connected molecules
bit-identical on both splits (final positions max diff 0). Heavy connected, train n 164: 61 better / 33 worse, 5
basin changes, same-basin −0.35 calls (dscore −0.0101); by heavy-bond fraction hb/nb 0–0.3 −0.09 calls, 0.3–0.7
−0.15, ≥ 0.7 −0.42; by reference length ref ≤ 12 (n 37) 23 better / 3 worse, −0.89 calls, rel 0.910 → 0.814; ref
12–20 ±0; 20–35 −0.63; ≥ 35 (n 15) +2.27 (3 better / 8 worse, incl. two basin changes). Valid heavy connected n 151:
50 / 37, −0.25 calls, ref ≤ 12 (n 10) 6 / 1 −0.5 calls, rel 0.871 → 0.822; hb/nb ≥ 0.7 only n 12 (±0). The smaller
valid gain is the class composition (valid has 10 tiny heavy molecules and 12 mostly-heavy ones against train's 37 /
45).

### Interpretation
- The stiff-mode learning limit was real: the tiny inorganic molecules fell from 8–9 to 6–7 calls (135156363 8 → 5,
  135020045 10 → 6, 135094396 11 → 8, 135078483 8 → 7 …) without any change of basin, and larger molecules with
  heavy bonds gained 1–3 calls in the same basins (des370k carboxylates__thiols 27 → 18, 135098427 27 → 20,
  135092350 18 → 15).
- 6–7 calls for a 4-atom molecule is still ~2× the harmonic minimum (3–4 calls including the displacement gate), so
  other mis-modelled modes remain: the bends at heavy centres (the Fischer–Almlöf bend formula grows with the
  covalent-radius product, giving H–S–H 0.74 vs 0.43, Cl–P–Cl 1.0 vs ~0.5, Cl–Si–Cl 1.1 vs ~0.4 mdyn Å/rad², while
  bends with heavy ligands at a first-row centre are about right: Cl–C–Cl 0.87 vs 1.0–1.2) — next cycle.
- Molecules with ref ≥ 35 and few heavy bonds worsened on train (+2.3 calls, n 15) but not on valid (−0.8, n 27):
  path sensitivity of long runs, no systematic sign.
- Uncertainty: the X–X bonds are 10–30 % soft under the product rule (Cl–Cl 0.71×); a per-pair rule would be more
  accurate but is not worth scanning (the row constants are not to be tuned on the evaluation).

### Next idea / control
Cycle 65: apply the same row factor of the apex atom to the Fischer–Almlöf bend guess (`_h0_angle`), connected
systems only — the diffuse valence shell that softens the stretches also lowers the angular stiffness at the centre
(H–S–H 0.43 vs H–O–H 0.76, PH3 0.33 vs NH3 0.6 mdyn Å/rad²); checks with the factor 0.58: H–S–H 0.43 (0.43),
Cl–P–Cl 0.59 (~0.5), H–P–H 0.41 (0.35), C–P–C 0.5 (0.5–0.6), C–S–C 0.5 (0.7, under), H–Si–H 0.40 (0.27–0.3, over).

**Amendment (written during cycle 65, before the apex factor was implemented).** The bend comparison above mixed
units (F–A value in Ha/rad² read against mdyn Å/rad² literature constants without the ×4.36 and the ×0.5
valence_scale). A static Wilson-GF fit of diagonal valence constants to textbook fundamentals (`/tmp/gf65.py`,
literature geometries/frequencies only, no PES evaluation) gives the ratio (F–A/2)/(fitted bend constant): CH4 1.17,
NH3 1.30, H2O 0.93, H2S 1.04, PH3 1.08, AsH3 1.25, CCl4 1.25, PCl3 0.78, PBr3 0.91, SCl2 0.89, SiF2 0.80 — i.e. the
bends at S and P centres are *about right* (≤ 10 % off, the halides even soft), so a blanket apex row factor 0.58
would make them 0.5–0.6× under-stiff. The genuinely over-stiff apex bends are the group-13/14 heavy centres:
SiH4 1.30, SiCl4 1.48, SiBr4 1.86, GeH4 1.51, GeCl4 1.82, BF3 1.23, BCl3 1.37, BBr3 1.50 (cause: the F–A factor
(r_cov,1 · r_cov,2)^0.42 grows with the bond lengths, SiCl4's value is 1.78× CH4's while the true constants are
similar). The apex-factor idea therefore survives only as a group-13/14 heavy-centre factor (Si/Ge ≈ 0.65, B ≈
0.75), a much smaller class (train Si 17 molecules share 0.0275, valid 11 share 0.0165; B 20 / 6) — see the backlog
entry `heavy-row-stiffness`. Cycle 65 took the vicinal pair term instead.

## Cycle 65 — vicinal (1,4) pair curvature in the geometry-tracked contact term of connected systems (keep)

Champion at start: `6c6efdf2e03120e3dbb7daebae4ba6baa2f73377` (train 0.7095286, valid 0.7225279). Candidate:
`2a8fb394fa396f8111c886bffafdaa97b190f84f` — **new champion** (train 0.7080763, valid 0.7210370).

### Hypothesis
The rotational stiffness of a single bond in the model is the Fischer–Almlöf torsional constant alone (class-scaled
and shared 1/√n over the dihedrals of the bond); it is fixed at the value of the starting geometry (cycle 46: tracking
the diagonal is a wash) and ignores the substituents' 1,4 contacts, whose repulsion is what makes the true torsional
profile amplitude- and conformer-dependent (the lesson of cycles 40/41/58: "the optimal model stiffness of a rotor
is amplitude-dependent"). Physical rotational stiffness k = 4.5·V₃ (ethane V₃ 2.9 kcal/mol → 0.021 Ha/rad²; the model
gives ≈ 0.026 with the ASE radii), butane anti well ≈ 0.025 (Fourier fit V₁ 1.6, V₂ −0.67, V₃ 3.6), CH₃–CCl₃ ≈ 0.040,
C₂Cl₆ ≈ 0.1. Adding the Almlöf pair curvature of the 1,4 pairs (graph distance 3, the same k(r) as the non-local
term, radial only) to the tracked contact block supplies (i) the substituent-dependent part of the rotor stiffness
that the F–A diagonal lacks — ethane +0.001 (6 gauche H···H × 1.7e-4), butane +0.003 (+13 %; H(C2)···C4 gauche
4 × 5.6e-4), CH₃–CCl₃ 0.023 → 0.033 (barrier 0.040), C₂Cl₆ 0.022 → 0.065 (~0.1) — and (ii) the geometry dependence
of that part (a contact that opens as a rotor relaxes releases its curvature through the analytic transport of the
secant pairs, cycle 53). Census (`/tmp/vic65.py`): 1,4/F–A stiffness ratio ≤ 0.1 for most C–C rotors (median 0.12),
0.2–0.5 for rotors with two heavy substituents on each end, > 0.5 for a handful of S/P-rich molecules (n 12 + 4 above
1.0). 1,5 pairs were excluded (they would add +11 % to butane, +6 % to propane through the 2-bond lever on the
bends, which are already 1.2–1.3× over-stiff per the GF fit) and the contact bends of the 1,4 X–H···Y pairs were
not added (they are not hydrogen bonds). Connected systems only (class rule; the dimers' paths and credits stay).

### Change (functions)
- `Internals._nonlocal_pairs(min_path, vicinal=False)` now returns `(ii, jj, far)`: with `vicinal=True` the pairs at
  graph distance 3 of every fragment are appended, `far` marks the non-local pairs (graph distance ≥ min_path or
  different fragments); cache key extended by `vicinal`.
- `Internals._h0_nonlocal_contacts(..., vicinal=None)`: `vicinal` defaults to `(ntrans + nrotations) == 0`
  (connected systems); the radial pair term H = DᵀKD covers all selected pairs, `_h0_contact_bends` is applied to the
  `far` subset only. Callers (`guess_hessian`, `_track_nonlocal_contacts`, `_analytic_model_from`) unchanged, so the
  1,4 term is geometry-tracked like the non-local contacts and enters the secant-pair transport.
- Static checks: py_compile, pyflakes diff empty, `/tmp/stub65.py` (pure-numpy stub of the pair selection).

### Result
Train 0.7080763 vs 0.7095286 (Δ −0.00145), 469/469 converged, mean rel energy 1.0014082 vs 1.0013962
(`evaluation_results/cycle-65-train.json`, 15ffa629bc374190ae4b401e48ad349d). Valid 0.7210370 vs 0.7225279
(Δ −0.00149), 465/465, mean rel energy 1.0078047 (`evaluation_results/cycle-65-valid.json`,
b353d518b734452383d219bd7b15ed63). Decision **keep** on both gates.
Paired read-out (`/tmp/read65.py cycle-65 cycle-64`): both controls bit-identical on both splits — multi-fragment
212 / 203 and connected molecules without a 1,4 pair within r_cov + 3 Å 30 / 5 (0 changed calls, 0 basin changes).
Connected with 1,4 pairs, train n 227: 103 changed, 54 better / 49 worse, Σ −12 calls; same basin (n 223) 53 / 46,
Σ −23 (dC_S −0.0026); four basin changes Σ +11 (135132759 11 → 25, 135039840 23 → 10, 104079126 48 → 56, 252089162
37 → 39 with rel_e 0.994 → 1.000, i.e. the cycle-64 deficit closed further);
by the max 1,4/F–A ratio of the molecule's rotors: 0–0.1 Σ −17 (n 68), 0.1–0.2 −7 (54), 0.2–0.35 +30 (29; 104079126
+8, 160853090 +5, 252657041 +5, 135101309 +5), 0.35–0.5 +2 (15), 0.5–1 −6 (12), > 1 −4 (10); by reference length
ref 10–20 −2, 20–35 −8, 35–60 −16, ≥ 60 +14 (n 3). Valid n 257: 140 changed, 71 / 69, Σ −20; same basin (254) 69 /
68, Σ −5; three basin changes Σ −15 (135050336 45 → 32, 125084169 37 → 31, 135101049 29 → 33 with dE +0.72
kcal/mol, still above the gate); ratio bins 0–0.1 +2 (91), 0.1–0.2 −12 (76), 0.2–0.35 −16 (29), 0.35–0.5 +7 (22),
0.5–1 0 (17), > 1 0 (4); ref 10–20 Σ −10 (27 / 17), 20–35 +4, 35–60 −17. Largest same-basin changes are ±14 (train
135043047 39 → 25, des370k_carboxylates 18 → 29, acalabrutinib 25 → 15; valid 135065494 43 → 51, des370k_acids 13 →
21, 134985308 32 → 26).

### Interpretation
- The gain is real but small (−0.0015 on both splits, twice the ±0.005-per-split lottery band only when the two
  splits are read together) and lottery-like in structure: the better : worse counts are near even, the sum comes
  from a few ±6–14 re-rolls of long runs, and the same-basin gain is larger on train (−23 calls) than on valid (−5).
- There is no monotone trend with the 1,4/F–A ratio: the classes where the term matters most (ratio ≥ 0.5, S/P-rich,
  n 12 + 10 train, 17 + 4 valid) gave −6 / −4 and 0 / 0 calls, i.e. the substituent-dependent rotor stiffness is not
  what limits those runs (consistent with cycles 40/41/58: rotor endgames are amplitude-limited, not
  stiffness-limited). The 0.2–0.35 bin lost +30 on train and gained −16 on valid — noise.
- Alternative reading: the effect could come mostly from the transport term (the 1,4 contacts now contribute
  T_now @ s − tbar corrections to the secant pairs) rather than from the guess itself; the read-out cannot separate
  the two, and it is not worth a control cycle at this effect size.
- Uncertainty: with 4 + 3 basin changes the split-level Δ has a ±0.001 basin component; the keep rests on the exact
  controls and the same-basin sums both being ≤ 0.

### Next idea / control
The contact term's follow-ups are now closed except 1,5 pairs (caveat above) and the inter-fragment construction
(dimer re-roll risk). Remaining candidates, all model calibrations of small classes: (B) group-13/14 heavy-centre
apex bend factor (Si/Ge ≈ 0.65, B ≈ 0.75 from the unit-corrected GF fit, cycle-64 amendment); (A) boron stretch
factor ≈ 0.65 (B–X bonds 1.4–1.7× over-stiff: BF₃ 2.6 measured vs the model's ~4, train 20 molecules share 0.033);
(C) linear-bend classes (sp carbon 0.07, cumulene/azide 0.12, H 0.05); (D) planar-centre improper. Cycle 66 takes (B)
+ (A) together as one "group-13/14 heavy-centre calibration" if the census shows a shared class, else (B) alone.

## Cycle 66 — group-13/14 heavy-centre calibration of the model Hessian: boron stretch factor and Si/Ge apex bend factors (non_generalizable)

Champion at start: `2a8fb394fa396f8111c886bffafdaa97b190f84f` (train 0.7080763, valid 0.7210370). Candidate:
`674a119482ad8325afdeb56c00d20a2684d3bdca` — rejected (train 0.7060164, valid 0.7213086); preserved in
`ideas/heavy-row-stiffness/674a119482ad8325afdeb56c00d20a2684d3bdca/algo.py`; champion restored.

### Hypothesis
The cycle-64 amendment (unit-corrected static Wilson–GF fit of textbook fundamentals, `/tmp/gf65.py`, `/tmp/gf66.py`)
left two over-stiff primitive classes in the model: (A) bonds to boron — the Almlöf exponential with the electron-
deficient element's radius (0.84 Å) gives B–H 4.8, B–C 6.5, B–N/B–O 9.0, B–F 8.1, B–Cl 5.0, B–Br 4.9 mdyn/Å against
3.5 / 3.5 / 5.5 / 6.5 / 7.2 / 3.7 / 2.9 measured, i.e. 1.1–1.7× over-stiff (boron has no row factor); (B) the
Fischer–Almlöf bend curvature at the heavier group-14 centres — the (r_cov,1·r_cov,2)^0.42 factor makes a Si/Ge apex
1.1–1.3× a carbon apex while the fitted valence bend constants hardly change with the centre (SiH₄ 0.53 vs CH₄ 0.52,
SiCl₄ 0.76 vs CCl₄ 0.70 mdyn Å/rad²): SiH₄ 1.30, SiCl₄ 1.48, SiBr₄ 1.86, GeH₄ 1.51, GeCl₄ 1.82 (carbon 1.17–1.25;
P/S/O/As within 10 %). A boron apex factor was dropped after the multi-start fit gave BF₃ 0.90 (the single-start
1.23 was a spurious minimum; BCl₃/BBr₃ fits are poor). Candidate: `_STIFFNESS_SCALE[5] = 0.65` (B–X → 3.2 / 4.2 / 5.9
/ 5.9 / 5.3 / 3.3 / 3.2 mdyn/Å, ratios 0.73–1.21) and an apex factor 0.70 at Si, 0.60 at Ge/Sn/Pb (SiH₄ 0.91, SiCl₄
1.03, SiBr₄ 1.31, GeH₄ 0.91, GeCl₄ 1.09), connected systems only, expected −0.001 to −0.003 (train classes B 20
molecules share 0.033, Si 17 share 0.0275; valid 6 / 11; nearly disjoint: overlap 135093107, 135095518).

### Change (functions)
- Module level after the `_STIFFNESS_SCALE` row loop: `_STIFFNESS_SCALE[5] = 0.65` (one factor per boron atom, as
  the row factors; the tracked stretch diagonal follows through `_h0_bond`); new table `_BEND_APEX_SCALE`
  (Z-indexed ones; 14 → 0.70, 32 / 50 / 82 → 0.60).
- `Internals._h0_angle`: for connected systems (`ntrans + nrotations == 0`) the F–A value is multiplied by
  `_BEND_APEX_SCALE[numbers[apex]]` (apex = `angle.indices[1]`; dummy atoms have number 0 → 1.0) before the
  valence_scale 0.5. Linear-bend dummies, torsions, contact terms unchanged.
- Static checks: py_compile, pyflakes diff empty, `/tmp/stub66.py` (exec of the table block with stand-in radii;
  calibration ratios recomputed).

### Result
Train 0.7060164 vs 0.7080763 (Δ −0.00206), 469/469 converged, mean rel energy 1.0014057 vs 1.0014082
(`evaluation_results/cycle-66-train.json`, 4cf71e4240084bca8ccd293f86a2fc66). Valid 0.7213086 vs 0.7210370
(Δ +0.00027), 465/465, mean rel energy 1.0078047 (`evaluation_results/cycle-66-valid.json`,
ae1f650c9eaa44b6891ca4d2083a8a3e). Decision **non_generalizable** ("valid improvement is not greater than 1e-4").
Paired read-out (`/tmp/read66.py cycle-66 cycle-65 train valid`): controls bit-identical on both splits (multi-
fragment 212 / 203, connected without B or Si 222 / 245: 0 changed calls, 0 basin changes, max |Δpos| 0).
Train, connected with B or Si, n 35: 21 changed, 16 better / 5 worse, Σ −22; same basin (33) 15 / 4, Σ −20
(dC_S −0.0026); two basin changes (135039840 Cl₆OSi₂ 10 → 20, the molecule that re-rolled 23 → 10 in cycle 65, dE
−0.01; 135095518 BF₇Si₂ 25 → 13, dE +0.07 kcal/mol). B only (18): 8 / 2, Σ −15 (B₁O₃ 10 → 9, BBr₃ 8 → 7, BClI₂ 8 → 7,
CB₄ 11 → 8, H₄B₆ 41 → 39, B₄O₇ 13 → 12, B₈S₁₆ 25 → 22, C₆H₁₈B₂N₄O₂ 33 → 28; HBBrF 6 → 7, C₂H₈BP 15 → 16). Si only (15):
7 / 3, Σ +5 (+10 from the basin change; same basin −5: F₂OSi 8 → 7, Br₄Si 7 → 6, I₄Si 7 → 6, Cl₈Si₄ 15 → 13, H₁₂Si₅
22 → 20, C₅H₆Si 8 → 7, C₆H₁₈P₄Si₃ 17 → 16; C₁₁H₂₈Si₃ 19 → 22, C₁₆H₂₄O₃Si 23 → 24). By reference length ref ≤ 12
(14) Σ −8, 13–25 (12) +5, > 25 (9) −19.
Valid, n 17 (all same basin): 10 changed, 5 / 5, Σ +2 (dC_S +0.00027). B only (6): 0 / 2, Σ +5 — 135100509
C₆H₁₀BNO₄ 13 → 17 (a boratrane-type cage: B(OR)₃ with the apex angles summing to 352° and a transannular N···B
contact at 2.54 Å that is a 1,5 pair, i.e. in no term of the model) and C₉H₂₁BO₃ 15 → 16; Si only (11): 5 / 3, Σ −3
(H₂F₄SSi₂ 17 → 15, H₅NOSi₂ 16 → 15, C₈H₈F₁₂O₄Si 46 → 45, C₁₁H₂₄N₂Si 13 → 12, C₁₁H₂₀N₄OSi₂ 13 → 12; O₇P₂Si 12 → 13,
C₆H₁₉NO₃SSi₂ 22 → 23, C₁₉H₂₆O₃Si 28 → 29). Max |Δpos| 9.4e-3 Å (same minima, slightly different endpoints).

### Interpretation
- The calibration behaves as predicted where the class is populated: the tiny boron and silicon molecules (ref ≤
  12) lose a call each, as the row factors did in cycle 64, and the boron-rich cages (CB₄, H₄B₆, B₈S₁₆) lose 2–3.
  Over both splits the same-basin sum is −18 calls on 52 affected molecules (−0.35 per molecule, the cycle-64
  magnitude), but the valid split holds only 6 boron and 11 silicon molecules, and a single +4 (the boratrane, whose
  dominant soft mode — B pyramidalisation against the dative N···B contact — is absent from the model) flips its
  sign. The gate cannot resolve an effect of this class size; the decision is a coin toss on 17 molecules.
- Alternative reading: the boron factor is right for trigonal boron (BX₃, boranes, borate esters — the train gains)
  and wrong for the four-coordinate / dative cases (the two valid losses are a borate cage and a boronic ester with
  a pendant amine); a tetracoordinate boron is a different bonding situation (B–O in borate anions ≈ 4–5 mdyn/Å,
  i.e. the factor should be even smaller there, not larger), so the loss is more likely the missing dative
  coordinate than the factor. Only one such molecule per split — unresolvable.
- Uncertainty: two train basin changes (Σ −2 net) do not decide the train result; the valid result rests on 10
  changed molecules with ±1 changes plus one +4.

### Next idea / control
The remaining mis-modelled primitive classes are all small (Si/B ≈ 3–4 % of the score, sp centres ≈ 2 %); each on
its own sits inside the per-split noise, so a further single-class calibration will only pass by luck. Options: (C)
linear-bend classes (terminal alkyne C≡C–H 0.10 → 0.04 Ha/rad², nitrile / internal alkyne 0.07, cumulene 0.12) —
the class is larger (train ≈ 30 sp-centre molecules) and the miscalibration is 2.5–3× rather than 1.3×, so its
signal should exceed the noise if it exists at all; (E) a "dative contact" term (lone-pair donor N/O/P at 2.0–2.8 Å
from an electron-deficient B/Al centre) — tiny class, parked; (G) planar-start amine/phosphine saddle starts (would
re-roll connected-system basins and endanger the train energy gate). Cycle 67 takes (C).

## Cycle 67 — class-resolved linear-bend constants for the dummy coordinates of near-linear centres (non_generalizable)

Champion at start: `2a8fb394fa396f8111c886bffafdaa97b190f84f` (train 0.7080763, valid 0.7210370). Candidate:
`3420ebd99102f8e217d263ab14066fe81c39c230` — rejected (train 0.7076505, valid 0.7219545); preserved in
`ideas/linear-bend-guess/3420ebd99102f8e217d263ab14066fe81c39c230/algo.py`; champion restored.

### Hypothesis
The cycle-39 linear-bend guess gives every dummy coordinate of a near-linear two-bonded centre the same curvature
(0.10 Ha/rad², 0.05 at hydrogen centres). A static Wilson–GF fit of textbook fundamentals with the unit-Jacobian
linear-bend coordinate (`/tmp/gflin67.py`, 1 Ha/rad² = 4.36 mdyn Å/rad²; the earlier docstring numbers were k/l²-type
values, unit-corrected here) resolves three classes: one multiple + one single bond X–C≡Y 0.062–0.087 Ha/rad²
(CH₃CN 0.271, propyne 0.289, NCCN 0.291, ClCN 0.347, FCN 0.381 mdyn Å/rad²) and H–C≡ 0.04–0.057 (HCN 0.249, HCCH
0.241, propyne ≡C–H 0.181); heterocumulenes 0.085–0.18 (CO₂ 0.773, N₂O 0.650, OCS 0.642, HNCO 0.60–0.77, CH₃NCS
0.63, HN₃ 0.53, ketene 0.37–0.54); all-carbon cumulene 0.061 (allene 0.267). So H–C≡ was 2× over-stiff, X–C≡Y
1.15–1.6× over-stiff, heterocumulenes 1.4–1.8× too soft. Class census (`/tmp/lin67b.py`, ratio r/(r_cov,i + r_cov,j)
< 0.895 = multiple): train 37 molecules with linear centres (share 0.058, rel 0.731; X–C≡Y 24, H–C≡Y 5,
heterocumulene 4, cumulene 2, two-single 4, H-centre 1), valid 42 (share 0.071, rel 0.771). Expected −0.0005 to
−0.0015 per split if the update spends a call per mis-modelled dummy mode, as for the stretches of cycle 64.

### Change (functions)
- `Internals._h0_linear_bend(centre, k_bend=0.10, k_bend_h=0.05, k_triple=0.07, k_triple_h=0.05, k_cumulene=0.14,
  k_allene=0.07)`: hydrogen centres keep 0.05; for the two real bonds of the centre a multiplicity weight ramps from
  0 (r ≥ 0.92 r_ref, r_ref = `_STIFFNESS_RADII` sum) to 1 (r ≤ 0.87 r_ref); the constant interpolates linearly in the
  number of multiple bonds from k_bend (0) through k_triple (1; k_triple_h with a hydrogen partner) to k_cumulene
  (2; k_allene for a carbon centre with two carbon partners). Callers (`guess_hessian` for the free dummy angle and
  the dummy dihedral, connected systems only) unchanged; multi-fragment systems unchanged.
- Static checks: py_compile, pyflakes diff empty, `/tmp/stub67.py` (rule re-implemented on the census centres:
  train 0.05 × 8, 0.07 × 44, 0.10 × 3, 0.14 × 2 plus 7 intermediate values; valid 0.05 × 7, 0.07 × 48, 0.14 × 2 plus 8).

### Result
Train 0.7076505 vs 0.7080763 (Δ −0.00043), 469/469 converged, mean rel energy 1.0014095 (unchanged;
`evaluation_results/cycle-67-train.json`, 17744faa48f24457a0a5ba4237c792ef). Valid 0.7219545 vs 0.7210370
(Δ +0.00092), 465/465, mean rel energy 1.0078050 (`evaluation_results/cycle-67-valid.json`,
4c18af3902f746bda7ae00e1cd72ea92). Decision **non_generalizable** ("valid improvement is not greater than 1e-4").
Paired read-out (`/tmp/read67.py cycle-67 cycle-65 train valid`): multi-fragment controls bit-identical (212 / 203);
connected molecules without a linear centre at the start: train 220 unchanged calls (five molecules with |Δpos| up to
3e-2 Å and equal calls — Cl₆OSi₂, B₂S₃, BO₃, SiO₂, SiS₂ — straighten during the run, so a dummy is added mid-run and
the new constant enters there), valid 220 with one change: 104129283 C₄H₆O₉P₂S₁ (ref 53) 32 → 40 after a P–O–P bridge
crosses 165° mid-run (final 162.9°; O centre with a 1.54 Å P–O bond → weight 0.6 → 0.082 instead of 0.10), dC_S
+0.00032. Affected at the start (constant changed), all same basin, no basin change on either split:
- train n 33: 18 changed, 11 better / 7 worse, Σ 0 (dC_S −0.00043); ref 13–25 (22) Σ −8, ref > 25 (9) Σ +8 of
  which acalabrutinib (internal alkyne, both sp carbons 0.10 → 0.07) 15 → 25; without it Σ −10 on 32 molecules.
  X–C≡Y/allene only (21) 5 / 3 Σ +7 (−3 without acalabrutinib); H–C≡Y (5) 3 / 2 Σ −3; heterocumulene (5) 2 / 1 Σ −4
  (ClC(O)NCO-type 17 → 15, an isothiocyanate 27 → 24, a sulfonyl-isothiocyanate 16 → 17); intermediate (2) 1 / 1.
- valid n 39: 15 changed, 5 better / 10 worse, Σ +1 (dC_S +0.00059); ref ≤ 25 (30) Σ +6 (nine +1 changes on
  ref 7–24 molecules: N₃ 5 → 6, C₂N₂S₃ 9 → 10, C₄H₂N₄ 12 → 13, …), ref > 25 (9) Σ −5 (135249279 C(N–O) 38 → 34,
  125084169 31 → 29, 135094940 28 → 26). X–C≡Y/allene only (27) 4 / 7 Σ +2; H–C≡Y (4) 0 / 1 Σ +1; heterocumulene
  (4) 1 / 1 Σ −3; intermediate (5) 1 / 1 Σ −3.
Over both splits: affected 72 molecules, 16 better / 17 worse, Σ +1 (+9 with the mid-run rebuild); heterocumulenes
(9) 3 / 2 Σ −7 are the only sub-class with a consistent sign.

### Interpretation
- No systematic effect: the sign flips between splits, the sub-classes are 5 / 3, 3 / 2, 4 / 7, 0 / 1, and the two
  largest changes (+10, +8) are path-sensitivity events on long runs of large molecules with the same final
  minimum. The dummy linear bends are soft, low-curvature modes that hardly couple to the rest; a 1.2–2× error in
  their guess is absorbed by the first update or never becomes the limiting mode, unlike the stiff stretches of
  cycle 64 (where the class effect was 23 : 3 on tiny molecules). The premise — one call per mis-modelled mode —
  holds for stiff modes that the trust region has to learn, not for soft ones.
- The one consistent sub-class is the stiffening of heterocumulene dummies (0.10 → 0.14, −7 calls on 9
  molecules): here the guess was too *soft*, which is the direction that costs steps (over-long steps along the
  mode, rejected by the trust region) — coherent with the stretch picture. Softening X–C≡Y (0.10 → 0.07) is neutral
  to slightly adverse (the update handles an over-stiff soft mode for free, and the softer guess opens larger
  steps that cost on long runs); softening H–C≡ 3 / 3 over both splits.
- Alternatives: (i) the multiplicity ramp assigns intermediate constants to the mid-run rebuild cases, which is
  where the +8 came from — a hard classification would have left that O(P–P) centre at 0.10, but the same
  path-noise would surface elsewhere; (ii) the class effect might be visible only for isolated sp centres in
  small molecules (train ref ≤ 12: 0 changes on 2 molecules; valid 1 / 0 / 1 on 4) — no evidence either way.
- Uncertainty: train −0.00043 and valid +0.00092 are both inside the ±0.001 path-noise band for a change
  touching 33–39 molecules; the read-out cannot resolve an effect smaller than ±0.3 calls per affected molecule.

### Next idea / control
The linear-bend line is closed for softening: keep 0.10 / 0.05 (cycle 39). A heterocumulene-only stiffening
(0.10 → 0.14 at centres with two multiple bonds and a heteroatom) is coherent with the stretch picture but touches
4 train / 4 valid molecules — unresolvable alone; bundle only if a larger textbook-calibration candidate is ever
assembled (with the cycle-66 B/Si factors; note that the evaluator is deterministic, so the cycle-66 valid +2 and
the cycle-67 heterocumulene −3 would re-contribute exactly). Cycle 68 leaves the small-class calibrations and
returns to the structural lines of the backlog (endgame / update / multi-fragment share 0.57).

## Cycle 68 — cosine-tracked three-fold torsional curvature (−h cos 3φ) in the analytic model, transported with the secant pairs (discard)

Champion at start: `2a8fb394fa396f8111c886bffafdaa97b190f84f` (train 0.7080763, valid 0.7210370). Candidate:
`e0226284cc800da499e17ebca802aa9325303ab1` — rejected on train (0.7104440, Δ +0.00237); preserved in
`ideas/torsion-cosine-transport/e0226284cc800da499e17ebca802aa9325303ab1/algo.py`; champion restored.

### Hypothesis
The last untested structural mechanism of the connected-system cost map is the chord lag of the secant pairs from
large rotor turns: a three-fold rotor V = V₃/2 (1 + cos 3φ) has the curvature −(9V₃/2) cos 3φ = −h cos 3φ (h the
guess at the staggered minimum, zero 30° away, −h at the eclipsed barrier), so a secant measured over a 30–60° turn
carries the path average (≈ 0.46 h for a 30° → 11° step, ≈ 0 for a full 60° turn), and the multi-secant memory
re-imposes that average in the endgame (the cycle-53 mechanism: model shifts along sampled directions are overridden
unless the pairs are transported). Cycle 46 tracked only the bond-length dependence of the torsional guess without
transport (a wash); the angle dependence with transport was never tested. Expected −1…−2 calls on class molecules
whose rotors start off-staggered and turn > 30° (train: 27 molecules with one such rotor, 24 with ≥ 2; big-turn
groups carry 0.22 / 0.27 of the score), ~0 for staggered starts; multi-fragment systems and connected molecules
without a class rotor bit-identical.

### Change (functions)
- Module: `_ROTOR_CENTRE_Z` (p-block groups 13–16), `_dihedral_angles(pos, ind)` (vectorised arctan2 form of
  `_dihedral_value`; stub-checked to 2e-16 and on built geometries, `/tmp/stub68.py`).
- `Internals._threefold_rotor(b, c, types, adj, positions, numbers, ring_max=8, pyramid_sum=350)`: class rule —
  neither centre 'pi', both in `_ROTOR_CENTRE_Z`, at least one tetrahedral centre (three other neighbours), the other
  tetrahedral, two-coordinate, or three-coordinate pyramidal (angle sum ≤ 350°; planar amide/aniline N excluded,
  its two substituents 180° apart cancel the cosines), bond not in a ring ≤ 8; two lone-pair centres (peroxide,
  disulfide, hydrazine: gauche minima) fall out because neither is tetrahedral.
- `Internals._h0_torsion_cosine_diagonal()`: vector over the internals with −h_i cos 3φ_i at the table entries
  (`_torsion_cos_table` = (idx, h, atom indices) built in `guess_hessian` for the proper dihedrals of class rotors,
  connected systems only, h_i the per-dihedral guess after class factor and 1/√n sharing).
- `guess_hessian`: the table entries of `H0` are overwritten with the signed cosine value at the start geometry
  (after the `abs`); `InternalPES._analytic_model_from`: `d = stretch diagonal + cosine torsional diagonal`, so
  `_track_analytic_model` moves the entries with the geometry and transports the stored pairs (Simpson average over
  the step, shadow copy shares the table via `shadow_copy`).
- Static checks: py_compile, pyflakes diff empty, class-rule stub (ethane / pyramidal amine True; planar N, disulfide,
  sp³–sp², C–Li, ring bond, five-coordinate S False), cosine sign (+h staggered, −h eclipsed, 0 at 30°).

### Result
Train 0.7104440 vs 0.7080763 (Δ +0.00237), 469/469 converged, mean rel energy 1.0014898 (vs 1.0014095;
`evaluation_results/cycle-68-train.json`, f4dc8b22085d48dea174f09852cf5b87). Decision **discard** ("train
improvement below 1e-4"). Paired read-out (`/tmp/read68.py cycle-68 cycle-65 train`): multi-fragment (212) and
connected molecules without a class rotor (142) bit-identical. Affected 115 molecules (class rotor at the start):
58 changed, 28 better / 30 worse, Σ +1 call, dC_S +0.00237, 3 basin changes (135122079 −4 calls dE −0.33 kcal/mol,
Cl₆OSi₂ 10 → 22 dE −0.01, one more). The score loss sits in short-reference molecules: ref 13–25 (65) Σ +21
(dC_S +0.00249), ref ≥ 26 (46) Σ −21; nat ≥ 41 (50) Σ −36 (dC_S −0.00134), nat 8–20 (22) Σ +19; ≥ 4 class
rotors (49) Σ −32, 1–3 rotors (66) Σ +33. By start phase: near-staggered starts (max cos 3φ < −0.5, 41) 3 / 5 Σ 0;
off-staggered ([−0.5, 0.5], 47) 15 / 13 Σ +11; near-eclipsed (> 0.5, 27) 10 / 12 Σ −10 but dC_S +0.00152 (Cl₆OSi₂).
By turn: nbig(> 30°) = 1 (27) 6 / 9 Σ +14, nbig ≥ 2 (24) 11 / 10 Σ −16. Largest changes: 384221749 59 → 34,
135140595 38 → 32, 135130097 40 → 35, 319495631 20 → 15; Cl₆OSi₂ 10 → 22, BF₇Si₂ 25 → 36, 135049744 (phosphonate
esters) 25 → 33, 125086620 (one C–S rotor) 11 → 16.
Static amplitude census (`/tmp/amp68.py`: Fischer–Almlöf rotor curvature √n·h_single of each class rotor at the
start geometry against a textbook three-fold barrier 9V₃/2 with V₃ = 0.33 kcal/mol per eclipsing substituent pair
× (1.54 Å / r)³ — ethane 2.9, methylamine 2.0, methanol 1.1, methylsilane 1.7, disilane 1.2): C–C rotors (232)
median ratio 1.12 (calibrated), O–C (84) 2.1, N–C (32) 1.7, C–N (19) 1.5, C–O (13) 2.4, O–P (14) 4.7, O–Si (5)
3.3, N–P 2.3, O–S 3.1; S–C (28) 1.0, C–P 1.0, C–Si (17) 0.71, Si–Si (5) 0.42. Outcome by the largest ratio of a
molecule's rotors: ≤ 1.5 (45) 9 / 13 Σ +15; 1.5–3 (47) 14 / 6 Σ −40 (dC_S −0.00161); > 3 (23) 5 / 11 Σ +26
(dC_S +0.00276). Molecules with an off-staggered rotor of ratio > 2 (31) Σ +22; the other 84 Σ −21.

### Interpretation
- The chord-lag hypothesis is not supported: rotors that turn > 30° do not consistently benefit (nbig = 1 Σ +14,
  nbig ≥ 2 Σ −16), near-staggered starts are untouched (Σ 0, as expected), and the well-calibrated C–C/S–C/C–Si
  rotors (ratio ≤ 1.5) are 9 / 13. The multi-secant memory replaces the lagging pair within a step or two of the
  rotor settling, so the lag costs less than assumed; what the tracked cosine adds is a *phase-dependent softening*
  along every class rotor for the whole approach (zero curvature 30° off, negative beyond).
- Where the guess is 1.5–3× over-stiff (mostly O–C / N–C hetero rotors) that softening lands nearer the truth on
  average along the path and gains (−40 calls on 47 molecules, 14 : 6); where it is > 3× over-stiff (O–P, O–Si
  esters; Cl₃Si–O–SiCl₃ with a nearly free rotor) the swing reaches −0.01…−0.07 Ha/rad² on modes whose true
  curvature is ~0 and the runs derail (+26); where it is right the mechanism is neutral with fat path-noise tails
  (BF₇Si₂ +11 on a 0.0015 Ha/rad² Si–Si rotor). The local curvature of a cosine well is the wrong stiffness for a
  quasi-Newton step far from the minimum: 30° off it is zero (free mode, trust-limited step) and beyond it
  negative, while the step that lands on the minimum needs the *secant* stiffness V′(ψ)/ψ = h·sinc(3ψ) (0.64 h at
  30°, positive everywhere) — the amplitude-conditioned model the cycle-41 lesson asked for.
- Alternatives: (i) the amplitude h of the swing is the Fischer–Almlöf guess, 2–5× the physical 9V₃/2 for hetero
  rotors — capping it at the pair-rule barrier would remove the > 3 losses but also most of the 1.5–3 gains (the
  static hetero-rotor recalibration itself is the closed class-factor family of cycles 40/41); (ii) the three basin
  changes and the two +11/+12 tiny molecules are lottery events that decide the sign of the score change on their
  own (without Cl₆OSi₂ and BF₇Si₂ the affected set is Σ −22, dC_S −0.0009).
- Uncertainty: 58 changed molecules with ±1–2 path noise; the group sums are within the noise except the > 3 group
  (5 : 11) and the ≥ 4-rotor / large-molecule group (Σ −32 / −36), which are consistent with the mechanism acting
  as a softening of over-stiff hetero rotors rather than as a lag correction.

### Next idea / control
Repair candidate 1 (cycle 69): the same table, class rule and transport with the *secant* stiffness h·sinc(3ψ)
(ψ the distance from the nearest staggered position; floor 0.05 h within 3° of the barrier, where sinc(171°) = 0.05,
so the model stays non-singular) instead of the local curvature −h cos 3φ. It keeps the guess h at staggered
starts and in the endgame (bit-identical there up to the transport), softens off-staggered rotors to the stiffness
that puts one Newton step on the minimum, never goes to zero or negative, and transports the pairs with the same
function. Controls as in cycle 68 (multi-fragment, no class rotor); the read-out groups by start phase and amplitude
ratio must show the > 3 group losses gone and the 1.5–3 gains kept for the line to survive.

## Cycle 69 — phase-tracked secant stiffness h·sinc(3ψ) of the three-fold rotors in the analytic model, transported with the secant pairs (non_generalizable)

Champion at start: `2a8fb394fa396f8111c886bffafdaa97b190f84f` (train 0.7080763, valid 0.7210370). Candidate:
`233eaf5f4692688b4502a8579fb8f20dc12b1ba9` — train 0.7043994 (Δ −0.00368, passed), valid 0.7219659 (Δ +0.00093),
decision **non_generalizable**; preserved in `ideas/torsion-cosine-transport/233eaf5f4692688b4502a8579fb8f20dc12b1ba9/algo.py`;
champion restored (`git reset --hard 2a8fb39…`).

### Hypothesis
Repair candidate 1 of the cycle-68 line: the phase-dependent rotor stiffness that puts one quasi-Newton step on the
staggered minimum is the *secant* V′(ψ)/ψ = h·sinc(3ψ) of the three-fold well (ψ = distance from the nearest
staggered position: h at ψ = 0, 0.64 h at 30°, 0.05 h within 3° of the barrier), not the local curvature −h cos 3φ
that cycle 68 tracked (zero at 30°, negative beyond). With the same table, class rule and pair transport, the model
keeps the guess h at staggered starts and in the endgame, softens off-staggered rotors, and never goes to zero or
negative — so the > 3 amplitude-ratio losses of cycle 68 (Cl₃Si–O–SiCl₃ 10 → 22, BF₇Si₂ 25 → 36, phosphonate esters)
should vanish while the 1.5–3 gains (Σ −40 on 47 molecules) stay. Controls (multi-fragment systems, connected
molecules without a class rotor) bit-identical.

### Change (functions)
- `Internals._h0_torsion_rotor_diagonal(self, floor=0.05)` replaces `_h0_torsion_cosine_diagonal`: for the table
  entries `phi = _dihedral_angles(positions, ind)`, `3ψ = arccos(−cos 3φ)` ∈ [0, π], value
  `h · max(sinc(3ψ/π), floor)` (numpy's normalised sinc; floor = sinc(171°) keeps the model non-singular at the
  barrier). Everything else as in cycle 68: `_ROTOR_CENTRE_Z`, `_dihedral_angles`, `_threefold_rotor` class rule
  (tetrahedral/pyramidal non-π p-block centres, rings ≤ 8 excluded), `_torsion_cos_table` built in `guess_hessian`
  for connected systems, `H0` entries overwritten with the start-geometry value, `_analytic_model_from` adds the
  rotor diagonal to the stretch diagonal so `_track_analytic_model` moves the entries and transports the pairs
  (Simpson average over the step, shared table in `shadow_copy`).
- Static checks: py_compile, pyflakes diff vs champion empty; stub of the formula (h at staggered, 0.64 h at 30°,
  floor at the eclipsed barrier, symmetric in ±ψ).

### Result
- Train 0.7043994 vs 0.7080763 (Δ −0.00368), 469/469 converged, mean rel energy 1.0013740 (vs 1.0014082);
  `evaluation_results/cycle-69-train.json` (5d2bfdd050154c55b53b9c36e7f33fdf). Valid 0.7219659 vs 0.7210370
  (Δ +0.00093), 465/465 converged, mean rel energy 1.0072680 (vs 1.0078047); `evaluation_results/cycle-69-valid.json`
  (da643e08687f417ab17a60be1493b3d3).
- Paired read-out (`/tmp/read68.py cycle-69 cycle-65 <split>`, code pointer at the preserved file): controls
  bit-identical on both splits (train 212 multi-fragment + 142 no-rotor connected; valid 203 + 120).
  Train affected 115: 46 changed, 29 better / 17 worse, Σ −55 calls; same basin (112) Σ −37 (dC_S −0.00271), 3 basin
  changes Σ −18 (dC_S −0.00097; 315516336 19 → 7 dE +0.08 kcal/mol, 135186987 22 → 12 dE +0.25, 384221749 59 → 47);
  near-eclipsed starts (27) 12 / 6 Σ −39; ≥ 4 class rotors (49) Σ −42; nat 8–20 (22) Σ −30 (dC_S −0.00224, incl. the
  two basin changes); amplitude ratio ≤ 1.5 (45) Σ −23, 1.5–3 (47) Σ −21, > 3 (23) Σ −11 — the cycle-68 tiny-molecule
  derailments are gone as predicted (Cl₆OSi₂ 10 → 14 instead of 22, BF₇Si₂ 25 → 26 instead of 36), the > 3 group now
  5 : 6 instead of 5 : 11.
  Valid affected 142: 54 changed, 26 better / 28 worse, Σ +8 calls; same basin (140) Σ +26 (dC_S +0.00248), 2 basin
  changes Σ −18 (135239978 37 → 19 landing 13 kcal/mol *above* the champion's minimum — a credit, not an improvement;
  one more); nbig = 1 (44) Σ +16, nbig ≥ 2 (30) Σ −9; near-eclipsed (33) 8 / 13 Σ +5; one class rotor (30) 1 / 7
  Σ +18 (dC_S +0.00144); nat ≥ 41 (63) Σ +30; amplitude ratio ≤ 1.5 (42) Σ +1, 1.5–3 (68) Σ −4, > 3 (32) 6 / 11
  Σ +11; off-staggered rotor with ratio > 2 (34) 7 / 12 Σ +17 (train: 10 / 9 Σ −15); no such rotor (108) Σ −9
  (train Σ −40). Largest valid losers: 103928880 20 → 31 (one N–S rotor, ratio 6.1), 135169446 15 → 26, 135065494
  51 → 60, three +7s; winners 135278208 46 → 39, 135249279 38 → 32, 104129283 32 → 27.
- Combined over both splits: affected 257, 55 better / 45 worse, Σ −47, of which 5 basin changes Σ −36; same-basin
  252 molecules 52 : 44, Σ −11 (≈ −0.04 calls/molecule, dC_S −0.0002 over 934 molecules) — indistinguishable from
  zero.

### Interpretation
- The repair did what it was designed to do on the failure mode it targeted (the > 3 over-stiff hetero rotors no
  longer derail: 5 : 6 / 6 : 11 instead of 5 : 11 with +11/+12 tiny-molecule outliers), but the mechanism has no
  same-basin signal: the whole combined gain is five basin changes (lottery events, one of them an energy *credit*),
  and every read-out group that gained on train flipped sign on valid (near-eclipsed Σ −39 → +5, off-staggered
  over-stiff Σ −15 → +17, nat ≥ 41 Σ −12 → +30, one-rotor Σ −17 → +18). Same-basin runs are 52 : 44 with Σ −11 over
  252 molecules. Phase-tracked rotor stiffness (either form) therefore does not change the number of calls of a
  connected run beyond ±1–2 path noise: as concluded in cycle 68, the multi-secant memory re-learns the rotor
  stiffness within a step or two of the rotor settling, and the endgame (where the calls are spent) runs at
  staggered phase where both forms equal the guess h.
- Repair candidate 2 (amplitude cap: h − a + a·sinc with a = min(h, 9V₃/2)) would remove the > 3 group (a
  coin-flip group, Σ −11 / +11) and keep the ≤ 3 groups, whose combined effect is Σ −23−21+1−4 = −47 on 202 molecules
  but with the sign carried by the train basin changes; its expected value is ≈ 0 ± noise, i.e. not worth a cycle.
  The static hetero-rotor recalibration behind it is the closed class-factor family (cycles 40/41).
- Alternatives considered: (i) the valid loss is concentrated in one-rotor / big molecules where a single
  softened rotor changes the *path* (Σ +18 on 30 one-rotor molecules, 1 : 7) — a genuine adverse effect of softening
  a rotor that then takes a different, longer route to the same basin; that argues against the mechanism, not for a
  repair. (ii) Noise: 100 changed molecules on valid with ±1–2 each gives σ ≈ 10–15 calls on Σ, so Σ +26 same-basin
  is a ~2σ adverse draw at most — the honest reading is "neutral", and neutral is a rejection for a mechanism that
  costs a table, a shadow-copy dependency and a transport term.

### Next idea / control
Line `torsion-cosine-transport` closed (both the local-curvature and the secant form tested; repair 2 has no
expected value). The rotor-turn premium (≈ +8 calls per big-turn molecule, ≈ 14 % of connected calls) is not a
stiffness-tracking problem; whatever remains of it is path length (number of steps a 60° turn needs under the
trust radius and the Cartesian bound), which the closed TR scans cover. Cycle 70 moves to a different mechanism.

## Cycle 70 — rigid-body docking of neutral multi-fragment starts on a point-charge + UFF 12-6 surrogate before the quasi-Newton run (keep)

Champion at start: `2a8fb394fa396f8111c886bffafdaa97b190f84f` (train 0.7080763, valid 0.7210370). Candidate:
`0cee2f51cc2eb67496f6cd7dd1cea8b8ed9803c8` — train 0.6974612 (Δ −0.01062, passed), valid 0.6973927 (Δ −0.02364),
decision **keep** → new champion `0cee2f5…`.

### Hypothesis
The remaining dimer cost (train share 0.28: intercept 0.19 + walk 0.09; walkers rmsd₀ > 0.6 Å train n 81 / valid
n 90 at ≈ 0.11 Å per call, half of it trust-cap-limited at 0.25 Å/step) is the *approach*: Sella spends 10–25 calls
walking a fragment from a far or loose DES370K start into contact along model-invisible rigid-body modes. A classical
intermolecular surrogate (point charges + UFF 12-6) knows where the contact is to within ≈ 0.5 Å for neutral
organic dimers, so relaxing only the six rigid-body degrees of freedom per fragment on the surrogate before the
first Sella step — and paying one xTB call to check that the docked pose is lower in energy — should remove most of
the walk for far/loose starts while leaving near-equilibrium starts (small docking move) and everything else
(connected systems, ionic complexes) bit-identical. The gate on the cosine between the xTB and surrogate rigid-body
force fields at the start refuses starts where the surrogate disagrees with xTB about the direction of approach.

### Change (functions)
- New module block after `_break_start_symmetry` (rationale comment, constants fixed a priori: `_DOCK_MIN_COS` 0.5,
  `_DOCK_MIN_RMSD` 0.5 Å, `_DOCK_MAX_MOVE` 6 Å, `_DOCK_CHARGE_PER_EN` 0.22 e/Δχ, `_DOCK_BOND_ORDER_LENGTH` 0.20 Å,
  `_DOCK_POLAR_H_EN` 2.57, Pauling `_PAULING_EN`, UFF `_UFF_LJ` (x, D), `_ALKALI`/`_ALKALINE_EARTH`/`_HALOGENS`):
  `_dock_formal_charges` (valence-rule formal charges from the 1.25·r_cov graph: alkali +1, alkaline earth +2, lone
  halide −1, N(4)/O(3) +1, sulfonium/phosphonium, amidinium/guanidinium/imidazolium, hydroxide/alkoxide/thiolate,
  C(3)–O(1) > 1.30 Å / C–S(1) > 1.75 Å → −1, shared carboxylate/carbonate/nitrate/nitro/N-oxide/sulfonate/sulfate/
  phosphate/oxo-halide charges, B/Al(4) −1), `_dock_surrogate_terms` (partial charges = formal + 0.22·bond order·Δχ
  per bond; Coulomb 332.0637 q_i q_j / r + UFF 12-6 D[(x/r)¹² − 2(x/r)⁶] with geometric combination over
  inter-fragment pairs; hydrogens on atoms with χ ≥ 2.57 (N, O, S, halogens) carry no 12-6 term; returns None if any
  fragment has a net formal charge or an odd electron count), `_dock_energy_gradient` (r floored at 0.1 Å, analytic
  gradient), `_dock_rigid_field` (least-squares translation + rotation field per fragment), `_dock_rotation`
  (SO(3) exponential and right Jacobian), `_dock_relax` (scipy L-BFGS-B over per-fragment translation and rotation
  vector scaled by the radius of gyration, exact gradient ∂E/∂ω = J_rᵀ Rᵀ τ; maxiter 500, gtol 1e-5), `_dock_start`
  (fragments from `_start_fragments`; force call 1 at the start; rigid-body-projected cosine ≥ 0.5; relaxed pose
  accepted if finite, max atomic move ≤ 6 Å, all-atom rmsd ≥ 0.5 Å; force call 2 at the docked pose, accepted if
  E₁ < E₀, else the start is restored and its cached evaluation re-installed on the wrapper).
- `minimize_func`: `_dock_start(atoms, wrapper)` after `atoms.calc = wrapper`; Sella budget
  `steps = max_force_calls − max(1, wrapper.call_count)`; Sella's first `PES.eval` is served from the ASE cache, so
  undocked starts cost no extra call and docked starts cost exactly one.
- Static checks (`/tmp/dockcheck70.py`, `/tmp/dockcensus70.py`): finite-difference gradient errors ≤ 9e-9 (energy
  and rigid parameters incl. large rotations), rigid placement preserves intramolecular distances to 1e-13,
  formal-charge rules on hand-built graphs (water 0, acetate −1, acetic acid 0, ammonium +1, guanidinium +1,
  urea 0, nitromethane 0, imidazolium +1, DMSO 0, methanesulfonate −1), parity refusal (acetate + water refused,
  acetic acid + water accepted); census on the dataset starts: accepted/refused = 133/79 train, 130/73 valid with
  0 mismatches against the name-based charge classes. pyflakes diff vs champion empty (only the new import).

### Result
- Train 0.6974612 vs 0.7080763 (Δ −0.01062), 469/469 converged, mean rel energy 1.0028371 (vs 1.0014082);
  `evaluation_results/cycle-70-train.json` (0c17fc8183ee496d81bd1ab709de47ea). Valid 0.6973927 vs 0.7210370
  (Δ −0.02364), 465/465 converged, mean rel energy 1.0062907 (vs 1.0078047); `evaluation_results/cycle-70-valid.json`
  (92c7852fa93241f9824b9c261d837ea1; the first valid run returned `status: pending` with an SSH exit-255
  infrastructure error at 44/465 and was repeated with the identical command).
- Controls (`/tmp/dock70cmp.py`): connected systems 0 changed, charged dimers 0 changed on both splits; neutral
  dimers 133 / 130 of which 44 / 43 bit-identical (docking refused by the cosine or move gates or the energy check).
- Train changed 89: 65 better / 20 worse / 4 equal, Σ −318 calls, dC_S −0.01087, Σ drelE +0.67. Same basin (54)
  42 : 9, Σ −193 (dC_S −0.00877); basin moved ≥ 0.3 Å (33) 21 : 11, Σ −100 (dC_S −0.00113, Σ drelE +0.67). By
  start displacement rmsd₀: [0, 0.5) Å (16) 7 : 7 Σ +35 (dC_S +0.0040); [0.5, 1) (36) 26 : 8 Σ −104; [1, 1.5) (19)
  15 : 4 Σ −79; [1.5, 2.5) (16) 15 : 1 Σ −134; ≥ 2.5 (2) Σ −36. By surrogate start energy (`/tmp/dock70feat.py`):
  bound contact E_s < −1 (27) 19 : 4 Σ −49; far/loose [−1, 0.5) (39) 32 : 7 Σ −232 (dC_S −0.0091); compressed ≥ 0.5
  (23) 14 : 9 Σ −37; starts the surrogate ranks below the champion's endpoint (10) 6 : 4 Σ +8. Largest winners
  amines__amines 32 → 8 (rmsd₀ 3.96), amides__benzene 32 → 9, esters__pyrrole 40 → 18 (same basin), pyrrole__pyrrole
  33 → 16, acids__ketones 28 → 12, alkanes__esters 38 → 23 (same basin), pyridine__pyridine 26 → 11, water__water
  25 → 10, pyridine__water 28 → 14 (basin credit +0.277); losers acids__esters 9 → 33 (rmsd₀ 0.24, basin 0.73),
  phenol__pyrrole 22 → 33 (basin 2.38, −0.228), esters__esters 18 → 28 (same basin), alcohols__pyridine 17 → 26
  (same basin), acids__benzene 23 → 32, alkenes__alkenes 11 → 20, alkenes__esters 15 → 23 (rmsd₀ 0.33, same basin).
- Valid changed 87: 68 better / 16 worse / 3 equal, Σ −471 calls, dC_S −0.02370, Σ drelE −0.70. Same basin (47)
  39 : 6, Σ −178 (dC_S −0.00944); basin moved ≥ 0.3 Å (39) 28 : 10, Σ −281 (dC_S −0.01362). rmsd₀ [0, 0.5) (11)
  4 : 7 Σ +47 (dC_S +0.0040); [0.5, 1) (31) 25 : 3 Σ −164; [1, 1.5) (29) 25 : 4 Σ −174; [1.5, 2.5) (12) 10 : 2
  Σ −129; ≥ 2.5 (4) 4 : 0 Σ −51. Bound contact (22) 19 : 3 Σ −102; far/loose (47) 38 : 7 Σ −322 (dC_S −0.0157);
  compressed (18) 11 : 6 Σ −47; surrogate-ranks-start-lower (10) 8 : 2 Σ −38. Winners amides__benzene 39 → 11,
  acids__benzene 37 → 15, alkenes__benzene 31 → 11, ethers__pyridine 30 → 10, alkenes__thiols 26 → 9,
  pyrrole__pyrrole 34 → 18, alkanes__water 30 → 14, ethers__ethers 32 → 17, esters__pyrrole 44 → 29,
  alcohols__benzene 27 → 12 (same basin); losers amides__pyridine 12 → 30 (rmsd₀ 0.23, same basin, compressed start
  E_s +107 kcal/mol at a 1.51 Å N–H···N contact), amines__water 16 → 33 (rmsd₀ 0.38, same basin), benzene__phenol
  15 → 32 (rmsd₀ 0.18, basin 0.62), pyrrole__thiols 20 → 28, alkenes__phenol 22 → 29.
- Energy gate (`/tmp/credits70.py`): train Σ(r−1) +0.660 → +1.331 (pyridine__water +0.277, ketones__ketones +0.227,
  benzene__phenol +0.178, alkenes__water +0.142, ethers__pyridine +0.117 gained; phenol__pyrrole −0.228,
  phenol__pyridine −0.086, pyridine__sulfides −0.083 lost); without monoatomics__pyrrole (+0.788, untouched) the
  margin is now +0.542. Valid +3.629 → +2.925 (acids__acids −0.513, alcohols__alcohols −0.279, amides__benzene
  −0.209, acids__water −0.166, acids__alkanes −0.161, pyrrole__pyrrole −0.148; ketones__water +0.483, ethers__water
  +0.414, alkenes__water +0.146 gained); without carboxylates__ketones (+3.216, a charged dimer, untouched) the valid
  margin would be −0.291.

### Interpretation
- The mechanism does what the hypothesis said and the effect is not a lottery: the same-basin dimers gain −0.0088 /
  −0.0094 per split (42 : 9 and 39 : 6), the gain grows monotonically with the start displacement (≈ −3 calls at
  rmsd₀ 0.5–1 Å, −4…−5 at 1–1.5, −8…−11 at 1.5–2.5, −13…−18 beyond) and with the surrogate's start energy class
  (far/loose starts −6 / −7 calls each on 39 / 47 dimers), and the endgame is unchanged (the docked runs end in the
  same minimum with the same call count once in contact). Valid gained more than train because its neutral dimers
  start farther (rmsd₀ ≥ 1 Å: 45 vs 37) and had more long walkers (30–46 calls).
- Replicated adverse class on both splits: starts already within 0.5 Å rmsd of the endpoint (16 / 11 dimers, 7 : 7
  and 4 : 7, Σ +35 / +47, dC_S +0.0040 each). There the docking move (≥ 0.5 Å by the gate) is larger than the
  distance to the minimum, i.e. the surrogate minimum lies farther from the xTB minimum than the start does; the xTB
  energy still decreases (the start is a perturbed/compressed pose, so the energy check is easy to pass) and the run
  then travels back through a soft region (amines__water 16 → 33 and amides__pyridine 12 → 30 end in the identical
  minimum). The three ≥ +17 losers are a compressed N–H···N start (1.51 Å, E_s +107 kcal/mol), an amine–water pose
  (lone-pair directionality missing from point charges) and a benzene–phenol π contact (no quadrupole beyond the
  C–H bond dipoles); the static residual analysis of the surrogate at the xTB minima (median rigid gradient 0.39 /
  0.36 kcal/mol/Å per atom, 90th percentile 2.4 / 2.1, tail from C–H···O/N contacts at 2.1–2.3 Å where the UFF H
  12-6 wall is repulsive) points the same way: the surrogate's contact geometry is systematically ≈ 0.3 Å too loose
  for xTB's short C–H···O/π contacts and too ideal for lone-pair-directed hydrogen bonds.
- The basin changes (33 / 39 dimers) are net favourable in calls (Σ −100 / −281) and mixed in energy (train +0.67,
  valid −0.70 rel-energy units): docking from far starts lands in the surrogate's preferred contact, not necessarily
  in the reference's basin. The valid margin now rests on the untouched charged dimer carboxylates__ketones
  (+3.216); the train margin is healthy for the first time since cycle 52 (+0.542 without the monoatomics__pyrrole
  credit). Consequence: **ionic complexes must stay outside the docking** (a charged-dimer path change would
  re-roll the +3.2 valid credit and the two train credits).
- Alternatives considered: (i) the gain could come from the symmetry-breaking start being replaced — no, the
  docking acts after `_break_start_symmetry` and the 44 / 43 undocked dimers are bit-identical; (ii) the extra call
  of a refused docked pose is not visible in the counts because refused poses re-install the cached start (the only
  cost is the docked-pose evaluation, which a docked run then reuses as its first point) — verified by the
  bit-identical undocked class; (iii) noise: ±0.005 per split against −0.0106 / −0.0236 with 65 : 20 and 68 : 16
  sign counts.

### Next idea / control
Two bounded follow-ups of the same mechanism, both general: (a) keep near-equilibrium starts where they are by a
physical criterion instead of the move size — the surrogate should only *place* a fragment that is not yet in
contact, so accept the docked pose only when the start is outside the contact range of the surrogate (its
inter-fragment 12-6/Coulomb energy near zero or its rigid xTB force small compared with the surrogate's descent),
or compare the xTB energy of the docked pose with the start's energy *after* the model's own rigid-body relaxation
estimate; (b) use the free xTB forces at the docked pose to seed a secant pair (start → docked pose) so the run
starts with the measured rigid-body curvature instead of the floor. Control for either: the far/loose class must
keep its −6/−7 calls and the undocked class stay bit-identical.

## Cycle 71 — trust-radius credit for an accepted docking move (docked multi-fragment runs start Sella at the fragment cap)

Champion at start: `0cee2f51cc2eb67496f6cd7dd1cea8b8ed9803c8` (train 0.6974612, valid 0.6973927).

### Hypothesis
A docked run (cycle 70) starts Sella at the docked pose with the multi-fragment initial radius δ₀ = 0.1 (max internal
component, Å or rad) and grows it ×1.5 per accepted step to the fragment cap 0.25: the first three steps are limited
to 0.1, 0.15, 0.225. The docked pose is by construction near a surrogate minimum, i.e. 0.3–0.5 Å of rigid-body
residual from the xTB minimum along soft rigid-body modes whose model curvature is the floor, so those first steps are
trust-limited (the model step along a floor mode is far longer than δ) and the residual walk costs three steps where
two would do at the cap. The docking move itself (≥ 0.5 Å rmsd by the gate, xTB energy verified lower) is an accepted
first step of the run in every sense of the trust-region policy (the realised drop is measured, not modelled), so the
radius after it should be what an accepted step earns: the cap. Undocked multi-fragment runs (refused poses, ionic
complexes) and connected systems keep δ₀ and stay bit-identical; the endgame of the docked runs is not touched (its
steps are far below δ). Expected: −1 call on the docked runs whose residual is 0.3–0.5 Å (about half of 89 / 87),
+1 on poses that are already within ≈ 0.1 Å (a 0.25 first step overshoots and is halved) — net ≈ −0.002…−0.004
per split; a small effect, but it is the only remaining bookkeeping inconsistency of the docking mechanism and it
carries no basin re-roll of the undocked classes (the energy gate's charged-dimer credits stay untouched).

Idea-search record (no cycle spent, see backlog `torsional-docking`): the connected analogue of the docking was refuted
statically — with SPICE thermal-snapshot starts the 30–176° rotor turns are not predictable by any classical
surrogate (turns ≥ 60° are uphill on the surrogate, cosine ≈ 0). The docked dimers' endgame length is uncorrelated
with the surrogate's residual (corr 0.16 / −0.03), so the surrogate's accuracy is not the limit either. Charged-dimer
docking was rejected by census: 63 % of the charged starts are within 0.6 Å of their endpoint and only 8–10 per
split are ≥ 1 Å away (gain ≈ 0.004) against a re-roll of the +3.2 valid credit.

### Change (functions)
- `_dock_start`: returns True when the docked pose is accepted (installed, second call spent), None otherwise.
- `minimize_func`: keeps the return value and, after constructing Sella, sets `opt.delta = opt.delta_max_tr`
  (0.25, the multi-fragment cap) for a docked run; nothing else changes.

Candidate: `5dfc93faf96694ea6f3ea1205ea258cc1b936c72` — train 0.6974449 (Δ −0.0000163, below the 1e-4 gate), valid
not run, decision **discard** (preserved in `ideas/docking-trust-credit/5dfc93f…/algo.py`).

### Result
- train `evaluation_results/cycle-71-train.json` (md5 9e0d7a28ccdd1dab571e2138961dc41c): mean rel_steps 0.6974449,
  469/469 converged, mean relE 1.0029, Σ(r−1) +1.347 (champion +1.331).
- Controls exactly as designed: connected (257), charged multi-fragment (79) and undocked neutral dimers (44) are
  bit-identical to cycle 70 (positions and calls). Only the 89 docked runs change (80 of them).
- Docked class: mean calls 17.43 → 17.10, Σ −29 calls; histogram of Δcalls: 30 runs faster (Σ −69, mean ref 48.3),
  43 unchanged, 16 slower (Σ +40, mean ref 36.6). In score the two halves cancel exactly (−0.00385 vs +0.00384):
  the winners are long H-bonded walkers with large references (esters__ethers 29 → 23, esters__esters 28 → 21,
  alcohols__pyridine 26 → 19, alcohols__pyrrole 33 → 28, acids__amines 21 → 17, acids__acids 21 → 18), the losers
  are weakly bound or near-converged poses with small references (acids__esters 33 → 44 with ref 14, +0.79 rel;
  benzene__phenol 8 → 13 with ref 17, +0.29; alkanes__ketones 13 → 19, +0.22; benzene__ethers 13 → 16, alkenes__water
  24 → 27). Three molecules carry +0.0028 of the +0.0038 loss.
- By start displacement: rmsd₀ [1.0, 1.5) Å gains −23 calls (13 faster / 4 slower); the other bins are ±3. By the
  surrogate's residual field at the xTB minimum: residual < 0.25 gains −26 (14/2), 0.5–1.0 loses +14 (2/6).
- Basin changes among docked runs: 2 (amides__pyridine same calls, same energy, geometry 0.1 Å off; pyridine__sulfides
  same calls, −0.25 kcal/mol, relE 0.917 → 0.934), energy credits +0.017.

### Interpretation
The mechanism does what it was built for — three trust-limited residual steps become one for the H-bonded docked
poses whose rigid-body residual is 0.3–0.5 Å — but the docked poses are bimodal: the weakly bound (dispersion,
π-stacked) and the near-converged poses have a residual of ≈ 0.1 Å, where δ₀ = 0.1 lands the first step by luck and
a 0.25 step (a 14° fragment rotation) overshoots the flat minimum, is halved and re-learnt over 3–6 further calls.
Because the near-converged poses are also the molecules with small references, the calls saved (−29 net) are worth
nothing in the score. A per-run credit would need the residual scale at the docked pose, i.e. a curvature estimate
along the rigid-body modes from the same surrogate that left the residual there — comparable crude models, no
bounded repair is warranted (the perfect discriminator is worth −0.0039). The docking bookkeeping line is closed:
credit for the docking move is neutral, and the secant-seeding variant (a chord over ≥ 0.5 Å along modes whose
curvature changes along the chord) has the same ≤ 0.004 upside with a poisoning risk. Rejected.

### Next idea / control
The remaining cost map is unchanged (docked-dimer endgame 17 calls/run, near-equilibrium connected endgame
8–11 calls at contraction ≈ 0.5, rotor-turn walks 12 calls/Å). Candidates: a fragment rigid-body curvature model
that replaces the 1e-3 floor by a physical estimate would only pay if it is right to better than 2× on the soft
rotations (the inertia-scaled floor was "mild"); the connected endgame is learning-limited (one direction per call)
and the model-Hessian calibrations are exhausted at the class level. Control for any dimer change: connected and
charged classes bit-identical; docked class judged by the winners/losers split above.

## Cycle 72 — ionic hydrogen-bond cap: contact bends of inter-fragment X–H···Y / X–H···π contacts with a charged fragment get twice the neutral cap

Champion at start: `0cee2f51cc2eb67496f6cd7dd1cea8b8ed9803c8` (train 0.6974612, valid 0.6973927).

### Hypothesis
The contact bends (`_h0_contact_bends`, cycles 45/48/49/62/63) give an inter-fragment hydrogen bond its sideways
curvature with a contact factor capped at the value of an *equilibrium neutral* hydrogen bond (bo_cap 0.05 for
N/O/S/halogen donors, calibrated on the water dimer's harmonic intermolecular frequencies at c = 0.027; the cap is
1.85× that). Ionic hydrogen bonds — a charged donor fragment (ammonium, guanidinium, imidazolium N–H⁺···O/N/S/π) or a
charged acceptor fragment (O–H···O⁻ carboxylate, X–H···Cl⁻/F⁻) — bind 3–4 times more strongly than neutral ones
(NH₄⁺···OH₂ 20 kcal/mol, CH₃COO⁻···HOH 16, Cl⁻···HOH 14, NH₄⁺···benzene 19, against 5 for the water dimer), and the
angular stiffness of an electrostatically held contact scales with its energy: the ion's field holds the neutral
partner's orientation with μE = μq/R² ≈ 0.02–0.05 Ha/rad² (water 1.85 D at 2.8 Å 0.026, ketone 0.04, amide 0.05),
against the model's acceptor-side total of 0.015 at the cap, and the donor's swing off the H···Y⁻ axis scales the same
way. The shorter contact of an ionic hydrogen bond (1.6–1.8 Å) already lifts the contact factor to the cap, i.e. to
1.85× the neutral calibration; the remaining factor ≈ 2 is what the cap forbids. Charged-dimer endgames cost the same
13–14 calls as neutral ones near equilibrium (cycle-72 census) although their reference is faster (pseudo-bond
model), and the acceptor-side rotations of the neutral partner are exactly the soft modes the update has to learn.
Change: for inter-fragment contacts of heteroatom donors (`strong`), when the donor's or the acceptor's fragment
carries a net formal charge (valence rules, `_dock_formal_charges`, per fragment of the covalent graph), the cap
(bo_cap, and bo_cap_weak for the π contacts) is doubled — one a-priori factor, not scanned. C–H donors, metal-ion
contacts (no bends), intramolecular contacts and the radial term are unchanged, so connected systems, neutral dimers
and the energy-gate credit walkers without heteroatom-donor contacts are bit-identical: monoatomics__pyrrole (+0.788),
carboxylates__ethers (+0.356), ketones__monoatomics (+0.233), carboxylates__ketones (train +0.199 / valid +3.216),
ethers__monoatomics (+0.852) all keep their runs (census `/tmp/ionhb72.py`: train 54 affected charged systems, share
of score 0.090, untouched credits +1.233; valid 57 affected, share 0.107, untouched credits +3.872). Expected
−0.003…−0.006 per split on the cation dimers (41/42), ion pairs (1/1), anion dimers with heteroatom donors (7/7)
and halide dimers (5/7); walkers (amides__guanidiniums 40, ammoniums__esters 48, imidazolium__ketones 36/33,
acids__guanidiniums 32, ammoniums__benzene 30, carboxylates__thiols 32) may re-roll — their credits sum to +0.17 /
+0.34, far inside the margins.

### Change (functions)
- `_fragment_charges` (new, Internals): net formal charge per fragment of the covalent graph from
  `_dock_formal_charges` at the geometry of the first call, cached with the graph key like `_covalent_graph_cache`.
- `_h0_contact_bends`: new parameter `ion_cap = 2.0`; for `inter[p]` contacts with a heteroatom donor whose donor or
  acceptor fragment has |q| > 0.5, `cap *= ion_cap` before `c = min(bo, cap) * scale` (applies to the bend at H, the
  acceptor bends, the lone-pair plane term and the π-face contacts alike).

Candidate: `6361324ae524526681fa99ba3f032c1e56b4cdae` — train 0.7012080 (Δ +0.0037467), valid not run, decision
**discard** (preserved in `ideas/ionic-hbond-cap/6361324…/algo.py`).

### Result
- train `evaluation_results/cycle-72-train.json` (md5 52f5246b2e6306330d7d38fa51a5fc9a): mean rel_steps 0.7012080,
  469/469 converged, mean relE 1.0028, Σ(r−1) +1.311 (champion +1.331; guanidiniums__pyrrole −0.020).
- Controls as designed: all 390 non-charged systems bit-identical except guanidiniums__water (a charged system that
  the census's 1.25-radius fragment test merges into one fragment; 14 → 14 calls, same minimum); the 25 charged
  systems without heteroatom-donor contacts (all the large credits) bit-identical.
- Affected charged systems (54): calls Σ +17, score +0.0037 = cation +0.0044, ion pair −0.0010, metal +0.0002,
  anion +0.0001. Same-basin (49): Σ 0 calls, 18 faster / 16 same / 15 slower (score +0.0003); by contact kind:
  X–H···Y 41 dimers Σ −8 (16/13/12), X–H···π 6 dimers Σ +8 (2/1/3). By acceptor type, same-basin: carbonyl O
  (single-neighbour, lone-pair term) Σ +3 over 16, two-neighbour O/N/S and halides Σ −3 over 27 — no sub-class moves.
  Basin changes 5: imidazolium__ketones 36 → 72 (ref 23, −0.10 kcal/mol, +0.0033 of the loss on its own),
  ammoniums__esters 48 → 55, amides__ammoniums 25 → 30, guanidiniums__pyrrole 35 → 10 (ref 65, +0.73 kcal/mol,
  relE 1.000 → 0.980), alcohols__imidazolium 25 → 19.
- Only contacts with bo > 0.05 (excess < 0.82 Å, i.e. H···O < 1.8, H···N < 1.85, H···S < 2.2 Å) feel the doubled
  cap; 36 of the 54 affected systems have such a contact at the endpoint (c 0.05 → 0.07–0.10, ×1.4–2).

### Interpretation
The ionic angular stiffness is not what limits the charged-dimer endgames: doubling the sideways curvature of the
ionic contacts (the physically estimated factor for the neutral partner's ion–dipole orientation) leaves the
same-basin cost exactly unchanged (Σ 0 over 49 runs, ±3 calls of per-run noise in both directions, no acceptor or
donor sub-class with a signal), and the score is decided by one walker re-roll (imidazolium__ketones +36 calls to a
minimum 0.1 kcal/mol lower). This matches the earlier reading that walkers creep along model-invisible modes (twist
about the contact axis, internal rotors of the neutral partner) rather than along the contact bends, and that the
update learns a 2× error on a mode that already has 15–25× the floor within the first steps. The reference's speed on
near-equilibrium cation dimers (9–15 calls) therefore does not come from its stiffer contact bends (its pseudo-bond
model gives them covalent-strength angles, 5–10× ours) but from its coordinate system. No bounded repair: a
different factor is a scan and a re-roll of the same walkers; an acceptor-only or hb-only variant would carry the
same ±0 same-basin signal. Ionic-contact stiffness line closed (see backlog `ionic-hbond-cap`).

### Next idea / control
The charged class's excess over its reference is a coordinate-system effect for the cation dimers (the reference
describes the contact by a pseudo-bond with bends and dihedrals across it, and its TS-BFGS update learns in those
coordinates; ours describes it by fragment rotations/translations plus the contact block). Candidate to examine
statically before any cycle: adding the contact's own bend/dihedral coordinates (H···Y–Z, X–H···Y–Z) as *redundant*
internals for strong inter-fragment contacts — the model curvature would then be learned along the contact
coordinates, which stay meaningful while the fragments librate, instead of along rigid-body rotations whose
projection on the contact changes with the geometry. Otherwise the cost map is unchanged: docked-dimer endgame,
near-equilibrium connected endgame, rotor-turn walks.

## Cycle 73 — per-eigenmode step caps in the max-internal-component trust region (connected systems; the Levenberg shift replaced by a box-constrained model step)

Champion at start: `0cee2f51cc2eb67496f6cd7dd1cea8b8ed9803c8` (train 0.6974612, valid 0.6973927).

### Hypothesis
The restricted step (`MaxInternalStep`, `rs='mis'`) bounds the largest weighted internal component of the step
(and, for connected systems, the largest linearised atomic displacement / cart_ratio) by the radius δ, but it
realises the bound with the Levenberg shift of the 2-norm trust region: s(α) = −Σ_i v_i (v_i·g)/(λ_i + α), with one
global α found by a root search on the max component. The shift damps *every* mode by λ_i/(λ_i + α), so while the
walk's leading coordinate is trust-limited (first 3–4 steps of the 155/257 connected far starts of the train split,
rmsd₀ ≥ 0.3 Å, 19.4 calls vs 25.9 reference, dominated by 30–178° turns of 2.3 acyclic rotors per molecule), every
other soft coordinate is damped as well: with a torsional mode of λ ≈ 0.01–0.02 Ha/rad² setting α ≈ 0.005–0.03, the
other rotors (λ 0.01–0.02) lose 20–70 % of their Newton amplitude, bends (0.1–0.2) 5–20 %, stretches 1–5 %, and the
softest modes (λ ≤ 0.005) creep at g_i/α, i.e. steepest-descent speed. The damped coordinates are exactly the
model-reliable ones for which the Newton amplitude was already within the box, so the damping buys nothing in
robustness — it is the geometry of the 2-norm ball applied to a box constraint. The cycle-28 gain of δ₀ 0.1 → 0.25
(−0.02) was a less-damping gain of the same kind. The box-constrained model minimisation is approximated per
eigenmode: every eigenmode v_i of the projected model gets the cap a_i = δ / max(m_i, c_i), with m_i the largest
weighted internal component of the unit mode u_i = Ufree v_i and c_i the largest atomic displacement of B⁺u_i over
cart_ratio; the mode's Newton amplitude is clipped to ±a_i, non-limiting modes keep their Newton amplitude. When
the composed step still violates the bound (several modes compounding on one coordinate or one lever arm), all caps
are scaled by a common factor t ∈ (0, 1) found by the existing root search (the stepper's derivative comes from the
clipped modes only; bisection after 4 Newton iterations, as for RFO). The Newton step is checked first without
caps, so steps that fit the trust region are bit-identical, and the caps are only applied for connected systems
(`mode_caps_mol`), so every multi-fragment system — including all energy-gate credit walkers — is bit-identical.
Expected: fewer far-start steps and a better-conditioned secant memory (steps span the soft modes instead of the
leading one), 1–2 calls on a third of the far starts, −0.01 to −0.025 on the train score; a failure mode is that
greedy full-cap steps along barely driven soft modes (the box solution takes the whole cap wherever the model
predicts a decrease) overshoot and trigger radius shrinks.

### Change (functions)
`CappedQuasiNewton` (new stepper: plain Newton step of the projected model without caps; with caps the modes with
|a_i| > t·cap_i are clipped, derivative from the clipped modes, alpha0 = 1, alphamin 0, alphamax 1, slope +1,
newton_safe False); `MaxInternalStep.__init__` (`mode_caps` flag selects the stepper for method 'qn'),
`MaxInternalStep.get_s` (uncapped Newton step first; caps only when it violates the bound),
`MaxInternalStep._mode_caps` (δ / max(m_i, c_i) from Ufree @ V and Binv); `_default_kwargs['minimum']['mode_caps_mol']`,
`Sella.__init__` (`self.mode_caps_mol`), `Sella._predict_step` (`rs_kwargs['mode_caps']` for order-0 connected
systems only).

### Result
Candidate `3fc0b796dd8b733321ee5609fb9c55768621d74c`: train 0.6975840 vs 0.6974612 (+0.00012, below the gate) →
discard; mean relE 1.00189 (champion 1.00284), Σ(r−1) +0.886 vs +1.331. Anchor restored; implementation kept in
`ideas/per-mode-caps/3fc0b796dd8b733321ee5609fb9c55768621d74c/algo.py`.

### Interpretation
Controls as designed: all 212 multi-fragment runs bit-identical, 39 connected runs whose steps always fit the trust
region bit-identical. Connected 257: 54 faster / 143 same / 60 slower, Σ −33 calls, but the whole gain is in 10
basin changes (Σ −68, e.g. 104079126 56 → 24 into a 22.8 kcal/mol *higher* minimum that matches the reference —
the champion had found a rearranged lower one, relE 1.022 → 1.000; 134997543 46 → 30 at +2.7 kcal/mol); the
same-basin effect is +35 over 247 (46 faster / 143 same / 58 slower), +0.3 calls per molecule for rmsd₀ 0.15–1.0 Å
and every size class below 35 atoms, only the very far starts (rmsd₀ ≥ 1 Å, n 25) gain (13:7, Σ −18). Two hops
cost the energy credit: 172113611 (C9H16N4O8) unfolded an intramolecular contact (O···H 5.0 → 9.4 Å, +48.8
kcal/mol, relE 0.793) and O4P2 lost a P–O bond (O–P 1.87 → 3.74 Å, +67 kcal/mol, relE 0.828) — bond breaking
from 0.25 Å per-mode caps compounding over successive steps along the row-factor-softened P–O stretches. The
mechanism is what the hypothesis flagged as its failure mode: the box step is the exact solution of the box trust
region for a diagonal model and therefore greedy — every mode with g_i/λ_i above its cap takes the whole cap,
however small its force, whenever the model curvature is small; the Levenberg shift gives such modes g_i/α
(steepest-descent amplitude, proportional to the force). The Tikhonov damping of the shift is a regulariser against
guess-curvature error, and the update learns the leading mode's curvature better from steps concentrated on it
than from steps spread over many capped modes. The far-start damping loss (rotors, bends) that the hypothesis
priced at 1–2 calls does not exist at the level of the same-basin statistics; cycle 28's δ₀ gain was about the
*leading* mode's cap, not about the damping of the others. Bounded repairs (gradient thresholds for which modes
may take their cap; box only for the leading modes) are scans of a constant or equivalent to the shift; a
principled per-mode trust criterion (box amplitude only inside the span of the measured secant pairs) would leave
the first — largest — far-start steps shift-damped and is not worth a cycle. Line closed (backlog `per-mode-caps`).

### Next idea / control
The step-composition levers of the connected walk are exhausted (trust radius policy, δ₀, Cartesian bound,
curved step, retrospective radius, step rejection, per-mode caps). What remains for connected systems is the
model itself along the rotor paths (closed after cycles 68/69) and the endgame (update family closed). The
remaining structural excess is the multi-fragment endgame (≈ 8 calls over the connected regression, flat
inter-fragment modes); the cycle-72 follow-up — redundant contact bend/dihedral internals across strong
inter-fragment contacts so that the update learns the contact curvature in coordinates that stay meaningful while
the fragments librate — is the next candidate to examine statically.

## Cycle 74 — surrogate docking extended to molecular-cation complexes (fragments with net formal charge +1 on a polyatomic ion; anions and bare ions stay undocked)

Champion at start: `0cee2f51cc2eb67496f6cd7dd1cea8b8ed9803c8` (train 0.6974612, valid 0.6973927).

### Hypothesis
Cycle 70's rigid-body docking (`_dock_start`) is refused for every start with a charged fragment. The census of the
charged multi-fragment classes on the champion's results (formula values at the saved start and final geometries
only) separates them: (i) 47 / 45 complexes of a *molecular cation* (ammonium, guanidinium, imidazolium) with a
neutral partner — 19.4 / 18.9 calls vs 30.1 / 28.1 reference (share 0.077 / 0.083 of the score), start
displacement from the champion's endpoint rmsd₀ 0.72 / 0.74 Å on average with 27 / 27 starts beyond 0.5 Å (the
docked-neutral bins gave −3 calls at 0.5–1 Å, −4…−5 at 1–1.5, −8…−11 at 1.5–2.5), closest inter-fragment
heavy-atom contact 3.2–5.9 Å at the start vs 2.7–2.9 Å at the end; energy-gate credits only +0.222 / +0.063;
(ii) 15 / 15 molecular-anion complexes (share 0.03, credits +0.035 / +3.354 — the validation margin, +2.925,
rests on carboxylates__ketones +3.216 alone); (iii) 15 / 12 complexes of a bare ion (K⁺/Na⁺/Li⁺, halides;
credits +0.956 / +0.811, near-equilibrium starts rmsd₀ 0.1–0.8 Å); (iv) 2 / 1 salt bridges (near starts). The
surrogate itself is as good for the cation complexes as for the neutral dimers: its rigid-body residual force
at the champion's xTB minima (surrogate terms rebuilt with the charge refusal removed) has median 0.53 / 0.39
kcal/mol/Å per atom, 90th percentile 1.17 / 0.90 (neutral dimers 0.40 / 0.37, 90th 2.45 / 2.10), and it pushes
the fragments apart at the xTB minimum by a median +1.6 kcal/mol/Å (neutral +2.7) — the 12-6 wall, not the
Coulomb pull, sets the docked contact, so charged N–H···O/N contacts are not over-compressed; the starts are
farther in the surrogate's own terms (E_s(start) − E_s(final) median 3.7 / 3.2 kcal/mol vs 2.1 / 1.6). The
non-polarisable model is *not* adequate for bare ions: the UFF 12-6 radii are neutral-atom radii, so K⁺···O has
its minimum at 3.65 Å against a 2.7 Å ionic contact and Cl⁻···O at 3.7 vs 3.1 Å, and the ion's polarisation of
the partner (cation–π, lone-pair coordination) is absent; these stay on the plain path. Molecular anions are
physically no worse than cations (same H-bond wall), but the class is small (potential ≈ 0.002 per split) and
carries the whole validation energy margin in one lottery molecule, so they stay undocked by scope; the same
holds for the salt bridges (a negative fragment). The rule is therefore: dock when every fragment has a
non-negative net formal charge, every charged fragment is polyatomic, and no fragment has an odd electron count;
all gates and constants unchanged (rigid-body cosine ≥ 0.5 with the first xTB force, docked move ≥ 0.5 Å rmsd,
≤ 6 Å max atomic move, xTB energy decrease, cached start evaluation). Expected: the 27 / 27 cation starts beyond
0.5 Å behave like the docked neutral dimers (−3 to −10 calls each, −0.005 to −0.007 per split), the 20 / 18
near-equilibrium starts replicate the small neutral loss (+0.004 per split at most); energy gate: the cation
class can lose at most its +0.22 / +0.06 credits and gains/loses new ones at the neutral docking's rate (train
−0.54 / +1.21 over 93 docked runs, valid −1.82 / +1.12 over 88) — well inside the +1.331 / +2.925 margins whose
carriers (monoatomics__pyrrole, carboxylates__ketones) are untouched; every neutral, anionic, bare-ion and
connected run must be bit-identical.

### Change (functions)
`_dock_surrogate_terms` (refusal rule: a fragment with a negative net formal charge, a charged single atom, or an
odd electron count returns None; a polyatomic cation is accepted), docstrings / rationale comment of the docking
block, `_dock_start` and the `minimize_func` comment updated. No constant changed.

### Result
Candidate `4d19b2e6863c0e4a844422976855666bbba88bc3`: train 0.6965562 vs 0.6974612 (−0.00091, passes the score gate)
but **invalid**: mean relE 0.99896 (Σ(r−1) −0.487 vs +1.331), energy_below_baseline → recorded as invalid.
Implementation preserved in `ideas/cation-docking/4d19b2e6863c0e4a844422976855666bbba88bc3/algo.py`; anchor restored.

### Interpretation
Controls as designed: 257 connected, 133 neutral, 15 anion, 2 salt-bridge and 15 bare-ion runs bit-identical. The
47 cation complexes: 16 refused by the gates (bit-identical), 17 docked into the same basin (Σ −8 calls; rmsd₀
< 0.5 Å +9 on 5 as in the neutral class, 0.5–1.5 Å −18 on 11), 14 basin changes (Σ −51 calls) with Σ credit
−1.818. By cation type: ammonium 16 (6 refused) −18 calls, credit −0.013; guanidinium 15 (7 refused) +14 calls
(acids__guanidiniums 17 → 39 into a 0.3 kcal/mol lower minimum), credit +0.008; **imidazolium 16 (3 refused)
−55 calls but credit −1.812** — seven docked imidazolium runs converged to minima 4.6–44.8 kcal/mol *above* the
champion's/reference's (amines__imidazolium +44.8, imidazolium__water +21.0, imidazolium__ketones +19.9,
alkenes__imidazolium +15.1, benzene__imidazolium +6.2, imidazolium__thiols +4.8, imidazolium__phenol +4.6). The
bond graphs are unchanged; the geometries show the cause: the champion's minima are the charged hydrogen bonds
N–H···N (amine N···H 1.41 Å, a nearly shared proton) / N–H···O (1.62–1.68 Å), while the docked runs end with the
partner's nucleophilic atom over the imidazolium ring carbon and no hydrogen bond at all (amine N···C2 2.28 Å — a
dative σ-adduct-like stationary point, water O···C 2.67, ketone O···C 2.65). The surrogate put the formal +1 on
the amidinium carbon C2 and the bond-polarity transfer left it at +0.70 e with the N–H protons at +0.19 e (no
more than an amide's), so the point-charge model draws the nucleophile onto the exposed ring carbon; from that
stacked pose xTB descends into the adduct minimum. Ammonium (N buried under its protons) and guanidinium (carbon
shielded by three NH₂) are not exposed the same way, and their docked runs behaved like neutral dimers. The
failure is therefore an error of the surrogate's charge placement for delocalised cations, not of the docking
mechanism (call gains −55 / −18 on the imidazolium / ammonium classes), and it is localised → bounded repair:
place the charge of a cationic centre on the hydrogens around it (ESP-fitted charges of ammonium, guanidinium
and imidazolium put +0.3–0.45 e on the N–H / ring C–H and leave the central atom neutral or negative). Ammonium
and guanidinium poses will re-roll under the repaired charges (expected neutral-like), the imidazolium poses
should become hydrogen-bonded.

### Next idea / control
Repair 1 (cycle 75): the formal charge of every cationic centre (metal ion, onium N/O/S/P, amidinium-type C) is
shared equally among the hydrogens bonded to the centre or to one of its neighbours before the bond-polarity
transfer (imidazolium: C2 +0.28, N −0.52, N–H +0.52, C2–H +0.41; guanidinium N–H +0.38, C +0.54; dimethylammonium
N −0.59, N–H +0.31, C–H +0.20). Neutral fragments contain no cationic centre in either split (their pair terms
are bit-identical to the champion's, checked statically), so every non-cation run stays a bit-identical control.

## Cycle 75 — repair 1 of the cation docking (cycle 74): the charge of a cationic centre is placed on its surrounding hydrogens

Champion at start: `0cee2f51cc2eb67496f6cd7dd1cea8b8ed9803c8` (train 0.6974612, valid 0.6973927).

### Hypothesis
Cycle 74's docking of polyatomic-cation complexes gained calls (imidazolium −55, ammonium −18 on 47 train
complexes) but seven docked imidazolium runs ended in nucleophile-on-ring-carbon minima 5–45 kcal/mol above the
reference's hydrogen-bonded ones, because the surrogate carried the formal +1 on the amidinium carbon (+0.70 e
after the polarity transfer) with amide-like N–H protons (+0.19 e). ESP-fitted charges of onium and amidinium
ions put the positive charge on the peripheral hydrogens (+0.3 to +0.45 e on the N–H and ring C–H) and leave the
central atom neutral or negative. Sharing the formal charge of every cationic centre equally among the hydrogens
bonded to it or to one of its neighbours (then the unchanged polarity transfer) gives imidazolium C2 +0.28, N
−0.52, N–H +0.52, C2–H +0.41; guanidinium C +0.54, N −0.5, N–H +0.38; dimethylammonium N −0.59, N–H +0.31, C–H
+0.20 — the nucleophile's best site becomes the N–H proton (no 12-6 wall, contact set by the heavy-atom wall at
N···O ≈ 2.5 Å against xTB's 2.7), i.e. the reference's basin. Expected: imidazolium runs keep most of their call
gain without the adduct minima; ammonium and guanidinium poses re-roll under the stronger proton charges with a
neutral-like outcome; energy credits of the cation class within ±0.3 of the champion's +0.222; all 422 non-cation
train runs bit-identical (neutral fragments contain no cationic centre in either split — pair terms checked
statically against the champion's, identical for all 133 / 130 neutral dimers). Score expectation −0.002 to
−0.004 per split.

### Change (functions)
`_dock_formal_charges` (returns `(q, onium)`; the metal-ion, onium N/O/S/P and amidinium-carbon branches record
their charge in `onium` and the charges are added at the end — `q` itself unchanged), `_dock_surrogate_terms`
(after the refusal checks the charge of every cationic centre is moved onto the hydrogens bonded to the centre or
to its neighbours, equal shares; centres without such hydrogens keep it), rationale comment. Cycle-74 scope rule
kept (polyatomic cations docked; anions, salt bridges, bare ions undocked); constants unchanged.

### Result
Candidate `df252caf787d3e1cbc8a25481589415f92ed9310`: train 0.7008388 vs 0.6974612 (**+0.00338**, fails the score
gate); valid: mean relE 1.00297, Σ(r−1) +1.393 (champion +1.331), 469 converged → recorded as discard. Implementation
preserved in `ideas/cation-docking/df252caf787d3e1cbc8a25481589415f92ed9310/algo.py`; anchor restored.

### Interpretation
The energy-gate defect is repaired: all seven imidazolium adduct minima are gone (imidazolium credits Σ 0.000, the
docked imidazolium runs end hydrogen-bonded like the reference; class −31 calls on 16, e.g. imidazolium__water 28 →
10, amines__imidazolium 23 → 15) and ammonium gains −11 (ammoniums__esters 48 → 37, ammoniums__water 32 → 23), but
the guanidinium class loses +41 and the whole +0.00338 is three basin changes plus three near-equilibrium re-rolls:
esters__guanidiniums 11 → 34 (rmsd₀ 0.48 Å; the docked pose keeps the champion's H7/H8···O13 bidentate contact but
the *ester* relaxes into another conformer, fragment-internal rmsd 0.46 Å, minimum 0.75 kcal/mol lower),
acids__guanidiniums 17 → 30 (acid conformer change 0.32 Å, −0.29), amides__ammoniums 25 → 47 (amide 0.40 Å,
−0.21), guanidiniums__ketones 13 → 21 (symmetric-equivalent pose), amides__imidazolium 21 → 30 and
guanidiniums__sulfides 12 → 19 (starts 0.2–0.6 Å from the endpoint, docked ≥ 0.5 Å away and walked back). All
422 non-cation runs bit-identical; 20 cation runs refused (17 within 1 Å of the endpoint); the 19 same-basin docked
runs gain Σ −33 (rmsd₀ 0.5–1 Å −31 on 13, > 1 Å −15 on 3), the 8 basin changes Σ +32 with credit +0.062. So the
cation docking works where the neutral one does (far starts, unique H-bond pose: imidazolium, ammonium) and fails
where the neutral one fails (starts within ≈0.5 Å of the endpoint) plus a class of its own: the multi-donor
guanidinium (six equivalent N–H) has several near-degenerate bidentate poses on the surrogate, the L-BFGS descent
picks one by the start's orientation, and from that pose xTB relaxes the *partner's* conformation (ester syn/anti,
acid O–H, amide C–N rotation) — a lower minimum reached by a long walk. None of the losses is a charge-placement
error (the surrogate charges are now ESP-like) and no gate separates them from the gains without a scan of the
0.5 Å move gate or a class rule, so the bounded repair stops here (2 of 4 candidates used): cation docking is a
wash (imidazolium/ammonium −0.0022 against guanidinium/near-equilibrium +0.0056 on train) → line closed.

### Next idea / control
No repair 2: the remaining losers are basin re-rolls (the neutral docking's known adverse class) and multi-pose
guanidinium starts, neither addressable without molecule-class rules or gate scans. The cation-docking entry in
`backlog.md` records the finding (dock only far, single-pose cation starts — not separable a priori) and both
implementations. Next: a new line from the remaining cost map (dimer endgame / large-amplitude rotor paths).

## Cycle 76 — symmetry-image secant pairs: the multi-secant memory is completed with the images of every stored pair under the point-group operations detected at the current geometry (connected systems)

Champion at start: `0cee2f51cc2eb67496f6cd7dd1cea8b8ed9803c8` (train 0.6974612, valid 0.6973927).

### Hypothesis
The endgame of a connected molecule costs ≈ k + 2 calls for k poorly modelled directions because every step's
secant pair teaches the model one direction (cycles 53/56: memory beyond four pairs is refuted, so the learning
rate is one direction per force call). xTB's energy is exactly invariant under permutations P of identical
nuclei and under every orthogonal map R of the coordinates (reflections included), so whenever the current
geometry is itself (approximately) invariant under some (R, P) — a point-group operation — every measured
secant pair (s, y) has an *exact* image pair under that operation: in redundant internals the image is the signed
permutation (DΠ s, DΠ y), Π the permutation of the coordinate list induced by P (bond (i,j) → (Pi,Pj) …) and D
= det R on the dihedrals (pseudo-scalars). The images are secant pairs of the PES displaced from the trajectory by
the symmetry residual ε (per-atom rmsd of R·P·x − x), i.e. their curvature error is ≈ 2aε (a ≈ 2 /Å, the stretch
anharmonicity; less for bends) — below the guess error (20–100 %) of any direction the real pairs have not
sampled, which is where the Gram–Schmidt filter lets them in (real pairs keep precedence). A step with components
in all irreps (the noisy starts guarantee that) therefore teaches up to |G| directions per call. This differs
from the refuted class-shared secant scaling (cycles 30/31): the images are exact, not analogies. Census on the
champion's endpoints (`/tmp/symdet76.py`, rmsd tolerance 0.05 Å, improper operations included): 99 train / 49
valid connected molecules end at a symmetric geometry (share 0.165 / 0.081 of the score, 12.3 vs 16.1 and 13.3
vs 17.9 calls; 22 / 7 of them planar only — the σ_h image separates in-plane from out-of-plane curvature), 42 / 15
are symmetric at the start (tiny molecules, 8.5 vs 11.1 calls), 57 / 34 become symmetric only during the run
(re-detection at every update catches their endgame); false symmetry at the start 0 / 1. Expected: −1 to −2
calls on the symmetric-final class, ≈ −0.008 … −0.015 train and −0.003 … −0.006 valid; multi-fragment systems and
asymmetric molecules bit-identical (the search returns nothing for them).

Tolerance, fixed a priori: an operation is used when ε ≤ max(0.05 Å, ½·rms atomic step just taken): at 0.05 Å
the worst stretch error is ≈ 20 % (bends a few %), and an image displaced by less than half the step is at
least as accurate as the real pair itself, whose secant averages the Hessian over the whole step. Permutations
are found by backtracking over same-element atoms with distance pruning (|d_ij − d_{Pi,Pj}| < 3ε_max), the
orthogonal map by a Kabsch fit in both determinant branches.

Image map (revised during implementation): Sella's coordinate list is *not* closed under the point group — a
single improper dihedral is added at a symmetric centre without proper dihedrals (SiHI3, PCl5, BF3-like
molecules, exactly the tiny inorganic class) — so the signed permutation drops the operations of 17 of the 96
symmetric train endpoints (228 of 403 operations survive). The image is therefore formed through Cartesian
space with the Jacobian of the current geometry: s' = B G B⁺ s, y' = B⁺ᵀ G Bᵀ y (atoms only; dummies carry no
force), G the permutation–rotation of the extended displacement field. This equals the signed permutation
whenever the list is closed and is the exact first-order image otherwise (offline check on the champion's
endpoints, `/tmp/symimg76b.py`: 390 of 403 operations kept, finite-difference vs linear image ≤ 2.6 % of the
displacement, gradient map exact). Dummy atoms of linear bends: the image dummy must land on the dummy itself
(u' = R u) or on its inversion through the centre (u' = 2R u_centre − R u; both satisfy the dummy's bond and
90° angle constraints), otherwise the operation is dropped — 13 operations on 6 molecules (chains on a rotation
axis, whose dummy azimuth has no symmetry-related partner). Cost: ≤ 40 ms per update on the largest molecules.

### Change (functions)
New module-level `_isometric_permutations` (backtracking search, node and solution caps) and
`_symmetry_operations` (Kabsch fit, both determinants, identity kept only as a reflection);
`Internals.symmetry_maps(tol)` (operations as extended Cartesian maps with the dummy permutation and inversion
flags; connected systems only — refused when translations/rotations exist); `InternalPES._symmetry_images`
(tolerance from the Cartesian step since the previous update, image maps s' = B G B⁺ s / y' = B⁺ᵀ G Bᵀ y built
from `int.jacobian()` and `_get_Binv()`); `PES._update_H` / `PES._collect_secant_pairs` take an `images` list
and enter, after every stored real pair, its images (s, y and the transport term tbar mapped alike), subject to
the unchanged dependence (0.3) and consistency (0.25) filters; `InternalPES._update_H` computes the images at
the current geometry before the transport; `InternalPES.__init__` records the geometry for the step measure.
Memory of real pairs (4) and all constants unchanged.

### Result
Candidate `1ba6177625d7e7ff315c5995188815383eda7a4b`: train **0.6868163** (−0.0106 vs 0.6974612), valid **0.6924513**
(−0.0049 vs 0.6973927); both valid (mean rel_energy 1.00284 / 1.00629; credit sums 1.3306 → 1.3306 and 2.9252 →
2.9252), all converged. Decision: **keep** (`python scripts/cycle.py record` → accepted). New champion 1ba6177.

Controls: every train dimer bit-identical (219/219); 4 valid dimers changed (alkanes__monoatomics 21 → 19,
acids__pyrrole 21 → 19, benzene__monoatomics 9 → 9, monoatomics__thiols 20 → 20) — Sella's bond detection joins
them into one fragment, so they are "connected" for the mechanism; asymmetric connected molecules identical
(174/250 and 215/250 unchanged; the only changed molecules without symmetry at the start or the champion's
endpoint pass through a symmetric geometry mid-run: 135121768 16 → 14, 135011106 24 → 19).

Symmetric-final class (`/tmp/cmp76.py`; operations at the champion endpoint, tol 0.05): train 96 molecules 1160
→ 1090 calls (43 : 25 : 8 better : same : worse), valid 43 molecules 541 → 508 (20 : 18 : 2). By group size:
1 operation 663 → 644 (47 molecules, −2.9 %) / 371 → 359; 2–3 operations 262 → 248 / 72 → 65; 4–7 operations
128 → 106 (13 molecules, −17 %, 8 : 5 : 0) / 91 → 79; 8–11 45 → 41 / 7 → 5; 12–23 operations 62 → 51 (7
molecules, −18 %, 6 : 1 : 0). Symmetric at the start 192 → 176 (25), symmetric only at the end 968 → 914 (71).
Largest gains 135100494 25 → 17 (7 ops), 135101309 26 → 22, 134978595 10 → 6, 135099518 15 → 11 (15 ops),
135100404 28 → 23, 135263244 20 → 15; largest losses +2 calls (135041910 7 → 9, 103954262 18 → 20). No basin
changes: every changed endpoint within 0.005 kcal/mol of the champion's. Evaluation durations (126 / 253 s)
are in line with cycles 72–75 (171–304 s; the evaluator has been slower since 04:57 today) — the symmetry
search itself costs ≤ 40 ms per update on the largest molecules (`/tmp/symtime76.py`).

### Interpretation
The learning-rate argument holds: the gain grows with the group order (−3 % at one operation, −17 … −18 % at
4–23 operations) and the images are exact enough at the 0.05 Å / half-step tolerance that losers are rare
(8 of 76 changed train molecules, ≤ 2 calls). The σ_h-only class (planar molecules) gains little because the
reflection only separates in-plane from out-of-plane curvature, and most planar endgames are in-plane. The
Cartesian pathway (B G B⁺) was necessary: Sella's single improper dihedral at symmetric centres breaks the
closure of the coordinate list exactly for the tiny inorganic molecules with the largest groups (a signed
permutation would have dropped 175 of 403 operations). Images cost nothing in force calls and are exact
information, so this is not a lottery: the gain is replicated across splits in proportion to the class size
(train 96 / valid 43 symmetric endpoints).

### Next idea / control
Remaining symmetric-line follow-ups are small: (i) operations dropped at dummy atoms of chains on a rotation
axis (13 operations, 6 train molecules; the image dummy could be rotated about the chain axis into the frame of
the current dummy) ≤ 0.001; (ii) the C3v-chain and near-symmetric (0.05 < ε < 0.3 Å) classes would need
weighted or component-wise images, which the filters cannot vet — not pursued. The cost map after this cycle is
recomputed before the next idea (connected asymmetric endgame vs dimer endgame vs approach walks).

## Cycle 77 — symmetry-image secant pairs for multi-fragment systems: the point-group operations of the whole complex (fragments included) feed the multi-secant memory through the same Cartesian-pathway image map

Champion at start: `1ba6177625d7e7ff315c5995188815383eda7a4b` (train 0.6868163, valid 0.6924513).

### Hypothesis
Cycle 76 refused multi-fragment systems in `Internals.symmetry_maps` (no operations when translation/rotation
coordinates exist) on the grounds that fragment translations and rotations "are not internal coordinates of a
Cartesian displacement field". They are: Sella's `Translation` (fragment centroid) and `Rotation` (exponential-map
rotation vector against a stored reference) coordinates are functions of the Cartesian positions with a Jacobian
that `Internals.jacobian()` already provides, so the image map of cycle 76, s' = B G B⁺ s and y' = B⁺ᵀ G Bᵀ y
with the current B and its pseudo-inverse, is the exact first-order image of a pair for TRIC coordinates as well
(the energy invariance E(Gx) = E(x) is a Cartesian statement; the map never needs the coordinate list to be closed
under the operation). Multi-fragment endgames are the class where free directions are worth most: a symmetric
dimer endgame costs 14–18 calls (train symmetric-final dimers: neutral 15.2 calls vs reference 40.8, charged 16.5
vs 23.8; valid 14.3 / 17.8) because the six inter-fragment modes are floor- or contact-modelled and each is
learned one per call. Census on the champion's endpoints (`/tmp/dimersym77.py`, `/tmp/dimersym77b.py`, tol
0.05 Å): 72 train / 79 valid multi-fragment runs end at a symmetric geometry (share 0.097 / 0.120 of the score;
neutral 37 / 37, charged 32 / 36), almost all with a single operation (a mirror plane: the ion on the plane of its
partner, a planar H-bonded pair, a homodimer with C2 or Ci), and 58 / 75 are symmetric at the raw start. Cycle 76's
one-operation class gained −2.9 %; with the endgame a larger fraction of a dimer run the expectation is −0.002 …
−0.006 per split. Connected systems are untouched by construction (their maps are unchanged).

Energy-gate exposure (the same census, credits Σ(r−1)): the train margin +1.331 rests on seven symmetric-start
dimers (+1.18: carboxylates__ethers +0.356, ketones__monoatomics +0.233, carboxylates__ketones +0.199,
benzene__phenol +0.178, amides__guanidiniums +0.089, acids__esters +0.073, alcohols__esters +0.052) besides
monoatomics__pyrrole (+0.788, asymmetric throughout); the valid margin +2.925 rests on carboxylates__ketones
(+3.216, mirror-symmetric perturbed start, asymmetric end) and ethers__monoatomics (+0.852, three operations
throughout). An image is admitted only when the step's non-symmetric component exceeds 15 % (the dependence
filter keeps a candidate whose residual after Gram–Schmidt is ≥ 0.3 of a unit step, and |Gs − (s·Gs)s| ≈ 2|s_anti|),
so the paths of these walkers change only in their symmetry-breaking phase, where the image lets the update
separate the antisymmetric curvature (H s_anti = (y − Gy)/2) from the symmetric one — the breaking accelerates
along the seeded sign, which is on average favourable for calls and energy but does not guarantee the same landing
basin. No credit-protecting exception is built (neutral-only or "operations present at the perturbed start"
variants have no mechanism rationale: the images do not oppose the seed of `_break_start_symmetry`, they teach
the negative curvature it exposes); the gates decide.

Constants unchanged and fixed a priori: tolerance max(0.05 Å, ½ rms atomic step since the previous update),
search caps 48 permutations / 20000 nodes, memory 4, filters 0.3 / 0.25. Lone-atom fragments (ions) enter the
permutation search like any atom (their element is unique in the complex, so they map onto themselves); linear
fragments keep the dummy rule of cycle 76.

### Change (functions)
`Internals.symmetry_maps`: the refusal of systems with `fragment_atom_groups` / translation or rotation
coordinates is removed (systems with `other` coordinates are still refused; none exist in this optimizer);
docstring updated. Nothing else changes: `InternalPES._symmetry_images`, `PES._collect_secant_pairs`, the search
and the constants are those of cycle 76.

### Result
Candidate `b672269cfca1cfede7e896b36a31e816bd9e2ddd`: train **0.6940499** (+0.0072 vs 0.6868163), valid (mean
rel_energy 1.002826, credit Σ 1.3306 → 1.3255), 469/469 converged; `python scripts/cycle.py record` → discard
("train improvement below 1e-4"); the valid split was not run. Candidate preserved in
`ideas/symmetry-image-secant-pairs/b672269cfca1cfede7e896b36a31e816bd9e2ddd/algo.py`; champion 1ba6177 restored.

Per molecule (`/tmp/cmp77.py`, `/tmp/cmp77b.py`, `/tmp/cmp77c.py`, `/tmp/cmp77d.py`): every connected molecule
bit-identical (as designed); 59 of 213 multi-fragment runs changed, 28 better : 30 worse, Σ +3.39 rel (+0.00723
of the score), three basin changes (esters__esters 28 → 24 calls but 3.0 kcal/mol *higher*, credit 0 → −0.037;
amides__carboxylates 15 → 29 and acids__guanidiniums 17 → 39, minima 1.44 / 0.24 kcal/mol deeper, credits +0.026
/ +0.006) — without them Σ +1.22 rel (+0.0026), still a loss. By symmetry class (tol 0.05 Å at the perturbed start
/ the champion endpoint): symmetric start and end, charged, n 29: 428 → 465 calls (14 : 9; +36 from the two basin
changers, +1 otherwise), neutral n 16: 192 → 191 (5 : 2); asymmetric start, symmetric end, neutral n 20: 372 → 388
(4 : 7; +20 without esters__esters), charged n 5: 109 → 117 (2 : 3); near-symmetric start (0.05–0.15 Å),
symmetric end, charged n 4: 63 → 76 (0 : 4); symmetric start, asymmetric end, charged n 9: 219 → 225 (2 : 4);
15 changed runs are ≥ 0.15 Å from any operation at both ends (mid-run detections at the ½-step tolerance, rigid-body
steps of 0.3–0.9 Å), mixed, Σ ≈ +6 calls. By fragment: water complexes 1 : 8 (415 → 444: alkenes__water 24 → 32,
imidazolium__water 28 → 35, benzene__water 36 → 41, guanidiniums__water 14 → 17, three more +2, pyridine__water
−1), carboxylates 1 : 7 (349 → 377), guanidiniums 5 : 5 (297 → 326 with the two basin changers); monoatomics 6 : 3
(265 → 258), amines 4 : 0 (306 → 300), pyridine 5 : 3, ammoniums 5 : 3. Largest gains acids__benzene 32 → 26,
carboxylates__esters 29 → 23, benzene__guanidiniums 26 → 21, acids__monoatomics 18 → 14, ammoniums__pyridine
25 → 22. Most changed runs end at the champion's geometry (|A − B| < 0.01 Å): the effect is on the path, not on
the basin. Offline checks (pure geometry, `/tmp/symimg77.py`, `/tmp/covT77nd.py`): the TRIC image map is exact to
first order on all 73 / 77 symmetric multi-fragment endpoints (87 / 81 operations, all kept; finite-difference vs
linear image ≤ 0.7 %, gradient map exact to 1e-14; search ≤ 2 ms), and the analytic model T is covariant under the
detected operations (dimers ≤ 5.5 %, connected molecules without dummies ≤ 3 %), so neither the map nor the order
of transport and image is the cause.

### Interpretation
The class carries no free information for the images to release. In a dimer endgame the steps are almost totally
symmetric — the seed of `_break_start_symmetry` is 0.02 Å of rigid-body motion and the approach leaves little
antisymmetric residual — so an image is rejected as dependent, and when it is admitted (≥ 15 % antisymmetric
content) what it separates, H s_A'' = (y − Gy)/2, is the curvature of the softest inter-fragment modes (fragment
rotations at the 1e-3 Ha/rad² floor) read from gradient differences over 0.002–0.02 Å, at the SCC noise level:
the class that loses most is the water complexes (8 : 1 worse, +29 calls), the lightest and softest rotor of the
set, and the symmetric endgames as a whole lose (+28 same-basin) instead of the −3 % the connected one-operation
class gained. For symmetric-start ionic complexes the images accelerate the departure from the symmetric point:
two runs left a symmetric endpoint that the champion and the reference accept as converged for minima 0.2–1.4
kcal/mol deeper at +14 / +22 calls (correct physics, punished by the metric, negligible credit), and the ½-step
tolerance admits loose "symmetries" during rigid-body approach steps of 0.3–0.9 Å, perturbing paths at random.
No subgroup gains, so no bounded repair is expected to clear the gate: a fixed 0.05 Å tolerance for fragment
systems removes only the random mid-run part (predicted ≈ +0.003 from the remaining subgroups), and a neutral-only
or perturbed-start exclusion is credit protection without a mechanism. Closed: symmetry images for multi-fragment
systems at the cycle-76 constants (tolerance, filters, memory). The cycle-76 result stands unchanged (connected
systems bit-identical).

### Next idea / control
The symmetric line is exhausted (connected: done; multi-fragment: refuted; near-symmetric and weighted images not
vettable). Back to the cost map: connected asymmetric endgame (share 0.30 / 0.355, ratio 0.78 / 0.75) and the far
connected walks (rmsd₀ ≥ 0.5 Å: share 0.148 / 0.206, ≈ 20 calls above the 5.6-call intercept of the fit calls ≈
5.6 + 0.13 heavy + 15.5 rmsd₀, `/tmp/connrmsd77.py`).

## Cycle 78 — predictor–corrector step under the geometry-following contact block (all systems): the quasi-Newton step is solved with the path average of the non-local contact curvature along the step itself, s = −(H + ⟨A(x(t)) − A(x)⟩)⁻¹ g

Champion at start: `1ba6177625d7e7ff315c5995188815383eda7a4b` (train 0.6868163, valid 0.6924513).

### Hypothesis
The analytic part of the model Hessian moves with the geometry (cycles 53/55: H ← H + T(x) − T(x_prev), the
stored pairs transported with the Simpson path average of T), but the step −H⁻¹g still assumes the curvature
of x_k over the whole step although the model itself says how it changes: the contact block A(x) of
`_h0_nonlocal_contacts` (radial pairs k(r) = 0.36 exp(−1.944 (r − r_cov)/Bohr) Ha/Bohr², e-fold per 0.27 Å,
the hydrogen-bond and lone-pair-plane bends, the X–H···π bends, and for connected systems the vicinal 1,4 pairs)
changes by 44 % over a 0.1 Å contact step and by ×2.5 over a 0.25 Å one (the multi-fragment cap), so the local
step overshoots a closing contact by ≈ 20 % of the step (into the wall: energy rise, ρ < 0.25, radius halved) and
undershoots an opening one, and a rotor whose 1,4 pairs open along its turn is stepped with a stiffness the
model itself abandons half-way. Cycle 57 tested this predictor–corrector step with the *whole* T (stretch
diagonal + contact block) for connected systems only: neutral overall (−0.0001), but the read-out was split —
the gains sat where the contact block changes along the step (long runs ≥ 16 calls −0.22 / −0.88 per molecule,
rmsd₀ ≥ 0.7 Å −0.45, folding chains), the losses in the short runs, the bonds that lengthen ≥ 0.1 Å (+0.71) and the
small hydrogen-free molecules (+0.20), i.e. where the Almlöf stretch exponential is not the anharmonicity of the
xTB bond and its error was applied twice (transport and step) for a residual that the next step removes anyway
(stretches are stiff and converge in 2–3 steps whatever their first-order error; the soft modes set the count).
The backlog's stated next experiment is the contact-only variant bundled with its multi-fragment extension, and
the multi-fragment approach steps are the largest curvature changes of any step in the sets (the backlog: a
contact wall whose curvature changes 6× along a 0.5–1 Å approach step).

Where the calls are (champion, `/tmp/costmap77.py`): neutral dimers share 0.133 / 0.136 at 17.4 calls per
docked run after the docking, charged dimers 0.109 / 0.115 (undocked approaches of 1–2.5 Å at the 0.25 cap),
connected far walks (rmsd₀ ≥ 0.5 Å) 0.148 / 0.206. The cycle-71 read-out named the docked runs that spend
20–30 calls settling a 0.3–0.5 Å residual along the hydrogen bond (esters__ethers, esters__esters,
alcohols__pyridine, alcohols__pyrrole) — exactly the steps whose contact curvature changes 40–100 % along the
way. Expected: −1 … −3 calls on H-bonded approaches and settlings that currently overshoot into the wall (the
contact stretch and the contact bends scale with the same exponential, so the libration stiffness follows too),
−0.2 … −0.9 on long connected runs as in cycle 57, nothing on near-equilibrium endgames (dT → 0 with the step),
i.e. −0.003 … −0.010 per split if the cycle-57 signal transfers. The walkers along the floor modes (water on a
π face) are not addressed (their model error is the floor itself, not the path dependence).

Design, fixed a priori (cycle 57's, restricted to the contact block and opened to fragment coordinates):
dA_g = ∫₀¹ [A(x(t)) − A(x)] dt and dA_e = 2 ∫₀¹ (1 − t) [A(x(t)) − A(x)] dt on the linearised Cartesian path
x + t B⁺ s of atoms and dummies (composite Simpson, panels of ≤ 0.4 Å atomic displacement, ≤ 4 — the transport's
rule, `_analytic_model_average`), A evaluated on the shadow internals with the frame (Q, B⁺) of each node so
that the fragment translation/rotation rows of the block follow the geometry; the step is the fixed point of
s = restricted_step(H + dA_g(s)) started from the quadratic step, one correction (Heun-type) removes the
first-order error, a second one is kept only when the iteration contracts (max |Δs| decreasing), convergence at
1e-4; the trust ratio of the step is judged against the same model's energy prediction (H + dA_e in
`PES.kick`). The stretch diagonal keeps moving H and transporting the pairs as before (cycle 53) but does not
enter the step correction. Unchanged trust region (radius, caps, Cartesian bound for connected systems), stepper,
update, filters, docking, symmetry images. Applies to every system whose contact block exists (connected
systems: intramolecular non-local + vicinal pairs; multi-fragment: inter-fragment pairs and bends, the fragment
translation/rotation coordinates included); no neutral-only or docked-only scoping (the mechanism is the same
wall for an ionic hydrogen bond; the charged-dimer credits — valid carboxylates__ketones +3.216 — are exposed to a
path re-roll and the gates decide). Constants (Simpson panel 0.4 Å, ≤ 2 corrections, 1e-4) are not to be scanned.

### Change (functions)
`PES.__init__` (`self._B_pred = None`), `PES.kick` (the trust-ratio prediction uses `_B_pred` when the step was
computed under the path-dependent model), `InternalPES._contact_model_from` / `_contact_model_at_positions` /
`_contact_model_ahead` (the contact block A on the shadow internals at the Simpson nodes of the step ahead, its
path averages dA_g / dA_e), `Sella._restricted_step_with` (the restricted step under a substitute model matrix
with the Hessian object restored), `Sella._curved_step` (the fixed-point iteration, all systems, minimisation
only), `_default_kwargs['minimum']['curved_step_contact'] = True`, `Sella.__init__` / `_predict_step` (hook after
the quadratic step, `_B_pred` reset). Adapted from
`ideas/curved-step-analytic-model/ed2664b53496ba09f754803e49ef5f796d06d9b8/algo.py` (cycle 57).

### Result
Candidate `cd9cffd2a038c0d42b7a3ae0df622df2de256fb5`, train **0.6877086** vs champion 0.6868163 (+0.00089): **discard**
(train gate). 469 / 469 converged, mean rel_energy 1.0025 (credit Σ(r − 1) +1.331 → +1.176: monoatomics__pyrrole
1.788 → 1.728, carboxylates__ketones 1.199 → 1.103, 104079126 1.022 → 1.000 (the champion's basin 22.8 kcal/mol
below the reference is lost, 56 → 43 calls), ammoniums__benzene 1.000 → 1.033); duration 364 s (253 s for cycle 76;
the corrector costs ≤ 0.1 s per step for the largest molecule, nint 627). Valid not run. Total calls 7737 → 7697
(−40) but the score rises: the losses sit on small-reference runs.
- Connected (n 257): 3979 → 3930 calls (−49, share −0.00209), 41 : 176 : 40 better : same : worse, two basin
  changes (104079126 −13, acalabrutinib 15 → 25 +10). By walk length: rmsd₀ ≥ 0.7 Å (n 54) 1430 → 1366
  (−64, −0.0031; 17 : 27 : 10, Σ −90 / +26; 384221749 59 → 42, 134997543 46 → 34, 135140595 38 → 30,
  160853090 66 → 60, paliperidone_palmitate 55 → 50; ketones__phenol 28 → 36); 0.3–0.7 Å (n 101) 1541 → 1549 (+8;
  17 : 67 : 17, Σ −25 / +33 with the basin changer); < 0.3 Å (n 102) 1008 → 1015 (+7; 7 : 82 : 13, ±1–2 calls).
  By size: heavy ≤ 6 +4, 7–15 −18, > 15 −35.
- Multi-fragment (n 212): 3758 → 3767 (+9, share +0.00298), 74 : 69 : 69. Same basin (n 199, |Δ endpoint| < 0.05
  Å): 3432 → 3383 (−49, −0.0029; 69 : 69 : 61); basin changes (n 13): 326 → 384 (+58, +0.0059; 5 : 0 : 8). Ionic
  (`_dock_formal_charges`, n 79): 1444 → 1478 (+34, +0.0039; 31 : 13 : 35) = same basin −21 (26 : 13 : 29;
  monoatomics__pyrrole 33 → 21, imidazolium__phenol 27 → 21, carboxylates__esters 29 → 23; amines__carboxylates
  13 → 19, amines__guanidiniums 11 → 16) + basin changes +55 (n 11, 5 : 6: carboxylates__water 13 → 43 (ref 49,
  both endpoints 2.9 kcal/mol above the reference's basin), ammoniums__benzene 17 → 41 (ref 50, a basin 1.6
  kcal/mol deeper, +0.033 credit), carboxylates__ketones 21 → 41 (ref 9, a basin between the reference's and the
  champion's), carboxylates__ethers 26 → 33, amides__guanidiniums 40 → 44; benzene__guanidiniums 26 → 11,
  ammoniums__water 32 → 25, amides__ammoniums 25 → 19). Neutral (n 133): 2314 → 2289 (−25, −0.0009; 43 : 56 : 34),
  same basin −28 (acids__benzene 32 → 26, amides__phenol 34 → 29, pyridine__thiols 23 → 18, alcohols__benzene
  18 → 14, esters__esters 28 → 24, esters__ethers 29 → 26, benzene__water 36 → 33; pyrrole__sulfides 17 → 21,
  alkenes__water 24 → 28), the docked near-equilibrium runs (rmsd₀ < 0.3 Å, n 11) +6.
- Offline fidelity check (`/tmp/curv78_arc.py`, pure geometry on saved endpoints): for a 0.25-rad rotation of one
  fragment (the multi-fragment cap) the linearised path x + t B⁺ s ends 0.05–0.09 Å off the exact arc, the
  along-step contact curvature averages differ by 2–8 % (×1.70 vs ×1.64, ×1.18 vs ×1.14, ×1.21 vs ×1.12); the
  corrections themselves are large (×0.7–1.9 path average) — the mechanism does change the rigid-body steps, and
  the linearisation is not what changed them. The implementation's Simpson average equals the recomputed one.

### Interpretation
The path dependence of the contact block is not a binding cost either, for the second time (cycle 57 with the
stretch diagonal, connected only). Its systematic part is small and lives in the long walks: connected far walks
gain −4.5 % (17 : 10, replicating cycle 57's long-run read-out), the same-basin dimers gain −1.4 % (69 : 61 — a
few settlings of 3–6 calls: acids__benzene, amides__phenol, esters__esters, esters__ethers, the cycle-71 residual
runs), and near-equilibrium endgames are untouched or ±1 (the correction vanishes with the step). Against this the
corrected rigid-body steps of the approach phase (×0.7–1.9 on the contact block at the 0.25 cap) re-route the
approach of ionic complexes, which have several basins within 1–3 kcal/mol: 11 ionic basin changes, 5 : 6 in
count, but the three expensive ones fall on references of 9–50 calls (+2.2, +0.6, +0.5 rel) — the credit lottery
of the energy-gate memory, now on the step side. A better path (the curvilinear arc) would change the corrections
by ≤ 8 % and re-roll the same lottery; an ionic exclusion would protect credits without a mechanism (the ionic
same-basin runs gain too); the connected part is a separate code path whose runs are bit-identical to a
connected-only candidate: implied train 0.6847241 (−0.00209, credit +1.309), the far-walk gain with the
near-equilibrium ±1 noise. Conditions tested: contact block only, fixed point ≤ 2 corrections, Simpson ≤ 4 panels
of 0.4 Å, trust ratio under the energy average, both classes at once.

### Next idea / control
Bounded repair 1 (cycle 79): the same mechanism restricted to connected systems (`_has_tr_internals` false), the
scope of the champion's model refinements (cycles 64, 65, 76): the multi-fragment part is a wash (same basin −49,
basin changes +58) with the charged-dimer credit exposure on valid, the connected part pays on the long walks. Train
is predicted exactly (bit-identical connected runs, champion dimers): 0.6847 (−0.0021); valid decides whether the
far-walk gain (0.206 share there) transfers. If valid fails, the line is closed at both scopes (57: connected with
the stretch; 78: all systems, contact only; 79: connected, contact only); the candidate of this cycle is preserved
in `ideas/curved-step-analytic-model/cd9cffd2a038c0d42b7a3ae0df622df2de256fb5/algo.py`.

## Cycle 79 — bounded repair 1 of cycle 78: the contact-block predictor–corrector step for connected systems only (multi-fragment systems keep the quadratic step)
Champion at start: `1ba6177625d7e7ff315c5995188815383eda7a4b` (train 0.6868163 / valid 0.6924513).

### Hypothesis
Cycle 78's read-out separates the two classes the candidate changed: the connected runs (a separate code path,
bit-identical here) gained −49 calls (share −0.00209; far walks rmsd₀ ≥ 0.7 Å −64, 17 : 10, replicating the
long-run gain of cycle 57), while the multi-fragment runs were a wash in count (74 : 69 : 69; same basin −49,
13 basin changes +58) whose expensive draws fell on ionic complexes with small references, and whose valid-split
counterpart carries the carboxylates__ketones +3.216 credit. The corrected rigid-body approach steps (×0.7–1.9 on
the inter-fragment block at the 0.25 cap, `/tmp/curv78_arc.py`) re-route the approach between basins 1–3 kcal/mol
apart — not a model error to fix (the arc-vs-linearised path differs by ≤ 8 %) but a lottery the mechanism cannot
avoid there. The connected scope is the scope of the champion's own model refinements (cycles 64, 65, 76) and is
selected by the structural gate `_has_tr_internals` (no molecule identities, no counts). Train is predicted
exactly from cycle 78: 0.6847241 (−0.00209, credit +1.309); valid decides (far connected walks share 0.206
there, near-equilibrium connected ±1 noise: 7 : 13 on train).

### Change (functions)
`Sella._curved_step`: returns the quadratic step when `self._has_tr_internals()` (fragment translation/rotation
coordinates present); docstring and the `_default_kwargs` comment record the scope and its reason. Everything
else as in cycle 78 (`InternalPES._contact_model_ahead`, `Sella._restricted_step_with`, `PES._B_pred`).

### Result
Candidate `336c47f6d09df544e3b314691af989f21ed6174e`: train **0.6846728** (−0.00214, mean rel_energy 1.00284, credit
+1.331 unchanged, 469 / 469 converged, 196 s), valid **0.6898550** (−0.00260, mean rel_energy 1.00629, credit
+2.925 unchanged, 465 / 465, 280 s): **keep** — new champion.
- Multi-fragment runs bit-identical to the champion on both splits (212 / 212, 203 / 203).
- Connected, train (n 257): 3979 → 3928 calls (−51), 42 : 176 : 39; by walk length rmsd₀ ≥ 0.7 Å (n 54) 1430 → 1362
  (−68, share −0.0033; 18 : 28 : 8; 384221749 59 → 42, 134997543 46 → 34, 135140595 38 → 30, 104079126 56 → 49
  (same deep basin, credit 1.022 kept), 160853090 66 → 60, paliperidone_palmitate 55 → 50, 252657041 30 → 25,
  315658749 28 → 23; 336270104 16 → 19), 0.3–0.7 Å (n 101) +10 (17 : 66 : 18; acalabrutinib 15 → 25 into a
  basin 0.03 kcal/mol deeper, 135094180 14 → 21), < 0.3 Å (n 102) +7 (7 : 82 : 13, ±1–2 calls). By size heavy ≤ 6
  +3, 7–15 −19, > 15 −35.
- Connected, valid (n 262): 4546 → 4502 (−44), 67 : 152 : 43, no basin change (all |Δ endpoint| < 0.03 Å); rmsd₀
  ≥ 0.7 Å (n 91) 2134 → 2094 (−40, share −0.0026; 38 : 35 : 18; 252633479 28 → 22, 135192963 22 → 17, 252656935
  24 → 20, 135239978 37 → 33; 135052949 27 → 31), 0.3–0.7 Å (n 113) +2 (22 : 69 : 22; 135071820 18 → 24,
  135098956 12 → 16), < 0.3 Å (n 58) −6 (7 : 48 : 3; 103921822 36 → 30). By size heavy ≤ 6 −4, 7–15 +11 (15 : 19),
  > 15 −51 (49 : 82 : 23).
- Four train runs differ from the cycle-78 candidate's connected runs (ketones__phenol 36 → 27, 104079126 43 → 49
  back in the champion's basin, pyridine__pyrrole 19 → 21, guanidiniums__water 15 → 14): three are dimers that
  Sella joins into one fragment at the start and the fourth a 1.1 Å walk; the gate `_has_tr_internals` is read
  at every step, so a mid-run rebuild that splits or joins fragments switches the correction — the class
  membership of the step, not of the molecule — and these runs took a different path. Everything else is
  bit-identical to cycle 78 (train predicted 0.6847241, got 0.6846728).

### Interpretation
The contact-block path average pays where cycle 57 and cycle 78 said it would — the long walks of connected
molecules, −4.8 % / −1.9 % of their calls (18 : 8 and 38 : 18), the large molecules on valid (heavy > 15: 49 : 23,
−1.7 %) — and costs nothing elsewhere (near-equilibrium ±1, mid-range a wash), with no basin changes on valid and
a single 0.03 kcal/mol one on train. The mechanism is the folding chain: a contact or a vicinal 1,4 pair that
closes during a 0.3–0.5 Å step is stepped with its path-averaged stiffness instead of the pre-step one, so the
step stops short of the wall and the next step is not a rejection or a shrink; opening pairs are stepped with the
stiffness the model abandons half-way, so the rotor turns further per call. It is the third mechanism of the
tracked contact block (tracking 34/35/42/43, transport 53/55, now the step), each worth about −0.002 … −0.003.
The multi-fragment scope stays closed (cycle 78): the inter-fragment corrections are physically the largest but
the approach phase of ionic complexes is a basin lottery and the credits there are the energy-gate margin.

### Next idea / control
The curved-step line is done at this scope (stretch part refuted in 57, multi-fragment in 78, contact-only
connected kept in 79); no constant scan (panel length, corrections, 1e-4). Remaining cost map after cycle 79 (to
be recomputed with `/tmp/costmap77.py` on the cycle-79 results): connected far walks ≈ 0.145 / 0.20, neutral
dimers 0.133 / 0.136, charged dimers 0.109 / 0.115, connected near-equilibrium 0.17 / 0.09.

## Cycle 80 — out-of-plane (improper) coordinates for the planar π centres whose proper dihedrals run about single bonds (carbonyl, carboxyl, carboxylate, thiocarbonyl, nitro and boron centres; Schlegel's out-of-plane guess 0.045 Ha/rad²; connected systems only)
Champion at start: `336c47f6d09df544e3b314691af989f21ed6174e` (train 0.6846728 / valid 0.6898550).

### Hypothesis
A static model-frequency check of the champion's guess Hessian (`/tmp/modelfreq80.py`, `/tmp/oop80.py`: H_cart =
BᵀH₀B on saved dimer-endpoint monomers, mass-weighted, no calculator) puts stretches at 0.86–1.12 of experiment,
most bends at 0.9–1.2, the sp² in-plane bends of ethylene/benzene at 1.15–1.35, the rotors at 1.5–2.2 (closed line,
cycles 40/41/68/69), the NH₃ umbrella at 1.52 — and one class 2–5× too *soft*: the out-of-plane wag of a planar
sp² centre whose π partner is a terminal atom. Sella's improper rule adds the umbrella coordinate only at centres
without proper dihedrals (formaldehyde: wag 1079 vs 1167 cm⁻¹, fine); a carbonyl carbon with substituents has
proper dihedrals about its single bonds, so its pyramidalisation is modelled only by the class-scaled rotor
constants of those bonds (0.01–0.03 Ha/rad² in total): acetone 217 vs 484 cm⁻¹, acetic acid 245 vs 535, acetaldehyde
330 vs 509 and 539 vs 763. Alkene, aromatic and amide centres get their wag from the dihedrals about the multiple /
conjugated bond (bond-order factor exp(−2.85 (r − r_cov)/Bohr) ≥ 1.9; benzene's out-of-plane modes are within
10 %) — an improper there would double stiffness that is already right (over-stiff is benign, over-soft makes the
quasi-Newton step overshoot until the update corrects it; the composition regression on cycle 79 charges +0.94
calls per carbonyl in ours and +0.89 in the reference, the same rule). Public practice (geomeTRIC, `internal.py`,
public repository): an OutOfPlane coordinate at every planar 3-neighbour centre with a Schlegel-type diagonal of
0.045 Ha/rad². Candidate: add the improper (n0, centre, n1, n2) — n1 the terminal neighbour with the largest
bond-order factor, i.e. the torsion about the terminal double bond the dihedral set cannot contain — at every
three-coordinate 'pi' centre (`_torsion_centre_types`) that has proper dihedrals, angle sum ≥ 345°, and no bond
of bond-order factor ≥ 1.7 to a neighbour with further neighbours; stiffness 0.045 Ha/rad² (fixed a priori; the
Fischer–Almlöf torsional formula is rejected for it because its (r·r_cov)⁻⁴ drops 6–8× from C=O to C=S). Static
result: acetone wag 217 → 472 (exp 484), acetic acid 245 → 511 (535), acetaldehyde 330 → 539 / 539 → 808
(509 / 763), methyl acetate ≈ 342 → 555; formaldehyde, formate, amides (NMA, acetamide, formamide), NH₃, ethylene,
benzene, toluene, phenol, pyridine, pyrrole, imidazolium, guanidinium, indole, pyrazine bit-identical (no new
internals). Census on the train start geometries (`/tmp/oop80_census.py`): 72 of the 250 connected molecules get
132 impropers (C=O 74, C=S 9, B=S 9, B=O 5, N=O 4, B=F 1, C=N 1, C=Br 2, C=Cl 1, and 26 planar carbons of
conjugated / polycyclic rings whose bonds are all 1.42–1.51 Å, bond-order 1.0–1.68 — borderline, partly anchored
centres; no aromatic C–H carbon with two 1.39–1.40 Å bonds fires); the affected molecules carry 1352 calls
(score share 0.116). Expected gain: a fraction of a call per carbonyl-type centre, −0.003 … −0.008 on train, more
on valid where carboxyl / ester centres are frequent; controls: multi-fragment runs and the 178 unaffected
connected molecules bit-identical.

### Change (functions)
`Internals.find_all_dihedrals` calls the new `Internals._add_planar_centre_impropers(neighbors, dihedral_centers,
bo_pi=1.7, planar_min=345.)` after Sella's umbrella impropers (returns immediately when fragment translations /
rotations exist); `Internals.guess_hessian` collects `proper_centres` while counting the dihedrals per bond and
gives an improper whose centre has proper dihedrals the new `Internals._h0_out_of_plane()` (0.045 Ha/rad²) instead
of the Fischer–Almlöf umbrella value. Nothing else changes; the umbrella impropers at centres without proper
dihedrals keep their value.

### Result
Candidate `b70bec7abfce13ad2b965c2006bb0ea1abf06c49`: train **0.6800287** (−0.00464, mean rel_energy 1.00274, credit
+1.287, 469 / 469 converged, 300 s), valid **0.6823292** (−0.00753, mean rel_energy 1.00629, credit +2.926, 465 / 465,
109 s): **keep** — new champion.
- Multi-fragment runs bit-identical on train (219 / 219); on valid 214 / 215 — des370k_acids__pyrrole__valid (19 → 19
  calls, max atom shift 0.005 Å, ΔE −2e-5 kcal/mol) is a dimer Sella joins into one fragment, so the acid's carboxyl
  carbon gets the improper at that rebuild (the connected-scope gate is structural and read at every rebuild).
- Connected molecules with new impropers at the start (census on start geometries): train n 72, 1352 → 1302 calls
  (−50, 41 : 19 : 12; C=O centres n 49 −41 (32 : 9 : 8), C=S n 6 −9, N=O n 3 −8, B=O n 2 −3, B=S n 2 +1, molecules with
  only the borderline conjugated-ring centres n 15 −1 (4 : 9 : 2)); valid n 99, 1794 → 1707 (−87, 61 : 26 : 12; C=O n 58
  −68 (43 : 8 : 7), N=O n 8 −11 (7 : 1 : 0), C=S n 4 −3, B=O n 3 +4, ring-only n 27 −4 (7 : 15 : 5)). The gain grows with
  the number of impropers per molecule: rel 0.751 → 0.743 / 0.757 → 0.726 (one), 0.784 → 0.730 / 0.724 → 0.691 (two),
  0.734 → 0.666 / 0.707 → 0.653 (three or more), and is spread over walk lengths (valid far −39 (25 : 11 : 5), mid −32
  (28 : 10 : 6), near −16 (8 : 5 : 1)) and sizes (heavy > 15: −73, 48 : 24 : 8). Biggest movers: 172113611 42 → 35,
  384221749 42 → 38, zanubrutinib 15 → 11, venetoclax / nintedanib / palbociclib 18 → 15 / 18 → 15 / 15 → 12
  (train); 135065494 53 → 45, 160845249 20 → 13, 135071820 24 → 18, 135079402 17 → 12 (valid). Losses: paliperidone
  palmitate 50 → 56, 160853090 60 → 64, 104129283 28 → 38 (valid, basin 0.018 kcal/mol lower), 135100509 13 → 18.
- Basin changes: train 134997543 (C=S ×2) 34 → 28 into a basin 2.70 kcal/mol *higher* (rel_energy 1.000 → 0.965) and
  135104646 17 → 20, +1.28 kcal/mol (rel 0.990) — the credit loss −0.044 is the whole train credit change; valid two
  changes, both to lower basins (−0.018, −0.166 kcal/mol; credit +0.0006).
- Connected molecules without new impropers at the start: train 178, 173 bit-identical, 5 small inorganic molecules
  (B₆H₄, P₂NF₆H, N₄S₄, PF₂N₂H₅, P₂C₃O₄H₆: 8–15 atoms) change by ±1–4 calls (Σ +5) because a rebuild at a later,
  planar geometry adds or drops the improper; valid 151, 150 identical, one Si/F molecule 9 → 10.

### Interpretation
The model deficiency the static frequency check exposed is real and costs calls: a carbonyl-type centre whose wag
the model holds 2–5× too soft makes the first quasi-Newton steps overshoot along that mode, and the update needs
one or two calls per centre to stiffen it — the fix removes ≈ 0.7–1.0 calls per centre (C=O −41 / 49 molecules
with ~1.5 centres each on train, −68 / 58 on valid; nitro centres −8 / 3 and −11 / 8), consistent with the +0.94
calls per carbonyl the composition regression charged and with the reference sharing the defect (+0.89 there). The
borderline conjugated / polycyclic centres (bond-order factor 1.0–1.68 on all bonds, ≈ 10 % of the additions) are a
wash (−1 / −4 over 42 molecules): an improper at a partly anchored centre over-stiffens a mode that is already
about right, which the update absorbs for free. The one expensive draw is the thioketone 134997543, whose stiffer
C=S wags route the run into a basin 2.7 kcal/mol higher in fewer calls (the basin lottery of far walks; credit
+1.287 stays far above the gate); the valid credits are unchanged because the charged dimers are bit-identical.
General lesson (adds to cycle 39): the static model-frequency check against gas-phase fundamentals finds the
> 2× soft modes that pay; the sp² in-plane bends (1.15–1.35×), NH₃-type umbrellas (1.5×) and rotors (1.5–2.2×)
are over-stiff and benign.

### Next idea / control
Extend the static model-frequency check (`/tmp/oop80.py`, now champion = cycle 80) to the remaining monomer classes
of the dimer endpoints with known fundamentals (methanol, methylamine, formamide / acetamide N–H wags, nitromethane,
acetonitrile, DMSO, methyl acetate, phenol, aniline, thiophene / furan, pyridine) and look for further modes ≥ 2×
too soft; over-stiff modes are not worth a cycle (cycles 38, 40/41, 67). Constants of this cycle (0.045 Ha/rad²,
bo 1.7, 345°) are fixed a priori and are not to be scanned.

## Cycle 81 — ion–dipole angular stiffness for the contacts of s-block metal ions in connected systems (bends with the ion q μ₀/(2 R²), ion-terminal dihedrals q μ₀/R² shared per contact bond; μ₀ = 0.73 au, the water dipole; covalent torsion shares no longer count the ion)
Champion at start: `b70bec7abfce13ad2b965c2006bb0ea1abf06c49` (train 0.6800287 / valid 0.6823292).

### Hypothesis
Idea search after cycle 80 (all static, on saved geometries: `/tmp/freq81.py`, `/tmp/inter81.py`,
`/tmp/inter81_census.py`, `/tmp/ion81_check.py`; no calculator, no optimizer step):
- The extended monomer frequency survey (methanol, methylamine, ethane, propane, methanethiol, H₂S, formamide,
  pyridine, phenol, dimethyl ether, ethanol, acetone, acetic acid, acetaldehyde, formic acid, formaldehyde, NH₃,
  ethylene, benzene, water, propene, CH₃Cl) finds no further mode ≥ 2× too soft: alcohol / thiol torsions and the
  amine wag are 1.3–2.2× over-stiff (benign, cycles 40/41/67), everything else 0.85–1.35. The organic-monomer line of
  the model-frequency check is closed.
- The intermolecular-mode survey of the 434 same-basin dimers (Eckart-projected guess Hessian at the cycle-80
  endpoints, modes with rigid-body content > 0.6, grouped by the strongest contact class hb / chb / vdw / ion) shows
  no class whose model softness correlates with the call count; the weak-contact (C–H···Y, vdW) dimers are
  over-stiff by 1.5–2.5× at most — no signal. The endgame of the neutral dimers stays anharmonicity / noise limited.
- The limited-memory multi-secant TS-BFGS is cumulative (`update_H` adds `Bplus` to the running H), so memory 4 does
  not discard old curvature — only the exact secant conditions of the last four pairs are re-imposed; frame-aware
  rotation pairs (transporting the rotation-coordinate components with the fragment frame) would change the
  endgame by a negligible amount. Not worth a cycle.
- The one class with a first-principles discrepancy is the s-block metal ions. The 14 `monoatomics` systems
  (Na ×5, K ×4, Li ×3, Mg, Ca; the dataset scan finds no other s-block atom) split into 8 that are *connected* at
  the start (`find_all_bonds` uses Cordero covalent radii × 1.25: K 2.03, Na 1.66, Li 1.28, Mg 1.41, Ca 1.76 Å, so a
  contact ion gets "bonds" to its nearest C/H/N/O/S atoms and the system is built with bends and dihedrals through
  the ion; rebuilds happen only via `check_for_bad_internals`, so the mode never changes during a run) and 6 that
  start as fragments (rigid-body coordinates, radial-only cation contacts — bit-identical under any change of the
  bonded-mode guess). Costs of the connected eight (ours / ref): Mg²⁺–phenol 32 / 52, Na⁺–S(CH₃)₂ 11 / 25,
  Na⁺–CH₄ 11 / 16, Li⁺–butane 22 / 37, K⁺–dithiane 23 / 40, Li⁺–benzene 9 / 13, K⁺–H₂S 20 / 22, Ca²⁺–ethene 6 / 9;
  K⁺–H₂S needs 20 calls for four atoms. Their radial contact terms already have the ionic scale (cycle 52, Shannon
  radii in `_STIFFNESS_RADII`), but every bend with the ion gets the Fischer–Almlöf bend guess `_h0_angle`
  (0.089 floor + 0.11 exp(−0.44 Δr) → 0.09–0.12 Ha/rad² regardless of the contact length), and every proper dihedral
  with the ion at an end gets the covalent torsion constant of the ligand bond X–Y (shared 1/√n with the covalent
  dihedrals, which it also dilutes). Physically the angular stiffness of a cation–ligand contact is the ion–dipole
  term E = −q μ cos φ / R², k_φ = q μ / R²: Na⁺···OH₂ at 2.3 Å gives 0.039 Ha/rad², identical with the explicit
  three-point-charge (TIP3P) water calculation (rock 398, wag 491 cm⁻¹ — the librations of the monohydrated ion);
  a rotation of the ligand in the plane of two of its bonds moves two M–X–Y bends by ±φ, so each bend carries
  q μ₀/(2 R²) with the water dipole μ₀ = 0.73 au (1.855 D) as the reference ligand: Li⁺ 2.0 Å 0.026, Na⁺ 2.3 Å 0.019,
  K⁺ 2.8 Å 0.013, Mg²⁺ 2.0 Å 0.051 Ha/rad² — the model was 5–15× over-stiff in k (2.2–4× in frequency) for these
  bends, outside the 1.5–2.2× regime shown to be free (cycles 38, 40/41, 67). In Cartesian terms the lateral
  stiffness of K⁺ against H₂S at 7.14 Bohr is 1.5e-4 Ha/Bohr² physically against 1.5–2.3e-3 in the champion model;
  the fragment-mode model of the same contact (radial-only) is ≈ 2.5e-3 for Na⁺ over pyrrole — the bonded mode was
  the inconsistent one.
Candidate: in connected systems a bend with an s-block ion (formal charge q = 1 for group 1, 2 for group 2, table
`_ION_CHARGE`) gets k = q μ₀ / (2 R_1 R_2) with the contact distance(s) (ion at the apex: both contacts; ion at an
end: the contact squared); a proper dihedral with an ion at an end (the wag of the ion about its contact atom X)
gets q μ₀ / R² shared equally by the n dihedrals through the contact bond (M, X), and these dihedrals no longer
enter the 1/√n share of the covalent torsion about X–Y. Constants (μ₀ = 0.73 au, the formal charges) are fixed a
priori and are not to be scanned. Static check (`/tmp/ion81_check.py`, champion vs candidate guess Hessians on
saved start / end geometries): ion-free systems (paliperidone palmitate, venetoclax, 160853090, 384221749,
alkanes__benzene, ethers__ethers, alkanes__esters) and the six fragment-start ion complexes bit-identical; the
connected eight softened as designed — K⁺–H₂S lateral block 0.0015 / 0.0018 → 0.0001 / 0.0005 Ha/Bohr² (model bends
629 / 684 → 173 / 320 cm⁻¹), Na⁺–CH₄ 551 / 751 → 208 / 502 cm⁻¹, Li⁺–benzene lateral 0.070 → 0.028 (the rest is the
six radial Almlöf terms), K⁺–dithiane 0.0062 / 0.0068 → 0.0009 / 0.0027, Na⁺–S(CH₃)₂ 0.0032 / 0.0058 →
0.0009 / 0.0011, Mg²⁺–phenol start 0.077 → 0.046 and end 0.020 / 0.050 → 0.017 / 0.031, Ca²⁺–ethene end
0.013 / 0.015 → 0.0045 / 0.0082, Li⁺–butane start 0.29 / 0.61 / 0.72 → 0.18 / 0.47 / 0.54 Ha/Bohr². Expected:
train −0.0005 … −0.001 (three connected ion systems on train, thin against the 1e-4 gate), valid ≈ −0.002 (five);
the credit walkers (monoatomics__pyrrole, ethers__monoatomics, the charged organic dimers) are fragment runs and
bit-identical; basin changes are possible only on the connected eight (the Mg²⁺ π → O migration in Mg²⁺–phenol).

### Change (functions)
Module table `_ION_CHARGE` (formal charges of Li/Na/K/Rb/Cs = 1, Be/Mg/Ca/Sr/Ba = 2); new `Internals._h0_ionic_bend`
(q μ₀ / (2 R_1 R_2), μ₀ = 0.73 au) and `Internals._h0_ionic_torsion` (q μ₀ / R² / n over the dihedrals through the
contact bond); `Internals.guess_hessian` reads `q_ion`, routes bends with an ion (connected systems, after the
dummy / linear-bend branch) to `_h0_ionic_bend`, counts the ion-terminal proper dihedrals per contact bond
(`ion_end`, `ndih_ion`) instead of per covalent bond, and gives them `_h0_ionic_torsion`. Multi-fragment systems,
ion-free systems, the radial contact terms, impropers and the linear-bend dummies are untouched.

### Result
Candidate `9988b8d39969333516b767fb8b7592d25976b765`: train **0.6796564** (−0.000372, mean rel_energy 1.00274, credit
+1.287, 469 / 469 converged, 268 s), valid **0.6816682** (−0.000661, mean rel_energy 1.00629, credit +2.926,
465 / 465, 108 s): **keep** — new champion.
- Exactly the eight connected ion complexes change (train 466 / 469 and valid 460 / 465 bit-identical, positions and
  energies): train Mg²⁺–phenol 32 → 25 (ref 52), Na⁺–S(CH₃)₂ 11 → 10 (25), Na⁺–CH₄ 11 → 11 (16; trajectory changed,
  max shift 0.001 Å); valid Li⁺–butane 22 → 16 (37), K⁺–dithiane 23 → 18 (40), Ca²⁺–ethene 6 → 5 (9), Li⁺–benzene
  9 → 9 (13; same basin, ΔE −0.001 kcal/mol), K⁺–H₂S 20 → 22 (22). Σ −8 calls on train, −10 on valid; no basin
  change (|ΔE| ≤ 0.001 kcal/mol everywhere), credits unchanged to 1e-5.
- The six fragment-start ion complexes (monoatomics__pyrrole 33, ethers__monoatomics__valid 27 and the rest) are
  bit-identical as designed; the charged organic dimers and all ion-free molecules are bit-identical.

### Interpretation
The bends and ion-terminal torsions of a bonded-mode cation contact were 5–15× over-stiff in k (2.2–4× in
frequency) — beyond the 1.5–2.2× regime shown to be free (cycles 38, 40/41, 67) — and the ion–dipole constant
removes 15–25 % of the calls of the affected runs (Mg²⁺–phenol −7 / 32, Li⁺–butane −6 / 22, K⁺–dithiane −5 / 23):
with a covalent-sized angular cage the quasi-Newton steps along the ion's sideways modes were tiny until the update
had softened every bend and torsion through the ion, which for the polydentate contacts (Mg²⁺ over the phenol ring,
Li⁺ in the butane cleft, K⁺ between two sulfurs) takes many secant pairs. The a-priori constant (water dipole,
0.73 au) is right to the factor that matters: the one loss, K⁺–H₂S +2 (H₂S dipole 0.97 D, half the reference,
but a large anisotropic polarisability and a K–S contact that ends at 3.78 Å), is a 4-atom endgame within the
same-basin noise; a ligand-resolved dipole would be a scan, not to be built. Small absolute gain (three train / five
valid molecules) as expected; consistent with cycle 52 (ionic radii for the same contacts) — the s-block model is
now consistent between the bonded and the fragment coordinate modes (radial: Shannon radii; angular: ion–dipole).

### Next idea / control
The connected-mode ion model is complete (radial + angular). Left in this class: the six fragment-start ion complexes
(radial-only contacts; the two credit walkers among them are not to be touched). The idea-search closures of this
cycle stand: organic-monomer model-frequency survey closed (no ≥ 2× soft mode left), intermolecular-mode survey
without a signal, cumulative update makes frame-aware rotation pairs negligible. Constants of this cycle (μ₀ = 0.73
au, formal charges) are fixed a priori and are not to be scanned.

## Cycle 82 — ion–dipole angular stiffness for the inter-fragment contacts of cationic fragments with the lone-pair atoms of neutral ones (Gauss–Newton form of E = k (1 − cos φ), k = q μ₀ max(cos φ, 0)/R², φ between the lone-pair axis of the acceptor atom and the direction to the charge site; s-block ions and the formal-charge sites of ammonium / guanidinium / imidazolium fragments; μ₀ = 0.73 au as cycle 81)
Champion at start: `9988b8d39969333516b767fb8b7592d25976b765` (train 0.6796564 / valid 0.6816682).

### Hypothesis
Idea search after cycle 81 (saved-data analysis only):
- Cost map unchanged: connected share 0.40 / 0.39 of the score (mean rel 0.753 / 0.725, uniform across walk lengths,
  at the one-mode-per-step limit; losers few and mild), dimers 0.28 / 0.29 with a same-basin endgame intercept of
  ≈ 14 calls (fit calls = 13.9 + 0.04 nat/10 + 3.1 rmsd₀ for the 175 same-basin neutral dimers, resid sd 6.9) that
  is uncorrelated with the model's soft-mode frequencies (cycle-81 census) — no offline-testable lever there.
- The near-equilibrium docking loss of cycle 70 sits in two valid molecules (amides__pyridine 12 → 30,
  amines__water 16 → 33, same basin); a contact-aware or ρ-ratio acceptance cannot be evaluated offline (docked
  poses and forces are not saved; diagnostics exist only for failed runs) — not pursued.
- The same-basin charged dimers by class (cycle-81 endpoints): cations (ammonium / guanidinium / imidazolium)
  n 75, 16.7 calls at rmsd₀ 0.56 against 15.7 predicted by the neutral fit, with a steeper rmsd₀ slope (9.4 vs 3.1
  per Å); anions n 23, 20.6 calls. The cation class is the larger one (52 train / 50 valid dimers, ours 973 / 943
  calls against ref 1495 / 1391 — ratio 0.65, worse than the neutral 0.48) and holds almost no energy credit
  (+0.223 / +0.063; the anion class holds the valid gate through carboxylates__ketones__valid +3.216 and is not
  touched).
- First-principles discrepancy of the fragment-mode model for a cation: the acceptor of a charged hydrogen bond
  turns against the charge–dipole term q μ/R² (0.026 Ha/rad² for a unit charge 2.8 Å from the acceptor atom; the
  united charge of an ammonium at its N) on top of the neutral-water-dimer electrostatics that the acceptor bends
  of `_h0_contact_bends` carry (2 × 0.0075 = 0.015 for a capped contact): ≈ 0.04 in total (the 400–650 cm⁻¹ water
  librations of NH₄⁺···OH₂, like Na⁺···OH₂), so the model holds the acceptor's orientation 2–3× too softly — in the
  regime where fixing pays (cycles 39, 45/48, 81), unlike the 1.5–2.2× benign band. The same term is what the six
  fragment-start s-block complexes lack: a single-atom contact (Na⁺···N amine, K⁺···O ether, Na⁺···N pyrazine) has
  no sideways stiffness beyond the rigid-body floor (2.8e-4 Ha/Bohr² against q μ/R⁴ = 2e-3 for Na⁺ at 2.3 Å; the
  cycle-72 note: 10–30× under-stiff), whereas the cation–π complexes (Na⁺–pyrrole, Li⁺–phenol, K⁺–propene) get
  their lateral stiffness from the inclined radial pair terms of five or six carbons (≈ 1.1 k_radial, near the
  physical 0.02–0.03 Ha/Bohr²) and need nothing. The connected-mode counterpart is cycle 81 (bends and terminal
  dihedrals through the contact bond); the fragment mode has no such coordinates, so the term must be a Cartesian
  Gauss–Newton form mapped through the pseudo-inverse Jacobian like the contact bends.
- Credit exposure (cycle-81 results): the affected molecules hold +0.146 of the train credit (+1.287 in total;
  monoatomics__pyrrole__train +0.788 is a cation–π complex, bit-identical by construction) and +1.003 of the valid
  credit (+2.926; ethers__monoatomics__valid +0.852 is affected — K⁺ walking from 3.66 to 2.31 Å while the
  reference stops at 3.24 Å after 9 calls; carboxylates__ketones__valid +3.216 is an anion dimer, bit-identical).
  Both gates survive the loss of every affected credit; adverse re-rolls of the 11 / 13 basin-changing cation walkers
  would have to sum to −1.14 / −1.92 to fail them.
Candidate: every inter-fragment pair (Q, X) in the non-local pair list with Q a charge site of a cationic fragment
(the fragment's net formal charge by the valence rules of `_dock_formal_charges`, distributed over its positively
charged atoms: a bare s-block ion, the N of an ammonium, the central C of a guanidinium / amidinium / imidazolium)
and X an uncharged lone-pair atom of a neutral fragment (N, O, S, halogen with |d| ≥ 0.5 for d = −Σ unit vectors to
its covalent neighbours: ethers, pyridine-type N, amines, carbonyls, thioethers; not planar amide / pyrrole /
aromatic N, not four-coordinate atoms; not anionic partners, whose field aligns the donor X–H dipole that the donor
bend already holds at 0.025 Ha/rad² against q μ/R² = 0.02–0.03) gets k Jᵀ J with J the Jacobian of v = u_XQ − d̂
(|v|² = 2 (1 − cos φ): isotropic in the two sideways motions of the charge relative to the lone-pair axis, free
about that axis, rigid-motion invariant, coupled to the ligand internals through the neighbours of X) and the true
ion–dipole curvature along φ as the weight, k = q μ₀ max(cos φ, 0)/R² (μ₀ = 0.73 au; zero for a charge on the far
side of the atom). Constants (μ₀, the valence-rule charges, the |d| threshold) are fixed a priori and not to be
scanned. Static check (`/tmp/ion82_check.py`, champion vs candidate guess Hessians at the start and the cycle-81
endpoint of all 934 systems; the Jacobian against central differences of v, max |H − H_num| ≤ 3e-10 of max |H| ≈ 1):
train 431 / 469 and valid 425 / 465 bit-identical (all connected systems, all neutral dimers, the anion dimers, the
salt bridges and cation–halide pairs, the eight connected ion complexes and the three cation–π fragment complexes);
the 38 / 40 changed are the cation dimers with a lone-pair partner (ours 721 / 766 calls, ref 1083 / 1102) and the
three s-block σ-complexes (amines__monoatomics__train 10 / 11, ethers__monoatomics__valid 27 / 9,
monoatomics__pyridine__valid 11 / 14). Model intermolecular modes at the endpoints: guanidinium–water librations
233 / 237 → 368 / 466 cm⁻¹, ammonium–water 251 / 286 → 292 / 339, ammonium–imidazole acceptor rotations
15 / 16 / 48 → 16 / 65 / 67, acids__ammoniums 21 / 38 / 50 → 43 / 53 / 71; the s-block σ-complexes at the start
10–30 → 23–59 cm⁻¹ (Na⁺···N amine at 3.33 Å k = 0.018, K⁺···O at 3.66 Å 0.015, Na⁺···N pyrazine at 3.84 Å 0.014
Ha/rad²); weights of the cation dimers 0.010–0.031 Ha/rad² (N⁺···O 2.6–2.7 Å 0.023–0.028, imidazolium C2···O 3.6 Å
0.014, thioether cos φ ≈ 0.5). Expected: a few tenths of a call per same-basin cation endgame (−0.001 … −0.003 per
split), re-rolls possible on the basin-changing walkers; unchanged model for the 431 / 425 others.

### Change (functions)
New `Internals._formal_charge_sites` (valence-rule formal charges of the real atoms via `_dock_formal_charges` on the
covalent graph, net charge per fragment; cached per graph) and `Internals._h0_ionic_contacts` (charge sites of the
cationic fragments, lone-pair atoms and axes of the neutral ones, the Jacobian of v = u_XQ − d̂ over Q, X and the
neighbours of X, k Jᵀ J mapped through Binv, eV units); `Internals._h0_nonlocal_contacts` adds it for the non-local
pairs next to `_h0_contact_bends` (so it is geometry-tracked and transported like the rest of the contact block);
docstrings of `_h0_nonlocal_contacts` / `_h0_contact_bends` updated. Connected systems (no inter-fragment pairs),
neutral and anionic dimers, and the covalent / radial / hydrogen-bond terms are untouched.

### Result
Candidate `6312ef0abba83727e237ccad6cb193cf25659862`: train 0.6821470 (Δ +0.0024906 vs 0.6796564; mean rel energy
1.0020753 vs 1.0027439), 469/469 converged — **discard** (valid not run). 39 runs changed (the 38 of the census
plus imidazolium__phenol__train, whose term switches on during the approach — same 27 calls, same minimum); 430
bit-identical. Affected set 748 → 769 calls (ref 1133). Same-basin 36: Σ +10 calls (13 faster, 8 same, 15 slower;
+0.00092); three basin changes +0.00157: imidazolium__ketones 36 → 55 (−0.10 kcal/mol, the cycle-72 re-roller again,
then 36 → 72), amides__ammoniums 25 → 37 (−0.21 kcal/mol), ammoniums__ketones 29 → 9 into a stationary point
4.99 kcal/mol above the reference minimum (0.14 Å rmsd from the start, 0.43 Å from the reference endpoint; same
N⁺···O 2.65 Å contact; credit +0.000 → −0.318, train credit +1.287 → +0.973). By charge site: ammonium (N⁺, the
donor atom itself) 13 runs 270 → 259 (same-basin 11: Σ −3 — ammoniums__water −6, ammoniums__pyridine −4,
acids__ammoniums −3 against ammoniums__ethers +5, ammoniums__esters +5, ammoniums__phenol +3); guanidinium (central
C, 3.6 Å from the acceptor) 13 runs 221 → 224; imidazolium (C2) 12 runs 247 → 275 (+19 of it the basin change;
alcohols__imidazolium +5, amides__imidazolium +3); the one s-block σ-complex amines__monoatomics 10 → 11 (Na⁺···N
from 3.33 Å, same minimum). Cycle-72 comparison (the doubled ionic H-bond caps, same walkers): same-basin Σ 0 / 49
then, Σ +10 / 36 now; the two together say the acceptor-orientation stiffness of a charged contact (0.015 → 0.03
then, 0.015 → 0.035–0.043 now, the physical value) is not a binding mode of the charged-dimer endgame.
Candidate preserved in `ideas/ionic-contact-dipole-fragment/6312ef0abba83727e237ccad6cb193cf25659862/algo.py`;
branch reset to `9988b8d`.

### Interpretation
The fragment-mode charge–dipole term is physically right for the acceptor rock/wag of NH₄⁺···OH₂-type contacts
(librations 233 / 237 → 368 / 466 cm⁻¹ in the model), yet the same-basin cost does not move — twice now (cycle 72
Σ 0 / 49, here Σ +10 / 36). Under-stiff by 2–3× on a mode that already sits 15–25× above the rigid-body floor is
in the benign band: the update fixes it within the first steps, and the endgame of a charged dimer runs along the
modes the term does not touch (twist about the contact, the rotors of the neutral partner, the ammonium's own
rotor). What the term does change is the path of the basin-changing walkers — an ion held sideways-stiff against
the acceptor's lone-pair axis can stop at the nearest stationary point (ammoniums__ketones 9 calls, +5 kcal/mol)
or take a longer route (imidazolium__ketones, amides__ammoniums) — and the re-roll decided the score. For the
delocalised cations the united charge at the central C is also the wrong site (the field is at the N–H's; the
term aligns the acceptor's lone pair with a point 3.6 Å away at 30° from the H-bond axis, imidazolium cos φ 0.68 at
the new endpoint), which explains the mild excess of the guanidinium / imidazolium classes but not the null of the
ammonium class where the site is right. The s-block σ-complex sample (one train run, +1 call) is too small for
any conclusion and cannot be evaluated alone under the gates (2 of the 3 σ-complexes are on the valid split, one of
them the +0.852 credit walker). No bounded repair: an ammonium-only or s-block-only restriction would be a class
selected by outcome (the same-basin ammonium Σ −3 / 11 is noise), and re-siting the delocalised charge on the N–H's
would still act on the acceptor orientation that both experiments found non-binding.

### Next idea / control
Fragment-mode angular stiffness of charged contacts closed (with cycle 72). The s-block model is complete for the
connected mode (cycles 52, 81); the fragment-start σ-complexes stay radial-only. The remaining cost of the charged
dimers (ratio 0.65 against 0.48 neutral, 16.7 same-basin calls at rmsd₀ 0.56 with a 9.4 calls/Å slope) is a path
cost along the modes of the neutral partner and the approach, not a model-stiffness cost of the contact.

## Cycle 83 — torsion-space pre-relaxation of neutral connected starts on a UFF-torsion + clash-wall surrogate (rotor angles by L-BFGS-B; gated by the rotor-torque cosine with the first xTB force field, a 0.5 Å minimum move after Kabsch superposition and an xTB energy decrease; start evaluation cached; charged, odd-electron and multi-fragment systems untouched)
Champion at start: `9988b8d39969333516b767fb8b7592d25976b765` (train 0.6796564 / valid 0.6816682).

### Hypothesis
Idea search after cycle 82 (saved-data analysis only; the cost map after cycles 81/82 has no model-stiffness lever
left in the connected class and none in the charged-dimer endgame):
- The connected starts are thermal SPICE conformers plus 0.02 Å noise: relative to the reference minima, rms bond
  change 0.04 Å, rms bond-angle change 4.4°, but the largest dihedral change has a median of 37° (rmsd₀ up to
  2 Å). The connected cost is a path cost in the torsions: 11.6 calls per Å of rmsd₀, and by the largest dihedral
  change of the walk < 10° 7.8 calls, 10–20° 11.3, 20–40° 14.7, 40–90° 19.0, > 90° 21.7 (cycle-81 endpoints).
  A start that already sits at the right torsions would save the rotor part of the walk — ceiling −0.04 … −0.10
  per split if every rotor walk were free, a few hundredths realistically.
- The cycle-70 docking (rigid-body surrogate, one verification call, gates) is the template: a surrogate minimised
  in the *soft* coordinates only, trusted only where the first xTB force field agrees with it. For a connected
  system the soft coordinates are the dihedrals of the rotatable bonds; bond lengths, angles and rings stay at the
  start values (the model handles those in a few steps).
- Surrogate design from static evidence (formula values at saved geometries only; 358 candidates): (i) the UFF
  torsion rules (sp³–sp³ √(V_i V_j) 3-fold staggered; group-16 pairs 2-fold; sp²–sp² 5 √(U_i U_j)(1 + 4.18 ln BO)
  2-fold planar; sp³-group-16 with sp² 2-fold; sp³–sp² 6-fold V = 1, and the propene rule V = 2 3-fold applied
  *per quad* to the dihedrals ending on the doubly bonded neighbour with the eclipsed minimum — the 3-fold term on
  both neighbours of a trigonal centre cancels exactly, and acetaldehyde / propene / acetic acid eclipse the C=X;
  eclipsed ranks the observed xTB paths better than staggered, 0.749 vs 0.720); (ii) a purely repulsive clash wall
  D_ij (0.65 x_ij / r)¹² on the pairs ≥ 4 bonds apart with the UFF 12-6 parameters, polar H excluded as in the
  docking, the 0.65 fixed a priori from the closest non-bonded contact at the xTB minima (0.01 % quantile of
  r / x_ij = 0.648 over the non-polar pairs) so that the wall does not act inside any minimum; (iii) *no* 12-6
  attraction and *no* point charges: at a thermally distorted skeleton both respond to the bond-angle noise rather
  than to the torsions (with the full 12-6 the fraction of molecules whose observed torsion path lowers the
  surrogate energy drops from 0.76 to 0.42; a screened ε = 4r Coulomb with the docking charges is neutral,
  18 / 353 paths form an intramolecular H-bond). The final surrogate has the observed xTB torsion path downhill for
  0.768 of the candidates (0.760 of those that move ≥ 0.5 Å) and barrier-free along the straight line in torsion
  space for 0.83 / 0.75. The xTB endpoints are not stationary points of the surrogate (median residual torque half
  the start value) and the torsion term alone ranks ~20–25 % of the observed endpoints above the start, so a wrong
  well is expected in about a quarter of the attempts — a basin lottery within the margins (train +1.287 / valid
  +2.926; the connected class holds −0.041 / +0.099 of it, extremes ±0.08 / +0.25).
- Gates as the docking (constants reused, not scanned): the rotor torques of the first xTB force field and of the
  surrogate gradient must have cosine ≥ 0.5; the relaxed geometry must move ≥ 0.5 Å rmsd after a Kabsch
  superposition and ≤ 6 Å per atom; the relaxed geometry must lower the xTB energy (one verification call; on
  rejection the cached start evaluation is re-installed, +1 call). Scope by mechanism: single-fragment starts with
  a net formal charge 0 and an even electron count (the charged conformations are set by electrostatics the
  surrogate does not contain); dimers keep the docking; connected systems without a rotatable bond are untouched
  by construction. Static census: 161 / 257 connected train systems and 200 / 262 valid ones reach the torque gate
  (charged / odd 32 / 38, rotor-free 64 / 24); they hold 0.26 / 0.31 of the score (ours 2763 / 3486 calls against
  ref 3763 / 4931, ratio 0.73 / 0.71) and 6 / 9 of them end in a basin different from the reference's. Three
  dimer-named systems whose shared proton (O···H 1.15–1.17 Å) puts them inside the 1.25 × Cordero bond cut are
  connected at the start and are candidates like any other (the H is untyped, so no rotor runs across the contact;
  the carbonyl O gains a second neighbour and its C=O appears as a 2-fold planar rotor of the acceptor about the
  hydrogen bond, which is the physical in-plane preference, over-stiff).
- Expected: a few tenths of a call to a few calls saved on the far-start walkers (the rotor part of 11.6 calls/Å),
  +1 call on every candidate that fails the energy gate, basin re-rolls on ~25 % of the accepted ones, all other
  runs bit-identical.

### Change (functions)
New block after `_dock_start`: constants `_TORS_WALL_SCALE` 0.65, `_TORS_GROUP16`, `_TORS_UFF_V1`;
`_tors_uff_u1`, `_tors_graph` (1.25 × Cordero bonds), `_tors_hybridisation` (sp³ / sp² / linear-terminal typing,
two-coordinate group-16 atoms typed per rotor), `_tors_rotors` (acyclic bonds with both ends of degree ≥ 2 and not
linear; the side not containing the graph centre rotates; topological distances), `_tors_bond_order` (from the
bond length against the Cordero sum, `_DOCK_BOND_ORDER_LENGTH`), `_tors_torsion_terms` (UFF rule groups per
rotor, V shared by the coherent quads), `_tors_dihedrals`, `_tors_wall_terms`, `_tors_place` (sequential rotations
by `_dock_rotation`), `_tors_torque` (dE/dφ_k = u_k · Σ_moving (r_a − r_j) × g_a), `_tors_energy_gradient`,
`_tors_relax` (L-BFGS-B in the rotor angles via `_scipy_minimize`, Kabsch superposition onto the start),
`_tors_start` (scope, gates, verification call, cache re-install). `minimize_func` calls `_tors_start(atoms,
wrapper)` right after `_dock_start`. Finite-difference check of the torsion-space gradient at 72 saved geometries:
worst relative error 1.7e-7; a rotation changes exactly the dihedrals about its own bond (< 1e-9 elsewhere).

### Result
Candidate `404554df501e3c04c63b4e911276eee1a1e6500f`: train 0.6816413 (Δ +0.0019849 vs 0.6796564; mean rel energy
1.0027262 vs 1.0027439), 469/469 converged — **discard** (valid not run). 40 runs changed, all in the census; 121
census members bit-identical (rejected by the call-free gates: torque cosine < 0.5 or a relaxation move < 0.5 Å);
429 bit-identical in total, dimers / charged / rotor-free untouched. Of the 40 verification calls 30 were rejected at
the energy gate (xTB energy above the start at the surrogate minimum: identical trajectory, +1 call, +0.00264) and 10
accepted: 8 same-basin Σ −6 calls (5 faster: 27 → 22, 25 → 22, 13 → 11, 16 → 14, 16 → 15; 3 slower: 11 → 12,
15 → 18, 11 → 14; −0.00023) and 2 basin changes into higher minima (135142801 30 → 22 at +1.13 kcal/mol, 135227471
16 → 17 at +0.41; credit −0.008; −0.00042). By rmsd₀ (start → cycle-81 endpoint): < 0.5 Å 13 attempts, 11 rejected,
+15 calls; 0.5–1 Å 19 attempts, 11 rejected, −6; 1–1.5 Å 5 / 5 rejected; > 1.5 Å 3 / 3 rejected — every far-start
walker, where the ceiling was, produced an xTB-uphill surrogate minimum. Affected set 800 → 817 calls (ref 1110).
Static diagnosis of the 40 (formula values only, `/tmp/tors83_snap.py`): the torsion-only surrogate is separable over
the rotors, so its descent-reachable minimum is the per-rotor snap of the group terms; the snapped image is *farther*
from the xTB endpoint than the start for 22 of the 30 rejected molecules (median Kabsch rmsd to the endpoint 0.60 →
1.04 Å) with 125 / 242 rotors turned toward the endpoint (a coin toss), against 7 / 10 closer and 48 / 67 rotors for
the accepted ones; and the snap runs substituents through each other — closest ≥ 1-5 contact r / x_ij of the image
0.08–0.31 in 11 of the 30 (start / endpoint 0.5–0.8) — which the wall at 0.65 x_ij (a C···C contact at 2.5 Å,
H···H at 1.9 Å) stops far too late for xTB. The one-rotor case 135076485 (Cl(O)C–N=C=O, C–N–C bent to 104° at the
start against 149° at the minimum) shows the angle side: the snap to planar is the right dihedral (the xTB endpoint
is planar) but the rigid image with the frozen angle puts O···C(1-4) at 2.28 Å (start 2.84, endpoint 3.45), and the
1-4 pairs carry no wall at all. Re-ranking the observed paths under the WCA repulsive branch of the UFF 12-6
(`/tmp/tors83_wca.py`): image below the start 0.77 → 0.48 of the movers (D ≥ 4) / 0.47 (1-4 included), best point of
the straight path below the start 0.90 → 0.82 — the repulsive part also responds to the frozen-angle distortions, so
a physical wall would mostly shrink the moves below the 0.5 Å gate rather than redirect them.
Candidate preserved in `ideas/torsional-docking/404554df501e3c04c63b4e911276eee1a1e6500f/algo.py`; branch reset to
`9988b8d`.

### Interpretation
The live test confirms the static refutation recorded under `torsional-docking` after cycle 70 with a surrogate that
removed the attraction and the electrostatics: the UFF torsional ideal is not where xTB's rotors go (half of the
rotors of the rejected set turn the wrong way; the observed-path ranking of 0.77 measures only whether the endpoint is
downhill, not whether the surrogate's minimum is near it), and with the bond angles frozen at a distorted start the
snapped conformations clash in ways no wall placed inside the observed contacts can see. Unlike the dimer approach
(cycle 70: long rigid-body paths, an electrostatic surrogate that is right about the approach direction, 89 / 87
accepted), the connected torsional path is short in xTB's own metric and the accepted moves save less than a call
each (Σ −6 over 8 same-basin walkers, three of them slower) — even a perfect acceptance of the 40 would have gained
< 0.002 per split. A WCA wall or 1-4 pairs would address only the clash side, cannot be evaluated offline (the
surrogate minima are not computable here), and leaves the 52 % rotor agreement untouched; no bounded repair.

### Next idea / control
Surrogate pre-relaxation of connected starts closed for UFF-class torsion terms with any wall (static after cycle 70,
live here). The verification call of a rejected pose is the only reusable by-product (a secant pair along a large
rotor move); it is not worth a mechanism of its own at 40 attempts per split. Remaining cost map unchanged: dimer
endgame (same-basin intercept ≈ 14 calls) and the large-amplitude rotor paths of connected far starts, which only
xTB's own forces can steer.

## Cycle 84 — class-resolved trust region for multi-fragment systems: the intramolecular coordinates of the fragments take the connected-system radius policy (δ₀ 0.25, cap 0.5, Cartesian bound 2δ on the intramolecular swing), the fragment translation/rotation components keep their tight radius through a weight of 2 in the max-component measure (0.125 → cap 0.25)
Champion at start: `9988b8d39969333516b767fb8b7592d25976b765` (train 0.6796564 / valid 0.6816682).

### Hypothesis
Idea search after cycle 83 (saved-data analysis only, cycle-81 endpoints):
- The first-row model calibration has no lever left. A survey of the guess-Hessian spectra of 22 monomers against
  the fundamentals (`/tmp/freq81.py`, `/tmp/sp2_84.py`) leaves only the aromatic / olefinic in-plane class
  (sp² C–H in-plane bends k 1.25–1.44×, sp²–sp² stretches 1.3–1.5×, ring modes 1.5–1.9×) and the O–H stretch
  (0.75×) outside ±20 %; but sp²-rich connected molecules carry no call excess (regression over 504 same-basin
  connected runs: n = 3.4 + 11.3·rmsd₀ + 2.2·ln N − 3.2·sp²-fraction; near-minimum starts 8.9 / 13.3 / 11.4 / 10.5
  calls for sp² fractions 0 / 0–0.3 / 0.3–0.6 / > 0.6), and the endgame mechanics say why: the stiff modes are bound
  by the force criterion (residual < 4.5e-4/k ≈ 5e-4 Å) and are the first the update learns (cycle 61's O–H
  softening null), the soft modes by the displacement criterion (the *model* step must fall under 1.8e-3 Å), where an
  over-stiff guess is free and a softened one costs a step — the cycle-11/38 size trades of the 1.2–1.3× bend
  refinements. Stiff-class calibrations of 1.3–1.5× are therefore not worth a cycle; the sp² line is closed static.
- Dimer census (`/tmp/dimer84.py`, `/tmp/dimerdih84.py`): after the docking the multi-fragment calls no longer depend
  on the rigid-body start displacement (2.6 / 5.5 calls per Å of whole-complex rmsd₀ against 23 / 28 for the
  reference) but on the *intramolecular* displacement of the fragments: 24 / 26 calls per Å of intra-fragment
  rmsd₀ (train / valid), twice the connected slope (12–15 per Å). At equal size (10–35 atoms) and equal largest
  fragment-dihedral deviation, dimers cost more than connected molecules by +5.9 / +3.4 calls (< 10°: the
  intermolecular endgame), +8.3 / +7.3 (10–20°), +8.1 / +10.6 (20–40°), +10.1 / +11.3 (40–70°) — an excess of
  2–5 calls per rotor-perturbed dimer beyond the endgame, present in the docked neutral and the undocked charged
  classes alike (70 / 52 dimers with a fragment dihedral ≥ 10° off).
- Mechanism: the trust region of a multi-fragment system (δ₀ 0.1, cap 0.25, no Cartesian bound; cycle 22, confirmed
  on the far starts by cycle 44) is a single max-component radius over *all* internal coordinates, so the fragment
  rotors are throttled like the rigid-body coordinates: a rotor 40–70° (0.7–1.2 rad) off needs 0.1 + 0.15 + 0.225 +
  0.25 + 0.25 + 0.25 — six trust-limited steps — where a connected molecule (0.25 + 0.375 + 0.5) needs three; 20–40°
  three to four against two; 10–20° two to three against one to two. That is the size of the observed excess. The
  tight radius was chosen for the fragment translation/rotation coordinates, whose model is a floor diagonal far from
  contact (a scaled steepest-descent step on an anisotropic long-range surface, cycle 44's reading); the
  intramolecular coordinates of the fragments carry the same calibrated model as a connected molecule and should take
  the connected policy.
- Candidate: one radius policy for all minimisations (δ₀ 0.25, growth ×1.5 to the cap 0.5, shrink ×0.5), with the
  fragment translation/rotation components weighted ×2 in the max-component measure (`MaxInternalStep` weight `wx`,
  previously 1) so that their effective radius is 0.125 at the start and 0.25 at the cap — the cap tested on both
  sides (cycles 18/22/44); the Cartesian bound (largest linearised atomic swing ≤ 2δ Å, the cycle-28 lever-arm guard
  now needed by the fragment dihedral chains) measured on the intramolecular part of the step only (fragment
  translation/rotation slots zeroed), so a rigid-body move is bounded by its weighted component alone. Connected
  systems have no fragment slots and are bit-identical by construction. Expected: −1 … −3 calls on the rotor-
  perturbed dimers (Σ ≈ −100 calls per split, −0.004 … −0.007); near-equilibrium dimers unchanged (their steps are
  below either radius); risks: fragment conformer re-rolls under the larger intramolecular steps (the connected class
  lives with the same policy), the valid energy gate resting on the untouched charged dimer carboxylates__ketones
  (+3.2), whose path may change through its intramolecular radius, and the 25 % larger first rigid-body step
  (0.125 against 0.1) of the undocked complexes.

### Change (functions)
- `_default_kwargs['minimum']`: `delta_max_tr` removed; new `wx_tr = 2.0` (weight of the fragment
  translation/rotation components); comments on the radius, the initial radius and the Cartesian bound rewritten.
- `Sella.__init__`: δ₀ = `delta0_mol` for every order-0 run (the multi-fragment exception dropped); `self.wx_tr`
  read from the defaults.
- `Sella._predict_step`: the Cartesian bound (`cart_ratio`) is passed for every order-0 `MaxInternalStep`; for
  systems with fragment coordinates `rs_kwargs['wx'] = self.wx_tr`.
- `Sella.step` (radius update): the growth cap is `delta_max_mol` for every system.
- `MaxInternalStep.cons`: the linearised swing uses the intramolecular part of the step (`_intra_mask`: translation
  slots first, rotation slots last, zeroed) when fragment coordinates exist; `_intra_mask` cached per
  (ntrans, nrotations, nint); connected systems take the previous code path (mask None).
Static check (`/tmp/cons84_check.py`, Jacobian pseudo-inverse of a saved amide–water endpoint, synthetic steps):
weights 2 on the 6 + 6 fragment slots and 1 elsewhere; a 0.1 fragment-rotation component measures 0.2 with its
0.08 Å swing ignored; a 0.25 rad fragment dihedral measures 0.25 with its swing 0.068 Å / 2 below it; the connected
build has no fragment slots (mask None).

### Result
- Train (`evaluation_results/cycle-84-train.json`, candidate ec1d42d): **0.6867286** vs champion 0.6796564
  (Δ +0.0070723), mean rel_energy 1.0027786 (champion 1.0027439), 469/469 converged. **Discard** (train gate).
  Valid not run.
- Decomposition (`/tmp/cmp84.py`, `/tmp/cmp84b.py`): true connected systems bit-identical (249/250 non-des370k
  runs; 104079126 51 → 50 and three des370k pairs that my 1.25 r_cov graph joins but Sella splits, 7 runs Σ −2).
  Multi-fragment 212 runs 70 : 71 : 71, Σ +69 calls, of which **6 basin changes Σ +68** (4 charged: ammoniums__benzene
  17 → 53 (−1.63 kcal/mol, a different cation–π pose), acids__guanidiniums 17 → 37 (−0.24), amides__ammoniums 25 → 42
  (−0.21), alkenes__monoatomics 17 → 13 (+0.48); 2 neutral Σ −1). Same-basin multi-fragment 206 runs 68 : 71 : 67,
  **Σ +1** — neutral −12 (42 : 52 : 37, −0.00045), charged +13 (26 : 19 : 30, +0.00198 through their small refs).
- The targeted class did not move: same-basin dimers by max fragment-dihedral deviation [0,10) Σ −3 (37 : 50 : 31),
  [10,20) +2 (11 : 8 : 15), [20,40) −5 (8 : 8 : 8), [40,181) +7 (12 : 5 : 13). By champion call count: [16,22)
  +15 (19 : 12 : 24), [22,30) −11, ≥ 30 −13 — the long walks gain a little, the mid-length runs lose it.
- Energy credits: Σ(r−1) +1.287 → +1.303 (ammoniums__benzene +0.033, 104079126 −0.022); margin without
  monoatomics__pyrrole +0.515.

### Interpretation
- The hypothesis is refuted on its own terms: the 2–5 call excess of rotor-perturbed dimers over equally perturbed
  monomers is **not** a trust-region throttle — giving the fragment rotors the connected policy (2.5× the initial
  radius, 2× the cap) leaves the [20,40) and [40,181) bins at a wash (−5 / +7 calls over 54 runs, 20 : 21). The
  excess must sit in the coupling the model does not carry: a rotor turn changes the H-bond geometry, so every rotor
  step is followed by rigid-body re-adjustment steps along the floor-stiff inter-fragment modes (an information-
  limited walk that a larger radius cannot shorten), and the endgame is displacement-bound in the coupled soft
  subspace. Consistent with the cycle-44/71 picture (rigid-body walks are neither radius- nor credit-limited).
- The loss is the classic charged-complex failure: with a 2.5× larger intramolecular first step the undocked ionic
  complexes re-roll poses (ammonium N–H excursions toward the benzene face, guanidinium/acid bidentate re-pairing) —
  the same class that decided cycles 74/75, now reached through the radius instead of the docking. A neutral-only
  scope would be a −0.0006 lottery (neutral same-basin −0.00045, basin changes −0.00016) with no mechanism behind
  it: not a repair, credit protection. No bounded repair candidate: the same-basin population is a wash even with
  the basin changes removed (≈ +0.0015).
- Multi-fragment trust-region line now closed on every side: cap 0.5 + bound for the fragment coordinates (cycle 44,
  +0.011), δ₀ 0.25 for docked runs (cycle 71, −0.00002), class-resolved intramolecular policy (this cycle, +0.0071).

### Next idea / control
- Preserved in `ideas/class-resolved-trust-region/ec1d42d85add3d73d39974d3624743f656d7321d/algo.py`; `algo.py` reset
  to 9988b8d (champion, md5 b7ceafbc535f66430674c5cf6ba7c9c5).
- The dimer excess is a coupling problem (rotor ↔ inter-fragment pose, both soft, both invisible to the model
  except through the floor and the secant pairs). Any further dimer lever must supply coupling information or
  shorten the displacement-bound endgame, not step sizes.

## Cycle 85 — torsional pre-relaxation of XH₃ rotors that start on their three-fold saddle: a rotor within 10° of the class saddle is rotated as a rigid group to the class minimum phase before the first force call (staggered for sp³ C / N and two-coordinate O / S frames, eclipsed with C=O for aldehyde / ketone / acid / ester carbons and with the acyl carbon for secondary-amide N-methyls; aromatic, alkene, tertiary-amide, planar-amine, carboxylate and Si / P / S / B frames untouched)
Champion at start: `9988b8d39969333516b767fb8b7592d25976b765` (train 0.6796564 / valid 0.6816682).

### Hypothesis
Idea search after cycle 84 (saved-data analysis only, cycle-81 endpoints):
- Dimer cost decomposition (`/tmp/chdecomp85.py`, `/tmp/dimint85.py`, `/tmp/mephase85*.py`): same-basin dimers
  with intramolecular rmsd < 0.1 Å cost 13.5 (neutral, docked) / 16.0 (charged) calls; intramolecular relaxation
  costs 22–35 calls/Å, 2–3× the connected class, and it is almost entirely methyl rotation of 40–80°. A methyl
  that turns ≥ 40° costs **+7 calls in a dimer** (26.9 vs 15.9 over 17 train dimers) against +1.2 in a connected
  molecule.
- Rotor census by frame class with the reference endpoints as the truth (`/tmp/xy3census85.py`,
  `/tmp/rotrule85.py`, `/tmp/amidetab85.py`): the DES370K template monomers carry every methyl at an *exactly*
  stationary phase (±60.0 or 0.0 to 0.1°). For sp³ frames the staggered template is the minimum (train 210 / valid
  247 stayed staggered, 2 / 2 went eclipsed), for ethers / sulfides too (49 / 39 stayed, 2 / 1 eclipsed), and
  acetyl-type methyls start eclipsed with C=O, their minimum (60 / 49 stayed, 5 / 10 environment-shifted). But the
  N-methyl of the secondary amides (N-methylacetamide-type) starts *staggered* with respect to the acyl carbon,
  which is its **saddle** — the minimum is eclipsed (train 6 of 7 such starts end within 17° of eclipsed in the
  reference, valid 6 of 12, the other 6 being reference runs that never left the saddle) — and the acetaldehyde
  "ketones" carry one methyl staggered with respect to C=O (saddle; train 3 / 3 and valid 3 / 4 end eclipsed).
  Tertiary amides are chemistry-dependent (DMA-type: one methyl eclipsed, one staggered; DMF-type: both eclipsed)
  and Si / P frames (trimethylsilyl) are geared (mixed endpoints, one rotated start would create a 1.68 Å H···H
  contact), so those classes are excluded.
- Mechanism (saddle creep): at the exact saddle the torque vanishes by symmetry, the curvature is negative, and
  the quasi-Newton step along the rotor is g/(h_model + α) ≈ 0; the rotor leaves the saddle only through the
  partner's torque, amplified ~(1 + |k|/h_model) per step (1.3 for a 1 kcal/mol barrier at h_model 0.025), and
  the 60° relaxation shows up in the endgame when everything else is converged: the triggered dimers cost 20–40
  calls against 10–18 for their class (train amides__phenol 34/71, amides__pyridine 31/45, amides__ethers 30/36,
  amides__amines 26/44, alkanes__amides 24/38, amides__ketones 20/30, ammoniums__ketones 29/29, ketones__phenol
  27/43, amides__guanidiniums 40/17). Some runs never leave it: valid amides__ammoniums 9/10, amides__ethers 10/12,
  amides__monoatomics 14/14, guanidiniums__ketones 14/19 converge on the saddle in both ours and the reference,
  and amides__benzene 11/88 (+5.70 kcal/mol vs the reference) / alkenes__amides 12/51 (+4.67) are our runs
  converging on the saddle while the reference escapes to a better basin. Cycles 68/69 (phase-tracked rotor
  stiffness, connected only) never touched this population, and the closure note there reserved the rotor census
  for "a different mechanism".
- Candidate: place saddle-start rotors at the class minimum phase before the first force call — the internal-
  rotor analogue of the cycle-70 docking (a surrogate whose minimum is known analytically by class, cf. the
  torsion rules of ETKDG and the Hehre–Pople–Devaquet eclipsing rule for sp² frames), zero force calls, and a
  pure gain wherever the rule matches the reference endpoint. Expected: train ≈ −0.006 (nine dimers −8 to −15
  calls each); valid ≈ −0.001 to −0.002 (nine escaping dimers −6 to −13 calls each against +2 to +5 calls on the
  four saddle-stayers whose reference is short, plus the two better-basin cases gaining energy credit). The rule
  is wrong for valid amides__carboxylates (reference stays at 59°, ours at 40°) — one exposure. Non-triggered
  molecules are bit-identical by construction (the start is only changed when a rotor is within the window).
- Constants fixed a priori: window 10° (a third of the ±30° negative-curvature region of the three-fold
  potential; the DES370K templates are exact), C=O / C=S thresholds 1.30 / 1.72 Å (the docking code's), planarity
  350°, terminal atoms H only (CF₃ excluded, no data), rotor phase = circular mean of exp(3iφ) with modulus ≥ 0.5.

### Change (functions)
- New section after `_break_start_symmetry`: constants `_ROTOR_SADDLE_WINDOW`, `_ROTOR_DOUBLE_BOND`,
  `_ROTOR_ETHER_PARTNERS`, `_ROTOR_PLANAR_ANGLE_SUM`; `_rotor_phase` (three-fold phase in (−60, 60] and the
  circular-mean modulus), `_rotor_class` (frame rule → reference substituent and 'eclipsed' / 'staggered'),
  `_prerelax_rotors` (bond graph at 1.25 r_cov as in `_start_fragments`; XH₃ rotors = centres with exactly four
  neighbours, three terminal H; rigid Rodrigues rotation of the three H about the frame → centre axis, the
  rotation sense verified on the resulting phase; rotors processed sequentially on the updated geometry).
- `minimize_func`: `_prerelax_rotors` runs before `_break_start_symmetry` (and hence before the docking).
Static checks (`/tmp/prerot85.py`, pure geometry on the saved starts): triggers train 10 molecules (11 rotors:
9 dimers + 135114602), valid 17 (16 dimers + 251920090); every rotated rotor lands within 0.00° of the class
minimum, bond lengths preserved to 1e-15 Å, shortest new non-bonded contact 1.91 Å (amides__amides); all other
starts unchanged.

### Result
- Train (`evaluation_results/cycle-85-train.json`, candidate 8c1813d): **0.6756649** vs champion 0.6796564
  (Δ −0.0039914), mean rel_energy 1.0027411 (champion 1.0027439), 469/469 converged. Train gate passed.
- Valid (`evaluation_results/cycle-85-valid.json`): **0.6796202** vs 0.6816682 (Δ −0.0020480), mean rel_energy
  1.0085721 (champion 1.0062922; Σ(r−1) +2.926 → +3.986), 465/465 converged. **Keep** — new champion
  `8c1813d4560e31e2f0e609ecc06c58ce51ee2d3a` (train 0.6756649 / valid 0.6796202).
- Decomposition (`/tmp/cmp85.py`): every untriggered run bit-identical (train 459/459, valid 448/448). Train
  triggered 10 runs 10 : 0 : 0, Σ −68 calls, all same basin: alkanes__amides 24 → 12, amides__amines 26 → 14,
  ammoniums__ketones 29 → 16, amides__ketones 20 → 14, amides__ethers 30 → 24, amides__phenol 34 → 28,
  amides__pyridine 31 → 25, ketones__phenol 27 → 23, amides__guanidiniums 40 → 38 (credit −6.19 kept), 135114602
  20 → 19. Valid triggered 17 runs 10 : 2 : 5, Σ −30 calls: escaping dimers amides__amines 22 → 12, amides__esters
  38 → 28, acids__amides 25 → 20, amides__sulfides 23 → 19, amides__imidazolium 35 → 32, amides__carboxylates
  38 → 35 (the "wrong-rule" case still gained), imidazolium__ketones 33 → 11 (now the reference's basin, +1.23
  kcal/mol against the champion's lower one); losses acids__ketones 22 → 27 (environment-shifted minimum at −46°,
  the rule's eclipsed phase is 46° off), amides__thiols 20 → 22, amides__amides 23 → 24. The four saddle-stayers
  now reach the minimum at no extra cost — amides__ammoniums 9 → 8 (−3.10 kcal/mol vs the reference),
  amides__ethers 10 → 12 (−4.24), amides__monoatomics 14 → 13 (−2.51), guanidiniums__ketones 14 → 14 (−5.11) —
  and the two better-basin cases follow the reference: amides__benzene 11 → 31 (−5.71 vs the champion, −0.01 vs
  the reference; +0.227 ratio), alkenes__amides 12 → 11 (−4.53, +0.14 vs the reference).

### Interpretation
- The saddle-creep mechanism is confirmed on its own population: with the N-methyl / acetaldehyde methyl placed at
  the class minimum the DES370K amide and aldehyde dimers cost 12–16 calls like their rigid siblings (train
  −6 to −13 calls per run, the mean 26.9 → ~19 of the "methyl turns ≥ 40°" class), and the runs that previously
  converged *on* the saddle go to the minimum for free — the saddle they sat on is 2.5–5 kcal/mol above the
  minimum, far more than a methyl barrier, i.e. the staggered N-methyl also locked the H-bond pattern of the
  complex; the reference sits there too (four valid cases), so the credit is real and large (+1.06 on the valid
  Σ(r−1)).
- The rule's exposure is the environment-shifted minimum (acids__ketones −46°, +5 calls; acetyl methyls in
  esters at 33–58° in a few reference endpoints): a rule cannot know it, and the loss is bounded (a slope start
  instead of a saddle start). A steric guard was considered and dropped — the only new contact below 2.0 Å
  (amides__amides 1.91 Å H···H) cost +1 call.
- Scope notes for the record: connected systems are in scope (135114602 −1, 251920090 ±0) but almost never
  trigger — SPICE starts are thermal, the idealised-template population is DES370K. Tertiary amides and Si / P
  frames are excluded on evidence (DMA vs DMF, geared trimethylsilyl), CF₃ for lack of data.

### Next idea / control
- Champion 8c1813d. Follow-ups in the same mechanism family, not yet tested: (i) 2-fold / 1-fold rotors (X–OH,
  X–SH, X–NH₂) at exact stationary template phases — census needed; (ii) CF₃ / other XY₃ with Y ≠ H (one
  connected case, 385453608 59.7° → 4.7°); (iii) the environment-shifted minima (ester acetyls, acids__ketones)
  are the rule's residual and not addressable by a rule.

## Cycle 86 — the symmetry-breaking displacement is applied to the docked pose too (2026-09-18)

### Hypothesis
- The docking (cycle 70) runs after `_break_start_symmetry`; its L-BFGS-B (gtol 1e-5, ftol 1e-12) converges the
  rigid-body pose to the surrogate minimum. When that minimum lies on a symmetry element (identical partners,
  a mirror-symmetric arrangement of two dipoles, an ion-free stack) the pose is re-symmetrised to ~1e-3 Å or
  better, i.e. the cycle-43 displacement is undone for docked starts, and the optimizer starts from a symmetric
  stationary point of xTB along every antisymmetric rigid-body mode (cycle-42/43 mechanism: zero gradient by
  symmetry, growth only by ~(1 + |k|/λ_model) per step from a 1e-3 Å seed).
- Evidence (pure geometry on saved endpoints, `/tmp/symend86.py`, `/tmp/geom86.py`, `/tmp/docked86.py`): docked
  runs (train 93 of 133 neutral dimers, identified as cycle-70 ≠ cycle-65 results) with a symmetric endpoint
  (`_symmetry_operations`, tol 0.02 Å) that was asymmetric before docking: train pyrrole__pyrrole 16 calls,
  +7.43 kcal/mol above the reference (cycle 65: 33 calls at the reference energy; reference 47) — an antiparallel
  stack; acids__ketones 12 calls, +2.36 (cycle 65: 28); valid pyrrole__pyrrole 18 calls, +17.33 (symmetric stack
  with two 2.42 Å N–H···N contacts vs the 1.82 Å H-bonded minimum), alcohols__alcohols 14 calls, +10.69 (two
  2.58 Å O···H contacts, the saddle between the two O–H···O minima), acids__acids 11 calls, +32.60 (single
  H-bond pose with one symmetry operation vs the C2h double H-bond dimer of reference / cycle 65). Empirical
  cost of the kick on symmetric *minima* from cycle 43 vs 39 (`/tmp/kick86.py`): near-equilibrium symmetric
  minima (rmsd₀ < 0.3, n 17) mean −0.24 calls, median 0; far ones (n 26) −4.46 — the displacement is not a cost
  on stable modes, so the expected effect is: saddle-stayers reach the minimum (energy credits up), creepers
  off a symmetric docked pose get faster, the 22 docked symmetric-minimum runs ≈ neutral.
- Score risk: the two train stayers (pyrrole__pyrrole, acids__ketones) may need cycle-65-like call counts
  (28–33) if their symmetric pose is a saddle; if it is a local minimum nothing changes. The train gate decides.

### Change (functions)
- `_break_start_symmetry(numbers, pos_ang, groups=None)`: optional fragment list (the pre-docking call is
  unchanged, `groups=None` → `_start_fragments`).
- `_dock_start`: after the finite / max-move / min-rmsd checks on the docked move, `pos1 =
  _break_start_symmetry(atoms.numbers, pos1, groups)` before `atoms.positions = pos1` and the verification call
  — the cached evaluation is the geometry Sella starts from, no extra call; rejected dockings, ionic complexes
  and connected systems are bit-identical. Same fixed seed, same 0.02 Å / 0.02 rad constants (never scanned).
- Static check: with the docking's `groups` the displacement equals the start's (`np.allclose`), max atomic move
  0.0585 Å for pyrrole__pyrrole.

### Result
- Train (`evaluation_results/cycle-86-train.json`, candidate 7f22357): **0.6777672** vs champion 0.6756649
  (Δ +0.0021022), mean rel_energy 1.0028358 (champion 1.0027411), 469/469 converged. Train gate failed —
  **discard**; valid not run. Champion 8c1813d restored (`0fff314`), implementation preserved in
  `ideas/docked-pose-symmetry-breaking/7f223571f3a91f6d7479a0eb6911afd5499ee8c2/algo.py`.
- Decomposition (`/tmp/cmp86.py`): 375 undocked / connected / ionic runs bit-identical (the one "unexpected" change,
  amides__amines 14 → 15, is a run docked only since the cycle-85 start change). Docked 93 runs 24 : 49 : 20,
  Σ +30 calls (+0.00205). Same-basin docked runs (90) Σ +10, 22 : 49 : 19 — a lottery. The cost sits in the docked
  runs whose champion endpoint is symmetric (24 runs, Σ +36, 7 : 9 : 8, +0.00259) while the asymmetric ones are
  noise (69 runs, Σ −6, 17 : 40 : 12). Individually: acids__ketones 12 → 38 — its symmetric docked pose *was* a
  saddle, the kick took it to the reference basin (−2.37 kcal/mol) at exactly the reference's 38 calls;
  pyrrole__pyrrole 16 → 16 unchanged at +7.43 — the antiparallel stack is a genuine xTB local minimum (a
  surrogate-basin error, not a symmetry effect); symmetric π-stack / benzene minima paid the endgame of the kicked
  soft antisymmetric modes: amides__benzene 9 → 21, ethers__pyrrole 8 → 13, alkenes__pyrrole 13 → 17,
  alkenes__phenol 8 → 10, pyridine__pyridine 11 → 13; gains esters__esters 28 → 23 but at a stationary point
  +3.00 kcal/mol above the champion's (r 1.00 → 0.96), pyridine__thiols 23 → 18, alcohols__pyridine 26 → 22,
  benzene__water 36 → 33. Energy: docked Σ(r−1) +0.149 → +0.194 (acids__ketones +0.079, esters__esters −0.037).

### Interpretation
- The mechanism is real but its sign under the score gate is negative: of the two train saddle-stayers one was a
  saddle (acids__ketones, fixed at +26 calls — the credit does not count in the score) and one a local minimum
  (pyrrole__pyrrole, unchanged), while the 22 docked runs whose symmetric pose is the minimum pay for the kick.
  Cycle 43's "cost-neutral on near-equilibrium symmetric minima" (−0.24 calls, rmsd₀ < 0.3) does not transfer to
  docked poses: a docked pose is *at* the surrogate minimum, so the kick is the only excitation of the soft
  antisymmetric modes (benzene / pyrrole face rotations, k ≈ 0.01 eV/Å², model floor ≫ k) and their
  displacement-bound endgame (rate |1 − 1/f| with f ≈ 10: ~10 steps to bring a 0.05 Å seed under 1.8e-3 Å)
  is no longer hidden under an approach walk. The cost is inherent to any seed along those modes: a smaller /
  rotation-free seed is a seed scan (excluded), and a saddle-vs-minimum discrimination of the symmetric pose
  needs the xTB curvature along the antisymmetric modes, which nothing in the run provides before sampling.
- Standing evidence for the record: docked symmetric poses are 24 of 93 docked train runs; two are saddle /
  wrong-basin stayers (acids__ketones saddle, pyrrole__pyrrole surrogate-basin error). The repair is a validity
  lever only (+0.045 train Σ(r−1) for +0.0021 score) and is parked for a validity emergency.

### Next idea / control
- Docking symmetry follow-up closed under the score gate (parked in `ideas/docked-pose-symmetry-breaking/`).
  The cost map is unchanged: connected share 0.41 at a uniform ratio 0.70–0.81, neutral dimers 0.46, charged
  dimers 0.80 (the worst class, undocked by scope).

## Cycle 87 — Aitken (multiple-root) acceleration of a creeping quasi-Newton endgame (2026-09-18)

### Hypothesis
- The persistent endgame excess of the champion sits on flat-bottomed modes: the acceptor-class regression
  (`/tmp/strength50.py cycle-85`, 158 same-basin het-donor H-bonded dimers, both splits) leaves +2.4…+3.8 calls on
  carbonyl-type (single-neighbour) acceptor contacts and +3.4 on short (1.3–1.7 Å) ionic contacts after the
  contact-bend terms of cycles 45–49; the shared-proton census (`/tmp/proton87.py`) puts the most centred protons
  (N–H 1.16–1.24 Å) at +3…+7 calls (ammoniums__pyridine train 25 vs 19, valid 18 vs 11, ammoniums__pyrrole 23 vs
  17, imidazolium__pyridine 22 vs 19, amines__ammoniums 17 vs 12); cycle 86 measured ~10 displacement-bound steps
  for a 0.05 Å seed along a face rotation whose model stiffness is ~10× the true one.
- Cycle-50 theory, re-derived: on a mode whose gradient has a multiple root (V ∝ x⁴, g ∝ x³) any secant/Newton
  scheme converges only linearly — Newton with the local curvature contracts at 2/3, the secant (interval-average)
  curvature at ρ = 0.7549 (root of ρ³ + ρ² − 1); a mode the model over-stiffens by f without the update sampling it
  contracts at 1 − 1/f. The displacement criterion (1.8e-3 Å) then binds for ≈ 7–10 calls from a 0.02–0.05 Å
  step. Attractive contacts approached from far and cosine torsions *bounce* (secant softer than local), which is
  not creep; relaxing out of a compressed Morse wall creeps only transiently.
- Remedy — Aitken's Δ² process (classical multiple-root acceleration, also Steffensen): for a linearly converging
  sequence with ratio ρ the remaining distance after the step d_n is d_n ρ/(1−ρ); equivalently the next model step's
  component along the creep direction (asymptotically ρ d_n) is short by 1/(1−ρ) (4.08 for the pure quartic). Not
  the GDIIS closure (a linear-model extrapolation already carried by the secant update): Aitken acts on the
  *iteration*, i.e. on exactly the modes where a quadratic model — however well updated — cannot converge fast.
- Detector (fixed a priori, not scanned): three consecutive descent steps of the current internal coordinate set,
  none trust-bounded or extrapolated, collinear with the latest one (cos ≥ 0.85), with ratios in [0.3, 0.9] that
  agree within 25 %, and gradient projections on the direction of one sign with strictly decreasing magnitude.
  Expected: −3…−6 calls per creeping endgame (carbonyl-acceptor and ionic dimers, shared protons, face rotations,
  inversion-type flat modes), misfires (ratio still falling → overshoot by ≤ a few small steps) ≈ +1 call, small
  energy-gate risk from a jump over a tiny barrier on a flat surface (the extra step is ≤ 3× a step the run was
  already taking and stays inside the trust region).

### Change (functions)
- `PES.__init__`: `_df_pred_corr` (additive correction of the next kick's energy prediction), `_g_cart_stash`.
- `PES.kick`: `df_pred += _df_pred_corr` when set (then reset).
- `InternalPES.eval`: stashes the Cartesian gradient with the state hash; new `InternalPES.creep_gradient()`
  returns it for the current geometry (None otherwise).
- `_default_kwargs['minimum']['creep_accel'] = True`; `Sella.__init__`: `creep_accel`, `_creep_hist`.
- `Sella._creep_record(special)`: appends (real-atom positions, Cartesian gradient, energy, special = trust-bounded
  or extrapolated step) — called after the initial evaluation in `_predict_step` (history reset there and in the
  rebuild branch of `step`) and after `kick` in `step`; four points kept.
- `Sella._creep_step(s, smag, rs_kwargs)` (called in `_predict_step` after `_curved_step`): the detector above on
  the last four points; the model step's Cartesian component c along the unit direction u of the latest step must
  be > 0; c_acc = min(min(1/(1−ρ), 4)·c, a₃ρ/(1−ρ)); the extra Cartesian displacement (c_acc − c)u is mapped
  through the Jacobian into the free internal subspace and added to s; the unchanged trust region bounds it (largest
  fraction of the extra that keeps the MaxInternalStep measure ≤ δ, by bisection); the energy-prediction correction
  −½ k (M_eff − 1) M_eff c² (k = uᵀBᵀHBu, the model curvature along u; M_eff = c_acc/c) makes the trust ratio judge
  the step against the model softened along u by M_eff (a landing at the minimum of a quartic gives ρ_trust ≈ 0.5,
  of an over-stiffened quadratic ≈ 1). Class constants `creep_cos` 0.85, `creep_rho_min/max` 0.3/0.9,
  `creep_rho_tol` 0.25, `creep_max_mult` 4.
- Commit `d22a111` (algo.py md5 98c7bae9d1d45caacfeff418f424214f).

### Result
- Train (`evaluation_results/cycle-87-train.json`, candidate d22a111): **0.6765739** vs champion 0.6756649
  (Δ +0.0009089), mean rel_energy 1.0027411 (identical to the champion), 469/469 converged. Train gate failed —
  **discard**; valid not run. Champion 8c1813d restored (`d3c5ac9`), implementation preserved in
  `ideas/creep-acceleration/d22a1110959bcaab1d780e3b3948ce665038414c/algo.py`.
- Decomposition (`/tmp/cmp87.py train 87 85`): 452 of 469 runs bit-identical (count and energy) — the detector
  fired in 17 runs only (connected 16 of 257, neutral dimers 1 of 133, charged dimers 0 of 79). Where it fired:
  1 better (apalutamide 15 → 14), 7 same count, 9 worse (+1 each; 252620869 +2): cariprazine 21 → 22, palbociclib
  12 → 13, abemaciclib 11 → 12, 252635561 15 → 16, 135099280 15 → 16, 135099604 11 → 12, 135099623 14 → 15,
  amines__thiols 12 → 13. All endpoints are the same minima (|ΔE| ≤ 3e-5 kcal/mol); the slower runs end slightly
  *deeper* (−1e-6…−3e-5 kcal/mol), i.e. the extrapolated step did move into the flat bottom and the displacement /
  energy criteria then needed one more settling step.

### Interpretation
- The premise — a clean, collinear, geometrically shrinking creep with a stable ratio — is rare: 3.6 % of train
  runs ever show the three-step pattern, and none of the 79 charged dimers and only one of the 133 neutral dimers,
  the classes carrying the endgame excess this cycle was aimed at (carbonyl-type acceptors +2.4…+3.8 calls,
  short ionic contacts +3.4, centred shared protons +3…+7). Their slow endgames are therefore not a 1-D multiple-
  root creep in Cartesian direction: the steps rotate (several soft modes decaying at comparable rates, or a
  curved valley such as the proton-transfer coordinate, whose N–H stretch and N···N contraction are not
  collinear), bounce, or are trust-bounded — none of which Aitken's Δ² addresses.
- Where the pattern does occur (drug-like connected molecules with piperazine / amine rotors: cariprazine,
  palbociclib, abemaciclib) it is the tail of a transient: the secant iteration was about to converge
  superlinearly (the champion needed ≤ 1 more call), so the extrapolation by 1/(1−ρ) overshoots and costs a
  settling step (9 : 7 : 1). The pure-quartic picture (ρ = 0.755 for many steps) does not describe xTB endgame
  modes, which are quadratic + quartic with a short quartic regime; the mixed-mode ratio is still falling when the
  25 % consistency window accepts it. A damped extrapolation would at best convert the +1s into 0s on 17 runs
  (ceiling ≈ 0.001) — not a cycle's worth; a two-step detector would fire earlier in the transient and overshoot
  more.
- Line closed: multiple-root / Aitken acceleration of the quasi-Newton endgame (three-step detector, factor ≤ 4,
  trust-bounded, softened-model trust ratio). The remaining dimer endgame excess needs a diagnosis of *which*
  pattern the slow endgames follow (rotating soft-mode directions vs curved valleys vs trust-bounded steps), which
  the saved results cannot provide (no trajectories).

### Next idea / control
- Cost map unchanged (champion 8c1813d). The creep/Aitken line joins the closed endgame lines (GDIIS, step
  rejection, energy-augmented secant, shrink-rule tightening). Next: idea search outside the endgame-step
  machinery — model content for the multi-fragment classes (the charged-dimer share 0.80 is the worst class)
  or the start geometry of connected systems (rotor turns drive the connected cost).

## Cycle 88 — ester acetyl methyls (CH₃–C(=O)–O–R) are placed staggered with respect to C=O by the torsional pre-relaxation (one C–H eclipsing the C–O(alkyl) bond, the GFN2-xTB minimum of the ester acetyl rotor; the C=O-eclipsed rule kept for aldehyde, ketone, acid and amide carbons and for ester carbons in systems with a formal charge) (2026-09-18)
Champion at start: `8c1813d4560e31e2f0e609ecc06c58ce51ee2d3a` (train 0.6756649 / valid 0.6796202).

### Hypothesis
Idea search after cycle 87 (saved-data analysis and code / log reading only; cycle-85 endpoints). Static closures,
for the record: 2-fold / 1-fold rotor saddle rule (X–OH, X–SH, X–NH₂ at exact stationary template phases — no
population); near-symmetric images (6 molecules); reference-bin share analysis (ref ≤ 20 carries share 0.367 at
ratio 0.84–0.90); Berny-type line search (no lever with the trust policy); charged ammonium-donor subset ≈ 0.004
ceiling; regression / intercept findings (connected same-basin: ours = 6.84 + 8.67 rmsd₀ + 0.17 nrot + 1.16 nturn
+ 0.10 nheavy, R² 0.617, vs reference 7.89 + 12.53 rmsd₀ + 0.37 nrot + 1.84 nturn + 0.17 nheavy; per-turner costs
sp³–sp³ 1.85 / 4.06, sp³–sp² 1.90 / 2.90, sp²–sp² 0.36 / 0.83, X–OH 1.96 / 4.09 — a path-length cost, cycle-41
interpretation); stretch-residual / Kekulé regression, noise-floor exemption, rebuild census, rigid-body floor,
GPR / PSB / coupling-term / shared-proton assessments, near-equilibrium class decomposition, strength50 class
residuals, composition regression, exocyclic N3 survey, nbt cluster regression, Almlöf calibration table and freq81
stiff-mode spread, neutral-dimer floor ≈ 9 calls, maxd regression, per-mode hybrid upper bound ≈ 0.0013, heavy-row
dimer residuals null, nlev-vs-nbt refutation, no trajectories in the results, pipeline scope check, charged-dimer
class decomposition (basin-changers ≈ credit walkers), near-eq ours ≥ ref only in inorganic H-free molecules;
DREIDING 12-10 surrogate term refuted statically (`/tmp/dockres88.py`: the residual rigid-body force of the
surrogate at the xTB endpoints is not reduced by a directional H-bond term), implied surrogate-minimum offset
uncorrelated with calls (`/tmp/dockres88b.py`), start bond-length deviations (0.045 Å rms) not predictive
(`/tmp/bondlen88.py`); torsion-class refinement / per-type LS fit / homo-dimer images / connected-refinement
extension to dimers not worth a cycle; full rotatable-bond census of connected runs (`/tmp/rotcensus88.py`,
`/tmp/rotcensus88b.py`: no saddle-start population in SPICE-type starts — sp³–sp³ eclipsed 3, aliphatic OH
eclipsed 2, sp³–sp² and sp²–sp² minima environment-dependent); XH₃ untouched-frame census (`/tmp/xy3census88.py`:
alkene methyls 32 of 38 already at the eclipsed minimum, 6 cis-crowded turners; aromatic methyls free rotors);
charged near-equilibrium excess = path lottery (`/tmp/trace88.py`: counts bounce across cycles); force-based
docking acceptance rejected in reasoning (it would reject the far-start winners); Sella's back-transformation
reviewed (standard); textbook small-class bundle (heterocumulene linear bends 0.14 + P-only torsion floor) kept
as a fallback.
- The residual after the cycle-85 pre-rotation (`/tmp/xh3resid88.py`, `/tmp/acyl88.py`: every XH₃ rotor of the
  same-basin runs, phase of the pre-rotated start (`algo._prerelax_rotors` applied offline — pure geometry) against
  our and the reference endpoints, by frame class and by the acyl carbon's third substituent Z):
  - acetyl methyls with Z = O–R (methyl / ethyl acetate) in dimers: 14 rotors, 12 leave the C=O-eclipsed template
    phase by 31–60° (end phases −60, 58, 51, −47, 45, −42, −40, 33, −33, −32, −31 — *identical in the reference
    endpoints to 0.1°*), the 2 stayers are ionic complexes (ammoniums__esters, esters__guanidiniums); including
    basin-changers: 14 of 14 neutral-partner complexes leave (acids__esters, alcohols__esters: ours leaves, the
    reference converged on the phase-0 start), 4 of 5 ionic ones stay (also esters__monoatomics,
    carboxylates__esters; esters__imidazolium goes to −60). Connected ester acetyls (251920090 ×4, 103938489) end
    16–30° off eclipsed.
  - Z = O–H (acetic acid) 17 of 20 stay within 15° of eclipsed; Z = C(sp³) (acetone) 15 stay / 9 drift 15–30° (mean
    |p| 11.7°, a flat well around eclipsed); Z = N (amides) stay. So "eclipsed with C=O" is right for aldehyde,
    ketone, acid and amide carbons and wrong for the ester carbon: on the GFN2-xTB surface the C=O-eclipsed phase
    of the ester acetyl methyl is the saddle and the group settles staggered to C=O (one C–H eclipsing the
    C–O(alkyl) bond), scattered ±30° by the partner (the same one-sided scatter the acetone methyls show about
    their eclipsed minimum). Physically the ester acetyl rotor has the smallest barrier of the acyl series
    (methyl acetate V₃ ≈ 0.28 kcal/mol from microwave spectroscopy vs ≈ 1.1 acetaldehyde, 0.78 acetone, 0.48 acetic
    acid), so its xTB minimum need not be the experimental one, and a preference that small is overridden by an
    ionic contact (cation on the carbonyl O or anion on the methyl H's keep it eclipsed: 4 of 5 cases).
  - These are the "environment-shifted ester acetyls" that cycle 85 filed as the rule's residual; the class-resolved
    census shows they are a *class* property, i.e. an analytic rule exists.
- Candidate: `_rotor_class` returns ('staggered' with respect to the carbonyl O) for a carbon frame with exactly one
  C=O and an O substituent that carries a heavy-atom partner (ester / anhydride carbon), unless the system has a
  formal charge (`_dock_formal_charges` by valence rules — the docking code's, so the ionic scope is the established
  one); acids, ketones, aldehydes, amides and ester carbons of ionic systems keep 'eclipsed'. Expected residual
  after a 60° pre-rotation 0–29° (mean 17°) instead of 31–60°; cycle-85 rates (−4 to −7 calls per dimer for a
  60° saddle start; +2 to +5 for a 30–45° miss) give train ≈ −0.001 (5 same-basin dimers + acids__esters), valid
  ≈ −0.001 to −0.002 (6 same-basin dimers, 251920090 with three rotors now placed further from their crowded
  endpoints, esters__phenol / alcohols__esters / amides__esters basin-changers). All other starts bit-identical by
  construction. No constants added; the window (10°), phase definition and rotation code are cycle 85's.

### Change (functions)
- Rotor pre-relaxation note: the acyl rule now names aldehyde / ketone / acid / amide carbons; the ester exception
  and its ionic scope are documented.
- `_rotor_class(numbers, pos, adj, dist, c, b, ionic=False)`: in the two-substituent carbon branch with exactly one
  C=O / C=S, the other substituent `other` is inspected — `numbers[double[0]] == 8 and numbers[other] == 8 and any
  heavy neighbour of other besides b` and `not ionic` → `(double[0], 'staggered')`; otherwise `(double[0],
  'eclipsed')` as before.
- `_prerelax_rotors`: `ionic = any(|_dock_formal_charges(numbers, adj, dist)| > 1e-6)` once per system, passed to
  `_rotor_class`.
- Static check (`/tmp/prerot88.py` before / after, pure geometry on all 934 starts): triggers 27 → 40 molecules
  (28 → 43 rotors); 25 unchanged, new: train acids__esters, alkanes__esters, amides__esters, amines__esters,
  esters__ketones, esters__thiols; valid alcohols__esters, alkanes__esters, alkenes__esters, benzene__esters,
  esters__ethers, esters__phenol, esters__pyrrole; changed: 251920090 (the −50° acetyl formerly rotated to 0 now
  left at its staggered-side phase, the two acetyls at −3° / 1° rotated to ±60) and amides__esters__valid (the
  acetyl rotated in addition to the amide N-methyl). Ionic ester complexes untouched (ammoniums__esters,
  esters__guanidiniums, esters__monoatomics, carboxylates__esters, esters__imidazolium). New intramolecular contact
  after the rotation: H···O(alkyl) 2.38 Å (was 2.64 Å to another H). Commit `8a30dec` (algo.py md5
  a0ce6137c1fc1875efb4e705427310b2).

### Result
- Train (`evaluation_results/cycle-88-train.json`, 72f2b8d7b736f4e13594d626f96c3916): **0.6740826** vs champion
  0.6756649 (Δ −0.0015824), mean rel_energy 1.0027410 (champion 1.0027411; Σ(r−1) +1.286 unchanged), 469/469
  converged. Train gate passed.
- Valid (`evaluation_results/cycle-88-valid.json`, 2083b06b35a507bd050183255db01dc2): **0.6765981** vs 0.6796202
  (Δ −0.0030221), mean rel_energy 1.0085820 (champion 1.0085721; Σ(r−1) +3.986 → +3.991), 465/465 converged.
  **Keep** — new champion `8a30dec9b7cf843c34cc0feefaa961e8a84b33a8` (train 0.6740826 / valid 0.6765981).
- Decomposition (`/tmp/cmp88.py`): train 463 / 469 bit-identical, the 6 triggered runs 6 : 0 : 0, Σ −25 calls
  (Σ Δrel_steps −0.742): esters__thiols 26 → 20 (ref 51), amides__esters 27 → 23 (41), esters__ketones 23 → 19
  (44), amines__esters 19 → 16 (63), alkanes__esters 23 → 20 (96), acids__esters 33 → 28 (ref 14; still its own
  basin 0.38 kcal/mol below the reference, r 1.073 kept). Valid 456 / 465 bit-identical, 9 triggered runs 8 : 0 : 1,
  Σ −41 calls (Σ Δrel_steps −1.405): esters__phenol 38 → 16 (ref 33; the pre-rotated start now converges in the
  reference's basin, r 0.995 → 1.000), alcohols__esters 33 → 24 (ref 18, own basin −0.05 kcal/mol), esters__ethers
  21 → 17 (51), amides__esters 28 → 24 (70), benzene__esters 28 → 25 (57), alkanes__esters 23 → 21 (39),
  esters__pyrrole 29 → 26 (68), 251920090 32 → 31 (64), alkenes__esters 20 → 22 (33; endpoint −32°, a 28° miss
  from the rotated start).
- Endpoint phases after the change (`/tmp/acyl88.py 88`): every rotated ester acetyl ends where it ended before
  (to 0.3°; the Kabsch same-basin flag of the analysis scripts now reports these runs as basin-changers only
  because the three methyl H labels are permuted by the 60° rotation — use phases or permutation-invariant
  comparisons for them); residual turns from the rotated start 2–29°.

### Interpretation
- The ester acetyl methyl is a class-level exception to the Hehre–Pople–Devaquet eclipsing rule on the GFN2-xTB
  surface, and treating it as such is worth −3 to −6 calls per neutral ester dimer (−0.0016 / −0.0030), the same
  saddle-creep mechanism as cycle 85 with the same rates: a 60° start-to-minimum path costs +4 to +7 calls in a
  dimer, a 20–30° path ≈ +1 to +2. The valid gain is larger because two basin-changers went the right way
  (esters__phenol reached the reference's basin 22 calls cheaper and its r 0.995 → 1.000; alcohols__esters −9) —
  a lottery component on top of the systematic −2 to −4 per same-basin dimer.
- The ionic scope was right on the evidence available (the 4 stayers are untouched, bit-identical) and cost
  nothing on the one ionic complex that does leave the eclipsed phase (esters__imidazolium 23 vs ref 44, unchanged).
  A local rule (contact of the carbonyl O / methyl H's with a charged group) would be a fit to 5 points; the
  system-level formal-charge scope is the docking precedent and is enough.
- The flat-well scatter (±30° around the class minimum for esters, ±20° for acetone) is the rule's floor: a rule
  cannot place a rotor better than the class minimum, and the remaining 2–29° residuals are the "environment-
  shifted" part cycle 85 correctly declared not a rule's business.

### Next idea / control
- Champion 8a30dec (train 0.6740826 / valid 0.6765981). The XH₃ pre-rotation family is now: sp³ / pyramidal /
  ether frames staggered, acyl frames eclipsed except esters (staggered) and secondary-amide N-methyls (eclipsed
  with the acyl carbon); untouched: aromatic / alkene / carboxylate / tertiary amide / planar amine / Si / P / S / B.
  Remaining census evidence: no other frame class has a systematic template-phase-vs-endpoint offset (ketones
  drift ±20° around eclipsed in a flat well — no rule; alkene methyls sit at their minimum). The XH₃ line is
  closed unless new molecules appear.
- Next: the same "class minimum known analytically, start placed there for zero calls" logic for the other
  start-geometry populations that the census found (none in connected SPICE starts) — or model content for the
  charged-dimer class (share 0.80).

## Cycle 89 — connected-only small-class calibration bundle: heterocumulene dummy linear bends 0.10 → 0.14 Ha/rad² and a 0.014 Ha/rad² rotational-stiffness floor for acyclic σ/lone-pair bonds with phosphorus at one end (non_generalizable) (2026-09-18)
Champion at start: `8a30dec9b7cf843c34cc0feefaa961e8a84b33a8` (train 0.6740826 / valid 0.6765981).

### Hypothesis
Idea search after cycle 88 (saved-data analysis and code / log reading only; cycle-88 endpoints). Static closures,
for the record: free antisymmetric seed pair for symmetric starts (no target population; saddle-return risk);
bond-length / angle start pre-correction (start bond deviations 0.045 Å rms do not predict calls, `/tmp/bondlen88.py`);
non-XH₃ rotor start rules (no class minimum: sp³–sp² and sp²–sp² minima are environment-dependent, `/tmp/rotcensus88.py`);
surrogate-physics improvements for docking (the implied surrogate-minimum offset does not predict calls,
`/tmp/dockres88b.py`); anion docking EV ≈ 0 (cycle-74 scope); rotor pre-relaxation already covers dimer fragments
(no scope extension); charged credit walkers (train 17 / valid 19 basin-changers with r > 1 at ≈ the reference's
cost) cost 0.016 / 0.023 in score but carry Σ(r−1) +1.81 / +4.77 of the +1.29 / +3.99 margins — untouchable;
a force- or surrogate-Hessian-based per-run trust-credit discriminator for docked poses (cycle-71 follow-up) too
crude (the surrogate H-bond radial curvature is dominated by the O···O 12-6 wall, 2–4× too stiff, so the far-start
winners would be under-credited; a force threshold would be a scan); docking secant-pair seeding closed for good
(a ≥ 0.5 Å chord from a far weak-force start under-reads the H-bond curvature 4–10×); gyration-radius scaling of
the rotation coordinates / multi-fragment Cartesian bound rejected on the cycle-44 evidence (would re-roll the
ionic walkers); neutral same-basin dimer cost = 14.71(±2.16) + 16.97(±2.37)·intra − 0.149(±0.095)·N +
1.91(±0.86)·rmsd₀ (n 161; intra = max per-fragment Kabsch rmsd start → end): the ≈ 14.7-call floor is the
inter-fragment soft-mode learning phase (≈ 6 calls over a connected near-equilibrium run, ≈ one call per rigid
mode learned), the intramolecular relaxation costs what it costs in connected systems.
- Candidate (the fallback bundle named in cycle 88): two textbook calibrations of small classes, connected systems
  only, constants from cycles 67 / 41 (not scanned):
  (i) `_h0_linear_bend`: a two-bonded near-linear centre whose two bonds are both shorter than 0.895 × the
  stiffness-radius sum and which is not an all-carbon allene (isocyanate, isothiocyanate, carbodiimide, ketene,
  ketenimine, azide, diazo) gets 0.14 Ha/rad² instead of 0.10 (Wilson-GF fits of the textbook fundamentals:
  HNCO 0.60–0.77, CH₃NCS 0.63, HN₃ 0.53, ketene 0.37–0.54 mdyn Å/rad² = 0.09–0.18 Ha/rad²); X–C≡Y, H–C≡, allene
  and hydrogen centres unchanged. Hard threshold instead of the cycle-67 ramp; the census (`/tmp/bundle89.py`,
  all 122 near-linear centres of both splits listed) puts the class at ratios ≤ 0.881 and the nearest excluded
  centre at 0.897 (a diazo N=C at the start), so the start-geometry partition is unambiguous.
  (ii) `_rotor_floor_factor` (cycle-41 code, P-only scope): an acyclic bond between two σ/lone-pair centres with
  phosphorus at one end and a main-group non-metal at the other, not in a ring ≤ 8, has its rotational stiffness
  k_rot = h_FA·√n lifted to 0.014 Ha/rad² (methylphosphine 9/2·V₃) when it is below it (census factors: C–P
  1.4–7.4, S–P 3.6–10.8, B–P 2.8–4.7, O–P 1.0–2.0, N–P 1.0–3.4). Thiols / disilane (below the floor physically)
  and the large-amplitude disulfide / polysilane detours of cycle 41 are outside the scope by construction.
- Scope (census + static guess comparison `/tmp/stub89.py`: Internals + `guess_hessian` of all 934 starts with the
  champion and the candidate modules, no calculator): train 9 P-floor + 4 heterocumulene molecules, valid 7 + 4
  (the two C–P bonds at near-linear carbons, 135093102 / 135093104, carry no proper dihedrals and are untouched);
  every multi-fragment system and every other connected molecule bit-identical at the start (24 of 934 guesses
  changed, only the listed dummy bends by exactly +0.04 Ha/rad² and the P-bond dihedrals). Mid-run straightening
  rebuilds can bring the constant in elsewhere.
- Expected ≈ −0.001 per split (cycle 41 P subset 6:0:4 Σ −12 calls post hoc = −0.0009 on train; cycle 67
  heterocumulenes −7 calls over 8 molecules = −0.0003 train / +0.0001 valid in score terms, the azide 5 → 6 with
  ref 7 outweighing 38 → 34 with ref 41); path noise ±1–5 calls per molecule; gate risk moderate.
- P-rotor amplitudes under the champion (`/tmp/amp_vs_change.py` on cycle-88 endpoints): 30–90° for all but two
  molecules (135009613 8°, 104125231 4–10°), i.e. the large-amplitude population where cycle 41 found the stiff
  model losing in general; the P subset's 6:0 was the exception being tested.

### Change (functions)
- `Internals._h0_linear_bend(centre, k_bend=0.10, k_bend_h=0.05, k_cumulene=0.14, r_multiple=0.895)`: scans
  `self.internals['bonds']` at the centre (dummy bond skipped), `bond.calc(self.all_atoms) < r_multiple ×
  (_STIFFNESS_RADII[z] + _STIFFNESS_RADII[z_other])` for both partners and `not (z == 6 and partners == [6, 6])`
  → `k_cumulene`; otherwise as before.
- `Internals._rotor_nonmetals` (class attribute, cycle-41 set) and `_rotor_floor_factor(b, c, adj, k_rot,
  k_floor=0.014, ring_max=8)`: 1.0 unless `15 in (zb, zc)`, both in `_rotor_nonmetals`, `k_rot < k_floor·Hartree`
  and not `_in_small_ring`; else `floor / k_rot`.
- `_torsion_class_factor(..., k_rot=None)`: the non-pi branch returns `_rotor_floor_factor` when `k_rot` is given.
- `guess_hessian`: `h_fa = self._h0_dihedral(dihedral, nbonds)` once per proper dihedral, `k_rot = h_fa·√ndih[key]`
  passed for connected systems (`scale_torsions`), `h0 = tfac·h_fa/√n` unchanged otherwise.
- Static checks: py_compile, pyflakes diff vs HEAD clean (only the pre-existing `dpos_first` line shift),
  `/tmp/stub89.py` as above. Commit `ee5c5798ee8fabf56a76dcfc31e369775c9d7969` (algo.py md5
  65b30b5fc9c82cf2ee4533634ee384f6).

### Result
- Train (`evaluation_results/cycle-89-train.json`, 70a9f2b16489cacd16a894acb09a1812; evaluation
  e3dd8e208f764673b9dd0336fd6643eb): **0.6734445** vs champion 0.6740826 (Δ −0.0006380), mean rel_energy 1.0027410
  unchanged (Σ(r−1) +1.286), 469/469 converged. Train gate passed.
- Valid (`evaluation_results/cycle-89-valid.json`, 0dc2c961d39b0067d8c0affbccafd5f5; evaluation
  fbbc4fd7a7464a70b6c8cbeaa4105aa0): **0.6772259** vs 0.6765981 (Δ +0.0006278), mean rel_energy 1.0085820 unchanged
  (Σ(r−1) +3.991), 465/465 converged. Valid gate failed → **non_generalizable**.
- Decomposition (`/tmp/cmp89.py`): train 453 / 469 bit-identical, 16 changed runs 8 : 5 : 3, Σ Δrel_steps −0.299;
  valid 453 / 465 bit-identical, 12 changed 4 : 2 : 6, Σ +0.292. No basin changes, all energies identical.
  - P-floor class: train 9 molecules 5 : 1 : 3, Σ +2 calls, Σ Δrel +0.054 (135098013 17 → 16, 135130097 38 → 36,
    135009613 20 → 19, 135092350 15 → 14, 252089162 39 → 38, 135049744 24 → 24, 135097653 17 → 18, 135101309
    22 → 23, 135316214 23 → 29); valid 7 molecules 3 : 2 : 2, Σ −1 call, Σ Δrel +0.029 (135092258 12 → 11,
    135106724 16 → 15, 135101049 34 → 33, 135094940 28 → 28, 104125231 13 → 13, 135100404 23 → 25, 135107889
    12 → 14). Over both splits 8 : 3 : 5, Σ +1 call — the cycle-41 6 : 0 : 4 did not replicate.
  - Heterocumulene class: train 4 molecules 2 : 2 : 0, Σ −4 calls, Σ Δrel −0.130 (440717067 28 → 25, 135076485
    17 → 16, 135104646 20 → 20, 104040973 4 → 4); valid 4 molecules 0 : 0 : 4, Σ +8 calls, Σ Δrel +0.321
    (251920090 31 → 32, 135239978 33 → 34, 135249279 33 → 38, 135041958 5 → 6). Over both splits 2 : 2 : 4, Σ +4;
    135249279 went 38 → 34 in cycle 67 and 33 → 38 now, 135041958 (azide, ref 7) 5 → 6 both times.
  - Mid-run straightening rebuilds (centres that cross 165° during the run and get the dummy constant at the
    rebuild geometry): train SiS₂ 135041910 9 → 7 (Si=S ratio 0.914 at the endpoint, i.e. the bond overshoots
    below the 0.895 threshold at the rebuild — the hard threshold's sensitivity), SiO₂ 134987421 8 → 8, OOBO
    135159584 10 → 10; valid 104129283 (a P–O–P bridge crossing 165°, its P–O bonds re-floored at the rebuild
    length) 38 → 35. Σ Δrel −0.222 train / −0.057 valid: the train pass is the SiS₂ lottery (−0.222 of the −0.299)
    plus the two heterocumulene gains; without the rebuild lottery train would be −0.077 → −0.00016.

### Interpretation
- Neither component carries a class signal under the present champion. The P-floor is a wash on both splits
  (Σ Δrel +0.05 / +0.03; the rotors turn 30–90°, the population where a stiff phase-blind guess has no advantage,
  and the transported 4-pair secant memory learns a 1.4–11× too soft rotor within the first steps anyway — cycle
  58's per-rotor stiffness learning found the same); the cycle-41 P subset 6 : 0 : 4 was small-sample luck, as its
  own log suspected. The heterocumulene stiffening is 2 : 2 : 4 over both splits with per-molecule signs flipping
  between cycles 67 and 89 — path noise of ±1–5 calls on molecules whose linear bend is not the slowest mode; the
  "over-stiff soft mode is free" premise does not hold for the isocyanate 135249279 (+5), whose N=C=O bend starts
  at 167° and carries the gradient. The two components were each ≈ −0.001 expected and ±0.0006 realised with
  opposite signs on the two splits: small-class calibrations below the ±0.001 path-noise floor cannot be resolved
  by this evaluator, even bundled, and the deterministic re-contribution argument of cycle 67 fails because the
  paths of the molecules are not deterministic across champions (every intervening keep re-rolls them).
- Both components are closed: the linear-bend constant stays 0.10 / 0.05 (cycles 39, 67, 89), the rotor floor is
  closed for every element class (cycles 40, 41, 89). No further textbook small-class calibration of the guess is
  worth a cycle: the remaining connected-system cost is path length (large-amplitude modes) and the near-
  equilibrium 4-call floor, neither of which a diagonal-guess constant changes.

### Next idea / control
- Champion 8a30dec unchanged (train 0.6740826 / valid 0.6765981); restored by `175c7c2`, implementation preserved in
  `ideas/textbook-small-class-bundle/ee5c5798ee8fabf56a76dcfc31e369775c9d7969/algo.py`.
- Guess-constant calibrations are closed as a family (stiff classes 81, soft classes 39/67/89, rotors 37/40/41/58/
  68/69/89, heavy rows 64/66). Next: the structural cost map — neutral-dimer soft-mode endgame (14.7-call floor,
  one call per rigid mode learned) and large-amplitude rotor paths — needs a mechanism that learns or predicts a
  mode's stiffness without sampling it; the remaining unexplored candidates are the low-priority connected-system
  bond radius 0.3 Å (one parameter, cycle-28 line) and bond-formation-aware internals (backlog, low).

## Cycle 90 — surrogate docking without the start-geometry force call: the docked pose is the optimizer's first evaluated point; the rigid-body force-cosine and energy-decrease gates (one call per docked run) are dropped and the surrogate-only gates decide (keep) (2026-09-18)
Champion at start: `8a30dec9b7cf843c34cc0feefaa961e8a84b33a8` (train 0.6740826 / valid 0.6765981). Candidate
`d86750369416b33503bb6d929ab6ed87592b5ae3` — new champion (train 0.6711216 / valid 0.6708908).

### Hypothesis
Idea search after cycle 89 (saved-data analysis, code and log reading only). Static closures, for the record:
rotor pre-relaxation window widening for mid-phase XH₃ starts (scanning), OH nearest-staggered rule (64 % only),
6-fold aromatic-methyl class too small; light-rotor cost ≈ 280 calls per split but diffuse (no class); charged
same-basin dimer cost equals the neutral one (the charged ratio 0.80 / 0.86 is a fast reference plus credit
walkers); same-basin connected losers ≤ 0.006; no failure tail (every run converges far below 200 calls); the
update-side pair-consistency filter is not a defect; the rotor-turn cost is an absolute excess of ≈ +8–10 calls
for ≥ 60° turners (44 / 65 molecules, contrib 0.067 / 0.100) with the ratio ≈ 0.7 in every hmax bin (no relative
excess); rowansci starts are 0.2–1.7 Å from the xTB minimum (25 molecules, 16.0 vs 23.8 calls); the endgame
accounting "1 (start) + one call per soft direction whose model is > 30 % off + ≈ 2 contraction steps + 1
verification call" fits tafamidis (7) and the near-equilibrium dimers (12); torsion-only cap 0.75 / trust ratio 3
(cycles 29, 32 evidence), partitioned or rigid-body-projected secant equations (polluted by inter-fragment force
changes), geminal 1,3 Urey–Bradley terms (a closed guess-calibration family) and bond-formation-aware internals
(no excess, basin risk) are not worth a cycle. The symmetry-image dummy-azimuth repair (backlog next experiment
(1)) was censused on both splits (`_symmetry_operations` vs `symmetry_maps` at the cycle-88 starts and endpoints):
operations are dropped at dummy atoms on 8 train / 2 valid connected molecules only (135099703, 135091854,
135093107, 135080011, 135093846, 134987421, 135098945, 135041910; 135077553, 135071396), all tiny and already at
5–13 calls against 8–19 — ≤ 0.0005 expected and a 2-molecule valid gate: parked.
- The cycle-70 docking pays one xTB call at the *start* geometry (rigid-body cosine gate against the surrogate,
  cached energy for the decrease check) and one at the docked pose; the docked-pose call is Sella's first point,
  so the start call is one wasted call on every docked run: 93 / 87 docked neutral dimers per split (cycle-70
  proxy), Σ 1/ref 2.54 / 2.58 → −0.0054 / −0.0056 deterministic if the gates are dropped. What the gates bought:
  the energy check refused 4 / 1 starts (cycle 76 read-out), the cosine gate an unknown share of the 26 / 32
  undocked dimers with rmsd₀ ≥ 0.3 Å (the 14 / 11 near-equilibrium ones are move-gate refusals, surrogate-only,
  and stay bit-identical), and cycle 70's own losers (near-equilibrium starts docked ≥ 0.5 Å away) had passed
  both gates. Hypothesis: the gates are not worth their call — docked runs lose exactly one call with identical
  trajectories (the docked-pose evaluation is the same xTB call either way), and the formerly refused starts dock
  like the far/loose class of cycle 70 (net gain) with a few near-equilibrium losers.

### Change (functions)
- `_dock_start(atoms)`: no force call; fragments, surrogate terms (charge parity), `_dock_relax`, the finite /
  ≤ 6 Å / ≥ 0.5 Å rmsd gates unchanged; the docked pose is installed on the Atoms and nothing is cached. The
  start evaluation, `_dock_rigid_field`, `_DOCK_MIN_COS`, the energy comparison and the cache re-installation are
  removed. `minimize_func`: `_dock_start(atoms)` before the calculator is attached; Sella budget
  `steps = max_force_calls − 1` (no call precedes the optimizer's first). Rationale comment updated.
- Static checks: py_compile; pyflakes diff vs the champion empty; the diff is a pure deletion of the two gates
  (25 insertions / 54 deletions, comments included). Read-out script `/tmp/cmp90.py` (classes by the cycle-70
  vs cycle-65 docked proxy; "exactly n−1 with the same endpoint" as the deterministic signature).

### Result
- Train 0.6711216 vs 0.6740826 (Δ −0.0029610), 469/469 converged, mean rel energy 1.0030701 (vs 1.0027410);
  `evaluation_results/cycle-90-train.json` (972397700e7cee0c002a9e6fe856fb68). Valid 0.6708908 vs 0.6765981
  (Δ −0.0057073), 465/465 converged, mean rel energy 1.0085197 (vs 1.0085820);
  `evaluation_results/cycle-90-valid.json` (13f7119bf5d7ad765342b646fb58f5a4). Decision **keep**.
- Controls: connected 257 / 262 and charged multi-fragment 79 / 73 bit-identical on both splits; move-gate
  refusals (undocked in both cycles) 32 / 38 bit-identical.
- Deterministic class: 89 / 87 runs end at the identical geometry with exactly one call less (Σ 1/ref 2.371 /
  2.594 → −0.00506 / −0.00558); this includes amides__amines (train) and alkanes__alkenes (valid), docked under
  the champion but not in cycle 70 (their starts changed with the rotor pre-relaxation), and excludes
  acids__esters (train, docked in cycle 70, refused by the move gate now).
- Cycle-70 energy-refused starts, now docked: train 4, all worse — amides__ketones 14 → 15, alcohols__amides
  18 → 21, amides__ethers 24 → 31, acids__pyrrole 13 → 23 (rmsd₀ 0.18 Å, docked away from a near-equilibrium
  start), Σ +21 calls, +0.593 rel (+0.00126), all same basin; valid 1, better — pyrrole__sulfides 21 → 16.
- Cosine-refused starts, now docked: train 7, 3 : 4 — acids__water 32 → 21 (rmsd₀ 1.42, basin 0.88 Å, r 0.994 →
  1.000), esters__water 22 → 17 (basin 0.54), amides__water 33 → 27 (ref 106; basin 0.99, r +0.029);
  ethers__sulfides 10 → 12, sulfides__water 17 → 26 (basin 1.68 Å, r +0.119), acids__sulfides 8 → 10
  (rmsd₀ 0.07), esters__sulfides 15 → 26 (rmsd₀ 0.47, same basin); Σ +0.366 rel (+0.00078). Valid 4, 3 : 1 —
  esters__esters 43 → 29 (rmsd₀ 2.50, basin 1.11, r 1.162 unchanged), ketones__pyrrole 28 → 24 (basin 1.61,
  r −0.029), esters__thiols 41 → 39; pyridine__pyridine 14 → 22 (rmsd₀ 0.31, same basin); Σ +0.036 (+0.00008).
- Energy gate: train Σ(r−1) 1.286 → 1.440 (+0.651 without monoatomics__pyrrole); valid 3.991 → 3.962 (+0.746
  without carboxylates__ketones). Largest credits unchanged (all charged, untouched).

### Interpretation
- The deterministic part is exactly as computed (−0.0051 / −0.0056: one call per docked run, identical
  trajectories — the docked-pose xTB call is the same call whether or not a start call precedes it), and it is the
  whole valid gain; on train the formerly gated starts cost back +0.0020: the four energy-refused dockings are all
  losers (+21 calls, same basins — the surrogate pose was higher in xTB energy than the start and the run walks
  back; acids__pyrrole is the near-equilibrium adverse class of cycle 70) and the cosine-refused set splits into
  far water complexes that gain (−5…−11 calls, three basin moves with credits +0.15) and sulfide pairs that lose
  (+2…+11 calls; the UFF S 12-6 wall places S contacts wrongly — esters__sulfides docks a 0.47 Å start into a
  same-basin 26-call walk). The cosine gate refused only 7 / 4 starts, i.e. it was cheap insurance that cost
  93 / 87 calls per split.
- Net: train −0.0030, valid −0.0057; the class read-out has no lottery component beyond the eleven / five re-rolled
  runs (five basin changes, all credits ≥ 0 except ketones__pyrrole −0.029). Both margins stay above +0.65 without
  their largest charged credit.
- The energy-refused class (docked pose above the start in xTB energy) is now the one identifiable loser class:
  4 / 1 runs, +21 / −5 calls. A surrogate-only criterion for it does not exist in the cycle-70 data (the losers'
  surrogate binding energies and moves are inside the winners' ranges — same conclusion as the cycle-72 static
  closure of the contact-aware acceptance); the sulfide losers point at the S 12-6 parameters (UFF x_S 4.035 Å,
  D 0.274 kcal/mol — the largest well among the organics, so S···H/C contacts are over-attracted and mis-placed),
  a physics fix of the surrogate, not a gate.

### Next idea / control
- Champion `d867503` (train 0.6711216 / valid 0.6708908). Docking follow-up worth one cycle: none with a start
  criterion (closed above); the sulfide/thiol surrogate class (S 12-6 well) is a small class (train 4 sulfide
  losers, valid thiol pairs mostly winners) — record, do not tune. The docking now costs nothing when refused and
  nothing extra when accepted, so the remaining docking cost is the near-equilibrium adverse class (≈ +0.004 per
  split, cycle 70) which has no start-side criterion.
- Next: the structural cost map is unchanged (neutral-dimer soft-mode endgame floor 14.7 calls, large-amplitude
  rotor paths); the symmetry-image dummy repair is parked (8 / 2 molecules); remaining low-priority candidates:
  connected-system bond radius 0.3 Å (one parameter), bond-formation-aware internals.

## Cycle 91 — class-resolved torsional guess applied to the fragments of multi-fragment systems as well (the cycle-37 rotatable-bond classes for acyclic bonds to a planar centre, same constants; the connected-only gate on `scale_torsions` removed) (discard) (2026-09-18)
Champion at start: `d86750369416b33503bb6d929ab6ed87592b5ae3` (train 0.6711216 / valid 0.6708908).

### Hypothesis
Idea search after cycle 90 (saved-data analysis only, cycle-90 endpoints; scripts `/tmp/xh3_91.py`, `/tmp/xh3b_91.py`,
`/tmp/sp3rot91.py`, `/tmp/small91.py`, `/tmp/tscale91.py`). Static closures first:
- An "every XH₃ rotor at its class minimum phase" start rule has no population: the dimer templates already start at the
  class minimum (neutral dimers 385 / 422 XH₃ rotors within 10° of it, charged 183 / 184); the connected starts are off
  by 10–50° in 124 / 288 rotors, but the same-basin call count does not depend on it at all (largest start deviation
  0–10° 18.2 calls at ratio 0.729, 10–20° 17.2 / 0.719, 20–30° 18.5 / 0.720, 30–40° 17.1 / 0.763, 40–50° 17.1 / 0.742;
  a rotor turn of a connected molecule is handled inside the ordinary steps).
- The dimer rotor turns of the cycle-90 census are endpoint effects: 74 neutral / 20 charged XH₃ rotors start at the
  class minimum and end 10–55° away, and the reference ends at the same phase (|ours − ref| < 10° for 59 / 74) — the
  partner's field sets the phase (esters__ethers O–CH₃ 55°, amides__sulfides acetyl 46°, amines__ketones 28°), so no
  start rule can anticipate it and the torsional docking of cycle 83 (UFF torsions ranked the xTB rotor direction at a
  coin toss) already refuted a surrogate for it.
- Substituted sp³–sp³ rotors near an eclipsed saddle in connected starts: 3 of 270 within 10° (2 molecules) — no
  population for a saddle rule beyond XH₃.
- Near-equilibrium connected molecules (rmsd₀ < 0.15, n 46): 7.67 calls vs 10.41 reference, 3–5-atom molecules at 4–6
  calls (S₄ 4, CNS 4, BF₄ 5, N₃ 5) — within 1–2 calls of the start + Newton + confirmation floor.
- Found instead: the class-resolved torsion scale of cycle 37 (`_torsion_class_factor`: sp²–sp³ ×0.6, sp²–lone pair
  ×0.45, carbonyl–lone pair ×0.7, sp²–sp² ×0.65, fading to 1 with the bond order) is gated to connected systems
  (`scale_torsions = connected`), so the fragments of every dimer keep the unscaled Fischer–Almlöf constant: the acetyl
  methyls of the ketone / ester / amide / acid fragments have Σh 0.025 Ha/rad² against 0.006–0.008 physical (acetone
  V₃ 0.78, acetaldehyde 1.16 kcal/mol; 3–4.5× over), i.e. the rotors that the partner's field moves 10–50° relax in
  steps 3–4× too short, hidden behind the intermolecular motion (the cycle-72 "creep along model-invisible modes").
  Census: 126 train / 120 valid multi-fragment systems have a scaled acyclic bond (48 / 48 charged); 252 sp²–sp³
  bonds, 117 of them turn ≥ 10° (71 ≥ 20°), against 5 / 142 sp²–lp and 5 / 125 carbonyl–lp; same-basin dimers by
  the largest scaled-bond turn: neutral 0–5° 14.5 calls, 10–20° 17.0, 20–40° 18.8, ≥ 40° 23.9 (rmsd₀ 1.08);
  charged 0–5° 14.5, 10–20° 26.0, 20–40° 24.1 — ≈ 260 excess calls over both splits (ceiling ≈ −0.007 per split,
  expected −0.001…−0.003). Touched Σ(r−1): train +0.163 of +1.458 (carboxylates__ketones +0.199, ketones__monoatomics
  +0.233, alkanes__carboxylates −0.327 …), valid +2.889 of +3.898 (carboxylates__ketones +3.216, acids__acids −0.513,
  ketones__water +0.483 …); the untouched part carries +1.295 / +1.009, so the energy gates survive even a total loss
  of the touched credits. Controls: 86 / 83 multi-fragment systems without a scaled bond and all connected molecules
  bit-identical. Mechanism-based scope: the class scale is an intramolecular rotor property, so it applies to every
  fragment regardless of charge (the neutral partner of an ionic complex carries the same rotors).

### Change (functions)
`Internals.guess_hessian`: `scale_torsions = True` (was `connected`), comments updated; `_torsion_centre_types`,
`_torsion_class_factor`, the linear-bend and ion-dipole routing untouched. Static check (`/tmp/gh91.py`, pure
construction of `Internals` + `guess_hessian()` at saved dimer starts): only proper-dihedral diagonals change
(ketones__water 12 entries ×0.64–0.93, amides__sulfides 10 ×0.60–0.82, benzene__water 6 ×0.60 — a toluene methyl,
carboxylates__ketones 6 ×0.60).

### Result
Candidate `20ba76e90db5bed0446e3399fe4869194cde6154`: train 0.6715393 (Δ +0.0004177 vs 0.6711216; mean rel energy
1.0035643 vs 1.0030701, Σ(r−1) +0.232), 469/469 converged — **discard** (valid not run). Controls: 86 untouched
multi-fragment systems bit-identical; connected 254 / 257 (ketones__phenol 23 → 25, guanidiniums__water 14 → 14 with
a 0.004 Å endpoint shift, 104079126 51 → 49 — starts connected by the 1.25 r_cov rule that rebuild as fragments
mid-run, so the fragment scale reached them). Touched: neutral 78 (−32 calls, −0.00105), charged 48 (+40 calls,
+0.00147). Same-basin read-out by the largest scaled-bond turn (cycle-90 run): neutral < 5° +8 calls over 37
(11 better / 8 worse), 5–10° +1 / 9, 10–20° −8 / 13 (5 : 2), 20–40° −12 / 9 (6 : 2), ≥ 40° −21 / 10 (6 : 2);
charged < 5° +15 / 29 (9 : 9), 5–10° +1 / 8, 10–20° −2 / 3, 20–40° −10 / 5 (4 : 1), ≥ 40° −1 / 1 — the turning
population (41 dimers) gained −54 calls (−1.3 per dimer: acids__alkenes 26 → 15, alkanes__esters 19 → 14,
acids__benzene 31 → 24, carboxylates__pyrrole 37 → 31, ketones__pyrrole 26 → 21), the non-turning one (66) lost
+24 (esters__sulfides 26 → 34, guanidiniums__pyrrole 35 → 43, alkanes__carboxylates 14 → 20, ammoniums__esters
48 → 53). Two charged walkers re-rolled into deeper basins at +37 calls: carboxylates__water 13 → 36 (from its
−0.206 basin into the reference's, r 1.000) and amides__carboxylates 15 → 29 (r +0.026). Without the two re-rolls
the candidate would read −0.0009. Candidate preserved in
`ideas/fragment-torsion-classes/20ba76e90db5bed0446e3399fe4869194cde6154/algo.py`; branch reset to `d867503`
(`50a90a2`).

### Interpretation
The mechanism is real but small: where a scaled rotor has to turn the softer model saves 1–2 calls (monotone in the
turn, 17 : 6 better : worse over the ≥ 10° bins on both charge classes), where it does not turn the change is a mild
path perturbation with a positive drift (+0.36 calls per dimer, 20 : 17), and the same-basin net (−30 calls,
−0.0009) is below the walker lottery of a 126-dimer path change (two charged re-rolls, +37 calls, credits +0.232).
This is the cycle-37 picture again (sp²–sp³ single-class molecules were neutral there because their acetyl methyls
do not turn in connected starts): a rotor-stiffness prior pays only along rotors that actually relax, and the dimer
population that does is 41 runs per split. No mechanism-based repair exists — restricting the scale to the carbonyl
acetyls keeps the carboxylate walkers in the touched set (a carboxylate carbon is a carbonyl by the class rule), a
charge-based exclusion is credit protection, and a rotor-selective scale needs the endpoint phase, which nothing at
the start predicts (closures above).

### Next idea / control
Torsion classes for dimer fragments: parked as a ≈ −0.001 same-basin mechanism inside the walker lottery (revisit
only bundled with another dimer-fragment change of the same sign, never alone). Rotor start rules closed for all
populations examined (XH₃ class minimum, substituted sp³–sp³ saddles, environment-set dimer phases). Cost map
unchanged: neutral-dimer soft-mode endgame floor 14.7 calls, large-amplitude rotor paths, charged credit walkers;
remaining low-priority candidates: connected-system bond radius 0.3 Å (one parameter), bond-formation-aware
internals, symmetry-image dummy repair (parked, 8 / 2 molecules).

## Cycle 92 — planar-centre out-of-plane coordinates for the fragments of multi-fragment systems as well (the cycle-80 improper rule — carbonyl, carboxyl, carboxylate, thiocarbonyl, nitro and boron centres whose proper dihedrals run about single bonds, Schlegel's 0.045 Ha/rad² — no longer gated on connected systems) (keep) (2026-09-18)
Champion at start: `d86750369416b33503bb6d929ab6ed87592b5ae3` (train 0.6711216 / valid 0.6708908).

### Hypothesis
Idea search after cycle 91 (all static, saved cycle-90 results and start geometries: `/tmp/costmap92.py`,
`/tmp/imp92.py`, `/tmp/imp92b.py`, `/tmp/imp92c.py`, `/tmp/imp92d.py`, `/tmp/gh92.py`; no calculator, no optimizer
step). Cost map: connected 0.410 of the score (same-basin 14.8 vs 20.5 calls), neutral dimers 0.125 (16.2 vs 42.7,
rel 0.446), charged dimers 0.135 (17.6 vs 27.8 same-basin, the walkers 1.33 / 1.70 carry the credits). The cycle-80
improper rule (`_add_planar_centre_impropers`) returned immediately for multi-fragment systems, so the acetic acid,
methyl acetate / methyl formate, acetone and acetate fragments of the DES370K pairs kept the model wag that the
cycle-80 frequency check had shown 2–5× too soft (acetone 217 vs 484 cm⁻¹, acetic acid 245 vs 535, methyl acetate
342 vs 555): the pyramidalisation of a carbonyl carbon is modelled only by the rotor dihedrals about its single
bonds. Census on the start geometries (same rule replicated on the fragment graphs): 41 / 40 neutral and 23 / 27
charged dimers per split carry 47 + 25 / 45 + 29 such centres (all C=O except 3 + 2 borderline ring carbons of
stretched DES370K benzene / phenol / pyrrole monomers); those dimers cost 20.1 / 20.4 calls same-basin against
14.3 / 15.1 for the neutral dimers without such a centre (references 40 / 41 vs 44 / 46; charged 20.8 vs 16.5 /
20.8 vs 17.0), and a regression over the 241 neutral same-basin dimers charges the carbonyl fragment +3.1 (se 1.1)
calls on top of the acetyl-methyl turn (+4.7 for turns ≥ 10°, se 1.2), the heavy-atom intramolecular relaxation
(+25 calls/Å) and the rigid displacement rmsd₀ (+1.2 calls/Å). The endpoints are planar to 0.3–0.6° (0.01 Å at
the oxygen): the partner's pull displaces the wag by a few hundredths of an ångström in the first steps, the
3–5× too soft model overshoots along it, and the mode has to be learned back before the displacement criteria
along it (1.8e-3 Å) are met — one to two calls per centre, the same mechanism as cycle 80 (−0.7…−1.0 calls per
centre there, "spread over walk lengths, near −16 (8 : 5 : 1)"). It also explains the cycle-91 read-out: the
class-scaled rotor dihedrals soften the fragments' only wag stiffness further (0.6–0.7×), and the non-turning
carbonyl fragments drifted (+24 calls) while the turning ones paid. Static check (`/tmp/gh92.py`, `/tmp/gh92b.py`:
`Internals(atoms, allow_fragments=True)` + guess_hessian, no calculator): the fragments get their impropers at
0.045 Ha/rad² (plus the projected contact term where the carbonyl oxygen is the acceptor), the Jacobian rank is
unchanged, everything else bit-identical. Expected ≈ −1 call per centre → −0.002 … −0.003 per split, with the
charged walker lottery (the touched charged dimers hold Σ(r−1) −0.040 train / +3.501 valid, i.e.
carboxylates__ketones +3.216; validity survives even a total loss of the touched credits: +0.46 without them).

### Change (functions)
`Internals._add_planar_centre_impropers`: the early return for systems with fragment translations / rotations is
removed (docstring updated). `guess_hessian` already routes an improper whose centre has proper dihedrals to
`_h0_out_of_plane()` (0.045 Ha/rad²) irrespective of the fragment count, and the transport / shadow-copy machinery
treats the coordinate like any dihedral. Constants unchanged (bo 1.7, 345°, 0.045 Ha/rad²; fixed a priori).

### Result
Candidate `78ac80e6d58c13bc9f4bb1e9c2a77cae68e9f5ca`: train **0.6670406** (−0.00408, mean rel_energy 1.0032037,
Σ(r−1) +1.503 (+0.714 without monoatomics__pyrrole), 469 / 469 converged, 57 s), valid **0.6648582** (−0.00603, mean
rel_energy 1.0085000, Σ(r−1) +3.952 (+0.736 without carboxylates__ketones), 465 / 465): **keep** — new champion.
- Multi-fragment systems without such a centre: 148 / 148 and 136 / 136 bit-identical. Connected: 255 / 257 and
  259 / 262 identical; the movers are the dimers that Sella joins into one fragment at the start and rebuilds as
  fragments mid-run (train ketones__phenol 23 → 28 (endpoint shifted 1.26 Å, same energy), 104079126 51 → 51
  (1.2 Å, same energy); valid esters__phenol 16 → 19, ammoniums__ketones 24 → 22, carboxylates__water 24 → 20, all
  same energy) — the rebuild lottery, Σ +5 / −3 calls.
- Neutral touched dimers (41 / 40): train −22 calls (31 : 6), valid −74 (28 : 7); same-basin 20.10 → 18.93 (ref
  40.2) and 19.46 → 17.87 (ref 44.3) = −1.04 / −1.41 calls per centre. The gain does not depend on the acetyl turn:
  turn < 5° −19 (12 : 3) / −24 (11 : 2), 10–40° −8 (8 : 2) / −15 (9 : 4), ≥ 40° −14 (8 : 0) / −23 (7 : 1); two
  centres −8 (4 : 0) / −21 (5 : 0). By fragment class (same basin): esters −35 (17 : 0) / −22 (13 : 4), acids −24
  (15 : 4) / −32 (14 : 3), ketones −18 (10 : 3) / −8 (8 : 6). Basin changes: train acids__ketones 11 → 36 into a
  basin 0.06 lower in r (r 0.937 → 1.000, +0.063 credit — the one +25-call draw of the split); valid alkenes__ketones
  47 → 35 (r 1.000 → 0.998, −0.002).
- Charged touched dimers (23 / 27): train −26 calls (13 : 5, no basin change; same-basin 19.9 → 18.8, −1.04 per
  centre), valid −53 (17 : 8; 21.0 → 19.9, −1.15 per centre; carboxylates −30 (11 : 3)); carboxylates__esters
  39 → 17 lost its +0.007 credit (r 1.007 → 1.000), carboxylates__ketones 29 → 39 kept r 4.216, carboxylates__thiols
  32 → 38 kept 1.128, alkanes__carboxylates (train) 14 → 20 kept 0.673; valid Σ(r−1) 3.962 → 3.952.
- Distribution of the per-dimer change (train, 64 touched): 44 faster (−1 … −6), 9 unchanged, 11 slower (+1 … +6,
  plus the +25 basin change).

### Interpretation
The carbonyl-fragment excess of the dimer class was a coordinate-set deficiency, not an endgame floor: giving the
fragments the out-of-plane coordinate at the Schlegel value removes ≈ 1.0–1.4 calls per centre on both splits, on
turning and non-turning acetyls alike and in every fragment class (esters cleanest, 30 : 4 over the splits), with
the charged carboxylates paying as well (16 : 7) and the credits intact. The connected-only scoping of the
cycle-80 rule had been credit protection dressed as a control: the fragment paths did change (every touched dimer
moved), but the change is a plain model repair whose basin re-rolls were 1 : 1 per split and cost-neutral in
credit. Lesson for the remaining connected-only intramolecular features: a model repair that is right for
monomers is right for the same molecules as fragments; the cycle-91 torsion-class extension lost on the
non-turning fragments because it softened the wag that this cycle now holds — that bundle condition is met.

### Next idea / control
Cycle 93: the cycle-37 torsion classes for the fragments (the parked cycle-91 change, `scale_torsions = True`) on
top of the fragment impropers — the mechanism paid on the 41 turning dimers (−54 calls) and its −0.6–0.7× softening
of the fragments' wag is now covered by the improper; controls: multi-fragment systems without a scaled bond and
connected molecules bit-identical. Other connected-only intramolecular features not yet applied to fragments:
stretch row factors (64: sulfides / thiols fragments), vicinal 1,4 pairs (65, touches only charged paths), rotor
class factors (40) and the third-row torsion floor (41), the class-resolved linear-bend constant (39) for
linear fragments — each to be censused for a fragment population and a displaced mode before a cycle.

## Cycle 93 — class-resolved torsional guess for the fragments of multi-fragment systems on top of the fragment out-of-plane coordinates (the cycle-37 rotatable-bond classes, same constants; `scale_torsions = True`; the parked cycle-91 change re-run now that the cycle-92 improper holds the fragments' carbonyl wag) (non_generalizable) (2026-09-18)
Champion at start: `78ac80e6d58c13bc9f4bb1e9c2a77cae68e9f5ca` (train 0.6670406 / valid 0.6648582).

### Hypothesis
Cycle 91 (the same change on champion d867503) paid −54 calls on the 41 turning acetyl-type methyls but lost +24 on
the 66 non-turning fragments and +37 on two charged walker re-rolls (train +0.0004, discarded, parked
"bundle-only"). Cycle 92 then showed that the fragments' carbonyl wag had been carried by the rotor dihedrals alone
(no improper for fragments), so the 0.6–0.7 class factors of cycle 91 had softened the wag as well — a mechanism for
the non-turner drift. With the improper in place (0.045 Ha/rad², unscaled: the improper branch of `guess_hessian`
precedes the torsion scaling), the class scaling touches only the proper-dihedral diagonals of the scaled bonds
(static check `/tmp/gh93.py`: coordinate sets and ranks identical to the champion, off-diagonal blocks identical,
impropers unchanged, e.g. acids__esters 16 proper diagonals 0.019 → 0.014 / 0.0048 → 0.0029 Ha/rad²). Expected: the
turning-dimer gain of cycle 91 (≈ −0.001 … −0.002 per split) without the non-turner drift; risk: the basin lottery
on the 126 / 120 touched paths (48 / 48 charged).

### Change (functions)
`Internals.guess_hessian`: `scale_torsions = connected` → `scale_torsions = True` (comments updated); identical to
the cycle-91 diff (`ideas/fragment-torsion-classes/20ba76e…`) applied on top of the cycle-92 champion. Preserved in
`ideas/fragment-torsion-classes/3a0cbb6158f66aa87054a3754edb0f6740ef3e9d/algo.py`.

### Result
Candidate `3a0cbb6158f66aa87054a3754edb0f6740ef3e9d`: train **0.6653582** (−0.00168, mean rel_energy 1.0031265,
Σ(r−1) +1.466, 469 / 469, 57 s) — passed; valid **0.6668016** (+0.00194, mean rel_energy 1.0083076, Σ(r−1) +3.863,
+0.647 without carboxylates__ketones, 465 / 465): **non_generalizable** — champion restored (`bc57bca`).
- Controls: multi-fragment systems without a scaled bond 86 / 86 and 83 / 83 bit-identical; connected 254 / 257 and
  259 / 262 (the rebuild-joined dimers again, −3 / −1 calls).
- Same-basin touched dimers: train n 124, −11 calls (43 : 31), valid n 114, −31 (33 : 21) — ≈ −0.0015 / −0.0019 of
  the score. By turn (cycle-92 endpoints): train turn 10–20° −20 (7 : 1 neutral), non-turners < 5° +7 (neutral −5
  at 8 : 8, charged +12 at 7 : 7); valid 5–40° −40, non-turners +7 (neutral +5 at 3 : 7), ≥ 40° +2.
- Versus cycle 91 on the same train dimers (`/tmp/cmp93b.py`): neutral non-turners d91 +8 (11 : 8) → d93 −5
  (8 : 8) — the drift is gone as hypothesised; but the turners' gain shrank (≥ 20°: d91 −33 → d93 −8) because the
  impropers had already taken −22 calls there; charged non-turners +15 → +12 with corr(d91, d93) 0.67
  (guanidiniums__pyrrole 35 → 43 and benzene__imidazolium 36 → 41 in both cycles: deterministic far-walk path
  effects, refs 65 / 82).
- By class composition (`/tmp/cmp93d.py`): sp²–sp³-only dimers train −27 (38, all same basin) / valid −6 same-basin
  (33) plus benzene__benzene 16 → 32; mixed (sp²–sp³ + carbonyl–lp / π–lp) train +20 (45) / valid −25 (43);
  no-sp²–sp³ −4 / 0 — the class attribution flips between the splits.
- Basin changes: train 2, score −0.00002 (acids__ketones 36 → 9 back into its r 0.937 basin, −0.063;
  amides__carboxylates 15 → 29 into a basin +0.026); valid 6, score **+0.00389** (amides__pyridine 29 → 20 into a
  higher basin −0.102, benzene__esters 26 → 18 −0.021; amides__carboxylates 22 → 33, carboxylates__esters 17 → 33
  +0.008, benzene__benzene 16 → 32 +0.010, ammoniums__esters 15 → 39 +0.016) — the valid failure is the lottery, the
  same-basin part of valid was better than train's.
- Recurrent adverse sub-class: the carbonyl–lp factor (esters 0.7, amides 0.8–0.87 after the bond-order fade) on
  non-turning fragments: amides__water 27 → 31 and amides__phenol 27 → 31 in both cycles 91 and 93, valid
  carbonyl–lp-only non-turners +7 (1 : 4) — the same adverse class as in cycle 37's regression (+0.33 / +0.30 calls
  per bond).

### Interpretation
Three evaluations (91 train; 93 train and valid) agree that the fragment torsion classes are a small same-basin
mechanism (−30 / −11 / −31 calls over 107 / 124 / 114 same-basin touched runs ≈ −0.001 … −0.002 per split) whose
class attribution is not stable (the sp²–sp³-only population paid −27 on train and −6 on valid; the mixed population
+20 / −25), sitting inside a basin lottery of ±0.004 on ≈ 125 touched paths per split (realised: +37, −13, +50
calls). The bundle hypothesis was right about the non-turner drift (gone) and wrong about the size: the impropers
took most of what the softer rotors used to win on the turning fragments — the two changes were substitutes for the
same first-step overshoot, not additive. A sub-class variant (sp²–sp³ only) inherits −0.0015 / +0.0007 from the
two split evaluations, i.e. expected ≈ 0 inside the lottery; the carbonyl–lp factor is adverse on fragments as it
was on monomers. **Fragment torsion classes closed in every variant**; the cycle-37 carbonyl–lp factor for connected
systems (0.7 → 1, adverse +0.3 calls per bond in cycle 37) remains a possible small connected-only refinement that
would have to be bundled.

### Next idea / control
Remaining connected-only intramolecular features to census for a displaced fragment mode before any cycle: stretch
row factors (64) for the C–S bonds of sulfide / thiol fragments (a stiff mode, the partner's field moves it
< 0.01 Å — likely ≈ 0), the class-resolved linear-bend constant (39) for linear fragments (none in DES370K), vicinal
1,4 pairs (65: only charged paths). Otherwise the cost map is unchanged: connected 0.41 (same-basin 14.8 vs 20.5
calls), neutral dimers 0.125 (rel 0.44), charged dimers 0.135 (rel 0.80, credits); the next idea search should
target the connected endgame again (soft modes with a real gradient) or the neutral dimer floor
(14.7 + 17·intra calls).

## Cycle 94 — rank-1 deflation of the pseudorotation (phase) mode of puckered five-membered rings in the guess Hessian (tangential pattern of the second-harmonic fit of the five ring dihedrals over all proper dihedrals of the ring bonds, curvature scaled to f = 0.2; rings with a conjugated, double or aromatic ring bond, bridged rings and near-planar rings excluded; connected systems only) (keep) (2026-09-18)
Champion at start: `78ac80e6d58c13bc9f4bb1e9c2a77cae68e9f5ca` (train 0.6670406 / valid 0.6648582).

### Hypothesis
The torsional guess treats every ring bond as an ethane-like rotor (0.026 Ha/rad² per bond after the 1/√n
sharing, unscaled in rings ≤ 8 since cycle 37). Model-Hessian frequencies at saved geometries (`/tmp/ringdec94.py`,
`/tmp/ringw94.py`: the guess projected with P = QQᵀ, Eckart/mass-weighted, no calculator) show that this is right
for the radial (amplitude) puckering of five-membered rings — cyclopentane 276 cm⁻¹ against the observed 283, THF
300, pyrrolidine 299 — and for six-membered chairs (cyclohexane 242/243/385/421 vs 248/248/383/426), but that the
pseudorotation (the phase motion in which the five ring dihedrals φ_j = Φ cos(P + 4πj/5), Φ ≈ 40–50°, change with
the tangential pattern dφ_j/dP) sits at 222–229 cm⁻¹ in the model, k_P = (5/2)KΦ² ≈ 0.035 Ha/rad², where the
physical curvature 2V₂ is 0 (cyclopentane, free), 0.001–0.002 (THF, pyrrolidine: 0.1–0.3 kcal/mol twist–envelope
barriers) and ≤ 0.01 Ha/rad² (1–3 kcal/mol in substituted, heteroatom and fused rings) — a mode 3–30× too stiff,
which a quasi-Newton walker creeps along (per-step error |1 − 1/ρ|). Both low modes are 95–99 % ring-dihedral in
origin, so a per-bond softening cannot separate them: uniform factors 0.25 on the ring torsions put the radial
mode 3.5× too soft (a dead end; erring soft is worse), the per-bond weights (φ/40°)² or −cos 3φ also soften the
radial mode (cyclopentane 157/263, 139/230). The rank-1 deflation H ← H − (1 − f)(Hu)(Hu)ᵀ/(uᵀHu) with u the
tangential pattern over all proper dihedrals about the ring bonds scales the pseudorotation alone (f = 0.2 →
0.30× in the projected Cartesian mode: cyclopentane 229 → 125 cm⁻¹ with 276 unchanged, THF 222 → 118 / 300,
pyrrolidine 229 → 125 / 299, cyclopentanol 149 → 85 / 253, thiolane 157 → 101 / 293, cyclohexane bit-identical),
keeps every H-conjugate direction and positive semi-definiteness, and f = 0.2 puts the model at the stiff end of
the physical range (k_P ≈ 0.01 Ha/rad², textbook-derived, not scanned). Census (`/tmp/ring94*.py`, cycle-92
results): connected molecules with a saturated 5-ring 21 / 29, with a 1–2-π-centre non-aromatic 5-ring 22 / 33,
relative cost no different from the rest (0.75 / 0.72 — the reference shares the defect, 3× stiffer without the
1/√n sharing); a pucker-displacement regression over 238 / 240 same-basin connected neutral molecules charges the
phase change of a 5-ring 15.9 (±13.1) / 46.4 (±15.6) calls per Å of ring-atom displacement (reference 26.2 / 95.8),
the dp ≥ 0.1 Å bin (10 / 7 molecules) paying 19.8 / 26.4 calls against 18.3 / 20.2 predicted — the mechanism of the
premium, attainable ≈ −0.001…−0.003 per split. Scope rule (composition census `/tmp/ring94d.py`): 5-ring, no π
centre or one π centre whose ring neighbours are both σ (cyclopentanones, methylene rings; lactams, lactones,
enamines, aromatic-fused rings excluded — their ring is planar at the conjugated bond and its soft mode is the
radial flap), every ring bond bo < 1.8 (aromatic ≈ 2.0, amide ≈ 1.9, C=C 2.6), not bridged (≥ 3 atoms shared with
another ring ≤ 8: norbornane/tropane-type cages are stiff), second-harmonic amplitude ≥ 25° (near-planar rings have
an ill-defined phase); 19 / 20 molecules per split qualify, of which 7 / 7 pseudorotate ≥ 24° during the cycle-92
run. Risk: the pattern is static (start geometry), so after a phase change ΔP the softening leaks sin²ΔP into the
radial direction (135264879 turns 108°, 135099868 76°, 160853090 71°).

### Change (functions)
`Internals._signed_dihedral` (new, static), `Internals._small_rings` (new, static: smallest ring ≤ ring_max through
every bond, ordered atom lists), `Internals._h0_ring_pseudorotation` (new: scope rule, second-harmonic fit
A = 0.4 Σ φ_j cos(4πj/5), B = 0.4 Σ φ_j sin(4πj/5), tangential pattern t_j = A sin(4πj/5) − B cos(4πj/5) placed on
every proper dihedral about ring bond j, sequential rank-1 deflation per ring; f_pseudo 0.2, amp_min 25°, bo_max
1.8, ring_max 8), `Internals.guess_hessian` (records `dih_index` — the h0 positions of the proper dihedrals of every
central bond — and applies the deflation to the diagonal guess for connected systems before the non-local contact
term is added, so the term is part of the static model, not of the tracked T(x)). Static check `/tmp/check94.py`:
238 / 244 connected molecules bit-identical, 19 / 18 changed (the census list minus two bridged rings the
in-code ring finder catches: 134981761 cage, 104006752 tropane-type), min eigenvalue of the projected guess
−3e-14, monomer frequencies as above.

### Result
Candidate `e8fd4b0ab1b834b07e505b8c04393665d7ce7b95`: train **0.6654223** (−0.0016183, mean rel_energy 1.0032024,
Σ(r−1) +1.502, +0.714 without monoatomics__pyrrole, 469 / 469, 56 s; evaluation `0f4488cefd8e44e093c8d1ddee6bb1ee`)
— passed; valid **0.6641096** (−0.0007487, mean rel_energy 1.0085000, Σ(r−1) +3.952, +0.736 without
carboxylates__ketones, 465 / 465, 36 s; evaluation `61eb689e71f74653addfe60e3fbe1221`) — passed: **keep**, new
champion `e8fd4b0` (train 0.6654223 / valid 0.6641096).
- Exactly the 19 / 18 touched molecules changed (450 / 447 bit-identical, including every multi-fragment system);
  no basin change, every touched run ends at the same pucker phase as before (Δφ identical to 0.1°), rel_energy
  sums −0.0006 / −0.0000.
- Train Σ −29 calls (9 : 2 : 8): acalabrutinib 25 → 11 (ref 38; the creeping run of the census), 160853090 64 → 56
  (ref 89, ring phase change 71°), 378166034 9 → 7, 135264879 24 → 22 (108°), 135099868 16 → 15 (76°), ribociclib,
  empagliflozin, 135264107, 135224095 −1 each; 135099703 13 → 14 and 135336542 11 → 12.
- Valid Σ −8 calls (6 : 3 : 9): 135052949 31 → 27 (ref 29, r 1.069 → 0.931), 135241634 24 → 21 (52°), 135229684
  14 → 12, 135084788 27 → 25, 135278702 and 242053505 −1; 381615095 19 → 22 (a second, non-eligible ring of small
  amplitude flips its phase 176° — a path re-roll inside the same basin), 378161490 30 → 31, 135150142 31 → 32 (46°).
- The molecules with the largest pucker-phase changes gained (108° −2, 76° −1, 71° −8, 52° −3, 48/72° 0, 46° +1): the
  static-pattern leak into the radial direction did not cost anything — the pairs learn the sampled direction
  (cycle-46 lesson), so a geometry-tracked pattern has an expected value ≈ 0 and is not to be run.

### Interpretation
The pseudorotation of puckered five-membered rings is a genuine over-stiff soft mode of the Fischer–Almlöf ring
model (the first ring-specific defect found: the ring exclusion of cycle 37 was justified by aromatic-ring
evidence only), and a rank-1 correction that leaves the accurate radial mode alone pays on both splits with every
control bit-identical — −1.5 / −0.4 calls per touched molecule, the largest gains on runs that were creeping
(acalabrutinib −14) or pseudorotating far (160853090 −8). The size is what the census predicted (−0.0016 / −0.0007
against −0.001…−0.003), so the remaining ring cost is the radial flap of near-planar partially unsaturated rings
(double-well puckering, cyclopentene-type: a positive-definite model cannot describe it) and it is not a lever.
Six-membered chairs are accurate; seven-membered and larger rings have no population worth a cycle (census).

### Next idea / control
Extension to the 5-rings of dimer fragments (multi-fragment saturated-5-ring census 12 / 3 molecules per split,
cycle-92 precedent for extending a connected-only model repair) is worth ≈ 0.0005 on train and nothing on valid —
only as a bundle partner with another same-sign fragment change. Ring line otherwise closed: uniform or per-bond
ring-torsion softening (radial mode), tracked pattern (≈ 0), 6-ring chairs (accurate), near-planar 'part' rings
(double well); saturated 7- and 8-membered rings: 1 + 1 / 1 + 0 connected molecules. Cost map after cycle 94
(`/tmp/cost94.py`): connected 257 / 262 molecules share 0.409 / 0.407, rel 0.746 / 0.722; neutral dimers 133 / 130,
share 0.124 / 0.124, rel 0.437 / 0.445; charged dimers 79 / 73, share 0.133 / 0.133, rel 0.786 / 0.849; next idea search targets the connected endgame (soft modes with a real
gradient that the four secant pairs cannot cover) and the neutral-dimer floor.

## Cycle 95 — ring pseudorotation deflation extended to the puckered five-membered rings of the fragments of multi-fragment systems (same scope rule, f = 0.2; centre labels computed for every system, torsion classes still connected-only) (keep) (2026-09-19)
Champion at start: `e8fd4b0ab1b834b07e505b8c04393665d7ce7b95` (train 0.6654223 / valid 0.6641096).

### Hypothesis
The cycle-94 deflation of the pseudorotation mode of puckered five-membered rings was scoped to connected
systems; the rings of dimer fragments kept the full ethane-like ring-torsion guess (pseudorotation 3–30× too
stiff). The defect is a property of the ring, not of the connectivity of the system, and cycle 92 established
the precedent for extending a connected-only model repair to the fragments of a complex (neutral and charged
alike) when the same defect shows in the fragment census. Census at the saved start geometries
(`/tmp/check95.py`, the candidate's own scope rule): 12 / 3 multi-fragment systems carry an eligible ring
(tetrahydrothiophene, tetrahydrofuran, cyclopentane, pyrrolidine, cyclopentanol fragments; amplitudes 39–53°),
of which 7 / 1 change their pucker phase by ≥ 34° during the cycle-94 run (`/tmp/phase95.py`: amides__ethers
−83°, alcohols__sulfides −83°, acids__ethers +81°, alcohols__alkanes +75°, alcohols__pyrrole +74°,
alcohols__monoatomics +66°, alcohols__imidazolium +39°; ammoniums__sulfides +34°, imidazolium__sulfides +25°).
Expected −0.0005…−0.001 on train and ≈ 0 on valid (three touched molecules, two charged) — a coin-flip valid gate.
A-priori credit analysis (`/tmp/credit95.py`): the touched runs carry no energy credit (train Σ(r−1) −0.060,
alcohols__sulfides already a 0.937 debit; valid +0.005), so a path change cannot lose credit and the margins
+1.502 / +3.952 survive any plausible debit.

### Change (functions)
`Internals.guess_hessian`: the centre labels `_torsion_centre_types(adj)` are computed for every system (they
were `None` for multi-fragment systems; the torsion classes stay gated by `scale_torsions = connected`), and
`_h0_ring_pseudorotation` is applied to the diagonal guess of every system instead of connected ones only.
`_h0_ring_pseudorotation` itself unchanged (f_pseudo 0.2, amp_min 25°, bo_max 1.8, ring_max 8). Static check
`/tmp/check95.py` (guess Hessians of champion vs candidate at the saved starts): 257 / 262 connected and 200 / 200
other multi-fragment systems bit-identical, 12 / 3 changed (max |ΔH| 2e-3…2e-2, min eigenvalue −2e-14). No
multi-fragment system carries a dummy atom (`/tmp/dummy95.py`), so the linear-bend constant has no fragment
extension to bundle.

### Result
Candidate `3828b19db0592282f6d43d68c7f4056a4be54e95`: train **0.6645181** (−0.0009041, mean rel_energy 1.0032024,
Σ(r−1) +1.502 unchanged, 469 / 469 converged, 56 s; evaluation `2b5d030fd8154e16981f0994fd81ea57`) — passed;
valid **0.6639533** (−0.0001563, mean rel_energy 1.0085000, Σ(r−1) +3.952 unchanged, 465 / 465, 37 s; evaluation
`ac2427c7c7dc480685bc9b2594056c12`) — passed: **keep**, new champion `3828b19` (train 0.6645181 / valid 0.6639533).
- Exactly the 12 / 3 touched molecules changed (457 / 462 bit-identical), every touched run ends in the same
  basin (rmsd ≤ 0.003 Å, |ΔE| < 1e-3 kJ/mol) with the same pucker phase; rel_energy sums unchanged.
- Train Σ −18 calls (7 : 2 : 3): amides__ethers 31 → 24 (ref 36, phase change −83°), alcohols__pyrrole 32 → 26
  (ref 49, +74°), alcohols__sulfides 28 → 24 (ref 44, −83°), alkanes__alkanes 14 → 12 (+14°), alcohols__monoatomics
  18 → 17 (+66°), alkanes__amines 13 → 12 (+19°), alcohols__alkanes 21 → 20 (+75°); sulfides__sulfides,
  amines__ethers, carboxylates__sulfides unchanged in count (phase change ≤ 3°); acids__ethers 24 → 25 (+81°),
  alcohols__imidazolium 25 → 28 (+39°, the run also flips its imidazolium ring 86°).
- Valid Σ −1 call (1 : 2 : 0): ammoniums__sulfides 28 → 25 (ref 24, +34°); alkenes__sulfides 10 → 11 (+5°) and
  imidazolium__sulfides 14 → 15 (+25°) — the two rings that hardly pseudorotate paid the ±1 path noise, and the
  gate cleared by 0.000056.
- Cost per touched molecule −1.5 / −0.3 calls, the same as cycle 94 (−1.5 / −0.4): the gain is again
  concentrated on the runs whose ring changes its pucker phase far (≥ 66°: −7, −6, −4, −1, −1, +1, +3), while
  rings that stay at their start phase are ±0–2.

### Interpretation
The ring model defect and its rank-1 repair carry over to dimer fragments unchanged (the fragments' intramolecular
model is the same Fischer–Almlöf guess, and the pseudorotation of a fragment ring is as free as in the monomer),
which is the third connected-only repair extended to fragments (cycle 92 impropers keep, cycle 93 torsion classes
non-generalisable, cycle 95 ring deflation keep) — the cycle-92 rule holds: extend when the fragment census shows
the same defect and the touched runs carry no credit. The valid result (−0.00016, three molecules) is at the gate
and consistent with the ≈ 0 prediction; the train result −0.0009 is at the top of the −0.0005…−0.001 range. The ring
line is now closed on every population: 5-ring pseudorotation (94/95), radial modes and 6-ring chairs accurate,
7-/8-rings 1 + 1 / 1 + 0 molecules, near-planar 'part' rings double-well.

### Next idea / control
Cost map after cycle 95 (`/tmp/cost94.py` on the cycle-95 results): connected 257 / 262 molecules share 0.409 /
0.407 (rel 0.746 / 0.722), neutral dimers 133 / 130 share 0.123 / 0.124 (rel 0.434 / 0.445), charged dimers 79 / 73
share 0.133 / 0.133 (rel 0.787 / 0.848). Remaining candidates from the cycle-95 idea search (all small): the
symmetry-image dummy-azimuth repair (8 / 2 tiny connected molecules, ≤ 0.0005, parked), the 0.3 Å connected bond
radius (one-parameter trust follow-up, low priority), an endgame-only SR1/PSB attribution switch (cycle 33
near-minimum ×1.12 says ≈ neutral at best; must not be convergence damping). The idea-search closures of this
cycle are recorded in the backlog entry `cycle-95 idea-search closures`.

## Cycle 96 — physical guess for the dummy-atom azimuth dihedrals of near-linear centres (torsion about the axis of the linear unit: Fischer–Almlöf floor 0.0015 Ha/rad² for triple-bond units and units with a terminal far end, the shared Fischer–Almlöf double-bond torsion for two-ended cumulenes, Sella's 0.5 kept for bridging hydrogens; connected systems only) (keep) (2026-09-19)
Champion at start: `3828b19db0592282f6d43d68c7f4056a4be54e95` (train 0.6645181 / valid 0.6639533).

### Hypothesis
Sella's `find_all_angles` adds a dummy atom X to every 2-bonded near-linear centre j (atol 15°) and
`find_all_dihedrals` then generates the terminal-dummy dihedrals X–j–A–S (the azimuth of every substituent S of
the neighbour A about the linear axis) and X–j–j'–X' for adjacent linear centres; all of them carried Sella's
fixed 0.5 Ha/rad² (a double-bond torsion). Physically these coordinates are the torsion about the axis of the
linear unit, which is free for triple-bond units (2-butyne 25 cm⁻¹, tolane V₂ ≈ 0.6 kcal/mol) and for units whose
far end is a terminal atom (nitriles, azides, isocyanates, ketenes: nothing to rotate against); only a two-ended
cumulene twists its π system (allene 865 cm⁻¹, 0.09–0.14 Ha/rad²). Because the differences of the same-side
azimuths are the valence angles at A, the stiff guess also over-stiffened those bends: the reduced model
(`/tmp/dumfreq96.py`, dummies eliminated exactly as the optimizer treats them — constrained dummy coordinates follow
the real atoms, the free azimuth minimises the model) put the CH₃ deformation of propyne/acetonitrile at ≈ 2480 cm⁻¹
(1450 observed, 2.9× in k), the rock at 1190 (1050), the allene torsion at 2054 (865) and the 2-butyne torsion at
≈ 1190 (25). Census (`/tmp/dummy96.py`, `/tmp/dummy96b.py`): 75 of the 500 connected molecules carry dummies and
no multi-fragment system does (connected-only is the whole population, no fragment-extension question); against
the heavy/rmsd0 cost regression the dummy molecules cost +1.03 ± 0.55 calls with the champion (+1.65 ± 0.84 with
the reference), two-sided units +1.79 / +2.48, one-sided +0.60 / +1.18 — expected gain ≈ −0.003 per split.
Safety: the internal gradient of a pure dummy rotation is exactly zero (dummy forces are zero), so a soft azimuth
never drives a step by itself; soft azimuths are damped by the trust-region α; the guess is the Fischer–Almlöf
value the same coordinates would get if they were proper dihedrals of the unit, not a scan.

### Change (functions)
New `Internals._h0_dummy_azimuth(dihedral, nbonds, adj, n_share, bo_triple=4.5, At=0.0015)`: walks the chain of
linear centres from the dummy end of the dihedral to the first non-linear atom on either side (`walk`), collects
the bonds of the unit and their Fischer–Almlöf bond orders; any hydrogen linear centre (the bridging proton of a
short hydrogen bond D–H···A, whose twist is an intermolecular coordinate outside the analysis) keeps Sella's 0.5
Ha/rad²; a unit with a terminal far end or a bond order ≥ bo_triple (C≡C 1.20 Å 5.6, C≡N 5.5, N≡N 4.8, C≡P 4.8;
cumulated C=C 3.1, C=N/C=O 3.9–4.1) gets the additive floor At of the torsional formula (0.0015 Ha/rad²); otherwise
(two-ended cumulene) the Fischer–Almlöf double-bond torsion of j–A (or j–j') shared over the n_share azimuths of
that bond (allene 0.0876 per azimuth). `Internals.guess_hessian`: a pre-pass counts the azimuth dihedrals per bond
(j, A) (`n_azimuth`); the dummy branch of the dihedral loop routes end-dummy dihedrals of connected systems to
`_h0_dummy_azimuth` (inner-dummy linear-bend dihedrals unchanged at `_h0_linear_bend`; multi-fragment systems keep
0.5). Static checks: `ast.parse`; `/tmp/check96.py` — 34 / 38 connected molecules change their guess at the start
geometry, every other molecule bit-identical (the four des370k proton bridges carboxylates__thiols,
imidazolium__pyridine, amides__imidazolium, acids__pyrrole stay at 0.5; the first version's bond-order covalent
test bo ≥ 0.5 was replaced by the hydrogen-centre exclusion because symmetric bridges reach H···A bond orders
0.50–0.64 while the covalent C–P bond of the alkyne 135093104 has 0.42); `/tmp/azcls96.py` per-molecule values
(free 0.0015 (+ contact projection), allenes/ketenimines 0.05–0.17, alkyne dummy–dummy 0.0015); reduced model
frequencies of the candidate (`/tmp/dumfreq96c.py`): propyne CH₃ deformation 1550 (1452 observed), rock 1087
(1053), 2-butyne torsion 65 (25), allene torsion 860 (865).

### Result
Candidate `460cd073c7a8f6614a2657b0b55ba6842e366b2b`: train **0.6595482** (−0.0049699, mean rel_energy 1.0032116,
Σ(r−1) +1.502 → +1.506, 469 / 469 converged, 54 s; evaluation `071cf905177646ee9714c0d583a2e806`) — passed;
valid **0.6555918** (−0.0083615, mean rel_energy 1.0079903, Σ(r−1) +3.952 → +3.716, 465 / 465, 37 s; evaluation
`7056b98dd9674219b5c1e0bfe6038cb3`) — passed: **keep**, new champion `460cd07` (train 0.6595482 / valid 0.6555918).
- 38 / 39 molecules changed (431 / 426 bit-identical), Σ −60 / −94 calls, gains : losses : same 19 : 6 : 13 /
  27 : 5 : 7; 34 / 38 of them carry a dummy at the start, 4 / 1 acquire a near-linear centre at a rebuild during
  the run (252089162 39 → 25, 104129283 38 → 28 among them).
- Per unit kind (`/tmp/gain96.py`): free units (triple bonds, terminal far ends) 32 / 35 molecules Σ −36 / −71
  calls; two-ended cumulenes 2 / 3 molecules Σ −9 / −13, all gains (135114602 19 → 14, 135099703 14 → 10,
  135169446 15 → 9, 135094940 28 → 23, 135234349 18 → 16).
- Largest gains: 252089162 39 → 25, 363892164 35 → 24, 134979308 25 → 15, 135239978 33 → 19, 125084169 31 → 21,
  104073139 29 → 21; largest losses acalabrutinib 11 → 16 (same basin), 135225351 28 → 35 (basin change, ΔE −0.001
  kJ/mol), 135249279 33 → 38, 135264337 11 → 15.
- Basins: 34 / 35 of the changed runs end in the same basin; train 104046121 moves to a lower basin (rel 1.000 →
  1.004), valid 135239978 leaves the lower basin the champion had found (rel 1.250 → 1.000, +54 kJ/mol; the valid
  margin −0.25 → +3.716) and 135060244 moves 0.987 → 1.000.
- Cost regression after the cycle (`/tmp/dummy96b_after.py`): dummy molecules −0.32 ± 0.53 calls against the
  heavy/rmsd0 fit (+1.03 before), two-sided units −1.18 (+1.79 before), one-sided +0.17 (+0.60 before) — the
  dummy penalty is gone.

### Interpretation
The 0.5 Ha/rad² Sella assigns to every dummy dihedral is a placeholder, not a calibration: for the coordinates
that describe the rotation about a triple bond or a one-sided linear unit the model was 300× too stiff, and because
those azimuths also carry the valence angles at the neighbour the neighbour's bends were 2.9× too stiff. The
repair pays −1.6 / −2.4 calls per touched molecule, three times the census estimate (the census measured the
residual of the whole molecule, not the walker steps lost to the mis-modelled bends), and holds on both splits
with the usual ±1–5 call basin/path noise on ≈ 25 % of the touched runs. This is the fourth model-frequency
defect found by comparing reduced model frequencies with observed vibrational data (linear bends 39, ring
pseudorotation 94/95, stretch row factors 64, dummy azimuths 96): the diagnostic — build the model, eliminate the
constrained coordinates, compare frequencies of textbook molecules — is the productive method for the connected
population; the only coordinates left with Sella placeholder constants are the dummy dihedrals of multi-fragment
systems (no such system carries a dummy) and the ionic bends/torsions.

### Next idea / control
Remaining cost map unchanged in structure (connected ≈ 0.40 share on each split, neutral dimers 0.12, charged
0.13). Candidates: (i) the dummy-azimuth guess for multi-fragment systems is moot (census: zero dummies), but a
rebuild during a dimer run can create a near-linear centre — the fragment branch keeps 0.5 and is a pure
bookkeeping extension if a dummy census of the final geometries finds cases; (ii) the same reduced-frequency
diagnostic for other Sella-placeholder coordinates: the improper/out-of-plane constant 0.045 (calibrated in 48),
linear-bend inner dihedrals (39), the angle guess of the dummy angles A–j–X (0.10 Ha) — the propyne/acetonitrile
model still shows the C≡C–H / C–C≡N bends at 947–1003 vs 633–931 observed (1.3–1.5× in frequency, ≈ 2× in k),
worth a census of what the linear-bend constant is for heavy-atom linear units vs H-terminated ones;
(iii) approach-only pre-move for undocked charged far starts (EV ≈ 0.002, lottery), (iv) endgame-only SR1/PSB
(≈ neutral).

## Cycle 97 — single-bond torsional bond order for the rotors of heavy sigma centres (Fischer–Almlöf bond-order factor capped at one for proper dihedrals about bonds with a p-block atom beyond the second row and a centre without a π orbital: phosphate/phosphonate P–O and P–N, silyl ether/siloxane Si–O, sulfonamide S–N, sulfonate S–O rotors; connected systems only) (non_generalizable) (2026-09-19)
Champion at start: `460cd073c7a8f6614a2657b0b55ba6842e366b2b` (train 0.6595482 / valid 0.6555918).

### Hypothesis
Idea search (static, cycle-96 result files): dimer cost is ≈ 14–16 calls for neutral dimers regardless of start
displacement or compression (`/tmp/nearstart97.py`, `/tmp/compressed97.py`), charged dimers cost 11 + 9.4·rmsd0 and
their rel 0.8 is the reference's efficiency, not our absolute cost; slow near-start dimers have heterogeneous causes
(`/tmp/slowdimer97.py`); connected near-start molecules sit at the 6–9 call floor. The largest positive residuals of
the connected cost regression (calls − (7.2 + 0.118·heavy + 11.0·rmsd0), `/tmp/resid97.py`) are P/S/Si/B-rich
molecules with rotors turning 90–177°. A census of the model's rotational stiffness Σh per bond class
(`/tmp/rotorcensus97.py`, sum of the proper-dihedral guess entries about each acyclic bond at the start geometries)
against the literature curvatures (n²V_n/2) found the classes O–P (median Σh 0.036 Ha/rad², physical 0.005–0.014),
O–Si (0.033 vs 0.001–0.005), N–S (0.030 vs 0.006–0.013), O–S (0.034 vs 0.005–0.014) 2–10× over-stiff, while C–S,
C–Si, C–P are at or below physical (0.007–0.014). The mechanism: the Fischer–Almlöf torsion reads the contraction of
a bond relative to the Cordero radius sum as multiplicity, bo = exp(−2.85 (r − r_cov)/Bohr), and the rotation carries
it twice (h0 ∝ bo²); P–O 1.58–1.62 Å against 1.73, Si–O 1.63–1.66 against 1.77, S(VI)–N 1.60–1.65 against 1.76 and
S(VI)–O 1.55–1.60 against 1.71 give bo 1.7–3 (4–9× the single-bond torsion) although the contraction is the polar
(Schomaker–Stevenson) and hypervalent one, not π bonding — a centre without a π orbital cannot form one. Proposed
rule (the stretch row-factor argument of cycle 64 applied to rotations): cap the torsional bond order at 1 for a
proper dihedral whose central bond has a p-block atom beyond the second row (`_STIFFNESS_SCALE < 1`) and a
'sigma'-typed centre (`_torsion_centre_types`); bonds with bo ≤ 1 and every lone-pair heavy centre (thioethers,
disulfides, thioanisoles) unchanged. Affected-set census (`/tmp/heavycap97.py`): 51 / 52 connected molecules carry
such a bond with bo > 1 (O–P 31 / 27 bonds, N–S 20 / 21, O–Si 8 / 12, N–P 8 / 8, O–S 3 / 10, plus the mild
C–S(VI)/C–Si/C–P/S–P/Br–P classes at bo 1.05–1.4); against the cost regression they cost +1.23 / +1.74 calls per
molecule (Σ +63 / +91), the untouched heavy-atom molecules −0.20 / −0.30. A-priori concerns noted: the mild classes
(bo 1.05–1.4, ρ ≈ 1 already) would become 0.5–0.9× physical for no expected gain, and the softening direction was
argued to help both amplitude populations (cycle 37) unlike the stiffening moves of cycles 41/89.

### Change (functions)
`Internals._h0_dihedral(..., bo_max=None)`: optional cap of the bond-order factor before the Fischer–Almlöf formula
(both bo factors capped). New `Internals._single_bond_rotor(b, c, types)`: True when one atom of the bond is a
p-block element beyond row 2 and one centre is 'sigma'. `Internals.guess_hessian`, proper-dihedral branch: for
connected systems the capped bo is also what `_torsion_class_factor` receives (a sulfonamide N(pi)–S(σ) bond then
gets the full s_pi_sigma), `tfac[key] = (fac, bo_max)`. Impropers, dummy azimuths, ionic torsions, multi-fragment
systems untouched. Static checks: `py_compile`; `/tmp/check97.py` (projected guess of champion vs candidate at every
start geometry): 52 / 53 molecules change, 417 / 412 bit-identical — the census set plus des370k_monoatomics__sulfides
(the ion contact makes the sulfide S three-bonded, hence 'sigma', and the compressed C–S start bond has bo 1.37);
`/tmp/diffprim97.py`: the primitive changes are confined to 1351 dihedral entries (median ratio 0.83, q10 0.39, min
0.03 on the S=N bo-6 bonds of two S–N cages) plus the ring-pseudorotation blocks of 7 heavy 5-rings; no negative
guess eigenvalues.

### Result
Candidate `702807879c10e0fa0c8388e8d1dc852e7008ce6d`: train **0.6591308** (−0.0004174, mean rel_energy 1.0030484,
Σ(r−1) +1.506 → +1.430, 469 / 469 converged, 54 s; evaluation `333bcb0727ee408b9998e1de7c4bbe60`) — passed the train
gate; valid **0.6571570** (+0.0015652, mean rel_energy 1.0079904, Σ(r−1) +3.716 → +3.716, 465 / 465, 38 s;
evaluation `6b1280c4d06c4fa6b782bc2221fb363c`) — failed: **non_generalizable**. Champion restored (`d93bf6f`),
implementation preserved in `ideas/heavy-sigma-torsion-cap/702807879c10e0fa0c8388e8d1dc852e7008ce6d/algo.py`.
- 52 / 53 molecules changed (417 / 412 bit-identical), Σ −18 / +38 calls, gains : losses : same 13 : 13 : 26 /
  10 : 20 : 23.
- Class read-out (`/tmp/classattr97.py`): target classes (P–O/P–N/Si–O/Si–N/S–N/S–O present) 12 : 8 : 5, Σ −24
  calls (Σ Δrel −0.52) on train but 7 : 15 : 7, Σ +25 (+0.43) on valid; collateral-only molecules
  (C–S(VI)/C–Si/C–P/S–P/S–S/B–S/halogen, bo 1.05–1.7) 1 : 5 : 10, Σ +6 (+0.32) / 3 : 5 : 8, Σ +13 (+0.30). By the
  strongest cap: bo ≥ 1.7 8 : 4 : 4, Σ −27 / 4 : 11 : 4, Σ +24; 1.3–1.7 3 : 7 : 8, Σ +7 / 4 : 5 : 5, Σ +9; < 1.3
  2 : 2 : 3, Σ +2 / 2 : 4 : 6, Σ +5.
- The train gain is two molecules: 104079126 51 → 37 (cyclic phosphate, P–O bo 2.1–2.8) — but in a different, higher
  basin (rel 1.022 → 0.983, +40.7 kJ/mol) — and 135130097 38 → 29 (phosphoramide, N–P 2.24, O–P 1.85–2.05, rotors
  turning ≤ 10°); 135097481 (S–N cage, bo 6.2) 27 → 28 also changed basin (rel 1.000 → 0.969, +30.8 kJ/mol),
  252089162 25 → 27 (0.994, +5.6 kJ/mol): three basin changes, all to higher energy (train margin −0.076).
- Valid losses: 135011106 19 → 29 (the compressed C(σ)–S(II) bond bo 1.56, collateral; same basin, the rel 0.918
  credit walker), 134985308 25 → 33 (eight P–O bonds bo 2.25–2.65 = 1.50–1.55 Å, rotors turning 58–83°),
  103928880 20 → 24 (S–N bo 2.81 = 1.55 Å, 91° turn), 135084788 25 → 29 (ring P–O 1.74, 112°), 104129283 28 → 31
  (ring P–O 2.2, 174–177° ring flips), 135150142 32 → 35, 135050336 34 → 37, 135052949 27 → 30 (S(VI)–C ring bond
  bo 1.07, ≤ 8 % change); gains 135106724 16 → 12, 135101049 34 → 31, 135239978 19 → 17, 252662734 21 → 19.
- Amplitude read-out (`/tmp/ampl97.py`, strongly capped bonds bo ≥ 1.5, largest start→final turn): near-minimum
  rotors (< 30°) 0 : 3 : 5, Σ +6 / 1 : 5 : 5, Σ +14; 30–60° 4 : 1 : 0, Σ −7 / 0 : 1 : 2, Σ +2; ≥ 60° 5 : 3 : 2,
  Σ −22 / 3 : 7 : 0, Σ +19.

### Interpretation
The inflation is real in the model (the FA bond order of P–O/Si–O/S(VI)–N/S(VI)–O single bonds is 1.7–3 and the
rotation carries it squared), but taking it out does not pay: the near-minimum rotors, which by the cycle-40/41
reading should want the physical (softer) value, lost on both splits (1 : 8, Σ +20), and the far-turning rotors
split 8 : 10 with the sign flipping between splits — a path lottery on top of a neutral endgame. Two readings,
both compatible with cycles 41 and 89: (i) the gas-phase prototype curvatures (trimethyl phosphate, methoxysilane,
methanesulfonamide) do not describe these rotors in multifunctional xTB molecules — cyclic phosphates, P–O–P
bridges, anionic P–O⁻ (1.50–1.55 Å) and sulfonyl-imine S=N (1.55 Å) bonds have anomeric, ionic or genuine
hypervalent-π rotational profiles with curvatures near the FA value, so a 4–9× softening overshoots into the
"too-soft stiff mode" regime that costs steps; (ii) where the guess is over-stiff, the transported secant memory
learns the rotor in one or two calls (cycle 89's P floor read-out), so the guess error is not the binding cost —
the +1.2/+1.7-call excess of these molecules in the census is the path length of their large-amplitude rotors, not
a stiffness cost. The collateral classes behaved as predicted a priori (mild changes of ρ ≈ 1 rotors are noise with
a slight cost: 4 : 10, Σ +19 over both splits), so a scope restriction to N/O partners would recover ≈ +0.0003 /
+0.0006 only and leave a sign-flipping target set — no repair candidate clears the noise floor. Together with
cycles 40, 41, 68/69 and 89 this closes the rotor-stiffness calibration of every element class in both directions
(stiffening: 41, 89; softening: 40, 97; phase-tracked: 68/69).

### Next idea / control
The rotor guess is closed in every population and direction; the excess of the P/S/Si-rich residual molecules is
rotor path length (turns of 60–177°), a walk problem whose step-composition levers are exhausted (memory). Remaining
candidates from the cycle-97 search: (i) H- vs heavy-terminated linear-bend constants (bundle-only, ≈ 2× in k on
alkyne C≡C–H bends), (ii) approach-only pre-move for undocked charged far starts (EV ≤ −0.004 / −0.0055, valid gate
hinges on far-start charged credits), (iii) endgame-only SR1/PSB (≈ neutral). The productive method remains the
reduced-model frequency check of fixed constants (`/tmp/dumfreq96.py`) — the remaining uninspected fixed constants
are the ionic bends/torsions of connected ion complexes and the fragment-rotation/translation floor.

## Cycle 98 — prior-uncertainty metric for the TS-BFGS update (E-weighted Powell direction)

### Hypothesis
The cycle-95 attribution reading ("soft modes are learned only once the stiff modes have converged") is a
property of the *metric* of the update, not of the formula family: every Powell-symmetric update
ΔB = UJᵀ + JUᵀ − U(JᵀS)Uᵀ (J = Y − BS, UᵀS = I) is the least change of the model in some weighted norm, and
TS-BFGS's direction U ∝ (SᵀY)Y + (Sᵀ|B|S)|B|S weights every coordinate by its curvature — a prior of uniform
*relative* uncertainty, under which the residual of a mixed approach step (bonds and bends not yet relaxed, a
rotor or two fragments moving) is charged to the stiff block. The calibration surveys of cycles 37–97 give a
different prior: the Almlöf stretch with row factors is within ≈ 20 % of the local curvature, the halved FA bend
within ≈ 30 %, the linear-bend/ion-contact classes spread 1.2–2×, the torsional and improper constants 1.5–2.2×
(amplitude-dependent), the azimuths and rigid-body coordinates are known to a factor of a few. Weighting the
Powell direction with that prior, u ∝ E x (E = diag(ε), ε bonds 0.2 / bends 0.3 / linear and ion bends 0.5 /
dihedrals 0.7 / azimuths and rigid-body 1.0, only ratios matter, fixed a priori) is the weighted least-change
(MAP) update and should attribute the soft-mode residual of mixed steps to the soft block. Hand algebra on
2-coordinate examples (bond k 0.5, torsion k 0.02; both guesses stiff, opposite-sign errors, stiff-only
residual): soft-mode recovery per step 6 → 25 % (opposite signs), 25/54 → 36/61 % (same sign), spurious
soft-block change from a stiff-only residual −0.7 % → −7 % (E-Greenstadt −25 %, rejected a priori). Expected
lever: the transition steps 2–5 of every walker; endgame-only variants pointless (already soft-dominated,
v_b/v_t ≈ 0.1). Risk named a priori: cycle 33's garbage attribution (SR1/Bofill ×1.6–1.9 in the approach phase)
and the lost stiff-block learning.

### Change (functions)
`_MS_TS_BFGS(B, S, Y, lams, vecs, eps=None, proj=None)`: X = X1 + X2 → X·ε (row-wise) → projected onto range(B)
(X Q Qᵀ), then U = lstsq(XS, X)ᵀ as before (eps None → bit-identical). `update_H(..., eps, proj)`;
`ApproximateHessian.prior_eps` (set by the PES) and `update(dx, dg, proj)`; `PES._secant_prior_proj()` (None)
and `InternalPES._secant_prior_proj()` (Q of `_get_jacobian_qr()` at the current geometry); `PES._update_H`
passes proj on both paths; `Internals._PRIOR_EPS` and `Internals.guess_uncertainty()` (ε per coordinate in the
index order of guess_hessian: translations, bonds, angles [dummy → linear, s-block ion → ion, else angle],
dihedrals [inner dummy → linear, end dummy → azimuth, else dihedral], other, rotations); `InternalPES.__init__`
attaches it to the fresh model after the guess (rebuilds refresh it). Every system (the mechanism is general).
Static checks: `py_compile`; `/tmp/check98.py` (934 start geometries): guess Hessian bit-identical to the
champion, ε present with the model's dimension and the class partition, projection basis orthonormal; class
counts train 10895 × 0.2, 18140 × 0.3, 233 × 0.5, 23599 × 0.7, 2616 × 1.0.

### Result
Candidate `ce5bf61eb592d46b00bce59220e88d0035585760`: train **0.6792257** (+0.0196775, mean rel_energy
1.0021246, Σ(r−1) +1.506 → +0.996, 468 / 469 converged — 384221749 hit the 200-call limit — 62 s; evaluation
`dcb46a4fd4534c028058a2f5b3afc3d5`): **discard** (train gate). Champion restored (`023f69d`), implementation
preserved in `ideas/prior-weighted-update/ce5bf61eb592d46b00bce59220e88d0035585760/algo.py`.
- All 469 walkers changed; gains : losses : same 120 : 171 : 178, Σ +422 calls. Populations (`/tmp/pop98.py`):
  connected +0.0043 (65 : 68, Σ +198 of which +162 is the stall; 2 basin changes), neutral dimers +0.0027
  (33 : 56, Σ +45, 3 basin changes), charged +0.0127 (22 : 47, Σ +179, 9 basin changes, credit −0.498:
  monoatomics__pyrrole 33 → 15 but rel 1.788 → 1.036; carboxylates__water 13 → 47 rel 0.794 → 1.000;
  amides__carboxylates 15 → 36, esters__guanidiniums 10 → 32, amides__ammoniums 25 → 42).
- Connected read-out (`/tmp/dum98.py`): no class signal — with linear units 7 : 9 (Σ +8), without 58 : 59
  (Σ +28 without the stall); by size ≤ 12 atoms 17 : 13 (−2), 12–25 14 : 18 (+33), 25–40 19 : 14 (−18),
  ≥ 40 15 : 23 (+185, +23 without the stall). The typical loss is +1–3 calls in the endgame, the typical gain
  −1–2; 135132759 27 → 11 and 135094473 20 → 12 are the only large connected gains.
- Stall (diagnostics `dcb46a4fd4534c028058a2f5b3afc3d5`, 384221749, 50 atoms, champion 38 calls): normal descent
  to call 38 (−403 kJ/mol), then 160 calls of creep — max force 3.8e-3 → 1.6e-3 Ha/Bohr, rms displacement
  1e-3–1e-2 Bohr per call, −1.8 kJ/mol in total: the model is far too stiff along the remaining force.

### Interpretation
Contradicted, and the mechanism is visible in the algebra once the sign of the stiff residual is taken into
account: the rank-one closure term −(jᵀs) u uᵀ puts |jᵀs| u_i u_j into every entry the direction u touches;
with u concentrated on the soft coordinates and the stiff guesses slightly too stiff (jᵀs < 0, the common case
for calibrated stretches and bends), the soft block in the *unsampled* directions is stiffened by
|jᵀs| u_t² at every mixed step — (ε_t/ε_b)² ≈ 12× more than under the |B| metric, whose small u_t is exactly what
keeps the soft block clean. The multi-secant memory corrects only the sampled span, so the stiffening
accumulates along the soft directions still to be walked, the steps there shrink, and in the worst case the
walker creeps (384221749). The |B| metric of TS-BFGS is therefore protective by construction: it charges the
garbage of mixed steps to the stiff block, where the next stiff-mode step re-learns it at no cost. The better
attribution of soft residuals is real but is worth less than the garbage placement — the same lesson as cycles 5
(BFGS) and 33 (SR1/Bofill), now for the metric itself. Connected systems came out neutral apart from the tail;
multi-fragment systems (rigid-body ε 1.0) lost outright. No bounded repair is indicated: a smaller ε ratio is a
scan of an a-priori constant, connected-only scoping is neutral, the E-Greenstadt form is worse by the same
algebra, and a floor or damping against the creep would be gaming. Line closed: the metric (prior) of the
secant update is not a lever in either direction — more soft-mode weight spreads garbage, less is what TS-BFGS
already does.

### Next idea / control
Update-family and update-metric both closed. Remaining candidates from the cycle-97 list: (i) H- vs
heavy-terminated linear-bend constants (bundle-only), (ii) approach-only pre-move for undocked charged far starts
(parked, valid gate hinges on far-start charged credits), (iii) endgame-only SR1/PSB (≈ neutral, pointless by the
attribution reading). The cycle-98 idea search closed: ionic bends/torsions already calibrated (81), tiny
molecules (slow ones are 0.5–0.65 Å rearrangements), connected symmetry-breaking start (not a call lever);
the rotor-turn path length remains the largest structural excess with its step-composition levers exhausted.

## Cycle 99 — pyramidalisation of planar-start three-coordinate P / As / Sb centres before the first force call

### Hypothesis
A trivalent pnictogen centre heavier than nitrogen is pyramidal at every minimum (X–P–X 92–101°: PH3 93.5, PF3
97.8, PMe3 98.6, PCl3 100.3; As, Sb smaller) with an inversion barrier of 30–40 kcal/mol, so a planar start is
the inversion saddle: zero gradient by symmetry and |k| ≈ 1.4 Ha/rad² of negative curvature against a model bend
curvature of ≈ 0.3. The cycle-85 saddle-creep mechanism (escape amplified by ≈ (1 + |k|/k_model) per step, then
the 0.8–0.9 Å of the centre's height still to travel) applies to connected systems here for the first time —
cycle 43's symmetry breaking is per-fragment rigid-body only and never touched a connected start. Census on the
cycle-96 files: all 15 three-coordinate P centres of both splits end pyramidal (280–318°); 12 molecules start
planar (sum ≥ 354°, |h| 0.007–0.2 Å; 7 train, 5 valid), +2.1 calls (median +0.7) above the cycle-95
size/displacement regression, the tiny-|h| ones ≈ +3; the final side equals the sign of the start height in 20/23
centres (11/11 at |h| ≥ 0.05 Å; the misses are |h| ≤ 0.026 Å, where the environment's torque decides). No
planar-start P/As/Sb in multi-fragment systems; no counter-example (no planar P minimum). Near-planar amine-type
N (≈ +4 calls) was examined and excluded: the planar-start N that end pyramidal end at 337–351° (anilines,
hydrazines, NF2, bridgehead amines — five all-sp³ cases end 340–351°, none at the 322–333° of a plain amine), so
no a-priori class target exists. Expected ≈ −0.002 train / −0.0015 valid; energy-gate risk = diastereomer
re-rolls on ≤ 12 molecules against margins of +1.5 / +3.7.

### Change (functions)
- New section after `_prerelax_rotors`: constants `_PYR_ANGLE = {15: 98°, 33: 96°, 51: 95°}` (textbook class
  angles), `_PYR_TERMINAL = (7, 8, 16, 34)`, `_PYR_CLASH_X = 0.5`; helpers `_pyr_subtree` (atoms reachable from a
  neighbour without passing through the centre), `_pyr_tilt` (rigid Rodrigues rotation of every substituent
  subtree about the axis normal × bond through the centre until the bond lies at elevation −τ below the neighbour
  plane, sin²τ = (2 cos θ + 1)/3, τ = 29.4° for 98°; bond lengths and substituent geometries preserved, the centre
  keeps its position), `_pyramidalise_centres(numbers, pos_ang)`: for every P/As/Sb with exactly three
  neighbours (1.25 × r_cov bond graph) whose angle sum exceeds `_ROTOR_PLANAR_ANGLE_SUM` (350°, reused), not in a
  ring through the centre, without a one-coordinate N/O/S/Se neighbour (P=O, P=S: sp² P(V)) or an s-block metal
  neighbour: tilt toward the side the centre already leans to (sign of its height over the neighbour plane);
  reject a trial that changes the bond graph or buries atoms (a non-bonded, non-geminal contact inside
  0.5 × the UFF 12-6 minimum distance x_ij — thousands of well depths; tighter contacts are ordinary template
  strain and are left to the walker); when the leaning side is rejected the other side is tried; centres are
  processed sequentially on the updated geometry.
- `minimize_func`: `pos_ang = _pyramidalise_centres(atomic_numbers, pos_ang)` after `_prerelax_rotors`, before
  `_break_start_symmetry` / docking.
- Static check (`/tmp/check99.py`, pure geometry on the saved starts): triggered exactly on the non-ring census
  population — train 135097653, 252089162 (2 P), 135101309, 135098013, 135094473, 135095493; valid 135101049,
  135100404 (3 P), 135093104 — angle sums 293.6–294.0°, bond-length deviations ≤ 1e-15 Å, max atomic move
  1.1–3.1 Å (subtree swings), rmsd to the champion's final geometry reduced in 8/9 molecules (135095493 0.48 →
  0.08 Å); 10/12 centres on the observed final side (misses 135098013 P8 at h −0.026 and 135100404 P8 at
  +0.012); all other starts bit-identical (463/462). The ring centres (valid 135150142, 135084788; train
  135094613's cage) are untouched by design. A first draft with `_PYR_CLASH_X` 0.6 flipped 135100404 P9 to the
  non-leaning side on an F···F contact of 2.00 Å (0.595 x) — the guard is for burials, not for the tight contacts
  of a crude template, hence 0.5 (fixed; no scan).

### Result
- Train: 0.6595482 → **0.6574843** (−0.0020638, gate passed); mean rel_energy 1.00321 → 1.00330 (margin
  Σ(r−1) 1.506 → 1.549), all 469 converged. 463 bit-identical; the 6 touched molecules Σ −22 calls:
  135094473 20 → 9, 252089162 25 → 17, 135101309 22 → 17, 135095493 8 → 5 (same basin, same energy);
  135098013 17 → 19 in a *different* basin 12.1 kJ/mol below the reference's (rel_energy 1.043 — the non-leaning
  side found the lower diastereomer); 135097653 17 → 20 (same basin, +3).
- Valid: 0.6555918 → **0.6545945** (−0.0009973, gate passed); mean rel_energy 1.00799 → 1.00800 (margin
  3.716 → 3.722), all 465 converged. 462 bit-identical; 3 touched, Σ −16, 3 gains: 135101049 34 → 22 (different
  basin, −3.0 kJ/mol), 135100404 23 → 21 (rmsd 1.03 Å to the champion's final at the same energy to 1 J/mol —
  the mirror-image / permuted C3 minimum), 135093104 11 → 9.
- Decision: **keep** → champion 192712c, train 0.6574843 / valid 0.6545945.

### Interpretation
The saddle-creep mechanism of cycle 85 transfers to the inversion mode of connected systems: removing the planar
P saddle saves 3–11 calls per molecule (the tiny-|h| starts 135094473 and 252089162 gain most, 20 → 9 and
25 → 17), with no energy cost — the two "wrong-side" diastereomers came out 12 and 3 kJ/mol *below* the
reference's minima, and the coupled N(PF2)3 case reached a C3 minimum of the same energy. The one loss
(135097653, +3, 30 atoms, ref 28) is a leaning start (|h| 0.09) whose creep was already short; the subtree swing
re-arranges a Cl···H contact and the walk pays for it — within the ±3 noise of a 20-call run. The gain per
touched molecule (−3.7 train / −5.3 valid) is far above the census estimate because the census's regression
"excess" already contained the displacement term of the pyramidalisation itself (rmsd0 0.4–1.4 Å), which the
pre-move removes as well. The 0.5 × x_ij burial guard and the bond-graph test never fired on the leaning side of
the census population; the sign-of-lean rule is now supported 10/12 on the pre-moved starts and by the energies.

### Next idea / control
Bounded follow-ups of this line: (a) ring centres — valid 135150142 (32 calls, ref 51, h −0.11) and 135084788
(25, ref 36, h +0.026) carry the largest planar-P excess (+11/+8.5 over the regression) but need a
ring-compatible construction (move the centre along the normal with the exocyclic subtrees and let the ring
bonds stretch by ≈ 0.15–0.2 Å, or pucker the ring); (b) amine-type N stays closed for lack of an a-priori target
(conjugation-dependent 337–351° finals); (c) the analogous three-coordinate S(III)/Se centres (sulfonium,
sulfoxide-type) have no planar starts in the data. The next idea search should look at the remaining
saddle-start classes of connected systems (planar carbanion 104079126 is charged and excluded; other
symmetric-start modes: e.g. eclipsed X–Y single bonds between two sp³ centres, ring puckers starting flat —
the ring line was closed on every population after 95, but a *saddle-start* ring (flat cyclohexane) is a
different mechanism from the ring-deflation model terms).

## Cycle 100 — pyramidalisation of planar-start difluoroamine nitrogen centres before the first force call

### Hypothesis
A nitrogen carrying two fluorine atoms is the first-row member of the cycle-99 class: the two fluorines raise the
amine's inversion barrier from ≈ 6 to tens of kcal/mol (NF3 ≈ 50–60, HNF2 / RNF2 ≈ 30–40) and every difluoroamine
minimum is strongly pyramidal (NF3 102.4°; HNF2, CH3NF2, CF3NF2, N2F4 102–105°) whatever the third substituent —
unlike the ordinary amine, whose planar-start finals (337–351°) are conjugation-dependent and were closed in cycle
99 for lack of an a-priori target. A planar NF2 start is therefore the inversion saddle of a mode with |k| ≈ 0.5–
0.7 Ha/rad² against the model's ≈ 0.3 (escape ≈ 3×/step), the cycle-85/99 saddle-creep mechanism. Census on the
cycle-99 files (`/tmp/nf2_100.py`): the class is one molecule per split, both geminal poly-NF2 carbons — train
135096784 (C16F2(NF2)–NF–C17(NF2)3: N11–N14 start 350–358°, heights 0.13–0.26 Å, end 320–323° [104/109/109]; 19
calls, ref 16, regression excess +7.0) and valid 135093932 (F–N=C(NF2)2: N7/N8 start 358/360°, heights 0.10/0.015
Å, end 334/327°; 16 calls, ref 20, excess +4.0); the monofluoroamine N15 (349 → 343°), aryl NCl2 (134983358,
360 → 351°) and the already-pyramidal NI3 / NBr3 / N(F)S2 centres are outside the class or below the planarity
threshold. Static check (`/tmp/check100.py`, `/tmp/side100.py`): the rule with θ = 103° moves exactly these two
starts (468 / 464 bit-identical), all six centres land on the side of the champion's final (signed volumes
±0.908 vs ±0.71–0.84), bond lengths preserved to 1e-15 Å, angle sums 309° — an overshoot of 11–25° against the
xTB endpoints (the geminal groups are flattened sterically / by conjugation on the sp² carbon), which brings the
fluorines of the adjacent NF2 groups of C17(NF2)3 to 1.91 / 1.95 Å (0.57 × x_FF; final 2.65 Å) and F2···F4 of the
valid molecule to 2.27 Å (final 2.64); the rmsd to the champion's final *rises* (0.248 → 0.327, 0.323 → 0.402 —
`/tmp/over100.py` shows it would not fall below the start's value even at 109°: the F azimuths and thermal
displacements dominate the rmsd, the inversion coordinate alone is 3× closer). Expected: train −0.0004…−0.0008
(3–6 calls on a ref-16 molecule), valid a coin flip (one net call on a ref-20 molecule clears the gate at
1.075e-4; the return from the overshoot on a stiff mode against a genuine saddle escape at h 0.015 Å). Energy
risk: 2⁴ / 2² side combinations on two molecules against margins +1.549 / +3.722.

Considered and set aside in the same design: (a) a ring-compatible construction for the two valid ring P centres
(135150142 32/51, 135084788 25/36) — an exocyclic-only variant (ring untouched, exocyclic subtree rotated about
the centre until its bond makes the class angle with both ring bonds, i.e. ≈ 75–78° out of the ring plane) was
implemented and checked statically: for 135150142 the leaning side is buried by the 13-atom exocyclic subtree,
the fall-back side inverts the centre's chirality against the final (signed volume −0.91 vs +0.97), and for
135084788 it places the exocyclic H 2.4 Å from its final position (the 7-ring puckers ≈ 1 Å at P; rmsd to the
final 0.60 → 0.69) — dropped before the run (population valid-only, no train signal; a ring-compatible
construction has to pucker the ring with the centre). (b) Extending the class to monofluoroamines (N15 of
135096784 ends 343°: no class target).

### Change (functions)
- `_pyr_class_angle(numbers, adj, c)`: P / As / Sb → `_PYR_ANGLE` as before; N with ≥ 2 fluorine neighbours
  (1.25 × r_cov bond graph) → `_PYR_ANGLE_NF2 = 103.0°` (NF3 102.4°; HNF2 / RNF2 102–105°, fixed a priori);
  otherwise None. `_pyramidalise_centres` now builds the bond graph for every molecule containing P / As / Sb
  or N, selects the centres by `_pyr_class_angle`, and uses that angle for τ; the construction (`_pyr_tilt`,
  now through the small helper `_pyr_rotation`), the side rule, the planarity threshold and every guard (three
  neighbours, no ring through the centre, no one-coordinate N/O/S/Se or s-block neighbour, bond graph unchanged,
  no burial inside 0.5 × x_ij, else the other side) are unchanged. Comment block updated. Commit
  `3ddd4df5afc80dac0650bb5b022b448daa8a7906`.

### Result
- Train: 0.6574843 → **0.6566848** (−0.0007996, gate passed); mean rel_energy 1.0033023 (margin Σ(r−1) 1.549,
  unchanged), 469 / 469 converged. 468 bit-identical; 135096784 19 → 13 (ref 16; same basin, same energy to
  1e-9 Ha, final geometry identical to the champion's to 0.00 Å). Evidence `evaluation_results/cycle-100-train.json`
  (38d0374521ea47fbbafd489deec7d662, md5 1b1c3f320c9969c6b9dbadfe3140f202).
- Valid: 0.6545945 → **0.6541644** (−0.0004301, gate passed); mean rel_energy 1.0080039 (margin 3.722,
  unchanged), 465 / 465 converged. 464 bit-identical; 135093932 16 → 12 (ref 20; same basin, same energy, same
  final geometry). Evidence `evaluation_results/cycle-100-valid.json` (91ffac26290b49edb979fdeebb077457, md5
  8e066d68c58c3d16b9848137c3aada02).
- Decision: **keep** → champion 3ddd4df, train 0.6566848 / valid 0.6541644.

### Interpretation
The saddle-creep mechanism holds for the difluoroamine class exactly as for the pnictogens: −6 and −4 calls on the
only two members, both walks ending in the champion's basin at the identical geometry. The 11–25° overshoot of the
textbook class angle against the sterically / conjugatively flattened xTB endpoints and the 1.9 Å F···F contacts
it creates in C17(NF2)3 cost nothing visible — the return along a stiff, well-sampled inversion mode and the
opening of a tracked short contact are ordinary first-step work, whereas the escape from a genuine saddle
(135093932's N8 at h = 0.015 Å) and the 0.3–0.5 Å of height to travel are what the walker paid for. The rmsd to
the final geometry is the wrong figure of merit for a pre-move: it is dominated by rotor azimuths and thermal
displacements that the walker corrects in the same steps anyway; what counts is the removal of the zero-gradient
mode. The rule's scope is now every three-coordinate inversion centre with an a-priori strongly pyramidal class
geometry and no ring through the centre; the remaining planar-start inversion populations have no such target
(amine N, NCl2, monofluoroamine) or are ring centres (valid-only, construction not built).

### Next idea / control
The zero-call start-rule line (cycles 85, 99, 100) is exhausted on the census populations: the eclipsed sp³–sp³
class is 1–2 train / 0 valid molecules, conjugated sp²–sp² phase classes carry no excess (`/tmp/conj100.py`),
flat-start six-rings are two molecules that stay flat, and the feature-class table (`/tmp/feat100.py`) has no
class with a consistent excess on both splits. The remaining cost is the dimer endgame (near-equilibrium
same-basin dimers at 13–14 calls; amides__pyridine__valid 29 vs 15, ammoniums__pyridine 25/18 vs 19/11 with
identical finals) and the charged diff-basin walkers that carry the energy margins — both closed at the level
of model/trust/update changes; a new lever there needs trajectory-level evidence that the passive diagnostics
do not provide for converged runs.

## Cycle 101 — surrogate docking of polyatomic-cation complexes from separated starts (keep)

### Hypothesis
The cycle-100 cost map (`/tmp/hist101.py`) puts 61 % of the cost in connected systems at a uniform rel 0.70–0.73
across call bins, neutral dimers at rel 0.45 (docked since cycle 70) and charged dimers at rel 0.77 / 0.85, where
the same-basin excess over the reference is only 14 / 17 calls per split and the slow runs are approach walks of
28–42 calls from separated starts. Cycle 75 (cation docking, discarded: train −0.0026 / valid +0.0033) was closed
with "no start criterion separates winners from losers". Re-binning its per-molecule results by the closest
inter-fragment contact at the start (`/tmp/cat75.py`, Bondi ratio bins) contradicts that closure: every run that
started separated (ratio ≥ 1.15) gained (Σ −51 calls), every loser (+13 … +23 calls) started in contact or at
mid range — the surrogate's charge sharing among the equivalent N–H / C–H donors of a polyatomic cation makes
the contact poses near-degenerate, so re-arranging an existing contact is a coin flip, while forming the contact
from a separated start replaces the approach walk. Reopened with a physical scope: dock a cation complex only when
no inter-fragment atom pair lies inside its UFF 12-6 minimum distance x_ij = sqrt(x_i x_j) (the surrogate's own
contact scale, the same x_ij as `_PYR_CLASH_X`; fraction 1.0, not scanned). Anions (diffuse, polarisable, partly
covalent H-bonds), bare ions (neutral-atom 12-6 radii, no polarisation) and odd-electron fragments stay undocked.
Census (`/tmp/census101.py`): 12 train / 8 valid cation complexes newly eligible, credits at stake +0.118 / +0.019
(valid margin 3.722 survives losing all of them); the neutral class is bit-identical (`/tmp/check101.py`: 133 /
130 identical eligibility and surrogate terms, none lost).

### Change (functions)
- Rationale block retitled "Rigid-body pre-relaxation ("docking") of multi-fragment starts on a classical
  intermolecular surrogate", new "Ionic complexes." paragraph (bare ions, anions, open-shell, polyatomic cations
  docked only from separated starts).
- `_dock_formal_charges` returns `(q, onium)`: the cationic centres (alkali +1, alkaline earth +2, N d4, O d3,
  sulfonium, phosphonium, amidinium / guanidinium / imidazolium C) are collected in `onium` and added to `q`;
  anion rules unchanged.
- `_dock_surrogate_terms` returns `(terms, charged)` or None: refuses anions (`net < −1e-6`), bare ions
  (`net > 1e-6` with one atom) and odd parity; places each onium charge on the hydrogens bonded to the centre or
  its neighbours (equal shares, ESP-like — the cycle-75 placement) before the 12-6 / Coulomb terms are built.
- `_dock_start`: after the surrogate terms, a charged complex is docked only if
  `min(|r_i − r_j| / x_ij) ≥ 1.0` over the inter-fragment pairs; otherwise untouched. Docstring updated.
- `_prerelax_rotors` caller indexes `_dock_formal_charges(...)[0]`; `minimize_func` comment updated.
- Commit `cf2439f9ee81bc0e65f361fad971390d682ad652`.

### Result
- Train: 0.6566848 → **0.6512909** (−0.0053939, gate passed); mean rel_energy 1.0033716 (margin Σ(r−1) 1.549 →
  1.581), 469 / 469 converged. 457 bit-identical, 12 changed, Σ −75 calls, no loser: imidazolium__water 28 → 9
  (ref 79), imidazolium__ketones 36 → 24 (ref 23; credit 1.029 kept at rmsd 1.96 Å from the old final),
  ammoniums__water 32 → 22, amides__guanidiniums 38 → 29 (credit 1.089 kept, same final),
  alcohols__guanidiniums 19 → 11, alkanes__imidazolium 14 → 10, ammoniums__pyrrole 16 → 12,
  ethers__guanidiniums 14 → 10, alkenes__imidazolium 16 → 14, ammoniums__benzene 17 → 15 (1.000 → 1.033, new
  basin), imidazolium__thiols 14 → 13, guanidiniums__thiols 13 → 13. Evidence
  `evaluation_results/cycle-101-train.json` (61b40cc9bc40462eae8ba474e87ed17a, md5
  6e63eadee0661ac0a5dc5f2b751634fc, 54.8 s).
- Valid: 0.6541644 → **0.6499489** (−0.0042155, gate passed); mean rel_energy 1.0079088 (margin 3.722 → 3.678),
  465 / 465 converged. 457 bit-identical, 8 changed, Σ −71 calls, 6 faster, 2 equal: ammoniums__ethers 25 → 6
  (ref 24; 1.013 → 1.000), benzene__imidazolium 26 → 11 (ref 68; 1.001 → 0.975), acids__imidazolium 35 → 23,
  amines__imidazolium 23 → 11 (1.004 kept at a different geometry, rmsd 1.58 Å), guanidiniums__thiols 19 → 10,
  alkanes__ammoniums 11 → 7, ammoniums__sulfides 25 → 25 (1.000 → 0.995, new basin), guanidiniums__pyrrole
  32 → 32 (pose identical, rmsd 0.00). Evidence `evaluation_results/cycle-101-valid.json`
  (0811ac7ebca8444bbf7760764f75b1af, md5 1d8ce30fb83534cef457ed93d5598f93, 42.3 s).
- Decision: **keep** → champion cf2439f, train 0.6512909 / valid 0.6499489.

### Interpretation
The separated-start scope isolates the part of cycle 75 that was a mechanism: for a cation whose charge sits on
several equivalent protons the surrogate cannot rank the contact poses, but it does not have to when the start
has no contact yet — any of the near-degenerate poses is a 10–20-call head start over the walker's own approach
along a flat Coulomb slope with a model that has no inter-fragment curvature until the secant pairs arrive. The
20 touched runs lost Σ −146 calls without a single loser; the three basin changes (ammoniums__benzene +0.033,
ammoniums__ethers −0.013, benzene__imidazolium −0.026, ammoniums__sulfides −0.005) net to zero within the
lottery band and the diff-basin margin carriers (imidazolium__ketones 1.029, amides__guanidiniums 1.089,
amines__imidazolium 1.004) kept their credits. The cycle-75 losers are exactly the runs the new gate excludes, so
the earlier closure was a scoping error, not a refutation of the surrogate for cations.

### Next idea / control
Two eligible valid runs did not move in calls (ammoniums__sulfides 25, new basin; guanidiniums__pyrrole 32,
identical pose — the docked pose either equalled the start within 0.5 Å rmsd or the 6 Å per-atom cap refused it).
Anion far starts stay undocked: the census shows them mostly already fast (13–20 calls), the carriers
carboxylates__ketones (+3.2 valid credit, separation ≈ 1.03 under the UFF criterion) and carboxylates__ethers
(+0.36) would be at risk, and anions are the most polarisable / charge-transfer-prone fragments for a fixed-charge
surrogate (EV ≈ −0.001 per split, gate risk on valid). Bare-ion far starts are already 10–18 calls with
neutral-atom 12-6 radii as the only contact scale — not run. The remaining cost is unchanged in kind: the
connected endgame (learning-limited on soft modes; the displacement criterion is the binding one for rotors) and
the charged diff-basin margin carriers.

## Cycle 102 — multi-start surrogate docking (24 octahedral orientations of the smallest fragment) — INVALID (discard)

**Hypothesis.** The local rigid-body relaxation of cycle 70/101 lands in the surrogate minimum nearest the start's arbitrary orientation; 5 train / 12 valid docked neutral dimers end above the reference minimum (Σ credits −0.52 / −2.29). Taking the lowest of 24 surrogate minima (identity + 23 proper octahedral rotations of the smallest fragment about its centroid, `_dock_start_rotations`, tie 1e-6 kcal/mol `_DOCK_ENERGY_TIE`) should land in the strongest contact pattern / deepest xTB basin.

**Change (functions).** `_dock_start` (multi-start loop, lowest surrogate energy, same 6 Å / 0.5 Å gates), new `_dock_start_rotations`, `_DOCK_ENERGY_TIE`, `import itertools`, rationale paragraph. Candidate 327825e, anchor cf2439f.

**Result.** Train 0.6512909 → 0.6595679 (Δ +0.0082770), mean rel_energy 0.9888733 (invalid, energy_below_baseline; Σ(r−1) −5.22 vs +1.58), 469/469 converged, 0 internal errors, evaluation 91fda751730a49858cbff5badc1c0eb7. Per-molecule: 95 dimers changed, ~90 end in a different basin (rmsd 0.8–2.5 Å); credit lost on 40 (Σ −8.08), gained on 15 (Σ +1.28); 45 faster / 46 slower, net +105 calls (e.g. alcohols__esters 1.052 → 0.160, acids__sulfides 9 → 26 calls).

**Interpretation.** Energy + cost failure of the mechanism, not an implementation error. The point-charge + UFF 12-6 surrogate ranks poses far worse than the start orientation does: its global minimum is a different xTB basin on almost every touched dimer, and those basins are on average higher, not lower. The near-start pose carries basin information from the dataset construction that the surrogate cannot reproduce; global-pose docking is refuted in general (losses on in-contact and loose starts alike), so no bounded repair (far-start restriction, acceptance margin = new constant) is justified. Champion restored: cf2439f (train 0.6512909 / valid 0.6499489). Implementation preserved in ideas/multi-start-docking/327825e…/algo.py.

**Next idea/control.** Docking stays local (nearest surrogate minimum). Remaining levers per the cycle-101 cost map are uniform across classes; next cycle should target connected-system endgame accuracy rather than start-pose search.

## Cycle 103 — torsion-class constants of connected systems pushed to the stiff end of their physical ranges (plain sp²–sp² 0.65 → 0.5 with carbonyl/nitro-conjugated sp²–sp² links kept at 0.65, sp²–lone-pair 0.45 → 0.35, carbonyl–lone-pair 0.7 → 1: the adverse class of cycles 37/91/93 returned to the full Fischer–Almlöf constant; multi-fragment systems untouched)

Candidate: `_torsion_class_factor` constants (connected only), from champion cf2439f (train 0.6512909 / valid 0.6499489).

### Hypothesis
Idea search before the cycle (saved data and static analysis only, no
PES calls; scripts `/tmp/feat103.py`, `/tmp/axial103.py`,
`/tmp/dockrefuse103.py`, `/tmp/rotors103.py`, `/tmp/torscls103.py`):
- Connected cost is diffuse: an 18-feature regression of the champion's
  calls (heavy atoms, rmsd₀, aromatic / saturated rings, aryl
  substituents, rotatable and turned bonds, heteroatoms, halogens, rows
  ≥ 3, polar H, linear centres, intramolecular H-bonds, XH₃ groups,
  charge) leaves rms residuals 4.35 / 4.54 calls (baseline ours = 6.16 +
  1.25 heavy/10 + 12.60 rmsd₀), rmsd₀ is the only strong term (t 11) and
  no structural feature carries a positive residual on both splits — no
  connected class to census for a model repair.
- Dimer endgame: the same-basin cost of two-fragment complexes is the
  same 15–16 calls in every docking class (docked 146 runs 15.5 / ref
  43.1, refused-small 59 at 15.3 / 40.5, cation-in-contact 56 at 15.9 /
  29.1, undocked anions / bare ions / open-shell 46 at 15.0 / 25.0):
  the higher rel of the charged classes is the fast reference, not a
  slow endgame, so no docking / acceptance lever remains
  (`_DOCK_MIN_RMSD` refusals cost −0.28 calls vs the cost model,
  compressed DES370K starts (12-6 ratio < 0.8, n 43) −0.95).  Fragment
  rotation about the contact line (start → endpoint) does not predict
  cost (rms residual unchanged with rotation terms; the largest axial
  rotations sit in fast docked runs).  Four compressed pyridine / phenol
  outliers (+49 calls, ≈ 0.0035) converged and are untraceable.
- Closed statically (this and the previous session): rotor↔pose
  coupling (already in the Gauss–Newton projection of the contact
  block), energy-augmented secants (four discards), rotor cosine-fit
  extrapolation, endgame-only SR1 (EV ≈ 0), noise-floor ρ guard,
  per-coordinate diagonal reading, surrogate accuracy / force-field
  libraries, parametric contact-scale fit, bend–bend coupling,
  local-symmetry images, symmetry-breaking kick for connected systems
  (connected walkers are environment-torque driven), tiny-inorganic
  anharmonic walks, near-start dimer floor decomposition, connected
  losers 4 / 257, rotor-turn census (111 / 250 molecules at rel 0.71,
  excess absolute), free six-fold rotors (3 train / 6 valid molecules;
  stiffening = gaming, softening walks further), amide N-methyl (3–7×)
  and acetyl (2–5×) over-stiff rotors (family closed 68/69/91/93/97),
  credit walkers (0.015 / 0.027 of the score, but they carry the energy
  margins +1.58 / +3.68).  The endgame is model-/anharmonicity-limited,
  not noise-limited (xTB energies smooth to < 1e-10 Ha).
- The one sanctioned open item with replicated evidence is the cycle-37
  torsion-class refinement (log of cycle 37: "carbonyl–lp back towards
  the full constant and the paying classes pushed towards their physical
  values (pi–pi 0.5, pi–lp 0.35)"; cycle 93: "remains a possible small
  connected-only refinement that would have to be bundled").

Mechanism.  `_torsion_class_factor` scales the shared Fischer–Almlöf
torsional constant of acyclic bonds to a planar centre.  Cycle 37 set
the classes to the stiff end of their physical ranges and measured the
class slopes on both splits: sp²–sp² −0.35 / −0.26 calls per bond,
sp²–lp −0.03 / −0.14, sp²–sp³ neutral, carbonyl–lp **+0.33 / +0.30**
(adverse on both splits); cycles 91 and 93 found the carbonyl–lp factor
adverse on dimer fragments as well, and cycle 93 ran with the fragment
out-of-plane coordinates in place — so the cycle-37 alternative (ii)
(the rotor dihedrals also carried the carbonyl wag) is refuted and
alternative (i) stands: GFN2-xTB's amide / ester conjugation is stiffer
than the experimental 15–20 kcal/mol barriers, and the full constant
(0.045–0.095 Ha/rad²) is the better guess.  The bundle therefore (a)
returns carbonyl–lp to 1 (60 train / 42 valid bonds in 40 / 27
molecules; −0.3 calls per bond → −18 / −13 calls), (b) moves plain
sp²–sp² links (biaryl / styrene / aryl–pyridyl, V₂ 2–3 kcal/mol,
0.006–0.010 Ha/rad² against the 0.65 guess 0.015–0.021) to 0.5 while
the carbonyl- or nitro-conjugated sp²–sp² links (enones, aryl ketones /
acids / amides, nitroarenes, 5–8 kcal/mol, 0.016–0.025) keep 0.65 — the
same carbonyl distinction the lone-pair class already makes (152 / 214
sp²–sp² bonds, 30 / 26 carbonyl-conjugated, 11 / 14 nitro), and (c)
moves sp²–lp (anisole / phenol / aniline, 0.010–0.017 against the 0.45
guess 0.015–0.022) to 0.35 (156 / 191 bonds).  The bond-order fade
(1.5–2.2), the ring exclusion, the 1/√n sharing and the sp²–sp³ factor
0.6 are unchanged; multi-fragment systems keep `scale_torsions =
False` (fragment torsion classes closed in every variant, cycle 93).

Prediction: by the cycle-37 slopes and a half-slope for the further
softening, ≈ −38 / −40 calls → −0.004 per split, at the ±0.005 noise
floor; credibility rests on the class-level replication across the
splits (carbonyl–lp molecules and plain sp²–sp² molecules better on
both), as in cycle 37.  Controls: every multi-fragment system and every
connected molecule without an acyclic sp²–sp², sp²–lp or carbonyl–lp
bond (130 train / 90 valid) must be bit-identical in calls and energy.
Energy gate: connected-only change, dimer credits untouched; basin
changes possible only on torsion-turned connected molecules.  Constants
are physical calibrations fixed a priori, one evaluation, no scan.

### Change (functions)
`_torsion_class_factor` defaults: `s_pi_lp` 0.45 → 0.35, `s_carbonyl_lp`
0.7 → 1.0, `s_pi_pi` 0.65 → 0.5, new `s_carbonyl_pi` 0.65 for sp²–sp²
links with a carbonyl-like (terminal O/S) centre on either side (shared
`carbonyl_like` helper, docstring).  Candidate `bbed8b6a05f885a2f5fd71070197580c28a3d66c`,
anchor `cf2439f9ee81bc0e65f361fad971390d682ad652`.  Static unit check
of the class logic on stub graphs (amide 1.0, acetyl 0.6, aryl–C(=O)
0.65, plain sp²–sp² 0.5 / 0.75 at bo 1.85, phenol 0.35, nitroarene
0.65) before the commit.

### Result
Train evaluation `8bf9953db6b14a4a926edd4f36872041`
([JSON](evaluation_results/cycle-103-train.json)): 469/469 converged,
**C_S = 0.6515659** (champion 0.6512909, Δ **+0.0002750**), mean
rel_energy 1.0034456 (valid; champion 1.0033716).  Decision
**discard** (train improvement below 1e-4); no validation run.
Champion cf2439f restored in `75fb2d3`.
- Controls held: 349 untouched runs, of which 345 bit-identical; the 4
  exceptions are touched by the rule, not by my census — 135100494 (a
  B/S cage whose 16-membered-ring B–S bonds are "acyclic" for the
  ring_max 8 rule; 15 → 15 calls, ΔE 3e-6 kcal/mol) and three in-contact
  DES370K complexes that Sella's 1.25 r_cov bonding rule treats as
  connected (ketones__phenol 28 → 29, monoatomics__phenol 25 → 25,
  guanidiniums__water 14 → 14; energy differences ≤ 4e-6 kcal/mol).
- Touched set (120 connected molecules, `/tmp/cmp103.py`): Σ −2 calls
  (29 better : 25 worse), score +0.00023 because the losers sit on low
  reference counts (134983358 16 → 19 at ref 15, zanubrutinib 11 → 14 at
  ref 18, 135122079 27 → 33 at ref 38) and the winners on high ones
  (ibrutinib 16 → 13 at ref 22, 135227471 16 → 13, 160853090 56 → 53 at
  ref 89).  Class regression of Δcalls on the bond counts: **carbonyl–lp
  +0.18 calls per bond** (single-class molecules +5 over 8, 2 : 3;
  172113611 35 → 39, 134997543 28 → 32), plain sp²–sp² −0.09 per bond
  (single-class −3 over 30, 5 : 3), sp²–lp +0.01 (single-class −4 over
  29, 8 : 5).
- One basin change, favourable: 134997543 (four carbonyl–lp bonds) left
  its +0.646 kcal/mol debit basin for the reference basin (28 → 32
  calls); the train energy margin improves accordingly (mean rel_energy
  1.00337 → 1.00345).

### Interpretation
- The carbonyl–lp class is adverse in both directions: 1 → 0.7 cost
  +0.33 / +0.30 calls per bond in cycle 37 (against champion fc3407d),
  0.7 → 1 costs +0.18 per bond now (against cf2439f).  A monotone
  stiffness response cannot produce that; the class "slope" is the
  path re-roll of the 40 amide / ester / acid molecules (each change of
  the first-step direction re-draws a ±2-call endgame), i.e. cycle 37's
  alternative (iii).  The cycle-93 fragment evidence (amides__water /
  amides__phenol +4 each in cycles 91 and 93) was the same re-roll
  read twice on the same paths.
- The paying classes did what the slopes said, but at a third of the
  cycle-37 size (plain sp²–sp² −0.09 per bond, sp²–lp ≈ 0): the
  remaining over-stiffness of 1.5–2× is inside the range the TS-BFGS
  update removes in one step, so the first-step gain has been
  harvested; further softening toward the physical minimum-curvature
  values only trades first-step accuracy for endgame re-rolls.  The
  carbonyl-conjugated sp²–sp² sub-class (kept at 0.65) cannot be read
  separately from a null result.
- Alternatives considered: (a) the carbonyl–lp change is right but
  masked by the sp²–sp² / sp²–lp changes on the same molecules — the
  single-class sets (8 / 30 / 29 molecules) show the same signs as the
  regression, so no; (b) noise — the touched-set Σ is −2 calls with a
  balanced 29 : 25, i.e. the whole bundle is inside the ±0.005 floor
  either way.  No bounded repair: every constant is a fixed
  calibration, and a sub-class split of the null classes would be a
  scan.
- Uncertainty: one train evaluation; the class regression rests on
  40 / 77 / 77 molecules with 1–6 bonds each.  What is certain is that
  the bundle does not reach the 1e-4 train gate, and that the class
  signs of cycle 37 do not replicate 66 cycles later.

### Next idea / control
**Torsion-class calibration closed in both directions** for connected
systems: the cycle-37 constants (0.6 / 0.45 / 0.7 / 0.65) stay; the
carbonyl–lp "adverse class" is a re-roll signature, not a model
defect, and is not to be reverted or scanned again.  With the torsion
line, the fragment torsion classes (93), the rotor stiffness family
(68/69/97), the guess-constant families (67/89), the start rules
(85/88/99/100), docking (70/90/101/102), the update (5/33/58/98), the
trust region (23–32/44/59/84) and the contact block (34–49/62–65/79)
all closed, the remaining cost is the diffuse learning floor of the
connected endgame (6.16 + 1.25 heavy/10 + 12.6 rmsd₀, rel 0.77–0.80 in
every reference bin) and the 15–16-call same-basin dimer endgame.  The
next cycle needs a mechanism that changes the learning floor itself
(how many calls a soft direction costs to learn), not another
calibration; candidates must be static-checked against the cycle-103
closure list before a cycle is spent.

## Cycle 104 — hydride factors of the Almlöf stretch guess: polar X–H bonds (N–H ×1.2, O–H ×1.4, F–H ×2, S–H ×1.25, H–Cl ×1.3) of connected systems set to the textbook harmonic force constants

Candidate: `_HYDRIDE_SCALE` table and `_h0_bond` (connected only), from champion cf2439f (train 0.6512909 / valid 0.6499489).

### Hypothesis
Idea search before the cycle (saved data and static analysis only, no
PES calls; scripts `/tmp/acetate104*.py`, `/tmp/worst104.py`,
`/tmp/bins104.py`, `/tmp/dockvictims104.py`, `/tmp/small104.py`,
`/tmp/freq104*.py`, `/tmp/xh104.py`, `/tmp/kmodel104.py`,
`/tmp/stretch104*.py`).  Static closures, for the record:
- Acetate / carboxylate methyl start rule (the cycle-88 ester exception
  extended to carboxylate frames): no train population is predictable
  from the start — the train carboxylate dimers start symmetric and far
  (pyrrole partner 5.84 / 5.68 Å to both oxygens), partial turners of
  6–20° would be overshot, ammoniums__carboxylates turns the wrong way,
  water is already eclipsed; the turners are credit walkers.  Dead on
  the train gate.
- Docking victims (cycle 70 vs 65, persistent in cycle 101):
  amides__pyridine valid 12 → 29, amines__water 16 → 32, benzene__phenol
  15 → 31, esters__esters train 18 → 25, alkenes__esters, acids__amines,
  alcohols__pyridine, acids__benzene, alkenes__alkenes — ≈ +0.003 train /
  +0.0045 valid in total; no start-side separator (surrogate energy,
  surrogate gain, contact bins) is consistent across the splits, as the
  cycle-88/102 closures said.  Left alone.
- Reference-bin structure of the champion: connected same-basin rel
  0.755–0.78 for ref < 16, 0.70–0.72 for 16–30, 0.65 for ≥ 30 (ours ≈
  7.4 / 10.5 / 12.4 / 16.9 / 26 calls against 9.6 / 13.8 / 17.6 / 23.6 /
  40); a uniform saving of one call is worth ≈ −0.045 (connected −0.028,
  dimers −0.017).
- Small inorganic molecules (ref ≤ 12) sit at 4–7 calls; the residual
  model errors of CI₄ / BF₄⁻ / SiBr₄ (t₂ stretches 1.1–1.4× stiff) are
  stretch–stretch coupling, not diagonal, and the symmetric starts do
  not excite them; BBr₃'s out-of-plane mode (84 vs 372 cm⁻¹) is not
  excited by a planar start.  Nothing to gain.
- Model-frequency table of the Almlöf stretch guess against the
  experimental harmonic force constants (`/tmp/kmodel104.py`, model /
  experiment): C–H 0.97–1.07, C–C 1.20, C=C 1.17, C=O 0.99, C–O 1.06,
  C–N 1.10, C≡N 0.97, C–Cl 0.96 (with the row factor) — the hydrocarbon
  and first-row heavy bonds are calibrated; **N–H 0.82–0.87, O–H
  0.67–0.74, F–H 0.50, S–H 0.81, H–Cl 0.76, C–F 0.80** are the too-soft
  classes.  (The cycle-84 survey saw the O–H at 0.75× and set it aside
  by reasoning — "stiff modes are learned first" — not by measurement;
  the cycle-88 bond-length regression pooled all X–H bonds and was
  null.)
- Connected starts are thermal snapshots: X–H bond errors against the
  xTB endpoints have rms 0.025–0.027 Å for C–H (5101 bonds), N–H (406),
  O–H (181) and S–H (29) alike, per-molecule maxima median 0.044 Å,
  90th percentile 0.084 Å (`/tmp/xh104.py`).
- The class-resolved cost regression (`/tmp/stretch104.py`, connected
  same-basin runs of cycle 101, ours = a + b·heavy + c·rmsd₀ +
  indicators): a molecule whose worst **N–H / O–H** start error exceeds
  0.04 Å costs **+2.9 calls (t 3.1) on train and +3.3 (t 3.8) on valid**
  beyond the fit (28 / 30 molecules, ours 18.7 vs ref 26.9), while the
  same indicator for C–H (≥ 0.05 Å, 143 molecules) carries −0.05 / +0.31
  and for heavy–heavy bonds +0.7 / +1.3 (t ≤ 1.3).  Cross-tab: C-H-hot
  molecules without polar bonds +0.9, with a polar error ≥ 0.04 Å +2.7;
  C-H-cold with a polar error ≥ 0.04 Å +3.3.  The excess is specific to
  the two bond classes whose guess is 1.2–1.4× too soft.

Mechanism.  A too-soft stretch guess is the expensive sign of a
calibration error: the quasi-Newton step along the bond is g/k₀ =
(k/k₀)·Δr, so the error contracts by |1 − k/k₀| per step — 0.4–0.5 for
O–H, 0.15–0.2 for N–H — instead of ≤ 0.1 for a calibrated C–H, and it
alternates sign, so the TS-BFGS update (whose |B| metric weights the
stiff components by k·s²) has to learn it from steps that the 10–20 C–H
bonds of the molecule dominate at first.  The max-force criterion needs
the O–H / N–H bonds inside 4.5e-4/k ≈ 5e-4 Å of equilibrium, the tightest
precision of any coordinate, so a 0.05–0.09 Å start error on such a bond
needs a 100–200× contraction: 2 steps at |r| 0.1, 4–6 at |r| 0.4 unless
the update learns the bond first.  An over-stiff guess (C–C 1.2×) is on
the free side (monotone contraction 0.17 per step).  This is the
cycle-64 mechanism (heavy-row stretch factors, −0.0082 train with every
control bit-identical) applied to the largest remaining stretch class:
N–H / O–H bonds occur in 265 of the 483 connected same-basin molecules.

Change.  One factor per heavy element of the hydride bond, fixed a
priori from the class means of the textbook harmonic force constants
(inverse of the model / experiment ratio, rounded to 0.05): N 1.2, O 1.4,
F 2.0, S 1.25, Cl 1.3 (S and Cl on top of the row factor 0.58, net 0.725
/ 0.754), applied in `_h0_bond` for connected systems only (the
cycle-64 scope; dimer fragments start from optimised templates and are
bound by their intermolecular endgame — a fragment extension is the
follow-up if this pays, as in cycle 92).  Ammonium N–H (5.8 mdyn/Å)
becomes 1.2× over-stiff, the harmless side; intramolecular H-bond donors
(k lower by 10–25 %) stay within 1.25× of the guess.  C–F (0.80) is the
same physics but a heavy–heavy class with a smaller correction; parked
as a bundle item, not added, to keep the attribution clean.

Prediction: −2 calls on the ≥ 0.04 Å polar-error molecules (≈ 28 per
split), −0.5 on the 0.02–0.04 Å ones (≈ 60 per split), nothing on
molecules without N–H / O–H bonds → ≈ −0.004 / −0.004 per split.
Controls: every multi-fragment system (bit-identical, the dimer branch
of `_h0_bond` is untouched) and every connected molecule without an
N–H / O–H / S–H / F–H / H–Cl bond must be bit-identical in calls and
energy.  Energy gate: connected-only change, dimer credits untouched;
basin changes possible on the touched connected molecules only.
Constants are physical calibrations fixed a priori, one evaluation, no
scan.

### Change (functions)
New module table `_HYDRIDE_SCALE` (N 1.2, O 1.4, F 2.0, S 1.25, Cl 1.3,
comment with the force-constant table and the census); `_h0_bond`
multiplies the connected-system stretch guess of a bond between a
hydrogen and a heavier atom by the heavy atom's hydride factor after the
row factors.  Static check (`Internals` on saved geometries, no PES):
connected 135122079 O–H 1.4 / all other classes 1.0, 135186987 N–H 1.2 /
S–H 0.725 / C–S 0.58, amides__water dimer all 1.0.

### Result
Train evaluation `4680be5dadc743dda3d581b7e40ba4f2`
([JSON](evaluation_results/cycle-104-train.json)): 469/469 converged,
**C_S = 0.6510486** (champion 0.6512909, Δ −0.0002423, train gate
passed), mean rel_energy 1.0034456.  Valid evaluation
`98b1f6bbd9c046359bacbab54b4f8a83`
([JSON](evaluation_results/cycle-104-valid.json)): 465/465 converged,
**C_S = 0.6522417** (champion 0.6499489, Δ **+0.0022928**), mean
rel_energy 1.0079088.  Decision **non_generalizable**.  Implementation
preserved in `ideas/hydride-stretch-factors/4d760937b87265f9a0cdb0761e9c16056cf6f87c/algo.py`;
champion cf2439f restored in `21c3497`.
- Controls held: every multi-fragment system and every connected
  molecule without a polar X–H bond bit-identical (353 train / 326
  valid), except the in-contact DES370K complexes that Sella's 1.25 r_cov
  bonding rule treats as connected (their donor O–H / N–H got the
  factor): train ketones__phenol 28 → 21 (ref 43), monoatomics__phenol
  25 → 28, pyridine__pyrrole 21 → 20, carboxylates__thiols 30 → 28;
  valid esters__phenol 19 → 15, ammoniums__ketones 22 → 30,
  acids__alcohols 20 → 29, carboxylates__water 20 → 28, acids__pyrrole
  19 → 18, monoatomics__thiols 22 → 20.
- Touched connected set (`/tmp/cmp104*.py`): train 111 molecules Σ +4
  calls (17 better : 11 worse : 83 unchanged), valid 132 molecules Σ +19
  (12 : 19 : 101).  **184 of the 243 touched molecules (76 %) keep
  their exact call count**; the changes are ±1–3 re-rolls plus two
  outliers (172113611 35 → 45 train, 135065494 45 → 52 valid).  One
  train basin change, favourable (134997543 to the reference basin,
  28 → 31 calls — the same molecule as in cycle 103).
- The predicted class did not respond: molecules whose worst N–H / O–H
  start error is ≥ 0.04 Å went Σ +4 on train (31 molecules, 9 : 5 : 17)
  and Σ +7 on valid (34, 8 : 7 : 19); the 0.02–0.04 Å class −2 / +8, the
  < 0.02 Å class +2 / +4.  Nothing correlates with the error size
  (135122079 O–H 0.065 Å: 27 → 27; 134979308 0.077 Å: 15 → 16;
  160850312 0.071 Å: 22 → 21).
- Class read-out over both splits: single polar bond −9 over 84 (7 : 2),
  2–3 polar bonds +20 over 110 (12 : 21), ≥ 4 +12 over 49 (10 : 7);
  O–H-only −4 / 47, N–H-only −1 / 154, both classes +25 / 32; H-bonded
  donors +24 / 116 vs −1 / 127 without — all inside the re-roll noise
  except the sign of the strong-donor in-contact complexes.
- Train score: the −0.00024 is the ketones__phenol in-contact complex
  (−7 calls at ref 43 = −0.00035) plus a null remainder; the valid
  +0.0023 is +19 touched calls plus +25 on the three strong-donor
  in-contact complexes (ammoniums__ketones, acids__alcohols,
  carboxylates__water: N–H⁺ / O–H donors in strong hydrogen bonds,
  whose true stretch constant is 30–50 % below the free value, so the
  ×1.2–1.4 made them 1.7–2× over-stiff).

### Interpretation
- The hydride stretch guess is not a binding constraint of the
  connected endgame: making the N–H / O–H diagonal physically right
  (1.2–1.4× stiffer) leaves three quarters of the affected molecules at
  the same call count and re-rolls the rest.  The cycle-84 argument is
  confirmed by measurement — the too-soft stiff bond overshoots on the
  first step, which makes it the dominant component of the first
  secant pair in the |B| metric, and the TS-BFGS update corrects it
  before the endgame; the max-force precision (5e-4 Å) is then reached
  in the same steps the rest of the molecule needs.  The cycle-64 row
  factors paid because their errors were 1.6–5× *and* the molecules
  were tiny (4–6 atoms, nothing else to learn); a 1.2–1.4× error on a
  bond class inside a 20–40-atom molecule is learned for free.
- The class signal that motivated the cycle (+2.9 / +3.3 calls for a
  polar-bond start error ≥ 0.04 Å, absent for C–H) is therefore not a
  stretch-contraction cost.  The polar X–H error is a marker of a
  thermally hot polar group: the same snapshot that stretched the O–H /
  N–H bond displaced its torsion and wag, and those are the soft modes
  the endgame pays for (X–OH rotor 1.96 / 4.09 calls per turner in the
  cycle-88 regression; hot C–H groups have no soft mode attached).  The
  marker is real, the lever is not the diagonal.
- Strong-donor in-contact complexes (the 1.25 r_cov "connected"
  H-bonded dimers) respond with the sign of cycle 61: a stiffer donor
  stretch costs +8 / +9 calls on the strong donors (ammonium, acid,
  carboxylate–water) and pays −4 / −7 on the weak ones (phenol) — the
  shared-proton endgame does depend on the donor stretch, but the class
  is 4–6 complexes per split with opposite signs, exactly the
  cycle-61 null.
- Alternatives considered: (a) too small a factor — the O–H factor 1.4
  is the full textbook correction and the response is flat in the
  error size, so no; (b) the update fighting the stiffer guess (a
  rank-2 correction meant for another coordinate landing on the
  now-heavier polar components) — possible for the 2–3-polar-bond class
  (+20 over 110) but indistinguishable from the re-roll noise; (c) a
  fragment extension — the dimer fragments start from optimised
  templates and the connected population already says the diagonal is
  free, so no census needed.  No bounded repair: the constants are
  fixed calibrations and a sub-class split (free vs H-bonded donors)
  would be a scan on a null.
- Uncertainty: one evaluation per split; the touched-set sums (+4 /
  +19) are inside the ±0.005 path-noise floor, and the two outliers are
  ordinary re-rolls (172113611 has re-rolled ±4–10 in cycles 103 and
  104 alike).  What is certain: the train pass was one in-contact
  complex, and the valid gate refused it.

### Next idea / control
**Stretch-guess calibration closed for every bond class**: first-row
heavy bonds within 10 % (cycle-64 table), heavy rows calibrated (64),
boron / Si small classes parked (66), hydrides null (104); the only
remaining ratio outside 15 % is C–F (0.80), a heavy–heavy class with a
smaller correction than the null O–H — not worth a cycle, bundle-only.
The polar-start-error marker (+3 calls) points at hot polar rotors /
wags, i.e. the soft-mode learning floor again; the guess-Hessian line
(stretches, bends, torsions, linear bends, impropers, ring puckers,
dummies) is now closed in every direction.  The next cycle needs a
mechanism that changes how many calls a soft direction costs to learn
(update, step composition or start construction), static-checked
against the cycle-103/104 closure lists before a cycle is spent.

## Cycle 105 — contact-block predictor-corrector step for complexes of neutral closed-shell fragments (the cycle-79 curved step applied to the inter-fragment block of multi-fragment systems whose fragments all carry zero formal charge; charged, bare-ion and odd-electron complexes keep the quadratic step)

Champion at start: `cf2439f9ee81bc0e65f361fad971390d682ad652` (train 0.6512909 /
valid 0.6499489; energy margins Σ(r−1) +1.581 / +3.678). Candidate:
`5f5145d362f2eda879ab20cebde4bddbf2f4bdb9`.

### Hypothesis
Idea search (saved data only, cycle-101 endpoints; scripts `/tmp/tight105.py`,
`/tmp/small105.py`, `/tmp/fragq105.py`, `/tmp/bins104.py`):
- Static closures of this search.  (a) The trust region is a max-norm over
  internal components (`MaxInternalStep.cons`: `max|s_i w_i|`), so the
  redundant dihedral count of a rotor does not throttle its turn — no
  redundancy lever.  (b) Endpoint contact-tightness census of the same-basin
  two-fragment dimers (closest inter-fragment pair / r_cov sum): tight
  contacts (< 1.6) are 19 train / 17 valid runs, nearly all charged credit
  walkers; the H-bond bin 1.6–2.2 costs 15.7–15.9 calls against 30–39 for
  the reference and the vdW bin > 2.2 costs 14.4–15.2 against 45–50 — the
  dimer endgame is ≈ 15 calls in every tightness, charge and docking class;
  there is no "connected treatment for tight contacts" population.  (c) No
  ≥ 3-fragment system exists (0 / 0), so nothing multi-body is at stake.
  (d) The low-reference connected bin is tiny inorganic species with
  distorted PubChem starts (E_start − E_min up to 290 kcal/mol; SiS₂ 9 / 9,
  FNO 9 / 9, 135158617 13 / 13, 135097481 27 / 30) — the cycle-103 closure.
  (e) Learning-floor assessment: in the soft subspace a secant pair fixes
  one direction per call and the multi-secant memory (optimum ≤ 4, cycles
  47 / 56) already keeps the last four exact; finite-difference sampling of
  soft modes (each sample costs a call and reads a stale/negative curvature
  on anharmonic rotors), local-symmetry images (no soft-mode information),
  per-type scale learning (30 / 31 / 58), anion docking (EV ≈ −0.001 with
  the valid gate at stake), bond/angle start pre-correction (deviations do
  not predict calls), energy-augmented secants (16 / 17 / 50 / 51), GPR /
  GEDIIS-type accelerators (87: firing costs a settling step) are all
  closed with their reasons.  The connected floor (6.2 + 1.25 heavy/10 +
  12.6 rmsd₀ calls) is the energy criterion: |ΔE| < 1e-6 Ha between calls
  needs Σ k δ² / 2 < 1e-6, i.e. an rms residual ≈ 2.6e-4 Å for a 60-mode
  molecule, ≈ 7× tighter than the force and displacement criteria, so the
  endgame is 2.3 decades of contraction at the rate set by the guess error
  in the directions the secant pairs have not sampled (±15–35 % per stiff
  mode, calibrations closed as a family) — no lever there either.
- Remaining concrete positive-expectation items are the charge-class scopes
  of two discarded multi-fragment candidates.  Decomposed by the docking's
  own fragment charge classes (`/tmp/fragq105.py`: neutral 133 train / 130
  valid complexes, polyatomic cations 47 / 45, anions 15 / 15, bare ions
  15 / 12, ion pairs 2 / 1):
  * cycle 78's inter-fragment curved step (78 relative to 79): neutral
    same-basin −27 calls over 125 runs, 40 better : 29 worse (−0.0010),
    neutral basin changers 8 runs +2 calls; the loss sat entirely in the
    charged classes — anion basin changers +54 calls (+0.0064), cation
    same-basin 12 : 17 (+0.0006 through their small references), bare ions
    ±0;
  * cycle 84's class-resolved trust region (84 vs 81): neutral same-basin
    −9 calls, 38 : 33 (−0.0002), changers −0.0004; charged +0.0096 — a
    wash on the neutral class, not worth re-running.
- Mechanism of the scope (not credit protection): the tracked inter-fragment
  block A(x) is the Lindh-type contact model (exponential pair curvature
  plus the H-bond / lone-pair-plane bends); the path-averaged step
  s = −(H + ⟨A(x(t)) − A(x)⟩)⁻¹ g is the step of that model.  For the
  docked neutral complexes the walk starts in contact, the steps close or
  open a contact by 0.1–0.25 Å (curvature ×1.4–2.5 along the step) and the
  average is the curvature the step actually meets.  For the charged
  complexes the inter-fragment surface is a Coulomb approach that the
  contact block does not represent; correcting the rigid-body approach
  steps ×0.7–1.9 at the 0.25 cap re-routed them between basins 1–3
  kcal/mol apart (cycle 78: +54 calls on five anion complexes, −0.129 of
  energy credit).  The class test is the docking's own formal-charge rule
  (`_dock_formal_charges` on the start bond graph): every fragment net 0
  and even electron count.
- Credit at stake (a-priori check, cycle-101 endpoints): the neutral class
  carries Σ(r−1) +0.307 train (28 runs with |r−1| > 1e-3, negatives
  −0.549) and −0.676 valid (45 runs, negatives −2.345); the margins
  +1.581 / +3.678 rest on the untouched charged classes (bare ions +0.956 /
  +0.865, anions +0.035 / +3.346, cations +0.255 / +0.310) — a full loss
  of the neutral credits would leave both margins positive.  Charged
  complexes, connected systems and the trust policy are bit-identical by
  construction.
- Expected: −0.0005 … −0.0010 train (the cycle-78 neutral same-basin gain
  after the shorter post-docking walks of cycles 90 / 92 / 95), the same
  order on valid; risk: the neutral basin lottery (±0.001) on either split.

### Change (functions)
- New `_fragments_formally_neutral(numbers, pos_ang)`: True when the start
  has ≥ 2 fragments (`_start_fragments`) and every fragment has zero net
  formal charge (`_dock_formal_charges` on the 1.25 r_cov bond graph) and an
  even electron count; False for connected starts.
- `Sella.__init__`: new kwarg `curved_step_fragments=False`, stored on the
  optimizer.
- `Sella._curved_step`: the gate `self._has_tr_internals()` becomes
  `self._has_tr_internals() and not self.curved_step_fragments`; the
  inter-fragment block of the tracked contact model (already part of
  `_h0_nonlocal_contacts`, `_contact_model_ahead` realises fragment
  translation/rotation steps to first order — the cycle-78 code path,
  untouched since) enters the path average for these systems; docstring and
  the `curved_step_contact` comment updated.
- `minimize_func`: passes `curved_step_fragments=_fragments_formally_neutral(
  atoms.numbers, atoms.get_positions())` after the docking.
Static check: the helper reproduces the census classes exactly on both
splits after the start rules and docking (133 / 130 neutral True, every
other class and all 257 / 262 connected starts False); the diff since
cycle 78 touches no line of `_contact_model_ahead`, the shadow internals or
the fragment-coordinate handling.

### Result
- Train (`evaluation_results/cycle-105-train.json`, evaluation
  91b80c41a49c497c88a4924c3a560fc8, candidate 5f5145d): **0.6518435** vs
  champion 0.6512909 (Δ +0.0005526), mean rel_energy 1.0033719 (champion
  1.0033716), 469/469 converged, 56 s. **Discard** (train gate). Valid not
  run.
- Scope check (`/tmp/cmp105.py`): every connected (257), cation (47), anion
  (15), bare-ion (15) and ion-pair (2) run bit-identical in calls and
  energy; only the 133 neutral complexes moved.
- Neutral complexes: same-basin 127 runs Σ −9 calls (43 better : 34 worse),
  6 basin changes Σ −2 calls (credit +0.0001); mean calls 16.04 → 15.95.
  By champion call count: [0,12) +5 (8 : 11), [12,16) −1 (8 : 6), [16,22)
  −18 (17 : 8), [22,30) −1 (9 : 8), ≥ 30 +6 (1 : 1).  By reference:
  ref < 20 +10 calls (+0.0013, 2 : 6), 20–30 +4 (+0.0003, 4 : 8), 30–45 −4
  (−0.0003, 17 : 10), ≥ 45 −19 (−0.0006, 20 : 10).  Largest moves:
  acids__esters 27 → 32 (ref 14), acids__ketones 36 → 45 (ref 38),
  acids__benzene 29 → 22 (ref 54, −0.44 kcal/mol), amides__esters 21 → 17,
  alcohols__amides 21 → 17, alkenes__water 23 → 27.

### Interpretation
- The mechanism behaves on the neutral complexes exactly as it did on
  connected systems in cycle 79 — the long walks gain (16–22 champion
  calls: −18, 17 : 8) and the near-equilibrium runs lose ±1–2 (≤ 12 calls:
  +5, 8 : 11) — and the net is a small saving (−0.07 calls per complex).
  The metric weights it the other way round: the short neutral runs are
  the ones with small references (ref < 30: +14 calls = +0.0016) while the
  long walks sit on references ≥ 30 (−23 calls = −0.0009).  On connected
  systems the short runs had references of 9–14 too, but there the long
  walks (rmsd₀ ≥ 0.7 Å) gained 8–17 calls each; on docked complexes the
  walks are short (docking removed the approach phase, cycles 70 / 90) and
  the per-run gain is 1–4 calls, not enough to carry the near-equilibrium
  losses at their small references.
- The a-priori estimate (−0.0005 … −0.0010 from the cycle-78 neutral
  same-basin bins) missed the reference weighting: cycle 78's neutral
  gains were spread over the reference bins (ref < 20 −3, ≥ 45 −18); after
  the docking changes the short runs are shorter still and their losses
  land where each call is worth 1/10 – 1/20 of the score.
- No bounded repair: the only discriminator left is the step length (the
  correction is significant when a contact changes by ≥ 0.1 Å along the
  step), and a threshold on it would be a scan on a ±0.001 signal against
  the fixed-constant discipline of this line; the same-basin saving is
  real but below the noise floor in the metric's weighting.  The
  multi-fragment curved step is closed in both scopes (78 all complexes,
  105 neutral only); the connected scope stays as kept in cycle 79.
- Uncertainty: single train evaluation; the 43 : 34 split and the bin
  pattern are consistent with cycle 78's read-out, so the sign of the
  same-basin effect (a small saving) is replicated, and its metric sign
  (a loss) follows from the reference weighting, not from noise.
- Champion restored (`git checkout cf2439f -- algo.py`, restore commit
  60b5785); implementation preserved in
  `ideas/curved-step-analytic-model/5f5145d362f2eda879ab20cebde4bddbf2f4bdb9/algo.py`;
  backlog entry `curved-step-analytic-model` updated (multi-fragment scope
  closed in both variants).

### Next idea / control
Both charge-class scopes of the discarded multi-fragment candidates are
now settled (84's class-resolved trust region: a wash on the neutral class
in its own data, not worth a run; 78's curved step: cycle 105).  With the
guess-Hessian line closed in every direction (cycle 104) and the step
composition closed for complexes, the remaining unexplored territory is
the *reference weighting* itself: the score is mean(n_ours / n_ref), so a
call saved on a ref-10 run is worth five on a ref-50 run, and the
short-reference population (ref ≤ 20 carries 0.37 of the score at ratio
0.84–0.90) is where the champion is weakest — near-equilibrium connected
molecules and the fast neutral complexes.  Their cost is the endgame
contraction rate under the energy criterion (2.3 decades at the
unsampled-direction guess error).  A next cycle should look for a
mechanism that shortens *specifically* the 6–10-call runs (start
construction from the model itself — e.g. one Newton step of the analytic
model before the first call is not possible without a gradient — or the
final-step accuracy of the quasi-Newton model at the sub-1e-3 Å scale),
static-checked against the cycle-103/104/105 closure lists.

## Cycle 106 — local (Morse) curvature–length slope for the tracked stretch part of the model (the transported stretch diagonal of connected systems moves from the start-geometry guess with exp(−3Bb/2 (r − r0)) = 5.5 /Å instead of the Almlöf cross-bond exponential Bb = 3.7 /Å; ion–ligand bonds, dummy bonds and multi-fragment systems unchanged)

Champion at start: `cf2439f9ee81bc0e65f361fad971390d682ad652` (train 0.6512909 /
valid 0.6499489; energy margins Σ(r−1) +1.581 / +3.678). Candidate:
`cefa1fcc68f29de64c681240c7c894255ac47e83`.

### Hypothesis
Idea search (three sessions, saved data only, cycle-101 endpoints; scripts
`/tmp/small105.py`, `/tmp/midbin106.py`, `/tmp/bondanh53.py cycle-101`,
`/tmp/anion106.py`, `/tmp/lowref106.py`, `/tmp/walk106.py`, `/tmp/tail106.py`):
- Source: Birkholz & Schlegel, *Theor. Chem. Acc.* 135:84 (2016),
  https://link.springer.com/article/10.1007/s00214-016-1847-3 (PDF
  https://schlegelgroup.wayne.edu/Pub_folder/390.pdf): the flowchart
  (Bofill / SR1 / PSB / BFGS) update, the scaled RFO step and the
  quasi-rotation propagation of the Hessian along the path.  All three are
  covered by existing closures (update formula 5 / 33 / 58 / 95 / 98; RFO
  stepper 76; the analytic-model transport of cycles 53 / 55 is the
  propagation this code needs) — nothing adaptable beyond what is in place.
- Score decomposition (cycle-101 results): the ref 13–20 bin holds 140 / 135
  molecules, 0.243 / 0.238 of the score, ours 13.2 / 13.4 calls at ratio
  0.81 / 0.82; one call on each of them is worth 0.019.  Inside it the
  connected molecules (104 / 99) run at 12.05 / 12.34 calls (ratio 0.753 /
  0.734, histogram peaked at 10–13, 6 / 3 at ratio ≥ 1), the neutral
  complexes at 13.9 / 14.3 (0.88) and the charged complexes at 17.6 / 18.0
  (1.02 / 1.18, 13 / 7 at ratio ≥ 1 — credit walkers and the
  symmetry-breaking dimers of the energy-gate lottery).  Same-minimum
  endpoints across the last three champions differ by 1e-4–3e-4 Å rmsd and
  ≈ 4e-9 Ha, so the displacement / force criteria bind, not the energy one;
  a run is a quantised contraction sequence s_0 e^k < 1.8e-3 Å with the
  effective e ≈ 0.55 of a 13-call run set by the guess error along the
  directions the secants have not sampled.
- Static closures (no cycle spent): endgame-only SR1 (≈ neutral by the
  cycle-95 attribution algebra), the 64 + 65 fragment bundle (EV ≈ 0 with
  the charged re-roll risk), ρ_post pair filter (no mechanism),
  H- vs heavy-terminated linear-bend constants (re-tread of cycle 67),
  memory 2–3 and scale-aware retention (≤ 0.3 %), Lindh full-model guess,
  stretch–bend / bend–bend coupling models, light-rotor cap relief,
  Cartesian-bound H exemption, trust-growth variants, cross-molecule online
  learning (order-dependent), self-instantiated calculators and engineered
  failures for trajectories (not passive), stiff-model early stopping
  (gaming, and the credit walkers carry the margins), reference weighting
  by molecule class (no mechanism that saves one call on every 12-call
  run), probe steps, flat-mode creep, low-ref dimer share, the 19
  'connected' DES370K pairs, rebuild carry-over population 4 / 1,
  back-transformation drift ≈ 1e-5 Å, stale chord cost, endgame-only
  diagonal read-out, surrogate Hessians, O1NumHess / GPR optimisers (need
  more calls than the reference), the anion / ion-pair docking population
  (3 / 5 and 2 / 1), the sinc-model cap, the trust-policy history.
- The one census signal with a mechanism behind it is the residual bond
  anharmonicity cost.  On the champion (`/tmp/bondanh53.py cycle-101`,
  connected same-endpoint 240 / 249 molecules, baseline calls = 7.2 + 0.49
  n20 + 5.1 maxdisp + 0.035 N train / 9.6 + 0.34 n20 + 3.9 maxdisp valid)
  every bond changing by > 0.1 Å start → end still costs **+0.92 (se 0.21)
  train / +0.62 (se 0.25) valid calls** (was +1.0 / +0.8 at cycle 52, before
  the transport), the largest lengthening +3.2 (se 1.4) / **+8.4 (se 1.7)
  calls per Å**, and the molecules with ≥ 2 such bonds sit +2.6 (se 0.8)
  train (n 27) / +1.0 (se 0.8) valid (n 31) above the baseline; 90 / 109
  molecules have at least one, Σ n10 = 167 / 181 bonds.  The tiny
  hydrogen-free inorganic molecules (36 train, PubChem starts with bonds
  0.1–0.5 Å off, E_start − E_min up to 230 kcal/mol; SiS₂ 9 / 9, FNO 9 / 9,
  SiO₂ 8 calls for three atoms) are the extreme of the same population and
  the one that responded to every stretch-model change (cycle 53 Σ −17
  calls, 57 Σ +7, 64 Σ −34).
- Mechanism (a-priori, no constant fitted): the Fischer–Almlöf exponential
  k = 0.36 exp(−1.944 (r − r_ref)) is a fit of *equilibrium* force constants
  against *equilibrium* lengths across bond types — Badger's rule, whose
  slope 3.7 /Å is the cross-bond one.  The curvature of *one* bond
  displaced from its minimum follows the Morse relation k = k_e (2e^{−2aΔ}
  − e^{−aΔ}) with d ln k/dr = −3a at the minimum (−3.6a at +0.1 Å, −2.7a at
  −0.1 Å) and a = √(k_e/2D_e) = 1.8–2.3 /Å for first-row bonds (C–H 1.9,
  C–C 2.0, C–O 2.1, N–H 2.2, O–H 2.3), 1.4–2.0 /Å for the heavier ones
  (Si–H 1.6, C–Cl 1.7, S–S 1.7, Cl–Cl 2.0): the local slope is 5.5–7 /Å,
  1.5–2× Badger's.  Cycle 53 chose Bb "as a conservative transport" and
  noted "Morse would give 3a ≈ 5–6 /Å".  With the cross-bond slope the
  transported pairs recover only ≈ 60 % of the curvature change along a
  relaxing bond: a bond lengthening by 0.13 Å from a compressed start
  (Morse 2.15 k_e at the start, 1.1 k_e at the end) leaves the transported
  secant 1.28 k_e against the true 1.10 (FA slope) versus 1.06 (local
  slope) — one contraction step more before the 1.8e-3 Å criterion,
  the +0.9 calls per such bond of the census.  Bb/2 = 0.97 /Bohr = 1.84 /Å
  lies in the middle of the Morse exponents above, so the local slope is
  taken as 3Bb/2 = 2.92 /Bohr = 5.5 /Å: no new constant.  The tracked
  stretch value of a covalent bond of a connected system becomes k(r0)
  exp(−3Bb/2 (r − r0)) with r0 the length at the anchor geometry (the
  positions at which the guess was evaluated — the start, or the rebuild
  geometry after a coordinate rebuild), so the model equals the guess
  there and only its motion along the run changes; the guess itself (the
  cross-bond relation, the right one for an unknown bond type at an
  unknown displacement) is untouched.  The pure exponential is preferred
  over the anchored Morse form: for a start displaced by ±0.1 Å both are
  within 3 % of each other (0.577 / 1.73 against 0.55 / 1.70 for the true
  0.57 / 1.96), and the exponential stays positive on any path while the
  Morse ratio vanishes at Δ = ln 2/a = 0.38 Å.
- Scope: covalent bonds (no dummy atom, no s-block metal: the ion–ligand
  "bonds" of the cycle-52/81 model are electrostatic, k ~ r⁻⁴, local slope
  ≈ 2 /Å, and keep the Almlöf exponential) of connected systems; all
  multi-fragment systems bit-identical (cycle-55 transport unchanged) —
  the credit decomposition of cycle 105 stands, no dimer path moves, the
  energy margins are not at stake.  The curved step (cycle 79) uses the
  contact block only, so the stretch slope enters through the transport
  and the model shift alone; cycle 57's double application (transport and
  step) of the stretch exponential is not repeated.
- Falsification test (connected same-endpoint census on the candidate):
  (i) the n10 ≥ 2 bins (train +2.6 se 0.8, n 27; valid +1.0, n 31) and the
  maxlong coefficient (+3.2 / +8.4 calls per Å) must fall; (ii) the 38
  train molecules with maxdb < 0.05 Å must stay within ±1 call (no
  mechanism there: the shift differs by 0.5 Bb Δ ≈ 5 % of k); (iii) every
  multi-fragment system bit-identical; (iv) the tiny hydrogen-free set
  (36 train) is the sensitivity read-out — a gain there without (i) is a
  lottery, a loss there is the cycle-57 signature (the Almlöf k_e wrong
  for an exotic bond, its error now moved 1.5× faster).
- Expected: −0.002 … −0.005 train (a third to a half of the +0.92 calls
  per n10 bond, 167 bonds), −0.001 … −0.004 valid (181 bonds, larger
  lengthening cost); risk: connected path re-rolls at ±0.002 and the
  exotic-bond magnitude errors.

### Change (functions)
- New module constant `_STRETCH_LOCAL_SLOPE = 1.5 * 1.944` /Bohr with the
  derivation (Badger cross-bond slope vs the Morse one-bond slope 3a,
  a = Bb/2 in the middle of the first-row / heavy-row Morse exponents).
- `Internals.__init__`: `self._stretch_anchor` = the positions at
  construction (the geometry of the guess); `Internals.copy` and
  `Internals.shadow_copy` share it (the shadow copies used for the Simpson
  path average are created after the first step and must not re-anchor);
  a rebuilt coordinate set (`Sella.initialize_pes` after
  `check_for_bad_internals`) constructs a new `Internals` at the current
  geometry, where its guess is evaluated, so it anchors there.
- `Internals._h0_bond(bond, Ab, Bb, rij=None)`: optional explicit length.
- `Internals._h0_stretch_diagonal`: for connected systems (`ntrans +
  nrotations == 0`) every bond between two real atoms (numbers > 0) with
  `_ION_CHARGE == 0` on both gets `_h0_bond(bond, rij=r0) *
  exp(−_STRETCH_LOCAL_SLOPE (r − r0)/Bohr)`, r0 from the anchor positions;
  every other bond, every multi-fragment system and every guess
  (`guess_hessian` → `_h0_bond` at the current geometry) unchanged.
- Docstrings of `_h0_stretch_diagonal` and `_track_analytic_model`.
- Static checks: py_compile; pyflakes diff vs the champion = line numbers
  only; `/tmp/stub106.py` execs the real `_h0_bond` /
  `_h0_stretch_diagonal` source on a stub (no optimizer, no calculator):
  at the anchor the tracked diagonal equals the Almlöf guess exactly, a
  C–C bond lengthening by 0.13 Å moves to 0.4885 of its guess (FA would
  give 0.620), a Na–O bond keeps the Almlöf ratio, a dummy bond keeps it,
  a multi-fragment stub keeps it for every bond.

### Result
- Train (`evaluation_results/cycle-106-train.json`, evaluation
  fdd0157f9099415193ecfde29fc90bbe, candidate cefa1fc): **0.6569924** vs
  champion 0.6512909 (Δ +0.0057015), mean rel_energy 1.0033042 (champion
  1.0033716), 469/469 converged, 58 s. **Discard** (train gate). Valid not
  run. Champion restored (`git checkout cf2439f -- algo.py`, commit
  1175141); implementation preserved in
  `ideas/stretch-transport-local-slope/cefa1fcc68f29de64c681240c7c894255ac47e83/algo.py`.
- Scope check (`/tmp/cmp106.py`): every multi-fragment system bit-identical
  in calls; 5 of the 7 DES370K train systems that are single-fragment by
  the bond graph (connected in Sella's coordinates, no translations /
  rotations) moved: ketones__phenol 28 → 25, pyridine__pyrrole 21 → 15,
  monoatomics__phenol 25 → 26, guanidiniums__water 14 → 15,
  carboxylates__thiols 30 → 42 (Σ +9, +0.0003).
- Connected (250): all Σ +39 (47 : 39, +0.0054) = same-endpoint 246 Σ +4
  (46 : 37, +0.0017) + 4 basin changes Σ +35 (+0.0037: 135039840 SiOCl
  10 → 33, 104079126 51 → 65 into a minimum 1.7 kcal/mol lower along a
  1.43 Å bond change, 2 others ±).
- Falsification test: (i) **failed** — n10 ≥ 2: Σ +12 (9 : 12), of which
  H-containing −2 (6 : 4, n 20) and H-free +14 (3 : 8, n 18); H-containing
  lengthening ≥ 0.1 Å −2 (3 : 2, n 12), shortening ≥ 0.1 Å +2 (11 : 6,
  n 47); lengthening-only −5 (4 : 2), "both" +13 (1 : 6). (ii) passed —
  maxdb < 0.05 Å: +1 over 40 (5 : 5). (iii) passed — multi-fragment
  bit-identical. (iv) the sensitivity read-out is a **loss**: H-free
  N ≤ 6 +9 (4 : 9), H-free lengthening +10 (2 : 6), H-free shortening +13
  (5 : 9), the champion's ≤ 9-call runs +19 (4 : 12). Largest moves:
  135078067 (B–C, N 5, maxdb 0.27 Å, n10 5) 9 → 16, 135205152 (B–O) 9 → 13,
  134980158 (P–S) 8 → 11, 135095518 (H-free, 10 atoms) 23 → 19, 135094180
  21 → 17, 135264337 15 → 12; large organics ±2–5 both ways (384221749
  38 → 44, 172113611 35 → 40, paliperidone_palmitate 56 → 52, 315516336
  14 → 11).
- By champion call count (same-endpoint): ≤ 9 +19 (4 : 12), 10–13 0
  (8 : 6), 14–20 −8 (15 : 9), > 20 −7 (19 : 10); by reference: ≤ 12 +9
  (5 : 9), 13–20 +2 (15 : 11), > 20 −7 (26 : 17). H-containing overall −6
  over 180 (34 : 21, −0.0011); H-free +10 over 66 (12 : 16, +0.0028).

### Interpretation
- The hypothesis is refuted where it was testable and unconfirmed where it
  was expected to pay. The organic molecules with bonds relaxing by > 0.1 Å
  do not respond to a 1.5× steeper transport (−2 calls over 20 molecules),
  so the census residual of +0.9 calls per such bond is not a
  stretch-transport lag: it is a proxy for distorted PubChem starts whose
  cost sits elsewhere (bends, wags, walkers). The stretch part of the model
  is not binding for the organic bulk in either slope — the same
  conclusion as cycle 57 ("stretches converge in 2–3 steps whatever their
  first-order error; the soft modes set the count").
- On the hydrogen-free inorganic molecules the transport magnitude, not
  its slope, is the error: the Almlöf k_e of a boron, P–S or Si–O bond is
  wrong by 1.2–2× (cycle-66 census), the additive shift (T(x) − T_avg) s
  carries that error, and moving it 1.5× faster costs a contraction step
  or a basin (cycle 57's loss on the same molecules with the same
  mechanism applied to the step). Both slope directions have now been
  measured (cycle 53 Bb pays −17 calls on this set, cycle 106 3Bb/2 loses
  +10): the Almlöf slope is the better-behaved one for an additive
  transport with an uncertain k_e, and the residual on the exotic-bond
  molecules would need a magnitude-free (multiplicative) transport of the
  measured curvature, y_i ← y_i k_i(x)/k_i(avg) on the bond components —
  a population of 36 train / share 0.012 valid molecules, i.e. a
  small-class candidate that cannot pass the valid gate on its own
  (backlog entry; bundle-only).
- The five single-fragment DES370K systems are connected systems by
  construction (Sella's bond graph) and re-rolled ±1–12 calls like any
  connected walker; nothing to scope there.
- The organic side's −0.0011 (34 : 21) is inside the connected path-noise
  band (±0.002) and is not evidence for the mechanism; the basin changes
  (+0.0037) are the usual lottery of a path change on 250 molecules.

### Next idea / control
The stretch model is closed in every direction (guess magnitude 64 / 66 /
104, slope 53 / 106, curved step 57). The organic endgame is set by the soft
modes and the exotic-bond molecules by the guess magnitude of bonds no
table calibrates. The next cycle should not touch the analytic model; the
remaining unmeasured candidates are (a) the magnitude-free transport
bundled with a second small-class repair (the cycle-66 boron / Si constants
parked in `ideas/heavy-row-stiffness`), evaluated as one cycle only if a
third independent small-class item appears, and (b) mechanisms that change
how a soft direction is learned (update, step composition or start
construction), static-checked against the cycle-103…106 closure lists.

## Cycle 107 — energy-corrected secant pairs (Zhang–Deng–Chen modified quasi-Newton equation): the newest pair of connected systems is moved from the chord curvature to the endpoint-local curvature along its step with the energy change of the step, y ← y + [(6a + 8b − 6ΔE) − y·s − s·(T(x)s − T̄s)] s/|s|²; gated on a positive chord, 2 meV of curvature energy and a ≥ 5 % change, clipped to 0.4–3× the chord; multi-fragment systems untouched

Champion at start: `cf2439f9ee81bc0e65f361fad971390d682ad652` (train 0.6512909 /
valid 0.6499489; energy margins Σ(r−1) +1.581 / +3.678).

### Hypothesis
Idea search (saved data only, cycle-101 endpoints; `/tmp/map107.py`,
`/tmp/feat107.py`, `/tmp/feat107b.py`):
- Cost map (train): connected 229 molecules ref 20.3 / ours 14.4 (rel 0.736,
  share 0.3596; same-basin 216 at 0.729), connected-charged 21 (0.724),
  dimer-charged 75 (0.736; same-basin 51 at 0.650, different-basin 24 at
  0.917), dimer-neutral 137 (ref 42.2 / ours 15.9, rel 0.454).  Reference
  bins [0,10)…[60,400): rel 0.83 / 0.83 / 0.83 / 0.81 / 0.71 / 0.59 / 0.56 /
  0.42 / 0.30 — the cheap references are where every remaining call is
  worth most.  Same-basin losers 22 (excess 29 calls, share 0.0042);
  different-basin 87 (Σ(ours − ref) −1790, Σ(relE − 1) +1.575 — the
  walkers carry the energy margin and are not to be touched).  Valid:
  connected 0.705 (0.3472), dimer-charged 0.824 (different-basin 21 at
  1.327), dimer-neutral 0.448.
- Connected same-basin regressions: train (n 235) ours = 5.96 + 10.83
  rmsd₀ + 0.09 nrot + 0.49 n20 + 0.113 heavy (R² 0.657) against ref = 7.15 +
  15.64 rmsd₀ + 0.26 nrot + 0.78 n20 + 0.189 heavy; valid (n 238) ours =
  7.30 + 7.84 rmsd₀ + 0.21 nrot + 0.37 n20 + 0.098 heavy.  By start
  displacement: rmsd₀ [0.15, 0.25) 54 molecules 10.3 calls, [0.25, 0.40)
  42 → 13.1, [0.40, 0.70) 63 → 14.6, [0.70, 5) 42 → 24.0 (ref 34.3, n20 ≈
  4).  Log form ours = 18.45 + 6.04 ln rmsd₀ + 0.123 heavy: the approach
  phase costs ≈ 6 calls per e-fold of start displacement against ≈ 1.7
  calls per e-fold of the endgame contraction (e ≈ 0.55) — the
  large-amplitude anharmonic soft-mode paths (rotors, puckers, distorted
  PubChem starts) are the remaining lever of the connected share, and the
  endgame is at its learning floor (cycle 95).
- What the closures leave open: the *model-side* rotor curvature was
  tracked in both forms (cycle 68 local −h cos 3φ, discard +0.0024; cycle
  69 secant h sinc 3ψ, non_generalizable) and the lesson was "the local
  curvature of a cosine well is the wrong stiffness far from the minimum
  (zero at 30°, negative beyond)" — a model quantity applied along the
  whole approach; the *measured* pairs were never corrected.  Cycle 21
  diagnosed the mid-phase rotor steps: "their secant curvature is the
  chord of the well and therefore too soft — the shrink protected exactly
  these steps", and cycle 59 (retrospective ρ_post trust rule, discard)
  named the reopening condition "a rotor model whose secant curvature is
  not systematically soft (a chord-corrected torsional update)".  The
  energy of every force call is used only in the trust ratio; the update
  never sees it.
- Source: Zhang, Deng & Chen, *J. Optim. Theory Appl.* 102, 147–167
  (1999), https://link.springer.com/article/10.1023/A:1021898630001 (the
  "new quasi-Newton equation" B_{k+1} s = ỹ with ỹ = y + θ s/(sᵀs), θ =
  6(f_k − f_{k+1}) + 3 sᵀ(g_k + g_{k+1})); Zhang & Xu, *J. Comput. Appl.
  Math.* 137, 269–278 (2001),
  https://www.sciencedirect.com/science/article/pii/S0377042700007135
  (properties, higher-order accuracy); the Yuan / Wei–Li–Qi family with
  θ' = 2(f_k − f_{k+1}) + sᵀ(g_k + g_{k+1}) and its Taylor derivation read
  in Babaie-Kafaki, *Sci. China Math.* 54, 2019–2036 (2011),
  https://doi.org/10.1007/s11425-011-4232-7; Yabe & Takano, *Comput.
  Optim. Appl.* 28, 203–225 (2004),
  https://link.springer.com/article/10.1023/B:COAP.0000026885.81997.88
  (the parametrised form).  Verified by Taylor expansion at x_{k+1}
  (T₃ = sᵀ(∇³E s)s): sᵀy = sᵀGs − ½T₃ + O(s⁴), 6(f_k − f_{k+1}) +
  3sᵀ(g_k + g_{k+1}) = +½T₃ + O(s⁴), so sᵀỹ = sᵀ G(x_{k+1}) s + O(|s|⁴):
  the modified pair carries the *local* curvature at the new point along
  the step, the plain pair the *chord* (error −½T₃, always soft on the
  descending side of a well whose curvature rises toward the minimum).
  Equivalent statement: the cubic Hermite interpolant through E₀, E₁ and
  the two slopes a = φ′(0), a + 2b = φ′(1) has residual c = 12(a + b −
  ΔE) (the trapezoid error is c/12) and endpoint curvature φ″(1) = y·s +
  c/2 = 6a + 8b − 6ΔE; for a quadratic it reduces to y·s.  None of the
  geometry-optimisation codes checked (Sella, geomeTRIC, ASE, pysisyphus,
  the Birkholz–Schlegel flowchart of cycle 106) uses the energy in the
  update.
- Worked case (cosine rotor V = h/9 (1 − cos 3ψ), step from ψ = 30° to
  11° off the minimum): chord 0.46 h, endpoint-local (exact) 0.84 h, the
  Hermite value 0.88 h, the ideal next chord (to the minimum) 0.94 h; the
  chord model overshoots to −11.7° (ρ ≈ −0.06, trust shrink), the
  corrected one lands at −0.8° — one call per large rotor turn when the
  rotor binds, and the same for every other direction sampled by a step
  (puckers, wags, bond relaxations beyond the analytic model).  On the
  concave side (chord ≤ 0) nothing is done, so the cycle-68 failure mode
  (a zero or negative stiffness imposed 30° off the minimum) cannot occur.
- Frame effects (static): the slopes are taken without gradient transport
  — a = g(x₀)·dx_initial and φ′(1) = g(x₁)·dx_final are exact directional
  derivatives of the energy along the internal geodesic at its two ends
  (the internal gradient lies in range(B) at each point, so the
  off-manifold part of the finite difference does not enter), whereas
  Sella's transported g_par = B₁B₀⁺g₀ loses cos Δφ of a rotor gradient
  (a rigid rotation keeps the Cartesian vector fixed and re-projects it:
  y·s is softened by ½Δφ²·g₀·Δφ ≈ 1.5 Δφ² of itself — 6 % at 0.2 rad,
  37 % at the 0.5-rad cap).  Writing the correction as y·s → φ″(1)
  removes that transport softening along s as a by-product; the
  components of y orthogonal to s keep Sella's transport.
- Interaction with the analytic transport (cycles 53/55): the stored pair
  is used as y + (T(x)s − T̄s), the analytic part's own chord → local
  move along s; the target y_used·s = φ″(1) therefore subtracts s·(T(x)s −
  T̄s) from the stored correction (no double counting), and later re-uses
  of the pair move only the analytic part further.  Where the model's
  stretch magnitude is wrong (the cycle-106 exotic bonds) the pair now
  carries the *measured* local curvature along s instead of the model's
  shift — the opposite of cycle 106's failure (a model error moved
  faster).
- Gates, all a priori: connected systems only (`ntrans + nrotations ==
  0`; the dimer endgames need the credit decomposition before any change,
  cycle 105); the newest pair at creation only; effective chord
  y·s + s·(T s − T̄ s) > 2 meV (≈ 7e-5 Ha: a 0.1-rad step on a 0.2 eV/rad²
  mode; endgame steps of 0.01–0.03 are 10–100× below it → bit-identical
  endgames, energy noise ≤ 1e-6 Ha is < 1 % of the gate); relative change
  |φ″(1)/chord − 1| ≥ 0.05 (near-quadratic steps untouched); ratio clipped
  to [0.4, 3] (a positive pair; the Hermite value is an interpolant and
  is not trusted beyond 3× the chord).  No fitted constant.
- Falsification tests: (i) every multi-fragment system bit-identical in
  calls and energies; (ii) connected near-equilibrium runs (rmsd₀ < 0.15
  Å, 45 train molecules) bit-identical or ±1; (iii) the rmsd₀ ≥ 0.4 Å
  bins (105 train molecules at 14.6 / 24.0 calls) must carry the gain, the
  n20 ≥ 2 subset first; (iv) the tiny hydrogen-free set (36 train) is the
  sensitivity read-out again — a loss there means the interpolant is
  wrong on strongly anharmonic bond paths (Morse, 0.1–0.5 Å) and the
  clip is too wide; (v) the trust-ratio distribution is not observable
  (passive), but a saved-run comparison of call counts by rmsd₀ bin is.
- Expected: −0.3 … −0.5 calls on ≈ 150 large-rmsd₀ connected molecules
  ≈ −0.005 … −0.010 train, valid alike (same mechanism, connected share
  0.347); risk: path re-rolls at ±0.002, the interpolant on bond paths.

### Change (functions)
- Module constants `_ESEC_MIN_CURV = 2e-3` (eV), `_ESEC_MIN_CHANGE = 0.05`,
  `_ESEC_RATIO = (0.4, 3.0)` after `_SYM_MAX_NODES`.
- `PES.__init__`: `self._energy_secant = False` (the correction is off for
  the Cartesian PES); `InternalPES.__init__`: `self._energy_secant =
  (self.int.ntrans + self.int.nrotations) == 0` (connected systems only).
- `PES.kick`: the two exact slopes and the energy change of the realised
  step are collected, `slopes = (g0 @ dx_initial, g1 @ dx_final,
  f1 − f0)`, and passed to `_update_H`.
- `InternalPES._update_H(dx, dg, slopes=None)` passes `slopes` through to
  `PES._update_H(dx, dg, T_now, tbar, images, slopes)`.
- `PES._update_H(..., slopes=None)`: after the `last` check, `dg =
  self._energy_corrected_dg(dx, dg, T_now, tbar, slopes)` when
  `slopes is not None and self._energy_secant`; everything else (pair
  memory, transport, symmetry images, TS-BFGS) unchanged.
- New `PES._energy_corrected_dg(dx, dg, T_now, tbar, slopes)`: shift =
  s·(T_now s − t̄), chord = y·s + shift (return unchanged when chord ≤
  2 meV), c = 12(½(φ₀ + φ₁) − ΔE), local = (φ₁ − φ₀) + c/2, ratio =
  local/chord (unchanged when |ratio − 1| < 0.05), clipped to [0.4, 3],
  β = (ratio − 1)·chord/|s|², return y + β s.  All non-finite inputs return
  the pair unchanged.  Static check of the algebra on hand numbers
  (`/tmp/stub107.py`: quadratic → unchanged, cosine rotor → 0.88 h,
  concave → unchanged, clip both ends) passed; py_compile and pyflakes
  clean.

### Result
- Candidate `533da61`, train: fitness **0.6587707** vs champion 0.6512909
  (**+0.0074798**), 469/469 converged, mean rel_energy 1.0087495 (gate
  passed), evaluation `af9f4c1dfe5c4f8fbc4bb36cad5a5fdc`, 52 s.
  Decision: **discard** (train improvement below 1e-4).  No validation
  run.
- Read-out (`/tmp/cmp107.py`, paired against cycle 101):
  - all 212 multi-fragment systems bit-identical in calls and energies
    (falsification test i passed — the gate is exactly what it was meant
    to be);
  - connected: 257 molecules, 52 bit-identical, Σ +68 calls (59 : 53),
    +0.0075 = same-endpoint +0.0027 (250 molecules, Σ +41, 57 : 48) +
    basin changes +0.0048 (7 molecules Σ +27, Σ(relE − 1) +2.52);
  - by start displacement (same endpoint): rmsd₀ < 0.15 Å 34 molecules
    Σ +4 (2 : 6; test ii failed — not bit-identical), [0.15, 0.25) Σ +7
    (11 : 7), [0.25, 0.40) Σ −6 (8 : 5), [0.40, 0.70) Σ +8 (15 : 16),
    ≥ 0.70 Å Σ +28 (21 : 14) — the bin that had to carry the gain has
    more winners than losers but two fat tails (384221749 38 → 54,
    135316214 23 → 34) and the charged connected complexes (24 molecules,
    2 : 6, Σ +25) turn it into a loss;
  - hydrogen-free Σ +10 (11 : 15; test iv: mild), H-containing Σ +31
    (46 : 33); ref [13, 21) −7 (21 : 13), ref ≥ 21 +44;
  - basin changes: deeper minima found on 135094396 (relE 3.54, 8 → 19),
    104079126 (51 → 42) and 134997543 (28 → 31); shallower on 135039840
    (SiOCl, 10 → 32 — the same molecule derailed 10 → 33 in cycle 106),
    135097481 (0.969, 27 → 32) and carboxylates__thiols (30 → 23).
- Preserved: `ideas/energy-corrected-secant/533da61d99ad0d9f1a3dd79dc2aa5694bdddfcec/algo.py`;
  backlog entry `energy-corrected-secant` (status: repair pending).

### Interpretation
- The formula is right and the gating is right; the *attribution* is
  wrong.  y' = y + β s puts the curvature-energy residual of the step on
  the coordinates in proportion to s_i — in Sella's mixed-unit internal
  coordinates the torsions (radians, 0.1–0.5 per step) dominate |s|²
  while the bond stretches (Å, 0.01–0.05) dominate the residual on the
  early connected steps (the Almlöf model vs the Morse curvature of a
  compressed or stretched bond, the cycle-106 mechanism).  A 0.04 eV
  stretch residual over |s|² ≈ 0.115 gives β ≈ 0.35 eV/rad² per unit of
  s, i.e. +0.1 eV/rad on a torsional component whose true y is ≈ 0.09
  eV/rad: the torsional curvature of the pair is doubled while the stretch
  component is barely moved.  The over-stiff torsion then costs the
  rotor turns a shortened step (the cycle-68 lesson in the opposite
  direction), the multi-secant memory takes one or two steps to overrule
  it, and the trust-region history is re-rolled — the ±3-call wash with
  fat tails seen on the large-rmsd₀ bin, and the non-identical
  near-equilibrium runs whose first step is stretch-dominated (the 2 meV
  gate is passed by the bond components alone).
- Test iv (hydrogen-free, +10): the Hermite interpolant's own error is
  small on rotor paths (≤ 10 %, always toward the ideal next chord) and on
  Morse steps ≤ 0.15 Å (≤ 5 %), 20 % under for −0.3 → −0.05 Å and fails
  (ratio 0.11 → clipped 0.4) for −0.5 → −0.1 Å: the clip protects the
  update but a 0.4× softened bond pair is still a 2.5× error on a single
  stretch — consistent with a mild H-free loss and with 135039840's
  derailment, not the driver of the train result.
- Charged connected complexes (s-block / ammonium-type single fragments,
  24 molecules, Σ +25): their early steps mix ion–ligand stretches (soft,
  anharmonic, large residual) with ligand torsions — the attribution flaw
  at its worst.
- What was learned that the hypothesis did not contain: an
  energy-informed secant correction must be *distributed* across the
  coordinates, and the plain Zhang–Deng–Chen rank-1 form (u = s) is only
  valid in a metric where the step's components are commensurate — for
  internal coordinates the Zhang–Xu vector-parameter form with u = y
  (each component scaled by the common ratio, i.e. the residual
  attributed in proportion to the component's own curvature energy) is
  the coordinate-free statement of the same equation.  The dimer gate
  works exactly, the energy noise did not enter (52 bit-identical
  connected runs, every endgame untouched).

### Next idea / control
- Bounded repair (candidate 1 of ≤ 3): (A) proportional attribution
  u = y_used, y' = y + (ratio − 1)(y + t) with t = T(x)s − T̄s; plus (C)
  apply only when the bond components carry ≤ ½ of the effective chord
  curvature energy (`_ESEC_MAX_STIFF_SHARE = 0.5`, bonds slice of the
  coordinate list), so that stretch-dominated early and near-equilibrium
  steps are untouched and the correction acts where it was derived —
  on the soft-mode approach steps.  Same gates and clip; multi-fragment
  systems untouched.  Falsification: near-equilibrium runs bit-identical
  (the stretch share of their first steps exceeds ½), the rmsd₀ ≥ 0.4 Å
  bins carry the gain, charged connected complexes no longer lose.
- Control for the repair: if the same rmsd₀ ≥ 0.7 bin is again a wash,
  the mechanism (chord too soft on the approach steps) is not binding
  with the multi-secant memory of 4 pairs, and the line closes after one
  more variant at most (per-block ratios are not observable).

## Cycle 108 — bounded repair of the energy-corrected secant pair (cycle 107): the newest pair of connected systems is rescaled as used, y_u ← (local/chord) y_u with y_u = y + (T(x₁)s − T̄s) (Zhang–Xu vector-parameter form, u = y_u — the residual attributed in proportion to each component's curvature energy instead of its step length), and only on steps whose bond components carry ≤ ½ of the chord curvature energy; same slopes, gates (2 meV, 5 %, clip 0.4–3) and scope as cycle 107; multi-fragment systems untouched

### Hypothesis
- Diagnosed cause of the cycle-107 discard (+0.0075): the rank-1 form
  y' = y + β s attributes the curvature-energy residual of a step to the
  coordinates in proportion to s_i.  In Sella's mixed-unit internal
  coordinates the torsional components (rad, 0.1–0.5) dominate |s|² while
  the residual of the early connected steps is the sum of many small
  bond-stretch contributions (Å, 0.01–0.05 each; Almlöf transport slope
  3.7 /Å vs Morse 5.5 /Å, magnitude errors on exotic bonds), so the
  stretch residual was dumped on the torsions (β ≈ 0.35 for a 0.04 eV
  residual over |s|² 0.115: the torsional curvature of the pair
  doubled), the near-equilibrium runs were not bit-identical (their
  stretch-dominated first steps passed the 2 meV gate) and the charged
  connected complexes (ion–ligand stretches mixed with ligand torsions)
  lost 2 : 6.
- Repair, two parts, each aimed at that cause: (A) the attribution —
  the pair as used by the update, y_u = y + (T(x₁)s − T̄s), is rescaled
  by the common factor local/chord (Zhang & Xu 2001, the modified secant
  equation with the vector parameter u = y_u: ỹ = y_u + θ u/(sᵀu) = (1 +
  θ/chord) y_u), so every component keeps its share of the curvature
  energy of the step: the residual lands where the curvature energy is.
  This is the |B|-metric statement of the same equation (u = |B|s ≈ y_u
  for an accurate model), consistent with the TS-BFGS update's own
  metric and with the cycle-98 lesson that errors put on the stiff block
  are re-learned for free by the next stiff step while errors on the
  soft block persist; u = s is the identity-metric statement and puts
  the errors on the soft block.  The price: on a mixed step the soft
  modes are under-corrected by their share of the chord, hence (C).
  (C) the scope — the correction is applied only when the bond components
  (slice [ntrans, ntrans + nbonds) of the coordinate vector) carry at most
  `_ESEC_MAX_STIFF_SHARE = 0.5` of the chord curvature energy y_u·s, i.e.
  when the soft coordinates (angles, torsions, out-of-plane, dummies)
  dominate the step.  Stretch-dominated early steps and the
  near-equilibrium runs keep the plain pair (the analytic transport
  already carries the stretch curvature along them, cycle 53); the
  mid-phase rotor/pucker steps — the population the cycle-107 hypothesis
  was written for (cycle 21/59: "their secant curvature is the chord of
  the well and therefore too soft") — are corrected with the residual
  distributed over the soft block by curvature-energy share.
- Constants: `_ESEC_MAX_STIFF_SHARE = 0.5` (the majority rule, set a
  priori; not to be scanned), the cycle-107 gates unchanged (chord > 2
  meV, |ratio − 1| ≥ 0.05, ratio ∈ [0.4, 3]).  Static algebra check
  (`ast`-extracted function on hand numbers): quadratic unchanged; cubic
  Hermite case exact (y·s 1.5 → 1.8); rotor 41° → 11° clipped at 3×;
  bond share 0.64 unchanged, 0.46 rescaled with every component of the
  used pair at the same factor 1.5 and the stored pair returned as
  y + (ratio − 1) y_u; tiny chord / non-finite inputs unchanged.
- Falsification tests: (i) every multi-fragment system bit-identical;
  (ii) the near-equilibrium connected runs (rmsd₀ < 0.15 Å, 34 train
  molecules) now bit-identical or ±1 (their first steps have bond share
  > ½ — thermal-snapshot X–H errors of 0.025 Å rms are 0.02 eV per bond
  against 0.004 eV per angle); (iii) the rmsd₀ ≥ 0.4 Å bins (119
  molecules at 14.8 / 23.3 calls) carry the gain; (iv) the hydrogen-free
  set is the sensitivity read-out (stretch-dominated: mostly skipped);
  (v) the charged connected complexes (24, 2 : 6, Σ +25 in cycle 107) no
  longer lose.
- Expected: −0.003 … −0.008 train (a fraction of the cycle-107 target
  −0.005 … −0.010 since the mixed steps are corrected by the soft-block
  average); if the rmsd₀ ≥ 0.7 bin is again a wash the chord softness of
  the approach steps is not binding under the memory-4 update and the
  line closes after at most one more variant (soft-block-only
  attribution, u = P_soft y_u).

### Change (functions)
- `_ESEC_MAX_STIFF_SHARE = 0.5` next to the cycle-107 constants (comment
  block updated: Zhang–Xu vector-parameter form, soft-step scope).
- `PES._update_H(..., slopes=None, stiff=None)` passes the bond slice to
  `_energy_corrected_dg(dx, dg, T_now, tbar, slopes, stiff)`.
- `PES._energy_corrected_dg`: y_u = y + (T_now s − t̄) (or y), chord =
  y_u·s (> 2 meV), share = y_u[stiff]·s[stiff]/chord (return the pair
  unchanged when share > 0.5 or non-finite), c, local, ratio as in cycle
  107 (|ratio − 1| ≥ 0.05, clip [0.4, 3]), return y + (ratio − 1) y_u.
- `InternalPES._update_H`: `stiff = slice(self.int.ntrans, self.int.ntrans
  + self.int.nbonds)` (coordinate order translations, bonds, angles,
  dihedrals, other, rotations — the slice `get_x` already uses for the
  dihedral block) passed through both branches.
- Everything else as cycle 107 (slopes from `PES.kick`, `_energy_secant`
  gate on `ntrans + nrotations == 0`).  py_compile / pyflakes clean
  (pre-existing warnings only); static stub checks listed in the
  Hypothesis passed.

### Result
- Candidate `cda4caa`, train: fitness **0.6548609** vs champion 0.6512909
  (**+0.0035700**), 469/469 converged, mean rel_energy 1.0033056,
  evaluation `7ded9bc93e714ace8b1b73c38e7a5872`, 54 s.  Decision:
  **discard** (train improvement below 1e-4).  No validation run.
- Read-out (`/tmp/cmp107.py cycle-108`; feature splits on
  `/tmp/feat107.pkl`):
  - all 212 multi-fragment systems bit-identical (test i passed); 83 of
    257 connected runs bit-identical (52 in cycle 107);
  - connected Σ +39 calls (42 : 42): same-endpoint 255 molecules Σ +27
    (42 : 41, +0.0020) + 2 basin changes Σ +12 (135039840 SiOCl 10 → 22,
    relE 0.9997 → 0.9998, the third re-roll of that floppy molecule in
    three cycles);
  - the diagnosed harms of cycle 107 are gone: near-equilibrium rmsd₀ <
    0.15 Å 34 molecules 2 : 1 Σ −1 (test ii passed), hydrogen-free 66
    molecules 7 : 8 Σ +3 (test iv passed), charged connected 26 molecules
    3 : 4 Σ 0 (test v passed);
  - the target population does not respond (test iii failed): rmsd₀ ≥
    0.7 Å 56 molecules 18 : 18 Σ +22, [0.40, 0.70) 14 : 10 Σ −5, [0.15,
    0.25) 3 : 7 Σ +11; by turning rotors: n20 = 0 7 : 11 Σ +9, n20 = 1
    5 : 6 Σ +2 (cycle 107: 9 : 9 Σ +8), n20 2–3 13 : 6 Σ −15 (cycle 107:
    14 : 9 Σ −12), n20 ≥ 4 11 : 9 Σ +12 (cycle 107: Σ +39) with the same
    long folding runs losing in both variants (172113611 35 → 46 / 42,
    135140595 30 → 39 / 38, 134997543 28 → 40 / 31) against
    paliperidone_palmitate 56 → 48, 336270104 19 → 15;
  - by run length: champion calls [14, 21) 19 : 10 Σ −14, [21, 400)
    15 : 17 Σ +30 — the mid-length runs gain a little, the long ones lose.
- Preserved: `ideas/energy-corrected-secant/cda4caa11b58b7d343b9628e4877f4debf9f9072/algo.py`;
  backlog entry `energy-corrected-secant` updated to closed.

### Interpretation
- The repair did what it was built for — every population the cycle-107
  attribution flaw had harmed is neutral now (near-equilibrium, H-free,
  charged connected, all within ±3 calls) — and exposed the ceiling of
  the mechanism: the energy of a step is one scalar.  It fixes the
  curvature of one soft mode exactly (the Hermite value is within 10 %
  on rotor paths) but has to be attributed on every mixed step, and the
  two attributions bracket the truth from opposite sides: by step length
  (u = s) the stretch residual of many small bond steps lands on the
  torsions (cycle 107), by curvature-energy share (u = y_u) the
  near-harmonic angles and the ≤ ½ bond block absorb the torsional
  residual and the rotor is under-corrected (cycle 108).  On multi-rotor
  steps the residual cancels between rotors on the concave and the
  convex side of their wells (local < chord past 30°, > chord inside),
  so the long folding runs (n20 ≥ 4) lose under both attributions.
- The single-rotor population (n20 = 1, 50 molecules), where the
  attribution is unambiguous, did not respond in either variant (5 : 6,
  9 : 9): during a 60–120° turn the trust cap (0.5 rad, less after a
  shrink) bounds the step, not the model's softness, and the last capped
  step lands close enough for the endgame to finish in the same number
  of calls; the overshoot of the worked example (−11.7° with the chord
  model) happens only when the turn's last step is uncapped, i.e.
  rarely.  The only consistent gain — n20 2–3 at −0.3 calls per molecule
  in both variants — is ≈ −0.001 of the score, below what a train +
  valid gate can confirm.
- Together with cycles 16/17/50/51 (trend-gated θ along s), the
  energy-informed secant correction is now closed in both attributions
  and with a-priori gates: the cycle-21/59 reopening condition ("a
  chord-corrected torsional update") has been met and found not binding
  under the memory-4 transported TS-BFGS update.  The exact-slope
  bookkeeping in `PES.kick` is the reusable part (a Hermite-based trust
  ratio would be the other consumer of the same three numbers; not
  planned — the trust policy is closed, cycle 28/59).

### Next idea / control
- No third repair candidate: the pre-registered closing condition (the
  rmsd₀ ≥ 0.7 bin a wash again) is met, and the population a
  single-dominant-rotor gate would target (n20 = 1) is non-responsive in
  both variants, so the last diagnosed variant has no population.  The
  line is closed in the backlog with the untested forms listed.
- Champion stays `cf2439f` (train 0.6512909 / valid 0.6499489).  Next
  cycle: back to the census — the remaining connected excess sits in
  the approach phase of large-rmsd₀ starts (≈ 6 calls per e-fold of
  start displacement) and in the dimer endgame (credit decomposition
  first); model-side and update-side torsional treatments are closed in
  every form (37/40/41/46/58/68/69/91/93/97/103/107/108), so the next
  candidate must come from a different mechanism (step composition on
  the approach, or the start geometry) or from a new census class.

## Cycle 109 — self-calibrated stretch transport of the secant pairs of connected systems: the stretch part of a pair's analytic transport is scaled by the pair's own measured / modelled bond-block curvature-energy ratio c_j = (y_j·s_j)_bonds / (T̄_s,j s_j·s_j)_bonds, clipped to [0.5, 2], i.e. y_j → y_j + (T(x) − T̄_j) s_j + (c_j − 1)(T_s(x) − T̄_s,j) s_j (the contact block stays additive); multi-fragment systems untouched

### Hypothesis
- Mechanism.  The cycle-53 transport moves every stored pair with the
  analytic model, y_j → y_j + (T(x) − T̄_j) s_j, so that the secant
  conditions describe the curvature at x instead of the path average of
  the old segment.  The move is by the *model's* magnitude: where the
  Almlöf stretch constant of a bond is off by 1/c (the first-row
  calibration is 1.2–2× off on bonds to the heavier p-block elements even
  after the cycle-64 row factors — cycle 57's signature, the cycle-66
  census of boron / Si–Ge, the cycle-106 read-out that the exotic-bond
  molecules lost under a 1.5× steeper transport), the transported pair is
  off by (1 − c)(T_s(x) − T̄_s,j) s_j.  Hand numbers (Bb = 3.674 /Å): a
  bond 2× too stiff in the model that lengthens by 0.15 Å in one step is
  transported to 0.67 of its true curvature (T₁/T₀ 0.576, T̄/T₀ 0.769:
  additive 0.192 vs ideal 0.288), one that shortens by 0.15 Å to 1.23×
  (1.068 vs 0.868), and an older pair whose bond has relaxed by a further
  0.3 Å since is transported to a *negative* curvature (−0.193 vs
  +0.096) — a pair the consistency filter does not necessarily drop, and
  which the |B|-metric update then imposes along the stiffest sampled
  direction.  The exponential *shape* (Badger) is generic; only the
  magnitude is uncertain, and the pair measures it along its own step:
  c_j = (y_j·s_j)_bonds / ((T̄_s,j s_j)·s_j)_bonds, the ratio of the
  measured to the modelled curvature energy on the bond block (both with
  the same projection P diag(d) P of the model, so the redundancy
  distribution cancels), clipped to `_STRETCH_CALIB = (0.5, 2)` — the
  a-priori range of the residual magnitude error (cycle 66: boron
  stretches 1.1–1.7× over-stiff, Si/Ge 1.3–1.9×; cycle 104: F–H 2×
  under-stiff).  With it the transport is exact for a bond whose
  curvature is c_j × the model's along the whole path; the contact block
  keeps the additive transport (its magnitude is not what the bond block
  measures).  c_j is a property of the pair (stored with it, invariant
  under the cycle-76 symmetry images), so each old pair is moved by the
  magnitude it measured itself.
- Why this is not the cycle-58 per-coordinate reading: one scalar per
  pair from the energy of the bond block along the step — no
  per-coordinate secant readings, no per-bond constants; on a step whose
  bond block is dominated by one exotic bond (0.15² × k against 30 × 0.02²
  × k for the thermal-snapshot bonds) the scalar is that bond's ratio, on
  a first-row step it is ≈ 1 (± 0.1) and multiplies a transport term that
  is itself 2–5 % of the pair (0.02–0.03 Å bond changes) — a ≤ 0.5 %
  perturbation, i.e. re-roll noise on the organic population.
- Population (census of 2026-09-19, `backlog.md`
  `cycle-109-idea-search-closures`): exotic-bond relaxers (a bond with an
  element outside H/C/N/O/F changing > 0.1 Å start → end): train 64
  connected molecules (share 0.0992, 34 hydrogen-free), valid 47 (share
  0.0694, 9 hydrogen-free); the > 0.05 Å class train 118 / valid 110.
  This is the cycle-106 "untested variant" (a magnitude-free transport)
  whose population there was read as the hydrogen-free set alone (share
  0.054 / 0.012, "non-generalizable unless bundled"); the exotic-bond
  census gives it a 4–6× larger valid exposure, so it runs alone — one
  mechanism, no constants beyond the clip.  Multi-fragment systems
  (`ntrans + nrotations > 0`) keep the plain transport of cycle 55 —
  bit-identical, the energy margin untouched.
- Public sources consulted for this cycle's search (nothing directly
  adaptable, recorded in the backlog entry): JCTC 2023 optimizer
  comparison https://pubs.acs.org/doi/abs/10.1021/acs.jctc.3c00188; Lindh
  model Hessian https://arxiv.org/pdf/cond-mat/0211509; QUICCA
  https://arxiv.org/pdf/cond-mat/0404627; Mones et al. model-Hessian
  study arXiv:1804.01590; xtb ANCopt
  https://xtb-docs.readthedocs.io/en/latest/optimization.html;
  restricted-variance GEK https://pubs.acs.org/doi/10.1021/acs.jctc.0c00257.
- Static checks (`/tmp/stub109.py`, `ast`-extracted
  `PES._stretch_calibration` on hand numbers, no optimizer code
  executed): calibration off → 1; a 2×-too-stiff bond dominating the
  block → 0.516; clips at 0.5 / 2; non-positive modelled energy and
  non-finite inputs → 1; the calibrated transport recovers c T₁ s exactly
  for the pure-stretch case (0.2882 vs additive 0.1919).  py_compile
  clean, pyflakes identical to the champion (line shift of one
  pre-existing warning).
- Falsification tests: (i) all 212 multi-fragment systems bit-identical;
  (ii) the exotic-bond relaxers > 0.1 Å (64 train) Σ calls < 0 with the
  hydrogen-free 34 as the sensitivity read-out (expected Σ ≤ −10: cycle
  106 lost +23 on them with a 1.5× steeper transport); (iii) connected
  molecules without any exotic bond changing > 0.05 Å (≈ 110 train) within
  the re-roll band (|Σ| ≤ 0.002 of the score, no drift); (iv) the
  near-equilibrium connected runs (rmsd₀ < 0.15 Å, 34) within ±1 each;
  (v) no basin change on the connected set beyond the known floppy
  re-rollers (135039840 SiOCl).  Expected −0.002 … −0.004 train, −0.001 …
  −0.002 valid (the thin valid exposure is the risk).  Closing condition:
  if (ii) fails (hydrogen-free Σ ≥ 0) the transport magnitude is not
  binding either, and the stretch-transport line (53 / 106 / 109) is
  closed with the slope and the magnitude both measured.

### Change (functions)
- `_STRETCH_CALIB = (0.5, 2.0)` next to the symmetry-image constants.
- `PES._stretch_calibration(dx, dg, tsbar, calib)` (static): c =
  clip((dx·dg)_calib / (dx·tsbar)_calib, 0.5, 2); 1 when calib / tsbar
  is None, the modelled bond-block energy ≤ 0 or anything non-finite.
- `PES._update_H(..., Ts_now=None, tsbar=None, calib=None)`: the
  single-pair branch applies the same calibrated transport; the
  multi-secant branch passes the new arguments on.
- `PES._collect_secant_pairs(..., Ts_now, tsbar, calib)`: c computed from
  the raw pair before normalisation, stored with the pair as
  (s, y, tb, tbs, c); at use y = y + (T_now s − tb) + (c − 1)(Ts_now s −
  tbs) when c ≠ 1 (multi-fragment pairs carry c = 1 and skip the term —
  bit-identical arithmetic to the champion).
- `InternalPES._analytic_model_from` / `_analytic_model` /
  `_analytic_model_at` return (T, T_s) with T_s = P diag(d) P the
  projected stretch part (same operations for T as before);
  `_analytic_model_average` averages both with the same Simpson nodes and
  weights; `_track_analytic_model` returns (T_now, tbar, Ts_now, tsbar);
  `_symmetry_images` maps tbs like tb and passes c through;
  `InternalPES._update_H` passes the bond slice `slice(ntrans, ntrans +
  nbonds)` as `calib` for connected systems and None otherwise.

### Result
- Candidate `f8bae55`, train: fitness **0.6546206** vs champion 0.6512909
  (**+0.0033297**), 469/469 converged, mean rel_energy 1.0033091,
  evaluation `267ca6f7578444359ae124143fef8199`, 69 s.  Decision:
  **discard** (train improvement below 1e-4).  No validation run.
- Read-out (`/tmp/cmp109.py`, exotic-bond census on the champion
  endpoints, rmsd₀ from `/tmp/turns109.pkl`):
  - test (i) passed: all 212 multi-fragment systems bit-identical in
    calls and final energy; the 7 DES370K systems that are single-fragment
    by Sella's bond graph (the same 7 as in cycle 107) re-rolled Σ +1
    (pyridine__pyrrole 21 → 15, alkanes__monoatomics 11 → 17);
  - test (ii) **failed**: exotic-bond relaxers > 0.1 Å, 61 same-endpoint
    molecules Σ +19 (6 : 10, +0.0019); hydrogen-free 33 Σ +4 (4 : 6):
    135078067 (B₄C cluster) 9 → 17 — the molecule that lost 9 → 16 under
    the steeper slope of cycle 106 loses under the calibrated magnitude
    too — against 135094180 (P₂O₄) 21 → 16 and 135041910 9 → 6;
    H-containing 28 Σ +15 (2 : 4); bond change ≥ 0.2 Å 15 molecules Σ +15
    (1 : 4);
  - test (iii): connected molecules without an exotic bond > 0.05 Å, 188
    same-endpoint, Σ +25 (14 : 18, +0.0014) — inside the band but with
    the drift concentrated on the long runs (champion > 20 calls: 40
    molecules Σ +33, 8 : 12; 135140595 30 → 40, 172113611 35 → 43,
    paliperidone_palmitate 56 → 60, nintedanib 15 → 17; all with
    maxdb ≤ 0.055 Å, i.e. no bond relaxation at all);
  - test (iv) passed: near-equilibrium rmsd₀ < 0.15 Å, 34 molecules Σ +2
    (2 : 3);
  - test (v) **failed**: one basin change, 135097481 (N₄S₄ cage) 27 → 22
    into a 7.35 kcal/mol *higher* minimum (relE 1.0000 → 0.9689; Σ(relE
    − 1) 1.581 → 1.552), the largest endpoint change of any cycle since
    the docking work; 135039840 (SiOCl) stayed at 10;
  - by rmsd₀: [0.4, 0.7) Σ −13 (7 : 3) the only gaining bin, ≥ 0.7 Σ +36
    (9 : 12), [0.15, 0.4) Σ +14 (3 : 10); 32 / 250 connected runs
    bit-identical (the calibration touches every connected pair).
- Preserved: `ideas/stretch-transport-self-calibration/f8bae554eaa9a28acc2e51539a383f665fb442f9/algo.py`
  (exposes the stretch part T_s of the analytic model, its path average
  and a per-pair scalar carried through the symmetry images — reusable
  scaffolding); backlog entry `stretch-transport-self-calibration`
  (closed) and the record-only `cycle-109-idea-search-closures`.
  Champion restored (`7b6a256`, algo.py bit-identical to `cf2439f`).

### Interpretation
- The closing condition was met: the transport magnitude is not binding
  on the exotic-bond population.  The newest pair is re-measured along a
  relaxing bond at the very next step, so the one-step 30 % curvature
  lag of the additive transport along the sampled direction is repaired
  by the update itself before it costs a call — the same lesson as
  cycle 104 (a 1.2–1.4× stiff-mode error is learned on the first step).
  The molecules with the largest exotic-bond changes are small inorganic
  cages and clusters (N₄S₄, B₄C, P₂O₄, SiOCl: multi-centre bonding,
  strongly redundant coordinate sets) for which the bond-block ratio is
  dominated by the projection couplings, so the "measured magnitude" is
  noise there and the calibrated transport re-rolls them (a 7 kcal/mol
  basin change, +8 calls on B₄C).  With cycle 53 (the exponential shape
  pays), cycle 106 (a steeper slope loses) and cycle 109 (the measured
  magnitude does not pay) the stretch-transport line is measured in
  every direction.
- Structural finding (cross-cycle table, `/tmp` analysis of the
  per-molecule Δcalls of cycles 103–109 against the champion): the three
  pair-side perturbations of connected systems (107 u = s, 108 u = y_u,
  109 c_j) lose on the *same* runs — 135140595 +8 / +9 / +10, 172113611
  +7 / +11 / +8, 135231613 +3 / +3 / +4, 135316214 +11 / +2 / +3 — and
  gain on the same one (135094180 −4 / −3 / −5); by champion call bin the
  mean Δcalls of the pair-side cycles is +0.2–0.8 for [21, 31) and +2 …
  +3 for ≥ 31 calls, while the guess-side cycles 103 / 104 (most runs
  bit-identical) sit at −0.1 … +0.2.  The champion's long folding paths
  are selected lucky draws at the train gate (100 cycles of "keep if
  better"), and any perturbation of the update along them regresses to
  the mean: **a pair-side candidate pays ≈ +0.002–0.003 on train before
  its mechanism acts**, so it needs an expected gain ≥ 0.004 to be
  measurable.  Guess- and start-side candidates that leave the majority
  of runs bit-identical do not pay this and are the only class the
  train gate can still resolve at the ±0.001 level.

### Next idea / control
- Champion stays `cf2439f` (train 0.6512909 / valid 0.6499489).  The
  stretch-transport line (53 / 106 / 109) and the energy-corrected pairs
  (107 / 108) are closed; the exotic-bond relaxers are not a class with
  a model-side lever (the cages need a bonding model, not a calibration).
- Given the structural finding, the next candidates must be scoped so
  that most connected runs stay bit-identical (start-geometry rules,
  class-gated guess repairs on a census population, or changes that
  fire only under a geometric condition), or must promise ≥ 0.004 on
  train.  The idea search resumes from the census populations with that
  filter: the remaining connected excess is the approach phase of
  large-rmsd₀ starts and the dimer endgame, neither of which a pair-side
  change can now address measurably.
