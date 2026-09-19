# Research backlog

## tric-allow-fragments: per-fragment translation/rotation internals (Sella `allow_fragments=True`)
Status: incorporated

**Hypothesis:** Give every disconnected fragment (non-covalent dimers, ~46 % of the
set, the most expensive molecules) explicit centroid-translation and rotation
coordinates (TRIC, Wang & Song 2016) instead of long inflated-radius pseudo-bonds
with angles/dihedrals through them; intermolecular DOF are then well conditioned and
the near-linear H-bond dummy-atom machinery is avoided, so dimers need fewer steps.
**Outcome and uncertainty:** Dimer cost fell 21 % (37.3 → 26.1 mean calls) with the
single-fragment control exact, but the cycle was invalid (one xtb SCC failure while a
Cl⁻ ion lingered 6 Å from a disulfide) and the energy gate would have failed
(mean rel_energy 0.9968): the constant 0.05 Ha guess curvature for the fragment
coordinates is far too stiff for dispersion-bound or distant fragments, so the
intermolecular steps are tiny and the external displacement/energy convergence test
fires before the dimer reaches the anchor's minimum. Unresolved: whether some early
stops are symmetric zero-gradient arrangements rather than stiffness.
**Reason to revisit:** The step reduction on dimers is large and the failure mechanism
is localized to the guess curvature of the fragment coordinates; a contact-derived
(distance-dependent) curvature or a faster trust-radius growth should keep the gain
while restoring the anchor's ability to slide along flat intermolecular surfaces.
**Next experiment:** Cycle 3 repair: contact-derived curvature for translations and
rotations (sum of Almlöf pseudo-bond curvatures over inter-fragment pairs projected on
the rigid-body motion). Compare dimer step counts and per-molecule rel_energy against
`anchors/train.json`; success = valid run, mean rel_energy ≥ 1, dimer cost still < 0.9.

**Attempts:**
- Cycle 2; candidate `a5b0bd56c0434408dc4c079157fd506862f43f59`; champion `62ff882376f00637eae97beb66cbee96274c8a38`; decision invalid.
  [Implementation](ideas/tric-allow-fragments/a5b0bd56c0434408dc4c079157fd506862f43f59/algo.py).
  [Training evidence](evaluation_results/cycle-2-train.json).
  [Narrative](full_log.md#cycle-2--fragment-translationrotation-internals-allow_fragmentstrue).
- Cycle 3; candidate `13cbc5cca5ed795a1793d0eb993a685d1e0dec4b`; champion `62ff882376f00637eae97beb66cbee96274c8a38`; decision keep.
  [Training evidence](evaluation_results/cycle-3-train.json).
  [Validation evidence](evaluation_results/cycle-3-valid.json).
  [Narrative](full_log.md#cycle-3--repair-1-of-tric-allow-fragments-contact-derived-curvature-for-fragment-coordinates).

## trust-region-textbook: Nocedal–Wright/geomeTRIC trust-radius policy
Status: accepted (cycle 21 connected systems, cycle 22 multi-fragment systems with TR cap 0.25); multi-fragment cap line closed (cycle 44: cap 0.5 + Cartesian bound under the tracked contact model → discard, far starts/walkers +3…+6 calls); see also `cartesian-trust-region` (cycle 28: δ₀ = 0.25 + Cartesian bound for connected systems)

**Hypothesis:** Sella grows the max-internal-step radius only ×1.15 per step and only
for 0.75 < ρ < 1.33, shrinks ×0.9 only for ρ < 0.01; a textbook policy (growth ×1.5 for
ρ > 0.75 with a cap, faster shrink on failure) should shorten the trust-limited approach
phase of unrelaxed conformers.
**Outcome and uncertainty:** Cycle 4 (growth ×2, cap 1.0, shrink ×0.5 for ρ < 0.25, all
systems, cycle-3 curvature model): invalid, neutral for PubChem, harmful for dimers.
Cycles 18–21 (cycle-13 curvature model): growth-only ×1.5 / cap 0.5 helps connected
molecules (−0.8 % same-basin) and hurts dimers (+10 %); a separate TR radius (cycle 19)
neutralises the dimer loss but the changed dimer paths lose lottery energy credits;
restricting the policy to systems without fragment TR internals (cycle 20) reproduces the
champion's dimers exactly but leaves a heavy loss tail (a few molecules +13…+35 calls);
adding ×0.5 shrink on failure (cycle 21) cuts the tail → keep (train 0.8749174, valid
0.8861966; same-basin non-dimer −0.9 % / −1.3 %). Remaining tail (135098013 18 → 33,
384221749 58 → 74) unexplained.
**Reason to revisit:** One-parameter follow-ups: cap 0.3, per-type caps (bonds 0.3 Å,
bends/torsions 0.5 rad), growth trigger ρ > 0.75 → window only, shrink trigger ρ < 0.25
(N&W). For multi-fragment systems the radius is not the limitation (the quadratic model is);
cycle 19's split-radius implementation (`delta_tr`, common shrink factor) is the base for
any TR-specific policy and was neutral (−0.4 % same-basin dimers) — reuse it only together
with a better TR curvature model.
**Next experiment:** none for the multi-fragment cap: cycle 44 (cap 0.5 with the
connected-system Cartesian bound, δ₀ 0.1 kept, tracked inter-fragment contact model of
cycle 43) made the far starts (> 3 Å excess +3.2 calls) and the reorientation walkers
(> 50° +2.7; > 3 Å with > 90° +6.0) slower while near-equilibrium dimers were unchanged —
the same sign as cycle 18, so the motion coefficients of the dimer-cost regression measure
the model's validity range on the anisotropic far field, not the radius. Larger caps only
help with far-field curvature information (see `inter-fragment-tracked-contacts` next
experiment). The connected-system bond radius (0.3 Å) is the only untested one-parameter
follow-up and is low priority; the shrink trigger must stay at Sella's ρ < 0.01 / ρ > 100
— a trigger tuned to the endgame energy resolution would be the prohibited damping (see
cycle 22 boundary observation); loosening it by model consistency (cycle 59,
`retrospective-trust-region`) re-creates the growth-only tail on rotor-rich mid-phases.

**Attempts:**
- Cycle 4; candidate `617c6f77a320cdbb904b5b02f50309e8a2afe736`; champion `13cbc5cca5ed795a1793d0eb993a685d1e0dec4b`; decision invalid.
  [Implementation](ideas/trust-region-textbook/617c6f77a320cdbb904b5b02f50309e8a2afe736/algo.py).
  [Training evidence](evaluation_results/cycle-4-train.json).
  [Narrative](full_log.md#cycle-4--textbook-trust-region-policy-expand-2-for-ρ--075-shrink-05-for-ρ--025-cap-10).
- Cycle 18; candidate `be5daad3baf6b70011aa3f989d170e48c1111a39`; champion `9c106c78fda1a4fcf8ec5a5d863bd5a770ca0d30`; decision invalid (energy_below_baseline; train C_S 0.9117889; PubChem −0.49 calls, dimers +2.54).
  [Implementation](ideas/trust-region-textbook/be5daad3baf6b70011aa3f989d170e48c1111a39/algo.py).
  [Training evidence](evaluation_results/cycle-18-train.json) (fa1406ea4c73417db625e19b88ab985b).
  [Narrative](full_log.md#cycle-18--growth-only-trust-radius-invalid).
- Cycle 19 (repair 1: separate TR radius); candidate `c5b89d72d82f36e02958ac43c35a1fd723945fdc`; champion `9c106c78fda1a4fcf8ec5a5d863bd5a770ca0d30`; decision invalid (energy_below_baseline; train C_S 0.8769714; dimers −0.02 calls, lottery losses 104079126/440717067/carboxylates__ketones).
  [Implementation](ideas/trust-region-textbook/c5b89d72d82f36e02958ac43c35a1fd723945fdc/algo.py).
  [Training evidence](evaluation_results/cycle-19-train.json) (d97d8604efb64267846a0550ce0961ba).
  [Narrative](full_log.md#cycle-19--growth-only-repair-1-separate-radius-for-fragment-tr-internals-invalid).
- Cycle 20 (repair 2: connected systems only); candidate `75cce0a859dd48530215d84187f7366099841410`; champion `9c106c78fda1a4fcf8ec5a5d863bd5a770ca0d30`; decision non_generalizable (train 0.8771281, valid 0.8939665 vs 0.8936084; heavy loss tail on valid).
  [Implementation](ideas/trust-region-textbook/75cce0a859dd48530215d84187f7366099841410/algo.py).
  [Training evidence](evaluation_results/cycle-20-train.json) (df090c1600e046cdaa3de5546b06650a); [validation evidence](evaluation_results/cycle-20-valid.json) (7481334f9a0a44f09533ae743b212288).
  [Narrative](full_log.md#cycle-20--growth-only-repair-2-connected-systems-only-non_generalizable).
- Cycle 21 (repair 3: ×0.5 shrink on failure); candidate `630dd79df90bd3681e846ca2de6ef318667f4f46`; champion `9c106c78fda1a4fcf8ec5a5d863bd5a770ca0d30`; decision keep (train 0.8749174, valid 0.8861966).
  [Training evidence](evaluation_results/cycle-21-train.json) (a7bab73733bf45b2a080dafd8db2e8dd); [validation evidence](evaluation_results/cycle-21-valid.json) (7944103de42d4be6a0f9445281af4376).
  [Narrative](full_log.md#cycle-21--growth-only-repair-3-05-shrink-on-failure-for-connected-systems-keep).
- Cycle 22 (textbook policy for multi-fragment systems, TR cap 0.25); candidate `a277fc1cbf2ebec46359133c7228210d71338a9d`; champion `630dd79df90bd3681e846ca2de6ef318667f4f46`; decision keep (train 0.8658181, valid 0.8829537; dimers −0.37 / −0.53 calls same-basin).
  [Training evidence](evaluation_results/cycle-22-train.json) (e2cb5469b28f4d76acc922eb1b6a966f); [validation evidence](evaluation_results/cycle-22-valid.json) (7ddbf17ed37f43a29da86eae10025ccc).
  [Narrative](full_log.md#cycle-22--textbook-trust-radius-policy-for-multi-fragment-systems-tr-cap-025-keep).
- Cycle 44 (TR cap 0.5 + Cartesian bound 2δ for multi-fragment systems, δ₀ 0.1 unchanged); candidate `af3ce6a94e560c9cc6d0b3f277dbe6fc39f31ed3`; champion `6e34e59e16b19c3538a760b9783d28081d2453b3`; decision discard (train 0.7996507 vs 0.7886692; connected 254/257 identical, dimers +205 calls, 86:90:36; far starts and walkers +3…+6, near-equilibrium ±0). Trivial parameter change, not preserved in `ideas/`.
  [Training evidence](evaluation_results/cycle-44-train.json) (4e01c2dafa894322ab988d9809a78c7f).
  [Narrative](full_log.md#cycle-44--same-trust-cap-05-and-cartesian-bound-2δ-for-multi-fragment-systems-as-for-connected-ones-discard).

## bfgs-update: BFGS instead of TS-BFGS for the internal Hessian update
Status: closed

**Hypothesis:** Bofill's TS-BFGS is a saddle-search compromise; plain BFGS (with a
curvature test on the step subspace, TS-BFGS fallback) should converge faster on the
soft modes that dominate the tight endgame.
**Outcome and uncertainty:** Contradicted decisively: C_S train 0.939 → 1.055, 53 % of
PubChem and 84 % of dimers slower, three runs ≥ 4× slower, one frozen (huge curvature
from a tiny positive yᵀs; displacements 1e-10 Bohr for 150 calls). Energy gate failed
marginally (0.99993). Not a bug: `_MS_BFGS` is the textbook formula; the loss is
systematic, not just in the tails.
**Reason to revisit:** Only as a Powell-damped variant if a future change makes the
secant data cleaner; low priority. Cycle 33 (bofill-sr1-update) rejected SR1/Bofill just as decisively; see that entry.
**Next experiment:** none planned.

**Attempts:**
- Cycle 5; candidate `e0360bc9b6d8a288b2a5d4636bbcf45c3f990653`; champion `13cbc5cca5ed795a1793d0eb993a685d1e0dec4b`; decision invalid.
  [Implementation](ideas/bfgs-update/e0360bc9b6d8a288b2a5d4636bbcf45c3f990653/algo.py) (also contains the `_h0_fragment` dummy-index repair).
  [Training evidence](evaluation_results/cycle-5-train.json).
  [Narrative](full_log.md#cycle-5--bfgs-update-for-minimization-bfgs_auto-with-a-subspace-curvature-test--dummy-index-repair-of-_h0_fragment).

## gdiis-endgame: controlled GDIIS (Farkas–Schlegel 2002) on the QN/TS-BFGS step
Status: parked

**Hypothesis:** A GDIIS extrapolation over the last ≤ 5 points/gradients, built with
the current TS-BFGS Hessian and the Farkas–Schlegel safeguards (cos thresholds by
subspace size, |GDIIS| ≤ 10|QN|, c_n > 0, Σ|c| ≤ 10), should shorten the tight endgame,
which is limited by soft-mode curvature accuracy.
**Outcome and uncertainty:** Cost-neutral on train (C_S 0.93860 vs 0.93888): big gains
on the longest runs (≥ 60 calls: 71 → 61; amides__ethers 77 → 53, carboxylates__esters
66 → 35) offset by slightly longer mid-range runs and two blow-ups (carboxylates__water
20 → 57). Invalid through the energy gate (mean rel_energy 0.99935): ammoniums__ketones
converged after 12 instead of 48 calls at a point 1.19 kcal/mol higher (rel 0.68) and
404345632 reached a different minimum (0.93). The TS-BFGS secant update already carries
the information GDIIS extrapolates, so the net effect is a path change, not a
speed-up. No traces (all runs converged).
**Reason to revisit:** The long-run gains are real and large; if a future Hessian model
makes the long runs rarer this becomes moot, otherwise a fragment-aware restriction
(no GDIIS while fragment translation/rotation gradients are significant, DIIS step
≥ 0.25 × QN step) could keep the gains without the early stops.
**Next experiment:** Repair 1 as above; success = train valid (mean rel_energy ≥ 1) and
C_S ≤ 0.9379 (gate 1e-4 below the champion) with the ≥ 60-call runs still improved.
**Closure (2026-09-18, static, before cycle 76, no cycle consumed):** the fragment-aware repair is redundant with
the champion's multi-secant update (cycles 53/55/56: the ≥ 60-call runs that GDIIS shortened no longer exist —
the longest docked-dimer runs are soft-mode walkers whose endpoints carry the energy credits, see
`surrogate-docking`), so the repair would be a path change on credit walkers, i.e. an energy-gate re-roll.
Not to be built. Also closed statically in the same search: noise-aware update skipping (xTB energies are
smooth to 1e-10 Ha — the dimer endgame is not noise-limited), sparse/partitioned block updates (ill-posed like
the per-rotor secant stiffness of cycle 58), contact internals across strong inter-fragment contacts (≈ 0 upside
on the 12 affected runs), a concave-mode trust-boundary step (near-eclipsed rotor starts carry no call premium
over the reference), the RFO stepper (uniform soft-mode damping while the trust bound is already not binding).

**Attempts:**
- Cycle 6; candidate `ec3782e85afea3ee86f0a448002d4354345fb6d7`; champion `13cbc5cca5ed795a1793d0eb993a685d1e0dec4b`; decision invalid.
  [Implementation](ideas/gdiis-endgame/ec3782e85afea3ee86f0a448002d4354345fb6d7/algo.py) (also contains the `_h0_fragment` dummy-index repair).
  [Training evidence](evaluation_results/cycle-6-train.json).
  [Narrative](full_log.md#cycle-6--controlled-gdiis-endgame-farkas--schlegel-2002-on-the-qnts-bfgs-step--_h0_fragment-repair).

## torsion-redundancy: share the Almlöf torsional constant over the redundant dihedrals of a bond
Status: incorporated (cycle 9)

**Hypothesis:** Sella gives every one of the n = (n_b − 1)(n_c − 1) dihedrals around a
bond the full Fischer–Almlöf torsional constant, so a rigid rotation is modelled n×
too stiff (sp³ C–C: 0.08 vs ≈ 0.02 Ha/rad²); the first quasi-Newton steps along soft
torsions are then 2–4× too short and the rank-2 update repairs one direction per
step. Dividing each dihedral's guess by √n should shorten floppy-molecule runs.
**Outcome and uncertainty:** Cycle 7 (1/√n for all bonds): valid but C_S 0.9511 vs
0.9389 (discard); molecules without short bonds improved in every cost bin (up to
−9 %), molecules with double/aromatic bonds got slower on short runs (+6–13 %).
Cycle 8 (× bond-order factor exp(−Ct (r − r_cov)/Bohr)): C_S 0.9094 but invalid on
the energy gate because the factor also hit Sella's improper dihedrals (umbrella
modes), flipping the formaldehyde dimer ketones__ketones into a shallower basin.
Cycle 9 (factor and sharing for proper dihedrals only): **keep**, train C_S
0.90994 (−2.9 %), valid 0.93389 (−2.6 %), energies 1.0001 / 1.0072; the gain
grows with molecule size (PubChem 40–80 atoms −9 %), dimers unchanged on train
(0.997) and +1.8 % on valid through basin changes. Remaining: the reactive
C6N6O6 case 135043047 closes only two of three forming N–O bonds (−0.16).
**Reason to revisit:** A 1/n split or a per-bond-type physical torsional constant
(geomeTRIC uses 0.023 per dihedral; xtb's Lindh-type 0.0075 with ring/bond-order
terms) could recover the remaining 2–4× stiffness of methyl rotors next to sp²
centres; ring torsions would need separate treatment.
Update (cycle 37): the per-bond-class scaling for acyclic bonds to a planar centre
is now in the champion (see `torsion-class-scale`); ring torsions were left at the
full constant as suggested here.
**Next experiment:** Only after the dimer/reactive levers are exhausted: compare
1/n with a floor for ring bonds against cycle 9 on the no-short-bond molecules.

**Attempts:**
- Cycle 7; candidate `6a301d6b92824d60a500ccf101e7772a07b7a871`; champion `13cbc5cca5ed795a1793d0eb993a685d1e0dec4b`; decision discard.
  [Implementation](ideas/torsion-redundancy/6a301d6b92824d60a500ccf101e7772a07b7a871/algo.py) (also contains the `_h0_fragment` dummy-index repair).
  [Training evidence](evaluation_results/cycle-7-train.json).
  [Narrative](full_log.md#cycle-7--share-the-fischeralmlöf-torsional-constant-over-the-redundant-dihedrals-of-a-bond-1n-scaling--_h0_fragment-repair).
- Cycle 8; candidate `c4972f1e0c508fafee12a2fdbeceb506786dd0d6`; champion `13cbc5cca5ed795a1793d0eb993a685d1e0dec4b`; decision invalid (energy_below_baseline; C_S 0.9094).
  [Implementation](ideas/torsion-redundancy/c4972f1e0c508fafee12a2fdbeceb506786dd0d6/algo.py) (also contains the `_h0_fragment` dummy-index repair).
  [Training evidence](evaluation_results/cycle-8-train.json).
  [Narrative](full_log.md#cycle-8--repair-1-of-torsion-redundancy-bond-order-factor-on-the-torsional-guess-1n-sharing-kept--_h0_fragment-repair).
- Cycle 9; candidate `19f0b01645625b416beed9a66fbae55bf58b6d49`; champion `13cbc5cca5ed795a1793d0eb993a685d1e0dec4b`; decision keep.
  [Training evidence](evaluation_results/cycle-9-train.json).
  [Validation evidence](evaluation_results/cycle-9-valid.json).
  [Narrative](full_log.md#cycle-9--repair-2-of-torsion-redundancy-bond-order-factor-and-1n-sharing-for-proper-dihedrals-only--_h0_fragment-repair).

## bond-formation-aware-internals: rebuild bonds/internals when covalent connectivity changes during a run
Status: open

**Hypothesis:** 23 train molecules change covalent connectivity during the run
(608 of 11026 champion calls; e.g. 135043047 forms three N–O bonds, boranes
104079126/135231613); Sella only re-detects near-linear angles, so a forming bond
is described through distant angles/dihedrals and the fragment coordinates, its
curvature is wrong, and which bonds close depends on the path (cycle 8 lost 38
kcal/mol on 135043047). Re-detecting bonds when a non-bonded pair comes within the
bonding radius (or a bond stretches far beyond it), rebuilding the internals and
re-initialising the Hessian (the machinery `check_for_bad_internals` →
`initialize_pes` already uses) should make these runs both cheaper and more
reproducible.
**Outcome and uncertainty:** not tried. Re-census against the cycle-45
champion (`/tmp/bondchange.py`, 1.25 r_cov graph at the input vs the final
geometries): the class is 24 train / 17 valid molecules (20 / 14 connected),
already cheaper than the reference (connected rel 0.866 / 0.859, Σ −98 /
−111 calls; share of C_S 0.037 / 0.026) and almost all "changes" are
threshold contacts of the input geometry that relax away (0/1 formed/broken)
rather than reactions; the true bond formers (135043047, 135231613,
135112265, 135094180) are 0.6–0.8 of the reference. Only
des370k_esters__phenol__valid (33 → 55) and 135065494 (92 → 78) are
expensive. The multi-fragment members (monoatomic ions binding, 4 / 3
molecules) are 1.19 / 1.25 of the reference but reach deeper basins
(−10 / −15 kJ/mol).
**Reason to revisit:** low — the 5.5 %-of-calls figure was pre-cycle-13; the
class now carries no systematic excess. Only if a rebuild mechanism is needed
for another reason (e.g. inter-fragment bond formation of the ion dimers).
**Next experiment:** implement the connectivity check in `Internals.check_for_bad_internals`
(or in `Sella.step` before `kick`), rebuild with the existing dummy/angle logic, and
evaluate; success = the 23 reactive molecules cheaper on average with no energy loss.

## angle-valence-scale: Fischer–Almlöf bend guess at valence force-field magnitude
Status: incorporated (cycle 10, champion `9d64f8b2809b7b9b878b18de1d8afc39a66fa40d`); two-level refinement rejected twice (cycles 11, 38) — size-dependent trade, parked.

**Hypothesis** The Fischer–Almlöf bend formula gives 0.28–0.40 Ha/rad² per
primitive angle, about twice the valence force-field constants for the same
angles (0.12–0.23 Ha/rad²; Schlegel 0.16/0.25). Sella's model energy over
redundant primitives has the form of a valence force field, so the primitive
diagonal should match the valence constants.

**Outcome and uncertainty** ×0.5 on every bend: train 0.90994 → 0.90497,
valid 0.93389 → 0.92182 (keep). Small molecules (<20 atoms) −3 to −4 %,
dimers −1.6 %/−2.4 % (2:1 better:worse, same basins), but PubChem ≥35 atoms
+2–3 % (+0.5–0.8 calls, broad and same-basin). The valence/Almlöf ratio is
not uniform: 0.43–0.47 for angles with a terminal H, 0.57–0.66 for
heavy-atom-only angles (C–C–C 0.66, C–C=O 0.58, C–O–H 0.57), so the single
factor over-softens the skeleton angles that dominate large molecules.

**Reason to revisit** A two-level scale (≈0.45 with a terminal H, ≈0.6–0.65
heavy-only) follows directly from the valence data and should recover part
of the large-molecule loss. Bend–bend coupling across an sp³ centre (valence
cross terms −0.02 to −0.05 Ha/rad²) is the next physical ingredient if the
diagonal refinements stall.

**Next experiment** The two-level scale was tested in cycle 11 (0.5/0.62,
all systems) and cycle 38 (0.5/0.65, connected systems only, dummy angles
heavy): both times the large-molecule gain appeared (≥ 35–40 atoms −0.3 to
−2.5 %) and both times the aggregate did not pass — cycle 11 through dimer
scatter and a conformer flip, cycle 38 through +0.16 calls on molecules
under 20 atoms at 2–3× the leverage (Σ Δcalls −19 but C_S +0.00035). No
angle class carries the small-molecule loss (class slopes within ±0.03
calls/angle), so there is nothing to repair; a size gate is a count
predicate and out of bounds. Only worth another look inside a bend-model
change that also helps small molecules (bend–bend coupling across an sp³
centre, or a heavy scale that fades with the number of heavy neighbours
of the centre in a physically justified way). Linear-centre (dummy-atom)
angles must not take the heavy scale: their physical constant is 2–7×
below the Almlöf value already (see linear-bend-guess).

**Attempts**
- cycle 10, candidate `9d64f8b2809b7b9b878b18de1d8afc39a66fa40d` vs champion
  `19f0b01645625b416beed9a66fbae55bf58b6d49`: keep (train 0.90497,
  valid 0.92182). Evidence: `evaluation_results/cycle-10-train.json`
  (71a115ea593347e8bb76e46752995a2d), `evaluation_results/cycle-10-valid.json`
  (dd2cae46930a474a843d60d4a825feb0); `full_log.md` cycle 10.
- cycle 11, candidate `ebfd6feffc3ac2d6cf4690bd31399615c0d8e83e` (two-level
  scale 0.5/0.62) vs champion `9d64f8b2809b7b9b878b18de1d8afc39a66fa40d`:
  discard (train 0.90740 vs 0.90497). PubChem −0.9 % as predicted (large
  molecules recovered), but dimer same-basin scatter (+58 calls, 52:51
  better:worse) and one rowansci conformer flip (venetoclax 41 → 82) outweigh
  it. Preserved in `ideas/angle-valence-scale/ebfd6feffc3ac2d6cf4690bd31399615c0d8e83e/algo.py`;
  evidence `evaluation_results/cycle-11-train.json`
  (c7f2fe103b0445569cddd6858a1286da); `full_log.md` cycle 11.
- cycle 38, candidate `6ce72a56796fb6c98855136c475699bb64241a09` (two-level
  scale 0.5/0.65, connected systems only, dummy angles heavy) vs champion
  `3ca2b9dd7c07605f17b891694e506742d101e5a6`: discard (train 0.8278188 vs
  0.8274668, Δ +0.00035; valid not run). Multi-fragment systems
  bit-identical; connected 71:63 better:worse, Σ Δcalls −19, size trend
  < 20 atoms +0.16 calls, 20–40 −0.21, 40–60 −0.21, ≥ 60 −1.11;
  acalabrutinib 16 → 31 (two linear centres). Preserved in
  `ideas/angle-valence-scale/6ce72a56796fb6c98855136c475699bb64241a09/algo.py`;
  evidence `evaluation_results/cycle-38-train.json`
  (8e757c9e8c934031862ed608010515b1); `full_log.md` cycle 38.

## multi-secant-update: limited-memory multi-secant TS-BFGS with consistency filter

Status: incorporated (cycle 13, champion `9c106c78fda1a4fcf8ec5a5d863bd5a770ca0d30`); memory 8 refuted for connected systems without transport (cycle 47) and again for all systems with the transported pairs (cycle 56: connected +0.0035, multi-fragment +0.0155) — the memory line is closed for good; the block-size and tolerance refinements stay parked.

**Hypothesis** A single-secant update makes the Hessian model exact only
along the newest step, so a soft subspace of n modes is learned in ≈1.5–2 n
steps. Imposing the last m independent secant pairs simultaneously (the
vendored `update_H`/`_MS_TS_BFGS` block form B S = Ỹ with `symmetrize_Y2`)
makes the model exact on the span of the recent steps and shortens the
endgame of dimers and floppy molecules. The pairs must be mutually
consistent with one symmetric Hessian, which fails for anharmonic soft
modes; an older pair is therefore only kept if it passes the symmetry test
|s_new·y_old − s_old·y_new| ≤ 0.25·sqrt((s_old·y_old)(s_new·y_new)) against
every newer kept pair (and the dependency test on its orthogonal remainder,
tolerance 0.3 after normalisation).

**Outcome and uncertainty** Candidate 1 (no consistency filter) helped the
covalent sets (PubChem −4 %) but lengthened dimer endgames (+1 call
same-basin, tails +30 calls); the filtered version keeps the covalent gain
and turns the dimers into a gain (−2.3 % train and valid, both sources
improve, 125:62 better:worse on train dimers). Near-minimum dimers
(ref < 15) are still slightly worse (1.44 → 1.60). The gain is consistent
across splits and sources; the energy sums are basin-lottery noise.

**Reason to revisit** Memory 6–8 was tested in cycle 47 (connected systems
only, dimers bit-identical): +0.44 calls per connected molecule, the loss
concentrated in short runs (9–16 calls: the first large relaxation steps
stay in the block through the whole endgame) and a wash in long runs
(consistency filter drops the old pairs). The marginal value of a pair is
positive at ages 2–4 and negative at 5–8, so the optimum is at or below 4.
Remaining untested refinements: a minimum block size of 3 pairs (the
2-pair blocks of the first steps may still act in the contact regime —
suspected cause of the near-minimum dimer loss), the consistency tolerance
(0.15 / 0.4), and a scale-aware retention (drop a pair once the newest step
is > 8× shorter — memory ≈ 3 in the geometric endgame, 4 in walks; ≤ 0.3 %
either way). None is measurable alone under the lottery; bundle with the
next H0 or endgame change. Reopened after cycles 53/55: the cycle-47 loss was
attributed to the first large relaxation steps being kept through the endgame
with their *averaged* bond-stretch and contact curvature (Morse ≈ 30 % per
0.05 Å) — exactly the part that the secant transport now moves to the current
geometry (composite Simpson average for the large early segments); the
torsional staleness (≈ 20 % per 0.2 rad) remains. For two-fragment systems
the six rigid-body modes exceed a memory of four, so the endgame model
reverts to the guess along the modes sampled ≥ 5 steps ago.

Cycle 56 answered this: with transported pairs the connected harm shrank
from +0.44 to +0.18 calls per molecule (38:51 better:worse, 168 identical;
the transport removed most but not all of the staleness — torsional
anharmonicity remains) and the multi-fragment systems lost +0.83 calls per
dimer (43:97), worst for approaches of 0.4–0.8 Å (+1.13, 8:43): the
approach-phase pairs are stale beyond the analytic contact block (long-range
electrostatic/dispersion curvature that the exponential contact term does
not carry, and rigid-body rotation coordinates whose local frame changes
along the approach), and the pairwise consistency filter does not catch
them. Marginal value of a pair is negative beyond age 4 in both classes.

**Next experiment** none on the memory count. A frame-aware treatment of
old rotation pairs or a dispersion/electrostatic long-range term in the
analytic model would be prerequisites for any longer memory.

**Attempts**
- cycle 12, candidate `ac30253ee6257494156a9af03f75fe334c61a817` (memory 4,
  dependency filter only) vs champion `9d64f8b2809b7b9b878b18de1d8afc39a66fa40d`:
  discard (train 0.91730 vs 0.90497; PubChem 0.970 → 0.960, dimers
  0.837 → 0.870). Evidence `evaluation_results/cycle-12-train.json`
  (0414ba73ec154cf2b4e11d676364017e); `full_log.md` cycle 12.
- cycle 13, repair 1 `9c106c78fda1a4fcf8ec5a5d863bd5a770ca0d30`
  (quadratic-consistency filter 0.25) vs the same champion: keep (train
  0.88450, valid 0.89361). Evidence `evaluation_results/cycle-13-train.json`
  (a30bfd3f6b044b89a36c5129b50a13b3), `evaluation_results/cycle-13-valid.json`
  (7f0fa4abd934431f9284e1d6008f280c); `full_log.md` cycle 13.
- cycle 47, candidate `9c4aade0c74c2ed5bcc9e5b35eba21519e648e82` (memory 8
  for covalently connected systems via `secant_memory_connected`, memory 4
  kept for systems with TR internals) vs champion
  `21a3273fee7548462d38cebd93dcfe77f0de31b4`: discard (train 0.7742212 vs
  0.7626995, Δ +0.0115; valid not run). Dimers 212/212 bit-identical;
  connected 34 better / 62 worse / 161 same, Σ +112 calls; champion runs of
  9–11 calls 0 better / 9 worse. Preserved in
  `ideas/multi-secant-update/9c4aade0c74c2ed5bcc9e5b35eba21519e648e82/algo.py`;
  evidence `evaluation_results/cycle-47-train.json`
  (e8396762e39b4b029cbee8d3e6c28759); `full_log.md` cycle 47.
- cycle 56, candidate `e332641c7bb3ac695a7e1a9ebc7de5da3971b32d` (memory 8 for all systems, pairs transported
  with the analytic model since cycles 53/55) vs champion
  `5d0a542040b30ac0f066135252c6bc80f13d072f`: discard (train 0.7384124 vs
  0.7193841, Δ +0.0190; mean rel energy 1.00169 valid; valid split not run).
  Connected +0.0035 (+0.18 calls each), dimers +0.0155 (+0.83 each; only 2
  basin changes). Evidence `evaluation_results/cycle-56-train.json`
  (ee622eeac737434fab0dee38b5c11e2f); `full_log.md` cycle 56.

## contact-bend-guess: Lindh-type bending stiffness of inter-fragment contacts in the fragment TR guess
Status: kept in cycle 45 as a geometry-tracked inter-fragment term, in cycle 48 extended to intramolecular heteroatom-donor contacts, in cycle 49 completed by the lone-pair-plane (out-of-plane) term for single-neighbour sp² acceptors, and in cycle 62 extended to the X–H···π contacts of heteroatom donors between fragments (champion 09bb577); cycle 63 π-face normalisation (a face is worth at most a double bond) keep, marginal (−0.0002 / −0.0005), π line closed; cycle 72 ionic cap ×2 for charged fragments discard (same-basin Σ 0, see `ionic-hbond-cap`); open: metal cations (excluded by the energy-gate credits); closed by census: 1,5 contacts, sp² N wag (covered by the bend at H), twist about H···Y (near the floor), butterfly modes (no missing mode); cycle 82 fragment-mode ion–dipole term for cation contacts discard (same-basin Σ +10 / 36, see `ionic-contact-dipole-fragment`) — metal cations closed too

**Hypothesis** The fragment translation/rotation guess (`_h0_fragment`,
cycle 3) models only the stretch projections of inter-fragment contacts, so
the libration and sliding stiffness of hydrogen-bonded and salt-bridge dimers
sits at the 1e-3 Ha floor, 5–30× too soft (water dimer libration
≈ 0.01–0.03 Ha/rad², benzene dimer ≈ 0.02). Adding a Lindh-type bending
stiffness k_φ·ρ_bond·ρ_contact for every angle k–i···j / i···j–l around a
contact (k_φ = 0.15 Ha/rad², isotropic ε = â + b̂ linear-bend form) should
shorten the dimer endgames, where the displacement criterion forces the
forces of these soft modes 100–1000× below threshold.

**Outcome and uncertainty** Train 0.89267 vs champion 0.88450 (dimers
1.034, PubChem/rowansci unchanged 1.000). Near-minimum salt bridges gained
enormously (ammoniums__carboxylates 29 → 11 = reference, acids__ammoniums
30 → 20, alcohols__carboxylates 36 → 26), but the net over the 219 dimers is
+0.61 calls: strong contacts (bo_max ≥ 0.02, n = 67) +0.96 (23 better /
35 worse), moderate (0.005–0.02, n = 31) +1.97 (8/20), weak (n = 114)
neutral. By initial→final RMSD of the champion run: < 0.3 Å +0.04 (9/11),
0.3–0.6 Å +0.14 (8/11), ≥ 0.6 Å +2.86 (6/13); the largest losses are far,
very short contacts (bo_max 0.19–0.35: acids__acids 41 → 54,
alcohols__pyridine 27 → 39, pyridine__pyridine 35 → 45) and a few
near-minimum ones that were already faster than the reference (pyrrole__water
16 → 27, amides__pyrrole 19 → 26). The effect is deterministic (95 dimers
identical) and not a basin lottery (energy 1.0004).

Cycle 15 (full rigid-body block, cross-fragment terms, true-angle
derivatives, bending only for X–H···Y contacts with per-donor caps): train
0.87726 vs 0.88450 (−0.0072) but valid 0.90397 vs 0.89361 (+0.0104) →
non_generalizable. The train gain was a basin lottery: 9 basin changes
contribute −59 of −73 calls (alcohols__sulfides 40 → 15, amides__phenol
78 → 44, carboxylates__guanidiniums 29 → 13), the 460 same-basin runs sum to
−14 (80 better / 86 worse). On valid the same-basin dimers are +0.38
calls/molecule (71 better / 87 worse), with amides__ammoniums 42 → 66 and
amides__phenol 41 → 62 against phenol__thiols 29 → 12 and acids__ammoniums
23 → 16. PubChem/rowansci paths are identical on both splits. Energies
1.0013 / 1.0078 (valid, unchanged).

**Reason to revisit** The two implementation defects of cycle 14 were
fixed (the full block is frame-invariant, PSD, the twist about a contact
axis sits at the floor, the relative step is the exact model step), and the
result is a wash with high per-molecule variance in both directions. So the
over-stiff hidden librations of the champion's diagonal stretch model are
not what costs the near-minimum dimers their calls, and a better rigid-body
guess does not change the endgame, which is governed by the displacement
criterion (soft modes have to be converged to |g| ≈ k·1.8e-3 Bohr, 100× below
the force threshold) and by the anharmonicity of the intermolecular surface
rather than by the initial curvature. The decomposition of the champion's
initial→final motion confirms this: for the ref < 15 dimers the relative
(rigid-body) RMSD is 0.26 Å (median) against 0.03 Å intramolecular — the
optimizer slides along the flat intermolecular surface while the reference
stops within ≈ 10 calls, presumably because its stiff model never moves
those coordinates. Physics-based guesses cannot close this gap; only a
different endgame strategy (convergence acceleration, not damping) can.

**Next experiment** Cycle 45 settled the question: the cycle-15 bend
block, rebuilt at every geometry inside `_h0_nonlocal_contacts` (so it
is tracked like the pair term) and with physically chosen constants
(A_φ,H 0.5 Ha/rad² isotropic at the hydrogen, A_φ,acc 0.15 true-angle at
the acceptor, contact factor capped 0.05 / 0.01 for C–H donors), is worth
−0.026 C_S on both splits, concentrated in the short-reference charged
and H-bonded dimers (train rel 1.263 → 1.058) and the same-endpoint
endgame (−1.4…−2.9 calls). Cycle 48 applied the same term, unchanged, to
the *intramolecular* non-local pairs (graph distance ≥ 5) with a
heteroatom donor: −0.0034 / −0.0035, all of it in the 36 / 36 connected
molecules with an H···Y contact < 2.6 Å (−1.4 / −2.3 calls each, −2.9 /
−4.1 with ≥ 2 contacts), everything else bit-identical; the class still
sits ≈ +3 calls above the no-contact baseline. Open follow-ups: (a) the
1,5 contacts (H···Y at graph distance 4 — 1,2-diols, α-hydroxy
carbonyls, 2-aminoethanols) are outside the min_path 5 pair set; census
of their excess cost first, and only the bend term (the FA torsion of
the central bond already stands for the 1,5 repulsion; cycle 7 found
min_path 4 doubtful for the pair term); (b) an acceptor lone-pair
direction term (the true-angle acceptor bends leave the precession of H
about the Y–Z axis at the floor; for sp² O/N the in-plane preference is
real) for inter- and intramolecular contacts alike; (c) metal-cation
contacts have no bend and a ~10× too stiff radial term (7 dimers).
Intramolecular C–H···Y contacts stay out (no excess cost in the census,
< 0.005 Ha/rad²). Constants are not to be scanned.

Cycle 49 did (b) for the class where the precession is genuinely
invisible: acceptors with a single covalent neighbour on a trigonal
centre (all carbonyl-type O=C in the data) got the term k s²/2, s = sine
of the angle between Y→H and the plane (W₁, Z, W₂), k = A_φ,acc
min(ρ_contact, cap)·min(ρ_YZ, 1)·planarity(Z), same prefactor 0.15 as
the acceptor bends, gradient over H, Y, Z, W₁, W₂
(`_h0_lone_pair_plane`, called from `_h0_contact_bends`, tracked):
−0.0060 / −0.0127. Census beforehand: single-neighbour-acceptor dimers
rel 0.95 / 1.09 vs 0.59 / 0.68 for two-neighbour acceptors, +5.5 / +9.5
calls above the same-basin baseline (n 26 / 22). Gain −2.0 / −1.8 calls
per single-contact dimer (39 / 31), −0.3 / −0.6 with ≥ 2 carbonyl
contacts (salt bridges, acid dimers), intramolecular neutral, three
charged walkers collapsed (53 → 27, 48 → 27, 44 → 9): walker creep runs
partly along model-invisible modes. (a) 1,5 contacts: census residual
−1.7 / +1.5 (X–H···Y angle ≈ 104°) — not a lever, closed. Remaining:
(d) the out-of-plane wag of two-neighbour sp² N acceptors (pyridine,
imine, azole N: both bends see the out-of-plane motion of a bisector
hydrogen only to second order; census nY = 2 N-acceptor dimers rel
0.68 / 0.72 vs O 0.56 / 0.70, n 14 / 16 — weak evidence, census the
same-basin residual first); (e) the twist about H···Y (donor rotation
about the contact; water-dimer donor torsion ~130 cm⁻¹ gives ≈ 0.002–
0.004 Ha/rad², near the floor — probably not a lever); (f) ring/butterfly
modes of doubly hydrogen-bonded dimers (the ≥ 2-contact class kept its
excess); (c) metal cations (≈ −0.002 per split).

Cycle 62 (2026-09-17) re-examined (d)–(f) analytically on the cycle-55
endpoints and closed them (the sp² N wag is held by the isotropic bend at
H; the twist sits within 2× of the floor; the doubly H-bonded dimers have
no mode both bends miss) and added the last missing mode class, the
X–H···π contacts between fragments: a hydrogen of a heteroatom donor
whose closest non-hydrogen atom of another fragment (by r − r_cov) is an
unsaturated carbon is a π donor towards that fragment, and every such
carbon within range gets the bend at H and the acceptor bends with the
weak cap 0.01 (`pi_donor` set in `_h0_contact_bends`). Keep, −0.0014 /
−0.0014, but the read-out is split by the face size (`/tmp/piface62.py`):
alkene faces (2 carbons, Σc ≤ 0.02 → k_H ≤ 0.01 Ha/rad², physical)
−0.8 / −1.1 calls same-basin; aromatic faces (6 carbons, 3× the sum,
2–3× the physical libration/sliding stiffness) +1.5 / +1.9 calls
same-basin, with the net gain coming from basin changes. Next: (g)
normalise a face's contact factors to a double bond's worth,
c_i ← c_i·min(1, 2·max c / Σc) per (hydrogen, fragment) — alkenes,
hydrogen bonds and connected molecules bit-identical (cycle 63).
Cycle 63 did (g): keep, but marginal (train −0.0002, valid −0.0005;
aromatic faces −0.13 / +0.85 calls same-basin, ±3–7 per dimer in both
directions) — the aromatic-face endgames are path-sensitive rather than
stiffness-limited; the π line is closed (class share 0.03–0.04).

**Attempts**
- cycle 14, candidate `6cc9adc8604c703313d050e93b5cb18c6a340a78` (diagonal
  ε-form bending term, k_φ 0.15) vs champion
  `9c106c78fda1a4fcf8ec5a5d863bd5a770ca0d30`: discard (train 0.89267 vs
  0.88450). Evidence `evaluation_results/cycle-14-train.json`
  (1b423bbcf946473ca508662153027436); implementation
  `ideas/contact-bend-guess/6cc9adc8604c703313d050e93b5cb18c6a340a78/algo.py`;
  `full_log.md` cycle 14.
- cycle 15, candidate `bb7ede468bab945528fd5010e2cc0ffeb8005471` (full
  rigid-body block `_h0_fragment_block`, stretch + H-bond bending, floor
  1e-3) vs champion `9c106c78fda1a4fcf8ec5a5d863bd5a770ca0d30`:
  non_generalizable (train 0.87726 vs 0.88450, valid 0.90397 vs 0.89361).
  Evidence `evaluation_results/cycle-15-train.json`
  (e703db0a26724480a1a0ab7f9bfc4c15) and
  `evaluation_results/cycle-15-valid.json`
  (d4bc0cd9ba3d4c1395a8b623728aed3d); implementation
  `ideas/contact-bend-guess/bb7ede468bab945528fd5010e2cc0ffeb8005471/algo.py`;
  `full_log.md` cycle 15.
- cycle 45, candidate `21a3273fee7548462d38cebd93dcfe77f0de31b4`
  (tracked bend term `_h0_contact_bends` inside `_h0_nonlocal_contacts`:
  isotropic X–H···Y bend A_φ 0.5, true-angle H···Y–Z bends A_φ 0.15,
  caps 0.05/0.01, ρ_contact ≥ 1e-4; connected systems bit-identical) vs
  champion `6e34e59e16b19c3538a760b9783d28081d2453b3`: **keep** (train
  0.7626995 vs 0.7886692, valid 0.7877605 vs 0.8137677; mean rel_energy
  1.0017 / 1.0092). Evidence `evaluation_results/cycle-45-train.json`
  (00a8a210346045c081c3e015340c04a8) and
  `evaluation_results/cycle-45-valid.json`
  (d69a42428806484db56bd81c07be8501); `full_log.md` cycle 45.
- cycle 48, candidate `4c05263c9a2a4f8b38faae186f0e3c2676311c97`
  (`_h0_contact_bends` over all non-local pairs with ρ_contact ≥ 1e-4;
  intramolecular pairs only with a heteroatom donor, C–H···Y skipped;
  constants unchanged) vs champion `21a3273fee7548462d38cebd93dcfe77f0de31b4`:
  **keep** (train 0.7592625 vs 0.7626995, valid 0.7842329 vs 0.7877605;
  mean rel_energy 1.0017 / 1.0092). Evidence
  `evaluation_results/cycle-48-train.json`
  (1721920d28a849cb8bda5ef4b236266a) and
  `evaluation_results/cycle-48-valid.json`
  (f9fd021384d04095bf3a237938d03ea0); `full_log.md` cycle 48.
- cycle 49, candidate `d148070066a818071b41626af514c0627710334b` (lone-pair-plane
  out-of-plane term for single-neighbour sp² acceptors, A_φ,acc 0.15,
  same contact set) vs champion `4c05263c9a2a4f8b38faae186f0e3c2676311c97`:
  keep (train 0.7532674 vs 0.7592625, valid 0.7715672 vs 0.7842329).
  Evidence `evaluation_results/cycle-49-train.json`
  (01093bbde69249c49e623ce80e3fb8e4), `evaluation_results/cycle-49-valid.json`
  (48ab639ada284e3b949b4eca75c691c1); `full_log.md` cycle 49.
- cycle 62, candidate `09bb577f62847bdc1fd519479f5456eb1b224841` (X–H···π
  contact bends for heteroatom donors between fragments, nearest-heavy-atom
  rule, weak cap 0.01 per unsaturated carbon) vs champion
  `5d0a542040b30ac0f066135252c6bc80f13d072f`: **keep** (train 0.7179724 vs
  0.7193841, valid 0.7256903 vs 0.7270871; mean rel_energy 1.0015 / 1.0078).
  Evidence `evaluation_results/cycle-62-train.json`
  (000623681fcf4d6482eeba0639a97931), `evaluation_results/cycle-62-valid.json`
  (e2a2a308dc4442a7ae10f527a9d5d27d); `full_log.md` cycle 62.
- cycle 63, candidate `a09dbb5d39a63bd0fdb4b4bdb176812d8af2c023` (π-face
  normalisation c_i·min(1, 2·max c/Σc) per (hydrogen, fragment) face) vs
  champion `09bb577f62847bdc1fd519479f5456eb1b224841`: **keep** (train
  0.7177764 vs 0.7179724, valid 0.7252031 vs 0.7256903; energies identical).
  Evidence `evaluation_results/cycle-63-train.json`
  (fbdd2b2272f54450904881c06c34fa1e), `evaluation_results/cycle-63-valid.json`;
  `full_log.md` cycle 63.

## inter-fragment-tracked-contacts: geometry-tracked coupled contact block for the fragment TR coordinates
Status: **kept** via repair 1 (cycle 43, champion `6e34e59e16b19c3538a760b9783d28081d2453b3`: tracked block + symmetry-breaking start, train 0.7886692 / valid 0.8137677)

**Hypothesis** The champion's inter-fragment model (`_h0_fragment`) is a
static, diagonal stretch projection built once at the input geometry: at
the floor for far starts, stale by 1–4 Å / 25–75° for mid starts by the
time the endgame begins. Letting every inter-fragment pair within
r_cov + 3 Å join the tracked Lindh-style pair term of cycle 35
(`_nonlocal_pairs` keeps pairs with different fragment labels;
`_h0_nonlocal_contacts` projects k·uuᵀ through pinv(B) onto all internals;
`_track_nonlocal_contacts` adds A(x) − A(x_prev) before each secant update)
gives the fragment translations/rotations their complete coupled
Gauss–Newton contact block *at the current geometry*, removing the
overshoot/shrink cycles of the approach and the stale-diagonal endgame.

**Outcome and uncertainty** Train C_S 0.79028 vs 0.81964 (−0.0294, the
largest gain of the programme), connected 254/257 bit-identical, dimers
Σ −483 calls (118:65:29), same-endpoint dimers −1.84 calls each (mid start
−2.91, far −1.17), basin changes −4.56 each. Invalid: Σ(rel−1) +0.414 →
−0.382. The whole swing sits in the 30 mirror-symmetric-start dimers
(+0.757 → −0.019): the reference and the candidate converge onto the
symmetric stationary point — often a saddle whose soft antisymmetric mode
carries a force below the gate at any displacement the run reaches —
while the champion's floor-soft/stale TR model spends 25–40 steps there
and amplifies the numerical asymmetry by ≈ 1 + |κ|/λ_floor per step,
leaving the saddle for the lower minimum (esters__monoatomics −0.248,
carboxylates__water −0.203, thiols__water −0.043; slide-past to a second
symmetric basin in carboxylates__ethers −0.356 and ketones__monoatomics
−0.151). The valid margin of the champion (+4.378) is likewise +3.774 from
29 symmetric-start dimers. Uncertainty: the candidate was not run on
valid; the slide-past cases may be reachable with larger approach steps
rather than symmetry breaking.

**Reason to revisit** The call gain is real, deterministic and general
(mid and far starts, same endpoints); only the energy lottery that the
champion's lineage accumulated at symmetric starts blocks it. Any clean
dimer optimizer will hit the same wall, so the repair — a general
saddle-avoidance mechanism — is a prerequisite for every future dimer
improvement, not just this one.

**Next experiment** Repair 1 (cycle 43, done, keep): deterministic
symmetry-breaking rigid-body displacement of the fragments of a
multi-fragment start (0.02 Å / 0.02 rad per fragment, seeded RNG, lone
atoms translation only) on top of the tracked block — see
[symmetry-breaking-start]. A larger trust cap with a Cartesian bound for
multi-fragment systems on top of the block was tested in cycle 44
(`trust-region-textbook`): discard, far starts and walkers +3…+6 calls,
near-equilibrium unchanged — the approach steps on the floor-diagonal far
field are steepest-descent-like in the rigid-body coordinates and a longer
step overshoots the orientation. Follow-ups on the block itself: far-field
curvature for the rigid-body coordinates (a directional long-range term
so the approach step points at the basin, e.g. an orientation term from
the contact geometry or a rotation floor scaled with the fragment's
geometric inertia); positive part of the tracked shift A(x) − A(x_prev)
only. The endgame intercept turned out to have a mechanism after all:
cycle 45 added the tracked H-bond *bending* curvature to this block
([contact-bend-guess], keep, −0.026 on both splits; same-endpoint
near-equilibrium H-bonded/ionic pairs −1.4…−2.9 calls). Remaining
directional terms: acceptor lone-pair direction, metal-cation contacts.

**Attempts**
- cycle 42, candidate `af40f3ce6f62182c304c4eb3f863f1ec661f5035` vs
  champion `94ea9fe5a46f07ba802f25ef882e33de49f54fb1`: invalid
  (train 0.7902817 vs 0.8196408, mean rel_energy 0.9991851,
  `energy_below_baseline`). Evidence `evaluation_results/cycle-42-train.json`
  (f0f79775c1614abba31a0026e6924566); implementation
  `ideas/inter-fragment-tracked-contacts/af40f3ce6f62182c304c4eb3f863f1ec661f5035/algo.py`;
  `full_log.md` cycle 42.
- cycle 43 (repair 1), candidate `6e34e59e16b19c3538a760b9783d28081d2453b3`
  vs champion `94ea9fe5a46f07ba802f25ef882e33de49f54fb1`: **keep**
  (train 0.7886692 vs 0.8196408, mean rel_energy 1.0023758; valid
  0.8137677 vs 0.8326029, mean rel_energy 1.0091846). Evidence
  `evaluation_results/cycle-43-train.json` (3635f19e5c9d48b0af35d1418f10a633),
  `evaluation_results/cycle-43-valid.json` (bc7d7c45f8154808b5e8fb23e33dd0f0);
  `full_log.md` cycle 43.
- cycle 86 (cross-reference) — the seed is undone for docked starts by the L-BFGS-B docking (surrogate minima are
  re-symmetrised to ≲ 1e-3 Å); re-applying it to the docked pose was tested as [docked-pose-symmetry-breaking]
  and discarded (train +0.0021: the kick along the soft antisymmetric modes of symmetric docked *minima* costs
  ≈ 10 endgame steps that no approach walk hides; one saddle-stayer fixed at the reference's call count).

## symmetry-breaking-start: deterministic rigid-body displacement of every fragment of a multi-fragment start
Status: **kept** (cycle 43, champion `6e34e59e16b19c3538a760b9783d28081d2453b3`, together with [inter-fragment-tracked-contacts])

**Hypothesis** DES370K starts that lie on a symmetry element (ion on the
bisector of two equivalent donors, molecule in its partner's mirror
plane) have an identically zero gradient along the symmetry-breaking
modes, so the reference and any clean optimizer converge onto the
symmetric stationary point, often a saddle whose antisymmetric force
|κ|δ stays below the 4.5e-4 Ha/Bohr gate. The standard remedy (ASE
`rattle`, VASP/CP2K practice) is a small displacement off the element
before the first force call: every fragment (Sella's 1.25·r_cov bond
pass, lone atoms as one-atom fragments) is translated 0.02 Å along a
fixed-seed random direction and rotated 0.02 rad about a random axis
through its centroid (`_start_fragments`, `_break_start_symmetry`,
`np.random.RandomState(0)`); connected systems are returned unchanged.
The seed is amplified by ≈ 1 + |κ|/λ_model per step and leaves saddles
with |κ| ≳ 0.03 eV/Å² within a 10–15-step run.

**Outcome and uncertainty** With the tracked block (cycle 42 code):
train 0.7886692 (−0.0310) / valid 0.8137677 (−0.0188), both valid;
symmetric-start credits restored (train +0.756 vs the champion's +0.757;
valid +3.874 vs +3.774), the calls on the symmetric-start dimers
themselves unchanged (valid 28.6 vs 28.7). Connected molecules
bit-identical. Uncertainty: the magnitude (0.02 Å / 0.02 rad) was set
once from the amplification estimate and is *not* to be scanned (a seed
scan would be a basin lottery, not research); the slide-past credits
(carboxylates__ethers, ketones__monoatomics train; carboxylates__ketones
valid) remain path lotteries that a shorter approach may re-roll.

**Reason to revisit** Only if a future candidate loses the symmetric
credits although its runs are longer than ≈ 10 steps (would indicate the
seed is damped during the approach and a rotation-only or larger seed is
needed), or to extend the idea to connected systems that start on a
symmetry element (not observed to matter: the connected margin is
+0.05 / −0.01).

**Next experiment** None planned. Diagnostic on any dimer-path-changing
candidate: recompute the symmetric-start credit decomposition from the
saved results (`symflag` tolerance 0.02 Å on two equivalent closest
contacts) before interpreting an energy-gate failure.

**Attempts**
- cycle 43, candidate `6e34e59e16b19c3538a760b9783d28081d2453b3` vs
  champion `94ea9fe5a46f07ba802f25ef882e33de49f54fb1`: **keep** (train
  0.7886692 vs 0.8196408, mean rel_energy 1.0023758; valid 0.8137677 vs
  0.8326029, mean rel_energy 1.0091846). Evidence
  `evaluation_results/cycle-43-train.json` (3635f19e5c9d48b0af35d1418f10a633),
  `evaluation_results/cycle-43-valid.json` (bc7d7c45f8154808b5e8fb23e33dd0f0);
  `full_log.md` cycle 43.

## energy-secant: energy-augmented (modified) secant condition for the TS-BFGS update

Status: closed after four attempts (cycles 16, 17, 50, 51 — all discard on
train; implementations preserved in `ideas/energy-secant/`). The design
space from "every step" (clip) to "only persistent model-limited creeps"
(cycle 51) shows no systematic same-basin gain anywhere.

**Hypothesis** The plain secant condition B s = y fixes the model curvature
along the step to the *average* curvature over the step, i.e. half a step
behind the current point. With the end-point energies the curvature at the
new point is available to third order (Zhang–Deng–Chen 1999; Wei–Li–Qi
2006): y* = y + θ s/(s·s), θ = 6(f0 − f1) + 3(g0 + g1)·s, exact for cubic
potentials (verified analytically: s·y* = G(x1)s²). On anharmonic modes
(Morse stretches, H-bond/salt-bridge approach coordinates) a model that is
locally exact at the current point should save ≈ 1 step per anharmonic
mode chain (secant order 1.62 → ≈ 2.4 in 1D). Implemented in `PES.kick`
via `PES._secant_dg` with g0 = the transported gradient `g_par`, s =
`dx_final`, θ clipped to ±0.5|s·y| (sign preserved).

**Outcome and uncertainty** Broad, systematic degradation: train C_S
0.8982 (cycle 16) and 0.9040 (cycle 17, with an energy-resolution floor
|s·y| ≥ 3e-4 eV that skips the endgame corrections) vs champion 0.8845.
Same-basin mean +0.75 / +0.91 calls per molecule, 194–200 worse vs
117–119 better, all three sources worse (PubChem 0.932 → 0.944, dimers
0.833 → 0.843/0.855, rowansci 0.914 → 0.966/0.970), and the loss is
carried by many small degradations (|d| < 8 contributes +0.017–0.019 of
the +0.023–0.027 same-basin C_S loss), not by a few outliers. The floor
made it slightly *worse*, so the endgame corrections (small |s·y|) were
neutral-to-helpful and the approach/mid-phase corrections are the harmful
part; the energy-noise explanation for cycle 16 is therefore refuted.
Side effect worth remembering: the corrected model stiffened the slide
coordinate of six stretched ionic dimers so that they stopped at the
stretched geometry like the reference (ammoniums__esters 58 → 15,
amides__carboxylates 42 → 15, carboxylates__water 44 → 20 (+12 kJ/mol),
esters__ethers 36 → 17, acids__guanidiniums 32 → 21,
carboxylates__guanidiniums 29 → 17) — a large call gain paid for with
energy-gate credit; the mean rel_energy still rose (1.0013/1.0017) through
unrelated deeper basins (monoatomics__pyrrole −11 kJ/mol, venetoclax −6).
Gains concentrated on deep H-bonded dimers (imidazolium__ketones 58 → 47,
acids__ketones 43 → 33, esters__thiols 40 → 30, ketones__phenol 60 → 50,
acids__acids 41 → 33). Uncertainty: the cause of the broad mid-phase harm
was not identified (candidates: inconsistency of end-point curvatures with
the interval-average curvatures of the older pairs in the multi-secant
block; rank-1 curvature changes of up to 50 % along trust-limited approach
steps where the cubic expansion is invalid; the next RFO step needing the
average curvature over the *next* interval, for which the interval average
of the last step is the better predictor on the compressed side of a
Morse-like mode).

**Cycle 50 re-analysis and outcome** The saved 200-call trace of 8030412
shows consecutive xTB energy changes decreasing monotonically to 1e-11 Ha
without sign flips — the energies are consistent to ≤ 1e-10 Ha, so energy
noise is not a factor at any stage. A paired class re-analysis of cycle 16
(`/tmp/paired_cls.py cycle-16 cycle-13`) does NOT support the earlier
reading that its gains sat on carbonyl-acceptor dimers: the acc-O nY=2
class was +3.9 calls (2 better / 11 worse), the nY=1 carbonyl class +0.85
(3 / 7); only four individual dimers gained. Cycle 50 therefore gated the
correction by theory instead of a clip: applied only when the step
continues the previous one (ŝ_prev·s₀ > 0; after a reversal the next
interval overlaps the last one and the interval average is the better
predictor — on a quartic bounce the corrected model contracts at 0.67
instead of 0.24) and only when 1/2 ≤ 1 + θ/(s·y) ≤ 2 (θ/s·y is −0.27 …
−0.37 on quartic creep, +0.8 on a cosine torsion, +8 / −1 across
inflection points). Result: train 0.7696717 vs 0.7532674 (discard),
469/469 converged, energy margin 0.913 vs 0.816. Paired vs cycle 49:
same-basin +0.13 calls (128 better / 137 worse / 192 identical, dC_S
+0.0052), basin changes +0.0112 (acids__esters 11 → 51 into a 1.6 kJ/mol
deeper basin, amides__ammoniums 27 → 54, …). The predicted creep classes
did gain — single-carbonyl-acceptor dimers −0.79 (5 / 2, n 14), "≥ 2 H to
one Y" −1.56 (6 / 1), "≥ 2 H-bonds" −1.29 (7 / 4), S acceptors −0.67 —
while dispersion-bound dimers lost +0.78 (8 / 18, n 37), two-neighbour O
acceptors +1.43 (4 / 8) and, robustly, the short connected runs (champion
< 12 calls: 4 better / 20 worse, mean +0.35; the cycle-47 signature of
early-step information persisting in the memory-4 block). Reading: the
retrospective trend test admits (i) the second/third relaxation step of
every short run and (ii) the overshoot step that closes a trend (g1·s > 0),
after which the next step reverses and the interval average is the better
predictor. Uncertainty: the gate fired rarely (192 identical), class
gains are n 9–14 and within the lottery; the 4 : 20 short-connected sign
is the one robust signal.

**Reason to revisit** The end-point-curvature idea is not refuted for the
creep regime (dimer endgames on flat contact modes); the harm comes from
relaxation-phase and overshoot corrections. Beyond the repair below, (a)
the sign-restricted variant y* = y + max(θ, 0) s/(s·s) (Yuan–Wei 2010) is
the wrong sign for creep (creep needs θ < 0), so it is dropped; (b)
energy-consistent step acceptance remains a separate idea
([[step-rejection]]).

**Cycle 51 outcome (repair 1)** Creep detector — model's own step
(`smag < delta`, plumbed from `Sella.step` as `pes._step_restricted`),
continues the previous step, df < 0, g0·s < 0, g1·s < 0, previous step
qualified the same way (`_creep_run ≥ 2`), c > 0, factor-2 window: train
0.7564724 (discard, +0.0032). 271 runs bit-identical; same-basin +0.0004
(36 : 32); the whole loss is one near-degenerate path change
(ammoniums__esters 27 → 52, −0.12 kJ/mol). The short-connected harm is
gone (3 : 1), so the repair hypothesis about relaxation steps was right,
but the creep classes did not gain (carbonyl 2 : 4, "≥ 2 H to one Y"
0 : 1, dimers 15 : 17 overall) and the large moves (−8, −7, +7, +5) have
random sign. Conclusion: the consecutive model-limited undershoots that
the detector finds in 40 % of runs are not stale 1-D curvature along a
flattening mode but a rotating step direction (subspace exploration —
each step's new component is in a direction the model has not sampled);
end-point curvature along the previous step cannot shorten that.

**Next experiment** None. If the line is ever reopened, the only untested
variant is applying the correction as an extra rank-1 term outside the
stored pairs (avoids the consistency-filter shift of stored y*), and only
with a creep diagnosis based on traces rather than saved endpoints.

**Attempts**
- cycle 16, candidate `7c0c154d68064fd1fe839244bf17fbfd2f376b5b` (clip
  ±0.5|s·y|, applied on every step) vs champion
  `9c106c78fda1a4fcf8ec5a5d863bd5a770ca0d30`: discard (train 0.8982460 vs
  0.8844994; energy 1.001287). Evidence
  `evaluation_results/cycle-16-train.json`
  (346b3366bd684533b9a9ba5d6d40f724); `full_log.md` cycle 16;
  implementation `ideas/energy-secant/7c0c154d68064fd1fe839244bf17fbfd2f376b5b/algo.py`.
- cycle 17, repair 1 `0eddb4a1ed75dba8f4dd72a8b006172d4ccf4e6e` (resolution
  floor |s·y| ≥ 3e-4 eV) vs the same champion: discard (train 0.9040092;
  energy 1.001738). Evidence `evaluation_results/cycle-17-train.json`
  (e4bfc2fb50da4d709cd4122c13fced9d); `full_log.md` cycle 17;
  implementation `ideas/energy-secant/0eddb4a1ed75dba8f4dd72a8b006172d4ccf4e6e/algo.py`.
- cycle 50, candidate `89ccf3494f442540c8781f2f6f15e5845d41e294` (trend
  gate ŝ_prev·s₀ > 0 + factor-2 credibility window, no clip) vs champion
  `d148070066a818071b41626af514c0627710334b`: discard (train 0.7696717 vs
  0.7532674; energy 1.0019465; same-basin +0.0052, basin changes +0.0112).
  Evidence `evaluation_results/cycle-50-train.json`
  (8ed606dee2e34c6b8f563db2c9be4385); `full_log.md` cycle 50;
  implementation `ideas/energy-secant/89ccf3494f442540c8781f2f6f15e5845d41e294/algo.py`.
- cycle 51, repair 1 `b2bce1e5d2ff39f00b9cd28c072fc83797651c0c` (persistent
  model-limited creep gate + window) vs champion
  `d148070066a818071b41626af514c0627710334b`: discard (train 0.7564724 vs
  0.7532674; energy 1.0017450; same-basin +0.0004, one path change
  +0.0028). Evidence `evaluation_results/cycle-51-train.json`
  (bc8cc2b69aa44d5fbad02c44abab5241); `full_log.md` cycle 51;
  implementation `ideas/energy-secant/b2bce1e5d2ff39f00b9cd28c072fc83797651c0c/algo.py`.

## step-rejection: undo uphill steps (geomeTRIC's ρ < −1 rule) from the cached previous point

Status: rejected after cycles 23–26 (cycle 23 discard, repairs 24/25 discard,
repair 26 non_generalizable); implementations preserved in
`ideas/step-rejection/af7101c283b901b6e22022dc6ecc251df8adf303/algo.py`
(plain rule) and
`ideas/step-rejection/6951d9e71fdd619a7ffcb906011d04cb61366479/algo.py`
(size guard 0.05 Å/rad, rise threshold 0.05 eV, connected systems only).

**Hypothesis** Sella accepts every step; when the energy rose by more than
the predicted drop (ρ < −1) the accepted uphill point can start a detour
(clash region, other basin). Undoing the step costs no force call
(`PES.reject_step`: restore the saved positions, `curr = last.copy()`,
state-hash cache), the Hessian keeps the failed pair, the radius is halved.

**Outcome and uncertainty** Plain rule: train 0.8757 vs 0.8658 (43 better /
101 worse, small losses +1…+5 everywhere, few large wins). Size guard
(cycle 24, 0.8733): changes 8 molecules — rejections happen in the mid
phase, not the endgame. Rise threshold 0.05 eV (cycle 25, 0.8666): the
big dimer wins vanish (they were shallow-rise path re-rolls), the large
flexible rowansci molecules win 4/4 (−18 calls), PubChem 7/10, dimers
23/38. Connected-only (cycle 26): train 0.8644 (+0.0014, passes) but valid
0.8855 vs 0.8830 (PubChem 13 better / 23 worse, dimers identical) →
non_generalizable. Reading: with the secant condition the next model
minimum is the same from the undone and from the uphill point
(x₁ − B′⁻¹g₁ = x₀ − B′⁻¹g₀), so the undo buys nothing on locally quadratic
surfaces and wastes the call, while the accepted path's fresh gradient
and its generous next ρ (measured from the high point) regrow the radius
faster. Uncertain: whether undos cascade on anharmonic walls (the +4
clusters), and whether a blow-up-only threshold (≥ 0.2 eV) is neutral or
slightly positive; per-molecule spread ±10 calls on valid makes the
expected gain sub-gate.

**Reason to revisit** Only with trajectory-level evidence (which events
cascade, what the undone points looked like) or together with a
line-search-style retry (cubic interpolation along the failed step using
E₀, g₀, E₁, g₁ instead of the halved-radius QN retry), which is the one
variant that uses information the accepted path does not.

**Next experiment** If reopened: blow-up-only rule (rise ≥ 0.2 eV, connected
systems) as a single cheap cycle; expect ≤ 20 molecules changed.

**Attempts**
- cycle 23 / `af7101c283b901b6e22022dc6ecc251df8adf303` vs champion
  `a277fc1cbf2ebec46359133c7228210d71338a9d` — discard (train 0.8756767 vs
  0.8658181); `evaluation_results/cycle-23-train.json`
  (5a7b1cd1ba314923bfcf4dbe41548eca).
- cycle 24 / `abf5ef6c2dd795255e1114962e1140eca7af2bdb` — discard (train
  0.8733385); `evaluation_results/cycle-24-train.json`
  (fcbee25f6bc6472a90bb9ec4e43d0cb7).
- cycle 25 / `560cf373b7bf0774db25c775533699b444295382` — discard (train
  0.8665735); `evaluation_results/cycle-25-train.json`
  (a2eb96c9ef22410388429286eefd5799).
- cycle 26 / `6951d9e71fdd619a7ffcb906011d04cb61366479` — non_generalizable
  (train 0.8643830, valid 0.8855485 vs 0.8829537);
  `evaluation_results/cycle-26-train.json`
  (9a6fd90369b44366a629a4d7a0f48fec),
  `evaluation_results/cycle-26-valid.json`
  (0b7ab797eb57406890f153e2153e029f).

## cartesian-trust-region: internal trust region that also bounds the linearised atomic displacement (Binv @ s)
Status: accepted (cycle 28, champion `291e48613748fb810ddabc67fb393ee65f2946a3`); ratio 1.5 non_generalizable (cycle 29) and cap 0.75 discard (cycle 32) — flat optimum; dimer cap extension closed (cycle 44); multi-fragment δ₀ = 0.25 discard (cycle 60, +0.0010: same-basin −0.0047 masked by two near-equilibrium basin escapes) — the δ₀ line is closed for both system classes

**Hypothesis:** the max-internal-component radius is blind to lever arms — 0.25 rad
changes of several dihedrals along a chain compound to multi-Å swings of distal groups
(cycle 27: rivaroxaban's Cl moved 3.33 Å in the first step, min non-bonded distance
0.86 Å, SCC failure). Bounding the largest atomic norm of `Binv @ s` (the minimum-norm
Cartesian realisation of the step, real atoms) by `cart_ratio · δ` makes a larger initial
radius (δ₀ = 0.25, connected systems) safe, so the far-from-minimum molecules reach the
0.5 cap in two accepted steps instead of four.
**Outcome and uncertainty:** cycle 27 (δ₀ = 0.25 alone) invalid (one SCC failure);
cycle 28 (δ₀ = 0.25 + Cartesian bound 2δ, connected systems) keep: train 0.8547638
(−0.01105), valid 0.8654398 (−0.01751); connected molecules −0.51 / −0.88 calls, gains
grow with reference length (ref ≥ 40: −5.75 / −3.12), dimers net 0 (7/12 changed through
mid-run internal rebuilds). The two mechanisms were not separated; ratio 2 and the scoping
were not scanned; a few molecules doubled their calls (path changes).
**Reason to revisit:** the same blind spot exists for multi-fragment systems (fragment
rotations of 0.25 rad move atoms by ~1 Å), where the TR cap 0.25 was chosen empirically
(cycle 22) — a Cartesian cap may allow a larger TR radius; the ratio trades long-lever
gains against path-change losses.
**Next experiment:** (1) ratio scan done (1.5 neutral; 3 not worth a cycle) and cap scan done (0.75 worse; 0.35 not worth a cycle); (2) extend the bound to
systems with fragment TR internals (`cart_ratio` also when `_has_tr_internals()`, possibly
with a larger TR cap) — done in cycle 44 (discard); (3) control:
Cartesian bound with δ₀ = 0.1 (separates the two mechanisms) — not run, the champion's
connected-system gains have since been re-based by the cycle-53 model change; (4) the
multi-fragment δ₀ control (0.25 = the cap, no ramp) — done in cycle 60 (discard).
Cycle-60 reading: multi-fragment same-basin runs −0.40 calls (192 molecules, −0.0047 of
C_S; near-equilibrium starts only −0.17, mid-range 1–2.5 Å starts −0.52, far starts
−0.33), i.e. the ramp costs less than one step even where the model directs the first
steps; the score rose because two near-equilibrium H-bonded starts (acids__esters
tr 0.47 Å, esters__guanidiniums 0.91 Å) left the reference basin for minima 0.2–0.4
kcal/mol deeper and converged in 38/31 calls instead of 9/11 (+0.0069 of C_S), which
the 0.1 Å first step had not done — the larger first step is itself a basin re-roll
for near-equilibrium starts, so the control cannot be separated from the energy
lottery by a single split. No intermediate value will be scanned (no-scan rule).

**Attempts:**
- Cycle 27 (δ₀ = 0.25 for connected systems, no Cartesian bound); candidate `73ea9a9c801d20e8a18ac7150d0cac1ead307346`; champion `a277fc1cbf2ebec46359133c7228210d71338a9d`; decision invalid (rivaroxaban SCC failure at the first step; 468/469 converged).
  [Training evidence](evaluation_results/cycle-27-train.json) (d3c74a7dc5054f1d84196935315088f9); diagnostic `diagnostics/d3c74a7dc5054f1d84196935315088f9/cef263ed6998aff73c7c3f60b0c657fad840ac32645bc44c2f01c3a6fbaede87.json`.
  [Narrative](full_log.md#cycle-27--initial-trust-radius-025-årad-for-connected-systems-invalid).
- Cycle 28 (repair 1: Cartesian bound 2δ); candidate `291e48613748fb810ddabc67fb393ee65f2946a3`; champion `a277fc1cbf2ebec46359133c7228210d71338a9d`; decision keep (train 0.8547638, valid 0.8654398).
  [Training evidence](evaluation_results/cycle-28-train.json) (7e98ad43caf041dd9ffb3945b180eb78); [validation evidence](evaluation_results/cycle-28-valid.json) (5e4cdc96837545a3b7f8910a7d250ee7).
  [Narrative](full_log.md#cycle-28--cycle-27-repair-1-cartesian-bound-2δ-å-on-the-internal-trust-region-keep).
- Cycle 29 (ratio 1.5); candidate `a25ea9c7953de0d518e145e57cbc82644630e22d`; champion `291e48613748fb810ddabc67fb393ee65f2946a3`; decision non_generalizable (train 0.8522352, valid 0.8675452; ±0.1 calls per connected molecule, opposite signs on the splits — the ratio sits on a flat optimum).
  [Training evidence](evaluation_results/cycle-29-train.json) (315b6aedf0844856959ac265fbcc2843); [validation evidence](evaluation_results/cycle-29-valid.json) (d218791111294ea5a3db2da61b35cf35).
  [Narrative](full_log.md#cycle-29--cartesian-bound-ratio-15-non_generalizable).
- Cycle 32 (connected-system cap 0.75); candidate `290dae279c8574f15d757ddb5ee1b6f9450264b8`; champion `291e48613748fb810ddabc67fb393ee65f2946a3`; decision discard (train 0.8574170; PubChem +0.16 calls, 16 better / 24 worse, dimers identical).
  [Training evidence](evaluation_results/cycle-32-train.json) (5d9c6aedcaca4a43adf7559ffaf01932).
  [Narrative](full_log.md#cycle-32--trust-radius-cap-075-årad-for-connected-systems-discard).
- Cycle 60 (δ₀ = 0.25 for multi-fragment systems too, i.e. start at the cap δ_max_tr); candidate `af12ce643644e02c49891dbd2dc93071ca5d02ef`; champion `5d0a542040b30ac0f066135252c6bc80f13d072f`; decision discard (train 0.7203612 vs 0.7193841, Δ +0.0010, mean rel energy 1.00208; connected 257/257 identical; multi-fragment 102 better / 71 worse / 39 identical, same-basin −0.0047, basin changes +0.0057).
  [Implementation](ideas/cartesian-trust-region/af12ce643644e02c49891dbd2dc93071ca5d02ef/algo.py).
  [Training evidence](evaluation_results/cycle-60-train.json) (bdbc981212b447128fc20cfcd3f7c488).
  [Narrative](full_log.md#cycle-60--initial-trust-radius-025-årad-for-multi-fragment-systems-discard).

## adaptive-type-scale: per-type curvature scale of the model Hessian from the newest secant pair
Status: rejected after cycles 30–31 (both discard on train)

**Hypothesis:** the diagonal guess is off by a molecule-dependent factor per coordinate type
(cycles 9–11: ±3–4 % from global type scales with opposite optima for small/large
molecules); the newest secant pair, decomposed by type, gives r_t = (s_t·y_t)/(s_t·(Hs)_t),
and H ← D H D with D = diag(√(r_t^p)) extrapolates the correction to the unexplored
directions of the type before the multi-secant update re-imposes the recent pairs.
**Outcome and uncertainty:** five damped updates (p = 0.5, clip [0.25, 4], ≥ 2 % energy
share, connected systems): train 0.8695 vs 0.8548 (PubChem +1.1 calls, 83:113
better:worse); first update only: 0.8605 (PubChem +0.5, 79:86, neutral on < 15 and > 40
atoms, worse in between); both lose the reactive borane 104079126's deep basin (+95
calls). Contradicted: model errors along a step are not shared by the other coordinates
of the same type, and the scaling distorts curvature already learned in other directions.
**Reason to revisit:** only with trajectory-level evidence of type-correlated errors, or as
a least-squares fit over several endgame pairs (small, quadratic steps) instead of the
anharmonic first steps.
**Next experiment:** none planned.

**Attempts:**
- Cycle 30; candidate `d1d43183addc53f9c1c9c6bc96c3e7867972dd94`; champion `291e48613748fb810ddabc67fb393ee65f2946a3`; decision discard (train 0.8695090).
  [Implementation](ideas/adaptive-type-scale/d1d43183addc53f9c1c9c6bc96c3e7867972dd94/algo.py).
  [Training evidence](evaluation_results/cycle-30-train.json) (44747228656a4add8236f89a950b7e7f).
  [Narrative](full_log.md#cycle-30--adaptive-per-type-curvature-scale-of-the-model-hessian-discard).
- Cycle 31 (repair 1: first update only); candidate `e783e4be52ab750cb28b15b15681d74edb953270`; champion `291e48613748fb810ddabc67fb393ee65f2946a3`; decision discard (train 0.8605085).
  [Training evidence](evaluation_results/cycle-31-train.json) (2ec64f33dab649c2854671f550f72a0f).
  [Narrative](full_log.md#cycle-31--adaptive-type-scale-repair-1-first-secant-update-only-discard).

## bofill-sr1-update: safeguarded multi-secant Bofill (SR1/PSB) update instead of TS-BFGS
Status: rejected (cycle 33, decisive)

**Hypothesis:** SR1 (blended with PSB by Bofill's φ) is the textbook trust-region
minimisation update, gives more accurate Hessians than BFGS-type updates and
reproduces a quadratic Hessian after n independent steps; a self-correcting model
should shorten the soft-mode endgame. Block form on the eigen-directions of JᵀS with
dual vectors keeps the multi-secant condition exact and bounds the SR1 term.
**Outcome and uncertainty:** train C_S 1.448 vs 0.855: 90 % of molecules slower
(median ×1.6), the loss concentrated in the approach phase of far-from-minimum
connected molecules (×1.7–1.9) and in dimers (25 → 39 calls); near-minimum molecules
only ×1.12. Together with the BFGS result (cycle 5, 1.055) this shows that TS-BFGS's
|B|-weighted Powell form — which keeps the stiff block close to the guess and spreads
the correction in the B metric — is essential with the large, anharmonic,
frame-transported secant pairs of trust-limited internal-coordinate steps. Not an
implementation defect (single-pair limit = Bofill; block secant condition exact; no
run failed).
**Reason to revisit:** only if the secant data become cleaner (small steps, e.g. an
endgame-only switch after the gradient falls below the force threshold, where the
×1.12 loss suggests near-neutrality) — low priority.
**Next experiment:** none planned; a possible endgame-only SR1 switch would need a
mechanism that is not convergence damping.

**Attempts:**
- Cycle 33; candidate `8cc5bca9ae123781cd0f5dee7b4319489987ea36`; champion `291e48613748fb810ddabc67fb393ee65f2946a3`; decision discard (train 1.4483809, energy 1.003991).
  [Implementation](ideas/bofill-sr1-update/8cc5bca9ae123781cd0f5dee7b4319489987ea36/algo.py).
  [Training evidence](evaluation_results/cycle-33-train.json) (4c9a07c63b404740827e2b22ebcfa673).
  [Narrative](full_log.md#cycle-33--safeguarded-multi-secant-bofill-sr1psb-hessian-update-discard).

## nonlocal-contact-guess: Lindh-style all-pair curvature of non-local intramolecular contacts in the guess Hessian
Status: adopted (cycle 34 `eacac3d65ab9dc5b36840c4d9c3ff0624a61531d`; geometry-tracked form cycle 35, champion `fc3407d8ecc67de5fe3a9a7ad64eaf6d6f3bad56`; vicinal 1-4 pairs of connected systems cycle 65, champion `2a8fb394fa396f8111c886bffafdaa97b190f84f`); follow-ups mostly closed

**Hypothesis:** the diagonal internal guess carries only 1-2/1-3/1-4 curvature; the
steric/hydrogen-bond curvature of the contacts that hold a fold together (graph
distance ≥ 5, 2–4 Å apart) is missing, so the model is too soft along the torsion
combinations that compress them. Adding the Fischer–Almlöf stretch curvature of every
such same-fragment pair as a Cartesian pair term mapped onto the internals
(H_nb = DᵀKD, D = u·(B⁺_j − B⁺_i)) supplies the off-diagonal information.
**Outcome and uncertainty:** train 0.8469 (Δ −0.0079), valid 0.8524 (Δ −0.0130);
folded molecules (≥ 0.3 qualifying pairs/atom) rel 0.90 → 0.84 with 3–7 : 1
better : worse, dimers/small molecules untouched. One basin re-roll (104079126,
29 → 142 calls, −130 kJ/mol) costs +0.006 C_S on train. Cycle 35 made the term
geometry dependent (H ← H + A(x) − A(x_prev) before every secant update, using
the stepper's cached pseudo-inverse Jacobian): train 0.8325 (Δ −0.0144), valid
0.8397 (Δ −0.0127); connected molecules moving 0.2–0.8 Å gained 0.3–0.8 calls
(20–38 : 4–15 better : worse), the far-moving valid class −0.84 calls, the
near-minimum class and the dimers unchanged; two train folding molecules took
long detours (363892164 +16, 384221749 +14 calls — indefinite shift when a
contact breaks, or interaction with the re-imposed secant pairs). 104079126
returned to the reference basin (train energy margin now Σ(r−1) ≈ 0.31).
Cycle 65 added the 1-4 pairs (graph distance 3, radial term only, no contact
bends, connected systems only, geometry-tracked and transported like the
non-local pairs): train −0.00145 / valid −0.00149 with both controls
(multi-fragment; connected without a 1-4 pair in range) bit-identical;
same-basin Σ −23 / −5 calls at 53:46 / 69:68 better:worse, 4 + 3 basin
changes, no monotone trend with the 1,4/F–A stiffness ratio (the S/P-rich
rotors with ratio ≥ 0.5 gave −10 / 0 calls) — a small, lottery-like gain, not a
class effect. Done since: angular part (contact bends, cycles 45/48/49/62/63),
tracking of the diagonal (cycle 46, wash), damped shift (superseded by the
cycle-53 transport). Untested: 1-5 pairs (`min_path=4`; would add ≈ +11 % to
butane and +6 % to propane via the two-bond lever on bends that the GF fit
already shows 1.2–1.3× over-stiff — expected adverse), a global scale on k
(a parameter scan, not to be done), and the same construction for
inter-fragment pairs (would couple the fragment TR coordinates with the
torsions and replace the diagonal `_h0_fragment` block; dimer re-roll risk).
**Reason to revisit:** the mechanism is confirmed and cheap; each follow-up is a
one-parameter or one-block change with a clear physical reading.
**Next experiment:** only the inter-fragment pair term for dimers remains
plausible (risk: re-rolls, cycles 13–15, and the train energy credit of
monoatomics__pyrrole); 1-5 pairs are expected adverse (see above).

**Attempts:**
- Cycle 34; candidate `eacac3d65ab9dc5b36840c4d9c3ff0624a61531d`; champion `291e48613748fb810ddabc67fb393ee65f2946a3`; decision keep (train 0.8468548 / valid 0.8524317; energy 1.001310 / 1.009560).
  [Training evidence](evaluation_results/cycle-34-train.json) (f1a90657e8694b3c986c57a17085d4b9).
  [Validation evidence](evaluation_results/cycle-34-valid.json) (f70f297b74dd477f81a19078dedf7131).
  [Narrative](full_log.md#cycle-34--non-local-contact-curvature-in-the-guess-hessian-lindh-style-pair-term-projected-onto-the-internals-keep).
- Cycle 35; candidate `fc3407d8ecc67de5fe3a9a7ad64eaf6d6f3bad56`; champion `eacac3d65ab9dc5b36840c4d9c3ff0624a61531d`; decision keep (train 0.8324971 / valid 0.8397486; energy 1.000665 / 1.009417).
  [Training evidence](evaluation_results/cycle-35-train.json) (7251b95c87de49f9838df92da65ee06f).
  [Validation evidence](evaluation_results/cycle-35-valid.json) (93e6f09374ac44cc8dc0461332ad3a64).
  [Narrative](full_log.md#cycle-35--geometry-tracked-non-local-contact-term-h--h--ax--ax_prev-before-every-secant-update).
- Cycle 65; candidate `2a8fb394fa396f8111c886bffafdaa97b190f84f`; champion `6c6efdf2e03120e3dbb7daebae4ba6baa2f73377`; decision keep (train 0.7080763 / valid 0.7210370; energy 1.0014082 / 1.0078047).
  [Training evidence](evaluation_results/cycle-65-train.json) (15ffa629bc374190ae4b401e48ad349d).
  [Validation evidence](evaluation_results/cycle-65-valid.json) (b353d518b734452383d219bd7b15ed63).
  [Narrative](full_log.md#cycle-65--vicinal-14-pair-curvature-in-the-geometry-tracked-contact-term-of-connected-systems-keep).

## stretch-row-factor: row-dependent prefactor of the Fischer–Almlöf stretch guess
Status: rejected (non_generalizable, cycle 36); preserved at `ideas/stretch-row-factor/2326fbaf173207b1b6755f6bc6dc3f92d88271a6/algo.py`
Superseded: cycle 64 (`heavy-row-stiffness`, factors 0.58/0.50/0.42/0.35, no boron factor, connected systems only) kept with −0.0082 / −0.0027 — the condition named under "Reason to revisit" was met by cycle 53's geometry-tracked stretch diagonal and transported secant pairs, which keep the guess diagonal in force beyond the first step instead of letting the update override it.

**Hypothesis:** the stretch guess `0.3601·exp(−1.944(r − r_cov))` was fitted to
first-row molecules; measured valence stretch constants of bonds to S/P/Cl/Si
(0.54–0.66×), Br (0.5×), I (0.4×) and B (0.6×) are far below it, and molecules rich
in such bonds are the slowest connected class per √N (+0.44/+0.71 champion calls
per heavy bond, train/valid). Scaling the prefactor per atom (row 3: 0.65, row 4:
0.55, row ≥ 5: 0.45, B: 0.65; product for two heavy atoms) should shorten the
bond-relaxation phase by 1–3 calls on heavy-bond-rich molecules.
**Outcome and uncertainty:** train 0.8277 (Δ −0.0048, passed; ≈ 0.8226 without the
104079126 basin flip) but valid 0.8416 (Δ +0.0018). Molecules without heavy bonds
bit-identical (255/267). The small paired differences (|Δ| ≤ 3 calls) sum to −34
on train and +3 on valid; per-class signs flip between splits (Br −16 vs +12, P(V)
−17 vs +7, thiols −6 vs +16, Cl −1 vs −27); valid lost 54 calls in four S(VI)/thiol
detours. Reading: the stiff bond diagonal is learned by the multi-secant TS-BFGS
update within the first steps, so the guess diagonal only changes the first step
and re-rolls the path; the noise floor of such changes is ≈ ±0.005 C_S per split.
Not excluded: hypervalent S(VI) centres (valid +18 over 8 molecules) may need a
different factor than 0.65; xTB heavy-atom bonds may be stiffer than the
force-field constants.
**Reason to revisit:** only if a mechanism is found that makes the first-step
accuracy matter (e.g. a first-step line search) or with a per-class split of the
factor supported by more than 8 molecules.
**Next experiment:** none planned; if revisited, test the scaling restricted to
S(II)/P(III)/halogen single bonds (leave hypervalent centres unscaled) and judge by
the |Δ| ≤ 3 paired sum on both splits, not by C_S.

**Attempts:**
- Cycle 36; candidate `2326fbaf173207b1b6755f6bc6dc3f92d88271a6`; champion `fc3407d8ecc67de5fe3a9a7ad64eaf6d6f3bad56`; decision non_generalizable (train 0.8277395 / valid 0.8415559; energy 1.001023 / 1.009468).
  [Training evidence](evaluation_results/cycle-36-train.json) (c307c0f1aa804da59ab9f9664823d739).
  [Validation evidence](evaluation_results/cycle-36-valid.json) (7bd02ef01d9d4f07b8039b864c3a4cde).
  [Narrative](full_log.md#cycle-36--row-dependent-prefactor-of-the-fischeralmlöf-stretch-guess-bonds-to-third-row-and-heavier-atoms-and-to-boron-045065-softer).

## torsion-class-scale: class-resolved torsional guess for rotatable bonds (acyclic single bonds to a planar sp² centre)
Status: incorporated (cycle 37); refinement open

**Hypothesis:** The 1/√n sharing of the Fischer–Almlöf torsional constant (cycle 9)
was calibrated by the n = 9 sp³–sp³ case; single bonds to a planar centre have
n = 2–6 dihedrals and are left 2–3.5× above their rotational-barrier curvature
(sp²–sp³ 0.025 vs 0.004–0.014 Ha/rad², aryl–O 0.034 vs 0.010, aryl–N 0.049 vs
0.017, biphenyl/styrene/butadiene 0.023–0.033 vs 0.008–0.019, amide 0.095 vs
0.057). Scaling each class to the stiff end of its physical range (too soft by 2×
means no progress along the mode; too stiff by 2× still halves the error per step)
should shorten the torsion-dominated endgames of drug-like molecules, where the
multi-secant update cannot cover 10–20 torsions with a few secant pairs.
**Outcome and uncertainty:** Cycle 37 (sp²–sp³ 0.6, sp²–lone pair 0.45, carbonyl–lone
pair 0.7, sp²–sp² 0.65, fading to 1 with the bond-order factor between 1.5 and 2.2;
ring bonds ≤ 8 atoms, impropers and multi-fragment systems unchanged): **keep**,
train 0.8274668 (−0.0050), valid 0.8379440 (−0.0018); all controls bit-identical.
Gain grows with the number of scaled bonds (≥ 7: −1.1 / −2.1 calls, 7:1 and 11:2
better:worse) and with size (40–60 atoms −0.9 / −0.5 calls; < 20 atoms +0.1 / +0.5).
Class regression replicated in sign on both splits: pi–pi −0.35 / −0.26 calls per
bond, pi–lp −0.03 / −0.14 (presence −0.5 vs 0), pi–sigma neutral, carbonyl–lp
**+0.33 / +0.30** (adverse; n = 40 / 29 molecules). The valid aggregate is below the
±0.005 per-split noise floor; credibility rests on the replicated class/size structure.
**Reason to revisit:** the paying classes (sp²–sp², sp²–lone pair) are still 1.3–1.9×
above their physical values and the amide/ester factor went the wrong way; the
mechanism is scoped to connected systems (intramolecular torsions of dimer monomers
still use the unscaled model).
**Next experiment:** one refinement (carbonyl–lp → 1.0 or 0.85, pi–pi 0.65 → 0.5,
pi–lp 0.45 → 0.35), expected ≈ −0.003 per split — at the noise floor, so bundle
with an independent change (e.g. extension to multi-fragment systems, accepting the
basin lottery, or 1/n sharing for the paying classes) and keep the controls
(molecules without scaled bonds bit-identical).

**Attempts:**
- Cycle 37; candidate `3ca2b9dd7c07605f17b891694e506742d101e5a6`; champion `fc3407d8ecc67de5fe3a9a7ad64eaf6d6f3bad56`; decision keep.
  [Training evidence](evaluation_results/cycle-37-train.json) (evaluation `a9c4f19458a640088df3d2ba085390dc`).
  [Validation evidence](evaluation_results/cycle-37-valid.json) (evaluation `0d288340dd2c4df68d6b4a3a10ff75af`).
  [Narrative](full_log.md#cycle-37--class-resolved-torsional-guess-for-rotatable-bonds-acyclic-single-bonds-to-a-planar-sp-centre-scaled-to-their-rotational-barrier-curvature-connected-systems-only).
- Cycle 91 (extension to the fragments of multi-fragment systems, same constants): discard, train +0.0004 —
  see `fragment-torsion-classes` (turning dimers pay, non-turning fragments drift, charged walker re-rolls decide).

## linear-bend-guess: physical linear-bend constant for the dummy-atom coordinates of near-linear centres
Status: incorporated (cycle 39, champion `94ea9fe5a46f07ba802f25ef882e33de49f54fb1`); class refinement tested in
cycle 67 — non_generalizable (train −0.0004, valid +0.0009, no systematic class effect; preserved in
`ideas/linear-bend-guess/3420ebd99102f8e217d263ab14066fe81c39c230/algo.py`); the softening line is closed; the
heterocumulene stiffening (4 + 4 molecules) was re-tested as a bundle component in cycle 89
(`textbook-small-class-bundle`, non_generalizable: 2:2:0 train / 0:0:4 valid, signs flipping per molecule
against cycle 67) — closed, the constant stays 0.10 / 0.05

**Hypothesis:** Sella describes a two-bonded centre within 15° of linear (alkynes,
nitriles, azides, cumulenes, proton-shared D–H···A bridges) with a dummy atom X: the
dummy angles A–j–X / C–j–X (bend towards X) and the dummy dihedral A–j–X–C (bend in
the plane of the initial A–j–C bend, unit Jacobian) are the free bending
coordinates. Their inherited guesses — Fischer–Almlöf bend value (0.16–0.18 Ha/rad²
after the cycle-10 halving) and a flat 0.5 Ha/rad² for the dihedral — are 2–3× and
≈ 7× above the physical linear-bend constants (C–C≡C / C–C≡N 0.06–0.08 Ha/rad²,
cumulene/azide 0.11–0.13, hydrogen-bond bridges 0.02–0.05), so the bend carrying the
initial gradient converges at 0.86 per step. Setting both to 0.10 Ha/rad² (0.05 at
hydrogen centres) in connected systems should shorten the endgame of every molecule
where that bend is the slowest mode; the reference shares the defect.
**Outcome and uncertainty:** Cycle 39: **keep**, train 0.8196408 (−0.0078), valid
0.8326029 (−0.0053). Multi-fragment systems bit-identical; the 37 / 42 linear-centre
molecules went 22:3 / 23:8 better:worse (Σ −65 / −46 calls, rel 0.83 → 0.75 / 0.88
→ 0.82), small molecules with one to three sp centres by −2 to −24 calls (12:1 on
both splits); six further molecules changed (−1 to −3 calls) because their angle
enters the linear window mid-run. One basin change (135255884, −3 kJ/mol);
acalabrutinib 16 → 25 same basin. Cycle 67 (class-resolved constants from a unit-corrected static Wilson–GF fit of
textbook fundamentals, `/tmp/gflin67.py`: X–C≡Y 0.07 Ha/rad², H–C≡ 0.05,
heterocumulene 0.14, allene 0.07, two single bonds 0.10, multiplicity from a length
ramp 0.92–0.87 r_ref): **non_generalizable**, train 0.7076505 (−0.00043), valid
0.7219545 (+0.00092). Controls bit-identical; affected 33 / 39 molecules all same
basin, 11:7 Σ 0 / 5:10 Σ +1 calls, plus one mid-run dummy rebuild (a P–O–P bridge
crossing 165°) 32 → 40 on valid and acalabrutinib 15 → 25 on train — path noise, no
class signal. Sub-classes over both splits: X–C≡Y softening 9:10, H–C≡ 3:3,
heterocumulene stiffening 3:2 Σ −7 (the only consistent sign: a too-soft guess
costs steps, an over-stiff soft mode is free). The dummy linear bends are soft
modes that the update absorbs; the one-call-per-mis-modelled-mode premise holds for
stiff modes only.
**Reason to revisit:** the physical constants split by centre class: triple-bond
carbon (nitrile, alkyne) 0.06–0.08, cumulene/azide/heterocumulene centre 0.10–0.13,
hydrogen bridge 0.02–0.05; the terminal-dummy dihedrals (substituent azimuths about
the axis) still carry 0.5 Ha/rad², immaterial for Newton steps (no gradient) but
untested; improper dihedrals at 3+-bonded linear centres are untouched.
**Next experiment:** none alone. Keep 0.10 / 0.05. If a bundle of textbook
calibrations is ever assembled (cycle-66 B/Si factors, dative contacts), include only
the heterocumulene stiffening (two multiple bonds with a heteroatom at or next to the
centre → 0.14 Ha/rad², 4 train / 4 valid molecules, −7 calls over both splits) —
the evaluator is deterministic, so the per-molecule changes of cycles 66 / 67
re-contribute exactly in a bundle; do not re-run the softening parts.

**Attempts:**
- Cycle 39; candidate `94ea9fe5a46f07ba802f25ef882e33de49f54fb1`; champion `3ca2b9dd7c07605f17b891694e506742d101e5a6`; decision keep.
  [Training evidence](evaluation_results/cycle-39-train.json) (evaluation `84b95d99cd5947058220b2a511e83bab`).
  [Validation evidence](evaluation_results/cycle-39-valid.json) (evaluation `03792890f6564afda687b8f0c1859c7d`).
  [Narrative](full_log.md#cycle-39--physical-linear-bend-guess-for-the-dummy-atom-coordinates-of-near-linear-centres-dummy-angles-and-the-ajxc-dummy-dihedral-010-harad-005-at-hydrogen-centres-connected-systems-only).
- Cycle 67; candidate `3420ebd99102f8e217d263ab14066fe81c39c230`; champion `2a8fb394fa396f8111c886bffafdaa97b190f84f`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-67-train.json) (evaluation `17744faa48f24457a0a5ba4237c792ef`).
  [Validation evidence](evaluation_results/cycle-67-valid.json) (evaluation `4c18af3902f746bda7ae00e1cd72ea92`).
  [Narrative](full_log.md#cycle-67--class-resolved-linear-bend-constants-for-the-dummy-coordinates-of-near-linear-centres-non_generalizable).
  [Implementation](ideas/linear-bend-guess/3420ebd99102f8e217d263ab14066fe81c39c230/algo.py).
- Cycle 89 (bundle with the P-only torsion floor, see `textbook-small-class-bundle`); candidate `ee5c5798ee8fabf56a76dcfc31e369775c9d7969`; champion `8a30dec9b7cf843c34cc0feefaa961e8a84b33a8`; decision non_generalizable (train −0.0006380 / valid +0.0006278; heterocumulene class Σ −4 calls train, +8 valid).
  [Training evidence](evaluation_results/cycle-89-train.json) (evaluation `e3dd8e208f764673b9dd0336fd6643eb`).
  [Validation evidence](evaluation_results/cycle-89-valid.json) (evaluation `fbbc4fd7a7464a70b6c8cbeaa4105aa0`).
  [Implementation](ideas/textbook-small-class-bundle/ee5c5798ee8fabf56a76dcfc31e369775c9d7969/algo.py).

## hydrogen-rotor-torsions: class factors for the sigma/lone-pair rotor bonds (hydroxyl, primary amine, bonds to two-coordinate S/Se, S–S)
Status: rejected as a bundle (cycle 40, null); the S(II) component was re-tested as `third-row-torsion-floor` (cycle 41, discard — the 8:2 did not replicate, 5:5)

**Hypothesis:** the Fischer–Almlöf-vs-physical accounting of the rotational
stiffness Σ_d h_d that found the linear bends (cycle 39) also flags the hydrogen
rotors on sp³ centres (C–O–H 0.020 Ha/rad² vs 0.008, ρ 2.6; C–NH₂ 0.028 vs 0.014,
ρ 2) and, in the soft direction, the bonds to a two-coordinate sulfur (C–S–H / C–S–C
0.005 vs 0.009 / 0.015, ρ 0.36–0.55 — overshoot per step) and disulfides (0.003 vs
≈ 0.025, ρ 0.12). Class factors hydroxyl-on-sp³-C/Si ×0.45, primary amine ×0.6, any
acyclic bond with one two-coordinate S/Se end ×2.5, S–S ×8, connected systems only,
should shorten the endgame of the 42 / 51 molecules that carry such bonds.
**Outcome and uncertainty:** Cycle 40: **discard**, train 0.8198194 (+0.00018),
controls exact (0 of 212 multi-fragment and 0 of 215 other connected molecules
changed), no basin change. By single-class molecules: hydroxyl Σ −9 (14 molecules,
5:5), amine Σ +14 (4; all of it 315516336 7 → 22, a fragile cycle-39 path), S(II)
Σ −14 (14, 8:2, −1.0 per molecule), disulfide Σ +2 (4, 1:1); mixed-class molecules
336270104 +7 and 135181905 +7. A per-rotor H···acceptor census found no H-bond
stiffening pattern (H-bonded hydroxyls improved 6:4, free ones did not). The hydroxyl
and amine rotors are evidently covered by the multi-secant update early (large,
simple gradients), unlike the small-gradient linear bend; only the ρ < 0.5 class
shows a consistent signal, and at 14 molecules × ±1 call it may not replicate.
**Reason to revisit:** none for the hydrogen rotors unless a later change makes the
hydroxyl torsion the slowest mode (e.g. after the stretched-bond fix); the S(II) /
S–S component was closed by `third-row-torsion-floor` (cycle 41: 5:5 on the S(II)
subset). Cross-cycle lesson: the hydroxyl ≥ 60° subset preferred the *soft* guess
4:1 (Σ −15) while the < 15° subsets of cycle 41 preferred the physical value 8:1 —
the right model stiffness of a rotor depends on its relaxation amplitude, not on
its bond class alone.
**Next experiment:** none planned for the hydroxyl/amine rules. If revisited: only
together with a mechanism that removes the 315516336-type single-path lottery from
the read-out (e.g. paired evaluation on both splits before judging a class).

**Attempts:**
- Cycle 40; candidate `0b61f0d96336ba8d5754eedd4893aaff0614c8a1`; champion `94ea9fe5a46f07ba802f25ef882e33de49f54fb1`; decision discard.
  [Training evidence](evaluation_results/cycle-40-train.json) (evaluation `02f8b72c87df4878b2129b6b4a6eb24a`).
  Implementation preserved: `ideas/hydrogen-rotor-torsions/0b61f0d96336ba8d5754eedd4893aaff0614c8a1/algo.py`.
  [Narrative](full_log.md#cycle-40--class-factors-for-the-sigmalone-pair-rotor-bonds-whose-fischeralmlöf-torsional-guess-is-off-by--2-hydroxyl-on-sp-csi-045-primary-amine-on-sp-c-06-bonds-to-two-coordinate-sse-25-ss-8-acyclic-bonds-connected-systems-only).

## third-row-torsion-floor: floor the rotational stiffness of acyclic sigma/lone-pair single bonds to third-row and heavier non-metals at 0.014 Ha/rad²
Status: rejected (cycle 41, discard); rotor class-factor family closed (cycles 37/40/41); the P-only floor was
re-tested as a bundle component in cycle 89 (`textbook-small-class-bundle`, non_generalizable: P class 5:1:3
Σ Δrel +0.054 train / 3:2:2 +0.029 valid — a wash, the 6:0 did not replicate) — closed for every element class

**Hypothesis:** the Fischer–Almlöf torsional constant carries (r·r_cov)⁻⁴ and the
champion's second bond-order factor, so for the long equilibrium bonds of the third
row and below it collapses to a fraction of the rotational-barrier curvature (C–S
0.005 Ha/rad² vs 0.009–0.015 physical, S–S 0.003 vs ≈ 0.025, C–P 0.0065 vs 0.014,
Si–Si 0.0025 vs 0.009); a model 2–8× too soft makes every quasi-Newton step
overshoot along the mode (|1 − 1/ρ| > 1). One rule: acyclic bond between two
sigma/lone-pair centres, max(Z) > 10, both atoms main-group non-metals, ring ≤ 8
excluded, per-dihedral guess × max(1, 0.014/Σ_d h_d); connected systems only.
Cycle-40 S(II) subset (8:2, −1 call per molecule) was the replication target.
**Outcome and uncertainty:** Cycle 41: **discard**, train 0.8203317 (+0.00069),
controls exact (0 of 212 multi-fragment and 0 of 214 untouched connected molecules
changed; 41 of 43 census molecules re-rolled, 16:12:13, Σ +20). The S(II) subset
went 5:5 (Σ −3) — no replication; phosphorus subset 6:0:4 (Σ −12, post hoc). The
sum is five detours in molecules with large-amplitude rotor relaxations: polysilane
cages 135095297 17 → 29 and 135095518 28 → 32 (Si–Si ×7–10), disulfide 134997543
35 → 47 (S–S ×6.6), 135098427 +6, 135181905 +4. Binned by the largest torsional
relaxation about a floored bond: < 15° 8:1 Σ −10, 15–30° Σ +20, 30–60° Σ −8,
≥ 60° Σ +18. Together with cycle 40 (hydroxyl ≥ 60° soft guess 4:1) and cycle 37
(gains growing with amplitude when softening) this says the optimal model stiffness
of a rotor is amplitude-dependent — the secant stiffness of a cosine rotor far from
its minimum is k_min·sinc(θ) — and the phase-blind class factor cannot serve both
populations.
**Reason to revisit:** only as an amplitude-conditioned model (a rotor whose first
secant pair reveals a much smaller stiffness than the guess should keep the soft
value, one near its minimum the physical one) — which is what the multi-secant
update already tries to do; or a P-only floor (10 molecules, ≤ −0.002 per split,
below the noise floor). Not worth a cycle on its own.
**Next experiment:** none planned. If the P subset is ever re-tested, bundle it with
an independent change and judge it on both splits' paired tables.

**Attempts:**
- Cycle 41; candidate `b4f2383a1571c6bae28171197c58c3fe6b4d1100`; champion `94ea9fe5a46f07ba802f25ef882e33de49f54fb1`; decision discard.
  [Training evidence](evaluation_results/cycle-41-train.json) (evaluation `2dadfb58a0034f1dac6b3d05c8f7da7a`).
  Implementation preserved: `ideas/third-row-torsion-floor/b4f2383a1571c6bae28171197c58c3fe6b4d1100/algo.py`.
- Cycle 89 (P-only scope, bundled with the heterocumulene linear-bend stiffening, see `textbook-small-class-bundle`); candidate `ee5c5798ee8fabf56a76dcfc31e369775c9d7969`; champion `8a30dec9b7cf843c34cc0feefaa961e8a84b33a8`; decision non_generalizable (train −0.0006380 / valid +0.0006278; P class Σ +2 calls train, −1 valid).
  [Training evidence](evaluation_results/cycle-89-train.json) (evaluation `e3dd8e208f764673b9dd0336fd6643eb`).
  [Validation evidence](evaluation_results/cycle-89-valid.json) (evaluation `fbbc4fd7a7464a70b6c8cbeaa4105aa0`).
  [Implementation](ideas/textbook-small-class-bundle/ee5c5798ee8fabf56a76dcfc31e369775c9d7969/algo.py).
  [Narrative](full_log.md#cycle-41--third-row-torsional-floor-the-rotational-stiffness-the-fischeralmlöf-guess-assigns-to-an-acyclic-sigmalone-pair-single-bond-involving-a-third-row-or-heavier-non-metal-is-floored-at-0014-harad-connected-systems-first-row-bonds-ring-bonds-bonds-to-planar-centres-and-metal-contacts-untouched).

## torsion-guess-relaxed-length: evaluate the torsional guess at the relaxed bond length instead of the (possibly stretched) initial one
Status: closed (tested as the geometry-tracked torsional diagonal in cycle 46: a wash, discarded; the clip variant is not worth a cycle)

**Hypothesis:** the Fischer–Almlöf torsional constant, with the champion's second
bond-order factor, scales as bo² (r r_cov)⁻⁴ with bo = exp(−2.85 (r − r_cov)/Bohr),
so a bond that starts 0.05 / 0.10 / 0.16 Å beyond the covalent-radius sum gets a
torsional guess 0.5 / 0.35 / 0.19 of its relaxed value; the bond relaxes in the
first steps but the diagonal guess is never revisited, so the torsion about it is
2–5× too soft in the endgame (overshoot, error factor > 1 per step). Evaluating the
formula at min(r, r_cov + δ) (or tracking the diagonal with the geometry as cycle 35
does for the contact term) removes the artefact; the same clip also stiffens the
genuinely long single bonds whose barriers are physically higher than ethane's
(crowded C–C, O–O, N–O, Si–Si).
**Outcome and uncertainty:** cycle 46 implemented the tracked form (the torsional
entries of the diagonal guess — proper dihedrals with bond-order, class and 1/√n
factors, impropers plain — re-evaluated at the current central-bond lengths before
every secant update, H ← H + P diag(h_t(x) − h_t(x_prev)) P with P the range(B)
projector; connected systems only): train 0.76561 vs 0.76270 (+0.0029, decision
discard, dimers bit-identical). Connected molecules: 66 better / 75 worse / 116
same, Σ dcalls +28 of which +15 is one lottery (315516336, champion 7 calls at a
point 0.34 kJ/mol above the reference, candidate 22 calls at the reference minimum)
and +24 two flat-mode wanders (135132759 11→24, 363892164 21→32, endpoints 0.01–0.05
kJ/mol lower); the rest cancels. No dependence on the census metric: the ≥ 1.4×
stretched-bond group (152 molecules) is +36 calls (45 better / 52 worse), the
stiffening-mass ≥ 0.3 group −15 (31/26), the softening groups flat. So the
bond-length dependence of the diagonal torsional guess is not a limiting factor of
the model: torsions are the directions the optimizer moves along from step 2 on,
so the secant pairs supersede the guess along them within a few steps, and shifting
the guess afterwards merely perturbs the learned curvature (unlike the contact term
of cycle 35, which appears in directions the pairs have not sampled). Earlier census
(cycle 40, 1.25 r_cov graph, connected molecules, dihedral-carrying bonds): with δ = 0.03 Å 134 train / 182 valid molecules
carry 391 / 529 such bonds (median ratio 1.27, i.e. mostly mild), with δ = 0.05 Å
99 / 124 molecules (206 / 261 bonds); C–C acyclic and ring bonds dominate (46 + 41
train at δ = 0.05), then C–S, C–O, C–N, N–O; the tail (ratios 5–30) is
polysulfide / P₄ / I₂-type cages and proton-shared O–H···O "bonds" (r − r_cov
0.2–0.65 Å) where the long "bond" is a weak contact and a soft torsion may be right.
Risk: a mostly-mild correction over half of the connected set is a re-roll (cycle 38
lesson) with an exotic tail that could overshoot; expected value unclear.
**Reason to revisit:** only if a diagnostic ever shows torsional overshoot in the
endgame of stretched-bond starts; cycle 46 says the learned curvature takes over
before the artefact matters. The class calibrations (cycles 9, 37, 40/41) were made
at start geometries and are therefore self-consistent as they stand.
**Next experiment:** none planned. If revisited, restrict the shift to the subspace
orthogonal to the stored secant directions (so it never touches learned curvature)
and apply it only during the first few steps (while the bonds relax); the
implementation of cycle 46 (`_torsion_guess_values`, `_track_torsion_guess`) is the
starting point.

**Attempts:**
- static census in cycle 40 (see the cycle-40 narrative).
- cycle 46: candidate 9e3179d8506861a6054ec4870b982444f96c445e vs champion
  21a3273fee7548462d38cebd93dcfe77f0de31b4 — train 0.76561 vs 0.76270 (+0.0029),
  decision discard (valid not run); evidence `evaluation_results/cycle-46-train.json`
  (evaluation 5faf0a10423542b0b536165d689b1e7c); implementation preserved in
  `ideas/torsion-guess-relaxed-length/9e3179d8506861a6054ec4870b982444f96c445e/algo.py`.

## metal-ionic-radii: ionic reference radii for the group-1/2 metals in the model-Hessian stiffness formulas
Status: accepted (cycle 52: train 0.7515590, valid 0.7677808; 14/14 metal systems changed, 11 better / 3 worse, Σ −23 / −43 calls; everything else bit-identical)

**Hypothesis:** the Almlöf stretch curvature k = 0.3601 exp(−1.944 (r − r_ref)/Bohr)
Ha/Bohr² assigns a covalent single-bond stiffness (0.36 Ha/Bohr²) at r = r_ref.
For the s-block metals the Cordero "covalent" radius is calibrated on their ionic
contacts in the CSD (Na 1.66 + O 0.66 = 2.32 Å is the Na⁺···O distance; Li 1.28 + C
0.76 = 2.04 ≈ the Li⁺···C(π) distance 2.3 Å; Mg 1.41 + C 0.76 = 2.17 ≈ Mg···C 2.22 Å),
so every metal–ligand contact and metal–ligand "bond" of the model gets 0.1–0.4
Ha/Bohr² (K···O at 2.31 Å in the valid K–ether even 1.47), 5–15× the physical
curvature of an ion–dipole / cation–π contact (0.02–0.07 Ha/Bohr² from the 200–500
cm⁻¹ M⁺···OH₂ stretches; ion–dipole model (2n−4)|U|/r² with n = 9 gives 0.029 for
Na⁺···OH₂, 3(n−3)|U|/r² ≈ 0.05 for Na⁺–benzene). The tangential stiffness that the
cage of radial pair terms builds around a cation on a π face (6 × k sin²θ/2 ≈ 0.07
Ha/Bohr² with Cordero radii) is 10× the physical slip curvature, and the tracked pair
term re-imposes it at every new geometry, so the cation "walkers" (Na–pyrrole train 29
calls vs ref 16, Li–phenol valid 34 vs 22) advance 0.01–0.05 Å per step. With the
Shannon ionic radius (CN 6: Li 0.76, Na 1.02, K 1.38, Rb 1.52, Cs 1.67, Be 0.45, Mg
0.72, Ca 1.00, Sr 1.18, Ba 1.35 Å) as the metal's reference radius the same formula
gives 0.037 (Na⁺···O 2.3 Å), 0.062 (Li⁺···O 1.9), 0.032 (K⁺···O 2.7), 0.026 (Na···C(π)
2.5), 0.02 (Mg···O 2.16) Ha/Bohr² — the physical range without a new constant. Halide
anions are the opposite case (Cordero Cl 1.02 is the covalent C–Cl radius, Cl⁻···H at
2.1 Å is already 0.8 Å outside r_ref → 0.015 Ha/Bohr², right) and keep the covalent
radii. Change: a module table `_STIFFNESS_RADII` (covalent radii except the ten
group-1/2 metals) used by `_h0_bond`, `_h0_angle`, `_h0_dihedral`, the torsion-class
bond order, `_h0_nonlocal_contacts` (cutoff and k) and the bond-order factors of
`_h0_contact_bends`; the bond criterion of `find_all_bonds` (connectivity, fragments,
`_start_fragments`) keeps the covalent radii, so the coordinate systems are unchanged
and every metal-free molecule is bit-identical.
**Outcome and uncertainty:** census (cycle-49 champion, `/tmp/metal52.py`): 6 train /
8 valid molecules contain Li/Na/K/Mg/Ca (share 0.0135 / 0.0176, mean rel 1.056 /
1.025 vs 0.75 overall); dimers Na–pyrrole 1.81, K–alkenes 1.10, Na–amines 1.00 (train),
Li–phenol 1.55, K–ethers 1.33, Na–pyridine 0.86 (valid); connected (metal within
1.25 r_cov of ligand atoms, 6 Mg–C / 6 Li–C bonds etc.) 0.70–1.11. Upper bound of the
gain (all to rel 0.75) −0.004 / −0.005; realistic −0.002 / −0.003 per split, i.e. at
the lottery-noise level but one-sided (all other molecules bit-identical). Risks: the
connected metal systems lose 10× on their metal–ligand bond diagonal and ~2× on the
angles at the metal (learned within the first steps, cycle-36 lesson — small effect
either way); a softer model can send a walker to another basin (energy gate margin
train 0.82 rel-energy units).
**Reason to revisit:** the connected metal systems gained as much as the dimers
(Mg–phenol 43 → 34, Li–alkanes 32 → 17, K–sulfides 28 → 21, Ca–alkenes 10 → 6): a 10×
overestimate spread over ~20 redundant metal coordinates is not "learned in the first
steps" (the cycle-36 lesson is for 1.5× errors on one bond). Two-call losses on
Na···H–C / K···S weak contacts (model now 0.005 Ha/Bohr² per pair) are noise-sized; no
repair planned. The train energy margin is now 0.087 rel-energy units (the Na–pyrrole
lottery credit is gone), so a future train `energy_below_baseline` on a path-changing
candidate may be this margin rather than the idea.
**Next experiment:** none. Other elements whose Cordero radius is an ionic-contact
radius do not occur (halide anions are covalently calibrated and already soft).

**Attempts:**
- Cycle 52; candidate `06e433a3b54220d8920e477ca43ed54ba1fef406`; champion
  `d148070066a818071b41626af514c0627710334b`; decision keep (train 0.7515590 vs
  0.7532674, valid 0.7677808 vs 0.7715672; train mean rel_energy 1.001739 → 1.000186,
  valid unchanged).
  [Training evidence](evaluation_results/cycle-52-train.json) (4fdaec24fcef4aa1bf69718a87aae4d5);
  [validation evidence](evaluation_results/cycle-52-valid.json) (59ddfb5869d0416d81a43e82c72cfaed).
  [Narrative](full_log.md#cycle-52--ionic-reference-radii-for-the-group-12-metals-in-the-model-hessian-stiffness-formulas-keep).

## analytic-secant-transport: geometry-tracked stretch diagonal and transport of the secant pairs with the analytic part of the model (connected systems)
Status: accepted (cycle 53: train 0.7446790, valid 0.7482450; multi-fragment bit-identical; connected 92:44 / 113:48 better:worse)

**Hypothesis:** the multi-secant TS-BFGS update imposes B⁺S = Ỹ exactly (UᵀS = I in
`_MS_TS_BFGS`), so any shift of the model along a sampled direction is overridden by
the measured pair, which is a path average over its segment. After a 0.1 Å bond step
the path-average curvature of a Morse-like bond lags the local curvature by 25–35 %
(Morse: 0.52 k_e at +0.1 Å, 1.76 k_e at −0.1 Å), which costs roughly one extra step per
anharmonic chain in the endgame of stiff molecules (small inorganic molecules take
7–10 calls at maxdisp 0.1–0.3 Å; connected same-endpoint regression on the cycle-52
champion: +7.6 (se 1.8) train / +14.0 (se 2.2) valid calls per Å of the largest bond
change, +1.0 / +0.8 calls per bond changing by > 0.1 Å, binned residual +2.9 / +6.1 for
maxdb ≥ 0.2 Å). The Fischer–Almlöf stretch exponential k(r) = 0.3601 exp(−1.944 (r −
r_ref)/Bohr) is a local curvature–length relation that matches Badger's rule (d ln k/dr
= −3/(r − d_ij) ≈ −3.3…−4.3/Å against Bb = 3.7/Å), so it is a physically justified,
conservative (Morse would give 3a ≈ 5–6/Å) transport of the bond curvature with the
geometry. Change (connected systems only, `InternalPES._transport`): the analytic part
of the model T(x) = P diag(d(x)) P + A(x) (projected Almlöf stretch diagonal from the new
`Internals._h0_stretch_diagonal`, plus the existing non-local contact block) is moved
with the geometry before every update, H ← H + T(x) − T(x_prev)
(`_track_analytic_model`, replacing `_track_nonlocal_contacts` on this class), and every
stored secant pair is transported with it before the update, y_j ← y_j + (T(x) − ½(T(x_j)
+ T(x_{j+1}))) s_j (`PES._collect_secant_pairs` stores the per-pair analytic average
tbar), so the secant conditions describe the curvature at the current point; the
dependence and consistency tests act on the transported pairs. Multi-fragment systems
keep the old path exactly (thin train energy margin +0.087).
**Outcome and uncertainty:** −0.0069 train / −0.0195 valid (the largest valid gain since
cycle 45). Multi-fragment systems bit-identical on both splits. Connected same-endpoint
gains by class (`/tmp/pairbond53.py`): small stiff molecules (< 6 heavy atoms, no
non-local contacts → stretch transport alone) −0.50 (18:5) / −0.62 (9:3) calls; ref ≥ 30
runs −1.28 (se 0.69) / −2.65 (se 0.65); ≥ 20 heavy atoms −0.71 (32:7) / −1.43 (57:20);
maxdb ≥ 0.2 Å −1.71 (se 0.93) / −1.05 (se 1.16); n10 ≥ 4 −2.33 / −2.83. So the predicted
anharmonic signature is present but the bulk of the gain comes from long runs of large
molecules, i.e. from transporting the pairs with the contact block A(x) (folding chains
change their contact curvature 2–6× along the path; the champion re-imposed the stale
segment averages along every sampled direction). The stretch/contact split is inferred
from the class pattern, not measured. Ten basin changes (+0.0012 / −0.0020) are lottery
noise; train energy margin now +0.159 (by lottery), valid +4.07.
**Reason to revisit:** if accepted, the same transport for multi-fragment systems (the
contact block changes 2–6× during the approach phase of a dimer) is the natural
follow-up, mindful of the train energy margin.
**Next experiment:** (a) the same transport for multi-fragment systems (contact block
during the approach; walkers creep along contact modes whose curvature grew after the
pairs were measured) — a dimer path change, so the train energy lottery applies; (b)
re-test the connected secant memory (5–6) with transported pairs (cycle 47's staleness
was measured without transport); (c) an A(x)-only control if the split matters later.

**Attempts:**
- Cycle 53; candidate `f38a8b6e79bc9856c9a87272d4b4e903ddfe2d7d`; champion
  `06e433a3b54220d8920e477ca43ed54ba1fef406`; decision keep (train 0.7446790 vs
  0.7515590, valid 0.7482450 vs 0.7677808; train mean rel_energy 1.000186 → 1.000339,
  valid 1.008717 → 1.008749).
  [Training evidence](evaluation_results/cycle-53-train.json) (170657c16c6c4979a5d77261a6bdcaab);
  [validation evidence](evaluation_results/cycle-53-valid.json) (26dda62539be4686ac3dbeb2896dc83a).
  [Narrative](full_log.md#cycle-53--geometry-tracked-stretch-diagonal-and-transport-of-the-secant-pairs-with-the-analytic-part-of-the-model-connected-systems-keep).

## simpson-transport-average: composite Simpson path average of the analytic model for the secant transport (shadow internals at interior points of the step)
Status: rejected (non_generalizable, cycle 54: train −0.0026, valid −0.0001); preserved at
`ideas/simpson-transport-average/394e459e6ab8eadc8bd60afce6f8a6d861796560/algo.py`.
**Hypothesis:** the cycle-53 transport moves every stored pair by (T(x) − T_avg,j) s_j with
T_avg,j the *trapezoid* average ½(T(x_prev) + T(x)) of the analytic model over the step.
The terms of T are exponentials of distances (k ∝ exp(−3.7/Å · r)), so for a contact or a
bond that opens by 0.5–1 Å in a single step (early torsional steps of a folding chain with
lever arms up to ~1 Å, a dissociating bond such as des370k_acids__alcohols with maxdb
0.84 Å that got worse 24→29 in cycle 53) the trapezoid overstates the true path average by
0.1–0.25 k_prev — more than the local value k(x) itself (b·Δr = 3/4/5 → +4/+14/+45 × k(x))
— leaving the transported pair too soft or of the wrong sign along the step. Simpson's rule
with the analytic model evaluated at the Cartesian midpoint (composite, panels ≤ 0.4 Å of
atomic displacement, at most four) is within 0.1 % of k_prev of the exact average of an
exponential for any step size (`/tmp/stub54.py`, real source extracted from algo.py).
Interior points are evaluated with a shadow `Internals` copy (own Atoms without a
calculator, shared coordinate lists and covalent graph → same contact pair set), so no
force call can be triggered; bond/contact steps with small b·Δr are unchanged to first
order, and multi-fragment systems (no transport) are bit-identical.
**Outcome and uncertainty:** train 0.7420590 vs 0.7446790 (−0.0026), valid 0.7481526 vs
0.7482450 (−0.0001, below the 1e-4 gate). Multi-fragment systems bit-identical on both
splits. Connected same-endpoint effect (`/tmp/pairbond53.py cycle-54 cycle-53`): dC_S
−0.0016 train (255 molecules, −0.075 calls, 30 better : 31 worse) / −0.0019 valid (259,
−0.035 calls, 40 : 43) — a small, consistent-in-sign but sign-count-neutral same-basin gain,
concentrated where predicted only weakly (train n10 [2,4) −0.47 se 0.30, maxdb ≥ 0.2 Å
−0.41 se 0.50; valid n10 [1,2) −0.51 se 0.24 (22 : 9), maxdb [0.12,0.2) −0.54 se 0.57;
des370k_acids__alcohols 29 → 25 as predicted, 135065494 (maxdb 1.53 Å) 55 → 63). The
rest is basin lottery: train 104079126 (bistable anion) 44 → 25 landing in the reference
basin (−0.0010 of the train gain), valid three deeper-basin runs lost (26 → 37, 29 → 40,
21 → 36; +0.0018). So the quadrature error of the trapezoid transport is real but small
in connected molecules: large steps are rare after the first few, and pairs older than
the memory (4) are gone before the accumulated average error matters. The remaining
approximation (Cartesian straight-line path compresses rotating bonds at the chord
midpoint) is harmless while the bond components of s vanish for torsional steps.
**Reason to revisit:** the multi-fragment approach phase takes 0.5–1 Å steps routinely,
so a dimer transport (which the champion does not have) needs this quadrature more than
the connected one did; the implementation is the natural base for it.
**Next experiment:** none alone; bundle with the dimer transport as one coherent change
(the transport with an accurate path average for the fragment systems) if that line is
opened, and attribute the connected part to this measurement (−0.002 same-basin).

**Attempts:**
- Cycle 54; candidate `394e459e6ab8eadc8bd60afce6f8a6d861796560`; champion
  `f38a8b6e79bc9856c9a87272d4b4e903ddfe2d7d`; decision non_generalizable (train
  0.7420590 vs 0.7446790, valid 0.7481526 vs 0.7482450; train mean rel_energy 1.000339 →
  1.000292, valid 1.008749 → 1.008769).
  [Implementation](ideas/simpson-transport-average/394e459e6ab8eadc8bd60afce6f8a6d861796560/algo.py).
  [Training evidence](evaluation_results/cycle-54-train.json) (39b185855ca644f8a472cda437aa1a53);
  [validation evidence](evaluation_results/cycle-54-valid.json) (154f265cd1fb42b28f2385e2e07c5810).
  [Narrative](full_log.md#cycle-54--composite-simpson-path-average-of-the-analytic-model-for-the-secant-transport-non_generalizable).

## dimer-secant-transport: secant-pair transport with the analytic model for multi-fragment systems (approach phase), with the composite Simpson path average
Status: accepted (cycle 55, champion `5d0a542040b30ac0f066135252c6bc80f13d072f`; train 0.7447 → 0.7194,
valid 0.7482 → 0.7271).
**Hypothesis:** the champion tracks the inter-fragment contact block A(x) for multi-fragment
systems by shifting the model (H ← H + A(x) − A(x_prev), cycles 42/43/45/49) but does not
transport the stored secant pairs, and the multi-secant TS-BFGS re-imposes B s_j = y_j
exactly (`/tmp/stub53.py`), so along every sampled direction — and the rigid-body
coordinates are sampled by every step of an approach — the model keeps the approach-phase
averages (contact curvature at 3–4 Å, 2–6× softer than at the formed hydrogen bond or ion
pair). That is the cycle-53 mechanism (connected: −0.0069 / −0.0195, gains concentrated in
folding chains) applied to the class where the contact curvature changes most: the dimer
endgame should bounce less along the approach directions and the model shift should no
longer be undone there. Because approach steps are 0.5–1 Å, the transport needs the
composite Simpson path average of cycle 54 (trapezoid error 0.1–0.25 of the larger end
value would leave approach pairs too soft or of the wrong sign); the same average applies
to connected systems, whose part of the result is therefore bit-identical to cycle 54's
connected results (train −0.0026 incl. one lucky basin, valid −0.0001; same-basin −0.002).
The shadow `Internals` gets independent `Rotation` objects (their quaternion-branch state
is mutated on evaluation) synced from the real ones before each interior evaluation, so
the real object's evaluation sequence is unchanged.
**Outcome and uncertainty:** confirmed, and the largest dimer gain of the whole line: train
−0.0253 (dimer part −0.0223, connected −0.0030), valid −0.0212 (dimer −0.0205, connected
−0.0007). Same-basin dimer effect −0.0222 / −0.0159 (134:44 and 127:45 better:worse, −1.6 /
−1.8 calls per dimer), monotone in the approach distance (rmsd start → end < 0.4 Å: −0.2 to
−0.9 calls; 0.4–0.8 Å: −1.3 / −1.5; 0.8–1.5 Å: −2.7 / −2.3; > 1.5 Å: −2.7 / −4.9), present for
charged (−1.0 / −1.3) and neutral (−1.9 / −2.1) pairs and every reference-length bin except
the valid short-reference bin (ref < 16: +0.3 ± 0.7, 8:9). Basin changes net −0.0001 / −0.0046
(train: carboxylates__water 50 → 13 into a shallower basin −0.206 credit, monoatomics__pyrrole
23 → 33 into a basin 11 kJ/mol deeper +0.788 credit; valid: alkanes__guanidiniums 37 → 14
shallower −0.155, guanidiniums__ketones 32 → 14 lost a 5 kJ/mol deeper basin −0.129). Train
energy margin +0.735 = +2.19 − 1.46, of which +0.788 is the single monoatomics__pyrrole
credit — without it the train split would have been invalid at −0.053; valid +3.836 (still
carried by carboxylates__ketones +3.216). Uncertainty: the cost of the interior model
evaluations is wall-clock only (no force calls); the transport is not applied across a
rebuild (the new InternalPES starts with fresh pairs).
**Reason to revisit:** the approach phase is now transported, but the memory is still 4 pairs
(cycles 5–6 showed 8 pairs hurt *without* transport, cycle 53's connected re-test was not done);
the valid short-reference dimers (ref < 16, rel 1.45, share 0.084 valid) remain the worst class
and did not move — they are same-basin slow charged dimers and slide-past runs whose cost is
decided in the first 3–5 steps, not in the endgame.
**Next experiment:** transported-memory re-test (5 or 6 pairs) for all systems; then a census of
the short-reference dimer runs (what the first five steps do) with the new champion's endpoints.

**Attempts:**
- cycle 55 — candidate `5d0a542040b30ac0f066135252c6bc80f13d072f`, champion
  `f38a8b6e79bc9856c9a87272d4b4e903ddfe2d7d`: train 0.7194 (Δ −0.0253, mean rel energy
  1.00157), valid 0.7271 (Δ −0.0212, 1.00825); decision **keep**. Evidence:
  `evaluation_results/cycle-55-train.json` (evaluation b6fb8eb0f1594e8d82e2bc85b5479c30),
  `evaluation_results/cycle-55-valid.json` (5de5a7b57d544968941d670ccc15ebc9);
  `full_log.md#cycle-55--secant-pair-transport-with-the-analytic-model-for-multi-fragment-systems-keep`.

## curved-step-analytic-model: predictor-corrector quasi-Newton step under the geometry-following analytic model (connected systems)
Status: **accepted at cycle 79** (contact block only, connected systems: train 0.6846728 / valid 0.6898550,
−0.00214 / −0.00260, champion `336c47f6d09df544e3b314691af989f21ed6174e`) after two rejections (cycle 57, the
whole analytic model T incl. the stretch diagonal, connected: train −0.0001; cycle 78, contact block only, all
systems: train +0.0009; cycle 105, contact block on the inter-fragment block of neutral closed-shell complexes
only: train +0.0006). Closed at the other scopes: the stretch part loses in the short runs, the multi-fragment
part is an approach-phase lottery of the ionic complexes and, on the neutral complexes, a small net call saving
(−9 over 127 same-basin runs, 43 : 34) that the metric turns into a loss because the short runs with small
references lose it (ref < 30 +14 calls) while the long walks with large references gain (ref ≥ 30 −23);
constants (Simpson panel 0.4 Å, ≤ 4 panels, ≤ 2 corrections, 1e-4) not to be scanned, and a step-length
threshold for the fragment scope would be a scan on a ±0.001 signal. **Multi-fragment scope closed in both
variants (78 all complexes, 105 neutral only).** Implementations preserved in `ideas/curved-step-analytic-model/ed2664b53496ba09f754803e49ef5f796d06d9b8/algo.py`
(cycle 57) and `ideas/curved-step-analytic-model/cd9cffd2a038c0d42b7a3ae0df622df2de256fb5/algo.py` (cycle 78:
`InternalPES._contact_model_from` / `_contact_model_at_positions` / `_contact_model_ahead` — the contact block A on
the shadow internals at the Simpson nodes of the step ahead and its path averages dA_g / dA_e, fragment
translation/rotation rows included; `Sella._curved_step` — the fixed-point iteration)
(reusable: `InternalPES._analytic_model_ahead` — Simpson path averages dT_g = ∫₀¹[T(x(t)) − T(x)]dt and
dT_e = 2∫₀¹(1 − t)[T(x(t)) − T(x)]dt along the linearised Cartesian path x + t·Binv·s, on the shadow
internals; `Sella._restricted_step_with(B)` — the restricted step under a substitute model matrix with the
Hessian object restored; `PES._B_pred` — model matrix for the trust-ratio prediction of the next kick).
**Hypothesis:** the model Hessian of connected systems moves with the geometry (H + T(x) − T(x_k), cycle 53),
but the step −H⁻¹g assumes the curvature at x_k over the whole step although the model itself says how it
changes: a bond that starts 0.06 Å too long (the typical start; C–C shorten:lengthen 101:26 at |dr| > 0.06 Å,
C–N 62:10, C–O 24:2) stiffens by 25 % while it shortens, so the local step overshoots the model's own
minimum by 12 % of the step (0.007 Å, 4× the displacement gate); compressed bonds are undershot; a closing
contact is met with the curvature it had before the step. The step of the path-dependent model,
s = −(H + dT_g(s))⁻¹g under the unchanged trust region, found by a fixed-point iteration from the
quadratic step (second correction only when contracting), removes the first-order error; the trust ratio
is judged with the same model's energy prediction (dT_e). Expected −0.2 to −0.5 calls per connected
molecule (C_S −0.003 to −0.008); multi-fragment systems bit-identical.
**Outcome and uncertainty:** neutral. Train 0.71929 (Δ −0.0001; connected −0.0001, same-basin −0.0012;
multi-fragment 212/212 bit-identical). Connected same-basin −0.12 ± 0.12 calls per molecule (58 better / 51
worse / 148 identical). The effect is not where the hypothesis put it: molecules whose largest bond change is
0.05–0.12 Å (the organic shorten-by-0.06 Å class, n 148) −0.09 to −0.13 each (29:28) — the bond
overshoot is not a binding constraint, because the 12 % residual (0.007 Å) is removed by the next step
while the soft modes still need 5–15 more; the gains sit in the long runs (champion ≥ 16 calls: −0.22 and
−0.88 each, 36:22; rmsd0 ≥ 0.7 Å −0.45, 23:14; n05 ≥ 6 bonds −0.36/−1.30), i.e. the folding chains where the
contact block changes along the step, and the losses in the short runs (6–12 calls: +0.16 to +0.21 each,
9:18), in bonds that lengthen ≥ 0.1 Å (+0.71, 7:8), maxdb ≥ 0.2 Å (+0.90, 5:8) and the small hydrogen-free
molecules (N < 8: +0.20, 5:11 — B–H, N–S, O–P, C–Cl, C–I, Br–S, B–Br, S₄, B–F bonds), where the Almlöf
exponential is not the anharmonicity of the xTB bond and its error is now applied twice (transport and
step). One basin change against the candidate (104079126 25 → 47 into a basin 22.7 kJ/mol deeper, +0.022
credit) cancels the same-basin gain; two small ones for it. Energy margin +0.69 (was +0.74). Wall time
56 s vs 35 s per split (2–8 shadow model evaluations per iteration, ≤ 2 iterations).
**Reason to revisit:** the contact-block part of the correction is the part with a signal (folding chains,
long runs); the stretch part is where the losses are. A contact-only curved step (dT from the non-local
block A(x) alone) would be the clean test, but its upper bound from this run is ≈ −0.002 per split — below
the gate on its own; worth bundling with an independent change to the same runs, or with a
multi-fragment extension (approach steps of 0.5–1 Å into a contact wall whose curvature changes 6×
along the step — the largest curvature change of any step in the sets; energy-lottery risk on train).
**Next experiment:** multi-fragment curved step (TR internals allowed, the fragment-coordinate model
included in T), judged first on the same-basin approach-distance bins of `/tmp/pairdimer55.py`; or the
contact-only variant bundled with the next connected-system change.

**Attempts:**
- cycle 57 — candidate `ed2664b53496ba09f754803e49ef5f796d06d9b8`, champion
  `5d0a542040b30ac0f066135252c6bc80f13d072f`: train 0.71929 (Δ −0.0001, mean rel energy 1.00148),
  469/469 converged; decision **discard** (train improvement below 1e-4; valid not run). Evidence:
  `evaluation_results/cycle-57-train.json` (evaluation e444d166623d4640be4e1c96a4e84cd2);
  `full_log.md#cycle-57--predictor-corrector-step-under-the-geometry-following-analytic-model-connected-systems-discard`.
- cycle 78 — candidate `cd9cffd2a038c0d42b7a3ae0df622df2de256fb5`, champion
  `1ba6177625d7e7ff315c5995188815383eda7a4b`: the contact-only variant for all systems (dA from the non-local /
  inter-fragment block A(x) alone, stretch diagonal excluded, fragment translation/rotation coordinates included,
  fixed point ≤ 2 corrections, Simpson ≤ 4 panels of 0.4 Å, trust ratio judged with H + dA_e). Train 0.6877086
  (Δ +0.00089, mean rel energy 1.00251, credit +1.331 → +1.176), 469/469 converged, 364 s; decision **discard**
  (train gate; valid not run). Read-out: total calls −40 but the score rises on small-reference runs. Connected
  −49 calls (−0.0021; 41 : 176 : 40): far walks rmsd₀ ≥ 0.7 Å −64 (17 : 10, 384221749 59 → 42, 134997543
  46 → 34, 135140595 38 → 30), mid +8 (acalabrutinib basin change +10), near-equilibrium +7 (7 : 13, ±1–2);
  multi-fragment +9 (+0.0030; 74 : 69 : 69) = same basin −49 (69 : 61; neutral −28, ionic −21) + 13 basin changes
  +58 (11 ionic, 5 : 6: carboxylates__water 13 → 43, ammoniums__benzene 17 → 41, carboxylates__ketones 21 → 41
  on references of 49 / 50 / 9; benzene__guanidiniums 26 → 11). Offline (`/tmp/curv78_arc.py`): the linearised
  path is 0.05–0.09 Å off the exact arc for 0.25-rad fragment rotations, the along-step contact averages differ by
  2–8 %, the corrections themselves ×0.7–1.9 — the rigid-body steps are changed by the mechanism, not by the
  linearisation. Conclusion: the path dependence of the contact block is not a binding cost; its systematic part
  (long walks −4.5 %, dimer settlings −1.4 %) is below the approach-phase lottery of the ionic complexes. Implied
  connected-only score 0.6847241 (−0.00209, bit-identical connected runs). Evidence:
  `evaluation_results/cycle-78-train.json` (evaluation 8b2b1177ffc94990953b2172705ff109);
  `full_log.md#cycle-78`.
- cycle 79 — candidate `336c47f6d09df544e3b314691af989f21ed6174e`, champion
  `1ba6177625d7e7ff315c5995188815383eda7a4b`: bounded repair 1 of cycle 78, the same correction gated by
  `_has_tr_internals` (connected systems only; the gate is read at every step, so a mid-run fragment split/join
  switches it — 4 train runs differ from cycle 78's connected runs). Train 0.6846728 (Δ −0.00214, credit +1.331
  unchanged), valid 0.6898550 (Δ −0.00260, credit +2.925 unchanged), 469 / 465 converged, 196 / 280 s; decision
  **keep**. Multi-fragment bit-identical on both splits. Connected train −51 calls (42 : 176 : 39): far walks
  rmsd₀ ≥ 0.7 Å −68 (18 : 8), mid +10, near-equilibrium +7 (±1–2); valid −44 (67 : 152 : 43): far walks −40
  (38 : 18), heavy > 15 −51 (49 : 23), no basin change. Evidence: `evaluation_results/cycle-79-train.json`
  (evaluation 7fe7b950db2e4974bf52187407342661), `evaluation_results/cycle-79-valid.json`; `full_log.md#cycle-79`.
- cycle 105 — candidate `5f5145d362f2eda879ab20cebde4bddbf2f4bdb9`, champion
  `cf2439f9ee81bc0e65f361fad971390d682ad652`: the cycle-79 step applied to the inter-fragment block as well for
  multi-fragment systems whose fragments are all neutral closed-shell molecules by the docking's formal-charge rules
  (`_fragments_formally_neutral`, `Sella(curved_step_fragments=...)`, gate in `_curved_step`); charged, bare-ion and
  odd-electron complexes and connected systems bit-identical (336 / 336 runs). Train 0.6518435 (Δ +0.00055, mean rel
  energy 1.0033719, credit unchanged), 469/469 converged, 56 s; decision **discard** (train gate; valid not run).
  Read-out (`/tmp/cmp105.py`): neutral same-basin 127 runs −9 calls (43 : 34), 6 basin changes −2 calls; by champion
  call count [16,22) −18 (17 : 8), [0,12) +5 (8 : 11), [22,30) −1; by reference: ref < 20 +10 calls (+0.0013,
  2 : 6), 20–30 +4 (+0.0003), 30–45 −4 (−0.0003, 17 : 10), ≥ 45 −19 (−0.0006, 20 : 10) — the cycle-79 pattern
  (long walks gain, near-equilibrium runs lose ±1–2) on a population whose short runs carry the small references.
  Implementation preserved in `ideas/curved-step-analytic-model/5f5145d362f2eda879ab20cebde4bddbf2f4bdb9/algo.py`.
  Evidence: `evaluation_results/cycle-105-train.json` (evaluation 91b80c41a49c497c88a4924c3a560fc8);
  `full_log.md#cycle-105`.

## rotor-secant-stiffness: per-rotor secant learning of the torsional stiffness from each step's torque change (connected systems)
Status: rejected (cycle 58, discard: train +0.0241, same-basin +1.18 ± 0.24 calls per connected molecule; valid not
run). Implementation preserved in `ideas/rotor-secant-stiffness/50710f5f24e0facec71ab0fb309b185b1d2787a6/algo.py`
(reusable: `Internals.guess_hessian` records the acyclic rotors as groups of dihedral indices (`_rotor_groups`,
ring bonds up to eight atoms excluded) and the guess diagonal (`_h0_diag_last`); `InternalPES._learn_rotor_stiffness`
— torque τ = u·g, rotation dphi = u·dx/n, secant K = u·dg/dphi per rotor, rank-one shift H += ((K_new − uᵀHu)/n²) uuᵀ
along the rigid-rotation indicator vector u, applied before the multi-secant update).
**Hypothesis:** the torsional guess is the weak part of the model (class factors bring the average within 1.3× of the
physical value, individual rotors scatter 2–5×) and the rotors are the slow modes whose model error sets the linear
rate of the endgame; the quasi-Newton update cannot extract a soft rotor's curvature from a step dominated by the
stiff relaxations (the rank-two correction changes the soft component by (h_A s_A)²/(h_B s_B²) ≈ nothing), whereas
the torque change of the rotor read directly off the gradient difference, u·dg/dphi, is one secant fact per moving
rotor per force call. Setting uᵀHu to that value (rails [0.2, 5] × guess, |dphi| ≥ 0.05 rad, no pair transport)
was expected to give −0.005 to −0.02 per split; rotor-free molecules and dimers bit-identical (controls).
**Outcome and uncertainty:** refuted, monotonically in the number of rotors. Train 0.74351 (Δ +0.0241; mean rel
energy 1.00151, margin +0.71). Controls hold: 212/212 dimers and 68/68 rotor-free connected molecules bit-identical.
Connected 42 better / 88 worse / 84 identical; same-basin +1.18 ± 0.24 calls (+0.0208), six endpoint changes
+0.0034 (104079126 25 → 67 into a basin 49 kJ/mol deeper, 104046121 25 → 42, 135186987 23 → 12). By number of
acyclic rotors: 1 rotor +0.19 (3:5), 2 −0.33 (4:8), 3–4 +0.62 (15:14), 5–7 +2.22 (12:23), ≥ 8 +3.98 (8:38). By
far-starting rotors (class phase m0 < 0.5): 0–2 far +0.1 to +0.5, ≥ 3 far +3.62 (17:53). By the largest rotation of
the champion path: 3–10° +0.77 (0:6), 10–30° +1.14 (15:27), 30–60° +2.02, ≥ 60° +2.95 — the damage is there for
small rotations too. By champion run length: 8–12 calls +0.23, 12–16 +1.01, 16–24 +1.03, 24–40 +4.72, ≥ 40 +9.33.
Even the single-rotor molecules (the cleanest case) show no gain. The two failure mechanisms are intrinsic: (i) for a
soft rotor (K ≈ 0.02–0.1 eV/rad²) the torque change of its own rotation, K·dphi ≈ 0.003 eV/rad at 0.05–0.1 rad, is
of the size of the torque changes the bends, stretches and the other rotors relaxing in the same step induce through
the 1–4/1–5 couplings, so the small-rotation "secant" is noise — the docstring's premise that the couplings are small
against K·dphi holds only for stiff coordinates; (ii) over a large rotation the secant of a cosine-type well is
always below the curvature at the minimum (|sin a − sin b| ≤ |a − b|), and because a rotor that has stopped turning
(|dphi| < 0.05) is never re-measured while the multi-secant update, by the hypothesis' own argument, barely touches
it, the too-soft large-span value is frozen into the endgame (oscillation instead of contraction). Same lesson as
the adaptive-type-scale (cycles 30/31): per-coordinate secant extraction from a full-molecule step is ill-posed;
the multi-secant update with its dependency/consistency filters is the right consumer of the pairs, and it does learn
the rotors where it matters (the endgame, when the soft modes dominate the steps).
**Reason to revisit:** none for the mechanism as such. A "stiffen-only" rail (K_new = max(K_cur, K_meas)), or a
window of small rotations only, are refuted by the same table (small rotations 0:6; the losses grow with the number
of simultaneously moving rotors, which no rail fixes). The residual form uᵀ(y − Hs)/dphi is identical for a model
without torsion–torsion couplings. Do not re-derive.
**Next experiment:** none. If the rotor model is to be improved, it must be through the *guess* (structural, no
per-step measurement) or through the multi-secant update's use of the pairs, not per-coordinate secants.

**Attempts:**
- cycle 58 — candidate `50710f5f24e0facec71ab0fb309b185b1d2787a6`, champion
  `5d0a542040b30ac0f066135252c6bc80f13d072f`: train 0.74351 (Δ +0.0241, mean rel energy 1.00151),
  469/469 converged; decision **discard** (train improvement below 1e-4; valid not run). Evidence:
  `evaluation_results/cycle-58-train.json` (evaluation 92e8fcb5254946659653314ed24ed27c);
  `full_log.md#cycle-58--per-rotor-secant-learning-of-the-torsional-stiffness-connected-systems-discard`.

## retrospective-trust-region: keep the trust radius after a failed step when the secant-updated model explains its energy change (connected systems)
Status: discarded (cycle 59, +0.0022 train); closed for connected systems; not tested on multi-fragment systems (their flat surfaces benefit from the fast shrink, cycle 22)

**Hypothesis:** A failed step (Sella's ρ < 0.01 / ρ > 100) whose cause is a soft mode modelled
m > 2 times too soft is repaired by the secant update itself; the ×0.5 shrink then only
truncates the corrected next step (|1 − m|/m > 0.5 of the overshoot for every m > 2). The
retrospective ratio ρ_post = ΔE/(g·s + ½ s·y) (Bastin et al., Math. Program. 123, 395
(2010): judge the step with the updated model — here exact along s, the trapezoid of the two
gradients) equals 1 on any quadratic surface and deviates only through anharmonicity, so
halving only when ρ_post is outside Sella's good-agreement window (0.75, 1.33) should save
the truncated step in the endgame while keeping the shrink for anharmonic failures.
**Outcome and uncertainty:** Train 0.7215782 vs 0.7193841: multi-fragment runs bit-identical,
connected 224/257 same call count, 14 better (−19) / 19 worse (+74). Runs of 8–24 calls
neutral (−0.03…+0.02), medium rotor bins ≤ −0.19; losses concentrated in long rotor-rich
runs (≥ 8 rotors +0.78, ≥ 3 far-starting rotors +0.72, rotations ≥ 60° +0.84, 24–40-call
runs +1.28; 134997543 32 → 51). Reading: the exemptions fire on mid-phase 0.2–0.5 rad rotor
steps (cosine surfaces are quadratic-consistent within the window, ρ_post ≈ 1.03–1.24) whose
secant curvature is the chord of the well and therefore too soft — the shrink of cycle 21
protected exactly these steps (cycle 20's growth-only tail reappears). The endgame case is
rare or cheap: moderate overshoots (m ≈ 2–2.5) give ρ_post ≈ 2/3 (quadratic part cancels) and
for m ≈ 2 the cap equals the needed correction anyway. Uncertainty: no trajectories (all runs
converged, no bundles), so the ρ_post distribution over failed steps is inferred, not measured.
**Reason to revisit:** Only with a rotor model whose secant curvature is not systematically
soft (a chord-corrected torsional update), or as a diagnostic quantity — ρ_post is a cheap,
model-independent anharmonicity meter per step that could qualify secant pairs (drop pairs
with ρ_post far from 1 instead of the pairwise consistency filter) rather than steer the radius.
**Next experiment:** none on the radius; a pair-quality use of ρ_post (secant filter) is a
different idea and untested.

**Attempts:**
- Cycle 59; candidate `d4a0cfde39ae85e0b945854754d31b24b4cfd44d`; champion `5d0a542040b30ac0f066135252c6bc80f13d072f`; decision discard (train 0.7215782 vs 0.7193841; connected net +55 calls, dimers identical).
  [Implementation](ideas/retrospective-trust-region/d4a0cfde39ae85e0b945854754d31b24b4cfd44d/algo.py).
  [Training evidence](evaluation_results/cycle-59-train.json) (9762f55cbaab805a6d19d3b794ee8e76).
  [Narrative](full_log.md#cycle-59--retrospective-trust-radius-shrink-test-for-connected-systems-discard).

## donor-stretch-softening: X–H stretch curvature of hydrogen-bond donors after the ν(XH)–d(H···Y) correlation (short contacts)
Status: discarded (cycle 61, +0.0006 train, same-basin +0.0013); closed as an equilibrium-correlation factor — the target class (shared-proton endgames) showed no systematic gain

**Hypothesis:** For the shared-proton ion pairs (N–H⁺···N/O⁻, O–H···O⁻ with d(H···Y) 1.35–1.5 Å,
X–H 1.07–1.24 Å; ammoniums/imidazolium/guanidiniums with pyridine, amines, carboxylates;
train share ≈ 0.03, mostly slower than the reference by 3–7 calls) the tracked Almlöf stretch
(0.36 exp(−3.67 Δr/Å)) is 2–6× too stiff along the proton coordinate: the Novak/Libowitzky
correlation ν(OH) = 3632 − 1.79·10⁶ exp(−d(H···O)/0.2146 Å) cm⁻¹ (Libowitzky, Monatsh.
Chem. 130, 1047 (1999)) gives the curvature ratio f(e) = (1 − 5.37 exp(−e/0.2146))² with
e = d − r_H − r_Y, which agrees with the tracked exponential at e ≈ 0.7 Å (d(H···O) 1.67,
both ≈ 0.34 Ha/Bohr²) and drops to 0.30 (vs 0.75) at 1.5 Å and 0.08 (vs 0.63) at 1.4 Å.
Scaling each donor X–H (X, Y heteroatoms; Y a non-local partner) by f(e_min)/f(0.7), floor
0.05, in the guess and the tracked stretch diagonal should shorten those endgames by 1–3
calls each (pot ≈ 0.003–0.005 per split) with little lottery exposure (acts in formed contacts).
**Outcome and uncertainty:** Train 0.7200242 vs 0.7193841 (Δ +0.0006), mean rel energy
1.00158, 469/469; connected −0.0002 (2 molecules changed), multi-fragment +0.0009 (20 better /
24 worse / 5 basin changes, net −0.0006 from the basin changes, +0.0013 same-basin). The 51
affected molecules (e_min < 0.7 at start or champion endpoint) split three ways: (a) contacts
short only at the start (16 same-basin, compressed non-equilibrium X–H···Y; pyrrole__water
14 → 18, ethers__phenol 9 → 14, alcohols__pyridine 17 → 19, amides/guanidiniums__sulfides +2)
+0.56 calls — the equilibrium correlation misfires there because the X–H bond is not weakened
in a compressed contact (the pair term already carries the wall); (b) contacts formed during
the run (26 same-basin) +0.19 (10/10), and within them the true shared protons (X–H ≥ 1.07,
13 molecules) 0.00 on average (amines__imidazolium 23 → 20, carboxylates__guanidiniums 14 →
10, acids__ammoniums 15 → 12, carboxylates__pyrrole 37 → 30 against acids__carboxylates 16 →
19, amides__imidazolium 21 → 25, ammoniums__ethers 13 → 17, guanidiniums__pyrrole 35 → 45 —
the last with the pyrrole nitrogen as "acceptor", not a hydrogen bond); (c) halide anion
acceptors (F⁻, Cl⁻: their stiffness radius is the C–X covalent one, so e = 0.6 is an
ordinary N–H···Cl⁻ contact at 1.94 Å) +4/+6. Reading: the secant transport already learns the
proton-coordinate curvature within one or two steps, so the initial 2–6× stiffness costs
nothing systematic; what limits those endgames is the anharmonic (flat/double-well) surface
along the proton coordinate, which no curvature scaling fixes. Uncertainty: single split,
±3–4-call scatter per molecule in the target class; no trajectories (all runs converged).
**Reason to revisit:** Only as part of a non-quadratic treatment of the proton coordinate
(e.g. a Morse/double-well line model along X–H···Y), or if a later model change makes the
early proton steps matter (they do not now). An elongation-gated variant (apply only when
r(X–H) − r_cov > 0.03 Å as well, and exclude halide acceptors) would remove classes (a) and
(c) but the remaining class (b) showed no gain, so it is not a repair worth a candidate.
**Next experiment:** none as a stiffness factor.

**Attempts:**
- Cycle 61; candidate `f4fd1efa1f715f94ffc5600745a7f8df1e9465a0`; champion `5d0a542040b30ac0f066135252c6bc80f13d072f`; decision discard (train 0.7200242 vs 0.7193841, Δ +0.0006, mean rel energy 1.00158; connected 255/257 identical; affected 51 molecules: compressed starts +0.56 calls, formed contacts +0.19, shared protons 0.00).
  [Implementation](ideas/donor-stretch-softening/f4fd1efa1f715f94ffc5600745a7f8df1e9465a0/algo.py).
  [Training evidence](evaluation_results/cycle-61-train.json) (f2e368a98799a56cb36701d5460e75f8).
  [Narrative](full_log.md#cycle-61--donor-xh-stretch-softening-after-the-νxhdhy-correlation-discard).

## heavy-row-stiffness: row-dependent stiffness of the model Hessian for heavier p-block elements
Status: cycle 64 (stretches) keep — champion `6c6efdf` (train 0.7095286 / valid 0.7225279); cycle 66 (boron
stretch factor 0.65 + Si 0.70 / Ge 0.60 apex bend factors) non_generalizable (train −0.0021, valid +0.0003; class
too small for the gate) — parked; remaining: torsions about heavy central bonds, s-block/d-block bonds of connected
systems, a dative-contact term (all tiny classes).

**Hypothesis.** The Almlöf/Fischer–Almlöf guesses are calibrated on first-row bonds; bonds to Si, P, S, Cl, Br, I …
are 1.6–5× softer than the universal exponential says (C–Cl 3.4 vs 5.8, Br–Br 2.5 vs 8.7, I–I 1.7 vs 8.4 mdyn/Å),
and the multi-secant learns one mis-modelled mode per step, so the tiny inorganic molecules (4–6 atoms, ref 8–11)
ran at the reference's own count. A per-row factor (row 3 0.58, row 4 0.50, row 5 0.42, row 6 0.35 per atom,
product) reproduces the C–X constants and leaves X–X / H–X 10–30 % soft. The bend statement first written here (H–S–H 0.43 vs
F–A 0.74, Cl–Si–Cl ~0.4 vs 1.1 mdyn Å/rad²) mixed units; the corrected static Wilson-GF fit (`/tmp/gf65.py`,
ratio (F–A/2)/(fitted valence bend constant)): CH4 1.17, NH3 1.30, H2O 0.93, H2S 1.04, PH3 1.08, AsH3 1.25, CCl4
1.25, PCl3 0.78, PBr3 0.91, SCl2 0.89, SiF2 0.80 — bends at S/P/O centres are about right (the halides soft), while
the group-13/14 heavy centres are over-stiff: SiH4 1.30, SiCl4 1.48, SiBr4 1.86, GeH4 1.51, GeCl4 1.82, BF3 1.23,
BCl3 1.37, BBr3 1.50 (the F–A factor (r_cov,1·r_cov,2)^0.42 grows with the bond lengths). Boron stretches are
1.4–1.7× over-stiff as well (BF3 k 2.6, BCl3 3.5, BBr3 2.8 mdyn/Å vs the model's 4–5 at those lengths; boron has no
row factor).

**Outcome and uncertainty.** Cycle 64: −0.0082 / −0.0027, controls bit-identical (multi-fragment, heavy-free), tiny
heavy molecules 8–9 → 6–7 calls, same basins; five train basin changes (net +0.002 dscore, monoatomics__sulfides
credit +0.057 lost, 104079126 credit +0.022 gained). The row constants are calibration values from textbook force
constants — not to be scanned on the evaluation.
Cycle 66 (`ideas/heavy-row-stiffness/674a119482ad8325afdeb56c00d20a2684d3bdca/algo.py`): controls bit-identical on
both splits; train B/Si class (35 molecules) 16 : 5, same basin Σ −20 calls (tiny B/Si molecules 8 → 7, CB₄ 11 → 8,
B₈S₁₆ 25 → 22, C₆H₁₈B₂N₄O₂ 33 → 28), two basin changes Σ −2; valid class (17) 5 : 5, Σ +2, driven by one boratrane
cage (135100509, 13 → 17; B(OR)₃ pyramidalised by a transannular N···B contact at 2.54 Å, a 1,5 pair in no model
term). Same-basin sum over both splits −18 calls on 52 molecules (−0.35 per molecule, the cycle-64 magnitude) but
the valid class (6 B + 11 Si molecules) cannot resolve it; the multi-start GF fit gave BF₃ 0.90 (the earlier 1.23 was
a spurious minimum), so no boron apex factor was included. The factors are textbook calibrations — not to be scanned.

**Reason to revisit.** 6–7 calls for a 4-atom molecule is ~2× the harmonic minimum; the bends at heavy centres are
the next mis-modelled class; the umbrella improper at a heavy centre keeps the plain F–A torsion value (0.002
Ha/rad², soft but redundant with the three bends).

**Next experiment.** Do not re-run the cycle-66 candidate alone (its class is 17 valid molecules; the gate is a coin
toss). If another textbook calibration of a small primitive class (linear bends, dative contacts) lands in the
noise too, bundle the boron/Si factors with it as one "textbook calibration of the remaining primitive classes"
candidate so the affected set is large enough for the gate — attribution then comes from the class read-out with
bit-identical controls. A dative-contact term (lone-pair N/O/P donor 2.0–2.8 Å from a trigonal B/Al acceptor,
graph distance ≥ 4) would fix the boratrane class but is a handful of molecules.

**Attempts**
- cycle 64, candidate `6c6efdf2e03120e3dbb7daebae4ba6baa2f73377` (row factors on `_h0_bond`, connected only) vs
  champion `a09dbb5d39a63bd0fdb4b4bdb176812d8af2c023`: **keep** (train 0.7095286 vs 0.7177764, valid 0.7225279 vs
  0.7252031). Evidence `evaluation_results/cycle-64-train.json` (1efefd0f933c48e59b2722a6fb6ecee5),
  `evaluation_results/cycle-64-valid.json` (ee8fa8398633404ba245e3359a82c4e0); `full_log.md` cycle 64.
- cycle 66, candidate `674a119482ad8325afdeb56c00d20a2684d3bdca` (boron stretch factor 0.65, Si 0.70 / Ge,Sn,Pb 0.60
  apex bend factors, connected only) vs champion `2a8fb394fa396f8111c886bffafdaa97b190f84f`: **non_generalizable**
  (train 0.7060164 vs 0.7080763, valid 0.7213086 vs 0.7210370). Evidence `evaluation_results/cycle-66-train.json`
  (4cf71e4240084bca8ccd293f86a2fc66), `evaluation_results/cycle-66-valid.json` (ae1f650c9eaa44b6891ca4d2083a8a3e);
  `full_log.md` cycle 66
  (#cycle-66--group-1314-heavy-centre-calibration-of-the-model-hessian-boron-stretch-factor-and-sige-apex-bend-factors-non_generalizable);
  implementation `ideas/heavy-row-stiffness/674a119482ad8325afdeb56c00d20a2684d3bdca/algo.py`.

## torsion-cosine-transport: phase-tracked torsional stiffness of the three-fold rotors in the analytic model, transported with the secant pairs (connected systems)
Status: closed (cycle 68 discard: train +0.00237; cycle 69 repair 1 non_generalizable: train −0.00368 / valid
+0.00093, same-basin effect neutral across both splits). Implementations preserved in
`ideas/torsion-cosine-transport/e0226284cc800da499e17ebca802aa9325303ab1/algo.py` (local curvature −h cos 3φ) and
`ideas/torsion-cosine-transport/233eaf5f4692688b4502a8579fb8f20dc12b1ba9/algo.py` (secant stiffness h·sinc(3ψ);
reusable parts: module-level `_ROTOR_CENTRE_Z`, `_dihedral_angles`; `Internals._threefold_rotor` class rule;
`_torsion_cos_table` built in `guess_hessian`; `_h0_torsion_rotor_diagonal`; the torsional diagonal added to `d`
in `InternalPES._analytic_model_from`; `shadow_copy` shares the table).

**Hypothesis:** a three-fold rotor's curvature is −h cos 3φ (h the staggered guess), so secant pairs measured over
a 30–60° turn carry a path average well below h, which the multi-secant memory re-imposes in the endgame; tracking
the cosine in T(x) and transporting the pairs with it (the cycle-53 mechanism) should save 1–2 calls on molecules
whose rotors start off-staggered and turn > 30°.
**Outcome and uncertainty:** controls bit-identical (212 multi-fragment, 142 connected without a class rotor);
affected 115: 28 better / 30 worse, Σ +1 call, dC_S +0.00237 (loss concentrated in short-reference and 1–3-rotor
molecules; nat ≥ 41 Σ −36, ≥ 4 rotors Σ −32). Turn-size groups do not support the lag hypothesis (nbig = 1 Σ +14,
nbig ≥ 2 Σ −16; staggered starts Σ 0). The static amplitude census (`/tmp/amp68.py`) explains the pattern as a
phase-dependent *softening*: the guess is calibrated for C–C rotors (ratio 1.12 to 9V₃/2 from a 0.33 kcal/mol
per-pair barrier rule) but 1.5–3× over-stiff for O–C/N–C/C–O and > 3× for O–P/O–Si/O–S; molecules whose largest
ratio is 1.5–3 gained (47, 14 : 6, Σ −40), > 3 lost (23, 5 : 11, Σ +26: Cl₃Si–O–SiCl₃ 10 → 22 with a basin change,
phosphonate esters 25 → 33), ≤ 1.5 neutral with fat tails (BF₇Si₂ 25 → 36 on a 0.0015 Ha/rad² Si–Si rotor). The
local curvature is the wrong stiffness for a quasi-Newton step far from the minimum (zero at 30°, negative beyond →
free/trust-limited modes); the step that lands on the minimum needs the secant V′(ψ)/ψ = h·sinc(3ψ).
Cycle 69 (secant form, floor 0.05): the targeted failure mode disappeared (Cl₆OSi₂ 10 → 14 instead of 22, BF₇Si₂
25 → 26 instead of 36; > 3 group 5 : 6 train / 6 : 11 valid), but there is no same-basin signal: train affected 115
29 : 17 Σ −55 of which 3 basin changes Σ −18; valid affected 142 26 : 28 Σ +8 of which 2 basin changes Σ −18 (one an
energy credit 13 kcal/mol above the champion's minimum); same-basin combined 252 molecules 52 : 44, Σ −11. Every
train group that gained flipped sign on valid (near-eclipsed −39 → +5, off-staggered over-stiff −15 → +17,
nat ≥ 41 −12 → +30, one-rotor −17 → +18: softening a single rotor changes the path, not the endgame). Phase-tracked
rotor stiffness of either form is neutral for same-basin runs because the pairs re-learn the rotor within a step
or two and the endgame runs at staggered phase where both forms equal h.
**Reason to revisit:** none identified. The rotor-turn premium (≈ +8 calls per big-turn molecule) is path length,
not stiffness lag; a stiffness change only re-rolls paths. Repair 2 (amplitude cap h − a + a·sinc, a = min(h,
9V₃/2)) removes a coin-flip group and keeps sub-noise groups — expected value ≈ 0; static hetero-rotor
recalibration is the closed class-factor family (cycles 40/41).
**Next experiment:** none (line closed). Reuse only the class rule `_threefold_rotor` / `_dihedral_angles` if a
different mechanism needs a rotor census (e.g. a rotor-aware step bound), never the phase-tracked diagonal.

**Attempts:**
- cycle 68 — candidate `e0226284cc800da499e17ebca802aa9325303ab1`, champion
  `2a8fb394fa396f8111c886bffafdaa97b190f84f`: train 0.7104440 (Δ +0.00237, mean rel energy 1.0014898), 469/469
  converged; decision **discard** (train improvement below 1e-4; valid not run). Evidence:
  `evaluation_results/cycle-68-train.json` (evaluation f4dc8b22085d48dea174f09852cf5b87);
  `full_log.md#cycle-68--cosine-tracked-three-fold-torsional-curvature-h-cos-3φ-in-the-analytic-model-transported-with-the-secant-pairs-discard`.
- cycle 69 — candidate `233eaf5f4692688b4502a8579fb8f20dc12b1ba9` (repair 1, secant stiffness h·max(sinc(3ψ), 0.05)),
  champion `2a8fb394fa396f8111c886bffafdaa97b190f84f`: train 0.7043994 (Δ −0.00368, mean rel energy 1.0013740),
  469/469 converged, passed; valid 0.7219659 (Δ +0.00093, mean rel energy 1.0072680), 465/465 converged; decision
  **non_generalizable**. Evidence: `evaluation_results/cycle-69-train.json` (5d2bfdd050154c55b53b9c36e7f33fdf),
  `evaluation_results/cycle-69-valid.json` (da643e08687f417ab17a60be1493b3d3);
  `full_log.md#cycle-69--phase-tracked-secant-stiffness-hsinc3ψ-of-the-three-fold-rotors-in-the-analytic-model-transported-with-the-secant-pairs-non_generalizable`.
- cycle 85 (cross-reference) — the rotor census was reused for a different mechanism as the closure note allowed:
  not the phase-tracked diagonal but a zero-call start repair (`rotor-saddle-prerelaxation`, kept). The three-fold
  saddle-start population of the DES370K templates is what the amplitude census of cycle 68 saw as "staggered starts
  Σ 0" for connected molecules — in dimers the same phase is the minimum's saddle.

## surrogate-docking: rigid-body docking of neutral multi-fragment starts on a point-charge + UFF 12-6 surrogate before the quasi-Newton run
Status: kept (cycle 70: train −0.01062 / valid −0.02364, champion `0cee2f51cc2eb67496f6cd7dd1cea8b8ed9803c8`, train
0.6974612 / valid 0.6973927); **cycle 90 (keep) removed the start-geometry force call and its two xTB gates** — see
`docking-without-start-call` below (champion `d86750369416b33503bb6d929ab6ed87592b5ae3`, train 0.6711216 / valid
0.6708908). **Cycle 101 (keep) extended the scope to polyatomic-cation complexes from separated starts** (UFF 12-6
contact criterion; champion `cf2439f9ee81bc0e65f361fad971390d682ad652`, train 0.6512909 / valid 0.6499489) — see
`cation-docking`. The remaining constants (min move 0.5 Å, max move 6 Å, charge 0.22 e/Δχ, bond-order length 0.20 Å,
polar-H χ 2.57, UFF/Pauling tables, C–O(−) 1.30 Å / C–S(−) 1.75 Å) were fixed a priori and are not to be scanned on
the evaluation; the cosine gate (0.5) no longer exists.

**Hypothesis:** the approach walk of far/loose DES370K starts (10–25 calls along model-invisible rigid-body modes)
can be replaced by a classical rigid-body relaxation on a surrogate that knows the contact to ≈ 0.5 Å, at the cost
of one xTB call to verify the docked pose; near-equilibrium starts, connected systems and ionic complexes stay
bit-identical.
**Outcome and uncertainty:** same-basin dimers −0.0088 / −0.0094 per split (42 : 9 / 39 : 6), gain monotone in the
start displacement (rmsd₀ 0.5–1 Å −3 calls, 1–1.5 −4…−5, 1.5–2.5 −8…−11, ≥ 2.5 −13…−18) and largest for far/loose
starts (surrogate start energy in [−1, 0.5) kcal/mol: 32 : 7 Σ −232 train, 38 : 7 Σ −322 valid). Replicated adverse
class: starts within 0.5 Å of the endpoint (7 : 7 Σ +35 / 4 : 7 Σ +47, dC_S +0.0040 per split) — the surrogate
minimum is farther from the xTB minimum than the perturbed start (missing lone-pair directionality, π quadrupole,
C–H···O contacts 0.3 Å too loose under the UFF H wall) and the energy check passes because the start is high. Basin
changes 33 / 39, calls favourable, energy mixed (train credits +0.67, valid −0.70); the valid gate now rests on the
untouched charged dimer carboxylates__ketones (+3.216).
**Reason to revisit:** the near-equilibrium loss (≈ +0.004 per split) and the residual approach cost of the
docked-but-not-quite-there class (docked pose ≈ 0.3–0.5 Å from the minimum) are both addressable with general
criteria; the docked-pose xTB forces are computed and unused.
**Next experiment:** (1) a contact-aware acceptance: dock only starts that are outside the surrogate's contact range
(a physical criterion on the start, not on the move) or require the docked pose to beat the start by more than the
rigid-body relaxation the model itself predicts from f₀; (2) seed the secant memory with the pair (start → docked
pose, f₀ → f₁) so the first Sella step already carries the measured rigid-body curvature; (3) never extend the
docking to ionic complexes (energy-gate credits) without a valid-margin analysis first.
**Closure of the follow-ups (2026-09-18, static, no cycle consumed):** (1) contact-aware acceptance has no
consistent start criterion — the docking gain (cycle 70 minus cycle 65, docked dimers only) binned by the surrogate
energy of the start E_s, by the surrogate gain of the docking move and by the closest inter-fragment heavy-atom
contact of the start changes sign between the splits in every bin that loses on one split (E_s ∈ [−4, −2) kcal/mol
+0.0029 train / −0.0028 valid; d_min ∈ [2.0, 2.5) Å +0.0058 / −0.0071; all other bins gain on both); the
near-equilibrium losers are recognisable only from the endpoint. (2) secant seeding closed by analogy with cycle 71
(same ≤ 0.004 upside, chord-poisoning risk). (4) a Coulomb-orientation rigid-body curvature block for docked poses
(replacing the 1e-3 floor) is unsupported: at the docked minima the surrogate's rigid-body Hessian is dominated by
the 12-6 contact (median Coulomb curvature 1.8e-3 / 1.1e-3 Ha/rad² per rotation mode, i.e. floor-sized), only
1.3 / 1.6 rigid-body modes per dimer are floor-dominated in the model, and the dimers whose floor modes are stiff
in the surrogate are not the slow ones (see `/tmp/rigidhess72.py`, non-persistent). Docking follow-ups closed.

**Attempts:**
- cycle 70 — candidate `0cee2f51cc2eb67496f6cd7dd1cea8b8ed9803c8`, champion
  `2a8fb394fa396f8111c886bffafdaa97b190f84f`: train 0.6974612 (Δ −0.01062, mean rel energy 1.0028371), 469/469
  converged, passed; valid 0.6973927 (Δ −0.02364, mean rel energy 1.0062907), 465/465 converged; decision **keep**.
  Evidence: `evaluation_results/cycle-70-train.json` (0c17fc8183ee496d81bd1ab709de47ea),
  `evaluation_results/cycle-70-valid.json` (92c7852fa93241f9824b9c261d837ea1);
  `full_log.md#cycle-70--rigid-body-docking-of-neutral-multi-fragment-starts-on-a-point-charge--uff-12-6-surrogate-before-the-quasi-newton-run-keep`.
- cycle 86 (cross-reference) — [docked-pose-symmetry-breaking] discarded (train +0.0021). Standing facts for the
  docking: 93 of 133 neutral train dimers are docked; 24 docked poses are symmetric, of which acids__ketones is
  a saddle (kick → reference basin at 38 calls) and pyrrole__pyrrole an xTB local minimum 7.43 kcal/mol above the
  H-bonded dimer (a surrogate-basin error of the point-charge stacking, not a symmetry effect).
- cycle 90 (cross-reference) — [docking-without-start-call] kept: the start call was one wasted call per docked
  run; docked set now 93 + 7 (train) / 87 + 4 (valid) neutral dimers, move-gate refusals 32 / 38.

## cation-docking: surrogate docking extended to polyatomic-cation complexes (ammonium, guanidinium, imidazolium partners)
Status: **reopened and kept in cycle 101** (2026-09-19, champion `cf2439f`, train 0.6566848 → 0.6512909 / valid
0.6541644 → 0.6499489) with a physical scope — polyatomic-cation complexes are docked only from *separated* starts
(no inter-fragment atom pair inside its UFF 12-6 minimum distance x_ij = sqrt(x_i x_j), the surrogate's own contact
scale; fraction 1.0, not scanned); cation starts already in contact, anions, bare ions and odd-electron fragments stay
undocked. The earlier closure "no start criterion separates winners from losers" (below) was a scoping error:
re-binning the cycle-75 per-molecule results by the closest inter-fragment contact at the start (`/tmp/cat75.py`,
Bondi ratio bins ≥ 1.15 / 1.0–1.15 / < 1.0) puts every gain in the far bin (Σ −51 calls, no loser) and every loss
(+13 … +23) in the contact / mid bins — the surrogate cannot rank the near-degenerate contact poses of a multi-donor
cation (re-arranging an existing contact is a coin flip) but does not have to when the contact is still to be formed.
The cycles 74–75 implementations remain preserved (`ideas/cation-docking/4d19b2e6…/algo.py`,
`ideas/cation-docking/df252caf…/algo.py`); the cycle-75 charge placement (equal shares on the hydrogens bonded to the
centre or its neighbours) is the one in the champion.

**Hypothesis:** the 47 / 45 polyatomic-cation complexes (share 0.077 / 0.083, calls 19.4 / 18.9 against reference
30.1 / 28.1, 27 / 27 starts beyond 0.5 Å rmsd₀) have the same far-start approach cost as the neutral dimers and the
same surrogate residual at the xTB minima (median 0.53 / 0.39 kcal/mol/Å per atom, the 12-6 wall sets the contact),
so docking them with unchanged gates should pay −0.005 to −0.007 per split; anions, salt bridges (valid-margin
carriers) and bare ions (neutral-atom 12-6 radii, no polarisation) stay undocked.
**Outcome and uncertainty:** cycle 74 (charge on the centre) train −0.00091 but invalid (Σ(r−1) −0.487): seven docked
imidazolium runs ended in nucleophile-over-ring-carbon minima 5–45 kcal/mol above the reference's charged hydrogen
bonds because the polarity transfer left C2 at +0.70 e with amide-like N–H protons; ammonium −18 / guanidinium +14 /
imidazolium −55 calls. Cycle 75 (charge shared equally among the hydrogens on the centre and its neighbours, ESP-like:
imidazolium N–H +0.52, C2 +0.28) repaired the gate (credits Σ 0 on imidazolium, mean relE 1.00297) with imidazolium
−31 and ammonium −11 calls, but guanidinium +41 → train +0.00338, discard. The losses are three basin changes into
lower minima reached by long walks (esters__guanidiniums 11 → 34: the docked pose keeps the bidentate N–H···O contact
but the ester relaxes into another conformer, fragment-internal rmsd 0.46 Å, −0.75 kcal/mol; acids__guanidiniums
17 → 30, amides__ammoniums 25 → 47) and three near-equilibrium re-rolls (starts 0.2–0.6 Å from the endpoint docked
≥ 0.5 Å away: guanidiniums__sulfides 12 → 19, guanidiniums__ketones 13 → 21, amides__imidazolium 21 → 30). Same-basin
docked cation runs gain like neutral ones (rmsd₀ 0.5–1 Å −31 on 13, > 1 Å −15 on 3). All 422 non-cation runs
bit-identical in both cycles.
**Reason it fails (conditions tested):** a rigid-body surrogate docking of a multi-donor cation (guanidinium: six
equivalent N–H, several near-degenerate bidentate poses) lands on a pose set by the start's orientation, from which
xTB relaxes the *partner's* conformation — a lower basin at +13…+23 calls; the near-equilibrium loss is the neutral
docking's known adverse class. Neither is a charge-placement error, and no start criterion separates the −0.0022
(imidazolium/ammonium far starts) from the +0.0056 without a scan of the 0.5 Å move gate or a cation-class rule.
**Reason to revisit:** only with a start-side criterion for pose multiplicity (e.g. refuse docking when several
rigid-body surrogate minima lie within kT of the docked pose — requires extra surrogate descents, not evaluations)
or if the neutral docking's near-equilibrium criterion is ever found; the imidazolium/ammonium gain (≈ −0.002 per
split) is the ceiling.

**Attempts:**
- cycle 74 — candidate `4d19b2e6863c0e4a844422976855666bbba88bc3`, champion
  `0cee2f51cc2eb67496f6cd7dd1cea8b8ed9803c8`: train 0.6965562 (Δ −0.00091, mean rel energy 0.99896), 469/469
  converged, **invalid** (energy below baseline). Evidence: `evaluation_results/cycle-74-train.json`
  (abb4eeac69614909967e5463614b7d51); `full_log.md` cycle 74.
- cycle 75 — candidate `df252caf787d3e1cbc8a25481589415f92ed9310`, champion `0cee2f51cc2eb67496f6cd7dd1cea8b8ed9803c8`:
  train 0.7008388 (Δ +0.00338, mean rel energy 1.00297), 469/469 converged; decision **discard** (train improvement
  below 1e-4). Evidence: `evaluation_results/cycle-75-train.json` (d0dd1abdb1fc4e4fa7d61ad87464fdde);
  `full_log.md` cycle 75.
- cycle 101 — candidate `cf2439f9ee81bc0e65f361fad971390d682ad652`, anchor `3ddd4df5afc80dac0650bb5b022b448daa8a7906`
  (separated-start scope, UFF 12-6 contact criterion): train 0.6512909 (Δ −0.0053939, mean rel energy 1.0033716,
  margin 1.581), 469/469 converged, 457 bit-identical, 12 changed Σ −75 calls, no loser; valid 0.6499489
  (Δ −0.0042155, mean rel energy 1.0079088, margin 3.678), 465/465 converged, 457 bit-identical, 8 changed Σ −71
  calls (6 faster, 2 equal); decision **keep**. Basin changes net to zero within the lottery band
  (ammoniums__benzene +0.033; ammoniums__ethers −0.013, benzene__imidazolium −0.026, ammoniums__sulfides −0.005);
  the diff-basin credits imidazolium__ketones 1.029, amides__guanidiniums 1.089, amines__imidazolium 1.004 kept.
  Evidence: `evaluation_results/cycle-101-train.json` (61b40cc9bc40462eae8ba474e87ed17a, md5
  6e63eadee0661ac0a5dc5f2b751634fc), `evaluation_results/cycle-101-valid.json` (0811ac7ebca8444bbf7760764f75b1af,
  md5 1d8ce30fb83534cef457ed93d5598f93); `full_log.md` cycle 101. Static checks `/tmp/census101.py` (12 / 8 newly
  eligible, credits at stake +0.118 / +0.019), `/tmp/check101.py` (neutral class bit-identical: 133 / 130).
  Not run, with reasons: anion far starts (mostly already 13–20 calls; the carriers carboxylates__ketones +3.2 valid
  credit at separation ≈ 1.03 and carboxylates__ethers +0.36 would be at risk; anions are the most polarisable /
  charge-transfer-prone fragments for a fixed-charge surrogate; EV ≈ −0.001 per split against a valid-gate risk);
  bare-ion far starts (neutral-atom 12-6 radii are the only contact scale, the far starts are already 10–18 calls).
  Open only if a mechanism appears for the two eligible valid runs that did not change in calls
  (guanidiniums__pyrrole 32, identical pose — refused by the 0.5 Å move gate or the 6 Å cap; ammoniums__sulfides 25,
  new basin).

## torsional-docking: rotor-space pre-relaxation of connected starts on a UFF torsion + point-charge/12-6 surrogate before the quasi-Newton run
Status: rejected twice — at the design-check stage (2026-09-18, no cycle consumed; champion `0cee2f5` unchanged) with
the point-charge / 12-6 surrogate, and live in cycle 83 (2026-09-18, train Δ +0.0019849, valid not run; candidate
preserved in `ideas/torsional-docking/404554df501e3c04c63b4e911276eee1a1e6500f/algo.py`) with the attraction-free
variant below. The first prototype (rotor detection, sequential rigid-rotor map, exact rotor-space gradients — all
finite-difference checked) was kept only in a non-persistent scratch file; the cycle-83 implementation is the
preserved file (`_tors_*` block after `_dock_start`, called from `minimize_func`).

**Cycle-83 variant (attraction-free surrogate, live test):** UFF torsion rule groups per rotor (sp³–sp³ 3-fold,
group-16 2-fold, sp²–sp² 5 √(U_i U_j)(1 + 4.18 ln BO) 2-fold planar, propene rule V = 2 3-fold *per quad* on the
dihedrals ending at the doubly bonded neighbour with the eclipsed minimum), a purely repulsive wall D_ij (0.65 x_ij / r)¹²
on pairs ≥ 4 bonds apart (0.65 = the 0.01 % quantile of r / x_ij over the non-polar pairs at the xTB minima, so that
the wall sits inside every observed contact), no 12-6 attraction and no charges (both respond to the bond-angle noise
of a thermal start: with the full 12-6 the observed torsion path is downhill for 0.42 of the movers, torsion-only
0.77); minimised in the rotor angles by L-BFGS-B with the bond lengths, angles and rings frozen; docking gates
(torque cosine ≥ 0.5 with the first xTB force field, ≥ 0.5 Å rmsd after Kabsch, ≤ 6 Å per atom, xTB energy below the
start on one verification call, cached start re-installed otherwise); scope: single-fragment starts with net formal
charge 0 and an even electron count (161 / 200 candidates of 257 / 262 connected systems per split).
**Outcome:** 40 verification calls on train: 30 rejected at the energy gate (+30 calls, +0.00264), 10 accepted — 8
same-basin Σ −6 calls (5 faster by 1–5, 3 slower by 1–3), 2 basin changes into minima 1.13 / 0.41 kcal/mol higher
(credit −0.008); all 8 attempts with rmsd₀ > 1 Å (the far starts the ceiling of −0.04 … −0.10 came from) rejected;
121 candidates stopped by the call-free gates, 429 runs bit-identical. Static diagnosis (the torsion-only surrogate
is separable over the rotors, so its minimum is the per-rotor snap of the group terms — `/tmp/tors83_snap.py`): the
snapped image lies farther from the xTB endpoint than the start for 22 / 30 rejected molecules (median rmsd 0.60 →
1.04 Å) with 125 / 242 rotors turned toward the endpoint, against 7 / 10 and 48 / 67 for the accepted; the snap
runs substituents through each other (closest ≥ 1-5 contact 0.08–0.31 x_ij in 11 / 30) and the 0.65 x_ij wall stops
a C···C contact only at 2.5 Å; 1-4 clashes from frozen distorted angles carry no wall at all (135076485: C–N–C 104°
at the start vs 149° at the minimum, the planar snap puts O···C(1-4) at 2.28 Å). Under the WCA repulsive branch of
the 12-6 (`/tmp/tors83_wca.py`) the observed path's image is below the start for only 0.48 of the movers (0.47 with
1-4 pairs), best path point 0.82: a physical wall shrinks the moves rather than redirecting them.
**Reason it fails (both attempts):** the UFF torsional ideal is not where xTB's rotors go (coin-toss rotor agreement
on the rejected set), and with the skeleton frozen at a thermally distorted start the snapped conformations clash in
ways no wall inside the observed contacts can see; the accepted moves save < 1 call each because the torsional path is
short in xTB's own metric (unlike the rigid-body approach of cycle 70). Even a perfect acceptance of the 40 attempts
would have gained < 0.002 per split. Not to be repaired with wall or 1-4 variants (offline-untestable, clash side
only); any further pre-relaxation of connected starts needs a surrogate that knows xTB's rotor minima, which only
xTB's own forces provide.

**Hypothesis:** the connected analogue of cycle 70 — the walk cost of connected starts is ≈ 12 calls per Å of rmsd₀
and comes from large rotor turns (30–176° on 25 % of the molecules), so relaxing the rotors on a classical torsion +
non-bonded surrogate (one verification call, same gates as the docking) would shorten the walk the way rigid-body
docking did for dimers.
**Outcome and uncertainty:** refuted statically on the champion's endpoints (same-basin connected molecules with a
neutral, even-electron graph and ≥ 1 rotor: 160 / 197 accepted of 250 per split). The surrogate energy of the start
placed at the final rotor angles minus the start energy, by the largest rotor turn: train [0°,15°) n 33 median
−0.00 kcal/mol (17 negative), [15°,30°) n 40 −0.38 (24), [30°,60°) n 52 +0.34 (23 negative, 23 above +1), [60°,90°)
n 18 +1.85 (7), ≥ 90° n 8 +5.75 (0 negative); cosine between the surrogate descent direction and the actual turn
vector 0.57 / 0.61 / 0.47 / 0.29 / 0.02 by bin; sign agreement of the largest rotor ≈ 70 %. Valid the same (60–90°:
11 / 29 negative, cos 0.24; ≥ 90°: 5 / 13, cos 0.21). Restricted to image-faithful molecules (rigid image within
0.35 Å of the endpoint) the picture does not improve ([60°,90°) train 4 / 7 negative, cos 0.25; ≥ 90° 0 / 3). The
variants without Coulomb, without 1-4 Coulomb and without torsion terms give the same ranking (no torsions is worse
everywhere). The huge positive outliers (+451, +3553, +3932 kcal/mol) are rigid images clashing where the endpoint
also changed a nitrogen inversion or a ring pucker.
**Reason it fails (do not retry):** the PubChem starts are MMFF94s conformers plus noise (bond sd ≈ 0.025 Å, angles
≈ 5°); the start→xTB-minimum rotor turns are MMFF94s-vs-xTB conformer disagreements that a UFF-class surrogate cannot
predict (its descent has cosine ≈ 0 with the actual turn for turns ≥ 60°). Unlike the dimer approach, the connected
walk is not an information problem a classical surrogate solves; the only surrogate that knows xTB's rotor minima is
xTB itself. Any surrogate pre-relaxation of connected starts, whatever the force field, should be treated as closed
unless the surrogate is learned from xTB endpoints (not available: molecule-identity-free training data would have to
come from the evaluation itself).
**Related read-out (docked dimers):** for the 89 / 87 docked dimers, the calls after docking (17.4 / 16.6) are
uncorrelated with the surrogate's residual rigid-body gradient at the xTB minimum (correlation with its log 0.16 /
−0.03; bins 17.6 / 15.5 / 15.3 / 18.8 / 23.4 calls for residuals [0,0.25) … ≥ 2); the slowest docked runs (29–38
calls: benzene__water, alkenes__water, alcohols__pyrrole, amides__pyridine) have tiny residuals. Improving the
surrogate's accuracy (lone-pair sites, π quadrupole) would therefore not shorten the dimer endgame — the long
endgames are soft-mode walkers (small partners, water), see [[hessian-update-and-dimer-endgame]] in memory.

**Attempts:**
- Design check after cycle 70 (2026-09-18); no candidate; static refutation on the champion's endpoints (above).
- Cycle 83; candidate `404554df501e3c04c63b4e911276eee1a1e6500f`; champion `9988b8d39969333516b767fb8b7592d25976b765`; decision discard (train Δ +0.0019849, valid not run).
  [Training evidence](evaluation_results/cycle-83-train.json).
  [Narrative](full_log.md#cycle-83--torsion-space-pre-relaxation-of-neutral-connected-starts-on-a-uff-torsion--clash-wall-surrogate-rotor-angles-by-l-bfgs-b-gated-by-the-rotor-torque-cosine-with-the-first-xtb-force-field-a-05-å-minimum-move-after-kabsch-superposition-and-an-xtb-energy-decrease-start-evaluation-cached-charged-odd-electron-and-multi-fragment-systems-untouched).

## docking-trust-credit: trust-radius credit for an accepted docking move (docked runs start Sella at the fragment cap)
Status: rejected (cycle 71, train Δ −0.0000163, below the gate; valid not run; candidate
`5dfc93faf96694ea6f3ea1205ea258cc1b936c72` preserved in `ideas/docking-trust-credit/`). Closes the docking
bookkeeping follow-ups of `surrogate-docking` (credit for the move; the secant-seeding variant has the same
≤ 0.004 upside and a chord-poisoning risk).

**Hypothesis:** the accepted docking move (≥ 0.5 Å rmsd, verified xTB energy drop) is the run's first successful step
under the trust-region policy, so the docked residual walk should start at δ = δ_max_tr = 0.25 instead of growing
0.1 → 0.15 → 0.225 over three trust-limited steps along the floor-curvature rigid-body modes.
**Outcome and uncertainty:** controls bit-identical (connected, ionic, undocked neutral dimers); docked class
17.43 → 17.10 calls (Σ −29; 30 faster Σ −69 with mean ref 48, 43 unchanged, 16 slower Σ +40 with mean ref 37);
score −0.00385 + 0.00384. Winners are long H-bonded walkers with 0.3–0.5 Å residual (esters__ethers −6,
esters__esters −7, alcohols__pyridine −7, alcohols__pyrrole −5); losers are near-converged or weakly bound poses
whose residual is ≈ 0.1 Å (benzene__phenol 8 → 13, alkanes__ketones 13 → 19, acids__esters 33 → 44 on ref 14) where
the 0.25 first step (a 14° fragment rotation) overshoots the flat minimum and is halved and re-learnt.
**Reason to revisit:** only with a per-run residual-scale estimate at the docked pose (xTB rigid-body force ÷ a
rigid-body curvature); the surrogate's own curvature is as crude as the residual it leaves, and the perfect
discriminator is worth −0.0039 on train. Not worth a bounded repair.
**Next experiment:** none on the bookkeeping; a physical rigid-body curvature model for docked poses would have to
replace the 1e-3 floor without double-counting the tracked contact term.

**Attempts:**
- cycle 71 — candidate `5dfc93faf96694ea6f3ea1205ea258cc1b936c72`, champion
  `0cee2f51cc2eb67496f6cd7dd1cea8b8ed9803c8`: train 0.6974449 (Δ −0.0000163, mean rel energy 1.0028731), 469/469
  converged, below the gate; decision **discard**. Evidence: `evaluation_results/cycle-71-train.json`
  (074fd8b1a8e14f12b8ac336b2abdf7e4, md5 9e0d7a28ccdd1dab571e2138961dc41c);
  `full_log.md#cycle-71--trust-radius-credit-for-an-accepted-docking-move-docked-multi-fragment-runs-start-sella-at-the-fragment-cap`.

## ionic-hbond-cap: doubled contact-bend cap for ionic hydrogen bonds (charged donor or acceptor fragment)
Status: discarded in cycle 72 (train +0.0037; same-basin Σ 0 over 49 affected charged dimers, one walker re-roll +36 calls); ionic-contact stiffness line closed

**Hypothesis:** The contact bends' caps are those of equilibrium neutral hydrogen bonds; ionic hydrogen bonds bind
3–4× more strongly and the ion's field holds the neutral partner's orientation with μq/R² ≈ 0.02–0.05 Ha/rad²
against 0.015 for the acceptor bends at the cap (the shorter contact already gives 1.85× the water-dimer value), so
inter-fragment X–H···Y / X–H···π contacts of heteroatom donors with a charged donor or acceptor fragment (valence-rule
fragment charges, `_fragment_charges`) should use twice the cap.
**Outcome and uncertainty:** Controls bit-identical (connected, neutral dimers, charged systems without heteroatom-
donor contacts — all the energy-gate credits). Same-basin effect exactly zero (49 runs, 18 faster / 16 same / 15
slower, Σ 0 calls; X–H···Y Σ −8 / 41, π Σ +8 / 6; carbonyl acceptors Σ +3 / 16, others Σ −3 / 27). Five basin changes
decide the score: imidazolium__ketones 36 → 72 (+0.0033 alone), ammoniums__esters +7, amides__ammoniums +5,
guanidiniums__pyrrole 35 → 10 into a 0.7 kcal/mol higher minimum (−0.020 credit), alcohols__imidazolium −6.
**Reason to revisit:** none for the stiffness itself — a 2× correction of a mode that already sits 15–25× above the
floor is learnt by the update within the first steps; the endgame of the charged dimers runs along other modes
(twist about the contact, rotors of the neutral partner). Any other factor is a scan plus a re-roll of the same
walkers. The reference's advantage on near-equilibrium cation dimers (9–15 calls vs 13–14) is a coordinate-system
effect (contact bends/dihedrals as internals), not a stiffness one — that is the follow-up, see full_log cycle 72.
**Next experiment:** none on this line.

**Attempts:**
- Cycle 72; candidate `6361324ae524526681fa99ba3f032c1e56b4cdae`; champion `0cee2f51cc2eb67496f6cd7dd1cea8b8ed9803c8`; decision discard.
  [Implementation](ideas/ionic-hbond-cap/6361324ae524526681fa99ba3f032c1e56b4cdae/algo.py).
  [Training evidence](evaluation_results/cycle-72-train.json).
  [Narrative](full_log.md#cycle-72--ionic-hydrogen-bond-cap-contact-bends-of-inter-fragment-xhy--xhπ-contacts-with-a-charged-fragment-get-twice-the-neutral-cap).

## per-mode-caps: per-eigenmode step caps (box-constrained model step) instead of the Levenberg shift in the max-internal-component trust region
Status: discarded in cycle 73 (train +0.00012; connected same-basin Σ +35 over 247, two bond-breaking/unfolding hops into 23–67 kcal/mol higher minima; multi-fragment controls bit-identical); box-step line closed

**Hypothesis:** `MaxInternalStep` bounds the largest weighted internal component (and the atomic displacement /
cart_ratio) but realises the bound with the 2-norm Levenberg shift, damping every mode by λ_i/(λ_i + α) while only
the leading coordinate is trust-limited; clipping each eigenmode of the projected model at δ / max(m_i, c_i) (m_i,
c_i the internal and Cartesian measures of the unit mode), leaving the non-limiting modes their Newton amplitude and
scaling all caps by a common factor when the composed step still violates, should save the far-start steps lost to
the damping of the other rotors (20–70 %) and bends (5–20 %).
**Outcome and uncertainty:** Newton-fitting steps and all 212 multi-fragment runs bit-identical. Connected 257: 54
faster / 143 same / 60 slower, Σ −33 calls, all of it from 10 basin changes (Σ −68); same-basin Σ +35 (+0.3
calls per molecule uniformly for rmsd₀ 0.15–1.0 Å and every size class below 35 atoms; only rmsd₀ ≥ 1 Å gained, 13:7,
Σ −18 over 25). Energy credit −0.445: 172113611 unfolded (O···H 5.0 → 9.4 Å, +48.8 kcal/mol, relE 0.793), O4P2
lost a P–O bond (+67 kcal/mol, relE 0.828), 104079126 lost the champion's rearranged lower minimum (+22.8). The box
step is the exact solution of the box trust region for a diagonal model and therefore *greedy*: every mode whose
Newton amplitude g_i/λ_i exceeds its cap takes the whole cap, including barely driven soft modes with unreliable
guess curvature (chain torsions, row-factor-softened P–O stretches), while the Levenberg shift gives them g_i/α —
steepest-descent amplitude proportional to the force. That Tikhonov damping is a regulariser against curvature
error, not a cost: the update learns the leading mode's curvature from steps concentrated on it, and the mid-range
starts lose 0.3 calls when the step is spread over many capped modes.
**Reason to revisit:** only with a principled trust criterion per mode (e.g. box amplitude for modes inside the span
of the measured secant pairs, shift-damped amplitude for guess-only modes); gradient-threshold hybrids are scans;
dogleg / 2-D subspace steps are force-proportional like the shift and would not change the leading-mode step.
**Next experiment:** none on this line.

**Attempts:**
- Cycle 73; candidate `3fc0b796dd8b733321ee5609fb9c55768621d74c`; champion `0cee2f51cc2eb67496f6cd7dd1cea8b8ed9803c8`; decision discard.
  [Implementation](ideas/per-mode-caps/3fc0b796dd8b733321ee5609fb9c55768621d74c/algo.py).
  [Training evidence](evaluation_results/cycle-73-train.json).
  [Narrative](full_log.md#cycle-73--per-eigenmode-step-caps-in-the-max-internal-component-trust-region-connected-systems-the-levenberg-shift-replaced-by-a-box-constrained-model-step).

## symmetry-image-secant-pairs: images of the stored secant pairs under the point-group operations of the current geometry
Status: kept (cycle 76: train −0.01064 / valid −0.00494, champion `1ba6177625d7e7ff315c5995188815383eda7a4b`, train
0.6868163 / valid 0.6924513); multi-fragment extension refuted (cycle 77, discard, train +0.00723: the class has no
free antisymmetric information above the noise of its softest modes — water complexes 1 : 8, symmetric endgames
+28 same-basin calls — and symmetric-start ionic complexes leave their symmetric endpoint at +14 / +22 calls).
Constants fixed a priori (rmsd tolerance max(0.05 Å, ½ rms step), distance-pruning 3× the tolerance, 48
permutations / 20000 nodes per search) — not to be scanned.

**Hypothesis:** the multi-secant update learns one direction per force call; xTB's energy is exactly invariant under
permutations of identical nuclei and every orthogonal map, so at a (nearly) symmetric geometry every stored pair has
an exact image under each point-group operation (permutation + rotation/reflection), and entering the images after
their real pairs (same dependence 0.3 / consistency 0.25 filters, memory of real pairs unchanged) multiplies the
learning rate by the group order for the symmetric-final class (96 train / 43 valid connected endpoints).
**Mechanism:** `_isometric_permutations` (distance-pruned backtracking over same-element atoms), `_symmetry_operations`
(Kabsch fit in both determinant branches), `Internals.symmetry_maps` (dummies of linear bends must map onto
themselves or their inversion through the centre), `InternalPES._symmetry_images` (image maps through Cartesian space
with the current Jacobian: s' = B G B⁺ s, y' = B⁺ᵀ G Bᵀ y — a signed permutation of the coordinates when the list is
closed under the operation, the exact first-order image otherwise; Sella's single improper dihedral at symmetric
centres breaks the closure for the tiny inorganic molecules, 175 of 403 operations), `PES._collect_secant_pairs`
(images after each real pair). Connected systems only (fragment translations/rotations are not internal coordinates
of a displacement field); asymmetric molecules and multi-fragment starts bit-identical.
**Outcome:** gain proportional to the group order: 1 operation −2.9 % of the class's calls (47 train molecules),
2–3 operations −5 %, 4–7 operations −17 % (13 molecules, 8 : 5 : 0), 12–23 operations −18 % (7 molecules, 6 : 1 : 0);
train 43 : 25 : 8 better : same : worse (losses ≤ 2 calls), valid 20 : 18 : 2; no basin changes; 4 valid dimers that
Sella joins into one fragment −4 calls. Symmetric at the start 192 → 176 calls, symmetric only at the end 968 → 914.
**Reason to revisit:** 13 operations on 6 train molecules are dropped at the dummy atoms of chains on a rotation axis
(the mirrored dummy lands at another azimuth); the near-symmetric class (0.05 < ε < 0.3 Å at the endpoint, 46 train
molecules) has no exact images.
**Next experiment:** (1) rotate the image dummy displacement about the chain axis into the frame of the current dummy
(general rule replacing the two-case identity/inversion rule; ≤ 0.001 expected); (2) do not build weighted or
component-wise images for near-symmetric geometries — the filters cannot vet their errors; (3) multi-fragment
systems: done (cycle 77, below) — not to be repaired: a fixed 0.05 Å tolerance for fragment systems would remove
only the random mid-run detections (15 runs ≥ 0.15 Å from any operation at both ends, Σ ≈ +6 calls) and leave the
symmetric-endgame (+28) and symmetric-start (+37 with two basin changers, +1 without) losses; neutral-only or
"operations present at the perturbed start" exclusions are credit protection without a mechanism.
**Cycle-77 reading (multi-fragment images, `/tmp/cmp77*.py`):** the TRIC image map is exact to first order (73 / 77
symmetric multi-fragment endpoints, 87 / 81 operations all kept, finite-difference vs linear ≤ 0.7 %, gradient map
exact) and the analytic model T is covariant under the detected operations (≤ 5.5 %), so the loss is not an
implementation artefact. Connected molecules bit-identical; 59 of 213 multi-fragment runs changed, 28 : 30 better :
worse, three basin changes (esters__esters 28 → 24 but 3.0 kcal/mol higher; amides__carboxylates 15 → 29 and
acids__guanidiniums 17 → 39 to minima 1.44 / 0.24 kcal/mol deeper — the images taught the antisymmetric curvature
of a symmetric endpoint the champion and the reference accept as converged); without them Σ +1.22 rel. In a dimer
endgame the steps are almost totally symmetric (0.02 Å rigid-body seed), so an image is rejected as dependent or,
when admitted (≥ 15 % antisymmetric content), separates the curvature of floor-stiffness fragment rotations from
gradient differences over 0.002–0.02 Å — noise-level information (water complexes 415 → 444 calls).

**Attempts:**
- cycle 76 — candidate `1ba6177625d7e7ff315c5995188815383eda7a4b`, champion
  `0cee2f51cc2eb67496f6cd7dd1cea8b8ed9803c8`: train 0.6868163 (Δ −0.01064, mean rel energy 1.0028371), 469/469
  converged, passed; valid 0.6924513 (Δ −0.00494, mean rel energy 1.0062908), 465/465 converged; decision **keep**.
  Evidence: `evaluation_results/cycle-76-train.json` (ab1de02139904c3585cd90945ed6058f),
  `evaluation_results/cycle-76-valid.json` (19c9d280a058447ab101694030b27e7f);
  `full_log.md#cycle-76--symmetry-image-secant-pairs`.
- cycle 77 (multi-fragment extension: `Internals.symmetry_maps` no longer refuses systems with fragment
  translation/rotation coordinates; everything else unchanged) — candidate `b672269cfca1cfede7e896b36a31e816bd9e2ddd`,
  champion `1ba6177625d7e7ff315c5995188815383eda7a4b`: train 0.6940499 (Δ +0.00723, mean rel energy 1.0028263),
  469/469 converged; decision **discard** (valid not run).
  [Implementation](ideas/symmetry-image-secant-pairs/b672269cfca1cfede7e896b36a31e816bd9e2ddd/algo.py).
  [Training evidence](evaluation_results/cycle-77-train.json) (6242c3168c1c4bf795116d642350b351).
  [Narrative](full_log.md#cycle-77--symmetry-image-secant-pairs-for-multi-fragment-systems-the-point-group-operations-of-the-whole-complex-fragments-included-feed-the-multi-secant-memory-through-the-same-cartesian-pathway-image-map).

## planar-centre-impropers: out-of-plane (improper) coordinates for planar π centres whose proper dihedrals run about single bonds
Status: kept (cycle 80: train −0.00464 / valid −0.00753, champion `b70bec7abfce13ad2b965c2006bb0ea1abf06c49`, train
0.6800287 / valid 0.6823292). Constants fixed a priori (Schlegel's out-of-plane guess 0.045 Ha/rad², bond-order
factor threshold 1.7 for an anchoring multiple bond, angle sum ≥ 345° for planarity) — not to be scanned.

**Hypothesis:** the static model-frequency check (`/tmp/modelfreq80.py`, `/tmp/oop80.py`: H_cart = BᵀH₀B of the guess
Hessian on saved dimer-endpoint monomers, mass-weighted eigenvalues against gas-phase fundamentals) finds one class
of modes ≥ 2× too soft: the out-of-plane wag of a planar sp² centre whose π partner is a terminal atom (C=O, C=S, N=O,
B=O/S). Sella adds the umbrella improper only at centres without proper dihedrals, so a substituted carbonyl carbon's
pyramidalisation is carried by the class-scaled rotor constants of its single bonds alone (acetone wag 217 vs 484
cm⁻¹, acetic acid 245 vs 535, acetaldehyde 330 / 539 vs 509 / 763; formaldehyde with its umbrella 1079 vs 1167).
Alkene / aromatic / amide centres are anchored by the dihedrals about their multiple or conjugated bond (bond-order
factor ≥ 1.9; benzene's out-of-plane modes within 10 %). Over-soft modes make the quasi-Newton step overshoot until
the update corrects them; the composition regression charged +0.94 calls per carbonyl (reference +0.89, same rule).
**Mechanism:** `Internals._add_planar_centre_impropers` (called from `find_all_dihedrals` after Sella's umbrella loop):
improper (n0, centre, n1, n2) at every three-coordinate 'pi' centre (`_torsion_centre_types`) that has proper
dihedrals, angle sum ≥ 345°, and no bond of bond-order factor ≥ 1.7 to a neighbour with further neighbours; n1 is the
terminal neighbour with the largest bond-order factor (the torsion about the terminal double bond the dihedral set
cannot contain); `guess_hessian` gives impropers whose centre has proper dihedrals `_h0_out_of_plane()` = 0.045
Ha/rad² (the Fischer–Almlöf torsional formula is not used: its (r·r_cov)⁻⁴ drops 6–8× from C=O to C=S); connected
systems only (structural gate, also active at a rebuild that joins a dimer). geomeTRIC's practice (OutOfPlane at
every planar 3-neighbour centre, 0.045) was not copied: impropers at aromatic / alkene centres would double a
stiffness that is already right.
**Outcome:** static: acetone 217 → 472, acetic acid 245 → 511, acetaldehyde 330 → 539 / 539 → 808, methyl acetate
≈ 342 → 555 cm⁻¹; amides, formaldehyde, formate, NH₃, ethylene, aromatics bit-identical. Runs: connected molecules
with new impropers train n 72 −50 calls (41 : 19 : 12; C=O −41 / 49 molecules, N=O −8 / 3, C=S −9 / 6), valid n 99 −87
(61 : 26 : 12; C=O −68 / 58, N=O −11 / 8); gain grows with the number of impropers (≥ 3: rel 0.734 → 0.666 /
0.707 → 0.653); the ≈ 10 % borderline conjugated-ring centres (all bonds bond-order 1.0–1.68) a wash (−1 / −4 over 42
molecules); multi-fragment runs bit-identical (one joined dimer 19 → 19); five small inorganic molecules ±1–4 calls
from rebuild-time impropers. Basin changes: train thioketone 134997543 34 → 28 into a basin 2.7 kcal/mol higher
(credit −0.035) and 135104646 17 → 20 (+1.28); valid two, both lower. Credits +1.287 / +2.926.
**Reason to revisit:** none for the constants. Remaining out-of-plane cases not covered: none found ≥ 2× soft.
**Next experiment:** the same static check for the other monomer classes of the dimer endpoints (alcohol / amine /
amide N–H wags, nitromethane, nitriles, DMSO, esters, phenol, aniline, five-membered heteroaromatics) to look for
further ≥ 2× soft modes; over-stiff modes (sp² in-plane bends 1.15–1.35×, NH₃ umbrella 1.5×, rotors 1.5–2.2×) are
not worth a cycle (cycles 38, 40/41, 67).

**Attempts:**
- cycle 80 — candidate `b70bec7abfce13ad2b965c2006bb0ea1abf06c49`, champion
  `336c47f6d09df544e3b314691af989f21ed6174e`: train 0.6800287 (Δ −0.00464, mean rel energy 1.0027438), 469/469
  converged, passed; valid 0.6823292 (Δ −0.00753, mean rel energy 1.0062922), 465/465 converged; decision **keep**.
  Evidence: `evaluation_results/cycle-80-train.json` (237a338d62ca43f381bed4cfe8b43803),
  `evaluation_results/cycle-80-valid.json` (924348b45a26464b9cc7a0cc6c1df134);
  `full_log.md#cycle-80--out-of-plane-improper-coordinates-for-the-planar-π-centres-whose-proper-dihedrals-run-about-single-bonds`.
- Cycle 92 (the same rule for the fragments of multi-fragment systems, connected-only gate removed, same
  constants): keep, train −0.00408 / valid −0.00603 — see `fragment-planar-centre-impropers` below
  (candidate `78ac80e6d58c13bc9f4bb1e9c2a77cae68e9f5ca`).

## ionic-angular-stiffness: ion–dipole angular stiffness for s-block metal contacts in connected systems
Status: incorporated (cycle 81 keep; constants μ₀ = 0.73 au and the formal charges fixed a priori, not to be scanned)

**Hypothesis:** A cation that starts within the covalent-radius bond cut-off of its ligand
(8 of the 14 s-block systems) is built as a connected system with bends and dihedrals through
the ion; cycle 52 gave the radial contacts the ionic scale (Shannon radii) but the bends kept the
Fischer–Almlöf guess (0.09–0.12 Ha/rad², a covalent floor) and the ion-terminal dihedrals the
covalent torsion constant of the ligand bond — 5–15× the ion–dipole stiffness q μ / R² (Na⁺···OH₂
at 2.3 Å 0.039 Ha/rad², the same as the explicit 3-point-charge water; librations 400 / 490 cm⁻¹).
Bends with the ion: q μ₀ / (2 R₁ R₂); ion-terminal proper dihedrals: q μ₀ / R² shared per contact
bond; μ₀ = 0.73 au (water); connected systems only, everything else bit-identical.
**Outcome and uncertainty:** keep — train 0.6800287 → 0.6796564 (−0.00037), valid 0.6823292 →
0.6816682 (−0.00066); exactly the eight connected ion complexes change: Mg²⁺–phenol 32 → 25,
Li⁺–butane 22 → 16, K⁺–dithiane 23 → 18, Na⁺–S(CH₃)₂ 11 → 10, Ca²⁺–ethene 6 → 5, K⁺–H₂S 20 → 22,
Na⁺–CH₄ 11 → 11, Li⁺–benzene 9 → 9; no basin change; credits unchanged. Over-stiffness of 5–15×
in k pays to remove, unlike the 1.5–2.2× cases (cycles 38, 40/41, 67).
**Reason to revisit:** none for the bonded mode (radial + angular now consistent with the
fragment-mode radial-only cation model); the six fragment-start ion complexes are untouched by
construction, two of them credit walkers (monoatomics__pyrrole__train, ethers__monoatomics__valid)
that must stay as they are.
**Next experiment:** none.

**Attempts:**
- Cycle 81; candidate `9988b8d39969333516b767fb8b7592d25976b765`; champion `b70bec7abfce13ad2b965c2006bb0ea1abf06c49`; decision keep.
  [Training evidence](evaluation_results/cycle-81-train.json).
  [Validation evidence](evaluation_results/cycle-81-valid.json).
  [Narrative](full_log.md#cycle-81--iondipole-angular-stiffness-for-the-contacts-of-s-block-metal-ions-in-connected-systems-bends-with-the-ion-q-μ₀2-r²-ion-terminal-dihedrals-q-μ₀r²-shared-per-contact-bond-μ₀--073-au-the-water-dipole-covalent-torsion-shares-no-longer-count-the-ion).

## ionic-contact-dipole-fragment: ion–dipole angular stiffness of the inter-fragment contacts of cationic fragments with lone-pair atoms (fragment mode)
Status: discarded (cycle 82, train +0.0025; same-basin Σ +10 / 36, three basin re-rolls; with cycle 72 the acceptor-orientation stiffness of charged contacts is closed; candidate preserved in `ideas/ionic-contact-dipole-fragment/6312ef0abba83727e237ccad6cb193cf25659862/algo.py`)

**Hypothesis:** The fragment-mode counterpart of `ionic-angular-stiffness` (cycle 81): every non-local
inter-fragment pair (Q, X) with Q a charge site of a cationic fragment (net valence-rule charge of
`_dock_formal_charges` spread over its positive atoms: a bare s-block ion, the N of an ammonium, the
central C of a guanidinium / imidazolium) and X an uncharged lone-pair atom (N, O, S, halogen; axis
d = −Σ û to the neighbours with |d| ≥ 0.5) of a neutral fragment gets k Jᵀ J with J the Jacobian of
v = u_XQ − d̂ (|v|² = 2 (1 − cos φ), Gauss–Newton form of the charge–dipole energy, mapped through Binv
like the contact bends) and k = q μ₀ max(cos φ, 0) / R², μ₀ = 0.73 au. Targets: the acceptor of a
charged hydrogen bond (NH₄⁺···OH₂ physical ≈ 0.04 Ha/rad² against the model's 0.015 from two capped
acceptor bends, 2–3× under-stiff) and the three fragment-start s-block σ-complexes (Na⁺···N amine,
K⁺···O ether, Na⁺···N pyrazine: lateral stiffness at the rigid-body floor, 10–30× under-stiff);
cation–π complexes, anions, salt bridges, neutral dimers and connected systems bit-identical
(census 431 / 469 and 425 / 465 identical at the start and the cycle-81 endpoints).
**Outcome and uncertainty:** train 0.6796564 → 0.6821470 (+0.0025), 469/469 converged, discard;
39 runs changed (the 38 of the census plus one whose term switches on during the approach), 430
bit-identical. Same-basin 36: Σ +10 calls (13 faster / 8 same / 15 slower), +0.00092; basin changes
+0.00157: imidazolium__ketones 36 → 55 (the cycle-72 re-roller), amides__ammoniums 25 → 37,
ammoniums__ketones 29 → 9 into a stationary point 5 kcal/mol above the reference (credit +0.000 →
−0.318). By site: ammonium (charge at the donor N, the physically right site) 270 → 259 with
same-basin Σ −3 / 11; guanidinium 221 → 224; imidazolium 247 → 275; the one train s-block σ-complex
10 → 11. The model librations moved to their physical values (guanidinium–water 233 / 237 →
368 / 466 cm⁻¹) without any same-basin gain — the second null on this mode after cycle 72's cap
doubling (Σ 0 / 49): the acceptor orientation of a charged contact already sits 15–25× above the
floor and is learnt by the update in the first steps; the charged-dimer endgame runs along the
partner's rotors, the twist about the contact and the approach path. A stiffer sideways hold can
also trap a walker at the nearest stationary point (ammoniums__ketones).
**Reason to revisit:** none for the acceptor orientation. The s-block σ-complexes (1 train / 2 valid
runs, one of them the +0.852 credit walker) cannot be evaluated alone under the gates; the
delocalised cations' charge sites (N–H's rather than the central C) would be a re-siting of a term
that is non-binding at its right site.
**Next experiment:** none. Not to be bundled again with the cation dimers; if a fragment-mode s-block
term is ever needed for another reason, it must come with a mechanism other than the acceptor
orientation.

**Attempts:**
- Cycle 82; candidate `6312ef0abba83727e237ccad6cb193cf25659862`; champion `9988b8d39969333516b767fb8b7592d25976b765`; decision discard (train Δ +0.0024906, valid not run).
  [Training evidence](evaluation_results/cycle-82-train.json).
  [Narrative](full_log.md#cycle-82--iondipole-angular-stiffness-for-the-inter-fragment-contacts-of-cationic-fragments-with-the-lone-pair-atoms-of-neutral-ones-gaussnewton-form-of-e--k-1--cos-φ-k--q-μ₀-maxcos-φ-0r²-φ-between-the-lone-pair-axis-of-the-acceptor-atom-and-the-direction-to-the-charge-site-s-block-ions-and-the-formal-charge-sites-of-ammonium--guanidinium--imidazolium-fragments-μ₀--073-au-as-cycle-81).

## class-resolved-trust-region: connected radius policy for the intramolecular coordinates of multi-fragment systems, weighted fragment translation/rotation components
Status: discarded (cycle 84, train +0.0071; same-basin dimers a wash in every rotor-perturbation bin, four charged basin re-rolls Σ +69; multi-fragment trust-region line closed on every side with cycles 44 / 71; candidate preserved in `ideas/class-resolved-trust-region/ec1d42d85add3d73d39974d3624743f656d7321d/algo.py`)

**Hypothesis:** After the docking (cycle 70) the multi-fragment runs are no longer approach-limited
(dimer calls vs whole-complex rmsd₀: 2.6 / 5.5 calls per Å against the reference's 22.7 / 28.1) but
intramolecular-relaxation-limited: 24 / 26 calls per Å of intra-fragment rmsd, twice the connected
slope, and at equal max fragment-dihedral deviation dimers cost +6 … +11 calls more than monomers
(10–35 atoms; 70 / 52 dimers with ≥ 10°). The single tight radius of the multi-fragment policy
(δ₀ 0.1, cap 0.25, both classes of coordinate) throttles the fragment rotors — a 0.7–1.2 rad rotor
needs six steps under it and three under the connected policy. Candidate: δ₀ 0.25 / cap 0.5 for all
order-0 runs, fragment translation/rotation components weighted ×2 in the max-component measure
(effective 0.125 → cap 0.25), Cartesian bound 2δ on the intramolecular part of the step only
(`_intra_mask`); connected systems bit-identical by construction.
**Outcome and uncertainty:** train 0.6796564 → 0.6867286 (+0.0071), 469/469 converged, discard;
true connected runs bit-identical (249 / 250), multi-fragment 70 : 71 : 71. Same-basin multi-fragment
206 runs Σ +1 (neutral −12 over 131, charged +13 over 75); by max fragment-dihedral deviation
[0,10) −3, [10,20) +2, [20,40) −5 (8 : 8 : 8), [40,181) +7 (12 : 5 : 13) — the targeted class did
not move. Loss = 6 basin changes Σ +68, four charged (ammoniums__benzene 17 → 53 into a −1.63 kcal/mol
cation–π pose, acids__guanidiniums 17 → 37, amides__ammoniums 25 → 42). The rotor-perturbed dimer
excess is not a step-size throttle: it is the model-invisible coupling between a fragment rotor and
the inter-fragment pose (each rotor turn is followed by rigid-body re-adjustment along floor-stiff
modes, information-limited), plus the displacement-bound endgame of the coupled soft subspace.
**Reason to revisit:** none for the radius policy (cycle 44 cap 0.5 + bound on the fragment
coordinates +0.011; cycle 71 δ₀ 0.25 for docked runs wash; this cycle the intramolecular policy
+0.0071). A neutral-only scope (−0.0006 on train) is credit protection, not a mechanism.
**Next experiment:** none on the trust region. A dimer lever must supply rotor ↔ pose coupling
information (secant pairs already do what they can) or shorten the displacement-bound endgame.

**Attempts:**
- Cycle 84; candidate `ec1d42d85add3d73d39974d3624743f656d7321d`; champion `9988b8d39969333516b767fb8b7592d25976b765`; decision discard (train Δ +0.0070723, valid not run).
  [Training evidence](evaluation_results/cycle-84-train.json) (md5 f5ea665488e6cda2b9e934e8927fc67b).
  [Narrative](full_log.md#cycle-84--class-resolved-trust-region-for-multi-fragment-systems-the-intramolecular-coordinates-of-the-fragments-take-the-connected-system-radius-policy-δ₀-025-cap-05-cartesian-bound-2δ-on-the-intramolecular-swing-the-fragment-translationrotation-components-keep-their-tight-radius-through-a-weight-of-2-in-the-max-component-measure-0125--cap-025).

## rotor-saddle-prerelaxation: XH₃ rotors that start on their three-fold saddle are placed at the class minimum phase before the first force call
Status: kept (cycle 85, 2026-09-18: train −0.0039914 / valid −0.0020480, champion
`8c1813d4560e31e2f0e609ecc06c58ce51ee2d3a`, train 0.6756649 / valid 0.6796202); extended by cycle 88
(ester acetyl class, 2026-09-18: train −0.0015824 / valid −0.0030221, champion
`8a30dec9b7cf843c34cc0feefaa961e8a84b33a8`, train 0.6740826 / valid 0.6765981). Constants fixed a priori, not to be
scanned: window 10° from the saddle (a third of the ±30° negative-curvature region of the three-fold potential;
the DES370K templates are exact), C=O / C=S thresholds 1.30 / 1.72 Å (the docking code's), planarity 350°,
circular-mean modulus ≥ 0.5, terminal atoms H only.

**Hypothesis:** the +7 calls that a methyl turning ≥ 40° costs in a dimer (against +1.2 in a connected molecule)
is saddle creep — the DES370K template monomers carry every methyl at an exactly stationary phase, which for the
N-methyl of secondary amides (staggered with respect to the acyl carbon) and one acetaldehyde methyl (staggered
with respect to C=O) is the saddle: zero torque by symmetry, negative curvature, a quasi-Newton step of ≈ 0 along
the rotor, and the 60° relaxation surfaces in the endgame (or never: runs converge on the saddle). A rule-based
three-fold surrogate whose minimum is known analytically by frame class (Hehre–Pople–Devaquet eclipsing rule for
sp² frames, staggered for sp³ and two-coordinate O / S frames; ETKDG-style torsion preferences) places such rotors
at the minimum for zero force calls; every other start is bit-identical.
**Outcome and uncertainty:** untriggered runs bit-identical (train 459 / valid 448). Train 10 triggered runs
10 : 0 : 0 Σ −68 (dimers −6 to −13 calls each, e.g. alkanes__amides 24 → 12, ammoniums__ketones 29 → 16); valid
17 runs 10 : 2 : 5 Σ −30. Saddle-stayers (ours and the reference converged on the saddle: amides__ammoniums,
amides__ethers, amides__monoatomics, guanidiniums__ketones) now reach minima 2.5–5.1 kcal/mol below the reference
at −1 to +2 calls — the staggered N-methyl also locked the complex's H-bond pattern; valid Σ(r−1) +2.926 → +3.986.
Better-basin cases follow the reference (amides__benzene 11 → 31, −5.71 kcal/mol; alkenes__amides 12 → 11, −4.53).
Exposure: environment-shifted minima the rule cannot know (acids__ketones true phase −46°, 22 → 27;
imidazolium__ketones 33 → 11 but +1.23 kcal/mol vs the champion's lower basin, equal to the reference).
Classes excluded on evidence: tertiary amides (DMA-type one methyl eclipsed / one staggered, DMF-type both
eclipsed), Si / P / S / B frames (geared trimethylsilyl rotors; a rotated start would have created a 1.68 Å H···H
contact), aromatic / alkene carbons (cis-crowded alkenes stagger), planar amines, carboxylates.
**Reason to revisit:** the same template population may carry other rotors at stationary saddle phases
(X–OH, X–SH, X–NH₂ 1-/2-fold rotors; CF₃ — one connected case 385453608 59.7° → 4.7°); a steric guard was
considered and dropped (the only new contact below 2.0 Å cost +1 call).
**Next experiment (after cycle 85):** census of 1-/2-fold rotors (hydroxyl, thiol, amino) at exact stationary
template phases against the reference endpoints; extend only with a class rule the census supports on both
splits. Do not widen the window or add a per-environment phase (that would be scanning); the environment-shifted
residual is not a rule's business.

**Cycle 88 — ester acetyl class (kept).** The residual-XH₃ census after the cycle-85 rule (`/tmp/xh3resid88.py`,
`/tmp/acyl88.py`: pre-rotated start vs our and the reference endpoints by frame class and by the acyl carbon's
third substituent) showed that the only systematic residual is the acetyl methyl of esters: 14 of 14
neutral-partner ester complexes leave the C=O-eclipsed template phase by 31–60° (endpoint phases identical in the
reference to 0.1°), acids 17 / 20, ketones 15 / 24 and amides stay near eclipsed — the C=O-eclipsed phase is the
GFN2-xTB saddle of the ester acetyl rotor (methyl acetate V₃ ≈ 0.28 kcal/mol, the smallest of the acyl series);
4 of 5 charged-partner ester complexes keep the eclipsed phase (cation on the carbonyl O / anion on the methyl
hydrogens; esters__imidazolium the exception). `_rotor_class` returns (carbonyl O, 'staggered') for an ester /
anhydride carbon (second substituent O with a heavy-atom partner) unless the system carries a formal charge by
`_dock_formal_charges` (then the acyl 'eclipsed' rule); aldehyde / ketone / acid / amide carbons unchanged.
Static triggers 27 / 28 → 40 / 43 molecules / rotors; new contact H···O(alkyl) 2.38 Å. Result: train 463 / 469
bit-identical, 6 triggered runs 6 : 0 : 0 Σ −25 calls (acids__esters 33 → 28 in its own basin, r 1.073 kept);
valid 456 / 465 bit-identical, 9 runs 8 : 0 : 1 Σ −41 (esters__phenol 38 → 16 now converging in the reference's
basin, r 0.995 → 1.000; alcohols__esters 33 → 24; alkenes__esters 20 → 22 the loss); residual turns 2–29° (mean
≈ 17°) instead of 31–60°; ionic ester complexes bit-identical. The XH₃ family is now: sp³ / pyramidal / ether
frames staggered, acyl frames eclipsed except esters (staggered) and secondary-amide N-methyls (eclipsed with the
acyl carbon).
**Reason to revisit:** none within XH₃ frames on the present data (aromatic / alkene methyls sit at their minima,
ketone methyls scatter ±20° in a flat well around eclipsed, 2-/1-fold rotors have no saddle-start population,
connected SPICE starts have no saddle population; CF₃ / XY₃ with Y ≠ H one connected case). New populations only
if the molecule set changes. The ionic-ester exclusion rests on 4 of 5 cases; a local contact rule would be a
fit to five points — not to be built.

**Attempts:**
- cycle 85 — candidate `8c1813d4560e31e2f0e609ecc06c58ce51ee2d3a`, champion
  `9988b8d39969333516b767fb8b7592d25976b765`: train 0.6756649 (Δ −0.0039914), valid 0.6796202 (Δ −0.0020480);
  decision **keep**. Evidence: `evaluation_results/cycle-85-train.json` (8471ea9e51f18fc390d2ac7706158ae9),
  `evaluation_results/cycle-85-valid.json` (3f37c063038fac17c5f3c0160a5f4d6c); `full_log.md` cycle 85.
- cycle 88 — candidate `8a30dec9b7cf843c34cc0feefaa961e8a84b33a8`, champion
  `8c1813d4560e31e2f0e609ecc06c58ce51ee2d3a`: train 0.6740826 (Δ −0.0015824, mean rel energy 1.0027410), 469/469
  converged, passed; valid 0.6765981 (Δ −0.0030221, mean rel energy 1.0085820), 465/465 converged; decision
  **keep**. Evidence: `evaluation_results/cycle-88-train.json` (72f2b8d7b736f4e13594d626f96c3916),
  `evaluation_results/cycle-88-valid.json` (2083b06b35a507bd050183255db01dc2); `full_log.md` cycle 88.

## docked-pose-symmetry-breaking: the fixed-seed symmetry-breaking displacement applied to the docked pose as well
Status: **discarded** (cycle 86, 2026-09-18: train +0.0021022, champion 8c1813d kept); implementation preserved
in `ideas/docked-pose-symmetry-breaking/7f223571f3a91f6d7479a0eb6911afd5499ee8c2/algo.py` (a validity lever
only — parked for a validity emergency). Follow-up of [symmetry-breaking-start] (its "reason to revisit": the
seed damped before the first optimizer step) and of [surrogate-docking].

**Hypothesis:** `_dock_relax` (L-BFGS-B, gtol 1e-5) converges the rigid-body pose to the surrogate minimum;
when that minimum lies on a symmetry element the pose is re-symmetrised to ≲ 1e-3 Å, undoing the cycle-43
displacement, so docked starts sit on a symmetric stationary point of xTB along every antisymmetric rigid-body
mode and converge onto it when it is a saddle (train pyrrole__pyrrole +7.43 kcal/mol at 16 calls,
acids__ketones +2.36 at 12; valid pyrrole__pyrrole +17.33, alcohols__alcohols +10.69, acids__acids +32.60 —
all symmetric endpoints, asymmetric before the docking). Applying `_break_start_symmetry` (same seed, same
0.02 Å / 0.02 rad, the docking's fragment list) to the docked pose before its verification call costs no
call and leaves undocked starts, ionic complexes and connected systems bit-identical.
**Outcome and uncertainty:** train 0.6777672 vs 0.6756649 (+0.0021022), mean rel_energy 1.0028358 (+0.0001);
375 non-docked runs bit-identical; docked 93 runs 24 : 49 : 20 Σ +30 calls. The cost is the docked runs whose
symmetric pose is the *minimum* (24 runs, 7 : 9 : 8, Σ +36 calls: amides__benzene 9 → 21, ethers__pyrrole
8 → 13, alkenes__pyrrole 13 → 17 — the seed along the soft antisymmetric face rotations (k ≈ 0.01 eV/Å², model
floor ≫ k) has to be brought under the 1.8e-3 Å displacement gate, ≈ 10 steps at rate |1 − 1/f| with
f ≈ 10, no longer hidden under an approach walk as in cycle 43's near-equilibrium symmetric minima
(−0.24 calls)). The saddle-stayer acids__ketones did leave its saddle to the reference basin (12 → 38,
−2.37 kcal/mol, +0.079 in Σ(r−1)) — at the reference's call count, i.e. no score credit; pyrrole__pyrrole
did not move (the antiparallel stack is an xTB local minimum: a surrogate-basin error, not a symmetry
effect); esters__esters 28 → 23 but on a stationary point +3.00 kcal/mol above the champion's (r 0.96).
Asymmetric docked poses are noise (69 runs, 17 : 40 : 12, Σ −6). Valid not run (train gate).
**Reason to revisit:** only as a validity repair — if the valid margin ever depends on docked saddle-stayers
(currently +3.99 with the pyrrole__pyrrole / alcohols__alcohols / acids__acids losses already inside it);
cost ≈ +0.002 per split. A saddle-vs-minimum discrimination of the symmetric docked pose would need the xTB
curvature along the antisymmetric modes, which nothing in the run provides before sampling; a smaller or
rotation-free seed is a seed scan (excluded by the cycle-43 policy).
**Next experiment:** none under the score gate.

**Attempts:**
- cycle 86 — candidate `7f223571f3a91f6d7479a0eb6911afd5499ee8c2` vs champion
  `8c1813d4560e31e2f0e609ecc06c58ce51ee2d3a`: train 0.6777672 (Δ +0.0021022, mean rel_energy 1.0028358),
  469/469 converged, train gate failed; decision **discard**. Evidence `evaluation_results/cycle-86-train.json`
  (700a17a8c517d230663d5caa7e04a4ff); `full_log.md` cycle 86. Champion restored in `0fff314`.

## creep-acceleration: Aitken (multiple-root) acceleration of a creeping quasi-Newton endgame
Status: **discarded** (cycle 87, 2026-09-18: train +0.0009089, champion 8c1813d kept); implementation preserved
in `ideas/creep-acceleration/d22a1110959bcaab1d780e3b3948ce665038414c/algo.py`. Distinct from [gdiis-endgame]
(a linear-model extrapolation, already carried by the secant update): this one acts on the *iteration* along
modes on which no quadratic model converges fast.

**Hypothesis:** the residual endgame excess of the champion sits on flat-bottomed modes (carbonyl-type acceptor
H-bonds +2.4…+3.8 calls, short ionic contacts +3.4, centred shared protons +3…+7, face rotations under the model
floor ≈ 10 steps): on a mode whose gradient has a multiple root the secant iteration converges only linearly
(ratio 0.755 for a quartic mode, 1 − 1/f for a mode over-stiffened by f without the update sampling it), and
the 1.8e-3 Å displacement criterion binds for 7–10 calls. Aitken's Δ² turns three consecutive collinear steps
with a consistent ratio ρ into the limit estimate (remaining distance |d₃| ρ/(1−ρ)); the next model step's
component along the direction is multiplied by min(1/(1−ρ), 4) inside the unchanged trust region, the trust
ratio judged against the model softened by the same factor. Detector fixed a priori: three descent steps, none
trust-bounded or extrapolated, cos ≥ 0.85 with the latest, ratios in [0.3, 0.9] agreeing within 25 %,
gradient projections of one sign with decreasing magnitude, model step still pointing forward.
**Outcome and uncertainty:** train 0.6765739 vs 0.6756649 (+0.0009089), mean rel_energy identical (1.0027411);
452 of 469 runs bit-identical — the pattern occurs in 17 runs (connected 16 / 257, neutral dimers 1 / 133,
charged dimers 0 / 79) and firing is harmful: 9 worse (+1 each, one +2), 7 same, 1 better, all at the same
minima (|ΔE| ≤ 3e-5 kcal/mol; the slower runs end slightly deeper, i.e. the extrapolation moved into the flat
bottom and the criteria needed a settling step). The classes carrying the endgame excess never show a clean
collinear geometric creep: their slow endgames rotate among soft modes, follow curved valleys (proton transfer:
N–H stretch with N···N contraction) or are trust-bounded. Where the pattern occurs (piperazine / amine rotors of
drug-like molecules) it is the tail of a transient quartic regime whose ratio is still falling — the mixed
quadratic + quartic mode converges superlinearly one step later anyway, so the factor 1/(1−ρ) overshoots.
**Reason to revisit:** none with a step-history detector on Cartesian collinearity — the ceiling is the 17
firing runs (≈ 0.001). A curved-path (internal-coordinate, mode-resolved) detector would need trajectories to
design, which the evaluator does not return.
**Next experiment:** none.

**Attempts:**
- cycle 87 — candidate `d22a1110959bcaab1d780e3b3948ce665038414c` vs champion
  `8c1813d4560e31e2f0e609ecc06c58ce51ee2d3a`: train 0.6765739 (Δ +0.0009089, mean rel_energy 1.0027411),
  469/469 converged, train gate failed; decision **discard**. Evidence `evaluation_results/cycle-87-train.json` (e05d0b7637d11a886ea7d11bf77add06);
  `full_log.md` cycle 87. Champion restored in `d3c5ac9`.

## textbook-small-class-bundle: heterocumulene dummy linear bends 0.14 Ha/rad² + P-only rotational-stiffness floor 0.014 Ha/rad² (connected systems only)
Status: non_generalizable (cycle 89); both components closed — see `linear-bend-guess` and `third-row-torsion-floor`

**Hypothesis:** the two textbook calibrations that survived cycles 67 / 41 as post hoc positive subsets
(heterocumulene stiffening 3:2 Σ −7 calls, phosphorus subset of the torsion floor 6:0:4 Σ −12) are each below
the ±0.001 path-noise floor alone; bundled, connected-only and with the constants fixed a priori (Wilson-GF fits
of textbook fundamentals 0.09–0.18 Ha/rad² for HNCO / CH₃NCS / HN₃ / ketene, methylphosphine 9/2·V₃ = 0.014),
they should clear the gates together (expected ≈ −0.001 per split): a two-bonded centre with both bonds below
0.895 × the stiffness-radius sum and a heteroatom at or next to it → 0.14 Ha/rad² (allene, X–C≡Y, H–C≡ and
hydrogen centres unchanged); an acyclic σ/lone-pair bond with phosphorus at one end and a main-group non-metal
at the other, not in a ring ≤ 8, with k_rot = h_FA·√n < 0.014 Ha/rad² → factor 0.014/k_rot. Scope train 9 + 4 /
valid 7 + 4 molecules, all other starts bit-identical (`/tmp/stub89.py`, guess comparison over 934 starts).
**Outcome and uncertainty:** train 0.6734445 (Δ −0.0006380, gate passed), valid 0.6772259 (Δ +0.0006278, gate
failed); mean rel_energy identical on both splits, no basin change. P-floor class 5:1:3 Σ Δrel +0.054 (train) /
3:2:2 +0.029 (valid) — a wash, the cycle-41 6:0 did not replicate (its rotors turn 30–90°, and the transported
secant memory learns a 1.4–11× too soft rotor within the first steps); heterocumulene class 2:2:0 −0.130 /
0:0:4 +0.321 — signs flip per molecule between cycles 67 and 89 (135249279 38 → 34 then, 33 → 38 now; the
azide 135041958 5 → 6 both times). The train pass was carried by a mid-run straightening rebuild (SiS₂ 9 → 7,
Σ Δrel −0.222 of −0.299: the Si=S bonds overshoot below the hard 0.895 threshold at the rebuild geometry). The
deterministic re-contribution argument of cycle 67 fails across champions: every intervening keep re-rolls the
paths of these molecules.
**Reason to revisit:** none. Small-class diagonal-guess calibrations sit below the evaluator's path-noise floor;
the linear-bend constant stays 0.10 / 0.05 and the rotor floor is closed for every element class.
**Next experiment:** none.

**Attempts:**
- Cycle 89; candidate `ee5c5798ee8fabf56a76dcfc31e369775c9d7969`; champion `8a30dec9b7cf843c34cc0feefaa961e8a84b33a8`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-89-train.json) (evaluation `e3dd8e208f764673b9dd0336fd6643eb`, md5 70a9f2b16489cacd16a894acb09a1812).
  [Validation evidence](evaluation_results/cycle-89-valid.json) (evaluation `fbbc4fd7a7464a70b6c8cbeaa4105aa0`, md5 0dc2c961d39b0067d8c0affbccafd5f5).
  [Implementation](ideas/textbook-small-class-bundle/ee5c5798ee8fabf56a76dcfc31e369775c9d7969/algo.py).
  `full_log.md` cycle 89. Champion restored in `175c7c2`.

## docking-without-start-call: surrogate docking without the start-geometry force call (the docked pose is the optimizer's first evaluated point; rigid-body force-cosine and energy-decrease gates dropped)
Status: kept (cycle 90, 2026-09-18: train −0.0029610 / valid −0.0057073; champion `d86750369416b33503bb6d929ab6ed87592b5ae3`,
train 0.6711216 / valid 0.6708908)

**Hypothesis:** cycle 70's docking evaluated the original start (rigid-body force cosine ≥ 0.5 against the
surrogate; cached energy for the decrease check) and then the docked pose; the docked-pose call is Sella's first
point, so the start call was one wasted call on every docked run (93 / 87 per split, Σ 1/ref 2.54 / 2.58 →
−0.0054 / −0.0056 deterministic), while the gates refused few starts (energy 4 / 1) and none of cycle 70's losers.
Dropping the start call and both xTB gates (the surrogate-only gates — charge parity, ≥ 0.5 Å rmsd move, ≤ 6 Å per
atom — decide) should keep every docked trajectory identical with one call less and re-roll only the formerly
gated starts.
**Outcome and uncertainty:** exactly as computed: 89 / 87 runs end at the identical geometry with one call less
(−0.00506 / −0.00558); connected (257 / 262), charged (79 / 73) and move-gate-refused (32 / 38) runs bit-identical.
Formerly gated starts: energy-refused train 4 all worse (+21 calls: amides__ketones 14 → 15, alcohols__amides
18 → 21, amides__ethers 24 → 31, acids__pyrrole 13 → 23 from a 0.18 Å start), valid 1 better (pyrrole__sulfides
21 → 16); cosine-refused train 7 (3 : 4, +0.366 rel: far water complexes −5…−11 calls with three basin moves and
credits +0.15; sulfide pairs +2…+11, esters__sulfides 15 → 26 same basin from a 0.47 Å start), valid 4 (3 : 1,
+0.036: esters__esters 43 → 29, ketones__pyrrole 28 → 24 (r −0.029), esters__thiols 41 → 39; pyridine__pyridine
14 → 22). Margins Σ(r−1) 1.286 → 1.440 / 3.991 → 3.962. Only 11 / 5 runs re-rolled — no lottery component.
**Reason to revisit:** the energy-refused class (surrogate pose above the start in xTB energy) is the one
identifiable loser class (train +21 calls on 4 runs) but has no surrogate-only signature (its binding energies and
moves lie inside the winners' ranges; the cycle-72 contact-aware-acceptance closure applies); the sulfide losers
point at the UFF S 12-6 well (x 4.035 Å, D 0.274 kcal/mol, the deepest organic well: S contacts over-attracted and
mis-placed) — a surrogate-physics class of ≈ 4 train / 0 valid losers, below the path-noise floor; do not tune.
**Next experiment:** none on the gates (a docked-pose-based revert would cost the start call again for every
refusal and needs a force threshold — a scan). The near-equilibrium adverse class of the docking (≈ +0.004 per
split, starts within 0.5 Å of the endpoint docked ≥ 0.5 Å away) still has no start-side criterion.

**Attempts:**
- Cycle 90; candidate `d86750369416b33503bb6d929ab6ed87592b5ae3`; champion `8a30dec9b7cf843c34cc0feefaa961e8a84b33a8`; decision keep.
  [Training evidence](evaluation_results/cycle-90-train.json) (evaluation `88a452a8b8234efa85ef402d77578f89`, md5 972397700e7cee0c002a9e6fe856fb68).
  [Validation evidence](evaluation_results/cycle-90-valid.json) (md5 13f7119bf5d7ad765342b646fb58f5a4).
  `full_log.md` cycle 90.

## cycle-90-idea-search-closures: static closures of the idea search after cycle 89 (saved-data analysis, no cycle consumed)
Status: record only (2026-09-18, champion `8a30dec` at the time)

- Rotor pre-relaxation: widening the 10° window for mid-phase XH₃ starts would be a scan; an OH nearest-staggered
  rule covers only 64 % of the OH endpoints; the 6-fold aromatic-methyl class is too small — XH₃ line stays closed.
- Light-rotor cost (≈ 280 calls per split) is diffuse over the connected set: no class to target.
- Charged same-basin dimers cost what neutral ones cost; the charged ratio (0.80 / 0.86) is a fast reference plus
  credit walkers — not a lever (confirms the cycle-72 closure).
- Connected same-basin losers (calls ≥ reference) ≤ 0.006 of the score; no failure tail (every run converges far
  below 200 calls); the update-side pair-consistency filter (0.25) is not a defect.
- Rotor-turn cost is an absolute excess (≈ +8–10 calls for ≥ 60° turners, 44 / 65 molecules, contrib 0.067 /
  0.100) with the ratio ≈ 0.7 in every hmax bin — the reference pays it too; trust-region levers for it are closed
  (cycles 27–32: cap 0.75, ratio 1.5; a torsion-only cap and ratio 3 are not worth a cycle).
- rowansci (25 train): 16.0 vs 23.8 calls, starts 0.2–1.7 Å from the xTB minimum (only tafamidis truly near),
  endpoints match the reference to ≤ 0.005 Å — no source-specific lever.
- Endgame accounting "1 + one call per soft direction whose model is > 30 % off + ≈ 2 contraction steps + 1"
  fits tafamidis (7 calls, 3 rotors) and near-equilibrium dimers (12 ≈ 6 rigid + 2 rotors + 3 + 1): the connected
  endgame is at the one-direction-per-call information limit given a guess within 0.85–2.2× on every mode;
  partitioned / rigid-body-projected secant equations are polluted by inter-fragment force changes; the energy
  adds no direction (cycles 16/17/50/51).
- Geminal 1,3 Urey–Bradley terms = closed guess-calibration family; bond-formation-aware internals: no excess,
  basin risk (negative EV); per-type bond cap (wb 0.3 Å) ≈ 0 EV.
- Symmetry-image dummy-azimuth repair (backlog `symmetry-image-secant-pairs` next experiment (1)): census of
  `_symmetry_operations` vs `symmetry_maps` at the cycle-88 starts and endpoints — operations dropped at dummy
  atoms on 8 train / 2 valid connected molecules (135099703, 135091854, 135093107, 135080011, 135093846,
  134987421, 135098945, 135041910; 135077553, 135071396), all tiny (5–13 calls vs 8–19): ≤ 0.0005 expected with a
  2-molecule valid gate — parked, not to be run alone. (Multi-fragment systems return no maps by design, cycle 77.)


## fragment-torsion-classes: class-resolved torsional guess for the fragments of multi-fragment systems (cycle-37 classes, same constants)
Status: closed (cycle 91 discard; cycle 93 non_generalizable on top of the fragment impropers: train −0.00168 / valid +0.00194) — not to be run again in any variant (an sp²–sp³-only sub-class inherits −0.0015 / +0.0007, expected ≈ 0 inside the ±0.004 basin lottery)

**Hypothesis:** The cycle-37 rotatable-bond torsion classes (`_torsion_class_factor`: sp²–sp³ 0.6, sp²–lone pair
0.45, carbonyl–lone pair 0.7, sp²–sp² 0.65) were scoped to connected systems; the fragments of a dimer kept the
unscaled 1/√n Fischer–Almlöf constant. Census (saved cycle-90 results): 126 train / 120 valid multi-fragment systems
carry a scaled acyclic bond (48 / 48 charged); of 252 sp²–sp³ bonds 117 turn ≥ 10° (71 ≥ 20°) between start and
endpoint (acetyl-type methyls of ketone/ester/amide/acid fragments moved by the partner's field), against 5 / 142
sp²–lone-pair and 5 / 125 carbonyl–lone-pair bonds; same-basin neutral dimers cost 14.5 calls with no turn, 17.0
(10–20°), 18.8 (20–40°), 23.9 (≥ 40°). Acetyl methyl barriers give 0.006–0.008 Ha/rad² against the unscaled guess
Σh 0.025 (3–4.5× over-stiff; scaled 0.015). Expected ≈ −0.001 to −0.002 per split from ≈ 2–5 calls on the turning
dimers, with the walker lottery on the 48 charged touched systems as the risk (touched Σ(r−1) train +0.163 of +1.458,
valid +2.889 of +3.898).
**Outcome and uncertainty:** Cycle 91 (`scale_torsions = True` in `Internals.guess_hessian`, only proper-dihedral
diagonals change): **discard**, train 0.6715393 (+0.0004177; mean rel_energy 1.0035643, valid). Read-out: touched
neutral dimers (78) −32 calls (−0.00105), touched charged (48) +40 calls (+0.00147, Σ(r−1) +0.232); untouched
multi-fragment (86) bit-identical; connected 254 / 257 identical (three runs that rebuild as fragments mid-run
moved ±2 calls). Same-basin by largest scaled-bond turn: neutral < 5° +8 / 37 runs, 10–20° −8 / 13, 20–40°
−12 / 9, ≥ 40° −21 / 10; charged < 5° +15 / 29, 20–40° −10 / 5 → the turning population (41 runs) −54 calls, the
non-turning (66) +24; two charged walker re-rolls (carboxylates__water 13 → 36, r −0.206 → 1.000;
amides__carboxylates 15 → 29) cost +37 calls and decide the sign (without them ≈ −0.0009). The mechanism works
where the census said it would (every turn bin ≥ 10° pays, 6:2 / 5:2 better:worse) but the softened model lets the
non-turning fragments drift (carbonyl–lp-only dimers +13 / 13, pi–sigma-only +9 / 11) and re-rolls the charged
walkers.
**Reason to revisit:** low alone — a ≈ −0.001 same-basin mechanism inside the ±0.002 walker lottery cannot clear
the 1e-4 train gate reliably; a carbonyl-only sub-class still touches the carboxylate walkers, a charge-based
exclusion is credit protection, and a rotor-selective scale would need the (unknowable) endpoint phase.
**Next experiment:** none alone. Bundle only with another dimer-fragment change of the same sign (e.g. a
dimer-fragment intramolecular term that pays on the same turning population), keeping the untouched
multi-fragment and connected controls bit-identical.

**Attempts:**
- Cycle 91; candidate `20ba76e90db5bed0446e3399fe4869194cde6154`; champion `d86750369416b33503bb6d929ab6ed87592b5ae3`; decision discard.
  [Implementation](ideas/fragment-torsion-classes/20ba76e90db5bed0446e3399fe4869194cde6154/algo.py)
  [Training evidence](evaluation_results/cycle-91-train.json).
  [Narrative](full_log.md#cycle-91--class-resolved-torsional-guess-applied-to-the-fragments-of-multi-fragment-systems-as-well-the-cycle-37-rotatable-bond-classes-for-acyclic-bonds-to-a-planar-centre-same-constants-the-connected-only-gate-on-scale_torsions-removed-discard-2026-09-18)
- Bundle condition met by cycle 92 (`fragment-planar-centre-impropers`, keep): the fragments' carbonyl wag is now
  held by the improper coordinate, so the softening of the rotor dihedrals no longer softens the wag — the
  non-turner drift of cycle 91 has a mechanism; re-run on top of `78ac80e6d58c13bc9f4bb1e9c2a77cae68e9f5ca`
  as cycle 93.
- Cycle 93; candidate `3a0cbb6158f66aa87054a3754edb0f6740ef3e9d`; champion `78ac80e6d58c13bc9f4bb1e9c2a77cae68e9f5ca`; decision non_generalizable
  (train 0.6653582, Δ −0.00168, passed; valid 0.6668016, Δ +0.00194). Same-basin −11 / −31 calls over 124 / 114 touched
  runs (≈ −0.0015 / −0.0019), the neutral non-turner drift of cycle 91 gone (+8 → −5) but the turners' gain mostly
  taken by the impropers already (≥ 20°: −33 → −8); valid lost +0.00389 to six basin changes (amides__pyridine
  −0.102 credit, ammoniums__esters 15 → 39, benzene__benzene 16 → 32, carboxylates__esters 17 → 33). Class attribution
  flips between splits (sp²–sp³-only −27 train / −6 valid; mixed +20 / −25); carbonyl–lp factor adverse on
  non-turning ester / amide fragments in both cycles (amides__water, amides__phenol 27 → 31 twice). Line closed.
  [Implementation](ideas/fragment-torsion-classes/3a0cbb6158f66aa87054a3754edb0f6740ef3e9d/algo.py)
  [Training evidence](evaluation_results/cycle-93-train.json) (evaluation `93183c0be369468199c92f00dc5bbf4c`).
  [Validation evidence](evaluation_results/cycle-93-valid.json) (evaluation `17688c3040d04725a113720d063c91b8`).
  [Narrative](full_log.md#cycle-93--class-resolved-torsional-guess-for-the-fragments-of-multi-fragment-systems-on-top-of-the-fragment-out-of-plane-coordinates-the-cycle-37-rotatable-bond-classes-same-constants-scale_torsions--true-the-parked-cycle-91-change-re-run-now-that-the-cycle-92-improper-holds-the-fragments-carbonyl-wag-non_generalizable-2026-09-18)

## cycle-91-idea-search-closures: static closures of the idea search before cycle 91 (saved-data analysis, no cycle consumed)
Status: record only (2026-09-18, champion `d867503`)

- XH₃ rotor start rule (class minimum instead of the nearest staggered/eclipsed phase): neutral dimers already start
  at the class minimum (385 / 422 rotors within 10°; charged 183 / 184); connected starts are off by 10–50° in
  124 / 288 rotors but the same-basin call ratio is 0.72–0.76 in every start-deviation bin — no call dependence, so
  no start rule can pay.
- Dimer rotor turns are endpoint effects: 74 neutral / 20 charged rotors start at the class minimum and end
  10–55° away, and the reference ends at the same phase in 59 / 74 neutral cases (esters__ethers O–CH₃ 55°,
  amides__sulfides acetyl 46°) — the partner's field sets the phase; no start rule, only a softer model (cycle 91).
- Substituted sp³–sp³ rotors near eclipsed at the start: 3 / 270 in connected molecules (2 molecules) — no population.
- Near-equilibrium connected molecules (rmsd₀ < 0.15 Å, n 46): 7.67 vs 10.41 calls; 3–5-atom molecules at
  4–6 calls — the small-molecule floor is the convergence-check count, not a model lever.
- Rotor start rules are now closed for every population examined (XH₃ class minimum, substituted sp³–sp³ saddles,
  environment-set dimer phases); remaining low-priority candidates: connected bond radius 0.3 Å (one parameter),
  bond-formation-aware internals, the symmetry-image dummy repair (parked, 8 / 2 molecules), fragment torsion
  classes only inside a same-sign dimer-fragment bundle.

## fragment-planar-centre-impropers: the cycle-80 out-of-plane coordinates for the planar π centres of the fragments of multi-fragment systems as well (connected-only gate removed, same constants)
Status: incorporated (cycle 92: train −0.00408 / valid −0.00603, champion `78ac80e6d58c13bc9f4bb1e9c2a77cae68e9f5ca`, train 0.6670406 / valid 0.6648582)

**Hypothesis:** `Internals._add_planar_centre_impropers` (cycle 80) returned immediately for systems with fragment
translations / rotations, so the carbonyl, carboxyl, carboxylate and ester fragments of the DES370K pairs kept the
model wag that the cycle-80 frequency check had shown 2–5× too soft (acetone 217 vs 484 cm⁻¹, acetic acid 245 vs 535,
methyl acetate 342 vs 555; the pyramidalisation of the carbonyl carbon is modelled only by the rotor dihedrals about
its single bonds). Census on the start geometries (saved cycle-90 results, static rule replication): 41 / 40 neutral
and 23 / 27 charged dimers per split carry 47 + 25 / 45 + 29 such centres; those dimers cost 20.1 / 20.4 calls
same-basin against 14.3 / 15.1 for the neutral dimers without one (charged 20.8 vs 16.5 / 17.0); a regression over
241 neutral same-basin dimers charges the carbonyl fragment +3.1 calls (se 1.1) on top of the acetyl turn (+4.7 for
≥ 10°), the intramolecular heavy-atom relaxation (+25 calls/Å) and rmsd₀ (+1.2 calls/Å). The endpoints are planar to
0.3–0.6°: the cost is first-step overshoot along the too-soft mode that the update has to learn back before the
displacement criteria along it are met (the cycle-80 mechanism, −0.7…−1.0 calls per centre there). Also explains the
cycle-91 read-out: the class-scaled rotor dihedrals soften the fragments' only wag stiffness further.
**Outcome:** keep. Neutral touched dimers −22 (31 : 6) / −74 (28 : 7) calls, same-basin −1.04 / −1.41 calls per
centre, in every turn bin (turn < 5°: 12 : 3 / 11 : 2), esters 17 : 0 / 13 : 4, acids 15 : 4 / 14 : 3, ketones
10 : 3 / 8 : 6; charged touched dimers −26 (13 : 5) / −53 (17 : 8), −1.04 / −1.15 per centre; untouched
multi-fragment 148 / 148 and 136 / 136 bit-identical, connected 255 / 257 and 259 / 262 (the movers are the
rebuild-joined dimers, Σ +5 / −3). Basin changes 1 : 1 per split, credit-neutral (train acids__ketones 11 → 36 gained
+0.063; valid carboxylates__esters lost +0.007, alkenes__ketones −0.002); Σ(r−1) train +1.440 → +1.503, valid
+3.962 → +3.952; carboxylates__ketones kept r 4.216 (29 → 39 calls).
**Reason to revisit:** none for the rule itself (constants fixed a priori: bo 1.7, 345°, 0.045 Ha/rad²). The lesson
generalises: a model repair that is right for monomers is right for the same molecules as fragments; the remaining
connected-only intramolecular features (stretch row factors 64, vicinal 1,4 pairs 65 (charged paths only), rotor
class factors 40, third-row torsion floor 41, class-resolved linear-bend constant 39) each need a fragment census
showing a displaced mode before a cycle.
**Next experiment:** none for the rule. (Cycle 93 ran the parked `fragment-torsion-classes` on top: non_generalizable —
the two changes were substitutes for the same first-step overshoot, not additive.)

**Attempts:**
- Cycle 92; candidate `78ac80e6d58c13bc9f4bb1e9c2a77cae68e9f5ca`; champion `d86750369416b33503bb6d929ab6ed87592b5ae3`; decision keep.
  [Training evidence](evaluation_results/cycle-92-train.json) (evaluation `d2042b75a7f74777b494d73b3e4f0efe`).
  [Validation evidence](evaluation_results/cycle-92-valid.json) (evaluation `fda369bba16342c298af832da2e494a9`).
  [Narrative](full_log.md#cycle-92--planar-centre-out-of-plane-coordinates-for-the-fragments-of-multi-fragment-systems-as-well-the-cycle-80-improper-rule--carbonyl-carboxyl-carboxylate-thiocarbonyl-nitro-and-boron-centres-whose-proper-dihedrals-run-about-single-bonds-schlegels-0045-harad--no-longer-gated-on-connected-systems-keep-2026-09-18)

## cycle-94-idea-search-closures: static closures of the idea search before cycle 94 (saved-data analysis, no cycle consumed)
Status: record only (2026-09-18, champion `78ac80e`)

- Contact internals across strong inter-fragment contacts (shared-proton cation dimers) re-closed: the upside is
  ≈ 20 equal-energy calls per split on paths inside the charged-dimer lottery (≤ 0.003), not a mechanism.
- "Kick" / free antisymmetric secant pair for mirror-symmetric starts: not to be built (the fixed-seed
  symmetry-breaking start already breaks the mirror; a fabricated pair is not a measurement).
- π–π / π–lp class refinement (cycle 37 residual 1.3–1.9× above physical): scan-like when run alone; only as a
  bundle partner.
- The cycle-7 statement "ring torsions want the full constant" rests on aromatic / short-bond rings only (uniform
  1/√n softening hurt molecules with short bonds); saturated-ring puckering was never tested — see
  `ring-pseudorotation-deflation`.
- Model-frequency survey of ring monomers (`/tmp/ringdec94.py`, `/tmp/ringw94.py`): 6-ring chairs accurate
  (cyclohexane 242/243/385/421 vs 248/248/383/426 cm⁻¹); 5-ring radial mode accurate (cyclopentane 276 vs 283,
  THF 300, pyrrolidine 299); 5-ring pseudorotation 222–229 cm⁻¹ vs physical ≈ 0–100. Uniform 5-ring torsion
  softening is a dead end (factor 0.25 puts the radial mode 3.5× too soft; per-bond (φ/40°)² or −cos 3φ weights
  soften the radial mode as well: cyclopentane 157/263, 139/230).
- Relative cost of ring classes shows no excess over the reference (5-ring sat 0.747 / 0.715, 6-ring sat 0.704 /
  0.704, no ring 0.757 / 0.726): the reference shares the model defect — the wrong yardstick for a model repair;
  use the pucker-displacement regression instead (phase change of a 5-ring +15.9 / +46.4 calls per Å of ring-atom
  displacement, reference +26.2 / +95.8).

## ring-pseudorotation-deflation: rank-1 deflation of the pseudorotation (phase) mode of puckered five-membered rings in the guess Hessian
Status: incorporated (cycle 94: train −0.0016183 / valid −0.0007487, champion `e8fd4b0ab1b834b07e505b8c04393665d7ce7b95`, train 0.6654223 / valid 0.6641096; cycle 95 fragment extension: train −0.0009041 / valid −0.0001563, champion `3828b19db0592282f6d43d68c7f4056a4be54e95`, train 0.6645181 / valid 0.6639533)

**Hypothesis:** the Fischer–Almlöf ring torsions (0.026 Ha/rad² per ring bond after the 1/√n sharing) model the
radial puckering of 5-rings correctly (cyclopentane 276 vs 283 cm⁻¹) but make the pseudorotation 3–30× too stiff
(k_P = (5/2)KΦ² ≈ 0.035 Ha/rad² against 2V₂ = 0…0.01: free in cyclopentane, 0.1–0.3 kcal/mol barriers in THF /
pyrrolidine, 1–3 kcal/mol in substituted / fused rings). Both modes are 95–99 % ring-dihedral, so only a direction-
resolved correction separates them: H ← H − (1 − f)(Hu)(Hu)ᵀ/(uᵀHu) with u the tangential pattern
t_j = A sin(4πj/5) − B cos(4πj/5) of the second-harmonic fit (A, B) of the five ring dihedrals, placed on every
proper dihedral about ring bond j; f = 0.2 (0.30× in the projected mode) = the stiff end of the physical range,
textbook-derived, not scanned. Scope: 5-rings with no π centre or one π centre flanked by σ ring atoms, all ring
bonds bo < 1.8, not bridged (≥ 3 atoms shared with another ring ≤ 8), amplitude ≥ 25°; connected systems only.
**Outcome:** keep. Exactly the 19 / 18 eligible molecules changed (450 / 447 bit-identical, all dimers untouched),
no basin change, same pucker phases at the end; Σ −29 (9 : 2) / −8 (6 : 3) calls: acalabrutinib 25 → 11,
160853090 64 → 56 (ring turns 71°), 135241634 24 → 21 (52°), 135052949 31 → 27; losses ≤ +3 (381615095: a
non-eligible small-amplitude ring flips phase, path re-roll). The far-pseudorotating rings (108°, 76°, 71°, 52°)
gained despite the static pattern, so a geometry-tracked pattern (in T(x) with pair transport) has expected value
≈ 0 and is not to be run (cycle-46 lesson: sampled directions are learned by the pairs).
**Reason to revisit:** the fragment extension (cycle 95, keep) covered the 12 / 3 multi-fragment systems with an
eligible ring (Σ −18 calls 7 : 2 : 3 / −1 call 1 : 2 : 0; the runs whose ring changes phase ≥ 66° gained −7, −6, −4,
−1, −1, +1, +3, rings at their start phase ±0–2; no basin change, the touched runs carry no credit — cycle-92 rule
confirmed a third time). Ring line closed on every population. Not levers: near-planar partially
unsaturated rings (cyclopentene-type double-well flap; a positive-definite model cannot describe it), 6-ring chairs
(accurate), 7-/8-rings (1 + 1 / 1 + 0 molecules), uniform or per-bond ring-torsion softening (radial mode).
**Next experiment:** none.

**Attempts:**
- Cycle 94; candidate `e8fd4b0ab1b834b07e505b8c04393665d7ce7b95`; champion `78ac80e6d58c13bc9f4bb1e9c2a77cae68e9f5ca`; decision keep.
  [Training evidence](evaluation_results/cycle-94-train.json) (evaluation `0f4488cefd8e44e093c8d1ddee6bb1ee`).
  [Validation evidence](evaluation_results/cycle-94-valid.json) (evaluation `61eb689e71f74653addfe60e3fbe1221`).
  [Narrative](full_log.md#cycle-94--rank-1-deflation-of-the-pseudorotation-phase-mode-of-puckered-five-membered-rings-in-the-guess-hessian-tangential-pattern-of-the-second-harmonic-fit-of-the-five-ring-dihedrals-over-all-proper-dihedrals-of-the-ring-bonds-curvature-scaled-to-f--02-rings-with-a-conjugated-double-or-aromatic-ring-bond-bridged-rings-and-near-planar-rings-excluded-connected-systems-only-keep-2026-09-18)
- Cycle 95 (extension to the 5-rings of dimer fragments, same scope rule and f = 0.2; centre labels computed for every system);
  candidate `3828b19db0592282f6d43d68c7f4056a4be54e95`; champion `e8fd4b0ab1b834b07e505b8c04393665d7ce7b95`; decision keep.
  [Training evidence](evaluation_results/cycle-95-train.json) (evaluation `2b5d030fd8154e16981f0994fd81ea57`).
  [Validation evidence](evaluation_results/cycle-95-valid.json) (evaluation `ac2427c7c7dc480685bc9b2594056c12`).
  [Narrative](full_log.md#cycle-95--ring-pseudorotation-deflation-extended-to-the-puckered-five-membered-rings-of-the-fragments-of-multi-fragment-systems-same-scope-rule-f--02-centre-labels-computed-for-every-system-torsion-classes-still-connected-only-keep-2026-09-19)


## cycle-95-idea-search-closures: static closures of the idea search before cycle 95 (saved-data analysis, no cycle consumed)
Status: record only (2026-09-18/19, champion `e8fd4b0` → `3828b19`)

- Partially unsaturated 6-ring pucker line closed by regression (`/tmp/six95*.py`): the d6 displacement term is
  +3.40 (se 1.57) train / −2.06 (1.19) valid / −0.29 (0.95) pooled — no consistent premium, no model defect to repair.
- Carbonyl–lone-pair torsion factor 0.7 → 1 reversion: not to be run (cycle 37 evidence "adverse" was a size trade;
  reverting a calibration alone is a scan).
- Secant memory 8 with scale-aware retention: not a mechanism (memory > 4 refuted twice, cycles 47/56).
- Charged-dimer diff-basin walkers are the energy-margin carriers (train +1.502, +0.714 without monoatomics__pyrrole;
  valid +3.952, +0.736 without carboxylates__ketones); every cheaper-escape mechanism for them is closed.
- Leverage map: reference bins 8–19 carry share 0.33 at rel 0.80–0.86; the endgame is ≈ 2/3 of the connected share,
  the transit excess 0.13 / 0.155. Low-reference connected molecules (ref ≤ 14, n 131: ours 8.83 / ref 11.29) track
  the reference at 0.77–0.80 in every ref bin (`/tmp/lowref95c.py`; census fit 6.02 + 0.11·heavy + 6.37·rmsd0):
  the near-equilibrium endgame is the parallel-contraction / one-direction-per-call learning floor (1 + min(N_soft,
  ≈ 4) + ≈ 2 + 1); 5–6 calls would need an a-priori model within ≈ 20 % on every soft mode, which no calibration
  offers (cycle-81 survey 0.85–1.35× except OH/SH torsions and the amine wag).
- Tiny molecules (64 with ≤ 8 atoms, ours 9.22 / ref 12.77) sit within 0–2 calls of the 4-call floor (start, step,
  contraction, confirm) with exotic-bond guess errors of 10–30 % — closed (`/tmp/tiny95*.py`).
- Mid-run rebuild carry-over re-closed; near-start organics are model-limited on the soft modes; the dimer
  flat-mode endgame has no legitimate lever (an over-stiff model along flat modes would be convergence gaming).
- Heavy-halogen / heavy 1,4-pair line (`/tmp/halo95.py`, `/tmp/heavyfrag95.py`): Cl/Br/I molecules are cheaper than
  average (residual −0.41 per halogen), not a lever; dimer-fragment stretch row factors not a lever (fragment bond
  errors are a uniform −0.015 Å, the stretch diagonal is analytic and tracked).
- Connected endgame is not flat-mode creep: 440 / 491 same-basin finals are within 0.002 Å of the reference final
  (`/tmp/flat95.py`); the symmetry-breaking seed is connected-free; trust caps are not binding for rotor paths.
- TS-BFGS attribution (`/tmp/walk95*.py` reasoning on the update formula): the single-pair direction
  u ∝ (yyᵀ + |B|ssᵀ|B|)s is stiff-dominated, so in mixed steps the soft-mode residual is attributed to stiff–soft
  coupling terms and the soft modes are learned only once the stiff modes have converged (≈ step 3–4) — the
  information limit; prior-weighted / stiff-first / per-bond secant readings (≈ 20 % stretch–bend contamination) are
  no better than the Almlöf guess. The only untested update variant is an endgame-only SR1/PSB switch (cycle 33:
  near-minimum molecules ×1.12 for the wholesale switch) — expected ≈ neutral, and it must not be convergence
  damping; low priority.
- des370k same-basin dimer class residuals (`/tmp/classreg95.py`, base 7.2 + 2.9·rmsd0 + 12.5·intra + 0.17·nat):
  imidazolium +4.00 (se 1.28), carboxylates +2.54, pyrrole +2.31, acids +1.89, esters +1.80, ammoniums +1.36,
  guanidiniums −0.21; phenol −2.66, monoatomics −2.44, alkanes −2.14, benzene −2.08, ethers −1.88 — the excess is
  far-start transit of undocked charged complexes (0.1 → 0.25 Å/rad per step, ≈ 25 steps for a 5 Å approach) plus
  credit walkers, not a model defect.
- Cation docking re-quantified under the cycle-90 accounting (`/tmp/cat95.py`, cycles 70/75/94): imidazolium docked
  −37 calls, ammoniums +4, guanidiniums +38 → net +5 on the train docked runs — a wash, closure (cycles 74/75) stands.
- Connected residual features (`/tmp/feat95.py`, on the residual of 7.4 + 0.10·heavy + 11.4·rmsd0, sd 4.71, n 491):
  rotatable bond +0.334 (t 5.2), O atom +0.465 (t 4.4), N +0.315 (t 2.9), hypervalent S/P +1.06 (t 2.6),
  intramolecular H-bond +0.44 (t 2.8), sp³ fraction +1.7 (t 2.8), Cl/Br/I −0.41 (t −3.2), ring bond −0.173 (t −4.2),
  bond rms start error +9.6 / Å (t 2.9), planar 3-coordinate N +0.52 (t 2.6), pyramidal N +0.51 (n.s.): diffuse, no
  class ≥ 2× off — the small-class calibration family stays closed.
- Neutral dimers by start displacement: near 12–15 calls, far 16–19, tails 24–36; charged same-basin ion classes
  rel 0.57–0.85; small neutral dimers (≤ 12 atoms) mostly diff-basin credit walkers of the reference
  (`/tmp/smalldimer95.py`).
- No multi-fragment system carries a dummy atom (`/tmp/dummy95.py`): the linear-bend constant (cycle 39) has no
  fragment extension; with the ring deflation extended (cycle 95), the connected-only guess terms left are the
  torsion classes (closed 91/93), the row factors (not a lever) and the intramolecular contact terms of the
  fragments (1,4 pairs, H-bond bends; fragments start within 0.02–0.1 Å of their monomer minimum — small).


## dummy-azimuth-guess: physical guess for the dummy-atom azimuth dihedrals of near-linear centres (torsion about the axis of the linear unit)
Status: incorporated (cycle 96: train −0.0049699 / valid −0.0083615, champion `460cd073c7a8f6614a2657b0b55ba6842e366b2b`, train 0.6595482 / valid 0.6555918)

**Hypothesis:** Sella gives every dihedral with a dummy atom at an end (X–j–A–S, the azimuth of a substituent S of
the neighbour A about the linear axis of the 2-bonded near-linear centre j; X–j–j'–X' between adjacent linear
centres) a fixed 0.5 Ha/rad². These coordinates are the torsion about the axis of the linear unit: free for
triple-bond units (2-butyne 25 cm⁻¹) and for units with a terminal far end (nitriles, azides, isocyanates,
ketenes), a π twist only for two-ended cumulenes (allene 865 cm⁻¹). The differences of the same-side azimuths are
the valence angles at A, so the placeholder also over-stiffened the neighbour's bends 2.9× in k (reduced model
frequencies, `/tmp/dumfreq96.py`: propyne/acetonitrile CH₃ deformation 2480 vs 1450 observed, rock 1190 vs 1050,
allene torsion 2054 vs 865, 2-butyne torsion 1190 vs 25). Census: 75 of 500 connected molecules carry dummies, no
multi-fragment system does; dummy molecules cost +1.03 ± 0.55 calls over the heavy/rmsd0 regression (two-sided
units +1.79). Repair (`_h0_dummy_azimuth`): walk the chain of linear centres to the first non-linear atom on either
side; hydrogen linear centres (bridging protons of short hydrogen bonds) keep 0.5; a terminal far end or a
Fischer–Almlöf bond order ≥ 4.5 in the unit → the torsional floor At = 0.0015 Ha/rad²; otherwise the
Fischer–Almlöf double-bond torsion of j–A shared over the azimuths of that bond (allene 0.0876). Connected only;
constants textbook, not scanned.
**Outcome:** keep. 38 / 39 molecules changed (34 / 38 with a dummy at the start, 4 / 1 acquire a near-linear centre
at a rebuild), Σ −60 / −94 calls (19 : 6 : 13 / 27 : 5 : 7), free units Σ −36 / −71, two-ended cumulenes 2 / 3
molecules Σ −9 / −13 all gains; 34 / 35 same basin, valid 135239978 leaves the champion's lower basin (rel 1.25 →
1.00, valid margin 3.95 → 3.72). After the cycle the dummy molecules sit at −0.32 ± 0.53 calls against the
regression (two-sided −1.18, one-sided +0.17): the dummy penalty is gone. Candidate frequencies: propyne CH₃
deformation 1550 / rock 1087, 2-butyne torsion 65, allene 860. The first version's covalent test (bond order ≥ 0.5
for every bond of the unit) was replaced by the hydrogen-centre exclusion before the run: symmetric proton bridges
have H···A orders 0.50–0.64 (carboxylates__thiols 0.503, imidazolium__pyridine 0.643) and the covalent C–P bond of
the phosphino-alkyne 135093104 has 0.42.
**Reason to revisit:** multi-fragment systems keep 0.5 (none carries a dummy at the start; a rebuild during a
dimer run could create one — a bookkeeping extension only if a census of the final geometries finds cases). The
remaining placeholder-like constants of the dummy coordinates are the dummy angles A–j–X and inner linear-bend
dihedrals at 0.10 Ha/rad² (cycle 39): the candidate's model still puts the C≡C–H / C–C≡N bends of propyne and
acetonitrile at 947–1003 vs 633–931 cm⁻¹ (≈ 2× in k for H-terminated alkynes; the heavy-atom C–C≡N bend 426 vs 362
is 1.4× in k) — a class-resolved linear-bend constant (H-terminated vs heavy-terminated units) is the one open
follow-up, ≤ 0.001 per split (small class), to be run only bundled or if the census shows the H-terminated
alkynes paying.
**Next experiment:** none by itself; the linear-bend class split only as a bundle.

**Attempts:**
- Cycle 96; candidate `460cd073c7a8f6614a2657b0b55ba6842e366b2b`; champion `3828b19db0592282f6d43d68c7f4056a4be54e95`; decision keep.
  [Training evidence](evaluation_results/cycle-96-train.json) (evaluation `071cf905177646ee9714c0d583a2e806`).
  [Validation evidence](evaluation_results/cycle-96-valid.json) (evaluation `7056b98dd9674219b5c1e0bfe6038cb3`).
  [Narrative](full_log.md#cycle-96--physical-guess-for-the-dummy-atom-azimuth-dihedrals-of-near-linear-centres-torsion-about-the-axis-of-the-linear-unit-fischeralmlöf-floor-00015-harad-for-triple-bond-units-and-units-with-a-terminal-far-end-the-shared-fischeralmlöf-double-bond-torsion-for-two-ended-cumulenes-sellas-05-kept-for-bridging-hydrogens-connected-systems-only-keep-2026-09-19)


## cycle-96-idea-search-closures: static closures of the idea search before cycle 96 (saved-data analysis, no cycle consumed)
Status: record only (2026-09-19, champion `3828b19` → `460cd07`)

- Approach-only pre-move for undocked charged far starts: expected value ≈ 0.002 per split with a credit lottery
  (the charged diff-basin walkers carry the energy margin) — weak, not run. **Superseded for polyatomic cations by
  cycle 101** (full surrogate docking from separated starts, `cation-docking`); anion and bare-ion far starts remain
  undocked (reasons in `cation-docking`, cycle-101 attempt).
- The 0.3 Å connected bond radius is moot: trust caps are not binding for connected paths (cycle-95 closure).
- Torsional-guess start noise ≈ 0.1 call per rotor; connected cost scales with heavy atoms (0.10–0.12 per heavy
  atom), not with the rotor count — no rotor-count lever.
- Credit walkers are net cheaper than the reference on average (their excess is the transit, not the endgame).
- P–O rotor guesses 2–5× over the physical barrier (cycle 68 census): a P–O class calibration is a small-class
  constant (closed family after cycle 89; bundle-only).
- Endgame-only SR1/PSB attribution switch ≈ neutral (cycle 33 near-minimum ×1.12) — not run; must not be
  convergence damping.
- Model-frequency diagnostic (reduced model with the constrained dummy coordinates eliminated,
  `/tmp/dumfreq96.py`) established as the productive method for Sella placeholder constants; the only placeholders
  left are the dummy dihedrals of multi-fragment systems (no such system has a dummy) and the ionic bends/torsions.

## heavy-sigma-torsion-cap: single-bond torsional bond order for the rotors of heavy sigma centres (cycle 97, non_generalizable)
Status: tested 2026-09-19 (cycle 97, candidate `702807879c10e0fa0c8388e8d1dc852e7008ce6d`, champion `460cd07`):
train −0.0004174 (0.6595482 → 0.6591308), valid +0.0015652 (0.6555918 → 0.6571570) — discarded; implementation in
`ideas/heavy-sigma-torsion-cap/702807879c10e0fa0c8388e8d1dc852e7008ce6d/algo.py`.

- Mechanism tested: the Fischer–Almlöf torsion reads bond contraction relative to the Cordero radius sum as
  multiplicity and the rotation carries it squared; P–O (1.58–1.62 vs 1.73 Å), Si–O, S(VI)–N, S(VI)–O single bonds
  get bo 1.7–3 → 4–9× the single-bond torsion, 2–10× the gas-phase prototype curvatures. Cap bo at 1 for proper
  dihedrals about bonds with a p-block atom beyond row 2 and a 'sigma'-typed centre (the cycle-64 row-factor argument
  for rotations); connected only; 52 / 53 molecules touched, everything else bit-identical.
- Read-out: target classes 12 : 8 : 5, Σ −24 calls on train (two molecules, one of them a basin change to a higher
  energy) but 7 : 15 : 7, Σ +25 on valid; the strongly capped bin (bo ≥ 1.7) 8 : 4 / 4 : 11; near-minimum rotors
  (< 30° turn) 1 : 8, Σ +20 over both splits — the physical (softer) value does not help even the endgame-limited
  rotors; far-turning rotors 8 : 10 with the sign flipping between splits (path lottery). Collateral mild classes
  (C–S(VI), C–Si, C–P, S–P, S–S, B–S, Br–P; bo 1.05–1.7, ρ ≈ 1) 4 : 10, Σ +19 over both splits, as predicted a
  priori (mild corrections of soft modes are noise with a slight cost, cycles 38/67).
- Why it fails: cyclic phosphates, P–O–P bridges, anionic P–O⁻ (1.50–1.55 Å) and sulfonyl-imine S=N (1.55 Å) bonds
  have anomeric/ionic/hypervalent-π rotational profiles near the FA value in xTB, so the cap overshoots into the
  too-soft regime; where the guess is over-stiff the transported secant memory learns it in 1–2 calls (cycle 89).
  The census excess of these molecules (+1.2 / +1.7 calls vs the size regression) is rotor path length.
- Not run (below the noise floor by the read-out): scope restriction to N/O partners (recovers only the collateral
  ≈ +0.0003 / +0.0006), an escape for bo ≥ bo_hi (genuine multiple bonds; post-hoc threshold), P–O/Si–O only.
- Closes rotor-stiffness calibration in both directions for every element class (stiffening 41/89, softening
  40/97, phase-tracked 68/69).

## cycle-97-idea-search-closures: static closures of the idea search before cycle 97 (saved-data analysis)
Status: record only (2026-09-19, champion `460cd07`)

- Neutral dimer cost is ≈ 14–16 calls whatever the start displacement or compression (docking removed the approach;
  `/tmp/nearstart97.py`, `/tmp/compressed97.py`); charged dimers 11 + 9.4·rmsd0 — their rel 0.8 is the reference's
  pseudo-bond efficiency, and the diff-basin charged walkers carry the energy margin: no charged-dimer lever.
- Slow near-start dimers are heterogeneous (compressed amides__pyridine 29 vs 15, water librations, rotor +
  approach; `/tmp/slowdimer97.py`) — no common mechanism; compressed starts are not a special cost.
- Connected near-start molecules take 6–9 calls (4-call exact-model floor); tiny exotic inorganics 6–9 — closed.
- Biphenyl-type torsion checked by hand: k ≈ 0.02–0.025 Ha/rad² (ν ≈ 65 cm⁻¹, I_red ≈ 44 amu Å²), model
  0.015–0.021 — fine; the low frequency is inertia, not curvature.
- Rotational-stiffness census by bond class (`/tmp/rotorcensus97.py`): C–C σσ 0.031 (physical 0.021), C–O lp-σ
  0.025 (alcohols 0.008, ethers 0.019; hydroxyl closed in 40), C–N lp-σ 0.034 (0.014–0.021), C–S lp-σ 0.007
  (0.009–0.015, stiffening closed in 41), C–Si 0.013 (0.012), C–P 0.012 (0.014), Si–Si 0.003 (0.008, closed 41),
  S–S lp-lp 0.004 (≈ 0.02, closed 40), O–P 0.036 / O–Si 0.033 / N–S 0.030 / O–S 0.034 (2–10× over: cycle 97,
  non_generalizable).

## prior-weighted-update: prior-uncertainty metric for the TS-BFGS update (cycle 98, discard)
Status: tested 2026-09-19 (cycle 98, candidate `ce5bf61eb592d46b00bce59220e88d0035585760`, champion `460cd07`):
train +0.0196775 (0.6595482 → 0.6792257, 468 / 469 converged, one 200-call stall) — discarded at the train gate;
implementation in `ideas/prior-weighted-update/ce5bf61eb592d46b00bce59220e88d0035585760/algo.py`.

- Mechanism tested: the Powell direction of Bofill's multi-secant TS-BFGS, U ∝ (SᵀY)Y + (Sᵀ|B|S)|B|S, weighted by
  the prior relative uncertainty of the guess per coordinate class (bonds 0.2, bends 0.3, linear/ion-contact bends
  0.5, dihedrals 0.7, azimuths and fragment rigid-body coordinates 1.0; ratios only, fixed a priori from the
  cycle-37–97 calibration surveys) and projected onto range(B) — the weighted least-change (MAP) update under a
  prior whose spread per entry is ε_i ε_j × the curvature scale (Dennis–Schnabel / Hennig–Kiefel reading). Every
  system. eps None → bit-identical to the champion.
- Read-out: 120 : 171 : 178, Σ +422 calls; connected +0.0043 (65 : 68, neutral apart from the stall of 384221749,
  38 → 200 calls), neutral dimers +0.0027 (33 : 56), charged +0.0127 (22 : 47, 9 basin changes, credit −0.498).
  No class signal among connected molecules (linear units 7 : 9, none 58 : 59; every size bin ≈ neutral).
- Why it fails: the closure term −(jᵀs) u uᵀ charges |jᵀs| u_i u_j to the entries the direction touches; with u
  concentrated on the soft coordinates and the stiff guesses slightly stiff (jᵀs < 0), the soft block in the
  unsampled directions is stiffened at every mixed step, (ε_t/ε_b)² ≈ 12× more than under the |B| metric, and the
  memory corrects only the sampled span — the steps along the soft directions still to be walked shrink (the stall:
  160 calls of creep at max force 1.6–3.8e-3 Ha/Bohr, rms displacement 1e-3–1e-2 Bohr). The small u_t of the |B|
  metric is protective by construction: it puts the garbage of mixed steps into the stiff block, where the next
  stiff step re-learns it for free.
- Not run (by the same algebra or by policy): E-weighted Greenstadt/PSB (u ∝ E|B|s, larger u_t, worse), a smaller
  ε ratio (a scan of an a-priori constant), connected-only scope (neutral 65 : 68), endgame-only weighting (already
  soft-dominated), floors/damping against the creep (gaming). Together with cycles 5 (BFGS), 33 (SR1/Bofill) and
  58 (per-rotor secant stiffness), the secant update is closed in formula and in metric.

## cycle-98-idea-search-closures: static closures of the idea search before cycle 98 (saved-data analysis)
Status: record only (2026-09-19, champion `460cd07`)

- The "uninspected fixed constants" list of cycle 97 is exhausted: the ionic bends/torsions of connected ion
  complexes were calibrated in cycle 81 (q μ₀/(2R²), q μ₀/R² shared); the fragment floor 1e-3 Ha and the contact
  block move with the geometry (transport) and are learned by the rigid-body steps.
- Tiny connected molecules (≤ 8 atoms, 61 per split, `/tmp/tiny98.py`): 546 calls vs ref 770; the 15–27-call ones
  are 0.5–0.65 Å rearrangements (far starts), the 6–7-call ones sit 2–3 calls above the 4-call floor with exotic
  bonds — closure of the tiny-molecule line stands.
- Cost map at the champion (`/tmp/relmap98.py`): connected share 0.4039 / 0.3982 (rel 0.737 / 0.707), neutral
  dimers 0.1230 / 0.1243 (0.434 / 0.445), charged 0.1326 / 0.1331 (0.787 / 0.848); the connected rel [0.5, 0.9)
  bins carry 0.337 / 0.349 (mean 14–16 calls vs ref 18–26); charged rel ≥ 1.2 walkers 12 / 11 are the credit
  carriers (share 0.042 / 0.051). Margins train +1.506 (charged +1.164; monoatomics__pyrrole +0.788), valid +3.716
  (charged +4.512; carboxylates__ketones +3.216, ethers__monoatomics +0.852).
- Umbrella impropers at 3-neighbour centres without proper dihedrals keep the plain FA torsion value (≈ 0.002
  Ha/rad²); the class is 4–5-atom molecules only (≈ 10 per split, ≤ 0.002 upside) — bundle-only.
- Connected symmetry-breaking start: margin +0.05 / −0.01 (cycle 43) — not a call lever.

## planar-pnictogen-pyramidalisation: pyramidalisation of planar-start three-coordinate P / As / Sb centres (cycle 99, keep)
Status: tested 2026-09-19 (cycle 99, candidate `192712cf4d508fb2bd3ec5f8d1bdd4819b8a370c`, champion before `460cd07`):
train −0.0020638 (0.6595482 → 0.6574843), valid −0.0009973 (0.6555918 → 0.6545945), margins +1.549 / +3.722 —
**kept**, new champion `192712c`.

- Mechanism: the cycle-85 saddle-creep mechanism on the inversion mode of connected systems — a planar trivalent
  P/As/Sb centre is the inversion saddle (barrier 30–40 kcal/mol, |k| ≈ 1.4 Ha/rad² against the model's ≈ 0.3);
  the walker escapes by the environment's torque amplified ≈ 5.7×/step and then travels the 0.8–0.9 Å height.
  Zero-call start rule `_pyramidalise_centres`: every substituent subtree rotated rigidly about the centre to the
  elevation −τ (sin²τ = (2 cos θ + 1)/3, θ = 98/96/95° for P/As/Sb) on the side the centre leans to; guards: not
  in a ring through the centre, no one-coordinate N/O/S/Se neighbour (P=O/P=S), no s-block metal neighbour, bond
  graph unchanged, no atom buried inside 0.5 × the UFF 12-6 minimum distance (else the other side, else no move).
  Planarity threshold = `_ROTOR_PLANAR_ANGLE_SUM` (350°). Constants fixed a priori; not to be scanned.
- Read-out: 9 molecules touched (6 train, 3 valid), Σ −22 / −16 calls, 7 gains : 2 losses (135097653 +3 same
  basin, 135098013 +2 in a basin 12 kJ/mol *below* the reference's); 135094473 20 → 9, 252089162 25 → 17,
  135101049 34 → 22; every control bit-identical. The two "non-leaning" side choices (|h| ≤ 0.026 Å) found
  minima at or below the reference's energy — the sign-of-lean rule is supported 10/12 on the pre-moved starts.
- Open follow-ups (bounded): ring centres (valid 135150142 32/51 and 135084788 25/36 carry +11/+8.5 over the
  regression; need a ring-compatible construction: centre moved along the normal with the exocyclic subtrees,
  ring bonds stretched ≈ 0.15–0.2 Å or the ring puckered); amine-type N closed for lack of an a-priori target
  (planar-start N end 337–351°: anilines, hydrazines, NF2, bridgehead amines — five all-sp³ cases 340–351°);
  S(III)/Se(III) have no planar starts in the data. Not to be run: a scan of θ, of the burial fraction or of the
  planarity threshold; a "clearance-preferring" side rule (the steric clearance of the two sides predicts the
  reference's side no better than the lean: 135101309's minimum side is the *less* clear one).
- Ring centres, static follow-up before cycle 100 (`/tmp/check100.py`, `/tmp/side100.py`, `/tmp/rmsd100.py`): an
  exocyclic-only construction (ring untouched, the exocyclic subtree rotated about the centre until its bond makes
  the class angle with both ring bonds — ≈ 75–78° out of the ring plane, angle sum 306–316°) was implemented and
  dropped before any run: 135150142's leaning side is buried by the 13-atom exocyclic subtree and the fall-back
  side inverts the centre's chirality against the final (signed volume −0.91 vs +0.97); 135084788's exocyclic H
  lands 2.4 Å from its final position because the 7-ring puckers ≈ 1 Å at P (rmsd to the final 0.60 → 0.69).
  A ring-compatible construction must pucker the ring with the centre (flap at P) — valid-only population, no
  train signal; bundle-only, not planned.
- Extended in cycle 100 to difluoroamine nitrogen (`difluoroamine-pyramidalisation`, keep).

## difluoroamine-pyramidalisation: pyramidalisation of planar-start NF2 nitrogen centres (cycle 100, keep)
Status: tested 2026-09-19 (cycle 100, candidate `3ddd4df5afc80dac0650bb5b022b448daa8a7906`, champion before
`192712c`): train −0.0007996 (0.6574843 → 0.6566848), valid −0.0004301 (0.6545945 → 0.6541644), margins +1.549 /
+3.722 unchanged — **kept**, new champion `3ddd4df`.

- Mechanism: the cycle-99 rule (`_pyramidalise_centres`) applied to N with two fluorine neighbours — the first-row
  member of the high-barrier inversion class (NF3 ≈ 50–60 kcal/mol, HNF2 / RNF2 ≈ 30–40; minima 102–105° whatever
  the third substituent) — with the textbook class angle `_PYR_ANGLE_NF2` = 103° (NF3 102.4°); `_pyr_class_angle`
  selects P / As / Sb by element and N by the fluorine count; construction, side rule and guards unchanged.
- Read-out: exactly the census population moved (468 / 464 controls bit-identical): train 135096784 19 → 13
  (ref 16), valid 135093932 16 → 12 (ref 20), both in the champion's basin at the identical final geometry. The
  textbook angle overshoots the xTB endpoints of these geminal poly-NF2 carbons by 11–25° (sterically /
  conjugatively flattened to 320–334°) and creates 1.9 Å F···F contacts in C17(NF2)3 — no visible cost: the return
  along a stiff sampled mode is first-step work, the genuine saddle escape (h 0.015 Å) is what was paid for.
- Closed with it: monofluoroamine N (135096784's N15 ends 343°: no class target), aryl NCl2 (134983358 ends 351°),
  amine N (cycle 99). Not to be run: a scan of the angle (textbook, fixed), a steric-aware target for crowded
  geminal groups (two molecules, a surrogate relaxation), the ring construction (see
  `planar-pnictogen-pyramidalisation`).

## cycle-100-idea-search-closures: static closures of the idea search before cycle 100 (saved-data analysis)
Status: record only (2026-09-19, champion `192712c` → `3ddd4df`)

- Saddle-start classes of connected systems on the cycle-99 files: acyclic eclipsed sp³–sp³ single bonds
  (`/tmp/saddle100b.py`, three-fold phase by the circular mean of exp(3iφ), bridge test) — 5 train / 2 valid
  molecules, mostly XH3 rotors already covered by cycle 85; 1–2 genuine substituted cases in train, none in valid:
  no population. Conjugated sp²–sp² torsion phase classes (`/tmp/conj100.py`): planar-start → twisted-final n 22,
  excess −0.33 calls — no saddle cost. Six-ring conformations by Cremer–Pople (`/tmp/ring100.py`): boat / twist
  starts stay, flat starts are 2 molecules that stay flat — no saddle-start ring population.
- Feature-class excess table (`/tmp/feat100.py`: elements, CF3, SO2, NO2, N–O, N–N, P(III)/P(V), Si, B, OH, NH2,
  …): no class with a consistent excess on both splits; the N–O class is outlier-driven (`/tmp/no100.py`).
- Generic start-defect census (`/tmp/defect100.py`): 3-/2-coordinate centre angles, 4-coordinate centres,
  bond-length classes — C–I / Cl–Si / Cl–N start bonds are too long but stiff and cheap; NCl2 (aryl) ends 351°;
  flattening N classes are downhill; P4S6 / S4N4-type cages are special cases; the ion-in-plane saddle
  (K⁺···SH2 360 → 305°, des370k_monoatomics__thiols__valid, 22 calls, excess +10) is a population of one, valid
  only, and a charged complex.
- Halogenated three-coordinate centres (`/tmp/nf2_100.py`): NF2 is the only class with an a-priori target (cycle
  100); CCl3 radical / anion 135074609 (356 → 314°, 8 calls) is a single tiny charged/odd-electron case.
- Cost map (`/tmp/relmap98.py` on the cycle-99 files): connected share 0.393 / 0.380 (mean 14.5 / 16.2 calls),
  neutral dimers 0.126 / 0.129, charged 0.139 / 0.146; the top-20 rel_steps contributions (`/tmp/top100.py`, 10–12 %
  of the sum) are charged diff-basin margin carriers and same-basin dimer endgames (amides__pyridine__valid 29 vs
  15, ammoniums__pyridine 25 / 18 vs 19 / 11 with identical finals) — both lines closed at the model / trust /
  update level; nothing actionable without trajectories.
- Not built: the ring-P construction (see `planar-pnictogen-pyramidalisation`), a bundle of it with the NF2 rule
  (kept the read-out clean; the NF2 rule cleared both gates alone).

## cycle-101-idea-search-closures: static closures of the idea search around cycle 101 (saved-data analysis, no cycle consumed)
Status: record only (2026-09-19, champion `3ddd4df` → `cf2439f`)

- Revert / control of the cycle-65 vicinal 1,4 contact term: effect size ±0.0015 per split is lottery-like (the
  keep was −0.0015 / −0.0015 with a small same-basin net); a control cycle cannot resolve it — not run.
- Endgame criterion analysis (reasoning on the fixed criteria): for a soft rotor (k ≈ 0.02 Ha/rad²) the force
  criterion (max 4.5e-4 Ha/Bohr) tolerates ≈ 7° of residual twist while the max-displacement criterion (1.8e-3 Å)
  tolerates ≈ 0.1° for an accurate model — the connected endgame is bound by the displacement criterion on soft
  modes, i.e. learning-limited; an over-stiff model along flat modes to pass it is gaming (self-imposed boundary),
  so no legitimate lever follows from this analysis.
- Charged dimers: the same-basin excess over the reference is only 14 / 17 calls per split (`/tmp/charged101.py`);
  the rest of the charged-class cost is diff-basin margin carriers (not to be shortened) and the approach walks of
  separated starts — the latter now docked for polyatomic cations (cycle 101).
- Cost map after cycle 100 (`/tmp/hist101.py`): connected 61 % of the cost at rel 0.70–0.73 uniform across call
  bins (mean 14.6 / 16.2 calls vs reference 20.7 / 23.9); neutral multi rel 0.45; charged multi rel 0.77 / 0.85.

## multi-start-docking — status: refuted (cycle 102, invalid)
- Attempt: cycle 102 (327825e, anchor cf2439f): lowest of 24 octahedral-orientation rigid-body surrogate minima of the smallest fragment. Train 0.6595679 (+0.0083), mean relE 0.9889 invalid; 95 dimers changed basin, credit Σ −8.08 lost / +1.28 gained, net +105 calls.
- Conditions tested: neutral + polyatomic-cation docked starts, all separations; surrogate = point charge + UFF 12-6 (cycle 70/101 constants). Refuted for the general case: the surrogate's global pose mis-ranks xTB basins; the near-start pose is the better basin predictor. Do not re-run without a surrogate that ranks H-bond/π contact patterns (DREIDING-type term not built, see cycle-101 closures).
- Related closures this segment: methyl-on-planar-centre class (no excess), 6 Å docking cap not binding, anion separated starts (6/8, margin risk), bare cations (2/3), dimer intra-relaxation ≈ 20 calls/Å vs connected 12.5.

## torsion-class-recalibration — status: closed (cycle 103, discard)
- Attempt: cycle 103 (`bbed8b6`, anchor `cf2439f`): `_torsion_class_factor` constants at the stiff end of their physical ranges — plain sp²–sp² 0.65 → 0.5 (carbonyl/nitro-conjugated sp²–sp² kept 0.65), sp²–lone-pair 0.45 → 0.35, carbonyl–lone-pair 0.7 → 1 (the cycle-37/91/93 "adverse class"); connected only. Train 0.6515659 (+0.000275), controls bit-identical, touched 120 molecules Σ −2 calls (29 : 25). [Evidence](evaluation_results/cycle-103-train.json), [Narrative](full_log.md#cycle-103--torsion-class-constants-of-connected-systems-pushed-to-the-stiff-end-of-their-physical-ranges-plain-sp²sp²-065--05-with-carbonylnitro-conjugated-sp²sp²-links-kept-at-065-sp²lone-pair-045--035-carbonyllone-pair-07--1-the-adverse-class-of-cycles-379193-returned-to-the-full-fischeralmlöf-constant-multi-fragment-systems-untouched).
- Why closed: the carbonyl–lp class is adverse in both directions (+0.33 / +0.30 per bond for 1 → 0.7 in cycle 37, +0.18 per bond for 0.7 → 1 now) — a path re-roll signature, not a stiffness response; the paying classes returned a third of the cycle-37 slope (plain sp²–sp² −0.09 per bond, sp²–lp ≈ 0): the first-step gain of the class scaling was harvested in cycle 37 and the TS-BFGS update removes the remaining 1.5–2× in one step. No implementation preserved (constants only, null result). Do not revert, split or scan the torsion classes again; the sp²–sp³ factor 0.6 was neutral in cycle 37 and untouched here.

## hydride-stretch-factors — status: closed (cycle 104, non_generalizable)
- Attempt: cycle 104 (`4d76093`, anchor `cf2439f`): `_HYDRIDE_SCALE` factors on `_h0_bond` for polar X–H bonds of connected systems (N–H ×1.2, O–H ×1.4, F–H ×2, S–H ×1.25, H–Cl ×1.3, from the textbook harmonic force constants; C–H, heavy–heavy bonds and dimer fragments untouched). Train 0.6510486 (−0.000242, gate passed on one in-contact complex), valid 0.6522417 (+0.002293). [Train](evaluation_results/cycle-104-train.json) `4680be5dadc743dda3d581b7e40ba4f2`, [Valid](evaluation_results/cycle-104-valid.json) `98b1f6bbd9c046359bacbab54b4f8a83`, [Narrative](full_log.md#cycle-104--hydride-factors-of-the-almlöf-stretch-guess-polar-xh-bonds-nh-12-oh-14-fh-2-sh-125-hcl-13-of-connected-systems-set-to-the-textbook-harmonic-force-constants), [Implementation](ideas/hydride-stretch-factors/4d760937b87265f9a0cdb0761e9c16056cf6f87c/algo.py).
- Why closed: 184 of 243 touched connected molecules keep their exact call count, the rest re-roll ±1–3 (train Σ +4, valid Σ +19); the ≥ 0.04 Å polar-error class that motivated the cycle (+2.9 / +3.3 residual calls on both splits) went +4 / +7. A 1.2–1.4× too-soft stiff bond is learned by the TS-BFGS update on the first step (its overshoot dominates the first secant pair in the |B| metric) — the cycle-84 reasoning, now measured. The polar-error marker is a hot polar rotor / wag, not stretch contraction. Strong-donor in-contact complexes (ammonium, acid, carboxylate–water) lose +8 / +9 calls with a stiffer donor stretch, weak donors (phenol) gain −4 / −7 — the cycle-61 shared-proton null with the opposite sign. Do not re-run, split (free vs H-bonded donors) or extend to fragments; C–F (0.80) is bundle-only. Stretch-guess calibration is closed for every bond class.
- Related static closures this cycle (no cycle consumed): acetate / carboxylate methyl start rule (no predictable train population, credit walkers), docking victims ≈ +0.003 / +0.0045 with no start-side separator, small inorganic molecules at 4–7 calls (stretch–stretch coupling, not diagonal), reference-bin structure (rel 0.755–0.78 / 0.70–0.72 / 0.65 for ref < 16 / 16–30 / ≥ 30), connected starts are thermal snapshots (X–H rms 0.025 Å, per-molecule max median 0.044 Å).

## stretch-transport-local-slope — status: closed (cycle 106, discard)
- Attempt: cycle 106 (`cefa1fc`, anchor `cf2439f`): the tracked stretch diagonal of the covalent bonds of connected systems (`Internals._h0_stretch_diagonal`, the analytic part T(x) of the cycle-53 model shift and secant transport) moves from the anchor-geometry guess with the Morse one-bond slope, k(r0) exp(−3Bb/2 (r − r0)) = 5.5 /Å (a = Bb/2 in the middle of the first-row / heavy-row Morse exponents, no fitted constant), instead of the Almlöf cross-bond exponential Bb = 3.7 /Å; anchor = the positions of the guess (start or rebuild geometry, shared by `copy` / `shadow_copy`); ion–ligand bonds of s-block metals, dummy bonds, all multi-fragment systems and every guess unchanged. Train 0.6569924 (+0.0057015), 469/469 converged, mean relE 1.0033042. [Evidence](evaluation_results/cycle-106-train.json) `fdd0157f9099415193ecfde29fc90bbe`, [Narrative](full_log.md#cycle-106--local-morse-curvaturelength-slope-for-the-tracked-stretch-part-of-the-model-the-transported-stretch-diagonal-of-connected-systems-moves-from-the-start-geometry-guess-with-exp3bb2-r--r0--55-å-instead-of-the-almlöf-cross-bond-exponential-bb--37-å-ionligand-bonds-dummy-bonds-and-multi-fragment-systems-unchanged), [Implementation](ideas/stretch-transport-local-slope/cefa1fcc68f29de64c681240c7c894255ac47e83/algo.py).
- Read-out (`/tmp/cmp106.py`): every multi-fragment system bit-identical (the 5 moved DES370K systems are single-fragment by the bond graph, i.e. connected in Sella's coordinates: Σ +9, carboxylates__thiols 30 → 42); connected same-endpoint 246 molecules Σ +4 (46 : 37, +0.0017), 4 basin changes Σ +35 (+0.0037: 135039840 SiOCl 10 → 33, 104079126 51 → 65 into a 1.7 kcal/mol lower minimum along a 1.4 Å bond change). The predicted population did not respond: n10 ≥ 2 H-containing −2 calls (6 : 4), H-containing lengthening ≥ 0.1 Å −2 (3 : 2), H-containing overall −6 over 180 (34 : 21, −0.0011, noise). The hydrogen-free inorganic molecules lost, both lengthening (+10, 2 : 6) and shortening (+13, 5 : 9), N ≤ 6 +9 (4 : 9), the champion's ≤ 9-call runs +19 (4 : 12): boron (135078067 B–C 9 → 16, 135205152 B–O 9 → 13), P–S (134980158 8 → 11) and Si–O–Cl bonds — the cycle-57 signature (an Almlöf k_e wrong by 1.2–2× for an exotic bond, its error now moved 1.5× faster).
- Why closed: the residual "+0.9 calls per bond changing > 0.1 Å" of the census is a proxy for distorted PubChem starts, not a stretch-transport lag (the organic population with such bonds is null under a 1.5× steeper transport); on the exotic-bond molecules the transport magnitude (the Almlöf k_e) dominates, and both slope directions have now been measured (cycle 53 Bb pays, cycle 106 3Bb/2 loses). Do not re-run with the anchored Morse form or an asymmetric slope. Untested variant, not worth a cycle on its own: a magnitude-free (multiplicative) transport of the bond components of the stored pairs, y_i ← y_i k_i(x)/k_i(avg), which would move the *measured* curvature of an exotic bond instead of the guess's — its population is the hydrogen-free set (share 0.054 train / 0.012 valid), i.e. non-generalizable by construction unless bundled.

## energy-corrected-secant — status: closed (cycles 107 + 108, discard twice; scalar attribution limit)
- Attempt: cycle 107 (`533da61`, anchor `cf2439f`): the newest secant pair of connected systems moved from the chord curvature of its step to the endpoint-local curvature with the energy change (Zhang–Deng–Chen modified quasi-Newton equation, `PES._energy_corrected_dg`): y ← y + β s with β = (ratio − 1)·chord/|s|², chord = y·s + s·(T(x)s − T̄s) (the analytic transport's own shift subtracted), local = (φ′(1) − φ′(0)) + 6(½(φ′(0) + φ′(1)) − ΔE) from the exact slopes g(x₀)·dx_initial, g(x₁)·dx_final; gates chord > 2 meV, |ratio − 1| ≥ 0.05, ratio ∈ [0.4, 3]; `ntrans + nrotations == 0` only. Train 0.6587707 (+0.0074798), 469/469 converged, mean relE 1.0087495. [Evidence](evaluation_results/cycle-107-train.json) `af9f4c1dfe5c4f8fbc4bb36cad5a5fdc`, [Narrative](full_log.md#cycle-107--energy-corrected-secant-pairs-zhangdengchen-modified-quasi-newton-equation-the-newest-pair-of-connected-systems-is-moved-from-the-chord-curvature-to-the-endpoint-local-curvature-along-its-step-with-the-energy-change-of-the-step-y--y--6a--8b--6δe--ys--stxs--t̄s-s-s²-gated-on-a-positive-chord-2-mev-of-curvature-energy-and-a--5--change-clipped-to-0430-the-chord-multi-fragment-systems-untouched), [Implementation](ideas/energy-corrected-secant/533da61d99ad0d9f1a3dd79dc2aa5694bdddfcec/algo.py).
- Sources: Zhang, Deng & Chen, J. Optim. Theory Appl. 102, 147 (1999), https://link.springer.com/article/10.1023/A:1021898630001; Zhang & Xu, J. Comput. Appl. Math. 137, 269 (2001), https://www.sciencedirect.com/science/article/pii/S0377042700007135 (vector-parameter form ỹ = y + θ u/(sᵀu) for any u with sᵀu ≠ 0); Babaie-Kafaki, Sci. China Math. 54, 2019 (2011), https://doi.org/10.1007/s11425-011-4232-7; Yabe & Takano, Comput. Optim. Appl. 28, 203 (2004), https://link.springer.com/article/10.1023/B:COAP.0000026885.81997.88.
- Read-out (`/tmp/cmp107.py`): all 212 multi-fragment systems bit-identical (test i passed); connected same-endpoint 250 molecules Σ +41 (57 : 48, +0.0027) — the gain expected on the large-rmsd₀ bins did not appear (rmsd₀ ≥ 0.7 Å: 53 molecules 21 : 14 but Σ +28 from two fat tails, 384221749 +16, 135316214 +11); 7 basin changes Σ +27 (+0.0048; deeper minima on 135094396 / 104079126 / 134997543, shallower on 135039840 10 → 32, 135097481, carboxylates__thiols); near-equilibrium runs not bit-identical (test ii failed: 34 molecules 2 : 6, Σ +4 — the gate is passed by stretch-dominated early steps); hydrogen-free +10 (11 : 15); charged connected Σ +25 over 24 (2 : 6). No feature split (rmsd₀, heavy, ref bin, n20, H content) explains the outcome.
- Diagnosed cause (static, from the formula and the coordinate metric): the rank-1 modification along u = s attributes the whole curvature-energy residual of the step to the components in proportion to s_i, i.e. in mixed units to the soft coordinates (torsions, radians) that dominate |s|², while the residual on early connected steps comes from the stiff bond components (Almlöf model ≠ true Morse curvature). A 0.04 eV stretch residual over |s|² ≈ 0.115 gives β ≈ 0.35 and doubles the torsional curvature of the pair. Hermite accuracy of the formula itself (static check): rotor steps ≤ 10 % error, always toward the ideal next chord; Morse steps ≤ 0.15 Å within 5 %, −0.3 → −0.05 Å 20 % under, −0.5 → −0.1 Å fails (ratio 0.11, clipped to 0.4) — consistent with the mild H-free loss but not the driver.
- Repair 1, cycle 108 (`cda4caa`, anchor `cf2439f`): (A) proportional attribution — the pair as used, y_u = y + (T(x₁)s − T̄s), rescaled by local/chord (Zhang–Xu vector parameter u = y_u, stored y' = y + (ratio − 1) y_u); (C) applied only when the bond components carry ≤ ½ of the chord curvature energy (`_ESEC_MAX_STIFF_SHARE = 0.5`); other gates and scope as cycle 107. Train 0.6548609 (+0.0035700), 469/469 converged, mean relE 1.0033056. [Evidence](evaluation_results/cycle-108-train.json) `7ded9bc93e714ace8b1b73c38e7a5872`, [Narrative](full_log.md#cycle-108--bounded-repair-of-the-energy-corrected-secant-pair-cycle-107-the-newest-pair-of-connected-systems-is-rescaled-as-used-y_u--localchord-y_u-with-y_u--y--tx₁s--t̄s-zhangxu-vector-parameter-form-u--y_u--the-residual-attributed-in-proportion-to-each-components-curvature-energy-instead-of-its-step-length-and-only-on-steps-whose-bond-components-carry--½-of-the-chord-curvature-energy-same-slopes-gates-2-mev-5--clip-043-and-scope-as-cycle-107-multi-fragment-systems-untouched), [Implementation](ideas/energy-corrected-secant/cda4caa11b58b7d343b9628e4877f4debf9f9072/algo.py).
- Read-out of the repair (`/tmp/cmp107.py cycle-108`, feature splits on `/tmp/feat107.pkl`): the diagnosed harms are gone — dimers 212/212 bit-identical, near-equilibrium runs (rmsd₀ < 0.15 Å) 2 : 1 Σ −1, hydrogen-free 7 : 8 Σ +3, charged connected 3 : 4 Σ 0, 83 connected runs bit-identical — but the target population does not respond: rmsd₀ ≥ 0.7 Å 18 : 18 Σ +22, n20 = 1 (single turning rotor) 5 : 6 Σ +2 (9 : 9 Σ +8 in cycle 107), n20 ≥ 4 11 : 9 Σ +12 (Σ +39 in 107) with the same long folding runs losing in both variants (172113611 35 → 46 / 42, 135140595 30 → 39 / 38, 134997543 28 → 40 / 31); the only consistent gain is n20 2–3: 13 : 6 Σ −15 (14 : 9 Σ −12 in 107), ≈ −0.3 calls on 48 molecules ≈ −0.001; the rest is ±1–3 re-rolls (rmsd₀ [0.15, 0.25) 3 : 7 Σ +11) and 135039840 (SiOCl, a floppy multi-minimum molecule) re-rolling 10 → 22 / 32 / 33 in cycles 108 / 107 / 106.
- Why closed: the energy of a step is one scalar; it corrects the curvature of a single soft mode exactly (Hermite ≤ 10 % on rotor paths) but has to be *attributed* on every mixed step — by step length (u = s, cycle 107: the stretch residual of many small bond steps lands on the torsions), by curvature-energy share (u = y_u, cycle 108: near-harmonic angles and the ≤ ½ bond block absorb the torsional residual, so the rotor is under-corrected, and on multi-rotor steps rotors on the concave and convex sides of their wells (local < chord and > chord) cancel in the scalar). The single-rotor population, where the attribution is unambiguous, shows no gain in either variant (the trust cap, not the chord softness, bounds those steps until the last one), and the multi-rotor folding runs lose under both. The reachable gain is the n20 2–3 bin, ≈ −0.001, below what a train + valid gate can confirm. Not tested and not planned (the same scalar limit applies): dihedral-block-only attribution (u = P_dih y_u) with a dihedral-share gate, a single-dominant-rotor gate (population = n20 = 1, non-responsive), the Yuan/Wei lower-order θ', per-block residuals (would need directional energy information the evaluator does not expose — one energy per force call), multi-fragment systems. The exact-slope bookkeeping in `PES.kick` (slopes = (g₀·dx_initial, g₁·dx_final, ΔE)) is reusable for any future energy-informed test of a step (e.g. a Hermite-based trust ratio), see the preserved implementations.

## cycle-109-idea-search-closures: static closures of the idea search before cycle 109 (saved-data analysis and public sources, no cycle consumed)
Status: record only (2026-09-19, champion `cf2439f`)

- Force-field Hessian as the initial guess (Mones, Csányi et al., "Exploration, sampling and optimisation of molecular geometries with model Hessians", arXiv:1804.01590 / Sci. Rep. 8, 13991 (2018)): the paper's own comparison puts empirical force-field Hessians level with Lindh / Almlöf-type model Hessians once the model is class-calibrated; every class of the present guess is already calibrated on this data (cycles 37–41, 49, 52, 64, 80, 81, 96) — closed, no cycle.
- Short-reference dimer excess is load-bearing: the DES370K systems that run slower than the reference on the same basin are few (train 24 Σ +10.21 rel-units, valid 24 Σ +15.89; same-endpoint part ≈ 0.007 / 0.008), while the different-basin walkers carry the energy margin (train 12 creditors Σ +2.005 / 16 debtors Σ −1.265; valid 21 creditors Σ +5.091 / 29 debtors Σ −2.747; all creditors train 36 Σ +2.839, valid 43 Σ +6.419) — any path change on them risks the valid gate (cycle-43 lesson). Not a lever.
- Connected cost regressions (cycle-101 endpoints, `/tmp/turns109.pkl`): ours ≈ 6.2–7.4 + 0.13–0.14 heavy + 10.7–12.6 (train) / 7.2–8.9 (valid) rmsd₀ + 0.5–0.8 n20 calls; maxturn adds nothing beyond rmsd₀ and n20 (turn steps run concurrently with the learning of the other modes); ours / reference 0.67–0.78 uniform over turn bins, element classes, sizes and rmsd₀ bins ([0.05, 0.1) 6.9 / 9.0 … [1.5, 10) 35.6 / 52.2); residuals of ours correlate 0.71 / 0.67 with the reference's residuals — the slow molecules are intrinsically hard, not a class the model misses. A cosine-fit rotor landing or a cap exemption for turning rotors (maxturn nil in the regressions; growth-only trust policies refuted in cycles 4 / 18–21) is closed.
- Neutral dimers: ours ≈ 15 + 19 · intra-fragment rmsd calls, the docking removed the rmsd₀ dependence; compressed DES370K starts show no excess (45 / 137 train, 42 / 132 valid, ratio 0.47 / 0.51 = the neutral mean); charged dimers ratio 0.74 / 0.82 with the same-basin excess 14 / 17 calls per split (cycle-101 closure) — no dimer class with a start-side or model-side excess left.
- Update-side variants: endgame-only SR1 / Bofill (cycle 33 measured SR1 / Bofill ×1.12 near the minimum and ×1.7–1.9 on the approach — negative expectation even when gated to the endgame; the TS-BFGS family is closed for good); structured diagonal-favouring (PSB-type) updates ≈ the cycle-58 fictitious-coupling algebra (≈ 0); QUICCA-type per-coordinate 1-D fits (Wittbrodt & Schlegel-type, https://arxiv.org/pdf/cond-mat/0404627) are the ill-posed per-coordinate reading of cycle 58; GDIIS = path change on the credit walkers (cycle 76 closure); soft-mode damping = gaming, over-stepping = the closed class factors (cycles 40 / 41).
- Guess-side: a surrogate rigid-body Hessian for the inter-fragment block (the cycle-81 intermolecular-mode survey found no correlation between model softness and calls); stiff-mode first-row calibration (O–H 28 % under-stiff, C–C 22 % over-stiff in the Almlöf guess — cycle 104 measured that such stiff-mode errors are learned on the first step, not binding); planar-start 6-ring saddle escapes (135132759: 27 calls vs reference 13, xTB puckers a 5,6-diaryl-1,2,4-triazine, our endpoint 0.012 kcal/mol lower — one molecule, no a-priori class); anion docking (expected ≈ −0.001, margin risk, cycle-102 closure); trust-region constant scans (bond-radius 0.3 Å, disfavoured by policy).
- Public sources read (nothing adaptable beyond what is implemented): optimizer comparison on xTB / DFT surfaces https://pubs.acs.org/doi/abs/10.1021/acs.jctc.3c00188; RL-trained optimizer https://pubs.acs.org/doi/abs/10.1021/acs.jctc.0c00971 (needs local training runs — forbidden here); ML leftmost-eigenvector TS optimizer https://arxiv.org/html/2603.21323v1 (saddle search); Lindh model Hessian https://arxiv.org/pdf/cond-mat/0211509 (the class-calibrated Almlöf guess already beats it on this data, cycle 5 / 64); xtb ANCopt documentation https://xtb-docs.readthedocs.io/en/latest/optimization.html (approximate normal coordinates = a Lindh-type model Hessian with a Cartesian-space update, the reference's coordinate choice is not ours to change); restricted-variance GEK https://pubs.acs.org/doi/10.1021/acs.jctc.0c00257 and GPR in internals https://pubs.acs.org/doi/abs/10.1021/acs.jctc.1c00517 (surrogate optimizers: full rewrite, no evidence they beat a calibrated quasi-Newton on ≤ 30-call runs; closed as a family in cycles 62 / 76). A Zenodo record that may mirror the benchmark data (https://zenodo.org/records/21135676) was not opened (repository restriction).
- New census, exotic-bond relaxers (a covalent bond with an element outside H / C / N / O / F changing > 0.05 Å between the start and the endpoint, connected systems): train 118 molecules (share 0.1817, 52 hydrogen-free, mean ours 13.8 / reference 19.4), 64 with a change > 0.1 Å (share 0.0992, 34 hydrogen-free); valid 110 (share 0.1635, 10 hydrogen-free), 47 with > 0.1 Å (share 0.0694, 9 hydrogen-free); recurrent pairs C–Cl, C–S, N–S, Br–C, O–P, O–S, P–S, S–S, B–O, C–I, C–Si, Cl–Si; s-block ion complexes are in the census (their ion–ligand bonds carry the cycle-52 ionic-radius Almlöf stretch and are transported like covalent bonds; `_ION_CHARGE` only reroutes their bends and terminal dihedrals). This is the population of the one mechanism-backed, never-measured candidate of the cycle-106 closure (a magnitude-free stretch transport), now with a valid exposure of 47 molecules instead of the 0.012 hydrogen-free share — taken up as cycle 109 in the per-step scalar form.

## stretch-transport-self-calibration — status: closed (cycle 109, discard)
- Attempt: cycle 109 (`f8bae55`, anchor `cf2439f`): the stretch part of the cycle-53 secant transport of connected systems scaled by the pair's own measured / modelled bond-block curvature-energy ratio, c_j = (y_j·s_j)_bonds / ((T̄_s,j s_j)·s_j)_bonds clipped to `_STRETCH_CALIB = (0.5, 2)`, y_j → y_j + (T(x) − T̄_j) s_j + (c_j − 1)(T_s(x) − T̄_s,j) s_j (T_s = P diag(d) P the projected stretch diagonal; contact block additive as before; c_j stored with the pair and passed through the symmetry images; multi-fragment systems untouched). Train 0.6546206 (+0.0033297), 469/469 converged, mean relE 1.0033091. [Evidence](evaluation_results/cycle-109-train.json) `6df5c4f480e5f54efeeb4a4f5b8c86cc`, [Narrative](full_log.md#cycle-109--self-calibrated-stretch-transport-of-the-secant-pairs-of-connected-systems-the-stretch-part-of-a-pairs-analytic-transport-is-scaled-by-the-pairs-own-measured--modelled-bond-block-curvature-energy-ratio-c_j--y_js_j_bonds--t̄_sj-s_js_j_bonds-clipped-to-05-2-ie-y_j--y_j--tx--t̄_j-s_j--c_j--1t_sx--t̄_sj-s_j-the-contact-block-stays-additive-multi-fragment-systems-untouched), [Implementation](ideas/stretch-transport-self-calibration/f8bae554eaa9a28acc2e51539a383f665fb442f9/algo.py).
- Read-out (`/tmp/cmp109.py`): all 212 multi-fragment systems bit-identical (the 7 DES370K systems that are single-fragment by Sella's bond graph re-rolled Σ +1: pyridine__pyrrole 21 → 15, alkanes__monoatomics 11 → 17); connected same-endpoint 249 molecules Σ +44 (20 : 28, +0.0033), 32 / 250 bit-identical; one basin change, 135097481 (N₄S₄ cage) 27 → 22 into a 7.35 kcal/mol *higher* minimum (relE 1.0000 → 0.9689). The target population did not respond: exotic-bond relaxers > 0.1 Å 61 same-endpoint molecules Σ +19 (6 : 10), hydrogen-free 33 Σ +4 (4 : 6; 135078067 B₄C cluster 9 → 17 as in cycle 106 9 → 16, 135094180 P₂O₄ 21 → 16, 135041910 9 → 6), H-containing 28 Σ +15 (2 : 4); bond change ≥ 0.2 Å 15 molecules Σ +15 (1 : 4). Non-exotic connected Σ +25 (14 : 18): the long folding runs lose again (champion > 20 calls 40 molecules Σ +33, 8 : 12; 135140595 30 → 40, 172113611 35 → 43, paliperidone_palmitate 56 → 60); near-equilibrium 34 molecules Σ +2 (2 : 3); rmsd₀ 0.4–0.7 Σ −13 (7 : 3) the only gaining bin.
- Why closed: the magnitude of the stretch transport is not binding on the exotic-bond population — the newest pair is re-measured along a relaxing bond on the next step, so a one-step 30 % curvature lag along the sampled direction costs nothing, while the small inorganic cages / clusters (N₄S₄, B₄C, P₂O₄, SiOCl: multi-centre bonding, strongly redundant coordinate sets) are the molecules where the bond-block ratio is dominated by the projection couplings and the calibration adds noise (a basin change and +8 on B₄C). With cycle 53 (the shape pays), cycle 106 (a steeper slope loses) and cycle 109 (the measured magnitude does not pay) the stretch-transport line is measured in every direction and closed. Not to be re-run with a per-bond, gated (bond-share) or unclipped estimator, or extended to fragments.
- Structural lesson (cross-cycle, `/tmp` analysis of cycles 103–109): every pair-side perturbation of connected systems costs +0.002–0.003 on train before any mechanism acts — cycles 107 / 108 / 109 (three different modifications of the pairs' bond block) lose on the same runs (135140595 +8 / +9 / +10, 172113611 +7 / +11 / +8, 135231613 +3 / +3 / +4, 135316214 +11 / +2 / +3; 135094180 gains −4 / −3 / −5 in all three), i.e. the champion's long folding paths are selected lucky draws at the train gate and regress to the mean under any perturbation of the update, whereas guess- / start-side changes that leave most runs bit-identical (cycles 103 / 104: Σ −2 / +4) do not pay this. A pair-side candidate must carry an expected mechanism gain ≥ 0.004 on train to be measurable; below that the train gate cannot pass it.
