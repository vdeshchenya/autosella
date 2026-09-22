# Research backlog

## tric-fragment-internals: Wang–Song TRICs for disconnected fragments
Status: revisiting

**Hypothesis:** Sella’s default internals grow the covalent cutoff until the graph is one fragment, adding chemically irrelevant intermolecular bonds/angles/dihedrals that restrict monomer translation/rotation. Enabling `allow_fragments=True` uses TRIC coordinates after one covalent pass and should reduce force calls on noncovalent dimers.

**Outcome and uncertainty:** Cycle 2 invalid (SCC crash) with a large unofficial DES370k cost win. Cycle 46 fragment-only TRIC + `wx=5` on the cycle-41 champion: crash-free, PubChem/Rowan identical, unofficial C=0.956, but 14 DES energy misses including amines–amines 6-step R=0.124.

**Reason to revisit:** Crash-free dimer cost win (cycle 46) with 14 localized energy misses. Full-run TRIC QN direction cannot be fixed by wx/δ/H0 (cycles 46–48). A short TRIC phase then connecting internals recovered amines in cycle 11.

**Next experiment:** Cycle 49: 3 force-call `wx=5` TRIC then connecting internals. Support: amines recovered, train V, Δ≥1e-4 vs `baa53f93`. Contradict: cycle-13-class hops or C≥1.

**Attempts:**
- Cycle 2; candidate `e89101bb077eed9e1a39a2aced9b54d0c985ccc9`; champion `7fa53b9aa21691454d13be8e506997cdffdbed1f`; decision invalid.
  [Implementation](ideas/tric-fragment-internals/e89101bb077eed9e1a39a2aced9b54d0c985ccc9/algo.py).
  [Training evidence](evaluation_results/cycle-2-train.json).
  [Narrative](full_log.md#cycle-2--tric-internals-for-disconnected-fragments).
- Cycle 46; candidate `7609b3d11b4188d68b308f7ef08341e00b3528a7`; champion `baa53f93e895f5dafb46eb48ef65fe583729f574`; decision invalid.
  [Implementation](ideas/tric-fragment-internals/7609b3d11b4188d68b308f7ef08341e00b3528a7/algo.py).
  [Training evidence](evaluation_results/cycle-46-train.json).
  [Narrative](full_log.md#cycle-46--fragment-only-tric-with-heavy-translation-weights).
- Cycle 47; candidate `db4f35a3c632e959783e7a10add9ca861f54b584`; champion `baa53f93e895f5dafb46eb48ef65fe583729f574`; decision invalid.
  [Training evidence](evaluation_results/cycle-47-train.json).
  [Narrative](full_log.md#cycle-47--fragment-tric-with-wx5-and-delta005).
- Cycle 48; candidate `474e6e4cc64d54d7359fdb0a70b3c56b67aee837`; champion `baa53f93e895f5dafb46eb48ef65fe583729f574`; decision invalid.
  [Training evidence](evaluation_results/cycle-48-train.json).
  [Narrative](full_log.md#cycle-48--stiffer-tric-translation-hessian-on-fragment-runs).




## tric-then-connected-internals: short TRIC phase then connecting internals
Status: revisiting

**Hypothesis:** A few TRIC force calls on disconnected fragments improve intermolecular packing; rebuilding champion connecting internals from the last evaluated geometry recovers the reference energy basin that full-TRIC missed.

**Outcome and uncertainty:** Cycles 11–13 invalid on energy. PubChem always matched. Amines recovered when the TRIC phase was long enough. A single TRIC kick still hops esters–monoatomics; three tiny kicks still hop alkanes–carboxylates. Hessian reset keeps DES cost ≥1. Mechanism contradicted for the energy+cost gates.

**Reason to revisit:** Only as a packing initializer if a later method can reject TRIC steps that leave the starting well *and* avoid rebuilding the Hessian (e.g. freeze translations instead of switching).

**Next experiment:** Do not two-stage TRIC until Hessian continuity is solved. See `sella-delta0-ode-halve` for the next cost-positive line.

**Attempts:**
- Cycle 11; candidate `b55c58cbc6bf05bfef563cf7cf1008cc2fceb6ad`; champion `7fa53b9aa21691454d13be8e506997cdffdbed1f`; decision invalid.
  [Implementation](ideas/tric-then-connected-internals/b55c58cbc6bf05bfef563cf7cf1008cc2fceb6ad/algo.py).
  [Training evidence](evaluation_results/cycle-11-train.json).
  [Narrative](full_log.md#cycle-11--two-stage-tric-then-connected-internals).
- Cycle 12; candidate `603c9c42ff99bf4ef986554f66542c85cf28e06d`; champion `7fa53b9aa21691454d13be8e506997cdffdbed1f`; decision invalid.
  [Implementation](ideas/tric-then-connected-internals/603c9c42ff99bf4ef986554f66542c85cf28e06d/algo.py).
  [Training evidence](evaluation_results/cycle-12-train.json).
  [Narrative](full_log.md#cycle-12--two-stage-tric-with-a-tiny-first-phase-trust-radius).
- Cycle 13; candidate `428d1a139bc205ed4c1196d277455e927558e301`; champion `7fa53b9aa21691454d13be8e506997cdffdbed1f`; decision invalid.
  [Implementation](ideas/tric-then-connected-internals/428d1a139bc205ed4c1196d277455e927558e301/algo.py).
  [Training evidence](evaluation_results/cycle-13-train.json).
  [Narrative](full_log.md#cycle-13--one-tiny-tric-kick-then-connecting-internals).

## tric-cartesian-step-cap: Cartesian max-atom cap on TRIC steps
Status: deferred

**Hypothesis:** Mixed-unit MaxInternalStep lets TRIC/internal steps realize large Cartesian motion; capping linearized max-atom displacement at 0.20 Å should prevent SCC-singular steps and basin hopping.

**Outcome and uncertainty:** Cycle 3 invalid, same anion–thiol crash after 7 successful 0.20 Å steps. Energy worse than cycle 2 (126 vs 57 unofficial failures) because the cap also hit connected PubChem/RowanSci molecules. Amines dimer path unchanged, so energy failure is not an oversized last step.

**Reason to revisit:** A *fragment-only* and/or tighter (~0.10 Å) cap might still prevent the SCC walk without touching connected molecules, if extra-redundant contacts leave a crash.

**Next experiment:** Apply `_cap_cartesian_step` only when `ntrans` or `nrotations` is nonzero, with `max_atom=0.10`, on top of TRICs (and contacts if those are kept).

**Attempts:**
- Cycle 3; candidate `495d68f34494fc4f278cc78169caac79f7b5e02a`; champion `7fa53b9aa21691454d13be8e506997cdffdbed1f`; decision invalid.
  [Implementation](ideas/tric-cartesian-step-cap/495d68f34494fc4f278cc78169caac79f7b5e02a/algo.py).
  [Training evidence](evaluation_results/cycle-3-train.json).
  [Narrative](full_log.md#cycle-3--tric-internals-with-cartesian-max-atom-step-cap).

## tric-extra-redundant-contacts: TRIC plus extra-redundant fragment distances
Status: deferred

**Hypothesis:** Adding three closest inter-fragment distances after angle/dihedral generation restrains starting dimer packing without intermolecular bends/torsions, fixing TRIC basin-hopping and the anion–thiol SCC crash while keeping some dimer cost win.

**Outcome and uncertainty:** Cycle 4 was `invalid` on energy (R=0.997) with no crash. Cost rose to 1.004. Amines dimer still 6 steps / rel_energy 0.124, so contacts do not determine that basin. PubChem/RowanSci matched the champion.

**Reason to revisit:** A single contact, or contacts without TRICs (distance-only intermolecular coordinates on champion internals), might recover energy with a smaller cost penalty. Combine with a fragment-only Cartesian cap only if a crash returns.

**Next experiment:** Cycle 18 tests extra-redundant contacts *without* TRICs on champion internals. If that fails, a single contact on TRICs is not justified.

**Attempts:**
- Cycle 4; candidate `e3b28b32d74b3afda8df54326f8928bc60704852`; champion `7fa53b9aa21691454d13be8e506997cdffdbed1f`; decision invalid.
  [Implementation](ideas/tric-extra-redundant-contacts/e3b28b32d74b3afda8df54326f8928bc60704852/algo.py).
  [Training evidence](evaluation_results/cycle-4-train.json).
  [Narrative](full_log.md#cycle-4--tric-plus-extra-redundant-inter-fragment-distances).

## sella-rfo-steps: RFO stepper on champion internals
Status: deferred

**Hypothesis:** RFO damps small/negative Hessian modes and can need fewer steps than Sella’s default `|λ|` quasi-Newton minimum stepper, without changing internals.

**Outcome and uncertainty:** Cycle 5 global RFO invalid (C=1.016, +38 kcal class). Cycle 63 dimer-only RFO invalid: connected safe, thiols +0.56 and acids–alkanes +0.20, C=1.005. Drop-in RFO is slower and hops two DES wells.

**Reason to revisit:** Only with a PD-eigenvalue switch plus a step cap, not as a dimer-wide method swap.

**Next experiment:** Do not use `method='rfo'` as a drop-in on dimers or the full set.

**Attempts:**
- Cycle 5; candidate `0e3b4be97e9e0cffe235605c518cbcdd00319f46`; champion `7fa53b9aa21691454d13be8e506997cdffdbed1f`; decision invalid.
  [Implementation](ideas/sella-rfo-steps/0e3b4be97e9e0cffe235605c518cbcdd00319f46/algo.py).
  [Training evidence](evaluation_results/cycle-5-train.json).
  [Narrative](full_log.md#cycle-5--rfo-steps-on-champion-internal-coordinates).
- Cycle 63; candidate `3e282497045a4d2324ebca61766d852ff8cfd3d3`; champion `e21f1d77e70cd5198ac760558e01de1d3d53139e`; decision invalid.
  [Implementation](ideas/sella-rfo-steps/3e282497045a4d2324ebca61766d852ff8cfd3d3/algo.py).
  [Training evidence](evaluation_results/cycle-63-train.json).
  [Narrative](full_log.md#cycle-63--dimer-only-rfo-steps-on-champion-internals).

## sella-larger-delta0: larger initial MaxInternalStep trust radius
Status: deferred

**Hypothesis:** Sella’s default `delta0=0.1` truncates early QN steps; 0.15 Å/rad lets them take more of the predicted displacement and reduces force calls on connected organics.

**Outcome and uncertainty:** Cycle 6 crashed on water-dimer geodesic ODE. Cycles 7–9 repaired the integrator; crash-free `delta0=0.15` (cycle 8) has C=0.987 and R=0.9999945 but still misses thiols/venetoclax energy. Cycle 9 (`delta0=0.12`) was cheaper (C=0.984) and worse on energy (R=0.99897). Larger MIS trust saves PubChem steps but misses the energy gate.

**Reason to revisit:** Pair a modest `delta0` increase with an energy-preserving polish (GDIIS, or two-stage small-trust finish) rather than further first-step tuning.

**Next experiment:** `delta0=0.15` with ODE-halve plus a late switch back to `delta0=0.10` / smaller steps after energy increases. Support: R≥1 and Δ≥1e-4. Contradict: energy still fails.

**Attempts:**
- Cycle 6; candidate `56cba87d7892b55e5dab0ce56b24f75d8130cc2f`; champion `7fa53b9aa21691454d13be8e506997cdffdbed1f`; decision invalid.
  [Implementation](ideas/sella-larger-delta0/56cba87d7892b55e5dab0ce56b24f75d8130cc2f/algo.py).
  [Training evidence](evaluation_results/cycle-6-train.json).
  [Narrative](full_log.md#cycle-6--larger-initial-maxinternalstep-trust-radius).

## sella-dimer-delta0-008: smaller initial dimer trust
Status: deferred

**Hypothesis:** Long DES jobs waste the first 0.1 Å connecting stretch. Starting dimers at `delta=0.08` with `delta_min=0.02` is more conservative than the champion.

**Outcome and uncertainty:** Cycle 72 discard. C=1.018; aggregate R passes via overshoots compensating thiols +0.56 and sulfides–water +0.17. Unofficial cheapening (alkanes–benzene 96→36) is not keepable.

**Reason to revisit:** Not as a first-step `delta0` change. A two-stage small-then-champion trust would be a different idea.

**Next experiment:** Do not interpolate dimer `delta0` to 0.09.

**Attempts:**
- Cycle 72; candidate `0cc494f8351365f4e53ae70c74dfeeab91adf789`; champion `a9c931a62562b4b5d904a2b373be7b3933200aa5`; decision discard.
  [Implementation](ideas/sella-dimer-delta0-008/0cc494f8351365f4e53ae70c74dfeeab91adf789/algo.py).
  [Training evidence](evaluation_results/cycle-72-train.json).
  [Narrative](full_log.md#cycle-72--dimer-delta008-smaller-initial-trust).

## sella-delta0-robust-geodesic: delta0=0.15 with iterative stepper and ODE halving
Status: deferred

**Hypothesis:** Cycle 6’s LSODA nfev>1000 is an implementation limit. Iterative Cartesian realization plus halved-step retry should keep the PubChem cost win without aborting.

**Outcome and uncertainty:** Cycle 7 invalid: amines–amines xTB SCC crash at call 105. Iterative stepper changed the path enough to hit a singular electronic-structure region.

**Reason to revisit:** Only if a later larger-trust run needs an alternative to ODE-halve that does not use `iterative_stepper=1`.

**Next experiment:** Do not combine `iterative_stepper=1` with enlarged `delta0` unless a fragment-only or energy-aware cap is added.

**Attempts:**
- Cycle 7; candidate `7df2c3fad396420816cd990a9789c3a374d5ffc0`; champion `7fa53b9aa21691454d13be8e506997cdffdbed1f`; decision invalid.
  [Implementation](ideas/sella-delta0-robust-geodesic/7df2c3fad396420816cd990a9789c3a374d5ffc0/algo.py).
  [Training evidence](evaluation_results/cycle-7-train.json).
  [Narrative](full_log.md#cycle-7--larger-trust-radius-with-robust-geodesic-updates).

## sella-delta0-ode-halve: delta0=0.15 with geodesic restore/halve only
Status: deferred

**Hypothesis:** Dropping the iterative stepper but keeping ODE restore/halve recovers cycle 6’s PubChem cost win without the cycle 7 SCC crash.

**Outcome and uncertainty:** Cycle 8 invalid on energy but crash-free: C=0.987, R=0.9999945. Water dimer recovered. Remaining misses: venetoclax +1.38 kcal, thiols +0.56, acids–alkanes +0.20.

**Reason to revisit:** Strongest cost signal in the delta0 family; combine with an energy-preserving finish rather than a smaller `delta0`.

**Next experiment:** Cycle 15: same restore with `df_actual > 1e-3` eV. Support: R≥1, no call-limit stall, Δ≥1e-4.

**Attempts:**
- Cycle 8; candidate `2a3a6545cb60cf75ee326293cb8c97bc29fdf132`; champion `7fa53b9aa21691454d13be8e506997cdffdbed1f`; decision invalid.
  [Implementation](ideas/sella-delta0-ode-halve/2a3a6545cb60cf75ee326293cb8c97bc29fdf132/algo.py).
  [Training evidence](evaluation_results/cycle-8-train.json).
  [Narrative](full_log.md#cycle-8--larger-trust-radius-with-ode-step-halving-only).

## sella-delta0-012: milder trust-radius increase with ODE halving
Status: deferred

**Hypothesis:** `delta0=0.12` keeps part of the PubChem cost win without venetoclax/thiol overshoot.

**Outcome and uncertainty:** Cycle 9 C=0.984 but R=0.99897, worse than cycle 8. Ammoniums–ketones appeared at rel_energy 0.682. End of the delta0 repair sequence.

**Reason to revisit:** Not as a standalone first-step change. Only as a parameter in a two-stage trust schedule.

**Next experiment:** See `sella-delta0-ode-halve`.

**Attempts:**
- Cycle 9; candidate `6758c719f3cd8b129e78710887e66277fbbc27fc`; champion `7fa53b9aa21691454d13be8e506997cdffdbed1f`; decision invalid.
  [Implementation](ideas/sella-delta0-012/6758c719f3cd8b129e78710887e66277fbbc27fc/algo.py).
  [Training evidence](evaluation_results/cycle-9-train.json).
  [Narrative](full_log.md#cycle-9--milder-trust-radius-increase-with-ode-step-halving).

## sella-sigma-inc-12: faster trust-radius growth after good steps
Status: incorporated

**Hypothesis:** Growing MaxInternalStep faster after well-predicted steps enlarges later Newton steps without changing the first-step cap.

**Outcome and uncertainty:** Cycle 10 invalid (dimer ODE timeout). Cycle 50 `σ_inc=1.18` after 6 connected steps: train pass, valid hops. Cycle 51 `nsteps>=20` + 1.18: train pass, valid cost loss (135065494 95→169). Cycle 52 `nsteps>=20` + `σ_inc=1.16` **keep** (current champion). Cycle 53 `nsteps>=16` + 1.16: discard; venetoclax 56→62 undoes the keep. Cycle 199 `σ_inc=1.17` after 20: discard, Δ=−1.38e-4; 172113611 50→52. Cycle 327 size-gated 1.17 on 30≤n<80: discard, Δ=−8.20e-5; only 172113611 50→52 extra; leftover unchanged.

**Reason to revisit:** Only if a later champion already has venetoclax-class long organics cheaper, so an earlier delay cannot inflate them. Do not raise 1.16. Do not interpolate 1.165 or re-size-gate 1.17 (cycle 327 extras the same amide).

**Next experiment:** Do not interpolate connected `σ_inc`. Cycle 200 dimer `σ_inc=1.16` after 80 was bit-identical.

**Attempts:**
- Cycle 10; candidate `7636e324179f349734b584975dd136105a5dadb0`; champion `7fa53b9aa21691454d13be8e506997cdffdbed1f`; decision invalid.
  [Implementation](ideas/sella-sigma-inc-12/7636e324179f349734b584975dd136105a5dadb0/algo.py).
  [Training evidence](evaluation_results/cycle-10-train.json).
  [Narrative](full_log.md#cycle-10--faster-trust-radius-growth-after-good-steps).
- Cycle 50; candidate `1a976a149a9bf3a4f8888d67759395ca7c2367da`; champion `baa53f93e895f5dafb46eb48ef65fe583729f574`; decision non_generalizable.
  [Implementation](ideas/sella-sigma-inc-12/1a976a149a9bf3a4f8888d67759395ca7c2367da/algo.py).
  [Training evidence](evaluation_results/cycle-50-train.json).
  [Validation evidence](evaluation_results/cycle-50-valid.json).
  [Narrative](full_log.md#cycle-50--delayed-sigmainc118-on-connected-molecules).
- Cycle 51; candidate `89206cc92924eb333f5a59eb9269659ffd6ad994`; champion `baa53f93e895f5dafb46eb48ef65fe583729f574`; decision non_generalizable.
  [Implementation](ideas/sella-sigma-inc-12/89206cc92924eb333f5a59eb9269659ffd6ad994/algo.py).
  [Training evidence](evaluation_results/cycle-51-train.json).
  [Validation evidence](evaluation_results/cycle-51-valid.json).
  [Narrative](full_log.md#cycle-51--later-delayed-sigmainc118-on-connected-molecules).
- Cycle 52; candidate `e21f1d77e70cd5198ac760558e01de1d3d53139e`; champion `baa53f93e895f5dafb46eb48ef65fe583729f574`; decision keep.
  [Implementation](ideas/sella-sigma-inc-12/e21f1d77e70cd5198ac760558e01de1d3d53139e/algo.py).
  [Training evidence](evaluation_results/cycle-52-train.json).
  [Validation evidence](evaluation_results/cycle-52-valid.json).
  [Narrative](full_log.md#cycle-52--milder-delayed-sigmainc116-after-20-connected-steps).
- Cycle 53; candidate `76152951ac0d2ae6ad3d5c555af5db0cd5868519`; champion `e21f1d77e70cd5198ac760558e01de1d3d53139e`; decision discard.
  [Implementation](ideas/sella-sigma-inc-12/76152951ac0d2ae6ad3d5c555af5db0cd5868519/algo.py).
  [Training evidence](evaluation_results/cycle-53-train.json).
  [Narrative](full_log.md#cycle-53--earlier-delayed-sigmainc116-after-16-connected-steps).
- Cycle 199; candidate `1fe57bac9bc74bfd034ea47e13136e5721b1b329`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-sigma-inc-12/1fe57bac9bc74bfd034ea47e13136e5721b1b329/algo.py).
  [Training evidence](evaluation_results/cycle-199-train.json).
  [Narrative](full_log.md#cycle-199--connected-σ_inc117-after-20-steps).
- Cycle 200; candidate `00f4a67d2e03185f76118d476f1c320d884b1b15`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-sigma-inc-12/00f4a67d2e03185f76118d476f1c320d884b1b15/algo.py).
  [Training evidence](evaluation_results/cycle-200-train.json).
  [Narrative](full_log.md#cycle-200--dimer-σ_inc116-after-80-steps).
- Cycle 327; candidate `234f07fdd7a45a78ed26c77c91ca36d056090254`; champion `362a4dd6db5ca11eea73f992d86141a50908bd75`; decision discard.
  [Implementation](ideas/sella-sigma-inc-117/234f07fdd7a45a78ed26c77c91ca36d056090254/algo.py).
  [Training evidence](evaluation_results/cycle-327-train.json).
  [Narrative](full_log.md#cycle-327--connected-σ_inc117-after-20-steps-on-30n_atoms80).

## sella-dimer-rho-dec: shrink dimer trust after mediocre ρ
Status: deferred

**Hypothesis:** Cycle 65’s `delta_min=0.02` only bites on rare ρ<0.01 shrinks. Lowering dimer `rho_dec` from 100 toward Sella’s saddle value of 5 makes mediocre predictions shrink, still floored at 0.02, which should cheapen 70–106 call DES jobs.

**Outcome and uncertainty:** Cycles 66–69 discard. `rho_dec=5`/`10` inflate mid-length DES; `20` is cost-positive but Δ=2.50e-5; `30` loses esters–esters and keeps two +1s. Nested low-ρ thresholds cannot isolate the 39→36 win.

**Reason to revisit:** Only with a non-threshold mechanism that can keep esters–esters without alkanes–amides/phenol–pyrrole +1s.

**Next experiment:** Do not interpolate `rho_dec` further.

**Attempts:**
- Cycle 66; candidate `c91fbcd55ed378df0d548892a52aa3d4975606d1`; champion `a9c931a62562b4b5d904a2b373be7b3933200aa5`; decision discard.
  [Implementation](ideas/sella-dimer-rho-dec/c91fbcd55ed378df0d548892a52aa3d4975606d1/algo.py).
  [Training evidence](evaluation_results/cycle-66-train.json).
  [Narrative](full_log.md#cycle-66--dimer-rho_dec5-with-the-002-trust-floor).
- Cycle 67; candidate `9cdcc4e9129adbbe08ffd775d014471dc785b220`; champion `a9c931a62562b4b5d904a2b373be7b3933200aa5`; decision discard.
  [Implementation](ideas/sella-dimer-rho-dec/9cdcc4e9129adbbe08ffd775d014471dc785b220/algo.py).
  [Training evidence](evaluation_results/cycle-67-train.json).
  [Narrative](full_log.md#cycle-67--milder-dimer-rho_dec10-with-the-002-trust-floor).
- Cycle 68; candidate `4b2cd78bbb240954e1c4a78194991bd9cf55278d`; champion `a9c931a62562b4b5d904a2b373be7b3933200aa5`; decision discard.
  [Implementation](ideas/sella-dimer-rho-dec/4b2cd78bbb240954e1c4a78194991bd9cf55278d/algo.py).
  [Training evidence](evaluation_results/cycle-68-train.json).
  [Narrative](full_log.md#cycle-68--milder-dimer-rho_dec20-with-the-002-trust-floor).
- Cycle 69; candidate `566f5a9b57314f5154efc685cd38369c5d6809f5`; champion `a9c931a62562b4b5d904a2b373be7b3933200aa5`; decision discard.
  [Implementation](ideas/sella-dimer-rho-dec/566f5a9b57314f5154efc685cd38369c5d6809f5/algo.py).
  [Training evidence](evaluation_results/cycle-69-train.json).
  [Narrative](full_log.md#cycle-69--milder-dimer-rho_dec30-with-the-002-trust-floor).

## sella-dimer-rho-inc: narrower dimer trust-expansion window
Status: deferred

**Hypothesis:** Long DES jobs expand δ on mediocre ρ in Sella’s 0.75–1.33 window. Narrowing dimer `rho_inc` to 1.25 should keep trust from inflating.

**Outcome and uncertainty:** Cycle 71 invalid. Sulfides–water and amides–phenol hop to cheaper higher wells; amines 99→112. Narrower expansion is energy-unsafe here.

**Reason to revisit:** Only if a well-preserving step cap is already in place.

**Next experiment:** Do not interpolate `rho_inc` on dimers.

**Attempts:**
- Cycle 71; candidate `eb5bc7d9cbaceeaa3a9d108c049c86ce7e13da7a`; champion `a9c931a62562b4b5d904a2b373be7b3933200aa5`; decision invalid.
  [Implementation](ideas/sella-dimer-rho-inc/eb5bc7d9cbaceeaa3a9d108c049c86ce7e13da7a/algo.py).
  [Training evidence](evaluation_results/cycle-71-train.json).
  [Narrative](full_log.md#cycle-71--dimer-rho_inc125-narrower-trust-expansion).

## sella-dimer-delayed-sigma-inc: late trust-radius growth on dimers
Status: revisiting

**Hypothesis:** Cycle 52’s delayed `σ_inc=1.16` never fires on dimers. Applying it after 20 dimer steps should cheapen long DES jobs without the connecting-bond enlargements that hopped in cycles 54–57.

**Outcome and uncertainty:** Cycle 58 `@20` on dimers: energy-safe, C worse. Cycle 59 `@40`: amines 99→128 still dominates; net Δ=−5.1e-4. Later delays cannot beat the champion without a general (non-identity) way to spare amines.

**Reason to revisit:** Only if a later method already has amines on a short energy-safe path, so late trust growth cannot inflate it.

**Next experiment:** Do not delay dimer `σ_inc` further. `sigma_dec` rarely fires while `rho_dec=100`; test dimer-only `rho_dec` first (`sella-dimer-rho-dec`).

**Attempts:**
- Cycle 58; candidate `74c854450cfc3ea9cc902ad60d6d52b296b3d6e0`; champion `e21f1d77e70cd5198ac760558e01de1d3d53139e`; decision discard.
  [Implementation](ideas/sella-dimer-delayed-sigma-inc/74c854450cfc3ea9cc902ad60d6d52b296b3d6e0/algo.py).
  [Training evidence](evaluation_results/cycle-58-train.json).
  [Narrative](full_log.md#cycle-58--delayed-sigmainc116-after-20-steps-on-dimers-too).
- Cycle 59; candidate `c71aba7b3d7f31eaf5a6f8197f658e7954ff685c`; champion `e21f1d77e70cd5198ac760558e01de1d3d53139e`; decision discard.
  [Implementation](ideas/sella-dimer-delayed-sigma-inc/c71aba7b3d7f31eaf5a6f8197f658e7954ff685c/algo.py).
  [Training evidence](evaluation_results/cycle-59-train.json).
  [Narrative](full_log.md#cycle-59--delayed-dimer-sigmainc116-after-40-steps).

## sella-dimer-stiff-long-h0: stiffer Lindh Hessian on connecting stretches
Status: revisiting

**Hypothesis:** Tiny Lindh h0 on long connecting bonds makes QN ride the weak connector. 4× those guesses on dimers should mix monomer internals without raising the 0.1 Å cap.

**Outcome and uncertainty:** Cycle 60 4× invalid (ammoniums–ketones R=0.682, C>1). Cycle 61 2× discard: aggregate R valid, C=1.002, acids–alkanes +0.20 kcal. Multiplier family exhausted.

**Reason to revisit:** Only combined with a well-preserving coordinate change, not another scale factor.

**Next experiment:** Do not interpolate 1.5×. Try dimer iterative stepper or Cartesian cap.

**Attempts:**
- Cycle 60; candidate `811bd5d311122ac484a46c76e41ec059a126b1ad`; champion `e21f1d77e70cd5198ac760558e01de1d3d53139e`; decision invalid.
  [Implementation](ideas/sella-dimer-stiff-long-h0/811bd5d311122ac484a46c76e41ec059a126b1ad/algo.py).
  [Training evidence](evaluation_results/cycle-60-train.json).
  [Narrative](full_log.md#cycle-60--stiffer-lindh-hessian-on-dimer-connecting-bonds).
- Cycle 61; candidate `9f6d613af826e54e8d359d745c7d3b450ed637a1`; champion `e21f1d77e70cd5198ac760558e01de1d3d53139e`; decision discard.
  [Implementation](ideas/sella-dimer-stiff-long-h0/9f6d613af826e54e8d359d745c7d3b450ed637a1/algo.py).
  [Training evidence](evaluation_results/cycle-61-train.json).
  [Narrative](full_log.md#cycle-61--milder-2-lindh-hessian-on-dimer-connecting-bonds).

## sella-dimer-interfrag-wb: larger MaxInternalStep on long connecting stretches
Status: deferred

**Hypothesis:** Connecting-bond stretches that join covalent fragments are capped at 0.1 Å like covalent bonds. OptKing-style larger interfragment stretch steps on dimers only should cheapen DES370k approach without touching connected organics.

**Outcome and uncertainty:** Cycle 54 `wb_long=0.4` (0.25 Å): unofficial ΔC≈0.013, 7 DES hops. Cycle 55 `wb_long=0.5` (0.20 Å): amines ODE after N–N 4.5 nm. Cycle 56 r≤3.5 Å window: aggregate R valid, C worse. Cycle 57 `wb_long=0.67` (0.15 Å): aggregate R valid, C=1.002, thiols/phenol/acids–alkanes still miss. Cycle 193 limiter-only wd=0.8 after 20: invalid, amines–amines 99→94 (R=0.998), benzene–ethers miss, Δ=−3.88e-4. Caps that unofficially cheapen DES leave the reference wells; limiter-only 0.8 is the same hop class.

**Reason to revisit:** Not as a connecting `wb` tweak, including limiter-only.

**Next experiment:** Do not interpolate connecting `wb`. See dummy-excluded TrustRegion.

**Attempts:**
- Cycle 54; candidate `7eb11a51baaac0692bcd99d11c142aee51c9ba38`; champion `e21f1d77e70cd5198ac760558e01de1d3d53139e`; decision invalid.
  [Implementation](ideas/sella-dimer-interfrag-wb/7eb11a51baaac0692bcd99d11c142aee51c9ba38/algo.py).
  [Training evidence](evaluation_results/cycle-54-train.json).
  [Narrative](full_log.md#cycle-54--dimer-only-larger-connecting-bond-stretch-steps).
- Cycle 55; candidate `141160341f0ddfa0f482985fd7d6c7ff3d0f83e3`; champion `e21f1d77e70cd5198ac760558e01de1d3d53139e`; decision invalid.
  [Implementation](ideas/sella-dimer-interfrag-wb/141160341f0ddfa0f482985fd7d6c7ff3d0f83e3/algo.py).
  [Training evidence](evaluation_results/cycle-55-train.json).
  [Narrative](full_log.md#cycle-55--tighter-dimer-long-bond-stretch-cap-020-å).
- Cycle 56; candidate `2682a542952eb23d7a2998b9d668893c1935465a`; champion `e21f1d77e70cd5198ac760558e01de1d3d53139e`; decision discard.
  [Implementation](ideas/sella-dimer-interfrag-wb/2682a542952eb23d7a2998b9d668893c1935465a/algo.py).
  [Training evidence](evaluation_results/cycle-56-train.json).
  [Narrative](full_log.md#cycle-56--contact-window-dimer-stretch-cap-020-å-r--35-å).
- Cycle 57; candidate `f10062d750f1821bebe6eb6ea2891d64acbf077f`; champion `e21f1d77e70cd5198ac760558e01de1d3d53139e`; decision discard.
  [Implementation](ideas/sella-dimer-interfrag-wb/f10062d750f1821bebe6eb6ea2891d64acbf077f/algo.py).
  [Training evidence](evaluation_results/cycle-57-train.json).
  [Narrative](full_log.md#cycle-57--milder-dimer-long-bond-stretch-cap-015-å).
- Cycle 193; candidate `f8144253e31f07c58c91c7c768a35561da1a4487`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision invalid.
  [Implementation](ideas/sella-dimer-interfrag-wb/f8144253e31f07c58c91c7c768a35561da1a4487/algo.py).
  [Training evidence](evaluation_results/cycle-193-train.json).
  [Narrative](full_log.md#cycle-193--limiter-connecting-stretch-wb08-after-20-dimer-steps).

## sella-dimer-delta-min-floor: prevent collapsed trust on dimers
Status: incorporated

**Hypothesis:** Poor-ρ shrinks can drop δ to η=1e-4 on DES dimers, stalling later QN steps. A dimer-only `delta_min` floor keeps useful step length after bad ρ.

**Outcome and uncertainty:** Cycle 64 `delta_min=0.05` passed train then wandered valid benzene–water 47→131. Cycle 65 `delta_min=0.02` **keep**: train Δ=1.175e-4, valid Δ=1.103e-4, benzene–water stayed 47. The 0.02–0.05 gap is ~2e-6 in train C, so further floor interpolation cannot keep vs this champion.

**Reason to revisit:** Only if a later method needs a different floor (e.g. with a tighter `rho_dec` that shrinks more often). Do not retune 0.02 vs 0.03 vs 0.05 alone.

**Next experiment:** See `sella-dimer-rho-dec`: lower dimer `rho_dec` so mediocre ρ shrinks, still floored at 0.02.

**Attempts:**
- Cycle 64; candidate `fd5f5683ad416dfac18c740818e2506caa9ab0fc`; champion `e21f1d77e70cd5198ac760558e01de1d3d53139e`; decision non_generalizable.
  [Implementation](ideas/sella-dimer-delta-min-floor/fd5f5683ad416dfac18c740818e2506caa9ab0fc/algo.py).
  [Training evidence](evaluation_results/cycle-64-train.json).
  [Validation evidence](evaluation_results/cycle-64-valid.json).
  [Narrative](full_log.md#cycle-64--dimer-trust-radius-floor-delta_min005).
- Cycle 65; candidate `a9c931a62562b4b5d904a2b373be7b3933200aa5`; champion `e21f1d77e70cd5198ac760558e01de1d3d53139e`; decision keep.
  [Training evidence](evaluation_results/cycle-65-train.json).
  [Validation evidence](evaluation_results/cycle-65-valid.json).
  [Narrative](full_log.md#cycle-65--milder-dimer-trust-radius-floor-delta_min002).

## sella-reject-uphill-steps: restore geodesic trials that raise energy
Status: deferred

**Hypothesis:** Cycle 8’s larger trust radius keeps some uphill geodesic steps; restoring the previous PES state when energy rises preserves the reference basin while retaining most of the cost win.

**Outcome and uncertainty:** Cycles 14–16 invalid. Venetoclax energy recovered at all three thresholds. Amines stalled at 200 calls even at 0.01 eV. Thiols remained +0.56 kcal (downhill). Cost always >1. Mechanism contradicted as a drop-in on `delta0=0.15`.

**Reason to revisit:** Only if combined with a line search that halves the step *before* evaluating a predicted-uphill move, so rejected trials are cheaper. Not as a post-eval restore.

**Next experiment:** Predicted-energy line search (scale s until df_pred < 0 and |s| small) without restoring after the eval.

**Attempts:**
- Cycle 14; candidate `2a8054d1dbd203e8bb1ed987cad596dd0d451fc4`; champion `7fa53b9aa21691454d13be8e506997cdffdbed1f`; decision invalid.
  [Implementation](ideas/sella-reject-uphill-steps/2a8054d1dbd203e8bb1ed987cad596dd0d451fc4/algo.py).
  [Training evidence](evaluation_results/cycle-14-train.json).
  [Narrative](full_log.md#cycle-14--larger-trust-radius-with-uphill-step-restore).
- Cycle 15; candidate `50658a54c78348e621d43e9970e3feeca1134eae`; champion `7fa53b9aa21691454d13be8e506997cdffdbed1f`; decision invalid.
  [Implementation](ideas/sella-reject-uphill-steps/50658a54c78348e621d43e9970e3feeca1134eae/algo.py).
  [Training evidence](evaluation_results/cycle-15-train.json).
  [Narrative](full_log.md#cycle-15--milder-uphill-restore-at-delta015).
- Cycle 16; candidate `ff62fb70edcb0f7df0f884d91a9998896d670255`; champion `7fa53b9aa21691454d13be8e506997cdffdbed1f`; decision invalid.
  [Implementation](ideas/sella-reject-uphill-steps/ff62fb70edcb0f7df0f884d91a9998896d670255/algo.py).
  [Training evidence](evaluation_results/cycle-16-train.json).
  [Narrative](full_log.md#cycle-16--001-ev-uphill-restore-at-delta015).

## sella-cartesian-ras: Cartesian RestrictedAtomicStep
Status: deferred

**Hypothesis:** Uniform Å trust on 3N Cartesian coordinates moves monomers without mixed-unit internal restrictions.

**Outcome and uncertainty:** Cycle 17 invalid: thiols SCC crash, 47 force-call limits, unofficial C≈4. Cartesian is much slower and less stable than internals here.

**Reason to revisit:** Only as a short Cartesian pre-relaxation with a tight max-atom cap, then internals — not as a full run.

**Next experiment:** At most 5 Cartesian RAS steps with a 0.1 Å atomic cap, then champion internals (if extra-redundant contacts fail).

**Attempts:**
- Cycle 17; candidate `aefc67d6b7c7d0fcde880af91f8a7f69566466c0`; champion `7fa53b9aa21691454d13be8e506997cdffdbed1f`; decision invalid.
  [Implementation](ideas/sella-cartesian-ras/aefc67d6b7c7d0fcde880af91f8a7f69566466c0/algo.py).
  [Training evidence](evaluation_results/cycle-17-train.json).
  [Narrative](full_log.md#cycle-17--sella-cartesian-restrictedatomicstep).

## sella-bakken-13-bonds: extra-redundant 1-3 distances on connected molecules
Status: deferred

**Hypothesis:** Bakken–Helgaker extra-redundant 1–3 stretches across valence angles cheapen connected organics without touching dimer connecting internals.

**Outcome and uncertainty:** Cycle 70 invalid. 135043047/252089162 hit 200 calls (+53/+86 kcal). Diagnostics: stall at max-force ~0.048 Eh/Bohr then a 0.55 bohr hop. Extra 1–3 as active coordinates are unsafe here, matching cycle 18’s extra-distance hops.

**Reason to revisit:** Not as active internals. A frozen 1–3 constraint or Hessian coupling without adding Bond objects would be a different idea.

**Next experiment:** Do not add angle-span bonds. Do not retune with an r cutoff. Cycle 75 tests the other Bakken class (extra impropers), not stretches.

**Attempts:**
- Cycle 70; candidate `9d225da3f8083e26d70590ee2e514ecc60dfb8c2`; champion `a9c931a62562b4b5d904a2b373be7b3933200aa5`; decision invalid.
  [Implementation](ideas/sella-bakken-13-bonds/9d225da3f8083e26d70590ee2e514ecc60dfb8c2/algo.py).
  [Training evidence](evaluation_results/cycle-70-train.json).
  [Narrative](full_log.md#cycle-70--connected-bakken-extra-redundant-1-3-distances).

## extra-redundant-no-tric: contacts on champion connecting internals
Status: deferred

**Hypothesis:** Extra-redundant closest distances between covalent fragments, added after angles/dihedrals on champion internals (no TRICs), shorten dimer packing steps without TRIC basin hops.

**Outcome and uncertainty:** Cycle 18 invalid: PubChem matched; DES C=1.163; amines 6-step hop (rel_energy 0.124) as in full TRIC. Extra intermolecular distances hop dimers even on connecting internals.

**Reason to revisit:** Not as active coordinates. A frozen contact constraint is a different idea.

**Next experiment:** Do not add extra intermolecular bonds. See `sella-delayed-delta015`.

**Attempts:**
- Cycle 18; candidate `df391d992661c1e1f7fa09fcbbfeac41de6f6e68`; champion `7fa53b9aa21691454d13be8e506997cdffdbed1f`; decision invalid.
  [Implementation](ideas/extra-redundant-no-tric/df391d992661c1e1f7fa09fcbbfeac41de6f6e68/algo.py).
  [Training evidence](evaluation_results/cycle-18-train.json).
  [Narrative](full_log.md#cycle-18--extra-redundant-covalent-fragment-contacts).

## sella-delayed-delta015: raise MIS trust to 0.15 after a few champion steps
Status: incorporated

**Hypothesis:** `delta0=0.15` from the first step overshoots a few basins; starting at 0.10 then flooring δ at 0.15 after several steps keeps the reference well and still cheapens later Newton steps.

**Outcome and uncertainty:** Cycle 19–24 on the original champion: only cycle 22 passed train (valid energy miss). Cycle 88 on the cycle-65 champion delayed the floor to nsteps>=20 with persistent `delta_min=0.15` and **kept** (train Δ=+2.47e-4, valid Δ=+2.40e-4).

**Reason to revisit:** Pair the 0.15 floor with `wa=1` after 20 so angle steps stay ≤0.15 rad while stretches/torsions use the larger trust (venetoclax 56→57, paliperidone 76→78 under wa=0.75). Cycle 198 0.16 floor: discard, paliperidone 78→85.

**Next experiment:** Do not raise the 0.15 connected floor.

**Attempts:**

**Hypothesis:** `delta0=0.15` from the first step overshoots a few basins; starting at 0.10 then flooring δ at 0.15 after several steps keeps the reference well and still cheapens later Newton steps.

**Outcome and uncertainty:** Cycle 19 discard (valid, C=1.000033). Cycle 20 earlier boost discard (C=1.0015). Cycles 21–22 fragment-gated 0.15: train pass (C=0.999756, Δ=2.44e-4) but valid R=0.999999684 (`des370k_monoatomics__sulfides__valid` +0.00093 kcal). Cycle 23 later floor and cycle 24 0.12 floor both failed train energy. Hard 0.15 floor on connected molecules is the remaining energy mechanism; timing/magnitude tweaks are exhausted.

**Reason to revisit:** Cycle 22 is still the strongest rejected keep-candidate (train+valid cost win). Combine only with a change that does not shift connected wells (e.g. dihedral-only trust, not another floor). Cycle 88 tests the 0.15 floor delayed to nsteps>=20 on the cycle-65 champion.

**Next experiment:** Cycle 94 switch `H.update_method` to `'BFGS'` at nsteps==20 on connected molecules (keep TS-BFGS for the first 20). Support: V_train, Δ≥1e-4 vs `453a761`. Contradict: cycle-31-class energy miss or Δ<1e-4.

**Attempts:**
- Cycle 19; candidate `11436a8a134c6225d2b62f7f9ccb2fa39a66cc2a`; champion `7fa53b9aa21691454d13be8e506997cdffdbed1f`; decision discard.
  [Implementation](ideas/sella-delayed-delta015/11436a8a134c6225d2b62f7f9ccb2fa39a66cc2a/algo.py).
  [Training evidence](evaluation_results/cycle-19-train.json).
  [Narrative](full_log.md#cycle-19--delayed-maxinternalstep-increase-to-015).
- Cycle 20; candidate `a42eb755edeaa80780aec436a9800d4fbcc54701`; champion `7fa53b9aa21691454d13be8e506997cdffdbed1f`; decision discard.
  [Implementation](ideas/sella-delayed-delta015/a42eb755edeaa80780aec436a9800d4fbcc54701/algo.py).
  [Training evidence](evaluation_results/cycle-20-train.json).
  [Narrative](full_log.md#cycle-20--earlier-delayed-maxinternalstep-increase).
- Cycle 21; candidate `d7aabfc2e443ef00f10646be5f3406d02d7676f9`; champion `7fa53b9aa21691454d13be8e506997cdffdbed1f`; decision non_generalizable.
  [Implementation](ideas/sella-delayed-delta015/d7aabfc2e443ef00f10646be5f3406d02d7676f9/algo.py).
  [Training evidence](evaluation_results/cycle-21-train.json).
  [Validation evidence](evaluation_results/cycle-21-valid.json).
  [Narrative](full_log.md#cycle-21--delayed-015-trust-floor-on-connected-molecules-only).
- Cycle 22; candidate `f791972145ad638a99ed51520b88e3b7bbaf7d27`; champion `7fa53b9aa21691454d13be8e506997cdffdbed1f`; decision non_generalizable.
  [Implementation](ideas/sella-delayed-delta015/f791972145ad638a99ed51520b88e3b7bbaf7d27/algo.py).
  [Training evidence](evaluation_results/cycle-22-train.json).
  [Validation evidence](evaluation_results/cycle-22-valid.json).
  [Narrative](full_log.md#cycle-22--fragment-gated-015-floor-on-champion-geodesics).
- Cycle 23; candidate `bc67bfee943002593aac79867bcf1f23b8ef5dbb`; champion `7fa53b9aa21691454d13be8e506997cdffdbed1f`; decision invalid.
  [Implementation](ideas/sella-delayed-delta015/bc67bfee943002593aac79867bcf1f23b8ef5dbb/algo.py).
  [Training evidence](evaluation_results/cycle-23-train.json).
  [Narrative](full_log.md#cycle-23--later-fragment-gated-015-trust-floor).
- Cycle 24; candidate `6d21bb2a78f12bdaacc84287556f7d6c11545ace`; champion `7fa53b9aa21691454d13be8e506997cdffdbed1f`; decision invalid.
  [Implementation](ideas/sella-delayed-delta015/6d21bb2a78f12bdaacc84287556f7d6c11545ace/algo.py).
  [Training evidence](evaluation_results/cycle-24-train.json).
  [Narrative](full_log.md#cycle-24--fragment-gated-012-trust-floor-after-five-steps).
- Cycle 88; candidate `453a761dd3eace02562f81dc72920ac1e38526cc`; champion `a9c931a62562b4b5d904a2b373be7b3933200aa5`; decision keep.
  [Training evidence](evaluation_results/cycle-88-train.json).
  [Validation evidence](evaluation_results/cycle-88-valid.json).
  [Narrative](full_log.md#cycle-88--connected-trust-radius-floor-015-after-20-steps).
- Cycle 198; candidate `b270883f18621472d190f30d8072d5c22d76e049`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-delayed-delta015/b270883f18621472d190f30d8072d5c22d76e049/algo.py).
  [Training evidence](evaluation_results/cycle-198-train.json).
  [Narrative](full_log.md#cycle-198--connected-trust-floor-016-after-20-steps).

## sella-rho-inc-15: wider trust-expansion ratio window
Status: deferred

**Hypothesis:** Growing δ when 1/1.5 < ρ < 1.5 instead of 1/(4/3) lets more nearly-quadratic steps expand MIS without a hard floor.

**Outcome and uncertainty:** Cycle 25 invalid: C=1.00068, R=0.999983, max +0.14 kcal. Wider expansion admits poorly predicted steps; no cost win.

**Reason to revisit:** Only if a method already has R=1 and is truncated by a too-narrow good-ρ window, with a cap on δ.

**Next experiment:** Do not widen ρ_inc alone. A narrower window (ρ_inc closer to 1) is a different, more conservative idea.

**Attempts:**
- Cycle 25; candidate `3b52044dbabb9c858aa4db1e969322e0b5161d98`; champion `7fa53b9aa21691454d13be8e506997cdffdbed1f`; decision invalid.
  [Implementation](ideas/sella-rho-inc-15/3b52044dbabb9c858aa4db1e969322e0b5161d98/algo.py).
  [Training evidence](evaluation_results/cycle-25-train.json).
  [Narrative](full_log.md#cycle-25--wider-trust-expansion-ratio-window).

## sella-sigma-dec-095: milder trust shrink after poor ρ
Status: deferred

**Hypothesis:** After ρ < 1/ρ_dec or ρ > ρ_dec, δ ← max(smag·0.95, δ_min) instead of 0.90 keeps more step length without changing the expansion rule.

**Outcome and uncertainty:** Cycle 26 discard: train valid (R=1.000092) but C=1.00585. Venetoclax +1.38 kcal; several dimers much slower. Milder shrink after rare bad-ρ steps increases wandering.

**Reason to revisit:** The opposite (more aggressive shrink, σ_dec=0.80) might stabilize a larger-trust method, not the champion.

**Next experiment:** Pair σ_dec<0.90 only with a crash-free cost-positive larger-trust candidate that still misses energy. Champion-only harder shrink (cycle 74, 0.80) was cost-negative; do not interpolate.

**Attempts:**
- Cycle 26; candidate `868930ec55aae141333fbd6b9f8b56affbc8f1c3`; champion `7fa53b9aa21691454d13be8e506997cdffdbed1f`; decision discard.
  [Implementation](ideas/sella-sigma-dec-095/868930ec55aae141333fbd6b9f8b56affbc8f1c3/algo.py).
  [Training evidence](evaluation_results/cycle-26-train.json).
  [Narrative](full_log.md#cycle-26--milder-trust-radius-shrink-after-poor-steps).

## sella-mis-dihedral-wd: larger torsion steps via MaxInternalStep wd=2/3
Status: deferred

**Hypothesis:** Cycle 8’s cost win came from larger torsions; its energy misses came from larger stretches. Weighted MIS with `wd=2/3` caps |s_d| at 0.15 and |s_bond| at 0.1.

**Outcome and uncertainty:** Cycle 27 invalid (C=0.99389, thiols +0.564 kcal). Cycle 28 delayed `wd` lost the cost win. Cycle 29 fragment-gated `wd=2/3` is the best: C=0.99323, R=0.999999888, dimers match champion, V fails only because 252618428 is +0.0026 kcal. Cycle 30 `wd=0.8` still hops 252618428. Cycle 34 early-only wd keeps C=0.99364 and still hops 252618428 — the miss is in the first four torsion-enlarged steps. Magnitude/timing of `wd` cannot separate that well from the cost win.

**Reason to revisit:** Combine cycle 29 with a well-preserving finish (GDIIS, Cartesian RMS cap on the linearized step, or energy-aware interpolation) rather than another `wd` value.

**Next experiment:** Do not retune `wd`. Revisit by adding a Cartesian RMS/max-atom cap on cycle 29’s connected-only torsion steps, or a GDIIS polish, evaluated as a new idea.

**Attempts:**
- Cycle 27; candidate `3bebc92e31fc377d30658bd92fba70e9a7bd72d2`; champion `7fa53b9aa21691454d13be8e506997cdffdbed1f`; decision invalid.
  [Implementation](ideas/sella-mis-dihedral-wd/3bebc92e31fc377d30658bd92fba70e9a7bd72d2/algo.py).
  [Training evidence](evaluation_results/cycle-27-train.json).
  [Narrative](full_log.md#cycle-27--dihedral-only-maxinternalstep-weight).
- Cycle 28; candidate `4c422305738c152d3d32f886fc038c8a93cefc1d`; champion `7fa53b9aa21691454d13be8e506997cdffdbed1f`; decision invalid.
  [Implementation](ideas/sella-mis-dihedral-wd/4c422305738c152d3d32f886fc038c8a93cefc1d/algo.py).
  [Training evidence](evaluation_results/cycle-28-train.json).
  [Narrative](full_log.md#cycle-28--delayed-dihedral-maxinternalstep-weight).
- Cycle 29; candidate `0938d3d50cb312294295acc6ec62e38bd4319ea8`; champion `7fa53b9aa21691454d13be8e506997cdffdbed1f`; decision invalid.
  [Implementation](ideas/sella-mis-dihedral-wd/0938d3d50cb312294295acc6ec62e38bd4319ea8/algo.py).
  [Training evidence](evaluation_results/cycle-29-train.json).
  [Narrative](full_log.md#cycle-29--fragment-gated-dihedral-maxinternalstep-weight).
- Cycle 30; candidate `766ee776d93ae852a354e79de1e03d5e965f31e6`; champion `7fa53b9aa21691454d13be8e506997cdffdbed1f`; decision invalid.
  [Implementation](ideas/sella-mis-dihedral-wd/766ee776d93ae852a354e79de1e03d5e965f31e6/algo.py).
  [Training evidence](evaluation_results/cycle-30-train.json).
  [Narrative](full_log.md#cycle-30--milder-fragment-gated-dihedral-weight).
- Cycle 34; candidate `f654b1feed84fd0d5b8e13946dea31de6adc9828`; champion `7fa53b9aa21691454d13be8e506997cdffdbed1f`; decision invalid.
  [Implementation](ideas/sella-mis-dihedral-wd/f654b1feed84fd0d5b8e13946dea31de6adc9828/algo.py).
  [Training evidence](evaluation_results/cycle-34-train.json).
  [Narrative](full_log.md#cycle-34--early-only-fragment-gated-dihedral-weight).

## sella-bfgs-auto: PD-preserving BFGS Hessian updates for minima
Status: deferred

**Hypothesis:** Replacing always-on TS-BFGS with `BFGS_auto` lets minima use standard BFGS while H and SᵀY are positive definite, taking better Newton steps without changing the trust radius.

**Outcome and uncertainty:** Cycle 31 invalid: C=1.005, R=0.99999958. 447/469 molecules unchanged cost. Cycle 94 late switch to standard BFGS at nsteps==20 (keep accumulated B) invalid: C=1.017, R=0.9999992, 252618428 hop, 160853090 88→155. TS-BFGS is required throughout, not only early.

**Reason to revisit:** Only if a later method makes the Hessian indefinite too often.

**Next experiment:** Do not swap TS-BFGS for BFGS, BFGS_auto, DFP, PSB, or SR1.

**Attempts:**
- Cycle 31; candidate `3b1c040aaf11ee658d80b08516c47742c208732b`; champion `7fa53b9aa21691454d13be8e506997cdffdbed1f`; decision invalid.
  [Implementation](ideas/sella-bfgs-auto/3b1c040aaf11ee658d80b08516c47742c208732b/algo.py).
  [Training evidence](evaluation_results/cycle-31-train.json).
  [Narrative](full_log.md#cycle-31--bfgs_auto-hessian-updates-for-minima).
- Cycle 94; candidate `14a94596030b651378be3d2895d2dafcf09b5fcb`; champion `453a761dd3eace02562f81dc72920ac1e38526cc`; decision invalid.
  [Implementation](ideas/sella-bfgs-auto/14a94596030b651378be3d2895d2dafcf09b5fcb/algo.py).
  [Training evidence](evaluation_results/cycle-94-train.json).
  [Narrative](full_log.md#cycle-94--bfgs-hessian-updates-after-20-connected-steps).

## sella-dihedral-wd-cart-cap: Cartesian max-atom cap on cycle-29 torsion steps
Status: closed

**Hypothesis:** Cycle 29’s 252618428 miss is a large Cartesian torsion. Capping linearized max-atom displacement at 0.15 Å on connected `wd=2/3` steps should keep that well.

**Outcome and uncertainty:** Cycle 32 invalid: C=1.039, +38 kcal on 135043047. Cycle 146 always-on connected Cartesian inside MIS: same +38 kcal hop. Cycle 147 Cartesian@20: energy-safe, Δ=−3.52e-5 (long tails cheapen; medium jobs inflate). Cycle 148 Cartesian@50: Δ=−1.13e-4 (venetoclax save lived in 20–50). Cycle 149 ρ-gated Cartesian@20: paliperidone ODE timeout. Mixing Cartesian and QN via ρ broke the geodesic.

**Reason to revisit:** No remaining nsteps/ρ interpolation is energy-and-ODE-safe with Δ≥1e-4. Do not add ODE-halve as a fourth Cartesian candidate.

**Next experiment:** Do not put Cartesian max-atom trust inside MaxInternalStep.cons.

**Attempts:**
- Cycle 32; candidate `4f5c571a1fa28cd7b6a6c1e3297c3369c755a95b`; champion `7fa53b9aa21691454d13be8e506997cdffdbed1f`; decision invalid.
  [Implementation](ideas/sella-dihedral-wd-cart-cap/4f5c571a1fa28cd7b6a6c1e3297c3369c755a95b/algo.py).
  [Training evidence](evaluation_results/cycle-32-train.json).
  [Narrative](full_log.md#cycle-32--cartesian-cap-on-fragment-gated-dihedral-steps).
- Cycle 146; candidate `44a884c2d7be42d315834c7f30c70aafba960041`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision invalid.
  [Implementation](ideas/sella-dihedral-wd-cart-cap/44a884c2d7be42d315834c7f30c70aafba960041/algo.py).
  [Training evidence](evaluation_results/cycle-146-train.json).
  [Narrative](full_log.md#cycle-146--connected-cartesian-max-atom-trust-inside-mis).
- Cycle 147; candidate `1b437e93b11a42f57acda7a6ccde9959b7503a0b`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision discard.
  [Implementation](ideas/sella-dihedral-wd-cart-cap/1b437e93b11a42f57acda7a6ccde9959b7503a0b/algo.py).
  [Training evidence](evaluation_results/cycle-147-train.json).
  [Narrative](full_log.md#cycle-147--cartesian-max-atom-trust-after-20-connected-steps).
- Cycle 148; candidate `dae1afe2abe56edb291ab54e5b75ac0928455111`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision discard.
  [Implementation](ideas/sella-dihedral-wd-cart-cap/dae1afe2abe56edb291ab54e5b75ac0928455111/algo.py).
  [Training evidence](evaluation_results/cycle-148-train.json).
  [Narrative](full_log.md#cycle-148--cartesian-max-atom-trust-after-50-connected-steps).
- Cycle 149; candidate `f6e7f3145eca849f78d5e2625070153f0173e893`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision invalid.
  [Implementation](ideas/sella-dihedral-wd-cart-cap/f6e7f3145eca849f78d5e2625070153f0173e893/algo.py).
  [Training evidence](evaluation_results/cycle-149-train.json).
  [Narrative](full_log.md#cycle-149--ρ-gated-cartesian-max-atom-trust-after-20-steps).

## sella-stiffer-bond-h0: doubled Lindh stretch force constants
Status: deferred

**Hypothesis:** 2× `_h0_bond` shrinks predicted stretches so more of the 0.1 MIS budget goes to torsions without raising the stretch cap.

**Outcome and uncertainty:** Cycle 33 invalid: C=1.044, +38 kcal on 135043047, ammoniums–ketones rel_energy 0.682. Guess-Hessian scaling is as unsafe as RFO here.

**Reason to revisit:** Not as a global H0 scale. A fragment-gated or bond-only scale on connected molecules still risks 135043047.

**Next experiment:** Do not scale `_h0_bond`.

**Attempts:**
- Cycle 33; candidate `1a398db5a665c7464789074132feaddf934846d3`; champion `7fa53b9aa21691454d13be8e506997cdffdbed1f`; decision invalid.
  [Implementation](ideas/sella-stiffer-bond-h0/1a398db5a665c7464789074132feaddf934846d3/algo.py).
  [Training evidence](evaluation_results/cycle-33-train.json).
  [Narrative](full_log.md#cycle-33--stiffer-lindh-bond-hessian-guess).

## sella-rho-gated-delta015: 0.15 floor only after a well-predicted step
Status: deferred

**Hypothesis:** Cycle 22’s valid energy miss comes from flooring δ at 0.15 after a poor ρ. Require the expansion window before boosting.

**Outcome and uncertainty:** Cycle 35 invalid: C=0.99922 (cost win) but R=0.999999877. The few molecules that fire the gated boost hop (venetoclax, 135264879). Unconditional cycle 22 remains better on train energy.

**Reason to revisit:** Only if combined with an energy-restoring finish, not as a ρ gate on the floor.

**Next experiment:** Do not ρ-gate the 0.15 floor.

**Attempts:**
- Cycle 35; candidate `1e9996a9a8c94b8bf0fc8bab7433796ea5d03635`; champion `7fa53b9aa21691454d13be8e506997cdffdbed1f`; decision invalid.
  [Implementation](ideas/sella-rho-gated-delta015/1e9996a9a8c94b8bf0fc8bab7433796ea5d03635/algo.py).
  [Training evidence](evaluation_results/cycle-35-train.json).
  [Narrative](full_log.md#cycle-35--ρ-gated-fragment-gated-015-trust-floor).

## sella-gdiis: Pulay GDIIS replacing quasi-Newton steps
Status: incorporated

**Hypothesis:** After three (x, g) iterates, GDIIS finds a lower-residual interpolated geometry and reduces tail force calls.

**Outcome and uncertainty:** Cycle 36 invalid. Cycle 97 hops. Cycle 98 no-op. Cycle 117/118 ungated milder interpolation: train keep, valid extras. Cycle 119 cosine 0.95 discard. Cycle 120 **keep**: two-point interpolation, cosine 0.90, `c_i≥0`, `||s||≤QN`, nsteps≥20, and `1/ρ_inc < ρ < ρ_inc`. Train Δ=+1.64e-4 (103954262 26→24); valid Δ=+2.69e-4 (123107365 32→28). Cycle 132 nsteps≥12 discard (Δ=0; only 135296936 energy wiggle). Cycle 133 drop ρ upper bound bit-identical. Cycle 137 cosine 0.80 discard (104046121 33→35). Cycle 150 always-on Newton residuals: discard, Δ=−1.40e-4; 103954262 24→26 and 160853090 88→87. Cycle 151 Newton fallback: discard, Δ=+2.40e-5; only 160853090 88→87. Cycle 160 current+lowest-residual earlier: discard, Δ=−2.65e-5; 103954262 24→26. Cycle 165 C2-DIIS 3-point fallback: discard, Δ=−7.19e-5; only 160853090 88→91. Cycle 188 nsteps≥15: discard, Δ=0, bit-identical. Cycle 190 rms(s)<0.0025 before 20: discard, Δ=0, energy-only 135296936. Cycle 333 delay connected GDIIS to 40 on 30≤n<80: discard, Δ=0, bit-identical. Cycle 340 skip connected GDIIS at n=18: discard, Δ=0, bit-identical. Cycle 370 retest vs `7140449`: discard, Δ=0, n_steps and energies bit-identical.

**Reason to revisit:** Cycle 423 cosine 0.85 n<18 bit-identical. Cycle 424 skip ρ n<18 non_generalizable (valid extra n=10). Cycle 425 skip ρ 12–18 non_generalizable (train −4, valid bit-identical). Close ρ-skip size cuts.

**Next experiment:** Do not interpolate 14–18 ρ-skip. N-oxide n<18 has no valid analog. See Sella 2.6.0 pivoting-QR guess-Hessian projector.

**Attempts:**
- Cycle 425; candidate `5bc03375762ec950634157ae837772250afcfb99`; champion `d2f7137d0609977f7ebd973704e68b8f419f8dad`; decision non_generalizable.
  [Implementation](ideas/sella-gdiis/5bc03375762ec950634157ae837772250afcfb99/algo.py).
  [Training evidence](evaluation_results/cycle-425-train.json).
  [Validation evidence](evaluation_results/cycle-425-valid.json).
  [Narrative](full_log.md#cycle-425--skip-gdiis-ρ-window-on-connected-12n_atoms18).
- Cycle 424; candidate `84ec7d2c08f1be44af8b39ddfb6808eefdc91d3a`; champion `d2f7137d0609977f7ebd973704e68b8f419f8dad`; decision non_generalizable.
  [Implementation](ideas/sella-gdiis/84ec7d2c08f1be44af8b39ddfb6808eefdc91d3a/algo.py).
  [Training evidence](evaluation_results/cycle-424-train.json).
  [Validation evidence](evaluation_results/cycle-424-valid.json).
  [Narrative](full_log.md#cycle-424--skip-gdiis-ρ-window-on-connected-n_atoms18).
- Cycle 36; candidate `4d95e1cdec73172aeee40dd63852f04d22b69032`; champion `7fa53b9aa21691454d13be8e506997cdffdbed1f`; decision invalid.
  [Implementation](ideas/sella-gdiis/4d95e1cdec73172aeee40dd63852f04d22b69032/algo.py).
  [Training evidence](evaluation_results/cycle-36-train.json).
  [Narrative](full_log.md#cycle-36--pulay-gdiis-steps-on-champion-internals).
- Cycle 97; candidate `be1725a8e78a2b876d9cee498fab2a83fbe9a015`; champion `453a761dd3eace02562f81dc72920ac1e38526cc`; decision invalid.
  [Implementation](ideas/sella-gdiis/be1725a8e78a2b876d9cee498fab2a83fbe9a015/algo.py).
  [Training evidence](evaluation_results/cycle-97-train.json).
  [Narrative](full_log.md#cycle-97--controlled-gdiis-after-20-connected-steps).
- Cycle 98; candidate `b1e8544f2f34674afc7d40cea127f26a54cc0343`; champion `453a761dd3eace02562f81dc72920ac1e38526cc`; decision discard.
  [Implementation](ideas/sella-gdiis/b1e8544f2f34674afc7d40cea127f26a54cc0343/algo.py).
  [Training evidence](evaluation_results/cycle-98-train.json).
  [Narrative](full_log.md#cycle-98--interpolation-only-gdiis-no-longer-than-qn).
- Cycle 117; candidate `16acbc6789857d774a64f78d1d07dac864a5d5da`; champion `73c4e88596f23f25a9b520c6158ac5764251c9c2`; decision non_generalizable.
  [Implementation](ideas/sella-gdiis/16acbc6789857d774a64f78d1d07dac864a5d5da/algo.py).
  [Training evidence](evaluation_results/cycle-117-train.json).
  [Validation evidence](evaluation_results/cycle-117-valid.json).
  [Narrative](full_log.md#cycle-117--milder-interpolation-only-gdiis-on-connected-tails).
- Cycle 118; candidate `d400e95d98a8c479c364515409d9bc4a4123559c`; champion `73c4e88596f23f25a9b520c6158ac5764251c9c2`; decision non_generalizable.
  [Implementation](ideas/sella-gdiis/d400e95d98a8c479c364515409d9bc4a4123559c/algo.py).
  [Training evidence](evaluation_results/cycle-118-train.json).
  [Validation evidence](evaluation_results/cycle-118-valid.json).
  [Narrative](full_log.md#cycle-118--two-point-interpolation-gdiis-on-connected-tails).
- Cycle 119; candidate `d606f97c4d5765b2e4824e1b23139ae2a6afe03b`; champion `73c4e88596f23f25a9b520c6158ac5764251c9c2`; decision discard.
  [Implementation](ideas/sella-gdiis/d606f97c4d5765b2e4824e1b23139ae2a6afe03b/algo.py).
  [Training evidence](evaluation_results/cycle-119-train.json).
  [Narrative](full_log.md#cycle-119--two-point-interpolation-gdiis-with-cosine-095).
- Cycle 120; candidate `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; champion `73c4e88596f23f25a9b520c6158ac5764251c9c2`; decision keep.
  [Training evidence](evaluation_results/cycle-120-train.json).
  [Validation evidence](evaluation_results/cycle-120-valid.json).
  [Narrative](full_log.md#cycle-120--ρ-gated-two-point-interpolation-gdiis).
- Cycle 121; candidate `23547d8a16eeec98c2910a7fa5ce2b35605a1236`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision discard.
  [Implementation](ideas/sella-gdiis/23547d8a16eeec98c2910a7fa5ce2b35605a1236/algo.py).
  [Training evidence](evaluation_results/cycle-121-train.json).
  [Narrative](full_log.md#cycle-121--ρ-gated-24-point-interpolation-gdiis).
- Cycle 124; candidate `e19b7d7525e37728623110137260c7df58ae39b9`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision discard.
  [Implementation](ideas/sella-gdiis/e19b7d7525e37728623110137260c7df58ae39b9/algo.py).
  [Training evidence](evaluation_results/cycle-124-train.json).
  [Narrative](full_log.md#cycle-124--ρ-gated-two-point-gdiis-on-dimers).
- Cycle 125; candidate `d978dd2ac4d948c4b67aab3ed7bede687e0efcd5`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision non_generalizable.
  [Implementation](ideas/sella-gdiis/d978dd2ac4d948c4b67aab3ed7bede687e0efcd5/algo.py).
  [Training evidence](evaluation_results/cycle-125-train.json).
  [Validation evidence](evaluation_results/cycle-125-valid.json).
  [Narrative](full_log.md#cycle-125--connected-δ-floor-005-plus-dimer-gdiis).
- Cycle 132; candidate `bf02118bc567ef919aae25d99ad6ef7d10c209ea`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision discard.
  [Implementation](ideas/sella-gdiis/bf02118bc567ef919aae25d99ad6ef7d10c209ea/algo.py).
  [Training evidence](evaluation_results/cycle-132-train.json).
  [Narrative](full_log.md#cycle-132--earlier-ρ-gated-gdiis-after-12-connected-steps).
- Cycle 133; candidate `410d10145983a13788913dd5d01acaa424f62213`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision discard.
  [Implementation](ideas/sella-gdiis/410d10145983a13788913dd5d01acaa424f62213/algo.py).
  [Training evidence](evaluation_results/cycle-133-train.json).
  [Narrative](full_log.md#cycle-133--gdiis-after-downhill-or-well-predicted-ρ).
- Cycle 137; candidate `4a9bc2964515bd7de00e7d9bc942fe5e9784b652`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision discard.
  [Implementation](ideas/sella-gdiis/4a9bc2964515bd7de00e7d9bc942fe5e9784b652/algo.py).
  [Training evidence](evaluation_results/cycle-137-train.json).
  [Narrative](full_log.md#cycle-137--ρ-gated-gdiis-cosine-080).
- Cycle 150; candidate `741620651fb5ff0782b4fd60a1d43662a1c1326d`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision discard.
  [Implementation](ideas/sella-gdiis/741620651fb5ff0782b4fd60a1d43662a1c1326d/algo.py).
  [Training evidence](evaluation_results/cycle-150-train.json).
  [Narrative](full_log.md#cycle-150--newton-metric-gdiis-residuals-on-connected-tails).
- Cycle 151; candidate `e8b5b53c33af35fece02b2022a18b8ec07014965`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision discard.
  [Implementation](ideas/sella-gdiis/e8b5b53c33af35fece02b2022a18b8ec07014965/algo.py).
  [Training evidence](evaluation_results/cycle-151-train.json).
  [Narrative](full_log.md#cycle-151--newton-metric-gdiis-as-fallback-after-gradient-reject).
- Cycle 160; candidate `f9cd5b03ba597c2c3188b34a9d95d8baec02ab26`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision discard.
  [Implementation](ideas/sella-gdiis/f9cd5b03ba597c2c3188b34a9d95d8baec02ab26/algo.py).
  [Training evidence](evaluation_results/cycle-160-train.json).
  [Narrative](full_log.md#cycle-160--two-lowest-residual-gdiis-points-on-connected-tails).
- Cycle 165; candidate `ad6721c7c62bde1b0c079b3f50dbc00441757122`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision discard.
  [Implementation](ideas/sella-gdiis/ad6721c7c62bde1b0c079b3f50dbc00441757122/algo.py).
  [Training evidence](evaluation_results/cycle-165-train.json).
  [Narrative](full_log.md#cycle-165--c2-diis-three-point-fallback-after-two-point-gdiis).
- Cycle 188; candidate `dd0bff317c40d8a59437cdbdebd11628d1cd75eb`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-gdiis/dd0bff317c40d8a59437cdbdebd11628d1cd75eb/algo.py).
  [Training evidence](evaluation_results/cycle-188-train.json).
  [Narrative](full_log.md#cycle-188--connected-gdiis-after-15-steps).
- Cycle 190; candidate `4d9ce1cb9cc063a3e8603ed9450f4d8dd0abc7b8`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-gdiis/4d9ce1cb9cc063a3e8603ed9450f4d8dd0abc7b8/algo.py).
  [Training evidence](evaluation_results/cycle-190-train.json).
  [Narrative](full_log.md#cycle-190--connected-gdiis-when-rmss--00025).
- Cycle 400; candidate `e48250fa433827e5c1741d264a9d9908c39499eb`; champion `8e55a3dd9c0982bebda9429fb00fea1ae261d650`; decision discard.
  [Implementation](ideas/sella-medium-gdiis-cos-085/e48250fa433827e5c1741d264a9d9908c39499eb/algo.py).
  [Training evidence](evaluation_results/cycle-400-train.json).
  [Narrative](full_log.md#cycle-400--gdiis-cosine-085-on-connected-12n30).
- Cycle 370; candidate `54d196b3310dbfe79c87e89d5f5a84d10a3438e5`; champion `7140449e9cbddb6f6589ec34fdac86f88dc413e1`; decision discard.
  [Implementation](ideas/sella-gdiis/54d196b3310dbfe79c87e89d5f5a84d10a3438e5/algo.py).
  [Training evidence](evaluation_results/cycle-370-train.json).
  [Narrative](full_log.md#cycle-370--skip-gdiis-on-connected-n_atoms18).
- Cycle 340; candidate `b53e9e27aaffe5ffad2d3439b3bbb9f865f5a252`; champion `2697c67f7818bfcd2955251e89152f51920c6aa0`; decision discard.
  [Implementation](ideas/sella-gdiis/b53e9e27aaffe5ffad2d3439b3bbb9f865f5a252/algo.py).
  [Training evidence](evaluation_results/cycle-340-train.json).
  [Narrative](full_log.md#cycle-340--skip-connected-gdiis-at-n_atoms18).
- Cycle 333; candidate `54366b3ae42f56a4c49a010d5f74909c1c9d5133`; champion `362a4dd6db5ca11eea73f992d86141a50908bd75`; decision discard.
  [Implementation](ideas/sella-gdiis/54366b3ae42f56a4c49a010d5f74909c1c9d5133/algo.py).
  [Training evidence](evaluation_results/cycle-333-train.json).
  [Narrative](full_log.md#cycle-333--delay-connected-gdiis-until-40-steps-on-30n_atoms80).

## sella-pred-uphill-halve: halve QN steps with predicted energy increase
Status: deferred

**Hypothesis:** Halving s while df_pred > 0 avoids uphill evals without a post-eval restore.

**Outcome and uncertainty:** Cycle 37 discard: bit-identical to the champion. No predicted-uphill MIS-QN steps on this set.

**Reason to revisit:** Pair with a stepper that does propose uphill steps (RFO did, but hopped +38 kcal).

**Next experiment:** Do not add predicted-uphill halving to the champion QN path.

**Attempts:**
- Cycle 37; candidate `8aa9602a1af4d2728f7872902618bbe9fbeb4439`; champion `7fa53b9aa21691454d13be8e506997cdffdbed1f`; decision discard.
  [Implementation](ideas/sella-pred-uphill-halve/8aa9602a1af4d2728f7872902618bbe9fbeb4439/algo.py).
  [Training evidence](evaluation_results/cycle-37-train.json).
  [Narrative](full_log.md#cycle-37--predicted-uphill-qn-step-halving).

## sella-iterative-stepper: iterative Cartesian realization at champion trust
Status: deferred

**Hypothesis:** `iterative_stepper=1` at δ=0.1 realizes internals without the ODE geodesic and may be cheaper or more stable than LSODA.

**Outcome and uncertainty:** Cycle 38 global discard (C=1.002). Cycle 62 dimer-only discard (C=1.001, amines 99→122, acids–alkanes +0.20 kcal). Not a cost win.

**Reason to revisit:** Only as an ODE-timeout fallback, not to reduce force calls.

**Next experiment:** Do not switch champion or dimer runs to iterative_stepper=1 for cost.

**Attempts:**
- Cycle 38; candidate `a623690b583155ac38cd083cdee6b73277beda70`; champion `7fa53b9aa21691454d13be8e506997cdffdbed1f`; decision discard.
  [Implementation](ideas/sella-iterative-stepper/a623690b583155ac38cd083cdee6b73277beda70/algo.py).
  [Training evidence](evaluation_results/cycle-38-train.json).
  [Narrative](full_log.md#cycle-38--iterative-cartesian-realization-of-internal-steps).
- Cycle 62; candidate `a80334e6450017199fb60dd987554568205175e5`; champion `e21f1d77e70cd5198ac760558e01de1d3d53139e`; decision discard.
  [Implementation](ideas/sella-iterative-stepper/a80334e6450017199fb60dd987554568205175e5/algo.py).
  [Training evidence](evaluation_results/cycle-62-train.json).
  [Narrative](full_log.md#cycle-62--dimer-only-iterative-cartesian-realization-of-internal-steps).

## sella-mis-angle-wa: larger angle steps via MaxInternalStep wa on connected molecules
Status: incorporated

**Hypothesis:** Downweighting angles (`wa=2/3`) on connected molecules cheapens organics without the torsion hop that killed cycle 29 on 252618428.

**Outcome and uncertainty:** Cycle 39 non_generalizable. Cycle 40 `wa=0.8` discard. Cycle 41 `wa=0.75` **keep** (current champion). Cycle 42 `wa=0.72` hopped 135255884. Cycle 43 `wa=0.74` valid cost regression. Cycle 44 delayed `wa=0.72` bit-identical. Cycle 45 `wa=0.72` plus Cartesian-in-cons invalid (+38 kcal on 135043047). Further static/delayed/Cartesian `wa` is exhausted.

**Reason to revisit:** Only if a later method already has R=1 on 135255884 and needs a slightly larger bend cap with a non-Cartesian limiter.

**Next experiment:** Do not lower `wa` below 0.75 or add linearized Cartesian to MIS. Cycle 73 late `wa=1` after 30 steps was bit-identical. Cycle 79 tests `wa=1` on the first Newton step only.

**Attempts:**
- Cycle 39; candidate `1e37f5eddd61993eae9960f24df3f2c8459134a1`; champion `7fa53b9aa21691454d13be8e506997cdffdbed1f`; decision non_generalizable.
  [Implementation](ideas/sella-mis-angle-wa/1e37f5eddd61993eae9960f24df3f2c8459134a1/algo.py).
  [Training evidence](evaluation_results/cycle-39-train.json).
  [Validation evidence](evaluation_results/cycle-39-valid.json).
  [Narrative](full_log.md#cycle-39--fragment-gated-maxinternalstep-angle-weight).
- Cycle 40; candidate `c610ace14492bbdde58282414bf8ed54fecee9b6`; champion `7fa53b9aa21691454d13be8e506997cdffdbed1f`; decision discard.
  [Implementation](ideas/sella-mis-angle-wa/c610ace14492bbdde58282414bf8ed54fecee9b6/algo.py).
  [Training evidence](evaluation_results/cycle-40-train.json).
  [Narrative](full_log.md#cycle-40--milder-fragment-gated-angle-weight).
- Cycle 41; candidate `baa53f93e895f5dafb46eb48ef65fe583729f574`; champion `7fa53b9aa21691454d13be8e506997cdffdbed1f`; decision keep.
  [Implementation](ideas/sella-mis-angle-wa/baa53f93e895f5dafb46eb48ef65fe583729f574/algo.py).
  [Training evidence](evaluation_results/cycle-41-train.json).
  [Validation evidence](evaluation_results/cycle-41-valid.json).
  [Narrative](full_log.md#cycle-41--intermediate-fragment-gated-angle-weight).
- Cycle 42; candidate `58192d16a213b44b8dc4f5ada022e172269f12d1`; champion `baa53f93e895f5dafb46eb48ef65fe583729f574`; decision non_generalizable.
  [Implementation](ideas/sella-mis-angle-wa/58192d16a213b44b8dc4f5ada022e172269f12d1/algo.py).
  [Training evidence](evaluation_results/cycle-42-train.json).
  [Validation evidence](evaluation_results/cycle-42-valid.json).
  [Narrative](full_log.md#cycle-42--slightly-larger-fragment-gated-angle-steps-on-the-new-champion).
- Cycle 43; candidate `471bddf0c52b3f04e37fee017601a50ae5bceafe`; champion `baa53f93e895f5dafb46eb48ef65fe583729f574`; decision non_generalizable.
  [Implementation](ideas/sella-mis-angle-wa/471bddf0c52b3f04e37fee017601a50ae5bceafe/algo.py).
  [Training evidence](evaluation_results/cycle-43-train.json).
  [Validation evidence](evaluation_results/cycle-43-valid.json).
  [Narrative](full_log.md#cycle-43--fragment-gated-angle-weight-between-keep-and-hop).
- Cycle 44; candidate `96df0509ac8a40b20907f066623d3716c1ad0f52`; champion `baa53f93e895f5dafb46eb48ef65fe583729f574`; decision discard.
  [Training evidence](evaluation_results/cycle-44-train.json).
  [Narrative](full_log.md#cycle-44--delayed-wa072-after-16-champion-angle-capped-steps).
- Cycle 45; candidate `f602bbfd64872f74227101d3a0487fa21a05af70`; champion `baa53f93e895f5dafb46eb48ef65fe583729f574`; decision invalid.
  [Training evidence](evaluation_results/cycle-45-train.json).
  [Narrative](full_log.md#cycle-45--wa072-with-linearized-cartesian-max-atom-in-mis).

## sella-late-wa-restore: default angle caps after 30 connected steps
Status: deferred

**Hypothesis:** Always-on `wa=0.75` adds tail calls on 135043047/venetoclax. Restoring `wa=1` after 30 steps should tighten late bends without touching jobs that finish earlier.

**Outcome and uncertainty:** Cycle 73 discard, bit-identical. Cycle 89 on the cycle-88 champion: only 135043047 45→44, Δ=+5.33e-5. Late wa is still not the extra-call mechanism even with δ=0.15.

**Reason to revisit:** Only if a later method makes late angle steps hit the 0.15 cap.

**Next experiment:** Do not move the wa=1 switch. See `σ_inc=1.18` with the 0.15 floor.

**Attempts:**
- Cycle 73; candidate `d79f179f48e7d56278ac80480d0b0b4a7d5ca284`; champion `a9c931a62562b4b5d904a2b373be7b3933200aa5`; decision discard.
  [Implementation](ideas/sella-late-wa-restore/d79f179f48e7d56278ac80480d0b0b4a7d5ca284/algo.py).
  [Training evidence](evaluation_results/cycle-73-train.json).
  [Narrative](full_log.md#cycle-73--restore-default-angle-caps-after-30-connected-steps).
- Cycle 89; candidate `cc2497b195da945a9c59dc8a80110ea2bf8f6921`; champion `453a761dd3eace02562f81dc72920ac1e38526cc`; decision discard.
  [Implementation](ideas/sella-late-wa-restore/cc2497b195da945a9c59dc8a80110ea2bf8f6921/algo.py).
  [Training evidence](evaluation_results/cycle-89-train.json).
  [Narrative](full_log.md#cycle-89--default-angle-caps-after-the-connected-015-trust-floor).

## sella-first-step-wa: default angle caps on the first connected Newton step
Status: deferred

**Hypothesis:** Extra connected cost vs original Sella is in the first 30 steps (cycle 73). Keeping `wa=1` for `nsteps==0` then `wa=0.75` matches the original first QN step.

**Outcome and uncertainty:** Cycle 79 discard. Energy-ok, Δ=−7.43e-4. venetoclax/135043047 unchanged; 252089162 +1; net +7 calls. First-step wa is not the 5–7 extra-call mechanism.

**Reason to revisit:** Only if a later method needs the original first Newton step as a well-preserving prefix.

**Next experiment:** Do not delay `wa=0.75` by more early steps.

**Attempts:**
- Cycle 79; candidate `0d233527c3a2e531c1788fc73eae209b3e79db94`; champion `a9c931a62562b4b5d904a2b373be7b3933200aa5`; decision discard.
  [Implementation](ideas/sella-first-step-wa/0d233527c3a2e531c1788fc73eae209b3e79db94/algo.py).
  [Training evidence](evaluation_results/cycle-79-train.json).
  [Narrative](full_log.md#cycle-79--first-connected-newton-step-keeps-default-angle-caps).

## sella-swart-hessian: Swart–Bickelhaupt model Hessian on connected molecules
Status: deferred

**Hypothesis:** Sella’s Fischer–Almlöf H0 decays too fast off equilibrium. Swart ρ=exp(1−r/rcov) is ORCA’s recommended guess for mixed strong/weak coordinates and can cheapen large organics without touching dimer connecting Hessians (cycles 60–61).

**Outcome and uncertainty:** Cycle 80 invalid. 135043047 +38 kcal; venetoclax +1.47. Cycle 99 dihedral-only Swart also invalid: 135043047 +38 kcal at 45 calls; venetoclax +1.38; paliperidone +1.65. Torsion H0 is sufficient for the canary hop.

**Reason to revisit:** Not as a drop-in or dihedral-only H0. Do not interpolate Swart k.

**Next experiment:** Do not swap connected H0 (full or hybrid).

**Attempts:**
- Cycle 80; candidate `77db5d2c2ca70982919009359178c2f871154ac1`; champion `a9c931a62562b4b5d904a2b373be7b3933200aa5`; decision invalid.
  [Implementation](ideas/sella-swart-hessian/77db5d2c2ca70982919009359178c2f871154ac1/algo.py).
  [Training evidence](evaluation_results/cycle-80-train.json).
  [Narrative](full_log.md#cycle-80--swart-model-hessian-on-connected-molecules).
- Cycle 99; candidate `8f6de5a65e9114fc966392c12f85b00e1286cbbb`; champion `453a761dd3eace02562f81dc72920ac1e38526cc`; decision invalid.
  [Implementation](ideas/sella-swart-hessian/8f6de5a65e9114fc966392c12f85b00e1286cbbb/algo.py).
  [Training evidence](evaluation_results/cycle-99-train.json).
  [Narrative](full_log.md#cycle-99--swart-dihedral-hessian-guess-on-connected-molecules).

## sella-powell-damp: Powell-damped TS-BFGS on connected molecules
Status: deferred

**Hypothesis:** Powell mixing of y with Bs when s·y < 0.2 s·B·s keeps TS-BFGS from absorbing noisy curvature on long connected jobs.

**Outcome and uncertainty:** Cycle 100 discard. Energy-ok; C=1.00003 (Δ=−1.23e-3). venetoclax/paliperidone/160853090 inflate. The 0.2 test fires on useful updates.

**Reason to revisit:** Only with a stricter s·y<0 skip (cycle 101), not a milder 0.2 mix.

**Next experiment:** Cycle 101 skip updates when s·y<0.

**Attempts:**
- Cycle 100; candidate `e37f649c149835e3ee3d4752aeabbc5feef390e8`; champion `453a761dd3eace02562f81dc72920ac1e38526cc`; decision discard.
  [Implementation](ideas/sella-powell-damp/e37f649c149835e3ee3d4752aeabbc5feef390e8/algo.py).
  [Training evidence](evaluation_results/cycle-100-train.json).
  [Narrative](full_log.md#cycle-100--powell-damped-ts-bfgs-on-connected-molecules).

## sella-dummy-dihedral-h0: softer dummy-atom dihedral guess Hessian
Status: incorporated

**Hypothesis:** Dummy-atom dihedrals at 2-coordinate linear centers use a hard 0.5 Ha H0. Softening to 0.25 Ha cheapens dummy-linear Newton steps without changing which coordinates exist (unlike atol).

**Outcome and uncertainty:** Cycle 103 **keep** vs cycle 88: train Δ=+5.54e-3, valid Δ=+3.93e-3, both energy-safe. Cycle 104 dropped the connected gate: invalid, ammoniums–ketones hop (R=0.682). Cycle 105 connected 0.20 Ha: discard, dummy-linear canaries cheaper but 104046121 33→55 undoes the win.

**Reason to revisit:** Not as a global dummy-H0 scale. Cycle 183 dimer H0=0.25: invalid, ammoniums–ketones 29→12 (+1.19 kcal, R=0.682), same hop class as cycle 104.

**Next experiment:** Do not interpolate dummy-dihedral H0 (cycle 426 0.18 n<18 extras carboxylates–thiols). Do not apply 0.25 Ha on dimers. Dummy-involving *angle* H0 (cycle 107) is energy-safe but cost-negative; do not interpolate it.

**Attempts:**
- Cycle 103; candidate `73c4e88596f23f25a9b520c6158ac5764251c9c2`; champion `453a761dd3eace02562f81dc72920ac1e38526cc`; decision keep.
  [Training evidence](evaluation_results/cycle-103-train.json).
  [Validation evidence](evaluation_results/cycle-103-valid.json).
  [Narrative](full_log.md#cycle-103--softer-dummy-atom-dihedral-hessian-on-connected-molecules).
- Cycle 104; candidate `9c908fc0b5a96787174fb6319065b74ad16cf70e`; champion `73c4e88596f23f25a9b520c6158ac5764251c9c2`; decision invalid.
  [Training evidence](evaluation_results/cycle-104-train.json).
  [Narrative](full_log.md#cycle-104--dummy-atom-dihedral-hessian-025-ha-on-dimers-too).
- Cycle 105; candidate `90bd72f86ab9c6ab520f94decc121b9e157db39b`; champion `73c4e88596f23f25a9b520c6158ac5764251c9c2`; decision discard.
  [Training evidence](evaluation_results/cycle-105-train.json).
  [Narrative](full_log.md#cycle-105--connected-dummy-atom-dihedral-hessian-020-ha).
- Cycle 183; candidate `5d3ba1d194626ae7d83e37505364d0ba71c658b0`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision invalid.
  [Implementation](ideas/sella-dummy-dihedral-h0/5d3ba1d194626ae7d83e37505364d0ba71c658b0/algo.py).
  [Training evidence](evaluation_results/cycle-183-train.json).
  [Narrative](full_log.md#cycle-183--dummy-dihedral-h0025-on-dimers).

## sella-dummy-angle-h0: softer dummy-involving angle Hessian on connected molecules
Status: incorporated

**Hypothesis:** Dummy-center–real angles still use Fischer–Almlöf (~0.16 Ha) after cycle 103’s dummy-dihedral keep. Setting those angles to 0.10 Ha on connected molecules should cheapen dummy-linear extras without cycle 105’s 0.20 dummy-dihedral wander.

**Outcome and uncertainty:** Cycle 107 discard. Energy-ok. Dummy-linear canaries improved (315516336 −5, 363892164 −2) but pyridine–pyrrole (probe-connected DES) 23→47 and 135043047 44→54 undo the win (net +14). 104046121 did not wander. Cycle 236 **keep**: same 0.10 Ha only on connected n_atoms≥30. Train Δ=+3.18e-4, valid Δ=+1.91e-3. 18–19-atom extras spared; 363892164 46→44 kept. Cycle 374 carbon-only dummy-angle discard, Δ=−6.43e-4; leftover unchanged; extras 319495631/small N-center dummy-angle saves.

**Reason to revisit:** Incorporated as a size-gated keep. Do not drop the size gate back to all connected (cycle 107). Do not apply dummy-angle H0 on dimers. Do not restrict dummy-angle 0.10 to carbon linear centers.

**Next experiment:** Do not interpolate N-center dummy-angle scales. Widen the size class is closed (237/277/369).

**Attempts:**
- Cycle 374; candidate `d2ff88635a991e8df4d6f114b59596dbaca33139`; champion `7140449e9cbddb6f6589ec34fdac86f88dc413e1`; decision discard.
  [Implementation](ideas/sella-dummy-angle-h0/d2ff88635a991e8df4d6f114b59596dbaca33139/algo.py).
  [Training evidence](evaluation_results/cycle-374-train.json).
  [Narrative](full_log.md#cycle-374--dummy-angle-010-ha-only-at-carbon-linear-centers).
- Cycle 107; candidate `7982551213ef8dab31d457242f207eb908d7fa3d`; champion `73c4e88596f23f25a9b520c6158ac5764251c9c2`; decision discard.
  [Implementation](ideas/sella-dummy-angle-h0/7982551213ef8dab31d457242f207eb908d7fa3d/algo.py).
  [Training evidence](evaluation_results/cycle-107-train.json).
  [Narrative](full_log.md#cycle-107--softer-dummy-involving-angle-hessian-on-connected-molecules).

## sella-dummy-dihedral-wd: milder dummy-dihedral MaxInternalStep weight
Status: incorporated

**Hypothesis:** Dummy-atom dihedrals share `wd=1` with real torsions. Downweighting only dummy-set dihedrals enlarges linear-bend steps without cycle 105’s 0.20 H0 wander.

**Outcome and uncertainty:** Cycle 111/122 always-on `wd_dummy=0.8` non_generalizable (train Δ=+5.96e-4; valid Δ=+8.71e-5). nsteps/ρ/linearity gates failed. Cycle 167 limiter-then-global dummy-wd was bit-identical to 122. Cycle 168 **keep**: downweight only the limiter dummy-dihedral index. Train Δ=+5.96e-4 (same 9/14/23/43-step saves as 122). Valid Δ=+2.96e-4: 104129283 51→46 kept, 135093104 and 134982070 extras gone; 135249279 still 40→41.

**Reason to revisit:** Incorporated. Cycle 422 n<18 wd=0.7 extras leftover 252089162 39→40; save is B2S3. Keep wd=0.8.

**Next experiment:** Do not interpolate dummy-wd below 0.8, including n<18.

**Attempts:**
- Cycle 422; candidate `c258a7d2c8e6b4add131d223b49ed604c6d569ac`; champion `d2f7137d0609977f7ebd973704e68b8f419f8dad`; decision discard.
  [Implementation](ideas/sella-dummy-dihedral-wd/c258a7d2c8e6b4add131d223b49ed604c6d569ac/algo.py).
  [Training evidence](evaluation_results/cycle-422-train.json).
  [Narrative](full_log.md#cycle-422--limiter-dummy-dihedral-wd07-on-connected-n_atoms18).
- Cycle 111; candidate `b232b737fcae11c7c9f1391d058247748ec845bc`; champion `73c4e88596f23f25a9b520c6158ac5764251c9c2`; decision non_generalizable.
  [Implementation](ideas/sella-dummy-dihedral-wd/b232b737fcae11c7c9f1391d058247748ec845bc/algo.py).
  [Training evidence](evaluation_results/cycle-111-train.json).
  [Validation evidence](evaluation_results/cycle-111-valid.json).
  [Narrative](full_log.md#cycle-111--milder-dummy-dihedral-maxinternalstep-weight-on-connected-molecules).
- Cycle 112; candidate `d8937136ee29ea7b7a71949bc518465d7d7d3c8c`; champion `73c4e88596f23f25a9b520c6158ac5764251c9c2`; decision non_generalizable.
  [Implementation](ideas/sella-dummy-dihedral-wd/d8937136ee29ea7b7a71949bc518465d7d7d3c8c/algo.py).
  [Training evidence](evaluation_results/cycle-112-train.json).
  [Validation evidence](evaluation_results/cycle-112-valid.json).
  [Narrative](full_log.md#cycle-112--dummy-dihedral-maxinternalstep-weight-075-on-connected-molecules).
- Cycle 114; candidate `5218fa468541dd8f748f76ed88221caf3ecce09a`; champion `73c4e88596f23f25a9b520c6158ac5764251c9c2`; decision discard.
  [Implementation](ideas/sella-dummy-dihedral-wd/5218fa468541dd8f748f76ed88221caf3ecce09a/algo.py).
  [Training evidence](evaluation_results/cycle-114-train.json).
  [Narrative](full_log.md#cycle-114--delayed-dummy-dihedral-maxinternalstep-weight-08-after-20-steps).
- Cycle 122; candidate `b5c569ddaf3ad2470a7cecc53f061a266003e88e`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision non_generalizable.
  [Implementation](ideas/sella-dummy-dihedral-wd/b5c569ddaf3ad2470a7cecc53f061a266003e88e/algo.py).
  [Training evidence](evaluation_results/cycle-122-train.json).
  [Validation evidence](evaluation_results/cycle-122-valid.json).
  [Narrative](full_log.md#cycle-122--dummy-dihedral-maxinternalstep-weight-08-on-the-gdiis-champion).
- Cycle 139; candidate `092710e897cf36cfa554176ab3314b2d430f9c6b`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision non_generalizable.
  [Implementation](ideas/sella-dummy-dihedral-wd/092710e897cf36cfa554176ab3314b2d430f9c6b/algo.py).
  [Training evidence](evaluation_results/cycle-139-train.json).
  [Validation evidence](evaluation_results/cycle-139-valid.json).
  [Narrative](full_log.md#cycle-139--early-only-dummy-dihedral-mis-weight-08).
- Cycle 144; candidate `5bbe075e7d7f382454c1a5d2cdd7a0ce77e6e6de`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision non_generalizable.
  [Implementation](ideas/sella-dummy-dihedral-wd/5bbe075e7d7f382454c1a5d2cdd7a0ce77e6e6de/algo.py).
  [Training evidence](evaluation_results/cycle-144-train.json).
  [Validation evidence](evaluation_results/cycle-144-valid.json).
  [Narrative](full_log.md#cycle-144--dummy-dihedral-wd08-plus-connected-rfo-after-50).
- Cycle 145; candidate `af10e6b42446c968e19d594c35841dd30badc61b`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision non_generalizable.
  [Implementation](ideas/sella-dummy-dihedral-wd/af10e6b42446c968e19d594c35841dd30badc61b/algo.py).
  [Training evidence](evaluation_results/cycle-145-train.json).
  [Validation evidence](evaluation_results/cycle-145-valid.json).
  [Narrative](full_log.md#cycle-145--dummy-dihedral-wd08-plus-connected-1d-quadratic-α).
- Cycle 152; candidate `9e9e85575e70c95842d5c59110be11f4b9a855f8`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision discard.
  [Implementation](ideas/sella-dummy-dihedral-wd/9e9e85575e70c95842d5c59110be11f4b9a855f8/algo.py).
  [Training evidence](evaluation_results/cycle-152-train.json).
  [Narrative](full_log.md#cycle-152--ρ-gated-dummy-dihedral-mis-weight-08).
- Cycle 153; candidate `07cc38f2fb3bb833cb1be25c2bf682395e902cfc`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision non_generalizable.
  [Implementation](ideas/sella-dummy-dihedral-wd/07cc38f2fb3bb833cb1be25c2bf682395e902cfc/algo.py).
  [Training evidence](evaluation_results/cycle-153-train.json).
  [Validation evidence](evaluation_results/cycle-153-valid.json).
  [Narrative](full_log.md#cycle-153--dummy-dihedral-wd08-only-on-still-linear-parents).
- Cycle 167; candidate `96ce60a1223904e02fe2917e6c0e06f123cbd6a2`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision non_generalizable.
  [Implementation](ideas/sella-dummy-dihedral-wd/96ce60a1223904e02fe2917e6c0e06f123cbd6a2/algo.py).
  [Training evidence](evaluation_results/cycle-167-train.json).
  [Validation evidence](evaluation_results/cycle-167-valid.json).
  [Narrative](full_log.md#cycle-167--dummy-dihedral-wd08-only-when-it-is-the-mis-limiter).
- Cycle 168; candidate `3dd5d21702bcb0ef6314d45201113465bf17ee7f`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision keep.
  [Training evidence](evaluation_results/cycle-168-train.json).
  [Validation evidence](evaluation_results/cycle-168-valid.json).
  [Narrative](full_log.md#cycle-168--wd08-on-only-the-limiter-dummy-dihedral).
- Cycle 182; candidate `a04dbcd16514ae20d905124fdcaf87d63ad3195d`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-dummy-dihedral-wd/a04dbcd16514ae20d905124fdcaf87d63ad3195d/algo.py).
  [Training evidence](evaluation_results/cycle-182-train.json).
  [Narrative](full_log.md#cycle-182--dummy-dihedral-limiter-wd-on-dimers).
- Cycle 187; candidate `f62916a4d306f2d601db213c4bf235bc60232d11`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-dummy-dihedral-wd/f62916a4d306f2d601db213c4bf235bc60232d11/algo.py).
  [Training evidence](evaluation_results/cycle-187-train.json).
  [Narrative](full_log.md#cycle-187--limiter-dummy-dihedral-wd07).

## sella-dummy-angle-wa: milder dummy-involving angle MaxInternalStep weight
Status: deferred

**Hypothesis:** Dummy-involving unconstrained angles share `wa=0.75`. Downweighting them to 0.7 enlarges linear-bend steps without cycle 107’s 0.10 H0.

**Outcome and uncertainty:** Cycle 115 discard, bit-identical. Dummy-angle MIS is a no-op; those angles are not the limiter. Cycle 203 limiter-only wa=0.65 is also n_steps bit-identical to cycle 180.

**Reason to revisit:** Not as a `wa_dummy` tweak. Dummy-angle H0 (cycle 107) is the direction change; MIS is not.

**Next experiment:** Do not interpolate `wa_dummy`. Do not retry limiter dummy-angle weights.

**Attempts:**
- Cycle 115; candidate `41249e1f0480062b497f18e9c1c1c191ed8adb07`; champion `73c4e88596f23f25a9b520c6158ac5764251c9c2`; decision discard.
  [Implementation](ideas/sella-dummy-angle-wa/41249e1f0480062b497f18e9c1c1c191ed8adb07/algo.py).
  [Training evidence](evaluation_results/cycle-115-train.json).
  [Narrative](full_log.md#cycle-115--milder-dummy-involving-angle-maxinternalstep-weight-on-connected-molecules).
- Cycle 203; candidate `30faf57b75df6e573961655794578b7096bc8d9c`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-dummy-angle-wa/30faf57b75df6e573961655794578b7096bc8d9c/algo.py).
  [Training evidence](evaluation_results/cycle-203-train.json).
  [Narrative](full_log.md#cycle-203--limiter-dummy-involving-angle-maxinternalstep-weight-065).

## sella-lindh-angle-h0: Lindh 1995 angle Hessian on connected molecules
Status: deferred

**Hypothesis:** Lindh k_θ=0.15 ρ_ab ρ_bc on connected non-dummy angles cheapens organics without cycle 113’s stretch H0 paliperidone inflation.

**Outcome and uncertainty:** Cycle 116 invalid. 135043047 +38.01 kcal (44→33); venetoclax +1.38 kcal; two other PubChem hops. Paliperidone 78→83 energy-safe. Soft Lindh angles hop the same canary as dummy-angle H0 / Swart torsions.

**Reason to revisit:** Not as a k_θ or α tweak. Dummy-only Lindh angles would still be a different idea, but cycle 107 already mixed dummy-angle H0.

**Next experiment:** Do not interpolate Lindh angle k=0.15. Do not combine with Lindh stretches.

**Attempts:**
- Cycle 116; candidate `dd0dbeb9e4050c10ba239994306afd6dc6330d4e`; champion `73c4e88596f23f25a9b520c6158ac5764251c9c2`; decision invalid.
  [Implementation](ideas/sella-lindh-angle-h0/dd0dbeb9e4050c10ba239994306afd6dc6330d4e/algo.py).
  [Training evidence](evaluation_results/cycle-116-train.json).
  [Narrative](full_log.md#cycle-116--lindh-1995-angle-hessian-on-connected-molecules).

## sella-schlegel-bond-h0: Schlegel 1984 stretch Hessian on connected molecules
Status: deferred

**Hypothesis:** OptKing’s default intrafragment stretch guess (Schlegel 1984, k=1.734/(R−B)³) on connected non-dummy bonds cheapens remaining organics vs Fischer–Almlöf without Swart torsions.

**Outcome and uncertainty:** Cycle 108 discard (energy-ok, paliperidone/venetoclax inflate, 135043047 44→39). Cycle 110 heavy-only Schlegel (Z≥3): invalid. Paliperidone 78→200 R=−6.13 / +163 kcal; venetoclax still R=0.948; 135043047 still 44→39. Excluding H–X made paliperidone worse. Close Schlegel stretch H0.

**Reason to revisit:** Not as a Z or B_period gate. A different model Hessian family (Lindh rho, not Schlegel) would be a new idea.

**Next experiment:** Do not interpolate Schlegel. Do not apply it on dimers.

**Attempts:**
- Cycle 108; candidate `546f75e7f1ba1151b5ac7e98e3e59a3c0d4a0d03`; champion `73c4e88596f23f25a9b520c6158ac5764251c9c2`; decision discard.
  [Implementation](ideas/sella-schlegel-bond-h0/546f75e7f1ba1151b5ac7e98e3e59a3c0d4a0d03/algo.py).
  [Training evidence](evaluation_results/cycle-108-train.json).
  [Narrative](full_log.md#cycle-108--schlegel-1984-stretch-hessian-on-connected-molecules).
- Cycle 110; candidate `0368b0e8157148ad48b2622c4a37111c60bc3df6`; champion `73c4e88596f23f25a9b520c6158ac5764251c9c2`; decision invalid.
  [Implementation](ideas/sella-schlegel-bond-h0/0368b0e8157148ad48b2622c4a37111c60bc3df6/algo.py).
  [Training evidence](evaluation_results/cycle-110-train.json).
  [Narrative](full_log.md#cycle-110--schlegel-stretch-hessian-on-connected-period2-bonds).

## sella-lindh-bond-h0: Lindh 1995 stretch Hessian on connected molecules
Status: deferred

**Hypothesis:** Lindh ρ=exp[α(r_cov²−r²)] stretch H0 on connected non-dummy bonds cheapens organics vs Fischer without Schlegel’s 1/R³ paliperidone hop.

**Outcome and uncertainty:** Cycle 113 discard. Energy-ok. C=1.00046 (Δ=−7.19e-3). Paliperidone 78→101; 135249644 15→29 R=1.079; net +52. Gaussian stretch H0 is cost-negative here.

**Reason to revisit:** Not as an α tweak. Dummy-bond-only Lindh would be a different idea.

**Next experiment:** Do not interpolate Lindh α or k=0.45. Cycle 116 Lindh *angles* hopped 135043047; do not combine stretch+angle Lindh.

**Attempts:**
- Cycle 113; candidate `c8580c90c721e68790f003d732da85ab6774c631`; champion `73c4e88596f23f25a9b520c6158ac5764251c9c2`; decision discard.
  [Implementation](ideas/sella-lindh-bond-h0/c8580c90c721e68790f003d732da85ab6774c631/algo.py).
  [Training evidence](evaluation_results/cycle-113-train.json).
  [Narrative](full_log.md#cycle-113--lindh-1995-stretch-hessian-on-connected-molecules).

## optking-interfrag-h0: OptKing DEFAULT stretch Hessian on dimer connectors
Status: deferred

**Hypothesis:** Cutoff-grown connecting stretches have Fischer H0 below OptKing’s 0.007 Ha/Bohr². Flooring them at 0.007 should cheapen long DES jobs without scaling covalent Fischer (cycles 60–61).

**Outcome and uncertainty:** Cycle 106 invalid. Connected identical. Unofficial DES cheapening was packing hops; in-well dimers net +402 calls. The floor is too stiff and cost-negative in-well.

**Reason to revisit:** Cycle 109 actual 1/R connecting Bonds also hopped packing (amines/water-water stalled at 200; imidazolium–water R=0.765). Pair-marking is reusable only with a well-preserving limiter, not another connecting-coordinate change.

**Next experiment:** Do not retry 1/R or r-space connecting H0 floors.

**Attempts:**
- Cycle 106; candidate `2914b6c4138322ffda8769e01a2ff3ebbae8af46`; champion `73c4e88596f23f25a9b520c6158ac5764251c9c2`; decision invalid.
  [Implementation](ideas/optking-interfrag-h0/2914b6c4138322ffda8769e01a2ff3ebbae8af46/algo.py).
  [Training evidence](evaluation_results/cycle-106-train.json).
  [Narrative](full_log.md#cycle-106--optking-interfragment-stretch-hessian-floor-on-dimers).

## baker-invbond-connecting: Baker/OptKing 1/R connecting stretches on dimers
Status: deferred

**Hypothesis:** q=1/R on cutoff-grown connectors with H_qq=H_rr R⁴ and MIS weights R² matches Fischer Cartesian stiffness while making long contacts more harmonic.

**Outcome and uncertainty:** Cycle 109 invalid. Connected identical. DES hops and stalls (amines 99→200 R=0.13, water–water 45→200 R=0.22, imidazolium–water 79→44 R=0.765). Finite 1/R steps leave the starting well.

**Reason to revisit:** Only with an energy-checked accept that does not spend extra evals, not a weaker R² cap.

**Next experiment:** Do not drop the R² MIS scale. Do not apply 1/R on connected molecules.

**Attempts:**
- Cycle 109; candidate `3eb05f3473d3435085fec1ed80f463085d2a5f89`; champion `73c4e88596f23f25a9b520c6158ac5764251c9c2`; decision invalid.
  [Implementation](ideas/baker-invbond-connecting/3eb05f3473d3435085fec1ed80f463085d2a5f89/algo.py).
  [Training evidence](evaluation_results/cycle-109-train.json).
  [Narrative](full_log.md#cycle-109--bakeroptking-1r-connecting-stretches-on-dimers).

## sella-dimer-sigma-dec: harder dimer trust shrink after poor ρ
Status: deferred

**Hypothesis:** Cycle 65 floors dimer δ at 0.02 after rare ρ<0.01 shrinks. Cutting `σ_dec` 0.90→0.80 makes those post-failure trusts smaller (still ≥0.02) and may cheapen long DES jobs that kept too much δ.

**Outcome and uncertainty:** Cycle 74 discard. Energy-ok, C=1.00277 (Δ=−3.72e-3). 30 DES losses vs 12 wins; alkenes/ethers/pyridine/benzene–water inflate; amines 99→102; ethers–ethers −2.74 kcal overshoot. Connected unchanged. Harder shrinks after rare bad ρ make later dimer steps too small.

**Reason to revisit:** Only paired with a method that leaves δ too large after a diagnosed poor-ρ event, not as a champion-only change. Do not interpolate 0.85.

**Next experiment:** Do not retune dimer `sigma_dec`.

**Attempts:**
- Cycle 74; candidate `fa35e86d27034ceee046c83ed3e3dc8d433780b6`; champion `a9c931a62562b4b5d904a2b373be7b3933200aa5`; decision discard.
  [Implementation](ideas/sella-dimer-sigma-dec/fa35e86d27034ceee046c83ed3e3dc8d433780b6/algo.py).
  [Training evidence](evaluation_results/cycle-74-train.json).
  [Narrative](full_log.md#cycle-74--dimer-sigma_dec080-harder-poor-ρ-shrink).

## sella-extra-impropers: extra-redundant impropers on connected 3-coordinate centers
Status: closed

**Hypothesis:** Sella skips an improper at 3-coordinate centers that already have a proper torsion. Extra-redundant out-of-plane internals (Bakken, not 1–3 stretches) should cheapen planar-rich organics that still cost 5–7 extra calls vs the original Sella.

**Outcome and uncertainty:** Cycles 75–77 invalid (ODE timeouts). Cycle 78 crash-free but energy-invalid: venetoclax +1.38 kcal, C=1.154. Extra OOP coordinates plus any working realization hop or inflate connected organics. Closed after three geodesic repairs.

**Reason to revisit:** Not as active extra dihedrals. A frozen out-of-plane constraint would be a different idea.

**Next experiment:** Do not add extra impropers. Do not retune the geodesic fallback.

**Attempts:**
- Cycle 75; candidate `ef9cc2cda05fe95395f959edc51894cecdfde933`; champion `a9c931a62562b4b5d904a2b373be7b3933200aa5`; decision invalid.
  [Implementation](ideas/sella-extra-impropers/ef9cc2cda05fe95395f959edc51894cecdfde933/algo.py).
  [Training evidence](evaluation_results/cycle-75-train.json).
  [Narrative](full_log.md#cycle-75--extra-impropers-on-connected-3-coordinate-centers).
- Cycle 76; candidate `82eac8c6dc06ab8db01ce9269092e5746d53614e`; champion `a9c931a62562b4b5d904a2b373be7b3933200aa5`; decision invalid.
  [Implementation](ideas/sella-extra-impropers/82eac8c6dc06ab8db01ce9269092e5746d53614e/algo.py).
  [Training evidence](evaluation_results/cycle-76-train.json).
  [Narrative](full_log.md#cycle-76--extra-impropers-with-geodesic-ode-restorehalve).
- Cycle 77; candidate `c724406e9eee94c58f269c033ecdebb9192756a6`; champion `a9c931a62562b4b5d904a2b373be7b3933200aa5`; decision invalid.
  [Implementation](ideas/sella-extra-impropers/c724406e9eee94c58f269c033ecdebb9192756a6/algo.py).
  [Training evidence](evaluation_results/cycle-77-train.json).
  [Narrative](full_log.md#cycle-77--extra-impropers-with-iterative-cartesian-realization).
- Cycle 78; candidate `9eb7535f1cfff1301c78154ddd704234f0c032a7`; champion `a9c931a62562b4b5d904a2b373be7b3933200aa5`; decision invalid.
  [Implementation](ideas/sella-extra-impropers/9eb7535f1cfff1301c78154ddd704234f0c032a7/algo.py).
  [Training evidence](evaluation_results/cycle-78-train.json).
  [Narrative](full_log.md#cycle-78--extra-impropers-with-linearized-b-fallback).

## sella-hbond-contacts: extra-redundant hydrogen-bond stretches on connected molecules
Status: deferred

**Hypothesis:** Intramolecular H...acceptor stretches added after covalent internals give QN an explicit H-bond coordinate on large organics without 1–3 covalent extras.

**Outcome and uncertainty:** Cycle 81 discard. venetoclax 56→161 with −1.66 kcal overshoot; 252089162 44→39 but +1.34 kcal. Extra H-bonds are not a keep.

**Reason to revisit:** Not as active Bond extras. A frozen H-bond constraint is a different idea.

**Next experiment:** Do not add H...acceptor stretches. Do not retune the 1.2–2.5 Å window.

**Attempts:**
- Cycle 81; candidate `eebfc4f5b9dcf4879d79b3fcf8b18beff9f66846`; champion `a9c931a62562b4b5d904a2b373be7b3933200aa5`; decision discard.
  [Implementation](ideas/sella-hbond-contacts/eebfc4f5b9dcf4879d79b3fcf8b18beff9f66846/algo.py).
  [Training evidence](evaluation_results/cycle-81-train.json).
  [Narrative](full_log.md#cycle-81--extra-redundant-hydrogen-bond-stretches-on-connected-molecules).

## sella-linear-atol: tighter linear-angle dummy threshold on connected molecules
Status: deferred

**Hypothesis:** Treating more near-linear bends as dummy linear coordinates cheapens some connected organics without extra Bond objects. `atol` is the half-width of that window.

**Outcome and uncertainty:** Cycle 82 `atol=10°` invalid (+38 kcal). Cycle 83 global `atol=20°` invalid on R by 4.5e-8 but cost Δ=+1.66e-4 with 135043047 46→38 energy-safe. Cycle 84 global 18° discard (104079126 37→50). Cycle 85 3-coord-only 20° discard (135043047 unchanged; 104079126 37→41). Cycle 86 2-coord-only 20° non_generalizable: train Δ=+1.79e-4 without moving 135043047; valid 135092258 15→26. Combined 20° needs both paths for the 135043047 win and is energy-unsafe; each split fails a gate.

**Reason to revisit:** Only if a well-preserving limiter already stops 135092258-class 2-coord dummy conversions *and* the 15–20° energy misses on 135078057/pyridine–pyrrole.

**Next experiment:** Do not interpolate atol or recombine the two 20° paths. See `sella-h0-reset-20`.

**Attempts:**
- Cycle 82; candidate `d3c4cd8655bcd95bf30f7731f83979687a9ddedb`; champion `a9c931a62562b4b5d904a2b373be7b3933200aa5`; decision invalid.
  [Implementation](ideas/sella-linear-atol/d3c4cd8655bcd95bf30f7731f83979687a9ddedb/algo.py).
  [Training evidence](evaluation_results/cycle-82-train.json).
  [Narrative](full_log.md#cycle-82--tighter-linear-angle-dummy-threshold-on-connected-molecules).
- Cycle 83; candidate `3baed08ad8962ee94e18c68a1a8380a441080a3d`; champion `a9c931a62562b4b5d904a2b373be7b3933200aa5`; decision invalid.
  [Implementation](ideas/sella-linear-atol/3baed08ad8962ee94e18c68a1a8380a441080a3d/algo.py).
  [Training evidence](evaluation_results/cycle-83-train.json).
  [Narrative](full_log.md#cycle-83--looser-linear-angle-dummy-threshold-on-connected-molecules).
- Cycle 84; candidate `335fc695e397348f202a67f782eb2910c618987f`; champion `a9c931a62562b4b5d904a2b373be7b3933200aa5`; decision discard.
  [Implementation](ideas/sella-linear-atol/335fc695e397348f202a67f782eb2910c618987f/algo.py).
  [Training evidence](evaluation_results/cycle-84-train.json).
  [Narrative](full_log.md#cycle-84--connected-linear-angle-dummy-threshold-atol18).
- Cycle 85; candidate `eb422fc51ffc8c6d83f48240b01ea2df015c8808`; champion `a9c931a62562b4b5d904a2b373be7b3933200aa5`; decision discard.
  [Implementation](ideas/sella-linear-atol/eb422fc51ffc8c6d83f48240b01ea2df015c8808/algo.py).
  [Training evidence](evaluation_results/cycle-85-train.json).
  [Narrative](full_log.md#cycle-85--multi-coordinate-linear-improper-threshold-atol20).
- Cycle 86; candidate `5fe601c2c9bf48d2e71b6c8202552407e0c659ab`; champion `a9c931a62562b4b5d904a2b373be7b3933200aa5`; decision non_generalizable.
  [Implementation](ideas/sella-linear-atol/5fe601c2c9bf48d2e71b6c8202552407e0c659ab/algo.py).
  [Training evidence](evaluation_results/cycle-86-train.json).
  [Validation evidence](evaluation_results/cycle-86-valid.json).
  [Narrative](full_log.md#cycle-86--two-coordinate-dummy-atom-linear-threshold-atol20).

## sella-h0-reset-20: reset connected BFGS Hessian to model H0 after 20 steps
Status: deferred

**Hypothesis:** Long connected jobs carry 20+ TS-BFGS updates that can stale. Replacing B with current-geometry Fischer–Almlöf H0 at nsteps==20 should cheapen remaining Newton steps without extra `calc()` calls.

**Outcome and uncertainty:** Cycle 87 invalid. Fresh H0 at 20 false-converges mid-length jobs at ~22 calls (acalabrutinib +0.007 kcal) and inflates the long targets (venetoclax 56→59, paliperidone 76→89). Mechanism contradicted.

**Reason to revisit:** Only with an energy-checked accept of the first post-reset step, not a later nsteps delay (targets already got worse).

**Next experiment:** Do not delay the H0 reset. See delayed connected δ=0.15 floor.

**Attempts:**
- Cycle 87; candidate `659ad401c71aacb956fd2df103748b97166ad370`; champion `a9c931a62562b4b5d904a2b373be7b3933200aa5`; decision invalid.
  [Implementation](ideas/sella-h0-reset-20/659ad401c71aacb956fd2df103748b97166ad370/algo.py).
  [Training evidence](evaluation_results/cycle-87-train.json).
  [Narrative](full_log.md#cycle-87--reset-connected-model-hessian-after-20-steps).

## sella-late-wd: milder connected dihedral weight after the 0.15 floor
Status: deferred

**Hypothesis:** On the cycle-88 0.15 floor, `wd=1` still caps |s_d| at 0.15. A mild `wd=0.9` after 20 lets torsions use |s_d|≤0.167 without cycle 29’s `wd=2/3` hop.

**Outcome and uncertainty:** Cycle 95 discard. Energy-ok; 252618428 unchanged at 35. 135043047 45→43 but paliperidone/160853090/172113611/venetoclax inflate. Net Δ=−3.25e-4. Late larger torsions are not a keep.

**Reason to revisit:** Only with a limiter that protects floppy tails, not a milder `wd`.

**Next experiment:** Do not interpolate `wd=0.95`. Late `wb=1.5` (cycle 96) was also not a keep.

**Attempts:**
- Cycle 95; candidate `a79e2da787e7dd3da76ff335eaefc74fa24031ca`; champion `453a761dd3eace02562f81dc72920ac1e38526cc`; decision discard.
  [Implementation](ideas/sella-late-wd/a79e2da787e7dd3da76ff335eaefc74fa24031ca/algo.py).
  [Training evidence](evaluation_results/cycle-95-train.json).
  [Narrative](full_log.md#cycle-95--connected-dihedral-weight-wd09-after-20-steps).

## sella-late-wb: original bond cap on the connected 0.15 floor
Status: deferred

**Hypothesis:** Cycle 88’s 0.15 floor enlarges stretches with bends/torsions. `wb=1.5` after 20 restores |s_b|≤0.10 (cycle 8 stretch-hop thesis) while keeping the floor for angles/torsions.

**Outcome and uncertainty:** Cycle 96 discard. Δ=−5.47e-5. paliperidone 78→77; phenol–monoatomic 52→54; venetoclax/135043047/252089162 unchanged. Late stretches are not the MIS limiter.

**Reason to revisit:** Only if a later method makes late Newton steps stretch-limited.

**Next experiment:** Do not interpolate `wb`. See controlled GDIIS (cycle 97).

**Attempts:**
- Cycle 96; candidate `5d4dfd77c99820f3b89045eddba338ce1d3ac1ab`; champion `453a761dd3eace02562f81dc72920ac1e38526cc`; decision discard.
  [Implementation](ideas/sella-late-wb/5d4dfd77c99820f3b89045eddba338ce1d3ac1ab/algo.py).
  [Training evidence](evaluation_results/cycle-96-train.json).
  [Narrative](full_log.md#cycle-96--connected-bond-weight-wb15-after-20-steps).

## sella-connected-delta-min-005: early connected trust-radius floor 0.05
Status: deferred

**Hypothesis:** Connected jobs collapse δ to eta=1e-4 on poor-ρ shrinks before the 0.15@20 floor. An early 0.05 floor cheapens those recoveries without dimer 0.02 or large-δ0 hops.

**Outcome and uncertainty:** Cycle 123 non_generalizable. Train Δ=+1.94e-4 from nintedanib 22→20 only. Valid bit-identical. Valid connected jobs do not have cost-changing pre-20 shrinks below 0.05.

**Reason to revisit:** Only if a later method makes pre-20 connected shrinks the limiter on valid, not by raising the floor. Cycle 125 stacked the floor with dimer GDIIS; the valid stall was dimer GDIIS (alkenes–alkenes 155→200), not the floor.

**Next experiment:** Do not interpolate 0.05–0.08. Do not restack with dimer GDIIS.

**Attempts:**
- Cycle 123; candidate `5c5d2537ed86cf28cfa3a30d80dc7552b4f482e9`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision non_generalizable.
  [Implementation](ideas/sella-connected-delta-min-005/5c5d2537ed86cf28cfa3a30d80dc7552b4f482e9/algo.py).
  [Training evidence](evaluation_results/cycle-123-train.json).
  [Validation evidence](evaluation_results/cycle-123-valid.json).
  [Narrative](full_log.md#cycle-123--connected-early-trust-radius-floor-005).
- Cycle 125; candidate `d978dd2ac4d948c4b67aab3ed7bede687e0efcd5`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision non_generalizable.
  [Implementation](ideas/sella-gdiis/d978dd2ac4d948c4b67aab3ed7bede687e0efcd5/algo.py).
  [Training evidence](evaluation_results/cycle-125-train.json).
  [Validation evidence](evaluation_results/cycle-125-valid.json).
  [Narrative](full_log.md#cycle-125--connected-δ-floor-005-plus-dimer-gdiis).

## sella-gediis: Li–Frisch GEDIIS interpolant on connected tails
Status: deferred

**Hypothesis:** Two-point interpolation GEDIIS (energy-model convex combination of the last two internals) cheapens overshooting connected tails when ρ-gated GDIIS does not accept.

**Outcome and uncertainty:** Cycle 126 invalid (hops). Cycle 127 cosine discard. Cycle 128 uphill discard (paliperidone +4). Cycle 129 ρ-gated fallback bit-identical (Δ=0). Last-segment GEDIIS is not a keep.

**Reason to revisit:** Only if a later method needs an energy interpolant that is *not* on the last two-point segment (3-point GEDIIS with a well-preserving limiter).

**Next experiment:** Do not add more GEDIIS gates. See connected Hessian |λ| floor.

**Attempts:**
- Cycle 126; candidate `59870bb30ef3b821555093fa2ae622a22a7bd686`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision invalid.
  [Implementation](ideas/sella-gediis/59870bb30ef3b821555093fa2ae622a22a7bd686/algo.py).
  [Training evidence](evaluation_results/cycle-126-train.json).
  [Narrative](full_log.md#cycle-126--two-point-interpolation-gediis-on-connected-tails).
- Cycle 127; candidate `b5c867dce24d25852b1a71f2d971dfdcecf679b0`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision discard.
  [Implementation](ideas/sella-gediis/b5c867dce24d25852b1a71f2d971dfdcecf679b0/algo.py).
  [Training evidence](evaluation_results/cycle-127-train.json).
  [Narrative](full_log.md#cycle-127--gediis-with-qn-cosine-090-on-connected-tails).
- Cycle 128; candidate `16e8e6fa588c1942acc4c45ae82ecfff8cd9bba7`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision discard.
  [Implementation](ideas/sella-gediis/16e8e6fa588c1942acc4c45ae82ecfff8cd9bba7/algo.py).
  [Training evidence](evaluation_results/cycle-128-train.json).
  [Narrative](full_log.md#cycle-128--uphill-only-cosine-gediis-on-connected-tails).
- Cycle 129; candidate `26209cc1c46e0e99f2feda9c8e7dae7df94a87c7`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision discard.
  [Implementation](ideas/sella-gediis/26209cc1c46e0e99f2feda9c8e7dae7df94a87c7/algo.py).
  [Training evidence](evaluation_results/cycle-129-train.json).
  [Narrative](full_log.md#cycle-129--ρ-gated-uphill-cosine-gediis-on-connected-tails).

## sella-eval-floor: Helgaker Hessian |λ| floor on connected QN steps
Status: deferred

**Hypothesis:** Flooring tiny TS-BFGS |λ| shortens floppy Newton modes on connected organics without changing internals.

**Outcome and uncertainty:** Cycle 130 0.01 Eh invalid (stalls/hops). Cycle 131 0.001 Eh discard: energy-safe, C=1.031, paliperidone 78→126. Raising |λ| lengthens connected jobs. Cycles 758–760 unique-tag floors of 1e-4/0.001/0.01 on 30–80 isocyanides were bit-identical leftover 33 because `get_HL_projected` dropped `eval_floor`. Cycle 761 copy plus 0.01 Eh extras leftover 363892164 33→41, energy-safe, Δ=−5.50e-4. Cycle 762 copy plus 0.001 Eh bit-identical leftover 33.

**Reason to revisit:** Not as a global floor. Unique-tag 0.01 extras leftover. Unique-tag 0.001 with the projected-Hessian copy is a no-op (leftover |λ| ≥ 0.001). Close leftover 363 Helgaker floors.

**Next experiment:** Do not interpolate 1e-4 Eh on this tag (would also be a no-op). Powell-damped Hessian updates from start on 30–80 isocyanides.

**Attempts:**
- Cycle 130; candidate `262bd262aa642c7cc9a1c37ab8c3d3b9c34e90f1`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision invalid.
  [Implementation](ideas/sella-eval-floor/262bd262aa642c7cc9a1c37ab8c3d3b9c34e90f1/algo.py).
  [Training evidence](evaluation_results/cycle-130-train.json).
  [Narrative](full_log.md#cycle-130--connected-qn-hessian-λ-floor-001-eh).
- Cycle 131; candidate `1144b1fc9f72a2de04915fe54681fc7dc8890ab8`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision discard.
  [Implementation](ideas/sella-eval-floor/1144b1fc9f72a2de04915fe54681fc7dc8890ab8/algo.py).
  [Training evidence](evaluation_results/cycle-131-train.json).
  [Narrative](full_log.md#cycle-131--connected-qn-hessian-λ-floor-0001-eh).
- Cycle 758; candidate `807dbf6d673d5a59df778ae0df43f3a4f43126eb`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision discard.
  [Training evidence](evaluation_results/cycle-758-train.json).
  [Narrative](full_log.md#cycle-758--cycle-728-plus-helgaker-λ-floor-1e-4-eh-on-30-80-isocyanides).
- Cycle 759; candidate `623f7415221fea9f529164f8a4ee260104396f08`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision discard.
  [Training evidence](evaluation_results/cycle-759-train.json).
  [Narrative](full_log.md#cycle-759--cycle-728-plus-helgaker-λ-floor-0001-eh-on-30-80-isocyanides).
- Cycle 760; candidate `5b29e2eb7db1015bfe57ada35d261f73dbcc464b`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision discard.
  [Training evidence](evaluation_results/cycle-760-train.json).
  [Narrative](full_log.md#cycle-760--cycle-728-plus-helgaker-λ-floor-001-eh-on-30-80-isocyanides).
- Cycle 761; candidate `17ad841e2fe8b92d3a2d623ba83961adb8be2ec3`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision discard.
  [Training evidence](evaluation_results/cycle-761-train.json).
  [Narrative](full_log.md#cycle-761--cycle-760-plus-copy-eval_floor-onto-the-projected-hessian).
- Cycle 762; candidate `033a4c9fe85f9db6b21ae59199b566e43cf8b429`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision discard.
  [Training evidence](evaluation_results/cycle-762-train.json).
  [Narrative](full_log.md#cycle-762--cycle-728-plus-helgaker-λ-floor-0001-eh-copied-onto-projected-hessian-on-30-80-isocyanides).

## sella-quadratic-alpha: 1D quadratic step scaling on connected MIS steps
Status: deferred

**Hypothesis:** After MaxInternalStep, the 1D quadratic along s has its minimum at α*=−(g·s)/(s·H·s). Cycle 37 only halved when df_pred>0 (α*<1/2) and never fired. Taking α*s when 0<α*<1 shortens overshooting connected Newton steps without extra force calls.

**Outcome and uncertainty:** Cycle 134 discard: paliperidone 78→76, 104079126 37→38, Δ=+2.67e-6. Cycle 135 truncated-only bit-identical. Cycle 136 nsteps≥40 paliperidone unchanged. Cycle 145 stacked with wd_dummy: train additive Δ=+5.99e-4; valid Δ=+2.32e-5 because α* cut 104129283 51→46 to 51→47 and added 135065494 93→94.

**Reason to revisit:** Not stacked with wd_dummy. Standalone α* cannot meet Δ≥1e-4.

**Next experiment:** Do not restack α* with dummy-dihedral weights. Do not apply α* to dimer RFO tails (cycle 206 bit-identical).

**Attempts:**
- Cycle 134; candidate `f28d9e5d2887b20b9d92e663138fb25cac0891c5`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision discard.
  [Implementation](ideas/sella-quadratic-alpha/f28d9e5d2887b20b9d92e663138fb25cac0891c5/algo.py).
  [Training evidence](evaluation_results/cycle-134-train.json).
  [Narrative](full_log.md#cycle-134--connected-1d-quadratic-α-scaling-of-mis-steps).
- Cycle 135; candidate `a910fda852cf50559a7c4fc2b3dfac2a5a78e01e`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision discard.
  [Implementation](ideas/sella-quadratic-alpha/a910fda852cf50559a7c4fc2b3dfac2a5a78e01e/algo.py).
  [Training evidence](evaluation_results/cycle-135-train.json).
  [Narrative](full_log.md#cycle-135--truncated-only-connected-1d-quadratic-α).
- Cycle 136; candidate `70801bff17ac43184d02acc8e52452bbf3b8d964`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision discard.
  [Implementation](ideas/sella-quadratic-alpha/70801bff17ac43184d02acc8e52452bbf3b8d964/algo.py).
  [Training evidence](evaluation_results/cycle-136-train.json).
  [Narrative](full_log.md#cycle-136--connected-1d-quadratic-α-after-40-steps).
- Cycle 145; candidate `af10e6b42446c968e19d594c35841dd30badc61b`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision non_generalizable.
  [Implementation](ideas/sella-quadratic-alpha/af10e6b42446c968e19d594c35841dd30badc61b/algo.py).
  [Training evidence](evaluation_results/cycle-145-train.json).
  [Validation evidence](evaluation_results/cycle-145-valid.json).
  [Narrative](full_log.md#cycle-145--dummy-dihedral-wd08-plus-connected-1d-quadratic-α).
- Cycle 206; candidate `206bc688335cc93a437104fb2f90c38f191f14de`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-quadratic-alpha/206bc688335cc93a437104fb2f90c38f191f14de/algo.py).
  [Training evidence](evaluation_results/cycle-206-train.json).
  [Narrative](full_log.md#cycle-206--1d-quadratic-α-on-dimer-rfo-after-80-steps).

## sella-psb-update: Powell-symmetric-Broyden Hessian updates on connected molecules
Status: deferred

**Hypothesis:** PSB is more stable than TS-BFGS for noisy curvature on long connected organics.

**Outcome and uncertainty:** Cycle 138 discard. C=1.278, 135043047 +38 kcal, paliperidone +1.44 kcal, five 200-call stalls. PSB hops the same canary as Swart/Lindh angle H0.

**Reason to revisit:** Not as a drop-in update. Only with a well-preserving accept of the first PSB step.

**Next experiment:** Do not try DFP/SR1/Greenstadt.

**Attempts:**
- Cycle 138; candidate `f4eaa983252e23d1ed85a616361f46431fd12eec`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision discard.
  [Implementation](ideas/sella-psb-update/f4eaa983252e23d1ed85a616361f46431fd12eec/algo.py).
  [Training evidence](evaluation_results/cycle-138-train.json).
  [Narrative](full_log.md#cycle-138--connected-powell-symmetric-broyden-hessian-updates).

## sella-late-rfo: RFO stepper after 50 connected steps
Status: closed

**Hypothesis:** Early RFO hops. Paliperidone (78) and 160853090 (88) still have long QN tails; RFO after 50 connected steps can cheapen those tails without 135043047 (44).

**Outcome and uncertainty:** Cycle 140/170 RFO@50 discard, Δ=+9.92e-5 on both champions. Cycle 141/171/173 RFO@45 non_generalizable and bit-identical: train paliperidone 78→73 and 160853090 88→89, valid 135278208 64→63 and 135065494 93→95. Cycle 173 PD-gate at 45 ≡ always-on (same as 166 at 50). Cycle 172 window 45–79: paliperidone still 73 but 160853090 88→90, Δ=+8.88e-5. Cycle 185 RFO@50 on 180: discard, Δ=+9.92e-5, same two movers. Cycle 186 RFO@45 on 180: non_generalizable, same as 171 (valid 135065494 93→95). ρ/truncation/window/PD cannot separate the paliperidone save from 88–95-step extras; dimer GDIIS/RFO does not absorb that extra.

**Reason to revisit:** Only if a later connected method already cheapens paliperidone without RFO, so RFO is not needed on 90-step jobs. Cycle 197 showed RFO@50 extras valid 135065494 93→96 even stacked with dimer TR.

**Next experiment:** Do not interpolate RFO start, window, ρ, truncation, or PD gates. Do not restack connected RFO@50.

**Attempts:**
- Cycle 140; candidate `7e5accc8932ca7187e28e2ca71f7b1495f838a2e`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision discard.
  [Implementation](ideas/sella-late-rfo/7e5accc8932ca7187e28e2ca71f7b1495f838a2e/algo.py).
  [Training evidence](evaluation_results/cycle-140-train.json).
  [Narrative](full_log.md#cycle-140--connected-rfo-steps-after-50-steps).
- Cycle 141; candidate `212e3863b89f8cee277691a12dbf12a36df00f61`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision non_generalizable.
  [Implementation](ideas/sella-late-rfo/212e3863b89f8cee277691a12dbf12a36df00f61/algo.py).
  [Training evidence](evaluation_results/cycle-141-train.json).
  [Validation evidence](evaluation_results/cycle-141-valid.json).
  [Narrative](full_log.md#cycle-141--connected-rfo-steps-after-45-steps).
- Cycle 142; candidate `011aa7a8612e57f096fb15a6f70b10778f01b272`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision discard.
  [Implementation](ideas/sella-late-rfo/011aa7a8612e57f096fb15a6f70b10778f01b272/algo.py).
  [Training evidence](evaluation_results/cycle-142-train.json).
  [Narrative](full_log.md#cycle-142--ρ-gated-connected-rfo-after-45-steps).
- Cycle 143; candidate `0928d61a059af2d1eecf9ab9c82ca8cd04c6d56a`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision discard.
  [Implementation](ideas/sella-late-rfo/0928d61a059af2d1eecf9ab9c82ca8cd04c6d56a/algo.py).
  [Training evidence](evaluation_results/cycle-143-train.json).
  [Narrative](full_log.md#cycle-143--rfo-after-50-ρ-gated-from-45).
- Cycle 144; candidate `5bbe075e7d7f382454c1a5d2cdd7a0ce77e6e6de`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision non_generalizable.
  [Implementation](ideas/sella-late-rfo/5bbe075e7d7f382454c1a5d2cdd7a0ce77e6e6de/algo.py).
  [Training evidence](evaluation_results/cycle-144-train.json).
  [Validation evidence](evaluation_results/cycle-144-valid.json).
  [Narrative](full_log.md#cycle-144--dummy-dihedral-wd08-plus-connected-rfo-after-50).
- Cycle 154; candidate `128432891d70bb6e2b00312aa3ed3c303e7eac60`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision discard.
  [Implementation](ideas/sella-late-rfo/128432891d70bb6e2b00312aa3ed3c303e7eac60/algo.py).
  [Training evidence](evaluation_results/cycle-154-train.json).
  [Narrative](full_log.md#cycle-154--truncated-only-connected-rfo-after-50-steps).
- Cycle 155; candidate `0d12aa1867070a0a56f4abfb6c88a6515dcf3bfe`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision discard.
  [Implementation](ideas/sella-late-rfo/0d12aa1867070a0a56f4abfb6c88a6515dcf3bfe/algo.py).
  [Training evidence](evaluation_results/cycle-155-train.json).
  [Narrative](full_log.md#cycle-155--untruncated-only-connected-rfo-after-50-steps).
- Cycle 166; candidate `e06536d08eaac08a252c481821317f16e524772c`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision discard.
  [Implementation](ideas/sella-late-rfo/e06536d08eaac08a252c481821317f16e524772c/algo.py).
  [Training evidence](evaluation_results/cycle-166-train.json).
  [Narrative](full_log.md#cycle-166--rfo-after-50-connected-steps-when-hessian-is-indefinite).
- Cycle 170; candidate `f20da46eb9b6e9ecda8f4a913177fbd9076c12ba`; champion `3dd5d21702bcb0ef6314d45201113465bf17ee7f`; decision discard.
  [Implementation](ideas/sella-late-rfo/f20da46eb9b6e9ecda8f4a913177fbd9076c12ba/algo.py).
  [Training evidence](evaluation_results/cycle-170-train.json).
  [Narrative](full_log.md#cycle-170--rfo-after-50-connected-steps-on-limiter-dummy-wd).
- Cycle 171; candidate `07a8fa027a0dd5ca15bc0d3f8c116abe312d00c6`; champion `3dd5d21702bcb0ef6314d45201113465bf17ee7f`; decision non_generalizable.
  [Implementation](ideas/sella-late-rfo/07a8fa027a0dd5ca15bc0d3f8c116abe312d00c6/algo.py).
  [Training evidence](evaluation_results/cycle-171-train.json).
  [Validation evidence](evaluation_results/cycle-171-valid.json).
  [Narrative](full_log.md#cycle-171--rfo-after-45-connected-steps-on-limiter-dummy-wd).
- Cycle 172; candidate `5c1c87867395bc5c999a308b5738fe79c94c8250`; champion `3dd5d21702bcb0ef6314d45201113465bf17ee7f`; decision discard.
  [Implementation](ideas/sella-late-rfo/5c1c87867395bc5c999a308b5738fe79c94c8250/algo.py).
  [Training evidence](evaluation_results/cycle-172-train.json).
  [Narrative](full_log.md#cycle-172--rfo-only-on-connected-steps-45-79).
- Cycle 173; candidate `61233bd270dfa52d779b01d19b4f0e54f433978a`; champion `3dd5d21702bcb0ef6314d45201113465bf17ee7f`; decision non_generalizable.
  [Implementation](ideas/sella-late-rfo/61233bd270dfa52d779b01d19b4f0e54f433978a/algo.py).
  [Training evidence](evaluation_results/cycle-173-train.json).
  [Validation evidence](evaluation_results/cycle-173-valid.json).
  [Narrative](full_log.md#cycle-173--rfo-after-45-when-hessian-is-indefinite).
- Cycle 185; candidate `8aa641eed93b5efa9bf3fde1df56e27087b0a369`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-late-rfo/8aa641eed93b5efa9bf3fde1df56e27087b0a369/algo.py).
  [Training evidence](evaluation_results/cycle-185-train.json).
  [Narrative](full_log.md#cycle-185--connected-rfo-after-50-on-the-dimer-gdiisrfo-champion).
- Cycle 186; candidate `ed556f509928eaa9b7fa0552a424d998e850b1d1`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision non_generalizable.
  [Implementation](ideas/sella-late-rfo/ed556f509928eaa9b7fa0552a424d998e850b1d1/algo.py).
  [Training evidence](evaluation_results/cycle-186-train.json).
  [Validation evidence](evaluation_results/cycle-186-valid.json).
  [Narrative](full_log.md#cycle-186--connected-rfo-after-45-on-the-dimer-gdiisrfo-champion).
- Cycle 197; candidate `9be3bf061baa2ae39fe6e0914f8397e8cef9fb1b`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision non_generalizable.
  [Implementation](ideas/sella-late-rfo/9be3bf061baa2ae39fe6e0914f8397e8cef9fb1b/algo.py).
  [Training evidence](evaluation_results/cycle-197-train.json).
  [Validation evidence](evaluation_results/cycle-197-valid.json).
  [Narrative](full_log.md#cycle-197--connected-rfo50-plus-dimer-trustregion-rfo80).

## sella-late-trust-region: Euclidean internal trust after 50 connected steps
Status: closed

**Hypothesis:** After 50 connected steps, MaxInternalStep’s max-component cap is the long-tail limiter. Sella TrustRegion (`||s||≤δ`) uses the full Euclidean internal budget.

**Outcome and uncertainty:** Cycle 156 @50: paliperidone save, 160853090 extra. Cycle 157 @20: large saves (135231613 −8, paliperidone −6, venetoclax −4) eaten by 30–45 extras, Δ=−2.98e-4. Cycle 158 @40: those saves gone (they lived in 20–40). Cycle 159 ρ-gate: venetoclax 57→67. Cycle 161 cosine/length 2×: Δ=−5.49e-4; phenol–monoatomic 52→65. Cycle 162 sticky after first 2× accept: Δ=−7.87e-4; paliperidone 78→72 but phenol still 65. Cycle 163 4× per-step: Δ=−2.29e-4; paliperidone 78→75, 157’s 52/57-step saves gone, phenol still 52→55. Cycle 169 always-on TR@20 on dummy-wd: Δ=−3.51e-4, undid 135043047. Cycle 194 skip TR when dummy is limiter: discard, Δ=−2.32e-4; dummy skip kept 135092350 but phenol still 52→57 and 135043047 still +1. Energy-safe throughout.

**Reason to revisit:** Cycle 157 remains the only TR variant with large localized saves. Dummy-limiter exclusion is not enough to pay for phenol/30–45 extras. Do not restack TR@20.

**Next experiment:** Do not restack always-on or dummy-excluded TrustRegion@20 on connected molecules.

**Attempts:**
- Cycle 156; candidate `132b369e2289825ad38e1c68981a74d4a29b04c9`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision discard.
  [Implementation](ideas/sella-late-trust-region/132b369e2289825ad38e1c68981a74d4a29b04c9/algo.py).
  [Training evidence](evaluation_results/cycle-156-train.json).
  [Narrative](full_log.md#cycle-156--connected-trustregion-after-50-steps).
- Cycle 157; candidate `ddb254ae9abdf505039e7b087264ae6a0883dd67`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision discard.
  [Implementation](ideas/sella-late-trust-region/ddb254ae9abdf505039e7b087264ae6a0883dd67/algo.py).
  [Training evidence](evaluation_results/cycle-157-train.json).
  [Narrative](full_log.md#cycle-157--connected-trustregion-after-20-steps).
- Cycle 158; candidate `fb85440e8ed0f6cddd14cda5461d57203a7dd91f`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision discard.
  [Implementation](ideas/sella-late-trust-region/fb85440e8ed0f6cddd14cda5461d57203a7dd91f/algo.py).
  [Training evidence](evaluation_results/cycle-158-train.json).
  [Narrative](full_log.md#cycle-158--connected-trustregion-after-40-steps).
- Cycle 159; candidate `14ac0a6208315daf27f91e88bd7c21e9b68e4888`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision discard.
  [Implementation](ideas/sella-late-trust-region/14ac0a6208315daf27f91e88bd7c21e9b68e4888/algo.py).
  [Training evidence](evaluation_results/cycle-159-train.json).
  [Narrative](full_log.md#cycle-159--ρ-gated-connected-trustregion-after-20-steps).
- Cycle 161; candidate `91e6f83a4d79332baac54f3a1c40dbef7d640404`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision discard.
  [Implementation](ideas/sella-late-trust-region/91e6f83a4d79332baac54f3a1c40dbef7d640404/algo.py).
  [Training evidence](evaluation_results/cycle-161-train.json).
  [Narrative](full_log.md#cycle-161--cosinelength-accepted-trustregion-vs-mis-after-20).
- Cycle 162; candidate `b9973b8fbcfceed147140cb52e682a0b1578ca1c`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision discard.
  [Implementation](ideas/sella-late-trust-region/b9973b8fbcfceed147140cb52e682a0b1578ca1c/algo.py).
  [Training evidence](evaluation_results/cycle-162-train.json).
  [Narrative](full_log.md#cycle-162--sticky-trustregion-after-first-cosinelength-accept).
- Cycle 163; candidate `f4e485dd21c2bc5af0353d85d279a8b4d71bd322`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision discard.
  [Implementation](ideas/sella-late-trust-region/f4e485dd21c2bc5af0353d85d279a8b4d71bd322/algo.py).
  [Training evidence](evaluation_results/cycle-163-train.json).
  [Narrative](full_log.md#cycle-163--4-length-accepted-trustregion-vs-mis-after-20).
- Cycle 169; candidate `920a2e95029a3cf51417392f8e5b14ca5cc80be7`; champion `3dd5d21702bcb0ef6314d45201113465bf17ee7f`; decision discard.
  [Implementation](ideas/sella-late-trust-region/920a2e95029a3cf51417392f8e5b14ca5cc80be7/algo.py).
  [Training evidence](evaluation_results/cycle-169-train.json).
  [Narrative](full_log.md#cycle-169--connected-trustregion-after-20-on-limiter-dummy-wd).
- Cycle 194; candidate `dfada3c46e8c55fbef5a0c6fe9e39915520eb3d0`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-late-trust-region/dfada3c46e8c55fbef5a0c6fe9e39915520eb3d0/algo.py).
  [Training evidence](evaluation_results/cycle-194-train.json).
  [Narrative](full_log.md#cycle-194--trustregion-after-20-except-dummy-dihedral-limiter).

## sella-poly-ls: quartic line-search fallback after GDIIS
Status: closed

**Hypothesis:** When ρ-gated two-point GDIIS rejects, a pysisyphus constrained quartic along the last internal step can still cheapen well-predicted connected tails without extra force calls.

**Outcome and uncertainty:** Cycle 164 invalid. Paliperidone 78→28 (+1.65 kcal) and 160853090 88→40 (+2.13 kcal). Extrapolation 1<x≤2 hops the long tails into higher wells, like cycle 126 GEDIIS.

**Reason to revisit:** Only with a well-preserving limiter that is *not* an energy interpolant on the last segment. Do not try 0<x<1; GEDIIS already hopped there.

**Next experiment:** Do not add cubic/quintic poly LS. See C2-DIIS GDIIS.

**Attempts:**
- Cycle 164; candidate `3b7d28b333483a8e9c405f0bf8a976fc905db590`; champion `d6fb6dd73ad4c7fe0182c54136abae56aa5168f8`; decision invalid.
  [Implementation](ideas/sella-poly-ls/3b7d28b333483a8e9c405f0bf8a976fc905db590/algo.py).
  [Training evidence](evaluation_results/cycle-164-train.json).
  [Narrative](full_log.md#cycle-164--quartic-line-search-fallback-after-gdiis-reject).

## sella-dimer-gdiis: champion two-point GDIIS on dimers
Status: incorporated

**Hypothesis:** Connected GDIIS is a keep; dimers never interpolate. The same two-point ρ/cosine/||s||≤QN GDIIS after 20 steps can skip redundant Newton steps on long DES jobs without changing dummy-wd or connected floors.

**Outcome and uncertainty:** Cycle 174 discard, Δ=+8.13e-5, energy-safe (three medium DES −1, amides–pyridine +1). Cycle 175 nsteps≥10: n_steps bit-identical to 174. Cycle 176 no ρ window: invalid, amines–amines 99→96 (+0.0036 kcal). Cycle 177 cosine 0.80 with ρ: invalid, acids–benzene 54→28 (+0.016 kcal). Cycle 180 stacked with RFO@80: **keep**. Cycle 181 stop GDIIS at 80: discard, Δ=0. Cycle 184 cap trans/rot to QN: discard, Δ=0. Cycle 191 three-then-two: discard, Δ=−9.28e-5, energy-safe; alcohols–alkanes 68→62, alkenes–alkenes 70→73, water–water 45→49. Cycle 192 two-then-three: n_steps bit-identical to 191. Cycle 201 C2 (allow c_i<0) on dimers: invalid, amines–amines 99→23 (R=0.131). Cycle 204 no ρ after 80: discard, Δ=−2.01e-5, amides–water 106→107, amines unchanged.

**Reason to revisit:** Incorporated in cycle 180. Loosening cosine/ρ/start hops packing; do not retry those gates. Do not allow negative Pulay coefficients on dimers. Do not drop ρ after 80.

**Next experiment:** Do not add three-point dimer C1 or C2-DIIS on dimers. Do not drop the ρ window on the RFO tail.

**Attempts:**
- Cycle 174; candidate `bcc66276ef1f714888c79769b428219a9d4299ad`; champion `3dd5d21702bcb0ef6314d45201113465bf17ee7f`; decision discard.
  [Implementation](ideas/sella-dimer-gdiis/bcc66276ef1f714888c79769b428219a9d4299ad/algo.py).
  [Training evidence](evaluation_results/cycle-174-train.json).
  [Narrative](full_log.md#cycle-174--champion-gdiis-on-dimers).
- Cycle 175; candidate `46abd86ed3e39872c7f7799cf452f0fb3306d1fc`; champion `3dd5d21702bcb0ef6314d45201113465bf17ee7f`; decision discard.
  [Implementation](ideas/sella-dimer-gdiis/46abd86ed3e39872c7f7799cf452f0fb3306d1fc/algo.py).
  [Training evidence](evaluation_results/cycle-175-train.json).
  [Narrative](full_log.md#cycle-175--dimer-gdiis-after-10-steps).
- Cycle 176; candidate `631e3c3212dc0c946ae11dd29882fcbcd023ebe6`; champion `3dd5d21702bcb0ef6314d45201113465bf17ee7f`; decision invalid.
  [Implementation](ideas/sella-dimer-gdiis/631e3c3212dc0c946ae11dd29882fcbcd023ebe6/algo.py).
  [Training evidence](evaluation_results/cycle-176-train.json).
  [Narrative](full_log.md#cycle-176--dimer-gdiis-without-the-ρ-window).
- Cycle 177; candidate `1af88884ef17b1068251d49c593b316a1cd5ae3d`; champion `3dd5d21702bcb0ef6314d45201113465bf17ee7f`; decision invalid.
  [Implementation](ideas/sella-dimer-gdiis/1af88884ef17b1068251d49c593b316a1cd5ae3d/algo.py).
  [Training evidence](evaluation_results/cycle-177-train.json).
  [Narrative](full_log.md#cycle-177--ρ-gated-dimer-gdiis-with-cosine-080).
- Cycle 180; candidate `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; champion `3dd5d21702bcb0ef6314d45201113465bf17ee7f`; decision keep.
  [Training evidence](evaluation_results/cycle-180-train.json).
  [Validation evidence](evaluation_results/cycle-180-valid.json).
  [Narrative](full_log.md#cycle-180--dimer-gdiis-plus-rfo-after-80-dimer-steps).
- Cycle 181; candidate `a5ad75cc117ac9878463f93a79146b61872137f0`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-dimer-gdiis/a5ad75cc117ac9878463f93a79146b61872137f0/algo.py).
  [Training evidence](evaluation_results/cycle-181-train.json).
  [Narrative](full_log.md#cycle-181--dimer-gdiis-only-before-step-80).
- Cycle 184; candidate `369974a2fdec0152fc82687cce1bc7e57d0e1dc5`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-dimer-gdiis/369974a2fdec0152fc82687cce1bc7e57d0e1dc5/algo.py).
  [Training evidence](evaluation_results/cycle-184-train.json).
  [Narrative](full_log.md#cycle-184--cap-dimer-gdiis-translationrotation-blocks).
- Cycle 191; candidate `bfe32e94388d979421bbc51fd1dfc9c3a910de67`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-dimer-gdiis/bfe32e94388d979421bbc51fd1dfc9c3a910de67/algo.py).
  [Training evidence](evaluation_results/cycle-191-train.json).
  [Narrative](full_log.md#cycle-191--three-point-dimer-gdiis-then-two-point-fallback).
- Cycle 192; candidate `fc95914f42be61623969869a48888cf82e88f2c5`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-dimer-gdiis/fc95914f42be61623969869a48888cf82e88f2c5/algo.py).
  [Training evidence](evaluation_results/cycle-192-train.json).
  [Narrative](full_log.md#cycle-192--two-point-dimer-gdiis-then-three-point-fallback).
- Cycle 201; candidate `5709093b3a89bc5ee81efeace54be4cfe4bf5ccc`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision invalid.
  [Implementation](ideas/sella-dimer-gdiis/5709093b3a89bc5ee81efeace54be4cfe4bf5ccc/algo.py).
  [Training evidence](evaluation_results/cycle-201-train.json).
  [Narrative](full_log.md#cycle-201--c2-diis-two-point-gdiis-on-dimers).
- Cycle 204; candidate `88e9f0db274445f4d8a1c9f67611e70f508102fd`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-dimer-gdiis/88e9f0db274445f4d8a1c9f67611e70f508102fd/algo.py).
  [Training evidence](evaluation_results/cycle-204-train.json).
  [Narrative](full_log.md#cycle-204--dimer-gdiis-without-the-ρ-window-after-80-steps).

## sella-dimer-late-rfo: RFO stepper after 80 dimer steps
Status: incorporated

**Hypothesis:** Always-on dimer RFO hopped (cycle 63). After 80 dimer steps, RFO can cheapen 95–106-call DES tails without early packing RFO or connected dummy-wd.

**Outcome and uncertainty:** Cycle 178 discard, Δ=+4.49e-5, ethers–ethers 95→93 energy-safe. Cycle 179 RFO@70: pyrrole–sulfides 88→86 but amides–water 106→109, Δ=+1.08e-5. Cycle 180 stacked with dimer GDIIS: **keep**.

**Reason to revisit:** Incorporated. Do not start dimer RFO before 80.

**Next experiment:** Close skip-GDIIS@80 on dimers (cycle 428 bit-identical). Do not drop the dimer ρ window after 80 (cycle 204). See connected-only `exact_geodesic`.

**Attempts:**
- Cycle 428; candidate `0a06be2932a59822b51f80e9ff897c184c1a1c36`; champion `d2f7137d0609977f7ebd973704e68b8f419f8dad`; decision discard.
  [Training evidence](evaluation_results/cycle-428-train.json).
  [Narrative](full_log.md#cycle-428--skip-dimer-gdiis-after-80-so-rfo-is-not-overwritten).
- Cycle 178; candidate `31ae756e5fd20b4e331576d8c5291fcdadae58e7`; champion `3dd5d21702bcb0ef6314d45201113465bf17ee7f`; decision discard.
  [Implementation](ideas/sella-dimer-late-rfo/31ae756e5fd20b4e331576d8c5291fcdadae58e7/algo.py).
  [Training evidence](evaluation_results/cycle-178-train.json).
  [Narrative](full_log.md#cycle-178--rfo-after-80-dimer-steps).
- Cycle 179; candidate `c1eb77342edded6af06a103c6eb8420d67efb5a5`; champion `3dd5d21702bcb0ef6314d45201113465bf17ee7f`; decision discard.
  [Implementation](ideas/sella-dimer-late-rfo/c1eb77342edded6af06a103c6eb8420d67efb5a5/algo.py).
  [Training evidence](evaluation_results/cycle-179-train.json).
  [Narrative](full_log.md#cycle-179--rfo-after-70-dimer-steps).
- Cycle 180; candidate `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; champion `3dd5d21702bcb0ef6314d45201113465bf17ee7f`; decision keep.
  [Training evidence](evaluation_results/cycle-180-train.json).
  [Validation evidence](evaluation_results/cycle-180-valid.json).
  [Narrative](full_log.md#cycle-180--dimer-gdiis-plus-rfo-after-80-dimer-steps).

## sella-dimer-late-trust-region: Euclidean trust on late dimer steps
Status: revisiting

**Hypothesis:** Late dimer RFO is still MIS-truncated. Euclidean TrustRegion on that tail can take more of the RFO (and late QN) step without early packing TR.

**Outcome and uncertainty:** Cycle 195 TR+RFO@80: discard, Δ=+6.87e-5, energy-safe, alkanes–benzene 96→94 and pyrrole–sulfides 88→87, no extras. Cycle 196 TR@70 + RFO@80: discard, Δ=+8.13e-5, amides–water 106→108 and alkanes–esters 96→97 extras.

**Reason to revisit:** TR+RFO@80 is a train-only near-miss; valid alkanes–water 95→130 undoes the cycle-180 keep. Do not start dimer TR before 80. Do not stack with connected RFO@50 (valid 135065494).

**Next experiment:** Do not take dimer Euclidean RFO trust to validation.

**Attempts:**
- Cycle 195; candidate `74b84c5b94fb6a98e3dbd3f7bc6346cb5ea092dc`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-dimer-late-trust-region/74b84c5b94fb6a98e3dbd3f7bc6346cb5ea092dc/algo.py).
  [Training evidence](evaluation_results/cycle-195-train.json).
  [Narrative](full_log.md#cycle-195--trustregion-on-dimer-rfo-after-80-steps).
- Cycle 196; candidate `47892113436396d91473bfbd5b4bf4ad322d848a`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-dimer-late-trust-region/47892113436396d91473bfbd5b4bf4ad322d848a/algo.py).
  [Training evidence](evaluation_results/cycle-196-train.json).
  [Narrative](full_log.md#cycle-196--trustregion-after-70-dimer-steps-rfo-still-at-80).
- Cycle 197; candidate `9be3bf061baa2ae39fe6e0914f8397e8cef9fb1b`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision non_generalizable.
  [Implementation](ideas/sella-dimer-late-trust-region/9be3bf061baa2ae39fe6e0914f8397e8cef9fb1b/algo.py).
  [Training evidence](evaluation_results/cycle-197-train.json).
  [Validation evidence](evaluation_results/cycle-197-valid.json).
  [Narrative](full_log.md#cycle-197--connected-rfo50-plus-dimer-trustregion-rfo80).

## sella-dimer-wx: milder dimer translation MaxInternalStep weight
Status: deferred

**Hypothesis:** Connecting stretches, not translations, are the dimer MIS limiter. Downweighting translations after packing (wx=0.8, nsteps≥20) is a no-op like dummy-angle `wa`.

**Outcome and uncertainty:** Cycle 189 discard, Δ=0, bit-identical.

**Reason to revisit:** Not as a wx tweak. The MIS limiter on dimers is the connecting stretch.

**Next experiment:** Do not interpolate dimer `wx`.

**Attempts:**
- Cycle 189; candidate `2e4e0c3b464f099032060b2690b553b173524104`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-dimer-wx/2e4e0c3b464f099032060b2690b553b173524104/algo.py).
  [Training evidence](evaluation_results/cycle-189-train.json).
  [Narrative](full_log.md#cycle-189--dimer-translation-weight-wx08-after-20-steps).

## sella-dimer-late-wd: milder dimer dihedral weight after 80 steps
Status: closed

**Hypothesis:** Late dimer RFO tails are torsion-limited under MIS. `wd=0.8` after 80 enlarges packing dihedrals without connecting `wb`.

**Outcome and uncertainty:** Cycle 205 discard, n_steps bit-identical. Dihedrals are not the late dimer MIS limiter.

**Reason to revisit:** Not as a `wd` tweak. Connecting stretches remain the likely limiter; enlarging them hops (cycle 193).

**Next experiment:** Do not interpolate dimer `wd`.

**Attempts:**
- Cycle 205; candidate `162641795f6774052a6563e0384e866ebc640029`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-dimer-late-wd/162641795f6774052a6563e0384e866ebc640029/algo.py).
  [Training evidence](evaluation_results/cycle-205-train.json).
  [Narrative](full_log.md#cycle-205--dimer-maxinternalstep-wd08-after-80-steps).

## sella-dimer-flowchart-hessian: Schlegel flowchart Hessian update on dimers
Status: closed

**Hypothesis:** TS-BFGS on packing secants is the wrong update. Schlegel flowchart (SR1 / BFGS / PSB) matches the secant after 20 dimer steps without touching connected TS-BFGS.

**Outcome and uncertainty:** Cycle 202 invalid (amines hop at 48). Cycle 207 after 80: energy-safe, Δ=−1.79e-4, ethers–ethers 93→108. Cycle 208 ρ-gated after 80: Δ=+5.09e-5, ethers extra gone, benzene–water 85→94. Cycle 209 ρ-gated after 86: Δ=+6.87e-5, benzene–water extra gone, pyrrole save gone. Cycle 210 BFGS-only ρ-gated after 80: Δ=+4.65e-5; benzene–water extra gone, 208 SR1/PSB saves gone except alkanes–esters −3.

**Reason to revisit:** Not as a discrete flowchart. The 208 saves and the benzene–water extra are the same SR1/PSB arms; nsteps and the BFGS arm cannot split them. See Bofill mixing.

**Next experiment:** Do not interpolate flowchart nsteps, ρ, or SR1/BFGS/PSB arms. Try continuous Bofill SR1/PSB mixing on the same late dimer tail.

**Attempts:**
- Cycle 202; candidate `efeb93336972552d441d37c13e8a44813ee4c3f1`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision invalid.
  [Implementation](ideas/sella-dimer-flowchart-hessian/efeb93336972552d441d37c13e8a44813ee4c3f1/algo.py).
  [Training evidence](evaluation_results/cycle-202-train.json).
  [Narrative](full_log.md#cycle-202--schlegel-flowchart-hessian-update-on-dimers-after-20-steps).
- Cycle 207; candidate `1a46741770d1e2b55c5ff453158c5b9ac3530056`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-dimer-flowchart-hessian/1a46741770d1e2b55c5ff453158c5b9ac3530056/algo.py).
  [Training evidence](evaluation_results/cycle-207-train.json).
  [Narrative](full_log.md#cycle-207--schlegel-flowchart-hessian-update-on-dimers-after-80-steps).
- Cycle 208; candidate `90f70a1940e8057a3680dfafcc12308474153887`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-dimer-flowchart-hessian/90f70a1940e8057a3680dfafcc12308474153887/algo.py).
  [Training evidence](evaluation_results/cycle-208-train.json).
  [Narrative](full_log.md#cycle-208--ρ-gated-flowchart-hessian-on-dimers-after-80-steps).
- Cycle 209; candidate `ff0b82e52fe2a1c64ed3b7ba0ebd4bb783193df2`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-dimer-flowchart-hessian/ff0b82e52fe2a1c64ed3b7ba0ebd4bb783193df2/algo.py).
  [Training evidence](evaluation_results/cycle-209-train.json).
  [Narrative](full_log.md#cycle-209--ρ-gated-flowchart-hessian-on-dimers-after-86-steps).
- Cycle 210; candidate `b53e9dfaa82d30e4f221a964a6d9aa5b9198e315`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-dimer-flowchart-hessian/b53e9dfaa82d30e4f221a964a6d9aa5b9198e315/algo.py).
  [Training evidence](evaluation_results/cycle-210-train.json).
  [Narrative](full_log.md#cycle-210--ρ-gated-bfgs-only-flowchart-on-dimers-after-80-steps).

## sella-dimer-bofill-hessian: Bofill SR1/PSB mix on late dimer Hessians
Status: closed

**Hypothesis:** Discrete flowchart SR1/PSB jumps (cycle 208) both save packing tails and hop benzene–water. Bofill’s continuous mix on the same ρ-gated RFO tail can keep mixed-curvature updates without a pure SR1 or PSB step.

**Outcome and uncertainty:** Cycle 211 discard, energy-safe, Δ=−2.80e-5. Pyrrole −4 and alkanes–esters −2; extras amides–water +4, ethers +3, amines +1. Cycles 212–213 φ bands: bit-identical to 180; all 211 movers were near-pure PSB (φ≤0.2). Cycle 214 pure PSB on φ<0.5: same saves, worse extras (ethers +5, amides–water +9). Residual SR1 in the 211 mix reduced extras.

**Reason to revisit:** Not as an SR1/PSB mixing formula. The PSB-side packing updates cannot split pyrrole/esters saves from ethers/amides extras.

**Next experiment:** Do not interpolate Bofill φ. Freeze or reset the dimer Hessian at the RFO switch instead.

**Attempts:**
- Cycle 211; candidate `5ec62c2603dca8322311458756c7a5e672c88f1a`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-dimer-bofill-hessian/5ec62c2603dca8322311458756c7a5e672c88f1a/algo.py).
  [Training evidence](evaluation_results/cycle-211-train.json).
  [Narrative](full_log.md#cycle-211--ρ-gated-bofill-hessian-on-dimers-after-80-steps).
- Cycle 212; candidate `a88f0126bfe9b63df9118d689ed613606f7eed57`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-dimer-bofill-hessian/a88f0126bfe9b63df9118d689ed613606f7eed57/algo.py).
  [Training evidence](evaluation_results/cycle-212-train.json).
  [Narrative](full_log.md#cycle-212--intermediate-φ-bofill-hessian-on-dimers-after-80-steps).
- Cycle 213; candidate `7b79ebbe779753dafb5a3f26f5e20bcb1b523647`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-dimer-bofill-hessian/7b79ebbe779753dafb5a3f26f5e20bcb1b523647/algo.py).
  [Training evidence](evaluation_results/cycle-213-train.json).
  [Narrative](full_log.md#cycle-213--sr1-dominant-bofill-hessian-on-dimers-after-80-steps).
- Cycle 214; candidate `bff7cf632cbf89b2dcca6c011d57e306e441496a`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-dimer-bofill-hessian/bff7cf632cbf89b2dcca6c011d57e306e441496a/algo.py).
  [Training evidence](evaluation_results/cycle-214-train.json).
  [Narrative](full_log.md#cycle-214--psb-hessian-on-near-psb-dimer-secants-after-80-steps).

## sella-dimer-freeze-hessian: freeze dimer TS-BFGS at the RFO switch
Status: closed

**Hypothesis:** Noisy Hessian updates after 80 on packing secants both save and extra DES tails. Freezing H at the RFO switch keeps packed TS-BFGS and skips those updates.

**Outcome and uncertainty:** Cycle 215 invalid. Energy misses on the unofficial cheaper tails and 200-call stalls on amides–water / ethers / benzene–water. Updates after 80 are required.

**Reason to revisit:** Not as a freeze. A one-shot H0 reset at 80 still allows later TS-BFGS.

**Next experiment:** Do not freeze dimer Hessian updates after 80. Try a one-shot model-H0 reset at the RFO switch.

**Attempts:**
- Cycle 215; candidate `74d9830f8a567c6f8cf91441ddf46332694b5452`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision invalid.
  [Implementation](ideas/sella-dimer-freeze-hessian/74d9830f8a567c6f8cf91441ddf46332694b5452/algo.py).
  [Training evidence](evaluation_results/cycle-215-train.json).
  [Narrative](full_log.md#cycle-215--freeze-dimer-hessian-at-the-rfo-switch-after-80-steps).

## sella-dimer-h0-reset-80: model Hessian reset at the dimer RFO switch
Status: closed

**Hypothesis:** Packed TS-BFGS at step 80 is the wrong curvature for Banerjee RFO. Replacing B once with current-geometry model H0, then continuing TS-BFGS, cheapens 80+ DES tails without freezing updates (cycle 215) or SR1/PSB mixing (211–214).

**Outcome and uncertainty:** Cycle 216 invalid, unofficial Δ=+1.52e-4 from false wells. Cycle 217 ρ-gate: unofficial Δ=+4.56e-4, extras gone, alkanes–benzene and pyrrole still hop. Cycle 218 QN-first ≡ 217. Cycle 219 restore packed H after one H0 step ≡ 217. The hop is the H0 displacement.

**Reason to revisit:** Not as an H0 reset on the RFO tail. Cycle 87’s energy-checked accept cannot help if the first H0 step already leaves the well.

**Next experiment:** Do not reset dimer H0 at 80. See negative-curvature update skip on the RFO tail.

**Hypothesis:** Packed TS-BFGS at step 80 is the wrong curvature for Banerjee RFO. Replacing B once with current-geometry model H0, then continuing TS-BFGS, cheapens 80+ DES tails without freezing updates (cycle 215) or SR1/PSB mixing (211–214).

**Outcome and uncertainty:** Cycle 216 invalid. Cycle 217 ρ-gate dropped extras; alkanes–benzene and pyrrole still hop. Cycle 218 QN-first ≡ 217. The hop is the H0 matrix, not the first RFO vs QN step.

**Reason to revisit:** Use H0 for one displacement only, then restore packed TS-BFGS.

**Next experiment:** Cycle 219: ρ-gated H0 step at 80, then restore the saved TS-BFGS Hessian.

**Attempts:**
- Cycle 216; candidate `9f5a0babcc7f92c65e2420f1b5dd9681f1481117`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision invalid.
  [Implementation](ideas/sella-dimer-h0-reset-80/9f5a0babcc7f92c65e2420f1b5dd9681f1481117/algo.py).
  [Training evidence](evaluation_results/cycle-216-train.json).
  [Narrative](full_log.md#cycle-216--model-h0-reset-at-the-dimer-rfo-switch-after-80-steps).
- Cycle 217; candidate `077fa38633b87689b001b8a683bd91e2b81284b3`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision invalid.
  [Implementation](ideas/sella-dimer-h0-reset-80/077fa38633b87689b001b8a683bd91e2b81284b3/algo.py).
  [Training evidence](evaluation_results/cycle-217-train.json).
  [Narrative](full_log.md#cycle-217--ρ-gated-model-h0-reset-at-the-dimer-rfo-switch).
- Cycle 218; candidate `85ff249f7b1ba3b32637ae5c3719f9b2f9c8056b`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision invalid.
  [Implementation](ideas/sella-dimer-h0-reset-80/85ff249f7b1ba3b32637ae5c3719f9b2f9c8056b/algo.py).
  [Training evidence](evaluation_results/cycle-218-train.json).
  [Narrative](full_log.md#cycle-218--ρ-gated-h0-reset-with-qn-first-step-then-rfo-from-81).
- Cycle 219; candidate `def7c486bef29d625ce573cd6ab92abc81b17ed1`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision invalid.
  [Implementation](ideas/sella-dimer-h0-reset-80/def7c486bef29d625ce573cd6ab92abc81b17ed1/algo.py).
  [Training evidence](evaluation_results/cycle-219-train.json).
  [Narrative](full_log.md#cycle-219--one-ρ-gated-h0-step-then-restore-packed-ts-bfgs).

## sella-dimer-skip-neg-curv: skip s·y<0 TS-BFGS on dimer RFO tails
Status: closed

**Hypothesis:** Indefinite packing secants after 80 should not update TS-BFGS. Skipping s·y<0 on dimers only, after RFO starts, avoids H0 hops and SR1/PSB extras.

**Outcome and uncertainty:** Cycle 220 discard, n_steps bit-identical to cycle 180. Negative-curvature skips do not fire on the RFO tail.

**Reason to revisit:** Not as a skip. Those tails are not s·y<0 limited.

**Next experiment:** Do not skip s·y<0 on dimer RFO tails. Floor late dimer trust instead.

**Attempts:**
- Cycle 220; candidate `4427af7e0d1d2be777a897a056a8ade1916e26fe`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-dimer-skip-neg-curv/4427af7e0d1d2be777a897a056a8ade1916e26fe/algo.py).
  [Training evidence](evaluation_results/cycle-220-train.json).
  [Narrative](full_log.md#cycle-220--skip-negative-curvature-hessian-updates-on-dimer-rfo-tails).

## sella-dimer-late-delta-min: raise dimer trust floor after 80 steps
Status: closed

**Hypothesis:** Late dimer RFO is truncated because δ sits near 0.02. Flooring δ at 0.10 after 80 lets Banerjee take more of the step without early TR.

**Outcome and uncertainty:** Cycle 221 floor after kick: bit-identical. Cycle 222 floor before RFO solve: bit-identical. δ is already ≥0.10 at nsteps≥80.

**Reason to revisit:** Not as a `delta_min` tweak on the RFO tail.

## sella-dimer-indefinite-rfo: RFO after 80 only on indefinite dimer Hessians
Status: closed

**Hypothesis:** PD packing tails should stay QN; RFO only when min λ<−1e-8.

**Outcome and uncertainty:** Cycle 223 discard, n_steps bit-identical. All dimer Hessians after 80 are indefinite.

**Reason to revisit:** Not as a PD gate on this tail.

**Next experiment:** Do not PD-gate dimer RFO@80. Try iterative Cartesian realization of those RFO steps.

**Attempts:**
- Cycle 223; candidate `ae05f89347767eaec9ee68f3c71c635e86af11e6`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-dimer-indefinite-rfo/ae05f89347767eaec9ee68f3c71c635e86af11e6/algo.py).
  [Training evidence](evaluation_results/cycle-223-train.json).
  [Narrative](full_log.md#cycle-223--indefinite-only-rfo-on-dimers-after-80-steps).

## sella-dimer-late-iterative-stepper: iterative Cartesian realization of dimer RFO after 80
Status: closed

**Hypothesis:** Cycle 62’s dimer `iterative_stepper=1` from step 0 was cost-negative because it changed packing. Enabling it only after Banerjee RFO@80 leaves early ODE packing unchanged.

**Outcome and uncertainty:** Cycle 224 discard, n_steps bit-identical. Seven 80+ DES energy wiggles only. Delayed iterative realization does not change force-call counts.

**Reason to revisit:** Only as an ODE-timeout fallback, not to reduce force calls.

**Next experiment:** Do not delay `iterative_stepper=1` on dimer RFO tails. Try Schlegel flowchart Hessian updates on connected molecules.

**Attempts:**
- Cycle 224; candidate `c62a34d71b072abdf3ca0ac55ed3165d81c7525b`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-dimer-late-iterative-stepper/c62a34d71b072abdf3ca0ac55ed3165d81c7525b/algo.py).
  [Training evidence](evaluation_results/cycle-224-train.json).
  [Narrative](full_log.md#cycle-224--iterative-cartesian-realization-of-dimer-rfo-steps-after-80).

## sella-connected-flowchart-hessian: Schlegel flowchart Hessian on connected molecules
Status: closed

**Hypothesis:** Covalent TS-BFGS leftover (363892164 46/31) needs SR1/BFGS/PSB matching; dimer packing must stay on TS-BFGS.

**Outcome and uncertainty:** Cycle 225 discard. Cost-negative; paliperidone/venetoclax/384221749 stall. Same class as cycle 94 BFGS after 20.

**Reason to revisit:** Not as a connected flowchart/BFGS/PSB/SR1 swap after 20.

**Next experiment:** Do not replace connected TS-BFGS with flowchart. Floor tiny Hessian |λ| on dimer RFO after 80 instead.

**Attempts:**
- Cycle 225; candidate `557e92007671bbcc9fe38f542f74ae870fe78134`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-connected-flowchart-hessian/557e92007671bbcc9fe38f542f74ae870fe78134/algo.py).
  [Training evidence](evaluation_results/cycle-225-train.json).
  [Narrative](full_log.md#cycle-225--schlegel-flowchart-hessian-on-connected-molecules-after-20-steps).

## sella-dimer-rfo-eval-floor: Helgaker |λ| floor on dimer RFO after 80
Status: closed

**Hypothesis:** Near-null packing modes make late Banerjee steps too large. Flooring |λ| at 0.001 Eh inside RFO damps those modes without touching connected QN.

**Outcome and uncertainty:** Cycle 226 discard, energy-safe, Δ=−5.03e-4. Pyrrole unofficial −6; extras on the other 80+ DES. Same lengthening as cycle 131.

**Reason to revisit:** Not as a |λ| floor on dimer RFO.

**Next experiment:** Do not interpolate the RFO eigenvalue floor. Skip negative-curvature Hessian updates on connected molecules instead.

**Attempts:**
- Cycle 226; candidate `b8e5303188fc72a08f7d2709ed67a2c43d250b2e`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-dimer-rfo-eval-floor/b8e5303188fc72a08f7d2709ed67a2c43d250b2e/algo.py).
  [Training evidence](evaluation_results/cycle-226-train.json).
  [Narrative](full_log.md#cycle-226--hessian-λ-floor-on-dimer-rfo-after-80-steps).

## sella-connected-skip-neg-curv: skip s·y<0 TS-BFGS on connected molecules
Status: closed

**Hypothesis:** Connected QN tails take negative-curvature updates that dimer RFO tails did not. Nocedal–Wright skip on connected only.

**Outcome and uncertainty:** Cycle 227 discard, Δ=−1.53e-3. Paliperidone 78→76 vs 160853090 88→98. Reproduces cycle 101 on the cycle-180 champion.

**Reason to revisit:** Not as a connected s·y<0 skip.

**Next experiment:** Do not skip connected negative-curvature updates. Powell-damp dimer RFO secants after 80.

**Attempts:**
- Cycle 227; candidate `ba359d2540e0d35146b90db416408902b62d2724`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-connected-skip-neg-curv/ba359d2540e0d35146b90db416408902b62d2724/algo.py).
  [Training evidence](evaluation_results/cycle-227-train.json).
  [Narrative](full_log.md#cycle-227--skip-negative-curvature-hessian-updates-on-connected-molecules).

## sella-dimer-powell-damp: Powell-damped TS-BFGS on dimer RFO after 80
Status: closed

**Hypothesis:** Weak-positive packing secants after 80 should be Powell-mixed; s·y<0 skip did not fire.

**Outcome and uncertainty:** Cycle 228 discard, bit-identical. s·y is never below 0.2 s·B·s on that tail.

**Reason to revisit:** Not as Powell damping on dimer RFO.

**Next experiment:** Do not Powell-damp dimer RFO secants. Try Newton-metric GDIIS after 80.

**Attempts:**
- Cycle 228; candidate `0631c592dedcc5122f4b5002f46cbb0373629076`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-dimer-powell-damp/0631c592dedcc5122f4b5002f46cbb0373629076/algo.py).
  [Training evidence](evaluation_results/cycle-228-train.json).
  [Narrative](full_log.md#cycle-228--powell-damped-ts-bfgs-on-dimer-rfo-after-80-steps).

## sella-dimer-newton-gdiis: Newton-metric GDIIS on dimer RFO after 80
Status: closed

**Hypothesis:** Gradient-GDIIS never accepts after 80. Newton residuals can unlock interpolants on those tails without changing connected GDIIS.

**Outcome and uncertainty:** Cycle 229 discard, bit-identical. Late dimer GDIIS still rejects in the Newton metric.

**Reason to revisit:** Not as a residual metric after 80.

**Next experiment:** Do not change GDIIS residuals on the RFO tail. Try limiter-only improper `wo` on connected molecules.

**Attempts:**
- Cycle 229; candidate `7cfcb17762c3c9f6411c6731add25f40fcf6eb93`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-dimer-newton-gdiis/7cfcb17762c3c9f6411c6731add25f40fcf6eb93/algo.py).
  [Training evidence](evaluation_results/cycle-229-train.json).
  [Narrative](full_log.md#cycle-229--newton-metric-gdiis-on-dimers-after-80-steps).

## sella-connected-limiter-wo: limiter-only improper MaxInternalStep weight
Status: closed

**Hypothesis:** Existing `nother` impropers can be the MIS limiter on leftover connected organics. Cycle-168-style downweight of that index only.

**Outcome and uncertainty:** Cycle 230 discard, bit-identical. `nother` is never the MIS limiter.

**Reason to revisit:** Not as a `wo` cap.

**Next experiment:** Do not interpolate `wo`. Clear GDIIS history at the dimer RFO switch instead.

**Attempts:**
- Cycle 230; candidate `512ac325325cef577cf0a0f8be843aa14c5b381b`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-connected-limiter-wo/512ac325325cef577cf0a0f8be843aa14c5b381b/algo.py).
  [Training evidence](evaluation_results/cycle-230-train.json).
  [Narrative](full_log.md#cycle-230--limiter-only-improper-weight-wo08-on-connected-molecules).

## sella-dimer-gdiis-reset-80: clear GDIIS history at dimer RFO switch
Status: closed

**Hypothesis:** Mixed QN/RFO residuals block GDIIS after 80. Flushing history at the switch lets two RFO points interpolate.

**Outcome and uncertainty:** Cycle 231 discard, bit-identical. Clean RFO history still does not accept.

**Reason to revisit:** Not as a history flush after 80.

**Next experiment:** Do not reset GDIIS at RFO. Try a slightly longer dimer interpolant cap.

**Attempts:**
- Cycle 231; candidate `368fe268a32c936f2d20f213f3b550ddc2675002`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-dimer-gdiis-reset-80/368fe268a32c936f2d20f213f3b550ddc2675002/algo.py).
  [Training evidence](evaluation_results/cycle-231-train.json).
  [Narrative](full_log.md#cycle-231--clear-gdiis-history-at-the-dimer-rfo-switch).

## sella-dimer-gdiis-long: slightly longer dimer GDIIS interpolants
Status: closed

**Hypothesis:** Champion ||s||≤QN rejects useful packing interpolants that are still cosine-aligned. 1.15× length on dimers only.

**Outcome and uncertainty:** Cycle 232 discard, bit-identical. Length cap is not the reject reason.

**Reason to revisit:** Not as a longer interpolant.

**Next experiment:** Do not loosen dimer GDIIS length. Try iterative Cartesian realization of connected tails.

**Attempts:**
- Cycle 232; candidate `51a4822a813af0e4dcf403109c9944e4051f8d26`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-dimer-gdiis-long/51a4822a813af0e4dcf403109c9944e4051f8d26/algo.py).
  [Training evidence](evaluation_results/cycle-232-train.json).
  [Narrative](full_log.md#cycle-232--dimer-gdiis-interpolants-up-to-115-times-the-qn-length).

## sella-connected-late-iterative-stepper: iterative Cartesian realization of connected tails
Status: closed

**Hypothesis:** Delayed iterative B⁺ maps on connected QN after 50 spare 135043047 and may cheapen paliperidone.

**Outcome and uncertainty:** Cycle 233 discard, paliperidone 78→84. Iterative realization lengthens that tail.

**Reason to revisit:** Not as delayed iterative on connected QN.

**Next experiment:** Do not delay `iterative_stepper=1` on connected tails. Widen late connected `rho_inc` instead.

**Attempts:**
- Cycle 233; candidate `a29c8ec6dd6494fa81b664961cd431aecec0b778`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-connected-late-iterative-stepper/a29c8ec6dd6494fa81b664961cd431aecec0b778/algo.py).
  [Training evidence](evaluation_results/cycle-233-train.json).
  [Narrative](full_log.md#cycle-233--iterative-cartesian-realization-of-connected-steps-after-50).

## sella-connected-late-rho-inc: wider trust-expansion window after 50 connected steps
Status: closed

**Hypothesis:** Cycle 25 hopped from step 0. Widening `rho_inc` to 1.4 after 50 spares 135043047 and may cheapen paliperidone tails.

**Outcome and uncertainty:** Cycle 234 discard, 160853090 88→91. Wider late expansion inflates the long tail.

**Reason to revisit:** Not as a wider late `rho_inc`.

**Next experiment:** Do not widen late `rho_inc`. Stop late connected trust growth instead.

**Attempts:**
- Cycle 234; candidate `c32ffa0b806854a9c020ce69cf9af48d1c9208df`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-connected-late-rho-inc/c32ffa0b806854a9c020ce69cf9af48d1c9208df/algo.py).
  [Training evidence](evaluation_results/cycle-234-train.json).
  [Narrative](full_log.md#cycle-234--wider-connected-trust-expansion-window-after-50-steps).

## sella-connected-late-sigma-hold: freeze connected trust expansion after 50
Status: closed

**Hypothesis:** Late `σ_inc=1.16` grows paliperidone/160853090 δ too far. Holding `σ_inc=1` after 50 freezes the radius.

**Outcome and uncertainty:** Cycle 235 discard, bit-identical. Late expansion is already saturated (`σ_inc * smag ≤ δ`).

**Reason to revisit:** Not as a late `σ_inc` hold.

**Next experiment:** Do not freeze late connected trust growth. Repair cycle 107 dummy-angle H0 with an n_atoms≥30 gate.

**Attempts:**
- Cycle 235; candidate `8bbddb0a647b43a48b30ee4b117b27669728f6e9`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision discard.
  [Implementation](ideas/sella-connected-late-sigma-hold/8bbddb0a647b43a48b30ee4b117b27669728f6e9/algo.py).
  [Training evidence](evaluation_results/cycle-235-train.json).
  [Narrative](full_log.md#cycle-235--stop-connected-trust-radius-expansion-after-50-steps).

## sella-dummy-angle-h0-large: dummy-angle H0 0.10 Ha on connected n_atoms≥30
Status: incorporated

**Hypothesis:** Cycle 107 extras were small (18–19 atom) canaries. n_atoms≥30 keeps 363892164 and spares them.

**Outcome and uncertainty:** Cycle 236 **keep**. Train Δ=+3.18e-4 (363892164/135191719/319495631 −2; 134979308/acalabrutinib +1). Valid Δ=+1.91e-3 (251920090 −10, 103958198 −5, 160845249 −5; 135249279 +2). Paliperidone/venetoclax/135043047/pyridine–pyrrole unchanged. Energy-safe.

**Reason to revisit:** Incorporated. Remaining train leftover 363892164 is still +13 vs ref. 252089162 (11) and 135043047 (18) stay ungated. Cycle 107 also saved 135149279 (20, −4) and 440717067 (25, −2) below the ≥30 cut.

**Next experiment:** Widen to n_atoms≥20 (`sella-dummy-angle-h0-medium`). Do not drop to all connected. Do not interpolate 0.10 on ≥30.

**Attempts:**
- Cycle 236; candidate `f5d05d318d217d46542a4473afeee3e30bd0c276`; champion `7064f55bde9f04bd3c7f0cc012a0e8b01492151c`; decision keep.
  [Training evidence](evaluation_results/cycle-236-train.json).
  [Validation evidence](evaluation_results/cycle-236-valid.json).
  [Narrative](full_log.md#cycle-236--dummy-involving-angle-hessian-010-ha-on-connected-n_atoms30).

## sella-dummy-angle-h0-medium: dummy-angle H0 0.10 Ha on connected n_atoms≥20
Status: closed

**Hypothesis:** Cycle 107 saved 135149279 (20, −4) and 440717067 (25, −2) that the ≥30 keep excludes. n_atoms≥20 still spares 135043047 (18) and pyridine–pyrrole (19).

**Outcome and uncertainty:** Cycle 237 non_generalizable. Cycle 277 n≥25 non_generalizable: train 440717067 38→36, valid 26/29 extras. 25–29 is still mixed.

**Reason to revisit:** Not as a 20–29 or 25–29 fill. Spare 18–29. Cycle 369 n≤18 extras 135043047 43→45.

**Next experiment:** Do not interpolate 18–29 dummy-angle size cuts. Do not fill n=18. Cycle 632 thiosulfonate-only 12–30 dummy-angle fill was bit-identical (leftover 433943527 has no dummy-involving angles).

**Attempts:**
- Cycle 369; candidate `3322eb546d695b2c4dd2dc1351c68c5427769219`; champion `7140449e9cbddb6f6589ec34fdac86f88dc413e1`; decision discard.
  [Implementation](ideas/sella-dummy-angle-h0-medium/3322eb546d695b2c4dd2dc1351c68c5427769219/algo.py).
  [Training evidence](evaluation_results/cycle-369-train.json).
  [Narrative](full_log.md#cycle-369--dummy-angle-h0-010-ha-on-connected-n_atoms18-or-30).
- Cycle 277; candidate `4ecc83fc85a62a89c89372a084967776708e6abf`; champion `cb0642bc324a4da62cfa93338d325865857c0742`; decision non_generalizable.
  [Implementation](ideas/sella-dummy-angle-h0-medium/4ecc83fc85a62a89c89372a084967776708e6abf/algo.py).
  [Training evidence](evaluation_results/cycle-277-train.json).
  [Validation evidence](evaluation_results/cycle-277-valid.json).
  [Narrative](full_log.md#cycle-277--dummy-angle-h0-010-ha-on-connected-n_atoms18-or-25).
- Cycle 237; candidate `66d03845da2afe36ae0a43be08877e75331ea09d`; champion `f5d05d318d217d46542a4473afeee3e30bd0c276`; decision non_generalizable.
  [Implementation](ideas/sella-dummy-angle-h0-medium/66d03845da2afe36ae0a43be08877e75331ea09d/algo.py).
  [Training evidence](evaluation_results/cycle-237-train.json).
  [Validation evidence](evaluation_results/cycle-237-valid.json).
  [Narrative](full_log.md#cycle-237--dummy-involving-angle-hessian-010-ha-on-connected-n_atoms20).

## sella-dummy-angle-h0-small: dummy-angle H0 0.10 Ha on connected n_atoms<18 plus ≥30
Status: incorporated

**Hypothesis:** Cycle 107 small dummy-linear saves (315516336 n=16 −5) sit below the 18–19 extras. Pair that band with the ≥30 keep; spare 18–29.

**Outcome and uncertainty:** Cycle 238 **keep**. Train Δ=+1.21e-3 (315516336 31→26; 252089162 42→41; small −1/−2). Valid Δ=+6.57e-4 (135169446 28→26, 134982070 −2, 135093104 −2; no extras). Energy-safe. 18–29 unchanged.

**Reason to revisit:** Incorporated. Do not fill the 18–29 gap (cycles 107 and 237 extras). Remaining 363892164 +13 is already ≥30; 135169446 still +8 vs ref at 0.10 Ha.

**Next experiment:** Soften ≥30 dummy-angle H0 to 0.08 Ha; keep 0.10 on n<18. Do not drop the 18–29 hole.

## sella-dummy-angle-h0-008-large: dummy-angle H0 0.08 Ha on connected n_atoms≥30
Status: closed

**Hypothesis:** 0.10 Ha still leaves 363892164 +13. 0.08 on ≥30 only, n<18 stays 0.10.

**Outcome and uncertainty:** Cycle 239 discard, Δ=−7.90e-5. 363892164 unchanged 44. Only extra 135191719 26→27. 104046121 stays 33. 0.08 does not cheapen the leftover and undoes a 0.10 save.

**Reason to revisit:** Not as an H0 scale on ≥30. 0.10 is the dummy-angle value.

**Next experiment:** Do not interpolate dummy-angle H0 on ≥30. Do not try 0.09.

## sella-dummy-angle-h0-008-small: dummy-angle H0 0.08 Ha on connected n_atoms<18
Status: closed

**Hypothesis:** Small leftovers 252089162 and 135169446 still improved at 0.10. 0.08 on n<18 only.

**Outcome and uncertainty:** Cycle 240 discard, Δ=−9.14e-4. 252089162 unchanged 41. 135039840 16→24 dominates. 315516336 26→27. 0.08 too soft on small dummy-linear jobs.

**Reason to revisit:** Not as an H0 scale on n<18. 0.10 is the dummy-angle value on both bands.

**Next experiment:** Do not interpolate dummy-angle H0. Try Lindh dummy-only angles on ≥30 instead of a constant scale.

## sella-lindh-dummy-angle-h0: Lindh dummy-involving angle Hessian on connected n_atoms≥30
Status: closed

**Hypothesis:** Geometry-dependent Lindh k_θ on dummy angles can cheapen leftover 363892164 after constant 0.10/0.08 saturated.

**Outcome and uncertainty:** Cycle 241 discard, Δ=−7.90e-5. n_steps bit-identical to cycle 239 (135191719 +1). Lindh dummy-angle ≡ 0.08 constant here.

**Reason to revisit:** Not as a Lindh dummy-angle form. 0.10 constant stays.

**Next experiment:** Do not interpolate dummy-angle H0 forms on ≥30. Loosen GDIIS cosine on n≥30 instead.

## sella-connected-gdiis-cos-large: GDIIS cosine 0.85 on connected n_atoms≥30
Status: closed

**Hypothesis:** Leftover 363892164/venetoclax reject GDIIS at cosine 0.90. 0.85 on large connected only.

**Outcome and uncertainty:** Cycle 242 discard, Δ=−1.22e-4. Only extra 104046121 33→35. Leftovers unchanged. 0.85 does not unlock those interpolants.

**Reason to revisit:** Not as a cosine tweak on ≥30. Keep 0.90.

**Next experiment:** Do not loosen GDIIS cosine. Skip the ρ window on n≥30 GDIIS instead.

## sella-connected-gdiis-no-rho-large: GDIIS without ρ gate on connected n_atoms≥30
Status: closed

**Hypothesis:** Leftovers fail the ρ window, not cosine. Skip ρ on n≥30, keep cosine 0.90.

**Outcome and uncertainty:** Cycle 243 discard, Δ=−1.41e-4. Paliperidone −1 vs 104046121 +2 and 135098427 +1. 363892164 unchanged. ρ skip is not the leftover gate.

**Reason to revisit:** Not as a ρ-ungated GDIIS on ≥30. Do not delay to nsteps≥50 (paliperidone −1 cannot meet δ).

**Next experiment:** Do not drop the GDIIS ρ window. Leave cosine 0.90.

## sella-connected-rfo-80-large: RFO after 80 connected steps on n_atoms≥30
Status: closed

**Hypothesis:** Long connected tails that reach 80 can take dimer-style Banerjee RFO without paliperidone (78).

**Outcome and uncertainty:** Cycle 244 discard, Δ=0, n_steps bit-identical. 160853090 energy-wiggles at 88. RFO@80 does not change connected force-call counts.

**Reason to revisit:** Not as an nsteps=80 connected RFO. Do not start earlier (RFO@50 closed).

**Next experiment:** Do not interpolate connected RFO start after 80. Leave connected QN.

## sella-dummy-bond-h0-large: dummy-involving stretch Hessian 0.10 Ha/Bohr² on connected n_atoms≥30
Status: closed

**Hypothesis:** Unconstrained dummy-set bonds still use Fischer. Soften them on n≥30 after dummy-angle 0.10 saturated.

**Outcome and uncertainty:** Cycle 245 discard, Δ=0, n_steps bit-identical. Dummy bonds are constrained; H0 on them is a no-op.

**Reason to revisit:** Not as a dummy-bond H0 scale. Those stretches are not free internals.

**Next experiment:** Do not interpolate dummy-bond H0. Raise the ≥30 connected trust floor after 20 instead.

## sella-connected-delta-min-018-large: trust floor 0.18 after 20 on connected n_atoms≥30
Status: closed

**Hypothesis:** 0.15 still truncates leftover 363892164/venetoclax. 0.18 on ≥30 only.

**Outcome and uncertainty:** Cycle 246 discard, Δ=−7.75e-4. Paliperidone/venetoclax +10. 363892164 unchanged. Larger floor inflates floppy drugs.

**Reason to revisit:** Not as a δ floor above 0.15 on ≥30.

**Next experiment:** Do not interpolate 0.16. Keep 0.15 after 20.

## sella-geometric-linear-angle: geomeTRIC LinearAngle on connected n_atoms≥30
Status: closed

**Hypothesis:** Dummy-atom linear-bend H0/GDIIS/RFO/δ are exhausted for leftover 363892164. Two orthogonal LinearAngle coordinates without dummy atoms (geomeTRIC) on n≥30 can cheapen that scaffold while n<18 keeps dummy-atom 0.10 H0.

**Outcome and uncertainty:** Cycle 247 invalid (dummy-free LinearAngle energy-unsafe / 135114602 stall). Cycle 248 invalid (additive dummy+LinearAngle hops/SCC/RSS). Cycle 249 invalid (stay-put `matmul`). Cycle 250 invalid: FD/frozen e0 without stay-put reproduces cycle 247 (363892164 44→31 at a higher well, 134979308/acalabrutinib early higher wells, 135114602 200-call stall). Dummy-free LinearAngle is not an energy-safe leftover save. Additive LinearAngle and hop-cap/stay-put are contradicted. Repair 3/3 done.

**Reason to revisit:** Not as dummy-free or additive LinearAngle coordinates. Dummy Cartesian DOF are required on these 2-coordinate linear centers. A stable dummy *placement* (geomeTRIC e0) that keeps dummy atoms is a different idea.

**Next experiment:** Do not retry dummy-free or additive LinearAngle. Place dummy atoms with a frozen-style e0 axis on connected n≥30 (`sella-e0-dummy-placement`).

**Attempts:**
- Cycle 250; candidate `5feae64ebbcf737bedc1832e5b2feeb6a858a7a5`; champion `63984199efc0776436013dcfe4f7a7dd1a8a586e`; decision invalid.
  [Implementation](ideas/sella-geometric-linear-angle/5feae64ebbcf737bedc1832e5b2feeb6a858a7a5/algo.py).
  [Training evidence](evaluation_results/cycle-250-train.json).
  [Narrative](full_log.md#cycle-250--dummy-free-linearangle-with-fd-hessian-no-stay-put).
- Cycle 249; candidate `8a92d2ec5eb96b890486ae98c6ecfbda5179ae33`; champion `63984199efc0776436013dcfe4f7a7dd1a8a586e`; decision invalid.
  [Implementation](ideas/sella-geometric-linear-angle/8a92d2ec5eb96b890486ae98c6ecfbda5179ae33/algo.py).
  [Training evidence](evaluation_results/cycle-249-train.json).
  [Narrative](full_log.md#cycle-249--dummy-free-linearangle-with-fd-hessian-and-hop-safe-geodesic).

**Attempts:**
- Cycle 248; candidate `521b08d91ab3fa3996f0ff337a80f96a90c67dc1`; champion `63984199efc0776436013dcfe4f7a7dd1a8a586e`; decision invalid.
  [Implementation](ideas/sella-geometric-linear-angle/521b08d91ab3fa3996f0ff337a80f96a90c67dc1/algo.py).
  [Training evidence](evaluation_results/cycle-248-train.json).
  [Narrative](full_log.md#cycle-248--additive-linearangle-with-dummy-atoms-kept-on-n30).

**Attempts:**
- Cycle 247; candidate `43e7cc2a7783c9fdc55945c84e099204f61c38b7`; champion `63984199efc0776436013dcfe4f7a7dd1a8a586e`; decision invalid.
  [Implementation](ideas/sella-geometric-linear-angle/43e7cc2a7783c9fdc55945c84e099204f61c38b7/algo.py).
  [Training evidence](evaluation_results/cycle-247-train.json).
  [Narrative](full_log.md#cycle-247--geometric-linearangle-coordinates-on-connected-n_atoms30).

## sella-e0-dummy-placement: geomeTRIC e0 dummy axis on connected n_atoms≥30
Status: deferred

**Hypothesis:** Dummy-free LinearAngle missed the lower well because it dropped dummy Cartesian DOF. Keeping dummy atoms but placing them on geomeTRIC's e0 axis (Cartesian basis most orthogonal to the linear frame) should stabilize 2-coordinate dummy planes without changing which coordinates exist.

**Outcome and uncertainty:** Cycle 251 discard (104046121 wander). Cycle 252 non_generalizable (train Δ=+1.15e-4, valid Δ=−2.41e-4). Cycle 253 non_generalizable (train Δ=+1.28e-4, valid Δ=+8.45e-5). Cycle 257 discard (e0 window + 1 Bohr, Δ=+4.60e-5). Cycle 259 discard (e0 on n<18 or ≥30, Δ=−7.38e-4). Cycle 266 non_generalizable on the cycle-265 champion: train Δ=+1.28e-4 additive, valid Δ=+8.45e-5 same 252634821 extra. Best e0 variant remains 253/266.

**Reason to revisit:** Still a near-miss on valid. Needs an independent valid save, not window interpolation.

**Next experiment:** Do not interpolate 0.04/0.10. Adjacent-substituent dummy (cycle 270) is a near-miss on this champion.

**Attempts:**
- Cycle 266; candidate `1511c02ec8f150b3f4bab6e19617e0b137bbb4f4`; champion `488f94cdf1a71e13529b7873dd559a470dbf54e2`; decision non_generalizable.
  [Implementation](ideas/sella-e0-dummy-placement/1511c02ec8f150b3f4bab6e19617e0b137bbb4f4/algo.py).
  [Training evidence](evaluation_results/cycle-266-train.json).
  [Validation evidence](evaluation_results/cycle-266-valid.json).
  [Narrative](full_log.md#cycle-266--windowed-e0-dummy-placement-on-the-p-neighbor-oxo-champion).
- Cycle 259; candidate `b691b48dc2779724b8e4a4798ca987bac7e7f052`; champion `63984199efc0776436013dcfe4f7a7dd1a8a586e`; decision discard.
  [Implementation](ideas/sella-e0-dummy-placement/b691b48dc2779724b8e4a4798ca987bac7e7f052/algo.py).
  [Training evidence](evaluation_results/cycle-259-train.json).
  [Narrative](full_log.md#cycle-259--windowed-e0-dummy-placement-on-dummy-angle-size-bands).
- Cycle 257; candidate `5bfc05bbe5b8db269ee421f2153f225fe5d404e9`; champion `63984199efc0776436013dcfe4f7a7dd1a8a586e`; decision discard.
  [Implementation](ideas/sella-e0-dummy-placement/5bfc05bbe5b8db269ee421f2153f225fe5d404e9/algo.py).
  [Training evidence](evaluation_results/cycle-257-train.json).
  [Narrative](full_log.md#cycle-257--windowed-e0-dummy-placement-plus-1-bohr-dummy-distance).
- Cycle 253; candidate `665e37aaf8b0935f6eda24fcbc22d9e35ba31759`; champion `63984199efc0776436013dcfe4f7a7dd1a8a586e`; decision non_generalizable.
  [Implementation](ideas/sella-e0-dummy-placement/665e37aaf8b0935f6eda24fcbc22d9e35ba31759/algo.py).
  [Training evidence](evaluation_results/cycle-253-train.json).
  [Validation evidence](evaluation_results/cycle-253-valid.json).
  [Narrative](full_log.md#cycle-253--e0-dummy-placement-in-a-moderate-linear-frame-window).
- Cycle 252; candidate `e91f257a8949cd644b2970b270eb99c7625e8ef9`; champion `63984199efc0776436013dcfe4f7a7dd1a8a586e`; decision non_generalizable.
  [Implementation](ideas/sella-e0-dummy-placement/e91f257a8949cd644b2970b270eb99c7625e8ef9/algo.py).
  [Training evidence](evaluation_results/cycle-252-train.json).
  [Validation evidence](evaluation_results/cycle-252-valid.json).
  [Narrative](full_log.md#cycle-252--e0-dummy-placement-only-for-ill-conditioned-linear-frames).
- Cycle 251; candidate `a05343dc9b55b38ecf2ff1984a16fbd3d9710bae`; champion `63984199efc0776436013dcfe4f7a7dd1a8a586e`; decision discard.
  [Implementation](ideas/sella-e0-dummy-placement/a05343dc9b55b38ecf2ff1984a16fbd3d9710bae/algo.py).
  [Training evidence](evaluation_results/cycle-251-train.json).
  [Narrative](full_log.md#cycle-251--geometric-e0-dummy-placement-on-connected-n_atoms30).


**Attempts:**
- Cycle 246; candidate `02222c27446313976f78678cc2cc16fc99eac34a`; champion `63984199efc0776436013dcfe4f7a7dd1a8a586e`; decision discard.
  [Implementation](ideas/sella-connected-delta-min-018-large/02222c27446313976f78678cc2cc16fc99eac34a/algo.py).
  [Training evidence](evaluation_results/cycle-246-train.json).
  [Narrative](full_log.md#cycle-246--connected-trust-floor-018-after-20-steps-on-n_atoms30).

**Attempts:**
- Cycle 245; candidate `78942f81ada3345193c18034315e72f5d5b354ef`; champion `63984199efc0776436013dcfe4f7a7dd1a8a586e`; decision discard.
  [Implementation](ideas/sella-dummy-bond-h0-large/78942f81ada3345193c18034315e72f5d5b354ef/algo.py).
  [Training evidence](evaluation_results/cycle-245-train.json).
  [Narrative](full_log.md#cycle-245--dummy-involving-stretch-hessian-010-habohr²-on-connected-n_atoms30).

**Attempts:**
- Cycle 244; candidate `c8bc3408f66f437d22938db1b63557f42ec82c3a`; champion `63984199efc0776436013dcfe4f7a7dd1a8a586e`; decision discard.
  [Implementation](ideas/sella-connected-rfo-80-large/c8bc3408f66f437d22938db1b63557f42ec82c3a/algo.py).
  [Training evidence](evaluation_results/cycle-244-train.json).
  [Narrative](full_log.md#cycle-244--connected-rfo-after-80-steps-on-n_atoms30).

**Attempts:**
- Cycle 243; candidate `96d993523abfd9a465359259ddc2beb422cd00a5`; champion `63984199efc0776436013dcfe4f7a7dd1a8a586e`; decision discard.
  [Implementation](ideas/sella-connected-gdiis-no-rho-large/96d993523abfd9a465359259ddc2beb422cd00a5/algo.py).
  [Training evidence](evaluation_results/cycle-243-train.json).
  [Narrative](full_log.md#cycle-243--connected-gdiis-without-ρ-gate-on-n_atoms30).

**Attempts:**
- Cycle 242; candidate `97988c43c44f3c9a7b0771ea4e8ed5bbfeb7b8b4`; champion `63984199efc0776436013dcfe4f7a7dd1a8a586e`; decision discard.
  [Implementation](ideas/sella-connected-gdiis-cos-large/97988c43c44f3c9a7b0771ea4e8ed5bbfeb7b8b4/algo.py).
  [Training evidence](evaluation_results/cycle-242-train.json).
  [Narrative](full_log.md#cycle-242--connected-gdiis-cosine-085-on-n_atoms30).

**Attempts:**
- Cycle 241; candidate `fc01feddf4066dc63ee37b58da06e33de607e70a`; champion `63984199efc0776436013dcfe4f7a7dd1a8a586e`; decision discard.
  [Implementation](ideas/sella-lindh-dummy-angle-h0/fc01feddf4066dc63ee37b58da06e33de607e70a/algo.py).
  [Training evidence](evaluation_results/cycle-241-train.json).
  [Narrative](full_log.md#cycle-241--lindh-1995-dummy-involving-angle-hessian-on-connected-n_atoms30).

**Attempts:**
- Cycle 240; candidate `12e7deb258d9cf5138691de1049a6c2bdc8edf1f`; champion `63984199efc0776436013dcfe4f7a7dd1a8a586e`; decision discard.
  [Implementation](ideas/sella-dummy-angle-h0-008-small/12e7deb258d9cf5138691de1049a6c2bdc8edf1f/algo.py).
  [Training evidence](evaluation_results/cycle-240-train.json).
  [Narrative](full_log.md#cycle-240--dummy-involving-angle-hessian-008-ha-on-connected-n_atoms18).

**Attempts:**
- Cycle 239; candidate `bc054b323f7b4b05076dd20947ac1a576956ff17`; champion `63984199efc0776436013dcfe4f7a7dd1a8a586e`; decision discard.
  [Implementation](ideas/sella-dummy-angle-h0-008-large/bc054b323f7b4b05076dd20947ac1a576956ff17/algo.py).
  [Training evidence](evaluation_results/cycle-239-train.json).
  [Narrative](full_log.md#cycle-239--dummy-involving-angle-hessian-008-ha-on-connected-n_atoms30).

**Attempts:**
- Cycle 238; candidate `63984199efc0776436013dcfe4f7a7dd1a8a586e`; champion `f5d05d318d217d46542a4473afeee3e30bd0c276`; decision keep.
  [Training evidence](evaluation_results/cycle-238-train.json).
  [Validation evidence](evaluation_results/cycle-238-valid.json).
  [Narrative](full_log.md#cycle-238--dummy-involving-angle-hessian-010-ha-on-connected-n_atoms18-or-30).

## sella-bohr-dummy-distance: 1 Bohr dummy–center distance on connected n_atoms≥30
Status: closed

**Hypothesis:** QUILD places dummy atoms 1 Bohr from linear centers; Sella uses 1 Å. Shortening the constrained dummy bond on n≥30 rescales dummy-angle/dihedral B rows without dropping dummy DOF.

**Outcome and uncertainty:** Cycle 254 discard, Δ=−8.20e-5. Leftover 363892164 unchanged. Dummy stretch is constrained; length is not the leftover limiter. Cycle 257 stacked Bohr on windowed e0: leftover save kept, 254 extras returned, train Δ=+4.60e-5.

**Reason to revisit:** Not as a dummy-distance scale, including on e0 planes.

**Next experiment:** Do not interpolate dummy distance or Bohr-only-on-e0.

**Attempts:**
- Cycle 254; candidate `b58df98f45c2df75d64286e36ff0e66aa42b443d`; champion `63984199efc0776436013dcfe4f7a7dd1a8a586e`; decision discard.
  [Implementation](ideas/sella-bohr-dummy-distance/b58df98f45c2df75d64286e36ff0e66aa42b443d/algo.py).
  [Training evidence](evaluation_results/cycle-254-train.json).
  [Narrative](full_log.md#cycle-254--1-bohr-dummy-distance-on-connected-n_atoms30).

## sella-large-wa-1: default angle caps after 20 on connected n_atoms≥80
Status: closed

**Hypothesis:** Restoring `wa=1` after 20 on n≥80 cheapens venetoclax/paliperidone without dummy-linear leftover.

**Outcome and uncertainty:** Cycle 255 discard, bit-identical. Angle caps are not the n≥80 MIS limiter.

**Reason to revisit:** Not as a wa tweak on n≥80.

**Next experiment:** Do not interpolate n≥80 wa. Do not retry wd=0.9 on that class (cycle 95 inflated venetoclax).

**Attempts:**
- Cycle 255; candidate `e3fed286262761374179ee9ce5f702006cac9662`; champion `63984199efc0776436013dcfe4f7a7dd1a8a586e`; decision discard.
  [Implementation](ideas/sella-large-wa-1/e3fed286262761374179ee9ce5f702006cac9662/algo.py).
  [Training evidence](evaluation_results/cycle-255-train.json).
  [Narrative](full_log.md#cycle-255--default-angle-caps-after-20-steps-on-connected-n_atoms80).

## sella-small-skip-gdiis: skip GDIIS on connected n_atoms<12
Status: closed

**Hypothesis:** Leftover 252089162 (n=11, 41) is GDIIS-limited after 20 steps. Skipping GDIIS on connected n<12 cheapens it without dummy-linear n≥30.

**Outcome and uncertainty:** Cycle 256 discard, bit-identical. GDIIS was not accepting on that band.

**Reason to revisit:** Not as an n<12 or n<18 GDIIS skip.

**Next experiment:** Do not interpolate the size cut. See cycle 253 windowed e0 as the leftover signal to combine later.

**Attempts:**
- Cycle 256; candidate `3046b7a74c776d25144ad1b88a9d97b998708400`; champion `63984199efc0776436013dcfe4f7a7dd1a8a586e`; decision discard.
  [Implementation](ideas/sella-small-skip-gdiis/3046b7a74c776d25144ad1b88a9d97b998708400/algo.py).
  [Training evidence](evaluation_results/cycle-256-train.json).
  [Narrative](full_log.md#cycle-256--skip-gdiis-on-connected-n_atoms12).

## sella-com-dummy-placement: QUILD COM dummy axis on connected n_atoms≥30
Status: closed

**Hypothesis:** Sella's dummy plane is `u×v` of nearly collinear 2-coordinate bonds (ill-conditioned). QUILD orients the dummy toward the molecular center of mass in the plane perpendicular to the linear axis. That direction is chemically defined (not a Cartesian basis like e0) and may split leftover alkynes from valid 252634821 without 104046121 wander if gated to `0.04 < ||u×v|| < 0.10`.

**Outcome and uncertainty:** Cycle 258 discard, Δ=−9.33e-5. Leftover 363892164 44→45 (wrong plane). 134979308/acalabrutinib −1 eaten by 135175432/135191719 +1. Window spared 104046121. Energy-safe.

**Reason to revisit:** Not as a COM dummy axis.

**Next experiment:** Do not interpolate always-on COM or COM+Bohr. Return to cycle 253 e0 window on the dummy-angle size bands.

**Attempts:**
- Cycle 258; candidate `29037da2450a12b757648d76bae16ae76e743568`; champion `63984199efc0776436013dcfe4f7a7dd1a8a586e`; decision discard.
  [Implementation](ideas/sella-com-dummy-placement/29037da2450a12b757648d76bae16ae76e743568/algo.py).
  [Training evidence](evaluation_results/cycle-258-train.json).
  [Narrative](full_log.md#cycle-258--quild-com-dummy-placement-in-a-moderate-linear-frame-window).

## sella-drop-unused-dummies: rebuild without dummy when the linear window is left
Status: closed

**Hypothesis:** Sella keeps 2-coordinate dummy atoms even after the real angle is a well-defined bend. QUILD drops the dummy and regenerates internals. Doing that on connected n≥30 (same rebuild as `bad_int`) can cheapen leftover alkynes after they leave ~175°.

**Outcome and uncertainty:** Cycle 260 discard, bit-identical. No n≥30 dummy left the 15° window on train.

**Reason to revisit:** Not as an `atol`-window drop.

**Next experiment:** Do not interpolate an earlier drop threshold.

**Attempts:**
- Cycle 260; candidate `0ccef6186843ea4650d495994accead59b068bba`; champion `63984199efc0776436013dcfe4f7a7dd1a8a586e`; decision discard.
  [Implementation](ideas/sella-drop-unused-dummies/0ccef6186843ea4650d495994accead59b068bba/algo.py).
  [Training evidence](evaluation_results/cycle-260-train.json).
  [Narrative](full_log.md#cycle-260--drop-unused-dummy-atoms-when-the-linear-window-is-left).

## sella-small-rfo: Banerjee RFO on connected n_atoms<12
Status: closed

**Hypothesis:** Small connected leftovers (252089162) need RFO rather than |λ| QN. n<12 spares 135043047.

**Outcome and uncertainty:** Cycle 261 discard, Δ=−7.10e-4. Cycle 262 RFO@20 discard, Δ=−3.40e-5. Cycle 263 RFO@30 discard, step-identical. Late RFO is not a force-call save. 252089162 unchanged throughout.

**Reason to revisit:** Not as a delayed n<12 RFO.

**Next experiment:** Do not interpolate the delay or widen to n<18. See 2-coordinate oxygen angle H0 on n<12 for leftover 252089162.

**Attempts:**
- Cycle 263; candidate `7aac51b17c523a4e0ab5a817560f2b8ab406508d`; champion `63984199efc0776436013dcfe4f7a7dd1a8a586e`; decision discard.
  [Implementation](ideas/sella-small-rfo/7aac51b17c523a4e0ab5a817560f2b8ab406508d/algo.py).
  [Training evidence](evaluation_results/cycle-263-train.json).
  [Narrative](full_log.md#cycle-263--banerjee-rfo-after-30-steps-on-connected-n_atoms12).

## sella-large-skip-gdiis: skip GDIIS on connected n_atoms≥80
Status: closed

**Hypothesis:** Venetoclax (n=111, +6 vs ref) is not dummy-linear. Two-point GDIIS after 20 may lengthen that floppy tail. Skipping GDIIS on connected n≥80 leaves paliperidone's early path and leftover 363892164 (n=43) unchanged.

**Outcome and uncertainty:** Cycle 268 skip after 50: discard, bit-identical. Cycle 269 skip from 20: discard, bit-identical. GDIIS never accepts on connected n≥80.

**Reason to revisit:** Not as an n≥80 GDIIS skip.

**Next experiment:** Do not interpolate n≥80 GDIIS skips. Three-point C1 GDIIS on n≥80 may accept where two-point never does.

**Attempts:**
- Cycle 269; candidate `bd2d11e49618135c1064db3f6c1c7ac65b956c57`; champion `488f94cdf1a71e13529b7873dd559a470dbf54e2`; decision discard.
  [Implementation](ideas/sella-large-skip-gdiis/bd2d11e49618135c1064db3f6c1c7ac65b956c57/algo.py).
  [Training evidence](evaluation_results/cycle-269-train.json).
  [Narrative](full_log.md#cycle-269--skip-gdiis-on-connected-n_atoms80).
- Cycle 268; candidate `b29c2fc26e67ce4db6eab443560fe0474bf02120`; champion `488f94cdf1a71e13529b7873dd559a470dbf54e2`; decision discard.
  [Implementation](ideas/sella-large-skip-gdiis/b29c2fc26e67ce4db6eab443560fe0474bf02120/algo.py).
  [Training evidence](evaluation_results/cycle-268-train.json).
  [Narrative](full_log.md#cycle-268--skip-gdiis-after-50-steps-on-connected-n_atoms80).

## sella-adj-dummy-placement: adjacent-substituent dummy plane on connected n_atoms≥30
Status: incorporated

**Hypothesis:** Ill-conditioned dummy planes (||u×v||~0.04–0.10) should use a local substituent to define a coplanar dummy direction (Schlegel φ_n12d=0), not Cartesian e0 or COM.

**Outcome and uncertainty:** Cycle 270 discard, energy-safe, Δ=+8.20e-5 (N-center extra). Cycle 271 carbon-center **keep**: train Δ=+1.35e-4, valid Δ=+3.27e-4. Leftover 363892164 unchanged; 252634821 31→30 (e0's extra became a save).

**Reason to revisit:** Incorporated as carbon-center windowed dummy. N-center dummy-plane interpolation is closed: 272 leftover −1 below gate, 278 e0 bit-identical, 279 LINP bit-identical. Do not drop the carbon gate back to all centers.

**Next experiment:** Close n<18 adj dummy (cycle 299 extras 315516336/135099623, no train save). Try 0.10 Ha C–N–O on 2-coordinate nitrogen for connected n<18.

**Attempts:**
- Cycle 299; candidate `e51f13e901fbda52d70fbf42e66aa0a9537caa57`; champion `7b0f85a2864652b033bd8929e169aabb0468c5b7`; decision discard.
  [Implementation](ideas/sella-adj-dummy-placement/e51f13e901fbda52d70fbf42e66aa0a9537caa57/algo.py).
  [Training evidence](evaluation_results/cycle-299-train.json).
  [Narrative](full_log.md#cycle-299--carbon-center-adjacent-dummy-on-connected-n_atoms18).
- Cycle 279; candidate `59c8d3118d8ad6f0dbd4f00df9d42aae22ca9646`; champion `cb0642bc324a4da62cfa93338d325865857c0742`; decision discard.
  [Implementation](ideas/sella-adj-dummy-placement/59c8d3118d8ad6f0dbd4f00df9d42aae22ca9646/algo.py).
  [Training evidence](evaluation_results/cycle-279-train.json).
  [Narrative](full_log.md#cycle-279--linp-dummy-on-cnc-nitrogen-centers).
- Cycle 278; candidate `3e05b619ac0b51c3a9f199445f039bae3b3983b9`; champion `cb0642bc324a4da62cfa93338d325865857c0742`; decision discard.
  [Implementation](ideas/sella-adj-dummy-placement/3e05b619ac0b51c3a9f199445f039bae3b3983b9/algo.py).
  [Training evidence](evaluation_results/cycle-278-train.json).
  [Narrative](full_log.md#cycle-278--e0-dummy-on-cnc-nitrogen-centers).
- Cycle 272; candidate `ffb0878ec8785d4ca72f6d59e33793be75c0037a`; champion `cb0642bc324a4da62cfa93338d325865857c0742`; decision discard.
  [Implementation](ideas/sella-adj-dummy-placement/ffb0878ec8785d4ca72f6d59e33793be75c0037a/algo.py).
  [Training evidence](evaluation_results/cycle-272-train.json).
  [Narrative](full_log.md#cycle-272--cnc-nitrogen-adjacent-substituent-dummy-plane).
- Cycle 271; candidate `cb0642bc324a4da62cfa93338d325865857c0742`; champion `488f94cdf1a71e13529b7873dd559a470dbf54e2`; decision keep.
  [Training evidence](evaluation_results/cycle-271-train.json).
  [Validation evidence](evaluation_results/cycle-271-valid.json).
  [Narrative](full_log.md#cycle-271--carbon-center-adjacent-substituent-dummy-plane).
- Cycle 270; candidate `06d4f79a728b68926c59f679468165bd6c22035a`; champion `488f94cdf1a71e13529b7873dd559a470dbf54e2`; decision discard.
  [Implementation](ideas/sella-adj-dummy-placement/06d4f79a728b68926c59f679468165bd6c22035a/algo.py).
  [Training evidence](evaluation_results/cycle-270-train.json).
  [Narrative](full_log.md#cycle-270--adjacent-substituent-dummy-plane-in-a-moderate-linear-frame-window).

## sella-large-iterative-stepper: iterative Cartesian realization on connected n_atoms≥80
Status: closed

**Hypothesis:** Iterative B⁺ maps on connected n≥80 cheapen venetoclax without dummy-linear leftover.

**Outcome and uncertainty:** Cycle 273 discard, Δ=−6.29e-4. Paliperidone 78→101; venetoclax unchanged. Realization is not venetoclax's limiter.

**Reason to revisit:** Not as n≥80 iterative from step 0. Do not delay it: venetoclax does not move.

**Next experiment:** Do not interpolate n≥80 iterative delay. Try three-point GDIIS on that band.

**Attempts:**
- Cycle 273; candidate `6a7cc16f7f975b4ad2f280d26ca5f12f8911fd04`; champion `cb0642bc324a4da62cfa93338d325865857c0742`; decision discard.
  [Implementation](ideas/sella-large-iterative-stepper/6a7cc16f7f975b4ad2f280d26ca5f12f8911fd04/algo.py).
  [Training evidence](evaluation_results/cycle-273-train.json).
  [Narrative](full_log.md#cycle-273--iterative-cartesian-realization-on-connected-n_atoms80).

## sella-large-3pt-gdiis: three-point C1 GDIIS on connected n_atoms≥80
Status: closed

**Hypothesis:** Two-point GDIIS never accepts on n≥80. Three-point C1 may still interpolate venetoclax.

**Outcome and uncertainty:** Cycle 274 discard, bit-identical. Three-point C1 also does not accept.

**Reason to revisit:** Not as a longer interpolant on n≥80.

**Next experiment:** Do not add GDIIS points on n≥80. Try size-gated RFO@50 on n≥80 to split paliperidone from 160853090.

**Attempts:**
- Cycle 274; candidate `eed031a47bbb4c470019c7ac91bf91f3513c4fc4`; champion `cb0642bc324a4da62cfa93338d325865857c0742`; decision discard.
  [Implementation](ideas/sella-large-3pt-gdiis/eed031a47bbb4c470019c7ac91bf91f3513c4fc4/algo.py).
  [Training evidence](evaluation_results/cycle-274-train.json).
  [Narrative](full_log.md#cycle-274--three-point-c1-gdiis-on-connected-n_atoms80).

## sella-large-rfo-50: Banerjee RFO after 50 steps on connected n_atoms≥80
Status: closed

**Hypothesis:** Connected RFO@50 extras were medium n=48–50 jobs. Size-gating to n≥80 keeps paliperidone's RFO save.

**Outcome and uncertainty:** Cycle 275 RFO@50 discard, paliperidone 78→77. Cycle 276 RFO@45 non_generalizable: train paliperidone 78→73, Δ=+1.37e-4; valid Δ=0 because this valid split has max n_atoms=50.

**Reason to revisit:** Not as an n≥80-only stepper. Valid cannot see n≥80.

**Next experiment:** Do not size-gate steppers at n≥80. Dummy-angle H0 on n≥25 (spare 18–24) can fire on valid.

**Attempts:**
- Cycle 276; candidate `e52400eddff57bebf6eb9736be9ed7b1a3202146`; champion `cb0642bc324a4da62cfa93338d325865857c0742`; decision non_generalizable.
  [Implementation](ideas/sella-large-rfo-50/e52400eddff57bebf6eb9736be9ed7b1a3202146/algo.py).
  [Training evidence](evaluation_results/cycle-276-train.json).
  [Validation evidence](evaluation_results/cycle-276-valid.json).
  [Narrative](full_log.md#cycle-276--banerjee-rfo-after-45-steps-on-connected-n_atoms80).
- Cycle 275; candidate `975985d30ecf4dc5e587a20a58003952ee6768da`; champion `cb0642bc324a4da62cfa93338d325865857c0742`; decision discard.
  [Implementation](ideas/sella-large-rfo-50/975985d30ecf4dc5e587a20a58003952ee6768da/algo.py).
  [Training evidence](evaluation_results/cycle-275-train.json).
  [Narrative](full_log.md#cycle-275--banerjee-rfo-after-50-steps-on-connected-n_atoms80).

## sella-oxo-angle-h0: 0.10 Ha Hessian on 2-coordinate oxygen angles for n<12
Status: incorporated

**Hypothesis:** Leftover 252089162 (pyrophosphate) is limited by stiff Fischer H0 on P–O–P / P–O–H. Soft 0.10 Ha on 2-coordinate oxygen angles for connected n<12 cheapens it without dummy-linear n≥30.

**Outcome and uncertainty:** Cycle 264 invalid (Si–O–Si hop). Cycle 265 **keep**: P-neighbor gate, train Δ=+1.15e-4, valid Δ=+1.08e-4. Cycle 267 0.08 Ha discard, bit-identical. Cycle 280 **keep**: complementary O–P–O at phosphorus, train Δ=+1.13e-4 (135094180 26→24; 252089162 39→40 extra), valid Δ=+1.08e-4 (135107889 19→18). New champion `1c692605a4ac0d6cc4ed04b97808bfd340bc6640`.

**Reason to revisit:** Incorporated. P–O–P 0.10 and O–P–O 0.10 both keep. Do not interpolate 0.08. Do not drop protonated phosphates (valid save is H-bearing).

**Next experiment:** Do not soften phosphate H0 further. Do not interpolate oxo 0.08. Complementary S–P–S at phosphorus is the thiophosphate test.

**Attempts:**
- Cycle 280; candidate `1c692605a4ac0d6cc4ed04b97808bfd340bc6640`; champion `cb0642bc324a4da62cfa93338d325865857c0742`; decision keep.
  [Training evidence](evaluation_results/cycle-280-train.json).
  [Validation evidence](evaluation_results/cycle-280-valid.json).
  [Narrative](full_log.md#cycle-280--010-ha-opo-hessian-on-connected-n_atoms12).
- Cycle 267; candidate `e41e77fb6ead88ecb4e744ec175a7bc570ce3da6`; champion `488f94cdf1a71e13529b7873dd559a470dbf54e2`; decision discard.
  [Implementation](ideas/sella-oxo-angle-h0/e41e77fb6ead88ecb4e744ec175a7bc570ce3da6/algo.py).
  [Training evidence](evaluation_results/cycle-267-train.json).
  [Narrative](full_log.md#cycle-267--008-ha-p-neighbor-oxo-hessian-on-connected-n_atoms12).
- Cycle 265; candidate `488f94cdf1a71e13529b7873dd559a470dbf54e2`; champion `63984199efc0776436013dcfe4f7a7dd1a8a586e`; decision keep.
  [Training evidence](evaluation_results/cycle-265-train.json).
  [Validation evidence](evaluation_results/cycle-265-valid.json).
  [Narrative](full_log.md#cycle-265--010-ha-hessian-on-p-neighbor-2-coordinate-oxygen-angles-for-n_atoms12).
- Cycle 264; candidate `2a8ecd25370f6a24bb7aa2711794604e3c14b24c`; champion `63984199efc0776436013dcfe4f7a7dd1a8a586e`; decision invalid.
  [Implementation](ideas/sella-oxo-angle-h0/2a8ecd25370f6a24bb7aa2711794604e3c14b24c/algo.py).
  [Training evidence](evaluation_results/cycle-264-train.json).
  [Narrative](full_log.md#cycle-264--010-ha-hessian-on-2-coordinate-oxygen-angles-for-n_atoms12).

- Cycle 262; candidate `5f3bae96c61f0878a727f9ab4c2849eb0e8aff3c`; champion `63984199efc0776436013dcfe4f7a7dd1a8a586e`; decision discard.
  [Implementation](ideas/sella-small-rfo/5f3bae96c61f0878a727f9ab4c2849eb0e8aff3c/algo.py).
  [Training evidence](evaluation_results/cycle-262-train.json).
  [Narrative](full_log.md#cycle-262--banerjee-rfo-after-20-steps-on-connected-n_atoms12).
- Cycle 261; candidate `dc95af35fffde0b5106c822474319373c121cb1a`; champion `63984199efc0776436013dcfe4f7a7dd1a8a586e`; decision discard.
  [Implementation](ideas/sella-small-rfo/dc95af35fffde0b5106c822474319373c121cb1a/algo.py).
  [Training evidence](evaluation_results/cycle-261-train.json).
  [Narrative](full_log.md#cycle-261--banerjee-rfo-on-connected-n_atoms12).

## sella-thiophosphate-angle-h0: 0.10 Ha P–S–P / S–P–S Hessian on connected n_atoms<12
Status: incorporated

**Hypothesis:** P4S6 extra is the sulfur analog of phosphate H0 keeps. 2-coordinate P–S–P or complementary S–P–S at P on n<12 cheapens the cage.

**Outcome and uncertainty:** Cycle 292 P–S–P discard, Δ=0, step-identical. Cycle 293 S–P–S non_generalizable: train P4S6 17→16, valid 135107889 18→19 extra. Cycle 620 **keep** restacked 0.10 Ha S–P–S gated to 3-coordinate P {S,S,S} on cycle 607: leftover 135094613 17→16, valid 135107889 stays 18.

**Reason to revisit:** Incorporated as oxygen-free 3-coord S–P–S. Do not drop the mixed S–P–O skip.

**Next experiment:** None. See leftover 135041973 stiffer O–S–C on n<12 sulfoxides.

**Attempts:**
- Cycle 620; candidate `17ef5a0334bd45194f97546bcbbb2ffc13b33951`; champion `4eb6cafb44eeb92e19753e82045a8573ed8f96cb`; decision keep.
  [Implementation](ideas/sella-n12-p4s6-sps-h0-stack/17ef5a0334bd45194f97546bcbbb2ffc13b33951/algo.py).
  [Training evidence](evaluation_results/cycle-620-train.json).
  [Validation evidence](evaluation_results/cycle-620-valid.json).
  [Narrative](full_log.md#cycle-620--cycle-607-plus-010-ha-oxygen-free-sps-on-n12-p4s6).
- Cycle 293; candidate `1fc637bedd205a1d109e13a976ac6c89be720cbf`; champion `7b0f85a2864652b033bd8929e169aabb0468c5b7`; decision non_generalizable.
  [Implementation](ideas/sella-thiophosphate-angle-h0/1fc637bedd205a1d109e13a976ac6c89be720cbf/algo.py).
  [Training evidence](evaluation_results/cycle-293-train.json).
  [Validation evidence](evaluation_results/cycle-293-valid.json).
  [Narrative](full_log.md#cycle-293--010-ha-sps-hessian-on-connected-n_atoms12).
- Cycle 292; candidate `15e25d9129e6a5e67d9a9950afd11764c4b12ce4`; champion `7b0f85a2864652b033bd8929e169aabb0468c5b7`; decision discard.
  [Implementation](ideas/sella-oxo-angle-h0/15e25d9129e6a5e67d9a9950afd11764c4b12ce4/algo.py).
  [Training evidence](evaluation_results/cycle-292-train.json).
  [Narrative](full_log.md#cycle-292--010-ha-p-neighbor-2-coordinate-sulfur-hessian-on-n_atoms12).

## sella-fluoride-angle-h0: 0.10 Ha F–Si–X / Cl–Si–X / F–B–F Hessian on connected n_atoms<12
Status: incorporated

**Hypothesis:** Remaining extra 135095518 is a B/Si fluoride. Soft F–Si–F / F–B–F H0 on n<12 is the analog of cycle 280 O–P–O. Mixed F–Si–X and Cl–Si–X follow.

**Outcome and uncertainty:** Cycle 283 non_generalizable (F–Si–F train-only). Cycle 284 **keep**: F–Si–X, train Δ=+1.12e-3 (135095518 33→26, 135094844 10→7), valid Δ=+2.15e-4 (135097864 30→27). Cycle 289 **keep**: Cl–Si–X, train Δ=+5.33e-4 (135039840 16→12), valid Δ=+1.96e-4 (135034037 11→10). Cycle 295 Br–Si–X non_generalizable: train SiBr4 9→7, valid Δ=0 (no n<12 Br–Si). New champion remains `7b0f85a2864652b033bd8929e169aabb0468c5b7`.

**Reason to revisit:** Incorporated as F–Si–X, Cl–Si–X, and F–B–F. Do not broaden to all Si-center angles. Br–Si–X / I–Si–X lack n<12 valid coverage on this split.

**Next experiment:** Do not interpolate Si-halide H0 further. Complementary O–S–O at sulfur is the sulfate analog of cycle 280 O–P–O.

**Attempts:**
- Cycle 295; candidate `6f0ea4f408bbba0e797f1d29cf495cf441d123b8`; champion `7b0f85a2864652b033bd8929e169aabb0468c5b7`; decision non_generalizable.
  [Implementation](ideas/sella-fluoride-angle-h0/6f0ea4f408bbba0e797f1d29cf495cf441d123b8/algo.py).
  [Training evidence](evaluation_results/cycle-295-train.json).
  [Validation evidence](evaluation_results/cycle-295-valid.json).
  [Narrative](full_log.md#cycle-295--010-ha-brsix-hessian-on-connected-n_atoms12).
- Cycle 289; candidate `7b0f85a2864652b033bd8929e169aabb0468c5b7`; champion `c0cdb7301cf263f5559e19ef89b48d4030fe675d`; decision keep.
  [Training evidence](evaluation_results/cycle-289-train.json).
  [Validation evidence](evaluation_results/cycle-289-valid.json).
  [Narrative](full_log.md#cycle-289--010-ha-clsix-hessian-on-connected-n_atoms12).
- Cycle 284; candidate `c0cdb7301cf263f5559e19ef89b48d4030fe675d`; champion `1c692605a4ac0d6cc4ed04b97808bfd340bc6640`; decision keep.
  [Training evidence](evaluation_results/cycle-284-train.json).
  [Validation evidence](evaluation_results/cycle-284-valid.json).
  [Narrative](full_log.md#cycle-284--010-ha-fsix-hessian-on-connected-n_atoms12).

## sella-fluoride-phosphorus-h0: 0.10 Ha F–P–X Hessian on connected n_atoms<12
Status: closed

**Hypothesis:** PF5/PF3 bending is the SiF4 analog of kept F–Si–X. Soft 0.10 Ha F–P–X on connected n<12 cheapens fluoride phosphorus without leftover dummy-linear n≥30.

**Outcome and uncertainty:** Cycle 290 discard, Δ=−3.77e-4 (135099841 21→20; 135101309 26→32). Cycle 291 PF4 skip discard, Δ=−6.10e-5 (135099841 21→20; 135101309 26→28). The extra is PF2–N on 135101309. Sparing that molecule leaves only 135099841 −1 below the train gate.

**Reason to revisit:** Not as F–P–X H0 interpolation. Train coverage is two molecules and they cannot both move the right way.

**Next experiment:** Do not interpolate n_F(P) or 0.08 Ha. Try 2-coordinate P-neighbor sulfur angles (P–S–P / P–S–H) analog of cycle 265.

**Attempts:**
- Cycle 291; candidate `4d024eeec4fa0f6c43f5b4cdcb9f615c534b38a3`; champion `7b0f85a2864652b033bd8929e169aabb0468c5b7`; decision discard.
  [Implementation](ideas/sella-fluoride-phosphorus-h0/4d024eeec4fa0f6c43f5b4cdcb9f615c534b38a3/algo.py).
  [Training evidence](evaluation_results/cycle-291-train.json).
  [Narrative](full_log.md#cycle-291--fpx-hessian-excluding-pf4pf5-centers).
- Cycle 290; candidate `d6673ca6431983582dbb0975f562d60702b0ce3d`; champion `7b0f85a2864652b033bd8929e169aabb0468c5b7`; decision discard.
  [Implementation](ideas/sella-fluoride-phosphorus-h0/d6673ca6431983582dbb0975f562d60702b0ce3d/algo.py).
  [Training evidence](evaluation_results/cycle-290-train.json).
  [Narrative](full_log.md#cycle-290--010-ha-fpx-hessian-on-connected-n_atoms12).

## sella-window-dummy-angle-h0: 0.08 Ha dummy-angle H0 on windowed alkyne dummies
Status: closed

**Hypothesis:** Dummy-angle 0.08 on windowed C–C–C alkynes cheapens leftover without dummy-dihedral 0.20 or nitrile extras.

**Outcome and uncertainty:** Cycle 294 discard, Δ=−1.02e-5. Leftover 44→43 eaten by 135191719 25→26. 134979308 unchanged.

**Reason to revisit:** Not as dummy-angle below 0.10 on alkynes. Cycle 286 dummy-dihedral 0.20 remains the better leftover unused.

**Next experiment:** Do not interpolate 0.09/0.05. Do not restack dummy-dihedral 0.20/0.18.

**Attempts:**
- Cycle 294; candidate `7fa78aad48d4a4f859dd95070fda97c0c87e418e`; champion `7b0f85a2864652b033bd8929e169aabb0468c5b7`; decision discard.
  [Implementation](ideas/sella-window-dummy-angle-h0/7fa78aad48d4a4f859dd95070fda97c0c87e418e/algo.py).
  [Training evidence](evaluation_results/cycle-294-train.json).
  [Narrative](full_log.md#cycle-294--008-ha-dummy-angle-h0-on-windowed-alkyne-dummies).

## sella-window-dummy-dihedral-h0: 0.20 Ha dummy-dihedral H0 on windowed linear-frame dummies
Status: incorporated

**Hypothesis:** Cycle 105 0.20 dummy-dihedral cheapened leftover but wandered 104046121. Windowed 0.04–0.10 dummies on n≥30 keep leftover and spare the canary. Near-collinear isocyanate N=C=O dummies need the same 0.20 scale without the alkyne window.

**Outcome and uncertainty:** Cycle 286 alkyne-only 0.20 non_generalizable (valid Δ=+5.12e-5). Cycle 304 windowed isocyanate no-op. Cycle 305 **keep**: unwindowed isocyanate {N,O} plus windowed C–C–C, train Δ=+1.87e-4, valid Δ=+1.56e-4 (135249279 42→40). New champion `a2c46dbb6142210fe6d4e0cd45c0254c7be95ae4`.

**Reason to revisit:** Incorporated. Do not apply 0.20 to nitriles (cycle 285 252634821 extra). Do not tag isocyanide C–N–C dummies (cycle 307/440 leftover twitch, n_steps unchanged). Leftover 363892164 still +12 vs ref.

**Next experiment:** Do not interpolate dummy-dihedral 0.18/0.16 on alkynes or isocyanide 0.20. Keep 0.20 on windowed alkynes and unwindowed isocyanates. Do not restack N-center adj dummy.

**Attempts:**
- Cycle 440; candidate `f270acd1902a1f93726025473a8eb773f3115a83`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision discard.
  [Implementation](ideas/sella-window-dummy-dihedral-h0/f270acd1902a1f93726025473a8eb773f3115a83/algo.py).
  [Training evidence](evaluation_results/cycle-440-train.json).
  [Narrative](full_log.md#cycle-440--window-isocyanide-n-dummies-at-020-ha-dummy-dihedral).
- Cycle 307; candidate `f5678cc5efac73df98fb69f38fd973314604fd02`; champion `a2c46dbb6142210fe6d4e0cd45c0254c7be95ae4`; decision discard.
  [Implementation](ideas/sella-window-dummy-dihedral-h0/f5678cc5efac73df98fb69f38fd973314604fd02/algo.py).
  [Training evidence](evaluation_results/cycle-307-train.json).
  [Narrative](full_log.md#cycle-307--020-ha-dummy-dihedral-on-isocyanide-cnc-dummies).
- Cycle 306; candidate `6aa3a840a621dff4b0597c84f30c260ae420cdde`; champion `a2c46dbb6142210fe6d4e0cd45c0254c7be95ae4`; decision discard.
  [Implementation](ideas/sella-window-dummy-dihedral-h0/6aa3a840a621dff4b0597c84f30c260ae420cdde/algo.py).
  [Training evidence](evaluation_results/cycle-306-train.json).
  [Narrative](full_log.md#cycle-306--018-ha-dummy-dihedral-on-windowed-alkynes-isocyanate-stays-020).
- Cycle 305; candidate `a2c46dbb6142210fe6d4e0cd45c0254c7be95ae4`; champion `7b0f85a2864652b033bd8929e169aabb0468c5b7`; decision keep.
  [Training evidence](evaluation_results/cycle-305-train.json).
  [Validation evidence](evaluation_results/cycle-305-valid.json).
  [Narrative](full_log.md#cycle-305--020-ha-dummy-dihedral-on-all-isocyanate-dummies-plus-windowed-alkynes).
- Cycle 304; candidate `02d01ed3f2cfd662855e30231c64b4b508005efd`; champion `7b0f85a2864652b033bd8929e169aabb0468c5b7`; decision non_generalizable.
  [Implementation](ideas/sella-window-dummy-dihedral-h0/02d01ed3f2cfd662855e30231c64b4b508005efd/algo.py).
  [Training evidence](evaluation_results/cycle-304-train.json).
  [Validation evidence](evaluation_results/cycle-304-valid.json).
  [Narrative](full_log.md#cycle-304--020-ha-dummy-dihedral-on-windowed-alkyne-and-isocyanate-dummies).
- Cycle 298; candidate `58a42e69f0979d647fa8dff787308d989ad5b9e3`; champion `7b0f85a2864652b033bd8929e169aabb0468c5b7`; decision non_generalizable.
  [Implementation](ideas/sella-window-dummy-dihedral-h0/58a42e69f0979d647fa8dff787308d989ad5b9e3/algo.py).
  [Training evidence](evaluation_results/cycle-298-train.json).
  [Validation evidence](evaluation_results/cycle-298-valid.json).
  [Narrative](full_log.md#cycle-298--020-ha-dummy-dihedral-h0-on-all-size-windowed-alkyne-dummies).
- Cycle 288; candidate `36c7d686893b61b6e4e2c394eab53a002bd429b6`; champion `c0cdb7301cf263f5559e19ef89b48d4030fe675d`; decision non_generalizable.
  [Implementation](ideas/sella-window-dummy-dihedral-h0/36c7d686893b61b6e4e2c394eab53a002bd429b6/algo.py).
  [Training evidence](evaluation_results/cycle-288-train.json).
  [Validation evidence](evaluation_results/cycle-288-valid.json).
  [Narrative](full_log.md#cycle-288--018-ha-dummy-dihedral-h0-on-windowed-alkyne-dummies).
- Cycle 287; candidate `ffdaa760c8189d2aebac46a7a57b823781190035`; champion `c0cdb7301cf263f5559e19ef89b48d4030fe675d`; decision non_generalizable.
  [Implementation](ideas/sella-window-dummy-dihedral-h0/ffdaa760c8189d2aebac46a7a57b823781190035/algo.py).
  [Training evidence](evaluation_results/cycle-287-train.json).
  [Validation evidence](evaluation_results/cycle-287-valid.json).
  [Narrative](full_log.md#cycle-287--alkyne-dummy-dihedral-020-plus-cnc-n-adj-dummy).
- Cycle 286; candidate `cc681c46673b9e6d6cdd57c3a53397e9c19835b8`; champion `c0cdb7301cf263f5559e19ef89b48d4030fe675d`; decision non_generalizable.
  [Implementation](ideas/sella-window-dummy-dihedral-h0/cc681c46673b9e6d6cdd57c3a53397e9c19835b8/algo.py).
  [Training evidence](evaluation_results/cycle-286-train.json).
  [Validation evidence](evaluation_results/cycle-286-valid.json).
  [Narrative](full_log.md#cycle-286--020-ha-dummy-dihedral-h0-on-windowed-alkyne-dummies).
- Cycle 285; candidate `ded936fb3e3ac10c4b28cd747507fb3229862ea8`; champion `c0cdb7301cf263f5559e19ef89b48d4030fe675d`; decision non_generalizable.
  [Implementation](ideas/sella-window-dummy-dihedral-h0/ded936fb3e3ac10c4b28cd747507fb3229862ea8/algo.py).
  [Training evidence](evaluation_results/cycle-285-train.json).
  [Validation evidence](evaluation_results/cycle-285-valid.json).
  [Narrative](full_log.md#cycle-285--020-ha-dummy-dihedral-h0-on-windowed-linear-frame-dummies).

- Cycle 283; candidate `409927b7e357bb3c0288d198698a15f77760597d`; champion `1c692605a4ac0d6cc4ed04b97808bfd340bc6640`; decision non_generalizable.
  [Implementation](ideas/sella-fluoride-angle-h0/409927b7e357bb3c0288d198698a15f77760597d/algo.py).
  [Training evidence](evaluation_results/cycle-283-train.json).
  [Validation evidence](evaluation_results/cycle-283-valid.json).
  [Narrative](full_log.md#cycle-283--010-ha-fsif--fbf-hessian-on-connected-n_atoms12).


## sella-free-dummy-angle: unconstrained dummy–center–real angle on connected n_atoms≥30
Status: closed

**Hypothesis:** Sella constrains one dummy–center–real angle. Leaving it free on n≥30 lets dummy-angle H0 0.10 act on leftover linear-bend frames.

**Outcome and uncertainty:** Cycle 281 discard, Δ=+6.17e-6. Leftover unchanged. Saves 134979308/319495631 −1 eaten by 104046121 33→35. Cycle 282 carbon-adj-only discard, Δ=−5.92e-5: 104046121 stays 33, 134979308 31→32 extra, 319495631 save gone.

**Reason to revisit:** Not as a dummy-angle-constraint skip. Leftover does not move; carbon-adj-only is harmful; all-n≥30 extras the canary.

**Next experiment:** Do not interpolate which dummy angles to unconstrain. Try F–Si–F / F–B–F 0.10 Ha on connected n<12.

**Attempts:**
- Cycle 282; candidate `c2e5cd4b18ac343365a5043087fdd5c9ace0737c`; champion `1c692605a4ac0d6cc4ed04b97808bfd340bc6640`; decision discard.
  [Implementation](ideas/sella-free-dummy-angle/c2e5cd4b18ac343365a5043087fdd5c9ace0737c/algo.py).
  [Training evidence](evaluation_results/cycle-282-train.json).
  [Narrative](full_log.md#cycle-282--unconstrained-dummy-angle-on-carbon-adj-dummies-only).
- Cycle 281; candidate `0695cca326fb8dd186be1d9937b20298263be30e`; champion `1c692605a4ac0d6cc4ed04b97808bfd340bc6640`; decision discard.
  [Implementation](ideas/sella-free-dummy-angle/0695cca326fb8dd186be1d9937b20298263be30e/algo.py).
  [Training evidence](evaluation_results/cycle-281-train.json).
  [Narrative](full_log.md#cycle-281--unconstrained-dummy-angle-on-connected-n_atoms30).

## sella-sulfate-angle-h0: 0.10 Ha O–S–O Hessian on connected n_atoms<12
Status: closed

**Hypothesis:** Cycle 280 kept 0.10 Ha O–P–O at phosphorus. Sulfate/sulfone O–S–O is the same tetrahedral oxyanion bend (Huige–Altona AMBER sulfate parameters; tetrahedral SO4 force fields). Soft 0.10 Ha on connected n<12 cheapens those frames without leftover dummy-linear n≥30 or S–P–S (135107889 has no O–S–O).

**Outcome and uncertainty:** Cycle 296 discard, Δ=−6.87e-4 (135093116 14→13; extras 135121768 17→19, 163350562 29→37). Cycle 297 S-neighbor non_generalizable: train 135093116 14→13, valid 135169088 15→16 extra. The only valid S-neighbor O–S–O molecule extras.

**Reason to revisit:** Not as O–S–O H0 interpolation. Sparing 135169088 leaves train-only 135093116.

**Next experiment:** Do not interpolate 0.15 or n_bonds≥5. Windowed C–C–C dummy-dihedral 0.20 on n<18 leftover 135169446 is next.

**Attempts:**
- Cycle 297; candidate `3929a49e883bb57a65e377049f26636596fffb48`; champion `7b0f85a2864652b033bd8929e169aabb0468c5b7`; decision non_generalizable.
  [Implementation](ideas/sella-sulfate-angle-h0/3929a49e883bb57a65e377049f26636596fffb48/algo.py).
  [Training evidence](evaluation_results/cycle-297-train.json).
  [Validation evidence](evaluation_results/cycle-297-valid.json).
  [Narrative](full_log.md#cycle-297--oso-hessian-only-at-sulfur-neighbor-s-centers).
- Cycle 296; candidate `09c3ae33178188ffcb8a44494f1fe634d0b829df`; champion `7b0f85a2864652b033bd8929e169aabb0468c5b7`; decision discard.
  [Implementation](ideas/sella-sulfate-angle-h0/09c3ae33178188ffcb8a44494f1fe634d0b829df/algo.py).
  [Training evidence](evaluation_results/cycle-296-train.json).
  [Narrative](full_log.md#cycle-296--010-ha-oso-hessian-on-connected-n_atoms12).

## sella-nitroso-angle-h0: 0.10 Ha C–N–O Hessian on connected n_atoms<18
Status: closed

**Hypothesis:** Nitroso C–N–O is a 2-coordinate nitrogen bend distinct from dummy-linear frames. Soft 0.10 Ha on connected n<18 cheapens those frames without leftover dummy-linear n≥30.

**Outcome and uncertainty:** Cycle 300 discard, Δ=−1.33e-4. Only 135099623 15→16 extra; 135247419/135238882 unchanged. Energy-safe. Cycle 308 invalid: 18–29 C–N–O hops 135043047 43→32 / +38.01 kcal. Oxime 135295734 15→16 energy-safe.

**Reason to revisit:** Not as a C–N–O H0 scale. n<18 extras; n=18 hops the 38 kcal canary.

**Next experiment:** Do not interpolate 0.13/0.15 or n<19. Close C–N–O.

**Attempts:**
- Cycle 308; candidate `66fff8a88e19986c093651441319c217ef39accc`; champion `a2c46dbb6142210fe6d4e0cd45c0254c7be95ae4`; decision invalid.
  [Implementation](ideas/sella-nitroso-angle-h0/66fff8a88e19986c093651441319c217ef39accc/algo.py).
  [Training evidence](evaluation_results/cycle-308-train.json).
  [Narrative](full_log.md#cycle-308--010-ha-cno-hessian-on-connected-18natoms30).
- Cycle 300; candidate `a993737fec7228297ab739023ea1c034b5e47bb2`; champion `7b0f85a2864652b033bd8929e169aabb0468c5b7`; decision discard.
  [Implementation](ideas/sella-nitroso-angle-h0/a993737fec7228297ab739023ea1c034b5e47bb2/algo.py).
  [Training evidence](evaluation_results/cycle-300-train.json).
  [Narrative](full_log.md#cycle-300--010-ha-cno-hessian-on-connected-n_atoms18).

## sella-azide-angle-h0: 0.10 Ha C–N–N / N–N–N Hessian on connected n_atoms<12
Status: closed

**Hypothesis:** Bent C–N–N / N–N–N still use Fischer H0 while linear azides already have dummy-angle 0.10. Soft 0.10 Ha on connected n<12 cheapens diazo/azide frames.

**Outcome and uncertainty:** Cycle 301 discard, Δ=−3.76e-4. Only 135264337 17→20 extra; 135098945 unchanged.

**Reason to revisit:** Not as C–N–N / N–N–N H0 on n<12. The polyazide extras; cyanogen does not save further.

**Next experiment:** Do not interpolate N–N–N vs C–N–N. Cycle 286 dummy-dihedral 0.20 remains the best unused leftover near-miss.

**Attempts:**
- Cycle 301; candidate `e18df70ab85ddd2456ced5921b8829ab9ddf8b8f`; champion `7b0f85a2864652b033bd8929e169aabb0468c5b7`; decision discard.
  [Implementation](ideas/sella-azide-angle-h0/e18df70ab85ddd2456ced5921b8829ab9ddf8b8f/algo.py).
  [Training evidence](evaluation_results/cycle-301-train.json).
  [Narrative](full_log.md#cycle-301--010-ha-cnn-hessian-on-connected-n_atoms12).

## sella-sulfoxide-angle-h0: 0.10 Ha O–S–C Hessian on connected n_atoms<12
Status: closed

**Hypothesis:** Leftover 135041973 is a sulfoxide limited by stiff Fischer O–S–C, not by O–S–O. Soft O–S–C on connected n<12 cheapens it and valid 442142528 without sulfate extras.

**Outcome and uncertainty:** Cycle 302 0.10 Ha discard, Δ=−1.78e-4, only 135041973 13→14 extra. Cycle 303 0.13 Ha (GAFF-like) discard, step-identical extra. Softening O–S–C extras the leftover.

**Reason to revisit:** Not as an O–S–C H0 scale. The leftover is not a too-stiff sulfoxide bend.

**Next experiment:** Do not interpolate 0.15. Restack cycle 286 dummy-dihedral 0.20 with isocyanate C–N–O windowed dummies.

**Attempts:**
- Cycle 303; candidate `11760eab7303a04e77371b0666a6efc5879b75a0`; champion `7b0f85a2864652b033bd8929e169aabb0468c5b7`; decision discard.
  [Implementation](ideas/sella-sulfoxide-angle-h0/11760eab7303a04e77371b0666a6efc5879b75a0/algo.py).
  [Training evidence](evaluation_results/cycle-303-train.json).
  [Narrative](full_log.md#cycle-303--013-ha-osc-hessian-on-connected-n_atoms12).
- Cycle 302; candidate `01edfbbd6282c60d7a419907c452805bf91bf417`; champion `7b0f85a2864652b033bd8929e169aabb0468c5b7`; decision discard.
  [Implementation](ideas/sella-sulfoxide-angle-h0/01edfbbd6282c60d7a419907c452805bf91bf417/algo.py).
  [Training evidence](evaluation_results/cycle-302-train.json).
  [Narrative](full_log.md#cycle-302--010-ha-osc-hessian-on-connected-n_atoms12).

## sella-po-stretch-h0: 0.25 Ha/Bohr² P–O stretch Hessian on connected n_atoms<12
Status: closed

**Hypothesis:** Diphosphite leftover 252089162 stays +3 after O–P–O / P–O–P angle 0.10. Soft P–O stretches (0.25 vs Fischer ~0.36 Ha/Bohr²) on the same n<12 band cheapen those last calls.

**Outcome and uncertainty:** Cycle 309 discard, Δ=−5.33e-4. Energy-safe. 252089162 unchanged 40. Extras 134989033 and 135156363 8→9. Stretch H0 is not the leftover limiter.

**Reason to revisit:** Not as a P–O stretch scale. The leftover did not move; phosphoryl/thiophosphate extras.

**Next experiment:** Do not interpolate 0.30/0.20. Try isolated gem-difluoro F–C–C 0.10 Ha on connected n≥30.

**Attempts:**
- Cycle 309; candidate `8444040b2a33a31b71352860d48f5e07f7cd9e76`; champion `a2c46dbb6142210fe6d4e0cd45c0254c7be95ae4`; decision discard.
  [Implementation](ideas/sella-po-stretch-h0/8444040b2a33a31b71352860d48f5e07f7cd9e76/algo.py).
  [Training evidence](evaluation_results/cycle-309-train.json).
  [Narrative](full_log.md#cycle-309--025-habohr-po-stretch-hessian-on-connected-n_atoms12).

## sella-isolated-cf2-angle-h0: isolated gem-difluoro F–C–C Hessian on connected n_atoms≥30
Status: closed

**Hypothesis:** Leftover 363892164’s unused coordinate is isolated CF2 F–C–C (exactly two F on 4-coord C; carbon neighbor not fluorinated). Soft 0.10 Ha cheapens it without perfluoro/CF3 extras.

**Outcome and uncertainty:** Cycle 310 discard, Δ=−6.88e-5. Energy-safe. Only leftover 43→44 extra. Isolation spared 135064751. Soft F–C–C overshoots; GAFF `c3-c3-f` is stiffer than Fischer. Cycle 311 0.21 Ha discard, Δ=0, leftover energy twitch only. Cycle 312 C–C–C 0.10 discard, Δ=0. Cycle 313 F–C–F 0.10 discard, Δ=0. Cycle 329 adjacent-CH2 C–C–C discard, Δ=0, leftover energy twitch only.

**Reason to revisit:** Not as isolated CF2 angle H0. F–C–C extras leftover; C–C–C / F–C–F / adjacent-CH2 C–C–C do not change n_steps.

**Next experiment:** Do not interpolate C–C–H at CF2-adjacent CH2. Close this family.

**Attempts:**
- Cycle 329; candidate `d7facbdf237b162b36079e7b8236bf7ff72dd69c`; champion `362a4dd6db5ca11eea73f992d86141a50908bd75`; decision discard.
  [Implementation](ideas/sella-isolated-cf2-angle-h0/d7facbdf237b162b36079e7b8236bf7ff72dd69c/algo.py).
  [Training evidence](evaluation_results/cycle-329-train.json).
  [Narrative](full_log.md#cycle-329--010-ha-ccc-at-ch2-adjacent-to-cf2-on-connected-30n_atoms80).
- Cycle 313; candidate `edb70db03d938e7d366356b33218e9ecbf2bc7a9`; champion `a2c46dbb6142210fe6d4e0cd45c0254c7be95ae4`; decision discard.
  [Implementation](ideas/sella-isolated-cf2-angle-h0/edb70db03d938e7d366356b33218e9ecbf2bc7a9/algo.py).
  [Training evidence](evaluation_results/cycle-313-train.json).
  [Narrative](full_log.md#cycle-313--010-ha-isolated-gem-difluoro-fcf-hessian-on-connected-n_atoms30).
- Cycle 312; candidate `85b80b92f7bedd7973b52df842565da558ec1e9c`; champion `a2c46dbb6142210fe6d4e0cd45c0254c7be95ae4`; decision discard.
  [Implementation](ideas/sella-isolated-cf2-angle-h0/85b80b92f7bedd7973b52df842565da558ec1e9c/algo.py).
  [Training evidence](evaluation_results/cycle-312-train.json).
  [Narrative](full_log.md#cycle-312--010-ha-isolated-gem-difluoro-ccc-hessian-on-connected-n_atoms30).
- Cycle 311; candidate `3cd0d5354edbf3582d6227bfc29866464512f79e`; champion `a2c46dbb6142210fe6d4e0cd45c0254c7be95ae4`; decision discard.
  [Implementation](ideas/sella-isolated-cf2-angle-h0/3cd0d5354edbf3582d6227bfc29866464512f79e/algo.py).
  [Training evidence](evaluation_results/cycle-311-train.json).
  [Narrative](full_log.md#cycle-311--021-ha-isolated-gem-difluoro-fcc-hessian-on-connected-n_atoms30).
- Cycle 310; candidate `9813df6456dac90aa8e48fdcf3d0f334badf5f7a`; champion `a2c46dbb6142210fe6d4e0cd45c0254c7be95ae4`; decision discard.
  [Implementation](ideas/sella-isolated-cf2-angle-h0/9813df6456dac90aa8e48fdcf3d0f334badf5f7a/algo.py).
  [Training evidence](evaluation_results/cycle-310-train.json).
  [Narrative](full_log.md#cycle-310--010-ha-isolated-gem-difluoro-fcc-hessian-on-connected-n_atoms30).

## sella-dimer-rfo40: earlier Banerjee RFO on dimers
Status: closed

**Hypothesis:** Champion dimers switch to Banerjee RFO only after 80 steps. Starting RFO earlier shortens long dimers (phenol–water 80, amides–water 106). A 40-step floor captures those tails; a 70-step floor avoids the 40–50 hop window.

**Outcome and uncertainty:** Cycle 314 RFO@40 invalid, unofficial Δ=+4.04e-4; amines–amines 99→97 / +0.0043 kcal. Cycle 315 RFO@50 invalid, unofficial Δ=+1.14e-3; phenol–water 80→64 / +0.123 kcal; amines recovered. Cycle 316 RFO@70 discard, Δ=−1.40e-5; energy-safe; pyrrole–sulfides 88→86 eaten by alkanes–esters 96→97 and amides–water 106→108. Cycle 317 ρ-gated RFO@40 discard, Δ=−7.92e-4; energy-safe; mixed RFO/QN flicker extras amines–amines 99→117 and benzene–esters 54→65 while alkanes–esters 96→76. Cycle 318 latch RFO after first good ρ at 40 invalid; amines 99→90 / +0.0045 kcal (same 314 hop). Earlier dimer RFO is either energy-unsafe (40/50/latch) or cost-negative (70/ρ-flicker).

**Reason to revisit:** Not as an earlier dimer RFO floor, ρ-gate, or latch. Keep unconditional RFO@80.

**Next experiment:** Do not interpolate latch@50 or consecutive-ρ counts. Close this family.

**Attempts:**
- Cycle 318; candidate `ee8664eacd20563a3bb8ee184c787b447f1f35f9`; champion `a2c46dbb6142210fe6d4e0cd45c0254c7be95ae4`; decision invalid.
  [Implementation](ideas/sella-dimer-rfo40/ee8664eacd20563a3bb8ee184c787b447f1f35f9/algo.py).
  [Training evidence](evaluation_results/cycle-318-train.json).
  [Narrative](full_log.md#cycle-318--latch-banerjee-rfo-after-first-good-ρ-dimer-step-at-40).
- Cycle 317; candidate `bb5a2f7679abb1e1fab3ac2e1d314e714dd4c953`; champion `a2c46dbb6142210fe6d4e0cd45c0254c7be95ae4`; decision discard.
  [Implementation](ideas/sella-dimer-rfo40/bb5a2f7679abb1e1fab3ac2e1d314e714dd4c953/algo.py).
  [Training evidence](evaluation_results/cycle-317-train.json).
  [Narrative](full_log.md#cycle-317--ρ-gated-banerjee-rfo-after-40-steps-on-dimers).
- Cycle 316; candidate `4550c984149c5a7ea6deb7f18e280a2ec7d127fc`; champion `a2c46dbb6142210fe6d4e0cd45c0254c7be95ae4`; decision discard.
  [Implementation](ideas/sella-dimer-rfo40/4550c984149c5a7ea6deb7f18e280a2ec7d127fc/algo.py).
  [Training evidence](evaluation_results/cycle-316-train.json).
  [Narrative](full_log.md#cycle-316--banerjee-rfo-after-70-steps-on-dimers).
- Cycle 315; candidate `a2df343ccbc29589f3d0a9afc1211580edbbe0a1`; champion `a2c46dbb6142210fe6d4e0cd45c0254c7be95ae4`; decision invalid.
  [Implementation](ideas/sella-dimer-rfo40/a2df343ccbc29589f3d0a9afc1211580edbbe0a1/algo.py).
  [Training evidence](evaluation_results/cycle-315-train.json).
  [Narrative](full_log.md#cycle-315--banerjee-rfo-after-50-steps-on-dimers).
- Cycle 314; candidate `2ea1ceb8650375387404b5496fdab32a1d4a523e`; champion `a2c46dbb6142210fe6d4e0cd45c0254c7be95ae4`; decision invalid.
  [Implementation](ideas/sella-dimer-rfo40/2ea1ceb8650375387404b5496fdab32a1d4a523e/algo.py).
  [Training evidence](evaluation_results/cycle-314-train.json).
  [Narrative](full_log.md#cycle-314--banerjee-rfo-after-40-steps-on-dimers).

## sella-ester-angle-h0: 0.10 Ha ester/carbamate C–O–C Hessian on connected n_atoms≥30
Status: closed

**Hypothesis:** Leftover 363892164 is an oxazolidinone. Dummy-linear/CF2/dimer RFO did not cheapen it. Soft 0.10 Ha on ester/carbamate C–O–C (2-coord O, two C, one C bonded to another O) on connected n≥30 targets that unused bend without 104046121 or n<30 ethers.

**Outcome and uncertainty:** Cycle 319 discard, Δ=−4.48e-4. Energy-safe. Leftover unchanged 43. Paliperidone palmitate 78→91; 104000591 23→24. Ester C–O–C is not the leftover limiter.

**Reason to revisit:** Not as an ester C–O–C H0 scale. Leftover did not move; the palmitate ester extras.

**Next experiment:** Do not interpolate 0.13/0.15 or 5-ring-only (still leftover, which is a no-op). Close this family.

**Attempts:**
- Cycle 319; candidate `04f52f5e34190973d6d7d6d56e0b8bd9f7400c60`; champion `a2c46dbb6142210fe6d4e0cd45c0254c7be95ae4`; decision discard.
  [Implementation](ideas/sella-ester-angle-h0/04f52f5e34190973d6d7d6d56e0b8bd9f7400c60/algo.py).
  [Training evidence](evaluation_results/cycle-319-train.json).
  [Narrative](full_log.md#cycle-319--010-ha-estercarbamate-coc-hessian-on-connected-n_atoms30).

## sella-pyridine-angle-h0: 0.10 Ha pyridine/imine C–N–C Hessian on connected n_atoms≥30
Status: incorporated

**Hypothesis:** Leftover 363892164 and venetoclax share 2-coordinate pyridine/imine C–N–C. Ester C–O–C did not move leftover. Soft 0.10 Ha on that class at n≥30.

**Outcome and uncertainty:** Cycle 320 n≥30 discard (paliperidone 78→153). Cycle 321 30≤n<80 non_generalizable (valid hops 135065494 +2.16 kcal, 11109414 +0.97). Cycle 322 guanidine-skip + 1–2 cap non_generalizable (valid V, Δ=+1.26e-5). Cycle 323 skip sulfur-substituted carbons **keep**: train Δ=+8.53e-4, valid Δ=+3.21e-4. New champion `362a4dd6db5ca11eea73f992d86141a50908bd75`.

**Reason to revisit:** Incorporated as 1–2 non-guanidinium, non-S C–N–C 0.10 on 30≤n<80. Do not drop the S/guanidine/count gates (321 hops; 322 thiazole extra). Cycle 324 filling 18–29 hops 252618428.

**Next experiment:** Do not interpolate n<27. Close the 18–29 extension.

**Attempts:**
- Cycle 324; candidate `3a069adf2380d572778c5709e4c9aee42d6bc566`; champion `362a4dd6db5ca11eea73f992d86141a50908bd75`; decision invalid.
  [Implementation](ideas/sella-pyridine-angle-h0/3a069adf2380d572778c5709e4c9aee42d6bc566/algo.py).
  [Training evidence](evaluation_results/cycle-324-train.json).
  [Narrative](full_log.md#cycle-324--same-cnc-010-ha-on-connected-18n_atoms80).
- Cycle 323; candidate `362a4dd6db5ca11eea73f992d86141a50908bd75`; champion `a2c46dbb6142210fe6d4e0cd45c0254c7be95ae4`; decision keep.

## sella-amide-angle-h0: 0.10 Ha 3-coordinate amide C–N–C on connected 30≤n_atoms<80
Status: closed

**Hypothesis:** Leftover 363892164 oxazolidinone NH is a 3-coordinate carbamate C–N–C. Soft 0.10 Ha on 1–2 such angles at 30≤n<80 cheapens leftover without paliperidone.

**Outcome and uncertainty:** Cycle 325 discard, Δ=−9.29e-4 (connected 30–80 extras). Cycle 442 connected 18–30 miss (leftover is a dimer). Cycle 443 dimer+pyridine: leftover 46→44, Δ=+9.48e-5, no extras. Cycle 444 imidazole partner bit-identical to 443. Cycle 445 H-bonded O: train Δ=+1.46e-4 (imidazolium 42→41) but valid amides–pyrrole 29→34. Cycles 446–447 partner XOR/count bit-identical to 445: extra is tagged by pyridine, same as leftover.

**Reason to revisit:** Not as a pyridine/imidazole-gated dimer amide H0. That class is train-only (valid analog extras; no valid save). Preserve 443/444/445.

**Next experiment:** Close this family. See unused connected leftovers (363892164 dummy-linear; carboxylates–esters dimer).

**Attempts:**
- Cycle 447; candidate `0257b497cf5f5489d96ad0042220556cbd5e1239`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-447-train.json).
  [Validation evidence](evaluation_results/cycle-447-valid.json).
  [Narrative](full_log.md#cycle-447--two-imidazole-cnc-or-pyridine-partner-on-dimer-amide-h0).
- Cycle 446; candidate `d0cdbccdc8905d776360fa413cc6d06b9bee43cd`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-446-train.json).
  [Validation evidence](evaluation_results/cycle-446-valid.json).
  [Narrative](full_log.md#cycle-446--pyridine-xor-imidazole-partner-on-dimer-amide-h0).
- Cycle 445; candidate `cba4888fe94b708583716073d8e84e3752c5f5b3`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision non_generalizable.
  [Implementation](ideas/sella-amide-angle-h0/cba4888fe94b708583716073d8e84e3752c5f5b3/algo.py).
  [Training evidence](evaluation_results/cycle-445-train.json).
  [Validation evidence](evaluation_results/cycle-445-valid.json).
  [Narrative](full_log.md#cycle-445--h-bonded-carbonyl-o-on-pyridineimidazole-amide-dimers).
- Cycle 444; candidate `d5b91f1545231da27123f1838bfb4933a1b2d48f`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision discard.
  [Implementation](ideas/sella-amide-angle-h0/d5b91f1545231da27123f1838bfb4933a1b2d48f/algo.py).
  [Training evidence](evaluation_results/cycle-444-train.json).
  [Narrative](full_log.md#cycle-444--010-ha-amide-cnc-on-pyridineimidazole-dimers).
- Cycle 443; candidate `3fd04630379cff5c61c245967cbddcd16e5494c0`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision discard.
  [Implementation](ideas/sella-amide-angle-h0/3fd04630379cff5c61c245967cbddcd16e5494c0/algo.py).
  [Training evidence](evaluation_results/cycle-443-train.json).
  [Narrative](full_log.md#cycle-443--010-ha-amide-cnc-on-pyridine-containing-dimers).
- Cycle 442; candidate `d18b0e7a35357c7e51313b41f244b12fb68a576d`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision discard.
  [Implementation](ideas/sella-amide-angle-h0/d18b0e7a35357c7e51313b41f244b12fb68a576d/algo.py).
  [Training evidence](evaluation_results/cycle-442-train.json).
  [Narrative](full_log.md#cycle-442--010-ha-3-coord-amide-cnc-on-connected-18n30).
- Cycle 325; candidate `4e849374cf6b55b9ba7ff244fc9b106c02cd323d`; champion `362a4dd6db5ca11eea73f992d86141a50908bd75`; decision discard.
  [Implementation](ideas/sella-amide-angle-h0/4e849374cf6b55b9ba7ff244fc9b106c02cd323d/algo.py).
  [Training evidence](evaluation_results/cycle-325-train.json).
  [Narrative](full_log.md#cycle-325--010-ha-3-coordinate-amide-cnc-on-connected-30n_atoms80).

  [Training evidence](evaluation_results/cycle-323-train.json).
  [Validation evidence](evaluation_results/cycle-323-valid.json).
  [Narrative](full_log.md#cycle-323--skip-sulfur-substituted-cnc-010-ha-on-connected-30n_atoms80).
- Cycle 322; candidate `7295b0c166637f1e8baecd602f1301625b318a98`; champion `a2c46dbb6142210fe6d4e0cd45c0254c7be95ae4`; decision non_generalizable.
  [Implementation](ideas/sella-pyridine-angle-h0/7295b0c166637f1e8baecd602f1301625b318a98/algo.py).
  [Training evidence](evaluation_results/cycle-322-train.json).
  [Validation evidence](evaluation_results/cycle-322-valid.json).
  [Narrative](full_log.md#cycle-322--at-most-two-non-guanidinium-cnc-010-ha-on-connected-30n_atoms80).
- Cycle 321; candidate `0b1bc737820ef552918c84bf1aaee8dd7a5f4812`; champion `a2c46dbb6142210fe6d4e0cd45c0254c7be95ae4`; decision non_generalizable.
  [Implementation](ideas/sella-pyridine-angle-h0/0b1bc737820ef552918c84bf1aaee8dd7a5f4812/algo.py).
  [Training evidence](evaluation_results/cycle-321-train.json).
  [Validation evidence](evaluation_results/cycle-321-valid.json).
  [Narrative](full_log.md#cycle-321--010-ha-pyridineimine-cnc-hessian-on-connected-30n_atoms80).
- Cycle 320; candidate `2b75fc642c10ad6fb4786fea84e2e6139eed2345`; champion `a2c46dbb6142210fe6d4e0cd45c0254c7be95ae4`; decision discard.
  [Implementation](ideas/sella-pyridine-angle-h0/2b75fc642c10ad6fb4786fea84e2e6139eed2345/algo.py).
  [Training evidence](evaluation_results/cycle-320-train.json).
  [Narrative](full_log.md#cycle-320--010-ha-pyridineimine-cnc-hessian-on-connected-n_atoms30).

## sella-window-dummy-wd: windowed dummy-dihedral limiter wd=0.7 on connected n≥30
Status: closed

**Hypothesis:** Leftover 363892164 extra calls are MIS-truncated dummy torsions. Cycle 187 global wd=0.7 undid 135043047 (n=18). Applying 0.7 only when the limiter dummy is windowed (n≥30 alkynes/isocyanates) cheapens leftover without that canary.

**Outcome and uncertainty:** Cycle 326 discard, Δ=0, bit-identical. Windowed dummy dihedrals are not the MIS limiter.

**Reason to revisit:** Not as a windowed dummy-wd interpolation. Do not try 0.6.

**Next experiment:** Close windowed dummy-wd. See carbamate O–C–N.

**Attempts:**
- Cycle 326; candidate `9e2c6000d6a7d036f8714f9c2bd58c55963af702`; champion `362a4dd6db5ca11eea73f992d86141a50908bd75`; decision discard.
  [Implementation](ideas/sella-window-dummy-wd/9e2c6000d6a7d036f8714f9c2bd58c55963af702/algo.py).
  [Training evidence](evaluation_results/cycle-326-train.json).
  [Narrative](full_log.md#cycle-326--windowed-dummy-dihedral-limiter-wd07-on-connected-n30).

## sella-phenol-angle-h0: 0.10 Ha phenol C–O–H on dimers with a carbonyl oxygen
Status: incorporated

**Hypothesis:** Dimer leftover ketones–phenol (46 vs 43) is limited by a stiff Fischer phenol C–O–H. Soft 0.10 Ha on 1–2 such angles on dimers that also have a 1-coordinate carbonyl oxygen cheapens that class without tagging phenol–water.

**Outcome and uncertainty:** Cycle 334 **keep**. Train Δ=+2.57e-4, valid Δ=+1.76e-4, both energy-safe. Leftover ketones–phenol untagged (acetaldehyde; ketone O is H-bonded / 2-coord). Movers are amide/acid dimers with a spare 1-coord carbonyl plus intermolecular C–O–H. New champion `2697c67f7818bfcd2955251e89152f51920c6aa0`. Cycle 335 H-bonded ketone discard, Δ=−7.65e-5 (phenol–phenol extra). Cycle 336 alkyl-alpha ≥2 H discard, Δ=+8.44e-5; leftover still 46. Cycle 337 cap 1–3 bit-identical to 336. Cycle 338 H-bonded alkyl aldehyde **invalid**: ammoniums–ketones 29→15 / +1.19 kcal (R=0.682); leftover still 46.

**Reason to revisit:** Incorporated 1-coord carbonyl gate. Do not add H-bonded aldehyde/ketone carbonyl detectors (338 hop; 335 extras; leftover is not this limiter).

**Next experiment:** Close leftover ketones–phenol phenol-H0. Cycle 340: skip connected GDIIS at n_atoms=18 only.

**Attempts:**
- Cycle 339; candidate `dcd6b7c129801c3c92ade8f5bbc65ac2ee2be08d`; champion `2697c67f7818bfcd2955251e89152f51920c6aa0`; decision discard.
  [Implementation](ideas/sella-phenol-angle-h0/dcd6b7c129801c3c92ade8f5bbc65ac2ee2be08d/algo.py).
  [Training evidence](evaluation_results/cycle-339-train.json).
  [Narrative](full_log.md#cycle-339--truephenol-coh-h0-when-an-alkyl-aldehydeketone-carbon-is-present).
- Cycle 338; candidate `c29c934ff22d5dfe4ac28d91f685b3a479619cac`; champion `2697c67f7818bfcd2955251e89152f51920c6aa0`; decision invalid.
  [Implementation](ideas/sella-phenol-angle-h0/c29c934ff22d5dfe4ac28d91f685b3a479619cac/algo.py).
  [Training evidence](evaluation_results/cycle-338-train.json).
  [Narrative](full_log.md#cycle-338--hbonded-alkyl-aldehyde-carbonyls-for-dimer-phenol-coh-h0).
- Cycle 337; candidate `18c46a63450bf110542b3e192309e7d665eeb1a4`; champion `2697c67f7818bfcd2955251e89152f51920c6aa0`; decision discard.
  [Implementation](ideas/sella-phenol-angle-h0/18c46a63450bf110542b3e192309e7d665eeb1a4/algo.py).
  [Training evidence](evaluation_results/cycle-337-train.json).
  [Narrative](full_log.md#cycle-337--alkyl-hbonded-ketone-gate-with-phenol-coh-cap-13).
- Cycle 336; candidate `25b06be49f1471a6f7340738ac75346210a5cbba`; champion `2697c67f7818bfcd2955251e89152f51920c6aa0`; decision discard.
  [Implementation](ideas/sella-phenol-angle-h0/25b06be49f1471a6f7340738ac75346210a5cbba/algo.py).
  [Training evidence](evaluation_results/cycle-336-train.json).
  [Narrative](full_log.md#cycle-336--hbonded-alkyl-ketone-carbonyls-alpha-c-has-2-h-for-dimer-phenol-coh-h0).
- Cycle 335; candidate `d69749753a7ff06af2a9ffde8cd4dcffc31d671c`; champion `2697c67f7818bfcd2955251e89152f51920c6aa0`; decision discard.
  [Implementation](ideas/sella-phenol-angle-h0/d69749753a7ff06af2a9ffde8cd4dcffc31d671c/algo.py).
  [Training evidence](evaluation_results/cycle-335-train.json).
  [Narrative](full_log.md#cycle-335--also-tag-hbonded-ketone-carbonyls-for-dimer-phenol-coh-h0).
- Cycle 334; candidate `2697c67f7818bfcd2955251e89152f51920c6aa0`; champion `362a4dd6db5ca11eea73f992d86141a50908bd75`; decision keep.
  [Training evidence](evaluation_results/cycle-334-train.json).
  [Validation evidence](evaluation_results/cycle-334-valid.json).
  [Narrative](full_log.md#cycle-334--010-ha-phenol-coh-hessian-on-dimers-with-a-carbonyl-oxygen).

## sella-carbamate-angle-h0: 0.10 Ha carbamate O–C–N on connected 30≤n_atoms<80
Status: closed

**Hypothesis:** Leftover 363892164 oxazolidinone unused bend is carbonyl O–C–N at 3-coordinate C with exactly two O and one N. Ester C–O–C and 3-coord amide C–N–C missed it. Soft 0.10 Ha on 1–2 such angles, skipping N bonded to S, on connected 30≤n<80.

**Outcome and uncertainty:** Cycle 328 discard, Δ=−8.20e-5. Energy-safe. Leftover unchanged 43. Only rivaroxaban 26→27 extra. The class fires on oxazolidinones but is not the leftover limiter.

**Reason to revisit:** Not as a carbamate O–C–N H0 scale. Leftover did not move; rivaroxaban extras.

**Next experiment:** Do not interpolate 0.13 or drop the 1–2 cap. Close this family.

**Attempts:**
- Cycle 328; candidate `8f02b880f0d7822ad0e9b67155fe770f957a147e`; champion `362a4dd6db5ca11eea73f992d86141a50908bd75`; decision discard.
  [Implementation](ideas/sella-carbamate-angle-h0/8f02b880f0d7822ad0e9b67155fe770f957a147e/algo.py).
  [Training evidence](evaluation_results/cycle-328-train.json).
  [Narrative](full_log.md#cycle-328--010-ha-carbamate-ocn-hessian-on-connected-30n_atoms80).

## sella-oxazolidinone-ocn-h0: 0.10 Ha 4-coordinate C–C–N / O–C–C on connected 30≤n_atoms<80
Status: incorporated

**Hypothesis:** Soft 0.10 Ha on 1–2 4-coordinate C–C–N (vertex C with one N, two C, one H) cheapens unused ring/backbone bends. An oxygen-bonded carbon neighbor keeps oxazolidinone/amino-acid Cα and drops kinase amines.

**Outcome and uncertainty:** Cycle 343–351 C–C–N: 350/355 carboxyl train keep-sized, valid no-op. Cycle 354 **keep**: cap all O–C–C, apply 0.10 only to ether members. Cycle 362 **keep**: carboxyl C–C–N plus siloxane C–O–Si skip plus N-benzylic fused-aryl 5-ring ether skip. Train Δ=+2.85e-4 (384221749 64→58, 135132770 22→21), valid Δ=+1.049e-4 (406793986 43→41; THF and 135169428 kept). New champion `7140449e9cbddb6f6589ec34fdac86f88dc413e1`. Cycle 359 fused-all valid-negative (135169428 relative-step weight); 360/361 ring-N no-ops (N is a vertex substituent, not in-ring).

**Reason to revisit:** Incorporated N-benzylic fused-aryl skip, siloxane skip, and carboxyl C–C–N on ether O–C–C. Do not drop the alcohol-inclusive cap (353 wander). Do not skip all 4-/5-rings (THF). Do not use ring-N gates (360/361).

**Next experiment:** Leftover 363892164 still 43. Oxazolidinone and isocyanide ipso angle H0 fire without changing n_steps (cycles 366–367). Methoxy O–C–C skip is train-invisible (cycle 368). Remaining valid extra 104073139. Do not interpolate leftover angle H0. Do not interpolate N-benzylic fused skip.

**Attempts:**
- Cycle 368; candidate `43ba11b3527c814ae99c717336dbdd4cd376a4d2`; champion `7140449e9cbddb6f6589ec34fdac86f88dc413e1`; decision discard.
  [Training evidence](evaluation_results/cycle-368-train.json).
  [Narrative](full_log.md#cycle-368--skip-methoxy-och3-in-ether-occ-010-ha).
- Cycle 367; candidate `c6ccbcc55e1f8029c8bfc70e3f4fc7f0fbead996`; champion `7140449e9cbddb6f6589ec34fdac86f88dc413e1`; decision discard.
  [Implementation](ideas/sella-isocyanide-ccn-h0/c6ccbcc55e1f8029c8bfc70e3f4fc7f0fbead996/algo.py).
  [Training evidence](evaluation_results/cycle-367-train.json).
  [Narrative](full_log.md#cycle-367--010-ha-arylisocyanide-ipso-ccc-hessian).
- Cycle 366; candidate `8fbf67e1cc6e6722fa30756a44deec81b96d2333`; champion `7140449e9cbddb6f6589ec34fdac86f88dc413e1`; decision discard.
  [Implementation](ideas/sella-oxazolidinone-ocn-h0/8fbf67e1cc6e6722fa30756a44deec81b96d2333/algo.py).
  [Training evidence](evaluation_results/cycle-366-train.json).
  [Narrative](full_log.md#cycle-366--010-ha-oxazolidinone-c4-ccc-hessian).
- Cycle 362; candidate `7140449e9cbddb6f6589ec34fdac86f88dc413e1`; champion `64af5cd30f3c66bfcf1612fc273d11b94f48e928`; decision keep.
  [Training evidence](evaluation_results/cycle-362-train.json).
  [Validation evidence](evaluation_results/cycle-362-valid.json).
  [Narrative](full_log.md#cycle-362--skip-n-benzylic-fused-aryl-small-ring-ethers).
- Cycle 361; candidate `6640851a98f0e98609b7377521205a60f5725f24`; champion `64af5cd30f3c66bfcf1612fc273d11b94f48e928`; decision non_generalizable.
  [Implementation](ideas/sella-oxazolidinone-ocn-h0/6640851a98f0e98609b7377521205a60f5725f24/algo.py).
  [Training evidence](evaluation_results/cycle-361-train.json).
  [Validation evidence](evaluation_results/cycle-361-valid.json).
  [Narrative](full_log.md#cycle-361--skip-if-any-fused-small-ring-through-the-ether-is-n-substituted).
- Cycle 360; candidate `e557c76b399940d43ab6db7765a4b351faa65e5d`; champion `64af5cd30f3c66bfcf1612fc273d11b94f48e928`; decision non_generalizable.
  [Implementation](ideas/sella-oxazolidinone-ocn-h0/e557c76b399940d43ab6db7765a4b351faa65e5d/algo.py).
  [Training evidence](evaluation_results/cycle-360-train.json).
  [Validation evidence](evaluation_results/cycle-360-valid.json).
  [Narrative](full_log.md#cycle-360--skip-only-n-substituted-fused-aryl-small-ring-ethers).
- Cycle 359; candidate `b014c6468c36151160a4f893f276d42542850a01`; champion `64af5cd30f3c66bfcf1612fc273d11b94f48e928`; decision non_generalizable.
  [Implementation](ideas/sella-oxazolidinone-ocn-h0/b014c6468c36151160a4f893f276d42542850a01/algo.py).
  [Training evidence](evaluation_results/cycle-359-train.json).
  [Validation evidence](evaluation_results/cycle-359-valid.json).
  [Narrative](full_log.md#cycle-359--carboxyl-ccn-plus-siloxane-and-fused-aryl-ether-skips).
- Cycle 358; candidate `9bc432ee0f65aed35f8a4b754d4c7f5fe17e2b1e`; champion `64af5cd30f3c66bfcf1612fc273d11b94f48e928`; decision discard.
  [Implementation](ideas/sella-oxazolidinone-ocn-h0/9bc432ee0f65aed35f8a4b754d4c7f5fe17e2b1e/algo.py).
  [Training evidence](evaluation_results/cycle-358-train.json).
  [Narrative](full_log.md#cycle-358--skip-siloxane-and-fused-aryl-small-ring-ethers-in-occ-010-ha).
- Cycle 354; candidate `64af5cd30f3c66bfcf1612fc273d11b94f48e928`; champion `2697c67f7818bfcd2955251e89152f51920c6aa0`; decision keep.
  [Training evidence](evaluation_results/cycle-354-train.json).
  [Validation evidence](evaluation_results/cycle-354-valid.json).
  [Narrative](full_log.md#cycle-354--cap-all-4-coord-occ-apply-010-ha-only-to-ether-members).
- Cycle 355; candidate `b63cad557f4db81770975ddcde0503685d4e8c5e`; champion `64af5cd30f3c66bfcf1612fc273d11b94f48e928`; decision non_generalizable.
  [Implementation](ideas/sella-oxazolidinone-ocn-h0/b63cad557f4db81770975ddcde0503685d4e8c5e/algo.py).
  [Training evidence](evaluation_results/cycle-355-train.json).
  [Validation evidence](evaluation_results/cycle-355-valid.json).
  [Narrative](full_log.md#cycle-355--add-carboxyl-ccn-010-ha-on-the-cycle-354-ether-occ-champion).
- Cycle 356; candidate `c5985d8f6a49fa879ca07104ae6104bb8d45e8ed`; champion `64af5cd30f3c66bfcf1612fc273d11b94f48e928`; decision discard.
  [Training evidence](evaluation_results/cycle-356-train.json).
  [Narrative](full_log.md#cycle-356--skip-45-membered-cyclic-ethers-in-occ-010-ha).
- Cycle 357; candidate `e87bf5242b2ab713edc80d6c6a5cf7662a3bae6d`; champion `64af5cd30f3c66bfcf1612fc273d11b94f48e928`; decision non_generalizable.
  [Implementation](ideas/sella-oxazolidinone-ocn-h0/e87bf5242b2ab713edc80d6c6a5cf7662a3bae6d/algo.py).
  [Training evidence](evaluation_results/cycle-357-train.json).
  [Validation evidence](evaluation_results/cycle-357-valid.json).
  [Narrative](full_log.md#cycle-357--carboxyl-ccn-plus-skip-45-membered-cyclic-ethers).
- Cycle 350; candidate `ce33f1f428a3c024da878ff688aa68dfa9c3cfe5`; champion `2697c67f7818bfcd2955251e89152f51920c6aa0`; decision non_generalizable.
  [Implementation](ideas/sella-oxazolidinone-ocn-h0/ce33f1f428a3c024da878ff688aa68dfa9c3cfe5/algo.py).
  [Training evidence](evaluation_results/cycle-350-train.json).
  [Validation evidence](evaluation_results/cycle-350-valid.json).
  [Narrative](full_log.md#cycle-350--4-coordinate-ccn-010-ha-only-next-to-a-carboxylester-carbon).
- Cycle 351; candidate `fbce9c6c57b314511378b8dd4fdda663ea1d5085`; champion `2697c67f7818bfcd2955251e89152f51920c6aa0`; decision non_generalizable.
  [Implementation](ideas/sella-oxazolidinone-ocn-h0/fbce9c6c57b314511378b8dd4fdda663ea1d5085/algo.py).
  [Training evidence](evaluation_results/cycle-351-train.json).
  [Validation evidence](evaluation_results/cycle-351-valid.json).
  [Narrative](full_log.md#cycle-351--4-coordinate-ccn-010-ha-next-to-any-3-coord-carbonyl-carbon).
- Cycle 352; candidate `768618e2aaddd538725c41f371d0ac21d4c72138`; champion `2697c67f7818bfcd2955251e89152f51920c6aa0`; decision non_generalizable.
  [Implementation](ideas/sella-oxazolidinone-ocn-h0/768618e2aaddd538725c41f371d0ac21d4c72138/algo.py).
  [Training evidence](evaluation_results/cycle-352-train.json).
  [Validation evidence](evaluation_results/cycle-352-valid.json).
  [Narrative](full_log.md#cycle-352--010-ha-oxazolidinone-c5-occ-hessian-on-connected-30n_atoms80).
- Cycle 353; candidate `0b8a971670d26c616a0cc6371e911b7995cab9b8`; champion `2697c67f7818bfcd2955251e89152f51920c6aa0`; decision non_generalizable.
  [Implementation](ideas/sella-oxazolidinone-ocn-h0/0b8a971670d26c616a0cc6371e911b7995cab9b8/algo.py).
  [Training evidence](evaluation_results/cycle-353-train.json).
  [Validation evidence](evaluation_results/cycle-353-valid.json).
  [Narrative](full_log.md#cycle-353--4-coord-occ-010-ha-only-at-ether-oxygen-two-heavy-neighbors).
- Cycle 349; candidate `bcef60a2c68cb2bcd6c9f53fb8f2eb31da044b2e`; champion `2697c67f7818bfcd2955251e89152f51920c6aa0`; decision discard.
  [Training evidence](evaluation_results/cycle-349-train.json).
  [Narrative](full_log.md#cycle-349--cycle-346-ccn-010-ha-only-when-exactly-one-such-angle-exists).
- Cycle 348; candidate `39598e7e0260fb32bb244b4a66e60d0761296996`; champion `2697c67f7818bfcd2955251e89152f51920c6aa0`; decision discard.
  [Implementation](ideas/sella-oxazolidinone-ocn-h0/39598e7e0260fb32bb244b4a66e60d0761296996/algo.py).
  [Training evidence](evaluation_results/cycle-348-train.json).
  [Narrative](full_log.md#cycle-348--cycle-346-ccn-010-ha-only-at-3-coordinate-nitrogen).
- Cycle 347; candidate `67b08f12f8cc9e3781ded1af8e099e617bccdfb4`; champion `2697c67f7818bfcd2955251e89152f51920c6aa0`; decision discard.
  [Implementation](ideas/sella-oxazolidinone-ocn-h0/67b08f12f8cc9e3781ded1af8e099e617bccdfb4/algo.py).
  [Training evidence](evaluation_results/cycle-347-train.json).
  [Narrative](full_log.md#cycle-347--4-coordinate-ccn-010-ha-only-if-a-carbon-neighbor-is-bonded-to-o-and-n).
- Cycle 346; candidate `12cab593f1780feb3b3cccdce29ca12cd3fd05f5`; champion `2697c67f7818bfcd2955251e89152f51920c6aa0`; decision discard.
  [Implementation](ideas/sella-oxazolidinone-ocn-h0/12cab593f1780feb3b3cccdce29ca12cd3fd05f5/algo.py).
  [Training evidence](evaluation_results/cycle-346-train.json).
  [Narrative](full_log.md#cycle-346--4-coordinate-ccn-010-ha-only-if-a-carbon-neighbor-is-oxygen-bonded).
- Cycle 345; candidate `8147f0e4c6ef7687db796f557b05174ea09a9b2f`; champion `2697c67f7818bfcd2955251e89152f51920c6aa0`; decision discard.
  [Implementation](ideas/sella-oxazolidinone-ocn-h0/8147f0e4c6ef7687db796f557b05174ea09a9b2f/algo.py).
  [Training evidence](evaluation_results/cycle-345-train.json).
  [Narrative](full_log.md#cycle-345--apply-4-coordinate-oxazolidinone-ccn-010-ha-in-guess_hessian).
- Cycle 344; candidate `044ba7afc4014900ddb5030349ac440d4a49ca3c`; champion `2697c67f7818bfcd2955251e89152f51920c6aa0`; decision discard.
  [Implementation](ideas/sella-oxazolidinone-ocn-h0/044ba7afc4014900ddb5030349ac440d4a49ca3c/algo.py).
  [Training evidence](evaluation_results/cycle-344-train.json).
  [Narrative](full_log.md#cycle-344--010-ha-oxazolidinone-ring-ccn-hessian-on-connected-30n_atoms80).
- Cycle 343; candidate `168517fbebf730ae0be91e85e07e0c5bbc7b1e54`; champion `2697c67f7818bfcd2955251e89152f51920c6aa0`; decision discard.
  [Implementation](ideas/sella-oxazolidinone-ocn-h0/168517fbebf730ae0be91e85e07e0c5bbc7b1e54/algo.py).
  [Training evidence](evaluation_results/cycle-343-train.json).
  [Narrative](full_log.md#cycle-343--010-ha-oxazolidinone-ring-ocn-hessian-on-connected-30n_atoms80).

## sella-thioether-scc-h0: 0.10 Ha 4-coordinate thioether S–C–C / C–S–C on connected 30≤n_atoms<80
Status: closed

**Hypothesis:** Ether O–C–C 0.10 kept; the sulfur analog is 4-coordinate S–C–C or 2-coordinate C–S–C. GAFF `c3-c3-ss` ≈ 0.123 Ha and `c3-ss-c3` ≈ 0.060 Ha, both below Fischer ~0.16.

**Outcome and uncertainty:** Cycles 363–365 discard, Δ=0. S–C–C and C–S–C (0.10 and 0.06) all fire on 384219853 with energy twitches and identical n_steps. Thioether angle H0 is not a limiter.

**Reason to revisit:** Not as an H0 scale or carbon-vs-sulfur center split.

**Next experiment:** Do not interpolate 0.04. Close this family.

**Attempts:**
- Cycle 365; candidate `e2c2d1521e54de7af7adcab6ac3c4658e50eb124`; champion `7140449e9cbddb6f6589ec34fdac86f88dc413e1`; decision discard.
  [Implementation](ideas/sella-thioether-scc-h0/e2c2d1521e54de7af7adcab6ac3c4658e50eb124/algo.py).
  [Training evidence](evaluation_results/cycle-365-train.json).
  [Narrative](full_log.md#cycle-365--006-ha-thioether-csc-hessian-matching-gaff-c3-ss-c3).
- Cycle 364; candidate `673e03bf53bcffcb36eb6c198fb8c69d73f75d97`; champion `7140449e9cbddb6f6589ec34fdac86f88dc413e1`; decision discard.
  [Implementation](ideas/sella-thioether-scc-h0/673e03bf53bcffcb36eb6c198fb8c69d73f75d97/algo.py).
  [Training evidence](evaluation_results/cycle-364-train.json).
  [Narrative](full_log.md#cycle-364--010-ha-thioether-csc-hessian-at-2-coordinate-sulfur).
- Cycle 363; candidate `150b8c06449d50f4d94d1035a93ccdc5df64e581`; champion `7140449e9cbddb6f6589ec34fdac86f88dc413e1`; decision discard.
  [Implementation](ideas/sella-thioether-scc-h0/150b8c06449d50f4d94d1035a93ccdc5df64e581/algo.py).
  [Training evidence](evaluation_results/cycle-363-train.json).
  [Narrative](full_log.md#cycle-363--010-ha-4-coordinate-thioether-scc-hessian-on-connected-30n_atoms80).

## sella-alkyne-aryl-angle-h0: 0.10 Ha aryl–alkyne C–C–C / C–C–N on connected 30≤n_atoms<80
Status: closed

**Hypothesis:** Leftover 363892164 dummy-linear extra is the aryl–alkyne joint, not CF2 or oxazolidinone. Soft 0.10 Ha on 1–2 C–C–C at a 3-coordinate carbon adjacent to a 2-coordinate C–C alkyne.

**Outcome and uncertainty:** Cycle 330 discard, Δ=0. n_steps bit-identical; leftover/104046121 energy twitches. C–C–C fires but is not the limiter. Cycle 331 C–C–N discard, Δ=0. Leftover untagged; 104046121/acalabrutinib energy twitches only. Cycle 332 4-coordinate alkyl–alkyne C–C–C discard, Δ=−6.09e-5. Leftover unchanged 43; 104046121 33→34 extra.

**Reason to revisit:** Not as an alkyne C–C–C/C–C–N H0 scale. 3-coord no-op; 4-coord extras the canary; leftover unchanged.

**Next experiment:** Do not interpolate 0.13. Close this family. Skip GDIIS on connected 18≤n<30 (hexanitroso leftover; dummy-angle gap).

**Attempts:**
- Cycle 332; candidate `220c8cf9caf1b69bd58b2a639733f553883b78c7`; champion `362a4dd6db5ca11eea73f992d86141a50908bd75`; decision discard.
  [Implementation](ideas/sella-alkyne-aryl-angle-h0/220c8cf9caf1b69bd58b2a639733f553883b78c7/algo.py).
  [Training evidence](evaluation_results/cycle-332-train.json).
  [Narrative](full_log.md#cycle-332--010-ha-alkylalkyne-ccc-hessian-on-connected-30n_atoms80).
- Cycle 331; candidate `427675582fda49e5134f2430d96206905cbd1610`; champion `362a4dd6db5ca11eea73f992d86141a50908bd75`; decision discard.
  [Implementation](ideas/sella-alkyne-aryl-angle-h0/427675582fda49e5134f2430d96206905cbd1610/algo.py).
  [Training evidence](evaluation_results/cycle-331-train.json).
  [Narrative](full_log.md#cycle-331--010-ha-arylalkyne-ccn-hessian-on-connected-30n_atoms80).
- Cycle 330; candidate `46c2259d53b764fbee4b758f5da64a2b7cf6bb9d`; champion `362a4dd6db5ca11eea73f992d86141a50908bd75`; decision discard.
  [Implementation](ideas/sella-alkyne-aryl-angle-h0/46c2259d53b764fbee4b758f5da64a2b7cf6bb9d/algo.py).
  [Training evidence](evaluation_results/cycle-330-train.json).
  [Narrative](full_log.md#cycle-330--010-ha-arylalkyne-ccc-hessian-on-connected-30n_atoms80).

## sella-azo-cnn-h0: 0.10 Ha 2-coordinate C–N–N on connected 30≤n_atoms<80
Status: incorporated

**Hypothesis:** Unused 2-coordinate C–N–N (imine N bonded to C and N) on 30≤n<80 is a different class from 3-coordinate pyrazole NH (cycle 371) and n<12 azide (cycle 301). Soft 0.10 Ha on 1–2 such angles cheapens those frames. Cycle 376 restricts to N–N both 2-coordinate in a C/N-only 5-ring.

**Outcome and uncertainty:** Cycle 375 discard, Δ=+8.12e-5 (acyclic azo / 3-coord N extras). Cycle 376 **keep**. Train Δ=+3.40e-4, valid Δ=+4.50e-4, both energy-safe. 315658749 32→27 and valid 381615095 35→28. Extra 252653184/440971005 +1 C2N3. New champion `7fd78d3b832786094a27ddb6422f54dc8d909fa7`.

**Reason to revisit:** Not by dropping C2N3 (valid save) or reopening unconstrained 2-coord C–N–N (375 extras).

**Next experiment:** Leave the 5-ring both-2-coord gate in the champion. Next class is not this family.

**Attempts:**
- Cycle 376; candidate `7fd78d3b832786094a27ddb6422f54dc8d909fa7`; champion `7140449e9cbddb6f6589ec34fdac86f88dc413e1`; decision keep.
  [Training evidence](evaluation_results/cycle-376-train.json).
  [Validation evidence](evaluation_results/cycle-376-valid.json).
  [Narrative](full_log.md#cycle-376--010-ha-5-ring-2-coord-cnn-hessian-on-30n80).
- Cycle 375; candidate `a5879b3e61b83a3be63a68d3af39b7e2c0318419`; champion `7140449e9cbddb6f6589ec34fdac86f88dc413e1`; decision discard.
  [Implementation](ideas/sella-azo-cnn-h0/a5879b3e61b83a3be63a68d3af39b7e2c0318419/algo.py).
  [Training evidence](evaluation_results/cycle-375-train.json).
  [Narrative](full_log.md#cycle-375--010-ha-2-coordinate-cnn-hessian-on-30n80).

## sella-nitro-ono-h0: 0.10 Ha nitro O–N–O on connected 30≤n_atoms<80
Status: closed

**Hypothesis:** Unused nitro O–N–O is stiffer than GAFF `o-no-o` (~0.123 Ha). Soft 0.10 Ha on 1–2 such angles, skipping 5-ring C–N–N keeps and CF3, cheapens nitro frames.

**Outcome and uncertainty:** Cycle 377 discard, Δ=−3.50e-4. Energy-safe. Only 135227471 19→21 and 384221749 58→62 extra. Soft O–N–O extras nitro.

**Reason to revisit:** Not as an O–N–O H0 scale. Do not interpolate 0.13.

**Next experiment:** Close this family.

**Attempts:**
- Cycle 377; candidate `5c8d3c00e5e44a01ae0837c5be2f7b74f54bb837`; champion `7fd78d3b832786094a27ddb6422f54dc8d909fa7`; decision discard.
  [Implementation](ideas/sella-nitro-ono-h0/5c8d3c00e5e44a01ae0837c5be2f7b74f54bb837/algo.py).
  [Training evidence](evaluation_results/cycle-377-train.json).
  [Narrative](full_log.md#cycle-377--010-ha-nitro-ono-hessian-on-30n80).

## sella-oxazole-cno-h0: 0.10 Ha 5-ring 2-coordinate C–N–O on connected 30≤n_atoms<80
Status: closed

**Hypothesis:** Unused 2-coordinate C–N–O in a C/N/O 5-ring is the oxygen analog of the cycle-376 C–N–N keep. Skipping 5-ring C–N–N molecules spares 381615095.

**Outcome and uncertainty:** Cycle 378 discard, Δ=−1.25e-4. Energy-safe. Only 125086620 17→18 extra. Soft 5-ring C–N–O extras the oxadiazole.

**Reason to revisit:** Not as a C–N–O H0 scale on this ring class.

**Next experiment:** Close this family.

**Attempts:**
- Cycle 378; candidate `8e1f7d182589d3b96f10b85637e8cafec9db48fa`; champion `7fd78d3b832786094a27ddb6422f54dc8d909fa7`; decision discard.
  [Implementation](ideas/sella-oxazole-cno-h0/8e1f7d182589d3b96f10b85637e8cafec9db48fa/algo.py).
  [Training evidence](evaluation_results/cycle-378-train.json).
  [Narrative](full_log.md#cycle-378--010-ha-5-ring-2-coord-cno-hessian-on-30n80).

## sella-sulfone-csc-h0: 0.10 Ha sulfone C–S–C on connected 30≤n_atoms<80
Status: closed

**Hypothesis:** Unused sulfone C–S–C at 4-coordinate S (two O, two C) is stiffer than needed. Soft 0.10 Ha on 1–2 such angles, skipping 5-ring C–N–N keeps, cheapens sulfone leftovers.

**Outcome and uncertainty:** Cycle 379 discard, Δ=0. n_steps bit-identical; energy twitches on 125086620 and 135224095. The class fires and is not a limiter.

**Reason to revisit:** Not as a C–S–C H0 scale at sulfone.

**Next experiment:** Do not interpolate 0.06 or O–S–O on 30–80. Try sulfonamide C–S–N.

**Attempts:**
- Cycle 379; candidate `fe110c0e51987e24227350991ef1a7a04e9f9f04`; champion `7fd78d3b832786094a27ddb6422f54dc8d909fa7`; decision discard.
  [Implementation](ideas/sella-sulfone-csc-h0/fe110c0e51987e24227350991ef1a7a04e9f9f04/algo.py).
  [Training evidence](evaluation_results/cycle-379-train.json).
  [Narrative](full_log.md#cycle-379--010-ha-sulfone-csc-hessian-on-30n80).

## sella-sulfonamide-csn-h0: 0.10 Ha sulfonamide C–S–N on connected 30≤n_atoms<80
Status: incorporated

**Hypothesis:** Unused sulfonamide C–S–N at 4-coordinate S {C, N, O, O} cheapens 313071992 and valid 252648738. Cycle 380 extras secondary sulfonamide NH and N bonded to two S.

**Outcome and uncertainty:** Cycle 380/381 non_generalizable (valid +6.14e-5). Cycle 432 **keep** stacked with geodesic 18–20: train Δ=+7.39e-4, valid Δ=+1.04e-4 (252648738 36→35 plus amides–imidazolium). New champion `544a2a450f4a9385c365598b8b1148144d4fb1d4`.

**Reason to revisit:** Incorporated. Do not interpolate 0.08 or combine O–S–N.

**Next experiment:** Do not drop the N gate. See leftover 135169446 alcohol C–C–O on n<18.

**Attempts:**
- Cycle 432; candidate `544a2a450f4a9385c365598b8b1148144d4fb1d4`; champion `d2f7137d0609977f7ebd973704e68b8f419f8dad`; decision keep.
  [Training evidence](evaluation_results/cycle-432-train.json).
  [Validation evidence](evaluation_results/cycle-432-valid.json).
  [Narrative](full_log.md#cycle-432--exact-geodesic-18-20-plus-sulfonamide-csn-010-ha).
- Cycle 382; candidate `ae3e5b87d59a7eef07b052ed9dbc78ceaea2978a`; champion `7fd78d3b832786094a27ddb6422f54dc8d909fa7`; decision discard.
  [Implementation](ideas/sella-sulfonamide-csn-h0/ae3e5b87d59a7eef07b052ed9dbc78ceaea2978a/algo.py).
  [Training evidence](evaluation_results/cycle-382-train.json).
  [Narrative](full_log.md#cycle-382--010-ha-tertiary2-coord-sulfonamide-osn-hessian).
- Cycle 381; candidate `ce2e8934edd7558195f18cc0fc5c3bcf2aeb069f`; champion `7fd78d3b832786094a27ddb6422f54dc8d909fa7`; decision non_generalizable.
  [Implementation](ideas/sella-sulfonamide-csn-h0/ce2e8934edd7558195f18cc0fc5c3bcf2aeb069f/algo.py).
  [Training evidence](evaluation_results/cycle-381-train.json).
  [Validation evidence](evaluation_results/cycle-381-valid.json).
  [Narrative](full_log.md#cycle-381--tertiary2-coord-sulfonamide-csn-hessian-010-ha).
- Cycle 380; candidate `8efdc1bfead5619c661aa8ef987d57f5376525b7`; champion `7fd78d3b832786094a27ddb6422f54dc8d909fa7`; decision non_generalizable.
  [Implementation](ideas/sella-sulfonamide-csn-h0/8efdc1bfead5619c661aa8ef987d57f5376525b7/algo.py).
  [Training evidence](evaluation_results/cycle-380-train.json).
  [Validation evidence](evaluation_results/cycle-380-valid.json).
  [Narrative](full_log.md#cycle-380--010-ha-sulfonamide-csn-hessian-on-30n80).

## sella-pyrrole-cnc-h0: 0.10 Ha 3-coordinate pyrrole/imidazole C–N–C on connected 30≤n_atoms<80
Status: closed

**Hypothesis:** Unused 3-coordinate pyrrole/imidazole NH C–N–C is the analog of kept pyridine 2-coord C–N–C. Soft 0.10 Ha on 1–2 such angles in a C/N 5-ring, skipping O/S carbons and 2-coord-N carbons, cheapens 135251716 and 313071992.

**Outcome and uncertainty:** Cycle 383 non_generalizable. Train Δ=+1.64e-4 (only 313071992 13→12; same molecule as cycle 381 C–S–N). Valid Δ=0; four hits energy-twitch only. Energy-safe. Valid pyrrole C–N–C is not a limiter.

**Reason to revisit:** Not as a pyrrole C–N–C H0 scale. Do not drop the 2-coord-N skip or combine with C–S–N (same train save, zero valid n_steps).

**Next experiment:** Close this family. Cycle 381 C–S–N remains the stacked near-miss if an independent valid save appears.

**Attempts:**
- Cycle 383; candidate `060e09d1a794164417d59a4e3fc575e5e33300cc`; champion `7fd78d3b832786094a27ddb6422f54dc8d909fa7`; decision non_generalizable.
  [Implementation](ideas/sella-pyrrole-cnc-h0/060e09d1a794164417d59a4e3fc575e5e33300cc/algo.py).
  [Training evidence](evaluation_results/cycle-383-train.json).
  [Validation evidence](evaluation_results/cycle-383-valid.json).
  [Narrative](full_log.md#cycle-383--010-ha-pyrroleimidazole-cnc-hessian-on-30n80).

## sella-poc-h0: 0.10 Ha phosphate-ester P–O–C on connected 30≤n_atoms<80
Status: closed

**Hypothesis:** n<12 P–O–H 0.10 does not apply on 30–80. Soft 1–2 P–O–C, skipping P–O–P and 5-ring C–N–N, cheapens phosphate esters.

**Outcome and uncertainty:** Cycle 384 discard, Δ=−1.94e-4. Energy-safe. Extra 135316214 33→36; 135130097 twitch only. Soft P–O–C extras the thiophosphate ester; the remaining hit is not a limiter.

**Reason to revisit:** Not as a P–O–C H0 scale on 30–80. Do not skip-S (would leave a no-op) or interpolate 0.13.

**Next experiment:** Close this family.

**Attempts:**
- Cycle 384; candidate `e01619e65c351cdd7624de221a1777dee64cfc78`; champion `7fd78d3b832786094a27ddb6422f54dc8d909fa7`; decision discard.
  [Implementation](ideas/sella-poc-h0/e01619e65c351cdd7624de221a1777dee64cfc78/algo.py).
  [Training evidence](evaluation_results/cycle-384-train.json).
  [Narrative](full_log.md#cycle-384--010-ha-phosphate-ester-poc-hessian-on-30n80).

## sella-imine-cnh-h0: 0.10 Ha 2-coordinate imine C–N–H on connected 30≤n_atoms<80
Status: closed

**Hypothesis:** Unused imine C–N–H is softer than Fischer. Soft 0.10 Ha on 1–2 such angles, skipping 2-coord C and guanidinium, cheapens imines.

**Outcome and uncertainty:** Cycle 385 discard, Δ=−1.67e-4. Energy-safe. Extras 104161681 25→26 and 336270104 26→27.

**Reason to revisit:** Not as an imine C–N–H H0 scale. Do not interpolate 0.13.

**Next experiment:** Close this family.

**Attempts:**
- Cycle 385; candidate `8feb0c6be9a6bcb9059f77717aef6f05fc4a0477`; champion `7fd78d3b832786094a27ddb6422f54dc8d909fa7`; decision discard.
  [Implementation](ideas/sella-imine-cnh-h0/8feb0c6be9a6bcb9059f77717aef6f05fc4a0477/algo.py).
  [Training evidence](evaluation_results/cycle-385-train.json).
  [Narrative](full_log.md#cycle-385--010-ha-imine-cnh-hessian-on-30n80).

## sella-urea-ncn-h0: 0.10 Ha urea N–C–N on connected 30≤n_atoms<80
Status: closed

**Hypothesis:** Unused urea N–C–N at 3-coordinate C {O, N, N} cheapens isolated ureas. Exactly one angle, skip 2-coord C / CF3 / 5-ring C–N–N.

**Outcome and uncertainty:** Cycle 386 discard, Δ=−1.33e-4. Energy-safe. Extra 104110975 16→17; others twitch. Soft urea extras the isolated urea.

**Reason to revisit:** Not as a urea N–C–N H0 scale. Do not open the 1–2 cap.

**Next experiment:** Close this family.

**Attempts:**
- Cycle 386; candidate `67e53c9818fd4197c8b3dfa02ec97951c9d3fa68`; champion `7fd78d3b832786094a27ddb6422f54dc8d909fa7`; decision discard.
  [Implementation](ideas/sella-urea-ncn-h0/67e53c9818fd4197c8b3dfa02ec97951c9d3fa68/algo.py).
  [Training evidence](evaluation_results/cycle-386-train.json).
  [Narrative](full_log.md#cycle-386--010-ha-urea-ncn-hessian-on-30n80).

## sella-furan-coc-h0: 0.10 Ha 5-ring furan C–O–C on connected 30≤n_atoms<80
Status: closed

**Hypothesis:** Unused furan C–O–C at 2-coordinate O in a C/O 5-ring is distinct from ether O–C–C at 4-coord C. Soft 0.10 Ha, skip 2-coord C and 5-ring C–N–N.

**Outcome and uncertainty:** Cycle 387 discard, Δ=0. Energy-safe. Four energy twitches, identical n_steps. Not a limiter.

**Reason to revisit:** Not as a furan C–O–C H0 scale. Do not interpolate 0.13.

**Next experiment:** Close this family.

**Attempts:**
- Cycle 387; candidate `9bbe90dce1f20b3058b44a6bb10face2bdeee907`; champion `7fd78d3b832786094a27ddb6422f54dc8d909fa7`; decision discard.
  [Implementation](ideas/sella-furan-coc-h0/9bbe90dce1f20b3058b44a6bb10face2bdeee907/algo.py).
  [Training evidence](evaluation_results/cycle-387-train.json).
  [Narrative](full_log.md#cycle-387--010-ha-5-ring-furan-coc-hessian-on-30n80).

## sella-alkyl-coh-h0: 0.10 Ha alkyl C–O–H on connected 30≤n_atoms<80
Status: closed

**Hypothesis:** Connected alkyl C–O–H is distinct from dimer phenol C–O–H. Soft 0.10 Ha, skip 2-coord C / guanidinium / 5-ring C–N–N.

**Outcome and uncertainty:** Cycle 388 discard, Δ=−6.27e-5. Energy-safe. Extra 8030412 33→34; four twitches. Soft alkyl C–O–H extras a previous save.

**Reason to revisit:** Not as an alkyl C–O–H H0 scale. Do not interpolate 0.13.

**Next experiment:** Close this family.

**Attempts:**
- Cycle 388; candidate `6aaeb8d9112b74784192f6661403141cd9ebd8a7`; champion `7fd78d3b832786094a27ddb6422f54dc8d909fa7`; decision discard.
  [Implementation](ideas/sella-alkyl-coh-h0/6aaeb8d9112b74784192f6661403141cd9ebd8a7/algo.py).
  [Training evidence](evaluation_results/cycle-388-train.json).
  [Narrative](full_log.md#cycle-388--010-ha-alkyl-coh-hessian-on-30n80).

## sella-carboxyl-oco-h0: 0.10 Ha carboxyl O–C–O on connected 30≤n_atoms<80
Status: closed

**Hypothesis:** Leftover 363892164 has unused carboxyl O–C–O. Soft 0.10 Ha on 1–2 such angles, skipping 5-ring C–N–N and CF3, cheapens leftover.

**Outcome and uncertainty:** Cycle 389 discard, Δ=−4.96e-4. Energy-safe. Leftover 363892164 stays 43 (energy twitch). Extra 384221749 58→66; also 135267488 +3 and 125089231 +1. One −1 on 160850312. Not the leftover limiter; extras the ether keep.

**Reason to revisit:** Not as a carboxyl O–C–O H0 scale. Do not interpolate 0.13.

**Next experiment:** Close this family. Leftover oxazolidinone angles are exhausted; try carbamate C–N dihedral H0.

**Attempts:**
- Cycle 389; candidate `e86f015cd16b8f2a399045c41d1beaefcb99d623`; champion `7fd78d3b832786094a27ddb6422f54dc8d909fa7`; decision discard.
  [Implementation](ideas/sella-carboxyl-oco-h0/e86f015cd16b8f2a399045c41d1beaefcb99d623/algo.py).
  [Training evidence](evaluation_results/cycle-389-train.json).
  [Narrative](full_log.md#cycle-389--010-ha-carboxyl-oco-hessian-on-30n80).

## sella-carbamate-cn-dihedral-h0: 0.005 Ha carbamate C–N dihedral on connected 30≤n_atoms<80
Status: closed

**Hypothesis:** Leftover oxazolidinone angles are exhausted. Soft 0.005 Ha (half Fischer) on dihedrals of the unique carbamate C {O,O,N}–N bond cheapens leftover 363892164.

**Outcome and uncertainty:** Cycle 390 discard, Δ=−1.98e-4. Energy-safe. No saves. Extras leftover 43→44, rivaroxaban 26→27, 135267488 45→46 — every 30–80 carbamate hit. Soft torsion extras leftover.

**Reason to revisit:** Not as a carbamate C–N dihedral scale. Do not interpolate 0.002/0.008.

**Next experiment:** Close this family. Do not restack with O–C–N angle H0.

**Attempts:**
- Cycle 390; candidate `653c32fcf145c249b66cb0d221d6fb041df0ef14`; champion `7fd78d3b832786094a27ddb6422f54dc8d909fa7`; decision discard.
  [Implementation](ideas/sella-carbamate-cn-dihedral-h0/653c32fcf145c249b66cb0d221d6fb041df0ef14/algo.py).
  [Training evidence](evaluation_results/cycle-390-train.json).
  [Narrative](full_log.md#cycle-390--0005-ha-carbamate-cn-dihedral-hessian-on-30n80).

## sella-medium-cso-h0: 0.10 Ha C–S–O on connected 18≤n_atoms<30
Status: closed

**Hypothesis:** Leftover 433943527 (n=27) has two unused C–S–O in the dummy-angle gap. Soft 0.10 Ha on 1–2 such angles cheapens it without n<12 sulfoxide extras.

**Outcome and uncertainty:** Cycle 391 discard, Δ=−4.43e-4. Energy-safe. Leftover twitch-only. Extras 135099868 22→25 and 135161672 14→15.

**Reason to revisit:** Not as a C–S–O H0 scale on 18–29. Do not interpolate 0.13 or drop the 1–2 cap.

**Next experiment:** Close this family.

**Attempts:**
- Cycle 391; candidate `552a89774ad37004bbc36e077ac7bc4813e89c00`; champion `7fd78d3b832786094a27ddb6422f54dc8d909fa7`; decision discard.
  [Implementation](ideas/sella-medium-cso-h0/552a89774ad37004bbc36e077ac7bc4813e89c00/algo.py).
  [Training evidence](evaluation_results/cycle-391-train.json).
  [Narrative](full_log.md#cycle-391--010-ha-cso-hessian-on-connected-18n30).

## sella-medium-css-h0: 0.10 Ha disulfide C–S–S on connected 12–30
Status: incorporated

**Hypothesis:** Disulfide C–S–S 0.10 on connected 12≤n<30 cheapens alkyl and O-substituted CSS. Cycle 392 was 18–30 train-only. Cycle 393’s 12–30 extraed 1,2-dithiole C{C,S,S}. Cycle 394 requires 4-coordinate or oxygen-substituted carbon.

**Outcome and uncertainty:** Cycle 392 non_generalizable (train Δ=+1.07e-4; valid Δ=0). Cycle 393 non_generalizable (valid extra 104039830). Cycle 394 keep. Train Δ=+1.07e-4 (135181905 20→19). Valid Δ=+2.05e-4 (135061043 21→19; extra gone). Energy-safe.

**Reason to revisit:** Not as an H0 interpolation or 30–80 expansion. Stack only with an independent 30–80 save such as cycle 381 C–S–N if a second valid n_steps appears.

**Next experiment:** Leave CSS 0.10 and the carbon gate. Do not open 30–80 CSS.

**Attempts:**
- Cycle 394; candidate `8e55a3dd9c0982bebda9429fb00fea1ae261d650`; champion `7fd78d3b832786094a27ddb6422f54dc8d909fa7`; decision keep.
  [Training evidence](evaluation_results/cycle-394-train.json).
  [Validation evidence](evaluation_results/cycle-394-valid.json).
  [Narrative](full_log.md#cycle-394--alkylo-substituted-css-010-ha-on-connected-12n30).
- Cycle 393; candidate `ace80a26cd542bee05a2970bab914f011c593576`; champion `7fd78d3b832786094a27ddb6422f54dc8d909fa7`; decision non_generalizable.
  [Implementation](ideas/sella-medium-css-h0/ace80a26cd542bee05a2970bab914f011c593576/algo.py).
  [Training evidence](evaluation_results/cycle-393-train.json).
  [Validation evidence](evaluation_results/cycle-393-valid.json).
  [Narrative](full_log.md#cycle-393--extend-disulfide-css-010-ha-to-connected-12n30).
- Cycle 392; candidate `edf28d4840bbbea701168a4daed56d74a8e21491`; champion `7fd78d3b832786094a27ddb6422f54dc8d909fa7`; decision non_generalizable.
  [Implementation](ideas/sella-medium-css-h0/edf28d4840bbbea701168a4daed56d74a8e21491/algo.py).
  [Training evidence](evaluation_results/cycle-392-train.json).
  [Validation evidence](evaluation_results/cycle-392-valid.json).
  [Narrative](full_log.md#cycle-392--010-ha-disulfide-css-hessian-on-connected-18n30).

## sella-medium-oss-h0: 0.10 Ha thiosulfonate O–S–S on connected 12–30
Status: closed

**Hypothesis:** Leftover 433943527 CSS twitched; unused 4-coordinate O–S–S is the limiter.

**Outcome and uncertainty:** Cycle 395 discard, Δ=−1.07e-4 vs `8e55a3d`. Energy-safe. Only extra leftover 433943527 21→22. Soft O–S–S extras the leftover.

**Reason to revisit:** Not as an O–S–S H0 scale on 12–30. Do not interpolate 0.13.

**Next experiment:** Close this family.

**Attempts:**
- Cycle 395; candidate `e432f4c1de924e6c48ef015a95ef438392b909b5`; champion `8e55a3dd9c0982bebda9429fb00fea1ae261d650`; decision discard.
  [Implementation](ideas/sella-medium-oss-h0/e432f4c1de924e6c48ef015a95ef438392b909b5/algo.py).
  [Training evidence](evaluation_results/cycle-395-train.json).
  [Narrative](full_log.md#cycle-395--010-ha-thiosulfonate-oss-hessian-on-connected-12n30).

## sella-medium-cno-h0: 0.10 Ha 2-coordinate C–N–O on connected 12–30
Status: closed

**Hypothesis:** Leftover 135043047 (n=18) has six 2-coordinate C–N–O in the dummy-angle gap. Soft 0.10 Ha with a 1–6 cap cheapens it without cycle 378’s 30–80 extras.

**Outcome and uncertainty:** Cycle 396 invalid (energy_below_baseline). Leftover hops 43→32, +38 kcal, R=0.839. Also extras 135099623/135295734 +1.

**Reason to revisit:** Not as a C–N–O H0 scale on 12–30. The leftover well is not C–N–O-stiff; softening hops it.

**Next experiment:** Close this family. Do not fill dummy-angle at n=18.

**Attempts:**
- Cycle 396; candidate `812895020f4c735523dbaf14ba1749f2fe8269f9`; champion `8e55a3dd9c0982bebda9429fb00fea1ae261d650`; decision invalid.
  [Implementation](ideas/sella-medium-cno-h0/812895020f4c735523dbaf14ba1749f2fe8269f9/algo.py).
  [Training evidence](evaluation_results/cycle-396-train.json).
  [Narrative](full_log.md#cycle-396--010-ha-2-coordinate-cno-hessian-on-connected-12n30).

## sella-medium-ccl-h0: 0.10 Ha 4-coordinate C–C–Cl on connected 12–30
Status: incorporated

**Hypothesis:** Valid leftover 135169446 has unused 4-coordinate C–C–Cl. Soft 0.10 Ha cheapens dummy-linear alkyl chlorides. Cycle 397 extraed 103954262 whose attached carbon is 3-coordinate. Cycle 398 skipped that extra; leftover C–C–Cl twitched only.

**Outcome and uncertainty:** Cycle 397 discard. Cycle 398 NG. Cycle 519 NG vs `55dbd7a` (train Δ=+1.066e-4, valid leftover twitch). Cycle 521 **keep** stacked with 1,3-dithiole S–C–S on 30–80.

**Reason to revisit:** Incorporated in champion `ffd575c`. Do not drop the skip of 3-coordinate carbon terminals.

**Next experiment:** None as a C–C–Cl scale interpolation.

**Attempts:**
- Cycle 521; candidate `ffd575c14ebd2a9da555e6bc8f786766fd428bed`; champion `55dbd7a57c878cb8f99a15468f3e3bb576af4403`; decision keep.
  [Training evidence](evaluation_results/cycle-521-train.json).
  [Validation evidence](evaluation_results/cycle-521-valid.json).
  [Narrative](full_log.md#cycle-521--ccl-plus-13-dithiole-scs-on-30n80).
- Cycle 519; candidate `7787214bd9f9dedfa9b042023b944ecdc9b5a5ba`; champion `55dbd7a57c878cb8f99a15468f3e3bb576af4403`; decision non_generalizable.
  [Implementation](ideas/sella-medium-ccl-h0/7787214bd9f9dedfa9b042023b944ecdc9b5a5ba/algo.py).
  [Training evidence](evaluation_results/cycle-519-train.json).
  [Validation evidence](evaluation_results/cycle-519-valid.json).
  [Narrative](full_log.md#cycle-519--restack-010-ha-4-coordinate-ccl-on-connected-12n30).
- Cycle 398; candidate `1e266540fa97b2e55eb59d2b8f7764bb174c69d4`; champion `8e55a3dd9c0982bebda9429fb00fea1ae261d650`; decision non_generalizable.
  [Implementation](ideas/sella-medium-ccl-h0/1e266540fa97b2e55eb59d2b8f7764bb174c69d4/algo.py).
  [Training evidence](evaluation_results/cycle-398-train.json).
  [Validation evidence](evaluation_results/cycle-398-valid.json).
  [Narrative](full_log.md#cycle-398--skip-3-coordinate-carbon-terminal-on-ccl-010-ha).

## sella-hydrocarbon-skip-gdiis: skip dimer GDIIS on C/H-only dimers
Status: incorporated

**Hypothesis:** Cycle 180 GDIIS after 20 caused leftover alkenes–alkenes 155→197. Train C/H dimers stay at-ref, so they do not need that interpolant. Skipping two-point GDIIS on C/H-only dimers leaves hetero DES on cycle-180.

**Outcome and uncertainty:** Cycle 540 NG: skip GDIIS only, leftover 197→195, valid Δ=+2.775e-5. Cycle 541 **keep** vs `ffd575c`: also skip RFO@80 on C/H dimers. Train Δ=+1.777e-4 (N-oxide). Valid Δ=+5.827e-4: leftover alkenes 197→155, reference well. Energy-safe.

**Reason to revisit:** Incorporated in champion `476747f`. Do not skip GDIIS or RFO@80 on hetero dimers.

**Next experiment:** None as a C/H interpolant interpolation. See dummy-linear leftover 363892164.

**Attempts:**
- Cycle 541; candidate `476747fdffbbabeb258760c23f8f2db49006a86d`; champion `ffd575c14ebd2a9da555e6bc8f786766fd428bed`; decision keep.
  [Training evidence](evaluation_results/cycle-541-train.json).
  [Validation evidence](evaluation_results/cycle-541-valid.json).
  [Narrative](full_log.md#cycle-541--n-oxide-n18-plus-skip-gdiis-and-rfo80-on-hydrocarbon-dimers).
- Cycle 540; candidate `e21e9bc5a2c26689684712913b627db988a97f53`; champion `ffd575c14ebd2a9da555e6bc8f786766fd428bed`; decision non_generalizable.
  [Implementation](ideas/sella-hydrocarbon-skip-gdiis/e21e9bc5a2c26689684712913b627db988a97f53/algo.py).
  [Training evidence](evaluation_results/cycle-540-train.json).
  [Validation evidence](evaluation_results/cycle-540-valid.json).
  [Narrative](full_log.md#cycle-540--n-oxide-n18-plus-skip-gdiis-on-hydrocarbon-dimers).

## sella-cf2-ccc-h0: 0.10 Ha cyclobutane CF2 C–C–C on connected 30≤n<80
Status: closed

**Hypothesis:** Leftover 363892164 unused C–C–C at 4-coordinate cyclobutane CF2 cheapens the dummy-linear extra after F–C–C closed.

**Outcome and uncertainty:** Cycle 542 discard vs `476747f`. Δ=0. Leftover 363892164 energy twitch at 43. CF2 C–C–C fires and is not the limiter.

**Reason to revisit:** Not as a 0.10/0.08 CF2 C–C–C interpolation.

**Next experiment:** Close leftover 363892164 CF2 C–C–C. See dummy-linear leftover 363892164.

**Attempts:**
- Cycle 542; candidate `c368ed1cf5e3c6a6ac542b01d0a6840e6a0bf9af`; champion `476747fdffbbabeb258760c23f8f2db49006a86d`; decision discard.
  [Implementation](ideas/sella-cf2-ccc-h0/c368ed1cf5e3c6a6ac542b01d0a6840e6a0bf9af/algo.py).
  [Training evidence](evaluation_results/cycle-542-train.json).
  [Narrative](full_log.md#cycle-542--010-ha-cyclobutane-cf2-ccc-on-connected-30n80).

## sella-isocyanide-skip-gdiis: skip GDIIS on connected isocyanides
Status: closed

**Hypothesis:** Leftover 363892164 GDIIS after 20 lengthens the dummy-linear isocyanide tail. Skipping GDIIS on connected 1-coordinate C–N isocyanides cheapens leftover without other interpolants.

**Outcome and uncertainty:** Cycle 543 discard vs `476747f`. Δ=0, bit-identical. GDIIS never accepts on leftover.

**Reason to revisit:** Not as an isocyanide GDIIS skip.

**Next experiment:** Close. See dummy-linear leftover 363892164.

**Attempts:**
- Cycle 543; candidate `1a4f1008e75c6f40eec50c1de8945755098c507b`; champion `476747fdffbbabeb258760c23f8f2db49006a86d`; decision discard.
  [Training evidence](evaluation_results/cycle-543-train.json).
  [Narrative](full_log.md#cycle-543--skip-gdiis-on-connected-isocyanides).

## sella-isocyanide-exact-geodesic: exact geodesic Binv on connected isocyanides
Status: closed

**Hypothesis:** Cycle 429 exact geodesic cheapened leftover 363892164 but extraed venetoclax. Gating it to connected isocyanides keeps the dummy-linear save and spares those extras.

**Outcome and uncertainty:** Cycle 544 discard vs `476747f`. Δ=0. Leftover 363892164 energy twitch at 43. Geodesic fires and is not the limiter.

**Reason to revisit:** Not as an isocyanide geodesic interpolation.

**Next experiment:** Close. See dummy-linear leftover 363892164.

**Attempts:**
- Cycle 544; candidate `667d5b3fcfb81113d223ca85cec5b89486464125`; champion `476747fdffbbabeb258760c23f8f2db49006a86d`; decision discard.
  [Implementation](ideas/sella-isocyanide-exact-geodesic/667d5b3fcfb81113d223ca85cec5b89486464125/algo.py).
  [Training evidence](evaluation_results/cycle-544-train.json).
  [Narrative](full_log.md#cycle-544--exact-geodesic-binv-on-connected-isocyanides).

## sella-windowed-dummy-dihedral-015: 0.15 Ha windowed dummy-dihedral H0
Status: incorporated

**Hypothesis:** Cycle 286 0.20 Ha windowed dummy-dihedrals still leave leftover 363892164 at 43/31. 0.15 Ha enlarges those linear-bend torsions.

**Outcome and uncertainty:** Cycle 545 discard vs `476747f`, Δ=−9.23e-5. Leftover 43→42 and acalabrutinib 32→31 undone by silyl triyne 135191719 +2 and propargyl alcohol 134979308 +1. Cycle 546 NG: Si/propargyl skip, train Δ=+1.249e-4 (363892164 43→42, acalabrutinib 32→31), valid Δ=0. Cycle 550 **keep** stacked n<18 allene dummy-dihedral 0.15 (leftover 135169446 23→22). New champion `d6f05c6`.

**Reason to revisit:** Incorporated. Do not re-enable Si or propargyl-alcohol 0.15. Do not interpolate 0.12 globally (cycle 566 leftover twitch). Cycle 565 kept n<18 allene 0.12.

**Next experiment:** None as a windowed 0.12 interpolation. See limiter wd 0.7 on n≥30 `alkyne_soft` dummy dihedrals.

**Attempts:**
- Cycle 550; candidate `d6f05c682acea05c3ff7dfe54f20f98d1c2bcd73`; champion `476747fdffbbabeb258760c23f8f2db49006a86d`; decision keep.
  [Training evidence](evaluation_results/cycle-550-train.json).
  [Validation evidence](evaluation_results/cycle-550-valid.json).
  [Narrative](full_log.md#cycle-550--cycle-546-stack-plus-015-dummy-dihedral-on-n18-allenes).
- Cycle 546; candidate `de7355c083174302afc64998282ffd2c892ed843`; champion `476747fdffbbabeb258760c23f8f2db49006a86d`; decision non_generalizable.
  [Implementation](ideas/sella-windowed-dummy-dihedral-015/de7355c083174302afc64998282ffd2c892ed843/algo.py).
  [Training evidence](evaluation_results/cycle-546-train.json).
  [Validation evidence](evaluation_results/cycle-546-valid.json).
  [Narrative](full_log.md#cycle-546--015-ha-windowed-alkyne-dummy-dihedral-skip-si-and-propargyl-alcohol).
- Cycle 545; candidate `91d8824ef2c737677c969ac4efba2c92a7d7eec5`; champion `476747fdffbbabeb258760c23f8f2db49006a86d`; decision discard.
  [Implementation](ideas/sella-windowed-dummy-dihedral-015/91d8824ef2c737677c969ac4efba2c92a7d7eec5/algo.py).
  [Training evidence](evaluation_results/cycle-545-train.json).
  [Narrative](full_log.md#cycle-545--015-ha-windowed-dummy-dihedral-hessian).

## sella-isocyanide-dummy-dihedral-015: 0.15 Ha isocyanide dummy-dihedral on n≥30
Status: closed

**Hypothesis:** Train leftover 363892164 unused isocyanide dummy dihedrals still use 0.25 Ha. Soft 0.15 Ha on 1–2 dummies at 2-coordinate N bonded to 1-coordinate C cheapens leftover after the 546 alkyne save.

**Outcome and uncertainty:** Cycle 552 discard vs `d6f05c6`. Δ=0. Leftover 363892164 energy twitch at 42. Class fires and is not the limiter.

**Reason to revisit:** Not as an isocyanide dummy-dihedral scale interpolation. Do not interpolate 0.12.

**Next experiment:** None. Close isocyanide dummy-dihedral 0.15. See dummy-angle 0.08 on n≥30 isocyanide dummies.

**Attempts:**
- Cycle 552; candidate `94bbe5bd835d4fad82bb0fac59344c81c93aa9dc`; champion `d6f05c682acea05c3ff7dfe54f20f98d1c2bcd73`; decision discard.
  [Implementation](ideas/sella-isocyanide-dummy-dihedral-015/94bbe5bd835d4fad82bb0fac59344c81c93aa9dc/algo.py).
  [Training evidence](evaluation_results/cycle-552-train.json).
  [Narrative](full_log.md#cycle-552--015-ha-isocyanide-dummy-dihedral-on-n30).

## sella-isocyanide-dummy-angle-008: 0.08 Ha dummy-angle on n≥30 isocyanides
Status: closed

**Hypothesis:** Leftover 363892164 dummy-involving angles at isocyanide N still use 0.10 Ha after dummy-dihedral 0.15 twitched. Soft 0.08 Ha on those dummy angles cheapens leftover.

**Outcome and uncertainty:** Cycle 553 discard vs `d6f05c6`. Δ=0. Leftover 363892164 energy twitch at 42. Class fires and is not the limiter.

**Reason to revisit:** Not as an isocyanide dummy-angle scale interpolation. Do not interpolate 0.05.

**Next experiment:** None. Close leftover 363892164 isocyanide dummy H0. See train leftover 252089162 extra=2.

**Attempts:**
- Cycle 553; candidate `8a48fd6795af18351e8a58a9b5a8daecc158409f`; champion `d6f05c682acea05c3ff7dfe54f20f98d1c2bcd73`; decision discard.
  [Implementation](ideas/sella-isocyanide-dummy-angle-008/8a48fd6795af18351e8a58a9b5a8daecc158409f/algo.py).
  [Training evidence](evaluation_results/cycle-553-train.json).
  [Narrative](full_log.md#cycle-553--008-ha-dummy-angle-on-n30-isocyanides).

## sella-n12-pop-opo-008: 0.08 Ha P–O–P / O–P–O on connected n<12
Status: incorporated

**Hypothesis:** Train leftover 252089162 (OP(O)OP(O)O, extra=2) still uses 0.10 Ha phosphate angles. Soft 0.08 Ha on P–O–P and O–P–O at n<12 cheapens leftover.

**Outcome and uncertainty:** Cycle 554 discard vs `d6f05c6`, Δ=+8.53e-5. 135094180 24→23; leftover 252089162 twitch. Cycle 562 valid extra 135259008 Si cage. Cycle 563 silicon skip restored valid. Cycle 565 **keep** stacked with carbamate N–C–O and n<18 allene dummy-dihedral 0.12.

**Reason to revisit:** Incorporated with silicon skip. Do not interpolate 0.05. Close P–O–H 0.08. Do not drop the Si skip (135259008 extra).

**Next experiment:** Leave P–O–P/O–P–O 0.08 with the silicon skip. Do not restack leftover 252089162 phosphate H0.

**Attempts:**
- Cycle 565; candidate `5b183d43d9c42b918e209ab6e11471afe1f07ba5`; champion `d6f05c682acea05c3ff7dfe54f20f98d1c2bcd73`; decision keep.
  [Training evidence](evaluation_results/cycle-565-train.json).
  [Validation evidence](evaluation_results/cycle-565-valid.json).
  [Narrative](full_log.md#cycle-565--cycle-563-stack-plus-012-dummy-dihedral-on-n18-allenes).
- Cycle 563; candidate `0a388dee47e3247e01423a4e749885c82607a6db`; champion `d6f05c682acea05c3ff7dfe54f20f98d1c2bcd73`; decision non_generalizable.
  [Implementation](ideas/sella-carbamate-nco-h0/0a388dee47e3247e01423a4e749885c82607a6db/algo.py).
  [Training evidence](evaluation_results/cycle-563-train.json).
  [Validation evidence](evaluation_results/cycle-563-valid.json).
  [Narrative](full_log.md#cycle-563--562-repair-skip-si-phosphates-and-n-sulfonyl-carbamates).
- Cycle 561; candidate `710a114c800a2b024f64219725334b22c5dd698d`; champion `d6f05c682acea05c3ff7dfe54f20f98d1c2bcd73`; decision discard.
  [Implementation](ideas/sella-n12-pop-opo-008/710a114c800a2b024f64219725334b22c5dd698d/algo.py).
  [Training evidence](evaluation_results/cycle-561-train.json).
  [Narrative](full_log.md#cycle-561--cycle-554-phosphate-008-plus-008-dummy-angle-on-n30-alkyne_soft).
- Cycle 557; candidate `63453960bc564d6c9112ddacdbfafe501e3ab11b`; champion `d6f05c682acea05c3ff7dfe54f20f98d1c2bcd73`; decision discard.
  [Implementation](ideas/sella-n12-pop-opo-008/63453960bc564d6c9112ddacdbfafe501e3ab11b/algo.py).
  [Training evidence](evaluation_results/cycle-557-train.json).
  [Narrative](full_log.md#cycle-557--cycle-554-phosphate-008-plus-exact-geodesic-on-1012-pop).
- Cycle 555; candidate `5776b0d00b767e7c02f37e1b1c9ef0521fe69958`; champion `d6f05c682acea05c3ff7dfe54f20f98d1c2bcd73`; decision discard.
  [Implementation](ideas/sella-n12-pop-opo-008/5776b0d00b767e7c02f37e1b1c9ef0521fe69958/algo.py).
  [Training evidence](evaluation_results/cycle-555-train.json).
  [Narrative](full_log.md#cycle-555--008-ha-pop-poh-and-opo-on-connected-n12).
- Cycle 554; candidate `5bb9f74e0f183d3e712a9ab089167521a471e55b`; champion `d6f05c682acea05c3ff7dfe54f20f98d1c2bcd73`; decision discard.
  [Implementation](ideas/sella-n12-pop-opo-008/5bb9f74e0f183d3e712a9ab089167521a471e55b/algo.py).
  [Training evidence](evaluation_results/cycle-554-train.json).
  [Narrative](full_log.md#cycle-554--008-ha-pop-and-opo-on-connected-n12).

## sella-carbamate-nco-h0: 0.10 Ha carbamate N–C–O on connected 30≤n<80
Status: incorporated

**Hypothesis:** Leftover 363892164 unused N–C–O at 3-coordinate carbamate carbon {N, O, O} with 2-coordinate ether oxygen is distinct from C5 O–C–C already at 0.10. Soft 0.10 Ha on 1–2 such angles at 30≤n<80, stacked on cycle-554 phosphate 0.08, can pass train δ.

**Outcome and uncertainty:** Cycle 562 NG (valid extras). Cycle 563 NG (train-passing, valid-safe). Cycle 565 **keep**: train 135267488 45→43 with N-sulfonyl skip; leftover oxazolidinone twitch. Do not skip all N–S.

**Reason to revisit:** Incorporated with N-sulfonyl skip. Leftover 363892164 N–C–O is not the limiter. Do not drop the N-sulfonyl skip (135065494 extra). Cycle 569 0.08 extraed leftover and 135267488. Keep 0.10.

**Next experiment:** Leave carbamate N–C–O 0.10. Do not interpolate 0.08. See leftover 363892164 pyridine C–N–C cap.

**Attempts:**
- Cycle 565; candidate `5b183d43d9c42b918e209ab6e11471afe1f07ba5`; champion `d6f05c682acea05c3ff7dfe54f20f98d1c2bcd73`; decision keep.
  [Training evidence](evaluation_results/cycle-565-train.json).
  [Validation evidence](evaluation_results/cycle-565-valid.json).
  [Narrative](full_log.md#cycle-565--cycle-563-stack-plus-012-dummy-dihedral-on-n18-allenes).
- Cycle 563; candidate `0a388dee47e3247e01423a4e749885c82607a6db`; champion `d6f05c682acea05c3ff7dfe54f20f98d1c2bcd73`; decision non_generalizable.
  [Implementation](ideas/sella-carbamate-nco-h0/0a388dee47e3247e01423a4e749885c82607a6db/algo.py).
  [Training evidence](evaluation_results/cycle-563-train.json).
  [Validation evidence](evaluation_results/cycle-563-valid.json).
  [Narrative](full_log.md#cycle-563--562-repair-skip-si-phosphates-and-n-sulfonyl-carbamates).
- Cycle 562; candidate `68d79150dcfdcf7d2352400b8410503a3b6f301a`; champion `d6f05c682acea05c3ff7dfe54f20f98d1c2bcd73`; decision non_generalizable.
  [Implementation](ideas/sella-carbamate-nco-h0/68d79150dcfdcf7d2352400b8410503a3b6f301a/algo.py).
  [Training evidence](evaluation_results/cycle-562-train.json).
  [Validation evidence](evaluation_results/cycle-562-valid.json).
  [Narrative](full_log.md#cycle-562--cycle-554-phosphate-008-plus-oxazolidinone-carbamate-nco).

## sella-n18-allene-ccc-h0: 0.10 Ha allene C–C–C on connected n<18
Status: closed

**Hypothesis:** Valid leftover 135169446 unused C–C–C at 2-coordinate allene carbon (both terminals 3-coord C) is the limiter after leftover H0 extras. Soft 0.10 Ha on 1–2 such angles at connected n<18, stacked on cycle-546 windowed-alkyne 0.15, would pass valid δ with the 546 train keep.

**Outcome and uncertainty:** Cycle 547 non_generalizable vs `476747f`. Train and valid n_steps-identical to cycle 546. Leftover 135169446 bit-identical at 23. `_allene_ccc` skips dummy-involving angles; linear allene uses Sella dummy-linear bends, so the valence C–C–C is absent.

**Reason to revisit:** Not as allene C–C–C valence H0. Do not interpolate 0.08 or drop the dummy skip.

**Next experiment:** None. Close leftover 135169446 allene C–C–C. Skip GDIIS on n<18 allenes is also closed (cycle 548). See exact geodesic on connected n<18 allenes stacked on cycle 546.

**Attempts:**
- Cycle 547; candidate `99f369043f55d20374a4b5adaf8bfae866e2d2fb`; champion `476747fdffbbabeb258760c23f8f2db49006a86d`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-547-train.json).
  [Validation evidence](evaluation_results/cycle-547-valid.json).
  [Narrative](full_log.md#cycle-547--cycle-546-stack-plus-allene-ccc-on-connected-n18).

## sella-n18-allene-skip-gdiis: skip GDIIS on connected n<18 allenes
Status: closed

**Hypothesis:** Leftover 135169446 finishes at 23, inside the post-20 GDIIS window. Skipping GDIIS on connected n<18 allenes cheapens leftover without other interpolants. Train has no n<18 allene.

**Outcome and uncertainty:** Cycle 548 non_generalizable vs `476747f`. Train and valid n_steps-identical to cycle 546. Leftover 135169446 bit-identical at 23. Matcher tags leftover; GDIIS never accepts under the ρ/cosine window (same no-op as cycle 543).

**Reason to revisit:** Not as an n<18 allene GDIIS skip.

**Next experiment:** None. Close skip-GDIIS on n<18 allenes. See exact geodesic on connected n<18 allenes stacked on cycle 546.

**Attempts:**
- Cycle 548; candidate `7ffe7f2dd148a7222dcf6b1b9b5ece5e47a36b4a`; champion `476747fdffbbabeb258760c23f8f2db49006a86d`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-548-train.json).
  [Validation evidence](evaluation_results/cycle-548-valid.json).
  [Narrative](full_log.md#cycle-548--cycle-546-stack-plus-skip-gdiis-on-n18-allenes).

## sella-n18-allene-exact-geodesic: exact geodesic Binv on connected n<18 allenes
Status: closed

**Hypothesis:** Leftover 135169446 is dummy-linear. Exact geodesic on connected n<18 allenes cheapens leftover without n≥30 extras. Train has no n<18 allene.

**Outcome and uncertainty:** Cycle 549 non_generalizable vs `476747f`. Train n_steps-identical to cycle 546. Valid leftover 135169446 energy twitch at 23. Cycle 575 twitch at 21. Cycle 658 extras leftover 20→21. Cycle 784 extras leftover 20→21 stacked on cycle 783 (Δ_valid=−1.19e-4). Geodesic extras leftover dummy-linear steps at leftover 20.

**Reason to revisit:** Not as an n<18 allene geodesic interpolation. Do not drop the 18–20 window. Do not restack geodesic on leftover 135169446 with cycle 783.

**Next experiment:** None. Close exact geodesic on n<18 allenes. Close leftover 135169446 `wd` below 0.70. See Helgaker |λ| floor 0.001 on `_has_allene` stacked on cycle 783.

**Attempts:**
- Cycle 549; candidate `81b35d564141e871d0f7c64e407fadf2037f13cd`; champion `476747fdffbbabeb258760c23f8f2db49006a86d`; decision non_generalizable.
  [Implementation](ideas/sella-n18-allene-exact-geodesic/81b35d564141e871d0f7c64e407fadf2037f13cd/algo.py).
  [Training evidence](evaluation_results/cycle-549-train.json).
  [Validation evidence](evaluation_results/cycle-549-valid.json).
  [Narrative](full_log.md#cycle-549--cycle-546-stack-plus-exact-geodesic-on-n18-allenes).
- Cycle 784; candidate `05016dbdd2296d66ba037f5dda9ee6456279c7f2`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-784-train.json).
  [Validation evidence](evaluation_results/cycle-784-valid.json).
  [Narrative](full_log.md#cycle-784--cycle-783-plus-exact-geodesic-on-n18-allenes).

## sella-noxide-onc-h0: 0.10 Ha pyridine N-oxide O–N–C on connected 12≤n<80
Status: incorporated

**Hypothesis:** Unused pyridine/imidazole N-oxide O–N–C cheapens leftover 135092053. Cycle 403 extraed valid 104063503 (n=47, one N-oxide). Valid leftover 252656292 has two N-oxides and was skipped by the 1–2 cap.

**Outcome and uncertainty:** Cycle 403/404/522 NG as train-only until stacked. Cycle 541 **keep** stacked with skip GDIIS and RFO@80 on C/H dimers. n<18 1–2 cap is in champion `476747f`.

**Reason to revisit:** Incorporated. Do not fill 12–80 or 12–40 (extras). Do not restack leftover 252656292 / 135169446 / esters–phenol H0.

**Next experiment:** None as an N-oxide scale interpolation. See dummy-linear leftover 363892164.

**Attempts:**
- Cycle 541; candidate `476747fdffbbabeb258760c23f8f2db49006a86d`; champion `ffd575c14ebd2a9da555e6bc8f786766fd428bed`; decision keep.
  [Training evidence](evaluation_results/cycle-541-train.json).
  [Validation evidence](evaluation_results/cycle-541-valid.json).
  [Narrative](full_log.md#cycle-541--n-oxide-n18-plus-skip-gdiis-and-rfo80-on-hydrocarbon-dimers).
- Cycle 539; candidate `3b055a6e4f160f7362a736ffd5420eb5bd1bc84f`; champion `ffd575c14ebd2a9da555e6bc8f786766fd428bed`; decision non_generalizable.
  [Implementation](ideas/sella-n18-vinyl-ch2oh-co-h0/3b055a6e4f160f7362a736ffd5420eb5bd1bc84f/algo.py).
  [Training evidence](evaluation_results/cycle-539-train.json).
  [Validation evidence](evaluation_results/cycle-539-valid.json).
  [Narrative](full_log.md#cycle-539--vinylallene-ch2oh-co-stretch-with-3-coordinate-carbon-neighbor).
- Cycle 538; candidate `dabf22f6594585e59b0a7dc678b64eaf518e7507`; champion `ffd575c14ebd2a9da555e6bc8f786766fd428bed`; decision non_generalizable.
  [Implementation](ideas/sella-n18-vinyl-ch2oh-co-h0/dabf22f6594585e59b0a7dc678b64eaf518e7507/algo.py).
  [Training evidence](evaluation_results/cycle-538-train.json).
  [Validation evidence](evaluation_results/cycle-538-valid.json).
  [Narrative](full_log.md#cycle-538--n-oxide-n18-plus-vinylallene-primary-alcohol-co-stretch).
- Cycle 537; candidate `9adf7618715ce4a7576cc28a388320e5ca0c54fa`; champion `ffd575c14ebd2a9da555e6bc8f786766fd428bed`; decision non_generalizable.
  [Implementation](ideas/sella-dimer-acetate-ester-cco-h0/9adf7618715ce4a7576cc28a388320e5ca0c54fa/algo.py).
  [Training evidence](evaluation_results/cycle-537-train.json).
  [Validation evidence](evaluation_results/cycle-537-valid.json).
  [Narrative](full_log.md#cycle-537--acetate-ester-cco-with-connecting-neighbor-and-medium-band-repair).
- Cycle 535; candidate `bd5c839c5d4dc6f45efba326899d21398abb1754`; champion `ffd575c14ebd2a9da555e6bc8f786766fd428bed`; decision non_generalizable.
  [Implementation](ideas/sella-dimer-acetate-ester-cco-h0/bd5c839c5d4dc6f45efba326899d21398abb1754/algo.py).
  [Training evidence](evaluation_results/cycle-535-train.json).
  [Validation evidence](evaluation_results/cycle-535-valid.json).
  [Narrative](full_log.md#cycle-535--n-oxide-n18-plus-phenol-gated-acetate-ester-cco-on-dimers).
- Cycle 534; candidate `c116f1d886c5ba6f8691332ee7831a7e1b9f5bcc`; champion `ffd575c14ebd2a9da555e6bc8f786766fd428bed`; decision non_generalizable.
  [Implementation](ideas/sella-fluoro-noxide-no-h0/c116f1d886c5ba6f8691332ee7831a7e1b9f5bcc/algo.py).
  [Training evidence](evaluation_results/cycle-534-train.json).
  [Validation evidence](evaluation_results/cycle-534-valid.json).
  [Narrative](full_log.md#cycle-534--n-oxide-n18-plus-fluoro-bis-n-oxide-no-stretch-on-30n80).
- Cycle 533; candidate `bd6a57768829c0764681263cf79e99643fd4aae5`; champion `ffd575c14ebd2a9da555e6bc8f786766fd428bed`; decision non_generalizable.
  [Implementation](ideas/sella-enyne-cc-stretch-h0/bd6a57768829c0764681263cf79e99643fd4aae5/algo.py).
  [Training evidence](evaluation_results/cycle-533-train.json).
  [Validation evidence](evaluation_results/cycle-533-valid.json).
  [Narrative](full_log.md#cycle-533--n-oxide-n18-plus-ch2clc2-stretch-on-n18).
- Cycle 532; candidate `89c4407a96b13320ebbb5ac48e46a8415d616cf6`; champion `ffd575c14ebd2a9da555e6bc8f786766fd428bed`; decision non_generalizable.
  [Implementation](ideas/sella-enyne-ccl-stretch-h0/89c4407a96b13320ebbb5ac48e46a8415d616cf6/algo.py).
  [Training evidence](evaluation_results/cycle-532-train.json).
  [Validation evidence](evaluation_results/cycle-532-valid.json).
  [Narrative](full_log.md#cycle-532--n-oxide-n18-plus-enyne-ccl-stretch-on-n18).
- Cycle 530; candidate `5d2b02cfeb4bc790c71c5619c7cff5445e81be7b`; champion `ffd575c14ebd2a9da555e6bc8f786766fd428bed`; decision non_generalizable.
  [Implementation](ideas/sella-naryl-alkyl-sch2-h0/5d2b02cfeb4bc790c71c5619c7cff5445e81be7b/algo.py).
  [Training evidence](evaluation_results/cycle-530-train.json).
  [Validation evidence](evaluation_results/cycle-530-valid.json).
  [Narrative](full_log.md#cycle-530--n-oxide-n18-plus-n-aryl-alkylaryl-sch2-stretch-on-30n80).
- Cycle 531; candidate `58c134697f84426b4a6133c8222b7d17444df05e`; champion `ffd575c14ebd2a9da555e6bc8f786766fd428bed`; decision non_generalizable.
  [Implementation](ideas/sella-protonated-ester-ome-h0/58c134697f84426b4a6133c8222b7d17444df05e/algo.py).
  [Training evidence](evaluation_results/cycle-531-train.json).
  [Validation evidence](evaluation_results/cycle-531-valid.json).
  [Narrative](full_log.md#cycle-531--n-oxide-n18-plus-protonated-ester-methoxy-co-stretch-on-30n80).
- Cycle 522; candidate `90068f480aa5a1c5f7744a57def2645a2434b53f`; champion `ffd575c14ebd2a9da555e6bc8f786766fd428bed`; decision non_generalizable.
  [Implementation](ideas/sella-noxide-onc-h0/90068f480aa5a1c5f7744a57def2645a2434b53f/algo.py).
  [Training evidence](evaluation_results/cycle-522-train.json).
  [Validation evidence](evaluation_results/cycle-522-valid.json).
  [Narrative](full_log.md#cycle-522--010-ha-n-oxide-onc-on-connected-n18).
- Cycle 404; candidate `f735de7acdd06755de49404edfed65fbd948824f`; champion `8e55a3dd9c0982bebda9429fb00fea1ae261d650`; decision non_generalizable.
  [Implementation](ideas/sella-noxide-onc-h0/f735de7acdd06755de49404edfed65fbd948824f/algo.py).
  [Training evidence](evaluation_results/cycle-404-train.json).
  [Validation evidence](evaluation_results/cycle-404-valid.json).
  [Narrative](full_log.md#cycle-404--n-oxide-onc-010-ha-on-12n40-with-14-cap).
- Cycle 403; candidate `0373e7e463b168facdb0994f8419b6a1f14307f7`; champion `8e55a3dd9c0982bebda9429fb00fea1ae261d650`; decision non_generalizable.
  [Implementation](ideas/sella-noxide-onc-h0/0373e7e463b168facdb0994f8419b6a1f14307f7/algo.py).
  [Training evidence](evaluation_results/cycle-403-train.json).
  [Validation evidence](evaluation_results/cycle-403-valid.json).
  [Narrative](full_log.md#cycle-403--010-ha-pyridine-n-oxide-onc-hessian-on-connected-12n80).

## sella-medium-alcohol-occ-h0: 0.10 Ha 4-coordinate alcohol O–C–C on connected 12–30
Status: closed

**Hypothesis:** Valid leftover 135169446 unused alcohol O–C–C is the limiter after C–O–H no-op. Cycle 401 extraed dimer monoatomics–phenol via connecting 4-coordinate carbons.

**Outcome and uncertainty:** Cycle 401 discard, Δ=−4.10e-5. Extra dimer 52→53; 135105134 twitch. Cycle 402 discard, Δ=−5.76e-5. Dimer extra gone; only extra 135086807 37→38. Isolated-alcohol/enyne gates would empty train.

**Reason to revisit:** Not as an alcohol O–C–C H0 scale on 12–30. Train hits extra; leftover is valid-only.

**Next experiment:** Close this family. Do not interpolate 0.13 or drop the 3-coord terminal.

**Attempts:**
- Cycle 402; candidate `9c246474b1d3e2bd2fcc65d993bc8ea1cde0c0d6`; champion `8e55a3dd9c0982bebda9429fb00fea1ae261d650`; decision discard.
  [Implementation](ideas/sella-medium-alcohol-occ-h0/9c246474b1d3e2bd2fcc65d993bc8ea1cde0c0d6/algo.py).
  [Training evidence](evaluation_results/cycle-402-train.json).
  [Narrative](full_log.md#cycle-402--allylic-3-coordinate-carbon-terminal-on-alcohol-occ-010-ha).
- Cycle 401; candidate `7739e1f9c82cd64d994b722b1465aed8a967b88b`; champion `8e55a3dd9c0982bebda9429fb00fea1ae261d650`; decision discard.
  [Implementation](ideas/sella-medium-alcohol-occ-h0/7739e1f9c82cd64d994b722b1465aed8a967b88b/algo.py).
  [Training evidence](evaluation_results/cycle-401-train.json).
  [Narrative](full_log.md#cycle-401--010-ha-4-coordinate-alcohol-occ-hessian-on-connected-12n30).

## sella-medium-gdiis-cos-085: GDIIS cosine 0.85 on connected 12≤n<30
Status: closed

**Hypothesis:** Dummy-gap leftovers reject GDIIS at cosine 0.90. 0.85 on connected 12≤n<30 admits those interpolants without cycle 242’s n≥30 extra.

**Outcome and uncertainty:** Cycle 400 discard, Δ=+7.61e-5. Energy-safe. Only save 135245411 28→27; leftovers unchanged. Same 12–30 save as cycle 137 cosine 0.80, without the n≥30 extras.

**Reason to revisit:** Not as a cosine floor below 0.90 on 12–30. Do not try 0.80 on that band.

**Next experiment:** Close this family.

**Attempts:**
- Cycle 400; candidate `e48250fa433827e5c1741d264a9d9908c39499eb`; champion `8e55a3dd9c0982bebda9429fb00fea1ae261d650`; decision discard.
  [Implementation](ideas/sella-medium-gdiis-cos-085/e48250fa433827e5c1741d264a9d9908c39499eb/algo.py).
  [Training evidence](evaluation_results/cycle-400-train.json).
  [Narrative](full_log.md#cycle-400--gdiis-cosine-085-on-connected-12n30).

## sella-medium-alkyl-coh-h0: 0.10 Ha alkyl C–O–H on connected 12–30
Status: closed

**Hypothesis:** Leftover 135169446 unused alkyl C–O–H is the limiter after C–C–Cl twitched.

**Outcome and uncertainty:** Cycle 399 discard, Δ=0. Energy-safe. n_steps bit-identical; two energy twitches. Not a limiter on train 12–30.

**Reason to revisit:** Not as an alkyl C–O–H H0 scale on 12–30. Do not interpolate 0.13.

**Next experiment:** Close this family.

**Attempts:**
- Cycle 399; candidate `ee6ccebf81004870bfdac282588049fd07f02912`; champion `8e55a3dd9c0982bebda9429fb00fea1ae261d650`; decision discard.
  [Implementation](ideas/sella-medium-alkyl-coh-h0/ee6ccebf81004870bfdac282588049fd07f02912/algo.py).
  [Training evidence](evaluation_results/cycle-399-train.json).
  [Narrative](full_log.md#cycle-399--010-ha-alkyl-coh-hessian-on-connected-12n30).
- Cycle 397; candidate `3f71f02c9a6fac4ef94db7bcf29a6928aee55010`; champion `8e55a3dd9c0982bebda9429fb00fea1ae261d650`; decision discard.
  [Implementation](ideas/sella-medium-ccl-h0/3f71f02c9a6fac4ef94db7bcf29a6928aee55010/algo.py).
  [Training evidence](evaluation_results/cycle-397-train.json).
  [Narrative](full_log.md#cycle-397--010-ha-4-coordinate-ccl-hessian-on-connected-12n30).

## sella-pyrazole-cnn-h0: 0.10 Ha 3-coordinate pyrazole C–N–N on connected 30≤n_atoms<80
Status: closed

**Hypothesis:** 440723624 extra +1 is unused 3-coordinate pyrazole NH C–N–N. Soft 0.10 Ha on 1–2 such angles at 30≤n<80 cheapens it without 2-coordinate azide C–N–N (cycle 301).

**Outcome and uncertainty:** Cycle 371 discard, Δ=0. n_steps bit-identical; energy twitches on the 1–2-cap hits including 440723624. The class fires and is not a limiter.

**Reason to revisit:** Not as a 3-coordinate C–N–N H0 scale.

**Next experiment:** Do not interpolate 0.13. Isolated single-CF3 F–C–C is next.

**Attempts:**
- Cycle 371; candidate `349024f854e7a86e86701fd1975499203e391d2f`; champion `7140449e9cbddb6f6589ec34fdac86f88dc413e1`; decision discard.
  [Implementation](ideas/sella-pyrazole-cnn-h0/349024f854e7a86e86701fd1975499203e391d2f/algo.py).
  [Training evidence](evaluation_results/cycle-371-train.json).
  [Narrative](full_log.md#cycle-371--010-ha-3-coordinate-pyrazole-cnn-hessian).

## sella-dummy-dihedral-020-n12: dummy-dihedral 0.20 Ha on connected n_atoms<12
Status: revisiting

**Hypothesis:** Windowed dummy-dihedrals already use 0.20 Ha on n≥30. Applying 0.20 on connected n<12 cheapens leftover 252089162 without cycle 105’s large-n extras.

**Outcome and uncertainty:** Cycles 415–417 **keep**. Cycle 417 train Δ=+5.02e-4 (135149279 18→14, 135043047 43→42; no extras). Valid Δ=+2.42e-4 (five 18–29 saves vs extra 104129283 46→50). Leftovers 433943527 and 135095297 unchanged. New champion `d2f7137d0609977f7ebd973704e68b8f419f8dad`. Cycle 426 dummy-dihedral 0.18 on n<18 discard, Δ=−7.12e-4 (carboxylates–thiols 27→41).

**Reason to revisit:** Remaining 0.25 is n≥30 non-windowed dummy dihedrals. Cycle 105 extras on 104046121; do not unwindow C–C–C. Do not interpolate below 0.20 on n<18.

**Next experiment:** Do not interpolate dummy-dihedral 0.18/0.16. Do not unwindow C–C–C.

**Attempts:**
- Cycle 426; candidate `d3a2a584a09a2c4d6c157fc148e330c50640a778`; champion `d2f7137d0609977f7ebd973704e68b8f419f8dad`; decision discard.
  [Implementation](ideas/sella-dummy-dihedral-020-n12/d3a2a584a09a2c4d6c157fc148e330c50640a778/algo.py).
  [Training evidence](evaluation_results/cycle-426-train.json).
  [Narrative](full_log.md#cycle-426--dummy-dihedral-018-ha-on-connected-n_atoms18).
- Cycle 417; candidate `d2f7137d0609977f7ebd973704e68b8f419f8dad`; champion `67c840172a2ef352eb2e87a657d146f84a40ca67`; decision keep.
  [Training evidence](evaluation_results/cycle-417-train.json).
  [Validation evidence](evaluation_results/cycle-417-valid.json).
  [Narrative](full_log.md#cycle-417--dummy-dihedral-020-ha-on-connected-n_atoms30).
- Cycle 416; candidate `67c840172a2ef352eb2e87a657d146f84a40ca67`; champion `77770b89368405cc71f3970d09ad2a571ef3c19d`; decision keep.
  [Training evidence](evaluation_results/cycle-416-train.json).
  [Validation evidence](evaluation_results/cycle-416-valid.json).
  [Narrative](full_log.md#cycle-416--dummy-dihedral-020-ha-on-connected-n_atoms18).
- Cycle 415; candidate `77770b89368405cc71f3970d09ad2a571ef3c19d`; champion `8e55a3dd9c0982bebda9429fb00fea1ae261d650`; decision keep.
  [Training evidence](evaluation_results/cycle-415-train.json).
  [Validation evidence](evaluation_results/cycle-415-valid.json).
  [Narrative](full_log.md#cycle-415--dummy-dihedral-020-ha-on-connected-n_atoms12).

## sella-medium-delta018: connected MIS floor 0.18 after 20 on n_atoms<30
Status: revisiting

**Hypothesis:** Dummy-dihedral 0.20 on n<30 is still truncated by the 0.15 MIS floor. 0.18 on that band enlarges dummy-linear steps without paliperidone (cycle 246 n≥30 extras).

**Outcome and uncertainty:** Cycle 418 discard, Δ=−3.0e-6 (perfluoro extra). Cycle 419 discard, Δ=+5.47e-5 (dummy gate; only 440717067 −1). Cycle 420 discard, Δ=+1.4e-6 (0.20 extras 135043047). Cycle 421 non_generalizable: train Δ=+1.97e-4 (440717067 and bromo leftover); valid bit-identical.

**Reason to revisit:** Dummy-gated 0.18 and bromo-ether C–O–C remain stacked train-only. Valid leftover 135169446 is not late-trust limited.

**Next experiment:** Limiter dummy-dihedral wd=0.7 on connected n<18.

**Attempts:**
- Cycle 453; candidate `584d8a459788dd0ead3cf7964c32d0e282381488`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision non_generalizable.
  [Implementation](ideas/sella-dummy-wd07-n18/584d8a459788dd0ead3cf7964c32d0e282381488/algo.py).
  [Training evidence](evaluation_results/cycle-453-train.json).
  [Validation evidence](evaluation_results/cycle-453-valid.json).
  [Narrative](full_log.md#cycle-453--dummy-dihedral-limiter-wd07-on-connected-n_atoms8).
- Cycle 452; candidate `b127694e9b49a190443922033fdcf4b511fc73e4`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision discard.
  [Implementation](ideas/sella-dummy-wd07-n18/b127694e9b49a190443922033fdcf4b511fc73e4/algo.py).
  [Training evidence](evaluation_results/cycle-452-train.json).
  [Narrative](full_log.md#cycle-452--dummy-dihedral-limiter-wd07-on-connected-n_atoms18).
- Cycle 451; candidate `386e8f989f6d84a50fc19aa52a6db9a31df67470`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision discard.
  [Training evidence](evaluation_results/cycle-451-train.json).
  [Narrative](full_log.md#cycle-451--connected-mis-floor-018-after-20-on-n_atoms18).

**Attempts:**
- Cycle 421; candidate `128bbb561e7709176a7a369de9a0039d1d418e39`; champion `d2f7137d0609977f7ebd973704e68b8f419f8dad`; decision non_generalizable.
  [Implementation](ideas/sella-medium-delta018/128bbb561e7709176a7a369de9a0039d1d418e39/algo.py).
  [Training evidence](evaluation_results/cycle-421-train.json).
  [Validation evidence](evaluation_results/cycle-421-valid.json).
  [Narrative](full_log.md#cycle-421--dummy-gated-018-floor-plus-3-coord-bromo-ether-coc).
- Cycle 420; candidate `4bc82af5d9d04008400bdaa5ff9df3a480141b81`; champion `d2f7137d0609977f7ebd973704e68b8f419f8dad`; decision discard.
  [Implementation](ideas/sella-medium-delta018/4bc82af5d9d04008400bdaa5ff9df3a480141b81/algo.py).
  [Training evidence](evaluation_results/cycle-420-train.json).
  [Narrative](full_log.md#cycle-420--dummy-gated-mis-floor-020-after-20-on-n_atoms30).
- Cycle 419; candidate `3c3fb71433711ac8fa4ece6c9ed3aa8ef707130f`; champion `d2f7137d0609977f7ebd973704e68b8f419f8dad`; decision discard.
  [Implementation](ideas/sella-medium-delta018/3c3fb71433711ac8fa4ece6c9ed3aa8ef707130f/algo.py).
  [Training evidence](evaluation_results/cycle-419-train.json).
  [Narrative](full_log.md#cycle-419--dummy-gated-mis-floor-018-after-20-on-n_atoms30).
- Cycle 418; candidate `3803dc4d8bf28d4dc4d6ff39611161dfb7ea5a1f`; champion `d2f7137d0609977f7ebd973704e68b8f419f8dad`; decision discard.
  [Implementation](ideas/sella-medium-delta018/3803dc4d8bf28d4dc4d6ff39611161dfb7ea5a1f/algo.py).
  [Training evidence](evaluation_results/cycle-418-train.json).
  [Narrative](full_log.md#cycle-418--connected-mis-floor-018-after-20-on-n_atoms30).

Status: closed

**Hypothesis:** A 1–2 cap skips all-three CF3 F–C–C. Applying two of three isolated CF3 F–C–C (no 2-coord C, siloxane, or aryl-F) cheapens 440723624 without perfluoro/siloxane extras.

**Outcome and uncertainty:** Cycle 372 discard, Δ=−1.33e-4. Only upadacitinib 16→17 extra; 440723624 stays 23. Soft F–C–C extras alkyl-CF3. Cycle 373 aryl-CF3 ipso discard, Δ=−9.69e-5; only 440723624 23→24 extra. Soft CF3 H0 overshoots the leftover.

**Reason to revisit:** Not as an F–C–C or ipso H0 scale at CF3. Do not interpolate 0.13/0.21.

**Next experiment:** Close this family.

**Attempts:**
- Cycle 373; candidate `1c8dca790d2d00e6bb7ed556c8df35ef2043ee7f`; champion `7140449e9cbddb6f6589ec34fdac86f88dc413e1`; decision discard.
  [Implementation](ideas/sella-isolated-cf3-fcc-h0/1c8dca790d2d00e6bb7ed556c8df35ef2043ee7f/algo.py).
  [Training evidence](evaluation_results/cycle-373-train.json).
  [Narrative](full_log.md#cycle-373--010-ha-aryl-cf3-ipso-angles-at-3-coord-carbon).
- Cycle 372; candidate `264372bdb8ab7be1cb2a60ad1fefd3bb29b682a1`; champion `7140449e9cbddb6f6589ec34fdac86f88dc413e1`; decision discard.
  [Implementation](ideas/sella-isolated-cf3-fcc-h0/264372bdb8ab7be1cb2a60ad1fefd3bb29b682a1/algo.py).
  [Training evidence](evaluation_results/cycle-372-train.json).
  [Narrative](full_log.md#cycle-372--isolated-cf3-fcc-010-ha-with-12-cap).

## sella-ocbr-h0: 0.10 Ha 3-coordinate ether O–C–Br / C–C–Br / C–O–C on connected 30≤n<80
Status: incorporated

**Hypothesis:** Leftover 252653184 unused bromo-ether angles are the limiter after the cycle 376 C2N3 extra.

**Outcome and uncertainty:** Carbon-center O–C–Br / C–C–Br / O–C–C all twitch (405–407). Cycle 408 C–O–C train leftover 16→15, valid extra 135192963 32→34. Cycle 409/433 Br-only: train-only. Cycle 434 **keep**: hetero/halo 3-coord ether. Cycle 436 **keep**: isolated 3-coord sulfide C–S–C skip hetero-ether. Train Δ=+2.27e-4 (252644426, brexpiprazole). Valid Δ=+1.19e-4 (103952397). New champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`. Leftover 194689238 unchanged.

**Reason to revisit:** Incorporated as `_hetero_coc` and `_isolated_csc`. Do not drop the exactly-one hetero gate, the no-S skip, or the isolated C/H-only gate. Cycle 437 mixed 3/4-coord C–S–C was a train-invisible no-op.

**Next experiment:** Close mixed C–S–C and hetero C–S–C H0. Window 2-coord N {C,C} isocyanide dummies at 0.20 Ha.

**Attempts:**
- Cycle 439; candidate `ba7c1a90a690d85b59e3f6c08547b533b0779e05`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision discard.
  [Implementation](ideas/sella-hetero-csc-h0/ba7c1a90a690d85b59e3f6c08547b533b0779e05/algo.py).
  [Training evidence](evaluation_results/cycle-439-train.json).
  [Narrative](full_log.md#cycle-439--010-ha-heterohalo-3-coord-csc-skip-p-and-nitrile).
- Cycle 437; candidate `d9e7c6040677e71032d6f0a61cbbcf059b6c6a3d`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision discard.
  [Training evidence](evaluation_results/cycle-437-train.json).
  [Narrative](full_log.md#cycle-437--isolated-mixed-34-coord-csc-on-diaryl-sulfide-molecules).
- Cycle 436; candidate `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; champion `d5533137bc7db63ba58f1dd2cef40098ae31e66a`; decision keep.
  [Training evidence](evaluation_results/cycle-436-train.json).
  [Validation evidence](evaluation_results/cycle-436-valid.json).
  [Narrative](full_log.md#cycle-436--010-ha-isolated-3-coord-sulfide-csc-skip-hetero-ether).
- Cycle 434; candidate `d5533137bc7db63ba58f1dd2cef40098ae31e66a`; champion `544a2a450f4a9385c365598b8b1148144d4fb1d4`; decision keep.
  [Training evidence](evaluation_results/cycle-434-train.json).
  [Validation evidence](evaluation_results/cycle-434-valid.json).
  [Narrative](full_log.md#cycle-434--010-ha-heterohalo-3-coord-ether-coc).
- Cycle 433; candidate `72b04b7b637b0fec65bf0ce8c4626e0bb8372653`; champion `544a2a450f4a9385c365598b8b1148144d4fb1d4`; decision non_generalizable.
  [Implementation](ideas/sella-ocbr-h0/72b04b7b637b0fec65bf0ce8c4626e0bb8372653/algo.py).
  [Training evidence](evaluation_results/cycle-433-train.json).
  [Validation evidence](evaluation_results/cycle-433-valid.json).
  [Narrative](full_log.md#cycle-433--010-ha-3-coord-bromo-ether-coc-stacked-on-432).
- Cycle 409; candidate `49eab6efff95371bbfe36e74f7cfb144ab9bf1a4`; champion `8e55a3dd9c0982bebda9429fb00fea1ae261d650`; decision non_generalizable.
  [Implementation](ideas/sella-ocbr-h0/49eab6efff95371bbfe36e74f7cfb144ab9bf1a4/algo.py).
  [Training evidence](evaluation_results/cycle-409-train.json).
  [Validation evidence](evaluation_results/cycle-409-valid.json).
  [Narrative](full_log.md#cycle-409--3-coordinate-br-carbon-on-bromo-ether-coc-010-ha).
- Cycle 408; candidate `40e8a7db8be60ff56755881afcf63f3d349ecc1b`; champion `8e55a3dd9c0982bebda9429fb00fea1ae261d650`; decision non_generalizable.
  [Implementation](ideas/sella-ocbr-h0/40e8a7db8be60ff56755881afcf63f3d349ecc1b/algo.py).
  [Training evidence](evaluation_results/cycle-408-train.json).
  [Validation evidence](evaluation_results/cycle-408-valid.json).
  [Narrative](full_log.md#cycle-408--010-ha-bromo-ether-coc-on-connected-30n80).
- Cycle 407; candidate `31c88a627591cce6e93eb8676c0a6ebfb6e18aee`; champion `8e55a3dd9c0982bebda9429fb00fea1ae261d650`; decision discard.
  [Implementation](ideas/sella-ocbr-h0/31c88a627591cce6e93eb8676c0a6ebfb6e18aee/algo.py).
  [Training evidence](evaluation_results/cycle-407-train.json).
  [Narrative](full_log.md#cycle-407--010-ha-3-coordinate-bromo-ether-occ-on-connected-30n80).
- Cycle 406; candidate `5a52c8a2791200cd2df88af1682c21e3a011566f`; champion `8e55a3dd9c0982bebda9429fb00fea1ae261d650`; decision discard.
  [Implementation](ideas/sella-ocbr-h0/5a52c8a2791200cd2df88af1682c21e3a011566f/algo.py).
  [Training evidence](evaluation_results/cycle-406-train.json).
  [Narrative](full_log.md#cycle-406--010-ha-3-coordinate-ether-ccbr-on-connected-30n80).
- Cycle 405; candidate `4211a73943a553bf898cc7caebc7c19585ad8b7b`; champion `8e55a3dd9c0982bebda9429fb00fea1ae261d650`; decision discard.
  [Implementation](ideas/sella-ocbr-h0/4211a73943a553bf898cc7caebc7c19585ad8b7b/algo.py).
  [Training evidence](evaluation_results/cycle-405-train.json).
  [Narrative](full_log.md#cycle-405--010-ha-3-coordinate-ether-ocbr-on-connected-30n80).

## sella-medium-sisi-h0: 0.10 Ha oligosilane Si–Si–Si / H–Si–H on connected 12≤n<30
Status: closed

**Hypothesis:** Leftover 135095297 pentasilane unused Si–Si–Si is the dummy-gap limiter.

**Outcome and uncertainty:** Cycle 410 Si–Si–Si discard, Δ=0, leftover twitch. Cycle 411 H–Si–H discard, Δ=0, leftover twitch. Cycle 412 H–Si–Si discard, Δ=0, leftover twitch. Not silane-angle limited.

**Reason to revisit:** Not as oligosilane angle H0 on 12–30. Do not interpolate 0.13.

**Next experiment:** Close this family.

**Attempts:**
- Cycle 412; candidate `57d6d3dd18c13d598ed9a6e473eebb8a3a9f731a`; champion `8e55a3dd9c0982bebda9429fb00fea1ae261d650`; decision discard.
  [Implementation](ideas/sella-medium-sisi-h0/57d6d3dd18c13d598ed9a6e473eebb8a3a9f731a/algo.py).
  [Training evidence](evaluation_results/cycle-412-train.json).
  [Narrative](full_log.md#cycle-412--010-ha-4-coordinate-hsisi-on-connected-12n30).
- Cycle 411; candidate `56c761e1ab50d920c766107dc1bdf35bfd21c23f`; champion `8e55a3dd9c0982bebda9429fb00fea1ae261d650`; decision discard.
  [Implementation](ideas/sella-medium-sisi-h0/56c761e1ab50d920c766107dc1bdf35bfd21c23f/algo.py).
  [Training evidence](evaluation_results/cycle-411-train.json).
  [Narrative](full_log.md#cycle-411--010-ha-4-coordinate-hsih-on-connected-12n30).
- Cycle 410; candidate `48a294fb3458bb3cd5a6e45054aeff7a603e935c`; champion `8e55a3dd9c0982bebda9429fb00fea1ae261d650`; decision discard.

**Hypothesis:** Leftover 135095297 pentasilane unused Si–Si–Si is the dummy-gap limiter.

**Outcome and uncertainty:** Cycle 410 discard, Δ=0. Leftover Si–Si–Si twitch at 37.

**Reason to revisit:** Complementary H–Si–H (9 angles, unique on train 12–30).

**Next experiment:** Cycle 411: 0.10 Ha H–Si–H, 1–9 cap, 12≤n<30.

**Attempts:**
- Cycle 410; candidate `48a294fb3458bb3cd5a6e45054aeff7a603e935c`; champion `8e55a3dd9c0982bebda9429fb00fea1ae261d650`; decision discard.
  [Implementation](ideas/sella-medium-sisi-h0/48a294fb3458bb3cd5a6e45054aeff7a603e935c/algo.py).
  [Training evidence](evaluation_results/cycle-410-train.json).
  [Narrative](full_log.md#cycle-410--010-ha-4-coordinate-sisis-on-connected-12n30).

## sella-medium-thiosulfonate-ccs-h0: 0.10 Ha thiosulfonate C–C–S with 4-coordinate N on 12≤n<30
Status: closed

**Hypothesis:** Leftover 433943527 unused C–C–S at the disulfide carbon is the limiter after CSS/O–S–S/C–S–O. A 4-coordinate N gate drops valid 135239086.

**Outcome and uncertainty:** Cycle 413 discard, Δ=0. Leftover energy twitch at 21. Not C–C–S limited.

**Reason to revisit:** Not as thiosulfonate C–C–S H0. Do not interpolate 0.13.

**Next experiment:** Close this family.

**Attempts:**
- Cycle 413; candidate `79b2af292012339314c1133fbccfe085eb4073fb`; champion `8e55a3dd9c0982bebda9429fb00fea1ae261d650`; decision discard.
  [Implementation](ideas/sella-medium-thiosulfonate-ccs-h0/79b2af292012339314c1133fbccfe085eb4073fb/algo.py).
  [Training evidence](evaluation_results/cycle-413-train.json).
  [Narrative](full_log.md#cycle-413--010-ha-thiosulfonate-ccs-with-4-coordinate-n-on-12n30).

## sella-oxo-brcbr-h0: 0.10 Ha 4-coordinate Br–C–Br on connected n<12
Status: closed

**Hypothesis:** Unused gem-dibromo Br–C–Br on leftover 135041973 is the limiter after O–S–C extraed it. Unique on train n<12.

**Outcome and uncertainty:** Cycle 414 discard, Δ=−1.78e-4. Only leftover 13→14 extra. Same overshoot as O–S–C. Energy-safe.

**Reason to revisit:** Not as Br–C–Br H0. Do not interpolate 0.13 or try S–C–Br (likely the same extra).

**Next experiment:** Close leftover 135041973 angle H0.

**Attempts:**
- Cycle 414; candidate `1b9382c97825dec29ce21667521d0618614f360a`; champion `8e55a3dd9c0982bebda9429fb00fea1ae261d650`; decision discard.
  [Implementation](ideas/sella-oxo-brcbr-h0/1b9382c97825dec29ce21667521d0618614f360a/algo.py).
  [Training evidence](evaluation_results/cycle-414-train.json).
  [Narrative](full_log.md#cycle-414--010-ha-4-coordinate-brcbr-on-connected-n12).

## sella-hessian-projector: Sella 2.6.0 pivoting-QR guess-Hessian projector
Status: closed

**Hypothesis:** Non-pivoting economic QR leaks near-null dummy-linear modes into H0. Rank-truncated pivoting QR (Sella 2.6.0 PR #81) should cheapen leftover dummy-linear jobs.

**Outcome and uncertainty:** Cycle 427 invalid. R=0.999306, max +2.78 kcal. Eight DES hops (esters–monoatomics 36→11 / R=0.752). Paliperidone 78→100. Connected leftover 252618428 35→25 with R=0.99979.

**Reason to revisit:** Not as a global H0 projector. Connected-only still energy-misses 252618428. Do not enable `exact_geodesic` as a follow-up.

**Next experiment:** Exact geodesic on connected 18≤n<22 (cycle 429 repair).

**Attempts:**
- Cycle 427; candidate `185a97369ff1d5b34d09dffb0ddbfd97f858e967`; champion `d2f7137d0609977f7ebd973704e68b8f419f8dad`; decision invalid.
  [Implementation](ideas/sella-hessian-projector/185a97369ff1d5b34d09dffb0ddbfd97f858e967/algo.py).
  [Training evidence](evaluation_results/cycle-427-train.json).
  [Narrative](full_log.md#cycle-427--pivoting-qr-guess-hessian-projector).

## sella-exact-geodesic: Sella 2.6.0 exact geodesic Binv on connected molecules
Status: incorporated

**Hypothesis:** Recomputing Binv at every geodesic ODE RHS (Sella 2.6.0 `exact_geodesic`) cheapens dummy-gap leftover 135043047 without hopping dimers.

**Outcome and uncertainty:** Cycle 429 discard, Δ=−1.73e-3. Cycle 430 non_generalizable (valid n=20 extras). Cycle 431 non_generalizable (valid Δ=+4.30e-5). Cycle 432 **keep** stacked with C–S–N: train Δ=+7.39e-4, valid Δ=+1.04e-4. New champion `544a2a450f4a9385c365598b8b1148144d4fb1d4`.

**Reason to revisit:** Incorporated as 18≤n<20 connected exact geodesic. Do not widen to n=20. Do not widen to n<18 (cycle 461 extras dummy-gap leftovers and 163350562).

**Next experiment:** Close n<18 geodesic.

**Attempts:**
- Cycle 461; candidate `5f3f71464bb0e21f46b6c820f88df67584fbd280`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision discard.
  [Implementation](ideas/sella-exact-geodesic/5f3f71464bb0e21f46b6c820f88df67584fbd280/algo.py).
  [Training evidence](evaluation_results/cycle-461-train.json).
  [Narrative](full_log.md#cycle-461--exact-geodesic-binv-on-connected-n_atoms20).
- Cycle 432; candidate `544a2a450f4a9385c365598b8b1148144d4fb1d4`; champion `d2f7137d0609977f7ebd973704e68b8f419f8dad`; decision keep.
  [Training evidence](evaluation_results/cycle-432-train.json).
  [Validation evidence](evaluation_results/cycle-432-valid.json).
  [Narrative](full_log.md#cycle-432--exact-geodesic-18-20-plus-sulfonamide-csn-010-ha).
- Cycle 431; candidate `93703e120a16201313fd188140bca00448a696de`; champion `d2f7137d0609977f7ebd973704e68b8f419f8dad`; decision non_generalizable.
  [Implementation](ideas/sella-exact-geodesic/93703e120a16201313fd188140bca00448a696de/algo.py).
  [Training evidence](evaluation_results/cycle-431-train.json).
  [Validation evidence](evaluation_results/cycle-431-valid.json).
  [Narrative](full_log.md#cycle-431--exact-geodesic-binv-on-connected-18n_atoms20).
- Cycle 430; candidate `92ae92dfb849df07cd388c0f148a204be778cc92`; champion `d2f7137d0609977f7ebd973704e68b8f419f8dad`; decision non_generalizable.
  [Implementation](ideas/sella-exact-geodesic/92ae92dfb849df07cd388c0f148a204be778cc92/algo.py).
  [Training evidence](evaluation_results/cycle-430-train.json).
  [Validation evidence](evaluation_results/cycle-430-valid.json).
  [Narrative](full_log.md#cycle-430--exact-geodesic-binv-on-connected-18n_atoms22).
- Cycle 429; candidate `d898fffad044c46c578abd8968a44dcd4c80cbcd`; champion `d2f7137d0609977f7ebd973704e68b8f419f8dad`; decision discard.
  [Implementation](ideas/sella-exact-geodesic/d898fffad044c46c578abd8968a44dcd4c80cbcd/algo.py).
  [Training evidence](evaluation_results/cycle-429-train.json).
  [Narrative](full_log.md#cycle-429--exact-geodesic-binv-on-connected-molecules).

## sella-restricted-step-isfinite: Sella 2.6 isfinite Newton α in restricted-step
Status: closed

**Hypothesis:** Dummy-linear leftover trust solves produce non-finite Newton α; Sella 2.6 bisects those instead of following inf.

**Outcome and uncertainty:** Cycle 435 non_generalizable. Train Δ=+1.93e-4 from dimer packing (amines–amines −7) not dummy leftovers. Valid Δ=−4.87e-4 (alkenes–water +16, alkenes–alkenes 197→200).

**Reason to revisit:** Not as a global isfinite gate. Connected-only drops the train δ save; dimer-only keeps the valid extras.

**Next experiment:** Close this family.

**Attempts:**
- Cycle 435; candidate `de3f00be1f6f508e2301e66536faab20c7b04083`; champion `d5533137bc7db63ba58f1dd2cef40098ae31e66a`; decision non_generalizable.
  [Implementation](ideas/sella-restricted-step-isfinite/de3f00be1f6f508e2301e66536faab20c7b04083/algo.py).
  [Training evidence](evaluation_results/cycle-435-train.json).
  [Validation evidence](evaluation_results/cycle-435-valid.json).
  [Narrative](full_log.md#cycle-435--sella-26-restricted-step-newton-α-hardening).

## sella-svd-rank-relative: Sella 2.6 relative SVD rank on Jacobian fallbacks
Status: closed

**Hypothesis:** Absolute `Si > 1e-6` SVD truncation mismatches the relative R-diagonal rank test and leaks dummy-linear modes.

**Outcome and uncertainty:** Cycle 441 discard, Δ=0, bit-identical including energies. Unused or rank-identical to the absolute cutoff.

**Reason to revisit:** Not as a relative SVD rank swap. Do not stack with pivoting QR H0.

**Next experiment:** Close this family.

**Attempts:**
- Cycle 441; candidate `5632cdb16470a0394cda02d2cb8d9ddce34abb04`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision discard.
  [Training evidence](evaluation_results/cycle-441-train.json).
  [Narrative](full_log.md#cycle-441--sella-26-relative-svd-rank-on-jacobian-fallbacks).

## sella-dummy-cone-projection: Sella 2.6 linear-bend dummy cone projection after set_x
Status: closed

**Hypothesis:** Dummy-linear leftover extra calls come from Newton IC dummy projection coupling into the free improper. Sella 2.6 cone projection should restore dummy constraints without that coupling.

**Outcome and uncertainty:** Cycle 438 discard, Δ=−1.51e-3. Dummy leftovers unchanged. Extras are n≤26 probe-connected DES (ethers–ethers 93→114). Energy-safe.

**Reason to revisit:** Not as a cone projection. The leftover is not dummy-constraint-projection limited. Do not gate n≥30 or port `get_scons` skip / fast dummy selection.

**Next experiment:** Close this family.

**Attempts:**
- Cycle 438; candidate `ebaed0c489109170f67a2660c6e917c06e56de0e`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision discard.
  [Implementation](ideas/sella-dummy-cone-projection/ebaed0c489109170f67a2660c6e917c06e56de0e/algo.py).
  [Training evidence](evaluation_results/cycle-438-train.json).
  [Narrative](full_log.md#cycle-438--sella-26-linear-bend-dummy-cone-projection).

## sella-ester-coc-h0: 0.10 Ha ester C–O–C on dimers with two carboxyl carbons
Status: closed

**Hypothesis:** Leftover carboxylates–esters is limited by ester C–O–C. Soft 0.10 Ha on 1–2 such angles when the dimer has ≥2 carboxyl/ester carbons cheapens it without single-ester packing jobs.

**Outcome and uncertainty:** Cycle 448 discard, Δ=−1.09e-4. Leftover n_steps unchanged 75 (energy twitch). Extra esters–esters 39→41.

**Reason to revisit:** Not as two-carboxyl ester C–O–C. Leftover is not that limiter; diester extras.

**Next experiment:** Close this family.

**Attempts:**
- Cycle 448; candidate `654d3d07834963b466b31cdc8a574ef4770d508d`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision discard.
  [Implementation](ideas/sella-ester-coc-h0/654d3d07834963b466b31cdc8a574ef4770d508d/algo.py).
  [Training evidence](evaluation_results/cycle-448-train.json).
  [Narrative](full_log.md#cycle-448--010-ha-ester-coc-on-dimers-with-two-carboxyl-carbons).

## sella-sulfoxide-osc-h0: 0.10 Ha sulfoxide O–S–C on connected n_atoms<18
Status: closed

**Hypothesis:** Leftover 135041973 is a CBr3 sulfoxide. Soft 0.10 Ha on 1–2 3-coordinate O–S–C at n<18 cheapens it without n≥18 O–S–O extras.

**Outcome and uncertainty:** Cycle 449 discard, Δ=−3.42e-4. Leftover 13→14 extra; 378166034 13→14 extra.

**Reason to revisit:** Not as 0.10 Ha O–S–C. Soft sulfoxide H0 overshoots.

**Next experiment:** Close. Do not interpolate 0.13.

**Attempts:**
- Cycle 449; candidate `d6f0d8b78319970ceb3bb6047494e0988a58af97`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision discard.
  [Implementation](ideas/sella-sulfoxide-osc-h0/d6f0d8b78319970ceb3bb6047494e0988a58af97/algo.py).
  [Training evidence](evaluation_results/cycle-449-train.json).
  [Narrative](full_log.md#cycle-449--010-ha-sulfoxide-osc-on-connected-n_atoms18).

## sella-noxide-cnc-h0: 0.10 Ha N-oxide C–N–C on connected n<18 or n≥30
Status: closed

**Hypothesis:** Leftover 135092053 is an N-oxide C–N–C limiter. Soft 0.10 Ha in the dummy-angle window cheapens it and valid 252656292.

**Outcome and uncertainty:** Cycle 450 discard, Δ=0. n_steps bit-identical; leftover energy twitch only.

**Reason to revisit:** Not as N-oxide C–N–C H0. The leftover is not that n_steps limiter.

**Next experiment:** Close. Dummy-linear n<18 limiter next.

**Attempts:**
- Cycle 450; candidate `8ad55ca7fb3b392e9e5e7612634d3b8952dabc29`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision discard.
  [Implementation](ideas/sella-noxide-cnc-h0/8ad55ca7fb3b392e9e5e7612634d3b8952dabc29/algo.py).
  [Training evidence](evaluation_results/cycle-450-train.json).
  [Narrative](full_log.md#cycle-450--010-ha-n-oxide-cnc-on-connected-n18-or-n30).

## sella-medium-delta018: connected MIS floor 0.18 after 20 on n<18
Status: closed

**Hypothesis:** Cycle 418 extraed perfluoro n=27. n<18 0.18 floor enlarges leftover 252089162 / 135092053 dummy-linear tails.

**Outcome and uncertainty:** Cycle 451 discard, Δ=0, bit-identical. Cycle 452 wd=0.7 n<18 discard (save 135078057 n=5, extra leftover 252089162 n=11). Cycle 453 n<8 non_generalizable (train −1 only, valid bit-identical).

**Reason to revisit:** Not as n<18/n<8 wd=0.7 or 0.18 floor. Do not widen n<12.

**Next experiment:** Close n-gated dummy-wd 0.7.

**Attempts:**
- Cycle 451; candidate `386e8f989f6d84a50fc19aa52a6db9a31df67470`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision discard.
  [Training evidence](evaluation_results/cycle-451-train.json).
  [Narrative](full_log.md#cycle-451--connected-mis-floor-018-after-20-on-n_atoms18).
- Cycle 452; candidate `b127694e9b49a190443922033fdcf4b511fc73e4`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision discard.
  [Implementation](ideas/sella-dummy-wd07-n18/b127694e9b49a190443922033fdcf4b511fc73e4/algo.py).
  [Training evidence](evaluation_results/cycle-452-train.json).
  [Narrative](full_log.md#cycle-452--dummy-dihedral-limiter-wd07-on-connected-n_atoms18).
- Cycle 453; candidate `584d8a459788dd0ead3cf7964c32d0e282381488`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision non_generalizable.
  [Implementation](ideas/sella-dummy-wd07-n18/584d8a459788dd0ead3cf7964c32d0e282381488/algo.py).
  [Training evidence](evaluation_results/cycle-453-train.json).
  [Validation evidence](evaluation_results/cycle-453-valid.json).
  [Narrative](full_log.md#cycle-453--dummy-dihedral-limiter-wd07-on-connected-n_atoms8).

## sella-pyridine-real-degree: pyridine C–N–C uses dummy-filtered neighbors
Status: closed

**Hypothesis:** `nbonds[N]==2` drops dummy-bonded pyridine N from the 0.10 Ha keep.

**Outcome and uncertainty:** Cycle 454 discard, Δ=0, bit-identical. Dummy-on-pyridine-N never fires on train.

**Reason to revisit:** Not as a dummy-filtered pyridine degree swap.

**Next experiment:** Close.

**Attempts:**
- Cycle 454; candidate `20af3e1384e4c671f3dc3b28311a33ae325266c8`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision discard.
  [Training evidence](evaluation_results/cycle-454-train.json).
  [Narrative](full_log.md#cycle-454--pyridine-cnc-uses-real-neighbors-not-nbonds).

## sella-connected-rfo40: Banerjee RFO after 40 steps on connected 30≤n<80
Status: closed

**Hypothesis:** Leftover 363892164 (n=43) never saw cycle 140 RFO@50. RFO@40 on 30–80 spare paliperidone/venetoclax.

**Outcome and uncertainty:** Cycle 455 discard, Δ=−6.27e-5. Leftover unchanged. Extra 384221749 58→60. Energy twitches only on 160853090.

**Reason to revisit:** Not as RFO@40/45/50 on 30–80. Leftover is not RFO-limited.

**Next experiment:** Close. Skip near-null QN modes instead.

**Attempts:**
- Cycle 455; candidate `9d0c9a6f98848ec9debac6487b79469cdfde69d0`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision discard.
  [Implementation](ideas/sella-connected-rfo40/9d0c9a6f98848ec9debac6487b79469cdfde69d0/algo.py).
  [Training evidence](evaluation_results/cycle-455-train.json).
  [Narrative](full_log.md#cycle-455--banerjee-rfo-after-40-steps-on-connected-30n80).

## sella-qn-tiny-eig-skip: skip near-null QN Hessian modes on connected 30≤n<80
Status: closed

**Hypothesis:** Dummy-linear leftover extra calls are MIS-truncated near-null TS-BFGS modes. Skipping them (pysisyphus/geomeTRIC) is the opposite of cycle 131’s |λ| floor.

**Outcome and uncertainty:** Cycle 456 discard, Δ=0, bit-identical. 1e-8 never fires in internals (Cartesian trans/rot scale). Cycle 457 discard, Δ=0, bit-identical. 1e-5 never fires either.

**Reason to revisit:** Not as a skip at 1e-8/1e-5/1e-4. Cycle 458 v0 shift at 0.001 extraed 160853090; leftover unchanged. Cycle 465 skip-neg-eig invalid (rivaroxaban SCF hop, 78 stalls). Close Hessian-spectrum / skip-neg gates on 30–80.

**Next experiment:** Close. Do not interpolate milder skip-neg.

**Attempts:**
- Cycle 456; candidate `3b06793d57c0c2d89eb605f331a07ecfbfd6b9ca`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision discard.
  [Training evidence](evaluation_results/cycle-456-train.json).
  [Narrative](full_log.md#cycle-456--skip-λ1e-8-qn-modes-on-connected-30n80).
- Cycle 457; candidate `f68db4e0e471316442d9bb47c6e0d9792d84700b`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision discard.
  [Training evidence](evaluation_results/cycle-457-train.json).
  [Narrative](full_log.md#cycle-457--skip-λ1e-5-qn-modes-on-connected-30n80).
- Cycle 458; candidate `3974d7bd4e7961050e67595d2d6af682900e608a`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision discard.
  [Implementation](ideas/sella-geometric-v0-shift/3974d7bd4e7961050e67595d2d6af682900e608a/algo.py).
  [Training evidence](evaluation_results/cycle-458-train.json).
  [Narrative](full_log.md#cycle-458--geometric-v0-hessian-shift-ε0001-on-connected-30n80).
- Cycle 462; candidate `27988e7590d35113a791a631e92acf2797b9bac9`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision discard.
  [Training evidence](evaluation_results/cycle-462-train.json).
  [Narrative](full_log.md#cycle-462--dummy-dihedral-limiter-wd07-on-connected-30n80).
- Cycle 465; candidate `97d6e51015f1706c8461f1228fb5cfc5157d1652`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision invalid.
  [Implementation](ideas/sella-skip-neg-eig-update/97d6e51015f1706c8461f1228fb5cfc5157d1652/algo.py).
  [Training evidence](evaluation_results/cycle-465-train.json).
  [Narrative](full_log.md#cycle-465--skip-ts-bfgs-updates-that-introduce-negative-eigenvalues-on-connected-30n80).

## sella-medium-ssc-h0: 0.10 Ha thiosulfonate S–S–C / O–S–O on connected 12–30
Status: closed

**Hypothesis:** Leftover 433943527 unused 4-coordinate S–S–C then O–S–O after CSS/O–S–S/C–S–O/C–C–S.

**Outcome and uncertainty:** Cycle 463 discard, Δ=0, leftover S–S–C energy twitch. Cycle 464 discard, Δ=0, leftover O–S–O energy twitch.

**Reason to revisit:** Not as thiosulfonate angle H0 on 12–30.

**Next experiment:** Close this family.

**Attempts:**
- Cycle 463; candidate `d2ac9a3e7b1ab74850acd990507a352ca590a924`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision discard.
  [Implementation](ideas/sella-medium-ssc-h0/d2ac9a3e7b1ab74850acd990507a352ca590a924/algo.py).
  [Training evidence](evaluation_results/cycle-463-train.json).
  [Narrative](full_log.md#cycle-463--010-ha-4-coordinate-thiosulfonate-ssc-on-connected-12n30).
- Cycle 464; candidate `0c73ad8565447411451fdb00011f3a8e78d1229a`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision discard.
  [Implementation](ideas/sella-medium-ssc-h0/0c73ad8565447411451fdb00011f3a8e78d1229a/algo.py).
  [Training evidence](evaluation_results/cycle-464-train.json).
  [Narrative](full_log.md#cycle-464--010-ha-thiosulfonate-oso-on-connected-12n30).

## sella-n18-alcohol-occ-h0: 0.10 Ha 4-coordinate alcohol O–C–C on connected n<18
Status: closed

**Hypothesis:** Valid leftover 135169446 unused alcohol O–C–C is the limiter. Cycle 401 extraed dimer/phenol on 12–30; n<18 still extraed monoatomics–phenol n=14.

**Outcome and uncertainty:** Cycle 459 discard, Δ=−4.10e-5. Same extra as 401, n=14. Cycle 460 discard, Δ=0, bit-identical. Allylic n<18 never fires on train.

**Reason to revisit:** Not as alcohol O–C–C on n<18. Valid leftover is train-invisible.

**Next experiment:** Close. Do not interpolate isolated-alcohol gates.

**Attempts:**
- Cycle 459; candidate `1f569cbf36cad634f412ee3440a519acf9bb4f5f`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision discard.
  [Implementation](ideas/sella-medium-alcohol-occ-h0/1f569cbf36cad634f412ee3440a519acf9bb4f5f/algo.py).
  [Training evidence](evaluation_results/cycle-459-train.json).
  [Narrative](full_log.md#cycle-459--010-ha-4-coordinate-alcohol-occ-on-connected-n18).
- Cycle 460; candidate `b2838e5f848401523936393074cde10b83b93d6e`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision discard.
  [Training evidence](evaluation_results/cycle-460-train.json).
  [Narrative](full_log.md#cycle-460--allylic-3-coord-terminal-on-alcohol-occ-n18).

## sella-skip-neg-eig-update: skip TS-BFGS updates that introduce negative eigenvalues
Status: closed

**Hypothesis:** geomeTRIC skip-bad-update on connected 30≤n<80 keeps a PD TS-BFGS model for dummy-linear leftover without cycle 186’s s·y<0 skip.

**Outcome and uncertainty:** Cycle 465 invalid. rivaroxaban xTB SCF hop; 78 force-call stalls including 160853090 and 384221749.

**Reason to revisit:** Not as skip-neg on 30–80. Do not interpolate a milder skip (the hop is a basin change).

**Next experiment:** Close. Leftover ketones–phenol aldehyde O–C–C on connected 18≤n<30.

**Attempts:**
- Cycle 465; candidate `97d6e51015f1706c8461f1228fb5cfc5157d1652`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision invalid.
  [Implementation](ideas/sella-skip-neg-eig-update/97d6e51015f1706c8461f1228fb5cfc5157d1652/algo.py).
  [Training evidence](evaluation_results/cycle-465-train.json).
  [Narrative](full_log.md#cycle-465--skip-ts-bfgs-updates-that-introduce-negative-eigenvalues-on-connected-30n80).

## sella-medium-aldehyde-occ-h0: 0.10 Ha aldehyde O–C–C on connected 18–30
Status: incorporated

**Hypothesis:** Leftover ketones–phenol is connected-classified acetaldehyde plus phenol; unused aldehyde O–C–C is the limiter after phenol H0 failed.

**Outcome and uncertainty:** Cycle 466 discard, Δ=+4.96e-5. Only leftover 46→45. Cycle 467/468 n_steps-identical interpolations. Cycle 469 non_generalizable: train leftover 46→43, Δ=+1.49e-4; valid bit-identical (analogs are n=42/48). Cycle 470 non_generalizable: train identical to 469; valid extra 104001731 32→33.

**Reason to revisit:** Cycle 469 is a train-only leftover keep because the aldehyde gate misses valid esters–phenol. Do not extend the phenol partner to 30–80 (104001731 extra). Isolated aryl phenol (exactly one; ipso two 3-coord C) on 18–30 is the next test of that leftover without the aldehyde requirement.

**Next experiment:** Close 30–80 phenol partner. Try exactly-one aryl phenol C–O–H 0.10 on connected 18≤n<30.

**Attempts:**
- Cycle 466; candidate `fe1aaf8eed98062abaf439515ce90b86bf377e31`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision discard.
  [Implementation](ideas/sella-medium-aldehyde-occ-h0/fe1aaf8eed98062abaf439515ce90b86bf377e31/algo.py).
  [Training evidence](evaluation_results/cycle-466-train.json).
  [Narrative](full_log.md#cycle-466--010-ha-aldehyde-occ-on-connected-18n30).
- Cycle 467; candidate `d2f8b7f97a18b52b05d0c7981852205eb80b274b`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision discard.
  [Implementation](ideas/sella-medium-aldehyde-occ-h0/d2f8b7f97a18b52b05d0c7981852205eb80b274b/algo.py).
  [Training evidence](evaluation_results/cycle-467-train.json).
  [Narrative](full_log.md#cycle-467--010-ha-aldehyde-occ-and-och-on-connected-18n30).
- Cycle 468; candidate `9b4339c0c062e8e64bf16b7a194219ac1d1ddaf2`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision discard.
  [Implementation](ideas/sella-medium-aldehyde-occ-h0/9b4339c0c062e8e64bf16b7a194219ac1d1ddaf2/algo.py).
  [Training evidence](evaluation_results/cycle-468-train.json).
  [Narrative](full_log.md#cycle-468--008-ha-aldehyde-occ-on-connected-18n30).
- Cycle 469; candidate `7a9bf9353b1aa25921e5428373337e270f8569f5`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision non_generalizable.
  [Implementation](ideas/sella-medium-aldehyde-occ-h0/7a9bf9353b1aa25921e5428373337e270f8569f5/algo.py).
  [Training evidence](evaluation_results/cycle-469-train.json).
  [Validation evidence](evaluation_results/cycle-469-valid.json).
  [Narrative](full_log.md#cycle-469--aldehyde-occ-plus-phenol-coh-when-aldehyde-is-present-18n30).
- Cycle 470; candidate `8b673e7f609a592c76524680c3c4a147418255d5`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision non_generalizable.
  [Implementation](ideas/sella-medium-aldehyde-occ-h0/8b673e7f609a592c76524680c3c4a147418255d5/algo.py).
  [Training evidence](evaluation_results/cycle-470-train.json).
  [Validation evidence](evaluation_results/cycle-470-valid.json).
  [Narrative](full_log.md#cycle-470--aldehyde-partner-phenol-coh-also-on-connected-30n80).

## sella-medium-carbamate-coc-h0: 0.10 Ha mixed 3-/4-coord carbamate C–O–C on connected 30–80
Status: closed

**Hypothesis:** Leftover 363892164 unused 2-coordinate C–O–C is mixed 3-coord carbamate C {O,O,N} plus 4-coord C. Soft 0.10 Ha on 1–2 such angles at connected 30≤n<80 tags leftover and rivaroxaban without paliperidone palmitate esters.

**Outcome and uncertainty:** Cycle 471 discard, Δ=0, n_steps bit-identical. Energy twitches on leftover and rivaroxaban; paliperidone untagged. The class fires and is not a limiter.

**Reason to revisit:** Not as a mixed carbamate C–O–C H0 scale. Do not interpolate 0.08 or drop the 1–2 cap.

**Next experiment:** Close this family. See isolated aryl phenol C–O–H on connected 18–30.

**Attempts:**
- Cycle 471; candidate `eab26996d9f28f2b48fefdebeeb349701481947f`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision discard.
  [Implementation](ideas/sella-medium-carbamate-coc-h0/eab26996d9f28f2b48fefdebeeb349701481947f/algo.py).
  [Training evidence](evaluation_results/cycle-471-train.json).
  [Narrative](full_log.md#cycle-471--010-ha-mixed-34-coord-carbamate-coc-on-connected-30n80).

## sella-medium-aryl-phenol-coh: 0.10 Ha isolated aryl phenol C–O–H on connected 18–30
Status: incorporated

**Hypothesis:** Probe-connected carbonyl–phenol leftovers in the dummy-angle gap are limited by aryl phenol C–O–H. Exactly one such angle at connected 18≤n<30 cheapens train ketones–phenol and valid esters–phenol without diphenol 135095125 or 30–80 104001731.

**Outcome and uncertainty:** Cycle 472 discard, Δ=+9.92e-5, leftover 46→44. Cycle 473 non_generalizable: train 46→43, Δ=+1.49e-4; valid esters–phenol 36→35, Δ=+6.52e-5. Cycle 474/477 stacking extra H0 onto esters–phenol undoes the phenol save. Cycle 475 0.08 discard. Cycle 478 stacked 473 with Si–O–S: **keep**.

**Reason to revisit:** Incorporated in champion `1cf4b725273d95a1f17f2e367a57199cbe3af436` with aldehyde O–C–C and Si–O–S. Do not stack ester C–O–C or O–H–O. Do not interpolate 0.08.

**Next experiment:** Leave aryl phenol at 0.10 Ha on 18–30.

**Hypothesis:** Probe-connected carbonyl–phenol leftovers in the dummy-angle gap are limited by aryl phenol C–O–H. Exactly one such angle at connected 18≤n<30 cheapens train ketones–phenol and valid esters–phenol without diphenol 135095125 or 30–80 104001731.

**Outcome and uncertainty:** Cycle 472 discard, Δ=+9.92e-5, leftover 46→44. Cycle 473 non_generalizable: train 46→43, Δ=+1.49e-4; valid esters–phenol 36→35, Δ=+6.52e-5. Cycle 474 non_generalizable: train unchanged; valid esters–phenol 35→36, ester C–O–C undoes the phenol save. Cycle 475 discard, Δ=+9.92e-5, leftover 43→44 (n_steps-identical to 472).

**Reason to revisit:** Cycle 473 remains the best unused. Do not stack ester C–O–C, 0.08 aryl phenol, n<18 alkynol C–O–H, or probe O–H–O onto esters–phenol.

**Next experiment:** Cycle 473 plus 0.10 Ha Si–O–S on connected 30≤n<80. Support: leftover still ≤43; esters–phenol still −1; 135042855 ≤37; both Δ gates vs `b050fa3`. Contradict: hop, extras, or Δ fails.

**Attempts:**
- Cycle 472; candidate `69c6eeb6af6f9f836984ed47dd52c998a28bcf11`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision discard.
  [Implementation](ideas/sella-medium-aryl-phenol-coh/69c6eeb6af6f9f836984ed47dd52c998a28bcf11/algo.py).
  [Training evidence](evaluation_results/cycle-472-train.json).
  [Narrative](full_log.md#cycle-472--exactly-one-aryl-phenol-coh-on-connected-18n30).
- Cycle 473; candidate `8a10ef4e9b3a2850c0dd82a4dd3c98885241f9b2`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision non_generalizable.
  [Implementation](ideas/sella-medium-aryl-phenol-coh/8a10ef4e9b3a2850c0dd82a4dd3c98885241f9b2/algo.py).
  [Training evidence](evaluation_results/cycle-473-train.json).
  [Validation evidence](evaluation_results/cycle-473-valid.json).
  [Narrative](full_log.md#cycle-473--aryl-phenol-coh-plus-aldehyde-occ-on-connected-18n30).
- Cycle 474; candidate `1add0c6a1dcd6b8945092e56c5790e49f694781b`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision non_generalizable.
  [Implementation](ideas/sella-medium-aryl-phenol-coh/1add0c6a1dcd6b8945092e56c5790e49f694781b/algo.py).
  [Training evidence](evaluation_results/cycle-474-train.json).
  [Validation evidence](evaluation_results/cycle-474-valid.json).
  [Narrative](full_log.md#cycle-474--exactly-one-mixed-ester-coc-on-connected-18n30).
- Cycle 475; candidate `f92c5293e0332c9f78cc76cc40108fc41b3e2af5`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision discard.
  [Implementation](ideas/sella-medium-aryl-phenol-coh/f92c5293e0332c9f78cc76cc40108fc41b3e2af5/algo.py).
  [Training evidence](evaluation_results/cycle-475-train.json).
  [Narrative](full_log.md#cycle-475--008-ha-aryl-phenol-coh-plus-aldehyde-occ-on-connected-18n30).

## sella-n18-alkynol-coh-h0: 0.10 Ha alkyl C–O–H on connected n<18 alkynols

Status: closed

**Hypothesis:** Valid leftover 135169446 is limited by alkyl C–O–H in the presence of 2-coordinate C–C carbons. Stacking that isolated class on connected n<18 with cycle 473’s train keep can add the missing valid call.

**Outcome and uncertainty:** Cycle 476 non_generalizable. Train and valid n_steps-identical to 473. 135169446 stays 23 with an energy twitch: alkyl C–O–H fires and is not a limiter.

**Reason to revisit:** Not as an alkyl C–O–H H0 scale on n<18 alkynols. Do not interpolate 0.08.

**Next experiment:** Close. See probe O–H–O on connected 18–30 stacked with cycle 473.

**Attempts:**
- Cycle 476; candidate `eda926503510cc3a46028ce74c3e9a1483061f1f`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision non_generalizable.
  [Implementation](ideas/sella-n18-alkynol-coh-h0/eda926503510cc3a46028ce74c3e9a1483061f1f/algo.py).
  [Training evidence](evaluation_results/cycle-476-train.json).
  [Validation evidence](evaluation_results/cycle-476-valid.json).
  [Narrative](full_log.md#cycle-476--cycle-473-plus-n18-alkynol-alkyl-coh).

## sella-medium-oho-h0: 0.10 Ha probe O–H–O on connected 18–30

Status: closed

**Hypothesis:** Probe-connected carbonyl–phenol leftovers are limited by the connecting 2-coordinate O–H–O H-bond under Fischer H0. Exactly one such angle at connected 18≤n<30 cheapens both leftovers without intramolecular 104079126 (two O–H–O).

**Outcome and uncertainty:** Cycle 477 non_generalizable. Train leftover stays 43 with a twitch; 104079126 untagged. Valid esters–phenol 35→36 undoes the aryl-phenol save.

**Reason to revisit:** Not as an O–H–O H0 stacked with aryl phenol. Do not interpolate 0.08.

**Next experiment:** Close stacking H0 onto the same probe-connected leftover. See Si–O–S on connected 30–80 stacked with cycle 473.

**Attempts:**
- Cycle 477; candidate `0374a264724124d96155db904463f434fc796deb`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision non_generalizable.
  [Implementation](ideas/sella-medium-oho-h0/0374a264724124d96155db904463f434fc796deb/algo.py).
  [Training evidence](evaluation_results/cycle-477-train.json).
  [Validation evidence](evaluation_results/cycle-477-valid.json).
  [Narrative](full_log.md#cycle-477--cycle-473-plus-exactly-one-probe-oho-on-connected-18n30).

## sella-pyridine-sios-h0: 0.10 Ha Si–O–S on connected 30–80

Status: incorporated

**Hypothesis:** Valid leftover 135042855 is limited by 2-coordinate Si–O–S. Train 12–80 has none. Stacking with cycle 473 supplies the missing valid call without retagging esters–phenol.

**Outcome and uncertainty:** Cycle 478 **keep**. Train Δ=+1.49e-4 (identical to 473). Valid Δ=+2.40e-4: 135042855 38→35 plus esters–phenol 36→35. Energy-safe. New champion `1cf4b725273d95a1f17f2e367a57199cbe3af436`.

**Reason to revisit:** Incorporated. Do not interpolate 0.08. Do not widen to Si–O–Si (cycle 264).

**Next experiment:** Leave Si–O–S at 0.10 Ha on 30–80.

**Attempts:**
- Cycle 478; candidate `1cf4b725273d95a1f17f2e367a57199cbe3af436`; champion `b050fa33d7706a952d7b12aedcaaff5a2765a0f7`; decision keep.
  [Training evidence](evaluation_results/cycle-478-train.json).
  [Validation evidence](evaluation_results/cycle-478-valid.json).
  [Narrative](full_log.md#cycle-478--cycle-473-plus-sios-on-connected-30n80).

## sella-pyridine-ether-hco-h0: 0.10 Ha isolated ether H–C–O on connected 30–80

Status: closed

**Hypothesis:** Leftover 363892164 unused H–C–O at the 4-coordinate ether carbon is the limiter after oxazolidinone O–C–C. Exactly one such angle at connected 30≤n<80 cheapens leftover without empagliflozin.

**Outcome and uncertainty:** Cycle 479 discard, Δ=−5.20e-5. Leftover twitch-only; extra 404345632 32→33. Class fires and extras a previous save.

**Reason to revisit:** Not as an ether H–C–O H0 scale. Do not interpolate 0.08.

**Next experiment:** Close this family.

**Attempts:**
- Cycle 479; candidate `de3c8ce83ffcc6543df41af674524fe7dea07bfb`; champion `1cf4b725273d95a1f17f2e367a57199cbe3af436`; decision discard.
  [Implementation](ideas/sella-pyridine-ether-hco-h0/de3c8ce83ffcc6543df41af674524fe7dea07bfb/algo.py).
  [Training evidence](evaluation_results/cycle-479-train.json).
  [Narrative](full_log.md#cycle-479--exactly-one-ether-hco-on-connected-30n80).

## sella-pyridine-carbamate-cnc-h0: 0.10 Ha carbamate NH C–N–C on connected 30–80

Status: closed

**Hypothesis:** Tight carbamate NH C–N–C (3-coord N {C, C, H}; one C is 3-coord {O, O, N}) cheapens leftover 363892164 without cycle 325’s amide extras.

**Outcome and uncertainty:** Cycle 480 discard, Δ=0. Leftover twitch-only; no extras. Not a limiter.

**Reason to revisit:** Not as a carbamate C–N–C H0 scale. Do not interpolate 0.08.

**Next experiment:** Close leftover oxazolidinone intramolecular H0.

**Attempts:**
- Cycle 480; candidate `09ab9c112c33cbf1cdc1e93f1c1aa7caf0f836a2`; champion `1cf4b725273d95a1f17f2e367a57199cbe3af436`; decision discard.
  [Implementation](ideas/sella-pyridine-carbamate-cnc-h0/09ab9c112c33cbf1cdc1e93f1c1aa7caf0f836a2/algo.py).
  [Training evidence](evaluation_results/cycle-480-train.json).
  [Narrative](full_log.md#cycle-480--exactly-one-carbamate-nh-cnc-on-connected-30n80).

## sella-n12-early-delta-floor: connected δ floor 0.15 after 10 steps on n_atoms<12
Status: closed

**Hypothesis:** Leftover 252089162 extra +2 is MIS-truncated in steps 10–20. Starting cycle 88’s 0.15 / σ_inc=1.16 floor at 10 on connected n<12 enlarges those steps without paliperidone.

**Outcome and uncertainty:** Cycle 481 discard, Δ=0. n_steps identical to champion 478. Leftover stays 39 with an energy twitch. The floor fires and is not a force-call limiter.

**Reason to revisit:** Not as an earlier 0.15 floor on n<12. Do not interpolate after 5.

**Next experiment:** Close. See GDIIS cosine 0.85 on connected n<12.

**Attempts:**
- Cycle 481; candidate `89526c52b9c5b5cca9f82634d86b213b4df144e0`; champion `1cf4b725273d95a1f17f2e367a57199cbe3af436`; decision discard.
  [Training evidence](evaluation_results/cycle-481-train.json).
  [Narrative](full_log.md#cycle-481--connected-δ-floor-015-after-10-steps-on-n_atoms12).

## sella-n12-gdiis-cos-085: GDIIS cosine 0.85 on connected n_atoms<12
Status: closed

**Hypothesis:** Leftover 252089162 rejects two-point GDIIS at cosine 0.90. 0.85 on connected n<12 admits interpolants without cycle 242 n≥30 extras.

**Outcome and uncertainty:** Cycle 482 discard, Δ=0, n_steps and energies bit-identical. Cosine is not the n<12 GDIIS gate.

**Reason to revisit:** Not as a cosine floor below 0.90 on n<12. Do not try 0.80.

**Next experiment:** Close. See skip-ρ GDIIS on connected n<12.

**Attempts:**
- Cycle 482; candidate `b784fe41b1fe6a36ea593b2bbc4063315a9c519f`; champion `1cf4b725273d95a1f17f2e367a57199cbe3af436`; decision discard.
  [Training evidence](evaluation_results/cycle-482-train.json).
  [Narrative](full_log.md#cycle-482--gdiis-cosine-085-on-connected-n_atoms12).

## sella-n12-gdiis-skip-rho: skip GDIIS ρ window on connected n_atoms<12
Status: closed

**Hypothesis:** Leftover 252089162 fails GDIIS at the ρ window. Skipping ρ on connected n<12 admits interpolants without cycle 243 paliperidone extras.

**Outcome and uncertainty:** Cycle 483 discard, Δ=0, n_steps and energies bit-identical. ρ is not the n<12 GDIIS gate. Combined with skip (256) and cosine 0.85 (482), n<12 GDIIS is inert.

**Reason to revisit:** Not as a GDIIS ρ/cosine/start-step tweak on n<12. Do not start GDIIS before 20 on that band.

**Next experiment:** Close n<12 GDIIS. See 2-coordinate S–N–S H0 on connected n<18.

**Attempts:**
- Cycle 483; candidate `414761e25b527cd0e7e8e500e4f479fc9fe60aff`; champion `1cf4b725273d95a1f17f2e367a57199cbe3af436`; decision discard.
  [Training evidence](evaluation_results/cycle-483-train.json).
  [Narrative](full_log.md#cycle-483--skip-gdiis-ρ-window-on-connected-n_atoms12).

## sella-ammonium-cnc-h0: 0.10 Ha tetrahedral NR4 C–N–C on connected 12≤n<30
Status: closed

**Hypothesis:** Train leftover 433943527 is limited by C–N–C at 4-coordinate N with four carbon neighbors and ≥2 methyl carbons.

**Outcome and uncertainty:** Cycle 499/501 discard, Δ=−1.07e-4: leftover 21→22 at 0.10 and 0.13 Ha on all six. Cycle 500 two-of-six discard, Δ=0, bit-identical. Cycle 502 methyl–methyl discard, Δ=−1.07e-4: leftover 21→22 again. Softening any NR4 C–N–C extras leftover.

**Reason to revisit:** Not as NR4 C–N–C H0. Leftover 433943527 is not C–N–C limited (extras). Do not interpolate 0.15.

**Next experiment:** Close.

**Attempts:**
- Cycle 502; candidate `2dab9b9699c23f5d235441efc9887b9e760db668`; champion `7f799f88e82511fa016c401d7128a7cd0e71d800`; decision discard.
  [Implementation](ideas/sella-ammonium-cnc-h0/2dab9b9699c23f5d235441efc9887b9e760db668/algo.py).
  [Training evidence](evaluation_results/cycle-502-train.json).
  [Narrative](full_log.md#cycle-502--010-ha-tetrahedral-nr4-methylmethyl-cnc-on-connected-12n30-with-13-cap).
- Cycle 501; candidate `640fce1f1dd87f436cd8fe61a84ccf9367179c0e`; champion `7f799f88e82511fa016c401d7128a7cd0e71d800`; decision discard.
  [Implementation](ideas/sella-ammonium-cnc-h0/640fce1f1dd87f436cd8fe61a84ccf9367179c0e/algo.py).
  [Training evidence](evaluation_results/cycle-501-train.json).
  [Narrative](full_log.md#cycle-501--013-ha-tetrahedral-nr4-cnc-on-connected-12n30-with-16-cap).
- Cycle 500; candidate `3cbbdcd3122d853cec73b97e3cb945c3b5f24438`; champion `7f799f88e82511fa016c401d7128a7cd0e71d800`; decision discard.
  [Implementation](ideas/sella-ammonium-cnc-h0/3cbbdcd3122d853cec73b97e3cb945c3b5f24438/algo.py).
  [Training evidence](evaluation_results/cycle-500-train.json).
  [Narrative](full_log.md#cycle-500--tetrahedral-nr4-cnc-applying-two-of-six-angles).
- Cycle 499; candidate `7b787c2f5dfdb4c4ce1073ba00d1b370e851c53e`; champion `7f799f88e82511fa016c401d7128a7cd0e71d800`; decision discard.
  [Implementation](ideas/sella-ammonium-cnc-h0/7b787c2f5dfdb4c4ce1073ba00d1b370e851c53e/algo.py).
  [Training evidence](evaluation_results/cycle-499-train.json).
  [Narrative](full_log.md#cycle-499--010-ha-tetrahedral-nr4-cnc-on-connected-12n30-with-16-cap).

## sella-n18-sns-h0: 0.10 Ha 2-coordinate S–N–S on connected n<18
Status: incorporated

**Hypothesis:** Valid leftover 135100218 is limited by 2-coordinate S–N–S. Soft 0.10 Ha on connected n<18 cheapens it and train 135093970 / 135097481.

**Outcome and uncertainty:** Cycle 484 invalid (sns=4 hop). Cycle 485 NG train-only. Cycle 492 NG: C2F5 saved leftover 135021635 31→30 (Δ_valid=+7.17e-5). Cycle 497 keep stacked S–N–S + organic C2F5 + hetero C–S–C. New champion `7f799f8`.

**Reason to revisit:** Incorporated. Do not drop the 1–3 S–N–S cap or the C2F5 unfluorinated-attach gate.

**Next experiment:** Close isoxazole C–C–O (train-empty). Target a train leftover with n_ref leverage vs `7f799f8`.

**Attempts:**
- Cycle 498; candidate `7b81f2ef8903d435ab8498710d708f981d261a7a`; champion `7f799f88e82511fa016c401d7128a7cd0e71d800`; decision discard.
  [Training evidence](evaluation_results/cycle-498-train.json).
  [Narrative](full_log.md#cycle-498--010-ha-isoxazole-cco-on-connected-30n80-with-12-cap).
- Cycle 497; candidate `7f799f88e82511fa016c401d7128a7cd0e71d800`; champion `1cf4b725273d95a1f17f2e367a57199cbe3af436`; decision keep.
  [Training evidence](evaluation_results/cycle-497-train.json).
  [Validation evidence](evaluation_results/cycle-497-valid.json).
  [Narrative](full_log.md#cycle-497--cycle-492-plus-heterohalo-csc-010-ha-on-connected-30n80).
- Cycle 496; candidate `7fb8d617ee3b0c12e5a262505f6de828fc930d89`; champion `1cf4b725273d95a1f17f2e367a57199cbe3af436`; decision non_generalizable.
  [Implementation](ideas/sella-n18-sns-h0/7fb8d617ee3b0c12e5a262505f6de828fc930d89/algo.py).
  [Training evidence](evaluation_results/cycle-496-train.json).
  [Validation evidence](evaluation_results/cycle-496-valid.json).
  [Narrative](full_log.md#cycle-496--amidine-ncn-skipping-fused-heteroaryl-next-nearest-6-ring).
- Cycle 495; candidate `3e1461336ca784a073850127482dea7f567c9d79`; champion `1cf4b725273d95a1f17f2e367a57199cbe3af436`; decision discard.
  [Training evidence](evaluation_results/cycle-495-train.json).
  [Narrative](full_log.md#cycle-495--amidine-ncn-skipping-fused-6-ring-carbon-attach).
- Cycle 494; candidate `1d007076eef5fde1f021ceabfd8cd3984cd2a9fa`; champion `1cf4b725273d95a1f17f2e367a57199cbe3af436`; decision discard.
  [Implementation](ideas/sella-n18-sns-h0/1d007076eef5fde1f021ceabfd8cd3984cd2a9fa/algo.py).
  [Training evidence](evaluation_results/cycle-494-train.json).
  [Narrative](full_log.md#cycle-494--cycle-493-amidine-ncn-skipping-56-membered-cn-rings).
- Cycle 493; candidate `6eeaa82a8d7bbfb5e1a4c122a1b787bc9fe99747`; champion `1cf4b725273d95a1f17f2e367a57199cbe3af436`; decision discard.
  [Implementation](ideas/sella-n18-sns-h0/6eeaa82a8d7bbfb5e1a4c122a1b787bc9fe99747/algo.py).
  [Training evidence](evaluation_results/cycle-493-train.json).
  [Narrative](full_log.md#cycle-493--cycle-492-plus-amidine-ncn-010-ha-on-connected-30n80-with-12-cap).
- Cycle 492; candidate `09f5ca444177c24e34bb2e397f02d17f429f9f5c`; champion `1cf4b725273d95a1f17f2e367a57199cbe3af436`; decision non_generalizable.
  [Implementation](ideas/sella-n18-sns-h0/09f5ca444177c24e34bb2e397f02d17f429f9f5c/algo.py).
  [Training evidence](evaluation_results/cycle-492-train.json).
  [Validation evidence](evaluation_results/cycle-492-valid.json).
  [Narrative](full_log.md#cycle-492--cycle-485-plus-organic-c2f5-fcc-unfluorinated-attach-on-3080).
- Cycle 491; candidate `1ac6288ca4cd1415259a61da7a97d1fcc0222535`; champion `1cf4b725273d95a1f17f2e367a57199cbe3af436`; decision discard.
  [Implementation](ideas/sella-n18-sns-h0/1ac6288ca4cd1415259a61da7a97d1fcc0222535/algo.py).
  [Training evidence](evaluation_results/cycle-491-train.json).
  [Narrative](full_log.md#cycle-491--cycle-485-plus-pentafluoroethyl-fcc-010-ha-on-connected-30n80-with-14-cap).
- Cycle 490; candidate `4fc0ec9332ce889dd16f5d8675eca2d17d4e1a5e`; champion `1cf4b725273d95a1f17f2e367a57199cbe3af436`; decision non_generalizable.
  [Implementation](ideas/sella-n18-sns-h0/4fc0ec9332ce889dd16f5d8675eca2d17d4e1a5e/algo.py).
  [Training evidence](evaluation_results/cycle-490-train.json).
  [Validation evidence](evaluation_results/cycle-490-valid.json).
  [Narrative](full_log.md#cycle-490--cycle-485-plus-clch-010-ha-on-connected-n18-with-12-cap).
- Cycle 488; candidate `922df572e0e85e2e23c7f976031f5488c488d5c0`; champion `1cf4b725273d95a1f17f2e367a57199cbe3af436`; decision non_generalizable.
  [Implementation](ideas/sella-n18-sns-h0/922df572e0e85e2e23c7f976031f5488c488d5c0/algo.py).
  [Training evidence](evaluation_results/cycle-488-train.json).
  [Validation evidence](evaluation_results/cycle-488-valid.json).
  [Narrative](full_log.md#cycle-488--cycle-485-plus-alkyne-gated-exact-geodesic-on-connected-n16).
- Cycle 487; candidate `f9381f72694bdf7732c82b3d7bc9a1d9a244312a`; champion `1cf4b725273d95a1f17f2e367a57199cbe3af436`; decision discard.
  [Implementation](ideas/sella-n18-sns-h0/f9381f72694bdf7732c82b3d7bc9a1d9a244312a/algo.py).
  [Training evidence](evaluation_results/cycle-487-train.json).
  [Narrative](full_log.md#cycle-487--cycle-485-plus-exact-geodesic-on-connected-n_atoms16).
- Cycle 486; candidate `c5c83a8520b6fe7bfdda35b065fe0997282e0ba6`; champion `1cf4b725273d95a1f17f2e367a57199cbe3af436`; decision non_generalizable.
  [Implementation](ideas/sella-n18-sns-h0/c5c83a8520b6fe7bfdda35b065fe0997282e0ba6/algo.py).
  [Training evidence](evaluation_results/cycle-486-train.json).
  [Validation evidence](evaluation_results/cycle-486-valid.json).
  [Narrative](full_log.md#cycle-486--cycle-485-plus-exactly-three-2-coord-ccc-on-connected-n18).
- Cycle 485; candidate `a1b6c162eec788afe6fc151c930b4fc761b3f010`; champion `1cf4b725273d95a1f17f2e367a57199cbe3af436`; decision non_generalizable.
  [Implementation](ideas/sella-n18-sns-h0/a1b6c162eec788afe6fc151c930b4fc761b3f010/algo.py).
  [Training evidence](evaluation_results/cycle-485-train.json).
  [Validation evidence](evaluation_results/cycle-485-valid.json).
  [Narrative](full_log.md#cycle-485--010-ha-2-coordinate-sns-on-connected-n18-with-13-cap).
- Cycle 484; candidate `54ae8f93259f9f0ed96bcd8dfb8329957b4da694`; champion `1cf4b725273d95a1f17f2e367a57199cbe3af436`; decision invalid.
  [Implementation](ideas/sella-n18-sns-h0/54ae8f93259f9f0ed96bcd8dfb8329957b4da694/algo.py).
  [Training evidence](evaluation_results/cycle-484-train.json).
  [Narrative](full_log.md#cycle-484--010-ha-2-coordinate-sns-on-connected-n18).

## sella-n30-gdiis-after-15: GDIIS after 15 steps on connected 30≤n<80
Status: closed

**Hypothesis:** Leftover 363892164 can interpolate five steps earlier than the nsteps=20 GDIIS gate on connected 30–80 without paliperidone.

**Outcome and uncertainty:** Cycle 489 discard, Δ=0, n_steps and energies bit-identical. GDIIS after 15 is a no-op on 30–80.

**Reason to revisit:** Not as a GDIIS start-step tweak on 30–80. Do not try after 10.

**Next experiment:** Close. See cycle 485 stacked with Cl–C–H on n<18.

**Attempts:**
- Cycle 489; candidate `bcae8a645188c63102563cf4a13ddfa8124411dc`; champion `1cf4b725273d95a1f17f2e367a57199cbe3af436`; decision discard.
  [Training evidence](evaluation_results/cycle-489-train.json).
  [Narrative](full_log.md#cycle-489--gdiis-after-15-steps-on-connected-30n80).

## sella-oxo-sulfoxide-csc-h0: 0.10 Ha 3-coordinate sulfoxide C–S–C on connected n<12
Status: closed

**Hypothesis:** Leftover 135041973 extraed under O–S–C and Br–C–Br. Unused C–S–C at 3-coordinate S {C, C, O} is the remaining S-center bend.

**Outcome and uncertainty:** Cycle 503 discard, Δ=−5.33e-4. Only leftover 13→16 extra. Soft C–S–C overshoots more than O–S–C.

**Reason to revisit:** Not as 0.10/0.13 C–S–C on n<12. Leftover is not C–S–C limited (extras).

**Next experiment:** Close. Try leftover Br–C–S instead.

**Attempts:**
- Cycle 503; candidate `a45dccff08b86d04d81fb023fa75bc5aee0850d7`; champion `7f799f88e82511fa016c401d7128a7cd0e71d800`; decision discard.
  [Implementation](ideas/sella-oxo-sulfoxide-csc-h0/a45dccff08b86d04d81fb023fa75bc5aee0850d7/algo.py).
  [Training evidence](evaluation_results/cycle-503-train.json).
  [Narrative](full_log.md#cycle-503--010-ha-3-coordinate-sulfoxide-csc-on-connected-n12-with-12-cap).

## sella-n18-imidazole-cnc-h0: 0.10 Ha 3-coordinate NH C–N–C on connected n<18
Status: closed

**Hypothesis:** Leftover 135092053 unused C–N–C at 3-coordinate NH {C, C, H} is distinct from N-oxide O–N–C extras and N-oxide C–N–C twitch.

**Outcome and uncertainty:** Cycle 506 discard, Δ=0. n_steps identical; leftover 135092053 and 135237854 energy twitch. Class fires and is not the limiter.

**Reason to revisit:** Not as 0.10 NH C–N–C on n<18. Do not restack N-oxide H0.

**Next experiment:** Close.

**Attempts:**
- Cycle 506; candidate `672af045985283de48a7c28c07a0b1c1d00ad40f`; champion `7f799f88e82511fa016c401d7128a7cd0e71d800`; decision discard.
  [Implementation](ideas/sella-n18-imidazole-cnc-h0/672af045985283de48a7c28c07a0b1c1d00ad40f/algo.py).
  [Training evidence](evaluation_results/cycle-506-train.json).
  [Narrative](full_log.md#cycle-506--010-ha-3-coordinate-nh-cnc-on-connected-n18-with-13-cap).

## sella-oxo-ps-stretch-h0: 0.25 Ha/Bohr² oxygen-free P–S stretch on connected n<12
Status: closed

**Hypothesis:** Leftover 135094613 P4S6 is stretch-limited after P–S–P twitch and S–P–S NG. Cycle 309 P–O 0.25 extraed oxygen-containing thiophosphate; skipping oxygen tags P4S6 only.

**Outcome and uncertainty:** Cycle 507 discard, Δ=−2.37e-4. Leftover stays 17. Extra 135095649 SPCl 9→10. Leftover is not P–S stretch-limited.

**Reason to revisit:** Not as 0.25/0.20 P–S stretch on n<12. Do not P/S-only (leftover already tagged).

**Next experiment:** Close.

**Attempts:**
- Cycle 507; candidate `b4746bddb3ae55e5900b12971ddc34a5183f942e`; champion `7f799f88e82511fa016c401d7128a7cd0e71d800`; decision discard.
  [Implementation](ideas/sella-oxo-ps-stretch-h0/b4746bddb3ae55e5900b12971ddc34a5183f942e/algo.py).
  [Training evidence](evaluation_results/cycle-507-train.json).
  [Narrative](full_log.md#cycle-507--025-habohr2-oxygen-free-ps-stretch-on-connected-n12).

## sella-oxo-tribromo-brcs-h0: 0.10 Ha CBr3 Br–C–S on connected n<12
Status: closed

**Hypothesis:** Leftover 135041973 extraed under O–S–C, Br–C–Br, and C–S–C. Unused Br–C–S at 4-coordinate C {Br, Br, Br, S} is the remaining carbon-center mixed bend.

**Outcome and uncertainty:** Cycle 504 discard, Δ=0, bit-identical (skip-if-more untagged leftover). Cycle 505 discard, Δ=0: leftover energy twitch at 13; Br–C–S fires and is not the limiter.

**Reason to revisit:** Not as Br–C–S 0.10. Leftover 135041973 valence angles extra or twitch.

**Next experiment:** Close leftover 135041973 angle H0.

**Attempts:**
- Cycle 505; candidate `ef4710ca1c4f9641c895a40e4ef6f689eed3b563`; champion `7f799f88e82511fa016c401d7128a7cd0e71d800`; decision discard.
  [Implementation](ideas/sella-oxo-tribromo-brcs-h0/ef4710ca1c4f9641c895a40e4ef6f689eed3b563/algo.py).
  [Training evidence](evaluation_results/cycle-505-train.json).
  [Narrative](full_log.md#cycle-505--cbr3-brcs-applying-two-of-six-angles-on-connected-n12).
- Cycle 504; candidate `a71e6641441997040f4fd86560acf7737dae2351`; champion `7f799f88e82511fa016c401d7128a7cd0e71d800`; decision discard.
  [Implementation](ideas/sella-oxo-tribromo-brcs-h0/a71e6641441997040f4fd86560acf7737dae2351/algo.py).
  [Training evidence](evaluation_results/cycle-504-train.json).
  [Narrative](full_log.md#cycle-504--010-ha-cbr3-brcs-on-connected-n12-with-12-cap).

## sella-fused-ch2-ccc-h0: 0.10 Ha fused CH2 C–C–C on connected 30≤n<80
Status: incorporated

**Hypothesis:** Leftover 440723624 unused C–C–C at 4-coordinate CH2 fused to a 3-coordinate carbon is distinct from extraing CF3 H0.

**Outcome and uncertainty:** Cycle 508 discard, Δ=+4.70e-5. Cycle 509 NG: train Δ=+1.99e-4; valid extras three sulfur frames. Cycle 510 NG: train Δ=+1.99e-4, valid Δ=0. Cycle 513 **keep** stacked fused CH2 skip-S with mixed C–S–C and CF3–S F–C–S.

**Reason to revisit:** Incorporated in champion `55dbd7a`. Do not drop the sulfur skip or interpolate 0.13.

**Next experiment:** None as a standalone fused-CH2 class.

**Attempts:**
- Cycle 513; candidate `55dbd7a57c878cb8f99a15468f3e3bb576af4403`; champion `7f799f88e82511fa016c401d7128a7cd0e71d800`; decision keep.
  [Training evidence](evaluation_results/cycle-513-train.json).
  [Validation evidence](evaluation_results/cycle-513-valid.json).
  [Narrative](full_log.md#cycle-513--fused-ch2-plus-mixed-csc-plus-cf3s-fcs-on-n18).
- Cycle 510; candidate `27768c6aed85155dafc18e86fd8a92a3fdef9266`; champion `7f799f88e82511fa016c401d7128a7cd0e71d800`; decision non_generalizable.
  [Implementation](ideas/sella-fused-ch2-ccc-h0/27768c6aed85155dafc18e86fd8a92a3fdef9266/algo.py).
  [Training evidence](evaluation_results/cycle-510-train.json).
  [Validation evidence](evaluation_results/cycle-510-valid.json).
  [Narrative](full_log.md#cycle-510--fused-ch2-ccc-skipping-sulfur-containing-molecules).
- Cycle 509; candidate `0e193964ed30aa787d147b2d5de42a968318818d`; champion `7f799f88e82511fa016c401d7128a7cd0e71d800`; decision non_generalizable.
  [Implementation](ideas/sella-fused-ch2-ccc-h0/0e193964ed30aa787d147b2d5de42a968318818d/algo.py).
  [Training evidence](evaluation_results/cycle-509-train.json).
  [Validation evidence](evaluation_results/cycle-509-valid.json).
  [Narrative](full_log.md#cycle-509--fused-ch2-ccc-requiring-a-3-coordinate-ring-carbon-with-two-3-coordinate-neighbors).
- Cycle 508; candidate `bdf01c76f9e80526e75effa28c7eef8aa766eff1`; champion `7f799f88e82511fa016c401d7128a7cd0e71d800`; decision discard.
  [Implementation](ideas/sella-fused-ch2-ccc-h0/bdf01c76f9e80526e75effa28c7eef8aa766eff1/algo.py).
  [Training evidence](evaluation_results/cycle-508-train.json).
  [Narrative](full_log.md#cycle-508--010-ha-fused-ch2-ccc-on-connected-30n80-with-12-cap).

## sella-alkyl-aryl-csc-h0: 0.10 Ha isolated alkyl–aryl mixed C–S–C on connected 30≤n<80
Status: incorporated

**Hypothesis:** Cycle 437 mixed C–S–C on diaryl-sulfide molecules was train-empty. Isolated alkyl–aryl C(3)–S–C(4) with CH2–alkyl, stacked on fused CH2 skip-S, tests leftover 194689238.

**Outcome and uncertainty:** Cycle 512 NG. Train Δ=+1.99e-4. Valid Δ=+5.25e-5: 135249279 40→39; leftover 194689238 twitch at 32. No extras. Cycle 513 **keep** stacked this save with CF3–S F–C–S.

**Reason to revisit:** Incorporated in champion `55dbd7a`. Leftover 194689238 mixed C–S–C is not the limiter.

**Next experiment:** None as mixed C–S–C H0-scale interpolation.

**Attempts:**
- Cycle 513; candidate `55dbd7a57c878cb8f99a15468f3e3bb576af4403`; champion `7f799f88e82511fa016c401d7128a7cd0e71d800`; decision keep.
  [Training evidence](evaluation_results/cycle-513-train.json).
  [Validation evidence](evaluation_results/cycle-513-valid.json).
  [Narrative](full_log.md#cycle-513--fused-ch2-plus-mixed-csc-plus-cf3s-fcs-on-n18).
- Cycle 512; candidate `6a5478485aaaf02b4c25513641603eaa5ad035c8`; champion `7f799f88e82511fa016c401d7128a7cd0e71d800`; decision non_generalizable.
  [Implementation](ideas/sella-alkyl-aryl-csc-h0/6a5478485aaaf02b4c25513641603eaa5ad035c8/algo.py).
  [Training evidence](evaluation_results/cycle-512-train.json).
  [Validation evidence](evaluation_results/cycle-512-valid.json).
  [Narrative](full_log.md#cycle-512--fused-ch2-skip-s-plus-isolated-alkylaryl-mixed-csc).


## sella-fluoro-aniline-cnc-h0: 0.10 Ha fluoroaniline N-aryl C–N–C on connected 30≤n<80
Status: closed

**Hypothesis:** Valid leftover 252656292 unused C–N–C at 3-coordinate N-aryl with two CH2, stacked on fused CH2 skip-S.

**Outcome and uncertainty:** Cycle 511 NG. Train n_steps-identical to 510. Valid leftover 252656292 energy twitch at 18: C–N–C fires and is not the limiter.

**Reason to revisit:** Not as an H0-scale interpolation. Do not restack on 252656292 C–N–C.

**Next experiment:** None. Close.

**Attempts:**
- Cycle 511; candidate `5b76980e103241129aaf17a7e2fc81847248f3c5`; champion `7f799f88e82511fa016c401d7128a7cd0e71d800`; decision non_generalizable.
  [Implementation](ideas/sella-fluoro-aniline-cnc-h0/5b76980e103241129aaf17a7e2fc81847248f3c5/algo.py).
  [Training evidence](evaluation_results/cycle-511-train.json).
  [Validation evidence](evaluation_results/cycle-511-valid.json).
  [Narrative](full_log.md#cycle-511--fused-ch2-skip-s-plus-fluoroaniline-n-aryl-cnc).

## sella-cf3s-fcs-h0: 0.10 Ha CF3–S F–C–S on connected n<18
Status: incorporated

**Hypothesis:** Leftover 135100218 unused CF3–S F–C–S, stacked on fused CH2 skip-S and mixed C–S–C.

**Outcome and uncertainty:** Cycle 513 **keep**. Train Δ=+1.99e-4 (fused CH2). Valid Δ=+1.42e-4: 135249279 −1 and leftover 135100218 25→24. No extras.

**Reason to revisit:** Incorporated in champion `55dbd7a`. Do not drop the 1–3 cap (train 135098013 has six F–C–S). Do not fill 30–80 (441095956 extra risk).

**Next experiment:** None as an F–C–S scale interpolation.

**Attempts:**
- Cycle 513; candidate `55dbd7a57c878cb8f99a15468f3e3bb576af4403`; champion `7f799f88e82511fa016c401d7128a7cd0e71d800`; decision keep.
  [Training evidence](evaluation_results/cycle-513-train.json).
  [Validation evidence](evaluation_results/cycle-513-valid.json).
  [Narrative](full_log.md#cycle-513--fused-ch2-plus-mixed-csc-plus-cf3s-fcs-on-n18).

## sella-thiosulfonate-oss-h0: 0.10/0.21 Ha thiosulfonate O–S–S on connected 18≤n<30
Status: closed

**Hypothesis:** Leftover 433943527 unused O–S–S at 4-coordinate S {O, O, S, C} is the limiter after NR4 extras.

**Outcome and uncertainty:** Cycle 514 discard, leftover 21→22 extra. Cycle 515 discard, still 21→22 with one 0.10 angle. Cycle 516 discard, Δ=0 twitch at 0.21 Ha.

**Reason to revisit:** Not as an H0-scale interpolation. Soft O–S–S extras leftover; stiff O–S–S is a no-op.

**Next experiment:** None. Close leftover 433943527 valence-angle H0.

**Attempts:**
- Cycle 516; candidate `6c73faeb055e35fd8a4d3956fc914e450b095961`; champion `55dbd7a57c878cb8f99a15468f3e3bb576af4403`; decision discard.
  [Implementation](ideas/sella-thiosulfonate-oss-h0/6c73faeb055e35fd8a4d3956fc914e450b095961/algo.py).
  [Training evidence](evaluation_results/cycle-516-train.json).
  [Narrative](full_log.md#cycle-516--thiosulfonate-oss-021-ha-on-the-first-angle).
- Cycle 515; candidate `0d8923fb6d2594d4acad0995b7cb01a9be01e7b4`; champion `55dbd7a57c878cb8f99a15468f3e3bb576af4403`; decision discard.
  [Implementation](ideas/sella-thiosulfonate-oss-h0/0d8923fb6d2594d4acad0995b7cb01a9be01e7b4/algo.py).
  [Training evidence](evaluation_results/cycle-515-train.json).
  [Narrative](full_log.md#cycle-515--thiosulfonate-oss-apply-first-of-two-angles).
- Cycle 514; candidate `e43c04c006a9b96b401bb7e4cbea207748a8052b`; champion `55dbd7a57c878cb8f99a15468f3e3bb576af4403`; decision discard.
  [Implementation](ideas/sella-thiosulfonate-oss-h0/e43c04c006a9b96b401bb7e4cbea207748a8052b/algo.py).
  [Training evidence](evaluation_results/cycle-514-train.json).
  [Narrative](full_log.md#cycle-514--thiosulfonate-oss-010-ha-on-connected-18n30).

## sella-dithiole-scs-h0: 0.10 Ha 1,3-dithiole S–C–S on connected 12≤n<80
Status: incorporated

**Hypothesis:** Valid leftover 104130706 unused S–C–S at 3-coordinate carbon with two 2-coordinate S and one C is the limiter.

**Outcome and uncertainty:** Cycle 517 discard, Δ=0 on 12–80 (train analogs twitch). Cycle 521 **keep** on 30–80 stacked with C–C–Cl: leftover 104130706 35→33.

**Reason to revisit:** Incorporated as 30–80 in champion `ffd575c`. Do not fill n<30 (cycle 517 twitch). Do not drop the carbon-third-neighbor gate or 1–2 cap.

**Next experiment:** None as an S–C–S scale interpolation.

**Attempts:**
- Cycle 521; candidate `ffd575c14ebd2a9da555e6bc8f786766fd428bed`; champion `55dbd7a57c878cb8f99a15468f3e3bb576af4403`; decision keep.
  [Training evidence](evaluation_results/cycle-521-train.json).
  [Validation evidence](evaluation_results/cycle-521-valid.json).
  [Narrative](full_log.md#cycle-521--ccl-plus-13-dithiole-scs-on-30n80).
- Cycle 517; candidate `eb8df4e83d823b4d88394bd644becefac5fdd5e2`; champion `55dbd7a57c878cb8f99a15468f3e3bb576af4403`; decision discard.
  [Implementation](ideas/sella-dithiole-scs-h0/eb8df4e83d823b4d88394bd644becefac5fdd5e2/algo.py).
  [Training evidence](evaluation_results/cycle-517-train.json).
  [Narrative](full_log.md#cycle-517--010-ha-13-dithiole-scs-on-connected-12n80).

## sella-aryl-f-fcc-h0: 0.10 Ha aryl F–C–C skipping Cl/Br/I and CF3
Status: closed

**Hypothesis:** Leftover 252656292 unused aryl F–C–C is the limiter after N-aryl C–N–C twitch and N-oxide O–N–C extras.

**Outcome and uncertainty:** Cycle 518 discard, Δ=0. Twitches on 125086620 and olaparib; Cl/CF3 skips held.

**Reason to revisit:** Not as an F–C–C H0-scale interpolation. Do not interpolate 0.13. Do not drop the Cl/CF3 skips.

**Next experiment:** None. Close aryl F–C–C as a train n_steps mechanism.

**Attempts:**
- Cycle 518; candidate `8f7a3a30213c15052ac7e087dcc516607c31f9b7`; champion `55dbd7a57c878cb8f99a15468f3e3bb576af4403`; decision discard.
  [Implementation](ideas/sella-aryl-f-fcc-h0/8f7a3a30213c15052ac7e087dcc516607c31f9b7/algo.py).
  [Training evidence](evaluation_results/cycle-518-train.json).
  [Narrative](full_log.md#cycle-518--010-ha-aryl-fcc-skipping-clbri-and-cf3).

## sella-fluoro-hetaryl-fcc-h0: 0.10 Ha fluoro-hetaryl F–C–C stacked on C–C–Cl
Status: closed

**Hypothesis:** Cycle 519 C–C–Cl is train-only. Leftover 252656292 unused F–C–C with nitrogen on a carbon neighbor is the limiter; cycle 518’s broader aryl F–C–C twitched on different molecules.

**Outcome and uncertainty:** Cycle 520 NG. Train identical to 519. Valid leftover 252656292 twitch at 18; 440971005 twitch at 34.

**Reason to revisit:** Not as an F–C–C H0-scale interpolation on leftover 252656292. Do not interpolate 0.13.

**Next experiment:** Close leftover 252656292 valence-angle H0. Keep C–C–Cl as the train stack.

**Attempts:**
- Cycle 520; candidate `48a0d867b770530079ff749c76333ce4dce671f4`; champion `55dbd7a57c878cb8f99a15468f3e3bb576af4403`; decision non_generalizable.
  [Implementation](ideas/sella-fluoro-hetaryl-fcc-h0/48a0d867b770530079ff749c76333ce4dce671f4/algo.py).
  [Training evidence](evaluation_results/cycle-520-train.json).
  [Validation evidence](evaluation_results/cycle-520-valid.json).
  [Narrative](full_log.md#cycle-520--ccl-plus-fluoro-hetaryl-fcc).

## sella-pyrrolidine-cnc-h0: 0.10 Ha pyrrolidine CH2–N–CH2 stacked on N-oxide n<18
Status: closed

**Hypothesis:** Leftover 252656292 unused CH2–N–CH2 in the N-aryl pyrrolidine is the limiter after closed valence H0. Stacked on cycle-522 N-oxide, leftover −1 would pass valid δ.

**Outcome and uncertainty:** Cycle 523 non_generalizable. Train identical to 522. Valid leftover 252656292 twitch at 18; extra 135174398; hop 135065494 90→119 / +2.10 kcal.

**Reason to revisit:** Not as pyrrolidine C–N–C. Leftover already twitched; do not N-aryl-repair.

**Next experiment:** None. Close leftover 252656292 pyrrolidine C–N–C.

**Attempts:**
- Cycle 523; candidate `f8dd30122aac914231e68318d7390b0d9a5c3735`; champion `ffd575c14ebd2a9da555e6bc8f786766fd428bed`; decision non_generalizable.
  [Implementation](ideas/sella-pyrrolidine-cnc-h0/f8dd30122aac914231e68318d7390b0d9a5c3735/algo.py).
  [Training evidence](evaluation_results/cycle-523-train.json).
  [Validation evidence](evaluation_results/cycle-523-valid.json).
  [Narrative](full_log.md#cycle-523--n-oxide-n18-plus-pyrrolidine-ch2nch2-on-30n80).

## sella-n18-alcohol-hco-h0: 0.10 Ha primary-alcohol H–C–O stacked on N-oxide n<18
Status: closed

**Hypothesis:** Leftover 135169446 unused H–C–O at CH2OH {O, C, H, H} is the limiter after closed C–O–H / O–C–C / C–C–Cl. Stacked on cycle-522 N-oxide, leftover −1 would pass valid δ.

**Outcome and uncertainty:** Cycle 524 non_generalizable. Train identical to 522. Valid leftover 135169446 extras 23→24. Energy-safe.

**Reason to revisit:** Not as alcohol H–C–O. Soft H0 extras leftover. Do not interpolate 0.13.

**Next experiment:** None. Close leftover 135169446 valence-angle H0.

**Attempts:**
- Cycle 524; candidate `26a7ca23a4f0a08309a1e6f2b94aa6304b47288f`; champion `ffd575c14ebd2a9da555e6bc8f786766fd428bed`; decision non_generalizable.
  [Implementation](ideas/sella-n18-alcohol-hco-h0/26a7ca23a4f0a08309a1e6f2b94aa6304b47288f/algo.py).
  [Training evidence](evaluation_results/cycle-524-train.json).
  [Validation evidence](evaluation_results/cycle-524-valid.json).
  [Narrative](full_log.md#cycle-524--n-oxide-plus-primary-alcohol-hco-on-connected-n18).

## sella-sulfonium-csc-h0: 0.10 Ha sulfonium C–S–C stacked on N-oxide n<18
Status: closed

**Hypothesis:** Leftover 163343378 unused C–S–C at 3-coordinate S {C, C, C} is the limiter. Stacked on cycle-522 N-oxide, leftover −2 would pass valid δ.

**Outcome and uncertainty:** Cycle 525 non_generalizable. Train identical to 522. Valid leftover 163343378 extras 33→56 / −5.75 kcal, same hop as cycle 353.

**Reason to revisit:** Not as sulfonium C–S–C H0. Soft H0 hops the polyol sugar.

**Next experiment:** None. Close leftover 163343378 sulfonium C–S–C.

**Attempts:**
- Cycle 525; candidate `461852d0511a2af23b191e86c8751528888542cb`; champion `ffd575c14ebd2a9da555e6bc8f786766fd428bed`; decision non_generalizable.
  [Implementation](ideas/sella-sulfonium-csc-h0/461852d0511a2af23b191e86c8751528888542cb/algo.py).
  [Training evidence](evaluation_results/cycle-525-train.json).
  [Validation evidence](evaluation_results/cycle-525-valid.json).
  [Narrative](full_log.md#cycle-525--n-oxide-n18-plus-sulfonium-csc-on-30n80).

## sella-alkyl-hnc-h0: 0.10 Ha alkyl secondary-amine H–N–C stacked on N-oxide n<18
Status: closed

**Hypothesis:** Leftover 194689238 unused H–N–C at the acyclic secondary amine is the limiter. Two such angles would pass valid δ stacked on cycle-522 N-oxide.

**Outcome and uncertainty:** Cycle 526 non_generalizable. Both-CH2 gate missed leftover and extraed 135255884. Cycle 527 non_generalizable. Methyl+CH2 tags leftover; leftover 194689238 energy twitch at 32, not the limiter; 135255884 extra gone.

**Reason to revisit:** Not as H–N–C H0. Leftover already twitched.

**Next experiment:** None. Close leftover 194689238 H–N–C.

**Attempts:**
- Cycle 527; candidate `3233ccc3564ada738a7b4a939c4c720207dc6cdf`; champion `ffd575c14ebd2a9da555e6bc8f786766fd428bed`; decision non_generalizable.
  [Implementation](ideas/sella-alkyl-hnc-h0/3233ccc3564ada738a7b4a939c4c720207dc6cdf/algo.py).
  [Training evidence](evaluation_results/cycle-527-train.json).
  [Validation evidence](evaluation_results/cycle-527-valid.json).
  [Narrative](full_log.md#cycle-527--n-oxide-n18-plus-n-methyl-secondary-amine-hnc-on-30n80).
- Cycle 526; candidate `ecdd8551f7ace39c70a11cc78dde2c7d54b8c7f0`; champion `ffd575c14ebd2a9da555e6bc8f786766fd428bed`; decision non_generalizable.
  [Implementation](ideas/sella-alkyl-hnc-h0/ecdd8551f7ace39c70a11cc78dde2c7d54b8c7f0/algo.py).
  [Training evidence](evaluation_results/cycle-526-train.json).
  [Validation evidence](evaluation_results/cycle-526-valid.json).
  [Narrative](full_log.md#cycle-526--n-oxide-n18-plus-alkyl-secondary-amine-hnc-on-30n80).

## sella-protonated-ester-cco-h0: 0.10 Ha protonated methyl-ester C–C–O stacked on N-oxide n<18
Status: closed

**Hypothesis:** Leftover 103943741 unused C–C–O at protonated methyl-ester carbonyl {O, O, C} (OH + OMe) is the limiter. Two such angles would pass valid δ stacked on cycle-522 N-oxide.

**Outcome and uncertainty:** Cycle 528 non_generalizable. Train identical to 522. Valid leftover 103943741 extras 33→35.

**Reason to revisit:** Not as protonated-ester C–C–O. Soft H0 extras leftover. Do not interpolate 0.13 or apply-first-1.

**Next experiment:** None. Close leftover 103943741 valence-angle H0.

**Attempts:**
- Cycle 528; candidate `bedb0862e36a7f2f75e4275909aa6dca7478d35c`; champion `ffd575c14ebd2a9da555e6bc8f786766fd428bed`; decision non_generalizable.
  [Implementation](ideas/sella-protonated-ester-cco-h0/bedb0862e36a7f2f75e4275909aa6dca7478d35c/algo.py).
  [Training evidence](evaluation_results/cycle-528-train.json).
  [Validation evidence](evaluation_results/cycle-528-valid.json).
  [Narrative](full_log.md#cycle-528--n-oxide-n18-plus-protonated-ester-cco-on-30n80).

## sella-trimethylene-ccc-h0: 0.10 Ha fused trimethylene C–C–C stacked on N-oxide n<18
Status: closed

**Hypothesis:** Leftover 252656292 unused C–C–C at the middle fused CH2 (both neighbors 4-coord CH2 with a 3-coord C) is the limiter after closed F–C–C / pyrrolidine / N-oxide. Leftover −1 would pass valid δ stacked on cycle-522 N-oxide.

**Outcome and uncertainty:** Cycle 529 non_generalizable. Train identical to 522. Valid Δ=0; leftover 252656292 twitch at 18. Class fires and is not the limiter.

**Reason to revisit:** Not as trimethylene C–C–C. Leftover already twitched.

**Next experiment:** None. Close leftover 252656292 valence-angle H0.

**Attempts:**
- Cycle 529; candidate `ead98555850f570010ea537a264b3dea610d6b80`; champion `ffd575c14ebd2a9da555e6bc8f786766fd428bed`; decision non_generalizable.
  [Implementation](ideas/sella-trimethylene-ccc-h0/ead98555850f570010ea537a264b3dea610d6b80/algo.py).
  [Training evidence](evaluation_results/cycle-529-train.json).
  [Validation evidence](evaluation_results/cycle-529-valid.json).
  [Narrative](full_log.md#cycle-529--n-oxide-n18-plus-fused-trimethylene-ccc-on-30n80).

## sella-naryl-alkyl-sch2-h0: 0.10 Ha/Bohr² N-aryl alkyl–aryl S–CH2 stretch stacked on N-oxide n<18
Status: closed

**Hypothesis:** Leftover 194689238 unused S–CH2 of the isolated alkyl–aryl sulfide is the limiter after mixed C–S–C twitched. Exactly one such stretch, N-aryl-gated, stacked on cycle-522 N-oxide would pass valid δ.

**Outcome and uncertainty:** Cycle 530 non_generalizable. Train identical to 522. Valid Δ=0; leftover 194689238 twitch at 32. Class fires and is not the limiter. 135249279 untagged.

**Reason to revisit:** Not as S–CH2 stretch H0. Leftover already twitched on the stretch and on mixed C–S–C / H–N–C.

**Next experiment:** None. Close leftover 194689238 H0. See leftover 103943741 methoxy C–O stretch.

**Attempts:**
- Cycle 530; candidate `5d2b02cfeb4bc790c71c5619c7cff5445e81be7b`; champion `ffd575c14ebd2a9da555e6bc8f786766fd428bed`; decision non_generalizable.
  [Implementation](ideas/sella-naryl-alkyl-sch2-h0/5d2b02cfeb4bc790c71c5619c7cff5445e81be7b/algo.py).
  [Training evidence](evaluation_results/cycle-530-train.json).
  [Validation evidence](evaluation_results/cycle-530-valid.json).
  [Narrative](full_log.md#cycle-530--n-oxide-n18-plus-n-aryl-alkylaryl-sch2-stretch-on-30n80).

## sella-protonated-ester-ome-h0: 0.10 Ha/Bohr² protonated-ester methoxy C–O stretch stacked on N-oxide n<18
Status: closed

**Hypothesis:** Leftover 103943741 unused methoxy C–O stretch is the limiter after C–C–O extraed. Exactly one such stretch stacked on cycle-522 N-oxide would pass valid δ.

**Outcome and uncertainty:** Cycle 531 non_generalizable. Train identical to 522. Valid leftover 103943741 extras 33→41, energy-safe. Soft stretch overshoots leftover more than C–C–O.

**Reason to revisit:** Not as methoxy C–O stretch H0. Soft H0 extras leftover. Do not interpolate 0.15.

**Next experiment:** None. Close leftover 103943741 H0. See leftover 135169446 enyne C–Cl stretch.

**Attempts:**
- Cycle 531; candidate `58c134697f84426b4a6133c8222b7d17444df05e`; champion `ffd575c14ebd2a9da555e6bc8f786766fd428bed`; decision non_generalizable.
  [Implementation](ideas/sella-protonated-ester-ome-h0/58c134697f84426b4a6133c8222b7d17444df05e/algo.py).
  [Training evidence](evaluation_results/cycle-531-train.json).
  [Validation evidence](evaluation_results/cycle-531-valid.json).
  [Narrative](full_log.md#cycle-531--n-oxide-n18-plus-protonated-ester-methoxy-co-stretch-on-30n80).

## sella-enyne-ccl-stretch-h0: 0.10 Ha/Bohr² enyne C–Cl stretch stacked on N-oxide n<18
Status: closed

**Hypothesis:** Leftover 135169446 unused C–Cl stretch at 4-coord carbon with a 2-coord C neighbor is the limiter after C–C–Cl twitched. Stacked on cycle-522 N-oxide, leftover −1 would pass valid δ.

**Outcome and uncertainty:** Cycle 532 non_generalizable. Train identical to 522. Valid leftover 135169446 twitch at 23. Class fires and is not the limiter. Cycle 533 adjacent C–C extraed leftover 23→24.

**Reason to revisit:** Not as C–Cl or CH2Cl–C(2) stretch H0. Leftover H0 extras or twitches.

**Next experiment:** None. Close leftover 135169446 H0.

**Attempts:**
- Cycle 532; candidate `89c4407a96b13320ebbb5ac48e46a8415d616cf6`; champion `ffd575c14ebd2a9da555e6bc8f786766fd428bed`; decision non_generalizable.
  [Implementation](ideas/sella-enyne-ccl-stretch-h0/89c4407a96b13320ebbb5ac48e46a8415d616cf6/algo.py).
  [Training evidence](evaluation_results/cycle-532-train.json).
  [Validation evidence](evaluation_results/cycle-532-valid.json).
  [Narrative](full_log.md#cycle-532--n-oxide-n18-plus-enyne-ccl-stretch-on-n18).

## sella-enyne-cc-stretch-h0: 0.10 Ha/Bohr² CH2Cl–C(2) stretch stacked on N-oxide n<18
Status: closed

**Hypothesis:** Leftover 135169446 unused C–C between 4-coord CH2Cl and 2-coord carbon is the limiter after C–Cl twitched.

**Outcome and uncertainty:** Cycle 533 non_generalizable. Train identical to 522. Valid leftover 135169446 extras 23→24, energy-safe.

**Reason to revisit:** Not as CH2Cl–C(2) stretch H0. Soft H0 extras leftover. Do not interpolate 0.15.

**Next experiment:** None. Close leftover 135169446 H0.

**Attempts:**
- Cycle 533; candidate `bd6a57768829c0764681263cf79e99643fd4aae5`; champion `ffd575c14ebd2a9da555e6bc8f786766fd428bed`; decision non_generalizable.
  [Implementation](ideas/sella-enyne-cc-stretch-h0/bd6a57768829c0764681263cf79e99643fd4aae5/algo.py).
  [Training evidence](evaluation_results/cycle-533-train.json).
  [Validation evidence](evaluation_results/cycle-533-valid.json).
  [Narrative](full_log.md#cycle-533--n-oxide-n18-plus-ch2clc2-stretch-on-n18).

## sella-fluoro-noxide-no-h0: 0.10 Ha/Bohr² fluoro bis-N-oxide N–O stretch stacked on N-oxide n<18
Status: closed

**Hypothesis:** Leftover 252656292 unused N–O stretches of the fused bis-N-oxide are the limiter after O–N–C extraed. Aryl-F plus exactly-two cap stacked on cycle-522 N-oxide would pass valid δ.

**Outcome and uncertainty:** Cycle 534 non_generalizable. Train identical to 522. Valid leftover 252656292 extras 18→19, energy-safe, same extra as cycle 404.

**Reason to revisit:** Not as N–O stretch H0. Soft H0 extras leftover. Do not interpolate 0.15.

**Next experiment:** None. Close leftover 252656292 H0.

**Attempts:**
- Cycle 534; candidate `c116f1d886c5ba6f8691332ee7831a7e1b9f5bcc`; champion `ffd575c14ebd2a9da555e6bc8f786766fd428bed`; decision non_generalizable.
  [Implementation](ideas/sella-fluoro-noxide-no-h0/c116f1d886c5ba6f8691332ee7831a7e1b9f5bcc/algo.py).
  [Training evidence](evaluation_results/cycle-534-train.json).
  [Validation evidence](evaluation_results/cycle-534-valid.json).
  [Narrative](full_log.md#cycle-534--n-oxide-n18-plus-fluoro-bis-n-oxide-no-stretch-on-30n80).

## sella-dimer-acetate-ester-cco-h0: 0.10 Ha phenol-gated acetate-ester C–C–O on dimers
Status: closed

**Hypothesis:** Dimer leftover esters–phenol (methyl acetate, extra=2, n_ref=33) is limited by unused C–C–O at the acetate carbonyl after champion phenol C–O–H. Phenol + alkoxy + methyl-acyl gates skip train formate–phenol. Stacked on cycle-522 N-oxide, leftover −2 would pass valid δ.

**Outcome and uncertainty:** Cycle 535 non_generalizable. Train identical to 522. Valid bit-identical; leftover untagged (1-coord carbonyl misses H-bonded =O). Cycle 536 non_generalizable. Valid still bit-identical after H-bonded =O repair. Cycle 537 non_generalizable. Train identical to 522. Valid leftover 35→36 (+1 twitch, Δ=−6.52e-5). Medium-band + neighbor-count + apply-first-2 tagged leftover; soft C–C–O is not the limiter (same undo class as cycle 474/477).

**Reason to revisit:** Not as acetate-ester C–C–O H0 on this leftover. Do not interpolate 0.13 or apply-first-1. Do not stack more H0 on esters–phenol.

**Next experiment:** None. Close this family. See leftover 135169446 primary-alcohol C–O stretch stacked on N-oxide n<18.

**Attempts:**
- Cycle 537; candidate `9adf7618715ce4a7576cc28a388320e5ca0c54fa`; champion `ffd575c14ebd2a9da555e6bc8f786766fd428bed`; decision non_generalizable.
  [Implementation](ideas/sella-dimer-acetate-ester-cco-h0/9adf7618715ce4a7576cc28a388320e5ca0c54fa/algo.py).
  [Training evidence](evaluation_results/cycle-537-train.json).
  [Validation evidence](evaluation_results/cycle-537-valid.json).
  [Narrative](full_log.md#cycle-537--acetate-ester-cco-with-connecting-neighbor-and-medium-band-repair).
- Cycle 536; candidate `889903462a5c03f4d07a834236a535295b72c3e4`; champion `ffd575c14ebd2a9da555e6bc8f786766fd428bed`; decision non_generalizable.
  [Implementation](ideas/sella-dimer-acetate-ester-cco-h0/889903462a5c03f4d07a834236a535295b72c3e4/algo.py).
  [Training evidence](evaluation_results/cycle-536-train.json).
  [Validation evidence](evaluation_results/cycle-536-valid.json).
  [Narrative](full_log.md#cycle-536--acetate-ester-cco-allowing-h-bonded-carbonyl-oxygen).
- Cycle 535; candidate `bd5c839c5d4dc6f45efba326899d21398abb1754`; champion `ffd575c14ebd2a9da555e6bc8f786766fd428bed`; decision non_generalizable.
  [Implementation](ideas/sella-dimer-acetate-ester-cco-h0/bd5c839c5d4dc6f45efba326899d21398abb1754/algo.py).
  [Training evidence](evaluation_results/cycle-535-train.json).
  [Validation evidence](evaluation_results/cycle-535-valid.json).
  [Narrative](full_log.md#cycle-535--n-oxide-n18-plus-phenol-gated-acetate-ester-cco-on-dimers).

## sella-oligosilane-sisi-stretch-h0: 0.25 Ha/Bohr² Si–Si stretch on 12–30 oligosilanes stacked on aryl-CF3 geodesic
Status: closed

**Hypothesis:** Leftover 135095297 unused Si–Si stretches still use Fischer. Soft 0.25 on 1–4 Si–Si at Si/H-only ≥4 Si stacked on cycle 607 may add leftover −1.

**Outcome and uncertainty:** Cycle 614 discard, Δ=+3.77e-5. Leftover 440723624 save kept. Leftover 135095297 extras 37→38. Soft Si–Si extras leftover.

**Reason to revisit:** Not as an oligosilane Si–Si stretch scale. Keep stacking from cycle 607 without this gate.

**Next experiment:** None. See cycle 607 plus leftover 135041973 iterative Cartesian B⁺ on n<12 sulfoxides.

**Attempts:**
- Cycle 614; candidate `0571cdc5e1a5b3137fce2e94cc2f3be77604220a`; champion `4eb6cafb44eeb92e19753e82045a8573ed8f96cb`; decision discard.
  [Training evidence](evaluation_results/cycle-614-train.json).
  [Narrative](full_log.md#cycle-614--cycle-607-plus-025-habohr-oligosilane-sisi-stretch-on-12-30).

## sella-oligosilane-skip-gdiis: skip GDIIS on n<18 oligosilanes stacked on aryl-CF3 geodesic
Status: closed

**Hypothesis:** Leftover 135095297 at 37 can interpolate after GDIIS@20. Geodesic extras leftover; iterative twitches. Skip GDIIS stacked on cycle 607 may add leftover −1.

**Outcome and uncertainty:** Cycle 613 discard, Δ=+9.69e-5. Bit-identical to 607. Leftover 135095297 stays 37. GDIIS never accepts on leftover.

**Reason to revisit:** Not as an n<18 oligosilane GDIIS skip. Keep stacking from cycle 607 without this gate.

**Next experiment:** None. See cycle 607 plus 0.25 Ha/Bohr² Si–Si stretch on leftover 135095297.

**Attempts:**
- Cycle 613; candidate `27f4bfd32867be4789d6ed421f866138e370e7ea`; champion `4eb6cafb44eeb92e19753e82045a8573ed8f96cb`; decision discard.
  [Training evidence](evaluation_results/cycle-613-train.json).
  [Narrative](full_log.md#cycle-613--cycle-607-plus-skip-gdiis-on-n18-oligosilanes).

## sella-oligosilane-iterative-stepper: iterative Cartesian B⁺ on n<18 oligosilanes stacked on aryl-CF3 geodesic
Status: closed

**Hypothesis:** Cycle 608 geodesic extras leftover 135095297. Iterative Cartesian B⁺ is the Newton realization opposite that geodesic. Stacked on cycle 607 it may add leftover −1 without extraing the silane.

**Outcome and uncertainty:** Cycle 611 discard, Δ=+9.69e-5. n_steps-identical to 607. Leftover 135095297 energy twitch at 37. Iterative B⁺ fires and is not the limiter.

**Reason to revisit:** Not as an n<18 oligosilane iterative interpolant. Keep stacking from cycle 607 without this gate.

**Next experiment:** None. See cycle 607 plus leftover 135094613 iterative Cartesian B⁺ on n<12 P4S6.

**Attempts:**
- Cycle 611; candidate `465362022c610b95aaa6cb33149603866d6fbcf8`; champion `4eb6cafb44eeb92e19753e82045a8573ed8f96cb`; decision discard.
  [Training evidence](evaluation_results/cycle-611-train.json).
  [Narrative](full_log.md#cycle-611--cycle-607-plus-iterative-cartesian-b-on-n18-oligosilanes).

## sella-oligosilane-exact-geodesic: exact geodesic on n<18 oligosilanes stacked on aryl-CF3 geodesic
Status: closed

**Hypothesis:** Cycle 607 near-miss needs another −1. Leftover 135095297 is Si/H-only pentasilane. Exact geodesic on n<18 oligosilanes (≥4 Si) stacked on 607 clears train δ.

**Outcome and uncertainty:** Cycle 608 discard, Δ=−2.15e-5. Leftover 440723624 save kept. Leftover 135095297 extras 37→39. Oligosilane geodesic extras the silane.

**Reason to revisit:** Not as n<18 oligosilane geodesic. Keep stacking from cycle 607 without this gate.

**Next experiment:** None. See cycle 607 plus iterative Cartesian B⁺ on n<18 oligosilanes.

**Attempts:**
- Cycle 608; candidate `cf152ae088c3bae04ac621196a6bee0ede2813d3`; champion `4eb6cafb44eeb92e19753e82045a8573ed8f96cb`; decision discard.
  [Training evidence](evaluation_results/cycle-608-train.json).
  [Narrative](full_log.md#cycle-608--cycle-607-plus-exact-geodesic-on-n18-oligosilanes).

## sella-aryl-cf3-exact-geodesic: exact geodesic on 30–80 aryl-CF3
Status: incorporated

**Hypothesis:** Leftover 440723624 extraed under CF3 F–C–C / ipso C–C–C. Exact geodesic on connected 30≤n<80 aryl-CF3 (4-coord C {F,F,F,C} bonded to 3-coord C) cheapens leftover without paliperidone or alkyl-CF3.

**Outcome and uncertainty:** Cycle 607 discard, Δ=+9.69e-5 below δ. Only save leftover 440723624 23→22. Cycle 619 valid showed 385453608 20→17 undone by leftover amides–pyrrole +5. Cycle 620 **keep** stacked oxygen-free S–P–S: train leftover 440723624 23→22 plus leftover 135094613 17→16, Δ=+2.302e-4; valid 385453608 20→17, Δ=+3.226e-4. New champion `17ef5a0334bd45194f97546bcbbb2ffc13b33951`.

**Reason to revisit:** Incorporated. Do not restack dimer amide+pyridine H0 (cycle 619 extraed leftover amides–pyrrole).

**Next experiment:** leftover 433943527 dummy-angle/dihedral/stretch/MIS are closed. Cycle 636 n<12 sulfoxide `wd=0.70` is a train-only near-miss; stack it with leftover 252656292.

**Attempts:**
- Cycle 620; candidate `17ef5a0334bd45194f97546bcbbb2ffc13b33951`; champion `4eb6cafb44eeb92e19753e82045a8573ed8f96cb`; decision keep.
  [Implementation](ideas/sella-n12-p4s6-sps-h0-stack/17ef5a0334bd45194f97546bcbbb2ffc13b33951/algo.py).
  [Training evidence](evaluation_results/cycle-620-train.json).
  [Validation evidence](evaluation_results/cycle-620-valid.json).
  [Narrative](full_log.md#cycle-620--cycle-607-plus-010-ha-oxygen-free-sps-on-n12-p4s6).
- Cycle 607; candidate `b72e2a5e70cdda1872f3b4073cc8ded8838c2beb`; champion `4eb6cafb44eeb92e19753e82045a8573ed8f96cb`; decision discard.
  [Implementation](ideas/sella-aryl-cf3-exact-geodesic/b72e2a5e70cdda1872f3b4073cc8ded8838c2beb/algo.py).
  [Training evidence](evaluation_results/cycle-607-train.json).
  [Narrative](full_log.md#cycle-607--exact-geodesic-on-30-80-aryl-cf3).
- Cycle 619; candidate `e7e40e36c5480368705fa348cc6e8c4643126635`; champion `4eb6cafb44eeb92e19753e82045a8573ed8f96cb`; decision non_generalizable.
  [Implementation](ideas/sella-aryl-cf3-amide-cnc-stack/e7e40e36c5480368705fa348cc6e8c4643126635/algo.py).
  [Training evidence](evaluation_results/cycle-619-train.json).
  [Validation evidence](evaluation_results/cycle-619-valid.json).
  [Narrative](full_log.md#cycle-619--cycle-607-plus-010-ha-pyridine-gated-amide-cnc-on-dimers).

## sella-isocyanide-ipso-ccc-h0: 0.10 Ha isocyanide ipso C–C–C on 30–80
Status: closed

**Hypothesis:** Cycle 604 pyridine C–C–N extraed leftover 363892164. Unused C–C–C at the isocyanide ipso carbon {C,C,N} is a different bend.

**Outcome and uncertainty:** Cycle 606 discard, Δ=0. Leftover 363892164 energy twitch at 42. Ipso C–C–C fires and is not the n_steps limiter.

**Reason to revisit:** Not as an isocyanide ipso C–C–C H0 scale. Leftover 363892164 ring H0 extras or twitches.

**Next experiment:** None. See leftover 440723624 exact geodesic on 30–80 aryl-CF3.

**Attempts:**
- Cycle 606; candidate `b6db80da7654619889bd00e0318eb090dc4794ae`; champion `4eb6cafb44eeb92e19753e82045a8573ed8f96cb`; decision discard.
  [Training evidence](evaluation_results/cycle-606-train.json).
  [Narrative](full_log.md#cycle-606--010-ha-isocyanide-ipso-ccc-on-30-80).

## sella-thiosulfonate-so-stretch-h0: 0.25 Ha/Bohr² thiosulfonate S–O stretch on 12–30
Status: closed

**Hypothesis:** Leftover 433943527 unused S–O stretches still use Fischer. Soft 0.25 on 1–2 S–O at 4-coordinate thiosulfonate S may cheapen leftover after S–S twitch.

**Outcome and uncertainty:** Cycle 605 discard, Δ=−1.066e-4. Only extra leftover 433943527 21→22, energy-safe. Soft S–O extras leftover.

**Reason to revisit:** Not as a thiosulfonate S–O stretch scale. Leftover 433943527 stretch H0 is closed.

**Next experiment:** None. See leftover 363892164 isocyanide ipso C–C–C on 30–80.

**Attempts:**
- Cycle 605; candidate `383d9ed4ce32acf2e9e5581f11f02ab142229e33`; champion `4eb6cafb44eeb92e19753e82045a8573ed8f96cb`; decision discard.
  [Training evidence](evaluation_results/cycle-605-train.json).
  [Narrative](full_log.md#cycle-605--025-habohr-thiosulfonate-so-stretch-on-12-30).

## sella-isocyanide-pyridine-ccn-h0: 0.10 Ha isocyanide-gated pyridine C–C–N on 30–80
Status: closed

**Hypothesis:** Leftover 363892164 unused 3-coordinate pyridine C–C–N still uses Fischer. Gate to 30–80 isocyanides.

**Outcome and uncertainty:** Cycle 603 discard, Δ=0, bit-identical. 1–2 cap skipped leftover because isocyanide ipso C–C–N also matched (3 candidates).

**Reason to revisit:** Repair by requiring both N carbons 3-coordinate so leftover pyridine C–C–N is 1–2.

**Next experiment:** None. Cycle 604 extras leftover.

**Attempts:**
- Cycle 604; candidate `6de13bc3a58e7b1600489792c33fa9a00ac7e942`; champion `4eb6cafb44eeb92e19753e82045a8573ed8f96cb`; decision discard.
  [Training evidence](evaluation_results/cycle-604-train.json).
  [Narrative](full_log.md#cycle-604--pyridine-ccn-requiring-3-coordinate-n-carbons).
- Cycle 603; candidate `b9a9d506caebd3bcd8482459d30b15ec2b06e44a`; champion `4eb6cafb44eeb92e19753e82045a8573ed8f96cb`; decision discard.
  [Training evidence](evaluation_results/cycle-603-train.json).
  [Narrative](full_log.md#cycle-603--010-ha-isocyanide-gated-pyridine-ccn-on-30-80).

## sella-thiosulfonate-ss-stretch-h0: 0.25 Ha/Bohr² thiosulfonate S–S stretch on 12–30
Status: closed

**Hypothesis:** Leftover 433943527 unused S–S stretch still uses Fischer (~0.36 Ha/Bohr²). Soft 0.25 on exactly one S–S at 4-coordinate thiosulfonate S {O,O,S,C} may cheapen leftover after interpolant twitch.

**Outcome and uncertainty:** Cycle 602 discard, Δ=0. Leftover 433943527 energy twitch at 21. Stretch fires and is not the n_steps limiter. 135181905 untagged.

**Reason to revisit:** Not as a thiosulfonate S–S stretch scale. Leftover 433943527 stretch H0 is closed with angle H0 and interpolants.

**Next experiment:** None. See leftover 363892164 0.10 Ha isocyanide-gated pyridine C–C–N on 30–80.

**Attempts:**
- Cycle 602; candidate `0aef8c41b19bcf33db2f741e37b748ed06e10306`; champion `4eb6cafb44eeb92e19753e82045a8573ed8f96cb`; decision discard.
  [Training evidence](evaluation_results/cycle-602-train.json).
  [Narrative](full_log.md#cycle-602--025-habohr-thiosulfonate-ss-stretch-on-12-30).

## sella-thiosulfonate-iterative-stepper: iterative Cartesian B⁺ on 12–30 thiosulfonates
Status: closed

**Hypothesis:** Cycle 590 geodesic twitched leftover 433943527. Iterative Cartesian B⁺ is the Newton realization of the same internal step.

**Outcome and uncertainty:** Cycle 601 discard, Δ=0. Leftover 433943527 energy twitch at 21. Realization fires and is not the n_steps limiter.

**Reason to revisit:** Not as a thiosulfonate iterative_stepper interpolation. Cycle 601 vs `4eb6caf` and cycle 864 vs `133f73f` (799 stack) both energy-twitch leftover 433943527 at 21. Combined with geodesic twitch and GDIIS no-op, leftover 433943527 Binv interpolants are closed.

**Next experiment:** None. Helgaker 0.01 Eh |λ| floor on connected 12–30 thiosulfonates stacked with cycle 799 vs leftover 433943527.

**Attempts:**
- Cycle 601; candidate `1660c801f3d7c2790c7aa4fcd4a3f101ff47032c`; champion `4eb6cafb44eeb92e19753e82045a8573ed8f96cb`; decision discard.
  [Training evidence](evaluation_results/cycle-601-train.json).
  [Narrative](full_log.md#cycle-601--iterative-cartesian-b-on-12-30-thiosulfonates).
- Cycle 864; candidate `5602b7fb5acb6c203c8c0bb17ae36f22ce4e5394`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-864-train.json).
  [Validation evidence](evaluation_results/cycle-864-valid.json).
  [Narrative](full_log.md#cycle-864--cycle-799-plus-iterative-b-on-connected-1230-thiosulfonates).

## sella-thiosulfonate-skip-gdiis: skip GDIIS on 12–30 thiosulfonates
Status: closed

**Hypothesis:** Leftover 433943527 extra=1 at 21 sits one step after GDIIS@20. Skip GDIIS on connected 12≤n<30 thiosulfonates may drop the extra. Cycle 589 GDIIS@15 was a no-op.

**Outcome and uncertainty:** Cycle 600 discard, Δ=0. Bit-identical to champion. Leftover is not GDIIS-limited.

**Reason to revisit:** Not as a thiosulfonate GDIIS skip. Combined with GDIIS@15 no-op, leftover 433943527 GDIIS interpolants are closed.

**Next experiment:** None. See iterative Cartesian B⁺ on 12–30 thiosulfonates vs leftover 433943527.

**Attempts:**
- Cycle 600; candidate `8ac6be202e9ec10aee6a9c2caa672752334f38f1`; champion `4eb6cafb44eeb92e19753e82045a8573ed8f96cb`; decision discard.
  [Training evidence](evaluation_results/cycle-600-train.json).
  [Narrative](full_log.md#cycle-600--skip-gdiis-on-12-30-thiosulfonates).

## sella-medium-ammonium-ncc-h0: 0.10 Ha choline N–C–C on connected 12–30
Status: closed

**Hypothesis:** Leftover 433943527 unused N–C–C at the choline methylene {N, C, H, H} with trimethylammonium N is the limiter after CSS/OSS/ammonium C–N–C extra or twitch. GAFF `c3-c3-n4` ≈ 0.10 Ha.

**Outcome and uncertainty:** Cycle 599 discard, Δ=0. Leftover 433943527 energy twitch at 21. N–C–C fires and is not the n_steps limiter. 135239137 untagged.

**Reason to revisit:** Not as a choline N–C–C H0 scale. Leftover 433943527 valence-angle H0 including the linker is closed.

**Next experiment:** None. See skip GDIIS on 12–30 thiosulfonates vs leftover 433943527 extra at n=21.

**Attempts:**
- Cycle 599; candidate `8cfc5499d0f42e5a3e58d015891534673be5536c`; champion `4eb6cafb44eeb92e19753e82045a8573ed8f96cb`; decision discard.
  [Training evidence](evaluation_results/cycle-599-train.json).
  [Narrative](full_log.md#cycle-599--010-ha-choline-ncc-on-connected-12-30).

## sella-n12-p4s6-iterative-stepper: iterative Cartesian B⁺ on n<12 P4S6 stacked on aryl-CF3 geodesic
Status: closed

**Hypothesis:** Cycle 598 geodesic twitched leftover 135094613. Iterative Cartesian B⁺ is the Newton realization. Stacked on cycle 607 it may add leftover −1.

**Outcome and uncertainty:** Cycle 612 discard, Δ=+9.69e-5. n_steps-identical to 607. Leftover 135094613 energy twitch at 17. Iterative B⁺ fires and is not the limiter.

**Reason to revisit:** Not as an n<12 P4S6 iterative interpolant. Keep stacking from cycle 607 without this gate.

**Next experiment:** None. See cycle 607 plus skip GDIIS on n<18 oligosilanes vs leftover 135095297.

**Attempts:**
- Cycle 612; candidate `a4b893ca73d3b265b4c99b66cbe33cae04186a8b`; champion `4eb6cafb44eeb92e19753e82045a8573ed8f96cb`; decision discard.
  [Training evidence](evaluation_results/cycle-612-train.json).
  [Narrative](full_log.md#cycle-612--cycle-607-plus-iterative-cartesian-b-on-n12-p4s6).

## sella-n12-p4s6-exact-geodesic: exact geodesic on n<12 P4S6
Status: closed

**Hypothesis:** Leftover 135094613 is not valence-H0 limited (cycle 592 extras). Exact geodesic on connected n<12 3-coordinate P {S,S,S} may cheapen leftover without leftover 252089162, leftover 135041973, or 135095649.

**Outcome and uncertainty:** Cycle 598 discard, Δ=0. Leftover 135094613 energy twitch at 17. Geodesic fires and is not the n_steps limiter.

**Reason to revisit:** Not as an n<12 P4S6 geodesic interpolant. Combined with valence-angle extras, leftover 135094613 is not geodesic-limited.

**Next experiment:** None. See leftover 433943527 0.10 Ha choline N–C–C on connected 12–30.

**Attempts:**
- Cycle 598; candidate `ab2eb91772695dbbba292227bce8285b86fb2e12`; champion `4eb6cafb44eeb92e19753e82045a8573ed8f96cb`; decision discard.
  [Training evidence](evaluation_results/cycle-598-train.json).
  [Narrative](full_log.md#cycle-598--exact-geodesic-on-n12-p4s6).

## sella-n12-sulfoxide-so-stretch-h0: 0.25 Ha/Bohr² sulfoxide S=O stretch stacked on aryl-CF3 geodesic
Status: closed

**Hypothesis:** Leftover 135041973 unused S=O still uses Fischer. Soft 0.25 on 1–2 terminal S=O at 3-coord S {C,C,O} stacked on cycle 607 may add leftover −1.

**Outcome and uncertainty:** Cycle 616 discard, Δ=−8.08e-5. Leftover 440723624 save kept. Leftover 135041973 extras 13→14. Soft S=O extras leftover.

**Reason to revisit:** Not as a sulfoxide S=O stretch scale. Keep stacking from cycle 607 without this gate.

**Next experiment:** None. See cycle 607 plus leftover 363892164 limiter dummy-wd 0.7 on 30–80 isocyanides.

**Attempts:**
- Cycle 616; candidate `42d8c2cfbc7175025d3195d3ecbfede289eae771`; champion `4eb6cafb44eeb92e19753e82045a8573ed8f96cb`; decision discard.
  [Training evidence](evaluation_results/cycle-616-train.json).
  [Narrative](full_log.md#cycle-616--cycle-607-plus-025-habohr-sulfoxide-so-stretch-on-n12).

## sella-n12-sulfoxide-iterative-stepper: iterative Cartesian B⁺ on n<12 sulfoxides stacked on aryl-CF3 geodesic
Status: closed

**Hypothesis:** Cycle 597 geodesic twitched leftover 135041973. Iterative Cartesian B⁺ stacked on cycle 607 may add leftover −1.

**Outcome and uncertainty:** Cycle 615 discard, Δ=+9.69e-5. n_steps-identical to 607. Leftover 135041973 energy twitch at 13. Iterative B⁺ fires and is not the limiter.

**Reason to revisit:** Not as an n<12 sulfoxide iterative interpolant. Keep stacking from cycle 607 without this gate.

**Next experiment:** None. See cycle 607 plus leftover 135041973 S=O stretch H0.

**Attempts:**
- Cycle 615; candidate `30e9627641ab851ad2994e89516fffdcaa8d19a8`; champion `4eb6cafb44eeb92e19753e82045a8573ed8f96cb`; decision discard.
  [Training evidence](evaluation_results/cycle-615-train.json).
  [Narrative](full_log.md#cycle-615--cycle-607-plus-iterative-cartesian-b-on-n12-sulfoxides).

## sella-n12-sulfoxide-exact-geodesic: exact geodesic on n<12 sulfoxides
Status: closed

**Hypothesis:** Leftover 135041973 is not valence-H0 limited (cycle 593 extras). Exact geodesic on connected n<12 3-coordinate sulfoxide S {C,C,O} may cheapen leftover without leftover 252089162 or leftover 135094613.

**Outcome and uncertainty:** Cycle 597 discard, Δ=0. Leftover 135041973 energy twitch at 13. Geodesic fires and is not the n_steps limiter.

**Reason to revisit:** Not as an n<12 sulfoxide geodesic interpolant. Combined with valence-angle extras, leftover 135041973 is not geodesic-limited.

**Next experiment:** None. See leftover 433943527 0.10 Ha choline N–C–C on connected 12–30.

**Attempts:**
- Cycle 597; candidate `ca996938d792c6b5a59ced4031e3135b37851ceb`; champion `4eb6cafb44eeb92e19753e82045a8573ed8f96cb`; decision discard.
  [Training evidence](evaluation_results/cycle-597-train.json).
  [Narrative](full_log.md#cycle-597--exact-geodesic-on-n12-sulfoxides).

## sella-n12-sulfoxide-wd-070: wd=0.70 on connected n<12 sulfoxides
Status: incorporated

**Hypothesis:** Leftover 135041973 extra=1 (`n_ref=12`) is not valence-H0 or geodesic limited. Lowering MaxInternalStep `wd` to 0.70 on connected n<12 3-coordinate S {C,C,O} enlarges dihedral steps and cheapens leftover.

**Outcome and uncertainty:** Cycle 636 non_generalizable. Train Δ=+1.777e-4, only leftover 135041973 13→12, energy-safe. Valid Δ=0, bit-identical. Cycle 637 **keep** stacked `wd=0.70` on 30–80 pyrrolidine N-oxides: train leftover 135041973 13→12, Δ=+1.777e-4; valid leftover 252656292 18→17, Δ=+1.265e-4. New champion `9eab9c6116bcedf7ce6da8351024bd3e5e418f3a`.

**Reason to revisit:** Incorporated. Do not drop the n<12 sulfoxide gate or the pyrrolidine+N-oxide 30–80 gate.

**Next experiment:** Scan leftover extras vs cycle 643. leftover 433943527 extra=1 still passes train δ; stretch/angle/dummy/interpolant/MIS closed. leftover 363892164 extra=+11 needs −2. leftover 135169446 extra=2 still passes valid δ on −1.

**Attempts:**
- Cycle 636; candidate `666c3f55840cdf70da073c962f41a819d9690c81`; champion `17ef5a0334bd45194f97546bcbbb2ffc13b33951`; decision non_generalizable.
  [Implementation](ideas/sella-n12-sulfoxide-wd-070/666c3f55840cdf70da073c962f41a819d9690c81/algo.py).
  [Training evidence](evaluation_results/cycle-636-train.json).
  [Validation evidence](evaluation_results/cycle-636-valid.json).
  [Narrative](full_log.md#cycle-636--wd070-on-n12-sulfoxides).
- Cycle 637; candidate `9eab9c6116bcedf7ce6da8351024bd3e5e418f3a`; champion `17ef5a0334bd45194f97546bcbbb2ffc13b33951`; decision keep.
  [Implementation](ideas/sella-n12-sulfoxide-wd-070/9eab9c6116bcedf7ce6da8351024bd3e5e418f3a/algo.py).
  [Training evidence](evaluation_results/cycle-637-train.json).
  [Validation evidence](evaluation_results/cycle-637-valid.json).
  [Narrative](full_log.md#cycle-637--cycle-636-plus-wd070-on-30-80-pyrrolidine-n-oxides).

## sella-gemcf2-alkyne-cc-h0: 0.25 Ha/Bohr² gem-CF2 alkyne C–C on 30–80
Status: revisiting

**Hypothesis:** Leftover 363892164 unused aryl–alkyne C≡C attached to gem-difluoro cyclobutane is the limiter after isocyanide C–N extraed leftover.

**Outcome and uncertainty:** Cycle 639 discard, Δ=0, bit-identical (alkyne does not attach at CF2). Cycle 640 repair tagged leftover and extraed it 42→43, Δ=−6.88e-5. Soft alkyne C–C extras leftover.

**Reason to revisit:** Not as an alkyne C–C stretch scale. Leftover 363892164 stretch H0 extras (isocyanide C–N and alkyne C–C). Cycle 644 `wd=0.70` also extras leftover 42→43.

**Next experiment:** leftover 363892164 `delta_min=0.18` after 20 on connected 30–80 isocyanides. Stretch/`wd` extra leftover. Leftover −2 clears train δ.

**Attempts:**
- Cycle 639; candidate `625b54022b0e40612f0e716bd33f2043cd320291`; champion `9eab9c6116bcedf7ce6da8351024bd3e5e418f3a`; decision discard.
  [Implementation](ideas/sella-gemcf2-alkyne-cc-h0/625b54022b0e40612f0e716bd33f2043cd320291/algo.py).
  [Training evidence](evaluation_results/cycle-639-train.json).
  [Narrative](full_log.md#cycle-639--025-habohr-gem-cf2-alkyne-cc-on-30-80).

## sella-n18-oligosilane-wd-070: wd=0.70 on connected n<18 oligosilanes
Status: revisiting

**Hypothesis:** Leftover 135095297 extra=1 is not Si–Si–Si limited. Lowering MaxInternalStep `wd` to 0.70 on connected n<18 Si/H-only oligosilanes (≥4 Si) cheapens leftover dihedral steps.

**Outcome and uncertainty:** Cycle 642 non_generalizable. Train Δ=+3.554e-4, leftover 135095297 37→31 (−6), energy-safe. leftover 135041973 stays 12. Valid Δ=0, bit-identical (no valid n<18 Si/H-only oligosilane). Cycle 643 **keep** stacked `wd=0.70` on n<18 allenes: train leftover 135095297 37→31, Δ=+3.554e-4; valid leftover 135169446 21→20, Δ=+1.195e-4. New champion `b099d91933b322e250f811c9525ba5e1d39a284a`.

**Reason to revisit:** Incorporated. Do not drop the n<18 oligosilane gate or the n<18 allene gate.

**Next experiment:** Scan leftover extras vs cycle 643. leftover 433943527 extra=1 still passes train δ; stretch/angle/dummy/interpolant/MIS closed. leftover 363892164 extra=+11 needs −2. leftover 135169446 extra=2 still passes valid δ on −1.

**Attempts:**
- Cycle 642; candidate `a576b14edd5d376628f548bd23a1a924634be4a8`; champion `9eab9c6116bcedf7ce6da8351024bd3e5e418f3a`; decision non_generalizable.
  [Implementation](ideas/sella-n18-oligosilane-wd-070/a576b14edd5d376628f548bd23a1a924634be4a8/algo.py).
  [Training evidence](evaluation_results/cycle-642-train.json).
  [Validation evidence](evaluation_results/cycle-642-valid.json).
  [Narrative](full_log.md#cycle-642--wd070-on-n18-oligosilanes).
- Cycle 643; candidate `b099d91933b322e250f811c9525ba5e1d39a284a`; champion `9eab9c6116bcedf7ce6da8351024bd3e5e418f3a`; decision keep.
  [Implementation](ideas/sella-n18-allene-wd-070/b099d91933b322e250f811c9525ba5e1d39a284a/algo.py).
  [Training evidence](evaluation_results/cycle-643-train.json).
  [Validation evidence](evaluation_results/cycle-643-valid.json).
  [Narrative](full_log.md#cycle-643--cycle-642-plus-wd070-on-n18-allenes).

## sella-n18-allene-wd-070: wd=0.70 on connected n<18 allenes
Status: incorporated

**Hypothesis:** Cycle 642 oligosilane `wd=0.70` is a train-only near-miss. Valid leftover 135169446 extra=3 is a dummy-linear allene. The same `wd=0.70` on connected n<18 allenes (2-coord C with two 3-coord C) cheapens leftover dihedrals without train oligosilanes.

**Outcome and uncertainty:** Cycle 643 **keep** vs `9eab9c6`. Train Δ=+3.554e-4, only leftover 135095297 37→31 (from 642). Valid Δ=+1.195e-4, only leftover 135169446 21→20. Energy-safe. New champion `b099d91933b322e250f811c9525ba5e1d39a284a`.

**Reason to revisit:** Incorporated. Do not drop the n<18 allene `wd=0.70` gate. Cycle 784 geodesic extras leftover 20→21. Cycle 785 `wd=0.60` energy-twitch leftover at 20. Cycle 674/887 `wo=0.70` never binds leftover dummy extras. Cycle 792 TrustRegion-from-start extras leftover 20→26. Cycle 888 skip dummy-limiter extras leftover 20→21; cycle 889 dummy-limiter 0.90 extras leftover 20→21. Dummy-limiter 0.8 is n_steps-optimal. Cycle 890 flowchart@17 extras leftover 20→21. Cycle 891 TrustRegion after 18 is a no-op at leftover 20. Cycle 892 GEDIIS after 16 extras leftover 20→22.

**Next experiment:** Newton-metric two-point GDIIS after 16 on `_has_allene` stacked with cycle 885 vs `e478d0c`. Cycle 834 after 15 never accepted leftover at 20. leftover 20→19 would keep.

**Attempts:**
- Cycle 643; candidate `b099d91933b322e250f811c9525ba5e1d39a284a`; champion `9eab9c6116bcedf7ce6da8351024bd3e5e418f3a`; decision keep.
  [Implementation](ideas/sella-n18-allene-wd-070/b099d91933b322e250f811c9525ba5e1d39a284a/algo.py).
  [Training evidence](evaluation_results/cycle-643-train.json).
  [Validation evidence](evaluation_results/cycle-643-valid.json).
  [Narrative](full_log.md#cycle-643--cycle-642-plus-wd070-on-n18-allenes).
- Cycle 785; candidate `bd891ca6ab8c5284dc33e8b9ef99cd2e15c2063f`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-785-train.json).
  [Validation evidence](evaluation_results/cycle-785-valid.json).
  [Narrative](full_log.md#cycle-785--cycle-783-plus-wd060-on-n18-allenes).
- Cycle 887; candidate `fadb3e59a947a633d99880c79f19624edd5cca58`; champion `e478d0c8c1a504a33dfa428d1e449f336c4b031f`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-887-train.json).
  [Validation evidence](evaluation_results/cycle-887-valid.json).
  [Narrative](full_log.md#cycle-887--cycle-885-plus-wo070-on-n18-allenes).
- Cycle 888; candidate `fb9e98cf379b62aedc9a0d59686ddc5a43bd9a95`; champion `e478d0c8c1a504a33dfa428d1e449f336c4b031f`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-888-train.json).
  [Validation evidence](evaluation_results/cycle-888-valid.json).
  [Narrative](full_log.md#cycle-888--cycle-885-plus-skip-dummy-limiter-on-n18-allenes).
- Cycle 889; candidate `db62ff3b4d6a5a0b53216e4b639c1f28b5cfe10d`; champion `e478d0c8c1a504a33dfa428d1e449f336c4b031f`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-889-train.json).
  [Validation evidence](evaluation_results/cycle-889-valid.json).
  [Narrative](full_log.md#cycle-889--cycle-885-plus-dummy-limiter-090-on-n18-allenes).
- Cycle 890; candidate `bad66261f73f1fc780d01975ffec3f36d6eb5033`; champion `e478d0c8c1a504a33dfa428d1e449f336c4b031f`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-890-train.json).
  [Validation evidence](evaluation_results/cycle-890-valid.json).
  [Narrative](full_log.md#cycle-890--cycle-885-plus-flowchart-after-17-on-n18-allenes).
- Cycle 891; candidate `8475e23bbf3b2dd587f6097165d1ce3b0b102f8e`; champion `e478d0c8c1a504a33dfa428d1e449f336c4b031f`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-891-train.json).
  [Validation evidence](evaluation_results/cycle-891-valid.json).
  [Narrative](full_log.md#cycle-891--cycle-885-plus-trustregion-after-18-on-n18-allenes).
- Cycle 892; candidate `e001c82b9a3e5369e03485df03497b2980847512`; champion `e478d0c8c1a504a33dfa428d1e449f336c4b031f`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-892-train.json).
  [Validation evidence](evaluation_results/cycle-892-valid.json).
  [Narrative](full_log.md#cycle-892--cycle-885-plus-gediis-after-16-on-n18-allenes).
- Cycle 893; candidate `0c3146ca9c1aab24e8cd9a1ea19b983f90e8d4f9`; champion `e478d0c8c1a504a33dfa428d1e449f336c4b031f`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-893-train.json).
  [Validation evidence](evaluation_results/cycle-893-valid.json).
  [Narrative](full_log.md#cycle-893--cycle-885-plus-newton-metric-gdiis-after-16-on-n18-allenes).
- Cycle 894; candidate `dc1ee1ee923800f0b690a6a8676095ded22ae36e`; champion `e478d0c8c1a504a33dfa428d1e449f336c4b031f`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-894-train.json).
  [Validation evidence](evaluation_results/cycle-894-valid.json).
  [Narrative](full_log.md#cycle-894--cycle-885-plus-rho_dec5-on-saturated-1830-alkanephenol).
- Cycle 895; candidate `efe3f6e83bfec5bedeb9c040cf12e193380dc56f`; champion `e478d0c8c1a504a33dfa428d1e449f336c4b031f`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-895-train.json).
  [Validation evidence](evaluation_results/cycle-895-valid.json).
  [Narrative](full_log.md#cycle-895--cycle-885-plus-predicted-uphill-step-halving-on-saturated-1830-alkanephenol).
- Cycle 896; candidate `788cbf166612eed84c57872cbd59bc318b528edb`; champion `e478d0c8c1a504a33dfa428d1e449f336c4b031f`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-896-train.json).
  [Validation evidence](evaluation_results/cycle-896-valid.json).
  [Narrative](full_log.md#cycle-896--cycle-885-plus-bfgs_auto-before-20-on-saturated-1830-alkanephenol).

## sella-n80-rfo-after-50: Banerjee RFO after 50 on connected n≥80
Status: incorporated

**Hypothesis:** Venetoclax extra=6 is a long floppy tail. Cycle 170 RFO@50 saved paliperidone. Restricting RFO after 50 to connected n≥80 cheapens paliperidone/venetoclax without leftover 363892164.

**Outcome and uncertainty:** Cycle 650 discard, Δ=+2.73e-5. Paliperidone 78→77; venetoclax energy twitch at 57. Cycle 651 **non_generalizable**: paliperidone 78→73 (train Δ=+1.367e-4), valid bit-identical. Cycle 655 nitro-CF3 `wd=0.70`: leftover 103943741 33→32. Cycle 662 **keep** stacked `wd=0.70` on 30–80 isoxazoles: paliperidone 78→73 plus valid leftover 434319675 47→46. New champion `2421717fc47361279dde7e172ead7b439968daaa`.

**Reason to revisit:** Incorporated in champion `2421717`. Do not drop n≥80 RFO@45 or 30–80 nitro-CF3/`isoxazole` `wd`.

**Next experiment:** Scan leftover extras vs cycle 662. leftover 363892164 extra=11; `wb` is a no-op. leftover 194689238 extra=2 still passes valid δ on −2. leftover 135169446 extra=2 still passes valid δ on −1. Close leftover 363892164 `wb`.

**Attempts:**
- Cycle 650; candidate `74dee916ed51a43593a381ff947d428757a6864b`; champion `b099d91933b322e250f811c9525ba5e1d39a284a`; decision discard.
  [Implementation](ideas/sella-n80-rfo-after-50/74dee916ed51a43593a381ff947d428757a6864b/algo.py).
  [Training evidence](evaluation_results/cycle-650-train.json).
  [Narrative](full_log.md#cycle-650--rfo-after-50-on-connected-n80).
- Cycle 651; candidate `bd1ad8aca2211d59e49bc7e683718a67d997435d`; champion `b099d91933b322e250f811c9525ba5e1d39a284a`; decision non_generalizable.
  [Implementation](ideas/sella-n80-rfo-after-50/bd1ad8aca2211d59e49bc7e683718a67d997435d/algo.py).
  [Training evidence](evaluation_results/cycle-651-train.json).
  [Validation evidence](evaluation_results/cycle-651-valid.json).
  [Narrative](full_log.md#cycle-651--rfo-after-45-on-connected-n80).
- Cycle 652; candidate `6951a348ee717a9f8acd2523315d391df202d1c2`; champion `b099d91933b322e250f811c9525ba5e1d39a284a`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-652-train.json).
  [Validation evidence](evaluation_results/cycle-652-valid.json).
  [Narrative](full_log.md#cycle-652--cycle-651-plus-skip-gdiis-on-n18-allenes).
- Cycle 653; candidate `fbeb1b4e509a81ae48623b201fb67a9252270a8d`; champion `b099d91933b322e250f811c9525ba5e1d39a284a`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-653-train.json).
  [Validation evidence](evaluation_results/cycle-653-valid.json).
  [Narrative](full_log.md#cycle-653--cycle-651-plus-wd070-on-30-80-benzothiazolines).
- Cycle 654; candidate `2894bf2d619aa48a9e857943d6bb271c3416f9d2`; champion `b099d91933b322e250f811c9525ba5e1d39a284a`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-654-train.json).
  [Validation evidence](evaluation_results/cycle-654-valid.json).
  [Narrative](full_log.md#cycle-654--cycle-651-plus-skip-gdiis-on-30-80-nitro-cf3).
- Cycle 655; candidate `f30299659579b71a705c9ac6f3449972ccc0e00f`; champion `b099d91933b322e250f811c9525ba5e1d39a284a`; decision non_generalizable.
  [Implementation](ideas/sella-n80-rfo-after-50/f30299659579b71a705c9ac6f3449972ccc0e00f/algo.py).
  [Training evidence](evaluation_results/cycle-655-train.json).
  [Validation evidence](evaluation_results/cycle-655-valid.json).
  [Narrative](full_log.md#cycle-655--cycle-651-plus-wd070-on-30-80-nitro-cf3).
- Cycle 656; candidate `93ac122f9c8e513d540a0312cf6f4718f579d9fd`; champion `b099d91933b322e250f811c9525ba5e1d39a284a`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-656-train.json).
  [Validation evidence](evaluation_results/cycle-656-valid.json).
  [Narrative](full_log.md#cycle-656--cycle-655-plus-skip-gdiis-on-30-80-nitro-cf3).
- Cycle 657; candidate `9ee6d456b329d24337a143a181101ef284e215ec`; champion `b099d91933b322e250f811c9525ba5e1d39a284a`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-657-train.json).
  [Validation evidence](evaluation_results/cycle-657-valid.json).
  [Narrative](full_log.md#cycle-657--cycle-655-plus-wa070-on-n18-allenes).
- Cycle 658; candidate `022e7836237cacfc572a971fbdc465d6ff0fa729`; champion `b099d91933b322e250f811c9525ba5e1d39a284a`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-658-train.json).
  [Validation evidence](evaluation_results/cycle-658-valid.json).
  [Narrative](full_log.md#cycle-658--cycle-655-plus-exact-geodesic-on-n18-allenes).
- Cycle 659; candidate `34d6dee47750181937cc7d04532ce497978d76cb`; champion `b099d91933b322e250f811c9525ba5e1d39a284a`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-659-train.json).
  [Validation evidence](evaluation_results/cycle-659-valid.json).
  [Narrative](full_log.md#cycle-659--cycle-655-plus-wa070-on-30-80-fused-benzothiazines).
- Cycle 660; candidate `faeaec048e115560c46c03afb66c181dbb1916d4`; champion `b099d91933b322e250f811c9525ba5e1d39a284a`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-660-train.json).
  [Validation evidence](evaluation_results/cycle-660-valid.json).
  [Narrative](full_log.md#cycle-660--cycle-655-plus-wd070-on-30-80-sulfoniums).
- Cycle 661; candidate `bb93e7cadf6045125cd596a92d24cee565cc0136`; champion `b099d91933b322e250f811c9525ba5e1d39a284a`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-661-train.json).
  [Validation evidence](evaluation_results/cycle-661-valid.json).
  [Narrative](full_log.md#cycle-661--cycle-655-plus-skip-gdiis-on-30-80-sulfoniums).
- Cycle 662; candidate `2421717fc47361279dde7e172ead7b439968daaa`; champion `b099d91933b322e250f811c9525ba5e1d39a284a`; decision keep.
  [Implementation](ideas/sella-isoxazole-wd-070/2421717fc47361279dde7e172ead7b439968daaa/algo.py).
  [Training evidence](evaluation_results/cycle-662-train.json).
  [Validation evidence](evaluation_results/cycle-662-valid.json).
  [Narrative](full_log.md#cycle-662--cycle-655-plus-wd070-on-30-80-isoxazoles).

## sella-isoxazole-wd-070: wd=0.70 on connected 30–80 isoxazoles
Status: incorporated

**Hypothesis:** Valid leftover 434319675 extra=1 is an isoxazole. Stacking `wd=0.70` on 2-coordinate O {2-coord N, 3-coord C} at 30≤n<80 with cycle 655's paliperidone/nitro-CF3 near-miss clears both δ gates.

**Outcome and uncertainty:** Cycle 662 **keep** vs `b099d91`. Train Δ=+1.841e-4 (paliperidone 78→73, 135267488 43→42). Valid Δ=+4.341e-4 (312525537 −2, 384575875 −1, leftover 434319675 −1, leftover 103943741 −1). Energy-safe. Matcher also tags 1,2,4-oxadiazole and oxime-ether O {N,C}. New champion `2421717`.

**Reason to revisit:** Incorporated. Do not drop 30–80 isoxazole `wd` or expand to n<30 / n≥80 without a new leftover.

**Next experiment:** Scan leftover extras vs cycle 662.

**Attempts:**
- Cycle 662; candidate `2421717fc47361279dde7e172ead7b439968daaa`; champion `b099d91933b322e250f811c9525ba5e1d39a284a`; decision keep.
  [Implementation](ideas/sella-isoxazole-wd-070/2421717fc47361279dde7e172ead7b439968daaa/algo.py).
  [Training evidence](evaluation_results/cycle-662-train.json).
  [Validation evidence](evaluation_results/cycle-662-valid.json).
  [Narrative](full_log.md#cycle-662--cycle-655-plus-wd070-on-30-80-isoxazoles).

## sella-oxadiazole-exact-geodesic: exact geodesic on 30–80 C-substituted 1,2,4-oxadiazoles
Status: revisiting

**Hypothesis:** Cycle 666 geodesic on the broad ONC isoxazole tag saved 125086620 (C-substituted 1,2,4-oxadiazole) but extraed oxime-ether 135267488 and CH-oxadiazole 384575875. Gating exact geodesic to the oxadiazole pentagon with a carbon substituent keeps the train save and drops those extras.

**Outcome and uncertainty:** Cycle 670 non_generalizable vs `2421717`. Train Δ=+2.508e-4 (only 125086620 17→15). Valid Δ=0, bit-identical. Energy-safe. Clean train-only near-miss. Cycle 671 geodesic twitched leftover 194689238. Cycles 672–673, 679, 681–682 fused interpolants/RFO/iterative no-op. Cycle 674–678 leftover 135169446 overlays closed. Cycle 680 n<18 allene RFO@15 bit-identical. Cycle 683 nitro-CF3 geodesic bit-identical. Cycles 684–686 nitro-CF3 MIS interpolants bit-identical. Cycle 687 nitro-CF3 iterative B⁺ energy twitch at 32. Cycle 688 RFO@20 saved leftover 103943741 32→31 (valid Δ=+6.94e-5; one short of δ). Cycles 689–690 RFO@15/@18 extraed leftover 32→33. Cycles 691 and 693 RFO@19/@22 match cycle 688 n_steps at 31. Cycle 692 skip GDIIS on RFO@20 bit-identical to 688. Cycles 694–696 dummy H0 overlays bit-identical to 688. Cycle 697 fused `wa=0.70` energy twitch at leftover 194689238 32. Cycle 698 nitro-CF3 dummy-dihedral 0.10 bit-identical to 688. Cycle 699 nitro-CF3 iterative B⁺ energy twitch at leftover 31. Cycle 700 sulfonium geodesic extraed leftover 163343378 33→34. Cycle 701 sulfonium RFO@20 energy twitch at leftover 33. Cycle 702 n<18 allene RFO@20 bit-identical to 688. Cycle 703 fused flowchart-from-start hopped leftover 194689238 32→200 / +0.144 kcal and stalled (maxD→0, maxF stuck at 1.97e-3 Eh/Bohr). Cycle 704 **keep**: delay flowchart until nsteps≥20; leftover 194689238 32→27 plus leftover 103943741 32→31. Train Δ=+2.508e-4; valid Δ=+4.278e-4. New champion `40e894e6688cea8b89f66cdb7eadd4a930ff5dee`.

**Reason to revisit:** Incorporated in champion `40e894e` (cycle 670 oxadiazole geodesic + cycle 688 nitro-CF3 RFO@20 + cycle 704 fused flowchart after 20). Remaining extras: train 363892164 +11, venetoclax +6, 433943527 +1, carboxylates–esters +1, amides–pyridine +1; valid 135169446 +2 and 163343378 +1.

**Next experiment:** Cycle 716: cycle 715 stack plus Schlegel flowchart after 15 on n<18 allenes vs leftover 135169446 extra=2. Support: leftover 363892164 stays 33; leftover 135169446 ≤19; both gates vs `40e894e`. Contradict: 715 train save lost, leftover extra/twitch, hop, or Δ fails.

**Attempts:**
- Cycle 670; candidate `710a57685fbac18ce3643055f89bbf2bb6803894`; champion `2421717fc47361279dde7e172ead7b439968daaa`; decision non_generalizable.
  [Implementation](ideas/sella-oxadiazole-exact-geodesic/710a57685fbac18ce3643055f89bbf2bb6803894/algo.py).
  [Training evidence](evaluation_results/cycle-670-train.json).
  [Validation evidence](evaluation_results/cycle-670-valid.json).
  [Narrative](full_log.md#cycle-670--exact-geodesic-on-30-80-c-substituted-124-oxadiazoles).
- Cycle 671; candidate `3eb04fba8ff53e3ca6556395bad01575e77521ae`; champion `2421717fc47361279dde7e172ead7b439968daaa`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-671-train.json).
  [Validation evidence](evaluation_results/cycle-671-valid.json).
  [Narrative](full_log.md#cycle-671--cycle-670-plus-exact-geodesic-on-30-80-fused-benzothiazines).
- Cycle 672; candidate `cc5b0c3e738542400a22a2d5d84ad8fdcdca3109`; champion `2421717fc47361279dde7e172ead7b439968daaa`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-672-train.json).
  [Validation evidence](evaluation_results/cycle-672-valid.json).
  [Narrative](full_log.md#cycle-672--cycle-670-plus-wo070-on-30-80-fused-benzothiazines).
- Cycle 673; candidate `70f7f17bbd0f4cb81aab2f1c986447c0319a729f`; champion `2421717fc47361279dde7e172ead7b439968daaa`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-673-train.json).
  [Validation evidence](evaluation_results/cycle-673-valid.json).
  [Narrative](full_log.md#cycle-673--cycle-670-plus-skip-gdiis-on-30-80-fused-benzothiazines).
- Cycle 674; candidate `a4aa991f49998940a08d9d866c23fff352888f03`; champion `2421717fc47361279dde7e172ead7b439968daaa`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-674-train.json).
  [Validation evidence](evaluation_results/cycle-674-valid.json).
  [Narrative](full_log.md#cycle-674--cycle-670-plus-wo070-on-n18-allenes).
- Cycle 675; candidate `7303792c3097fa843874c323b75a9be3f05ba284`; champion `2421717fc47361279dde7e172ead7b439968daaa`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-675-train.json).
  [Validation evidence](evaluation_results/cycle-675-valid.json).
  [Narrative](full_log.md#cycle-675--cycle-670-plus-wb070-on-n18-allenes).
- Cycle 676; candidate `4334d748e49d642d625a085be06e7f240c48c0ec`; champion `2421717fc47361279dde7e172ead7b439968daaa`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-676-train.json).
  [Validation evidence](evaluation_results/cycle-676-valid.json).
  [Narrative](full_log.md#cycle-676--cycle-670-plus-008-ha-dummy-angle-on-n18-allenes).
- Cycle 677; candidate `d4ce23d18d539a82db4dd6d40aaaa969fe518d0c`; champion `2421717fc47361279dde7e172ead7b439968daaa`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-677-train.json).
  [Validation evidence](evaluation_results/cycle-677-valid.json).
  [Narrative](full_log.md#cycle-677--cycle-670-plus-010-ha-dummy-dihedral-on-n18-allenes).
- Cycle 678; candidate `ea4b3c44f7d93612e7a4aae9e47282d781ccecc9`; champion `2421717fc47361279dde7e172ead7b439968daaa`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-678-train.json).
  [Validation evidence](evaluation_results/cycle-678-valid.json).
  [Narrative](full_log.md#cycle-678--cycle-670-plus-iterative-cartesian-b-on-n18-allenes).
- Cycle 679; candidate `7640a896f7999c9037f81216e03193f0202f2f29`; champion `2421717fc47361279dde7e172ead7b439968daaa`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-679-train.json).
  [Validation evidence](evaluation_results/cycle-679-valid.json).
  [Narrative](full_log.md#cycle-679--cycle-670-plus-wb070-on-30-80-fused-benzothiazines).
- Cycle 680; candidate `6470e6e56e6298a5caf3f766142d2bd869545b13`; champion `2421717fc47361279dde7e172ead7b439968daaa`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-680-train.json).
  [Validation evidence](evaluation_results/cycle-680-valid.json).
  [Narrative](full_log.md#cycle-680--cycle-670-plus-rfo-after-15-on-n18-allenes).
- Cycle 681; candidate `309e1895f4fa80c8a10042c6a2eeb213e4becc73`; champion `2421717fc47361279dde7e172ead7b439968daaa`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-681-train.json).
  [Validation evidence](evaluation_results/cycle-681-valid.json).
  [Narrative](full_log.md#cycle-681--cycle-670-plus-rfo-after-20-on-30-80-fused-benzothiazines).
- Cycle 682; candidate `efb30737c29466af9e557d6fc1837f9a08ab9585`; champion `2421717fc47361279dde7e172ead7b439968daaa`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-682-train.json).
  [Validation evidence](evaluation_results/cycle-682-valid.json).
  [Narrative](full_log.md#cycle-682--cycle-670-plus-iterative-cartesian-b-on-30-80-fused-benzothiazines).
- Cycle 683; candidate `216ff0773ef2e6e5f6243296b33d73af3dc6e293`; champion `2421717fc47361279dde7e172ead7b439968daaa`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-683-train.json).
  [Validation evidence](evaluation_results/cycle-683-valid.json).
  [Narrative](full_log.md#cycle-683--cycle-670-plus-exact-geodesic-on-30-80-nitro-cf3).
- Cycle 684; candidate `ac8c6392773a5d53ae441db744d44626c87deb6e`; champion `2421717fc47361279dde7e172ead7b439968daaa`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-684-train.json).
  [Validation evidence](evaluation_results/cycle-684-valid.json).
  [Narrative](full_log.md#cycle-684--cycle-670-plus-wo070-on-30-80-nitro-cf3).
- Cycle 685; candidate `3a59174edfe17a4b741f34a9ac857dda165023ec`; champion `2421717fc47361279dde7e172ead7b439968daaa`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-685-train.json).
  [Validation evidence](evaluation_results/cycle-685-valid.json).
  [Narrative](full_log.md#cycle-685--cycle-670-plus-wb070-on-30-80-nitro-cf3).
- Cycle 686; candidate `4194545663308cc6d27a83124cfbc6c9371ad3d4`; champion `2421717fc47361279dde7e172ead7b439968daaa`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-686-train.json).
  [Validation evidence](evaluation_results/cycle-686-valid.json).
  [Narrative](full_log.md#cycle-686--cycle-670-plus-wa070-on-30-80-nitro-cf3).
- Cycle 687; candidate `47f7b58d21a4b5d17fb9f540ed1944a15c73403c`; champion `2421717fc47361279dde7e172ead7b439968daaa`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-687-train.json).
  [Validation evidence](evaluation_results/cycle-687-valid.json).
  [Narrative](full_log.md#cycle-687--cycle-670-plus-iterative-cartesian-b-on-30-80-nitro-cf3).
- Cycle 688; candidate `12ae8344b0642325c5fa1a59016036ee69083ba1`; champion `2421717fc47361279dde7e172ead7b439968daaa`; decision non_generalizable.
  [Implementation](ideas/sella-oxadiazole-exact-geodesic/12ae8344b0642325c5fa1a59016036ee69083ba1/algo.py).
  [Training evidence](evaluation_results/cycle-688-train.json).
  [Validation evidence](evaluation_results/cycle-688-valid.json).
  [Narrative](full_log.md#cycle-688--cycle-670-plus-rfo-after-20-on-30-80-nitro-cf3).
- Cycle 689; candidate `24cf3053ac7aa099c0de0ae54e25c5d0d18f5335`; champion `2421717fc47361279dde7e172ead7b439968daaa`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-689-train.json).
  [Validation evidence](evaluation_results/cycle-689-valid.json).
  [Narrative](full_log.md#cycle-689--cycle-670-plus-rfo-after-15-on-30-80-nitro-cf3).
- Cycle 690; candidate `9c03b22e2f1034ac1e8c5686be05c35493f569da`; champion `2421717fc47361279dde7e172ead7b439968daaa`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-690-train.json).
  [Validation evidence](evaluation_results/cycle-690-valid.json).
  [Narrative](full_log.md#cycle-690--cycle-670-plus-rfo-after-18-on-30-80-nitro-cf3).
- Cycle 691; candidate `5986c818189b29cc59826da223b90f12f177e7e8`; champion `2421717fc47361279dde7e172ead7b439968daaa`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-691-train.json).
  [Validation evidence](evaluation_results/cycle-691-valid.json).
  [Narrative](full_log.md#cycle-691--cycle-670-plus-rfo-after-19-on-30-80-nitro-cf3).
- Cycle 692; candidate `352aa5133146ebae4d7ba052614922dcb4f2f6cb`; champion `2421717fc47361279dde7e172ead7b439968daaa`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-692-train.json).
  [Validation evidence](evaluation_results/cycle-692-valid.json).
  [Narrative](full_log.md#cycle-692--cycle-688-plus-skip-gdiis-on-30-80-nitro-cf3).
- Cycle 693; candidate `1781f6355ac1c00b25c5fde48cd732fd62428bc4`; champion `2421717fc47361279dde7e172ead7b439968daaa`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-693-train.json).
  [Validation evidence](evaluation_results/cycle-693-valid.json).
  [Narrative](full_log.md#cycle-693--cycle-670-plus-rfo-after-22-on-30-80-nitro-cf3).
- Cycle 694; candidate `eef5c39e503ed42374c65074aa49c57fcbc92d51`; champion `2421717fc47361279dde7e172ead7b439968daaa`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-694-train.json).
  [Validation evidence](evaluation_results/cycle-694-valid.json).
  [Narrative](full_log.md#cycle-694--cycle-688-plus-008-ha-dummy-angle-on-30-80-nitro-cf3).
- Cycle 695; candidate `4d6c884179f7b4c589152fcda1f37200836634ba`; champion `2421717fc47361279dde7e172ead7b439968daaa`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-695-train.json).
  [Validation evidence](evaluation_results/cycle-695-valid.json).
  [Narrative](full_log.md#cycle-695--cycle-688-plus-008-ha-dummy-angle-on-30-80-fused-benzothiazines).
- Cycle 696; candidate `1fc51fda9733a9e55e6edd98a54b7063c80203a0`; champion `2421717fc47361279dde7e172ead7b439968daaa`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-696-train.json).
  [Validation evidence](evaluation_results/cycle-696-valid.json).
  [Narrative](full_log.md#cycle-696--cycle-688-plus-010-ha-dummy-dihedral-on-30-80-fused-benzothiazines).
- Cycle 697; candidate `51c1cafbabca95e8ce6e8226451ec7819b22c82a`; champion `2421717fc47361279dde7e172ead7b439968daaa`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-697-train.json).
  [Validation evidence](evaluation_results/cycle-697-valid.json).
  [Narrative](full_log.md#cycle-697--cycle-688-plus-wa070-on-30-80-fused-benzothiazines).
- Cycle 698; candidate `3758b839b2e5d68406c1dcba303afc88fb451467`; champion `2421717fc47361279dde7e172ead7b439968daaa`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-698-train.json).
  [Validation evidence](evaluation_results/cycle-698-valid.json).
  [Narrative](full_log.md#cycle-698--cycle-688-plus-010-ha-dummy-dihedral-on-30-80-nitro-cf3).
- Cycle 699; candidate `a41fbcc92ed3b701ec103f113304d73b769243c1`; champion `2421717fc47361279dde7e172ead7b439968daaa`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-699-train.json).
  [Validation evidence](evaluation_results/cycle-699-valid.json).
  [Narrative](full_log.md#cycle-699--cycle-688-plus-iterative-cartesian-b-on-30-80-nitro-cf3).
- Cycle 700; candidate `a31bb29181ec9866b685e2d10b04800562c1d8ae`; champion `2421717fc47361279dde7e172ead7b439968daaa`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-700-train.json).
  [Validation evidence](evaluation_results/cycle-700-valid.json).
  [Narrative](full_log.md#cycle-700--cycle-688-plus-exact-geodesic-on-30-80-sulfoniums).
- Cycle 701; candidate `bd850c5fea0c183e5c5e48e71b49af20a5a42a1e`; champion `2421717fc47361279dde7e172ead7b439968daaa`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-701-train.json).
  [Validation evidence](evaluation_results/cycle-701-valid.json).
  [Narrative](full_log.md#cycle-701--cycle-688-plus-rfo-after-20-on-30-80-sulfoniums).
- Cycle 702; candidate `3af14c52ae8a1d19a2fedef98222b56974ee759f`; champion `2421717fc47361279dde7e172ead7b439968daaa`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-702-train.json).
  [Validation evidence](evaluation_results/cycle-702-valid.json).
  [Narrative](full_log.md#cycle-702--cycle-688-plus-rfo-after-20-on-n18-allenes).
- Cycle 703; candidate `b0e0b0fee5ae95a8ebe18eed622d8062d14d28b6`; champion `2421717fc47361279dde7e172ead7b439968daaa`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-703-train.json).
  [Validation evidence](evaluation_results/cycle-703-valid.json).
  [Narrative](full_log.md#cycle-703--cycle-688-plus-schlegel-flowchart-hessian-on-30-80-fused-benzothiazines).
- Cycle 704; candidate `40e894e6688cea8b89f66cdb7eadd4a930ff5dee`; champion `2421717fc47361279dde7e172ead7b439968daaa`; decision keep.
  [Implementation](algo.py).
  [Training evidence](evaluation_results/cycle-704-train.json).
  [Validation evidence](evaluation_results/cycle-704-valid.json).
  [Narrative](full_log.md#cycle-704--cycle-688-plus-schlegel-flowchart-hessian-after-20-on-30-80-fused-benzothiazines).

## sella-isocyanide-flowchart-hessian: Schlegel flowchart after 20 on 30–80 isocyanides
Status: incorporated

**Hypothesis:** Cycle 704 delayed flowchart cheapened fused leftover 194689238. The same SR1/BFGS/PSB updates after 20 on 30–80 1-coordinate C bonded to N cheapen dummy-linear leftover 363892164 without dimers or n≥80.

**Outcome and uncertainty:** Cycle 705 discard vs `40e894e`. Train V, Δ=+6.88e-5. Only leftover 363892164 42→41, energy-safe twitch. One short of train δ. Cycle 706 after 15: leftover stays 41. Cycle 707 Bofill after 20: leftover 42→169, Δ=−8.74e-3. Cycle 708 thiosulfonate flowchart@15 twitches leftover 433943527 at 21. Cycle 709 n≥80 flowchart@45 stalls paliperidone 73→167 and extras venetoclax 57→58. Cycle 710 skip GDIIS on 705: bit-identical to 705. Cycle 711 RFO@20 from 704: bit-identical to champion. Cycle 712 RFO@20 on 705: leftover stays 41 with an energy twitch vs 705. Cycle 713 thiosulfonate RFO@15: bit-identical to champion. Cycle 714 `wa=0.70` on 705: leftover 41→42, undoes the flowchart save. Cycle 715 **non_generalizable**: flowchart@20 plus iterative B⁺ saved leftover 363892164 42→33 (train Δ=+6.19e-4); valid Δ=0, canary 312525537 energy twitch at 19. Cycles 716–718 n<18 allene flowchart extras leftover 135169446 20→23; iterative extras 20→21; skip GDIIS bit-identical at 20. Cycles 719–722 leftover esters–phenol tagging repairs closed. Cycle 723 leftover 163343378 iterative extras 33→35, Δ_valid=−1.34e-4. Cycle 728 **keep** stacked disconnected 18–30 aryl-phenol iterative on the 715 train stack. Cycle 730 skip GDIIS with iterative: bit-identical leftover 33. Cycle 731 RFO@20 with iterative: leftover 33→42 extras. Cycle 732 dummy-angle 0.08 with iterative: leftover 33→42 extras. Cycles 733–736/739 late-trust/ρ_inc/σ_dec with iterative: bit-identical leftover 33. Cycle 738 `diag_every_n=1` invalid (stub `rayleigh_ritz`). Cycle 741 `wb=0.70` with iterative: bit-identical leftover 33. Cycles 758–760 Helgaker |λ| floors of 1e-4/0.001/0.01 without copying `eval_floor` onto `get_HL_projected` were bit-identical leftover 33. Cycle 761 copy plus 0.01 Eh extras leftover 33→41, Δ=−5.50e-4. Cycle 762 copy plus 0.001 Eh bit-identical leftover 33. Cycle 763 skip s·y<0 from start bit-identical leftover 33. Cycle 764 from-start Powell extras leftover 33→41, Δ=−5.50e-4. Cycle 765 Powell after 20 is bit-identical to 764. Cycle 766 three-point C1 GDIIS bit-identical leftover 33. Cycle 767 GDIIS start 15 bit-identical leftover 33. Cycles 768–770 Powell η=0.05/0.10/0.15 after 20 bit-identical leftover 33; only η=0.2 extras leftover 33→41. Cycle 771 skip s·y < 0.2 s·Bs after 20 extras leftover 33→36, Δ=−2.06e-4. Cycle 781/782 TrustRegion after 20 bit-identical leftover 33. Cycle 783 **non_generalizable**: exact geodesic WITH iterative saved leftover 363892164 33→27 (train Δ=+4.13e-4); valid Δ=0, canary 312525537 energy twitch at 19. Cycle 784 exact geodesic on n<18 allenes extras leftover 135169446 20→21, Δ_valid=−1.19e-4.

**Reason to revisit:** Incorporated in champion `133f73f`. Do not drop 30–80 isocyanide flowchart@20+iterative. Do not interpolate leftover 363892164 Hessian nsteps/Bofill/`wa`/`wd`/skip GDIIS/RFO/dummy-angle 0.08 with or without iterative B⁺. Do not apply flowchart, iterative, skip GDIIS, or exact geodesic to n<18 allenes. Close leftover esters–phenol skip GDIIS/iterative tagging. Close leftover 163343378 interpolants (flowchart extras leftover 33→51).

**Next experiment:** Keep cycle 799. Cycle 813 Helgaker 0.01 on leftover esters–phenol energy-twitches at 35. Repair with 0.1 Eh |λ| floor (ChemShell Baker default) on the same unique connected 18–30 aryl-phenol plus ester tag stacked with 799 vs leftover esters–phenol extra=2. leftover 35→33 would pass valid δ. Support: leftover 363 ≤27; venetoclax ≤56; train alkanes–phenol ≤35; leftover esters–phenol ≤33. Contradict: leftover extra/twitch/hop, 799 train saves lost, or Δ fails.

**Attempts:**
- Cycle 705; candidate `8a62d14890bfb230d7434d1e7628ed3365170c14`; champion `40e894e6688cea8b89f66cdb7eadd4a930ff5dee`; decision discard.
  [Implementation](ideas/sella-isocyanide-flowchart-hessian/8a62d14890bfb230d7434d1e7628ed3365170c14/algo.py).
  [Training evidence](evaluation_results/cycle-705-train.json).
  [Narrative](full_log.md#cycle-705--cycle-704-plus-schlegel-flowchart-hessian-after-20-on-30-80-isocyanides).
- Cycle 706; candidate `fe72ecc266f3b91b4f55ccc2d5c53cb987e29b95`; champion `40e894e6688cea8b89f66cdb7eadd4a930ff5dee`; decision discard.
  [Training evidence](evaluation_results/cycle-706-train.json).
  [Narrative](full_log.md#cycle-706--cycle-704-plus-schlegel-flowchart-hessian-after-15-on-30-80-isocyanides).
- Cycle 707; candidate `52eb5f9317a3ac2e7d915a2bee6f43a08819f029`; champion `40e894e6688cea8b89f66cdb7eadd4a930ff5dee`; decision discard.
  [Training evidence](evaluation_results/cycle-707-train.json).
  [Narrative](full_log.md#cycle-707--cycle-704-plus-bofill-hessian-after-20-on-30-80-isocyanides).
- Cycle 710; candidate `77741344100ff8adbbdceb1c2c1bacfdc6075fff`; champion `40e894e6688cea8b89f66cdb7eadd4a930ff5dee`; decision discard.
  [Training evidence](evaluation_results/cycle-710-train.json).
  [Narrative](full_log.md#cycle-710--cycle-705-plus-skip-gdiis-on-30-80-isocyanides).
- Cycle 711; candidate `ec8c39b8fbc2774bdc65b17a0d1f3c12293e90c0`; champion `40e894e6688cea8b89f66cdb7eadd4a930ff5dee`; decision discard.
  [Training evidence](evaluation_results/cycle-711-train.json).
  [Narrative](full_log.md#cycle-711--cycle-704-plus-banerjee-rfo-after-20-on-30-80-isocyanides).
- Cycle 712; candidate `9ec76656863443504eb626a5d2ed7442a7c7ae6b`; champion `40e894e6688cea8b89f66cdb7eadd4a930ff5dee`; decision discard.
  [Training evidence](evaluation_results/cycle-712-train.json).
  [Narrative](full_log.md#cycle-712--cycle-705-plus-banerjee-rfo-after-20-on-30-80-isocyanides).
- Cycle 714; candidate `42407721edd650ba0c64b4263be6c7bde3e2af77`; champion `40e894e6688cea8b89f66cdb7eadd4a930ff5dee`; decision discard.
  [Training evidence](evaluation_results/cycle-714-train.json).
  [Narrative](full_log.md#cycle-714--cycle-705-plus-wa070-on-30-80-isocyanides).
- Cycle 715; candidate `19e479163450ade23283e21d7cf4983ec627792f`; champion `40e894e6688cea8b89f66cdb7eadd4a930ff5dee`; decision non_generalizable.
  [Implementation](ideas/sella-isocyanide-flowchart-iterative/19e479163450ade23283e21d7cf4983ec627792f/algo.py).
  [Training evidence](evaluation_results/cycle-715-train.json).
  [Validation evidence](evaluation_results/cycle-715-valid.json).
  [Narrative](full_log.md#cycle-715--cycle-705-plus-iterative-cartesian-b-on-30-80-isocyanides).
- Cycle 716; candidate `2d95756e6ff8fb6cf3bbf5f2a37d1f829ba2d7c2`; champion `40e894e6688cea8b89f66cdb7eadd4a930ff5dee`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-716-train.json).
  [Validation evidence](evaluation_results/cycle-716-valid.json).
  [Narrative](full_log.md#cycle-716--cycle-715-plus-schlegel-flowchart-after-15-on-n18-allenes).
- Cycle 717; candidate `0f8a30cc907abdeb088c861641a24163ef0a8701`; champion `40e894e6688cea8b89f66cdb7eadd4a930ff5dee`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-717-train.json).
  [Validation evidence](evaluation_results/cycle-717-valid.json).
  [Narrative](full_log.md#cycle-717--cycle-715-plus-iterative-cartesian-b-on-n18-allenes).
- Cycle 718; candidate `c82b2732bae2f8271714332f636069375db77a81`; champion `40e894e6688cea8b89f66cdb7eadd4a930ff5dee`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-718-train.json).
  [Validation evidence](evaluation_results/cycle-718-valid.json).
  [Narrative](full_log.md#cycle-718--cycle-715-plus-skip-gdiis-on-n18-allenes).
- Cycle 719; candidate `f9b2ea525fa1a8ff2412286d5c6e1643ca3a589c`; champion `40e894e6688cea8b89f66cdb7eadd4a930ff5dee`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-719-train.json).
  [Validation evidence](evaluation_results/cycle-719-valid.json).
  [Narrative](full_log.md#cycle-719--cycle-715-plus-skip-gdiis-on-1830-esterphenol).
- Cycle 720; candidate `a27eb71e7011a311f61fe4eac1f62f9eea096891`; champion `40e894e6688cea8b89f66cdb7eadd4a930ff5dee`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-720-train.json).
  [Validation evidence](evaluation_results/cycle-720-valid.json).
  [Narrative](full_log.md#cycle-720--cycle-715-plus-iterative-cartesian-b-on-1830-esterphenol).
- Cycle 721; candidate `3d0b95f530135cc8f93a0f0264f90c889edef1e4`; champion `40e894e6688cea8b89f66cdb7eadd4a930ff5dee`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-721-train.json).
  [Validation evidence](evaluation_results/cycle-721-valid.json).
  [Narrative](full_log.md#cycle-721--cycle-715-plus-iterative-cartesian-b-on-disconnected-1830-esterphenol).
- Cycle 722; candidate `f4e24fa65b34792fa71d02c6dce18422bffee4e0`; champion `40e894e6688cea8b89f66cdb7eadd4a930ff5dee`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-722-train.json).
  [Validation evidence](evaluation_results/cycle-722-valid.json).
  [Narrative](full_log.md#cycle-722--cycle-715-plus-iterative-b-on-connected-1830-esterphenol-with-h-bond-phenol-o).
- Cycle 723; candidate `9681875792cba578b314248b019fea50103044b0`; champion `40e894e6688cea8b89f66cdb7eadd4a930ff5dee`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-723-train.json).
  [Validation evidence](evaluation_results/cycle-723-valid.json).
  [Narrative](full_log.md#cycle-723--cycle-715-plus-iterative-cartesian-b-on-3080-sulfoniums).
- Cycle 724; candidate `af00248a18c2eefe24e56c293e5c91a6b7952ebf`; champion `40e894e6688cea8b89f66cdb7eadd4a930ff5dee`; decision non_generalizable.
  [Implementation](ideas/sella-carboxylate-phenol-iterative/af00248a18c2eefe24e56c293e5c91a6b7952ebf/algo.py).
  [Training evidence](evaluation_results/cycle-724-train.json).
  [Validation evidence](evaluation_results/cycle-724-valid.json).
  [Narrative](full_log.md#cycle-724--cycle-715-plus-iterative-b-on-disconnected-1830-carboxylatephenol).

## sella-carboxylate-phenol-iterative: iterative B⁺ on disconnected 18–30 carboxylate–phenol
Status: incorporated

**Hypothesis:** Cycle 715 is a train-only near-miss. Valid leftover carboxylates–phenol extra=1 (`n_ref=27`) would pass valid δ on −2. Iterative Cartesian B⁺ on disconnected 18–30 dimers with both an aryl phenol and a carboxylate carbon {two 1-coordinate O, one C} cheapens leftover without tagging train n=17.

**Outcome and uncertainty:** Cycle 724 non_generalizable vs `40e894e`. Train V, Δ=+6.19e-4, bit-identical to cycle 715. Valid V, Δ=+7.96e-5. Only leftover carboxylates–phenol 28→27, energy-safe. One short of valid δ. Cycle 725 skip GDIIS bit-identical to 724. Cycle 726 flowchart extras leftover 27→28, undoes the save. Cycle 727 RFO@20 bit-identical to 724. Cycle 728 **keep** widened the tag to disconnected 18–30 aryl phenol (dropped carboxylate): train Δ=+6.66e-4, valid Δ=+2.93e-4 (carboxylates–phenol 28→27, amides–phenol 43→40, benzene–phenol 34→33). New champion `133f73f`.

**Reason to revisit:** Incorporated in champion `133f73f` as the wider aryl-phenol tag. Do not restore the carboxylate conjunction. Close skip GDIIS, flowchart, and RFO@20 on the carboxylate–phenol subset. Leftover alkanes–phenol extra=1 still fires with an energy twitch; iterative is not that leftover’s limiter.

**Next experiment:** leftover 433943527 extra=1 (`n_ref=20`) still passes train δ on −1. Helgaker 0.01 Eh |λ| floor on connected 12–30 thiosulfonates stacked with cycle 799 vs `133f73f`.

**Attempts:**
- Cycle 724; candidate `af00248a18c2eefe24e56c293e5c91a6b7952ebf`; champion `40e894e6688cea8b89f66cdb7eadd4a930ff5dee`; decision non_generalizable.
  [Implementation](ideas/sella-carboxylate-phenol-iterative/af00248a18c2eefe24e56c293e5c91a6b7952ebf/algo.py).
  [Training evidence](evaluation_results/cycle-724-train.json).
  [Validation evidence](evaluation_results/cycle-724-valid.json).
  [Narrative](full_log.md#cycle-724--cycle-715-plus-iterative-b-on-disconnected-1830-carboxylatephenol).
- Cycle 725; candidate `b61941a5604b7d92492d8f3e7f124cad6df6101e`; champion `40e894e6688cea8b89f66cdb7eadd4a930ff5dee`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-725-train.json).
  [Validation evidence](evaluation_results/cycle-725-valid.json).
  [Narrative](full_log.md#cycle-725--cycle-724-plus-skip-gdiis-on-disconnected-1830-carboxylatephenol).
- Cycle 726; candidate `6e5c10b904bb066ebce67fc2357ac09297cb4c11`; champion `40e894e6688cea8b89f66cdb7eadd4a930ff5dee`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-726-train.json).
  [Validation evidence](evaluation_results/cycle-726-valid.json).
  [Narrative](full_log.md#cycle-726--cycle-724-plus-schlegel-flowchart-after-20-on-disconnected-1830-carboxylatephenol).
- Cycle 727; candidate `ef78760c467acdf49683c43a19350e1c6bdb3410`; champion `40e894e6688cea8b89f66cdb7eadd4a930ff5dee`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-727-train.json).
  [Validation evidence](evaluation_results/cycle-727-valid.json).
  [Narrative](full_log.md#cycle-727--cycle-724-plus-banerjee-rfo-after-20-on-disconnected-1830-carboxylatephenol).
- Cycle 728; candidate `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; champion `40e894e6688cea8b89f66cdb7eadd4a930ff5dee`; decision keep.
  [Implementation](algo.py).
  [Training evidence](evaluation_results/cycle-728-train.json).
  [Validation evidence](evaluation_results/cycle-728-valid.json).
  [Narrative](full_log.md#cycle-728--cycle-715-plus-iterative-b-on-disconnected-1830-aryl-phenol).
- Cycle 783; candidate `5f42ae5410cf91bfaeeba1e685bdc5368ee5d36e`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Implementation](ideas/sella-isocyanide-exact-geodesic/5f42ae5410cf91bfaeeba1e685bdc5368ee5d36e/algo.py).
  [Training evidence](evaluation_results/cycle-783-train.json).
  [Validation evidence](evaluation_results/cycle-783-valid.json).
  [Narrative](full_log.md#cycle-783--cycle-728-plus-exact-geodesic-on-30-80-isocyanides).

## sella-isocyanide-exact-geodesic: exact geodesic on 30–80 isocyanides WITH iterative
Status: revisiting

**Hypothesis:** Leftover 363892164 extra=2 after flowchart@20+iterative is dummy-linear geodesic-limited. Exact geodesic WITH iterative cheapens leftover 33→27.

**Outcome and uncertainty:** Cycle 783 non_generalizable vs `133f73f`. Train V, Δ=+4.13e-4, leftover 363892164 33→27 energy-safe. Valid V, Δ=0, only 312525537 energy twitch at 19. Cycle 784 geodesic on n<18 allenes extras leftover 135169446 20→21. Train-only near-miss to stack with a valid leftover save that is not geodesic.

**Reason to revisit:** Promising train Δ=+4.13e-4. Close leftover 135169446 geodesic. Stack with allene `wd=0.60` or cycle 751 RFO@20 (valid has no n≥80).

**Next experiment:** Keep cycle 799. Cycle 814 Helgaker 0.1 extras leftover esters–phenol 35→55. Repair 2/3 with 0.03 Eh |λ| floor on the same unique connected 18–30 aryl-phenol plus ester tag stacked with 799. leftover 35→33 would pass valid δ. Support: leftover 363 ≤27; venetoclax ≤56; train alkanes–phenol ≤35; leftover esters–phenol ≤33. Contradict: leftover extra/twitch/hop, 799 train saves lost, or Δ fails.

**Attempts:**
- Cycle 783; candidate `5f42ae5410cf91bfaeeba1e685bdc5368ee5d36e`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Implementation](ideas/sella-isocyanide-exact-geodesic/5f42ae5410cf91bfaeeba1e685bdc5368ee5d36e/algo.py).
  [Training evidence](evaluation_results/cycle-783-train.json).
  [Validation evidence](evaluation_results/cycle-783-valid.json).
  [Narrative](full_log.md#cycle-783--cycle-728-plus-exact-geodesic-on-30-80-isocyanides).
- Cycle 784; candidate `05016dbdd2296d66ba037f5dda9ee6456279c7f2`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-784-train.json).
  [Validation evidence](evaluation_results/cycle-784-valid.json).
  [Narrative](full_log.md#cycle-784--cycle-783-plus-exact-geodesic-on-n18-allenes).
- Cycle 785; candidate `bd891ca6ab8c5284dc33e8b9ef99cd2e15c2063f`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-785-train.json).
  [Validation evidence](evaluation_results/cycle-785-valid.json).
  [Narrative](full_log.md#cycle-785--cycle-783-plus-wd060-on-n18-allenes).
- Cycle 786; candidate `41abf789b28deb3d00fd84ef21a54df83036baa8`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-786-train.json).
  [Validation evidence](evaluation_results/cycle-786-valid.json).
  [Narrative](full_log.md#cycle-786--cycle-783-plus-helgaker-λ-floor-0001-eh-on-n18-allenes).
- Cycle 787; candidate `642162c7f844e782414cfd7432f28abc35cef822`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-787-train.json).
  [Validation evidence](evaluation_results/cycle-787-valid.json).
  [Narrative](full_log.md#cycle-787--cycle-783-plus-helgaker-λ-floor-001-eh-on-n18-allenes).
- Cycle 788; candidate `a116c424f9cf2e6cd2f575b81bcc7c16c4e7b467`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-788-train.json).
  [Validation evidence](evaluation_results/cycle-788-valid.json).
  [Narrative](full_log.md#cycle-788--cycle-783-plus-schlegel-flowchart-after-18-on-n18-allenes).
- Cycle 789; candidate `c4380424dba9b401649a3c918574beadc4ee06e7`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-789-train.json).
  [Validation evidence](evaluation_results/cycle-789-valid.json).
  [Narrative](full_log.md#cycle-789--cycle-783-plus-schlegel-flowchart-after-16-on-n18-allenes).
- Cycle 790; candidate `1b457c4d5589731aae72c53f2adadee15f0c15be`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-790-train.json).
  [Validation evidence](evaluation_results/cycle-790-valid.json).
  [Narrative](full_log.md#cycle-790--cycle-783-plus-skip-sy0-hessian-updates-on-n18-allenes).
- Cycle 791; candidate `475da9ce3cda5425bf7a43439efb0d4616949e10`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-791-train.json).
  [Validation evidence](evaluation_results/cycle-791-valid.json).
  [Narrative](full_log.md#cycle-791--cycle-783-plus-powell-damped-hessian-updates-on-n18-allenes).
- Cycle 792; candidate `038cf04b43e07eebfdbb721396aa57b55e06eadd`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-792-train.json).
  [Validation evidence](evaluation_results/cycle-792-valid.json).
  [Narrative](full_log.md#cycle-792--cycle-783-plus-euclidean-trustregion-from-start-on-n18-allenes).
- Cycle 793; candidate `51a243e628de87f07ac0d3ade56440c6af390396`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Implementation](ideas/sella-isocyanide-geodesic-sulfonamide-rfo/51a243e628de87f07ac0d3ade56440c6af390396/algo.py).
  [Training evidence](evaluation_results/cycle-793-train.json).
  [Validation evidence](evaluation_results/cycle-793-valid.json).
  [Narrative](full_log.md#cycle-793--cycle-783-plus-banerjee-rfo-after-20-on-n80-sulfonamides).
- Cycle 794; candidate `83b49690fd490c618df1bbe3b770eeb22a9c5f01`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-794-train.json).
  [Validation evidence](evaluation_results/cycle-794-valid.json).
  [Narrative](full_log.md#cycle-794--cycle-793-plus-exact-geodesic-on-disconnected-1830-aryl-phenols).
- Cycle 795; candidate `0a9b21d4d5c1f92e4e35a9d320f1203a1003c084`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-795-train.json).
  [Validation evidence](evaluation_results/cycle-795-valid.json).
  [Narrative](full_log.md#cycle-795--cycle-793-plus-flowchart-after-20-on-30-80-sulfoniums).
- Cycle 796; candidate `a5479142b30a22f176b670a9184bce1aeb01b611`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-796-train.json).
  [Validation evidence](evaluation_results/cycle-796-valid.json).
  [Narrative](full_log.md#cycle-796--cycle-793-plus-rfo-after-20-on-1830-alkanephenol-dimers).
- Cycle 797; candidate `1865e9e4efb6d0760af01515127a7232cb1894d8`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-797-train.json).
  [Validation evidence](evaluation_results/cycle-797-valid.json).
  [Narrative](full_log.md#cycle-797--cycle-793-plus-rfo-after-20-on-saturated-1830-alkanephenol).
- Cycle 798; candidate `dbf729203fa7647b007b6ce1ed76b136457f8408`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-798-train.json).
  [Validation evidence](evaluation_results/cycle-798-valid.json).
  [Narrative](full_log.md#cycle-798--cycle-793-plus-exact-geodesic-on-saturated-1830-alkanephenol).
- Cycle 799; candidate `674b77ca0f758e3cf1650389d9f6c2ea4ae4f0ef`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Implementation](ideas/sella-alkane-phenol-flowchart-hessian/674b77ca0f758e3cf1650389d9f6c2ea4ae4f0ef/algo.py).
  [Training evidence](evaluation_results/cycle-799-train.json).
  [Validation evidence](evaluation_results/cycle-799-valid.json).
  [Narrative](full_log.md#cycle-799--cycle-793-plus-flowchart-after-20-on-saturated-1830-alkanephenol).
- Cycle 800; candidate `268e21130832d5382ff571ac7cc68c3d22aa0d16`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-800-train.json).
  [Validation evidence](evaluation_results/cycle-800-valid.json).
  [Narrative](full_log.md#cycle-800--cycle-799-plus-skip-gdiis-on-saturated-1830-alkanephenol).
- Cycle 801; candidate `fa9902addfd096da0b6d2f3c6e24a3a1f1f3770c`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-801-train.json).
  [Validation evidence](evaluation_results/cycle-801-valid.json).
  [Narrative](full_log.md#cycle-801--cycle-799-plus-rfo-after-20-on-n18-allenes).
- Cycle 802; candidate `5b1a25bdc24531473ef720e5218a5423ad4ed10f`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-802-train.json).
  [Validation evidence](evaluation_results/cycle-802-valid.json).
  [Narrative](full_log.md#cycle-802--cycle-799-plus-rfo-after-18-on-n18-allenes).
- Cycle 803; candidate `c18e81629ad53d364dc747fb94adcc3e46351eb4`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-803-train.json).
  [Validation evidence](evaluation_results/cycle-803-valid.json).
  [Narrative](full_log.md#cycle-803--cycle-799-plus-trustregion-after-20-on-saturated-1830-alkanephenol).
- Cycle 804; candidate `3e2c84e23906f4f713ad611681eb50253ed83a28`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-804-train.json).
  [Validation evidence](evaluation_results/cycle-804-valid.json).
  [Narrative](full_log.md#cycle-804--cycle-799-plus-rfo-after-20-on-connected-1830-esterphenol).
- Cycle 805; candidate `2c8e30129c966f880496115f8c3ea1c0ff4e54b0`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-805-train.json).
  [Validation evidence](evaluation_results/cycle-805-valid.json).
  [Narrative](full_log.md#cycle-805--cycle-799-plus-flowchart-after-20-on-connected-1830-esterphenol).
- Cycle 806; candidate `7865b07e2e3be6b42cc1683a43ae87aecdf3b42c`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-806-train.json).
  [Validation evidence](evaluation_results/cycle-806-valid.json).
  [Narrative](full_log.md#cycle-806--cycle-799-plus-exact-geodesic-on-connected-1830-esterphenol).
- Cycle 807; candidate `3730eaf65db274009d5d16e0290c673631f016a5`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-807-train.json).
  [Validation evidence](evaluation_results/cycle-807-valid.json).
  [Narrative](full_log.md#cycle-807--cycle-799-plus-skip-gdiis-on-connected-1830-esterphenol).
- Cycle 808; candidate `2fb65f8f3bca9982ab3affd7d64dc5e929afe711`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-808-train.json).
  [Validation evidence](evaluation_results/cycle-808-valid.json).
  [Narrative](full_log.md#cycle-808--cycle-799-plus-trustregion-after-20-on-connected-1830-esterphenol).
- Cycle 809; candidate `004b6a78126ad5417e3d4416b17c298167b88e76`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-809-train.json).
  [Validation evidence](evaluation_results/cycle-809-valid.json).
  [Narrative](full_log.md#cycle-809--cycle-799-plus-helgaker-0001-eh-floor-on-saturated-1830-alkanephenol).
- Cycle 810; candidate `6a09d97007b2f24696798fbc03d8be20f5a406dc`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Implementation](ideas/sella-alkane-phenol-helgaker-001/6a09d97007b2f24696798fbc03d8be20f5a406dc/algo.py).
  [Training evidence](evaluation_results/cycle-810-train.json).
  [Validation evidence](evaluation_results/cycle-810-valid.json).
  [Narrative](full_log.md#cycle-810--cycle-799-plus-helgaker-001-eh-floor-on-saturated-1830-alkanephenol).
- Cycle 811; candidate `66dbd7a243ae696895abd708ce0104e41f4c63cc`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision invalid.
  [Training evidence](evaluation_results/cycle-811-train.json).
  [Narrative](full_log.md#cycle-811--cycle-799-plus-helgaker-0005-eh-floor-on-saturated-1830-alkanephenol).

## sella-sulfonium-flowchart-hessian: Schlegel flowchart after 20 on 30–80 sulfoniums
Status: closed

**Hypothesis:** Leftover 163343378 extra=1 is not geodesic/iterative/H0 limited. Schlegel flowchart after 20 on connected 30–80 3-coordinate S {C,C,C} stacked with 793 cheapens leftover 33→31.

**Outcome and uncertainty:** Cycle 795 non_generalizable vs `133f73f`. Train bit-identical to 793. Valid leftover 163343378 33→51 extra +18, energy-safe.

**Reason to revisit:** Not as a sulfonium flowchart. Combined with geodesic/iterative extras, leftover 163343378 is not flowchart-limited.

**Next experiment:** None. Close leftover 163343378 interpolants. Repair cycle 796: Banerjee RFO after 20 on disconnected 18–30 aryl phenol plus a saturated C/H fragment (every carbon 4-coordinate) vs leftover alkanes–phenol extra=1. leftover 35→33 would pass valid δ. Support: leftover 363 ≤27; venetoclax ≤56; leftover alkanes–phenol ≤33; both gates vs `133f73f`. Contradict: leftover extra/twitch/hop, packing extras, or Δ fails.

**Attempts:**
- Cycle 795; candidate `0a9b21d4d5c1f92e4e35a9d320f1203a1003c084`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-795-train.json).
  [Validation evidence](evaluation_results/cycle-795-valid.json).
  [Narrative](full_log.md#cycle-795--cycle-793-plus-flowchart-after-20-on-30-80-sulfoniums).

## sella-alkane-phenol-rfo: Banerjee RFO after 20 on disconnected 18–30 alkane–phenol
Status: revisiting

**Hypothesis:** Leftover alkanes–phenol extra=1 is not iterative/geodesic limited. Banerjee RFO after 20 on disconnected 18–30 aryl phenol plus a C/H fragment cheapens leftover 35→33 without water or ester dimers.

**Outcome and uncertainty:** Cycle 796 non_generalizable vs `133f73f`. Train extras alkenes–phenol 36→38 (C/H matcher too broad). Valid leftover alkanes–phenol energy twitch at 35 (RFO fires, not the n_steps limiter). Benzene–phenol and alkenes–phenol also twitch.

**Reason to revisit:** Cycle 874 **keep** vs `133f73f`. Train Δ=+9.43e-4 (leftover 433 21→18 plus leftover 363 27, venetoclax 56, train alkanes–phenol 35). Valid Δ=+1.30e-4 (leftover esters–phenol 35→33). leftover 135169446 extra=2 independently passes valid δ on −1. leftover 135169446 GEDIIS after 15 extras 20→22; skip s·y<0 no-op; Baker no-op; Cartesian stall; RFO from start extras.

**Next experiment:** `delta0=0.15` from the first step on disconnected 18–30 saturated alkane–phenol stacked with cycle 885 vs `e478d0c`. leftover alkanes–phenol extra=1 needs −2. leftover 35→33 would keep. Support: leftover venetoclax ≤51; leftover 433 ≤18; leftover 363 ≤27; leftover alkanes–phenol ≤33; leftover esters–phenol ≤33; leftover 135169446 stays 20. Contradict: leftover extra/twitch/hop, 885 train save lost, or Δ fails.

**Next experiment:** Li–Frisch two-point GEDIIS after 20 on connected 18–30 ester–phenol stacked with cycle 870. leftover esters–phenol 35→33 would keep. Support: leftover 433 ≤18; leftover 363 ≤27; venetoclax ≤56; train alkanes–phenol ≤35; leftover esters–phenol ≤33. Contradict: leftover extra/twitch/hop, 870 train saves lost, or Δ fails.

**Attempts:**
- Cycle 796; candidate `a5479142b30a22f176b670a9184bce1aeb01b611`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-796-train.json).
  [Validation evidence](evaluation_results/cycle-796-valid.json).
  [Narrative](full_log.md#cycle-796--cycle-793-plus-rfo-after-20-on-1830-alkanephenol-dimers).
- Cycle 797; candidate `1865e9e4efb6d0760af01515127a7232cb1894d8`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-797-train.json).
  [Validation evidence](evaluation_results/cycle-797-valid.json).
  [Narrative](full_log.md#cycle-797--cycle-793-plus-rfo-after-20-on-saturated-1830-alkanephenol).
- Cycle 798; candidate `dbf729203fa7647b007b6ce1ed76b136457f8408`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-798-train.json).
  [Validation evidence](evaluation_results/cycle-798-valid.json).
  [Narrative](full_log.md#cycle-798--cycle-793-plus-exact-geodesic-on-saturated-1830-alkanephenol).
- Cycle 799; candidate `674b77ca0f758e3cf1650389d9f6c2ea4ae4f0ef`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Implementation](ideas/sella-alkane-phenol-flowchart-hessian/674b77ca0f758e3cf1650389d9f6c2ea4ae4f0ef/algo.py).
  [Training evidence](evaluation_results/cycle-799-train.json).
  [Validation evidence](evaluation_results/cycle-799-valid.json).
  [Narrative](full_log.md#cycle-799--cycle-793-plus-flowchart-after-20-on-saturated-1830-alkanephenol).
- Cycle 800; candidate `268e21130832d5382ff571ac7cc68c3d22aa0d16`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-800-train.json).
  [Validation evidence](evaluation_results/cycle-800-valid.json).
  [Narrative](full_log.md#cycle-800--cycle-799-plus-skip-gdiis-on-saturated-1830-alkanephenol).
- Cycle 809; candidate `004b6a78126ad5417e3d4416b17c298167b88e76`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-809-train.json).
  [Validation evidence](evaluation_results/cycle-809-valid.json).
  [Narrative](full_log.md#cycle-809--cycle-799-plus-helgaker-0001-eh-floor-on-saturated-1830-alkanephenol).
- Cycle 810; candidate `6a09d97007b2f24696798fbc03d8be20f5a406dc`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Implementation](ideas/sella-alkane-phenol-helgaker-001/6a09d97007b2f24696798fbc03d8be20f5a406dc/algo.py).
  [Training evidence](evaluation_results/cycle-810-train.json).
  [Validation evidence](evaluation_results/cycle-810-valid.json).
  [Narrative](full_log.md#cycle-810--cycle-799-plus-helgaker-001-eh-floor-on-saturated-1830-alkanephenol).
- Cycle 811; candidate `66dbd7a243ae696895abd708ce0104e41f4c63cc`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision invalid.
  [Training evidence](evaluation_results/cycle-811-train.json).
  [Narrative](full_log.md#cycle-811--cycle-799-plus-helgaker-0005-eh-floor-on-saturated-1830-alkanephenol).
- Cycle 812; candidate `a317d91881faea950dced86c2da2cb4a45704dba`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-812-train.json).
  [Validation evidence](evaluation_results/cycle-812-valid.json).
  [Narrative](full_log.md#cycle-812--cycle-799-plus-powell-damping-after-20-on-saturated-1830-alkanephenol).
- Cycle 813; candidate `ba0ad3dafb795c90e093aba93af93ed7fe455bfe`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-813-train.json).
  [Validation evidence](evaluation_results/cycle-813-valid.json).
  [Narrative](full_log.md#cycle-813--cycle-799-plus-helgaker-001-eh-floor-on-connected-1830-esterphenol).
- Cycle 814; candidate `7544cf31df4652641fb600b39498cd7cea4e724c`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-814-train.json).
  [Validation evidence](evaluation_results/cycle-814-valid.json).
  [Narrative](full_log.md#cycle-814--cycle-799-plus-helgaker-01-eh-floor-on-connected-1830-esterphenol).
- Cycle 815; candidate `b1b19236e4b26275077f0cb08a8f7e2c79807ea6`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-815-train.json).
  [Validation evidence](evaluation_results/cycle-815-valid.json).
  [Narrative](full_log.md#cycle-815--cycle-799-plus-helgaker-003-eh-floor-on-connected-1830-esterphenol).
- Cycle 816; candidate `24c936846592ad91c974ceb72b2f51cd0516af7a`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-816-train.json).
  [Validation evidence](evaluation_results/cycle-816-valid.json).
  [Narrative](full_log.md#cycle-816--cycle-799-plus-baker-hessian_shift-01-eh-on-connected-1830-esterphenol).
- Cycle 817; candidate `1c2d80d072fbf8b006a29e88ea34c43523da21d7`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-817-train.json).
  [Validation evidence](evaluation_results/cycle-817-valid.json).
  [Narrative](full_log.md#cycle-817--cycle-799-plus-dummy-angle-010-ha-on-connected-1830-esterphenol).
- Cycle 818; candidate `74103a0db38cc178899abeb54d342eac6a75f3c2`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-818-train.json).
  [Validation evidence](evaluation_results/cycle-818-valid.json).
  [Narrative](full_log.md#cycle-818--cycle-799-plus-powell-damping-after-20-on-connected-1830-esterphenol).
- Cycle 819; candidate `5514e7922d85c3fa21142e485a6a2a3d1ad54233`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-819-train.json).
  [Validation evidence](evaluation_results/cycle-819-valid.json).
  [Narrative](full_log.md#cycle-819--cycle-799-plus-wa070-on-connected-1830-esterphenol).
- Cycle 820; candidate `297e0f1f899c3c00c7b3f937c075e6bf87638b01`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-820-train.json).
  [Validation evidence](evaluation_results/cycle-820-valid.json).
  [Narrative](full_log.md#cycle-820--cycle-799-plus-iterative-b-on-connected-1830-esterphenol).
- Cycle 821; candidate `d1cac28f8670615109549be4d0850d8081a66a97`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-821-train.json).
  [Validation evidence](evaluation_results/cycle-821-valid.json).
  [Narrative](full_log.md#cycle-821--cycle-799-plus-wd070-on-connected-1830-esterphenol).
- Cycle 822; candidate `09904376333c01ac91d9bf62526f986443f578b1`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-822-train.json).
  [Validation evidence](evaluation_results/cycle-822-valid.json).
  [Narrative](full_log.md#cycle-822--cycle-799-plus-wd23-on-connected-1830-esterphenol).
- Cycle 823; candidate `efdf11b0fe17aa72079638d1bc09b50db49769f4`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-823-train.json).
  [Validation evidence](evaluation_results/cycle-823-valid.json).
  [Narrative](full_log.md#cycle-823--cycle-799-plus-wb070-on-cutoff-joined-1830-esterphenol).
- Cycle 824; candidate `76e322a1690cd6dd9b02b6d9b34e38b529c4f78f`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-824-train.json).
  [Validation evidence](evaluation_results/cycle-824-valid.json).
  [Narrative](full_log.md#cycle-824--cycle-799-plus-wb23-on-cutoff-joined-1830-esterphenol).
- Cycle 825; candidate `b79d73f07cc36a4fd4c11443aed4e04beea6d271`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-825-train.json).
  [Validation evidence](evaluation_results/cycle-825-valid.json).
  [Narrative](full_log.md#cycle-825--cycle-799-plus-wb05-on-cutoff-joined-1830-esterphenol).
- Cycle 826; candidate `8d3373e5aa19acb4866c5cd7275313b394115f33`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-826-train.json).
  [Validation evidence](evaluation_results/cycle-826-valid.json).
  [Narrative](full_log.md#cycle-826--cycle-799-plus-gdiis-start15-on-n18-allenes).
- Cycle 827; candidate `39308a9f5e8ea5ef64a97abf06ee1d030ca4c3a0`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-827-train.json).
  [Validation evidence](evaluation_results/cycle-827-valid.json).
  [Narrative](full_log.md#cycle-827--cycle-799-plus-gdiis-start15-cosine-080-on-n18-allenes).
- Cycle 828; candidate `c78edc21579c45f12f73352ab636a0d7e03b3a30`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-828-train.json).
  [Validation evidence](evaluation_results/cycle-828-valid.json).
  [Narrative](full_log.md#cycle-828--cycle-799-plus-gdiis-start15-cosine-080-skip-ρ-on-n18-allenes).
- Cycle 829; candidate `565dd1b55477ad0cdd7efeddaf790bacf56522d1`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-829-train.json).
  [Validation evidence](evaluation_results/cycle-829-valid.json).
  [Narrative](full_log.md#cycle-829--cycle-799-plus-gdiis-start15-cosine-080-skip-ρ-allow-ci0-on-n18-allenes).
- Cycle 830; candidate `8e5910117147d2fdb03320e564bed2783d655cea`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-830-train.json).
  [Validation evidence](evaluation_results/cycle-830-valid.json).
  [Narrative](full_log.md#cycle-830--cycle-799-plus-three-point-gdiis-on-n18-allenes).
- Cycle 831; candidate `a3fc89bb1e12d2dc590c2090327b598d7b9494d0`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-831-train.json).
  [Validation evidence](evaluation_results/cycle-831-valid.json).
  [Narrative](full_log.md#cycle-831--cycle-799-plus-skip-sy0-on-n18-allenes).
- Cycle 832; candidate `7b44c1e45085f0283a77868c292e389c2ce6527f`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-832-train.json).
  [Validation evidence](evaluation_results/cycle-832-valid.json).
  [Narrative](full_log.md#cycle-832--cycle-799-plus-rfo-from-start-on-n18-allenes).
- Cycle 833; candidate `1ad0a54dd67b9fa8d5c4953dac08d51691679aac`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-833-train.json).
  [Validation evidence](evaluation_results/cycle-833-valid.json).
  [Narrative](full_log.md#cycle-833--cycle-799-plus-gediis-after-15-on-n18-allenes).
- Cycle 834; candidate `d6ec43a0ceb380128ec614d039f0f95b3c5c7ac0`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-834-train.json).
  [Validation evidence](evaluation_results/cycle-834-valid.json).
  [Narrative](full_log.md#cycle-834--cycle-799-plus-newton-metric-gdiis-after-15-on-n18-allenes).
- Cycle 835; candidate `b8ee13ffdee1a43676945ae9f54cc37307c65eab`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-835-train.json).
  [Validation evidence](evaluation_results/cycle-835-valid.json).
  [Narrative](full_log.md#cycle-835--cycle-799-plus-bfgs_auto-from-start-on-n18-allenes).
- Cycle 836; candidate `061f0ee98fc67eed13be42fac4590731889e1673`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-836-train.json).
  [Validation evidence](evaluation_results/cycle-836-valid.json).
  [Narrative](full_log.md#cycle-836--cycle-799-plus-psb-hessian-from-start-on-n18-allenes).
- Cycle 837; candidate `651f30ed5061890df7b43a368041f1759f4a577f`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-837-train.json).
  [Validation evidence](evaluation_results/cycle-837-valid.json).
  [Narrative](full_log.md#cycle-837--cycle-799-plus-wa080-on-n18-allenes).
- Cycle 838; candidate `e9dc91c68db308ab25a7344fe7960c425b4f3908`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-838-train.json).
  [Validation evidence](evaluation_results/cycle-838-valid.json).
  [Narrative](full_log.md#cycle-838--cycle-799-plus-dummy-angle-h0-008-on-n18-allenes).
- Cycle 839; candidate `341062f8babaa9517f5a69254b962ba5ca227b54`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-839-train.json).
  [Validation evidence](evaluation_results/cycle-839-valid.json).
  [Narrative](full_log.md#cycle-839--cycle-799-plus-dummy-dihedral-h0-015-on-n18-allenes).
- Cycle 840; candidate `fad0b3502d54a1ab4e6ac19a559750895d01ce89`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-840-train.json).
  [Validation evidence](evaluation_results/cycle-840-valid.json).
  [Narrative](full_log.md#cycle-840--cycle-799-plus-alkyne-soft-dummy-dihedral-h0-010-on-n18-allenes).
- Cycle 841; candidate `981f15dab25ff38648c7f2bfc64248bb4a8d1408`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-841-train.json).
  [Validation evidence](evaluation_results/cycle-841-valid.json).
  [Narrative](full_log.md#cycle-841--cycle-799-plus-alkyne-soft-dummy-dihedral-h0-014-on-n18-allenes).
- Cycle 842; candidate `f796db5427602b93ccdbbe9441403a33f96a3d69`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-842-train.json).
  [Validation evidence](evaluation_results/cycle-842-valid.json).
  [Narrative](full_log.md#cycle-842--cycle-799-plus-adj-dummy-placement-on-n18-allenes).
- Cycle 843; candidate `45ccc84238209b62d6e99bb2a06e348ba8165b88`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-843-train.json).
  [Validation evidence](evaluation_results/cycle-843-valid.json).
  [Narrative](full_log.md#cycle-843--cycle-799-plus-trust-growth-after-15-on-n18-allenes).
- Cycle 844; candidate `4d5828abe9494271261c5f30d282ad389eecaddd`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-844-train.json).
  [Validation evidence](evaluation_results/cycle-844-valid.json).
  [Narrative](full_log.md#cycle-844--cycle-799-plus-delta0-015-on-n18-allenes).
- Cycle 845; candidate `fda962360bb5187a4f22f6c8463df5888fc460b9`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-845-train.json).
  [Validation evidence](evaluation_results/cycle-845-valid.json).
  [Narrative](full_log.md#cycle-845--cycle-799-plus-cartesian-internals-on-n18-allenes).
- Cycle 846; candidate `b68cfc3a065268a5ab54d4280832b2a0de5ac3c6`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-846-train.json).
  [Validation evidence](evaluation_results/cycle-846-valid.json).
  [Narrative](full_log.md#cycle-846--cycle-799-plus-cc-stretch-h0-010-on-n18-allenes).
- Cycle 847; candidate `efd924885bde6a5dd53b4520eda33ba001626f4a`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-847-train.json).
  [Validation evidence](evaluation_results/cycle-847-valid.json).
  [Narrative](full_log.md#cycle-847--cycle-799-plus-rho_dec5-on-n18-allenes).
- Cycle 848; candidate `37ecd6c19b5b9728a7b7ea708b9db94562d3f082`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-848-train.json).
  [Validation evidence](evaluation_results/cycle-848-valid.json).
  [Narrative](full_log.md#cycle-848--cycle-799-plus-e0-dummy-placement-on-n18-allenes).
- Cycle 849; candidate `83ed8a324f3bfa9ef43ef5e826ca3464ad1bf2f7`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-849-train.json).
  [Validation evidence](evaluation_results/cycle-849-valid.json).
  [Narrative](full_log.md#cycle-849--cycle-799-plus-alcohol-co-stretch-h0-010-on-n18-allenes).
- Cycle 850; candidate `955e51ed3a76dc7ac7763392431d93c1de091406`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-850-train.json).
  [Validation evidence](evaluation_results/cycle-850-valid.json).
  [Narrative](full_log.md#cycle-850--cycle-799-plus-predicted-uphill-step-halving-on-n18-allenes).
- Cycle 851; candidate `802ffd9243c7ab712d4e8c60c5fd64e35a81aa5e`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-851-train.json).
  [Validation evidence](evaluation_results/cycle-851-valid.json).
  [Narrative](full_log.md#cycle-851--cycle-799-plus-alcohol-allene-cc-stretch-h0-010-on-n18-allenes).
- Cycle 852; candidate `1f690282a9afd8699ddb709284855dfffcb1c98b`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-852-train.json).
  [Validation evidence](evaluation_results/cycle-852-valid.json).
  [Narrative](full_log.md#cycle-852--cycle-799-plus-delta0-015-on-connected-1830-esterphenol).
- Cycle 853; candidate `98418401c6c49fc9c38bb8608551788adc55ba4d`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-853-train.json).
  [Validation evidence](evaluation_results/cycle-853-valid.json).
  [Narrative](full_log.md#cycle-853--cycle-799-plus-predicted-uphill-step-halving-on-connected-1830-esterphenol).
- Cycle 854; candidate `dea914bd786433ebeeb3f4ddf1656b44171544be`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-854-train.json).
  [Validation evidence](evaluation_results/cycle-854-valid.json).
  [Narrative](full_log.md#cycle-854--cycle-799-plus-rho_dec5-on-connected-1830-esterphenol).
- Cycle 855; candidate `4f59b8f7c410a5999bebf3ada9e220346e3ae939`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-855-train.json).
  [Validation evidence](evaluation_results/cycle-855-valid.json).
  [Narrative](full_log.md#cycle-855--cycle-799-plus-trust-growth-after-15-on-connected-1830-esterphenol).
- Cycle 856; candidate `8f6e397dc5ceb7175694e15441ff739ca128edaf`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-856-train.json).
  [Validation evidence](evaluation_results/cycle-856-valid.json).
  [Narrative](full_log.md#cycle-856--cycle-799-plus-bfgs_auto-from-start-on-connected-1830-esterphenol).
- Cycle 857; candidate `797ce9e00becc5cc80911d7934017038d0945b80`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-857-train.json).
  [Validation evidence](evaluation_results/cycle-857-valid.json).
  [Narrative](full_log.md#cycle-857--cycle-799-plus-wo070-on-connected-1830-esterphenol).
- Cycle 858; candidate `1be65f5b944032408128e8cb011335ba0ffd942c`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-858-train.json).
  [Validation evidence](evaluation_results/cycle-858-valid.json).
  [Narrative](full_log.md#cycle-858--cycle-799-plus-newton-metric-gdiis-after-15-on-connected-1830-esterphenol).
- Cycle 859; candidate `9c88f1bee738cba784192e2f3fad32469bdf1b58`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-859-train.json).
  [Validation evidence](evaluation_results/cycle-859-valid.json).
  [Narrative](full_log.md#cycle-859--cycle-799-plus-gediis-after-15-on-connected-1830-esterphenol).
- Cycle 860; candidate `de368792b8e4d7dc8e5b1190c15e81653275edaa`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-860-train.json).
  [Validation evidence](evaluation_results/cycle-860-valid.json).
  [Narrative](full_log.md#cycle-860--cycle-799-plus-e0-dummy-placement-on-connected-1830-esterphenol).
- Cycle 861; candidate `f92f5444b290dd4e37b49f11e31b74bdd5ccce5d`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-861-train.json).
  [Validation evidence](evaluation_results/cycle-861-valid.json).
  [Narrative](full_log.md#cycle-861--cycle-799-plus-phenol-co-stretch-h0-010-on-connected-1830-esterphenol).
- Cycle 862; candidate `2fd763ba21fc499ba9edc21582b672ab0fcdb461`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-862-train.json).
  [Validation evidence](evaluation_results/cycle-862-valid.json).
  [Narrative](full_log.md#cycle-862--cycle-799-plus-baker-hessian_shift-01-eh-on-n18-allenes).
- Cycle 863; candidate `6f0cd771393a771b8e1f9b524f05c22fa6f8fef8`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-863-train.json).
  [Validation evidence](evaluation_results/cycle-863-valid.json).
  [Narrative](full_log.md#cycle-863--cycle-799-plus-cartesian-internals-on-connected-1830-esterphenol).
- Cycle 864; candidate `5602b7fb5acb6c203c8c0bb17ae36f22ce4e5394`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-864-train.json).
  [Validation evidence](evaluation_results/cycle-864-valid.json).
  [Narrative](full_log.md#cycle-864--cycle-799-plus-iterative-b-on-connected-1230-thiosulfonates).
- Cycle 865; candidate `b8c51f52381b6f3370ff06d0be66884225eb9ee1`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision discard.
  [Training evidence](evaluation_results/cycle-865-train.json).
  [Narrative](full_log.md#cycle-865--cycle-799-plus-cartesian-internals-on-connected-1230-thiosulfonates).
- Cycle 866; candidate `7f01e3238a870ba4c65d843017de4253ab9f2f73`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-866-train.json).
  [Validation evidence](evaluation_results/cycle-866-valid.json).
  [Narrative](full_log.md#cycle-866--cycle-799-plus-helgaker-001-eh-floor-on-connected-1230-thiosulfonates).
- Cycle 867; candidate `69672f4d502bea2c82f4c10fb13035362f9753b0`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-867-train.json).
  [Validation evidence](evaluation_results/cycle-867-valid.json).
  [Narrative](full_log.md#cycle-867--cycle-799-plus-baker-hessian_shift-01-eh-on-connected-1230-thiosulfonates).
- Cycle 868; candidate `b3017cf1c0dba24e28515e84742eb2f095aa1fc6`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-868-train.json).
  [Validation evidence](evaluation_results/cycle-868-valid.json).
  [Narrative](full_log.md#cycle-868--cycle-799-plus-trustregion-after-15-on-connected-1230-thiosulfonates).
- Cycle 869; candidate `e3421bdcf1d6aa845f83b32918711bda541e7b58`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-869-train.json).
  [Validation evidence](evaluation_results/cycle-869-valid.json).
  [Narrative](full_log.md#cycle-869--cycle-799-plus-wo070-on-connected-1230-thiosulfonates).
- Cycle 870; candidate `b94cc9528e9f42e9a3ece3fe9f3451c92fd10132`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Implementation](ideas/sella-thiosulfonate-gediis/b94cc9528e9f42e9a3ece3fe9f3451c92fd10132/algo.py).
  [Training evidence](evaluation_results/cycle-870-train.json).
  [Validation evidence](evaluation_results/cycle-870-valid.json).
  [Narrative](full_log.md#cycle-870--cycle-799-plus-gediis-after-15-on-connected-1230-thiosulfonates).
- Cycle 871; candidate `8ce3e248bc5b1b6ba06155b7e6bcb83ff0204dda`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-871-train.json).
  [Validation evidence](evaluation_results/cycle-871-valid.json).
  [Narrative](full_log.md#cycle-871--cycle-870-plus-skip-sy0-on-connected-1830-esterphenol).
- Cycle 872; candidate `14bb5e5bbffdb36c8c549a394e218cd626eb3a3f`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-872-train.json).
  [Validation evidence](evaluation_results/cycle-872-valid.json).
  [Narrative](full_log.md#cycle-872--cycle-870-plus-baker-hessian_shift-01-eh-on-connected-1830-esterphenol).
- Cycle 873; candidate `160e4930212060fd6f944fe6e124527a5f73e2fa`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-873-train.json).
  [Validation evidence](evaluation_results/cycle-873-valid.json).
  [Narrative](full_log.md#cycle-873--cycle-870-plus-gediis-after-15-on-n18-allenes).
- Cycle 874; candidate `e478d0c8c1a504a33dfa428d1e449f336c4b031f`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision keep.
  [Implementation](algo.py).
  [Training evidence](evaluation_results/cycle-874-train.json).
  [Validation evidence](evaluation_results/cycle-874-valid.json).
  [Narrative](full_log.md#cycle-874--cycle-870-plus-gediis-after-20-on-connected-1830-esterphenol).
- Cycle 875; candidate `8b3f8f9d4a76a249528070bd13d5d0d730da5884`; champion `e478d0c8c1a504a33dfa428d1e449f336c4b031f`; decision discard.
  [Training evidence](evaluation_results/cycle-875-train.json).
  [Narrative](full_log.md#cycle-875--cycle-874-plus-gediis-after-18-on-n18-allenes).
- Cycle 876; candidate `5d7ce9f56889c887d45a06cd801f99712ff35201`; champion `e478d0c8c1a504a33dfa428d1e449f336c4b031f`; decision invalid.
  [Training evidence](evaluation_results/cycle-876-train.json).
  [Narrative](full_log.md#cycle-876--cycle-874-plus-gediis-after-20-on-saturated-1830-alkanephenol).
- Cycle 877; candidate `19595aa4d12cc618db39f87b77b6cb8e13c4600c`; champion `e478d0c8c1a504a33dfa428d1e449f336c4b031f`; decision discard.
  [Training evidence](evaluation_results/cycle-877-train.json).
  [Narrative](full_log.md#cycle-877--cycle-874-plus-baker-hessian_shift-01-eh-on-saturated-1830-alkanephenol).
- Cycle 878; candidate `343daba56119ba8b8df5ebd01c0b643a978e01d4`; champion `e478d0c8c1a504a33dfa428d1e449f336c4b031f`; decision discard.
  [Training evidence](evaluation_results/cycle-878-train.json).
  [Narrative](full_log.md#cycle-878--cycle-874-plus-wo070-on-saturated-1830-alkanephenol).
- Cycle 879; candidate `468ec78cb028d0ad080cbcb6513fc0cecdf1bd48`; champion `e478d0c8c1a504a33dfa428d1e449f336c4b031f`; decision discard.
  [Training evidence](evaluation_results/cycle-879-train.json).
  [Narrative](full_log.md#cycle-879--cycle-874-plus-trustregion-after-20-on-saturated-1830-alkanephenol).
- Cycle 880; candidate `a2fff374d01b5650764441616a761965b10f7415`; champion `e478d0c8c1a504a33dfa428d1e449f336c4b031f`; decision discard.
  [Training evidence](evaluation_results/cycle-880-train.json).
  [Narrative](full_log.md#cycle-880--cycle-874-plus-skip-sy0-on-saturated-1830-alkanephenol).
- Cycle 881; candidate `2f2ffe5e8376aff3e9bccee4f767a9a5b38b0ad2`; champion `e478d0c8c1a504a33dfa428d1e449f336c4b031f`; decision invalid.
  [Training evidence](evaluation_results/cycle-881-train.json).
  [Narrative](full_log.md#cycle-881--cycle-874-plus-gediis-after-20-on-n80-sulfonamides).
- Cycle 882; candidate `667fcae536e895f39d347e5fa58e38b3f237fe35`; champion `e478d0c8c1a504a33dfa428d1e449f336c4b031f`; decision discard.
  [Training evidence](evaluation_results/cycle-882-train.json).
  [Narrative](full_log.md#cycle-882--cycle-874-plus-newton-metric-gdiis-after-20-on-saturated-1830-alkanephenol).
- Cycle 883; candidate `0984652074c88c62b1ab5836cce0dfb0cde36e46`; champion `e478d0c8c1a504a33dfa428d1e449f336c4b031f`; decision discard.
  [Training evidence](evaluation_results/cycle-883-train.json).
  [Narrative](full_log.md#cycle-883--cycle-874-plus-wa070-on-saturated-1830-alkanephenol).
- Cycle 884; candidate `7e7e93e3f483b6594c9471c45e5890155d6adf53`; champion `e478d0c8c1a504a33dfa428d1e449f336c4b031f`; decision discard.
  [Training evidence](evaluation_results/cycle-884-train.json).
  [Narrative](full_log.md#cycle-884--cycle-874-plus-wd070-on-saturated-1830-alkanephenol).

## sella-disconnected-phenol-iterative: iterative B⁺ on disconnected 18–30 aryl phenol
Status: incorporated

**Hypothesis:** Widening cycle 724’s carboxylate+phenol tag to any disconnected 18–30 aryl phenol also tags leftover alkanes–phenol and other phenol dimers whose iterative B⁺ Newton realization is cheaper than geodesic LSODA.

**Outcome and uncertainty:** Cycle 728 **keep** vs `40e894e`. Train V, Δ=+6.66e-4 (leftover 363892164 42→33 plus phenol–pyrrole 45→44). Valid V, Δ=+2.93e-4 (carboxylates–phenol 28→27, amides–phenol 43→40, benzene–phenol 34→33). Leftover alkanes–phenol stays 35 extra=1 with an energy twitch. New champion `133f73f`.

**Reason to revisit:** Incorporated. Do not drop the disconnected 18–30 aryl-phenol iterative gate. Do not interpolate skip GDIIS/flowchart/RFO on this wider tag (flowchart extras leftover carboxylates–phenol).

**Next experiment:** leftover 433943527 extra=1 (`n_ref=20`) still passes train δ on −1. Helgaker 0.01 Eh |λ| floor on connected 12–30 thiosulfonates stacked with cycle 799 vs `133f73f`.

**Attempts:**
- Cycle 728; candidate `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; champion `40e894e6688cea8b89f66cdb7eadd4a930ff5dee`; decision keep.
  [Implementation](algo.py).
  [Training evidence](evaluation_results/cycle-728-train.json).
  [Validation evidence](evaluation_results/cycle-728-valid.json).
  [Narrative](full_log.md#cycle-728--cycle-715-plus-iterative-b-on-disconnected-1830-aryl-phenol).

## sella-n80-sulfonamide-rfo: Banerjee RFO after 20 on n≥80 sulfonamides
Status: deferred

**Hypothesis:** Train venetoclax extra=6 lives in the QN window before n≥80 RFO@45. Banerjee RFO after 20 on connected n≥80 4-coordinate sulfonamide S {two 1-coordinate O, N, C} tags venetoclax uniquely (paliperidone has no S) and may take ≥3 leftover steps.

**Outcome and uncertainty:** Cycle 751 discard vs `133f73f`. Train V, Δ=+4.18e-5. Only venetoclax 57→56, energy-safe. Paliperidone stays 73. Cycle 752 RFO@10 extras leftover 57→59. Cycle 753 from-start `wd=0.70` hops venetoclax (57→61, +1.38 kcal/mol). Cycle 754 delayed `wd` extras leftover 57→63. RFO@20 is a real one-call save; `wd` is not. Cycle 772 `sigma_inc=1.20` after 20 bit-identical venetoclax at 57. Cycle 773 `rho_inc=1.5` after 20 bit-identical venetoclax at 57. Cycle 774 skip GDIIS bit-identical venetoclax at 57. Cycle 775 iterative B⁺ energy-twitch venetoclax at 57. Cycle 776 Helgaker 0.01 with copy bit-identical venetoclax at 57. Cycle 777 exact geodesic extras venetoclax 57→61, Δ=−1.67e-4. Cycle 779 skip s·y<0 extras venetoclax 57→59, Δ=−8.36e-5. Cycle 881 GEDIIS after 20 invalid: leftover venetoclax hops 56→82 (+0.0088 kcal). Cycle 885 TrustRegion after 20 **non_generalizable**: leftover venetoclax 56→51, train Δ=+2.09e-4, valid Δ=0.

**Reason to revisit:** Cycle 897 **keep** stacked leftover venetoclax TrustRegion (56→51) with leftover alkanes–phenol `delta0=0.15` (valid 35→30). New champion `f9d546f`. Do not drop sulfonamide TrustRegion after 20 or leftover alkanes–phenol `delta0=0.15`. Close leftover 135169446 Newton-metric interpolants. leftover 135169446 extra=2 still independently passes valid δ on −1 but leftover-135169446-only interpolants cannot pass Δ_train vs `f9d546f`. Cycle 894 leftover alkanes–phenol `rho_dec=5` was a no-op at `delta0=0.1`.

**Next experiment:** MaxInternalStep `wo=0.70` on disconnected 18–30 saturated alkane–phenol stacked with cycle 897 vs `f9d546f`. `delta0=0.15` enlarges leftover’s first dummy extras so `wo` may now bind. Train leftover 35→33 plus valid leftover 30→28 would keep.

**Attempts:**
- Cycle 751; candidate `053aaa616c800772b7c2e399e38785298ad7eafd`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision discard.
  [Implementation](ideas/sella-n80-sulfonamide-rfo/053aaa616c800772b7c2e399e38785298ad7eafd/algo.py).
  [Training evidence](evaluation_results/cycle-751-train.json).
  [Narrative](full_log.md#cycle-751--cycle-728-plus-banerjee-rfo-after-20-on-n80-sulfonamides).
- Cycle 752; candidate `df108dd35e2927efc6cf4e4ceeba71323a61653c`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision discard.
  [Training evidence](evaluation_results/cycle-752-train.json).
  [Narrative](full_log.md#cycle-752--cycle-728-plus-banerjee-rfo-after-10-on-n80-sulfonamides).
- Cycle 753; candidate `acb835c6b2d507e92abb8c168fa6b3406566b7c8`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision invalid.
  [Training evidence](evaluation_results/cycle-753-train.json).
  [Narrative](full_log.md#cycle-753--cycle-751-plus-wd070-on-n80-sulfonamides).
- Cycle 754; candidate `ad0986223de8bda69ca4a3e3e68a34d09d4f6f09`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision discard.
  [Training evidence](evaluation_results/cycle-754-train.json).
  [Narrative](full_log.md#cycle-754--cycle-751-plus-delayed-wd070-after-20-on-n80-sulfonamides).
- Cycle 772; candidate `163a2f0d544bb43970a396755f59f8fb2f424667`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision discard.
  [Training evidence](evaluation_results/cycle-772-train.json).
  [Narrative](full_log.md#cycle-772--cycle-728-plus-sigma_inc120-after-20-on-n80-sulfonamides).
- Cycle 773; candidate `07b7ce4cc471c0c11d26375c0a2e93af49c0a545`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision discard.
  [Training evidence](evaluation_results/cycle-773-train.json).
  [Narrative](full_log.md#cycle-773--cycle-728-plus-rho_inc15-after-20-on-n80-sulfonamides).
- Cycle 774; candidate `c8157caf6d2dda2fbc6d34bd0851d769cb7a03eb`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision discard.
  [Training evidence](evaluation_results/cycle-774-train.json).
  [Narrative](full_log.md#cycle-774--cycle-728-plus-skip-gdiis-on-n80-sulfonamides).
- Cycle 775; candidate `8a84da51efc448e4a5aead3e4366851ccd64a0f1`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision discard.
  [Training evidence](evaluation_results/cycle-775-train.json).
  [Narrative](full_log.md#cycle-775--cycle-728-plus-iterative-cartesian-b-on-n80-sulfonamides).
- Cycle 776; candidate `21499d9e7f84cd87882a346ee3d8d4a82d4d2d71`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision discard.
  [Training evidence](evaluation_results/cycle-776-train.json).
  [Narrative](full_log.md#cycle-776--cycle-728-plus-helgaker-λ-floor-001-eh-with-projected-hessian-copy-on-n80-sulfonamides).
- Cycle 777; candidate `2535415580b41dc0039ada8b3f581b9caaa7e7b0`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision discard.
  [Training evidence](evaluation_results/cycle-777-train.json).
  [Narrative](full_log.md#cycle-777--cycle-728-plus-exact-geodesic-on-n80-sulfonamides).
- Cycle 779; candidate `35f411f119887cadb1efd2444338ed695450e805`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision discard.
  [Training evidence](evaluation_results/cycle-779-train.json).
  [Narrative](full_log.md#cycle-779--cycle-728-plus-skip-sy0-hessian-updates-from-start-on-n80-sulfonamides).
- Cycle 780; candidate `0da2060f49f7115d7a6e96fd0d18c9d5e5f165c9`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision discard.
  [Training evidence](evaluation_results/cycle-780-train.json).
  [Narrative](full_log.md#cycle-780--cycle-728-plus-gdiis-start-15-on-n80-sulfonamides).
- Cycle 881; candidate `2f2ffe5e8376aff3e9bccee4f767a9a5b38b0ad2`; champion `e478d0c8c1a504a33dfa428d1e449f336c4b031f`; decision invalid.
  [Training evidence](evaluation_results/cycle-881-train.json).
  [Narrative](full_log.md#cycle-881--cycle-874-plus-gediis-after-20-on-n80-sulfonamides).
- Cycle 885; candidate `125dd80d91a8b2b36292fc5919f5d14bf28d9189`; champion `e478d0c8c1a504a33dfa428d1e449f336c4b031f`; decision non_generalizable.
  [Implementation](ideas/sella-n80-sulfonamide-trust-region/125dd80d91a8b2b36292fc5919f5d14bf28d9189/algo.py).
  [Training evidence](evaluation_results/cycle-885-train.json).
  [Validation evidence](evaluation_results/cycle-885-valid.json).
  [Narrative](full_log.md#cycle-885--cycle-874-plus-trustregion-after-20-on-n80-sulfonamides).
- Cycle 886; candidate `e67cc92409d39e53754157c0a26bb3e3e2cfe7ef`; champion `e478d0c8c1a504a33dfa428d1e449f336c4b031f`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-886-train.json).
  [Validation evidence](evaluation_results/cycle-886-valid.json).
  [Narrative](full_log.md#cycle-886--cycle-885-plus-gediis-after-18-on-n18-allenes).
- Cycle 887; candidate `fadb3e59a947a633d99880c79f19624edd5cca58`; champion `e478d0c8c1a504a33dfa428d1e449f336c4b031f`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-887-train.json).
  [Validation evidence](evaluation_results/cycle-887-valid.json).
  [Narrative](full_log.md#cycle-887--cycle-885-plus-wo070-on-n18-allenes).
- Cycle 888; candidate `fb9e98cf379b62aedc9a0d59686ddc5a43bd9a95`; champion `e478d0c8c1a504a33dfa428d1e449f336c4b031f`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-888-train.json).
  [Validation evidence](evaluation_results/cycle-888-valid.json).
  [Narrative](full_log.md#cycle-888--cycle-885-plus-skip-dummy-limiter-on-n18-allenes).
- Cycle 889; candidate `db62ff3b4d6a5a0b53216e4b639c1f28b5cfe10d`; champion `e478d0c8c1a504a33dfa428d1e449f336c4b031f`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-889-train.json).
  [Validation evidence](evaluation_results/cycle-889-valid.json).
  [Narrative](full_log.md#cycle-889--cycle-885-plus-dummy-limiter-090-on-n18-allenes).
- Cycle 890; candidate `bad66261f73f1fc780d01975ffec3f36d6eb5033`; champion `e478d0c8c1a504a33dfa428d1e449f336c4b031f`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-890-train.json).
  [Validation evidence](evaluation_results/cycle-890-valid.json).
  [Narrative](full_log.md#cycle-890--cycle-885-plus-flowchart-after-17-on-n18-allenes).
- Cycle 891; candidate `8475e23bbf3b2dd587f6097165d1ce3b0b102f8e`; champion `e478d0c8c1a504a33dfa428d1e449f336c4b031f`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-891-train.json).
  [Validation evidence](evaluation_results/cycle-891-valid.json).
  [Narrative](full_log.md#cycle-891--cycle-885-plus-trustregion-after-18-on-n18-allenes).
- Cycle 892; candidate `e001c82b9a3e5369e03485df03497b2980847512`; champion `e478d0c8c1a504a33dfa428d1e449f336c4b031f`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-892-train.json).
  [Validation evidence](evaluation_results/cycle-892-valid.json).
  [Narrative](full_log.md#cycle-892--cycle-885-plus-gediis-after-16-on-n18-allenes).
- Cycle 893; candidate `0c3146ca9c1aab24e8cd9a1ea19b983f90e8d4f9`; champion `e478d0c8c1a504a33dfa428d1e449f336c4b031f`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-893-train.json).
  [Validation evidence](evaluation_results/cycle-893-valid.json).
  [Narrative](full_log.md#cycle-893--cycle-885-plus-newton-metric-gdiis-after-16-on-n18-allenes).
- Cycle 894; candidate `dc1ee1ee923800f0b690a6a8676095ded22ae36e`; champion `e478d0c8c1a504a33dfa428d1e449f336c4b031f`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-894-train.json).
  [Validation evidence](evaluation_results/cycle-894-valid.json).
  [Narrative](full_log.md#cycle-894--cycle-885-plus-rho_dec5-on-saturated-1830-alkanephenol).
- Cycle 895; candidate `efe3f6e83bfec5bedeb9c040cf12e193380dc56f`; champion `e478d0c8c1a504a33dfa428d1e449f336c4b031f`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-895-train.json).
  [Validation evidence](evaluation_results/cycle-895-valid.json).
  [Narrative](full_log.md#cycle-895--cycle-885-plus-predicted-uphill-step-halving-on-saturated-1830-alkanephenol).
- Cycle 896; candidate `788cbf166612eed84c57872cbd59bc318b528edb`; champion `e478d0c8c1a504a33dfa428d1e449f336c4b031f`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-896-train.json).
  [Validation evidence](evaluation_results/cycle-896-valid.json).
  [Narrative](full_log.md#cycle-896--cycle-885-plus-bfgs_auto-before-20-on-saturated-1830-alkanephenol).
- Cycle 897; candidate `f9d546f8780f42c2f7178ac63711672ce38a5e77`; champion `e478d0c8c1a504a33dfa428d1e449f336c4b031f`; decision keep.
  [Implementation](algo.py).
  [Training evidence](evaluation_results/cycle-897-train.json).
  [Validation evidence](evaluation_results/cycle-897-valid.json).
  [Narrative](full_log.md#cycle-897--cycle-885-plus-delta0015-on-saturated-1830-alkanephenol).

## sella-thiosulfonate-flowchart-hessian: Schlegel flowchart after 15 on 12–30 thiosulfonates
Status: revisiting

**Hypothesis:** Train leftover 433943527 extra=1 (`n_ref=20`) would pass train δ on −1. Gating Schlegel flowchart after 15 to connected 12–30 4-coordinate S with two 1-coordinate O and one S neighbor tags leftover uniquely.

**Outcome and uncertainty:** Cycle 708 discard vs `40e894e`. Train V, Δ=0. Leftover stays 21 with an energy twitch. Class fires and is not the leftover n_steps limiter. Cycle 713 RFO after 15 is bit-identical. Cycle 729 iterative B⁺ vs `133f73f`: Δ=0, leftover energy twitch at 21. Iterative is not the leftover limiter. Cycle 735 0.18 trust floor bit-identical. Cycle 740 σ_dec=0.95 bit-identical. Cycle 755 from-start skip s·y<0 vs `133f73f`: Δ=0, bit-identical including leftover at 21; leftover Hessian updates are never s·y<0. Cycle 756 from-start Powell vs `133f73f`: Δ=0, bit-identical; leftover updates always have s·y ≥ 0.2 s·Bs. Cycle 757 from-start flowchart vs `133f73f`: Δ=−1.07e-4, leftover 21→22 energy-safe. Cycle 778 Helgaker 0.01 with copy vs `133f73f`: Δ=0, bit-identical leftover at 21; leftover |λ| ≥ 0.01.

**Reason to revisit:** Not as a thiosulfonate flowchart (after-15 twitch, from-start extra), RFO, iterative, late-trust, MIS, skip s·y<0, Powell damping, or Helgaker |λ| floor. Leftover 433943527 Hessian-update interpolants extras leftover. Close this unique-tag Hessian family.

**Next experiment:** Do not interpolate leftover 433943527 Hessian updates or Helgaker floors. Stack cycle 751 RFO@20 with leftover 363892164 −1 vs `133f73f`. See GDIIS start 15 on n≥80 sulfonamides.

**Attempts:**
- Cycle 708; candidate `cbea751286a5c6564603016af5db4cbddbeeb2f2`; champion `40e894e6688cea8b89f66cdb7eadd4a930ff5dee`; decision discard.
  [Training evidence](evaluation_results/cycle-708-train.json).
  [Narrative](full_log.md#cycle-708--cycle-704-plus-schlegel-flowchart-hessian-after-15-on-12-30-thiosulfonates).
- Cycle 713; candidate `f8f9ccda9019f2208355b260cd9e357bd383fcab`; champion `40e894e6688cea8b89f66cdb7eadd4a930ff5dee`; decision discard.
  [Training evidence](evaluation_results/cycle-713-train.json).
  [Narrative](full_log.md#cycle-713--cycle-704-plus-banerjee-rfo-after-15-on-12-30-thiosulfonates).
- Cycle 750; candidate `f51168331fd8a03f283dc2265f392f1dd1771a15`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision discard.
  [Training evidence](evaluation_results/cycle-750-train.json).
  [Narrative](full_log.md#cycle-750--cycle-728-plus-skip-sy0-hessian-updates-after-20-on-12-30-thiosulfonates).
- Cycle 755; candidate `04d1d6ff4675fbcf3120d53a776d81eaae0a9802`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision discard.
  [Training evidence](evaluation_results/cycle-755-train.json).
  [Narrative](full_log.md#cycle-755--cycle-728-plus-skip-sy0-hessian-updates-from-start-on-12-30-thiosulfonates).
- Cycle 756; candidate `413afebb042ad860d912edec1715e27c3f929850`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision discard.
  [Training evidence](evaluation_results/cycle-756-train.json).
  [Narrative](full_log.md#cycle-756--cycle-728-plus-powell-damped-hessian-updates-from-start-on-12-30-thiosulfonates).
- Cycle 757; candidate `56e19389e8207e5177fcd80918cc70e8cfcb6f4b`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision discard.
  [Training evidence](evaluation_results/cycle-757-train.json).
  [Narrative](full_log.md#cycle-757--cycle-728-plus-schlegel-flowchart-hessian-from-start-on-12-30-thiosulfonates).
- Cycle 778; candidate `50f17400fb68cd305a5765742c6dce9c5c813ad8`; champion `133f73fbcaaa6460616dd98afd89f2b61c2dee2b`; decision discard.
  [Training evidence](evaluation_results/cycle-778-train.json).
  [Narrative](full_log.md#cycle-778--cycle-728-plus-helgaker-λ-floor-001-eh-with-projected-hessian-copy-on-12-30-thiosulfonates).

**Attempts:**
- Cycle 708; candidate `cbea751286a5c6564603016af5db4cbddbeeb2f2`; champion `40e894e6688cea8b89f66cdb7eadd4a930ff5dee`; decision discard.
  [Training evidence](evaluation_results/cycle-708-train.json).
  [Narrative](full_log.md#cycle-708--cycle-704-plus-schlegel-flowchart-hessian-after-15-on-12-30-thiosulfonates).
- Cycle 713; candidate `f8f9ccda9019f2208355b260cd9e357bd383fcab`; champion `40e894e6688cea8b89f66cdb7eadd4a930ff5dee`; decision discard.
  [Training evidence](evaluation_results/cycle-713-train.json).
  [Narrative](full_log.md#cycle-713--cycle-704-plus-banerjee-rfo-after-15-on-12-30-thiosulfonates).

## sella-large-flowchart-hessian: Schlegel flowchart after 45 on n≥80
Status: closed

**Hypothesis:** Train venetoclax extra=6 sits on the n≥80 RFO@45 tail. Delaying Schlegel flowchart until that existing RFO gate keeps early TS-BFGS and may cheapen the floppy tail without cycle 225’s from-20 stall.

**Outcome and uncertainty:** Cycle 709 discard vs `40e894e`. Train V, Δ=−2.61e-3. paliperidone_palmitate 73→167 and venetoclax 57→58. Same stall class as cycle 225.

**Reason to revisit:** Not as an n≥80 flowchart interpolation. Do not start flowchart earlier on `_large`.

**Next experiment:** None. See Banerjee RFO after 20 on 30–80 isocyanides.

**Attempts:**
- Cycle 709; candidate `3ca186915596bf05e51980cdcdd609c08860c36e`; champion `40e894e6688cea8b89f66cdb7eadd4a930ff5dee`; decision discard.
  [Training evidence](evaluation_results/cycle-709-train.json).
  [Narrative](full_log.md#cycle-709--cycle-704-plus-schlegel-flowchart-hessian-after-45-on-n80-molecules).

## sella-isocyanide-iterative-stepper: iterative Cartesian B⁺ on 30–80 isocyanides
Status: closed

**Hypothesis:** Cycle 587 exact geodesic extras leftover 363892164. Iterative Cartesian B⁺ is the Newton realization of the same internal step and may cheapen leftover dummy-linear paths.

**Outcome and uncertainty:** Cycle 596 discard, Δ=0. Leftover 363892164 energy twitch at 42. Realization fires and is not the n_steps limiter.

**Reason to revisit:** Not as an isocyanide iterative_stepper interpolation. Combined with geodesic extra, leftover 363892164 is not Binv-realization limited.

**Next experiment:** None. See leftover 135094613 exact geodesic on n<12 P4S6.

**Attempts:**
- Cycle 596; candidate `dc2fa0382fa42a44dd59b3292ac775beeaf30751`; champion `4eb6cafb44eeb92e19753e82045a8573ed8f96cb`; decision discard.
  [Training evidence](evaluation_results/cycle-596-train.json).
  [Narrative](full_log.md#cycle-596--iterative-cartesian-b-on-30-80-isocyanides).

## sella-medium-ammonium-cnc-h0: 0.10 Ha trimethylammonium C–N–C on connected 12–30
Status: closed

**Hypothesis:** Leftover 433943527 unused 4-coordinate trimethylammonium C–N–C (N {C,C,C,C} with ≥3 methyl carbons) is the limiter after CSS/`wa`/GDIIS/geodesic twitch. GAFF `c3-n4-c3` ≈ 0.10 Ha vs Fischer ~0.16. A 1–6 cap matches C(4,2)=6.

**Outcome and uncertainty:** Cycle 595 discard, Δ=−1.066e-4. Only extra leftover 433943527 21→22, energy-safe. Soft C–N–C extras leftover.

**Reason to revisit:** Not as an ammonium C–N–C H0 scale. Do not interpolate 0.13. Leftover 433943527 valence-angle H0 is closed.

**Next experiment:** None. See leftover 363892164 iterative Cartesian B⁺ on 30–80 isocyanides.

**Attempts:**
- Cycle 595; candidate `6a203625d23c578e66fec5b0d48699831e902368`; champion `4eb6cafb44eeb92e19753e82045a8573ed8f96cb`; decision discard.
  [Training evidence](evaluation_results/cycle-595-train.json).
  [Narrative](full_log.md#cycle-595--010-ha-trimethylammonium-cnc-on-connected-12-30).

## sella-n12-wa-070: wa=0.70 on connected n<12
Status: incorporated

**Hypothesis:** Champion connected `wa=0.75` caps |s_a|≤0.133. Lowering `wa` to 0.70 on connected n<12 enlarges leftover 252089162 (`OP(O)OP(O)O`) angle steps without venetoclax/paliperidone.

**Outcome and uncertainty:** Cycle 571 NG extras P–F. Cycle 572 P–F skip train-passing valid-safe. Cycle 582 geodesic on 30–80 bis-N-oxides valid near-miss (135149595 19→18, Δ=+9.775e-5). Cycle 584 **keep** stacked geodesic with `wa=0.70` on the same bis-N-oxide class: 135149595 19→17, Δ_valid=+1.955e-4. New champion `4eb6caf`. Leftover 252089162 at-ref 37. Leftover 252656292 still +1.

**Reason to revisit:** Incorporated in champion `4eb6caf`. Do not drop n<12 `wa=0.70` except P–F or 30–80 bis-N-oxide geodesic+`wa=0.70`. After this keep, valid-only leftover 252656292 −1 fails Δ_train.

**Next experiment:** Cycle 607: exact geodesic on connected 30≤n<80 aryl-CF3 vs leftover 440723624. Support: leftover ≤21 plus another save so Δ_train≥1e-4 vs `4eb6caf`. Contradict: leftover extra/twitch, hop, or Δ fails.

**Attempts:**
- Cycle 595; candidate `6a203625d23c578e66fec5b0d48699831e902368`; champion `4eb6cafb44eeb92e19753e82045a8573ed8f96cb`; decision discard.
  [Training evidence](evaluation_results/cycle-595-train.json).
  [Narrative](full_log.md#cycle-595--010-ha-trimethylammonium-cnc-on-connected-12-30).
- Cycle 594; candidate `c55111b013c58956f6d352ba702db05eb63f9042`; champion `4eb6cafb44eeb92e19753e82045a8573ed8f96cb`; decision discard.
  [Training evidence](evaluation_results/cycle-594-train.json).
  [Narrative](full_log.md#cycle-594--wa070-on-12-30-thiosulfonates).
- Cycle 593; candidate `0cb1e63c52870914ede4998829d12b417597d0ab`; champion `4eb6cafb44eeb92e19753e82045a8573ed8f96cb`; decision discard.
  [Training evidence](evaluation_results/cycle-593-train.json).
  [Narrative](full_log.md#cycle-593--010-ha-sulfoxide-csc-cso-on-n12).
- Cycle 592; candidate `c701103c4575c4472d6e0da9d222ff1224870622`; champion `4eb6cafb44eeb92e19753e82045a8573ed8f96cb`; decision discard.
  [Training evidence](evaluation_results/cycle-592-train.json).
  [Narrative](full_log.md#cycle-592--010-ha-sps-and-psp-on-n12).
- Cycle 591; candidate `31ca0a2e3bbcceefca145f4da2749e87f41b21d7`; champion `4eb6cafb44eeb92e19753e82045a8573ed8f96cb`; decision discard.
  [Training evidence](evaluation_results/cycle-591-train.json).
  [Narrative](full_log.md#cycle-591--gdiis-start-15-on-n12-without-pf).
- Cycle 590; candidate `5260f348ed7710002751c6f0135a3063be3f13bc`; champion `4eb6cafb44eeb92e19753e82045a8573ed8f96cb`; decision discard.
  [Training evidence](evaluation_results/cycle-590-train.json).
  [Narrative](full_log.md#cycle-590--exact-geodesic-on-12-30-thiosulfonates).
- Cycle 589; candidate `f1a8783cd8b30f169f280ce07ad08687ec802470`; champion `4eb6cafb44eeb92e19753e82045a8573ed8f96cb`; decision discard.
  [Training evidence](evaluation_results/cycle-589-train.json).
  [Narrative](full_log.md#cycle-589--gdiis-start-15-on-12-30-thiosulfonates).
- Cycle 588; candidate `6f4adde1adb908c63c76267467b867d9e018715d`; champion `4eb6cafb44eeb92e19753e82045a8573ed8f96cb`; decision discard.
  [Training evidence](evaluation_results/cycle-588-train.json).
  [Narrative](full_log.md#cycle-588--dummy-angle-025-on-30-80-isocyanides).
- Cycle 587; candidate `ae5d27042b182ed233ee8d1cbc8cbaf553d05228`; champion `4eb6cafb44eeb92e19753e82045a8573ed8f96cb`; decision discard.
  [Training evidence](evaluation_results/cycle-587-train.json).
  [Narrative](full_log.md#cycle-587--exact-geodesic-on-30-80-isocyanides).
- Cycle 586; candidate `37c88b3e9c9b7ca4508b5956d7a1ee9749fd893d`; champion `4eb6cafb44eeb92e19753e82045a8573ed8f96cb`; decision discard.
  [Training evidence](evaluation_results/cycle-586-train.json).
  [Narrative](full_log.md#cycle-586--wa070-on-30-80-isocyanides).
- Cycle 585; candidate `8468791269f7f7856ce26772fc5d72df19c4724e`; champion `4eb6cafb44eeb92e19753e82045a8573ed8f96cb`; decision discard.
  [Training evidence](evaluation_results/cycle-585-train.json).
  [Narrative](full_log.md#cycle-585--expand-wa070-except-pf-from-n12-to-n18).
- Cycle 584; candidate `4eb6cafb44eeb92e19753e82045a8573ed8f96cb`; champion `5b183d43d9c42b918e209ab6e11471afe1f07ba5`; decision keep.
  [Implementation](algo.py).
  [Training evidence](evaluation_results/cycle-584-train.json).
  [Validation evidence](evaluation_results/cycle-584-valid.json).
  [Narrative](full_log.md#cycle-584--582-geodesic-plus-wa070-on-30-80-bis-n-oxides).
- Cycle 583; candidate `0f63abc62bae07dda38924c28f2da9e66049016c`; champion `5b183d43d9c42b918e209ab6e11471afe1f07ba5`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-583-train.json).
  [Validation evidence](evaluation_results/cycle-583-valid.json).
  [Narrative](full_log.md#cycle-583--582-geodesic-plus-gdiis-start-15-on-30-80-bis-n-oxides).
- Cycle 582; candidate `ae935b297d9b25d8581203e7d383e242d66f9049`; champion `5b183d43d9c42b918e209ab6e11471afe1f07ba5`; decision non_generalizable.
  [Implementation](ideas/sella-n12-wa-070/ae935b297d9b25d8581203e7d383e242d66f9049/algo.py).
  [Training evidence](evaluation_results/cycle-582-train.json).
  [Validation evidence](evaluation_results/cycle-582-valid.json).
  [Narrative](full_log.md#cycle-582--572-stack-plus-exact-geodesic-on-30-80-bis-n-oxides).
- Cycle 581; candidate `a14e8ec02ece8124b5347745af2a3bd08a53d3cc`; champion `5b183d43d9c42b918e209ab6e11471afe1f07ba5`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-581-train.json).
  [Validation evidence](evaluation_results/cycle-581-valid.json).
  [Narrative](full_log.md#cycle-581--572-stack-plus-wa070-on-30-80-bis-n-oxides).
- Cycle 580; candidate `0b143591aab864f1cc7e6cd62d0a58924042c921`; champion `5b183d43d9c42b918e209ab6e11471afe1f07ba5`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-580-train.json).
  [Validation evidence](evaluation_results/cycle-580-valid.json).
  [Narrative](full_log.md#cycle-580--572-stack-plus-gdiis-start-15-on-30-80-bis-n-oxides-with-all_atoms).
- Cycle 579; candidate `6d504baa4c6b615b4c532159eec7195b25987540`; champion `5b183d43d9c42b918e209ab6e11471afe1f07ba5`; decision invalid.
  [Implementation](ideas/sella-n12-wa-070/6d504baa4c6b615b4c532159eec7195b25987540/algo.py).
  [Training evidence](evaluation_results/cycle-579-train.json).
  [Narrative](full_log.md#cycle-579--572-stack-plus-gdiis-start-15-on-30-80-bis-n-oxides).
- Cycle 578; candidate `2c6b78d015ae7b741094b7e897edb902c81b5482`; champion `5b183d43d9c42b918e209ab6e11471afe1f07ba5`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-578-train.json).
  [Validation evidence](evaluation_results/cycle-578-valid.json).
  [Narrative](full_log.md#cycle-578--572-stack-plus-dummy-angle-025-on-30-80-bis-n-oxides).
- Cycle 577; candidate `6c216e2caa28cf180a758e7d6a7c8f3a91273207`; champion `5b183d43d9c42b918e209ab6e11471afe1f07ba5`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-577-train.json).
  [Validation evidence](evaluation_results/cycle-577-valid.json).
  [Narrative](full_log.md#cycle-577--572-stack-plus-gdiis-start-15-on-n18-allenes).
- Cycle 576; candidate `f2cb453`; champion `5b183d43d9c42b918e209ab6e11471afe1f07ba5`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-576-train.json).
  [Validation evidence](evaluation_results/cycle-576-valid.json).
  [Narrative](full_log.md#cycle-576--572-stack-plus-skip-gdiis-rho-on-n18-allenes).
- Cycle 573; candidate `88dafe8a5f27fefd8919ab4ad3b0ece3a4af27b3`; champion `5b183d43d9c42b918e209ab6e11471afe1f07ba5`; decision non_generalizable.
  [Training evidence](evaluation_results/cycle-573-train.json).
  [Validation evidence](evaluation_results/cycle-573-valid.json).
  [Narrative](full_log.md#cycle-573--572-stack-plus-limiter-wd-07-on-n18-allene-dummy-dihedrals).
- Cycle 572; candidate `93c51156cc3542e50c7c70f124db8e6238ff7adc`; champion `5b183d43d9c42b918e209ab6e11471afe1f07ba5`; decision non_generalizable.
  [Implementation](ideas/sella-n12-wa-070/93c51156cc3542e50c7c70f124db8e6238ff7adc/algo.py).
  [Training evidence](evaluation_results/cycle-572-train.json).
  [Validation evidence](evaluation_results/cycle-572-valid.json).
  [Narrative](full_log.md#cycle-572--wa070-on-n12-without-a-pf-bond).
- Cycle 571; candidate `a10f906f7eb4ae6bea7f2e9c9120d27c9eebfb23`; champion `5b183d43d9c42b918e209ab6e11471afe1f07ba5`; decision non_generalizable.
  [Implementation](ideas/sella-n12-wa-070/a10f906f7eb4ae6bea7f2e9c9120d27c9eebfb23/algo.py).
  [Training evidence](evaluation_results/cycle-571-train.json).
  [Validation evidence](evaluation_results/cycle-571-valid.json).
  [Narrative](full_log.md#cycle-571--wa070-on-connected-n12).















































