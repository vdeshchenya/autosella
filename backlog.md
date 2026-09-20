# Research backlog

## damped-bfgs: Positive-curvature internal Hessian updates
Status: closed

**Hypothesis:** Damped BFGS can produce a more useful positive curvature model for minimum searches.
**Outcome and uncertainty:** Cycle 2 failed both energy and cost requirements despite complete convergence; broad slowdown, no retained failure bundles. No evidence supports a damping-only repair.
**Reason to revisit:** None currently. The proposed independent RFO control (cycle 3) also failed, so the condition for the original combination hypothesis was not met. A different, evidence-supported metric/update coupling would be needed to reopen this entry.
**Next experiment:** No scheduled repair. Preserve the implementation and failed evidence; do not combine rejected variants without a new mechanism.

**Attempts:**
- Cycle 2; candidate `de54017baf76882c62f9a69bef2a05574489f986`; champion `71a74e891073ff65f11406a38a0abdff764815d0`; decision invalid.
  [Implementation](ideas/damped-bfgs/de54017baf76882c62f9a69bef2a05574489f986/algo.py).
  [Training evidence](evaluation_results/cycle-2-train.json).
  [Narrative](full_log.md#cycle-2-damped-bfgs-internal-hessian).

## rational-function-step: Augmented-Hessian RFO steps
Status: closed

**Hypothesis:** RFO regularizes soft curvature more effectively than absolute-eigenvalue Newton steps.
**Outcome and uncertainty:** Cycle 3 failed energy and cost despite all molecules converging without errors. Gains on 89 molecules do not establish aggregate usefulness. Converged paths have no passive traces.
**Reason to revisit:** RFO can follow negative curvature and may require stronger trust-radius contraction than the original policy; this interaction could explain basin changes.
**Next experiment:** Cycle 18 tested the proposed contraction repair with the current accepted model and failed both cost and energy. No further tuning scheduled; a new step-model/globalization mechanism would be needed to reopen.

**Attempts:**
- Cycle 3; candidate `771d38ae3c9dc008b93f7f68ec7dc5ba393d1c66`; champion `71a74e891073ff65f11406a38a0abdff764815d0`; decision invalid.
  [Implementation](ideas/rational-function-step/771d38ae3c9dc008b93f7f68ec7dc5ba393d1c66/algo.py).
  [Training evidence](evaluation_results/cycle-3-train.json).
  [Narrative](full_log.md#cycle-3-rational-function-steps).

- Cycle 18; candidate `aad65c90f91a996bbf0a474d0ac45970ed092294`; champion `103f2fdfee3ef9f388cd082b4a7c4a2eefb59b6d`; decision invalid. Repair 1 with stronger contraction.
  [Implementation](ideas/rational-function-step/aad65c90f91a996bbf0a474d0ac45970ed092294/algo.py).
  [Training evidence](evaluation_results/cycle-18-train.json).
  [Narrative](full_log.md#cycle-18-rfo-with-stronger-trust-model-contraction).

## fragment-tric: Explicit intermolecular translation and rotation
Status: incorporated

**Hypothesis:** Replace pseudo-bond fragment coupling with centroid/quaternion coordinates to improve intermolecular conditioning.
**Outcome and uncertainty:** Cycles 4–5 invalid from one SCF failure and energy regressions. Cycle 6 with softer curvature passed train/validation and became champion. Passive step-size repair did not resolve the error; softer collective curvature did in the tested evaluations, though the detailed electronic-path mechanism remains uncertain.
**Reason to revisit:** Cost signal warrants bounded geometric step control to stabilize newly exposed collective modes.
**Next experiment:** Softer TRIC curvature passed both gates in cycle 6 and is incorporated. The cap remains rejected. Next assess trust-radius growth/initialization with this accepted coordinate model under the same gates.

**Attempts:**
- Cycle 4; candidate `5db45c8a14616f06cf1fd56be16ac8115262d590`; champion `71a74e891073ff65f11406a38a0abdff764815d0`; decision invalid.
  [Implementation](ideas/fragment-tric/5db45c8a14616f06cf1fd56be16ac8115262d590/algo.py).
  [Training evidence](evaluation_results/cycle-4-train.json).
  [Narrative](full_log.md#cycle-4-explicit-fragment-translation-and-rotation).

- Cycle 5; candidate `ba4ffa67bc4867c0d847da7a177c2b49fb470410`; champion `71a74e891073ff65f11406a38a0abdff764815d0`; decision invalid. Repair 1; cap did not cure SCF failure.
  [Implementation](ideas/fragment-tric/ba4ffa67bc4867c0d847da7a177c2b49fb470410/algo.py).
  [Training evidence](evaluation_results/cycle-5-train.json).
  [Narrative](full_log.md#cycle-5-cartesian-motion-cap-for-fragment-coordinates).

- Cycle 6; candidate `2077e365ba0dd5cc51bb719b3ed82b128e5da840`; champion `71a74e891073ff65f11406a38a0abdff764815d0`; decision keep. Repair 2 with softer curvature resolves the tested numerical failure and passes both aggregate energy/cost gates.
  Implementation retained as accepted Git commit.
  [Training evidence](evaluation_results/cycle-6-train.json).
  [Validation evidence](evaluation_results/cycle-6-valid.json).
  [Narrative](full_log.md#cycle-6-softer-fragment-curvature).

## coordinate-aware-radius: Initial radius respects coordinate types
Status: incorporated

**Hypothesis:** Larger initial radii accelerate bonded-coordinate correction but collective translation/rotation modes need smaller initial steps.
**Outcome and uncertainty:** Uniform radius 0.2 in cycle 7 passes energy, loses aggregate cost. Saved comparisons show bonded structures improve while complexes slow down; individual trajectories are not retained for converged runs.
**Reason to revisit:** Coordinate types define different physical displacements per internal unit, so a generic coordinate-aware radius could retain early gains without destabilizing collective motion.
**Next experiment:** Repair 1 passed both gates as cycle 8. Retain the coordinate-aware radius in the champion; future parameter changes require a fresh hypothesis and full evaluation. No molecular identities or dataset categories enter the implementation.

**Attempts:**
- Cycle 7; candidate `4ee07740c283194fafaa547b093dccb5a8229337`; champion `2077e365ba0dd5cc51bb719b3ed82b128e5da840`; decision discard.
  [Implementation](ideas/coordinate-aware-radius/4ee07740c283194fafaa547b093dccb5a8229337/algo.py).
  [Training evidence](evaluation_results/cycle-7-train.json).
  [Narrative](full_log.md#cycle-7-larger-initial-trust-radius).

- Cycle 8; candidate `46ea0b05db756c6f9842a66e5ce3cf27a848c764`; champion `2077e365ba0dd5cc51bb719b3ed82b128e5da840`; decision keep. Repair 1 accepted.
  Implementation retained as accepted Git commit.
  [Training evidence](evaluation_results/cycle-8-train.json).
  [Validation evidence](evaluation_results/cycle-8-valid.json).
  [Narrative](full_log.md#cycle-8-coordinate-aware-initial-radius).

**360 mixed-chart control:**Apply8's existing2:1 intrinsic/collective scale ratio within mixed charts, consistently in ordinary/GP bounds and feedback. This tests unnecessary intrinsic restriction when collective rows exist; not a growth-factor or radius-ratio sweep. Pure intrinsic behavior stays accepted8.

**360 outcome:**Invalid aggregate energy and slightly worse cost with all469 converged, zero errors and empty failure_details. The fixed mixed-chart ratio fails; no ratio/subtype/controller sweep. Accepted8 remains incorporated. Revisit mixed bounds only with independent evidence for coupled intrinsic/collective prediction error.
- Cycle360; candidate `30cafa58f9c928a3f63d2d398407b1e854e6ef05`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision invalid. [Implementation](ideas/coordinate-aware-radius/30cafa58f9c928a3f63d2d398407b1e854e6ef05/algo.py), [train](evaluation_results/cycle-360-train.json), [narrative](full_log.md#cycle-360-two-family-intrinsic-and-collective-step-bounds).

## cartesian-model: Molecular-model Cartesian quasi-Newton optimization
Status: deferred

**Hypothesis:** Map physical internal curvature into Cartesian space and learn Cartesian secants to avoid nonlinear transport.
**Outcome and uncertainty:** Cycle 9 valid energy but severe broad cost failure, five call-limit cases. Passive traces show sustained slow descent, few uphill trials; rejection overhead is insufficient to explain slowdown.
**Reason to revisit:** Updating the geometric preconditioner at each geometry and using limited-memory corrections may retain moving bond/angle stiffness that the fixed initial metric loses. This is a different method than the tested full-BFGS initialization.
**Next experiment:** The dynamic-preconditioner repair was tested in cycle 13 and worsened cost further. No current repair is justified. Reopen only with evidence for a different metric or curvature model, not another step-cap change.

**Attempts:**
- Cycle 9; candidate `78d9ff9546cb1c0fba52bf18a992bd333dde5fe4`; champion `46ea0b05db756c6f9842a66e5ce3cf27a848c764`; decision discard.
  [Implementation](ideas/cartesian-model/78d9ff9546cb1c0fba52bf18a992bd333dde5fe4/algo.py).
  [Training evidence](evaluation_results/cycle-9-train.json).
  [Narrative](full_log.md#cycle-9-model-preconditioned-cartesian-optimizer).

- Cycle 13; candidate `925a1daa8df2a6d0d3321d479633b9b92bf8075a`; champion `46ea0b05db756c6f9842a66e5ce3cf27a848c764`; decision discard. Repair 1, moving model with L-BFGS, worsens cost.
  [Implementation](ideas/cartesian-model/925a1daa8df2a6d0d3321d479633b9b92bf8075a/algo.py).
  [Training evidence](evaluation_results/cycle-13-train.json).
  [Narrative](full_log.md#cycle-13-moving-cartesian-molecular-preconditioner).

**381 revisit:**Accepted physical priors and calibration since9/13 supply independent model evidence. Transfer current blocks through a common dummy lift into physical-whitened Cartesian TS/SR1 learning, retaining9's trust controller. Tests a distinct calibrated curvature model; no radius sweep.

**381 reassessment:**The accepted physical-prior/calibration prerequisite was tested in fixed physical-whitened Cartesian TS/SR1 learning. Broad cost loss remains:456 slower, six limits, valid energy and zero errors. Two passive trajectories show ongoing late relaxation rather than a localized exception. A new independently supported moving-Cartesian curvature or consistent nonlinear model is needed; no trust/floor/fit-count sweep.
- Cycle381; candidate `f79d535b1a9d4d5e1d15ff919df830f368955ef5`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard.
  [Implementation](ideas/cartesian-model/f79d535b1a9d4d5e1d15ff919df830f368955ef5/algo.py), [train](evaluation_results/cycle-381-train.json), [narrative](full_log.md#cycle-381-calibrated-physical-metric-cartesian-ts-learning).
  [Passive manifest](diagnostics/82ff4cbfed94496993675fb965f68a15/manifest.json).

## model-scale-calibration: First-secant scaling of a physical Hessian
Status: incorporated

**Hypothesis:** Calibrate the initial chemical model's global stiffness from observed curvature at no added force cost.
**Outcome and uncertainty:** Cycle 10 passed training by a small margin but lost validation cost; both splits energy-valid/error-free. A single direction may not represent other modes.
**Reason to revisit:** A robust fit over several independent directions or separate physical mode classes could reduce first-direction bias while retaining model anisotropy.
**Next experiment:** Physical-block fitting over six secants passed both gates in cycle 15 and is incorporated. Keep the failed scalar variant as evidence; future changes should compare to the accepted block fit rather than the old anchor.

**Attempts:**
- Cycle 10; candidate `6742314709fc7d31ea6585b9ceb2a28de63e4fca`; champion `46ea0b05db756c6f9842a66e5ce3cf27a848c764`; decision non_generalizable.
  [Implementation](ideas/model-scale-calibration/6742314709fc7d31ea6585b9ceb2a28de63e4fca/algo.py).
  [Training evidence](evaluation_results/cycle-10-train.json).
  [Validation evidence](evaluation_results/cycle-10-valid.json).
  [Narrative](full_log.md#cycle-10-first-secant-model-scale-calibration).

- Cycle 15; candidate `103f2fdfee3ef9f388cd082b4a7c4a2eefb59b6d`; champion `46ea0b05db756c6f9842a66e5ce3cf27a848c764`; decision keep. Repair 1 uses regularized physical blocks and several early secants.
  Implementation retained as accepted Git commit.
  [Training evidence](evaluation_results/cycle-15-train.json).
  [Validation evidence](evaluation_results/cycle-15-valid.json).
  [Narrative](full_log.md#cycle-15-regularized-physical-block-hessian-calibration).

## energy-secants: Use energy changes in directional Hessian learning
Status: deferred

**Hypothesis:** Cubic-Hermite endpoint curvature from energies and gradients improves ordinary average-curvature secants without extra force calls.
**Outcome and uncertainty:** Cycle 11 energy-valid/error-free but cost failure. Gains on 113 cases do not compensate slower convergence on 174; no retained failure traces. Curved-coordinate metric compatibility remains uncertain.
**Reason to revisit:** A covariant derivation or application to a straight Cartesian metric would separate energy-information utility from geodesic transport assumptions. Could combine with a future successful Cartesian optimizer.
**Next experiment:** Derive a metric-consistent correction before another internal-coordinate attempt, or test it with improved dynamic Cartesian preconditioning; retain unchanged gates.

**Attempts:**
- Cycle 11; candidate `bd0cf6def371fb4404b735560d0cd7ff7e7c9328`; champion `46ea0b05db756c6f9842a66e5ce3cf27a848c764`; decision discard.
  [Implementation](ideas/energy-secants/bd0cf6def371fb4404b735560d0cd7ff7e7c9328/algo.py).
  [Training evidence](evaluation_results/cycle-11-train.json).
  [Narrative](full_log.md#cycle-11-energy-informed-geodesic-secants).





**Completed revisit:** Cycle 63 derives actual-path endpoint acceleration and pairing corrections, preserves native early fitting, and tests against champion 41. These terms were absent in cycle 11.

**Reassessment after cycle 63:** Actual-path acceleration/pairing correction is energy-valid and error-free but loses cost (72 faster, 103 slower, 294 unchanged); all cases converge and no failure traces exist. Defer further internal Hermite secants until an independently improved path/model connection or energy interpolant is available. No clipping or timing sweep is justified.
- Cycle 63; candidate `a3480744809b118ecc9acbe8073cc73ad14ed96a`; champion `f56535e29a1c699bb1a1ea59ae6f57f047fda88e`; decision discard.
  [Implementation](ideas/energy-secants/a3480744809b118ecc9acbe8073cc73ad14ed96a/algo.py).
  [Training evidence](evaluation_results/cycle-63-train.json).
  [Narrative](full_log.md#cycle-63-path-consistent-energy-informed-curvature).

**Revisit199:**Independently accepted182/186 changes the bond metric and connection; apply63's unchanged post-calibration Hermite correction with endpoint acceleration from that actual current connection. Reset path state on each collision retry; clipping/eligibility constants remain fixed. Predeclared during198.

**Reassessment199:**Current-connection acceleration in the accepted Badger chart remains fully converged and energy-valid but loses cost. The declared improved-path prerequisite does not rescue this interpolation. Require a different independently supported interpolant/error model before another revisit; no clipping/timing sweep.
- Cycle199; candidate `1f39a0633364bcd0a1056b640b6d1316c6b57709`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision discard.
  [Implementation](ideas/energy-secants/1f39a0633364bcd0a1056b640b6d1316c6b57709/algo.py), [train](evaluation_results/cycle-199-train.json), [narrative](full_log.md#cycle-199-energy-secants-with-the-current-badger-connection).

**Revisit553:** A distinct use of the energy interpolant in the exact fixed GP objective eliminates physical path acceleration and vector transport entirely. This tests the earlier flat-chart prerequisite at surrogate level, rather than reapplying199 to the physical Hessian. BFGS uses endpoint-curvature secants with unchanged Wolfe controls, twelve steps and original proposal critics. This is structural motivation, not a diagnosed physical-path repair.

**553 reassessment:** Exact fixed GP chart still loses cost;2faster7slower460same,allconvergedvalidzeroerrors. No passive microstate diagnoses interpolation failure. Future use needs independent endpoint-curvature accuracy evidence; no strength,safeguard,budget orlinesearch sweep.
- Cycle553; candidate `58b8ff78b0042f41988e142140f38dea52f24247`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/energy-secants/58b8ff78b0042f41988e142140f38dea52f24247/algo.py), [training](evaluation_results/cycle-553-train.json), [narrative](full_log.md#cycle-553-energy-informed-surrogate-bfgs-secants).

## local-gdiis: Safeguarded geometry-history extrapolation
Status: deferred

**Hypothesis:** Recent geometries and preconditioned gradients improve local steps beyond one secant at no added force cost.
**Outcome and uncertainty:** Cycle 16 passes training and reduces validation cost but loses validation energy. One training call-limit bundle shows lengthy nonlinear descent; converged validation paths are unavailable.
**Reason to revisit:** Speed gains on both splits justify delaying extrapolation until the trust model indicates local quadratic behavior.
**Next experiment:** Repair 1 failed training cost and retained the same call-limit case. Do not adjust more activation thresholds. Revisit only with a metric-consistent way to transport history across changing internal-coordinate tangent spaces, then repeat full gates.

**Attempts:**
- Cycle 16; candidate `ac78a1e098f2e430c8536ae6797d1d52a9d9f79e`; champion `103f2fdfee3ef9f388cd082b4a7c4a2eefb59b6d`; decision non_generalizable.
  [Implementation](ideas/local-gdiis/ac78a1e098f2e430c8536ae6797d1d52a9d9f79e/algo.py).
  [Training evidence](evaluation_results/cycle-16-train.json).
  [Validation evidence](evaluation_results/cycle-16-valid.json).
  [Narrative](full_log.md#cycle-16-safeguarded-geometry-diis).

- Cycle 17; candidate `7d2a6e5070b7bf92745bad3088a033d53ee90368`; champion `103f2fdfee3ef9f388cd082b4a7c4a2eefb59b6d`; decision discard. Repair 1 loses training speed; energy-valid.
  [Implementation](ideas/local-gdiis/7d2a6e5070b7bf92745bad3088a033d53ee90368/algo.py).
  [Training evidence](evaluation_results/cycle-17-train.json).
  [Narrative](full_log.md#cycle-17-activate-diis-only-with-a-reliable-local-model).

**Cycle215 outcome:**Developed path displacements and parallel gradients with186 retain full convergence/energy but lose training cost. No isolated failure supports further activation/ridge changes. Revisit only with a separately supported covariant preconditioner or diagnosed extrapolation failure; transported history is reusable, but its benefit is unestablished.
- Cycle215; candidate `2d13b2ba8e1163347ea22930b05837331f43767b`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision discard.
  [Implementation](ideas/local-gdiis/2d13b2ba8e1163347ea22930b05837331f43767b/algo.py), [train](evaluation_results/cycle-215-train.json), [narrative](full_log.md#cycle-215-developed-history-gdiis).

## subspace-gp: Gradient-enhanced nonlinear surrogate in recent motion space
Status: incorporated

**Hypothesis:** A small energy-and-gradient GP with the current quadratic model as prior can improve steps using nonlinear history, without extra force calls or full-dimensional GP scaling.
**Outcome and uncertainty:** Cycle 19 fails before activation from a missing Hessian accessor. The surrogate mechanism remains unmeasured.
**Reason to revisit:** A precise API correction allows the proposed method to be evaluated; no optimizer or contract redesign is needed.
**Next experiment:** Repair 1 passed both splits as cycle 20. The GP is incorporated. Next compare residual-based uncertainty calibration against this accepted implementation, with the mean model unchanged.

**Attempts:**
- Cycle 19; candidate `a41daef1b418e964afce5729263cf10c4d5fdd98`; champion `103f2fdfee3ef9f388cd082b4a7c4a2eefb59b6d`; decision invalid.
  [Implementation](ideas/subspace-gp/a41daef1b418e964afce5729263cf10c4d5fdd98/algo.py).
  [Training evidence](evaluation_results/cycle-19-train.json).
  [Narrative](full_log.md#cycle-19-low-dimensional-gradient-enhanced-surrogate-steps).

- Cycle 20; candidate `c94c425f8c57f8fddbe81012e20febe7625547a5`; champion `103f2fdfee3ef9f388cd082b4a7c4a2eefb59b6d`; decision keep. Repair 1 fixes only the accessor and establishes both force-call gains and energy validity.
  Implementation retained in accepted Git history.
  [Training evidence](evaluation_results/cycle-20-train.json).
  [Validation evidence](evaluation_results/cycle-20-valid.json).
  [Narrative](full_log.md#cycle-20-repair-the-surrogate-hessian-interface).

## gp-uncertainty: Residual-calibrated surrogate confidence
Status: deferred

**Hypothesis:** Empirical residual amplitude can distinguish unreliable quadratic-prior histories from accurate local models and improve GP proposal selection.
**Outcome and uncertainty:** Cycle 21 gains training cost but loses validation cost; both splits energy-valid, fully converged and error-free. No surrogate-decision traces exist. Large training improvements coexist with more small regressions.
**Reason to revisit:** The residual statistic depends on kernel fit and retained-history geometry. A separately accepted change to the covariance model could alter this interaction; current evidence does not justify tuning the confidence threshold.
**Next experiment:** First test the independent Matérn-kernel hypothesis against the champion. Reassess residual calibration only if a new accepted kernel provides a concrete reason, with all gates repeated.

**Attempts:**
- Cycle 21; candidate `26f93c15025242f7e2c8ff9501541b26cf5be9a2`; champion `c94c425f8c57f8fddbe81012e20febe7625547a5`; decision non_generalizable.
  [Implementation](ideas/gp-uncertainty/26f93c15025242f7e2c8ff9501541b26cf5be9a2/algo.py).
  [Training evidence](evaluation_results/cycle-21-train.json).
  [Validation evidence](evaluation_results/cycle-21-valid.json).
  [Narrative](full_log.md#cycle-21-calibrate-gp-uncertainty-from-observed-residuals).


**Revisit cycle 143:** Cycle 106 independently accepted tangent-direction derivative covariances, changing the model on which the residual statistic depends. Test the exact cycle-21 amplitude formula on that champion without changing the confidence threshold or floor.


**Cycle 143 reassessment:** Exact cycle-21 amplitude calibration with accepted tangent derivative maps passes energy and fully converges without errors, but loses training cost. The proposed covariance interaction does not help. No confidence or amplitude sweep; require independently diagnosed predictive-error behavior before another attempt.
- Cycle 143; candidate `a9b6d3bb63bf81eea1547df1c64a0661f3e87a2e`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision discard.
  [Implementation](ideas/gp-uncertainty/a9b6d3bb63bf81eea1547df1c64a0661f3e87a2e/algo.py).
  [Training evidence](evaluation_results/cycle-143-train.json).
  [Narrative](full_log.md#cycle-143-residual-confidence-calibration-with-tangent-observations).

**Revisit555:** An independent Student-t process model marginalizes one shared inverse-Gamma scale of both latent residual and observation nugget. This is a structural Bayesian alternative to21/143's plug-in scale beta/n, not a threshold/floor adjustment; it also differs from420's separate heavy-tailed block likelihood. Fixednu5 gives matched unit prior covariance and finite fourth moments. No claim that prior failures diagnosed scale uncertainty.

**555 reassessment:** Shared-scale Studenttprocess also loses cost;32faster61slower376same,allconvergedvalidzeroerrors. This tests analytic marginalization rather than the former plug-in estimator. Require actual predictive-error calibration evidence before further change; no degrees/noise/strength/threshold sweep.
- Cycle555; candidate `540cb97d66a07b910618387544550c4e7bc2b027`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/gp-uncertainty/540cb97d66a07b910618387544550c4e7bc2b027/algo.py), [training](evaluation_results/cycle-555-train.json), [narrative](full_log.md#cycle-555-analytically-marginalized-student-t-process-scale).

## matern-surrogate: Less restrictive smoothness for GP residuals
Status: deferred

**Hypothesis:** Matérn 5/2 interpolation reduces nearby extrapolation oscillations compared with the squared-exponential kernel while retaining a physical quadratic prior.
**Outcome and uncertainty:** Cycle 22 gains training cost but loses validation cost. Energy-valid and fully converged on both splits; no failure traces. Local gradient covariance was matched, but global kernel range and projected-history consistency differ.
**Reason to revisit:** Training gains in complexes suggest useful behavior, but extrapolation paths are unobserved. A separately validated coordinate mapping or history representation could change the kernel interaction.
**Next experiment:** No kernel-length sweep. First test consistency between surrogate targets and coordinate conversion on the accepted kernel. Reassess Matérn only after an independent improvement establishes a concrete combination hypothesis; repeat every gate.

**Attempts:**
- Cycle 22; candidate `da1ff4766b6bc2974934d5c284ff1ddf3b0a6a47`; champion `c94c425f8c57f8fddbe81012e20febe7625547a5`; decision non_generalizable.
  [Implementation](ideas/matern-surrogate/da1ff4766b6bc2974934d5c284ff1ddf3b0a6a47/algo.py).
  [Training evidence](evaluation_results/cycle-22-train.json).
  [Validation evidence](evaluation_results/cycle-22-valid.json).
  [Narrative](full_log.md#cycle-22-matern-covariance-for-the-compact-surrogate).

Cycle 79 tests the same matched-gradient-variance kernel after cycle 77 independently accepts retained uphill observations. This changes the sampled history across reversals while leaving kernel scale fixed; the earlier validation failure is retained.

**Reassessment after cycle 79:** The accepted nonmonotone history permits a new cost gain but the combined Matérn variant fails aggregate training energy. All runs converge; dominant endpoint changes have no path traces. Stop this combination without length or confidence tuning. Revisit only after evidence for a specific chart-consistent covariance construction or diagnosed extrapolation mechanism, not another unqualified model combination.

- Cycle 79; candidate `5ef66d5388a46eb25aa73a0afaad853591871bc7`; champion `07a8c991a656290029ea2aee428d3b4f66890938`; decision invalid.
  [Implementation](ideas/matern-surrogate/5ef66d5388a46eb25aa73a0afaad853591871bc7/algo.py).
  [Training evidence](evaluation_results/cycle-79-train.json).
  [Narrative](full_log.md#cycle-79-matern-covariance-with-retained-uphill-history).

**Reassessment after cycle 121:** Combining the same kernel scale with accepted tangent derivative maps passes training energy but loses cost. This resolves that specific pending interaction test negatively. No new length or confidence tuning is justified. A future revisit needs independently observed kernel extrapolation failure and a derived repair.
- Cycle 121; candidate `189ffd0f0f664ff48aedf846f48198a0d064be42`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision discard.
  [Implementation](ideas/matern-surrogate/189ffd0f0f664ff48aedf846f48198a0d064be42/algo.py).
  [Training evidence](evaluation_results/cycle-121-train.json).
  [Narrative](full_log.md#cycle-121-matern-covariance-with-tangent-aware-derivative-observations).

## surrogate-target-conversion: Realize GP internal-coordinate targets
Status: deferred

**Hypothesis:** Iterative internal-to-Cartesian conversion can better realize the surrogate minimizer than interpreting its displacement as a geodesic tangent.
**Outcome and uncertainty:** Cycle 23 improves training cost but loses validation cost; both energy-valid, fully converged and error-free. No retained trajectories isolate the cause. Redundant-target feasibility and approximate gradient transport remain unresolved.
**Reason to revisit:** A metric-consistent derivation of history/gradient transport for the target chart could resolve the mismatch; simply tightening conversion tolerances has no supporting evidence.
**Next experiment:** No immediate repair. Revisit with a defined secant/transport convention or a separately validated fixed-chart surrogate, then repeat full gates.

**Attempts:**
- Cycle 23; candidate `058d5d6234243b38593b40b86b207dd25c864da4`; champion `c94c425f8c57f8fddbe81012e20febe7625547a5`; decision non_generalizable.
  [Implementation](ideas/surrogate-target-conversion/058d5d6234243b38593b40b86b207dd25c864da4/algo.py).
  [Training evidence](evaluation_results/cycle-23-train.json).
  [Validation evidence](evaluation_results/cycle-23-valid.json).
  [Narrative](full_log.md#cycle-23-iterative-realization-of-surrogate-coordinate-targets).

## distance-product-model: Lindh-style initial internal curvature
Status: closed

**Hypothesis:** Distance-product stiffness for angle/torsion coordinates can represent coupled bond softening more accurately before secant fitting.
**Outcome and uncertainty:** Cycle 24 is energy-valid but slower in training; 157 improve, 218 regress. All converge with no errors or retained traces. It substituted covalent-radius sums for the reference distances paired with published force constants.
**Reason to revisit:** Restoring that parameter pairing tests whether the adaptation, rather than the distance-product construction, causes the loss. This is a mechanistic source comparison; endpoint evidence alone does not prove it.
**Next experiment:** Repair 1 (cycle 25) worsened cost further despite energy validity. No further force-constant tuning is justified. Reopen only with evidence for a different interaction with curvature learning or coordinates.

**Attempts:**
- Cycle 24; candidate `17ffe8074faada78ba31233300bec89024415bf2`; champion `c94c425f8c57f8fddbe81012e20febe7625547a5`; decision discard.
  [Implementation](ideas/distance-product-model/17ffe8074faada78ba31233300bec89024415bf2/algo.py).
  [Training evidence](evaluation_results/cycle-24-train.json).
  [Narrative](full_log.md#cycle-24-distance-product-initial-curvature-model).

- Cycle 25; candidate `7979620aa813a324fd3a936027183e72012a3ba5`; champion `c94c425f8c57f8fddbe81012e20febe7625547a5`; decision discard. Repair 1 restores published parameter pairing and worsens cost.
  [Implementation](ideas/distance-product-model/7979620aa813a324fd3a936027183e72012a3ba5/algo.py).
  [Training evidence](evaluation_results/cycle-25-train.json).
  [Narrative](full_log.md#cycle-25-restore-the-published-distance-model-parameter-pairing).

## surrogate-trust-feedback: Use the selecting model for radius feedback
Status: deferred

**Hypothesis:** A GP-selected step should use GP-predicted energy change when judging model agreement for the trust radius.
**Outcome and uncertainty:** Cycle 26 barely improves training and slightly worsens validation cost. Both splits energy-valid, fully converged and error-free. Most trajectories have unchanged call counts; no GP-decision traces exist.
**Reason to revisit:** A separately accepted change in surrogate representation or target realization could improve prediction accuracy and change this feedback interaction.
**Next experiment:** Cycle 119 now tests the accepted tangent-aware observation model and fails training cost. No threshold tuning. Revisit only after a demonstrated improvement in predicting the realized geodesic endpoint, distinct from historical derivative consistency.

**Attempts:**
- Cycle 26; candidate `ecdfe4038e1d5ac0dd285937f0fbbb8191be1ab9`; champion `c94c425f8c57f8fddbe81012e20febe7625547a5`; decision non_generalizable.
  [Implementation](ideas/surrogate-trust-feedback/ecdfe4038e1d5ac0dd285937f0fbbb8191be1ab9/algo.py).
  [Training evidence](evaluation_results/cycle-26-train.json).
  [Validation evidence](evaluation_results/cycle-26-valid.json).
  [Narrative](full_log.md#cycle-26-trust-feedback-from-the-selecting-surrogate-model).

- Cycle 119; candidate `3eda736503077bb55c3e8805111daa4980de0974`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision discard. Fully converged and energy-valid, with higher training cost.
  [Implementation](ideas/surrogate-trust-feedback/3eda736503077bb55c3e8805111daa4980de0974/algo.py).
  [Training evidence](evaluation_results/cycle-119-train.json).
  [Narrative](full_log.md#cycle-119-gp-selected-trust-feedback-with-tangent-observations).

## inverse-distance-surrogate: Pair-geometry GP with physical motion prior
Status: deferred

**Hypothesis:** An inverse-pair-distance kernel can provide invariant, contact-sensitive interpolation while retaining learned physical curvature as its mean model.
**Outcome and uncertainty:** Cycle 27 is fully converged/error-free but fails energy and cost. Fifty-six cases improve and 227 slow down. No failed-path bundles exist. Cartesian prior curvature and geodesic realization are approximate.
**Reason to revisit:** A consistent Cartesian Hessian transformation, including the appropriate geometric curvature terms, or an intrinsic feature Jacobian compatible with the internal step chart could remove a specific model/mapping mismatch. The tested hybrid does not disprove inverse-distance kernels generally.
**Next experiment:** Derive and statically validate that metric/curvature connection before another attempt. No length-scale sweep or immediate numerical repair is justified by the endpoint evidence alone.

**Attempts:**
- Cycle 27; candidate `50acf635218ad0e21904c4ad697feec2ce816f54`; champion `c94c425f8c57f8fddbe81012e20febe7625547a5`; decision invalid.
  [Implementation](ideas/inverse-distance-surrogate/50acf635218ad0e21904c4ad697feec2ce816f54/algo.py).
  [Training evidence](evaluation_results/cycle-27-train.json).
  [Narrative](full_log.md#cycle-27-inverse-distance-surrogate-in-a-physical-motion-subspace).

**Cycle 123 reassessment:** The derived Cartesian chain-rule curvature term is now tested with the preserved inverse-distance model on champion 106. It again fails energy and cost with full convergence and no errors/artifacts. That prior correction alone is insufficient. Any future revisit needs a consistent query-to-realized geometry formulation, not scale tuning.
- Cycle 123; candidate `af7c267f3dd878fa1598d1fc850d5163ba2302b7`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision invalid.
  [Implementation](ideas/inverse-distance-surrogate/af7c267f3dd878fa1598d1fc850d5163ba2302b7/algo.py).
  [Training evidence](evaluation_results/cycle-123-train.json).
  [Narrative](full_log.md#cycle-123-cartesian-chain-rule-curvature-for-inverse-distance-gp).

**397 reassessment:** Exact scored Cartesian endpoints, developed native secants and transported gradients fulfill123's query/realization prerequisite. All469converge with validenergy andzeroerrors, but31faster/263slower/175unchanged lose cost broadly. Emptyfailure_details cannot distinguish descriptor/mean accuracy from loss of geodesic motion. Future revisit needs independent finite-response model evidence; no length,confidence,history,solverbudget or eligibility sweep.
- Cycle397; candidate `593bdfd441a37de54ce131d85e2a6777771ff140`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/inverse-distance-surrogate/593bdfd441a37de54ce131d85e2a6777771ff140/algo.py), [train](evaluation_results/cycle-397-train.json), [narrative](full_log.md#cycle-397-exact-cartesian-realization-of-inverse-distance-gp-proposals).

## diverse-gp-history: Select distinct nearby observations
Status: deferred

**Hypothesis:** A diverse five-point subset of a nearby twelve-point pool improves information content without enlarging the surrogate solve.
**Outcome and uncertainty:** Cycle 28 passes energy but loses training cost, with full convergence and no errors or failure traces. Older history can increase chart error; endpoints do not isolate it.
**Reason to revisit:** A separately accepted history transport or coordinate-chart correction could make older points more trustworthy.
**Next experiment:** No pool-size tuning. Revisit selection only after a concrete history-consistency improvement, keeping the newest two points and repeating full gates.

**Attempts:**
- Cycle 28; candidate `e1575a6bc8cbf3d180b64cc27ad25738ac12911d`; champion `c94c425f8c57f8fddbe81012e20febe7625547a5`; decision discard.
  [Implementation](ideas/diverse-gp-history/e1575a6bc8cbf3d180b64cc27ad25738ac12911d/algo.py).
  [Training evidence](evaluation_results/cycle-28-train.json).
  [Narrative](full_log.md#cycle-28-diverse-nearby-history-for-the-accepted-surrogate).

**Cycle 111 reassessment:** Combining the same selection rule with independently accepted tangent derivative maps again loses training cost despite valid energy and full convergence. No passive trace identifies a repair; a more specific local information criterion is needed before revisiting, not pool-size tuning.

- Cycle 111; candidate `6720d42b69095b78f9f89e12d76c81a1f9ea2aab`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision discard.
  [Implementation](ideas/diverse-gp-history/6720d42b69095b78f9f89e12d76c81a1f9ea2aab/algo.py).
  [Training evidence](evaluation_results/cycle-111-train.json).
  [Narrative](full_log.md#cycle-111-diverse-nearby-history-with-tangent-observations).

**Cycle 130 reassessment:** Query-energy posterior variance selection also loses cost with full convergence and valid energy. The scoring span and final span differ, but no retained path evidence diagnoses this as the cause. Defer selection until a model-consistent criterion or passive selection trace supports a specific change; no pool-size or query-weight sweep.

- Cycle 130; candidate `89736f36e5f8e5be0ab63869494db28a71b4e87c`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision discard.
  [Implementation](ideas/diverse-gp-history/89736f36e5f8e5be0ab63869494db28a71b4e87c/algo.py).
  [Training evidence](evaluation_results/cycle-130-train.json).
  [Narrative](full_log.md#cycle-130-predictive-variance-selection-of-tangent-gp-history).


**254 fixed-model reassessment:**Selection and prediction now use the same native recent-five span, covariance and target submatrix. This supplies130's declared model-consistency prerequisite, but all469 converged/energy-valid runs still lose cost. No localized numerical defect is observed. Defer until independently demonstrated older-data reliability, without pool/query/length/confidence sweeps.
- Cycle254; candidate `48d0b203800d452ed559779cfefee3b2e7ec407e`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision discard.
  [Implementation](ideas/diverse-gp-history/48d0b203800d452ed559779cfefee3b2e7ec407e/algo.py), [train](evaluation_results/cycle-254-train.json), [narrative](full_log.md#cycle-254-select-gp-history-in-the-same-model-used-for-prediction).

## positive-gp-prior: Project full positive curvature into the surrogate
Status: closed

**Hypothesis:** Applying spectral absolute value before subspace projection makes the GP prior consistent with ordinary QN and prevents cancellation of signed modes.
**Outcome and uncertainty:** Cycle 29 passes training with only seven changed call counts, but loses validation cost slightly. Both splits fully converged, error-free and energy-valid. No failure traces exist.
**Reason to revisit:** A separately accepted change to Hessian learning may alter negative curvature and make this consistency useful. Training gains are concentrated, so the current result does not justify spectral-floor tuning.
**Next experiment:** Cycle 37 repeats the test with the accepted auxiliary model and again loses validation cost. No further spectral-floor tuning or immediate combination is justified; close this variant pending fundamentally different evidence.

**Attempts:**
- Cycle 29; candidate `8b8321d37acd5e698e5a87d3fe0786ce983c3e82`; champion `c94c425f8c57f8fddbe81012e20febe7625547a5`; decision non_generalizable.
  [Implementation](ideas/positive-gp-prior/8b8321d37acd5e698e5a87d3fe0786ce983c3e82/algo.py).
  [Training evidence](evaluation_results/cycle-29-train.json).
  [Validation evidence](evaluation_results/cycle-29-valid.json).
  [Narrative](full_log.md#cycle-29-project-a-consistent-positive-curvature-prior).

- Cycle 37; candidate `d06ed2091e256d219b58e8346bb899a461bde1f6`; champion `b1714c735090ec636c9accf8a541bd5559bd8577`; decision non_generalizable. Revisit under an independently accepted physical model still loses validation cost.
  [Implementation](ideas/positive-gp-prior/d06ed2091e256d219b58e8346bb899a461bde1f6/algo.py).
  [Training evidence](evaluation_results/cycle-37-train.json).
  [Validation evidence](evaluation_results/cycle-37-valid.json).
  [Narrative](full_log.md#cycle-37-positive-surrogate-prior-with-the-accepted-auxiliary-model).

**237 physical-metric follow-up:**The separately accepted234 changes the reflection itself, rather than merely improving the Hessian prior. Sharing that physically reflected full model with the GP again passes train and narrowly loses valid cost, with all molecules converged, valid energy and no errors. This is the same consistency family as29/37. Retain the shared-majorizer variant closed; a useful step majorizer need not be an accurate energy mean or kernel metric. No floor, confidence or kernel-length sweep. The spectrum handoff is reusable only for an independently derived separation of those roles.

- Cycle237; candidate `6778c4d7e11ceb4cadee87ee561618513c2eb00b`; champion `73cadc1ae8fd47e796123cafe93956f0e2c67043`; decision non_generalizable.
  [Implementation](ideas/positive-gp-prior/6778c4d7e11ceb4cadee87ee561618513c2eb00b/algo.py), [train](evaluation_results/cycle-237-train.json), [valid](evaluation_results/cycle-237-valid.json), [narrative](full_log.md#cycle-237-share-the-reflected-curvature-model-with-the-gp).

## joint-secant-fit: Regularized symmetric early curvature learning
Status: deferred

**Hypothesis:** Jointly fit the first six secants around the calibrated physical prior, avoiding information loss during sequential replay.
**Outcome and uncertainty:** Cycle 30 is energy-valid and fully converged/error-free but has a broad, substantial cost loss. There are 80 faster and 315 slower cases. No failure traces isolate a cause; shared-chart error and loss of exact newest-secant interpolation remain possible.
**Reason to revisit:** A covariant common-history representation or a derivation enforcing the newest secant while regularizing older information could change this interaction. Mere regularizer tuning lacks support.
**Next experiment:** Cycle 57 tests a concrete common-chart, newest-secant-preserving correction and still loses cost with full convergence and no errors. No immediate strength/overlap sweep; revisit only with a new, consistently derived relation between historical and native geodesic secants or passive evidence identifying a repair.

**Attempts:**
- Cycle 30; candidate `4cf145de34544b2830af3b96a0d533010dfbf1c3`; champion `c94c425f8c57f8fddbe81012e20febe7625547a5`; decision discard.
  [Implementation](ideas/joint-secant-fit/4cf145de34544b2830af3b96a0d533010dfbf1c3/algo.py).
  [Training evidence](evaluation_results/cycle-30-train.json).
  [Narrative](full_log.md#cycle-30-joint-symmetric-fitting-of-early-secants).

**Revisit cycle 57:** Derived a two-observation common-chart correction projected perpendicular to the newest step, weighted by squared angular independence. This retains the accepted early fit and newest equation, unlike cycle 30; full remote comparison pending.

- Cycle 57; candidate `5786273da35519d9b0cf8e57ae6dd557206ebc59`; champion `f56535e29a1c699bb1a1ea59ae6f57f047fda88e`; decision discard. Two-observation chart correction preserves the newest secant.
  [Implementation](ideas/joint-secant-fit/5786273da35519d9b0cf8e57ae6dd557206ebc59/algo.py).
  [Training evidence](evaluation_results/cycle-57-train.json).
  [Narrative](full_log.md#cycle-57-preserve-the-newest-secant-while-correcting-older-curvature).

## auxiliary-distance-prior: Map extra distance curvature into existing internals
Status: incorporated

**Hypothesis:** Nearby auxiliary distance springs add useful coupling/contact information to the calibrated initial curvature without enlarging the optimization manifold.
**Outcome and uncertainty:** Cycle 31 has substantial cost gains (197 faster, 106 slower) but fails aggregate energy with full convergence and zero errors. Cost gains occur in complexes and other structures, while endpoint energy losses concentrate in complexes. No failed-path traces exist.
**Reason to revisit:** Cross-fragment springs may counteract the accepted soft collective prior. Restricting auxiliary pairs to original bonded components isolates this interaction while retaining intramolecular information.
**Next experiment:** Repair 1 (cycle 32) passes both gates and is incorporated. Keep the intracomponent restriction; the unrestricted model stays rejected. Further changes require an independent hypothesis and full evaluation.

**Attempts:**
- Cycle 31; candidate `bfc98b479bc758fed7540f3ae4ed258c9758535b`; champion `c94c425f8c57f8fddbe81012e20febe7625547a5`; decision invalid.
  [Implementation](ideas/auxiliary-distance-prior/bfc98b479bc758fed7540f3ae4ed258c9758535b/algo.py).
  [Training evidence](evaluation_results/cycle-31-train.json).
  [Narrative](full_log.md#cycle-31-auxiliary-distance-curvature-in-the-initial-model).

- Cycle 32; candidate `b1714c735090ec636c9accf8a541bd5559bd8577`; champion `c94c425f8c57f8fddbe81012e20febe7625547a5`; decision keep. Repair 1 removes intercomponent springs and restores energy validity while retaining cost gains.
  Implementation retained in accepted Git history.
  [Training evidence](evaluation_results/cycle-32-train.json).
  [Validation evidence](evaluation_results/cycle-32-valid.json).
  [Narrative](full_log.md#cycle-32-keep-auxiliary-curvature-within-bonded-components).

## current-geodesic-connection: Update the connection along the geometry path
Status: incorporated

**Hypothesis:** Recompute the Jacobian pseudoinverse along the ODE to improve geodesic realization and parallel gradient transport.
**Outcome and uncertainty:** Cycle 33 improves training cost but fails energy. All converge without errors; 72 faster, 73 slower, 324 unchanged. Cost gains occur in both broad groups, while material endpoint energy losses occur in complexes. No failure trajectories exist.
**Reason to revisit:** Collective translation/rotation coordinates may interact differently with the changed connection. A coordinate-family control can retain the purely bonded benefit without recognizing molecular identities.
**Next experiment:** Repair 3 (cycle 36) corrects the buffer, passes training, then loses validation cost despite energy validity and full convergence. The bounded sequence is complete. Defer until a consistent accumulated-curvature transport derivation or another independently accepted metric change motivates a new hypothesis; no tolerance sweep.

**Attempts:**
- Cycle 33; candidate `354ee7e41136d671960b5a5327a55a33686c88a6`; champion `b1714c735090ec636c9accf8a541bd5559bd8577`; decision invalid.
  [Implementation](ideas/current-geodesic-connection/354ee7e41136d671960b5a5327a55a33686c88a6/algo.py).
  [Training evidence](evaluation_results/cycle-33-train.json).
  [Narrative](full_log.md#cycle-33-current-jacobian-in-geodesic-integration).

- Cycle 34; candidate `71ac587a2e8b245c2896b7b3ade186f2a067aa68`; champion `b1714c735090ec636c9accf8a541bd5559bd8577`; decision discard. Repair 1 restores energy but removes the cost gain.
  [Implementation](ideas/current-geodesic-connection/71ac587a2e8b245c2896b7b3ade186f2a067aa68/algo.py).
  [Training evidence](evaluation_results/cycle-34-train.json).
  [Narrative](full_log.md#cycle-34-current-geodesic-connection-for-bonded-coordinates).

- Cycle 35; candidate `56d0aabd804781c2c0c6157c8a3f601ccfcff81a`; champion `b1714c735090ec636c9accf8a541bd5559bd8577`; decision invalid. Repair 2 fails before the first step due to an ODE derivative-buffer row mismatch, confirmed by passive diagnostics.
  [Implementation](ideas/current-geodesic-connection/56d0aabd804781c2c0c6157c8a3f601ccfcff81a/algo.py).
  [Training evidence](evaluation_results/cycle-35-train.json).
  [Narrative](full_log.md#cycle-35-current-connection-secants-on-the-accepted-geometry-path).

- Cycle 36; candidate `727c02dd094ec5f273c1cd04c182fd5ba6fa1da1`; champion `b1714c735090ec636c9accf8a541bd5559bd8577`; decision non_generalizable. Repair 3 fixes only the buffer allocation; corrected transport passes training but loses validation cost.
  [Implementation](ideas/current-geodesic-connection/727c02dd094ec5f273c1cd04c182fd5ba6fa1da1/algo.py).
  [Training evidence](evaluation_results/cycle-36-train.json).
  [Validation evidence](evaluation_results/cycle-36-valid.json).
  [Narrative](full_log.md#cycle-36-repair-the-transported-secant-ode-allocation).

**Revisit186:**182 independently accepts inverse-cubic bond arc-length coordinates. Their changing radial Jacobian factors supply the previously required new metric mechanism. Repeat the full current-Jacobian path/gradient connection with unchanged tolerances and gates; retain33–36 negative evidence.

**Reassessment186:** The predeclared Badger-metric interaction passes both gates, all934 cases converged without errors. Retain current inverse-Jacobian path/gradient transport with existing ODE controls. Earlier raw-coordinate failures remain relevant; no general claim across metrics.
- Cycle186; candidate `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; champion `3fbddaadd7fef774f7b0effe823c962e6334376d`; decision keep.
  [Train](evaluation_results/cycle-186-train.json), [valid](evaluation_results/cycle-186-valid.json), [narrative](full_log.md#cycle-186-current-geodesic-connection-in-the-accepted-badger-metric). Accepted source retained in Git.

## tangent-hessian-transport: Move accumulated curvature between tangent planes
Status: deferred

**Hypothesis:** Endpoint-isometric transport of the learned tangent Hessian preserves curvature memory as the coordinate manifold turns.
**Outcome and uncertainty:** Cycle 38 is energy-valid and fully converged/error-free but slightly loses training cost (23 faster, 41 slower). No failure traces exist. The endpoint operator transport and native secant-vector transport are different approximations.
**Reason to revisit:** A derivation using one consistent transport for both the Hessian and its secants could resolve that mismatch; combining rejected transport variants without this derivation would not be justified.
**Next experiment:** Defer until that joint convention is defined and statically audited. No activation/rank-threshold tuning from current endpoint evidence.

**Attempts:**
- Cycle 38; candidate `42aae5d1bd8a620f3a6751913119482ff43bac55`; champion `b1714c735090ec636c9accf8a541bd5559bd8577`; decision discard.
  [Implementation](ideas/tangent-hessian-transport/42aae5d1bd8a620f3a6751913119482ff43bac55/algo.py).
  [Training evidence](evaluation_results/cycle-38-train.json).
  [Narrative](full_log.md#cycle-38-endpoint-isometric-transport-of-accumulated-curvature).

## element-pair-calibration: Chemical sharing of early bond stiffness scales
Status: deferred

**Hypothesis:** Separate fitted scales for unordered element pairs avoid averaging distinct bond-model errors during early curvature calibration.
**Outcome and uncertainty:** Cycle 39 passes training but loses validation cost; both splits are energy-valid, fully converged and error-free. Additional scales may be weakly identified, but no recorded fit-conditioning diagnostic establishes that explanation.
**Reason to revisit:** A physically justified hierarchical or uncertainty-aware fit could share evidence between sparse bond classes while allowing supported deviations. It should be derived from identifiability rather than validation-specific parameter tuning.
**Next experiment:** Cycle 45 tested the coupled shared/contrast prior and again lost validation cost after passing training. Defer until richer curvature observations or a directly diagnosed fit-uncertainty mechanism is available; no prior-strength sweep.

**Attempts:**
- Cycle 39; candidate `d200398823c8945de758de7710712621027287ad`; champion `b1714c735090ec636c9accf8a541bd5559bd8577`; decision non_generalizable.
  [Implementation](ideas/element-pair-calibration/d200398823c8945de758de7710712621027287ad/algo.py).
  [Training evidence](evaluation_results/cycle-39-train.json).
  [Validation evidence](evaluation_results/cycle-39-valid.json).
  [Narrative](full_log.md#cycle-39-element-pair-bond-stiffness-calibration).

- Cycle 45; candidate `85440e936e24d8cdf1754a9bfc5a98fd95addea3`; champion `f56535e29a1c699bb1a1ea59ae6f57f047fda88e`; decision non_generalizable. Repair 1 preserves the shared bond-scale prior penalty through a coupled precision matrix.
  [Implementation](ideas/element-pair-calibration/85440e936e24d8cdf1754a9bfc5a98fd95addea3/algo.py).
  [Training evidence](evaluation_results/cycle-45-train.json).
  [Validation evidence](evaluation_results/cycle-45-valid.json).
  [Narrative](full_log.md#cycle-45-shared-and-contrast-prior-for-chemical-bond-calibration).

## gradient-gp-subspace: Include current descent in the GP search space
Status: deferred

**Hypothesis:** Adding the projected current gradient permits useful nonlinear proposals beyond ordinary-step/history directions.
**Outcome and uncertainty:** Cycle 40 loses training cost, with 15 faster and 20 slower cases, zero errors and valid energy. One 200-call trace shows continuing descent with force/displacement criteria unmet. It does not reveal which GP proposals were accepted.
**Reason to revisit:** A consistent transported observation model or directional uncertainty rule could determine whether the new direction is supported. Existing passive evidence does not justify dimension/rank/acceptance tuning.
**Next experiment:** Defer until that modeling improvement is specified, then test the added direction independently against the current accepted GP.

**Attempts:**
- Cycle 40; candidate `ef974aeaea6ef33f75fc956471cee21862083b71`; champion `b1714c735090ec636c9accf8a541bd5559bd8577`; decision discard.
  [Implementation](ideas/gradient-gp-subspace/ef974aeaea6ef33f75fc956471cee21862083b71/algo.py).
  [Training evidence](evaluation_results/cycle-40-train.json).
  [Passive trace](diagnostics/91e3c67fdecc4e1abe39985dee4774c3/9397e7612d708c137e5a7477345e723a21223a14af08e4c24489ea20cfe2ac8f.json).
  [Narrative](full_log.md#cycle-40-gradient-enriched-surrogate-search-subspace).

## box-quadratic-step: Refine boundary steps inside the component trust box
Status: deferred

**Hypothesis:** Full tangent-space minimization within existing component bounds can improve on the scalar-shifted ordinary-step path.
**Outcome and uncertainty:** Cycle 42 is valid and converged but loses training cost, concentrated in complexes. Non-complex molecules gain slightly. No passive failures exist; excessive distributed collective motion is a hypothesis from the method and endpoint grouping, not an observed internal trace.
**Reason to revisit:** The small gain outside complexes suggests a probe restricted by actual coordinate representation could retain benefit where the physical model is stronger.
**Next experiment:** Repair 1 (cycle 43) passed training but lost validation cost, with valid energy and full convergence. Defer until model fidelity can constrain aggressive boundary directions on a derived basis; current endpoints do not justify further parameter tuning.

**Attempts:**
- Cycle 42; candidate `6b76b1eb5ba8bfb8b8c103e66056470b5676d992`; champion `f56535e29a1c699bb1a1ea59ae6f57f047fda88e`; decision discard.
  [Implementation](ideas/box-quadratic-step/6b76b1eb5ba8bfb8b8c103e66056470b5676d992/algo.py).
  [Training evidence](evaluation_results/cycle-42-train.json).
  [Narrative](full_log.md#cycle-42-quadratic-minimization-inside-the-internal-step-box).

- Cycle 43; candidate `e7648efc936ab0e68355951e3cbc65368add7048`; champion `f56535e29a1c699bb1a1ea59ae6f57f047fda88e`; decision non_generalizable. Repair 1 restricts refinement to actual bonded-only coordinate systems.
  [Implementation](ideas/box-quadratic-step/e7648efc936ab0e68355951e3cbc65368add7048/algo.py).
  [Training evidence](evaluation_results/cycle-43-train.json).
  [Validation evidence](evaluation_results/cycle-43-valid.json).
  [Narrative](full_log.md#cycle-43-component-box-refinement-only-for-bonded-coordinates).

## common-chart-gp: Differentiate historical observations in one tangent chart
Status: deferred

**Hypothesis:** Pull historical gradients through the inverse overlap of old/current tangent bases to match the GP's tangent-projected history positions.
**Outcome and uncertainty:** Cycle 44 slightly loses training cost (six faster, nine slower), with valid energy, full convergence and no errors. There are no failure traces. Native geodesic realization still differs from the orthographic chart used by the surrogate.
**Reason to revisit:** A matching inverse-chart geometry solver and coherent curvature-secant convention could remove this remaining inconsistency. The old full-redundant target converter does not satisfy the derived chart equation.
**Next experiment:** Cycle 56 tested the derived inverse chart and dual secant/Hessian transport but still lost training cost, fully converged and error-free. No immediate repair or overlap/tolerance sweep is supported; revisit only with a separately improved surrogate model or diagnosed chart failure.

**Attempts:**
- Cycle 44; candidate `25ee3d6b59bf3db31e30c6edada6d3c6df1bc5b7`; champion `f56535e29a1c699bb1a1ea59ae6f57f047fda88e`; decision discard.
  [Implementation](ideas/common-chart-gp/25ee3d6b59bf3db31e30c6edada6d3c6df1bc5b7/algo.py).
  [Training evidence](evaluation_results/cycle-44-train.json).
  [Narrative](full_log.md#cycle-44-common-chart-historical-gradients-for-the-gp).

**Revisit now:** A bounded inverse of the tangent-projection chart and matching dual secant/Hessian transport have been derived (cycle 55 notes). Cycle 56 will test them jointly with the historical-gradient pullback, preparing geometry without force calls and falling back before evaluation.

- Cycle 56; candidate `5e90fe2a50ebe1ceffb1b302b2d895f243a53bb6`; champion `f56535e29a1c699bb1a1ea59ae6f57f047fda88e`; decision discard. Matched chart realization and dual transport.
  [Implementation](ideas/common-chart-gp/5e90fe2a50ebe1ceffb1b302b2d895f243a53bb6/algo.py).
  [Training evidence](evaluation_results/cycle-56-train.json).
  [Narrative](full_log.md#cycle-56-match-gp-coordinates-to-an-orthographic-geometry-step).

**377 revisit:**Accepted298/307 independently improve the scalar mean and its covariance, satisfying56's declared surrogate-model prerequisite. Reuse56's unchanged inverse-chart bounds and dual transport with these models; retain307's early GP before six secants, activating the matched chart only afterward as56 did. Preserve current collision checks and use ordinary fallback before force evaluation on chart failure. This is a component interaction, not a diagnosed repair or tolerance sweep.

**377 outcome:**The accepted bending surrogate does not rescue the matched chart:469 converged,zero errors, energy0.9997926630450247 and cost0.751048741503069 fail. One higher-energy converged endpoint dominates recovery loss; empty failure_details do not identify its mechanism. Further work requires actual chart/model/transport evidence, not another surrogate composition or tolerance/activation sweep.
- Cycle377; candidate `a144bd9e8061158c3c3b843557b80eae47794529`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision invalid. [Implementation](ideas/common-chart-gp/a144bd9e8061158c3c3b843557b80eae47794529/algo.py), [train](evaluation_results/cycle-377-train.json), [narrative](full_log.md#cycle-377-matched-orthographic-gp-geometry-with-the-bending-model).

## full-frame-geodesic: One parallel frame for path and curvature state
Status: deferred

**Hypothesis:** Integrating a full tangent frame makes velocity, transported gradient/step and accumulated Hessian share one connection, avoiding the mismatches of earlier transport variants.
**Outcome and uncertainty:** Cycle 46 is energy-valid, fully converged and error-free but slightly loses training cost in both broad groups (44 faster, 59 slower). No passive failure bundle exists; successful fallback rates are unobserved.
**Reason to revisit:** A separately supported geometric metric or model-fidelity improvement might make consistent transport useful. Numerical efficiency alone would not address the measured force-call loss. A prediction-timing control is being isolated first to clarify that separate mechanism.
**Next experiment:** Defer full-frame changes; no ODE tolerance sweep. Test direct initial-model prediction independently from the champion. Any later metric change must derive both tangent and covector/operator transports consistently.

**Attempts:**
- Cycle 46; candidate `9dec621e1a563ef3c6743876f7ca05f5770c4e37`; champion `f56535e29a1c699bb1a1ea59ae6f57f047fda88e`; decision discard.
  [Implementation](ideas/full-frame-geodesic/9dec621e1a563ef3c6743876f7ca05f5770c4e37/algo.py).
  [Training evidence](evaluation_results/cycle-46-train.json).
  [Narrative](full_log.md#cycle-46-full-frame-geodesic-curvature-transport).

**Revisit189:**182/186 supply independently accepted nonlinear geometry/current connection. Integrate a shared frame in that connection and transport H, physical blocks and cached secants together, fixing46's early-refit mismatch and132's endpoint-only connection. Use a derived orthogonal ambient extension to preserve fit statistics; retain original trust prediction after188's negative control.

**Reassessment189:**The shared geodesic frame with joint H/block/secant transport fully converges and passes energy but loses cost. End this revisit. A future attempt needs independently supported covariant physical modeling or diagnosed transport error; no tolerance or activation sweep. Explicit-integrator and fallback effects remain unisolated.
- Cycle189; candidate `f50fca2885de01827811cb369064df71c727d9c3`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision discard.
  [Implementation](ideas/full-frame-geodesic/f50fca2885de01827811cb369064df71c727d9c3/algo.py), [train](evaluation_results/cycle-189-train.json), [narrative](full_log.md#cycle-189-geodesic-transport-of-the-complete-curvature-state).

## initial-model-prediction: Keep trust prediction in the initial Taylor model
Status: deferred

**Hypothesis:** Using the saved initial gradient, Hessian and step directly avoids projecting that model through the moved tangent basis.
**Outcome and uncertainty:** Cycle 47 is fully converged, energy-valid and error-free but slightly loses training cost. Mathematical prediction consistency does not improve this accepted approximate geometry/model pipeline.
**Reason to revisit:** A separately accepted geometry or model change may require this initial-model convention. Preserve the small control for such a derivation, rather than tuning radius thresholds to make it pass.
**Next experiment:** No immediate repair. Continue independent globalization work; retain current champion prediction until a new mechanism passes both gates.

**Attempts:**
- Cycle 47; candidate `47fa4c545f30e66a3c0a27d94c44bc80e1f1d6a1`; champion `f56535e29a1c699bb1a1ea59ae6f57f047fda88e`; decision discard.
  [Implementation](ideas/initial-model-prediction/47fa4c545f30e66a3c0a27d94c44bc80e1f1d6a1/algo.py).
  [Training evidence](evaluation_results/cycle-47-train.json).
  [Narrative](full_log.md#cycle-47-predict-energy-change-in-the-initial-quadratic-model).

**Revisit188:**182 and186 independently accept a nonlinear bond chart and current-Jacobian path/gradient connection. Retest47's isolated initial contraction under that geometry, without changing transported secants, model learning or controller constants.

**Reassessment188:**The accepted nonlinear chart/current connection does not make initial contraction improve cost. All469 converge and pass energy; end this revisit without radius tuning. Require a distinct derived prediction model or diagnosed controller failure before revisiting again.
- Cycle188; candidate `7339c2cac0d080ad12fdf5dc4fd8ffda4413b821`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision discard.
  [Implementation](ideas/initial-model-prediction/7339c2cac0d080ad12fdf5dc4fd8ffda4413b821/algo.py), [train](evaluation_results/cycle-188-train.json), [narrative](full_log.md#cycle-188-initial-taylor-prediction-with-the-accepted-current-connection).

## bounded-backtracking: Retry insufficient-decrease internal steps
Status: deferred

**Hypothesis:** A bounded number of real smaller-step trials may avoid expensive recovery after poor full steps, while updating curvature only from the final trial.
**Outcome and uncertainty:** Cycle 48 is fully converged and error-free but fails energy and loses cost, concentrated in complexes. Non-complex cases gain slightly. No trial/failure traces are retained. The state, call-count and last-evaluated-geometry contract passes this complete evaluation.
**Reason to revisit:** Weak collective rearrangements may be harmed by sufficient-decrease selection while bonded internal motion benefits. Actual primitive types can test this localized explanation without dataset labels.
**Next experiment:** No immediate repair. Cycle 49 removes the energy loss but still loses cost (11 faster, 24 slower, 434 unchanged), contradicting the proposed bonded-only benefit. Revisit only with evidence of repeated failed trials or a separately accepted model that can predict when real retries pay for themselves; no radius-memory tuning is supported.

**Attempts:**
- Cycle 48; candidate `936398aa53ba66ed14627c742cfb2147dff359fc`; champion `f56535e29a1c699bb1a1ea59ae6f57f047fda88e`; decision invalid.
  [Implementation](ideas/bounded-backtracking/936398aa53ba66ed14627c742cfb2147dff359fc/algo.py).
  [Training evidence](evaluation_results/cycle-48-train.json).
  [Narrative](full_log.md#cycle-48-bounded-backtracking-with-real-trial-evaluations).

- Cycle 49; candidate `7036305cbd4231c40b6ce6f2ff4a42098565f8d2`; champion `f56535e29a1c699bb1a1ea59ae6f57f047fda88e`; decision discard.
  [Implementation](ideas/bounded-backtracking/7036305cbd4231c40b6ce6f2ff4a42098565f8d2/algo.py).
  [Training evidence](evaluation_results/cycle-49-train.json).
  [Narrative](full_log.md#cycle-49-backtracking-limited-to-bonded-coordinate-systems).

## morse-gp-mean: Centered anharmonic bond prior for the GP
Status: deferred

**Hypothesis:** A Morse-minus-harmonic correction preserves current gradient/curvature while improving finite bond-displacement predictions.
**Outcome and uncertainty:** Cycle 50 is fully converged and energy-valid with zero errors but slightly loses cost; 11 faster, 17 slower, 441 unchanged. No failure traces or surrogate-acceptance diagnostics identify a specific repair.
**Reason to revisit:** An accepted change to surrogate coordinates or observation consistency could allow this nonlinear physical mean to matter more reliably. The complete mean and derivative implementation is reusable.
**Next experiment:** No immediate exponent/strength sweep. Reconsider after an independently accepted surrogate representation change, comparing against that new champion under both gates.

**Attempts:**
- Cycle 50; candidate `fcba597d4e3300b27b7ad8a2919446c8dd164e79`; champion `f56535e29a1c699bb1a1ea59ae6f57f047fda88e`; decision discard.
  [Implementation](ideas/morse-gp-mean/fcba597d4e3300b27b7ad8a2919446c8dd164e79/algo.py).
  [Training evidence](evaluation_results/cycle-50-train.json).
  [Narrative](full_log.md#cycle-50-anharmonic-bond-prior-in-the-gp-surrogate).

**Cycle 127 reassessment:** The unchanged Morse prior with accepted tangent derivative maps again loses training cost while remaining energy-valid and fully converged. Observation consistency alone is insufficient. Revisit only with independent evidence for finite bond-displacement mean error or a derived representation that realizes it; no exponent/strength tuning.
- Cycle 127; candidate `23e2dd67def9204034ba42c88421b38e4a3d39d3`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision discard.
  [Implementation](ideas/morse-gp-mean/23e2dd67def9204034ba42c88421b38e4a3d39d3/algo.py).
  [Training evidence](evaluation_results/cycle-127-train.json).
  [Narrative](full_log.md#cycle-127-morse-nonlinear-mean-with-tangent-aware-gp-observations).

## weighted-coordinate-regression: Weighted scalar gradient trends in internal coordinates
Status: deferred

**Hypothesis:** Neighbor-weighted scalar gradient regressions can exploit approximate coordinate independence and smooth noisy curvature information.
**Outcome and uncertainty:** Cycle 51 has 13 faster but 420 slower cases, 25 force-call limits, zero errors and failed energy recovery. Two passive traces show long descent with late oscillations. Fit states are absent, so intercept drift and coordinate coupling are explanations to investigate, not established causes. This adaptation replaces GP refinement but retains native geodesic realization and quadratic trust feedback.
**Reason to revisit:** A current-gradient-anchored model in a consistent coordinate chart could remove drifting fitted roots, while explicit coupling correction may avoid the severe scalar-independence failure. This is a formulation change requiring derivation, not a window/weight sweep.
**Next experiment:** No immediate repair after broad loss. Revisit only with a coherent coupled/anchored formulation, retaining native fallback and full gates. Preserve the source-inspired implementation and both traces as controls.

**Attempts:**
- Cycle 51; candidate `ce3fd31b6acd71ca50217e5819232c72b97c0ae0`; champion `f56535e29a1c699bb1a1ea59ae6f57f047fda88e`; decision invalid.
  [Implementation](ideas/weighted-coordinate-regression/ce3fd31b6acd71ca50217e5819232c72b97c0ae0/algo.py).
  [Training evidence](evaluation_results/cycle-51-train.json).
  [Single-molecule trace](diagnostics/7f9474a65da6403abf87e7ddcb3dc382/0962c6552e47396d113183652eb491e9efaa7daebcb32af202d6b50d4979b8aa.json).
  [Complex trace](diagnostics/7f9474a65da6403abf87e7ddcb3dc382/0e8ec1f89fa24ee9673db386e8d7c70161f2273062bf7c4bba8475dad2a1f038.json).
  [Narrative](full_log.md#cycle-51-weighted-primitive-coordinate-gradient-regression).

## metric-hessian-blend: Stiffness-scaled secant residual alignment
Status: deferred

**Hypothesis:** Positive diagonal curvature scales balance the primal step and dual gradient residual in the accepted Hessian mixture.
**Outcome and uncertainty:** Cycle 52 is converged, energy-valid and error-free but loses total cost: 107 faster, 98 slower, 264 unchanged. Complexes lose cost while other systems gain; no trajectory/metric diagnostics establish why.
**Reason to revisit:** Collective coordinates use a much softer prior and may distort this diagonal metric, while bonded-coordinate scaling may retain the observed benefit.
**Next experiment:** Repair 1 (cycle 53) passes training but loses validation cost, both fully converged and energy-valid. No immediate further repair; retain the accepted Euclidean criterion. Revisit only after an independently justified curvature representation change, not a metric-floor or mixing sweep.

**Attempts:**
- Cycle 52; candidate `30e1cc5e3c9a7f71ee7ab19047c5aa23e360c6ca`; champion `f56535e29a1c699bb1a1ea59ae6f57f047fda88e`; decision discard.
  [Implementation](ideas/metric-hessian-blend/30e1cc5e3c9a7f71ee7ab19047c5aa23e360c6ca/algo.py).
  [Training evidence](evaluation_results/cycle-52-train.json).
  [Narrative](full_log.md#cycle-52-stiffness-scaled-alignment-for-mixed-hessian-updates).

- Cycle 53; candidate `9182da7e4c744223a2b3e59a6c38ec0d69d68acd`; champion `f56535e29a1c699bb1a1ea59ae6f57f047fda88e`; decision non_generalizable. Repair 1 restricts the metric to bonded coordinates.
  [Implementation](ideas/metric-hessian-blend/9182da7e4c744223a2b3e59a6c38ec0d69d68acd/algo.py).
  [Training evidence](evaluation_results/cycle-53-train.json).
  [Validation evidence](evaluation_results/cycle-53-valid.json).
  [Narrative](full_log.md#cycle-53-metric-blending-only-for-bonded-coordinate-systems).


**Revisit cycle 146:** Replace the earlier diagonal approximation by the full current spectral primal/dual metric, preserving mode coupling after accepted stretch-bend calibration. This is a newly derived metric, not a floor or bonded-only activation adjustment; use the same denominator-free blend and unchanged gates.

**Reassessment 146:** Full spectral metric also loses training cost with complete convergence and no errors; no immediate floor or mixing repair. Revisit only with evidence that identifies unreliable curvature modes and a general model for them.
- Cycle 146; candidate `f3f78deb14dc15926f555e1f1be0e9d37e76a404`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision discard.
  [Implementation](ideas/metric-hessian-blend/f3f78deb14dc15926f555e1f1be0e9d37e76a404/algo.py).
  [Training evidence](evaluation_results/cycle-146-train.json).
  [Narrative](full_log.md#cycle-146-full-spectral-metric-for-secant-residual-blending).

**247 fixed-primitive-metric control:**234/244 independently support initial-D reflection in steps and TS correction directions. Replacing only SR1 blend alignment by that fixed primal/dual metric still loses cost broadly, with all469 converged and valid energy.52 used learned diagonals and146 learned full spectra; this distinct fixed-metric control also fails. Defer until observed secant-error/mode information, not another metric/floor/weight/subgroup variant.
- Cycle247; candidate `bdc3a2d2acea4528b1a8147d6866f2d778cd38fd`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision discard. [Implementation](ideas/metric-hessian-blend/bdc3a2d2acea4528b1a8147d6866f2d778cd38fd/algo.py), [train](evaluation_results/cycle-247-train.json), [narrative](full_log.md#cycle-247-fixed-physical-metric-for-ts-sr1-residual-alignment).

## collective-contact-correlations: Bounded interfragment motion correlations
Status: deferred

**Hypothesis:** A contact-derived correlation matrix can improve collective-coordinate coupling without adding raw interfragment stiffness.
**Outcome and uncertainty:** Cycle 54 narrowly passes training (39 faster, 36 slower, 394 unchanged) but loses validation cost. Both splits are fully converged, energy-valid and error-free; no failure traces exist. Primitive diagonals are preserved before projection, with half-to-double quadratic-form bounds after projection.
**Reason to revisit:** A more faithful directional interaction model may distinguish meaningful collective couplings from mere contact geometry while retaining this stable bounded construction.
**Next experiment:** No cutoff or shrinkage sweep. Revisit after a concrete improved interaction model is derived; compare against the then-current champion under full gates. Keep the accepted isotropic collective block meanwhile.

**Attempts:**
- Cycle 54; candidate `8d6325fa6bed67cefcdf052495148766632bc2b5`; champion `f56535e29a1c699bb1a1ea59ae6f57f047fda88e`; decision non_generalizable.
  [Implementation](ideas/collective-contact-correlations/8d6325fa6bed67cefcdf052495148766632bc2b5/algo.py).
  [Training evidence](evaluation_results/cycle-54-train.json).
  [Validation evidence](evaluation_results/cycle-54-valid.json).
  [Narrative](full_log.md#cycle-54-bounded-contact-correlations-in-the-collective-prior).

**Revisit 149:** Derive and include transverse spectral curvature from the existing exponential central interaction. Radial weights, cutoff and bounded correlation rule remain those of 54; this supplies a concrete directional interaction model instead of tuning shrinkage.

**Reassessment 149:** The transverse repulsive contribution remains slower with full convergence and valid energy. Revisit only after a physically supported attractive/electrostatic interaction model is derived; no immediate cutoff/exponent/shrink sweep.
- Cycle 149; candidate `9f0b31313cc0a30447b1cbd0c647e764ddc1d57e`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision discard.
  [Implementation](ideas/collective-contact-correlations/9f0b31313cc0a30447b1cbd0c647e764ddc1d57e/algo.py).
  [Training evidence](evaluation_results/cycle-149-train.json).
  [Narrative](full_log.md#cycle-149-full-central-pair-curvature-in-bounded-collective-correlations).

**Revisit194:**UFF12-6 attractive/repulsive radial and transverse spectral curvature replaces149's pure repulsion. All bounded normalization and contact rules remain fixed, satisfying the declared attractive-model prerequisite. Full remote comparison pending.

**Reassessment194:**The UFF12-6 attractive/repulsive tensor remains slower with complete convergence, valid energy and no errors. This tested central-force model is insufficient; defer until independent directional/electrostatic evidence supports a different model. No cutoff, interaction-strength or normalization sweep.
- Cycle194; candidate `480dcc4720fc17031dce6e1efe00487a071cc87e`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision discard.
  [Implementation](ideas/collective-contact-correlations/480dcc4720fc17031dce6e1efe00487a071cc87e/algo.py), [train](evaluation_results/cycle-194-train.json), [narrative](full_log.md#cycle-194-dispersion-aware-bounded-collective-correlations).

**Cycle222 prerequisite:**Use Gaussian QEq screened electrostatic pair Hessian magnitudes as the directional interaction model; retain194 mapping/cutoff/normalization/shrinkage/diagonals. This supplies the explicit electrostatic alternative, with a neutral initial charge-model assumption and fixed-charge Hessian limitation documented in the log.

**Cycle222 outcome:**The independently derived neutral Gaussian-QEq electrostatic tensor fully converges and passes energy but loses cost. Missing charge response, neutrality/charge-transfer assumptions and normalized shape remain unresolved, with no diagnostic charge/model state. Revisit only with an independently supported charge-response/interaction model; no charge/cutoff/shrinkage/strength tuning.
- Cycle222; candidate `d09f6be4c76fe57df12147c9368103d240aa24b2`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision discard.
  [Implementation](ideas/collective-contact-correlations/d09f6be4c76fe57df12147c9368103d240aa24b2/algo.py), [train](evaluation_results/cycle-222-train.json), [narrative](full_log.md#cycle-222-screened-electrostatic-collective-correlations).

**Revisit228:**222's charge-response prerequisite is supplied by constrained implicit differentiation. Replace frozen contact-pair tensors with the fully relaxed scalar Gaussian-QEq Hessian and its native chain-rule correction, before the same bounded collective normalization. No charge/cutoff/shrinkage-strength sweep.

**228 outcome/reassessment:**Fully differentiated relaxed scalar QEq model still loses cost with all469 converged and valid energy. Explicit charge response fulfills222's prerequisite but does not rescue the prior. Await an independently supported charge-state/charge-transfer model or measured model defect; no parameter sweeps.
- Cycle228; candidate `0937d8a98854fc10ae52cd5ebab31799408bfa4f`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision discard.
  [Implementation](ideas/collective-contact-correlations/0937d8a98854fc10ae52cd5ebab31799408bfa4f/algo.py), [train](evaluation_results/cycle-228-train.json), [narrative](full_log.md#cycle-228-relaxed-electrostatic-collective-curvature).

**441 revisit:**QTPIE supplies228's independently supported charge-transfer prerequisite through normalized-overlap effective electronegativities. Differentiate the full scalar, including electronegativity response, with the same OpenBabel parameters and collective normalization; compare against430. No charge,cutoff ormixing sweep.

**441 reassessment:**Fully differentiated QTPIE passes energy and converges all469 but loses cost. The independent charge-transfer prerequisite does not rescue this collective prior; no localized passive failure. Further work requires independently supported collective curvature or charge-state evidence, not overlap,radius,charge,mixing ornormalization sweeps.
- Cycle441; candidate `8535a4ee633c4ed65330554671ecc7cd094b3d47`; champion `bb770d9c7c9e59d39007f71a106af77cd9eb73a1`; decision discard. [Implementation](ideas/collective-contact-correlations/8535a4ee633c4ed65330554671ecc7cd094b3d47/algo.py), [train](evaluation_results/cycle-441-train.json), [narrative](full_log.md#cycle-441-qtpie-relaxed-collective-correlations).

## shifted-newton-spectrum: Uniform regularization of indefinite Newton steps
Status: deferred

**Hypothesis:** A uniform shift preserves signed curvature ordering, allowing strongly negative directions to remain soft within the existing component restriction.
**Outcome and uncertainty:** Cycle 55 is fully converged and energy-valid with zero errors but slightly loses cost; 48 faster, 48 slower and 373 unchanged. No failure traces establish a step-size or numerical issue.
**Reason to revisit:** An independently improved curvature model could make negative-mode ordering more reliable. The small step-model control is distinct from rejected RFO.
**Next experiment:** No immediate trust-threshold or spectral-floor sweep. Reconsider after a justified model change, comparing with the accepted absolute-spectrum path.

**Attempts:**
- Cycle 55; candidate `7358c01aaba92d3061576e73294af719dcb4d490`; champion `f56535e29a1c699bb1a1ea59ae6f57f047fda88e`; decision discard.
  [Implementation](ideas/shifted-newton-spectrum/7358c01aaba92d3061576e73294af719dcb4d490/algo.py).
  [Training evidence](evaluation_results/cycle-55-train.json).
  [Narrative](full_log.md#cycle-55-uniformly-shift-an-indefinite-newton-spectrum).

**Cycle488 prerequisite/revisit:** Accepted234 physical metric and450/463 electronic response supply independently improved curvature. Replace reflection by a uniform shift in that same physical congruence, preserving55's1e-8 floor and fallback without retuning the controller. [Narrative](full_log.md#cycle-488-signed-curvature-shift-in-the-physical-metric).

**488 reassessment:** Metric-consistent uniform shifting with the accepted electronic bond model fully converges and passes energy, but loses cost;{'slow': 56, 'same': 363, 'fast': 50}. The independent-curvature prerequisite is insufficient. Await measured negative-mode/model-error evidence; no metric, floor, radius or activation sweeps.
- Cycle488; candidate `b09174de47a6913d76a154df75f975ef7c618229`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/shifted-newton-spectrum/b09174de47a6913d76a154df75f975ef7c618229/algo.py), [training](evaluation_results/cycle-488-train.json), [narrative](full_log.md#cycle-488-signed-curvature-shift-in-the-physical-metric).

## bounded-physical-fit: Solve coupled coefficient bounds jointly
Status: deferred

**Hypothesis:** A true bounded least-squares solution can improve the physical prior when clipping an unconstrained fit leaves interior coefficients nonstationary.
**Outcome and uncertainty:** Cycle 58 loses training cost with valid energy, full convergence and zero errors. No failure traces exist. More accurate calibration is not necessarily a better optimization model.
**Reason to revisit:** Improved, consistently represented fit observations could make the exact bounded objective more useful; the small solver control is reusable.
**Next experiment:** No bound, ridge or solver-tolerance sweep. Revisit only after an independently accepted change to the calibration model or observations.

**Attempts:**
- Cycle 58; candidate `a529c7a2b27941cc8a2166acc7bd09f69816952f`; champion `f56535e29a1c699bb1a1ea59ae6f57f047fda88e`; decision discard.
  [Implementation](ideas/bounded-physical-fit/a529c7a2b27941cc8a2166acc7bd09f69816952f/algo.py).
  [Training evidence](evaluation_results/cycle-58-train.json).
  [Narrative](full_log.md#cycle-58-solve-the-bounded-physical-calibration-jointly).


**Revisit cycle 144:** Accepted cycle 82 adds a zero-centered signed stretch-bend feature to the calibration, meeting the declared model-change condition. Test the same BVLS solver with the champion's per-variable positive/signed bounds; do not change ridge, bounds or solver settings.


**Cycle 144 reassessment:** Joint bounds with the accepted signed stretch-bend feature again lose training cost, despite full convergence, energy validity and no errors. The declared interaction did not help. Defer until independently improved calibration observations or passive fit evidence justify further work; no bound/ridge/solver sweep.
- Cycle 144; candidate `aa72f56e94d9843ea798f92fa3e5533d7b0c54d7`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision discard.
  [Implementation](ideas/bounded-physical-fit/aa72f56e94d9843ea798f92fa3e5533d7b0c54d7/algo.py).
  [Training evidence](evaluation_results/cycle-144-train.json).
  [Narrative](full_log.md#cycle-144-joint-bounded-calibration-with-signed-stretch-bend-coupling).

**Revisit cycle231:**182 changes the bond chart and186 changes the connection used for gradient transport, directly improving the observation representation since144. Test the exact previous bounded solver unchanged on186. This fulfills the recorded observation-change condition; if it loses broadly again, require observed fit/model evidence before another revisit.

**Cycle231 reassessment:**The182 bond chart and186 current path connection do not rescue the unchanged BVLS control: full convergence and valid energy, but another cost loss. Further revisits require observed calibration/model evidence, not another unrelated accepted ancestor change. No parameter sweep.
- Cycle231; candidate `30bce7db9c37f4151c55623470ab551e262ac539`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision discard.
  [Implementation](ideas/bounded-physical-fit/30bce7db9c37f4151c55623470ab551e262ac539/algo.py), [train](evaluation_results/cycle-231-train.json), [narrative](full_log.md#cycle-231-joint-coefficient-bounds-with-current-geodesic-observations).

## weighted-coordinate-metric: Fixed stiffness-weighted internal geometry
Status: deferred

**Hypothesis:** Transform the full redundant-coordinate representation by fixed initial stiffness weights to improve its metric, while retaining physical displacement limits.
**Outcome and uncertainty:** Cycle 59 is energy-valid and error-free but loses cost in both complexes and other systems; 141 faster, 184 slower, 144 unchanged. One force-call limit has sustained late descent. The converged subset also loses cost; passive data lacks weight/connection states.
**Reason to revisit:** A metric tied more faithfully to learned curvature or a consistent connection/transport model could change this result. The adapter already handles values, derivatives, Hessian units, periodic wrapping and physical safeguards.
**Next experiment:** No normalization, rank-threshold or collective-guard sweep. Revisit only after deriving a new metric/transport interaction supported by evidence; compare to the then-current champion under full gates.

**Attempts:**
- Cycle 59; candidate `de94d210617aa58b137927b543cf76516216f729`; champion `f56535e29a1c699bb1a1ea59ae6f57f047fda88e`; decision discard.
  [Implementation](ideas/weighted-coordinate-metric/de94d210617aa58b137927b543cf76516216f729/algo.py).
  [Training evidence](evaluation_results/cycle-59-train.json).
  [Passive trace](diagnostics/9a0ec509b61e4134b557245b8fd02f3f/742a200e8bc36d3890ed373425d8d414106d41cf4b157a57ce2f3ca64b9b3621.json).
  [Narrative](full_log.md#cycle-59-fixed-force-constant-weighted-internal-coordinates).

**Cycle219 prerequisite:**Compose the exact59 constant-weight adapter with accepted182 nonlinear bonds and186 current weighted connection. Preserve59 normalization and displacement guards; no failed218 bond prior. The independently accepted connection supplies the recorded metric/transport prerequisite.

**Cycle219 outcome:**The fixed59 weights composed with accepted182/186 fully converge and pass aggregate energy but still lose cost. The current connection prerequisite is now tested. Require an independently supported learned metric or diagnosed weighted-model defect before revisiting; no normalization/rank/guard sweep.
- Cycle219; candidate `14b751653892cb38a18cbb8c3729e24cb7fbae20`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision discard.
  [Implementation](ideas/weighted-coordinate-metric/14b751653892cb38a18cbb8c3729e24cb7fbae20/algo.py), [train](evaluation_results/cycle-219-train.json), [narrative](full_log.md#cycle-219-force-weighted-badger-geometry-with-current-connection).

**Revisit283:**Test a derived frozen full learned-curvature metric with consistent vector transport and native covector conversion. Accepted234/244 support its positive curvature component; whether that transfers to geodesic geometry remains unproven. This is distinct from59/219 fixed diagonal coordinate weights; no failed normalization or rank parameter is tuned.

**Cycle283 outcome:**Learned full curvature with metric-consistent gradient transport is error-free, fully converged and energy-valid but loses cost. This tests the learned-metric extension promised after219. Defer pending actual metric/connection discrepancy evidence or an independently justified model-consistent full-tensor transport; no numerical-floor/normal-completion/activation/tolerance tuning.
- Cycle283; candidate `7b8ec8ac16480ead87baa4a2996c11e7664d1cbc`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision discard.
  [Implementation](ideas/weighted-coordinate-metric/7b8ec8ac16480ead87baa4a2996c11e7664d1cbc/algo.py), [train](evaluation_results/cycle-283-train.json), [narrative](full_log.md#cycle-283-learned-metric-geodesics-with-dual-consistent-gradient-transport).

## admissible-gp-iterates: Retain safe intermediate surrogate candidates
Status: deferred

**Hypothesis:** A useful admissible intermediate GP point can be retained when the final microiterate violates a safeguard.
**Outcome and uncertainty:** Cycle 60 is fully converged, energy-valid and error-free but loses training cost. No failure traces isolate a repair; lower model energy is not sufficient for better physical progress.
**Reason to revisit:** An independently more accurate surrogate or calibrated confidence model could make intermediate-point selection useful. The callback changes no solver limit or physical budget.
**Next experiment:** No confidence-threshold or callback-frequency sweep. Reconsider only after an accepted surrogate-model improvement, comparing to its unchanged selection rule.

**Attempts:**
- Cycle 60; candidate `90fcac0649656748e02735f28e27f42c724db1d5`; champion `f56535e29a1c699bb1a1ea59ae6f57f047fda88e`; decision discard.
  [Implementation](ideas/admissible-gp-iterates/90fcac0649656748e02735f28e27f42c724db1d5/algo.py).
  [Training evidence](evaluation_results/cycle-60-train.json).
  [Narrative](full_log.md#cycle-60-retain-admissible-intermediate-gp-steps).

**Revisit305:**Accepted298 supplies60's required independent surrogate-model improvement. Reuse the same callback selection and all solver/endpoint controls, adding only per-point angular-domain exception handling.300's two-start loss remains contrary evidence for broad selection gains.

**305 reassessment:**The required independent mean improvement does not rescue callback selection: all469 converge with valid energy and no errors but cost worsens. Defer until direct search-path/confidence evidence distinguishes why an intermediate point is useful; no further budget, threshold or frequency sweep.
- Cycle305; candidate `ffb532810c948704326ce6c2809b1f960e30b0d0`; champion `7c0bbf8190d01c916f081326421250f6b036918c`; decision discard.
  [Implementation](ideas/admissible-gp-iterates/ffb532810c948704326ce6c2809b1f960e30b0d0/algo.py).
  [Training evidence](evaluation_results/cycle-305-train.json).
  [Narrative](full_log.md#cycle-305-admissible-gp-iterates-with-the-accepted-bending-mean).

## secant-cubic-damping: Model-error-calibrated cubic Newton steps
Status: deferred

**Hypothesis:** Previous secant prediction error estimates a useful cubic penalty, limiting unreliable model steps before physical evaluation.
**Outcome and uncertainty:** Cycle 61 is fully converged and error-free but has large cost loss and fails energy recovery. No failure traces exist. Pre-update error can be corrected by the ensuing Hessian update, so its use as future uncertainty may over-damp.
**Reason to revisit:** A justified uncertainty estimate or a complete adaptive cubic controller with matched geometry could make the scalar step solver useful. The tested heuristic is not a full ARC algorithm.
**Next experiment:** No coefficient/floor sweep. Revisit only with a new model-error estimate or complete controller derivation; retain all force-cost and energy gates.

**Attempts:**
- Cycle 61; candidate `a9358f40ae89b94f13ae76717488ad26a3a86bc1`; champion `f56535e29a1c699bb1a1ea59ae6f57f047fda88e`; decision invalid.
  [Implementation](ideas/secant-cubic-damping/a9358f40ae89b94f13ae76717488ad26a3a86bc1/algo.py).
  [Training evidence](evaluation_results/cycle-61-train.json).
  [Narrative](full_log.md#cycle-61-secant-error-calibrated-cubic-step-damping).

**Revisit211:**The complete-controller prerequisite was supplied by a signed cubic secular solver, accepted/rejected trials, ARC ratio/sigma updates and the independently accepted186 geometry. This replaces the macro optimizer/GP while retaining the physical learner. All469 converge without errors, but energy and broad cost both fail. No successful-path model state identifies which controller/information loss is responsible. Revisit now requires an independently established model/globalization coupling or informative rejection diagnostics; no sigma/threshold/cap sweep.
- Cycle211; candidate `4c18129ba36b9503939ee87b0aaf28ad1b5f2a93`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision invalid.
  [Implementation](ideas/secant-cubic-damping/4c18129ba36b9503939ee87b0aaf28ad1b5f2a93/algo.py), [train](evaluation_results/cycle-211-train.json), [narrative](full_log.md#cycle-211-adaptive-cubic-macro-optimizer).

## cartesian-force-fit: Physical-block fitting in atomic-force units
Status: deferred

**Hypothesis:** Pulling the existing internal secant residual through the Cartesian Jacobian transpose improves physical curvature scale calibration.
**Outcome and uncertainty:** Cycle 62 is fully converged, energy-valid and error-free but loses training cost; 121 faster, 128 slower, 220 unchanged. No failure traces isolate a repair.
**Reason to revisit:** Independently improved transported observations or a richer physical model might benefit from this force metric. The current data still inherits native connection error and may overweight stiff components.
**Next experiment:** Revisit only with a justified observation/model improvement; no loss-interpolation, ridge or coefficient-bound sweep.

**Attempts:**
- Cycle 62; candidate `8e6953dd7120fed1a1f7735fab8b83c4c611a4fb`; champion `f56535e29a1c699bb1a1ea59ae6f57f047fda88e`; decision discard.
  [Implementation](ideas/cartesian-force-fit/8e6953dd7120fed1a1f7735fab8b83c4c611a4fb/algo.py).
  [Training evidence](evaluation_results/cycle-62-train.json).
  [Narrative](full_log.md#cycle-62-calibrate-physical-blocks-in-cartesian-force-units).

**Revisit226:**Independently accepted182/186 changed the nonlinear chart and corrected the connection;154/165/167/174/178 improved physical priors. These fulfill62's recorded observation/model prerequisite. Test exactly its original loss and guards against186, without changing ridge, clipping, schedule or replay.

**226 outcome/reassessment:**Full convergence and valid energy but cost loss; current connection and richer prior do not rescue the Cartesian norm. Revisit only with an independently supported response-error covariance; no loss-interpolation or fit-strength sweep.
- Cycle226; candidate `96f135aae6f0e97b5dad891a94f23044b1ed0a4a`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision discard.
  [Implementation](ideas/cartesian-force-fit/96f135aae6f0e97b5dad891a94f23044b1ed0a4a/algo.py), [train](evaluation_results/cycle-226-train.json), [narrative](full_log.md#cycle-226-cartesian-force-calibration-with-current-geodesics).

## physical-spectrum-metric: Reflect negative curvature in a physical metric
Status: incorporated

**Hypothesis:** An absolute Hessian formed by physical-metric congruence handles mixed stiff/soft negative modes better than Euclidean reflection.
**Outcome and uncertainty:** Cycle 64 passes energy and converges all cases without errors, but loses training cost. There are 46 faster, 42 slower and 381 unchanged cases. No failure traces identify a repair.
**Reason to revisit:** An independently improved physical metric or curvature uncertainty estimate may make the congruence useful; the current fixed primitive prior is only approximate.
**Next experiment:** No normalization, spectral floor or subgroup sweep. Revisit after a justified metric improvement and compare against the current native absolute spectrum.

**Attempts:**
- Cycle 64; candidate `39dcd45a2b6bab4f2bbb15e70db6386ff63899f5`; champion `f56535e29a1c699bb1a1ea59ae6f57f047fda88e`; decision discard.
  [Implementation](ideas/physical-spectrum-metric/39dcd45a2b6bab4f2bbb15e70db6386ff63899f5/algo.py).
  [Training evidence](evaluation_results/cycle-64-train.json).
  [Narrative](full_log.md#cycle-64-physical-metric-absolute-hessian-for-indefinite-steps).

**Revisit cycle234:**The accepted Badger/Schlegel primitive laws, linear-bend treatment and182 nonlinear chart improve the initial physical metric since64;186 improves transported curvature observations. Reuse64 exactly, changing no normalization, floor, fit or damping parameter. Another broad loss will require observed negative-mode/metric evidence before further revisits.

- Cycle234; candidate `73cadc1ae8fd47e796123cafe93956f0e2c67043`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision keep. Both splits improve cost with full convergence, valid energy and zero errors. Incorporated unchanged64 construction with the accepted primitive prior/chart. [train](evaluation_results/cycle-234-train.json), [valid](evaluation_results/cycle-234-valid.json), [narrative](full_log.md#cycle-234-physical-metric-reflection-with-the-accepted-primitive-prior).

**235 extension:**Replacing the primitive metric with the existing full fitted physical prior loses training cost despite full convergence and valid energy. Keep234 incorporated; defer this extension until independently observed metric/mode uncertainty can justify it, with no coefficient/mixing/floor sweep.
- Cycle235; candidate `0d9a50688deefb40ace7fc5e8db6b9ae68a17e94`; champion `73cadc1ae8fd47e796123cafe93956f0e2c67043`; decision discard. [Implementation](ideas/physical-spectrum-metric/0d9a50688deefb40ace7fc5e8db6b9ae68a17e94/algo.py), [train](evaluation_results/cycle-235-train.json), [narrative](full_log.md#cycle-235-calibrated-physical-prior-for-negative-curvature-reflection).

**453 fixed electronic extension:** Unlike 235, this uses only 450's newly accepted, fixed primitive electronic bond correlations; learned coefficients and auxiliary springs are excluded. All cases converge without errors and energy passes, but the cost gain is below the gate. The diagonal construction remains incorporated; defer the matrix extension pending independent negative-mode uncertainty evidence, without metric/blend/scope sweeps.
- Cycle 453; candidate `b79566d043ca6d5486ecc64d682cef0936d43607`; champion `19ef38c155eee70f9872b95841931f5a303e74c4`; decision discard. [Implementation](ideas/physical-spectrum-metric/b79566d043ca6d5486ecc64d682cef0936d43607/algo.py), [training](evaluation_results/cycle-453-train.json), [narrative](full_log.md#cycle-453-electronic-metric-for-negative-curvature).

## full-observation-gp: Compact full-gradient surrogate inference
Status: deferred

**Hypothesis:** Retaining the complete whitened historical gradient-residual span learns useful coupled motion omitted by a displacement-only surrogate.
**Outcome and uncertainty:** Cycle 65 is fully converged, energy-valid and error-free but loses cost; 37 faster, 37 slower, 395 unchanged. No failure diagnostics exist. Exact span arguments do not ensure physically accurate historical derivatives or a better prior.
**Reason to revisit:** Independently improved chart compatibility or a more accurate physical mean could make the retained derivative information useful. The compact span avoids a full ambient derivative Gram matrix.
**Next experiment:** Cycle 93 tested the changed physical-prior and early-activation regime and again lost cost. Defer until a specific derivative-chart or observational-consistency mechanism is supported; no rank/noise/length sweep.

**Attempts:**
- Cycle 65; candidate `1861d86408e2c0096eefc047d426da12e67983cd`; champion `f56535e29a1c699bb1a1ea59ae6f57f047fda88e`; decision discard.
  [Implementation](ideas/full-observation-gp/1861d86408e2c0096eefc047d426da12e67983cd/algo.py).
  [Training evidence](evaluation_results/cycle-65-train.json).
  [Narrative](full_log.md#cycle-65-full-observation-gp-in-its-derivative-span).

**Revisit now:** Cycles 74/75/82 improve the physical mean model; cycles 77/81 retain reversal observations and activate proposals during calibration. The original cycle-65 experiment used the cycle-41 prior and waited for six calibration pairs. Full derivative coverage could supply useful early directions in the current regime; historical-chart inconsistency remains unresolved and the earlier rejection remains evidence.

**Cycle 93 reassessment:** The new accepted physical mean and history policy still produce worse cost (61 faster, 72 slower), despite valid energy and full convergence. A broader prior improvement alone is insufficient for this construction; further work needs a more specific model/observation diagnosis.

- Cycle 93; candidate `86d3aa7e48f1b985d8b451c90873db8707558e47`; champion `71bc71cd0f3fa378394283b4eb42c5596eceece4`; decision discard.
  [Implementation](ideas/full-observation-gp/86d3aa7e48f1b985d8b451c90873db8707558e47/algo.py).
  [Training evidence](evaluation_results/cycle-93-train.json).
  [Narrative](full_log.md#cycle-93-full-gradient-gp-during-early-physical-calibration).

**Cycle 109 revisit:** Champion 106 independently validates tangent directional observations. Test a residual-enriched search span using old tangent projectors and a current positive physical metric, retaining the accepted reduced directional GP. This is approximate search enrichment, not the earlier exact full-isotropic posterior construction.

**Cycle 109 reassessment:** The independently accepted tangent maps do not rescue this residual-enriched span: fully converged, energy-valid training still loses cost, with no failure bundles. Defer until a diagnosed directional model deficiency motivates another construction; no rank/noise sweep.

- Cycle 109; candidate `386f5ced15062dd31c3b9bb180502bb374a29d52`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision discard.
  [Implementation](ideas/full-observation-gp/386f5ced15062dd31c3b9bb180502bb374a29d52/algo.py).
  [Training evidence](evaluation_results/cycle-109-train.json).
  [Narrative](full_log.md#cycle-109-tangent-aware-gradient-residual-search-directions).

**395 fixed-frame revisit:**A complete physical Cartesian GP optimizer puts every paid gradient in one exact affine frame, supplying the outstanding observation-consistency prerequisite. Retain compact full residual inference; remove native transport, quasi-Newton curvature andpaidprobes. Prior failures remain counterevidence, and the fixed physical mean may become stale.

**395 reassessment:**The complete fixed-frameCartesianGP loses broadly;181converged,288limits,zeroerrors,failedenergy and[0,468,1] faster/slower/unchanged. A retained trace has persistent nonzero forces and continuing relaxation. Exact observational consistency alone does not suffice; await independently improved changing nonlinear prior or persistent curvature evidence. No history,rank,noise,length,floor,radius orconfidence sweep.
- Cycle395; candidate `8d4a8ba80d681db19b7b67ec11e9ceb816e10195`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision invalid. [Implementation](ideas/full-observation-gp/8d4a8ba80d681db19b7b67ec11e9ceb816e10195/algo.py), [train](evaluation_results/cycle-395-train.json), [narrative](full_log.md#cycle-395-fixed-frame-full-gradient-cartesian-gp-optimizer), [passive bundle](diagnostics/bf305acfe0ab4eb389dd777bcc93c463/0102b7b1541ce0c12c095629ee703063570eec609b64e7e6bb7e6dbfa9de828b.json).

## least-change-curvature-blend: PSB/SR1 latest-secant update
Status: deferred

**Hypothesis:** A Frobenius least-change PSB component better preserves useful prior curvature when residual alignment is poor, while retaining the accepted SR1 weight.
**Outcome and uncertainty:** Cycle 66 passes energy with zero errors but loses cost broadly (43 faster, 392 slower, 34 unchanged); two force-call limits. Both passive traces show sustained late descent and substantial remaining motion. The converged subset also has large cost loss.
**Reason to revisit:** A separately justified least-change metric could change which prior components should be preserved; the tested Euclidean PSB does not provide an effective replacement for TS-BFGS.
**Next experiment:** No blend-coefficient or stopping/step threshold sweep. Revisit only after an independent metric improvement and compare its least-change rule under full gates.

**Attempts:**
- Cycle 66; candidate `1e69d7d282c6c4e47c2fa80dfdc5f8970c28787e`; champion `f56535e29a1c699bb1a1ea59ae6f57f047fda88e`; decision discard.
  [Implementation](ideas/least-change-curvature-blend/1e69d7d282c6c4e47c2fa80dfdc5f8970c28787e/algo.py).
  [Training evidence](evaluation_results/cycle-66-train.json).
  [Passive trace 1](diagnostics/8f26a17355824dbd805be9f68509629d/2debab1b176f3f33272dfad9bb23aa76b273e13d032307bf1422f363e6289bc7.json).
  [Passive trace 2](diagnostics/8f26a17355824dbd805be9f68509629d/21910cce10533bab03f8a33a928af85be1096b5a7a23f77b4fcb732f3b1034f7.json).
  [Narrative](full_log.md#cycle-66-least-change-psb-component-in-the-curvature-blend).

**Physical-metric revisit270:**Accepted234/244 supply a supported primitive stiffness metric absent in66. Replace Euclidean PSB by the least-change correction in the D-scaled Frobenius norm, retaining the original native SR1 weight and term. This tests a metric prerequisite, not a blend-weight change;187's complete TS/SR1 congruence is a different rule.

**Reassessment270:**The independent physical metric prerequisite was tested and failed cost substantially despite complete convergence and valid energy. No passive failure buffers exist. Defer further variants until observed model-error structure supports a different covariance; no metric-exponent, blend-weight or activation sweep.
- Cycle270; candidate `ba7ea2619bf97de35f7f881e72410ab7e4baba6d`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision discard.
  [Implementation](ideas/least-change-curvature-blend/ba7ea2619bf97de35f7f881e72410ab7e4baba6d/algo.py), [training](evaluation_results/cycle-270-train.json), [narrative](full_log.md#cycle-270-physical-least-change-component-in-secant-learning).

## rebuild-curvature-memory: Carry a centered learned quadratic across resets
Status: deferred

**Hypothesis:** Preserve learned curvature outside a near-linear coordinate change while fresh physical blocks initialize the rebuilt representation.
**Outcome and uncertainty:** Cycle 67 is energy-valid/error-free but loses cost (4 faster, 11 slower, 454 unchanged), with one force-call limit. The converged subset also loses. The passive trace shows substantial late descent and no exception; rebuild/matrix states are unavailable.
**Reason to revisit:** A principled identification of curvature that remains reliable through rank changes could improve transfer. The background-aware calibration already prevents the carried model being silently overwritten.
**Next experiment:** No amplitude or pseudoinverse-cutoff sweep. Revisit with an independently justified compatibility/uncertainty model and compare to native reset under full gates.

**Attempts:**
- Cycle 67; candidate `bc92aae36316986bb19f2ce27dc443b496075378`; champion `f56535e29a1c699bb1a1ea59ae6f57f047fda88e`; decision discard.
  [Implementation](ideas/rebuild-curvature-memory/bc92aae36316986bb19f2ce27dc443b496075378/algo.py).
  [Training evidence](evaluation_results/cycle-67-train.json).
  [Passive trace](diagnostics/4c446147617b48c8bd237ffc530d23f4/dd96173173ae8d92dbc592a45c8b2446039656a8e7efa1a6569d2fdf718bb2c1.json).
  [Narrative](full_log.md#cycle-67-preserve-learned-curvature-across-coordinate-rebuilds).

## fragment-size-bounds: Physical scaling of rotational step limits
Status: deferred

**Hypothesis:** Reference fragment gyration lengths balance rotation limits with translation and reduce poorly modeled collective moves.
**Outcome and uncertainty:** Cycle 68 improves aggregate training cost but fails energy despite full convergence and zero errors. Most total speed gain comes from one changed basin; several complexes also lose recovery. There are no retained successful trajectories to identify a bound-specific failure.
**Reason to revisit:** A physically consistent collective model or additional passive path evidence could distinguish productive rotation scaling from altered basin selection. Size-factor tuning alone is unsupported.
**Next experiment:** No immediate repair. Revisit only with an independently justified collective metric/model coupling, retaining full gates and avoiding molecule or size-regime exceptions inferred from outcomes.

**Attempts:**
- Cycle 68; candidate `05a32c9f8bcc4e19692d37ace32d69779ae9ee65`; champion `f56535e29a1c699bb1a1ea59ae6f57f047fda88e`; decision invalid.
  [Implementation](ideas/fragment-size-bounds/05a32c9f8bcc4e19692d37ace32d69779ae9ee65/algo.py).
  [Training evidence](evaluation_results/cycle-68-train.json).
  [Narrative](full_log.md#cycle-68-fragment-size-scaled-rotational-step-bounds).

Cycle259 pending: apply reference gyration scaling to the full rotational coordinate chart, Jacobian, Hessians and primitive force constants, while preserving physical bounds and GP admission norms. This supplies the declared collective metric/model coupling prerequisite absent in68; radius is in native Angstrom units, with no fitted scale factor.

Cycle259 outcome: `8e6fef4808d34bb5fd8482e95cff818e20c9112f`, [implementation](ideas/fragment-size-bounds/8e6fef4808d34bb5fd8482e95cff818e20c9112f/algo.py), [train](evaluation_results/cycle-259-train.json), [narrative](full_log.md#cycle-259-radius-of-gyration-rotational-coordinate-chart). Cost 0.7592285794717533, energy 1.0023058697989555, all469 converged and zero errors; [77, 99, 293] faster/slower/unchanged. Discard. Consistent source-supported collective scaling meets the earlier prerequisite but loses cost. Require independently supported nonlinear rotation geometry or observed model discrepancy; no radius/unit/tolerance sweeps.

## morse-bond-coordinates: Anharmonic stretching representation
Status: deferred

**Hypothesis:** Exponential bond coordinates absorb stretching anharmonicity throughout ordinary and surrogate optimization, reducing curvature-learning work.
**Outcome and uncertainty:** Cycle 69 has complete convergence, valid energy and zero errors, but slightly worse aggregate cost. Gains/losses are balanced; no retained failure trace identifies a repair. Initial Jacobian and physical prior match the champion, but the finite-step geometry and restriction meaning differ.
**Reason to revisit:** The exact-derivative adapter provides a stable tested implementation for a future independently supported anharmonic metric or model coupling. The failed GP-mean-only cycle 50 does not test this representation, and neither failure establishes a universal result about anharmonic coordinates.
**Next experiment:** No immediate repair or exponent sweep. Revisit with evidence that a specific physical step metric or improved bond model resolves the representation's measured cost tradeoff; compare against the current champion under both gates.

**Attempts:**
- Cycle 69; candidate `f2d90136b3d92f5eba635f30b500418bf7a87c24`; champion `f56535e29a1c699bb1a1ea59ae6f57f047fda88e`; decision discard.
  [Implementation](ideas/morse-bond-coordinates/f2d90136b3d92f5eba635f30b500418bf7a87c24/algo.py).
  [Training evidence](evaluation_results/cycle-69-train.json).
  [Narrative](full_log.md#cycle-69-morse-bond-coordinates-throughout-the-optimizer).

## exact-gp-microcurvature: Analytic posterior Hessian in microiterations
Status: deferred

**Hypothesis:** Exploit known GP curvature instead of relearning it with BFGS during the same twelve surrogate iterations.
**Outcome and uncertainty:** Cycle 70 passes validity with full convergence but improves training cost by only 0.00004841, short of the unchanged gate. Seven call counts change; no retained failure traces. Better model minimization has little demonstrated physical-call effect.
**Reason to revisit:** The analytic Hessian is a usable component for a separately justified posterior-curvature or confidence-constrained solver. No evidence supports simply tuning the microiteration budget or solver trust parameters.
**Next experiment:** Cycle 89 tested the distinct local-curvature proposal and lost cost with full convergence. Defer until posterior-curvature quality or a different model connection is independently diagnosed; no eigenvalue-floor or step-cap sweep.

**Attempts:**
- Cycle 70; candidate `73f6297867a48eeb87847a79d29ff324b406434f`; champion `f56535e29a1c699bb1a1ea59ae6f57f047fda88e`; decision discard.
  [Implementation](ideas/exact-gp-microcurvature/73f6297867a48eeb87847a79d29ff324b406434f/algo.py).
  [Training evidence](evaluation_results/cycle-70-train.json).
  [Narrative](full_log.md#cycle-70-exact-hessian-gp-microiterations).

**Revisit now:** The distinct local-curvature proposal is derived under cycle 88. It targets reliance on a remote GP stationary point, while preserving the physical Hessian and rejecting proposals outside the original safeguards.

**Cycle 89 reassessment:** The local Newton proposal passes energy but worsens cost (76 faster, 113 slower), with no failed-path evidence. This does not support the tested use; the analytic Hessian remains available without a current further experiment.

- Cycle 89; candidate `09701e80e4b0a50c171bedc5346c887c1a5d79fd`; champion `71bc71cd0f3fa378394283b4eb42c5596eceece4`; decision discard.
  [Implementation](ideas/exact-gp-microcurvature/09701e80e4b0a50c171bedc5346c887c1a5d79fd/algo.py).
  [Training evidence](evaluation_results/cycle-89-train.json).
  [Narrative](full_log.md#cycle-89-local-posterior-curvature-gp-proposal).

**504 revisit:** Explicit posterior Hessian covariance now supplies an RMS curvature metric. This is a new uncertainty-aware local proposal, not89's floor or cap tuning; compare against463 under unchanged gates.

**504 outcome:** Explicit posterior second moment still loses cost with all469 converged,valid energy,zero errors;{'slow': 113, 'fast': 80, 'same': 276}. Empty diagnostics do not isolate damping from model error. Await independent curvature-calibration evidence; no uncertainty multiplier,floor,cap or blend sweeps.
- Cycle504; candidate `0d54f34ba672e41d8354e9a5cc3dd0ed569ea3a7`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/exact-gp-microcurvature/0d54f34ba672e41d8354e9a5cc3dd0ed569ea3a7/algo.py), [training](evaluation_results/cycle-504-train.json), [narrative](full_log.md#cycle-504-posterior-rms-curvature-proposal).

## nonmonotone-gp-history: Learn from energy-increasing steps
Status: incorporated

**Hypothesis:** Retain nearby paid-for observations across energy increases so the GP can learn an overshoot bracket rather than restarting.
**Outcome and uncertainty:** Cycle 71 yields broad cost gains (103 faster, 31 slower) but fails energy at R=0.99996844. All cases converge without errors; several changed basin endpoints dominate energy loss. No successful-run traces identify the model/path cause.
**Reason to revisit:** A separately supported history-transport or physical-step improvement could let retained nonmonotone information reduce cost while limiting basin changes. The current result supports retaining the idea but not accepting its energy deficit.
**Next experiment:** Cycle 77 passes both gates when combined with independently accepted physical-prior changes 74/75. Retain it in the champion; future history changes require a distinct hypothesis and full comparison. No length/energy-threshold sweep.

**Attempts:**
- Cycle 71; candidate `d760c91e83b1de26e15da1c51c54e0bed9e83e5d`; champion `f56535e29a1c699bb1a1ea59ae6f57f047fda88e`; decision invalid.
  [Implementation](ideas/nonmonotone-gp-history/d760c91e83b1de26e15da1c51c54e0bed9e83e5d/algo.py).
  [Training evidence](evaluation_results/cycle-71-train.json).
  [Narrative](full_log.md#cycle-71-retain-gp-observations-across-energy-increases).

Cycle 77 revisits the mechanism after independently accepted physical-prior changes in cycles 74/75. Their saved endpoints differ materially from the cycle-41 baseline, motivating an interaction test under the original gates; the cycle-71 failure remains evidence.

- Cycle 77; candidate `07a8c991a656290029ea2aee428d3b4f66890938`; champion `20a882f1b5fe2f948a28f155ea5e68dad9665efd`; decision keep. Both splits fully converge, pass energy and improve cost.
  Implementation retained in accepted Git history.
  [Training evidence](evaluation_results/cycle-77-train.json).
  [Validation evidence](evaluation_results/cycle-77-valid.json).
  [Narrative](full_log.md#cycle-77-retain-gp-history-with-the-improved-physical-prior).

## cartesian-rms-trust: Physical ellipsoidal step restriction
Status: deferred

**Hypothesis:** Restrict the native positive quadratic by predicted Cartesian RMS rather than mixed-unit primitive components.
**Outcome and uncertainty:** Cycle 72 has five geodesic integration exceptions and worse successful-subset mean cost, though 144 successful cases improve. Early and later passive traces show large localized motion; the latter includes a large energy/force spike. Failed conversion tangents are unobserved, and nonlinear realized motion can exceed the tangent bound.
**Reason to revisit:** A maximum-atom safeguard directly addresses motion concentration while preserving the new model direction. This may also reduce large excursions in successful cases; that is unproven and must pass all gates.
**Next experiment:** Repair 1 (cycle 73) eliminates all five exceptions and passes energy, but worsens cost. End this sequence without radius/cap tuning. Revisit only with an independently justified nonlinear realization or model coupling; the stability repair alone is insufficient.

**Attempts:**
- Cycle 72; candidate `334f9cb0e9273213b3f798717b696cbaecd62bbe`; champion `f56535e29a1c699bb1a1ea59ae6f57f047fda88e`; decision invalid.
  [Implementation](ideas/cartesian-rms-trust/334f9cb0e9273213b3f798717b696cbaecd62bbe/algo.py).
  [Training evidence](evaluation_results/cycle-72-train.json).
  [Narrative](full_log.md#cycle-72-cartesian-rms-ellipsoidal-trust-steps).

- Cycle 73; candidate `d0e1d73afd6274eb219bd67ae16abfbc14517b20`; champion `f56535e29a1c699bb1a1ea59ae6f57f047fda88e`; decision discard. Repair 1 adds the maximum-atom safeguard; full convergence and valid energy, but cost fails.
  [Implementation](ideas/cartesian-rms-trust/d0e1d73afd6274eb219bd67ae16abfbc14517b20/algo.py).
  [Training evidence](evaluation_results/cycle-73-train.json).
  [Narrative](full_log.md#cycle-73-maximum-atom-safeguard-for-physical-trust-steps).

## anchored-gp-uncertainty: Covariance of the anchored residual
Status: deferred

**Hypothesis:** Propagating covariance through the same origin-value/gradient subtraction used by the mean could improve confidence-based selection.
**Outcome and uncertainty:** Cycle 76 reproduces every train result row exactly, with complete convergence and valid energy. It fails the improvement gate; no successful-path surrogate diagnostics identify which guard is decisive.
**Reason to revisit:** The analytically consistent variance is a reusable component if an independently justified solver constrains uncertainty during its trajectory, where endpoint-only equivalence need not persist. It is not evidence for relaxing the current threshold.
**Next experiment:** Cycle 88 used this covariance during uncertainty-penalized microoptimization and failed. Defer further use until an independently justified model/geometry connection is available; no penalty, cap or tolerance sweep. Preserve the original full gates.

**Attempts:**
- Cycle 76; candidate `46c174da5910a16cab07f22b66c48da52061ac6d`; champion `20a882f1b5fe2f948a28f155ea5e68dad9665efd`; decision discard.
  [Implementation](ideas/anchored-gp-uncertainty/46c174da5910a16cab07f22b66c48da52061ac6d/algo.py).
  [Training evidence](evaluation_results/cycle-76-train.json).
  [Narrative](full_log.md#cycle-76-anchored-functional-gp-uncertainty).

**Revisit now:** The analytic uncertainty gradient and penalized objective m+2*sigma are derived under cycle 87. Unlike cycle 76, uncertainty now affects the search trajectory and not only endpoint filtering. The current mean and its affine-anchored covariance are used consistently.

**Cycle 88 reassessment:** One early SCF failure follows an O-Cl collision at 0.5603 Angstrom. Passive geometry identifies the collision but not the surrogate decision. The matched 468 successful cases also worsen cost (+0.01806) and have mean recovery below one. A localized collision repair alone does not address the broader contrary evidence, so no immediate repair is justified.

- Cycle 88; candidate `1b96f8a49e1534188b7f479925dbd862c23d4c84`; champion `71bc71cd0f3fa378394283b4eb42c5596eceece4`; decision invalid.
  [Implementation](ideas/anchored-gp-uncertainty/1b96f8a49e1534188b7f479925dbd862c23d4c84/algo.py).
  [Training evidence](evaluation_results/cycle-88-train.json).
  [Passive bundle](diagnostics/fa9088cd2ae64f6f934511d0ebc2e241/ad282655ad6d06b8f378c64b4c4dd8e329d405ec623dc84f8f4df62856066e8a.json).
  [Narrative](full_log.md#cycle-88-uncertainty-penalized-anchored-gp-search).

## inverse-gradient-gp: Direct stationary-location inference
Status: deferred

**Hypothesis:** Learn a nonlinear conservative inverse gradient map around the physical inverse quadratic and query zero gradient, using the forward GP as a proposal critic.
**Outcome and uncertainty:** Cycle 78 has valid energy and complete convergence, but 162 slower versus 56 faster cases and worse aggregate cost. There are no failure traces or inverse-model states to isolate the mechanism.
**Reason to revisit:** The inverse regression is a tested component for a future formulation that establishes a consistent gradient chart or diagnoses when inverse inference supplies information missed by forward minimization. Retaining a critic alone did not deliver efficiency.
**Next experiment:** Defer until such evidence supports a specific formulation; no immediate kernel/noise/initial-seed sweep. A changed algorithm must pass the ordinary full gates.

**Attempts:**
- Cycle 78; candidate `f9e6ea3a3b7f0fba6128cb9562780854fc8f3c5a`; champion `07a8c991a656290029ea2aee428d3b4f66890938`; decision discard.
  [Implementation](ideas/inverse-gradient-gp/f9e6ea3a3b7f0fba6128cb9562780854fc8f3c5a/algo.py).
  [Training evidence](evaluation_results/cycle-78-train.json).
  [Narrative](full_log.md#cycle-78-inverse-gradient-gp-proposal-with-forward-model-checks).

**442 revisit:**A common affine Cartesian chart gives exact gradients of one restricted physical energy at all selected paid points; a scalar Legendre GP uses both conjugate values and inverse gradients. This supplies78's chart prerequisite. Project its proposal into430's unchanged native forward critic/realization, with all original gates; no kernel/noise sweep.

**442 reassessment:**Common affine Cartesian data plus scalar Legendre observations still lose cost, with full convergence, valid energy and no errors. The chart prerequisite has now been tested. Further work requires independently supported inverse-branch or query/realization evidence; no kernel,noise,rank,metricfloor,projection oractivation sweep.
- Cycle442; candidate `f1e2adb8ee3db3b6a635cfb50998f1dba6d4e623`; champion `bb770d9c7c9e59d39007f71a106af77cd9eb73a1`; decision discard. [Implementation](ideas/inverse-gradient-gp/f1e2adb8ee3db3b6a635cfb50998f1dba6d4e623/algo.py), [train](evaluation_results/cycle-442-train.json), [narrative](full_log.md#cycle-442-cartesian-legendre-inverse-proposals).

## moving-primitive-prior: Forecast geometry-dependent stiffness changes
Status: deferred

**Hypothesis:** Add the incremental calibrated primitive-model change to the learned Hessian before its newest-secant update.
**Outcome and uncertainty:**80's ordinary-secant forecast loses training cost.168's structured endpoint correction passes training but misses validation cost; both splits fully converge without errors. Frozen projectors and approximate finite-path model differences remain limitations.
**Reason to revisit:** A more faithful known-model gradient difference or consistent changing native chart could improve the residual secant; current successful endpoints do not diagnose a repair.
**Next experiment:** Defer until independently supported model/chart consistency evidence exists. Do not sweep forecast amplitude or activation count.

**Attempts:**
- Cycle 80; candidate `a0cd93fa428eaaba87ef42b8ab934f44b894a692`; champion `07a8c991a656290029ea2aee428d3b4f66890938`; decision discard.
  [Implementation](ideas/moving-primitive-prior/a0cd93fa428eaaba87ef42b8ab934f44b894a692/algo.py).
  [Training evidence](evaluation_results/cycle-80-train.json).
  [Narrative](full_log.md#cycle-80-forecast-changes-in-calibrated-primitive-curvature).

**Structured revisit168:** Independently accepted Badger bonds and a primary structured-secant derivation supply a new mechanism: ordinary secants conflate model curvature variation with residual learning. Test a bond-only forecast plus the matching trapezoidal end-point target, retaining the initial six-step calibration and fixed projector. This is an approximate native adaptation; no claim of exact structured-BFGS convergence.

- Cycle168; candidate `44251160b0c0a2bad493dc0455f0e8e6a05bc049`; champion `02a03f929ee5c99ff8e2085c1a742484e750b42a`; decision non_generalizable.
  [Implementation](ideas/moving-primitive-prior/44251160b0c0a2bad493dc0455f0e8e6a05bc049/algo.py), [train](evaluation_results/cycle-168-train.json), [valid](evaluation_results/cycle-168-valid.json), [narrative](full_log.md#cycle-168-structured-bond-curvature-forecast-with-end-point-secants).
  Both splits fully converge, but the training gain does not extend to validation. The trapezoidal known-model approximation, frozen projector and native chart remain limitations; no failure trace supports a bounded repair. Defer without amplitude or activation tuning.

**Revisit275:**Use an actual integrated Badger potential with current-chart covariant Hessians and separately parallel-transported exact model gradients. This replaces168's trapezoidal gradient approximation and frozen known-model projector, preserving six-pair calibration timing. Native residual-Hessian identification remains approximate; full gates decide.

**275 outcome/reassessment:**Exact known-potential gradients, parallel transport and current covariant Hessian fulfill168's stated consistency prerequisite, but all469 converged runs still fail aggregate energy and lose cost. One changed endpoint dominates energy loss; empty failure_details provides no mechanism. Defer pending independent residual-model/transport evidence; no amplitude/timing/support sweep. The model derivative and transport components remain reusable.
- Cycle275; candidate `630dbea790aecd54cd13adddf70ac4c42a74b5b9`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision invalid. [Implementation](ideas/moving-primitive-prior/630dbea790aecd54cd13adddf70ac4c42a74b5b9/algo.py), [train](evaluation_results/cycle-275-train.json), [narrative](full_log.md#cycle-275-integrable-bond-model-with-covariant-residual-secants).

**328 prerequisite:**Independently accepted298/307 supports a different known scalar model: finite angular bending. Reuse275 exact covariant derivatives and separately transported model gradient, freezing angle references after the same six-pair fit and using initial angular stiffness as298. No bond amplitude/timing revision. This directly tests whether finite-model fidelity changes the earlier structured-secant loss.

**328 outcome/reassessment:**The independently supported angular potential also loses cost despite complete convergence and valid energy. Exact known-model derivatives alone do not establish a useful residual Hessian forecast. Defer until independent residual-transport evidence; no model amplitude/timing sweep.
- Cycle328; candidate `88c1698af03a4e6abfb260faa0291afe3819e452`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/moving-primitive-prior/88c1698af03a4e6abfb260faa0291afe3819e452/algo.py), [training](evaluation_results/cycle-328-train.json), [narrative](full_log.md#cycle-328-structured-bending-potential-secants).

## regularized-gp-length: Infer a local correlation length
Status: closed

**Hypothesis:** Penalized marginal likelihood can adapt the GP residual length to cached local observations and improve proposals.
**Outcome and uncertainty:** Cycle 83 worsens cost and fails energy, with 91 faster and 78 slower cases. All converge; the largest changed endpoint dominates energy loss. Length fits and successful trajectories are unobserved.
**Reason to revisit:** A separately justified observation-error or chart-consistency model could prevent likelihood from explaining transport/model defects through length. Current evidence does not diagnose that mechanism.
**Next experiment:** Defer until such a model or passive evidence supports a specific repair; no immediate prior, length-bound or noise sweep. Any future adaptation must pass the unchanged gates.

**Attempts:**
- Cycle 83; candidate `45910dae070ca9a514c62f64500fb6aa325a5865`; champion `71bc71cd0f3fa378394283b4eb42c5596eceece4`; decision invalid.
  [Implementation](ideas/regularized-gp-length/45910dae070ca9a514c62f64500fb6aa325a5865/algo.py).
  [Training evidence](evaluation_results/cycle-83-train.json).
  [Narrative](full_log.md#cycle-83-regularized-local-gp-correlation-length-inference).

**Cycle 129 reassessment:** Accepted tangent maps do not rescue inference: aggregate energy fails and cost worsens, despite full convergence and no errors. No passive fit evidence supports tuning. This interaction is closed pending a materially different observation-error model.

- Cycle 129; candidate `41421e97150d5ac31cbfdddd593762293f34cba0`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision invalid.
  [Implementation](ideas/regularized-gp-length/41421e97150d5ac31cbfdddd593762293f34cba0/algo.py).
  [Training evidence](evaluation_results/cycle-129-train.json).
  [Narrative](full_log.md#cycle-129-regularized-gp-length-inference-with-tangent-observations).

## quadratic-bayesian-residual: Finite local curvature inference
Status: deferred

**Hypothesis:** A finite symmetric quadratic residual can infer shared local curvature from energy and gradient history without spatial interpolation.
**Outcome and uncertainty:** Cycle 84 passes energy with full convergence, but worsens cost: 112 faster, 151 slower. No failure traces distinguish misspecification from redundant physical curvature learning.
**Reason to revisit:** The origin-anchored quadratic covariance supplies a reusable curvature component for a separately derived hierarchical model that also represents nonquadratic residuals; this could avoid forcing all history into a single quadratic.
**Next experiment:** Defer until the separation and prior are independently justified. Do not immediately sweep amplitude or noise, or add kernels solely to rescue this result. Compare any new model against the current champion under both gates.

**Attempts:**
- Cycle 84; candidate `3a92093fe518e5e20e46ad908d8ce7eeb4147504`; champion `71bc71cd0f3fa378394283b4eb42c5596eceece4`; decision discard.
  [Implementation](ideas/quadratic-bayesian-residual/3a92093fe518e5e20e46ad908d8ce7eeb4147504/algo.py).
  [Training evidence](evaluation_results/cycle-84-train.json).
  [Narrative](full_log.md#cycle-84-finite-quadratic-bayesian-residual).

## constrained-gp-search: Optimize the surrogate inside physical limits
Status: deferred

**Hypothesis:** A constrained nonlinear search can find feasible boundary proposals missed by filtering unconstrained microiterations.
**Outcome and uncertainty:** Cycle 85 passes energy and fully converges but worsens cost, with 99 faster and 99 slower cases. Successful-path model and solver states are unavailable.
**Reason to revisit:** This analytic-constraint implementation is reusable if a separately supported uncertainty-aware solver or model identifies why physical feasibility alone is insufficient. No such diagnosis is currently available.
**Next experiment:** Defer until that model or passive evidence justifies a specific formulation. No immediate solver-budget, tolerance or margin sweep; all future changes require both unchanged gates.

**Attempts:**
- Cycle 85; candidate `84874b08f571f551e9f41fffda0799bb074c58c2`; champion `71bc71cd0f3fa378394283b4eb42c5596eceece4`; decision discard.
  [Implementation](ideas/constrained-gp-search/84874b08f571f551e9f41fffda0799bb074c58c2/algo.py).
  [Training evidence](evaluation_results/cycle-85-train.json).
  [Narrative](full_log.md#cycle-85-constrained-nonlinear-gp-microoptimization).

**Revisit289:**The uncertainty-aware solver prerequisite was supplied by the analytic original relative-confidence constraint, together with85's physical constraints and unchanged twelve-iteration search. It fully converges and passes energy but loses cost; no passive failure supports a repair. Revisit now needs independently calibrated predictive uncertainty or successful-path evidence identifying a specific constraint/search defect, not margin/threshold/budget tuning.
- Cycle289; candidate `c514c8d4d47d6a31e801930c097446dcee7604d0`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision discard.
  [Implementation](ideas/constrained-gp-search/c514c8d4d47d6a31e801930c097446dcee7604d0/algo.py), [train](evaluation_results/cycle-289-train.json), [narrative](full_log.md#cycle-289-confidence-constrained-gp-microsearch).

## stretch-stretch-calibration: Learn adjacent bond correlations
Status: deferred

**Hypothesis:** A bounded signed feature between constituent bonds of valence angles can improve the early physical Hessian fit.
**Outcome and uncertainty:** Cycle 86 fails energy and slightly worsens cost, with 34 faster and 42 slower cases. All converge; the largest endpoint energy loss is 6.6143 kJ/mol. There are no successful-path or fitted-coefficient diagnostics.
**Reason to revisit:** A separately validated calibration or model-consistency improvement could change the interaction between this correlated feature and existing across-angle springs. The normalized feature remains a bounded reusable construction.
**Next experiment:** Defer pending such independent evidence; no immediate coupling-amplitude or ridge sweep. Any later combination must be compared afresh under both original gates.

**Attempts:**
- Cycle 86; candidate `1c4b6aa305bd56f4526a795b2b4d991b80988a59`; champion `71bc71cd0f3fa378394283b4eb42c5596eceece4`; decision invalid.
  [Implementation](ideas/stretch-stretch-calibration/1c4b6aa305bd56f4526a795b2b4d991b80988a59/algo.py).
  [Training evidence](evaluation_results/cycle-86-train.json).
  [Narrative](full_log.md#cycle-86-bounded-stretch-stretch-calibration).

**Revisit282:**182's accepted Badger bond chart and186's current path connection supply the independently validated model/observation change requested by86.154/165/167/174 also revise the physical features. Test exactly86's normalized feature, coefficient bounds, zero prior and ridge on244, without changing any coupling strength or calibration count. This is an interaction control, not a diagnosis of86's endpoint energy loss.

**Cycle282 outcome:**The unchanged86 feature on accepted182/186 and revised physical priors passes energy and fully converges but loses cost. The previous consistency prerequisite is now tested; defer until new interaction-model evidence identifies how to separate adjacent-bond coupling from auxiliary-spring curvature, without coefficient/ridge/count sweeps.
- Cycle282; candidate `e5648f3d7746f687af21596c71e514f067f175ac`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision discard.
  [Implementation](ideas/stretch-stretch-calibration/e5648f3d7746f687af21596c71e514f067f175ac/algo.py), [train](evaluation_results/cycle-282-train.json), [narrative](full_log.md#cycle-282-stretch-stretch-calibration-in-the-accepted-badger-chart).

## cumulative-physical-calibration: Infer the starting model from full history
Status: deferred

**Hypothesis:** Later directions can improve bounded physical coefficients if the complete secant history is replayed from the refitted starting model.
**Outcome and uncertainty:**87 narrowly loses training cost with older physical features.169 with independently improved features lowers cost on both splits but fails validation energy at converged endpoints. No successful-path fit states or localized errors identify a repair.
**Reason to revisit:** The all-history policy and chronological replay have a real cost signal on the new priors. A separately accepted model or path-consistency treatment could justify a future combination, but endpoint changes alone cannot identify it.
**Next experiment:** Defer; keep the original all-history policy and coefficient bounds if revisiting. Do not sweep calibration counts, windows or forgetting factors.

**Attempts:**
- Cycle 87; candidate `6992ed17eef099c64a15d645c80cfdf479ed8788`; champion `71bc71cd0f3fa378394283b4eb42c5596eceece4`; decision discard.
  [Implementation](ideas/cumulative-physical-calibration/6992ed17eef099c64a15d645c80cfdf479ed8788/algo.py).
  [Training evidence](evaluation_results/cycle-87-train.json).
  [Narrative](full_log.md#cycle-87-cumulative-physical-calibration-with-full-replay).

**Revisit169:** Accepted154/165/167 independently changed bond, angle and torsion feature laws. Test the unchanged87 all-history policy on these new features as a single interaction control. No coefficient bounds, ridge, windows or forgetting factors are changed; existing chart uncertainty remains.

- Cycle169; candidate `e7d793d0e975b1201bc2f5e791db877ffb44d9d0`; champion `02a03f929ee5c99ff8e2085c1a742484e750b42a`; decision non_generalizable.
  [Implementation](ideas/cumulative-physical-calibration/e7d793d0e975b1201bc2f5e791db877ffb44d9d0/algo.py), [train](evaluation_results/cycle-169-train.json), [valid](evaluation_results/cycle-169-valid.json), [narrative](full_log.md#cycle-169-full-history-calibration-with-accepted-physical-priors).
  New physical features produce cost gains on both splits, but validation energy fails at converged endpoints. A general independently accepted method for better basin/model consistency could justify a future combination; no localized error or successful-path diagnostics justify a repair now. Keep the original bounds and all-history policy intact when revisiting, rather than tuning counts.

**Revisit196:**Independently accepted182/186 supplies a changed chart and path connection: inverse-cubic bond stiffness is constant in native units, and secants follow the current connection. Repeat the unchanged all-history policy from169; no bounds, ridge or memory settings change. Predeclared before195 completed.

**Reassessment196:**The accepted Badger/current-connection interaction passes training and validation energy, but loses validation cost. All934 converge without errors. This resolves169's observed energy failure without producing a keep. No count/window/bounds sweep; require a materially new model/observation-consistency derivation before revisiting again.
- Cycle196; candidate `01eab33491e4e274406992892cab232147483370`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision non_generalizable.
  [Implementation](ideas/cumulative-physical-calibration/01eab33491e4e274406992892cab232147483370/algo.py), [train](evaluation_results/cycle-196-train.json), [valid](evaluation_results/cycle-196-valid.json), [narrative](full_log.md#cycle-196-full-history-calibration-in-the-accepted-badger-metric).

## directional-gp-search: Surrogate step length on the physical direction
Status: deferred

**Hypothesis:** Keep the physical quasi-Newton direction and use nonlinear GP information only to choose a feasible step length without extra physical trials.
**Outcome and uncertainty:** Cycle 90 passes energy and fully converges but worsens cost, with 84 faster and 162 slower cases. No scalar-fit or failure traces diagnose the mechanism.
**Reason to revisit:** A separately derived path-consistent one-dimensional energy/gradient model could improve the relation between the surrogate ray and the realized geodesic. The current ray inherits the existing approximate chart.
**Next experiment:** Defer until that model is supported; no scalar interval or iteration-budget sweep. Any future directional proposal must pass both unchanged gates.

**Attempts:**
- Cycle 90; candidate `bc45f1df5362667868e5fae5c77434719446b5c0`; champion `71bc71cd0f3fa378394283b4eb42c5596eceece4`; decision discard.
  [Implementation](ideas/directional-gp-search/bc45f1df5362667868e5fae5c77434719446b5c0/algo.py).
  [Training evidence](evaluation_results/cycle-90-train.json).
  [Narrative](full_log.md#cycle-90-gp-line-search-along-the-ordinary-direction).

## energy-interpolated-trust: Infer the correction radius after uphill motion
Status: deferred

**Hypothesis:** A quadratic through the last physical energy pair and initial slope can estimate the correction distance from the retained uphill endpoint.
**Outcome and uncertainty:** Cycle 91 fully converges and passes energy but worsens cost, with 51 faster and 92 slower cases. No successful-path slope or radius state diagnoses the reason.
**Reason to revisit:** A separately derived path-aware globalization method could use the scalar interpolation in a way consistent with the next correction direction. The present radius-only use is unproven and unsuccessful.
**Next experiment:** Defer until that connection is supported; no factor or agreement-trigger sweep. Any future controller must pass the original full gates.

**Attempts:**
- Cycle 91; candidate `e5c3bba5e85bb47a9d650de3edda61d1a5083bb2`; champion `71bc71cd0f3fa378394283b4eb42c5596eceece4`; decision discard.
  [Implementation](ideas/energy-interpolated-trust/e5c3bba5e85bb47a9d650de3edda61d1a5083bb2/algo.py).
  [Training evidence](evaluation_results/cycle-91-train.json).
  [Narrative](full_log.md#cycle-91-energy-interpolated-uphill-trust-contraction).

## normalized-volume-umbrella: Isolate one pyramidalization curvature mode
Status: deferred

**Hypothesis:** A normalized-volume gradient per trigonal center can represent umbrella stiffness more directly than correlated improper gradients, at matched per-center native-space trace.
**Outcome and uncertainty:** Cycle 92 fully converges and passes energy but slightly worsens cost (21 faster, 37 slower). No passive failure or model-state evidence explains the difference.
**Reason to revisit:** The smooth permutation-invariant gradient is a reusable coordinate component if a separately derived in-plane/out-of-plane decomposition or moving model requires a single pyramidalization coordinate.
**Next experiment:** Defer pending that independent use; no immediate amplitude or rank sweep. Compare any new representation under both unchanged gates.

**Attempts:**
- Cycle 92; candidate `627d25d2d02330bf9b53a31aba3eef1b045b4ba7`; champion `71bc71cd0f3fa378394283b4eb42c5596eceece4`; decision discard.
  [Implementation](ideas/normalized-volume-umbrella/627d25d2d02330bf9b53a31aba3eef1b045b4ba7/algo.py).
  [Training evidence](evaluation_results/cycle-92-train.json).
  [Narrative](full_log.md#cycle-92-trace-matched-normalized-volume-umbrella-mode).

## compliance-native-prior: Project compliance before native inversion
Status: deferred

**Hypothesis:** An inverse projected-compliance prior can represent relaxation among redundant primitives, with positive class features and bounded signed coupling.
**Outcome and uncertainty:** Cycle 94 fully converges but fails energy and broadly loses cost (46 faster, 340 slower). Empirical stiffness reciprocals are not measured physical compliances; no model-state trace isolates the cause.
**Reason to revisit:** A separately derived physically calibrated compliance model could supply appropriate relaxed constants instead of reciprocating the existing stiffness guess. The factorization is reusable for such a model.
**Next experiment:** Defer pending that derivation; no interpolation or scale sweep from these endpoints. Any revised prior must pass both unchanged gates.

**Attempts:**
- Cycle 94; candidate `fd09ca0f8f88bbf7c48a2c3d343ba38efed8538e`; champion `71bc71cd0f3fa378394283b4eb42c5596eceece4`; decision invalid.
  [Implementation](ideas/compliance-native-prior/fd09ca0f8f88bbf7c48a2c3d343ba38efed8538e/algo.py).
  [Training evidence](evaluation_results/cycle-94-train.json).
  [Narrative](full_log.md#cycle-94-compliance-based-native-physical-prior).

## exactly-conditioned-gp: Fit a residual with exact current anchors
Status: deferred

**Hypothesis:** Condition the residual covariance on zero current value/gradient before fitting history, so mean and uncertainty share the same exact anchors.
**Outcome and uncertainty:** Cycle 95 narrowly passes train but loses validation cost; both splits fully converge and pass energy. Only 13 training call counts change. No successful model-state diagnostics identify a repair.
**Reason to revisit:** Exact conditioning is a reusable algebraic component for a separately justified residual model, such as separating fixed physical curvature from higher-order corrections.
**Next experiment:** Test that distinct curvature-preserving model under full gates; do not tune this variant's noise or confidence.

**Attempts:**
- Cycle 95; candidate `340cc43b37247a4f6243c6e16f5510dbbcdf15bb`; champion `71bc71cd0f3fa378394283b4eb42c5596eceece4`; decision non_generalizable.
  [Implementation](ideas/exactly-conditioned-gp/340cc43b37247a4f6243c6e16f5510dbbcdf15bb/algo.py).
  [Training evidence](evaluation_results/cycle-95-train.json).
  [Validation evidence](evaluation_results/cycle-95-valid.json).
  [Narrative](full_log.md#cycle-95-exactly-conditioned-current-point-gp-residual).

## curvature-preserving-gp: Learn only higher-order residual structure
Status: deferred

**Hypothesis:** Fix the estimated physical current Hessian and use a conditioned GP for cubic-and-higher corrections, separating the roles of secant and nonlinear models.
**Outcome and uncertainty:** Cycle 96 fully converges and passes energy but loses cost; 49 faster, 241 slower. The Hessian was an estimate, so fixing it may remove useful corrections, but model states are unobserved.
**Reason to revisit:** Independently reliable curvature observations or a quantified Hessian uncertainty could support a principled residual separation. The positive feature-removal construction can encode exact constraints when justified.
**Next experiment:** Defer until such information is available; no derivative-degree or constraint-strength sweep. Full original gates remain required.

**Attempts:**
- Cycle 96; candidate `19c32bb967b308c5d74e77caf5904fa934ffc987`; champion `71bc71cd0f3fa378394283b4eb42c5596eceece4`; decision discard.
  [Implementation](ideas/curvature-preserving-gp/19c32bb967b308c5d74e77caf5904fa934ffc987/algo.py).
  [Training evidence](evaluation_results/cycle-96-train.json).
  [Narrative](full_log.md#cycle-96-curvature-preserving-nonlinear-gp-residual).

## physical-metric-calibration: Balance secant fit errors by starting curvature
Status: deferred

**Hypothesis:** The inverse starting-Hessian norm can balance stiff and soft response errors during the accepted physical-block fit.
**Outcome and uncertainty:** Cycle 97 fully converges and passes energy but worsens cost; 133 faster, 128 slower. Successful fit/fallback states are unavailable.
**Reason to revisit:** Independently supported response-error covariance could justify a physical loss metric more directly than the empirical prior. The fixed-map implementation provides a controlled component.
**Next experiment:** Defer pending that observation model; no metric interpolation, class restriction or floor sweep. Full gates are unchanged.

**Attempts:**
- Cycle 97; candidate `6091ca071d570c43ac2b5f2dae139b3b0d304bc6`; champion `71bc71cd0f3fa378394283b4eb42c5596eceece4`; decision discard.
  [Implementation](ideas/physical-metric-calibration/6091ca071d570c43ac2b5f2dae139b3b0d304bc6/algo.py).
  [Training evidence](evaluation_results/cycle-97-train.json).
  [Narrative](full_log.md#cycle-97-physical-metric-calibration-loss).

## bend-bend-calibration: Learn common-vertex angle correlations
Status: deferred

**Hypothesis:** A bounded signed block can represent coupled bending absent from independently scaled native angles.
**Outcome and uncertainty:** Cycle 98 passes energy and fully converges, but slightly loses training cost (10 faster, 20 slower, 439 unchanged). No failure/model-state evidence identifies a repair.
**Reason to revisit:** An independently derived separation of native bending from across-angle distance curvature could make the feature identifiable; its spectral bound is reusable.
**Next experiment:** Defer pending that separation or passive evidence. No amplitude or ridge sweep; retain full gates.

**Attempts:**
- Cycle 98; candidate `5ac8c52907caade09855d01a8c2fade2b7ce4fc2`; champion `71bc71cd0f3fa378394283b4eb42c5596eceece4`; decision discard.
  [Implementation](ideas/bend-bend-calibration/5ac8c52907caade09855d01a8c2fade2b7ce4fc2/algo.py).
  [Training evidence](evaluation_results/cycle-98-train.json).
  [Narrative](full_log.md#cycle-98-bounded-bend-bend-calibration).

## scale-mixture-gp: Integrate residual correlation lengths
Status: deferred

**Hypothesis:** A rational-quadratic residual averages spatial scales while preserving local value/gradient variances.
**Outcome and uncertainty:** Cycle 99 slightly improves training cost but fails energy with full convergence; 100 faster, 111 slower. The largest energy losses also save calls. No trajectory bundles are available, so basin-change causation remains uncertain.
**Reason to revisit:** A separately justified uncertainty/trajectory model might retain mixed scales while detecting harmful extrapolation. The derivative implementation provides a controlled component.
**Next experiment:** Defer until such a model or passive diagnostics exists; no immediate shape, length or confidence sweep.

**Attempts:**
- Cycle 99; candidate `8cfd9cea6aa3e7dfe61f681ad64cd7cec9d79041`; champion `71bc71cd0f3fa378394283b4eb42c5596eceece4`; decision invalid.
  [Implementation](ideas/scale-mixture-gp/8cfd9cea6aa3e7dfe61f681ad64cd7cec9d79041/algo.py).
  [Training evidence](evaluation_results/cycle-99-train.json).
  [Narrative](full_log.md#cycle-99-scale-mixture-gp-residual).

**Cycle180 audit/reassessment:** The later unit-shape mixture repeated99's kernel on178, which additionally contains tangent derivative maps, collision protection and revised physical priors. I overlooked99 during selection, so180 should not be presented as a new kernel family or a diagnosed repair. It now passes energy but loses training cost (83 faster,112 slower,274 unchanged), leaving no support for scale-mixture adoption or parameter tuning. The earlier source copy under scale-mixture-surrogate is retained intact; this canonical entry groups both attempts.
- Cycle180; candidate `0ea1b6c16de6b481322260c7fd1e0d142ee2315c`; champion `4807cb0fb347c39d9c849ada2e3f404d9e557930`; decision discard.
  [Implementation](ideas/scale-mixture-gp/0ea1b6c16de6b481322260c7fd1e0d142ee2315c/algo.py).
  [Training evidence](evaluation_results/cycle-180-train.json).
  [Narrative](full_log.md#cycle-180-exponential-scale-mixture-surrogate-covariance).

**349 interaction revisit:**Apply the unchanged99/180 shape1/beta0.25 kernel to the independently accepted307 bending-energy feature map and298 finite mean, with182/186 trajectory changes retained. This is one prespecified representation interaction, not a new kernel or parameter sweep. The prior failures remain counterevidence; broad loss ends this revisit.

**349 reassessment:**The fixed mixture composed with307's accepted bending features and298 mean still loses cost with full convergence, valid energy and no errors. The prespecified interaction revisit is exhausted; await independent uncertainty evidence without shape/length/noise/alternate-kernel sweeps.
- Cycle349; candidate `7aa366da6a12133d2e80f24f2ceb5e3fcdf607e2`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/scale-mixture-gp/7aa366da6a12133d2e80f24f2ceb5e3fcdf607e2/algo.py), [training](evaluation_results/cycle-349-train.json), [narrative](full_log.md#cycle-349-scale-mixture-in-bending-energy-coordinates).

## additive-mode-gp: Separate nonlinear corrections by physical mode
Status: deferred

**Hypothesis:** After the physical quadratic, independent nonlinear mode corrections may use a short gradient history more efficiently.
**Outcome and uncertainty:** Cycle 100 loses energy and broadly worsens cost (39 faster, 368 slower), with full convergence and no failure traces. Eigenmodes may not separate nonlinear couplings.
**Reason to revisit:** A stable, physically derived nonlinear-mode decomposition could supply an appropriate additive basis. The consistent derivative covariance is reusable for that representation.
**Next experiment:** Defer until such a basis is derived; no immediate interaction-order, amplitude or length sweep.

**Attempts:**
- Cycle 100; candidate `ba08e39e383ec501e0d1f3f69cf000adc26234d7`; champion `71bc71cd0f3fa378394283b4eb42c5596eceece4`; decision invalid.
  [Implementation](ideas/additive-mode-gp/ba08e39e383ec501e0d1f3f69cf000adc26234d7/algo.py).
  [Training evidence](evaluation_results/cycle-100-train.json).
  [Narrative](full_log.md#cycle-100-additive-spectral-mode-gp-residual).

**Revisit 147:** Replace independent eigenaxis additivity by structural primitive-kind participation with matched total value/gradient prior variance. This derives the requested physical grouping; all old cycle-100 failure evidence remains relevant.

**Reassessment 147:** Structural grouping with matched prior moments remains slower, despite complete convergence and valid energy. No immediate weights/length/interaction-order sweep. Revisit only if a more local physical interaction decomposition is independently supported; primitive kind alone does not provide a beneficial decomposition here.
- Cycle 147; candidate `1084c2d68c972cf7cd31ed87df0c0ee054edcbb6`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision discard.
  [Implementation](ideas/additive-mode-gp/1084c2d68c972cf7cd31ed87df0c0ee054edcbb6/algo.py).
  [Training evidence](evaluation_results/cycle-147-train.json).
  [Narrative](full_log.md#cycle-147-structural-group-additive-gp-with-matched-local-variance).

**Revisit277:**Use overlapping real-atom primitive supports and a separate collective group, motivated by additive local-energy models. Preserve147's value/gradient covariance moments and all other244 GP settings. This supplies the local-interaction decomposition requested after147, not another primitive-type weight.

**277 reassessment:**Overlapping atom-local nonlinear interactions with a separate collective component fully converge with valid energy but lose cost. This tests147's local-decomposition prerequisite. Missing nonlocal coupling, independent-site assumption and moment matching are unresolved; require independent residual-locality evidence before another decomposition. No support/weight/length sweep.
- Cycle277; candidate `94d48d14493114974f722a5a5dfb3dd3472e4364`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision discard. [Implementation](ideas/additive-mode-gp/94d48d14493114974f722a5a5dfb3dd3472e4364/algo.py), [train](evaluation_results/cycle-277-train.json), [narrative](full_log.md#cycle-277-atom-local-additive-surrogate-covariance).

**Revisit321:**An independently accepted finite bending coordinate and real Cartesian tangent partition provide an atomic-motion decomposition that retains local cross-kind interactions. Preserve matched value/gradient moments, all mean/search/gates; this is neither100's spectral axes nor147's global primitive kinds. Broad loss ends the decomposition without weights/cutoffs/length sweeps.

**Reassessment321:**The real atomic-motion partition plus accepted bending map still loses cost, with all469 converged and valid energy. Locality and omitted nonlocal response are unresolved. Await independently supported nonlinear local-environment features or a diagnosed covariance defect; no atomic weights, grouping, length or cutoff sweep.
- Cycle321; candidate `d9d3f77fdfc5606dca4360b3d90856ad2d695620`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard.
  [Implementation](ideas/additive-mode-gp/d9d3f77fdfc5606dca4360b3d90856ad2d695620/algo.py), [training](evaluation_results/cycle-321-train.json), [narrative](full_log.md#cycle-321-atomic-motion-additive-gp-covariance).

## exponential-gradient-flow: Exact integration of the local quadratic flow
Status: deferred

**Hypothesis:** Exact stiff-mode relaxation under the same trust bound may outperform backward-Euler spectral damping.
**Outcome and uncertainty:** Cycle 101 passes energy and fully converges without errors but loses training cost (86 faster, 119 slower). No solver failure or passive trajectory evidence supports a bounded repair.
**Reason to revisit:** A derived nonlinear remainder correction, using already observed gradients consistently, could improve the flow model beyond a frozen reflected quadratic. The spectral phi-function step is reusable.
**Next experiment:** Defer until that correction is justified; no time, radius or spectral-floor sweep.

**Attempts:**
- Cycle 101; candidate `8b7c6c1b32db761d514a2431f4649fb8e4dbf836`; champion `71bc71cd0f3fa378394283b4eb42c5596eceece4`; decision discard.
  [Implementation](ideas/exponential-gradient-flow/8b7c6c1b32db761d514a2431f4649fb8e4dbf836/algo.py).
  [Training evidence](evaluation_results/cycle-101-train.json).
  [Narrative](full_log.md#cycle-101-exponential-local-gradient-flow-steps).

**340 revisit:**The accepted GP307 supplies a complete paid-data nonlinear gradient remainder. Test a two-stage ETDRK2 proposal in its whitened chart, replacing GP micro-minimization while preserving the full model and admission gates. This directly tests101's nonlinear-remainder prerequisite, although its preconditioned reduced flow differs from101's full native quadratic flow.304's implicit discrete-gradient equation and solver are not reused.

**340 reassessment:**The complete nonlinear GP remainder and explicit ETDRK2 stage fully converge with valid energy and no errors, but lose cost. This tests the stated nonlinear-correction prerequisite. No saved stage/field path isolates horizon matching versus one-stage model error. Require independently supported finite-flow/model agreement before another attempt; no horizon, stage-count, weight or confidence sweeps.
- Cycle340; candidate `a8560cb2566dac8f33f970e46d36909f005a36c6`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/exponential-gradient-flow/a8560cb2566dac8f33f970e46d36909f005a36c6/algo.py), [training](evaluation_results/cycle-340-train.json), [narrative](full_log.md#cycle-340-exponential-nonlinear-gp-flow-proposal).

## range-projected-physical-prior: Remove redundant QR completion at initialization
Status: deferred

**Hypothesis:** Initial physical features restricted to the actual B range avoid arbitrary redundant responses during calibration.
**Outcome and uncertainty:** Cycle 102 passes energy but narrowly loses cost; 27 faster, 35 slower, 407 unchanged. No failure traces identify a repair.
**Reason to revisit:** A coherent moving-range curvature representation may require this initialization to match its transport convention. Initial projection alone is insufficient evidence of benefit.
**Next experiment:** Defer until that joint representation is derived; no rank threshold or gauge stiffness sweep.

**Attempts:**
- Cycle 102; candidate `4b7b34958954c1c0ad054c244ff7276805a611c5`; champion `71bc71cd0f3fa378394283b4eb42c5596eceece4`; decision discard.
  [Implementation](ideas/range-projected-physical-prior/4b7b34958954c1c0ad054c244ff7276805a611c5/algo.py).
  [Training evidence](evaluation_results/cycle-102-train.json).
  [Narrative](full_log.md#cycle-102-rank-revealing-initial-physical-projector).

## linear-bend-calibration: Share angle scaling with dummy linear bends
Status: incorporated

**Hypothesis:** Parent-dummy-centered impropers model linear bending and may share more useful calibration with angles than torsions.
**Outcome and uncertainty:**103's grouping-only variant loses cost.177 adds a derived isotropic starting model but fails from unhashable coordinate keys before forces.178 fixes only those keys and passes both full gates with no errors and full convergence.
**Reason to revisit:** The derived constrained prior supplied the missing connection between the physical bend stiffness and its dummy representation; this formulation is now incorporated.
**Next experiment:** Retain178's matched physical stiffness and shared angle coefficient as the champion component. Further changes need a distinct representation/model hypothesis rather than dummy-stiffness tuning.

**Attempts:**
- Cycle 103; candidate `f649b24d208c2ac89ab2632439d664c8a407feb8`; champion `71bc71cd0f3fa378394283b4eb42c5596eceece4`; decision discard.
  [Implementation](ideas/linear-bend-calibration/f649b24d208c2ac89ab2632439d664c8a407feb8/algo.py).
  [Training evidence](evaluation_results/cycle-103-train.json).
  [Narrative](full_log.md#cycle-103-calibrate-dummy-linear-bends-as-bending).

**Revisit177:** Derive a matching isotropic starting stiffness for the two dummy-represented physical bend directions, using the independently accepted Schlegel real-angle law, then share the angle-class fit. This supplies the constrained prior absent in103; nearly zero-angle constructions are excluded because they do not represent an opposing linear pair.

- Cycle177; candidate `fe3357128be9fcd80fdbf419c953951557e326f7`; champion `ece05361773e587139cedc78c9528bdd567d7775`; decision invalid.
  [Implementation](ideas/linear-bend-calibration/fe3357128be9fcd80fdbf419c953951557e326f7/algo.py), [train](evaluation_results/cycle-177-train.json), [trace](diagnostics/aa15f858d38d430bb073f29333a8da95/acbda262ac8ace24eccbbf15ff087355873ac3418a76cc850c2a90d392c5653a.json), [narrative](full_log.md#cycle-177-isotropic-physical-prior-for-dummy-linear-bends).
  Implementation error before forces: Dihedral is unhashable. Bounded repair1 will use immutable index keys, without changing the model.

- Cycle178; candidate `4807cb0fb347c39d9c849ada2e3f404d9e557930`; champion `ece05361773e587139cedc78c9528bdd567d7775`; decision keep, bounded repair1.
  [Train](evaluation_results/cycle-178-train.json), [valid](evaluation_results/cycle-178-valid.json), [narrative](full_log.md#cycle-178-immutable-keys-for-the-linear-bend-prior). Accepted code retained in Git. Fixing immutable keys removes all177 errors; matched physical starting stiffness plus angle-class calibration improves both splits. This incorporates the derived model, not103's grouping-only variant.

## one-sided-trust-feedback: Expand after unexpectedly good downhill steps
Status: deferred

**Hypothesis:** A model that underpredicts downhill progress need not prevent trust-radius growth during minimization.
**Outcome and uncertainty:** Cycle 104 passes train, loses validation cost, and passes energy/full convergence on both splits. Validation has 34 faster and 33 slower cases. No ratio/radius traces diagnose the loss.
**Reason to revisit:** A separately validated error model or step-path prediction could distinguish useful underprediction from misleading agreement and justify asymmetric feedback.
**Next experiment:** Defer until that model or passive evidence exists; no radius-factor or ratio-threshold sweep.

**Attempts:**
- Cycle 104; candidate `07e2cf4cbc0dceba39c07becf08dfd9db4ae6c60`; champion `71bc71cd0f3fa378394283b4eb42c5596eceece4`; decision non_generalizable.
  [Implementation](ideas/one-sided-trust-feedback/07e2cf4cbc0dceba39c07becf08dfd9db4ae6c60/algo.py).
  [Training evidence](evaluation_results/cycle-104-train.json).
  [Validation evidence](evaluation_results/cycle-104-valid.json).
  [Narrative](full_log.md#cycle-104-one-sided-downhill-trust-feedback).

## aligned-gp-extension: Extend aligned surrogate minima within current guards
Status: deferred

**Hypothesis:** Directional extension can turn repeated extrapolation into informative interpolation without violating current proposal bounds.
**Outcome and uncertainty:** Cycle 105 improves cost but fails energy; 25 faster and 16 slower. One higher-energy endpoint dominates energy loss and saves 38 calls. No trajectory bundles identify the causal extension or support a targeted repair.
**Reason to revisit:** Independently improved historical derivative consistency or model uncertainty could preserve the small acquisition benefit while changing the problematic path. This is an interaction hypothesis requiring fresh gates.
**Next experiment:** Cycle 107 tests the accepted tangent-observation combination and repeats the energy loss. Defer further extension work until new trajectory evidence or a separately supported acquisition model exists; no scale/cosine/confidence sweep.

**Attempts:**
- Cycle 105; candidate `072bcfc75ba2f5bcfafa7811639a1d0434ba97c3`; champion `71bc71cd0f3fa378394283b4eb42c5596eceece4`; decision invalid.
  [Implementation](ideas/aligned-gp-extension/072bcfc75ba2f5bcfafa7811639a1d0434ba97c3/algo.py).
  [Training evidence](evaluation_results/cycle-105-train.json).
  [Narrative](full_log.md#cycle-105-bounded-aligned-gp-step-extension).

**Reassessment:** Cycle 107 retains the cost signal but repeats the dominant endpoint energy loss after independently accepted derivative consistency. That specific combination fails; no passive trajectory evidence supports another immediate repair.
- Cycle 107; candidate `6e84cc96466c2008f042486e7656c80f842992de`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision invalid.
  [Implementation](ideas/aligned-gp-extension/6e84cc96466c2008f042486e7656c80f842992de/algo.py).
  [Training evidence](evaluation_results/cycle-107-train.json).
  [Narrative](full_log.md#cycle-107-aligned-extension-with-tangent-observations).

## internal-box-dogleg: Cauchy-to-Newton path under component bounds
Status: deferred

**Hypothesis:** A positive-quadratic dogleg can use limited internal motion more effectively than uniform spectral damping.
**Outcome and uncertainty:** Cycle 108 passes energy and fully converges but broadly worsens cost. No numerical failure or passive trace identifies a repair. Mixed internal units may make the Cauchy segment inefficient.
**Reason to revisit:** An independently justified metric for the Cauchy segment could resolve that mismatch. The exact affine weighted-box intersection remains reusable.
**Next experiment:** Defer until such a metric is independently supported; no arbitrary segment mixing or radius sweep.

**Attempts:**
- Cycle 108; candidate `9994056b9337ba5a293ddbfadb44819841c5633c`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision discard.
  [Implementation](ideas/internal-box-dogleg/9994056b9337ba5a293ddbfadb44819841c5633c/algo.py).
  [Training evidence](evaluation_results/cycle-108-train.json).
  [Narrative](full_log.md#cycle-108-reflected-hessian-dogleg-inside-the-internal-box).

**Cycle201 prerequisite:**Independently accepted physical primitive laws154/165/167/178 and Badger chart182 now motivate M=P*diag(D_initial)*P.T for the Cauchy direction. Same108 box/Newton construction and current186 champion; no radius or mixture tuning.

**Cycle201 outcome:**Physical Cauchy scaling still loses cost broadly, with full convergence and valid energy. No localized repair is supported. Defer until independent evidence identifies why box-constrained physical-model minimization predicts actual progress; no metric/segment tuning.
- Cycle201; candidate `a86f25ac7e7b995358d1f74a6edcc0fd83bd1a56`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision discard.
  [Implementation](ideas/internal-box-dogleg/a86f25ac7e7b995358d1f74a6edcc0fd83bd1a56/algo.py), [train](evaluation_results/cycle-201-train.json), [narrative](full_log.md#cycle-201-physical-metric-internal-dogleg).

## two-seed-gp-search: Search the surrogate from the current geometry too
Status: deferred

**Hypothesis:** A second local GP search from the origin can find a lower admissible endpoint missed by the ordinary-step seed.
**Outcome and uncertainty:** Cycle 110 is fully converged, energy-valid and error-free, but its cost gain falls below the unchanged training gate. Only seven call counts change and no failure traces exist.
**Reason to revisit:** A separately supported change to the surrogate may make its local minima more sensitive to the seed; this two-start selection is a retained control for that situation.
**Next experiment:** Defer until that model is justified; no seed-count, microiteration-budget or tolerance sweep to bridge the narrow gate miss.

**Attempts:**
- Cycle 110; candidate `7c760ba051bb2c9c532a129d983a4374934a5f85`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision discard.
  [Implementation](ideas/two-seed-gp-search/7c760ba051bb2c9c532a129d983a4374934a5f85/algo.py).
  [Training evidence](evaluation_results/cycle-110-train.json).
  [Narrative](full_log.md#cycle-110-two-seed-local-search-on-the-accepted-gp).

**Revisit300:**Accepted298 supplies the deferred independent nonlinear-mean change. Reuse110 exact seeds, solver settings and guards; adapt only per-seed angular-domain exceptions. No budget/count/seed sweep.

**Cycle300 reassessment:**The accepted298 mean meets110's independent-model prerequisite, but the exact two-start control now loses training cost with all469 converged, valid energy and no errors. Empty failure details supply no localized repair. Require direct basin-selection evidence before another search modification; no seed or budget sweep.
- Cycle300; candidate `b93c00225615884ad4abe08727215c99d811f557`; champion `7c0bbf8190d01c916f081326421250f6b036918c`; decision discard.
  [Implementation](ideas/two-seed-gp-search/b93c00225615884ad4abe08727215c99d811f557/algo.py).
  [Training evidence](evaluation_results/cycle-300-train.json).
  [Narrative](full_log.md#cycle-300-two-start-gp-search-with-the-accepted-bending-mean).

## internal-preconditioned-fire: Inertial motion with a learned curvature mass
Status: deferred

**Hypothesis:** Positive-metric FIRE momentum can reduce repeated steps while the physical Hessian removes stiff oscillations.
**Outcome and uncertainty:** Cycle 112 fails energy and loses cost broadly; 403 cases slow down, one reaches the call limit. Its passive trace shows sustained late descent and unconverged forces, but no mass/velocity state. The converged subset also regresses strongly.
**Reason to revisit:** A consistently transported inertial state or an independently supported mass/GP coupling could make the engine useful; simple current-tangent projection does not establish that connection.
**Next experiment:** Defer until that mechanism is derived or diagnosed. No dt/alpha/latency sweep from this broad negative result; this failure concerns the tested moving-metric hybrid, not every inertial method.

**Attempts:**
- Cycle 112; candidate `884f6b72a4ee8b057797772ce97eb9161e0220e7`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision invalid.
  [Implementation](ideas/internal-preconditioned-fire/884f6b72a4ee8b057797772ce97eb9161e0220e7/algo.py).
  [Training evidence](evaluation_results/cycle-112-train.json).
  [Passive trace](diagnostics/84b9e184016f48ce87d7a77f8a88cc2f/15192444eacab5f069a5c67e2ee201b8aba334e6ee62336ae07a07f14253e3e4.json).
  [Narrative](full_log.md#cycle-112-curvature-preconditioned-internal-inertial-motion).

**Revisit296:**Accepted186 now exposes the endpoint-parallel selected step, including realized truncation. Reuse112 settings but carry this actual velocity after the paid kick; this meets the consistent-transport prerequisite. Passive evidence does not establish transport as the old cost cause. No parameter sweep.

**296 reassessment:**The endpoint-parallel transport prerequisite is now implemented, but the engine still broadly loses cost and fails energy despite all469 converging with zero errors. Faster/slower/unchanged=[45, 362, 62]; failure_details is empty. Correct transport alone is insufficient for this learned-mass/GP coupling. Defer until an independently supported mass/dynamics/model coupling is available; no dt, alpha, latency or activation sweep.
- Cycle296; candidate `021cbae039eb70b9f0856393490c5af3f89fcaa0`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision invalid.
  [Implementation](ideas/internal-preconditioned-fire/021cbae039eb70b9f0856393490c5af3f89fcaa0/algo.py), [training](evaluation_results/cycle-296-train.json), [narrative](full_log.md#cycle-296-endpoint-parallel-momentum-for-internal-fire).

## directional-gp-noise: Correlated noise for mapped derivative observations
Status: deferred

**Hypothesis:** Isotropic embedded derivative discrepancy transforms through M_i into correlated directional noise, improving relative confidence across tangent changes.
**Outcome and uncertainty:** Cycle 113 passes training narrowly with four changed call counts, then fails validation cost. Both splits converge fully, with valid energy and no errors or traces. The assumed latent noise geometry is not independently measured.
**Reason to revisit:** A separately justified observation-error model could indicate where directional correlations matter; the preserved congruence provides a control for that model.
**Next experiment:** Defer until such evidence exists; no noise amplitude, conditioning or tangent-threshold sweep.

**Attempts:**
- Cycle 113; candidate `b7beb0ee04fa97e4516d6c6d70e0da7c2d3221b3`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision non_generalizable.
  [Implementation](ideas/directional-gp-noise/b7beb0ee04fa97e4516d6c6d70e0da7c2d3221b3/algo.py).
  [Training evidence](evaluation_results/cycle-113-train.json).
  [Validation evidence](evaluation_results/cycle-113-valid.json).
  [Narrative](full_log.md#cycle-113-transport-derivative-noise-with-the-observation-map).

## full-affine-gp-mean: Retain transverse current-gradient information
Status: deferred

**Hypothesis:** Historical directional derivatives leaving the search span should subtract the known ambient affine mean before fitting nonlinear residuals.
**Outcome and uncertainty:** Cycle 114 passes training but worsens validation cost. Both splits fully converge, with valid energy and no errors or failure traces. Formal mean consistency does not ensure improved physical interpolation.
**Reason to revisit:** A consistently derived and independently supported transverse-curvature model could make the known off-span mean more useful; this affine implementation isolates the first-order part.
**Next experiment:** Defer until that curvature/observation mechanism is supported. No strength or activation sweep from endpoints alone; future changes require both original gates.

**Attempts:**
- Cycle 114; candidate `b274b35cb47451416a3ac4cbd69be508cb994c6f`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision non_generalizable.
  [Implementation](ideas/full-affine-gp-mean/b274b35cb47451416a3ac4cbd69be508cb994c6f/algo.py).
  [Training evidence](evaluation_results/cycle-114-train.json).
  [Validation evidence](evaluation_results/cycle-114-valid.json).
  [Narrative](full_log.md#cycle-114-full-affine-prior-outside-the-gp-search-span).

## periodic-torsion-gp: Sine/cosine extension of the tangent residual embedding
Status: deferred

**Hypothesis:** Physical torsion periodicity can improve GP residual interpolation without changing the current local metric or enlarging the search space.
**Outcome and uncertainty:** Cycle 115 converges fully with valid energy and no errors, but loses training cost. No diagnostic artifacts identify the cause; the current-centered mixed feature metric is an approximation.
**Reason to revisit:** An independently supported nonlinear prior or intrinsic angular metric could make periodic residual features more useful. The analytic directional/query chain rules remain a reusable implementation.
**Next experiment:** Defer until that geometric/mean connection is derived and supported. No period or length-scale sweep.

**Attempts:**
- Cycle 115; candidate `7dc426d1d2e7f4138b56c990c9393705cb0de5e6`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision discard.
  [Implementation](ideas/periodic-torsion-gp/7dc426d1d2e7f4138b56c990c9393705cb0de5e6/algo.py).
  [Training evidence](evaluation_results/cycle-115-train.json).
  [Narrative](full_log.md#cycle-115-periodic-torsion-features-for-the-gp-residual).

## expected-gradient-gp: Acquire low posterior expected squared gradient
Status: deferred

**Hypothesis:** A stationarity objective including gradient covariance avoids selecting inaccurate low-energy surrogate minima.
**Outcome and uncertainty:** Cycle 116 fails aggregate energy and worsens cost; full convergence and no errors or diagnostic failures. The calibrated quality of posterior gradient variance remains unobserved.
**Reason to revisit:** An independently supported gradient-error model or physical weighting could make risk-aware stationarity useful. The mapped Hessians and anchored gradient covariance are exact reusable algebra for the tested GP.
**Next experiment:** Defer until that observational/metric evidence exists. No confidence coefficient, solver-budget or activation sweep.

**Attempts:**
- Cycle 116; candidate `38f6cf45848e09bc6f8d66266c89c65bf216863c`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision invalid.
  [Implementation](ideas/expected-gradient-gp/38f6cf45848e09bc6f8d66266c89c65bf216863c/algo.py).
  [Training evidence](evaluation_results/cycle-116-train.json).
  [Narrative](full_log.md#cycle-116-expected-gradient-risk-gp-acquisition).

## soft-secant-curvature: Positive curvature from a soft secant penalty
Status: deferred

**Hypothesis:** A softened secant observation retains the physical prior when transported gradient differences are inconsistent, while preserving positive curvature for either observed curvature sign.
**Outcome and uncertainty:** Cycle 117 fully converges without errors but fails energy and worsens cost substantially. The manifest contains no failure artifacts; penalty, reflection and transport effects cannot be separated from endpoints.
**Reason to revisit:** An independently supported secant-error estimate could determine a meaningful relaxation scale. The exact small-span inverse formula remains available for such a noise model.
**Next experiment:** Defer until that error model is justified; no penalty-strength or spectral-floor sweep from broad negative evidence.

**Attempts:**
- Cycle 117; candidate `b57687885aa31f60547a11475ad358ea76d0569b`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision invalid.
  [Implementation](ideas/soft-secant-curvature/b57687885aa31f60547a11475ad358ea76d0569b/algo.py).
  [Training evidence](evaluation_results/cycle-117-train.json).
  [Narrative](full_log.md#cycle-117-energy-normalized-soft-quasi-newton-curvature).

## preconditioned-conjugate-step: Internal PR+ with a learned current metric
Status: deferred

**Hypothesis:** Conjugate directions correct persistent Newton-model errors while the accepted Hessian provides preconditioning.
**Outcome and uncertainty:** Cycle 118 fully converges with valid energy and no errors, but loses training cost. No failure evidence identifies a localized repair; projection and model-only line length remain approximations.
**Reason to revisit:** A supported transport/line-search connection could make conjugacy useful; the retained direction engine has no inertial time-scale parameters.
**Next experiment:** Defer until that connection is justified. No beta cap, restart threshold or line-length sweep from endpoints alone.

**Attempts:**
- Cycle 118; candidate `44978d43cf128fb09f3d5189f54a86e40caf7fcb`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision discard.
  [Implementation](ideas/preconditioned-conjugate-step/44978d43cf128fb09f3d5189f54a86e40caf7fcb/algo.py).
  [Training evidence](evaluation_results/cycle-118-train.json).
  [Narrative](full_log.md#cycle-118-preconditioned-conjugate-directions-inside-the-internal-box).

**Cycle181 formulation/reassessment:** Hager–Zhang theta=1 now uses actual native geodesic secants and the inferred ordinary trust-shifted metric, vanishing under a compatible inverse secant. This replaces118's PR+ projected proposal memory. It fully converges and passes energy but loses training cost substantially; no diagnosed failure supports another beta, shift or line-length repair. Defer both formulations.
- Cycle181; candidate `b841e1cac9d8119647f4e93c219d17a8647eb86f`; champion `4807cb0fb347c39d9c849ada2e3f404d9e557930`; decision discard.
  [Implementation](ideas/preconditioned-conjugate-step/b841e1cac9d8119647f4e93c219d17a8647eb86f/algo.py).
  [Training evidence](evaluation_results/cycle-181-train.json).
  [Narrative](full_log.md#cycle-181-native-secant-hager-zhang-directions-in-the-shifted-metric).

**Cycle486 revisiting:** Full fixed-physical Cartesian Hager–Zhang directions coupled to paid strong-Wolfe search replaces model-only lengths and approximate native transport. This explicitly addresses the recorded prerequisite; unchanged gates apply. [Narrative](full_log.md#cycle-486-physical-conjugate-gradients-with-paid-wolfe-search).

**Cycle486 reassessment:** The full fixed-physical Hager–Zhang/Wolfe coupling fails energy with297 limits172 converged0 errors;every case slower. The missing line-search prerequisite alone is insufficient. Passive endpoint trace lacks direction/acceptance state. Revisit requires independently supported evolving-metric conjugacy or diagnosed trial waste, with no beta, budget, cap or floor sweeps.
- Cycle486; candidate `9deddbecd5b2d9e52b37bb71cf0f7160601966f1`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision invalid. [Implementation](ideas/preconditioned-conjugate-step/9deddbecd5b2d9e52b37bb71cf0f7160601966f1/algo.py), [training](evaluation_results/cycle-486-train.json), [narrative](full_log.md#cycle-486-physical-conjugate-gradients-with-paid-wolfe-search).

**Cycle576 reassessment:** Currentchemical Cartesian metric plus published nonlinear hat-PR+ addresses486 evolvingmetric prerequisite. Improves486 convergence172→415, but54limits,energyinvalid,464slower/1faster/4same against463. Passive200row/8frame trace shows force/motion oscillations without beta/acceptance/metricstate; no localized repair. Revisitrequires suchdiagnosis or independently coupled globalization; no beta,Wolfe,cap/floor sweeps.
- Cycle576; candidate `c1ef6b20dfe3549e1afaa287985a88adc8e1acea`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision invalid. [Implementation](ideas/preconditioned-conjugate-step/c1ef6b20dfe3549e1afaa287985a88adc8e1acea/algo.py), [training](evaluation_results/cycle-576-train.json), [trace](diagnostics/173f0b70ba594bf592875f8aa3bc6c4e/02b04a7d2c247610d69ca647d86466e721e8038ab9e84c2e48e058a657182717.json), [narrative](full_log.md#cycle-576-moving-chemical-preconditioned-conjugate-gradients).

## torsion-multiplicity-prior: Normalize proper torsional stiffness per central bond
Status: deferred

**Hypothesis:** Coordinate multiplicity should not amplify one physical torsion's prior stiffness; adapt UFF normalization to the empirical diagonal Hessian.
**Outcome and uncertainty:** Cycle 120 fully converges and passes energy but worsens cost. Direct normalization may double-count coordination corrections already in the empirical formula. No failure artifacts support a localized repair.
**Reason to revisit:** A collective torsional prior derived from a physical rotation mode could make its barrier normalization internally consistent; this implementation provides a redundancy control for that reformulation.
**Next experiment:** Derive such a collective model before another test. Do not sweep normalization exponents for this failed transfer.

**Attempts:**
- Cycle 120; candidate `8e7fd1e079dfe49b97149c584a776da72ab27cd0`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision discard.
  [Implementation](ideas/torsion-multiplicity-prior/8e7fd1e079dfe49b97149c584a776da72ab27cd0/algo.py).
  [Training evidence](evaluation_results/cycle-120-train.json).
  [Narrative](full_log.md#cycle-120-normalize-proper-torsion-prior-by-central-bond-multiplicity).

## transported-secant-mixing: Native secant history with tangent transport
Status: deferred

**Hypothesis:** Use physically observed geodesic steps/gradient changes in a transported multisecant inverse action to avoid raw chord GDIIS and nonlinear GP model errors.
**Outcome and uncertainty:** Cycle 122 fails aggregate energy and loses cost despite complete convergence and zero errors. No diagnostic artifacts. Polar transport and changing preconditioner are approximate; endpoints cannot isolate their contributions.
**Reason to revisit:** This implements native secant collection and consistent old-pair transport for a future independently diagnosed locality or transport improvement. It is also a control against cycle 16's chord GDIIS.
**Next experiment:** Require direct evidence of transport or inverse-fit conditioning failure before a targeted repair; no ridge or activation sweep follows this broad failure.

**Attempts:**
- Cycle 122; candidate `6e425c6a1182264dfd0a4c09bcacbb16cbf2987f`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision invalid.
  [Implementation](ideas/transported-secant-mixing/6e425c6a1182264dfd0a4c09bcacbb16cbf2987f/algo.py).
  [Training evidence](evaluation_results/cycle-122-train.json).
  [Narrative](full_log.md#cycle-122-transported-native-multisecant-step-selection).

## ordinary-bfgs-sr1: Published BFGS component in the accepted residual blend
Status: deferred

**Hypothesis:** Ordinary BFGS on safely positive secants can improve minimum-search curvature while the accepted SR1 mixture retains indefinite information.
**Outcome and uncertainty:** Cycle 124 passes energy and fully converges without errors but loses cost. This direct formula control gives no evidence for denominator or damping repairs; changing secant geometry could alter its behavior.
**Reason to revisit:** Useful as a published-formula control after an independently established improvement in secant transport or coordinate representation, not a blend-weight tuning target.
**Next experiment:** No immediate repair. Require such an independently supported interaction before another full-gate test.

**Attempts:**
- Cycle 124; candidate `d7723446d36b88e3dafd9a7795199e365bea5162`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision discard.
  [Implementation](ideas/ordinary-bfgs-sr1/d7723446d36b88e3dafd9a7795199e365bea5162/algo.py).
  [Training evidence](evaluation_results/cycle-124-train.json).
  [Narrative](full_log.md#cycle-124-ordinary-bfgs-component-in-the-residual-aligned-sr1-blend).

**Cycle216 prerequisite:**Accepted182 changes bond coordinates and186 corrects the moving connection. Reapply124 verbatim to this independently improved secant geometry, with all curvature guards and mixing weights unchanged.

**Cycle216 outcome:**Accepted182/186 geometry does not rescue the unchanged124 formula: full convergence and valid energy, substantial cost loss. No guard/mixing tuning. Revisit only with independently diagnosed model-update error or a materially different curvature learner requiring this formula control.
- Cycle216; candidate `9a3e60629e9098d272ae05d1adaee7a31bf46808`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision discard.
  [Implementation](ideas/ordinary-bfgs-sr1/9a3e60629e9098d272ae05d1adaee7a31bf46808/algo.py), [train](evaluation_results/cycle-216-train.json), [narrative](full_log.md#cycle-216-ordinary-bfgs-blend-with-current-geodesic-secants).

## energy-defect-fit-weight: Downweight nonquadratic physical-fit observations
Status: deferred

**Hypothesis:** The energy/endpoint-work trapezoidal defect estimates secant nonlinearity and can improve shared physical coefficients by reducing unreliable observation weights.
**Outcome and uncertainty:** Cycle 125 passes energy with full convergence and no errors, but loses cost. No fit-state traces establish whether weighting is too strong or the directional error model is inappropriate.
**Reason to revisit:** The paid-for path-defect estimate is a reusable reliability feature if a future model connects directional and transverse error or supplies passive fit-state evidence.
**Next experiment:** No amplitude/exponent sweep. Derive or diagnose that observation-error relationship before a new weighting formulation.

**Attempts:**
- Cycle 125; candidate `d8d2ed6a0ed3d1ad5ebfb94cb6781f47c5a5ed32`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision discard.
  [Implementation](ideas/energy-defect-fit-weight/d8d2ed6a0ed3d1ad5ebfb94cb6781f47c5a5ed32/algo.py).
  [Training evidence](evaluation_results/cycle-125-train.json).
  [Narrative](full_log.md#cycle-125-energy-defect-uncertainty-in-physical-coefficient-calibration).

**236 revisit:**An isotropic cubic Taylor-kernel model derives3:1 longitudinal/transverse error covariance from the magnitude-only paid energy defect. Test this relationship on234 with unchanged first-six budget, ridge, bounds and raw replay; no scalar-amplitude sweep.

**236 reassessment:**Derived cubic3:1 error anisotropy still loses cost, with valid energy and full convergence. The amplitude, isotropy and fixed-prior/curved-path mismatch remain uncalibrated. Require independently measured error structure or an explicitly signed, common-frame model before another attempt; no amplitude/exponent/ratio sweep.
- Cycle236; candidate `1c1020fdcedbd5c64367e96d3c22c8cb849fb88b`; champion `73cadc1ae8fd47e796123cafe93956f0e2c67043`; decision discard. [Implementation](ideas/energy-defect-fit-weight/1c1020fdcedbd5c64367e96d3c22c8cb849fb88b/algo.py), [train](evaluation_results/cycle-236-train.json), [narrative](full_log.md#cycle-236-cubic-energy-defect-covariance-in-physical-calibration).

## adaptive-quadratic-gp-trend: Marginalize one shared curvature-scale correction
Status: deferred

**Hypothesis:** An uncertain scalar physical curvature scale plus a nonlinear SE residual separates coherent mean error from spatial interpolation.
**Outcome and uncertainty:** Cycle 126 is energy-valid and fully converged without errors but loses cost. Posterior scale/acceptance states are not retained, leaving redundancy and confidence effects unresolved.
**Reason to revisit:** The rank-one trend component is usable if independently observed mean-scale error or a derived uncertainty prior supports this separation. It differs from cycle 84's purely quadratic replacement.
**Next experiment:** No prior-amplitude sweep. Require that specific mean-error evidence before another hierarchical model.

**Attempts:**
- Cycle 126; candidate `32195f1e3f11580169fff30bc091e9e220c831e1`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision discard.
  [Implementation](ideas/adaptive-quadratic-gp-trend/32195f1e3f11580169fff30bc091e9e220c831e1/algo.py).
  [Training evidence](evaluation_results/cycle-126-train.json).
  [Narrative](full_log.md#cycle-126-adaptive-scalar-quadratic-trend-with-nonlinear-gp-residual).

**Revisit 152:** Derive a correlated coefficient covariance from the existing physical-fit Gram and normalized residuals, replacing the former single scalar fixed-amplitude trend. This addresses the requested derived uncertainty prior, while treating clipping, redundant components and observation reuse as explicit limitations.

**Reassessment 152:** The correlated physical-fit uncertainty model also loses training cost, with complete convergence and valid energy. A residual covariance in redundant native components is only a working model; there is no calibration evidence supporting amplitude/denominator tuning. Revisit only with independently justified curvature-error statistics.
- Cycle 152; candidate `c721bdc6a9f9942d27ed913ebad989c4d0753a3f`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision discard.
  [Implementation](ideas/adaptive-quadratic-gp-trend/c721bdc6a9f9942d27ed913ebad989c4d0753a3f/algo.py).
  [Training evidence](evaluation_results/cycle-152-train.json).
  [Narrative](full_log.md#cycle-152-physical-fit-uncertainty-as-a-correlated-quadratic-gp-component).

## stretch-torsion-calibration: Bounded Fourier cross-curvature features
Status: deferred

**Hypothesis:** Phase-aware central-stretch/torsion couplings can improve early physical curvature learning, using three Class II-inspired harmonics with a fixed total correlation budget.
**Outcome and uncertainty:** Cycle 128 passes energy and fully converges/error-free, but training cost gain 0.0000409145 is below the fixed gate. No coefficient traces establish whether features are weakly identified or redundant.
**Reason to revisit:** The bounded phase-consistent construction can be useful with an independently supported calibration-identifiability improvement. A small positive endpoint gain alone does not justify strengthening it.
**Next experiment:** Require such evidence or a diagnosed coefficient-fit limitation before altering the model; no harmonic/amplitude/ridge sweep and no validation of this rejected candidate.

**Attempts:**
- Cycle 128; candidate `8f5ce04e25f9088cd3743182de2e134a18bec55c`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision discard.
  [Implementation](ideas/stretch-torsion-calibration/8f5ce04e25f9088cd3743182de2e134a18bec55c/algo.py).
  [Training evidence](evaluation_results/cycle-128-train.json).
  [Narrative](full_log.md#cycle-128-bounded-fourier-stretch-torsion-calibration).

## symmetric-inverse-multisecant: Symmetric inverse learning in a physical metric
Status: deferred

**Hypothesis:** Joint symmetric inverse fitting can combine native geodesic secants into a potential-consistent step model with regularized weak modes.
**Outcome and uncertainty:** Cycle 131 passes energy with full convergence and no errors but substantially loses cost. Symmetry does not resolve the observed inferential limitations; successful fit/transport states are not retained.
**Reason to revisit:** The derived thin-SVD inverse fit supplies a controlled symmetric alternative if transport/locality or conditioning evidence later identifies why raw history is unreliable. No such evidence currently establishes benefit.
**Next experiment:** Defer; require that independent evidence before changing ridge, memory or guards. Compare under both unchanged gates.

**Attempts:**
- Cycle 131; candidate `6ac061c228a96b1fa41c49835c60a096c90f5b06`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision discard.
  [Implementation](ideas/symmetric-inverse-multisecant/6ac061c228a96b1fa41c49835c60a096c90f5b06/algo.py).
  [Training evidence](evaluation_results/cycle-131-train.json).
  [Narrative](full_log.md#cycle-131-physically-scaled-symmetric-inverse-multisecant-steps).

## transported-physical-calibration: One orthogonal frame for the early fit
Status: deferred

**Hypothesis:** Jointly transport physical blocks and cached secants to preserve the six-step fit objective across tangent-space changes.
**Outcome and uncertainty:** Cycle 132 passes energy and fully converges without errors but loses cost. The constructed Gram/RHS invariance is algebraic; endpoint rotations remain an approximate geometric model.
**Reason to revisit:** The complete-state transport is usable if a later independently justified connection or moving physical-feature model needs consistent regression statistics. It does not itself show a force-call benefit.
**Next experiment:** Defer until that interaction is derived or passive evidence supports a particular repair; no overlap or activation sweep.

**Attempts:**
- Cycle 132; candidate `4056c3baff113de74acdffb93b15fd33afa7e6c6`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision discard.
  [Implementation](ideas/transported-physical-calibration/4056c3baff113de74acdffb93b15fd33afa7e6c6/algo.py).
  [Training evidence](evaluation_results/cycle-132-train.json).
  [Narrative](full_log.md#cycle-132-joint-frame-transport-during-physical-calibration).

## local-curvature-memory: Structural locality for learned corrections
Status: deferred

**Hypothesis:** Atomic-support overlap can forget inaccurate nonlocal secant corrections while preserving the calibrated physical prior and exact newest secant.
**Outcome and uncertainty:** Cycle 133 passes energy, has no errors, but loses cost broadly (70 faster, 239 slower); one call limit. Passive evidence shows slow continuing descent and substantial motion, with no Hessian states proving which couplings were lost.
**Reason to revisit:** Structural overlap is a reusable locality prior if a future diagnostic or structured secant derivation identifies unreliable nonlocal information. It cannot currently distinguish useful collective curvature from erroneous memory.
**Next experiment:** Defer until that distinction is supported; do not tune overlap strength, activation or force thresholds to rescue this broad failure.

**Attempts:**
- Cycle 133; candidate `070d27f764866d86b9c7b9780a953efb21e89d00`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision discard.
  [Implementation](ideas/local-curvature-memory/070d27f764866d86b9c7b9780a953efb21e89d00/algo.py).
  [Training evidence](evaluation_results/cycle-133-train.json).
  [Passive trace](diagnostics/5ef4f2fd47d14a5988753a03420a507f/40d014386c9d4c1e27ad1114b81749972d3519648e1098e09bd2f3390e8c8a95.json).
  [Narrative](full_log.md#cycle-133-local-atomic-support-for-learned-curvature-memory).

## gradient-history-gp: Historical derivatives with one scalar anchor
Status: deferred

**Hypothesis:** Directional gradient history plus current energy can avoid unhelpful scalar constraints in an approximate GP chart.
**Outcome and uncertainty:** Cycle 134 fails energy and cost despite full convergence and no errors; no diagnostic artifacts. The joint observation model remains superior in this configuration.
**Reason to revisit:** The exact observation-subset control is reusable if future chart or observation-error evidence specifically identifies historical scalar constraints as unreliable. Current endpoints provide no such diagnosis.
**Next experiment:** No noise/confidence sweep. First complete the complementary historical-energy control, independently from the champion; defer this variant until new model evidence.

**Attempts:**
- Cycle 134; candidate `ac3c1795e5a04c2e5c45a123777d62037d1912d0`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision invalid.
  [Implementation](ideas/gradient-history-gp/ac3c1795e5a04c2e5c45a123777d62037d1912d0/algo.py).
  [Training evidence](evaluation_results/cycle-134-train.json).
  [Narrative](full_log.md#cycle-134-historical-gradient-gp-with-a-current-energy-anchor).

## energy-history-gp: Scalar history with current gradient anchors
Status: deferred

**Hypothesis:** Historical energies could complement native Hessian learning without fitting approximate old derivative constraints in the nonlinear GP.
**Outcome and uncertainty:** Cycle 135 fails aggregate energy and loses cost with full convergence and no errors. Together with 134, the observations support the joint model; endpoint evidence does not diagnose a repair.
**Reason to revisit:** This subset construction is a controlled baseline if a future derivative-error model identifies unreliable historical gradients. It has no demonstrated advantage now.
**Next experiment:** Defer; no relative-noise or confidence tuning. Require independent observation-error evidence for a new formulation.

**Attempts:**
- Cycle 135; candidate `9fef9c537494d27cec1fc4b315cedfc6bdb3a2d5`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision invalid.
  [Implementation](ideas/energy-history-gp/9fef9c537494d27cec1fc4b315cedfc6bdb3a2d5/algo.py).
  [Training evidence](evaluation_results/cycle-135-train.json).
  [Narrative](full_log.md#cycle-135-historical-energy-gp-with-current-derivative-anchors).

## physical-mode-gp: Add one missing physical mode to nonlinear search
Status: deferred

**Hypothesis:** A large predicted-lowering Hessian mode outside the history span can provide a useful nonlinear search direction.
**Outcome and uncertainty:** Cycle 136 passes energy and fully converges without errors, but loses cost. No successful surrogate states diagnose whether the selected mode lacks observations or harms the metric.
**Reason to revisit:** The mode score can support future direction selection if an independently validated directional-information criterion establishes which curvature modes the GP can learn reliably.
**Next experiment:** Defer; no mode-count, spectral-floor or confidence sweep without that evidence.

**Attempts:**
- Cycle 136; candidate `25d48f73fc76548cc30fea948fa4ca4aed2d0318`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision discard.
  [Implementation](ideas/physical-mode-gp/25d48f73fc76548cc30fea948fa4ca4aed2d0318/algo.py).
  [Training evidence](evaluation_results/cycle-136-train.json).
  [Narrative](full_log.md#cycle-136-dominant-missing-physical-mode-gp-enrichment).

## quadratic-complement-gp: Full physical mean with a reduced nonlinear residual
Status: deferred

**Hypothesis:** Analytic minimization outside a whitened GP residual span can retain full physical response without learning nonlinear behavior in unsupported directions.
**Outcome and uncertainty:** Cycle 137 fails energy and narrowly loses cost despite full convergence and no errors. No successful model/path traces identify whether transverse motion or changed residual extension causes the endpoint losses.
**Reason to revisit:** The complete mean, derivative maps and complement elimination provide a consistent component if an independently supported physical-curvature or geometry mapping improvement changes the model's reliability.
**Next experiment:** Defer until such evidence or derivation exists. Do not sweep complement strength or confidence from endpoint basin changes alone.

**Attempts:**
- Cycle 137; candidate `b6c8a913df02834415f1ed8c6b819e4bd18cb4a4`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision invalid.
  [Implementation](ideas/quadratic-complement-gp/b6c8a913df02834415f1ed8c6b819e4bd18cb4a4/algo.py).
  [Training evidence](evaluation_results/cycle-137-train.json).
  [Narrative](full_log.md#cycle-137-full-quadratic-mean-with-an-eliminated-gp-complement).

## angle-torsion-calibration: Phase-aware bending and twisting curvature
Status: deferred

**Hypothesis:** Three bounded Fourier cross-curvature features can learn coupled adjacent-angle/torsion motion during early physical calibration.
**Outcome and uncertainty:** Cycle 138 passes training but loses validation cost; both splits pass energy and fully converge without errors. No successful coefficient/path traces isolate the generalization failure.
**Reason to revisit:** The bounded physical feature construction could become useful with independently supported improvements in calibration identifiability or locality. Training gain alone does not justify an amplitude sweep.
**Next experiment:** Defer until such evidence exists; preserve the three-feature model and compare a diagnosed fit change under both fixed gates.

**Attempts:**
- Cycle 138; candidate `a997b451376fb31b6197a29408b4345d11dba5c1`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision non_generalizable.
  [Implementation](ideas/angle-torsion-calibration/a997b451376fb31b6197a29408b4345d11dba5c1/algo.py).
  [Training evidence](evaluation_results/cycle-138-train.json).
  [Validation evidence](evaluation_results/cycle-138-valid.json).
  [Narrative](full_log.md#cycle-138-bounded-fourier-angle-torsion-calibration).

## cartesian-force-residual: Choose motion by predicted physical stationarity
Status: deferred

**Hypothesis:** A constrained Cartesian force-norm model with exact coordinate chain terms can select faster stationarity-seeking steps from learned native curvature.
**Outcome and uncertainty:** Cycle 139 fails energy and loses cost broadly (67 faster, 305 slower); three call limits, no errors. Their bounded traces show above-threshold maximum forces and substantial motion despite low RMS forces, without model/solver states identifying a repair.
**Reason to revisit:** The analytic Cartesian residual map is reusable if a future curvature/globalization model independently establishes a reliable connection between predicted and actual force reduction. The current first-order energy guard is insufficient evidence.
**Next experiment:** Defer; no norm-weight, solver-budget or guard sweep from these endpoint failures. Require a diagnosed model improvement before reusing the force objective.

**Attempts:**
- Cycle 139; candidate `ef80c4401f6e00ce45b4f0243c05d95a3f7660e3`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision invalid.
  [Implementation](ideas/cartesian-force-residual/ef80c4401f6e00ce45b4f0243c05d95a3f7660e3/algo.py).
  [Training evidence](evaluation_results/cycle-139-train.json).
  [Diagnostic manifest](diagnostics/b0678d651deb4ed7918913d863171074/manifest.json).
  [Narrative](full_log.md#cycle-139-cartesian-force-residual-step-with-native-curvature).

## nearest-positive-spectrum: Preserve positive modes while flooring negative curvature
Status: deferred

**Hypothesis:** A Frobenius-nearest positive step model can soften negative modes without shifting reliable positive stiffness.
**Outcome and uncertainty:** Cycle 140 passes energy and fully converges without errors but loses cost. No passive model traces diagnose which indefinite steps cause the loss.
**Reason to revisit:** The minimal spectral modification is a reusable control if independently validated curvature-quality or globalization changes alter treatment of negative modes.
**Next experiment:** Defer; no spectral-floor or trust-threshold sweep without that evidence.

**Attempts:**
- Cycle 140; candidate `cb96c77cd23e4e86efc3c29aebbe7f219ac960ac`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision discard.
  [Implementation](ideas/nearest-positive-spectrum/cb96c77cd23e4e86efc3c29aebbe7f219ac960ac/algo.py).
  [Training evidence](evaluation_results/cycle-140-train.json).
  [Narrative](full_log.md#cycle-140-nearest-positive-spectrum-for-indefinite-ordinary-steps).

## cartesian-chord-curvature: Learn in a Cartesian frame while stepping in internals
Status: deferred

**Hypothesis:** Cartesian chord secants and exact Hessian chain transformations can avoid incompatible internal tangent transports while retaining native geometry.
**Outcome and uncertainty:** Cycle 141 passes energy but loses cost substantially (12 faster, 338 slower), with three call limits and no errors. Passive traces show continuing descent; two retain high forces, one low force with above-threshold displacement. No matrix states diagnose a frame repair.
**Reason to revisit:** The paired matrix/secant conversion can serve as a control if independent evidence supports a Cartesian curvature model or rotational treatment. Coordinate-chain correctness alone does not deliver efficiency.
**Next experiment:** Defer; no alignment, activation or update-strength tuning without specific model evidence.

**Attempts:**
- Cycle 141; candidate `6bd6de19a6b239262c45c97c9e477af9629e365d`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision discard.
  [Implementation](ideas/cartesian-chord-curvature/6bd6de19a6b239262c45c97c9e477af9629e365d/algo.py).
  [Training evidence](evaluation_results/cycle-141-train.json).
  [Diagnostic manifest](diagnostics/cdd3bd1c345a4c3c8148cc833d65fae2/manifest.json).
  [Narrative](full_log.md#cycle-141-cartesian-chord-curvature-learning-with-native-geometry).

## cubic-gaussian-features: Finite local Gaussian feature expansion
Status: deferred

**Hypothesis:** Gaussian-weighted features through cubic degree can regularize poorly identified higher-order response while matching local SE derivative statistics through order three.
**Outcome and uncertainty:** Cycle 142 passes energy and fully converges without errors but loses cost substantially. No passive model traces isolate omitted features versus nonstationary variance.
**Reason to revisit:** The analytic finite covariance is a useful component if independent evidence identifies which local derivative orders are reliable or justifies a finite-feature error model.
**Next experiment:** Defer; no degree, envelope, amplitude or noise sweep absent that evidence.

**Attempts:**
- Cycle 142; candidate `339a31d718f5419f9a5ea9fc612cbdc4f6719fb0`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision discard.
  [Implementation](ideas/cubic-gaussian-features/339a31d718f5419f9a5ea9fc612cbdc4f6719fb0/algo.py).
  [Training evidence](evaluation_results/cycle-142-train.json).
  [Narrative](full_log.md#cycle-142-cubic-order-gaussian-feature-prior).

## extra-distance-coordinates: Across-angle distances in the geometry metric
Status: deferred

**Hypothesis:** Explicit extra redundant distances can improve native geometry paths while keeping the original Cartesian primitive quadratic and mapped spring terms.
**Outcome and uncertainty:** Cycle 145 passes energy with all runs converged and no errors but loses cost. No retained path/model diagnostics explain the loss; the coordinate implementation is operationally viable.
**Reason to revisit:** The additional-coordinate construction can support an independently derived coordinate metric or path-model interaction without changing chemical connectivity or double-counting springs.
**Next experiment:** Defer until such a derivation or diagnosis exists; no cutoff, coordinate count or weight sweep.

**Attempts:**
- Cycle 145; candidate `a8936cc3566d737a2f29290c723bc7338d2fa631`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision discard.
  [Implementation](ideas/extra-distance-coordinates/a8936cc3566d737a2f29290c723bc7338d2fa631/algo.py).
  [Training evidence](evaluation_results/cycle-145-train.json).
  [Narrative](full_log.md#cycle-145-extra-redundant-across-angle-distance-coordinates).

**Warped-metric revisit273:**Zhu/Woo's published exp[-1.7(r/R-1)]+.01R/r coordinate defines an independently derived nonlinear metric. Reuse145's across-angle pair list and zero additional physical spring contribution; use exact derivatives throughout geometry. Existing auxiliary spring curvature provides positive coordinate-unit metric entries for new rows, while physical springs remain counted only once in the auxiliary block. This tests the declared metric prerequisite, not pair-count/cutoff/weight tuning.

**Reassessment273:**The published warped-metric prerequisite was tested with the same pair set and physical spring law. All469 converge with valid energy but cost worsens. No passive failure localizes a repair. Defer until an independently supported finite-path/learning compatibility mechanism; no warp/cutoff/weight sweep.
- Cycle273; candidate `ba186e9640d824a8eb9613f1f1f12b2c6482c133`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision discard.
  [Implementation](ideas/extra-distance-coordinates/ba186e9640d824a8eb9613f1f1f12b2c6482c133/algo.py), [training](evaluation_results/cycle-273-train.json), [narrative](full_log.md#cycle-273-warped-extra-distance-geometry-metric).

## periodic-physical-mean: Coupled periodic torsional extension of the GP prior
Status: deferred

**Hypothesis:** Replace the coupled quadratic's finite torsional response by a sine/cosine extension preserving its complete current gradient and Hessian, to reduce anharmonic residual learning.
**Outcome and uncertainty:** Cycle 148 passes training but loses validation cost; both splits converge fully with valid energy and no errors. The passive validation manifest is empty. Primitive redundancy and omitted chemical Fourier multiplicities remain model uncertainties, not established causes.
**Reason to revisit:** An independently derived way to infer a physically meaningful coupled torsion phase/multiplicity from existing observations could improve the extension. The analytic mean and consistent target derivatives are reusable.
**Next experiment:** Defer until such a model is supported; no period, amplitude or Fourier-order sweep. Compare any later revision against the then-current champion under unchanged gates.

**Attempts:**
- Cycle 148; candidate `c70229b7395de77edc3258723e762d6de1aeb10b`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision non_generalizable.
  [Implementation](ideas/periodic-physical-mean/c70229b7395de77edc3258723e762d6de1aeb10b/algo.py).
  [Training evidence](evaluation_results/cycle-148-train.json).
  [Validation evidence](evaluation_results/cycle-148-valid.json).
  [Narrative](full_log.md#cycle-148-periodic-extension-of-the-coupled-physical-gp-mean).

## self-scaled-secant: Repeated one-sided adaptation of learned curvature
Status: deferred

**Hypothesis:** Reduce an over-stiff learned Hessian before installing a lower positive directional secant, extending scale adaptation beyond initialization.
**Outcome and uncertainty:** Cycle 150 passes energy but severely loses cost (273 slower, 59 faster), with three call-limit outcomes and no errors. Passive traces show continued large displacements and uphill moves; matrix/scaling state is unavailable. Exact directional scaling also zeroes the residual cosine, an explicit property of this tested recurrence.
**Reason to revisit:** A derived scaling restricted to unlearned directions could preserve accumulated secants and separate background amplitude from blend orientation. No such implementation is currently supported.
**Next experiment:** Defer until that representation is derived; no scalar floor, activation or timing sweep. Retain the accepted unscaled update.

**Attempts:**
- Cycle 150; candidate `f773fe358bb8994016bfb71267f30641f7193df9`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision discard.
  [Implementation](ideas/self-scaled-secant/f773fe358bb8994016bfb71267f30641f7193df9/algo.py).
  [Training evidence](evaluation_results/cycle-150-train.json).
  [Passive manifest](diagnostics/a69797e948854d3f8baa518cfdf6dd64/manifest.json).
  [Narrative](full_log.md#cycle-150-one-sided-self-scaling-before-mixed-secant-learning).

**Revisit285:**The independently sourced sampled/unsampled decomposition supplies150's requested secant-preserving representation. Test a PSD floor based on observed dimensionless spectral curvature only in the current tangent complement of all saved steps, after the native update and six-pair fit. This is conservative stiffening with protected actions, not another global-softening factor. No history window or scaling multiplier is tuned.

**285 reassessment:**The derived secant-protected complement floor fully converges with zero errors but fails aggregate energy and substantially loses cost. Protecting learned actions is insufficient to justify transferring the largest observed scale to unseen modes. Require independently measured directional uncertainty before revisiting; no maximum/scale/rank/window/activation sweep.
- Cycle285; candidate `59c4eee25830488c10114745fc65b98ce6e154a3`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision invalid.
  [Implementation](ideas/self-scaled-secant/59c4eee25830488c10114745fc65b98ce6e154a3/algo.py), [train](evaluation_results/cycle-285-train.json), [narrative](full_log.md#cycle-285-secant-protected-unsampled-curvature-floor).

## quartic-surrogate-search: Momentum and scaled PSB for GP microoptimization
Status: deferred

**Hypothesis:** Quartic stabilization and gradient momentum can solve the nonconvex dimensionless GP more effectively within the existing twelve-step budget.
**Outcome and uncertainty:** Cycle 151 passes energy and fully converges without errors but loses cost. No passive traces or microstep states identify whether finite-budget search, candidate bounds or model minima cause the loss. 431 tests the full physical method and fails broadly:50calculator errors,325call limits,94converged; every returned trajectory is slower than430. Representative passive bundles show early instability and persistent residual forces, not an isolated solver defect.
**Reason to revisit:** Evidence of a particular nonconvex search failure or a principled physical-unit formulation could motivate a distinct use of the spectral quartic solver. Its hard-case algebra is reusable.
**Next experiment:** Defer until an independently justified molecular globalization/curvature formulation addresses both stability and slow progress; no epoch, sigma, reference-length, cap, PSB or averaging sweep.

**Attempts:**
- Cycle 151; candidate `44f7536dfa1af7a727e88d3ec24a9f92dba1cf9b`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision discard.
  [Implementation](ideas/quartic-surrogate-search/44f7536dfa1af7a727e88d3ec24a9f92dba1cf9b/algo.py).
  [Training evidence](evaluation_results/cycle-151-train.json).
  [Narrative](full_log.md#cycle-151-quartic-momentum-search-of-the-gp-surrogate).

**431 prerequisite:** Supply fixed physical Cartesian whitening, initial-direction length normalization and dimensionless energy units, then test the complete paid outer method. Preserve151's dimensional schedule constants; pay for epoch-average checks. This is the previously untested physical formulation, not a microiteration or parameter repair.

- Cycle431; candidate `4514630f970397c19c4047a5265c3d2f86ce010e`; champion `bb770d9c7c9e59d39007f71a106af77cd9eb73a1`; decision invalid.
  [Implementation](ideas/quartic-surrogate-search/4514630f970397c19c4047a5265c3d2f86ce010e/algo.py), [train](evaluation_results/cycle-431-train.json), [narrative](full_log.md#cycle-431-physical-quartic-momentum-with-scaled-psb-learning).

## cubic-hermite-proposals: Polyharmonic residual search under the existing GP check
Status: deferred

**Hypothesis:** A cubic Hermite residual with an affine tail learns finite gradient trends without a correlation-length choice, potentially giving better candidates under the accepted GP checks.
**Outcome and uncertainty:** Cycle 153 fully converges with valid energy and no errors but loses cost. No passive traces or interpolation/acceptance states identify conditioning or model disagreement as a localized cause.
**Reason to revisit:** A demonstrated finite-gradient interpolation failure of the SE model or a supported common confidence model could motivate using this interpolation differently. Mapped Hermite derivatives and polynomial side constraints are reusable.
**Next experiment:** Defer until that evidence exists; no radial-degree, regularizer or model-mixing sweep.

**Attempts:**
- Cycle 153; candidate `31a1ef6633480b0802bd1ddc20b9f1f9de593be2`; champion `33018bfb615e02a0d30eda737c0bbaac4cddbe81`; decision discard.
  [Implementation](ideas/cubic-hermite-proposals/31a1ef6633480b0802bd1ddc20b9f1f9de593be2/algo.py).
  [Training evidence](evaluation_results/cycle-153-train.json).
  [Narrative](full_log.md#cycle-153-cubic-hermite-residual-proposals-with-the-accepted-gp-check).

## circular-torsion-coordinates: Native sine/cosine torsional embedding
Status: deferred

**Hypothesis:** A unit-circle representation preserves local torsional metric/stiffness while making finite history globally periodic, potentially improving curvature learning and surrogate proposals.
**Outcome and uncertainty:**155–158 diagnose and repair indexing and circle-frame errors, but no version passes both gates. With independently accepted collision control and new angle priors,166 fully converges without errors yet fails aggregate energy and loses training cost. Thus the supported robustness revisit does not recover general usefulness. Successful paths provide no failure traces for a further repair.
**Reason to revisit:** An independently justified finite-history/chart formulation could address the remaining difference between ambient chords and native angular observations. Exact embedding derivatives and frame controls remain reusable; collision repair alone is insufficient.
**Next experiment:** Defer pending such a formulation. No further frame, cutoff, radius or blend sweep; retain165's raw-angle representation.

**Attempts:**
- Cycle 155; candidate `1eb0136faeaf8a7ed5d7a4bf69c5b515cfb89748`; champion `19d5b8d17667205a8b8fdadc55b55a965cd5467d`; decision invalid.
  [Implementation](ideas/circular-torsion-coordinates/1eb0136faeaf8a7ed5d7a4bf69c5b515cfb89748/algo.py).
  [Training evidence](evaluation_results/cycle-155-train.json).
  [Passive manifest](diagnostics/d63cbade305e41349638f0f6b8b0262a/manifest.json).
  [Narrative](full_log.md#cycle-155-native-unit-circle-torsional-coordinates).

**Cycle 156 repair 1:** Active-order wrapping eliminates all 214 index errors. Training remains invalid because the single SCF failure repeats to numerical precision; the other 468 cases converge. Its untruncated four-frame bundle supports a representation-level investigation. Static derivation shows the frozen initial inverse maps the changing circle's radial curvature into physical acceleration. Next test E(phi0)*E(phi)^T on ODE products, annihilating this radial term while retaining the original raw-angle connection; no radius/tolerance changes. Status remains revisiting.
- Cycle 156; candidate `f2a75f6a73095382eba9160ec4ff94690366f282`; champion `19d5b8d17667205a8b8fdadc55b55a965cd5467d`; decision invalid.
  [Implementation](ideas/circular-torsion-coordinates/f2a75f6a73095382eba9160ec4ff94690366f282/algo.py).
  [Training evidence](evaluation_results/cycle-156-train.json).
  [Passive manifest](diagnostics/f43762f2d2ab4fef91143c6fa2dfdb96/manifest.json).
  [Narrative](full_log.md#cycle-156-active-order-wrapping-for-circular-torsional-coordinates).

**Cycle 157 repair 2:** Correcting the ODE circle frame removes the repeated training SCF error and passes training cost/energy with full convergence. Validation is non_generalizable due to one different SCF failure; 464 other cases converge. Its ten scalar records and eight saved frames show increasing displacements without direct atom collision, but no matrix states establish the cause. Static audit identifies an exact remaining mismatch: stored curvature/fit history and old-model trust prediction do not follow the known circle rotation. Next test this complete-state orthogonal map as the third and final bounded repair, without changing thresholds. Status remains revisiting.
- Cycle 157; candidate `0c7b32aec36bdf6d8af83486c98fee7584a83bf8`; champion `19d5b8d17667205a8b8fdadc55b55a965cd5467d`; decision non_generalizable.
  [Implementation](ideas/circular-torsion-coordinates/0c7b32aec36bdf6d8af83486c98fee7584a83bf8/algo.py).
  [Training evidence](evaluation_results/cycle-157-train.json).
  [Validation evidence](evaluation_results/cycle-157-valid.json).
  [Passive manifest](diagnostics/7092e59d772e44cab72d2e2b77984e95/manifest.json).
  [Narrative](full_log.md#cycle-157-circle-frame-correction-of-the-frozen-ode-connection).

**Cycle 158 repair 3:** Exact orthogonal transport of all retained model state and matching trust projection yields valid, fully converged training but loses cost. No validation; bounded sequence complete.
- Cycle 158; candidate `3d4b5c3a18724441b96688e42ca8dd5f43c9f862`; champion `19d5b8d17667205a8b8fdadc55b55a965cd5467d`; decision discard.
  [Implementation](ideas/circular-torsion-coordinates/3d4b5c3a18724441b96688e42ca8dd5f43c9f862/algo.py).
  [Training evidence](evaluation_results/cycle-158-train.json).
  [Narrative](full_log.md#cycle-158-complete-circle-frame-covariance-of-the-retained-model).

**Accepted-guard revisit166:** The original train-qualified157 representation with165’s accepted physical/geometry priors fully converges but fails energy and cost. No new localized repair is established.
- Cycle166; candidate `7d22c6292ff5795c3b2af3a222416836aa5d93a2`; champion `04697384ef14d7cdbc19360b22275c0f359e2479`; decision invalid.
  [Implementation](ideas/circular-torsion-coordinates/7d22c6292ff5795c3b2af3a222416836aa5d93a2/algo.py).
  [Training evidence](evaluation_results/cycle-166-train.json).
  [Narrative](full_log.md#cycle-166-native-circle-coordinates-with-accepted-physical-and-collision-priors).

## badger-gp-mean: Integrated inverse-cubic bond law in the GP prior
Status: deferred

**Hypothesis:** An anharmonic mean obtained by integrating the accepted Badger stiffness reduces finite stretching residuals while preserving current energy, gradient and Hessian.
**Outcome and uncertainty:** Cycle 159 passes training but loses validation cost; all 934 cases converge with valid energy and zero errors. No localized failure is diagnosed. The equilibrium stiffness correlation may not describe finite within-bond potential shape.
**Reason to revisit:** Independent finite-curvature evidence from ordinary optimizer observations could support adapting the shape to actual local anharmonicity. The analytic mean, positive remainder and consistent derivative-target plumbing are reusable.
**Next experiment:** Defer until such a model is derived; no amplitude or exponent sweep. A future observation-derived shape must be compared against the then-current champion under unchanged gates.

**Attempts:**
- Cycle 159; candidate `b1a3bdca67b55099e3540abf008f756a7be16dc1`; champion `19d5b8d17667205a8b8fdadc55b55a965cd5467d`; decision non_generalizable.
  [Implementation](ideas/badger-gp-mean/b1a3bdca67b55099e3540abf008f756a7be16dc1/algo.py).
  [Training evidence](evaluation_results/cycle-159-train.json).
  [Validation evidence](evaluation_results/cycle-159-valid.json).
  [Narrative](full_log.md#cycle-159-integrated-badger-bond-gp-mean).

## schlegel-angle-prior: Published categorical bending stiffness
Status: incorporated

**Hypothesis:** Original Schlegel real-angle constants complement the accepted Badger bond prior and reduce early calibration work.
**Outcome and uncertainty:**160 is invalid from a saved new Cl–O overlap despite a promising successful subset.165 retains exactly the published constants with independently accepted collision backtracking and passes train/validation, becoming champion. All cases converge without errors; the prior's contribution interacts with existing fitted stretch-bend normalization.
**Reason to revisit:** The diagnosed failure had a general geometry remedy, now independently accepted and tested in combination.
**Next experiment:** Retain165’s accepted real-angle prior. Assess the separate periodic-coordinate idea under the improved physical/geometry model; no angle-constant tuning.

**Attempts:**
- Cycle 160; candidate `a56e588599dbb4ec3c0614754cc7af00d49d5029`; champion `19d5b8d17667205a8b8fdadc55b55a965cd5467d`; decision invalid.
  [Implementation](ideas/schlegel-angle-prior/a56e588599dbb4ec3c0614754cc7af00d49d5029/algo.py).
  [Training evidence](evaluation_results/cycle-160-train.json).
  [Passive manifest](diagnostics/6b3e95f0f5854c29b7c3eadd872f4350/manifest.json).
  [Narrative](full_log.md#cycle-160-published-schlegel-real-angle-stiffness-prior).

- Cycle165; candidate `04697384ef14d7cdbc19360b22275c0f359e2479`; champion `ed8615cd901420f128424a78729a579e637028d7`; decision keep.
  Implementation retained as accepted Git commit.
  [Training evidence](evaluation_results/cycle-165-train.json).
  [Validation evidence](evaluation_results/cycle-165-valid.json).
  [Narrative](full_log.md#cycle-165-schlegel-angle-prior-with-accepted-collision-backtracking).

## square-root-sr1-weight: Published absolute-cosine curvature blend
Status: deferred

**Hypothesis:** Stronger SR1 weighting captures residual cross-curvature suppressed by the accepted squared cosine, using the published square-root rule with the accepted TS-BFGS component.
**Outcome and uncertainty:**161 fails from a newly formed Cl–O collision; repair162 prevents it but stalls at a false O–H boundary. Current-contact recognition in163 removes both failures and fully converges with valid energy, but loses training cost. No matrix states establish whether the blend discontinuity contributes to slow cases.
**Reason to revisit:** The geometry-only collision repair is independently reusable and now merits isolation with the champion update. The stronger SR1 weight itself has no accepted gain.
**Next experiment:** Final component control removes the square-root weight while retaining163's guard unchanged. No further weight, cutoff or step-factor tuning in this bounded sequence.

**Attempts:**
- Cycle161; candidate `f5a5d43fa272e093366fa527a9cf5b82d8065cd0`; champion `19d5b8d17667205a8b8fdadc55b55a965cd5467d`; decision invalid.
  [Implementation](ideas/square-root-sr1-weight/f5a5d43fa272e093366fa527a9cf5b82d8065cd0/algo.py).
  [Training evidence](evaluation_results/cycle-161-train.json).
  [Passive manifest](diagnostics/30ee09ec18894718bdbb87814e292496/manifest.json).
  [Narrative](full_log.md#cycle-161-square-root-residual-weight-in-ts-bfgs-and-sr1-learning).

**Cross-entry diagnostic update from cycle162:** In `schlegel-angle-prior`160, the failed short pair is Cl–O (0.9587 Angstrom versus3.7902 previously). In `circular-torsion-coordinates`157 validation, it is Cl–C (1.0541 Angstrom versus4.2472 previously). This element-aware saved-result analysis establishes new severe heavy-atom contacts, which the earlier distance-only summaries did not identify. Reconsider a combination only if the independent collision backtracking in162 first establishes useful full-gate behavior; this is not permission to extend the exhausted circle repair sequence with arbitrary caps.

**square-root-sr1-weight repair1:** Cycle162 removes the original SCF failure but is invalid from collision-backtracking exhaustion elsewhere. A saved O–H distance tends to exactly0.97 Angstrom while force stays large, demonstrating the static unbonded classification blocks gradual bond formation. Next repair2 adds current-geometry contact recognition at the existing topology threshold1.25 radii sums, keeping the guard boundary and backtracking fixed.
- Cycle162; candidate `6b14589779ff5d55ce3f5a8205eea46777f0b5b1`; champion `19d5b8d17667205a8b8fdadc55b55a965cd5467d`; decision invalid.
  [Implementation](ideas/square-root-sr1-weight/6b14589779ff5d55ce3f5a8205eea46777f0b5b1/algo.py).
  [Training evidence](evaluation_results/cycle-162-train.json).
  [Passive manifest](diagnostics/3730ce753a844f0b84a020879d0731bc/manifest.json).
  [Narrative](full_log.md#cycle-162-pre-force-collision-backtracking-for-square-root-sr1).

**Repair2 outcome:** Current-contact recognition in163 removes both diagnosed errors, but complete cost fails.
- Cycle163; candidate `5ffd38ef2d283d1f402104e77c09bbfc8e832f49`; champion `19d5b8d17667205a8b8fdadc55b55a965cd5467d`; decision discard.
  [Implementation](ideas/square-root-sr1-weight/5ffd38ef2d283d1f402104e77c09bbfc8e832f49/algo.py).
  [Training evidence](evaluation_results/cycle-163-train.json).
  [Narrative](full_log.md#cycle-163-current-contact-recognition-in-collision-backtracking).

**square-root-sr1-weight final component control:**164 restores the champion squared-cosine weight while retaining the collision repair, and passes both gates. Thus the weight remains deferred/rejected; the independently useful guard is incorporated below. This completes the bounded sequence.

## pre-force-collision-backtracking: Realized geometry feasibility before force calls
Status: incorporated

**Hypothesis:** Reject a large newly formed nonbonded overlap before an expensive force calculation, while allowing existing/current chemical contacts to relax normally.
**Outcome and uncertainty:**162 removes the original SCF failure but stalls at a false O–H boundary.163 recognizes current contacts and fully converges, but its square-root update loses cost.164 isolates the geometry rule with the original update and passes both gates, becoming champion. Only three training and a few validation call counts change; broad robustness beyond these evaluations is unproven.
**Reason to revisit:** The accepted repair directly addresses saved heavy-atom overlaps in earlier promising rejected coordinate/angle variants.
**Next experiment:** Retain the accepted guard; test the original160 angle prior against164, without altering geometric thresholds or published stiffness constants.

**Attempts:**
- Cycle162; candidate `6b14589779ff5d55ce3f5a8205eea46777f0b5b1`; champion `19d5b8d17667205a8b8fdadc55b55a965cd5467d`; decision invalid.
  [Implementation](ideas/square-root-sr1-weight/6b14589779ff5d55ce3f5a8205eea46777f0b5b1/algo.py).
  [Training evidence](evaluation_results/cycle-162-train.json).
  [Narrative](full_log.md#cycle-162-pre-force-collision-backtracking-for-square-root-sr1).
- Cycle163; candidate `5ffd38ef2d283d1f402104e77c09bbfc8e832f49`; champion `19d5b8d17667205a8b8fdadc55b55a965cd5467d`; decision discard.
  [Implementation](ideas/square-root-sr1-weight/5ffd38ef2d283d1f402104e77c09bbfc8e832f49/algo.py).
  [Training evidence](evaluation_results/cycle-163-train.json).
  [Narrative](full_log.md#cycle-163-current-contact-recognition-in-collision-backtracking).
- Cycle164; candidate `ed8615cd901420f128424a78729a579e637028d7`; champion `19d5b8d17667205a8b8fdadc55b55a965cd5467d`; decision keep.
  Implementation retained as accepted Git commit.
  [Training evidence](evaluation_results/cycle-164-train.json).
  [Validation evidence](evaluation_results/cycle-164-valid.json).
  [Narrative](full_log.md#cycle-164-isolated-collision-backtracking-with-the-champion-update).

**Accepted consistency control172:** `f46f61edc66308b06e6c17ff02513444139cce43` propagates the accepted dyadic step fraction to trust-radius size feedback. Both gates pass, fully converged and error-free. [Train](evaluation_results/cycle-172-train.json), [valid](evaluation_results/cycle-172-valid.json), [narrative](full_log.md#cycle-172-accepted-collision-step-size-in-trust-feedback). Geometry thresholds and all no-backtrack paths are unchanged.

## swart-auxiliary-prior: Radius-screened auxiliary stiffness
Status: deferred

**Hypothesis:** Published radius-normalized pair stiffness could improve the element-size dependence of existing auxiliary contacts.
**Outcome and uncertainty:**170 converges fully with valid energy but broadly loses cost. This selected-pair adaptation coexists with explicit real-angle/torsion priors; it does not test the source's complete all-pair model.
**Reason to revisit:** A future coherent all-pair physical representation that replaces overlapping primitive terms might use the law without double counting. That requires an independently supported model decomposition.
**Next experiment:** No immediate decay/cutoff tuning; defer until such a decomposition is derived, then compare under full gates.
**Attempts:**
- Cycle170; candidate `dced17a93a915b7e7ac0446ea5eb980a7f377df0`; champion `02a03f929ee5c99ff8e2085c1a742484e750b42a`; decision discard.
  [Implementation](ideas/swart-auxiliary-prior/dced17a93a915b7e7ac0446ea5eb980a7f377df0/algo.py), [train](evaluation_results/cycle-170-train.json), [narrative](full_log.md#cycle-170-radius-screened-swart-auxiliary-pair-stiffness).

## fixed-physical-prior: Fixed-weight physical model control
Status: deferred

**Hypothesis:** Improved published primitive priors might remove the need for early coefficient fitting.
**Outcome and uncertainty:**171 fully converges and passes energy but loses cost. Learned coefficients remain useful for this model; the control also fixes the signed coupling to its initial zero coefficient.
**Reason to revisit:** This ablation provides a clean fixed-model control for a future independently derived physical prior or calibration-bias treatment.
**Next experiment:** No calibration-count tuning. Reuse only if the prior or model identification changes enough to justify a new control.
**Attempts:**
- Cycle171; candidate `bc376ac459904d23c08b3fa33bbe5ad2f79994a2`; champion `02a03f929ee5c99ff8e2085c1a742484e750b42a`; decision discard.
  [Implementation](ideas/fixed-physical-prior/bc376ac459904d23c08b3fa33bbe5ad2f79994a2/algo.py), [train](evaluation_results/cycle-171-train.json), [narrative](full_log.md#cycle-171-fixed-published-physical-prior-without-coefficient-fitting).

## dense-distance-curvature: Replace torsional priors by all-pair radial stiffness
Status: deferred

**Hypothesis:** Dense nonbonded distance springs can supply correlated torsional/umbrella/long-range curvature without overlapping explicit priors.
**Outcome and uncertainty:**173 fully converges and passes energy but substantially loses cost. No passive failure path identifies a specific repair. Near-planar radial rank and overly stiff collective motion are limitations, not proven causal diagnoses.
**Reason to revisit:** A separately derived model with controlled transverse curvature and collective response might use the dense pullback. The tested hybrid omits the source's original angular model and retains accepted bond/angle laws.
**Next experiment:** Defer; do not add an arbitrary torsion floor or fit pair decay to these endpoints.
**Attempts:**
- Cycle173; candidate `9cb437d06ed748d19f371072db5aee6522d74b67`; champion `f46f61edc66308b06e6c17ff02513444139cce43`; decision discard.
  [Implementation](ideas/dense-distance-curvature/9cb437d06ed748d19f371072db5aee6522d74b67/algo.py), [train](evaluation_results/cycle-173-train.json), [narrative](full_log.md#cycle-173-dense-distance-spring-replacement-of-torsional-model-curvature).

**364 complete-sector control:**173 omitted the source angular model. Test paper47–52 radial/angular blocks jointly within each covalent component while preserving307 collective stiffness; source-defined linear/near-zero handling addresses the transverse prerequisite. This is a full intrinsic-model replacement, not radial decay/cutoff tuning. Source code/paper branch discrepancy is documented in the log; implement the paper.

**364 reassessment:**The complete source radial/angular intrinsic model with accepted collective response converges fully and passes energy but loses cost substantially. The published-code branch discrepancy is corrected according to the paper; no force calculation error remains to repair. The tested completion does not satisfy the performance hypothesis. Revisit only with independent mode-quality evidence, not constants/radii/angle/mixture sweeps.
- Cycle364; candidate `69b1d2f539f458b462aad5b21be534a53612e66a`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/dense-distance-curvature/69b1d2f539f458b462aad5b21be534a53612e66a/algo.py), [train](evaluation_results/cycle-364-train.json), [narrative](full_log.md#cycle-364-complete-modified-swart-intrafragment-prior).

## collective-torsion-curvature: Common bridge rotation without torsional contrasts
Status: deferred

**Hypothesis:** A rank-one torsional block preserves common bridge rotation stiffness while leaving deformation contrasts to angle and umbrella priors.
**Outcome and uncertainty:**175 fully converges with valid energy but loses cost. No failure trace identifies missing physical contrast modes.
**Reason to revisit:** The common-mode decomposition may be useful if independent information identifies which contrast modes deserve separate curvature; the exact common-action identity is reusable.
**Next experiment:** Defer without an arbitrary blend or contrast stiffness; derive the missing mode model before another attempt.
**Attempts:**
- Cycle175; candidate `368ec64d6f6f1fb76a460111e3d0ed54326a0521`; champion `ece05361773e587139cedc78c9528bdd567d7775`; decision discard.
  [Implementation](ideas/collective-torsion-curvature/368ec64d6f6f1fb76a460111e3d0ed54326a0521/algo.py), [train](evaluation_results/cycle-175-train.json), [narrative](full_log.md#cycle-175-collective-bridge-torsion-curvature-with-preserved-common-stiffness).

## secant-restored-positive-metric: Restore the observed secant after reflection
Status: deferred

**Hypothesis:** A positive BFGS correction to the reflected stepping Hessian preserves the latest positive-curvature geodesic secant.
**Outcome and uncertainty:**176 improves training cost but loses validation cost; both splits fully converge and pass energy. No passive model-state evidence identifies an error to repair.
**Reason to revisit:** An independently supported finite-step transport or constraint-curvature model could improve the secant supplied to this reusable algebraic restoration.
**Next experiment:** Defer until that evidence exists; no restoration damping or eigenvalue threshold sweep.
**Attempts:**
- Cycle176; candidate `25afaecaf49557784fb2766166e671368fe69061`; champion `ece05361773e587139cedc78c9528bdd567d7775`; decision non_generalizable.
  [Implementation](ideas/secant-restored-positive-metric/25afaecaf49557784fb2766166e671368fe69061/algo.py), [train](evaluation_results/cycle-176-train.json), [valid](evaluation_results/cycle-176-valid.json), [narrative](full_log.md#cycle-176-restore-positive-secants-after-stepping-metric-reflection).

**Cycle202 prerequisite:**Accepted182/186 changes the nonlinear chart and finite-step connection independently of restoration, satisfying the documented transport condition. Test176 unchanged against186; no damping or threshold tuning.

**Cycle202 outcome:**The accepted182/186 path does not rescue restoration: fully converged energy-valid training loses cost. Defer pending independent evidence distinguishing averaged secants from local reflected curvature; no strength/threshold sweep.
- Cycle202; candidate `db5f3b4afe57f9ebe155be3fc8fa97b40c972824`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision discard.
  [Implementation](ideas/secant-restored-positive-metric/db5f3b4afe57f9ebe155be3fc8fa97b40c972824/algo.py), [train](evaluation_results/cycle-202-train.json), [narrative](full_log.md#cycle-202-positive-secant-restoration-with-the-current-connection).

## cosine-bend-coordinates: Initially matched cosine bending chart
Status: deferred

**Hypothesis:** An affine cosine angle chart can change finite geodesic and surrogate motion while matching the initial tangent and force constants.
**Outcome and uncertainty:**179 fully converges without errors but loses cost and fails aggregate energy. No failed path identifies a chain-rule or singularity defect; this does not test every intrinsic bending representation.
**Reason to revisit:** A separately derived finite-history transport or near-linear coordinate model could make this smooth chart useful. Its consistent value/Jacobian/Hessian chain rule is reusable.
**Next experiment:** Require that geometric mechanism or passive failure evidence before changing the representation; no scale, threshold or activation sweep from endpoint loss.

**Attempts:**
- Cycle179; candidate `03ae5ed6d685bd66abf2b370099fae0848fba328`; champion `4807cb0fb347c39d9c849ada2e3f404d9e557930`; decision invalid.
  [Implementation](ideas/cosine-bend-coordinates/03ae5ed6d685bd66abf2b370099fae0848fba328/algo.py).
  [Training evidence](evaluation_results/cycle-179-train.json).
  [Narrative](full_log.md#cycle-179-initially-matched-cosine-bend-native-coordinates).


## nonstationary-chart-prior: Gradient-dependent starting curvature
Status: deferred

**Hypothesis:** Include the known nonlinear-coordinate second derivative of the starting model's linear energy term, retaining it as a fixed background in physical fitting/replay.
**Outcome and uncertainty:**183 passes energy and converges all molecules without errors but loses training cost. Algebraic consistency for the chosen redundant primitive extension does not establish an improved empirical Hessian.
**Reason to revisit:** An independently supported raw-coordinate energy model or measured curvature transformation defect could make this exact chain-rule component useful.
**Next experiment:** Require that model evidence before another correction; no background strength or projector threshold sweep.

**Attempts:**
- Cycle183; candidate `70e198cdb2e0d8bfe9a567dcdc349b7b67bec872`; champion `3fbddaadd7fef774f7b0effe823c962e6334376d`; decision discard.
  [Implementation](ideas/nonstationary-chart-prior/70e198cdb2e0d8bfe9a567dcdc349b7b67bec872/algo.py).
  [Training evidence](evaluation_results/cycle-183-train.json).
  [Narrative](full_log.md#cycle-183-nonstationary-initial-hessian-in-the-badger-chart).

## signed-quadratic-trust: Negative curvature within the ordinary norm budget
Status: deferred

**Hypothesis:** A global signed-quadratic trust-ball solution can exploit negative curvature while staying inside the ordinary free-step norm and existing component bounds.
**Outcome and uncertainty:**184 fully converges and passes energy without errors but loses training cost. No solver/trajectory failure identifies a repair; a better approximate quadratic need not predict a better physical step.
**Reason to revisit:** Independently measured reliability of negative modes or a better nonlinear acceptance model could make the hard-case-aware solver useful.
**Next experiment:** Require that model evidence; no radius, eigenvalue or prediction-margin sweep.

**Attempts:**
- Cycle184; candidate `1cb49dc62c28501ddcde4e20d91870819b18de5b`; champion `3fbddaadd7fef774f7b0effe823c962e6334376d`; decision discard.
  [Implementation](ideas/signed-quadratic-trust/1cb49dc62c28501ddcde4e20d91870819b18de5b/algo.py).
  [Training evidence](evaluation_results/cycle-184-train.json).
  [Narrative](full_log.md#cycle-184-signed-quadratic-trust-ball-at-the-ordinary-motion-budget).

## local-angle-trust: Bound common and contrast bending motion
Status: deferred

**Hypothesis:** A center-local angular norm can control concerted bending and redistribution more usefully than independent primitive limits.
**Outcome and uncertainty:**185 passes energy and fully converges without errors but loses training cost. The coherent trust/GP norm does not show benefit; endpoint results do not diagnose excessive common versus contrast restriction.
**Reason to revisit:** A supported local-mode curvature or uncertainty model could distinguish admissible common/contrast motion; the basis-independent norm and analytic derivative are reusable.
**Next experiment:** Require such mode-specific evidence; no group scaling, radius or chemistry activation sweep.

**Attempts:**
- Cycle185; candidate `85c861994755d78259c057ede60d760783e5d03a`; champion `3fbddaadd7fef774f7b0effe823c962e6334376d`; decision discard.
  [Implementation](ideas/local-angle-trust/85c861994755d78259c057ede60d760783e5d03a/algo.py).
  [Training evidence](evaluation_results/cycle-185-train.json).
  [Narrative](full_log.md#cycle-185-common-and-contrast-local-angle-trust-bounds).

## physical-secant-congruence: Complete matrix updates in fixed physical units
Status: deferred

**Hypothesis:** Apply the entire accepted TS/SR1 matrix update under a fixed primitive-stiffness congruence, balancing stiff and soft coupling without changing geometry or calibration loss.
**Outcome and uncertainty:**187 fully converges with valid energy and zero errors, but loses training cost. No passive failure or model state identifies a numerical repair; all fixed diagonal physical metrics are not ruled out.
**Reason to revisit:** An independently supported model of correction covariance or physical coupling could replace the diagonal approximation with a derived least-change metric. The complete primal/dual congruence is reusable.
**Next experiment:** Require that independent evidence before another metric. No diagonal floor, strength or coordinate-class sweep from this broad loss.

**Attempts:**
- Cycle187; candidate `d9854802da18fbfda9cffaeea5b90c1858b1f379`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision discard.
  [Implementation](ideas/physical-secant-congruence/d9854802da18fbfda9cffaeea5b90c1858b1f379/algo.py), [train](evaluation_results/cycle-187-train.json), [narrative](full_log.md#cycle-187-physical-congruence-for-complete-secant-updates).

## angle-regularized-torsion-prior: Cancel near-linear torsional prior singularity
Status: deferred

**Hypothesis:**Multiply real proper-torsion prior stiffness by the squared sines of both contained angles, bounding its Cartesian Hessian response near collinearity.
**Outcome and uncertainty:**190 fully converges, passes energy and has zero errors, but loses training cost. The geometric regularization is not enough to improve the accepted empirical prior; successful-path stiffness states are unavailable.
**Reason to revisit:**A separately derived regular torsional representation or a physical angular/torsional model could use bounded plane-area response consistently, rather than changing the stiffness alone.
**Next experiment:**Require that model or passive singularity evidence; no exponent, strength or angle-activation sweep from the present loss.

**Attempts:**
- Cycle190; candidate `a30f2fd267b24c52f6ffae9e0948a2c1791f24c7`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision discard.
  [Implementation](ideas/angle-regularized-torsion-prior/a30f2fd267b24c52f6ffae9e0948a2c1791f24c7/algo.py), [train](evaluation_results/cycle-190-train.json), [narrative](full_log.md#cycle-190-bounded-cartesian-response-from-proper-torsion-priors).

**347 prerequisite:**A smooth bond-plane invariant pair now replaces raw proper torsions while matching their initial tangent and spring response. This supplies the independently derived regular representation requested above; initial priors are retained rather than damped. Exact construction/risks are in cycle347.

**347 reassessment:**The independently derived smooth invariant representation retains initial stiffness and fully converges with valid energy, but loses cost. Its implementation is indexed under [smooth-bond-plane-torsions](#smooth-bond-plane-torsions). This prerequisite did not establish a benefit; no damping/coordinate/threshold sweep follows.

## triangle-prior-redundancy: Avoid duplicate triangle-angle springs
Status: deferred

**Hypothesis:**Omit triangle-angle prior diagonals because the existing three bond lengths determine each angle, retaining all geometric coordinates.
**Outcome and uncertainty:**191 fully converges without errors but fails energy and loses cost. Geometric dependence does not establish physical double counting; omitted angular stiffness or its signed coupling may remain useful.
**Reason to revisit:**An independently derived transformation of triangle bending curvature into a coupled bond model could retain its physical content while reducing redundant parameterization. Simple deletion is unsupported.
**Next experiment:**Require that equivalent coupled model and a reason it changes calibration usefully; no partial damping or ring-size sweep.

**Attempts:**
- Cycle191; candidate `568476d8ce6e6878bb546c12e4fb77608878f8e2`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision invalid.
  [Implementation](ideas/triangle-prior-redundancy/568476d8ce6e6878bb546c12e4fb77608878f8e2/algo.py), [train](evaluation_results/cycle-191-train.json), [narrative](full_log.md#cycle-191-remove-dependent-triangle-angle-prior-springs).

**Revisit287:**Retain every initial physical feature through the exact tangent map from dependent triangle angles to their three active bond lengths, then remove only those geometric rows. This supplies191's required equivalent coupled model and isolates reduced observation/geometry representation from physical stiffness deletion. No ring-size, stiffness or cutoff tuning.

**287 reassessment:**Equivalent initial physical-curvature transfer plus geometric row reduction converges fully and lowers cost but fails energy. Almost all cost gain comes from one inferior converged endpoint; empty passive manifest cannot diagnose its path. Require an independently established coordinate-invariant physical step-bound/model formulation before revisiting, without triangle/rank/damping sweeps.
- Cycle287; candidate `5c14c3d3d55fe585eee02591245c4b0bd00b2abb`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision invalid.
  [Implementation](ideas/triangle-prior-redundancy/5c14c3d3d55fe585eee02591245c4b0bd00b2abb/algo.py), [train](evaluation_results/cycle-287-train.json), [manifest](diagnostics/e58a09c658294e429d6d00132782c703/manifest.json), [narrative](full_log.md#cycle-287-triangle-coordinate-reduction-with-preserved-physical-curvature).

## contact-constrained-secant: Local support with exact newest secants
Status: deferred

**Hypothesis:** Project each new accepted matrix correction onto contact-supported symmetric entries while enforcing its secant, avoiding unsupported long-range learning without erasing old memory.
**Outcome and uncertainty:**192 passes energy without errors but loses broadly (67 faster/214 slower) and adds two force-call limits. Complete scalar traces show residual-force relaxation, no exception; internal projection conditioning is unobserved. Distinct from133's repeated stored-memory attenuation.
**Reason to revisit:** A derived tangent-compatible sparse model or independent correction-covariance evidence could replace the initial atom-contact approximation; exact constrained projection is reusable.
**Next experiment:** Require that structural evidence; no contact-radius or regularization sweep from this broad loss.

**Attempts:**
- Cycle192; candidate `1042b508e0bc591db59192e911f835b5cae5cbdf`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision discard.
  [Implementation](ideas/contact-constrained-secant/1042b508e0bc591db59192e911f835b5cae5cbdf/algo.py), [train](evaluation_results/cycle-192-train.json), [diagnostics](diagnostics/3ca4850539074a6983f9fca2115b79e1/manifest.json), [narrative](full_log.md#cycle-192-contact-supported-exact-secant-corrections).

**446 structural revisit:** Replace raw native-entry contact masks by physically normalized atomic element factors projected into the current tangent before imposing the total secant. This supplies192's explicit tangent-compatible-model prerequisite; no distance cutoff or sparsity-strength sweep. Latent element Hessians are inferred jointly from aggregate forces, with unavailable element gradients left explicit. Full derivation and limitations are in cycle446 narrative.

**446 reassessment:** The tangent-compatible atomic covariance loses cost broadly with467converged,2calllimits,validenergy andzeroerrors. The available trace shows ongoing relaxation; no element/Hessian states diagnose a repair. This resolves the stated structural prerequisite negatively for the tested aggregate-force least-change construction. Revisit only with independent evidence for element force decomposition or missing nonlocal curvature; no support,normalization,rank,cutoff orblend sweeps.
- Cycle446; candidate `304eeb8ce04e5080e9762991ed77ff950167494c`; champion `bb770d9c7c9e59d39007f71a106af77cd9eb73a1`; decision discard. [Implementation](ideas/contact-constrained-secant/304eeb8ce04e5080e9762991ed77ff950167494c/algo.py), [train](evaluation_results/cycle-446-train.json), [manifest](diagnostics/ceb4770db6534c7886172c9580289feb/manifest.json), [narrative](full_log.md#cycle-446-tangent-compatible-atomic-secant-covariance).

## chemical-reference-badger: Normalize nonlinear bond geometry at chemical lengths
Status: deferred

**Hypothesis:**Set the Badger-chart reference to covalent-radius sums, making its physical metric independent of supplied initial bond distortion.
**Outcome and uncertainty:**193 fully converges with valid energy and no errors but loses training cost. Chemical references need not match local equilibrium; the initial-scale change remains entangled with trust and secant norms.
**Reason to revisit:**A chart-invariant trust/update construction could distinguish metric conditioning from unintended coordinate-scale effects; this reference adapter is reusable.
**Next experiment:**Require that independent formulation; no reference interpolation, exponent or normalization sweep.

**Attempts:**
- Cycle193; candidate `f6f621aecb83b546107e18d71e7a8358bb5cece9`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision discard.
  [Implementation](ideas/chemical-reference-badger/f6f621aecb83b546107e18d71e7a8358bb5cece9/algo.py), [train](evaluation_results/cycle-193-train.json), [narrative](full_log.md#cycle-193-chemical-reference-badger-bond-coordinates).

## energy-simplex-proposals: Energy-based barycentric history search
Status: deferred

**Hypothesis:**Optimize quadratic-exact barycentric energy over the history simplex, then apply one current physical correction, using the accepted GP as common chart and final guard.
**Outcome and uncertainty:**195 fully converges with valid energy and zero errors but loses training cost. The finite face solver avoids arbitrary coefficient iterations; posterior correction accuracy and interpolation error are unobserved.
**Reason to revisit:**A separately derived nonlinear history-interpolation model or reliable curvature correction could make the constrained energy representation useful beyond this one-step approximation.
**Next experiment:**Require that model/evidence; no coefficient, activation or correction-count sweep.

**Attempts:**
- Cycle195; candidate `ea11ccfe0b3e1db8e4679e2045c622f2619e9ef3`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision discard.
  [Implementation](ideas/energy-simplex-proposals/ea11ccfe0b3e1db8e4679e2045c622f2619e9ef3/algo.py), [train](evaluation_results/cycle-195-train.json), [narrative](full_log.md#cycle-195-energy-simplex-surrogate-proposals).

## uff-bending-prior: Chemical and geometric bending stiffness
Status: deferred

**Hypothesis:**Use the UFF three-center effective-charge bending law at actual initial geometry to distinguish stiffness beyond Schlegel's two classes.
**Outcome and uncertainty:**197 fully converges with valid energy and zero errors but loses cost. Current geometry replaces typed equilibrium data; physical-prior quality and interaction with shared calibration remain unresolved.
**Reason to revisit:**An independently justified equilibrium-geometry estimate or complete coupled valence model could make this stiffness law useful without distorted-geometry bias.
**Next experiment:**Require that model/evidence; no scale, angle-threshold or element-activation sweep.

**Attempts:**
- Cycle197; candidate `3d4d0b1ab31e25426ebdf5eacc39fa5a2a425ac6`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision discard.
  [Implementation](ideas/uff-bending-prior/3d4d0b1ab31e25426ebdf5eacc39fa5a2a425ac6/algo.py), [train](evaluation_results/cycle-197-train.json), [narrative](full_log.md#cycle-197-geometry-adapted-uff-bending-prior).

## curvature-preserving-conic: Energy-interpolating rational proposals
Status: deferred

**Hypothesis:**A conic model preserving the current physical gradient/Hessian and interpolating one historical energy can produce a useful analytic stationary proposal under the accepted GP guards.
**Outcome and uncertainty:**198 fully converges with valid energy and no errors but loses training cost. No horizon/pole instability is observed; fit activation and successful-path model quality are unrecorded.
**Reason to revisit:**An independently supported horizon estimate from a consistent multipoint model could supply directional anharmonicity beyond the one-displacement assumption, reusing the exact Taylor correction and branch-safe minimizer.
**Next experiment:**Require that derivation/evidence; no root, guard or interpolation-window sweep from this broad loss.

**Attempts:**
- Cycle198; candidate `4f3dee57bfd91c46703a83fd76d1464e203556dd`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision discard.
  [Implementation](ideas/curvature-preserving-conic/4f3dee57bfd91c46703a83fd76d1464e203556dd/algo.py), [train](evaluation_results/cycle-198-train.json), [narrative](full_log.md#cycle-198-curvature-preserving-conic-proposals).

**Revisit284:**Replace the one-displacement energy root by a bounded full-span nonlinear horizon fit to all existing retained energies and mapped gradients. Exact current Taylor data and the GP admission model stay fixed. This derives the multipoint observation relation requested after198, with analytic fit Jacobians and positive-branch/positive-matrix domain guarantees.

**284 reassessment:**A bounded multipoint fit to saved energies and mapped gradients meets the prior horizon-identification prerequisite but still loses cost, with full convergence, valid energy and no errors. Defer until independently demonstrated local rational-model accuracy; no fit-weight/horizon-radius/microbudget/window sweeps.
- Cycle284; candidate `efd6861b596f88e2f9e1576bc598c69b0937b45d`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision discard.
  [Implementation](ideas/curvature-preserving-conic/efd6861b596f88e2f9e1576bc598c69b0937b45d/algo.py), [train](evaluation_results/cycle-284-train.json), [narrative](full_log.md#cycle-284-multipoint-energy-gradient-conic-proposals).

## directional-physical-calibration: Fit physical scales from directional work
Status: deferred

**Hypothesis:** Fit the few physical coefficients to scalar step curvature, leaving transverse responses to full secant replay.
**Outcome and uncertainty:** Cycle200 fully converges with valid energy but loses cost. Projection discards information and weakens data relative to ridge; no instability supports a repair.
**Reason to revisit:** An independently supported observation covariance could separate unmodeled transverse coupling from informative physical-scale responses.
**Next experiment:** Await that model; no ridge, window or projection-mixture sweep.

**Attempts:**
- Cycle200; candidate `7a108f45f98ec4b5c3c5a9acbc1b6b8f0377e18c`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision discard.
  [Implementation](ideas/directional-physical-calibration/7a108f45f98ec4b5c3c5a9acbc1b6b8f0377e18c/algo.py).
  [Training evidence](evaluation_results/cycle-200-train.json).
  [Narrative](full_log.md#cycle-200-directional-curvature-physical-calibration).

## residual-stabilized-gp: Off-span residual curvature in GP scaling
Status: deferred

**Hypothesis:**Stabilize projected GP curvature against coupling to omitted free directions.
**Outcome and uncertainty:**203 fully converges but fails energy and broadly worsens cost. Mean and kernel embedding change together; no failed paths diagnose a repair.
**Reason to revisit:**An independently supported separation of the GP prior mean from its kernel metric could isolate which role should use residual coupling.
**Next experiment:**Await that model; no residual-strength, floor or activation sweep.
**Attempts:**
- Cycle203; candidate `95497f97ba66f6b31dfe57ed6264f340f8d051f3`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision invalid.
  [Implementation](ideas/residual-stabilized-gp/95497f97ba66f6b31dfe57ed6264f340f8d051f3/algo.py), [train](evaluation_results/cycle-203-train.json), [narrative](full_log.md#cycle-203-residual-stabilized-surrogate-curvature).

## stabilized-subspace-learning: Physical-prior direct Ritz batch learner
Status: deferred

**Hypothesis:**Joint normalized recent secants with residual-stabilized curvature can replace sequential rank-two learning while retaining a physical complement.
**Outcome and uncertainty:**204 fails energy and broadly loses cost, with41 call limits and no errors. Two passive traces show persistent low-force motion; no saved model/history state separates forgetting, tangent inconsistency or complement stiffness.
**Reason to revisit:**An independently established common-frame observation model and a derivation preserving accumulated complementary curvature might address the information loss; current traces do not justify a parameter repair.
**Next experiment:**Await that mechanism; no memory, rank, floor or stopping-step sweep.
**Attempts:**
- Cycle204; candidate `a0f1e6e611db9928ef8eccccdb4851cf17cdfaea`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision invalid.
  [Implementation](ideas/stabilized-subspace-learning/a0f1e6e611db9928ef8eccccdb4851cf17cdfaea/algo.py), [train](evaluation_results/cycle-204-train.json), [narrative](full_log.md#cycle-204-physical-prior-stabilized-subspace-learning).
  [Diagnostic manifest](diagnostics/3df70da8e1264c2db6e900c9fbfbb3e0/manifest.json).

## rank-preserving-torsion-reduction: Reduce redundant proper bridge torsions
Status: deferred

**Hypothesis:**A stable representative and rank-required extra torsions provide a less redundant finite geometry chart.
**Outcome and uncertainty:**205 invalid with192 ODE-limit exceptions,148 after one force call; returned subset also more often slower. Passive traces locate failures inside geometry integration but lack rank/solver states. Initial rank is insufficient evidence for finite-path regularity.
**Reason to revisit:**A derived chart construction retaining out-of-plane regularity throughout planar limits, or a robust rank-changing realization method, could make reduction useful; current rank/order selection is not supported.
**Next experiment:**Await that construction; no rank/order threshold or ODE-budget sweep.
**Attempts:**
- Cycle205; candidate `60777668643acbcae0441bf3b3ba74104a9b4dcf`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision invalid.
  [Implementation](ideas/rank-preserving-torsion-reduction/60777668643acbcae0441bf3b3ba74104a9b4dcf/algo.py), [train](evaluation_results/cycle-205-train.json), [narrative](full_log.md#cycle-205-rank-preserving-acyclic-torsion-reduction).
  [Diagnostic manifest](diagnostics/3d5d1ba9097543d19f24fbd22d14421f/manifest.json).

## separate-collective-calibration: Separate translation and rotation coefficients
Status: deferred

**Hypothesis:**Distinct collective physical responses need independent fitted correction scales.
**Outcome and uncertainty:**206 fully converges with valid energy but loses cost. No fit-state observations identify whether extra contrast is poorly identified.
**Reason to revisit:**A separately supported collective observation model could identify independent translational and rotational errors without weakening inference.
**Next experiment:**Await that evidence; no ridge/bounds/class activation sweep.
**Attempts:**
- Cycle206; candidate `8ef837ff9a660a39c298383e8ba5fa7b037b7563`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision discard.
  [Implementation](ideas/separate-collective-calibration/8ef837ff9a660a39c298383e8ba5fa7b037b7563/algo.py), [train](evaluation_results/cycle-206-train.json), [narrative](full_log.md#cycle-206-separate-translation-and-rotation-calibration).

## category-aware-bend-calibration: Separate hydrogen and heavier bend corrections
Status: deferred

**Hypothesis:**The two published Schlegel stiffness categories may have distinct remaining model errors.
**Outcome and uncertainty:**207 fully converges with valid energy but loses cost. Six observations may not identify the extra contrast; no fit-state diagnostics distinguish this from unnecessary flexibility.
**Reason to revisit:**An independently supported angular observation model that identifies category-dependent errors could make the additional coefficient useful.
**Next experiment:**Await that evidence; no category/ridge/bounds sweep.
**Attempts:**
- Cycle207; candidate `102e6a2be6556e17719867e2099faf07d6816118`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision discard.
  [Implementation](ideas/category-aware-bend-calibration/102e6a2be6556e17719867e2099faf07d6816118/algo.py), [train](evaluation_results/cycle-207-train.json), [narrative](full_log.md#cycle-207-category-aware-schlegel-bend-calibration).

## transverse-intrafragment-curvature: Full positive contact tensors inside fragments
Status: deferred

**Hypothesis:**Transverse curvature omitted from radial auxiliary springs could improve angular/contact conditioning.
**Outcome and uncertainty:**208 fully converges with valid energy but broad cost loss. The repulsive-potential inference may add inappropriate angular stiffness; no instability diagnoses a local repair.
**Reason to revisit:**An independently supported partition between contact geometry and existing bending/torsion priors could avoid double counting and specify the actual missing tensor response.
**Next experiment:**Await that model; no tensor-strength, cutoff or exponent sweep.
**Attempts:**
- Cycle208; candidate `7f72a5868ebe36eb24cb95aa5903b166f68818f3`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision discard.
  [Implementation](ideas/transverse-intrafragment-curvature/7f72a5868ebe36eb24cb95aa5903b166f68818f3/algo.py), [train](evaluation_results/cycle-208-train.json), [narrative](full_log.md#cycle-208-transverse-intrafragment-contact-curvature).

## two-step-endpoint-curvature: Vector endpoint inference from two paid secants
Status: deferred

**Hypothesis:**Quadratic interpolation of three gradients/iterates gives a better latest Hessian action than an averaged secant.
**Outcome and uncertainty:**209 fully converges with valid energy but loses cost. Two historical vectors are transported along the accepted connection; matrix/replay chart treatment remains approximate and added ODE rows may alter adaptivity. No failure/model state diagnoses a repair.
**Reason to revisit:**An independently established covariant model/replay scheme or reliable path-parameter model might make higher-order observations useful without losing robust average curvature.
**Next experiment:**Await that mechanism; no knot, order, interpolation-weight or activation sweep.
**Attempts:**
- Cycle209; candidate `2c2af3271eecb3062a11c245fad6d51785b57e45`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision discard.
  [Implementation](ideas/two-step-endpoint-curvature/2c2af3271eecb3062a11c245fad6d51785b57e45/algo.py), [train](evaluation_results/cycle-209-train.json), [narrative](full_log.md#cycle-209-transported-two-step-endpoint-curvature).

**Revisit557:** Fixed GP surrogate removes209's moving-vector/matrix replay and ODE-adaptivity uncertainty. Three exact gradients/iterates share one flat chart; chord-length parameterization gives a derived endpoint pair. The old physical loss remains counterevidence, not a diagnosed repair. No knot/order/blend sweep.

**Completed557:** Exact fixedGP chart still loses cost, all469converged/validenergy/zeroerrors;{'faster': 4, 'same': 458, 'slower': 7}. No diagnosed repair. Revisit only with direct evidence that endpoint inference improves predictivecurvature; no generic chartchange or knot/order/weight/budget sweep.
- Cycle557; candidate `972eb283ead293494f897b06b9d867d21430e683`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/two-step-endpoint-curvature/972eb283ead293494f897b06b9d867d21430e683/algo.py), [training](evaluation_results/cycle-557-train.json), [narrative](full_log.md#cycle-557-two-step-curve-secants-in-surrogate-bfgs).

## stretch-derived-bending: Geometric-mean bond strength defines bending stiffness
Status: deferred

**Hypothesis:**The published bending/stretching relation with accepted Badger bonds supplies better continuous angular stiffness.
**Outcome and uncertainty:**210 fully converges with valid energy but loses cost. The source's fixed ratio may not transfer from its own radial model; no failure state identifies a repair.
**Reason to revisit:**A separately justified coupled radial/angular model could support a consistent relation rather than this hybrid.
**Next experiment:**Await that model; no bending-ratio, exponent or element-class sweep.
**Attempts:**
- Cycle210; candidate `4facff868b705f68a2b18b53b4c4a1ba24e6cda8`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision discard.
  [Implementation](ideas/stretch-derived-bending/4facff868b705f68a2b18b53b4c4a1ba24e6cda8/algo.py), [train](evaluation_results/cycle-210-train.json), [narrative](full_log.md#cycle-210-stretch-derived-bending-stiffness).

**Cycle211 cubic-controller cross-reference:**The complete-controller prerequisite of [secant-cubic-damping](#secant-cubic-damping-model-error-calibrated-cubic-newton-steps) was tested and rejected on energy/cost; [design](full_log.md#cycle-211-adaptive-cubic-macro-optimizer). It replaces the heuristic sigma estimate with accepted/rejected ARC updates and retains the independently accepted186 geometry.

## continuous-torsional-prior: Smooth positive continuation of Schlegel torsional stiffness
Status: deferred

**Hypothesis:**C1 matching at the covalent length removes an abrupt stiffness reset and improves initial-model consistency.
**Outcome and uncertainty:**213 fully converges with valid energy but loses cost. Continuous geometry dependence alone is insufficient; over-softened long bonds may contribute, but no model trace isolates it.212 was unchanged-source administrative evidence, not a test.
**Reason to revisit:**An independently supported torsional law with a physical long-bond limit could provide continuity without unsupported exponential softening.
**Next experiment:**Await that law; no transition/decay/floor sweep.
**Attempts:**
- Cycle213; candidate `3e2336cd18ebe27682b01c3798f7588b5eddbb96`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision discard.
  [Implementation](ideas/continuous-torsional-prior/3e2336cd18ebe27682b01c3798f7588b5eddbb96/algo.py), [train](evaluation_results/cycle-213-train.json), [narrative](full_log.md#cycle-213-continuous-positive-torsional-prior).

## correlated-physical-observations: Shared directional discrepancy in early calibration
Status: deferred

**Hypothesis:**Correlated normalized secant errors avoid overcounting similar step directions while retaining all outputs.
**Outcome and uncertainty:**214 fully converges with valid energy but loses cost. Equal shared/local error variance was a modeling assumption; no saved fit state supports tuning it.
**Reason to revisit:**Independent observations of residual correlation or a covariant discrepancy model could justify a more faithful correlation structure.
**Next experiment:**Await that evidence; no covariance-strength, noise or history-count sweep.
**Attempts:**
- Cycle214; candidate `b7fb79d3cfbad9636287325c5994ce9cd14f8171`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision discard.
  [Implementation](ideas/correlated-physical-observations/b7fb79d3cfbad9636287325c5994ce9cd14f8171/algo.py), [train](evaluation_results/cycle-214-train.json), [narrative](full_log.md#cycle-214-correlated-physical-calibration-observations).

**371 structural revisit:**Replace214's nonsymmetric latent response by a symmetrized Gaussian matrix, standardizing each output marginal back to identity. Preserve its equal independent/shared mixture. Reciprocal-component covariance follows from symmetry and survives orthogonal step directions. Exact span/complement reduction needs at most a36x36 solve; this supplies the previously deferred covariance derivation, not empirical noise identification.

**371 outcome:**Fully converged and energy-valid but slower. Symmetry alone does not supply a useful discrepancy prior. Require independent calibration residual/transport evidence before further covariance changes; no noise, rank, mixture or schedule sweep.
- Cycle371; candidate `d59ba66a5c716699bbfd923101e09592904f260b`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/correlated-physical-observations/d59ba66a5c716699bbfd923101e09592904f260b/algo.py), [train](evaluation_results/cycle-371-train.json), [narrative](full_log.md#cycle-371-symmetric-response-covariance-in-physical-calibration).

## developed-tangent-gp: Local GP on transported path development
Status: deferred

**Hypothesis:**Use current-tangent developed displacements and parallel gradients to reduce chord and changing-tangent artifacts in nonlinear GP proposals.
**Outcome and uncertainty:**217 fully converges with valid energy and zero errors but loses training cost. Transported gradients are only approximate derivatives of the flat-development scalar model; path dependence and ODE adaptivity remain unmeasured.
**Reason to revisit:**An independently derived finite-curvature pullback or diagnosed inconsistency could support this reusable small-vector transport/history carrier.
**Next experiment:**Await that mechanism; no history, noise or transport-tolerance sweep. Compare any new geometry-aware observation model under both original gates.
**Attempts:**
- Cycle217; candidate `b8368d5fc343d973f98e7324fd8edcad8522a25c`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision discard.
  [Implementation](ideas/developed-tangent-gp/b8368d5fc343d973f98e7324fd8edcad8522a25c/algo.py), [train](evaluation_results/cycle-217-train.json), [narrative](full_log.md#cycle-217-developed-tangent-gp-history).

**Related303 control:**A source-derived cubic logarithm supplies a finite-curvature scalar-chart differential, unlike transported path-development gradients. It also slightly loses training cost with complete valid convergence; see [geodesic-log-gp](#geodesic-log-gp-cubic-geodesic-logarithm-and-its-differential). This does not support an immediate transport or expansion-order repair.

## effective-charge-bond-prior: Element-resolved UFF radial stiffness
Status: deferred

**Hypothesis:**Published UFF effective charges distinguish within-period bond stiffness and improve early metric quality.
**Outcome and uncertainty:**218 fully converges and passes energy without errors but loses cost. The current actual bond length replaces UFF equilibrium lengths, and the accepted chart still follows Badger.
**Reason to revisit:**An independently supported equilibrium-length model or coherent metric/prior construction could remove those limitations; the element constants and units are reusable.
**Next experiment:**Await that mechanism; no prefactor, exponent or element-selection sweep. Full gates remain unchanged.
**Attempts:**
- Cycle218; candidate `e2e7ca76859aae8be2ff9b30e0cd8641e50a0fe5`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision discard.
  [Implementation](ideas/effective-charge-bond-prior/e2e7ca76859aae8be2ff9b30e0cd8641e50a0fe5/algo.py), [train](evaluation_results/cycle-218-train.json), [narrative](full_log.md#cycle-218-effective-charge-bond-stiffness).

## locality-ordered-replay: Nearby secants get later replay priority
Status: deferred

**Hypothesis:**Farthest-to-nearest replay can preserve more local curvature when the early path turns; newest secant always remains last.
**Outcome and uncertainty:**220 fully converges with valid energy and zero errors but slightly loses cost. Midpoint distance and native old-pair geometry are approximate; no retained model trace diagnoses a repair.
**Reason to revisit:**A separately justified curvature-relevance measure or common-frame history could make information ordering meaningful beyond chord proximity.
**Next experiment:**Await that model; no midpoint, norm or order heuristic sweep. Full gates unchanged.
**Attempts:**
- Cycle220; candidate `09d04c5d8615cce1788c025a61c9c919e0e9ad5a`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision discard.
  [Implementation](ideas/locality-ordered-replay/09d04c5d8615cce1788c025a61c9c919e0e9ad5a/algo.py), [train](evaluation_results/cycle-220-train.json), [narrative](full_log.md#cycle-220-locality-ordered-early-curvature-replay).

## tangent-only-curvature: Residual updates confined to the current tangent
Status: deferred

**Hypothesis:**Restricting residual alignment and updates to the geometric tangent avoids learning arbitrary redundant response.
**Outcome and uncertainty:**221 fully converges with valid energy and zero errors but loses cost. Old replay uses current projection; normal/cross base memory is retained, so the model is not fully covariant.
**Reason to revisit:**An independently established covariant base/history model could make the reusable tangent correction part of a complete operator learner.
**Next experiment:**Await that model or informative passive defect; no rank, normal-curvature or correction-strength sweep.
**Attempts:**
- Cycle221; candidate `1b1d27281b4749aa89063df9c8c8507efd432cf5`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision discard.
  [Implementation](ideas/tangent-only-curvature/1b1d27281b4749aa89063df9c8c8507efd432cf5/algo.py), [train](evaluation_results/cycle-221-train.json), [narrative](full_log.md#cycle-221-tangent-only-curvature-corrections).

## bond-orthogonal-auxiliary: Separate auxiliary stiffness from pure stretching
Status: deferred

**Hypothesis:**A fixed-bond tangent projection avoids counting stretching twice and improves physical-block identifiability.
**Outcome and uncertainty:**223 fully converges with valid energy but loses training cost. Physical stretch/bend coupling may be removed; no passive defect supports repair.
**Reason to revisit:**An independently derived coupled physical decomposition could preserve mixed responses while separating diagonal stiffness.
**Next experiment:**Await that model; no projection-strength, rank or cutoff sweep.
**Attempts:**
- Cycle223; candidate `e04188ed18fc8aaa2d46ca90bf4135626ed04bf4`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision discard.
  [Implementation](ideas/bond-orthogonal-auxiliary/e04188ed18fc8aaa2d46ca90bf4135626ed04bf4/algo.py), [train](evaluation_results/cycle-223-train.json), [narrative](full_log.md#cycle-223-bond-orthogonal-auxiliary-curvature).

## affine-difference-gp: Fit the pathwise affine-removed GP process
Status: deferred

**Hypothesis:**Transform observations, covariance and shared anchor noise consistently before fitting to avoid changing historical interpolation after fitting.
**Outcome and uncertainty:**224 fully converges with valid energy and zero errors but loses training cost. Native derivative-map error and reused physical mean remain unobserved.
**Reason to revisit:**An independently supported common-chart likelihood or measured anchor-noise model could make this exact linear-operator construction useful.
**Next experiment:**Await that mechanism; no noise, length or confidence sweep.
**Attempts:**
- Cycle224; candidate `f0c61dbe1d691d5af1c0b1990962b1d1787e683f`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision discard.
  [Implementation](ideas/affine-difference-gp/f0c61dbe1d691d5af1c0b1990962b1d1787e683f/algo.py), [train](evaluation_results/cycle-224-train.json), [narrative](full_log.md#cycle-224-affine-difference-gp-observations).

## modified-cholesky-step: Preserve off-diagonal curvature using GMW81
Status: deferred

**Hypothesis:**A nonnegative diagonal correction can supply better ordinary descent directions than spectral reflection while preserving learned free-basis coupling.
**Outcome and uncertainty:**225 fully converges with valid energy and zero errors but loses training cost. Basis dependence, modification magnitude and fallback frequency are unobserved.
**Reason to revisit:**An independently supported natural basis or informative curvature diagnostics could identify when preserving component couplings matters.
**Next experiment:**Await that evidence; no pivot, floor or strength sweep.
**Attempts:**
- Cycle225; candidate `b5b9b4ca12365e48b4afa979fd23d0526c2c1bf8`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision discard.
  [Implementation](ideas/modified-cholesky-step/b5b9b4ca12365e48b4afa979fd23d0526c2c1bf8/algo.py), [train](evaluation_results/cycle-225-train.json), [narrative](full_log.md#cycle-225-modified-cholesky-ordinary-step-model).

## evidence-physical-regularization: Infer common prior strength from secant evidence
Status: deferred

**Hypothesis:**Gaussian marginal likelihood can choose a more appropriate shared physical regularizer from paid data than one fixed strength.
**Outcome and uncertainty:**227 passes train cost/energy but fails valid cost and aggregate energy; both splits fully converge without errors. Independent scalar-output noise, redundant outputs and post-evidence clipping are unverified approximations.
**Reason to revisit:**An independently derived correlated response-error model or bounded-coefficient evidence could address overconfidence without tuning a fixed regularizer.
**Next experiment:**Await that model or passive inference diagnostics; no machine-bound/grid/noise/activation sweep. Retain unchanged full gates.
**Attempts:**
- Cycle227; candidate `31350880226b5c66d6be91ae59e4046838eed2e1`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision non_generalizable.
  [Implementation](ideas/evidence-physical-regularization/31350880226b5c66d6be91ae59e4046838eed2e1/algo.py), [train](evaluation_results/cycle-227-train.json), [valid](evaluation_results/cycle-227-valid.json), [narrative](full_log.md#cycle-227-evidence-selected-physical-regularization).

## realized-surrogate-comparison: Compare predictions after geodesic realization
Status: deferred

**Hypothesis:**The nonlinear path can change a GP proposal's modeled advantage over the ordinary direction; comparing realized endpoints can avoid poor selections.
**Outcome and uncertainty:**229 fully converges with valid energy and zero errors but slightly loses cost. Endpoint-only admission has little demonstrated benefit; no local failure or successful model trace diagnoses repair.
**Reason to revisit:**The state-restoring geometry preview and full quadratic extension are reusable in an explicit pullback optimizer, which can search the realized model rather than merely filter tangent proposals.
**Next experiment:**Test the separately motivated derivative-free geodesic-pullback search under unchanged full gates. No comparison margin or confidence tuning.
**Attempts:**
- Cycle229; candidate `cff6479cf884aa52fd05b9bd50a55acdaa4dd730`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision discard.
  [Implementation](ideas/realized-surrogate-comparison/cff6479cf884aa52fd05b9bd50a55acdaa4dd730/algo.py), [train](evaluation_results/cycle-229-train.json), [narrative](full_log.md#cycle-229-compare-realized-surrogate-endpoints).

## geodesic-pullback-gp: Optimize the GP through realized geodesic endpoints
Status: deferred

**Hypothesis:**Derivative-free optimization of the realized scalar model can choose better curved-path targets than tangent BFGS.
**Outcome and uncertainty:**230 passes train but loses validation cost; both splits fully converge with valid energy and no errors. Runtime rises sharply. Initial-tangent chord charts, historical derivative maps and off-span mean remain approximate; no passive defect supports repair.
**Reason to revisit:**A consistently differentiated common curvilinear chart or an independently improved scalar model could make the state-restoring pullback search useful. Merely resolving small third-order chord corrections more accurately may not justify its cost.
**Next experiment:**Await that model or informative passive evidence; no microbudget, tolerance, confidence or radius sweep. Preserve the full unchanged gates.
**Attempts:**
- Cycle230; candidate `a515f14bea2b4bdee02c8a5f22aee1e20b4d09d5`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision non_generalizable.
  [Implementation](ideas/geodesic-pullback-gp/a515f14bea2b4bdee02c8a5f22aee1e20b4d09d5/algo.py), [train](evaluation_results/cycle-230-train.json), [valid](evaluation_results/cycle-230-valid.json), [narrative](full_log.md#cycle-230-geodesic-pullback-gp-microsearch).

**294 joint-model reassessment:**One ambient scalar mean is now fitted to history and evaluated at full realized geodesic chords, retaining normal covariance at both. This satisfies the earlier compatibility prerequisite, but loses training cost despite full convergence, valid energy and zero errors (96 faster,77 slower,296 unchanged). No retained failure identifies a repair. The tested coupling is deferred without normal weights, search budgets, confidence or history sweeps. An analytic Jacobi-field query derivative might reduce runtime but does not establish better force-call efficiency; revisit only with independently supported model/search evidence.
- Cycle294; candidate `9ddbb9bd1c7de1ec05dac0905a4fdf02ace6a90b`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision discard.
  [Joint implementation](ideas/geodesic-pullback-gp/9ddbb9bd1c7de1ec05dac0905a4fdf02ace6a90b/algo.py), [training](evaluation_results/cycle-294-train.json), [narrative](full_log.md#cycle-294-ambient-history-gp-with-realized-geodesic-queries).

**378 revisit:**Accepted298/307 independently improve the physical mean/covariance after230 and294, supplying the declared scalar-model prerequisite. Reuse230's original40-query search directly on307, with no294 ambient covariance or377 chart replacement and no solver/gate tuning. Restore quaternion continuation state along with230's saved geometry/PES state. Another broad loss will require direct model/search evidence before further composition.

**378 reassessment:**The improved298/307 scalar surrogate still loses training cost; all469 converge with valid energy and no errors. Runtime remains about23minutes. Empty passive evidence supplies no localized repair. Await direct model/search evidence; no microbudget, tolerance, radius, confidence or further composition sweep.
- Cycle378; candidate `071586a51680c65eda5cda095be6de3b57f49cfe`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/geodesic-pullback-gp/071586a51680c65eda5cda095be6de3b57f49cfe/algo.py), [train](evaluation_results/cycle-378-train.json), [narrative](full_log.md#cycle-378-realized-geodesic-search-with-the-bending-gp).

## torsion-multiplicity-metric: Normalize the torsional geometry metric per central bond
Status: deferred

**Hypothesis:**Inverse-square-root row weights remove repeated common-rotation contributions without deleting proper dihedrals or weakening the physical prior.
**Outcome and uncertainty:**232 fully converges with valid energy and no errors but slightly loses training cost. Keeping every row avoids205's failures; bending contrasts, Euclidean learning changes and finite rank thresholds remain unresolved.
**Reason to revisit:**An independently derived decomposition separating common rotation from bending contrasts could retain useful contrast geometry while normalizing duplication. The fixed weighted-coordinate framework is reusable.
**Next experiment:**Await that model or diagnostic evidence; no exponent/group/rank-threshold sweep and no validation of this rejection.
**Attempts:**
- Cycle232; candidate `752a383bb6c3c45ebcb1306f698772cb177efa81`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision discard.
  [Implementation](ideas/torsion-multiplicity-metric/752a383bb6c3c45ebcb1306f698772cb177efa81/algo.py), [train](evaluation_results/cycle-232-train.json), [narrative](full_log.md#cycle-232-multiplicity-normalized-torsion-geometry).

## integral-hessian-gp: Learn varying curvature from line-integral secants
Status: deferred

**Hypothesis:**A nonparametric Hessian can interpret recent gradient differences as path averages while keeping older learned information in a TS/SR1 base.
**Outcome and uncertainty:**233 fully converges with valid energy and zero errors but loses cost broadly. Native common-frame approximation, median bandwidth, kernel covariance and dual posterior consistency remain unverified; no localized failure supports repair.
**Reason to revisit:**An independently established covariant common frame or calibrated Hessian-variation model could make integral observations useful. The bounded posterior helper and once-only older-base assimilation remain reusable.
**Next experiment:**Await such evidence; no window/bandwidth/noise/quadrature-order sweep. Preserve full gates.
**Attempts:**
- Cycle233; candidate `a8a188602d39c18cd2825f5518f44c16d9235fa7`; champion `84e94edf6b970ec3eb3bc422a128bec4b6bcaac6`; decision discard.
  [Implementation](ideas/integral-hessian-gp/a8a188602d39c18cd2825f5518f44c16d9235fa7/algo.py), [train](evaluation_results/cycle-233-train.json), [narrative](full_log.md#cycle-233-nonparametric-integral-observation-hessian-learning).

## common-torsion-spectrum-metric: Separate common rotation from torsion contrasts
Status: deferred

**Hypothesis:**Normalize the common central-bond rotation metric by multiplicity while retaining contrast stiffness, only in negative-curvature reflection.
**Outcome and uncertainty:**238 lowers training cost but fails aggregate energy; all469 converge with zero errors. One31.3 kcal/mol higher endpoint accounts for more cost gain than the aggregate. Passive manifest is empty; no trajectory diagnosis supports repair.
**Reason to revisit:**A physically supported coupling of common torsion motion to the energy landscape could identify an appropriate metric; the positive common/contrast decomposition is reusable.
**Next experiment:**Await that evidence. No multiplicity exponent, topology subgroup, activation or stiffness sweep; current apparent gain is dominated by an energy-losing path.
**Attempts:**
- Cycle238; candidate `73eee7aea750a49d5eda84077f242c156ee66bed`; champion `73cadc1ae8fd47e796123cafe93956f0e2c67043`; decision invalid.
  [Implementation](ideas/common-torsion-spectrum-metric/73eee7aea750a49d5eda84077f242c156ee66bed/algo.py), [train](evaluation_results/cycle-238-train.json), [narrative](full_log.md#cycle-238-common-rotation-torsion-metric-for-spectral-reflection).

## signed-gp-mean: Retain negative curvature in the GP energy mean
Status: deferred

**Hypothesis:**Separate signed energy Taylor curvature from positive kernel distances and descent-step curvature.
**Outcome and uncertainty:**239 fully converges with valid energy and zero errors, but slightly loses training cost. Successful microsearch/model traces are unavailable; negative-mode reliability and unbounded extrapolation remain uncertain.
**Reason to revisit:**A independently established bounded nonconvex surrogate solver or negative-mode error model could make the exact mean/kernel separation useful.
**Next experiment:**Await that model or passive evidence; no floor, confidence, length or iteration sweep from this result.
**Attempts:**
- Cycle239; candidate `cdeccd5911b68597d673094a1a46243745b06777`; champion `73cadc1ae8fd47e796123cafe93956f0e2c67043`; decision discard.
  [Implementation](ideas/signed-gp-mean/cdeccd5911b68597d673094a1a46243745b06777/algo.py), [train](evaluation_results/cycle-239-train.json), [narrative](full_log.md#cycle-239-signed-quadratic-mean-with-positive-gp-kernel-geometry).

## event-driven-topology: Refresh coordinates when new covalent contacts appear
Status: incorporated

**Hypothesis:**A newly formed contact warrants rediscovering bonds, angles, torsions and fragments so the optimizer directly represents its stiff motion.
**Outcome and uncertainty:**240 passes both cost and energy gates with full convergence and zero errors. It changes relatively few trajectories; topology correction and curvature-reset effects are not separately observable.
**Reason to revisit:**The accepted event detector provides a controlled basis for future topology-aware memory transfer if an independently consistent chart transformation is derived.
**Next experiment:**Retain the existing distance rule and seen-pair suppression; no cutoff or activation tuning. Any memory transfer must preserve paid observations and pass full gates.
**Attempts:**
- Cycle240; candidate `6b55fe37d19e209e52357a3ffa5a59402eb8fa26`; champion `73cadc1ae8fd47e796123cafe93956f0e2c67043`; decision keep. Source retained in accepted Git history.
  [train](evaluation_results/cycle-240-train.json), [valid](evaluation_results/cycle-240-valid.json), [narrative](full_log.md#cycle-240-refresh-internal-coordinates-after-new-covalent-contacts).

**Contact-loss extension:**242 fully converges with valid energy but loses training cost; defer once-per-pair disappearing-contact resets without cutoff/hysteresis tuning.240 remains incorporated.
- Cycle242; candidate `0e0288d121957faf23b3924f0df270fd5b46166f`; champion `6b55fe37d19e209e52357a3ffa5a59402eb8fa26`; decision discard.
  [Implementation](ideas/event-driven-topology/0e0288d121957faf23b3924f0df270fd5b46166f/algo.py), [train](evaluation_results/cycle-242-train.json), [narrative](full_log.md#cycle-242-refresh-coordinates-after-disappearing-covalent-contacts).

## hermite-tensor-gp-mean: Learn a bounded cubic and quartic GP mean
Status: deferred

**Hypothesis:**One past energy/full-gradient Hermite interpolation captures anharmonicity while preserving the current quadratic expansion.
**Outcome and uncertainty:**241 fully converges with valid energy and no errors but loses training cost. Approximate derivative maps, learned Hessian and unmodeled mean-estimation uncertainty remain limitations.
**Reason to revisit:**A independently supported common-chart derivative model or uncertainty-aware tensor fit could make the interpolation component useful.
**Next experiment:**Await such evidence; no amplitude, rank threshold, kernel or search-budget sweep. The analytic mean/gradient and coercivity check remain reusable.
**Attempts:**
- Cycle241; candidate `5942d53303604fadc88455a75ccb1c160c7ea6ba`; champion `6b55fe37d19e209e52357a3ffa5a59402eb8fa26`; decision discard.
  [Implementation](ideas/hermite-tensor-gp-mean/5942d53303604fadc88455a75ccb1c160c7ea6ba/algo.py), [train](evaluation_results/cycle-241-train.json), [narrative](full_log.md#cycle-241-coercive-hermite-tensor-mean-for-the-gp).

## rebuild-gp-observations: Re-express paid GP data in rebuilt coordinates
Status: deferred

**Hypothesis:**Reconstructing coordinates and gradients from saved paid Cartesian data preserves useful GP information across topology changes without carrying an approximate Hessian.
**Outcome and uncertainty:**243 fully converges without errors but fails energy and slightly loses cost. The available passive manifest is empty; chart/retention suitability is unresolved.
**Reason to revisit:**An independently established compatibility or observation-error model across topology changes could determine which data remain reliable. The geometry-only reconstruction and state-restoration helper are reusable.
**Next experiment:**Await that evidence; no history/radius/rank/activation sweep. Keep240’s fresh starts.
**Attempts:**
- Cycle243; candidate `1a71f55505f21c92a47259b23d66ba0fbf7a4404`; champion `6b55fe37d19e209e52357a3ffa5a59402eb8fa26`; decision invalid.
  [Implementation](ideas/rebuild-gp-observations/1a71f55505f21c92a47259b23d66ba0fbf7a4404/algo.py), [train](evaluation_results/cycle-243-train.json), [narrative](full_log.md#cycle-243-reconstruct-gp-observations-after-coordinate-rebuilds).

## physical-ts-update: Physical reflection in the TS correction direction
Status: incorporated

**Hypothesis:**The physical negative-curvature metric supported by234 can improve the TS rank-two update direction while preserving signed secants and native residual blending.
**Outcome and uncertainty:**244 passes both cost/energy gates with full convergence and zero errors. Per-mode correction and successful trajectory details are unobserved; improvement is measured at complete-split level.
**Reason to revisit:**Accepted physical reflection now has evidence in learning and step construction. Any further extension needs a distinct metric or observation mechanism.
**Next experiment:**Keep the native signed residual, Euclidean blend and no-floor reflection. Test independent physical priors under full gates rather than tuning this update.
**Attempts:**
- Cycle244; candidate `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; champion `6b55fe37d19e209e52357a3ffa5a59402eb8fa26`; decision keep. Source retained in accepted Git history.
  [train](evaluation_results/cycle-244-train.json), [valid](evaluation_results/cycle-244-valid.json), [narrative](full_log.md#cycle-244-physical-reflection-in-the-ts-hessian-update).

**375 full-prior learning control:**After244's accepted diagonal correction, replacing only its negative-curvature absolute action by the fitted physical model plus initial redundant-complement completion fully converges and passes energy but loses cost. This differs from235's stepping experiment; neither supports full fitted-metric superiority. Defer this extension until independent mode-quality evidence, without completion/floor/coefficient sweeps.244 remains incorporated.
- Cycle375; candidate `b7e29fbb99c9071e8a3834792b25f5dac58e4fc5`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/physical-ts-update/b7e29fbb99c9071e8a3834792b25f5dac58e4fc5/algo.py), [train](evaluation_results/cycle-375-train.json), [narrative](full_log.md#cycle-375-calibrated-physical-metric-in-ts-curvature-learning).

## uff-tetrahedral-torsion: Barrier-derived minimum curvature for tetrahedral torsions
Status: deferred

**Hypothesis:**UFF threefold barriers and their matched path normalization give a better torsional prior for four-coordinate group14 centers.
**Outcome and uncertainty:**245 passes train but loses validation cost. Both splits fully converge with valid energy and no errors. Distance typing, minimum-phase curvature and transfer to xTB remain uncertain; no localized defect supports repair.
**Reason to revisit:**An independently supported torsional phase/typing model could improve this chemical prior. The complete barrier/normalization construction is reusable.
**Next experiment:**Await that evidence; no element-subset, barrier, multiplicity or phase sweep.
**Attempts:**
- Cycle245; candidate `ce531850618f37e50e58dd10087b254535752fdb`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision non_generalizable.
  [Implementation](ideas/uff-tetrahedral-torsion/ce531850618f37e50e58dd10087b254535752fdb/algo.py), [train](evaluation_results/cycle-245-train.json), [valid](evaluation_results/cycle-245-valid.json), [narrative](full_log.md#cycle-245-uff-barrier-curvature-for-tetrahedral-torsions).

**359 finite-phase revisit:**Use245's same typing/barriers in the source threefold phase potential, subtracting its complete current quadratic Taylor polynomial and adding only the remainder to307's GP mean. This supplies the declared phase-model prerequisite without restoring245's rejected minimum-curvature Hessian, or tuning chemical coverage/barriers.

**359 reassessment:**The published chemical phase applied only as a higher-order GP remainder fully converges and passes energy but loses cost. The declared phase prerequisite has now been tested; no phase/barrier/chemical-support sweep. Further work requires independent finite-model prediction evidence.
- Cycle359; candidate `d162b9bbcf569b9cffb309abe305aa7a6c9950a9`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/uff-tetrahedral-torsion/d162b9bbcf569b9cffb309abe305aa7a6c9950a9/algo.py), [training](evaluation_results/cycle-359-train.json), [narrative](full_log.md#cycle-359-chemically-phased-tetrahedral-torsion-mean).

## global-rigid-trial-projection: Remove overall motion from collective trial directions
Status: deferred

**Hypothesis:**Project initial Cartesian trial velocities off global translation/rotation while retaining relative fragment motion.
**Outcome and uncertainty:**246 fully converges with valid energy and no errors but loses cost. Only initial tangents are horizontal; finite geodesics, full-space curvature and history remain native. No localized defect is observed.
**Reason to revisit:**A derived quotient geometry with compatible curvature/history could address the incomplete finite-path/model treatment. The generator and mapped-nullspace helper are reusable.
**Next experiment:**Await that independent formulation; no rank, radius, atom weighting or subgroup sweep.
**Attempts:**
- Cycle246; candidate `7651d0f82bdf2edf6b80530bfd958a64cc883a00`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision discard.
  [Implementation](ideas/global-rigid-trial-projection/7651d0f82bdf2edf6b80530bfd958a64cc883a00/algo.py), [train](evaluation_results/cycle-246-train.json), [narrative](full_log.md#cycle-246-exclude-global-rigid-motion-from-tric-trial-tangents).

Cycle 256 pending: existing null-space SQP with global real-atom centroid and best-fit rotation constraints supplies Lagrangian curvature and finite endpoint correction, addressing the limited tangent-only formulation of 246. Native approximate transport remains; this is not an exact quotient solver.

Cycle 256 outcome: `1e2049177e435566e776d5176b47152ad5e8e81a`, [implementation](ideas/global-rigid-trial-projection/1e2049177e435566e776d5176b47152ad5e8e81a/algo.py), [train](evaluation_results/cycle-256-train.json), [narrative](full_log.md#cycle-256-global-rigid-gauge-through-constrained-internal-optimization). Cost 0.7640350167869745, energy 1.0015929126874747, 469 converged and zero errors; [68, 99, 302] faster/slower/unchanged. Discard. Native SQP is operational but loses cost; no localized repair evidence. Defer until consistent finite gauge/transport treatment is derived; no tolerance sweep.

## uff-carbon-inversion: Carbonyl-sensitive UFF inversion minimum curvature
Status: deferred

**Hypothesis:**Published carbonyl-specific inversion stiffness improves the generic geometry-only out-of-plane prior at three-coordinate carbon centers.
**Outcome and uncertainty:**248 passes train but loses validation cost, with both splits fully converged, energy-valid and error-free. Graph typing and minimum-curvature transfer to xTB are unverified; no localized implementation failure is observed.
**Reason to revisit:**An independently supported bond-order/phase model or observed inversion-curvature discrepancy could justify a different prior. The permutation normalization and existing gradient pullback are reusable.
**Next experiment:**Await that evidence; no chemical subset, K value, phase or ridge sweep.
**Attempts:**
- Cycle248; candidate `2298f4582878d2cf2f2c1626ee1cfde815a785df`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision non_generalizable.
  [Implementation](ideas/uff-carbon-inversion/2298f4582878d2cf2f2c1626ee1cfde815a785df/algo.py), [train](evaluation_results/cycle-248-train.json), [valid](evaluation_results/cycle-248-valid.json), [narrative](full_log.md#cycle-248-carbonyl-sensitive-uff-inversion-curvature).

## expected-improvement-gp: Acquire expected energy improvement in the local GP
Status: deferred

**Hypothesis:**Expected improvement can choose a more useful paid observation than posterior-mean minimization under existing physical admission.
**Outcome and uncertainty:**249 fully converges with valid energy and no errors, but slightly loses training cost. Flat acquisition tails, chart approximation and posterior calibration remain unresolved.
**Reason to revisit:**An independently supported uncertainty model could make energy acquisition useful; analytic EI and its deterministic limit remain reusable.
**Next experiment:**Await that evidence; no incumbent, offset, confidence or search-budget sweep.
**Attempts:**
- Cycle249; candidate `28083c504db865eeac87c68adf6dfd5d267776bc`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision discard.
  [Implementation](ideas/expected-improvement-gp/28083c504db865eeac87c68adf6dfd5d267776bc/algo.py), [train](evaluation_results/cycle-249-train.json), [narrative](full_log.md#cycle-249-expected-improvement-gp-microsearch).

## statistical-quasi-newton: Relative least-change learning with uncertainty step prediction
Status: deferred

**Hypothesis:**SQN's negative Broyden update avoids unnecessary transverse stiffness, and its paired Wishart step predictor limits aggressive directions.
**Outcome and uncertainty:**250 fails energy and broad cost, with one force-call limit and no errors. Passive late steps stagnate while forces remain high; internal scale/matrix state is unavailable.
**Reason to revisit:**Static review identifies shortening after trust restriction, whereas the published estimate scales the unconstrained Newton direction. Correct placement could remove unintended double shortening.
**Next experiment:**Bounded repair1 applies the same factor inside the restricted solver, keeping all other learning/admission and epsilon constants fixed; broad remaining loss ends the repair.
**Attempts:**
- Cycle250; candidate `974a537e43adca1ce4b43203d4e592040c37e845`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision invalid.
  [Implementation](ideas/statistical-quasi-newton/974a537e43adca1ce4b43203d4e592040c37e845/algo.py), [train](evaluation_results/cycle-250-train.json), [narrative](full_log.md#cycle-250-statistical-quasi-newton-learning-with-paired-step-prediction).
  [Passive bundle](diagnostics/ce72a02a3b69474994d3c351e3b60142/db03403fa3980a65d74a7612cf9a16e50054b7616a902b8289e570c10e481a30.json).

**Repair1 reassessment:**251 moves scaling before the radius solve. All469 now converge and energy passes, but broad cost loss increases. The former limited case converges in16 calls; no retained failure traces remain. The diagnosed placement defect was real, but its repair does not make this adaptation beneficial. Revisit only with an independently supported moving-chart/globalization coupling; no epsilon, eligibility, activation or scale sweep.
- Cycle251; candidate `9ce218f1c43229f93eca0ec8e6db7089deb9a16d`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision discard, bounded repair1.
  [Implementation](ideas/statistical-quasi-newton/9ce218f1c43229f93eca0ec8e6db7089deb9a16d/algo.py), [train](evaluation_results/cycle-251-train.json), [narrative](full_log.md#cycle-251-apply-sqn-shortening-before-trust-restriction).

## hydrogen-bond-angular-prior: Directional weak-interaction bending curvature
Status: deferred

**Hypothesis:**A donor-H-acceptor angular block supplements radial auxiliary and isotropic fragment priors without new coordinates.
**Outcome and uncertainty:**252 fully converges with valid energy and no errors but narrowly loses cost. Chemical recognition, transferred Lindh stiffness and omitted numerically collinear angles remain limitations.
**Reason to revisit:**An independently supported directional interaction Hessian or observed H-bond curvature defect could improve this reusable Cartesian-gradient pullback block.
**Next experiment:**Await that evidence; no donor/acceptor subset, decay, radius, stiffness or ridge sweep.
**Attempts:**
- Cycle252; candidate `860d84e7d6b50d0419e594a1d3b5bf6116dadc2a`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision discard.
  [Implementation](ideas/hydrogen-bond-angular-prior/860d84e7d6b50d0419e594a1d3b5bf6116dadc2a/algo.py), [train](evaluation_results/cycle-252-train.json), [narrative](full_log.md#cycle-252-directional-hydrogen-bond-curvature-prior).

## rotation-matrix-gp: Couple fragment rotation components in the residual kernel
Status: deferred

**Hypothesis:**Rodrigues chord features represent finite rotation similarity while matching the current GP infinitesimal metric.
**Outcome and uncertainty:**253 fully converges with valid energy and no errors but loses training cost. Relative rotation-vector subtraction is not exact group composition; mixed physical projection is not an isotropic SO(3) kernel.
**Reason to revisit:**A coherently derived noncommutative history chart with matching physical realization could make rotation features useful. Analytic Rodrigues feature derivatives are reusable.
**Next experiment:**324 tests exact group composition in a current normalized logarithm chart for both historical/query covariance derivatives, retaining native physical realization and307 bending features. This resolves the explicit253 component-subtraction issue; no feature-weight, angle scale, kernel length or microbudget sweep.
**Attempts:**
- Cycle253; candidate `52a35fd29751219a5acb8c9d42fbcd8aa7df95b9`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision discard.
  [Implementation](ideas/rotation-matrix-gp/52a35fd29751219a5acb8c9d42fbcd8aa7df95b9/algo.py), [train](evaluation_results/cycle-253-train.json), [narrative](full_log.md#cycle-253-coupled-rotation-matrix-features-for-the-gp-residual).

**324 reassessment:**Exact group composition, a normalized current log chart and matching history/query derivatives pass training but lose validation cost. All934 converge with valid energy; no diagnostic failure.253's coordinate-subtraction limitation is now tested directly and is insufficient for general gain. Defer until observational/physical metric evidence identifies a further mechanism; do not infer a missing SE3 coupling from endpoints.
- Cycle324; candidate `befca672ef2a71681bdceb1a2edb811a35cab5d6`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision non_generalizable.
  [Implementation](ideas/rotation-matrix-gp/befca672ef2a71681bdceb1a2edb811a35cab5d6/algo.py). [Training evidence](evaluation_results/cycle-324-train.json). [Validation evidence](evaluation_results/cycle-324-valid.json). [Narrative](full_log.md#cycle-324-exact-rotation-composition-for-gp-covariance).

## group-huber-calibration: Bounded influence in early physical calibration
Status: deferred

**Hypothesis:** A group-Huber residual loss reduces the effect of strongly mismatched early secants on physical coefficient fitting while retaining every raw secant in Hessian replay.
**Outcome and uncertainty:** Cycle 255 converges all 469 training molecules without errors and passes energy, but cost 0.749337930214813 loses to 0.7492669586281289. Only four faster and six slower molecules; 459 unchanged. No localized defect is observed.
**Reason to revisit:** Independent evidence of contamination in coefficient observations could support a principled noise model. The invariant group-residual IRLS implementation is reusable.
**Next experiment:** Await that evidence; no cutoff sweep from this small cost regression.
**Attempts:**
- Cycle 255; candidate `0015404410f7cc5498c472449b22ae9aec20ec05`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision discard.
  [Implementation](ideas/group-huber-calibration/0015404410f7cc5498c472449b22ae9aec20ec05/algo.py), [training evidence](evaluation_results/cycle-255-train.json), [narrative](full_log.md#cycle-255-group-huber-calibration-of-the-physical-prior). Source https://osqp.org/docs/examples/huber.html; the group norm is our extension.

## nonsymmetric-root-directions: Broyden gradient-root learning
Status: deferred

**Hypothesis:** Nonsymmetric rank-one learning of finite transported gradient secants provides better post-calibration directions than symmetric energy-curvature learning.
**Outcome and uncertainty:** Cycle257 is valid, fully converged and error-free, but cost 0.7913009275241293 loses to 0.7492669586281289. Faster/slower/unchanged [64, 185, 220]. Ray restriction, native energy-model feedback and finite-history transport are approximations, without a localized failure diagnosis.
**Reason to revisit:** A consistent root-merit globalization or common-frame Jacobian construction could make this independent root learner useful.
**Next experiment:** Await that independently derived formulation or informative passive evidence; no memory, damping or activation sweep.
**Attempts:**
- Cycle257; candidate `240fbc9dbbb02a8772622c2945bfeb9198bb538e`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision discard.
  [Implementation](ideas/nonsymmetric-root-directions/240fbc9dbbb02a8772622c2945bfeb9198bb538e/algo.py), [train](evaluation_results/cycle-257-train.json), [narrative](full_log.md#cycle-257-nonsymmetric-broyden-root-directions-after-physical-calibration).

## secant-covariance-metric: Positive curvature from balanced secant moments
Status: deferred

**Hypothesis:** A regularized solution of P Rg P=Rx using recent paid secants improves the current positive step metric while leaving energy learning intact.
**Outcome and uncertainty:** Cycle258 converges fully with valid energy and no errors but loses cost, 0.7659237223115666 versus 0.7492669586281289; faster/slower/unchanged [80, 100, 289]. Finite optimization-selected pairs are biased samples, and their native transport is approximate.
**Reason to revisit:** An independently supported sampling/transport model could make this exact low-rank matrix-square-root moment solution useful.
**Next experiment:** Await that model or localized passive evidence; no ridge, normalization or window sweep.
**Attempts:**
- Cycle258; candidate `499aa41f9e735cbfb3509edebd12d6b1e0f502ae`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision discard.
  [Implementation](ideas/secant-covariance-metric/499aa41f9e735cbfb3509edebd12d6b1e0f502ae/algo.py), [train](evaluation_results/cycle-258-train.json), [narrative](full_log.md#cycle-258-positive-step-metric-correction-by-secant-covariance-matching).

## stereographic-rotation-chart: Locally matched modified Rodrigues coordinates
Status: deferred

**Hypothesis:** The rational map r=4q_vector/(1+q_scalar), with consistent analytic derivatives, improves finite rotational geometry and curvature learning.
**Outcome and uncertainty:** Cycle260 converges all469 without errors but fails energy (0.99982461922416) and cost (0.7524768953697083). Faster/slower/unchanged [35, 38, 396]; empty failure_details. Branch resets and derivative/model states are unavailable on successful paths.
**Reason to revisit:** Independent evidence of a rotation-history/branch-model defect could motivate a consistent rotation atlas or common-frame observation model; the exact rational derivative implementation is reusable.
**Next experiment:** Await that model or localized passive evidence; no reset-angle, trust-factor or chart-scale sweep.
**Attempts:**
- Cycle260; candidate `ee775e06452787ae0fea5e78f29eeee09d042e45`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision invalid.
  [Implementation](ideas/stereographic-rotation-chart/ee775e06452787ae0fea5e78f29eeee09d042e45/algo.py), [train](evaluation_results/cycle-260-train.json), [narrative](full_log.md#cycle-260-stereographic-rotation-coordinates-with-exact-derivatives).

## normal-geometry-gp: Preserve curved normal history in covariance
Status: deferred

**Hypothesis:** Retain physically scaled normal coordinate offsets and their derivative maps in a positive embedding kernel without expanding the search span.
**Outcome and uncertainty:**261 passes training but misses validation cost; both splits fully converge with valid energy and no errors. No retained failure bundle identifies a repair.
**Reason to revisit:** A separately supported common curvilinear query and mean model could give the extra covariance dimensions a faithful physical interpretation.
**Next experiment:** Await that model or localized passive evidence; no normal-weight, length or history sweep. Compare against the current champion under unchanged gates.

**Attempts:**
- Cycle261; candidate `2d4eb77fa7344194b6e693e53694eb2f6bb5d4c2`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision non_generalizable.
  [Implementation](ideas/normal-geometry-gp/2d4eb77fa7344194b6e693e53694eb2f6bb5d4c2/algo.py). [Training evidence](evaluation_results/cycle-261-train.json). [Validation evidence](evaluation_results/cycle-261-valid.json). [Narrative](full_log.md#cycle-261-retain-normal-geometry-in-the-gp-covariance).

**294 joint-model reassessment:**One ambient scalar mean is now fitted to history and evaluated at full realized geodesic chords, retaining normal covariance at both. This satisfies the earlier compatibility prerequisite, but loses training cost despite full convergence, valid energy and zero errors (96 faster,77 slower,296 unchanged). No retained failure identifies a repair. The tested coupling is deferred without normal weights, search budgets, confidence or history sweeps. An analytic Jacobi-field query derivative might reduce runtime but does not establish better force-call efficiency; revisit only with independently supported model/search evidence.
- Cycle294; candidate `9ddbb9bd1c7de1ec05dac0905a4fdf02ace6a90b`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision discard.
  [Joint implementation](ideas/geodesic-pullback-gp/9ddbb9bd1c7de1ec05dac0905a4fdf02ace6a90b/algo.py), [training](evaluation_results/cycle-294-train.json), [narrative](full_log.md#cycle-294-ambient-history-gp-with-realized-geodesic-queries).

## projected-vector-gp: Nonconservative projected gradient-field surrogate
Status: deferred

**Hypothesis:** Relax reduced-chart integrability by learning mapped gradient residuals with a projected vector kernel and searching its posterior expected stationarity.
**Outcome and uncertainty:**262 has valid energy, full convergence and zero errors but severe broad cost loss. No passive traces distinguish observation-model and acquisition errors.
**Reason to revisit:** An independently measured nonintegrable chart error or a supported conservative/nonconservative decomposition could justify this vector construction.
**Next experiment:** Require that evidence; no uncertainty, kernel or search-budget sweep after this broad regression.

**Attempts:**
- Cycle262; candidate `c850f5c19a2e4bf2258310a913be4d79406a11b1`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision discard.
  [Implementation](ideas/projected-vector-gp/c850f5c19a2e4bf2258310a913be4d79406a11b1/algo.py). [Training evidence](evaluation_results/cycle-262-train.json). [Narrative](full_log.md#cycle-262-projected-gradient-field-gp-stationarity-search).

## curvature-marked-gp: Nonstationary covariance from historical Hessians
Status: deferred

**Hypothesis:** Combine inverse positive historical curvatures in a determinant-normalized Gaussian convolution kernel to represent varying local scales.
**Outcome and uncertainty:**263 converges all469 with valid energy and zero errors but loses cost. Curvature marks are learned approximations, reinterpreted in today's basis and held fixed during derivative observations; no passive state isolates these limitations.
**Reason to revisit:** A separately supported spatial curvature field with consistent derivative information could replace fixed historical marks.
**Next experiment:** Await that model or informative diagnostics; no mark strength, length or history sweep.

**Attempts:**
- Cycle263; candidate `c9ffae85cb2cf03bca226a64a2af33782a72955f`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision discard.
  [Implementation](ideas/curvature-marked-gp/c9ffae85cb2cf03bca226a64a2af33782a72955f/algo.py). [Training evidence](evaluation_results/cycle-263-train.json). [Narrative](full_log.md#cycle-263-historical-curvature-marked-gp-covariance).

**338 revisit:**Accepted307 supplies a smooth explicit spatial metric from the bending map. Replace263 frozen historical Hessian marks with Sigma(y)=4*(JF(y).T*JF(y))^-1 and differentiate that complete field at both covariance arguments. This directly supplies the declared spatial-field/derivative prerequisite; no mark-strength or length sweep.

**338 outcome/reassessment:**The exact spatial bending field converges all469 with valid energy and no errors, but cost gain0.0000484152591718 misses the1e-4 gate. No passive failure localizes a repair. This tests the previously requested consistent spatial-field prerequisite without yielding acceptance. Revisit only after independent evidence identifies the remaining covariance discrepancy; no metric-strength, length, noise or conditioning sweep.
- Cycle338; candidate `50435e0ddac89383a09ceef43cfdcc0d5d4a4d58`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/curvature-marked-gp/50435e0ddac89383a09ceef43cfdcc0d5d4a4d58/algo.py), [training](evaluation_results/cycle-338-train.json), [narrative](full_log.md#cycle-338-differentiable-bending-metric-convolution-covariance).

## retrospective-trust-feedback: Assess radius with the updated endpoint quadratic
Status: deferred

**Hypothesis:** A new Hessian should set its next trust radius using retrospective agreement on the already paid step.
**Outcome and uncertainty:**264 passes energy and fully converges without errors but loses cost. Endpoint tangent and native two-sided controller differ from the source's complete algorithm; no passive failure localizes a repair.
**Reason to revisit:** An independently supported reverse-path scalar model or a complete acceptance/globalization mechanism could make retrospective feedback useful.
**Next experiment:** Await that model; no ratio threshold or radius-factor sweep.

**Attempts:**
- Cycle264; candidate `e9291f050c235860f4fa2b6718fbfb5448e71416`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision discard.
  [Implementation](ideas/retrospective-trust-feedback/e9291f050c235860f4fa2b6718fbfb5448e71416/algo.py). [Training evidence](evaluation_results/cycle-264-train.json). [Narrative](full_log.md#cycle-264-retrospective-endpoint-quadratic-trust-feedback).

## cautious-gp-proposals: Remove first-order uphill physical eigencomponents
Status: deferred

**Hypothesis:** Masking GP proposal components inconsistent with current descent can avoid nonlinear extrapolation costs while retaining scalar-model admission.
**Outcome and uncertainty:**265 fully converges without errors but fails energy and loses cost. No passive trace shows whether removed coupled motion or downstream basin changes dominate.
**Reason to revisit:** An independently supported mode-coupling criterion could distinguish necessary uphill components from inaccurate extrapolation.
**Next experiment:** Await that criterion; no sign threshold, scaling or basis sweep.

**Attempts:**
- Cycle265; candidate `cddbb96bdd1884876414fc69e449de5d1158e2d5`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision invalid.
  [Implementation](ideas/cautious-gp-proposals/cddbb96bdd1884876414fc69e449de5d1158e2d5/algo.py). [Training evidence](evaluation_results/cycle-265-train.json). [Narrative](full_log.md#cycle-265-cautious-physical-eigenmode-gp-proposals).

## stretch-schur-reflection: Preserve stretch stiffness during indefinite regularization
Status: deferred

**Hypothesis:** Reflect only relaxed soft curvature to retain useful learned stretch coupling.
**Outcome and uncertainty:**266 fully converges without errors but fails energy and loses cost; no passive failure trace isolates the source.
**Reason to revisit:** Independent evidence that positive bond curvature is accurate and soft Schur stiffness is excessive could justify a different bounded relaxed-block model.
**Next experiment:** First obtain such diagnostic evidence; no block, rank or reflection-strength sweep from the present aggregate result.

**Attempts:**
- Cycle266; candidate `a2ed565cfdc5ffe3845e263d3bc40efcf327002f`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision invalid.
  [Implementation](ideas/stretch-schur-reflection/a2ed565cfdc5ffe3845e263d3bc40efcf327002f/algo.py), [training](evaluation_results/cycle-266-train.json), [narrative](full_log.md#cycle-266-stretch-preserving-schur-curvature-reflection).

## compact-endpoint-curvature: Average endpoint Hessians in the secant relation
Status: deferred

**Hypothesis:**(Bnew+Bold)s/2=y estimates instantaneous endpoint curvature better than Bnew*s=y.
**Outcome and uncertainty:**267 passes energy but sharply loses cost;467 converge and2 hit200 calls with persistent motion. Passive bundles lack Hessian state and cannot establish alternating model error.
**Reason to revisit:** An independently accurate or uncertainty-qualified old Hessian could make the compact relation reliable; recursively approximate curvature is insufficient here.
**Next experiment:**Require such evidence or a separately derived stable endpoint inference scheme; no extrapolation-factor or late-activation sweep.

**Attempts:**
- Cycle267; candidate `3a80a445ed88f5b897a436bfa0f538cb5e5be6ab`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision discard.
  [Implementation](ideas/compact-endpoint-curvature/3a80a445ed88f5b897a436bfa0f538cb5e5be6ab/algo.py), [training](evaluation_results/cycle-267-train.json), [diagnostics](diagnostics/87487bf2e1f5455d9cc9f7289ca4ddd6/manifest.json), [narrative](full_log.md#cycle-267-compact-finite-difference-endpoint-curvature).

## block-loo-gp-noise: Conditional discrepancy as historical block noise
Status: deferred

**Hypothesis:** Leave-one-block-out energy/gradient disagreement can identify unreliable historical surrogate information.
**Outcome and uncertainty:**268 fully converges without errors but fails energy and loses cost. Successful model states are unobserved; simultaneous conditional estimates and real nonlinear information remain uncertainties.
**Reason to revisit:**Predictive diagnostics distinguishing observation-chart discrepancy from genuine curvature could support an identifiable noise model.
**Next experiment:**Await that evidence; no noise multiplier, iteration count or confidence sweep.

**Attempts:**
- Cycle268; candidate `31e4194689ae97975dd73229d1c37005d6576acd`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision invalid.
  [Implementation](ideas/block-loo-gp-noise/31e4194689ae97975dd73229d1c37005d6576acd/algo.py), [training](evaluation_results/cycle-268-train.json), [narrative](full_log.md#cycle-268-block-leave-one-out-gp-covariance-matching).

## gp-advantage-confidence: Correlated uncertainty of replacing the ordinary step
Status: deferred

**Hypothesis:**Pairwise anchored posterior variance can reject uncertain GP advantages that pass absolute-energy confidence.
**Outcome and uncertainty:**269 fully converges with valid energy and zero errors but loses cost; conservative selection and model calibration are unresolved.
**Reason to revisit:**An independently calibrated probability of relative improvement could change the cost of this risk criterion.
**Next experiment:**Require predictive calibration evidence; no confidence-ratio or acquisition-budget sweep.

**Attempts:**
- Cycle269; candidate `bdfe590600cad9e1f44e3cb0d7ee1e143325538b`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision discard.
  [Implementation](ideas/gp-advantage-confidence/bdfe590600cad9e1f44e3cb0d7ee1e143325538b/algo.py), [training](evaluation_results/cycle-269-train.json), [narrative](full_log.md#cycle-269-correlated-gp-advantage-confidence).

## curvature-flowchart: Discrete signed-curvature Hessian updates
Status: deferred

**Hypothesis:**Full SR1 for overestimated curvature, BFGS for reliable positive secants and PSB otherwise can outperform continuous mixing.
**Outcome and uncertainty:**271 passes energy with all469 converged and zero errors but loses cost substantially. No passive traces or branch histories identify the source of loss.
**Reason to revisit:**Measured regime-specific curvature/model error could justify a different update-selection criterion; pure broad performance loss cannot.
**Next experiment:**Require that diagnostic evidence or a supported curvature-consistency model; no .1 threshold, branch or activation sweep.

**Attempts:**
- Cycle271; candidate `d935f6df17062b48eb473aca8633635acee66b71`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision discard.
  [Implementation](ideas/curvature-flowchart/d935f6df17062b48eb473aca8633635acee66b71/algo.py), [training](evaluation_results/cycle-271-train.json), [narrative](full_log.md#cycle-271-curvature-sign-flowchart-hessian-learning).

## explicit-umbrella-coordinates: Native Fischer out-of-plane geometry
Status: deferred

**Hypothesis:**Realizing supported out-of-plane springs as signed geometry coordinates can improve geodesic motion and secant learning.
**Outcome and uncertainty:**272 passes energy with all469 converged and zero errors but loses cost. No passive traces identify singularities, model error or metric coupling.
**Reason to revisit:**An independently supported redundant-coordinate metric or observed poor umbrella-path realization could justify a targeted new representation. The exact derivative and physical-curvature-matching construction is reusable.
**Next experiment:**Require such evidence; no weight, singularity cutoff or prior-strength sweep.

**Attempts:**
- Cycle272; candidate `3a1009ba9ce83a024a64e00eeaeeddd41ac67bb7`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision discard.
  [Implementation](ideas/explicit-umbrella-coordinates/3a1009ba9ce83a024a64e00eeaeeddd41ac67bb7/algo.py), [training](evaluation_results/cycle-272-train.json), [narrative](full_log.md#cycle-272-explicit-fischer-out-of-plane-geometry-coordinates).

## gradient-regularized-newton: Endpoint-gradient regularization controller
Status: deferred

**Hypothesis:**Gradient-proportional Newton regularization with endpoint-gradient acceptance can adapt efficiently to nonlinear model error.
**Outcome and uncertainty:**274 has broad cost/energy failure,454 converged and15 call limits, no errors. Two retained traces show low forces with large continuing motion, but lack sigma/Hessian/acceptance state; the reason for late inefficient motion is unisolated.
**Reason to revisit:**An independently supported model-error metric or controller diagnosis could make endpoint tests useful for nonconvex molecular quasi-Newton geometry; the convex source does not establish that prerequisite.
**Next experiment:**Await that evidence. No sigma/cap/acceptance sweep after465 slower cases; do not restrict motion merely to trigger stopping. Preserve full gates.
**Attempts:**
- Cycle274; candidate `8942ddb72b3e1335d151d526e02b29ffe1269ae2`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision invalid. [Implementation](ideas/gradient-regularized-newton/8942ddb72b3e1335d151d526e02b29ffe1269ae2/algo.py), [train](evaluation_results/cycle-274-train.json), [narrative](full_log.md#cycle-274-gradient-regularized-newton-macro-optimizer).

## newton-ray-restriction: Curvature-metric ordinary step damping
Status: deferred

**Hypothesis:**Preserve learned coupled Newton motion when shrinking to a component bound instead of changing direction through additive damping.
**Outcome and uncertainty:**276 fully converges with valid energy and no errors but loses cost. No retained successful model states explain whether soft-direction error or interaction with GP causes the loss.
**Reason to revisit:**Independent reliable curvature/anisotropy evidence may make Newton-ray restriction useful; the source's quadratic argument alone is insufficient here.
**Next experiment:**Await such evidence. No damping mixture, eigenvalue floor or activation sweep. Full gates unchanged.
**Attempts:**
- Cycle276; candidate `d4c514cdccf2f6f34f4a2bba418410205e157d49`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision discard. [Implementation](ideas/newton-ray-restriction/d4c514cdccf2f6f34f4a2bba418410205e157d49/algo.py), [train](evaluation_results/cycle-276-train.json), [narrative](full_log.md#cycle-276-curvature-metric-newton-ray-restriction).

## gradient-model-trust-feedback: Gradient prediction for radius feedback
Status: deferred

**Hypothesis:**Use transported signed-Hessian action and paid endpoint gradients to judge vector-model progress and set the trust radius.
**Outcome and uncertainty:**Cycle278 fully converges without errors and passes energy but substantially worsens cost. No successful-path model diagnostics are retained. Native norm scaling and gradient/energy progress disagreement remain unisolated.
**Reason to revisit:**Independent evidence for a physically meaningful residual norm that predicts useful energy progress could support a different globalization rule. Merely retuning thresholds is unsupported.
**Next experiment:**Only after such evidence, derive a consistent residual metric and compare against244 under full gates; broad loss currently ends this configuration.

**Attempts:**
- Cycle278; candidate `cdd906be7932e91cb79142ae7eacbfbc0380f596`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision discard.
  [Implementation](ideas/gradient-model-trust-feedback/cdd906be7932e91cb79142ae7eacbfbc0380f596/algo.py), [training](evaluation_results/cycle-278-train.json), [narrative](full_log.md#cycle-278-transported-gradient-model-trust-feedback).

## posterior-curvature-seed: GP curvature as the next secant seed
Status: deferred

**Hypothesis:**Persist multiple-observation curvature by replacing the GP-covered Hessian block before the newly paid native secant correction.
**Outcome and uncertainty:**279 fully converges without errors and passes energy but substantially worsens cost. Repeated data reuse, derivative mapping and second-derivative noise are unisolated; no successful-path diagnostic states survive.
**Reason to revisit:**Independent evidence of calibrated posterior Hessian quality could justify a different information-transfer rule. Current evidence does not support seed-weight/timing changes.
**Next experiment:**Require that information-quality evidence first; compare a derived assimilation model under the full gates. The exact mapped posterior Hessian remains reusable.

**Attempts:**
- Cycle279; candidate `dc5594591bcd717468be5f0c9b2163b2e3b09a9c`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision discard.
  [Implementation](ideas/posterior-curvature-seed/dc5594591bcd717468be5f0c9b2163b2e3b09a9c/algo.py), [training](evaluation_results/cycle-279-train.json), [narrative](full_log.md#cycle-279-posterior-curvature-seed-for-native-secant-learning).

## two-level-residual-gp: Recent residual GP over an older-data mean
Status: deferred

**Hypothesis:**An older GP mean corrected by a recent GP retains useful history without requiring the new model to interpolate old observations.
**Outcome and uncertainty:**280 fully converges without errors and passes energy but worsens training cost. Successful-path model states are unavailable. Prior-mean uncertainty, historical derivative maps and older-data reliability remain limitations.
**Reason to revisit:**A supported common-chart model or calibrated older-mean uncertainty could improve this information split. More points/levels alone are unsupported.
**Next experiment:**Require that independent improvement first, then compare its standalone and two-level versions against the champion under full gates; no window, depth or kernel-width sweep.

**Attempts:**
- Cycle280; candidate `25ab1b95d5b75249ab5c096740991255077ba23c`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision discard.
  [Implementation](ideas/two-level-residual-gp/25ab1b95d5b75249ab5c096740991255077ba23c/algo.py), [training](evaluation_results/cycle-280-train.json), [narrative](full_log.md#cycle-280-two-level-recent-residual-gp).

## fixed-delocalized-chart: Fixed DIC geometry and history space
Status: deferred

**Hypothesis:**A fixed projection of the complete Badger/collective internal coordinate set aligns the geodesic, gradient and GP history in a common flat chart.
**Outcome and uncertainty:**281 fully converges without errors and passes energy but worsens cost. Successful-path conditioning and model states are absent; the manifest is empty. The chart's global conditioning and projected component bounds remain limitations, not diagnosed failures.
**Reason to revisit:**Independent evidence for when the fixed chart becomes inadequate could justify chart management; a demonstrated common-chart surrogate benefit could support a different model connection.
**Next experiment:**Await that evidence; no rank cutoff, reset frequency or radius sweep after this converged loss. The complete projected value/Jacobian/Hessian and periodic-reference adapter remain reusable.

**Attempts:**
- Cycle281; candidate `7e7ca35605f0be89ef5fc1e12d2dad194b21a0c5`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision discard.
  [Implementation](ideas/fixed-delocalized-chart/7e7ca35605f0be89ef5fc1e12d2dad194b21a0c5/algo.py), [training](evaluation_results/cycle-281-train.json), [narrative](full_log.md#cycle-281-fixed-delocalized-internal-chart).

## derivative-information-selection: Value of the next gradient observation
Status: deferred

**Hypothesis:**Expected information about the relative energy of ordinary and GP candidates can improve query selection beyond immediate predicted energy.
**Outcome and uncertainty:**286 passes energy and fully converges without errors, but loses cost. Successful query covariance/selection states are unavailable; model calibration and the two-alternative horizon remain uncertain.
**Reason to revisit:**Independent calibration of gradient-query information or a derived policy for updating the next search span could make the exact two-choice calculation useful.
**Next experiment:**Await that evidence; no score weights, noise levels or arbitrary alternative-count sweeps.

**Attempts:**
- Cycle286; candidate `23ae77d40823ee8bf35365b234fe28e21877ad08`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision discard.
  [Implementation](ideas/derivative-information-selection/23ae77d40823ee8bf35365b234fe28e21877ad08/algo.py), [train](evaluation_results/cycle-286-train.json), [narrative](full_log.md#cycle-286-derivative-information-selection-between-native-proposals).

## fixed-origin-physical-calibration: Fit one anchored physical potential
Status: deferred

**Hypothesis:**Use current projected gradients of one fixed scalar potential to calibrate physical features, reserving transported secants for replay.
**Outcome and uncertainty:**288 passes energy and fully converges without errors but loses cost. Successful fit states are unavailable; model mismatch and anchor-distance effects are unisolated.
**Reason to revisit:**An independently supported finite-displacement physical potential or observed projection inconsistency could make this observation framework useful.
**Next experiment:**Await that evidence, then compare a derived potential against the current champion; no anchor-reset, weight, history or ridge sweep.

**Attempts:**
- Cycle288; candidate `32bbb6083378e8e6635fafbe0cc8921dce1a1c2f`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision discard.
  [Implementation](ideas/fixed-origin-physical-calibration/32bbb6083378e8e6635fafbe0cc8921dce1a1c2f/algo.py), [train](evaluation_results/cycle-288-train.json), [narrative](full_log.md#cycle-288-fixed-origin-physical-potential-gradient-calibration).

**330 prerequisite:**298/307 independently accepts a finite angular potential, supplying288's deferred physical-model prerequisite. Reuse its anchored-gradient observation relation but replace the angular quadratic by that potential in the initial projected chart. Keep initial Hessian, six-pair replay, regression and GP fixed. This is early coefficient inference, distinct from328's later known-Hessian forecast.

**330 reassessment:**Finite angular gradients satisfy288's independently supported model prerequisite, but complete convergence and valid energy still come with broad cost loss. The anchored chart/linear reference remains approximate and no diagnostic defect supports a repair. Revisit only with independent observation-relation evidence, without fit-weight, ridge or reference-reset sweeps.
- Cycle330; candidate `fa66b235a176525d0a6a806d9153535d1a844df8`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/fixed-origin-physical-calibration/fa66b235a176525d0a6a806d9153535d1a844df8/algo.py), [training](evaluation_results/cycle-330-train.json), [narrative](full_log.md#cycle-330-finite-bending-potential-calibration).

## collision-boundary-refinement: Refine the safe geometric step fraction
Status: deferred

**Hypothesis:**Use the existing geometry-only retry budget to refine a collision bracket and retain more useful proposed motion.
**Outcome and uncertainty:**290 fully converges without errors but fails energy and loses cost; only two call counts change, both slower. The passive manifest is empty. Near-contact model stiffness and trajectory branching are unisolated.
**Reason to revisit:**A separately supported collision-aware direction/model could make boundary refinement useful without pushing an unchanged direction toward a stiff contact.
**Next experiment:**Await that directional model or path evidence; no clearance margin, fraction, contact subset or budget sweep.

**Attempts:**
- Cycle290; candidate `f3531f98d42b7acda9dffe99b3a6e0b73e845fdc`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision invalid.
  [Implementation](ideas/collision-boundary-refinement/f3531f98d42b7acda9dffe99b3a6e0b73e845fdc/algo.py), [train](evaluation_results/cycle-290-train.json), [narrative](full_log.md#cycle-290-bracketed-collision-boundary-refinement).

## step-curvature-information: Learn curvature acting on the proposed step
Status: deferred

**Hypothesis:**Select a query by expected information about H*s, a first-order proxy for Newton-step uncertainty.
**Outcome and uncertainty:**291 fully converges with valid energy and zero errors but loses cost. Native covariance calibration, changing future charts and use of the information remain unobserved.
**Reason to revisit:**Independent calibration of curvature-action uncertainty or a matched posterior-based step learner could give the acquisition a reliable downstream use.
**Next experiment:**Await that coupling or evidence; no information weights, query count, noise or activation sweep. Exact third-derivative conditional information code is reusable.

**Attempts:**
- Cycle291; candidate `f4695afa170855c1103106a4d24b6b8f84676470`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision discard.
  [Implementation](ideas/step-curvature-information/f4695afa170855c1103106a4d24b6b8f84676470/algo.py), [train](evaluation_results/cycle-291-train.json), [narrative](full_log.md#cycle-291-ordinary-step-curvature-information-for-query-selection).

## paid-objective-acceleration: O-ACCEL around paid native steps
Status: deferred

**Hypothesis:**Alternate native optimization with a separately paid objective-stationarity correction from several outer iterates.
**Outcome and uncertainty:**292 fully converges without errors but fails energy and substantially worsens cost. Successful acceleration/model traces are absent; additional-call overhead and Cartesian/native model/globalization mismatch remain unisolated.
**Reason to revisit:**An independently supported common geometry and acceleration-specific globalization could make the paid-point schedule useful, or informative trajectories could isolate a concrete defect.
**Next experiment:**Await that coupling/evidence; no history size, cadence, ridge, cap or activation sweep. The explicit state machine and exact paid-gradient accounting remain reusable.

**Attempts:**
- Cycle292; candidate `147aaf8612893efbd52f5fc04f402e3ad48a2991`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision invalid.
  [Implementation](ideas/paid-objective-acceleration/147aaf8612893efbd52f5fc04f402e3ad48a2991/algo.py), [train](evaluation_results/cycle-292-train.json), [narrative](full_log.md#cycle-292-paid-preconditioner-objective-acceleration).

**Cycle501 revisit:** Apply the preconditioner-plus-acceleration schedule entirely within the fixed GP objective/chart, with cheap bounded Armijo globalization. This supplies the previously missing common geometry and removes extra paid points; no292 cadence/history/ridge/cap tuning.

**501 outcome:** All469 converge,valid energy,zero errors, but cost worsens;{'fast': 4, 'same': 460, 'slow': 5}. The common-GP geometry and cheap globalization do not establish useful acceleration. Await retained inner-search defects; no macro-budget,history,ridge,preconditioner or line-search sweeps.
- Cycle501; candidate `702ade5f55165fd9cadca9072f08fe464436994e`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/paid-objective-acceleration/702ade5f55165fd9cadca9072f08fe464436994e/algo.py), [training](evaluation_results/cycle-501-train.json), [narrative](full_log.md#cycle-501-objective-accelerated-surrogate-search).

## physical-kernel-metric: Separate physical covariance and learned mean
Status: deferred

**Hypothesis:**A fixed primitive physical metric gives steadier residual correlation lengths while learned curvature remains in the GP mean.
**Outcome and uncertainty:**293 fully converges without errors but fails energy and loses cost. No residual/covariance states establish how primitive scaling differs from actual smoothness.
**Reason to revisit:**Independent residual correlation measurements or a derived discrepancy field could support physically specified covariance separate from the mean.
**Next experiment:**Await that evidence; no metric mixtures, normalization, lengths, noise or subgroup sweep. The analytic matrix-precision kernel and derivative implementation remain reusable.

**Attempts:**
- Cycle293; candidate `39cd2b5ffc5e94055de819f22f8c96abc7c4e801`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision invalid.
  [Implementation](ideas/physical-kernel-metric/39cd2b5ffc5e94055de819f22f8c96abc7c4e801/algo.py), [train](evaluation_results/cycle-293-train.json), [narrative](full_log.md#cycle-293-fixed-physical-covariance-metric-with-unchanged-gp-mean).

## burg-physical-calibration: Positive relative-entropy stiffness prior
Status: deferred

**Hypothesis:**Multiplicative positive stiffness ratios benefit from a Burg rather than Gaussian coefficient prior with the same local curvature.
**Outcome and uncertainty:**295 fully converges with valid energy and zero errors, but loses training cost; faster/slower/unchanged=[75, 76, 318]. Empty failure_details does not identify whether nonlinear shrinkage, clipping, or model error causes the loss.
**Reason to revisit:**An independently supported positive coefficient distribution or calibration observation model could make the convex positive-prior solver useful; current physical intuition alone is insufficient.
**Next experiment:**Await that evidence; no prior-power, bound, ridge, history or solver-budget sweep. Compare under unchanged gates.

**Attempts:**
- Cycle295; candidate `d6241cfe2ca6b379ab0137b3f9c421dc3d523a73`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision discard.
  [Implementation](ideas/burg-physical-calibration/d6241cfe2ca6b379ab0137b3f9c421dc3d523a73/algo.py), [training](evaluation_results/cycle-295-train.json), [narrative](full_log.md#cycle-295-burg-prior-for-positive-physical-calibration).

## dual-physical-calibration: Learn inverse response with the native initial model
Status: deferred

**Hypothesis:**Fit displacement response to gradient changes using compliance features derived from the complete native physical Hessian, then invert and replay ordinary secants.
**Outcome and uncertainty:**297 fully converges with valid energy and zero errors but loses cost; faster/slower/unchanged=[98, 253, 118]. Empty failure_details does not isolate response weighting, fixed initial range, or inverse model error.
**Reason to revisit:**A independently supported error model for displacement response or a physical compliance representation could make this bounded feature fit useful. The initial-Hessian-preserving congruence differs from94's rejected prior replacement.
**Next experiment:**Await that evidence; no mixed-loss, reciprocal-bound, ridge, memory or activation sweep. Keep full unchanged gates.

**Attempts:**
- Cycle297; candidate `e421499e15f8e0192b3e4a34e3edab15da3afc16`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision discard.
  [Implementation](ideas/dual-physical-calibration/e421499e15f8e0192b3e4a34e3edab15da3afc16/algo.py), [training](evaluation_results/cycle-297-train.json), [narrative](full_log.md#cycle-297-dual-physical-calibration-from-displacement-response).

## smooth-bending-gp-mean: Source-derived anharmonic angular mean
Status: incorporated

**Hypothesis:**A locally matched smooth bending potential reduces the nonlinear residual that a short-history GP must learn while preserving its current gradient and learned Hessian.
**Outcome and uncertainty:**298 passes both full cost/energy gates with all934 cases converged and zero errors. The aggregate gains are small; no successful model-state traces identify which aspect of the finite bending shape helps.
**Reason to revisit:**The accepted mean provides measured support for this specific angular model and consistent historical derivative treatment, not arbitrary forcefield extensions.
**Next experiment:**Keep the exact accepted construction; test the independently derived DFP correction control against this new champion. No angular weight, phase, stiffness, confidence or microsearch sweep.

**Attempts:**
- Cycle298; candidate `7c0bbf8190d01c916f081326421250f6b036918c`; champion `025b5e2d5f90c39a0116e86fa03d3ff41da036ae`; decision keep. Source retained in accepted Git history.
  [Training](evaluation_results/cycle-298-train.json), [validation](evaluation_results/cycle-298-valid.json), [narrative](full_log.md#cycle-298-smooth-anharmonic-bending-mean-for-the-gp).

**432 source audit:** The accepted formula applies the tanh ratio to both denominator terms; published157/165 applies it only to the reference-sine term. Accepted empirical evidence stands. Cycle432 tests the corrected mean and matched energy coordinate together, preserving local stiffness and chain rules. See [narrative](full_log.md#cycle-432-published-bending-denominator-in-mean-and-covariance).

## dfp-residual-blend: Measured-response DFP component with native SR1 mixing
Status: deferred

**Hypothesis:**A DFP least-change rank-two component may preserve useful inverse response when positive directional curvature is well resolved.
**Outcome and uncertainty:**299 passes energy but broadly loses cost, with448 converged,21 force-call limits and zero errors; faster/slower/unchanged=[46, 310, 113]. Three passive traces show sustained descent with either unconverged force or excessive late motion. No retained Hessian state diagnoses a repair.
**Reason to revisit:**An independently supported inverse-metric/observation model might justify DFP's correction direction; positive secant eligibility alone is insufficient.
**Next experiment:**Defer without denominator, damping, blend or activation sweeps. A future formulation needs curvature-state evidence or an independent model derivation and full unchanged gates.

**Attempts:**
- Cycle299; candidate `3bdb04119ba10f52f0dffcbafeb9dea155814b09`; champion `7c0bbf8190d01c916f081326421250f6b036918c`; decision discard.
  [Implementation](ideas/dfp-residual-blend/3bdb04119ba10f52f0dffcbafeb9dea155814b09/algo.py), [training](evaluation_results/cycle-299-train.json), [manifest](diagnostics/af248e9ea7d646a5ac1d935d0f8a3b84/manifest.json), [narrative](full_log.md#cycle-299-dfp-component-in-residual-aligned-curvature-updates).

## taylor-remainder-gp: Infinite Taylor-series energy remainder
Status: deferred

**Hypothesis:**An origin-centered exponential Taylor remainder learns curvature and higher response while its uncertainty grows away from the current point.
**Outcome and uncertainty:**301 fully converges with valid energy and no errors but loses training cost. No passive conditioning/confidence states identify the source; no validation.
**Reason to revisit:**Independently measured local derivative uncertainty could justify a calibrated Taylor coefficient prior; the analytic covariance and derivative machinery remain reusable.
**Next experiment:**Await that evidence; no amplitude, scale, degree, noise or confidence sweep. Repeat full gates against the current champion for a supported revision.

**Attempts:**
- Cycle301; candidate `f786a5e05813c1b339444e0f7c70a4bbe16ac3be`; champion `7c0bbf8190d01c916f081326421250f6b036918c`; decision discard.
  [Implementation](ideas/taylor-remainder-gp/f786a5e05813c1b339444e0f7c70a4bbe16ac3be/algo.py).
  [Training evidence](evaluation_results/cycle-301-train.json).
  [Narrative](full_log.md#cycle-301-taylor-remainder-covariance-for-the-local-gp).

## polar-multisecant-replay: Joint polar reconciliation of early secants
Status: deferred

**Hypothesis:**Polar overlap reconciliation fits early observations jointly while retaining the newest equation and calibrated physical seed.
**Outcome and uncertainty:**302 fully converges with valid energy and no errors, but loses training cost broadly. No saved overlap/factorization state isolates a repair.538 supplies fixed GP secants but also loses cost,4faster6slower459same, all converged and valid.
**Reason to revisit:**The common-chart prerequisite was tested in538 and is insufficient. Independent evidence of benign overlap spectra versus endpoint model error could make this batch construction useful. Its PSD polar kernel and exact-newest relation remain reusable.
**Next experiment:**Await such evidence; no threshold, damping, pair-count or timing sweep. Use full gates against the current champion for any justified revision.

**Attempts:**
- Cycle302; candidate `3b1dc31ede787e483c48951414d979a91ecf1d1f`; champion `7c0bbf8190d01c916f081326421250f6b036918c`; decision discard.
  [Implementation](ideas/polar-multisecant-replay/3b1dc31ede787e483c48951414d979a91ecf1d1f/algo.py).
  [Training evidence](evaluation_results/cycle-302-train.json).
  [Narrative](full_log.md#cycle-302-polar-multisecant-replay-of-the-calibrated-prior).

**Revisit538 prerequisite:** A GP-only search supplies302's explicitly requested consistent common secant chart. Reuse its exact-newest polar kernel and guards on accepted model steps, with an identity physical-normalized seed; no native physical secant replay or timing/threshold tuning. Gubarev2026sections2–3 corroborate the construction and motivate oldest-pair removal for rank loss. Compare with463 under full gates.

- Cycle538; candidate `b72acca4977fbe19d3f3a6f9c7d5325944ab3a21`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/polar-multisecant-replay/b72acca4977fbe19d3f3a6f9c7d5325944ab3a21/algo.py), [training](evaluation_results/cycle-538-train.json), [narrative](full_log.md#cycle-538-common-chart-polar-multisecant-surrogate-search).

## geodesic-log-gp: Cubic geodesic logarithm and its differential
Status: deferred

**Hypothesis:**Correct projected historical chords toward current geodesic initial tangents and differentiate the same chart for gradient observations.
**Outcome and uncertainty:**303 converges all469 with valid energy and no errors but slightly loses cost. Empty diagnostics provide no localized repair; approximation order and reduced-span effects remain unmeasured.
**Reason to revisit:**A measured finite-path chart discrepancy or an independently supported full logarithm construction could justify reusing the primitive-HVP shape operator and analytic differential.
**Next experiment:**Await that evidence; no correction-weight, degree, span, history-range or activation sweep. Repeat full unchanged gates for a justified revision.

**Attempts:**
- Cycle303; candidate `c25552cf488daae49230c33d1fdb6821e9c00b83`; champion `7c0bbf8190d01c916f081326421250f6b036918c`; decision discard.
  [Implementation](ideas/geodesic-log-gp/c25552cf488daae49230c33d1fdb6821e9c00b83/algo.py).
  [Training evidence](evaluation_results/cycle-303-train.json).
  [Narrative](full_log.md#cycle-303-cubic-geodesic-logarithm-for-gp-history).

## discrete-gradient-gp: Finite dissipative surrogate flow step
Status: deferred

**Hypothesis:**A quadratic-matched Gonzalez discrete-gradient step can use nonlinear GP information more reliably than its stationary point.
**Outcome and uncertainty:**304 fully converges with valid energy and no errors, but loses cost. Empty passive diagnostics do not distinguish solver fallback from unhelpful solved proposals.
**Reason to revisit:**Direct surrogate-solve evidence or an independently derived globally convergent scalar-model solver could make the dissipative equation useful.
**Next experiment:**Await that evidence; no time, relaxation, solver-budget, tolerance or confidence sweep. Repeat full unchanged gates for any justified revision.

**Attempts:**
- Cycle304; candidate `8a31fa7ecd1408ef053a3426852ef33995472980`; champion `7c0bbf8190d01c916f081326421250f6b036918c`; decision discard.
  [Implementation](ideas/discrete-gradient-gp/8a31fa7ecd1408ef053a3426852ef33995472980/algo.py).
  [Training evidence](evaluation_results/cycle-304-train.json).
  [Narrative](full_log.md#cycle-304-dissipative-discrete-gradient-gp-proposal).


**Revisit552:** A source-supported sequential Itoh–Abe construction replaces the coupled Gonzalez equation by scalar difference-quotient roots. Bracketing plus Brent solves supplies a concrete independent solver mechanism rather than changing304's fixed-point relaxation or time. Keep time2 matched to unit quadratic and all463critics. This structural revisit is not a passive diagnosis of304.

**552 reassessment:** Sequential scalar implicit equations still lose cost (24faster,34slower,411same); all converge with validenergy andzeroerrors. No passive model-state evidence distinguishes root termination from proposal quality. Await independently diagnosed model-solver error; no time,order,tolerance or budget sweep.
- Cycle552; candidate `843c3acd2b6ce822dbcc0240e7afa841ba1c208f`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/discrete-gradient-gp/843c3acd2b6ce822dbcc0240e7afa841ba1c208f/algo.py), [training](evaluation_results/cycle-552-train.json), [narrative](full_log.md#cycle-552-coordinate-discrete-gradient-surrogate-flow).

## exponential-contact-gp-mean: Integrated native auxiliary repulsion in the GP mean
Status: deferred

**Hypothesis:**The native exponential contact stiffness determines a higher-order repulsive mean correction without changing the current local model.
**Outcome and uncertainty:**306 has sufficient training cost gain but fails aggregate energy despite all469 converged and no errors. One endpoint dominates energy loss; empty passive diagnostics do not reveal its cause.
**Reason to revisit:**A measured finite-contact chart discrepancy or independent repulsion/attraction decomposition could improve the promising cost effect. The current model uses radial tangents and repulsion only.
**Next experiment:**Await supporting evidence; no exponent, amplitude, cutoff, support or stopping sweeps. Any justified repair must pass full unchanged gates.

**Attempts:**
- Cycle306; candidate `b8d77e656c65b9fa99046da36a7523386c5229ea`; champion `7c0bbf8190d01c916f081326421250f6b036918c`; decision invalid.
  [Implementation](ideas/exponential-contact-gp-mean/b8d77e656c65b9fa99046da36a7523386c5229ea/algo.py).
  [Training evidence](evaluation_results/cycle-306-train.json).
  [Narrative](full_log.md#cycle-306-exponential-auxiliary-contact-mean-for-the-gp).

**327 prerequisite:**A full source-defined UFF12–6 potential supplies the attraction/repulsion alternative. Its1–3 exclusion is imposed on the unchanged auxiliary candidate pair set; native Hessian support is unchanged. Test the finite mean with zero current second-order correction and307 covariance. This is an independent potential-shape test, not a localized diagnosis of306.

**327 reassessment:**The attractive/repulsive UFF mean passes energy with full convergence but loses training cost. The independent decomposition prerequisite is now tested; source-consistent1–3 exclusion does not yield a gain. Revisit now requires measured finite-chart discrepancy or another independently supported contact model, not amplitude/radius/cutoff/support tuning.
- Cycle327; candidate `103b71d772a0fca1dd33e7aebe86a593f6f68900`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard.
  [Implementation](ideas/exponential-contact-gp-mean/103b71d772a0fca1dd33e7aebe86a593f6f68900/algo.py). [Training evidence](evaluation_results/cycle-327-train.json). [Narrative](full_log.md#cycle-327-attractive-repulsive-contact-gp-mean).

## bending-energy-gp-coordinates: Physical bending shape in residual covariance
Status: incorporated

**Hypothesis:**The accepted anharmonic bending potential defines locally normalized nonlinear coordinates for a more uniform GP residual metric.
**Outcome and uncertainty:**307 passes both cost/energy gates, fully converged and error-free. Most cases are unchanged, so endpoint evidence supports this particular warp but does not identify its pathwise mechanism.
**Reason to revisit:**The accepted coordinate gives a measured basis for future compatible scalar means; historical and query derivatives must continue to use the same chain rule.
**Next experiment:**Retain exactly307 and test a separately derived scalar angle–torsion coupling. No warp-strength, scale or confidence sweep.

**Attempts:**
- Cycle307; candidate `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; champion `7c0bbf8190d01c916f081326421250f6b036918c`; decision keep. Source retained in accepted Git history.
  [Training evidence](evaluation_results/cycle-307-train.json).
  [Validation evidence](evaluation_results/cycle-307-valid.json).
  [Narrative](full_log.md#cycle-307-bending-energy-coordinates-for-gp-covariance).


**432 source audit:** The accepted formula applies the tanh ratio to both denominator terms; published157/165 applies it only to the reference-sine term. Accepted empirical evidence stands. Cycle432 tests the corrected mean and matched energy coordinate together, preserving local stiffness and chain rules. See [narrative](full_log.md#cycle-432-published-bending-denominator-in-mean-and-covariance).

## angle-modulated-torsion-mean: Finite contained-angle coupling in the GP mean
Status: deferred

**Hypothesis:**Published smooth angular envelopes modulate local torsional stiffness without altering the current Taylor model.
**Outcome and uncertainty:**308 passes energy, converges all469 without errors, but loses cost. Empty failure_details supplies no localized model/domain diagnosis.
**Reason to revisit:**Independently measured mixed angle–torsion response or a supported harmonic mixture could identify a suitable amplitude model. This local n=1 envelope is not the full ADDT potential.
**Next experiment:**Await such evidence; no weight, harmonic order, domain, support or activation sweep. Full unchanged gates apply to any revision.

**Attempts:**
- Cycle308; candidate `a50f378f1b101efa5334d63c6d56ec62a8728b55`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard.
  [Implementation](ideas/angle-modulated-torsion-mean/a50f378f1b101efa5334d63c6d56ec62a8728b55/algo.py).
  [Training evidence](evaluation_results/cycle-308-train.json).
  [Narrative](full_log.md#cycle-308-angle-modulated-torsional-gp-mean).


## coupled-bending-coordinate-mean: Squared nonlinear-coordinate mean
Status: deferred

**Hypothesis:**The accepted bending-energy map can express a coupled mean using the learned local quadratic.
**Outcome and uncertainty:**309 passes energy and converges all469 without errors, but loses cost. Empty failure_details does not isolate finite-curvature calibration or reduced-span projection effects.
**Reason to revisit:**Independent evidence for a coupled nonlinear energy representation could justify this reusable scalar/gradient construction; covariance success alone is insufficient.
**Next experiment:**Await that evidence; no mean-blending or warp-strength sweep. The independent covariance-only ablation is next, to assess the accepted model's interaction.

**Attempts:**
- Cycle309; candidate `d35e390358d375401f8d55aa2fe4046c5c476f91`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard.
  [Implementation](ideas/coupled-bending-coordinate-mean/d35e390358d375401f8d55aa2fe4046c5c476f91/algo.py).
  [Training evidence](evaluation_results/cycle-309-train.json).
  [Narrative](full_log.md#cycle-309-coupled-mean-in-bending-energy-coordinates).


## quadratic-mean-bending-covariance: Covariance-only factorial control
Status: deferred

**Hypothesis:**The accepted angular covariance might make the independent nonlinear mean unnecessary.
**Outcome and uncertainty:**310 fully converges, passes energy and has no errors, but loses training cost. This ablation favors the joint307 model; empty diagnostics cannot identify its pathwise mechanism.
**Reason to revisit:**This control is reusable if a separately accepted nonlinear energy representation replaces the mean's role; current evidence supports retaining307 unchanged.
**Next experiment:**No immediate follow-up or partial mean weight. Await such an independent model change and use full gates.

**Attempts:**
- Cycle310; candidate `334fddf7eefe59a559e6c50327c7fb0a99f9be5d`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard.
  [Implementation](ideas/quadratic-mean-bending-covariance/334fddf7eefe59a559e6c50327c7fb0a99f9be5d/algo.py).
  [Training evidence](evaluation_results/cycle-310-train.json).
  [Narrative](full_log.md#cycle-310-quadratic-mean-control-with-bending-covariance).


## physical-spectral-secant: Calibrated physical seed with spectral scaling
Status: deferred

**Hypothesis:**A physical preconditioner and short nonlinear GP could replace accumulated dense secant curvature with a single spectral scale.
**Outcome and uncertainty:**311 fails energy and cost, with409 slower cases and six call limits. The converged subset also loses substantially. Two passive bundles show continuing oscillatory progress but no scalar/fallback states.
**Reason to revisit:**An independently improved anisotropic preconditioner or evidence identifying missing curvature modes could make a low-memory spectral model useful. Current scalar scaling cannot learn cross-mode structure.
**Next experiment:**Await that evidence; no BB-family, clipping, activation or radius sweep. Any revised model must pass full unchanged gates.

**Attempts:**
- Cycle311; candidate `1f52cb5bbbe048e610dd798fd51bf7c53076b1d1`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision invalid.
  [Implementation](ideas/physical-spectral-secant/1f52cb5bbbe048e610dd798fd51bf7c53076b1d1/algo.py).
  [Training evidence](evaluation_results/cycle-311-train.json).
  [Narrative](full_log.md#cycle-311-physical-preconditioner-with-spectral-secant-scaling).


## calibrated-bending-mean: Share the fitted physical angular coefficient
Status: deferred

**Hypothesis:**The existing paid-secant angular calibration also estimates higher-order bending amplitude.
**Outcome and uncertainty:**312 converges all469 with valid energy and no errors, but slightly loses cost. Empty failure_details cannot distinguish finite-amplitude mismatch from coefficient path dependence.
**Reason to revisit:**Direct evidence linking fitted harmonic and anharmonic response could justify shared calibration; current physical block fitting alone does not.
**Next experiment:**Await that evidence; no mean-weight, damping, fit-bound or timing sweep.

**Attempts:**
- Cycle312; candidate `81c0808c98584aacbcec3a942a76471b0a6b9631`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard.
  [Implementation](ideas/calibrated-bending-mean/81c0808c98584aacbcec3a942a76471b0a6b9631/algo.py).
  [Training evidence](evaluation_results/cycle-312-train.json).
  [Narrative](full_log.md#cycle-312-share-fitted-angular-amplitude-with-the-gp-mean).


## normal-bending-features: Keep the finite angular map outside the search span
Status: deferred

**Hypothesis:**Retaining discarded nonlinear angular components makes residual similarities more physically faithful.
**Outcome and uncertainty:**313 fully converges with valid energy and no errors, but loses cost. Empty diagnostics cannot identify a normal-metric or over-separation mechanism.
**Reason to revisit:**Independent covariance-distance evidence could support a complete nonlinear embedding; the rectangular derivative kernel remains reusable with analytic features.
**Next experiment:**Await that evidence; no normal weights, diagonal strength, feature support or kernel length sweep.

**Attempts:**
- Cycle313; candidate `d3e182333b432ea82db1f510e79aff0abe1295fe`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard.
  [Implementation](ideas/normal-bending-features/d3e182333b432ea82db1f510e79aff0abe1295fe/algo.py).
  [Training evidence](evaluation_results/cycle-313-train.json).
  [Narrative](full_log.md#cycle-313-retain-normal-bending-features-in-gp-covariance).


## bending-coordinate-microsearch: Change variables in the GP solver
Status: deferred

**Hypothesis:**The accepted angular energy coordinate can flatten finite response during BFGS microsearch.
**Outcome and uncertainty:**314 fully converges with valid energy and no errors, but loses cost. No saved inverse/solver paths identify a branch or fallback cause.
**Reason to revisit:**Direct evidence of conditioning or inverse-chart failures could support a different derived local chart; the analytic inverse gradient transformation is reusable.
**Next experiment:**Await that evidence; no inverse/root/search budget, tolerance, starting-guess or scale sweep.

**Attempts:**
- Cycle314; candidate `f04cdc19b92e1f4e1cca8a269ec9167ed5d735e7`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard.
  [Implementation](ideas/bending-coordinate-microsearch/f04cdc19b92e1f4e1cca8a269ec9167ed5d735e7/algo.py).
  [Training evidence](evaluation_results/cycle-314-train.json).
  [Narrative](full_log.md#cycle-314-optimize-the-gp-in-bending-energy-coordinates).


## full-space-bending-model: Nonlinear physical ordinary proposal
Status: deferred

**Hypothesis:**The accepted bending shape can improve full free-space proposals before GP refinement.
**Outcome and uncertainty:**315 fully converges with valid energy and no errors but loses cost. Empty diagnostics do not distinguish dependence on the GP residual from unsupported full-space directions.
**Reason to revisit:**Independent evidence for standalone finite physical-model accuracy could make this full-dimensional refinement useful. Its exact native-spectrum whitening and scalar gradient remain reusable.
**Next experiment:**Await that evidence; no amplitude, activation, boundary, norm or solver-budget sweep.

**Attempts:**
- Cycle315; candidate `150252338fb2861bf46ff1b9274e7d1174bafe50`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard.
  [Implementation](ideas/full-space-bending-model/150252338fb2861bf46ff1b9274e7d1174bafe50/algo.py).
  [Training evidence](evaluation_results/cycle-315-train.json).
  [Narrative](full_log.md#cycle-315-full-space-anharmonic-ordinary-step-model).


## oriented-signed-newton: Preserve and orient the signed inverse-curvature ray
Status: deferred

**Hypothesis:**Whole-ray descent orientation preserves useful relative mode curvature lost through individual reflection.
**Outcome and uncertainty:**316 passes energy and fully converges without errors but substantially loses cost. No successful Hessian/globalization paths isolate a failure mechanism.
**Reason to revisit:**An independently supported signed-curvature accuracy or inexpensive Wolfe globalization mechanism could make the published ray treatment useful.
**Next experiment:**Await that evidence; no sign mixtures, spectral floors, radius or activation sweep. Native-controller results do not test the source's full line-search algorithm.

**Attempts:**
- Cycle316; candidate `e15db8f91ad1cf44fec07232798263dccf266790`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard.
  [Implementation](ideas/oriented-signed-newton/e15db8f91ad1cf44fec07232798263dccf266790/algo.py).
  [Training evidence](evaluation_results/cycle-316-train.json).
  [Narrative](full_log.md#cycle-316-oriented-signed-newton-ray-within-native-bounds).

## paid-negative-mode-probe: One paid directional curvature measurement
Status: deferred

**Hypothesis:**A single downhill probe of the first learned negative mode can improve subsequent geodesic secant information enough to repay its force-call cost.
**Outcome and uncertainty:**317 fails before probing with159 undefined-Bohr exceptions. This is an implementation defect, not a scientific rejection.
**Reason to revisit:**The unambiguous passive traceback admits a one-line namespace repair without changing the hypothesis.
**Next experiment:**Bounded repair1 uses units.Bohr; compare full training with307. Broad cost/energy loss ends this variant without displacement, cadence, mode-count or activation sweeps.

**Attempts:**
- Cycle317; candidate `3ce44037095f6001fc701b09bd03a8505732df0e`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision invalid.
  [Implementation](ideas/paid-negative-mode-probe/3ce44037095f6001fc701b09bd03a8505732df0e/algo.py).
  [Training evidence](evaluation_results/cycle-317-train.json).
  [Narrative](full_log.md#cycle-317-one-paid-probe-of-the-first-learned-negative-mode).

**Repair1 reassessment:**318 qualifies units.Bohr and converges every case without errors. Valid energy accompanies broad cost loss, so the scientific hypothesis is unsupported in this once-only native-secanted variant. Revisit only with an independently supported measurement-to-curvature learner that can plausibly repay the paid call; no displacement, cadence, mode-count or force-threshold sweep.
- Cycle318; candidate `77e7709153d460dbd29e96a22c16eef19cf121ed`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard, bounded repair1.
  [Implementation](ideas/paid-negative-mode-probe/77e7709153d460dbd29e96a22c16eef19cf121ed/algo.py), [training](evaluation_results/cycle-318-train.json), [narrative](full_log.md#cycle-318-repair-the-paid-probe-unit-namespace).

**389 prerequisite:** Test a symmetric initial physical-Newton Hessian action with an explicit nonstationary native pullback, learned through the accepted physical calibration. This separates actual differential curvature from318's extra finite downhill trajectory. Two paid calls; unchanged full gates, no displacement or timing sweep after broad loss.

**389 reassessment:** Symmetric initial curvature action fully converges but fails energy and costs more; [27,377,65] faster/slower/unchanged. Removing a nominal two-call overhead from saved counts leaves only0.003316 cost gain, so its information fails to pay for measurement. One higher endpoint has159.015kJ/mol regression; emptyfailure_details cannot diagnose the trajectory. Await independently demonstrated information-to-cost value; no displacement,timing,direction or weighting sweep.
- Cycle389; candidate `10254b119499467dac7d0f3078a822647b65718f`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision invalid. [Implementation](ideas/paid-negative-mode-probe/10254b119499467dac7d0f3078a822647b65718f/algo.py), [train](evaluation_results/cycle-389-train.json), [narrative](full_log.md#cycle-389-paid-initial-symmetric-curvature-action).

## force-consistent-bending-mean: Nonstationary angular energy-coordinate trend
Status: deferred

**Hypothesis:**Expressing the current angular force in the accepted energy coordinate, with exact second-order compensation, improves finite GP mean accuracy.
**Outcome and uncertainty:**319 fully converges with valid energy and no errors, but loses cost. Redundant primitive force allocation and finite model accuracy are unobserved.
**Reason to revisit:**Independent evidence for a physically meaningful angular force decomposition could make this analytically compensated remainder useful.
**Next experiment:**Await that evidence; no weight, coefficient clamp or activation sweep.

**Attempts:**
- Cycle319; candidate `f94d0f12293962dcef9d683ca36be440332f239d`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard.
  [Implementation](ideas/force-consistent-bending-mean/f94d0f12293962dcef9d683ca36be440332f239d/algo.py), [training](evaluation_results/cycle-319-train.json), [narrative](full_log.md#cycle-319-force-consistent-bending-energy-gp-mean).

**401 revisit:** Use the independently derived minimum complementary elastic-energy allocation c=sqrt(D)*(J.T*sqrt(D))^+*(J.T*g), with champion physical D. This replaces319's Euclidean gauge only in its compensated mean remainder; no fitted strength or change to native forces/Hessian. Compare full gates against307.

**401 reassessment:** Elastic force allocation gives full convergence/valid energy but loses cost slightly;[7, 7, 455] faster/slower/unchanged, no errors or failure traces. This tests the model-defined allocation prerequisite without improvement; await independent finite-response evidence, no weighting/rank/clamp/timing sweep.
- Cycle401; candidate `a78383eb5a7135a1152e72855db9760bcde13b65`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/force-consistent-bending-mean/a78383eb5a7135a1152e72855db9760bcde13b65/algo.py), [train](evaluation_results/cycle-401-train.json), [narrative](full_log.md#cycle-401-elastic-force-allocation-for-the-nonlinear-bending-mean).

## finite-stretch-bend-mean: Nonlinear continuation of calibrated mixed curvature
Status: deferred

**Hypothesis:**A finite bond-length/bending-energy product can extend the accepted signed stretch-bend fit beyond its quadratic role.
**Outcome and uncertainty:**320 fully converges with valid energy and no errors, but loses cost. Primitive incidence and fitted local coefficients may not describe finite mixed energy.
**Reason to revisit:**Independent nonlinear mixed-response evidence could identify a useful physical cross model; exact inverse-Badger length features and mean derivative remain reusable.
**Next experiment:**Await that evidence; no amplitude, exponent, coefficient or activation sweep.

**Attempts:**
- Cycle320; candidate `112bf465734f553f8d4f58c469e85de5ed97cf99`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard.
  [Implementation](ideas/finite-stretch-bend-mean/112bf465734f553f8d4f58c469e85de5ed97cf99/algo.py), [training](evaluation_results/cycle-320-train.json), [narrative](full_log.md#cycle-320-finite-physical-stretch-bend-gp-mean).

## bending-response-subspace: Physical nonlinear gradient enrichment
Status: deferred

**Hypothesis:**The accepted bending remainder can identify an additional useful GP direction outside ordinary/history motion.
**Outcome and uncertainty:**322 fully converges with valid energy and no errors, but loses cost. Transverse information and changed local whitening/rank are unobserved.
**Reason to revisit:**An independently supported directional observation model could make this cheap nonlinear response useful.
**Next experiment:**Await that evidence; no direction weight, rank cutoff, history or admission sweep.

**Attempts:**
- Cycle322; candidate `c0ab330ed2dd8cbac279f2100b7b02bc34d7ca31`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard.
  [Implementation](ideas/bending-response-subspace/c0ab330ed2dd8cbac279f2100b7b02bc34d7ca31/algo.py), [training](evaluation_results/cycle-322-train.json), [narrative](full_log.md#cycle-322-bending-response-direction-for-gp-subspace).

## physical-rkcd-descent: Stabilized actual-gradient integration
Status: deferred

**Hypothesis:**Chebyshev polynomial stages with a physical Jacobi metric can damp stiff modes efficiently without Hessian learning or a GP.
**Outcome and uncertainty:**323 is invalid:190 backend numerical errors and155 call limits; all279 returned cases lose cost, including124 converged. Two passive bundles show oscillations. Initial spectral estimates, changing stiffness and Cartesian geometry remain confounded; this is no localized implementation defect.
**Reason to revisit:**A changing nonlinear preconditioner with a justified stability controller could make polynomial stages useful; it must address slow converged cases as well as unsafe geometries.
**Next experiment:**Only after deriving such a controller, compare its full paid-stage cost against307. Do not sweep damping/stage counts on the current formulation.

**Attempts:**
- Cycle323; candidate `b23f9aab985614bc78db5d2702a0c4b24f45438d`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision invalid.
  [Implementation](ideas/physical-rkcd-descent/b23f9aab985614bc78db5d2702a0c4b24f45438d/algo.py). [Training evidence](evaluation_results/cycle-323-train.json). [Narrative](full_log.md#cycle-323-physical-jacobi-preconditioned-rkcd-descent).

## bond-energy-gp-coordinates: Finite Badger energy in residual covariance
Status: deferred

**Hypothesis:**Use a normalized signed square-root of the integrated inverse-cubic bond potential in the GP covariance, preserving the current metric and accepted angular map.
**Outcome and uncertainty:**325 passes training but loses validation cost; all934 converge with valid energy and no errors. The native arc length may already suffice, and the finite potential model is not verified by equilibrium stiffness agreement.
**Reason to revisit:**Observed finite bond response could support a physically identified feature shape beyond the equilibrium stiffness relation. The analytic rational map is reusable.
**Next experiment:**Require such independent model evidence before a new shape; no amplitude/exponent or activation sweep and no automatic combination with other rejected covariance maps.

**Attempts:**
- Cycle325; candidate `ba2800819476f8298e3bd1c8c515643c8308a92c`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision non_generalizable.
  [Implementation](ideas/bond-energy-gp-coordinates/ba2800819476f8298e3bd1c8c515643c8308a92c/algo.py). [Training evidence](evaluation_results/cycle-325-train.json). [Validation evidence](evaluation_results/cycle-325-valid.json). [Narrative](full_log.md#cycle-325-badger-energy-coordinates-for-gp-covariance).

## condition-bounded-gp: Standardized covariance with a bounded condition number
Status: deferred

**Hypothesis:**Prior-variance scaling and a row-bound nugget preserve more derivative information than a fixed absolute nugget while controlling solve conditioning.
**Outcome and uncertainty:**326 fully converges with valid energy but loses training cost. No solver/model states diagnose whether reduced regularization amplified chart or mean mismatch.
**Reason to revisit:**An independently supported observation-error model could separate necessary statistical regularization from numerical stabilization; the standardized solve is reusable.
**Next experiment:**Require that model before another attempt; no condition-limit or nugget sweep.

**Attempts:**
- Cycle326; candidate `28b79e7c2f8954d0c4f3a1042d75c7abcca45c3e`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard.
  [Implementation](ideas/condition-bounded-gp/28b79e7c2f8954d0c4f3a1042d75c7abcca45c3e/algo.py). [Training evidence](evaluation_results/cycle-326-train.json). [Narrative](full_log.md#cycle-326-scale-aware-bounded-gp-conditioning).

## evidence-selected-bending-kernel: Choose native or bending covariance from paid observations
Status: deferred

**Hypothesis:**Conditional marginal likelihood can identify when the nonlinear angular warp improves local residual modeling.
**Outcome and uncertainty:**329 fully converges, passes energy and has no errors, but loses cost. Short-history selection and unmodeled chart discrepancy remain uncertainties; no predictive traces identify a repair.
**Reason to revisit:**Independent evidence that a model-selection statistic predicts query error could justify a revised selector; the shared-observation two-model implementation is reusable.
**Next experiment:**Await that evidence; no prior odds, temperature, length or noise sweep. Any revision must pass full gates against the champion.

**Attempts:**
- Cycle329; candidate `376727986a8561bf2367a11e466f9235234489db`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/evidence-selected-bending-kernel/376727986a8561bf2367a11e466f9235234489db/algo.py), [training](evaluation_results/cycle-329-train.json), [narrative](full_log.md#cycle-329-evidence-selected-bending-covariance).

## native-bending-energy-coordinates: Native signed angular energy chart
Status: deferred

**Hypothesis:**Move307's physical angular nonlinearity into actual geodesics and curvature learning while avoiding a second GP warp.
**Outcome and uncertainty:**331 fully converges without errors but fails aggregate energy and narrowly loses cost. Several changed endpoints dominate energy loss; failure_details and the manifest are empty, so singularity or chart defects are unconfirmed.
**Reason to revisit:**A supported chart/branch mechanism or passive trajectory diagnosis could justify a bounded revision. The analytic native derivative adapter is reusable.
**Next experiment:**Await that evidence; no chart amplitude, linear-angle threshold or reference-reset sweep. Any revision requires the full unchanged gates.

**Attempts:**
- Cycle331; candidate `f02c7d8bbf6128cc55b8766945cfd8e52ab2d401`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision invalid. [Implementation](ideas/native-bending-energy-coordinates/f02c7d8bbf6128cc55b8766945cfd8e52ab2d401/algo.py), [training](evaluation_results/cycle-331-train.json), [diagnostic manifest](diagnostics/43e419f7d5114dc397acb7c30eb2fecb/manifest.json), [narrative](full_log.md#cycle-331-native-bending-energy-geometry).

## redundant-quaternion-geometry: Four-component native fragment orientation
Status: deferred

**Hypothesis:**A unit-quaternion sphere embedding can improve finite rotational geometry and curvature learning while matching initial angular scale.
**Outcome and uncertainty:**332 fully converges without errors and passes energy, but loses cost. Finite trust geometry and redundant normal curvature remain unobserved; no passive failure supports a bounded repair.
**Reason to revisit:**A supported tangent-space treatment of redundancy or a diagnosed branch mechanism could make the exact quaternion derivative adapter useful.
**Next experiment:**Await that evidence; no scale, trust or branch threshold sweep. Compare any revision with the current champion under full gates.

**Attempts:**
- Cycle332; candidate `449f481abe5598393882c7e2c99172746854977c`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/redundant-quaternion-geometry/449f481abe5598393882c7e2c99172746854977c/algo.py), [training](evaluation_results/cycle-332-train.json), [narrative](full_log.md#cycle-332-redundant-quaternion-fragment-geometry).

## ring-torsion-calibration: Separate cyclic and graph-bridge torsion scales
Status: deferred

**Hypothesis:**Ring closure creates a distinct primitive torsion-model error that early secants can identify independently.
**Outcome and uncertainty:**333 fully converges without errors but fails aggregate energy and slightly loses cost. No passive artifacts diagnose coefficient identifiability or changed basin selection.
**Reason to revisit:**Independent evidence of class-specific torsional response or a supported collective ring model could make this partition useful.
**Next experiment:**Await that evidence; no ring-size, element or regularization sweep. Preserve the full gates.

**Attempts:**
- Cycle333; candidate `425404d5a257ee8393b5533b99db8fc9f23931ac`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision invalid. [Implementation](ideas/ring-torsion-calibration/425404d5a257ee8393b5533b99db8fc9f23931ac/algo.py), [training](evaluation_results/cycle-333-train.json), [diagnostic manifest](diagnostics/64159088d1c54ecbbd1ecfdd3d8a8388/manifest.json), [narrative](full_log.md#cycle-333-separate-ring-torsion-calibration).

## collective-ring-puckering: Native Cartesian puckering amplitudes
Status: deferred

**Hypothesis:**Collective ring geometry can improve native geodesics and curvature learning without adding physical spring energy.
**Outcome and uncertainty:**334 converges all training cases with valid energy and no errors, but loses force-call cost and adds runtime. Redundancy, finite bounds and learned metric effects remain unisolated.
**Reason to revisit:**A supported rank-preserving ring replacement or a diagnosed path/model interaction could use the exact scalar-coordinate derivative implementation.
**Next experiment:**Await that derivation/evidence; no coordinate weights, ring-size cutoffs or trust sweep. Preserve the full gates.

**Attempts:**
- Cycle334; candidate `6bd75769d4c96ac55704649b8d0187f7f10ac3b4`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/collective-ring-puckering/6bd75769d4c96ac55704649b8d0187f7f10ac3b4/algo.py), [training](evaluation_results/cycle-334-train.json), [narrative](full_log.md#cycle-334-collective-ring-puckering-geometry).

**345 prerequisite:**Replace consecutive endocyclic torsions with334's collective puckering coordinates under an unchanged absolute-rank audit. Retain their initial Cartesian spring factors through the new Jacobian pseudoinverse and the existing torsion fitting class. This removes augmentation redundancy while preserving physical spring response; finite-path conditioning still requires remote evidence.

**345 reassessment:**Curvature-preserving ring replacement fully converges and lowers cost, but fails energy;54 faster/43 slower/372 unchanged. One higher-energy endpoint supplies most energy loss and much of the gain. Empty passive manifest offers no localized defect to repair. Revisit only with a supported finite-path/curvature compatibility mechanism, without coordinate weights, rank thresholds, ring subsets or trust sweeps.
- Cycle345; candidate `2f0fde58f5d133c2e53113a7561aadce5f68b05e`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision invalid. [Implementation](ideas/collective-ring-puckering/2f0fde58f5d133c2e53113a7561aadce5f68b05e/algo.py), [training](evaluation_results/cycle-345-train.json), [manifest](diagnostics/1fdcefc75e50430bba1933bab6b9cd08/manifest.json), [narrative](full_log.md#cycle-345-curvature-preserving-ring-coordinate-replacement).

## collision-direction-projection: Cure blocked motion through tangent inequalities
Status: deferred

**Hypothesis:**Project a collision-blocked native direction onto linearized pair-clearance constraints to retain useful motion elsewhere.
**Outcome and uncertainty:**335 fully converges with valid energy and no errors; the cost gain0.0000520047844401 misses the fixed training gate. No passive failure diagnoses a repair or indicates how often the correction succeeds.
**Reason to revisit:**An independently supported finite-path constraint model could address nonlinear geodesic collisions that the initial tangent inequalities cannot identify.
**Next experiment:**Await that derivation or passive diagnosis; no activation, margin, solver-budget or contact-subset sweep and no validation of335.

**Attempts:**
- Cycle335; candidate `7b26dd5bcf87ed7da6059a4dcb11df8435db3341`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/collision-direction-projection/7b26dd5bcf87ed7da6059a4dcb11df8435db3341/algo.py), [training](evaluation_results/cycle-335-train.json), [narrative](full_log.md#cycle-335-project-collision-blocked-directions).

## finite-carbon-inversion-mean: Cosine remainder for planar carbon inversion
Status: deferred

**Hypothesis:**A physical finite inversion potential can improve the residual GP mean while preserving current gradient and learned Hessian.
**Outcome and uncertainty:**336 fully converges with valid energy and zero errors but loses cost. Approximate carbon typing, the auxiliary linear angle chart and the potential shape remain unresolved; no passive failure diagnoses a repair.
**Reason to revisit:**Independent finite inversion-response evidence could identify a better justified potential or chart while reusing the exact remainder plumbing.
**Next experiment:**Await that evidence; no amplitude, atom-type extension or angle-threshold sweep. Compare any revision against the current champion under unchanged gates.

**Attempts:**
- Cycle336; candidate `a41d1d0da51c145fac61c25b4e80b0cee6650ab7`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/finite-carbon-inversion-mean/a41d1d0da51c145fac61c25b4e80b0cee6650ab7/algo.py), [training](evaluation_results/cycle-336-train.json), [narrative](full_log.md#cycle-336-finite-carbon-inversion-gp-mean).

## native-fragment-screw: Coupled native SE3 logarithm coordinates
Status: deferred

**Hypothesis:**A square screw-log chart can model finite fragment translation and rotation more efficiently than separate coordinates.
**Outcome and uncertainty:**337 fully converges with valid energy and no errors but loses training cost. Branch behavior, finite component bounds and path curvature remain unobserved; no passive failure diagnoses a repair.
**Reason to revisit:**A supported frame-invariant metric or diagnosed branch mechanism could make the exact six-coordinate derivative adapter useful.
**Next experiment:**Await that derivation/evidence; no scale, branch, radius or activation sweep. Preserve unchanged gates.

**Attempts:**
- Cycle337; candidate `4261910fdaeeb7d3bffbae9edd7a66548f4ed8d4`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/native-fragment-screw/4261910fdaeeb7d3bffbae9edd7a66548f4ed8d4/algo.py), [training](evaluation_results/cycle-337-train.json), [narrative](full_log.md#cycle-337-native-fragment-screw-geometry).

## projected-in-plane-prior: Four-atom bending directions in the physical prior
Status: deferred

**Hypothesis:**Separate projected in-plane bending from existing Fischer inversion stiffness at three-neighbor centers.
**Outcome and uncertainty:**339 fully converges with valid energy and zero errors, but loses training cost. Original scalar stiffness and raw-angle coupling/GP may not suit the new bending directions; no passive mechanism isolates these possibilities.
**Reason to revisit:**An independently supported stiffness decomposition or coherent fitted in-plane potential could make the exact four-atom gradient reusable.
**Next experiment:**Await that physical model; no atom-type, angular-threshold or weight sweep. Preserve the full gates.

**Attempts:**
- Cycle339; candidate `74633d85ed4cca75b58c85964b9a1fb339b801c7`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/projected-in-plane-prior/74633d85ed4cca75b58c85964b9a1fb339b801c7/algo.py), [training](evaluation_results/cycle-339-train.json), [narrative](full_log.md#cycle-339-projected-in-plane-bending-prior).

## self-scaled-broyden-update: Coupled positive-tangent inverse learning
Status: deferred

**Hypothesis:**Joint Broyden direction and determinant-aware scaling could balance learned positive curvature after physical calibration.
**Outcome and uncertainty:**341 has a broad cost loss,222 slower cases and two force-call limits despite valid aggregate energy and zero errors. Passive traces show force/energy oscillation without exceptions; they do not isolate transport, globalization or prior-scale loss.
**Reason to revisit:**A supported compatible globalization or secant-transport formulation could make the source's SPD update assumptions relevant here.
**Next experiment:**Await that derivation or specific passive evidence; no scaling, activation or damping sweep. Test any coherent revision against the current champion under unchanged full gates.

**Attempts:**
- Cycle341; candidate `390fda029645782824c410fe3736e6444fe39a32`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/self-scaled-broyden-update/390fda029645782824c410fe3736e6444fe39a32/algo.py), [training](evaluation_results/cycle-341-train.json), [manifest](diagnostics/bae20022219c4335a6984c80956a5580/manifest.json), [narrative](full_log.md#cycle-341-positive-tangent-self-scaled-broyden-learning).

## optimal-spectral-memory: Least-change scalar-plus-low-rank curvature
Status: deferred

**Hypothesis:**A physical-metric spectral approximation can forget stale cross-mode detail while retaining exceptional accumulated curvature.
**Outcome and uncertainty:**342 fully converges with valid energy and zero errors but substantially loses cost. Five-mode compression precedes each newest secant update; no passive model states distinguish information loss from moving-tangent effects.
**Reason to revisit:**An independently supported curvature relevance criterion or a formulation preserving physically important complementary response could improve this least-change representation.
**Next experiment:**Await that evidence; no rank, norm, frequency or activation sweep. Keep the unchanged full gates.

**Attempts:**
- Cycle342; candidate `aab761827d0bcf21c44aea7afeb413000cdedc47`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/optimal-spectral-memory/aab761827d0bcf21c44aea7afeb413000cdedc47/algo.py), [training](evaluation_results/cycle-342-train.json), [narrative](full_log.md#cycle-342-optimal-spectral-curvature-memory).

## paid-geodesic-wolfe: Derivative-tested paid geodesic line search
Status: deferred

**Hypothesis:**Wolfe acceptance with expansion and interpolation could improve native secant quality enough to repay additional force calls.
**Outcome and uncertainty:**343 fully converges without errors but fails energy and increases cost. Empty passive diagnostics cannot separate changed basins, trial overhead, or retained approximate curvature transport.
**Reason to revisit:**A separately supported direction/transport method requiring actual Wolfe conditions, or retained trial evidence showing a correctable mechanism, could make the line-search machinery useful.
**Next experiment:**Await that support; no acceptance constants, retry budget, activation or trust-memory sweep, and no automatic combination with341.

**Attempts:**
- Cycle343; candidate `0330c171dbbea5535d65af70e56c9ea4d18ffae6`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision invalid. [Implementation](ideas/paid-geodesic-wolfe/0330c171dbbea5535d65af70e56c9ea4d18ffae6/algo.py), [training](evaluation_results/cycle-343-train.json), [manifest](diagnostics/0f38ac8c7a9b4bfd9eac0b62abc006da/manifest.json), [narrative](full_log.md#cycle-343-paid-geodesic-wolfe-search).

**Cycle486 revisiting:** Full fixed-physical Cartesian Hager–Zhang directions coupled to paid strong-Wolfe search replaces model-only lengths and approximate native transport. This explicitly addresses the recorded prerequisite; unchanged gates apply. [Narrative](full_log.md#cycle-486-physical-conjugate-gradients-with-paid-wolfe-search).

**Cycle486 reassessment:** The full fixed-physical Hager–Zhang/Wolfe coupling fails energy with297 limits172 converged0 errors;every case slower. The missing line-search prerequisite alone is insufficient. Passive endpoint trace lacks direction/acceptance state. Revisit requires independently supported evolving-metric conjugacy or diagnosed trial waste, with no beta, budget, cap or floor sweeps.
- Cycle486; candidate `9deddbecd5b2d9e52b37bb71cf0f7160601966f1`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision invalid. [Implementation](ideas/preconditioned-conjugate-step/9deddbecd5b2d9e52b37bb71cf0f7160601966f1/algo.py), [training](evaluation_results/cycle-486-train.json), [narrative](full_log.md#cycle-486-physical-conjugate-gradients-with-paid-wolfe-search).

## atomic-tail-bond-mean: Ionization-derived finite bond stretch prior
Status: deferred

**Hypothesis:**Neutral-atom electronic decay scales could improve finite bond anharmonicity modeling beyond a stiffness extrapolation.
**Outcome and uncertainty:**344 fully converges without errors but fails energy and loses cost. No passive path/model state identifies a defect; neutral versus bonded density tails and local equilibrium centering are unresolved approximations.
**Reason to revisit:**Independently available environment-specific decay information or paid-data evidence of finite stretch response could validate this physical shape while reusing the exact Badger-chart remainder.
**Next experiment:**Await that support; no decay multiplier, amplitude, atom-subset or activation sweep. Compare any revision under unchanged full gates.

**Attempts:**
- Cycle344; candidate `a420ec452f01338c2d13a6a8827d85377b0cef08`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision invalid. [Implementation](ideas/atomic-tail-bond-mean/a420ec452f01338c2d13a6a8827d85377b0cef08/algo.py), [training](evaluation_results/cycle-344-train.json), [manifest](diagnostics/b4b213afe2164a90a04d6084ab559b46/manifest.json), [narrative](full_log.md#cycle-344-atomic-tail-bond-gp-mean).

## newest-protected-two-secant: Compatible old curvature with exact newest priority
Status: deferred

**Hypothesis:**Restore the adjustable part of one preceding secant in the newest step's physical-metric complement without overwriting the newest native action.
**Outcome and uncertainty:**346 fully converges with valid energy and no errors, but broadly loses cost (48 faster/203 slower/218 unchanged). No failure_details identify a defect; tangent projection and old response staleness remain unisolated.
**Reason to revisit:**An independently supported common-frame or finite-secant error model could make the exact priority-preserving algebra useful.
**Next experiment:**Await that evidence; no correction weight, memory, rank or activation sweeps.

**Attempts:**
- Cycle346; candidate `49b7b673655add99c3ce6326b1b2886f378d469d`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/newest-protected-two-secant/49b7b673655add99c3ce6326b1b2886f378d469d/algo.py), [training](evaluation_results/cycle-346-train.json), [narrative](full_log.md#cycle-346-newest-protected-two-secant-curvature).

## smooth-bond-plane-torsions: Native smooth plane invariants for proper torsions
Status: deferred

**Hypothesis:**Remove current plane-area divisions from torsional geometry while exactly matching initial raw-angle tangent and Cartesian spring curvature.
**Outcome and uncertainty:**347 converges all469 with valid energy and no errors but loses cost and increases runtime; faster/slower/unchanged=[84, 115, 270]. Finite radial geometry, approximate retained-state transport and generic-derivative overhead remain different from307.
**Reason to revisit:**An independently supported coordinate-covariant history/curvature model could use the analytic invariant pair; a local singularity diagnosis could also justify targeted use through a general method.
**Next experiment:**Await that evidence, without area threshold, topology subset, coordinate-weight, trust or transport sweeps.

**Attempts:**
- Cycle347; candidate `688ba2ad996d82d0bd67cf095b13456cfed967b3`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/smooth-bond-plane-torsions/688ba2ad996d82d0bd67cf095b13456cfed967b3/algo.py), [training](evaluation_results/cycle-347-train.json), [narrative](full_log.md#cycle-347-smooth-bond-plane-torsion-geometry).

## keating-bending-curvature: Dot-product bending with retained angular stiffness
Status: deferred

**Hypothesis:**Normalized Keating bond-vector factors supply geometric stretch/bend coupling while retaining the original pure angular stiffness.
**Outcome and uncertainty:**348 fully converges without errors but fails energy and loses cost. Empty failure_details do not isolate coupling mismatch or overlap with auxiliary/signed mixed features.
**Reason to revisit:**An independently supported molecular coupling decomposition could use the PSD factor and QR-completion-preserving mapping.
**Next experiment:**Await that model or localized passive evidence; no coupling strength, angle subset or diagonal compensation sweeps.

**Attempts:**
- Cycle348; candidate `acdf17d62b33eaaca92dfc44897c0abb7e92138c`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision invalid. [Implementation](ideas/keating-bending-curvature/acdf17d62b33eaaca92dfc44897c0abb7e92138c/algo.py), [training](evaluation_results/cycle-348-train.json), [narrative](full_log.md#cycle-348-keating-coupled-bending-curvature).

## native-rational-eigenmode: Rational modes normalized by native bounds
Status: deferred

**Hypothesis:**Separate rational saturation of low-curvature modes can preserve motion suppressed by a common shift.
**Outcome and uncertainty:**350 fully converges without errors and passes energy but loses cost. Empty failure_details provide no localized defect; over-damping versus direction changes remains unresolved.
**Reason to revisit:**An independently supported mode-reliability model could use this dimensionally consistent rational construction.
**Next experiment:**Await that evidence; no saturation, mode-subset, trust-history or activation sweep.

**Attempts:**
- Cycle350; candidate `d4529aea342bc5b9a7297d5299160b68726ec767`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/native-rational-eigenmode/d4529aea342bc5b9a7297d5299160b68726ec767/algo.py), [training](evaluation_results/cycle-350-train.json), [narrative](full_log.md#cycle-350-native-bound-rational-eigenmode-steps).

## contact-weighted-fragment-centers: Fixed linear centers near initial contacts
Status: deferred

**Hypothesis:**Initial contact-stiffness centers reduce translation/rotation coupling while retaining exactly linear translation coordinates.
**Outcome and uncertainty:**351 passes training but loses validation cost; both splits fully converge with valid energy and no errors. Isotropic-tether approximation, deformation coupling and frozen weights are unresolved. Empty failure_details supply no localized repair.
**Reason to revisit:**A separately supported anisotropic center-of-stiffness model or measured coupling information could improve pivot choice while reusing these exact linear coordinates.
**Next experiment:**Await that model; no exponent, weight floor, reweighting frequency or subgroup sweeps.

**Attempts:**
- Cycle351; candidate `ba3d660d12c873017cf6381bc8240db2af45936c`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision non_generalizable. [Implementation](ideas/contact-weighted-fragment-centers/ba3d660d12c873017cf6381bc8240db2af45936c/algo.py), [training](evaluation_results/cycle-351-train.json), [validation](evaluation_results/cycle-351-valid.json), [narrative](full_log.md#cycle-351-contact-weighted-fragment-translation-centers).

**354 anisotropic revisit:**The published symmetric-coupling stiffness center supplied351's model prerequisite, retaining its contact law. A minimum-change affine realization instead produces20 errors (15 ODE/5 SCF),one call limit and broad non-error cost loss (65 faster/103 slower/281 same). Saved SCF trial geometry shows a199.10Angstrom atom displacement; the one-frame ODE failure happens before a second force call. The affine weights/Jacobians were not retained, so near-planar amplification remains inferred. No bound/rank/clipping sweep or immediate repair is justified. A mathematically controlled body-fixed center representation avoiding affine reconstruction is the prerequisite for future work; it must still beat307 under full gates.
- Cycle354; candidate `434f16c119184b28c7514d3d2ce0db17d99717d2`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision invalid. [Implementation](ideas/contact-weighted-fragment-centers/434f16c119184b28c7514d3d2ce0db17d99717d2/algo.py), [training](evaluation_results/cycle-354-train.json), [manifest](diagnostics/62ddc1ee876445b78980e1f0cf51cadf/manifest.json), [narrative](full_log.md#cycle-354-anisotropic-contact-stiffness-centers).

**356 structural repair in progress:**Direct center solve and body-point action t+Exp(-phi)cbody avoid near-planar atom-affine reconstruction, satisfying354's mathematical prerequisite. First bounded repair, same contact model and full gates; no clipping/rank/bound sweep.

**356 reassessment:**Body-point action removes all354 errors and converges469, but loses cost with valid energy. End the bounded structural repair sequence; stable realization alone does not establish useful anisotropic coupling. No rank/center/contact/bound tuning. Revisit requires independent coupling evidence.
- Cycle356; candidate `424b4609d7c3c588a1106ba3d3a3d3b096a8f415`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/contact-weighted-fragment-centers/424b4609d7c3c588a1106ba3d3a3d3b096a8f415/algo.py), [training](evaluation_results/cycle-356-train.json), [narrative](full_log.md#cycle-356-body-fixed-anisotropic-contact-centers).

## invariant-collective-bounds: Euclidean bounds for fragment vector triples
Status: deferred

**Hypothesis:**Laboratory-axis invariant bounds improve collective trust control.
**Outcome and uncertainty:**352 fully converges with valid energy and no errors but loses cost. Conservative group balls and orientation effects remain confounded; empty failure_details gives no localized repair.
**Reason to revisit:**A separately supported collective model or invariant radius controller could use these consistent norm and derivative primitives.
**Next experiment:**Await that model; no radius, volume compensation or subset sweep.

**Attempts:**
- Cycle352; candidate `332f1d3805119f5e61e0480ddc5471da38ce52be`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/invariant-collective-bounds/332f1d3805119f5e61e0480ddc5471da38ce52be/algo.py), [training](evaluation_results/cycle-352-train.json), [narrative](full_log.md#cycle-352-orientation-invariant-collective-step-bounds).

## rigid-symmetry-curvature: Exact differential rigid-motion Hessian actions
Status: deferred

**Hypothesis:**Current-gradient symmetry identities correct global/relative curvature coupling without extra force calls.
**Outcome and uncertainty:**353 fully converges with valid energy and no errors but loses cost. Empty passive diagnostics cannot separate finite-secant conflict, signed reflection or null-space force noise.
**Reason to revisit:**An independently supported quotient model or compatible finite-secant construction could exploit these exact nonstationary identities.
**Next experiment:**Await that support; no strength, floor, timing or subset sweep.

**Attempts:**
- Cycle353; candidate `54e8085a706cdd337658783898d5af9cfaa2e1f1`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/rigid-symmetry-curvature/54e8085a706cdd337658783898d5af9cfaa2e1f1/algo.py), [training](evaluation_results/cycle-353-train.json), [manifest](diagnostics/923279b64ea04448bc3a1d9538e7165a/manifest.json), [narrative](full_log.md#cycle-353-exact-rigid-symmetry-curvature-actions).

## physical-metric-damping: Continuous primitive-stiffness regularization path
Status: deferred

**Hypothesis:**Hplus+alpha*M allocates restricted motion better than a uniform shift while preserving full Newton and native bounds.
**Outcome and uncertainty:**355 fully converges with valid energy and no errors but loses cost. Initial-metric staleness and component-path geometry remain unisolated; empty failure_details provides no repair.
**Reason to revisit:**Independent directional reliability or model-error evidence could justify a different damping metric using this generalized-eigenvalue implementation.
**Next experiment:**Await that information; no metric powers, interpolation, damping-scale or radius sweep.

**Attempts:**
- Cycle355; candidate `745248a9bcdb18d64259bf4acc7a02f7f1baa815`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/physical-metric-damping/745248a9bcdb18d64259bf4acc7a02f7f1baa815/algo.py), [training](evaluation_results/cycle-355-train.json), [narrative](full_log.md#cycle-355-physical-metric-continuous-step-damping).

## gradient-active-motion: Gradient-selected atomic tangent subspaces
Status: deferred

**Hypothesis:**Prioritizing high-force atomic motion while retaining coupled curvature reduces early weak-mode work.
**Outcome and uncertainty:**357 loses cost broadly (28 faster/370 slower), fails energy and reaches three call limits without optimizer errors. Temporary initial-velocity restriction differs from the source's exact restrained paths; active masks are not retained.
**Reason to revisit:**An independently justified coupling-aware active-set criterion could distinguish truly dispensable motion from necessary collective relaxation.
**Next experiment:**Await that criterion; no force fraction, release threshold or check-frequency tuning.

**Attempts:**
- Cycle357; candidate `f7b88a1f71184c9d3fdba48702f24035f12287ba`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision invalid. [Implementation](ideas/gradient-active-motion/f7b88a1f71184c9d3fdba48702f24035f12287ba/algo.py), [training](evaluation_results/cycle-357-train.json), [manifest](diagnostics/14fd30cde79d4c80a0d32f1be4c265b6/manifest.json), [narrative](full_log.md#cycle-357-gradient-selected-atomic-motion-subspaces).

**357 passive follow-up:**The representative45-atom bundle shows negligible final movement with max force0.00915Eh/Bohr, far above convergence. This is high-force stagnation; active masks/radius/model states are unavailable. Coupling-aware repair remains deferred given broad cost loss. [Trace](diagnostics/14fd30cde79d4c80a0d32f1be4c265b6/c3eb4f8da8e2fc3be0a2a883127bb473a89474c862f0b40bf371cfced11367e1.json).

## inverse-separation-chart: Locally matched collective vector inversion
Status: deferred

**Hypothesis:**Inverse fragment separation coordinates improve finite collective trajectories with initial Cartesian curvature unchanged.
**Outcome and uncertainty:**358 fully converges without errors and passes energy but loses cost. Local matching alone is insufficient; no passive defect localizes a repair.
**Reason to revisit:**A separately supported inverse-distance interaction model or measured trajectory discrepancy could motivate consistent model/chart coupling.
**Next experiment:**Await that model or evidence; no inverse power, tree, scaling or radius sweeps.

**Attempts:**
- Cycle358; candidate `a3fabe585e9edbc7c40ff90b9cdf8f427aede993`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/inverse-separation-chart/a3fabe585e9edbc7c40ff90b9cdf8f427aede993/algo.py), [training](evaluation_results/cycle-358-train.json), [narrative](full_log.md#cycle-358-locally-matched-inverse-separation-translation-chart).

## signed-pseudo-transient: Signed physical gradient-flow continuation
Status: deferred

**Hypothesis:**Implicit gradient flow with force-residual time-step feedback can replace energy-ratio globalization and avoid paid endpoint rejection.
**Outcome and uncertainty:**361 fully converges469 without errors but has major cost loss and invalid aggregate energy. Empty failure_details and pending manifest provide no controller/curvature trace. The signed physical-metric nonrejecting controller differs from274 and355, yet is unsuccessful.
**Reason to revisit:**Independent evidence that the learned Jacobian or moving metric is inconsistent with physical gradient flow could justify a different continuation formulation. A small parameter adjustment cannot address broad loss.
**Next experiment:**Require that evidence; no initial-time, cap, residual exponent or stability-factor sweep. Preserve full gates and genuine geometry relaxation.

**Attempts:**
- Cycle361; candidate `ef35f277f73f8d7ed6aa11896e04b54f716f724d`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision invalid. [Implementation](ideas/signed-pseudo-transient/ef35f277f73f8d7ed6aa11896e04b54f716f724d/algo.py), [train](evaluation_results/cycle-361-train.json), [narrative](full_log.md#cycle-361-signed-curvature-pseudo-transient-continuation).

## convex-energy-history: Energy-simplex history proposals
Status: deferred

**Hypothesis:**Nonnegative history weights chosen by a GEDIIS energy functional can give useful ordinary Newton proposals without signed-coefficient extrapolation.
**Outcome and uncertainty:**362 fully converges with valid energy and zero errors but loses cost; empty failure_details. Common-tangent developed histories are approximate, and GP can override the proposal, so the cause is unisolated.
**Reason to revisit:**Independent evidence for a consistent finite energy representation or measured proposal/model discrepancy could make the small simplex solver useful. Transported history alone has now failed both215 and362.
**Next experiment:**Await that evidence; no coefficient, activation, history-window or solver sweep.

**Attempts:**
- Cycle362; candidate `c048d0e53704444f82d972da9b775503c82385bc`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/convex-energy-history/c048d0e53704444f82d972da9b775503c82385bc/algo.py), [train](evaluation_results/cycle-362-train.json), [narrative](full_log.md#cycle-362-convex-energy-history-newton-proposals).

## directional-undershoot: Aligned-step covariance loosening
Status: deferred

**Hypothesis:**A source-defined aligned-step controller can lengthen directional GP correlations and avoid systematically short surrogate steps.
**Outcome and uncertainty:**363 converges all469, passes energy and has zero errors, but loses cost. The covariance-only adaptation and extra transported vector both differ from the orbital source; empty failure_details cannot isolate their effects.
**Reason to revisit:**Actual retained covariance/prediction evidence of systematic undershoot, or an independently supported trend model compatible with the source controller, could justify a new formulation.
**Next experiment:**Await that evidence; no angle, growth, factor bound or covariance parameter sweep.

**Attempts:**
- Cycle363; candidate `f938e767792fbea773fa44cdfbfb6585602befb9`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/directional-undershoot/f938e767792fbea773fa44cdfbfb6585602befb9/algo.py), [train](evaluation_results/cycle-363-train.json), [narrative](full_log.md#cycle-363-directional-gp-undershoot-mitigation).

## transported-nesterov: One-call restarted accelerated proposals
Status: deferred

**Hypothesis:**A virtual corrected point and transported extrapolation offset accelerate slow modes without extra force calls.
**Outcome and uncertainty:**365 fully converges with valid energy and zero errors but loses cost. No retained failure state separates momentum from curved-coordinate transport or final-proposal admission mismatch.
**Reason to revisit:**An independently supported acceleration/model coupling or a diagnosed transport discrepancy could justify a different recurrence.
**Next experiment:**Await that evidence; no momentum, restart, activation or cap sweeps.

**Attempts:**
- Cycle365; candidate `fd6f8e508e2349dee64a8f84153509a7b9ea8124`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/transported-nesterov/fd6f8e508e2349dee64a8f84153509a7b9ea8124/algo.py), [train](evaluation_results/cycle-365-train.json), [narrative](full_log.md#cycle-365-transported-restarted-nesterov-proposals).

## centroid-polar-chart: Distance and spherical direction for fragment centroids
Status: deferred

**Hypothesis:**A locally identity radial/spherical chart can separate collective approach from sideways motion without changing initial curvature.
**Outcome and uncertainty:**366 fully converges without errors but fails aggregate energy and loses cost. Empty failure_details supplies no localized chart/domain failure.
**Reason to revisit:**Independent finite-path and interaction-model evidence could motivate a compatible radial/angular scalar model. Exact chain derivatives and square coordinate carrier are reusable.
**Next experiment:**Await that evidence; no angular scale, reference, tree or radius sweeps.

**Attempts:**
- Cycle366; candidate `ba9a6bfeafe642a580ccdc2eb3cacedb13c3de3c`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision invalid. [Implementation](ideas/centroid-polar-chart/ba9a6bfeafe642a580ccdc2eb3cacedb13c3de3c/algo.py), [train](evaluation_results/cycle-366-train.json), [narrative](full_log.md#cycle-366-radial-and-spherical-fragment-separation-coordinates).

## chemical-cubic-residual: Shared local anharmonic coefficient inference
Status: deferred

**Hypothesis:**Same-element-pair bonds share an uncertain normalized cubic response, complementing the coupled307 GP.
**Outcome and uncertainty:**367 fully converges with valid energy and zero errors but loses cost. Empty failure_details cannot separate local-environment mismatch, Badger chart normalization or cubic extrapolation.
**Reason to revisit:**Independent data or theory supporting a shared finite local descriptor could make correlated chemical learning useful. The exact finite-feature covariance and uncertainty integration are reusable.
**Next experiment:**Await that descriptor/evidence; no chemical subsets, degree, normalization or amplitude sweeps.

**Attempts:**
- Cycle367; candidate `896a37221bc9a0b44806da0f5ef081e8d71c5833`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/chemical-cubic-residual/896a37221bc9a0b44806da0f5ef081e8d71c5833/algo.py), [train](evaluation_results/cycle-367-train.json), [narrative](full_log.md#cycle-367-chemically-shared-cubic-residual-covariance).

**394 finite-process revisit:**Replace the cubic-only shared assumption by an independent finite-distance SE process per element pair, adding only its second-order Taylor remainder to307. The source-derived sum covariance and exact inverse-Badger descriptor supply the declared descriptor prerequisite. No strength/degree/chemical-subset sweep.

**394 reassessment:**Finite-distance sharedSE functions with quadratic jets removed fully converge and passenergy but losecost; [11, 15, 443] faster/slower/unchanged. Runtime also increases. No diagnostic source identifies a defect. Require independently demonstrated local environment dependence or finite-process prediction error before revisiting; no amplitude,length,quadrature,feature orchemical-subset sweep.
- Cycle394; candidate `d4c0dd3551c046d6fbca1ed165c2409be649f3c6`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/chemical-cubic-residual/d4c0dd3551c046d6fbca1ed165c2409be649f3c6/algo.py), [train](evaluation_results/cycle-394-train.json), [narrative](full_log.md#cycle-394-shared-finite-distance-bond-residual-process).

## calibrated-secant-blending: Physical-prior projection of learned secants
Status: deferred

**Hypothesis:**Blend measured curvature toward the six-pair calibrated physical response when directional responses agree, reducing spurious coupling.
**Outcome and uncertainty:**368 fails energy/cost broadly with55 force-call limits. One passive bundle shows low-force motion exceeding displacement thresholds; no retained curvature/blend states identify causality. Equation12 projection need not decay near a minimum, unlike the full source method.
**Reason to revisit:**A derived molecular analogue of the full source asymptotically vanishing blend, supported by independent response-error evidence, could retain early robustness without suppressing final curvature learning.
**Next experiment:**Await that error model; no beta, proxy, timing or update-family sweep after broad loss. Any revisit must beat307 under full gates.

**Attempts:**
- Cycle368; candidate `5e8740d2c24240552965674c0850297b9cc87e17`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision invalid. [Implementation](ideas/calibrated-secant-blending/5e8740d2c24240552965674c0850297b9cc87e17/algo.py), [train](evaluation_results/cycle-368-train.json), [narrative](full_log.md#cycle-368-calibrated-prior-secant-projection-blending).

## paid-rosenbrock-relaxation: Two-stage Cartesian gradient-flow optimization
Status: deferred

**Hypothesis:**Actual intermediate gradients and published energy-model timestep control improve nonlinear implicit relaxation.
**Outcome and uncertainty:**369 fails energy/cost with61 limits,467 slower cases and zero errors. One retained trace continues high-force relaxation. Stage, damping and matrix states are absent; fixed-chart curvature loss and paid overhead remain unseparated.
**Reason to revisit:**An independently supported moving preconditioner with a consistent two-stage geometry, or passive evidence of a specific stage defect, might make this integration principle useful. The source controller alone is insufficient.
**Next experiment:**Await that formulation/evidence; no lambda, cap, stage fraction or Hessian-update sweep after broad loss.

**Attempts:**
- Cycle369; candidate `a562c41f5472f40b1f1fdaa7571503c55ca7ccfd`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision invalid. [Implementation](ideas/paid-rosenbrock-relaxation/a562c41f5472f40b1f1fdaa7571503c55ca7ccfd/algo.py), [train](evaluation_results/cycle-369-train.json), [narrative](full_log.md#cycle-369-paid-two-stage-rosenbrock-relaxation).

## fragment-swing-twist: Separate axial spin and tilt in the native rotation chart
Status: deferred

**Hypothesis:**A square shape-axis swing–twist map with identity initial Jacobian improves finite fragment relaxation paths.
**Outcome and uncertainty:**370 fully converges with valid energy and zero errors but loses cost. No retained traces establish axis relevance or path/model mismatch; absence of branch exceptions does not establish broad chart regularity.
**Reason to revisit:**Independent interaction-anisotropy evidence or a compatible finite rotational scalar model could give the decomposition a useful role.
**Next experiment:**Await that evidence/model; no axis, scaling, bounds, group or singularity-threshold sweep.

**Attempts:**
- Cycle370; candidate `b926c3e3934a40f0f2d502e4f7e5c2b3e2fc7f2b`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/fragment-swing-twist/b926c3e3934a40f0f2d502e4f7e5c2b3e2fc7f2b/algo.py), [train](evaluation_results/cycle-370-train.json), [narrative](full_log.md#cycle-370-native-fragment-swing-twist-coordinates).

## entropic-gp-search: Penalize energy uncertainty during proposal search
Status: deferred

**Hypothesis:**Minimizing Gaussian entropic energy mu+variance/2 can find more reliable proposals before the existing admission gate.
**Outcome and uncertainty:**372 lowers total cost but fails energy, all469 converged with zero errors. One inferior endpoint supplies more cost gain than the total. Empty passive manifest gives no path/model diagnosis.
**Reason to revisit:**Independent calibration of posterior energy uncertainty or evidence linking admission failures to microsearch could justify risk-sensitive selection. The analytic variance-gradient wrapper remains reusable.
**Next experiment:**Await such evidence; no risk-scale, anchor, confidence or solver-budget sweep.

**Attempts:**
- Cycle372; candidate `c54bf81286afea20d39ffa6b6019430e3fbc8e11`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision invalid. [Implementation](ideas/entropic-gp-search/c54bf81286afea20d39ffa6b6019430e3fbc8e11/algo.py), [train](evaluation_results/cycle-372-train.json), [manifest](diagnostics/b4fca33858754e5d812799300b62715c/manifest.json), [narrative](full_log.md#cycle-372-entropic-risk-gp-proposal-search).

## relative-fragment-pose-tree: Body-relative pose tree with matched initial tangent
Status: deferred

**Hypothesis:**Relative orientation and body-frame displacement can represent interfragment coupled motion better than independent absolute poses.
**Outcome and uncertainty:**373 fully converges with valid energy and zero errors, but substantially loses cost; [38, 135, 296] faster/slower/unchanged. Empty passive failure evidence cannot separate finite metric, bounds and approximate history effects.
**Reason to revisit:**A derived interaction metric or compatible finite curvature/history representation could exploit the exact relative-pose chart and full derivatives.
**Next experiment:**Require that independent construction; no tree/root/weight/bounds/activation sweep.

**Attempts:**
- Cycle373; candidate `2a5c41907a962f3c67cc7ea972d98961fb3d18ad`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/relative-fragment-pose-tree/2a5c41907a962f3c67cc7ea972d98961fb3d18ad/algo.py), [train](evaluation_results/cycle-373-train.json), [narrative](full_log.md#cycle-373-relative-fragment-pose-tree-coordinates).

## accelerated-spectral-residuals: Fixed physical chart with accelerated DF-SANE
Status: deferred

**Hypothesis:**A physical Cartesian preconditioner and short secant residual extrapolation can remove the cost of repeatedly building full learned curvature.
**Outcome and uncertainty:**374 is invalid:218 converged,247 limits,four xTB electronic convergence errors. Saved trajectories show high-force oscillation and a late paid excursion from low force. They do not expose proposal acceptance or isolate extrapolation from the spectral direction. No Python/accounting error is established.
**Reason to revisit:**A derived energy-compatible globalization or consistent moving metric could address stationary-residual merit and frozen-geometry weaknesses while reusing the independent rewrite. Current evidence does not support parameter tuning.
**Next experiment:**Only after deriving that component, compare the coherent solver against the current champion; require full energy and cost gates. No eta/memory/cap/controller sweeps from these results.

**Attempts:**
- Cycle374; candidate `74ec5d466d2f988f142149bab70638aba4495f9d`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision invalid.
  [Implementation](ideas/accelerated-spectral-residuals/74ec5d466d2f988f142149bab70638aba4495f9d/algo.py), [train](evaluation_results/cycle-374-train.json), [narrative](full_log.md#cycle-374-physically-preconditioned-accelerated-spectral-residuals).

## finite-outer-distance-mean: Joint finite bond-angle coupling in the GP mean
Status: deferred

**Hypothesis:**The exact cosine-law outer distance, with inverse Badger bonds, supplies useful higher-order coupling from the existing across-angle auxiliary springs.
**Outcome and uncertainty:**376 converges all469 with zero errors and valid energy but loses cost. Empty failure_details leave surrogate prediction and admission effects unobserved. Quadratic curvature is preserved exactly; this does not prove the remaining finite shape is accurate.
**Reason to revisit:**Independent evidence for joint finite bond-angle response could support a physical shape beyond the local harmonic outer spring. The exact chart inversion and analytic derivatives are reusable.
**Next experiment:**Await that evidence; no spring-strength, path-count, support, domain or GP-setting sweeps. Compare a derived response under full gates against the then-current champion.

**Attempts:**
- Cycle376; candidate `15de561a8a983c462d58dafcd6451195e3c84e77`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/finite-outer-distance-mean/15de561a8a983c462d58dafcd6451195e3c84e77/algo.py), [train](evaluation_results/cycle-376-train.json), [narrative](full_log.md#cycle-376-finite-outer-distance-coupling-in-the-gp-mean).

## local-density-gp-coordinates: First-order matched bonded density moments in GP covariance
Status: deferred

**Hypothesis:**Species-resolved radial and vector-density Gram moments capture finite coupled environment variation in a native scalar GP.
**Outcome and uncertainty:**379 converges all469 with valid energy and no errors, but loses cost; faster/slower/unchanged=[35, 26, 408]. Empty failure_details cannot distinguish descriptor incompleteness, finite normalization and off-manifold history effects.
**Reason to revisit:**Independent evidence for a useful finite local-environment response or a compatible full geometry chart could give the stable native descriptor and analytic derivatives a role.
**Next experiment:**Await that evidence; no radial-scale, moment-order, rank, species, support or amplitude sweep after this loss.

**Attempts:**
- Cycle379; candidate `2e32c28a558ba8eccd248cfb897d75a95cd0ab8d`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/local-density-gp-coordinates/2e32c28a558ba8eccd248cfb897d75a95cd0ab8d/algo.py), [train](evaluation_results/cycle-379-train.json), [narrative](full_log.md#cycle-379-locally-matched-density-moment-gp-coordinates).

**Cycle577 reassessment:** Same bondedmoments/radialscale/species/complete-angle eligibility nowuse exact finiteCartesianchart withanalyticqueryJacobian, supplying379 geometry prerequisite. All469converged,validenergy,zeroerrors,butcostloss;{'faster': 39, 'slower': 224, 'same': 206}. Emptydetails do notisolate incomplete descriptor or historical/chart mismatch. Furtherrevisitrequires independently measureddescriptor/modelresponse or history-chart consistency, no scale/order/support/rank sweep.
- Cycle577; candidate `437a855fead07cba6a54343877bfb7c89a6b0e57`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/local-density-gp-coordinates/437a855fead07cba6a54343877bfb7c89a6b0e57/algo.py), [training](evaluation_results/cycle-577-train.json), [narrative](full_log.md#cycle-577-finite-cartesian-density-covariance-coordinates).

## physical-locking-transport: Metric-isometric Hessian transport locked to path velocity
Status: deferred

**Hypothesis:**Use one physical primal/dual isometry and normalized endpoint work for late curvature memory.
**Outcome and uncertainty:**380 converges all469 with valid energy and no errors but loses cost; faster/slower/unchanged=[22, 24, 423]. Empty failure_details leave transport quality, empirical metric and globalization unobserved.
**Reason to revisit:**Independent covariant-model evidence or a supported compatibility mechanism with globalization could make the exact reflection/locking construction useful.
**Next experiment:**Await that evidence; no metric, activation, overlap or normalization sweeps.

**Attempts:**
- Cycle380; candidate `9bb50d22f08ab73a5ab3e96be19841a5991f105c`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/physical-locking-transport/9bb50d22f08ab73a5ab3e96be19841a5991f105c/algo.py), [train](evaluation_results/cycle-380-train.json), [narrative](full_log.md#cycle-380-physical-metric-locking-transport-for-curvature-learning).


## adjacent-torsion-calibration: Phase-aware neighboring-bond torsion curvature
Status: deferred

**Hypothesis:**One bounded signed phase-difference harmonic block captures coupled rotations about distinct adjacent bonds from existing paid secants.
**Outcome and uncertainty:**382 converges all469 with valid energy and zero errors but loses cost; faster/slower/unchanged=[3, 9, 457]. Emptyfailure_details does not identify fit or phase mismatch.
**Reason to revisit:**Independent evidence for local coupled torsion response or a diagnosed fitting limitation could make the bounded graph construction useful.
**Next experiment:**Await such evidence; no phase,harmonic,amplitude,degree normalization or ridge sweep. Retain unchanged full gates.
**Attempts:**
- Cycle382; candidate `5f0837760ba6166868231b7cd57d429181cf7c84`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard.
  [Implementation](ideas/adjacent-torsion-calibration/5f0837760ba6166868231b7cd57d429181cf7c84/algo.py), [train](evaluation_results/cycle-382-train.json), [narrative](full_log.md#cycle-382-adjacent-torsion-phase-coupling-in-physical-calibration).


## elementary-interfragment-pose: Distance,bends and dihedrals for relative fragment motion
Status: deferred

**Hypothesis:**An initially matched elementary dimer chart can represent finite intermolecular motion more effectively than absolute centroid/rotation vectors.
**Outcome and uncertainty:**383 converges469 without errors,but fails energy and loses cost; faster/slower/unchanged=[48,85,336]. Emptyfailure_details and passive manifest provide no branch/domain diagnosis.
**Reason to revisit:**Independent evidence linking relative-coordinate geometry to finite model/transport accuracy could justify a consistent learning or globalization formulation. Exact wrapper derivatives remain reusable.
**Next experiment:**Require such evidence; no tree,root,reference-axis,scale or branch-threshold sweep. Preserve fixed full gates.
**Attempts:**
- Cycle383; candidate `afa3e7d9c835111e5e36fcc668d18d0a17530ad4`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision invalid.
  [Implementation](ideas/elementary-interfragment-pose/afa3e7d9c835111e5e36fcc668d18d0a17530ad4/algo.py), [train](evaluation_results/cycle-383-train.json), [narrative](full_log.md#cycle-383-elementary-interfragment-pose-coordinates), [passive manifest](diagnostics/d0187ac6cda74a67a9f9b7e055bd7beb/manifest.json).


## orthogonal-secant-calibration: Structured total-least-squares physical fitting
Status: deferred

**Hypothesis:**Orthogonal distance to the physical secant graph avoids privileging gradient-response error when learning early model coefficients.
**Outcome and uncertainty:**384 converges all469,passes energy and has zero errors,but loses cost; faster/slower/unchanged=[128, 155, 186]. Emptyfailure_details cannot separate discrepancy-model error,fit quality and approximate history.
**Reason to revisit:**Independent evidence for two-sided secant discrepancy or a measured coefficient-estimation defect could justify a calibrated graph-distance model. The exact profiled objective and gradient are reusable.
**Next experiment:**Await such evidence; no metric,discrepancy ratio,ridge,fit-count or solver-budget sweep. Preserve full gates.
**Attempts:**
- Cycle384; candidate `04fa577e1de0e59aa54aa7e34fc22b280ce333b1`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard.
  [Implementation](ideas/orthogonal-secant-calibration/04fa577e1de0e59aa54aa7e34fc22b280ce333b1/algo.py), [train](evaluation_results/cycle-384-train.json), [narrative](full_log.md#cycle-384-two-sided-orthogonal-physical-secant-calibration).


## projective-torsion-coordinates: Scalar half-angle native torsional chart
Status: deferred

**Hypothesis:** An initially matched scalar projective torsion chart supports finite response without unit-circle redundancy.
**Outcome and uncertainty:**385 has426 identical pre-force initialization errors;43 converge. No meaningful performance assessment yet.
**Reason to revisit:** Passive evidence identifies a redundant metadata read before its creation. Real-atom indexing already excludes all dummy linear-bend torsions.
**Next experiment:** Remove only that lookup as bounded repair1; unchanged full remote gates.
**Attempts:**
- Cycle385; candidate `c1ef82d146a23b84db29d76152396cc85aa0a419`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision invalid. [Implementation](ideas/projective-torsion-coordinates/c1ef82d146a23b84db29d76152396cc85aa0a419/algo.py), [train](evaluation_results/cycle-385-train.json), [narrative](full_log.md#cycle-385-scalar-projective-torsion-coordinates), [passive bundle](diagnostics/c4a38df42b9f4c60b23715438d6d6efa/d98d0118ffdc805ce2876fcd88205439b590850e436f143015613e4c3153d838.json).

**Repair1 reassessment:**386 removes initialization errors but loses cost broadly, with468converged andone200-call limit. Excluding that case still loses cost. Passive late motion is small with excessive residual forces, without chart-state evidence of a pole. End sequence; independent finite-chart/learning compatibility evidence is required, with no recentering/scale/phase/activation sweeps.
- Cycle386; candidate `e69cd7e20ae614994af3d7de1ec8b4c6fe710a87`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/projective-torsion-coordinates/e69cd7e20ae614994af3d7de1ec8b4c6fe710a87/algo.py), [train](evaluation_results/cycle-386-train.json), [narrative](full_log.md#cycle-386-projective-torsion-initialization-repair), [passive bundle](diagnostics/90b1e9fe003f441eb2c824555e8d252e/6f0258c5f4b17841e24fb439b8467ca517c02f73e0b258f941a8d182230a9048.json).


## eckart-fragment-frames: Consistent mass-weighted collective coordinates
Status: deferred

**Hypothesis:** Center-of-mass translation and mass-weighted Eckart orientation reduce coupling of fragment poses with internal deformations.
**Outcome and uncertainty:**387 fully converges with valid energy and zero errors but loses cost; faster/slower/unchanged=[79, 68, 322]. Emptyfailure_details cannot isolate potential-model or geometry-metric effects.
**Reason to revisit:** An independently supported coupling between this physical frame and a compatible learning metric could make the exact reference-coefficient formulation useful.
**Next experiment:** Await that derivation/evidence; no mass exponent,atom subset,reference-refresh or strength sweep.
**Attempts:**
- Cycle387; candidate `fa3ca2aa0cbe8890fe16c60fc6226983a37c6740`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/eckart-fragment-frames/fa3ca2aa0cbe8890fe16c60fc6226983a37c6740/algo.py), [train](evaluation_results/cycle-387-train.json), [narrative](full_log.md#cycle-387-mass-weighted-eckart-fragment-frames).


## downshifted-bundle-proposals: Piecewise-affine residual tangents under GP admission
Status: deferred

**Hypothesis:** Conservative envelopes of paid residual tangents generate useful finite proposals without smooth posterior minimization.
**Outcome and uncertainty:**388 fully converges with valid energy and no errors,but loses cost; faster/slower/unchanged=[52, 214, 203]. Emptyfailure_details leaves cut activity,GP rejection and chart inversion unobserved.
**Reason to revisit:** Independent evidence linking cut predictions to finite molecular response, or a compatible proposal/admission model, could justify a full bundle formulation. The exact small-simplex dual solver is reusable.
**Next experiment:** Require that evidence; no cut-strength,history,map-tolerance or fallback-policy sweep.
**Attempts:**
- Cycle388; candidate `22e8d62d7927e91d3955d9df96c5122581e5eb8e`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/downshifted-bundle-proposals/22e8d62d7927e91d3955d9df96c5122581e5eb8e/algo.py), [train](evaluation_results/cycle-388-train.json), [narrative](full_log.md#cycle-388-downshifted-bundle-proposals-from-paid-history).


## paid-nonlinear-conjugate-residual: Measured Hessian actions in a physical Cartesian solver
Status: deferred

**Hypothesis:** Recurring paid directional curvature and response-orthogonal memory improve force relaxation enough to cover their cost.
**Outcome and uncertainty:**390 loses cost broadly and fails energy, with151limits andzeroerrors. A retained trace has ongoing energy descent and alternating probe/optimization moves; no localized exception. Fixed geometry, nonlinear history mismatch and probe overhead remain unisolated.
**Reason to revisit:** An independently established response-transport or nonlinear preconditioner could make measured actions useful with fewer optimization iterations; accurate differential curvature alone did not suffice.
**Next experiment:** Await that mechanism/evidence; no memory,probe step,metric floor,cap or backtracking sweep.
**Attempts:**
- Cycle390; candidate `964c41caf18e05466c4f52549f6859d80c9702a9`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision invalid. [Implementation](ideas/paid-nonlinear-conjugate-residual/964c41caf18e05466c4f52549f6859d80c9702a9/algo.py), [train](evaluation_results/cycle-390-train.json), [narrative](full_log.md#cycle-390-physical-cartesian-nonlinear-conjugate-residual-optimizer), [passive bundle](diagnostics/549468f0760a4807aab159d90fdc43ba/01ce5d846e148cb7fd823507c54ecb96ffac3027c08b6952a07753fab9341122.json).


## relative-condition-curvature: Complement-preserving optimal spectral secant correction
Status: deferred

**Hypothesis:** A rank-two secant correction minimizing relative condition number can retain useful accumulated native curvature without global rescaling.
**Outcome and uncertainty:**391 fully converges with valid energy andzeroerrors but loses cost; [69,131,269] faster/slower/unchanged. No passive model evidence separates curvature accuracy from moving-tangent effects.
**Reason to revisit:** Independently demonstrated relative spectral distortion or a compatible tangent transport could make the exact small-block optimizer useful.
**Next experiment:** Await that evidence; no positivity,blend,timing orcondition-threshold sweep.
**Attempts:**
- Cycle391; candidate `0ff50c7a721b45aff0390d201977f86886ea842d`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/relative-condition-curvature/0ff50c7a721b45aff0390d201977f86886ea842d/algo.py), [train](evaluation_results/cycle-391-train.json), [narrative](full_log.md#cycle-391-relative-condition-rank-two-curvature-learning).


## projected-native-flow: First-order projected geometry flow with developed secants
Status: deferred

**Hypothesis:** Projected native velocity fields with compatible gradient transport and integrated endpoint development can improve finite molecular realization.
**Outcome and uncertainty:**392 fully converges with valid energy andzeroerrors but loses cost slightly; faster/slower/unchanged=[23, 30, 416]. No passive path evidence identifies attenuation or prediction mismatch.
**Reason to revisit:** Independently established finite-path model errors or compatible finite-coordinate predictions could make the developed-displacement machinery useful.
**Next experiment:** Require that evidence; no flow-time, projection-strength, tolerance or eligibility sweep.
**Attempts:**
- Cycle392; candidate `7adc2d16d7df0309646e64f9bacbc2d2721234e6`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/projected-native-flow/7adc2d16d7df0309646e64f9bacbc2d2721234e6/algo.py), [train](evaluation_results/cycle-392-train.json), [narrative](full_log.md#cycle-392-projected-native-flow-with-developed-secants).


## atomic-force-filter: Nondominated atomic force progress for native trust steps
Status: deferred

**Hypothesis:** A nonmonotone componentwise force filter can accept physically useful model-inaccurate steps while rejecting unproductive paid trials.
**Outcome and uncertainty:**393 fully converges with valid energy andzeroerrors but increases cost; faster/slower/unchanged=[93, 206, 170]. Rejection overhead,ceiling andradius effects remain unisolated.
**Reason to revisit:** Recorded proposal/controller evidence or a model explicitly designed for filter progress could establish which route saves evaluations. State restoration and invariant force-vector filtering remain reusable.
**Next experiment:** Require that evidence; no margin,threshold,radius orfilter-size sweeps.
**Attempts:**
- Cycle393; candidate `d223978addd34fc1d7bab6157ac4b60c2d818019`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/atomic-force-filter/d223978addd34fc1d7bab6157ac4b60c2d818019/algo.py), [train](evaluation_results/cycle-393-train.json), [narrative](full_log.md#cycle-393-atomic-force-filter-trust-region-controller).


## structured-native-elastic: Known nonlinear elastic curvature with Cartesian residual learning
Status: deferred

**Hypothesis:** Exact changing native-potential curvature lets Cartesian secants learn only unknown response.
**Outcome and uncertainty:**396passesenergy but loses cost broadly,451slower,8limits,zeroerrors. A passive trace retains continuing late motion; no localized implementation defect. Fixed topology,dummy lift,residual updates andglobalization remain unisolated.
**Reason to revisit:** Independent evidence for a compatible residual model or nonlinear preconditioner could use this exact value/gradient/Hessian component.
**Next experiment:** Await that evidence; no strength,reset,floor,radius or update-family sweep.
**Attempts:**
- Cycle396; candidate `2cae9732d3274c5c427835de3f44e46330ce30c7`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/structured-native-elastic/2cae9732d3274c5c427835de3f44e46330ce30c7/algo.py), [train](evaluation_results/cycle-396-train.json), [narrative](full_log.md#cycle-396-structured-cartesian-learning-over-a-native-elastic-potential), [passive manifest](diagnostics/547dbd192e454521b53ce98e06f5ffaf/manifest.json).


## nonlinear-elastic-preconditioning: Broyden acceleration of a force-corrected coarse relaxation
Status: deferred

**Hypothesis:** Cheapnonlinearelasticrelaxationdefinesabetterconditionedfixedpointresidualthanphysicalforcesforoutersecantlearning.
**Outcome and uncertainty:**398invalidwith2innernodescenterrors,186limitsand281converged. Noneofthe467returnedcasesbeats307;passiveerrortracehashighforce,andlimittracehasunfinishedsmallmotion. Innerstateisnotcaptured.
**Reason to revisit:** Independentlyvalidatedcoarsenonlinearresponse or a compatible stablefixedpointmapcouldmakeleftpreconditionedlearninguseful;thepresentUandinnermapdonotshowusefulsignal.
**Next experiment:** Awaitthatmodel/evidence;noforcingtolerance,innerbudget,cap,Jacobianfloorormeritcontrollertuning.
**Attempts:**
- Cycle398;candidate `37bebade59ef265cd978ae9d884f698b3338e1d2`;champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`;decision invalid. [Implementation](ideas/nonlinear-elastic-preconditioning/37bebade59ef265cd978ae9d884f698b3338e1d2/algo.py), [train](evaluation_results/cycle-398-train.json), [narrative](full_log.md#cycle-398-nonlinear-elastic-preconditioning-with-broyden-acceleration), [manifest](diagnostics/dcefefafdb0442aba6a6901a00ddb42a/manifest.json).


## contact-distance-pose: Interfragment inverse-distance collective coordinates
Status: deferred

**Hypothesis:** Contact distances give a more useful finite pose chart while initial left-inverse normalization preserves the champion tangent and prior.
**Outcome and uncertainty:**399 has45 unpaid geometry failures; returned424 show74 faster/59 slower/291 unchanged. This subset is not a score. Bundles identify finite-path breakdown, but do not expose chart singular values.
**Reason to revisit:** The localized failure and useful subset signal support bounded chart-exit handling without changing physical evaluations or gates.
**Next experiment:** Repair1: restore the paid geometry after a recognized cheap realization failure, rebuild champion coordinates and retry in the same iteration; disable the contact chart for the remaining trajectory. Stop if repaired results show broad loss.
**Attempts:**
- Cycle399; candidate `bb4d612e3c6e00383d7f79e343ba272f8a8c8f79`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision invalid. [Implementation](ideas/contact-distance-pose/bb4d612e3c6e00383d7f79e343ba272f8a8c8f79/algo.py), [train](evaluation_results/cycle-399-train.json), [narrative](full_log.md#cycle-399-interfragment-inverse-distance-pose-chart), [manifest](diagnostics/144f1672568c4ff99e3d809a405bcc1b/manifest.json).

**400 reassessment:** Bounded repair1 removes all45 exceptions;469 converge, zeroerrors. Cost improves .00254489 but energy0.9999063145922878 fails. Empty failure_details do not diagnose higher endpoints. Await independent finite-chart/model evidence before another redesign; no failure-threshold, refresh, radius or scaling sweep.
- Cycle400; candidate `d13e43a4dc5680162f9f2d5f3de0d7397a52315a`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision invalid. [Implementation](ideas/contact-distance-pose/d13e43a4dc5680162f9f2d5f3de0d7397a52315a/algo.py), [train](evaluation_results/cycle-400-train.json), [narrative](full_log.md#cycle-400-exit-failed-contact-charts-at-the-last-paid-geometry).


## decrement-shifted-quasi-newton: Coupled SCORE secants and paid line search
Status: deferred

**Hypothesis:** One learned-metric decrement can coordinate finite step damping and positive secant regularization, improving stable curvature learning.
**Outcome and uncertainty:**402passesenergy but costs2.4519;26limits,zeroerrors,463slower. Passive trace shows unfinished motion; shift, fixed geometry and search overhead remain unisolated.
**Reason to revisit:** Independently supported dimensional curvature/regularization model or trajectory-level secant/search evidence could make the coupled solver useful.
**Next experiment:** Await that evidence; no shift,normalization,cap,Wolfe orupdate-family sweeps.
**Attempts:**
- Cycle402; candidate `a11692dfc4d26d1e69c7b884502e4ae14a3a7c9e`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/decrement-shifted-quasi-newton/a11692dfc4d26d1e69c7b884502e4ae14a3a7c9e/algo.py), [train](evaluation_results/cycle-402-train.json), [narrative](full_log.md#cycle-402-dimensionless-score-with-paid-wolfe-globalization), [manifest](diagnostics/826ca9ba50bb451389e2926d491b8579/manifest.json).

## shepard-residual-proposals: Local Taylor blending under GP admission
Status: deferred

**Hypothesis:** Inverse-distance blending of paid residual jets supplies useful finite proposals beyond the SE posterior minimum.
**Outcome and uncertainty:**403 is energy-valid and fully converged but increases cost; successful-path model states are unavailable. GP rejection and interpolation shape cannot be separated from endpoints.
**Reason to revisit:** A independently supported common-coordinate gradient representation or proposal/admission model could address the mismatch; source local-jet interpolation remains reusable.
**Next experiment:** Only after such evidence, compare a consistently represented local interpolant against307; do not tune weights or admission on this loss.

**Attempts:**
- Cycle403; candidate `eae143c9146f6dc7fb034ec232481eafa82ee43b`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard.
  [Implementation](ideas/shepard-residual-proposals/eae143c9146f6dc7fb034ec232481eafa82ee43b/algo.py). [Training evidence](evaluation_results/cycle-403-train.json). [Narrative](full_log.md#cycle-403-modified-shepard-local-jet-proposals-under-gp-admission).

## direct-transverse-bends: Dummy-free near-linear bend components
Status: incorporated

**Hypothesis:** Two smooth physical bend components avoid dummy constraint coupling while retaining isotropic angular stiffness.
**Outcome and uncertainty:**404 has8unpaid ODE-budget errors,6at the first step;461returned cases lose cost overall despite11improvements. Inner frame/rank states are unavailable.
**Reason to revisit:** A rotation-compatible gauge with consistent path and curvature transport could address finite rotational sensitivity; source component/chain-torsion machinery is reusable.
**Next experiment:** Derive such a chart before testing again; compare against307 with identical prior and thresholds. No automatic ODE-budget or gauge sweeps.

**Attempts:**
- Cycle404; candidate `8dd409236b822cf91d979bbd8c395013006ef2e7`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision invalid.
  [Implementation](ideas/direct-transverse-bends/8dd409236b822cf91d979bbd8c395013006ef2e7/algo.py). [Training evidence](evaluation_results/cycle-404-train.json). [Narrative](full_log.md#cycle-404-direct-transverse-bends-without-dummy-atoms).

**430 prerequisite/repair1:** Re-read404 failure_details,manifest and two saved bundles. Replace its laboratory-fixed gauge with an exact four-atom molecular frame: dot/cross scalar invariance supplies the deferred compatible chart, with all reference derivatives included. Same-fragment off-axis reference eligibility is structural; otherwise retain native dummies. No claimed passive proof of a gauge pole. Keep404 stiffness/threshold/chain torsions and307 gates.

**430 repair1 outcome:** Keep after both gates pass, all934train/valid molecules converge without errors. Molecular-reference chart resolves the observed404 errors and improves cost across both splits; initial reference eligibility and frame coupling remain part of the accepted specification. No immediate reference or threshold tuning.
- Cycle430; candidate/newchampion `bb770d9c7c9e59d39007f71a106af77cd9eb73a1`; precedingchampion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision keep. Implementation is committed in algo.py. [Train](evaluation_results/cycle-430-train.json). [Validation](evaluation_results/cycle-430-valid.json). [Narrative](full_log.md#cycle-430-molecular-reference-transverse-bends).

**439 extension pending:**430 remains the accepted champion. Test its unchanged molecular-reference chart for each eligible near-pi pair at centers with additional neighbors, retaining native handling of unresolved pairs and degree-two-only chain torsions. No coordinate threshold, reference ranking or stiffness tuning. This extension changes redundant physical stiffness too; full gates determine utility. See [narrative](full_log.md#cycle-439-transverse-bends-at-multi-neighbor-centers).

**439 extension outcome:**All469converged,validenergy,zeroerrors,butcostloss;{'same': 465, 'faster': 1, 'slower': 3}. No passive failure states diagnose redundant stiffness versus finite-chart effects. Keep430 incorporated and defer this extension without reference,threshold,stiffness,subset orODEbudget sweeps. Revisit only with an independently supported redundant-curvature decomposition.
- Cycle439; candidate `5e0fdb7ff2a407baedaaedf1ea4b94e4741c326b`; champion `bb770d9c7c9e59d39007f71a106af77cd9eb73a1`; decision discard. [Implementation](ideas/direct-transverse-bends/5e0fdb7ff2a407baedaaedf1ea4b94e4741c326b/algo.py), [train](evaluation_results/cycle-439-train.json), [narrative](full_log.md#cycle-439-transverse-bends-at-multi-neighbor-centers).

## quotient-cartesian-learning: Parallel Cartesian shape frames
Status: deferred

**Hypothesis:** Removing global rigid motions with a consistent parallel frame improves physical Cartesian curvature learning.
**Outcome and uncertainty:**405 encounters3singular-inertia errors and7calllimits, with426/466returned cases slower. Complete transport does not make this Cartesian model competitive; inner frame drift and curvature states are unavailable.
**Reason to revisit:** A useful Cartesian model that remains valid across linear shape strata would be needed; the derived angular-momentum transport may be reusable independently.
**Next experiment:** Require that independent model improvement before revisiting; no tolerance, threshold, radius or fit sweep from this broad loss.

**Attempts:**
- Cycle405; candidate `1bf7a369fa3c2b8cb3a186161e5d65dbdfa8ae13`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision invalid.
  [Implementation](ideas/quotient-cartesian-learning/1bf7a369fa3c2b8cb3a186161e5d65dbdfa8ae13/algo.py). [Training evidence](evaluation_results/cycle-405-train.json). [Narrative](full_log.md#cycle-405-parallel-frame-cartesian-quotient-quasi-newton).

## chordal-rotation-mean: Matched finite orientation energy prior
Status: deferred

**Hypothesis:** A chordal orientation potential supplies useful rotational higher orders without changing current gradient/curvature.
**Outcome and uncertainty:**406 is fully converged, energy-valid and error-free but loses cost; no diagnostic failure identifies the source of mismatch.
**Reason to revisit:** Independently established rotational energy-shape evidence could justify this local prior; exact composed-quaternion derivative and quadratic compensation are reusable.
**Next experiment:** Await that evidence, then compare a physically identified finite law against307. No stiffness, potential shape or kernel sweep.

**Attempts:**
- Cycle406; candidate `87a5c6875ac623cd80934b4914590d547e0159ac`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard.
  [Implementation](ideas/chordal-rotation-mean/87a5c6875ac623cd80934b4914590d547e0159ac/algo.py). [Training evidence](evaluation_results/cycle-406-train.json). [Narrative](full_log.md#cycle-406-matched-chordal-rotation-energy-in-the-gp-mean).

## atomic-budget-quadratic: Quadratic redistribution under atomic norm bounds
Status: deferred

**Hypothesis:** Joint native-box and per-atom norm constraints spend the ordinary step's existing largest atomic-motion budget more effectively.
**Outcome and uncertainty:**407 fully converges with valid energy and no errors but loses cost. Tangent feasibility does not establish actual geodesic endpoint quality.
**Reason to revisit:** An independently useful finite-displacement model could exploit the convex constraint machinery without relying only on the current quadratic.
**Next experiment:** Await that model, then compare its feasible proposal against307 under unchanged gates; no budget multiplier or inner-iteration sweep.

**Attempts:**
- Cycle407; candidate `8fad9c05ced1e83a1badeb5c82bb7e79fdc5f309`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard.
  [Implementation](ideas/atomic-budget-quadratic/8fad9c05ced1e83a1badeb5c82bb7e79fdc5f309/algo.py). [Training evidence](evaluation_results/cycle-407-train.json). [Narrative](full_log.md#cycle-407-quadratic-redistribution-within-the-ordinary-atomic-budget).

## collision-constrained-gp: Reoptimize collision-blocked surrogate proposals
Status: deferred

**Hypothesis:** A nonlinear GP chooses better feasible motion than shortening or Euclidean-projecting a collision-blocked direction.
**Outcome and uncertainty:**408 reproduces champion aggregate metrics, fully converges and has no errors; successful-path activation is unobserved. No gain meets the gate.
**Reason to revisit:** Independently observed GP blockage could establish whether a nonlinear realization model would make these constraints useful.
**Next experiment:** Await such evidence; no activation, contact-set, margin or search-budget sweep.

**Attempts:**
- Cycle408; candidate `ffe26c633d13e6c451b41b6aab897933f7429cf2`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard.
  [Implementation](ideas/collision-constrained-gp/ffe26c633d13e6c451b41b6aab897933f7429cf2/algo.py). [Training evidence](evaluation_results/cycle-408-train.json). [Narrative](full_log.md#cycle-408-collision-constrained-gp-proposal-refinement).

## bayesian-kernel-averaging: Marginalize native versus bending covariance
Status: deferred

**Hypothesis:** Posterior model averaging and between-model uncertainty can handle uncertain residual geometry more effectively than one selected covariance.
**Outcome and uncertainty:**409 fully converges with valid energy and no errors, but loses cost. Shared chart/mean error and probability calibration remain unresolved.
**Reason to revisit:** Independently validated complementary residual models could make the exact two-model mixture machinery useful.
**Next experiment:** Await complementary model evidence; no odds, temperature, length or noise tuning.

**Attempts:**
- Cycle409; candidate `9a38a3fa5788ec53cb3253fcc1ebdd6161c67510`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard.
  [Implementation](ideas/bayesian-kernel-averaging/9a38a3fa5788ec53cb3253fcc1ebdd6161c67510/algo.py). [Training evidence](evaluation_results/cycle-409-train.json). [Narrative](full_log.md#cycle-409-bayesian-averaging-of-native-and-bending-gp-models).

## physically-scaled-conjugate-gradients: Paid directional curvature with SCG globalization
Status: deferred

**Hypothesis:** Actual directional curvature and conjugacy can avoid full-Hessian model error and repay an extra paid probe per accepted move.
**Outcome and uncertainty:**410 loses energy and broad cost,125calllimits,zeroerrors;466cases slower. Two complete scalar traces show nonzero late forces with probe/trial alternation, but curvature and damping states are absent.
**Reason to revisit:** Independently supported low-cost curvature information or a useful changing physical preconditioner would be required; this implementation supplies correct combined-call accounting.
**Next experiment:** Await that evidence; no probe-size, damping, cap, restart or floor sweep.

**Attempts:**
- Cycle410; candidate `80f6056f463398109b3129dd8d73b691d1325250`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision invalid.
  [Implementation](ideas/physically-scaled-conjugate-gradients/80f6056f463398109b3129dd8d73b691d1325250/algo.py). [Training evidence](evaluation_results/cycle-410-train.json). [Narrative](full_log.md#cycle-410-physically-normalized-scaled-conjugate-gradients).

**Revisit585:** Cheap analyticGPgradients nowreplace410paidcurvature probes; allphysicaldynamics remain463. TestsourceSCG onlyas12-attemptmodelsearch withunchangedcritics. This resolvespaidcurvaturecost structurally, notbychanging410probe size orphysicalpreconditioner.

**585 reassessment:** Model-onlySCGsuppliescheapcurvaturebutlosescostwithall469converged,validenergy0errors;{'slower': 10, 'faster': 4, 'same': 455}. No localmodeltrajectorydiagnosis. Awaitdirectcurvature/conjugacyfailureevidence; no probe,damping,restart,budget sweep.
- Cycle585; candidate `28004705f3b9cb66b35c6b9b6127a2e2e8962af0`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decisiondiscard. [Implementation](ideas/physically-scaled-conjugate-gradients/28004705f3b9cb66b35c6b9b6127a2e2e8962af0/algo.py), [training](evaluation_results/cycle-585-train.json), [narrative](full_log.md#cycle-585-scaled-conjugate-gradient-surrogate-search).

## finite-stretch-bend-covariance: Mixed bond-vector geometry for GP covariance
Status: deferred

**Hypothesis:** A compensated Keating invariant adds finite stretch/bend correlations while preserving307's current tangent and pure-bending covariance.
**Outcome and uncertainty:**411 fully converges,passes energy,zeroerrors;cost gain0.0000485230 is below1e-4. No successful model traces diagnose a defect or repair.
**Reason to revisit:** Independent evidence of stretch/bend residual correlation or a compatible newly accepted physical model could support a controlled combination.
**Next experiment:** Await that evidence; do not tune coupling strength, eligible bonds/angles or kernel settings to pursue the threshold.

**Attempts:**
- Cycle411; candidate `2438424af2b3831f520c8272ef5da8c20ed747da`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard.
  [Implementation](ideas/finite-stretch-bend-covariance/2438424af2b3831f520c8272ef5da8c20ed747da/algo.py). [Training evidence](evaluation_results/cycle-411-train.json). [Narrative](full_log.md#cycle-411-finite-stretch-bend-coupling-in-gp-covariance).

**Revisit533:**450/463 accepted electronic bond correlations after411, supplying its declared compatible-physical-model prerequisite. Test the exact preserved411 feature on463; no feature or kernel tuning. The local GP normalization now includes electronic stretch correlations, which may complement the invariant's finite mixed geometry. A further failure exhausts this interaction control.

**533 outcome:** Train passes (16fast9slow444same) but validation loses cost ({'slower': 18, 'same': 428, 'faster': 19}), bothfullyconverged,validenergy,zeroerrors. This exhausts the accepted450/463 electronic-model interaction prerequisite. Future work needs independently observed finite residual coupling or a diagnosed chart/derivative defect, not another generic accepted-model combination or feature/kernel/domain sweep.
- Cycle533; candidate `3e13dceb7839ba5ebcdbe51520b47e1bea88611d`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision non_generalizable. [Implementation](ideas/finite-stretch-bend-covariance/3e13dceb7839ba5ebcdbe51520b47e1bea88611d/algo.py), [training](evaluation_results/cycle-533-train.json), [validation](evaluation_results/cycle-533-valid.json), [narrative](full_log.md#cycle-533-mixed-covariance-with-electronic-bond-correlations).

## gfn2-repulsion-mean: Published GFN2 repulsion as a higher-order GP mean
Status: deferred

**Hypothesis:** A known target-method component explains finite residual energy beyond the learned quadratic.
**Outcome and uncertainty:**412 fully converges with valid energy and no errors but loses cost; component cancellation and affine-chart error are unresolved.
**Reason to revisit:** Independent evidence that a compatible electronic residual or faithful finite chart offsets those errors would justify using the exact component.
**Next experiment:** Await that evidence; no amplitude, pair-set, cutoff or chart-radius tuning. Structured derivative learning at actual points is a different hypothesis.

**Attempts:**
- Cycle412; candidate `1a4aef45f1e983b8fe38814236375b9df50490ba`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard.
  [Implementation](ideas/gfn2-repulsion-mean/1a4aef45f1e983b8fe38814236375b9df50490ba/algo.py). [Training evidence](evaluation_results/cycle-412-train.json). [Narrative](full_log.md#cycle-412-published-gfn2-repulsion-remainder-in-the-gp-mean).

## structured-gfn2-repulsion: Learn the unknown curvature around published repulsion
Status: deferred

**Hypothesis:** Use actual known target-method derivatives in a structured secant, rather than relearning their geometry variation.
**Outcome and uncertainty:**413 fully converges with valid energy and no errors but loses cost. Repulsion/electronic cancellation and approximate native operator identification remain unresolved.
**Reason to revisit:** Independent evidence of a consistent residual-transport or complementary electronic-curvature model could justify the analytic component.
**Next experiment:** Await that evidence; no amplitude, pair-support, activation-count or transport-tolerance sweep.

**Attempts:**
- Cycle413; candidate `7495e1d5b55eca38418353d0d49c44a702b9fc11`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard.
  [Implementation](ideas/structured-gfn2-repulsion/7495e1d5b55eca38418353d0d49c44a702b9fc11/algo.py). [Training evidence](evaluation_results/cycle-413-train.json). [Narrative](full_log.md#cycle-413-structured-native-learning-with-published-gfn2-repulsion).

## error-oriented-broyden: Fixed physical chart with NLEQ-ERR globalization
Status: deferred

**Hypothesis:**Natural Newton-correction contraction plus exact common-frame Broyden secants improves force relaxation.
**Outcome and uncertainty:**414 fully converges with no errors but fails aggregate energy and loses cost. No successful-path model/fallback states isolate the cause.
**Reason to revisit:**A demonstrated energy-faithful Jacobian/root globalization would be needed; exact-Newton theory does not establish this for learned molecular gradients.
**Next experiment:**Await that evidence; no damping, cap, floor, acceptance or fallback-threshold sweep.

**Attempts:**
- Cycle414;candidate `a9f4ffda83d1fdf2c2ce305d904ea98a6c3056f2`;champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`;decision invalid.
  [Implementation](ideas/error-oriented-broyden/a9f4ffda83d1fdf2c2ce305d904ea98a6c3056f2/algo.py). [Training evidence](evaluation_results/cycle-414-train.json). [Narrative](full_log.md#cycle-414-error-oriented-broyden-optimization-in-a-fixed-physical-chart).

## energy-dominating-minimax-force: Lower worst predicted force within ordinary energy and motion budgets
Status: deferred

**Hypothesis:**An energy-dominating convex proposal can improve the worst atomic force without sacrificing modeled descent or motion limits.
**Outcome and uncertainty:**415 fully converges with valid energy and no errors but loses cost. Model force quality and finite geodesic agreement remain unobserved.
**Reason to revisit:**Independent evidence of accurate finite Cartesian force prediction could make this constrained Pareto construction useful.
**Next experiment:**Await that evidence; no norm mixtures, budgets, constraint slack, activation or inner-iteration tuning.

**Attempts:**
- Cycle415;candidate `39e12471768d849b8b3ff5a006bcbf6a12c396b3`;champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`;decision discard.
  [Implementation](ideas/energy-dominating-minimax-force/39e12471768d849b8b3ff5a006bcbf6a12c396b3/algo.py). [Training evidence](evaluation_results/cycle-415-train.json). [Narrative](full_log.md#cycle-415-energy-dominating-minimax-atomic-force-proposals).

## minimum-operator-secant: Minimum spectral-norm physical secant correction
Status: deferred

**Hypothesis:** Enforce each secant with the smallest worst-direction curvature correction in fixed physical units.
**Outcome and uncertainty:** Cycle416 converges all cases without errors and satisfies energy, but loses cost strongly. The indefinite transverse correction may damage useful curvature; no retained diagnostic establishes that cause.
**Reason to revisit:** A theoretical selection among minimum-norm mappings that also controls curvature inertia could resolve a structural weakness. Evidence is needed before another variant.
**Next experiment:** Derive an inertia-preserving constrained mapping, if feasible for indefinite secants, before comparing a complete update against the current champion; no blend/norm/threshold sweep.

**Attempts:**
- Cycle416; candidate `061c840d69f204a700536afe3791e677735839b9`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard.
  [Implementation](ideas/minimum-operator-secant/061c840d69f204a700536afe3791e677735839b9/algo.py).
  [Training evidence](evaluation_results/cycle-416-train.json).
  [Narrative](full_log.md#cycle-416-minimum-operator-norm-physical-secant-learning).

## individual-primitive-fit: Independent local primitive stiffness calibration
Status: deferred

**Hypothesis:** Independent native-coordinate stiffnesses capture local curvature differences hidden by shared type scales.
**Outcome and uncertainty:**417 converges all cases,passesenergyandcorrectness,butlosescost. Six normalized secants may not identify the larger model, and independent priors reduce pooling; no retained fit states isolate either mechanism.
**Reason to revisit:** Independent evidence for heterogeneous response together with a principled identifiable low-dimensional local model could justify selective flexibility.39/45 and417 discourage arbitrary partition or ridge changes.
**Next experiment:** Derive such a response model and its identifiability condition before evaluation against the current champion; no coefficient-bound,regularization,history or grouping sweep.

**Attempts:**
- Cycle417; candidate `979ef0b079fe804b0c0f817579b51e2444064032`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard.
  [Implementation](ideas/individual-primitive-fit/979ef0b079fe804b0c0f817579b51e2444064032/algo.py).
  [Training evidence](evaluation_results/cycle-417-train.json).
  [Narrative](full_log.md#cycle-417-individually-learned-primitive-stiffnesses).

## newton-continuation-root: Residual-controlled explicit Newton continuation
Status: deferred

**Hypothesis:** Persistent pseudo-time and measured residual reduction globalize a common-chart Newton root solve with reusable secant Jacobians.
**Outcome and uncertainty:**418 converges all469 without errors but fails energy and loses cost. Saved endpoints cannot separate nonminimum roots, basin choice, approximate-Jacobian reuse and fixed-chart inefficiency.
**Reason to revisit:** A supported energy-descent root Jacobian or nonlinear coordinate realization could address the root/energy mismatch structurally. No present evidence justifies controller tuning.
**Next experiment:** Derive the energy-descent condition and a consistent moving-chart continuation before testing against the current champion; retain full gates and paid-rejection accounting.

**Attempts:**
- Cycle418; candidate `2360600a3951fcbcca563775bdc343559cab509a`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision invalid.
  [Implementation](ideas/newton-continuation-root/2360600a3951fcbcca563775bdc343559cab509a/algo.py).
  [Training evidence](evaluation_results/cycle-418-train.json).
  [Narrative](full_log.md#cycle-418-physical-newton-continuation-with-residual-ratio-control).

## cartesian-completed-geometry: Full-rank hybrid embedding with transferred curvature
Status: deferred

**Hypothesis:** Cartesian completion regularizes native paths while exact tangent transfer preserves the initial physical quadratic form.
**Outcome and uncertainty:**419 is energy-valid and error-free but loses cost broadly:14faster/447slower/8same;onecalllimit. Passive final rows show ongoing energy/displacement variation, without internal metric or Hessian states.
**Reason to revisit:** A formulation separating global rigid motion and the physical update metric from embedding regularization could address structural coupling. This is a theoretical prerequisite, not a diagnosis from endpoints.
**Next experiment:** Derive that separation with a common path/transport model before testing; no Cartesian-weight,metric,subset,trust orrank-threshold sweep.

**Attempts:**
- Cycle419; candidate `bb9a74e3d994258af996ad5d47662946fec60c80`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard.
  [Implementation](ideas/cartesian-completed-geometry/bb9a74e3d994258af996ad5d47662946fec60c80/algo.py). [Training evidence](evaluation_results/cycle-419-train.json).
  [Manifest](diagnostics/a7bceaf982574e4c86a91593f3281e72/manifest.json).
  [Narrative](full_log.md#cycle-419-cartesian-completed-native-geometry-with-transferred-physical-curvature).

## variational-block-student-gp: Joint historical-block reliability inference
Status: deferred

**Hypothesis:** A Student-t likelihood with one inferred precision per historical value/gradient block can suppress chart-inconsistent observations while preserving the current anchor.
**Outcome and uncertainty:** Cycle420 loses training cost, with all469converged and no errors. Approximate variational uncertainty may be miscalibrated; no passive inference state distinguishes this from an unnecessary robust likelihood. No validation.
**Reason to revisit:** Independent evidence that geometric transport error has a heavy-tailed block structure, together with calibrated posterior uncertainty, could make robust historical inference useful.
**Next experiment:** First derive a common-frame transport-error model and justified predictive uncertainty. No degree-of-freedom,noise,iteration,history,block-subset orconfidence sweep from this result.

**Attempts:**
- Cycle420; candidate `395f158d02a72e7e9f024a34bf9c0f1ea7674979`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/variational-block-student-gp/395f158d02a72e7e9f024a34bf9c0f1ea7674979/algo.py). [Train](evaluation_results/cycle-420-train.json). [Narrative](full_log.md#cycle-420-variational-student-t-likelihood-for-historical-gp-blocks).

## directional-trust-dilation: Apply scalar agreement only along the paid direction
Status: deferred

**Hypothesis:** Rank-one dilation of the native trust box avoids extrapolating one paid model-agreement ratio to unobserved directions.
**Outcome and uncertainty:**421 loses cost with complete convergence,valid energy,zeroerrors. No passive trust-shape state identifies a localized cause.
**Reason to revisit:** Independent evidence about directional model reliability, plus a uniformly controlled shape transport, could make directional feedback useful. The exact relative-map construction preserves perpendicular bounds.
**Next experiment:** Await that model; no dilation blend,condition limits,controller constants,coordinate subset orreset-frequency sweep. Compare any distinct formulation against307 under full gates.

**Attempts:**
- Cycle421; candidate `9398b2d9a3b546b09c998def1baac446de01e654`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/directional-trust-dilation/9398b2d9a3b546b09c998def1baac446de01e654/algo.py). [Train](evaluation_results/cycle-421-train.json). [Narrative](full_log.md#cycle-421-directional-dilation-of-the-native-trust-box).

## d2-dispersion-mean: Published damped attraction beyond the GP quadratic
Status: deferred

**574 reassessment:** Published D3coordinationresponse andfiniteCartesianquery supplythe oldprerequisite butstill losecost withall469converged. See [coordination-dispersion-mean](#coordination-dispersion-mean-environment-dependent-pair-dispersion-gp-remainder). Futureuse needs independentlycalibrated chargeresponse ormean-error diagnostics; no D2/D3parametersweep.

**Hypothesis:** A fixed physical dispersion remainder captures finite nonbonded response without changing the paid current value,gradient or learned quadratic.
**Outcome and uncertainty:**422 loses cost;all469converge,validenergy,zeroerrors. D2-versus-D4 mismatch and affine query geometry remain unisolated.
**Reason to revisit:** Independently available charge/coordination-dependent dispersion response with consistent finite query geometry could provide a better component prior; the analytic central-force remainder is reusable.
**Next experiment:** Await that independently justified model,without D2 amplitude,damping,radii,C6,pair-support orkernel sweeps. Test against307 with unchanged gates.

**Attempts:**
- Cycle422; candidate `a7a59b0477979aaeeebe7d8061ad0f454095f849`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/d2-dispersion-mean/a7a59b0477979aaeeebe7d8061ad0f454095f849/algo.py). [Train](evaluation_results/cycle-422-train.json). [Narrative](full_log.md#cycle-422-published-d2-dispersion-remainder-in-the-gp-mean).

## geometric-mean-secant: Self-dual positive BFGS–DFP mean
Status: deferred

**Hypothesis:** The matrix geometric mean balances Hessian and inverse-Hessian least-change views, preserving the shared secant and complementary curvature.
**Outcome and uncertainty:**423 loses cost and fails aggregate energy,with all469converged andzeroerrors. Positive-domain eligibility,learned curvature and native transport remain unisolated; no passive state diagnoses a repair.
**Reason to revisit:** A independently supported compatible transport/globalization framework could make the positive-domain self-dual update useful. The exact two-dimensional reduction needs no dense matrix square root.
**Next experiment:** Await that framework or localized evidence; no blend,positivity,timing,scaling or eligibility sweep. Full gates unchanged.

**Attempts:**
- Cycle423; candidate `6315780d7176367e1c5f9ba5c4dc712bfafe6d69`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision invalid. [Implementation](ideas/geometric-mean-secant/6315780d7176367e1c5f9ba5c4dc712bfafe6d69/algo.py). [Train](evaluation_results/cycle-423-train.json). [Narrative](full_log.md#cycle-423-self-dual-geometric-mean-secant-learning).

## erf-network-covariance: Moment-matched infinite-width network prior
Status: deferred

**Hypothesis:** Saturating nonstationary covariance may model finite molecular response better than radial stationarity while matching current prior moments.
**Outcome and uncertainty:**424 loses cost with complete convergence,valid energy andzeroerrors. Recentered nonstationarity and derivative-chart error remain unisolated.
**Reason to revisit:** An independently justified persistent feature chart or calibrated saturation model could make this analytic network prior useful.
**Next experiment:** Await that model; no bias,amplitude,input-scale,noise,history orconfidence sweeps.

**Attempts:**
- Cycle424; candidate `e88e1c4de01ffad3bb319721969ec88dab6be350`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/erf-network-covariance/e88e1c4de01ffad3bb319721969ec88dab6be350/algo.py). [Train](evaluation_results/cycle-424-train.json). [Narrative](full_log.md#cycle-424-moment-matched-neural-network-gp-covariance).

## skew-gradient-nuisance: Conservative energy with marginalized chart curl
Status: deferred

**Hypothesis:** Correlated antisymmetric linear mapping error can be separated from a scalar GP without optimizing a nonconservative field.
**Outcome and uncertainty:**425 loses cost slightly;all469converged,validenergy,zeroerrors. No retained inference state identifies curl prevalence or a localized defect.
**Reason to revisit:** Independent chart-error observations or a derived higher-order consistent transport likelihood could establish identifiable nuisance structure. Low-rank evidence inference is reusable.
**Next experiment:** Await that evidence; no variance,rank,history,noise,confidence or inference-budget sweeps. Compare to current champion under full gates.

**Attempts:**
- Cycle425; candidate `64c1d50bfa2926d326cb1114032ea44ef4a0b0e0`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/skew-gradient-nuisance/64c1d50bfa2926d326cb1114032ea44ef4a0b0e0/algo.py). [Train](evaluation_results/cycle-425-train.json). [Narrative](full_log.md#cycle-425-marginalized-skew-mapping-error-in-historical-gradients).

## neutral-h4-mean: Finite directional hydrogen-bond GP mean
Status: deferred

**Hypothesis:** A published radial/angular/proton-transfer shape captures nonlinear hydrogen-bond response after preserving the current quadratic.
**Outcome and uncertainty:**426 loses cost with all469converged,validenergy andzeroerrors. PM6-to-GFN2 transfer,neutral-core approximation and affine chart are unresolved.
**Reason to revisit:** Independently validated environment-dependent hydrogen-bond response with a compatible finite query chart could make the analytic distance-jet remainder useful.
**Next experiment:** Await that model/evidence; no strength,radial,angular,support,charge-factor,noise orconfidence sweep.

**Attempts:**
- Cycle426; candidate `2ca1756aa1ac281a0dea072fdeac1e378632cf95`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/neutral-h4-mean/2ca1756aa1ac281a0dea072fdeac1e378632cf95/algo.py). [Train](evaluation_results/cycle-426-train.json). [Narrative](full_log.md#cycle-426-neutral-h4-hydrogen-bond-remainder-in-the-gp-mean).

## compact-wendland-gp: Finite residual correlation support
Status: deferred

**Hypothesis:** A smooth positive-definite compact kernel prevents inappropriate remote residual influence while matching current value/gradient variances.
**Outcome and uncertainty:**427 loses cost with complete convergence,validenergy,zeroerrors. Compact support and near-field shape changes remain inseparable.
**Reason to revisit:** Independent residual-correlation evidence with a justified locality scale could support finite-support modeling. The division-free Hermite covariance is reusable.
**Next experiment:** Await that evidence; no support,order,noise,history orconfidence sweeps.

**Attempts:**
- Cycle427; candidate `9c105141642032dd8dfd3e5f5935c8462794641e`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/compact-wendland-gp/9c105141642032dd8dfd3e5f5935c8462794641e/algo.py). [Train](evaluation_results/cycle-427-train.json). [Narrative](full_log.md#cycle-427-moment-matched-compact-support-gp-covariance).

## paid-path-hermite-tensor: Native directional quartic from paid path data
Status: deferred

**Hypothesis:** A coercive rank-one Hermite tensor adds anharmonic response using actual paid endpoint path slopes while preserving current gradient/Hessian.
**Outcome and uncertainty:**428 loses cost with all469converged,validenergy,zeroerrors; no passive failure localizes a repair. Quartic extrapolation and learned-Hessian error remain unisolated.
**Reason to revisit:** Independently calibrated higher-order model uncertainty or a justified continuation validity domain could make directional interpolation useful.
**Next experiment:** Await that model/evidence; no quartic scaling,coercivity,activation,root orbound sweeps.

**Attempts:**
- Cycle428; candidate `b917bb50cf3d669bb4eeeabbd3dcbf26ab099326`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/paid-path-hermite-tensor/b917bb50cf3d669bb4eeeabbd3dcbf26ab099326/algo.py). [Train](evaluation_results/cycle-428-train.json). [Narrative](full_log.md#cycle-428-paid-path-directional-hermite-tensor-steps).

## heat-continuation-gp: Analytic residual diffusion during microsearch
Status: deferred

**Hypothesis:** A sequence of exact Gaussian convolutions guides residual search from smooth to detailed response while retaining the native final model.
**Outcome and uncertainty:**429 loses cost slightly with all469converged,validenergy,zeroerrors. No passive evidence distinguishes branch selection from stage-budget/memory effects.
**Reason to revisit:** Independently observed model-basin problems or a reliable continuation solver could use the closed-form value/gradient heat operator.
**Next experiment:** Await that evidence; no smoothing-scale,stage-count,budget,tolerance orseed sweeps.

**Attempts:**
- Cycle429; candidate `d85cab89d0cbbc27c339581fe0c8571048e4bc90`; champion `a049250808c5f46c1a6fb12844e8b1e900c1a7da`; decision discard. [Implementation](ideas/heat-continuation-gp/d85cab89d0cbbc27c339581fe0c8571048e4bc90/algo.py). [Train](evaluation_results/cycle-429-train.json). [Narrative](full_log.md#cycle-429-heat-continuation-search-of-the-learned-gp-residual).

## published-bending-denominator: Faithful Manz damping placement
Status: deferred

**Hypothesis:** Applying the published tanh ratio only to the reference-sine denominator term improves finite angular shape while preserving local stiffness and the matched mean/covariance relation.
**Outcome and uncertainty:**432 fully converges without errors and passes energy, but its0.00005581train cost gain is below the fixed gate. No validation or passive failure evidence. Accepted298/307-derived formula remains the empirical champion component despite its source discrepancy.
**Reason to revisit:** Preserve a faithful primary-source control for an independently justified physical angular extension; source fidelity alone does not justify acceptance.
**Next experiment:** Await such a separate modeling connection; no tanh, mixture, activation or exponent sweep and no automatic rebaseline.

**Attempts:**
- Cycle432; candidate `c3273f28c386c62bdc720124bf67496d268093a5`; champion `bb770d9c7c9e59d39007f71a106af77cd9eb73a1`; decision discard.
  [Implementation](ideas/published-bending-denominator/c3273f28c386c62bdc720124bf67496d268093a5/algo.py), [train](evaluation_results/cycle-432-train.json), [narrative](full_log.md#cycle-432-published-bending-denominator-in-mean-and-covariance).

## finite-transverse-bending-mean: Exact near-linear radial shape in the GP mean
Status: deferred

**Hypothesis:**430's real two-component bends support the published linear-equilibrium angular potential via an exact radial inversion at frozen current bond-length ratio; subtracting its full current second-order expansion preserves native force/curvature.
**Outcome and uncertainty:**433 fully converges without errors and passes energy but loses cost. No passive failure or inner model states isolate coupling omission or finite radial shape as the cause.
**Reason to revisit:** A separately supported coupled stretch–linear-bend model could reuse the geometric inversion and full vector-jet subtraction; current scalar radial theory alone is insufficient.
**Next experiment:** Await that model/evidence; no stiffness, branch, mixture or eligibility sweep.

**Attempts:**
- Cycle433; candidate `257375db47d1dc07d205a6b0227db315539d3d31`; champion `bb770d9c7c9e59d39007f71a106af77cd9eb73a1`; decision discard.
  [Implementation](ideas/finite-transverse-bending-mean/257375db47d1dc07d205a6b0227db315539d3d31/algo.py), [train](evaluation_results/cycle-433-train.json), [narrative](full_log.md#cycle-433-finite-radial-transverse-bending-mean).

## normal-slope-nuisance-gp: Marginalized historical normal response
Status: deferred

**Hypothesis:** A scalar extension f(p)+n.h(p) models energetic response to historical components omitted by the current tangent search, while querying only the unchanged f process.
**Outcome and uncertainty:**434 fully converges with valid energy and zeroerrors but loses cost. No inner covariance or residual data identifies a localized repair. This differs from261/294's full ambient metric and425's antisymmetric gradient field.
**Reason to revisit:** Independent evidence calibrating normal energetic response or distinguishing it from tangent variation could make the exact positive-semidefinite latent covariance useful.
**Next experiment:** Await that evidence; no normal strength,length,rank,jitter,history oractivation sweeps.

**Attempts:**
- Cycle434; candidate `0d3bc9ce6fedbbe302c62c2441814e81534be59f`; champion `bb770d9c7c9e59d39007f71a106af77cd9eb73a1`; decision discard.
  [Implementation](ideas/normal-slope-nuisance-gp/0d3bc9ce6fedbbe302c62c2441814e81534be59f/algo.py), [train](evaluation_results/cycle-434-train.json), [narrative](full_log.md#cycle-434-marginalized-normal-slope-gp-observations).

## radially-invariant-bend-frame: Unit-bond transverse axis
Status: deferred

**Hypothesis:** Define430's frame axis by the difference of unit bond directions so pure radial stretching cannot change its bend components.
**Outcome and uncertainty:**435 fully converges without errors and passes energy but loses cost. No passive failure identifies an integration or gauge defect; cleaner coordinate semantics do not ensure better geometry paths.
**Reason to revisit:** An independently supported model requiring genuinely pure angular components could use this exact invariant representation.
**Next experiment:** Await that connection; no axis mixture, reference ranking, angle threshold or ODE-budget sweep.

**Attempts:**
- Cycle435; candidate `8ceb8c561319fec96f4e198b97d5d0ffcbaecec0`; champion `bb770d9c7c9e59d39007f71a106af77cd9eb73a1`; decision discard.
  [Implementation](ideas/radially-invariant-bend-frame/8ceb8c561319fec96f4e198b97d5d0ffcbaecec0/algo.py), [train](evaluation_results/cycle-435-train.json), [narrative](full_log.md#cycle-435-radially-invariant-transverse-bend-frame).

## physical-adaptive-gradient: Chemical preconditioning with adaptive first-order steps
Status: deferred

**Hypothesis:** Exact fixed-chart secant smoothness permits line-search-free first-order optimization using an initial chemical metric.
**Outcome and uncertainty:**436 has199converged,270calllimits,energy failure and468/469slower than430. Passive saved force history shows unfinished relaxation without errors; metric/stepsize states are unavailable. This tests fixed-metric AdGD, not all adaptive-gradient methods.
**Reason to revisit:** An independently supported metric-learning or acceleration derivation could address unresolved anisotropy while preserving a common-chart gradient relation.
**Next experiment:** Await that mechanism; no stepsize,growth,initiallength,metricfloor orcap sweep from broadloss.

**Attempts:**
- Cycle436; candidate `8943dad0f192a9a71dd9ccbb4e61a768cef2e92f`; champion `bb770d9c7c9e59d39007f71a106af77cd9eb73a1`; decision invalid. [Implementation](ideas/physical-adaptive-gradient/8943dad0f192a9a71dd9ccbb4e61a768cef2e92f/algo.py), [train](evaluation_results/cycle-436-train.json), [narrative](full_log.md#cycle-436-physically-preconditioned-adaptive-gradient-descent).

**447 acceleration revisit:** NAG-free2025 supplies an independent two-extrema curvature/momentum derivation, replacing436's AdGD recurrence in the identical fixed physical chart. Use its published nonconvex100-step restart and436's unchanged initialization, with no extra derivative probes. This tests the stated acceleration prerequisite, not a stepsize or growth-factor sweep.

- Cycle447; candidate `e3abdf16b9a1cfdb478b444e407ff2ceea7ecdbc`; champion `bb770d9c7c9e59d39007f71a106af77cd9eb73a1`; decision invalid. [Implementation](ideas/physical-adaptive-gradient/e3abdf16b9a1cfdb478b444e407ff2ceea7ecdbc/algo.py), [train](evaluation_results/cycle-447-train.json), [narrative](full_log.md#cycle-447-physical-nag-free-acceleration).
**447 outcome/update:**217converged,205limits,47backend failures;420slower/1faster/1same returned cases. Early passive trace shows multi-Bohr overshoot preceding failure. Acceleration alone did not rescue the fixed metric. Defer pending independently justified evolving geometry/metric with globalization; no curvature-extrema,restart,initialization orcap sweeps from this broad failure.

## posterior-minimum-location: Average differentiable posterior-path minima
Status: deferred

**Hypothesis:** Accounting for nonlinear uncertainty in the minimum location improves physical progress over minimizing the mean surface.
**Outcome and uncertainty:**437 fully converges with valid energy but loses cost; {'slower': 165, 'faster': 64, 'same': 240} versus430. Empty failure_details cannot separate sampler approximation, posterior calibration and decision-loss mismatch. Runtime increases.
**Reason to revisit:** An independently justified physical decision loss or observed posterior-calibration evidence could make the reusable mapped-derivative path sampler useful.
**Next experiment:** Await that evidence; no seed,frequency,count,amplitude,budget oracceptance sweep.

**Attempts:**
- Cycle437; candidate `efbe6210ca9a920034c63de19fc23872b25668b0`; champion `bb770d9c7c9e59d39007f71a106af77cd9eb73a1`; decision discard. [Implementation](ideas/posterior-minimum-location/efbe6210ca9a920034c63de19fc23872b25668b0/algo.py), [train](evaluation_results/cycle-437-train.json), [narrative](full_log.md#cycle-437-posterior-mean-minimum-location).

**475 revisit:** Independent LES2026 supplies a trajectory-information decision criterion instead of437's endpoint averaging. Retain the exact same sampler, seed and microbudget; rank the five-support path union using anchored scalar posterior information, only after unchanged physical proposal guards. This tests the decision-loss prerequisite, not a sampler sweep.

**475 reassessment:** LES trajectory-information selection tests the decision-loss prerequisite, but fully converges with valid energy and zero errors while losing cost; {'slower': 143, 'same': 246, 'faster': 80}. No passive defect supports repair. Revisit needs independent posterior calibration or a physically meaningful information-to-cost relation; no sampler/support/noise/budget/admission sweep.
- Cycle475; candidate `70644cc1c74a8e9fa6a3deae3aa73c14bd5d1395`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/posterior-minimum-location/70644cc1c74a8e9fa6a3deae3aa73c14bd5d1395/algo.py), [training](evaluation_results/cycle-475-train.json), [narrative](full_log.md#cycle-475-local-descent-sequence-entropy-search).

## congruence-physical-calibration: Fit marginal scales of the full physical seed
Status: deferred

**Hypothesis:** Geometrically coupled scaling of complete physical curvature retains useful correlation better than additive component weights.
**Outcome and uncertainty:**438 converges all469 with valid energy but loses cost; {'slower': 97, 'same': 295, 'faster': 77} versus430. Emptyfailure_details leave parameterization, identifiability and nonlinear-fit effects unresolved.
**Reason to revisit:** Independent physical evidence for a marginal/correlation decomposition could justify this positive congruence model with its analytic small-parameter fit.
**Next experiment:** Await that evidence; no group,ridge,bounds,solver-budget ormixing sweep.

**Attempts:**
- Cycle438; candidate `cb7ca9b77a2ba911857e0f0ab1f850e37ef68618`; champion `bb770d9c7c9e59d39007f71a106af77cd9eb73a1`; decision discard. [Implementation](ideas/congruence-physical-calibration/cb7ca9b77a2ba911857e0f0ab1f850e37ef68618/algo.py), [train](evaluation_results/cycle-438-train.json), [narrative](full_log.md#cycle-438-congruence-physical-model-calibration).


## sobolev-neural-proposals: Trainable scalar residual with a GP critic
Status: deferred

**Hypothesis:** Four tanh units trained on paid values and mapped derivatives learn useful nonlinear proposal directions while the exact GP retains admission control.
**Outcome and uncertainty:**440 fully converges with valid energy but loses cost; empty failure_details cannot separate fit,feature and critic effects.
**Reason to revisit:** Independently demonstrated nonlinear feature structure could justify the reusable analytically differentiated, origin-anchored Sobolev model.
**Next experiment:** Await that evidence; no width,seed,prior,noise,trainingbudget orcritic sweeps.

**Attempts:**
- Cycle440; candidate `c5e2dbe969ade765df4f1a25c5cfbd22c849cb06`; champion `bb770d9c7c9e59d39007f71a106af77cd9eb73a1`; decision discard. [Implementation](ideas/sobolev-neural-proposals/c5e2dbe969ade765df4f1a25c5cfbd22c849cb06/algo.py), [train](evaluation_results/cycle-440-train.json), [narrative](full_log.md#cycle-440-sobolev-neural-residual-proposals).

## gaussian-well-mean: Curvature-matched finite joint-mode mean
Status: deferred

**Hypothesis:** A one-point SE well preserving current value,gradient andcurvature improves finite coupled response relative to the harmonic mean.
**Outcome and uncertainty:**443 passes energy andconverges all469without errors but increases cost substantially. Emptyfailure_details leaves no retained proposal evidence.
**Reason to revisit:** Independent physical evidence for a bounded joint-mode energy law or documented surrogate overprediction could justify a different finite continuation. Exact local curvature alone is insufficient.
**Next experiment:** Await that evidence; no trend,width,amplitude,mixing oractivation sweeps. The independently sourced RVO linked model/controller is a different hypothesis.

**Attempts:**
- Cycle443; candidate `6f93a5234b9bbd2778844df6be0caf6879658db1`; champion `bb770d9c7c9e59d39007f71a106af77cd9eb73a1`; decision discard. [Implementation](ideas/gaussian-well-mean/6f93a5234b9bbd2778844df6be0caf6879658db1/algo.py), [train](evaluation_results/cycle-443-train.json), [narrative](full_log.md#cycle-443-curvature-matched-gaussian-well-mean).

## restricted-variance-controller: Linked physical GEK and uncertainty-limited motion
Status: deferred

**Hypothesis:** A physical constant trend andmatched Matern lengths withRFO variance-limited microiterations can use learned total-energy structure beyond a residual proposal gate.
**Outcome and uncertainty:**444 invalid from8ODE/2backenderrors,2calllimits. Returned-subset112faster220slower127same; passive traces showlarge growing movements andfragmentseparation but lack inner modelstates.
**Reason to revisit:** Useful call savings coexist with a localizedmissinggeometricdomain demonstrated by retained trajectories. A cumulative existingtrustbox can limit realization while retaining the new model/controller.
**Next experiment:**Boundedrepair1:intersect microsteps with430 cumulative native delta box andstop atboundary. Keep every444model/varianceconstant. End onpersistingbroadloss; no threshold/rank/history sweeps.

**Attempts:**
- Cycle444; candidate `a4ff9de977252fe6ca6814942121901fad8e6c0e`; champion `bb770d9c7c9e59d39007f71a106af77cd9eb73a1`; decision invalid. [Implementation](ideas/restricted-variance-controller/a4ff9de977252fe6ca6814942121901fad8e6c0e/algo.py), [train](evaluation_results/cycle-444-train.json), [manifest](diagnostics/240ec8ec3b474e15a701c7fd8bdf5b40/manifest.json), [narrative](full_log.md#cycle-444-compact-restricted-variance-gek-controller).

**445 reassessment:**Repair1 converges all469 andremoves8ODE/2backenderrors and2limits, but loses cost andfails energy. Emptyfailure_details/manifest do not identify convergedpath errors. End this repairsequence. Revisit only after independent chart-consistent uncertainty orrealization evidence; no model/domainconstant sweeps.
- Cycle445; candidate `9eb1e9460211f29ecca0d2c02383ab85789503f6`; champion `bb770d9c7c9e59d39007f71a106af77cd9eb73a1`; decision invalid. [Implementation](ideas/restricted-variance-controller/9eb1e9460211f29ecca0d2c02383ab85789503f6/algo.py), [train](evaluation_results/cycle-445-train.json), [manifest](diagnostics/3c2b1f5560a64f0d8af7a2ebd2686146/manifest.json), [narrative](full_log.md#cycle-445-geometric-domain-for-restricted-variance-gek).

## exponential-auxiliary-gp-search: Modified-energy dissipating GP microiterations
Status: deferred

**Hypothesis:** Linearly implicit exponential SAV flow supplies better nonlinear GP proposals without extra paid forces.
**Outcome and uncertainty:**448 fullyconverges with validenergy andzeroerrors butlosescost;{'slower': 23, 'faster': 18, 'same': 428}. Emptyfailure_details leave model-trajectory quality and auxiliary consistency unobserved.
**Reason to revisit:** Independent surrogate-trajectory evidence or a translation-covariant auxiliary formulation could clarify whether modifiedenergy consistency improves search.
**Next experiment:** Await that mechanism/evidence; no timestep,relaxation,microbudget oradmission sweeps.

**Attempts:**
- Cycle448; candidate `85090a13f1b19ceeeceba5ddb6ca54fdbb5568c4`; champion `bb770d9c7c9e59d39007f71a106af77cd9eb73a1`; decision discard. [Implementation](ideas/exponential-auxiliary-gp-search/85090a13f1b19ceeeceba5ddb6ca54fdbb5568c4/algo.py), [train](evaluation_results/cycle-448-train.json), [narrative](full_log.md#cycle-448-exponential-auxiliary-variable-gp-search).

## regional-energy-secant: Least integrated quadratic model change
Status: deferred

**Hypothesis:**A physical-tangent regionalenergy norm givesbettersecant curvature thanmatrix-only leastchange.
**Outcome and uncertainty:**449 losesbroadly:18faster410slower41same,3limits,zeroerrors,validenergy. Passive tracehaslowforces butcontinuedoscillatorymotion; missingmatrix/controllerstatespreventattribution.
**Reason to revisit:** Independentlymeasured regionalcurvature-error statistics or a compatible movingchart could distinguishwhy the inferredcomplementarytrace isunhelpful.
**Next experiment:** Awaitthatevidence/formulation; no normweight,tracefactor,radius oractivation sweep.

**Attempts:**
- Cycle449; candidate `e7c8bb363e3ad5930d2fc60d55aa53e45cc7b62d`; champion `bb770d9c7c9e59d39007f71a106af77cd9eb73a1`; decision discard. [Implementation](ideas/regional-energy-secant/e7c8bb363e3ad5930d2fc60d55aa53e45cc7b62d/algo.py), [train](evaluation_results/cycle-449-train.json), [narrative](full_log.md#cycle-449-regional-energy-model-secant-update).

## delocalized-pi-bond-correlations: Electronic response in the bond prior
Status: incorporated

**Hypothesis:** Nonlocal neutral pi-electron response supplies useful bond correlations while preserving Badger marginal stiffnesses.
**Outcome and uncertainty:** Cycle 450 improves both splits, satisfies both energy gates and converges all 934 without errors. Equal-bond reference and valence typing omit charge, heteroatom and twist response; the geometric slope convention is an explicit inference.
**Reason to revisit:** A separately established orbital-overlap or heteroatom response model could extend the physics without empirical motif rules.
**Next experiment:** No parameter or scope sweep. Continue independent methods against the new champion.

**Attempts:**
- Cycle 450; candidate `19ef38c155eee70f9872b95841931f5a303e74c4`; champion `bb770d9c7c9e59d39007f71a106af77cd9eb73a1`; decision keep. Implementation is the accepted git commit. [Training](evaluation_results/cycle-450-train.json), [validation](evaluation_results/cycle-450-valid.json), [narrative](full_log.md#cycle-450-delocalized-pi-electron-bond-correlations).

**459 heteroatom extension:** Van-Catledge parameters and published neutral coordination types meet the earlier independent-model prerequisite, but universal length slopes and sigma scaling remain assumptions. All training cases converge without errors and energy passes; 48 improve, 54 regress and 367 are unchanged, with an aggregate cost loss. Empty passive failure details do not support parameter/scope repairs. The carbon-only baseline remains incorporated; defer heterogeneous response until independent charge/orientation/length-response evidence resolves its model mismatch.
- Cycle 459; candidate `f1301abf9f081abd897596fa8374be2dbb28d730`; champion `19ef38c155eee70f9872b95841931f5a303e74c4`; decision discard. [Implementation](ideas/delocalized-pi-bond-correlations/f1301abf9f081abd897596fa8374be2dbb28d730/algo.py), [training](evaluation_results/cycle-459-train.json), [narrative](full_log.md#cycle-459-heteroatom-orbital-response-in-bond-correlations).

**463 revisit:** POAV1 and Slater–Koster geometry independently supply current orbital alignment for carbon-only radial response. This tests a spatial coupling mechanism absent from450 without changing452 coordinates or459 chemistry. No overlap/projection/scope sweep is planned.

**463 accepted:** Current POAV1/Slater–Koster alignment improves both gates with full convergence, valid energy and no errors. Retain this radial-response extension; sigma mixing and orbital-axis derivatives remain outside its scope.
- Cycle463; candidate `99a2d8b5482466729ddde624fc4253fc98aba249`; champion `19ef38c155eee70f9872b95841931f5a303e74c4`; decision keep. Accepted source in Git. [Training](evaluation_results/cycle-463-train.json), [validation](evaluation_results/cycle-463-valid.json), [narrative](full_log.md#cycle-463-orbital-alignment-in-carbon-bond-response).

**474 current-geometry reference:** Linear HSSH hopping evaluated at actual initial bond lengths retains463 orientation and radial derivatives. Training passes but validation loses cost, despite all934 converged, valid energies and no errors. Empty failure_details does not support a length, overlap, slope, scope, stiffness, refresh or blend repair. Actual-state evaluation alone does not resolve460's reference-state issue; future extension needs independent sigma-stress or response evidence. Accepted463 remains incorporated.
- Cycle474; candidate `0df8670bde1d97602aa6e30f761817ecd3c9c11d`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision non_generalizable. [Implementation](ideas/delocalized-pi-bond-correlations/0df8670bde1d97602aa6e30f761817ecd3c9c11d/algo.py), [training](evaluation_results/cycle-474-train.json), [validation](evaluation_results/cycle-474-valid.json), [narrative](full_log.md#cycle-474-current-geometry-electronic-bond-response).

## full-matrix-hypergradient: Learn paid energy response of a gradient-to-step map
Status: deferred

**Hypothesis:** Full-matrix online hypergradients learn coupled relaxation directions in a chemical metric without explicit curvature secants.
**Outcome and uncertainty:**451 is slower on every case and fails energy, with 185 call limits and no errors. Passive late force/displacement evidence shows unfinished relaxation but contains no learned-map states.
**Reason to revisit:** A separately supported faster matrix-learning rule or changing chemical metric could address the short-budget warmup.
**Next experiment:** Await that formulation; no hyperstepsize, bound, floor, symmetry or momentum sweep from this broad loss.

**Attempts:**
- Cycle 451; candidate `cb291831dfe15a37f40391c27206a26e7b33786b`; champion `19ef38c155eee70f9872b95841931f5a303e74c4`; decision invalid. [Implementation](ideas/full-matrix-hypergradient/cb291831dfe15a37f40391c27206a26e7b33786b/algo.py), [training](evaluation_results/cycle-451-train.json), [narrative](full_log.md#cycle-451-physical-full-matrix-hypergradient-descent).

## orbital-frame-coordinates: Separate pi twist from carbon pyramidalization
Status: deferred

**Hypothesis:** POAV1 axes define collective carbon deformation coordinates with better finite motion than overlapping proper torsions, while preserving their initial Cartesian spring factors.
**Outcome and uncertainty:**452 passes training but loses validation cost. All 934 converge with valid energy and no errors; passive failure_details are empty. Initial rank/spring preservation does not isolate finite metric and trust effects.
**Reason to revisit:** An independently supported finite-frame metric or a retained coordinate-path error could make the physical decomposition useful.
**Next experiment:** Await that mechanism; no valence scope, rank, geometric threshold or weight sweep.

**Attempts:**
- Cycle 452; candidate `41add323090ea0f5ba14376d3699df575203875e`; champion `19ef38c155eee70f9872b95841931f5a303e74c4`; decision non_generalizable. [Implementation](ideas/orbital-frame-coordinates/41add323090ea0f5ba14376d3699df575203875e/algo.py), [training](evaluation_results/cycle-452-train.json), [validation](evaluation_results/cycle-452-valid.json), [narrative](full_log.md#cycle-452-orbital-frame-twist-and-pyramidalization-coordinates).

## finite-electronic-mean: Higher-order delocalized bond response
Status: deferred

**Hypothesis:** Extend 450's accepted quadratic electronic bond response to an occupied-eigenvalue GP mean, subtracting its complete second-order Taylor jet.
**Outcome and uncertainty:** 454 converges all cases without errors and passes energy, but its training cost gain is below the fixed gate. Finite effective native lengths and the re-centered equal-hopping reference are approximations; no passive failure state identifies their error.
**Reason to revisit:** Independent finite-response or orbital-reference evidence could support a more accurate local collective model. Marginal matching, stable subspace-leakage energy remainder and analytic derivatives are reusable.
**Next experiment:** Await such evidence; no amplitude, x/y, length normalization, scope or confidence sweep to cross the gate. Compare any independently justified revision against the current champion under full gates.

**Attempts:**
- Cycle 454; candidate `684e143f16257b7fb51f4551ab467d530891ac1f`; champion `19ef38c155eee70f9872b95841931f5a303e74c4`; decision discard. [Implementation](ideas/finite-electronic-mean/684e143f16257b7fb51f4551ab467d530891ac1f/algo.py), [training](evaluation_results/cycle-454-train.json), [narrative](full_log.md#cycle-454-finite-delocalized-electronic-surrogate-mean).

**465 revisit:** Accepted463 independently supports an orbital-alignment reference, satisfying the prior reference-evidence condition. Apply the preserved454 finite-response correction with overlap factors consistently present in the Hamiltonian, gradient and Hessian; no amplitude, x/y, normalization or admission tuning.

**465 reassessment:** The independently accepted orbital-reference revision passes training but slightly loses validation cost. Both splits fully converge, pass energy and have no errors or failure_details. This tested reference prerequisite does not make the finite mean generalize; defer until independent evidence identifies finite native-length or re-centering error. No amplitude, x/y, normalization, scope or confidence sweep.
- Cycle465; candidate `2a3a36960d980c638e90564ed5449c51db83fc73`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision non_generalizable. [Implementation](ideas/finite-electronic-mean/2a3a36960d980c638e90564ed5449c51db83fc73/algo.py), [training](evaluation_results/cycle-465-train.json), [validation](evaluation_results/cycle-465-valid.json), [narrative](full_log.md#cycle-465-oriented-finite-electronic-gp-mean).

## feature-coordinate-search: Minimize the GP in its bending features
Status: deferred

**Hypothesis:** Searching the unchanged surrogate in its nonlinear covariance coordinates may improve a bounded BFGS solve.
**Outcome and uncertainty:** 455 fully converges without errors and passes energy but loses cost. The evidence does not separate inverse fallback, search trajectory or changed local minima.
**Reason to revisit:** A retained surrogate-search diagnostic or independently supported inverse branch could justify a more reliable parameterization. The inverse-transpose gradient and deterministic damped inverse are reusable.
**Next experiment:** Await that evidence; no inverse tolerance, rank, microiteration or acceptance sweep. Use the full gates.

**Attempts:**
- Cycle 455; candidate `5c95cf9e1c403dab5d10a27e04b06094a6445827`; champion `19ef38c155eee70f9872b95841931f5a303e74c4`; decision discard. [Implementation](ideas/feature-coordinate-search/5c95cf9e1c403dab5d10a27e04b06094a6445827/algo.py), [training](evaluation_results/cycle-455-train.json), [narrative](full_log.md#cycle-455-surrogate-minimization-in-bending-feature-coordinates).

## innovation-covariance-memory: Conditional symmetric Hessian uncertainty
Status: deferred

**Hypothesis:** Retain learned uncertainty between exact secants and renew it from response innovations to preserve useful curvature information.
**Outcome and uncertainty:** 456 passes energy without errors but loses cost broadly and reaches three call limits. Passive evidence shows unfinished relaxation; it does not retain covariance states to distinguish drift estimation from moving-frame error.
**Reason to revisit:** An independently justified common-frame uncertainty/drift model could address the mismatch between matrix inference and changing native geometry.
**Next experiment:** Await that model or evidence; no inflation, floor, rank, blend or activation sweep. Compare a justified revision against the current champion under unchanged gates.

**Attempts:**
- Cycle 456; candidate `d94619fc4082cb24fd5b56e88e58d07511deaedc`; champion `19ef38c155eee70f9872b95841931f5a303e74c4`; decision discard. [Implementation](ideas/innovation-covariance-memory/d94619fc4082cb24fd5b56e88e58d07511deaedc/algo.py), [training](evaluation_results/cycle-456-train.json), [narrative](full_log.md#cycle-456-innovation-adaptive-covariance-memory-for-secants).

## accelerated-surrogate-search: AdaNAQ on the fitted GP
Status: deferred

**Hypothesis:** Cheap exact look-ahead gradients and adaptive momentum can find better surrogate proposals within twelve microiterations.
**Outcome and uncertainty:** 457 fully converges without errors and passes energy but loses training cost. No passive failure state separates search behavior from prediction error.
**Reason to revisit:** Retained surrogate-search evidence or an independently supported model where momentum finds more reliable minima could justify this solver.
**Next experiment:** Await that evidence; no momentum, gamma, line-search, budget or admission sweep.

**Attempts:**
- Cycle 457; candidate `0d3b302034d5312eb2af2f96c8645ad74a6525d2`; champion `19ef38c155eee70f9872b95841931f5a303e74c4`; decision discard. [Implementation](ideas/accelerated-surrogate-search/0d3b302034d5312eb2af2f96c8645ad74a6525d2/algo.py), [training](evaluation_results/cycle-457-train.json), [narrative](full_log.md#cycle-457-accelerated-quasi-newton-surrogate-search).

## fixed-metric-abc-fire: Bias-corrected inertial engine in chemical units
Status: deferred

**Hypothesis:** A fixed chemical frame and ABC-FIRE reset correction can accelerate coherent inertial relaxation without changing masses or GP replacement.
**Outcome and uncertainty:** 458 is invalid with one backend SCC error, 201 call limits and broad cost regression. The complete short error trace does not establish a general cap/dynamics defect.
**Reason to revisit:** Independent evidence for adapting chemical inertia while retaining a consistent state, or faster reset dynamics under short budgets, could address this engine's limitations. Fixed whitening and counted rollback code are reusable.
**Next experiment:** Await that formulation; no dt, alpha, latency, cap, floor or activation sweep.

**Attempts:**
- Cycle 458; candidate `62d7993ebe807639b75269029ba59fd2174f5c67`; champion `19ef38c155eee70f9872b95841931f5a303e74c4`; decision invalid. [Implementation](ideas/fixed-metric-abc-fire/62d7993ebe807639b75269029ba59fd2174f5c67/algo.py), [training](evaluation_results/cycle-458-train.json), [narrative](full_log.md#cycle-458-fixed-chemical-metric-abc-fire).

## self-consistent-pi-reference: Relax the topological electronic reference
Status: deferred

**Hypothesis:** Self-consistent HSSH bond orders give more useful initial bond correlations than equal hopping, while retaining Badger marginals.
**Outcome and uncertainty:** 460 passes training but loses validation cost; every case converges without errors and both energy gates pass. No passive model state separates successful inner solves from fallback or actual geometry mismatch.
**Reason to revisit:** Independent evidence connecting reference bond response to current geometry could justify a more representative state. The bounded analytic HSSH solve remains reusable.
**Next experiment:** Await that evidence; no tolerance, budget, damping, length, x/y or scope sweep. Full gates for any independently justified model.

**Attempts:**
- Cycle 460; candidate `4a04199c384a736507c536383a5b17dee92c1883`; champion `19ef38c155eee70f9872b95841931f5a303e74c4`; decision non_generalizable. [Implementation](ideas/self-consistent-pi-reference/4a04199c384a736507c536383a5b17dee92c1883/algo.py), [training](evaluation_results/cycle-460-train.json), [validation](evaluation_results/cycle-460-valid.json), [narrative](full_log.md#cycle-460-self-consistent-carbon-electronic-reference).

## local-shape-bend-correlations: Angular stiffness from local rotational quotients
Status: deferred

**Hypothesis:** Eliminating local rotations from unit-bond deformation gives useful bend correlations while retaining marginal chemical stiffnesses.
**Outcome and uncertainty:** 461 fully converges with valid energy but broad cost loss. No passive failure state identifies a numerical defect; redundant angle conditioning and interaction with native geometry remain unresolved.
**Reason to revisit:** Independent evidence for a stable nonredundant shape representation could avoid pseudoinverse-induced soft physical directions.
**Next experiment:** Await that representation or retained curvature evidence; no rank, blend, stiffness, element or activation sweep.

**Attempts:**
- Cycle 461; candidate `26fb4327336136a2af18994f3f92f3e080759b1d`; champion `19ef38c155eee70f9872b95841931f5a303e74c4`; decision discard. [Implementation](ideas/local-shape-bend-correlations/26fb4327336136a2af18994f3f92f3e080759b1d/algo.py), [training](evaluation_results/cycle-461-train.json), [narrative](full_log.md#cycle-461-local-shape-correlations-for-bending).

## cubic-surrogate-search: Adaptive cubic minimization of the fitted model
Status: deferred

**Hypothesis:** Cheap cubic model rejection and signed curvature improve surrogate proposals within twelve microiterations.
**Outcome and uncertainty:** 462 converges all cases with valid energy and zero errors but loses cost. No retained microsearch traces identify a solver defect versus surrogate error.
**Reason to revisit:** Independent evidence relating model curvature and admissible proposal quality could justify the solver; the bounded spectral implementation is reusable.
**Next experiment:** Await that evidence; no sigma, stencil, threshold, budget or admission sweep. It does not unlock239's signed-mean prerequisite.

**Attempts:**
- Cycle462; candidate `54163554867445bbd98b0286b27fd54884e81fc0`; champion `19ef38c155eee70f9872b95841931f5a303e74c4`; decision discard. [Implementation](ideas/cubic-surrogate-search/54163554867445bbd98b0286b27fd54884e81fc0/algo.py), [training](evaluation_results/cycle-462-train.json), [narrative](full_log.md#cycle-462-cubic-regularization-inside-the-gp-search).

## electronic-torsion-stiffness: Delocalized bond order in the torsion prior
Status: deferred

**Hypothesis:** A positive 2*t0*|p| electronic twist curvature supplements the generic proper-torsion background.
**Outcome and uncertainty:**464 loses cost and fails aggregate energy despite full convergence and zero errors. No passive trace separates basin change, redundant sharing or over-stiffening.
**Reason to revisit:** An independently supported projection onto collective orbital twist, with compatible marginal curvature, could avoid contaminating other deformations.
**Next experiment:** Await that model/evidence; no t0, exponent, stiffness, scope, distribution or fit sweep. Full gates for a justified revision.

**Attempts:**
- Cycle464; candidate `31ff24ade9ad1cb6f6215acb72c56ea1cd68f1ee`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision invalid. [Implementation](ideas/electronic-torsion-stiffness/31ff24ade9ad1cb6f6215acb72c56ea1cd68f1ee/algo.py), [training](evaluation_results/cycle-464-train.json), [narrative](full_log.md#cycle-464-electronic-bond-order-torsional-stiffness).

**492 revisit:** Test collective POAV-twist Jacobians with native marginals preserved by diagonal normalization. This supplies the recorded projection/marginal prerequisite without452 coordinate replacement or464 extra diagonal stiffness.

**492 outcome:** Collective twist correlations with preserved marginal curvature also lose cost despite full convergence, valid energy and zero errors; {'same': 426, 'slow': 22, 'fast': 21}. No retained prior state identifies a repair. Revisit now requires independent evidence of the missing orbital response or a better-supported projected physical metric; no projection/scope/strength tuning.
- Cycle492; candidate `b5b6b63940eccdcfdbb1579ab90407e9ccb7e622`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/electronic-torsion-stiffness/b5b6b63940eccdcfdbb1579ab90407e9ccb7e622/algo.py), [training](evaluation_results/cycle-492-train.json), [narrative](full_log.md#cycle-492-collective-orbital-twist-correlations).

**507 revisit:** The same accepted occupied/virtual electronic response used radially in450/463 gives a full signed twist Hessian. Test this missing electronic relaxation while keeping492 projection,2.5eV scale and torsion marginal normalization. This is a distinct physical-model prerequisite, not a strength or scope sweep.

**507 outcome:** Full signed occupied/virtual twist response also loses cost with469 converged,valid energy,zero errors;{'same': 428, 'fast': 22, 'slow': 19}. No retained prior state diagnoses a repair. The missing-relaxation prerequisite has now been tested; require independently supported finite-coordinate electronic curvature or diagnostic model error before further work, without response,t0,scope,rank/floor or fit sweeps.
- Cycle507; candidate `7a19c2082e789694fd81928776a98acf9000ae42`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/electronic-torsion-stiffness/7a19c2082e789694fd81928776a98acf9000ae42/algo.py), [training](evaluation_results/cycle-507-train.json), [narrative](full_log.md#cycle-507-relaxed-electronic-orbital-twist-curvature).

## occupied-orbital-features: Nonlinear electronic covariance geometry
Status: deferred

**Hypothesis:** Occupied-subspace projector response changes finite GP similarity while preserving its local feature Jacobian.
**Outcome and uncertainty:**466 converges all cases with valid energy and zero errors but loses cost. Empty failure_details cannot separate response normalization, feature folding and reference-geometry approximations.
**Reason to revisit:** Independent evidence that orbital-state similarity predicts residual-energy correlation, or retained feature/Jacobian diagnostics, could justify a better representation.
**Next experiment:** Await that evidence; no rank, scale, width, scope or blending sweep.

**Attempts:**
- Cycle466; candidate `cfc7ddf90f973f0b2c1cd0656f27f15282fb97e8`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/occupied-orbital-features/cfc7ddf90f973f0b2c1cd0656f27f15282fb97e8/algo.py), [training](evaluation_results/cycle-466-train.json), [narrative](full_log.md#cycle-466-occupied-orbital-covariance-features).

## forecast-weighted-curvature: Online prediction weighting of Hessian models
Status: deferred

**Hypothesis:** Out-of-sample secant forecast losses identify which curvature update is useful before mixing models.
**Outcome and uncertainty:**467 fully converges with valid energy and no errors but loses training cost. Empty failure_details leaves frame drift, loss mismatch and model correlation unresolved.
**Reason to revisit:** Independent evidence connecting calibrated gradient forecasts with useful optimization directions, or a common-frame model, could make prediction weighting meaningful.
**Next experiment:** Await that evidence; no loss normalization, expert-set, forgetting, rate or activation sweep.

**Attempts:**
- Cycle467; candidate `05fed6d07d7fb13b352bfe0c507d478aa78e3b2f`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/forecast-weighted-curvature/05fed6d07d7fb13b352bfe0c507d478aa78e3b2f/algo.py), [training](evaluation_results/cycle-467-train.json), [narrative](full_log.md#cycle-467-forecast-weighted-curvature-experts).

## independent-electronic-calibration: Separate marginal and electronic scales
Status: deferred

**Hypothesis:** Existing paid secants can distinguish electronic correlation from marginal bond stiffness.
**Outcome and uncertainty:**468 fully converges with valid energy and no errors but loses cost. Weak identifiability and changed regularization remain unresolved; no passive fit state.
**Reason to revisit:** Independently informative electronic-response observations or retained coefficient uncertainty could justify this extra parameter.
**Next experiment:** Await that evidence; no prior, bounds, ridge, grouping, timing or strength sweep.

**Attempts:**
- Cycle468; candidate `25f3017b17862ac570bf0ee3f6e30f0f36ba5492`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/independent-electronic-calibration/25f3017b17862ac570bf0ee3f6e30f0f36ba5492/algo.py), [training](evaluation_results/cycle-468-train.json), [narrative](full_log.md#cycle-468-independent-electronic-correlation-calibration).

## powell-hybrid-surrogate: Root-merit search on the GP gradient
Status: deferred

**Hypothesis:** Hybrid Newton/dogleg root finding gives useful stationary proposals under the existing energy and uncertainty critic.
**Outcome and uncertainty:**469 fully converges with valid energy and no errors but slightly loses cost. No passive search state distinguishes saddle roots, model mismatch and solver paths.
**Reason to revisit:** Independent evidence linking predicted stationarity with accepted physical progress could support this bounded alternative search.
**Next experiment:** Await evidence; no factor, tolerance, budget, seed, restart or admission sweep.

**Attempts:**
- Cycle469; candidate `93fb8d45d2223b7b1b79b40ad6db054186dfa744`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/powell-hybrid-surrogate/93fb8d45d2223b7b1b79b40ad6db054186dfa744/algo.py), [training](evaluation_results/cycle-469-train.json), [narrative](full_log.md#cycle-469-powell-hybrid-surrogate-stationarity).

## critical-quadratic-flow: Modewise damped restricted Newton path
Status: deferred

**Hypothesis:** Exact critically damped quadratic response allocates bounded movement better across stiff and soft modes.
**Outcome and uncertainty:**470 fully converges with valid energy and no errors but loses cost. No restriction/integration failure supports repair.
**Reason to revisit:** Independent evidence for useful mode relaxation times or a justified nonlinear flow correction could change the trajectory.
**Next experiment:** Await that evidence; no damping, radius, duration, floor or activation sweep.

**Attempts:**
- Cycle470; candidate `29f6f2dbffeffe4a1895ab2d881a41e5a99cda22`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/critical-quadratic-flow/29f6f2dbffeffe4a1895ab2d881a41e5a99cda22/algo.py), [training](evaluation_results/cycle-470-train.json), [narrative](full_log.md#cycle-470-critically-damped-quadratic-step-path).

## log-distance-secant: Positive least multiplicative curvature change
Status: deferred

**Hypothesis:** Minimize squared logarithmic relative eigenvalue changes while exactly fitting the newest positive tangent secant.
**Outcome and uncertainty:**471 fully converges with valid energy and no errors but loses cost. No passive update state distinguishes frame error and unsuitable least-change geometry.
**Reason to revisit:** Independent multiplicative curvature-error evidence or a consistent frame model could justify this exact bounded projection.
**Next experiment:** Await that evidence; no norm, floor, bounds, timing, blending or eligibility sweep.

**Attempts:**
- Cycle471; candidate `6890f87d9e6ab1a148d99dc8243a3a75850049e3`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/log-distance-secant/6890f87d9e6ab1a148d99dc8243a3a75850049e3/algo.py), [training](evaluation_results/cycle-471-train.json), [narrative](full_log.md#cycle-471-least-logarithmic-spectral-secant-change).

## neural-tangent-covariance: Hidden-parameter sensitivity covariance
Status: deferred

**Hypothesis:** A smooth neural tangent kernel gives more useful residual correlations than spatial SE or random-network output covariance.
**Outcome and uncertainty:**472 fully converges with valid energy and no errors but loses cost. Empty passive failure evidence leaves extrapolation, feature geometry and model error unresolved.
**Reason to revisit:** Independent evidence that parameter sensitivities predict molecular residual correlation could justify this covariance.
**Next experiment:** Await that evidence; no depth, bias, amplitude, length, noise or confidence sweep.

**Attempts:**
- Cycle472; candidate `4c648c9b1f20fddd628ad59c00de266c12815a07`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/neural-tangent-covariance/4c648c9b1f20fddd628ad59c00de266c12815a07/algo.py), [training](evaluation_results/cycle-472-train.json), [narrative](full_log.md#cycle-472-neural-tangent-residual-covariance).

## physical-distance-weighted-descent: DoWG in a fixed chemical metric
Status: deferred

**Hypothesis:** Distance-weighted normalization adapts gradient step length without paid searches or moving secant models.
**Outcome and uncertainty:**473 is invalid:464 limits,5 converged,zero errors, every case slower and energy fails. A passive final trace retains large force and motion; no metric/schedule state identifies a local defect.
**Reason to revisit:** Independently supported distance estimates or a compatible evolving physical metric might connect this schedule to fast molecular relaxation; current evidence supplies no useful cost signal.
**Next experiment:** Await that mechanism/evidence; no initial radius, cap, floor, weighting, decay or restart sweep.

**Attempts:**
- Cycle473; candidate `dabaccd54e5a8bd63eb04123755cb436e1e369f2`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision invalid. [Implementation](ideas/physical-distance-weighted-descent/dabaccd54e5a8bd63eb04123755cb436e1e369f2/algo.py), [training](evaluation_results/cycle-473-train.json), [narrative](full_log.md#cycle-473-physical-metric-distance-weighted-descent).

## relevance-physical-calibration: Independent coefficient uncertainty in early physical fitting
Status: deferred

**Hypothesis:** Bayesian relevance inference preserves unsupported physical prior terms while learning supported coefficient corrections.
**Outcome and uncertainty:**476 fully converges with valid energy and no errors but loses cost. Empty passive evidence leaves prior identification, residual correlations and finite EM convergence unresolved.
**Reason to revisit:** Independently supported coefficient-error statistics or a consistent observation covariance could make selective shrinkage useful.
**Next experiment:** Await that evidence; no prior, bounds, noise, iteration, grouping or clip sweep.

**Attempts:**
- Cycle476; candidate `de8848a555a2a43bed45b318e6165e0301fce41b`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/relevance-physical-calibration/de8848a555a2a43bed45b318e6165e0301fce41b/algo.py), [training](evaluation_results/cycle-476-train.json), [narrative](full_log.md#cycle-476-relevance-learning-for-physical-calibration).

## taylor-coefficient-covariance: Independent local expansion uncertainties
Status: deferred

**Hypothesis:** An entire Taylor-coefficient prior represents residual anharmonicity better than a stationary kernel.
**Outcome and uncertainty:**477 passes energy with full convergence and zero errors but loses training cost. No passive failure evidence; unknown coefficient statistics and approximate charts remain.
**Reason to revisit:** Independently justified tensor-coefficient error statistics or a common historical chart may improve the local prior.
**Next experiment:** Await that mechanism; no order, scale, noise or confidence sweep.

**Attempts:**
- Cycle477; candidate `1667f001825626c1d413d314d333d13b004f7850`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/taylor-coefficient-covariance/1667f001825626c1d413d314d333d13b004f7850/algo.py), [training](evaluation_results/cycle-477-train.json), [narrative](full_log.md#cycle-477-taylor-coefficient-residual-covariance).

## nonconvex-energy-adaptive-descent: AdaPGNC in fixed physical coordinates
Status: deferred

**Hypothesis:** Energy-based lower curvature plus gradient smoothness can choose efficient one-call steps without a surrogate or search.
**Outcome and uncertainty:**478 fails energy,250 limits219 converged0 errors and broad cost loss. A retained final trace has unresolved force and motion; no schedule/metric state diagnoses a local defect.
**Reason to revisit:** A supported coupling between evolving physical conditioning and the curvature adaptation may make this principle useful.
**Next experiment:** Await that mechanism/evidence; no cap, floor, starting size, schedule or branch-constant sweep.

**Attempts:**
- Cycle478; candidate `94099c9a0954f0aa9f2dbc0b628dbfb06313d236`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision invalid. [Implementation](ideas/nonconvex-energy-adaptive-descent/94099c9a0954f0aa9f2dbc0b628dbfb06313d236/algo.py), [training](evaluation_results/cycle-478-train.json), [narrative](full_log.md#cycle-478-energy-aware-nonconvex-adaptive-descent).

**Cycle500 revisit:** Independent AdaBBNC source2608.22430 supplies a signed-BB/energy curvature bound and a fixed published growth sequence, beyond478’s norm/energy truncation. Test the complete formulation under the same physical metric and cap. This is not a localized repair claim; no parameter sweeps planned.

**500 outcome:** Independent AdaBBNC has287 converged182 limits0 errors, invalid energy and465 slower2 faster2same; passive late trace remains above force/motion/energy thresholds without internal curvature state. No parameter repair justified. Await an independently supported evolving geometry/metric; no tau/theta,rho,initial-size,cap or floor sweeps.
- Cycle500; candidate `6338b64ad83c681c7b5e3a34a279b744be8aaa0d`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision invalid. [Implementation](ideas/nonconvex-energy-adaptive-descent/6338b64ad83c681c7b5e3a34a279b744be8aaa0d/algo.py), [training](evaluation_results/cycle-500-train.json), [narrative](full_log.md#cycle-500-nonconvex-bb-physical-descent).

## orbital-strength-bending-mean: Locally normalized VALBOND pair energy
Status: deferred

**Hypothesis:** Orbital orthogonality strength describes finite bending with fewer residual force observations.
**Outcome and uncertainty:**479 pairwise and511 joint-shell means both fully converge with valid energy/zero errors but lose cost.511 has4 faster,1 slower,464 same; no mean-state diagnostics. Its joint square root now represents mixed orbital response, so the missing joint-shell prerequisite has been tested.
**Reason to revisit:** Independent finite-angle electronic response or orbital-error statistics could calibrate the frozen-reference hybrid approximation.
**Next experiment:** Await that calibration; no scope,angle-domain,weights,hybrid participation or mean/kernel blending sweeps.

**Attempts:**
- Cycle479; candidate `5f062ac5f924a4a06f3cee0592607a3c2b403412`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/orbital-strength-bending-mean/5f062ac5f924a4a06f3cee0592607a3c2b403412/algo.py), [training](evaluation_results/cycle-479-train.json), [narrative](full_log.md#cycle-479-orbital-strength-bending-mean).
- Cycle511; candidate `53602b421754800916dbaf55677199cdc87b0677`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/orbital-strength-bending-mean/53602b421754800916dbaf55677199cdc87b0677/algo.py), [training](evaluation_results/cycle-511-train.json), [narrative](full_log.md#cycle-511-joint-shell-bending-mean).

## minimum-transport-compliance: Bures projection of inverse Hessian
Status: deferred

**Hypothesis:** Minimum positional-Gaussian transport under the latest secant preserves useful transverse compliance.
**Outcome and uncertainty:**480 fails energy with70 limits and broad cost loss;41 faster277 slower151 same. Passive late trace shows force plateau but no Hessian/compliance states.
**Reason to revisit:** Independent evidence linking the secant observation and Gaussian compliance evolution could resolve stiffening versus curved-secant mismatch.
**Next experiment:** Await that evidence; no metric blends, inversion variants, root budgets, activation or floors.

**Attempts:**
- Cycle480; candidate `1c3dd631fcfe85e01a323901d400d757bd2c9577`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision invalid. [Implementation](ideas/minimum-transport-compliance/1c3dd631fcfe85e01a323901d400d757bd2c9577/algo.py), [training](evaluation_results/cycle-480-train.json), [narrative](full_log.md#cycle-480-minimum-transport-compliance-secant).

## affine-nuisance-kriging: Integrated affine residual trend
Status: deferred

**Hypothesis:** Removing affine nuisance coefficients before interpolation can improve learned nonlinear curvature.
**Outcome and uncertainty:**481 fully converges with valid energy and zero errors, but loses cost;122 faster137 slower210 same. No passive trend or posterior state diagnoses the cause.
**Reason to revisit:** Independent evidence of affine residual contamination or a consistent anchoring/uncertainty model could justify revisiting.
**Next experiment:** Await that evidence; no trend order, scale, ridge, noise or confidence sweeps.

**Attempts:**
- Cycle481; candidate `207b997d02bd5771ef39d0bdadd6740159313ac2`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/affine-nuisance-kriging/207b997d02bd5771ef39d0bdadd6740159313ac2/algo.py), [training](evaluation_results/cycle-481-train.json), [narrative](full_log.md#cycle-481-affine-nuisance-universal-kriging).

## hessian-damped-surrogate: Gradient-difference inertial GP search
Status: deferred

**Hypothesis:** Hessian-like damping from gradient differences can improve a bounded cheap search.
**Outcome and uncertainty:**482 fully converges with valid energy and zero errors, but loses cost;{'slower': 11, 'same': 447, 'faster': 11}. No retained microsearch state diagnoses the small regression.
**Reason to revisit:** Independently supported nonconvex inertial step control or recorded search-error evidence could distinguish damping from under-solving.
**Next experiment:** Await that evidence; no inertia, damping, microbudget or backtrack sweeps.

**Attempts:**
- Cycle482; candidate `ad49bf059674bd576be7f44b939cf9efd200d9b5`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/hessian-damped-surrogate/ad49bf059674bd576be7f44b939cf9efd200d9b5/algo.py), [training](evaluation_results/cycle-482-train.json), [narrative](full_log.md#cycle-482-hessian-damped-inertial-surrogate-search).

## physical-projected-dynamics: Force-parallel momentum in fixed physical coordinates
Status: deferred

**Hypothesis:** Immediate velocity projection retains downhill inertia while removing transverse oscillation without learned secants.
**Outcome and uncertainty:**483 fails energy,462 limits7 converged0 errors;all469 slower. A bounded late trace shows alternating energy/force with large finite motion; no velocity/spectrum state.
**Reason to revisit:** Independent spectral-stability control coupled to a justified evolving physical metric could address the observed oscillation.
**Next experiment:** Await that mechanism; no timestep, cap, floor or projection sweeps given the broad absence of useful signal.

**Attempts:**
- Cycle483; candidate `9f69a00d8ee1c78b0239f72d4d4b548d0dc0fe8d`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision invalid. [Implementation](ideas/physical-projected-dynamics/9f69a00d8ee1c78b0239f72d4d4b548d0dc0fe8d/algo.py), [training](evaluation_results/cycle-483-train.json), [narrative](full_log.md#cycle-483-physical-metric-projected-dynamics).

## rectified-feature-covariance: First-order arc-cosine GP residual
Status: deferred

**Hypothesis:** A mean-square differentiable random-feature residual learns nonlinear variation with less data than an analytic prior.
**Outcome and uncertainty:**484 fully converges with valid energy and zero errors but loses cost;{'slower': 237, 'faster': 89, 'same': 143}. No retained covariance/derivative state isolates the reason.
**Reason to revisit:** Independently supported residual regularity or calibrated prediction-error evidence could motivate a compatible rough prior.
**Next experiment:** Await that evidence; no order, depth, bias, scale, noise or confidence sweeps.

**Attempts:**
- Cycle484; candidate `08f29c775d184f522af52a25db81b76b1f363699`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/rectified-feature-covariance/08f29c775d184f522af52a25db81b76b1f363699/algo.py), [training](evaluation_results/cycle-484-train.json), [narrative](full_log.md#cycle-484-rectified-feature-residual-covariance).

## shared-atom-secant: Sparse primitive correction with exact free secant
Status: deferred

**Hypothesis:** Restrict new curvature coupling to shared-atom coordinates while preserving existing physical correlations.
**Outcome and uncertainty:**485 passes energy,468 converged1 limit0 errors, but365 slower10 faster94 same. Late motion remains finite; no Hessian/normal-system state identifies a repair.
**Reason to revisit:** Independent residual-locality evidence could support a chemically justified support model or coupling mechanism.
**Next experiment:** Await that evidence; no support-radius, weighting, damping, rank or activation sweeps.

**Attempts:**
- Cycle485; candidate `32c93c9aee7d3fd24515298c290a41ee3ac33f36`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/shared-atom-secant/32c93c9aee7d3fd24515298c290a41ee3ac33f36/algo.py), [training](evaluation_results/cycle-485-train.json), [narrative](full_log.md#cycle-485-shared-atom-sparse-secant-corrections).

## gradient-history-surrogate: Adagrad boxes for the cheap GP search
Status: deferred

**Hypothesis:** Gradient-history bounds with signed model curvature improve proposals within twelve microiterations.
**Outcome and uncertainty:**487 fully converges with valid energy and zero errors but loses cost;{'slow': 7, 'fast': 14, 'same': 448}. Empty failure_details; no inner trajectory identifies the cause.
**Reason to revisit:** Independent evidence that the surrogate is under-solved, or a supported finite-domain Cauchy model, could justify a different search formulation.
**Next experiment:** Await that evidence; no weight, power, curvature cap, stencil, budget or admission sweeps.

**Attempts:**
- Cycle487; candidate `5692d16809c5f3c71cf8a5bb7691f439ecdebf7d`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/gradient-history-surrogate/5692d16809c5f3c71cf8a5bb7691f439ecdebf7d/algo.py), [training](evaluation_results/cycle-487-train.json), [narrative](full_log.md#cycle-487-gradient-history-surrogate-trust-boxes).

## full-matrix-physical-adagrad: Correlation-adaptive gradient steps in a fixed chemical chart
Status: deferred

**Hypothesis:** A cumulative full gradient matrix damps repeatedly active directions while retaining less-observed physical modes.
**Outcome and uncertainty:**489 has469 limits0 converged0 errors, fails energy and all469 slower. A late passive trace shows alternating energy and large motion; no history-matrix state.
**Reason to revisit:** Independent evidence or a source-supported local curvature connection could explain how a gradient-moment metric should evolve for deterministic molecular forces.
**Next experiment:** Await that mechanism; no normalization, decay, history, rate, cap or floor sweeps after complete absence of speed signal.

**Attempts:**
- Cycle489; candidate `7b7e8b6b88eba870f318e5eba80712c5d999fc73`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision invalid. [Implementation](ideas/full-matrix-physical-adagrad/7b7e8b6b88eba870f318e5eba80712c5d999fc73/algo.py), [training](evaluation_results/cycle-489-train.json), [narrative](full_log.md#cycle-489-full-matrix-physical-gradient-adaptation).

## strain-energy-covariance: Prior strain energy as a covariance feature
Status: deferred

**Hypothesis:** Distinguish physical strain levels in the GP feature space to avoid misleading residual correlations.
**Outcome and uncertainty:**490 fully converges with valid energy and zero errors, but loses cost; {'slow': 118, 'fast': 107, 'same': 244}. Empty failure_details; no residual-correlation state.
**Reason to revisit:** Independent evidence that residual errors depend on physical strain could justify a physically calibrated feature.
**Next experiment:** Await that evidence; no lift-strength, mean-mixture, shape, noise or confidence sweeps.

**Attempts:**
- Cycle490; candidate `ded216c90e3c501386243fe28533c0a94b2f7550`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/strain-energy-covariance/ded216c90e3c501386243fe28533c0a94b2f7550/algo.py), [training](evaluation_results/cycle-490-train.json), [narrative](full_log.md#cycle-490-physical-strain-energy-kernel-feature).

## rescaled-hermite-covariance: Constant-normalized residual GP with derivative data
Status: deferred

**Hypothesis:** Normalize the covariance by its constant interpolant to reduce extrapolation distortion.
**Outcome and uncertainty:**491 fully converges with valid energy and zero errors, but loses cost; {'fast': 70, 'same': 335, 'slow': 64}. Empty failure_details; no denominator or posterior state.
**Reason to revisit:** Independently supported evidence of constant-extrapolation bias or normalization-domain failures could justify a changed formulation.
**Next experiment:** Await that evidence; no denominator-threshold, amplitude, noise, length or trend sweeps.

**Attempts:**
- Cycle491; candidate `13363cf2b968747238847fadeaa6b40db4841b3d`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/rescaled-hermite-covariance/13363cf2b968747238847fadeaa6b40db4841b3d/algo.py), [training](evaluation_results/cycle-491-train.json), [narrative](full_log.md#cycle-491-constant-normalized-hermite-covariance).

## weak-inverse-bfgs: Weak inverse curvature correction before BFGS
Status: deferred

**Hypothesis:** Correct inverse directional curvature before the full secant while avoiding global scaling.
**Outcome and uncertainty:**493 fully converges with valid energy and zero errors but loses cost; {'slow': 114, 'same': 307, 'fast': 48}. Empty failure_details leaves interaction with transport and the native controller unresolved.
**Reason to revisit:** Independent evidence of inverse-curvature bias or an energy-faithful globalization mechanism could justify this positive update.
**Next experiment:** Await that evidence; no scaling, damping, blend or activation sweeps.

**Attempts:**
- Cycle493; candidate `037e01ccadefe84a4da136345b1ffc17a0815e33`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/weak-inverse-bfgs/037e01ccadefe84a4da136345b1ffc17a0815e33/algo.py), [training](evaluation_results/cycle-493-train.json), [narrative](full_log.md#cycle-493-weak-inverse-secant-followed-by-bfgs).

## relativistic-surrogate-search: Conformal relativistic dynamics on the GP
Status: deferred

**Hypothesis:** Smooth bounded momentum improves model minimization within twelve cheap iterations.
**Outcome and uncertainty:**494 fully converges with valid energy and zero errors but loses cost; {'same': 443, 'fast': 13, 'slow': 13}. Empty failure_details; no microtrajectory supports a targeted repair.
**Reason to revisit:** Retained model-search errors or an independently supported curvature-matched kinetic model could justify this search.
**Next experiment:** Await that evidence; no damping, speed, timestep, iteration or admission sweeps.

**Attempts:**
- Cycle494; candidate `c04296b323180f32a457a7eb31ceffafb2d5fd48`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/relativistic-surrogate-search/c04296b323180f32a457a7eb31ceffafb2d5fd48/algo.py), [training](evaluation_results/cycle-494-train.json), [narrative](full_log.md#cycle-494-relativistic-momentum-surrogate-search).

## physical-silver-descent: Long and short gradient steps in a fixed physical metric
Status: deferred

**Hypothesis:** A deterministic silver schedule yields useful multistep acceleration after chemical whitening.
**Outcome and uncertainty:**495 has172 converged297 limits0 errors, fails energy and468 slower1 faster. Passive late trace shows small displacement but substantial residual force; no smoothness or phase state.
**Reason to revisit:** An independently supported nonconvex smoothness mechanism or evolving chemical metric could preserve useful long-step progress.
**Next experiment:** Await that mechanism; no L multiplier, phase, cap, floor or schedule sweeps.

**Attempts:**
- Cycle495; candidate `3574e1feea6e7406ee26f6b5e48478dd9f50ecd9`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision invalid. [Implementation](ideas/physical-silver-descent/3574e1feea6e7406ee26f6b5e48478dd9f50ecd9/algo.py), [training](evaluation_results/cycle-495-train.json), [narrative](full_log.md#cycle-495-silver-steps-in-a-fixed-physical-metric).

## band-limited-residual-covariance: Isotropic compact-spectrum Hermite GP
Status: deferred

**Hypothesis:** Excluding high frequencies improves smooth residual interpolation from short history.
**Outcome and uncertainty:**496 fully converges and passes energy with zero errors, but loses cost;{'slow': 120, 'fast': 76, 'same': 273}. No passive interpolation state.
**Reason to revisit:** Independent residual spectral evidence or a diagnosed numerical covariance failure could justify the analytic uniform-ball construction.
**Next experiment:** Await that evidence; no bandwidth, dimension, spectrum-mixture, noise or admission sweeps.

**Attempts:**
- Cycle496; candidate `4919efbc03bbc35159a3a485133a4f04bee78b8b`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/band-limited-residual-covariance/4919efbc03bbc35159a3a485133a4f04bee78b8b/algo.py), [training](evaluation_results/cycle-496-train.json), [narrative](full_log.md#cycle-496-band-limited-residual-covariance).

## interacting-pi-response: Coulomb and exchange feedback in electronic bond susceptibility
Status: deferred

**Hypothesis:** Coupled occupied/virtual transitions improve bond correlations while preserving the accepted orbital reference and Badger marginals.
**Outcome and uncertainty:**497 fully converges and passes energy with zero errors but loses cost;{'same': 462, 'slow': 6, 'fast': 1}. No response-state diagnostics.
**Reason to revisit:** Independently supported reference-interaction consistency or retained susceptibility errors could justify this coupled response.
**Next experiment:**549 addresses the reference-interaction mismatch structurally: exact neutral PPP reference and its own excited-state susceptibility, with unchanged U/V/t0 and frozen geometry integrals. Dense dimension cap512 limits work; larger components retain463. No parameter sweep.

**Attempts:**
- Cycle497; candidate `3268a2b85c4f716aadbd79c56e4e6964c5953e49`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/interacting-pi-response/3268a2b85c4f716aadbd79c56e4e6964c5953e49/algo.py), [training](evaluation_results/cycle-497-train.json), [narrative](full_log.md#cycle-497-interacting-pi-electron-bond-response).

**Revisit549:** Exact neutral PPP diagonalization and its full spectral susceptibility now share a Hamiltonian, fulfilling the former reference-consistency prerequisite. It still loses cost (5faster14slower450same), with full convergence, validenergy,zeroerrors and no passive response diagnostics. This exhausts the current consistency lead; await independent evidence for a different physical response mechanism, without U/V/t0,workcap,screening or tolerance sweeps.
- Cycle549; candidate `bad41a8eb3951bdb43505365a195dc95e518eefe`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/interacting-pi-response/bad41a8eb3951bdb43505365a195dc95e518eefe/algo.py), [training](evaluation_results/cycle-549-train.json), [narrative](full_log.md#cycle-549-correlated-pi-electron-reference-and-response).

## physical-cat-trust: Signed quasi-Newton CAT in a fixed physical chart
Status: deferred

**Hypothesis:** Force-aware energy progress and signed trust steps globalize secant curvature efficiently.
**Outcome and uncertainty:**498 passes energy with461 converged8 limits0 errors but457 slower7 faster5same. Passive trace has late motion and no radius/Hessian state; no localized repair.
**Reason to revisit:** An independently supported compatibility condition between learned curvature, physical metric and CAT's progress rule could use this complete solver.
**Next experiment:** Await that condition; no radius,ratio,cap,floor or update sweeps.

**Attempts:**
- Cycle498; candidate `de8fc6f22d8a38cef4f308f5a319def123c08028`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/physical-cat-trust/de8fc6f22d8a38cef4f308f5a319def123c08028/algo.py), [training](evaluation_results/cycle-498-train.json), [narrative](full_log.md#cycle-498-physical-cat-trust-region-solver).

## stationary-branch-surrogate: Follow a convex-to-GP stationary branch
Status: deferred

**Hypothesis:** A positive stationary branch can reach a useful GP minimum from the ordinary step.
**Outcome and uncertainty:**499 fully converges with valid energy and zero errors; gain0.0000875134 is below gate;{'fast': 10, 'same': 454, 'slow': 5}. No inner-state diagnostics.
**Reason to revisit:** Independent evidence of continuation truncation or a better-supported branch-following formulation could retain this small useful signal.
**Next experiment:** Await that evidence; no increment, corrector, tolerance, finite-difference, budget or fallback sweeps.

**Attempts:**
- Cycle499; candidate `80d23e4f004651278cb793dd80a2888d35d48003`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/stationary-branch-surrogate/80d23e4f004651278cb793dd80a2888d35d48003/algo.py), [training](evaluation_results/cycle-499-train.json), [narrative](full_log.md#cycle-499-convex-to-gp-stationary-continuation).

## opposite-bond-angle-fit: Learn nonincident central bond-angle response
Status: deferred

**Hypothesis:** Other-ligand bending responds to stretching a central bond through shared valence.
**Outcome and uncertainty:**502 fully converges with valid energy and zero errors but loses cost;{'slow': 42, 'same': 380, 'fast': 47}. No coefficient traces identify overfit versus transfer error.
**Reason to revisit:** Independent cross-response data or a calibrated joint valence model could justify the missing-topology channel.
**Next experiment:** Await that evidence; no topology/element restrictions,sign,amplitude,ridge,bounds or fitting-window sweeps.

**Attempts:**
- Cycle502; candidate `025caf2ff863b08a2b8a121bb28d5f8833ac2219`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/opposite-bond-angle-fit/025caf2ff863b08a2b8a121bb28d5f8833ac2219/algo.py), [training](evaluation_results/cycle-502-train.json), [narrative](full_log.md#cycle-502-opposite-ligand-stretch-bend-calibration).

## two-plane-physical-momentum: Two-plane proximal momentum in physical coordinates
Status: deferred

**Hypothesis:** Paid energy progress determines momentum through a two-plane dual solution.
**Outcome and uncertainty:**503 has467 limits,two backend SCC errors,no convergence; all returned cases slower. Passive trace shows persistent force and motion but lacks plane state.
**Reason to revisit:** An independently supported compatibility of the two-plane intercept with nonconvex physical curvature and bounded moves could make this complete solver useful.
**Next experiment:** Await that derivation; no eta,lambda,beta ceiling,cap,floor or restart sweeps.

**Attempts:**
- Cycle503; candidate `08a4c21a5006e89807cc1cc3f1e2b57981ad8672`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision invalid. [Implementation](ideas/two-plane-physical-momentum/08a4c21a5006e89807cc1cc3f1e2b57981ad8672/algo.py), [training](evaluation_results/cycle-503-train.json), [narrative](full_log.md#cycle-503-two-plane-adaptive-physical-momentum).

## arcsinh-output-process: Invertible residual-output process
Status: deferred

**Hypothesis:** Exact inverse-normal output moments capture heavy-tailed physical-model error.
**Outcome and uncertainty:**505 has valid energy,469 convergence,zero errors but loses cost;{'slow': 65, 'fast': 47, 'same': 357}. No retained latent/model traces.
**Reason to revisit:** Independently calibrated physical residual/noise units could separate output-distribution benefit from uncertain energy normalization.
**Next experiment:** Await that calibration; no pivot,scale,tail,kernel/noise or confidence sweeps.

**Attempts:**
- Cycle505; candidate `17809cee457ba0a72de8b896d357e744d2aa95c8`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/arcsinh-output-process/17809cee457ba0a72de8b896d357e744d2aa95c8/algo.py), [training](evaluation_results/cycle-505-train.json), [narrative](full_log.md#cycle-505-arcsinh-output-warped-residual-process).

## physical-two-restart-acceleration: Paid accelerated descent with two restart mechanisms
Status: deferred

**Hypothesis:** Epoch energy decrease and observed Hessian variation determine when to reset accelerated descent.
**Outcome and uncertainty:**506 has128 converged,341 limits,no errors,energy failure and468 slower cases. Passive late forces remain above threshold; no restart state is retained.
**Reason to revisit:** An independently supported moving physical metric or lower-cost variation observation may address the complete solver's force-call overhead.
**Next experiment:** Await that mechanism; no L/M,alpha/beta,cap,floor or restart-threshold sweeps.

**Attempts:**
- Cycle506; candidate `94932ddf87defe9fe21ad9cf292c5454a27305e4`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision invalid. [Implementation](ideas/physical-two-restart-acceleration/94932ddf87defe9fe21ad9cf292c5454a27305e4/algo.py), [training](evaluation_results/cycle-506-train.json), [narrative](full_log.md#cycle-506-physical-accelerated-descent-with-two-restarts).

## physical-resilient-propagation: Sign-adaptive physical mode descent
Status: deferred

**Hypothesis:** Per-mode sign persistence and selective rollback adapt to different curvatures without Hessian learning.
**Outcome and uncertainty:**508 has394 converged,75 limits,no errors,energy failure and467 slower cases. Passive late forces/motion remain too large; no mode-width state is retained.
**Reason to revisit:** An independently supported basis-invariant physical sign metric could address sensitivity to fixed eigenvectors while retaining magnitude robustness.
**Next experiment:** Await that formulation; no widths,growth/shrink,cap,floor,basis or rollback sweeps.

**Attempts:**
- Cycle508; candidate `ce283f85ef8edde5a0739d361bc8e9244852c822`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision invalid. [Implementation](ideas/physical-resilient-propagation/ce283f85ef8edde5a0739d361bc8e9244852c822/algo.py), [training](evaluation_results/cycle-508-train.json), [narrative](full_log.md#cycle-508-resilient-propagation-in-physical-modes).

## orthogonal-hybrid-orbital-axes: POAV2 axes in radial electronic response
Status: deferred

**Hypothesis:** Joint sigma/pi orthogonality improves the frozen orbital alignment for unequal bond angles.
**Outcome and uncertainty:**509 passes training but slightly loses validation cost; all molecules converge with valid energy and zero errors. No orbital-state diagnostics.
**Reason to revisit:** A separately justified full hybrid hopping model could couple the POAV2 orientation to the omitted s/p participation rather than only its normalized p axis.
**Next experiment:** Await that independently supported Hamiltonian; no angular-domain,hybrid-weight,response-strength or scope sweeps.

**Attempts:**
- Cycle509; candidate `cfb592608f7eff102e3cdcecff871e261ce01d1e`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision non_generalizable. [Implementation](ideas/orthogonal-hybrid-orbital-axes/cfb592608f7eff102e3cdcecff871e261ce01d1e/algo.py), [training](evaluation_results/cycle-509-train.json), [validation](evaluation_results/cycle-509-valid.json), [narrative](full_log.md#cycle-509-orthogonal-hybrid-orbital-axes).

## physical-rms-ratio-descent: ADADELTA in physical normal coordinates
Status: deferred

**Hypothesis:** Recent realized-update/gradient RMS supplies positive adaptive compliance without permanent gradient accumulation.
**Outcome and uncertainty:**510 has4 converged,464 limits,1 backend SCC error; all468 returned cases slower. Passive late force/motion remain large; accumulator traces unavailable.
**Reason to revisit:** A separately supported coupling-aware compliance estimator could retain the finite-memory benefit while representing rotating soft directions.
**Next experiment:** Await that formulation; no decay,epsilon,cap,floor or basis sweeps.

**Attempts:**
- Cycle510; candidate `375ef405d538152cf4eb2ff2adbad300b6885561`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision invalid. [Implementation](ideas/physical-rms-ratio-descent/375ef405d538152cf4eb2ff2adbad300b6885561/algo.py), [training](evaluation_results/cycle-510-train.json), [narrative](full_log.md#cycle-510-physical-rms-ratio-descent).

## strong-inverse-least-change: Full inverse secant in the current Hessian norm
Status: deferred

**Hypothesis:** A direct symmetric projection of compliance preserves unobserved response directions while enforcing the current secant.
**Outcome and uncertainty:**512 fully converges with valid energy/zero errors but loses cost;109 faster,211 slower,149 same. No retained matrix/domain-state traces.
**Reason to revisit:** Independent inverse-response error estimates could justify a compliance norm different from the uncertain current Hessian.
**Next experiment:** Await that evidence; no metric,positivity,blend,activation or scaling sweeps.

**Attempts:**
- Cycle512; candidate `fc6c284fa7ad7edffe27019624ac1ab91478461b`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/strong-inverse-least-change/fc6c284fa7ad7edffe27019624ac1ab91478461b/algo.py), [training](evaluation_results/cycle-512-train.json), [narrative](full_log.md#cycle-512-strong-inverse-least-change-curvature).

## minimum-residual-surrogate: Newton-MR search of the scalar GP
Status: deferred

**Hypothesis:** Signed surrogate curvature with gradient-merit globalization can improve stationary proposals without positive BFGS updates.
**Outcome and uncertainty:**513 fully converges with valid energy/zero errors but loses cost;19 faster,17 slower,433 same. No retained microsearch state distinguishes stationary-branch selection from derivative or model error.
**Reason to revisit:** Independent evidence of singular surrogate curvature and reliable curvature derivatives could justify minimum-residual search.
**Next experiment:** Await that evidence; no rank threshold,spacing,budget,merit or line-search sweeps.

**Attempts:**
- Cycle513; candidate `16a6e46f80a3d87733241846ce4987d076ce1f39`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/minimum-residual-surrogate/16a6e46f80a3d87733241846ce4987d076ce1f39/algo.py), [training](evaluation_results/cycle-513-train.json), [narrative](full_log.md#cycle-513-minimum-residual-newton-surrogate-search).

## global-start-surrogate: DIRECT initialization of surrogate refinement
Status: deferred

**Hypothesis:** Spatial exploration can find a better basin for the existing GP local minimizer.
**Outcome and uncertainty:**514 fully converges with valid energy/zero errors but loses cost;5 faster,10 slower,454 same. No retained search-state diagnostics.
**Reason to revisit:** Independent evidence that useful posterior basins are missed, together with calibration of their model error, could justify global initialization.
**Next experiment:** Await that evidence; no budget,box,epsilon,local-bias or seed sweeps.

**Attempts:**
- Cycle514; candidate `5a46d46ac3fa38922af748cc25217913cb833760`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/global-start-surrogate/5a46d46ac3fa38922af748cc25217913cb833760/algo.py), [training](evaluation_results/cycle-514-train.json), [narrative](full_log.md#cycle-514-global-start-surrogate-refinement).

## canonical-thermal-bond-response: Fixed-particle thermal electronic response
Status: deferred

**Hypothesis:** Canonical Fermi occupations regularize near-degenerate radial response through physical electronic entropy.
**Outcome and uncertainty:**515 saves one training call and passes; validation slightly loses cost (0 faster,1 slower,464 same). Both splits fully converge with valid energy/zero errors; no orbital diagnostics.
**Reason to revisit:** Independently calibrated spectral energies and occupations could distinguish physical thermal response from errors in the simple neutral hopping model.
**Next experiment:** Await that calibration; no temperature,hopping-scale,gap or scope sweeps.

**Attempts:**
- Cycle515; candidate `4c36771fcc8ac314e8bb20737558a863dc93b64a`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision non_generalizable. [Implementation](ideas/canonical-thermal-bond-response/4c36771fcc8ac314e8bb20737558a863dc93b64a/algo.py), [training](evaluation_results/cycle-515-train.json), [validation](evaluation_results/cycle-515-valid.json), [narrative](full_log.md#cycle-515-canonical-thermal-bond-response).

## physical-angle-adaptive-relaxation: AARE with a fixed chemical metric
Status: deferred

**Hypothesis:** Force-angle control retains useful inertia and retries severe overshoot with reduced speed.
**Outcome and uncertainty:**516 has259 converged,210 limits,no errors,energy failure and468 slower cases. Passive late force/motion remain large; no controller-state diagnosis.
**Reason to revisit:** An independently supported moving metric with compatible velocity and conjugate-direction transport could address stale conditioning.
**Next experiment:** Await that formulation; no time-step,angle,cap,CG-formula or metric-floor sweeps.

**Attempts:**
- Cycle516; candidate `15aad9ff5950a35b5b91f7da405f7be690ab2ff1`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision invalid. [Implementation](ideas/physical-angle-adaptive-relaxation/15aad9ff5950a35b5b91f7da405f7be690ab2ff1/algo.py), [training](evaluation_results/cycle-516-train.json), [narrative](full_log.md#cycle-516-physical-angle-adaptive-relaxation).

## learned-polynomial-flow: Polynomial vector dynamics with a scalar GP critic
Status: deferred

**Hypothesis:** Fitted gradient dynamics can forecast a useful endpoint beyond a local GP minimum.
**Outcome and uncertainty:**517 fully converges with valid energy/zero errors but loses cost;45 faster,200 slower,224 same. No retained learned-field/integration states.
**Reason to revisit:** Independently demonstrated low-dimensional force dynamics with an integrability or stability constraint could justify predictive flow.
**Next experiment:** Await that mechanism; no degree,fit rank,horizon,tolerances or RHS-budget sweeps.

**Attempts:**
- Cycle517; candidate `12c9b60fabb0729d5338c53d3c80a9e677038b42`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/learned-polynomial-flow/12c9b60fabb0729d5338c53d3c80a9e677038b42/algo.py), [training](evaluation_results/cycle-517-train.json), [narrative](full_log.md#cycle-517-learned-polynomial-relaxation-flow).

**544 reassessment:** A source-derived model-descent halfspace projection supplies a structural stability mechanism. All469converge with validenergy/zeroerrors but broad cost loss persists and exceeds517; {'slower': 228, 'faster': 47, 'same': 194}. Emptyfailure_details. This prerequisite is exhausted; await independently demonstrated prediction accuracy, without projection/fit/integration parameter sweeps.
- Cycle544; candidate `4bf90e747c38a00a9f65e40678ff3ab704edb24b`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/learned-polynomial-flow/4bf90e747c38a00a9f65e40678ff3ab704edb24b/algo.py), [training](evaluation_results/cycle-544-train.json), [narrative](full_log.md#cycle-544-descent-constrained-learned-polynomial-flow).

## curvature-adaptive-surrogate: Analytical curvature-based GP step lengths
Status: deferred

**Hypothesis:** Directional curvature can set useful inverse-BFGS step lengths within a short surrogate search.
**Outcome and uncertainty:**518 passes training but loses validation cost; all934 converge with valid energy and zero errors. No search-state diagnosis.
**Reason to revisit:** Independent evidence of reliable model directional curvature and a suitable local self-concordance scale could justify the analytical rule.
**Next experiment:** Await that evidence; no curvature floor,step scaling,spacing or iteration sweeps.

**Attempts:**
- Cycle518; candidate `1ec73657a8d2ea084d9cd5ed6037fb0ba82dbbf2`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision non_generalizable. [Implementation](ideas/curvature-adaptive-surrogate/1ec73657a8d2ea084d9cd5ed6037fb0ba82dbbf2/algo.py), [training](evaluation_results/cycle-518-train.json), [validation](evaluation_results/cycle-518-valid.json), [narrative](full_log.md#cycle-518-curvature-adaptive-surrogate-bfgs).

## relaxed-bond-normalization: Calibrate electronic bond compliance
Status: deferred

**Hypothesis:** Match empirical Badger stiffness to reciprocal diagonal compliance of the coupled electronic reference model.
**Outcome and uncertainty:**519 fully converges with valid energy/zero errors but loses cost;{'same': 450, 'faster': 10, 'slower': 9}. No electronic/curvature state diagnostics.
**Reason to revisit:** Independent calibration of local relaxed force constants, including the effect of redundant projection, could establish the intended physical stiffness meaning.
**Next experiment:** Await that calibration; no blend,normalization exponent,scope or fit-weight sweeps.

**Attempts:**
- Cycle519; candidate `2a967273b7ffa947ffe4230fa89fcae75e119527`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/relaxed-bond-normalization/2a967273b7ffa947ffe4230fa89fcae75e119527/algo.py), [training](evaluation_results/cycle-519-train.json), [narrative](full_log.md#cycle-519-relaxed-bond-stiffness-normalization).

## regularized-surrogate-extrapolation: Extrapolate the GP microiteration limit
Status: deferred

**Hypothesis:** Regularized minimal-polynomial extrapolation can improve the final BFGS proposal while retaining all model critics.
**Outcome and uncertainty:**520 fully converges with valid energy/zero errors, but gain is below1e-4;{'same': 468, 'faster': 1}. No microiterate diagnostics.
**Reason to revisit:** Independent evidence of approximately fixed linear dynamics in a stalled surrogate search could support a reliable extrapolated limit.
**Next experiment:** Await that evidence; no ridge,window,coefficient-bound or budget sweeps.

**Attempts:**
- Cycle520; candidate `2e1b15277ee222e4feafc73cc0bb20caaca58788`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/regularized-surrogate-extrapolation/2e1b15277ee222e4feafc73cc0bb20caaca58788/algo.py), [training](evaluation_results/cycle-520-train.json), [narrative](full_log.md#cycle-520-regularized-surrogate-sequence-extrapolation).

## physical-golden-ratio-relaxation: Adaptive averaged-anchor relaxation
Status: deferred

**Hypothesis:** Golden-ratio averaged positions and local gradient-change steps can stabilize one-call physical relaxation.
**Outcome and uncertainty:**521 has44 converged,425 limits,no errors,energy failure and468 slower cases. Passive late steps are small while forces remain large; controller state is absent.
**Reason to revisit:** A separately justified evolving metric or an explicit diagnosis of step-collapse under clipping could connect this controller to molecular conditioning.
**Next experiment:** Await that evidence; no phi,growth,initial-size,cap or metric-floor sweeps.

**Attempts:**
- Cycle521; candidate `9e4347b5f9563a2cf4ddbd06993dd673e8c51664`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision invalid. [Implementation](ideas/physical-golden-ratio-relaxation/9e4347b5f9563a2cf4ddbd06993dd673e8c51664/algo.py), [training](evaluation_results/cycle-521-train.json), [narrative](full_log.md#cycle-521-physical-adaptive-golden-ratio-relaxation).

## third-order-surrogate-search: Directional Chebyshev correction of GP Newton steps
Status: deferred

**Hypothesis:** Directional third curvature improves local surrogate steps without more physical calls.
**Outcome and uncertainty:**522 fully converges with valid energy/zero errors but loses cost;{'faster': 14, 'same': 441, 'slower': 14}. No derivative/search-state evidence.
**Reason to revisit:** Independently accurate higher derivatives and evidence that curvature change limits the GP search could justify this correction.
**Next experiment:** Await that evidence; no spacing,correction scale,line-search or budget sweeps.

**Attempts:**
- Cycle522; candidate `5feb267cd895a82a3ebb931bad7249e9a63f2fea`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/third-order-surrogate-search/5feb267cd895a82a3ebb931bad7249e9a63f2fea/algo.py), [training](evaluation_results/cycle-522-train.json), [narrative](full_log.md#cycle-522-third-order-chebyshev-surrogate-search).

## hoshino-curvature-adaptation: Directional adaptation of inverse-Broyden mixing
Status: deferred

**Hypothesis:** A curvature-ratio-dependent positive inverse blend could improve learned directions.
**Outcome and uncertainty:**523 fully converges with valid energy/zero errors but broad cost loss; {'slower': 165, 'faster': 44, 'same': 260}. No matrix-state evidence.
**Reason to revisit:** Independent evidence identifying overreaction to particular curvature mismatches could support a noise-aware secant treatment.
**Next experiment:** Await that evidence; no blend,activation,floor or scope sweeps.

**Attempts:**
- Cycle523; candidate `d09180f12c353756fc0c72577949df04101f8e4f`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/hoshino-curvature-adaptation/d09180f12c353756fc0c72577949df04101f8e4f/algo.py), [training](evaluation_results/cycle-523-train.json), [narrative](full_log.md#cycle-523-curvature-adaptive-hoshino-secant-update).

## covariant-orbital-bond-response: Full orbital geometry in the bond prior
Status: deferred

**Hypothesis:** Exact orbital derivatives and a consistent native connection could improve463's frozen-axis bond correlations.
**Outcome and uncertainty:**524 fully converges with valid energy/zero errors but loses cost; {'same': 445, 'faster': 8, 'slower': 16}. No curvature diagnostics.
**Reason to revisit:** An independently justified complete joint-coordinate prior could retain cross-type curvature lost by extracting the bond block alone. This needs consistent scale/fit identification first.
**Next experiment:** Await that joint formulation; no normalization,scope,blend or derivative-threshold sweeps.

**Attempts:**
- Cycle524; candidate `90822fae9b4be829b9118a9bf54b515c81a0a63e`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/covariant-orbital-bond-response/90822fae9b4be829b9118a9bf54b515c81a0a63e/algo.py), [training](evaluation_results/cycle-524-train.json), [narrative](full_log.md#cycle-524-covariant-orbital-geometry-bond-response).

**Revisit556:** Derive a joint-coordinate prior rather than extracting524's bond block. Set radial slopes from BadgerD so the frozen-axis independent-bond response matches463 analytically; source t0=2.5eV supplies angular/cross units. Native full electronic Hessian adds to diagonal mechanicalD, normalized to original marginals. A symmetric correlation square root partitions the complete prior into positive existing chemical blocks, preserving four shared fit parameters rather than adding unidentified cross-type coefficients. This supplies the deferred joint-model/scale/fit construction, not a diagnosis of524.

**Completed556:** Joint-coordinate construction also loses cost despite all469converging and validenergy,zeroerrors;23faster22slower424same. The deferred scale/fit construction is now tested. Revisit only with independent evidence identifying a systematic joint-curvature error; no magnitude,normalization,scope orfit sweep.
- Cycle556; candidate `356fcea0186d75c0f53e440532bee633518e06fa`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/covariant-orbital-bond-response/356fcea0186d75c0f53e440532bee633518e06fa/algo.py), [training](evaluation_results/cycle-556-train.json), [narrative](full_log.md#cycle-556-joint-coordinate-orbital-response-physical-prior).

## physical-gradient-norm-acceleration: Horizon-based OGM-G physical relaxation
Status: deferred

**Hypothesis:** Direct gradient-norm acceleration could improve physical force convergence.
**Outcome and uncertainty:**525 has298 converged,171 limits,no errors,energy failure and468 slower cases. Passive trace shows late continued relaxation; no controller state.
**Reason to revisit:** Independently supported curvature estimation or evolving metric could connect finite-horizon acceleration to changing molecular conditioning.
**Next experiment:** Await that evidence; no horizon,curvature safety,cap or metric-floor sweeps.

**Attempts:**
- Cycle525; candidate `b30d17ba573b1748293e27361d9e909371253f57`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision invalid. [Implementation](ideas/physical-gradient-norm-acceleration/b30d17ba573b1748293e27361d9e909371253f57/algo.py), [training](evaluation_results/cycle-525-train.json), [narrative](full_log.md#cycle-525-physical-optimized-gradient-norm-acceleration).

## backtracking-gradient-norm-surrogate: OBL-G cheap gradient-norm search
Status: deferred

**Hypothesis:** Trial-checked gradient-norm acceleration could improve GP endpoints without more paid forces.
**Outcome and uncertainty:**526 fully converges with valid energy/zero errors but loses cost; {'faster': 18, 'slower': 16, 'same': 435}. No microsearch diagnostics.
**Reason to revisit:** Independent evidence of convex, acceleration-limited microdynamics could establish the conditions in which the source recurrence is useful.
**Next experiment:** Await that evidence; no growth,budget,horizon or confidence sweeps.

**Attempts:**
- Cycle526; candidate `21902c553559f286cd8e67dc123319eb7251f5ad`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/backtracking-gradient-norm-surrogate/21902c553559f286cd8e67dc123319eb7251f5ad/algo.py), [training](evaluation_results/cycle-526-train.json), [narrative](full_log.md#cycle-526-backtracking-gradient-norm-surrogate-acceleration).

## longitudinal-orbital-hopping: Longitudinal p-orbital coupling
Status: deferred

**Hypothesis:** The missing pp-sigma projection improves bond correlations in tilted orbital geometry.
**Outcome and uncertainty:**527 fully converges with valid energy/zero errors, but gain is below1e-4; {'same': 466, 'faster': 2, 'slower': 1}. No electronic-state diagnostics.
**Reason to revisit:** A consistent independently calibrated multi-orbital Hamiltonian could resolve the hybrid radial law and omitted sigma-band response together.
**Next experiment:** Await that formulation; no ratio,projection,scope or radial-law sweeps.

**Attempts:**
- Cycle527; candidate `7946bb0cebab993b147170e3ed54a09af64e3491`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/longitudinal-orbital-hopping/7946bb0cebab993b147170e3ed54a09af64e3491/algo.py), [training](evaluation_results/cycle-527-train.json), [narrative](full_log.md#cycle-527-longitudinal-p-orbital-hopping-in-bond-response).

**542 reassessment:** Full C/H valence Hamiltonian with independently sourced onsite/hopping parameters and locally balanced r^-4 repulsion supplies527's prerequisite. All469converge with validenergy/zeroerrors, but aggregate cost worsens; {'same': 442, 'faster': 18, 'slower': 9}. No orbital-state diagnostics. Await independent charge/repulsion/radial-response calibration; no parameter/scope sweeps.
- Cycle542; candidate `bc7dd4857c24525f00ec5e7fecd23c35faf748ce`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/longitudinal-orbital-hopping/bc7dd4857c24525f00ec5e7fecd23c35faf748ce/algo.py), [training](evaluation_results/cycle-542-train.json), [narrative](full_log.md#cycle-542-full-valence-bond-response-prior).

## directional-cubic-surrogate: Directional cubic gradient microsearch
Status: deferred

**Hypothesis:** A directional curvature certificate selects useful GP gradient steps without full micro-Hessian learning.
**Outcome and uncertainty:**528 fully converges with valid energy/zero errors but loses cost; {'slower': 8, 'faster': 12, 'same': 449}. No microsearch diagnostics.
**Reason to revisit:** Independent evidence of useful directional curvature and cubic-regime transitions could support a model solver matched to these conditions.
**Next experiment:** Await that evidence; no alpha,growth,spacing,initialH,budget or confidence sweeps.

**Attempts:**
- Cycle528; candidate `5bd3a23e35db5d9767a619e530b28133138bb8c5`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/directional-cubic-surrogate/5bd3a23e35db5d9767a619e530b28133138bb8c5/algo.py), [training](evaluation_results/cycle-528-train.json), [narrative](full_log.md#cycle-528-directional-cubic-surrogate-descent).

## positive-diagonal-physical-learning: Positive quasi-Cauchy mode stiffness
Status: deferred

**Hypothesis:** Square-root least-change mode stiffness learning improves physical conditioning without full secant interpolation.
**Outcome and uncertainty:**529 has273 converged,196 limits,zero errors,energy failure and broad cost loss; {'slower': 468, 'faster': 1}. Passive late force/energy oscillation lacks metric and line-search states.
**Reason to revisit:** Independent evidence of nearly fixed modal eigenvectors or diagnosed line-search waste could separate useful stiffness adaptation from the controller cost.
**Next experiment:** Await that evidence; no line,cap,root,metric-floor or activation sweeps.

**Attempts:**
- Cycle529; candidate `597b33afa899278762bb07260235edc7f4fbc192`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision invalid. [Implementation](ideas/positive-diagonal-physical-learning/597b33afa899278762bb07260235edc7f4fbc192/algo.py), [training](evaluation_results/cycle-529-train.json), [narrative](full_log.md#cycle-529-positive-diagonal-physical-curvature-learning).

## relative-hessian-entropy: Relative-Hessian matrix entropy projection
Status: deferred

**Hypothesis:** Entropy regularity of relative curvature reduces unnecessary changes while enforcing measured secants.
**Outcome and uncertainty:**530 fully converges with valid energy and zero errors, but loses cost; {'slower': 204, 'same': 230, 'faster': 35}. No curvature diagnostics.
**Reason to revisit:** Independently supported curvature-error geometry could distinguish useful entropy constraints from generic exact secant interpolation.
**Next experiment:** Await that evidence; no blend,activation,root,floor or scope sweeps.

**Attempts:**
- Cycle530; candidate `553f34cd1d00d3aeac7bf7bad2e59377c4525266`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/relative-hessian-entropy/553f34cd1d00d3aeac7bf7bad2e59377c4525266/algo.py), [training](evaluation_results/cycle-530-train.json), [narrative](full_log.md#cycle-530-relative-hessian-entropy-secant-projection).

## gradient-normalized-surrogate: Approximate Newton GP steps with a function certificate
Status: deferred

**Hypothesis:** Gradient-normalized regularization and function decrease relative to endpoint gradient improve surrogate endpoints.
**Outcome and uncertainty:**531 passes train (6fast2slow461same) but loses validation cost ({'same': 451, 'slower': 9, 'faster': 5}); both fully converge with valid energy and no errors. No microsearch diagnostics.
**Reason to revisit:** Independent evidence of approximation error or retained search states could show where this certificate improves physical proposal quality.
**Next experiment:** Await that evidence; no gamma,growth,spacing,budget,PSD-map or confidence sweeps.

**Attempts:**
- Cycle531; candidate `a8874a6ec7e1f41b3300a28458a20b25906b0009`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision non_generalizable. [Implementation](ideas/gradient-normalized-surrogate/a8874a6ec7e1f41b3300a28458a20b25906b0009/algo.py), [training](evaluation_results/cycle-531-train.json), [validation](evaluation_results/cycle-531-valid.json), [narrative](full_log.md#cycle-531-gradient-normalized-surrogate-newton-search).

## stabilized-typei-surrogate: Stabilized type-I GP Jacobian learning
Status: deferred

**Hypothesis:** Independent multisecant Jacobian directions improve model stationarity search.
**Outcome and uncertainty:**532 fully converges with valid energy/zero errors, but loses cost; {'slower': 6, 'same': 455, 'faster': 8}. Empty passive diagnostics.
**Reason to revisit:** Independent evidence of solver stagnation or near-dependent microsteps could distinguish a useful stabilization from model/proposal error.
**Next experiment:** Await that evidence; no history,theta,tau,line,budget or confidence sweeps.

**Attempts:**
- Cycle532; candidate `261e86ee87becaaaa56dc7dfb67645cafe40e864`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/stabilized-typei-surrogate/261e86ee87becaaaa56dc7dfb67645cafe40e864/algo.py), [training](evaluation_results/cycle-532-train.json), [narrative](full_log.md#cycle-532-stabilized-type-i-multisecant-surrogate-search).

## vector-auxiliary-surrogate: Positive-objective vector auxiliary GP dynamics
Status: deferred

**Hypothesis:** Coordinatewise auxiliary energies regulate local physical modes independently without a guessed objective lower bound.
**Outcome and uncertainty:**534 fully converges with valid energy/zero errors but loses cost; {'slower': 29, 'faster': 28, 'same': 412}. Empty failure_details, no inner-state evidence.
**Reason to revisit:** Independent evidence of auxiliary collapse or exponential stiffness could separate a useful vector dissipation mechanism from the positive transform.
**Next experiment:** Await that evidence; no exponential-scale,dt,psi,budget or admission sweeps.

**Attempts:**
- Cycle534; candidate `035f416ccefc8f6358e06d2351517c18f1db9d5a`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/vector-auxiliary-surrogate/035f416ccefc8f6358e06d2351517c18f1db9d5a/algo.py), [training](evaluation_results/cycle-534-train.json), [narrative](full_log.md#cycle-534-positive-objective-vector-auxiliary-surrogate-search).

## articulated-bond-vector-chart: Explicit rotation-carrying molecular chart
Status: deferred

**Hypothesis:** Hierarchical local bond vectors with complete frame derivatives and full physical metric matching improve finite collective motion.
**Outcome and uncertainty:**535 invalid:466converged,2limits,1unrepresentable-step error;5fast461slow2same among returned cases. Error trace shows frozen geometry at force.0088Eh/Bohr but no controller/chart states.
**Reason to revisit:** Independent evidence of finite-chart curvature conditioning or identified domain/collision obstruction could separate chart geometry from controller failures. Analytic recursive Jacobian and metric matching are reusable.
**Next experiment:** Require that evidence; broad cost loss does not support radius/floor/tree/root/domain/scope tuning or automatic retry of one error.

**Attempts:**
- Cycle535; candidate `1c823ae37cf6eeb57ff7cc9f79897193a6ad7aaf`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision invalid. [Implementation](ideas/articulated-bond-vector-chart/1c823ae37cf6eeb57ff7cc9f79897193a6ad7aaf/algo.py), [training](evaluation_results/cycle-535-train.json), [narrative](full_log.md#cycle-535-articulated-bond-vector-coordinate-optimizer), [passive bundle](diagnostics/a7bab9b7d0f64369a83ebd694c456e88/905bf1bd6b1f2f2e1e9c394ab642756ad1a3aeb61a58dd9cc4302b66f36d47c9.json).

## inertial-corrected-surrogate: FISC search inside the physical GP
Status: deferred

**Hypothesis:** Descent-preserving inertial direction correction improves model endpoints without physical dynamics or estimated Hessians.
**Outcome and uncertainty:**536 had469missing-initial NameErrors. Repair537 fully converges with zeroerrors and passes train; validation cost improves but energy fails. One higher-energy converged outcome accounts for more than the full cost gain; other cases collectively lose cost. Empty passive diagnostics.
**Reason to revisit:** Independently diagnosed basin-selection or surrogate endpoint error could distinguish a useful direction correction from invalid apparent speedup. The binding repair is resolved.
**Next experiment:** Await that evidence; no r,age,line,budget,confidence or case-specific changes. The one bounded implementation repair is complete.

**Attempts:**
- Cycle536; candidate `2e919a3d6d90df9bc30bd8384bc7d1f0f14dcb10`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision invalid. [Implementation](ideas/inertial-corrected-surrogate/2e919a3d6d90df9bc30bd8384bc7d1f0f14dcb10/algo.py), [training](evaluation_results/cycle-536-train.json), [narrative](full_log.md#cycle-536-inertial-direction-corrected-surrogate-search).
- Cycle537; candidate `77cdea7611d9d6b99d40b9cbd59346d45e51acaf`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision non_generalizable. [Implementation](ideas/inertial-corrected-surrogate/77cdea7611d9d6b99d40b9cbd59346d45e51acaf/algo.py), [training](evaluation_results/cycle-537-train.json), [validation](evaluation_results/cycle-537-valid.json), [narrative](full_log.md#cycle-537-restore-fisc-proposal-baseline-binding).

## anisotropic-surrogate-descent: Cosh-reference nonlinear dual GP preconditioning
Status: deferred

**Hypothesis:** Symmetric nonlinear gradient compression and its anisotropic decrease certificate improve finite model search.
**Outcome and uncertainty:**539 fully converges with validenergy and noerrors but loses cost; {'slower': 11, 'faster': 10, 'same': 448}. Emptyfailure_details.
**Reason to revisit:** Independently observed component-gradient scales or majorizer failures could distinguish useful compression from model endpoint errors.
**Next experiment:** Await that evidence; no reference-scale,basis,rate,budget or critic sweep.

**Attempts:**
- Cycle539; candidate `dc951725d0fc75e83e7937aa9ce68826802872d9`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/anisotropic-surrogate-descent/dc951725d0fc75e83e7937aa9ce68826802872d9/algo.py), [training](evaluation_results/cycle-539-train.json), [narrative](full_log.md#cycle-539-cosh-reference-anisotropic-surrogate-descent).

## greedy-coordinate-surrogate: Greedy local Newton coordinate updates in the GP
Status: deferred

**Hypothesis:** Concentrating each model update on the coordinate with greatest predicted decrease avoids unhelpful nonlinear cross-mode motion.
**Outcome and uncertainty:**540 fully converges with validenergy and noerrors but loses cost; {'slower': 15, 'faster': 12, 'same': 442}. Emptyfailure_details.
**Reason to revisit:** Independent model-coupling or inner-iteration error evidence could identify a physically useful decomposition.
**Next experiment:** Await that evidence; no block-size,selection,spacing,floor,budget or critic sweep.

**Attempts:**
- Cycle540; candidate `a4443c649ca742d61f45b9cd2a30b34f33d2ed32`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/greedy-coordinate-surrogate/a4443c649ca742d61f45b9cd2a30b34f33d2ed32/algo.py), [training](evaluation_results/cycle-540-train.json), [narrative](full_log.md#cycle-540-greedy-coordinate-newton-surrogate-search).

## nonorthogonal-orbital-response: Generalized orbital eigenproblem for bond response
Status: deferred

**Hypothesis:** Neighboring orbital overlap contributes useful occupied-state curvature missing from an orthogonal pi model.
**Outcome and uncertainty:**541 fully converges with validenergy and noerrors but loses cost; {'same': 445, 'faster': 15, 'slower': 9}. No orbital-state diagnostics.
**Reason to revisit:** Independently calibrated joint radial hopping and overlap laws could address the tested proportional-decay assumption. The degeneracy-safe spectral Hessian is reusable.
**Next experiment:** Await that physical evidence; no overlap,radialscale,occupation,scope or threshold sweep.

**Attempts:**
- Cycle541; candidate `1c10f2f9e002dcfb0da315ca1ef0ed2640a82a90`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/nonorthogonal-orbital-response/1c10f2f9e002dcfb0da315ca1ef0ed2640a82a90/algo.py), [training](evaluation_results/cycle-541-train.json), [narrative](full_log.md#cycle-541-nonorthogonal-orbital-bond-response).

## forward-backward-envelope-search: PANOC+ model proposal search
Status: deferred

**Hypothesis:** Envelope-globalized quasi-Newton steps and a gradient-corrected endpoint improve finite surrogate search.
**Outcome and uncertainty:**543 fully converges with validenergy/zeroerrors but loses cost; {'same': 460, 'slower': 7, 'faster': 2}. No passive inner-state diagnostics.
**Reason to revisit:** Independent evidence of envelope or surrogate endpoint error could identify useful proximal globalization.
**Next experiment:** Await that evidence; no alpha,beta,gamma,bound,budget or critic sweep.

**Attempts:**
- Cycle543; candidate `7aa038abed7d01c81428112bc930fcdf2549344b`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/forward-backward-envelope-search/7aa038abed7d01c81428112bc930fcdf2549344b/algo.py), [training](evaluation_results/cycle-543-train.json), [narrative](full_log.md#cycle-543-forward-backward-envelope-surrogate-search).

## physical-coin-betting: Wealth-adaptive physical dual averaging
Status: deferred

**Hypothesis:** Signed-gradient accumulation and coordinate wealth adapt motion in an initial physical metric without curvature learning.
**Outcome and uncertainty:**545 fails energy with 333 limits and 468 slower cases; a downloaded trace confirms finite oscillation, but does not identify its wealth or metric cause.
**Reason to revisit:** An independently supported metric/noise model could explain when wealth feedback is useful; the exact optimizer remains reusable.
**Next experiment:** Await that evidence. No alpha, wealth, cap, floor or restart sweep on the broad failure.

**Attempts:**
- Cycle545; candidate `9c23d26f401bb43244d9e0c09b2dd9d41741719e`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision invalid. [Implementation](ideas/physical-coin-betting/9c23d26f401bb43244d9e0c09b2dd9d41741719e/algo.py), [training](evaluation_results/cycle-545-train.json), [narrative](full_log.md#cycle-545-physical-coin-betting-optimizer).

## certified-nonconvex-surrogate: PF-AGD model search
Status: deferred

**Hypothesis:** Regularized acceleration and explicit nonconvexity certificates improve finite GP search beyond BFGS.
**Outcome and uncertainty:**546 converges all with zero errors but fails energy and loses cost;21 faster,16 slower,432same. Empty passive diagnostics cannot identify certificate or endpoint error.
**Reason to revisit:** Independent surrogate-certification evidence could justify this explicit negative-curvature mechanism; complete bounded implementation preserved.
**Next experiment:** Await that evidence, without epsilon,M,L,backtracking,budget or certificate-frequency sweeps.

**Attempts:**
- Cycle546; candidate `811aebdc0881e27bccc8df59cdbbd6dd52d86c25`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision invalid. [Implementation](ideas/certified-nonconvex-surrogate/811aebdc0881e27bccc8df59cdbbd6dd52d86c25/algo.py), [training](evaluation_results/cycle-546-train.json), [narrative](full_log.md#cycle-546-certified-nonconvex-surrogate-acceleration).

## singular-vector-surrogate: W4SV gradient-root dynamics
Status: deferred

**Hypothesis:** Delayed momentum in changing singular-vector frames traverses weak model curvature effectively.
**Outcome and uncertainty:**547 fully converges with validenergy and zeroerrors but loses cost;23faster30slower416same. No passive model-state diagnostics.
**Reason to revisit:** Independently demonstrated singularity-limited model search could justify the frame-based dynamics.
**Next experiment:** Await that evidence; no timestep,threshold,budget,matching or critic sweep.

**Attempts:**
- Cycle547; candidate `a5c3bc27e591c55dce8c9c55de037d58510a0960`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/singular-vector-surrogate/a5c3bc27e591c55dce8c9c55de037d58510a0960/algo.py), [training](evaluation_results/cycle-547-train.json), [narrative](full_log.md#cycle-547-singular-vector-surrogate-dynamics).

## quartic-cubic-surrogate: AR3 model search
Status: deferred

**Hypothesis:** Cubic curvature coupling with quartic regularization improves finite surrogate search.
**Outcome and uncertainty:**548 passes training but loses validation cost; all converge with valid energy and zero errors. Empty passive diagnostics do not identify polynomial-subsolver behavior.
**Reason to revisit:** Independent evidence of a cubic-subproblem or derivative accuracy failure could justify a bounded repair of this complete higher-order mechanism.
**Next experiment:** Await that evidence; no regularization,finite-difference,solver or budget sweep.

**Attempts:**
- Cycle548; candidate `8afbd18cc2499b8b0aa70257769ec1a9920d2f6e`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision non_generalizable. [Implementation](ideas/quartic-cubic-surrogate/8afbd18cc2499b8b0aa70257769ec1a9920d2f6e/algo.py), [training](evaluation_results/cycle-548-train.json), [validation](evaluation_results/cycle-548-valid.json), [narrative](full_log.md#cycle-548-quartic-regularized-cubic-surrogate-models).

## space-dilation-surrogate: Gradient-difference metric contraction
Status: deferred

**Hypothesis:** Crossing model valleys and contracting their transformed gradient differences improves finite model search.
**Outcome and uncertainty:**550 loses cost with28faster42slower399same; fullconvergence,validenergy,zeroerrors. No passive inner-state diagnostics.
**Reason to revisit:** Independent evidence of a transverse model-search error could justify this reusable non-secant metric mechanism.
**Next experiment:** Await that evidence; no scale,contraction,growth,budget orcritic sweep.

**Attempts:**
- Cycle550; candidate `85381d0d43a57f1418f63c1ea3fad59a17a859be`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/space-dilation-surrogate/85381d0d43a57f1418f63c1ea3fad59a17a859be/algo.py), [training](evaluation_results/cycle-550-train.json), [narrative](full_log.md#cycle-550-gradient-difference-space-dilation-in-surrogate-search).

## physical-auto-conditioned: Running energy-curvature bound for physical descent
Status: deferred

**Hypothesis:** Direct energy linearization errors provide a stable upper-curvature step rule in an initial chemical metric.
**Outcome and uncertainty:**551has381limits and failsenergy,1faster468slowerzeroerrors. One retained paliperidonepalmitate trace has passing force/displacement criteria but energy change slightly above threshold at200calls; slow finite progress in this case does not diagnose the broad curvature-state mechanism.
**Reason to revisit:** Independently measured moving-metric or curvature-memory mismatch could motivate a different nonconvex coarse solver; the complete implementation is preserved.
**Next experiment:** Await that evidence; no alpha,L0,cap,floor or memory sweep on broadloss.

**Attempts:**
- Cycle551; candidate `f6ab5b13223ed7bbe1e444e113c6c3b2cbd6a90a`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision invalid. [Implementation](ideas/physical-auto-conditioned/f6ab5b13223ed7bbe1e444e113c6c3b2cbd6a90a/algo.py), [training](evaluation_results/cycle-551-train.json), [narrative](full_log.md#cycle-551-auto-conditioned-physical-gradient-descent).

## physical-matrix-polar: Adaptive Muon in symmetric chemical coordinates
Status: deferred

**Hypothesis:** Matrix-polar directions balance collective spatial response without an arbitrary eigenvector-coordinate reshape.
**Outcome and uncertainty:**554has469limits,zeroerrorsandenergyfailure;allslower. The incomplete manifest provides a pyridine/pyrrole trace with finite oscillatory final motion (.248Bohr) and forces(.006–.007Hartree/Bohr), but no matrix/direction states to isolate the cause.
**Reason to revisit:** Independent evidence linking matrix-polar direction balancing to an evolving chemical metric could justify a different complete mechanism; symmetric tensor-preserving whitening is reusable.
**Next experiment:** Await that evidence; no eta,gamma,rank,cap,floor or momentum sweep.

**Attempts:**
- Cycle554; candidate `94038dc63f9f14d27d125f0530d9329a26988dee`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision invalid. [Implementation](ideas/physical-matrix-polar/94038dc63f9f14d27d125f0530d9329a26988dee/algo.py), [training](evaluation_results/cycle-554-train.json), [narrative](full_log.md#cycle-554-adaptive-matrix-polar-physical-descent).

**Cycle578 reassessment:** Publishedpreconditioned-norm framework suppliesevolvingcurvature link. BFGScurvature plusquadratic-optimalspectralray/paidArmijo yields455converged14limits,validenergy0errors,but466slower/2faster/1same. Passive trace shows continuedrelaxationandtrialwaste withoutcurvature/rank/acceptance diagnosis. Requiresindependently supported control ofmatrixmode response orsuchdiagnosis; no rank,cap,floor,raylength,Armijo/blendsweep.
- Cycle578; candidate `3bdf0a55eb7e1329bd3b029a96e0d0ffc1705d98`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/physical-matrix-polar/3bdf0a55eb7e1329bd3b029a96e0d0ffc1705d98/algo.py), [training](evaluation_results/cycle-578-train.json), [trace](diagnostics/c978202f51a942ba8c351cf3a7cd439f/be07ece8b81c59fb3b5525d5f2ab16b00e801c22dd080da2018b7aa5508e7a32.json), [narrative](full_log.md#cycle-578-curvature-scaled-spectral-physical-descent).

## nitrogen-umbrella-double-well: Reflection-symmetric nitrogen height mean
Status: deferred

**Hypothesis:** A physical quartic inversion remainder explains finite nitrogen displacement without replacing current learned quadraticcurvature.
**Outcome and uncertainty:**558 fullyconverges withvalidenergy/zeroerrors butnear-neutralcostloss;{'faster': 12, 'same': 449, 'slower': 8}. No mean-state diagnostic; chemicaltyping,wellheight andlinearchart remainapproximate.
**Reason to revisit:** Independent inversionresponse orbond-order evidence could establish a coupledheight model beyondthe current frozenlinearprojection.
**Next experiment:** Await such evidence; no atomscope,wellheight,stiffness ordomain sweep.
**Attempts:**
- Cycle558; candidate `2780cfaff8e8fdf1f0098e93342c014c6b721674`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/nitrogen-umbrella-double-well/2780cfaff8e8fdf1f0098e93342c014c6b721674/algo.py), [training](evaluation_results/cycle-558-train.json), [narrative](full_log.md#cycle-558-nitrogen-umbrella-double-well-surrogate-mean).

## physical-forward-cubic: Paid orthogonal curvature with explicit age correction
Status: deferred

**Hypothesis:** An orthogonally sampled physicalcurvature model withknownsamplinggeometry canrepayextraoraclecalls.
**Outcome and uncertainty:**559invalidenergy,360converged109limitszeroerrors,4fast465slow. Manifesthas11availableartifacts. Selected[trace](diagnostics/6051182a2e474c1e838648d0ccca7a8f/0737d21641fc2b593e5b5bb2a4b7c2fc1b11fd3ed70bf6790037229365e9bc3f.json) has8frames/192omittedand200scalars;lateforce/displacementabovecriteria,finiteenergyalternation. No curvature/controllerstateorcallroles tolocalizethecause.
**Reason to revisit:** Independent evidence distinguishing finite-difference noise from age-bound conservatism couldsupport a new economicalobservation model.
**Next experiment:** Await such evidence; no h,M,memory,cap,metricfloor orcontroller sweep.
**Attempts:**
- Cycle559; candidate `b4e1160191b2a9376c4d1797a6254e7bbcb75b19`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision invalid. [Implementation](ideas/physical-forward-cubic/b4e1160191b2a9376c4d1797a6254e7bbcb75b19/algo.py), [training](evaluation_results/cycle-559-train.json), [manifest](diagnostics/6051182a2e474c1e838648d0ccca7a8f/manifest.json), [narrative](full_log.md#cycle-559-physical-cubic-optimization-with-orthogonal-forward-curvature).

## residual-adaptive-anderson-surrogate: Regularized type-II fixed-point mixing
Status: deferred

**Hypothesis:** Adapt residual regularization using actual/predicted nonmonotone reduction to improve finite GP stationarity search.
**Outcome and uncertainty:**560 converges all469 with validenergy and zeroerrors but losescost. Emptyfailure_details; full GP contraction is unestablished.
**Reason to revisit:** A derived contraction-controlled map or observed residual/model mismatch could justify a new formulation.
**Next experiment:** Require that prerequisite before another map or regularization change; no step,history,threshold orbudget sweep. Compare fixed-map and derived-map searches under full gates.

**Attempts:**
- Cycle560; candidate `c73734563a668fc6bb3672a5425eb40c8d2902cf`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/residual-adaptive-anderson-surrogate/c73734563a668fc6bb3672a5425eb40c8d2902cf/algo.py), [train](evaluation_results/cycle-560-train.json), [narrative](full_log.md#cycle-560-residual-adaptive-nonmonotone-anderson-surrogate-search).

## nonmonotone-spectral-surrogate: Spectral secants with GLL energy search
Status: deferred

**Hypothesis:** Scalar GP secantcurvature andnonmonotoneenergyglobalization can improve finite proposals without changing physicalHessian memory.
**Outcome and uncertainty:**561all469converge,validenergy,zeroerrors,butcostloss. No microsearchdiagnostics.
**Reason to revisit:** A supported model/query geometry or diagnosed directionalcurvature failure could justify a scalarsearch variant.
**Next experiment:** Awaitthat evidence; no BBvariant,clip,history,line-search orbudget sweep.

**Attempts:**
- Cycle561; candidate `25115560c1490760211f5233548433950324e1bb`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/nonmonotone-spectral-surrogate/25115560c1490760211f5233548433950324e1bb/algo.py), [train](evaluation_results/cycle-561-train.json), [narrative](full_log.md#cycle-561-nonmonotone-spectral-gradient-surrogate-search).

## powell-direction-surrogate: Conjugate-direction scalar energy search
Status: deferred

**Hypothesis:** Finite line minimizations canbuildusefulGPdirections missedbygradient-driven searches.
**Outcome and uncertainty:**562all469converge,validenergy,zeroerrors,butcostloss. No search-leveldiagnostics.
**Reason to revisit:** An independentlysupported finiteenergylandscape or diagnosedlinebracketing defect couldjustifyreusingtheenergy-onlysearch.
**Next experiment:** Requirethatprerequisite; no directions,tolerances,budgets,bounds orpolishing sweeps.

**Attempts:**
- Cycle562; candidate `c4f2591608293156b3caa1d71ad0837677011160`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/powell-direction-surrogate/c4f2591608293156b3caa1d71ad0837677011160/algo.py), [train](evaluation_results/cycle-562-train.json), [narrative](full_log.md#cycle-562-powell-direction-set-surrogate-energy-search).

## physical-kinetic-friction: Auxiliary kinetic damping in a chemical metric
Status: deferred

**Hypothesis:** Kinetic-driven auxiliaryfriction canstabilizemomentumandacceleratephysicalrelaxation.
**Outcome and uncertainty:**563fiveconverged,463limits,oneSCCerror; all468returnedcasesslower. Pomalidomidetracehaslargefiniteoscillations;SCCgeometryunavailable. Friction/momentumstatesunknown.
**Reason to revisit:** Independentlyimproved anisotropicdynamicmetric or retaineddynamicalstate identifyingamissingcoupling couldmake thisauxiliarymechanismuseful.
**Next experiment:** Awaitthatprerequisite; no friction,time,cap,floor orintegratorsweep. Fullgatesrequired.

**Attempts:**
- Cycle563; candidate `dc7bb34c16f3b5d90bee043b7ceae7d69ac8170f`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision invalid. [Implementation](ideas/physical-kinetic-friction/dc7bb34c16f3b5d90bee043b7ceae7d69ac8170f/algo.py), [train](evaluation_results/cycle-563-train.json), [manifest](diagnostics/42c7f1a3452b4a8e8469fe62f8c3b642/manifest.json), [narrative](full_log.md#cycle-563-physical-kinetic-friction-adaptive-descent).

## gradient-certificate-surrogate: Trial-gradient certificate for regularized Newton proposals
Status: deferred

**Hypothesis:** An implicit trial-gradient certificate can adapt regularization to finite GP curvature and improve proposal quality.
**Outcome and uncertainty:**564 passes training but loses validation cost; every case converges with valid energy and zero errors. Empty passive failure details do not identify a microsearch defect.
**Reason to revisit:** A derived nonconvex certificate or retained model-state evidence could distinguish regularization mismatch from ordinary variation in proposal quality.
**Next experiment:** Require that evidence before revision; no exponent, H0, spacing, positivity, budget or threshold sweep.

**Attempts:**
- Cycle564; candidate `a1bcde4f3774bc5c69e4a311a639f65cc1500eb9`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision non_generalizable. [Implementation](ideas/gradient-certificate-surrogate/a1bcde4f3774bc5c69e4a311a639f65cc1500eb9/algo.py), [training](evaluation_results/cycle-564-train.json), [validation](evaluation_results/cycle-564-valid.json), [narrative](full_log.md#cycle-564-trial-gradient-certified-surrogate-newton-search).

## cubic-metric-surrogate: BFGS-metric cubic damping with energy certification
Status: deferred

**Hypothesis:** A metric-based cubic decrease rule can improve finite GP proposals with learned BFGS directions.
**Outcome and uncertainty:**565all469converge, validenergy, zeroerrors butcostloss; {'faster': 4, 'slower': 7, 'same': 458}. Emptyfailure_details; no microsearch-state diagnosis.
**Reason to revisit:** Independently derived model normalization or observed curvature-inexactness evidence could support a different calibrated certificate.
**Next experiment:** Await that evidence; no L,alpha,multiplier,mode,budget or metric sweep.

**Attempts:**
- Cycle565; candidate `d9b61a3639316651f09c69f61823b4f860a06fb5`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/cubic-metric-surrogate/d9b61a3639316651f09c69f61823b4f860a06fb5/algo.py), [training](evaluation_results/cycle-565-train.json), [narrative](full_log.md#cycle-565-cubic-metric-damped-surrogate-bfgs).

## three-body-dispersion-mean: Angular triple dispersion beyond the local quadratic
Status: deferred

**Hypothesis:** A consistent finite triangle model can capture collective nonlinear response absentfrom pairwise wells.
**Outcome and uncertainty:**566all469converge withvalidenergy/zeroerrors butcostloss;{'faster': 13, 'slower': 12, 'same': 444}. No passive mean-state diagnostics. D2atomicC6/radii in anATM-shaped prior remainapproximate.
**Reason to revisit:** Independent environment-dependent three-body coefficients and damping calibratedto a consistentgeometry couldsupport a lessapproximate collective model.
**Next experiment:** Awaitthatphysicalevidence; no amplitude,radius,C9mixing,cutoff,scope orspacing sweeps.

**Attempts:**
- Cycle566; candidate `47edc2d27f15261a37fb7f13a442001ac7b38d23`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/three-body-dispersion-mean/47edc2d27f15261a37fb7f13a442001ac7b38d23/algo.py), [training](evaluation_results/cycle-566-train.json), [narrative](full_log.md#cycle-566-three-body-dispersion-remainder-in-the-surrogate-mean).

## nonlinear-coordinate-preconditioning: Implicitly coupled scalar minimization maps
Status: deferred

**Hypothesis:** Solving local nonlinearcoordination before a coupledNewton correction can reduce finite GP modelsearch error.
**Outcome and uncertainty:**567all469converge withvalidenergy/zeroerrors butcostloss;{'faster': 7, 'slower': 8, 'same': 454}. No innersearch diagnostics distinguish localinexactness,querybudget orproposalquality.
**Reason to revisit:** Independent evidence of a nonlinear blockstructure or observed subproblem-solver defect couldjustify a supported nonlinearpreconditioner.
**Next experiment:** Awaitthat evidence; no block,subsolver,tolerance,spacing,merit orbudget sweep.

**Attempts:**
- Cycle567; candidate `84516117de2620ed1fd58c3510623c51c89df23c`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/nonlinear-coordinate-preconditioning/84516117de2620ed1fd58c3510623c51c89df23c/algo.py), [training](evaluation_results/cycle-567-train.json), [narrative](full_log.md#cycle-567-nonlinear-coordinate-preconditioning-of-surrogate-newton).

## gaussian-multipole-mean: Joint relaxed charge-dipole nonlinear GP mean
Status: deferred

**Hypothesis:** Collective anisotropic electronic response can improve finite surrogate proposals beyond scalarchargecurvature.
**Outcome and uncertainty:**568all469converged withvalidenergy/zeroerrors butcostloss;{'faster': 39, 'slower': 261, 'same': 169}. No passive modelstatediagnosis. Atomic parameters transferredfromPQEq into a self-width-matched Gaussianmultipolefunctional remainuncalibrated.
**Reason to revisit:** Independently supported environment-dependent electronic parameters or direct model-response diagnostics could resolve the transferredprior/finitechart uncertainty.
**Next experiment:** Await that evidence; no amplitude,width,charge,scope,spacing orsolversweep. Fullgatesremain.

**Attempts:**
- Cycle568; candidate `f0398e400437e35df2d8bafba1acf85abbb1d94c`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/gaussian-multipole-mean/f0398e400437e35df2d8bafba1acf85abbb1d94c/algo.py), [training](evaluation_results/cycle-568-train.json), [narrative](full_log.md#cycle-568-joint-gaussian-charge-dipole-surrogate-remainder).

## simplex-surrogate-search: Local downhill simplex GP proposals
Status: deferred

**Hypothesis:** Finite local shape adaptation may find usefulproposals missedby derivative-directedsearch.
**Outcome and uncertainty:**569all469converged withvalidenergy/zeroerrors butcostloss;{'faster': 32, 'slower': 45, 'same': 392}. No passivemicrosearch state identifies a defect.
**Reason to revisit:** Independent finite-domain modelgeometry or observed simplexdegeneration couldsupport a specific domain-aware search.
**Next experiment:** Awaitthat evidence; no simplex,coefficient,tolerance,budget orsearchportfolio sweep.

**Attempts:**
- Cycle569; candidate `8a96f6e9a9a2aa3ae5c75d1e7039c9dd84bdc87e`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/simplex-surrogate-search/8a96f6e9a9a2aa3ae5c75d1e7039c9dd84bdc87e/algo.py), [training](evaluation_results/cycle-569-train.json), [narrative](full_log.md#cycle-569-downhill-simplex-surrogate-search).

## nonlinear-proximal-surrogate: Residual-certified nonlinear proximal GP search
Status: deferred

**Hypothesis:** The nonlinear proximal resolvent can improve finite proposals through changingcurvature.
**Outcome and uncertainty:**570all469converged withvalidenergy/zeroerrors butcostloss;{'faster': 31, 'slower': 26, 'same': 412}. No innersolverdiagnostic. BoundedBFGS replacesSCAR-PM; thisdoesnot testfullNASCAR.
**Reason to revisit:** A supported inneraccuracy/work controller or retained subproblemstate could make the nonlinearproximalmethod useful.
**Next experiment:** Awaitthat evidence; no regularization,accuracy,budget orcertificateconstant sweep.

**Attempts:**
- Cycle570; candidate `65cded16eebd0a4671d42ee6065471ba965f2bf6`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/nonlinear-proximal-surrogate/65cded16eebd0a4671d42ee6065471ba965f2bf6/algo.py), [training](evaluation_results/cycle-570-train.json), [narrative](full_log.md#cycle-570-nonlinear-proximal-surrogate-search).

## covariance-population-surrogate: Rank-selected full covariance GP search
Status: deferred

**Hypothesis:** Population covariance adaptation can explore correlated finite proposals missed by local gradient search.
**Outcome and uncertainty:**571 all469 converged,validenergy,zeroerrors,but highercost; {'faster': 59, 'slower': 69, 'same': 341}. No passive search-state diagnosis; fixedseed path depends on GPbasis.
**Reason to revisit:** Independent finite-domain geometry or observed covariance degeneration could support a specific adaptation.
**Next experiment:** Await evidence; no population,seed,scale,learning-rate orbudget sweep.

**Attempts:**
- Cycle571; candidate `ac7ea64c5db736873bcc3cdb439203544e2fa307`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/covariance-population-surrogate/ac7ea64c5db736873bcc3cdb439203544e2fa307/algo.py), [training](evaluation_results/cycle-571-train.json), [narrative](full_log.md#cycle-571-covariance-adaptive-population-surrogate-search).

## physical-lie-group-secants: Fixed-chart multiplicative positive preconditioning
Status: deferred

**Hypothesis:** Acceptedsecants in one chemicalCartesianchart can trackcurvature via Lie-group factors without native transport.
**Outcome and uncertainty:**572 passesenergy,zeroerrors,435converged34limits,but468slower/1faster. Passive trace shows persistentforce/motion; Q/acceptance states absent. Source isotropic-sampling guarantees do not apply to selected secants.
**Reason to revisit:** Independently supported unbiased curvature acquisition or direct metric-state diagnostics could resolve selected-pair bias/learning-speed uncertainty.
**Next experiment:** Await that prerequisite; no rate,line-search,cap,floor/regularization sweep.

**Attempts:**
- Cycle572; candidate `29012b06ca9f06dbfb617ca211827242272caa76`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/physical-lie-group-secants/29012b06ca9f06dbfb617ca211827242272caa76/algo.py), [training](evaluation_results/cycle-572-train.json), [trace](diagnostics/a3365b1565f842049867999b6351b7a2/bb888d07226263ea0da3837ec36ec74df07110bcd2a1fdecab0057d826b165d5.json), [narrative](full_log.md#cycle-572-lie-group-physical-secant-preconditioning).

## feasibility-ellipsoid-surrogate: Geometric feasibility cuts during GP search
Status: deferred

**Hypothesis:** An enclosing ellipsoid andexact geometriccuts could avoidunusable finiteproposals.
**Outcome and uncertainty:**573 all469converged,validenergy,zeroerrors,butcostloss; {'faster': 117, 'slower': 100, 'same': 252}. No retainedcut/shape states; nonconvexobjectivecuts can discard usefulbasins.
**Reason to revisit:** A supported nonconvex separationoracle or direct discarded-basin diagnosis could make geometricellipsoidsearch useful.
**Next experiment:** Await such evidence; no cutdepth,radius,budget orconstraintportfolio sweep.

**Attempts:**
- Cycle573; candidate `00e7046b4ef8da0af29e655150ea7b1f207b3014`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/feasibility-ellipsoid-surrogate/00e7046b4ef8da0af29e655150ea7b1f207b3014/algo.py), [training](evaluation_results/cycle-573-train.json), [narrative](full_log.md#cycle-573-feasibility-cut-ellipsoid-surrogate-search).

## coordination-dispersion-mean: Environment-dependent pair dispersion GP remainder
Status: deferred

**Hypothesis:** Published D3 coordination response can supplymissing nonlinearcollective pairattraction beyond thecurrentquadratic.
**Outcome and uncertainty:**574 all469converged,validenergy,zeroerrors,costloss; {'faster': 36, 'slower': 46, 'same': 387}. No modelstatediagnosis. Thismeets422coordination prerequisite but PBE-BJtransferandchargeabsence remainapproximate.
**Reason to revisit:** Independently calibrated charge-dependent response or a direct mean-error diagnosis could support a moreconsistent physicalcomponent.
**Next experiment:** Await that evidence; no amplitude,damping,C6,radius,scope,spacing orkernel sweep.

**Attempts:**
- Cycle574; candidate `54de5ea481cf315def027fadf386e97d955df8bb`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/coordination-dispersion-mean/54de5ea481cf315def027fadf386e97d955df8bb/algo.py), [training](evaluation_results/cycle-574-train.json), [narrative](full_log.md#cycle-574-coordination-dependent-dispersion-surrogate-remainder).

## doubly-optimistic-surrogate: Midpoint displacement learning with extrapolated hints
Status: deferred

**Hypothesis:** Online displacement learning could exploit changing GPcurvature without Hessians orline searches.
**Outcome and uncertainty:**575 all469converged,validenergy,zeroerrors,butcostloss; {'faster': 75, 'slower': 85, 'same': 309}. No search-state diagnostics identify an error; geometric D/T adaptation doesnot carry the source complexity guarantee.
**Reason to revisit:** Independent finite-domain online regret controls or direct hint-error diagnosis could support a justified adaptation.
**Next experiment:** Await evidence; no episode,radius,rate/accumulator oraveraging sweep.

**Attempts:**
- Cycle575; candidate `5dc85a65878e3cca271880ab7b5606ce58c0a7f6`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decision discard. [Implementation](ideas/doubly-optimistic-surrogate/5dc85a65878e3cca271880ab7b5606ce58c0a7f6/algo.py), [training](evaluation_results/cycle-575-train.json), [narrative](full_log.md#cycle-575-doubly-optimistic-midpoint-surrogate-search).

## coordination-charge-mean: Environment-dependent electrostatic GP remainder
Status: deferred

**Hypothesis:** Published calibrated coordination-dependent charge response supplies nonlinear physical mean without changing localquadraticcurvature.
**Outcome and uncertainty:**579all469converge,energyvalid,zeroerrors,butcostloss;{'slower': 105, 'faster': 78, 'same': 286}. Charge fit is not total-energy calibration; neutralprior andlongrangeCT remainlimitations. Empty passivefailure_details givesno localizedrepair.
**Reason to revisit:** EEQBC2025 independently calibrated bondcapacitance/localcharge resolves longrangeCT mechanism and can test whether charge-locality matters for this remainder.
**Next experiment:** Auditandimplement complete source EEQBC energy/gradient withsamefinitechart/originjet; compare463fullgates. No arbitraryamplitude,scope,charge,CN,width orsolver sweeps.

**Attempts:**
- Cycle579; candidate `deaa3057409c8764df50cd935912b811c9802147`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decisiondiscard. [Implementation](ideas/coordination-charge-mean/deaa3057409c8764df50cd935912b811c9802147/algo.py), [training](evaluation_results/cycle-579-train.json), [narrative](full_log.md#cycle-579-coordination-dependent-eeq-electrostatic-mean).

**580 reassessment:** FullEEQBC2025independentcapacity/localcharge/widthmodelall469converge,energyvalid0errorsbutcostloss;{'slower': 278, 'same': 155, 'faster': 36}. Sourcecalibration+longrangeCTcontrolinsufficient. Awaitindependentlycalibrated electrostaticenergy response or passive modelerror diagnosis; no parameter/scope/units sweep.
- Cycle580; candidate `caee340cd917a598fa1acc080bfcb9f6400fec0e`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decisiondiscard. [Implementation](ideas/coordination-charge-mean/caee340cd917a598fa1acc080bfcb9f6400fec0e/algo.py), [training](evaluation_results/cycle-580-train.json), [narrative](full_log.md#cycle-580-bond-capacitance-electrostatic-mean).

## extragradient-newton-surrogate: Energy-certified lookahead with hybrid Newton correction
Status: deferred

**Hypothesis:** GradientlookaheadbeforeNewtonhelpsescapeunhelpfulcurvatureregionswithinthecheapGP.
**Outcome and uncertainty:**581all469converge,energyvalid0errors,butcostloss;{'faster': 4, 'same': 458, 'slower': 7}. Emptyfailure_details,noinnertrajectorydiagnosis.
**Reason to revisit:** Independent diagnosisoflookahead/curvatureerror or modelsearchdifficultycouldjustifyaboundedrepair; sourceArmijovariantaddresses464initialscaleobstaclealready.
**Next experiment:** Awaitsuchevidence; no lookaheadfraction,zeta,Armijo,Hspacing,budgetorblendsweep.

**Attempts:**
- Cycle581; candidate `a4d4d257a20207f650b88fed29937023d91f38f4`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decisiondiscard. [Implementation](ideas/extragradient-newton-surrogate/a4d4d257a20207f650b88fed29937023d91f38f4/algo.py), [training](evaluation_results/cycle-581-train.json), [narrative](full_log.md#cycle-581-extragradient-newton-surrogate-search).

## two-channel-orbital-response: Variable orbital basis for linear carbon centers
Status: deferred

**Hypothesis:** Two transverse pi channels atC2 withoneC3channelrepresent collectivebondcurvaturemissingfrom463.
**Outcome and uncertainty:**582all469converge,validenergy0errors,butcostloss;{'same': 468, 'slower': 1}. Publishedstraight-chainparametersaretransferredtofinite/mixed/bentcomponents;bondalternation,endgroupenergiesandchargemissing. No passiveorbitalstates.
**Reason to revisit:** Independently supported finitebondalternation/endgroup response andconsistentmixed-carboncalibrationmayresolvetheextensionmismatch.
**Next experiment:** Awaitthoseparameters/directdiagnostics; no amplitude,scope,angle,gap orrank sweep.

**Attempts:**
- Cycle582; candidate `79df6a6acc49920f1d405581bb3acf12692783c3`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decisiondiscard. [Implementation](ideas/two-channel-orbital-response/79df6a6acc49920f1d405581bb3acf12692783c3/algo.py), [training](evaluation_results/cycle-582-train.json), [narrative](full_log.md#cycle-582-two-channel-carbon-orbital-bond-response).

## secant-damped-dynamics: Persistent dynamics controlled by learned spectral bounds
Status: deferred

**Hypothesis:** Chemical whitening plus curvature-controlled time/damping accelerates physical relaxation without paid trials.
**Outcome and uncertainty:**583invalid:184converged,284limits,1SCCfailure; all468returnedcases slower. Benzene/water passive late trajectory oscillates with large forces/moves; spectral/velocity states unavailable, so no localized model defect is established.
**Reason to revisit:** Independently justified nonlinear stability or gauge-aware spectral controls could address persistent motion; exact state diagnostics could distinguish curvature from integration failure.
**Next experiment:** Await that prerequisite; no step,damping,floor,cap orreset sweep based on broadloss.

**Attempts:**
- Cycle583; candidate `0132439ef96d00c2b2ff70d9bfb18da128380ae9`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decisioninvalid. [Implementation](ideas/secant-damped-dynamics/0132439ef96d00c2b2ff70d9bfb18da128380ae9/algo.py), [training](evaluation_results/cycle-583-train.json), [narrative](full_log.md#cycle-583-secant-controlled-dynamical-functional-particles).

## implicit-radau-surrogate: Stiffly damped integration of the GP gradient
Status: deferred

**Hypothesis:** Implicit RadauIIA can suppress stiff surrogate modes and provide better proposals than short BFGS search.
**Outcome and uncertainty:**584all469converge,validenergy0errors,butcostloss;{'same': 445, 'faster': 11, 'slower': 13}. No passive innertrajectorydiagnosis; integrationaccuracy doesnotestablish physical modelaccuracy.
**Reason to revisit:** Independent evidenceofnumericalstiffnessor an integration failure could justify a targeted solver change.
**Next experiment:** Awaitsuchdiagnostics; no horizon,tolerance,budget orstage sweep.

**Attempts:**
- Cycle584; candidate `65766722f6dc6ce90d41d39c5451148d67c79e88`; champion `99a2d8b5482466729ddde624fc4253fc98aba249`; decisiondiscard. [Implementation](ideas/implicit-radau-surrogate/65766722f6dc6ce90d41d39c5451148d67c79e88/algo.py), [training](evaluation_results/cycle-584-train.json), [narrative](full_log.md#cycle-584-implicit-radau-surrogate-gradient-flow).
