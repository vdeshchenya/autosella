# Geometry optimizer research

## Objective and environment

Develop a general molecular geometry optimizer that achieves the fixed reference
energy lowering with fewer force evaluations. Improve the optimizer supplied in
`algo.py`; you may replace its entire implementation.

You work inside an isolated container. Your Git workspace and research artifacts
are in `/workspace`; the supplied dataset and metadata are read-only at
`/dataset`. Public-web research is available subject to the repository
restrictions described below. Submit numerical evaluations through the provided
client. A trusted service outside the container runs the fixed GFN2-xTB evaluator
and returns results to your workspace. Starting geometries are source SPICE
conformers and RowanSci structures with reproducible 0.02 Å coordinate noise;
no GFN-FF preoptimization is applied.

Do not run numerical experiments inside this container. This includes local
candidate screening, molecular optimization, energy/force calculations,
synthetic or toy-potential benchmarks, and diagnostic optimizer executions,
regardless of sample size or worker count. Obtain all optimizer measurements
through the supplied remote evaluation client, following the training and
validation gates.
Do not build or run local evaluation harnesses, including background jobs or
subprocesses. Local work may include editing code, static checks such as syntax
parsing without executing the optimizer, public-web research, and calculations
on saved data to analyze results already returned by the evaluator. Such analysis
must not execute the optimizer, call an energy/force calculator, or run a mock
or toy experiment.

The dataset has two subsets, called **splits**: `train` and `valid` (validation).
Test candidates on train first. Only candidates that pass the training gates
proceed to validation, which checks whether their improvement extends beyond
training. Each molecule has fixed reference energies and a reference force-call
count in `/dataset/train_XTB.json` or `/dataset/valid_XTB.json`. These references
stay fixed throughout research. The **champion** is the most recently accepted
optimizer, or the supplied starting optimizer before the first accepted change.

## Optimizer contract

Expose this function at module scope in `algo.py`. Fix seeds if using randomness.

```python
def minimize_func(
    positions: np.ndarray,       # (N, 3), float64, nm
    atomic_numbers: np.ndarray,  # (N,), integers
    calc: Callable,              # calc(pos) -> (energy, forces)
    max_force_calls: int,        # 200 in evaluation
    converged: Callable,         # converged() -> bool; no force-call cost
) -> tuple[np.ndarray, int]:
    # Energy: kJ/mol; forces = -grad(E): kJ/mol/nm.
    return final_positions, n_force_calls
```

The evaluator measures cost by counting force evaluations. Every call to
`calc(positions)` that starts an energy-and-force calculation counts, including
calculations at trial geometries that your method later rejects. On normal
return, provide `(final_positions, n_force_calls)`, where `n_force_calls` is the
integer number of these calculations and `final_positions` is the geometry from
the last successfully completed calculation. This ensures that the returned
geometry has the energy recorded by the evaluator. Do not change those
coordinates after their final evaluation; if you restore an earlier geometry or
construct a different one, evaluate it again before returning it, using a
remaining force call. The evaluator checks coordinate equality after conversion
to float64. Continue until external convergence, the limit of 200 force
evaluations per molecule, or the evaluator's time limit, whichever comes first.
Let evaluator stop signals propagate. When a limit interrupts optimization, the
evaluator scores the last successfully evaluated geometry, provided there is
one and no invalid calculation has occurred. Reaching a limit without convergence
does not by itself invalidate the result: the energy requirement still applies.
Optimizer exceptions, nonfinite numerical results, malformed returned values,
and an incorrect reported call count are errors that invalidate the evaluation.

You need not predict whether a new method will achieve sufficient accuracy within
200 calls. Design it to operate within `max_force_calls`; evaluation determines
whether it succeeds. Keep the interface and fixed evaluation rules unchanged.

### Convergence

The optimization is considered converged if all five conditions hold
simultaneously, using strict `<` comparisons:

| Quantity | Threshold |
| --- | --- |
| Absolute energy change | `1e-6` Hartree |
| Maximum atomic force norm | `4.5e-4` Hartree/Bohr |
| RMS atomic force norm | `3e-4` Hartree/Bohr |
| Maximum atomic displacement | `1.8e-3` Å |
| RMS atomic displacement | `1.2e-3` Å |

Energy change and displacement compare consecutive successful `calc()` calls.
Displacements are measured after removing translation and optimally aligning
rotation, with corresponding atoms equally weighted. Force and displacement
statistics use per-atom Euclidean vector norms; RMS is over atoms, not Cartesian
components. The first successful call initializes history and cannot establish
convergence. `converged()` reads the latest result without a force evaluation.

The convergence criteria are fixed and available for inspection. You may use
them to understand the required accuracy and design the optimizer. The evaluator
alone determines convergence. Do not modify its criteria or fabricate progress
through repeated or artificially small moves whose purpose is only to trigger
stopping conditions. Optimize the actual molecular geometry under the fixed
energy and cost requirements.

## Metrics and acceptance

For molecule $i$, let $n_i(a)$ be the force-call count of optimizer $a$, and
$n_i^{\mathrm{ref}}$ the fixed reference count. Let $E_i^0$ be the reference
starting energy, $E_i(a)$ the evaluated final energy, and
$E_i^{\mathrm{ref}}$ the reference final energy. The evaluator reports

$$
c_i(a)=\frac{n_i(a)}{n_i^{\mathrm{ref}}},\qquad
r_i(a)=\frac{E_i^0-E_i(a)}{E_i^0-E_i^{\mathrm{ref}}}.
$$

For a split $S$ containing $N_S$ molecules, its cost and energy recovery are

$$
C_S(a)=\frac{1}{N_S}\sum_{i\in S}c_i(a),\qquad
R_S(a)=\frac{1}{N_S}\sum_{i\in S}r_i(a).
$$

These correspond to `mean_rel_steps` and `mean_rel_energy` in evaluator JSON;
per-molecule values are `rel_steps` and `rel_energy`. Lower cost is better.
Energy regression is not clipped. Every molecule has equal weight; an incomplete
split has no final score. Define validity as

$$
V_S(a)\iff\text{evaluation complete}\ \land\
\texttt{internal_error_count}=0\ \land\ R_S(a)\ge 1-10^{-9}.
$$

This is an aggregate energy requirement. The convergence fraction is diagnostic,
not an additional acceptance gate. Infrastructure interruptions leave evaluation
pending rather than creating a scientific rejection.

Before research starts, the launcher evaluates the starting optimizer on both
splits. Its force-call counts must reproduce the per-molecule references, giving
$C_{\mathrm{train}}=C_{\mathrm{valid}}=1$. Energy recovery must also reproduce 1
within the startup comparison tolerance of `1e-9`, per molecule and in aggregate.
A mismatch is a setup problem, not a new baseline to accept or renormalize.
The launcher requires complete, valid, matching starting evidence, records the
starting champion as cycle 1, and only then starts research.

For later candidates, compare against the current champion $a^\star$, using its
recorded train and validation evaluations from the same dataset release. Define

$$
\Delta_S(a)=C_S(a^\star)-C_S(a),\qquad\delta=10^{-4}.
$$

A candidate proceeds to validation only if
$V_{\mathrm{train}}(a)\land\Delta_{\mathrm{train}}(a)\ge\delta$.
It becomes the new champion only if validation also satisfies
$V_{\mathrm{valid}}(a)\land\Delta_{\mathrm{valid}}(a)>\delta$.
A keep therefore already means that both stages passed.

## Workspace files and tools

Paths below are relative to `/workspace` unless absolute.

| File or directory | Purpose |
| --- | --- |
| `algo.py` | Optimizer implementation; the only file you commit. |
| `anchors/train.json`, `anchors/valid.json` | Starting evaluations prepared by the launcher. |
| `evaluation_results/cycle-N-train.json`, `evaluation_results/cycle-N-valid.json` | Candidate evaluation evidence: aggregate metrics, per-molecule `results`, `errors`, and evaluation identifiers. |
| `diagnostics/<evaluation-ID>/` | Downloaded manifests and passive failure/nonconvergence bundles from your own evaluations; diagnostic evidence, not scores. |
| `results.tsv` | Machine-written decisions and pending records for all cycles. |
| `generalizable.tsv` | Starting champion and subsequent keeps. |
| `non_generalizable.tsv` | Candidates that passed training but were rejected on validation. |
| `full_log.md` | Your chronological account of each hypothesis, implementation change, interpretation, and next experiment. |
| `backlog.md` | Promising unsuccessful ideas, linked to evidence and preserved implementations, with reasons to revisit them. |
| `ideas/<idea-id>/<candidate-SHA>/algo.py` | Preserved versions of promising rejected implementations; never overwrite a saved version. |
| `cycle.json` | Machine-managed current cycle number, champion, and evaluation paths; do not edit manually. |

Per-molecule evidence is in the JSON files above. Do not assume a local `run.log`
exists. Keep research notes, saved implementations, evaluation files and generated
tables uncommitted so they survive restoring the champion.

`evaluation_access/client.py` submits and reconnects to evaluations.
`scripts/cycle.py` allocates cycle numbers and supplies the recorded paths and
commits to `scripts/record_decision.py`. The latter computes the gates from
evaluator JSON and is the sole writer of the three TSV tables. Never hand-enter
metrics, edit a generated decision, or fabricate evaluation evidence.

### Passive failure diagnostics

When enabled by the service operator, workers capture bounded diagnostic bundles
for failures and nonconverged outcomes during ordinary evaluations; successful
converged runs do not retain these buffers. Receipts identify unavailable or
disabled capture. The trusted service stores bundles under their evaluation ID.
The evaluation JSON's `failure_details` contains compact error/traceback evidence
and a `diagnostics` receipt with available call counts and upload status. Inspect
these details and relevant available bundles before choosing a targeted repair.
Use an ID from your own evaluation receipt to download its manifest:

```bash
python evaluation_access/client.py diagnostics --evaluation-id <ID>
```

The manifest is saved to `diagnostics/<ID>/manifest.json`. Use its molecule name
and artifact ID to retrieve a selected bundle:

```bash
python evaluation_access/client.py diagnostics --evaluation-id <ID> --molecule <name> --artifact-id <SHA256>
```

The client downloads bounded pages, verifies the hash and saves the bundle to
`diagnostics/<ID>/<SHA256>.json`. The default output directory is
`/workspace/diagnostics`; `--output-dir` can change it. Manifest results can be
filtered with `--molecule` and paged with `--offset` and `--limit`. Use names and
IDs returned by the manifest. Bundles retain at most eight full frames and 200
scalar records; check availability and truncation before drawing conclusions.
Missing evidence is not evidence that a failure mechanism was absent.

These reads perform no calculations and neither submit nor resume evaluations.
Analyze the saved data locally within the boundary above. Separate remote
interface-check or replay jobs are not available. Partial results and bundles
can guide a repair but cannot supply a score or bypass a full-split gate.

## Research direction and boundaries

Pursue fundamental improvements as well as incremental ones. The starting
optimizer is a reference point, not a design you must preserve. You may replace
any part of `algo.py`, including the entire implementation.

Implementation repairs and small, justified parameter experiments are legitimate
research. Classify an unsuccessful attempt before choosing the next experiment: an
implementation or contract error, numerical instability, energy or cost failure,
or an infrastructure interruption (pending, not a method rejection). When a method
shows useful signal and its failure is localized, pursue a bounded repair sequence
of up to three further candidates before reassessing the direction. Each must
target a diagnosed cause; stop earlier if evidence contradicts the mechanism or
no justified repair remains. Use general repairs, not molecule-specific exceptions.
Every refinement consumes the normal evaluation budget and must pass the same
full train/validation gates; partial results are diagnostic only. Record
conclusions about the variant and conditions actually tested, rather than closing
an entire method family, and append attempts to the existing backlog without
erasing earlier evidence.

Explore substantially different formulations when justified: new coordinate
representations, continuous-time or ODE-based approaches, different optimization
principles, or ideas beyond these examples. These are possibilities, not a
prescribed search space. One coherent experiment may be a complete rewrite.
Explain the central hypothesis and why it could reach the required energy with
fewer force evaluations, then test the smallest implementation that meaningfully
evaluates that hypothesis. A failed implementation does not by itself disprove
its underlying idea; retain promising ideas for later revision or combination.

You may inspect the supplied train/validation metadata and per-molecule evaluation
results to understand successes and failures. Use those observations to develop
general methods; do not introduce molecule-specific recognition or special cases.
State an explanation and an experiment that could show it is wrong. Never gate
behavior on known molecule identities, exact formulas/count tables, or identifying
graph/fingerprint predicates. Never game convergence, the energy gate or the
force-call budget.

Actively search the public web for ideas relevant to this research. Explore
publications, preprints, technical blog posts, documentation, and public
implementations, including approaches outside the current optimizer's method
family. Follow promising references and investigate unfamiliar methods. Record
useful sources with their URLs and explain what could be adapted, why it might
help, and how to test it. Treat claims from sources as hypotheses until supported
by evaluation evidence. Before browsing, read the
[repository blacklist](/opt/autoresearch/isolated_runs/repository_access_policy.md).
It also applies to mirrors, forks, raw files, APIs, archives, caches and alternate
URLs. Approved workspace/data remain permitted.

Use `full_log.md` for the chronological account of every cycle. Use `backlog.md`
as an index of unsuccessful ideas with a concrete reason to revisit them. For
example, a rewrite may introduce a useful capability but need a better component
or another improvement before it reduces cost. Explain that proposed connection
as a hypothesis, supported by the available evidence; do not list every rejected
edit merely because it could be tried again.

Before restoring the champion, give each promising idea a stable, descriptive
ID such as `<idea-id>`. Save the rejected candidate's complete, committed
`algo.py` at `ideas/<idea-id>/<candidate-SHA>/algo.py`, using the full commit SHA.
Obtain the exact source with `git show <candidate-SHA>:algo.py`; do not save a
reconstructed patch or an edited working copy. This file must remain usable even
after the candidate commit is no longer in the active Git history. Never
overwrite a saved version: later attempts get their own commit directories.
Keep explanations in `backlog.md` and link to existing evaluation JSON and
`full_log.md` rather than copying logs or numerical tables into `ideas/`.

Create one backlog entry per idea using this template. Keep each field brief
but specific enough to resume the work without reconstructing the entire cycle:

```markdown
## <idea-id>: <short title>
Status: deferred

**Hypothesis:** What changes in the optimizer, and why could it reduce cost?
**Outcome and uncertainty:** What happened, why was it rejected, and what
does the evidence leave unresolved?
**Reason to revisit:** What evidence, implementation improvement, or specific
combination could make this idea useful? Explain the expected interaction.
**Next experiment:** What would you change or combine, what would you compare
against, and what result would support or contradict the hypothesis?

**Attempts:**
- Cycle <N>; candidate `<full-SHA>`; champion `<full-SHA>`; decision <decision>.
  [Implementation](ideas/<idea-id>/<candidate-SHA>/algo.py).
  [Training evidence](evaluation_results/cycle-N-train.json).
  [Validation evidence](evaluation_results/cycle-N-valid.json), if evaluated.
  [Narrative](full_log.md#<cycle-section-anchor>).
```

Include only evidence links that exist. Consult the backlog when choosing
experiments. Set status to `revisiting` when testing an idea again, `incorporated`
when it contributes to an accepted optimizer, or `closed` when there is no
current reason to pursue it; use `deferred` while it awaits a useful next test.
Append each new attempt and update the interpretation and next experiment
without erasing previous failures. State what changed since the rejected attempt.
Evaluate revised or combined candidates against the current champion under the
same gates; do not assume separately useful changes will work together. Preserve
promising failures without accepting them merely to continue their development.

Use the supplied workspace, evaluation evidence and permitted public sources.
Do not inspect evaluator internals or modify evaluator code, clients, scripts,
prompts, metadata or datasets. Never open or search `.ralph/` or Ralph
state/history/context files. Do not operate SSH shells, Redis, queues, workers or
process pools. Install additional tools only in your private home or `/tools`.

## Research loop

Read the starting anchor files and cycle-1 table entry before your first change.
If they are missing or incomplete, report the setup problem; do not manufacture
replacement anchors. A research cycle is one committed candidate and its decision;
it is distinct from a Ralph iteration or a retry of an evaluation.

For candidate $a$ based on champion $a^\star$, apply the following decision rule
in order. An incomplete required evaluation pauses the decision at that stage.

$$
\operatorname{decision}(a)=
\begin{cases}
\text{pending}, & \text{training incomplete},\\
\text{invalid}, & \neg V_{\mathrm{train}}(a),\\
\text{discard}, & \Delta_{\mathrm{train}}(a)<\delta,\\
\text{pending}, & \text{validation incomplete or not yet evaluated},\\
\text{non\_generalizable}, & \neg V_{\mathrm{valid}}(a)
\ \lor\ \Delta_{\mathrm{valid}}(a)\le\delta,\\
\text{keep}, & \text{otherwise}.
\end{cases}
$$

Do not run validation for a training rejection. A completed rejection leaves
the champion unchanged; a keep replaces it with the candidate and its recorded
train/validation evidence. Pending means there is no final decision yet.

1. Read the tables, evaluation evidence, `full_log.md` and `backlog.md`. Ask the
   cycle tool for the current or next cycle:

   ```bash
   python scripts/cycle.py begin
   ```

   It returns the cycle number, champion commit, champion evidence paths and
   candidate output paths. Repeating it resumes an unfinished cycle; only a
   completed decision permits the next number. Use `python scripts/cycle.py status`
   to inspect the allocation. For shell commands, obtain the number with
   `N=$(python scripts/cycle.py begin --number)`; do not guess it or use the Ralph
   iteration number.
2. If resuming, inspect the Git state and existing evidence before editing or
   evaluating anything. Otherwise state one hypothesis in `full_log.md`, implement
   one coherent idea from the champion or a preserved implementation under repair,
   and commit only `algo.py` as `cycle N: <description>`. Do not change the
   candidate while its evaluation is unresolved.
3. Evaluate the committed candidate on train:

   ```bash
   python evaluation_access/client.py evaluate --split train --program algo.py --output-json "evaluation_results/cycle-${N}-train.json"
   ```

   Record the training outcome using the current allocation:

   ```bash
   python scripts/cycle.py record --description "<mechanism>"
   ```

   For `invalid` or `discard`, skip validation. For incomplete training, follow
   the interruption instructions below. For complete training that passes both
   gates, the record is pending because validation is now required; run:

   ```bash
   python evaluation_access/client.py evaluate --split valid --program algo.py --output-json "evaluation_results/cycle-${N}-valid.json"
   python scripts/cycle.py record --description "<mechanism>"
   ```

   The cycle tool supplies the cycle number, current candidate commit, champion
   commit and evidence paths to the decision writer. The writer checks provenance
   and computes all metrics and decisions. Preserve the same description when
   recording a pending evaluation and its completion.
4. Follow the completed decision. Keep accepted code. For a rejection, first
   preserve any promising implementation and update the backlog, then restore
   the champion's optimizer code and commit state while retaining uncommitted
   artifacts. Do not start another candidate while the current one is pending.
5. Finish the `full_log.md` section with cycle, candidate and champion commits,
   decision and evidence links, then **Hypothesis**, **Change** (functions),
   **Interpretation** (including alternatives and uncertainty), and
   **Next idea/control**. Link numerical evidence instead of copying tables.

## Interrupted evaluations

If an evaluation returns `status: pending`, it has no final score. The service
may still be computing molecules, the client may have reached its waiting limit,
or the connection may have failed. Keep the candidate unchanged and repeat the
same evaluation command with the same commit, split and output path. The client
reconnects to the existing request; preserve its `.evaluation_access/` state.
The service retains successful molecular results and handles infrastructure
retries. You do not need to manage individual tasks.

When a confirmed evaluation receipt exists, the cycle tool can record pending
status. For example, training can pass while validation remains incomplete: the
cycle receives a pending row, then a final decision for the same cycle after
validation completes. The writer appends that resolution and preserves the
pending history; repeating an identical record does not duplicate it.

If the cycle tool reports inconsistent decision tables after an interrupted
write, run `python scripts/cycle.py recover`, then retry the original command.
Recovery copies already recorded decisions into missing secondary-table entries;
it does not rescore candidates or change decisions. If it refuses conflicting
records, report the problem instead of editing the tables manually.

A connection failure before receipt confirmation may produce only diagnostic
JSON without an evaluation ID. In that case the writer refuses a table row.
Record the interruption in `full_log.md` and reconnect; never invent identifiers
or scores. If recovery continues to fail, report that research is blocked and
wait for the operator. Do not reject a method for an infrastructure interruption
or reset the evaluation service. A correction after a completed optimizer error
is a new committed candidate and cycle.

Continue until the human stops the loop. An improvement is not completion.
