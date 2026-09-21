# Astra high autoresearch progress

Stopped snapshot of `astra-high-new-data` captured at **2026-09-21T14:17:01.637474+00:00**, using `gpt-6-astra` with **high** reasoning effort. The run was stopped at **2026-09-21T13:26:11.306441+00:00**; research container and relay are stopped and evaluation access is paused.

## Final research state

- Last recorded completed cycle: **739**, decision **discard** (Frozen-reference onsite coordination mean).
- Working cycle at stop: **740**, commit `f3fb1debe771cce207350cb2410681a643931575`. Root `algo.py` is its exact source. It has no recorded final decision and is not the accepted champion.
- The last local cycle-740 training receipt was pending; its saved progress and timestamp are recorded in `progress-manifest.json`. This archive does not turn that partial receipt into a completed result.
- Accepted champion remains **cycle 463**, `99a2d8b5482466729ddde624fc4253fc98aba249`: [champion source](candidates/99a2d8b5482466729ddde624fc4253fc98aba249/algo.py).
- Champion mean relative force-call cost: **0.745101515** on train and **0.766560528** on validation.
- Frozen dataset release: `main-f59f2b89a-gfn2-v1` (469 training and 465 validation molecules).

## Saved research

The full log, backlog, all decision summary tables, current program instructions, runner scripts/templates, cycle record and research-time checkpoint are included. `analysis/` contains the small repeated-geometry consistency summary, not its underlying diagnostic frames.

`ideas/` contains 709 Python source snapshots, including the full-precision ANI-1x preflight implementation. `candidates/<commit>/algo.py` preserves 739 distinct candidate/anchor sources, including every commit referenced by the result tables and the final working HEAD. Previous archived sources are retained. Git ancestry includes the stopped source HEAD as well as the earlier archive.

`research_sources/` preserves downloaded implementation code, model parameters, source manifests and licenses. Paper downloads remain local; citations are in `full_log.md` and their hashes are in `progress-manifest.json`. `molecules/` preserves the experiment's frozen benchmark inputs, verified against the stopped run.

## Excluded artifacts

Raw `evaluation_results/`, generated `anchors/`, diagnostic dumps, private runtime state, credentials, caches and external service installations remain local and are excluded from Git. Historical links to those artifacts in logs are not bundled result files. No scientific evaluations were rerun for this commit. The archive preserves research progress and provenance, not a standalone deployment of the gateway and workers.
