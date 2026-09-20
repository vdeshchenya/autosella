# Astra high autoresearch progress

Snapshot of `astra-high-new-data` at **2026-09-20T19:11:04.892696+00:00**, using `gpt-6-astra` with **high** reasoning effort. The running experiment was not stopped or changed.

## Snapshot and accepted champion

- Latest recorded cycle: **586**, decision **invalid** (Positive scaled rank-one physical secants).
- Root `algo.py` is the exact working source from commit `662f85875c0af3912e3407069350bfccca6b2a63` at capture time. This candidate is not the accepted champion.
- Latest accepted champion: **cycle 463**, `99a2d8b5482466729ddde624fc4253fc98aba249`. Its source is [candidates/99a2d8b5482466729ddde624fc4253fc98aba249/algo.py](candidates/99a2d8b5482466729ddde624fc4253fc98aba249/algo.py).
- Champion mean relative force-call cost: **0.745101515** on training and **0.766560528** on validation, as recorded in `results.tsv`.
- Frozen dataset release: `main-f59f2b89a-gfn2-v1` (469 training and 465 validation molecules). This is the experiment's original dataset, not a substitution from a later repository revision.

## Saved research

`full_log.md`, `backlog.md`, `results.tsv`, `generalizable.tsv`, and `non_generalizable.tsv` preserve the research narrative, plans, and decision summaries. The current program instructions, runner scripts/templates, and ordinary `cycle.json` record are included.

`ideas/` contains 554 Python source snapshots. `candidates/<commit>/algo.py` preserves every candidate and anchor referenced by the captured results, plus the current HEAD (585 distinct commits), including sources that may no longer be reachable through the accepted Git history. The branch also retains the source HEAD's Git ancestry.

`molecules/` contains the frozen input geometries, selection/reference data, release metadata, and verified checksums. These are benchmark inputs, distinct from generated per-evaluation outputs.

## Excluded outputs and reproduction limits

Per-evaluation outputs (`evaluation_results/`), generated anchor evaluations (`anchors/`), and diagnostic dumps (`diagnostics/`) remain local and are excluded from Git. Runtime state, credentials, and caches are also excluded. Paths to those outputs inside the preserved logs and decision tables are historical references, not files bundled here.

This archive preserves source and research evidence, not a standalone deployment of the external evaluation gateway and worker services. No scientific evaluations were rerun for this archival commit. `progress-manifest.json` records provenance and SHA-256 checksums; public-source downloads are represented by a citation and hashes.
