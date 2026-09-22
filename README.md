# AutoSella-G: Grok 4.6 (xhigh) research archive

This snapshot preserves the research run `grok-46-xhigh-new-data-77e061f4f75ce6ce` through 900 completed decisions (1,142 receipt rows, including 242 pending entries), captured on 22 September 2026. Research commits retain their original ancestry; the final archive commit collects the decision tables, narrative log, backlog, prompts, scripts, candidate sources, and frozen training/validation inputs.

## Current champion

The root **`algo.py` is the accepted champion**, cycle 897, commit `f9d546f8780f42c2f7178ac63711672ce38a5e77`. Its SHA-256 is `6e2eab4b8c3a1323883c9295f3ae3a11dfe59cf1b45095cb637826e3845fa0d4`. It is byte-identical to `candidates/f9d546f8780f42c2f7178ac63711672ce38a5e77/algo.py` and to the AutoSella-G implementation used on `xtb_ff_evaluation`.

The recorded per-molecule relative force-call costs are **97.7878% training** and **98.1403% validation**, relative to the run's initial Sella anchor. All 469 training and 465 validation molecules converged; both evaluations passed the run's validity gate. These are research-set scores, not the held-out evaluation results.

Later rejected candidates remain in Git history and `candidates/`. The uncompleted cycle-901 working experiment is preserved separately as `pending/algo.py`; it is not the champion. This is a frozen capture and makes no claim that the source run has stopped.

## Contents

- `results.tsv`, `generalizable.tsv`, and `non_generalizable.tsv`: recorded decisions and scores.
- `full_log.md`, `backlog.md`, and `ideas/`: hypotheses, results, and implementation history.
- `candidates/<commit>/algo.py`: exact sources for every recorded candidate and anchor.
- `molecules/`: the frozen `main-f59f2b89a-gfn2-v1` inputs and reference metadata.
- `research_manifest.json`: source identities, champion selection, hashes, and exclusions.
- `grok_progress.json`: 900 verified candidate points with evaluation-receipt hashes and elapsed timestamps.
- `research_time.jsonl`: partial instrumentation added late in the run; it must not be mistaken for the full research duration.

Raw per-evaluation dumps remain local and are deliberately excluded from Git. Log links into `evaluation_results/` and `anchors/` therefore refer to the original local run. Runtime credentials, private controller state, and caches are excluded. The software is LGPL-3.0-only; molecular data retain their upstream terms recorded with the dataset.
