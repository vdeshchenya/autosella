# Fable 5.1 max autoresearch progress

Saved progress for `fable-51-max-new-data-fd6051477f2627c8`, using
`claude-fable-5-1` with `max` reasoning and dataset
`main-f59f2b89a-gfn2-v1`. Started 2026-09-16; stopped by the operator on
2026-09-19 after the provider's monthly spending limit blocked further work.

The latest recorded cycle is 109. Its candidate was discarded, and `algo.py`
was restored to champion `cf2439f9ee81bc0e65f361fad971390d682ad652`
(cycle 101): training cost 0.6512909040 and validation cost 0.6499488938
relative to the reference. The recorded results contain 160 rows, including
earlier records with reused cycle numbers; they are preserved without rewriting.

## Saved material

- [full_log.md](full_log.md): complete saved research narrative.
- [backlog.md](backlog.md): research ideas, evidence, and pending directions.
- [results.tsv](results.tsv), [generalizable.tsv](generalizable.tsv), and
  [non_generalizable.tsv](non_generalizable.tsv): original decision tables.
- `anchors/` and `diagnostics/`: initial reference results and diagnostic outputs.
- `ideas/`: retained implementations of research ideas.
- `candidates/<commit>/algo.py`: source snapshots for all 109 unique candidate
  and anchor commits referenced by the decision table, including discarded
  candidates outside the final branch's ancestry.
- `molecules/`: the exact frozen dataset and metadata used by this run,
  verified against its launch hashes.
- `program.md`, prompt templates, and `scripts/`: the run's research instructions
  and workflow code.
- [progress-manifest.json](progress-manifest.json): run identity, revisions,
  dataset release, and SHA-256 checksums for the saved artifacts.

This branch preserves the run's 74-commit final ancestry, followed by the
progress snapshot commit. It is an archive of the isolated run, not a merge
into the current main branch. Evaluation paths beginning `/workspace/` are
original container paths. Per-evaluation JSON files in `evaluation_results/`
are retained locally and excluded from Git; the decision tables and research
log retain the result summaries. The latest upstream test-set changes were not
part of this experiment. Provider credentials, private runner state, and runtime
caches are not included.
