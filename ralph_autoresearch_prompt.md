Follow `program.md`. Resume from the workspace's git state, generated tables,
evaluation evidence, `full_log.md` and `backlog.md`; recover pending work first.
Explore fundamental changes, including complete rewrites, as well as incremental
improvements. Follow promising public sources within the repository restrictions.
Implement one coherent idea per cycle and commit only `algo.py`; preserve
promising rejected implementations in `ideas/` and the backlog. Use passive
diagnostics to guide bounded repairs of promising failures under the
unchanged evaluation gates.
Use `scripts/cycle.py begin` to reserve/resume the research cycle, the evaluation
client for measurements, and `scripts/cycle.py record` to invoke the decision
writer. Respect the research and repository-access boundaries. Continue
until manually interrupted.

Do not run local numerical screening, energy/force calculations, optimizer tests,
or toy benchmarks. Use only the supplied remote evaluation client for measurements;
local code editing, static checks, and calculations on saved results are allowed.
