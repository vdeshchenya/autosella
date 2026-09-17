# Research iteration {{iteration}} / {{max_iterations}}

{{prompt}}

## Previous iteration context

{{context}}

Follow `program.md`; recover pending evaluations before new work and consult
the backlog. Use `scripts/cycle.py` for research cycle numbers; the Ralph
iteration number above is not the research cycle. Read the permitted workspace
evidence and public sources.
Preserve promising rejected implementations before restoring the anchor.
Use passive diagnostics to guide bounded repairs of promising failures under the
unchanged evaluation gates.
Never open or search `.ralph/` or Ralph state/history/context files.
Record each completed cycle and continue until
the human stops the loop; a new best result is not completion.

Do not run local numerical screening, energy/force calculations, optimizer tests,
or toy benchmarks. Use only the supplied remote evaluation client for measurements;
local code editing, static checks, and calculations on saved results are allowed.
