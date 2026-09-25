# Geometry optimizer autoresearch

Research code changes only `algo.py`; notes and saved ideas remain uncommitted.
Each run has a fresh local Docker workspace,
private home and approved read-only data. A restricted gateway evaluates the
committed optimizer on remote GFN2-xTB workers.

The objective is lower mean relative force-call cost. Complete evaluations need
`mean_rel_energy >= 1 - 1e-9` and zero internal errors; convergence is diagnostic.
Against the current champion, train must improve by at least `1e-4` and
valid by more than `1e-4`. Infrastructure interruptions remain pending/unscored.

- [Research contract](program.md): interface, acceptance, evidence and recovery.
- [Operator guide](LOCAL_REMOTE_AUTORESEARCH.md): image, isolation, lifecycle and dashboard.
- [Gateway](evaluation_access/README.md): trusted evaluation service and client.
- [Prepared data](molecules/README.md): immutable release format.

The dashboard shows each run's progress, decision tables and retained logs.
Fresh runs start with an empty `backlog.md`; promising rejected implementations
are preserved under `ideas/` for evidence-based revision or combination.
`scripts/cycle.py` reserves/resumes cycle numbers and records decisions through
the decision writer. Startup requires per-molecule reference reproduction:
force-call ratios exactly 1 and energy recovery within `1e-9` of 1.

## Launch an experiment

Run these commands from the repository root in the project's Python environment.
Keep Docker Desktop running on the Mac; the research agent runs locally and
molecular evaluations run remotely.

1. Follow the [operator setup](LOCAL_REMOTE_AUTORESEARCH.md#configure) to build
   the runtime image and deploy the matching gateway, evaluator and prepared
   data. The pool uses **196 workers: 49 per host** across four hosts. Each host has 48 physical
   cores; the 49th worker uses one additional SMT sibling. Pin every worker to
   one logical CPU, set numerical library threads to 1 and `JAX_ENABLE_X64=1`.
   Match the loaded math libraries (`glibc`/`libm`) across worker hosts as well
   as Python packages and xTB; matching package versions alone is insufficient.
2. Copy [the example config](isolated_runs/config.example.json) to a private
   location and replace its placeholders with the approved image ID, snapshot
   and dataset manifests, gateway/relay settings and local UID/GID. Set
   `purpose` to `research` (the example defaults to `verification`), choose
   `provider` (`codex` or `claude`), and set the desired `model` and supported
   reasoning `effort`. Give each experiment a new `name`; keep `auth.kind`
   as `interactive` for account sign-in.
3. Set the paths below and make `AR_NAME` match the config's `name`:

   ```bash
   AR_ROOT="$HOME/.local/share/gigaopt/ar-runs"
   AR_CONFIG="/absolute/path/to/your-run.json"
   AR_NAME="my-experiment"

   python3 -m isolated_runs.launcher --root "$AR_ROOT" create --config "$AR_CONFIG"
   python3 -m isolated_runs.launcher --root "$AR_ROOT" auth "$AR_NAME"
   python3 -m isolated_runs.launcher --root "$AR_ROOT" start "$AR_NAME"
   python3 -m isolated_runs.launcher --root "$AR_ROOT" status "$AR_NAME"
   ```

Complete the sign-in requested by `auth`. `start` obtains the remote train and
validation starting metrics and checks reference agreement before launching
Ralph. A running container alone does not mean the starting checks have passed;
inspect the dashboard's starting-anchor and research status.

To pause and later continue the same experiment, preserving its records:

```bash
python3 -m isolated_runs.launcher --root "$AR_ROOT" stop "$AR_NAME"
python3 -m isolated_runs.launcher --root "$AR_ROOT" resume "$AR_NAME"
```

## View progress and logs

In a separate terminal, use the **same run root** as the launcher:

```bash
AR_ROOT="$HOME/.local/share/gigaopt/ar-runs"
python3 -m isolated_runs.monitor --root "$AR_ROOT" --port 8767 --gateway-status
```

Open **[http://127.0.0.1:8767/](http://127.0.0.1:8767/)**. If the dashboard is
already running, open it directly; use another port for a separate viewer.
The page refreshes automatically and shows the champion's training/validation
costs and progress curves. Hover over a marker for cycle costs, energy ratios
and the decision. Buttons open the three decision tables, **Full log** and
**Backlog**; expand **Logs** for Ralph transcripts and debugging output.

Run files persist under `$AR_ROOT/$AR_NAME/workspace`; Ralph transcripts are
in `.ralph/logs/` there. Closing the dashboard does not stop the experiment.
