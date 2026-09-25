# Local isolated runs, remote evaluation

Each run uses one local Docker research container with a fresh Git workspace,
private home/memory/tools, approved read-only data and a unique restricted
evaluation key. Prior histories, host credentials, agent configuration, SSH
agents and the Docker socket are excluded. Public Internet remains available.

Research instructions prohibit local numerical screening, calculations and toy
benchmarks. Agents must use the supplied remote evaluator for measurements;
local editing, static checks and analysis of returned results remain allowed.

The agent is non-root with a read-only root filesystem, no capabilities and
explicit resource limits. Its network namespace blocks private networks and
host services except the pinned evaluation IPv4/SSH port. The operator, Docker
engine/kernel, image, gateway and worker execution boundary remain trusted.

## Configure

1. Build with `scripts/install_autoresearch_tools.sh`. Required explicit settings
   are `BASE_IMAGE` (digest-pinned Node Debian image), `BUN_VERSION`,
   `RALPH_COMMIT`, `CODEX_VERSION`, `CLAUDE_VERSION` and `IMAGE_TAG`; `--help`
   describes them. The script builds a minimal runtime and prints its image ID.
2. Install the [trusted gateway](evaluation_access/README.md) separately. Its
   absolute `authorized_keys` configuration must support automatic provisioning.
3. Copy [config.example.json](isolated_runs/config.example.json), supplying a
   unique name, provider/model/effort, immutable image ID, limits, non-root
   UID/GID, authentication and pinned evaluation endpoint. Keep `admin_command`
   and ordinary SSH credentials on the Mac. Remote Docker contexts are rejected.
4. Approve explicit snapshot/data file lists and hashes; include the example's
   required source files and only necessary train/valid inputs. Generate entries:

   ```bash
   python3 -m isolated_runs.manifest --source /absolute/approved/snapshot --files-from /private/source-allowlist.txt
   python3 -m isolated_runs.manifest --dataset --source /absolute/approved/dataset --files-from /private/data-allowlist.txt
   ```

The manifest hashes named files, without deciding what is approved. The launcher
rechecks bytes and creates a Git root with one initial commit and no remote.
Worker runtime checks must include system `glibc`/`libm`, numerical-library
versions, CPU configuration and thread settings, in addition to source and xTB
hashes. Check reference reproduction when adding a host; if it differs, exclude
that host and preserve the failed evidence rather than relax the startup gate.
The approved dataset must include `release.json`; its release ID is pinned in
the private runtime configuration. Evaluation receipts must match that release.
Data is mounted at `/dataset`, linked from `molecules`.
Each new workspace starts with an empty, uncommitted `backlog.md` and `ideas/`.
Prior backlogs, preserved implementations and decision tables cannot seed a new
run. Backlog entries link evidence and versioned candidate code; these private
artifacts persist through stop/resume and sanitized export.
The snapshot includes `scripts/cycle.py`. It reserves cycle numbers in an
uncommitted `cycle.json`, which cannot seed a fresh run. `begin` resumes an
unfinished cycle; `status` inspects it; `record --description` passes the
reservation and evaluator evidence to the existing decision writer.
If a process stops between decision-table appends, `python scripts/cycle.py
recover` replays already recorded decisions into missing secondary-table entries.
It refuses conflicting records and does not alter scores or allocate a cycle.

For Mac Docker Desktop with a remote coordinator, add `evaluation.relay`:

```json
{
  "host": "192.168.65.254",
  "port": 42391,
  "relay": {
    "ssh_target": "OPERATOR_SSH_ALIAS",
    "target_host": "127.0.0.1",
    "target_port": 22
  }
}
```

Replace `host` with the verified IPv4 of `host.docker.internal`; use a distinct
listener port per run and the coordinator's pinned host key. The Mac's private
SSH master forwards its loopback listener to the coordinator. No operator private
key, agent/control socket or unrestricted SSH credential enters the research run.
Before starting, probes verify the host IP and restricted gateway authentication
through the container network. Failure blocks startup. Stop removes the relay;
resume recreates it. Unconfirmed teardown must be resolved by retrying stop.

Start/resume also starts one host-side relay recovery process, independent of
the dashboard. It checks the owned SSH master every five seconds and replaces
it only when unavailable. Failed starts use 5/10/20/40/60-second backoff, capped
at 60 seconds; one minute of stability resets the delay. Recovery uses the
existing lifecycle lock, never submits evaluator requests and never restarts
the research container. Stop intent is saved before teardown, so an interrupted
stop cannot cause an automatic relay restart.

For an already running run created before this feature, enable recovery without
replacing a healthy relay:

```bash
python3 -m isolated_runs.launcher --root /private/ar-runs ensure-relay RUN
```

Private `RUN/control/relay-recovery.log` records timestamped restart attempts and
outcomes (64 KiB per file, two backups). Newly launched SSH masters also append
their own errors to `relay-ssh.log`, trimmed to its last 64 KiB during recovery
checks. The existing healthy master cannot acquire logging retroactively; it
gets that logging on its next restart. These logs help diagnose interruptions;
they are not required to choose when to reconnect and are not shown to the
research agent. This small process recovers SSH exits, not its own termination
or a host reboot; `ensure-relay` can start it again without restarting research.

## September 16 dataset refresh

The sole supported dataset is `molecules/releases/main-f59f2b89a-gfn2-v1`: 469 train
and 465 validation inputs. The dataset uses raw SPICE conformers and noisy
RowanSci structures, with no GFN-FF preoptimization. Reference keys are exact
XYZ stems, without `_mm`. Keep the existing reproduction gate unchanged.

Create new snapshot and dataset manifests from the chosen AR branch and this
release. A new release requires a fresh gateway state root/configuration and
installation at the same absolute path on every participating worker host.
Do not repoint an old run or reuse its anchors, gateway state or result cache.
The previous datasets were withdrawn as invalid by the dataset owner on
2026-09-16. Their payloads and exporter profiles are removed; launch and gateway
checks reject their release identities. Old experiments must remain retired.
Preserve their historical results for provenance only; do not use their anchors,
scores or caches as evidence under the current dataset. Replace the retired
service configuration with the current release when deploying the next run.

The pool was expanded on 2026-09-16 to **196 workers on four hosts**, with
49 workers on each host.
Each host has 48 physical cores and 96 logical CPUs. Use one worker per physical
core plus one worker on a distinct SMT sibling, for 49 pinned logical CPUs.
This is 196 worker processes on 192 physical cores, not 196 physical cores.
Numerical threads remain 1 and `JAX_ENABLE_X64=1` on every worker.

One worker host retains glibc 2.31. Its restored workers run Python and
xTB through isolated glibc 2.35 launchers in the private service directory.
The loaded libc/libm bytes match the other three hosts. Both launchers are
required: wrapping Python alone does not change the libraries loaded by its
xTB subprocess. The restored runtime reproduced all 934 new train/validation
references with exact force-call counts and energy ratios within `1e-9` before
joining the shared queue. Do not restart these workers with the host Python or
an unwrapped xTB executable.

The active pool was moved to a dedicated Redis instance on **port 16382,
DB 0**, prefix `validate:prepared-v1`, on 2026-09-16. The previous port 16381
also had workers on an unapproved fifth host; they consumed new-release
tasks without having its files. Do not use that shared queue for this pool.
Its state and the failed startup evidence remain preserved.

The service root is a private deployment directory. The gateway uses
`config-isolated.json` and `state-isolated/`; its diagnostic collector listens
on loopback port 18765. All four hosts record the 49 current worker PIDs,
creation times, CPU affinity and launch commands in `workers/migration.json`
under this service root. The other three hosts forward loopback port 16382 to
the Redis host through the owned `redis-ssh.sock` connection. Workers retain the
verified numerical code/runtime, including both glibc launchers on the affected host.

The current dataset is installed at the same absolute path on all four hosts. Old release payloads
have been removed from the previous service deployments and stopped local
experiments. Recheck all 196 worker processes, their Redis server identity,
collector access and dataset hashes at launch. Before submitting work, an idle
Redis connection census should match the 196 intended workers plus the operator
probe; counting processes on only four hosts cannot detect other queue consumers.

The coordinator wait window is 3600 seconds so a complete split can finish in
one session. This does not change optimizer limits: each worker still uses a
2100-second task wall limit, 8 GiB RSS limit, one numerical thread and x64.
Create a fresh run and pass both baseline checks before research begins.

## Lifecycle

To rename a running experiment in the dashboard, set its private
`control/state.json` field `display_name` to the desired run-style label.
The dashboard and launcher status expose this label while the directory name,
run ID, evaluation ownership and running containers remain stable. A label
change does not restart research.

Finished runs are displayed only from their hash-pinned sanitized exports.
Verify those exports before removing a dataset or its gateway configuration;
marking a run `finished` without an export hides its still-preserved results.

Use one private root (mode 700) on the Mac:

```bash
python3 -m isolated_runs.launcher --root /private/ar-runs create --config /private/run.json
python3 -m isolated_runs.launcher --root /private/ar-runs auth RUN
python3 -m isolated_runs.launcher --root /private/ar-runs start RUN
python3 -m isolated_runs.launcher --root /private/ar-runs status RUN
python3 -m isolated_runs.launcher --root /private/ar-runs stop RUN
python3 -m isolated_runs.launcher --root /private/ar-runs resume RUN
python3 -m isolated_runs.launcher --root /private/ar-runs export RUN
python3 -m isolated_runs.launcher --root /private/ar-runs finish RUN
```

`create` provisions files and the run key without starting research. Interactive
`auth` logs into the selected provider in the private home. Alternatively use
`auth: {"kind":"api_key","credential_file":"/private/run-key"}` with a mode-600
key file. The two `launch_local_*_remote_validate.sh` wrappers accept this same
CLI and check the provider.

`start` evaluates fresh train/valid starting anchors for this run. Research
requires both complete and valid, with zero internal errors and matching source,
commit and release provenance; then it records cycle 1 and starts Ralph.
Before recording cycle 1, it also requires every per-molecule and aggregate
force-call ratio to equal 1 and energy recovery to be within `1e-9` of 1.
Pending, invalid or reference-mismatching anchors block research.
`anchors/*.json` retains the evidence. These checks establish call-count and
reference-relative final-energy agreement, not identical trajectories or an
independent measurement of the starting energies.

`stop` first saves `stopping`, then pauses submissions, cancels this run's queued tasks and removes its
containers/relay; already claimed numerical work may finish. Missing service or
teardown acknowledgement leaves `stopped_unconfirmed`, blocking resume/export
until stop succeeds. `resume` enables the same run and recovers its unfinished
evaluation IDs before research; its workspace and memory persist.

For an interrupted Claude conversation, the operator may set
`claude_resume_session` in `RUN/control/runtime.json` to its exact session UUID
before `resume`. This passes Claude's `--resume` option; omit it for fresh
conversations. Verify the session belongs to that run's `/workspace`.
Optional `resources.cpuset_cpus` (for example `"0,1"`) pins the local agent's
CPU affinity as well as its existing CPU quota and survives container recreation.
Runtime builds apply the guarded Ralph stream-cancellation patch to prevent
retaining a Bun backing buffer per output chunk.


`export` requires a stopped/created/finished run. It writes a private sanitized
archive of scientific artifacts, memory and diagnostics with a manifest;
credentials, host configuration, Git objects and symlinks are excluded.
`finish` stops, exports, revokes the evaluation key and removes mounted secrets.
Incomplete cleanup stays retryable as `finishing`. Provider credential revocation
is separate. Finished exports retain sanitized logs.

For `purpose: verification`, fresh anchors need complete provenance and zero
internal errors; scientific invalidity remains explicitly recorded. Ralph and
research decision tables are not created. At `verification_waiting_for_permit`:

```bash
python3 -m isolated_runs.launcher --root /private/ar-runs verification-permit RUN
```

The host confirms the anchors and pauses submissions before one bounded provider
session reads its own evidence and a public primary source. `algo.py` stays
read-only. Permit and provider limits are ten minutes each; a new attempt needs
a fresh disposable run.

## Dashboard and logs

```bash
python3 -m isolated_runs.monitor --root /private/ar-runs --port 8765 --gateway-status
```

Open [the dashboard](http://127.0.0.1:8765/). Each card starts with model, lifecycle
and progress chart; per-cycle statistics appear on marker hover/keyboard focus.
Run summaries remain visible and operational details expand below the chart.
Training/validation champion lines use verified decision records; pending,
unverified and error records carry no scientific score. Empty history stays empty.

Buttons open `results.tsv`, `generalizable.tsv` and `non_generalizable.tsv`.
**Full log** and **Backlog** open `full_log.md` and `backlog.md` in the same
reading panel, refreshing automatically while preserving scroll position.
Notes use the redacted 256 KiB log view and flag omitted earlier content.
**All evidence** reveals every column; raw view and **Download TSV** retain the
redacted source within the 4 MiB/5,000-row bound. The viewer flags malformed or
truncated data and preserves selection/scroll positions across refreshes.
Page refresh is five seconds; timestamped gateway snapshots refresh every thirty.

Logs start collapsed. Ralph output in `RUN/workspace/.ralph/logs/` retains up to
ten 50 MiB segments across restarts; verification uses
`RUN/workspace/verification/provider.log`. Stop archives available Docker output
under `RUN/operator-logs/`. Finished views use only the verified sanitized export.
Stopping the read-only localhost dashboard leaves saved logs intact.
The log selector also opens the run's current `backlog.md`.

## Verification

```bash
python3 -m pytest tests/isolated_runs/test_launcher.py tests/test_evaluation_access.py -q
python3 -m isolated_runs.verify_isolation --root /private/ar-runs FIRST SECOND
```

The Docker harness needs two created, never-started disposable runs. It probes
filesystem/network separation, hardening, gateway routing and stop independence
without launching an LLM or numerical work. FIRST ends stopped, SECOND created.
Filesystem canaries do not establish provider memory isolation. Record actual
check outputs separately; these instructions do not assert successful deployment
or valid scientific anchors.
