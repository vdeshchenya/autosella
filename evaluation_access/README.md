# Restricted evaluation gateway

Each run receives one SSH key with `restrict` and a forced command bound to its
run ID. Requests can select source and train/valid, but cannot select shell
commands, filesystem paths, Redis, evaluator code or another owner. Nonempty
`SSH_ORIGINAL_COMMAND` is rejected. Candidate source is parsed without importing
it on the coordinator, then executed by the worker's optimization child.
Worker containment remains a separate security boundary.

## Trusted operator setup

Install the service in an operator-owned checkout that agents cannot edit, with
an immutable approved release available at matching worker paths. Example config:

```json
{
  "state_root": "/srv/evaluation/state",
  "molecules_dir": "/srv/evaluation/releases/APPROVED/molecules",
  "release_id": "APPROVED-RELEASE-ID",
  "redis_host": "localhost",
  "redis_port": 6379,
  "redis_db": 0,
  "allowed_splits": ["train", "valid"],
  "max_source_bytes": 1048576,
  "task_timeout_seconds": 60,
  "authorized_keys": "/home/evaluation/.ssh/authorized_keys"
}
```

The configured release identity must match the installed `release.json`.
Each gateway state root is bound to one release; use a fresh state root for a
new release or an unbound legacy database with existing jobs. Run requests and
returned receipts must match the run's approved release identity.
`task_timeout_seconds` is a retained
work wait window, not an optimizer kill limit. Prepared workers use Redis prefix
`validate:prepared-v1`; install them from a separate fixed checkout, verify
capacity and `JAX_ENABLE_X64=1`, and preserve running workers' code.

```bash
python -I /srv/evaluation/code/evaluation_access/server.py --config /srv/evaluation/config.json provision --run-id RUN --public-key /private/run.pub
python -I /srv/evaluation/code/evaluation_access/server.py --config /srv/evaluation/config.json status --run-id RUN
python -I /srv/evaluation/code/evaluation_access/server.py --config /srv/evaluation/config.json set-state --run-id RUN --state paused
```

Provision atomically installs the tagged key and returns
`authorized_key_installed: true`; repeated provisioning must use the same run/key.
States are `enabled`, `paused`, `revoked`. Pause withdraws only owned queued
tasks; claimed work may finish. Revoke also removes the run key. Enabling permits
explicit resume without submitting automatically. Host administration and
ordinary SSH credentials stay outside research containers.

## Client and recovery

The launcher supplies `EVALUATION_HOST`, `EVALUATION_PORT`, `EVALUATION_KEY`,
`EVALUATION_KNOWN_HOSTS` and `EVALUATION_EXPECTED_RELEASE_ID`. The client sends
`expected_release_id` with each request and checks the returned release identity.
The client pins host keys, disables SSH agents/shared
connections/forwarding/user config, and sends no remote shell command.

```bash
python evaluation_access/client.py evaluate --program algo.py --split train --request-id cycle-N-train --output-json evaluation_results/cycle-N-train.json
python evaluation_access/client.py results --evaluation-id ID
python evaluation_access/client.py resume --evaluation-id ID
python evaluation_access/client.py cancel --evaluation-id ID
```

`evaluate` saves its idempotency token before submission. Repeating the same
source/split/commit/request/endpoint reconnects to the same evaluation; preserve
client state. Source hash identifies content and Git HEAD supplies commit
provenance. After `--wait-seconds` (default 600), the client saves pending JSON
and exits 75. A transport failure before any server receipt has no evaluation ID
and cannot serve as scientific evidence.

`resume` dispatches and returns; poll `results`. Wait for `cancelling` to settle
to `cancelled` or `complete` before resuming. Paused runs refuse new submissions
and resumes. Recovery preserves successful tasks, retries only affected
infrastructure tasks with at most three total attempts per molecule (initial
attempt plus two retries), and retains evaluation/program IDs.
Complete internal optimizer errors remain scientific failures; infrastructure
interruptions stay pending/unscored. Use [program.md](../program.md) for decisions.
Candidate execution that exceeds the per-task RSS budget or raises `MemoryError`
during optimization is a terminal numerical failure (`optimizer_error`). It
invalidates the candidate without scoring a partial trajectory or retrying the
molecule. Recovery also recognizes explicit RSS-guard receipts from older
workers and pending jobs, retaining the original diagnostics and recording the
classification change. Unknown process deaths remain infrastructure failures.

Redis task inputs, states and results expire after 12 hours (43,200 seconds).
Submission, claim, withdrawal/resume and completion establish a fresh retention
window; owned running heartbeats renew inputs and state so active work remains
available. Read-only polling and duplicate submissions/results do not extend retention.
Downloaded results remain in durable evaluation files after Redis expiry.
Resuming a submitted task whose Redis records expired leaves it pending for
operator diagnosis, without silently rerunning it. This also applies to an
interrupted submission whose acknowledgement was never recorded. Resume or
collect pending work within the retention window; queue names and worker lease
durations are unchanged.

The wire API accepts one JSON request on stdin: `submit` takes `request_id`,
`source`, `split`, `candidate_commit`; `status`, `results`, `resume`, `cancel`
take `evaluation_id`. Responses contain `ok` and `result` or `error`. Request IDs
cannot be rebound to different content; lookups are scoped to the forced run ID.
SQLite persists ownership/idempotency; per-job artifacts and an execution lock
support detached coordination and recovery without clearing shared queues.

Offline checks: `python -m pytest tests/test_evaluation_access.py -o addopts= -q`.
They use temporary state and fake transports, without installing live keys or
launching numerical work.

## Optional passive diagnostics

The Astra research prompt uses saved failure evidence to guide bounded repair
attempts. This is an optional addition to ordinary evaluations, with no new
numerical job type. Interface checks and selected-case replays remain deferred.

The guarded worker keeps a fixed shared-memory file journal: the last eight
completed position/force frames, up to 200 scalar rows, and the latest attempted
input, written before entering the calculator. Scalars include attempted and
completed force-call counts, energies, elapsed time, and the existing convergence
metrics. Source tracebacks identify the exact candidate hash. The parent reads
the journal after child termination, including timeout/RSS-limit/crash paths,
before deleting temporary files. This survives a child failure, not loss of the
whole worker host. Internal optimizer state is not instrumented.

Only errors and nonconverged outcomes retain a bundle; converged successes drop
their buffers. Nonconvergence remains diagnostic and does not add an acceptance
gate. Geometry capture is limited to 2,048 atoms; larger inputs retain scalar
evidence with explicit omissions. Each serialized bundle is capped at 1 MiB,
including its identity envelope; size trimming removes whole frames and records
the omission. There are no extra calculator calls, changes to convergence
thresholds, or changes to scoring and recovery policy. Capture adds bounded work
inside the wall-clock limit, so production timing parity still needs checking
before drawing performance conclusions from an enabled run.

The worker parent uploads bundles to a separate trusted collector. The collector
shares the gateway's state directory; workers do not need a shared filesystem.
It checks the run, evaluation, candidate source, molecule and task attempt against
the durable evaluation state, then stores immutable, hash-addressed files in:

```
<state_root>/diagnostics/<run_id>/<evaluation_id>/<artifact_sha256>.json
```

Only small descriptors go through Redis and into `failure_details` in evaluation
receipts. Stored bundles never travel through Redis. The default service quota
is 1 GiB per run, including metadata. Bundles expire twelve hours after collector
ingestion by default (`diagnostics_retention_seconds: 43200`). The collector
sweeps even idle runs at startup and at most every 60 seconds while serving, and
also cleans before uploads. When an upload would exceed the run quota, the
collector removes the oldest bundles in that run until it fits. A bundle larger
than the quota is rejected without evicting existing bundles. Payloads and their
metadata are removed together; empty evaluation directories are removed too.

Downloads and duplicate uploads of an existing bundle do not renew its lifetime.
For bundles predating this policy, age starts at the payload file modification
time. Once deleted, a later upload is a new ingestion; there are no permanent
tombstones. Existing bundles older than the limit are therefore removed when
the updated collector starts. This policy covers collector copies only: files
already downloaded by clients remain in their chosen download directories.

Failed uploads use an operator-owned
worker spool, capped at 64 MiB overall and 16 MiB per run; retries attempt at most
one pending bundle per minute outside calculations. Storage or upload failures
produce explicit unavailable/pending receipts without changing numerical results.
A later successful upload appears in a fresh manifest; an older evaluation
receipt may still say `upload_pending`. Likewise, old receipts may say `stored`
after expiry or eviction; a fresh manifest lists the artifacts currently held.
Chunk reads are locked against cleanup, but a bundle can expire between chunk
requests. An unavailable-artifact error means the client should refresh the
manifest; downloads do not pin artifacts. Archive useful bundles before expiry.
The gateway never deletes files or launches work when diagnostics are read;
collector cleanup is independent of Redis expiry and evaluation results.

To enable this on a newly deployed trusted service, add the following to its
config (the token file must contain a private random ASCII token of at least
32 characters):

```json
{
  "diagnostics_token_file": "/private/evaluation-diagnostics.token",
  "diagnostics_max_run_bytes": 1073741824,
  "diagnostics_retention_seconds": 43200
}
```

Start the optional collector explicitly from the trusted checkout:

```bash
python -I /srv/evaluation/code/evaluation_access/diagnostics.py --config /srv/evaluation/config.json --bind 127.0.0.1 --port 8765
```

On each trusted worker host, configure all three environment variables before
starting workers from the updated checkout:

```bash
export GIGAOPT_DIAGNOSTICS_URL=http://127.0.0.1:8765
export GIGAOPT_DIAGNOSTICS_TOKEN_FILE=/private/evaluation-diagnostics.token
export GIGAOPT_DIAGNOSTICS_SPOOL_DIR=/var/lib/evaluation/diagnostic-spool
```

For remote workers, the localhost URL must reach the collector through an
operator-managed tunnel; an HTTPS reverse proxy is also supported. Plain HTTP
uploads to a non-loopback host and redirects are rejected. Keep the token and
collector access outside research containers. No deployment, tunnel, worker
restart, or collector startup happens merely by checking out this branch.
Rebuild the isolated research image so its installed client includes the new
command; copy the updated research prompts into each new run's approved snapshot.

The research agent reads artifacts through its existing restricted SSH client:

```bash
python evaluation_access/client.py diagnostics --evaluation-id ID
python evaluation_access/client.py diagnostics --evaluation-id ID --molecule NAME
python evaluation_access/client.py diagnostics --evaluation-id ID --molecule NAME --artifact-id SHA256
```

The first commands save a bounded manifest under `diagnostics/ID/`; use its
`next_offset` with `--offset` for another page (`--limit` defaults to 32, maximum
64 entries). The last command downloads one bundle in bounded chunks, checks
ownership/provenance consistency and its SHA-256, and atomically saves
`diagnostics/ID/SHA256.json`. `--output-dir` changes the destination. Console
output contains a summary, not full geometries. These commands do not submit,
resume, reconcile, or score jobs, and can read a paused run's stored evidence.
The agent has no direct worker filesystem, collector credential, or Redis access.

Offline coverage includes trace-on/off numerical parity with deterministic test
calculators, first-call and caught failures, worker interruption, strict bundle
and storage caps, upload failure, ownership and read-only gateway behavior,
chunk integrity, and safe local file output. Tests use temporary state and mock
calculators; the collector tests may use a local loopback HTTP server.
