"""Container entry point: recover, establish fresh run anchors, then start Ralph."""
from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
import subprocess
import signal
import selectors
import stat
import sys
import time
import uuid
from urllib.parse import urlparse

from evaluation_access.client import Client, TransportError, atomic_json, evaluate, ssh_command

REPOSITORY_ACCESS_POLICY = Path(__file__).resolve().with_name("repository_access_policy.md")


class GateError(RuntimeError):
    pass


class RalphLog:
    """Durable bounded stdout/stderr history, independent of Docker rotation."""
    def __init__(self, workspace: Path, *, max_bytes: int = 50 * 1024 * 1024, keep: int = 10):
        self.max_bytes, self.keep = max_bytes, keep
        self.directory = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        self.output = None
        self.size = 0
        try:
            for component in (".ralph", "logs"):
                try:
                    os.mkdir(component, mode=0o700, dir_fd=self.directory)
                except FileExistsError:
                    pass
                child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=self.directory)
                os.close(self.directory)
                self.directory = child
            self.rotate()
        except BaseException:
            self.close()
            raise

    def rotate(self):
        if self.output is not None:
            os.fsync(self.output)
            os.close(self.output)
            self.output = None
        # Only our exact file names are pruned; never follow a link or inspect
        # provider homes. The directory stays private to this isolated run.
        names = []
        with os.scandir(self.directory) as entries:
            for item in entries:
                middle = item.name.removeprefix("ralph-").removesuffix(".log")
                if (item.name == f"ralph-{middle}.log" and middle.isdigit()
                        and item.is_file(follow_symlinks=False)):
                    names.append(item.name)
        for name in sorted(names)[:max(0, len(names) - self.keep + 1)]:
            os.unlink(name, dir_fd=self.directory)
        self.output = os.open(f"ralph-{time.time_ns()}.log",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self.directory)
        self.size = 0

    def write(self, data: bytes):
        while data:
            if self.size >= self.max_bytes:
                self.rotate()
            part = data[:self.max_bytes - self.size]
            offset = 0
            while offset < len(part):
                offset += os.write(self.output, part[offset:])
            self.size += len(part)
            data = data[len(part):]

    def close(self):
        if self.output is not None:
            os.fsync(self.output)
            os.close(self.output)
            self.output = None
        if self.directory is not None:
            os.close(self.directory)
            self.directory = None


def run_ralph(config: dict, env: dict, *, workspace: Path = Path("/workspace")) -> int:
    """Tee Ralph into run-local rotating files and propagate its exit/signals."""
    log = RalphLog(workspace)
    process = None
    previous = {}
    stopped_at = None
    exited_at = None

    def forward(signum, _frame):
        nonlocal stopped_at
        stopped_at = stopped_at or time.monotonic()
        if process is not None:
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass

    try:
        process = subprocess.Popen(ralph_command(config), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, env=env, start_new_session=True, cwd=workspace)
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, forward)
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while selector.get_map():
                for key, _ in selector.select(.2):
                    data = os.read(key.fileobj.fileno(), 65536)
                    if not data:
                        selector.unregister(key.fileobj)
                        break
                    log.write(data)
                    # Preserve the normal live Docker stream as well.
                    try:
                        sys.stdout.buffer.write(data)
                        sys.stdout.buffer.flush()
                    except BrokenPipeError:
                        pass
                if process.poll() is not None:
                    exited_at = exited_at or time.monotonic()
                if ((stopped_at is not None and time.monotonic() - stopped_at > 3)
                        or (exited_at is not None and time.monotonic() - exited_at > 2)):
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    # Descendants cannot hold the log pipe indefinitely.
                    if exited_at is not None and time.monotonic() - exited_at > 4:
                        break
        code = process.wait()
        return code if code >= 0 else 128 - code
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
            process.stdout.close()
        log.close()


def environment(config: dict, secrets: Path = Path("/run/secrets")) -> dict:
    env = dict(os.environ)
    ev = config["evaluation"]
    if not isinstance(config.get("expected_release_id"), str) or not config["expected_release_id"]:
        raise GateError("runtime lacks the approved dataset release identity")
    env.update(EVALUATION_HOST=f'{ev["user"]}@{ev["host"]}', EVALUATION_PORT=str(ev["port"]),
               EVALUATION_KEY=str(secrets / "evaluation_key"), EVALUATION_KNOWN_HOSTS=str(secrets / "known_hosts"),
               EVALUATION_CLIENT="/opt/autoresearch/evaluation_access/client.py", PYTHON="python3",
               EVALUATION_STATE_FILE=".evaluation_state/client.json",
               EVALUATION_EXPECTED_RELEASE_ID=config["expected_release_id"],
               GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL="/dev/null", GIT_TERMINAL_PROMPT="0",
               # Bind-mount ownership can differ in child tool processes. Trust
               # only the workspace explicitly supplied by this launcher.
               GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="safe.directory",
               GIT_CONFIG_VALUE_0="/workspace")
    if config.get("auth_kind") == "api_key":
        credential = (secrets / "provider").read_text().strip()
        if not credential or "\n" in credential:
            raise GateError("provider credential must be a single API key")
        env["OPENAI_API_KEY" if config["provider"] == "codex" else "ANTHROPIC_API_KEY"] = credential
    if config["provider"] == "claude":
        env["CLAUDE_CODE_EFFORT_LEVEL"] = config["effort"]
    return env


def recover(client: Client, evaluation_ids: list[str], output: Path, *, wait_seconds: float = 600) -> None:
    """Resume only this run's IDs; the server independently enforces ownership."""
    for evaluation_id in evaluation_ids:
        deadline = time.monotonic() + wait_seconds
        result = client.request({"method": "results", "evaluation_id": evaluation_id})
        while result["status"] == "cancelling":
            if time.monotonic() >= deadline:
                raise GateError("owned cancellation is still settling; no new research may start")
            time.sleep(2)
            result = client.request({"method": "results", "evaluation_id": evaluation_id})
        if result["status"] in {"pending", "cancelled"}:
            client.request({"method": "resume", "evaluation_id": evaluation_id})
        while result["status"] != "complete":
            if time.monotonic() >= deadline:
                raise GateError("owned evaluation remains pending; no new research may start")
            time.sleep(2)
            result = client.request({"method": "results", "evaluation_id": evaluation_id})
            if result["status"] in {"pending", "cancelled", "cancelling"}:
                raise GateError("owned evaluation needs infrastructure recovery; no new research may start")
        atomic_json(output / f"{evaluation_id}.json", result)


def check_anchor_identity(result: dict, split: str, config: dict) -> None:
    """Validate a complete receipt without interpreting scientific eligibility."""
    if result.get("status") != "complete" or result.get("split") != split:
        raise GateError(f"{split} starting anchor is pending; research remains blocked")
    if result.get("run_id") != config["run_id"]:
        raise GateError("starting anchor belongs to a different run")
    if result.get("candidate_commit") != config["initial_commit"]:
        raise GateError("starting anchor does not identify the approved initial commit")
    expected_source = config.get("initial_source_sha256")
    if (not expected_source or result.get("source_sha256") != expected_source
            or result.get("program_id") != expected_source):
        raise GateError("starting anchor does not identify the approved optimizer source")
    if not result.get("evaluation_id") or not result.get("release_id") or not result.get("timestamp"):
        raise GateError("starting anchor lacks evaluation provenance")
    expected_release = config.get("expected_release_id")
    if not expected_release or result["release_id"] != expected_release:
        raise GateError("starting anchor does not identify the approved dataset release")
    if result.get("dataset_release_id", expected_release) != expected_release:
        raise GateError("starting anchor has inconsistent dataset release provenance")
    if result.get("is_valid") not in (0, 1):
        raise GateError("starting anchor lacks an explicit scientific validity flag")


def check_anchor(result: dict, split: str, config: dict, *, require_scientific_validity: bool = True) -> None:
    check_anchor_identity(result, split, config)
    if result.get("internal_error_count") != 0:
        raise GateError(f"{split} starting anchor has internal errors; diagnose before proceeding")
    if require_scientific_validity and result["is_valid"] != 1:
        raise GateError(f"{split} starting anchor is scientifically invalid; research remains blocked")
    if require_scientific_validity:
        check_anchor_reproduction(result, split)


def check_anchor_reproduction(result: dict, split: str) -> None:
    """Starting code must reproduce references, not merely pass candidate gates.

    Equal means alone can conceal opposite per-molecule discrepancies. Calls
    are discrete and must match exactly; energy recovery allows the same 1e-9
    numerical tolerance used by the energy validity gate, on both sides of 1.
    These receipt checks do not establish identical optimization trajectories.
    """
    def fail(detail: str) -> None:
        raise GateError(f"{split} starting anchor/reference mismatch: {detail}; research remains blocked")

    def ratio_is_one(value, *, energy: bool = False) -> bool:
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value)
                and (math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-9) if energy else value == 1))

    measurements = result.get("results")
    count = result.get("molecule_count")
    if (not isinstance(measurements, list) or not measurements
            or type(count) is not int or count != len(measurements)
            or type(result.get("num_results")) is not int or result["num_results"] != count
            or result.get("num_errors") != 0 or result.get("errors") != []):
        fail("complete per-molecule evidence with zero errors is required")
    names = set()
    for measurement in measurements:
        if not isinstance(measurement, dict):
            fail("malformed per-molecule evidence")
        name = measurement.get("mol_name")
        if not isinstance(name, str) or not name or name in names:
            fail("missing or duplicate molecule identity")
        names.add(name)
        if not ratio_is_one(measurement.get("rel_steps")):
            fail(f"{name}: force-call ratio must equal 1")
        if not ratio_is_one(measurement.get("rel_energy"), energy=True):
            fail(f"{name}: energy recovery must equal 1 within 1e-9")
    if not ratio_is_one(result.get("mean_rel_steps")):
        fail("mean force-call ratio must equal 1")
    if not ratio_is_one(result.get("mean_rel_energy"), energy=True):
        fail("mean energy recovery must equal 1 within 1e-9")


def prepare_starting_anchors(client: Client, config: dict, env: dict, *,
                             program: Path = Path("/etc/starting-algo.py"),
                             workspace: Path = Path("/workspace")) -> dict:
    """Collect both complete split observations before applying research policy."""
    config["initial_source_sha256"] = hashlib.sha256(program.read_bytes()).hexdigest()
    anchors = {}
    for split in ("train", "valid"):
        result = evaluate(client, program=program, split=split,
                          state_file=workspace / ".evaluation_state/anchors.json",
                          candidate_commit=config["initial_commit"], request_id="anchor-" + split,
                          namespace=config["run_id"], output_json=workspace / f"anchors/{split}.json", wait_seconds=3600)
        # Pending infrastructure or mismatched identity is still a hard stop.
        # A complete is_valid=0 result is preserved and does not suppress the
        # other split's fresh observation.
        check_anchor_identity(result, split, config)
        anchors[split] = result
    if anchors["train"]["release_id"] != anchors["valid"]["release_id"]:
        raise GateError("starting anchors were evaluated with different releases")
    verification = config.get("purpose", "research") == "verification"
    for split, result in anchors.items():
        check_anchor(result, split, config, require_scientific_validity=not verification)
    if not verification:
        subprocess.run([sys.executable, "scripts/record_decision.py", "--cycle", "1", "--starting-anchor",
                        "--candidate-commit", config["initial_commit"], "--anchor-commit", config["initial_commit"],
                        "--train-json", "anchors/train.json", "--valid-json", "anchors/valid.json",
                        "--description", "Fresh starting anchor"], check=True, env=env, cwd=workspace)
    # Verification stores raw observations only. It never creates an accepted
    # anchor, rejection table or any other optimizer research history.
    return anchors


def repository_policy_arguments(provider: str) -> list[str]:
    """Add behavioral instructions from the image, independent of run files."""
    try:
        policy = REPOSITORY_ACCESS_POLICY.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        raise GateError("runtime repository access policy is unavailable; provider was not started") from exc
    if not policy:
        raise GateError("runtime repository access policy is empty; provider was not started")
    if provider == "codex":
        # JSON quoting encodes this text as a TOML basic string, preserving
        # newlines and quotes inside one argv value without a host shell.
        return ["--config", "developer_instructions=" + json.dumps(policy, ensure_ascii=False)]
    if provider == "claude":
        return ["--append-system-prompt", policy]
    raise GateError("unsupported repository access policy provider")


def ralph_command(config: dict) -> list[str]:
    if not config.get("model") or not config.get("effort"):
        raise GateError("model and effort must be explicit")
    command = ["ralph", "--prompt-file", "ralph_autoresearch_prompt.md",
               "--prompt-template", "ralph_autoresearch_template.md", "--agent",
               "codex" if config["provider"] == "codex" else "claude-code",
               "--model", config["model"], "--max-iterations", "0", "--completion-promise",
               "MANUAL_STOP_ONLY_" + config["run_id"], "--no-commit", "--no-questions",
               "--last-activity-timeout", "45m", "--no-allow-all", "--"]
    if config["provider"] == "codex":
        # This bypass is ONLY inside the externally restricted Docker process.
        # No host CLI is ever launched with a sandbox/approval bypass.
        command += ["--dangerously-bypass-approvals-and-sandbox", "--config",
                    'model_reasoning_effort="' + config["effort"] + '"', "--cd", "/workspace"]
    else:
        command += ["--dangerously-skip-permissions", "--effort", config["effort"]]
        # The trusted operator can reconnect an interrupted research session.
        # Use an explicit ID; never infer a conversation from another run.
        if config.get("claude_resume_session") is not None:
            session = config["claude_resume_session"]
            try:
                if not isinstance(session, str) or str(uuid.UUID(session)) != session:
                    raise ValueError("noncanonical session ID")
            except (ValueError, AttributeError, TypeError) as exc:
                raise GateError("claude_resume_session must be a canonical UUID") from exc
            command += ["--resume", session]
    command += repository_policy_arguments(config["provider"])
    return command


def verification_prompt(config: dict, anchors: dict, algo_sha256: str) -> str:
    ids = {split: anchors[split]["evaluation_id"] for split in ("train", "valid")}
    validity = {split: anchors[split]["is_valid"] for split in ("train", "valid")}
    return f'''This is one bounded environment verification session, not an optimizer research cycle.
Run ID: {config["run_id"]}. Exact provider/model/effort: {config["provider"]}/{config["model"]}/{config["effort"]}.
Leave algo.py and all scientific/protocol files unchanged. Do not commit, propose or edit an optimizer,
submit evaluations, call evaluate/submit/resume/cancel, or start Ralph/research iterations.
The gateway is paused: only status/results reads for this run's existing anchors are permitted.
These are observed starting scores, not accepted research anchors: train is_valid={validity["train"]},
valid is_valid={validity["valid"]}. Preserve these flags exactly, including any zero.
Scientific invalidity does not prevent this infrastructure check. Do not claim optimizer acceptance
or generate results.tsv, generalizable.tsv, non_generalizable.tsv or research decisions.
Use tools to complete these finite checks, then stop:
1. Read this run's anchors/train.json and anchors/valid.json and inspect the starting algo.py.
   Its required SHA256 is {algo_sha256}. Do not inspect other runs, host homes or credentials.
2. Through python evaluation_access/client.py, request both status and results for each own anchor:
   train evaluation ID {ids["train"]}; valid evaluation ID {ids["valid"]}.
   Use --evaluation-id and --output-json to save the four responses at
   verification/train-status.json, verification/train-results.json,
   verification/valid-status.json, verification/valid-results.json.
3. Use a shell/network tool to fetch one publicly accessible primary research source related to
   numerical or molecular optimization. Save a nonempty response excerpt (at most 1 MiB) to
   verification/public-source.txt. Record its actual HTTPS URL and title. Do not invent a fetch.
4. Write verification/evidence.json with exactly these fields:
   run_id, provider, model, effort, algo_sha256, anchor_evaluation_ids (train/valid mapping),
   anchor_validity (the exact train/valid is_valid flags above),
   public_source (url, title, response_file="verification/public-source.txt").
   Use the exact run/model/effort/hash/anchor identifiers above. Briefly summarize actual tool outcomes
   in verification/summary.md; distinguish observations from unverified claims. Do not copy credentials.
Finish after these checks. Do not continue research. The process is stopped after ten minutes.
'''


def verification_command(config: dict, prompt: str) -> list[str]:
    if not config.get("model") or not config.get("effort"):
        raise GateError("verification requires the explicitly selected model and effort")
    if config["provider"] == "codex":
        return ["codex", "exec", "--json", "--model", config["model"], "--dangerously-bypass-approvals-and-sandbox",
                "--config", 'model_reasoning_effort="' + config["effort"] + '"', "--cd", "/workspace",
                *repository_policy_arguments("codex"), prompt]
    if config["provider"] == "claude":
        return ["claude", "-p", prompt, "--model", config["model"], "--effort", config["effort"],
                "--dangerously-skip-permissions", "--output-format", "stream-json", "--verbose",
                *repository_policy_arguments("claude")]
    raise GateError("unsupported verification provider")


def wait_for_verification_permit(config: dict, anchors: dict, algo_sha256: str,
                                 permit_path: Path, *, wait_seconds: float = 600) -> None:
    expected = {"run_id": config["run_id"], "algo_sha256": algo_sha256, "gateway_state": "paused",
                "anchor_evaluation_ids": {split: anchors[split]["evaluation_id"] for split in ("train", "valid")},
                "anchor_validity": {split: anchors[split]["is_valid"] for split in ("train", "valid")}}
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if permit_path.is_file():
            permit = json.loads(permit_path.read_text())
            if any(permit.get(key) != value for key, value in expected.items()):
                raise GateError("trusted verification permit does not match this run and its fresh anchors")
            return
        time.sleep(min(1, max(0, deadline - time.monotonic())))
    raise GateError("verification permit was not issued within ten minutes; provider was not started")


def bounded_provider(command: list[str], env: dict, log_path: Path, *, timeout: float = 600) -> None:
    with log_path.open("x") as log:
        os.chmod(log_path, 0o600)
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except BaseException as exc:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
            if isinstance(exc, subprocess.TimeoutExpired):
                raise GateError("verification provider exceeded its finite time budget; process group stopped") from exc
            raise
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # No background tool processes remain after the session.
        if code:
            raise GateError(f"verification provider exited with status {code}; inspect this run's provider log")


def verify_once(config: dict, anchors: dict, env: dict, client: Client, *, workspace: Path = Path("/workspace"),
                permit_path: Path = Path("/run/verification-permit/permit.json")) -> None:
    output = workspace / "verification"
    output.mkdir(exist_ok=True)
    if (output / "invocation.json").exists():
        raise GateError("this disposable verification already attempted its single provider session")
    algo = workspace / "algo.py"
    initial_hash = hashlib.sha256(algo.read_bytes()).hexdigest()
    config = {**config, "initial_source_sha256": initial_hash}
    for split in ("train", "valid"):
        check_anchor(anchors[split], split, config, require_scientific_validity=False)
    if anchors["train"]["release_id"] != anchors["valid"]["release_id"]:
        raise GateError("starting observations use different releases")
    ready = {"run_id": config["run_id"], "algo_sha256": initial_hash,
             "anchor_evaluation_ids": {split: anchors[split]["evaluation_id"] for split in ("train", "valid")},
             "anchor_validity": {split: anchors[split]["is_valid"] for split in ("train", "valid")}}
    atomic_json(output / "ready.json", ready)
    atomic_json(workspace / ".run-ready.json", {**ready, "status": "verification_waiting_for_permit"})
    wait_for_verification_permit(config, anchors, initial_hash, permit_path)
    # Exclusive marker prevents an interrupted/resumed disposable run from
    # silently starting a second billable provider session.
    try:
        with (output / "invocation.json").open("x") as handle:
            json.dump({**ready, "provider": config["provider"], "model": config["model"],
                       "effort": config["effort"], "started_at": time.time()}, handle)
    except FileExistsError as exc:
        raise GateError("this disposable verification already attempted its single provider session") from exc
    prompt = verification_prompt(config, anchors, initial_hash)
    (output / "prompt.txt").write_text(prompt)
    bounded_provider(verification_command(config, prompt), env, output / "provider.log")
    if hashlib.sha256(algo.read_bytes()).hexdigest() != initial_hash:
        raise GateError("verification changed the optimizer")
    evidence = json.loads((output / "evidence.json").read_text())
    expected = {**ready, "provider": config["provider"], "model": config["model"], "effort": config["effort"]}
    if any(evidence.get(key) != value for key, value in expected.items()):
        raise GateError("verification evidence does not match the requested run/model/settings/anchors")
    source = evidence.get("public_source", {})
    url = urlparse(source.get("url", ""))
    if (url.scheme != "https" or not url.hostname or not source.get("title")
            or source.get("response_file") != "verification/public-source.txt"
            or not 0 < (output / "public-source.txt").stat().st_size <= 1024 * 1024):
        raise GateError("verification lacks a recorded public source fetch")
    for split in ("train", "valid"):
        for method in ("status", "results"):
            saved = json.loads((output / f"{split}-{method}.json").read_text())
            if saved.get("run_id") != config["run_id"] or saved.get("evaluation_id") != ready["anchor_evaluation_ids"][split]:
                raise GateError("verification gateway evidence belongs to another evaluation")
        # Independently re-read the server result; agent-written evidence alone
        # is not used as the authority for scientific anchor identity.
        actual = client.request({"method": "results", "evaluation_id": ready["anchor_evaluation_ids"][split]})
        check_anchor(actual, split, config, require_scientific_validity=False)
        if actual["is_valid"] != ready["anchor_validity"][split] or actual["release_id"] != anchors[split]["release_id"]:
            raise GateError("confirmed starting observation changed its scientific validity or release")
        atomic_json(output / f"{split}-confirmed.json", actual)
    atomic_json(output / "complete.json", {**expected, "status": "verification_complete",
        "optimizer_unchanged": True, "gateway_access": "paused_read_only", "provider_invocations": 1,
        "research_history_created": False,
        "public_source_url": source["url"], "source_content_sha256": hashlib.sha256((output / "public-source.txt").read_bytes()).hexdigest(),
        "source_fetch_evidence": "agent transcript and saved response; inspect provider.log", "completed_at": time.time()})
    atomic_json(workspace / ".run-ready.json", {**ready, "status": "verification_complete"})


def main() -> int:
    if os.geteuid() == 0 or not Path("/etc/autoresearch.json").is_file():
        raise GateError("runner requires the non-root isolated container contract")
    config = json.loads(Path("/etc/autoresearch.json").read_text())
    env = environment(config)
    os.environ.update(env)
    os.chdir("/workspace")
    # The dataset was separately approved and copied, then mounted read-only.
    molecules = Path("molecules")
    if not molecules.exists():
        molecules.symlink_to("/dataset", target_is_directory=True)
    elif not molecules.is_symlink() or str(molecules.readlink()) != "/dataset":
        raise GateError("unexpected mutable molecules path")
    client_path = Path("evaluation_access")
    if not client_path.exists():
        client_path.symlink_to("/opt/autoresearch/evaluation_access", target_is_directory=True)
    elif not client_path.is_symlink() or str(client_path.readlink()) != "/opt/autoresearch/evaluation_access":
        raise GateError("unexpected evaluation client path")
    client = Client(ssh_command(env["EVALUATION_HOST"], env["EVALUATION_KEY"],
                               port=int(env["EVALUATION_PORT"]), known_hosts=env["EVALUATION_KNOWN_HOSTS"]),
                    expected_release_id=config["expected_release_id"])
    recover(client, config.get("recover_evaluation_ids", []), Path(".evaluation_state/recovered"))
    anchors = prepare_starting_anchors(client, config, env)
    atomic_json(Path(".run-ready.json"), {"run_id": config["run_id"], "status": "ready",
        "anchor_train_evaluation_id": anchors["train"]["evaluation_id"],
        "anchor_valid_evaluation_id": anchors["valid"]["evaluation_id"]})
    if config["provider"] == "codex" and config.get("auth_kind") == "api_key":
        # Establish only this run's API-key login; stdout is intentionally not
        # archived and no host auth/cache is imported.
        login = subprocess.run(["codex", "login", "--with-api-key"], input=env["OPENAI_API_KEY"],
                               text=True, capture_output=True, env=env)
        if login.returncode:
            raise GateError("Codex per-run API credential login failed")
    if config.get("purpose", "research") == "verification":
        verify_once(config, anchors, env, client)
        return 0
    return run_ralph(config, env)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (GateError, TransportError, RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Research gate blocked: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(75)
