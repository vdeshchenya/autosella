"""Reconnectable JSON-over-SSH evaluation client; sends source data only."""
from __future__ import annotations

import argparse
import base64
import binascii
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
import uuid

MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_DIAGNOSTIC_RESPONSE_BYTES = 128 * 1024
MAX_DIAGNOSTIC_BYTES = 1024 * 1024
MAX_DIAGNOSTIC_CHUNK_BYTES = 65536
MAX_DIAGNOSTIC_PAGE_ENTRIES = 64


class TransportError(RuntimeError):
    """A reconnectable failure, not an optimizer error."""


def check_release_identity(result, expected_release_id):
    if expected_release_id is not None and (
            not isinstance(result, dict) or result.get("release_id") != expected_release_id
            or result.get("dataset_release_id") != expected_release_id):
        raise ValueError("Evaluation receipt does not identify the approved dataset release")


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            os.chmod(temporary, 0o600)
            json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def stdout_summary(result, output_json: Path):
    """Keep aggregate outcomes visible without duplicating per-molecule data."""
    fields = (
        "status", "service_status", "evaluation_id", "run_id", "program_id", "source_sha256",
        "candidate_commit", "split", "release_id", "dataset_release_id", "timestamp", "method",
        "fitness", "mean_rel_steps", "mean_rel_energy", "max_final_energy_delta_kcal_mol",
        "converged", "converged_fraction", "converged_count", "is_valid", "invalid_reason",
        "validity_reason", "internal_error_count", "stop_reason_counts", "duration_s",
        "molecule_count", "num_results", "num_errors", "infrastructure_retries",
        "infrastructure_error", "pending_reason", "progress",
    )
    return {**{field: result[field] for field in fields if field in result},
            "evaluation_log": str(output_json.resolve())}


def resolve_candidate_commit(commit):
    if commit is None:
        git = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, check=False)
        if git.returncode:
            diagnostic = git.stderr.strip() or f"git exited with status {git.returncode}"
            raise ValueError("Cannot infer --candidate-commit from Git HEAD; fix repository access "
                             f"or provide a full commit ID. {diagnostic}")
        commit = git.stdout.strip()
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        raise ValueError("--candidate-commit must be a full hexadecimal Git commit ID")
    return commit


def ssh_command(host, key, *, port=22, jump_host=None, known_hosts=None):
    pattern = r"(?:[A-Za-z0-9_.-]+@)?[A-Za-z0-9][A-Za-z0-9_.:-]*"
    if not isinstance(host, str) or host.startswith("-") or not re.fullmatch(pattern, host):
        raise ValueError("EVALUATION_HOST must be a fixed SSH host or user@host")
    if not 0 < int(port) < 65536:
        raise ValueError("Invalid evaluation SSH port")
    key_path = Path(key).expanduser().resolve()
    if not key_path.is_file():
        raise ValueError("Evaluation run key is missing")
    command = ["ssh", "-F", "/dev/null", "-T", "-p", str(port), "-i", str(key_path),
               "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "IdentityAgent=none",
               "-o", "StrictHostKeyChecking=yes", "-o", "ClearAllForwardings=yes",
               "-o", "ControlMaster=no", "-o", "ControlPath=none", "-o", "ConnectionAttempts=1",
               "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2"]
    if known_hosts:
        command += ["-o", f"UserKnownHostsFile={Path(known_hosts).expanduser().resolve()}"]
    if jump_host:
        if jump_host.startswith("-") or not re.fullmatch(pattern, jump_host):
            raise ValueError("Invalid evaluation jump host")
        command += ["-J", jump_host]
    return command + [host]


class Client:
    def __init__(self, command, *, request_timeout=90, expected_release_id=None):
        self.command = command
        self.request_timeout = request_timeout
        if expected_release_id is not None and (
                not isinstance(expected_release_id, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", expected_release_id)):
            raise ValueError("Invalid expected dataset release identity")
        self.expected_release_id = expected_release_id

    def request(self, request):
        if self.expected_release_id is not None:
            request = {**request, "expected_release_id": self.expected_release_id}
        body = json.dumps(request, allow_nan=False).encode()
        if len(body) > MAX_REQUEST_BYTES:
            raise ValueError("Request exceeds 2 MiB")
        try:
            process = subprocess.run(self.command, input=body, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, timeout=self.request_timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise TransportError(f"SSH evaluation connection interrupted: {exc}") from exc
        response_limit = (MAX_DIAGNOSTIC_RESPONSE_BYTES if request.get("method") == "diagnostics"
                          else MAX_RESPONSE_BYTES)
        if len(process.stdout) > response_limit:
            raise TransportError("Evaluation response exceeds size limit")
        try:
            response = json.loads(process.stdout)
        except (ValueError, UnicodeError) as exc:
            raise TransportError(f"Invalid SSH evaluation response (exit {process.returncode})") from exc
        if not isinstance(response, dict):
            raise TransportError("Invalid evaluation response object")
        if response.get("ok") is not True:
            raise RuntimeError(f"Evaluation service rejected request: {response.get('error', 'unknown error')}")
        if process.returncode:
            raise TransportError(f"SSH connection failed after response (exit {process.returncode}); reconnect with same ID")
        result = response["result"]
        check_release_identity(result, self.expected_release_id)
        return result


def _diagnostic_identifier(value, name, *, sha256=False):
    pattern = r"[0-9a-f]{64}" if sha256 else r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}"
    if not isinstance(value, str) or not re.fullmatch(pattern, value):
        raise ValueError(f"Invalid diagnostic {name}")
    return value


def _diagnostic_integer(value, name, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"Invalid diagnostic {name}: expected {lower} through {upper}")
    return value


def _diagnostic_provenance(result, evaluation_id, client, expected=None):
    if not isinstance(result, dict):
        raise ValueError("Invalid diagnostic response object")
    if len(json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")) > MAX_DIAGNOSTIC_RESPONSE_BYTES:
        raise ValueError("Diagnostic response exceeds size limit")
    if result.get("evaluation_id") != evaluation_id:
        raise ValueError("Diagnostic evaluation identity mismatch")
    provenance = {name: result.get(name) for name in (
        "evaluation_id", "run_id", "source_sha256", "release_id", "dataset_release_id")}
    for name, value in provenance.items():
        _diagnostic_identifier(value, name, sha256=name == "source_sha256")
    if provenance["release_id"] != provenance["dataset_release_id"]:
        raise ValueError("Diagnostic dataset release identity mismatch")
    check_release_identity(result, getattr(client, "expected_release_id", None))
    if expected is not None and provenance != expected:
        raise ValueError("Diagnostic provenance changed between chunks")
    return provenance


def _diagnostic_identity(identity, provenance, molecule):
    fields = {"run_id", "evaluation_id", "task_id", "source_sha256", "mol_name"}
    if not isinstance(identity, dict) or set(identity) != fields or any(
            not isinstance(value, str) or not value or len(value) > 512
            or any(ord(character) < 32 for character in value)
            for value in identity.values()):
        raise ValueError("Invalid diagnostic artifact identity")
    for name in ("run_id", "evaluation_id", "source_sha256"):
        if identity[name] != provenance[name]:
            raise ValueError(f"Diagnostic artifact {name} identity mismatch")
    if molecule is not None and identity["mol_name"] != molecule:
        raise ValueError("Diagnostic molecule identity mismatch")
    return identity


def _diagnostic_atomic_write(output_dir, evaluation_id, filename, payload):
    """Walk directories without following links; publish only a validated file."""
    destination = Path(os.path.abspath(Path(output_dir).expanduser())) / evaluation_id
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory = os.open(destination.anchor, directory_flags)
    temporary = f".{filename}.{uuid.uuid4().hex}.tmp"
    created = False
    try:
        for component in destination.parts[1:]:
            try:
                os.mkdir(component, mode=0o700, dir_fd=directory)
            except FileExistsError:
                pass
            child = os.open(component, directory_flags, dir_fd=directory)
            os.close(directory)
            directory = child

        def check_destination():
            try:
                existing = os.stat(filename, dir_fd=directory, follow_symlinks=False)
            except FileNotFoundError:
                return
            if not stat.S_ISREG(existing.st_mode):
                raise ValueError("Diagnostic output must be a regular file, never a symlink")

        check_destination()
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=directory)
        created = True
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        check_destination()
        os.replace(temporary, filename, src_dir_fd=directory, dst_dir_fd=directory)
        created = False
        os.fsync(directory)
    finally:
        if created:
            os.unlink(temporary, dir_fd=directory)
        os.close(directory)
    return destination / filename


def _diagnostic_json(payload):
    def unique_object(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError("Duplicate key in diagnostic JSON")
            result[name] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Non-finite number in diagnostic JSON: {value}")

    try:
        return json.loads(payload.decode("utf-8"), object_pairs_hook=unique_object,
                          parse_constant=invalid_constant)
    except (UnicodeError, RecursionError) as exc:
        raise ValueError("Invalid diagnostic JSON") from exc


def _diagnostic_small_summary(value):
    """Only bounded scalar observations belong on stdout, never trace arrays."""
    fields = ("attempted_calls", "completed_calls", "n_started", "n_completed", "n_steps",
              "n_completed_steps", "attempted", "completed", "calls_started", "calls_completed",
              "failure_kind", "stop_reason", "truncated", "capture_incomplete",
              "geometry_omitted_atom_limit", "mol_name", "task_id", "availability", "reason",
              "artifact_id", "size_bytes")
    if not isinstance(value, dict):
        return {}
    return {key: (item[:512] if isinstance(item, str) else item)
            for key in fields if key in value for item in [value[key]]
            if item is None or type(item) in (str, bool, int, float)}


def _diagnostic_capture_evidence(result, provenance, molecule):
    evidence = result.get("capture_evidence", "not_recorded")
    if evidence not in ("available", "not_recorded", "unavailable"):
        raise ValueError("Invalid diagnostic capture evidence")
    counts = result.get("capture_status_counts", {})
    if (not isinstance(counts, dict) or len(counts) > 64 or any(
            not isinstance(key, str) or len(key) > 64 or type(value) is not int or not 0 <= value < 2 ** 31
            for key, value in counts.items())):
        raise ValueError("Invalid diagnostic capture status counts")
    output = {"capture_evidence": evidence, "capture_status_counts": counts,
              "missing_capture": "unknown; absent artifacts do not establish absence of failures"}
    receipt = result.get("capture_receipt")
    if receipt is not None:
        if not isinstance(receipt, dict) or molecule is None or receipt.get("mol_name") != molecule:
            raise ValueError("Diagnostic capture receipt molecule mismatch")
        output["capture_receipt"] = _diagnostic_small_summary(receipt)
        if "identity" in receipt:
            output["capture_receipt"]["identity"] = _diagnostic_identity(receipt["identity"], provenance, molecule)
        if "summary" in receipt:
            output["capture_receipt"]["summary"] = _diagnostic_small_summary(receipt["summary"])
        if receipt.get("availability") == "missing":
            output["missing_capture"] = True
    return output


def _diagnostic_trace_summary(trace):
    fields = ("frames_dropped_for_size", "scalar_rows_dropped_for_size",
              "older_completed_frames_not_retained", "older_scalar_rows_not_retained",
              "attempted_input_omitted_for_size", "bundle_omitted_for_size")
    truncation = trace.get("truncation")
    omitted = {key: value for key in fields if isinstance(truncation, dict) and key in truncation
               for value in [truncation[key]] if type(value) in (bool, int) and value >= 0}
    for source, key in ((trace, "geometry_omitted_atom_limit"),
                        (trace.get("exception"), "traceback_truncated"),
                        (trace.get("invalid_input"), "truncated")):
        if isinstance(source, dict) and type(source.get(key)) is bool:
            omitted["invalid_input_truncated" if key == "truncated" else key] = source[key]
    truncated = trace.get("truncated") if type(trace.get("truncated")) is bool else "not_reported"
    if any(omitted.values()):
        truncated = True
    elif isinstance(truncation, dict) and truncated == "not_reported":
        truncated = False
    incomplete = trace.get("capture_incomplete")
    return {"summary": _diagnostic_small_summary(trace), "truncated": truncated, "omitted": omitted,
            "missing_capture": incomplete if type(incomplete) is bool else "not_reported"}


def diagnostics(client, *, evaluation_id, molecule=None, artifact_id=None,
                output_dir=Path("diagnostics"), offset=0, limit=32):
    """Fetch one manifest page or one bounded artifact, using read-only requests."""
    _diagnostic_identifier(evaluation_id, "evaluation_id")
    if molecule is not None and (not isinstance(molecule, str) or not molecule
                                 or len(molecule) > 512 or any(ord(c) < 32 for c in molecule)):
        raise ValueError("Invalid diagnostic molecule")
    _diagnostic_integer(offset, "offset", 0, 2 ** 31 - 1)
    _diagnostic_integer(limit, "limit", 1, MAX_DIAGNOSTIC_PAGE_ENTRIES)
    request = {"method": "diagnostics", "evaluation_id": evaluation_id}
    if molecule is not None:
        request["molecule"] = molecule
    if artifact_id is None:
        result = client.request({**request, "offset": offset, "limit": limit})
        provenance = _diagnostic_provenance(result, evaluation_id, client)
        entries = result.get("artifacts")
        if (type(result.get("schema_version")) is not int or result["schema_version"] != 1
                or not isinstance(entries, list) or len(entries) > limit):
            raise ValueError("Invalid diagnostic manifest page")
        total = _diagnostic_integer(result.get("total"), "total", 0, 2 ** 31 - 1)
        next_offset = _diagnostic_integer(result.get("next_offset"), "next_offset", 0, 2 ** 31 - 1)
        if (next_offset != min(offset + len(entries), total) or (entries and offset + len(entries) > total)
                or type(result.get("eof")) is not bool or result["eof"] != (next_offset >= total)
                or (not entries and not result["eof"])):
            raise ValueError("Inconsistent diagnostic manifest pagination")
        summaries, artifact_ids = [], set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Invalid diagnostic manifest entry")
            digest = _diagnostic_identifier(entry.get("artifact_id"), "artifact_id", sha256=True)
            if digest in artifact_ids:
                raise ValueError("Duplicate diagnostic artifact in manifest")
            artifact_ids.add(digest)
            size = _diagnostic_integer(entry.get("size_bytes"), "size_bytes", 1, MAX_DIAGNOSTIC_BYTES)
            identity = _diagnostic_identity(entry.get("identity"), provenance, molecule)
            if type(entry.get("truncated")) is not bool or not isinstance(entry.get("summary"), dict):
                raise ValueError("Invalid diagnostic manifest summary")
            summaries.append({"artifact_id": digest, "molecule": identity["mol_name"],
                              "size_bytes": size, "truncated": entry["truncated"],
                              "summary": _diagnostic_small_summary(entry["summary"])})
        capture = _diagnostic_capture_evidence(result, provenance, molecule)
        filename = "manifest.json" if offset == 0 else f"manifest-offset-{offset}.json"
        payload = json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
        path = _diagnostic_atomic_write(output_dir, evaluation_id, filename, payload)
        return {**provenance, "method": "diagnostics", "diagnostic_manifest": str(path),
                "artifacts": summaries, "page_entries": len(entries), "total_artifacts": total,
                "omitted_from_page": max(0, total - len(entries)), "next_offset": next_offset,
                "eof": result["eof"], "truncated": any(item["truncated"] for item in summaries),
                **capture}

    _diagnostic_identifier(artifact_id, "artifact_id", sha256=True)
    if molecule is None:
        raise ValueError("--molecule is required when selecting --artifact-id")
    if offset != 0 or limit != 32:
        raise ValueError("--offset and --limit apply only to manifest pages")
    request["artifact_id"] = artifact_id
    payload, provenance, identity, expected_size = bytearray(), None, None, None
    while True:
        result = client.request({**request, "offset": len(payload), "limit": MAX_DIAGNOSTIC_CHUNK_BYTES})
        provenance = _diagnostic_provenance(result, evaluation_id, client, provenance)
        chunk_identity = _diagnostic_identity(result.get("identity"), provenance, molecule)
        if identity is not None and chunk_identity != identity:
            raise ValueError("Diagnostic artifact identity changed between chunks")
        identity = dict(chunk_identity)
        size = _diagnostic_integer(result.get("size_bytes"), "size_bytes", 1, MAX_DIAGNOSTIC_BYTES)
        if expected_size is not None and size != expected_size:
            raise ValueError("Diagnostic artifact size changed between chunks")
        expected_size = size
        if result.get("artifact_id") != artifact_id or result.get("sha256") != artifact_id:
            raise ValueError("Diagnostic artifact hash identity mismatch")
        if result.get("encoding") != "base64" or not isinstance(result.get("data"), str):
            raise ValueError("Invalid diagnostic chunk encoding")
        try:
            chunk = base64.b64decode(result["data"], validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Invalid diagnostic chunk base64") from exc
        start = _diagnostic_integer(result.get("offset"), "chunk offset", 0, MAX_DIAGNOSTIC_BYTES)
        end = _diagnostic_integer(result.get("next_offset"), "chunk next_offset", 0, MAX_DIAGNOSTIC_BYTES)
        if (start != len(payload) or not 0 < len(chunk) <= MAX_DIAGNOSTIC_CHUNK_BYTES
                or end != start + len(chunk) or end > size
                or type(result.get("eof")) is not bool or result["eof"] != (end == size)):
            raise ValueError("Inconsistent diagnostic chunk offsets, size, or eof")
        payload.extend(chunk)
        if result["eof"]:
            break
    if hashlib.sha256(payload).hexdigest() != artifact_id:
        raise ValueError("Diagnostic artifact SHA-256 mismatch")
    bundle = _diagnostic_json(payload)
    if (not isinstance(bundle, dict) or type(bundle.get("schema_version")) is not int or bundle["schema_version"] != 1
            or bundle.get("identity") != identity or not isinstance(bundle.get("trace"), dict)):
        raise ValueError("Diagnostic bundle schema or identity mismatch")
    path = _diagnostic_atomic_write(output_dir, evaluation_id, f"{artifact_id}.json", payload)
    return {**provenance, "method": "diagnostics", "artifact_id": artifact_id,
            "molecule": molecule, "diagnostic_artifact": str(path), "size_bytes": len(payload),
            "sha256": artifact_id, **_diagnostic_trace_summary(bundle["trace"])}


@contextmanager
def state_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def evaluate(client, *, program: Path, split: str, state_file: Path, candidate_commit="unknown",
             request_id=None, evaluation_id=None, wait_seconds=600.0, poll_seconds=2.0,
             namespace="", output_json: Path | None = None):
    source = program.read_text(encoding="utf-8")
    if len(source.encode()) > 1024 * 1024:
        raise ValueError("Candidate source exceeds 1 MiB")
    digest = hashlib.sha256(source.encode()).hexdigest()
    identity = hashlib.sha256(json.dumps([namespace, digest, split, candidate_commit, request_id],
                                         separators=(",", ":")).encode()).hexdigest()
    result = None
    try:
        with state_lock(state_file):
            saved = json.loads(state_file.read_text()) if state_file.exists() else {"version": 1, "evaluations": {}}
            if saved.get("version") != 1:
                raise ValueError("Unsupported client state format")
            item = saved["evaluations"].setdefault(identity, {"request_id": request_id or uuid.uuid4().hex})
            if evaluation_id:
                item["evaluation_id"] = evaluation_id
            result = item.get("receipt")
            # Persist the idempotency token before networking: even a dropped
            # submit reply can be replayed without creating another evaluation.
            atomic_json(state_file, saved)
            if not item.get("evaluation_id"):
                result = client.request({"method": "submit", "request_id": item["request_id"],
                                         "source": source, "split": split, "candidate_commit": candidate_commit})
                item.update(evaluation_id=result["evaluation_id"], receipt=result)
                atomic_json(state_file, saved)
            evaluation_id = item["evaluation_id"]
        deadline = time.monotonic() + wait_seconds
        result = client.request({"method": "results", "evaluation_id": evaluation_id})
        if result["source_sha256"] != digest or result["split"] != split:
            raise ValueError("Saved evaluation does not match candidate source and split")
        if result["status"] in {"pending", "cancelled"}:
            client.request({"method": "resume", "evaluation_id": evaluation_id})
            result = client.request({"method": "results", "evaluation_id": evaluation_id})
        while result["status"] not in {"complete", "cancelled", "cancelling"}:
            if time.monotonic() >= deadline:
                break
            time.sleep(min(poll_seconds, max(0, deadline - time.monotonic())))
            result = client.request({"method": "results", "evaluation_id": evaluation_id})
            if result["status"] == "pending":
                break
    except TransportError as exc:
        if result is None:
            raise
        result = dict(result)
        result["infrastructure_error"] = str(exc)
    # A cached submit receipt can be used after a disconnect, but must still
    # belong to the run's approved release before any result file is emitted.
    check_release_identity(result, getattr(client, "expected_release_id", None))
    if result["status"] != "complete":
        result["service_status"] = result["status"]
        result["status"] = "pending"
    if output_json:
        result["evaluation_log"] = str(output_json.resolve())
        atomic_json(output_json, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("evaluate", "status", "results", "resume", "cancel", "diagnostics"))
    parser.add_argument("--program", type=Path, default=Path("algo.py"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--state-file", type=Path, default=Path(".evaluation_access/client-state.json"))
    parser.add_argument("--output-json", type=Path,
                        help="Save the full receipt here and print only a summary to stdout")
    parser.add_argument("--request-id")
    parser.add_argument("--evaluation-id")
    parser.add_argument("--molecule", help="Restrict diagnostics to an exact molecule name")
    parser.add_argument("--artifact-id", help="Download this diagnostic SHA-256 bundle; requires --molecule")
    parser.add_argument("--output-dir", type=Path, default=Path("diagnostics"))
    parser.add_argument("--offset", type=int, default=0, help="Diagnostic manifest starting entry")
    parser.add_argument("--limit", type=int, default=32, help="Diagnostic manifest entries (maximum 64)")
    parser.add_argument("--candidate-commit")
    parser.add_argument("--wait-seconds", type=float, default=600.0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args(argv)
    try:
        if args.method == "diagnostics" and args.output_json:
            raise ValueError("Diagnostics use --output-dir; --output-json is reserved for evaluation receipts")
        if args.wait_seconds < 0 or args.poll_seconds <= 0:
            raise ValueError("Wait must be nonnegative and poll interval positive")
        commit = resolve_candidate_commit(args.candidate_commit) if args.method == "evaluate" else None
        command = ssh_command(os.environ.get("EVALUATION_HOST"), os.environ.get("EVALUATION_KEY", ""),
                              port=int(os.environ.get("EVALUATION_PORT", "22")),
                              jump_host=os.environ.get("EVALUATION_JUMP_HOST"),
                              known_hosts=os.environ.get("EVALUATION_KNOWN_HOSTS"))
        client = Client(command, expected_release_id=os.environ.get("EVALUATION_EXPECTED_RELEASE_ID"))
        if args.method == "diagnostics":
            result = diagnostics(client, evaluation_id=args.evaluation_id, molecule=args.molecule,
                                 artifact_id=args.artifact_id, output_dir=args.output_dir,
                                 offset=args.offset, limit=args.limit)
        elif args.method == "evaluate":
            result = evaluate(client, program=args.program, split=args.split, state_file=args.state_file,
                              candidate_commit=commit, request_id=args.request_id, evaluation_id=args.evaluation_id,
                              wait_seconds=args.wait_seconds, poll_seconds=args.poll_seconds,
                              namespace=json.dumps(command), output_json=args.output_json)
        else:
            if not args.evaluation_id:
                raise ValueError("--evaluation-id is required")
            result = client.request({"method": args.method, "evaluation_id": args.evaluation_id})
            if args.output_json:
                atomic_json(args.output_json, result)
        printable = stdout_summary(result, args.output_json) if args.output_json else result
        print(json.dumps(printable, sort_keys=True, allow_nan=False))
        return 75 if args.method == "evaluate" and result["status"] != "complete" else 0
    except TransportError as exc:
        # No confirmed receipt means no scientific record can yet be emitted.
        diagnostic = {"status": "pending", "infrastructure_error": str(exc),
                      "state_file": str(args.state_file)}
        if args.output_json and args.method != "diagnostics":
            # Replace a possibly stale output with an explicitly incomplete
            # receipt: the decision writer rejects a missing evaluation_id.
            atomic_json(args.output_json, diagnostic)
        print(json.dumps(diagnostic), file=sys.stderr)
        return 75
    except (ValueError, RuntimeError, OSError) as exc:
        if args.output_json and args.method != "diagnostics":
            # Failed requests must not leave a prior completed receipt at the
            # path consumed by the decision writer.
            atomic_json(args.output_json, {"status": "pending", "infrastructure_error": str(exc),
                                           "state_file": str(args.state_file)})
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
