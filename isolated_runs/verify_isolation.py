"""Opt-in real Docker boundary verification using two CREATED disposable runs.

This starts only sleep/probe containers, never an LLM or a scientific evaluation.
It makes no claims about provider memory unless an authenticated LLM check is
performed separately. Gateway ownership/failure tests live in its test suite.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import secrets

from isolated_runs.launcher import Launcher, LaunchError


PROBE = r'''
import json, os, pathlib, socket, urllib.request
cfg=json.loads(os.environ["PROBE"])
assert os.geteuid()!=0
for path in (cfg["other_root"], cfg["host_canary"], "/var/run/docker.sock", "/host", "/root/.ssh"):
    try:
        pathlib.Path(path).read_bytes()
    except (FileNotFoundError, PermissionError, IsADirectoryError):
        pass
    else:
        raise AssertionError("host/other-run path is visible: "+path)
assert pathlib.Path("/memory/isolation-canary").read_text()==cfg["own_secret"]
assert cfg["other_secret"] not in pathlib.Path("/memory/isolation-canary").read_text()
assert not pathlib.Path("/workspace/results.tsv").exists()
assert not pathlib.Path("/home/research/.codex/auth.json").exists()
status=pathlib.Path("/proc/self/status").read_text()
assert "CapEff:\t0000000000000000" in status and "NoNewPrivs:\t1" in status
try:
    pathlib.Path("/usr/local/isolation-write-check").write_text("must fail")
except (OSError, PermissionError):
    pass
else:
    raise AssertionError("root filesystem is writable")
for host,port in cfg["blocked"]:
    try:
        sock=socket.create_connection((host,port),timeout=2)
    except OSError:
        continue
    sock.close()
    raise AssertionError("forbidden network target is reachable: "+host+":"+str(port))
assert urllib.request.urlopen("https://example.com",timeout=20).status==200
print(json.dumps({"nonroot":True,"host_and_other_run_hidden":True,"own_memory_only":True,
                  "fresh_results_and_codex_home":True,"rootfs_readonly":True,"caps_dropped":True,
                  "private_network_targets_blocked":True,"public_https":True}))
'''


def confirm_listener(launcher: Launcher, container: str) -> None:
    script = r'''
import socket,time
for attempt in range(50):
    try:
        connection=socket.create_connection(("127.0.0.1",8765),timeout=.2)
        connection.close()
        print("LISTENER_READY",flush=True)
        break
    except OSError:
        time.sleep(.1)
else:
    raise SystemExit("positive-control listener failed")
'''
    result = launcher.docker("exec", container, "python3", "-c", script)
    if result.stdout.strip() != "LISTENER_READY":
        raise LaunchError("cross-run network test requires a confirmed listening positive control")


def verify(launcher: Launcher, names: list[str]) -> dict:
    if len(set(names)) != 2:
        raise LaunchError("two different disposable run names are required")
    states, configs = [], []
    for name in names:
        state = launcher.state(name)
        if state["status"] != "created":
            raise LaunchError("verify requires two CREATED, never-started disposable runs")
        cfg = json.loads((launcher.path(name) / "control/config.json").read_text())
        launcher.preflight(cfg["image"])
        states.append(state)
        configs.append(cfg)
    host_canary = launcher.root / (".host-canary-" + secrets.token_hex(8))
    host_canary.write_text(secrets.token_hex(32))
    own = [secrets.token_hex(32), secrets.token_hex(32)]
    results = {}
    try:
        for i, name in enumerate(names):
            launcher.remove_containers(states[i])
            launcher.start_network(states[i], configs[i])
            (launcher.path(name) / "memory/isolation-canary").write_text(own[i])
            command = launcher.agent_command(name, states[i], configs[i])
            command[command.index("--entrypoint") + 1] = "sleep"
            command[-1] = "infinity"
            launcher.docker(*command)
        for i, name in enumerate(names):
            other_spec = json.loads(launcher.docker("inspect", states[1-i]["network_container"]).stdout)[0]
            own_spec = json.loads(launcher.docker("inspect", states[i]["network_container"]).stdout)[0]
            other_ip = next(iter(other_spec["NetworkSettings"]["Networks"].values()))["IPAddress"]
            gateway = next(iter(own_spec["NetworkSettings"]["Networks"].values()))["Gateway"]
            # Establish an actual listening service in B so rejecting its IP
            # tests the boundary, rather than merely testing a closed port.
            launcher.docker("exec", "--detach", states[1-i]["container"], "python3", "-m", "http.server", "8765")
            confirm_listener(launcher, states[1-i]["container"])
            probe = {"other_root": str(launcher.path(names[1-i]) / "memory/isolation-canary"),
                     "host_canary": str(host_canary), "own_secret": own[i], "other_secret": own[1-i],
                     "blocked": [(other_ip, 8765), (gateway, 22), (configs[i]["evaluation"]["host"], 6379),
                                 (configs[i]["evaluation"]["host"], 6380),
                                 ("169.254.169.254", 80)]}
            if "relay" in configs[1-i]["evaluation"]:
                other_ev = configs[1-i]["evaluation"]
                probe["blocked"].append((other_ev["host"], other_ev["port"]))
            output = launcher.docker("exec", "--env", "PROBE=" + json.dumps(probe),
                                     states[i]["container"], "python3", "-c", PROBE)
            results[name] = json.loads(output.stdout)
        launcher.stop(names[0])
        other_running = launcher.docker("inspect", "--format", "{{.State.Running}}", states[1]["container"]).stdout.strip()
        other_gateway = launcher.admin(names[1], "status")
        other_https = launcher.docker("exec", states[1]["container"], "python3", "-c",
            'import urllib.request; print(urllib.request.urlopen("https://example.com",timeout=20).status)').stdout.strip()
        if other_running != "true" or other_gateway.get("state") != "enabled" or other_https != "200":
            raise LaunchError("stopping the first run did not preserve the second run's independent operation")
        if "relay" in configs[1]["evaluation"]:
            if launcher.relay(states[1], configs[1]).status()["state"] != "running":
                raise LaunchError("stopping the first run disrupted the second run's SSH relay")
            launcher.verify_relay_endpoint(states[1], configs[1], existing_container=True)
        return {"verification": "real-docker", "runs": results, "scientific_evaluations": 0,
                "stop_one_other_alive": {"stopped_run": names[0], "unaffected_run": names[1],
                                         "other_container_running": True, "other_gateway_enabled": True,
                                         "other_public_https": True},
                "provider_memory_check": "not performed; requires authenticated separate LLM canary check"}
    finally:
        cleanup_errors = []
        for i, name in enumerate(names):
            try:
                launcher.remove_containers(states[i])
            except Exception:
                cleanup_errors.append(f"{name}: container removal unconfirmed")
            try:
                launcher.stop_relay(states[i])
            except Exception:
                cleanup_errors.append(f"{name}: SSH relay removal unconfirmed")
            try:
                (launcher.path(name) / "memory/isolation-canary").unlink(missing_ok=True)
            except Exception:
                cleanup_errors.append(f"{name}: private canary cleanup failed")
        try:
            host_canary.unlink(missing_ok=True)
        except Exception:
            cleanup_errors.append("host canary cleanup failed")
        if cleanup_errors:
            raise LaunchError("; ".join(cleanup_errors))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("names", nargs=2)
    args = p.parse_args()
    launcher = Launcher(args.root)
    names = sorted(args.names)
    with launcher.lock(names[0]), launcher.lock(names[1]):
        print(json.dumps(verify(launcher, args.names), indent=2))


if __name__ == "__main__":
    main()
