"""Operator-only adapter; this module and SSH identity stay outside the agent."""
from __future__ import annotations
import argparse
import shlex
import subprocess


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", required=True)
    p.add_argument("--server", required=True)
    p.add_argument("--python", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--identity")
    p.add_argument("--local", action="store_true")
    p.add_argument("action", choices=("provision", "set-state", "status"))
    args, rest = p.parse_known_args()
    command = [args.python, "-I", args.server, args.action, "--config", args.config, *rest]
    if not args.local:
        ssh = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ClearAllForwardings=yes",
               "-o", "ForwardAgent=no", "-o", "ConnectTimeout=15"]
        if args.identity:
            ssh += ["-o", "IdentitiesOnly=yes", "-i", args.identity]
        command = [*ssh, args.host, shlex.join(command)]
    return subprocess.run(command).returncode


if __name__ == "__main__":
    raise SystemExit(main())
