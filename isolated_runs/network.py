"""Install a namespace-local egress policy, then permanently discard privilege.

This runs ONLY as the init process of the run's network helper container.
Research processes never receive NET_ADMIN or access to the Docker API.
"""
from __future__ import annotations

import ipaddress
import os
import subprocess
import sys

# Includes link-local/cloud metadata, Docker/host/LAN ranges, multicast and
# special-purpose addresses. Public DNS cannot bypass destination filtering.
BLOCKED_V4 = (
    "0.0.0.0/8", "10.0.0.0/8", "100.64.0.0/10", "127.0.0.0/8",
    "169.254.0.0/16", "172.16.0.0/12", "192.0.0.0/24", "192.0.2.0/24",
    "192.168.0.0/16", "198.18.0.0/15", "198.51.100.0/24", "203.0.113.0/24",
    "224.0.0.0/4", "240.0.0.0/4",
)


def firewall_rules(endpoint: str, port: int) -> list[list[str]]:
    ip = ipaddress.IPv4Address(endpoint)
    if ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_unspecified:
        raise ValueError("evaluation endpoint must be a reachable, pinned IPv4 address")
    if not 1 <= port <= 65535:
        raise ValueError("invalid evaluation SSH port")
    rules = [["-P", "OUTPUT", "DROP"], ["-P", "INPUT", "DROP"], ["-P", "FORWARD", "DROP"]]
    rules += [["-A", "INPUT", "-m", "conntrack", "--ctstate", "ESTABLISHED,RELATED", "-j", "ACCEPT"]]
    # Agent processes may use their own loopback, but cannot reach Docker's
    # embedded DNS or host-local services through 127.0.0.11.
    rules += [["-A", "OUTPUT", "-d", "127.0.0.11", "-j", "REJECT"],
              ["-A", "OUTPUT", "-o", "lo", "-j", "ACCEPT"],
              ["-A", "INPUT", "-i", "lo", "-j", "ACCEPT"]]
    rules += [["-A", "OUTPUT", "-d", str(ip), "-p", "tcp", "--dport", str(port), "-j", "ACCEPT"]]
    # The evaluator exception is SSH-only even when its address is public;
    # Redis/admin/other ports must not fall through to public Internet access.
    rules += [["-A", "OUTPUT", "-d", str(ip), "-j", "REJECT"]]
    for cidr in BLOCKED_V4:
        rules.append(["-A", "OUTPUT", "-d", cidr, "-j", "REJECT"])
    rules.append(["-A", "OUTPUT", "-j", "ACCEPT"])
    return rules


def main() -> None:
    endpoint, port = sys.argv[1:]
    # Flush only this newly created container's tables. No host firewall access.
    for family in ("iptables", "ip6tables"):
        subprocess.run([family, "-w", "-F"], check=True)
        for chain in ("INPUT", "OUTPUT", "FORWARD"):
            subprocess.run([family, "-w", "-P", chain, "DROP"], check=True)
    for rule in firewall_rules(endpoint, int(port)):
        subprocess.run(["iptables", "-w", *rule], check=True)
    print("ISOLATED_NETWORK_READY", flush=True)
    os.execvp("setpriv", ["setpriv", "--reuid=65534", "--regid=65534", "--clear-groups",
        "--bounding-set=-all", "--inh-caps=-all", "--ambient-caps=-all",
        "--no-new-privs", "sleep", "infinity"])


if __name__ == "__main__":
    main()
