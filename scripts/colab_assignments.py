#!/usr/bin/env python3
"""List the Colab runtimes currently assigned to this account (read only: it never assigns or unassigns).

Usage: python scripts/colab_assignments.py
Exit code 0 = no runtime assigned, 3 = one or more assigned, 1 = the list could not be read.
Prints the endpoint and accelerator of each runtime; the access token is never printed.
"""
import sys

import requests

from mcp_server_colab_exec.colab_runtime import COLAB_API, _colab_headers, _strip_xssi, get_credentials


def main() -> int:
    try:
        token = get_credentials().token  # refreshed on each call
        r = requests.get(f"{COLAB_API}/tun/m/assignments", headers=_colab_headers(token),
                         params={"authuser": "0"}, timeout=30)
        r.raise_for_status()
        items = _strip_xssi(r.text).get("assignments", [])
    except Exception as exc:  # noqa: BLE001
        print(f"could not read assignments: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if not items:
        print("assigned runtimes: 0")
        return 0
    print(f"assigned runtimes: {len(items)}")
    for a in items:
        print(f"  {a.get('endpoint')}  accelerator={a.get('accelerator', '?')}")
    return 3


if __name__ == "__main__":
    sys.exit(main())
