#!/usr/bin/env python3
"""Run a local .py on a Colab GPU runtime and save the FULL stdout and the decoded archive to disk.

Usage: python scripts/run_colab_file.py <file.py> <out_dir> [--gpu L4] [--timeout 1800]

The Colab script prints one line 'ARCHIVE_B64 <crc32-hex> <base64(gzip(json))>'. It is extracted programmatically
(no manual copying), CRC-verified and written to <out_dir>/<file stem>.json. Full stdout goes to <out_dir>/<stem>.stdout.txt.
"""
import argparse
import base64
import gzip
import os
import pathlib
import tempfile
import re
import sys
import time
import zlib

import requests

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from run_colab_benchmark import execute_with_live_streaming  # noqa: E402
from mcp_server_colab_exec.colab_runtime import (  # noqa: E402
    COLAB_API,
    _colab_headers,
    _strip_xssi,
    allocate_runtime,
    create_session,
    get_credentials,
    start_keepalive,
    unassign_runtime,
)

ap = argparse.ArgumentParser()
ap.add_argument("file")
ap.add_argument("out_dir")
ALLOWED_GPUS = ("T4", "L4")  # A100 is banned by the user (2026-11-07): its compute-unit cost per hour is too high
ap.add_argument("--gpu", default="L4", type=str.upper, choices=ALLOWED_GPUS)
ap.add_argument("--timeout", type=int, default=1800)
args = ap.parse_args()

code = pathlib.Path(args.file).read_text(encoding="utf-8")
out = pathlib.Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
stem = pathlib.Path(args.file).stem

def list_endpoints(token: str):
    """Endpoints currently assigned to this account, or None if the list cannot be read."""
    try:
        r = requests.get(f"{COLAB_API}/tun/m/assignments", headers=_colab_headers(token),
                         params={"authuser": "0"}, timeout=30)
        r.raise_for_status()
        return {a.get("endpoint") for a in _strip_xssi(r.text).get("assignments", []) if a.get("endpoint")}
    except Exception as exc:  # noqa: BLE001
        print(f"[run_colab_file] could not list assignments: {exc}", file=sys.stderr)
        return None


def release(token: str, endpoint: str) -> None:
    """Unassign with retries, then confirm the endpoint is gone from the assignment list."""
    for i in range(5):
        # the access token from the start of a long job expires (~1 h): unassign then fails with 401
        try:
            token = get_credentials().token
        except Exception as exc:  # noqa: BLE001
            print(f"[run_colab_file] could not refresh credentials: {exc}", file=sys.stderr)
        if unassign_runtime(token, endpoint):
            now = list_endpoints(token)
            if now is not None and endpoint not in now:
                return
        time.sleep(10 * (i + 1))
    print(f"[run_colab_file] WARNING: runtime {endpoint} may still be assigned; release it manually", file=sys.stderr)


def _long_assign_timeout(fn):
    """The library uses a 30 s read timeout for /tun/m/assign; A100 assignment often answers later (ReadTimeout).
    Allow 120 s for the assign endpoint only."""
    def wrapped(url, *a, **k):
        if "/tun/m/assign" in str(url):
            k["timeout"] = max(float(k.get("timeout") or 0), 120.0)
        return fn(url, *a, **k)
    return wrapped


requests.get = _long_assign_timeout(requests.get)
requests.post = _long_assign_timeout(requests.post)

LOCK = pathlib.Path(tempfile.gettempdir()) / "usir_colab_assign.lock"
LATE_WINDOW_SEC = 120  # a runtime created by a failed assign can appear in the list well after the error


def acquire_lock(max_wait: float = 1800.0) -> None:
    """Serialise assign requests across runner processes, so that a runtime appearing during a failed assign
    belongs to this process and not to a concurrent job.  A lock older than 10 minutes is treated as stale."""
    t = time.time()
    while True:
        try:
            os.close(os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            return
        except FileExistsError:
            try:
                if time.time() - LOCK.stat().st_mtime > 600:
                    LOCK.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.time() - t > max_wait:
                raise RuntimeError("timed out waiting for the Colab assign lock")
            time.sleep(5)


t0 = time.time()
creds = get_credentials()
acquire_lock()
try:
    before = list_endpoints(creds.token)
    try:
        assignment = allocate_runtime(creds.token, accelerator=args.gpu)
    except Exception:
        # The assign request may have created a runtime even though the response was lost (ReadTimeout,
        # ConnectionError).  Watch the list for a while; anything new belongs to this process (lock held).
        released = set()
        if before is None:
            print("[run_colab_file] WARNING: assign failed and assignments could not be listed; check Colab", file=sys.stderr)
        else:
            t_fail = time.time()
            while time.time() - t_fail < LATE_WINDOW_SEC:
                time.sleep(15)
                after = list_endpoints(creds.token)
                for ep in sorted((after or set()) - before - released):
                    print(f"[run_colab_file] releasing runtime created by the failed assign: {ep}", file=sys.stderr)
                    release(creds.token, ep)
                    released.add(ep)
        raise
finally:
    LOCK.unlink(missing_ok=True)
endpoint = assignment["endpoint"]
stop_event = start_keepalive(creds.token, endpoint)
rc = 1
try:
    kernel_id = create_session(assignment["proxy_url"], assignment["proxy_token"])
    stdout, stderr, exit_code = execute_with_live_streaming(
        assignment["proxy_url"], assignment["proxy_token"], kernel_id, code,
        timeout=args.timeout, access_token=creds.token, endpoint=endpoint,
    )
    (out / f"{stem}.stdout.txt").write_text(stdout, encoding="utf-8")
    (out / f"{stem}.stderr.txt").write_text(stderr, encoding="utf-8")
    m = re.search(r"^ARCHIVE_B64 ([0-9a-f]{8}) (\S+)$", stdout, re.M)
    if not m:
        print(f"[run_colab_file] no ARCHIVE_B64 line (exit_code={exit_code}); stdout saved", file=sys.stderr)
    else:
        raw = gzip.decompress(base64.b64decode(m.group(2)))
        crc = format(zlib.crc32(raw) & 0xFFFFFFFF, "08x")
        if crc != m.group(1):
            print(f"[run_colab_file] CRC mismatch {crc} != {m.group(1)}", file=sys.stderr)
        else:
            (out / f"{stem}.json").write_bytes(raw)
            print(f"[run_colab_file] wrote {out / (stem + '.json')} ({len(raw)} bytes, crc ok)")
            rc = 0
    print(f"[run_colab_file] total wall {time.time() - t0:.0f}s (kernel exit_code={exit_code})")
finally:
    stop_event.set()
    if assignment.get("reused") and before and endpoint in before:
        print(f"[run_colab_file] runtime {endpoint} existed before this run; not released", file=sys.stderr)
    else:
        release(creds.token, endpoint)
sys.exit(rc)
