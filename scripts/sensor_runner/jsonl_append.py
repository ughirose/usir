"""Concurrency-safe append of one line to a shared JSONL log (measurement tools: timing / attempts / rounds).

Parallel annotator processes append to the same file. A plain `open(p, "a")` + write is not atomic on Windows (the CRT emulates
append by seek + write), so records were lost. Here the append is done under an exclusive lock file (O_CREAT|O_EXCL, retry) with
one binary write + fsync. A lock older than STALE_S (crashed writer) is removed."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

STALE_S = 10.0
TIMEOUT_S = 30.0


def append_line(path: Path, line: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + ".lock")
    data = (line.rstrip("\r\n") + "\n").encode("utf-8")
    deadline = time.monotonic() + TIMEOUT_S
    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except (FileExistsError, PermissionError):  # PermissionError: Windows delete-pending lock file
            try:
                if time.time() - lock.stat().st_mtime > STALE_S:
                    lock.unlink()
                    continue
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError(f"could not lock {lock}")
            time.sleep(0.005)
    try:
        with open(path, "ab") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    finally:
        os.close(fd)
        try:
            lock.unlink()
        except OSError:
            pass


def append_json(path: Path, obj: Any) -> None:
    append_line(path, json.dumps(obj, ensure_ascii=False))
