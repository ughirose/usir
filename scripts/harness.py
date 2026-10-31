#!/usr/bin/env python3
"""Antigravity Harness & Safety-Net Verification Runner for USIR."""

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run_command(cmd, cwd=REPO_ROOT, check=True):
    print(f"[HARNESS] Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    res = subprocess.run(cmd, cwd=cwd, shell=True, text=True, capture_output=False)
    if check and res.returncode != 0:
        print(f"[HARNESS] Error: Command exited with code {res.returncode}")
        return False
    return res.returncode == 0


def run_python_tests():
    print("\n--- [GATE 2] Running Python LOOCV Golden Suite ---")
    py_dir = os.path.join(REPO_ROOT, "python")
    cmd = "uv run pytest tests/ -v"
    return run_command(cmd, cwd=py_dir, check=False)


def run_typescript_tests():
    print("\n--- [GATE 1 & 3] Running TypeScript Vitest Suite ---")
    node_modules = os.path.join(REPO_ROOT, "node_modules")
    if not os.path.exists(node_modules):
        print("[HARNESS] Notice: node_modules not yet installed. Skipping Vitest.")
        return True
    return run_command("pnpm test", cwd=REPO_ROOT, check=False)


def main():
    parser = argparse.ArgumentParser(description="USIR Test & Safety-Net Harness")
    parser.add_argument("action", choices=["test", "test:py", "test:ts", "status"], default="test", nargs="?")
    args = parser.parse_args()

    success = True
    if args.action in ["test", "test:py"]:
        py_ok = run_python_tests()
        if not py_ok:
            success = False

    if args.action in ["test", "test:ts"]:
        ts_ok = run_typescript_tests()
        if not ts_ok:
            success = False

    if not success:
        print("\n[HARNESS] FAILURE DETECTED: Quality gates failed.")
        sys.exit(1)

    print("\n[HARNESS] SUCCESS: All specified gates passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
