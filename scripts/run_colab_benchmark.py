#!/usr/bin/env python3
"""Run USIR v4.0 Scaled Bidirectional Benchmark on Google Colab T4 GPU.

Directly orchestrates the Colab GPU runtime via WebSocket streaming:
1. Allocates Tesla T4 GPU runtime
2. Clones latest commit from github.com/ughirose/usir
3. Executes python/colab/run_v4_bidirectional_experiment.py across all 30 benchmark instances
4. Collects and extracts the 4 authentic artifacts
5. Unassigns the Colab GPU runtime cleanly
"""

import base64
import io
import json
import os
import re
import sys
import tarfile
import time
import uuid

# Ensure UTF-8 console output on Windows
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass



from mcp_server_colab_exec.colab_runtime import (
    allocate_runtime,
    create_session,
    get_credentials,
    start_keepalive,
    unassign_runtime,
    _make_colab_input_reply,
    propagate_credentials,
    EPHEMERAL_AUTH_TYPES,
)
import websocket

COLAB_CODE = """
import os
import sys
import subprocess
import tarfile
import base64

print("=== 1. PREPARING REPOSITORY ===", flush=True)
if os.path.exists("/tmp/v4_checkpoint.json"):
    try:
        os.remove("/tmp/v4_checkpoint.json")
        print("Removed stale /tmp/v4_checkpoint.json", flush=True)
    except Exception:
        pass

if os.path.exists("usir"):
    subprocess.run(["rm", "-rf", "usir"], check=True)

subprocess.run(["git", "clone", "https://github.com/ughirose/usir.git"], check=True)

res_rev = subprocess.run(["git", "-C", "usir", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
head_commit = res_rev.stdout.strip()
print(f"Cloned HEAD commit: {head_commit}", flush=True)

print("=== 2. RUNNING EXPERIMENT ===", flush=True)
cmd = [sys.executable, "-u", "python/colab/run_v4_bidirectional_experiment.py"]
proc = subprocess.Popen(
    cmd,
    cwd="usir",
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)

for line in iter(proc.stdout.readline, ''):
    sys.stdout.write(line)
    sys.stdout.flush()

proc.wait()
print(f"Process finished with exit code: {proc.returncode}", flush=True)
if proc.returncode != 0:
    raise RuntimeError(f"Experiment failed with exit code {proc.returncode}")

print("=== 3. PACKAGING ARTIFACTS ===", flush=True)
artifact_files = [
    "v4_bidirectional_experiment_results.json",
    "v4_bidirectional_experiment_report.md",
    "blind_evaluation_sheet.md",
    "blind_evaluation_sheet.html",
]

tar_path = "/tmp/v4_artifacts.tar.gz"
with tarfile.open(tar_path, "w:gz") as tar:
    for a in artifact_files:
        p = os.path.join("usir", "artifacts", a)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing artifact: {p}")
        sz = os.path.getsize(p)
        print(f"Packaging {a} ({sz} bytes)", flush=True)
        tar.add(p, arcname=a)

with open(tar_path, "rb") as f:
    b64_content = base64.b64encode(f.read()).decode("ascii")

print("=== BEGIN_ARTIFACTS_BASE64 ===", flush=True)
print(b64_content, flush=True)
print("=== END_ARTIFACTS_BASE64 ===", flush=True)
print("ALL_DONE_SUCCESSFULLY", flush=True)
"""


def execute_with_live_streaming(
    proxy_url: str,
    proxy_token: str,
    kernel_id: str,
    code: str,
    timeout: int = 2400,
    access_token: str = None,
    endpoint: str = None,
) -> tuple[str, str, int]:
    """Execute code on Colab kernel while streaming output to local stdout."""
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    execute_session_id = uuid.uuid4().hex
    ws_url = proxy_url.replace("https://", "wss://").replace("http://", "ws://")
    ws_url = f"{ws_url}/api/kernels/{kernel_id}/channels?session_id={execute_session_id}"

    ws = websocket.create_connection(
        ws_url,
        header=[
            f"X-Colab-Runtime-Proxy-Token: {proxy_token}",
            "X-Colab-Client-Agent: vscode",
        ],
        timeout=timeout,
        ping_interval=20,
        ping_timeout=10,
    )

    msg_id = uuid.uuid4().hex
    execute_msg = {
        "header": {
            "msg_id": msg_id,
            "msg_type": "execute_request",
            "username": "colab-exec",
            "session": execute_session_id,
            "version": "5.3",
        },
        "parent_header": {},
        "metadata": {},
        "content": {
            "code": code,
            "silent": False,
            "store_history": True,
            "user_expressions": {},
            "allow_stdin": False,
            "stop_on_error": True,
        },
        "channel": "shell",
    }
    ws.send(json.dumps(execute_msg))

    had_error = False
    saw_idle = False
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            raw = ws.recv()
        except websocket.WebSocketTimeoutException:
            continue
        except websocket.WebSocketConnectionClosedException as e:
            sys.stderr.write(f"\n[colab-exec] Connection closed: {e}. Attempting reconnect...\n")
            sys.stderr.flush()
            reconnect_success = False
            for retry in range(1, 16):
                time.sleep(3)
                try:
                    ws = websocket.create_connection(
                        ws_url,
                        header=[
                            f"X-Colab-Runtime-Proxy-Token: {proxy_token}",
                            "X-Colab-Client-Agent: vscode",
                        ],
                        timeout=timeout,
                        ping_interval=20,
                        ping_timeout=10,
                    )
                    sys.stderr.write(f"[colab-exec] Reconnected to kernel (attempt {retry})\n")
                    sys.stderr.flush()
                    reconnect_success = True
                    break
                except Exception as rec_err:
                    sys.stderr.write(f"[colab-exec] Reconnect attempt {retry} failed: {rec_err}\n")
                    sys.stderr.flush()
            if reconnect_success:
                continue
            stderr_buf.write(f"\n[colab-exec] Connection lost: {e}\n")
            had_error = True
            break
        except Exception as e:
            sys.stderr.write(f"\n[colab-exec] Unexpected WebSocket error: {e}\n")
            sys.stderr.flush()
            stderr_buf.write(f"\n[colab-exec] Unexpected WebSocket error: {e}\n")
            had_error = True
            break
        if not raw:
            continue

        msg = json.loads(raw)
        msg_type = msg.get("msg_type") or msg.get("header", {}).get("msg_type", "")
        content = msg.get("content", {})

        if msg_type == "colab_request":
            metadata = msg.get("metadata", {})
            request_type = metadata.get("colab_request_type")
            colab_msg_id = metadata.get("colab_msg_id")
            auth_type = (
                content.get("request", {}).get("authType", "")
                if isinstance(content, dict) else ""
            )
            auth_type = str(auth_type).lower()

            if request_type == "request_auth" and colab_msg_id is not None:
                error_text = None
                if not access_token or not endpoint:
                    error_text = "missing auth context for credentials propagation"
                elif auth_type not in EPHEMERAL_AUTH_TYPES:
                    error_text = f"unsupported auth type: {auth_type}"
                else:
                    try:
                        dry = propagate_credentials(access_token, endpoint, auth_type, dry_run=True)
                        if dry.get("success"):
                            propagate_credentials(access_token, endpoint, auth_type, dry_run=False)
                        elif dry.get("unauthorizedRedirectUri"):
                            error_text = (
                                f"{auth_type} requires interactive browser consent: "
                                f"{dry['unauthorizedRedirectUri']}"
                            )
                        else:
                            error_text = f"{auth_type} dry-run failed: {dry}"
                    except Exception as e:
                        error_text = f"{auth_type} propagation failed: {e}"

                reply = _make_colab_input_reply(execute_session_id, colab_msg_id, error_text)
                ws.send(json.dumps(reply))
                if error_text:
                    stderr_buf.write(f"[colab-exec] Warning: {error_text}\n")
            continue

        parent_msg_id = msg.get("parent_header", {}).get("msg_id")
        if parent_msg_id != msg_id:
            continue

        def safe_write(stream, s: str):
            try:
                stream.write(s)
                stream.flush()
            except Exception:
                try:
                    stream.buffer.write(s.encode("utf-8", errors="replace"))
                    stream.buffer.flush()
                except Exception:
                    pass

        if msg_type == "stream":
            stream_name = content.get("name", "stdout")
            text = content.get("text", "")
            if stream_name == "stdout":
                stdout_buf.write(text)
                safe_write(sys.stdout, text)
            else:
                stderr_buf.write(text)
                safe_write(sys.stderr, text)

        elif msg_type == "execute_result":
            data = content.get("data", {})
            text = data.get("text/plain", "")
            if text:
                stdout_buf.write(text + "\n")
                safe_write(sys.stdout, text + "\n")

        elif msg_type == "display_data":
            data = content.get("data", {})
            text = data.get("text/plain", "")
            if text:
                stdout_buf.write(text + "\n")
                safe_write(sys.stdout, text + "\n")

        elif msg_type == "error":
            had_error = True
            ename = content.get("ename", "Error")
            evalue = content.get("evalue", "")
            traceback_lines = content.get("traceback", [])
            err_msg = f"\n{ename}: {evalue}\n"
            stderr_buf.write(err_msg)
            safe_write(sys.stderr, err_msg)
            for line in traceback_lines:
                clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
                stderr_buf.write(clean + "\n")
                safe_write(sys.stderr, clean + "\n")


        elif msg_type == "status":
            state = content.get("execution_state")
            if state == "idle":
                saw_idle = True
                break

    ws.close()
    if not saw_idle:
        msg = "[colab-exec] ERROR: Timed out waiting for kernel execution to finish.\n"
        stderr_buf.write(msg)
        sys.stderr.write(msg)
        had_error = True

    exit_code = 1 if had_error else 0
    return stdout_buf.getvalue(), stderr_buf.getvalue(), exit_code


def main() -> int:
    start_time = time.time()
    print("=" * 70)
    print("USIR v4.0 REMOTE COLAB BENCHMARK LAUNCHER")
    print("=" * 70)

    creds = get_credentials()
    access_token = creds.token

    assignment = allocate_runtime(access_token, accelerator="T4")
    endpoint = assignment["endpoint"]
    stop_event = start_keepalive(access_token, endpoint)

    try:
        kernel_id = create_session(assignment["proxy_url"], assignment["proxy_token"])
        print(f"Allocated T4 GPU runtime: {endpoint}, kernel: {kernel_id}")

        stdout, stderr, exit_code = execute_with_live_streaming(
            assignment["proxy_url"],
            assignment["proxy_token"],
            kernel_id,
            COLAB_CODE,
            timeout=2400,
            access_token=access_token,
            endpoint=endpoint,
        )

        total_elapsed = time.time() - start_time
        print(f"\nRemote execution finished in {total_elapsed:.1f}s (Exit code: {exit_code})")

        if exit_code != 0:
            print(f"ERROR: Execution failed with exit code {exit_code}")
            if stderr:
                print(f"STDERR:\n{stderr}")
            return 1

        # Extract base64 artifacts
        m = re.search(r"=== BEGIN_ARTIFACTS_BASE64 ===\s*\n(.*?)\n\s*=== END_ARTIFACTS_BASE64 ===", stdout, re.DOTALL)
        if not m:
            print("ERROR: Could not find base64 artifact markers in stdout!")
            return 1

        b64_str = m.group(1).strip()
        tar_bytes = base64.b64decode(b64_str)
        print(f"Decoded artifact tarball: {len(tar_bytes)} bytes")

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        artifacts_dir = os.path.join(repo_root, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
            tar.extractall(path=artifacts_dir)
            for member in tar.getmembers():
                dest = os.path.join(artifacts_dir, member.name)
                print(f"Extracted: {dest} ({os.path.getsize(dest)} bytes)")

        print("\nAll 4 authentic artifacts successfully extracted to artifacts/ directory!")
        return 0

    finally:
        stop_event.set()
        unassign_runtime(access_token, endpoint)
        print("Colab GPU runtime unassigned cleanly.")


if __name__ == "__main__":
    sys.exit(main())
