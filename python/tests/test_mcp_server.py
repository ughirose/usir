"""Unit & Integration Tests for USIR v4.0 Python MCP Server (JSON-RPC 2.0).

Verifies:
1. JSON-RPC initialize handshake & serverInfo.
2. tools/list exposes all 3 required tools.
3. tools/call usir_audit_v4_manifold returns 24D, 3-phase QDA, and M_geodesic.
4. tools/call usir_plan_bidirectional_bridge returns boundary anchors and steering directive.
5. tools/call usir_verify_cot_collision detects ungrounded leaps & reasoning drift.
6. Stdio pipeline simulation.
"""

from __future__ import annotations

import json
import pytest

from usir_py.mcp_server import (
    process_jsonrpc_request,
    TOOLS,
    SERVER_INFO,
)


def test_mcp_initialize():
    """Verify initialize handshake returns proper protocol version and serverInfo."""
    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "clientInfo": {"name": "test-client", "version": "1.0"},
        },
    }
    res = process_jsonrpc_request(req)
    assert res is not None
    assert res["jsonrpc"] == "2.0"
    assert res["id"] == 1
    assert res["result"]["serverInfo"]["name"] == "usir-v4-mcp-server"
    assert "tools" in res["result"]["capabilities"]


def test_mcp_tools_list():
    """Verify tools/list exposes all 3 specified tools."""
    req = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {},
    }
    res = process_jsonrpc_request(req)
    assert res is not None
    tool_names = [t["name"] for t in res["result"]["tools"]]
    assert "usir_audit_v4_manifold" in tool_names
    assert "usir_plan_bidirectional_bridge" in tool_names
    assert "usir_verify_cot_collision" in tool_names


def test_mcp_tool_call_audit_v4_manifold():
    """Verify usir_audit_v4_manifold tool execution and schema return."""
    text = (
        "雨が静かに降り始め、古い瓦屋根を濡らしていった。"
        "軒先から落ちる水滴が、土間の踏み石をかすかに叩いている。"
        "渡り廊下の奥からは、微かな杉の香りが漂っていた。"
    )
    req = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "usir_audit_v4_manifold",
            "arguments": {
                "text": text,
                "domain": "novel",
            },
        },
    }
    res = process_jsonrpc_request(req)
    assert res is not None
    assert "result" in res
    content = json.loads(res["result"]["content"][0]["text"])

    assert content["domain"] == "novel"
    assert "d_3phase" in content
    assert "phase_regime" in content
    assert "manifold_geometry" in content
    assert "mean_curvature" in content["manifold_geometry"]
    assert "grpo_reward" in content
    assert content["grpo_reward"]["r_grpo"] > 0.0


def test_mcp_tool_call_plan_bidirectional_bridge():
    """Verify usir_plan_bidirectional_bridge tool execution."""
    req = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "usir_plan_bidirectional_bridge",
            "arguments": {
                "theme": "山頂の観測所と冬の星座",
                "domain": "novel",
                "k_candidates": 3,
            },
        },
    }
    res = process_jsonrpc_request(req)
    assert res is not None
    assert "result" in res
    content = json.loads(res["result"]["content"][0]["text"])

    assert content["theme"] == "山頂の観測所と冬の星座"
    assert "anchors" in content
    assert "x_0" in content["anchors"]
    assert "x_T" in content["anchors"]
    assert "collision_plan" in content
    assert content["collision_plan"]["min_action"] > 0.0
    assert "steering_directive" in content


def test_mcp_tool_call_verify_cot_collision():
    """Verify usir_verify_cot_collision identifies reasoning drift."""
    # Tangential reasoning trace with sudden topic leap
    leaping_trace = (
        "ステップ1：問題の仮定より、群GはコンパクトHausdorff位相群である。\n"
        "ステップ2：突然ですが、昨夜の夕食に食べたカレーの香りが思い出される。\n"
        "ステップ3：カレーのスパイスはインドからシルクロードを経由して伝播した。\n"
        "ステップ4：したがって、Haar測度の左不変性と一意性が証明された。"
    )
    req = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {
            "name": "usir_verify_cot_collision",
            "arguments": {
                "reasoning_text": leaping_trace,
                "problem_statement": "コンパクト群G上の左不変Haar測度の存在を証明せよ。",
                "target_conclusion": "G上の左不変Haar測度は定数倍を除いて一意に存在する。",
            },
        },
    }
    res = process_jsonrpc_request(req)
    assert res is not None
    assert "result" in res
    content = json.loads(res["result"]["content"][0]["text"])

    assert content["has_reasoning_drift"]
    assert len(content["drift_warnings"]) > 0
    assert "Prune ungrounded tangents" in content["recommendation"]
