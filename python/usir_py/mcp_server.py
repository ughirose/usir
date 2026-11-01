"""USIR v4.0 Standard Model Context Protocol (MCP) Server.

JSON-RPC 2.0 (stdio) compliant server directly registrable with:
- Claude Desktop
- Antigravity
- Cursor
- Dify

Exposes 3 Core Tools:
1. usir_audit_v4_manifold: 24D symplectic state, 3-phase QDA distance, semantic geodesic curvature, and multiplicative GRPO barrier reward.
2. usir_plan_bidirectional_bridge: Meet-in-the-middle Schrödinger bridge planner across novel, article, and cot_reasoning domains.
3. usir_verify_cot_collision: Verifies LLM reasoning trace (<think>) against forward premise and backward goal to prune reasoning drift.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional

import numpy as np

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from .extractor_24d import extract_usir_24d
from .manifold_geodesic import (
    _embed_sentence_unit,
    compute_semantic_geodesic_curvature,
    compute_three_phase_qda_distance,
    compute_multiplicative_grpo_reward,
)
from .bidirectional_bridge import (
    pin_boundary_anchors,
    solve_bidirectional_collision_bridge,
)

SERVER_INFO = {
    "name": "usir-v4-mcp-server",
    "version": "4.0.0",
    "description": "USIR v4.0 Universal Style Genome & Bidirectional Schrödinger Bridge MCP Engine",
}

TOOLS = [
    {
        "name": "usir_audit_v4_manifold",
        "description": (
            "Audits text style and reasoning dynamics via 24D symplectic geometry, "
            "asymmetric 3-phase QDA distance (Deadzone freeze vs Critical vs Chaos), "
            "semantic geodesic curvature M_geodesic, and 4-factor non-hackable GRPO reward."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Input natural language text or chain-of-thought steps.",
                },
                "domain": {
                    "type": "string",
                    "enum": ["novel", "article", "cot_reasoning"],
                    "default": "novel",
                    "description": "Target evaluation domain.",
                },
                "x_0": {
                    "type": "string",
                    "description": "Optional start anchor for bridge evaluation.",
                },
                "x_T": {
                    "type": "string",
                    "description": "Optional end goal anchor for bridge evaluation.",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "usir_plan_bidirectional_bridge",
        "description": (
            "Designs a meet-in-the-middle bidirectional Schrödinger bridge for a theme or problem. "
            "Pins boundary anchors x_0 and x_T, plans forward wave psi and backward wave phi, "
            "and generates midpoint collision action criteria and next-step logit steering guidance."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "theme": {
                    "type": "string",
                    "description": "Core theme, topic, or target problem statement.",
                },
                "domain": {
                    "type": "string",
                    "enum": ["novel", "article", "cot_reasoning"],
                    "default": "novel",
                    "description": "Domain of bridge (novel, analytical article, or deductive reasoning).",
                },
                "start_premise": {
                    "type": "string",
                    "description": "Optional custom start anchor.",
                },
                "end_goal": {
                    "type": "string",
                    "description": "Optional custom end goal anchor.",
                },
                "k_candidates": {
                    "type": "integer",
                    "default": 4,
                    "description": "Number of candidate waves to evaluate.",
                },
            },
            "required": ["theme"],
        },
    },
    {
        "name": "usir_verify_cot_collision",
        "description": (
            "Verifies an LLM reasoning trace (<think> or steps) by inspecting both "
            "forward deduction from initial premises and backward abduction from the target goal. "
            "Detects reasoning drift, tangents, and invalid hops, outputting pruned collision guidance."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "reasoning_text": {
                    "type": "string",
                    "description": "Raw or structured CoT reasoning steps or <think> block content.",
                },
                "problem_statement": {
                    "type": "string",
                    "description": "Initial problem statement or premises.",
                },
                "target_conclusion": {
                    "type": "string",
                    "description": "Target goal proposition or claimed conclusion.",
                },
            },
            "required": ["reasoning_text"],
        },
    },
]


# ==============================================================================
# Tool Implementations
# ==============================================================================

def tool_audit_v4_manifold(args: Dict[str, Any]) -> Dict[str, Any]:
    text = args.get("text", "")
    domain = args.get("domain", "novel")
    x_0 = args.get("x_0")
    x_T = args.get("x_T")

    extracted = extract_usir_24d(text)
    qda_res = compute_three_phase_qda_distance(text)
    geodesic_res = compute_semantic_geodesic_curvature(text)
    grpo_res = compute_multiplicative_grpo_reward(text, x_0=x_0, x_T=x_T)

    return {
        "domain": domain,
        "text_length": len(text),
        "phase_regime": qda_res["phase_regime"],
        "is_healthy_critical": qda_res["is_healthy_critical"],
        "d_3phase": qda_res["d_3phase"],
        "penalties": {
            "freeze_barrier": qda_res["penalty_freeze"],
            "chaos_barrier": qda_res["penalty_chaos"],
            "sigma_norm": qda_res["sigma_norm"],
        },
        "manifold_geometry": {
            "mean_curvature": geodesic_res["mean_curvature"],
            "max_curvature": geodesic_res["max_curvature"],
            "mean_torsion": geodesic_res["mean_torsion"],
            "brownian_ratio": geodesic_res["brownian_ratio"],
            "is_brownian_catalog": geodesic_res["is_brownian_catalog"],
            "m_geodesic": geodesic_res["m_geodesic"],
        },
        "grpo_reward": {
            "r_grpo": grpo_res["reward"],
            "m_syntax": grpo_res["m_syntax"],
            "m_geodesic": grpo_res["m_geodesic"],
            "m_bridge": grpo_res["m_bridge"],
            "m_cleanroom": grpo_res["m_cleanroom"],
            "is_rejected": grpo_res["is_rejected"],
        },
        "order_parameters": extracted["order_params"],
    }


def tool_plan_bidirectional_bridge(args: Dict[str, Any]) -> Dict[str, Any]:
    theme = args.get("theme", "")
    domain = args.get("domain", "novel")
    start_premise = args.get("start_premise")
    end_goal = args.get("end_goal")
    k_candidates = args.get("k_candidates", 4)

    anchors = pin_boundary_anchors(
        theme=theme,
        domain=domain,
        custom_start=start_premise,
        custom_end=end_goal,
    )

    fwd_cands = args.get("forward_candidates")
    bwd_cands = args.get("backward_candidates")
    if not fwd_cands or not bwd_cands:
        # Default prompt engineering fixture for planning
        fwd_cands = [[anchors.x_0, f"{theme}の気配が深まり、冷気が空間を満たしていく。"]]
        bwd_cands = [[f"{theme}の物理平衡が定着し、", anchors.x_T]]

    bridge_res = solve_bidirectional_collision_bridge(
        anchors,
        forward_candidates=fwd_cands,
        backward_candidates=bwd_cands,
        k_candidates=k_candidates,
    )

    steering_directive = (
        f"[USIR v4.0 Schrödinger Bridge Steering Directive]\n"
        f"Domain: {domain} | Theme: {theme}\n"
        f"Forward Anchor x_0: {anchors.x_0}\n"
        f"Backward Anchor x_T: {anchors.x_T}\n"
        f"Optimal Collision Pair: Forward Wave #{bridge_res.optimal_pair[0]} x Backward Wave #{bridge_res.optimal_pair[1]}\n"
        f"Minimal Action S_collision: {bridge_res.min_action:.4f} (Bridge Coupling M_bridge: {bridge_res.m_bridge:.4f})\n"
        f"Midpoint Infilling Rule: Weld forward progress seamlessly into backward precondition without tangential drift."
    )

    return {
        "domain": domain,
        "theme": theme,
        "anchors": {
            "x_0": anchors.x_0,
            "x_T": anchors.x_T,
            "kamae_target": anchors.kamae_coords.tolist(),
        },
        "collision_plan": {
            "optimal_pair": list(bridge_res.optimal_pair),
            "min_action": bridge_res.min_action,
            "m_bridge": bridge_res.m_bridge,
            "d_geodesic": bridge_res.d_geodesic,
            "d_causal": bridge_res.d_causal,
            "d_3phase": bridge_res.d_3phase,
            "welded_text_preview": bridge_res.welded_text[:180] + ("..." if len(bridge_res.welded_text) > 180 else ""),
        },
        "steering_directive": steering_directive,
    }


def tool_verify_cot_collision(args: Dict[str, Any]) -> Dict[str, Any]:
    reasoning_text = args.get("reasoning_text", "")
    problem_statement = args.get("problem_statement")
    target_conclusion = args.get("target_conclusion")

    # Split reasoning into steps
    lines = [line.strip() for line in reasoning_text.splitlines() if len(line.strip()) >= 5]
    if len(lines) < 2:
        lines = [s.strip() for s in reasoning_text.split("。") if len(s.strip()) >= 5]

    geodesic_res = compute_semantic_geodesic_curvature(lines)
    qda_res = compute_three_phase_qda_distance(reasoning_text)

    # Check forward alignment from problem_statement
    fwd_score = 1.0
    if problem_statement and lines:
        v_prob = _embed_sentence_unit(problem_statement)
        v_start = _embed_sentence_unit(lines[0])
        fwd_score = float(np.dot(v_prob, v_start) / (np.linalg.norm(v_prob) * np.linalg.norm(v_start) + 1e-7))
        fwd_score = float(np.clip(fwd_score, 0.0, 1.0))

    # Check backward alignment to target_conclusion
    bwd_score = 1.0
    if target_conclusion and lines:
        v_goal = _embed_sentence_unit(target_conclusion)
        v_end = _embed_sentence_unit(lines[-1])
        bwd_score = float(np.dot(v_goal, v_end) / (np.linalg.norm(v_goal) * np.linalg.norm(v_end) + 1e-7))
        bwd_score = float(np.clip(bwd_score, 0.0, 1.0))

    # Reasoning Drift Detection:
    # High curvature spikes between steps indicate ungrounded leaps or wandering tangents
    drift_steps: List[Dict[str, Any]] = []
    for idx, kappa in enumerate(geodesic_res["curvatures"]):
        if kappa > 0.50:
            step_a = lines[idx] if idx < len(lines) else ""
            step_b = lines[idx + 1] if idx + 1 < len(lines) else ""
            drift_steps.append({
                "step_index": idx + 1,
                "curvature": kappa,
                "warning": "High semantic curvature: associative leap or topic tangent detected.",
                "step_before": step_a[:60],
                "step_after": step_b[:60],
            })

    has_drift = len(drift_steps) > 0 or geodesic_res["is_brownian_catalog"]
    collision_action = (1.0 - fwd_score) + (1.0 - bwd_score) + geodesic_res["mean_curvature"] + 0.1 * qda_res["d_3phase"]

    recommendation = (
        "CoT collision verified: continuous directed deduction from premise to goal."
        if not has_drift and collision_action < 2.0
        else f"Reasoning drift detected ({len(drift_steps)} high-curvature hops). Prune ungrounded tangents."
    )

    return {
        "step_count": len(lines),
        "forward_premise_alignment": fwd_score,
        "backward_goal_alignment": bwd_score,
        "collision_action": float(collision_action),
        "mean_step_curvature": geodesic_res["mean_curvature"],
        "brownian_ratio": geodesic_res["brownian_ratio"],
        "has_reasoning_drift": has_drift,
        "drift_warnings": drift_steps,
        "recommendation": recommendation,
    }


# ==============================================================================
# JSON-RPC 2.0 Dispatcher
# ==============================================================================

def process_jsonrpc_request(req: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    req_id = req.get("id")
    method = req.get("method")
    params = req.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": SERVER_INFO,
                "capabilities": {
                    "tools": {},
                },
            },
        }

    elif method == "notifications/initialized":
        # Notification, no response needed
        return None

    elif method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": TOOLS,
            },
        }

    elif method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments", {})

        try:
            if tool_name == "usir_audit_v4_manifold":
                output = tool_audit_v4_manifold(args)
            elif tool_name == "usir_plan_bidirectional_bridge":
                output = tool_plan_bidirectional_bridge(args)
            elif tool_name == "usir_verify_cot_collision":
                output = tool_verify_cot_collision(args)
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method or tool not found: {tool_name}",
                    },
                }

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(output, ensure_ascii=False, indent=2),
                        }
                    ],
                },
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32000,
                    "message": f"Internal tool execution error: {str(e)}",
                },
            }

    else:
        if req_id is not None:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not supported: {method}",
                },
            }
        return None


def run_stdio_server() -> None:
    """Reads JSON-RPC 2.0 lines from stdin and writes responses to stdout."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            res = process_jsonrpc_request(req)
            if res is not None:
                sys.stdout.write(json.dumps(res, ensure_ascii=False) + "\n")
                sys.stdout.flush()
        except Exception as e:
            err_res = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {str(e)}"},
            }
            sys.stdout.write(json.dumps(err_res, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    run_stdio_server()
