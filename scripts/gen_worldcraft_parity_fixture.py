"""Generates packages/usir-core/tests/fixtures/worldcraft_arbitrator_parity.json (Python reference).

Run: cd python; uv run python ../scripts/gen_worldcraft_parity_fixture.py
Both python/tests/test_worldcraft_parity.py and packages/usir-core/tests/worldcraft-parity.test.ts
assert against this file.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from usir_py.hsmm_longform_director import HSMMLongformDirector
from usir_py.nishiki_unified_field import WorldCraftArbitrator

OUT = Path(__file__).resolve().parents[1] / "packages" / "usir-core" / "tests" / "fixtures" / "worldcraft_arbitrator_parity.json"


def unit(*idx: int) -> list:
    rows = []
    for i in idx:
        r = [0.0] * 12
        r[i] = 1.0
        rows.append(r)
    return rows


def row(entries: dict) -> list:
    r = [0.0] * 12
    for k, v in entries.items():
        r[k] = float(v)
    return r


R1 = row({0: 1.0, 3: 2.0, 7: -1.0})
R2 = row({1: 0.5, 3: 1.0, 11: 2.0})
CONSTRAINT_SETS = {
    "empty": [],
    "unit_evidence": unit(7, 11),
    "unit_all_modes": unit(1, 2, 5, 6, 7, 10, 11),
    "lincomb_2row": [row({0: 1.0, 4: 1.0}), row({4: 1.0, 9: -1.0, 11: 0.5})],
    "rank_deficient_3row": [R1, R2, [a + 2.0 * b for a, b in zip(R1, R2)]],
}

MOMENTA = [
    [0.68, 0.72, 0.65, 0.75, 0.70, 0.62, 0.78, 0.69, 0.71, 0.74, 0.66, 0.70],
    [0.15, -0.12, 0.18, -0.14, 0.15, 0.10, -0.16, 0.12, -0.14, 0.18, 0.11, -0.15],
    [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0, 11.0, -12.0],
]

DIRECTOR_CONFIG = {
    "transition_matrix": [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
    "refractory_cooldown": 1,
    "projector_method": "svd",
}

DIRECTOR_OPS = [
    {"op": "arbitrate_rows", "target_p": MOMENTA[0], "rows": CONSTRAINT_SETS["lincomb_2row"]},
    {"op": "accumulate", "target_p": MOMENTA[0], "measured_p": MOMENTA[1]},
    {"op": "arbitrate_rows", "target_p": MOMENTA[2], "rows": CONSTRAINT_SETS["rank_deficient_3row"]},
    {"op": "step", "current_well": 0},
    {"op": "discharge", "base_climax_p": MOMENTA[0]},
    {"op": "step", "current_well": 2},
    {"op": "step", "current_well": 0},
    {"op": "arbitrate_rows", "target_p": MOMENTA[0], "rows": CONSTRAINT_SETS["unit_evidence"]},
    {"op": "step", "current_well": 0},
    {"op": "discharge", "base_climax_p": MOMENTA[1]},
]


def main() -> None:
    svd = WorldCraftArbitrator(projector_method="svd")
    tik = WorldCraftArbitrator(projector_method="tikhonov")
    sets = {}
    for name, rows in CONSTRAINT_SETS.items():
        j = WorldCraftArbitrator.build_general_jacobian(rows)
        p_svd = WorldCraftArbitrator.compute_null_space_projector_svd(j)
        p_tik = WorldCraftArbitrator.compute_null_space_projector(j)
        arb = []
        for m in MOMENTA:
            s_safe, s_cut, _ = svd.arbitrate_momentum(np.array(m), j)
            t_safe, t_cut, _ = tik.arbitrate_momentum(np.array(m), j)
            arb.append({"svd_safe": s_safe.tolist(), "svd_cut": s_cut.tolist(),
                        "tik_safe": t_safe.tolist(), "tik_cut": t_cut.tolist()})
        rank = int(round(12 - np.trace(p_svd)))
        sets[name] = {"rows": rows, "rank": rank, "p_null_svd": p_svd.tolist(),
                      "p_null_tikhonov": p_tik.tolist(), "arbitration": arb}

    d = HSMMLongformDirector(
        transition_matrix=np.array(DIRECTOR_CONFIG["transition_matrix"]),
        refractory_cooldown=DIRECTOR_CONFIG["refractory_cooldown"],
        seed=0,
        arbitrator=WorldCraftArbitrator(projector_method="svd"),
    )
    steps = []
    for op in DIRECTOR_OPS:
        rec = dict(op)
        if op["op"] == "arbitrate_rows":
            j = WorldCraftArbitrator.build_general_jacobian(op["rows"])
            safe, cut = d.arbitrate_with_jacobian(np.array(op["target_p"]), j)
            rec["out"] = {"safe_p": safe.tolist(), "cut_p": cut.tolist()}
        elif op["op"] == "accumulate":
            delta = d.accumulate_unreleased_momentum(np.array(op["target_p"]), np.array(op["measured_p"]))
            rec["out"] = {"delta": delta.tolist()}
        elif op["op"] == "step":
            nxt = d.step_transition(op["current_well"])
            rec["out"] = {"next_well": int(nxt)}
        elif op["op"] == "discharge":
            boosted, discharged = d.discharge_momentum(np.array(op["base_climax_p"]))
            rec["out"] = {"boosted_p": boosted.tolist(), "discharged_p": discharged.tolist()}
        rec["state"] = {"p_banked": d.p_banked.tolist(), "refractory_counter": int(d.refractory_counter),
                        "banked_energy": d.get_banked_energy()}
        steps.append(rec)

    fixture = {"version": 1, "generator": "scripts/gen_worldcraft_parity_fixture.py",
               "momenta": MOMENTA, "constraint_sets": sets,
               "director": {"config": DIRECTOR_CONFIG, "steps": steps}}
    OUT.write_text(json.dumps(fixture, indent=1), encoding="utf-8")
    print("wrote", OUT, "sets", {k: v["rank"] for k, v in sets.items()})


if __name__ == "__main__":
    main()
