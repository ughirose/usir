"""Phase 49 Colab job generator: GRPO step time with the rollouts of both prompts of a step in one generate() call
(8 rows, left-padded) vs one call per prompt (4 rows each).  L4, Qwen3-8B; both arms use share_ref_forward=True.

Writes artifacts/colab_run/p49_merge_rollouts_bench.py (gitignored).  Same settings as the Phase 48 bench
(build_p48_share_ref_bench.py): fixed PPL threshold 1.702, 6 steps per arm, one discarded warm-up step, same initial
LoRA weights and torch seed per arm.
Run: python scripts/run_colab_file.py artifacts/colab_run/p49_merge_rollouts_bench.py docs/handoff/v10 --gpu L4 --timeout 3600
"""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_colab_p33 import build_zip_b64  # noqa: E402
from build_p48_share_ref_bench import BODY as P48_BODY  # noqa: E402


def _sub(s: str, old: str, new: str) -> str:
    assert s.count(old) == 1, old
    return s.replace(old, new)


BODY = P48_BODY
BODY = _sub(BODY, '"p48_share_ref_bench_v1"', '"p49_merge_rollouts_bench_v1"')
BODY = _sub(BODY, "def arm(name, share, steps):", "def arm(name, merge, steps):")
BODY = _sub(BODY, "store_rollouts=True, reward_fn=reward, share_ref_forward=share,",
            "store_rollouts=True, reward_fn=reward, share_ref_forward=True, merge_step_rollouts=merge,")
BODY = _sub(BODY, '"share_ref_forward": share,', '"merge_step_rollouts": merge,')
BODY = _sub(BODY, 'arm("warmup", False, 1)', 'arm("warmup", True, 1)')
BODY = _sub(BODY, 'for name, share in (("unshared", False), ("shared", True)):', 'for name, merge in (("per_prompt", False), ("merged", True)):')
BODY = _sub(BODY, "res[\"arms\"].append(arm(name, share, 6))", "res[\"arms\"].append(arm(name, merge, 6))")


def main() -> None:
    out = HERE / "p49_merge_rollouts_bench.py"
    out.write_text("B64 = " + repr(build_zip_b64()) + "\n" + BODY, encoding="utf-8")
    print(out, out.stat().st_size)


if __name__ == "__main__":
    main()
