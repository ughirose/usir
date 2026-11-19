"""CLI runner for Step 1 sensor evaluation on F9 gold excerpts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Any

from usir_py.sensor_eval.f9_loader import F9Dataset, load_f9_dataset
from usir_py.sensor_eval.metrics import (
    compute_politeness_pairwise_agreement,
    evaluate_model_predictions,
    evaluate_step1_gate,
    load_sensor_v2_validator,
)
from usir_py.sensor_eval.ollama_client import call_ollama_chat
from usir_py.sensor_eval.prompt import (
    DEFAULT_PROMPT_KEY,
    build_ollama_utterances_format_schema,
    build_work_messages,
    compute_prompt_hash,
    get_prompt,
)


def parse_model_spec(spec: str) -> tuple[str, bool | None, str]:
    """Parse `model_name[::think=true|false]` into `(model, think, safe_tag)`."""
    think: bool | None = None
    model = spec
    if "::" in spec:
        model, opt = spec.split("::", 1)
        if opt == "think=true":
            think = True
        elif opt == "think=false":
            think = False
        else:
            raise ValueError(f"Unsupported model option {opt!r} in {spec!r}")
    label = spec if think is not None else model
    safe_tag = re.sub(r"[^a-zA-Z0-9._-]+", "_", label).strip("_")
    return model, think, safe_tag


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_repo_path(path_str: str | None, repo_root: Path | None = None) -> str | None:
    if path_str is None:
        return None
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    root = repo_root or _default_repo_root()
    return str(root / p)


def load_eval_config(config_path: Path) -> dict[str, Any]:
    """Load reusable evaluation JSON config and normalize model specs/options."""
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Eval config must be a JSON object: {config_path}")

    repo_root = _default_repo_root()
    cfg = dict(raw)
    for path_key in ("gold_dir", "split_file", "out_dir", "politeness_gold", "nishiki_bin"):
        if path_key in cfg and cfg[path_key] is not None:
            cfg[path_key] = _resolve_repo_path(str(cfg[path_key]), repo_root)

    raw_models = cfg.get("models")
    if raw_models is not None:
        model_specs: list[str] = []
        per_model_options: dict[str, dict[str, Any]] = {}
        for entry in raw_models:
            if isinstance(entry, str):
                model_specs.append(entry)
            elif isinstance(entry, dict) and isinstance(entry.get("spec"), str):
                spec = entry["spec"]
                model_specs.append(spec)
                opts: dict[str, Any] = {}
                for opt_key in ("num_ctx", "num_predict", "timeout", "seed"):
                    if opt_key in entry and entry[opt_key] is not None:
                        opts[opt_key] = entry[opt_key]
                if opts:
                    per_model_options[spec] = opts
            else:
                raise ValueError(f"Invalid model entry in config {config_path}: {entry!r}")
        cfg["models"] = model_specs
        cfg["per_model_options"] = per_model_options
    return cfg


def _extract_utterances(parsed_json: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    if not isinstance(parsed_json, dict):
        return None
    utts = parsed_json.get("utterances")
    if not isinstance(utts, list):
        return None
    return utts


def _build_model_summary_entry(
    *,
    label: str,
    model_name: str,
    think: bool | None,
    result_file_name: str,
    elapsed_sec: float,
    prompt_eval_count: int,
    eval_count: int,
    metrics: dict[str, Any],
    prompt_version: str,
    prompt_hash: str,
    context_window_warning: bool,
    gate: dict[str, Any],
) -> dict[str, Any]:
    ov = metrics["overall"]
    return {
        "model_spec": label,
        "model_name": model_name,
        "think": think,
        "prompt_version": prompt_version,
        "prompt_hash": prompt_hash,
        "result_file": result_file_name,
        "elapsed_sec": round(elapsed_sec, 3),
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
        "context_window_warning": context_window_warning,
        "schema_valid_works": metrics["schema_validation"]["valid_works"],
        "total_works": metrics["schema_validation"]["total_works"],
        "non_roster_speaker_count": len(metrics["non_roster_speaker_ids"]),
        "null_string_speaker_count": ov.get(
            "null_string_speaker_count", metrics.get("null_string_speaker_count", 0)
        ),
        "explicit_error_rate": ov["explicit_error_rate"],
        "explicit_errors": ov["explicit_errors"],
        "explicit_denominator": ov["explicit_denominator"],
        "overall_error_rate": ov["overall_error_rate"],
        "overall_errors": ov["overall_errors"],
        "overall_error_rate_raw_speaker": ov["overall_error_rate_raw_speaker"],
        "overall_errors_raw_speaker": ov["overall_errors_raw_speaker"],
        "attributed_total": ov["attributed_total"],
        "politeness_agreement_rate": ov["politeness_agreement_rate"],
        "politeness_gold_available": ov["politeness_gold_available"],
        "ambiguous_rate": ov["ambiguous_rate"],
        "ambiguous_attributed_count": ov["ambiguous_attributed_count"],
        "ambiguous_rate_all_quotes": ov["ambiguous_rate_all_quotes"],
        "excluded_abstention_rate": ov["excluded_abstention_rate"],
        "not_speech_abstention_rate": ov.get("not_speech_abstention_rate"),
        "not_speech_abstained": ov.get("not_speech_abstained"),
        "not_speech_total": ov.get("not_speech_total"),
        "undetermined_abstention_rate": ov.get("undetermined_abstention_rate"),
        "undetermined_abstained": ov.get("undetermined_abstained"),
        "undetermined_total": ov.get("undetermined_total"),
        "gate": gate,
        "by_evidence": ov["by_evidence"],
        "by_narration": {
            k: {
                "explicit_error_rate": v["explicit_error_rate"],
                "overall_error_rate": v["overall_error_rate"],
                "overall_error_rate_raw_speaker": v["overall_error_rate_raw_speaker"],
                "ambiguous_rate": v["ambiguous_rate"],
                "attributed_total": v["attributed_total"],
            }
            for k, v in metrics["by_narration"].items()
        },
    }


def _write_summary(
    dataset: F9Dataset,
    out_dir: Path,
    prompt_version: str,
    prompt_hash: str,
    model_runs: list[dict[str, Any]],
    predictions_index: dict[str, dict[str, list[dict[str, Any]] | None]],
) -> dict[str, Any]:
    pairwise_pol = compute_politeness_pairwise_agreement(
        dataset.works, predictions_index, attributed_only=True
    )
    summary = {
        "run_timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "split": dataset.split,
        "split_version": dataset.split_version,
        "gold_json_sha256": dataset.gold_json_sha256,
        "work_ids_evaluated": [w.work_id for w in dataset.works],
        "prompt_version": prompt_version,
        "prompt_hash": prompt_hash,
        "step1_pass_thresholds": {
            "explicit_error_rate_lt": 0.05,
            "overall_error_rate_lt": 0.10,
            "politeness_agreement_rate_ge": 0.90,
        },
        "models": model_runs,
        "politeness_pairwise_agreement": pairwise_pol,
    }
    summary_file = out_dir / f"summary_{dataset.split}.json"
    summary_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def merge_result_files_to_summary(
    dataset: F9Dataset,
    model_specs: list[str],
    out_dir: Path,
    prompt_version: str | None = None,
) -> dict[str, Any]:
    """Build combined summary JSON from existing per-model result JSON files in `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fallback_pv = get_prompt(prompt_version or DEFAULT_PROMPT_KEY)[0]
    fallback_ph = compute_prompt_hash(prompt_version=prompt_version or DEFAULT_PROMPT_KEY)
    model_runs: list[dict[str, Any]] = []
    predictions_index: dict[str, dict[str, list[dict[str, Any]] | None]] = {}
    expected_work_ids = [w.work_id for w in dataset.works]
    seen_pvs: set[str] = set()
    seen_phs: set[str] = set()

    for spec in model_specs:
        model_name, think, safe_tag = parse_model_spec(spec)
        result_file = out_dir / f"{safe_tag}_{dataset.split}.json"
        if not result_file.exists():
            raise FileNotFoundError(f"Expected model result file not found: {result_file}")
        report = json.loads(result_file.read_text(encoding="utf-8"))
        if report.get("work_ids_evaluated") != expected_work_ids:
            raise ValueError(
                f"Work IDs mismatch in {result_file.name}: "
                f"{report.get('work_ids_evaluated')} != {expected_work_ids}"
            )
        label = str(report.get("model_spec", spec))
        rt = report.get("runtime_totals") or {}
        metrics = report["metrics"]
        rep_pv = str(report.get("prompt_version") or fallback_pv)
        rep_ph = str(report.get("prompt_hash") or fallback_ph)
        seen_pvs.add(rep_pv)
        seen_phs.add(rep_ph)

        raw_outputs = report.get("raw_outputs") or {}
        if "any_context_window_warning" in rt:
            ctx_warn = bool(rt["any_context_window_warning"])
        else:
            inf_opts = report.get("inference_options") or {}
            rep_ctx = int(inf_opts.get("num_ctx") or 0)
            rep_pred = int(inf_opts.get("num_predict") or 0)
            ctx_warn = any(
                bool(rec.get("context_window_warning"))
                or (
                    rep_ctx > 0
                    and (int(rec.get("prompt_eval_count") or 0) + int(rec.get("eval_count") or 0))
                    >= rep_ctx
                )
                or (rep_pred > 0 and int(rec.get("eval_count") or 0) >= rep_pred)
                for rec in raw_outputs.values()
                if isinstance(rec, dict)
            )
        gate = report.get("gate") or evaluate_step1_gate(metrics)

        model_runs.append(
            _build_model_summary_entry(
                label=label,
                model_name=str(report.get("model_name", model_name)),
                think=report.get("think", think),
                result_file_name=result_file.name,
                elapsed_sec=float(rt.get("elapsed_sec", 0.0)),
                prompt_eval_count=int(rt.get("prompt_eval_count", 0)),
                eval_count=int(rt.get("eval_count", 0)),
                metrics=metrics,
                prompt_version=rep_pv,
                prompt_hash=rep_ph,
                context_window_warning=ctx_warn,
                gate=gate,
            )
        )
        predictions_index[label] = {
            wid: (raw_outputs.get(wid) or {}).get("parsed_utterances") for wid in expected_work_ids
        }

    summary_pv = next(iter(seen_pvs)) if len(seen_pvs) == 1 else ("mixed" if seen_pvs else fallback_pv)
    summary_ph = next(iter(seen_phs)) if len(seen_phs) == 1 else ("mixed" if seen_phs else fallback_ph)
    return _write_summary(dataset, out_dir, summary_pv, summary_ph, model_runs, predictions_index)


def run_evaluation(
    dataset: F9Dataset,
    model_specs: list[str],
    out_dir: Path,
    ollama_url: str,
    num_ctx: int,
    num_predict: int,
    seed: int,
    timeout_sec: float,
    dry_run: bool,
    prompt_version: str = DEFAULT_PROMPT_KEY,
    per_model_options: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run all specified models against `dataset` and write per-model and summary JSONs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ver_label, _ = get_prompt(prompt_version)
    prompt_hash = compute_prompt_hash(prompt_version=prompt_version)
    format_schema = build_ollama_utterances_format_schema()
    validator = load_sensor_v2_validator()

    if dry_run:
        work_rows = [
            {
                "work_id": w.work_id,
                "author": w.author,
                "title": w.title,
                "narration": w.narration,
                "narrator": w.narrator,
                "pov": w.pov,
                "characters": len(w.characters),
                "total_quotes": len(w.utterances),
                "attributed": sum(1 for u in w.utterances if u.status == "attributed"),
                "undetermined": sum(1 for u in w.utterances if u.status == "undetermined"),
                "not_speech": sum(1 for u in w.utterances if u.status == "not_speech"),
                "scene_hash": w.scene_hash,
            }
            for w in dataset.works
        ]
        summary = {
            "dry_run": True,
            "split": dataset.split,
            "split_version": dataset.split_version,
            "gold_dir": dataset.gold_dir,
            "gold_json_sha256": dataset.gold_json_sha256,
            "prompt_version": ver_label,
            "prompt_hash": prompt_hash,
            "works": work_rows,
            "totals": {
                "works": len(dataset.works),
                "total_quotes": sum(r["total_quotes"] for r in work_rows),
                "attributed": sum(r["attributed"] for r in work_rows),
                "undetermined": sum(r["undetermined"] for r in work_rows),
                "not_speech": sum(r["not_speech"] for r in work_rows),
            },
        }
        return summary

    model_runs: list[dict[str, Any]] = []
    predictions_index: dict[str, dict[str, list[dict[str, Any]] | None]] = {}

    for spec in model_specs:
        model_name, think, safe_tag = parse_model_spec(spec)
        label = spec
        m_opts = (per_model_options or {}).get(spec, {})
        eff_num_ctx = int(m_opts.get("num_ctx", num_ctx))
        eff_num_predict = int(m_opts.get("num_predict", num_predict))
        eff_seed = int(m_opts.get("seed", seed))
        eff_timeout = float(m_opts.get("timeout", timeout_sec))

        predictions_by_work: dict[str, list[dict[str, Any]] | None] = {}
        raw_work_records: dict[str, Any] = {}
        total_elapsed = 0.0
        total_prompt_tokens = 0
        total_eval_tokens = 0
        any_ctx_warning = False

        for work in dataset.works:
            messages = build_work_messages(work, prompt_version=prompt_version)
            resp = call_ollama_chat(
                model=model_name,
                messages=messages,
                format_schema=format_schema,
                ollama_url=ollama_url,
                num_ctx=eff_num_ctx,
                num_predict=eff_num_predict,
                seed=eff_seed,
                temperature=0.0,
                timeout_sec=eff_timeout,
                think=think,
            )
            utts = _extract_utterances(resp.parsed_json)
            predictions_by_work[work.work_id] = utts
            total_elapsed += resp.elapsed_sec
            p_tok = int(resp.prompt_eval_count or 0)
            e_tok = int(resp.eval_count or 0)
            total_prompt_tokens += p_tok
            total_eval_tokens += e_tok
            ctx_warn = bool((p_tok + e_tok) >= eff_num_ctx or e_tok >= eff_num_predict)
            if ctx_warn:
                any_ctx_warning = True

            raw_work_records[work.work_id] = {
                "format_mode": resp.format_mode,
                "elapsed_sec": resp.elapsed_sec,
                "prompt_eval_count": resp.prompt_eval_count,
                "eval_count": resp.eval_count,
                "effective_num_ctx": eff_num_ctx,
                "effective_num_predict": eff_num_predict,
                "effective_timeout": eff_timeout,
                "context_window_warning": ctx_warn,
                "total_duration_ns": resp.total_duration_ns,
                "load_duration_ns": resp.load_duration_ns,
                "eval_duration_ns": resp.eval_duration_ns,
                "parse_error": resp.parse_error,
                "thinking_chars": len(resp.thinking_content) if resp.thinking_content else 0,
                "parsed_utterances": utts,
                "raw_content": resp.raw_content,
            }

        metrics = evaluate_model_predictions(
            works=dataset.works,
            predictions_by_work=predictions_by_work,
            model_id=label,
            prompt_hash=prompt_hash,
            schema_validator=validator,
        )
        gate = evaluate_step1_gate(metrics)
        predictions_index[label] = predictions_by_work

        model_report = {
            "run_timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "split": dataset.split,
            "split_version": dataset.split_version,
            "gold_json_sha256": dataset.gold_json_sha256,
            "work_ids_evaluated": [w.work_id for w in dataset.works],
            "model_spec": label,
            "model_name": model_name,
            "think": think,
            "prompt_version": ver_label,
            "prompt_hash": prompt_hash,
            "inference_options": {
                "num_ctx": eff_num_ctx,
                "num_predict": eff_num_predict,
                "seed": eff_seed,
                "timeout": eff_timeout,
                "temperature": 0.0,
            },
            "runtime_totals": {
                "elapsed_sec": round(total_elapsed, 3),
                "prompt_eval_count": total_prompt_tokens,
                "eval_count": total_eval_tokens,
                "any_context_window_warning": any_ctx_warning,
            },
            "gate": gate,
            "metrics": metrics,
            "raw_outputs": raw_work_records,
        }
        out_file = out_dir / f"{safe_tag}_{dataset.split}.json"
        out_file.write_text(
            json.dumps(model_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        model_runs.append(
            _build_model_summary_entry(
                label=label,
                model_name=model_name,
                think=think,
                result_file_name=out_file.name,
                elapsed_sec=total_elapsed,
                prompt_eval_count=total_prompt_tokens,
                eval_count=total_eval_tokens,
                metrics=metrics,
                prompt_version=ver_label,
                prompt_hash=prompt_hash,
                context_window_warning=any_ctx_warning,
                gate=gate,
            )
        )

    return _write_summary(dataset, out_dir, ver_label, prompt_hash, model_runs, predictions_index)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    root = _default_repo_root()
    parser = argparse.ArgumentParser(description="Run Step 1 F9 speaker/politeness evaluation.")
    parser.add_argument("--config", default=None, help="Optional JSON config file for model-swap regression runs")
    parser.add_argument("--gold-dir", default=None, help="Directory containing F9 gold.json and *.txt")
    parser.add_argument("--split-file", default=None, help="Path to f9_split.json")
    parser.add_argument("--split", choices=["dev", "holdout"], default=None)
    parser.add_argument(
        "--unlock-holdout",
        action="store_true",
        help="Required to unlock sealed holdout split.",
    )
    parser.add_argument("--politeness-gold", default=None, help="Optional politeness gold JSON path")
    parser.add_argument("--nishiki-bin", default=None, help="Optional nishiki binary for segment cross-check")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Ollama model specs (e.g. qwen3:8b::think=true qwen2.5:14b qwen3:14b)",
    )
    parser.add_argument("--prompt-version", default=None, help="Prompt version key or label (e.g. v1, v2)")
    parser.add_argument("--ollama-url", default=None)
    parser.add_argument("--num-ctx", type=int, default=None)
    parser.add_argument("--num-predict", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--out-dir", default=None, help="Directory for JSON outputs")
    parser.add_argument("--dry-run", action="store_true", help="Verify data & split without calling Ollama")
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Rebuild summary JSON from existing per-model result files without calling Ollama",
    )
    parser.add_argument(
        "--check-gate",
        action="store_true",
        help="Exit 0 if any evaluated model passes speaker gate (<5% explicit, <10% overall, valid), else exit 2",
    )

    args = parser.parse_args(argv)
    cfg: dict[str, Any] = load_eval_config(Path(args.config)) if args.config else {}

    gold_dir = args.gold_dir if args.gold_dir is not None else cfg.get("gold_dir")
    if not gold_dir:
        parser.error("--gold-dir is required unless specified in --config")

    split_file = (
        args.split_file
        if args.split_file is not None
        else cfg.get("split_file", str(root / "scripts" / "step1" / "f9_split.json"))
    )
    split = args.split if args.split is not None else cfg.get("split", "dev")
    politeness_gold = args.politeness_gold if args.politeness_gold is not None else cfg.get("politeness_gold")
    nishiki_bin = args.nishiki_bin if args.nishiki_bin is not None else cfg.get("nishiki_bin")
    model_specs = args.models if args.models is not None else cfg.get("models", ["qwen3:8b::think=false"])
    prompt_version = args.prompt_version if args.prompt_version is not None else cfg.get("prompt_version", DEFAULT_PROMPT_KEY)
    ollama_url = args.ollama_url if args.ollama_url is not None else cfg.get("ollama_url", "http://127.0.0.1:11434")
    num_ctx = args.num_ctx if args.num_ctx is not None else int(cfg.get("num_ctx", 8192))
    num_predict = args.num_predict if args.num_predict is not None else int(cfg.get("num_predict", 4096))
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    timeout = args.timeout if args.timeout is not None else float(cfg.get("timeout", 240.0))
    out_dir = Path(
        args.out_dir
        if args.out_dir is not None
        else cfg.get("out_dir", str(root / "scripts" / "step1" / "results"))
    )

    per_model_options: dict[str, dict[str, Any]] = {
        k: dict(v) for k, v in (cfg.get("per_model_options") or {}).items()
    }
    for cli_val, opt_key in (
        (args.num_ctx, "num_ctx"),
        (args.num_predict, "num_predict"),
        (args.seed, "seed"),
        (args.timeout, "timeout"),
    ):
        if cli_val is not None:
            for opts in per_model_options.values():
                opts.pop(opt_key, None)

    dataset = load_f9_dataset(
        gold_dir=gold_dir,
        split_file=split_file,
        split=split,
        unlock_holdout=args.unlock_holdout,
        politeness_gold_path=politeness_gold,
        nishiki_bin=nishiki_bin,
    )
    if args.merge_only:
        summary = merge_result_files_to_summary(
            dataset=dataset,
            model_specs=model_specs,
            out_dir=out_dir,
            prompt_version=prompt_version,
        )
    else:
        summary = run_evaluation(
            dataset=dataset,
            model_specs=model_specs,
            out_dir=out_dir,
            ollama_url=ollama_url,
            num_ctx=num_ctx,
            num_predict=num_predict,
            seed=seed,
            timeout_sec=timeout,
            dry_run=args.dry_run,
            prompt_version=prompt_version,
            per_model_options=per_model_options,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.check_gate and not args.dry_run:
        any_passed = any(
            bool((m.get("gate") or {}).get("speaker_gate_passed"))
            for m in summary.get("models", [])
        )
        return 0 if any_passed else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
