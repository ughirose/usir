"""Local-side utterance annotation for a work that has NO F9 gold, using the Step 1 three-stage pipeline
(stage 1 qwen2.5:14b JSON extraction -> stage 2 deterministic postprocess -> stage 3 per-utterance speaker recheck),
with the settings of scripts/step1/step1_eval_config_v5_threestage.json.

Difference from local_from_step1.py: no gold is read at all. The work is cut into scene-aligned chunks
(<= --max-chars characters and <= --max-quotes outer quotes) because the frozen config (num_ctx 9216, num_predict 4096)
cannot take a whole 30k-character work in one call. Each chunk is run as one pseudo-work with placeholder gold
(status=undetermined, speaker=None); predictions never use the placeholder.
Writes scripts/sensor_runner/local/<work>.utterances.json (sensor.local_utterances/1) after every chunk, and
raw per-chunk records to --out-dir.

Usage: python local_from_step1_nogold.py WORK_ID [--max-chars 6000] [--max-quotes 55] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import REPO, find_nishiki, load_work, segment, write_json  # noqa: E402

sys.path.insert(0, str(REPO / "python"))
from usir_py.sensor_eval.f9_loader import (  # noqa: E402
    F9Character,
    F9UtteranceGold,
    F9Work,
    extract_outer_quote_spans,
    number_outer_quotes,
)
from usir_py.sensor_eval.features import (  # noqa: E402
    FIRST_PERSON_NARRATION_PRONOUNS,
    _find_first_unblocked_pronoun,
    extract_work_utterance_features,
    find_speakers_in_text,
)
from usir_py.sensor_eval.ollama_client import call_ollama_chat, parse_model_spec  # noqa: E402
from usir_py.sensor_eval.postprocess import PostprocessConfig, apply_postprocessing_to_work  # noqa: E402
from usir_py.sensor_eval.prompt import build_ollama_utterances_format_schema, build_work_messages  # noqa: E402
from usir_py.sensor_eval.run_step1_f9 import _extract_utterances, load_eval_config  # noqa: E402
from usir_py.sensor_eval.speaker_recheck import SpeakerRecheckConfig, apply_speaker_recheck_to_work  # noqa: E402

CONFIG = REPO / "scripts" / "step1" / "step1_eval_config_v5_threestage.json"
_U_MARKER_RE = re.compile(r"\[U(\d+)\]")
_FIRST_PERSON_QUOTE_PRONOUNS: tuple[str, ...] = FIRST_PERSON_NARRATION_PRONOUNS + (
    "我々",
    "われわれ",
    "わし",
    "おいら",
    "うち",
)


def make_chunks(scenes, max_chars, max_quotes):
    chunks, cur, c_chars, c_q = [], [], 0, 0
    for s in scenes:
        nq, nc = len(s["utterances"]), len(s["scene_text"])
        if cur and (c_chars + nc > max_chars or c_q + nq > max_quotes):
            chunks.append(cur)
            cur, c_chars, c_q = [], 0, 0
        cur.append(s)
        c_chars += nc
        c_q += nq
    if cur:
        chunks.append(cur)
    return chunks


def join_scenes_text(scenes, raw_text: str | None = None) -> str:
    """Join consecutive scenes preserving exact original newlines via `raw_text` spans when valid."""
    if not scenes:
        return ""
    if raw_text is not None:
        raw_len = len(raw_text)
        spans = [s.get("span") for s in scenes]
        valid = all(
            isinstance(sp, dict)
            and isinstance(sp.get("start"), int)
            and isinstance(sp.get("end"), int)
            and 0 <= sp["start"] <= sp["end"] <= raw_len
            for sp in spans
        )
        if valid and all(spans[i - 1]["end"] <= spans[i]["start"] for i in range(1, len(spans))):
            sliced = raw_text[spans[0]["start"] : spans[-1]["end"]]
            return sliced.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(s["scene_text"] for s in scenes)


def narration_only_text(text: str) -> str:
    """Return non-quote narrative text by excluding outermost 「...」/『...』 spans."""
    spans = extract_outer_quote_spans(text)
    if not spans:
        return text
    parts: list[str] = []
    cur = 0
    for start, end in spans:
        if start > cur:
            parts.append(text[cur:start])
        cur = end
    if cur < len(text):
        parts.append(text[cur:])
    return "\n".join(parts)


def narration_speakers_in_text(
    text: str,
    characters: list[F9Character],
    narration: str,
    narrator: str | None,
) -> tuple[str, ...]:
    """Return roster speaker IDs whose aliases appear in narration (outside quotes)."""
    return find_speakers_in_text(narration_only_text(text), characters, narration, narrator)


def scene_has_opening_narration_speaker(
    scene: dict,
    raw_text: str | None,
    characters: list[F9Character],
    narration: str,
    narrator: str | None,
    *,
    opening_quotes: int = 5,
) -> bool:
    """Return True if a roster character alias appears in narration up through after the `opening_quotes`-th quote."""
    scene_text = join_scenes_text([scene], raw_text)
    spans = extract_outer_quote_spans(scene_text)
    if not spans or opening_quotes <= 0:
        return bool(narration_speakers_in_text(scene_text, characters, narration, narrator))
    end_pos = spans[opening_quotes][0] if len(spans) > opening_quotes else len(scene_text)
    return bool(narration_speakers_in_text(scene_text[:end_pos], characters, narration, narrator))


def quote_has_first_person_pronoun(quote_text: str) -> bool:
    """Return True if `quote_text` explicitly contains a Japanese first-person pronoun."""
    inner = quote_text[1:-1] if len(quote_text) >= 2 and quote_text[0] in "「『" and quote_text[-1] in "」』" else quote_text
    return any(_find_first_unblocked_pronoun(inner, p) is not None for p in _FIRST_PERSON_QUOTE_PRONOUNS)


def _trim_anchor_line(line: str, cap: int, characters: list[F9Character], narration: str, narrator: str | None) -> str:
    if len(line) <= cap:
        return line
    tail = line[-cap:]
    if narration_speakers_in_text(tail, characters, narration, narrator):
        return tail
    return line[:cap]


def build_lookback_context(
    all_scenes,
    chunk_start_idx: int,
    raw_text: str | None,
    characters: list[F9Character],
    narration: str,
    narrator: str | None,
    *,
    max_chars: int = 1500,
) -> tuple[str, list[int]]:
    """Collect raw text backwards from `chunk_start_idx` to the nearest scene with narration character names."""
    if chunk_start_idx <= 0 or max_chars <= 0:
        return "", []

    anchor_idx = 0
    for k in range(chunk_start_idx - 1, -1, -1):
        scene_text = join_scenes_text([all_scenes[k]], raw_text)
        if narration_speakers_in_text(scene_text, characters, narration, narrator):
            anchor_idx = k
            break

    ctx_scenes = all_scenes[anchor_idx:chunk_start_idx]
    ctx = join_scenes_text(ctx_scenes, raw_text).strip()
    if len(ctx) > max_chars:
        tail = ctx[-max_chars:]
        nl = tail.find("\n")
        trimmed = tail[nl + 1 :].strip() if (nl != -1 and tail[nl + 1 :].strip()) else tail
        has_raw_anchor = bool(narration_speakers_in_text(ctx, characters, narration, narrator))
        if not has_raw_anchor or narration_speakers_in_text(trimmed, characters, narration, narrator):
            ctx = trimmed
        else:
            anchor_text = join_scenes_text([all_scenes[anchor_idx]], raw_text)
            anchor_line = ""
            for line in reversed(anchor_text.splitlines()):
                if narration_speakers_in_text(line, characters, narration, narrator):
                    anchor_line = line.strip()
                    break
            if not anchor_line:
                anchor_line = narration_only_text(anchor_text).strip()
            anchor_line = _trim_anchor_line(anchor_line, min(400, max_chars // 3), characters, narration, narrator)
            sep = "\n……\n"
            rem = max_chars - len(anchor_line) - len(sep)
            if rem > 40:
                tail_part = ctx[-rem:]
                nl2 = tail_part.find("\n")
                if nl2 != -1 and tail_part[nl2 + 1 :].strip():
                    tail_part = tail_part[nl2 + 1 :].strip()
                ctx = f"{anchor_line}{sep}{tail_part}"
            else:
                ctx = anchor_line[:max_chars]

    return _U_MARKER_RE.sub(r"(U\1)", ctx), [s["index"] for s in ctx_scenes]


def format_numbered_with_context(numbered_body: str, context_text: str) -> str:
    """Prepend unnumbered raw reference context before the numbered chunk body."""
    if not context_text:
        return numbered_body
    return (
        "### 直前の参照文脈（注釈対象外・番号なし原文）\n"
        f"{context_text}\n\n"
        "### 注釈対象本文（番号付き発話のみ出力対象）\n"
        f"{numbered_body}"
    )


def all_null_scene_warnings(
    scenes,
    stage1,
    final=None,
    *,
    features=None,
    min_standalone_lines: int = 5,
    high_null_rate: float = 0.8,
    long_quote_min_chars: int = 10,
):
    """Detect stage-1 all-null, high-null, or politeness-template-collapse scenes.

    `scenes` is the chunk's scene list in order, as (scene_index, scene_hash, utterance_count); utterance numbers run
    through the chunk. When `features` is None (legacy signature), flags scenes where all utterances have speaker_id
    null. When `features` is provided, requires `standalone_line >= min_standalone_lines` to avoid false positives on
    essays with inline quotes, and flags `null_rate >= high_null_rate` or all long quotes (`quote_chars >=
    long_quote_min_chars`) collapsed to `politeness == "unknown"`.
    """
    s1 = {int(u["utterance_index"]): u for u in stage1 or []}
    fin = {int(u["utterance_index"]): u for u in final or []}
    feat_by_idx = {f.utterance_index: f for f in features} if features is not None else None
    out, n = [], 0
    for idx, h, cnt in scenes:
        rng = range(n, n + cnt)
        n += cnt
        if cnt == 0:
            continue
        if feat_by_idx is None:
            if any((s1.get(i) or {}).get("speaker_id") is not None for i in rng):
                continue
            out.append({"warning": "stage1_all_speakers_null", "scene_index": idx, "scene_hash": h, "utterances": cnt,
                        "final_speaker_null": sum(1 for i in rng if (fin.get(i) or {}).get("speaker_id") is None)})
            continue

        null_cnt = sum(1 for i in rng if (s1.get(i) or {}).get("speaker_id") is None)
        null_rate = null_cnt / cnt
        standalone_cnt = sum(
            1 for i in rng
            if feat_by_idx.get(i) is not None and feat_by_idx[i].position == "standalone_line"
        )
        long_idxs = [
            i for i in rng
            if feat_by_idx.get(i) is not None and feat_by_idx[i].quote_chars >= long_quote_min_chars
        ]
        long_pol_unknown_cnt = sum(
            1 for i in long_idxs
            if (((s1.get(i) or {}).get("register") or {}).get("politeness") == "unknown")
        )
        is_all_null = null_cnt == cnt
        is_high_null = standalone_cnt >= min_standalone_lines and null_rate >= high_null_rate
        is_pol_collapse = (
            standalone_cnt >= min_standalone_lines
            and len(long_idxs) > 0
            and long_pol_unknown_cnt == len(long_idxs)
        )
        reasons: list[str] = []
        if is_high_null and is_all_null:
            reasons.append("all_speakers_null")
        elif is_high_null:
            reasons.append("high_null_rate")
        if is_pol_collapse:
            reasons.append("all_long_politeness_unknown")
        if not reasons:
            continue

        if is_high_null and is_all_null:
            warning = "stage1_all_speakers_null"
        elif is_high_null:
            warning = "stage1_high_null_speakers"
        else:
            warning = "stage1_template_collapse"

        out.append({
            "warning": warning,
            "reasons": reasons,
            "scene_index": idx,
            "scene_hash": h,
            "utterances": cnt,
            "standalone_lines": standalone_cnt,
            "stage1_speaker_null": null_cnt,
            "stage1_null_rate": round(null_rate, 4),
            "long_quotes": len(long_idxs),
            "long_politeness_unknown": long_pol_unknown_cnt,
            "final_speaker_null": sum(1 for i in rng if (fin.get(i) or {}).get("speaker_id") is None),
        })
    return out


def _stage1_rank(
    raw_utts,
    exp_cnt: int,
    warns: list[dict],
    *,
    features=None,
    utterances=None,
    long_quote_min_chars: int = 10,
    min_standalone_lines: int = 5,
) -> tuple[int, int, int, int, int, int]:
    """Lower-is-better deterministic anomaly score for selecting between no-context and with-context Stage 1 runs.

    Components:
    (0) missing utterance indices
    (1) speaker_id is None count (missing counts as null)
    (2) A2 warning count (all-null / high-null / politeness template collapse)
    (3) long quotes (quote_chars >= long_quote_min_chars) with politeness == "unknown"
    (4) quotes explicitly containing a 1st-person pronoun whose register.first_person is None
    (5) addressee_id all-null collapse flag (1 if standalone_lines >= min_standalone_lines and all addressee_id are None)
    """
    by_i = {
        int(u["utterance_index"]): u
        for u in raw_utts or []
        if isinstance(u, dict) and isinstance(u.get("utterance_index"), int)
    }
    missing = sum(1 for i in range(exp_cnt) if i not in by_i)
    nulls = sum(1 for i in range(exp_cnt) if (by_i.get(i) or {}).get("speaker_id") is None)
    if features is not None:
        long_pol_unk = sum(
            1 for f in features
            if f.quote_chars >= long_quote_min_chars
            and (((by_i.get(f.utterance_index) or {}).get("register") or {}).get("politeness") == "unknown")
        )
        standalone_cnt = sum(1 for f in features if f.position == "standalone_line")
    else:
        long_pol_unk = sum(int(w.get("long_politeness_unknown", 0)) for w in warns)
        standalone_cnt = 0

    if utterances is not None:
        fp_missed = sum(
            1 for u in utterances
            if quote_has_first_person_pronoun(u.quote_text)
            and (((by_i.get(u.n) or {}).get("register") or {}).get("first_person") is None)
        )
    else:
        fp_missed = 0

    addr_all_null = int(
        standalone_cnt >= min_standalone_lines
        and exp_cnt > 0
        and all((by_i.get(i) or {}).get("addressee_id") is None for i in range(exp_cnt))
    )
    return (missing, nulls, len(warns), long_pol_unk, fp_missed, addr_all_null)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("work_id")
    ap.add_argument("--max-chars", type=int, default=6000)
    ap.add_argument("--max-quotes", type=int, default=55)
    ap.add_argument("--max-context-chars", type=int, default=1500)
    ap.add_argument("--opening-cue-quotes", type=int, default=5)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--local-out", default=None)
    a = ap.parse_args()

    out_dir = Path(a.out_dir) if a.out_dir else HERE / "results" / "local_step1" / a.work_id
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_eval_config(CONFIG)
    model_spec = cfg["models"][0]
    model_name, think, _ = parse_model_spec(model_spec)
    opts = cfg["per_model_options"].get(model_spec, {})
    num_ctx, num_predict = int(opts.get("num_ctx", cfg["num_ctx"])), int(opts.get("num_predict", cfg["num_predict"]))
    seed, timeout = int(opts.get("seed", cfg["seed"])), float(opts.get("timeout", cfg["timeout"]))
    pp_cfg = PostprocessConfig.from_dict(cfg["postprocess"])
    sr_cfg = SpeakerRecheckConfig.from_dict(cfg["speaker_recheck"])
    fmt = build_ollama_utterances_format_schema()

    work = load_work(a.work_id)
    raw_text = work["text_abs"].read_bytes().decode("utf-8")
    seg = segment(find_nishiki(), work["text_abs"])
    narration = str(work.get("narration") or "third")
    narrator = work.get("narrator")
    chars = [F9Character(id=c["id"], aliases=list(c.get("aliases", [])), gender=str(c.get("gender", "Unknown"))) for c in work["characters"]]
    chunks = make_chunks(seg["scenes"], a.max_chars, a.max_quotes)
    print(f"scenes={len(seg['scenes'])} chunks={len(chunks)} quotes={sum(len(s['utterances']) for s in seg['scenes'])}", flush=True)

    local_out = Path(a.local_out) if a.local_out else HERE / "local" / f"{a.work_id}.utterances.json"
    scenes_out: dict[str, list[dict]] = {}
    totals = {"elapsed_wall_s": 0.0, "stage1_s": 0.0, "stage3_s": 0.0, "stage1_prompt_tok": 0, "stage1_eval_tok": 0,
              "stage3_prompt_tok": 0, "stage3_eval_tok": 0, "pp_forced": 0, "pp_changed": 0, "pp_downgraded": 0,
              "sr_target": 0, "sr_rechecked": 0, "sr_changed": 0, "sr_rejected": 0, "ctx_warn_chunks": 0, "parse_error_chunks": 0,
              "stage1_all_null_scenes": 0, "stage1_anomaly_warnings": 0,
              "c1_context_chunks": 0, "c1_dual_run_chunks": 0, "c1_context_adopted": 0, "c1_no_context_kept": 0,
              "c2_retry_chunks": 0, "c2_retry_adopted": 0,
              "dual_pass2_s": 0.0, "dual_pass2_prompt_tok": 0, "dual_pass2_eval_tok": 0}
    chunk_log = []
    warnings: list[dict] = []
    t_all = time.time()
    chunk_start_idx = 0

    for ci, ch in enumerate(chunks):
        t0 = time.time()
        text = join_scenes_text(ch, raw_text)
        numbered, spans = number_outer_quotes(text)
        exp = sum(len(s["utterances"]) for s in ch)
        if len(spans) != exp:
            raise SystemExit(f"chunk {ci}: outer quotes {len(spans)} != segment utterances {exp}")
        gold = [
            F9UtteranceGold(n=i, open_offset=sp[0], close_offset=sp[1], head="", status="undetermined", speaker=None, quote_text=text[sp[0]:sp[1]])
            for i, sp in enumerate(spans)
        ]
        scene_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        def _make_work(ctx_text: str) -> F9Work:
            return F9Work(
                work_id=f"{a.work_id}#c{ci}", author=None, title=None, narration=narration,
                narrator=narrator, pov=work.get("pov"), characters=chars, text=text,
                numbered_text=format_numbered_with_context(numbered, ctx_text),
                scene_hash=scene_hash, utterances=gold,
            )

        feats = extract_work_utterance_features(_make_work(""))
        chunk_nar_spks = list(narration_speakers_in_text(text, chars, narration, narrator))
        first_scene_has_opening_cue = scene_has_opening_narration_speaker(
            ch[0], raw_text, chars, narration, narrator, opening_quotes=a.opening_cue_quotes,
        )

        # Pass 1: always run without lookback context first.
        w = _make_work("")
        resp = call_ollama_chat(model=model_name, messages=build_work_messages(w, prompt_version=cfg["prompt_version"]),
                                format_schema=fmt, ollama_url=cfg["ollama_url"], num_ctx=num_ctx, num_predict=num_predict,
                                seed=seed, temperature=0.0, timeout_sec=timeout, think=think)
        raw = _extract_utterances(resp.parsed_json)
        p_tok, e_tok = int(resp.prompt_eval_count or 0), int(resp.eval_count or 0)
        chunk_stage1_s = float(resp.elapsed_sec)
        chunk_p_tok, chunk_e_tok = p_tok, e_tok
        chunk_ctx_warn = bool(p_tok + e_tok >= num_ctx or e_tok >= num_predict)
        chunk_parse_error = resp.parse_error
        chunk_attempts = 1
        context_text, context_scenes = ("", [])
        context_source = "none"

        scene_meta = [(s["index"], s["scene_hash"], len(s["utterances"])) for s in ch]
        s1_warn1 = all_null_scene_warnings(scene_meta, raw, features=feats)
        rank1 = _stage1_rank(raw, exp, s1_warn1, features=feats, utterances=gold)

        c1_trigger = (not first_scene_has_opening_cue) and (chunk_start_idx > 0)
        c2_trigger = (not c1_trigger) and bool(s1_warn1) and (chunk_start_idx > 0)
        dual_run = None
        if c1_trigger or c2_trigger:
            cand_ctx, cand_ctx_scenes = build_lookback_context(
                seg["scenes"], chunk_start_idx, raw_text, chars, narration, narrator, max_chars=a.max_context_chars,
            )
            if cand_ctx:
                chunk_attempts = 2
                w_ctx = _make_work(cand_ctx)
                resp2 = call_ollama_chat(model=model_name, messages=build_work_messages(w_ctx, prompt_version=cfg["prompt_version"]),
                                         format_schema=fmt, ollama_url=cfg["ollama_url"], num_ctx=num_ctx, num_predict=num_predict,
                                         seed=seed, temperature=0.0, timeout_sec=timeout, think=think)
                raw2 = _extract_utterances(resp2.parsed_json)
                p_tok2, e_tok2 = int(resp2.prompt_eval_count or 0), int(resp2.eval_count or 0)
                chunk_stage1_s += float(resp2.elapsed_sec)
                chunk_p_tok += p_tok2
                chunk_e_tok += e_tok2
                totals["dual_pass2_s"] += float(resp2.elapsed_sec)
                totals["dual_pass2_prompt_tok"] += p_tok2
                totals["dual_pass2_eval_tok"] += e_tok2
                ctx_warn2 = bool(p_tok2 + e_tok2 >= num_ctx or e_tok2 >= num_predict)
                s1_warn2 = all_null_scene_warnings(scene_meta, raw2, features=feats)
                rank2 = _stage1_rank(raw2, exp, s1_warn2, features=feats, utterances=gold)
                adopted = (not resp2.parse_error) and (rank2 < rank1)
                if resp2.parse_error:
                    selection_reason = "parse_error_on_context"
                elif adopted:
                    selection_reason = "context_lower_anomaly"
                else:
                    selection_reason = "no_context_tie_or_better"

                if c1_trigger:
                    totals["c1_dual_run_chunks"] += 1
                    if adopted:
                        totals["c1_context_adopted"] += 1
                        totals["c1_context_chunks"] += 1
                    else:
                        totals["c1_no_context_kept"] += 1
                else:
                    totals["c2_retry_chunks"] += 1
                    if adopted:
                        totals["c2_retry_adopted"] += 1

                dual_run = {
                    "trigger": "c1_no_opening_cue" if c1_trigger else "c2_a2_anomaly",
                    "context_scenes": cand_ctx_scenes,
                    "context_chars": len(cand_ctx),
                    "pass1_warnings": [x["warning"] for x in s1_warn1],
                    "pass2_warnings": [x["warning"] for x in s1_warn2],
                    "pass1_rank": list(rank1),
                    "pass2_rank": list(rank2),
                    "pass1_elapsed_s": resp.elapsed_sec,
                    "pass2_elapsed_s": resp2.elapsed_sec,
                    "pass1_prompt_tok": p_tok,
                    "pass1_eval_tok": e_tok,
                    "pass2_prompt_tok": p_tok2,
                    "pass2_eval_tok": e_tok2,
                    "pass2_ctx_warning": ctx_warn2,
                    "pass2_parse_error": resp2.parse_error,
                    "adopted": adopted,
                    "selection_reason": selection_reason,
                }
                if adopted:
                    w, resp, raw = w_ctx, resp2, raw2
                    context_text, context_scenes = cand_ctx, cand_ctx_scenes
                    context_source = "c1_dual_adopted" if c1_trigger else "c2_retry"
                    chunk_ctx_warn = ctx_warn2
                    chunk_parse_error = resp2.parse_error

        s2, pp_audit = apply_postprocessing_to_work(w, raw, pp_cfg, features=feats) if pp_cfg.is_enabled else (raw, None)
        final, sr_audit = apply_speaker_recheck_to_work(w, s2, sr_cfg, ollama_url=cfg["ollama_url"], features=feats) if sr_cfg.is_enabled else (s2, None)
        by_n = {int(u["utterance_index"]): u for u in final or []}
        missing = sum(1 for i in range(exp) if i not in by_n)
        n = 0
        for s in ch:
            rows = []
            for u in s["utterances"]:
                p = by_n.get(n)
                rows.append({"utterance_index": u["index"], "speaker_id": p.get("speaker_id") if p else None,
                             "addressee_id": p.get("addressee_id") if p else None,
                             "evidence": (p.get("evidence") if p else None) or "ambiguous"})
                n += 1
            scenes_out[s["scene_hash"]] = rows
        dt = time.time() - t0
        totals["stage1_s"] += chunk_stage1_s
        totals["stage1_prompt_tok"] += chunk_p_tok
        totals["stage1_eval_tok"] += chunk_e_tok
        if pp_audit:
            totals["pp_forced"] += len(pp_audit["inline_non_speech_forced"])
            totals["pp_changed"] += sum(1 for r in pp_audit["inline_non_speech_forced"] if r["changed"])
            totals["pp_downgraded"] += len(pp_audit["explicit_downgraded"])
        if sr_audit:
            totals["stage3_s"] += float(sr_audit["elapsed_sec"])
            totals["stage3_prompt_tok"] += int(sr_audit["prompt_eval_count"])
            totals["stage3_eval_tok"] += int(sr_audit["eval_count"])
            totals["sr_target"] += int(sr_audit["target_count"])
            totals["sr_rechecked"] += int(sr_audit["rechecked_count"])
            totals["sr_changed"] += int(sr_audit["changed_count"])
            totals["sr_rejected"] += int(sr_audit["rejected_count"])
        totals["ctx_warn_chunks"] += int(chunk_ctx_warn)
        totals["parse_error_chunks"] += int(bool(chunk_parse_error))
        null_warn = [{"chunk": ci, **x} for x in all_null_scene_warnings(scene_meta, raw, final, features=feats)]
        warnings.extend(null_warn)
        row = {"chunk": ci, "scenes": [s["index"] for s in ch], "chars": len(text), "quotes": exp, "missing_predictions": missing,
               "wall_s": round(dt, 1), "stage1_s": round(chunk_stage1_s, 3), "stage1_attempts": chunk_attempts,
               "stage1_prompt_tok": chunk_p_tok, "stage1_eval_tok": chunk_e_tok, "ctx_warning": chunk_ctx_warn,
               "parse_error": chunk_parse_error, "first_scene_has_opening_cue": first_scene_has_opening_cue,
               "context_source": context_source, "context_scenes": context_scenes,
               "context_chars": len(context_text), "chunk_narration_speakers": chunk_nar_spks,
               "pass1_rank": list(rank1),
               "pass2_rank": dual_run["pass2_rank"] if dual_run else None,
               "adopted_context": bool(dual_run and dual_run["adopted"]),
               "stage1_all_null_scenes": [x["scene_index"] for x in null_warn if x["warning"] == "stage1_all_speakers_null"],
               "stage1_anomaly_scenes": [{"scene_index": x["scene_index"], "warning": x["warning"]} for x in null_warn]}
        chunk_log.append(row)
        c2_compat = dual_run if c2_trigger else None
        write_json(out_dir / f"chunk_{ci:03d}.json", {**row, "dual_run": dual_run, "c2_retry": c2_compat, "stage1_parsed": raw, "final": final, "pp_audit": pp_audit, "sr_audit": sr_audit})
        write_json(local_out, {"schema": "sensor.local_utterances/1",
                               "provenance": f"step1-threestage-nogold:{model_spec} (chunked; stage1 + postprocess + stage3 recheck_v2), config {CONFIG.name}, chunks_done {ci + 1}/{len(chunks)}",
                               "scenes": scenes_out})
        print(json.dumps(row, ensure_ascii=False), flush=True)
        chunk_start_idx += len(ch)

    totals["elapsed_wall_s"] = round(time.time() - t_all, 1)
    totals["dual_pass2_s"] = round(totals["dual_pass2_s"], 3)
    totals["stage1_all_null_scenes"] = sum(1 for x in warnings if x["warning"] == "stage1_all_speakers_null")
    totals["stage1_anomaly_warnings"] = len(warnings)
    write_json(out_dir / "summary.json", {"work_id": a.work_id, "model": model_spec, "num_ctx": num_ctx, "num_predict": num_predict,
                                           "max_chars": a.max_chars, "max_quotes": a.max_quotes, "max_context_chars": a.max_context_chars,
                                           "opening_cue_quotes": a.opening_cue_quotes,
                                           "chunks": chunk_log, "totals": totals, "warnings": warnings})
    for x in warnings:
        print(f"WARNING {x['warning']} for scene s{x['scene_index']} (chunk {x['chunk']}, reasons={x.get('reasons')}, "
              f"null={x.get('stage1_speaker_null')}/{x['utterances']}); stage 3 does not recheck nulls", flush=True)
    print("DONE", json.dumps(totals), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
