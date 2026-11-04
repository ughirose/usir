# USIR (Claude Code 用エントリ)

アーキテクチャ不変条件の正本は `.antigravity/rules.md`（Antigravity と共用）。必ず従うこと。

@.antigravity/rules.md

## 引き継ぎ（オンデマンド読み込み）
- 作業開始時に `docs/handoff/` の最新MDを読み、決定事項と未完了を引き継ぐ。
- 既存の詳細な現状文書: `HANDOFF_V4.md`（長いので必要時のみ読む）。
- 作業の節目に `docs/handoff/` の最新ファイルへ追記する（形式は `docs/handoff/README.md`）。

## Claude Code 向け補足
- 構成: pnpm monorepo（`packages/usir-core`, `usir-mcp`, `usir-bridges`）+ Python（`python/usir_py`, uv 管理）。
- コマンド:
  - TS テスト: `pnpm test`（個別: `pnpm test:core` / `test:mcp` / `test:bridges`）
  - Python テスト: `cd python && uv run pytest`
  - 全部: `pnpm test:all`
- TS 実装と Python 実装は数式パリティを保つ（例: `bidirectionalBridge.ts` と `bidirectional_bridge.py`）。片方を変えたら他方も確認する。
- `.antigravity/skills/`（cleanroom-auditor, usir-math-verifier）は Antigravity 用スキル。内容は SKILL.md を読めば手順として流用できる。
- 作業ツリーに未コミットの変更あり（benchmark artifacts, `realtime_decoder.py` 等）。触る前に `git status` / `git diff` で確認し、意図を確かめる。
- 重い実験は Colab で実行（`scripts/run_colab_benchmark.py`）。ローカルで GPU 実験をしない。
- 実測値の捏造・都合の良い報告は禁止。結果は限界・誤爆も含めて報告する。
