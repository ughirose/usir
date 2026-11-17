# STATUS_faw（USIR v5 Stage FAW 設計）

- 開始: 2026-11-17
- 作業ブランチ: `stage/faw-design`
- 起点: `stage/p2-gate3a` 先端 `f008df5`（P-2c 完了）。本流（main）は未マージ・未参照。
- 作業ディレクトリ: C:\dev\usir
- 制約: コード変更禁止、仕様書・討議パック編集禁止、本流マージ禁止。

## advisor
- `/advisor fable` は Claude Code のスラッシュコマンドであり、このセッションから実行できるツールが無い（ToolSearch "advisor" で該当なし）。
- 代替: Agent（model=fable）に相談内容を渡して助言を得る。記録は下記「相談ログ」。

## 進捗
- [x] ブランチ作成
- [x] advisor 初回相談（Fable サブエージェント経由）
- [x] 現行定義の特定（仕様書・Python・Rust）
- [x] タスク A 現状の定量化
- [x] タスク B 設計
- [x] タスク C 計測計画
- [x] タスク D 推奨と未決事項
- [x] ハーネス・パリティ検証
- [x] レポート作成・inbox 複製

## 相談ログ
### 相談1（着手直後、Fable サブエージェント経由。助言は model 出力であり正本ではない）
聞いたこと: (1) 候補案を出す軸 (2) 退化ケース契約の罠 (3) S_haz 踏襲範囲と Awase/Nagare の設計分離。
助言の根拠の範囲: `python/usir_py/v5/six_pole_atlas.py` の :112-175, :301-327, :512-534 のみ。repo 内で `S_haz`/`ii-a`/`ii-g` の Grep は 0 件（B-22/B-23 正本は助言側では未確認）。
採用（コードで確認すべき点として扱う。検証前は仮説）:
- 候補案の4軸: クリップ位置（場面内 ReLU か符号付き生値保持か）／閾値の扱い／基準（絶対・作品内分位・コーパス）／集約単位。
- 退化契約: 欠測既定値（irony 0.1、detachment 0.12、phi 0.45/0.5、speed_mean 0.5、speed_ratio 1.0）を実測値と区別し、有効フラグまたは NaN で返す。場面数による定義分岐（irony_grad 5場面、speed_ratio）を連続指標に持ち込まない。ε 依存域は未定義にする。「入力ゼロ」と「信号ゼロ」を区別する。
- Awase と Nagare は次元別に仕様を書く。共通化は契約の書式のみ。
採らない/保留:
- (ii-a)/(ii-g) の対応づけ: 助言側の推測を含む。B-22 正本の所在確認が済むまで採用しない。
- 「Awase の低再現性は張り付きの帰結か」は未検証。設計書に未検証として記載する。


## 完了状態（2026-11-17）
- 設計書: docs/handoff/2026-11-17_usir_v5_faw_design.md
- レポート: docs/handoff/2026-11-17_usir_v5_stage_faw_report.md（inbox へ複製）
- 計測: scripts/faw/（faw_measure.py, faw_pilot.py, 結果 JSON, logs/）
- ゲート: harness exit 0（PASSED 672）、パリティ 不一致 0（F17/C216）
- advisor: 付帯終了のため追加相談なし（Sonnet 単独で完了）
- 未検証・判断待ちはレポート §7・§8
