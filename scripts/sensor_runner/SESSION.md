# SESSION.md — Claude Code セッションによる強いモデル注釈のバッチ処理手順

対象: `scripts/sensor_runner/` の prepare が出力した「パック」を、Claude Code セッションが読んで注釈 JSON（`sensor.strong/1`）を書き、ingest で検証・取り込む。API キーは使わない。

## 分担（固定）
| 項目 | 担当 |
|---|---|
| 発話の話者・宛先・根拠（evidence） | ローカルモデル＋決定論（`local/<work>.utterances.json`、場面ハッシュ→発話配列） |
| 口調（politeness）・一人称 | Policy X（Janome 表層規則）・表層正規表現。merge が付与 |
| 同席の補完・時間・場所・視点・事実・特性の変動・出来事／風情コード・Awase | **強いモデル（このセッション）** |
| 特性の基準値（baseline） | 台帳 `cache/ledger/<work>.traits.json`。場面側は変動（axis, delta）のみ |

## 用語
- パック: `scripts/sensor_runner/packs/<work>/<hash12>.pack.json`。場面本文、`[S#]/[U#]` 付き本文、関係する名簿、台帳の基準値、ローカル発話注釈、直前場面の状態サマリ、注釈指示、出力テンプレート、出力先 `output_path` を含む。
- キャッシュ: `scripts/sensor_runner/cache/strong/<scene_hash>.json`。キーは場面ハッシュ（nishiki segment の `scene_hash`）。本文が変わった場面はハッシュが変わるため、その場面だけ未キャッシュになる。

## 前提
```bash
cd C:/dev/usir-sensor-runner/python        # uv 環境
export PYTHONIOENCODING=utf-8
export NISHIKI_BIN=C:/dev/nishiki-hybrid/target/debug/nishiki.exe   # segment / diagnose を持つビルド
```
以降 `R="uv run python ../scripts/sensor_runner/cli.py"`。

## 1回の処理単位
**1作品につき最大 5 場面（`--limit 5`）を1バッチ**とする。理由: 直前場面の状態サマリ（prior）を次のパックに反映させるには、前の場面を ingest してからパックを再生成する必要があるため、バッチを短く保つ。場面は index 昇順で処理する。

## 手順（1バッチ）
1. 状況確認: `$R status`（作品ごとの cached 数）。
2. パック生成: `$R prepare --work <WORK_ID> --limit 5`。出力される pack パスが処理対象。未キャッシュ場面が 0 なら完了。
3. 場面 index 昇順に、1パックずつ:
   1. パックを Read する（`scene_text`, `roster`, `ledger_baseline`, `local_utterances`, `prior`, `instructions`, `output_template`）。ログや本文を会話に貼り直さない。
   2. `instructions` と `output_template` に従い、`sensor.strong/1` の JSON を `pack.output_path` に Write する（`scene_hash` はパックの値をそのまま、`annotator` は `claude-code-session:<モデルID>`、`provenance` は `session`）。
   3. 直ちに ingest: `$R ingest <output_path>`。
      - `ACCEPT`: キャッシュと manifest に取り込まれる。
      - `REJECT`: 理由が `rejects/<name>.reject.json` と標準出力に出る。**理由に従って出力ファイルを書き直し**、再度 ingest する（ingest は修復しない。同じ理由で2回失敗したら方針を変えて本文を読み直す）。
   4. 次の場面のパックを再生成する: `$R prepare --work <WORK_ID> --limit 5`（直前場面が cached になり、prior が `cached` の状態サマリになる。残りのパックは再生成されるので、最新のものを読むこと）。
4. バッチ終了後: `$R status` で cached 数を確認。作品が全場面 cached になったら merge へ。

## 書き方の要点（ingest が機械的に検証する規則）
- すべての `basis` は本文に実在する連続文字列の逐語コピー（空白差のみ可）。要約・言い換え・連結は不可。見つからない項目は出さない（facts を空配列にする等）。`present` の basis は必須。
- `event_top3` / `mood_top3` は重複なし3件・コードは E01–E16 / M01–M16。`awase` は各 -2〜2 の整数＋ F_basis / f_basis。
- `trait_changes`: 能動的な人物ごとに1件。`changes` は台帳の baseline に対する変動のみ（delta は -2〜2、0 不可、axis 0〜5、同じ axis を重複させない）。適用後は [-1,1] に丸められる。変動が無ければ `changes: []`。台帳に無い人物は `ledger_additions` に baseline を添える（既存人物の上書きは不可）。
- `facts`: predicate は7種。`STATE_CHANGED` のみ `target` / `category`（`body_health, emotion, relation_status, object_state, identity_form, situation, other`）/ `text` の半構造化。`unclassified` は seed 専用で session では不可。
- `present`: ローカル発話の話者・宛先が居合わせるなら含める。merge は自動追加しない（`local_ids_absent_from_present` として件数が出るので、0 でなければ本文を見直す。本文上その人物が不在の発話者（手紙の文面など）は、含めないのが正しい場合がある）。
- 台詞の話者などローカル側の項目は書かない（書いても merge は使わない）。

## 検証の通し方
1. 取り込み: 各場面で `ingest` が `ACCEPT`（strong 注釈の規則と、sensor.v2 スキーマへの dry merge の両方を通る）。
2. 作品が全場面 cached になったら:
   ```bash
   $R merge --work <WORK_ID> --diagnose
   ```
   - `merged=N/N errors=0` であること（sensor.v2 スキーマ・名簿検証）。
   - `missing_local_utt=0`（ローカル発話注釈の欠落が無い）。
   - `lenses=` に `kasane, sasae, kamae, sadame, nagare, awase` が全て `ok`。
   - 出力: `scripts/sensor_runner/merged/<work>.annotations.json`（場面ごとの sensor.v2 記録）。
3. テスト: `uv run pytest tests/test_sensor_runner.py -q`。

## 差分の再実行
- 本文が変わった作品: `prepare` を再実行すると、ハッシュが変わった場面（と新規場面）のパックだけが出る。変わっていない場面は cached のまま。
- 注釈指示（`strong.py` の `INSTRUCTIONS` / スキーマ）を変えて全場面を取り直す場合のみ `common.INSTRUCTION_VERSION` を上げる（旧エントリは未キャッシュ扱いになる）。
- 特定場面だけ取り直す: `cache/strong/<scene_hash>.json` を削除して `prepare`。

## 限界（既知）
- prior（直前場面の状態）は、直前場面が cached のときだけ入る。バッチ内で順に ingest しない場合は `pending` の注記になる。
- キャッシュキーは場面ハッシュのみ。名簿・台帳を変えても自動では無効化されない（変えたら該当場面のエントリを削除する）。
- seed エントリ（`provenance: seed:hybrid-full-run@6043212`）は basis が null の項目を含み、STATE_CHANGED は `category: unclassified`。session で取り直した場面は basis 必須の規則を通っている。
- 注釈の妥当性（事実・特性・風情の正しさ）は、この仕組みでは検証しない。検証するのは書式・根拠の逐語性・名簿・スキーマ整合のみ。
