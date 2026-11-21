# `sensor.v2` Phase 0 最小契約ノート

- 作成日: 2026-11-19
- 下書きスキーマ: `schemas/sensor_v2.schema.json`（Draft 2020-12）
- 対象フェーズ: Phase 0（Sasae + Sadame 縦貫通の先行契約）

---

## 1. 位置づけとライフサイクル

1. `schemas/sensor_v2.schema.json` は **Step 1〜Step 3 の検証期間中における USIR 側の下書き正本**である。
2. Step 3 通過＝契約凍結の時点で、最終の JSON Schema をもとに Nishiki の `crates/usir-schema` へ型定義を正式移行し、USIR 側の下書きは廃止する。
3. **2 層＋決定論射影方式**:
   - 原稿側（`.nishiki/annotations/<章ID>/<場面ID>.json`）の保存形式は本スキーマ（`sensor.v2`）とする。
   - Nishiki の `usir-metrics`（Sadame / Nagare / Awase / Kamae B）は全構造体 `#[serde(deny_unknown_fields)]` の `usir_schema::SceneAnnotation` を入力とするため、判定側では決定論的な射影関数で 4 極部分だけを `SceneAnnotation` 互換データへ変換して流す。
   - Sasae の判定と `scene_hash` 鮮度照合は `sensor.v2` を直接参照する。

---

## 2. 確定した裁定 3 件（2026-11-19 ユーザー承認済み）

1. **価値観 6 軸の物差し（`values_6d` / `axes_q`）**:
   - `character_states[].values_6d` は **`{-1, 0, +1}` の 3 段階**（6 要素整数配列）に統一する。
   - これにより、既存の `usir-metrics/src/sadame.rs`（`INTER_NORM = 24.0`、`intra_variance ∈ [0, 1]`）を無改造・無再正規化で駆動する（粒度の細分化が必要と実証された場合のみ Step 3 以降に再検討）。
2. **発話から `DialogueEdge` への集約と重み**:
   - 同一場面内の発話から `SadameAnn.dialogue_edges` へ射影する際、**同一の有向ペア `(speaker_id, addressee_id)`（`u != v`、両方非 `null`）を 1 本に集約**する。
   - `evidence == "ambiguous"` の発話は辺の生成から除外する。
   - ペア内に複数発話がある場合、`DialogueEdge.confidence: u8` は**ペア内の最大重み**を採用する。
   - `evidence` → 重みの初期値は **`explicit = 100` / `inferred = 50` / `ambiguous = 0`** とし、Step 1 の実測誤り率に基づいて較正する。
   - 同様に `character_states[]` で `evidence == "ambiguous"` の要素は `CharacterStalk` への射影から除外（観測欠損扱い）する。
3. **`diagnose` への追加入力と出力版管理**:
   - `DiagnoseOptions`（CLI / Wasm / Core 共通）への追加入力は、既存の `artifact_json` に倣い JSON 文字列の任意オプション（`characters_json`, `annotations_json`）とする。
   - 新オプション未指定時の既定出力 `nishiki.diagnose/2` は 1 ビットも変更しない。
   - 新入力を使う場合の出力（`sadame` の `ok` 化、`sasae` の名簿・注釈連動拡張）は次版（`nishiki.diagnose/3` 候補）として開発し、Phase 2 通過後に凍結を判断する。

---

## 3. Phase 0 最小契約（必須 vs 将来任意枠）

| 区分 | スキーマ領域 | 主な中身 |
|---|---|---|
| **必須（Sasae + Sadame + 再現性）** | `contract` | `schema_version` (`"sensor.v2"`), `splitter_version`, `scene_id` (`ch03:s02` 等), `scene_hash` (64桁 SHA-256 hex), `ledger_hash`, `model_id`, `prompt_hash`。`author_id` / `work_id` / `scene_index` / `scene_count` / `char_len` / `seed` / `stale_context` / `override_applied` は任意（機械補完・運用フラグ）。 |
| **必須（Sasae 場面メタ）** | `scene_meta.{time, location_id, present_character_ids, pov_character}` | 時間の相対関係（`continuous` / `flashback` / `parallel` / `unknown`）・経過手がかり、場所 ID、同席者配列、視点人物 ID（`null` 許容、不明は不明のまま記録する開世界仮説）。 |
| **必須（Sasae / Sadame 発話）** | `utterances[]` | `utterance_index` (`[U0]` 連番), `speaker_id`, `addressee_id`, `evidence` (`explicit`/`inferred`/`ambiguous`), `register` (`politeness`, `first_person`)。 |
| **必須（Sadame 価値観）** | `character_states[]` | 能動人物のみ記録。`character_id`, `values_6d` (`[i8; 6]`, 各 `-1..=1`), `evidence`。 |
| **必須（Sasae 事実更新）** | `fact_updates[]` | 7 述語 enum（`LEARNED` / `ACQUIRED` / `LOST` / `MOVED_TO` / `STATE_CHANGED` / `HOOK_OPENED` / `HOOK_RESOLVED`）＋ `subject`, `object`, `evidence`, 任意 `sentence_index` (`[S0]`)。 |
| **将来枠（任意・`null` 許容）** | `scene_meta.{nagare, awase, kamae}`, `provisional_entities[]` | `nagare`（`NagareAnn` 同型 `event_top3`/`mood_top3`）、`awase`（`AwaseAnn` 同型 4 評価）、`kamae`（文字列 ID 伏線要約、`HOOK_OPENED`/`HOOK_RESOLVED` との併用可）、`provisional_entities`（台帳未登録エンティティの仮 ID）。 |

---

## 4. 不変条件と運用ルール

- **分割・連番・ハッシュの正本は Nishiki**: センサー実行系は場面分割やオフセット計算を再実装しない。Nishiki が正規化後テキストから `usir_v5_splitter@1` で分割し、`scene_hash` と `[S0]` / `[U0]` 連番を供給する。注釈ファイルに原文は保存しない。
- **鮮度照合と上書き層**: `diagnose` 実行時に実本文の `scene_hash` と `contract.scene_hash` を突き合わせ、不一致場面は注釈適用をスキップして警告を出す。人間の確定修正（`.nishiki/overrides/`）は自動生成注釈（`.nishiki/annotations/`）に常に優先する。
- **台帳連携（Plotailor `characters.yaml` → `CharacterEntry[]`）**:
  - `status == "confirmed"` のみ既定採用（`rejected` 除外、`candidate` は既定除外）。
  - `aliases` には必ず `[name, ...aliases]` を結合する。一般代名詞・一般役職名詞単独・複数人物間で重複する表層は除外＋警告する。
  - `characters.yaml` に無い `gender` / `honored` は既定値（`Unknown` / `false`）で補完する。
- **伏線 ID の変換規約（Phase 4 向け）**: Plotailor `hooks.yaml` および `sensor.v2` は文字列 ID を用い、`usir_schema::KamaeAnn` への射影時に作品内で決定論的な `String -> u32` 連番マップへ変換する。

---

## 5. JSON Schema 外で実行時バリデータが強制する制約

JSON Schema（Draft 2020-12）の構造検査に加え、以下はランナーおよび Nishiki 側の実行時検証で強制する。

1. `scene_meta.nagare` が非 `null` の場合、`event_top3` 内および `mood_top3` 内で `code` が重複しないこと（`usir_schema::validate` 準拠）。
2. `utterances[].utterance_index` が場面内で重複しないこと。
3. `contract.scene_index` と `contract.scene_count` が両方指定された場合、`scene_index < scene_count` を満たすこと。
4. `contract.scene_hash` が Nishiki 分割器が算出した正規化場面本文の SHA-256 と一致すること（不一致時は陳腐化扱い）。
