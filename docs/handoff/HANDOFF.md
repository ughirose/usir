# USIR v5 実装引継ぎ台帳 (HANDOFF)

## 1. タスク原文
> あなたは `C:\dev\usir` の実装担当サブエージェントです。以下のステージ B-1 および B-2 の全実装・生成・テスト・同期・コミットを1パスで完遂し、完了結果を簡潔に報告してください。
> - 対象コーパス: `C:\dev\usir\data\` の3つのマニフェスト（計 10 作家 106 作品、3,584,940 字）。
> - キャッシュテキスト: `novel_{author_id}_{work_id}.txt`（46件）および `heldout_{author_id}_{work_id}.txt`（60件）全 106 作品。
> - スキーマ仕様: `usir-schema`（`SceneAnnotation` v5.0、`deny_unknown_fields` 厳密準拠）。
> - タスクA: `python/usir_py/v5/__init__.py`, `aozora_scene_splitter.py` の実装。
> - タスクB: `python/tests/v5/test_scene_splitter.py` の作成と検証（pytest 100% PASS）。
> - タスクC: ステージ B-2 パイロット 50 場面生成・コピー・HANDOFF 更新・Git コミット。

## 2. 現在の意図
USIR v5 / Nishiki 統合における青空文庫 106 作品（3,584,940 字）の決定論的場面分割（200〜1600 字クランプ）および初期ヒューリスティック注釈（awase / nagare / sadame / kamae）パイプラインを確立し、Rust クレート `usir-schema` との 100% バイナリ/スキーマ整合性を実証するとともに、人間・Gemini 監査用のパイロット 50 場面（`pilot_50_scenes.jsonl`）を同期してステージ B-3（本格アノテーション）への移行条件を完備すること。

## 3. 計画と進捗
- [x] **ステージ B-1-1**: プロトタイプ検証と境界条件アルゴリズムの確定（空行、見出し、約物、時空間マーカー、文分割、端数再配分）。
- [x] **ステージ B-1-2**: `python/usir_py/v5/__init__.py` および `python/usir_py/v5/aozora_scene_splitter.py` 実装。
- [x] **ステージ B-1-3**: Rust `usir-schema` との厳密結合テスト（`aozora_106works_scenes_parses_and_validates`）の追加と実行検証。
- [x] **ステージ B-1-4**: `python/tests/v5/test_scene_splitter.py`（6項目）作成と pytest 全件 PASS。
- [x] **ステージ B-2-1**: 全 4,576 場面データセット（`data/v5/aozora_106works_scenes.jsonl`）およびパイロット 50 場面（`data/v5/pilot_50_scenes.jsonl`）の生成。
- [x] **ステージ B-2-2**: パイロット 50 場面の `F:\マイドライブ\Claude-Sync\inbox\pilot_50_scenes.jsonl` への物理同期。
- [x] **ステージ B-2-3**: 引継ぎ台帳（`docs/handoff/HANDOFF.md`）作成および `F:\マイドライブ\Claude-Sync\inbox\HANDOFF_usir.md` 同期。
- [x] **ステージ B-2-4**: Git コミットの作成。

## 4. 完了条件と検証結果
1. **全 106 作品・10 作家の全数保持**:
   - マニフェスト（46 + 30 + 30）全 106 作品を欠損なく処理。
   - 総場面数: **4,576 場面**（総文字数 3,576,966 字）。
2. **場面文字数の厳密クランプ**:
   - 最小文字数: **200 字**
   - 最大文字数: **1,594 字**（<= 1,600 字）
   - 中央値: **784.0 字**（平均 781.7 字）
   - 範囲外（<200 または >1600）: **0 件**。
3. **Rust `usir-schema` 厳密パース・意味検証**:
   - `cargo test --manifest-path C:\dev\nishiki\crates\usir-schema\Cargo.toml --test schema`: **8 passed, 0 failed in 0.06s**。
   - `deny_unknown_fields`、`awase`（-2..=2）、`nagare`（Top-3 重複なし、rank_weight 1..=3）、`group_by_work`（全 106 作品で scene_index が 0 から scene_count-1 まで昇順連続・欠番重複ゼロ）を完全通過。
4. **Python pytest テストスイート**:
   - `uv run pytest tests/v5/ -v`: **6 passed in 1.57s**。
5. **パイロット 50 場面の整合性**:
   - 10 作家 × 5 場面 ＝ **50 場面**。
   - 序盤（0.05）、前半（0.25）、中盤（0.50）、後半（0.75）、終盤（0.95）を均等網羅。
   - 人間・Gemini 監査用メタデータ（`scene_id`, `author_ja`, `title`, `text`, `story_position`, `f_event_stream`, `f_style_stream`）完備。

## 5. 成果物・変更ファイルポインタ
- [`python/usir_py/v5/__init__.py`](file:///C:/dev/usir/python/usir_py/v5/__init__.py): v5 パッケージエントリポイント
- [`python/usir_py/v5/aozora_scene_splitter.py`](file:///C:/dev/usir/python/usir_py/v5/aozora_scene_splitter.py): 場面分割・ヒューリスティック注釈エンジン・コーパススプリッター
- [`python/tests/v5/test_scene_splitter.py`](file:///C:/dev/usir/python/tests/v5/test_scene_splitter.py): 分割境界・スキーマ準拠・境界値テストスイート
- [`crates/usir-schema/tests/schema.rs`](file:///C:/dev/nishiki/crates/usir-schema/tests/schema.rs): Rust 側での実コーパス（4,576 場面）直接検証テスト
- [`data/v5/aozora_106works_scenes.jsonl`](file:///C:/dev/usir/data/v5/aozora_106works_scenes.jsonl): 全 106 作品・4,576 場面の厳密スキーマ注釈 JSONL（3.05 MB）
- [`data/v5/pilot_50_scenes.jsonl`](file:///C:/dev/usir/data/v5/pilot_50_scenes.jsonl): パイロット監査用 50 場面 JSONL（148 KB）
- `F:\マイドライブ\Claude-Sync\inbox\pilot_50_scenes.jsonl`: ドライブ側監査用コピー
- `F:\マイドライブ\Claude-Sync\inbox\HANDOFF_usir.md`: ドライブ側引継ぎ台帳コピー

## 6. 確定事項と除外・却下した方針
- **`aozora_106works_scenes.jsonl` への `text` フィールド混入の完全排除**:
  - `usir-schema` の `SceneAnnotation` は `#[serde(deny_unknown_fields)]` であり、Rust の下流パイプラインで直接消費される JSONL に本文 `text` や `title` を含めると即座にパースエラーとなる。そのため本番 JSONL はスキーマ厳密準拠とし、監査用本文付きレコードはパイロット JSONL（`pilot_50_scenes.jsonl`）に局所化させた。
- **末尾端数場面のハードカット排除と再配分（Rebalance）アルゴリズム採用**:
  - 末尾の端数が 200 字未満になった場合、単に直前場面に結合すると 1,600 字の上限を突破するケースが存在した。そのため「結合して <= 1600 字なら結合、超える場合は 2 場面を結合して中央付近の文末記号（`。！？`）で均等 2 分割」する自己補正ロジックを確定実装した。これにより 4,576 場面すべてで 200〜1600 字の厳密遵守が達成された。
- **ヒューリスティック Top-3 の決定論的重複排除**:
  - `nagare` の `event_top3` および `mood_top3` において同点スコアが発生しても、コード名によるタイブレークを行い、必ず 3 つの異なるコードが選出されるよう保証した。

## 7. 未解決・未検証・限界（自己批判・死角の開示）
1. **誤爆 (False Positive)**:
   - 会話文が連続する場面において、`「`〜`」` の割合が高いために自動的に `E03`（対話）や高い覚醒度が割り当てられるが、深刻な告白や静かな別れの対話であっても表層的な疑問符・感嘆符の欠如や語彙の偏りにより、意図と異なるムードコードが Top-1 に選ばれるケースが存在する（これはあくまでヒューリスティック・初期ドラフト値の限界であり、ステージ B-3 の LLM/Gemini 高度アノテーションで上書きされる前提）。
2. **取りこぼし (False Negative)**:
   - 時空間マーカー（`RE_TIME_MARKER`）は主要な表現（「翌朝」「その夜」「数年後」等）をカバーしているが、古風な擬古文体（森鴎外・泉鏡花）における「去ぬる明治何年」「折しも」「ほどなく」等のマイナーな文語的転換表現の一部は weak boundary として拾われず、通常の文字数閾値（750 字）で機械的に切られている箇所がある。
3. **物理的事実と設計意図の乖離**:
   - 場面平均文字数は 781.7 字、中央値 784.0 字と目標値（750 字）に対して約 4% 上振れしている。これは青空文庫の段落が比較的長く、文境界を跨いで切ることを避けた結果として少し長めに蓄積されてから切られる傾向があるためである。ただし上限 1,600 字に対して最大 1,594 字であり、物理的制約は完全に満たされている。

## 8. 次の1手 (NEXT)
- ステージ B-3: 人間・Gemini による `pilot_50_scenes.jsonl` の目視監査・プロンプト精度検証と、106 作品全 4,576 場面に対する本番 LLM/Gemini 注釈実行パイプラインの構築。
