# STATUS hybrid full run（強いモデル全項目注釈 × 全レンズ）
- ブランチ: phase2/hybrid-full-run（phase2/sensor-ceiling-strong-model 3cd4fde から分岐）、worktree C:\dev\usir-hybrid
- nishiki 側: phase2/hybrid-full-run（phase2/nagare-awase-fill a19bfe4 から分岐 + feat/sasae-location-v2 を取り込み）、worktree C:\dev\nishiki-hybrid。既存ブランチへのマージなし
- updated: 2026-11-20
- state: 開始（WIP）
- done: worktree 作成。nishiki 側は location-v2 取り込み時に diagnose_slice.rs / expected_v3.json が競合、解消中
- next: (1) scene_meta.nagare/awase を本文のみから注釈しコミット（event_top3_gold・stage_b_gold は未開封）(2) 採点 (3) nishiki diagnose (4) 比較表・レポート
- 方針: 事実・特性・同席・場所は phase2/sensor-ceiling-strong-model の注釈（stage_b_gold を開く前にコミット済み: dccc026）を再利用。出来事 top3・Awase を新規に注釈する
