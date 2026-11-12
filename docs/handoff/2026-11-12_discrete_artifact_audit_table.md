# 全極・全空間 離散カウント固定化（ゼロ張り付き・2値化）総点検レポート

点検対象スケール:
- ① D10 多訳データ (16件, 400〜800字)
- ② Class A ミクロ推敲スパン (84件, 30〜150字)
- ③ 青空文庫 106作品 場面 (4,576件)
- ④ 青空文庫 106作品 スライディング窓 (11,723件)

## 1. 高リスク特徴量（短文・小窓でのゼロ張り付き率 > 80% またはユニーク値 <= 3）

| 特徴量名 | カテゴリ | D10 ゼロ率 (ユニーク数) | Class A ゼロ率 (ユニーク数) | 4576場面 ゼロ率 | 11723窓 ゼロ率 | リスク要因・機構解釈 |
|---|---|---|---|---|---|---|
| `6pole_kamae` | 6pole | 50.0% (2) | 91.7% (2) | 43.0% (2) | 49.8% (2) | フック生存率（開閉フックの離散比） |
| `kamae_hook_survival` | Deep15 | 50.0% (2) | 91.7% (2) | 43.0% (2) | 49.8% (2) | フック生存率（短文で開フック0または1に固定） |
| `kamae_opened_hooks` | Deep15 | 50.0% (2) | 91.7% (2) | 43.0% (3) | 49.8% (3) | 開フック数（短文で0〜1カウント） |
| `awase_detachment_mean` | Deep15 | 62.5% (3) | 97.6% (2) | 83.0% (6) | 83.8% (7) | 視点乖離度（短文で特定助詞欠如時0） |
| `sadame_e_inter` | Deep15 | 75.0% (4) | 95.2% (2) | 27.7% (176) | 34.1% (168) | 場面間エッジ（単一場面・短文では定義上0固定） |
| `ortho28_v2_final_ぬ` | ortho28 | 100.0% (1) | 95.2% (3) | 87.9% (107) | 89.5% (96) | 文末助動詞「ぬ」（短文でほぼ出現せず0張り付き） |
| `ortho28_v2_final_う` | ortho28 | 75.0% (3) | 96.4% (2) | 46.9% (167) | 52.5% (141) | 文末助動詞「う」（短文でほぼ出現せず0張り付き） |
| `ortho28_v2_final_す` | ortho28 | 75.0% (2) | 98.8% (2) | 62.5% (230) | 65.7% (201) | 文末助動詞「す」（短文でほぼ出現せず0張り付き） |
| `ortho28_v2_final_か` | ortho28 | 100.0% (1) | 97.6% (2) | 67.4% (134) | 71.3% (134) | 文末助動詞「か」（短文でほぼ出現せず0張り付き） |
| `ortho28_v2_quote_ratio` | ortho28 | 75.0% (4) | 95.2% (5) | 22.6% (2543) | 27.0% (1849) | 会話文比率（会話文のない地文短文で0固定） |
| `hw25_gamma_uid` | HW-25 | 0.0% (16) | 51.2% (38) | 1.0% (4434) | 0.8% (8915) | UID比率（1文・短文でステップ幅粗大） |
| `6pole_kasane` | 6pole | 0.0% (12) | 13.1% (49) | 0.1% (3752) | 0.2% (63) | 読点密度（連続値、低リスク） |
| `6pole_sasae` | 6pole | 0.0% (12) | 0.0% (54) | 0.1% (3318) | 0.0% (57) | 句点密度（短文でも文末に必ず1個あり） |
| `6pole_nagare` | 6pole | 0.0% (1) | 0.0% (1) | 0.0% (1) | 0.0% (1) | 世阿弥エントロピー（連続値） |
| `6pole_awase` | 6pole | 93.8% (2) | 100.0% (1) | 89.8% (4) | 91.5% (4) | アイロニー平均（感情極性反転） |

## 2. 全 50 特徴量の完全点検マトリクス

| 特徴量名 | D10 ゼロ率 / ユニーク数 | Class A ゼロ率 / ユニーク数 | 4576場面 ゼロ率 / ユニーク数 | 11723窓 ゼロ率 / ユニーク数 | リスク判定 |
|---|---|---|---|---|---|
| `6pole_awase` | 93.8% / 2 | 100.0% / 1 | 89.8% / 4 | 91.5% / 4 | HIGH_ARTIFACT_RISK |
| `6pole_kamae` | 50.0% / 2 | 91.7% / 2 | 43.0% / 2 | 49.8% / 2 | HIGH_ARTIFACT_RISK |
| `6pole_kasane` | 0.0% / 12 | 13.1% / 49 | 0.1% / 3752 | 0.2% / 63 | LOW_CONTINUOUS |
| `6pole_nagare` | 0.0% / 1 | 0.0% / 1 | 0.0% / 1 | 0.0% / 1 | HIGH_ARTIFACT_RISK |
| `6pole_sadame` | 75.0% / 4 | 95.2% / 2 | 27.7% / 194 | 34.1% / 186 | HIGH_ARTIFACT_RISK |
| `6pole_sasae` | 0.0% / 12 | 0.0% / 54 | 0.1% / 3318 | 0.0% / 57 | LOW_CONTINUOUS |
| `awase_detachment_max` | 62.5% / 3 | 97.6% / 2 | 83.0% / 6 | 83.8% / 7 | HIGH_ARTIFACT_RISK |
| `awase_detachment_mean` | 62.5% / 3 | 97.6% / 2 | 83.0% / 6 | 83.8% / 7 | HIGH_ARTIFACT_RISK |
| `awase_irony_max` | 93.8% / 2 | 100.0% / 1 | 89.8% / 4 | 91.5% / 4 | HIGH_ARTIFACT_RISK |
| `awase_irony_mean` | 93.8% / 2 | 100.0% / 1 | 89.8% / 4 | 91.5% / 4 | HIGH_ARTIFACT_RISK |
| `hw25_gamma_uid` | 0.0% / 16 | 51.2% / 38 | 1.0% / 4434 | 0.8% / 8915 | MEDIUM_DISCRETE_STEP |
| `kamae_hook_survival` | 50.0% / 2 | 91.7% / 2 | 43.0% / 2 | 49.8% / 2 | HIGH_ARTIFACT_RISK |
| `kamae_hook_survival_idf` | 50.0% / 2 | 91.7% / 2 | 43.0% / 2 | 49.8% / 2 | HIGH_ARTIFACT_RISK |
| `kamae_opened_hooks` | 50.0% / 2 | 91.7% / 2 | 43.0% / 3 | 49.8% / 3 | HIGH_ARTIFACT_RISK |
| `nagare_event_accel` | 100.0% / 1 | 100.0% / 1 | 100.0% / 1 | 100.0% / 1 | HIGH_ARTIFACT_RISK |
| `nagare_event_speed` | 100.0% / 1 | 100.0% / 1 | 100.0% / 1 | 100.0% / 1 | HIGH_ARTIFACT_RISK |
| `nagare_mood_accel` | 100.0% / 1 | 100.0% / 1 | 100.0% / 1 | 100.0% / 1 | HIGH_ARTIFACT_RISK |
| `nagare_mood_speed` | 100.0% / 1 | 100.0% / 1 | 100.0% / 1 | 100.0% / 1 | HIGH_ARTIFACT_RISK |
| `nagare_phi_zeami` | 0.0% / 1 | 0.0% / 1 | 0.0% / 1 | 0.0% / 1 | HIGH_ARTIFACT_RISK |
| `ortho28_v2_dialogue_char_ratio` | 75.0% / 4 | 95.2% / 4 | 25.4% / 3303 | 31.8% / 6123 | HIGH_ARTIFACT_RISK |
| `ortho28_v2_final_other` | 75.0% / 2 | 92.9% / 3 | 21.7% / 375 | 3.7% / 392 | HIGH_ARTIFACT_RISK |
| `ortho28_v2_final_」` | 100.0% / 1 | 100.0% / 1 | 85.2% / 46 | 56.2% / 337 | HIGH_ARTIFACT_RISK |
| `ortho28_v2_final_い` | 75.0% / 2 | 84.5% / 4 | 37.1% / 212 | 39.4% / 185 | HIGH_ARTIFACT_RISK |
| `ortho28_v2_final_う` | 75.0% / 3 | 96.4% / 2 | 46.9% / 167 | 52.5% / 141 | HIGH_ARTIFACT_RISK |
| `ortho28_v2_final_か` | 100.0% / 1 | 97.6% / 2 | 67.4% / 134 | 71.3% / 134 | HIGH_ARTIFACT_RISK |
| `ortho28_v2_final_す` | 75.0% / 2 | 98.8% / 2 | 62.5% / 230 | 65.7% / 201 | HIGH_ARTIFACT_RISK |
| `ortho28_v2_final_た` | 50.0% / 2 | 39.3% / 3 | 7.8% / 443 | 8.7% / 374 | HIGH_ARTIFACT_RISK |
| `ortho28_v2_final_だ` | 100.0% / 1 | 90.5% / 3 | 56.8% / 186 | 61.3% / 172 | HIGH_ARTIFACT_RISK |
| `ortho28_v2_final_ぬ` | 100.0% / 1 | 95.2% / 3 | 87.9% / 107 | 89.5% / 96 | HIGH_ARTIFACT_RISK |
| `ortho28_v2_final_る` | 50.0% / 5 | 69.0% / 4 | 31.7% / 341 | 33.5% / 299 | MEDIUM_DISCRETE_STEP |
| `ortho28_v2_hiragana_ratio` | 0.0% / 16 | 0.0% / 72 | 0.6% / 4195 | 0.6% / 5992 | LOW_CONTINUOUS |
| `ortho28_v2_kana_が` | 25.0% / 13 | 45.2% / 39 | 1.1% / 3000 | 0.8% / 1645 | MEDIUM_DISCRETE_STEP |
| `ortho28_v2_kana_た` | 0.0% / 12 | 14.3% / 49 | 0.9% / 3378 | 0.7% / 2200 | LOW_CONTINUOUS |
| `ortho28_v2_kana_て` | 6.2% / 16 | 26.2% / 49 | 0.8% / 3172 | 0.7% / 1905 | LOW_CONTINUOUS |
| `ortho28_v2_kana_で` | 12.5% / 15 | 35.7% / 44 | 1.3% / 3168 | 1.0% / 1823 | MEDIUM_DISCRETE_STEP |
| `ortho28_v2_kana_と` | 0.0% / 12 | 39.3% / 39 | 0.8% / 3112 | 0.7% / 1831 | MEDIUM_DISCRETE_STEP |
| `ortho28_v2_kana_に` | 0.0% / 12 | 29.8% / 48 | 0.8% / 3123 | 0.7% / 1800 | LOW_CONTINUOUS |
| `ortho28_v2_kana_の` | 0.0% / 16 | 11.9% / 58 | 0.8% / 3494 | 0.7% / 2479 | LOW_CONTINUOUS |
| `ortho28_v2_kana_は` | 0.0% / 12 | 21.4% / 49 | 0.8% / 3012 | 0.7% / 1709 | LOW_CONTINUOUS |
| `ortho28_v2_kana_も` | 18.8% / 14 | 57.1% / 34 | 1.5% / 2939 | 0.9% / 1582 | MEDIUM_DISCRETE_STEP |
| `ortho28_v2_kana_を` | 0.0% / 16 | 27.4% / 43 | 0.9% / 3094 | 0.8% / 1766 | LOW_CONTINUOUS |
| `ortho28_v2_kanji_ratio` | 0.0% / 16 | 0.0% / 72 | 0.0% / 4177 | 0.0% / 5921 | LOW_CONTINUOUS |
| `ortho28_v2_log_mean_sent_len` | 0.0% / 12 | 0.0% / 58 | 0.0% / 3401 | 0.0% / 2193 | LOW_CONTINUOUS |
| `ortho28_v2_narr_log_mean_sent_len` | 0.0% / 16 | 2.4% / 50 | 0.2% / 2973 | 0.3% / 5022 | LOW_CONTINUOUS |
| `ortho28_v2_narr_log_p90_sent_len` | 0.0% / 13 | 2.4% / 50 | 0.2% / 1216 | 0.8% / 1572 | LOW_CONTINUOUS |
| `ortho28_v2_quote_ratio` | 75.0% / 4 | 95.2% / 5 | 22.6% / 2543 | 27.0% / 1849 | HIGH_ARTIFACT_RISK |
| `ortho28_v2_sent_len_cv` | 0.0% / 16 | 50.0% / 40 | 0.1% / 4550 | 0.0% / 11590 | MEDIUM_DISCRETE_STEP |
| `sadame_e_inter` | 75.0% / 4 | 95.2% / 2 | 27.7% / 176 | 34.1% / 168 | HIGH_ARTIFACT_RISK |
| `sadame_e_intra` | 100.0% / 1 | 100.0% / 1 | 100.0% / 1 | 100.0% / 1 | HIGH_ARTIFACT_RISK |
| `sadame_edges_eff` | 75.0% / 2 | 95.2% / 2 | 27.7% / 7 | 34.1% / 7 | HIGH_ARTIFACT_RISK |
| `sadame_total` | 75.0% / 4 | 95.2% / 2 | 27.7% / 194 | 34.1% / 186 | HIGH_ARTIFACT_RISK |
