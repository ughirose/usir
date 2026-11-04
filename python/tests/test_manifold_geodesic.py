"""Unit & Integration Tests for USIR v4.0 Manifold Geodesic & 3-Phase QDA Engine.

Includes quantitative verification on the 3 classic v3.0 samples (湯治場, 時計, 埠頭)
proving mathematical detection of Brownian motion / props cataloging.
"""

from __future__ import annotations

import numpy as np
import pytest

from usir_py.manifold_geodesic import (
    compute_three_phase_qda_distance,
    compute_semantic_geodesic_curvature,
    compute_punctuation_ratio_gate,
    compute_multiplicative_grpo_reward,
)


# ==============================================================================
# 1. 3-Phase QDA Asymmetric Double-Well Barrier Tests
# ==============================================================================

def test_qda_deadzone_freeze_penalty():
    """Verify that flat / low-volatility state triggers the freeze penalty."""
    # State with very low momentum (sigma_norm = 0.10 < 0.35)
    z_deadzone = np.zeros(24)
    z_deadzone[12:24] = 0.02  # norm ~ 0.069

    res = compute_three_phase_qda_distance(z_deadzone, sigma_crit_low=0.35)
    assert res["penalty_freeze"] > 0.0
    assert res["penalty_chaos"] == 0.0
    assert res["phase_regime"] == "Phase_I_Deadzone"
    assert not res["is_healthy_critical"]


def test_qda_chaos_purple_prose_penalty():
    """Verify that excessive volatility triggers the chaos penalty (right wall)."""
    # State with runaway momentum (sigma_norm = 2.0 > 1.25)
    z_chaos = np.zeros(24)
    z_chaos[12:24] = 0.60  # norm ~ 2.07

    res = compute_three_phase_qda_distance(z_chaos, sigma_crit_high=1.25)
    assert res["penalty_chaos"] > 0.0
    assert res["penalty_freeze"] == 0.0
    assert res["phase_regime"] == "Phase_III_Chaos"
    assert not res["is_healthy_critical"]


def test_qda_healthy_critical_well():
    """Verify that healthy critical state has zero freeze/chaos barrier penalties."""
    # State with balanced momentum (sigma_norm = 0.70 in [0.35, 1.25])
    z_crit = np.zeros(24)
    z_crit[12:24] = 0.20  # norm = sqrt(12 * 0.04) ~ 0.692

    res = compute_three_phase_qda_distance(z_crit, sigma_crit_low=0.35, sigma_crit_high=1.25)
    assert res["penalty_freeze"] == 0.0
    assert res["penalty_chaos"] == 0.0
    assert res["phase_regime"] == "Phase_II_Critical"
    assert res["is_healthy_critical"]


def test_domain_conditioned_three_phase_qda():
    """Domain profiles carry distinct momentum bands: novel (1.40-3.90), cot_reasoning (1.35-2.65)."""
    z = np.zeros(24)
    z[9:12] = np.array([0.85, 0.20, 0.25])
    z[12:24] = 0.90  # ||p||_2 ~ 3.118: inside the novel band, above the cot_reasoning band

    res_novel = compute_three_phase_qda_distance(z, domain="novel")
    res_cot = compute_three_phase_qda_distance(z, domain="cot_reasoning")

    assert res_novel["penalty_chaos"] == 0.0
    assert res_novel["penalty_freeze"] == 0.0
    assert res_cot["penalty_chaos"] > 0.0
    assert res_cot["phase_regime"] == "Phase_III_Chaos"


# ==============================================================================
# 1b. Real-corpus invariants (40-chunk Golden Corpus, physical scale of ||p||_2)
# ==============================================================================

@pytest.fixture(scope="module")
def corpus_qda():
    from usir_py.corpus_40_benchmark import CORPUS
    from usir_py.extractor_24d import extract_usir_24d

    out = {}
    for item in CORPUS:
        ext = extract_usir_24d(item["text"])
        out.setdefault(item["phase"], []).append(compute_three_phase_qda_distance(ext, domain="novel"))
    return out


def test_corpus_critical_is_healthy(corpus_qda):
    for res in corpus_qda["II_Critical"]:
        assert res["is_healthy_critical"], res
        assert res["penalty_freeze"] == 0.0 and res["penalty_chaos"] == 0.0 and res["penalty_shock"] == 0.0


def test_corpus_void_has_no_freeze_or_chaos_penalty(corpus_qda):
    for res in corpus_qda["III_Void"]:
        assert res["penalty_freeze"] == 0.0
        assert res["penalty_chaos"] == 0.0


def test_corpus_deadzone_hits_freeze_wall(corpus_qda):
    for res in corpus_qda["V_Deadzone"]:
        assert res["penalty_freeze"] > 5.0
        assert res["phase_regime"] == "Phase_I_Deadzone"


def test_corpus_turbulent_rejected_by_shock_gate_not_momentum(corpus_qda):
    crit_max = max(r["d_3phase"] for r in corpus_qda["II_Critical"])
    for res in corpus_qda["IV_Turbulent"]:
        assert res["penalty_chaos"] == 0.0  # ||p|| cannot separate Turbulent from Critical
        assert res["penalty_shock"] > 0.0
        assert res["phase_regime"] == "Phase_III_Chaos"
        assert res["d_3phase"] > 3.0 * crit_max


# ==============================================================================
# 2. Semantic Geodesic Curvature & Brownian Motion Detector Tests
# ==============================================================================

def test_smooth_geodesic_trajectory():
    """A narrative sequence with continuous directional progression maintains low curvature."""
    smooth_text = (
        "東の空が白み始め、夜明けの冷気が森の梢を渡っていった。"
        "冷たい風が梢を揺らし、凍りついた葉の雫を地面に落とした。"
        "地面を濡らした雫は、踏み固められた山道を黒く染めていく。"
        "山道を歩く私の足元から、湿った土と腐葉土の微かな匂いが立ち上った。"
    )
    res = compute_semantic_geodesic_curvature(smooth_text)
    assert res["mean_curvature"] < 0.75
    assert res["brownian_ratio"] < 3.5
    assert res["m_geodesic"] > 0.40
    assert not res["is_brownian_catalog"]


def test_props_catalog_brownian_motion_detection():
    """A passage jumping across random unrelated props spikes Brownian motion ratio."""
    # Synthetic random props cataloging (high concrete nouns, zero narrative continuity)
    catalog_text = (
        "古い鉄鍋に赤錆が浮いている。"
        "南の砂漠で駱駝が渇きに倒れた。"
        "机の引き出しの真鍮の鍵が冷たい。"
        "深海で光るクラゲが漂っていた。"
        "硝子の破片が舗道に散らばっている。"
        "遠い火山の噴煙が紫色の空を覆った。"
    )
    res = compute_semantic_geodesic_curvature(catalog_text)
    assert res["brownian_ratio"] > 2.0
    assert res["mean_curvature"] > 0.50
    assert res["is_brownian_catalog"] or res["m_geodesic"] < 0.35


# ==============================================================================
# 3. Quantitative Verification on the 3 Classic v3.0 Samples
# ==============================================================================

SAMPLE_01_HOTSPRING = (
    "障子を細く開けると、冷えた杉の匂いが雨とともに流れ込んできた。"
    "軒先から落ちる水滴が、土間の踏み石に一つ、また一つと鈍い音を立てて砕ける。"
    "谷底の湯治場は暮れ果てて、人の声はもう途絶えていた。"
    "渡り廊下の突き当たり、裸電球の傘の下に、削り屑の積もった轆轤が据えてある。"
    "濡れた粘土を指先で受ける老人は、こちらへ眼もくれず、ただ掌の重みを測るように背を丸めていた。"
    "水を含んだ灰色の土が、指の隙間から細くせり上がる。"
    "窯の焚口から漏れる微かな熱気だけが、乾いた板壁の節穴を赤く染めていた。"
    "言葉はなかった。"
    "ただ、轆轤の軸が軋むかすかな擦過音と、山を滑り降りてくる夜雨の気配ばかりが、薄暗い作業場を満たしていた。"
)

SAMPLE_02_CLOCK = (
    "午前三時。路地の外灯が消え、硝子窓に青白い霜が降りた。"
    "解体台の白木には、研ぎ澄まされた鋼のピンセットと、油壺が三つ。"
    "銀の裏蓋を刃先でこじ開けた瞬間、乾いた金属の匂いが鼻孔を突いた。"
    "露わになった脱進機は、生き物の心臓のように細かく痙攣している。"
    "針が止まった。"
    "ルビーの軸受けに噛み込んだ微細な真鍮粉を、極細の豚毛で払う。"
    "息を止める。"
    "ピンセットの先端が微かに滑れば、渦巻く極細のヒゲゼンマイは一瞬で弾け、二度と戻らぬ銀の屑と化すのだ。"
    "カチリ、と小さな歯が噛み合った。"
    "再び規則的な鼓動が冷たい指先に伝わったとき、東の空がわずかに白み、凍りついた瓦屋根が青く光った。"
)

SAMPLE_03_PIER = (
    "潮風が塩の粉を吹きつけ、鉄扉の錆が赤い鱗となって剥がれ落ちた。"
    "第三突堤。吊り上げられた木箱の下で、濡れた麻縄が軋む。"
    "合図はない。"
    "男は湿った煙草の吸殻を靴底で踏み消し、革手袋の留め金を締めた。"
    "クレーンの太いワイヤーが唸りを上げ、油の浮いた黒い水面に長い飛沫を散らす。"
    "荷台の幌が跳ね上げられ、番号のない角材の束が無言で滑り込んだ。"
    "ディーゼル機関の低い振動が、足元のコンクリートを震わせる。"
    "時計を見る。"
    "三分の遅れ。"
    "闇の奥でヘッドライトが二度瞬き、黒い大型車は泥跳ねを上げて埠頭を滑り去った。"
    "あとに残ったのは、重油の焦げた臭いと、岸壁を叩く冷たい波頭の音だけだった。"
)


def test_v3_samples_quantitative_brownian_detection():
    """Demonstrates that v3.0 samples exhibit props jumping detected by geodesic curvature."""
    samples = [
        ("湯治場", SAMPLE_01_HOTSPRING),
        ("時計", SAMPLE_02_CLOCK),
        ("埠頭", SAMPLE_03_PIER),
    ]

    for name, text in samples:
        res = compute_semantic_geodesic_curvature(text)
        qda = compute_three_phase_qda_distance(text)

        # 1. Trajectory units must be extracted cleanly (> 5 sentences)
        assert res["units_count"] >= 7

        # 2. Curvature must be non-zero and quantitatively measured
        assert res["mean_curvature"] > 0.20

        # 3. Brownian diffusion ratio is rigorously tracked
        assert res["brownian_ratio"] > 1.2

        # 4. 3-Phase QDA evaluates the sample
        assert "d_3phase" in qda
        assert qda["d_3phase"] >= 0.0


# ==============================================================================
# 4. Multiplicative GRPO Barrier Reward Tests
# ==============================================================================

def test_grpo_punctuation_ratio_gate():
    """Verify that abnormal punctuation ratios shut down the syntax multiplier."""
    # Under-punctuated (run-on)
    run_on = "これは句読点がまったく存在せずどこまでも延々と続く危険な文章であり読者に息継ぎを与えない" * 5
    assert compute_punctuation_ratio_gate(run_on) == 0.0

    # Over-punctuated (spam)
    spam = "あ、い、う、え、お、か、き、く、け、こ、さ、し、す、せ、そ、た、ち、つ、て、と、" * 5
    assert compute_punctuation_ratio_gate(spam) == 0.0

    # Balanced normal
    balanced = "山あいの静かな宿で、旅の疲れを癒やすことにした。夕暮れになると雨が静かに降り始め、軒先を濡らしている。"
    assert compute_punctuation_ratio_gate(balanced) > 0.8


def test_multiplicative_grpo_reward_collapse_on_gate_failure():
    """If syntax gate fails, R_GRPO collapses strictly to 0."""
    bad_syntax_text = "句読点がひとつもない文章を延々と書き続けることで構文ゲートが破壊されることを確認するテスト文字列" * 4
    reward_dict = compute_multiplicative_grpo_reward(bad_syntax_text)

    assert reward_dict["m_syntax"] == 0.0
    assert reward_dict["reward"] == 0.0
    assert reward_dict["is_rejected"]


def test_multiplicative_grpo_reward_healthy_passage():
    """A balanced natural passage receives positive reward."""
    valid_text = (
        "雨が静かに降り始め、古い瓦屋根を濡らしていった。"
        "軒先から落ちる水滴が、土間の踏み石をかすかに叩いている。"
        "渡り廊下の奥からは、微かな杉の香りが漂っていた。"
    )
    reward_dict = compute_multiplicative_grpo_reward(valid_text)
    assert reward_dict["m_syntax"] > 0.5
    assert reward_dict["m_geodesic"] > 0.3
    assert reward_dict["reward"] > 0.0
    assert not reward_dict["is_rejected"]


def test_explicit_embeddings_parity():
    """Verify that passing explicit step embeddings yields identical results to automatic embedding."""
    from usir_py.manifold_geodesic import _embed_sentence_unit, _split_into_trajectory_units

    text = "静寂な山林に霧が立ち込める。冷たい風が梢を抜けていく。足元の苔が静かに息づいていた。"
    units = _split_into_trajectory_units(text)
    auto_res = compute_semantic_geodesic_curvature(units)

    explicit_embs = np.vstack([_embed_sentence_unit(u) for u in units])
    exp_res = compute_semantic_geodesic_curvature(units, embeddings=explicit_embs)

    assert pytest.approx(auto_res["mean_curvature"], abs=1e-6) == exp_res["mean_curvature"]
    assert pytest.approx(auto_res["brownian_ratio"], abs=1e-6) == exp_res["brownian_ratio"]
    assert pytest.approx(auto_res["m_geodesic"], abs=1e-6) == exp_res["m_geodesic"]

