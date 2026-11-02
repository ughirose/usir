import { describe, it, expect } from 'vitest';
import {
  computeThreePhaseQdaDistance,
  computeSemanticGeodesicCurvature,
  computePunctuationRatioGate,
  computeMultiplicativeGrpoReward,
  embedSentenceUnit,
  splitIntoTrajectoryUnits,
} from '../src/dynamics/manifoldGeodesic.js';

describe('USIR v4.0 Manifold Geodesic & 3-Phase QDA Engine (TS)', () => {
  it('correctly applies freeze penalty to low-volatility deadzone states', () => {
    const zDeadzone = new Array<number>(24).fill(0.0);
    for (let i = 12; i < 24; i++) zDeadzone[i] = 0.02; // norm ~ 0.069 < 0.35

    const res = computeThreePhaseQdaDistance(zDeadzone, { sigma_crit_low: 0.35 });
    expect(res.penalty_freeze).toBeGreaterThan(0.0);
    expect(res.penalty_chaos).toBe(0.0);
    expect(res.phase_regime).toBe('Phase_I_Deadzone');
    expect(res.is_healthy_critical).toBe(false);
  });

  it('correctly applies chaos penalty to over-excited high-volatility states', () => {
    const zChaos = new Array<number>(24).fill(0.0);
    for (let i = 12; i < 24; i++) zChaos[i] = 0.60; // norm ~ 2.07 > 1.25

    const res = computeThreePhaseQdaDistance(zChaos, { sigma_crit_high: 1.25 });
    expect(res.penalty_chaos).toBeGreaterThan(0.0);
    expect(res.penalty_freeze).toBe(0.0);
    expect(res.phase_regime).toBe('Phase_III_Chaos');
    expect(res.is_healthy_critical).toBe(false);
  });

  it('recognizes healthy critical phase within double-well basin', () => {
    const zCrit = new Array<number>(24).fill(0.0);
    for (let i = 12; i < 24; i++) zCrit[i] = 0.20; // norm ~ 0.692 in [0.35, 1.25]

    const res = computeThreePhaseQdaDistance(zCrit, { sigma_crit_low: 0.35, sigma_crit_high: 1.25 });
    expect(res.penalty_freeze).toBe(0.0);
    expect(res.penalty_chaos).toBe(0.0);
    expect(res.phase_regime).toBe('Phase_II_Critical');
    expect(res.is_healthy_critical).toBe(true);
  });

  it('switches centroids and volatility bounds by domain (novel, article, cot_reasoning)', () => {
    const zCot = new Array<number>(24).fill(0.0);
    zCot[9] = 0.85;
    zCot[10] = 0.20;
    zCot[11] = 0.25;
    for (let i = 12; i < 24; i++) zCot[i] = 0.45; // norm ~ 1.5588

    const resNovel = computeThreePhaseQdaDistance(zCot, { domain: 'novel' });
    const resArticle = computeThreePhaseQdaDistance(zCot, { domain: 'article' });
    const resCot = computeThreePhaseQdaDistance(zCot, { domain: 'cot_reasoning' });

    expect(resNovel.penalty_chaos).toBeGreaterThan(0.0);
    expect(resNovel.phase_regime).toBe('Phase_III_Chaos');
    expect(resArticle.penalty_chaos).toBeGreaterThan(0.0);
    expect(resCot.penalty_chaos).toBe(0.0);
    expect(resCot.phase_regime).toBe('Phase_II_Critical');
    expect(resCot.d_3phase).toBeLessThan(resNovel.d_3phase);
  });

  it('measures low curvature and low Brownian ratio on smooth intentional narrative', () => {
    const smoothText =
      '東の空が白み始め、夜明けの冷気が森の梢を渡っていった。' +
      '冷たい風が梢を揺らし、凍りついた葉の雫を地面に落とした。' +
      '地面を濡らした雫は、踏み固められた山道を黒く染めていく。' +
      '山道を歩く私の足元から、湿った土と腐葉土の微かな匂いが立ち上った。';

    const res = computeSemanticGeodesicCurvature(smoothText);
    expect(res.mean_curvature).toBeLessThan(0.75);
    expect(res.brownian_ratio).toBeLessThan(3.5);
    expect(res.m_geodesic).toBeGreaterThan(0.40);
    expect(res.is_brownian_catalog).toBe(false);
  });

  it('quantitatively detects Brownian jumping on props cataloging passage', () => {
    const catalogText =
      '古い鉄鍋に赤錆が浮いている。' +
      '南の砂漠で駱駝が渇きに倒れた。' +
      '机の引き出しの真鍮の鍵が冷たい。' +
      '深海で光るクラゲが漂っていた。' +
      '硝子の破片が舗道に散らばっている。' +
      '遠い火山の噴煙が紫色の空を覆った。';

    const res = computeSemanticGeodesicCurvature(catalogText);
    expect(res.brownian_ratio).toBeGreaterThan(2.0);
    expect(res.mean_curvature).toBeGreaterThan(0.50);
  });

  it('evaluates classic v3.0 hot spring passage and measures props jump curvature', () => {
    const hotspringText =
      '障子を細く開けると、冷えた杉の匂いが雨とともに流れ込んできた。' +
      '軒先から落ちる水滴が、土間の踏み石に一つ、また一つと鈍い音を立てて砕ける。' +
      '谷底の湯治場は暮れ果てて、人の声はもう途絶えていた。' +
      '渡り廊下の突き当たり、裸電球の傘の下に、削り屑の積もった轆轤が据えてある。' +
      '濡れた粘土を指先で受ける老人は、こちらへ眼もくれず、ただ掌の重みを測るように背を丸めていた。' +
      '水を含んだ灰色の土が、指の隙間から細くせり上がる。' +
      '窯の焚口から漏れる微かな熱気だけが、乾いた板壁の節穴を赤く染めていた。' +
      '言葉はなかった。' +
      'ただ、轆轤の軸が軋むかすかな擦過音と、山を滑り降りてくる夜雨の気配ばかりが、薄暗い作業場を満たしていた。';

    const res = computeSemanticGeodesicCurvature(hotspringText);
    expect(res.units_count).toBeGreaterThanOrEqual(7);
    expect(res.mean_curvature).toBeGreaterThan(0.20);
    expect(res.brownian_ratio).toBeGreaterThan(1.2);
  });

  it('strictly collapses multiplicative reward to 0 if syntax punctuation gate fails', () => {
    const runOnText =
      '句読点がまったく存在せずどこまでも延々と続く危険な文章であり読者に息継ぎを与えない'.repeat(4);
    const rewardDict = computeMultiplicativeGrpoReward(runOnText);

    expect(rewardDict.m_syntax).toBe(0.0);
    expect(rewardDict.reward).toBe(0.0);
    expect(rewardDict.is_rejected).toBe(true);
  });

  it('yields healthy positive reward for well-formed natural passage', () => {
    const validText =
      '山あいの静かな宿で、旅の疲れを癒やすことにした。' +
      '夕暮れになると雨が静かに降り始め、古い軒先を濡らしている。' +
      '渡り廊下の奥からは、微かな杉の香りが漂っていた。';

    const rewardDict = computeMultiplicativeGrpoReward(validText);
    expect(rewardDict.m_syntax).toBeGreaterThan(0.5);
    expect(rewardDict.m_geodesic).toBeGreaterThan(0.3);
    expect(rewardDict.reward).toBeGreaterThan(0.0);
    expect(rewardDict.is_rejected).toBe(false);
  });

  it('verifies explicit embeddings parity with automatic unit embeddings', () => {
    const text = '静寂な山林に霧が立ち込める。冷たい風が梢を抜けていく。足元の苔が静かに息づいていた。';
    const units = splitIntoTrajectoryUnits(text);
    const autoRes = computeSemanticGeodesicCurvature(units);

    const explicitEmbs = units.map((u) => embedSentenceUnit(u));
    const expRes = computeSemanticGeodesicCurvature(units, { embeddings: explicitEmbs });

    expect(Math.abs(autoRes.mean_curvature - expRes.mean_curvature)).toBeLessThan(1e-6);
    expect(Math.abs(autoRes.brownian_ratio - expRes.brownian_ratio)).toBeLessThan(1e-6);
    expect(Math.abs(autoRes.m_geodesic - expRes.m_geodesic)).toBeLessThan(1e-6);
  });
});
