"""Exact Python port of nishiki Tier 2 RG (block_bootstrap L_0=64, stride=32) for HVG and RQA.

Matches:
- crates/nishiki-linalg/src/rg.rs (block_bootstrap, quantile_sorted)
- crates/nishiki-fractal/src/hvg.rs (hvg_edges, hvg_stats)
- crates/nishiki-rqa/src/rqa.rs (recurrence)
- crates/nishiki-cli/tests/fable_support/mod.rs (sentences, zscore_cloud, fractal_block, rqa_block, spearman)
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Sequence

import numpy as np

BLOCK_LEN = 64
BLOCK_STRIDE = 32


def _is_kanji(c: str) -> bool:
    code = ord(c)
    return (0x4E00 <= code <= 0x9FFF) or (0x3400 <= code <= 0x4DBF) or code == 0x3005


def _is_hiragana(c: str) -> bool:
    code = ord(c)
    return 0x3041 <= code <= 0x309F


def sentences(text: str) -> tuple[list[list[float]], list[str]]:
    """Extract sentence features [char_len, commas, kanji_ratio, hira_ratio] and sentence strings."""
    feats: list[list[float]] = []
    texts: list[str] = []
    cur: list[str] = []
    n = 0
    commas = 0
    kanji = 0
    hira = 0
    pending = False

    def flush() -> None:
        nonlocal cur, n, commas, kanji, hira
        if n > 0:
            length = float(n)
            feats.append([length, float(commas), kanji / length, hira / length])
            texts.append("".join(cur))
        cur = []
        n = 0
        commas = 0
        kanji = 0
        hira = 0

    for c in text:
        if pending and c not in ("」", "』", "）", "。", "！", "？", "!", "?"):
            flush()
            pending = False
        if c == "\n":
            flush()
            pending = False
            continue
        if c.isspace():
            continue
        cur.append(c)
        n += 1
        if c == "、":
            commas += 1
        elif c in ("。", "！", "？", "!", "?"):
            pending = True
        if _is_kanji(c):
            kanji += 1
        elif _is_hiragana(c):
            hira += 1

    flush()
    return feats, texts


def char_count_non_ws(text: str) -> int:
    return sum(1 for c in text if not c.isspace())


def quantile_sorted(xs: Sequence[float], q: float) -> float | None:
    if not xs:
        return None
    last = len(xs) - 1
    q_clamped = min(1.0, max(0.0, q)) if math.isfinite(q) else 0.5
    pos = q_clamped * last
    lo = min(int(math.floor(pos)), last)
    hi = min(lo + 1, last)
    frac = pos - lo
    v = (1.0 - frac) * xs[lo] + frac * xs[hi]
    return v if math.isfinite(v) else None


def block_bootstrap(
    series: Sequence,
    block_len: int,
    stride: int,
    g: Callable[[Sequence], float | None],
) -> dict:
    block_len = max(1, block_len)
    stride = max(1, stride)
    values: list[float] = []
    n_blocks = 0
    dropped = 0
    undersized = False

    if not series:
        return {
            "values": values,
            "n_blocks": 0,
            "dropped": 0,
            "undersized": False,
            "median": None,
            "iqr": None,
        }

    def push(block: Sequence) -> None:
        nonlocal n_blocks, dropped
        n_blocks += 1
        v = g(block)
        if v is not None and math.isfinite(v):
            values.append(float(v))
        else:
            dropped += 1

    if len(series) < block_len:
        undersized = True
        push(series)
    else:
        start = 0
        while start + block_len <= len(series):
            push(series[start : start + block_len])
            start += stride

    s_vals = sorted(values)
    med = quantile_sorted(s_vals, 0.5)
    q75 = quantile_sorted(s_vals, 0.75)
    q25 = quantile_sorted(s_vals, 0.25)
    iqr = (q75 - q25) if (q75 is not None and q25 is not None and math.isfinite(q75 - q25)) else None
    return {
        "values": values,
        "n_blocks": n_blocks,
        "dropped": dropped,
        "undersized": undersized,
        "median": med,
        "iqr": iqr,
    }


def hvg_stats(series: Sequence[float]) -> dict[str, float]:
    n = len(series)
    vals = [x if math.isfinite(x) else float("-inf") for x in series]
    edges: list[tuple[int, int]] = []
    stack: list[int] = []
    for j in range(n):
        xj = vals[j]
        while stack and vals[stack[-1]] < xj:
            edges.append((stack.pop(), j))
        if stack:
            top = stack[-1]
            edges.append((top, j))
            if vals[top] == xj:
                stack.pop()
        stack.append(j)

    deg = [0] * n
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1

    kmax = max(deg) if deg else 0
    hist = [0] * (kmax + 1)
    for d in deg:
        hist[d] += 1

    mean_degree = (2.0 * len(edges) / n) if n > 0 else 0.0
    total = float(sum(hist[2:]))
    kl_bits = 0.0
    if total > 0.0:
        log2_3 = math.log2(3.0)
        log2_2_3 = math.log2(2.0 / 3.0)
        for k in range(2, len(hist)):
            c = hist[k]
            if c > 0:
                p = c / total
                log2_th = -log2_3 + (k - 2.0) * log2_2_3
                kl_bits += p * (math.log2(p) - log2_th)

    m = 2.0 * len(edges)
    assortativity = 0.0
    if m > 0.0:
        sx = 0.0
        sxx = 0.0
        sxy = 0.0
        for a, b in edges:
            da = float(deg[a])
            db = float(deg[b])
            sx += da + db
            sxx += da * da + db * db
            sxy += 2.0 * da * db
        mean = sx / m
        var = sxx / m - mean * mean
        if var > 1e-12:
            assortativity = (sxy / m - mean * mean) / var

    return {
        "kl_bits": kl_bits,
        "assortativity": assortativity,
        "mean_degree": mean_degree,
    }


def zscore_cloud(rows: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(rows, dtype=np.float64)
    if arr.size == 0:
        return arr
    mu = arr.mean(axis=0)
    var = ((arr - mu) ** 2).mean(axis=0)
    out = np.zeros_like(arr)
    for k in range(arr.shape[1]):
        if var[k] > 1e-12:
            out[:, k] = (arr[:, k] - mu[k]) / math.sqrt(var[k])
    return out


def recurrence(
    points: np.ndarray | Sequence[Sequence[float]],
    target_rr: float = 0.05,
    l_min: int = 2,
    v_min: int = 2,
    max_points: int = 2000,
) -> dict[str, float]:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] == 0:
        return {"det": 0.0, "lam": 0.0, "tt": 0.0, "l_max_ratio": 0.0}

    finite_mask = np.all(np.isfinite(pts), axis=1)
    pts = pts[finite_mask]
    cap = max(3, max_points)
    if len(pts) > cap:
        idx = [(k * len(pts)) // cap for k in range(cap)]
        pts = pts[idx]

    n = len(pts)
    if n < 3:
        return {"det": 0.0, "lam": 0.0, "tt": 0.0, "l_max_ratio": 0.0}

    # Pairwise Euclidean distance matrix
    diff = pts[:, None, :] - pts[None, :, :]
    dmat = np.sqrt(np.sum(diff * diff, axis=-1))
    iu = np.triu_indices(n, k=1)
    all_d = dmat[iu]
    m_pairs = len(all_d)
    clamped_rr = min(1.0, max(0.0, target_rr))
    rank = max(1, min(m_pairs, math.ceil(clamped_rr * m_pairs))) - 1
    eps = float(np.partition(all_d, rank)[rank])

    bits = dmat <= eps
    np.fill_diagonal(bits, False)
    rec_pairs = int(np.sum(bits[iu]))

    # Diagonal lines (upper triangle k=1..n-1)
    l_min = max(1, l_min)
    diag_pts = 0
    l_max = 0
    for k in range(1, n):
        diag = np.diagonal(bits, offset=k)
        run = 0
        for on in diag:
            if on:
                run += 1
            elif run > 0:
                if run > l_max:
                    l_max = run
                if run >= l_min:
                    diag_pts += run
                run = 0
        if run > 0:
            if run > l_max:
                l_max = run
            if run >= l_min:
                diag_pts += run

    det = (diag_pts / rec_pairs) if rec_pairs > 0 else 0.0

    # Vertical lines (include main diagonal)
    v_min = max(1, v_min)
    bits_v = bits.copy()
    np.fill_diagonal(bits_v, True)
    vert_pts = 0
    vert_lines = 0
    col_pts = int(np.sum(bits_v))
    for j in range(n):
        col = bits_v[:, j]
        run = 0
        for on in col:
            if on:
                run += 1
            elif run > 0:
                if run >= v_min:
                    vert_pts += run
                    vert_lines += 1
                run = 0
        if run > 0:
            if run >= v_min:
                vert_pts += run
                vert_lines += 1

    lam = (vert_pts / col_pts) if col_pts > 0 else 0.0
    tt = (vert_pts / vert_lines) if vert_lines > 0 else 0.0
    return {
        "det": det,
        "lam": lam,
        "tt": tt,
        "l_max_ratio": l_max / max(1, n),
    }


def compute_tier2_all(feats: Sequence[Sequence[float]]) -> dict[str, float]:
    """Compute all 6 Tier 2 RG metrics (block_bootstrap L_0=64, stride=32) for a sentence feature list."""
    sent_len = [f[0] for f in feats]
    # HVG per block (compute once per block for efficiency)
    bs_hvg = block_bootstrap(
        sent_len,
        BLOCK_LEN,
        BLOCK_STRIDE,
        lambda b: hvg_stats(b)["kl_bits"],
    )
    # Wait: we can compute all 3 HVG stats per block cleanly:
    hvg_blocks: list[dict[str, float]] = []
    if sent_len:
        if len(sent_len) < BLOCK_LEN:
            hvg_blocks.append(hvg_stats(sent_len))
        else:
            start = 0
            while start + BLOCK_LEN <= len(sent_len):
                hvg_blocks.append(hvg_stats(sent_len[start : start + BLOCK_LEN]))
                start += BLOCK_STRIDE

    def med(vals: Iterable[float]) -> float:
        s = sorted(vals)
        v = quantile_sorted(s, 0.5)
        return v if v is not None else 0.0

    hvg_kl = med(b["kl_bits"] for b in hvg_blocks)
    hvg_assort = med(b["assortativity"] for b in hvg_blocks)
    hvg_mean_degree = med(b["mean_degree"] for b in hvg_blocks)

    cloud = zscore_cloud(feats)
    rqa_blocks: list[dict[str, float]] = []
    if len(cloud) > 0:
        if len(cloud) < BLOCK_LEN:
            rqa_blocks.append(recurrence(cloud))
        else:
            start = 0
            while start + BLOCK_LEN <= len(cloud):
                rqa_blocks.append(recurrence(cloud[start : start + BLOCK_LEN]))
                start += BLOCK_STRIDE

    rqa_det = med(b["det"] for b in rqa_blocks)
    rqa_lam = med(b["lam"] for b in rqa_blocks)
    rqa_tt = med(b["tt"] for b in rqa_blocks)

    return {
        "hvg_kl": hvg_kl,
        "hvg_assort": hvg_assort,
        "hvg_mean_degree": hvg_mean_degree,
        "rqa_det": rqa_det,
        "rqa_lam": rqa_lam,
        "rqa_tt": rqa_tt,
        "n_sentences": len(feats),
        "n_blocks": len(hvg_blocks),
    }


def _ranks(v: Sequence[float]) -> list[float]:
    idx = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and v[idx[j + 1]] == v[idx[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            r[idx[k]] = avg
        i = j + 1
    return r


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    ma = sum(a[:n]) / n
    mb = sum(b[:n]) / n
    sab = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    saa = sum((a[i] - ma) ** 2 for i in range(n))
    sbb = sum((b[i] - mb) ** 2 for i in range(n))
    if saa <= 0.0 or sbb <= 0.0:
        return 0.0
    return sab / math.sqrt(saa * sbb)


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    return pearson(_ranks(a), _ranks(b))
