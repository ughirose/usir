#!/usr/bin/env python
"""Aozora Bunko fetcher / cleaner (Phase 18).

Downloads public-domain texts from the official site (https://www.aozora.gr.jp/cards/<person id>/files/<file>.zip);
the GitHub mirror raw URL pattern (https://github.com/aozorabunko/aozorabunko/raw/master/cards/<person id>/files/<file>.zip)
returned HTTP 404 on 2026-11-05 and is kept only as a fallback.
decodes Shift_JIS (cp932), strips Aozora notation (ruby, ｜, ［＃…］, header block, 底本 trailer),
caches cleaned text in ``data/aozora_cache/`` (git-ignored) and writes ``data/aozora_manifest.json``.

Work ids are resolved from the official index ``list_person_all_extended_utf8.zip`` (author id +
title + character type), so no file id is hard-coded.  Only works with the copyright flag "なし"
whose author died in 1967 or earlier are accepted.

Author / work names appear only in the manifest (labels); they never enter a feature extractor.

Usage:
    python scripts/fetch_aozora.py            # fetch everything missing from the cache, write manifest
    python scripts/fetch_aozora.py --offline  # build the manifest from the cache only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "aozora_cache"
MANIFEST_PATH = ROOT / "data" / "aozora_manifest.json"

USER_AGENT = "usir-research-fetcher"
DELAY_SEC = 1.0
INDEX_URL = "https://www.aozora.gr.jp/index_pages/list_person_all_extended_utf8.zip"
MIRROR_RAW = "https://github.com/aozorabunko/aozorabunko/raw/master/cards/{pid}/files/{fname}"
AOZORA_PREFIX = "https://www.aozora.gr.jp/cards/"
MAX_DEATH_YEAR = 1967  # life+50 had already expired before the 2018 extension to life+70

# (aozora person id, Japanese author name, genre, [titles in priority order], number of works wanted)
PLAN = [
    ("000148", "夏目漱石", "novel", ["坊っちゃん", "夢十夜", "草枕", "三四郎", "こころ"], 5),
    ("000879", "芥川龍之介", "novel", ["羅生門", "鼻", "芋粥", "地獄変", "藪の中", "杜子春"], 5),
    ("000035", "太宰治", "novel", ["走れメロス", "人間失格", "斜陽", "女生徒", "富嶽百景"], 4),
    ("000081", "宮沢賢治", "novel", ["銀河鉄道の夜", "注文の多い料理店", "風の又三郎", "セロ弾きのゴーシュ", "よだかの星"], 4),
    ("000074", "梶井基次郎", "novel", ["檸檬", "桜の樹の下には", "城のある町にて", "闇の絵巻", "愛撫"], 5),
    ("000129", "森鴎外", "novel", ["高瀬舟", "山椒大夫", "阿部一族", "雁", "舞姫"], 5),
    ("000050", "泉鏡花", "novel", ["高野聖", "外科室", "夜行巡査", "草迷宮", "化鳥", "日本橋"], 4),
    ("001383", "谷崎潤一郎", "novel", ["刺青", "秘密", "少年", "春琴抄", "痴人の愛"], 5),
    ("001095", "坂口安吾", "novel", ["白痴", "桜の森の満開の下", "青鬼の褌を洗う女", "風博士", "夜長姫と耳男", "不連続殺人事件"], 4),
    ("000119", "中島敦", "novel", ["山月記", "李陵", "弟子", "名人伝", "光と風と夢"], 5),
    # genre "article": essays / criticism / lectures
    ("000042", "寺田寅彦", "article", ["科学者とあたま", "天災と国防", "津浪と人間", "地震雑感", "日本人の自然観", "藤棚の陰から", "物理学と感覚"], 7),
    ("001095", "坂口安吾", "article", ["堕落論", "日本文化私観"], 2),
    ("000148", "夏目漱石", "article", ["私の個人主義", "現代日本の開化"], 2),
    ("001395", "和辻哲郎", "article", ["鎖国", "古寺巡礼"], 2),
]

# ---------------------------------------------------------------------------
# Cleaner
# ---------------------------------------------------------------------------

_DASH_LINE = re.compile(r"^-{20,}\s*$", re.M)
_RUBY = re.compile(r"《[^》\n]*》")
_ANNOT = re.compile(r"［＃[^］\n]*］")
_GAIJI = re.compile(r"※(?=［＃)")
_FOOTER = re.compile(r"^底本[：:]", re.M)
_ACCENT = re.compile(r"[〔〕]")


def clean_aozora_text(raw: str) -> str:
    """Strip Aozora notation from decoded text.

    Order: header block -> 底本 trailer -> gaiji marker + ［＃…］ annotations -> ruby 《…》 -> ｜.
    The header is everything up to and including the closing line of the "-------" block that
    explains notation symbols (title / author lines precede it).  If no such block exists the
    text is kept from its start.
    """
    text = raw.replace("\r\n", "\n").replace("\r", "\n")

    dashes = list(_DASH_LINE.finditer(text))
    if len(dashes) >= 2:
        text = text[dashes[1].end():]

    footers = [m.start() for m in _FOOTER.finditer(text)]
    if footers:
        tail_zone = max(len(text) - 2500, 0)
        in_zone = [s for s in footers if s >= tail_zone]
        text = text[: (in_zone[0] if in_zone else footers[-1])]

    text = _GAIJI.sub("", text)
    text = _ANNOT.sub("", text)
    text = _RUBY.sub("", text)
    text = text.replace("｜", "")
    text = _ACCENT.sub("", text)

    lines = [ln.strip().lstrip("　").rstrip() for ln in text.split("\n")]
    out: List[str] = []
    blank = 0
    for ln in lines:
        if not ln:
            blank += 1
            continue
        if blank and out:
            out.append("")  # a single blank line is kept as a paragraph break
        blank = 0
        out.append(ln)
    return "\n".join(out).strip() + "\n"


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------

_last_request = [0.0]


def _http_get(url: str) -> bytes:
    wait = DELAY_SEC - (time.time() - _last_request[0])
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read()
    finally:
        _last_request[0] = time.time()


def load_index(cache_dir: Path = CACHE_DIR) -> List[Dict[str, str]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / "_index.csv"
    if not csv_path.exists():
        data = _http_get(INDEX_URL)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            name = next(n for n in zf.namelist() if n.endswith(".csv"))
            csv_path.write_bytes(zf.read(name))
    with open(csv_path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _death_year(row: Dict[str, str]) -> Optional[int]:
    m = re.match(r"\s*(\d{4})", row.get("没年月日", ""))
    return int(m.group(1)) if m else None


_CHAR_PREF = {"新字新仮名": 0, "新字旧仮名": 1, "旧字新仮名": 2, "旧字旧仮名": 3}


def resolve_work(index: List[Dict[str, str]], pid: str, title: str) -> Optional[Dict[str, str]]:
    cands = []
    for r in index:
        if r["人物ID"] != pid or r["作品名"] != title or r["役割フラグ"] != "著者":
            continue
        if r["作品著作権フラグ"] != "なし":
            continue
        dy = _death_year(r)
        if dy is None or dy > MAX_DEATH_YEAR:
            continue
        if not r["テキストファイルURL"].startswith(AOZORA_PREFIX):
            continue
        cands.append(r)
    if not cands:
        return None
    cands.sort(key=lambda r: (_CHAR_PREF.get(r["文字遣い種別"], 9), int(r["作品ID"])))
    return cands[0]


def fetch_clean_text(row: Dict[str, str]) -> Dict[str, str]:
    aozora_url = row["テキストファイルURL"]
    rel = aozora_url[len(AOZORA_PREFIX):]  # <pid>/files/<file>.zip
    pid, _, fname = rel.split("/", 2)
    mirror_url = MIRROR_RAW.format(pid=pid, fname=fname)
    # As of 2026-11-05 the GitHub mirror raw URLs return HTTP 404 (repo contents API as well), so the
    # official site is tried first and the mirror pattern is kept as a fallback.
    try:
        data = _http_get(aozora_url)
        used = aozora_url
    except Exception as exc:  # noqa: BLE001
        print(f"  official site failed ({exc}); falling back to {mirror_url}", file=sys.stderr)
        data = _http_get(mirror_url)
        used = mirror_url
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".txt"))
        raw = zf.read(name).decode("cp932", errors="replace")
    return {"text": clean_aozora_text(raw), "zip_url": used}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _cache_file(author_id: str, genre: str, work_id: str) -> Path:
    return CACHE_DIR / f"{genre}_{author_id}_{work_id}.txt"


def build(offline: bool = False) -> List[Dict]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    index = load_index()
    manifest: List[Dict] = []
    for pid, author_ja, genre, titles, want in PLAN:
        got = 0
        for title in titles:
            if got >= want:
                break
            row = resolve_work(index, pid, title)
            if row is None:
                print(f"[skip] {author_ja} 「{title}」: not found / not public domain in index")
                continue
            work_id = row["作品ID"]
            cpath = _cache_file(pid, genre, work_id)
            meta_path = cpath.with_suffix(".json")
            if cpath.exists() and meta_path.exists():
                text = cpath.read_text(encoding="utf-8")
                zip_url = json.loads(meta_path.read_text(encoding="utf-8"))["zip_url"]
            elif offline:
                print(f"[skip] {author_ja} 「{title}」: not cached (offline)")
                continue
            else:
                try:
                    fetched = fetch_clean_text(row)
                except Exception as exc:  # noqa: BLE001
                    print(f"[fail] {author_ja} 「{title}」: {exc}; trying next candidate", file=sys.stderr)
                    continue
                text, zip_url = fetched["text"], fetched["zip_url"]
                cpath.write_text(text, encoding="utf-8")
                meta_path.write_text(json.dumps({"zip_url": zip_url}), encoding="utf-8")
            if len(text) < 1500:
                print(f"[skip] {author_ja} 「{title}」: only {len(text)} chars after cleaning")
                continue
            manifest.append({
                "author_id": pid,
                "author_ja": author_ja,
                "title": title,
                "work_id": work_id,
                "char_type": row["文字遣い種別"],
                "card_url": row["図書カードURL"],
                "zip_url": zip_url,
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "char_count": len(text),
                "genre": genre,
            })
            got += 1
            print(f"[ok] {genre} {author_ja} 「{title}」 id={work_id} chars={len(text)}")
        if got < want:
            print(f"[warn] {author_ja} ({genre}): wanted {want}, got {got}")
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"manifest: {len(manifest)} works -> {MANIFEST_PATH}")
    return manifest


def load_manifest_texts(genre: Optional[str] = None) -> List[Dict]:
    """Return manifest entries with a ``text`` field read from the cache (raises if missing)."""
    entries = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    out = []
    for e in entries:
        if genre and e["genre"] != genre:
            continue
        p = _cache_file(e["author_id"], e["genre"], e["work_id"])
        text = p.read_text(encoding="utf-8")
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != e["sha256"]:
            raise ValueError(f"cache sha256 mismatch for {e['title']} ({p})")
        out.append({**e, "text": text})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--offline", action="store_true", help="use the cache only")
    build(offline=ap.parse_args().offline)
