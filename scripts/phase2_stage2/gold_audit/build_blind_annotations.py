"""Step 1: Build and validate blind Google-model annotations for 3 modern works (11 scenes).

Must be run and committed BEFORE opening scripts/phase2_stage2/gold/stage_b_gold.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "python"))

from usir_py.sensor_eval.scene_annotator import (  # noqa: E402
    dumps,
    scene_prompt_hash,
    validate_annotation_schema,
    validate_scene_output,
)

NISHIKI = "C:/dev/nishiki-phase2/target/release/nishiki.exe"
TEXT_DIR = REPO / "scripts/step1/modern_gold"
ROSTER_DIR = REPO / "scripts/phase2_stage2/annotations_rerun"
SCHEMA_PATH = REPO / "schemas/sensor_v2.schema.json"
OUT_DIR = HERE / "annotations"

WORKS = [
    ("001798_056779", "boku"),
    ("001801_001786", "shinichi"),
    ("001875_059897", None),
]

RAW_ANNOTATIONS: dict[str, dict[str, Any]] = {
    "001798_056779:s0": {
        "time": {
            "relative_order": "unknown",
            "explicit_time": "不破と知合いになって三日目",
            "elapsed_hint": None,
            "evidence": "explicit",
        },
        "location": {"surface": "代田橋"},
        "present_characters": [
            {"id": "boku", "basis": "僕は世帯道具や画の道具一式をオート三輪に積みこんで、はるばる八王子から代田橋にバタバタバタと引越して来ました"},
            {"id": "fuwa", "basis": "その夜不破とかんたんな契約書を作成し、四万円を手交し"},
        ],
        "character_states": [
            {"character_id": "boku", "values_6d": [0, 0, 0, 0, 0, 0], "evidence": "inferred", "basis": "その夜不破とかんたんな契約書を作成し、四万円を手交し"},
            {"character_id": "fuwa", "values_6d": [0, 0, 0, 0, 0, 0], "evidence": "inferred", "basis": "その夜不破とかんたんな契約書を作成し"},
        ],
        "fact_updates": [
            {"subject": "boku", "predicate": "MOVED_TO", "object": "代田橋", "evidence": "explicit", "basis": "はるばる八王子から代田橋にバタバタバタと引越して来ました"},
            {"subject": "boku", "predicate": "ACQUIRED", "object": "契約書", "evidence": "explicit", "basis": "その夜不破とかんたんな契約書を作成し"},
            {"subject": "fuwa", "predicate": "ACQUIRED", "object": "四万円", "evidence": "explicit", "basis": "四万円を手交し"},
        ],
    },
    "001798_056779:s1": {
        "time": {
            "relative_order": "continuous",
            "explicit_time": "失踪の前の晩",
            "elapsed_hint": "一週間目に",
            "evidence": "explicit",
        },
        "location": {"surface": "僕の部屋"},
        "present_characters": [
            {"id": "boku", "basis": "なけなしの財布から千円札二枚渡してやりました"},
            {"id": "fuwa", "basis": "不破数馬はすこし酔って僕の部屋にやってきて"},
        ],
        "character_states": [
            {"character_id": "boku", "values_6d": [0, 1, 0, 0, 0, 0], "evidence": "inferred", "basis": "間代前納の形式ならそれでもいいと思った"},
            {"character_id": "fuwa", "values_6d": [0, -1, -1, -1, -1, 0], "evidence": "explicit", "basis": "一週間目に夫妻もろとも、夜逃げというか昼逃げというか、どこかに失踪してしまったのです"},
        ],
        "fact_updates": [
            {"subject": "fuwa", "predicate": "STATE_CHANGED", "object": "夫妻で失踪した", "evidence": "explicit", "basis": "一週間目に夫妻もろとも、夜逃げというか昼逃げというか、どこかに失踪してしまったのです"},
            {"subject": "fuwa", "predicate": "ACQUIRED", "object": "二千円", "evidence": "explicit", "basis": "なけなしの財布から千円札二枚渡してやりました"},
            {"subject": "fuwa", "predicate": "HOOK_OPENED", "object": "野呂が留守番として来る約束", "evidence": "explicit", "basis": "その間の留守番として、遠縁の野呂という男をここに寝泊りさせるから、仲良くしてやってくれ"},
        ],
    },
    "001798_056779:s2": {
        "time": {
            "relative_order": "continuous",
            "explicit_time": "翌朝、十一時頃",
            "elapsed_hint": "翌朝",
            "evidence": "explicit",
        },
        "location": {"surface": "板の間"},
        "present_characters": [
            {"id": "boku", "basis": "僕が板の間にイーゼルを立てて庭の写生をしていると"},
            {"id": "noro", "basis": "野呂がやって来たのは、それから二三日経った正午頃でした"},
        ],
        "character_states": [
            {"character_id": "boku", "values_6d": [0, 1, 0, 0, 0, 0], "evidence": "inferred", "basis": "「留守番は君じゃないか。僕はここの居住者だ」"},
            {"character_id": "noro", "values_6d": [0, 0, -1, -1, 0, 0], "evidence": "explicit", "basis": "この家は僕のものだよ。不破氏から僕が買い受けたんだ。君は早々に出て行って呉れ」"},
        ],
        "fact_updates": [
            {"subject": "boku", "predicate": "ACQUIRED", "object": "不破の便箋", "evidence": "explicit", "basis": "直ぐその便箋は折り畳んでポケットにしまいました"},
            {"subject": "noro", "predicate": "MOVED_TO", "object": "この家", "evidence": "explicit", "basis": "野呂がやって来たのは、それから二三日経った正午頃でした"},
            {"subject": "noro", "predicate": "LEARNED", "object": "不破が兵庫県に立ったこと", "evidence": "explicit", "basis": "「不破さんは昨日兵庫県に立ちましたよ」"},
            {"subject": "boku", "predicate": "LEARNED", "object": "野呂が不破からこの家を買い受けたこと", "evidence": "explicit", "basis": "この家は僕のものだよ。不破氏から僕が買い受けたんだ。君は早々に出て行って呉れ」"},
            {"subject": "noro", "predicate": "STATE_CHANGED", "object": "急に言葉がぞんざいになった", "evidence": "explicit", "basis": "急に言葉がぞんざいになりました"},
            {"subject": None, "predicate": "HOOK_OPENED", "object": "家の居住権と所有権の食い違い", "evidence": "explicit", "basis": "この家は僕のものだよ。不破氏から僕が買い受けたんだ。君は早々に出て行って呉れ」"},
        ],
    },
    "001798_056779:s3": {
        "time": {
            "relative_order": "continuous",
            "explicit_time": None,
            "elapsed_hint": None,
            "evidence": "explicit",
        },
        "location": {"surface": "板の間"},
        "present_characters": [
            {"id": "boku", "basis": "今度は僕がびっくりして棒立ちになりました"},
            {"id": "noro", "basis": "野呂はそれを受取って調べ始めた"},
        ],
        "character_states": [
            {"character_id": "boku", "values_6d": [0, 1, 0, 0, 0, 0], "evidence": "explicit", "basis": "僕だってちゃんと権利金を払い、きちんと賃貸借の契約をしたんだぞ"},
            {"character_id": "noro", "values_6d": [0, 0, -1, 0, 0, -1], "evidence": "inferred", "basis": "彼はがっくりと板の間に腰をおろし、嘆息するように言いました。「何か誤解がある。まるでこの世には誤解が充ち満ちているようだ」"},
        ],
        "fact_updates": [
            {"subject": "noro", "predicate": "LEARNED", "object": "僕が権利金を払い賃貸借契約をしたこと", "evidence": "explicit", "basis": "領収証と契約書を急いで取出し、引返して野呂の前につきつけてやりました。野呂はそれを受取って調べ始めた"},
            {"subject": "noro", "predicate": "STATE_CHANGED", "object": "不安と困惑の色が拡がり腰をおろした", "evidence": "explicit", "basis": "やがて不安と困惑の色が彼の顔いっぱいに拡がって来たようでした"},
            {"subject": None, "predicate": "HOOK_OPENED", "object": "野呂が家を買い受けた事情と誤解の真相", "evidence": "explicit", "basis": "「君が買い受けたってのは、どんな事情だね」"},
        ],
    },
    "001801_001786:s0": {
        "time": {
            "relative_order": "unknown",
            "explicit_time": "深夜",
            "elapsed_hint": None,
            "evidence": "explicit",
        },
        "location": {"surface": "海"},
        "present_characters": [
            {"id": "shinichi", "basis": "慎一は、知らぬまにかなり沖にきていた"},
            {"id": "onna", "basis": "まちがいなく若い女がひとり、深夜の海を泳いでいるのだった"},
        ],
        "character_states": [
            {"character_id": "shinichi", "values_6d": [0, 0, 0, 1, 0, 0], "evidence": "explicit", "basis": "「……あぶないですよ、この海は。すぐうねりが変わるんです。もっと岸の近くで……」"},
            {"character_id": "onna", "values_6d": [-1, 0, 0, 0, 0, 0], "evidence": "inferred", "basis": "「いいの。ほっといてよ」"},
        ],
        "fact_updates": [
            {"subject": "shinichi", "predicate": "LEARNED", "object": "若い女がひどく疲れ喘ぎながら泳いでいること", "evidence": "explicit", "basis": "はじめて慎一は気づいた。女の声はひどく疲れ、喘いでいた"},
            {"subject": "shinichi", "predicate": "LEARNED", "object": "女が今日真赤なスポーツ・カーで来た女であること", "evidence": "explicit", "basis": "今日、真赤なスポーツ・カーにひとりで乗ってきた女だった"},
            {"subject": "onna", "predicate": "LEARNED", "object": "慎一がこの町の出身で親父が漁師だったこと", "evidence": "explicit", "basis": "「……親父が漁師でした」"},
            {"subject": "onna", "predicate": "HOOK_OPENED", "object": "深夜の沖で助けを拒んで泳ぎ続ける理由", "evidence": "inferred", "basis": "「いいの。ほっといてよ」"},
        ],
    },
    "001801_001786:s1": {
        "time": {
            "relative_order": "continuous",
            "explicit_time": None,
            "elapsed_hint": None,
            "evidence": "explicit",
        },
        "location": {"surface": None},
        "present_characters": [
            {"id": "shinichi", "basis": "あわてて、慎一はいった"},
            {"id": "onna", "basis": "女は、ヒステリックにいった"},
        ],
        "character_states": [
            {"character_id": "shinichi", "values_6d": [0, 0, 0, 1, -1, 1], "evidence": "inferred", "basis": "その人を好きなように死なしてやるほうが、ずっと親切だし、ほんとうは、ずっと勇気のいることなんだ"},
            {"character_id": "onna", "values_6d": [-1, 0, 0, 0, 1, 0], "evidence": "explicit", "basis": "女は、ヒステリックにいった"},
        ],
        "fact_updates": [
            {"subject": "onna", "predicate": "LEARNED", "object": "慎一の親父が去年死んだこと", "evidence": "explicit", "basis": "「死にました。去年」"},
            {"subject": "shinichi", "predicate": "LEARNED", "object": "女が自殺するつもりであること", "evidence": "inferred", "basis": "「……あなたは、自殺するつもりですか？」"},
            {"subject": "onna", "predicate": "LEARNED", "object": "慎一が自殺をとめないという考え", "evidence": "explicit", "basis": "「べつに、やめなさい、っていうつもりじゃないんですよ」"},
            {"subject": "onna", "predicate": "STATE_CHANGED", "object": "ヒステリックに反発したのち無言になる", "evidence": "explicit", "basis": "女は、ヒステリックにいった"},
        ],
    },
    "001801_001786:s2": {
        "time": {
            "relative_order": "continuous",
            "explicit_time": "真暗な夜",
            "elapsed_hint": None,
            "evidence": "explicit",
        },
        "location": {"surface": "海"},
        "present_characters": [
            {"id": "shinichi", "basis": "彼は岸に顔を向けた。そのままゆっくりと引きかえした"},
            {"id": "onna", "basis": "女は怒ったような目つきで、海をみつめていた"},
        ],
        "character_states": [
            {"character_id": "shinichi", "values_6d": [0, 0, 0, 0, -1, 0], "evidence": "explicit", "basis": "彼は岸に顔を向けた。そのままゆっくりと引きかえした"},
            {"character_id": "onna", "values_6d": [-1, 0, 0, 0, 1, 0], "evidence": "explicit", "basis": "女は怒ったような目つきで、海をみつめていた"},
        ],
        "fact_updates": [
            {"subject": "onna", "predicate": "LEARNED", "object": "慎一の親父もこの沖で自殺したこと", "evidence": "explicit", "basis": "ぼくの親父も、自殺したんです。背骨を打ってもう漁ができなくなって、この沖で銛をからだに結えつけてとびこんじゃったんです"},
            {"subject": "shinichi", "predicate": "MOVED_TO", "object": "岸", "evidence": "explicit", "basis": "彼は岸に顔を向けた。そのままゆっくりと引きかえした"},
        ],
    },
    "001875_059897:s0": {
        "time": {
            "relative_order": "unknown",
            "explicit_time": "斎木少年がきて一か月目ぐらいの時",
            "elapsed_hint": None,
            "evidence": "explicit",
        },
        "location": {"surface": "安江の家"},
        "present_characters": [
            {"id": "yasue", "basis": "その日音枝とふたりで赤ん坊をつれにいった安江が、三人になって帰宅すると"},
            {"id": "otoe", "basis": "その日音枝とふたりで赤ん坊をつれにいった安江が、三人になって帰宅すると"},
            {"id": "heiroku", "basis": "思いがけない兵六さんがきて、待っていた"},
            {"id": "susumu", "basis": "やむを得ず居すわりの形になった斎木少年に留守をたのんで"},
        ],
        "character_states": [
            {"character_id": "yasue", "values_6d": [0, 1, 0, 1, 0, 0], "evidence": "explicit", "basis": "戦時中に安江たちが仲人で結婚した若いふうふが、ふたりとも肺病で寝こんでいるのを見かねて、あずかった子供だった"},
            {"character_id": "heiroku", "values_6d": [0, -1, -1, 0, -1, 0], "evidence": "inferred", "basis": "その虚をつきでもするように、兵六さんはいんぎんに両手をつき"},
        ],
        "fact_updates": [
            {"subject": "yasue", "predicate": "ACQUIRED", "object": "夏樹という赤ん坊をあずかる", "evidence": "explicit", "basis": "ふたりとも肺病で寝こんでいるのを見かねて、あずかった子供だった"},
            {"subject": "heiroku", "predicate": "MOVED_TO", "object": "安江の家", "evidence": "explicit", "basis": "思いがけない兵六さんがきて、待っていた"},
            {"subject": "heiroku", "predicate": "LEARNED", "object": "進君から赤ん坊の事情", "evidence": "explicit", "basis": "進君から話を聞きましてな、今、あんたの物語を一席していたとこですよ"},
        ],
    },
    "001875_059897:s1": {
        "time": {
            "relative_order": "continuous",
            "explicit_time": None,
            "elapsed_hint": None,
            "evidence": "explicit",
        },
        "location": {"surface": None},
        "present_characters": [
            {"id": "yasue", "basis": "恨みがましさも忘れて安江はつい許す気になり"},
            {"id": "heiroku", "basis": "そしてほんとにそのあとしばらく、兵六さんはせっせとミルクを運んできてくれた"},
            {"id": "otoe", "basis": "音枝がそういうと、文吉は文吉で"},
            {"id": "bunkichi", "basis": "にやっとする文吉に、安江はくってかかる調子で"},
        ],
        "character_states": [
            {"character_id": "yasue", "values_6d": [0, 0, 0, 1, 1, 0], "evidence": "explicit", "basis": "夫と進君との間を円滑にしてゆきたい安江の気弱さでもあった"},
            {"character_id": "heiroku", "values_6d": [0, -1, -1, -1, -1, 0], "evidence": "explicit", "basis": "そういう事情をさとった兵六さんはうまくそれを利用し、文吉のいる前ではミルク代も大っぴらにうけとりながら、恩きせがましいことまでいったものだ"},
            {"character_id": "bunkichi", "values_6d": [0, 1, 0, 0, -1, 0], "evidence": "explicit", "basis": "「かしたさ。お前の姉婿だった人だと思ったからさ。」"},
            {"character_id": "otoe", "values_6d": [0, 0, 0, 0, 0, 0], "evidence": "inferred", "basis": "「警察につかまったりして、はずかしいのかしら。」"},
        ],
        "fact_updates": [
            {"subject": "yasue", "predicate": "STATE_CHANGED", "object": "恨みがましさも忘れて兵六を許す気になる", "evidence": "explicit", "basis": "恨みがましさも忘れて安江はつい許す気になり"},
            {"subject": "heiroku", "predicate": "STATE_CHANGED", "object": "食糧統制違反で新聞に出てぱったり来なくなった", "evidence": "explicit", "basis": "まもなく食糧統制違反で新聞にも出るなどやみ屋なりのはなやかさでうわさにのぼるようになって以来、兵六さんの足はだんだん遠のき、やがてぱったり来なくなってしまった"},
            {"subject": "heiroku", "predicate": "ACQUIRED", "object": "文吉から二千円", "evidence": "explicit", "basis": "だって、二千円ぐらいの金を、かしてくれっていったもんな"},
            {"subject": "yasue", "predicate": "LEARNED", "object": "文吉がへそくりから兵六に二千円貸したこと", "evidence": "explicit", "basis": "安江が顔色をかえる"},
            {"subject": "heiroku", "predicate": "LEARNED", "object": "安江が権利金未受領を夫に隠している事情", "evidence": "explicit", "basis": "そういう事情をさとった兵六さんはうまくそれを利用し"},
            {"subject": "yasue", "predicate": "HOOK_OPENED", "object": "斎木少年の権利金を夫に隠していること", "evidence": "explicit", "basis": "安江だって斎木少年のへやの権利金のことはいまだに夫にかくしている"},
        ],
    },
    "001875_059897:s2": {
        "time": {
            "relative_order": "continuous",
            "explicit_time": None,
            "elapsed_hint": None,
            "evidence": "inferred",
        },
        "location": {"surface": None},
        "present_characters": [
            {"id": "yasue", "basis": "音枝にはぽんぽんいうくせに、当の進君が頭を下げると、安江は柔和な顔で五百円をわたしながら"},
            {"id": "otoe", "basis": "音枝は素直に引きさがるのだが、やがてまた安江の顔色をうかがうようにして"},
            {"id": "susumu", "basis": "当の進君が頭を下げると、安江は柔和な顔で五百円をわたしながら"},
        ],
        "character_states": [
            {"character_id": "yasue", "values_6d": [0, 0, 0, 1, 1, 0], "evidence": "explicit", "basis": "音枝にはぽんぽんいうくせに、当の進君が頭を下げると、安江は柔和な顔で五百円をわたしながら"},
            {"character_id": "otoe", "values_6d": [0, 1, 0, 1, 0, 0], "evidence": "explicit", "basis": "でも、かわいそうみたいよ"},
            {"character_id": "bunkichi", "values_6d": [0, 0, -1, 0, -1, 0], "evidence": "explicit", "basis": "だが文吉は文吉で別のむな算用をしていた"},
            {"character_id": "susumu", "values_6d": [0, 1, 0, 0, 0, 0], "evidence": "inferred", "basis": "当の進君が頭を下げると"},
        ],
        "fact_updates": [
            {"subject": "yasue", "predicate": "LEARNED", "object": "進君の家が兵六に生活費を預けたが渡してもらえないこと", "evidence": "explicit", "basis": "進君の何年間かの生活費なんですって。月々わたしてもらう約束なのにさ、兵六おじさん、なかなかくれないらしいわ"},
            {"subject": "susumu", "predicate": "ACQUIRED", "object": "五百円", "evidence": "explicit", "basis": "当の進君が頭を下げると、安江は柔和な顔で五百円をわたしながら"},
            {"subject": "yasue", "predicate": "STATE_CHANGED", "object": "柔和な顔になり五百円を貸した", "evidence": "explicit", "basis": "音枝にはぽんぽんいうくせに、当の進君が頭を下げると、安江は柔和な顔で五百円をわたしながら"},
            {"subject": "heiroku", "predicate": "HOOK_OPENED", "object": "兵六が進君の生活費を渡さず会えない問題", "evidence": "explicit", "basis": "月々わたしてもらう約束なのにさ、兵六おじさん、なかなかくれないらしいわ"},
        ],
    },
    "001875_059897:s3": {
        "time": {
            "relative_order": "continuous",
            "explicit_time": "きょう",
            "elapsed_hint": None,
            "evidence": "explicit",
        },
        "location": {"surface": None},
        "present_characters": [
            {"id": "yasue", "basis": "安江はつい慰めずにいられなくなり"},
            {"id": "susumu", "basis": "ぼく、ここのところ、毎日のように行ってるんです"},
        ],
        "character_states": [
            {"character_id": "yasue", "values_6d": [0, 0, 0, 1, 1, 0], "evidence": "explicit", "basis": "安江はつい慰めずにいられなくなり"},
            {"character_id": "susumu", "values_6d": [0, 1, 0, 0, 0, 1], "evidence": "explicit", "basis": "今時の若い人に珍らしく、両手をついての切口上"},
        ],
        "fact_updates": [
            {"subject": "yasue", "predicate": "LEARNED", "object": "進君が毎日兵六を訪ねており今日も会いに行くつもりであること", "evidence": "explicit", "basis": "ぼく、ここのところ、毎日のように行ってるんです。きょうもこれから出かけるところです。きょうは泊りこんででもあってくるつもりです"},
            {"subject": "yasue", "predicate": "STATE_CHANGED", "object": "進君を慰める気持ちになり日ぎりを緩めた", "evidence": "explicit", "basis": "安江はつい慰めずにいられなくなり"},
            {"subject": "susumu", "predicate": "HOOK_OPENED", "object": "兵六に相談してことしうちぐらいに部屋をさがすこと", "evidence": "explicit", "basis": "ま、ここまできて日ぎりはしませんけどね、ことしうちぐらいには、さがしてくださいね"},
        ],
    },
}

PERSON_VALUES_6D: dict[str, dict[str, Any]] = {
    "001798_056779:boku":     {"values_6d": [0,  1,  0,  0,  0,  0], "rationale": "契約書・領収証・間代前納などの正規手続きを重視し、礼儀・規範意識に沿って行動する。"},
    "001798_056779:fuwa":     {"values_6d": [0, -1, -1, -1, -1,  0], "rationale": "権利金4万円＋前納2千円を得て夜逃げ同然に失踪し、二重契約で利得を図る打算的・逸脱的人物。"},
    "001798_056779:noro":     {"values_6d": [0,  0, -1, -1,  0,  0], "rationale": "家屋購入の所有権を前面に出し、当初は横柄に立ち退きを迫る世俗的・自己中心的振る舞い。"},
    "001801_001786:shinichi": {"values_6d": [0,  0,  0,  1, -1,  0], "rationale": "見知らぬ女を気遣いつつも（親切・尊重）、感情的介入を抑えて静かに引き返す強い抑制を持つ。"},
    "001801_001786:onna":     {"values_6d": [-1, 0,  0,  0,  1,  0], "rationale": "深夜の沖で自殺を意図し助けを拒む死への傾斜と、ヒステリック・怒りに揺れる情念の強さ。"},
    "001875_059897:yasue":    {"values_6d": [0,  0,  0,  1,  1,  0], "rationale": "肺病の夫婦の赤ん坊を引き取り、進君にも五百円を貸し慰める献身・庇護と情に脆い性格。"},
    "001875_059897:heiroku":  {"values_6d": [0, -1, -1, -1, -1,  0], "rationale": "食糧統制違反のやみ屋で、安江の弱みや進君の生活費を打算的に利用・不履行する利己的逸脱者。"},
    "001875_059897:otoe":     {"values_6d": [0,  1,  0,  1,  0,  0], "rationale": "主婦代りに赤ん坊の世話をし母に素直に従う規範性と、進君を哀れみ取り成す思いやり。"},
    "001875_059897:bunkichi": {"values_6d": [0,  1, -1,  0, -1,  0], "rationale": "義理で金を貸す一方、権利金と相殺するむな算用や冷静な人物評を示す世俗的・抑制的気質。"},
    "001875_059897:susumu":   {"values_6d": [0,  1,  0,  0,  0,  1], "rationale": "両手をついての切口上という礼儀正しさと、毎日訪ね泊りこんででも会うとする主体的な行動力。"},
}


def segment(txt_path: Path) -> dict[str, Any]:
    out = subprocess.run([NISHIKI, "segment", str(txt_path), "--format", "json"], capture_output=True, check=True)
    return json.loads(out.stdout.decode("utf-8"))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    pr_hash = scene_prompt_hash()

    total_scenes = 0
    total_rejected = 0
    total_schema_errs = 0
    traces_and_audit: list[dict[str, Any]] = []

    for wid, pov in WORKS:
        txt_path = TEXT_DIR / f"{wid}.txt"
        roster_path = ROSTER_DIR / f"{wid}.characters.json"
        roster = json.loads(roster_path.read_text(encoding="utf-8"))
        roster_ids = [c["id"] for c in roster]

        seg = segment(txt_path)
        scenes = seg["scenes"]
        assert (wid, len(scenes)) in {
            ("001798_056779", 4),
            ("001801_001786", 3),
            ("001875_059897", 4),
        }, (wid, len(scenes))

        work_anns: list[dict[str, Any]] = []
        for si, sc in enumerate(scenes):
            sid = f"{wid}:s{si}"
            raw = RAW_ANNOTATIONS[sid]
            scene_text = sc["scene_text"]
            cleaned, audit = validate_scene_output(raw, roster_ids, scene_text)
            sec = cleaned["sections"]
            trace = cleaned["trace"]

            assert audit["parse_ok"] is True, (sid, audit)
            if audit["rejected"]:
                print(f"REJECTED in {sid}: {audit['rejected']}", file=sys.stderr)
            total_rejected += len(audit["rejected"])

            ann = {
                "contract": {
                    "schema_version": "sensor.v2",
                    "splitter_version": seg["splitter"],
                    "scene_id": sid,
                    "scene_hash": sc["scene_hash"],
                    "ledger_hash": "blind-gold-audit-v1",
                    "model_id": "google-gemini-independent-audit",
                    "prompt_hash": pr_hash,
                    "work_id": wid,
                    "scene_index": si,
                    "scene_count": len(scenes),
                    "char_len": sc["char_len"],
                    "seed": 0,
                },
                "scene_meta": {
                    "time": sec["time"],
                    "location_id": sec["location_id"],
                    "present_character_ids": sec["present_character_ids"],
                    "pov_character": pov,
                },
                "utterances": [],
                "character_states": sec["character_states"],
                "fact_updates": sec["fact_updates"],
            }
            errs = validate_annotation_schema(ann, schema, roster_ids)
            if errs:
                print(f"SCHEMA ERRORS in {sid}: {errs}", file=sys.stderr)
            total_schema_errs += len(errs)

            audit["scene_id"] = sid
            audit["schema_errors"] = errs
            work_anns.append(ann)
            traces_and_audit.append({"scene_id": sid, "trace": trace, "audit": audit})
            total_scenes += 1
            print(f"{sid}: kept={audit['kept']} rejected={len(audit['rejected'])} schema_errors={len(errs)}")

        (OUT_DIR / f"{wid}.annotations.json").write_text(dumps(work_anns), encoding="utf-8")

    (OUT_DIR / "person_values_6d.json").write_text(dumps(PERSON_VALUES_6D), encoding="utf-8")
    (OUT_DIR / "traces_and_audit.json").write_text(dumps(traces_and_audit), encoding="utf-8")

    print(f"SUMMARY: scenes={total_scenes} total_rejected={total_rejected} total_schema_errors={total_schema_errs}")
    if total_rejected != 0 or total_schema_errs != 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
