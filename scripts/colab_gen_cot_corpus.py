"""Phase 18 M18.2: generate a Japanese chain-of-thought corpus on Colab for cot_reasoning QDA calibration.

Self-contained (no repo code needed on Colab). Run:
    python scripts/run_colab_file.py scripts/colab_gen_cot_corpus.py artifacts/qda_calibration --gpu L4 --timeout 2400
Output JSON: {"model", "seed", "problems": [...], "outputs": [...], "elapsed_s"}.
Problems are written for this repo (no external dataset). Greedy decoding, so the corpus is reproducible
up to library/GPU numerics.
"""
import base64, gzip, json, sys, time, zlib, subprocess

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "accelerate"], check=False)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-3B-Instruct"
PROBLEMS = [
    "りんごが3個入った袋が7袋あります。そのうち2袋を友人に渡しました。残りのりんごは何個ですか。",
    "時速48kmで2時間30分走ると、何km進みますか。",
    "ある数に5を足して3倍すると36になります。ある数はいくつですか。",
    "1から50までの整数のうち、3でも5でも割り切れない数はいくつありますか。",
    "Aさんの年齢はBさんの2倍です。6年後、Aさんの年齢はBさんの1.5倍になります。今のBさんは何歳ですか。",
    "定価2400円の品物を2割引きで買い、さらにその価格から300円引いてもらいました。支払いはいくらですか。",
    "鶴と亀が合わせて20匹いて、足の数は合計56本です。亀は何匹ですか。",
    "3人が1回ずつ握手をすると握手は何回ですか。では8人ではどうですか。",
    "水槽に毎分12リットル入れる管と毎分4リットル抜く管があります。240リットルの水槽が空から満杯になるまで何分かかりますか。",
    "連続する3つの整数の和が84です。最も大きい数はいくつですか。",
    "全員が嘘つきか正直者のどちらかの村で、Aは「Bは嘘つきだ」と言い、Bは「AとCは同じ種類だ」と言いました。Cが正直者のとき、AとBはそれぞれどちらですか。",
    "赤・青・白の3つの箱があり、1つにだけ景品が入っています。赤には「景品はここにない」、青には「景品は白にある」、白には「景品はここにない」と書かれ、正しい札は1枚だけです。景品はどこですか。",
    "月曜日の100日後は何曜日ですか。",
    "2の10乗を7で割った余りを求めてください。",
    "長方形の周の長さが36cmで、縦は横より4cm短いです。面積を求めてください。",
    "ある仕事をAは12日、Bは18日で終えます。2人で一緒に行うと何日で終わりますか。",
    "4枚のカード1,2,3,4から2枚を並べて2桁の数を作るとき、偶数は何通りできますか。",
    "サイコロを2回振って、出た目の和が7になる確率を求めてください。",
    "ある商品を原価の3割の利益を見込んで定価をつけましたが、売れないので定価の1割引きで売ったところ利益は170円でした。原価はいくらですか。",
    "兄は分速80m、弟は分速60mで同じ地点から同じ方向に歩きます。弟が出発して5分後に兄が出発すると、兄は何分後に弟に追いつきますか。",
    "濃度8%の食塩水300gに水を何g加えると濃度6%になりますか。",
    "1から100までの整数の和を求め、その求め方も説明してください。",
    "正方形の対角線の長さが10cmのとき、面積はいくらですか。",
    "A,B,C,Dの4人が一列に並ぶとき、AとBが隣り合う並び方は何通りですか。",
    "すべての猫は動物であり、ある動物は泳げます。このとき「ある猫は泳げる」は必ず正しいと言えますか。",
    "雨が降れば地面は濡れます。地面が濡れています。雨が降ったと結論してよいですか。理由も述べてください。",
    "時計が3時ちょうどを指すとき、長針と短針のなす角は何度ですか。3時30分ではどうですか。",
    "数列2, 6, 12, 20, 30の次の数は何ですか。規則も説明してください。",
    "3で割ると2余り、5で割ると3余る最小の正の整数は何ですか。",
    "ある本を1日目に全体の1/3、2日目に残りの1/4を読み、残りは60ページでした。本は全部で何ページですか。",
    "毎年10%ずつ増える量が、何年後に初めて最初の2倍を超えますか。",
    "5人の平均点が72点で、1人加わると平均点が74点になりました。加わった人の点数は何点ですか。",
    "縦3cm、横4cm、高さ12cmの直方体の対角線の長さを求めてください。",
    "12と18の最大公約数と最小公倍数を求めてください。",
    "箱の中に赤玉3個と白玉2個があります。同時に2個取り出すとき、2個とも赤である確率を求めてください。",
    "列車が長さ300mの鉄橋を渡り始めてから渡り終わるまでに20秒、長さ500mのトンネルに入り始めてから出終わるまでに30秒かかります。列車の長さと速さを求めてください。",
    "AはBより背が高く、CはAより背が高く、DはBより背が低いです。2番目に背が高いのは誰ですか。",
    "1000円で1個80円のパンと1個120円の牛乳を合わせて10個買い、おつりが0円になる組み合わせはありますか。",
    "ある整数を2乗すると、一の位が6になりました。元の整数の一の位として考えられる数をすべて挙げてください。",
    "池の周りを、AとBが同じ地点から反対方向に歩くと12分後に出会い、同じ方向に歩くと60分後にAがBに1周差をつけます。Aの速さはBの何倍ですか。",
]
SYS = "あなたは丁寧な数学・論理の家庭教師です。答えを出す前に、考え方を日本語の文章で段階的に説明してください。箇条書きや数式記号の多用は避け、地の文で推論を書いてください。"

torch.manual_seed(0)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
outs = []
for i, q in enumerate(PROBLEMS):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": q}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
    with torch.no_grad():
        g = model.generate(ids, max_new_tokens=600, do_sample=False, repetition_penalty=1.05)
    text = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)
    outs.append(text)
    print(f"[{i}] {len(text)} chars", flush=True)
res = {"model": MODEL, "seed": 0, "decoding": "greedy, max_new_tokens=600, repetition_penalty=1.05",
       "system": SYS, "problems": PROBLEMS, "outputs": outs, "elapsed_s": time.time() - t0}
raw = json.dumps(res, ensure_ascii=False).encode("utf-8")
print("ARCHIVE_B64", format(zlib.crc32(raw) & 0xFFFFFFFF, "08x"), base64.b64encode(gzip.compress(raw)).decode())
