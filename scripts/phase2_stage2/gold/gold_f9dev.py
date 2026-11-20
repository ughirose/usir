# Stage B gold for 4 F9 Dev works (24 scenes). Written by the session model by reading the scene text directly,
# before opening any sensor (qwen) annotation output.
from gold_modern import S, F, G  # noqa: F401  (shares the registry G)

w = "000035_000301"
S(f"{w}:s0", "unknown", ["二年目の十一月"], ["下宿"], ["yozo"],
  {"yozo":[-1,-1,0,0,0,-1]},
  [F("yozo","ACQUIRED","行動隊長の地位"), F("yozo","STATE_CHANGED","情死事件で身の上が一変した"), F("yozo","STATE_CHANGED","運動から逃げ、死ぬ事にした")])
S(f"{w}:s1", "parallel", ["その頃"], ["仙遊館","自分の部屋"], ["yozo","musume"],
  {"yozo":[0,0,0,0,-1,0], "musume":[0,0,0,0,1,0]}, [])
S(f"{w}:s2", "parallel", ["或る夏の夜","同じ頃"], ["銀座","大カフエ","ビルの事務所"], ["yozo","doshi","tsuneko"],
  {"yozo":[0,0,0,-1,0,-1], "doshi":[0,0,0,0,1,0]},
  [F("yozo","MOVED_TO","銀座の大カフエ"), F("doshi","STATE_CHANGED","狂乱の如く興奮した"), F("yozo","ACQUIRED","女給の恩")])
S(f"{w}:s3", "continuous", ["秋の、寒い夜"], ["銀座裏","屋台のお鮨や","屋台"], ["yozo","tsuneko"],
  {"yozo":[0,0,0,0,0,0], "tsuneko":[0,0,0,1,0,0]},
  [F("yozo","STATE_CHANGED","震えおののく心が静まった"), F("yozo","MOVED_TO","銀座裏の屋台の鮨や")])
S(f"{w}:s4", "continuous", [], ["本所の大工さんの二階","二階"], ["yozo","tsuneko"],
  {"yozo":[0,0,0,0,0,0], "tsuneko":[0,0,0,0,0,-1]},
  [F("yozo","MOVED_TO","本所の二階"), F("yozo","LEARNED","ツネ子の身の上（広島、夫は詐欺罪で服役中）"), F("yozo","STATE_CHANGED","恐怖と不安から離れる")])
S(f"{w}:s5", "continuous", ["朝"], [], ["yozo","tsuneko"],
  {"yozo":[0,0,0,0,-1,0]},
  [F("yozo","STATE_CHANGED","もとのお道化者に戻る"), F("yozo","MOVED_TO","引上げ"), F(None,"HOOK_OPENED","「金の切れめが縁の切れめ」の放言")])
S(f"{w}:s6", "continuous", ["十一月の末","ひとつき"], ["神田","屋台"], ["yozo","horiki"],
  {"yozo":[0,0,0,0,1,0], "horiki":[0,0,0,0,1,0]},
  [F(None,"HOOK_OPENED","カフエ行きの約束"), F("yozo","STATE_CHANGED","酔って大胆になる")])

w = "001383_056643"
S(f"{w}:s0", "unknown", [], [], ["hagiwara","shinichi","senkichi"],
  {"shinichi":[0,-1,0,-1,1,0], "senkichi":[0,0,0,0,0,-1], "hagiwara":[0,0,0,0,1,0]},
  [F("senkichi","STATE_CHANGED","縛られ額に炭を塗られる"), F("hagiwara","STATE_CHANGED","不思議な快感に襲われる")])
S(f"{w}:s1", "continuous", ["暫くしてから"], ["物置小屋","お堂"], ["hagiwara","shinichi","senkichi"],
  {"shinichi":[0,-1,0,-1,1,0], "senkichi":[0,0,0,0,0,-1], "hagiwara":[0,0,0,0,0,-1]},
  [F("senkichi","STATE_CHANGED","遊戯で狼に喰い殺される")])
S(f"{w}:s2", "continuous", [], ["土間"], ["hagiwara","shinichi","senkichi"],
  {"shinichi":[0,-1,0,-1,1,0], "hagiwara":[0,0,0,0,0,-1], "senkichi":[0,0,0,0,0,-1]},
  [F("hagiwara","STATE_CHANGED","遊戯で狼に喰い殺される"), F("senkichi","STATE_CHANGED","体中泥だらけになる")])
S(f"{w}:s3", "continuous", [], [], ["hagiwara","shinichi","senkichi"],
  {"hagiwara":[0,0,0,0,1,-1], "shinichi":[0,-1,0,-1,1,0], "senkichi":[0,0,0,0,0,-1]},
  [F("hagiwara","STATE_CHANGED","恐怖が快感に変わり信一の傀儡となることを喜ぶ")])
S(f"{w}:s4", "continuous", ["夕靄"], ["小屋","門"], ["hagiwara","shinichi","senkichi","jochu"],
  {"jochu":[0,1,0,0,0,0], "hagiwara":[0,0,0,0,1,-1], "shinichi":[0,-1,0,-1,0,0]},
  [F("jochu","MOVED_TO","小屋"), F("hagiwara","MOVED_TO","家"), F("hagiwara","STATE_CHANGED","信一に心を奪われる")])
S(f"{w}:s5", "continuous", ["明くる日","四五日立った或る日"], ["学校","運動場","中二階の十畳の間"], ["hagiwara","shinichi","senkichi","jochu","mitsuko"],
  {"shinichi":[0,-1,0,-1,1,0], "mitsuko":[0,-1,0,0,0,0]},
  [F("hagiwara","MOVED_TO","信一の家"), F("shinichi","STATE_CHANGED","学校ではいつもの意気地なしに戻る"), F(None,"HOOK_OPENED","仙吉を白酒で酔わせる企み")])

w = "000050_003586"
S(f"{w}:s0", "unknown", ["今日"], [], ["tabiso","uba"],
  {"tabiso":[0,0,1,0,0,0], "uba":[0,1,0,0,0,0]},
  [F("tabiso","MOVED_TO","鎌倉"), F("tabiso","LEARNED","嘉吉の話")])
S(f"{w}:s1", "continuous", [], ["お店"], ["tabiso","uba"],
  {"tabiso":[0,0,1,1,0,0], "uba":[0,0,1,0,0,0]},
  [F("tabiso","STATE_CHANGED","念仏の声が出る"), F(None,"HOOK_OPENED","黒門の別宅での回向の依頼")])
S(f"{w}:s2", "unknown", ["夜"], ["小川べり","黒門","秋谷"], ["niemon","saihachi","kundo"],
  {"kundo":[0,1,0,0,-1,0], "saihachi":[0,0,0,0,1,0]},
  [F(None,"HOOK_OPENED","手毬と旅の書生をめぐる出来事")])
S(f"{w}:s3", "continuous", [], ["川"], ["niemon","saihachi","kundo"],
  {"saihachi":[0,0,0,0,1,0]},
  [F(None,"HOOK_OPENED","手毬の下の猫")])
S(f"{w}:s4", "continuous", [], [], ["niemon","saihachi","kundo"],
  {"kundo":[0,1,0,0,-1,0]},
  [F("kundo","LEARNED","猫は死骸だったこと"), F(None,"HOOK_RESOLVED","手毬の下の猫の正体")])

w = "000081_000456"
S(f"{w}:s0", "unknown", [], ["車のなか","車室","銀河"], ["giovanni","campanella","tabibito"],
  {"giovanni":[0,0,1,0,0,0], "campanella":[0,0,1,1,0,1]},
  [F("campanella","STATE_CHANGED","決心している")])
S(f"{w}:s1", "continuous", ["十一時"], ["白鳥停車場","停車場","河原"], ["giovanni","campanella"],
  {"giovanni":[0,0,0,0,0,1], "campanella":[0,0,1,0,0,0]},
  [F("giovanni","MOVED_TO","白鳥停車場"), F("giovanni","MOVED_TO","河原"), F("campanella","MOVED_TO","河原")])
S(f"{w}:s2", "continuous", [], ["プリオシン海岸","河原","渚"], ["giovanni","campanella"],
  {"giovanni":[0,0,0,0,0,1], "campanella":[0,0,0,0,0,1]},
  [F("giovanni","MOVED_TO","プリオシン海岸"), F("campanella","ACQUIRED","くるみの実")])
S(f"{w}:s3", "continuous", [], ["白い岩","岩"], ["giovanni","campanella","daigakushi"],
  {"daigakushi":[0,1,0,0,-1,0]},
  [F("giovanni","LEARNED","くるみが百二十万年前のものであること"), F("giovanni","LEARNED","ボスは牛の先祖であること")])
S(f"{w}:s4", "continuous", [], ["白い岩","河原","車室"], ["giovanni","campanella","daigakushi"],
  {"giovanni":[1,0,0,0,0,1]},
  [F("giovanni","MOVED_TO","車室の席"), F("campanella","MOVED_TO","車室の席")])
S(f"{w}:s5", "continuous", [], ["汽車"], ["giovanni","campanella","toritori"],
  {"giovanni":[0,0,0,0,1,0], "toritori":[0,0,0,1,0,0]},
  [F("toritori","MOVED_TO","二人の席のうしろ"), F("giovanni","STATE_CHANGED","さびしいようなかなしい気持になる")])
