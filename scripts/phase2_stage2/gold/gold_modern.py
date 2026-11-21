# Stage B gold for 3 modern works (11 scenes). Written by the session model by reading the scene text directly,
# before opening any sensor (qwen) annotation output. Conventions: see gold/README.md.
G = {}
def S(sid, order, etime, locs, present, states, facts):
    G[sid] = dict(relative_order=order, explicit_time=etime, location_surfaces=locs, present=present, states=states, facts=facts)
def F(sub, pred, obj): return dict(subject=sub, predicate=pred, object=obj)

w = "001798_056779"
S(f"{w}:s0", "unknown", "三日目", ["代田橋"], ["boku","fuwa"],
  {"boku":[0,1,0,0,0,0], "fuwa":[0,0,0,0,1,0]},
  [F("boku","MOVED_TO","代田橋")])
S(f"{w}:s1", "continuous", "失踪の前の晩", ["僕の部屋"], ["boku","fuwa"],
  {"boku":[0,1,0,0,0,0], "fuwa":[0,-1,0,-1,0,0]},
  [F("fuwa","STATE_CHANGED","失踪"), F("fuwa","ACQUIRED","二千円"), F(None,"HOOK_OPENED","不破の借金の事情")])
S(f"{w}:s2", "continuous", "翌朝", ["板の間","台所","庭"], ["boku","noro"],
  {"boku":[0,1,0,0,0,0], "noro":[0,0,0,-1,0,0]},
  [F("noro","MOVED_TO","この家"), F("noro","LEARNED","不破が出発したこと"), F(None,"HOOK_OPENED","家の所有をめぐる食い違い")])
S(f"{w}:s3", "continuous", None, ["板の間"], ["boku","noro"],
  {"boku":[0,1,0,0,0,0], "noro":[0,0,0,0,0,-1]},
  [F("noro","STATE_CHANGED","不安と困惑"), F(None,"HOOK_OPENED","家の二重契約の誤解")])

w = "001801_001786"
S(f"{w}:s0", "unknown", "深夜", ["海"], ["shinichi","onna"],
  {"shinichi":[1,0,0,1,0,0], "onna":[-1,0,0,0,0,0]},
  [F("shinichi","LEARNED","女が今日スポーツ・カーで来た女であること"), F("shinichi","MOVED_TO","女の方")])
S(f"{w}:s1", "continuous", None, [], ["shinichi","onna"],
  {"shinichi":[0,0,0,1,0,0], "onna":[-1,0,0,0,1,0]},
  [F("shinichi","LOST","父"), F(None,"HOOK_OPENED","女の自殺の意図"), F("onna","LEARNED","慎一の父の死生観")])
S(f"{w}:s2", "continuous", None, ["海","沖"], ["shinichi","onna"],
  {"shinichi":[0,0,0,0,0,-1], "onna":[-1,0,0,0,0,0]},
  [F("onna","LEARNED","慎一の父が自殺したこと"), F("shinichi","MOVED_TO","岸")])

w = "001875_059897"
S(f"{w}:s0", "unknown", "斎木少年がきて一か月目ぐらい", ["安江の家"], ["yasue","otoe","heiroku","susumu"],
  {"yasue":[0,0,0,1,0,0], "heiroku":[0,0,0,-1,0,0]},
  [F("yasue","ACQUIRED","夏樹（赤ん坊）"), F("heiroku","MOVED_TO","安江の家")])
S(f"{w}:s1", "continuous", None, [], ["yasue","heiroku","otoe","bunkichi"],
  {"yasue":[0,0,0,0,1,0], "bunkichi":[0,0,0,-1,0,0], "heiroku":[0,-1,0,-1,0,0]},
  [F("bunkichi","LOST","二千円"), F("heiroku","STATE_CHANGED","食糧統制違反で新聞に出る"), F(None,"HOOK_OPENED","兵六への貸金の行方")])
S(f"{w}:s2", "continuous", None, [], ["yasue","otoe","susumu"],
  {"yasue":[0,0,0,1,0,0], "otoe":[0,0,0,1,0,0], "susumu":[0,1,0,0,0,0]},
  [F("yasue","LEARNED","進君の生活費が兵六に預けられていること"), F("susumu","ACQUIRED","五百円"), F(None,"HOOK_OPENED","兵六が進君の生活費を渡さない")])
S(f"{w}:s3", "continuous", None, [], ["yasue","susumu"],
  {"yasue":[0,0,0,1,0,0], "susumu":[0,1,0,0,0,1]},
  [F(None,"HOOK_OPENED","進君の部屋探し"), F("susumu","MOVED_TO","鷲さんのところ")])
