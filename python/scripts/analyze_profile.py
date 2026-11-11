import json
from collections import defaultdict

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
profile_path = ROOT / 'docs' / 'handoff' / 'v10' / 'nishiki_profile_106works.json'
data = json.load(open(profile_path, encoding='utf-8-sig'))
works = data['works']

print('=== 106 WORKS PROFILE COMPREHENSIVE AUDIT ===')
print(f'Total works: {len(works)}')
total_scenes = sum(w.get('scene_count', 0) for w in works)
print(f'Total scenes: {total_scenes}')

author_stats = defaultdict(lambda: {
    'works': 0, 'scenes': 0,
    'sadame_intra_chars': [], 'sadame_total': [], 'sadame_edges_eff': [],
    'awase_irony': [], 'awase_detach': [],
    'kamae_hooks': [], 'kamae_survival': [], 'kamae_survival_idf': [],
    'nagare_phi_zeami': [], 'nagare_still_runs': []
})

all_sadame_intra_chars = []
all_sadame_total = []
all_sadame_edges_eff = []
all_irony = []
all_detach = []
all_hooks = []
all_survival = []
all_survival_idf = []
all_phi_zeami = []
all_still_runs = []

for w in works:
    aid = w.get('author_id', 'unknown')
    sc = w.get('scene_count', 0)
    sadame = w.get('sadame', {})
    awase = w.get('awase', {})
    kamae = w.get('kamae', {})
    nagare = w.get('nagare', {})
    
    intra_chars = sadame.get('intra_characters', 0)
    sad_tot = sadame.get('total', 0.0)
    edges_eff = sadame.get('edges_eff', 0.0)
    
    irony = awase.get('irony', {}).get('mean', 0.0)
    detach = awase.get('detachment', {}).get('mean', 0.0)
    
    hooks = kamae.get('opened_hooks', 0)
    surv = kamae.get('hook_survival', 0.0)
    surv_idf = kamae.get('hook_survival_idf', 0.0)
    
    phi = nagare.get('phi_zeami', 0.0)
    n_still = len(nagare.get('still_runs', []))
    
    ast = author_stats[aid]
    ast['works'] += 1
    ast['scenes'] += sc
    ast['sadame_intra_chars'].append(intra_chars)
    ast['sadame_total'].append(sad_tot)
    ast['sadame_edges_eff'].append(edges_eff)
    ast['awase_irony'].append(irony)
    ast['awase_detach'].append(detach)
    ast['kamae_hooks'].append(hooks)
    ast['kamae_survival'].append(surv)
    ast['kamae_survival_idf'].append(surv_idf)
    ast['nagare_phi_zeami'].append(phi)
    ast['nagare_still_runs'].append(n_still)
    
    all_sadame_intra_chars.append(intra_chars)
    all_sadame_total.append(sad_tot)
    all_sadame_edges_eff.append(edges_eff)
    all_irony.append(irony)
    all_detach.append(detach)
    all_hooks.append(hooks)
    all_survival.append(surv)
    all_survival_idf.append(surv_idf)
    all_phi_zeami.append(phi)
    all_still_runs.append(n_still)

def mean(lst):
    return sum(lst) / len(lst) if lst else 0.0

print(f'\n--- GLOBAL AGGREGATE (106 works, {total_scenes} scenes) ---')
print(f'Sadame intra_characters: mean={mean(all_sadame_intra_chars):.2f}, min={min(all_sadame_intra_chars)}, max={max(all_sadame_intra_chars)}, works>0={sum(1 for x in all_sadame_intra_chars if x > 0)}/106')
print(f'Sadame total (polyphony/intra): mean={mean(all_sadame_total):.4f}, min={min(all_sadame_total):.4f}, max={max(all_sadame_total):.4f}, works>0={sum(1 for x in all_sadame_total if x > 0)}/106')
print(f'Sadame edges_eff: mean={mean(all_sadame_edges_eff):.2f}, min={min(all_sadame_edges_eff):.2f}, max={max(all_sadame_edges_eff):.2f}, works>0={sum(1 for x in all_sadame_edges_eff if x > 0)}/106')
print(f'Awase irony mean: global_mean={mean(all_irony):.4f}, works>0={sum(1 for x in all_irony if x > 0)}/106 ({sum(1 for x in all_irony if x > 0)/106*100:.1f}%)')
print(f'Awase detachment mean: global_mean={mean(all_detach):.4f}, works>0={sum(1 for x in all_detach if x > 0)}/106 ({sum(1 for x in all_detach if x > 0)/106*100:.1f}%)')
print(f'Kamae opened_hooks: total={sum(all_hooks)}, mean={mean(all_hooks):.1f}, works>0={sum(1 for x in all_hooks if x > 0)}/106')
print(f'Kamae hook_survival: mean={mean(all_survival):.4f}, min={min(all_survival):.4f}, max={max(all_survival):.4f}')
print(f'Kamae hook_survival_idf: mean={mean(all_survival_idf):.4f}, min={min(all_survival_idf):.4f}, max={max(all_survival_idf):.4f}')
print(f'Nagare phi_zeami: mean={mean(all_phi_zeami):.4f}, min={min(all_phi_zeami):.4f}, max={max(all_phi_zeami):.4f}')
print(f'Nagare still_runs: total={sum(all_still_runs)}, mean={mean(all_still_runs):.2f}, works>0={sum(1 for x in all_still_runs if x > 0)}/106')

author_names = {
    '000148': '夏目漱石',
    '000879': '芥川龍之介',
    '000035': '太宰治',
    '000081': '宮沢賢治',
    '000074': '梶井基次郎',
    '000129': '森鴎外',
    '000050': '泉鏡花',
    '001383': '谷崎潤一郎',
    '001095': '坂口安吾',
    '000119': '中島敦',
}

print('\n--- PER-AUTHOR BREAKDOWN ---')
print(f"{'Author':<10} | {'Works':>5} | {'Scenes':>6} | {'Sadame_intra':>12} | {'Sadame_tot':>10} | {'Irony_mean':>10} | {'Detach_mean':>11} | {'Hooks':>5} | {'Hook_surv':>9} | {'Phi_zeami':>9}")
for aid in sorted(author_stats.keys()):
    ast = author_stats[aid]
    aname = author_names.get(aid, aid)
    print(f"{aname:<10} | {ast['works']:>5} | {ast['scenes']:>6} | {mean(ast['sadame_intra_chars']):>12.2f} | {mean(ast['sadame_total']):>10.4f} | {mean(ast['awase_irony']):>10.4f} | {mean(ast['awase_detach']):>11.4f} | {mean(ast['kamae_hooks']):>5.1f} | {mean(ast['kamae_survival']):>9.4f} | {mean(ast['nagare_phi_zeami']):>9.4f}")
