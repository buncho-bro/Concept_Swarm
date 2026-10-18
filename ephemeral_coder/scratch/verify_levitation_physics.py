import torch
from sentence_transformers import SentenceTransformer
import sys

from benchmark_magnetic_levitation import DualSurfaceConceptField, SkyMagnetExtender, load_dataset

embedder = SentenceTransformer('all-MiniLM-L6-v2')
base_anchors, test_words = load_dataset()
base_embs = embedder.encode(base_anchors, convert_to_tensor=True).cpu()
test_embs = embedder.encode(test_words, convert_to_tensor=True).cpu()

field = DualSurfaceConceptField(dim=384)
for label, emb in zip(base_anchors, base_embs):
    field.add_ground_anchor(emb, weight=0.6, sigma=0.28, label=f'Ground:{label[:12]}')

extender = SkyMagnetExtender(field, dim=384)
for i in range(60):
    extender.deploy_sky_magnet(test_embs[i], label=test_words[i][:15], K=3, steps=15)

gc = torch.stack(field.ground_centers)
sc = torch.stack(field.sky_centers)

results = []
for i in range(60):
    emb = test_embs[i]
    pos = field.roll_marble(emb, steps=80)
    
    sim_to_target_mag = torch.dot(pos, sc[i]).item()
    sim_to_all_mag = torch.matmul(pos, sc.T)
    best_mag_idx = torch.argmax(sim_to_all_mag).item()
    best_mag_sim = sim_to_all_mag[best_mag_idx].item()
    
    sim_to_ground = torch.matmul(pos, gc.T)
    best_g_idx = torch.argmax(sim_to_ground).item()
    best_g_sim = sim_to_ground[best_g_idx].item()
    
    # 粒子は磁石と地表アンカーのどちらに近いか？
    fell_to_ground = best_g_sim > best_mag_sim
    captured_by_target = (not fell_to_ground) and (best_mag_idx == i)
    
    results.append({
        'word': test_words[i],
        'target_sim': sim_to_target_mag,
        'best_mag_idx': best_mag_idx,
        'best_mag_sim': best_mag_sim,
        'best_g_idx': best_g_idx,
        'best_g_sim': best_g_sim,
        'fell_to_ground': fell_to_ground,
        'captured': captured_by_target
    })

num_captured = sum(1 for r in results if r['captured'])
num_ground = sum(1 for r in results if r['fell_to_ground'])
num_crosstalk = sum(1 for r in results if not r['fell_to_ground'] and not r['captured'])

print(f'=== 厳密判定 (磁石 vs 地表アンカー競合) ===')
print(f'正しく自機磁石に浮遊捕捉: {num_captured}/60 ({num_captured/60*100:.1f}%)')
print(f'地表アンカーの谷に滑落: {num_ground}/60 ({num_ground/60*100:.1f}%)')
print(f'他磁石への混線吸着: {num_crosstalk}/60 ({num_crosstalk/60*100:.1f}%)')

print('\nサンプル詳細 (最初の15件):')
for r in results[:15]:
    status = 'CAPTURED' if r['captured'] else ('GROUND_FALL' if r['fell_to_ground'] else 'CROSSTALK')
    w = r['word']
    ms = r['target_sim']
    gs = r['best_g_sim']
    gi = r['best_g_idx']
    print(f'[{status:11s}] {w:32s} | TargetMagCos: {ms:.4f} | GroundCos: {gs:.4f} (G_{gi}: {base_anchors[gi][:15]})')
