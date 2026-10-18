import torch
import numpy as np
from sentence_transformers import SentenceTransformer

# 抽出時の直交デカップリング (Orthogonal Decoupled Interference) の検証
embedder = SentenceTransformer("all-MiniLM-L6-v2")
dim = 384

concept_words = [
    "Function", "Variable", "Loop", "Recursion", "Asynchronous", 
    "Concurrency", "Thread", "Process", "Lock", "Mutex"
]
raw_embs = embedder.encode(concept_words, convert_to_tensor=True).cpu()
teacher_v = raw_embs / torch.norm(raw_embs, dim=-1, keepdim=True)

W_proj = torch.randn(dim, dim) * 0.05

# 従来の連成干渉 vs 直交デカップリング干渉
drifts_coupled = []
drifts_decoupled = []

for i in range(len(concept_words)):
    v_i = teacher_v[i]
    
    # 近傍3概念
    cos_sims = torch.matmul(v_i, teacher_v.T)
    cos_sims[i] = -999.0
    topk = torch.topk(cos_sims, 3).indices.tolist()
    neighbors = [teacher_v[k] for k in topk]

    # 1. 従来の連成干渉 (他概念の法線をそのまま加算)
    coupled_normals = []
    for c_k in neighbors:
        proj = torch.dot(c_k, v_i) * v_i
        ortho = c_k - proj
        n_ortho = torch.norm(ortho)
        coupled_normals.append(ortho / n_ortho if n_ortho > 1e-5 else torch.randn(dim)/10.0)

    x_c = (v_i + 0.1 * torch.matmul(W_proj, v_i)).clone().detach().requires_grad_(True)
    for _ in range(25):
        phi = torch.zeros(1)
        for n_k in coupled_normals:
            dot = torch.dot(x_c, n_k)
            phi += torch.exp(- (dot**2) / (2 * 0.25**2))
        grad = torch.autograd.grad(phi, x_c)[0]
        proj_grad = grad - torch.dot(x_c, grad) * x_c
        x_next = x_c + 0.08 * proj_grad
        x_c = (x_next / torch.norm(x_next)).detach().requires_grad_(True)

    cos_c = torch.dot(x_c.detach(), v_i).item()
    drifts_coupled.append(np.degrees(np.arccos(np.clip(cos_c, -1.0, 1.0))))

    # 2. 直交デカップリング干渉 (Gram-Schmidt による近傍引力の相互直交化・バイアス相殺)
    decoupled_normals = []
    for c_k in neighbors:
        proj = torch.dot(c_k, v_i) * v_i
        ortho = c_k - proj
        # 既存法線成分を除去して純粋な直交補空間を形成
        for prev_n in decoupled_normals:
            ortho = ortho - torch.dot(ortho, prev_n) * prev_n
        n_ortho = torch.norm(ortho)
        if n_ortho > 1e-5:
            decoupled_normals.append(ortho / n_ortho)

    x_d = (v_i + 0.1 * torch.matmul(W_proj, v_i)).clone().detach().requires_grad_(True)
    for _ in range(25):
        phi = torch.zeros(1)
        # LogSumExp (Smooth-Max) で法線引力の偏りを相殺
        pulls = []
        for n_k in decoupled_normals:
            dot = torch.dot(x_d, n_k)
            pulls.append(torch.exp(- (dot**2) / (2 * 0.25**2)))
        phi = torch.logsumexp(torch.stack(pulls) * 4.0, dim=0) / 4.0
        grad = torch.autograd.grad(phi, x_d)[0]
        proj_grad = grad - torch.dot(x_d, grad) * x_d
        x_next = x_d + 0.08 * proj_grad
        x_d = (x_next / torch.norm(x_next)).detach().requires_grad_(True)

    cos_d = torch.dot(x_d.detach(), v_i).item()
    drifts_decoupled.append(np.degrees(np.arccos(np.clip(cos_d, -1.0, 1.0))))

print("=== ズレの比較結果 ===")
print(f"従来の大円干渉抽出の平均ズレ: {np.mean(drifts_coupled):.2f} 度")
print(f"直交デカップリング干渉の平均ズレ: {np.mean(drifts_decoupled):.2f} 度")
for w, dc, dd in zip(concept_words[:5], drifts_coupled[:5], drifts_decoupled[:5]):
    print(f"  ・{w:12s}: 従来 {dc:5.2f}° => 改善後 {dd:5.2f}° (低減率 {((dc-dd)/dc)*100:.1f}%)")
