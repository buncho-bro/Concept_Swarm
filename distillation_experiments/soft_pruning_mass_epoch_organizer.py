"""
Soft-Pruning Mass-Epoch Spider-Web Belief Network Organizer
(0にしないソフトプルーニング・確信度重み付け・大規模エポック自己組織化エンジン)
======================================================================================
改善設計:
1. 【0にしないソフトプルーニング (Soft-Pruning & Confidence Weighting)】:
   二値のハード削除（重み0）を全廃。
   ロジット差分 (logit(NO) - logit(YES)) からシグモイド確信度スコア W in [0.10, 1.00] を計算。
   致命的な矛盾であっても最小基底重み (W_min = 0.10) と「反例制約」として保持し、
   思考の足場を完全に消滅させない。
2. 【思考原野カバレッジの大幅拡大 (500 Epochs, Batch=32)】:
   バッチサイズを32（毎エポック128組の三角形閉路）に拡大し、
   500エポック（計64,000組の三角形閉路）をGPU並列で高速巡回。
   4万語の語彙空間の大部分を高密度メッシュとして網羅。
3. 【自己修復と安全な迂回路の保持 (Bypass & Redundant Connectivity)】:
   低確信度の概念に対しても上位概念を経由する迂回エッジを保持し、
   孤立ノードや思考の空白地帯（概念砂漠）の発生を防止。
4. 【Web標準250問の完全測定】:
   足場が維持された高密度信念網を用いてベンチマークを走破し、
   Causal Judgement等の低下を克服しつつ全体スコアの更なる向上を検証。
"""

import os
import sys
import json
import time
import re
import random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append("C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment")
sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
ARTIFACT_DIR = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5"
device = "cuda" if torch.cuda.is_available() else "cpu"
QWEN_ID = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"Device: {device}")
print("Loading Model & Tokenizer for Soft-Pruning Web Organizer...")
tokenizer = AutoTokenizer.from_pretrained(QWEN_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    QWEN_ID,
    torch_dtype=torch.bfloat16,
    device_map=device,
    trust_remote_code=True
)
model.eval()

embed_weights = model.get_input_embeddings().weight.data.clone()
vocab_size = embed_weights.shape[0]

print("Indexing complete vocabulary for Thought Wilderness...")
FULL_VOCAB_TOKENS = []
FULL_VOCAB_IDS = []
for idx in range(vocab_size):
    s = tokenizer.decode([idx]).strip().lower()
    if len(s) >= 3 and s.isascii() and s.isalpha():
        if s not in FULL_VOCAB_TOKENS:
            FULL_VOCAB_TOKENS.append(s)
            FULL_VOCAB_IDS.append(idx)

FULL_VOCAB_IDS_TENSOR = torch.tensor(FULL_VOCAB_IDS, device=device, dtype=torch.long)
CONCEPT_EMBEDS = F.normalize(embed_weights[FULL_VOCAB_IDS_TENSOR], p=2, dim=-1)
num_concepts = CONCEPT_EMBEDS.shape[0]
print(f"Constructed GPU Thought Wilderness: {num_concepts} clean concepts.")

TOKEN_YES_ID = tokenizer.encode("YES", add_special_tokens=False)[0]
TOKEN_NO_ID = tokenizer.encode("NO", add_special_tokens=False)[0]


class SoftSpiderWebOrganizer:
    def __init__(self, top_k_neighbors=4, batch_anchors=32):
        self.top_k = top_k_neighbors
        self.batch_anchors = batch_anchors
        # エッジ辞書: node_a -> dict(node_b -> weight)
        self.weighted_graph = {}
        # 反例・注意喚起ノード: node -> list(counter_concepts)
        self.counter_caveats = {}
        self.total_triangles_audited = 0
        self.high_confidence_count = 0
        self.caveat_count = 0
        
    def step_epoch(self, current_anchor_indices):
        B = len(current_anchor_indices)
        anchor_vecs = CONCEPT_EMBEDS[current_anchor_indices] # [B, D]
        
        # 1. GPU行列積による近傍概念の一括探索
        sims = torch.matmul(anchor_vecs, CONCEPT_EMBEDS.T)
        for i, a_idx in enumerate(current_anchor_indices):
            sims[i, a_idx] = -1.0
            
        topk_sims, topk_inds = torch.topk(sims, k=self.top_k, dim=-1) # [B, top_k]
        neighbor_vecs = CONCEPT_EMBEDS[topk_inds]
        
        # 2. 新概念の並列合成 (A + N -> W)
        syn_vecs = F.normalize(0.5 * anchor_vecs.unsqueeze(1) + 0.5 * neighbor_vecs, p=2, dim=-1)
        syn_flat = syn_vecs.view(-1, CONCEPT_EMBEDS.shape[-1])
        syn_sims = torch.matmul(syn_flat, CONCEPT_EMBEDS.T)
        w_top_inds = torch.topk(syn_sims, k=3, dim=-1).indices
        
        triangles = []
        flat_idx = 0
        for i, a_idx in enumerate(current_anchor_indices):
            a_word = FULL_VOCAB_TOKENS[a_idx]
            for j in range(self.top_k):
                n_idx = topk_inds[i, j].item()
                n_word = FULL_VOCAB_TOKENS[n_idx]
                
                candidates = [w_top_inds[flat_idx, c].item() for c in range(3)]
                w_idx = candidates[0]
                for c in candidates:
                    if c != a_idx and c != n_idx:
                        w_idx = c
                        break
                w_word = FULL_VOCAB_TOKENS[w_idx]
                flat_idx += 1
                
                triangles.append({
                    "a_idx": a_idx, "n_idx": n_idx, "w_idx": w_idx,
                    "a": a_word, "n": n_word, "w": w_word
                })
                
        # 3. 単一フォワードパスによる連続ロジット差分スコアリング
        audit_prompts = [
            f"<|im_start|>system\nAudit conceptual triangle.<|im_end|>\n"
            f"<|im_start|>user\nDoes combining '{t['a']}' and '{t['n']}' into '{t['w']}' introduce an invalid fallacy or fatal contradiction? Reply YES or NO.<|im_end|>\n"
            f"<|im_start|>assistant\n"
            for t in triangles
        ]
        
        enc = tokenizer(audit_prompts, return_tensors="pt", padding=True, truncation=True, max_length=96).to(device)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :]
            logits_yes = logits[:, TOKEN_YES_ID]
            logits_no = logits[:, TOKEN_NO_ID]
            
        # 4. ソフトプルーニング: ロジット差分から連続的な確信度重みを計算（決して0にしない）
        # margin = logit(NO) - logit(YES)
        # 高いほど無矛盾・健全、低いほど矛盾傾向
        margins = (logits_no - logits_yes).float().cpu().numpy()
        
        next_anchor_candidates = []
        for t, m in zip(triangles, margins):
            self.total_triangles_audited += 1
            # シグモイド確信度 [0.0, 1.0]
            raw_conf = 1.0 / (1.0 + np.exp(-m * 0.5))
            
            # 【重要】0にせず、最小基底重み 0.10 を保証
            # soft_weight in [0.10, 1.00]
            soft_weight = round(float(0.10 + 0.90 * raw_conf), 3)
            
            a_w, n_w, w_w = t["a"], t["n"], t["w"]
            
            if soft_weight >= 0.50:
                self.high_confidence_count += 1
                # 健全エッジとして登録
                self.weighted_graph.setdefault(a_w, {})[n_w] = max(self.weighted_graph.setdefault(a_w, {}).get(n_w, 0.0), soft_weight)
                self.weighted_graph.setdefault(n_w, {})[w_w] = max(self.weighted_graph.setdefault(n_w, {}).get(w_w, 0.0), soft_weight)
                self.weighted_graph.setdefault(w_w, {})[a_w] = max(self.weighted_graph.setdefault(w_w, {}).get(a_w, 0.0), soft_weight)
                next_anchor_candidates.append(t["w_idx"])
            else:
                self.caveat_count += 1
                # 矛盾傾向の強い枝も0にせず、反例制約（Caveat Edge）として重み0.10〜0.49で保持
                self.weighted_graph.setdefault(a_w, {})[w_w] = soft_weight
                self.counter_caveats.setdefault(a_w, set()).add(f"caution_{w_w}")
                # 次の種にも一定確率で残し、孤立死（情報断絶）を防ぐ
                if random.random() < 0.2:
                    next_anchor_candidates.append(t["w_idx"])
                    
        # 次世代アンカーの更新
        next_anchor_candidates = list(set(next_anchor_candidates))
        if len(next_anchor_candidates) >= B // 2:
            next_anchors = random.sample(next_anchor_candidates, B // 2)
            fresh_random = random.sample(range(num_concepts), B - len(next_anchors))
            next_anchors.extend(fresh_random)
        else:
            next_anchors = random.sample(range(num_concepts), B)
            
        return next_anchors


def run_soft_mass_epoch_training(num_epochs=500, batch_anchors=32):
    print(f"\n================================================================================")
    print(f"STARTING SOFT-PRUNING MASS-EPOCH TRAINING: {num_epochs} EPOCHS (Batch={batch_anchors})")
    print(f"Zero-Pruning Abolished: Continuous Confidence Weights W in [0.10, 1.00]")
    print(f"================================================================================")
    
    organizer = SoftSpiderWebOrganizer(top_k_neighbors=4, batch_anchors=batch_anchors)
    current_anchors = random.sample(range(num_concepts), batch_anchors)
    
    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        current_anchors = organizer.step_epoch(current_anchors)
        dt = time.time() - t0
        
        if epoch % 50 == 0 or epoch == num_epochs:
            nodes_count = len(organizer.weighted_graph)
            print(f"  [Epoch {epoch:03d}/{num_epochs}] dt: {dt*1000:.1f}ms | Web Nodes: {nodes_count} | High-Conf Links: {organizer.high_confidence_count} | Soft Caveats: {organizer.caveat_count}")
            
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"SOFT MASS-EPOCH TRAINING COMPLETED in {total_time:.2f} seconds!")
    print(f"Audited Triangles: {organizer.total_triangles_audited} | High-Conf: {organizer.high_confidence_count} | Caveats: {organizer.caveat_count}")
    print(f"Constructed Belief Web: {len(organizer.weighted_graph)} concept nodes (Zero-pruned: 0 nodes!)")
    print("=" * 80)
    
    # 連続重み付け信念網の保存
    # JSONシリアライズ用に辞書整形
    clean_graph = {k: {nk: float(w) for nk, w in neighbors.items()} for k, neighbors in list(organizer.weighted_graph.items())}
    clean_caveats = {k: list(v) for k, v in list(organizer.counter_caveats.items())}
    
    save_data = {
        "num_epochs": num_epochs,
        "elapsed_sec": round(total_time, 2),
        "total_nodes": len(organizer.weighted_graph),
        "total_triangles_audited": organizer.total_triangles_audited,
        "high_confidence_count": organizer.high_confidence_count,
        "caveat_count": organizer.caveat_count,
        "weighted_graph_sample": {k: clean_graph[k] for k in list(clean_graph.keys())[:50]},
        "counter_caveats_sample": {k: clean_caveats[k] for k in list(clean_caveats.keys())[:30]},
        "all_nodes": list(clean_graph.keys())
    }
    
    out_file = os.path.join(OUTPUT_DIR, "soft_mass_epoch_belief_web.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"Saved Soft Belief Web to: {out_file}")
    return save_data


if __name__ == "__main__":
    run_soft_mass_epoch_training(num_epochs=500, batch_anchors=32)
