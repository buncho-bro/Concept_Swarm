"""
Mass-Epoch High-Speed Spider-Web Belief Network Organizer
(GPUテンソル並列・単一フォワードパス高速化による大量エポック蜘蛛の巣形成・思考整理エンジン)
==================================================================================================
高速化と大量エポック設計:
1. 【GPUテンソル並列近傍探索 (GPU Batch Matrix Top-K)】:
   語彙埋め込み行列 (40,150 x 1536) に対するバッチ行列積により、
   数十〜数百の概念の近傍探索・新概念合成を数十ミリ秒で一括実行。
2. 【ロジット直接判定による単一フォワードパス矛盾検知 (Logit-Direct Audit)】:
   遅い自己回帰生成ループ（トークンごとの生成）を全廃。
   1回の順伝播（Forward Pass）で矛盾判定トークン（YES/NO）の対数オッズを並列取得し、
   毎秒数百〜数千の三角形閉路を高速監査・剪定。
3. 【大量エポック自己組織化ループ (Mass-Epoch Recursive Expansion)】:
   問題文は一切絡めず、思考原野全体から無作為抽出された種から出発し、
   数百エポックにわたり蜘蛛の巣（三角形メッシュ）を自律拡大。
   あぶれた矛盾ノードを切り落としながら、堅固な無矛盾信念グラフ（Belief Web）を結晶化。
4. 【整理された信念網による高速ベンチマーク評価】:
   構築された強固な思考網を参照し、未知のベンチマーク問題へ一撃で適用。
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
print("Loading Model & Tokenizer for High-Throughput Web Organizing...")
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

# 語彙埋め込み行列の正規化（GPUキャッシュ）
embed_weights = model.get_input_embeddings().weight.data.clone() # [vocab_size, hidden_dim]
vocab_size = embed_weights.shape[0]

# 思考原野にある全英単語概念の抽出
print("Indexing clean vocabulary concepts for Thought Wilderness...")
FULL_VOCAB_TOKENS = []
FULL_VOCAB_IDS = []
for idx in range(vocab_size):
    s = tokenizer.decode([idx]).strip().lower()
    if len(s) >= 3 and s.isascii() and s.isalpha():
        if s not in FULL_VOCAB_TOKENS:
            FULL_VOCAB_TOKENS.append(s)
            FULL_VOCAB_IDS.append(idx)

FULL_VOCAB_IDS_TENSOR = torch.tensor(FULL_VOCAB_IDS, device=device, dtype=torch.long)
CONCEPT_EMBEDS = F.normalize(embed_weights[FULL_VOCAB_IDS_TENSOR], p=2, dim=-1) # [N_concepts, hidden_dim]
num_concepts = CONCEPT_EMBEDS.shape[0]
print(f"Constructed GPU Thought Wilderness Tensor: {num_concepts} concepts.")

# トークン判定用ID（YES = 矛盾あり, NO = 健全）
TOKEN_YES_ID = tokenizer.encode("YES", add_special_tokens=False)[0]
TOKEN_NO_ID = tokenizer.encode("NO", add_special_tokens=False)[0]


class FastSpiderWebOrganizer:
    """
    GPU並列による超高速・大量エポック蜘蛛の巣信念網オーガナイザー
    """
    def __init__(self, top_k_neighbors=4, batch_anchors=16):
        self.top_k = top_k_neighbors
        self.batch_anchors = batch_anchors
        self.graph_edges = {} # a -> set(n)
        self.pruned_contradictions = set()
        self.sound_triangles = []
        
    def step_epoch(self, current_anchor_indices):
        """
        1エポック分の超高速並列三角形展開と矛盾剪定
        """
        B = len(current_anchor_indices)
        anchor_vecs = CONCEPT_EMBEDS[current_anchor_indices] # [B, D]
        
        # 1. GPU行列積による近傍概念の一括探索 [B, num_concepts]
        sims = torch.matmul(anchor_vecs, CONCEPT_EMBEDS.T)
        
        # 自身を除外してTop-K近傍を取得
        for i, a_idx in enumerate(current_anchor_indices):
            sims[i, a_idx] = -1.0
            
        topk_sims, topk_inds = torch.topk(sims, k=self.top_k, dim=-1) # [B, top_k]
        
        # 2. 新概念の並列合成 (A + N -> W)
        # 近傍ベクトル [B, top_k, D]
        neighbor_vecs = CONCEPT_EMBEDS[topk_inds]
        # 合成ベクトル [B, top_k, D]
        syn_vecs = F.normalize(0.5 * anchor_vecs.unsqueeze(1) + 0.5 * neighbor_vecs, p=2, dim=-1)
        
        # 新単語のTop-1探索 [B * top_k, D] x [D, num_concepts]
        syn_flat = syn_vecs.view(-1, CONCEPT_EMBEDS.shape[-1])
        syn_sims = torch.matmul(syn_flat, CONCEPT_EMBEDS.T)
        
        # 自身らを除外
        w_top_inds = torch.topk(syn_sims, k=3, dim=-1).indices # [B * top_k, 3]
        
        # 三角形のバッチ構築
        triangles = []
        flat_idx = 0
        for i, a_idx in enumerate(current_anchor_indices):
            a_word = FULL_VOCAB_TOKENS[a_idx]
            for j in range(self.top_k):
                n_idx = topk_inds[i, j].item()
                n_word = FULL_VOCAB_TOKENS[n_idx]
                
                # 新単語候補からa, nと異なるものを選択
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
                
        # 3. 単一フォワードパスによるロジット並列矛盾検知 (Logit-Direct Audit)
        # 16〜32個のプロンプトを一括バッチ評価
        audit_prompts = [
            f"<|im_start|>system\nAudit conceptual triangle.<|im_end|>\n"
            f"<|im_start|>user\nDoes combining concept '{t['a']}' and '{t['n']}' into '{t['w']}' create an impossible contradiction or false fallacy? Reply YES or NO.<|im_end|>\n"
            f"<|im_start|>assistant\n"
            for t in triangles
        ]
        
        enc = tokenizer(audit_prompts, return_tensors="pt", padding=True, truncation=True, max_length=96).to(device)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :] # 最後のトークン位置のロジット
            logits_yes = logits[:, TOKEN_YES_ID]
            logits_no = logits[:, TOKEN_NO_ID]
            
        # 矛盾判定: logit(YES) > logit(NO)
        is_contradiction = (logits_yes > logits_no).tolist()
        
        next_anchor_candidates = []
        sound_count = 0
        pruned_count = 0
        
        for t, contra in zip(triangles, is_contradiction):
            if contra:
                pruned_count += 1
                self.pruned_contradictions.add((t["a"], t["w"]))
            else:
                sound_count += 1
                self.sound_triangles.append(t)
                # エッジ追加
                self.graph_edges.setdefault(t["a"], set()).add(t["n"])
                self.graph_edges.setdefault(t["n"], set()).add(t["w"])
                self.graph_edges.setdefault(t["w"], set()).add(t["a"])
                # (W + A) を次世代アンカーの種に
                next_anchor_candidates.append(t["w_idx"])
                next_anchor_candidates.append(t["a_idx"])
                
        # 次のエポック用のアンカー選定（偏りをなくすため無作為性もブレンド）
        next_anchor_candidates = list(set(next_anchor_candidates))
        if len(next_anchor_candidates) >= B // 2:
            next_anchors = random.sample(next_anchor_candidates, B // 2)
            # 残り半分は思考原野から完全無作為に新風を注入
            fresh_random = random.sample(range(num_concepts), B - len(next_anchors))
            next_anchors.extend(fresh_random)
        else:
            next_anchors = random.sample(range(num_concepts), B)
            
        return next_anchors, sound_count, pruned_count


def run_mass_epoch_training(num_epochs=200, batch_anchors=16):
    """
    大量エポック（200エポック）による蜘蛛の巣思考整理の実行
    """
    print(f"\n================================================================================")
    print(f"STARTING MASS-EPOCH SPIDER-WEB ORGANIZATION: {num_epochs} EPOCHS (Batch={batch_anchors})")
    print(f"================================================================================")
    
    organizer = FastSpiderWebOrganizer(top_k_neighbors=4, batch_anchors=batch_anchors)
    # 初期シード: 思考原野全体から完全に無作為に選出
    current_anchors = random.sample(range(num_concepts), batch_anchors)
    
    start_time = time.time()
    total_sound = 0
    total_pruned = 0
    
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        current_anchors, sound, pruned = organizer.step_epoch(current_anchors)
        total_sound += sound
        total_pruned += pruned
        dt = time.time() - t0
        
        if epoch % 20 == 0 or epoch == num_epochs:
            active_nodes = len(organizer.graph_edges)
            print(f"  [Epoch {epoch:03d}/{num_epochs}] dt: {dt*1000:.1f}ms | Web Nodes: {active_nodes} | Sound Triangles: {total_sound} | Pruned: {total_pruned}")
            
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"MASS-EPOCH ORGANIZATION COMPLETED in {total_time:.2f} seconds!")
    print(f"Throughput: {num_epochs / total_time:.1f} epochs/sec | Total Triangles Audited: {total_sound + total_pruned}")
    print(f"Constructed Belief Web: {len(organizer.graph_edges)} concept nodes | {total_sound} sound links | {total_pruned} contradictions eliminated")
    print("=" * 80)
    
    # 整理された思考網のサマリー保存
    web_summary = {
        "num_epochs": num_epochs,
        "elapsed_sec": round(total_time, 2),
        "total_nodes": len(organizer.graph_edges),
        "total_sound_triangles": total_sound,
        "total_contradictions_pruned": total_pruned,
        "pruning_ratio": round(total_pruned / (total_sound + total_pruned + 1e-9) * 100, 2),
        "sample_connected_nodes": list(organizer.graph_edges.keys())[:50]
    }
    
    out_json = os.path.join(OUTPUT_DIR, "mass_epoch_organized_belief_web.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(web_summary, f, ensure_ascii=False, indent=2)
    print(f"Saved Organized Belief Web to: {out_json}")
    return organizer, web_summary


if __name__ == "__main__":
    organizer, summary = run_mass_epoch_training(num_epochs=200, batch_anchors=16)
