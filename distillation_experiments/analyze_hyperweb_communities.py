"""
analyze_hyperweb_communities.py
-------------------------------
全概念ハイパーWebにおける「まとまり（コミュニティ・意味大陸）」の厳密な検出とプロファイリング
"""

import sys
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
from networkx.algorithms import community

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from transformers import AutoModelForCausalLM, AutoTokenizer

def analyze_communities():
    model_name = "Qwen/Qwen2.5-7B"
    print(f"[*] モデルロード中: {model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    W_embed = model.get_input_embeddings().weight.data.to(torch.float32)
    vocab = tokenizer.get_vocab()
    
    purified_words = []
    purified_ids = []
    for raw_s, tid in vocab.items():
        decoded = tokenizer.decode([tid])
        s_strip = decoded.strip()
        if not s_strip.isalpha() or not s_strip.isascii():
            continue
        has_space = raw_s.startswith('Ġ') or decoded.startswith(' ')
        if not has_space or len(s_strip) < 3:
            continue
        clean_w = s_strip.lower()
        if clean_w not in purified_words:
            purified_words.append(clean_w)
            purified_ids.append(tid)

    num_words = len(purified_words)
    ids_tensor = torch.tensor(purified_ids, device=device, dtype=torch.long)
    selected_embeds = W_embed[ids_tensor]
    mean_vec = selected_embeds.mean(dim=0, keepdim=True)
    centered_embeds = selected_embeds - mean_vec
    concept_space = F.normalize(centered_embeds, p=2, dim=-1)

    # グラフ構築 (k=6)
    k_links = 6
    G = nx.DiGraph()
    for i, w in enumerate(purified_words):
        G.add_node(w, id=i)

    batch_size = 4096
    edge_list = []
    for b_start in range(0, num_words, batch_size):
        b_end = min(b_start + batch_size, num_words)
        sub_space = concept_space[b_start:b_end]
        sims = torch.matmul(sub_space, concept_space.T)
        for row_i in range(b_end - b_start):
            sims[row_i, b_start + row_i] = -1.0
        topk = torch.topk(sims, k=k_links, dim=-1)
        top_indices = topk.indices.cpu().numpy()
        top_scores = topk.values.cpu().numpy()

        for row_i in range(b_end - b_start):
            src_w = purified_words[b_start + row_i]
            for rank in range(k_links):
                dst_w = purified_words[top_indices[row_i, rank]]
                edge_list.append((src_w, dst_w, float(top_scores[row_i, rank])))

    G.add_weighted_edges_from(edge_list)
    
    # 無向グラフにして主要コミュニティを検出 (Label Propagation: O(m) で超高速)
    G_undirected = G.to_undirected()
    print("[*] 意味のコミュニティ（大陸・まとまり）を検出中 (Label Propagation)...")
    communities = list(community.label_propagation_communities(G_undirected))

    # 規模順にソート
    communities.sort(key=len, reverse=True)
    print(f"\n[+] 検出された総コミュニティ数: {len(communities)}")
    
    # PageRankの計算
    pagerank = nx.pagerank(G, alpha=0.85, max_iter=50)

    community_profiles = []
    for c_idx, comm in enumerate(communities[:10], 1): # 上位10大コミュニティ
        comm_nodes = list(comm)
        # コミュニティ内のPageRank上位単語
        sorted_nodes = sorted(comm_nodes, key=lambda w: pagerank.get(w, 0.0), reverse=True)
        top_reps = sorted_nodes[:15]
        size = len(comm_nodes)
        share = size / num_words * 100

        print(f"\n🌍 【大陸 #{c_idx}】 規模: {size} 語 ({share:.1f}%)")
        print(f"   代表ハブ概念: {top_reps[:8]}")
        print(f"   主要構成単語例: {top_reps[8:15]}")

        community_profiles.append({
            "cluster_id": c_idx,
            "size": size,
            "share_percent": round(share, 2),
            "top_hubs": top_reps[:8],
            "sample_words": top_reps[8:15]
        })

    with open("distillation_experiments/hyperweb_community_analysis.json", "w", encoding="utf-8") as f:
        json.dump(community_profiles, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    analyze_communities()
