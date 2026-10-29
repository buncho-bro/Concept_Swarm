"""
build_concept_hyperweb.py
-------------------------
インターネット型思考原野 (Concept Hyper-Web) プロトタイプ

【パラダイムシフト】
- 絶対的な3次元空間座標（ユークリッド幾何学）を完全撤廃。
- 概念を「Webページ（ノード）」とし、関係性を「有向ハイパーリンク（エッジ＋アンカーテキスト）」として並べる。
- 2つの概念を比較する際、スカラー距離を測るのではなく、
  「どのような共通リンクを持ち、どのリンクで枝分かれしているか」というWebトポロジーの差分として
  「どう違うのか」を完全可視化する。
"""

import sys
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from transformers import AutoModelForCausalLM, AutoTokenizer

class ConceptHyperWebEngine:
    def __init__(self, model_name="Qwen/Qwen2.5-7B"):
        print(f"[*] 概念言語モデルをロード中: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()

        print("[*] 真正概念語彙空間を準備中...")
        W_embed = self.model.get_input_embeddings().weight.data.to(torch.float32)
        vocab = self.tokenizer.get_vocab()
        
        purified_words = []
        purified_ids = []
        for raw_s, tid in vocab.items():
            decoded = self.tokenizer.decode([tid])
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

        self.words_list = purified_words
        self.ids_tensor = torch.tensor(purified_ids, device=self.device, dtype=torch.long)
        
        # 脱中心化語彙空間
        selected_embeds = W_embed[self.ids_tensor]
        self.mean_vec = selected_embeds.mean(dim=0, keepdim=True)
        centered_embeds = selected_embeds - self.mean_vec
        self.concept_space = F.normalize(centered_embeds, p=2, dim=-1)
        print(f"[+] 真正概念語彙プール: {len(self.words_list)} 語")

    def get_concept_representation(self, word: str):
        """単語の深層文脈ベクトル（推論層 Layer 18）を抽出"""
        inputs = self.tokenizer(f" {word}", return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
            # 中間推論層（Layer 18）のベクトル
            h = out.hidden_states[18][0, -1, :].to(torch.float32) - self.mean_vec.squeeze(0)
            return F.normalize(h, p=2, dim=0)

    def extract_anchor_text(self, vec_A: torch.Tensor, vec_B: torch.Tensor, top_k=2):
        """
        2つのノード間の有向リンクに付与する「アンカーテキスト（どういう関係か）」を抽出。
        差分ベクトル Δv = vec_B - vec_A を語彙空間へ逆射影。
        """
        delta = F.normalize(vec_B - vec_A, p=2, dim=0)
        sims = torch.mv(self.concept_space, delta)
        top_indices = torch.topk(sims, k=15).indices.tolist()
        
        anchors = []
        for idx in top_indices:
            w = self.words_list[idx]
            if len(w) >= 3 and w not in anchors:
                anchors.append(w)
            if len(anchors) >= top_k:
                break
        return "/".join(anchors)

    def build_hyperweb(self, core_concepts: list[str], top_links_per_node=4):
        """
        指定されたコア概念群から、有向ハイパーリンクWebグラフを自発的に組織化
        """
        print(f"\n[*] 概念ハイパーWebを組織化中: {core_concepts}")
        G = nx.DiGraph()

        concept_vecs = {}
        for c in core_concepts:
            concept_vecs[c] = self.get_concept_representation(c)
            G.add_node(c, label=c)

        # 全ノード間のリンク強度（遷移確率）を計算
        for src in core_concepts:
            v_src = concept_vecs[src]
            
            # 他の全ノードとの類似度（Attentionリンク候補）
            scores = []
            for dst in core_concepts:
                if src == dst:
                    continue
                v_dst = concept_vecs[dst]
                sim = torch.dot(v_src, v_dst).item()
                scores.append((dst, sim, v_dst))
            
            # 上位リンクを抽出
            scores.sort(key=lambda x: x[1], reverse=True)
            for dst, sim, v_dst in scores[:top_links_per_node]:
                # アンカーテキスト（関係ラベル）の生成
                anchor = self.extract_anchor_text(v_src, v_dst)
                G.add_edge(src, dst, weight=sim, anchor=anchor)
                print(f"  🔗 [Link] {src} ➔ {dst} | Weight={sim:.3f} | Anchor='{anchor}'")

        return G, concept_vecs

    def compare_via_topology(self, G: nx.DiGraph, concept_A: str, concept_B: str):
        """
        2つの概念の「どう違うのか」をWebトポロジー（リンクの共通点と差分）から導出
        """
        neighbors_A = set(G.successors(concept_A)) | set(G.predecessors(concept_A))
        neighbors_B = set(G.successors(concept_B)) | set(G.predecessors(concept_B))

        common = neighbors_A & neighbors_B
        only_A = neighbors_A - neighbors_B
        only_B = neighbors_B - neighbors_A

        report = {
            "pair": f"{concept_A} vs {concept_B}",
            "common_links": list(common),
            "only_A_links": list(only_A),
            "only_B_links": list(only_B)
        }
        return report

def run_hyperweb_demo():
    engine = ConceptHyperWebEngine("Qwen/Qwen2.5-7B")

    # ユーザーの例を取り入れた概念群
    # 1. 植物・食文化クラスタ: apple, orange, rose, banana
    # 2. 社会・司法クラスタ: crime, police, arrest, law
    # 3. 物理・燃焼クラスタ: fire, wood, ash, smoke
    target_concepts = [
        "apple", "orange", "rose", "banana",
        "crime", "police", "arrest", "law",
        "fire", "wood", "ash", "smoke"
    ]

    G, vecs = engine.build_hyperweb(target_concepts, top_links_per_node=3)

    # トポロジー差分分析（どう違うのかの抽出）
    comp1 = engine.compare_via_topology(G, "apple", "orange")
    comp2 = engine.compare_via_topology(G, "apple", "rose")

    print("\n" + "="*70)
    print("📊 Webトポロジーによる『どう違うのか』の分析レポート")
    print("="*70)
    print(f"【比較 1: apple vs orange (同カテゴリ内の属性差)】")
    print(f"  ・共通リンク（共有するWeb圏）: {comp1['common_links']}")
    print(f"  ・apple固有リンク: {comp1['only_A_links']}")
    print(f"  ・orange固有リンク: {comp1['only_B_links']}")

    print(f"\n【比較 2: apple vs rose (カテゴリを超えた飛躍差)】")
    print(f"  ・共通リンク（共有するWeb圏）: {comp2['common_links']}")
    print(f"  ・apple固有リンク: {comp2['only_A_links']}")
    print(f"  ・rose固有リンク: {comp2['only_B_links']}")
    print("="*70)

    # ネットワーク可視化（Force-Directed Spring Layout: つながりが空間配置を決める）
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G, k=0.85, seed=42)

    # クラスタごとの色分け
    color_map = []
    for node in G.nodes():
        if node in ["apple", "orange", "rose", "banana"]:
            color_map.append("#a1d99b" if node != "rose" else "#fc9272")
        elif node in ["crime", "police", "arrest", "law"]:
            color_map.append("#9ecae1")
        else:
            color_map.append("#fdae6b")

    # ノードとエッジの描画
    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=2800, edgecolors="black", linewidths=1.5)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold", font_family="sans-serif")
    
    # 有向エッジ（矢印）
    nx.draw_networkx_edges(G, pos, edge_color="gray", width=2.0, alpha=0.7, arrowsize=18, arrowstyle="->", connectionstyle="arc3,rad=0.15")

    # アンカーテキスト（関係ラベル）の描画
    edge_labels = nx.get_edge_attributes(G, 'anchor')
    # 見やすさのため一部の主要ラベルのみ表示
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color="darkblue", bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    plt.title("🌐 Concept Hyper-Web (Internet-Style Semantic Wilderness)\nArrangement by Connectivity & Hyperlinks instead of Absolute Spatial Coordinates", fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()

    out_png = "distillation_experiments/concept_hyperweb_map.png"
    plt.savefig(out_png, dpi=300)
    print(f"\n[+] 概念ハイパーWebマップを保存しました: {out_png}")

    # アーティファクトへコピー
    import shutil
    art_path = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5/concept_hyperweb_map.png"
    shutil.copyfile(out_png, art_path)
    print(f"[+] アーティファクトディレクトリにコピー完了: {art_path}")

if __name__ == "__main__":
    run_hyperweb_demo()
