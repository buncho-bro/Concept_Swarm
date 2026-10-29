"""
build_concept_hyperweb_v2.py
----------------------------
真・インターネット型思考原野 (True Concept Hyper-Web)

【核心的改良】
1. 深層ドリフトバイアスを完全排除:
   概念間のハイパーリンクを、LLM本来の「文脈的連想確率（アウトリンク生成確率）」から直接抽出。
2. 各概念ノード（Webページ）は、自身から自然に伸びる外部リンク（Top Out-Links）を持つ。
3. 2つの概念（例: apple と orange、apple と rose）の「どう違うのか」を、
   - 共通して参照しているハブノード（Co-citation / 共通リンク）
   - 片方しか参照していない固有リンク（Distinctive Out-Links）
   というWebトポロジーの集合演算によって完全自動導出する。
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

class TrueConceptHyperWeb:
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

        print("[*] 真正概念語彙プールを構築中...")
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
        self.valid_ids_tensor = torch.tensor(purified_ids, device=self.device)
        
        # 脱中心化埋め込み
        selected_embeds = W_embed[self.valid_ids_tensor]
        self.mean_vec = selected_embeds.mean(dim=0, keepdim=True)
        centered_embeds = selected_embeds - self.mean_vec
        self.concept_space = F.normalize(centered_embeds, p=2, dim=-1)
        print(f"[+] 真正概念語彙プール: {len(self.words_list)} 語")

    def extract_hyperlinks_from_concept(self, concept: str, top_k=5):
        """
        単語（Webページ）から自然に伸びている外部リンク（アウトリンク）を抽出。
        多様な観点（属性、分類、共起）から多角的にリンクを取得。
        """
        prompts = [
            f" {concept} is a",
            f" {concept} relates to",
            f" {concept} and"
        ]
        
        links = []
        stop_words = {"the", "and", "that", "this", "with", "from", "for", "are", "also", "very", "most", "type", "kind", "often", "such", "other", "well", "form", "part", "good"}

        for p in prompts:
            inputs = self.tokenizer(p, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**inputs)
                logits = out.logits[0, -1, :]
                selected_logits = logits[self.valid_ids_tensor]
                probs = F.softmax(selected_logits, dim=-1)
                top_scores, top_indices = torch.topk(probs, k=15)

            for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
                w = self.words_list[idx]
                if w == concept.lower() or w in stop_words or len(w) < 4:
                    continue
                if w not in [l[0] for l in links]:
                    links.append((w, round(float(score) * 100, 2)))
                if len(links) >= top_k:
                    break
            if len(links) >= top_k:
                break
        return links

    def get_concept_representation(self, word: str):
        """文脈表現（Layer 18）の抽出（平均ベクトル引き算）"""
        inputs = self.tokenizer(f" {word}", return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
            h = out.hidden_states[18][0, -1, :].to(torch.float32)
            return F.normalize(h - self.mean_vec.squeeze(0), p=2, dim=0)

    def build_semantic_web(self, core_concepts: list[str]):
        """
        概念Webグラフの構築
        """
        print(f"\n[*] 意味ハイパーWeb（インターネット型思考原野）を生成中...")
        G = nx.DiGraph()

        concept_outlinks = {}
        concept_vecs = {}
        for c in core_concepts:
            links = self.extract_hyperlinks_from_concept(c, top_k=5)
            concept_outlinks[c] = links
            concept_vecs[c] = self.get_concept_representation(c)
            G.add_node(c, node_type="core", size=3200)
            print(f"  🌐 [Node: {c}] ➔ Out-Links: {links}")

            # アウトリンク先をハブノードとして追加
            for dst, weight in links:
                if not G.has_node(dst):
                    G.add_node(dst, node_type="hub", size=1400)
                G.add_edge(c, dst, weight=weight, link_type="semantic_link")

        # コア概念同士の直接リンク
        # 意味的な引き合い（内積）が上位のペアを直接エッジ化
        for i, c1 in enumerate(core_concepts):
            for j, c2 in enumerate(core_concepts):
                if i < j:
                    sim = torch.dot(concept_vecs[c1], concept_vecs[c2]).item()
                    if sim > 0.35: # 意味的引力が強い場合のみリンク
                        G.add_edge(c1, c2, weight=sim*10, link_type="direct_semantic")
                        G.add_edge(c2, c1, weight=sim*10, link_type="direct_semantic")

        return G, concept_outlinks

    def analyze_difference(self, G: nx.DiGraph, outlinks: dict, concept_A: str, concept_B: str):
        """
        Webトポロジー（リンク構造）から『どう違うのか』を論理的に抽出
        """
        links_A = set([l[0] for l in outlinks.get(concept_A, [])])
        links_B = set([l[0] for l in outlinks.get(concept_B, [])])

        common = links_A & links_B
        diff_A = links_A - links_B
        diff_B = links_B - links_A

        # 2ホップでの共通ハブ（共通して繋がっているWebページ）
        succ_A = set(G.successors(concept_A))
        succ_B = set(G.successors(concept_B))
        shared_hubs = succ_A & succ_B

        return {
            "pair": f"{concept_A} vs {concept_B}",
            "direct_common_links": list(common),
            "shared_hubs": list(shared_hubs),
            "only_A_links": list(diff_A),
            "only_B_links": list(diff_B)
        }

def run_experiment():
    engine = TrueConceptHyperWeb("Qwen/Qwen2.5-7B")

    core_concepts = [
        "apple", "orange", "rose", "banana",
        "police", "crime", "fire", "wood"
    ]

    G, outlinks = engine.build_semantic_web(core_concepts)

    diff_apple_orange = engine.analyze_difference(G, outlinks, "apple", "orange")
    diff_apple_rose = engine.analyze_difference(G, outlinks, "apple", "rose")

    print("\n" + "="*70)
    print("🕸️ Webトポロジーによる『どう違うのか』の比較分析レポート")
    print("="*70)
    print("【比較 1: apple vs orange】 (同一カテゴリ内の属性ズレ)")
    print(f"  ・共有しているハブノード (Shared Hubs): {diff_apple_orange['shared_hubs']}")
    print(f"  ・apple 固有のアウトリンク: {diff_apple_orange['only_A_links']}")
    print(f"  ・orange 固有のアウトリンク: {diff_apple_orange['only_B_links']}")

    print("\n【比較 2: apple vs rose】 (カテゴリの断絶と存在様式の飛躍)")
    print(f"  ・共有しているハブノード (Shared Hubs): {diff_apple_rose['shared_hubs']}")
    print(f"  ・apple 固有のアウトリンク: {diff_apple_rose['only_A_links']}")
    print(f"  ・rose 固有のアウトリンク: {diff_apple_rose['only_B_links']}")
    print("="*70)

    # 可視化 (Force-Directed Spring Layout)
    plt.figure(figsize=(18, 14))
    pos = nx.spring_layout(G, k=1.1, iterations=80, seed=42)

    core_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "core"]
    hub_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "hub"]

    core_colors = {
        "apple": "#4caf50", "orange": "#ff9800", "rose": "#e91e63", "banana": "#cddc39",
        "police": "#2196f3", "crime": "#f44336", "fire": "#ff5722", "wood": "#795548"
    }
    c_list = [core_colors.get(n, "#9c27b0") for n in core_nodes]

    # コアノード
    nx.draw_networkx_nodes(G, pos, nodelist=core_nodes, node_color=c_list, node_size=3400, edgecolors="black", linewidths=2.5)
    nx.draw_networkx_labels(G, pos, labels={n: n.upper() for n in core_nodes}, font_size=12, font_weight="bold", font_color="black")

    # ハブノード（薄いグレーのWebリンク先）
    nx.draw_networkx_nodes(G, pos, nodelist=hub_nodes, node_color="#f0f0f0", node_size=1600, edgecolors="#666666", linewidths=1.2)
    nx.draw_networkx_labels(G, pos, labels={n: n for n in hub_nodes}, font_size=9, font_color="#222222")

    # エッジの描画
    semantic_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("link_type") == "semantic_link"]
    direct_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("link_type") == "direct_semantic"]

    nx.draw_networkx_edges(G, pos, edgelist=semantic_edges, edge_color="#3f51b5", width=1.8, alpha=0.6, arrowsize=16, connectionstyle="arc3,rad=0.08")
    nx.draw_networkx_edges(G, pos, edgelist=direct_edges, edge_color="#e91e63", width=2.8, alpha=0.8, arrowsize=22, style="dashed", connectionstyle="arc3,rad=0.15")

    plt.title("Concept Hyper-Web Wilderness (Internet-Style Semantic Topology)\nArrangement by Hyperlinks & Out-Link Overlap instead of Absolute Coordinates", fontsize=15, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()

    out_png = "distillation_experiments/concept_hyperweb_v2_map.png"
    plt.savefig(out_png, dpi=300)
    print(f"\n[+] 真・概念ハイパーWebマップを保存しました: {out_png}")

    import shutil
    art_path = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5/concept_hyperweb_v2_map.png"
    shutil.copyfile(out_png, art_path)
    print(f"[+] アーティファクトディレクトリにコピー完了: {art_path}")

if __name__ == "__main__":
    run_experiment()
