"""
build_neuro_circuit_wilderness.py
---------------------------------
太さと抵抗を持つ思考電気回路網 (Neuro-Circuit Wilderness)

【物理・数理モデル】
1. 各リンクに「太さ（コンダクタンス G）」と「抵抗（レジスタンス R = 1/G）」を付与。
   - G_uv: モデルの連想確率 P(v|u) （よくある自然な連想ほど太いパイプ）
   - R_uv: 1 / G_uv （意外・飛躍した連想ほど高抵抗）
2. キルヒホッフの電流則とオームの法則による電位・電流シミュレーション:
   - 電源（起電力 V=1.0）を概念A（例: apple）に印加。
   - 接地（GND V=0.0）を比較対象B（例: orange, rose, police）に接続。
   - グラフ・ラプラシアン方程式 L · V = I_ext を解き、網全体を流れる電流 I と合成抵抗 R_eff を算出。
3. 「どう違うか」を、思考回路の【合成抵抗値（推論コスト）】と【電流が流れるバイパス経路】として完全定量化！
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

class NeuroCircuitEngine:
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

        print("[*] 真正概念語彙プールを抽出中...")
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
        print(f"[+] 真正概念語彙プール: {len(self.words_list)} 語")

    def extract_weighted_hyperlinks(self, concept: str, top_k=6):
        """
        単語から伸びる外部リンクの『太さ（連想確率 P）』を抽出
        """
        prompts = [f" {concept} is a", f" {concept} relates to", f" {concept} and"]
        links = {}
        stop_words = {"the", "and", "that", "this", "with", "from", "for", "are", "also", "very", "most", "type", "kind", "often", "such", "other", "well", "form", "part"}

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
                p_val = float(score)
                if w not in links or p_val > links[w]:
                    links[w] = p_val
                if len(links) >= top_k:
                    break
            if len(links) >= top_k:
                break
        return links

    def build_circuit_graph(self, core_concepts: list[str]):
        """
        太さ（G）と抵抗（R）を持つ思考電気回路グラフの構築
        """
        print(f"\n[*] 思考電気回路網を構築中...")
        G = nx.Graph() # 電流は双方向に流れるため無向回路として構築

        for c in core_concepts:
            G.add_node(c, node_type="core")

        for c in core_concepts:
            links = self.extract_weighted_hyperlinks(c, top_k=5)
            for dst, prob in links.items():
                if not G.has_node(dst):
                    G.add_node(dst, node_type="hub")
                
                # 太さ (Conductance G) と 抵抗 (Resistance R = 1 / G)
                conductance = max(0.005, prob) # 最小コンダクタンス
                resistance = 1.0 / conductance  # オーム抵抗

                if G.has_edge(c, dst):
                    # 並列回路の合成コンダクタンス G_total = G1 + G2
                    G[c][dst]['conductance'] += conductance
                    G[c][dst]['resistance'] = 1.0 / G[c][dst]['conductance']
                else:
                    G.add_edge(c, dst, conductance=conductance, resistance=resistance)

        print(f"[+] 回路構築完了: ノード数 = {G.number_of_nodes()}, 素子（抵抗リンク）数 = {G.number_of_edges()}")
        return G

    def solve_circuit_current(self, G: nx.Graph, source: str, target: str):
        """
        キルヒホッフ方程式を解き、source(V=1.0) から target(V=0.0) への
        電位分布、枝電流、および全体の有効合成抵抗 R_eff を算出
        """
        nodes = list(G.nodes())
        n = len(nodes)
        node2idx = {node: i for i, node in enumerate(nodes)}

        s_idx = node2idx[source]
        t_idx = node2idx[target]

        # コンダクタンス・ラプラシアン行列 L = D - W
        L = np.zeros((n, n), dtype=np.float64)
        for u, v, data in G.edges(data=True):
            i, j = node2idx[u], node2idx[v]
            g = data['conductance']
            L[i, j] -= g
            L[j, i] -= g
            L[i, i] += g
            L[j, j] += g

        # 境界条件: V_source = 1.0, V_target = 0.0
        A = L.copy()
        b = np.zeros(n, dtype=np.float64)

        # source行を固定: V_s = 1.0
        A[s_idx, :] = 0.0
        A[s_idx, s_idx] = 1.0
        b[s_idx] = 1.0

        # target行を固定: V_t = 0.0
        A[t_idx, :] = 0.0
        A[t_idx, t_idx] = 1.0
        b[t_idx] = 0.0

        # 連立一次方程式を解いて各ノードの電位 V を求める
        V = np.linalg.solve(A, b)

        # 各枝（エッジ）を流れる電流 I_uv = g_uv * |V_u - V_v| を計算
        total_current_in = 0.0
        edge_currents = {}
        for u, v, data in G.edges(data=True):
            i, j = node2idx[u], node2idx[v]
            v_diff = V[i] - V[j]
            current = data['conductance'] * abs(v_diff)
            edge_currents[(u, v)] = current
            if u == source:
                total_current_in += data['conductance'] * (V[i] - V[j])
            elif v == source:
                total_current_in += data['conductance'] * (V[j] - V[i])

        # 合成有効抵抗 R_eff = (V_s - V_t) / I_total = 1.0 / total_current_in
        r_eff = 1.0 / (abs(total_current_in) + 1e-12)

        # 最も電流が多く流れた主要回路（上位パス）を特定
        sorted_branches = sorted(edge_currents.items(), key=lambda x: x[1], reverse=True)

        return {
            "source": source,
            "target": target,
            "effective_resistance": r_eff,
            "total_current": abs(total_current_in),
            "potentials": {node: float(V[node2idx[node]]) for node in nodes},
            "top_current_branches": [(edge, round(c, 5)) for edge, c in sorted_branches[:8]]
        }

def run_neuro_circuit_experiment():
    engine = NeuroCircuitEngine("Qwen/Qwen2.5-7B")

    core_concepts = [
        "apple", "orange", "rose", "banana",
        "police", "crime", "fire", "wood"
    ]

    G = engine.build_circuit_graph(core_concepts)

    # 3つの実験:
    # 1. apple ➔ orange (同カテゴリ・常識の回路)
    # 2. apple ➔ rose (カテゴリ越境・バラ科の回路)
    # 3. apple ➔ police (完全無関係の遠隔回路)
    res_orange = engine.solve_circuit_current(G, "apple", "orange")
    res_rose = engine.solve_circuit_current(G, "apple", "rose")
    res_police = engine.solve_circuit_current(G, "apple", "police")

    print("\n" + "="*70)
    print("⚡ 思考電気回路シミュレーション結果（オームの法則による差分解明）")
    print("="*70)
    print(f"【回路 1: apple ➔ orange (リンゴとミカン)】")
    print(f"  ・有効合成抵抗 (R_eff): {res_orange['effective_resistance']:.2f} Ω (思考の摩擦)")
    print(f"  ・流れた全電流 (Total Current): {res_orange['total_current']:.4f} A")
    print(f"  ・電気が激しく流れた主幹線: {res_orange['top_current_branches'][:4]}")

    print(f"\n【回路 2: apple ➔ rose (リンゴとバラ)】")
    print(f"  ・有効合成抵抗 (R_eff): {res_rose['effective_resistance']:.2f} Ω (思考の摩擦)")
    print(f"  ・流れた全電流 (Total Current): {res_rose['total_current']:.4f} A")
    print(f"  ・電気が流れた主幹線: {res_rose['top_current_branches'][:4]}")

    print(f"\n【回路 3: apple ➔ police (リンゴと警察)】")
    print(f"  ・有効合成抵抗 (R_eff): {res_police['effective_resistance']:.2f} Ω (思考の摩擦)")
    print(f"  ・流れた全電流 (Total Current): {res_police['total_current']:.4f} A")
    print(f"  ・電気が流れた主幹線: {res_police['top_current_branches'][:4]}")
    print("="*70)

    # 思考回路マップの可視化
    # 線の太さ = コンダクタンス（太さ）、線の色 = 抵抗値、ハイライト = apple➔orangeを流れる電流
    plt.figure(figsize=(18, 14))
    pos = nx.spring_layout(G, k=1.0, seed=42)

    # ノード電位に応じたカラーリング (apple=赤, orange=青)
    node_potentials = [res_orange['potentials'].get(n, 0.5) for n in G.nodes()]
    node_sizes = [3200 if n in core_concepts else 1400 for n in G.nodes()]

    # エッジの太さと色
    conductances = [G[u][v]['conductance'] * 80 for u, v in G.edges()]
    resistances = [min(100, G[u][v]['resistance']) for u, v in G.edges()]

    # ノード描画
    nodes = nx.draw_networkx_nodes(
        G, pos, node_size=node_sizes,
        node_color=node_potentials, cmap=plt.cm.coolwarm,
        edgecolors="black", linewidths=1.8
    )
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

    # 回路抵抗エッジの描画
    edges = nx.draw_networkx_edges(
        G, pos, width=conductances, edge_color=resistances,
        edge_cmap=plt.cm.viridis_r, alpha=0.75
    )

    cbar = plt.colorbar(nodes)
    cbar.set_label("Electric Potential (Apple=1.0V High ➔ Orange=0.0V GND)", fontsize=11, fontweight="bold")

    plt.title("⚡ Neuro-Circuit Wilderness (Thinking Electric Network)\nThickness = Conductance | Resistance = Semantic Friction | Colors = Electric Potential V", fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()

    out_png = "distillation_experiments/neuro_circuit_wilderness_map.png"
    plt.savefig(out_png, dpi=300)
    print(f"\n[+] 思考電気回路マップを保存しました: {out_png}")

    import shutil
    art_path = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5/neuro_circuit_wilderness_map.png"
    shutil.copyfile(out_png, art_path)
    print(f"[+] アーティファクトディレクトリにコピー完了: {art_path}")

if __name__ == "__main__":
    run_neuro_circuit_experiment()
