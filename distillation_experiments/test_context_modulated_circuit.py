"""
test_context_modulated_circuit.py
---------------------------------
文脈（コンテキスト）による思考電気抵抗の動的変容シミュレーション

【実証内容】
1. 中立文脈 (Neutral): "apple"
2. 植物学文脈 (Botanical): "In botanical taxonomy, plants like apple"
3. 食品売り場文脈 (Grocery): "In the supermarket grocery fruit section, fresh apple"
4. 事件捜査文脈 (Theft Investigation): "In the orchard theft case, the stolen apple"

上記4つの文脈が入力されたとき、
apple から [orange], [rose], [police] への「有効合成抵抗 R_eff（思考の摩擦）」が
どのように劇的に揺らぎ、回路が開閉・逆転するかをオームの法則で完全定量化する。
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

class ContextModulatedCircuitEngine:
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

        # 真正概念語彙プールの抽出
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
        self.word2idx = {w: i for i, w in enumerate(purified_words)}
        print(f"[+] 真正概念語彙プール: {len(self.words_list)} 語")

    def compute_contextual_resistance(self, context_prompt: str, source="apple", targets=["orange", "rose", "police"]):
        """
        文脈 context_prompt の下で、source から各 target への直接およびハブ経由の
        コンダクタンス G(Context) と 抵抗 R = 1/G を計算
        """
        inputs = self.tokenizer(context_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
            logits = out.logits[0, -1, :] # 最後のトークンからの予測
            selected_logits = logits[self.valid_ids_tensor]
            probs = F.softmax(selected_logits, dim=-1)

        probs_cpu = probs.float().cpu().numpy()
        
        # ターゲット単語への直接コンダクタンス
        results = {}
        for tgt in targets:
            tgt_prob = 1e-6 # 最小確率（無限大抵抗の防止）
            if tgt in self.word2idx:
                w_idx = self.word2idx[tgt]
                tgt_prob = max(tgt_prob, float(probs_cpu[w_idx]))

            # 抵抗 R = 1 / P (オーム)
            resistance = 1.0 / tgt_prob
            # 扱いやすい対数スケール抵抗（情報量サプライザル -log2 P）も併記
            surprisal_r = -np.log2(tgt_prob)
            
            results[tgt] = {
                "probability": tgt_prob,
                "resistance_ohm": resistance,
                "surprisal_resistance": surprisal_r
            }

        # 文脈下で最も強く発火したTop 5ハブ単語
        top_k = torch.topk(probs, k=8)
        top_hubs = []
        for idx in top_k.indices.tolist():
            w = self.words_list[idx]
            if w not in [source] + targets and len(w) >= 4:
                top_hubs.append(w)
            if len(top_hubs) >= 4:
                break

        return results, top_hubs

def run_context_experiment():
    engine = ContextModulatedCircuitEngine("Qwen/Qwen2.5-7B")

    scenarios = [
        {
            "name": "① 中立文脈 (Neutral Baseline)",
            "prompt": " apple",
            "desc": "単語単体としてのデフォルト状態"
        },
        {
            "name": "② 食品売り場文脈 (Grocery Fruit)",
            "prompt": " In the supermarket grocery fruit section, fresh apple",
            "desc": "可食・甘味・デザートの文脈"
        },
        {
            "name": "③ 植物学分類文脈 (Botanical Taxonomy)",
            "prompt": " In botanical classification, angiosperms like apple",
            "desc": "被子植物・バラ科（Rosaceae）の学術文脈"
        },
        {
            "name": "④ 農園盗難事件文脈 (Theft Investigation)",
            "prompt": " In the orchard theft investigation, the stolen apple",
            "desc": "事件・盗難・警察捜査の文脈"
        }
    ]

    targets = ["orange", "rose", "police"]
    all_results = []

    print("\n" + "="*75)
    print("⚡ 文脈（コンテキスト）による思考電気抵抗の動的揺らぎ測定実験")
    print("="*75)

    matrix_surprisal = []

    for sc in scenarios:
        res, hubs = engine.compute_contextual_resistance(sc["prompt"], source="apple", targets=targets)
        print(f"\n【{sc['name']}】")
        print(f"  ・文脈プロンプト: \"{sc['prompt']}\"")
        print(f"  ・活性化した主要ハブ: {hubs}")
        row = []
        for tgt in targets:
            r_info = res[tgt]
            p_pct = r_info["probability"] * 100
            s_r = r_info["surprisal_resistance"]
            row.append(s_r)
            print(f"    ➔ [{tgt}]: 抵抗 = {s_r:.2f} bit (生抵抗: {r_info['resistance_ohm']:.1f} Ω | 伝導確率: {p_pct:.4f}%)")
        matrix_surprisal.append(row)

        all_results.append({
            "scenario": sc["name"],
            "prompt": sc["prompt"],
            "hubs": hubs,
            "resistances": res
        })

    print("="*75)

    # 棒グラフによる可視化（文脈による抵抗の逆転劇）
    plt.figure(figsize=(14, 8))
    x = np.arange(len(scenarios))
    width = 0.25

    matrix_surprisal = np.array(matrix_surprisal) # (4, 3)

    plt.bar(x - width, matrix_surprisal[:, 0], width, label="apple ➔ orange (Fruit)", color="#ff9800", edgecolor="black")
    plt.bar(x, matrix_surprisal[:, 1], width, label="apple ➔ rose (Botanical/Rose)", color="#e91e63", edgecolor="black")
    plt.bar(x + width, matrix_surprisal[:, 2], width, label="apple ➔ police (Law/Crime)", color="#2196f3", edgecolor="black")

    plt.ylabel("Contextual Resistance (Surprisal Resistance in Bits, Lower = Superconducting)", fontsize=11, fontweight="bold")
    plt.title("Dynamic Resistance Fluctuation by Context (Semiconductor Gate Effect)\nLower Resistance = Circuits Open and Current Surges", fontsize=13, fontweight="bold")
    plt.xticks(x, [s["name"] for s in scenarios], fontsize=9, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(fontsize=10, loc="upper right")

    # 注釈
    plt.annotate("💥 Rose Resistance Drops!\n(Botanical Connection Opens)", xy=(2, matrix_surprisal[2, 1]), xytext=(1.8, matrix_surprisal[2, 1] + 2),
                 arrowprops=dict(facecolor='magenta', shrink=0.08, width=1.5, headwidth=6), fontweight='bold', color='darkmagenta')

    plt.annotate("🚨 Police Resistance Crashes!\n(Investigation Circuit Surges)", xy=(3 + width, matrix_surprisal[3, 2]), xytext=(2.6, matrix_surprisal[3, 2] - 3),
                 arrowprops=dict(facecolor='blue', shrink=0.08, width=1.5, headwidth=6), fontweight='bold', color='darkblue')

    plt.tight_layout()
    out_png = "distillation_experiments/context_modulated_circuit_comparison.png"
    plt.savefig(out_png, dpi=300)
    print(f"\n[+] 文脈可変抵抗比較グラフを保存しました: {out_png}")

    import shutil
    art_path = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5/context_modulated_circuit_comparison.png"
    shutil.copyfile(out_png, art_path)
    print(f"[+] アーティファクトディレクトリにコピー完了: {art_path}")

    # JSON保存
    with open("distillation_experiments/context_circuit_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    run_context_experiment()
