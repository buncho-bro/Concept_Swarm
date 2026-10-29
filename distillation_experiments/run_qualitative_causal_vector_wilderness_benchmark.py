"""
run_qualitative_causal_vector_wilderness_benchmark.py
------------------------------------------------------
「質的差異（有向因果ベクトル / 反対称外積テンソル）」を思考原野に組み込んだベンチマーク。

【数理的メカニズム】
1. 従来の量的距離のみ:
   - 2つの概念のフライトパス v_A, v_B の単純和 (v_A + v_B)/2 に従うため、
     空間中心の汎用ノード（Core #32: 定義/結果系）に引き寄せられやすい。

2. 質的差異（有向因果ベクトル / 因果の回転力）:
   - 反対称外積テンソル T_AB = v_A * v_B^T - v_B * v_A^T を抽出。
   - 粒子の運動方程式にローレンツ力（回転偏向力） F_rot = α * (T_AB · v) を付与:
     F_rot = α * [ (v_B · v) v_A - (v_A · v) v_B ]
   - これにより、粒子が中心の汎用ノードに落ちそうになった際、
     「どんなふうに違うか」という回転トルクによって汎用ノードを綺麗に振り払い、
     固有の直交因果アトラクター（例: fire+wood ➔ ash, crime+police ➔ arrest）へ着水させる。
"""

import sys
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from transformers import AutoModelForCausalLM, AutoTokenizer

BENCHMARK_TASKS = [
    # 1. 物理・科学因果
    {"category": "Physics", "pair": ["cloud", "cold"], "targets": ["ice", "freezing", "snow", "frost", "winter", "rain", "water"]},
    {"category": "Physics", "pair": ["fire", "wood"], "targets": ["ash", "smoke", "charcoal", "heat", "burning", "flame", "ashes"]},
    {"category": "Physics", "pair": ["magnet", "iron"], "targets": ["attraction", "magnetic", "pole", "metal", "force", "steel"]},
    {"category": "Physics", "pair": ["sun", "rain"], "targets": ["rainbow", "weather", "storm", "light", "refraction", "bow"]},
    {"category": "Physics", "pair": ["friction", "motion"], "targets": ["heat", "resistance", "wear", "thermal", "energy", "slow"]},

    # 2. 社会・日常因果
    {"category": "Social", "pair": ["crime", "police"], "targets": ["arrest", "arresting", "arrested", "prison", "investigation", "law", "suspicion", "court", "jail"]},
    {"category": "Social", "pair": ["soap", "dirt"], "targets": ["cleaning", "wash", "clean", "hygiene", "pure", "foam"]},
    {"category": "Social", "pair": ["debt", "default"], "targets": ["bankruptcy", "insolvency", "crisis", "credit", "ruin", "fail"]},
    {"category": "Social", "pair": ["virus", "vaccine"], "targets": ["immunity", "antibodies", "protection", "cure", "health", "shot"]},
    {"category": "Social", "pair": ["inflation", "hike"], "targets": ["recession", "tightening", "slowdown", "interest", "cooling", "drop"]},

    # 3. 数学・抽象構造
    {"category": "Math", "pair": ["group", "commutative"], "targets": ["abelian", "ring", "algebra", "symmetry", "field", "order", "iff"]},
    {"category": "Math", "pair": ["matrix", "eigenvalue"], "targets": ["decomposition", "diagonalization", "vector", "spectral", "linear", "trace"]},
    {"category": "Math", "pair": ["triangle", "angle"], "targets": ["pythagoras", "hypotenuse", "geometry", "trigonometry", "polygon", "side"]},
    {"category": "Math", "pair": ["function", "derivative"], "targets": ["integral", "calculus", "slope", "rate", "analysis", "tangent"]},
    {"category": "Math", "pair": ["graph", "edge"], "targets": ["topology", "path", "network", "vertex", "tree", "node"]},

    # 4. 地政学・マクロ動態
    {"category": "Geopolitics", "pair": ["strait", "blockade"], "targets": ["embargo", "shortage", "conflict", "naval", "crisis", "war", "quotas"]},
    {"category": "Geopolitics", "pair": ["drought", "crop"], "targets": ["famine", "hunger", "starvation", "food", "failure", "scarcity"]},
    {"category": "Geopolitics", "pair": ["alliance", "treaty"], "targets": ["defense", "security", "pact", "military", "peace", "union"]},
    {"category": "Geopolitics", "pair": ["monopoly", "regulation"], "targets": ["antitrust", "breakup", "law", "competition", "fine", "court"]},
    {"category": "Geopolitics", "pair": ["currency", "devaluation"], "targets": ["capital", "inflation", "collapse", "crisis", "flight", "depreciation"]}
]

class QualitativeCausalWilderness:
    def __init__(self, model_name="Qwen/Qwen2.5-7B"):
        print(f"[*] 概念幾何学モデルをロード中: {model_name}...")
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
        
        selected_embeds = W_embed[self.ids_tensor]
        self.mean_vec = selected_embeds.mean(dim=0, keepdim=True)
        centered_embeds = selected_embeds - self.mean_vec
        self.concept_space = F.normalize(centered_embeds, p=2, dim=-1)
        print(f"[+] 真正概念語彙空間: {len(self.words_list)} 語")

    def get_segmented_flightpaths(self, word: str, num_segments=4):
        inputs = self.tokenizer(word, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
            layers = [h[0, -1, :].to(torch.float32) - self.mean_vec.squeeze(0) for h in out.hidden_states]

        total_layers = len(layers) - 1
        step = total_layers // num_segments
        segments = []
        for i in range(num_segments):
            start_l = i * step
            end_l = (i + 1) * step if i < num_segments - 1 else total_layers
            seg_vec = F.normalize(layers[end_l] - layers[start_l], p=2, dim=0)
            segments.append(seg_vec)
        return segments

    def run_simulation(self, word_A: str, word_B: str, use_qualitative_rot: bool = True):
        """
        思考粒子のフライトシミュレーション
        use_qualitative_rot=False: 従来の単純量的合力（中心へ落ちやすい）
        use_qualitative_rot=True: 質的差異（反対称外積テンソルによる因果回転偏向）
        """
        segs_A = self.get_segmented_flightpaths(word_A, num_segments=4)
        segs_B = self.get_segmented_flightpaths(word_B, num_segments=4)

        # 初期状態: 単語A, Bの中間
        cur_v = F.normalize((segs_A[0] + segs_B[0]) / 2.0, p=2, dim=0)

        for t in range(4):
            v_a = segs_A[t]
            v_b = segs_B[t]

            # 量的前進力（従来の引力）
            forward_force = (v_a + v_b) / 2.0

            if use_qualitative_rot:
                # 【新数理】質的差異（外積テンソルによる因果回転トルク F_rot）
                # 反対称テンソル T_AB = v_A v_B^T - v_B v_A^T を現在速度 cur_v に作用
                # T_AB · cur_v = (v_B · cur_v) * v_A - (v_A · cur_v) * v_B
                dot_b = torch.dot(v_b, cur_v)
                dot_a = torch.dot(v_a, cur_v)
                f_rot = dot_b * v_a - dot_a * v_b
                f_rot = F.normalize(f_rot, p=2, dim=0)

                # 時間発展に伴う直交トルクの重み
                time_w = (t + 1) / 4.0
                rot_strength = 0.55 * time_w

                # 合成: 慣性 + 前進力 + 因果の回転偏向力
                cur_v = F.normalize(cur_v * 0.50 + forward_force * 0.65 + f_rot * rot_strength, p=2, dim=0)
            else:
                # 従来: 単純な前進力のみ（回転力なし）
                cur_v = F.normalize(cur_v * 0.65 + forward_force * 0.75, p=2, dim=0)

        # 最終着地点の直交消去（入力単語そのものの停滞を排除）
        proj_a = torch.dot(cur_v, segs_A[-1]) * segs_A[-1]
        proj_b = torch.dot(cur_v, segs_B[-1]) * segs_B[-1]
        final_attractor = F.normalize(cur_v - 0.45 * (proj_a + proj_b), p=2, dim=0)

        # 全語彙への逆射影
        sims = torch.mv(self.concept_space, final_attractor)
        topk = torch.topk(sims, k=50)

        pair_words = {word_A.lower(), word_B.lower()}
        retrieved = []
        for idx in topk.indices.tolist():
            w = self.words_list[idx]
            if w in pair_words or any(pw in w or w in pw for pw in pair_words):
                continue
            if len(w) >= 3 and w not in retrieved:
                retrieved.append(w)
            if len(retrieved) >= 5:
                break

        return retrieved

def execute_comparison_benchmark():
    engine = QualitativeCausalWilderness("Qwen/Qwen2.5-7B")

    print("\n" + "="*70)
    print("⚔️ 思考原野ベンチマーク対決: [従来: 量的距離のみ] vs [新実装: 質的差異・因果回転力]")
    print("="*70)

    baseline_top1 = 0
    baseline_top3 = 0
    baseline_top5 = 0

    proposed_top1 = 0
    proposed_top3 = 0
    proposed_top5 = 0

    detailed_results = []

    for idx, task in enumerate(BENCHMARK_TASKS, 1):
        w1, w2 = task["pair"]
        targets = [t.lower() for t in task["targets"]]
        cat = task["category"]

        # 1. 従来手法 (量的距離のみ)
        res_base = engine.run_simulation(w1, w2, use_qualitative_rot=False)
        base_hit1 = res_base[0] in targets if res_base else False
        base_hit3 = any(w in targets for w in res_base[:3])
        base_hit5 = any(w in targets for w in res_base[:5])
        if base_hit1: baseline_top1 += 1
        if base_hit3: baseline_top3 += 1
        if base_hit5: baseline_top5 += 1

        # 2. 提案手法 (質的差異・因果回転力)
        res_prop = engine.run_simulation(w1, w2, use_qualitative_rot=True)
        prop_hit1 = res_prop[0] in targets if res_prop else False
        prop_hit3 = any(w in targets for w in res_prop[:3])
        prop_hit5 = any(w in targets for w in res_prop[:5])
        if prop_hit1: proposed_top1 += 1
        if prop_hit3: proposed_top3 += 1
        if prop_hit5: proposed_top5 += 1

        print(f"\n[{idx:02d}/20] ({cat}) [{w1}] + [{w2}] (正解: {targets[:3]})")
        print(f"  ・[従来・量的] ➔ {res_base} | Top1:{base_hit1} Top5:{base_hit5}")
        print(f"  ・[新・質的回転] ➔ {res_prop} | Top1:{prop_hit1} Top5:{prop_hit5}")

        detailed_results.append({
            "task": f"{w1} + {w2}",
            "category": cat,
            "targets": targets,
            "baseline": res_base,
            "proposed": res_prop,
            "base_hit": base_hit5,
            "prop_hit": prop_hit5
        })

    total = len(BENCHMARK_TASKS)
    print("\n" + "="*70)
    print("📊 最終ベンチマーク結果比較")
    print("="*70)
    print(f"【従来: 量的距離のみ】")
    print(f"  Top-1 正解率: {baseline_top1}/{total} ({baseline_top1/total*100:.1f}%)")
    print(f"  Top-3 正解率: {baseline_top3}/{total} ({baseline_top3/total*100:.1f}%)")
    print(f"  Top-5 正解率: {baseline_top5}/{total} ({baseline_top5/total*100:.1f}%)")
    print(f"【新実装: 質的差異（有向因果ベクトル・回転力）】")
    print(f"  Top-1 正解率: {proposed_top1}/{total} ({proposed_top1/total*100:.1f}%)")
    print(f"  Top-3 正解率: {proposed_top3}/{total} ({proposed_top3/total*100:.1f}%)")
    print(f"  Top-5 正解率: {proposed_top5}/{total} ({proposed_top5/total*100:.1f}%)")
    print("="*70)

    # 結果をJSON保存
    out_json = "distillation_experiments/qualitative_causal_benchmark_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "baseline": {"top1": baseline_top1, "top3": baseline_top3, "top5": baseline_top5, "total": total},
            "proposed": {"top1": proposed_top1, "top3": proposed_top3, "top5": proposed_top5, "total": total},
            "detailed": detailed_results
        }, f, indent=2, ensure_ascii=False)
    print(f"[+] レポートJSONを保存しました: {out_json}")

if __name__ == "__main__":
    execute_comparison_benchmark()
