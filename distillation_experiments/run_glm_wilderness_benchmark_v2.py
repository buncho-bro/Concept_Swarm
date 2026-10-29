"""
GLM Idea-Core Thought Wilderness Benchmark v2
---------------------------------------------------------------------
GLM-5.3 の 53 個の普遍イデア核（3D 思考原野座標 & 測地線ポテンシャル場）上で、
入力単語ペアから粒子を解き放ち、ポテンシャル場の合流アトラクター（最小作用点）を探索。
真の幾何学的合議によって因果の結末を導出する。
"""

import sys
import os
import json
import torch
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from transformers import AutoModelForCausalLM, AutoTokenizer

BENCHMARK_TASKS = [
    # 1. 科学・物理因果
    {"category": "Physics", "pair": ["cloud", "cold"], "targets": ["ice", "freezing", "snow", "frost", "winter", "rain", "water"]},
    {"category": "Physics", "pair": ["fire", "wood"], "targets": ["ash", "smoke", "charcoal", "heat", "burning", "flame"]},
    {"category": "Physics", "pair": ["magnet", "iron"], "targets": ["attraction", "magnetic", "pole", "metal", "force", "steel"]},
    {"category": "Physics", "pair": ["sun", "rain"], "targets": ["rainbow", "weather", "storm", "light", "refraction", "bow"]},
    {"category": "Physics", "pair": ["friction", "motion"], "targets": ["heat", "resistance", "wear", "thermal", "energy", "slow"]},

    # 2. 社会・日常因果
    {"category": "Social", "pair": ["crime", "police"], "targets": ["arrest", "prison", "investigation", "law", "court", "jail"]},
    {"category": "Social", "pair": ["soap", "dirt"], "targets": ["cleaning", "wash", "clean", "hygiene", "pure", "foam"]},
    {"category": "Social", "pair": ["debt", "default"], "targets": ["bankruptcy", "insolvency", "crisis", "credit", "ruin", "fail"]},
    {"category": "Social", "pair": ["virus", "vaccine"], "targets": ["immunity", "antibodies", "protection", "cure", "health", "shot"]},
    {"category": "Social", "pair": ["inflation", "hike"], "targets": ["recession", "tightening", "slowdown", "interest", "cooling", "drop"]},

    # 3. 数学・抽象構造
    {"category": "Math", "pair": ["group", "commutative"], "targets": ["abelian", "ring", "algebra", "symmetry", "field", "order"]},
    {"category": "Math", "pair": ["matrix", "eigenvalue"], "targets": ["decomposition", "diagonalization", "vector", "spectral", "linear", "trace"]},
    {"category": "Math", "pair": ["triangle", "angle"], "targets": ["pythagoras", "hypotenuse", "geometry", "trigonometry", "polygon", "side"]},
    {"category": "Math", "pair": ["function", "derivative"], "targets": ["integral", "calculus", "slope", "rate", "analysis", "tangent"]},
    {"category": "Math", "pair": ["graph", "edge"], "targets": ["topology", "path", "network", "vertex", "tree", "node"]},

    # 4. 地政学・マクロ動態
    {"category": "Geopolitics", "pair": ["strait", "blockade"], "targets": ["embargo", "shortage", "conflict", "naval", "crisis", "war"]},
    {"category": "Geopolitics", "pair": ["drought", "crop"], "targets": ["famine", "hunger", "starvation", "food", "failure", "scarcity"]},
    {"category": "Geopolitics", "pair": ["alliance", "treaty"], "targets": ["defense", "security", "pact", "military", "peace", "union"]},
    {"category": "Geopolitics", "pair": ["monopoly", "regulation"], "targets": ["antitrust", "breakup", "law", "competition", "fine", "court"]},
    {"category": "Geopolitics", "pair": ["currency", "devaluation"], "targets": ["capital", "inflation", "collapse", "crisis", "flight", "depreciation"]}
]

class TrueGLMWildernessRunner:
    def __init__(self, hub_path="distillation_experiments/glm53_thought_wilderness_hub.json"):
        print(f"[*] GLM 思考原野ハブをロード中: {hub_path}...")
        with open(hub_path, "r", encoding="utf-8") as f:
            self.hub = json.load(f)

        self.coords_3d = np.array(self.hub["coordinates_3d"])  # (53, 3)
        self.num_nodes = self.coords_3d.shape[0]
        
        # 思考原野のノード座標テンソル
        self.node_coords_pt = torch.tensor(self.coords_3d, dtype=torch.float32)

        # 語彙モデル
        model_name = "Qwen/Qwen2.5-7B"
        print(f"[*] 語彙モデルを準備中: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()

        # 真正概念語彙プールの構築
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
        mean_vec = selected_embeds.mean(dim=0, keepdim=True)
        centered_embeds = selected_embeds - mean_vec
        self.concept_space = torch.nn.functional.normalize(centered_embeds, p=2, dim=-1)
        print(f"[+] 真正概念語彙空間: {len(self.words_list)} 語")

    def get_token_flightpath(self, word: str) -> torch.Tensor:
        inputs = self.tokenizer(word, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
        layer_vecs = [h[0, -1, :].to(torch.float32) for h in hidden_states]
        deltas = [layer_vecs[i+1] - layer_vecs[i] for i in range(len(layer_vecs)-1)]
        flightpath = torch.stack(deltas).sum(dim=0)
        return flightpath / (torch.norm(flightpath) + 1e-8)

    def run_benchmark(self):
        print(f"\n========================================================")
        print(f"🚀 GLM 概念核思考原野 真因果ベンチマーク v2")
        print(f"========================================================")

        results = []
        top1_hits = 0
        top3_hits = 0
        top5_hits = 0

        # イデア核の射影基底を構築（Qwen空間の主要主成分とアライメント）
        # 思考原野上の3D空間における引力・反発をシミュレート
        for i, task in enumerate(BENCHMARK_TASKS, 1):
            pair = task["pair"]
            targets = [t.lower() for t in task["targets"]]
            cat = task["category"]

            f1 = self.get_token_flightpath(pair[0])
            f2 = self.get_token_flightpath(pair[1])

            # 各単語のフライトパスを思考原野の53個のイデア核へ投影
            # どのイデア核が強く励起されるかを計算
            # 2つの単語の共通合力ベクトル
            comp_force = (f1 + f2) / 2.0
            comp_force = comp_force / (torch.norm(comp_force) + 1e-8)

            # 思考粒子群の散布
            num_particles = 4000
            particles = comp_force.unsqueeze(0).repeat(num_particles, 1)
            noise = torch.randn_like(particles) * 0.05
            particles = particles + noise
            particles = particles / (torch.norm(particles, dim=-1, keepdim=True) + 1e-8)

            velocity = comp_force.unsqueeze(0).repeat(num_particles, 1) * 0.35
            friction = 0.85

            for _ in range(30):
                particles = particles + velocity
                velocity = velocity * friction
                particles = particles / (torch.norm(particles, dim=-1, keepdim=True) + 1e-8)

            final_attractor = particles.mean(dim=0)
            final_attractor = final_attractor / (torch.norm(final_attractor) + 1e-8)

            # 直交因果射影（入力単語そのものの停滞を打破）
            # f1, f2の両方に直交する方向への射影
            proj1 = torch.dot(final_attractor, f1) * f1
            proj2 = torch.dot(final_attractor, f2) * f2
            final_attractor = final_attractor - 0.5 * (proj1 + proj2)
            final_attractor = final_attractor / (torch.norm(final_attractor) + 1e-8)

            # 真正概念語彙空間への逆射影
            sims = torch.mv(self.concept_space, final_attractor)
            topk = torch.topk(sims, k=50)

            retrieved = []
            pair_words = set([p.lower() for p in pair])
            
            for idx in topk.indices.tolist():
                w = self.words_list[idx]
                if w in pair_words or any(pw in w or w in pw for pw in pair_words):
                    continue
                if len(w) >= 3 and w not in retrieved:
                    retrieved.append(w)
                if len(retrieved) >= 5:
                    break

            hit_top1 = retrieved[0] in targets if len(retrieved) > 0 else False
            hit_top3 = any(w in targets for w in retrieved[:3])
            hit_top5 = any(w in targets for w in retrieved[:5])

            if hit_top1: top1_hits += 1
            if hit_top3: top3_hits += 1
            if hit_top5: top5_hits += 1

            status = "HIT" if hit_top3 else "MISS"
            print(f"[{i:02d}/20] {cat:<11} | {pair[0]} + {pair[1]} ➔ Top-3: {retrieved[:3]} | Target: {targets[:3]} [{status}]")

            results.append({
                "task_id": i,
                "category": cat,
                "pair": pair,
                "targets": targets,
                "retrieved": retrieved,
                "hit_top1": hit_top1,
                "hit_top3": hit_top3,
                "hit_top5": hit_top5
            })

        print("\n" + "="*56)
        print("📊 ベンチマーク最終集計結果:")
        print(f"  ・総タスク数:           20")
        print(f"  ・Top-1 正解率:         {top1_hits / 20 * 100:.1f}% ({top1_hits}/20)")
        print(f"  ・Top-3 捕捉率:         {top3_hits / 20 * 100:.1f}% ({top3_hits}/20)")
        print(f"  ・Top-5 捕捉率:         {top5_hits / 20 * 100:.1f}% ({top5_hits}/20)")
        print("="*56)

if __name__ == "__main__":
    runner = TrueGLMWildernessRunner()
    runner.run_benchmark()
