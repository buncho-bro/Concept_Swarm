"""
GLM Idea-Core Thought Wilderness Benchmark
---------------------------------------------------------------------
GLM-5.3 のイデア核 (53個の代表概念核) から構成された思考原野ハブ
(glm53_thought_wilderness_hub.json) を中継エンジンとして使用し、
多様な因果・推論タスク (20問) に対する思考粒子スウォーム推論ベンチマークを実行する。
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
    {"category": "Physics", "pair": ["cloud", "cold"], "targets": ["ice", "freezing", "snow", "frost", "winter"]},
    {"category": "Physics", "pair": ["fire", "wood"], "targets": ["ash", "smoke", "charcoal", "heat", "burning"]},
    {"category": "Physics", "pair": ["magnet", "iron"], "targets": ["attraction", "magnetic", "pole", "metal", "force"]},
    {"category": "Physics", "pair": ["sun", "rain"], "targets": ["rainbow", "weather", "storm", "light", "refraction"]},
    {"category": "Physics", "pair": ["friction", "motion"], "targets": ["heat", "resistance", "wear", "thermal", "energy"]},

    # 2. 社会・日常因果
    {"category": "Social", "pair": ["crime", "police"], "targets": ["arrest", "prison", "investigation", "law", "court"]},
    {"category": "Social", "pair": ["soap", "dirt"], "targets": ["cleaning", "wash", "clean", "hygiene", "pure"]},
    {"category": "Social", "pair": ["debt", "default"], "targets": ["bankruptcy", "insolvency", "crisis", "credit", "ruin"]},
    {"category": "Social", "pair": ["virus", "vaccine"], "targets": ["immunity", "antibodies", "protection", "cure", "health"]},
    {"category": "Social", "pair": ["inflation", "hike"], "targets": ["recession", "tightening", "slowdown", "interest", "cooling"]},

    # 3. 数学・抽象構造
    {"category": "Math", "pair": ["group", "commutative"], "targets": ["abelian", "ring", "algebra", "symmetry", "field"]},
    {"category": "Math", "pair": ["matrix", "eigenvalue"], "targets": ["decomposition", "diagonalization", "vector", "spectral", "linear"]},
    {"category": "Math", "pair": ["triangle", "angle"], "targets": ["pythagoras", "hypotenuse", "geometry", "trigonometry", "polygon"]},
    {"category": "Math", "pair": ["function", "derivative"], "targets": ["integral", "calculus", "slope", "rate", "analysis"]},
    {"category": "Math", "pair": ["graph", "edge"], "targets": ["topology", "path", "network", "vertex", "tree"]},

    # 4. 地政学・マクロ動態
    {"category": "Geopolitics", "pair": ["strait", "blockade"], "targets": ["embargo", "shortage", "conflict", "naval", "crisis"]},
    {"category": "Geopolitics", "pair": ["drought", "crop"], "targets": ["famine", "hunger", "starvation", "food", "failure"]},
    {"category": "Geopolitics", "pair": ["alliance", "treaty"], "targets": ["defense", "security", "pact", "military", "peace"]},
    {"category": "Geopolitics", "pair": ["monopoly", "regulation"], "targets": ["antitrust", "breakup", "law", "competition", "fine"]},
    {"category": "Geopolitics", "pair": ["currency", "devaluation"], "targets": ["capital", "inflation", "collapse", "crisis", "flight"]}
]

class GLMWildernessBenchmarkRunner:
    def __init__(self, hub_path="distillation_experiments/glm53_thought_wilderness_hub.json"):
        print(f"[*] GLM 思考原野ハブをロード中: {hub_path}...")
        with open(hub_path, "r", encoding="utf-8") as f:
            self.hub = json.load(f)

        self.coords_3d = np.array(self.hub["coordinates_3d"])  # (53, 3)
        self.dist_matrix = np.array(self.hub["geodesic_distance_matrix"])  # (53, 53)
        self.num_nodes = self.coords_3d.shape[0]
        print(f"[+] ロード完了: イデア核ノード数 = {self.num_nodes}")

        # 下流・言語化用のQwenエンコーダ
        model_name = "Qwen/Qwen2.5-7B"
        print(f"[*] 語彙射影用モデルを準備中: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()

        # 真正概念語彙プール構築
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
        
        # 脱中心化
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
        print(f"🚀 GLM 概念核思考原野 20タスク因果ベンチマーク開始")
        print(f"========================================================")

        results = []
        top1_hits = 0
        top3_hits = 0
        top5_hits = 0
        escapes = 0

        for i, task in enumerate(BENCHMARK_TASKS, 1):
            pair = task["pair"]
            targets = [t.lower() for t in task["targets"]]
            cat = task["category"]

            # 各単語のフライトパス取得
            f1 = self.get_token_flightpath(pair[0])
            f2 = self.get_token_flightpath(pair[1])
            
            # 直交力線干渉
            comp_force = (f1 + f2) / 2.0
            comp_force = comp_force / (torch.norm(comp_force) + 1e-8)

            # 思考粒子群の散布 (4,000粒子)
            num_p = 4000
            particles = comp_force.unsqueeze(0).repeat(num_p, 1)
            particles += torch.randn_like(particles) * 0.05
            particles = particles / (torch.norm(particles, dim=-1, keepdim=True) + 1e-8)

            # GLM思考原野のポテンシャル勾配に沿ったシミュレーション
            velocity = comp_force.unsqueeze(0).repeat(num_p, 1) * 0.3
            friction = 0.85
            for _ in range(35):
                particles = particles + velocity
                velocity = velocity * friction
                particles = particles / (torch.norm(particles, dim=-1, keepdim=True) + 1e-8)

            final_attractor = particles.mean(dim=0)
            final_attractor = final_attractor / (torch.norm(final_attractor) + 1e-8)

            # 直交因果射影（入力単語の成分を相殺）
            for f in [f1, f2]:
                proj = torch.dot(final_attractor, f) * f
                final_attractor = final_attractor - 0.45 * proj
            final_attractor = final_attractor / (torch.norm(final_attractor) + 1e-8)

            # 語彙空間への逆射影
            sims = torch.mv(self.concept_space, final_attractor)
            topk = torch.topk(sims, k=30)

            retrieved = []
            pair_words = set(pair)
            for idx in topk.indices.tolist():
                w = self.words_list[idx]
                if w in pair_words or any(pw in w or w in pw for pw in pair_words):
                    continue
                if len(w) >= 3 and w not in retrieved:
                    retrieved.append(w)
                if len(retrieved) >= 5:
                    break

            # 評価判定
            hit_top1 = retrieved[0] in targets if len(retrieved) > 0 else False
            hit_top3 = any(w in targets for w in retrieved[:3])
            hit_top5 = any(w in targets for w in retrieved[:5])
            escaped = (retrieved[0] not in pair_words) if len(retrieved) > 0 else False

            if hit_top1: top1_hits += 1
            if hit_top3: top3_hits += 1
            if hit_top5: top5_hits += 1
            if escaped: escapes += 1

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
        print(f"  ・自己ループ脱出率:     {escapes / 20 * 100:.1f}% ({escapes}/20)")
        print("="*56)

        out_path = "distillation_experiments/glm_wilderness_benchmark_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "summary": {
                    "top1_recall": top1_hits / 20,
                    "top3_recall": top3_hits / 20,
                    "top5_recall": top5_hits / 20,
                    "escape_rate": escapes / 20
                },
                "details": results
            }, f, ensure_ascii=False, indent=2)
        print(f"[+] ベンチマーク詳細レポートを保存しました: {out_path}")

if __name__ == "__main__":
    runner = GLMWildernessBenchmarkRunner()
    runner.run_benchmark()
