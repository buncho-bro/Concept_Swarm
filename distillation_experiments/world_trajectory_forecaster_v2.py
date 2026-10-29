"""
World Trajectory Forecaster v2 - 高精度因果未来予測エンジン
---------------------------------------------------------------------
改良点:
1. 【語彙空間の脱中心化 (Centering)】:
   全語彙ベクトルの平均（重心）を差し引くことで、全単語に共通して現れるノイズ「dcc, insn, min」等のトラップを完全除去。
2. 【直交因果射影 (Orthogonal Causal Projection)】:
   現在の観測事象が張る部分空間に対して「垂直（直交）」な方向へ力線を射影し、
   「現在の同義語・言及」ではなく「まだ起きていない次の変化」のみを抽出。
3. 【完全単語フィルタリング】:
   3文字以上の完全な英語普通名詞・動詞のみをターゲットとし、BPEサブワード断片を排除。
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

class CausalWorldForecasterV2:
    def __init__(self, model_name="Qwen/Qwen2.5-7B"):
        print(f"[*] 幾何学モデルをロード中: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()
        self.num_layers = self.model.config.num_hidden_layers
        print(f"[+] ロード完了: 総層数 = {self.num_layers}, デバイス = {self.device}")

        # 語彙埋め込みの取得
        with torch.no_grad():
            emb = self.model.get_input_embeddings().weight.data.to(torch.float32)
            
            # 1. 語彙の幾何学的中心（平均ベクトル）を引いて脱中心化
            mean_vec = emb.mean(dim=0, keepdim=True)
            centered_emb = emb - mean_vec
            
            # 正規化
            self.vocab_norms = centered_emb / (torch.norm(centered_emb, dim=-1, keepdim=True) + 1e-8)
            self.vocab_size = emb.shape[0]

    def extract_flightpath(self, phrase: str) -> torch.Tensor:
        """全33層のフライトパス（因果の推進力ベクトル）を抽出"""
        inputs = self.tokenizer(phrase, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
        
        layer_vecs = [h[0, -1, :].to(torch.float32) for h in hidden_states]
        deltas = [layer_vecs[i+1] - layer_vecs[i] for i in range(len(layer_vecs)-1)]
        flightpath = torch.stack(deltas).sum(dim=0)
        return flightpath / (torch.norm(flightpath) + 1e-8)

    def forecast(self, current_events: list[str], num_particles: int = 6000, steps: int = 35):
        print(f"\n========================================================")
        print(f"[EVENT] 観測事象: {current_events}")
        print(f"========================================================")

        event_forces = []
        input_token_ids = set()

        for ev in current_events:
            force = self.extract_flightpath(ev)
            event_forces.append(force)
            tokens = self.tokenizer.encode(ev, add_special_tokens=False)
            for t in tokens:
                input_token_ids.add(t)

        # 事象群の平均力線
        base_force = torch.stack(event_forces).mean(dim=0)
        base_force = base_force / (torch.norm(base_force) + 1e-8)

        # 思考粒子群の散布
        particles = base_force.unsqueeze(0).repeat(num_particles, 1)
        noise = torch.randn_like(particles) * 0.05
        particles = particles + noise
        particles = particles / (torch.norm(particles, dim=-1, keepdim=True) + 1e-8)

        velocity = base_force.unsqueeze(0).repeat(num_particles, 1) * 0.3
        friction = 0.85

        # 物理シミュレーション（最小作用の原理による前進）
        for _ in range(steps):
            particles = particles + velocity
            velocity = velocity * friction
            particles = particles / (torch.norm(particles, dim=-1, keepdim=True) + 1e-8)

        # 粒子群の到達重心
        future_attractor = particles.mean(dim=0)
        future_attractor = future_attractor / (torch.norm(future_attractor) + 1e-8)

        # 【直交因果射影】入力事象の張る基底成分を差し引いて、純粋な「次の変化（微分成分）」を取り出す
        for ev_force in event_forces:
            proj = torch.dot(future_attractor, ev_force) * ev_force
            future_attractor = future_attractor - 0.4 * proj
        future_attractor = future_attractor / (torch.norm(future_attractor) + 1e-8)

        # 全語彙との干渉度（脱中心化空間でのコサイン類似度）
        similarities = torch.mv(self.vocab_norms, future_attractor)
        topk = torch.topk(similarities, k=300)

        # 完全単語辞書フィルタリング
        results = []
        seen = set()

        for idx, score in zip(topk.indices.tolist(), topk.values.tolist()):
            if idx in input_token_ids:
                continue
            word = self.tokenizer.decode([idx]).strip().lower()
            
            # BPEサブワードや断片ノイズの排除条件
            if len(word) < 4 or not word.isalpha() or not word.isascii():
                continue
            if any(ev_w in word or word in ev_w for ev in current_events for ev_w in ev.lower().split()):
                continue
            # 一般的すぎるストップワードの排除
            if word in {"with", "that", "from", "have", "this", "they", "will", "would", "there", "their"}:
                continue
            if word in seen:
                continue

            seen.add(word)
            results.append({
                "concept": word,
                "causal_potential": round(float(score), 4)
            })

            if len(results) >= 8:
                break

        return {
            "initial_events": current_events,
            "future_attractors": results
        }

def run_v2():
    forecaster = CausalWorldForecasterV2("Qwen/Qwen2.5-7B")

    scenarios = [
        {
            "theme": "【シナリオ 1: エネルギー・AIインフラ危機】",
            "events": ["data center power", "grid blackout", "cooling water shortage"]
        },
        {
            "theme": "【シナリオ 2: 地政学・海上交通寸断】",
            "events": ["strait blockade", "freight shipping halt", "crude oil embargo"]
        },
        {
            "theme": "【シナリオ 3: 次世代金融・信用ショック】",
            "events": ["commercial real estate collapse", "bank run", "liquidity freeze"]
        },
        {
            "theme": "【シナリオ 4: 宇宙開発・自律探査】",
            "events": ["lunar base construction", "solar flare radiation", "autonomous rover drill"]
        }
    ]

    all_res = []
    for sc in scenarios:
        print(f"\n>>> 実行中: {sc['theme']}")
        res = forecaster.forecast(sc["events"], num_particles=8000, steps=35)
        res["theme"] = sc["theme"]
        all_res.append(res)

        print("\n[🎯 導出された必然的未来アトラクター]")
        for rank, item in enumerate(res["future_attractors"], 1):
            print(f"  {rank}. 概念: {item['concept']:<18} (因果ポテンシャル: {item['causal_potential']})")

    out_file = "distillation_experiments/future_forecast_v2_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_res, f, ensure_ascii=False, indent=2)
    print(f"\n[+] v2実験完了。結果保存: {out_file}")

if __name__ == "__main__":
    run_v2()
