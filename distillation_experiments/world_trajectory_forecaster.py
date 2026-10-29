"""
World Trajectory Forecaster (WTF) - 概念地形を用いた純粋未来予測エンジン
---------------------------------------------------------------------
Web検索や現実のニュースから得られる複数の観測事象（キーワード群）を注入し、
単語が持つ全層フライトパス（因果の運動量）を積分して合成力場を形成。
思考原野上に数千個の思考粒子を散布し、ポテンシャル勾配に沿って落下させることで、
「まだ起きていないが不可避的に引きずり込まれる未来の谷底（アトラクター）」を特定・言語化する。
"""

import sys
import os
import json
import torch
import numpy as np

# Windowsコンソール文字化け・絵文字対策
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from transformers import AutoModelForCausalLM, AutoTokenizer

class WorldTrajectoryForecaster:
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
        self.num_layers = self.model.config.num_hidden_layers
        print(f"[+] ロード完了: 総層数 = {self.num_layers}, デバイス = {self.device}")

        # 語彙埋め込みの正規化キャッシュ（逆引き用）
        with torch.no_grad():
            emb = self.model.get_input_embeddings().weight.data.to(torch.float32)
            self.vocab_norms = emb / (torch.norm(emb, dim=-1, keepdim=True) + 1e-8)
            self.vocab_size = emb.shape[0]

    def extract_flightpath(self, phrase: str) -> torch.Tensor:
        """
        単語・フレーズの全層フライトパス（因果の推進力ベクトル）を抽出。
        始層(Layer 0)から最終層(Layer 32)までの状態変化ベクトル Δh を積分。
        """
        inputs = self.tokenizer(phrase, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # (num_layers + 1, batch, seq, dim)
        
        # 最終トークンの各層軌跡
        layer_vecs = [h[0, -1, :].to(torch.float32) for h in hidden_states]
        
        # 軌道の累積変位（全工程の推進ベクトル）
        # 単なる最終層 - 初層 だけでなく、各層間の差分ベクトルの曲率も反映
        deltas = [layer_vecs[i+1] - layer_vecs[i] for i in range(len(layer_vecs)-1)]
        flightpath = torch.stack(deltas).sum(dim=0)
        
        # 正規化
        return flightpath / (torch.norm(flightpath) + 1e-8)

    def simulate_future(self, current_events: list[str], num_particles: int = 5000, steps: int = 40):
        """
        複数事象の合成力線を思考原野に解き放ち、思考粒子群の重力落下により未来アトラクターを特定。
        """
        print(f"\n========================================================")
        print(f"[EVENT] Initial Events: {current_events}")
        print(f"========================================================")

        event_forces = []
        input_token_ids = set()
        
        for ev in current_events:
            force = self.extract_flightpath(ev)
            event_forces.append(force)
            # 入力語彙のトークンIDを記録（自己想起・同義語ループの排除用）
            tokens = self.tokenizer.encode(ev, add_special_tokens=False)
            for t in tokens:
                input_token_ids.add(t)

        # 複数事象の直交干渉合成（合力場）
        composite_force = torch.stack(event_forces).mean(dim=0)
        composite_force = composite_force / (torch.norm(composite_force) + 1e-8)

        # 思考粒子群の散布（初期位置: 現在の合力場 + 不確実性のゆらぎノイズ）
        dim = composite_force.shape[0]
        particles = composite_force.unsqueeze(0).repeat(num_particles, 1)
        noise = torch.randn_like(particles) * 0.08  # 探索の広がり
        particles = particles + noise
        particles = particles / (torch.norm(particles, dim=-1, keepdim=True) + 1e-8)

        # 物理シミュレーション（運動量・摩擦・勾配落下）
        velocity = composite_force.unsqueeze(0).repeat(num_particles, 1) * 0.25
        friction = 0.88
        
        # ポテンシャル場での時間発展
        for step in range(steps):
            # 慣性移動
            particles = particles + velocity
            velocity = velocity * friction
            
            # 超球面上への拘束
            particles = particles / (torch.norm(particles, dim=-1, keepdim=True) + 1e-8)

        # 粒子群が到達した谷底（重心アトラクター）
        future_attractor = particles.mean(dim=0)
        future_attractor = future_attractor / (torch.norm(future_attractor) + 1e-8)

        # 谷底の座標を全語彙空間へ逆射影
        # 各語彙とのコサイン類似度（干渉度）
        similarities = torch.mv(self.vocab_norms, future_attractor)

        # 上位候補の抽出
        topk = torch.topk(similarities, k=100)
        
        future_outcomes = []
        seen_words = set()
        
        for idx, score in zip(topk.indices.tolist(), topk.values.tolist()):
            if idx in input_token_ids:
                continue
            
            word = self.tokenizer.decode([idx]).strip().lower()
            
            # クリーニング: 記号・短小トークン・サブワード断片・入力語の重複排除
            if len(word) < 3 or not word.isascii() or not word.isalpha():
                continue
            if any(ev_w in word or word in ev_w for ev in current_events for ev_w in ev.lower().split()):
                continue
            if word in seen_words:
                continue

            seen_words.add(word)
            future_outcomes.append({
                "concept": word,
                "attractor_gravity": float(score)
            })

            if len(future_outcomes) >= 8:
                break

        return {
            "initial_events": current_events,
            "future_attractors": future_outcomes
        }

def run_future_prediction_suite():
    forecaster = WorldTrajectoryForecaster("Qwen/Qwen2.5-7B")

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

    all_results = []

    for sc in scenarios:
        print(f"\n\n>>> 実行中: {sc['theme']}")
        res = forecaster.simulate_future(sc["events"], num_particles=8000, steps=45)
        res["theme"] = sc["theme"]
        all_results.append(res)
        
        print("\n[FUTURE ATTRACTOR] Inevitable Consequences / Spillovers:")
        for rank, item in enumerate(res["future_attractors"], 1):
            print(f"  {rank}. Concept: {item['concept']:<15} (Gravity Score: {item['attractor_gravity']:.4f})")

    # 結果のJSON保存
    output_path = "distillation_experiments/future_forecast_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n[+] 全実験結果を保存しました: {output_path}")

if __name__ == "__main__":
    run_future_prediction_suite()
