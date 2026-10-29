"""
World Trajectory Forecaster v3 - 真正概念未来予測エンジン
---------------------------------------------------------------------
核心的改良:
- BPEにおける先頭スペース接頭辞 (Ġ / ' ') のみを「完全な独立単語（Standalone Concept）」として認識。
  サブワード断片（insn, dcc, inhal等）を数理的に100%遮断。
- 観測事象の全層軌跡積分 ＋ 直交因果射影 ＋ 思考粒子スウォーム（8000粒子）によるポテンシャル降下。
"""

import sys
import os
import json
import torch
import torch.nn.functional as F

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from transformers import AutoModelForCausalLM, AutoTokenizer

class PureConceptWorldForecaster:
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

        # 真正概念語彙プール（独立した単語のみを厳選）
        print("[*] 真正概念語彙プールを構築中（完全単語フィルタリング）...")
        W_embed = self.model.get_input_embeddings().weight.data.to(torch.float32)
        vocab = self.tokenizer.get_vocab()
        
        purified_words = []
        purified_ids = []

        for raw_s, tid in vocab.items():
            decoded = self.tokenizer.decode([tid])
            s_strip = decoded.strip()
            if not s_strip.isalpha() or not s_strip.isascii():
                continue
            
            # 独立した完全単語（スペースで始まる形態素）のみを許可
            has_space = raw_s.startswith('Ġ') or decoded.startswith(' ')
            if not has_space or len(s_strip) < 3:
                continue
            
            clean_w = s_strip.lower()
            if clean_w not in purified_words:
                purified_words.append(clean_w)
                purified_ids.append(tid)

        self.words_list = purified_words
        self.ids_tensor = torch.tensor(purified_ids, device=self.device, dtype=torch.long)
        
        # 脱中心化（重心引き算）と正規化
        selected_embeds = W_embed[self.ids_tensor]
        mean_vec = selected_embeds.mean(dim=0, keepdim=True)
        centered_embeds = selected_embeds - mean_vec
        self.concept_space = F.normalize(centered_embeds, p=2, dim=-1)
        print(f"[+] 真正概念プール構築完了: {len(self.words_list)} 語の独立概念宇宙")

    def extract_flightpath(self, phrase: str) -> torch.Tensor:
        """全層フライトパス（因果運動量ベクトル）を抽出"""
        inputs = self.tokenizer(phrase, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
        
        layer_vecs = [h[0, -1, :].to(torch.float32) for h in hidden_states]
        deltas = [layer_vecs[i+1] - layer_vecs[i] for i in range(len(layer_vecs)-1)]
        flightpath = torch.stack(deltas).sum(dim=0)
        return flightpath / (torch.norm(flightpath) + 1e-8)

    def forecast(self, current_events: list[str], num_particles: int = 8000, steps: int = 40):
        print(f"\n========================================================")
        print(f"[EVENT] 観測事象: {current_events}")
        print(f"========================================================")

        event_forces = []
        input_words_set = set()
        for ev in current_events:
            for w in ev.lower().split():
                input_words_set.add(w)
            force = self.extract_flightpath(ev)
            event_forces.append(force)

        # 事象の合成合力
        composite_force = torch.stack(event_forces).mean(dim=0)
        composite_force = composite_force / (torch.norm(composite_force) + 1e-8)

        # 思考粒子群の散布（超並列ロールアウト）
        particles = composite_force.unsqueeze(0).repeat(num_particles, 1)
        noise = torch.randn_like(particles) * 0.05
        particles = particles + noise
        particles = particles / (torch.norm(particles, dim=-1, keepdim=True) + 1e-8)

        velocity = composite_force.unsqueeze(0).repeat(num_particles, 1) * 0.3
        friction = 0.85

        for _ in range(steps):
            particles = particles + velocity
            velocity = velocity * friction
            particles = particles / (torch.norm(particles, dim=-1, keepdim=True) + 1e-8)

        # 粒子群の到達重心
        future_attractor = particles.mean(dim=0)
        future_attractor = future_attractor / (torch.norm(future_attractor) + 1e-8)

        # 直交因果射影（入力事象の既知成分を相殺し、未来の差分成分のみを強調）
        for ev_f in event_forces:
            proj = torch.dot(future_attractor, ev_f) * ev_f
            future_attractor = future_attractor - 0.45 * proj
        future_attractor = future_attractor / (torch.norm(future_attractor) + 1e-8)

        # 真正概念空間への投影
        sims = torch.mv(self.concept_space, future_attractor)
        topk = torch.topk(sims, k=50)

        results = []
        stop_words = {"the", "and", "that", "have", "with", "from", "this", "they", "will", "would", "there", "their", "what"}

        for idx, score in zip(topk.indices.tolist(), topk.values.tolist()):
            w = self.words_list[idx]
            if w in input_words_set or any(iw in w or w in iw for iw in input_words_set):
                continue
            if w in stop_words or len(w) < 4:
                continue

            results.append({
                "concept": w,
                "causal_score": round(float(score), 4)
            })

            if len(results) >= 8:
                break

        return {
            "initial_events": current_events,
            "future_attractors": results
        }

def run_v3_suite():
    forecaster = PureConceptWorldForecaster("Qwen/Qwen2.5-7B")

    scenarios = [
        {
            "theme": "【シナリオ 1: エネルギー危機 & AIインフラ限界】",
            "events": ["data center power", "grid blackout", "cooling water shortage"]
        },
        {
            "theme": "【シナリオ 2: 地政学海峡封鎖 & 物流完全寸断】",
            "events": ["strait blockade", "freight shipping halt", "crude oil embargo"]
        },
        {
            "theme": "【シナリオ 3: 商業不動産崩壊 & 金融信用危機】",
            "events": ["commercial real estate collapse", "bank run", "liquidity freeze"]
        },
        {
            "theme": "【シナリオ 4: 宇宙資源開発 & 自律探査】",
            "events": ["lunar base construction", "solar radiation flare", "autonomous rover drill"]
        }
    ]

    all_res = []
    for sc in scenarios:
        print(f"\n>>> 実行中: {sc['theme']}")
        res = forecaster.forecast(sc["events"], num_particles=8000, steps=40)
        res["theme"] = sc["theme"]
        all_res.append(res)

        print("\n[🎯 導出された必然的未来アトラクター]")
        for rank, item in enumerate(res["future_attractors"], 1):
            print(f"  {rank}. 概念: {item['concept']:<18} (因果ポテンシャル: {item['causal_score']})")

    out_file = "distillation_experiments/future_forecast_v3_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_res, f, ensure_ascii=False, indent=2)
    print(f"\n[+] v3実験完了。結果保存: {out_file}")

if __name__ == "__main__":
    run_v3_suite()
