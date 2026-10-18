import os
import sys
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import time
import json
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ==============================================================================
# 根本原因究明 & 改善策（Gated Levitation / Topic-Aligned）の検証ベンチマーク
# ==============================================================================

# 1. 現行の無条件強制 Hook (Fixed English Attractors, No Gate)
class UngatedHook:
    def __init__(self, attractors, gamma=0.08, sigma=0.25):
        self.attractors = attractors
        self.gamma = gamma
        self.sigma = sigma
        self.active = True

    def __call__(self, module, inputs, output):
        if not self.active or self.attractors is None:
            return output
        last_h = output[:, -1, :]
        h_norm = F.normalize(last_h, p=2, dim=-1)

        cos_sim = torch.matmul(h_norm, self.attractors.T)
        dist_sq = 2.0 - 2.0 * cos_sim
        pulls = torch.exp(- dist_sq / (2.0 * (self.sigma ** 2)))

        weights = F.softmax(pulls * 8.0, dim=-1)
        target_dir = torch.matmul(weights, self.attractors)
        target_dir = F.normalize(target_dir, p=2, dim=-1)

        tangent_nudge = target_dir - torch.sum(target_dir * h_norm, dim=-1, keepdim=True) * h_norm
        tangent_nudge = F.normalize(tangent_nudge, p=2, dim=-1)

        h_magnitude = torch.norm(last_h, p=2, dim=-1, keepdim=True)
        nudged_last_h = last_h + self.gamma * h_magnitude * tangent_nudge

        new_output = output.clone()
        new_output[:, -1, :] = nudged_last_h
        return new_output

# 2. 改善版: 磁界ゲート付き Hook (Gated Hook)
# コサイン類似度が threshold 未満なら引力は厳密に 0。関連するときのみ磁界が発動。
class GatedHook:
    def __init__(self, attractors, gamma=0.08, threshold=0.20, sigma=0.25):
        self.attractors = attractors
        self.gamma = gamma
        self.threshold = threshold
        self.sigma = sigma
        self.active = True
        self.interventions = 0
        self.total_tokens = 0

    def __call__(self, module, inputs, output):
        if not self.active or self.attractors is None:
            return output
        self.total_tokens += 1
        last_h = output[:, -1, :]
        h_norm = F.normalize(last_h, p=2, dim=-1)

        cos_sim = torch.matmul(h_norm, self.attractors.T) # (B, N)
        max_cos, best_idx = torch.max(cos_sim, dim=-1)

        # ゲート判定: 類似度が閾値未満なら磁力はゼロ（自然な生成を一切邪魔しない）
        if max_cos.item() < self.threshold:
            return output

        self.interventions += 1
        # 閾値を超えた分だけ滑らかに磁力を付与 (0.0 〜 1.0)
        gate_strength = (max_cos - self.threshold) / (1.0 - self.threshold)

        dist_sq = 2.0 - 2.0 * cos_sim
        pulls = torch.exp(- dist_sq / (2.0 * (self.sigma ** 2)))
        weights = F.softmax(pulls * 8.0, dim=-1)
        target_dir = torch.matmul(weights, self.attractors)
        target_dir = F.normalize(target_dir, p=2, dim=-1)

        tangent_nudge = target_dir - torch.sum(target_dir * h_norm, dim=-1, keepdim=True) * h_norm
        tangent_nudge = F.normalize(tangent_nudge, p=2, dim=-1)

        h_magnitude = torch.norm(last_h, p=2, dim=-1, keepdim=True)
        effective_gamma = self.gamma * gate_strength.unsqueeze(-1)
        nudged_last_h = last_h + effective_gamma * h_magnitude * tangent_nudge

        new_output = output.clone()
        new_output[:, -1, :] = nudged_last_h
        return new_output

def run_root_cause_analysis():
    print("=" * 80)
    print(" 【正確性低下の根本原因解明 & ゲート付き磁気浮上改善実験】")
    print("=" * 80)

    # 1. Qwen2.5-0.5B と Qwen2.5-1.5B の両方で検証
    models_to_test = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]

    # 厳格な正誤判定ができる知識・推論テストケース
    eval_cases = [
        {
            "id": "Cap_Australia",
            "q": "オーストラリアの首都はどこですか？都市名のみで答えてください。",
            "correct_ans": "キャンベラ",
            "wrong_trap": "シドニー"
        },
        {
            "id": "CS_2PC_Deadlock",
            "q": "2相コミット（2PC）プロトコルにおいて、コーディネーター（調停者）がコミット準備完了（Prepared）受信後にクラッシュした場合、参加者ノードに発生する最大の問題は何ですか？",
            "correct_ans": "ブロック", # または ブロッキング、停止
            "wrong_trap": "正常に完了"
        },
        {
            "id": "Math_Order",
            "q": "3 + 5 × 4 の計算結果を数字のみで答えてください。",
            "correct_ans": "23",
            "wrong_trap": "32"
        },
        {
            "id": "Attention_Scale",
            "q": "TransformerのScaled Dot-Product Attentionで、内積を√d_kで割る主な目的は何の増大を防ぐためですか？1単語で答えてください。",
            "correct_ans": "分散",
            "wrong_trap": "平均"
        }
    ]

    for model_id in models_to_test:
        print(f"\n######################################################################")
        print(f"■ モデル検証: {model_id}")
        print(f"######################################################################")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cuda")
        model.eval()
        device = model.device

        num_layers = getattr(model.config, "num_hidden_layers", len(model.model.layers))
        l_low = int(num_layers * 0.25)
        l_mid = int(num_layers * 0.50)
        l_high = int(num_layers * 0.75)

        # アトラクターA: 固定の英語Universalアトラクター (現行WebUI)
        fixed_english_prompts = [
            "Fundamental definitions, explicit premises, factual axioms, syntactic tokens",
            "Step-by-step causal deduction, intermediate reasoning mechanisms, relational inference",
            "Holistic synthesis, comprehensive coherent conclusion, abstract invariants"
        ]
        # アトラクターB: 日本語の論理整合アトラクター (質問言語と整合)
        aligned_jp_prompts = [
            "前提条件の厳密な定義、客観的事実の公理、正確な用語の同定",
            "段階的な論理的因果関係の検証、反例の検討、中間推論ステップの確認",
            "客観的結論の厳密な導出、矛盾のない統合的回答"
        ]

        def get_attractors(prompts):
            vecs = {}
            for target_layer in [l_low, l_mid, l_high]:
                v_list = []
                for p in prompts:
                    inp = tokenizer(p, return_tensors="pt").to(device)
                    with torch.no_grad():
                        out = model(**inp, output_hidden_states=True)
                        h = out.hidden_states[target_layer][0, -1, :]
                        v_list.append(F.normalize(h, p=2, dim=-1))
                vecs[target_layer] = torch.stack(v_list)
            return vecs

        attractors_fixed = get_attractors(fixed_english_prompts)
        attractors_aligned = get_attractors(aligned_jp_prompts)

        # テストする条件一覧
        configs = [
            {"name": "1. Base Model (素のモデル, 無介入)", "type": "none"},
            {"name": "2. 現行積層 (英語固定, ゲート無/常時強制介入 γ=0.08)", "type": "ungated_fixed"},
            {"name": "3. 改善積層 (日本語整合, ゲート無/常時介入 γ=0.08)", "type": "ungated_aligned"},
            {"name": "4. 新アーキテクチャ (日本語整合 + 磁界ゲート付き介入 γ=0.08, th=0.25)", "type": "gated_aligned"},
        ]

        for cfg in configs:
            print(f"\n--- [条件: {cfg['name']}] ---")
            correct_cnt = 0
            
            for ec in eval_cases:
                # フックの準備
                handles = []
                hooks = []
                if cfg["type"] == "ungated_fixed":
                    for l_idx in [l_low, l_mid, l_high]:
                        hk = UngatedHook(attractors_fixed[l_idx], gamma=0.08)
                        h = model.model.layers[l_idx].mlp.register_forward_hook(hk)
                        handles.append(h); hooks.append(hk)
                elif cfg["type"] == "ungated_aligned":
                    for l_idx in [l_low, l_mid, l_high]:
                        hk = UngatedHook(attractors_aligned[l_idx], gamma=0.08)
                        h = model.model.layers[l_idx].mlp.register_forward_hook(hk)
                        handles.append(h); hooks.append(hk)
                elif cfg["type"] == "gated_aligned":
                    for l_idx in [l_low, l_mid, l_high]:
                        hk = GatedHook(attractors_aligned[l_idx], gamma=0.08, threshold=0.25)
                        h = model.model.layers[l_idx].mlp.register_forward_hook(hk)
                        handles.append(h); hooks.append(hk)

                inp = tokenizer(tokenizer.apply_chat_template([{"role": "user", "content": ec["q"]}], tokenize=False, add_generation_prompt=True), return_tensors="pt").to(device)

                torch.manual_seed(42)
                with torch.no_grad():
                    out = model.generate(**inp, max_new_tokens=80, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                ans_text = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()

                for h in handles: h.remove()

                is_ok = ec["correct_ans"].lower() in ans_text.lower()
                is_trap = ec["wrong_trap"].lower() in ans_text.lower() if ec["wrong_trap"] else False

                mark = "✅ 正解" if is_ok else ("❌ 罠誤答" if is_trap else "⚠️ 不正解")
                if is_ok: correct_cnt += 1

                print(f"  ・[{ec['id']:15s}] {mark:6s} | 回答: {repr(ans_text[:50])}")

            print(f"  => 総合正解率: {correct_cnt}/{len(eval_cases)} ({correct_cnt/len(eval_cases)*100:.1f}%)")

        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    run_root_cause_analysis()
