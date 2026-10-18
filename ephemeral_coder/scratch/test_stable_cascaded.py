import os
import sys
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import time
import math
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ==============================================================================
# 積層全大円干渉エンジン (NaN防御 & 安定化版)
# ==============================================================================

class CascadedGreatCircleLayer:
    def __init__(self, layer_id, name, basis_words, model, tokenizer, device):
        self.layer_id = layer_id
        self.name = name
        self.basis_words = basis_words
        self.device = device
        
        vecs = []
        for w in basis_words:
            inp = tokenizer(w, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
                h = out.hidden_states[layer_id][0, -1, :].to(torch.float32)
                h = F.normalize(h, p=2, dim=-1)
                vecs.append(h)
        self.basis = torch.stack(vecs) # (K, D)

    def step_geodesic_interference(self, v_init, n_steps=2, eta=0.10, sigma=0.30):
        """
        全大円干渉による測地線滑走 (NaN完全防止)
        """
        v = F.normalize(v_init.to(torch.float32), p=2, dim=-1)
        
        for step in range(n_steps):
            # 基底とのコサイン類似度
            cos_theta = torch.matmul(v, self.basis.T) # (K,)
            cos_theta = torch.clamp(cos_theta, -0.999, 0.999)
            
            # 磁気浮上引力 (基底に引き寄せられる)
            dist_sq = 2.0 - 2.0 * cos_theta
            pulls = torch.exp(- dist_sq / (2.0 * (sigma ** 2)))
            
            # 閾値ゲート (無関係な基底は干渉させない)
            mask = cos_theta > 0.10
            pulls = pulls * mask.float()
            
            total_pull = torch.sum(pulls)
            if total_pull > 1e-5:
                weights = pulls / total_pull
                
                # 大円接線ベクトル: t_k = (c_k - cos * v) / sin
                sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta ** 2, min=1e-8))
                tangents = (self.basis - cos_theta.unsqueeze(-1) * v.unsqueeze(0)) / sin_theta.unsqueeze(-1)
                
                interference_dir = torch.sum(weights.unsqueeze(-1) * tangents, dim=0)
                norm_dir = torch.norm(interference_dir, p=2)
                if norm_dir > 1e-6:
                    interference_dir = interference_dir / norm_dir
                    v = F.normalize(v + eta * interference_dir, p=2, dim=-1)
            else:
                # 干渉が起きない（無関係）な場合は位置を維持（歪めない）
                pass
                
        return v


class CascadedGreatCircleHook:
    def __init__(self, circle_layer, gamma=0.05):
        self.circle_layer = circle_layer
        self.gamma = gamma
        self.active = True

    def __call__(self, module, inputs, output):
        if not self.active:
            return output
            
        last_h = output[:, -1, :] # (B, D)
        h_norm = F.normalize(last_h, p=2, dim=-1)
        
        # 1. 現在の思考ベクトルを初期位置として全大円干渉
        attractor = self.circle_layer.step_geodesic_interference(h_norm[0], n_steps=2, eta=0.08)
        attractor = attractor.to(last_h.dtype).unsqueeze(0)
        
        # 2. 接平面射影 (Tangent Nudge)
        proj = torch.sum(attractor * h_norm, dim=-1, keepdim=True) * h_norm
        tangent = attractor - proj
        norm_t = torch.norm(tangent, p=2, dim=-1, keepdim=True)
        
        # 安全弁: ゼロ除算・NaN防止
        if norm_t.item() < 1e-6:
            return output # 変化なしなら元のまま
            
        tangent_nudge = tangent / norm_t
        
        # 3. 磁気浮上介入
        h_mag = torch.norm(last_h, p=2, dim=-1, keepdim=True)
        nudged = last_h + self.gamma * h_mag * tangent_nudge
        
        new_output = output.clone()
        new_output[:, -1, :] = nudged
        return new_output


def run_stable_cascaded_test():
    print("=" * 80)
    print(" 【安定化版】積層全大円干渉エンジン 実証実験")
    print("=" * 80)

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cuda")
    model.eval()
    device = model.device

    l_low, l_mid, l_high = 6, 12, 18

    # 言語普遍の純粋な幾何学概念基底（文章指定・アンカー化なし）
    basis_layer_low = ["Entity", "Object", "Number", "Name", "Place", "Time", "Property", "State", "Definition", "Data"]
    basis_layer_mid = ["Cause", "Effect", "Condition", "Relation", "Difference", "Mechanism", "Process", "Rule", "Change"]
    basis_layer_high = ["Conclusion", "Principle", "System", "Truth", "Consistency", "Summary", "Invariance", "Resolution"]

    print("\n[1] 3層の全大円干渉モジュールを構築...")
    circle_l6 = CascadedGreatCircleLayer(l_low, "Layer 6 (要素)", basis_layer_low, model, tokenizer, device)
    circle_l12 = CascadedGreatCircleLayer(l_mid, "Layer 12 (関係)", basis_layer_mid, model, tokenizer, device)
    circle_l18 = CascadedGreatCircleLayer(l_high, "Layer 18 (統合)", basis_layer_high, model, tokenizer, device)

    hook_l6 = CascadedGreatCircleHook(circle_l6, gamma=0.05)
    hook_l12 = CascadedGreatCircleHook(circle_l12, gamma=0.05)
    hook_l18 = CascadedGreatCircleHook(circle_l18, gamma=0.05)

    handles = [
        model.model.layers[l_low].mlp.register_forward_hook(hook_l6),
        model.model.layers[l_mid].mlp.register_forward_hook(hook_l12),
        model.model.layers[l_high].mlp.register_forward_hook(hook_l18)
    ]

    test_queries = [
        "日本の首都はどこですか？簡潔に答えてください。",
        "オーストラリアの首都はどこですか？簡潔に答えてください。",
        "3 + 5 × 4 の計算結果を数字のみで答えてください。",
        "2相コミット（2PC）プロトコルで、コーディネーターがPrepared受信後にクラッシュした際の最大の問題は何ですか？",
        "TransformerのSelf-Attentionにおいて、内積を√d_kで割る数学的理由は何ですか？"
    ]

    for q in test_queries:
        print(f"\n=======================================================")
        print(f"◆ 問い: \"{q}\"")
        inp = tokenizer(tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True), return_tensors="pt").to(device)

        # 1. Base Model (素)
        hook_l6.active = False; hook_l12.active = False; hook_l18.active = False
        torch.manual_seed(42)
        with torch.no_grad():
            out_base = model.generate(**inp, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        base_resp = tokenizer.decode(out_base[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()

        # 2. 積層全大円篩 (Cascaded Great Circle: 前層吸引点 ➔ 次層初期位置)
        hook_l6.active = True; hook_l12.active = True; hook_l18.active = True
        torch.manual_seed(42)
        with torch.no_grad():
            out_cascaded = model.generate(**inp, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        cascaded_resp = tokenizer.decode(out_cascaded[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()

        print(f"  ▶ [Base Model (素)]:")
        print(f"    {repr(base_resp[:100])}")
        print(f"  ▶ [新・積層全大円篩 (Cascaded)]:")
        print(f"    {repr(cascaded_resp[:100])}")

    for h in handles:
        h.remove()

if __name__ == "__main__":
    run_stable_cascaded_test()
