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

# 現行 Hook
class CurrentLayerTerrainHook:
    def __init__(self, layer_idx, layer_name, attractors, gamma=0.08, sigma=0.25):
        self.layer_idx = layer_idx
        self.layer_name = layer_name
        self.attractors = attractors
        self.gamma = gamma
        self.sigma = sigma
        self.active = True
        self.drift_angles = []

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

        # 介入による角度偏向 (ドリフト角度 degree) を記録
        nudged_norm = F.normalize(nudged_last_h, p=2, dim=-1)
        cos_drift = torch.clamp(torch.sum(h_norm * nudged_norm, dim=-1), -1.0, 1.0)
        drift_deg = torch.acos(cos_drift).item() * (180.0 / np.pi)
        self.drift_angles.append(drift_deg)

        new_output = output.clone()
        new_output[:, -1, :] = nudged_last_h
        return new_output

class MultiLayerSynchronizedController:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.hook_handles = []

    def attach_layer_terrains(self, layer_specs, gamma=0.08):
        self.detach()
        for l_idx, attractors in layer_specs.items():
            hook = CurrentLayerTerrainHook(
                layer_idx=l_idx,
                layer_name=f"Layer_{l_idx}_MLP",
                attractors=attractors,
                gamma=gamma
            )
            handle = self.model.model.layers[l_idx].mlp.register_forward_hook(hook)
            self.hooks.append(hook)
            self.hook_handles.append(handle)

    def set_active(self, active: bool):
        for h in self.hooks:
            h.active = active

    def clear_records(self):
        for h in self.hooks:
            h.drift_angles.clear()

    def detach(self):
        for handle in self.hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.hooks.clear()
        self.hook_handles.clear()

def run_dialogue_accuracy_benchmark():
    print("=" * 80)
    print(" 【実対話シナリオ】積層地形の回答正確性・文脈脱線（ドリフト）詳細検証")
    print("=" * 80)

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cuda")
    model.eval()
    device = model.device

    l_low, l_mid, l_high = 6, 12, 18

    # WebUI デフォルトの Universal Reasoning アトラクター
    univ_prompts = {
        "low": [
            "Fundamental definitions, explicit premises, factual axioms, syntactic tokens, empirical observations",
            "Base primitives, ground truth constraints, precise data inputs, logical conditions"
        ],
        "mid": [
            "Step-by-step causal deduction, intermediate reasoning mechanisms, relational inference, counterfactual validation",
            "Transition logic, algorithmic transformations, comparative analysis, structural dependencies"
        ],
        "high": [
            "Holistic synthesis, comprehensive coherent conclusion, abstract invariants, deep principles, meta-cognitive review",
            "Unified theoretical framework, architectural design judgment, balanced long-term trade-offs"
        ]
    }
    def enc(prompts, layer):
        vecs = []
        for p in prompts:
            inp = tokenizer(p, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
                h = out.hidden_states[layer][0, -1, :]
                vecs.append(F.normalize(h, p=2, dim=-1))
        return torch.stack(vecs)

    attractors = {
        l_low: enc(univ_prompts["low"], l_low),
        l_mid: enc(univ_prompts["mid"], l_mid),
        l_high: enc(univ_prompts["high"], l_high)
    }

    controller = MultiLayerSynchronizedController(model)

    # ユーザーが実際にWebUIで問いかける代表的な5つのシナリオ
    queries = [
        {
            "id": "Q1_2PC_Paxos",
            "prompt": "分散データベースにおいて、2相コミット（2PC）とPaxosプロトコルの根本的な違いを正確に解説してください。",
            "target_facts": ["2PCはアトミックコミット（全会一致）", "Paxosはコンセンサス（過半数合意）", "2PCはコーディネーター障害でブロックする", "Paxosは過半数が生きていれば進行可能"],
            "irrelevant_traps": ["Axiom", "premise", "syntactic"]
        },
        {
            "id": "Q2_Attention_Scale",
            "prompt": "TransformerのSelf-Attentionにおいて、内積を次元数d_kの平方根（√d_k）で割る（スケーリングする）数学的な理由は何ですか？",
            "target_facts": ["分散がd_k倍に増大するのを防ぐ", "ソフトマックスの勾配消失（極端な値）を防ぐ", "分散を1に正規化"],
            "irrelevant_traps": ["philosophical", "moral"]
        },
        {
            "id": "Q3_SOLID_Principle",
            "prompt": "オブジェクト指向プログラミングにおけるSOLID原則のうち、「単一責任の原則（SRP）」とは何ですか？具体例を交えて説明してください。",
            "target_facts": ["クラスを変更する理由は1つだけでなければならない", "1つの責務のみを持つ", "変更理由の分離"],
            "irrelevant_traps": ["cosmological", "particle"]
        },
        {
            "id": "Q4_Capital_Fact",
            "prompt": "オーストラリアの首都はどこですか？シドニーと誤解されやすいですが、正しい都市名を挙げてください。",
            "target_facts": ["キャンベラ", "Canberra"],
            "irrelevant_traps": ["シドニーが首都", "メルボルンが首都"]
        },
        {
            "id": "Q5_Code_BinarySearch",
            "prompt": "ソート済み配列から特定数値を探索する「二分探索（バイナリサーチ）」の計算量O(log N)の理由を簡潔に説明してください。",
            "target_facts": ["探索範囲が毎回半分（1/2）になる", "2の何乗か（対数）", "log2 N ステップ"],
            "irrelevant_traps": ["O(N)", "線形探索"]
        }
    ]

    benchmark_records = []

    print("\n[実対話プロンプトでの比較実験を開始]")
    for q_item in queries:
        qid = q_item["id"]
        q_text = q_item["prompt"]
        print(f"\n" + "-" * 70)
        print(f"◆ 課題 [{qid}]: {q_text}")

        # 1. Base Model 生成
        controller.detach()
        controller.set_active(False)
        inp = tokenizer(tokenizer.apply_chat_template([{"role": "user", "content": q_text}], tokenize=False, add_generation_prompt=True), return_tensors="pt").to(device)
        
        torch.manual_seed(42)
        with torch.no_grad():
            out_base = model.generate(**inp, max_new_tokens=250, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        base_text = tokenizer.decode(out_base[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()

        # 2. Stacked Terrains 生成 (γ = 0.08)
        controller.attach_layer_terrains(attractors, gamma=0.08)
        controller.set_active(True)
        controller.clear_records()

        torch.manual_seed(42)
        with torch.no_grad():
            out_stacked = model.generate(**inp, max_new_tokens=250, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        stacked_text = tokenizer.decode(out_stacked[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()

        # ドリフト角の平均
        l6_drifts = controller.hooks[0].drift_angles if controller.hooks else []
        l12_drifts = controller.hooks[1].drift_angles if len(controller.hooks) > 1 else []
        l18_drifts = controller.hooks[2].drift_angles if len(controller.hooks) > 2 else []

        mean_drift_deg = np.mean(l6_drifts + l12_drifts + l18_drifts) if l6_drifts else 0.0

        # 事実の網羅性チェック
        base_hits = [f for f in q_item["target_facts"] if f.lower() in base_text.lower() or any(w in base_text for w in f.split("（")[0].split("・"))]
        stacked_hits = [f for f in q_item["target_facts"] if f.lower() in stacked_text.lower() or any(w in stacked_text for w in f.split("（")[0].split("・"))]

        print(f"\n  [A. Base Model (素のモデル)] (事実合致: {len(base_hits)}/{len(q_item['target_facts'])})")
        print(f"  {base_text[:200]}...\n")

        print(f"  [B. Stacked Terrains (現行積層)] (事実合致: {len(stacked_hits)}/{len(q_item['target_facts'])}, 平均偏向角: {mean_drift_deg:.2f}°)")
        print(f"  {stacked_text[:200]}...\n")

        benchmark_records.append({
            "id": qid,
            "prompt": q_text,
            "base_text": base_text,
            "stacked_text": stacked_text,
            "base_hits": len(base_hits),
            "stacked_hits": len(stacked_hits),
            "total_facts": len(q_item["target_facts"]),
            "mean_drift_deg": mean_drift_deg
        })

    print("=" * 80)
    print(" 【対話ベンチマーク総括】")
    for r in benchmark_records:
        print(f"・[{r['id']:18s}] Base 正確性: {r['base_hits']}/{r['total_facts']} | Stacked 正確性: {r['stacked_hits']}/{r['total_facts']} | 隠れ層偏向角: {r['mean_drift_deg']:.2f}°")
    print("=" * 80)

    with open("dialogue_benchmark_records.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_records, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    run_dialogue_accuracy_benchmark()
