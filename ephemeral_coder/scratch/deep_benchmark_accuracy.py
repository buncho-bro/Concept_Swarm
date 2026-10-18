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

# 現行の LayerTerrainHook（バグ検証用）
class CurrentLayerTerrainHook:
    def __init__(self, layer_idx, layer_name, attractors, gamma=0.08, sigma=0.25):
        self.layer_idx = layer_idx
        self.layer_name = layer_name
        self.attractors = attractors
        self.gamma = gamma
        self.sigma = sigma
        self.active = True
        self.records = []

    def __call__(self, module, inputs, output):
        if not self.active or self.attractors is None:
            return output
        last_h = output[:, -1, :]
        h_norm = F.normalize(last_h, p=2, dim=-1)

        cos_sim = torch.matmul(h_norm, self.attractors.T) # (B, N)
        dist_sq = 2.0 - 2.0 * cos_sim
        pulls = torch.exp(- dist_sq / (2.0 * (self.sigma ** 2)))

        # 【検証ポイント】Softmax で正規化すると、pulls の絶対値（磁界強度）が消滅する！
        weights = F.softmax(pulls * 8.0, dim=-1)
        target_dir = torch.matmul(weights, self.attractors)
        target_dir = F.normalize(target_dir, p=2, dim=-1)

        tangent_nudge = target_dir - torch.sum(target_dir * h_norm, dim=-1, keepdim=True) * h_norm
        tangent_nudge = F.normalize(tangent_nudge, p=2, dim=-1)

        h_magnitude = torch.norm(last_h, p=2, dim=-1, keepdim=True)
        # ここで常に gamma の強さで介入している
        nudged_last_h = last_h + self.gamma * h_magnitude * tangent_nudge

        max_cos = torch.max(cos_sim).item()
        max_pull = torch.max(pulls).item()
        self.records.append({"cos_sim": max_cos, "pull": max_pull})

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
            h.records.clear()

    def detach(self):
        for handle in self.hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.hooks.clear()
        self.hook_handles.clear()

def detailed_accuracy_benchmark():
    print("=" * 75)
    print(" [DEEP BENCHMARK] 積層地形の正確性・事実歪曲メカニズムの徹底検証")
    print("=" * 75)

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

    # 多彩な事実・知識・推論ベンチマークセット (12問)
    test_suite = [
        # 1. 確定歴史・地理・常識
        {"id": "Geo_1", "q": "日本で一番高い山の名前は何ですか？山名のみで答えてください。", "ans": "富士山", "type": "Fact"},
        {"id": "Hist_1", "q": "江戸幕府を開いた初代征夷大将軍は誰ですか？人名のみで答えてください。", "ans": "徳川家康", "type": "Fact"},
        {"id": "Sci_1", "q": "太陽系で最も太陽に近い惑星は何ですか？惑星名のみで答えてください。", "ans": "水星", "type": "Fact"},
        {"id": "Sci_2", "q": "ヒトの平熱はおよそ何度ですか？数字（小数含む）で答えてください。", "ans": "36", "type": "Fact"},

        # 2. 算術・論理
        {"id": "Math_1", "q": "7 × 8 の計算結果は何ですか？数字のみで答えてください。", "ans": "56", "type": "Math"},
        {"id": "Math_2", "q": "100から27を引くといくつになりますか？数字のみで答えてください。", "ans": "73", "type": "Math"},
        {"id": "Logic_1", "q": "「すべての鳥は飛ぶ。ペンギンは鳥である。」この前提から演繹される結論は何ですか？", "ans": "ペンギンは飛ぶ", "type": "Logic"},

        # 3. コード・アルゴリズム
        {"id": "Code_1", "q": "Pythonでリスト [3, 1, 2] を昇順にソートする関数名またはメソッド名は何ですか？", "ans": "sort", "type": "Code"},

        # 4. 専門推論・概念
        {"id": "CS_1", "q": "ACID特性のAは何の略ですか？英単語1語で答えてください。", "ans": "Atomicity", "type": "CS"},
        {"id": "CS_2", "q": "HTTPステータスコードで「Not Found」を表す数字は何ですか？", "ans": "404", "type": "CS"},
        {"id": "CS_3", "q": "Gitで変更を一時退避させるコマンドは何ですか？", "ans": "stash", "type": "CS"},
        {"id": "General_1", "q": "1年は何日ですか？（平年）数字のみで答えてください。", "ans": "365", "type": "Fact"}
    ]

    print(f"\n[2] ベンチマーク実行: 全 {len(test_suite)} 問")
    print("    条件 A: 素の Base Model (無介入, γ=0.0)")
    print("    条件 B: 現行 Stacked Terrains (Universal, γ=0.08, 常時強制介入)")

    base_correct = 0
    stacked_correct = 0
    drift_details = []

    for item in test_suite:
        qid = item["id"]
        q = item["q"]
        ans = item["ans"]

        # --- A. Base Model ---
        controller.detach()
        controller.set_active(False)
        inp = tokenizer(tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True), return_tensors="pt").to(device)

        torch.manual_seed(42)
        with torch.no_grad():
            out_base = model.generate(**inp, max_new_tokens=60, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        base_resp = tokenizer.decode(out_base[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()

        # --- B. Stacked Terrains ---
        controller.attach_layer_terrains(attractors, gamma=0.08)
        controller.set_active(True)
        controller.clear_records()

        torch.manual_seed(42)
        with torch.no_grad():
            out_stacked = model.generate(**inp, max_new_tokens=60, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        stacked_resp = tokenizer.decode(out_stacked[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()

        # アトラクターとの類似度統計
        avg_cos_l6 = np.mean([r["cos_sim"] for r in controller.hooks[0].records]) if controller.hooks[0].records else 0
        avg_pull_l6 = np.mean([r["pull"] for r in controller.hooks[0].records]) if controller.hooks[0].records else 0

        # 正誤判定
        base_ok = ans.lower() in base_resp.lower()
        stacked_ok = ans.lower() in stacked_resp.lower()

        if base_ok: base_correct += 1
        if stacked_ok: stacked_correct += 1

        print(f"\n[{qid}] 問い: {q}")
        print(f"     期待正解: {ans}")
        print(f"  ・Base     ({ 'OK ' if base_ok else 'NG ' }): {repr(base_resp[:60])}")
        print(f"  ・Stacked  ({ 'OK ' if stacked_ok else 'NG ' }): {repr(stacked_resp[:60])}")
        print(f"    [統計] Layer6 平均コサイン類似度: {avg_cos_l6:.3f} | 引力値: {avg_pull_l6:.4e}")

        drift_details.append({
            "id": qid,
            "q": q,
            "ans": ans,
            "base_ok": base_ok,
            "stacked_ok": stacked_ok,
            "base_resp": base_resp,
            "stacked_resp": stacked_resp,
            "avg_cos_l6": avg_cos_l6,
            "avg_pull_l6": avg_pull_l6
        })

    total = len(test_suite)
    print("\n" + "=" * 75)
    print(f" 【検証結果サマリー】")
    print(f"  ・Base Model 正解数:      {base_correct}/{total} ({base_correct/total*100:.1f}%)")
    print(f"  ・Stacked Model 正解数:   {stacked_correct}/{total} ({stacked_correct/total*100:.1f}%)")
    print("=" * 75)

    with open("deep_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(drift_details, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    detailed_accuracy_benchmark()
