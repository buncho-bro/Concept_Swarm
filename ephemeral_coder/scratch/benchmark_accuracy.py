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
# 積層地形の正確性・事実性ベンチマークテスト (Factuality & Reasoning Benchmark)
# ==============================================================================

# 現行の LayerTerrainHook
class CurrentLayerTerrainHook:
    def __init__(self, layer_idx, layer_name, attractors, gamma=0.08, sigma=0.25):
        self.layer_idx = layer_idx
        self.layer_name = layer_name
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

    def set_gamma(self, gamma: float):
        for h in self.hooks:
            h.gamma = gamma

    def detach(self):
        for handle in self.hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.hooks.clear()
        self.hook_handles.clear()

def run_benchmark():
    print("=" * 70)
    print(" [BENCHMARK] 積層地形の事実性・正確性・指示追従性 厳格検証テスト")
    print("=" * 70)

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"\n[1] モデルロード: {model_id} (FP16)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    model.eval()
    device = model.device

    # レイヤー選定 (24層中 6, 12, 18)
    l_low, l_mid, l_high = 6, 12, 18

    # 現行 WebUI と同一のアトラクター構築
    TERRAIN_DOMAINS = {
        "Universal": {
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
        },
        "Distributed": {
            "low": [
                "Primitive locking primitives: mutex, spinlock, compare-and-swap atomic operations, memory barriers",
                "Deadlock detection algorithms using wait-for graphs, cycle detection, edge allocation"
            ],
            "mid": [
                "Two-phase locking protocol (2PL) ensuring strict serializable schedules, lock escalation",
                "Multi-version concurrency control (MVCC) snapshot isolation with timestamp ordering and undo logs"
            ],
            "high": [
                "Strict serializability and linearizability in high-throughput distributed architectures, consensus Paxos Raft",
                "Deadlock-free non-blocking consensus with bounded latency guarantees and CAP trade-off optimization"
            ]
        }
    }

    def build_attractors(domain_key):
        d = TERRAIN_DOMAINS[domain_key]
        def enc(prompts, layer):
            vecs = []
            for p in prompts:
                inp = tokenizer(p, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model(**inp, output_hidden_states=True)
                    h = out.hidden_states[layer][0, -1, :]
                    vecs.append(F.normalize(h, p=2, dim=-1))
            return torch.stack(vecs)
        return {
            l_low: enc(d["low"], l_low),
            l_mid: enc(d["mid"], l_mid),
            l_high: enc(d["high"], l_high)
        }

    print("    ・汎用ドメイン (Universal) アトラクター生成完了")
    print("    ・分散ドメイン (Distributed) アトラクター生成完了")
    attractors_univ = build_attractors("Universal")
    attractors_dist = build_attractors("Distributed")

    controller = MultiLayerSynchronizedController(model)

    # テスト問題群 (事実性、算術・論理、指示追従、日本語)
    test_cases = [
        {
            "id": "Fact_JP_1",
            "category": "確定事実 (地理・常識)",
            "prompt": "日本の首都はどこですか？簡潔に一言で答えてください。",
            "expected_keywords": ["東京"],
            "forbidden_keywords": ["Axiom", "Premise", "Definition", "Tokyo is", "capital"]
        },
        {
            "id": "Math_1",
            "category": "算術・論理計算",
            "prompt": "リンゴが5個あります。2個食べました。残りは何個ですか？数字で答えてください。",
            "expected_keywords": ["3", "3個"],
            "forbidden_keywords": ["5", "mutex", "lock", "concurrency"]
        },
        {
            "id": "Fact_Sci_1",
            "category": "科学的事実",
            "prompt": "水の化学式は何ですか？記号のみで答えてください。",
            "expected_keywords": ["H2O", "H₂O"],
            "forbidden_keywords": ["Hydrogen", "Oxygen", "Axiomatic", "deduction"]
        },
        {
            "id": "Reasoning_CS_1",
            "category": "ドメイン推論 (CS)",
            "prompt": "デッドロックが発生する4つの必要十分条件（コフマン条件）を箇条書きで挙げてください。",
            "expected_keywords": ["相互排除", "占有と待機", "非横領", "循環待機", "Mutual exclusion", "Hold and wait", "No preemption", "Circular wait"],
            "forbidden_keywords": []
        },
        {
            "id": "Lang_Consist_1",
            "category": "言語一貫性 (日本語維持)",
            "prompt": "自己紹介をしてください。あなたの役割は何ですか？日本語で丁寧に答えてください。",
            "expected_keywords": ["AI", "アシスタント", "モデル"],
            "forbidden_keywords": []
        }
    ]

    # 検証条件
    conditions = [
        {"name": "Base Model (無介入, γ=0.0)", "active": False, "attractors": None, "gamma": 0.0},
        {"name": "Stacked (Universal, γ=0.04 - 弱)", "active": True, "attractors": attractors_univ, "gamma": 0.04},
        {"name": "Stacked (Universal, γ=0.08 - 現行)", "active": True, "attractors": attractors_univ, "gamma": 0.08},
        {"name": "Stacked (Universal, γ=0.15 - 強)", "active": True, "attractors": attractors_univ, "gamma": 0.15},
        {"name": "Stacked (Distributed, γ=0.08 - 現行)", "active": True, "attractors": attractors_dist, "gamma": 0.08},
    ]

    results = []

    print("\n[2] ベンチマーク実行中...")
    for tc in test_cases:
        print(f"\n=======================================================")
        print(f"■ テスト項目: [{tc['id']}] {tc['category']}")
        print(f"  入力プロンプト: \"{tc['prompt']}\"")
        print(f"  期待キーワード: {tc['expected_keywords']}")
        print(f"=======================================================")

        for cond in conditions:
            if cond["active"]:
                controller.attach_layer_terrains(cond["attractors"], gamma=cond["gamma"])
                controller.set_active(True)
            else:
                controller.set_active(False)
                controller.detach()

            messages = [{"role": "user", "content": tc["prompt"]}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(device)

            torch.manual_seed(42)
            with torch.no_grad():
                out_tokens = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False, # 決定論的比較のため Greedy
                    pad_token_id=tokenizer.eos_token_id
                )
            gen_text = tokenizer.decode(out_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

            # 正確性評価
            has_expected = any(kw.lower() in gen_text.lower() for kw in tc["expected_keywords"])
            has_forbidden = any(kw.lower() in gen_text.lower() for kw in tc["forbidden_keywords"])
            
            # 日本語比率チェック (漢字・ひらがな・カタカナの含有率)
            jp_chars = sum(1 for c in gen_text if '\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9fff')
            total_chars = max(len(gen_text.replace(" ", "")), 1)
            jp_ratio = jp_chars / total_chars

            print(f"\n  ▶ 条件: {cond['name']}")
            print(f"    生成結果: {repr(gen_text[:120])}...")
            print(f"    正解判定: {'PASS' if has_expected and not has_forbidden else 'FAIL'} | 期待一致: {has_expected} | 混入禁止語: {has_forbidden} | 日本語率: {jp_ratio:.1%}")

            results.append({
                "test_id": tc["id"],
                "category": tc["category"],
                "condition": cond["name"],
                "gamma": cond["gamma"],
                "active": cond["active"],
                "output": gen_text,
                "passed": has_expected and not has_forbidden,
                "has_expected": has_expected,
                "has_forbidden": has_forbidden,
                "jp_ratio": jp_ratio
            })

    # 結果集計
    print("\n" + "=" * 70)
    print(" 【ベンチマーク結果総合サマリー】")
    print("=" * 70)

    summary_by_cond = {}
    for r in results:
        cname = r["condition"]
        if cname not in summary_by_cond:
            summary_by_cond[cname] = {"total": 0, "passed": 0, "expected_hits": 0, "jp_ratios": []}
        summary_by_cond[cname]["total"] += 1
        if r["passed"]:
            summary_by_cond[cname]["passed"] += 1
        if r["has_expected"]:
            summary_by_cond[cname]["expected_hits"] += 1
        summary_by_cond[cname]["jp_ratios"].append(r["jp_ratio"])

    for cname, s in summary_by_cond.items():
        pass_rate = s["passed"] / s["total"] * 100
        hit_rate = s["expected_hits"] / s["total"] * 100
        avg_jp = np.mean(s["jp_ratios"]) * 100
        print(f"・{cname:40s} | 正解率(厳格): {pass_rate:5.1f}% | 期待語網羅: {hit_rate:5.1f}% | 平均日本語率: {avg_jp:5.1f}%")

    with open("benchmark_accuracy_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n詳細結果を benchmark_accuracy_results.json に保存しました。")

if __name__ == "__main__":
    run_benchmark()
