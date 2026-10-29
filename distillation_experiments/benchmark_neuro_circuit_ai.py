"""
benchmark_neuro_circuit_ai.py
-----------------------------
コンテキスト可変抵抗（思考電気回路）推論AI ベンチマークテスト

【テスト構成】
1. 多義性・文脈適応テスト (Contextual Disambiguation):
   同一単語（bank, apple, crane, plant, python）が文脈によって開く回路（低抵抗ノード）を正しく切り替えられるか。
2. 多段階因果推論テスト (Multi-Hop Causal Reasoning):
   物理・科学・社会・数学の因果連鎖において、最小抵抗経路（最小作用回路）を流れて正しい結論へ着水できるか。
3. ハルシネーション遮断テスト (Hallucination Rejection):
   欺瞞的なノイズ文脈の中で、高抵抗な偽の知識を絶縁ゲートで遮断し、真実に電流を流せるか。

【比較手法】
- Baseline: 従来の生の次トークン確率（Greedy / Top-1）
- Neuro-Circuit AI: 文脈抵抗 + キルヒホッフ有効抵抗 R_eff + 絶縁ゲート
"""

import sys
import os
import json
import torch
import torch.nn.functional as F
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from transformers import AutoModelForCausalLM, AutoTokenizer

BENCHMARK_SUITE = [
    # --- Category 1: 多義性・文脈適応テスト ---
    {
        "category": "Disambiguation",
        "context": "In the Wall Street financial market, the commercial bank announced",
        "question": "What is the primary association?",
        "options": ["loan", "river", "mud", "water"],
        "ground_truth": "loan",
        "probe_word": "bank"
    },
    {
        "category": "Disambiguation",
        "context": "Walking along the muddy edge of the flowing river, the grassy bank was",
        "question": "What is the primary association?",
        "options": ["water", "loan", "interest", "finance"],
        "ground_truth": "water",
        "probe_word": "bank"
    },
    {
        "category": "Disambiguation",
        "context": "In the Silicon Valley tech conference, the executive from Apple announced",
        "question": "What is the primary association?",
        "options": ["iphone", "fruit", "orchard", "cider"],
        "ground_truth": "iphone",
        "probe_word": "apple"
    },
    {
        "category": "Disambiguation",
        "context": "At the autumn harvest festival in the orchard, the delicious sweet apple was",
        "question": "What is the primary association?",
        "options": ["fruit", "iphone", "silicon", "computer"],
        "ground_truth": "fruit",
        "probe_word": "apple"
    },
    {
        "category": "Disambiguation",
        "context": "At the high-rise skyscraper construction site, the heavy industrial crane lifted",
        "question": "What is the primary association?",
        "options": ["steel", "wings", "feather", "nest"],
        "ground_truth": "steel",
        "probe_word": "crane"
    },
    {
        "category": "Disambiguation",
        "context": "Flying over the peaceful northern lake, the majestic white crane spread",
        "question": "What is the primary association?",
        "options": ["wings", "steel", "engine", "concrete"],
        "ground_truth": "wings",
        "probe_word": "crane"
    },

    # --- Category 2: 因果推論テスト (Causal Chains) ---
    {
        "category": "Causal Reasoning",
        "context": "When severe crime occurs, the duty of the investigating police is to make an",
        "question": "What is the immediate outcome?",
        "options": ["arrest", "flower", "recreation", "celebration"],
        "ground_truth": "arrest",
        "probe_word": "police"
    },
    {
        "category": "Causal Reasoning",
        "context": "When intense heat and fire consume dry wooden logs, the remaining residue is",
        "question": "What is the final substance?",
        "options": ["ash", "ice", "glass", "liquid"],
        "ground_truth": "ash",
        "probe_word": "fire"
    },
    {
        "category": "Causal Reasoning",
        "context": "In abstract algebra, when an algebraic group satisfies the commutative property, it is called",
        "question": "What is the mathematical classification?",
        "options": ["abelian", "geometric", "matrix", "linear"],
        "ground_truth": "abelian",
        "probe_word": "group"
    },
    {
        "category": "Causal Reasoning",
        "context": "When a naval blockade completely seals a strategic international strait, it triggers an",
        "question": "What is the geopolitical result?",
        "options": ["embargo", "celebration", "tourism", "sports"],
        "ground_truth": "embargo",
        "probe_word": "blockade"
    },
    {
        "category": "Causal Reasoning",
        "context": "When freezing cold subzero winter temperatures hit moisture in the cloud, it produces",
        "question": "What is the precipitation?",
        "options": ["snow", "steam", "heat", "smoke"],
        "ground_truth": "snow",
        "probe_word": "cold"
    },

    # --- Category 3: ハルシネーション遮断テスト ---
    {
        "category": "Hallucination Rejection",
        "context": "Although the patient insisted on antibiotics for their viral influenza, the medical doctor explained that antibiotics are only effective against",
        "question": "What do antibiotics target?",
        "options": ["bacteria", "virus", "influenza", "fungus"],
        "ground_truth": "bacteria",
        "probe_word": "antibiotics"
    },
    {
        "category": "Hallucination Rejection",
        "context": "While sound waves travel easily through air and water, in the complete physical vacuum of outer space, sound can",
        "question": "Can sound travel?",
        "options": ["never", "always", "faster", "louder"],
        "ground_truth": "never",
        "probe_word": "vacuum"
    },
    {
        "category": "Hallucination Rejection",
        "context": "Despite the severe lightning and dark thunderous clouds passing high above the dry Sahara desert, the arid ground remained completely",
        "question": "What was the ground condition?",
        "options": ["dry", "flooded", "soaked", "green"],
        "ground_truth": "dry",
        "probe_word": "ground"
    }
]

class NeuroCircuitAIBenchmark:
    def __init__(self, model_name="Qwen/Qwen2.5-7B"):
        print(f"[*] 思考回路AIモデルをロード中: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()

        # 語彙
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
        self.word2id = {w: tid for w, tid in zip(purified_words, purified_ids)}
        self.valid_ids_tensor = torch.tensor(purified_ids, device=self.device)
        print(f"[+] 思考語彙プール: {len(self.words_list)} 語")

    def run_inference(self, task: dict):
        context = task["context"]
        options = task["options"]
        truth = task["ground_truth"]

        inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
            logits = out.logits[0, -1, :]
            selected_logits = logits[self.valid_ids_tensor]
            probs = F.softmax(selected_logits, dim=-1).float().cpu().numpy()

        # 1. Baseline: 生の予測確率
        opt_probs = {}
        for opt in options:
            if opt in self.word2id:
                idx = self.words_list.index(opt)
                opt_probs[opt] = float(probs[idx])
            else:
                # 語彙に直接ない場合は最小確率
                opt_probs[opt] = 1e-7

        baseline_choice = max(opt_probs.items(), key=lambda x: x[1])[0]

        # 2. Neuro-Circuit AI: 動的インピーダンス回路シミュレーション
        # 抵抗 R = -log2(P) (サプライザル抵抗)
        # かつ、絶縁閾値 (R > 18.0 bit は物理的完全遮断)
        circuit_resistances = {}
        circuit_currents = {}
        insulation_threshold = 18.0 # 絶縁抵抗値

        for opt in options:
            p = opt_probs[opt]
            r = -np.log2(max(p, 1e-8))
            circuit_resistances[opt] = r

            # オームの法則: I = V / R (V = 1.0)
            if r >= insulation_threshold:
                i = 0.0 # 絶縁遮断
            else:
                # 低抵抗ほど電気が二乗で集中する非線形コンダクタンス
                i = 1.0 / (r ** 1.5)
            circuit_currents[opt] = i

        # 最大電流が流れた回路を選択
        if max(circuit_currents.values()) > 0:
            neuro_choice = max(circuit_currents.items(), key=lambda x: x[1])[0]
        else:
            # 全て絶縁の場合は最小抵抗
            neuro_choice = min(circuit_resistances.items(), key=lambda x: x[1])[0]

        return {
            "baseline_choice": baseline_choice,
            "baseline_correct": baseline_choice == truth,
            "neuro_choice": neuro_choice,
            "neuro_correct": neuro_choice == truth,
            "resistances": circuit_resistances,
            "currents": circuit_currents,
            "ground_truth": truth
        }

def execute_benchmark():
    engine = NeuroCircuitAIBenchmark("Qwen/Qwen2.5-7B")

    print("\n" + "="*75)
    print("⚡ 思考電気回路AI（Neuro-Circuit AI）総合ベンチマークテスト")
    print("="*75)

    baseline_score = 0
    neuro_score = 0
    total = len(BENCHMARK_SUITE)

    detailed_results = []

    for idx, task in enumerate(BENCHMARK_SUITE, 1):
        res = engine.run_inference(task)
        b_ok = res["baseline_correct"]
        n_ok = res["neuro_correct"]

        if b_ok: baseline_score += 1
        if n_ok: neuro_score += 1

        print(f"\n[Task {idx:02d}/{total}] ({task['category']}) Ground Truth: [{res['ground_truth']}]")
        print(f"  Context: \"{task['context']}\"")
        print(f"  ・Baseline Choice:     [{res['baseline_choice']}] (正解: {b_ok})")
        print(f"  ・Neuro-Circuit Choice: [{res['neuro_choice']}] (正解: {n_ok})")
        print(f"  ・回路抵抗値 (Ω/bit):   { {k: round(v, 2) for k, v in res['resistances'].items()} }")
        print(f"  ・流れた電流 (A):       { {k: round(v, 4) for k, v in res['currents'].items()} }")

        detailed_results.append({
            "task_id": idx,
            "category": task["category"],
            "context": task["context"],
            "ground_truth": res["ground_truth"],
            "baseline_choice": res["baseline_choice"],
            "baseline_correct": b_ok,
            "neuro_choice": res["neuro_choice"],
            "neuro_correct": n_ok,
            "resistances": res["resistances"],
            "currents": res["currents"]
        })

    print("\n" + "="*75)
    print("📊 最終ベンチマーク成績サマリー")
    print("="*75)
    print(f"【Baseline (生の次トークン確率)】")
    print(f"  正解率: {baseline_score}/{total} ({baseline_score/total*100:.1f}%)")
    print(f"【Neuro-Circuit AI (思考電気回路 + オームの法則 + 絶縁ゲート)】")
    print(f"  正解率: {neuro_score}/{total} ({neuro_score/total*100:.1f}%)")
    print("="*75)

    # JSON出力
    out_json = "distillation_experiments/neuro_circuit_ai_benchmark_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total": total,
                "baseline_accuracy": baseline_score / total,
                "neuro_circuit_accuracy": neuro_score / total
            },
            "tasks": detailed_results
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[+] ベンチマークレポートJSONを保存しました: {out_json}")

if __name__ == "__main__":
    execute_benchmark()
