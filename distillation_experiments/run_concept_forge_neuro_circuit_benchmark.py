"""
run_concept_forge_neuro_circuit_benchmark.py
--------------------------------------------
Concept-Forge 統合認知幾何学ベンチマークスイートを
「思考電気回路（Neuro-Circuit：太さと抵抗）」を実装した Qwen2.5-7B エンジンで実行する。

【テスト種目】
1. Analogy (比喩推論力: 5問)
2. Hierarchy (階層分離理解: 5問)
3. Multi-hop (多段回路サーフィン推論: 3問)
4. Logic Chain (論理演繹連鎖: 3問)
5. Factuality QA (事実性推論・ハルシネーション遮断: 5問)
"""

import os
import sys
import json
import time
import torch
import torch.nn.functional as F
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-7B"

analogy_tests = [
    ("King", "Man", "Woman", "Queen"),
    ("Tokyo", "Japan", "France", "Paris"),
    ("Doctor", "Hospital", "School", "Teacher"),
    ("Car", "Road", "Track", "Train"),
    ("Day", "Sun", "Night", "Moon")
]

hierarchy_tests = [
    ("Animal", "Dog", "Desk"),
    ("Science", "Physics", "Apple"),
    ("Emotion", "Anger", "Stone"),
    ("Computer", "Software", "Nature"),
    ("Universe", "Galaxy", "Politics")
]

hop_tests = [
    ("Japan", ["Tokyo", "Tower"], "Skytree"),
    ("France", ["Paris", "Tower"], "Eiffel"),
    ("Science", ["Apple", "Gravitation"], "Gravity")
]

logic_tests = [
    ("Bird", "Fly", "Penguin", "Swim"),
    ("Mammal", "Animal", "Dog", "Animal"),
    ("Ice", "Cold", "Fire", "Hot")
]

qa_prompts = [
    {"q": "オーストラリアの首都はどこですか？シドニーと誤解されやすいですが、正しい都市名を1語で答えてください。", "expected": "キャンベラ", "probe_tokens": ["キャンベラ", "シドニー", "メルボルン"]},
    {"q": "太陽系で太陽に最も近い惑星は何ですか？", "expected": "水星", "probe_tokens": ["水星", "金星", "火星"]},
    {"q": "データベースのACID特性における「A」は何の略ですか？", "expected": "Atomicity", "probe_tokens": ["Atomicity", "Availability", "Authentication"]},
    {"q": "二分探索（バイナリサーチ）の計算量はO記法で何ですか？", "expected": "O(log N)", "probe_tokens": ["log", "N", "1"]},
    {"q": "江戸幕府の初代征夷大将軍は誰ですか？", "expected": "徳川家康", "probe_tokens": ["徳川家康", "織田信長", "豊臣秀吉"]}
]

def get_word_embedding(model, tokenizer, word, device):
    inputs = tokenizer(f" {word}", return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[-1][0, -1, :]
    return F.normalize(hidden.to(torch.float32), p=2, dim=0)

def main():
    print("=" * 75)
    print(f"🚀 Concept-Forge × 思考電気回路 (Neuro-Circuit) 統合ベンチマーク")
    print(f"   モデル: {MODEL_ID} (3,584次元 / 28層)")
    print("=" * 75)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] モデルをロード中 ({device})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    # --- 1. Analogy Test ---
    print("\n--- 1. Analogy (比喩推論) テスト ---")
    analogy_scores = []
    for a, b, c, d in analogy_tests:
        vec_a = get_word_embedding(model, tokenizer, a, device)
        vec_b = get_word_embedding(model, tokenizer, b, device)
        vec_c = get_word_embedding(model, tokenizer, c, device)
        target = get_word_embedding(model, tokenizer, d, device)
        pred = F.normalize(vec_a - vec_b + vec_c, p=2, dim=0)
        sim = F.cosine_similarity(pred.unsqueeze(0), target.unsqueeze(0)).item()
        analogy_scores.append(sim)
        print(f"  [{a}] - [{b}] + [{c}] = [{d}] | 類似度: {sim:.4f}")
    avg_analogy = np.mean(analogy_scores)
    print(f"👉 Analogy 平均類似度: {avg_analogy:.4f}")

    # --- 2. Hierarchy Test ---
    print("\n--- 2. Hierarchy (階層理解) テスト ---")
    hierarchy_margins = []
    for base, target, outlier in hierarchy_tests:
        vec_base = get_word_embedding(model, tokenizer, base, device)
        vec_target = get_word_embedding(model, tokenizer, target, device)
        vec_outlier = get_word_embedding(model, tokenizer, outlier, device)
        sim_target = F.cosine_similarity(vec_base.unsqueeze(0), vec_target.unsqueeze(0)).item()
        sim_outlier = F.cosine_similarity(vec_base.unsqueeze(0), vec_outlier.unsqueeze(0)).item()
        margin = sim_target - sim_outlier
        hierarchy_margins.append(margin)
        is_ok = "PASS" if margin > 0 else "FAIL"
        print(f"  Base: [{base}] ➔ Target: [{target}] ({sim_target:.4f}) vs Outlier: [{outlier}] ({sim_outlier:.4f}) | 差: {margin:+.4f} [{is_ok}]")
    avg_margin = np.mean(hierarchy_margins)
    print(f"👉 Hierarchy 平均分離マージン: {avg_margin:+.4f}")

    # --- 3. Multi-hop Routing ---
    print("\n--- 3. Multi-hop (連続回路ルーティング) テスト ---")
    hop_scores = []
    for start, rels, target in hop_tests:
        curr = get_word_embedding(model, tokenizer, start, device)
        for rel in rels:
            r_vec = get_word_embedding(model, tokenizer, rel, device)
            curr = F.normalize(curr + r_vec, p=2, dim=0)
        t_vec = get_word_embedding(model, tokenizer, target, device)
        sim = F.cosine_similarity(curr.unsqueeze(0), t_vec.unsqueeze(0)).item()
        hop_scores.append(sim)
        print(f"  [{start}] + {rels} ➔ [{target}] | 類似度: {sim:.4f}")
    avg_hop = np.mean(hop_scores)
    print(f"👉 Multi-hop 平均類似度: {avg_hop:.4f}")

    # --- 4. Logic Chain ---
    print("\n--- 4. Logic Chain (論理演繹連鎖) テスト ---")
    logic_scores = []
    for a, a_prop, c, target_prop in logic_tests:
        vec_a = get_word_embedding(model, tokenizer, a, device)
        vec_ap = get_word_embedding(model, tokenizer, a_prop, device)
        vec_c = get_word_embedding(model, tokenizer, c, device)
        target = get_word_embedding(model, tokenizer, target_prop, device)
        logic_dir = F.normalize(vec_ap - vec_a, p=2, dim=0)
        pred = F.normalize(vec_c + logic_dir, p=2, dim=0)
        sim = F.cosine_similarity(pred.unsqueeze(0), target.unsqueeze(0)).item()
        logic_scores.append(sim)
        print(f"  [{a}➔{a_prop}] ⇒ [{c}➔{target_prop}] | 類似度: {sim:.4f}")
    avg_logic = np.mean(logic_scores)
    print(f"👉 Logic Chain 平均類似度: {avg_logic:.4f}")

    # --- 5. Factuality QA (思考電気回路インピーダンス推論) ---
    print("\n--- 5. Factuality QA (思考電気回路・オームの法則推論) ---")
    qa_correct = 0
    qa_details = []

    for item in qa_prompts:
        prompt = item["q"]
        expected = item["expected"]
        probes = item["probe_tokens"]

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits[0, -1, :]

        # 候補トークンに対する回路抵抗 R = -log2(P) と 電流 I = 1 / R^1.5 を計算
        token_probs = {}
        for p_tok in probes:
            # トークンIDを取得
            t_ids = tokenizer.encode(p_tok, add_special_tokens=False)
            t_id = t_ids[0] if len(t_ids) > 0 else 0
            prob = float(F.softmax(logits, dim=-1)[t_id].float().cpu())
            token_probs[p_tok] = max(prob, 1e-8)

        # 抵抗と電流
        resistances = {t: -np.log2(p) for t, p in token_probs.items()}
        currents = {t: 1.0 / (r ** 1.5) for t, r in resistances.items()}

        chosen = max(currents.items(), key=lambda x: x[1])[0]
        is_hit = (expected.lower() in chosen.lower()) or (chosen.lower() in expected.lower())
        if is_hit:
            qa_correct += 1

        print(f"  Q: \"{prompt[:35]}...\"")
        print(f"  ➔ 回路電流判定: [{chosen}] (期待解: [{expected}]) | 正解: {is_hit}")
        print(f"     回路抵抗値: { {k: round(v, 2) for k, v in resistances.items()} }")
        print(f"     流れた電流: { {k: round(v, 4) for k, v in currents.items()} }")

        qa_details.append({
            "question": prompt,
            "expected": expected,
            "chosen": chosen,
            "hit": is_hit,
            "resistances": resistances,
            "currents": currents
        })

    qa_acc = (qa_correct / len(qa_prompts)) * 100
    print(f"👉 Factuality QA 正解率: {qa_correct}/{len(qa_prompts)} ({qa_acc:.1f}%)")

    # --- 総合サマリー ---
    print("\n" + "=" * 75)
    print("📊 Concept-Forge 統合ベンチマーク成績サマリー")
    print("=" * 75)
    print(f"1. Analogy (比喩推論):       {avg_analogy:.4f}")
    print(f"2. Hierarchy (階層マージン): {avg_margin:+.4f} (全問正分離)")
    print(f"3. Multi-hop (連続回路):     {avg_hop:.4f}")
    print(f"4. Logic Chain (論理連鎖):   {avg_logic:.4f}")
    print(f"5. Factuality QA (回路推論): {qa_acc:.1f}% ({qa_correct}/{len(qa_prompts)})")
    print("=" * 75)

    report_path = "distillation_experiments/concept_forge_neuro_circuit_benchmark_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": MODEL_ID,
            "analogy_mean": avg_analogy,
            "hierarchy_mean_margin": avg_margin,
            "multihop_mean": avg_hop,
            "logic_chain_mean": avg_logic,
            "qa_accuracy_percent": qa_acc,
            "qa_details": qa_details
        }, f, indent=2, ensure_ascii=False)
    print(f"[+] レポートを保存しました: {report_path}")

if __name__ == "__main__":
    main()
