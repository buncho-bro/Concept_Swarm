"""
Concept-Forge Benchmark with Qwen2.5-1.5B-Instruct
--------------------------------------------------
Concept-Forge WebUI で実装されていた認知幾何学ベンチマークスイート:
1. Analogy (比喩推論力: 5問)
2. Hierarchy (階層分離理解: 5問)
3. Multi-hop (連続軌道推論: 3問)
4. Logic Chain (論理演繹連鎖: 3問)
5. 文章補完・事実性推論 (Factuality & Reasoning: 5問)
をダウンロード済みの Qwen2.5-1.5B-Instruct を用いて実行・測定する。
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

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

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
    ("France", ["Paris", "Tower"], "Eiffel_Tower"),
    ("Science", ["Apple", "Universal_Gravitation"], "Gravity")
]

logic_tests = [
    ("Bird", "Move", "Penguin", "Move"),
    ("Mammal", "Animal", "Dog", "Animal"),
    ("Ice", "Cold", "Water", "Cold")
]

qa_prompts = [
    {"q": "オーストラリアの首都はどこですか？シドニーと誤解されやすいですが、正しい都市名を1語で答えてください。", "expected": "キャンベラ"},
    {"q": "太陽系で太陽に最も近い惑星は何ですか？", "expected": "水星"},
    {"q": "データベースのACID特性における「A」は何の略ですか？", "expected": "Atomicity"},
    {"q": "二分探索（バイナリサーチ）の計算量はO記法で何ですか？", "expected": "O(log N)"},
    {"q": "江戸幕府の初代征夷大将軍は誰ですか？", "expected": "徳川家康"}
]

def get_word_embedding(model, tokenizer, word, device):
    inputs = tokenizer(word, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[-1][0, -1, :]
    return F.normalize(hidden.to(torch.float32), p=2, dim=0)

def test_multihop_routing(model, tokenizer, start, relations, target, device):
    current_vec = get_word_embedding(model, tokenizer, start, device)
    for rel in relations:
        rel_vec = get_word_embedding(model, tokenizer, rel, device)
        current_vec = F.normalize(current_vec + rel_vec, p=2, dim=0)
    target_vec = get_word_embedding(model, tokenizer, target, device)
    return F.cosine_similarity(current_vec.unsqueeze(0), target_vec.unsqueeze(0)).item()

def test_logic_chain(model, tokenizer, a, a_is_b, c, target_b, device):
    vec_a = get_word_embedding(model, tokenizer, a, device)
    vec_b = get_word_embedding(model, tokenizer, a_is_b, device)
    vec_c = get_word_embedding(model, tokenizer, c, device)
    logic_direction = F.normalize(vec_b - vec_a, p=2, dim=0)
    pred_b = F.normalize(vec_c + logic_direction, p=2, dim=0)
    target_vec = get_word_embedding(model, tokenizer, target_b, device)
    return F.cosine_similarity(pred_b.unsqueeze(0), target_vec.unsqueeze(0)).item()

def main():
    print("=" * 70)
    print(f"🚀 Concept-Forge 統合ベンチマーク実行 (モデル: {MODEL_ID})")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] モデルをロード中 ({device})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    print("[+] ロード完了\n")

    # 1. 比喩推論 (Analogy)
    print("--- [1] Analogy (比喩推論テスト: a - b + c ≈ d) ---")
    analogy_scores = []
    for a, b, c, d in analogy_tests:
        va = get_word_embedding(model, tokenizer, a, device)
        vb = get_word_embedding(model, tokenizer, b, device)
        vc = get_word_embedding(model, tokenizer, c, device)
        vd = get_word_embedding(model, tokenizer, d, device)
        v_pred = F.normalize(va - vb + vc, p=2, dim=0)
        score = F.cosine_similarity(v_pred.unsqueeze(0), vd.unsqueeze(0)).item()
        analogy_scores.append(score)
        print(f"  ・{a} - {b} + {c} ➔ {d}: コサイン類似度 = {score:.4f}")
    avg_analogy = np.mean(analogy_scores)
    print(f"  ==> Analogy 平均スコア: {avg_analogy:.4f}\n")

    # 2. 階層理解 (Hierarchy)
    print("--- [2] Hierarchy (階層分離度テスト: sim(Base, Target) - sim(Base, Outlier)) ---")
    hierarchy_scores = []
    for base, target, outlier in hierarchy_tests:
        v_base = get_word_embedding(model, tokenizer, base, device)
        v_tgt = get_word_embedding(model, tokenizer, target, device)
        v_out = get_word_embedding(model, tokenizer, outlier, device)
        sim_tgt = F.cosine_similarity(v_base.unsqueeze(0), v_tgt.unsqueeze(0)).item()
        sim_out = F.cosine_similarity(v_base.unsqueeze(0), v_out.unsqueeze(0)).item()
        score = sim_tgt - sim_out
        hierarchy_scores.append(score)
        print(f"  ・{base} ➔ 近接: {target} (sim={sim_tgt:.3f}) vs 外れ値: {outlier} (sim={sim_out:.3f}) => 分離度 = {score:+.4f}")
    avg_hierarchy = np.mean(hierarchy_scores)
    print(f"  ==> Hierarchy 平均分離度: {avg_hierarchy:.4f}\n")

    # 3. 連続軌道推論 (Multi-hop Routing)
    print("--- [3] Multi-hop (連続軌道追跡テスト) ---")
    hop_scores = []
    for s, r, t in hop_tests:
        score = test_multihop_routing(model, tokenizer, s, r, t, device)
        hop_scores.append(score)
        print(f"  ・{s} + {' + '.join(r)} ➔ {t}: コサイン類似度 = {score:.4f}")
    avg_hop = np.mean(hop_scores)
    print(f"  ==> Multi-hop 平均スコア: {avg_hop:.4f}\n")

    # 4. 論理演繹連鎖 (Logic Chain)
    print("--- [4] Logic Chain (論理演繹ベクトル保存テスト) ---")
    logic_scores = []
    for a, b, c, t in logic_tests:
        score = test_logic_chain(model, tokenizer, a, b, c, t, device)
        logic_scores.append(score)
        print(f"  ・{a}:{b} :: {c}:{t} => 演繹保存度 = {score:.4f}")
    avg_logic = np.mean(logic_scores)
    print(f"  ==> Logic Chain 平均スコア: {avg_logic:.4f}\n")

    # 5. 文章補完・事実性生成テスト (Text Completion / QA)
    print("--- [5] Text Completion (文章補完・事実知識テスト) ---")
    qa_results = []
    qa_hits = 0
    for item in qa_prompts:
        q = item["q"]
        exp = item["expected"]
        messages = [
            {"role": "system", "content": "あなたは簡潔かつ極めて正確に回答するAIアシスタントです。余計な前置きは省き、結論を述べてください。"},
            {"role": "user", "content": q}
        ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=45,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        res_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        ans = tokenizer.decode(res_tokens, skip_special_tokens=True).strip()
        hit = exp.lower() in ans.lower()
        if hit:
            qa_hits += 1
        print(f"  [Q] {q}")
        print(f"      ➔ 解答: {ans}")
        print(f"      ➔ 期待: {exp} [{'正解 (PASS)' if hit else '不正解 (FAIL)'}]\n")
        qa_results.append({"q": q, "ans": ans, "expected": exp, "hit": hit})

    qa_acc = qa_hits / len(qa_prompts)

    # 総合集計
    print("=" * 70)
    print("📊 Concept-Forge ベンチマーク最終サマリー (Qwen2.5-1.5B-Instruct)")
    print(f"  1. Analogy (比喩推論力):           {avg_analogy:.4f}")
    print(f"  2. Hierarchy (階層概念分離度):     {avg_hierarchy:+.4f}")
    print(f"  3. Multi-hop (連続軌道追跡):       {avg_hop:.4f}")
    print(f"  4. Logic Chain (論理演繹保存度):   {avg_logic:.4f}")
    print(f"  5. Factuality QA (事実性・正確率): {qa_acc * 100:.1f}% ({qa_hits}/{len(qa_prompts)})")
    print("=" * 70)

    # レポート保存
    report = {
        "model": MODEL_ID,
        "analogy_avg": avg_analogy,
        "hierarchy_avg": avg_hierarchy,
        "multihop_avg": avg_hop,
        "logic_chain_avg": avg_logic,
        "qa_accuracy": qa_acc,
        "qa_details": qa_results
    }
    with open("distillation_experiments/concept_forge_qwen15b_benchmark_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("[+] ベンチマークレポートを保存しました: distillation_experiments/concept_forge_qwen15b_benchmark_report.json")

if __name__ == "__main__":
    main()
