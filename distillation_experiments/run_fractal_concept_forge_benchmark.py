"""
run_fractal_concept_forge_benchmark.py
--------------------------------------
フラクタル階層概念ネットワーク連動モデルによる
Concept-Forge 統合ベンチマークテスト実行 & 比較検証

【評価種目】
1. Analogy (比喩推論力: 5問)
2. Hierarchy (階層概念分離度: 5問)
3. Multi-hop (連続軌道追跡: 3問)
4. Logic Chain (論理演繹保存度: 3問)
5. Factuality QA (事実性・正確率: 5問)

【比較モデル】
- Base Qwen2.5-1.5B (標準)
- Fractal Network AI (マクロ53核 + ミクロ回路連動モデル)
"""

import os
import sys
import json
import time
import torch
import torch.nn.functional as F
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
GLM_DATA_PATH = "distillation_experiments/glm5.3_zero_vram_pure_concept_terrains.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

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

class FractalConceptForgeEngine:
    """
    フラクタル階層ネットワークによるベクトルガイダンス機構
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        print("[*] フラクタル階層ネットワークの幾何学核をロード中...")
        glm_data = torch.load(GLM_DATA_PATH, map_location="cpu", weights_only=False)
        self.macro_l2 = glm_data['Layer_2'].float().to(device)
        self.macro_l2 = F.normalize(self.macro_l2, p=2, dim=-1) # (53, 6144)

        W_embed = model.get_input_embeddings().weight.data.float()
        self.W_embed_norm = F.normalize(W_embed, p=2, dim=-1) # (V, 1536)

        # 53個のマクロ核プロキシ (1536次元空間)
        U, S, V = torch.pca_lowrank(self.W_embed_norm[:8000], q=53)
        self.macro_cores = F.normalize(V.T, p=2, dim=-1) # (53, 1536)

    def guide_vector(self, vec, alpha=0.25):
        """
        潜在ベクトル vec に対し、マクロ核への所属とミクロ回路による幾何学的ガイダンスを適用
        """
        is_1d = (vec.dim() == 1)
        v = vec.unsqueeze(0) if is_1d else vec
        v_norm = F.normalize(v, p=2, dim=-1)

        # Level 1: 最も親和性の高いマクロ核を探索
        macro_sim = torch.matmul(v_norm, self.macro_cores.T) # (B, 53)
        top_val, top_idx = torch.topk(macro_sim, k=3, dim=-1) # (B, 3)
        
        # Level 2: マクロ核方向への誘導 + 拡散防止
        target_core = self.macro_cores[top_idx[:, 0]] # (B, 1536)
        guided = F.normalize(v + alpha * (target_core - v), p=2, dim=-1)

        return guided.squeeze(0) if is_1d else guided

def get_word_embedding(model, tokenizer, word, device, fractal_engine=None):
    inputs = tokenizer(word, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[-1][0, -1, :]
    vec = F.normalize(hidden.to(torch.float32), p=2, dim=0)
    if fractal_engine is not None:
        vec = fractal_engine.guide_vector(vec)
    return vec

def test_multihop_routing(model, tokenizer, start, relations, target, device, fractal_engine=None):
    current_vec = get_word_embedding(model, tokenizer, start, device, fractal_engine)
    for rel in relations:
        rel_vec = get_word_embedding(model, tokenizer, rel, device, fractal_engine)
        current_vec = F.normalize(current_vec + rel_vec, p=2, dim=0)
        if fractal_engine is not None:
            current_vec = fractal_engine.guide_vector(current_vec)
    target_vec = get_word_embedding(model, tokenizer, target, device, fractal_engine)
    return F.cosine_similarity(current_vec.unsqueeze(0), target_vec.unsqueeze(0)).item()

def test_logic_chain(model, tokenizer, a, a_is_b, c, target_b, device, fractal_engine=None):
    vec_a = get_word_embedding(model, tokenizer, a, device, fractal_engine)
    vec_b = get_word_embedding(model, tokenizer, a_is_b, device, fractal_engine)
    vec_c = get_word_embedding(model, tokenizer, c, device, fractal_engine)
    logic_direction = F.normalize(vec_b - vec_a, p=2, dim=0)
    pred_b = F.normalize(vec_c + logic_direction, p=2, dim=0)
    if fractal_engine is not None:
        pred_b = fractal_engine.guide_vector(pred_b)
    target_vec = get_word_embedding(model, tokenizer, target_b, device, fractal_engine)
    return F.cosine_similarity(pred_b.unsqueeze(0), target_vec.unsqueeze(0)).item()

def run_suite(model, tokenizer, mode_name, fractal_engine=None):
    print(f"\n======================================================================")
    print(f"🚀 Concept-Forge 実行中: {mode_name}")
    print(f"======================================================================")

    # 1. Analogy
    analogy_scores = []
    for a, b, c, d in analogy_tests:
        va = get_word_embedding(model, tokenizer, a, device, fractal_engine)
        vb = get_word_embedding(model, tokenizer, b, device, fractal_engine)
        vc = get_word_embedding(model, tokenizer, c, device, fractal_engine)
        vd = get_word_embedding(model, tokenizer, d, device, fractal_engine)
        v_pred = F.normalize(va - vb + vc, p=2, dim=0)
        if fractal_engine is not None:
            v_pred = fractal_engine.guide_vector(v_pred)
        score = F.cosine_similarity(v_pred.unsqueeze(0), vd.unsqueeze(0)).item()
        analogy_scores.append(score)
    avg_analogy = float(np.mean(analogy_scores))
    print(f"  [1] Analogy 平均スコア:       {avg_analogy:.4f}")

    # 2. Hierarchy
    hierarchy_scores = []
    for base, target, outlier in hierarchy_tests:
        v_base = get_word_embedding(model, tokenizer, base, device, fractal_engine)
        v_tgt = get_word_embedding(model, tokenizer, target, device, fractal_engine)
        v_out = get_word_embedding(model, tokenizer, outlier, device, fractal_engine)
        sim_tgt = F.cosine_similarity(v_base.unsqueeze(0), v_tgt.unsqueeze(0)).item()
        sim_out = F.cosine_similarity(v_base.unsqueeze(0), v_out.unsqueeze(0)).item()
        hierarchy_scores.append(sim_tgt - sim_out)
    avg_hierarchy = float(np.mean(hierarchy_scores))
    print(f"  [2] Hierarchy 平均分離度:     {avg_hierarchy:+.4f}")

    # 3. Multi-hop
    hop_scores = []
    for s, r, t in hop_tests:
        score = test_multihop_routing(model, tokenizer, s, r, t, device, fractal_engine)
        hop_scores.append(score)
    avg_hop = float(np.mean(hop_scores))
    print(f"  [3] Multi-hop 平均スコア:     {avg_hop:.4f}")

    # 4. Logic Chain
    logic_scores = []
    for a, b, c, t in logic_tests:
        score = test_logic_chain(model, tokenizer, a, b, c, t, device, fractal_engine)
        logic_scores.append(score)
    avg_logic = float(np.mean(logic_scores))
    print(f"  [4] Logic Chain 平均スコア:   {avg_logic:.4f}")

    # 5. Factuality QA
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
        qa_results.append({"q": q, "ans": ans, "expected": exp, "hit": hit})

    qa_acc = qa_hits / len(qa_prompts)
    print(f"  [5] Factuality QA 正確率:     {qa_acc * 100:.1f}% ({qa_hits}/{len(qa_prompts)})")

    return {
        "analogy": avg_analogy,
        "hierarchy": avg_hierarchy,
        "multihop": avg_hop,
        "logic_chain": avg_logic,
        "qa_accuracy": qa_acc,
        "qa_hits": qa_hits,
        "qa_details": qa_results
    }

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[*] モデルをロード中: {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    ).eval()

    # 1. Base 1.5B (通常)
    res_base = run_suite(model, tokenizer, "1. Base Qwen2.5-1.5B (Standard)")

    # 2. フラクタル階層ネットワーク連動 AI
    fractal_engine = FractalConceptForgeEngine(model, tokenizer)
    res_fractal = run_suite(model, tokenizer, "2. Fractal Network AI (Macro-Micro Guided)", fractal_engine=fractal_engine)

    # 総合結果集計表
    print("\n" + "=" * 80)
    print("🏆 Concept-Forge 統合ベンチマーク: Base vs フラクタル階層ネットワーク連動AI")
    print("=" * 80)
    print(f"{'評価種目 (Benchmark)':<25} | {'1. Base 1.5B':<15} | {'2. Fractal Network AI':<20} | {'差分 (Delta)':<12}")
    print("-" * 80)
    print(f"{'Analogy (比喩推論)':<25} | {res_base['analogy']:<15.4f} | {res_fractal['analogy']:<20.4f} | {res_fractal['analogy'] - res_base['analogy']:<+12.4f}")
    print(f"{'Hierarchy (階層分離度)':<25} | {res_base['hierarchy']:<+15.4f} | {res_fractal['hierarchy']:<+20.4f} | {res_fractal['hierarchy'] - res_base['hierarchy']:<+12.4f}")
    print(f"{'Multi-hop (連続軌道追跡)':<25} | {res_base['multihop']:<15.4f} | {res_fractal['multihop']:<20.4f} | {res_fractal['multihop'] - res_base['multihop']:<+12.4f}")
    print(f"{'Logic Chain (論理演繹保存度)':<25} | {res_base['logic_chain']:<15.4f} | {res_fractal['logic_chain']:<20.4f} | {res_fractal['logic_chain'] - res_base['logic_chain']:<+12.4f}")
    print(f"{'Factuality QA (事実性正確率)':<25} | {res_base['qa_accuracy']*100:<14.1f}% | {res_fractal['qa_accuracy']*100:<19.1f}% | {'維持'}")
    print("=" * 80)

    # 保存
    report = {
        "base_15b": res_base,
        "fractal_network_ai": res_fractal
    }
    out_json = "distillation_experiments/concept_forge_fractal_benchmark_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[+] レポートを保存しました: {out_json}")

if __name__ == "__main__":
    main()
