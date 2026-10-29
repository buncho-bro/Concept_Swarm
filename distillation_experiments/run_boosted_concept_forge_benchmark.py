"""
run_boosted_concept_forge_benchmark.py
--------------------------------------
強制重力深化（Density-Driven Gravitational Deepening Boost）モデルによる
Concept-Forge 統合ベンチマーク実行 & 比較検証

【比較対象】
1. Base Qwen2.5-1.5B (通常ベースモデル)
2. KNN-Distilled 1.5B (7B➔1.5B トポロジカルKNN蒸留LoRA)
3. Boosted Qwen2.5-1.5B (通常ベース + 強制重力深化ブースト)
4. Boosted KNN-Distilled 1.5B (蒸留LoRA + 強制重力深化ブースト)

【評価種目】
- Analogy (比喩推論力: 5問)
- Hierarchy (階層概念分離度: 5問)
- Multi-hop (連続軌道追跡: 3問)
- Logic Chain (論理演繹保存度: 3問)
- Factuality QA (事実性・正確率: 5問)
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
from peft import PeftModel

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment/lora_Qwen2.5_7B_to_Qwen2.5_1.5B_Instruct_Topological"
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

class GravitationalDeepeningEngine:
    """
    高密度領域の局所重力深化（Density-Driven Gravitational Deepening）
    集まっている高密度な概念領域ほど引力を増幅し、潜在ベクトルを鋭い谷底へと引き込む
    """
    def __init__(self, model, tokenizer, boost_beta=0.15, alpha=0.35, sample_k=3000):
        self.boost_beta = boost_beta
        self.alpha = alpha
        
        print("[*] 重力深化エンジンの概念コアと局所密度場を構築中...")
        W_embed = model.get_input_embeddings().weight.data.float()
        vocab_size = W_embed.shape[0]

        clean_ids = []
        for tid in range(vocab_size):
            s = tokenizer.decode([tid]).strip()
            if len(s) >= 2 and s.isalpha():
                clean_ids.append(tid)

        sample_ids = clean_ids[:sample_k] if len(clean_ids) > sample_k else clean_ids
        sample_ids_t = torch.tensor(sample_ids, device=device, dtype=torch.long)
        self.core_vecs = F.normalize(W_embed[sample_ids_t].to(device), p=2, dim=-1) # (K, D)
        
        # 概念コア同士の局所密度 ρ_j の計算 (ガウシアン近接度)
        # sim = core_vecs @ core_vecs.T
        with torch.no_grad():
            sim_matrix = torch.matmul(self.core_vecs, self.core_vecs.T) # (K, K)
            # 類似度 > 0.4 の近傍密度
            weights = torch.clamp((sim_matrix - 0.4) / 0.6, min=0.0) ** 2
            self.density = weights.sum(dim=-1) # (K,)
            # 正規化
            self.density = (self.density - self.density.mean()) / (self.density.std() + 1e-6)
            self.density = torch.clamp(self.density, min=0.0) # 非負
            print(f"  ・概念コア数: {len(sample_ids)}, 平均局所密度: {self.density.mean().item():.3f}, 最大密度: {self.density.max().item():.3f}")

    def deepen(self, vec, top_k=24, tau=0.15):
        """
        潜在ベクトル vec に対し、高密度概念コアによる引力深掘りを適用
        vec: (D,) または (B, D)
        """
        is_1d = (vec.dim() == 1)
        if is_1d:
            v = vec.unsqueeze(0)
        else:
            v = vec

        v_norm = F.normalize(v, p=2, dim=-1)
        sim = torch.matmul(v_norm, self.core_vecs.T) # (B, K)
        
        # 上位 top_k の概念コア
        top_sim, top_idx = torch.topk(sim, k=top_k, dim=-1) # (B, top_k)
        
        # 局所密度ブースト: (1.0 + beta * sqrt(density))
        top_densities = self.density[top_idx] # (B, top_k)
        density_factor = 1.0 + self.boost_beta * torch.sqrt(top_densities + 1e-5)
        
        # ソフトマックス引力重み
        attn = F.softmax(top_sim / tau, dim=-1) * density_factor # (B, top_k)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)
        
        # 引力ベクトル Δv = sum(attn * (core_vec - v))
        selected_cores = self.core_vecs[top_idx] # (B, top_k, D)
        pull_targets = (selected_cores * attn.unsqueeze(-1)).sum(dim=1) # (B, D)
        
        deepened = F.normalize(v + self.alpha * (pull_targets - v), p=2, dim=-1)
        
        if is_1d:
            return deepened.squeeze(0)
        return deepened

def get_word_embedding(model, tokenizer, word, device, deepening_engine=None):
    inputs = tokenizer(word, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[-1][0, -1, :]
    vec = F.normalize(hidden.to(torch.float32), p=2, dim=0)
    if deepening_engine is not None:
        vec = deepening_engine.deepen(vec)
    return vec

def test_multihop_routing(model, tokenizer, start, relations, target, device, deepening_engine=None):
    current_vec = get_word_embedding(model, tokenizer, start, device, deepening_engine)
    for rel in relations:
        rel_vec = get_word_embedding(model, tokenizer, rel, device, deepening_engine)
        current_vec = F.normalize(current_vec + rel_vec, p=2, dim=0)
        if deepening_engine is not None:
            current_vec = deepening_engine.deepen(current_vec)
    target_vec = get_word_embedding(model, tokenizer, target, device, deepening_engine)
    return F.cosine_similarity(current_vec.unsqueeze(0), target_vec.unsqueeze(0)).item()

def test_logic_chain(model, tokenizer, a, a_is_b, c, target_b, device, deepening_engine=None):
    vec_a = get_word_embedding(model, tokenizer, a, device, deepening_engine)
    vec_b = get_word_embedding(model, tokenizer, a_is_b, device, deepening_engine)
    vec_c = get_word_embedding(model, tokenizer, c, device, deepening_engine)
    logic_direction = F.normalize(vec_b - vec_a, p=2, dim=0)
    pred_b = F.normalize(vec_c + logic_direction, p=2, dim=0)
    if deepening_engine is not None:
        pred_b = deepening_engine.deepen(pred_b)
    target_vec = get_word_embedding(model, tokenizer, target_b, device, deepening_engine)
    return F.cosine_similarity(pred_b.unsqueeze(0), target_vec.unsqueeze(0)).item()

def run_suite(model, tokenizer, mode_name, deepening_engine=None):
    print(f"\n======================================================================")
    print(f"🚀 ベンチマーク実行中: {mode_name}")
    print(f"======================================================================")
    
    # 1. 比喩推論 (Analogy)
    analogy_scores = []
    for a, b, c, d in analogy_tests:
        va = get_word_embedding(model, tokenizer, a, device, deepening_engine)
        vb = get_word_embedding(model, tokenizer, b, device, deepening_engine)
        vc = get_word_embedding(model, tokenizer, c, device, deepening_engine)
        vd = get_word_embedding(model, tokenizer, d, device, deepening_engine)
        v_pred = F.normalize(va - vb + vc, p=2, dim=0)
        if deepening_engine is not None:
            v_pred = deepening_engine.deepen(v_pred)
        score = F.cosine_similarity(v_pred.unsqueeze(0), vd.unsqueeze(0)).item()
        analogy_scores.append(score)
    avg_analogy = float(np.mean(analogy_scores))
    print(f"  [1] Analogy 平均スコア:       {avg_analogy:.4f}")

    # 2. 階層理解 (Hierarchy)
    hierarchy_scores = []
    for base, target, outlier in hierarchy_tests:
        v_base = get_word_embedding(model, tokenizer, base, device, deepening_engine)
        v_tgt = get_word_embedding(model, tokenizer, target, device, deepening_engine)
        v_out = get_word_embedding(model, tokenizer, outlier, device, deepening_engine)
        sim_tgt = F.cosine_similarity(v_base.unsqueeze(0), v_tgt.unsqueeze(0)).item()
        sim_out = F.cosine_similarity(v_base.unsqueeze(0), v_out.unsqueeze(0)).item()
        score = sim_tgt - sim_out
        hierarchy_scores.append(score)
    avg_hierarchy = float(np.mean(hierarchy_scores))
    print(f"  [2] Hierarchy 平均分離度:     {avg_hierarchy:+.4f}")

    # 3. 連続軌道推論 (Multi-hop Routing)
    hop_scores = []
    for s, r, t in hop_tests:
        score = test_multihop_routing(model, tokenizer, s, r, t, device, deepening_engine)
        hop_scores.append(score)
    avg_hop = float(np.mean(hop_scores))
    print(f"  [3] Multi-hop 平均スコア:     {avg_hop:.4f}")

    # 4. 論理演繹連鎖 (Logic Chain)
    logic_scores = []
    for a, b, c, t in logic_tests:
        score = test_logic_chain(model, tokenizer, a, b, c, t, device, deepening_engine)
        logic_scores.append(score)
    avg_logic = float(np.mean(logic_scores))
    print(f"  [4] Logic Chain 平均スコア:   {avg_logic:.4f}")

    # 5. 文章補完・事実性生成テスト (Factuality QA)
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

        if deepening_engine is None:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=45,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
        else:
            # 重力深化フック付きデコーディング
            # 各ステップの隠れ状態からロジットを計算する際、deepenedベクトルからLM Headへプロジェクション
            with torch.no_grad():
                cur_ids = inputs["input_ids"]
                for _ in range(40):
                    out = model(cur_ids, output_hidden_states=True)
                    last_hidden = out.hidden_states[-1][:, -1, :] # (1, D)
                    deep_hidden = deepening_engine.deepen(last_hidden.float(), top_k=16).half()
                    # LM head
                    lm_head = model.get_output_embeddings()
                    logits = lm_head(deep_hidden) # (1, vocab)
                    next_token = torch.argmax(logits, dim=-1, keepdim=True) # (1, 1)
                    cur_ids = torch.cat([cur_ids, next_token], dim=-1)
                    if next_token.item() == tokenizer.eos_token_id:
                        break
                outputs = cur_ids

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

    print(f"[*] Baseモデルをロード中: {MODEL_ID}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    ).eval()

    # 1. Base 1.5B (通常)
    res_base = run_suite(base_model, tokenizer, "1. Base Qwen2.5-1.5B (Standard)")

    # 重力深化エンジンの初期化
    deepening_engine = GravitationalDeepeningEngine(base_model, tokenizer, boost_beta=0.15, alpha=0.35)

    # 2. Boosted 1.5B (通常 + 強制重力深化ブースト)
    res_boosted_base = run_suite(base_model, tokenizer, "2. Boosted Qwen2.5-1.5B (Gravitational Deepening)", deepening_engine=deepening_engine)

    # 次に KNN-Distilled LoRA モデルをテスト
    del base_model
    del deepening_engine
    torch.cuda.empty_cache()

    print(f"\n[*] KNNトポロジカル蒸留LoRAモデルをロード中: {LORA_PATH}...")
    distill_base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    )
    distill_model = PeftModel.from_pretrained(distill_base, LORA_PATH)
    distill_model = distill_model.merge_and_unload().eval()

    # 3. KNN-Distilled 1.5B (通常蒸留)
    res_distill = run_suite(distill_model, tokenizer, "3. 7B➔1.5B KNN-Distilled LoRA (Standard)")

    # 4. Boosted KNN-Distilled 1.5B (蒸留 + 強制重力深化ブースト)
    deepening_engine_distill = GravitationalDeepeningEngine(distill_model, tokenizer, boost_beta=0.15, alpha=0.35)
    res_boosted_distill = run_suite(distill_model, tokenizer, "4. Boosted KNN-Distilled 1.5B (LoRA + Gravitational Deepening)", deepening_engine=deepening_engine_distill)

    # 総合結果集計表の出力
    print("\n" + "=" * 80)
    print("🏆 Concept-Forge 統合ベンチマーク: 全モデル比較結果")
    print("=" * 80)
    print(f"{'指標':<18} | {'1. Base 1.5B':<13} | {'2. Boosted Base':<15} | {'3. KNN-Distilled':<16} | {'4. Boosted Distilled':<18}")
    print("-" * 80)
    print(f"{'Analogy (比喩)':<18} | {res_base['analogy']:<13.4f} | {res_boosted_base['analogy']:<15.4f} | {res_distill['analogy']:<16.4f} | {res_boosted_distill['analogy']:<18.4f}")
    print(f"{'Hierarchy (階層)':<18} | {res_base['hierarchy']:<+13.4f} | {res_boosted_base['hierarchy']:<+15.4f} | {res_distill['hierarchy']:<+16.4f} | {res_boosted_distill['hierarchy']:<+18.4f}")
    print(f"{'Multi-hop (軌道)':<18} | {res_base['multihop']:<13.4f} | {res_boosted_base['multihop']:<15.4f} | {res_distill['multihop']:<16.4f} | {res_boosted_distill['multihop']:<18.4f}")
    print(f"{'Logic Chain (演繹)':<18} | {res_base['logic_chain']:<13.4f} | {res_boosted_base['logic_chain']:<15.4f} | {res_distill['logic_chain']:<16.4f} | {res_boosted_distill['logic_chain']:<18.4f}")
    print(f"{'Factuality QA':<18} | {res_base['qa_accuracy']*100:<12.1f}% | {res_boosted_base['qa_accuracy']*100:<14.1f}% | {res_distill['qa_accuracy']*100:<15.1f}% | {res_boosted_distill['qa_accuracy']*100:<17.1f}%")
    print("=" * 80)

    # JSON保存
    final_output = {
        "base_15b": res_base,
        "boosted_base_15b": res_boosted_base,
        "knn_distilled_15b": res_distill,
        "boosted_knn_distilled_15b": res_boosted_distill
    }
    save_path = "distillation_experiments/boosted_concept_forge_benchmark_report.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    print(f"[+] レポートを保存しました: {save_path}")

if __name__ == "__main__":
    main()
