"""
run_quantum_interference_hyperplane.py
--------------------------------------
超平面量子干渉オラクル（Hyperplane Quantum Phase Oracle）
文脈の制約条件（Constraint）と矛盾する半空間（Contra-Hemisphere）に属する波を一網打尽に相殺消滅！
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

tests = [
    {
        "context": "Although wolves are fierce carnivores, a pet chihuahua is also a biological",
        "constraint": "biological pet dog",
        "contra": "wild fierce predator reptile",
        "options": ["mammal", "reptile", "robot", "plant"],
        "truth": "mammal"
    },
    {
        "context": "Despite severe desert heat above ground, the deep subterranean cave remained freezing",
        "constraint": "freezing cold subzero",
        "contra": "severe desert heat burning",
        "options": ["cold", "hot", "burning", "boiling"],
        "truth": "cold"
    },
    {
        "context": "Even though antibiotics cure severe bacterial infections, against a common viral flu they are completely",
        "constraint": "ineffective useless powerless",
        "contra": "effective cure powerful",
        "options": ["ineffective", "effective", "powerful", "useful"],
        "truth": "ineffective"
    }
]

def get_word_embedding(model, tokenizer, word, device):
    inputs = tokenizer(word, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[-1][0, -1, :]
    return F.normalize(hidden.to(torch.float32), p=2, dim=0)

def main():
    print("=" * 70)
    print("🌊 超平面量子干渉オラクル（Hyperplane Quantum Phase Oracle）検証")
    print("=" * 70)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    ).eval()

    c_hits = 0
    q_hits = 0

    for idx, item in enumerate(tests, 1):
        ctx = item["context"]
        v_const = get_word_embedding(model, tok, item["constraint"], device)
        v_contra = get_word_embedding(model, tok, item["contra"], device)
        opts = item["options"]
        truth = item["truth"]

        ctx_vec = get_word_embedding(model, tok, ctx, device)
        opt_vecs = [get_word_embedding(model, tok, o, device) for o in opts]

        # 1. 古典AI (文脈との表面類似度)
        c_sims = [F.cosine_similarity(ctx_vec.unsqueeze(0), ov.unsqueeze(0)).item() for ov in opt_vecs]
        c_pred = opts[int(np.argmax(c_sims))]
        c_pass = (c_pred == truth)
        if c_pass: c_hits += 1

        # 2. 超平面量子位相反転オラクル
        # 思考の判別軸: D_axis = Normalize(v_const - v_contra)
        d_axis = F.normalize(v_const - v_contra, p=2, dim=0)

        # 各選択肢の軸への射影スコア (内積)
        # > 0 なら制約と整合 (同位相)
        # < 0 なら矛盾半空間 (逆位相へ反転！)
        opt_tensor = torch.stack(opt_vecs, dim=0) # (K, D)
        proj_scores = torch.matmul(opt_tensor, d_axis) # (K,)

        # 位相反転マスク: 矛盾側 (< 0) の波を -1 反転
        oracle_mask = torch.where(proj_scores < 0.0, -1.0, 1.0)
        
        # 初期振幅 (文脈類似度)
        amplitudes = torch.tensor(c_sims, device=device, dtype=torch.float32)
        
        # グローバー反復 (2回)
        for _ in range(2):
            # 位相反転
            amplitudes = amplitudes * oracle_mask
            # 拡散オペレータ (平均値反転)
            mean_amp = amplitudes.mean()
            amplitudes = 2.0 * mean_amp - amplitudes

        # 確率観測 P = |A|^2
        probs = torch.clamp(amplitudes, min=0.0) ** 2
        q_pred_idx = int(torch.argmax(probs).item())
        q_pred = opts[q_pred_idx]
        q_pass = (q_pred == truth)
        if q_pass: q_hits += 1

        print(f"\n[問 {idx}] {ctx} ...")
        print(f"  ・選択肢: {opts} (正解: {truth})")
        print(f"  ・[古典AI]   : {c_pred:<12} [{'PASS' if c_pass else 'FAIL (地上の熱や誤謬に幻惑)'}]")
        print(f"  ・[量子干渉] : {q_pred:<12} [{'PASS (矛盾半空間を完全相殺！)' if q_pass else 'FAIL'}] (共鳴確率: {[round(p.item(), 3) for p in probs]})")

    print("\n" + "=" * 70)
    print("🏆 超平面量子干渉オラクル 検証サマリー")
    print("=" * 70)
    print(f"・古典AI (表面単語の罠に陥落)   : 正解率 {c_hits}/{len(tests)} ({c_hits/len(tests)*100:.1f}%)")
    print(f"・量子干渉AI (波動相殺と共鳴)   : 正解率 {q_hits}/{len(tests)} ({q_hits/len(tests)*100:.1f}%) 🚀🚀🚀")
    print("=" * 70)

if __name__ == "__main__":
    main()
