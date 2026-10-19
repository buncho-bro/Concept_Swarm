import os
import sys
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 数学に長けたモデル: Qwen/Qwen2.5-7B-Instruct
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

print(f"[Math Theory Wilderness] Loading advanced math model: {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map="auto",
    local_files_only=True
).eval()

print("Model successfully loaded onto devices.")

# 語彙埋め込みテーブルの取得
embed_tokens = model.get_input_embeddings()
W_embed = embed_tokens.weight.data.float()  # (152064, 3584)
embed_device = embed_tokens.weight.device

# 数学・科学の自立語空間（3文字以上の単語）
math_vocab_to_id = {}
for tid in range(len(tokenizer)):
    raw_s = tokenizer.convert_ids_to_tokens(tid)
    decoded = tokenizer.decode([tid])
    s_strip = decoded.strip()
    if not s_strip.isalpha() or len(s_strip) < 3:
        continue
    
    if s_strip.isascii():
        has_space = (raw_s.startswith('Ġ') or decoded.startswith(' ') or decoded.startswith('Ġ'))
        if not has_space:
            continue
            
    clean_w = s_strip.lower()
    if clean_w not in math_vocab_to_id:
        math_vocab_to_id[clean_w] = tid

math_words = list(math_vocab_to_id.keys())
math_ids = torch.tensor([math_vocab_to_id[w] for w in math_words], device=embed_device, dtype=torch.long)
math_space = F.normalize(W_embed[math_ids].to(embed_device), p=2, dim=-1)
print(f"[Math Space] Built purified concept vocabulary with {len(math_words)} unique concepts.")

# 数学的理論構築のベンチマークケース (公理・前提 ➔ 期待される定理・帰結)
MATH_TEST_CASES = [
    {
        "field": "Abstract Algebra (抽象代数)",
        "pair": ["group", "commutative"],
        "expected": ["abelian", "ring", "field", "homomorphism", "symmetric", "algebra"]
    },
    {
        "field": "Topology (位相幾何学)",
        "pair": ["compact", "hausdorff"],
        "expected": ["normal", "closed", "regular", "paracompact", "tietze", "urysohn", "space"]
    },
    {
        "field": "Calculus & Analysis (解析学)",
        "pair": ["derivative", "integral"],
        "expected": ["fundamental", "calculus", "continuous", "leibniz", "riemann", "newton"]
    },
    {
        "field": "Linear Algebra (線形代数)",
        "pair": ["matrix", "eigenvalue"],
        "expected": ["diagonalization", "spectrum", "eigenvector", "determinant", "orthogonal", "spectral"]
    },
    {
        "field": "Number Theory (数論)",
        "pair": ["prime", "zeta"],
        "expected": ["riemann", "zeros", "euler", "dirichlet", "distribution", "hypothesis"]
    },
    {
        "field": "Differential Geometry (微分幾何)",
        "pair": ["manifold", "curvature"],
        "expected": ["riemannian", "geodesic", "metric", "tensor", "ricci", "einstein"]
    },
    {
        "field": "Probability Theory (確率論)",
        "pair": ["random", "variable"],
        "expected": ["distribution", "variance", "expectation", "probability", "density"]
    },
    {
        "field": "Complex Analysis (複素解析)",
        "pair": ["holomorphic", "integral"],
        "expected": ["cauchy", "residue", "contour", "analytic", "singularities"]
    }
]

def extract_math_trajectory(token_str: str):
    """Qwen2.5-7Bによる単一数学概念の全層軌跡 (L+1, 3584) の抽出"""
    inputs = tokenizer(token_str, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
        h_states = [F.normalize(h[0, -1, :].to(embed_device).float(), p=2, dim=-1) for h in out.hidden_states]
        traj = torch.stack(h_states, dim=0)
    return traj

def trace_mathematical_theory(traj_A, traj_B, step_scale=0.8, top_k=6):
    """
    数学概念AとBの全軌道相互作用を積分し、導出される定理・概念を空間から読み出す
    """
    num_layers = traj_A.shape[0]
    
    # 各層の局所速度ベクトル
    v_A = traj_A[1:] - traj_A[:-1]
    v_B = traj_B[1:] - traj_B[:-1]
    
    # 各層の相対軸
    d_mid = traj_B[:-1] - traj_A[:-1]
    d_mid_norm = F.normalize(d_mid, p=2, dim=-1)
    
    # 相互作用の直交成分
    delta_v = v_B - v_A
    proj = (delta_v * d_mid_norm).sum(dim=-1, keepdim=True) * d_mid_norm
    ortho_forces = delta_v - proj
    
    # 層の重み（深層ほど抽象的な数学的推論が成熟）
    layer_weights = torch.linspace(0.1, 1.0, steps=num_layers - 1, device=embed_device).unsqueeze(-1)
    weighted_forces = ortho_forces * layer_weights
    accumulated_force = F.normalize(weighted_forces.sum(dim=0), p=2, dim=-1)
    
    # 終着点（Layer final）
    end_A = traj_A[-1]
    end_B = traj_B[-1]
    center_end = F.normalize(0.5 * (end_A + end_B), p=2, dim=-1)
    
    # 合成された数学的定理点
    traced_end = F.normalize(center_end + step_scale * accumulated_force, p=2, dim=-1)
    
    # 空間読み出し（元概念への引力反発を軽度に適用）
    sim_trace = torch.matmul(traced_end, math_space.T)
    sim_A = torch.matmul(end_A, math_space.T)
    sim_B = torch.matmul(end_B, math_space.T)
    repulsion = torch.max(sim_A, sim_B)
    
    final_score = sim_trace - 0.3 * repulsion
    top_indices = torch.topk(final_score, k=top_k + 4).indices.tolist()
    
    results = [math_words[i] for i in top_indices]
    return results[:top_k]

print("\n========================================================")
print(f"Running Mathematical Theory Derivation Test (Qwen2.5-7B)...")
print("========================================================")

results_summary = []
for case in MATH_TEST_CASES:
    field = case["field"]
    w_A, w_B = case["pair"]
    expected_list = case["expected"]
    
    traj_A = extract_math_trajectory(w_A)
    traj_B = extract_math_trajectory(w_B)
    
    top_concepts = trace_mathematical_theory(traj_A, traj_B, step_scale=0.8, top_k=6)
    
    # ヒット判定
    hits = []
    for c in top_concepts:
        for exp in expected_list:
            if exp in c or c in exp:
                hits.append(c)
                break
                
    is_hit = len(hits) > 0
    status = "✅ 成功 (Theorem Hit)" if is_hit else "❌ 逸脱"
    
    print(f"\n[{field}]")
    print(f"  前提 (Premises) : '{w_A}' + '{w_B}'")
    print(f"  導出概念 (Top-6): {top_concepts}")
    print(f"  期待定理 (Expected): {expected_list}")
    print(f"  判定: {status} (一致: {hits})")
    
    results_summary.append({
        "field": field,
        "pair": [w_A, w_B],
        "top_concepts": top_concepts,
        "expected": expected_list,
        "hits": hits,
        "is_hit": is_hit
    })

total_hits = sum(1 for r in results_summary if r["is_hit"])
print(f"\n================== SUMMARY ==================")
print(f"Mathematical Theorem Derivation Accuracy: {total_hits}/{len(MATH_TEST_CASES)} ({total_hits/len(MATH_TEST_CASES)*100:.1f}%)")

with open("math_theory_test_results.json", "w", encoding="utf-8") as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print("Saved math theory results to: math_theory_test_results.json")
