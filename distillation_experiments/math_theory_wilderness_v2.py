import os
import sys
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

print(f"[Math Theory Wilderness v2] Initializing on {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map="auto",
    local_files_only=True
).eval()

# 語彙の数学的純化マスクの構築
# 記号、1〜2文字、非英字、一般的な助詞・接続詞を排除し、
# 3文字以上の純粋な実質概念（名詞・形容詞・数学用語）のトークンIDのみを許可
stop_words = {"the", "and", "that", "for", "with", "from", "this", "are", "not", "have", "can", "will", "which", "into", "also"}
valid_token_ids = []
id_to_concept = {}

for tid in range(len(tokenizer)):
    raw = tokenizer.convert_ids_to_tokens(tid)
    dec = tokenizer.decode([tid])
    s = dec.strip().lower()
    # 英字のみ、3文字以上、ストップワード除外
    if s.isascii() and s.isalpha() and len(s) >= 3 and s not in stop_words:
        # 先頭スペースまたはĠがある自立語
        if raw.startswith('Ġ') or dec.startswith(' '):
            valid_token_ids.append(tid)
            id_to_concept[tid] = s

valid_ids_tensor = torch.tensor(valid_token_ids, device=model.device, dtype=torch.long)
print(f"[Concept Mask] Filtered {len(valid_token_ids)} pure mathematical/academic concept tokens.")

MATH_TEST_CASES = [
    {
        "field": "Abstract Algebra (抽象代数)",
        "pair": ["group", "commutative"],
        "expected": ["abelian", "ring", "field", "homomorphism", "symmetric", "algebra", "subgroup"]
    },
    {
        "field": "Linear Algebra (線形代数)",
        "pair": ["matrix", "eigenvalue"],
        "expected": ["diagonalization", "spectrum", "eigenvector", "determinant", "orthogonal", "spectral", "operator"]
    },
    {
        "field": "Calculus & Analysis (解析学)",
        "pair": ["derivative", "integral"],
        "expected": ["fundamental", "calculus", "continuous", "leibniz", "riemann", "newton", "differential"]
    },
    {
        "field": "Topology (位相幾何学)",
        "pair": ["compact", "hausdorff"],
        "expected": ["normal", "closed", "regular", "paracompact", "tietze", "urysohn", "space", "manifold"]
    },
    {
        "field": "Differential Geometry (微分幾何)",
        "pair": ["manifold", "curvature"],
        "expected": ["riemannian", "geodesic", "metric", "tensor", "ricci", "einstein", "flat"]
    },
    {
        "field": "Number Theory (数論)",
        "pair": ["prime", "zeta"],
        "expected": ["riemann", "zeros", "euler", "dirichlet", "distribution", "hypothesis", "function"]
    },
    {
        "field": "Probability Theory (確率論)",
        "pair": ["random", "variable"],
        "expected": ["distribution", "variance", "expectation", "probability", "density", "sample"]
    },
    {
        "field": "Complex Analysis (複素解析)",
        "pair": ["holomorphic", "integral"],
        "expected": ["cauchy", "residue", "contour", "analytic", "singularities", "differential"]
    }
]

def extract_trajectory_hidden(token_str: str):
    inp = tokenizer(token_str, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.model(**inp, output_hidden_states=True)
        # 全層の隠れ状態
        h_states = [F.normalize(h[0, -1, :].float(), p=2, dim=-1) for h in out.hidden_states]
        traj = torch.stack(h_states, dim=0)  # (num_layers, 3584)
    return traj

def trace_math_concept(traj_A, traj_B, step_scale=1.0, top_k=6):
    num_layers = traj_A.shape[0]
    
    # 速度ベクトル
    v_A = traj_A[1:] - traj_A[:-1]
    v_B = traj_B[1:] - traj_B[:-1]
    
    # 相対軸
    d_mid = traj_B[:-1] - traj_A[:-1]
    d_mid_norm = F.normalize(d_mid, p=2, dim=-1)
    
    # 直交相互作用力線
    delta_v = v_B - v_A
    proj = (delta_v * d_mid_norm).sum(dim=-1, keepdim=True) * d_mid_norm
    ortho_forces = delta_v - proj
    
    # 層の重み付け
    layer_weights = torch.linspace(0.1, 1.0, steps=num_layers - 1, device=model.device).unsqueeze(-1)
    weighted_forces = ortho_forces * layer_weights
    accumulated_force = F.normalize(weighted_forces.sum(dim=0), p=2, dim=-1)
    
    # 終着点の重心
    end_A = traj_A[-1]
    end_B = traj_B[-1]
    center_end = F.normalize(0.5 * (end_A + end_B), p=2, dim=-1)
    
    # 合成された因果定理点
    traced_point = F.normalize(center_end + step_scale * accumulated_force, p=2, dim=-1)
    
    # LM Head を通してロジットを算出
    with torch.no_grad():
        logits_traced = model.lm_head(traced_point.to(model.lm_head.weight.dtype))
        logits_A = model.lm_head(end_A.to(model.lm_head.weight.dtype))
        logits_B = model.lm_head(end_B.to(model.lm_head.weight.dtype))
        
        # 自立概念マスクの適用
        masked_logits = logits_traced[valid_ids_tensor]
        repulsion = torch.max(logits_A[valid_ids_tensor], logits_B[valid_ids_tensor])
        
        # 元単語引力に対する適度な相殺
        final_scores = masked_logits - 0.25 * repulsion
        top_k_indices = torch.topk(final_scores, k=top_k + 4).indices.tolist()
        
    chosen_concepts = []
    seen = set()
    for idx in top_k_indices:
        real_tid = valid_token_ids[idx]
        w = id_to_concept[real_tid]
        if w not in seen:
            seen.add(w)
            chosen_concepts.append(w)
            if len(chosen_concepts) == top_k:
                break
                
    return chosen_concepts

print("\n================== MATHEMATICAL THEORY DERIVATION (v2) ==================")
v2_results = []
for case in MATH_TEST_CASES:
    field = case["field"]
    w_A, w_B = case["pair"]
    exp_list = case["expected"]
    
    traj_A = extract_trajectory_hidden(w_A)
    traj_B = extract_trajectory_hidden(w_B)
    
    derived = trace_math_concept(traj_A, traj_B, step_scale=1.0, top_k=6)
    
    # 一致判定
    hits = []
    for d in derived:
        for exp in exp_list:
            if exp in d or d in exp:
                hits.append(d)
                break
                
    is_hit = len(hits) > 0
    mark = "✅ 成功 (Hit)" if is_hit else "❌"
    
    print(f"\n[{field}]")
    print(f"  前提: '{w_A}' + '{w_B}'")
    print(f"  導出された数学概念 (Top-6): {derived}")
    print(f"  期待される定理・帰結: {exp_list}")
    print(f"  判定: {mark} (一致: {hits})")
    
    v2_results.append({
        "field": field,
        "pair": [w_A, w_B],
        "derived": derived,
        "expected": exp_list,
        "hits": hits,
        "is_hit": is_hit
    })

total_hit = sum(1 for r in v2_results if r["is_hit"])
print(f"\n================== FINAL BENCHMARK SUMMARY ==================")
print(f"Mathematical Theorem Derivation Accuracy: {total_hit}/{len(MATH_TEST_CASES)} ({total_hit/len(MATH_TEST_CASES)*100:.1f}%)")

with open("math_theory_v2_results.json", "w", encoding="utf-8") as f:
    json.dump(v2_results, f, ensure_ascii=False, indent=2)
