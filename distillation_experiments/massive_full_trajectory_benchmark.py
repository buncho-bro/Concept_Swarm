import os
import sys
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

STUDENT_MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[Massive Full-Trajectory Benchmark] Initializing model on {device}...")
tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    STUDENT_MODEL_ID,
    dtype=torch.float16,
    device_map={"": device},
    local_files_only=True
).eval()

embed_tokens = model.get_input_embeddings()
W_embed = embed_tokens.weight.data.float()  # (vocab_size, 2048)

# 自立語空間の構築（3文字以上の自立語、または2文字以上のCJK漢字・かな）
purified_word_to_id = {}
for tid in range(len(tokenizer)):
    raw_s = tokenizer.convert_ids_to_tokens(tid)
    decoded = tokenizer.decode([tid])
    s_strip = decoded.strip()
    if not s_strip.isalpha():
        continue
    
    is_latin = s_strip.isascii()
    if is_latin:
        # 英語は先頭スペース(またはĠ)があり、かつ3文字以上（ic, gs, js などの極短ノイズを除外）
        has_space_prefix = (raw_s.startswith('Ġ') or decoded.startswith(' ') or decoded.startswith('Ġ'))
        if not has_space_prefix or len(s_strip) < 3:
            continue
    else:
        # 非ラテン（漢字・かななど）は2文字以上
        if len(s_strip) < 2:
            continue
    
    clean_word = s_strip.lower()
    if clean_word not in purified_word_to_id:
        purified_word_to_id[clean_word] = tid

words_list = list(purified_word_to_id.keys())
ids_tensor = torch.tensor([purified_word_to_id[w] for w in words_list], device=device, dtype=torch.long)
word_space = F.normalize(W_embed[ids_tensor], p=2, dim=-1)
print(f"[Word Space] Built clean purified concept space with {len(words_list)} unique words.")

from massive_causal_benchmark import BENCHMARK_PAIRS

def extract_full_trajectory(token_str: str):
    """入力トークン単体が始点から止まるまでの全層軌跡 (33, 2048) を取得"""
    inputs = tokenizer(token_str, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
        h_states = [F.normalize(h[0, -1, :].float(), p=2, dim=-1) for h in out.hidden_states]
        traj = torch.stack(h_states, dim=0)
    return traj

# 100ペアの単語軌跡を事前キャッシュ（高速化）
print("Extracting full trajectories for all benchmark tokens...")
cached_trajectories = {}
all_tokens = set()
for item in BENCHMARK_PAIRS:
    for t in item["pair"]:
        all_tokens.add(t)

for t in sorted(list(all_tokens)):
    cached_trajectories[t] = extract_full_trajectory(t)
print(f"Cached full trajectories for {len(cached_trajectories)} unique tokens.")

def trace_trajectory_interaction(traj_A, traj_B, step_scale=1.0, top_k=5):
    """全層の軌跡を丸ごと掛け合わせ、因果の終着点をトレース"""
    num_layers = traj_A.shape[0]
    
    # 局所速度ベクトル
    v_A = traj_A[1:] - traj_A[:-1]  # (32, 2048)
    v_B = traj_B[1:] - traj_B[:-1]  # (32, 2048)
    
    # 各層の相対軸
    d_mid = traj_B[:-1] - traj_A[:-1]
    d_mid_norm = F.normalize(d_mid, p=2, dim=-1)
    
    # 相互作用速度と直交成分
    delta_v = v_B - v_A
    proj = (delta_v * d_mid_norm).sum(dim=-1, keepdim=True) * d_mid_norm
    ortho_forces = delta_v - proj
    
    # 層ごとの重み付け（概念の成熟度合い）
    layer_weights = torch.linspace(0.1, 1.0, steps=num_layers - 1, device=device).unsqueeze(-1)
    weighted_forces = ortho_forces * layer_weights
    accumulated_force = F.normalize(weighted_forces.sum(dim=0), p=2, dim=-1)
    
    # 終着点（静止点）
    end_A = traj_A[-1]
    end_B = traj_B[-1]
    center_end = F.normalize(0.5 * (end_A + end_B), p=2, dim=-1)
    
    # 合成終着点
    traced_end = F.normalize(center_end + step_scale * accumulated_force, p=2, dim=-1)
    
    # 空間読み出し
    sim_trace = torch.matmul(traced_end, word_space.T)
    sim_A = torch.matmul(end_A, word_space.T)
    sim_B = torch.matmul(end_B, word_space.T)
    repulsion = torch.max(sim_A, sim_B)
    
    final_score = sim_trace - 0.35 * repulsion
    top_indices = torch.topk(final_score, k=top_k + 4).indices.tolist()
    
    results = [words_list[i] for i in top_indices]
    return results[:top_k]

# ========================================================
# ベンチマーク実行（複数スケール探索）
# ========================================================
scales = [0.6, 1.0, 1.4, 1.8]
benchmark_summary = {}

print(f"\n========================================================")
print(f"Running Full-Trajectory Benchmark on 100 pairs across scales {scales}...")
print(f"========================================================")

for sc in scales:
    hit_count = 0
    escape_count = 0
    details = []
    
    for idx, item in enumerate(BENCHMARK_PAIRS):
        t_A, t_B = item["pair"]
        expected_list = item["expected"]
        
        traj_A = cached_trajectories[t_A]
        traj_B = cached_trajectories[t_B]
        
        top_words = trace_trajectory_interaction(traj_A, traj_B, step_scale=sc, top_k=5)
        
        is_escaped = not any(w.lower() in [t_A.lower(), t_B.lower()] for w in top_words[:3])
        if is_escaped:
            escape_count += 1
            
        matched_hits = []
        for w in top_words:
            for exp in expected_list:
                if exp.lower() in w.lower() or w.lower() in exp.lower():
                    matched_hits.append(w)
                    break
        is_hit = len(matched_hits) > 0
        if is_hit:
            hit_count += 1
            
        details.append({
            "pair": f"{t_A}+{t_B}",
            "category": item["category"],
            "expected": expected_list,
            "top5": top_words,
            "is_escaped": is_escaped,
            "is_hit": is_hit,
            "hits": matched_hits
        })
        
    benchmark_summary[str(sc)] = {
        "hit_count": hit_count,
        "escape_count": escape_count,
        "hit_rate": hit_count / len(BENCHMARK_PAIRS),
        "escape_rate": escape_count / len(BENCHMARK_PAIRS),
        "details": details
    }
    print(f"Scale lambda={sc:3.1f}: Hit Rate = {hit_count:2d}/100 ({hit_count:4.1f}%), Escape Rate = {escape_count:2d}/100 ({escape_count:4.1f}%)")

output_json = "full_trajectory_benchmark_results_100.json"
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(benchmark_summary, f, ensure_ascii=False, indent=2)

print(f"\nFull benchmark results saved to: {output_json}")

# ベストスケールでの代表例表示
best_scale = max(scales, key=lambda s: benchmark_summary[str(s)]["hit_count"])
print(f"\n=== Detailed Samples at Best Scale lambda={best_scale} ===")
for d in benchmark_summary[str(best_scale)]["details"][:25]:
    status = "✅" if d["is_hit"] else "❌"
    print(f"  {status} {d['pair']:20s} ➔ {d['top5']} (Hits: {d['hits']})")
