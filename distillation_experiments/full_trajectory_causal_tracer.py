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

print(f"[Full Trajectory Causal Tracer] Initializing model on {device}...")
tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    STUDENT_MODEL_ID,
    dtype=torch.float16,
    device_map={"": device},
    local_files_only=True
).eval()

embed_tokens = model.get_input_embeddings()
W_embed = embed_tokens.weight.data.float()  # (vocab_size, 2048)

# 自立語空間の構築（英語自立語 + CJK自立語、サブワード断片を排除）
purified_word_to_id = {}
for tid in range(len(tokenizer)):
    raw_s = tokenizer.convert_ids_to_tokens(tid)
    decoded = tokenizer.decode([tid])
    s_strip = decoded.strip()
    if len(s_strip) < 2 or not s_strip.isalpha():
        continue
    
    is_latin = s_strip.isascii()
    if is_latin:
        has_space_prefix = (raw_s.startswith('Ġ') or decoded.startswith(' ') or decoded.startswith('Ġ'))
        if not has_space_prefix:
            continue
    
    clean_word = s_strip.lower()
    if clean_word not in purified_word_to_id:
        purified_word_to_id[clean_word] = tid

words_list = list(purified_word_to_id.keys())
ids_tensor = torch.tensor([purified_word_to_id[w] for w in words_list], device=device, dtype=torch.long)
word_space = F.normalize(W_embed[ids_tensor], p=2, dim=-1)
print(f"[Word Space] Built clean concept space with {len(words_list)} unique words.")

def extract_full_trajectory(token_str: str):
    """
    入力トークン単体が始点から止まるまでの全層軌跡 (L+1, 2048) を取得
    """
    inputs = tokenizer(token_str, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
        # 隠れ層 0 (embedding) から 32 (final layer) までの全33個の状態
        h_states = [F.normalize(h[0, -1, :].float(), p=2, dim=-1) for h in out.hidden_states]
        # (num_layers, hidden_dim)
        traj = torch.stack(h_states, dim=0)
    return traj

def combine_trajectories_and_trace(traj_A, traj_B, step_scale=1.0, top_k=6):
    """
    各単語が止まるまでの2本の軌跡 (traj_A, traj_B) を丸ごと組み合わせ、
    第3の因果の軌跡を生成して終着点の単語を読み出す
    """
    num_layers = traj_A.shape[0]  # 通常33層 (0〜32)
    
    # 各層での局所速度ベクトル（接線）: v[l] = h[l+1] - h[l]
    v_A = traj_A[1:] - traj_A[:-1]  # (32, 2048)
    v_B = traj_B[1:] - traj_B[:-1]  # (32, 2048)
    
    # 各層での2単語間の相対軸ベクトル d[l]
    d_mid = traj_B[:-1] - traj_A[:-1]  # (32, 2048)
    d_mid_norm = F.normalize(d_mid, p=2, dim=-1)
    
    # 各層での運動の差（相互作用速度）
    delta_v = v_B - v_A  # (32, 2048)
    
    # 相対軸に対する直交成分（軸に垂直に飛び出す因果力線）
    proj = (delta_v * d_mid_norm).sum(dim=-1, keepdim=True) * d_mid_norm
    ortho_force_layers = delta_v - proj  # (32, 2048)
    
    # 層ごとの重み付け（中層〜深層で概念の相互作用が成熟するため、線形重み付け）
    layer_weights = torch.linspace(0.1, 1.0, steps=num_layers - 1, device=device).unsqueeze(-1)
    weighted_forces = ortho_force_layers * layer_weights
    
    # 累積因果力（軌道全体を通じて蓄積された相互作用エネルギー）
    accumulated_force = F.normalize(weighted_forces.sum(dim=0), p=2, dim=-1)
    
    # 2単語がそれぞれ止まった終着点 (Layer 32)
    end_A = traj_A[-1]
    end_B = traj_B[-1]
    center_end = F.normalize(0.5 * (end_A + end_B), p=2, dim=-1)
    
    # 合成された第3の因果の終着点
    traced_end = F.normalize(center_end + step_scale * accumulated_force, p=2, dim=-1)
    
    # 空間から単語を読み出し（元単語の終着点に対する適度な反発場も併用）
    sim_trace = torch.matmul(traced_end, word_space.T)
    sim_A = torch.matmul(end_A, word_space.T)
    sim_B = torch.matmul(end_B, word_space.T)
    repulsion = torch.max(sim_A, sim_B)
    
    final_score = sim_trace - 0.35 * repulsion
    top_indices = torch.topk(final_score, k=top_k + 4).indices.tolist()
    
    results = [words_list[i] for i in top_indices]
    return results[:top_k]

# 代表的なペアで即時テスト
test_pairs = [
    ("sun", "ice"),
    ("fire", "wood"),
    ("cow", "grass"),
    ("lightning", "forest"),
    ("earthquake", "ocean"),
    ("light", "prism"),
    ("crime", "police")
]

print("\n================== FULL TRAJECTORY COMBINATION TEST ==================")
for word_A, word_B in test_pairs:
    traj_A = extract_full_trajectory(word_A)
    traj_B = extract_full_trajectory(word_B)
    
    results = combine_trajectories_and_trace(traj_A, traj_B, step_scale=1.2, top_k=5)
    print(f"[{word_A} + {word_B}] ➔ {results}")
