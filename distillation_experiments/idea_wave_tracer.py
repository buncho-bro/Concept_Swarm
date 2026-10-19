import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

STUDENT_MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[Idea Wave Tracer] Initializing on {device}...")
tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    STUDENT_MODEL_ID,
    dtype=torch.float16,
    device_map={"": device},
    local_files_only=True
).eval()

# 1. 「場所そのものが単語を持っている空間」の構築
# 各単語トークンが空間上の1つの点（埋め込みベクトル）に対応する
embed_tokens = model.get_input_embeddings()
W_embed = embed_tokens.weight.data.float()  # (vocab_size, 2048)
vocab_size = W_embed.shape[0]

# 意味のある主要語彙（英単語・日本語単語）のマスク作成
# 記号や空文字を排除し、場所と単語の対応辞書を整える
valid_token_ids = []
token_words = []
for tid in range(min(vocab_size, 60000)):
    w = tokenizer.decode([tid]).strip()
    if len(w) >= 2 and not w.startswith("<|"):
        valid_token_ids.append(tid)
        token_words.append(w)

valid_token_ids = torch.tensor(valid_token_ids, device=device, dtype=torch.long)
valid_space = F.normalize(W_embed[valid_token_ids], p=2, dim=-1)  # (V_valid, 2048)
print(f"[Word Space] Built word-space with {len(valid_token_ids)} located words.")

def observe_token_fluctuation_idea(token_str: str, n_steps: int = 5, delta: float = 0.05):
    """
    【ステップ1: 単一トークン固有の揺らぎ（イデア軌跡）を観測】
    入力されるのは1単語のみ。
    そのトークン自身がモデルの幾何空間（Layer 1..L）を通過するときに
    自然に発生する微小な層間ドリフト（内部の揺らぎの軌跡）を観測する。
    """
    inputs = tokenizer(token_str, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
        # 各隠れ層 (L=0..36) でのこのトークン自身の状態ベクトルを収集
        hidden_states = out.hidden_states
        
        # 代表的な層間遷移（浅層 -> 中層 -> 深層）から揺らぎの軌跡（ベクトル線）を抽出
        steps_idx = [0, 6, 12, 18, 24, 30, 36]
        traj_points = []
        for l_idx in steps_idx:
            h = hidden_states[l_idx][0, -1, :].float()  # (2048,)
            traj_points.append(F.normalize(h, p=2, dim=-1))
            
    traj_tensor = torch.stack(traj_points)  # (n_steps, 2048)
    # 揺らぎの運動方向（速度ベクトル線: 各ステップの差分）
    delta_vectors = traj_tensor[1:] - traj_tensor[:-1]  # (n_steps-1, 2048)
    return traj_tensor, delta_vectors

def trace_synthesized_line_to_words(tokens: list[str], top_k: int = 8):
    """
    【ステップ2〜4: 揺らぎの違いの掛け合わせ ➔ 空間への落とし込み（トレース）】
    """
    print(f"\n==================================================")
    print(f"入力トークン群: {tokens}")
    
    # 1. 各トークンから独立に固有の揺らぎ（軌跡＝イデア）を観測
    trajectories = []
    deltas = []
    for t_str in tokens:
        t_traj, t_delta = observe_token_fluctuation_idea(t_str)
        trajectories.append(t_traj)
        deltas.append(t_delta)
        
        # その単語単独の終着点にある場所の単語を確認
        sim_solo = torch.matmul(t_traj[-1], valid_space.T)
        top_solo_idx = torch.topk(sim_solo, k=3).indices.tolist()
        solo_words = [token_words[i] for i in top_solo_idx]
        print(f"  ・[{t_str}] 単体の揺らぎ終着点: {solo_words}")

    # 2. 言葉の違い（それぞれの揺らぎの線の違い）を表す
    # トークン i とトークン j の揺らぎベクトルの差異・関係線
    diff_lines = []
    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):
            # 揺らぎの軌跡全体の幾何学的差異（線の違い）
            line_diff = deltas[j] - deltas[i]  # (steps-1, 2048)
            diff_lines.append(line_diff)

    # 3. それらの線（イデアの違い）を掛け合わせ、1本の合成線を紡ぎ出す
    if len(diff_lines) == 0:
        # 単一単語の場合はその単語自身の揺らぎ線
        synthesized_line = deltas[0]
        start_point = trajectories[0][0]
    else:
        # 複数単語の違いの線をテンソル的に掛け合わせ・干渉させて合成
        # 相対差異線の総和と、重心からの固有干渉
        stacked_diffs = torch.stack(diff_lines)  # (pairs, steps-1, 2048)
        # 非線形な干渉（サイン・コサイン位相での掛け合わせ）
        interaction = torch.mean(stacked_diffs, dim=0)  # (steps-1, 2048)
        
        # 全トークンの初期位置の合成から出発する
        start_point = F.normalize(torch.stack([t[0] for t in trajectories]).mean(dim=0), p=2, dim=-1)
        synthesized_line = interaction

    # 4. 【トレース】合成された線を「場所そのものが単語を持つ空間」に落とし込む
    # 線の始点から、掛け合わされた揺らぎの運動に沿って空間をトレース（なぞる）
    curr_point = start_point
    traced_points = [curr_point]
    for step in range(synthesized_line.shape[0]):
        step_dir = F.normalize(synthesized_line[step], p=2, dim=-1)
        # 空間上の測地線移動（線のトレース）
        curr_point = F.normalize(curr_point + 0.45 * step_dir, p=2, dim=-1)
        traced_points.append(curr_point)

    # 5. 線が通過した場所、および最終到達場所にある単語を読み出し
    print(f"\n  [線のトレース結果]")
    for step_i, pt in enumerate(traced_points):
        sims = torch.matmul(pt, valid_space.T)
        top_indices = torch.topk(sims, k=top_k).indices.tolist()
        traced_words = [token_words[idx] for idx in top_indices]
        if step_i == 0:
            print(f"    - 始点（言葉の出会い地点）: {traced_words[:5]}")
        elif step_i == len(traced_points) - 1:
            print(f"    - ★ 終着地点（合わさった線の帰結）: {traced_words}")
        else:
            print(f"    - 中間通過点 Step {step_i}: {traced_words[:4]}")

# 複数の実験ケースで検証
test_cases = [
    ["猫", "魚"],                     # 捕食・生物関係
    ["太陽", "氷"],                   # 融解・熱力学的因果
    ["雨", "傘"],                     # 道具・防護関係
    ["火", "木"],                     # 燃焼因果
    ["需要", "供給"],                 # 経済因果
]

for tc in test_cases:
    trace_synthesized_line_to_words(tc)
