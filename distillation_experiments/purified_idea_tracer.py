import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

STUDENT_MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[Purified Idea Tracer] Initializing on {device}...")
tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    STUDENT_MODEL_ID,
    dtype=torch.float16,
    device_map={"": device},
    local_files_only=True
).eval()

# 1. 【データ量の適正化】純粋で重複のない単語空間の構築
# 表記揺れ（_cat, (cat など）やサブワード断片を排除し、正しい言葉の場所のみを保持
embed_tokens = model.get_input_embeddings()
W_embed = embed_tokens.weight.data.float()  # (vocab_size, 2048)

clean_word_to_id = {}
for tid in range(len(tokenizer)):
    t_str = tokenizer.decode([tid]).strip()
    # 2文字以上、アルファベットのみ（または純粋な漢字/ひらがな）、小文字正規化
    if len(t_str) >= 2 and t_str.isalpha():
        clean_word = t_str.lower()
        if clean_word not in clean_word_to_id:
            clean_word_to_id[clean_word] = tid

selected_words = list(clean_word_to_id.keys())
selected_ids = torch.tensor([clean_word_to_id[w] for w in selected_words], device=device, dtype=torch.long)
# 語彙空間（各単語が厳密に固有の場所を持つ）
word_space = F.normalize(W_embed[selected_ids], p=2, dim=-1)  # (N_clean, 2048)
print(f"[Word Space] Built clean purified word space: {len(selected_words)} unique concepts (compact & accurate).")

def observe_token_momentum(token_str: str):
    """
    単一トークン固有の揺らぎと運動量ベクトルを抽出
    """
    inputs = tokenizer(token_str, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
        h_states = out.hidden_states
        # 浅層（語彙定義）から中層（関係性）への固有遷移ベクトル
        h_start = F.normalize(h_states[0][0, -1, :].float(), p=2, dim=-1)
        h_mid = F.normalize(h_states[18][0, -1, :].float(), p=2, dim=-1)
        h_deep = F.normalize(h_states[32][0, -1, :].float(), p=2, dim=-1)
        
        # 固有の揺らぎ速度ベクトル（モメンタム）
        momentum = F.normalize((h_mid - h_start) + (h_deep - h_mid), p=2, dim=-1)
    return h_start, momentum

def trace_causal_interaction(tokens: list[str], top_k: int = 6):
    """
    言葉の違い（線の差異）と相互作用直交成分を掛け合わせ、因果の場所をトレースする
    """
    print(f"\n==================================================")
    print(f"入力トークン: {tokens}")
    
    positions = []
    momenta = []
    for t_str in tokens:
        pos, mom = observe_token_momentum(t_str)
        positions.append(pos)
        momenta.append(mom)

    if len(tokens) == 2:
        pos_A, pos_B = positions[0], positions[1]
        mom_A, mom_B = momenta[0], momenta[1]
        
        # 1. 2つの言葉の差分ベクトル（直線軸）
        direct_axis = F.normalize(pos_B - pos_A, p=2, dim=-1)
        center_pos = F.normalize(0.5 * (pos_A + pos_B), p=2, dim=-1)
        
        # 2. 揺らぎの違い（モメンタムの差分・相互作用線）
        interaction_diff = mom_B - mom_A
        
        # 3. 直線軸に垂直な直交成分（相互作用が外へと向かう因果の力線）
        # 内積成分（互いを打ち消すか引っぱる力）を除去し、真の相互作用方向を抽出
        proj_on_axis = torch.dot(interaction_diff, direct_axis) * direct_axis
        orthogonal_causal_force = F.normalize(interaction_diff - proj_on_axis, p=2, dim=-1)
        
        # 4. 合成された線をトレース（中心点から直交因果方向へ進む）
        steps = [0.0, 0.35, 0.70, 1.05]
        print(f"  [幾何学的トレースの推移]")
        for s_idx, step_scale in enumerate(steps):
            # 中心から因果力線に沿って空間を移動
            traced_point = F.normalize(center_pos + step_scale * orthogonal_causal_force, p=2, dim=-1)
            
            # その場所にある単語を読み出し（入力自身は除外して純粋な因果結果を観測）
            sims = torch.matmul(traced_point, word_space.T)
            top_indices = torch.topk(sims, k=top_k + 4).indices.tolist()
            
            # 入力語そのものを除いた純粋な結果
            result_words = []
            for idx in top_indices:
                w = selected_words[idx]
                if w not in [t.lower() for t in tokens] and w not in result_words:
                    result_words.append(w)
                if len(result_words) >= top_k:
                    break
                    
            if s_idx == 0:
                print(f"    - 始点（両単語の中間点）: {result_words[:4]}")
            elif s_idx == len(steps) - 1:
                print(f"    - ★ 終着地点（因果の帰結場所）: {result_words}")
            else:
                print(f"    - 相互作用進行中 (step {s_idx}): {result_words[:4]}")

# 正しい因果関係の代表テストケース
test_cases = [
    ["fire", "wood"],       # 火 ＋ 木 ➔ 燃焼・灰・炭
    ["sun", "ice"],         # 太陽 ＋ 氷 ➔ 融解・水・蒸発
    ["rain", "umbrella"],   # 雨 ＋ 傘 ➔ 濡れ防止・天候・保護
    ["cat", "fish"],        # 猫 ＋ 魚 ➔ 捕食・餌・水生
    ["demand", "supply"],   # 需要 ＋ 供給 ➔ 価格・市場・均衡
]

for tc in test_cases:
    trace_causal_interaction(tc)
