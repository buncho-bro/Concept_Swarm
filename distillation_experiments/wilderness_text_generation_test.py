import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

STUDENT_MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[Wilderness Text Generation Test] Initializing on {device}...")
tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    STUDENT_MODEL_ID,
    dtype=torch.float16,
    device_map={"": device},
    local_files_only=True
).eval()

# 1. 思考原野データのロード
with open("thought_wilderness_data.json", "r", encoding="utf-8") as f:
    wilderness_data = json.load(f)

concept_words = wilderness_data["words"]
word_to_coord = {w: np.array(coord, dtype=np.float32) for w, coord in wilderness_data["coords"].items()}
coord_matrix = np.array([word_to_coord[w] for w in concept_words], dtype=np.float32)  # (N, 2)
N_words = len(concept_words)
print(f"[Wilderness] Loaded {N_words} concepts with 2D stabilized coordinates.")

# 2. 思考原野における「文章生成ダイナミクス」の設計
# 単語列 w_1, w_2, ..., w_t が与えられたとき、
# 原野上での「慣性（モメンタム）」と「局所ポテンシャルの谷筋」に沿って次の単語座標 P_{t+1} を予測する。

def generate_sentence_in_wilderness(start_word: str, max_steps: int = 8, momentum_decay: float = 0.6):
    """
    思考原野上の力学軌道による連続単語生成
    """
    start_word = start_word.lower()
    if start_word not in word_to_coord:
        return [f"'{start_word}' not in wilderness"]
    
    generated_tokens = [start_word]
    current_pos = word_to_coord[start_word].copy()
    current_momentum = np.zeros(2, dtype=np.float32)
    
    # 訪問済み単語の過剰ループを防ぐ局所斥力履歴
    history_coords = [current_pos]
    
    for step in range(max_steps):
        # 1. 現在位置の周囲にある最近傍候補たち
        dists = np.linalg.norm(coord_matrix - current_pos, axis=-1)
        
        # 2. 思考原野のポテンシャル勾配:
        # 直前トークンからの推進力 + 過去トークンへの反発（同じ場所での停滞防止）
        repulsion_force = np.zeros(2, dtype=np.float32)
        for h_pos in history_coords[-3:]:  # 直近3単語から離れる斥力
            diff = current_pos - h_pos
            d = np.linalg.norm(diff) + 1e-4
            repulsion_force += (diff / (d**2 + 0.5)) * 0.8
            
        # 進行方向（慣性と斥力の合成）
        current_momentum = current_momentum * momentum_decay + repulsion_force
        norm_m = np.linalg.norm(current_momentum)
        if norm_m > 2.0:
            current_momentum = current_momentum / norm_m * 2.0
            
        next_target_pos = current_pos + current_momentum
        
        # 3. 目的座標に最も自然に適合する単語の検索
        # 距離スコア - 直前単語ペナルティ
        candidate_scores = np.linalg.norm(coord_matrix - next_target_pos, axis=-1)
        
        # 直前数単語の除外
        for past_w in generated_tokens[-2:]:
            if past_w in concept_words:
                idx = concept_words.index(past_w)
                candidate_scores[idx] += 10.0
                
        best_idx = np.argmin(candidate_scores)
        chosen_word = concept_words[best_idx]
        
        generated_tokens.append(chosen_word)
        current_pos = word_to_coord[chosen_word].copy()
        history_coords.append(current_pos)
        
        # 終了判定（自然なピリオド相当または特定終了概念）
        if len(generated_tokens) >= max_steps:
            break
            
    return generated_tokens

# 3. 通常のLLM（33層トランスフォーマー自己回帰）との比較検証
def generate_llm_baseline(prompt: str, max_tokens: int = 8):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()

# 4. 複数シチュエーションでのテスト
test_prompts = [
    ("sun", "The sun"),
    ("fire", "The fire"),
    ("rain", "Heavy rain"),
    ("crime", "The crime"),
    ("cow", "A cow"),
    ("earthquake", "An earthquake"),
    ("cold", "Extreme cold")
]

print("\n================== WILDERNESS TEXT GENERATION STRESS-TEST ==================")
results = []
for start_w, full_prompt in test_prompts:
    wilderness_words = generate_sentence_in_wilderness(start_w, max_steps=7)
    wilderness_phrase = " ".join(wilderness_words)
    
    llm_phrase = generate_llm_baseline(full_prompt, max_tokens=7)
    
    print(f"\n[Prompt Starting: '{start_w}']")
    print(f"  🌲 思考原野による生成: \"{wilderness_phrase}\"")
    print(f"  🤖 LLM通常生成 (33層) : \"{llm_phrase}\"")
    
    results.append({
        "start": start_w,
        "wilderness": wilderness_phrase,
        "llm": llm_phrase
    })

# 5. 「致命的なずれ（Drift）」の定量的測定
# 思考原野で生成された単語列の意味的一貫性（意味空間の散逸度）
drift_scores = []
for r in results:
    w_list = r["wilderness"].split()
    # 思考原野上の座標移動分散（過度なジャンプや無限遠への発散がないか）
    pts = [word_to_coord[w] for w in w_list if w in word_to_coord]
    if len(pts) >= 2:
        step_dists = [np.linalg.norm(pts[i+1] - pts[i]) for i in range(len(pts)-1)]
        max_jump = max(step_dists)
        mean_jump = np.mean(step_dists)
        drift_scores.append((r["start"], mean_jump, max_jump))

print("\n================== DRIFT & COHERENCE METRICS ==================")
print("Step Jump Distance on Wilderness (Stability Check):")
for start_w, mean_j, max_j in drift_scores:
    status = "✅ 安定 (Stable)" if max_j < 6.0 else "⚠️ 乖離あり (Drift)"
    print(f"  Prompt '{start_w:10s}': Mean Step = {mean_j:4.2f}, Max Jump = {max_j:4.2f} ➔ {status}")

with open("wilderness_generation_test_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
