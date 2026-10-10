"""
【ファイル概要】
step7_fractal_routing_demo.py

フラクタル構造を用いた新しいMoEルーティング手法のデモンストレーション。大局的な概念から局所概念へ再帰的にExpertを選択します。
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ==========================================
# 1. 谷 (Valleys) の定義
# マクロ（大分類）とミクロ（詳細データ）の分離
# ==========================================
VALLEYS = {
    "Science": ["Physics", "Quantum", "Gravity", "Biology", "Chemistry", "Mathematics", "Geometry", "Astronomy", "Machine", "Technology"],
    "Society": ["Economy", "Law", "Nation", "Democracy", "Market", "Currency", "Industry", "Culture", "Trade", "Capital"],
    "Emotion": ["Love", "Fear", "Joy", "Sadness", "Hope", "Despair", "Anger", "Happiness", "Loneliness", "Pleasure"],
    "Nature":  ["Forest", "Ocean", "Mountain", "Climate", "Weather", "Planet", "Earth", "Star", "Water", "Fire"]
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_word_embedding(model, tokenizer, word):
    inputs = tokenizer(word, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # Last token hidden state
    hidden = outputs.hidden_states[-1][0, -1, :]
    return F.normalize(hidden.to(torch.float32), p=2, dim=0)

# ==========================================
# 2. フラクタル・ルーティング (Dynamic Paging)
# ==========================================
def fractal_routing(model, tokenizer, start_word, steps=4):
    print(f"\n[{start_word}] の思考軌跡を開始...")
    current_vec = get_word_embedding(model, tokenizer, start_word)
    
    # マクロ（大分類）アンカーの抽出
    macro_anchors = {valley: get_word_embedding(model, tokenizer, valley) for valley in VALLEYS.keys()}
    
    route = [start_word]
    current_valley = None
    
    for step in range(steps):
        # --- [Step A] マクロ判定 (今どの谷にいるか) ---
        best_valley = None
        best_sim = -1.0
        for valley, anchor_vec in macro_anchors.items():
            sim = F.cosine_similarity(current_vec.unsqueeze(0), anchor_vec.unsqueeze(0)).item()
            if sim > best_sim:
                best_sim = sim
                best_valley = valley
                
        # 谷が変わった場合（ダイナミック・ローディングの発動）
        if current_valley != best_valley:
            if current_valley is not None:
                print(f"  --> [Handoff] 境界を越えました。<{current_valley}> のデータを破棄し、<{best_valley}> の詳細データを読み込みます。")
            else:
                print(f"  --> [Load] <{best_valley}> の谷に降り立ちました。詳細データを展開します。")
            current_valley = best_valley
            
        # --- [Step B] ミクロ探索 (谷の中での精緻な思考) ---
        # この谷のデータ "だけ" を使って最も近い概念を探すため、他の谷のノイズに邪魔されない
        micro_concepts = VALLEYS[current_valley]
        micro_vecs = [get_word_embedding(model, tokenizer, w) for w in micro_concepts]
        
        best_micro = None
        best_micro_sim = -1.0
        best_micro_vec = None
        
        for w, v in zip(micro_concepts, micro_vecs):
            if w in route:
                continue # 通った道はスキップ
            sim = F.cosine_similarity(current_vec.unsqueeze(0), v.unsqueeze(0)).item()
            if sim > best_micro_sim:
                best_micro_sim = sim
                best_micro = w
                best_micro_vec = v
                
        if best_micro is None:
            print("  --> [Stop] この谷の思考の果てに到達しました。")
            break
            
        print(f"      Hop {step+1}: -> {best_micro} (Sim: {best_micro_sim:.4f})")
        route.append(best_micro)
        
        # ビー玉の移動（次の概念の方向へ引力が働く）
        current_vec = F.normalize(current_vec + best_micro_vec * 0.3, p=2, dim=0)
        
    return route

# ==========================================
# 3. テスト実行
# ==========================================
def main():
    print("Loading Base Model (0.5B)...")
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
    model.eval()
    
    # テスト単語
    test_words = [
        "Apple",      # -> Scienceの谷に落ちるか？
        "Tears",      # -> Emotionの谷に落ちるか？
        "Bank",       # -> Societyの谷に落ちるか？
        "Pollution",  # -> Societyから始まり、Natureへ移動(Handoff)するか？
    ]
    
    for w in test_words:
        fractal_routing(model, tokenizer, w)

if __name__ == "__main__":
    main()
