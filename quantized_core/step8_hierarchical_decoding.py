"""
【ファイル概要】
step8_hierarchical_decoding.py

フラクタルルーティングに基づき、階層的にテキストをデコード（生成）していく実験的アルゴリズム。
"""

import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss
import numpy as np

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print("Loading Model (FP16 for speed comparison)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # 純粋な計算速度を測るため、量子化なしのFP16でロードします
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    
    # 1. LM Head (全15万語の辞書ベクトル) の抽出
    lm_head_weight = model.lm_head.weight.detach() # Shape: [Vocab, Hidden]
    V, D = lm_head_weight.shape
    print(f"\nExtracted Output Layer -> Vocab Size: {V}, Hidden Size: {D}")
    
    # 2. FAISS インデックスの構築（概念地形の地図作成）
    print("Building FAISS Concept Index (Mapping 150,000 words into vector space)...")
    start_faiss = time.time()
    lm_head_np = lm_head_weight.cpu().to(torch.float32).numpy()
    faiss.normalize_L2(lm_head_np) # コサイン類似度（角度）で検索するための正規化
    index = faiss.IndexFlatIP(D)   # 内積（IP）検索インデックス
    index.add(lm_head_np)
    print(f"FAISS Index Ready. (Took {time.time()-start_faiss:.2f}s)\n")
    
    # ==========================================
    # カスタム推論ループ
    # ==========================================
    def custom_generate(prompt, max_new_tokens=30, method="standard", top_k=100):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        past_key_values = None
        generated_tokens = []
        
        # 最初のプロンプトを処理してKVキャッシュを初期化
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=True, output_hidden_states=True)
        past_key_values = outputs.past_key_values
        hidden = outputs.hidden_states[-1][0, -1, :]
        
        start_time = time.time()
        
        for i in range(max_new_tokens):
            if method == "standard":
                # --- Method A: 従来方式 (全15万語との計算) ---
                # 15万語の重みをすべて計算器に運んで内積をとる
                logits = torch.matmul(hidden.unsqueeze(0), lm_head_weight.T)
                next_token = torch.argmax(logits, dim=-1).item()
                
            elif method == "topographical":
                # --- Method B: フラクタル探査＋文法評価 (あなたの方式) ---
                # 1. FAISSによる高速探査 (意味が近い候補 K 個だけを抽出)
                hidden_np = hidden.cpu().to(torch.float32).numpy().reshape(1, -1)
                faiss.normalize_L2(hidden_np)
                _, I = index.search(hidden_np, top_k)
                candidates = I[0] # Top-K トークンIDの配列
                
                # 2. 局所的な文章整合性の評価
                # 拾ってきた候補(例:100語)の重みだけを取り出して計算する
                local_weights = lm_head_weight[candidates] # [K, D]
                local_logits = torch.matmul(hidden.unsqueeze(0), local_weights.T) # [1, K]
                
                # K個の中で最も文法整合性の高い（Logits最大の）ものを選ぶ
                best_local_idx = torch.argmax(local_logits, dim=-1).item()
                next_token = candidates[best_local_idx]
            
            generated_tokens.append(next_token)
            
            # 次のステップのための準備（自己回帰）
            input_ids = torch.tensor([[next_token]]).to(device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
            past_key_values = outputs.past_key_values
            hidden = outputs.hidden_states[-1][0, -1, :]
            
        elapsed = time.time() - start_time
        tps = max_new_tokens / elapsed
        result_text = prompt + tokenizer.decode(generated_tokens)
        return result_text, tps

    # ==========================================
    # 比較テストの実行
    # ==========================================
    test_prompt = "日本の首都はどこですか？また、そこには何が"
    
    print("=== Method A: Standard Decoding (全15万語の毎ターン計算) ===")
    res_a, tps_a = custom_generate(test_prompt, method="standard")
    print(f"Result: {res_a}")
    print(f"Speed:  {tps_a:.2f} tokens/sec\n")
    
    print("=== Method B: Topographical Decoding (高速探査 K=100 ＋ 文法評価) ===")
    res_b, tps_b = custom_generate(test_prompt, method="topographical", top_k=100)
    print(f"Result: {res_b}")
    print(f"Speed:  {tps_b:.2f} tokens/sec\n")

if __name__ == "__main__":
    main()
