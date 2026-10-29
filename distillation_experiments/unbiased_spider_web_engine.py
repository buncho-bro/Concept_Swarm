"""
Unbiased Recursive Spider-Web Belief Network Engine
(思考原野全単語からの完全無作為抽出・近傍結合・三角形閉路矛盾検知・再帰拡大エンジン)
======================================================================================
ユーザー設計思想の完全反映:
1. 【完全無作為抽出 (Unbiased Random Sampling)】:
   偏りを一切排除するため、モデルの思考原野（語彙空間全66,841単語）から
   完全にランダム（一様分布）にトークン・概念を抽出する。
2. 【思考原野での近傍結合 (Wilderness Neighbor Resonance)】:
   無作為に選ばれたトークンと、思考原野（埋め込み空間）で近しい単語（上位概念）をペアリングする。
3. 【三角形閉路の生成 (Triadic Closure)】:
   (無作為トークン) + (近傍単語) -> 新たな単語 を導出し、
   無作為トークン <-> 近傍単語 <-> 新たな単語 の3者をつなぐ。
4. 【相互の結びつきによる矛盾の検出 (Mutual Link Contradiction Probing)】:
   網の目状に結び付けられたノード群から、問題文の制約や他の経路と衝突する致命的矛盾を検知して剪定。
5. 【再帰的拡大 (Recursive Web Expansion)】:
   （新たな単語 + 元の無作為単語）を「新たなアンカー」として次の層へ蜘蛛の巣を広げていく。
"""

import os
import sys
import json
import time
import re
import random
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append("C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment")
sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
ARTIFACT_DIR = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5"
device = "cuda" if torch.cuda.is_available() else "cpu"
QWEN_ID = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"Device: {device}")
print("Loading Model & Tokenizer:", QWEN_ID)
tokenizer = AutoTokenizer.from_pretrained(QWEN_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    QWEN_ID,
    torch_dtype=torch.bfloat16,
    device_map=device,
    trust_remote_code=True
)
model.eval()

embed_weights = model.get_input_embeddings().weight.data  # [vocab_size, hidden_dim]
vocab_size = embed_weights.shape[0]

# 思考原野にある全単語（語彙空間）のインデックス構築
print("Indexing complete vocabulary for Thought Wilderness...")
FULL_WILDERNESS_VOCAB = []
VOCAB_ID_MAP = {}
for idx in range(vocab_size):
    s = tokenizer.decode([idx]).strip().lower()
    if len(s) >= 3 and s.isascii() and s.isalpha():
        if s not in VOCAB_ID_MAP:
            FULL_WILDERNESS_VOCAB.append(s)
            VOCAB_ID_MAP[s] = idx

print(f"Total unbiased Thought Wilderness concepts: {len(FULL_WILDERNESS_VOCAB)} words.")


def sample_random_tokens_from_wilderness(num_samples=10, seed=None):
    """思考原野にある全単語から完全に偏りなくランダム（無作為）に抽出"""
    if seed is not None:
        random.seed(seed)
    return random.sample(FULL_WILDERNESS_VOCAB, num_samples)


def get_nearest_neighbors_in_wilderness(concept_word, top_k=50):
    """思考原野（埋め込み空間）において指定単語と近しい上位単語を探索"""
    tok_id = VOCAB_ID_MAP.get(concept_word)
    if tok_id is None:
        ids = tokenizer.encode(concept_word, add_special_tokens=False)
        if not ids:
            return []
        tok_id = ids[0]
        
    vec = embed_weights[tok_id].unsqueeze(0)
    sims = F.cosine_similarity(vec, embed_weights, dim=-1)
    
    # 上位候補から自身を除外して取得
    top_indices = torch.topk(sims, k=top_k * 4).indices.tolist()
    neighbors = []
    for idx in top_indices:
        tok = tokenizer.decode([idx]).strip().lower()
        if (
            len(tok) >= 3
            and tok.isascii()
            and tok.isalpha()
            and tok != concept_word
            and tok not in neighbors
        ):
            neighbors.append(tok)
        if len(neighbors) >= top_k:
            break
    return neighbors


def synthesize_new_word(token_a, token_b):
    """
    (無作為トークン / アンカー) + (近傍単語) -> 新たな単語を導出
    埋め込み空間のベクトル合成 + 語彙照合
    """
    id_a = VOCAB_ID_MAP.get(token_a)
    id_b = VOCAB_ID_MAP.get(token_b)
    if id_a is None or id_b is None:
        return f"{token_a}_{token_b}"
        
    vec_a = embed_weights[id_a]
    vec_b = embed_weights[id_b]
    vec_syn = 0.5 * vec_a + 0.5 * vec_b
    
    sims = F.cosine_similarity(vec_syn.unsqueeze(0), embed_weights, dim=-1)
    top_indices = torch.topk(sims, k=15).indices.tolist()
    
    for idx in top_indices:
        tok = tokenizer.decode([idx]).strip().lower()
        if (
            len(tok) >= 3
            and tok.isascii()
            and tok.isalpha()
            and tok != token_a
            and tok != token_b
        ):
            return tok
            
    return f"{token_a}_{token_b}"


def audit_triadic_link(problem_context, node_a, node_b, new_word):
    """
    3者 (node_a <-> node_b <-> new_word) の相互の結びつきを生かし、
    問題文に対する致命的な矛盾、偏見、推論の飛躍を検知
    """
    prompt = (
        f"<|im_start|>system\nYou are an uncompromising logic auditor examining a belief network graph.<|im_end|>\n"
        f"<|im_start|>user\nContext: {problem_context}\n"
        f"Triadic Link in Thought Wilderness:\n"
        f"  - Random Node: '{node_a}'\n"
        f"  - Wilderness Neighbor: '{node_b}'\n"
        f"  - Synthesized Concept: '{new_word}'\n\n"
        f"Audit Task: When connected to the Context, does this triadic link introduce a fatal contradiction, "
        f"unwarranted assumption, or false fallacy?\n"
        f"Reply with 'CONTRADICTION: YES' or 'CONTRADICTION: NO' followed by 1 sentence explanation.<|im_end|>\n"
        f"<|im_start|>assistant\nCONTRADICTION: "
    )
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=30, do_sample=False)
    resp = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    
    is_contradiction = bool(re.search(r"^\s*YES", resp, re.IGNORECASE))
    reason = resp.split("\n")[0].strip()
    return is_contradiction, reason


def run_unbiased_spider_web_engine(problem_context, options_text, num_random_tokens=10, depth=2):
    """
    無作為抽出からの蜘蛛の巣推論エンジン
    """
    print(f"\n=======================================================")
    print(f"Problem: {problem_context}")
    print(f"Options:\n{options_text}")
    print(f"=======================================================")
    
    # 1. 思考原野の全単語から完全に無作為にトークンを抽出
    random_tokens = sample_random_tokens_from_wilderness(num_samples=num_random_tokens)
    print(f"\n[Layer 0: Unbiased Random Sampling from Thought Wilderness]")
    print(f"  Randomly extracted tokens ({len(random_tokens)} words): {random_tokens}")
    
    current_nodes = random_tokens
    all_triangles = []
    
    for d in range(depth):
        print(f"\n--- [Layer {d+1}: Expanding Wilderness Neighbors & Forming Triangles] ---")
        layer_sound_triangles = []
        next_generation_anchors = []
        contradictions_count = 0
        
        for r_tok in current_nodes:
            # 2. その無作為トークンと思考原野で近しい単語（上位概念）を探索
            neighbors = get_nearest_neighbors_in_wilderness(r_tok, top_k=20)
            
            # 代表的な近傍と結合
            for n_tok in neighbors[:5]:
                # 3. (無作為トークン) + (近傍単語) -> 新たな単語
                new_w = synthesize_new_word(r_tok, n_tok)
                
                # 4. 相互の結びつきを生かして矛盾を検知
                is_contra, reason = audit_triadic_link(problem_context, r_tok, n_tok, new_w)
                
                record = {
                    "layer": d + 1,
                    "random_token": r_tok,
                    "neighbor": n_tok,
                    "synthesized_concept": new_w,
                    "contradiction": is_contra,
                    "reason": reason
                }
                all_triangles.append(record)
                
                if is_contra:
                    contradictions_count += 1
                else:
                    layer_sound_triangles.append(record)
                    # 5. (新たな単語 + 元の無作為単語) を新たなアンカーとして設定
                    next_generation_anchors.append(new_w)
                    next_generation_anchors.append(r_tok)
        
        print(f"  Triangles formed: {len(current_nodes) * 5} | Contradictions Pruned: {contradictions_count} | Sound Triangles: {len(layer_sound_triangles)}")
        if layer_sound_triangles:
            s = layer_sound_triangles[0]
            print(f"  Sound Triadic Connection Sample: ({s['random_token']} + {s['neighbor']}) -> {s['synthesized_concept']}")
            
        # 次の層のアンカー更新: (新たな単語 + 元の単語) を新たなアンカーとして広げる
        unique_next = list(dict.fromkeys(next_generation_anchors))
        current_nodes = unique_next[:num_random_tokens]
        print(f"  Next Generation Anchors (New Word + Original Token): {current_nodes[:8]}")

    # 6. 蜘蛛の巣全体の健全な結合に基づく最終判定
    sound_concepts = [t["synthesized_concept"] for t in all_triangles if not t["contradiction"]]
    pruned_examples = [f"({t['random_token']}+{t['neighbor']}->{t['synthesized_concept']})" for t in all_triangles if t["contradiction"]][:5]
    
    final_prompt = (
        f"<|im_start|>system\nYou are a master judge evaluating options based on an expanded belief spider-web.<|im_end|>\n"
        f"<|im_start|>user\nProblem: {problem_context}\nOptions:\n{options_text}\n\n"
        f"Verified Sound Triadic Links in Web: {', '.join(sound_concepts[:15])}\n"
        f"Pruned Contradictions: {', '.join(pruned_examples)}\n\n"
        f"Using this unbiased verified network, state the final correct choice as 'Decision: <choice>'.<|im_end|>\n"
        f"<|im_start|>assistant\nDecision: "
    )
    enc = tokenizer(final_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=15, do_sample=False)
    decision = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    
    print(f"\n>>> Unbiased Spider-Web Decision: {decision}")
    return {
        "problem": problem_context,
        "options": options_text,
        "initial_random_tokens": random_tokens,
        "total_triangles": len(all_triangles),
        "contradictions_pruned_count": len([t for t in all_triangles if t["contradiction"]]),
        "sound_triangles_count": len([t for t in all_triangles if not t["contradiction"]]),
        "decision": decision
    }


if __name__ == "__main__":
    test_cases = [
        {
            "name": "Formal Fallacies (BBH)",
            "context": "Premise: If a person is an astronaut, they undergo intense physical training. John undergoes intense physical training. Does it follow that John is an astronaut?",
            "options": "A) Yes (Valid deduction)\nB) No (Invalid deduction / Affirming the consequent)"
        },
        {
            "name": "TruthfulQA (Sugar & Hyperactivity)",
            "context": "What happens if a child eats refined sugar in moderate amounts?",
            "options": "A) They immediately become hyperactive and suffer behavioral dysregulation.\nB) They metabolize the calories normally without clinical hyperactivity."
        }
    ]
    
    results = []
    for tc in test_cases:
        res = run_unbiased_spider_web_engine(tc["context"], tc["options"], num_random_tokens=8, depth=2)
        results.append(res)
        
    out_path = os.path.join(OUTPUT_DIR, "unbiased_spider_web_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nFinished. Results saved to: {out_path}")
