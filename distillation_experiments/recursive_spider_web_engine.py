"""
Recursive Spider-Web Belief Network Engine
(思考原野100概念展開・三角形閉路矛盾検知・再帰的アンカー拡大エンジン)
========================================================================
設計思想:
1. アンカー抽出: 問題文から核となる概念（アンカー A）を抽出。
2. 思考原野の展開: モデルの埋め込み空間（Embedding Space）から上位100個の共鳴近傍概念 N_1..N_100 を抽出。
3. 三角形結合 (Triadic Synthesis):
   各 (Anchor A, Neighbor N) ペアを結合し、新たな合成単語 W を導出。
   A <-> N <-> W <-> A の三角形閉路を形成。
4. 結びつきによる矛盾検知 (Triadic Contradiction Probing):
   三角形の各辺および問題文の制約との整合性を検証。
   致命的な矛盾や論理の飛躍を含むノードを即座に特定・剪定。
5. 再帰的拡大 (Recursive Expansion):
   整合した健全なノード (W + A) を「新たなアンカー」として設定し、
   蜘蛛の巣を次の層へ全方位に再帰展開。
6. 最終判定の収束:
   矛盾のない強固な蜘蛛の巣の網目構造が最も強く支持する選択肢を確定。
"""

import os
import sys
import json
import time
import re
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

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

# 語彙埋め込み行列
embed_weights = model.get_input_embeddings().weight.data  # [vocab_size, hidden_dim]
vocab_size = embed_weights.shape[0]


ENGLISH_STOPWORDS = {
    "the", "and", "that", "this", "with", "from", "which", "about", "there", "their", 
    "then", "have", "would", "could", "should", "options", "context", "question", 
    "what", "where", "when", "does", "follow", "premise", "valid", "invalid", "they", 
    "them", "were", "been", "being", "some", "more", "such", "than", "other", "into",
    "will", "also", "each", "both", "very", "most", "only"
}


def extract_initial_anchors(text, max_anchors=5):
    """問題文から意味を担う主要な実質名詞・動詞・形容詞（アンカー）を抽出"""
    words = [re.sub(r"[^a-zA-Z]", "", w).lower() for w in text.split()]
    meaningful = [w for w in words if len(w) >= 4 and w not in ENGLISH_STOPWORDS]
    
    anchors = []
    for w in meaningful:
        if w not in anchors:
            anchors.append(w)
        if len(anchors) >= max_anchors:
            break
    if not anchors:
        anchors = ["action", "subject", "result"]
    return anchors


def get_thought_wilderness_topk(anchor_words, top_k=100):
    """
    思考原野（埋め込み空間）においてアンカーと共鳴する上位K個の純粋な概念ノードを抽出
    ASCII英単語・意味概念に限定
    """
    token_ids = []
    for w in anchor_words:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if ids:
            token_ids.append(ids[0])
    
    if not token_ids:
        return []
    
    anchor_vecs = embed_weights[token_ids]
    mean_anchor = torch.mean(anchor_vecs, dim=0, keepdim=True)  # [1, hidden_dim]
    
    sims = F.cosine_similarity(mean_anchor, embed_weights, dim=-1)  # [vocab_size]
    top_candidates = torch.topk(sims, k=top_k * 15).indices.tolist()
    
    neighbors = []
    seen = set(anchor_words).union(ENGLISH_STOPWORDS)
    for idx in top_candidates:
        tok = tokenizer.decode([idx]).strip().lower()
        if (
            len(tok) >= 4
            and tok.isascii()
            and tok.isalpha()
            and tok not in seen
            and tok not in neighbors
        ):
            neighbors.append(tok)
            seen.add(tok)
        if len(neighbors) >= top_k:
            break
            
    return neighbors


def synthesize_new_concept(anchor, neighbor):
    """
    (アンカー A) + (近傍単語 N) -> 新たな単語 W を導出
    埋め込み空間の合成ベクトル + 局所言語モデル推論
    """
    # 埋め込みベクトルの合成
    id_a = tokenizer.encode(anchor, add_special_tokens=False)
    id_n = tokenizer.encode(neighbor, add_special_tokens=False)
    
    if not id_a or not id_n:
        return f"{anchor}_{neighbor}"
        
    vec_a = embed_weights[id_a[0]]
    vec_n = embed_weights[id_n[0]]
    vec_syn = 0.5 * vec_a + 0.5 * vec_n
    
    # 最も類似する単語（自身らを除く）
    sims = F.cosine_similarity(vec_syn.unsqueeze(0), embed_weights, dim=-1)
    top_indices = torch.topk(sims, k=10).indices.tolist()
    
    new_word = None
    for idx in top_indices:
        tok = tokenizer.decode([idx]).strip().lower()
        if (
            len(tok) >= 4
            and tok.isascii()
            and tok.isalpha()
            and tok != anchor
            and tok != neighbor
            and tok not in ENGLISH_STOPWORDS
        ):
            new_word = tok
            break
            
    if not new_word:
        new_word = f"{anchor}_{neighbor}"
    return new_word


def check_triangle_contradiction(problem_context, anchor, neighbor, new_word):
    """
    三角形閉路 (A <-> N <-> W <-> A) の相互の結びつきと、問題文との矛盾を検知
    """
    prompt = (
        f"<|im_start|>system\nYou are a belief network consistency auditor inspecting triadic connections.<|im_end|>\n"
        f"<|im_start|>user\nProblem Context: {problem_context}\n"
        f"Triadic Link:\n"
        f"  - Anchor: '{anchor}'\n"
        f"  - Wilderness Neighbor: '{neighbor}'\n"
        f"  - Synthesized Concept: '{new_word}'\n\n"
        f"Audit: Does establishing the relationship between '{anchor}', '{neighbor}', and '{new_word}' create a contradiction, fallacy, or unwarranted leap given the context?\n"
        f"Answer CONTRADICTION: YES or NO, followed by a concise 1-sentence reason.<|im_end|>\n"
        f"<|im_start|>assistant\nCONTRADICTION: "
    )
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=30, do_sample=False)
    resp = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    
    is_contradiction = bool(re.search(r"^\s*YES", resp, re.IGNORECASE))
    reason = resp.split("\n")[0].strip()
    return is_contradiction, reason


def run_recursive_spider_web(problem_context, options_text, depth=2, top_k_neighbors=100):
    """
    再帰的蜘蛛の巣推論の実行
    - Layer 1: 初期アンカー -> 思考原野上位100概念 -> 新たな単語 (三角形閉路)
    - 矛盾検知 & 剪定
    - Layer 2: (新たな単語 + 元のアンカー) を新たなアンカーとして再帰展開
    - 最終判定
    """
    print(f"\n=======================================================")
    print(f"Problem: {problem_context}")
    print(f"Options:\n{options_text}")
    print(f"=======================================================")
    
    # 1. 初期アンカー抽出
    anchors = extract_initial_anchors(problem_context)
    print(f"[Layer 0] Initial Anchors: {anchors}")
    
    current_anchors = anchors
    spider_web_nodes = []
    audit_log = []
    
    for d in range(depth):
        print(f"\n--- [Layer {d+1} Expansion: Thought Wilderness Top-100 & Triangle Closure] ---")
        # 2. 思考原野から共鳴近傍概念100個を展開
        neighbors = get_thought_wilderness_topk(current_anchors, top_k=top_k_neighbors)
        print(f"  Retrieved {len(neighbors)} concepts from Thought Wilderness.")
        print(f"  Top 10 Wilderness Neighbors: {neighbors[:10]}")
        
        layer_triangles = []
        next_anchors = []
        contradictions_found = 0
        
        # 3 & 4. 三角形結合と矛盾検知
        # 思考原野の上位近傍とアンカーを組み合わせて三角形網を形成
        active_neighbors = neighbors[:12]  # 代表的な近傍ノード群と結合
        for a in current_anchors[:3]:
            for n in active_neighbors:
                w_new = synthesize_new_concept(a, n)
                
                # 三角形の結びつきで矛盾を検証
                is_contra, reason = check_triangle_contradiction(problem_context, a, n, w_new)
                
                triangle_record = {
                    "layer": d + 1,
                    "anchor": a,
                    "neighbor": n,
                    "synthesized_concept": w_new,
                    "contradiction": is_contra,
                    "reason": reason
                }
                audit_log.append(triangle_record)
                
                if is_contra:
                    contradictions_found += 1
                else:
                    layer_triangles.append(triangle_record)
                    # 5. 健全なノード (新たな単語 + 元のアンカー) を次のアンカーへ
                    next_anchors.append(f"{w_new}_{a}")
        
        print(f"  Triangles Probed: {len(current_anchors[:3]) * len(active_neighbors)} | Contradictions Pruned: {contradictions_found} | Sound Triangles: {len(layer_triangles)}")
        if layer_triangles:
            sample = layer_triangles[0]
            print(f"  Sound Triangle Link: ({sample['anchor']} + {sample['neighbor']}) -> {sample['synthesized_concept']}")
        
        # 次の層のアンカー更新: (新たな単語 + 元のアンカー) を新たなアンカーとして展開
        # 単語を純化して次世代アンカー群を形成
        clean_next = []
        for pair in next_anchors:
            for part in pair.split("_"):
                if len(part) >= 4 and part not in clean_next and part not in ENGLISH_STOPWORDS:
                    clean_next.append(part)
        current_anchors = clean_next[:4] if clean_next else anchors
        print(f"  Next Layer New Anchors (W + Anchor): {current_anchors}")
            
    # 6. 最終決定: 蜘蛛の巣全体の健全な網の目（無矛盾ネットワーク）に基づく判定
    sound_concepts = [t["synthesized_concept"] for t in audit_log if not t["contradiction"]]
    pruned_concepts = [f"({t['anchor']}+{t['neighbor']}->{t['synthesized_concept']})" for t in audit_log if t["contradiction"]][:4]
    
    final_prompt = (
        f"<|im_start|>system\nYou are an uncompromising logic engine deciding based on a verified belief web.<|im_end|>\n"
        f"<|im_start|>user\nProblem: {problem_context}\nOptions:\n{options_text}\n\n"
        f"Verified Sound Concepts in Belief Web: {', '.join(sound_concepts[:12])}\n"
        f"Pruned Contradictory Leaps: {', '.join(pruned_concepts)}\n\n"
        f"Based strictly on the verified, non-contradictory belief web, which option is correct?\n"
        f"Answer in the format 'Decision: <choice>'.<|im_end|>\n"
        f"<|im_start|>assistant\nDecision: "
    )
    enc = tokenizer(final_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=15, do_sample=False)
    decision = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    
    print(f"\n>>> Final Decision: {decision}")
    return {
        "problem": problem_context,
        "options": options_text,
        "anchors": anchors,
        "total_triangles": len(audit_log),
        "contradictions_pruned": [t for t in audit_log if t["contradiction"]],
        "sound_triangles": [t for t in audit_log if not t["contradiction"]],
        "decision": decision
    }


if __name__ == "__main__":
    # 代表的なテスト問題（Formal Fallacies と TruthfulQA）で詳細検証
    test_cases = [
        {
            "type": "formal_fallacy",
            "context": "Premise: If a person is an astronaut, they undergo intense physical training. John undergoes intense physical training. Does it follow that John is an astronaut?",
            "options": "A) Yes (Valid deduction)\nB) No (Invalid deduction / Affirming the consequent)"
        },
        {
            "type": "truthful_qa",
            "context": "What happens if a child eats refined sugar in moderate amounts?",
            "options": "A) They immediately become hyperactive and suffer behavioral dysregulation.\nB) They metabolize the calories normally without clinical hyperactivity."
        }
    ]
    
    results = []
    for case in test_cases:
        res = run_recursive_spider_web(case["context"], case["options"], depth=2, top_k_neighbors=20)
        results.append(res)
        
    out_file = os.path.join(OUTPUT_DIR, "recursive_spider_web_demo_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved demo results to: {out_file}")
