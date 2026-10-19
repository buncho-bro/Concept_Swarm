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

print(f"[Ablation Benchmark] Initializing model on {device}...")
tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    STUDENT_MODEL_ID,
    dtype=torch.float16,
    device_map={"": device},
    local_files_only=True
).eval()

embed_tokens = model.get_input_embeddings()
W_embed = embed_tokens.weight.data.float()  # (vocab_size, 2048)

# ========================================================
# 1. 2つの語彙空間の準備
# ========================================================
# (A) 従来空間 (Standard Vocab, ~74,563語): サブワード断片を含む
std_word_to_id = {}
for tid in range(len(tokenizer)):
    t_str = tokenizer.decode([tid]).strip()
    if len(t_str) >= 2 and t_str.isalpha():
        clean_word = t_str.lower()
        if clean_word not in std_word_to_id:
            std_word_to_id[clean_word] = tid

std_words = list(std_word_to_id.keys())
std_ids = torch.tensor([std_word_to_id[w] for w in std_words], device=device, dtype=torch.long)
std_space = F.normalize(W_embed[std_ids], p=2, dim=-1)
print(f"[Word Space A (Standard)] {len(std_words)} words (contains subword fragments).")

# (B) 改良1空間 (Purified Independent Vocab): 自立語のみ、サブワード語尾片を除外
# 英語は先頭スペース(またはĠ)がある自立語トークン、CJKは2文字以上の自立熟語
purified_word_to_id = {}
for tid in range(len(tokenizer)):
    raw_s = tokenizer.convert_ids_to_tokens(tid)
    decoded = tokenizer.decode([tid])
    s_strip = decoded.strip()
    if len(s_strip) < 2 or not s_strip.isalpha():
        continue
    
    # 英語・ラテン文字の場合: 先頭にスペースがあるか確認（自立語）
    is_latin = s_strip.isascii()
    if is_latin:
        # raw tokenがĠまたは先頭空白で始まっているか
        has_space_prefix = (raw_s.startswith('Ġ') or decoded.startswith(' ') or decoded.startswith('Ġ'))
        if not has_space_prefix:
            # 先頭スペースなしの小文字トークンはサブワード断片なので除外
            continue
    
    clean_word = s_strip.lower()
    if clean_word not in purified_word_to_id:
        purified_word_to_id[clean_word] = tid

purified_words = list(purified_word_to_id.keys())
purified_ids = torch.tensor([purified_word_to_id[w] for w in purified_words], device=device, dtype=torch.long)
purified_space = F.normalize(W_embed[purified_ids], p=2, dim=-1)
print(f"[Word Space B (Purified Independent)] {len(purified_words)} words (subwords eliminated).")

# ========================================================
# 2. 100ケースの因果ベンチマークペア
# ========================================================
from massive_causal_benchmark import BENCHMARK_PAIRS

def observe_token_momentum(token_str: str):
    """単一トークンの固有揺らぎ軌跡と運動量ベクトル"""
    inputs = tokenizer(token_str, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
        h_states = out.hidden_states
        h_start = F.normalize(h_states[0][0, -1, :].float(), p=2, dim=-1)
        h_mid = F.normalize(h_states[18][0, -1, :].float(), p=2, dim=-1)
        h_deep = F.normalize(h_states[32][0, -1, :].float(), p=2, dim=-1)
        momentum = F.normalize((h_mid - h_start) + (h_deep - h_mid), p=2, dim=-1)
    return h_start, momentum

def compute_causal_trace(pos_A, mom_A, pos_B, mom_B, step_scale=0.7):
    """言葉の差異線と直交因果力線の合成トレース点"""
    direct_axis = F.normalize(pos_B - pos_A, p=2, dim=-1)
    center_pos = F.normalize(0.5 * (pos_A + pos_B), p=2, dim=-1)
    interaction_diff = mom_B - mom_A
    proj = torch.dot(interaction_diff, direct_axis) * direct_axis
    ortho_force = F.normalize(interaction_diff - proj, p=2, dim=-1)
    traced_point = F.normalize(center_pos + step_scale * ortho_force, p=2, dim=-1)
    return traced_point

def query_space(traced_point, pos_A, pos_B, words_list, space_tensor, use_repulsion=False, alpha=0.4, top_k=5):
    """
    指定された空間から単語を読み出し。
    use_repulsion=True の場合は元単語A, Bへの引力を相殺（反発場を適用）。
    """
    sim_trace = torch.matmul(traced_point, space_tensor.T)  # (N,)
    
    if use_repulsion:
        sim_A = torch.matmul(pos_A, space_tensor.T)
        sim_B = torch.matmul(pos_B, space_tensor.T)
        repulsion = torch.max(sim_A, sim_B)  # AまたはBに近いほどペナルティ
        score = sim_trace - alpha * repulsion
    else:
        score = sim_trace
        
    top_indices = torch.topk(score, k=top_k + 5).indices.tolist()
    results = [words_list[idx] for idx in top_indices]
    return results[:top_k]

# ========================================================
# 3. 4つの比較条件（Ablation Study）の実行
# ========================================================
MODES = {
    "Baseline": {
        "desc": "従来空間 (74,563語) + 反発なし",
        "words": std_words,
        "space": std_space,
        "use_repulsion": False
    },
    "Mode1_PurifiedOnly": {
        "desc": "改良1のみ: 自立語空間 (サブワード断片排除) + 反発なし",
        "words": purified_words,
        "space": purified_space,
        "use_repulsion": False
    },
    "Mode2_RepulsionOnly": {
        "desc": "改良2のみ: 従来空間 (74,563語) + 元単語反発ポテンシャル(α=0.4)",
        "words": std_words,
        "space": std_space,
        "use_repulsion": True
    },
    "Mode12_Combined": {
        "desc": "改良1+2統合: 自立語空間 + 元単語反発ポテンシャル(α=0.4)",
        "words": purified_words,
        "space": purified_space,
        "use_repulsion": True
    }
}

print(f"\n========================================================")
print(f"Starting Ablation Benchmark on 100 pairs across 4 configurations...")
print(f"========================================================")

ablation_results = {mode_name: {"hit_count": 0, "escape_count": 0, "details": []} for mode_name in MODES}

# 100ペアの単語表現を一度キャッシュ
cached_token_data = {}
for item in BENCHMARK_PAIRS:
    for t_str in item["pair"]:
        if t_str not in cached_token_data:
            cached_token_data[t_str] = observe_token_momentum(t_str)

for idx, item in enumerate(BENCHMARK_PAIRS):
    t_A, t_B = item["pair"]
    expected_list = item["expected"]
    pos_A, mom_A = cached_token_data[t_A]
    pos_B, mom_B = cached_token_data[t_B]
    
    traced_point = compute_causal_trace(pos_A, mom_A, pos_B, mom_B, step_scale=0.7)
    
    for mode_name, cfg in MODES.items():
        top_words = query_space(
            traced_point, pos_A, pos_B,
            cfg["words"], cfg["space"],
            use_repulsion=cfg["use_repulsion"],
            alpha=0.4,
            top_k=5
        )
        
        # 評価: 元単語自身およびその直接形から脱出できたか
        is_escaped = not any(w.lower() in [t_A.lower(), t_B.lower()] for w in top_words[:3])
        if is_escaped:
            ablation_results[mode_name]["escape_count"] += 1
            
        # 評価: 期待される因果帰結語が含まれたか (部分一致/完全一致)
        matched_hits = []
        for w in top_words:
            for exp in expected_list:
                if exp.lower() in w.lower() or w.lower() in exp.lower():
                    matched_hits.append(w)
                    break
        is_hit = len(matched_hits) > 0
        if is_hit:
            ablation_results[mode_name]["hit_count"] += 1
            
        ablation_results[mode_name]["details"].append({
            "pair": f"{t_A}+{t_B}",
            "category": item["category"],
            "expected": expected_list,
            "top5": top_words,
            "is_escaped": is_escaped,
            "is_hit": is_hit,
            "hits": matched_hits
        })

    if (idx + 1) % 20 == 0:
        print(f"Processed {idx + 1}/100 pairs...")

print(f"\n================== ABLATION BENCHMARK SUMMARY (N=100) ==================")
for mode_name, cfg in MODES.items():
    hits = ablation_results[mode_name]["hit_count"]
    escapes = ablation_results[mode_name]["escape_count"]
    hit_rate = (hits / 100.0) * 100.0
    escape_rate = (escapes / 100.0) * 100.0
    print(f"[{mode_name}] ({cfg['desc']})")
    print(f"  ➔ Hit Rate: {hits}/100 ({hit_rate:.1f}%) | Escape Rate: {escapes}/100 ({escape_rate:.1f}%)\n")

# 代表的な問題ペアでの各モード比較
print("================== REPRESENTATIVE CASE COMPARISON ==================")
pick_pairs = ["fire+wood", "sun+ice", "earthquake+ocean", "light+prism", "cow+grass", "crime+police"]
for target_pair in pick_pairs:
    print(f"\nTarget: [{target_pair}]")
    for mode_name in MODES:
        det = next(d for d in ablation_results[mode_name]["details"] if d["pair"] == target_pair)
        hit_mark = "✅" if det["is_hit"] else "❌"
        print(f"  {hit_mark} {mode_name:22s}: {det['top5']} (Hits: {det['hits']})")

output_file = "ablation_benchmark_results.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(ablation_results, f, ensure_ascii=False, indent=2)
print(f"\nSaved full ablation comparison to: {output_file}")
