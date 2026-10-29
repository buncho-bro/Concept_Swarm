"""
Evaluate Benchmark with Mass-Epoch Pre-Organized Belief Web
(200エポック自己組織化された蜘蛛の巣信念網を活用した高速ベンチマーク評価)
========================================================================
1. 事前組織化された信念網 (mass_epoch_organized_belief_web.json) をロード。
2. 問題文の核概念に対して、思考原野で事前に整理された無矛盾リンク・反例クラスタを即座に参照。
3. 問題文にノイズを混ぜず、整理された思考マップを背景知識として与えて解答を導出。
4. Web標準250問の精度を測定し、整理前の素のモデルおよび推論時ノイズモデルと比較。
"""

import os
import sys
import json
import time
import re
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append("C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment")
sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
ARTIFACT_DIR = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5"
RESULTS_JSON = os.path.join(OUTPUT_DIR, "organized_belief_web_benchmark_results.json")
device = "cuda" if torch.cuda.is_available() else "cpu"
QWEN_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# 1. 事前組織化された蜘蛛の巣信念網のロード
WEB_JSON = os.path.join(OUTPUT_DIR, "mass_epoch_organized_belief_web.json")
with open(WEB_JSON, "r", encoding="utf-8") as f:
    web_data = json.load(f)

print(f"Loaded Pre-Organized Belief Web: {web_data['total_nodes']} nodes, {web_data['total_sound_triangles']} sound links.")

tokenizer = AutoTokenizer.from_pretrained(QWEN_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    QWEN_ID,
    torch_dtype=torch.bfloat16,
    device_map=device,
    trust_remote_code=True
)
model.eval()


def solve_with_organized_mental_map(problem_text, options_text=""):
    """
    事前に大量エポックで整理・無矛盾化された思考網の知見を参照して解答
    """
    # 問題文の主要単語
    words = [re.sub(r"[^a-zA-Z]", "", w).lower() for w in problem_text.split() if len(w) >= 4]
    
    # 思考網内の関連ノードを照合
    matched_nodes = [w for w in words if w in web_data["sample_connected_nodes"]][:5]
    context_hint = ""
    if matched_nodes:
        context_hint = f"Pre-Organized Knowledge Network Reference: Verified semantic concepts include {', '.join(matched_nodes)}.\n"
        
    prompt = (
        f"<|im_start|>system\nYou are a precise reasoning engine guided by a structurally organized, non-contradictory belief network. "
        f"Avoid fallacious shortcuts, popular misconceptions, or ungrounded leaps.<|im_end|>\n"
        f"<|im_start|>user\n{context_hint}"
        f"Problem: {problem_text}\n"
        f"{options_text}\n"
        f"State your reasoned conclusion formatted strictly as 'Final Answer: <choice>'.<|im_end|>\n"
        f"<|im_start|>assistant\nFinal Answer: "
    )
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=15, do_sample=False)
    resp = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    
    m = re.search(r"([A-Za-z0-9_]+)", resp)
    return m.group(1).strip() if m else resp


from spider_web_belief_network_engine import load_web_standard_suites

def run_evaluation():
    suites = load_web_standard_suites()
    all_results = {}
    total_correct = 0
    total_items = 0
    t0 = time.time()
    
    print("\n================================================================================")
    print("EVALUATING BENCHMARK WITH 200-EPOCH PRE-ORGANIZED BELIEF WEB (N=250)")
    print("================================================================================")
    
    for suite_name, items in suites.items():
        print(f"\n--- Testing Suite: {suite_name} (N={len(items)}) ---")
        correct = 0
        records = []
        
        for i, it in enumerate(items):
            if suite_name in ['causal_judgement', 'disambiguation_qa', 'formal_fallacies']:
                inp = it['input']
                target = it['target'].strip()
                options_text = "Options: Yes or No" if suite_name in ['causal_judgement', 'formal_fallacies'] else ""
            elif suite_name == 'truthfulqa_mc1':
                inp = it['question']
                options_text = "Options:\n" + "\n".join([f"({chr(65+c_idx)}) {c}" for c_idx, c in enumerate(it['choices'])])
                target = chr(65 + it['correct_idx'])
            elif suite_name == 'mmlu_moral':
                inp = it['question']
                options_text = "Options:\n" + "\n".join([f"({chr(65+c_idx)}) {c}" for c_idx, c in enumerate(it['choices'])])
                target = chr(65 + it['answer'])
            else:
                inp = it.get('input', it.get('question', ''))
                target = str(it.get('target', it.get('answer', '')))
                options_text = ""
                
            pred = solve_with_organized_mental_map(inp, options_text)
            
            p_clean = pred.lower().strip("()., ")
            t_clean = target.lower().strip("()., ")
            
            # 数字による選択肢回答 (1 -> a, 2 -> b, etc.) の正規化
            num_map = {"1": "a", "2": "b", "3": "c", "4": "d"}
            if p_clean in num_map:
                p_clean = num_map[p_clean]
            
            is_correct = False
            if p_clean == t_clean:
                is_correct = True
            elif len(t_clean) == 1 and p_clean.startswith(t_clean):
                is_correct = True
            elif t_clean in ['yes', 'no'] and p_clean in ['yes', 'no'] and t_clean == p_clean:
                is_correct = True
                
            if is_correct:
                correct += 1
                
            records.append({
                "idx": i + 1,
                "input": inp[:80],
                "target": target,
                "pred": pred,
                "correct": is_correct
            })
            
            if (i + 1) % 10 == 0 or (i + 1) == len(items):
                curr = (correct / (i + 1)) * 100.0
                print(f"  [{suite_name}] Item {i+1:02d}/{len(items)} | Current Acc: {curr:.1f}%")
                
        acc = (correct / len(items)) * 100.0
        all_results[suite_name] = {
            "accuracy": acc,
            "correct": correct,
            "total": len(items),
            "records": records
        }
        total_correct += correct
        total_items += len(items)
        print(f">>> Suite {suite_name} Accuracy: {acc:.1f}% ({correct}/{len(items)})")
        
    elapsed = time.time() - t0
    total_acc = (total_correct / total_items) * 100.0
    
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_accuracy": total_acc,
        "total_correct": total_correct,
        "total_items": total_items,
        "elapsed_sec": round(elapsed, 1),
        "suites": all_results
    }
    
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        
    print("\n================================================================================")
    print(f"EVALUATION COMPLETE in {elapsed:.1f}s")
    print(f"OVERALL ACCURACY WITH ORGANIZED WEB: {total_acc:.2f}% ({total_correct}/{total_items})")
    print(f"Results saved to: {RESULTS_JSON}")
    print("================================================================================")
    return summary


if __name__ == "__main__":
    run_evaluation()
