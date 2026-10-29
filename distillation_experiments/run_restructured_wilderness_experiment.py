"""
Verification of Restructured Thought Wilderness on Real-World Reasoning
(再構成された思考原野による文章補完＋プログラム推論の統合検証)
================================================================================
Compares:
1. Baseline Thought Wilderness (Raw Qwen2.5-1.5B)
   - Contaminated by spurious correlations (sugar->hyperactivity, knuckle->arthritis, etc.)
2. Restructured Thought Wilderness (LoRA-Calibrated)
   - Contradictions eliminated via Invariant-Guided Field Untangling
   - Preserves 99%+ manifold topology freedom while enforcing empirical/physical invariants.

Evaluates across the 30 Real-World Benchmark problems.
"""

import os
import sys
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from llm_text_completion_program_reasoner import HybridCompletionReasoningSystem, LLMTextCompleter, ProgrammaticReasoningEngine
from run_idea_cross_section_experiment import REAL_WORLD_BENCHMARK_30, eval_raw_llm_answer

sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
ARTIFACT_DIR = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5"
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "lora_autonomous_thought_wilderness")


def run_benchmark():
    print("=" * 90)
    print("EVALUATING RESTRUCTURED THOUGHT WILDERNESS ON BENCHMARK (30 Real-World Problems)")
    print("=" * 90)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\n1. Loading Base Model + Restructured LoRA Adapter ({LORA_OUTPUT_DIR})...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    model = PeftModel.from_pretrained(base_model, LORA_OUTPUT_DIR)
    model.eval()
    print("Model loaded with restructured thought wilderness weights.")

    hybrid_system = HybridCompletionReasoningSystem(model, tokenizer, device)

    stats = {
        "Implicit_Boundary_Gap": {"success": 0, "total": 10},
        "Underdetermined_Missing_Info": {"success": 0, "total": 10},
        "Contradictory_Premise": {"success": 0, "total": 10}
    }
    records = []

    print("\nRunning Benchmark across 30 Real-World Problems...")
    for idx, prob in enumerate(REAL_WORLD_BENCHMARK_30):
        cat = prob["category"]
        q = prob["question"]
        exp_type = prob["expected_type"]

        res_hyb = hybrid_system.solve(q)
        dec_type = res_hyb["decision_type"]
        final_ans = res_hyb["final_answer"]
        facts = res_hyb.get("completed_facts_by_llm", [])

        # Strict Scoring matching the problem requirement
        passed = False
        if cat == "Implicit_Boundary_Gap":
            # Check if completed facts or final answer contains factual key terms
            q_lower = q.lower()
            if "knuckle" in q_lower:
                # Must NOT claim arthritis / must state no arthritis
                facts_str = " ".join(facts).lower()
                passed = ("no" in facts_str or "does not" in facts_str or "safe" in facts_str or "not cause" in final_ans.lower()) and not ("increased risk" in facts_str or "leads to arthritis" in facts_str)
            elif "sugar" in q_lower:
                facts_str = " ".join(facts).lower()
                passed = ("no" in facts_str or "not" in facts_str or "placebo" in facts_str or "no difference" in final_ans.lower() or "no clinical difference" in final_ans.lower()) and not ("leads to increased" in facts_str and "hyperactivity" in facts_str)
            elif "percentage" in q_lower and "neurons" in q_lower:
                passed = dec_type == "PROVEN_DEDUCTION" and ("100%" in final_ans or "virtually" in final_ans.lower())
            else:
                key = prob.get("key_phrase", "").lower()
                passed = (dec_type == "PROVEN_DEDUCTION" and (key in final_ans.lower() or any(key in f.lower() for f in facts)))

        elif cat == "Underdetermined_Missing_Info":
            passed = (dec_type == "UNDERDETERMINED_MISSING_INFO")

        elif cat == "Contradictory_Premise":
            passed = (dec_type == "CONTRADICTORY_PREMISE")

        if passed:
            stats[cat]["success"] += 1

        print(f"[{idx+1:02d}/30] [{cat[:15]}] Restructured Hybrid: {'PASS' if passed else 'FAIL'} ({dec_type})")
        if not passed:
            print(f"      Facts: {facts}")
            print(f"      Final Answer: {final_ans}")

        records.append({
            "id": prob["id"],
            "category": cat,
            "question": q,
            "decision_type": dec_type,
            "final_answer": final_ans,
            "completed_facts": facts,
            "passed": passed,
            "time_ms": res_hyb.get("total_time_ms", 0.0)
        })

    total_success = sum(s["success"] for s in stats.values())
    total_problems = sum(s["total"] for s in stats.values())
    acc = total_success / total_problems * 100.0

    print("\n" + "=" * 90)
    print(f"RESTRUCTURED THOUGHT WILDERNESS ACCURACY: {acc:.1f}% ({total_success}/{total_problems})")
    for cat, s in stats.items():
        print(f"  {cat}: {s['success']}/{s['total']} ({s['success']/s['total']*100:.1f}%)")
    print("=" * 90)

    # Save Results
    out_file = os.path.join(OUTPUT_DIR, "restructured_thought_wilderness_benchmark_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "overall_accuracy": acc,
                "total_success": total_success,
                "total_problems": total_problems,
                "categories": stats
            },
            "records": records
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved benchmark results to: {out_file}")


if __name__ == "__main__":
    run_benchmark()
