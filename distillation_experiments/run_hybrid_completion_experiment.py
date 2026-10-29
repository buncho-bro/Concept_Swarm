"""
Verification of Hybrid Architecture: LLM Text Completion + Program Reasoning
(文章補完のみLLM、推論はプログラム完結アーキテクチャの実戦検証)
================================================================================
Evaluates:
1. Raw Base LLM (1.5B) -> attempts to reason directly via text generation (prone to hallucination).
2. Hybrid Architecture -> LLM completes text gaps, Program executes 100% of reasoning.
"""

import os
import sys
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_text_completion_program_reasoner import HybridCompletionReasoningSystem
from run_idea_cross_section_experiment import REAL_WORLD_BENCHMARK_30, eval_raw_llm_answer

sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
ARTIFACT_DIR = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5"


def main():
    print("=" * 95)
    print("TESTING: LLM FOR TEXT COMPLETION ONLY + PROGRAM SYSTEM FOR 100% REASONING")
    print("=" * 95)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"Loading {model_name} on {device} (bfloat16)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    model.eval()

    hybrid_system = HybridCompletionReasoningSystem(model, tokenizer, device)

    stats = {
        "Implicit_Boundary_Gap": {"raw_success": 0, "hybrid_success": 0, "total": 10},
        "Underdetermined_Missing_Info": {"raw_success": 0, "hybrid_success": 0, "total": 10},
        "Contradictory_Premise": {"raw_success": 0, "hybrid_success": 0, "total": 10}
    }
    sample_records = []

    print("\nRunning Benchmark across 30 Real-World Problems...")
    for idx, prob in enumerate(REAL_WORLD_BENCHMARK_30):
        cat = prob["category"]
        q = prob["question"]
        exp_type = prob["expected_type"]

        # 1. Raw LLM
        raw_ans = eval_raw_llm_answer(model, tokenizer, device, q)

        # 2. Hybrid System (LLM Text Completion + Program Reasoning)
        res_hyb = hybrid_system.solve(q)
        dec_type = res_hyb["decision_type"]
        final_ans = res_hyb["final_answer"]

        # Scoring
        raw_pass = False
        hyb_pass = False

        if cat == "Implicit_Boundary_Gap":
            key = prob["key_phrase"].lower()
            if key in raw_ans.lower():
                raw_pass = True
            if dec_type == "PROVEN_DEDUCTION" and key in final_ans.lower():
                hyb_pass = True

        elif cat == "Underdetermined_Missing_Info":
            if any(w in raw_ans.lower() for w in ["cannot be determined", "depends on", "need to know", "not enough information", "missing"]):
                raw_pass = True
            if dec_type == "UNDERDETERMINED_MISSING_INFO":
                hyb_pass = True

        elif cat == "Contradictory_Premise":
            if any(w in raw_ans.lower() for w in ["impossible", "cannot exist", "contradiction", "violates", "no such"]):
                raw_pass = True
            if dec_type == "CONTRADICTORY_PREMISE":
                hyb_pass = True

        if raw_pass:
            stats[cat]["raw_success"] += 1
        if hyb_pass:
            stats[cat]["hybrid_success"] += 1

        print(f"[{idx+1:02d}/30] [{cat[:15]}] Raw: {'PASS' if raw_pass else 'FAIL (Hallucinated)'} | Hybrid: {'PASS' if hyb_pass else 'FAIL'} ({dec_type})")

        sample_records.append({
            "id": prob["id"],
            "category": cat,
            "question": q,
            "raw_answer": raw_ans,
            "raw_passed": raw_pass,
            "hybrid_decision_type": dec_type,
            "hybrid_final_answer": final_ans,
            "llm_completed_facts": res_hyb.get("completed_facts_by_llm", []),
            "program_identified_gap": res_hyb.get("identified_gap_by_program", ""),
            "hybrid_passed": hyb_pass,
            "time_ms": res_hyb.get("total_time_ms", 0)
        })

    print("\n" + "=" * 90)
    print("HYBRID SYSTEM VERIFICATION SUMMARY:")
    print("=" * 90)
    tot_raw, tot_hyb = 0, 0
    for cat, d in stats.items():
        tot_raw += d["raw_success"]
        tot_hyb += d["hybrid_success"]
        print(f"  * {cat:<30}: Raw LLM: {d['raw_success']/10*100:5.1f}% | Hybrid Architecture: {d['hybrid_success']/10*100:5.1f}%")

    print(f"\n  OVERALL SUCCESS RATE (N=30):")
    print(f"    - Raw Base LLM (1.5B):               {tot_raw/30*100:.1f}% ({tot_raw}/30)")
    print(f"    - Hybrid System (LLM Text + Prog):   {tot_hyb/30*100:.1f}% ({tot_hyb}/30)")

    out_file = os.path.join(OUTPUT_DIR, "hybrid_completion_experiment_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "stats": stats,
            "overall_accuracy": {"raw_llm": tot_raw/30.0*100.0, "hybrid": tot_hyb/30.0*100.0},
            "records": sample_records
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to: {out_file}")


if __name__ == "__main__":
    main()
