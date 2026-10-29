"""
Verification of 4-Valued Epistemic Unprovability & Premise Error Diagnosis
(証明不能時の不足情報逆算特定・質問／前提誤り検出の厳格検証)
================================================================================
Tests whether the system can distinguish between:
1. `PROVEN_TRUE`: Uniquely proven True after contradiction-free Idea-Mapping completion.
2. `PROVEN_FALSE`: Explicitly refuted by negative boundary observations (Modus Tollens / polarity clash).
3. `UNDERDETERMINED_MISSING_INFO`: Cannot be proven True or False because multiple branches survive;
   must identify the exact missing observations required to resolve the question.
4. `INVALID_PREMISE_OR_QUERY`: The user's question or the provided premises themselves are invalid
   (all candidate continuations collapse into contradiction, base premises self-contradict, or
   the query presupposes an unknown entity/concept).
"""

import os
import sys
import json
import random
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from unified_cognitive_architecture import UnifiedCognitiveArchitecture
from idea_mapping_completion_system import ObservationCompletedCognitiveSystem
from run_observation_completion_benchmark import SyntheticNonceGenerator

sys.stdout.reconfigure(encoding='utf-8')


def build_epistemic_diagnosis_suite(arch: UnifiedCognitiveArchitecture) -> List[Dict]:
    gen = SyntheticNonceGenerator(seed=77777)
    rng = random.Random(77777)
    problems = []

    # ------------------------------------------------------------------
    # Group 1: Determinate Cases (N = 20: 10 PROVEN_TRUE, 10 PROVEN_FALSE)
    # ------------------------------------------------------------------
    for i in range(20):
        c1, c2 = gen.next_concepts(2)
        b1 = gen.next_concepts(2)
        b2 = gen.next_concepts(2)
        b3 = gen.next_concepts(2)
        subj = gen.next_entity()

        # b1 and b2 are refuted, leaving b3 as the unique proven bridge
        context = (
            f"{subj} is a {c1}. Each {c1} is a {c2}. "
            f"Each {b1[0]} is a {b1[1]}. Each {b2[0]} is a {b2[1]}. Each {b3[0]} is a {b3[1]}. "
            f"{subj} is not a {b1[1]}. {subj} is not a {b2[1]}."
        )
        if i % 2 == 0:
            question = f"Is {subj} a {b3[1]}?"
            expected_status = "PROVEN_TRUE"
        else:
            # Query about b1[0], which is actively refuted via Modus Tollens (since subj is not b1[1])
            question = f"Is {subj} a {b1[0]}?"
            expected_status = "PROVEN_FALSE"

        problems.append({
            "id": f"diag_determinate_{i+1:02d}",
            "group": "Group1_Determinate_True_Or_False",
            "context": context,
            "question": question,
            "expected_status": expected_status
        })

    # ------------------------------------------------------------------
    # Group 2: Underdetermined / Missing Information Cases (N = 20)
    # ------------------------------------------------------------------
    for i in range(20):
        c1, c2 = gen.next_concepts(2)
        b1 = gen.next_concepts(2)
        b2 = gen.next_concepts(2)
        b3 = gen.next_concepts(2)
        subj = gen.next_entity()

        # Only b1 is refuted; both b2 and b3 survive without contradiction!
        context = (
            f"{subj} is a {c1}. Each {c1} is a {c2}. "
            f"Each {b1[0]} is a {b1[1]}. Each {b2[0]} is a {b2[1]}. Each {b3[0]} is a {b3[1]}. "
            f"{subj} is not a {b1[1]}."
        )
        question = f"Is {subj} a {b2[1]}?"
        problems.append({
            "id": f"diag_underdetermined_{i+1:02d}",
            "group": "Group2_Underdetermined_Missing_Info",
            "context": context,
            "question": question,
            "expected_status": "UNDERDETERMINED_MISSING_INFO",
            "expected_surviving_branches": sorted([b2[0], b3[0]])
        })

    # ------------------------------------------------------------------
    # Group 3: Invalid Premise or Flawed User Query Cases (N = 20)
    #   - 7 problems: All candidate continuations refuted (No consistent world exists)
    #   - 7 problems: Direct self-contradiction in premises
    #   - 6 problems: User query presupposes a non-existent concept/entity
    # ------------------------------------------------------------------
    for i in range(20):
        c1, c2 = gen.next_concepts(2)
        b1 = gen.next_concepts(2)
        b2 = gen.next_concepts(2)
        subj = gen.next_entity()

        if i < 7:
            # Subcase 3a: All candidate branches (b1 and b2) are refuted by negative observations
            context = (
                f"{subj} is a {c1}. Each {c1} is a {c2}. "
                f"Each {b1[0]} is a {b1[1]}. Each {b2[0]} is a {b2[1]}. "
                f"{subj} is not a {b1[1]}. {subj} is not a {b2[1]}."
            )
            question = f"Is {subj} a {b1[1]}?"
            err_subtype = "all_candidate_branches_contradicted"
        elif i < 14:
            # Subcase 3b: Base premises themselves entail a direct contradiction (subj is c2 and NOT c2)
            context = (
                f"{subj} is a {c1}. Each {c1} is a {c2}. "
                f"Each {b1[0]} is a {b1[1]}. {subj} is not a {c2}."
            )
            question = f"Is {subj} a {b1[1]}?"
            err_subtype = "self_contradictory_premises"
        else:
            # Subcase 3c: User query asks about a concept or entity completely absent from the world
            unknown_concept = gen.next_concept()
            context = (
                f"{subj} is a {c1}. Each {c1} is a {c2}. "
                f"Each {b1[0]} is a {b1[1]}."
            )
            question = f"Is {subj} a {unknown_concept}?"
            err_subtype = "unknown_query_predicate"

        problems.append({
            "id": f"diag_invalid_{i+1:02d}",
            "group": "Group3_Invalid_Premise_Or_Query",
            "err_subtype": err_subtype,
            "context": context,
            "question": question,
            "expected_status": "INVALID_PREMISE_OR_QUERY"
        })

    return problems


def eval_raw_4way(model, tokenizer, device, context: str, question: str) -> str:
    prompt = (
        f"Context: {context}\n"
        f"Question: {question}\n\n"
        f"Classify the logical status of this question into strictly ONE of the following 4 labels:\n"
        f"1. PROVEN_TRUE (definitely true from the context)\n"
        f"2. PROVEN_FALSE (definitely false / refuted from the context)\n"
        f"3. UNDERDETERMINED_MISSING_INFO (cannot be proven true or false because necessary rules/observations are missing)\n"
        f"4. INVALID_PREMISE_OR_QUERY (the context is self-contradictory, all branches are contradicted, or the question asks about an unknown concept)\n"
        f"Output ONLY the label name."
    )
    messages = [
        {"role": "system", "content": "Output strictly one of: PROVEN_TRUE, PROVEN_FALSE, UNDERDETERMINED_MISSING_INFO, INVALID_PREMISE_OR_QUERY."},
        {"role": "user", "content": prompt}
    ]
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(chat, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    resp = tokenizer.decode(out[0][len(inp["input_ids"][0]):], skip_special_tokens=True).strip().upper()
    for label in ["UNDERDETERMINED_MISSING_INFO", "INVALID_PREMISE_OR_QUERY", "PROVEN_TRUE", "PROVEN_FALSE"]:
        if label in resp:
            return label
    return "PROVEN_FALSE"


def main():
    print("=" * 85)
    print("VERIFYING 4-VALUED EPISTEMIC DIAGNOSIS (MISSING INFO & INVALID PREMISE DETECTION)")
    print("=" * 85)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    model.eval()

    existing_5step = UnifiedCognitiveArchitecture(model, tokenizer, device)
    completed_system = ObservationCompletedCognitiveSystem(existing_5step)

    problems = build_epistemic_diagnosis_suite(existing_5step)

    stats = {
        "Group1_Determinate_True_Or_False": {"raw": 0, "ex": 0, "comp": 0, "total": 0},
        "Group2_Underdetermined_Missing_Info": {"raw": 0, "ex": 0, "comp": 0, "total": 0, "exact_missing_pinpointed": 0},
        "Group3_Invalid_Premise_Or_Query": {"raw": 0, "ex": 0, "comp": 0, "total": 0},
    }
    sample_diagnostics = {}

    for p in problems:
        grp = p["group"]
        stats[grp]["total"] += 1

        # 1. Raw LLM 4-way classification
        raw_status = eval_raw_4way(model, tokenizer, device, p["context"], p["question"])

        # 2. Existing 5-Step System Alone (Only produces binary True/False -> maps to PROVEN_TRUE / PROVEN_FALSE)
        existing_5step.step3.proven_lemmas.clear()
        res_ex = existing_5step.solve(question=p["question"], context=p["context"])
        ex_status = "PROVEN_TRUE" if res_ex["decision"] else "PROVEN_FALSE"

        # 3. Completed System with 4-Valued Epistemic Diagnosis
        existing_5step.step3.proven_lemmas.clear()
        res_comp = completed_system.solve(question=p["question"], context=p["context"])
        comp_status = res_comp["epistemic_status"]
        diag = res_comp["epistemic_diagnosis"]

        if raw_status == p["expected_status"]:
            stats[grp]["raw"] += 1
        if ex_status == p["expected_status"]:
            stats[grp]["ex"] += 1
        if comp_status == p["expected_status"]:
            stats[grp]["comp"] += 1

        if grp == "Group2_Underdetermined_Missing_Info" and comp_status == "UNDERDETERMINED_MISSING_INFO":
            warnings = res_comp["mapping_completion"].get("underdetermined_warnings", [])
            if warnings:
                surv = sorted(warnings[-1].get("surviving_branches", []))
                if surv == p["expected_surviving_branches"]:
                    stats[grp]["exact_missing_pinpointed"] += 1

        if grp not in sample_diagnostics:
            sample_diagnostics[grp] = {
                "problem_id": p["id"],
                "context": p["context"],
                "question": p["question"],
                "expected_status": p["expected_status"],
                "raw_llm_status": raw_status,
                "existing_5step_status": ex_status,
                "completed_epistemic_diagnosis": diag
            }

    print("\n--- 4-Valued Epistemic Diagnosis Results (N = 60) ---")
    total_raw, total_ex, total_comp = 0, 0, 0
    for grp, d in stats.items():
        tot = d["total"]
        total_raw += d["raw"]
        total_ex += d["ex"]
        total_comp += d["comp"]
        print(
            f"  * {grp} (N={tot}):\n"
            f"      Raw LLM (4-way):                  {d['raw']/tot*100:.1f}% ({d['raw']}/{tot})\n"
            f"      Existing 5-Step (Binary Only):    {d['ex']/tot*100:.1f}% ({d['ex']}/{tot})\n"
            f"      Idea-Mapping Diagnostic System:   {d['comp']/tot*100:.1f}% ({d['comp']}/{tot})"
        )
        if "exact_missing_pinpointed" in d:
            print(
                f"      -> Exact Missing Observation Set Pinpointed: "
                f"{d['exact_missing_pinpointed']/tot*100:.1f}% ({d['exact_missing_pinpointed']}/{tot})"
            )

    print(
        f"\n  OVERALL 4-WAY DIAGNOSTIC ACCURACY (N=60):\n"
        f"    - Raw Base LLM:                   {total_raw/60*100:.2f}% ({total_raw}/60)\n"
        f"    - Existing 5-Step System Alone:   {total_ex/60*100:.2f}% ({total_ex}/60)\n"
        f"    - Idea-Mapping Diagnostic System: {total_comp/60*100:.2f}% ({total_comp}/60)"
    )

    out_path = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment/epistemic_diagnosis_verification_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "stats": stats,
            "overall_accuracy": {
                "raw_llm": total_raw / 60.0 * 100.0,
                "existing_5step": total_ex / 60.0 * 100.0,
                "completed_diagnostic_system": total_comp / 60.0 * 100.0
            },
            "sample_diagnostics": sample_diagnostics
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved diagnostic verification results to: {out_path}")


if __name__ == "__main__":
    main()
