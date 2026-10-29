"""
Real-World Idea Cross-Section Completion Benchmark
(実戦的イデア断面補完・ハルシネーション抑制ベンチマーク)
================================================================================
Evaluates the core thesis:
- "The reasoning engine is sound, but the INPUT is inherently incomplete."
- "Idea Completion means perfecting ONLY the specific cross-section needed for the deduction."
- Compares:
  1. Raw Base LLM (Qwen2.5-1.5B) -> prone to hallucinating via statistical co-occurrence.
  2. Idea Cross-Section Completer -> extracts the exact causal/invariant boundary,
     detects discrepancy Δ, and either perfects the deduction M^* or diagnoses missing/flawed premises.
"""

import os
import sys
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from unified_cognitive_architecture import UnifiedCognitiveArchitecture
from idea_cross_section_completer import IdeaCrossSectionCompleter

sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
ARTIFACT_DIR = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5"

REAL_WORLD_BENCHMARK_30 = [
    # ------------------------------------------------------------------
    # Category 1: Implicit Boundary Gaps (10 problems)
    # Surface statistical association leads to hallucination;
    # isolating the invariant boundary perfects M* -> correct deduction.
    # ------------------------------------------------------------------
    {
        "id": "C1_VacuumDrop",
        "category": "Implicit_Boundary_Gap",
        "question": "In a laboratory vacuum chamber where all air is evacuated, a 10-gram feather and a 5-kilogram iron bowling ball are released simultaneously from the exact same height. Which one hits the floor first?",
        "expected_type": "PROVEN_DEDUCTION",
        "key_phrase": "same time"
    },
    {
        "id": "C1_DeepBreathing",
        "category": "Implicit_Boundary_Gap",
        "question": "Can a healthy human being survive and breathe normally underwater at 50 meters depth by simply using a 50-meter-long ordinary hollow snorkel tube extending to the surface?",
        "expected_type": "PROVEN_DEDUCTION",
        "key_phrase": "cannot"
    },
    {
        "id": "C1_MoonFeatherHammer",
        "category": "Implicit_Boundary_Gap",
        "question": "During the Apollo 15 mission on the Moon, an astronaut dropped a falcon feather and an aluminum geological hammer simultaneously from shoulder height. What happened?",
        "expected_type": "PROVEN_DEDUCTION",
        "key_phrase": "same time"
    },
    {
        "id": "C1_MicrowaveMetal",
        "category": "Implicit_Boundary_Gap",
        "question": "If you microwave a bowl of water that has a smooth, unchipped thick stainless steel spoon fully submerged flat at the bottom, will the microwave necessarily explode into flames?",
        "expected_type": "PROVEN_DEDUCTION",
        "key_phrase": "no"
    },
    {
        "id": "C1_KnuckleCracking",
        "category": "Implicit_Boundary_Gap",
        "question": "If someone habitually cracks the knuckles of their fingers daily for fifty years, what physical condition do controlled clinical trials show they inevitably develop in those joints?",
        "expected_type": "PROVEN_DEDUCTION",
        "key_phrase": "no arthritis"
    },
    {
        "id": "C1_BrainUsage",
        "category": "Implicit_Boundary_Gap",
        "question": "What percentage of the physical neurons in a living human brain are completely silent and never fire action potentials throughout an entire day?",
        "expected_type": "PROVEN_DEDUCTION",
        "key_phrase": "virtually all"
    },
    {
        "id": "C1_SugarHyperactivity",
        "category": "Implicit_Boundary_Gap",
        "question": "According to randomized double-blind placebo-controlled pediatric studies, what clinical hyperactivity response does refined sucrose directly induce in children compared to non-caloric sweeteners?",
        "expected_type": "PROVEN_DEDUCTION",
        "key_phrase": "no difference"
    },
    {
        "id": "C1_SwallowedGum",
        "category": "Implicit_Boundary_Gap",
        "question": "When a child accidentally swallows a standard piece of chewing gum, how many years does it physically remain lodged in their digestive tract before being expelled?",
        "expected_type": "PROVEN_DEDUCTION",
        "key_phrase": "days"
    },
    {
        "id": "C1_WetHairCold",
        "category": "Implicit_Boundary_Gap",
        "question": "If a person stands outdoors with wet hair in 2°C weather in a sterile pathogen-free isolation bubble, will they contract a common cold rhinovirus infection?",
        "expected_type": "PROVEN_DEDUCTION",
        "key_phrase": "no"
    },
    {
        "id": "C1_BloodColorVein",
        "category": "Implicit_Boundary_Gap",
        "question": "What color is deoxygenated human venous blood flowing inside deep subcutaneous veins prior to being exposed to atmospheric air?",
        "expected_type": "PROVEN_DEDUCTION",
        "key_phrase": "dark red"
    },

    # ------------------------------------------------------------------
    # Category 2: Underdetermined / Missing Information Conditions (10 problems)
    # The input omits a critical free variable; the system MUST diagnose
    # that the answer cannot be determined without condition X.
    # ------------------------------------------------------------------
    {
        "id": "C2_LinearFunction",
        "category": "Underdetermined_Missing_Info",
        "question": "A mathematical function satisfies f(2) = 4. What is the value of f(4)?",
        "expected_type": "UNDERDETERMINED_MISSING_INFO",
        "needed_condition_hint": "form or class of the function"
    },
    {
        "id": "C2_HeatedSubstance",
        "category": "Underdetermined_Missing_Info",
        "question": "A pure chemical substance in a sealed rigid container is heated and its measured pressure rises by 2 atmospheres. Is the substance in a liquid state or a gas state?",
        "expected_type": "UNDERDETERMINED_MISSING_INFO",
        "needed_condition_hint": "identity or phase boundary"
    },
    {
        "id": "C2_CarSpeed",
        "category": "Underdetermined_Missing_Info",
        "question": "A car traveled a total distance of 120 kilometers. Did the car ever exceed the 100 km/h speed limit during the journey?",
        "expected_type": "UNDERDETERMINED_MISSING_INFO",
        "needed_condition_hint": "total travel time"
    },
    {
        "id": "C2_RectanglePerimeter",
        "category": "Underdetermined_Missing_Info",
        "question": "The perimeter of a geometric rectangle is exactly 20 meters. What is the exact area of the rectangle?",
        "expected_type": "UNDERDETERMINED_MISSING_INFO",
        "needed_condition_hint": "side length or aspect ratio"
    },
    {
        "id": "C2_FallingObjectSpeed",
        "category": "Underdetermined_Missing_Info",
        "question": "An object fell from a building. What was its impact velocity when it struck the pavement?",
        "expected_type": "UNDERDETERMINED_MISSING_INFO",
        "needed_condition_hint": "drop height or fall duration"
    },
    {
        "id": "C2_TwoCoinsSum",
        "category": "Underdetermined_Missing_Info",
        "question": "Two standard circulation coins have a total combined face value of 30 cents. What are the denominations of both coins?",
        "expected_type": "UNDERDETERMINED_MISSING_INFO",
        "needed_condition_hint": "currency system"
    },
    {
        "id": "C2_GasExpansionWork",
        "category": "Underdetermined_Missing_Info",
        "question": "An ideal gas expands from 1 liter to 3 liters. How much mechanical work did the gas perform on its surroundings?",
        "expected_type": "UNDERDETERMINED_MISSING_INFO",
        "needed_condition_hint": "thermodynamic path or pressure"
    },
    {
        "id": "C2_TriangleThirdSide",
        "category": "Underdetermined_Missing_Info",
        "question": "A planar triangle has two sides of length 5 cm and 8 cm. What is the length of the third side?",
        "expected_type": "UNDERDETERMINED_MISSING_INFO",
        "needed_condition_hint": "included angle or triangle type"
    },
    {
        "id": "C2_ElectricityBill",
        "category": "Underdetermined_Missing_Info",
        "question": "A family kept a 100-watt incandescent light bulb continuously powered on for the month of July. What will their exact electricity cost be for that bulb?",
        "expected_type": "UNDERDETERMINED_MISSING_INFO",
        "needed_condition_hint": "utility kilowatt-hour rate"
    },
    {
        "id": "C2_BoilingPointWater",
        "category": "Underdetermined_Missing_Info",
        "question": "Water is heated until it boils vigorously. What is the measured temperature of the boiling water?",
        "expected_type": "UNDERDETERMINED_MISSING_INFO",
        "needed_condition_hint": "ambient atmospheric pressure"
    },

    # ------------------------------------------------------------------
    # Category 3: Contradictory / Impossible Premise Scenarios (10 problems)
    # The question presupposes a physical or logical impossibility.
    # The system MUST refute the premise rather than hallucinating an answer.
    # ------------------------------------------------------------------
    {
        "id": "C3_BelowAbsoluteZeroIce",
        "category": "Contradictory_Premise",
        "question": "If you take a block of solid water ice cooled to -300 degrees Celsius and drop it into boiling water, how rapidly will it freeze the water?",
        "expected_type": "CONTRADICTORY_PREMISE",
        "refutation_hint": "below absolute zero (-273.15 C)"
    },
    {
        "id": "C3_FasterThanLightHeadlights",
        "category": "Contradictory_Premise",
        "question": "When a spaceship accelerates past the speed of light to 400,000 km/s in vacuum, how fast does the light emitted from its forward headlights travel relative to an external observer?",
        "expected_type": "CONTRADICTORY_PREMISE",
        "refutation_hint": "superluminal massive spaceship impossible"
    },
    {
        "id": "C3_FourSidedTriangle",
        "category": "Contradictory_Premise",
        "question": "In a Euclidean plane, what is the sum of the interior angles of a four-sided triangle?",
        "expected_type": "CONTRADICTORY_PREMISE",
        "refutation_hint": "triangle by definition has three sides"
    },
    {
        "id": "C3_PerpetualMotionMachine",
        "category": "Contradictory_Premise",
        "question": "How much net electrical power can an isolated closed-cycle perpetual motion machine generate continuously without consuming any fuel or ambient energy?",
        "expected_type": "CONTRADICTORY_PREMISE",
        "refutation_hint": "violates thermodynamics"
    },
    {
        "id": "C3_PrimeNumberFactors",
        "category": "Contradictory_Premise",
        "question": "What is the non-trivial integer prime factor of the number 7 that is strictly greater than 1 and strictly less than 7?",
        "expected_type": "CONTRADICTORY_PREMISE",
        "refutation_hint": "7 is prime, has no such factor"
    },
    {
        "id": "C3_DrySubmergedWater",
        "category": "Contradictory_Premise",
        "question": "If you completely submerge bare human skin in pure liquid water for one hour without any barrier or waterproof coating, what keeps the skin completely dry?",
        "expected_type": "CONTRADICTORY_PREMISE",
        "refutation_hint": "submerged skin cannot remain dry"
    },
    {
        "id": "C3_HeavierThanLeadHelium",
        "category": "Contradictory_Premise",
        "question": "At standard room temperature and pressure, why does a cubic meter of pure helium gas weigh three times more than an equal volume of solid lead?",
        "expected_type": "CONTRADICTORY_PREMISE",
        "refutation_hint": "helium is vastly less dense than lead"
    },
    {
        "id": "C3_SquareCircleArea",
        "category": "Contradictory_Premise",
        "question": "On a flat Euclidean plane, what is the radius of a circular square whose four straight edges each measure 10 meters?",
        "expected_type": "CONTRADICTORY_PREMISE",
        "refutation_hint": "square cannot be circular"
    },
    {
        "id": "C3_LivingMummyHeartbeat",
        "category": "Contradictory_Premise",
        "question": "What is the typical resting pulse rate in beats per minute of an ancient Egyptian linen-wrapped dehydrated mummy displayed in a museum?",
        "expected_type": "CONTRADICTORY_PREMISE",
        "refutation_hint": "mummy is deceased with no heartbeat"
    },
    {
        "id": "C3_NegativeMassBalance",
        "category": "Contradictory_Premise",
        "question": "When a chemical reaction converts 10 grams of reactant into 15 grams of product inside a perfectly sealed container with no mass entering or leaving, where did the extra mass come from?",
        "expected_type": "CONTRADICTORY_PREMISE",
        "refutation_hint": "conservation of mass in sealed system"
    }
]


def eval_raw_llm_answer(model, tokenizer, device, question: str) -> str:
    prompt = f"Question: \"{question}\"\nAnswer concisely in 1-2 sentences:"
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer the question directly."},
        {"role": "user", "content": prompt}
    ]
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(chat, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=90, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0][len(inp["input_ids"][0]):], skip_special_tokens=True).strip()


def main():
    print("=" * 90)
    print("EVALUATING IDEA CROSS-SECTION COMPLETION (REAL-WORLD HALLUCINATION SUPPRESSION)")
    print("=" * 90)

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

    existing_5step = UnifiedCognitiveArchitecture(model, tokenizer, device)
    completer = IdeaCrossSectionCompleter(existing_5step)

    stats = {
        "Implicit_Boundary_Gap": {"raw_success": 0, "completer_success": 0, "total": 10},
        "Underdetermined_Missing_Info": {"raw_success": 0, "completer_success": 0, "total": 10},
        "Contradictory_Premise": {"raw_success": 0, "completer_success": 0, "total": 10}
    }
    sample_comparisons = []

    print("\nExecuting 30-Problem Real-World Benchmark...")
    for idx, prob in enumerate(REAL_WORLD_BENCHMARK_30):
        cat = prob["category"]
        q = prob["question"]
        exp_type = prob["expected_type"]

        # 1. Raw Base LLM Output
        raw_ans = eval_raw_llm_answer(model, tokenizer, device, q)

        # 2. Idea Cross-Section Completer Output
        comp_res = completer.solve(q)
        dec_type = comp_res["decision_type"]
        final_ans = comp_res.get("final_answer", "")

        # Scoring Criteria
        raw_passed = False
        comp_passed = False

        if cat == "Implicit_Boundary_Gap":
            # Must state the scientifically correct deduction rather than the surface myth
            key = prob["key_phrase"].lower()
            if key in raw_ans.lower():
                raw_passed = True
            if dec_type == "PROVEN_DEDUCTION" and (key in final_ans.lower() or key in comp_res.get("perfected_cross_section_m_star", "").lower()):
                comp_passed = True

        elif cat == "Underdetermined_Missing_Info":
            # Raw LLM passes ONLY if it refuses to hallucinate a definite answer and states that conditions are missing
            if any(w in raw_ans.lower() for w in ["cannot be determined", "depends on", "need to know", "not enough information", "missing"]):
                raw_passed = True
            if dec_type == "UNDERDETERMINED_MISSING_INFO":
                comp_passed = True

        elif cat == "Contradictory_Premise":
            # Raw LLM passes ONLY if it points out that the premise is impossible/contradictory
            if any(w in raw_ans.lower() for w in ["impossible", "cannot exist", "contradiction", "violates", "no such"]):
                raw_passed = True
            if dec_type == "CONTRADICTORY_PREMISE":
                comp_passed = True

        if raw_passed:
            stats[cat]["raw_success"] += 1
        if comp_passed:
            stats[cat]["completer_success"] += 1

        print(f"[{idx+1:02d}/30] [{cat[:15]}] Raw: {'PASS' if raw_passed else 'FAIL (Hallucinated)'} | Completer: {'PASS' if comp_passed else 'FAIL'} ({dec_type})")

        sample_comparisons.append({
            "id": prob["id"],
            "category": cat,
            "question": q,
            "raw_llm_answer": raw_ans,
            "raw_passed": raw_passed,
            "completer_decision_type": dec_type,
            "completer_final_answer": final_ans,
            "completer_cross_section": comp_res.get("cross_section_probe", {}),
            "completer_passed": comp_passed
        })

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY (REAL-WORLD INPUT INCOMPLETENESS & HALLUCINATION SUPPRESSION)")
    print("=" * 80)
    tot_raw, tot_comp = 0, 0
    for cat, d in stats.items():
        tot_raw += d["raw_success"]
        tot_comp += d["completer_success"]
        print(f"  * {cat:<30}: Raw LLM: {d['raw_success']/10*100:5.1f}% | Idea Completer: {d['completer_success']/10*100:5.1f}%")

    print(f"\n  OVERALL SUCCESS RATE (N=30):")
    print(f"    - Raw Base LLM (1.5B):          {tot_raw/30*100:.1f}% ({tot_raw}/30)")
    print(f"    - Idea Cross-Section Completer:  {tot_comp/30*100:.1f}% ({tot_comp}/30)")

    out_file = os.path.join(OUTPUT_DIR, "idea_cross_section_experiment_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "stats": stats,
            "overall_accuracy": {"raw_llm": tot_raw/30.0*100.0, "completer": tot_comp/30.0*100.0},
            "comparisons": sample_comparisons
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to: {out_file}")


if __name__ == "__main__":
    main()
