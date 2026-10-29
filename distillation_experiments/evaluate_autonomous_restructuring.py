"""
Evaluate Contradiction Reduction in Restructured Thought Wilderness
================================================================================
Compares Base Model vs Restructured Model across the 12 Diagnostic Probes.
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from diagnose_thought_wilderness_contradictions import CONTRADICTION_PROBES, extract_concept_vector

sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
ARTIFACT_DIR = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5"
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
AUTONOMOUS_LORA_DIR = os.path.join(OUTPUT_DIR, "lora_autonomous_thought_wilderness")


def evaluate():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    model = PeftModel.from_pretrained(base_model, AUTONOMOUS_LORA_DIR)
    model.eval()

    # Load baseline
    base_data = json.load(open(os.path.join(OUTPUT_DIR, "thought_wilderness_contradiction_diagnostics.json"), encoding="utf-8"))
    base_records = {r["id"]: r for r in base_data["records"]}

    results = []
    print("=" * 85)
    print("RESTRUCTURED THOUGHT WILDERNESS: POST-ANNEALING CONTRADICTION AUDIT")
    print("=" * 85)

    for probe in CONTRADICTION_PROBES:
        p_id = probe["id"]
        v_a = extract_concept_vector(model, tokenizer, probe["concept_a"])
        v_myth = extract_concept_vector(model, tokenizer, probe["concept_b_myth"])
        v_fact = extract_concept_vector(model, tokenizer, probe["concept_b_fact"])

        cos_myth = float(torch.dot(v_a, v_myth).item())
        cos_fact = float(torch.dot(v_a, v_fact).item())
        distortion = cos_myth - cos_fact

        base_r = base_records.get(p_id, {})
        base_dist = base_r.get("contradiction_distortion", 0.0)

        results.append({
            "id": p_id,
            "category": probe["category"],
            "base_distortion": base_dist,
            "post_distortion": distortion,
            "delta_improvement": base_dist - distortion,
            "is_distorted": distortion > 0.0
        })

        status = "RESOLVED" if (base_dist > 0 and distortion <= 0) else ("STILL DISTORTED" if distortion > 0 else "CONSISTENT")
        print(f"[{p_id}] Baseline: {base_dist:+.4f} --> Post: {distortion:+.4f} (Shift: {base_dist - distortion:+.4f}) [{status}]")

    post_distorted = sum(1 for r in results if r["is_distorted"])
    avg_base_dist = np.mean([r["base_distortion"] for r in results])
    avg_post_dist = np.mean([r["post_distortion"] for r in results])

    print("=" * 85)
    print(f"SUMMARY AUDIT:")
    print(f"  Distorted Probes: {base_data['summary']['distorted_count']}/12 ({base_data['summary']['distorted_ratio']*100:.1f}%) --> {post_distorted}/12 ({post_distorted/12*100:.1f}%)")
    print(f"  Mean Distortion: {avg_base_dist:+.4f} --> {avg_post_dist:+.4f}")
    print("=" * 85)

    # Save
    out_file = os.path.join(OUTPUT_DIR, "autonomous_restructuring_comparison.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "pre_distorted_count": base_data['summary']['distorted_count'],
                "post_distorted_count": post_distorted,
                "pre_mean_distortion": float(avg_base_dist),
                "post_mean_distortion": float(avg_post_dist)
            },
            "records": results
        }, f, indent=2, ensure_ascii=False)

    # Plot Comparison Chart
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 6))

    labels = [r["id"].replace("Myth_", "").replace("Physics_", "").replace("Formal_", "") for r in results]
    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width/2, [r["base_distortion"] for r in results], width, label="Baseline Thought Wilderness", color="#e74c3c", alpha=0.85)
    ax.bar(x + width/2, [r["post_distortion"] for r in results], width, label="Autonomously Restructured", color="#2ecc71", alpha=0.85)

    ax.axhline(0, color="black", linestyle="--", linewidth=1.0)
    ax.set_ylabel("Contradiction Distortion [cos(Myth) - cos(Fact)]\n(Values > 0 indicate erroneous attraction to myth)")
    ax.set_title("Autonomous Thought Wilderness Restructuring: Contradiction Elimination")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.legend(loc="upper left")
    plt.tight_layout()

    chart_path = os.path.join(ARTIFACT_DIR, "autonomous_thought_wilderness_comparison.png")
    plt.savefig(chart_path, dpi=300)
    print(f"Saved comparison chart to: {chart_path}")


if __name__ == "__main__":
    evaluate()
