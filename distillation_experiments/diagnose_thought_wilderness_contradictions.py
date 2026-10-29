"""
Thought Wilderness Contradiction Diagnostics
(思考原野の幾何学的矛盾診断スクリプト)
================================================================================
Measures latent semantic forces (cosine similarity, potential energy, directional alignment)
between concepts across 3 core contradiction domains:
1. Scientific Myths & Spurious Correlations (e.g. Sugar-Hyperactivity, Knuckle-Arthritis)
2. Naive Intuitive Physics vs Invariant Physical Laws (e.g. Vacuum Fall, Deep Snorkel)
3. Presupposition Failures vs Formal Reality (e.g. Superluminal Travel, Below Absolute Zero)
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
ARTIFACT_DIR = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5"
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# Benchmark of Contradiction Probes in the Thought Wilderness
CONTRADICTION_PROBES = [
    # Category 1: Scientific Myths vs Empirical Reality
    {
        "id": "Myth_SugarHyperactivity",
        "category": "Scientific_Myth",
        "concept_a": "refined sugar intake in children",
        "concept_b_myth": "clinical behavioral hyperactivity and attention deficit",
        "concept_b_fact": "caloric energy metabolism without behavioral alteration in double-blind trials",
        "expected_true_relation": "independent/repulsive (myth is false)",
    },
    {
        "id": "Myth_KnuckleArthritis",
        "category": "Scientific_Myth",
        "concept_a": "habitual knuckle joint cracking",
        "concept_b_myth": "osteoarthritis and permanent joint damage",
        "concept_b_fact": "synovial fluid cavitation without articular cartilage degeneration",
        "expected_true_relation": "independent/repulsive (myth is false)",
    },
    {
        "id": "Myth_BrainTenPercent",
        "category": "Scientific_Myth",
        "concept_a": "human brain neurological functioning",
        "concept_b_myth": "ninety percent silent inactive unused neurons",
        "concept_b_fact": "continuous distributed action potential firing across entire cortex",
        "expected_true_relation": "independent/repulsive (myth is false)",
    },
    {
        "id": "Myth_SwallowedGumSevenYears",
        "category": "Scientific_Myth",
        "concept_a": "swallowed insoluble chewing gum",
        "concept_b_myth": "trapped in digestive tract for seven years",
        "concept_b_fact": "normal gastrointestinal transit and excretion within days",
        "expected_true_relation": "independent/repulsive (myth is false)",
    },
    {
        "id": "Myth_WetHairCausesCold",
        "category": "Scientific_Myth",
        "concept_a": "cold ambient exposure with wet hair",
        "concept_b_myth": "direct contraction of upper respiratory viral infection",
        "concept_b_fact": "pathogen transmission via viral agents independent of hair dampness",
        "expected_true_relation": "independent/repulsive (myth is false)",
    },
    {
        "id": "Myth_BloodTypePersonality",
        "category": "Scientific_Myth",
        "concept_a": "ABO erythrocyte surface antigen blood group",
        "concept_b_myth": "inherent leadership temperament and psychological traits",
        "concept_b_fact": "biological hematology devoid of behavioral psychological correlation",
        "expected_true_relation": "independent/repulsive (myth is false)",
    },

    # Category 2: Naive Physics vs Invariant Physical Laws
    {
        "id": "Physics_VacuumFreeFall",
        "category": "Physics_Invariant",
        "concept_a": "free fall in complete vacuum",
        "concept_b_myth": "heavier massive object falls significantly faster than light feather",
        "concept_b_fact": "mass-independent identical gravitational acceleration",
        "expected_true_relation": "repulsive to naive mass-dependence, attractive to invariant g",
    },
    {
        "id": "Physics_DeepSnorkelBreathing",
        "category": "Physics_Invariant",
        "concept_a": "breathing surface air at fifty meters underwater",
        "concept_b_myth": "easy regular pulmonary inhalation via long simple tube",
        "concept_b_fact": "fatal thoracic compression from six atmospheres hydrostatic pressure gradient",
        "expected_true_relation": "repulsive to naive tube breathing, attractive to barotrauma",
    },
    {
        "id": "Physics_MicrowaveSubmergedMetal",
        "category": "Physics_Invariant",
        "concept_a": "submerged smooth rounded stainless steel spoon in water bowl inside microwave",
        "concept_b_myth": "immediate catastrophic electrical explosion and fire",
        "concept_b_fact": "dissipated electrical charge without arcing or ignition",
        "expected_true_relation": "repulsive to explosion myth, attractive to safe dissipation",
    },
    {
        "id": "Physics_SuperluminalMass",
        "category": "Physics_Invariant",
        "concept_a": "physical spaceship with rest mass",
        "concept_b_myth": "accelerating beyond light speed to four hundred thousand km per second",
        "concept_b_fact": "asymptotic infinite relativistic energy barrier strictly bounding velocity below c",
        "expected_true_relation": "repulsive to superluminal velocity, attractive to relativistic limit",
    },

    # Category 3: Epistemic & Formal Presupposition Consistency
    {
        "id": "Formal_FourSidedTriangle",
        "category": "Formal_Logic",
        "concept_a": "geometric planar triangle",
        "concept_b_myth": "quadrilateral polygon with four distinct straight sides",
        "concept_b_fact": "three-sided closed polygonal shape by Euclidean definition",
        "expected_true_relation": "strictly repulsive (definitionally contradictory)",
    },
    {
        "id": "Formal_BelowAbsoluteZero",
        "category": "Formal_Logic",
        "concept_a": "thermodynamic kinetic temperature of gas",
        "concept_b_myth": "cooling down to negative three hundred degrees Celsius",
        "concept_b_fact": "strictly bounded lower fundamental limit at zero Kelvin (-273.15 C)",
        "expected_true_relation": "strictly repulsive (physically impossible)",
    }
]


def extract_concept_vector(model, tokenizer, concept_text: str) -> torch.Tensor:
    """Extracts the deep Thought Wilderness representation (late hidden state mean)."""
    prompt = f"Concept representation: {concept_text}"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
        # Deep representation (Layer 24 / -2 hidden state)
        h = out.hidden_states[-2][0].float()
        # Mean pooling across concept tokens
        h_vec = h.mean(dim=0)
        h_norm = F.normalize(h_vec, p=2, dim=-1)
    return h_norm


def run_diagnostics():
    print("=" * 80)
    print("THOUGHT WILDERNESS CONTRADICTION DIAGNOSTICS")
    print(f"Model: {MODEL_ID} on {device}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    model.eval()

    results = []
    print("\nScanning Latent Potentials across Concept Pairs...\n")

    for probe in CONTRADICTION_PROBES:
        vec_a = extract_concept_vector(model, tokenizer, probe["concept_a"])
        vec_myth = extract_concept_vector(model, tokenizer, probe["concept_b_myth"])
        vec_fact = extract_concept_vector(model, tokenizer, probe["concept_b_fact"])

        cos_myth = float(torch.dot(vec_a, vec_myth).item())
        cos_fact = float(torch.dot(vec_a, vec_fact).item())

        # Contradiction Bias = cos(A, Myth) - cos(A, Fact)
        # If > 0, the Thought Wilderness erroneously pulls toward the myth more than the fact!
        contradiction_distortion = cos_myth - cos_fact

        results.append({
            "id": probe["id"],
            "category": probe["category"],
            "concept_a": probe["concept_a"],
            "concept_b_myth": probe["concept_b_myth"],
            "concept_b_fact": probe["concept_b_fact"],
            "cos_with_myth": cos_myth,
            "cos_with_fact": cos_fact,
            "contradiction_distortion": contradiction_distortion,
            "is_distorted": contradiction_distortion > 0.0
        })

        status = "DISTORTED (Myth Attraction > Fact)" if contradiction_distortion > 0 else "CONSISTENT (Fact > Myth)"
        print(f"[{probe['id']}]")
        print(f"  cos(A, Myth) = {cos_myth:+.4f} | cos(A, Fact) = {cos_fact:+.4f}")
        print(f"  Distortion (Myth - Fact): {contradiction_distortion:+.4f} --> {status}\n")

    # Aggregate Statistics
    distorted_count = sum(1 for r in results if r["is_distorted"])
    avg_myth_cos = np.mean([r["cos_with_myth"] for r in results])
    avg_fact_cos = np.mean([r["cos_with_fact"] for r in results])
    avg_distortion = np.mean([r["contradiction_distortion"] for r in results])

    print("=" * 80)
    print("DIAGNOSTIC SUMMARY:")
    print(f"  Total Probes Evaluated: {len(results)}")
    print(f"  Distorted Probes (Myth > Fact): {distorted_count} / {len(results)} ({distorted_count/len(results)*100:.1f}%)")
    print(f"  Mean cos(A, Myth): {avg_myth_cos:.4f}")
    print(f"  Mean cos(A, Fact): {avg_fact_cos:.4f}")
    print(f"  Mean Contradiction Distortion: {avg_distortion:+.4f}")
    print("=" * 80)

    # Save Results
    out_file = os.path.join(OUTPUT_DIR, "thought_wilderness_contradiction_diagnostics.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total": len(results),
                "distorted_count": distorted_count,
                "distorted_ratio": distorted_count / len(results),
                "avg_myth_cos": float(avg_myth_cos),
                "avg_fact_cos": float(avg_fact_cos),
                "avg_distortion": float(avg_distortion)
            },
            "records": results
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved diagnostic report to: {out_file}")

    return results


if __name__ == "__main__":
    run_diagnostics()
