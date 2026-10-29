"""
Metric Inconsistency & Triangle Inequality Violation Detector
(思考原野における三つ組構成要素間の幾何学的自己矛盾検知スクリプト)
================================================================================
Evaluates Triangle Inequality Violations and Metric Inconsistency:
If d(A, B) is small (close) and d(A, C) is small (close),
then geometrically d(B, C) MUST be <= d(A, B) + d(A, C).
If d(B, C) is extremely large, this is a fatal INTERNAL SELF-CONTRADICTION of the metric space.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

print("==================================================================")
print(" GEOMETRIC SELF-CONTRADICTION DETECTOR (TRIANGLE INEQUALITY SCAN) ")
print(f" Model: {MODEL_ID} on {device}")
print("==================================================================")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None
)
model.eval()


def get_concept_vector(concept_name: str) -> torch.Tensor:
    prompt = f"The core concept of {concept_name} is"
    inp = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
        # Deep representation (Layer -2)
        h = out.hidden_states[-2][0].float()
        h_vec = h.mean(dim=0)
        return F.normalize(h_vec, p=2, dim=-1)


# Triplet test candidates: (A, B, C)
# Testing cases where A is related to B and A is related to C, but B and C are seemingly distant
TRIPLETS = [
    ("apple", "orange", "rose"),        # Apple-Orange (fruit), Apple-Rose (Rosaceae plant) -> Orange vs Rose?
    ("penguin", "bird", "submarine"),   # Penguin-Bird, Penguin-Submarine (underwater navigator) -> Bird vs Submarine?
    ("light", "wave", "particle"),      # Light-Wave, Light-Particle (wave-particle duality) -> Wave vs Particle?
    ("mercury", "planet", "liquid metal"), # Mercury-Planet, Mercury-Liquid Metal -> Planet vs Liquid Metal?
    ("bat", "mammal", "baseball"),      # Polysemy / homonym clash
    ("bank", "river", "money"),         # Polysemy / homonym clash
    ("virus", "biological entity", "computer program"), # Polysemy / metaphor clash
    ("energy", "mass", "motion")
]

print("\nTesting Triplets (A, B, C):")
print("-" * 80)

for a, b, c in TRIPLETS:
    v_a = get_concept_vector(a)
    v_b = get_concept_vector(b)
    v_c = get_concept_vector(c)

    # Angular distance on unit sphere: theta = arccos(cos_sim) in radians
    cos_ab = torch.dot(v_a, v_b).clamp(-1.0, 1.0).item()
    cos_ac = torch.dot(v_a, v_c).clamp(-1.0, 1.0).item()
    cos_bc = torch.dot(v_b, v_c).clamp(-1.0, 1.0).item()

    theta_ab = np.arccos(cos_ab)
    theta_ac = np.arccos(cos_ac)
    theta_bc = np.arccos(cos_bc)

    # Euclidean distance on unit sphere: dist = ||v1 - v2|| = sqrt(2 - 2*cos)
    dist_ab = np.sqrt(max(0.0, 2.0 - 2.0 * cos_ab))
    dist_ac = np.sqrt(max(0.0, 2.0 - 2.0 * cos_ac))
    dist_bc = np.sqrt(max(0.0, 2.0 - 2.0 * cos_bc))

    # Metric Triangle Inequality: d(B, C) <= d(A, B) + d(A, C)
    # Metric Tension / Excess = d(B, C) - (d(A, B) + d(A, C))
    excess = dist_bc - (dist_ab + dist_ac)
    angular_excess = theta_bc - (theta_ab + theta_ac)

    # Ratio of disparity
    ratio = dist_bc / (max(dist_ab, dist_ac) + 1e-8)

    print(f"Triplet: A='{a}' | B='{b}' | C='{c}'")
    print(f"  cos(A, B) = {cos_ab:+.4f} (dist: {dist_ab:.4f})")
    print(f"  cos(A, C) = {cos_ac:+.4f} (dist: {dist_ac:.4f})")
    print(f"  cos(B, C) = {cos_bc:+.4f} (dist: {dist_bc:.4f})")
    print(f"  Metric Sum [d(A,B) + d(A,C)]: {dist_ab + dist_ac:.4f}")
    print(f"  Actual Distance d(B, C):      {dist_bc:.4f}")
    print(f"  Disparity Ratio [d(B,C) / max(d(A,B), d(A,C))]: {ratio:.2f}x")
    
    if ratio > 1.5:
        print(f"  --> FATAL GEOMETRIC ANOMALY: B and C are disproportionately FAR ({ratio:.2f}x) despite both being close to A!")
    print("-" * 80)
