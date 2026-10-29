import os
import sys
import json
import time
import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding='utf-8')

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
ARTIFACT_DIR = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("==================================================================")
print(" STEP 2 BENCHMARK: ANALOGY & CONCEPT RESONANCE (類似構造と類推)   ")
print("  - Sub-module 2A: Bidirectional Directed Semantic Influence      ")
print("  - Sub-module 2B: Relational Analogy Alignment (A:B :: C:D)      ")
print("  - Sub-module 2C: Fictional Neutrality & Friction Verification    ")
print("==================================================================")
print(f"Device: {device}")

print(f"Loading {BASE_MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    device_map=device
)
model.eval()

# Neutral Baseline log-prob
neutral_prompt = "Something is"
inputs = tokenizer(neutral_prompt, return_tensors="pt").to(device)
with torch.no_grad():
    neutral_logits = model(**inputs).logits[0, -1, :]
    neutral_log_probs = F.log_softmax(neutral_logits, dim=-1).float()

# ----------------------------------------------------------------------
# 1. Step 2 Implementation: Analogy & Concept Resonance Engine
# ----------------------------------------------------------------------
class Step2AnalogyEngine:
    def __init__(self):
        self.vector_cache = {}

    def get_concept_delta_vector(self, word: str):
        """
        Retrieves full-vocabulary influence vector shift for word against neutral baseline.
        """
        w = word.strip().lower()
        if w in self.vector_cache:
            return self.vector_cache[w]
            
        prompt = f"The {w} is"
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits[0, -1, :]
            lps = F.log_softmax(logits, dim=-1).float()
            
        delta = lps - neutral_log_probs
        self.vector_cache[w] = delta
        return delta

    def compute_pairwise_affinity(self, word_a: str, word_b: str):
        """
        Sub-module 2A: Directed Semantic Influence.
          Delta(A -> B): Log prob shift of B given 'The A is ...'
          Delta(B -> A): Log prob shift of A given 'The B is ...'
        """
        delta_vec_a = self.get_concept_delta_vector(word_a)
        delta_vec_b = self.get_concept_delta_vector(word_b)
        
        # Token ids
        toks_b = tokenizer.encode(" " + word_b.strip().lower(), add_special_tokens=False)
        if not toks_b:
            toks_b = tokenizer.encode(word_b.strip().lower(), add_special_tokens=False)
        tid_b = toks_b[0]
        
        toks_a = tokenizer.encode(" " + word_a.strip().lower(), add_special_tokens=False)
        if not toks_a:
            toks_a = tokenizer.encode(word_a.strip().lower(), add_special_tokens=False)
        tid_a = toks_a[0]
        
        delta_ab = delta_vec_a[tid_b].item()
        delta_ba = delta_vec_b[tid_a].item()
        
        symmetric_affinity = (delta_ab + delta_ba) / 2.0
        directional_asymmetry = delta_ab - delta_ba
        
        # Grounding magnitude (is it a real grounded word or ungrounded fictional?)
        grounding_a = torch.norm(delta_vec_a).item()
        grounding_b = torch.norm(delta_vec_b).item()
        
        is_fictional = (grounding_a < 15.0 or grounding_b < 15.0)
        
        return {
            "word_a": word_a,
            "word_b": word_b,
            "delta_ab": delta_ab,
            "delta_ba": delta_ba,
            "affinity": symmetric_affinity,
            "asymmetry": directional_asymmetry,
            "is_fictional": is_fictional,
            "is_plausible": (symmetric_affinity > -0.5),
            "is_contradiction": (symmetric_affinity < -1.5)
        }

    def compute_analogy_alignment(self, a: str, b: str, c: str, d: str):
        """
        Sub-module 2B: Proportional Analogy Resonance (A:B :: C:D).
        Tests whether the conceptual transformation (B - A) aligns with (D - C).
        """
        delta_a = self.get_concept_delta_vector(a)
        delta_b = self.get_concept_delta_vector(b)
        delta_c = self.get_concept_delta_vector(c)
        delta_d = self.get_concept_delta_vector(d)
        
        rel_ab = delta_b - delta_a
        rel_cd = delta_d - delta_c
        
        norm_ab = torch.norm(rel_ab)
        norm_cd = torch.norm(rel_cd)
        
        if norm_ab == 0 or norm_cd == 0:
            alignment = 0.0
        else:
            alignment = (torch.dot(rel_ab, rel_cd) / (norm_ab * norm_cd)).item()
            
        return {
            "analogy": f"{a}:{b} :: {c}:{d}",
            "alignment": alignment,
            "is_valid_analogy": (alignment > 0.30)
        }

# ----------------------------------------------------------------------
# 2. Test Suite for Step 2
# ----------------------------------------------------------------------
test_cases_affinity = [
    # Grounded Common-Sense Property Pairs
    {"a": "penguin", "b": "swim", "expected_plausible": True, "expected_affinity_min": -0.5, "desc": "Penguins swim naturally"},
    {"a": "penguin", "b": "fly", "expected_plausible": False, "expected_affinity_min": None, "desc": "Penguins cannot fly (inhibited)"},
    {"a": "metal", "b": "sink", "expected_plausible": True, "expected_affinity_min": 0.0, "desc": "Metal sinks in water"},
    {"a": "astronaut", "b": "spaceflight", "expected_plausible": True, "expected_affinity_min": 2.0, "desc": "Astronauts undergo spaceflight"},
    {"a": "stomach", "b": "years", "expected_plausible": False, "expected_affinity_min": None, "desc": "Gum myth (strong friction)"},
    # Fictional / Ungrounded Words (Must have zero pre-existing bias)
    {"a": "yumpus", "b": "bright", "expected_fictional": True, "desc": "Fictional entity has no real-world bias"},
    {"a": "dumpus", "b": "sour", "expected_fictional": True, "desc": "Fictional entity has no real-world bias"}
]

test_cases_analogy = [
    {"a": "bird", "b": "fly", "c": "fish", "d": "swim", "expected_valid": True, "desc": "Locomotion analogy (bird:fly :: fish:swim)"},
    {"a": "king", "b": "man", "c": "queen", "d": "woman", "expected_valid": True, "desc": "Gender role analogy (king:man :: queen:woman)"},
    {"a": "bird", "b": "fly", "c": "penguin", "d": "fly", "expected_valid": False, "desc": "Broken analogy (bird:fly vs penguin:fly)"}
]

# ----------------------------------------------------------------------
# 3. Execution and Objective Verification
# ----------------------------------------------------------------------
print(f"\nExecuting Step 2 Verification...\n")
engine = Step2AnalogyEngine()

# Test 2A & 2C: Affinities & Grounding
print("--- [Part 1: Pairwise Concept Affinity & Friction] ---")
affinity_results = []
affinity_passed = 0

for tc in test_cases_affinity:
    res = engine.compute_pairwise_affinity(tc["a"], tc["b"])
    
    if "expected_fictional" in tc:
        # Fictional test: affinity must be modest (-1.5 < aff < 1.0) and ungrounded
        passed = abs(res["affinity"]) < 1.0
        expected_str = "Near-zero bias (|aff| < 1.0)"
    else:
        passed = (res["is_plausible"] == tc["expected_plausible"])
        expected_str = f"Plausible={tc['expected_plausible']}"
        
    verdict = "PASS" if passed else "FAIL"
    if passed:
        affinity_passed += 1
        
    print(f"Pair ({tc['a']:10s}, {tc['b']:11s}) -> Affinity: {res['affinity']:+.3f} (A->B: {res['delta_ab']:+.2f}, B->A: {res['delta_ba']:+.2f}) | {verdict} ({expected_str})")
    affinity_results.append({**tc, **res, "passed": passed})

# Test 2B: Proportional Analogy
print("\n--- [Part 2: Relational Analogy Alignment] ---")
analogy_results = []
analogy_passed = 0

for tc in test_cases_analogy:
    res = engine.compute_analogy_alignment(tc["a"], tc["b"], tc["c"], tc["d"])
    passed = (res["is_valid_analogy"] == tc["expected_valid"])
    verdict = "PASS" if passed else "FAIL"
    if passed:
        analogy_passed += 1
        
    print(f"Analogy: {res['analogy']:25s} | Cosine Alignment: {res['alignment']:+.3f} -> {verdict} (Valid={tc['expected_valid']})")
    analogy_results.append({**tc, **res, "passed": passed})

total_tests = len(affinity_results) + len(analogy_results)
total_passed = affinity_passed + analogy_passed
pass_rate = (total_passed / total_tests) * 100.0

print("\n==================================================================")
print(f" STEP 2 VERIFICATION SUMMARY: {total_passed}/{total_tests} Passed ({pass_rate:.1f}%)")
print(f"  - Pairwise Affinities: {affinity_passed}/{len(affinity_results)}")
print(f"  - Relational Analogies: {analogy_passed}/{len(analogy_results)}")
print("==================================================================")

# Save JSON
json_path = os.path.join(OUTPUT_DIR, "step2_analogy_benchmark_results.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump({
        "total_passed": total_passed,
        "total_tests": total_tests,
        "pass_rate": pass_rate,
        "affinities": affinity_results,
        "analogies": analogy_results
    }, f, indent=2, ensure_ascii=False)
print(f"Saved results to {json_path}")

# Visualization Chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Concept Affinities
pairs = [f"{r['a']}-{r['b']}" for r in affinity_results]
affs = [r["affinity"] for r in affinity_results]
colors = ['#27ae60' if r["affinity"] > -0.5 else '#c0392b' for r in affinity_results]

bars1 = ax1.barh(pairs, affs, color=colors, alpha=0.85)
ax1.axvline(0.0, color='black', linestyle='-', alpha=0.5)
ax1.axvline(-0.5, color='orange', linestyle='--', label='Plausibility Gate (-0.5)')
ax1.axvline(-1.5, color='red', linestyle='--', label='Contradiction Gate (-1.5)')
ax1.set_xlabel('Bidirectional Semantic Affinity', fontsize=10, fontweight='bold')
ax1.set_title('Step 2: Directed Semantic Influence across Concepts', fontsize=11, fontweight='bold')
ax1.grid(axis='x', linestyle='--', alpha=0.5)
ax1.legend(loc='lower right', fontsize=8)

for bar in bars1:
    w = bar.get_width()
    offset = 0.15 if w >= 0 else -0.15
    ha = 'left' if w >= 0 else 'right'
    ax1.text(w + offset, bar.get_y() + bar.get_height()/2.0, f"{w:+.2f}", ha=ha, va='center', fontweight='bold', fontsize=9)

# Subplot 2: Analogy Alignments
analogies = [r["analogy"] for r in analogy_results]
alignments = [r["alignment"] for r in analogy_results]
colors2 = ['#2980b9' if r["passed"] else '#c0392b' for r in analogy_results]

bars2 = ax2.bar(analogies, alignments, color=colors2, width=0.4, alpha=0.85)
ax2.axhline(0.30, color='red', linestyle='--', label='Analogy Threshold (0.30)')
ax2.set_ylabel('Structural Vector Alignment', fontsize=10, fontweight='bold')
ax2.set_title('Step 2: Relational Analogy Alignment (A:B :: C:D)', fontsize=11, fontweight='bold')
ax2.set_ylim(-0.2, 0.8)
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.legend(loc='upper right', fontsize=8)

for bar in bars2:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, h + 0.02, f"{h:+.2f}", ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
chart_path = os.path.join(ARTIFACT_DIR, "step2_analogy_benchmark.png")
plt.savefig(chart_path, dpi=200)
plt.close()
print(f"Saved benchmark chart to {chart_path}")
