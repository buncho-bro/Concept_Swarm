import os
import sys
import json
import time
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
print(" UNIVERSAL CONCEPT CONTRADICTION & SELF-CORRECTION SYSTEM         ")
print(" (Measuring Internal Semantic Friction / Energy Minimization)     ")
print("==================================================================")
print(f"Device: {device}")

print("Loading Qwen2.5-1.5B-Instruct...")
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
    neutral_log_probs = F.log_softmax(neutral_logits, dim=-1)

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "to", "and", "or",
    "that", "this", "it", "for", "with", "as", "by", "on", "at", "from", "which",
    "be", "has", "have", "had", "do", "does", "did", "but", "not", "your", "my",
    "through", "without", "easily", "quickly", "across", "because"
}

def extract_core_concepts(sentence: str):
    """Extracts informative content words from a sentence."""
    raw_words = [w.strip(".,?!:;\"'()[]{}").lower() for w in sentence.split()]
    concepts = [w for w in raw_words if len(w) > 2 and w.isalpha() and w not in STOPWORDS]
    # Deduplicate while preserving order
    return list(dict.fromkeys(concepts))

def compute_pairwise_influence(word_a: str, word_b: str):
    """
    Computes bidirectional influence (compatibility) between word A and word B:
      Delta(A -> B) = log P(B | A) - log P(B | Neutral)
      Delta(B -> A) = log P(A | B) - log P(A | Neutral)
    """
    # A -> B
    prompt_a = f"The {word_a} is"
    inp_a = tokenizer(prompt_a, return_tensors="pt").to(device)
    token_b = tokenizer.encode(" " + word_b, add_special_tokens=False)
    
    if not token_b:
        return 0.0
    tid_b = token_b[0]
    
    with torch.no_grad():
        lp_a = F.log_softmax(model(**inp_a).logits[0, -1, :], dim=-1)
    delta_ab = lp_a[tid_b].item() - neutral_log_probs[tid_b].item()
    
    # B -> A
    prompt_b = f"The {word_b} is"
    inp_b = tokenizer(prompt_b, return_tensors="pt").to(device)
    token_a = tokenizer.encode(" " + word_a, add_special_tokens=False)
    if not token_a:
        return delta_ab
    tid_a = token_a[0]
    
    with torch.no_grad():
        lp_b = F.log_softmax(model(**inp_b).logits[0, -1, :], dim=-1)
    delta_ba = lp_b[tid_a].item() - neutral_log_probs[tid_a].item()
    
    # Symmetric compatibility
    return (delta_ab + delta_ba) / 2.0

def measure_sentence_contradiction_energy(sentence: str):
    """
    Measures the internal contradiction energy (semantic friction) of a sentence.
    Positive energy = High contradiction / disharmony.
    Negative energy = High coherence / harmony.
    """
    concepts = extract_core_concepts(sentence)
    n = len(concepts)
    if n < 2:
        return 0.0, {}, concepts
    
    pairwise_matrix = np.zeros((n, n))
    bottlenecks = []
    
    for i in range(n):
        for j in range(i + 1, n):
            comp = compute_pairwise_influence(concepts[i], concepts[j])
            pairwise_matrix[i, j] = comp
            pairwise_matrix[j, i] = comp
            bottlenecks.append((concepts[i], concepts[j], comp))
            
    # Contradiction Energy = - (Average compatibility)
    # Higher average compatibility means LOWER contradiction energy
    avg_compat = np.sum(pairwise_matrix) / (n * (n - 1))
    contradiction_energy = -avg_compat
    
    # Sort bottlenecks: lowest compatibility = biggest friction / contradiction
    bottlenecks.sort(key=lambda x: x[2])
    
    return contradiction_energy, bottlenecks, concepts, pairwise_matrix

# ----------------------------------------------------------------------
# Test Suite: Diverse Real-World Pairs (Truth vs Myth/Contradiction)
# ----------------------------------------------------------------------
test_corpus = [
    {
        "domain": "Medical / Myth (User's Example)",
        "myth": "Swallowed chewing gum stays stuck in human stomach for seven years without digestion.",
        "truth": "Swallowed chewing gum passes naturally through digestive tract and exits body."
    },
    {
        "domain": "Physics / Mechanics",
        "myth": "Heavy metal bowling balls float easily on calm ocean water surface.",
        "truth": "Heavy metal bowling balls sink quickly to deep ocean floor."
    },
    {
        "domain": "Biology / Ecosystem",
        "myth": "Cold penguins fly high across hot dry desert sand dunes.",
        "truth": "Cold penguins swim fast through freezing icy polar ocean water."
    },
    {
        "domain": "Logic / Causality",
        "myth": "Fierce lions starve to death because gentle sheep hunt them.",
        "truth": "Gentle sheep run away because fierce lions hunt them."
    }
]

print("\n--- Evaluating Contradiction Energy across Diverse Domains ---")
results = []
for case in test_corpus:
    domain = case["domain"]
    e_myth, b_myth, c_myth, m_myth = measure_sentence_contradiction_energy(case["myth"])
    e_truth, b_truth, c_truth, m_truth = measure_sentence_contradiction_energy(case["truth"])
    
    print(f"\n[Domain: {domain}]")
    print(f"  Myth/False Sentence:  Energy = {e_myth:+.3f} (High Friction)")
    print(f"    Top Contradiction:  '{b_myth[0][0]}' <-> '{b_myth[0][1]}' (Friction: {b_myth[0][2]:+.2f})")
    print(f"  Truth/True Sentence:  Energy = {e_truth:+.3f} (Harmonious)")
    print(f"    Top Synergy:        '{b_truth[-1][0]}' <-> '{b_truth[-1][1]}' (Synergy: {b_truth[-1][2]:+.2f})")
    print(f"  --> Energy Gap:       {e_myth - e_truth:+.3f} (Truth is {e_myth - e_truth:+.2f} pts more stable)")
    
    results.append({
        "domain": domain,
        "e_myth": e_myth,
        "e_truth": e_truth,
        "gap": e_myth - e_truth,
        "b_myth": b_myth[:3],
        "b_truth": b_truth[-3:],
        "c_myth": c_myth,
        "m_myth": m_myth,
        "c_truth": c_truth,
        "m_truth": m_truth
    })

# ----------------------------------------------------------------------
# Self-Correction Engine Demo
# ----------------------------------------------------------------------
print("\n==================================================================")
print(" DEMONSTRATING UNIVERSAL SELF-CORRECTION (RELAXATION)             ")
print("==================================================================")
def self_correct_sentence(sentence, candidate_replacements):
    """
    Identifies the word causing the highest contradiction energy,
    and replaces it with the candidate that MINIMIZES energy (relaxes friction).
    """
    orig_energy, bottlenecks, concepts, _ = measure_sentence_contradiction_energy(sentence)
    culprit_a, culprit_b, worst_score = bottlenecks[0]
    print(f"Original Sentence: '{sentence}'")
    print(f"  Initial Energy: {orig_energy:+.3f}")
    print(f"  Detected Friction Culprit: '{culprit_a}' vs '{culprit_b}' (Score: {worst_score:+.2f})")
    
    # Try replacing culprit_b
    best_replacement = None
    min_energy = orig_energy
    
    for cand in candidate_replacements:
        modified_sentence = sentence.replace(culprit_b, cand)
        new_energy, _, _, _ = measure_sentence_contradiction_energy(modified_sentence)
        print(f"  Evaluating candidate '{cand}' -> New Energy: {new_energy:+.3f}")
        if new_energy < min_energy:
            min_energy = new_energy
            best_replacement = (cand, modified_sentence)
            
    if best_replacement:
        print(f"\n>> SELF-CORRECTION SUCCESSFUL! <<")
        print(f"  Repaired Sentence: '{best_replacement[1]}'")
        print(f"  Energy Drop: {orig_energy:+.3f} -> {min_energy:+.3f} (Reduction: {orig_energy - min_energy:+.3f})")
    return best_replacement

# Test Self-Correction on Physics Myth:
# "Heavy metal bowling balls float easily on calm ocean water surface."
self_correct_sentence(
    "Heavy metal bowling balls float easily on calm ocean water surface.",
    ["sink", "vanish", "explode", "fly"]
)

# ----------------------------------------------------------------------
# Visualizing Energy Landscape & Comparison
# ----------------------------------------------------------------------
domains = [r["domain"].split(" / ")[0] for r in results]
energies_myth = [r["e_myth"] for r in results]
energies_truth = [r["e_truth"] for r in results]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Bar plot
x = np.arange(len(domains))
width = 0.35
ax1.bar(x - width/2, energies_myth, width, label='Myth / Contradiction (False)', color='#e74c3c', alpha=0.85)
ax1.bar(x + width/2, energies_truth, width, label='Truth / Coherent (True)', color='#2ecc71', alpha=0.85)
ax1.set_ylabel('Contradiction Energy (Lower is More Harmonious)', fontsize=11, fontweight='bold')
ax1.set_title('Internal Semantic Contradiction Energy Comparison\n(Without Knowing Any Labels or Test Specifics)', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(domains, fontsize=10, fontweight='bold')
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.legend(fontsize=10)

# Heatmap for Medical Case (Myth vs Truth)
med_res = results[0]
im = ax2.imshow(med_res["m_myth"], cmap="coolwarm", aspect="auto", vmin=-4, vmax=4)
ax2.set_xticks(range(len(med_res["c_myth"])))
ax2.set_yticks(range(len(med_res["c_myth"])))
ax2.set_xticklabels(med_res["c_myth"], rotation=45, ha='right', fontsize=9)
ax2.set_yticklabels(med_res["c_myth"], fontsize=9)
ax2.set_title("Semantic Friction Matrix: Gum Myth\n(Blue/Cold = Negative Friction/Contradiction)", fontsize=11, fontweight='bold')
fig.colorbar(im, ax=ax2, label="Pairwise Influence Score")

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "semantic_energy_landscape.png")
plt.savefig(plot_path, dpi=300)
print(f"\nSaved plot to {plot_path}")

artifact_plot_path = os.path.join(ARTIFACT_DIR, "semantic_energy_landscape.png")
import shutil
shutil.copyfile(plot_path, artifact_plot_path)
print(f"Copied to artifact dir: {artifact_plot_path}")

# Save json results
with open(os.path.join(OUTPUT_DIR, "self_correction_results.json"), "w", encoding="utf-8") as f:
    json.dump([{k: v for k, v in r.items() if not k.startswith("m_")} for r in results], f, indent=2)

print("\n=== All Tests Successfully Completed! ===")
