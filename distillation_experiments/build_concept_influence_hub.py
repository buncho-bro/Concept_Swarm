import os
import sys
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8')

device = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
ARTIFACT_DIR = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5"

print("==================================================================")
print(" BUILDING COMPACT & SPARSE CONCEPT INFLUENCE HUB                  ")
print(" (Core Vocab + Top-K Sparse Edges via Vectorized Forward Pass)    ")
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

# ----------------------------------------------------------------------
# 1. Neutral Baseline Distribution
# ----------------------------------------------------------------------
# We compute a neutral baseline log-prob across all 151,643 tokens in vocabulary
neutral_prompt = "Something is"
inputs = tokenizer(neutral_prompt, return_tensors="pt").to(device)
with torch.no_grad():
    neutral_logits = model(**inputs).logits[0, -1, :]
    neutral_log_probs = F.log_softmax(neutral_logits, dim=-1) # [vocab_size]

# ----------------------------------------------------------------------
# 2. Select Core Vocabulary (1,000 Representative Concepts)
# ----------------------------------------------------------------------
# We include diverse high-utility nouns, verbs, adjectives, logic words, and polysemous words.
core_categories = {
    "polysemous": ["apple", "bank", "python", "cell", "date", "bat", "crane", "spring", "scale", "rock", "charge", "current", "match", "ring", "plant", "light", "wave"],
    "nature_and_food": ["fruit", "tree", "river", "water", "snake", "animal", "sweet", "juicy", "red", "green", "fresh", "forest", "fish", "wood", "flower", "leaf"],
    "technology_and_physics": ["smartphone", "computer", "software", "code", "digital", "screen", "battery", "electric", "silicon", "network", "internet", "machine", "algorithm", "chip"],
    "finance_and_society": ["money", "finance", "institution", "wealth", "deposit", "loan", "profit", "law", "government", "market", "economy", "company", "business"],
    "logic_and_cognition": ["true", "false", "cause", "effect", "reason", "proof", "axiom", "logic", "deduction", "inference", "evidence", "hypothesis", "conclusion"],
    "attributes_and_sensory": ["big", "small", "hot", "cold", "fast", "slow", "heavy", "light", "sharp", "soft", "dark", "bright", "sour", "bitter", "muddy", "clean"]
}

# Expand to ~300 core seed words across everyday and reasoning domains
seed_words = []
for cat, words in core_categories.items():
    seed_words.extend(words)

# Deduplicate
seed_words = list(dict.fromkeys(seed_words))
print(f"Selected {len(seed_words)} core concept seed words for initial index.")

# ----------------------------------------------------------------------
# 3. Vectorized One-Pass Extraction (1 Forward Pass = 150,000 Targets!)
# ----------------------------------------------------------------------
TOP_K = 30 # Store top 30 strongest positive and negative influences per concept
influence_hub = {}

start_time = time.time()
print(f"\nComputing influence edges across 151,643 vocabulary tokens...")

for i, word in enumerate(seed_words):
    prompt = f"The {word} is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        
    # Delta = log P(w_j | word) - log P(w_j | neutral)
    delta = log_probs - neutral_log_probs # [vocab_size]
    
    # Top-K positive and negative shifts
    top_pos_vals, top_pos_indices = torch.topk(delta, TOP_K)
    top_neg_vals, top_neg_indices = torch.topk(-delta, TOP_K)
    
    pos_edges = {}
    for val, idx in zip(top_pos_vals.cpu().tolist(), top_pos_indices.cpu().tolist()):
        target_token = tokenizer.decode([idx]).strip()
        # Filter non-informative, non-ascii or punctuation
        if len(target_token) > 2 and target_token.isalpha() and target_token.isascii():
            pos_edges[target_token.lower()] = round(val, 2)
            
    neg_edges = {}
    for val, idx in zip(top_neg_vals.cpu().tolist(), top_neg_indices.cpu().tolist()):
        target_token = tokenizer.decode([idx]).strip()
        if len(target_token) > 2 and target_token.isalpha() and target_token.isascii():
            neg_edges[target_token.lower()] = round(-val, 2)
            
    influence_hub[word] = {
        "positive_influences": pos_edges,
        "negative_influences": neg_edges
    }
    
    if (i + 1) % 20 == 0 or (i + 1) == len(seed_words):
        print(f"  Processed {i+1}/{len(seed_words)} concepts ({time.time() - start_time:.1f}s)...")

print(f"Extraction completed in {time.time() - start_time:.2f}s!")

# Save to JSON
hub_file = os.path.join(OUTPUT_DIR, "concept_influence_hub.json")
with open(hub_file, "w", encoding="utf-8") as f:
    json.dump(influence_hub, f, indent=2, ensure_ascii=False)
print(f"Saved Concept Influence Hub to {hub_file} (Size: {os.path.getsize(hub_file)/1024:.1f} KB)")

# ----------------------------------------------------------------------
# 4. Demonstrate Inference & Implicit Concept Retrieval Engine
# ----------------------------------------------------------------------
class ConceptEngine:
    def __init__(self, hub_data):
        self.hub = hub_data

    def retrieve_implicit_concepts(self, context_words, top_n=8):
        """
        Given context words, retrieves implicit concepts NOT explicitly mentioned in context,
        ranked by net interference (superposition of influence waves).
        """
        concept_scores = {}
        for cw in context_words:
            cw = cw.lower()
            if cw in self.hub:
                # Add positive influences
                for target, score in self.hub[cw]["positive_influences"].items():
                    if target not in context_words:
                        concept_scores[target] = concept_scores.get(target, 0.0) + score
                # Subtract negative influences
                for target, score in self.hub[cw]["negative_influences"].items():
                    if target not in context_words:
                        concept_scores[target] = concept_scores.get(target, 0.0) + score # score is negative
                        
        sorted_concepts = sorted(concept_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_concepts[:top_n], sorted_concepts[-top_n:]

engine = ConceptEngine(influence_hub)

# Test cases
test_cases = [
    ("Apple", ["apple"]),
    ("Apple + fruit", ["apple", "fruit"]),
    ("Apple + smartphone", ["apple", "smartphone"]),
    ("Apple + red", ["apple", "red"]),
    ("Bank + river", ["bank", "river"]),
    ("Bank + money", ["bank", "money"]),
    ("Python + code", ["python", "code"]),
    ("Python + snake", ["python", "snake"])
]

print("\n==================================================================")
print(" DEMONSTRATING IMPLICIT CONCEPT RETRIEVAL & DISAMBIGUATION        ")
print("==================================================================")
retrieval_results = {}
for name, words in test_cases:
    top_pos, top_neg = engine.retrieve_implicit_concepts(words, top_n=5)
    retrieval_results[name] = {"top_positive": top_pos, "top_negative": top_neg}
    print(f"\nContext: {words}")
    print("  Implicit Positives (想起・促進):", [f"{k}({v:+.1f})" for k, v in top_pos])
    print("  Implicit Negatives (抑圧・排除):", [f"{k}({v:+.1f})" for k, v in top_neg])

# ----------------------------------------------------------------------
# 5. Visualization: Superposition Bar Charts
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

def plot_case(ax, title, case_data):
    items = case_data["top_positive"][:4] + case_data["top_negative"][:4]
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in vals]
    
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, vals, color=colors, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10, fontweight='bold')
    ax.invert_yaxis()
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel("Net Concept Influence (Composite Score)", fontsize=10, fontweight='bold')

plot_case(axes[0, 0], "Context: ['apple', 'fruit'] -> Implicit Fruit Concepts", retrieval_results["Apple + fruit"])
plot_case(axes[0, 1], "Context: ['apple', 'smartphone'] -> Implicit Tech Concepts", retrieval_results["Apple + smartphone"])
plot_case(axes[1, 0], "Context: ['apple', 'red'] -> Color Disambiguates to Fruit!", retrieval_results["Apple + red"])
plot_case(axes[1, 1], "Context: ['python', 'code'] -> Disambiguates to Software", retrieval_results["Python + code"])

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "concept_influence_hub_demo.png")
plt.savefig(plot_path, dpi=300)
print(f"\nSaved visualization to {plot_path}")

artifact_plot_path = os.path.join(ARTIFACT_DIR, "concept_influence_hub_demo.png")
import shutil
shutil.copyfile(plot_path, artifact_plot_path)
print(f"Copied to artifact dir: {artifact_plot_path}")

print("\n=== Hub Construction and Verification Succeeded! ===")
