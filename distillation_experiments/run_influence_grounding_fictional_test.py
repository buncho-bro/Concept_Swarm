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
print(" INTRINSIC INFLUENCE GROUNDING TEST (変更能ネットワーク接続度)    ")
print(" Detecting In-Task Ephemeral Entities via Maximum Influence Energy")
print(" ZERO DICTIONARIES - PURELY BASED ON MODEL'S INTERNAL CONNECTIONS ")
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

def measure_word_influence_grounding(word: str):
    """
    User's Idea: '変更能に当てはめたときにどこにもない単語があったらそれはその場でのみ使われる固有名詞'
    Measures how strongly this word is grounded in the model's pre-trained influence network.
    A grounded real-world word has strong positive and negative influence spikes (|Delta| > 5.0).
    An ungrounded/fictional entity has weak, flat influence (|Delta| is small across vocabulary).
    """
    prompt = f"The {word} is"
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**enc).logits[0, -1, :]
        lps = F.log_softmax(logits, dim=-1)
        
    delta = lps - neutral_log_probs
    # Measure maximum absolute influence across top-100 vocabulary shifts
    top_pos, _ = torch.topk(delta, k=50)
    top_neg, _ = torch.topk(-delta, k=50)
    
    max_pos = top_pos[0].item()
    max_neg = top_neg[0].item()
    mean_top_influence = (top_pos.mean().item() + top_neg.mean().item()) / 2.0
    
    # Grounding Energy Score
    grounding_score = mean_top_influence
    is_ephemeral_entity = (grounding_score < 4.0) # Threshold for completely disconnected token
    
    return {
        "word": word,
        "max_positive": max_pos,
        "max_negative": -max_neg,
        "grounding_score": grounding_score,
        "is_ephemeral_entity": is_ephemeral_entity
    }

# ----------------------------------------------------------------------
# 1. Test Grounding of Real Words vs ProntoQA Fictional Entities
# ----------------------------------------------------------------------
test_words = [
    # Real-world common entities
    "apple", "fruit", "stomach", "gum", "ocean", "penguin", "metal",
    # ProntoQA imaginary entities
    "yumpus", "dumpus", "wumpus", "impus", "tumpus", "zumpus", "jompus", "numpus", "vumpus", "rompus"
]

print("\n--- Measuring Influence Grounding of Individual Words ---")
word_stats = []
for w in test_words:
    res = measure_word_influence_grounding(w)
    status = "EPHEMERAL / UNGROUNDED" if res["is_ephemeral_entity"] else "GROUNDED (Real)"
    print(f"  Word: {w:10s} | Grounding Score: {res['grounding_score']:.2f} (Max +{res['max_positive']:.1f} / {res['max_negative']:.1f}) -> {status}")
    word_stats.append(res)

# ----------------------------------------------------------------------
# 2. Document-Level Ephemeral Ratio (Virtual Degree via Ungrounded Entities)
# ----------------------------------------------------------------------
def estimate_virtual_degree_via_ungrounded_count(text: str):
    raw_words = [w.strip(".,?!:;\"'()[]{}").lower() for w in text.split()]
    # Simple length filter, no dictionaries
    candidates = [w for w in raw_words if len(w) > 2 and w.isalpha() and w not in ["the", "are", "each", "every", "not", "was", "for", "and", "that", "this"]]
    unique_candidates = list(dict.fromkeys(candidates))
    
    if not unique_candidates:
        return 0.0, 0, 0
        
    ephemeral_count = 0
    grounding_scores = []
    
    for w in unique_candidates[:15]: # Sample up to 15 key words
        stats = measure_word_influence_grounding(w)
        grounding_scores.append(stats["grounding_score"])
        if stats["is_ephemeral_entity"]:
            ephemeral_count += 1
            
    ephemeral_ratio = ephemeral_count / len(grounding_scores)
    # Virtual degree is directly proportional to ungrounded entity ratio
    virtual_degree = float(np.clip(ephemeral_ratio * 1.5, 0.0, 1.0))
    return virtual_degree, ephemeral_count, len(grounding_scores)

print("\n--- Evaluating Entire Texts via Ephemeral Entity Ratio ---")
benchmark_docs = [
    {
        "name": "Everyday Apple Sentence",
        "text": "The fresh red apple is a delicious and healthy fruit that grows on orchard trees."
    },
    {
        "name": "Gum Medical Myth (User's Example)",
        "text": "Swallowed chewing gum stays stuck in human stomach for seven years without digestion."
    },
    {
        "name": "Bowling Ball in Ocean",
        "text": "Heavy metal bowling balls sink quickly to deep ocean water floor due to gravity."
    },
    {
        "name": "ProntoQA Q0 (Max is sour)",
        "text": "Jompuses are not shy. Jompuses are yumpuses. Each yumpus is aggressive. Each yumpus is a dumpus. Dumpuses are not wooden. Dumpuses are wumpuses. Wumpuses are red. Max is a yumpus."
    },
    {
        "name": "ProntoQA Q1 (Stella is bright)",
        "text": "Every tumpus is not angry. Tumpuses are rompuses. Every numpus is not bright. Rompuses are yumpuses. Yumpuses are zumpuses. Zumpuses are impuses. Each impus is a dumpus. Stella is a yumpus."
    }
]

doc_results = []
for doc in benchmark_docs:
    v_deg, eph_cnt, total_cnt = estimate_virtual_degree_via_ungrounded_count(doc["text"])
    routed = "Engine A (Common Sense / Reality)" if v_deg < 0.5 else "Engine B (Pure Deductive Sandbox)"
    print(f"\n[{doc['name']}]")
    print(f"  Ungrounded Ephemeral Entities: {eph_cnt} / {total_cnt} ({eph_cnt/total_cnt*100:.1f}%)")
    print(f"  --> Virtual Degree λ:         {v_deg:.3f} | Routing: {routed}")
    doc_results.append({
        "name": doc["name"],
        "ephemeral_ratio": eph_cnt / total_cnt,
        "virtual_degree": v_deg,
        "routed": routed
    })

# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Individual Word Grounding
w_names = [s["word"] for s in word_stats]
w_scores = [s["grounding_score"] for s in word_stats]
w_colors = ['#27ae60' if not s["is_ephemeral_entity"] else '#c0392b' for s in word_stats]
ax1.barh(w_names, w_scores, color=w_colors, alpha=0.85)
ax1.axvline(4.0, color='black', linestyle='--', label='Ungrounded Threshold (4.0)')
ax1.set_xlabel('Influence Grounding Score (Higher = Deeply Rooted in Reality)', fontsize=10, fontweight='bold')
ax1.set_title('Word-Level Grounding: Real Words vs ProntoQA Entities\n(No Dictionaries Used)', fontsize=11, fontweight='bold')
ax1.grid(axis='x', linestyle='--', alpha=0.5)
ax1.legend(loc='lower right')

# Document Virtual Degree
d_names = [d["name"] for d in doc_results]
d_lambdas = [d["virtual_degree"] for d in doc_results]
d_colors = ['#2ecc71' if l < 0.5 else '#9b59b6' for l in d_lambdas]
ax2.barh(d_names, d_lambdas, color=d_colors, alpha=0.85)
ax2.axvline(0.5, color='red', linestyle='--', label='Routing Boundary (0.5)')
ax2.set_xlabel('Document Virtual Degree λ', fontsize=10, fontweight='bold')
ax2.set_title("User's Formulation: Virtual Degree via Ephemeral Entity Ratio", fontsize=11, fontweight='bold')
ax2.set_xlim(0, 1.05)
ax2.grid(axis='x', linestyle='--', alpha=0.5)
ax2.legend(loc='lower right')

for b in ax2.patches:
    w = b.get_width()
    ax2.annotate(f'{w:.2f}', xy=(w, b.get_y() + b.get_height()/2), xytext=(5, 0), textcoords="offset points", va='center', fontweight='bold')

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "influence_grounding_fictional_detection.png")
plt.savefig(plot_path, dpi=300)
print(f"\nSaved plot to {plot_path}")

artifact_plot_path = os.path.join(ARTIFACT_DIR, "influence_grounding_fictional_detection.png")
import shutil
shutil.copyfile(plot_path, artifact_plot_path)
print(f"Copied to artifact dir: {artifact_plot_path}")

print("\n=== Experiment Successfully Completed! ===")
