import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
ARTIFACT_DIR = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5"

print("==================================================================")
print(" CONCEPT INFLUENCE (変更能) & DISAMBIGUATION EXPERIMENT           ")
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

def get_next_token_log_probs(prompt: str):
    """Returns log-softmax probabilities for the next token given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
        log_probs = F.log_softmax(logits, dim=-1)
    return log_probs

def get_token_id(word: str):
    """Encodes a word with leading space (typical for continuation)."""
    tokens = tokenizer.encode(" " + word.strip(), add_special_tokens=False)
    return tokens[0] if len(tokens) > 0 else None

def compute_influence_shift(base_prompt: str, modifier_prompt: str, target_words: list):
    """
    Computes:
      1. Directional Shift for specific target concepts:
         Delta = log P(Y | base + modifier) - log P(Y | base)
      2. Global KL Divergence:
         KL(P(comb) || P(base))
    """
    lp_base = get_next_token_log_probs(base_prompt)
    combined_prompt = f"{base_prompt} {modifier_prompt}".strip()
    lp_comb = get_next_token_log_probs(combined_prompt)
    
    # Global KL Divergence: sum P_comb * (lp_comb - lp_base)
    p_comb = torch.exp(lp_comb)
    kl_div = torch.sum(p_comb * (lp_comb - lp_base)).item()
    
    shifts = {}
    for word in target_words:
        tid = get_token_id(word)
        if tid is not None:
            delta = lp_comb[tid].item() - lp_base[tid].item()
            prob_before = torch.exp(lp_base[tid]).item()
            prob_after = torch.exp(lp_comb[tid]).item()
            shifts[word] = {
                "delta": delta,
                "prob_before": prob_before,
                "prob_after": prob_after,
                "ratio": prob_after / (prob_before + 1e-12)
            }
        else:
            shifts[word] = {"delta": 0.0, "ratio": 1.0}
            
    return shifts, kl_div

# ----------------------------------------------------------------------
# Test Case 1: Apple (Fruit vs Smartphone Company)
# ----------------------------------------------------------------------
case1_base = "The Apple is"
case1_modifiers = {
    "No modifier (Base)": "",
    "+ Fruit context": "a fresh botanical fruit that is",
    "+ Smartphone context": "a leading technology smartphone that is",
    "+ Red modifier": "bright red and"
}
case1_targets = [
    "red", "sweet", "juicy", "delicious", "healthy",
    "smart", "digital", "expensive", "electric", "portable"
]

print("\n--- Running Case 1: 'Apple' Disambiguation ---")
case1_results = {}
for mod_name, mod_text in case1_modifiers.items():
    if mod_name == "No modifier (Base)":
        continue
    shifts, kl = compute_influence_shift(case1_base, mod_text, case1_targets)
    case1_results[mod_name] = shifts
    print(f"\nModifier: {mod_name} (Global KL Influence: {kl:.4f})")
    for tw, data in shifts.items():
        print(f"  Target: {tw:12s} | Delta: {data['delta']:+6.2f} | Multiplier: x{data['ratio']:.2f}")

# ----------------------------------------------------------------------
# Test Case 2: Bank (River Bank vs Financial Bank)
# ----------------------------------------------------------------------
case2_base = "The bank was"
case2_modifiers = {
    "+ River context": "situated along the flowing river water where",
    "+ Financial context": "a commercial financial institution where"
}
case2_targets = [
    "muddy", "grassy", "flowing", "wet", "natural",
    "wealthy", "secure", "profitable", "closed", "regulated"
]

print("\n--- Running Case 2: 'Bank' Disambiguation ---")
case2_results = {}
for mod_name, mod_text in case2_modifiers.items():
    shifts, kl = compute_influence_shift(case2_base, mod_text, case2_targets)
    case2_results[mod_name] = shifts
    print(f"\nModifier: {mod_name} (Global KL Influence: {kl:.4f})")
    for tw, data in shifts.items():
        print(f"  Target: {tw:12s} | Delta: {data['delta']:+6.2f} | Multiplier: x{data['ratio']:.2f}")

# ----------------------------------------------------------------------
# Test Case 3: Python (Snake vs Programming Language)
# ----------------------------------------------------------------------
case3_base = "The Python was"
case3_modifiers = {
    "+ Biology context": "a large tropical reptile in the wild that was",
    "+ Programming context": "an open-source software programming language that was"
}
case3_targets = [
    "crawling", "dangerous", "coiled", "sleeping", "hunting",
    "compiled", "interpreted", "fast", "flexible", "scripted"
]

print("\n--- Running Case 3: 'Python' Disambiguation ---")
case3_results = {}
for mod_name, mod_text in case3_modifiers.items():
    shifts, kl = compute_influence_shift(case3_base, mod_text, case3_targets)
    case3_results[mod_name] = shifts
    print(f"\nModifier: {mod_name} (Global KL Influence: {kl:.4f})")
    for tw, data in shifts.items():
        print(f"  Target: {tw:12s} | Delta: {data['delta']:+6.2f} | Multiplier: x{data['ratio']:.2f}")

def draw_heatmap(ax, matrix, x_labels, y_labels, title):
    im = ax.imshow(matrix, cmap="coolwarm", aspect="auto", vmin=-5, vmax=5)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=9, fontweight='bold')
    ax.set_yticklabels(y_labels, fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    
    # Loop over data dimensions and create text annotations.
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            val = matrix[i, j]
            text = ax.text(j, i, f"{val:+.1f}",
                           ha="center", va="center", color="white" if abs(val) > 2.5 else "black",
                           fontweight="bold", fontsize=9)
    return im

# ----------------------------------------------------------------------
# Plot Heatmaps & Comparisons
# ----------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(11, 13))

# Heatmap 1: Apple
mod_labels_1 = list(case1_results.keys())
matrix_1 = np.array([[case1_results[m][t]['delta'] for t in case1_targets] for m in mod_labels_1])
im1 = draw_heatmap(axes[0], matrix_1, case1_targets, mod_labels_1, "Case 1: 'Apple' Semantic Influence (変更能) Matrix")

# Heatmap 2: Bank
mod_labels_2 = list(case2_results.keys())
matrix_2 = np.array([[case2_results[m][t]['delta'] for t in case2_targets] for m in mod_labels_2])
im2 = draw_heatmap(axes[1], matrix_2, case2_targets, mod_labels_2, "Case 2: 'Bank' Semantic Influence (変更能) Matrix")

# Heatmap 3: Python
mod_labels_3 = list(case3_results.keys())
matrix_3 = np.array([[case3_results[m][t]['delta'] for t in case3_targets] for m in mod_labels_3])
im3 = draw_heatmap(axes[2], matrix_3, case3_targets, mod_labels_3, "Case 3: 'Python' Semantic Influence (変更能) Matrix")

fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im3, cax=cbar_ax)
cbar.set_label('Semantic Log-Odds Delta (変更能: +促進 / -抑制)', fontsize=11, fontweight='bold')

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "concept_influence_disambiguation_heatmap.png")
plt.savefig(plot_path, dpi=300)
print(f"\nSaved heatmap to {plot_path}")

artifact_plot_path = os.path.join(ARTIFACT_DIR, "concept_influence_disambiguation_heatmap.png")
import shutil
shutil.copyfile(plot_path, artifact_plot_path)
print(f"Copied to artifact dir: {artifact_plot_path}")

# Save full results json
with open(os.path.join(OUTPUT_DIR, "concept_influence_results.json"), "w", encoding="utf-8") as f:
    json.dump({
        "case1_apple": case1_results,
        "case2_bank": case2_results,
        "case3_python": case3_results
    }, f, indent=2)

print("\n=== Experiment Successfully Completed! ===")
