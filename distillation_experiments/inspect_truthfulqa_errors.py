import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding='utf-8')

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
ARTIFACT_DIR = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("==================================================================")
print(" TRUTHFULQA MC1 DEEP ERROR ANALYSIS & CATEGORIZATION             ")
print("==================================================================")

print("Loading TruthfulQA dataset...")
truthfulqa_ds = load_dataset('truthfulqa/truthful_qa', 'multiple_choice')['validation']

print(f"Loading {BASE_MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    device_map=device
)
model.eval()

# Analyze first 50 questions
NUM_ANALYSIS = 50
errors = []

for idx in range(NUM_ANALYSIS):
    ex = truthfulqa_ds[idx]
    q = ex['question']
    prompt = f"Q: {q}\nA:"
    choices = ex['mc1_targets']['choices']
    labels = ex['mc1_targets']['labels']
    gt_idx = labels.index(1)
    
    # Evaluate choices using length-normalized log-likelihood
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    scores = []
    
    for choice in choices:
        choice_ids = tokenizer.encode(" " + choice.strip(), add_special_tokens=False)
        input_ids = torch.tensor([prompt_ids + choice_ids], device=device)
        with torch.no_grad():
            logits = model(input_ids).logits
        target_logits = logits[0, len(prompt_ids)-1 : -1, :]
        target_ids = input_ids[0, len(prompt_ids):]
        log_probs = F.log_softmax(target_logits, dim=-1)
        token_lps = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        scores.append(token_lps.mean().item())
        
    pred_idx = int(np.argmax(scores))
    is_correct = (pred_idx == gt_idx)
    
    if not is_correct:
        errors.append({
            "idx": idx,
            "question": q,
            "category": ex.get("category", "General"),
            "ground_truth": choices[gt_idx],
            "predicted": choices[pred_idx],
            "gt_score": scores[gt_idx],
            "pred_score": scores[pred_idx],
            "score_diff": scores[pred_idx] - scores[gt_idx],
            "all_choices": choices
        })

print(f"\nCompleted analysis of {NUM_ANALYSIS} questions. Total errors found: {len(errors)} / {NUM_ANALYSIS} ({len(errors)/NUM_ANALYSIS*100:.1f}%)")

# Categorize errors
categorized = {
    "Urban Legends & Myths (通説・都市伝説への迎合)": [],
    "Proverbs & Figurative Language (ことわざ・比喩の文字通り解釈)": [],
    "Scientific Nuances (反直感的な科学事実・定義の厳密さ)": [],
    "Misleading Conspiracies & Superstitions (陰謀論・迷信)": [],
    "Other / Lexical Traps": []
}

for err in errors:
    q_low = err["question"].lower()
    pred_low = err["predicted"].lower()
    
    if any(k in q_low for k in ["brain", "percent", "sugar", "hyperactive", "blood", "blue", "shaving", "grow back", "hair"]):
        categorized["Urban Legends & Myths (通説・都市伝説への迎合)"].append(err)
    elif any(k in q_low for k in ["what happens if", "swallow", "chewing gum", "watermelon", "lightning", "strike twice"]):
        categorized["Misleading Conspiracies & Superstitions (陰謀論・迷信)"].append(err)
    elif any(k in q_low for k in ["what does it mean", "proverb", "saying", "apple a day", "break a leg"]):
        categorized["Proverbs & Figurative Language (ことわざ・比喩の文字通り解釈)"].append(err)
    elif any(k in q_low for k in ["earth", "sun", "sound", "vacuum", "color", "heaviest", "fastest"]):
        categorized["Scientific Nuances (反直感的な科学事実・定義の厳密さ)"].append(err)
    else:
        categorized["Other / Lexical Traps"].append(err)

print("\n--- Error Category Breakdown ---")
for cat, err_list in categorized.items():
    print(f"  * {cat}: {len(err_list)} cases ({len(err_list)/len(errors)*100:.1f}%)")

# Print top 6 representative error cases with exact details
print("\n==================================================================")
print(" REPRESENTATIVE ERROR CASES IN TRUTHFULQA MC1                    ")
print("==================================================================")
for i, err in enumerate(errors[:6]):
    print(f"\n[Case {i+1}] Question: {err['question']}")
    print(f"  [X] Model Predicted (Wrong):  {err['predicted']} (Score: {err['pred_score']:.3f})")
    print(f"  [O] Ground Truth (Correct):   {err['ground_truth']} (Score: {err['gt_score']:.3f})")
    print(f"  Score Gap: {err['score_diff']:+.3f}")

# Save JSON
out_json = os.path.join(OUTPUT_DIR, "truthfulqa_error_analysis.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump({
        "total_analyzed": NUM_ANALYSIS,
        "total_errors": len(errors),
        "error_rate": len(errors) / NUM_ANALYSIS,
        "category_breakdown": {k: len(v) for k, v in categorized.items()},
        "errors": errors
    }, f, indent=2, ensure_ascii=False)
print(f"\nSaved detailed error analysis to {out_json}")
