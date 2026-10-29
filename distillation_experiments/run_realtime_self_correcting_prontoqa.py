import os
import sys
import json
import time
import re
import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding='utf-8')

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
ARTIFACT_DIR = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("==================================================================")
print(" STEP-LEVEL REAL-TIME SELF-CORRECTION & BACKTRACKING REASONER     ")
print(" Monitoring Concept Friction at Each Deduction Step               ")
print("==================================================================")
print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

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
    "all", "each", "every", "statement", "following", "true", "false", "question",
    "step", "so", "therefore", "since", "we", "know", "can", "see"
}

def extract_core_concepts(text: str):
    words = [w.strip(".,?!:;\"'()[]{}").lower() for w in text.split()]
    concepts = [w for w in words if len(w) > 2 and w.isalpha() and w not in STOPWORDS]
    return list(dict.fromkeys(concepts))

def compute_step_friction(premise_concepts, step_concepts):
    """
    Measures negative contradiction (friction) between a newly proposed step
    and the established premises.
    Returns: maximum negative friction score (more negative = severe contradiction).
    """
    if not premise_concepts or not step_concepts:
        return 0.0
    
    # Check pairwise influence between premise words and step words
    min_compat = 0.0
    
    # We take top 5 premise concepts and top 3 step concepts
    p_tokens = [(c, tokenizer.encode(" " + c, add_special_tokens=False)) for c in premise_concepts[:6]]
    s_tokens = [(c, tokenizer.encode(" " + c, add_special_tokens=False)) for c in step_concepts[:4]]
    
    p_valid = [(c, t[0]) for c, t in p_tokens if t]
    s_valid = [(c, t[0]) for c, t in s_tokens if t]
    
    if not p_valid or not s_valid:
        return 0.0
        
    # Batch forward pass for step words
    prompts = [f"The {c} is" for c, _ in s_valid]
    batch_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        logits = model(**batch_inputs).logits[:, -1, :]
        lps = F.log_softmax(logits, dim=-1)
        deltas = lps - neutral_log_probs.unsqueeze(0) # [len(s_valid), vocab_size]
        
    compats = []
    for s_idx, (s_word, _) in enumerate(s_valid):
        for p_word, p_tid in p_valid:
            score = deltas[s_idx, p_tid].item()
            compats.append(score)
            
    if not compats:
        return 0.0
        
    # Friction is measured by the severity of the most negative contradiction
    worst_score = float(np.min(compats))
    return worst_score

def extract_conclusion_from_text(text: str):
    text_lower = text.lower()
    patterns = [
        r"therefore[,\s]+(?:the\s+statement\s+is\s+|it\s+is\s+)?(true|false)",
        r"the\s+statement\s+is\s+(true|false)",
        r"conclusion[:\s]+(?:the\s+statement\s+is\s+)?(true|false)",
        r"answer[:\s]+(true|false)",
        r"\b(true|false)\b\s*[\.\!\)]*$"
    ]
    for pat in patterns:
        matches = list(re.finditer(pat, text_lower))
        if matches:
            return True if matches[-1].group(1) == "true" else False
    tail = text_lower[-150:]
    if "true" in tail and "false" not in tail:
        return True
    if "false" in tail and "true" not in tail:
        return False
    return None

def generate_step_by_step_with_self_correction(context: str, question: str, max_corrections=3):
    """
    Generates reasoning with real-time contradiction monitoring.
    If a generated step contradicts premises, it triggers a backtrack!
    """
    premise_concepts = extract_core_concepts(context)
    
    initial_user_msg = f"""Context:
{context}

Question:
{question}

Please deduce the answer step-by-step. List each deduction step starting with 'Step 1:', 'Step 2:', etc.
End your reasoning with:
Therefore, the statement is [True/False]."""

    messages = [{"role": "user", "content": initial_user_msg}]
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    corrections_triggered = 0
    generated_text = ""
    
    # We generate in chunks of steps
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=450,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_text = tokenizer.decode(out_ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    # Step-level verification
    steps = re.split(r"(Step\s*\d+:)", gen_text)
    verified_steps = []
    has_contradiction = False
    
    for k in range(1, len(steps), 2):
        step_header = steps[k]
        step_body = steps[k+1] if k+1 < len(steps) else ""
        step_full = f"{step_header} {step_body.strip()}"
        
        step_concepts = extract_core_concepts(step_body)
        worst_friction = compute_step_friction(premise_concepts, step_concepts)
        
        # If severe contradiction detected (worst_friction < -4.5)
        if worst_friction < -4.5 and corrections_triggered < max_corrections:
            corrections_triggered += 1
            has_contradiction = True
            # Backtrack & append self-correction thought
            correction_note = f"\n[Self-Correction Monitor: '{step_concepts[:2]}' conflicts with premise rules (Friction: {worst_friction:.1f}). Backtracking.]\nWait, let me re-check the premises carefully:"
            verified_steps.append(correction_note)
            break
        else:
            verified_steps.append(step_full)
            
    # If contradiction was caught, re-prompt model to complete from the verified point
    if has_contradiction and corrections_triggered > 0:
        partial_chain = "\n".join(verified_steps)
        followup_msg = f"{initial_user_msg}\n\nHere is your verified deduction so far:\n{partial_chain}\n\nPlease proceed logically and give the final conclusion:"
        followup_input = tokenizer(tokenizer.apply_chat_template([{"role": "user", "content": followup_msg}], tokenize=False, add_generation_prompt=True), return_tensors="pt").to(device)
        
        with torch.no_grad():
            followup_ids = model.generate(
                **followup_input,
                max_new_tokens=300,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        corrected_gen = tokenizer.decode(followup_ids[0, followup_input.input_ids.shape[1]:], skip_special_tokens=True).strip()
        final_text = f"{partial_chain}\n{corrected_gen}"
    else:
        final_text = gen_text
        
    final_verdict = extract_conclusion_from_text(final_text)
    return final_text, final_verdict, corrections_triggered

def run_self_correction_benchmark(num_samples=30):
    print(f"\nLoading ProntoQA dataset ({num_samples} samples for Self-Correction testing)...")
    dataset = load_dataset('renma/ProntoQA')['validation']
    items = dataset.select(range(min(num_samples, len(dataset))))
    
    baseline_correct = 0
    corrected_correct = 0
    total_corrections = 0
    
    start_time = time.time()
    for i, ex in enumerate(items):
        context = ex['context']
        question = ex['question']
        ground_truth = True if ex['answer'].strip() == 'A' else False
        
        # 1. Baseline generation (uncorrected)
        base_prompt = tokenizer.apply_chat_template([{"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}\n\nReason step-by-step. End with 'Therefore, the statement is [True/False].'"}], tokenize=False, add_generation_prompt=True)
        inp = tokenizer(base_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=350, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        base_text = tokenizer.decode(out[0, inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
        pred_base = extract_conclusion_from_text(base_text)
        
        if pred_base == ground_truth:
            baseline_correct += 1
            
        # 2. Real-time Self-Correcting Reasoner
        sc_text, pred_sc, num_corr = generate_step_by_step_with_self_correction(context, question)
        total_corrections += num_corr
        
        if pred_sc == ground_truth:
            corrected_correct += 1
            
        if (i + 1) % 5 == 0 or (i + 1) == len(items):
            elapsed = time.time() - start_time
            print(f"  [{i+1}/{len(items)}] Base Acc: {baseline_correct/(i+1)*100:.1f}% | Self-Corrected Acc: {corrected_correct/(i+1)*100:.1f}% | Corrections: {total_corrections} ({elapsed:.1f}s)", flush=True)
            
    base_acc = baseline_correct / len(items) * 100.0
    sc_acc = corrected_correct / len(items) * 100.0
    
    print("\n==================================================================")
    print(" SUMMARY: REAL-TIME SELF-CORRECTION BENCHMARK RESULTS             ")
    print("==================================================================")
    print(f"Problems Evaluated:          {len(items)}")
    print(f"Base 1.5B (Uncorrected):    {base_acc:.2f}%")
    print(f"Self-Corrected Reasoner:    {sc_acc:.2f}% (Gap: {sc_acc - base_acc:+.2f}%)")
    print(f"Total Corrections Triggered: {total_corrections}")
    
    # Save results
    results_data = {
        "num_samples": len(items),
        "base_accuracy": base_acc,
        "self_corrected_accuracy": sc_acc,
        "total_corrections": total_corrections
    }
    with open(os.path.join(OUTPUT_DIR, "self_correction_prontoqa_results.json"), "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
        
    # Plotting
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(['Base 1.5B (Uncorrected)', 'Self-Corrected Reasoner\n(Energy Backtracking)'], [base_acc, sc_acc], color=['#95a5a6', '#27ae60'], alpha=0.85, width=0.5)
    ax.set_ylabel('ProntoQA Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('Real-Time Energy Self-Correction vs Base Model\n(ProntoQA Deductive Proof Chain)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    for b in bars:
        h = b.get_height()
        ax.annotate(f'{h:.1f}%', xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "self_correction_prontoqa_comparison.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved plot to {plot_path}")
    
    import shutil
    shutil.copyfile(plot_path, os.path.join(ARTIFACT_DIR, "self_correction_prontoqa_comparison.png"))

if __name__ == "__main__":
    run_self_correction_benchmark(num_samples=30)
