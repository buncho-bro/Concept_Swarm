import os
import sys
import json
import time
import re
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.stdout.reconfigure(encoding='utf-8')

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment/lora_Qwen2.5_7B_to_Qwen2.5_1.5B_Instruct_Topological"
R1_DISTILL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
OUTPUT_DIR = "C:/Users/lop3l/.gemini/antigravity/scratch/concept_core/distillation_experiment"
ARTIFACT_DIR = "C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("==================================================================")
print(" PRONTOQA AUTONOMOUS CHAIN-OF-THOUGHT & PROOF BENCHMARK           ")
print(" Multi-Step Deductive Proof Generation (Base vs LoRA vs R1)       ")
print("==================================================================")
print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

print("\nLoading ProntoQA validation dataset...")
dataset = load_dataset('renma/ProntoQA')['validation']
print(f"Total available validation problems: {len(dataset)}")

def extract_conclusion_from_generation(text: str):
    """
    Extracts the final True/False verdict from model generated reasoning chain.
    """
    text_lower = text.lower()
    
    # Priority patterns for explicit conclusions
    patterns = [
        r"therefore[,\s]+(?:the\s+statement\s+is\s+|it\s+is\s+)?(true|false)",
        r"the\s+statement\s+is\s+(true|false)",
        r"conclusion[:\s]+(?:the\s+statement\s+is\s+)?(true|false)",
        r"answer[:\s]+(true|false)",
        r"\b(true|false)\b\s*[\.\!\)]*$" # Trailing word
    ]
    
    for pat in patterns:
        matches = list(re.finditer(pat, text_lower))
        if matches:
            last_match = matches[-1].group(1)
            return True if last_match == "true" else False
            
    # Fallback search in the last 200 characters
    tail = text_lower[-200:]
    has_true = "true" in tail
    has_false = "false" in tail
    if has_true and not has_false:
        return True
    elif has_false and not has_true:
        return False
    elif has_true and has_false:
        # Check which appears later
        return tail.rfind("true") > tail.rfind("false")
    return None

def evaluate_model_on_prontoqa_cot(model, tokenizer, items, model_name, is_deepseek_r1=False):
    print(f"\n>>> Evaluating {model_name} on ProntoQA Proof Generation ({len(items)} Qs) ...")
    correct = 0
    valid_format = 0
    total_tokens = 0
    results_detail = []
    
    start_time = time.time()
    for i, ex in enumerate(items):
        context = ex['context']
        question = ex['question']
        ground_truth = True if ex['answer'].strip() == 'A' else False
        
        # Build prompt suitable for chat/instruction
        if is_deepseek_r1:
            # DeepSeek-R1 naturally outputs <think> ... </think>
            user_msg = f"Context:\n{context}\n\nQuestion:\n{question}\n\nPlease reason step-by-step and deduce whether the statement is True or False. End your response with 'Therefore, the statement is [True/False].'"
            prompt = f"<｜User｜>{user_msg}<｜Assistant｜><think>\n"
        else:
            user_msg = f"Context:\n{context}\n\nQuestion:\n{question}\n\nPlease reason step-by-step through the logical rules to prove whether the statement is True or False. End your response with 'Therefore, the statement is [True/False].'"
            messages = [{"role": "user", "content": user_msg}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=False, # Deterministic greedy decoding
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id
            )
            
        gen_ids = output_ids[0, prompt_len:]
        total_tokens += len(gen_ids)
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        
        predicted_bool = extract_conclusion_from_generation(gen_text)
        is_correct = (predicted_bool == ground_truth)
        
        if predicted_bool is not None:
            valid_format += 1
        if is_correct:
            correct += 1
            
        results_detail.append({
            "idx": i,
            "ground_truth": ground_truth,
            "predicted": predicted_bool,
            "is_correct": is_correct,
            "gen_length": len(gen_ids),
            "sample_reasoning": gen_text[:250] + "..." if len(gen_text) > 250 else gen_text
        })
        
        if (i + 1) % 10 == 0 or (i + 1) == len(items):
            elapsed = time.time() - start_time
            acc = correct / (i + 1) * 100.0
            avg_tok = total_tokens / (i + 1)
            print(f"  [{i+1}/{len(items)}] Acc: {acc:.2f}% | Avg Tokens: {avg_tok:.1f} | Format Valid: {valid_format/(i+1)*100:.1f}% ({elapsed:.1f}s)", flush=True)
            
    final_acc = correct / len(items) * 100.0
    se = math.sqrt(final_acc * (100 - final_acc) / len(items))
    avg_gen_tokens = total_tokens / len(items)
    
    return {
        "model_name": model_name,
        "accuracy": final_acc,
        "se": se,
        "avg_tokens": avg_gen_tokens,
        "valid_format_rate": valid_format / len(items) * 100.0,
        "details": results_detail
    }

def main():
    NUM_SAMPLES = 100 # Rigorous 100-question sample
    eval_items = dataset.select(range(NUM_SAMPLES))
    all_results = {}
    
    # ------------------------------------------------------------------
    # 1. Base 1.5B (Qwen2.5-1.5B-Instruct)
    # ------------------------------------------------------------------
    print("\n[1/3] Loading Base Model: Qwen2.5-1.5B-Instruct ...")
    tokenizer_base = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map=device
    )
    model_base.eval()
    
    res_base = evaluate_model_on_prontoqa_cot(model_base, tokenizer_base, eval_items, "Base 1.5B", is_deepseek_r1=False)
    all_results["Base 1.5B"] = res_base
    
    # Save checkpoint
    out_json = os.path.join(OUTPUT_DIR, "prontoqa_cot_benchmark_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Checkpointed Base 1.5B results to {out_json}", flush=True)
    
    del model_base
    torch.cuda.empty_cache()
    
    # ------------------------------------------------------------------
    # 2. KNN 7B to 1.5B LoRA
    # ------------------------------------------------------------------
    print("\n[2/3] Loading KNN 7B to 1.5B Distilled LoRA ...", flush=True)
    model_lora = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map=device
    )
    model_lora = PeftModel.from_pretrained(model_lora, LORA_PATH)
    model_lora.eval()
    
    res_lora = evaluate_model_on_prontoqa_cot(model_lora, tokenizer_base, eval_items, "KNN 7to1.5B LoRA", is_deepseek_r1=False)
    all_results["KNN 7to1.5B LoRA"] = res_lora
    
    # Save checkpoint
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Checkpointed KNN LoRA results to {out_json}", flush=True)
    
    del model_lora
    torch.cuda.empty_cache()
    
    # ------------------------------------------------------------------
    # 3. DeepSeek-R1-Distill-Qwen-1.5B (Real Line 2 Model)
    # ------------------------------------------------------------------
    print("\n[3/3] Loading DeepSeek-R1-Distill-Qwen-1.5B ...", flush=True)
    tokenizer_r1 = AutoTokenizer.from_pretrained(R1_DISTILL_ID)
    model_r1 = AutoModelForCausalLM.from_pretrained(
        R1_DISTILL_ID,
        torch_dtype=torch.float16,
        device_map=device
    )
    model_r1.eval()
    
    res_r1 = evaluate_model_on_prontoqa_cot(model_r1, tokenizer_r1, eval_items, "DeepSeek-R1-Distill-1.5B", is_deepseek_r1=True)
    all_results["DeepSeek-R1-Distill-1.5B"] = res_r1
    
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    del model_r1
    torch.cuda.empty_cache()
    
    # ------------------------------------------------------------------
    # Summary & Plotting
    # ------------------------------------------------------------------
    print("\n==================================================================")
    print(" SUMMARY: PRONTOQA CHAIN-OF-THOUGHT PROOF GENERATION ACCURACY     ")
    print("==================================================================")
    for name, r in all_results.items():
        print(f"Model: {name:28s} | Acc: {r['accuracy']:.2f}% (±{r['se']:.2f}%) | Avg Reasoning Tokens: {r['avg_tokens']:.1f}")
        
    out_json = os.path.join(OUTPUT_DIR, "prontoqa_cot_benchmark_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved detailed results to {out_json}")
    
    # Bar Chart
    models = list(all_results.keys())
    accs = [all_results[m]['accuracy'] for m in models]
    errs = [all_results[m]['se'] for m in models]
    tokens = [all_results[m]['avg_tokens'] for m in models]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Acc plot
    x = np.arange(len(models))
    bars = ax1.bar(x, accs, yerr=errs, capsize=6, color=['#95a5a6', '#3498db', '#e67e22'], alpha=0.85)
    ax1.set_ylabel('Deductive Reasoning Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_title('ProntoQA Multi-Step Deductive Proof Generation\n(Step-by-Step Chain-of-Thought Reasoning | 100 Qs)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=10, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.axhline(50, color='red', linestyle='--', alpha=0.6, label='Random Chance (50%)')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right')
    
    for b in bars:
        h = b.get_height()
        ax1.annotate(f'{h:.1f}%', xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    # Tokens plot
    bars2 = ax2.bar(x, tokens, color=['#bdc3c7', '#5dade2', '#f39c12'], alpha=0.85)
    ax2.set_ylabel('Average Generated Reasoning Tokens', fontsize=11, fontweight='bold')
    ax2.set_title('Reasoning Effort / Chain Depth\n(Average Thinking Tokens Generated per Problem)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=10, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    
    for b in bars2:
        h = b.get_height()
        ax2.annotate(f'{h:.1f}', xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "prontoqa_cot_proof_comparison.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved plot to {plot_path}")
    
    artifact_plot_path = os.path.join(ARTIFACT_DIR, "prontoqa_cot_proof_comparison.png")
    import shutil
    shutil.copyfile(plot_path, artifact_plot_path)
    print(f"Copied to artifact dir: {artifact_plot_path}")

if __name__ == "__main__":
    main()
