import os
import json
import re
import multiprocessing

# ==============================================================================
# EvalPlus (MBPP+) Distilled Benchmark
# ==============================================================================

BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
LORA_MODEL_ID = "./lora_glm5.3_to_Qwen2.5_Coder_3B_Topological"
OUTPUT_FILE = "benchmark_results_mbpp_plus.jsonl"

def load_distilled_model():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    
    print(f"Loading Base Model: {BASE_MODEL_ID} (16bit)")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, device_map="auto", torch_dtype=torch.float16)
    
    if os.path.exists(LORA_MODEL_ID):
        print(f"Applying Distilled Topology (LoRA): {LORA_MODEL_ID}")
        model = PeftModel.from_pretrained(model, LORA_MODEL_ID)
    else:
        print(f"⚠️ LoRA directory not found at {LORA_MODEL_ID}. Benchmarking BASE model only.")
        
    model.eval()
    return tokenizer, model

def extract_python_code(text):
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

import multiprocessing
from eval_worker import _run_exec

def evaluate_code(generated_code, base_test, plus_test):
    full_code = f"{generated_code}\n\n{base_test}\n\n"
    if plus_test:
        full_code += f"{plus_test}\n\n"
    
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_run_exec, args=(full_code, queue))
    p.start()
    p.join(timeout=3.0)
    
    if p.is_alive():
        p.terminate()
        p.join()
        return False, "Execution Timeout (Infinite Loop?)"
    
    if not queue.empty():
        return queue.get()
    return False, "Unknown Execution Error"

def get_signature(code):
    for line in code.split('\n'):
        if line.startswith('def '):
            return line
    return ""

def run_benchmark():
    import torch
    from tqdm import tqdm
    from datasets import load_dataset
    
    tokenizer, model = load_distilled_model()
    
    print("Downloading EvalPlus (MBPP+) dataset from HuggingFace...")
    dataset = load_dataset("evalplus/mbppplus", split="test")
    
    total_tasks = len(dataset)
    
    completed_tasks = set()
    passed_tasks = 0
    results = []
    
    if os.path.exists(OUTPUT_FILE):
        print(f"Found existing {OUTPUT_FILE}. Loading progress...")
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    completed_tasks.add(data["task_id"])
                    if data["passed"]:
                        passed_tasks += 1
        print(f"Resuming from task {len(completed_tasks)}/{total_tasks}")
    
    print(f"Starting EvalPlus Benchmark on MBPP+ hardcore tasks...")
    
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for i, task in enumerate(tqdm(dataset, desc="Evaluating")):
            task_id = task["task_id"]
            if task_id in completed_tasks:
                continue
                
            prompt = task["prompt"]
            base_test = task["test_list"]
            plus_test = task.get("test", "")
            reference_code = task["code"]
            
            if isinstance(base_test, list):
                base_test_str = "\n".join(base_test)
            else:
                base_test_str = str(base_test)
                
            signature = get_signature(reference_code)
            
            messages = [
                {"role": "system", "content": "You are an expert Python developer. Output ONLY the valid Python code inside a markdown block. Make sure to handle all edge cases correctly."},
                {"role": "user", "content": f"Instruction: {prompt}\n\nComplete this code:\n\n```python\n{signature}\n"}
            ]
            
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=True, top_p=0.95)
            
            generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            python_code = extract_python_code(generated_text)
            
            if signature and not python_code.startswith("def "):
                final_code = signature + "\n" + python_code
            else:
                final_code = python_code
            
            passed, reason = evaluate_code(final_code, base_test_str, plus_test)
            
            if passed:
                passed_tasks += 1
                
            result_dict = {
                "task_id": task_id,
                "passed": passed,
                "reason": reason,
                "generated_code": final_code
            }
            results.append(result_dict)
            f.write(json.dumps(result_dict) + "\n")
            f.flush()
            
    pass_at_1 = (passed_tasks / total_tasks) * 100
    
    print("\n==================================================")
    print(" 🏆 EVALPLUS BENCHMARK RESULTS (MBPP+) ")
    print("==================================================")
    print(f"Model: {BASE_MODEL_ID} + GLM-5.3 Distilled Topology")
    print(f"Total Tasks: {total_tasks}")
    print(f"Passed Tasks: {passed_tasks}")
    print(f"Score (Pass@1): {pass_at_1:.2f}%")
    print("==================================================")
    print(f"Detailed logs saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_benchmark()
