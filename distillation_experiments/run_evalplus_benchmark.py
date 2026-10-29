import os
import torch
import json
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset

# ==============================================================================
# EvalPlus (HumanEval+) Distilled Benchmark
# 
# 蒸留済みモデルで、境界値テストが大幅に強化された HumanEval+ を実行します。
# 元のテストに加え、数百の過酷な追加テスト(plus)をサンドボックス内で評価します。
# ==============================================================================

BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
LORA_MODEL_ID = "./lora_glm5.3_to_Qwen2.5_Coder_3B_Topological"
OUTPUT_FILE = "benchmark_results_humaneval_plus.jsonl"

def load_distilled_model():
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
    # マークダウンのコードブロックからPythonコードを抽出
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

import multiprocessing

def _run_exec(full_code, queue):
    # 標準モジュール群を提供 (EvalPlusのテスト実行に必要なもの)
    import math
    from typing import List, Dict, Tuple, Set, Optional, Union, Any
    import collections
    import itertools
    import re
    import hashlib
    
    exec_globals = {
        'math': math, 'List': List, 'Dict': Dict, 'Tuple': Tuple, 'Set': Set,
        'Optional': Optional, 'Union': Union, 'Any': Any,
        'collections': collections, 'itertools': itertools, 're': re,
        'hashlib': hashlib
    }
    try:
        exec(full_code, exec_globals)
        queue.put((True, "Passed"))
    except AssertionError as e:
        queue.put((False, "Assertion Error (Edge Case Failed)"))
    except Exception as e:
        queue.put((False, f"Execution Error: {type(e).__name__}"))

def evaluate_code(generated_code, base_test, plus_test, entry_point):
    """
    生成されたコードを実行する際、3秒のタイムアウトを設ける。
    無限ループなどのバグでプロセスがハングし、GPUが止まるのを防ぐため。
    """
    full_code = f"{generated_code}\n\n{base_test}\n\n"
    if plus_test:
        full_code += f"{plus_test}\n\n"
    full_code += f"check({entry_point})"
    
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_run_exec, args=(full_code, queue))
    p.start()
    p.join(timeout=3.0)  # 3秒で強制タイムアウト
    
    if p.is_alive():
        p.terminate()
        p.join()
        return False, "Execution Timeout (Infinite Loop?)"
    
    if not queue.empty():
        return queue.get()
    return False, "Unknown Execution Error"

def run_benchmark():
    tokenizer, model = load_distilled_model()
    
    print("Downloading EvalPlus (HumanEval+) dataset from HuggingFace...")
    # EvalPlusのデータセットをロード
    dataset = load_dataset("evalplus/humanevalplus", split="test")
    
    total_tasks = len(dataset)
    passed_tasks = 0
    results = []
    
    print(f"Starting EvalPlus Benchmark on {total_tasks} hardcore tasks...")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i, task in enumerate(tqdm(dataset, desc="Evaluating")):
            task_id = task["task_id"]
            prompt = task["prompt"]
            base_test = task["test"]
            # EvalPlus特有の過酷な追加テストコード
            plus_test = task.get("plus", "")
            entry_point = task["entry_point"]
            
            messages = [
                {"role": "system", "content": "You are an expert Python developer. Complete the following Python function exactly as requested. Output ONLY the valid Python code inside a markdown block. Make sure to handle all edge cases correctly."},
                {"role": "user", "content": f"Complete this code:\n\n```python\n{prompt}\n```"}
            ]
            
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                # 境界値の考慮が必要なため、少し長めに設定
                outputs = model.generate(**inputs, max_new_tokens=768, temperature=0.1, do_sample=True, top_p=0.95)
            
            generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            python_code = extract_python_code(generated_text)
            
            if f"def {entry_point}" not in python_code:
                 final_code = prompt + "\n" + python_code
            else:
                 final_code = python_code
            
            passed, reason = evaluate_code(final_code, base_test, plus_test, entry_point)
            
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
    print(" 🏆 EVALPLUS BENCHMARK RESULTS (HumanEval+) ")
    print("==================================================")
    print(f"Model: {BASE_MODEL_ID} + GLM-5.3 Distilled Topology")
    print(f"Total Tasks: {total_tasks}")
    print(f"Passed Tasks: {passed_tasks}")
    print(f"Score (Pass@1): {pass_at_1:.2f}%")
    print("==================================================")
    print(f"Detailed logs saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_benchmark()
