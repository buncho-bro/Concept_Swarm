import os
import torch
import json
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset

# ==============================================================================
# Distilled Model Benchmark (HumanEval Pass@1)
# 
# 蒸留済みモデル(Qwen + GLM LoRA)でHumanEvalの164問を解かせ、
# 実際にPythonのテストコード(assert)を実行して正答率(Pass@1)を算出します。
# ==============================================================================

BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
LORA_MODEL_ID = "./lora_glm5.3_to_Qwen2.5_Coder_3B_Topological"
OUTPUT_FILE = "benchmark_results_humaneval.jsonl"

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

def evaluate_code(generated_code, test_code, entry_point):
    """
    生成されたコードとテストコードを結合し、簡易的なサンドボックス(exec)内で実行する。
    """
    full_code = f"{generated_code}\n\n{test_code}\n\ncheck({entry_point})"
    
    # 実行環境の制限と標準モジュールの提供
    import math
    from typing import List, Dict, Tuple, Set, Optional, Union, Any
    import collections
    import itertools
    import re
    
    exec_globals = {
        'math': math, 'List': List, 'Dict': Dict, 'Tuple': Tuple, 'Set': Set,
        'Optional': Optional, 'Union': Union, 'Any': Any,
        'collections': collections, 'itertools': itertools, 're': re
    }
    
    try:
        exec(full_code, exec_globals)
        return True, "Passed"
    except AssertionError as e:
        return False, "Assertion Error (Test Failed)"
    except Exception as e:
        return False, f"Execution Error: {type(e).__name__}"

def run_benchmark():
    tokenizer, model = load_distilled_model()
    
    print("Downloading HumanEval dataset from HuggingFace...")
    dataset = load_dataset("openai/openai_humaneval", split="test")
    
    total_tasks = len(dataset)
    passed_tasks = 0
    results = []
    
    print(f"Starting Benchmark on {total_tasks} HumanEval tasks...")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i, task in enumerate(tqdm(dataset, desc="Evaluating")):
            task_id = task["task_id"]
            prompt = task["prompt"]
            test_code = task["test"]
            entry_point = task["entry_point"]
            
            # Coderモデル用にシステムプロンプトを設定
            messages = [
                {"role": "system", "content": "You are an expert Python developer. Complete the following Python function exactly as requested. Output ONLY the valid Python code inside a markdown block."},
                {"role": "user", "content": f"Complete this code:\n\n```python\n{prompt}\n```"}
            ]
            
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=True, top_p=0.95)
            
            generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            python_code = extract_python_code(generated_text)
            
            # 関数が再定義されているかチェック（重複を防ぐ）
            if f"def {entry_point}" not in python_code:
                 final_code = prompt + "\n" + python_code
            else:
                 final_code = python_code
            
            # テスト実行
            passed, reason = evaluate_code(final_code, test_code, entry_point)
            
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
            
    # スコア計算
    pass_at_1 = (passed_tasks / total_tasks) * 100
    
    print("\n==================================================")
    print(" 🏆 BENCHMARK RESULTS (HumanEval) ")
    print("==================================================")
    print(f"Model: {BASE_MODEL_ID} + GLM-5.3 Distilled Topology")
    print(f"Total Tasks: {total_tasks}")
    print(f"Passed Tasks: {passed_tasks}")
    print(f"Score (Pass@1): {pass_at_1:.2f}%")
    print("==================================================")
    print(f"Detailed logs saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_benchmark()
