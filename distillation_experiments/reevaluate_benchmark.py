import json
import re
import multiprocessing
from tqdm import tqdm
from datasets import load_dataset

# ==============================================================================
# Re-evaluate Benchmark
# 
# 既に生成済みの benchmark_results_humaneval_plus.jsonl を読み込み、
# 構文バグ（先頭の 'python' などのゴミ）を除去した上で、
# テストコードだけを高速に再評価するスクリプトです。
# タイムアウトは10秒に延長しています。
# ==============================================================================

INPUT_FILE = "benchmark_results_humaneval_plus.jsonl"
OUTPUT_FILE = "benchmark_results_humaneval_plus_reevaluated.jsonl"
TIMEOUT_SECONDS = 10.0

def _run_exec(full_code, queue):
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
    except AssertionError:
        queue.put((False, "Assertion Error (Edge Case Failed)"))
    except Exception as e:
        queue.put((False, f"Execution Error: {type(e).__name__}"))

def evaluate_code(cleaned_code, base_test, plus_test, entry_point):
    full_code = f"{cleaned_code}\n\n{base_test}\n\n"
    if plus_test:
        full_code += f"{plus_test}\n\n"
    full_code += f"check({entry_point})"
    
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_run_exec, args=(full_code, queue))
    p.start()
    p.join(timeout=TIMEOUT_SECONDS)
    
    if p.is_alive():
        p.terminate()
        p.join()
        return False, "Execution Timeout"
    
    if not queue.empty():
        return queue.get()
    return False, "Unknown Execution Error"

def clean_generated_code(code):
    """
    LLMが出力したコードに含まれるパースバグ（先頭の 'python' など）を除去する。
    """
    code = code.strip()
    # 先頭が 'python' だけの行で始まっている場合、それを削除
    if code.startswith("python\n") or code.startswith("python\r\n"):
        code = code[6:].strip()
    
    # 複数回プロンプトが連結されてしまっているケースへの対策
    # 'def function_name' が2回出てくる場合、後ろのものを採用するなど（今回は簡易的に対処）
    return code

def run_reevaluation():
    print("Loading original test data (EvalPlus)...")
    dataset = load_dataset("evalplus/humanevalplus", split="test")
    # task_id をキーにしてテストコードを辞書化
    test_cases = {}
    for task in dataset:
        test_cases[task["task_id"]] = {
            "base_test": task["test"],
            "plus_test": task.get("plus", ""),
            "entry_point": task["entry_point"]
        }
        
    print(f"Loading generated results from {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    results_data = [json.loads(line) for line in lines]
    total_tasks = len(results_data)
    
    passed_count = 0
    timeout_count = 0
    assertion_count = 0
    other_error_count = 0
    
    print(f"Re-evaluating {total_tasks} tasks with {TIMEOUT_SECONDS}s timeout...")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for res in tqdm(results_data, desc="Re-evaluating"):
            task_id = res["task_id"]
            raw_code = res["generated_code"]
            
            # コードのクリーニング
            cleaned_code = clean_generated_code(raw_code)
            
            # テストケースの取得
            tests = test_cases.get(task_id)
            if not tests:
                continue
                
            passed, reason = evaluate_code(cleaned_code, tests["base_test"], tests["plus_test"], tests["entry_point"])
            
            if passed:
                passed_count += 1
            elif "Timeout" in reason:
                timeout_count += 1
            elif "Assertion" in reason:
                assertion_count += 1
            else:
                other_error_count += 1
                
            new_res = {
                "task_id": task_id,
                "passed": passed,
                "reason": reason,
                "generated_code": cleaned_code
            }
            out_f.write(json.dumps(new_res) + "\n")
            
    pass_at_1 = (passed_count / total_tasks) * 100
    
    print("\n==================================================")
    print(" 🔄 RE-EVALUATION RESULTS (HumanEval+)")
    print("==================================================")
    print(f"Total Tasks: {total_tasks}")
    print(f"Passed Tasks: {passed_count} ({pass_at_1:.2f}%)")
    print(f"Failed (Edge Case / Assertion): {assertion_count}")
    print(f"Failed (Timeout >= {TIMEOUT_SECONDS}s): {timeout_count}")
    print(f"Failed (Syntax/Name Error): {other_error_count}")
    print("==================================================")

if __name__ == "__main__":
    run_reevaluation()
