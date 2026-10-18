"""
run_mbppplus_benchmark.py
==============================================================================
EvalPlus (MBPP+) ベンチマーク実行スクリプト
GLM-5.3 概念地形スライス・トポロジー蒸留 LoRA のコーディング性能測定
==============================================================================
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import json
import re
import argparse
import multiprocessing
from tqdm import tqdm
from eval_worker import _run_exec

DEFAULT_BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
DEFAULT_LORA_PATH = r"C:\Users\lop3l\.gemini\antigravity\scratch\Antigravity_Concept_Swarm\distillation_experiments\lora_glm5.3_slice_topological_to_Qwen2.5_Coder_3B"
DEFAULT_OUTPUT_FILE = "benchmark_results_mbpp_plus_slice_topological.jsonl"

def load_distilled_model(base_model_id, lora_path):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    
    print(f"Loading Base Model: {base_model_id} (FP16)...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    if lora_path and os.path.exists(lora_path):
        print(f"Applying Distilled Topology (LoRA): {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
    else:
        print(f"⚠️ LoRA path '{lora_path}' not found or not provided. Benchmarking BASE model only.")
        
    model.eval()
    return tokenizer, model

def extract_python_code(text):
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def evaluate_code(generated_code, base_test, plus_test):
    full_code = f"{generated_code}\n\n{base_test}\n\n"
    if plus_test:
        full_code += f"{plus_test}\n\n"
    
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_run_exec, args=(full_code, queue))
    p.start()
    p.join(timeout=4.0) # 4秒タイムアウト
    
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

def assemble_code(signature, python_code):
    if not signature:
        return python_code
        
    func_name_match = re.search(r"def\s+([a-zA-Z0-9_]+)\s*\(", signature)
    func_name = func_name_match.group(1) if func_name_match else None
    
    # 1. すでに同名関数の定義が含まれている、または def 文が存在する場合
    if func_name and re.search(rf"\bdef\s+{func_name}\b", python_code):
        return python_code
    if re.search(r"^\s*def\s+[a-zA-Z0-9_]+", python_code, re.MULTILINE):
        return python_code
        
    # 2. 関数本体のみが出力された場合
    lines = python_code.splitlines()
    first_non_empty = next((l for l in lines if l.strip()), None)
    if first_non_empty and not (first_non_empty.startswith(" ") or first_non_empty.startswith("\t")):
        indented_body = "\n".join("    " + l if l.strip() else l for l in lines)
        return signature + "\n" + indented_body
    else:
        return signature + "\n" + python_code


def main():
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser(description="Run MBPP+ Benchmark for Distilled Model")
    parser.add_argument("--base", type=str, default=DEFAULT_BASE_MODEL_ID, help="Base Model ID")
    parser.add_argument("--lora", type=str, default=DEFAULT_LORA_PATH, help="LoRA Directory Path")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_FILE, help="Output JSONL Path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks to evaluate")
    parser.add_argument("--start_idx", type=int, default=0, help="Start task index")
    parser.add_argument("--use_idea_hook", action="store_true", help="Attach Tangent Nudge Attractor Hook during inference")
    parser.add_argument("--hook_alpha", type=float, default=0.08, help="Attractor tangent nudge strength")
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    
    print("=" * 70)
    print(" 🏆 EVALPLUS BENCHMARK: MBPP+ (Python Coding Evaluation)")
    print("=" * 70)
    print(f"Base Model: {args.base}")
    print(f"LoRA: {args.lora}")
    print(f"Idea Hook Enabled: {args.use_idea_hook} (alpha={args.hook_alpha})")
    print(f"Output File: {args.output}")

    tokenizer, model = load_distilled_model(args.base, args.lora)
    
    nudge_hook = None
    if args.use_idea_hook:
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "distillation_experiments"))
            from universal_idea_core_engine import LanguageAgnosticIdeaCore, LatentOrthogonalBridge, TangentNudgeHook
            
            print("\n[Idea Core] Initializing Language-Agnostic Idea Core & Tangent Hook...")
            idea_core = LanguageAgnosticIdeaCore(canonical_dim=2048, device=model.device)
            idea_core.load_teacher_terrains()
            
            # 直交ブリッジ
            bridge = LatentOrthogonalBridge(in_dim=2048, idea_dim=2048).to(model.device)
            teacher_kernels = idea_core.get_layer_kernels("Layer_1") # 中間関係層
            if teacher_kernels is None:
                teacher_kernels = idea_core.get_layer_kernels("Layer_0")
            
            # 生徒モデルの埋め込みでプロクラステス較正
            embed_w = model.get_input_embeddings().weight[:54].detach()
            bridge.solve_procrustes(embed_w, teacher_kernels)
            
            # フック装着 (深層レイヤー 12, 18, 24, 30)
            target_layers = [12, 18, 24, 30]
            nudge_hook = TangentNudgeHook(bridge, teacher_kernels, alpha=args.hook_alpha, beta=8.0)
            nudge_hook.register(model, target_layers)
            print(f"✨ Tangent Nudge Hook successfully activated on layers {target_layers}!")
        except Exception as e:
            print(f"⚠️ Failed to activate Idea Hook: {e}")

    
    print("\nLoading EvalPlus (MBPP+) dataset from HuggingFace...")
    dataset = load_dataset("evalplus/mbppplus", split="test")
    total_dataset_tasks = len(dataset)
    print(f"Total tasks in dataset: {total_dataset_tasks}")

    tasks_to_run = list(dataset)
    if args.start_idx > 0:
        tasks_to_run = tasks_to_run[args.start_idx:]
    if args.limit:
        tasks_to_run = tasks_to_run[:args.limit]

    total_tasks = len(tasks_to_run)
    completed_tasks = set()
    passed_tasks = 0
    results = []

    if os.path.exists(args.output):
        print(f"Found existing {args.output}. Loading progress...")
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        completed_tasks.add(data["task_id"])
                        if data.get("passed"):
                            passed_tasks += 1
                    except Exception:
                        pass
        print(f"Resuming: {len(completed_tasks)} tasks already completed (Passed: {passed_tasks})")

    print(f"\nStarting evaluation of {total_tasks} tasks...\n")
    
    with open(args.output, "a", encoding="utf-8") as f:
        for i, task in enumerate(tqdm(tasks_to_run, desc="Evaluating MBPP+")):
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
                {"role": "system", "content": "You are an expert Python developer. Output ONLY valid, executable Python code inside a markdown block. Make sure to handle all edge cases correctly."},
                {"role": "user", "content": f"Instruction: {prompt}\n\nComplete this code:\n\n```python\n{signature}\n"}
            ]
            
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=400,
                    temperature=0.1,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            python_code = extract_python_code(generated_text)
            
            final_code = assemble_code(signature, python_code)

            
            passed, reason = evaluate_code(final_code, base_test_str, plus_test)
            completed_tasks.add(task_id)
            
            if passed:
                passed_tasks += 1
                
            result_dict = {
                "task_id": task_id,
                "passed": passed,
                "reason": reason,
                "generated_code": final_code
            }
            results.append(result_dict)
            f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")
            f.flush()

            # 10問ごとに進捗率を出力
            if len(completed_tasks) % 10 == 0:
                cur_rate = (passed_tasks / len(completed_tasks)) * 100
                print(f"\n[Progress: {len(completed_tasks)}/{total_tasks}] Passed: {passed_tasks} ({cur_rate:.2f}%)")
            
    pass_at_1 = (passed_tasks / len(completed_tasks)) * 100 if completed_tasks else 0.0
    
    print("\n" + "=" * 70)
    print(" 🏆 FINAL EVALPLUS BENCHMARK RESULTS (MBPP+) ")
    print("=" * 70)
    print(f"Model: {args.base}")
    print(f"LoRA: {args.lora}")
    print(f"Total Evaluated Tasks: {len(completed_tasks)}")
    print(f"Passed Tasks: {passed_tasks}")
    print(f"Score (Pass@1): {pass_at_1:.2f}%")
    print("=" * 70)
    print(f"Detailed logs saved to: {args.output}")

if __name__ == "__main__":
    main()
