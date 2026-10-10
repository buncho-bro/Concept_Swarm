"""
【ファイル概要】
env_setter.py

Project Ripple実行時の環境変数、パス、ハードウェア（CUDA/CPU）設定等を自動構成し、安定したデプロイ環境を提供するユーティリティ。
"""

import json
import re
import torch
import os

class EnvironmentConfigurator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.history_file = "config_eval_history.json"
        self.knowledge_base = self._load_history()

    def _load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return []

    def _save_history(self):
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base[-10:], f, ensure_ascii=False, indent=2) # 直近10件を記憶

    def analyze_task(self, user_task):
        past_lessons = ""
        if self.knowledge_base:
            lessons = [f"Task: {k['task']} -> Learned: {k['lesson']}" for k in self.knowledge_base[-3:]]
            past_lessons = "Previous Self-Evaluations:\n" + "\n".join(lessons) + "\n\n"

        sys_prompt = (
            "You are the Environment Configurator for 'Project Ripple'.\n"
            "Analyze the user's task and output optimal physical parameters for the LLM.\n"
            f"{past_lessons}"
            "1. terrain_injection: Choose ONE from ['Quantum Mechanics', 'Evolution', 'Fluid Dynamics', 'Thermodynamics', 'None'].\n"
            "2. viscosity (0.0 to 1.0): Concept retention power. High(0.8-1.0) for reading long documents/web search. Low(0.0-0.3) for rapid brainstorm.\n"
            "3. decay (0.0 to 1.0): Speed of context forgetting. Fast(high) for rapid switching. Slow(low) to keep context.\n\n"
            "CRITICAL RULE: The 'reasoning' value MUST be written in the exact same language as the user's Task.\n\n"
            "Output ONLY JSON: {'terrain_injection': '...', 'viscosity': 0.0, 'decay': 0.0, 'reasoning': '...'}"
        )
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": f"Task: {user_task}"}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=300, temperature=0.1)
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        try:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception as e:
            pass
        return {"terrain_injection": "None", "viscosity": 0.5, "decay": 0.5, "reasoning": "Fallback config."}

    def evaluate_performance(self, task, output_text, config):
        """
        自己評価プロトコル：タスク完了後に呼び出され、設定パラメータが適切だったかを反省する。
        """
        sys_prompt = (
            "You are the Self-Evaluation Protocol.\n"
            "Review the Task, the generated Output, and the applied Config (viscosity, decay, terrain).\n"
            "Did the output collapse? Was the terrain too strong? Was viscosity too high making it ignore the main task?\n"
            "Provide a short 'lesson' to improve future configs. Output ONLY JSON: {'lesson': '...'}"
        )
        user_content = f"Task: {task}\nConfig: {config}\nOutput Snippet: {output_text[:200]}..."
        
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_content}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=150, temperature=0.3)
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                eval_data = json.loads(match.group(0))
                lesson = eval_data.get('lesson', 'No clear lesson.')
                self.knowledge_base.append({"task": task, "lesson": lesson})
                self._save_history()
                print(f"[Self-Evaluation] Learned: {lesson}")
        except Exception as e:
            print(f"[Self-Evaluation] Failed: {str(e)}")
