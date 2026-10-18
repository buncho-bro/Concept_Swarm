import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"
SLICE_LORA_PATH = r"C:\Users\lop3l\.gemini\antigravity\scratch\Antigravity_Concept_Swarm\distillation_experiments\lora_glm5.3_slice_topological_to_Qwen2.5_Coder_3B"

def main():
    print(f"Loading Base: {BASE_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print(f"Attaching Pure Slice & Topological LoRA: {SLICE_LORA_PATH}...")
    model = PeftModel.from_pretrained(base_model, SLICE_LORA_PATH)
    model.eval()
    
    prompt = "Pythonで高並行性に対応したスレッドセーフな非同期イベントバス（EventBus）を実装してください。設計の要点とクリーンなコードを示してください。"
    messages = [
        {"role": "system", "content": "You are a master software architect."},
        {"role": "user", "content": prompt}
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    
    print("Generating response from Slice-Distilled Model...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.05,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
    res_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(res_tokens, skip_special_tokens=True)
    print("\n" + "=" * 50)
    print("【Pure Slice & Topological Distilled Model Output】")
    print("=" * 50)
    print(text)

if __name__ == "__main__":
    main()
