import os
import sys
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread

# テスト: 積層地形コントローラーと TextIteratorStreamer の統合動作確認
class LayerTerrainHook:
    def __init__(self, layer_idx, layer_name, attractors, gamma=0.08, sigma=0.25):
        self.layer_idx = layer_idx
        self.layer_name = layer_name
        self.attractors = attractors
        self.gamma = gamma
        self.sigma = sigma
        self.active = True

    def __call__(self, module, inputs, output):
        if not self.active or self.attractors is None:
            return output
        last_h = output[:, -1, :]
        h_norm = F.normalize(last_h, p=2, dim=-1)
        cos_sim = torch.matmul(h_norm, self.attractors.T)
        dist_sq = 2.0 - 2.0 * cos_sim
        pulls = torch.exp(- dist_sq / (2.0 * self.sigma**2))
        weights = F.softmax(pulls * 8.0, dim=-1)
        target_dir = torch.matmul(weights, self.attractors)
        target_dir = F.normalize(target_dir, p=2, dim=-1)
        tangent_nudge = target_dir - torch.sum(target_dir * h_norm, dim=-1, keepdim=True) * h_norm
        tangent_nudge = F.normalize(tangent_nudge, p=2, dim=-1)
        h_magnitude = torch.norm(last_h, p=2, dim=-1, keepdim=True)
        nudged_last_h = last_h + self.gamma * h_magnitude * tangent_nudge
        new_output = output.clone()
        new_output[:, -1, :] = nudged_last_h
        return new_output

class MultiLayerSynchronizedController:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.hook_handles = []

    def attach_layer_terrains(self, layer_specs, gamma=0.08):
        self.detach()
        for l_idx, attractors in layer_specs.items():
            hook = LayerTerrainHook(
                layer_idx=l_idx,
                layer_name=f"Layer_{l_idx}_MLP",
                attractors=attractors,
                gamma=gamma
            )
            handle = self.model.model.layers[l_idx].mlp.register_forward_hook(hook)
            self.hooks.append(hook)
            self.hook_handles.append(handle)

    def set_active(self, active: bool):
        for h in self.hooks:
            h.active = active

    def detach(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hooks.clear()
        self.hook_handles.clear()

def test_integration():
    print("Testing Stacked Terrains Integration with Streamer...")
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cuda")
    model.eval()

    num_layers = getattr(model.config, "num_hidden_layers", len(model.model.layers))
    l_low = int(num_layers * 0.25)
    l_mid = int(num_layers * 0.50)
    l_high = int(num_layers * 0.75)
    print(f"Num layers: {num_layers}, Selected: Low={l_low}, Mid={l_mid}, High={l_high}")

    # Generate dummy attractors
    attractors = {
        l_low: torch.randn(2, model.config.hidden_size, device="cuda", dtype=torch.float16),
        l_mid: torch.randn(2, model.config.hidden_size, device="cuda", dtype=torch.float16),
        l_high: torch.randn(2, model.config.hidden_size, device="cuda", dtype=torch.float16),
    }
    for k in attractors:
        attractors[k] = F.normalize(attractors[k], p=2, dim=-1)

    controller = MultiLayerSynchronizedController(model)
    controller.attach_layer_terrains(attractors, gamma=0.08)

    prompt = "Explain in one brief sentence how locks prevent race conditions."
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(inputs, streamer=streamer, max_new_tokens=60, do_sample=True, temperature=0.7)

    thread = Thread(target=model.generate, kwargs=kwargs)
    thread.start()

    collected = ""
    for new_text in streamer:
        collected += new_text
    thread.join()

    print("Generated output:")
    print(collected)
    controller.detach()
    print("Test passed successfully!")

if __name__ == "__main__":
    test_integration()
