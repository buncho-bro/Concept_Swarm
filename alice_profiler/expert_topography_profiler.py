"""
【ファイル概要】
expert_topography_profiler.py

160GBのAliceAIモデルに対し、HTTP Rangeリクエストを用いてsafetensorsから必要なゲート重みだけをストリーミング取得し、ローカル容量・RAMをほぼ消費せずにExpertの専門性を抽出するスクリプト。
"""

import requests
import json
import struct
import numpy as np
import time
from transformers import AutoTokenizer

class HTTP_SafeOpen:
    """
    Zero-Full-Load Safetensors Reader via HTTP Range requests.
    Fetches only the required tensor bytes over the network without downloading the entire shard.
    """
    def __init__(self, repo_id, file_name):
        self.repo_id = repo_id
        self.file_name = file_name
        self.url = f'https://huggingface.co/{repo_id}/resolve/main/{file_name}'
        
        # Read header length (8 bytes)
        r = requests.get(self.url, headers={'Range': 'bytes=0-7'})
        if r.status_code not in (200, 206):
            raise Exception(f'Failed to read safetensors header length for {file_name}')
        n = struct.unpack('<Q', r.content)[0]
        
        # Read JSON header
        r = requests.get(self.url, headers={'Range': f'bytes=8-{8+n-1}'})
        self.header = json.loads(r.content)
        self.data_start = 8 + n

    def _bytes_to_numpy(self, data, dtype_str, shape):
        if dtype_str == 'BF16':
            arr = np.frombuffer(data, dtype=np.uint16)
            arr_f32 = np.zeros(arr.size, dtype=np.uint32)
            arr_f32[:] = arr
            arr_f32 = arr_f32 << 16
            return arr_f32.view(np.float32).reshape(shape)
        elif dtype_str == 'F16':
            return np.frombuffer(data, dtype=np.float16).reshape(shape).astype(np.float32)
        elif dtype_str == 'F32':
            return np.frombuffer(data, dtype=np.float32).reshape(shape)
        else:
            raise Exception(f'Unsupported dtype {dtype_str}')

    def get_tensor(self, tensor_name):
        t_info = self.header[tensor_name]
        start, end = t_info['data_offsets']
        r = requests.get(self.url, headers={'Range': f'bytes={self.data_start + start}-{self.data_start + end - 1}'})
        return self._bytes_to_numpy(r.content, t_info['dtype'], t_info['shape'])

    def get_embedding_row(self, tensor_name, row_idx):
        t_info = self.header[tensor_name]
        start_offset = t_info['data_offsets'][0]
        row_size = t_info['shape'][1] * 2  # assuming 2 bytes per element (BF16/F16)
        
        start = start_offset + row_idx * row_size
        end = start + row_size - 1
        
        r = requests.get(self.url, headers={'Range': f'bytes={self.data_start + start}-{self.data_start + end}'})
        return self._bytes_to_numpy(r.content, t_info['dtype'], (t_info['shape'][1],))


def get_safetensors_index(repo_id):
    url = f'https://huggingface.co/{repo_id}/resolve/main/model.safetensors.index.json'
    r = requests.get(url)
    if r.status_code == 404:
        raise Exception("Model is not sharded or index not found.")
    return r.json()['weight_map']

def compute_concept_core(f_embed, embed_tensor_name, token_ids):
    vectors = []
    for tid in token_ids:
        vec = f_embed.get_embedding_row(embed_tensor_name, tid)
        # L2 normalize
        vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
        vectors.append(vec_norm)
    
    # Spherical mean (Great Circle Interference center)
    core = np.sum(vectors, axis=0)
    core = core / (np.linalg.norm(core) + 1e-8)
    return core

def main():
    repo_id = "yandex/AliceAI-Foundation-80B-A3B-Base"
    print(f"--- AliceAI Expert Topography Profiler ---")
    print(f"Target Model: {repo_id}")
    
    print("\n1. Fetching Tokenizer and generating concept anchors...")
    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
    
    math_words = ['algorithm', 'calculus', 'formula', 'proof', 'series', 'math', 'equation', 'geometry', 'theorem']
    code_words = ['def', 'return', 'implementation', 'syntax', 'function', 'loop', 'class', 'import', 'yield', 'async']
    
    math_tokens = [tokenizer.encode(w, add_special_tokens=False)[0] for w in math_words]
    code_tokens = [tokenizer.encode(w, add_special_tokens=False)[0] for w in code_words]
    
    print("\n2. Fetching Safetensors Index via HTTP...")
    weight_map = get_safetensors_index(repo_id)
    
    embed_file = weight_map["model.embed_tokens.weight"]
    f_embed = HTTP_SafeOpen(repo_id, embed_file)
    
    print("Computing Mathematical Concept Core (x*_math)...")
    x_math = compute_concept_core(f_embed, "model.embed_tokens.weight", math_tokens)
    
    print("Computing Coding Concept Core (x*_code)...")
    x_code = compute_concept_core(f_embed, "model.embed_tokens.weight", code_tokens)
    
    math_manifest = {"domain": "math", "experts": []}
    code_manifest = {"domain": "code", "experts": []}
    
    print("\n3. Scanning 48 Layers (Gate Weights) via HTTP Streaming...")
    
    total_math_experts = set()
    total_code_experts = set()
    
    K = 16 # Top 16 experts per layer
    
    for l in range(48):
        t0 = time.time()
        gate_name = f"model.layers.{l}.mlp.gate.weight"
        if gate_name not in weight_map:
            print(f"Warning: {gate_name} not found.")
            continue
            
        file_name = weight_map[gate_name]
        f_gate = HTTP_SafeOpen(repo_id, file_name)
        
        gate_w = f_gate.get_tensor(gate_name) # shape (512, 2048)
        
        # Normalize each expert's gate vector
        norms = np.linalg.norm(gate_w, axis=1, keepdims=True) + 1e-8
        gate_w_norm = gate_w / norms
        
        # Compute Cosine Similarities
        s_math = np.dot(gate_w_norm, x_math)
        s_code = np.dot(gate_w_norm, x_code)
        
        # Get top K indices
        top_math = np.argsort(s_math)[-K:][::-1]
        top_code = np.argsort(s_code)[-K:][::-1]
        
        math_manifest["experts"].append({
            "layer": l,
            "indices": top_math.tolist(),
            "scores": s_math[top_math].tolist()
        })
        
        code_manifest["experts"].append({
            "layer": l,
            "indices": top_code.tolist(),
            "scores": s_code[top_code].tolist()
        })
        
        for idx in top_math: total_math_experts.add(f"L{l}_E{idx}")
        for idx in top_code: total_code_experts.add(f"L{l}_E{idx}")
        
        t1 = time.time()
        
        # Summary for standard output
        math_mean = np.mean(s_math[top_math])
        code_mean = np.mean(s_code[top_code])
        print(f"Layer {l:02d} | Shard: {file_name[-15:]} | Top16 Math Sim: {math_mean:.3f} | Top16 Code Sim: {code_mean:.3f} | Time: {t1-t0:.2f}s")
    
    print("\n4. Saving Manifests and Overlap Analysis...")
    with open("math_experts_manifest.json", "w") as f:
        json.dump(math_manifest, f, indent=2)
    with open("code_experts_manifest.json", "w") as f:
        json.dump(code_manifest, f, indent=2)
        
    overlap = total_math_experts.intersection(total_code_experts)
    overlap_rate = len(overlap) / (48 * K) * 100
    
    analysis = {
        "total_math_selected": len(total_math_experts),
        "total_code_selected": len(total_code_experts),
        "overlap_count": len(overlap),
        "overlap_rate_percent": round(overlap_rate, 2),
        "overlapping_experts": list(overlap)
    }
    with open("expert_overlap_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
        
    print(f"Overlap Rate: {overlap_rate:.1f}%")
    if overlap_rate < 30.0:
        print("✅ Domain-specific expert differentiation is highly successful.")
    else:
        print("⚠️ High overlap detected (Polymath experts dominating).")
        
    print("\nProfiling complete. Memory footprint stayed minimal.")

if __name__ == "__main__":
    main()
