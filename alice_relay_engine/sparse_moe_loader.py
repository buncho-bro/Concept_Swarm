import requests
import json
import struct
import numpy as np
import torch
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

class SparseMoELoader:
    """
    On-Demand Sparse MoE Loader for AliceAI 80B.
    Selectively downloads only specific experts via HTTP Range Streaming.
    Converts downloaded bytes directly to PyTorch tensors for the physics engine.
    """
    def __init__(self, repo_id="yandex/AliceAI-Foundation-80B-A3B-Base"):
        self.repo_id = repo_id
        self.index_url = f"https://huggingface.co/{repo_id}/resolve/main/model.safetensors.index.json"
        
        r = requests.get(self.index_url)
        if r.status_code != 200:
            raise Exception("Failed to fetch safetensors index.")
        self.weight_map = r.json()["weight_map"]
        self.header_cache = {}
        
    def _get_safetensors_header(self, file_name):
        if file_name in self.header_cache:
            return self.header_cache[file_name]
            
        url = f"https://huggingface.co/{self.repo_id}/resolve/main/{file_name}"
        r = requests.get(url, headers={"Range": "bytes=0-7"})
        n = struct.unpack('<Q', r.content)[0]
        r2 = requests.get(url, headers={"Range": f"bytes=8-{8+n-1}"})
        header = json.loads(r2.content)
        
        self.header_cache[file_name] = {
            "url": url,
            "header": header,
            "data_start": 8 + n
        }
        return self.header_cache[file_name]

    def _bytes_to_tensor(self, data, dtype_str, shape):
        if dtype_str == 'BF16':
            arr = np.frombuffer(data, dtype=np.uint16)
            arr_f32 = np.zeros(arr.size, dtype=np.uint32)
            arr_f32[:] = arr
            arr_f32 = arr_f32 << 16
            f32_np = arr_f32.view(np.float32).reshape(shape)
            return torch.from_numpy(f32_np)
        elif dtype_str == 'F16':
            f16_np = np.frombuffer(data, dtype=np.float16).reshape(shape).astype(np.float32)
            return torch.from_numpy(f16_np)
        else:
            raise Exception(f"Unsupported dtype {dtype_str}")

    def fetch_expert_slice(self, tensor_name, expert_idx):
        if tensor_name not in self.weight_map:
            raise Exception(f"Tensor {tensor_name} not found in map.")
            
        file_name = self.weight_map[tensor_name]
        info = self._get_safetensors_header(file_name)
        url = info["url"]
        t_info = info["header"][tensor_name]
        
        start_offset = t_info["data_offsets"][0]
        full_shape = t_info["shape"] # [512, D_intermediate, D_hidden]
        
        # Bytes per element = 2 (for BF16/F16)
        expert_size_elements = full_shape[1] * full_shape[2]
        expert_size_bytes = expert_size_elements * 2
        
        slice_start = start_offset + expert_idx * expert_size_bytes
        slice_end = slice_start + expert_size_bytes - 1
        
        data_start = info["data_start"]
        r = requests.get(url, headers={"Range": f"bytes={data_start + slice_start}-{data_start + slice_end}"})
        
        if r.status_code not in (200, 206):
            raise Exception(f"Failed to fetch {tensor_name} for expert {expert_idx}")
            
        return self._bytes_to_tensor(r.content, t_info["dtype"], (full_shape[1], full_shape[2]))

    def hot_swap_experts(self, manifest_file, max_layers=48):
        """
        Reads a manifest JSON and concurrently downloads all specified experts.
        Returns a nested dictionary: {layer_idx: {expert_idx: {'gate_up': Tensor, 'down': Tensor}}}
        """
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
            
        print(f"[Sparse Loader] Hot-swapping domain: {manifest['domain'].upper()}")
        t0 = time.time()
        
        loaded_layers = {}
        fetch_tasks = []
        
        # We will use ThreadPoolExecutor to blast HTTP requests concurrently
        with ThreadPoolExecutor(max_workers=32) as executor:
            for layer_data in manifest["experts"][:max_layers]:
                l = layer_data["layer"]
                indices = layer_data["indices"]
                loaded_layers[l] = {}
                
                name_gate_up = f"model.layers.{l}.mlp.experts.gate_up_proj"
                name_down = f"model.layers.{l}.mlp.experts.down_proj"
                
                for idx in indices:
                    # Submit gate_up
                    fetch_tasks.append((
                        l, idx, 'gate_up',
                        executor.submit(self.fetch_expert_slice, name_gate_up, idx)
                    ))
                    # Submit down
                    fetch_tasks.append((
                        l, idx, 'down',
                        executor.submit(self.fetch_expert_slice, name_down, idx)
                    ))
                    loaded_layers[l][idx] = {}
            
            # Wait for all tasks
            completed = 0
            total = len(fetch_tasks)
            for task_info in fetch_tasks:
                l, idx, proj_type, future = task_info
                tensor = future.result()
                loaded_layers[l][idx][proj_type] = tensor
                completed += 1
                if completed % 100 == 0:
                    print(f"  Downloaded {completed}/{total} expert tensors...")

        t1 = time.time()
        
        # Calculate memory footprint
        total_bytes = 0
        for l in loaded_layers:
            for idx in loaded_layers[l]:
                total_bytes += loaded_layers[l][idx]['gate_up'].element_size() * loaded_layers[l][idx]['gate_up'].nelement()
                total_bytes += loaded_layers[l][idx]['down'].element_size() * loaded_layers[l][idx]['down'].nelement()
                
        print(f"[Sparse Loader] Hot-swap complete in {t1-t0:.2f}s.")
        print(f"[Sparse Loader] RAM Footprint: {total_bytes / (1024**3):.2f} GB")
        
        return loaded_layers
