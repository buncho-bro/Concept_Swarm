import time
import os
import psutil
import torch
import numpy as np
from typing import List, Tuple

from shallow_benchmark_data import SINGLE, MULTI, ANCHORS
from shallow_extractor import ShallowConceptExtractor, InterferenceConfig, cuda_was_initialized
from high_dim_concept_core import HighDimConceptCore, HDParams

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def run_benchmark():
    print("==================================================")
    print("🚀 Initializing Concept Extractors (CPU ONLY)...")
    print("==================================================")
    
    # Enforce CPU only
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    t_start = time.perf_counter()
    ex = ShallowConceptExtractor(verbose=False)
    print(f"Model loaded in {time.perf_counter() - t_start:.2f}s")
    
    # ---------------------------------------------------------
    # 1. Single-Domain Benchmark (Niche Term Retention)
    # ---------------------------------------------------------
    print("\n[1/2] Running Single-Domain Benchmark (48 sentences)...")
    
    metrics = {
        "mean": {"sim": [], "time": []},
        "physics": {"sim": [], "time": []},
        "great_circle": {"sim": [], "time": []}
    }
    
    for text, domain, niche_term in SINGLE:
        ids, toks, V = ex.token_vectors(text)
        
        # Get niche term index
        niche_idx = -1
        for i, t in enumerate(toks):
            if niche_term.lower() in t.lower():
                niche_idx = i
                break
        
        if niche_idx == -1:
            continue
            
        def get_rr(scores: np.ndarray, target_idx: int) -> float:
            ranks = np.argsort(-scores)
            rank = np.where(ranks == target_idx)[0][0] + 1
            return 1.0 / rank

        # 1. Conventional (Mean)
        t0 = time.perf_counter()
        mean_vec = V.mean(0)
        mean_vec /= max(np.linalg.norm(mean_vec), 1e-12)
        t_mean = (time.perf_counter() - t0) * 1000
        scores_mean = V @ mean_vec
        metrics["mean"]["sim"].append(get_rr(scores_mean, niche_idx))
        metrics["mean"]["time"].append(t_mean)
        
        # 2. Physics Simulation
        t0 = time.perf_counter()
        params = HDParams(dim=V.shape[1], sigma=0.8, marble_steps=30, depth_gain=0.1)
        model = HighDimConceptCore(params)
        model.erode(torch.from_numpy(V).float())
        model.weather()
        marble_start = torch.from_numpy(mean_vec).float()
        core_tensor, _, _ = model.roll_marble(marble_start)
        physics_vec = core_tensor.numpy()
        physics_vec /= max(np.linalg.norm(physics_vec), 1e-12)
        t_physics = (time.perf_counter() - t0) * 1000
        scores_phys = V @ physics_vec
        metrics["physics"]["sim"].append(get_rr(scores_phys, niche_idx))
        metrics["physics"]["time"].append(t_physics)
        
        # 3. Great Circle Interference
        t0 = time.perf_counter()
        gc_result = ex.extract(text)
        t_gc = (time.perf_counter() - t0) * 1000
        # Best reciprocal rank among all cores
        best_rr = max([get_rr(c.contributions, niche_idx) for c in gc_result.cores])
        metrics["great_circle"]["sim"].append(best_rr)
        metrics["great_circle"]["time"].append(t_gc)

    # ---------------------------------------------------------
    # 2. Multi-Domain Benchmark (Concept Separation)
    # ---------------------------------------------------------
    print("\n[2/2] Running Multi-Domain Separation Test (16 sentences)...")
    success_2 = 0
    success_3 = 0
    for text, domains in MULTI:
        # Request up to 3 cores
        res = ex.extract(text, config=InterferenceConfig(k_max=3, sigma_scale=0.4))
        cores = res.cores
        if len(cores) >= 2:
            success_2 += 1
        if len(cores) == 3:
            success_3 += 1

    # ---------------------------------------------------------
    # Results formatting
    # ---------------------------------------------------------
    print("\n==================================================")
    print(" 🏆 SHALLOW EXTRACTION BENCHMARK RESULTS")
    print("==================================================")
    
    print(f"{'Method':<25} | {'Niche Retention (MRR)':<25} | {'CPU Time / Sent':<15}")
    print("-" * 70)
    
    m_mean = np.mean(metrics['mean']['sim'])
    t_mean = np.mean(metrics['mean']['time'])
    print(f"{'Conventional (Mean)':<25} | {m_mean:<25.4f} | {t_mean:<10.2f} ms")
    
    m_phys = np.mean(metrics['physics']['sim'])
    t_phys = np.mean(metrics['physics']['time'])
    print(f"{'Physics (Concept Core)':<25} | {m_phys:<25.4f} | {t_phys:<10.2f} ms")
    
    m_gc = np.mean(metrics['great_circle']['sim'])
    t_gc = np.mean(metrics['great_circle']['time'])
    print(f"{'Great Circle Interference':<25} | {m_gc:<25.4f} | {t_gc:<10.2f} ms")
    print("==================================================")
    
    print("\nMulti-Domain Concept Separation (Great Circle):")
    print(f"- Successfully identified >= 2 cores: {success_2}/{len(MULTI)} cases")
    print(f"- Identified 3 cores: {success_3}/{len(MULTI)} cases")
    
    mem_usage = get_memory_usage()
    cuda_init = cuda_was_initialized()
    print("\nHardware & Footprint:")
    print(f"- Total Memory Consumed (RAM): {mem_usage:.1f} MB")
    print(f"- CUDA Initialized: {cuda_init}")
    if not cuda_init:
        print("- VRAM Consumed: 0 MB (Strictly CPU-Only)")
    
    print("==================================================\n")

if __name__ == '__main__':
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    run_benchmark()
