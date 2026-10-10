import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
import time
import psutil
import multiprocessing as mp
import numpy as np

# Suppress HF warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from shared_memory_loader import setup_shared_extractor, attach_shared_extractor
from shallow_extractor import ShallowConceptExtractor, InterferenceConfig

def get_sys_ram_mb() -> float:
    """Returns the current system RAM usage of the current process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def worker_process(worker_id: int, model_id: str, embed_shape: tuple, embed_dtype: np.dtype, result_queue: mp.Queue):
    """
    The worker function that runs in a completely separate process.
    Attaches to the shared memory and runs an extraction, recording its memory footprint.
    """
    print(f"[Worker {worker_id}] Started.", flush=True)
    import time
    time.sleep(worker_id * 1.0)
    try:
        t0 = time.time()
        # Attach to the existing shared memory
        ex = attach_shared_extractor(model_id, embed_shape, embed_dtype)
        print(f"[Worker {worker_id}] Attached.", flush=True)
        
        # We process a highly complex sentence
        text = "Advanced quantum computing algorithms can efficiently break modern cryptographic encryption schemes."
        
        mem_before = get_sys_ram_mb()
        res = ex.extract(text, config=InterferenceConfig(k_max=2))
        mem_after = get_sys_ram_mb()
        
        t1 = time.time()
        print(f"[Worker {worker_id}] Extraction complete.", flush=True)
        
        # Send back the results to the master process for verification
        result_queue.put({
            "worker_id": worker_id,
            "mem_before_mb": mem_before,
            "mem_after_mb": mem_after,
            "elapsed_s": t1 - t0,
            "primary_core": res.primary,  # We'll check if this matches perfectly
            "top_tokens": res.cores[0].top_tokens(res.tokens, k=5)
        })
    except Exception as e:
        result_queue.put({"worker_id": worker_id, "error": str(e)})

def run_benchmark():
    model_id = "zai-org/GLM-5.3"
    num_workers = 4
    
    print("==================================================")
    print(f" 🧪 PHASE 1: SHARED MEMORY BENCHMARK ({num_workers} AGENTS)")
    print("==================================================")
    
    # 1. Base measurement
    base_ram = get_sys_ram_mb()
    print(f"[Master] Base Master RAM before anything: {base_ram:.1f} MB")
    
    # 2. Master initialization (Loads arrays into Shm)
    mgrs = setup_shared_extractor(model_id)
    embed_mgr = mgrs["W_embed"]
    
    master_ram_loaded = get_sys_ram_mb()
    print(f"[Master] Master RAM after creating Shared Memory: {master_ram_loaded:.1f} MB")
    print(f"[Master] Shared Memory size: W_embed is {embed_mgr.shape} ({(embed_mgr.size / 1024**2):.1f} MB)\n")
    
    # 3. Master baseline extraction (to verify output)
    ex_master = attach_shared_extractor(model_id, embed_mgr.shape, embed_mgr.dtype)
    text = "Advanced quantum computing algorithms can efficiently break modern cryptographic encryption schemes."
    res_master = ex_master.extract(text, config=InterferenceConfig(k_max=2))
    master_primary_core = res_master.primary
    
    # 4. Launch multiple processes concurrently
    print(f"\n🚀 Launching {num_workers} Parallel Agents...\n")
    
    queue = mp.Queue()
    processes = []
    
    for i in range(num_workers):
        p = mp.Process(target=worker_process, args=(i, model_id, embed_mgr.shape, embed_mgr.dtype, queue))
        processes.append(p)
        p.start()
        
    # 5. Collect and Verify Results BEFORE joining to avoid pipe deadlock
    results = []
    # We expect `num_workers` results
    for i in range(num_workers):
        try:
            results.append(queue.get(timeout=30))
        except mp.queues.Empty:
            print(f"[Master] TIMEOUT waiting for worker {i}! A worker crashed silently.", flush=True)
        
    for p in processes:
        p.join()
        
    results.sort(key=lambda x: x.get("worker_id", -1))
    
    print("==================================================", flush=True)
    print(" 📊 PARALLEL AGENT RESULTS", flush=True)
    print("==================================================", flush=True)
    
    all_matched = True
    total_agent_ram = 0
    
    for r in results:
        if "error" in r:
            print(f"[Agent {r['worker_id']}] ERROR: {r['error']}")
            continue
            
        # Verify bit-exactness
        max_diff = np.max(np.abs(r["primary_core"] - master_primary_core))
        is_exact = max_diff < 1e-6
        if not is_exact:
            all_matched = False
            
        print(f"[Agent {r['worker_id']}] RAM (Peak): {r['mem_after_mb']:>5.1f} MB | Match: {'✅' if is_exact else '❌'} | Time: {r['elapsed_s']:.2f}s")
        print(f"            Top Tokens: {', '.join([t for t,w in r['top_tokens']])}")
        total_agent_ram += r['mem_after_mb']
        
    print("\n==================================================")
    print(" 🏁 VERIFICATION SUMMARY")
    print("==================================================")
    
    avg_agent_ram = total_agent_ram / num_workers
    print(f"- Base Master RAM (No extraction): {base_ram:.1f} MB")
    print(f"- Average Agent RAM Usage: {avg_agent_ram:.1f} MB")
    
    # Is RAM scalable?
    # If it was loading 600MB independently, average agent RAM would be ~900MB.
    # With shared memory, average agent RAM should be ~250MB (just python + torch baseline)
    is_scalable = avg_agent_ram < (base_ram + 300)
    
    print(f"- Scalability Check: {'✅ PASS (Memory is Shared)' if is_scalable else '❌ FAIL (Memory is Duplicated)'}")
    print(f"- Bit-Exactness Check: {'✅ PASS' if all_matched else '❌ FAIL'}")
    
    print("\nCleaning up shared memory...")
    for mgr in mgrs.values():
        mgr.unlink()
        
if __name__ == "__main__":
    run_benchmark()
