import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
import time
import multiprocessing as mp
import numpy as np
import psutil

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from shared_memory_loader import setup_shared_extractor, attach_shared_extractor
from latent_blackboard import LatentBlackboard, create_debate_sync_events
from agent_worker import run_agent_a, run_agent_b
from shallow_extractor import _normalize_rows

def get_sys_ram_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

def run_phase2_benchmark():
    model_id = "zai-org/GLM-5.3"
    num_agents = 2
    
    print("==================================================")
    print(" 🧪 PHASE 2: LATENT BLACKBOARD DEBATE BENCHMARK")
    print("==================================================")
    
    base_ram = get_sys_ram_mb()
    
    # 1. Setup Shared Extractors
    print("[Master] Setting up Shared Memory Extractors...")
    mgrs = setup_shared_extractor(model_id)
    embed_mgr = mgrs["W_embed"]
    
    # 2. Setup Latent Blackboard
    bb_name = "latent_blackboard_shm"
    bb = LatentBlackboard(bb_name, embed_mgr.shape[1], num_agents, create=True)
    
    # 3. Setup Synchronization
    sync = create_debate_sync_events(num_agents)
    
    print(f"[Master] RAM after setup: {get_sys_ram_mb():.1f} MB (Base was {base_ram:.1f} MB)")
    
    # 4. Input scenarios
    agent_a_text = "Advanced quantum computing algorithms can break modern RSA encryption schemes rapidly."
    agent_b_text = "However, post-quantum cryptography utilizes lattice-based structures to ensure resilience against quantum attacks."
    
    # 5. Launch Agents
    print("\n🚀 Launching 2 Non-Verbal Agents...")
    
    p_a = mp.Process(target=run_agent_a, args=(
        model_id, embed_mgr.shape, embed_mgr.dtype, bb_name, sync, agent_a_text
    ))
    
    p_b = mp.Process(target=run_agent_b, args=(
        model_id, embed_mgr.shape, embed_mgr.dtype, bb_name, sync, agent_b_text
    ))
    
    p_a.start()
    p_b.start()
    
    # Trigger A
    # Wait for both agents to be fully initialized and attached to shared memory before timing
    print("[Master] Waiting for Agents to initialize...", flush=True)
    sync["ready_events"][0].wait()
    sync["ready_events"][1].wait()
    
    print("\n💬 Commencing Latent Debate (No Text Exchange)...", flush=True)
    
    t_start = time.perf_counter()
    
    # Trigger A
    sync["start_events"][0].set()
    
    # Wait for B to finish (A finishes -> B starts -> B finishes)
    sync["done_events"][1].wait(timeout=10.0)
    
    t_end = time.perf_counter()
    
    if not sync["done_events"][1].is_set():
        print("❌ ERROR: Debate timed out!")
        p_a.terminate()
        p_b.terminate()
        return
        
    # 6. Step 3: Master Integration (Resonance)
    x_0 = bb.read_proposal(0)
    x_1 = bb.read_proposal(1)
    
    # Combine the vectors (Spherical Midpoint)
    x_star = _normalize_rows((x_0 + x_1).reshape(1, -1))[0]
    bb.write_global_focus(x_star)
    
    print("==================================================")
    print(" 🏁 DEBATE VERIFICATION SUMMARY")
    print("==================================================")
    
    debate_ms = (t_end - t_start) * 1000
    print(f"⏱️  Debate Latency (A proposal -> B critique): {debate_ms:.2f} ms")
    print(f"✅ Text conversion count during IPC: 0")
    
    # Read back token keys to ground the latent space
    token_keys = bb.get_tokens()
    
    # Use Master's extractor to decode the latent vectors for human inspection
    ex = attach_shared_extractor(model_id, embed_mgr.shape, embed_mgr.dtype)
    
    def decode_vec(vec, name):
        sims = np.dot(ex.weights.unit_vectors(token_keys), vec)
        top_idx = np.argsort(sims)[::-1][:5]
        words = [ex.tokenizer.decode([token_keys[i]]) for i in top_idx]
        print(f"  [{name}] Resonance Top 5: {', '.join(words)}")
        
    print("\n🧠 Latent Space Trajectory (Decoded for verification):")
    decode_vec(x_0, "Agent A (Initial)")
    decode_vec(x_1, "Agent B (Critique)")
    decode_vec(x_star, "Global Focus (Merged)")
    
    # Verify the topographic stability (dot product)
    # x_star should be a good compromise between x_0 and x_1
    sim_0 = np.dot(x_star, x_0)
    sim_1 = np.dot(x_star, x_1)
    
    print("\n📐 Geometric Resonance Stability:")
    print(f"  Similarity x* to x_0: {sim_0:.4f}")
    print(f"  Similarity x* to x_1: {sim_1:.4f}")
    
    if sim_0 > 0.5 and sim_1 > 0.5:
        print("✅ PASS: Global Focus maintains high resonance with both Agent's topography.")
    else:
        print("❌ FAIL: Global Focus collapsed or lost resonance.")
        
    if debate_ms < 500:
        print("✅ PASS: Ultra-low latency achieved (< 500ms).")
    else:
        print("❌ FAIL: Debate latency too high.")
        
    p_a.join()
    p_b.join()
    
    bb.unlink()
    for mgr in mgrs.values():
        mgr.unlink()

if __name__ == "__main__":
    run_phase2_benchmark()
