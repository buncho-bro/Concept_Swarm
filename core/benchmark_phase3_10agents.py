"""
【ファイル概要】
benchmark_phase3_10agents.py

10体並列エージェントの合議とテキストデコードの全パイプラインをテストするための実行スクリプト。
"""

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
from multi_agent_swarm import worker_expert, worker_critic, worker_arbiter
from agent_worker import run_agent_a # We can reuse Agent A as Proposer
from dual_stream_decoder import DualStreamDecoder

def get_sys_ram_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

def run_phase3_benchmark():
    model_id = "zai-org/GLM-5.3"
    num_agents = 10
    
    print("==================================================")
    print(" 🧪 PHASE 3: 10-AGENT SWARM & DUAL-STREAM DECODER")
    print("==================================================")
    
    base_ram = get_sys_ram_mb()
    
    # 1. Setup Shared Extractors
    print("[Master] Setting up Shared Memory Extractors...")
    mgrs = setup_shared_extractor(model_id)
    embed_mgr = mgrs["W_embed"]
    
    bb_name = "latent_blackboard_shm_p3"
    bb = LatentBlackboard(bb_name, embed_mgr.shape[1], num_agents, create=True)
    sync = create_debate_sync_events(num_agents)
    
    print(f"[Master] RAM after setup: {get_sys_ram_mb():.1f} MB (Base was {base_ram:.1f} MB)")
    
    # Text Inputs
    input_text = "Machine learning algorithms applied to high-frequency trading."
    domains = [
        "Software engineering and algorithms.",
        "Financial markets and volatility.",
        "Statistical arbitrage and risk.",
        "Latency and network infrastructure.",
        "Regulatory compliance in finance.",
        "Data streaming and Kafka.",
        "GPU acceleration for neural networks.",
    ]
    
    # 2. Launch 10 Agents
    print(f"\n🚀 Launching {num_agents} Parallel Agents (Strict VRAM 0MB)...")
    processes = []
    
    # Proposer (Worker 0)
    p = mp.Process(target=run_agent_a, args=(model_id, embed_mgr.shape, embed_mgr.dtype, bb_name, sync, input_text))
    processes.append(p)
    
    # Experts (Worker 1-7)
    for i in range(1, 8):
        p = mp.Process(target=worker_expert, args=(i, model_id, embed_mgr.shape, embed_mgr.dtype, bb_name, sync, domains[i-1]))
        processes.append(p)
        
    # Critic (Worker 8)
    p = mp.Process(target=worker_critic, args=(8, model_id, embed_mgr.shape, embed_mgr.dtype, bb_name, sync))
    processes.append(p)
    
    # Arbiter (Worker 9)
    p = mp.Process(target=worker_arbiter, args=(9, model_id, embed_mgr.shape, embed_mgr.dtype, bb_name, sync))
    processes.append(p)
    
    for p in processes:
        p.start()
        
    print("[Master] Waiting for all 10 Agents to initialize...")
    for i in range(num_agents):
        sync["ready_events"][i].wait()
        
    print("\n💬 Commencing 10-Agent Latent Swarm Debate...")
    t_start = time.perf_counter()
    
    # 1. Start Proposer
    sync["start_events"][0].set()
    sync["done_events"][0].wait() # Wait for proposal
    
    # 2. Start Experts and Critic simultaneously
    for i in range(1, 9):
        sync["start_events"][i].set()
        
    # 3. Start Arbiter (it will internally wait for Experts to finish)
    sync["start_events"][9].set()
    
    # Wait for Arbiter to finish
    sync["done_events"][9].wait(timeout=20.0)
    
    t_end = time.perf_counter()
    
    if not sync["done_events"][9].is_set():
        print("❌ ERROR: 10-Agent Swarm timed out!")
        for p in processes: p.terminate()
        return
        
    x_star = bb.global_focus
    token_keys = bb.get_tokens()
    
    ex = attach_shared_extractor(model_id, embed_mgr.shape, embed_mgr.dtype)
    keywords = [ex.tokenizer.decode([tk]).strip() for tk in token_keys[:5] if tk != -1]
    
    print("==================================================")
    print(" 🏁 10-AGENT SWARM VERIFICATION SUMMARY")
    print("==================================================")
    print(f"⏱️  Swarm Latency (10 Agents): {(t_end - t_start) * 1000:.2f} ms")
    print(f"💾 Total RAM (Master + 10 Agents): {get_sys_ram_mb():.1f} MB (Expected < 1.0GB limit for agents)")
    print(f"🔍 Extracted Keywords (Stream 2): {keywords}")
    
    # 3. Dual Stream Decoding
    print("\n📝 Initializing Dual-Stream Decoder (GPT-2)...")
    decoder = DualStreamDecoder()
    
    final_text = decoder.decode(x_star, keywords)
    
    print("\n✨ [FINAL OUTPUT (Dual-Stream)]")
    print("-" * 50)
    print(final_text)
    print("-" * 50)
    
    keyword_retention = all([k.lower() in final_text.lower() for k in keywords])
    if keyword_retention:
        print("✅ PASS: All niche keywords successfully retained in the final output!")
    else:
        print("❌ FAIL: Keywords were destroyed or lost by the decoder.")
        
    if (t_end - t_start) * 1000 < 500:
        print("✅ PASS: 10-Agent ultra-low latency achieved (< 500ms).")
    else:
        print("❌ FAIL: Swarm latency too high.")
        
    for p in processes:
        p.join()
        
    bb.unlink()
    for mgr in mgrs.values():
        mgr.unlink()

if __name__ == "__main__":
    run_phase3_benchmark()
