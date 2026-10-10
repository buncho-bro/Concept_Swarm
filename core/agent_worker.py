import time
import numpy as np
import multiprocessing as mp
from typing import Dict
from shallow_extractor import GreatCircleField, _normalize_rows
from shared_memory_loader import attach_shared_extractor
from latent_blackboard import LatentBlackboard

def run_agent_a(
    model_id: str,
    embed_shape: tuple,
    embed_dtype: np.dtype,
    bb_name: str,
    sync: Dict[str, mp.Event],
    input_text: str
):
    """
    Agent A (Proposer):
    Extracts the concept core from the input text using Great Circle Interference,
    and proposes it to the Latent Blackboard.
    """
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # 1. Attach to shared memory (0 RAM copy)
    ex = attach_shared_extractor(model_id, embed_shape, embed_dtype)
    # Get num_agents from sync size or just hardcode to 10 for Phase 3 since we are testing 10
    num_agents = len(sync["start_events"])
    bb = LatentBlackboard(bb_name, embed_shape[1], num_agents, create=False)
    
    sync["ready_events"][0].set()
    # Wait for start signal
    sync["start_events"][0].wait()
    t0 = time.perf_counter()
    
    # 2. Extract initial concept core
    ids, tokens, V = ex.token_vectors(input_text)
    field = GreatCircleField(V)
    
    # Fast heuristic: just take the top pairwise focus
    foci, _ = field.pairwise_foci(max_pairs=10, rng=np.random.default_rng(42))
    
    # Gradient ascent to find local maximum (Physical convergence)
    x = foci[0:1] # [1, D]
    for _ in range(10):
        grad, _ = field.riemannian_grad(x, sigma=0.1)
        x = _normalize_rows(x + 0.5 * grad)
        
    x_0 = x[0]
    
    # 3. Write to blackboard (Non-verbal communication)
    bb.write_proposal(0, x_0)
    bb.write_tokens(ids)
    
    t_el = time.perf_counter() - t0
    
    # Signal done
    sync["done_events"][0].set()
    print(f"[Agent A] Proposal submitted. ({t_el*1000:.1f}ms)", flush=True)


def run_agent_b(
    model_id: str,
    embed_shape: tuple,
    embed_dtype: np.dtype,
    bb_name: str,
    sync: Dict[str, mp.Event],
    verifier_text: str
):
    """
    Agent B (Verifier):
    Reads Agent A's proposal directly from the blackboard.
    Exposes the proposal to its own Concept Topography (verifier_text)
    and applies physical simulation (gradient descent/ascent) to adjust it.
    """
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    import time
    time.sleep(1.0)
    
    # 1. Attach to shared memory
    ex = attach_shared_extractor(model_id, embed_shape, embed_dtype)
    num_agents = len(sync["start_events"])
    bb = LatentBlackboard(bb_name, embed_shape[1], num_agents, create=False)
    
    # Pre-compute verifier topography
    _, _, V_B = ex.token_vectors(verifier_text)
    field_B = GreatCircleField(V_B)
    
    sync["ready_events"][1].set()
    # Wait for Agent A to finish proposing
    t0 = time.perf_counter()
    sync["done_events"][0].wait()
    t_init = time.perf_counter()
    
    # 2. Read proposal x_0 from Latent Blackboard (No text involved)
    x_0 = bb.read_proposal(0)
    x_1 = x_0.copy().reshape(1, -1)
    
    # 3. Physical Simulation (Convergence on Agent B's Topography)
    # Exert "gravity" from Agent B's perspective onto Agent A's concept
    # INCREASE SIGMA to 0.5 so it can pull from further away
    for _ in range(5):
        grad, _ = field_B.riemannian_grad(x_1, sigma=0.5)
        x_1 = _normalize_rows(x_1 + 0.3 * grad)
        
    x_1 = x_1[0]
    
    # 4. Write back adjusted vector
    bb.write_proposal(1, x_1)
    
    t_el = time.perf_counter() - t0
    t_sim = time.perf_counter() - t_init
    
    # Signal done
    sync["done_events"][1].set()
    print(f"[Agent B] Verification completed. (Total: {t_el*1000:.1f}ms, Sim: {t_sim*1000:.1f}ms)", flush=True)
