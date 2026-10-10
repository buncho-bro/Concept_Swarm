"""
【ファイル概要】
multi_agent_swarm.py

10体の専門エージェント群を並列稼働させ、各々のポテンシャル地形を用いた大円干渉（物理シミュレーション）で合議ベクトルを算出するオーケストレーター。
"""

import time
import os
import numpy as np
import multiprocessing as mp
from typing import Dict, List
from shallow_extractor import GreatCircleField, _normalize_rows
from shared_memory_loader import attach_shared_extractor
from latent_blackboard import LatentBlackboard

def worker_expert(worker_id: int, model_id: str, embed_shape: tuple, embed_dtype: np.dtype, bb_name: str, sync: Dict[str, mp.Event], domain_text: str):
    """
    Worker 0-7: Domain Experts.
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import time
    time.sleep(worker_id * 0.2) # Stagger initialization
    
    ex = attach_shared_extractor(model_id, embed_shape, embed_dtype)
    num_agents = len(sync["start_events"])
    bb = LatentBlackboard(bb_name, embed_shape[1], num_agents, create=False)
    
    # Pre-compute expert topography
    _, _, V_E = ex.token_vectors(domain_text)
    field_E = GreatCircleField(V_E)
    
    sync["ready_events"][worker_id].set()
    
    # Wait for Proposer (Master) to write initial vector
    sync["start_events"][worker_id].wait()
    
    # Read the initial global focus
    x_0 = bb.read_proposal(0) # Master writes to slot 0 for simplicity
    x_i = x_0.copy().reshape(1, -1)
    
    # Physical Simulation: Gradient descent into the expert's knowledge basin
    for _ in range(3):
        grad, _ = field_E.riemannian_grad(x_i, sigma=0.5)
        x_i = _normalize_rows(x_i + 0.3 * grad)
        
    bb.write_proposal(worker_id, x_i[0])
    sync["done_events"][worker_id].set()


def worker_critic(worker_id: int, model_id: str, embed_shape: tuple, embed_dtype: np.dtype, bb_name: str, sync: Dict[str, mp.Event]):
    """
    Worker 8: Logic/Critic (Applies repulsive forces for contradictions)
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import time
    time.sleep(worker_id * 0.2)
    
    ex = attach_shared_extractor(model_id, embed_shape, embed_dtype)
    bb = LatentBlackboard(bb_name, embed_shape[1], len(sync["start_events"]), create=False)
    
    sync["ready_events"][worker_id].set()
    sync["start_events"][worker_id].wait()
    
    x_0 = bb.read_proposal(0)
    x_i = x_0.copy().reshape(1, -1)
    
    # Critic repels extreme outliers or illogical topologies
    # Simplified: applies a slight regularization/smoothing back to the mean
    x_i = _normalize_rows(x_i - 0.05 * x_i) 
    
    bb.write_proposal(worker_id, x_i[0])
    sync["done_events"][worker_id].set()


def worker_arbiter(worker_id: int, model_id: str, embed_shape: tuple, embed_dtype: np.dtype, bb_name: str, sync: Dict[str, mp.Event]):
    """
    Worker 9: Arbiter / Integrator.
    Waits for all experts (1-8) to finish, then integrates their proposals into a final x*
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import time
    time.sleep(worker_id * 0.2)
    
    ex = attach_shared_extractor(model_id, embed_shape, embed_dtype)
    bb = LatentBlackboard(bb_name, embed_shape[1], len(sync["start_events"]), create=False)
    
    sync["ready_events"][worker_id].set()
    sync["start_events"][worker_id].wait()
    
    # Wait for workers 1 to 8 to finish
    for i in range(1, 9):
        sync["done_events"][i].wait()
        
    # Read all proposals
    proposals = []
    for i in range(1, 9):
        proposals.append(bb.read_proposal(i))
        
    proposals = np.stack(proposals) # [8, D]
    
    # Global integration (Geodesic Resonance)
    # Simply normalized sum of all expert vectors
    x_star = np.sum(proposals, axis=0)
    x_star = x_star / np.linalg.norm(x_star)
    
    bb.write_global_focus(x_star)
    sync["done_events"][worker_id].set()
