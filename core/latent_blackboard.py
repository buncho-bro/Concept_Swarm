"""
【ファイル概要】
latent_blackboard.py

プロセス間でテキストではなく「潜在ベクトル（概念核座標）」を直接送受信するための共有通信基盤。IPCオーバーヘッドをミリ秒単位に抑えます。
"""

import numpy as np
from multiprocessing import shared_memory
import multiprocessing as mp

class LatentBlackboard:
    """
    Shared memory blackboard for zero-copy, non-verbal communication between agents.
    Agents exchange high-dimensional latent vectors (concept cores) instead of text.
    """
    def __init__(self, name: str, dim: int, num_agents: int, create: bool = False):
        self.name = name
        self.dim = dim
        self.num_agents = num_agents
        
        self.bytes_per_float = 4
        self.bytes_per_int = 4
        
        # Memory Layout
        self.size_global = dim * self.bytes_per_float
        self.size_proposals = num_agents * dim * self.bytes_per_float
        self.size_tokens = 256 * self.bytes_per_int
        self.size_status = num_agents * self.bytes_per_int
        self.size_step = 1 * self.bytes_per_int
        
        self.total_size = (self.size_global + self.size_proposals + 
                           self.size_tokens + self.size_status + self.size_step)
        
        if create:
            try:
                self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.total_size)
            except FileExistsError:
                self.shm = shared_memory.SharedMemory(name=self.name)
                self.shm.unlink()
                self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.total_size)
        else:
            self.shm = shared_memory.SharedMemory(name=self.name)
            
        # Map NumPy arrays directly onto the shared buffer offsets
        offset = 0
        self.global_focus = np.ndarray((dim,), dtype=np.float32, buffer=self.shm.buf, offset=offset)
        offset += self.size_global
        
        self.agent_proposals = np.ndarray((num_agents, dim), dtype=np.float32, buffer=self.shm.buf, offset=offset)
        offset += self.size_proposals
        
        # Fixed slot of up to 256 crucial token IDs to preserve semantic grounding
        self.token_keys = np.ndarray((256,), dtype=np.int32, buffer=self.shm.buf, offset=offset)
        offset += self.size_tokens
        
        self.status_flags = np.ndarray((num_agents,), dtype=np.int32, buffer=self.shm.buf, offset=offset)
        offset += self.size_status
        
        self.step_counter = np.ndarray((1,), dtype=np.int32, buffer=self.shm.buf, offset=offset)
        
        if create:
            # Initialize with zeros
            self.global_focus.fill(0)
            self.agent_proposals.fill(0)
            self.token_keys.fill(-1)
            self.status_flags.fill(0)
            self.step_counter[0] = 0

    def write_proposal(self, agent_id: int, vector: np.ndarray):
        """Zero-copy write to agent's proposal slot."""
        np.copyto(self.agent_proposals[agent_id], vector)
        
    def read_proposal(self, agent_id: int) -> np.ndarray:
        """Zero-copy read from agent's proposal slot."""
        return self.agent_proposals[agent_id]
        
    def write_global_focus(self, vector: np.ndarray):
        np.copyto(self.global_focus, vector)
        
    def write_tokens(self, tokens: list):
        """Preserve up to 256 key token IDs."""
        count = min(len(tokens), 256)
        self.token_keys[:count] = tokens[:count]
        if count < 256:
            self.token_keys[count:] = -1
            
    def get_tokens(self) -> list:
        return [int(t) for t in self.token_keys if t != -1]

    def close(self):
        self.shm.close()
        
    def unlink(self):
        try:
            self.shm.unlink()
        except Exception:
            pass

def create_debate_sync_events(num_agents: int):
    """
    Creates lightweight IPC synchronization events to coordinate non-verbal debate.
    """
    return {
        "ready_events": [mp.Event() for _ in range(num_agents)],
        "start_events": [mp.Event() for _ in range(num_agents)],
        "done_events": [mp.Event() for _ in range(num_agents)],
        "master_done": mp.Event()
    }
