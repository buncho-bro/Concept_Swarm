import numpy as np
from multiprocessing import shared_memory
from typing import Dict, Tuple, Optional
import os
import torch
from shallow_extractor import ShallowConceptExtractor, ShallowWeights, InterferenceConfig

class SharedTensorManager:
    """Manages NumPy arrays in POSIX/Windows shared memory."""
    def __init__(self, name: str, shape: Tuple[int, ...], dtype: np.dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.size = int(np.prod(self.shape) * np.dtype(self.dtype).itemsize)
        self.shm: Optional[shared_memory.SharedMemory] = None
        self.array: Optional[np.ndarray] = None

    def create(self, data: np.ndarray) -> None:
        """Create the shared memory and populate it with data."""
        if data.shape != self.shape or data.dtype != self.dtype:
            raise ValueError(f"Data shape/dtype mismatch. Expected {self.shape} {self.dtype}, got {data.shape} {data.dtype}")
        
        try:
            self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.size)
        except FileExistsError:
            # Clean up dangling shared memory from previous crashes
            self.shm = shared_memory.SharedMemory(name=self.name)
            self.shm.unlink()
            self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.size)

        self.array = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)
        np.copyto(self.array, data)

    def attach(self) -> None:
        """Attach to an existing shared memory block (Read-Only)."""
        self.shm = shared_memory.SharedMemory(name=self.name, create=False)
        self.array = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)
        self.array.flags.writeable = False  # Ensure 100% read-only

    def close(self) -> None:
        if self.shm is not None:
            self.shm.close()

    def unlink(self) -> None:
        if self.shm is not None:
            try:
                self.shm.unlink()
            except Exception:
                pass


def setup_shared_extractor(model_id: str = "zai-org/GLM-5.3") -> Dict[str, SharedTensorManager]:
    """
    [Master Process] Loads the heavy embeddings ONCE, registers them into Shared Memory.
    Returns a dictionary of SharedTensorManagers.
    """
    print(f"[Master] Initializing base ShallowConceptExtractor for {model_id}...")
    ex = ShallowConceptExtractor(model_id=model_id, verbose=False)
    
    # Ex doesn't load W_embed entirely into memory by default, we construct it:
    shape = (ex.weights.vocab_rows, ex.weights.dim)
    dtype = np.float32
    
    embed_mgr = SharedTensorManager("W_embed_shm", shape, dtype)
    
    print(f"[Master] Constructing W_embed {shape} into Shared Memory...")
    # Temporarily create local array or just populate directly
    # Wait, embed_mgr.create() needs the data. Let's create it.
    W_embed_full = np.zeros(shape, dtype=dtype)
    for s, block in ex.weights.iter_vocab_blocks():
        W_embed_full[s:s+len(block)] = block
    
    embed_mgr.create(W_embed_full)
    
    # Vocab mean is cached
    vocab_mean = ex.weights._vocab_mean()
    mean_mgr = SharedTensorManager("vocab_mean_shm", vocab_mean.shape, vocab_mean.dtype)
    
    print(f"[Master] Writing vocab_mean {vocab_mean.shape} to Shared Memory...")
    mean_mgr.create(vocab_mean)
    
    return {"W_embed": embed_mgr, "vocab_mean": mean_mgr}


def attach_shared_extractor(model_id: str, embed_shape: Tuple[int, int], embed_dtype: np.dtype) -> ShallowConceptExtractor:
    """
    [Child Process] Creates a ShallowConceptExtractor but hijacks its weights
    to point to the Read-Only Shared Memory arrays. Zero RAM duplication!
    """
    embed_mgr = SharedTensorManager("W_embed_shm", embed_shape, embed_dtype)
    mean_mgr = SharedTensorManager("vocab_mean_shm", (embed_shape[1],), np.float64)
    
    embed_mgr.attach()
    mean_mgr.attach()
    
    # Hijack the original class temporarily
    import shallow_extractor
    original_cls = shallow_extractor.ShallowWeights
    
    class HijackedShallowWeights(original_cls):
        def _load_weights(self):
            # Still load layer norm (very small 6KB)
            super()._load_weights()
            self._mu = mean_mgr.array
            self.tensor_bytes = 0  # No local footprint
            
        def raw_rows(self, ids):
            # Read directly from shared memory, bypassing disk and row_cache
            return embed_mgr.array[list(ids)].astype(np.float64)
            
        def _vocab_mean(self, chunk=8192):
            return mean_mgr.array

    shallow_extractor.ShallowWeights = HijackedShallowWeights
    
    ex = ShallowConceptExtractor(model_id=model_id, verbose=False)
    
    # Restore
    shallow_extractor.ShallowWeights = original_cls
    
    # Keep managers alive
    ex._shared_mgrs = [embed_mgr, mean_mgr]
    
    return ex
