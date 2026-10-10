"""
【ファイル概要】
export_micro_topo.py

RP2350などのSRAM制約の厳しいエッジデバイス向けに、対象ドメインの概念地形を「局所100語展開用」のバイナリ（micro_topo.bin）へエクスポートするPC側スクリプト。
"""

import os
import struct
import numpy as np

def generate_toy_vocabulary(num_words=1000, dim=128):
    """
    Generates a synthetic vocabulary and embeddings for Yes/No sentiment domain.
    """
    np.random.seed(42)
    # Generate random embeddings normalized to unit length
    embeddings = np.random.randn(num_words, dim).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Generate synthetic words (e.g., word_000, word_001...)
    # We will inject some specific words for testing
    vocab = [f"word_{i:03d}" for i in range(num_words)]
    
    # Inject specific domain words
    injections = {
        0: "good", 1: "great", 2: "excellent", 3: "amazing", 4: "yes", 5: "agree",
        10: "bad", 11: "terrible", 12: "awful", 13: "worst", 14: "no", 15: "disagree",
        20: "is", 21: "this", 22: "the", 23: "a", 24: "an", 25: "very"
    }
    for idx, word in injections.items():
        vocab[idx] = word
        
    # Make Yes/No cluster
    yes_vec = np.random.randn(dim).astype(np.float32)
    yes_vec /= np.linalg.norm(yes_vec)
    no_vec = np.random.randn(dim).astype(np.float32)
    no_vec /= np.linalg.norm(no_vec)
    
    # Bias the injected words towards Yes/No
    for idx in range(0, 10):
        embeddings[idx] = (embeddings[idx] + 2.0 * yes_vec)
        embeddings[idx] /= np.linalg.norm(embeddings[idx])
    for idx in range(10, 20):
        embeddings[idx] = (embeddings[idx] + 2.0 * no_vec)
        embeddings[idx] /= np.linalg.norm(embeddings[idx])
        
    return vocab, embeddings, yes_vec, no_vec

def compute_nearest_neighbors(embeddings, k=20):
    """
    Compute k nearest neighbors for each word.
    """
    num_words = embeddings.shape[0]
    neighbors = np.zeros((num_words, k), dtype=np.uint16)
    
    # Brute force cosine similarity
    for i in range(num_words):
        sims = np.dot(embeddings, embeddings[i])
        # Sort descending, ignore self (which is at index 0 after sort)
        top_indices = np.argsort(sims)[::-1][1:k+1]
        neighbors[i] = top_indices.astype(np.uint16)
        
    return neighbors

def export_binary(filepath, vocab, embeddings, neighbors, yes_vec, no_vec):
    """
    Exports to a lightweight binary format designed for MicroPython f.seek()
    
    Format:
    - Magic (4 bytes): b'RPAI'
    - N_WORDS (uint16)
    - DIM (uint16)
    - Vocab Table: N_WORDS * 16 bytes (null-padded ascii)
    - Embeddings: N_WORDS * DIM * 4 bytes (float32)
    - Neighbors: N_WORDS * 20 * 2 bytes (uint16)
    - Yes Vector: DIM * 4 bytes
    - No Vector: DIM * 4 bytes
    """
    n_words = len(vocab)
    dim = embeddings.shape[1]
    k_neighbors = neighbors.shape[1]
    
    with open(filepath, 'wb') as f:
        # Header
        f.write(b'RPAI')
        f.write(struct.pack('<HH', n_words, dim))
        
        # Vocab Table
        for word in vocab:
            # truncate to 15 chars, pad to 16
            b_word = word[:15].encode('ascii', errors='ignore')
            b_word += b'\x00' * (16 - len(b_word))
            f.write(b_word)
            
        # Embeddings
        f.write(embeddings.tobytes())
        
        # Neighbors
        f.write(neighbors.tobytes())
        
        # Yes/No
        f.write(yes_vec.tobytes())
        f.write(no_vec.tobytes())
        
    print(f"Exported {n_words} words (dim={dim}) to {filepath}")
    
    file_size = os.path.getsize(filepath)
    print(f"Total File Size: {file_size / 1024:.1f} KB")
    
if __name__ == "__main__":
    print("Generating Micro Topography...")
    dim = 128
    vocab, embeddings, yes_vec, no_vec = generate_toy_vocabulary(num_words=1000, dim=dim)
    neighbors = compute_nearest_neighbors(embeddings, k=20)
    
    export_binary("micro_topo.bin", vocab, embeddings, neighbors, yes_vec, no_vec)
