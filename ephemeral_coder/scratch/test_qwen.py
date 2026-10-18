import sys
import os
sys.path.insert(0, os.path.abspath("."))
import time
from engine.ollama_client import query_ollama

t0 = time.time()
r = query_ollama('qwen2.5-coder:3b', 'def add(a, b): return a + b')
t_elapsed = time.time() - t0
print(f"Time: {t_elapsed:.1f}s, len: {len(r or '')}")
