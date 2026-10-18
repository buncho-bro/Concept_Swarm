import sys
import os
sys.path.insert(0, os.path.abspath("."))
import time
import subprocess
from engine.ollama_client import query_ollama

print("1. Running query to llama3.1:8b...")
t0 = time.time()
res = query_ollama("llama3.1:8b", "Count from 1 to 50 in Python list.")
t1 = time.time() - t0
print(f"Query finished in {t1:.2f}s, Response length: {len(res or '')}")

print("\n2. Checking 'ollama ps' output:")
subprocess.run(["ollama", "ps"], shell=True)
