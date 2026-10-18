import sys
import os
import urllib.request
import json
import time
import subprocess

payload = {
    "model": "llama3.1:8b",
    "prompt": "Count from 1 to 50 in Python list.",
    "stream": False,
    "options": {
        "num_ctx": 8192
    }
}

print("Testing llama3.1:8b with options.num_ctx = 8192...")
req = urllib.request.Request(
    "http://localhost:11434/api/generate",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
t0 = time.time()
with urllib.request.urlopen(req, timeout=120) as res:
    data = json.loads(res.read().decode("utf-8"))
t1 = time.time() - t0
print(f"Finished in {t1:.2f}s, Response length: {len(data.get('response', ''))}")

print("\nChecking 'ollama ps':")
subprocess.run(["ollama", "ps"], shell=True)
