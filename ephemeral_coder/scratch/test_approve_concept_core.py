import sys
import urllib.request
import json
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

url_approve = "http://localhost:8085/api/chat/approve"
spec_id = "spec_15a8bb"

print(f"=== Approving Spec: {spec_id} ===")
req = urllib.request.Request(
    url_approve,
    data=json.dumps({"spec_id": spec_id}).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)

t0 = time.time()
with urllib.request.urlopen(req, timeout=600) as resp:
    res = json.loads(resp.read().decode('utf-8'))
t_exec = time.time() - t0

print(f"Approval executed in {t_exec:.1f}s")
print(f"Success: {res.get('success')}")
print(f"Total lines generated: {res.get('line_count')}")
print("\n--- Worker Logs ---")
for log in res.get("logs", []):
    print(" ", log)

print("\n--- Final Code Snippet ---")
code = res.get("final_code", "")
print(code[:1500])
if len(code) > 1500:
    print("\n... [truncated] ...")
