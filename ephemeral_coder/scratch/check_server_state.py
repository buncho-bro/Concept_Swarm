import urllib.request
import json

try:
    with urllib.request.urlopen("http://localhost:8085/api/state", timeout=5) as r:
        data = json.loads(r.read().decode('utf-8'))
        print("Server state responded:")
        print("Code Store keys:", list(data.get("code_store", {}).keys()))
        for sym, d in data.get("code_store", {}).items():
            print(f"  {sym}: v{d.get('version')} ({len(d.get('code','').splitlines())} lines)")
        print("Failures:", len(data.get("failures", [])))
except Exception as e:
    print("Error querying state:", e)
