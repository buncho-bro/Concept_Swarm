import re

with open(r"C:\Users\lop3l\.gemini\antigravity\scratch\Antigravity_Concept_Swarm\ephemeral_coder\scratch\timeline_summary.txt", "r", encoding="utf-8") as f:
    text = f.read()

steps = text.split("########################################\n### STEP ")

for s in steps:
    if not s.strip():
        continue
    s_id = s.split("\n")[0].strip()
    content = s[len(s_id)+1:].replace("########################################\n", "").strip()
    
    # print step ID and first 300 chars
    print(f"=== STEP {s_id} ===")
    print(content[:400])
    print("-" * 50)
